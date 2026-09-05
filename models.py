"""Local inference only. Public model weights are downloaded on first use."""
import json
import math
import os
import gc
import threading
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".model-cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_IDS = {
    "sim": "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    "toxicity": "smilegate-ai/kor_unsmile",
    "ppl": "skt/kogpt2-base-v2",
}


class ModelEngine:
    def __init__(self, offline=False):
        self.offline = offline
        self.loaded = {}
        self.failed = {}
        self.low_memory = os.environ.get("DETOX_LOW_MEMORY", "0") == "1"
        self._lock = threading.RLock()
        lockfile = ROOT / "model-lock.json"
        self.revisions = json.loads(lockfile.read_text("utf-8")) if lockfile.exists() else {}

    @contextmanager
    def session(self, offline=False):
        # A cached engine is shared by browser sessions. Keep inference and
        # metadata reads together while another request waits for the engine.
        with self._lock:
            if self.offline != offline:
                self.failed.clear()
            self.offline = offline
            yield self

    def reset(self):
        with self._lock:
            self.loaded.clear()
            self.failed.clear()
            gc.collect()

    def _options(self, name):
        locked = self.revisions.get(name, {})
        revision = locked.get("revision", "main") if locked.get("id") == MODEL_IDS[name] else "main"
        return {"revision": revision,
                "local_files_only": self.offline, "trust_remote_code": False}

    def load(self, name):
        if name in self.failed:
            raise RuntimeError(self.failed[name])
        if name in self.loaded:
            return self.loaded[name]
        if self.low_memory:
            self.loaded.clear()
            gc.collect()
        import torch
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        try:
            # Load a complete cached checkpoint by local path. Some optional
            # processor probes in SentenceTransformer ignore local_files_only
            # when a Hub ID is supplied, so a local directory avoids them.
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import LocalEntryNotFoundError
            location, options = MODEL_IDS[name], self._options(name)
            try:
                cached = Path(snapshot_download(location, revision=options["revision"], local_files_only=True))
                has_weights = any((cached / f).exists() for f in ("model.safetensors", "pytorch_model.bin"))
                if has_weights and (cached / "config.json").exists():
                    location = str(cached)
                    options = {**options, "local_files_only": True}
            except (FileNotFoundError, LocalEntryNotFoundError):
                if self.offline:
                    raise
            if name == "sim":
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(location, device="cpu", **options)
                model.eval()
                value = (model.tokenizer, model)
            else:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
                tokenizer = AutoTokenizer.from_pretrained(location, **options)
                cls = AutoModelForSequenceClassification if name == "toxicity" else AutoModelForCausalLM
                model = cls.from_pretrained(location, **options).to("cpu").eval()
                if name == "toxicity":
                    labels = set(model.config.id2label.values())
                    expected = {"여성/가족", "남성", "성소수자", "인종/국적", "연령", "지역", "종교", "기타 혐오", "악플/욕설", "clean"}
                    if labels != expected:
                        raise ValueError("공식 UnSmile 라벨과 실제 모델 라벨이 다릅니다. 라벨 매핑을 확인하세요.")
                value = (tokenizer, model)
            self.loaded[name] = value
            return value
        except Exception as exc:
            self.failed[name] = f"{type(exc).__name__}: {exc}"
            raise

    def metadata(self, name):
        tokenizer, model = self.loaded[name]
        config = model[0].auto_model.config if name == "sim" else model.config
        return {"id": MODEL_IDS[name], "revision": getattr(config, "_commit_hash", None)
                or self._options(name)["revision"], "device": "cpu",
                "max_tokens": model.max_seq_length if name == "sim" else getattr(config, "max_position_embeddings", None),
                "toxicity_definition": "max(sigmoid(logits)) over 9 non-clean labels; not union probability" if name == "toxicity" else None,
                "label_mapping": config.id2label if name == "toxicity" else None,
                "ppl_definition": "exp(mean next-token NLL), no BOS/EOS added; first token unscored" if name == "ppl" else None}

    @staticmethod
    def _check_length(tokenizer, texts, limit, add_special_tokens=True):
        for text in texts:
            count = len(tokenizer.encode(text, add_special_tokens=add_special_tokens, truncation=False))
            if count > limit:
                raise ValueError(f"모델 입력 한도 {limit}토큰을 넘었습니다({count}토큰). 문장을 나누어 평가하세요. 자동으로 자르지 않습니다.")

    def sim(self, source, candidate):
        tokenizer, model = self.load("sim")
        self._check_length(tokenizer, [source, candidate], model.max_seq_length)
        embeddings = model.encode([source, candidate], normalize_embeddings=True, show_progress_bar=False)
        return max(-1.0, min(1.0, float(embeddings[0] @ embeddings[1])))

    def toxicity(self, source, candidate):
        import torch
        tokenizer, model = self.load("toxicity")
        self._check_length(tokenizer, [source, candidate], model.config.max_position_embeddings)
        inputs = tokenizer([source, candidate], padding=True, truncation=False, return_tensors="pt")
        with torch.inference_mode():
            scores = torch.sigmoid(model(**inputs).logits)
        label_scores = [{model.config.id2label[i]: float(v) for i, v in enumerate(row)} for row in scores]
        maxima = [max(v for k, v in row.items() if k != "clean") for row in label_scores]
        return {"scores": maxima, "source_labels": label_scores[0], "candidate_labels": label_scores[1]}

    def ppl(self, source, candidate):
        import torch
        tokenizer, model = self.load("ppl")
        limit = model.config.n_positions
        self._check_length(tokenizer, [source, candidate], limit, add_special_tokens=False)
        values = []
        for text in (source, candidate):
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            ids = inputs["input_ids"]
            if ids.shape[1] < 2:
                raise ValueError("PPL은 최소 2개의 모델 토큰이 필요합니다.")
            with torch.inference_mode():
                nll = float(model(**inputs, labels=ids).loss)
            if not math.isfinite(nll) or nll > 700:
                raise ValueError("유한한 PPL을 계산하지 못했습니다.")
            values.append(math.exp(nll))
        return values
