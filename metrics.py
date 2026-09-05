"""Reproducible Korean sentence-pair evaluation; missing metrics remain None."""
from __future__ import annotations

import csv
import difflib
import html
import io
import math
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version

from sacrebleu.metrics import BLEU, CHRF

MAX_CHARS = 1500
MAX_ROWS = 100
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
NEGATION = re.compile(r"\b(?:안|못)\s|않\w*|없\w*|아니\w*|\b(?:not|never|no)\b", re.I)
NUMBERS = re.compile(r"(?<!\w)[+-]?\d+(?:[,\.]\d+)*")


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(normalize(text))


def validate(text: str, label: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{label}을 입력하세요. 빈 문장은 평가하지 않습니다.")
    if len(text) > MAX_CHARS:
        raise ValueError(f"{label}: 최대 {MAX_CHARS}자까지 입력할 수 있습니다.")
    if not any(c.isalnum() for c in text):
        raise ValueError(f"{label}: 글자 또는 숫자가 포함되어야 합니다.")
    return normalize(text)


def bleu_metric() -> BLEU:
    return BLEU(tokenize="none", smooth_method="exp", effective_order=True)


def rouge_l(reference: list[str], candidate: list[str]) -> float:
    # Unicode-aware tokenization: default English ROUGE tokenizers discard Hangul.
    previous = [0] * (len(candidate) + 1)
    for left in reference:
        current = [0]
        for j, right in enumerate(candidate, 1):
            current.append(previous[j - 1] + 1 if left == right else max(previous[j], current[-1]))
        previous = current
    return 2 * previous[-1] / (len(reference) + len(candidate)) if reference or candidate else 0.0


def lexical(reference: str, candidate: str) -> dict:
    a, b = tokens(reference), tokens(candidate)
    ca, cb = Counter(a), Counter(b)
    matched = sum((ca & cb).values())
    bleu, chrf = bleu_metric(), CHRF(char_order=6, word_order=0, beta=2)
    bv = bleu.sentence_score(" ".join(b), [" ".join(a)]).score
    cv = chrf.sentence_score(candidate, [reference]).score
    return {
        "bleu": bv, "chrf": cv, "rouge_l": rouge_l(a, b),
        "token_precision": matched / len(b) if b else 0,
        "token_recall": matched / len(a) if a else 0,
        "token_f1": 2 * matched / (len(a) + len(b)) if a or b else 0,
        "common_tokens": list((ca & cb).elements()),
        "removed_tokens": list((ca - cb).elements()),
        "added_tokens": list((cb - ca).elements()),
        "bleu_signature": str(bleu.get_signature()),
        "chrf_signature": str(chrf.get_signature()),
    }


def checks(source: str, candidate: str, keywords: str) -> dict:
    terms = list(dict.fromkeys(normalize(k) for k in keywords.split(",") if k.strip()))
    applicable = [k for k in terms if k in source]
    kept = [k for k in applicable if k in candidate]
    src_num, out_num = NUMBERS.findall(source), NUMBERS.findall(candidate)
    src_neg, out_neg = NEGATION.findall(source), NEGATION.findall(candidate)
    messages = []
    if Counter(src_num) != Counter(out_num):
        messages.append("숫자 표현이 달라졌습니다. 수량·날짜·수치의 의미를 확인하세요.")
    if src_neg != out_neg:
        messages.append("부정 표현의 형태가 달라졌습니다. 주장 반전인지 표현 변경인지 직접 확인하세요.")
    if len(candidate) < len(source) * 0.5:
        messages.append("순화문의 글자 수가 원문의 절반 미만입니다. 정보가 빠졌는지 확인하세요.")
    if source == candidate:
        messages.append("정규화 후 원문과 순화문이 같습니다. 높은 유사도만으로 순화 성공을 판단할 수 없습니다.")
    missing = [k for k in applicable if k not in candidate]
    if missing:
        messages.append("사용자가 지정한 핵심 표현 일부가 문자 그대로 유지되지 않았습니다.")
    return {
        "messages": messages,
        "source_numbers": src_num, "candidate_numbers": out_num,
        "source_negation": src_neg, "candidate_negation": out_neg,
        "keywords_kept": kept, "keywords_missing": missing,
        "keywords_not_in_source": [k for k in terms if k not in source],
        "keyword_retention": len(kept) / len(applicable) if applicable else None,
        "length_ratio": len(candidate) / len(source),
    }


def diff_html(source: str, candidate: str) -> str:
    a, b = tokens(source), tokens(candidate)
    parts = []
    for op, i, j, k, l in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        old, new = html.escape(" ".join(a[i:j])), html.escape(" ".join(b[k:l]))
        if op == "equal":
            parts.append(old)
        else:
            if op in {"delete", "replace"}:
                parts.append(f'<del style="background:#ffe1e1;color:#8f2525;padding:2px 4px">{old}</del>')
            if op in {"insert", "replace"}:
                parts.append(f'<ins style="background:#d6f5e8;color:#135c43;padding:2px 4px">{new}</ins>')
    return '<div style="line-height:2.3;font-size:17px">' + " ".join(parts) + "</div>"


def evaluate_pair(source, candidate, keywords="", engine=None,
                  use_sim=True, use_toxicity=True, use_ppl=True, threshold=0.5) -> dict:
    src, out = validate(source, "원문"), validate(candidate, "순화문")
    if not 0 < threshold < 1:
        raise ValueError("독성 판정 임계값은 0과 1 사이여야 합니다.")
    result = {
        "source": source, "candidate": candidate,
        "normalized_source": src, "normalized_candidate": out,
        "source_overlap": lexical(src, out),
        "checks": checks(src, out, keywords),
        "sim": None, "toxicity_source": None, "toxicity_candidate": None,
        "toxicity_reduction": None, "sta": None,
        "toxicity_labels_source": None, "toxicity_labels_candidate": None,
        "ppl_source": None, "ppl_candidate": None, "fl_proxy": None, "j_proxy": None,
        "errors": {}, "settings": {"toxicity_threshold": threshold, "keywords": keywords,
            "use_sim": use_sim, "use_toxicity": use_toxicity, "use_ppl": use_ppl},
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {name: version(name) for name in ["sacrebleu"]},
        "models": {},
    }
    operations = [("sim", use_sim), ("toxicity", use_toxicity), ("ppl", use_ppl)]
    for name, enabled in operations:
        if not enabled:
            continue
        try:
            if engine is None:
                raise RuntimeError("모델 엔진이 연결되지 않았습니다.")
            value = getattr(engine, name)(src, out)
            if name == "sim":
                result["sim"] = value
            elif name == "toxicity":
                original, rewritten = value["scores"]
                result.update(toxicity_source=original, toxicity_candidate=rewritten,
                              toxicity_reduction=original - rewritten, sta=int(rewritten < threshold),
                              toxicity_labels_source=value["source_labels"], toxicity_labels_candidate=value["candidate_labels"])
            else:
                result.update(ppl_source=value[0], ppl_candidate=value[1], fl_proxy=1 / value[1])
            result["models"][name] = engine.metadata(name)
        except Exception as exc:
            result["errors"][name] = f"{type(exc).__name__}: {exc}"
    if all(result[k] is not None for k in ("sta", "sim", "fl_proxy")):
        result["j_proxy"] = result["sta"] * max(0, result["sim"]) * result["fl_proxy"]
    if engine is not None:
        for name in ("torch", "transformers", "sentence-transformers"):
            result["versions"][name] = version(name)
    return result


def parse_csv(data: bytes) -> list[dict]:
    if len(data) > 2_000_000:
        raise ValueError("CSV는 2 MB 이하로 업로드하세요.")
    try:
        raw = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raw = data.decode("cp949")
    reader = csv.DictReader(io.StringIO(raw))
    if not reader.fieldnames or not {"source", "candidate"}.issubset(reader.fieldnames):
        raise ValueError("CSV에 source, candidate 열이 필요합니다. keywords 열은 선택 사항입니다.")
    rows = []
    for i, row in enumerate(reader, 2):
        if len(rows) >= MAX_ROWS:
            raise ValueError(f"한 번에 최대 {MAX_ROWS}쌍까지 평가할 수 있습니다.")
        if None in row:
            raise ValueError(f"CSV {i}행의 열 수가 맞지 않습니다. 쉼표가 있는 문장은 큰따옴표로 감싸세요.")
        rows.append({k: row.get(k) or "" for k in ("source", "candidate", "keywords")})
    if not rows:
        raise ValueError("CSV에 평가할 행이 없습니다.")
    return rows


def flatten(result: dict) -> dict:
    keys = ("source", "candidate", "sim", "toxicity_source", "toxicity_candidate",
            "toxicity_reduction", "sta", "ppl_source", "ppl_candidate", "fl_proxy", "j_proxy")
    row = {k: result.get(k) for k in keys}
    for metric in ("bleu", "chrf", "rouge_l", "token_f1"):
        row[f"source_{metric}"] = (result.get("source_overlap") or {}).get(metric)
    row["warnings"] = " | ".join(result.get("checks", {}).get("messages", []))
    row["errors"] = str(result.get("errors", {}))
    return row


def results_csv(results: list[dict]) -> bytes:
    rows = [flatten(r) for r in results]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        # Prevent spreadsheet formula execution in user-controlled text fields.
        writer.writerow({k: ("'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v)
                         for k, v in row.items()})
    return buffer.getvalue().encode("utf-8-sig")


def aggregate(results: list[dict]) -> dict:
    summary = {"total_rows": len(results)}
    for name in ("sim", "sta", "toxicity_reduction", "ppl_candidate", "j_proxy"):
        values = [r[name] for r in results if r.get(name) is not None]
        summary[name + "_macro_mean"] = sum(values) / len(values) if values else None
        summary[name + "_valid_n"] = len(values)
    eligible = [r for r in results if r.get("toxicity_source") is not None
                and r["toxicity_source"] >= r["settings"]["toxicity_threshold"]]
    summary["initially_toxic_n"] = len(eligible)
    summary["toxic_to_nontoxic_rate"] = sum(r["sta"] for r in eligible) / len(eligible) if eligible else None
    pairs = [(r["normalized_candidate"], normalize(r["source"]))
             for r in results if r.get("normalized_candidate")]
    summary["source_corpus_n"] = len(pairs)
    if pairs:
        candidates, sources = zip(*pairs)
        summary["source_corpus_bleu"] = bleu_metric().corpus_score(
            [" ".join(tokens(s)) for s in candidates], [[" ".join(tokens(s)) for s in sources]]).score
        summary["source_corpus_chrf"] = CHRF().corpus_score(list(candidates), [list(sources)]).score
    return summary
