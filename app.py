from pathlib import Path
import json
import os
import streamlit as st

from metrics import (evaluate_pair, aggregate, parse_csv, results_csv, flatten, diff_html, MAX_CHARS)
from models import ModelEngine, MODEL_IDS

st.set_page_config(page_title="문장 보존 실험실", page_icon="🔬", layout="wide")
if st.session_state.get("result_schema") != 2:
    st.session_state.pop("single_result", None)
    st.session_state.pop("batch_result", None)
    st.session_state.result_schema = 2
st.markdown("""<style>
.stApp { background: #f7f9fc; }
.block-container { max-width: 1240px; padding-top: 4.5rem; }
h1,h2,h3 { letter-spacing: -.04em; }
div[data-testid="stMetric"] { background:white; border:1px solid #e1e7ef; border-radius:14px; padding:18px; }
div[data-testid="stMetricLabel"] { color:#536177; }
div[data-testid="stMetricValue"] { color:#143c66; white-space:normal; text-overflow:clip; }
div[data-testid="stMetricValue"] > div, div[data-testid="stMetricValue"] p { white-space:inherit; overflow-wrap:anywhere; text-overflow:clip; }
div[data-testid="stMetricValue"] p { display:block; }
[class*="st-key-comparison-card-"] div[data-testid="stMetricValue"] { font-size:1.55rem; line-height:1.6; white-space:pre-line; }
[class*="st-key-score-card-"] { background:white; border:1px solid #e1e7ef; border-radius:14px; padding:18px; }
[class*="st-key-score-card-"] div[data-testid="stMetric"] { background:transparent; border:0; border-radius:0; padding:0; }
.eyebrow { color:#256b91; font-size:12px; font-weight:700; letter-spacing:.16em; }
</style>""", unsafe_allow_html=True)


@st.cache_resource
def get_engine():
    return ModelEngine()


def fmt(value, digits=3):
    return "미측정" if value is None else f"{value:.{digits}f}"


def evaluate(row):
    options = dict(use_sim=use_sim, use_toxicity=use_toxicity, use_ppl=use_ppl, threshold=threshold)
    if not any([use_sim, use_toxicity, use_ppl]):
        return evaluate_pair(**row, engine=None, **options)
    with get_engine().session(offline=offline) as engine:
        return evaluate_pair(**row, engine=engine, **options)


def download(results, key):
    a, b = st.columns(2)
    with a:
        st.download_button("결과 CSV 저장", results_csv(results), "evaluation.csv", "text/csv", key=key+"csv")
    with b:
        st.download_button("설정·모델 정보 포함 JSON 저장", json.dumps({"results": results,
            "summary": aggregate(results)}, ensure_ascii=False, indent=2, allow_nan=False),
            "evaluation.json", "application/json", key=key+"json")


def render_result(r):
    st.subheader("분석 결과")
    st.caption("아래 결과는 마지막으로 ‘문장 비교하기’를 누른 입력과 설정 기준입니다. 입력을 바꾼 뒤에는 다시 분석하세요.")
    with st.expander("독성 변화와 자연성 상세", expanded=True):
        a, b, c = st.columns(3)
        with a, st.container(key="comparison-card-toxicity"):
            st.metric("독성 점수 변화", f"원문 {fmt(r['toxicity_source'])}\n순화문 {fmt(r['toxicity_candidate'])}")
            st.caption("**독성 점수:** 혐오·욕설을 모델이 평가한 점수입니다. **0~1** 사이이며, 낮을수록 독성을 낮게 평가합니다.")
        with b:
            st.metric("독성 감소량", fmt(r["toxicity_reduction"]))
            st.caption("**원문 독성 점수 − 순화문 독성 점수**")
            st.caption("**양수(0보다 큼):** 독성 감소 · **0:** 변화 없음 · **음수(0보다 작음):** 독성 증가")
        with c, st.container(key="comparison-card-ppl"):
            st.metric("PPL 변화", f"원문 {fmt(r['ppl_source'], 2)}\n순화문 {fmt(r['ppl_candidate'], 2)}")
            st.caption("**PPL:** 언어 모델이 문장을 예측하기 어려운 정도를 나타냅니다. **낮을수록 예측하기 쉬우며**, 문법 정확도 점수는 아닙니다.")
        if r["sta"] is not None:
            st.write(f"비독성 판정 STAᵢ = {r['sta']} (순화문 독성 점수 < {r['settings']['toxicity_threshold']}). 여러 문장의 평균이 비독성 판정 비율 STA입니다.")
        st.caption("UnSmile 다중 라벨 분류기의 혐오·욕설 9개 점수 중 최댓값을 사용합니다(clean 제외). 인용·반어 등에서 오판할 수 있으며, 이 프로젝트 데이터의 정확도는 별도 검증이 필요합니다.")
        if r.get("toxicity_labels_source"):
            st.dataframe([{"분류": label, "원문 점수": score, "순화문 점수": r["toxicity_labels_candidate"][label]}
                          for label, score in r["toxicity_labels_source"].items()], hide_index=True, width="stretch")
    with st.expander("평가한 문장과 설정 확인"):
        st.write("원문", r["source"])
        st.write("순화문", r["candidate"])
        st.json(r["settings"])
    cards = st.columns(4)
    with cards[0], st.container(key="score-card-sim"):
        st.metric("의미 유사도 · SIM", fmt(r["sim"]), help="한국어 KR-SBERT 임베딩 코사인 유사도. −1~1. 의미 보존 확률이나 정확도가 아닙니다.")
        st.caption("최소 **−1** · 최대 **1**")
        st.caption("↑ 높을수록 모델이 의미를 유사하게 평가")
    with cards[1], st.container(key="score-card-chrf"):
        st.metric("문자 중복 · chrF", fmt(r["source_overlap"]["chrf"], 1), help="원문 대비 문자 1~6-gram Fβ(β=2). 0~100. 의미 판단과 별개입니다.")
        st.caption("최소 **0** · 최대 **100**")
        st.caption("↑ 높을수록 겹치는 문자 표현이 많음")
    with cards[2], st.container(key="score-card-toxicity"):
        st.metric("순화문 독성 점수", fmt(r["toxicity_candidate"]), help="UnSmile의 혐오·욕설 9개 라벨 sigmoid 점수 중 최댓값. 0~1. 하나 이상 유해할 확률이나 보정된 실제 독성 확률이 아닙니다.")
        st.caption("최소 **0** · 최대 **1**")
        st.caption("↓ 낮을수록 모델이 독성을 낮게 평가")
    with cards[3], st.container(key="score-card-ppl"):
        st.metric("순화문 PPL ↓", fmt(r["ppl_candidate"], 2), help="KoGPT2가 평가한 perplexity. 낮을수록 모델 관점에서 예측하기 쉽습니다. 문법 판정이나 0~1 FL이 아닙니다.")
        st.caption("최소 **1** (이론값) · 최대 **상한 없음**")
        st.caption("↓ 낮을수록 언어 모델이 예측하기 쉬움")
    st.info("SIM이 높아도 주장·대상·인과관계가 같다고 보장하지 않습니다. 변경 표시와 확인 항목을 함께 읽으세요.")

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("#### 바뀐 표현")
        st.caption("빨간 취소선: 삭제 · 초록 밑줄: 추가 / 비교용 토큰 사이에 공백을 표시합니다.")
        st.markdown(diff_html(r["normalized_source"], r["normalized_candidate"]), unsafe_allow_html=True)
        st.markdown("#### 표현 중복 지표")
        s = r["source_overlap"]
        names = [("BLEU · 0~100", "bleu"), ("chrF · 0~100", "chrf"),
                 ("ROUGE-L F1 · 0~1", "rouge_l"), ("토큰 중복 F1 · 0~1", "token_f1")]
        st.dataframe([{"지표": name, "원문 대비": fmt(s[key])}
                      for name, key in names], hide_index=True, width="stretch")
        st.caption("원문 대비 BLEU는 self-BLEU 성격의 중복도입니다. 원문 복사도 높은 점수를 받으므로 순화 품질로 해석하지 않습니다.")
        with st.expander("겹침·삭제·추가 토큰 보기"):
            st.write("겹침", s["common_tokens"])
            st.write("삭제", s["removed_tokens"])
            st.write("추가", s["added_tokens"])
    with right:
        st.markdown("#### 문맥 확인 항목")
        for message in r["checks"]["messages"]:
            st.warning(message)
        if not r["checks"]["messages"]:
            st.write("현재 단순 규칙에서 확인 항목이 발견되지 않았습니다. 의미 일치를 인증하는 결과는 아닙니다.")
        st.caption("숫자·일부 부정 표현·길이·사용자 핵심어를 살펴보는 규칙 검사입니다. 동의어·주체 교체·반어·부정 범위를 완전히 판별하지 못합니다.")
        check = r["checks"]
        st.write("핵심어 유지율", fmt(check["keyword_retention"]))
        st.write("누락된 핵심 표현", ", ".join(check["keywords_missing"]) or "없음 / 미지정")
        if check["keywords_not_in_source"]:
            st.write("원문에 없어 계산에서 제외한 표현", check["keywords_not_in_source"])
        st.write("숫자 비교", {"원문": check["source_numbers"], "순화문": check["candidate_numbers"]})

    with st.expander("실험용 종합 점수 · 논문의 J와 다름"):
        st.latex(r"FL_{proxy}=1/PPL_y,\quad J_{proxy}=STA_i\times\max(0,SIM)\times FL_{proxy}")
        st.write("J_proxy", fmt(r["j_proxy"], 6))
        st.caption("PPL을 역수로 바꾼 임의의 보조 점수입니다. ParaDetox의 FL·J 재현값이 아니며 타 논문의 점수와 비교할 수 없습니다. 세 모델 지표 중 하나라도 미측정이면 계산하지 않습니다.")
    if r["errors"]:
        st.error("일부 모델 지표를 측정하지 못했습니다. 다른 값을 대입하지 않고 ‘미측정’으로 남겼습니다.")
        with st.expander("모델 오류와 해결 방법", expanded=True):
            st.json(r["errors"])
            st.write("최초 실행은 모델 다운로드를 위해 인터넷이 필요합니다. 다운로드 후에는 오프라인 모드를 쓸 수 있습니다. 입력 한도 오류는 문장을 나눠 해결하세요. 재시도하려면 왼쪽 ‘모델 다시 불러오기’를 누르세요.")
    with st.expander("재현 정보"):
        st.json({"models": r["models"], "versions": r["versions"],
                 "bleu_signature": r["source_overlap"]["bleu_signature"],
                 "chrf_signature": r["source_overlap"]["chrf_signature"],
                 "measured_at_utc": r["measured_at_utc"]})
    download([r], "single")


with st.sidebar:
    st.header("평가 설정")
    st.caption("한국어 문장 비교 · CPU 실행")
    use_sim = st.checkbox("의미 유사도 · KR-SBERT", value=True)
    use_toxicity = st.checkbox("독성 변화 · 혐오 분류기", value=True)
    use_ppl = st.checkbox("자연성 보조 · KoGPT2 PPL", value=True)
    threshold = st.slider("독성 판정 임계값", min_value=0.05, max_value=0.95, value=0.5, step=0.05)
    st.caption("0.5는 초기 설정값입니다. 연구 데이터의 사람 평가에 맞춰 별도 검증하세요.")
    offline = st.checkbox("오프라인 모드 · 받은 모델만 사용", value=False)
    if st.button("모델 다시 불러오기"):
        get_engine().reset()
        st.success("모델 캐시를 초기화했습니다. 비교 버튼을 다시 누르세요.")
    st.divider()
    st.write("첫 모델 실행에는 다운로드 시간이 필요합니다. 입력 문장은 이 앱을 실행하는 서버에서 계산하며 외부 추론 API에 보내지 않습니다.")
    if os.environ.get("DETOX_CLOUD") == "1":
        st.caption("입력 문장은 분석을 위해 클라우드 서버로 전송됩니다. 이 앱 코드는 입력·분석 결과를 파일이나 데이터베이스에 저장하지 않습니다.")
        st.caption("서버 메모리를 절약하기 위해 모델을 하나씩 불러옵니다. 다른 사용자가 분석 중이면 순서대로 처리하므로 시간이 걸릴 수 있습니다.")
    st.caption("서버를 종료하면 메모리의 결과가 사라집니다. 필요한 결과는 CSV·JSON으로 저장하세요.")

st.markdown('<div class="eyebrow">TEXT DETOXIFICATION · EVALUATION</div>', unsafe_allow_html=True)
st.title("말은 부드럽게, 의미는 그대로일까?")
st.write("원문과 순화문을 나란히 놓고 의미, 겹치는 표현, 독성 변화, 자연성을 측정합니다.")
single, batch, guide = st.tabs(["문장 한 쌍 비교", "CSV 일괄 평가", "지표 설명·연구에 사용하기"])

with single:
    with st.form("pair_form"):
        a, b = st.columns(2)
        source = a.text_area("원문", "네 보고서는 쓰레기야. 근거가 부족해.", height=150, max_chars=MAX_CHARS)
        candidate = b.text_area("순화문", "보고서의 근거가 부족합니다.", height=150, max_chars=MAX_CHARS)
        with st.expander("선택 입력 · 보존할 핵심 표현"):
            keywords = st.text_input("보존을 확인할 핵심 표현 · 쉼표로 구분", placeholder="보고서, 근거",
                                     help="원문에 존재하는 표현만 분모에 포함합니다. 부분 문자열의 존재 여부를 확인합니다.")
        submitted = st.form_submit_button("문장 비교하기", type="primary", width="stretch")
    if submitted:
        with st.spinner("지표를 계산하고 있습니다. 첫 실행에는 모델을 내려받습니다."):
            try:
                st.session_state.single_result = evaluate(dict(source=source, candidate=candidate, keywords=keywords))
            except Exception as exc:
                st.session_state.pop("single_result", None)
                st.error(str(exc))
    if "single_result" in st.session_state:
        render_result(st.session_state.single_result)

with batch:
    st.subheader("여러 순화 결과를 한 번에 비교")
    st.write("필수 열은 source(원문), candidate(순화문)입니다. keywords(핵심 표현)는 선택입니다. 최대 100쌍까지 평가합니다.")
    st.download_button("예제 CSV 받기", Path(__file__).with_name("examples.csv").read_bytes(), "examples.csv", "text/csv")
    uploaded = st.file_uploader("CSV 파일 · UTF-8 또는 CP949", type=["csv"])
    if st.button("CSV 평가 시작", type="primary", disabled=uploaded is None):
        st.session_state.pop("batch_result", None)
        try:
            rows = parse_csv(uploaded.getvalue())
            results = []
            progress = st.progress(0, text="평가 준비 중")
            for i, row in enumerate(rows):
                try:
                    result = evaluate(row)
                except Exception as exc:
                    result = {**row, "errors": {"input": str(exc)}}
                results.append(result)
                progress.progress((i + 1) / len(rows), text=f"{i+1} / {len(rows)}쌍 완료")
            st.session_state.batch_result = results
        except Exception as exc:
            st.error(str(exc))
    if "batch_result" in st.session_state:
        results = st.session_state.batch_result
        st.caption("마지막으로 ‘CSV 평가 시작’을 누른 파일·설정 기준입니다. 실패한 지표는 빈칸이며 평균에서 제외됩니다.")
        summary = aggregate(results)
        a, b, c = st.columns(3)
        a.metric("평가 행 수", len(results))
        b.metric("평균 SIM", fmt(summary["sim_macro_mean"]))
        c.metric("STA · 비독성 판정 비율", fmt(summary["sta_macro_mean"]))
        st.dataframe([flatten(r) for r in results], hide_index=True, width="stretch")
        with st.expander("집계값과 지표별 유효 표본 수", expanded=True):
            st.json(summary)
            st.caption("STA는 순화문 비독성 판정의 평균입니다. toxic_to_nontoxic_rate는 원문이 독성으로 판정된 쌍만의 성공 비율입니다. J_proxy는 문장별 곱을 계산한 뒤 평균합니다. PPL 평균은 문장별 산술평균이며 corpus PPL이 아닙니다.")
        download(results, "batch")

with guide:
    st.subheader("점수의 뜻과 한계")
    st.markdown("""
| 항목 | 계산 | 해석할 때 확인할 것 |
|---|---|---|
| SIM | KR-SBERT 임베딩의 코사인 유사도, −1~1 | 의미 보존 확률이 아님. 부정·대상 변경은 직접 확인 |
| BLEU | Unicode 토큰 1~4-gram, exp smoothing, effective order, 0~100 | 원문과 순화문의 표현 중복도. 짧은 문장에 민감 |
| chrF | 공백 제외 문자 1~6-gram, β=2, 0~100 | 한국어 표현 중복을 살피는 보조 지표 |
| ROUGE-L | Unicode 토큰 최장 공통 부분수열 F1, 0~1 | 형태소 분석을 하지 않으므로 조사가 바뀌면 별도 토큰 |
| 독성 점수 | UnSmile의 혐오·욕설 9개 라벨 sigmoid 중 최댓값 | 실제 독성 확률이나 라벨 합집합 확률이 아님 |
| STA | 순화문 독성 점수가 임계값 미만이면 1, 아니면 0. 데이터셋에서는 평균 | toxicity rate와 방향이 반대. 순화로 인해 개선됐는지는 감소량도 확인 |
| PPL | KoGPT2의 exp(평균 다음 토큰 음의 로그우도) | 낮을수록 모델이 예측하기 쉬움. 문법 정확도·FL과 같지 않음 |
| J_proxy | STA × max(0,SIM) × 1/PPL | 이 프로그램의 실험용 정의. ParaDetox의 J가 아님 |

**연구 설계:** 좋은 순화, 단순 복사, 과도한 삭제, 주장 반전, 숫자 변경, 무관한 문장 쌍을 함께 평가하세요.
사람이 별도로 평가한 의미 보존·독성·자연성과 자동 점수를 비교해야 이 프로그램의 평가 성능을 검증할 수 있습니다.
핵심 주장이 혐오 자체인 문장은 독성 제거와 주장 보존이 충돌할 수 있습니다. 원문 전체 보존과 보존할 비공격적 정보의 범위를 구분하세요.

**재현:** 결과 JSON에 모델 ID·리비전·라이브러리 버전·임계값·지표 설정이 들어갑니다.
공백은 한 칸으로 합치고 Unicode NFC 정규화를 적용합니다. 모델 입력 길이를 넘으면 해당 모델 점수를 미측정 처리합니다.
한 문장 비교만으로 정확도·F1 같은 평가기 성능을 측정할 수는 없습니다.

**근거 자료**
- [ParaDetox 논문, §5.2](https://aclanthology.org/2022.acl-long.469/) — FL은 CoLA 기반 문장 수용성 분류기이며 본 프로그램 PPL과 다릅니다.
- [KR-SBERT 모델 카드](https://huggingface.co/snunlp/KR-SBERT-V40K-klueNLI-augSTS)
- [UnSmile 공식 모델·추론 방법](https://github.com/smilegate-ai/korean_unsmile_dataset) — 다중 라벨 sigmoid를 사용합니다.
- [KoGPT2](https://github.com/SKT-AI/KoGPT2), [PPL 정의](https://huggingface.co/docs/transformers/perplexity)
- [SacreBLEU 구현](https://github.com/mjpost/sacrebleu)
""")
    st.json(MODEL_IDS)
