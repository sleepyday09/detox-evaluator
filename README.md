# 문장 보존 실험실

**인터넷에서 사용:** https://sleepyday09-detox-evaluator.streamlit.app/

브라우저만 있으면 이용할 수 있습니다. 2026-09-06 공개 서버에서 SIM·독성·PPL·J_proxy 계산과 로그인하지 않은 별도 브라우저의 분석을 확인했습니다. 처음 접속하거나 서버가 다시 시작된 직후에는 준비 시간이 걸릴 수 있습니다.

한국어 **원문과 순화문**을 입력해 의미 유사도와 표현 중복, 독성 변화, 자연성 보조 지표를 계산하는 프로그램입니다. 자동 순화 모델을 만드는 프로그램이 아니라, 이미 작성된 순화 결과를 평가하는 프로그램입니다. 로컬 실행과 클라우드 배포를 지원하며 배포 설정은 [DEPLOY.md](DEPLOY.md)에 있습니다.

## 실행

이 PC에서는 `브라우저로 실행.bat` 또는 `start.bat`을 더블클릭합니다. 프로그램이 이미 실행 중이면 브라우저만 열고, 실행 중이 아니면 서버를 시작한 뒤 기본 브라우저를 엽니다. 브라우저가 열리지 않으면 http://127.0.0.1:8517 에 접속하세요. 새로 시작한 서버의 실행 창을 닫으면 해당 서버가 종료됩니다.

다른 PC에서는 Python 3.11 또는 3.12를 설치한 뒤 `start.bat`을 실행하세요. 전용 가상환경과 라이브러리를 설치하며, 첫 모델 분석에는 인터넷 연결이 필요합니다. API 키는 필요하지 않습니다. 모델을 받은 뒤에는 화면 왼쪽의 오프라인 모드를 선택할 수 있습니다. 모델 다운로드 실패 시 해당 지표가 미측정으로 표시됩니다.

직접 실행할 경우:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8517 --browser.gatherUsageStats false
```

프로그램은 CPU에서 실행합니다. 추론에 사용하는 입력 문장은 외부 추론 API에 전송하지 않습니다. 공개 모델 다운로드에는 Hugging Face 연결을 사용합니다. 사용자가 저장한 CSV·JSON에는 입력 문장이 포함됩니다.

## 사용

1. 원문과 순화문을 입력하고 **문장 비교하기**를 누릅니다.
2. 필요하면 사람이 작성한 기준 순화문과 보존할 핵심 표현을 입력합니다.
3. SIM, 표현 중복 지표, 변경된 토큰, 숫자·부정 표현 확인 항목을 함께 살펴봅니다.
4. CSV 또는 JSON으로 결과를 저장합니다. JSON에는 모델 리비전과 설정이 포함됩니다.
5. CSV 일괄 평가에서는 `examples.csv` 양식을 사용합니다. 필수 열은 `source,candidate`, 선택 열은 `reference,keywords`입니다. 쉼표가 들어가는 셀은 CSV 규칙에 따라 큰따옴표로 감쌉니다.

한 번에 최대 100쌍, 파일 2 MB, 입력 하나당 1,500자입니다. 모델별 토큰 한도가 더 짧을 수 있습니다. 한도를 넘으면 해당 모델 점수를 계산하지 않으며 텍스트를 몰래 자르지 않습니다.

## 지표 정의

| 지표 | 구현·범위 | 주의점 |
|---|---|---|
| SIM | KR-SBERT 정규화 임베딩의 내적 = 코사인 유사도, −1~1 | 의미 보존의 확률이 아니며 명제 일치를 보장하지 않음 |
| BLEU | SacreBLEU, Unicode 정규식 토큰, 1~4-gram, exp smoothing, effective order, 0~100 | 원문 대비는 self-BLEU 성격의 중복 측정. 기준 순화문 대비와 분리 |
| chrF | SacreBLEU, 문자 1~6-gram, 공백 제외, β=2, 0~100 | 문자 중복이며 의미 이해가 아님 |
| ROUGE-L | Unicode 토큰의 최장 공통 부분수열 F1, 0~1 | 영어 전용 토크나이저로 한글이 사라지는 문제를 피함. 형태소 분석은 하지 않음 |
| 토큰 F1 | 중복 횟수를 고려한 원문·순화문 토큰 중복 F1, 0~1 | 대소문자·조사·어미 차이를 그대로 반영 |
| 독성 | UnSmile 다중 라벨 분류기의 혐오·욕설 9개 sigmoid 점수 중 최댓값(clean 제외), 0~1 | 하나 이상 유해할 확률이나 보정된 실제 독성 확률이 아님. 상세 라벨 점수도 제공 |
| 독성 감소 | 원문 독성 점수 − 순화문 독성 점수, −1~1 | 양수일 때 감소. 성공률과 별개 |
| STA | 순화문 독성 점수가 임계값 미만이면 1, 아니면 0 | 데이터셋에서는 평균. 기본 임계값 0.5는 이 프로젝트에서 보정하지 않은 초기값 |
| PPL | KoGPT2, exp(평균 다음 토큰 음의 로그우도), 동일 모델로 전후 비교 | BOS/EOS 추가 없이 첫 토큰을 제외한 다음 토큰을 평가. 낮다고 문법이 정확한 것은 아님 |
| J_proxy | STA × max(0,SIM) × 1/PPL | 이 프로그램에서 정의한 실험용 수치. **ParaDetox의 FL·J와 다름** |

NFC 정규화와 연속 공백 축약을 적용합니다. 토큰 규칙은 `\w+|[^\w\s]`이며 한국어 어절과 문장부호를 분리합니다. 원래 대소문자는 유지합니다. 핵심어 유지는 사용자가 입력한 표현의 부분 문자열 포함 여부이고, 원문에 없는 표현은 분모에서 제외합니다. 숫자·일부 부정 표현·길이 검사는 **단순 규칙에 따른 확인 항목**이며 논리적 모순을 검증하는 NLI가 아닙니다.

일괄 평가에서 실패한 지표는 평균에서 제외하고 유효 표본 수를 표시합니다. 원래 독성이 있는 것으로 판정된 문장 중 비독성으로 바뀐 비율은 `toxic_to_nontoxic_rate`로 따로 계산합니다. 문장별 J_proxy를 계산한 뒤 평균하며, 평균 지표끼리 곱하지 않습니다. 문장 PPL의 산술평균은 corpus PPL이 아닙니다. corpus BLEU·chrF는 각 문장 점수 평균과 별도로 계산합니다.

## 연구에서 해석하기

- SIM을 ‘의미가 90% 보존되었다’처럼 해석하지 마세요. 높은 유사도에서도 주체·부정·수치·인과관계가 달라질 수 있습니다.
- 단순 복사, 자연스러운 순화, 과도한 삭제, 주장 반전, 숫자 변경, 무관한 문장 등의 대조군을 준비하세요.
- 사람이 평가한 의미 보존·독성·자연성 정답을 별도로 확보하고, 자동 평가와의 일치·상관을 검증해야 평가기 성능을 주장할 수 있습니다. 예제 작동 확인은 그런 성능 검증을 대체하지 않습니다.
- 원문의 주장이 혐오 자체인 경우에는 독성을 없애면서 모든 주장을 유지하기 어려울 수 있습니다. 평가 전에 보존할 비공격적 정보의 범위를 정하세요.
- J_proxy는 PPL의 임의 역수 변환을 사용합니다. 논문 간 비교용 최종 성능이나 합격 기준으로 사용하지 마세요. ParaDetox FL은 CoLA 기반 수용성 분류기로 측정했으며, 이 프로그램에 해당 영어 분류기를 한국어용인 것처럼 적용하지 않았습니다.

## 이번 실행에서 확인한 사항

- 계산·입력 검증·CSV 처리 등 단위 검사 12개 통과.
- 실제 앱의 초기 화면, 표현 지표만 계산, 빈 입력 오류, 세 모델을 사용한 오프라인 분석 확인.
- 예제 4쌍 모두 SIM·독성·PPL 계산 성공. 실제 결과는 `sample-results.json`에 저장.
- 동일 문장 SIM 약 1.000, 검사에 사용한 무관한 문장 쌍 SIM 약 0.079. 입력 토큰 한도 초과 거부 확인.
- 위 내용은 작동 검사이며, 사람 평가와 비교한 정확도 검증은 수행하지 않았습니다. 기계 판독용 기록은 `verification.json`에 있습니다.

## 파일

- `app.py`: 사용자 화면, 한 쌍·CSV 평가
- `metrics.py`: 중복 지표, 집계, 규칙 검사, 내보내기
- `models.py`: 한국어 임베딩·혐오 분류·PPL 로컬 추론
- `model-lock.json`: 사용할 모델 리비전 고정
- `requirements.txt`: 이 PC에서 검증한 주요 라이브러리 버전 고정
- `requirements-lock.txt`: 이 PC에서 확인한 설치 버전
- `examples.csv`, `sample-results.json`: 예제 입력과 실제 실행 결과
- `tests/test_metrics.py`: 기본 계산·오류 처리 검증

## 출처

- [ParaDetox: Detoxification with Parallel Data (ACL 2022), §5.2](https://aclanthology.org/2022.acl-long.469/)
- [Sentence Transformers: Semantic Textual Similarity](https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html)
- [KR-SBERT 모델 카드](https://huggingface.co/snunlp/KR-SBERT-V40K-klueNLI-augSTS)
- [UnSmile 공식 모델·추론 방법](https://github.com/smilegate-ai/korean_unsmile_dataset)
- [KoGPT2 공식 저장소](https://github.com/SKT-AI/KoGPT2)
- [Hugging Face: PPL 정의](https://huggingface.co/docs/transformers/perplexity)
- [SacreBLEU 공식 구현](https://github.com/mjpost/sacrebleu)

공개 모델의 가중치는 배포 ZIP에 포함하지 않습니다. 각 모델·라이브러리·학습 데이터의 이용 조건은 해당 원문을 확인하세요.
