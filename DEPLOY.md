# 인터넷 배포

공개 주소: https://sleepyday09-detox-evaluator.streamlit.app/

저장소: https://github.com/sleepyday09/detox-evaluator

2026-09-06 확인: 실제 배포 로그의 실행 환경은 Python 3.14.7 / CPU PyTorch 2.14.0+cpu입니다. 기본 예제의 SIM 0.698, 순화문 독성 0.071, PPL 1379.05, J_proxy 0.000506을 화면에서 확인했습니다. 별도 비로그인 브라우저에서도 전체 지표 계산과 숫자 변경 경고를 확인했습니다. 실제 휴대폰 접속과 대규모 동시 접속 부하 테스트는 수행하지 않았습니다.

## Streamlit Community Cloud 설정

- 저장소: 이 폴더의 코드만 포함한 GitHub 저장소
- 브랜치: `main`
- Main file path: `cloud_app.py`
- Advanced settings → Python version: `3.14` (현재 서버에서 확인한 버전; 로컬 테스트는 3.12)
- Secrets: 필요 없음

`requirements.txt`로 라이브러리를 설치하고 첫 분석 때 고정된 리비전의 공개 모델을 내려받습니다. Linux에서는 CPU용 PyTorch를 사용합니다. 로컬 환경을 기록한 `requirements-lock.txt`는 클라우드 설치 파일로 사용하지 않습니다.

`cloud_app.py`는 메모리를 절약하기 위해 모델을 하나씩 유지합니다. 공유 모델 엔진은 요청을 순서대로 처리하며 입력과 결과는 각 브라우저 세션에서 관리합니다. 모델을 바꿀 때 다시 불러오므로 일괄 평가에는 시간이 걸릴 수 있습니다. 무료 호스팅의 자원 제한 안에서 실제로 작동하는지는 배포 후 모든 지표로 확인해야 합니다.

## 입력 데이터 처리

공개 URL로 접속하면 입력 문장과 CSV가 호스팅 서버로 전송됩니다. 앱은 서버에 설치된 모델로 계산하며 외부 추론 API를 호출하지 않습니다. 앱 코드는 입력·결과를 파일이나 데이터베이스에 저장하지 않지만, 호스팅 서비스 자체의 로그·개인정보 정책은 별도로 적용됩니다. 세션이 끝나거나 서버가 재시작되기 전에 필요한 결과를 내려받으세요.

## 로컬 실행 유지

기존 `start.bat` 또는 `브라우저로 실행.bat`은 계속 `127.0.0.1:8517`에서 `app.py`를 실행합니다. 서버 주소·포트는 해당 실행 스크립트가 지정합니다. `.streamlit/config.toml`에는 클라우드 배포를 방해하는 로컬 주소·포트를 고정하지 않습니다.

## 배포 후 확인

1. 공개 URL을 로그인하지 않은 브라우저에서도 열 수 있는지 확인합니다.
2. 기본 예제의 SIM·독성·PPL과 J_proxy가 모두 계산되는지 확인합니다.
3. CSV 평가 및 결과 다운로드를 확인합니다.
4. 재시작 뒤 모델 다운로드·메모리 오류가 없는지 호스팅 로그를 확인합니다.

공식 절차: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
