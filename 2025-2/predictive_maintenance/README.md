# 🛠️ predictive_maintenance

**AI4I 2020 설비 데이터를 활용한 머신러닝 기반 고장 예측 및 InfluxDB·Grafana 모니터링 프로젝트**

Python으로 설비 운전 데이터를 InfluxDB에 저장하고, **XGBoost로 고장 발생 여부**를 판별한 뒤 **LightGBM으로 고장 유형**을 분류합니다. 예측 결과는 다시 InfluxDB에 기록하여 Grafana에서 시각화할 수 있도록 구성했습니다.

> 이 프로젝트는 공개된 **합성(synthetic) 설비 데이터**를 사용한 예제입니다. 실제 설비에서 실시간으로 데이터를 수집하거나 미래의 고장을 미리 예측하는 상용 시스템은 아닙니다.

---

## 📌 프로젝트 개요

주요 작업은 다음과 같습니다.

- AI4I 2020 데이터셋을 InfluxDB에 적재하고 설비 운전 상태를 시계열 형태로 저장
- InfluxDB에서 데이터를 조회하여 이상 범위 및 결측값 처리
- **고장 여부 이진 분류(XGBoost)** 및 **고장 유형 다중 분류(LightGBM)** 학습·평가
- 예측 결과와 고장 확률을 InfluxDB에 다시 저장
- Grafana와 InfluxDB를 연동하여 설비 상태 및 예측 결과를 시각화할 수 있는 환경 구성

## 📂 프로젝트 폴더 구조

```text
predictive_maintenance/
├── docker-compose.yml          # InfluxDB 및 Grafana 컨테이너 실행 설정
├── requirements.txt            # Python 라이브러리 목록
├── README.md                   # 프로젝트 설명 문서
├── data/                       # 원본 데이터 및 전처리 결과 (사용자가 생성)
│   ├── ai4i2020.csv            # AI4I 2020 원본 데이터 (별도 다운로드)
│   └── ai4i_preprocessed.csv   # 02단계 실행 후 생성
└── src/
    ├── 01_csv_to_influx.py       # 원본 CSV → InfluxDB 적재
    ├── 02_fetch_and_preprocess.py # InfluxDB 조회 및 데이터 전처리
    ├── 03_train_models.py        # XGBoost·LightGBM 학습 및 평가
    └── 04_predict_to_influx.py   # 예측 생성 및 InfluxDB 저장
```

> `data/` 폴더 및 위에 기재된 두 CSV는 **프로젝트 실행에 필요한/실행 후 생성되는 파일을 보여주는 구조**입니다. 사용자가 별도로 `data/` 폴더를 만들고 원본 CSV를 넣어야 합니다.

## 🗂️ 데이터셋 및 원본 데이터 출처

- **데이터셋:** AI4I 2020 Predictive Maintenance Dataset
- **제공처:** UCI Machine Learning Repository
- **데이터 규모:** 10,000개 레코드의 합성 설비 데이터
- **사용 파일:** `ai4i2020.csv`
- **라이선스:** CC BY 4.0 (출처 표기 필요)

**🔗 [AI4I 2020 공식 데이터셋 다운로드](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)**

### 📥 데이터 다운로드 및 준비 방법

1. 위 UCI 데이터셋 페이지에 접속해 파일을 다운로드합니다.
2. 다운로드한 압축파일이 있다면 해제한 뒤 `ai4i2020.csv`를 찾습니다.
3. 프로젝트 최상위 경로에 `data` 폴더를 생성합니다.
4. `ai4i2020.csv`를 `data/` 폴더에 넣습니다.
5. `src/01_csv_to_influx.py`가 `../data/ai4i2020.csv`를 읽으므로 **실행 시 작업 디렉터리는 `src`**로 설정합니다.

> ⚠️ 이전 README에 기재되어 있던 `https://www.unb.ca/cic/datasets/vpn.html`은 **VPN 네트워크 트래픽 데이터셋** 페이지이며, 본 프로젝트의 AI4I 데이터 출처가 아닙니다. 위의 **UCI AI4I 2020** 링크를 사용해야 합니다.

### 📋 주요 입력 변수 및 정답 레이블

| 구분 | 원본 데이터 컬럼 | 내용 |
| --- | --- | --- |
| 입력 | `Air temperature [K]` | 주변 공기 온도 |
| 입력 | `Process temperature [K]` | 공정 온도 |
| 입력 | `Rotational speed [rpm]` | 회전 속도 |
| 입력 | `Torque [Nm]` | 토크 |
| 입력 | `Tool wear [min]` | 공구 마모 시간 |
| 정답 | `Machine failure` | 설비 고장 여부 (0/1) |
| 정답 | `TWF`, `HDF`, `PWF` | 공구 마모·열 방출·동력 관련 고장 표시 |
| 부가 정보 | `Type` | 설비/제품 유형 (InfluxDB 태그로 저장) |

> 원본 데이터에는 `OSF`, `RNF` 등의 다른 고장 표시도 있지만, **현재 코드에서 세부 고장 유형 모델의 학습 대상으로 사용하는 것은 TWF·HDF·PWF 세 유형**입니다.

## ⚙️ 시스템 구성 및 데이터 처리 흐름

```text
AI4I 원본 데이터 (data/ai4i2020.csv)
        │
        ▼
01_csv_to_influx.py
        │  센서값·고장 레이블 저장
        ▼
InfluxDB (bucket: ai4i / measurement: machine_status)
        │
        ▼
02_fetch_and_preprocess.py
        │  데이터 조회, 이상 범위 필터링, 결측치 제거, 레이블 생성
        ▼
data/ai4i_preprocessed.csv
        │
        ├──► 03_train_models.py
        │         ├─ XGBoost: 고장 여부 분류·평가
        │         └─ LightGBM: 고장 유형 분류·평가
        │
        └──► 04_predict_to_influx.py
                  ├─ 전체 데이터로 두 모델을 별도 재학습
                  ├─ 고장 확률·고장 여부·고장 유형 예측
                  └─ InfluxDB (measurement: prediction)에 기록
                                  │
                                  ▼
                            Grafana 시각화
```

### 🔍 Python 파일별 주요 기능

| 실행 단계 | 파일 | 주요 내용 |
| --- | --- | --- |
| **1. 데이터 적재** | `01_csv_to_influx.py` | 원본 CSV를 읽어 설비 센서값과 고장 레이블을 `machine_status` 측정값으로 저장 |
| **2. 데이터 전처리** | `02_fetch_and_preprocess.py` | Flux 쿼리로 InfluxDB 데이터 조회, 이상 범위·결측값 필터링, 고장 유형 라벨 구성 후 CSV 저장 |
| **3. 모델 학습·평가** | `03_train_models.py` | XGBoost 이진 분류 및 LightGBM 3종 고장 유형 분류; 혼동행렬과 분류 리포트 출력 |
| **4. 예측 결과 저장** | `04_predict_to_influx.py` | 전체 전처리 데이터를 사용하여 모델을 새로 학습하고, 예측 레코드를 InfluxDB의 `prediction`에 저장 |

**타임스탬프 참고:** `01_csv_to_influx.py`는 원본에 실제 시각 정보가 없으므로 **2020-01-01 00:00:00부터 레코드별 1분 간격의 가상 시각**을 생성합니다. 그래프의 날짜·시간은 실제 장비 측정 시각이 아닙니다.

## 🧠 적용 예측 모델

| 모델 | 예측 대상 | 구현 내용 |
| --- | --- | --- |
| **XGBoost** (`XGBClassifier`) | 고장 발생 여부 (`machine_failure`: 0/1) | 불균형 클래스에 `scale_pos_weight` 적용, 고장 확률 및 예측 클래스 계산 |
| **LightGBM** (`LGBMClassifier`) | 고장 유형 (`failure_type`) | 고장 샘플 중 **TWF·HDF·PWF**만 사용하여 3개 클래스로 분류 |

### 🔧 주요 학습 및 평가 설정

| 항목 | 설정값 |
| --- | --- |
| 입력 특징 | 공기 온도, 공정 온도, 회전 속도, 토크, 공구 마모 시간 (**5개**) |
| 학습 / 테스트 비율 | **80% / 20%** |
| 데이터 분할 | `random_state=42`, 클래스별 비율을 고려한 `stratify` |
| XGBoost | `n_estimators=300`, `max_depth=5`, `learning_rate=0.05` |
| LightGBM | `n_estimators=300`, `learning_rate=0.05`, `num_class=3` |
| 성능 확인 | 혼동행렬(`confusion_matrix`), 정밀도·재현율·F1 등을 포함한 `classification_report` |

고장 유형은 전처리 과정에서 `1×TWF + 2×HDF + 3×PWF`로 만들어집니다. 세부 분류 모델은 이 중 **1(TWF), 2(HDF), 3(PWF)**에 해당하는 샘플만 사용하며 다른 값은 학습에서 제외합니다.

> **모델 재사용 참고:** `03_train_models.py`는 학습·평가 결과를 출력하지만 학습된 모델 파일을 저장하지 않습니다. `04_predict_to_influx.py`는 3단계 모델을 로드하는 대신 **전체 전처리 데이터로 모델을 다시 학습**합니다. 따라서 4단계 결과는 독립 테스트셋의 예측 성능을 의미하지 않습니다.

## 🐳 개발 환경 및 사용 기술

| 구분 | 도구·버전 | 역할 |
| --- | --- | --- |
| 운영 환경 | Ubuntu, Docker / Docker Compose | 분석 및 서비스 실행 환경 |
| Python | 3.12.3 (기존 README 기준) | 데이터 처리 및 머신러닝 구현 |
| InfluxDB | 기존 실행 환경 2.7.12 / Compose 이미지 `influxdb:2.7` | 시계열 데이터 저장 |
| Grafana | 기존 실행 환경 12.2.1 / Compose 이미지 `grafana/grafana:latest` | 대시보드 시각화 |
| pandas·NumPy | `requirements.txt` 참조 | 데이터 처리 |
| XGBoost·LightGBM·scikit-learn | `requirements.txt` 참조 | 분류 모델 및 성능 평가 |
| influxdb-client | `requirements.txt` 참조 | InfluxDB 데이터 읽기·쓰기 |

> `docker-compose.yml`은 Grafana 버전을 `latest`로 사용하므로 **항상 Grafana 12.2.1이 설치된다는 의미는 아닙니다.** 재현성을 높이려면 검증한 버전으로 이미지 태그를 고정해야 합니다.

## 🔐 실행 전 설정

`docker-compose.yml` 및 InfluxDB에 접근하는 Python 파일에는 다음 설정값이 사용됩니다.

| 설정 | 현재 코드 값 | 설명 |
| --- | --- | --- |
| InfluxDB URL | `http://localhost:8086` | InfluxDB 접속 주소 |
| Organization | `my-org` | InfluxDB 조직명 |
| Bucket | `ai4i` | 데이터 저장 버킷 |
| InfluxDB Token | `change-me` | 실제 사용할 토큰으로 교체 필요 |
| InfluxDB / Grafana 비밀번호 | `change-me` | 운영 환경에 맞게 변경 필요 |

1. `docker-compose.yml`의 `change-me`로 표시된 비밀번호 및 InfluxDB 관리자 토큰을 변경합니다.
2. `src/01_csv_to_influx.py`, `src/02_fetch_and_preprocess.py`, `src/04_predict_to_influx.py`의 `TOKEN`도 동일한 실제 토큰에 맞게 수정합니다.
3. `ORG` 및 `BUCKET` 값을 수정한 경우, 각 Python 파일과 Compose 설정을 동일하게 맞춥니다.
4. **실제 비밀번호·토큰이 포함된 수정 파일은 공개 GitHub 저장소에 커밋하지 마세요.** 현재 코드는 환경변수를 통해 토큰을 읽도록 구현되어 있지 않으므로 직접 수정 시 특히 주의가 필요합니다.

## 🚀 설치 및 실행 방법

아래 명령어는 **Ubuntu / Linux 터미널에서 프로젝트 최상위 폴더를 기준**으로 실행하는 예시입니다. 사전에 Docker 및 Docker Compose, Python 3가 설치되어 있어야 합니다.

### 1. 프로젝트 폴더로 이동

```bash
cd predictive_maintenance
```

### 2. InfluxDB 및 Grafana 컨테이너 실행

```bash
docker compose up -d
```

컨테이너가 정상 실행 중인지 확인하려면 다음 명령어를 사용합니다.

```bash
docker compose ps
```

### 3. 가상환경 생성 및 활성화

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. 필요한 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 5. Python 소스 폴더로 이동

```bash
cd src
```

### 6. 데이터 적재 → 전처리 → 학습·평가 → 예측 저장

아래 명령어를 **순서대로** 실행합니다.

```bash
python 01_csv_to_influx.py
python 02_fetch_and_preprocess.py
python 03_train_models.py
python 04_predict_to_influx.py
```

> 실행 전에 반드시 `data/ai4i2020.csv`를 준비해야 합니다. 2단계에서 생성한 `data/ai4i_preprocessed.csv`는 3·4단계에서 사용됩니다. 명령어를 `src` 폴더에서 실행하도록 작성되어 있으므로 다른 작업 디렉터리에서는 상대경로 오류가 날 수 있습니다.

## 📊 InfluxDB 및 Grafana 확인

| 서비스 | 접속 주소 | 용도 |
| --- | --- | --- |
| InfluxDB | [http://localhost:8086](http://localhost:8086) | 시계열 측정값·예측값 확인 |
| Grafana | [http://localhost:3000](http://localhost:3000) | 예측 결과 시각화 |

**InfluxDB에 저장되는 주요 measurement**

| Measurement | 데이터 내용 | 주요 필드 |
| --- | --- | --- |
| `machine_status` | 원본 센서값과 실제 고장 표시 | `air_temp`, `process_temp`, `rotational_speed`, `torque`, `tool_wear`, `machine_failure`, `twf`, `hdf`, `pwf` |
| `prediction` | 머신러닝 모델의 예측 결과 | `prob_failure`, `pred_failure`, `pred_failure_type` |

`prediction`의 주요 값은 **고장 확률(0~1)**, **고장 예측 여부(0/1)**, **예측된 고장 유형(0/1/2/3)**입니다. `pred_failure_type=0`은 예측된 고장이 없거나 유형을 지정하지 않은 경우이며, 1·2·3은 각각 TWF·HDF·PWF를 나타냅니다.

Grafana에서 시각화를 하려면 **InfluxDB 2.x 데이터 소스를 연결하고 적절한 패널·대시보드를 별도로 구성**해야 합니다. 현재 `docker-compose.yml`은 Grafana 서버를 실행하지만, 완성된 대시보드를 자동으로 생성하거나 가져오지는 않습니다.

## ⚠️ 사용 시 주의사항

- **학습용 데이터:** AI4I 2020은 공개된 합성 데이터이므로 실제 설비 현장에서의 성능을 보장하지 않습니다.
- **가상 타임스탬프:** 1단계에서 만든 시간축은 시각화를 위한 임의의 1분 간격으로, 원본 설비의 실제 시간 정보가 아닙니다.
- **고장 유형 범위:** 세부 고장 분류는 TWF·HDF·PWF만 대상으로 합니다. 원본의 모든 고장 원인을 분류하는 코드는 아닙니다.
- **평가와 시각화 구분:** 3단계는 학습·테스트 분할 평가를 수행하고, 4단계는 전체 데이터 재학습 및 예측 시각화용 결과 저장을 수행합니다.
- **Grafana 설정:** 별도의 InfluxDB 데이터 소스 연결과 대시보드 구성이 필요합니다.
- **보안:** `change-me`를 실제 값으로 교체하되, 토큰·비밀번호를 GitHub에 공개하지 마세요.
- **플랫폼 의존성:** `requirements.txt`에 CUDA/Linux 관련 `nvidia-nccl-cu12`가 포함되어 있어 운영체제에 따라 설치 호환성 확인이 필요합니다.

## 📚 참고 자료

- [AI4I 2020 Predictive Maintenance Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- [InfluxDB 2.x 문서](https://docs.influxdata.com/influxdb/v2/)
- [Grafana 문서](https://grafana.com/docs/grafana/latest/)

---

**프로젝트 목적:** 설비 데이터를 수집·저장하고, 고장 여부 및 고장 유형을 분류한 뒤, 예측 결과를 모니터링 환경으로 연결하는 **데이터 처리·머신러닝·시각화의 통합 흐름**을 학습합니다.
