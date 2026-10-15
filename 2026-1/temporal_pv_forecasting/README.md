# ☀️ Temporal PV Forecasting

### 시간적 특징 학습 기반 태양광 발전량 예측 및 특징 선택 실험

전국 태양광 발전량·기상 데이터를 대상으로 **시간적 특징(Lag, Rolling, Diff)** 을 생성하고, 입력 특징 구성에 따른 태양광 발전량 추정 성능을 비교하는 Python 연구 프로젝트입니다.

충북 지역 데이터를 예시로 **LightGBM, LSTM, PatchTST**를 학습·평가하며, **SHAP 중요도 분석**과 **특징 간 상관관계 필터링**을 이용해 유용한 입력 특징 조합을 탐색합니다. 원본 데이터의 정제부터 지역별 분리, 시간적 특징 생성, 모델 비교, 특징 중요도 분석, Top-K 재학습까지 단계별 실험 코드로 구성되어 있습니다.

> ⚠️ **실행 전 확인:** 이번에 제공된 파일은 `01`~`06` Python 스크립트 6개와 기존 `README.md`입니다. 실험 2·3이 직접 import하는 **`pv_exp_common.py`는 첨부되지 않았습니다.** 해당 파일을 별도로 준비하지 않으면 `05_exp2_shap_analysis.py`, `06_exp3_shap_corr_retrain.py`는 실행되지 않습니다. 원본 데이터 CSV도 별도 다운로드가 필요합니다.

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 연구 주제 | 시간적 특징 학습 기반의 고정밀 태양광 발전량 예측 모델 개발 |
| 분석 데이터 | 한국동서발전(주) 전국 태양광 발전량 예측값 학습데이터 |
| 분석 지역 | 전국 데이터 전처리 후 **충북 지역** 데이터를 이용한 실험 예시 |
| 예측 대상 | `정규화발전량 = 발전량(MWh) / 설비용량(MW)` |
| 입력 특징 | Time, Current, Lag, Rolling, Diff의 5개 그룹 |
| 특징 조합 | C1~C6 (6개 입력 특징 구성) |
| 비교 모델 | LightGBM, LSTM, PatchTST |
| 특징 분석 | SHAP 중요도 및 특징 간 상관관계 필터링 |
| 데이터 분할 | Train 2020~2022년 / Validation 2023년 / Test 2024년 |
| 주요 평가지표 | 주간 구간 NMAE(`DAY_NMAE_pct`) 및 MAE, RMSE, NRMSE, MBE, R² |

## 📁 프로젝트 폴더 구조

```text
temporal_pv_forecasting/
├── README.md
├── 01_data_preprocess.py         # 전국 태양광 데이터 정제
├── 02_region_split.py           # 시도별 CSV 분할
├── 03_feature_engineering.py    # 시간 공백 검출 및 특징 생성
├── 04_exp1_train.py             # 실험 1: C1~C6 · 3개 모델 성능 비교
├── 05_exp2_shap_analysis.py     # 실험 2: LightGBM SHAP 중요도
├── 06_exp3_shap_corr_retrain.py # 실험 3: SHAP + 상관관계 Top-K 재학습
│
├── 01. 전국 태양광 발전량 예측값 학습데이터.csv  # 별도 다운로드
├── solar_preprocessed.csv        # 01 실행 후 생성
├── 한국동서발전_지역별/             # 02 실행 후 생성
│   └── PV_Dataset_<지역명>.csv
├── chungbuk_pv_dataset.csv      # 지역별 결과에서 사용자가 준비
├── time_gap_report.csv          # 03 실행 후 생성
├── chungbuk_pv_features.csv     # 03 실행 후 생성
│
├── pv_training_outputs/         # 04 실행 후 생성
├── pv_exp2_outputs/             # 05 실행 후 생성
└── pv_exp3_outputs_LGM/         # 06 실행 후 생성
```

위 폴더 구조는 **실제 제공된 6개 스크립트의 원래 파일명을 기준**으로 작성했으며, 데이터 및 출력 폴더는 실행·준비 과정에서 추가됩니다. 첨부파일의 `(9)`, `(10)`, `(20261015-...)`와 같은 이름 구분자는 GitHub에 올릴 파일명이 아니라면 제거하여 위와 같이 정리하세요.

**별도로 확인할 파일**

| 파일 | 상태 및 역할 |
|---|---|
| `pv_exp_common.py` | **이번 첨부에 없음.** `05`, `06`에서 사용하는 공통 데이터 처리·학습·SHAP·평가 함수를 제공해야 함 |
| `requirements.txt` | 기존 README에는 언급되어 있으나 **이번 첨부에 없음.** 아래 패키지 설치 명령으로 환경 구성 가능 |

### 🗂️ Python 파일별 역할

| 파일 | 주요 기능 |
|---|---|
| `01_data_preprocess.py` | 지역·시간 중복 제거, 결측 처리, 짧은 연속 결측 보간, 물리 범위 확인, 야간 값 보정, 불완전한 날짜 제거 |
| `02_region_split.py` | `시도명` 기준으로 전처리 결과를 분리해 `PV_Dataset_<지역명>.csv` 저장 |
| `03_feature_engineering.py` | 시간 간격 확인, 연속 구간 ID 생성, 정규화 발전량 및 Time/Lag/Rolling/Diff 생성 |
| `04_exp1_train.py` | C1~C6 특징 조합으로 LightGBM·LSTM·PatchTST 학습, 예측 파일·모델·성능 결과 저장 |
| `05_exp2_shap_analysis.py` | C6_Full LightGBM을 고정된 최적 파라미터로 학습하고 검증 구간 기준 SHAP 분석 |
| `06_exp3_shap_corr_retrain.py` | SHAP 순위 우선으로 높은 상관관계의 특징을 제외한 뒤 Top-K 조합으로 모델 재학습 |

---

## 🔗 데이터셋 안내

| 항목 | 내용 |
|---|---|
| 데이터셋 | 한국동서발전(주)_전국 태양광 발전량 예측값 학습데이터 |
| 제공처 | 공공데이터포털 |
| 다운로드 | [공공데이터포털 데이터 페이지](https://www.data.go.kr/data/15151005/fileData.do?recommendDataYn=Y) |
| 원본 입력 파일명 | `01. 전국 태양광 발전량 예측값 학습데이터.csv` |
| 원본 읽기 인코딩 | `cp949` |
| 주요 컬럼 | `시도명`, `발전일자`, `설비용량(MW)`, `발전량(MWh)`, 기온·습도·풍속·일조·일사량 등 |

### 📥 다운로드 및 준비 방법

1. 위 데이터 페이지에서 CSV 파일을 다운로드합니다.
2. 압축파일로 제공되면 압축을 풀고, 원본 CSV를 프로젝트 최상위 폴더에 넣습니다.
3. 파일명을 `01. 전국 태양광 발전량 예측값 학습데이터.csv`로 맞추거나 `01_data_preprocess.py`의 `INPUT_PATH`를 변경합니다.
4. `01_data_preprocess.py` → `02_region_split.py` 순서로 실행합니다.
5. `한국동서발전_지역별/`에서 사용할 지역의 CSV를 선택해 **`chungbuk_pv_dataset.csv`** 라는 이름으로 프로젝트 최상위 폴더에 복사하거나, `03_feature_engineering.py`의 `INPUT_FILE`을 해당 실제 파일 경로로 수정합니다.

> 원본 데이터 파일은 이번 첨부에 포함되지 않았습니다. 데이터의 공개·활용 조건은 공공데이터포털 제공 페이지에서 확인해 주세요.

---

## 🔄 전체 데이터 처리 및 실험 흐름

```text
전국 태양광 발전량 원본 CSV
           │
           ▼
① 데이터 품질 처리 (01_data_preprocess.py)
   중복·결측·물리 범위·야간 조건·시간별 데이터 정리
           │  solar_preprocessed.csv
           ▼
② 지역별 데이터 분할 (02_region_split.py)
   전국 → 시도별 PV_Dataset_<지역명>.csv
           │  사용할 지역 CSV 선택
           ▼
③ 시간적 특징 생성 (03_feature_engineering.py)
   Time + Current + Lag + Rolling + Diff
           │  chungbuk_pv_features.csv
           ├────────────────────────────────┐
           ▼                                │
④ 실험 1: C1~C6 × 3개 예측 모델              │
   LightGBM · LSTM · PatchTST                │
           │                                │
           ▼                                ▼
⑤ 실험 2: C6 LightGBM의 SHAP 중요도 분석
           │  SHAP 순위 CSV
           ▼
⑥ 실험 3: SHAP 순위 + 상관관계 필터링
   Top-K 특징 집합 구성 및 모델 재학습
           │
           ▼
모델 · 예측 CSV · 성능 비교표 · SHAP 그래프
```

실험 2는 실험 1에서 도출한 설정값을 활용하지만, **현재 제공된 `05` 코드 안에 LightGBM 파라미터가 고정되어 있으므로** 실험 1의 출력 폴더를 직접 읽는 구조는 아닙니다. 반면 실험 3은 실험 2의 `shap_feature_importance.csv` 파일이 반드시 필요합니다.

---

## 🧹 1단계: 데이터 품질 처리 및 지역 분리

### 전국 데이터 전처리 — `01_data_preprocess.py`

| 단계 | 코드에 구현된 처리 |
|---|---|
| 기본 형식 | 발전일자 `datetime` 변환, 문자열 공백 제거, 수치형 데이터 변환 |
| 유일성 | (`시도명`, `발전일자`)가 같은 행은 **결측이 가장 적은 행**을 우선 유지 |
| 완전성 | 결측률 80% 이상인 컬럼 제거, 야간 결측값 0 대체, 지역별 단기 결측 선형보간 |
| 단시간 보간 | 기온·습도·풍속 최대 6개 연속 관측치, 적운량 최대 3개, 일조·일사량 최대 2개 |
| 유효성 | 설비용량·발전량·기온·습도·풍속·운량·일사량 등의 허용 범위 확인 |
| 일관성 | 대기권밖일사량과 일사량이 모두 0이면 발전량을 0으로 보정 |
| 정확성 | 1시간 자료를 전제로 `발전량(MWh) <= 설비용량(MW)` 규칙 적용 |
| 날짜 단위 정리 | 지역·날짜별 남은 관측 행이 24개가 아닌 날짜 제거 |

보간 허용 길이는 **시간차를 직접 측정하는 것이 아니라 연속 결측 관측치 개수**로 판단합니다. 데이터가 실제 1시간 간격일 때 위의 6개·3개·2개는 각각 최대 6시간·3시간·2시간에 대응합니다.

### 지역별 분할 — `02_region_split.py`

전처리 결과 `solar_preprocessed.csv`를 `시도명` 컬럼 기준으로 나누어 `한국동서발전_지역별/`에 저장합니다. 출력 파일명은 실제 데이터에 기록된 **지역명**을 사용합니다.

---

## 🧩 2단계: 시간적 특징 생성

`03_feature_engineering.py`는 충북 데이터를 예시로 다음 특징을 생성합니다.

| 특징 종류 | 주요 구성 | 코드 설정 |
|---|---|---|
| **Time** | 연·월·일·시각·연중 일수, 시각의 sin/cos 변환 | `year`, `month`, `day`, `hour`, `doy`, `sin_hour`, `cos_hour` |
| **Current** | 현재 기상·일조·일사량 관련 변수 | 기온, 습도, 풍속, 적운량, 일조, 대기권밖일사량, 일사량 |
| **Lag** | 과거 관측값 | 1, 3, 24개 관측치 전 |
| **Rolling** | 과거 관측값의 이동평균 | 직전 시점부터 3개, 24개 구간 |
| **Diff** | 과거 값 사이의 변화량 | `shift(1) - shift(1 + d)`, d=1, 24 |

Lag·Rolling·Diff는 **8개 기준 변수**(기온, 습도, 풍속, 적운량, 일조, 대기권밖일사량, 일사량, 정규화발전량)에서 생성합니다. 각 기준 변수마다 Lag 3개, Rolling 2개, Diff 2개로 **총 56개 시간적 파생 특징**을 만듭니다.

### 시간 공백 처리

시간 간격이 1시간이 아니면 새로운 `segment_id`를 부여하고, 파생 특징을 각 연속 구간 안에서 계산하여 시간적으로 떨어진 관측치를 연결하지 않도록 합니다. 시간 간격이 다른 지점은 `time_gap_report.csv`에 기록합니다.

또한 파생 특징 생성 결과에서 결측값이 발생한 날짜는 **해당 날짜의 행 전체를 제거**하므로, 실제 학습에 사용하는 날짜와 샘플 수는 원본 데이터보다 줄어들 수 있습니다.

### 예측 대상 및 단위 복원

모델은 설비용량의 차이를 고려하기 위해 정규화한 발전량을 사용합니다.

```text
정규화발전량 = 발전량(MWh) / 설비용량(MW)
예측 발전량(MWh) = 예측 정규화발전량 × 설비용량(MW)
```

원본 코드에서는 정규화발전량을 목표값으로 모델을 학습하고, 예측치를 설비용량으로 환산해 **MWh 단위 오차**를 평가합니다. 이러한 환산은 데이터가 1시간 간격으로 정리되어 있다는 전제에서 해석해야 합니다.

---

## 🧠 3단계: 입력 특징 조합 및 예측 모델 — 실험 1

### 특징 그룹 및 조합

| 그룹 | 의미 |
|---|---|
| `G1_Time` | 시간 및 주기성 특징 |
| `G2_Current` | 현재 시점 기상·일사 특징 |
| `G3_Lag` | 과거 값 특징 |
| `G4_Rolling` | 과거 이동평균 특징 |
| `G5_Diff` | 과거 변화량 특징 |

| Case | 사용 그룹 | 구성 목적 |
|---|---|---|
| **C1** | Time + Current | 현재 정보 기반 기준 실험 |
| **C2** | Time + Current + Lag | 과거 값 추가 효과 |
| **C3** | Time + Current + Rolling | 이동평균 추가 효과 |
| **C4** | Time + Current + Diff | 변화량 추가 효과 |
| **C5** | Time + Current + Lag + Rolling | 두 시간 특징의 결합 효과 |
| **C6** | Time + Current + Lag + Rolling + Diff | 전체 특징 사용 기준 |

모든 예정 컬럼이 존재할 경우 위 조합의 입력 차원은 각각 **13 / 37 / 29 / 29 / 53 / 69개**입니다. 실제 사용 특징 수는 CSV의 컬럼 존재 여부에 따라 달라질 수 있습니다. `year`는 데이터 분할용으로 사용되지만 학습 특징에서는 제외됩니다.

### 비교 모델

| 모델 | 분석 방식 |
|---|---|
| **LightGBM** | 입력 특징을 사용하는 트리 기반 부스팅 회귀 |
| **LSTM** | 연속 시계열을 입력받는 순환 신경망 |
| **PatchTST** | 시계열을 패치 단위로 처리하는 Transformer 계열 모델 |

`04_exp1_train.py`는 Optuna 기반 하이퍼파라미터 탐색, 모델 저장, Validation/Test 예측 CSV 생성, 성능 요약 및 모델별 선택된 조합의 **공통 시각 교집합 비교**를 수행합니다.

### 학습·검증·테스트 데이터 구분

| 구분 | 기간 | 용도 |
|---|---|---|
| **Train** | 2020~2022년 | 모델 학습 |
| **Validation** | 2023년 | 하이퍼파라미터 및 모델 설정 선택 |
| **Test** | 2024년 | 최종 성능 확인 |

평가는 전체 시간(`ALL_`)과 **대기권밖일사량계산값이 0보다 큰 주간 구간(`DAY_`)** 을 나누어 진행합니다. 주요 비교 지표는 주간 NMAE이며, MAE·RMSE·NRMSE·MBE·R²도 함께 계산합니다.

> 주간 NMAE는 코드에서 각 시각의 `|예측 MWh − 실제 MWh| / 설비용량(MW)`를 평균해 100을 곱하는 방식입니다. 일반적인 발전량 범위 기준 NMAE 등과 분모가 다를 수 있으므로 다른 연구와 수치를 비교할 때 정의를 확인해야 합니다.

---

## 🔍 4단계: SHAP 특징 중요도 분석 — 실험 2

`05_exp2_shap_analysis.py`는 **C6_Full의 LightGBM 모델**을 대상으로, 검증 기간에서 추출한 샘플의 SHAP 값을 계산합니다.

- 개별 특징의 SHAP 중요도 순위를 산출합니다.
- 특징을 Time·Current·Lag·Rolling·Diff 그룹으로 구분해 그룹별 중요도를 집계합니다.
- 상위 특징의 막대그래프와 SHAP Beeswarm 시각화를 생성합니다.
- 다음 실험에서 사용할 특징 중요도 순위 CSV를 저장합니다.

실험 2의 모델 파라미터는 스크립트의 `best_params`에 **고정된 값**이 정의되어 있습니다. `--n_trials_lgb` 옵션이 존재하더라도 고정 파라미터 전달 여부를 함께 확인해야 하며, 이를 매번 새로운 탐색을 수행한다는 의미로 해석하면 안 됩니다.

### 주요 출력 파일

```text
pv_exp2_outputs/
├── experiment_config.json
├── experiment_2_shap/
│   ├── shap_feature_importance.csv
│   ├── shap_group_importance.csv
│   ├── shap_values_valid_sample.npy
│   ├── shap_input_valid_sample.csv
│   ├── shap_top30_bar.png
│   └── shap_beeswarm_top30.png
└── results/
    ├── exp2_c6_lightgbm_summary.csv
    ├── exp2_c6_features.csv
    └── exp2_top_features_preview.csv
```

실제 SHAP 계산 및 그래프 저장 기능은 `pv_exp_common.py`의 `compute_shap_importance` 구현에 의존합니다. 위 파일 목록은 제공된 스크립트에 기재된 **예상 산출물**입니다.

---

## 🎯 5단계: SHAP + 상관관계 필터링 — 실험 3

`06_exp3_shap_corr_retrain.py`는 실험 2에서 얻은 SHAP 중요도 순위를 입력받아 상관성이 높은 특징을 정리한 뒤 Top-K 조합을 구성합니다.

1. SHAP 순위가 높은 특징을 먼저 후보로 유지합니다.
2. **Train(2020~2022년)** 데이터로 특징 간 Pearson 상관계수를 계산합니다.
3. 이미 유지된 특징과의 **절댓값 상관계수가 0.95 이상**인 후순위 특징을 제외합니다.
4. 필터링된 후보 특징을 SHAP 순서대로 Top-K로 구성합니다.
5. 각 특징 조합으로 요청한 모델을 재학습하여 주간 예측 성능을 비교합니다.

| 항목 | 기본 설정 |
|---|---|
| 상관계수 방법 | Pearson (`--corr_method pearson`) |
| 제외 기준 | `abs(correlation) >= 0.95` |
| Top-K 후보 | 1~58 (`--topk_list`에서 변경 가능) |
| 기본 재학습 모델 | **LightGBM만 실행** |
| 선택 가능한 모델 | LightGBM, LSTM, PatchTST |
| 모델 선택 지표 | Validation 주간 NMAE |
| 출력 폴더 | `pv_exp3_outputs_LGM/` |

여기서 **Top-K는 상관관계 필터링 후 실제 남은 특징 수를 초과하지 않도록 조정**됩니다. 따라서 Top-58까지 설정되어 있어도 최종 후보가 58개보다 적으면 그보다 적은 특징까지만 실험합니다. `--models lightgbm,lstm,patchtst`로 세 모델을 모두 포함할 수 있지만, 학습 시간과 계산 자원이 크게 늘어납니다.

### 주요 결과 파일

```text
pv_exp3_outputs_LGM/
├── experiment_config.json
├── results/
│   ├── exp3_feature_target_correlation.csv
│   ├── exp3_abs_feature_correlation_matrix.csv
│   ├── exp3_shap_corr_candidate_features.csv
│   ├── exp3_high_corr_excluded_features.csv
│   ├── features_SHAP_Corr0p95_Top<K>.csv
│   ├── exp3_all_topk_all_models_summary.csv
│   └── MAIN_exp3_individual_daytime_results.csv
└── models/   # 재학습 모델 저장 폴더
```

---

## 🛠️ 실행 환경 및 설치

| 구분 | 기술 |
|---|---|
| 언어 | Python |
| 데이터 처리 | pandas, NumPy |
| 회귀·전처리 | scikit-learn, LightGBM |
| 딥러닝 | PyTorch |
| 최적화 | Optuna |
| 설명가능성 분석 | SHAP |
| 산출물 저장 | joblib, CSV, JSON, NumPy |

### 1. 가상환경 생성 (선택)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 2. 라이브러리 설치

```bash
python -m pip install pandas numpy scikit-learn lightgbm optuna shap matplotlib joblib torch
```

PyTorch는 CPU/CUDA 환경에 따라 설치 방법이 달라질 수 있습니다. `requirements.txt`는 이번에 제공되지 않았으므로, 위 명령은 **제공된 코드의 주요 의존성을 기준으로 작성한 설치 예시**입니다.

### 3. 실행 전 점검

- 프로젝트 최상위 폴더를 현재 작업 디렉터리로 지정합니다.
- `01`~`06` 파일을 각각 `01_data_preprocess.py` ... `06_exp3_shap_corr_retrain.py`라는 **원래 파일명**으로 배치합니다.
- 실험 2·3까지 진행하려면 `pv_exp_common.py`와 그 안의 공통 함수를 별도로 준비합니다.
- 데이터셋에 2020~2024년 자료가 실제로 들어 있는지 확인합니다.

---

## ▶️ 단계별 실행 방법

모든 명령은 **`temporal_pv_forecasting` 프로젝트 최상위 폴더**에서 실행하는 예시입니다.

### ① 전국 데이터 전처리

```bash
python 01_data_preprocess.py
```

생성: `solar_preprocessed.csv`

### ② 지역별 CSV 분할

```bash
python 02_region_split.py
```

생성: `한국동서발전_지역별/PV_Dataset_<지역명>.csv`

**중요:** 분할 결과의 지역명을 먼저 확인하고, 사용하려는 지역 데이터를 `chungbuk_pv_dataset.csv`로 복사하거나 다음 단계 코드의 입력 경로를 변경합니다.

### ③ 시간적 특징 생성

```bash
python 03_feature_engineering.py
```

생성: `chungbuk_pv_features.csv`, `time_gap_report.csv`

### ④ 실험 1 — C1~C6 특징 조합별 모델 비교

```bash
python 04_exp1_train.py --csv chungbuk_pv_features.csv
```

빠른 동작 확인용:

```bash
python 04_exp1_train.py --csv chungbuk_pv_features.csv --models lightgbm --quick_test
```

기본 설정은 **6개 특징 조합 × 3개 모델**, 모델·조합별 Optuna 최대 50회 탐색이며, LSTM/PatchTST의 최대 Epoch는 200입니다. 전체 실행은 계산량이 큽니다. `--quick_test`는 정식 실험 결과를 위한 설정이 아닙니다.

### ⑤ 실험 2 — SHAP 중요도 분석

**필수:** `pv_exp_common.py`가 있어야 합니다.

```bash
python 05_exp2_shap_analysis.py --csv chungbuk_pv_features.csv
```

빠른 점검용:

```bash
python 05_exp2_shap_analysis.py --csv chungbuk_pv_features.csv --quick_test
```

생성 예정: `pv_exp2_outputs/experiment_2_shap/shap_feature_importance.csv` 등

### ⑥ 실험 3 — 상관관계 필터링 및 Top-K 재학습

**필수:** `pv_exp_common.py`와 실험 2에서 만든 SHAP 순위 CSV가 있어야 합니다.

```bash
python 06_exp3_shap_corr_retrain.py
```

선택한 특징 개수만 빠르게 확인하려면:

```bash
python 06_exp3_shap_corr_retrain.py --topk_list 5,10,15 --quick_test
```

세 모델을 대상으로 실행하려면:

```bash
python 06_exp3_shap_corr_retrain.py --models lightgbm,lstm,patchtst --topk_list 5,10,15
```

> `--quick_test`는 내부에서 Top-K 목록을 `5,10`으로 덮어쓰므로, **원하는 `--topk_list`를 그대로 테스트하려면 `--quick_test`를 빼야 합니다.**

---

## 📊 생성 결과 및 평가 항목

| 실험 | 주요 출력 | 활용 |
|---|---|---|
| 실험 1 | `pv_training_outputs/results/summary_results.csv` | 전체 모델·특징 조합 성능 비교 |
| 실험 1 | `pv_training_outputs/results/MAIN_individual_daytime_results.csv` | 주간 성능 요약 |
| 실험 1 | `pv_training_outputs/results/best_case_by_model_validation.csv` | 모델별 Validation 기준 최적 특징 조합 확인 |
| 실험 1 | `pv_training_outputs/models/`, `predictions/` | 학습 모델과 시각별 예측 결과 |
| 실험 2 | `shap_feature_importance.csv`, `shap_group_importance.csv` | 개별 특징 및 특징 그룹의 중요도 |
| 실험 2 | `shap_top30_bar.png`, `shap_beeswarm_top30.png` | SHAP 시각화 |
| 실험 3 | `exp3_shap_corr_candidate_features.csv` | 상관관계 필터링 후 후보군 |
| 실험 3 | `exp3_high_corr_excluded_features.csv` | 높은 상관관계로 제외된 특징 |
| 실험 3 | `exp3_all_topk_all_models_summary.csv` | Top-K 재학습 결과 비교 |

**이 README에는 실제 학습 성능 수치를 기재하지 않았습니다.** 이번 첨부에는 원본 데이터와 실행 결과 CSV가 포함되지 않았기 때문에, MAE/NMAE 등의 수치를 임의로 작성하지 않았습니다.

---

## ⚠️ 실행 및 결과 해석 시 주의사항

1. **공통 모듈 누락:** 현재 제공된 `05`, `06`은 `pv_exp_common.py`의 함수에 의존합니다. 기존 README에는 해당 파일이 있다고 적혀 있지만 이번 첨부에는 없어, 실제 파일을 준비하기 전에는 해당 실험을 실행할 수 없습니다. `04_exp1_train.py`의 함수명과 `05`·`06`이 요구하는 공통 함수명도 일부 다르므로 단순히 파일명을 변경하는 방식으로 해결되지 않습니다.
2. **데이터 파일과 경로:** `02`가 생성하는 파일은 `PV_Dataset_<실제 시도명>.csv`이지만 `03`은 `chungbuk_pv_dataset.csv`를 읽습니다. 파일명·경로를 맞춘 후 실행해야 합니다.
3. **시간적 특징의 의미:** 과거 발전량의 Lag·Rolling·Diff가 입력에 포함됩니다. 미래 시점 예측에 활용하려면 예측 시점에 실제로 확보 가능한 정보만 사용하는지 확인해야 하며, 현재 시점의 기상 변수도 실제 가용 시점을 검토해야 합니다.
4. **날짜 전체 삭제:** 전처리 단계에서는 24개 시간 관측값을 갖추지 않은 날짜를 제외하고, 특징 생성 후에는 결측 특징이 하나라도 존재하는 날짜를 통째로 제거합니다. 제거 전후 데이터 수와 계절별 분포를 검토하세요.
5. **1시간 간격 전제:** 24시간 Lag, 하루 24개 관측값 검사, MWh/MW 정규화와 성능 해석은 1시간 간격 자료를 전제로 합니다. 실제 데이터 주기 및 단위를 확인해야 합니다.
6. **데이터 누수 방지:** 상관관계 필터링은 학습 구간에만 적용하지만, 전처리의 선형보간은 해당 행의 미래 관측값을 참조할 수 있습니다. 엄격한 미래 예측을 평가하려면 정보 사용 시점을 추가로 점검해야 합니다.
7. **실험 2의 하이퍼파라미터:** `05`에는 고정 LightGBM 파라미터가 입력되어 있습니다. 실험 1 결과에서 자동으로 불러와 재최적화하는 과정으로 오해하지 않도록 주의해야 합니다.
8. **Top-K 설정:** 실험 3의 기본 목록은 1~58개이나 실제 후보 수에 따라 축소되며, `--quick_test` 사용 시에는 입력한 목록과 관계없이 `5,10`으로 변경됩니다.
9. **결과 비교 기준:** 모델마다 예측 가능한 날짜·시각이 다를 수 있으므로, 개별 결과와 동일 시각 기준의 공통 비교 결과를 구분해서 해석해야 합니다.
10. **실행 검증 상태:** 이 문서는 제공된 **소스코드와 기존 README를 검토하여 작성한 사용 안내**입니다. 원본 CSV, 공통 모듈 및 실행 결과가 모두 제공되지 않아 실제 학습 성공이나 모델 성능을 검증한 것은 아닙니다.

---

## 🎯 프로젝트 목적

시간적 특징의 구성 방식에 따른 태양광 발전량 예측 성능 변화를 비교하고, SHAP 중요도와 특징 간 상관관계를 이용해 불필요한 중복 정보를 줄이면서 효과적인 특징 집합을 찾는 것이 목적입니다. 이를 통해 **입력 특징의 구성과 선택이 예측 모델 성능에 미치는 영향**을 체계적으로 분석할 수 있는 실험 환경을 제공합니다.

### 📚 데이터 출처

한국동서발전(주), *전국 태양광 발전량 예측값 학습데이터*, 공공데이터포털.  
https://www.data.go.kr/data/15151005/fileData.do?recommendDataYn=Y
