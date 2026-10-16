# ☀️ pv_shap_topk_feature_analysis

**SHAP 중요도 기반 Top-K 특징 선택 및 모델 간 교차 적용을 통한 태양광 발전량 추정 성능 분석 프로젝트**

한국동서발전(주)의 태양광 발전 데이터를 바탕으로 **Random Forest, LightGBM, LSTM**의 입력 특징 중요도를 SHAP으로 분석합니다. 각 모델의 SHAP 순위에서 상위 K개 특징을 선택해 재학습하고, **Validation 성능이 가장 좋은 K**를 결정합니다. 이어서 한 모델이 선택한 특징 조합을 다른 모델에도 적용하여 **특징 선택의 모델 의존성 및 전이 가능성**을 분석합니다.

전체 **69개 특징(Full69)**을 사용하는 모델과 SHAP Top-K 모델의 발전량 추정 성능을 비교합니다.

---

## 📌 프로젝트 개요

주요 작업은 다음과 같습니다.

- **데이터 전처리:** 전국 태양광 발전 데이터의 결측·중복값 정리 및 충청북도 데이터 추출
- **특징 생성:** Time, Current, Lag, Rolling, Diff의 5개 그룹으로 **69개 입력 특징** 구성
- **기준 모델 학습:** Random Forest, LightGBM, LSTM의 Full69 모델 학습 및 성능 평가
- **모델별 SHAP 분석:** 동일한 검증 구간을 활용하여 특징 중요도와 순위 산출
- **SHAP Top-K 탐색:** 각 모델의 중요도 순위에 따라 Top-1부터 Top-69까지 후보를 평가하고 최적 K 선택
- **교차 적용 실험:** 모델별 최적 특징 집합을 다른 모델에 적용하여 추정 성능 비교
- **최종 성능 비교:** Full69와 SHAP Top-K 모델의 결과를 동일한 평가 조건에서 비교

| 항목 | 내용 |
| --- | --- |
| 분석 지역 | 충청북도 |
| 데이터 기간 | 2020~2024년 |
| 비교 모델 | Random Forest, LightGBM, LSTM |
| 기본 입력 특징 | 5개 그룹, 총 69개 |
| 특징 분석·선택 | SHAP 중요도, 모델별 Top-K 탐색 |
| 특징 선택 기준 | 2023년 Validation 주간 NMAE |
| 평가 지표 | MAE, RMSE, NMAE, NRMSE, MBE, R² |
| 개발 언어 | Python |

## 📂 프로젝트 폴더 구조

아래는 주요 Python 파일을 **저장소용 표준 파일명**으로 정리한 구조입니다.

```text
pv_shap_topk_feature_analysis/
├── 01_data_preprocess.py                  # 전국 태양광 발전 데이터 전처리
├── 02_region_split.py                     # 지역별 데이터 분리
├── 03_feature_engineering.py              # 충북 데이터의 69개 특징 생성
├── 04_exp1_train.py                       # LightGBM·LSTM Full69 모델 학습
├── 04_exp1_train_random_forest.py         # Random Forest 모델 학습
├── 04_compare_3models_common_test.py      # 3모델 기준 성능 비교
├── 05_exp2_run_shap_3models.py            # 모델별 SHAP 중요도 분석
├── 06_exp3_shap_topk_retrain.py           # SHAP Top-K 탐색 및 최적 K 재학습
├── 07_exp4_cross_model_best_features.py   # 최적 특징의 모델 간 교차 적용
├── 08_compare_full69_vs_shap_topk_3models.py # Full69와 Top-K 성능 비교
├── shap_common.py                         # SHAP 공통 함수·시각화
├── shap_lightgbm.py                       # LightGBM SHAP 계산
├── shap_random_forest.py                  # Random Forest SHAP 계산
├── shap_lstm.py                           # LSTM SHAP 계산
└── README.md                              # 프로젝트 설명 및 실행 안내
```

`shap_*.py` 보조 모듈은 `05_exp2_run_shap_3models.py`와 같은 폴더에 배치합니다. 원본 데이터와 실행 결과 폴더는 아래 안내에 따라 준비하거나 실행 과정에서 생성합니다.

## 🔗 데이터셋 및 다운로드 안내

**한국동서발전(주)의 전국 태양광 발전량 예측값 학습데이터**를 사용합니다.

| 항목 | 내용 |
| --- | --- |
| 제공기관 | 한국동서발전(주) |
| 데이터명 | 한국동서발전(주)_전국 태양광 발전량 예측값 학습데이터 |
| 다운로드 | [공공데이터포털 데이터 페이지](https://www.data.go.kr/data/15151005/fileData.do?recommendDataYn=Y) |
| 주요 정보 | 발전일자, 지역, 발전량, 설비용량, 기온, 습도, 풍속, 적운량, 일조, 일사량 등 |
| 분석 대상 | 전처리 후 추출한 충청북도 데이터 |

### 📥 데이터 준비 방법

1. [공공데이터포털](https://www.data.go.kr/data/15151005/fileData.do?recommendDataYn=Y)에서 원본 CSV 파일을 내려받습니다.
2. 다운로드한 CSV를 프로젝트 폴더에 배치하고, `01_data_preprocess.py`의 `INPUT_PATH`에 지정된 파일명과 일치시킵니다.
3. 다음 단계의 전처리·지역 분리·특징 생성 스크립트를 순서대로 실행합니다.

```text
pv_shap_topk_feature_analysis/
└── 01. 전국 태양광 발전량 예측값 학습데이터.csv
```

`01_data_preprocess.py`는 기본적으로 원본 CSV를 **CP949 인코딩**으로 읽도록 설정되어 있습니다. 내려받은 파일의 이름이나 인코딩이 다르면 해당 설정을 조정해야 합니다.

## ⚙️ 데이터 전처리 및 특징 생성

### 1. 데이터 처리 단계

| 단계 | 코드 | 주요 처리 | 생성 파일 |
| --- | --- | --- | --- |
| 1 | `01_data_preprocess.py` | 날짜·수치형 정리, 중복 제거 및 짧은 결측 구간 보간 | `solar_preprocessed.csv` |
| 2 | `02_region_split.py` | 전국 데이터를 시도별로 분리 | `한국동서발전_지역별/PV_Dataset_충북.csv` 등 |
| 3 | `03_feature_engineering.py` | 연속 시간 구간 구분, 시계열 파생 특징 생성 | `chungbuk_pv_features.csv` |

특징 생성 단계에서는 시간 간격이 끊어진 구간을 구분하고, Lag·Rolling·Diff 특징이 서로 다른 연속 구간을 가로질러 계산되지 않도록 처리합니다.

### 2. 입력 특징 구성

| 특징 그룹 | 구성 | 특징 수 |
| --- | --- | ---: |
| **Time** | 월·일·시·연중일 및 시간의 `sin/cos` 변환 | 6 |
| **Current** | 기온, 습도, 풍속, 적운량, 일조, 대기권밖일사량, 일사량 | 7 |
| **Lag** | 기초 변수 8개의 과거 1·3·24시간 값 | 24 |
| **Rolling** | 기초 변수 8개의 과거 3·24시간 이동평균 | 16 |
| **Diff** | 기초 변수 8개의 1·24시간 변화량 | 16 |
| **합계** | **C6_Full** | **69** |

Lag·Rolling·Diff의 기초 변수 8개에는 기상·일사 관련 7개 변수와 정규화발전량이 포함됩니다.

모델은 **정규화발전량**을 학습하고, 평가 시 설비용량을 이용해 실제 발전량 단위로 복원합니다.

```text
정규화발전량 = 발전량(MWh) / 설비용량(MW)
추정 발전량(MWh) = 추정 정규화발전량 × 설비용량(MW)
```

### 3. 데이터 분할

| 구분 | 기간 | 목적 |
| --- | --- | --- |
| Train | 2020~2022년 | 모델 학습 |
| Validation | 2023년 | 하이퍼파라미터 탐색 및 최적 K 선택 |
| Test | 2024년 | 최종 성능 평가 |

## 🧠 SHAP 기반 특징 분석 및 실험 구성

### 1. 전체 분석 흐름

```text
공공데이터포털 원본 CSV
         ↓
데이터 전처리 및 충북 데이터 추출
         ↓
Time / Current / Lag / Rolling / Diff (69개)
         ↓
Random Forest · LightGBM · LSTM 기준 모델 학습
         ↓
모델별 SHAP 특징 중요도 및 순위 분석
         ↓
SHAP Top-1 ~ Top-69 특징 조합 탐색
         ↓
2023년 Validation 기준 최적 K 선택
         ↓
선택 특징으로 모델 재학습 및 2024년 Test 평가
         ↓
모델 간 최적 특징 조합 교차 적용
         ↓
Full69 대비 성능 및 모델별 특징 선택 효과 비교
```

### 2. 주요 실험

| 실험 | 내용 | 관련 코드 |
| --- | --- | --- |
| **Full69 기준 모델** | 69개 특징을 사용하는 RF·LightGBM·LSTM 학습 및 비교 | `04_exp1_train.py`, `04_exp1_train_random_forest.py`, `04_compare_3models_common_test.py` |
| **SHAP 중요도 분석** | 기존 학습 모델에서 특징별 기여도와 모델별 중요도 순위 계산 | `05_exp2_run_shap_3models.py` |
| **SHAP Top-K 탐색** | 모델별 SHAP 상위 K개 특징으로 재학습하여 Validation 오차 비교 | `06_exp3_shap_topk_retrain.py` |
| **최적 특징 교차 적용** | 한 모델에서 선정한 최적 특징 집합을 다른 모델에 적용 | `07_exp4_cross_model_best_features.py` |
| **Full69 vs Top-K** | 저장된 기준 모델과 최종 Top-K 모델의 예측 성능 비교 | `08_compare_full69_vs_shap_topk_3models.py` |

### 3. 모델별 SHAP 분석 방식

| 모델 | SHAP 방식 | 중요도 산출 |
| --- | --- | --- |
| Random Forest | `TreeExplainer` | 특징별 SHAP 절댓값 평균 |
| LightGBM | `TreeExplainer` | 특징별 SHAP 절댓값 평균 |
| LSTM | `GradientExplainer` | 시간축 SHAP 절댓값을 합산한 뒤 샘플 평균 |

SHAP 분석에는 **2023년 Validation의 공통 주간 시점**을 사용합니다. 모델별 원시 SHAP 값의 크기를 직접 비교하기보다, **정규화된 중요도와 특징 순위**를 이용해 모델 간 차이를 살펴봅니다.

### 4. 최적 K 선택 및 교차 적용

`06_exp3_shap_topk_retrain.py`는 각 모델에서 SHAP 중요도 순위 상위 K개만 사용하는 여러 모델을 평가한 뒤 **Validation 주간 NMAE가 최소인 K**를 선택합니다. 최적 K를 고른 이후에 선택된 구성에 대해서만 실제 Test 데이터를 이용한 최종 평가를 수행합니다.

이어서 `07_exp4_cross_model_best_features.py`는 모델별로 선택된 최적 특징을 서로 교차 적용합니다.

| 특징 선택 기준 ↓ / 학습 모델 → | Random Forest | LightGBM | LSTM |
| --- | --- | --- | --- |
| RF의 최적 Top-K | 적용 | 적용 | 적용 |
| LightGBM의 최적 Top-K | 적용 | 적용 | 적용 |
| LSTM의 최적 Top-K | 적용 | 적용 | 적용 |

최적 특징 선택 결과가 준비된 모델부터 교차 적용할 수 있으며, 미완료 모델의 결과는 이후 실행에서 추가할 수 있도록 구성되어 있습니다.

## 🚀 설치 및 실행 방법

### 1. Python 및 라이브러리 설치

Python 3.10 이상 환경을 권장합니다.

```bash
pip install numpy pandas matplotlib scikit-learn lightgbm optuna torch shap joblib
```

실행은 `pv_shap_topk_feature_analysis` 폴더에서 진행하며, 스크립트 간에 저장된 모델·특징 중요도·예측 결과를 순차적으로 사용합니다.

### 2. 전처리 및 특징 생성

```bash
python 01_data_preprocess.py
python 02_region_split.py
python 03_feature_engineering.py
```

### 3. Full69 기준 모델 학습 및 성능 비교

```bash
python 04_exp1_train.py --models lightgbm,lstm
python 04_exp1_train_random_forest.py
python 04_compare_3models_common_test.py
```

`04_exp1_train.py`는 명령행에서 `--models lightgbm,lstm`을 명시해 두 모델만 실행합니다. Random Forest는 별도 코드에서 학습합니다. 모델 학습에는 Optuna 탐색이 포함되어 실행 시간이 길어질 수 있습니다.

### 4. 모델별 SHAP 중요도 분석

```bash
python 05_exp2_run_shap_3models.py
```

저장된 Full69 모델을 불러와 SHAP 값을 계산하며, **이 단계에서는 모델을 재학습하지 않습니다.** `shap_common.py`, `shap_lightgbm.py`, `shap_random_forest.py`, `shap_lstm.py` 파일이 필요합니다.

### 5. SHAP Top-K 탐색 및 재학습

```bash
python 06_exp3_shap_topk_retrain.py --models lightgbm,random_forest,lstm --base_train_script 04_exp1_train.py
```

기본 탐색 범위는 각 모델의 **Top-1~Top-69**이며, 모든 후보에 대해 학습 및 검증을 수행하므로 시간이 오래 걸릴 수 있습니다. 실행 전에 이전 단계의 SHAP 순위 결과를 준비해야 합니다.

### 6. 모델 간 최적 특징 교차 적용 및 성능 비교

```bash
python 07_exp4_cross_model_best_features.py
python 08_compare_full69_vs_shap_topk_3models.py
```

`07`번은 선택된 특징을 대상 모델에 적용하여 **새로 학습**합니다. `08`번은 이미 저장된 예측 결과를 읽어 **재학습 없이 성능을 비교**합니다.

## 📊 성능 평가 및 생성 결과

### 1. 평가 지표

| 지표 | 설명 |
| --- | --- |
| **MAE** | 평균 절대오차 |
| **RMSE** | 평균 제곱근 오차 |
| **NMAE** | 설비용량으로 정규화한 평균 절대오차(%) |
| **NRMSE** | 설비용량으로 정규화한 제곱근 평균제곱오차(%) |
| **MBE** | 평균 편향오차 |
| **R²** | 결정계수 |

전체 구간(`ALL`)과 주간 구간(`DAY`)을 구분해 평가하며, 주간은 **`대기권밖일사량계산값 > 0`** 조건을 사용합니다.

### 2. 주요 결과 폴더

| 폴더 | 생성 내용 |
| --- | --- |
| `pv_training_outputs/` | LightGBM·LSTM의 학습 모델, 예측 결과 및 학습 설정 |
| `pv_training_outputs_rf/` | Random Forest의 학습 모델 및 예측 결과 |
| `pv_training_outputs_final_comparison/` | Full69 3모델 공통 평가 결과 |
| `pv_shap_lgbm_rf_lstm_outputs/` | 모델별 SHAP 중요도 순위·시각화 및 중요도 비교 |
| `pv_exp3_1_modelwise_shap_topk/` | K별 Validation 성능, 최적 K, 최종 재학습 결과 |
| `pv_exp4_cross_model_best_features/` | 모델별 최적 특징 교차 적용 결과와 성능 비교 |
| `pv_exp3_1_modelwise_shap_topk/results_same_test_as_C6_Full/` | Full69와 SHAP Top-K 성능 비교 결과 |

주요 산출물의 예시는 다음과 같습니다.

```text
pv_shap_lgbm_rf_lstm_outputs/
├── random_forest/shap_feature_importance.csv
├── lightgbm/shap_feature_importance.csv
├── lstm/shap_feature_importance.csv
└── comparison/...

pv_exp3_1_modelwise_shap_topk/results/
├── BEST_K_by_validation_random_forest.csv
├── BEST_K_by_validation_lightgbm.csv
├── BEST_K_by_validation_lstm.csv
├── BEST_K_all_models_by_validation.csv
└── FINAL_TEST_all_models.csv

pv_exp4_cross_model_best_features/results/
├── FINAL_FIXED_TEST_all_cross_combinations.csv
├── MATRIX_DAY_NMAE_feature_source_x_target_model.csv
└── COMPARE_own_vs_transferred_feature_sets.csv
```

## 🛠️ 사용 기술

| 구분 | 기술 |
| --- | --- |
| 데이터 처리 | Python, pandas, NumPy |
| 모델 학습 | scikit-learn, LightGBM, PyTorch |
| 하이퍼파라미터 탐색 | Optuna |
| 특징 중요도 분석 | SHAP (`TreeExplainer`, `GradientExplainer`) |
| 결과 시각화 | Matplotlib |
| 모델 및 결과 저장 | joblib, PyTorch 모델 파일, CSV, JSON |

## ⚠️ 실행 및 결과 해석 시 참고사항

- **원본 데이터:** 공공데이터포털에서 내려받은 CSV를 별도로 준비해야 하며, 원본 데이터는 README나 코드에 포함되어 있지 않습니다.
- **선행 결과:** SHAP 분석에는 학습된 Full69 모델이 필요하고, Top-K 탐색과 교차 적용에는 이전 단계의 결과 파일이 필요합니다.
- **학습 시간:** Top-K 후보마다 학습·튜닝을 반복하므로 전체 탐색에는 상당한 시간이 소요될 수 있습니다.
- **코드 파일명:** Python 파일명이 다운로드 과정에서 변경되었다면 명령행 인자 또는 코드에 지정된 경로를 실제 파일명에 맞춰야 합니다.
- **연구 범위:** 이 프로젝트는 발전량 추정과 모델별 특징 중요도·특징 선택 효과를 분석하며, 실제 고장 라벨을 이용한 고장 분류는 수행하지 않습니다.

## 🎯 프로젝트 목적

**모델별 SHAP 특징 중요도 순위가 발전량 추정 성능에 미치는 영향을 정량적으로 확인**하고, Validation 기반 최적 Top-K 선택 및 다른 모델로의 특징 조합 전이 효과를 분석하는 데 목적이 있습니다. 이를 통해 입력 특징의 수와 구성, 예측 모델의 특성 사이의 관계를 살펴보고, 향후 태양광 발전시스템의 상태 모니터링과 고장진단에 활용할 수 있는 특징 분석 방법을 검토합니다.
