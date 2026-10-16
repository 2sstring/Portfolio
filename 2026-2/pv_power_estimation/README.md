# ☀️ pv_power_estimation

**SHAP·SFFS 기반 단계적 특징 선택을 활용한 태양광 발전량 추정 모델 비교 프로젝트**

기상·일사량 정보와 과거 발전 상태를 활용하여 **현재 운전 조건에서 기대되는 태양광 발전량**을 추정합니다. **Random Forest, LightGBM, LSTM**을 학습하고, 모델별 **SHAP 특징 중요도 분석**과 **SFFS(Sequential Forward Floating Selection)**를 이용해 효과적인 입력 특징 조합을 탐색합니다.

전체 69개 특징을 사용하는 기준 모델과 특징 선택 모델을 비교하고, 서로 다른 모델에서 선택된 특징 집합의 활용 가능성도 살펴봅니다. 추정 발전량은 향후 실제 발전량과 비교하는 **상태 모니터링 및 고장진단 연구의 기준값**으로 활용할 수 있습니다.

---

## 📌 프로젝트 개요

주요 작업은 다음과 같습니다.

- **공공 태양광 발전 데이터 전처리:** 결측치·중복값 처리 및 충청북도 자료 추출
- **입력 특징 생성:** Time, Current, Lag, Rolling, Diff의 5개 그룹, 총 **69개 특징** 구성
- **기준 모델 학습:** Random Forest, LightGBM, LSTM 모델 학습 및 성능 비교
- **SHAP 중요도 분석:** 모델별 상위 **20개 후보 특징** 선정
- **SFFS 특징 선택:** 검증 데이터의 오차를 기준으로 모델별 특징 부분집합 탐색
- **추가 비교 실험:** Full69 하이퍼파라미터 고정 조건에서 선택 특징과 교차 적용 특징의 성능 비교

| 항목 | 내용 |
| --- | --- |
| 분석 지역 | 충청북도 |
| 데이터 기간 | 2020~2024년 |
| 학습 모델 | Random Forest, LightGBM, LSTM |
| 입력 특징 | 5개 그룹, 총 69개 |
| 특징 선택 | SHAP Top20 → SFFS |
| 주요 평가 지표 | MAE, RMSE, NMAE, NRMSE, MBE, R² |
| 개발 언어 | Python |

## 📂 프로젝트 폴더 구조

아래는 주요 실행 코드와 필요한 보조 모듈을 중심으로 한 구조입니다.

```text
pv_power_estimation/
├── 01_data_preprocess.py                    # 원본 전국 데이터 정제
├── 02_region_split.py                       # 지역별 데이터 분리
├── 03_feature_engineering.py                # 충북 데이터 특징 생성
├── 04_exp1_train.py                         # LightGBM·LSTM 기준 모델 학습
├── 04_exp1_train_random_forest.py           # Random Forest 기준 모델 학습
├── 04_compare_3models_common_test.py        # 기준 모델 3종의 성능 비교 자료 생성
├── 05_exp2_run_shap_3models.py              # 모델별 SHAP 분석
├── 06_exp5_shap_sffs_modelwise.py            # SHAP Top20 기반 SFFS 특징 선택
├── 07_compare_full69_3models_same_test.py   # Full69 모델 3종 비교(재학습 없음)
├── 08_3models_sffs_fixed_hp_same_test.py    # SFFS 특징의 고정 HP 실험
├── 09_cross_model_sffs_fixed_hp_same_test.py # SFFS 특징의 모델 간 교차 적용
├── 10_3models_shap_top20_fixed_hp_same_test.py # SHAP Top20 고정 HP 실험
├── shap_common.py                           # SHAP 공통 처리 모듈
├── shap_lightgbm.py                         # LightGBM SHAP 모듈
├── shap_random_forest.py                    # Random Forest SHAP 모듈
├── shap_lstm.py                             # LSTM SHAP 모듈
└── README.md                                # 프로젝트 설명 및 실행 안내
```

데이터 파일과 학습 결과 폴더는 아래 절차에 따라 준비하거나 실행 과정에서 생성됩니다. 위 보조 모듈은 `05_exp2_run_shap_3models.py`와 같은 폴더에 있어야 합니다.

## 🔗 데이터셋 및 다운로드 안내

**한국동서발전(주)의 전국 태양광 발전량 예측값 학습데이터**를 사용합니다.

| 항목 | 내용 |
| --- | --- |
| 제공기관 | 한국동서발전(주) |
| 데이터명 | 한국동서발전(주)_전국 태양광 발전량 예측값 학습데이터 |
| 데이터 출처 | [공공데이터포털 다운로드 페이지](https://www.data.go.kr/data/15151005/fileData.do?recommendDataYn=Y) |
| 주요 변수 | 발전일자, 시도명, 설비용량, 발전량, 기온, 습도, 풍속, 적운량, 일조, 일사량 등 |
| 분석 대상 | 전처리 후 충청북도 데이터 |

### 📥 데이터 준비 방법

1. [공공데이터포털](https://www.data.go.kr/data/15151005/fileData.do?recommendDataYn=Y)에서 원본 CSV 파일을 다운로드합니다.
2. 다운로드한 CSV를 `pv_power_estimation` 프로젝트 폴더에 배치합니다.
3. `01_data_preprocess.py`에 설정된 입력 파일명과 실제 다운로드 파일명이 일치하도록 합니다.

```text
pv_power_estimation/
└── 01. 전국 태양광 발전량 예측값 학습데이터.csv
```

기본 스크립트는 위 파일명을 사용하며 원본 파일을 `cp949` 인코딩으로 읽습니다. 데이터 파일명이 다르면 `01_data_preprocess.py`의 `INPUT_PATH`를 수정합니다.

## ⚙️ 데이터 전처리 및 특징 생성

### 1. 전처리 및 지역별 데이터 분리

| 단계 | 파일 | 주요 처리 | 결과 |
| --- | --- | --- | --- |
| 1 | `01_data_preprocess.py` | 날짜·수치형 변환, 중복 제거, 짧은 결측 구간 보간 | `solar_preprocessed.csv` |
| 2 | `02_region_split.py` | 전국 데이터를 시도별로 분리 | `한국동서발전_지역별/PV_Dataset_충북.csv` 등 |
| 3 | `03_feature_engineering.py` | 시간 연속 구간 식별, 파생 특징 생성 및 결측 구간 정리 | `chungbuk_pv_features.csv` |

시간 간격이 연속되지 않는 구간에는 `segment_id`를 부여하고, 과거 시점의 특징은 해당 연속 구간 안에서 계산합니다. 특징 생성으로 결측값이 발생한 날짜는 코드에 정의된 기준에 따라 정리합니다.

### 2. 입력 특징 그룹

| 특징 그룹 | 주요 구성 | 특징 수 |
| --- | --- | ---: |
| **Time** | 월, 일, 시, 연중일, 시간의 사인·코사인 변환 | 6 |
| **Current** | 기온, 습도, 풍속, 적운량, 일조, 대기권밖일사량, 일사량 | 7 |
| **Lag** | 8개 기초 변수의 1·3·24시간 이전 값 | 24 |
| **Rolling** | 8개 기초 변수의 과거 3·24시간 이동평균 | 16 |
| **Diff** | 8개 기초 변수의 1·24시간 변화량 | 16 |
| **합계** | **C6_Full** | **69** |

Lag·Rolling·Diff의 8개 기초 변수에는 기상·일사량 관련 7개 변수와 **정규화발전량**이 포함됩니다.

모델은 발전량을 설비용량으로 나눈 값을 학습하며, 평가할 때는 발전량 단위로 복원합니다.

```text
정규화발전량 = 발전량(MWh) / 설비용량(MW)
추정 발전량(MWh) = 추정 정규화발전량 × 설비용량(MW)
```

### 3. 데이터 분할

| 구분 | 기간 | 용도 |
| --- | --- | --- |
| Train | 2020~2022년 | 모델 학습 |
| Validation | 2023년 | 하이퍼파라미터 및 특징 선택 |
| Test | 2024년 | 최종 성능 평가 |

## 🧠 예측 모델 및 특징 선택 실험

### 1. 전체 분석 흐름

```text
공공데이터포털 원본 CSV
        ↓
데이터 전처리 → 충북 자료 추출
        ↓
Time / Current / Lag / Rolling / Diff (69개)
        ↓
RF · LightGBM · LSTM 기준 모델 학습
        ↓
모델별 SHAP 중요도 분석
        ↓
모델별 SHAP Top20 후보 특징
        ↓
SFFS 기반 특징 부분집합 탐색
        ↓
선택 특징 재학습 및 모델별 성능 비교
        ↓
고정 하이퍼파라미터·교차 적용 실험
```

### 2. 주요 실험 구성

| 실험 | 목적 | 관련 코드 |
| --- | --- | --- |
| **Full69 기준 모델** | 전체 69개 특징을 사용한 RF·LightGBM·LSTM 학습 및 평가 | `04_exp1_train.py`, `04_exp1_train_random_forest.py`, `07_compare_full69_3models_same_test.py` |
| **SHAP 중요도** | 학습된 기준 모델별 중요 특징 분석 | `05_exp2_run_shap_3models.py` |
| **SHAP + SFFS** | SHAP Top20을 후보로 SFFS 특징 부분집합 선택 | `06_exp5_shap_sffs_modelwise.py` |
| **SFFS 고정 HP** | Full69 최적 하이퍼파라미터를 고정한 SFFS 특징 재학습 | `08_3models_sffs_fixed_hp_same_test.py` |
| **Cross-SFFS 고정 HP** | 한 모델에서 선택한 특징을 다른 모델에도 적용 | `09_cross_model_sffs_fixed_hp_same_test.py` |
| **SHAP Top20 고정 HP** | Top20 특징의 독립적인 성능 비교 | `10_3models_shap_top20_fixed_hp_same_test.py` |

**SFFS**는 특징을 순차적으로 추가하고 조건에 따라 제거하는 탐색 방법입니다. 이 프로젝트는 검증 데이터의 **주간 NMAE**를 기준으로 특징 부분집합을 선택합니다.

`06`번은 SFFS 탐색 중 Full69의 하이퍼파라미터를 활용하고, 최종 선택 특징에 대해서는 재튜닝을 수행합니다. 반면 `08`~`10`번은 Full69 하이퍼파라미터를 **다시 최적화하지 않고**, 새로운 특징 집합에 맞춰 **모델 가중치만 새로 학습**합니다.

> `04_exp1_train.py`에는 PatchTST 구현도 포함되어 있지만, 이 프로젝트에서 정리하는 주요 비교 모델은 **Random Forest, LightGBM, LSTM**입니다.

## 🚀 설치 및 실행 방법

### 1. 실행 환경 및 라이브러리 설치

Python 3 환경에서 다음 라이브러리를 설치합니다.

```bash
pip install numpy pandas matplotlib scikit-learn lightgbm optuna torch shap joblib
```

프로젝트 폴더에서 다음 순서로 실행합니다. 모델 학습 및 SFFS 탐색은 데이터 크기와 실행 환경에 따라 시간이 오래 걸릴 수 있습니다.

### 2. 데이터 전처리 및 특징 생성

```bash
python 01_data_preprocess.py
python 02_region_split.py
python 03_feature_engineering.py
```

### 3. Full69 기준 모델 학습 및 비교

```bash
python 04_exp1_train.py --models lightgbm,lstm
python 04_exp1_train_random_forest.py
python 04_compare_3models_common_test.py
python 07_compare_full69_3models_same_test.py
```

`07`번은 앞선 모델의 저장된 예측 결과를 읽어 비교하므로 **재학습을 수행하지 않습니다.**

### 4. SHAP 중요도 및 SFFS 특징 선택

```bash
python 05_exp2_run_shap_3models.py
python 06_exp5_shap_sffs_modelwise.py --base_train_script 04_exp1_train.py
```

### 5. 고정 하이퍼파라미터 추가 실험

```bash
python 08_3models_sffs_fixed_hp_same_test.py
python 09_cross_model_sffs_fixed_hp_same_test.py
python 10_3models_shap_top20_fixed_hp_same_test.py
```

각 코드는 이전 단계에서 생성한 학습 모델, SHAP 결과 및 특징 선택 목록을 사용합니다.

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

평가는 전체 시간대(`ALL`)와 주간 시간대(`DAY`)를 나누어 수행하며, 주간은 코드에서 `대기권밖일사량계산값 > 0` 조건으로 구분합니다.

### 2. 주요 결과 폴더

| 폴더 | 생성 내용 |
| --- | --- |
| `pv_training_outputs/` | LightGBM·LSTM 학습 모델 및 예측값 |
| `pv_training_outputs_rf/` | Random Forest 학습 모델 및 예측값 |
| `pv_training_outputs_final_comparison/` | 기준 모델 성능 비교 자료 |
| `pv_shap_lgbm_rf_lstm_outputs/` | 모델별 SHAP 중요도 및 순위 |
| `pv_exp5_shap_sffs/` | SFFS 탐색 기록, 선택 특징 및 평가 결과 |
| `pv_final_full69_3models_same_test/` | Full69 3개 모델 성능 비교 |
| `pv_exp7_3models_sffs_fixed_hp/` | SFFS 특징의 고정 하이퍼파라미터 결과 |
| `pv_exp8_cross_model_sffs_fixed_hp/` | 모델 간 SFFS 특징 교차 적용 결과 |
| `pv_exp9_3models_shap_top20_fixed_hp/` | SHAP Top20 고정 하이퍼파라미터 결과 |

위 폴더는 관련 코드를 실행하면 생성되며, 재현에 필요한 선행 결과 파일이 준비되어 있어야 합니다.

## 🛠️ 사용 기술

| 구분 | 기술 |
| --- | --- |
| 데이터 처리 | Python, pandas, NumPy |
| 머신러닝 | scikit-learn, LightGBM |
| 딥러닝 | PyTorch |
| 하이퍼파라미터 탐색 | Optuna |
| 특징 중요도 | SHAP |
| 특징 선택 | Sequential Forward Floating Selection (SFFS) |
| 결과 분석 | CSV, Matplotlib |

## ⚠️ 실행 및 결과 해석 시 주의사항

- **원본 데이터:** GitHub에서 코드를 내려받은 뒤 공공데이터포털의 CSV를 별도로 준비해야 합니다.
- **실행 순서:** SHAP 및 특징 선택 단계는 사전에 저장된 Full69 학습 모델에 의존합니다.
- **고정 하이퍼파라미터 실험:** 하이퍼파라미터 탐색을 생략하지만 새 특징 조합으로 **모델을 재학습**합니다.
- **연구 범위:** 실제 발전소의 고장 라벨을 활용한 고장 분류 모델은 포함하지 않습니다. 결과는 발전량 추정과 입력 특징 선택 효과를 분석하기 위한 것입니다.

## 🎯 프로젝트 목적

다양한 특징을 단순히 모두 사용하는 대신, **모델별로 중요한 특징을 체계적으로 선택**하여 입력 차원과 발전량 추정 성능의 관계를 분석하는 데 목적이 있습니다. 이를 통해 태양광 발전시스템의 기대 발전량 산정과 후속 상태 모니터링 연구에 활용할 수 있는 모델 구성 방법을 제시합니다.
