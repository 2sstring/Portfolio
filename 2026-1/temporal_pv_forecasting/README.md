# Temporal PV Forecasting

시간적 특징 학습 기반의 태양광 발전량 예측 모델 개발을 위한 Python 코드 저장소입니다.

본 저장소는 다음 연구 주제와 관련된 실험 코드를 포함합니다.

> 시간적 특징 학습 기반의 고정밀 태양광 발전량 예측 모델 개발

본 연구의 목적은 Lag, Rolling, Diff 등 시간적 특징을 반영한 입력 구성이 태양광 발전량 예측 성능에 미치는 영향을 비교·분석하는 것입니다.

---

## 1. 데이터셋

본 프로젝트에서는 공공데이터포털에서 제공하는 다음 데이터를 사용합니다.

* **데이터셋명:** 한국동서발전(주)_전국 태양광 발전량 예측값 학습데이터
* **제공처:** 공공데이터포털
* **다운로드 URL:** https://www.data.go.kr/data/15151005/fileData.do?recommendDataYn=Y

본 저장소에는 원본 데이터 파일을 포함하지 않습니다.
데이터는 공공데이터포털에서 직접 다운로드한 후 프로젝트 폴더에 넣어 사용해야 합니다.

전처리 코드에서 사용하는 기본 입력 파일명은 다음과 같습니다.

```text
01. 전국 태양광 발전량 예측값 학습데이터.csv
```

다운로드한 파일명이 다를 경우, 파일명을 위와 같이 변경하거나 `01_data_preprocess.py` 파일의 `INPUT_PATH` 값을 수정하면 됩니다.

---

## 2. 프로젝트 개요

전체 실험 흐름은 다음과 같습니다.

1. 전국 태양광 발전량 원본 데이터 전처리
2. 전처리된 데이터를 지역별로 분할
3. 시간적 특징 생성
4. 입력 특징 조합별 예측 모델 학습 및 성능 평가
5. SHAP 기반 주요 특징 분석
6. SHAP 중요도와 상관관계 필터링을 이용한 Top-K 특징 재학습

본 연구에서 사용하는 예측 대상은 다음과 같습니다.

```text
정규화발전량 = 발전량(MWh) / 설비용량(MW)
```

모델이 예측한 정규화 발전량은 다음 식을 이용하여 실제 발전량 단위인 MWh로 복원합니다.

```text
예측 발전량(MWh) = 예측 정규화발전량 × 설비용량(MW)
```

---

## 3. 저장소 구조

```text
temporal-pv-forecasting/
├─ 01_data_preprocess.py
├─ 02_region_split.py
├─ 03_feature_engineering.py
├─ 04_exp1_train.py
├─ 05_exp2_shap_analysis.py
├─ 06_exp3_shap_corr_retrain.py
├─ pv_exp_common.py
├─ README.md
└─ requirements.txt
```

### 파일 설명

| 파일명                            | 설명                                                                 |
| ------------------------------ | ------------------------------------------------------------------ |
| `01_data_preprocess.py`        | 원본 전국 데이터를 불러와 결측 처리, 유효성 검증, 중복 제거 등을 수행하고 전처리 데이터를 저장합니다.        |
| `02_region_split.py`           | 전처리된 전국 데이터를 지역별 CSV 파일로 분할합니다.                                    |
| `03_feature_engineering.py`    | 정규화 발전량과 Lag, Rolling, Diff 등의 시간적 특징을 생성합니다.                      |
| `04_exp1_train.py`             | C1~C6 입력 특징 조합에 대해 LightGBM, LSTM, PatchTST 모델을 학습하고 성능을 비교합니다.    |
| `05_exp2_shap_analysis.py`     | 선정된 LightGBM 최적 파라미터를 사용하여 C6_Full 모델을 학습하고 SHAP 기반 특징 중요도를 분석합니다. |
| `06_exp3_shap_corr_retrain.py` | SHAP 중요도와 상관관계 필터링을 적용한 Top-K 특징 조합으로 모델을 재학습합니다.                  |
| `pv_exp_common.py`             | 여러 실험 코드에서 공통으로 사용하는 함수들을 포함합니다.                                   |

---

## 4. 특징 그룹

본 연구에서는 입력 특징을 다음과 같은 그룹으로 구분합니다.

| 그룹         | 설명                                   |
| ---------- | ------------------------------------ |
| G1_Time    | 월, 일, 시간, 연중 일수, 시간 주기 특징 등 기본 시간 특징 |
| G2_Current | 현재 시점의 기상 및 일사 관련 특징                 |
| G3_Lag     | 과거 시점의 값으로 구성한 지연 특징                 |
| G4_Rolling | 과거 일정 구간의 이동평균 특징                    |
| G5_Diff    | 과거 값의 변화량을 나타내는 차분 특징                |

주요 입력 특징 조합은 다음과 같습니다.

| Case | 입력 특징 구성                              |
| ---- | ------------------------------------- |
| C1   | Time + Current                        |
| C2   | Time + Current + Lag                  |
| C3   | Time + Current + Rolling              |
| C4   | Time + Current + Diff                 |
| C5   | Time + Current + Lag + Rolling        |
| C6   | Time + Current + Lag + Rolling + Diff |

---

## 5. 실험 설계

데이터는 연도 기준으로 다음과 같이 분할합니다.

| 구분         | 기간            |
| ---------- | ------------- |
| Train      | 2020년 ~ 2022년 |
| Validation | 2023년         |
| Test       | 2024년         |

주요 평가 지표는 주간 구간 기준 NMAE입니다.

```text
DAY_NMAE_pct
```

이 외에도 MAE, RMSE, NRMSE, MBE, R² 등을 함께 계산합니다.

---

## 6. 설치 방법

Python 3.10 이상 사용을 권장합니다.

필요한 주요 패키지는 다음과 같습니다.

```bash
pip install pandas numpy scikit-learn lightgbm optuna shap matplotlib joblib torch
```

`requirements.txt` 파일을 사용하는 경우 다음 명령어로 설치할 수 있습니다.

```bash
pip install -r requirements.txt
```

---

## 7. 실행 방법

### 1단계. 원본 데이터 전처리

```bash
python 01_data_preprocess.py
```

출력 파일 예시는 다음과 같습니다.

```text
solar_preprocessed.csv
```

### 2단계. 지역별 데이터 분할

```bash
python 02_region_split.py
```

출력 예시는 다음과 같습니다.

```text
한국동서발전_지역별/PV_Dataset_충북.csv
```

### 3단계. 시간적 특징 생성

`03_feature_engineering.py`는 기본적으로 다음 입력 파일을 사용합니다.

```text
chungbuk_pv_dataset.csv
```

지역별 분할 결과 중 사용할 지역 데이터를 위 파일명으로 변경한 뒤 실행합니다.

```bash
python 03_feature_engineering.py
```

출력 파일은 다음과 같습니다.

```text
chungbuk_pv_features.csv
```

### 4단계. 실험 1: 입력 특징 조합별 모델 학습

```bash
python 04_exp1_train.py --csv chungbuk_pv_features.csv
```

실험 1에서는 C1~C6 입력 특징 조합에 대해 LightGBM, LSTM, PatchTST 모델의 예측 성능을 비교합니다.

### 5단계. 실험 2: SHAP 기반 특징 중요도 분석

```bash
python 05_exp2_shap_analysis.py --csv chungbuk_pv_features.csv
```

실험 2에서는 C6_Full 입력 특징을 사용한 LightGBM 모델을 대상으로 SHAP 분석을 수행합니다.

### 6단계. 실험 3: SHAP + 상관관계 필터링 기반 Top-K 재학습

```bash
python 06_exp3_shap_corr_retrain.py
```

기본 설정에서는 LightGBM 모델만 Top-K 특징 조합별로 재학습합니다.

---

## 8. 참고 사항

* 원본 데이터 파일은 저장소에 포함하지 않습니다.
* 데이터는 공공데이터포털에서 직접 다운로드해야 합니다.
* 일부 코드에는 기본 파일명이 지정되어 있으므로, 필요에 따라 코드 상단의 경로 변수를 수정해야 합니다.
* 본 실험의 기본 지역 예시는 충북 데이터를 기준으로 작성되어 있습니다.
* 학습된 모델, 예측 결과, 중간 산출물 등 용량이 큰 파일은 `.gitignore`를 이용해 GitHub 업로드 대상에서 제외하는 것을 권장합니다.

---

## 9. 데이터 출처 표기

본 프로젝트에서 사용한 데이터는 다음과 같습니다.

```text
한국동서발전(주)_전국 태양광 발전량 예측값 학습데이터,
공공데이터포털.
https://www.data.go.kr/data/15151005/fileData.do?recommendDataYn=Y
```

---

## 10. 라이선스

본 저장소의 코드는 학술 연구 및 실험 목적으로 작성되었습니다.
데이터 사용 시에는 공공데이터포털의 데이터 이용 조건과 라이선스를 확인한 후 사용해야 합니다.
::: 
