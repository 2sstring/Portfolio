# 💡 uv_led_optical_output_compensation

**머신러닝 기반 UV LED 광출력 예측 및 국소 탐색을 이용한 ADC 보상량 산출**

## 📌 프로젝트 개요

이 프로젝트는 **UV LED의 온도 변화에 따른 광출력 변화를 예측**하고, 기준 온도에서의 광출력을 유지하기 위한 **보상 ADC Level**을 계산하는 실험 코드입니다.

온도와 ADC Level을 입력으로 하는 회귀 모델을 학습한 뒤, 학습된 모델을 이용해 현재 온도에서의 예상 광출력을 산정합니다. 이어서 **25℃에서 측정한 광출력**을 목표값으로 설정하고, 예상 보상 ADC 주변을 탐색해 **목표 광출력 오차와 ADC 증가량을 함께 고려한 보상값**을 선택합니다.

| 구분 | 내용 |
|---|---|
| 연구 대상 | UV LED의 온도 변화와 광출력 특성 |
| 입력 변수 | `Temperature`, `ADC_Level` |
| 예측 대상 | `Optical_Power` |
| 예측 방법 | 5종 회귀 모델 비교 및 하이퍼파라미터 탐색 |
| 검증 방법 | ADC Level 단위 Leave-One-ADC-Out 검증 |
| 보상 방법 | 25℃ 기준 광출력 + 국소 기울기 기반 추정 + 목적함수 국소 탐색 |
| 결과 | 광출력 예측 모델, 성능 평가표, 보상 ADC, 시각화 그래프 |

> **데이터 공개 안내:** 학습·보상에 사용하는 엑셀 파일은 회사 내부 측정 자료로서 **공개할 수 없습니다.** 이 저장소에는 원본 측정값을 포함하지 않습니다. 예측 결과 CSV, 그래프, 학습 모델에도 원본 측정 특성이 반영될 수 있으므로 외부 공개 전 별도 검토가 필요합니다.

---

## 📂 프로젝트 폴더 구조

```text
uv_led_optical_output_compensation/
├── README.md
├── 01_train_predict_model.py        # 광출력 회귀 모델 학습·검증·테스트
├── 02_local_search_objective.py     # 예측 모델 기반 ADC 보상량 국소 탐색
│
├── led_dataset.xlsx                 # 내부 측정 데이터: 비공개, 별도 준비
├── adc_dataset.xlsx                 # 25℃ ADC-광출력 관계: 비공개, 별도 준비
│
├── outputs_uv_led_ai/              # 첫 번째 코드 실행 후 생성
│   ├── best_model.pkl
│   ├── validation_best_model.pkl
│   ├── best_model_info.json
│   ├── final_model_selection_summary.json
│   ├── data_split_info.json
│   ├── led_dataset_long_format.csv
│   ├── temperature_drop_rate.csv
│   ├── cv_hyperparameter_results_with_train.csv
│   ├── cv_summary_by_model.csv
│   ├── cv_best_params_by_model.csv
│   ├── test_metrics_best_each_model.csv
│   ├── test_predictions_best_each_model.csv
│   ├── test_metrics.csv
│   ├── test_predictions.csv
│   └── graphs/                     # 광출력 특성 및 예측 결과 PNG
│
└── outputs_uv_led_05_local_search_objective/  # 두 번째 코드 실행 후 생성
    ├── led_dataset_long_format.csv
    ├── adc_dataset_clean.csv
    ├── target_power_25c_by_adc.csv
    ├── local_compensation_results.csv
    ├── local_adc_optimization_candidates.csv
    ├── local_compensation_summary.csv
    ├── local_compensation_summary_by_adc.csv
    ├── local_optimization_info.json
    └── graphs/                     # 보상 전후 오차·ADC·출력 그래프 PNG
```

`outputs_uv_led_ai/`와 `outputs_uv_led_05_local_search_objective/`는 **프로그램 실행 후 생성되는 결과 폴더**입니다. 위 구조는 결과 파일까지 포함한 예시이며, 실행 전부터 모두 존재하는 파일을 뜻하지 않습니다.

### 🗂️ 코드별 역할

| 파일 | 주요 기능 |
|---|---|
| `01_train_predict_model.py` | 엑셀 데이터 변환, 모델 후보 비교, ADC 단위 교차검증, 테스트, 최종 회귀 모델 저장 |
| `02_local_search_objective.py` | 학습 모델 및 25℃ 기준 데이터 로드, 보상 ADC 추정, 국소 목적함수 탐색, 결과 저장 |

---

## 🗂️ 입력 데이터 및 준비 방법

이 프로젝트는 공개 데이터셋을 사용하는 것이 아니라 **실측 UV LED 광출력 데이터**를 활용합니다.

| 파일 | 용도 | 코드에서 기대하는 형태 |
|---|---|---|
| `led_dataset.xlsx` | 온도·ADC Level별 광출력 학습 및 25℃ 목표 광출력 산정 | 첫 번째 시트에 온도 열과 `120-LEVEL`, `ADC 120`처럼 숫자를 포함한 ADC 열들이 있는 가로형 데이터 |
| `adc_dataset.xlsx` | 25℃에서 ADC 변화에 따른 광출력 관계 및 국소 기울기 계산 | 첫 번째 시트에 ADC Level 열과 광출력 열이 있는 데이터 |

### 데이터 변환

첫 번째 코드는 `led_dataset.xlsx`의 Wide Format을 다음과 같은 Long Format으로 변환합니다.

```text
[변환 전: 형식 예시]
온도 | 120-LEVEL | 150-LEVEL | 180-LEVEL | ...

[변환 후]
Temperature | ADC_Level | Optical_Power
```

온도·ADC·광출력 값은 수치형으로 변환되며, 변환할 수 없는 값 또는 결측값은 제외됩니다. 광출력의 **측정 단위는 제공된 원본 파일의 정의에 따르며**, 코드에서 별도의 단위를 확정하지 않습니다.

**데이터 준비 순서**

1. 내부에서 사용 권한이 있는 `led_dataset.xlsx` 및 `adc_dataset.xlsx`를 준비합니다.
2. 두 파일을 Python 스크립트와 같은 폴더에 배치합니다.
3. `led_dataset.xlsx`에 테스트용 ADC Level **170, 230** 및 목표값 계산에 필요한 **25℃ 행**이 포함되어 있는지 확인합니다.
4. 파일명·열 이름이 다를 경우 코드 상단의 경로와 데이터 로드 부분을 실제 형식에 맞게 조정합니다.

---

## ⚙️ 전체 처리 과정

```text
UV LED 내부 측정 데이터
        │
        ▼
[1] 온도·ADC·광출력 데이터 변환
        │
        ▼
[2] ADC Level별 Train/Validation · Test 분리
        │
        ▼
[3] 5종 회귀 모델 및 하이퍼파라미터 비교
        │
        ▼
[4] 미학습 ADC Level 예측 평가 · 모델 저장
        │
        ▼
[5] 25℃ 기준 목표 광출력 설정
        │
        ▼
[6] 온도 변화에 따른 광출력 부족량 예측
        │
        ▼
[7] 예상 보상 ADC 산출 · 주변 국소 탐색
        │
        ▼
[8] 최종 ADC 보상값 및 보상 전후 결과 저장
```

## 🤖 1단계: 광출력 예측 모델 학습

### 비교 모델

| 모델 | 주요 설정 |
|---|---|
| Linear Regression | 기본 선형 회귀 |
| Polynomial Regression | 1~3차 다항 특징 |
| Support Vector Regression | RBF 커널, `C`·`epsilon`·`gamma` 조합 비교 |
| Gaussian Process Regression | RBF + WhiteKernel, 길이 척도·잡음 수준 조합 비교 |
| Random Forest Regression | 트리 수·최대 깊이·최소 리프 샘플 수 비교 |

코드상 총 **88개 회귀 후보 조합**을 생성합니다. 이는 모델 5종의 하이퍼파라미터 조합 개수를 합한 값이며, 학습 완료 또는 특정 성능을 의미하지 않습니다.

### ADC Level 기준 데이터 분리 및 검증

- **Test ADC:** `170`, `230`
- **Train/Validation:** 나머지 ADC Level
- **검증 방식:** Train/Validation에 속하는 ADC Level 하나의 모든 온도 데이터를 검증용으로 남기고, 나머지 ADC Level로 학습하는 *Leave-One-ADC-Out* 방식
- **하이퍼파라미터 선택 기준:** 검증 데이터에서 계산한 평균 `MAPE(%)` 최소화

이러한 분리는 동일 ADC Level의 측정값이 학습·검증 양쪽에 혼재하는 상황을 피하고, **학습에 사용되지 않은 ADC Level에서의 예측 가능성**을 확인하기 위한 것입니다.

### 성능 지표

| 지표 | 설명 |
|---|---|
| MAE | 실제값과 예측값의 평균 절대오차 |
| RMSE | 큰 오차에 더 민감한 평균제곱근오차 |
| MAPE (%) | 광출력 실측값에 대한 평균 절대 백분율 오차 |
| R² | 회귀 예측의 결정계수 |

MAPE 계산에서는 실제 광출력이 0에 가까울 때 분모에 `1e-8`을 사용합니다. 따라서 **작은 광출력 영역의 MAPE가 매우 커질 수 있으므로** MAE·RMSE도 함께 확인하는 것이 좋습니다.

### 최종 모델 선택 방식

`01_train_predict_model.py`의 `FINAL_MODEL_SELECTION_MODE`에 따라 저장할 모델을 지정할 수 있습니다.

| 설정값 | 의미 |
|---|---|
| `test_best_each_model` (**현재 기본값**) | 모델별 검증 최상위 하이퍼파라미터 중 Test MAPE가 가장 낮은 모델 선택 |
| `validation_best` | 전체 후보 중 Validation 평균 MAPE가 가장 낮은 모델 선택 |
| `manual_model` | 지정한 모델 계열의 Validation 최상위 조합 선택 |

학습이 완료되면 보상 단계에서 사용하는 **`outputs_uv_led_ai/best_model.pkl`**과 검증 최우수 모델 **`validation_best_model.pkl`**이 각각 저장됩니다.

> **평가 시 주의:** 기본 설정인 `test_best_each_model`은 Test ADC Level을 하이퍼파라미터 탐색에는 사용하지 않지만, **최종 모델 계열을 선택할 때 Test 결과를 사용합니다.** 따라서 선택된 모델의 Test 성능을 완전히 독립적인 최종 일반화 성능으로 해석해서는 안 됩니다. 독립 평가가 중요한 경우 `validation_best`로 모델을 먼저 확정하고, 별도의 미사용 ADC 데이터로 평가해야 합니다.

---

## 🎯 2단계: 국소 탐색 기반 ADC 광출력 보상

`02_local_search_objective.py`는 첫 번째 단계에서 저장한 모델과 **25℃ ADC-광출력 관계**를 사용하여 온도 변화에 대응하는 보상 ADC를 계산합니다.

### 보상량 산출 절차

1. 기준 ADC Level의 **25℃ 측정 광출력**을 목표 광출력으로 설정합니다.
2. 현재 온도와 기준 ADC를 예측 모델에 입력하여 보상 전 광출력을 추정합니다.
3. 목표 광출력과 예측 광출력의 차이에서 부족 광출력을 계산합니다.
4. `adc_dataset.xlsx`의 25℃ 데이터를 사용해 기준 ADC 주변의 국소 기울기를 추정합니다.
5. 부족 광출력을 ADC 증가량으로 환산해 **예상 보상 ADC**를 계산합니다.
6. 예상 ADC 주변의 정수 후보들을 평가하여 목적함수가 최소인 ADC를 선택합니다.

### 목적함수

\[
\min_{A\in\mathcal A_{\mathrm{local}}}
\left(\left|P_{\mathrm{target}}-\hat P(T,A)\right|+\lambda(A-A_{\mathrm{base}})\right)
\]

- \(P_{\mathrm{target}}\): 기준 ADC에서의 25℃ 측정 광출력
- \(\hat P(T,A)\): 현재 온도 \(T\) 및 후보 ADC \(A\)에 대해 모델이 예측한 광출력
- \(A_{\mathrm{base}}\): 보상 전 기준 ADC Level
- \(\lambda\): ADC 증가에 대한 벌점 계수
- \(\mathcal A_{\mathrm{local}}\): 예상 보상 ADC 주변에서 구성한 탐색 후보 집합

기본 설정은 다음과 같습니다.

| 설정 | 값 | 의미 |
|---|---:|---|
| `ADC_MIN`, `ADC_MAX` | 0, 255 | ADC 허용 범위 |
| `LOCAL_WINDOW` | 5 | 예상 보상 ADC 주변 ±5 탐색 |
| `SLOPE_WINDOW` | 10 | 기울기 추정 시 기준 ADC 주변 ±10 활용 |
| `LAMBDA_ADC` | 0.01 | ADC 증가 벌점 계수 |
| `TOLERANCE` | 0.10 | 목표 광출력과의 절대오차 허용 기준(결과 판별용) |
| `ALLOW_ADC_DECREASE` | `False` | 기준 ADC보다 작은 ADC로는 보상하지 않음 |

**`TOLERANCE`는 목적함수의 제약조건으로 강제하지 않고**, 계산된 후보가 허용오차 안에 들어왔는지 표시하는 용도로 사용합니다. 탐색 결과가 허용오차를 충족하지 못할 수도 있습니다.

### 보상 결과 평가

보상 전후의 예측 광출력을 기준으로 평균·최대 상대오차, 절대오차, 출력 범위 및 표준편차를 비교합니다. 또한 보상 ADC 증가량, 허용오차 충족률, 조건별 탐색 후보 개수를 저장합니다.

이 결과는 **예측 모델을 활용한 보상 시뮬레이션**입니다. 보상 후 ADC에서 UV LED 광출력을 다시 측정한 실제 폐루프 제어 실험 결과를 의미하지 않습니다.

---

## 🛠️ 사용 기술

| 기술 | 용도 |
|---|---|
| Python | 학습 및 보상 시뮬레이션 |
| pandas, NumPy | 엑셀 측정값 변환 및 데이터 계산 |
| scikit-learn | 회귀 학습, 검증, 성능 평가 |
| Matplotlib | 온도·광출력·보상 결과 시각화 |
| openpyxl | `.xlsx` 파일 읽기 |
| joblib | 학습된 회귀 모델 저장·불러오기 |

## 🚀 설치 및 실행 방법

프로젝트 폴더에서 아래 명령어를 실행합니다.

### 1. 라이브러리 설치

```bash
pip install pandas numpy scikit-learn matplotlib openpyxl joblib
```

### 2. 광출력 예측 모델 학습 및 저장

```bash
python 01_train_predict_model.py
```

정상 실행 후 `outputs_uv_led_ai/best_model.pkl`이 생성되어야 합니다. 모델 비교 대상과 교차검증 횟수에 따라 실행 시간이 길어질 수 있습니다.

### 3. ADC 보상량 계산

```bash
python 02_local_search_objective.py
```

두 번째 코드는 `outputs_uv_led_ai/best_model.pkl`, `led_dataset.xlsx`, `adc_dataset.xlsx`가 있어야 실행됩니다. 코드 내부에 이전 실험명(`05_local_search_objective`)이 남아 있지만 **실제 실행할 파일명은 `02_local_search_objective.py`**입니다.

---

## 📊 주요 결과 파일

### 광출력 예측 (`outputs_uv_led_ai/`)

| 결과 파일 | 내용 |
|---|---|
| `best_model.pkl` | 선택 모드에 따라 최종 저장된 예측 모델 |
| `validation_best_model.pkl` | Validation 최우수 모델 |
| `best_model_info.json` | 최종 모델·입력 변수·선택 기준 정보 |
| `data_split_info.json` | Train/Validation과 Test ADC 분리 정보 |
| `cv_hyperparameter_results_with_train.csv` | 후보별 ADC 단위 학습·검증 상세 지표 |
| `cv_summary_by_model.csv` | 후보별 검증 평균 및 표준편차 |
| `cv_best_params_by_model.csv` | 모델 계열별 최상위 파라미터 |
| `test_metrics_best_each_model.csv` | 모델 계열별 Test 성능 비교 |
| `test_metrics.csv` | 최종 저장 모델의 Test 성능 |
| `test_predictions.csv` | 최종 모델의 실제·예측 광출력 및 오차 |
| `graphs/` | 광출력 특성, Validation MAPE, Test 예측 비교 그래프 |

### ADC 보상 (`outputs_uv_led_05_local_search_objective/`)

| 결과 파일 | 내용 |
|---|---|
| `target_power_25c_by_adc.csv` | 기준 ADC별 25℃ 목표 광출력 |
| `local_adc_optimization_candidates.csv` | 국소 탐색의 후보 ADC 및 목적함수 값 |
| `local_compensation_results.csv` | 조건별 보상 전후 광출력·오차·최적 ADC |
| `local_compensation_summary.csv` | 전체 조건의 보상 효과 요약 |
| `local_compensation_summary_by_adc.csv` | 기준 ADC별 보상 효과 비교 |
| `local_optimization_info.json` | 탐색 폭, 벌점, 모델 정보 등 실험 설정 |
| `graphs/` | 보상 전후 오차, 최적 ADC 변화 및 광출력 안정성 그래프 |

원본 측정 데이터가 제공되지 않아 **실제 모델별 MAE·RMSE·MAPE·R² 수치나 보상 개선율은 이 README에 기재하지 않았습니다.** 해당 값은 코드를 실제 데이터로 실행한 결과 파일에서 확인해야 합니다.

---

## ⚠️ 참고 및 주의사항

- **비공개 데이터:** 회사 내부 `led_dataset.xlsx`, `adc_dataset.xlsx`는 GitHub에 업로드하지 않습니다. 측정값을 포함하는 파생 CSV·그래프·모델 파일도 공개 가능 여부를 검토해야 합니다.
- **모델 평가 독립성:** 기본 모델 선택 방식은 Test 데이터를 모델 계열 선택에 활용합니다. 독립 테스트가 필요한 보고서·논문에서는 별도의 미사용 데이터를 확보해야 합니다.
- **측정 조건 일치:** 두 엑셀 파일의 ADC 정의, 25℃ 측정 조건 및 광출력 단위가 일치해야 합니다.
- **범위 밖 예측:** 측정되지 않은 온도·ADC 구간에서는 회귀 모델의 외삽 결과가 부정확할 수 있습니다.
- **보상 후보의 범위:** 국소 탐색은 전체 ADC를 완전탐색하지 않으므로 전역 최적해를 보장하지 않습니다. 기본 설정에서는 기준 ADC 이하로 감소시키지 않습니다.
- **광출력 제어 검증:** 계산된 보상 ADC는 모델 예측에 기반한 권장값이며, 실제 장비 적용 전 추가 계측 및 동작 범위 검증이 필요합니다.
- **실행 환경:** 두 스크립트는 Python 소스와 엑셀 파일이 동일한 폴더에 있다고 가정합니다. `joblib` 모델 파일은 신뢰할 수 있는 출처에서 생성된 것만 불러와야 합니다.
- **재현성:** 원본 측정 데이터가 비공개이므로 공개 저장소만으로 동일한 학습 결과를 완전히 재현할 수는 없습니다.

---

## 🎯 프로젝트 목적

온도 변화에 따른 UV LED 광출력을 회귀 모델로 추정하고, **목표 광출력 오차와 ADC 보상 부담의 균형**을 고려하여 정출력 유지에 활용할 수 있는 ADC 보상값을 산출하는 것이 목적입니다. 본 저장소는 그중 **데이터 기반 모델 학습 및 보상 알고리즘 검토 단계**를 다룹니다.
