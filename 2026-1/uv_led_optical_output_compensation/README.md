# UV LED 정출력 유지를 위한 온도-광출력 데이터 분석 및 보상량 산출

## 프로젝트 개요

본 프로젝트는 UV LED의 온도 변화에 따른 광출력 변화를 분석하고, 정출력 유지를 위한 ADC 보상량을 산출하는 프로젝트입니다.

UV LED는 온도 변화에 따라 광출력이 변할 수 있습니다. 따라서 일정한 광출력을 유지하기 위해 온도와 ADC Level에 따른 광출력 데이터를 분석하고 AI 회귀 모델을 학습합니다.

학습된 예측 모델을 기반으로 현재 온도에서 목표 광출력을 유지하기 위한 보상 ADC Level을 계산합니다.

## 데이터셋 정보

본 저장소에는 원본 측정 데이터셋을 포함하지 않습니다.

본 프로젝트의 데이터셋은 회사 내부 측정 자료를 사용하였으며, 공개가 불가능합니다.

* 데이터 종류: UV LED 온도, ADC Level, 광출력 측정 데이터
* 데이터 출처: 회사 내부 측정 자료
* 공개 여부: 비공개
* 사용 파일:

  * `led_dataset.xlsx`
  * `adc_dataset.xlsx`

데이터 파일을 아래와 같이 프로젝트 폴더에 배치합니다.

```text
06-uv-led-optical-output-compensation/
├── README.md
├── 01_train_predict_model.py
├── 02_local_search_objective.py
├── led_dataset.xlsx       # GitHub 업로드 제외
└── adc_dataset.xlsx       # GitHub 업로드 제외
```

## 주요 내용

### 1. 데이터 변환

* wide-format 형태의 LED 측정 데이터를 long-format으로 변환합니다.
* 입력 변수는 `Temperature`, `ADC_Level`로 구성합니다.
* 출력 변수는 `Optical_Power`로 구성합니다.

### 2. 광출력 예측 모델 학습

다음 회귀 모델을 비교합니다.

* Linear Regression
* Polynomial Regression
* Support Vector Regression
* Gaussian Process Regression
* Random Forest Regression

각 모델에 대해 하이퍼파라미터 후보를 설정하고, Validation 성능을 기준으로 모델을 비교합니다.

### 3. 검증 방법

* Test ADC Level을 학습에서 완전히 제외합니다.
* Train/Validation 데이터에서 ADC Level 단위 Leave-One-Out 검증을 수행합니다.
* 미측정 ADC Level에 대한 예측 성능을 확인합니다.

### 4. 최종 Test 평가

* 제외한 Test ADC Level에 대해 최종 예측 성능을 평가합니다.
* MAE, RMSE, MAPE, R2 지표를 계산합니다.
* 실제 광출력과 예측 광출력을 비교하는 그래프를 저장합니다.

### 5. ADC 보상량 산출

학습된 광출력 예측 모델을 이용하여 정출력 유지를 위한 ADC 보상량을 산출합니다.

보상량 산출 과정은 다음과 같습니다.

1. 각 기준 ADC Level의 25℃ 광출력을 목표 광출력으로 설정합니다.
2. 현재 온도와 기준 ADC에서의 광출력을 예측합니다.
3. 목표 광출력 대비 부족량을 계산합니다.
4. 25℃ ADC-광출력 관계를 이용해 예상 보상 ADC를 계산합니다.
5. 예상 ADC 주변 후보만 국소 탐색합니다.
6. 목표 광출력 오차와 ADC 증가량을 함께 고려하여 최적 ADC Level을 선택합니다.

적용한 목적함수는 다음과 같습니다.

```text
|목표 광출력 - 예측 광출력| + λ × ADC 증가량
```

## 실행 방법

먼저 광출력 예측 모델을 학습합니다.

```bash
pip install pandas numpy scikit-learn matplotlib openpyxl joblib
python 01_train_predict_model.py
```

이후 학습된 모델을 이용하여 ADC 보상량을 계산합니다.

```bash
python 02_local_search_objective.py
```

## 생성 파일

예측 모델 학습 후 다음과 같은 결과 폴더가 생성됩니다.

```text
outputs_uv_led_ai/
├── best_model.pkl
├── best_model_info.json
├── cv_results_all_candidates.csv
├── cv_summary_all_candidates.csv
├── test_metrics.csv
└── graphs/
```

ADC 보상량 계산 후 다음과 같은 결과 폴더가 생성됩니다.

```text
outputs_uv_led_05_local_search_objective/
├── local_compensation_results.csv
├── local_compensation_summary.csv
├── local_compensation_summary_by_adc.csv
├── local_optimization_info.json
└── graphs/
```

## 사용 기술

* Python
* pandas
* NumPy
* scikit-learn
* Matplotlib
* openpyxl
* joblib

## 정리

본 프로젝트는 UV LED의 온도 변화에 따른 광출력 저하를 예측하고, 목표 광출력을 유지하기 위한 ADC 보상량을 계산하는 프로젝트입니다. 회귀 모델 기반 광출력 예측과 국소 탐색 기반 목적함수 최적화를 결합하여 정출력 유지 제어에 활용할 수 있는 보상값 산출 흐름을 구성합니다.
