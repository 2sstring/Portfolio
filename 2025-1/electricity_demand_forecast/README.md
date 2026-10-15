# ⚡ electricity_demand_forecast

**가정용 전력 소비 데이터를 활용한 시계열 예측 및 모델별 성능 비교 프로젝트**

Python을 이용해 가정용 전력 소비 시계열 데이터를 전처리하고, 통계 기반 모델과 딥러닝 기반 모델의 예측 성능을 비교합니다. **ARMA, ARIMA, RNN, LSTM, GRU, 1D CNN** 총 6개 모델을 적용하여 일별 평균 유효전력(`Global_active_power`)을 예측하고, 평균제곱오차(MSE)와 그래프를 통해 결과를 분석합니다.

---

## 📌 프로젝트 개요

이 프로젝트는 UCI Machine Learning Repository에서 제공하는 가정용 전력 소비 데이터를 사용합니다. 분 단위로 측정된 전력 데이터를 **일별 평균값**으로 변환한 뒤, 과거의 전력 소비 패턴으로부터 이후 시점의 전력 수준을 예측합니다.

주요 작업은 다음과 같습니다.

- 원본 전력 소비 데이터의 결측치 처리 및 일별 시계열 변환
- 시계열 순서를 유지한 학습 데이터와 테스트 데이터 분리
- 통계 기반 모델(ARMA, ARIMA)과 딥러닝 모델(RNN, LSTM, GRU, 1D CNN) 학습
- 테스트 구간 예측 및 MSE 기반 성능 비교
- 실제값과 예측값, 모델별 MSE 시각화

## 📂 프로젝트 폴더 구조

```text
electricity_demand_forecast/
├── main.py                         # 데이터 전처리, 모델 학습 및 성능 평가
├── README.md                       # 프로젝트 설명 문서
└── household_power_consumption.txt # 원본 데이터 (별도 다운로드 필요)
```

> `household_power_consumption.txt`는 GitHub 저장소에 포함되어 있지 않으며, 사용자가 별도로 다운로드해야 합니다.

## 🗂️ 데이터셋

- **데이터셋:** Individual Household Electric Power Consumption
- **제공처:** UCI Machine Learning Repository
- **데이터 링크:** [UCI 데이터셋 바로가기](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
- **사용 변수:** `Global_active_power` (가정의 유효전력 측정값)
- **예측 대상:** `Global_active_power`의 일별 평균값

### 📥 데이터 다운로드 및 준비 방법

1. 위 데이터 링크에 접속합니다.
2. `individual+household+electric+power+consumption.zip` 파일을 다운로드합니다.
3. 압축을 해제하고 `household_power_consumption.txt` 파일을 찾습니다.
4. 해당 파일을 `main.py`와 같은 `electricity_demand_forecast` 폴더에 넣습니다.

> ⚠️ 원본 데이터 파일은 25MB 이상으로, 본 GitHub 저장소에는 포함하지 않았습니다. 프로그램 실행 전에 반드시 준비해 주세요.

## ⚙️ 데이터 전처리

`main.py`의 `load_and_preprocess()` 함수에서는 다음 과정을 수행합니다.

| 단계 | 처리 내용 |
| --- | --- |
| 1. 데이터 로드 | 세미콜론(`;`)으로 구분된 TXT 파일 읽기 |
| 2. 날짜 및 시간 처리 | `Date`와 `Time`을 결합하여 날짜·시간 인덱스 생성 |
| 3. 결측치 처리 | `?`를 결측치로 인식하고 `Global_active_power`의 결측값 제거 |
| 4. 일별 변환 | 분 단위 유효전력을 일별 평균(`resample('D').mean()`)으로 집계 |
| 5. 데이터 분리 | 시간 순서대로 학습용 80%, 테스트용 20%로 분리 |

**참고:** 일별 평균 유효전력을 예측하는 것이며, 일별 누적 에너지 사용량(kWh)을 합산하는 과정은 아닙니다.

## 🧠 적용 예측 모델

| 구분 | 모델 | 주요 특징 |
| --- | --- | --- |
| 통계 모델 | ARMA | 자기회귀 및 이동평균을 결합한 시계열 모델 |
| 통계 모델 | ARIMA | 차분을 포함하여 시계열을 모델링하는 방법 |
| 딥러닝 모델 | RNN | 순차적으로 입력되는 시계열 정보 학습 |
| 딥러닝 모델 | LSTM | 장·단기 의존성을 고려하는 순환 신경망 |
| 딥러닝 모델 | GRU | 게이트 구조를 활용한 순환 신경망 |
| 딥러닝 모델 | 1D CNN | 1차원 합성곱으로 시계열 패턴 추출 |

코드에서 ARMA는 `ARIMA(order=(2, 0, 2))`, ARIMA는 `ARIMA(order=(2, 1, 2))`로 구현되어 있습니다.

### 🔧 주요 실험 설정

| 항목 | 설정값 |
| --- | --- |
| 학습 / 테스트 비율 | 80% / 20% (시간 순서 유지) |
| 딥러닝 입력 시퀀스 | 이전 30일의 일별 평균 유효전력 |
| 예측 대상 | 다음 1일의 일별 평균 유효전력 |
| RNN / LSTM / GRU 은닉 크기 | 32 |
| RNN / LSTM / GRU 계층 수 | 1 |
| 1D CNN 필터 수 / 커널 크기 | 32 / 3 |
| 학습 횟수 | 200 Epochs |
| 배치 크기 | 64 |
| 옵티마이저 | Adam (학습률 0.001) |
| 손실 함수 | MSELoss |
| 연산 장치 | CUDA 사용 가능 시 GPU, 그 외 CPU |

## 🚀 실행 방법

### 1. 프로젝트 폴더로 이동

```bash
cd electricity_demand_forecast
```

### 2. 필요한 라이브러리 설치

```bash
pip install numpy pandas matplotlib scikit-learn statsmodels torch
```

### 3. 데이터 파일 확인

`household_power_consumption.txt` 파일이 `main.py`와 같은 폴더에 있는지 확인합니다.

### 4. 프로그램 실행

```bash
python main.py
```

실행하면 데이터 로드 및 전처리, 6개 모델의 학습·예측, MSE 출력, 결과 그래프 표시가 순서대로 진행됩니다.

## 📊 성능 평가 및 시각화

### 평가 지표: MSE (Mean Squared Error)

모델별 예측 성능은 실제값과 예측값의 차이를 제곱한 뒤 평균한 **평균제곱오차(MSE)**로 평가합니다.

$$
\mathrm{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

MSE가 작을수록 실제 관측값과 예측값 사이의 평균 제곱 오차가 작음을 의미합니다.

### 출력 결과

- **실제값과 모델별 예측값 비교 그래프:** 테스트 구간의 실제 전력 값과 6개 모델의 예측 결과를 선 그래프로 비교
- **모델별 MSE 막대그래프:** 6개 모델의 오차 크기를 한눈에 비교
- **콘솔 출력:** 데이터 크기, 학습 과정의 손실값, 모델별 최종 MSE 표시

> 실제 MSE 수치는 실행 환경과 모델 학습 결과에 따라 달라질 수 있으므로 README에 고정된 성능 수치를 제시하지 않았습니다.

## 🛠️ 사용 기술

- **언어:** Python
- **데이터 처리:** pandas, NumPy
- **통계적 시계열 모델:** statsmodels
- **딥러닝:** PyTorch
- **성능 평가:** scikit-learn
- **시각화:** Matplotlib

## 📝 참고 사항

- 본 프로젝트는 가정용 전력 소비 데이터를 이용한 **시계열 예측 모델 학습 및 비교 실습**을 목적으로 합니다.
- 예측에는 `Global_active_power` 한 변수의 과거 값만 사용합니다.
- 딥러닝 모델은 테스트 시점 이전의 **실제 관측값을 포함한 30일 입력 시퀀스**로 예측하지만, ARMA/ARIMA는 학습 구간을 기준으로 테스트 구간을 예측합니다. 따라서 두 모델군의 평가 조건이 완전히 같지는 않으며, MSE 비교 시 이 차이를 고려해야 합니다.
- 별도의 난수 시드 고정이나 모델 파일 저장 기능은 현재 코드에 포함되어 있지 않습니다.
- 사용 중인 `statsmodels` 버전에 따라 ARIMA 예측 함수의 옵션 호환성을 확인해야 할 수 있습니다.

## 🔗 데이터 출처

[UCI Machine Learning Repository — Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
