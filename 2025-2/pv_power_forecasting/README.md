# ☀️ pv_power_forecasting

**LightGBM·SARIMAX 및 가중 앙상블을 활용한 태양광 발전량 추정·예측 프로젝트**

태양광 발전 데이터의 기상·일사·운전 정보를 전처리하고 시간 관련 특징을 생성한 뒤, **LightGBM**, **SARIMAX** 및 두 모델의 **성능 기반 가중 앙상블**을 비교합니다. 모델별 평가 지표와 학습된 모델, 앙상블 가중치를 파일로 저장하여 추후 다른 시스템에서 활용할 수 있도록 구성했습니다.

> ⚠️ **비공개 데이터 안내:** `SolarData_merged.csv`와 `SolarData_60min_for_train.csv`의 원본 데이터는 공개할 수 없습니다. GitHub에 업로드한 두 CSV는 **내용이 없는 빈 파일**이며, 실행하려면 별도로 사용 권한이 있는 실제 데이터를 준비해야 합니다.

---

## 📌 프로젝트 개요

이 프로젝트의 주요 작업은 다음과 같습니다.

- 시간 정보와 태양광 관련 측정값을 통합하여 **60분 간격 데이터** 생성
- 결측치·물리적 범위를 벗어난 값 제거 및 **태양 위치를 이용한 주간 데이터 추출**
- 시간, 과거 시점(Lag), 이동 통계(Rolling), 태양 위치, 추정 일사량 등의 **입력 특징 생성**
- **LightGBM 회귀 모델**과 **SARIMAX 시계열 모델**의 발전량 예측 성능 비교
- 개별 모델의 MAE를 이용한 **가중 앙상블** 결과 계산
- 성능 평가 CSV, 모델 파일 및 특징 목록 저장

## 📂 프로젝트 폴더 구조

```text
pv_power_forecasting/
├── 01_pre_processing.py                    # 데이터 정제·리샘플링·특징 생성
├── 02_lgbm+sarimax_model.py                # 모델 학습·평가·가중 앙상블
├── SolarData_merged.csv                    # 비공개 원본 데이터의 빈 파일
├── SolarData_60min_for_train.csv           # 비공개 전처리 데이터의 빈 파일
└── README.md                               # 프로젝트 설명 및 실행 안내
```

분석에 필요한 실제 데이터를 준비하고 스크립트를 실행하면 다음과 같은 결과 파일이 **추가로 생성**됩니다.

```text
pv_power_forecasting/
├── SolarData_60min_for_train_lgbm_sarimax.csv  # 모델 성능 비교표
└── models/                                      # 실행 시 생성
    ├── lgbm_model.pkl                          # 학습된 LightGBM 모델
    ├── sarimax_model.pkl                       # 학습된 SARIMAX 결과 객체
    ├── ensemble_weights.json                   # 앙상블 가중치
    └── feature_cols.json                       # LightGBM 입력 특징 목록
```

| 파일 | 역할 |
| --- | --- |
| `01_pre_processing.py` | 원본 데이터를 60분 간격으로 정리하고 예측에 사용할 특징을 생성 |
| `02_lgbm+sarimax_model.py` | 데이터 기간 분리, LightGBM·SARIMAX 학습, 앙상블 및 성능 평가 |
| `SolarData_merged.csv` | 전처리 입력 파일명 안내용 빈 CSV |
| `SolarData_60min_for_train.csv` | 전처리 결과이자 모델 학습 입력 파일명 안내용 빈 CSV |
| `README.md` | 프로젝트 구성, 실행 방법, 데이터 공개 제한 및 주의사항 |

## 🗂️ 사용 데이터

| 구분 | 내용 |
| --- | --- |
| 원본 입력 | `SolarData_merged.csv` |
| 전처리 결과 | `SolarData_60min_for_train.csv` |
| 분석 시간 간격 | **60분** (`RESAMPLE_RULE = "60min"`) |
| 시간 열 | `Datetime` |
| 예측 대상 | `Target` |
| 핵심 측정 열 | `DNI`, `DHI`, `Temperature`, `Module_Temperature`, `WS`, `RH` |

- **DNI / DHI:** 직달일사량 / 산란일사량
- **Temperature / Module_Temperature:** 기온 / 모듈 온도
- **WS / RH:** 풍속 / 상대습도
- **Target:** 코드에서 예측 대상으로 사용하는 태양광 발전량 관련 값

> `Target`의 정확한 단위 및 정규화 여부는 비공개 원본 데이터의 정의를 확인해야 합니다. 빈 CSV 파일에는 실제 관측값이나 분석 가능한 열 데이터가 포함되어 있지 않습니다.

### 🔒 원본 데이터 안내

`SolarData_merged.csv`와 `SolarData_60min_for_train.csv`는 **공개할 수 없는 데이터**에 해당합니다. 저장소에 포함된 빈 파일은 필요한 파일명을 안내하기 위한 것이며, 실제 측정 데이터가 아닙니다.

실행하려면 사용 권한이 있는 데이터를 해당 파일명과 코드에 맞는 컬럼 구조로 준비해야 합니다. 비공개 원본 자료를 저장소에 업로드하지 않도록 주의해 주세요.

## ⚙️ 데이터 전처리 및 특징 생성

`01_pre_processing.py`는 다음 순서로 동작합니다.

| 단계 | 처리 내용 |
| --- | --- |
| 1. 데이터 로드 | `Datetime` 파싱, 시간순 정렬, 중복 시각 제거 |
| 2. 리샘플링 | 숫자형 변수의 **60분 평균값** 생성 |
| 3. 데이터 정제 | 핵심 변수 결측값과 지정된 물리 범위를 벗어난 관측값 제거 |
| 4. 주간 필터링 | `pvlib`의 태양 위치 계산을 이용하여 태양이 떠 있는 시각만 선택 |
| 5. 시간 특징 생성 | 연·월·일·시·분, 연중 일자, 시간의 `sin/cos` 변환 |
| 6. 과거 정보 특징 | `Target`, 일사량, 모듈 온도에 대한 Lag 및 Rolling 특징 생성 |
| 7. 추가 특징 | 과거 날짜 기반 Target 평균, 이슬점, 추정 수평면 일사량 등 계산 |
| 8. 최종 저장 | 특징 생성으로 발생한 결측 행을 제거하고 CSV로 저장 |

### 주요 특징 예시

| 특징 그룹 | 예시 |
| --- | --- |
| 시간 | `Hour`, `hour_sin`, `hour_cos`, `doy` |
| 이전 시점 | `Target_lag1`, `DNI_lag3`, `Module_Temperature_lag1` |
| 이동 통계 | `Target_roll_mean3`, `DHI_roll_std3` |
| 태양 위치·일사량 | `solar_zenith`, `solar_altitude`, `GHI_est` |
| 기상 파생 정보 | `DewPoint` |
| 과거 날짜 기반 | `Target_mean_3d_same_time`, `Target_mean_5d_same_time` |

**추정 수평면 일사량**은 코드에서 다음과 같이 계산합니다.

```text
GHI_est = DNI × max(cos(solar_zenith), 0) + DHI
```

`LATITUDE`, `LONGITUDE`, `TIMEZONE`은 태양 위치 계산에 사용됩니다. 스크립트에는 예시 좌표와 `Asia/Seoul`이 설정되어 있으므로 실제 데이터의 발전소 위치 및 시간대에 맞게 확인해야 합니다.

> 참고: 과거 동일 시간대 평균은 정제·주간 필터링된 데이터의 추정 샘플 간격을 바탕으로 `shift`를 사용합니다. 누락 시각이나 일별 주간 샘플 수가 달라지는 경우 정확한 전일 동일 시각과 일치하는지 확인이 필요합니다.

## 🤖 예측 모델 및 실험 설정

`02_lgbm+sarimax_model.py`에서 아래 세 가지 예측 결과를 비교합니다.

| 모델 | 설명 |
| --- | --- |
| **LightGBM** | Lag, Rolling, 기상, 일사량, 시간 특징 등을 입력하는 회귀 모델 |
| **SARIMAX** | `Target`의 과거 시계열을 사용하는 통계 모델 |
| **LightGBM + SARIMAX** | 두 모델의 예측값을 성능 기반 가중치로 결합한 앙상블 |

### 데이터 분할

| 구분 | 기간 |
| --- | --- |
| 학습 데이터 | **2016~2017년** |
| 테스트 데이터 | **2018년** |
| 정렬 기준 | `Datetime` 오름차순 |

### 주요 모델 설정

| 항목 | 설정값 |
| --- | --- |
| LightGBM | `n_estimators=1000`, `learning_rate=0.05`, `subsample=0.9`, `colsample_bytree=0.9` |
| SARIMAX | `order=(1, 0, 1)` |
| 난수 시드 | `42` |
| 앙상블 가중치 | 각 모델의 MAE 역수에 비례하여 계산 |

SARIMAX 코드는 현재 **외생 변수(`exog`)를 넣지 않고 `Target` 시계열만 사용**합니다. 또한 계절 차수를 지정하는 `seasonal_order` 부분은 주석 처리되어 있어 현재 실행 설정에서 계절 성분은 별도로 지정하지 않습니다.

### 가중 앙상블 방식

개별 모델의 오차가 작을수록 높은 가중치를 부여합니다.

```text
가중치(모델 i) = (1 / MAE_i) / Σ(1 / MAE_j)
앙상블 예측값 = Σ(가중치_i × 모델별 예측값_i)
```

> ⚠️ **평가 시 주의:** 현재 코드는 **2018년 테스트 데이터에서 계산한 MAE로 앙상블 가중치를 정한 뒤, 동일한 2018년 데이터에서 앙상블 성능을 평가**합니다. 따라서 앙상블 결과는 독립적인 최종 테스트 성능으로 해석하기 어렵습니다. 엄밀한 일반화 성능 평가를 위해서는 가중치를 별도 검증 데이터에서 정해야 합니다.

## 📊 성능 평가 및 결과 파일

모델별로 다음 지표를 계산하여 CSV로 저장합니다.

| 지표 | 설명 |
| --- | --- |
| MAE | 평균 절대오차 |
| RMSE | 평균 제곱근 오차 |
| MSE | 평균 제곱오차 |
| R² | 결정계수 |
| NMAE | 테스트 `Target`의 최댓값과 최솟값 차이로 MAE를 나눈 값 |
| ACC_within_tol | 절대오차가 지정된 허용오차 이내인 샘플 비율 |
| KPX_NMAE / FER | 스크립트에 구현된 별도 계산식 기반 지표 |

- 허용오차 설정은 **`TOLERANCE = 0.08`**이며, 이는 `Target`과 동일한 값의 척도에서 사용하는 **절대오차 기준**입니다.
- 코드의 `KPX_NMAE`는 `Target >= 0.1` 구간에서 **절대오차 평균 × 100**으로 계산합니다. 설비용량 기반 정규화가 포함되어 있지 않으므로 공식 평가 지표와 동일하다고 간주하지 않아야 합니다.
- 실제 성능 수치는 원본 데이터가 공개되지 않았으므로 README에 기재하지 않았습니다.

**실행 후 생성 파일**

| 결과 파일 | 내용 |
| --- | --- |
| `SolarData_60min_for_train_lgbm_sarimax.csv` | LightGBM, SARIMAX, 앙상블의 성능 비교 결과 |
| `models/lgbm_model.pkl` | LightGBM 모델 |
| `models/sarimax_model.pkl` | SARIMAX 학습 결과 객체 |
| `models/ensemble_weights.json` | 모델별 앙상블 가중치 |
| `models/feature_cols.json` | 학습에 사용된 LightGBM 입력 변수 목록 |

## 🛠️ 사용 기술

| 도구·라이브러리 | 용도 |
| --- | --- |
| Python | 전처리 및 모델 학습 실행 |
| pandas / NumPy | 데이터 처리, 파생 특징 생성 및 수치 계산 |
| pvlib | 태양 위치 계산 |
| LightGBM | 트리 앙상블 기반 발전량 회귀 |
| statsmodels | SARIMAX 시계열 모델 |
| scikit-learn | 예측 성능 평가 |
| joblib | LightGBM 모델 저장 |

## 🚀 설치 및 실행 방법

### 1. 프로젝트 폴더로 이동

```bash
cd pv_power_forecasting
```

### 2. 필요한 라이브러리 설치

```bash
pip install pandas numpy pvlib lightgbm statsmodels scikit-learn joblib
```

### 3. 실제 데이터 준비

사용 권한이 있는 원본 데이터의 열 이름을 코드에서 사용하는 형식에 맞춰 `SolarData_merged.csv`로 저장하고 프로젝트 최상위 폴더에 배치합니다.

최소한 `Datetime`, `DHI`, `DNI`, `Temperature`, `Module_Temperature`, `WS`, `RH`, `Target` 등의 입력 열이 필요합니다. 원본 빈 CSV로는 전처리나 모델 학습을 진행할 수 없습니다.

### 4. 전처리 출력 경로 관련 코드 확인

현재 `01_pre_processing.py`의 기본 출력 파일명에는 폴더 경로가 없습니다.

```python
OUTPUT_CSV_PATH = r"SolarData_60min_for_train.csv"
```

그런데 스크립트 마지막에는 `os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)`가 있어, **빈 디렉터리 이름으로 폴더를 만들려고 하면서 오류가 발생할 수 있습니다.** 출력 위치를 현재 폴더로 유지하려면 이 부분을 다음과 같이 수정합니다.

```python
output_dir = os.path.dirname(OUTPUT_CSV_PATH)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

df_feat.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
```

수정 전후의 데이터 입출력 파일명이 일치하는지도 확인합니다.

### 5. 데이터 전처리 실행

```bash
python 01_pre_processing.py
```

실제 데이터가 준비되어 있다면 `SolarData_60min_for_train.csv`가 생성됩니다.

### 6. 모델 학습 및 성능 비교 실행

```bash
python "02_lgbm+sarimax_model.py"
```

실행 후 프로젝트 폴더에 성능 비교 CSV와 `models/` 폴더가 생성됩니다. 운영체제나 터미널에 따라 특수문자가 포함된 파일명은 위와 같이 따옴표로 감싸서 실행할 수 있습니다.

## ⚠️ 참고 및 주의사항

- **비공개 데이터:** 업로드된 두 CSV는 빈 파일입니다. 원본 태양광 발전 측정값과 실제 예측 결과는 제공되지 않습니다.
- **데이터 기간:** 모델은 **2016·2017년 학습, 2018년 테스트**로 고정되어 있습니다. 해당 기간의 유효 데이터가 있어야 합니다.
- **위치 및 시간대:** 태양 위치 계산용 좌표와 시간대는 실제 발전소에 맞게 확인해야 합니다.
- **전처리 출력 경로:** 현재 스크립트의 빈 디렉터리 생성 문제는 실행 전 수정해야 합니다.
- **라이브러리 호환성:** 사용하는 `scikit-learn` 버전에 따라 `mean_squared_error(..., squared=False)`가 지원되지 않을 수 있습니다. 이 경우 `np.sqrt(mean_squared_error(y_true, y_pred))`로 RMSE를 계산하도록 변경할 수 있습니다.
- **앙상블 평가:** 테스트 데이터로 계산한 오차를 가중치 산정에 다시 사용하므로 결과 해석에 주의해야 합니다.
- **모델 활용 범위:** 모델 파일을 저장하지만 **ROS2 연동 또는 실시간 서비스 노드는 이 폴더의 두 Python 스크립트에 포함되어 있지 않습니다.**

---

**이 프로젝트는 태양광 발전 데이터의 전처리, 특징 생성, 개별 예측 모델 비교 및 가중 앙상블 실험을 위한 코드입니다.**
