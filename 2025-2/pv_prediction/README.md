# ☀️ pv_prediction

**ROS2 기반 태양광 발전량 예측 및 고장 상태 진단(AI Node) 프로젝트**

## 📌 프로젝트 소개

**pv_prediction**은 ROS2 환경에서 태양광 인버터의 발전량과 기상·환경 데이터를 수신하여, **LightGBM + SARIMAX 가중 앙상블**로 발전량을 예측하고 **실제 발전량과 예측값의 차이**를 바탕으로 운전 상태를 판정하는 프로젝트입니다.

`pv_data_publisher`의 시뮬레이터에서 발행하는 **`/pv/sample`** 토픽(JSON 문자열)을 구독하여 입력 특징을 생성하고, 예측값을 **`/pv/forecast`** 토픽으로 발행합니다. 설정에 따라 예측·고장 판정 결과를 **InfluxDB**에도 기록할 수 있습니다.

주요 목적은 실시간 스트리밍 환경에서 태양광 발전량 추정, 이상 상태 탐지, 모니터링 연동 과정을 시험하는 것입니다.

## 📁 프로젝트 폴더 구조

```text
pv_prediction/
├── README.md
└── src/
    └── pv_ai/
        ├── package.xml
        ├── setup.py
        ├── setup.cfg
        ├── test/
        │   ├── test_copyright.py
        │   ├── test_flake8.py
        │   └── test_pep257.py
        ├── launch/
        │   └── pv_forecast.launch.py
        ├── pv_ai/
        │   ├── __init__.py
        │   ├── pv_forecast_node.py
        │   ├── pv_sub_print.py
        │   └── __pycache__/                  # Python 실행 시 생성되는 캐시
        │       ├── __init__.cpython-310.pyc
        │       └── pv_sub_print.cpython-310.pyc
        └── resource/
            └── pv_ai
```

> **참고:** 위 구조는 저장소 파일 구성 기준입니다. 기존 실행 안내의 `/root/pv_ws`는 ROS2 컨테이너 내부의 워크스페이스 경로로, GitHub 폴더 이름과 다를 수 있습니다. 빌드 후에는 워크스페이스 최상위에 `build/`, `install/`, `log/`가 생성됩니다. `__pycache__/`와 `.pyc`는 자동 생성 파일로 일반적으로 Git 추적 대상에서 제외합니다.

### 🧩 주요 파일 설명

| 파일 및 폴더 | 역할 |
|---|---|
| `pv_ai/pv_forecast_node.py` | ROS2 토픽 구독, 입력 특징 생성, 발전량 예측, 고장 판정 및 InfluxDB 기록 |
| `pv_ai/pv_sub_print.py` | `/pv/sample` 수신 JSON의 주요 정보를 콘솔에 출력하는 확인용 노드 |
| `launch/pv_forecast.launch.py` | 발전량 예측 노드를 실행하도록 정의한 ROS2 launch 파일 |
| `setup.py` | ROS2 Python 패키지 설치 및 `ros2 run` 실행 진입점 등록 |
| `setup.cfg` | ROS2 실행 스크립트 설치 위치 설정 |
| `package.xml` | 패키지 메타데이터와 ROS2 의존성 정의 |
| `resource/pv_ai` | ROS2 패키지 인덱스 등록용 파일 |
| `test/` | 저작권·코드 스타일·문서화 형식 검사 파일 |

## 🔄 시스템 동작 구조

```text
태양광 데이터 Publisher (pv_data_publisher)
                  │
                  ▼
      /pv/sample (std_msgs/msg/String)
                  │
                  ▼
       PVForecastNode (pv_ai)
                  │
        ┌─────────┴───────────┐
        ▼                     ▼
   FeatureBuffer        태양 위치·일사량 계산
        └─────────┬───────────┘
                  ▼
         Lag / Rolling 등
           입력 특징 생성
                  │
                  ▼
     LightGBM + SARIMAX 앙상블
                  │
         ┌────────┴────────┐
         ▼                 ▼
 /pv/forecast 발행     실제값과 비교하여
  (발전량 예측값)       운전 상태 판정
                           │
                           ▼
                   InfluxDB 기록(선택)
```

### 📡 ROS2 토픽

| 구분 | 토픽 | 메시지 타입 | 설명 |
|---|---|---|---|
| Subscribe | `/pv/sample` | `std_msgs/msg/String` | 발전량·기상 정보를 담은 JSON 문자열 수신 |
| Publish | `/pv/forecast` | `std_msgs/msg/Float32` | 앙상블 발전량 예측값 발행 (W) |

입력 JSON 예시:

```json
{
  "ts": "2025-11-30T12:00:00",
  "inverter_id": "INV1",
  "ac_power": 1234.5,
  "dni": 800.0,
  "dhi": 100.0,
  "ws": 2.3,
  "rh": 45.6,
  "module_temp": 35.0,
  "hour": 12,
  "status": "OK"
}
```

> `ac_power`는 **W** 단위로 수신하며 모델 입력·내부 예측에는 **kW** 단위를 사용합니다. `/pv/forecast`로 발행할 때는 다시 **W**로 변환합니다. `module_temp`, 일사량 등 주요 입력이 없으면 특징 생성이 불가능하여 예측이 생략될 수 있습니다. 현재 코드는 `/pv/sample`의 **JSON 문자열**을 처리하므로, 별도로 정의된 `PVSample.msg` 타입을 곧바로 구독하는 구조는 아닙니다.

## 🧠 예측 모델 및 특징 생성

`pv_forecast_node.py`는 미리 학습한 모델을 로드하고, 수신 데이터에서 예측 입력을 구성합니다.

| 구성 요소 | 처리 내용 |
|---|---|
| `FeatureBuffer` | 최근 입력 데이터를 저장하며, 최소한의 이력이 쌓인 뒤 특징 생성 |
| 시간 특징 | 시각 및 주기성을 나타내는 `hour_cos` 등 |
| 기상·일사 특징 | `DNI`, `DHI`, `WS`, `RH`, `Module_Temperature`, `GHI_est` |
| 태양 위치 | `pvlib`를 이용한 태양고도(`solar_altitude`), 태양천정각(`solar_zenith`) |
| 시계열 특징 | 이전 시점 값(Lag), 이동 평균(Rolling mean), 일부 이동 표준편차 |
| LightGBM | 현재 특징 벡터를 이용한 발전량 추정 |
| SARIMAX | 저장된 시계열 모델을 이용한 예측 |
| 가중 앙상블 | `ensemble_weights.json`의 가중치로 두 모델의 결과 결합 |

예측 노드 실행 전에 **`pv_power_forecasting` 프로젝트에서 생성한 모델 파일**을 준비해야 합니다. 이 저장소 자체에는 모델 학습 스크립트가 포함되어 있지 않습니다.

## 🚦 고장 및 운전 상태 판정

발전량 예측값과 실제값, 태양고도, 추정 수평면일사량 등을 이용하여 다음 상태를 처리합니다.

| 상태 코드 | 의미 |
|---|---|
| `NORMAL` | 실제 발전량과 예측 발전량의 차이가 허용 범위 이내 |
| `FAULT_OVER` | 예측 발전량이 실제 발전량보다 높으며 허용 오차 초과 |
| `FAULT_UNDER` | 예측 발전량이 실제 발전량보다 낮으며 허용 오차 초과 |
| `FAULT_UNDER_IRR` | 낮에 일사량이 충분하지만 실제 발전 출력이 매우 낮은 경우 |
| `NIGHT` | 태양고도가 0° 이하인 야간 상태 |
| `NO_ACTUAL` | 실제 발전량이 없어 비교할 수 없는 경우를 위한 상태 코드 |

현재 소스의 주요 판정 기준은 다음과 같습니다.

- **오차 허용 기준:** `max(30 W, 실제 발전량의 30%)`
- **주간 판정:** 태양고도 `> 0°`
- **일사량 기반 저출력 판정:** 추정 `GHI >= 50 W/m²`이고 실제 발전량 `< 20 W`
- **야간:** 일반적인 발전량 오차 기반 고장 판정 대신 `NIGHT` 기록

> 이 기준은 코드에 설정된 **실험용 판정 규칙**이며, 실제 발전소의 설비 용량·환경·진단 기준에 따라 조정이 필요합니다. `NO_ACTUAL` 등의 상태는 정의되어 있어도 입력 수신 및 특징 생성 조건에 따라 실제 발생 여부가 달라질 수 있습니다.

## 🛠️ 실행 환경 및 사전 준비

| 항목 | 내용 |
|---|---|
| 실행 환경 | Ubuntu 기반 Docker 컨테이너 (기존 README 기준) |
| ROS2 | Humble |
| 패키지 빌드 | `colcon`, `ament_python` |
| ROS2 Python 패키지 | `rclpy`, `std_msgs` |
| 주요 Python 라이브러리 | `numpy`, `pandas`, `joblib`, `lightgbm`, `statsmodels`, `pvlib`, `influxdb-client` |
| 외부 입력 | `pv_data_publisher`의 `/pv/sample` 데이터 |
| 선택적 저장소 | InfluxDB 2.x |

ROS2가 설치된 Python 환경에 필요한 라이브러리가 없다면 설치합니다.

```bash
python3 -m pip install numpy pandas joblib lightgbm statsmodels pvlib influxdb-client
```

> ROS2 Humble에서 사용하는 Python 환경과 패키지 버전의 호환성을 확인해야 합니다. 위 명령은 코드에 필요한 주요 외부 패키지 안내이며, 모든 환경에서의 설치 성공을 보장하지는 않습니다.

### 📦 필수 모델 파일

컨테이너 내부 **`/root/pv_models/`** 경로에 아래 파일을 준비합니다.

```text
/root/pv_models/
├── lgbm_model.pkl
├── sarimax_model.pkl
├── ensemble_weights.json
└── feature_cols.json
```

| 파일 | 용도 |
|---|---|
| `lgbm_model.pkl` | 학습된 LightGBM 모델 |
| `sarimax_model.pkl` | 학습된 SARIMAX 결과 객체 |
| `ensemble_weights.json` | LightGBM과 SARIMAX 앙상블 가중치 |
| `feature_cols.json` | 학습 시 사용한 입력 특징의 이름 및 순서 |

> 모델을 저장할 때와 추론할 때의 특징 구성, 값의 단위, 전처리 규칙 및 주요 라이브러리 버전이 일치해야 합니다. 모델 파일이 없으면 예측 노드를 시작할 수 없습니다.

## 🚀 실행 방법

다음 명령은 기존 README의 **Docker + ROS2 Humble + `/root/pv_ws`** 환경을 기준으로 한 예시입니다. 컨테이너 이름과 경로는 실제 환경에 맞게 변경합니다.

### 1. Docker 컨테이너 시작

```bash
# InfluxDB에 기록하는 경우 (컨테이너가 이미 생성되어 있을 때)
docker start influxdb

# ROS2 컨테이너 실행
docker start <ROS2_컨테이너명>
```

### 2. ROS2 컨테이너 접속 및 워크스페이스 빌드

```bash
docker exec -it <ROS2_컨테이너명> bash

source /opt/ros/humble/setup.bash
cd /root/pv_ws

# 최초 실행 또는 소스 변경 시
colcon build --packages-select pv_ai --symlink-install
source install/setup.bash

export ROS_DOMAIN_ID=10
```

> `pv_prediction/src/pv_ai`가 ROS2 워크스페이스의 `src/` 아래에 배치되어 있어야 합니다. 이미 빌드된 환경이라면 재빌드 없이 `source install/setup.bash`를 실행할 수 있습니다.

### 3. 발전량 예측 노드 실행

```bash
ros2 run pv_ai pv_forecast_node
```

별도 터미널에서 수신 데이터만 확인하려면:

```bash
ros2 run pv_ai pv_sub_print
```

Publisher 역시 실행되어 있어야 데이터가 전달됩니다. 양쪽 ROS2 환경에서 **`ROS_DOMAIN_ID=10`**을 일치시키고 컨테이너 간 ROS2 통신이 가능한지 확인합니다.

### 4. ROS2 토픽 확인

```bash
ros2 topic list
ros2 topic info /pv/sample
ros2 topic echo /pv/sample
ros2 topic info /pv/forecast
ros2 topic echo /pv/forecast
```

> `launch/pv_forecast.launch.py` 파일이 존재하지만, **현재 `setup.py`의 `data_files`에는 launch 파일 설치 설정이 없습니다.** 따라서 `ros2 launch pv_ai pv_forecast.launch.py` 명령은 별도 설치 설정을 추가하지 않으면 파일을 찾지 못할 수 있습니다. 기본 실행은 위의 `ros2 run` 명령을 사용하세요.

## 🗃️ InfluxDB 연동 (선택)

예측 노드는 환경 변수를 통해 InfluxDB 접속 정보를 읽습니다.

```bash
export INFLUX_URL="http://localhost:8086"
export INFLUX_ORG="my-org"
export INFLUX_BUCKET="pv"
export INFLUX_TOKEN="<본인_InfluxDB_토큰>"
```

| 항목 | 코드에서 사용하는 값 |
|---|---|
| Measurement | `pv_forecast` |
| 태그(Tag) | `inverter_id`, `status` |
| 발전량·오차 | `p_actual_w`, `p_pred_w`, `p_pred_kw`, `err_w`, `rel_err_percent`, `threshold_w` |
| 고장 여부 | `is_fault` (`FAULT`로 시작하는 상태는 1, 그 외는 0) |
| 환경·일사량 | `dni`, `dhi`, `ws`, `rh`, `module_temp`, `ghi_est`, `solar_altitude`, `solar_zenith` |

> **중요:** `INFLUX_TOKEN`이 설정되지 않으면 예측 노드는 InfluxDB 기록을 비활성화하고 동작을 계속합니다. InfluxDB 서버가 다른 컨테이너에서 실행되는 경우, `localhost:8086`이 올바른 주소인지 컨테이너 네트워크 구성을 확인해야 합니다. 토큰은 GitHub에 커밋하지 마세요.

## 🧪 테스트 및 확인

기본적인 패키지 검사 파일은 `test/` 디렉터리에 포함되어 있습니다. ROS2 환경에서 필요 시 아래 명령을 사용할 수 있습니다.

```bash
cd /root/pv_ws
colcon test --packages-select pv_ai
colcon test-result --verbose
```

`test_copyright.py`는 현재 코드에서 건너뛰도록 설정되어 있습니다. 나머지 코드 스타일·문서화 검사도 실행 환경 및 소스 상태에 따라 결과가 달라질 수 있습니다.

## ⚠️ 실행 및 활용 시 주의사항

- **모델 파일 필수:** 4개 모델·메타데이터 파일을 `/root/pv_models/`에 준비해야 합니다.
- **입력 타입 확인:** 현재 노드는 `/pv/sample`의 `std_msgs/msg/String` JSON 메시지를 구독합니다. Publisher가 사용자 정의 `PVSample` 메시지를 직접 발행하는 경우에는 구독 타입을 맞추어야 합니다.
- **충분한 이력 필요:** Lag/Rolling 특징 생성에 필요한 수신 이력이 부족하거나 필수 특징이 누락되면 예측이 생략될 수 있습니다.
- **ROS2 환경 일치:** Publisher와 예측 노드가 같은 ROS Domain 및 통신 가능한 네트워크에 있어야 합니다.
- **시간·발전량 단위:** `ac_power`는 W, 내부 예측은 kW를 사용하며, 타임존과 태양광 설비 위치가 학습 환경과 맞는지 확인해야 합니다.
- **Launch 설치 설정:** 현재 `setup.py`는 `launch/` 파일을 설치하지 않으므로 필요 시 패키징 설정을 보완해야 합니다.
- **GitHub 관리:** `__pycache__/`, `.pyc`, `build/`, `install/`, `log/` 등은 보통 버전 관리에서 제외합니다.
- **진단 결과 해석:** 이 프로젝트는 프로토타입입니다. 고장 판정 코드만으로 실제 설비 고장을 확정하지 않으며, 현장 기준에 따른 검증이 필요합니다.

## 🎯 프로젝트 목적

- ROS2 기반 태양광 발전 데이터의 **실시간 수신 및 온라인 추론**
- **LightGBM + SARIMAX 앙상블**을 이용한 발전량 예측
- 실제 출력과 예측값, 일사량·태양고도 등을 활용한 **운전 상태 진단**
- **InfluxDB 기반 기록 및 향후 모니터링 시스템 연동**
- 실제 센서 연결 전 시뮬레이션 환경에서의 기능 검증

---

**관련 프로젝트:** `pv_data_publisher` (ROS2 데이터 발행) · `pv_power_forecasting` (오프라인 모델 학습 및 모델 파일 생성)
