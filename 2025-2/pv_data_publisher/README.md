# ☀️ PV Data Publisher (Simulator)

ROS2 기반 태양광 발전 데이터 시뮬레이터 및 데이터 발행(Publisher) 프로젝트입니다.

## 📌 프로젝트 소개

**pv_data_publisher**는 태양광 발전소의 인버터 운전 정보와 기상·환경 데이터를 시뮬레이션하여 ROS2 토픽 **`/pv/sample`**로 발행하기 위한 워크스페이스입니다. 발행된 데이터는 AI 발전량 예측 노드(`PVForecastNode`)의 입력 데이터로 활용하는 것을 목적으로 합니다.

실제 현장 센서와 연결하지 않고도 데이터 수신, 예측 노드 연동 및 고장 판정 흐름을 점검할 수 있도록 구성한 실습·시험용 프로젝트입니다.

## 📁 폴더 구조

```text
pv_data_publisher/
├── README.md
└── src/
    ├── pv_interfaces/
    │   ├── CMakeLists.txt
    │   ├── package.xml
    │   ├── msg/
    │   │   └── PVSample.msg
    │   ├── src/
    │   │   └── empty
    │   └── include/
    │       └── pv_interfaces/
    │           └── empty
    └── pv_edge/
        └── (세부 구현 파일은 이번 제공 자료에 포함되지 않음)
```

> **참고:** `empty` 파일은 GitHub에서 비어 있는 폴더가 사라지는 것을 방지하기 위해 넣은 자리표시자입니다. ROS2의 `colcon build` 실행 후에는 워크스페이스 최상위에 `build/`, `install/`, `log/` 폴더가 자동 생성됩니다.
>
> 기존 실행 안내에서는 컨테이너 내부 워크스페이스 경로로 `/root/pv_ws_pub`을 사용합니다. **GitHub 저장소 이름(`pv_data_publisher`)과 컨테이너 내부 경로는 다를 수 있습니다.**

## 🧩 주요 구성 요소

| 구성 요소 | 역할 |
|---|---|
| `pv_interfaces/package.xml` | ROS2 사용자 정의 메시지 패키지 메타데이터 및 의존성 정의 |
| `pv_interfaces/CMakeLists.txt` | `PVSample.msg` 인터페이스 생성 설정 |
| `pv_interfaces/msg/PVSample.msg` | PV 센서 및 인버터 샘플 데이터의 필드 정의 |
| `pv_edge/` | 태양광 센서 데이터 시뮬레이터 노드를 포함하는 패키지(기존 README 기준) |
| `empty` | GitHub에 빈 디렉터리 구조를 보존하기 위한 파일 |

## 📡 ROS2 토픽 및 메시지

기존 README에 기록된 발행 토픽은 다음과 같습니다.

| 항목 | 내용 |
|---|---|
| 토픽 이름 | `/pv/sample` |
| 기존 README의 발행 타입 | `std_msgs/msg/String` (JSON 문자열) |
| 데이터 | 측정 시각, 인버터 ID, AC 출력, 일사량, 풍속, 습도, 상태 등 |
| 활용 대상 | AI 발전량 예측 노드(`PVForecastNode`) |

기존 README의 JSON 메시지 예시는 다음과 같습니다.

```json
{
  "ts": "2025-11-30T12:00:00",
  "inverter_id": "INV1",
  "ac_power": 1234.5,
  "dni": 800.0,
  "dhi": 100.0,
  "ws": 2.3,
  "rh": 45.6,
  "status": "OK"
}
```

### 사용자 정의 인터페이스: `PVSample.msg`

별도로 제공된 `pv_interfaces/msg/PVSample.msg`에는 다음 필드가 정의되어 있습니다.

| 필드 | 타입 | 설명 |
|---|---|---|
| `stamp` | `builtin_interfaces/Time` | ROS2 시각 |
| `plant_id` | `string` | 발전소 식별자 |
| `inverter_id` | `string` | 인버터 식별자 |
| `hour` | `int32` | 시간 정보 |
| `dhi` | `float32` | 산란일사량 (W/m²) |
| `dni` | `float32` | 직달일사량 (W/m²) |
| `module_temp` | `float32` | 모듈 온도 (°C) |
| `ambient_temp` | `float32` | 외기 온도 (°C) |
| `ws` | `float32` | 풍속 (m/s) |
| `rh` | `float32` | 상대습도 (%) |
| `ac_power` | `float32` | AC 출력 (W) |
| `status` | `string` | 운전 상태 (`OK`, `TRIP`, `LOW_OUTPUT` 등) |

> **메시지 타입 주의:** `PVSample.msg`가 정의되어 있다고 해서 `/pv/sample` 토픽이 반드시 `pv_interfaces/msg/PVSample` 타입으로 발행되는 것은 아닙니다. 기존 README는 JSON 문자열을 발행한다고 설명하고 있으나, `pv_edge`의 실제 노드 코드는 이번 자료에 포함되지 않아 사용 중인 토픽 타입을 확인할 수 없습니다. 실행 후 `ros2 topic info /pv/sample` 명령으로 확인하세요.

## 🛠️ 실행 환경

| 항목 | 환경 |
|---|---|
| 운영 환경 | Ubuntu 기반 Docker 컨테이너 (기존 안내 기준) |
| ROS2 | Humble |
| 빌드 도구 | `colcon` |
| 데이터 통신 | ROS2 Topic Publish/Subscribe |
| 선택적 연동 서비스 | InfluxDB (기존 실행 안내에 포함) |

## 🚀 실행 방법

아래 절차는 기존 README에 기록된 ROS2 Docker 실행 환경을 기준으로 합니다. **`pv_edge` 패키지의 실행 파일과 의존성이 준비되어 있어야** 시뮬레이터를 실행할 수 있습니다.

### 1. Docker 컨테이너 시작

```bash
# InfluxDB를 함께 사용하는 환경인 경우
# docker start influxdb

# ROS2가 설치된 컨테이너 시작
# 실제 컨테이너 이름으로 바꾸세요.
docker start <ROS2_컨테이너명>
```

### 2. ROS2 컨테이너 접속 및 환경 설정

```bash
docker exec -it <ROS2_컨테이너명> bash

# ROS2 Humble 환경 로드
source /opt/ros/humble/setup.bash

# 컨테이너의 실제 워크스페이스 경로로 이동
cd /root/pv_ws_pub
```

### 3. 워크스페이스 빌드 및 환경 로드

최초 실행이거나 인터페이스를 변경한 경우, 의존성 설정을 확인한 뒤 빌드합니다.

```bash
colcon build --symlink-install
source install/setup.bash

# 기존 프로젝트에서 사용하는 Domain ID
export ROS_DOMAIN_ID=10
```

> 기존에 `install/`이 준비되어 있다면 재빌드 없이 `source install/setup.bash`를 사용할 수 있습니다. 빌드하기 전에 아래의 `package.xml` 주의사항을 확인하세요.

### 4. 시뮬레이터(Publisher) 실행

```bash
ros2 run pv_edge pv_simulator
```

이 실행 명령은 **기존 README에 기록된 명령**입니다. 실제 실행 파일 이름과 설치 여부는 `pv_edge` 패키지 구현을 확인해야 합니다.

### 5. 다른 터미널에서 토픽 확인

같은 ROS2 환경에서 별도 터미널을 열어 아래 명령을 실행합니다.

```bash
source /opt/ros/humble/setup.bash
source /root/pv_ws_pub/install/setup.bash
export ROS_DOMAIN_ID=10

ros2 topic list
ros2 topic info /pv/sample
ros2 topic echo /pv/sample
```

사용자 정의 인터페이스의 정의는 다음 명령으로 확인할 수 있습니다.

```bash
ros2 interface show pv_interfaces/msg/PVSample
```

## 📊 활용 목적

- **태양광 데이터 스트림 시험:** 실제 센서 없이 ROS2 환경에서 데이터 발행 및 수신 절차 검증
- **AI 예측 노드 연동:** `PVForecastNode`의 입력 토픽 수신 시험
- **운전 상태 시나리오 검토:** `OK`, `TRIP`, `LOW_OUTPUT` 등 상태 정보 전달 구조 확인
- **ROS2 인터페이스 학습:** 사용자 정의 메시지와 Topic 기반 노드 간 데이터 전달 구조 이해

## 🔒 데이터 공개 및 준비 안내

기존 README에 따르면 **`SolarData_Sample_60min.csv`는 공개할 수 없는 원본 데이터**이며, GitHub에 업로드한 파일은 실제 수치가 없는 빈 파일입니다.

따라서 이 저장소만으로 원본 데이터에 의존하는 시뮬레이션 결과를 그대로 재현할 수는 없습니다. 실제 데이터 사용 권한이 있는 경우 별도로 데이터를 준비해야 하며, 파일의 위치와 읽기 방식은 `pv_edge` 구현에 맞춰 확인해야 합니다.

## ⚠️ 실행 시 주의사항

1. **패키지 의존성 확인:** 현재 제공된 `pv_interfaces/package.xml`에는 의존성 이름 뒤에 불필요한 `>` 문자가 포함되어 있습니다. 빌드 전에 `rosidl_default_generators`와 `rosidl_default_runtime` 이름을 올바르게 수정해야 합니다.
2. **실제 Publisher 구현 확인:** 이번에 전달된 파일에는 `pv_edge`의 Python 실행 파일과 패키지 설정이 없어, `ros2 run pv_edge pv_simulator`의 정상 실행 여부를 검증하지 못했습니다.
3. **토픽 타입 확인:** 기존 설명의 `std_msgs/msg/String`과 별도로 정의된 `pv_interfaces/msg/PVSample`은 서로 다른 타입입니다. 실제 노드와 구독 노드에서 동일한 토픽 타입을 사용해야 합니다.
4. **ROS Domain 통일:** Publisher와 Subscriber가 통신하려면 동일한 `ROS_DOMAIN_ID`와 호환되는 ROS2 통신 환경을 사용해야 합니다.
5. **데이터 보안:** 비공개 태양광 원본 데이터 및 식별 가능한 운영 정보는 저장소에 올리지 않도록 주의하세요.

---

본 프로젝트는 **ROS2 기반 태양광 데이터 발행 파이프라인 및 예측 노드 연동 시험**을 목적으로 합니다.
