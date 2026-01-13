# PV Data Publisher (Simulator)

이 워크스페이스는 **태양광 발전소 센서 데이터를 시뮬레이션하여 ROS2 토픽 `/pv/sample` 으로 Publish** 하는 역할을 합니다.  
AI 예측 노드(PVForecastNode)의 입력 데이터 소스로 사용됩니다.

---

## Workspace 구조

```
pv_ws_pub/
 ├── src/
 │    ├── pv_interfaces/      # 메시지 정의(PVSample.msg)
 │    └── pv_edge/            # 시뮬레이터 노드(pv_simulator_node.py)
 ├── install/ (자동 생성)
 ├── build/   (자동 생성)
 └── log/     (자동 생성)
```

---

## 실행 준비

### 1. 도커 컨테이너 실행

```bash
docker start influxdb
docker start <도커 컨테이너 이름>
```

---

## 실행 절차 (Publisher)

**터미널 1에서 아래를 실행합니다:**

```bash
docker exec -it <도커 컨테이너 이름> bash

# ROS2 기본 설정
source /opt/ros/humble/setup.bash

# 워크스페이스 이동
cd /root/pv_ws_pub

# 워크스페이스 환경 설정
source install/setup.bash

# ROS Domain 설정
export ROS_DOMAIN_ID=10

# 시뮬레이터 실행
ros2 run pv_edge pv_simulator 
```

---

## 발행되는 ROS 토픽

| 토픽 | 타입 | 설명 |
|------|------|------|
| `/pv/sample` | `std_msgs/String` | 태양광 센서·기상 정보 JSON 메시지 |

메시지 예시는 아래와 같습니다:

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

---

##  목적

- ROS2 기반의 실시간 PV 데이터 스트림 생성
- 예측 노드 테스트 및 고장 판정 검증
- 실제 센서 데이터 없이도 동일 구조로 실험 가능

---

> ⚠️ `SolarData_Sample_60min.csv`는 공개할 수 없는 데이터이며, 올린 파일은 빈파일입니다.

