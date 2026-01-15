# pv_ws - PV Forecasting + Fault Detection (AI Node)

이 워크스페이스는 **태양광 발전량 예측(LightGBM + SARIMAX 앙상블 모델)** 과  
**고장 상태 판정(FAULT_OVER / FAULT_UNDER / NIGHT / UNDER_IRR 등)** 을 수행하는 ROS2 노드를 포함합니다.

`pv_ws_pub` 에서 Publish하는 `/pv/sample` 을 Subscribe하여 예측을 수행합니다.

---

## Workspace 구조

```
pv_ws/
 ├── src/
 │    └── pv_ai/
 │         ├── pv_forecast_node.py   # 메인 예측/고장판정 노드
 │         ├── pv_sub_print.py         # sample 출력 유틸
 │         └── launch/
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

## 실행 절차 (Forecasting Node)

**터미널 2에서 아래를 실행하세요:**

```bash
docker exec -it <도커 컨테이너 이름> bash

# ROS2 설정
source /opt/ros/humble/setup.bash

# 워크스페이스 이동
cd /root/pv_ws

# 환경 설정
source install/setup.bash

# ROS Domain 지정
export ROS_DOMAIN_ID=10

# 예측 노드 실행
ros2 run pv_ai pv_forecast_node
```

---

## 모델 파일 요구사항

이 워크스페이스는 `/root/pv_models/` 에 다음 파일들이 있어야 합니다:

```
lgbm_model.pkl
sarimax_model.pkl
ensemble_weights.json
feature_cols.json
```

이들은 학습된 모델 및 피처 구성을 담고 있으며 예측에 필수적입니다.

---

## ⚙ 수행 기능 요약

- 버퍼(FeatureBuffer)를 이용한 recent time window 구성
- LightGBM + SARIMAX 앙상블 예측
- 낮/밤 구분을 통한 서로 다른 에러기준 적용
- 다음 상태 분류 수행:
  - `NORMAL`
  - `FAULT_OVER`
  - `FAULT_UNDER`
  - `FAULT_UNDER_IRR`
  - `NIGHT`
  - `NO_ACTUAL`
- InfluxDB에 상세 로그 기록 (예측값, 실제값, 오차, 고도, 기상 값 등)

---

## ✔ 목적

- AI 기반 태양광 발전량 예측 및 이상 탐지
- 실시간 스트리밍 데이터를 이용한 온라인 추론
- 산업용 PV 모니터링/진단 시스템 프로토타입

---

