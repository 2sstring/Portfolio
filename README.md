# AI-Ex 포트폴리오

**데이터 분석부터 머신러닝·딥러닝 모델 개발, 산업·에너지 분야의 응용 및 시스템 연동까지 수행한 프로젝트 모음**

2025년부터 2026년까지 진행한 AI·데이터 관련 프로젝트를 정리한 저장소입니다. 각 프로젝트에는 데이터 전처리, 분석·모델링 코드 및 개별 `README.md`를 담았으며, 프로젝트별 목적과 실행 방법은 각 폴더에서 확인할 수 있습니다.

## 📌 포트폴리오 소개

| 분야 | 주요 내용 |
|---|---|
| ☀️ 에너지·태양광 | 발전량 추정·예측, 시계열 특징 분석, 특징 선택, 발전 이상 패턴 탐지 |
| 🏭 산업 AI | 설비 고장 예측, 음향·이미지 이상 탐지, 반도체 공정 데이터 품질 관리 |
| 📊 데이터 분석 | 공기질·전력 소비·토지이용·네트워크 트래픽 데이터 분석 및 분류 |
| 🔗 시스템 연동 | ROS2 데이터 발행·예측 노드, InfluxDB·Grafana 모니터링, Flask 웹 애플리케이션 |

프로젝트는 **학기별 폴더(2025-1 ~ 2026-2)**로 구분되어 있습니다. 전체 **23개 프로젝트**를 아래 목록에서 확인할 수 있습니다.

## ⭐ 주요 프로젝트

| 프로젝트 | 핵심 내용 | 주요 기술 |
|---|---|---|
| [pv_power_estimation](2026-2/pv_power_estimation/) | SHAP·SFFS 기반 특징 선택과 태양광 발전량 추정 모델 비교 | Random Forest, LightGBM, LSTM, SHAP, SFFS |
| [pv_shap_topk_feature_analysis](2026-2/pv_shap_topk_feature_analysis/) | 모델별 SHAP Top-K 특징 선정 및 교차 적용 분석 | SHAP, Random Forest, LightGBM, LSTM |
| [pv_power_anomaly_detection](2026-2/pv_power_anomaly_detection/) | 일사량·발전량 관계에 따른 비정상 발전 패턴 탐지 및 시각화 | Python, 통계·규칙 기반 분석 |
| [predictive_maintenance](2025-2/predictive_maintenance/) | 설비 고장 예측과 모니터링 데이터 파이프라인 구성 | XGBoost, LightGBM, InfluxDB, Grafana, Docker |
| [pv_data_publisher](2025-2/pv_data_publisher/) · [pv_prediction](2025-2/pv_prediction/) | 태양광 데이터 시뮬레이션 발행과 예측·고장 판정 노드 연동 | ROS2, Python, LightGBM, SARIMAX |
| [mimii_sound_anomaly_detection_optimization](2026-1/mimii_sound_anomaly_detection_optimization/) | 설비 음향의 Mel·FFT 특징을 활용한 이상 탐지 모델 비교 | PyTorch, CNN Autoencoder, FFT |

## 📂 전체 프로젝트 목록

### 2025-1 · 기초 데이터 분석 및 모델링 (3개)

| 프로젝트 | 주요 내용 |
|---|---|
| [air_quality_prediction](2025-1/air_quality_prediction/) | 공기질 측정 데이터 분석 및 RNN·LSTM·GRU·1D CNN 기반 벤젠 농도 예측 |
| [bank_web_server](2025-1/bank_web_server/) | Flask를 이용한 로그인, 계좌 조회 및 입출금 웹 애플리케이션 |
| [electricity_demand_forecast](2025-1/electricity_demand_forecast/) | 가정용 전력 소비 시계열에 대한 ARMA·ARIMA 및 딥러닝 모델 비교 |

### 2025-2 · 응용 머신러닝 및 시스템 연동 (8개)

| 프로젝트 | 주요 내용 |
|---|---|
| [chungbuk_land_use_data_analysis](2025-2/chungbuk_land_use_data_analysis/) | 충북 토지이용·인구·도로 데이터의 통계 분석 및 시각화 |
| [chungbuk_land_use_classification](2025-2/chungbuk_land_use_classification/) | 지역별 토지이용 특징을 활용한 K-Means 군집 분석 |
| [predictive_maintenance](2025-2/predictive_maintenance/) | 설비 고장 예측 및 InfluxDB·Grafana 연계 모니터링 |
| [pv_correlation_analysis](2025-2/pv_correlation_analysis/) | 태양광 발전 데이터의 변수 간 상관관계 분석 |
| [pv_power_forecasting](2025-2/pv_power_forecasting/) | LightGBM·SARIMAX 및 가중 앙상블 기반 발전량 추정·예측 |
| [pv_data_publisher](2025-2/pv_data_publisher/) | ROS2 기반 태양광 발전 데이터 시뮬레이터 및 Publisher |
| [pv_prediction](2025-2/pv_prediction/) | ROS2 기반 발전량 예측·고장 판정 AI 노드 |
| [traffic_classification](2025-2/traffic_classification/) | 네트워크 트래픽의 서비스 유형 분류 및 모델 비교 |

### 2026-1 · 모델 고도화 및 데이터 품질 관리 (9개)

| 프로젝트 | 주요 내용 |
|---|---|
| [ai4i_predictive_maintenance_model_optimization](2026-1/ai4i_predictive_maintenance_model_optimization/) | 설비 고장 예측 모델 구조 및 판정 임계값 비교·최적화 |
| [mimii_sound_anomaly_detection_optimization](2026-1/mimii_sound_anomaly_detection_optimization/) | 산업 설비 음향 데이터의 Mel-Spectrogram·FFT 기반 이상 탐지 |
| [mvtec_ad_anomaly_detection_optimization](2026-1/mvtec_ad_anomaly_detection_optimization/) | 산업 제품 이미지의 Autoencoder 기반 이상 탐지 및 모델 비교 |
| [solar_power_data_quality_assurance](2026-1/solar_power_data_quality_assurance/) | 태양광 발전 데이터 품질 검증 및 전처리 |
| [solar_power_forecasting_reproduction](2026-1/solar_power_forecasting_reproduction/) | 태양광 발전량 예측 관련 논문 실험 재현 및 회귀 모델 비교 |
| [temporal_pv_forecasting](2026-1/temporal_pv_forecasting/) | Lag·Rolling·Diff 시간 특징을 활용한 발전량 예측 실험 |
| [uci_secom_data_quality_assurance](2026-1/uci_secom_data_quality_assurance/) | 반도체 제조 센서 데이터의 품질 검증 및 정제 |
| [using_opencv](2026-1/using_opencv/) | OpenCV를 활용한 영상 입출력·색상 변환·필터링 실습 |
| [uv_led_optical_output_compensation](2026-1/uv_led_optical_output_compensation/) | UV LED 광출력 예측과 온도 변화에 따른 ADC 보상량 탐색 |

### 2026-2 · 태양광 발전량 추정 및 특징 분석 (3개)

| 프로젝트 | 주요 내용 |
|---|---|
| [pv_power_anomaly_detection](2026-2/pv_power_anomaly_detection/) | 발전량·일사량 기반 이상 패턴 후보 자동 탐지 및 시각화 |
| [pv_power_estimation](2026-2/pv_power_estimation/) | SHAP·SFFS 기반 모델별 특징 선택 및 발전량 추정 성능 비교 |
| [pv_shap_topk_feature_analysis](2026-2/pv_shap_topk_feature_analysis/) | SHAP Top-K 특징 선택과 모델 간 교차 적용 성능 분석 |

## 🛠️ 주요 사용 기술

| 구분 | 기술 |
|---|---|
| 언어·데이터 처리 | Python, NumPy, pandas |
| 머신러닝·딥러닝 | scikit-learn, LightGBM, XGBoost, PyTorch |
| 모델 분석·최적화 | SHAP, SFFS, Optuna, 시계열 분석 |
| 시각화·영상 처리 | Matplotlib, OpenCV |
| 시스템·서비스 | ROS2, Flask, Docker, InfluxDB, Grafana |

## 📖 프로젝트 확인 및 실행 안내

각 프로젝트의 폴더에 있는 `README.md`에서 **데이터셋 출처, 필요한 라이브러리, 실행 순서, 주요 코드 및 출력 결과**를 확인할 수 있습니다. 프로젝트별로 사용 데이터와 실행 환경이 다르므로, 원하는 프로젝트의 README를 먼저 확인한 후 실행하는 것을 권장합니다.

일부 프로젝트는 공개 데이터셋을 별도로 다운로드해야 하며, 비공개 측정 데이터를 활용한 프로젝트는 실제 원본 데이터가 저장소에 포함되어 있지 않을 수 있습니다. 따라서 저장소의 코드와 문서를 중심으로 프로젝트의 구성과 분석 방법을 확인해 주세요.

---

*AI-Ex 포트폴리오 · 2025–2026*
