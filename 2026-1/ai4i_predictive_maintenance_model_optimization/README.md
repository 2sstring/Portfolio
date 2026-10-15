# 🏭 ai4i_predictive_maintenance_model_optimization

## 📌 프로젝트 소개

**ai4i_predictive_maintenance_model_optimization**은 **AI4I 2020 Predictive Maintenance Dataset**의 제조 설비 센서 데이터를 활용하여 **설비 고장 여부를 예측하고, 모델 구조와 판단 임계값(Threshold)을 개선하는 머신러닝·딥러닝 프로젝트**입니다.

기초 탐색적 데이터 분석(EDA)부터 전처리, **기본 MLP·개선 MLP·XGBoost 모델 학습**, 검증 데이터 기반 임계값 선택, 테스트 성능 평가, 새로운 센서 입력에 대한 추론 예시까지 단계별로 구현했습니다.

> **분류 목표:** 설비 상태를 `정상(0)`과 `고장(1)`으로 구분하는 **이진 분류(Binary Classification)**입니다. 고장 유형별 다중 분류를 수행하는 프로젝트는 아닙니다.

---

## 📁 프로젝트 폴더 구조

```text
ai4i_predictive_maintenance_model_optimization/
├── README.md                    # 프로젝트 설명
├── step1_eda.py                 # 데이터 탐색 및 시각화
├── step2_data_prep.py           # 데이터 전처리 및 분할
├── step3_train_model.py         # 기본 MLP 학습 및 평가
├── step3_train_model_MLP.py     # 개선 MLP 학습 및 임계값 최적화
├── step3_train_model_XGB.py     # XGBoost 학습 및 임계값 최적화
├── step4_inference.py           # 기본 MLP 추론 예시
├── step4_inference_MLP.py       # 개선 MLP 추론 예시
├── step4_inference_XGB.py       # XGBoost 추론 예시
├── ai4i2020.csv                 # 원본 데이터 (별도 다운로드)
├── models/                      # 학습 후 모델·스케일러 저장 (자동 생성)
└── runs/                        # TensorBoard 로그 (자동 생성)
```

`ai4i2020.csv`는 저장소에 포함하지 않았으며, `models/`와 `runs/`는 관련 코드를 실행하면 생성됩니다. 위 Python 파일명은 코드에서 참조하는 **프로젝트 내 원래 파일명**을 기준으로 표시했습니다.

### 🗂️ 주요 파일 설명

| 파일 | 역할 |
|---|---|
| `step1_eda.py` | 센서 변수의 분포, 상관관계, 정상/고장 비율 및 고장 유형별 발생 현황 확인 |
| `step2_data_prep.py` | 식별자·누수 가능 변수 제거, 원-핫 인코딩, 데이터 분할, 표준화 및 DataLoader 구성 |
| `step3_train_model.py` | 기본 MLP 모델을 학습하고 테스트 데이터로 성능 평가 |
| `step3_train_model_MLP.py` | 확장된 MLP를 학습하고 검증 F1-score 기반 임계값 탐색 |
| `step3_train_model_XGB.py` | XGBoost의 설정값을 교차검증한 후 모델 학습·평가 및 임계값 탐색 |
| `step4_inference.py` | 기본 MLP 모델·스케일러를 불러와 예시 센서 데이터 분류 |
| `step4_inference_MLP.py` | 개선 MLP를 이용한 단일 센서 입력 추론 |
| `step4_inference_XGB.py` | XGBoost를 이용한 단일 센서 입력 추론 |

---

## 🔗 데이터셋 및 다운로드 안내

본 프로젝트는 **AI4I 2020 Predictive Maintenance Dataset**을 사용합니다.

- **UCI 데이터 페이지:** [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- **Kaggle 데이터 페이지:** [Predictive Maintenance Dataset (AI4I 2020)](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020/data)
- **사용 파일명:** `ai4i2020.csv`

### 📥 다운로드 및 준비 방법

1. 위 데이터 페이지에서 **AI4I 2020** 데이터를 다운로드합니다.
2. 압축 파일이라면 압축을 해제합니다.
3. `ai4i2020.csv`를 프로젝트의 **최상위 폴더**에 배치합니다.
4. `step1_eda.py`를 실행해 데이터가 정상적으로 로드되는지 확인합니다.

> ⚠️ **데이터 로딩 방식이 다릅니다.** `step1_eda.py`는 프로젝트 안의 로컬 `ai4i2020.csv`를 읽는 반면, `step2_data_prep.py`는 코드에 지정된 **UCI 다운로드 URL에서 CSV를 직접 읽습니다.** 인터넷이 없는 환경에서 학습하려면 `step2_data_prep.py`의 `pd.read_csv(url)`을 `pd.read_csv('ai4i2020.csv')`로 변경해야 합니다.

### 📊 주요 데이터 항목

| 변수 | 설명 |
|---|---|
| `Type` | 제품 등급(L, M, H) |
| `Air temperature [K]` | 주변 공기 온도 |
| `Process temperature [K]` | 공정 온도 |
| `Rotational speed [rpm]` | 회전 속도 |
| `Torque [Nm]` | 토크 |
| `Tool wear [min]` | 공구 마모 시간 |
| `Machine failure` | 예측 대상: 정상 0 / 고장 1 |
| `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | 고장 유형을 나타내는 지표. 고장 여부 학습 입력에서는 제외 |

최종 학습 입력은 **연속형 센서 변수 5개 + `Type`의 원-핫 인코딩 변수 2개 = 총 7개 특징**으로 구성됩니다.

---

## ⚙️ 데이터 분석 및 전처리

### 1. 탐색적 데이터 분석(EDA)

`step1_eda.py`에서 다음 사항을 확인합니다.

- 데이터 크기, 컬럼 구조 및 결측값·자료형
- 온도·회전속도·토크·공구 마모 등 주요 센서 변수의 분포
- 센서 변수 간 Pearson 상관계수와 히트맵
- 정상/고장 클래스 비율 및 세부 고장 유형별 발생 건수

### 2. 데이터 전처리

`step2_data_prep.py`의 전처리 구성은 다음과 같습니다.

| 단계 | 처리 내용 |
|---|---|
| 식별자 제거 | `UDI`, `Product ID` 삭제 |
| 범주형 처리 | `Type`에 `One-Hot Encoding` 적용 (`drop_first=True`) |
| 데이터 누수 방지 | 고장 여부를 직접 나타내는 `TWF`, `HDF`, `PWF`, `OSF`, `RNF` 입력에서 제외 |
| 데이터 분할 | Train 60% / Validation 20% / Test 20% (클래스 비율 유지) |
| 스케일링 | 5개 연속형 센서 변수에 `StandardScaler` 적용. 학습 세트에만 `fit` |
| 클래스 불균형 처리 | PyTorch 학습용 `WeightedRandomSampler`로 정상·고장 샘플의 추출 빈도 조정 |
| 데이터 로더 | 배치 크기 64, 검증·테스트는 원래의 클래스 분포 유지 |

학습 스크립트들은 `step2_data_prep`에서 전처리 변수들을 `import`하므로, 해당 파일을 **먼저 단독 실행하는 것이 필수는 아닙니다.** 학습 코드를 실행할 때 전처리도 함께 수행됩니다.

---

## 🧠 예측 모델 및 최적화 구성

| 모델 | 학습 파일 | 핵심 구조 및 설정 | 판정 임계값 |
|---|---|---|---|
| 기본 MLP | `step3_train_model.py` | 은닉층 `32 → 16`, BatchNorm·Dropout, 30 Epochs | 0.50 |
| 개선 MLP | `step3_train_model_MLP.py` | 은닉층 `128 → 64 → 32`, BatchNorm·Dropout, 130 Epochs | 검증 F1 최대화로 탐색 |
| XGBoost | `step3_train_model_XGB.py` | `XGBClassifier` 교차검증 및 `xgb.train`, 900 boosting rounds | 검증 F1 최대화로 탐색 |

### 🔹 기본 MLP

PyTorch 기반 `BCEWithLogitsLoss`와 Adam 최적화 알고리즘을 사용해 설비 고장 확률을 학습합니다. 기본 구현은 판단 임계값을 **0.5**로 사용합니다.

### 🔹 개선 MLP

은닉층의 크기와 학습 Epoch를 확장하고, **Validation 데이터의 F1-score를 가장 크게 하는 임계값**을 `0.00`부터 `1.00`까지 `0.01` 간격으로 탐색합니다. 이후 선택된 임계값을 적용하여 테스트 세트 성능을 계산합니다.

### 🔹 XGBoost

`StratifiedKFold` 5분할과 `GridSearchCV`를 사용합니다. 다만 **현재 `param_grid`는 각 하이퍼파라미터에 값이 하나씩만 지정되어 있으므로 여러 조합을 폭넓게 탐색하는 Grid Search는 아닙니다.** 지정된 조합을 교차검증한 후 `xgb.train`으로 학습하고 검증 F1-score 기반 임계값을 선택합니다.

> 학습 결과에는 **Accuracy, Precision, Recall, F1-score, ROC-AUC**가 출력됩니다. 실제 실험 결과 수치는 환경과 실행에 따라 달라지므로 README에 임의로 기재하지 않았습니다.

---

## 📈 모델 평가 및 시각화

학습 스크립트에는 다음 결과를 확인하는 기능이 있습니다.

| 항목 | 내용 |
|---|---|
| Accuracy | 전체 샘플 중 올바르게 분류한 비율 |
| Precision | 고장으로 예측한 샘플 중 실제 고장 비율 |
| Recall | 실제 고장 중 모델이 검출한 비율 |
| F1-score | Precision과 Recall의 조화평균 |
| ROC-AUC | 다양한 임계값에서 고장/정상 구분 능력 평가 |
| Confusion Matrix | 정상·고장의 정분류 및 오분류 현황 시각화 |
| ROC Curve | FPR과 TPR 관계 시각화 |
| TensorBoard | 학습·검증 Loss 및 Validation F1-score 기록 |

특히 고장 데이터는 정상 데이터보다 적을 수 있으므로, **Accuracy만이 아니라 고장 검출과 관련된 Recall 및 F1-score를 함께 확인**하도록 구성했습니다.

---

## 💻 실행 환경 및 설치

**주요 사용 기술:** Python, pandas, NumPy, scikit-learn, PyTorch, XGBoost, Matplotlib, Seaborn, TensorBoard, joblib

프로젝트 폴더에서 다음 명령을 실행합니다.

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

필요 패키지 설치:

```bash
pip install pandas numpy scikit-learn torch xgboost matplotlib seaborn tensorboard joblib
```

### ▶️ 단계별 실행 방법

**1. 데이터 탐색**

```bash
python step1_eda.py
```

**2. 전처리 점검 (선택)**

```bash
python step2_data_prep.py
```

**3. 모델 학습 — 원하는 모델을 선택하거나 각각 실행**

```bash
# 기본 MLP
python step3_train_model.py

# 개선 MLP
python step3_train_model_MLP.py

# XGBoost
python step3_train_model_XGB.py
```

**4. 저장한 모델로 예시 센서 데이터 추론**

```bash
# 기본 MLP 결과 사용
python step4_inference.py

# 개선 MLP 결과 사용
python step4_inference_MLP.py

# XGBoost 결과 사용
python step4_inference_XGB.py
```

학습을 완료하지 않은 상태에서 추론 코드를 먼저 실행하면 모델·스케일러 파일이 없다는 오류가 발생할 수 있습니다.

**5. TensorBoard 확인 (선택)**

```bash
tensorboard --logdir runs
```

안내되는 로컬 주소로 접속하여 학습 과정을 확인합니다.

---

## 📦 학습 후 생성 파일

```text
models/
├── fault_diagnosis.pth          # 기본 MLP 가중치
├── sensor_scaler.pkl            # 기본 MLP용 전처리 스케일러
├── fault_diagnosis_mlp.pth      # 개선 MLP 가중치
├── sensor_scaler_mlp.pkl        # 개선 MLP용 스케일러
├── fault_diagnosis_xgb.json     # XGBoost 모델
└── sensor_scaler_xgb.pkl        # XGBoost용 스케일러

runs/
├── fault_diagnosis_experiment/      # 기본 MLP 로그
├── fault_diagnosis_experiment_mlp/  # 개선 MLP 로그
└── fault_diagnosis_experiment_xgb/  # XGBoost 로그
```

모델 파일은 해당 학습 코드를 실행한 경우에만 생성되며, `.pth` 및 `.json` 모델과 스케일러 `.pkl` 파일의 조합을 일치시켜 사용해야 합니다.

---

## 🧪 추론 예시

각 추론 스크립트는 현장 센서 신호 수집을 직접 연결하는 대신, **한 건의 예시 입력 데이터를 코드에 지정하여 단일 샘플 분류를 시뮬레이션**합니다.

```python
incoming_data = {
    'Type': 'H',
    'Air temperature [K]': 302.5,
    'Process temperature [K]': 311.2,
    'Rotational speed [rpm]': 1350,
    'Torque [Nm]': 70.0,
    'Tool wear [min]': 215
}
```

신규 데이터를 학습 시와 동일한 7개 특징으로 변환하고 스케일링한 뒤 **고장 확률과 정상/고장 판정**을 출력합니다.

| 추론 코드 | 연동 모델 | 코드에 지정된 임계값 |
|---|---|---:|
| `step4_inference.py` | 기본 MLP | 0.50 |
| `step4_inference_MLP.py` | 개선 MLP | 0.93 |
| `step4_inference_XGB.py` | XGBoost | 0.74 |

`0.93`과 `0.74`는 **추론 스크립트에 고정 기입된 값**입니다. 학습 때 검증으로 계산되는 최적 임계값이 자동 저장·전달되는 구조는 아니므로, 다시 학습한 경우 실제 출력된 최적값과 맞는지 확인해야 합니다.

---

## ⚠️ 실행 및 해석 시 주의사항

1. **로컬 데이터와 온라인 데이터의 차이:** EDA는 `ai4i2020.csv`를 로컬에서 읽지만, 전처리는 UCI 주소에서 직접 다운로드합니다. 사용할 데이터의 버전과 경로를 일치시키세요.
2. **기본 MLP의 평가 방식:** `step3_train_model.py`는 학습 진행 중 `test_loader`를 Validation처럼 사용한 뒤 같은 데이터에서 최종 성능을 계산합니다. 엄밀한 독립 테스트 평가가 필요하면 `val_loader`로 바꾸는 것이 좋습니다.
3. **MLP의 그래프 기록:** `step3_train_model.py`와 `step3_train_model_MLP.py`의 `writer.add_graph()`에 전달하는 더미 입력은 배치 크기 1인데, 모델에 `BatchNorm1d`가 포함돼 있어 오류가 발생할 수 있습니다. 예를 들어 더미 입력을 `torch.randn(2, input_dim)`으로 변경하거나 그래프 기록을 생략할 수 있습니다.
4. **Grid Search 탐색 범위:** XGBoost의 현재 파라미터 그리드는 값이 한 조합뿐입니다. 모델 최적화 범위를 넓히려면 각 파라미터의 후보 값을 추가해야 합니다.
5. **모델과 전처리의 일치:** 학습·추론의 입력 특징 순서, 범주형 인코딩, 스케일러, 모델 아키텍처가 동일해야 합니다. 각 모델에 맞는 추론 스크립트를 사용하세요.
6. **실시간 시스템과의 구분:** 현재 추론 예시는 코드에 입력된 가상 센서값으로 판정하는 방식으로, 실제 PLC·OPC-UA와 연동한 실시간 수집 기능은 구현되어 있지 않습니다.
7. **판정 기준의 한계:** 이진 분류 결과는 데이터 기반 고장 가능성 추정이며, 실제 산업 설비의 안전 관련 제어 또는 정비 판단에 단독으로 사용하기 전에 별도 검증이 필요합니다.

---

## 🎯 프로젝트 목적

본 프로젝트는 AI4I 제조 설비 데이터를 바탕으로 **센서 특징 분석 → 전처리 및 클래스 불균형 대응 → 모델별 학습 → 검증 기반 임계값 조정 → 성능 평가 → 신규 데이터 추론**까지의 전체 과정을 구현합니다. 기본 MLP, 개선 MLP, XGBoost를 비교하고 설비 고장 검출 성능의 개선 가능성을 검토하는 데 목적이 있습니다.
