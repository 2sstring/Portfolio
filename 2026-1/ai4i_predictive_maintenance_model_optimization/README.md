# 제조 설비 예지보전 AI 모델 성능 최적화 및 고도화

## 프로젝트 개요

본 프로젝트는 제조 설비의 센서 데이터를 활용하여 설비 고장 여부를 예측하는 AI 예지보전 모델을 개발하고 성능을 최적화하는 프로젝트입니다.

AI4I 2020 Predictive Maintenance Dataset을 사용하여 제조 설비의 공정 변수와 고장 여부 간의 관계를 분석합니다. 또한 다양한 AI 모델을 적용하여 고장 진단 성능을 비교합니다.

제조 고장 데이터는 정상 데이터에 비해 고장 데이터 수가 적은 클래스 불균형 특성을 가질 수 있습니다. 따라서 본 프로젝트에서는 단순 정확도뿐만 아니라 Precision, Recall, F1-score, ROC-AUC 등을 함께 평가합니다.

## 데이터셋 정보

본 저장소에는 원본 데이터셋을 포함하지 않습니다.

* 데이터셋명: AI4I 2020 Predictive Maintenance Dataset
* 다운로드 링크: https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020/data
* 사용 파일명: `ai4i2020.csv`

데이터셋을 다운로드한 후 아래와 같이 프로젝트 폴더에 배치합니다.

```text
03-ai4i-predictive-maintenance-model-optimization/
├── README.md
├── step1_eda.py
├── step2_data_prep.py
├── step3_train_model.py
├── step3_train_model_MLP.py
├── step3_train_model_XGB.py
├── step4_inference.py
├── step4_inference_MLP.py
├── step4_inference_XGB.py
└── ai4i2020.csv          # GitHub 업로드 제외
```

## 주요 내용

### 1. 데이터 탐색

* 데이터 크기 및 컬럼 구조를 확인합니다.
* 센서 변수 분포를 확인합니다.
* 고장 여부 클래스 분포를 확인합니다.
* 고장 유형별 데이터 분포를 확인합니다.
* feature 간 상관관계를 분석합니다.

### 2. 데이터 전처리

* 불필요한 식별자 컬럼을 제거합니다.
* 범주형 변수를 one-hot encoding합니다.
* 모델 학습에 부적절한 누출 가능 컬럼을 제거합니다.
* Train, Validation, Test 데이터를 분리합니다.
* 수치형 feature에 scaling을 적용합니다.

### 3. 모델 학습

본 프로젝트에서는 다음 모델을 비교합니다.

* 기본 PyTorch 기반 고장 진단 모델
* MLP 기반 고장 진단 모델
* XGBoost 기반 고장 진단 모델

### 4. 성능 최적화

* Validation 데이터 기준으로 threshold를 탐색합니다.
* F1-score 기반으로 최적 threshold를 선정합니다.
* 클래스 불균형을 고려하여 성능을 평가합니다.
* 모델별 confusion matrix를 비교합니다.
* ROC curve 및 ROC-AUC를 확인합니다.

### 5. 추론

* 학습된 모델과 scaler를 불러옵니다.
* 신규 센서 데이터에 대한 고장 확률을 예측합니다.
* 예측 확률과 threshold를 이용하여 정상/고장 여부를 판단합니다.

## 실행 방법

필요 라이브러리를 설치한 후 단계별로 실행합니다.

```bash
pip install pandas numpy scikit-learn torch xgboost matplotlib seaborn tensorboard joblib
python step1_eda.py
python step2_data_prep.py
python step3_train_model_MLP.py
python step3_train_model_XGB.py
```

추론 코드는 다음과 같이 실행합니다.

```bash
python step4_inference_MLP.py
python step4_inference_XGB.py
```

## 생성 파일

모델 학습 후 다음과 같은 파일이 생성될 수 있습니다.

```text
models/
├── fault_diagnosis.pth
├── fault_diagnosis_mlp.pth
├── fault_diagnosis_xgb.json
├── sensor_scaler.pkl
├── sensor_scaler_mlp.pkl
└── sensor_scaler_xgb.pkl
```

해당 파일들은 학습 결과물이므로 GitHub에는 업로드하지 않는 것을 권장합니다.

## 사용 기술

* Python
* pandas
* NumPy
* scikit-learn
* PyTorch
* XGBoost
* Matplotlib
* Seaborn
* TensorBoard
* joblib

## 정리

본 프로젝트는 제조 설비의 센서 데이터를 활용하여 고장 여부를 예측하는 예지보전 AI 모델을 개발하는 프로젝트입니다. 데이터 전처리, 모델 학습, threshold 최적화, 성능 평가, 추론 과정을 포함하며, 제조 현장에서 발생할 수 있는 고장 진단 문제를 AI 기반으로 해결하는 흐름을 구성합니다.
