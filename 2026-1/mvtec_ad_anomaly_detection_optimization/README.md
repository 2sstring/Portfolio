# MVTec AD 모델 성능 최적화 및 고도화

## 프로젝트 개요

본 프로젝트는 이미지 기반 산업 제품 이상 탐지를 위한 Autoencoder 모델을 개발하고 성능을 평가하는 프로젝트입니다.

MVTec AD 데이터셋을 활용하여 정상 제품 이미지만으로 Autoencoder를 학습하고, 재구성 오차를 이용하여 비정상 제품을 탐지합니다.

본 프로젝트에서는 `bottle` 카테고리를 기준으로 정상 이미지와 불량 이미지를 비교하고, 재구성 오차 기반 anomaly score를 산출합니다.

## 데이터셋 정보

본 저장소에는 원본 이미지 데이터셋을 포함하지 않습니다.

* 데이터셋명: MVTec AD
* 다운로드 링크: https://www.mvtec.com/research-teaching/datasets/mvtec-ad
* 사용 카테고리: `bottle`
* 사용 데이터 경로: `mvtec_ad/bottle/`

데이터셋을 다운로드한 후 아래와 같이 프로젝트 폴더에 배치합니다.

```text
04-mvtec-ad-anomaly-detection-optimization/
├── README.md
├── step1_data_eda.py
├── step2_train.py
├── step2_train_cnn.py
├── step3_evaluate.py
└── mvtec_ad/                 # GitHub 업로드 제외
    └── bottle/
        ├── train/
        │   └── good/
        ├── test/
        │   ├── good/
        │   ├── broken_large/
        │   ├── broken_small/
        │   └── contamination/
        └── ground_truth/
```

## 주요 내용

### 1. 데이터 탐색

* 정상 이미지와 불량 이미지 구조를 확인합니다.
* 학습 데이터와 테스트 데이터 구성을 확인합니다.
* 이미지 크기 및 샘플 이미지를 확인합니다.

### 2. Autoencoder 모델 학습

* 정상 이미지 기반 Autoencoder를 학습합니다.
* CNN 기반 Autoencoder 구조를 적용합니다.
* 입력 이미지를 재구성하도록 모델을 학습합니다.
* 학습 loss를 통해 모델 수렴 상태를 확인합니다.

### 3. 이상 탐지 평가

* 테스트 이미지를 입력합니다.
* 원본 이미지와 재구성 이미지를 비교합니다.
* 재구성 오차를 계산합니다.
* 재구성 오차 map을 생성합니다.
* anomaly score를 산출합니다.
* 정상/비정상 분류 성능을 평가합니다.

### 4. 시각화

* 원본 이미지
* 재구성 이미지
* 재구성 오차 map
* heatmap overlay
* 정상 및 불량 샘플 비교

## 실행 방법

필요 라이브러리를 설치한 후 단계별로 실행합니다.

```bash
pip install torch torchvision numpy opencv-python scikit-learn matplotlib pillow
python step1_data_eda.py
python step2_train_cnn.py
python step3_evaluate.py
```

## 생성 파일

학습 및 평가 후 다음과 같은 파일이 생성됩니다.

```text
autoencoder_model.pth
autoencoder_model_skipless.pth
runs/
```

## 사용 기술

* Python
* PyTorch
* torchvision
* OpenCV
* NumPy
* scikit-learn
* Matplotlib
* Pillow

## 정리

본 프로젝트는 정상 이미지만을 이용해 Autoencoder를 학습한 후, 재구성 오차를 기반으로 산업 제품의 이상 여부를 탐지하는 프로젝트입니다. 이미지 기반 제조 품질 검사 문제에서 비지도학습 기반 이상 탐지 모델을 적용하고, anomaly score와 시각화 결과를 통해 모델 성능을 확인합니다.
