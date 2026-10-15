# 🏭 mvtec_ad_anomaly_detection_optimization

**MVTec AD 산업 제품 이미지 기반 이상 탐지 및 Autoencoder 모델 성능 개선 프로젝트**

본 프로젝트는 산업 제품의 **정상 이미지로 합성곱 오토인코더(Convolutional Autoencoder)를 학습**하고, 입력 이미지와 재구성 이미지 사이의 오차를 이용해 제품의 이상 여부를 판단합니다. MVTec AD 데이터셋의 **`bottle` 카테고리**를 대상으로 기본 CNN Autoencoder와 구조를 개선한 **Skipless Autoencoder**를 구현하고, 복원 오차 기반 이상 점수와 시각화 결과를 분석합니다.

> ⚠️ **데이터 안내:** MVTec AD 원본 이미지는 이 저장소에 포함하지 않습니다. 공식 제공처에서 데이터셋을 내려받아 아래 폴더 구조로 준비해야 합니다.

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 분석 대상 | MVTec AD 산업 이미지 데이터셋의 `bottle` 카테고리 |
| 입력 데이터 | 정상(`train/good`) 이미지 및 정상·불량(`test`) 이미지 |
| 학습 방식 | 정상 이미지만 사용하는 재구성 기반 비지도 이상 탐지 |
| 예측 모델 | 기본 ConvAutoencoder, SkiplessAutoencoder |
| 입력 이미지 크기 | RGB `3 × 256 × 256` |
| 이상 점수 | 재구성 오차 맵에 Gaussian Blur를 적용한 후 **최댓값(Max)** 사용 |
| 성능 평가 | 이미지 단위 AUROC, F1-score, Precision–Recall 기반 임계값 |
| 시각화 | 원본 이미지, 재구성 이미지, 오차 맵, 오버레이 히트맵 |
| 활용 목적 | 제조 제품의 불량 탐지 및 비전 기반 품질 검사 모델 실험 |

## 📁 프로젝트 폴더 구조

```text
mvtec_ad_anomaly_detection_optimization/
├── README.md                       # 프로젝트 안내
├── step1_data_eda.py               # 데이터 로더 및 이미지 EDA
├── step2_train.py                  # 기본 CNN Autoencoder 학습
├── step2_train_cnn.py              # Skipless Autoencoder 학습·검증
├── step3_evaluate.py               # 테스트 평가 및 이상 영역 시각화
├── mvtec_ad/                       # 원본 데이터 (별도 다운로드)
│   └── bottle/
│       ├── train/
│       │   └── good/               # 정상 학습 이미지
│       ├── test/
│       │   ├── good/               # 정상 테스트 이미지
│       │   ├── broken_large/       # 불량 유형 예시
│       │   ├── broken_small/
│       │   └── contamination/
│       └── ground_truth/           # 데이터셋의 결함 위치 마스크
├── autoencoder_model.pth           # 기본 모델 학습 후 생성
├── autoencoder_model_skipless.pth  # 개선 모델 학습 후 생성
└── runs/                           # TensorBoard 로그 (개선 모델 학습 후 생성)
```

`mvtec_ad/`는 사용자가 별도로 준비해야 하는 데이터 폴더입니다. 두 `.pth` 파일과 `runs/`는 실행 후 생성되는 산출물이므로 최초 저장소에 없어도 됩니다. `ground_truth/` 마스크는 원본 데이터셋 구성요소이며, **현재 평가 코드는 이를 이용한 픽셀 단위 성능 지표를 계산하지 않습니다.**

### 🗂️ 주요 파일 설명

| 파일 | 역할 |
|---|---|
| `step1_data_eda.py` | `MVTecDataset` 클래스 정의, 정상/불량 이미지 로드와 샘플 시각화 |
| `step2_train.py` | 기본 합성곱 오토인코더를 정상 이미지로 학습하고 가중치 저장 |
| `step2_train_cnn.py` | Skip 연결이 없는 개선형 오토인코더를 학습/검증하고 최적 검증 손실 모델 저장 |
| `step3_evaluate.py` | 복원 오차 기반 이미지별 점수, AUROC 및 F1 임계값 계산, 결함 히트맵 시각화 |

---

## 🔗 데이터셋 및 다운로드 안내

- **데이터셋:** MVTec Anomaly Detection (MVTec AD)
- **공식 다운로드:** [MVTec AD Dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- **프로젝트에서 사용하는 카테고리:** `bottle`
- **코드의 기본 데이터 경로:** `./mvtec_ad/bottle/`

### 📥 다운로드 및 준비 방법

1. 위 공식 페이지에서 MVTec AD 데이터셋의 제공 조건을 확인하고 데이터를 다운로드합니다.
2. 다운로드한 파일의 압축을 해제합니다.
3. 프로젝트 폴더 아래에 `mvtec_ad/bottle/` 구조가 만들어지도록 배치합니다.
4. `bottle/train/good/`와 `bottle/test/`의 각 하위 폴더에 **`.png` 이미지가 존재하는지** 확인합니다.
5. 아래 실행 명령은 **프로젝트 최상위 폴더**에서 실행합니다.

> 데이터가 다른 폴더에 있다면 각 스크립트 하단의 `ROOT_DIR = './mvtec_ad'`와 `CATEGORY = 'bottle'` 설정을 수정하세요. 데이터셋의 라이선스 및 사용 조건은 공식 제공처를 따릅니다.

---

## 🔍 데이터 탐색 및 전처리

`step1_data_eda.py`의 `MVTecDataset`은 학습 시 `train/good` 폴더의 이미지를 모두 **정상(0)** 으로 읽습니다. 테스트 시에는 `test/good`을 **정상(0)**, 그 밖의 결함 유형 폴더를 **이상(1)** 으로 처리합니다.

전처리는 `torchvision.transforms`를 사용하며, 모든 모델이 이미지를 `256 × 256`으로 조정한 후 텐서로 변환합니다. 개선 모델의 학습 데이터에만 **50% 확률의 좌우 반전(RandomHorizontalFlip)** 이 적용되며, 검증 및 테스트에는 적용되지 않습니다.

```bash
python step1_data_eda.py
```

이 스크립트는 테스트 데이터에서 일부 샘플을 선택하여 원본 이미지와 정상·이상 라벨을 화면에 표시합니다.

---

## 🧠 Autoencoder 모델 학습

두 모델 모두 **입력 이미지를 다시 복원**하도록 학습하며, 재구성 오차가 큰 이미지를 이상 후보로 판단합니다.

| 구분 | 기본 CNN Autoencoder | Skipless Autoencoder |
|---|---|---|
| 학습 코드 | `step2_train.py` | `step2_train_cnn.py` |
| 구조 | Conv2d 인코더 + ConvTranspose2d 디코더 | 다단계 합성곱 인코더, Bottleneck, 디코더 (**Skip 연결 없음**) |
| 입력 크기 | `3 × 256 × 256` | `3 × 256 × 256` |
| 손실 함수 | MSELoss | MSELoss |
| 최적화 알고리즘 | Adam (`1e-3`) | Adam (`1e-3`) |
| Epoch 설정 | 50 | 최대 1,000 |
| 검증 데이터 | 별도 검증 절차 없음 | 정상 학습 데이터에서 20% 분리 (`seed=42`) |
| 학습 제어 | 고정 Epoch | ReduceLROnPlateau 학습률 조절 및 최저 검증 손실 가중치 저장 |
| 학습 기록 | 콘솔 손실 출력 | TensorBoard 손실·학습률·입력/복원 이미지 기록 |
| 모델 저장 | `autoencoder_model.pth` | `autoencoder_model_skipless.pth` |

### 기본 모델 학습

```bash
python step2_train.py
```

### 개선 모델 학습

```bash
python step2_train_cnn.py
```

개선 모델은 훈련/검증 데이터를 분리하고, 검증 손실이 가장 낮아질 때의 모델을 `autoencoder_model_skipless.pth`에 저장합니다. **1,000 Epoch는 최대 반복 설정**이며, 현재 코드에는 조기 종료(Early Stopping)가 구현되어 있지 않습니다.

TensorBoard 학습 기록을 확인하려면 다음 명령을 사용합니다.

```bash
tensorboard --logdir runs
```

---

## 📊 이상 탐지 및 성능 평가

`step3_evaluate.py`는 기본 설정에서 **Skipless Autoencoder**와 `autoencoder_model_skipless.pth`를 불러옵니다. 모델이 입력 이미지를 재구성한 후 다음 과정으로 이상 점수를 계산합니다.

1. 입력과 복원 이미지의 **채널 평균 제곱오차(MSE)**를 픽셀별로 계산합니다.
2. 픽셀 오차 맵에 `15 × 15` Gaussian Blur를 적용합니다.
3. 오차 맵의 **최댓값**을 해당 이미지의 이상 점수로 사용합니다.
4. 테스트 이미지 전체를 대상으로 **이미지 단위 AUROC**를 계산합니다.
5. Precision–Recall 값에서 F1-score가 높은 분류 임계값을 탐색합니다.
6. 결함 샘플에 원본·복원·오차 맵·히트맵 오버레이를 나란히 표시합니다.

```bash
python step3_evaluate.py
```

### 평가 시각화 구성

| 시각화 | 의미 |
|---|---|
| Original | 입력 원본 이미지 및 이상 점수·판정 결과 |
| Reconstructed | 오토인코더가 복원한 이미지 |
| Error Map | 픽셀별 복원 오차 분포 |
| Overlay Heatmap | 원본 이미지 위에 이상 점수 분포를 중첩한 결과 |

평가 스크립트에서 기본 모델을 비교하려면 `MODEL_PATH`를 `autoencoder_model.pth`로 변경하고, 생성하는 모델 클래스도 `ConvAutoencoder`로 함께 변경해야 합니다. 모델 구조와 저장된 가중치가 서로 같아야 합니다.

> **평가 해석:** 현재 코드는 테스트 이미지의 실제 라벨을 사용해 최적 임계값을 선택하고 같은 테스트 이미지에서 F1-score를 계산합니다. 따라서 해당 F1-score는 **독립적인 테스트 성능으로 해석하기 어렵습니다.** 최종적인 성능 비교를 위해서는 별도의 검증 데이터에서 임계값을 결정하고 독립 테스트 데이터에 고정 적용하는 방식이 권장됩니다.

---

## ⚙️ 실행 환경 및 실행 순서

### 1. 필수 라이브러리 설치

Python 환경에서 아래 패키지를 설치합니다.

```bash
pip install torch torchvision numpy opencv-python scikit-learn matplotlib pillow tensorboard
```

PyTorch 설치 방식은 CPU/GPU 환경에 따라 달라질 수 있습니다. GPU가 없다면 두 학습 코드는 CPU로 실행됩니다.

### 2. 전체 실행 순서

```bash
# 1. MVTec AD 이미지 탐색
python step1_data_eda.py

# 2. 기본 모델 학습 (선택)
python step2_train.py

# 3. 개선 모델 학습
python step2_train_cnn.py

# 4. 개선 모델 성능 평가 및 이상 영역 시각화
python step3_evaluate.py
```

평가 전에 `step2_train_cnn.py`를 먼저 실행하여 `autoencoder_model_skipless.pth` 파일을 생성해야 합니다.

---

## 📦 생성 파일 안내

| 결과 파일 또는 폴더 | 생성 단계 | 설명 |
|---|---|---|
| `autoencoder_model.pth` | 기본 모델 학습 | 기본 ConvAutoencoder의 모델 가중치 |
| `autoencoder_model_skipless.pth` | 개선 모델 학습 | 최저 검증 손실 기준 SkiplessAutoencoder의 가중치 |
| `runs/bottle_skipless_autoencoder/` | 개선 모델 학습 | TensorBoard 학습 손실, 학습률 및 이미지 기록 |

평가 결과 그래프는 현재 코드에서 `plt.show()`로 출력하며, **PNG 파일로 자동 저장하지는 않습니다.** 모델 성능 수치 또한 실제 실행 결과가 제공되지 않아 이 문서에 임의로 기재하지 않았습니다.

---

## ⚠️ 실행 및 실험 시 주의사항

- **데이터 별도 준비:** MVTec AD 원본 이미지가 없으면 실행할 수 없습니다. 특히 테스트 폴더에는 정상과 결함 이미지가 모두 필요합니다.
- **실행 경로:** 상대경로를 사용하므로 Python 스크립트는 프로젝트 최상위 폴더에서 실행합니다.
- **평가 모델 일치:** 기본 평가 코드는 Skipless 모델을 로드합니다. 기본 CNN 모델을 평가하려면 클래스와 가중치 경로를 함께 바꿔야 합니다.
- **모델 파일 로드 환경:** GPU에서 저장된 모델을 CPU에서 평가할 경우 `torch.load(MODEL_PATH, map_location=device)`처럼 장치를 명시하는 것이 안전합니다.
- **F1 임계값 계산:** `precision_recall_curve`의 Precision/Recall 배열은 임계값 배열보다 원소가 하나 많습니다. 현행 평가 코드에서 `np.argmax(f1_scores)`가 마지막 원소를 선택하면 `thresholds[best_idx]`에 접근할 때 오류가 발생할 수 있으므로, 임계값에 대응하는 구간(`precisions[:-1]`, `recalls[:-1]`)만 사용하도록 수정해야 합니다.
- **평가 독립성:** 테스트 라벨로 임계값을 최적화하면 성능이 낙관적으로 측정될 수 있습니다. 별도의 검증/테스트 분리가 필요합니다.
- **평가 범위:** 현재 AUROC와 F1-score는 **이미지 수준의 정상·불량 분류 지표**이며, `ground_truth` 마스크를 이용한 결함 위치 검출 정확도는 별도로 평가하지 않습니다.
- **모델 수렴:** Skipless 학습은 최대 1,000 Epoch로 설정되어 있으며, 실행 시간이 길 수 있습니다. 검증 손실과 TensorBoard 기록을 함께 확인하세요.

---

## ✅ 프로젝트 정리

본 프로젝트는 산업 제품 이미지를 정상 상태 중심으로 학습하는 두 가지 CNN Autoencoder를 구현하고, **재구성 오차 기반 이상 점수로 불량 여부를 판정**합니다. 기본 모델과 Skipless 모델의 구조 및 학습 설정을 비교하고, 오차 맵과 히트맵으로 이상이 발생한 위치를 시각적으로 살펴볼 수 있습니다. 제조 비전 검사 및 이미지 기반 비지도 이상 탐지 모델의 개발·평가 과정을 실습하는 데 목적이 있습니다.
