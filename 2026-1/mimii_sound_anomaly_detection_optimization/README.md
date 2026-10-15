# 🔊 mimii_sound_anomaly_detection_optimization

**MIMII 설비 음향 데이터를 활용한 이상음 탐지 및 Autoencoder 모델 비교 프로젝트**

본 프로젝트는 설비의 정상·비정상 음향을 분석하고, **Mel-Spectrogram 기반 CNN Autoencoder**와 **FFT 기반 Autoencoder**를 학습하여 이상음을 탐지합니다. 정상 소리만으로 모델을 학습한 뒤, 입력 특징과 복원된 특징의 차이인 **복원 오차(Reconstruction Error)**를 이상 점수로 사용합니다.

> ⚠️ **데이터 안내:** MIMII 원본 WAV 파일은 이 저장소에 포함되어 있지 않습니다. 데이터 제공처에서 별도로 내려받아야 합니다.

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 분석 대상 | MIMII Dataset의 밸브(`valve`) 작동 소리 |
| 분석 조건 | `0 dB`, `id_02` 예시 |
| 데이터 형식 | 정상(`normal`) / 비정상(`abnormal`) WAV |
| 음향 특징 | Waveform, STFT, Mel-Spectrogram, FFT |
| 모델 | CNN Autoencoder / FFT Autoencoder |
| 학습 방식 | 정상 음향만 사용한 비지도 이상 탐지 모델 학습 |
| 이상 판단 | 복원 오차(MSE)와 임계값 비교 |
| 성능 평가 | ROC-AUC, F1-score, 혼동행렬, 오차 분포 |
| 추론 예시 | WAV 파일을 구간별로 분석하는 모니터링 시뮬레이션 |

## 📁 프로젝트 폴더 구조

```text
mimii_sound_anomaly_detection_optimization/
├── README.md                         # 프로젝트 설명
├── step1_eda_audio.py                # Waveform·STFT·Mel-Spectrogram 탐색
├── step1_fft_eda_audio.py            # FFT 주파수 특징 탐색
├── step2_train_audio_ae.py           # Mel CNN Autoencoder 학습
├── step2_fft_train_audio.py          # FFT Autoencoder 학습
├── step3_eval_audio_ae.py            # Mel 모델 성능 평가
├── step3_fft_eval_audio.py           # FFT 모델 성능 평가
├── step4_inference_audio_ae.py       # Mel 모델 구간별 추론
├── step4_fft_inference_audio_ae.py   # FFT 모델 구간별 추론
├── 0_dB_valve/                       # 원본 데이터 (별도 다운로드)
│   └── valve/
│       └── id_02/
│           ├── normal/               # 정상 WAV
│           └── abnormal/             # 비정상 WAV
└── audio_models/                     # 학습 시 자동 생성
    ├── audio_autoencoder_real.pth
    ├── fft_autoencoder_real.pth
    └── fft_feature_stats.npz
```

`0_dB_valve/`는 저장소에 포함하지 않는 데이터 폴더이며, `audio_models/`는 학습 후 생성됩니다. 파일명은 코드에서 사용하는 **원래 프로젝트 파일명**을 기준으로 표기했습니다.

### 🗂️ 주요 파일 설명

| 파일 | 역할 |
|---|---|
| `step1_eda_audio.py` | 정상·비정상 파형, STFT 및 Mel-Spectrogram 시각화와 청음 |
| `step1_fft_eda_audio.py` | 1초 윈도우의 평균 FFT 스펙트럼 비교 및 청음 |
| `step2_train_audio_ae.py` | 정상 WAV를 Mel-Spectrogram으로 변환하여 CNN Autoencoder 학습 |
| `step2_fft_train_audio.py` | 정상 WAV의 FFT 특징으로 Autoencoder 학습 및 정규화 통계 저장 |
| `step3_eval_audio_ae.py` | Mel 모델의 MSE 분포, ROC-AUC, F1 최적 임계값, 혼동행렬 분석 |
| `step3_fft_eval_audio.py` | FFT 모델의 성능 평가 및 학습 기반 임계값 비교 |
| `step4_inference_audio_ae.py` | Mel 모델을 이용한 구간별 이상 점수 계산·시각화 |
| `step4_fft_inference_audio_ae.py` | FFT 모델을 이용한 구간별 이상 점수 계산·시각화 |

---

## 🔗 데이터셋 및 다운로드 안내

- **데이터셋:** MIMII Dataset
- **다운로드 링크:** [MIMII Dataset — Zenodo](https://zenodo.org/records/3384388)
- **사용 설비:** `valve`
- **사용 조건:** `0 dB`, `id_02`
- **사용 경로:** `0_dB_valve/valve/id_02/`

### 📥 다운로드 및 준비 방법

1. 위 Zenodo 링크에서 MIMII 데이터를 다운로드합니다.
2. 압축 파일을 해제하여 `valve`의 **0 dB 데이터**를 준비합니다.
3. 프로젝트 최상위 폴더에 `0_dB_valve/valve/id_02/normal/` 및 `0_dB_valve/valve/id_02/abnormal/` 구조로 배치합니다.
4. 각 폴더에 `.wav` 파일이 있는지 확인한 후 **프로젝트 최상위 폴더에서** 코드를 실행합니다.

> 다른 설비, 잡음 수준 또는 ID를 사용하려면 각 Python 파일에 설정된 `normal_dir`, `abnormal_dir`, `normal_test_dir`, `abnormal_test_dir` 경로를 수정해야 합니다. 원본 데이터 재배포 여부는 데이터 제공처의 조건을 확인하세요.

---

## 🔎 음향 탐색 및 특징 생성

`step1_eda_audio.py`는 정상·비정상 WAV를 각각 선택하여 **시간영역 파형(Waveform)**, **STFT Spectrogram**, **Mel-Spectrogram**을 시각화하고 소리를 재생합니다. `step1_fft_eda_audio.py`는 같은 음향의 **평균 FFT 진폭 스펙트럼**을 비교합니다.

### 모델별 입력 전처리

| 구분 | Mel-Spectrogram 방식 | FFT 방식 |
|---|---|---|
| 오디오 로드 | `librosa`로 16,000 Hz 리샘플링 | `librosa`로 16,000 Hz 리샘플링 |
| 특징 생성 | 128개 Mel 대역, dB 변환 | 1초 구간별 Hann 윈도우·FFT 후 평균 진폭 |
| 입력 크기 | `1 × 128 × 128` | 8,000차원 벡터 |
| 전처리 | `(mel_db + 80) / 80`, 자르기·패딩 | `log1p` 변환 후 평균·표준편차로 표준화 |
| 학습 데이터 | 정상(`normal`) 파일 | 정상(`normal`) 파일 |

두 방식의 복원 오차는 서로 다른 특징 공간에서 계산되므로, **오차값 자체를 직접 비교하기보다 동일한 평가 조건에서 ROC-AUC나 F1-score를 비교**해야 합니다.

## 🧠 Autoencoder 학습 모델

| 항목 | Mel-Spectrogram CNN Autoencoder | FFT Autoencoder |
|---|---|---|
| 학습 파일 | `step2_train_audio_ae.py` | `step2_fft_train_audio.py` |
| 인코더 | 합성곱 신경망(Conv2D) | 완전연결층 `8000 → 4000 → 2000 → 1000 → 200` |
| 잠재 표현 | 64차원 | 200차원 |
| 디코더 | 전치 합성곱(ConvTranspose2D) | 완전연결층으로 8,000차원 복원 |
| 손실 함수 | MSELoss | MSELoss |
| 최적화 | Adam (`lr=0.001`) | Adam (`lr=0.0001`) |
| 학습 Epoch | 50 | 100 |
| 배치 크기 | 32 | 64 |
| 저장 파일 | `audio_autoencoder_real.pth` | `fft_autoencoder_real.pth` |

### 이상 탐지 원리

```text
정상 WAV ──→ 음향 특징 추출 ──→ Autoencoder 학습
                                      │
평가 WAV ──→ 동일한 특징 추출 ──→ 특징 복원
                                      │
                            입력 ↔ 복원 특징 비교
                                      │
                              MSE(이상 점수)
                                      │
                            Threshold와 비교
                              ├─ NORMAL
                              └─ ABNORMAL
```

정상 데이터로 학습한 모델의 **복원 오차가 기준보다 높으면 비정상으로 판단**합니다. 복원 오차는 이상 징후를 나타내는 점수이며, 고장의 구체적인 원인을 직접 분류하는 것은 아닙니다.

---

## 📈 성능 평가 및 임계값 설정

`step3_eval_audio_ae.py`와 `step3_fft_eval_audio.py`에서는 정상·비정상 파일에 대해 복원 오차를 계산합니다.

| 평가 항목 | 설명 |
|---|---|
| Reconstruction Error | 입력과 복원 결과의 평균제곱오차(MSE) |
| Error Distribution | 정상·비정상 이상 점수 분포 |
| ROC Curve / ROC-AUC | 임계값 변화에 따른 정상·비정상 구분 능력 |
| F1-score | Precision과 Recall의 조화평균 |
| Optimal Threshold | 평가 데이터에서 F1을 최대화하도록 계산한 임계값 |
| Confusion Matrix | 정상·비정상의 분류 및 오분류 현황 |
| Reconstruction Visualization | Mel-Spectrogram 또는 FFT 원본·복원 특징 비교 |

**추론 코드의 임계값 설정은 서로 다릅니다.**

- **Mel 방식:** `step4_inference_audio_ae.py`에 `THRESHOLD = 0.0016`이 고정되어 있습니다. 평가 코드의 최적 임계값을 자동으로 읽어오는 것은 아닙니다.
- **FFT 방식:** `step2_fft_train_audio.py`에서 **정상 학습 복원 오차의 95백분위수(p95)**를 계산하고 `fft_feature_stats.npz`에 보관합니다. FFT 추론 코드는 이 임계값을 읽어 사용합니다. 별도로 FFT 평가 코드는 평가 데이터 기준 F1 최적 임계값도 계산합니다.

> ⚠️ **평가 방법 주의:** 현재 정상 평가 파일은 학습에 사용한 `normal` 폴더에서 가져옵니다. 또한 평가 데이터로 F1 최적 임계값을 찾고 같은 데이터에서 평가하면 결과가 낙관적으로 나타날 수 있습니다. 독립적인 성능 확인을 위해 **파일 단위 Train / Validation / Test 분리**와 검증 데이터 기반 임계값 결정이 권장됩니다.

---

## 💻 실행 환경 및 설치

**주요 사용 기술:** Python, PyTorch, librosa, NumPy, scikit-learn, Matplotlib, Seaborn, sounddevice

프로젝트 최상위 폴더에서 가상환경을 준비합니다.

```bash
python -m venv venv
```

Linux / macOS:

```bash
source venv/bin/activate
```

Windows:

```powershell
venv\Scripts\activate
```

필요한 라이브러리를 설치합니다.

```bash
pip install numpy librosa soundfile sounddevice torch scikit-learn matplotlib seaborn
```

> 청음(`sounddevice`)에는 출력 장치와 시스템 오디오 라이브러리(예: PortAudio)가 필요할 수 있습니다. 재생이 불가능하다면 EDA의 `sd.play()`, `sd.wait()` 호출을 비활성화할 수 있습니다.

### ▶️ 방법 A — Mel-Spectrogram CNN Autoencoder

```bash
# 1. 음향 데이터 탐색
python step1_eda_audio.py

# 2. 정상 음향 기반 학습
python step2_train_audio_ae.py

# 3. 정상·비정상 음향 평가
python step3_eval_audio_ae.py

# 4. 구간별 이상 탐지 시뮬레이션
python step4_inference_audio_ae.py
```

### ▶️ 방법 B — FFT Autoencoder

```bash
# 1. FFT 특징 탐색
python step1_fft_eda_audio.py

# 2. 정상 음향 기반 학습
python step2_fft_train_audio.py

# 3. 정상·비정상 음향 평가
python step3_fft_eval_audio.py

# 4. 구간별 이상 탐지 시뮬레이션
python step4_fft_inference_audio_ae.py
```

두 파이프라인은 **독립적으로 실행**할 수 있습니다. 다만 평가·추론 전에 각 방식의 모델 학습을 완료해야 합니다.

---

## 📦 생성 파일 및 추론 결과

| 생성 파일 | 생성 스크립트 | 역할 |
|---|---|---|
| `audio_models/audio_autoencoder_real.pth` | `step2_train_audio_ae.py` | Mel CNN Autoencoder 모델 가중치 |
| `audio_models/fft_autoencoder_real.pth` | `step2_fft_train_audio.py` | FFT Autoencoder 모델 가중치 |
| `audio_models/fft_feature_stats.npz` | `step2_fft_train_audio.py` | FFT 특징 평균·표준편차와 학습 기반 p95 임계값 |

평가와 추론은 **콘솔 메시지 및 Matplotlib 그래프**로 결과를 확인합니다. 현재 스크립트에 평가 그래프의 자동 파일 저장 기능은 구현되어 있지 않습니다.

| 추론 스크립트 | 분석 간격 | 결과 |
|---|---|---|
| `step4_inference_audio_ae.py` | 2초 윈도우를 1초씩 이동 | Mel 복원 오차와 NORMAL/ABNORMAL 출력 |
| `step4_fft_inference_audio_ae.py` | 1초 윈도우를 1초씩 이동 | FFT 복원 오차와 NORMAL/ABNORMAL 출력 |

여기서 **실시간 모니터링은 WAV 파일을 시간 순서대로 분석하는 오프라인 시뮬레이션**을 뜻합니다. 마이크나 현장 센서로부터 음향을 직접 실시간 수집하는 기능은 포함되어 있지 않습니다.

---

## ⚠️ 실행 및 결과 해석 시 주의사항

1. **원본 데이터 별도 준비:** MIMII WAV는 저장소에 없으며, 코드가 참조하는 정상·비정상 폴더 경로를 맞춰야 합니다.
2. **학습·평가 데이터 중복:** 정상 평가 파일이 학습 폴더와 같으므로 독립적인 정상 테스트 데이터로 분리하는 것이 필요합니다.
3. **평가 데이터로 임계값 탐색:** 같은 평가 데이터에서 최적 Threshold를 선택하고 F1-score를 계산하면 성능이 낙관적으로 나타날 수 있습니다.
4. **Mel 임계값 고정:** 추론 코드의 `THRESHOLD = 0.0016`은 자동 최적화된 결과가 아니라 코드에 지정된 값이므로 별도 검증이 필요합니다.
5. **Mel 전처리 순서 차이:** 학습에서는 정규화 **후** 0 패딩, 추론에서는 패딩 **후** 정규화를 적용하므로 패딩된 영역의 값이 달라질 수 있습니다.
6. **FFT 학습·추론 특징 차이:** FFT 학습은 파일의 **1초 구간 평균 스펙트럼**, 추론은 **개별 1초 구간 스펙트럼**을 사용합니다. 정상 학습 기반 임계값의 적용을 별도로 검증해야 합니다.
7. **시스템 자원:** FFT Autoencoder는 8,000차원 입력과 큰 전결합층을 사용하므로 학습 시 메모리 부담이 클 수 있습니다.
8. **실험 수치:** 실제 학습·평가를 실행한 결과가 제공되지 않았으므로 ROC-AUC·F1-score 수치를 임의로 작성하지 않았습니다.

## 🧩 사용 기술

| 기술 | 활용 목적 |
|---|---|
| **Python** | 음향 분석 및 실험 실행 |
| **librosa** | WAV 로드, STFT, Mel-Spectrogram |
| **NumPy** | FFT, 윈도우 처리 및 수치 연산 |
| **PyTorch** | CNN 및 FFT Autoencoder 학습·추론 |
| **scikit-learn** | ROC-AUC, F1-score, 혼동행렬 계산 |
| **Matplotlib / Seaborn** | 파형·스펙트럼·오차 분포 시각화 |
| **sounddevice** | 정상·비정상 음향 청음 |

## ✅ 정리

**mimii_sound_anomaly_detection_optimization**은 MIMII 밸브 음향을 대상으로 **데이터 탐색 → Mel/FFT 특징 생성 → 정상음 기반 Autoencoder 학습 → 복원 오차 평가 → 구간별 이상 탐지 시뮬레이션**을 구현한 프로젝트입니다. 두 가지 음향 표현 방식을 비교하여 산업 설비의 이상음 탐지 모델 특성을 분석하는 데 목적이 있습니다.
