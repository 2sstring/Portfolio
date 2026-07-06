# MIMII 모델 최적화 및 고도화

## 프로젝트 개요

본 프로젝트는 설비 음향 데이터를 활용하여 기계 이상음을 탐지하는 AI 모델을 개발하고 성능을 평가하는 프로젝트입니다.

MIMII 데이터셋을 사용하여 정상 음향과 비정상 음향을 비교하고, Autoencoder 기반 비지도 이상 탐지 모델을 학습합니다.

음향 데이터는 시간 영역 파형, 주파수 영역 FFT, Mel-spectrogram 등으로 변환하여 분석합니다.

## 데이터셋 정보

본 저장소에는 원본 음향 데이터셋을 포함하지 않습니다.

* 데이터셋명: MIMII Dataset
* 다운로드 링크: https://zenodo.org/records/3384388
* 사용 설비 예시: `valve`
* 사용 조건 예시: `0 dB`, `id_02`
* 사용 데이터 경로: `0_dB_valve/valve/id_02/`

데이터셋을 다운로드한 후 아래와 같이 프로젝트 폴더에 배치합니다.

```text
05-mimii-sound-anomaly-detection-optimization/
├── README.md
├── step1_eda_audio.py
├── step1_fft_eda_audio.py
├── step2_train_audio_ae.py
├── step2_fft_train_audio.py
├── step3_eval_audio_ae.py
├── step3_fft_eval_audio.py
├── step4_inference_audio_ae.py
├── step4_fft_inference_audio_ae.py
└── 0_dB_valve/               # GitHub 업로드 제외
    └── valve/
        └── id_02/
            ├── normal/
            └── abnormal/
```

## 주요 내용

### 1. 음향 데이터 탐색

* 정상 음향과 비정상 음향 파일을 확인합니다.
* waveform을 시각화합니다.
* spectrogram을 시각화합니다.
* Mel-spectrogram을 시각화합니다.

### 2. FFT 기반 분석

* 정상 음향과 비정상 음향의 주파수 성분을 비교합니다.
* FFT amplitude feature를 생성합니다.
* 주파수 영역에서 이상음 특성을 확인합니다.

### 3. Mel-spectrogram Autoencoder 학습

* 음향 데이터를 Mel-spectrogram으로 변환합니다.
* 정상 데이터 기반 Autoencoder를 학습합니다.
* 재구성 오차를 이용하여 이상 탐지를 수행합니다.

### 4. FFT Autoencoder 학습

* FFT feature를 입력으로 사용하는 Autoencoder를 학습합니다.
* 정상 데이터의 주파수 패턴을 학습합니다.
* 비정상 음향에 대한 재구성 오차 증가 여부를 확인합니다.

### 5. 성능 평가

* 정상/비정상 데이터의 reconstruction error를 비교합니다.
* ROC-AUC를 계산합니다.
* 최적 threshold를 탐색합니다.
* confusion matrix를 시각화합니다.

### 6. 추론

* 학습된 모델을 불러옵니다.
* 신규 음향 데이터의 reconstruction error를 계산합니다.
* threshold 기준으로 정상/비정상을 판단합니다.

## 실행 방법

필요 라이브러리를 설치한 후 단계별로 실행합니다.

```bash
pip install torch numpy librosa scikit-learn matplotlib soundfile
python step1_eda_audio.py
python step1_fft_eda_audio.py
python step2_train_audio_ae.py
python step2_fft_train_audio.py
python step3_eval_audio_ae.py
python step3_fft_eval_audio.py
```

추론 코드는 다음과 같이 실행합니다.

```bash
python step4_inference_audio_ae.py
python step4_fft_inference_audio_ae.py
```

## 생성 파일

학습 후 다음과 같은 파일이 생성됩니다.

```text
audio_models/
├── audio_autoencoder_real.pth
├── fft_autoencoder_real.pth
└── fft_feature_stats.npz
```

## 사용 기술

* Python
* PyTorch
* librosa
* NumPy
* scikit-learn
* Matplotlib
* soundfile

## 정리

본 프로젝트는 설비 음향 데이터를 활용하여 정상음과 이상음을 구분하는 비지도 이상 탐지 모델을 개발하는 프로젝트입니다. Mel-spectrogram과 FFT feature를 각각 활용하여 Autoencoder 모델을 학습하고, reconstruction error 기반으로 설비 이상 여부를 판단합니다.
