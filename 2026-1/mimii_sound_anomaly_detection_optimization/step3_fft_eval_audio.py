# %% [1] 라이브러리 임포트 및 모델 로드
import os
import glob
import numpy as np
import librosa
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# 장치 설정
device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

# %% [2] 학습 시와 동일한 FFT AutoEncoder 정의
SR = 16000
WINDOW_SEC = 1.0
WINDOW_SIZE = int(SR * WINDOW_SEC)
FFT_FEATURE_DIM = WINDOW_SIZE // 2


class FFTAutoEncoder(nn.Module):
    def __init__(self, input_dim: int = FFT_FEATURE_DIM, latent_dim: int = 200):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4000, bias=False),
            nn.ReLU(),
            nn.Linear(4000, 2000, bias=False),
            nn.ReLU(),
            nn.Linear(2000, 1000, bias=False),
            nn.ReLU(),
            nn.Linear(1000, latent_dim, bias=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1000, bias=True),
            nn.ReLU(),
            nn.Linear(1000, 2000, bias=False),
            nn.ReLU(),
            nn.Linear(2000, 4000, bias=False),
            nn.ReLU(),
            nn.Linear(4000, input_dim, bias=False),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


model = FFTAutoEncoder().to(device)
model.load_state_dict(torch.load('audio_models/fft_autoencoder_real.pth', map_location=device))
model.eval()
print("FFT AutoEncoder 모델 로드 완료")

stats = np.load('audio_models/fft_feature_stats.npz')
feature_mean = stats['feature_mean'].astype(np.float32)
feature_std = stats['feature_std'].astype(np.float32)
train_threshold = float(stats['train_threshold'][0])
print(f"저장된 train 기반 threshold(참고용): {train_threshold:.6f}")

# %% [3] 평가용 데이터 경로 설정 (정상/비정상)
normal_test_dir = '0_dB_valve/valve/id_02/normal'
abnormal_test_dir = '0_dB_valve/valve/id_02/abnormal'

normal_test_files = sorted(glob.glob(os.path.join(normal_test_dir, '*.wav')))[:50]
abnormal_test_files = sorted(glob.glob(os.path.join(abnormal_test_dir, '*.wav')))[:50]

if not normal_test_files or not abnormal_test_files:
    raise FileNotFoundError("평가용 정상/비정상 wav 파일을 확인해주세요.")

print(f"정상 평가 파일 수: {len(normal_test_files)}")
print(f"비정상 평가 파일 수: {len(abnormal_test_files)}")

# %% [4] FFT 특징 추출 함수 (학습 시와 동일)
def positive_fft(input_rawdata: np.ndarray, sampling_frequency: int = 16000, length: int = 16000):
    if len(input_rawdata) < length:
        input_rawdata = np.pad(input_rawdata, (0, length - len(input_rawdata)))
    elif len(input_rawdata) > length:
        input_rawdata = input_rawdata[:length]

    input_rawdata = np.hanning(length) * input_rawdata
    y = np.fft.fft(input_rawdata)
    n = len(y) // 2
    y = 2 * np.abs(y[:n]) / (length / 2)
    freq = np.linspace(0, sampling_frequency / 2, n, endpoint=True)
    return y.astype(np.float32), freq.astype(np.float32)


def fft_result(data: np.ndarray, stride: int = WINDOW_SIZE, fs: int = SR):
    samples = [data[i:i + fs] for i in range(0, len(data), stride)]
    amps = []
    freq = None
    for samp in samples:
        y, freq = positive_fft(samp, sampling_frequency=fs, length=fs)
        amps.append(y)
    amp_mean = np.mean(np.stack(amps, axis=0), axis=0)
    return freq, amp_mean


# %% [5] 파일별 복원 오차 계산 함수
criterion = nn.MSELoss(reduction='mean')


def compute_errors(file_list):
    errors = []
    raw_features = []
    reconstructed_features = []

    with torch.no_grad():
        for path in file_list:
            y, _ = librosa.load(path, sr=SR)
            _, feat = fft_result(y)
            feat = np.log1p(feat).astype(np.float32)
            feat_norm = ((feat - feature_mean) / feature_std).astype(np.float32)

            input_tensor = torch.tensor(feat_norm, dtype=torch.float32).unsqueeze(0).to(device)
            reconstructed, _ = model(input_tensor)

            loss = criterion(reconstructed, input_tensor)
            errors.append(loss.item())

            raw_features.append(feat_norm)
            reconstructed_features.append(reconstructed.cpu().squeeze(0).numpy())

    return np.array(errors), np.array(raw_features), np.array(reconstructed_features)


print("정상 및 비정상 데이터의 복원 오차 계산 중...")
normal_errors, normal_raw, normal_recon = compute_errors(normal_test_files)
abnormal_errors, abnormal_raw, abnormal_recon = compute_errors(abnormal_test_files)

# %% [6] 오차 분포 시각화 (기존 평가 흐름 유지)
plt.figure(figsize=(10, 6))
plt.hist(normal_errors, bins=20, alpha=0.6, density=True, label='Normal (Valve)')
plt.hist(abnormal_errors, bins=20, alpha=0.6, density=True, label='Abnormal (Valve)')
plt.title('Reconstruction Error Distribution (FFT + AutoEncoder)')
plt.xlabel('Mean Squared Error (MSE)')
plt.legend()
plt.show()

# %% [7] ROC-AUC 성능 평가 (기존과 동일)
y_true = np.concatenate([np.zeros(len(normal_errors)), np.ones(len(abnormal_errors))])
y_scores = np.concatenate([normal_errors, abnormal_errors])

fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# %% [8] F1-Score 및 최적 임계값 탐색 (기존 평가 방식 유지)
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
optimal_f1 = f1_scores[optimal_idx]

print(f"\n최적의 이상치 탐지 임계값(Threshold): {optimal_threshold:.6f}")
print(f"해당 임계값에서의 최고 F1-Score: {optimal_f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"참고용 train 기반 threshold: {train_threshold:.6f}")

# 최적 threshold 기준 예측
y_pred = (y_scores >= optimal_threshold).astype(int)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Abnormal'])
disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
plt.title(f'Confusion Matrix (Optimal Threshold: {optimal_threshold:.6f})')
plt.show()

# %% [9] train 기반 threshold도 함께 비교 출력
train_based_pred = (y_scores >= train_threshold).astype(int)
train_based_f1 = f1_score(y_true, train_based_pred)
print(f"train 기반 threshold에서의 F1-Score(비교용): {train_based_f1:.4f}")

# %% [10] 원본 vs 재구성 FFT 특징 시각화 (Explainability)
def visualize_reconstruction(raw_feature, reconstructed_feature, title):
    plt.figure(figsize=(12, 4))
    plt.plot(raw_feature, label='Original FFT Feature')
    plt.plot(reconstructed_feature, label='Reconstructed FFT Feature', alpha=0.8)
    plt.title(title)
    plt.xlabel('Frequency Bin')
    plt.ylabel('Normalized Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.show()


print("\n정상 및 비정상 FFT 특징 복원 결과 시각화 중...")
visualize_reconstruction(normal_raw[0], normal_recon[0], 'Normal Valve - FFT Feature Reconstruction')
visualize_reconstruction(abnormal_raw[0], abnormal_recon[0], 'Abnormal Valve - FFT Feature Reconstruction')
