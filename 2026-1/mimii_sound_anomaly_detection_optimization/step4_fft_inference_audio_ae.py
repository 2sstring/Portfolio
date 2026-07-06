# %% [1] 라이브러리 임포트 및 모델 로드
import os
import glob
import numpy as np
import librosa
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# 장치 설정
device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

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

stats = np.load('audio_models/fft_feature_stats.npz')
feature_mean = stats['feature_mean'].astype(np.float32)
feature_std = stats['feature_std'].astype(np.float32)
THRESHOLD = float(stats['train_threshold'][0])

print(f"모델 로드 완료. 설정된 결함 임계값(train 기반): {THRESHOLD:.6f}")


def positive_fft(input_rawdata: np.ndarray, sampling_frequency: int = 16000, length: int = 16000):
    if len(input_rawdata) < length:
        input_rawdata = np.pad(input_rawdata, (0, length - len(input_rawdata)))
    elif len(input_rawdata) > length:
        input_rawdata = input_rawdata[:length]

    input_rawdata = np.hanning(length) * input_rawdata
    y = np.fft.fft(input_rawdata)
    n = len(y) // 2
    y = 2 * np.abs(y[:n]) / (length / 2)
    return y.astype(np.float32)


# %% [2] 추론용 비정상 오디오 파일 선택
abnormal_files = sorted(glob.glob('0_dB_valve/valve/id_02/abnormal/*.wav'))
if not abnormal_files:
    raise FileNotFoundError("비정상 추론용 wav 파일을 확인해주세요.")

test_file = abnormal_files[0]
print(f"추론 대상 파일: {test_file}")

y, sr = librosa.load(test_file, sr=SR)

# %% [3] 슬라이딩 윈도우 추론 시뮬레이션
hop_size = SR  # 1초씩 이동
scores = []
times = []

print("\n실시간 모니터링 시뮬레이션 중...")
for start in range(0, max(len(y) - WINDOW_SIZE + 1, 1), hop_size):
    window = y[start:start + WINDOW_SIZE]
    if len(window) < WINDOW_SIZE:
        window = np.pad(window, (0, WINDOW_SIZE - len(window)))

    feat = positive_fft(window, sampling_frequency=SR, length=WINDOW_SIZE)
    feat = np.log1p(feat).astype(np.float32)
    feat_norm = ((feat - feature_mean) / feature_std).astype(np.float32)

    input_tensor = torch.tensor(feat_norm, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        reconstructed, _ = model(input_tensor)
        loss = torch.mean((reconstructed - input_tensor) ** 2).item()

    scores.append(loss)
    current_time = start / sr
    times.append(current_time)

    status = "[ABNORMAL]" if loss > THRESHOLD else "[NORMAL]"
    print(f"Time: {current_time:4.1f}s | Anomaly Score: {loss:.6f} | Status: {status}")

# %% [4] 시간에 따른 결함 점수 추이 시각화
plt.figure(figsize=(12, 5))
plt.plot(times, scores, marker='o', label='Anomaly Score')
plt.axhline(y=THRESHOLD, color='red', linestyle='--', label='Detection Threshold')
plt.title('Machine Status Monitoring over Time (FFT + AutoEncoder)')
plt.xlabel('Time (sec)')
plt.ylabel('Reconstruction Error (MSE)')
plt.legend()
plt.show()
