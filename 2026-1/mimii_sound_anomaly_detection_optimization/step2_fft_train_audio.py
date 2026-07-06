# %% [1] 라이브러리 임포트 및 장치 설정
import os
import glob
import random
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 재현성 고정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 장치 설정
device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)
print(f"학습 장치(Device): {device}")

# %% [2] 데이터 경로 설정
normal_dir = '0_dB_valve/valve/id_02/normal'
normal_files = sorted(glob.glob(os.path.join(normal_dir, '*.wav')))

if not normal_files:
    raise FileNotFoundError(f"정상 학습 파일이 없습니다: {normal_dir}")

print(f"로드된 정상 학습 오디오 파일 개수: {len(normal_files)}개")

# %% [3] FFT 기반 특징 추출 함수
SR = 16000
WINDOW_SEC = 1.0
WINDOW_SIZE = int(SR * WINDOW_SEC)
FFT_FEATURE_DIM = WINDOW_SIZE // 2  # 양의 주파수 성분만 사용 -> 8000

def positive_fft(input_rawdata: np.ndarray, sampling_frequency: int = 16000, length: int = 16000):
    """참고 노트북 구조를 반영한 양의 주파수 FFT 계산 함수"""
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
    """1초씩 나누어 FFT 후 평균"""
    samples = [data[i:i + fs] for i in range(0, len(data), stride)]
    amps = []
    freq = None
    for samp in samples:
        y, freq = positive_fft(samp, sampling_frequency=fs, length=fs)
        amps.append(y)
    amp_mean = np.mean(np.stack(amps, axis=0), axis=0)
    return freq, amp_mean

# %% [4] 전체 학습 특징 행렬 생성
print("FFT 특징 추출 중...")
train_features = []
for path in normal_files:
    wav, _ = librosa.load(path, sr=SR)
    _, amp = fft_result(wav)
    train_features.append(amp)

X_train = np.stack(train_features, axis=0).astype(np.float32)
print(f"추출된 FFT 특징 shape: {X_train.shape}")

# 선택 사항: 학습 안정화를 위해 log scaling 적용
X_train = np.log1p(X_train).astype(np.float32)
print("log1p 스케일링 적용 완료")

# 학습 시 사용할 정규화 통계 저장 (평가 시 동일 적용)
feature_mean = X_train.mean(axis=0)
feature_std = X_train.std(axis=0) + 1e-8
X_train_norm = ((X_train - feature_mean) / feature_std).astype(np.float32)
print("특징 표준화 완료")

# %% [5] Dataset 및 DataLoader 구축
class FFTDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.x = data

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32)


train_dataset = FFTDataset(X_train_norm)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)
print("학습용 DataLoader 구축 완료")

# %% [6] FFT AutoEncoder 정의 (참고 노트북 구조 반영)
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
print("\n[FFT AutoEncoder 모델 준비 완료]")

# %% [7] 손실 함수 및 최적화 설정
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# %% [8] 학습 루프
EPOCHS = 100
print("\n[모델 학습 시작 - FFT 특징 기반 정상 작동 소리 학습]")
model.train()
loss_history = []

for epoch in range(EPOCHS):
    running_loss = 0.0

    for batch_x in train_loader:
        batch_x = batch_x.to(device)
        outputs, _ = model(batch_x)
        loss = criterion(outputs, batch_x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)

    avg_loss = running_loss / len(train_loader.dataset)
    loss_history.append(avg_loss)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | Reconstruction Loss (MSE): {avg_loss:.6f}")

print("학습 완료!")

# %% [9] 학습 데이터 score 계산 및 보조 threshold 저장
@torch.no_grad()
def compute_scores(model: nn.Module, dataloader: DataLoader):
    model.eval()
    scores = []
    for batch_x in dataloader:
        batch_x = batch_x.to(device)
        outputs, _ = model(batch_x)
        score = torch.mean((outputs - batch_x) ** 2, dim=1)
        scores.extend(score.cpu().numpy())
    return np.array(scores, dtype=np.float32)


train_scores = compute_scores(model, train_loader)
#train_threshold = float(train_scores.max())
#print(f"학습 정상 데이터 최대 score (참고용 threshold): {train_threshold:.6f}")
train_threshold = float(np.percentile(train_scores, 95))
print(f"학습 정상 데이터 p95 score (참고용 threshold): {train_threshold:.6f}")

# %% [10] 모델 및 전처리 통계 저장
os.makedirs('audio_models', exist_ok=True)
model_path = 'audio_models/fft_autoencoder_real.pth'
stats_path = 'audio_models/fft_feature_stats.npz'

# 모델 가중치 저장
torch.save(model.state_dict(), model_path)
# 평가/추론 시 동일 전처리를 위한 통계 저장
np.savez(
    stats_path,
    feature_mean=feature_mean,
    feature_std=feature_std,
    train_threshold=np.array([train_threshold], dtype=np.float32),
)

print(f"\n[저장 완료] 모델: {model_path}")
print(f"[저장 완료] 전처리 통계: {stats_path}")
