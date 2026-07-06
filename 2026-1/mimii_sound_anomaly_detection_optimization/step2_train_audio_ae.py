# %% [1] 라이브러리 임포트 및 장치 설정
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"학습 장치(Device) 설정 완료: {device}")

# %% [2] 실제 MIMII 데이터셋 경로 설정 및 커스텀 Dataset 구축
normal_dir = '0_dB_valve/valve/id_02/normal'
normal_files = glob.glob(os.path.join(normal_dir, '*.wav'))

if not normal_files:
    print(f"에러: '{normal_dir}' 경로에서 .wav 파일을 찾을 수 없습니다.")
    exit()

print(f"로드된 정상 학습 오디오 파일 개수: {len(normal_files)}개")

class MIMIIMelDataset(Dataset):
    def __init__(self, file_paths, sr=16000, n_mels=128, max_frames=128):
        self.file_paths = file_paths
        self.sr = sr
        self.n_mels = n_mels
        self.max_frames = max_frames # CNN 입력 크기를 고정하기 위함 (128x128)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 1. 파일 로드
        path = self.file_paths[idx]
        y, _ = librosa.load(path, sr=self.sr)
        
        # 2. Mel-Spectrogram 변환
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # [추가됨] 정규화 (Normalization): -80 ~ 0 dB 범위를 0 ~ 1 범위로 스케일링
        mel_db = (mel_db + 80.0) / 80.0
        
        # 3. 텐서 크기 고정 (128x128)
        # 10초짜리 오디오의 경우 프레임이 300개를 넘어가므로, 앞부분 128프레임만 잘라서 사용합니다.
        # 현장 데이터의 특성에 따라 중간을 자르거나, 슬라이딩 윈도우를 적용할 수도 있습니다.
        if mel_db.shape[1] > self.max_frames:
            mel_db = mel_db[:, :self.max_frames]
        else:
            # 혹시 길이가 짧은 파일이 있다면 0으로 패딩(Padding)을 채웁니다.
            pad_width = self.max_frames - mel_db.shape[1]
            mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode='constant')

        # 4. PyTorch 텐서 변환 (Channel, Mels, Time) -> (1, 128, 128)
        mel_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)
        
        # 오토인코더는 입력 데이터를 그대로 정답(Target)으로 사용하므로 두 번 반환합니다.
        return mel_tensor, mel_tensor

# 데이터셋 및 DataLoader 생성
train_dataset = MIMIIMelDataset(normal_files)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
print("학습용 DataLoader 구축 완료 (128x128 크기로 자동 전처리 적용)")

# %% [3] CNN 기반 오토인코더(Autoencoder) 아키텍처 정의
class AudioAutoencoder(nn.Module):
    def __init__(self):
        super(AudioAutoencoder, self).__init__()
        
        # 인코더 (Encoder): 128x128 이미지를 압축
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 64) # 매우 강력한 압축 (64차원으로 축소)
        )
        
        # 디코더 (Decoder): 다시 원본 크기인 128x128로 복원
        self.decoder_fc = nn.Sequential(
            nn.Linear(64, 256 * 4 * 4),
            nn.ReLU()
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() # 출력을 0 ~ 1 사이로 제한
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder_fc(encoded)
        decoded = decoded.view(-1, 256, 4, 4) # 다시 2D 형태로 변환
        reconstructed = self.decoder_conv(decoded)
        return reconstructed

model = AudioAutoencoder().to(device)
print("\n[오토인코더 모델 준비 완료]")

# %% [4] 손실 함수 및 최적화 설정
# 픽셀 단위의 복원 오차를 측정하는 MSE 사용
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %% [5] 오토인코더 학습 루프 (Training Loop)
epochs = 50
print("\n[모델 학습 시작 - 실제 밸브의 정상 작동 소리 학습]")
model.train()

for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_x, batch_target in train_loader:
        batch_x = batch_x.to(device)
        batch_target = batch_target.to(device)
        
        # Forward, Loss, Backward, Step
        outputs = model(batch_x)
        loss = criterion(outputs, batch_target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / len(train_loader)
    
    if (epoch + 1) % 2 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:2d}/{epochs}] | Reconstruction Loss (MSE): {avg_loss:.4f}")

print("학습 완료!")

# %% [6] 모델 가중치 저장
os.makedirs('audio_models', exist_ok=True)
model_path = 'audio_models/audio_autoencoder_real.pth'
torch.save(model.state_dict(), model_path)
print(f"\n[저장 완료] 밸브 정상음 학습 모델이 '{model_path}'에 저장되었습니다.")