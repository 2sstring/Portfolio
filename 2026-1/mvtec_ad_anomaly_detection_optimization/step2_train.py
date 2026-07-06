# 02_train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from step1_data_eda import MVTecDataset # 앞서 만든 Dataset 클래스 임포트

class ConvAutoencoder(nn.Module):
    """합성곱 오토인코더 아키텍처"""
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

if __name__ == "__main__":
    ROOT_DIR = './mvtec_ad' 
    CATEGORY = 'bottle' 
    BATCH_SIZE = 16
    NUM_EPOCHS = 50

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_dataset = MVTecDataset(ROOT_DIR, CATEGORY, is_train=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"학습 디바이스: {device}")

    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("모델 학습 시작...")
    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        for images, _, _ in train_loader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images) # 자기 자신을 타겟으로 학습
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss/len(train_loader):.4f}')

    # 학습된 모델 가중치 저장
    SAVE_PATH = 'autoencoder_model.pth'
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"모델 저장 완료: {SAVE_PATH}")