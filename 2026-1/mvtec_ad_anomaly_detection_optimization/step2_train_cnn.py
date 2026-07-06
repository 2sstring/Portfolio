# 02_train_skipless.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from step1_data_eda import MVTecDataset


class SkiplessAutoencoder(nn.Module):
    def __init__(self):
        super(SkiplessAutoencoder, self).__init__()

        # -------- Encoder --------
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),   # 256x256
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                        # 128x128

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                        # 64x64

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                        # 32x32

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                        # 16x16
        )

        # -------- Bottleneck --------
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # -------- Decoder --------
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2),  # 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),    # 128x128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),    # 256x256
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.bottleneck(z)
        out = self.decoder(z)
        return out


if __name__ == "__main__":
    ROOT_DIR = "./mvtec_ad"
    CATEGORY = "bottle"

    BATCH_SIZE = 16
    NUM_EPOCHS = 1000
    VAL_RATIO = 0.2
    LR = 1e-3
    SEED = 42

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # 기존 비교를 위해 train augmentation은 너무 많이 바꾸지 않음
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # split용 원본 dataset
    full_dataset_for_split = MVTecDataset(
        ROOT_DIR, CATEGORY, is_train=True, transform=None
    )

    total_len = len(full_dataset_for_split)
    val_len = int(total_len * VAL_RATIO)
    train_len = total_len - val_len

    train_subset, val_subset = random_split(
        full_dataset_for_split,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(SEED)
    )

    # transform 다르게 적용하기 위해 다시 생성
    full_train_dataset = MVTecDataset(
        ROOT_DIR, CATEGORY, is_train=True, transform=transform_train
    )
    full_val_dataset = MVTecDataset(
        ROOT_DIR, CATEGORY, is_train=True, transform=transform_val
    )

    train_dataset = Subset(full_train_dataset, train_subset.indices)
    val_dataset = Subset(full_val_dataset, val_subset.indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"학습 디바이스: {device}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    model = SkiplessAutoencoder().to(device)

    # 구조 효과를 먼저 보기 위해 우선 MSE 유지
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )

    log_dir = f"runs/{CATEGORY}_skipless_autoencoder"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard 로그 디렉토리 설정: {log_dir}")

    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    writer.add_graph(model, dummy_input)

    print("모델 학습 시작...")

    best_val_loss = float("inf")
    best_epoch = -1
    save_path = "autoencoder_model_skipless.pth"

    for epoch in range(NUM_EPOCHS):
        # -------------------
        # Train
        # -------------------
        model.train()
        train_loss_sum = 0.0

        for images, _, _ in train_loader:
            images = images.to(device)

            outputs = model(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)

        # -------------------
        # Validation
        # -------------------
        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for images, _, _ in val_loader:
                images = images.to(device)
                outputs = model(images)
                loss = criterion(outputs, images)
                val_loss_sum += loss.item()

        avg_val_loss = val_loss_sum / len(val_loader)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalars(
            "Loss",
            {"Train": avg_train_loss, "Val": avg_val_loss},
            epoch
        )
        writer.add_scalar("Learning Rate", current_lr, epoch)

        # 10 epoch마다 validation 샘플 기록
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                sample_imgs, _, _ = next(iter(val_loader))
                sample_imgs = sample_imgs.to(device)
                sample_outs = model(sample_imgs)

            writer.add_images("Input", sample_imgs[:4], epoch)
            writer.add_images("Reconstruction", sample_outs[:4], epoch)

        # best model 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_path)

        print(
            f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}] | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"LR: {current_lr:.6e}"
        )

    writer.close()

    print("\n학습 완료")
    print(f"최저 Val Loss: {best_val_loss:.6f} (Epoch {best_epoch})")
    print(f"Best 모델 저장 완료: {save_path}")