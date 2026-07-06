import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve
from torch.utils.data import DataLoader
from torchvision import transforms
from step1_data_eda import MVTecDataset
from step2_train import ConvAutoencoder
from step2_train_cnn import SkiplessAutoencoder

def evaluate_performance(model, test_loader, device):
    """테스트 데이터셋 전체를 평가하여 정량적 지표를 산출합니다."""
    model.eval()
    y_true = []
    y_scores = []
    
    print("전체 테스트 데이터셋 정량 평가를 진행합니다...")
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            # 1. 픽셀 단위 오차 계산 및 노이즈 제거 (가우시안 블러)
            error = torch.mean((images - outputs) ** 2, dim=1) 
            error_map = error.squeeze().cpu().numpy()
            error_map = cv2.GaussianBlur(error_map, (15, 15), 0)
            
            # 2. 이미지 레벨 이상치 점수(Anomaly Score) 산출
            # 제조품은 아주 작은 결함 하나만 있어도 전체가 불량입니다. 
            # 따라서 오차 맵에서 '가장 오차가 큰 픽셀의 값(Max)'을 해당 이미지의 대표 불량 점수로 사용합니다.
            anomaly_score = np.max(error_map)
            
            y_scores.append(anomaly_score)
            y_true.append(labels.item()) # 0: 정상, 1: 불량

    # 3. 정량적 지표 계산
    # AUROC: 임계값에 상관없이 모델의 전반적인 정상/불량 분류 능력을 평가
    auroc = roc_auc_score(y_true, y_scores)
    
    # Precision-Recall 기반 최적 임계값 및 F1-Score 탐색
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    # F1-Score 계산 (0으로 나누어지는 것 방지)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_threshold = thresholds[best_idx]
    
    print("-" * 40)
    print(f"[전체 평가 결과]")
    print(f"AUROC Score          : {auroc:.4f}")
    print(f"Best F1-Score        : {best_f1:.4f}")
    print(f"Optimal Threshold    : {best_threshold:.4f}")
    print("-" * 40)
    
    return best_threshold

def visualize_anomaly(model, test_loader, device, threshold, num_samples=3):
    """결함 탐지 시각화 및 판정 결과를 출력합니다."""
    model.eval()
    samples_shown = 0
    
    print(f"\n최적 임계값({threshold:.4f})을 적용하여 시각화를 시작합니다.")
    
    with torch.no_grad():
        for images, labels, _ in test_loader:
            if labels.item() == 0: 
                continue
                
            images = images.to(device)
            outputs = model(images)
            
            error = torch.mean((images - outputs) ** 2, dim=1) 
            error_map = error.squeeze().cpu().numpy()
            error_map = cv2.GaussianBlur(error_map, (15, 15), 0)
            
            anomaly_score = np.max(error_map)
            
            # 산출된 Threshold를 바탕으로 불량(NG) / 정상(OK) 판정
            prediction = "NG (Defect)" if anomaly_score >= threshold else "OK (Normal)"
            
            error_map_norm = cv2.normalize(error_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap = cv2.applyColorMap(error_map_norm, cv2.COLORMAP_JET)
            
            img_np = images.squeeze().cpu().permute(1, 2, 0).numpy()
            out_np = outputs.squeeze().cpu().permute(1, 2, 0).numpy()
            
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted((img_np * 255).astype(np.uint8), 0.5, heatmap_rgb, 0.5, 0)
            
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            # 타이틀에 예측 판정 결과 및 점수 표시
            axes[0].imshow(img_np); axes[0].set_title(f'Original\nScore: {anomaly_score:.4f} -> {prediction}')
            axes[1].imshow(out_np); axes[1].set_title('Reconstructed')
            axes[2].imshow(error_map, cmap='hot'); axes[2].set_title('Error Map')
            axes[3].imshow(overlay); axes[3].set_title('Overlay Heatmap')
            
            for ax in axes:
                ax.axis('off')
            plt.show()
            
            samples_shown += 1
            if samples_shown >= num_samples:
                break

if __name__ == "__main__":
    ROOT_DIR = './mvtec_ad' 
    CATEGORY = 'bottle' 
    MODEL_PATH = 'autoencoder_model_skipless.pth'

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    test_dataset = MVTecDataset(ROOT_DIR, CATEGORY, is_train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = ConvAutoencoder().to(device)
    #model = ImprovedAutoencoder().to(device)
    model = SkiplessAutoencoder().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    
    # 1. 정량 평가 수행 및 최적 임계값 도출
    optimal_thresh = evaluate_performance(model, test_loader, device)
    
    # 2. 도출된 임계값을 시각화 함수에 전달하여 실제 판정 시뮬레이션
    visualize_anomaly(model, test_loader, device, optimal_thresh, num_samples=3)