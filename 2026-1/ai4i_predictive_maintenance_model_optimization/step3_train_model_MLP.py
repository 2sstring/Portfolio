# 라이브러리 임포트 및 전처리 데이터 불러오기
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 로깅용
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

import joblib


# 전처리 모듈 임포트
try:
    from step2_data_prep import train_loader, val_loader, test_loader, X_train, scaler
    print("데이터 전처리 모듈(step2_data_prep) 로드 성공")
except ModuleNotFoundError:
    print("에러: 'step2_data_prep.py' 파일을 찾을 수 없습니다. 같은 폴더에 있는지 확인해주세요.")
    exit()

# 학습 장치(Device) 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"학습 장치(Device) 설정 완료: {device}")

# 모델 아키텍처 정의 및 초기화
class FaultDiagnosisMLP(nn.Module):
    def __init__(self, input_dim):
        super(FaultDiagnosisMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128), 
            nn.Dropout(0.2),    
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
            
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

input_dim = X_train.shape[1] 
model = FaultDiagnosisMLP(input_dim).to(device)

print(f"\n[모델 구조 확인]\n{model}")

# 손실 함수, 최적화 알고리즘 및 TensorBoard 설정
criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

# TensorBoard 기록을 위한 SummaryWriter 초기화 (저장 경로 지정)
log_dir = "runs/fault_diagnosis_experiment_mlp"
writer = SummaryWriter(log_dir)
print(f"TensorBoard 로그 디렉토리 설정: {log_dir}")

# 모델 그래프(구조)를 TensorBoard에 기록
# (더미 데이터를 하나 통과시켜서 그래프를 그립니다)
dummy_input = torch.randn(1, input_dim).to(device)
writer.add_graph(model, dummy_input)

# 모델 학습 및 검증 루프 (Training & Validation Loop)
epochs = 130
print("\n [모델 학습 시작]")

for epoch in range(epochs):
    # --- 1. Training Phase ---
    model.train() 
    train_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad() 
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()       
        optimizer.step()      
        
        train_loss += loss.item()
        
    avg_train_loss = train_loss / len(train_loader)
    
    # --- 2. Validation Phase ---
    model.eval()
    val_loss = 0.0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            v_loss = criterion(outputs, batch_y)
            val_loss += v_loss.item()
            
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            
    avg_val_loss = val_loss / len(val_loader)
    val_f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    # --- 3. TensorBoard에 지표 기록 ---
    writer.add_scalars('Loss', {'Train': avg_train_loss, 'Validation': avg_val_loss}, epoch)
    writer.add_scalar('Metrics/Validation_F1', val_f1, epoch)
    
    # 진행 상황 출력
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:2d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}") #| Patience: {patience_counter}/{patience}")

writer.close()
print("학습 및 TensorBoard 기록 완료!")

# Validation 세트에서 최적 임계값 탐색
model.eval()
val_targets, val_probs = [], []
with torch.no_grad():
    for batch_X, batch_y in val_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        probs = torch.sigmoid(outputs)
        val_probs.extend(probs.cpu().numpy())
        val_targets.extend(batch_y.cpu().numpy())

val_targets = np.array(val_targets).reshape(-1)
val_probs = np.array(val_probs).reshape(-1)
thresholds = np.linspace(0.0, 1.0, 101)
f1_scores = [
    f1_score(val_targets, (val_probs >= t).astype(int), zero_division=0)
    for t in thresholds
]
best_idx = int(np.argmax(f1_scores))
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]
print(f"Best validation threshold: {best_threshold:.2f}, Validation F1: {best_f1:.4f}")

# 최종 모델 평가 (Evaluation)
model.eval()
all_preds, all_probs, all_targets = [], [], []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        outputs = model(batch_X)
        probs = torch.sigmoid(outputs)
        preds = (probs >= best_threshold).float()
        
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(batch_y.cpu().numpy())

# 평가지표 계산
acc = accuracy_score(all_targets, all_preds)
prec = precision_score(all_targets, all_preds, zero_division=0)
rec = recall_score(all_targets, all_preds, zero_division=0)
f1 = f1_score(all_targets, all_preds, zero_division=0)
auc = roc_auc_score(all_targets, all_probs)

print("\n[최종 테스트 데이터셋 평가 결과]")
print(f"Accuracy (정확도):  {acc:.4f}")
print(f"Precision (정밀도): {prec:.4f}")
print(f"Recall (재현율):    {rec:.4f}")
print(f"F1-Score:           {f1:.4f}")
print(f"ROC-AUC:            {auc:.4f}")

# 모델 가중치 및 스케일러 저장
# 디렉토리가 없으면 생성 (선택 사항)
os.makedirs('models', exist_ok=True)
model_path = 'models/fault_diagnosis_mlp.pth'
scaler_path = 'models/sensor_scaler_mlp.pkl'

torch.save(model.state_dict(), model_path)
joblib.dump(scaler, scaler_path)

print(f"\n[저장 완료] 모델 가중치('{model_path}')와 스케일러('{scaler_path}')가 저장되었습니다.")

# 모델 평가 결과 시각화 (Confusion Matrix 및 평가지표)


print("\n평가 결과 시각화 그래프를 생성합니다...")

# 시각화 환경 및 레이아웃 설정 (1행 3열)
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- 1. Confusion Matrix (혼동 행렬) ---
cm = confusion_matrix(all_targets, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Normal (0)', 'Fault (1)'],
            yticklabels=['Normal (0)', 'Fault (1)'],
            cbar=False, annot_kws={"size": 14})
axes[0].set_title('Confusion Matrix', fontsize=14, pad=10)
axes[0].set_ylabel('Actual Status', fontsize=12)
axes[0].set_xlabel('Predicted Status', fontsize=12)

# --- 2. 5가지 평가지표 요약 바 차트 ---
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
metrics_values = [acc, prec, rec, f1, auc]

sns.barplot(x=metrics_names, y=metrics_values, ax=axes[1], palette='viridis')
axes[1].set_title('Evaluation Metrics Summary', fontsize=14, pad=10)
axes[1].set_ylim(0, 1.1) # y축 범위를 0~1.1로 고정하여 여백 확보

# 막대 그래프 위에 정확한 수치 텍스트 표시
for i, v in enumerate(metrics_values):
    axes[1].text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# --- 3. ROC Curve (수신자 조작 특성 곡선) ---
fpr, tpr, thresholds = roc_curve(all_targets, all_probs)
axes[2].plot(fpr, tpr, color='crimson', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
axes[2].plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', alpha=0.7)
axes[2].set_xlim([-0.02, 1.0])
axes[2].set_ylim([0.0, 1.05])
axes[2].set_xlabel('False Positive Rate (FPR)', fontsize=12)
axes[2].set_ylabel('True Positive Rate (TPR)', fontsize=12)
axes[2].set_title('Receiver Operating Characteristic (ROC)', fontsize=14, pad=10)
axes[2].legend(loc="lower right", fontsize=11)

# 그래프 간격 조절 및 출력
plt.tight_layout()
plt.show()