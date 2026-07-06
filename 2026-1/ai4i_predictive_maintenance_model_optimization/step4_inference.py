# 라이브러리 임포트 및 저장된 객체 불러오기
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os

print("실시간 추론(Inference) 환경 준비 중...")

# 학습 장치(Device) 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# 모델 아키텍처 재정의 (학습 시와 동일해야 함)
class FaultDiagnosisMLP(nn.Module):
    def __init__(self, input_dim):
        super(FaultDiagnosisMLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),    
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.3),    
            
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)

# 파일 경로 설정 (train_model.py에서 저장한 경로와 일치)
model_path = 'models/fault_diagnosis.pth'
scaler_path = 'models/sensor_scaler.pkl'

# 1. 스케일러 로드
try:
    scaler = joblib.load(scaler_path)
    print(f"스케일러 로드 완료: {scaler_path}")
except FileNotFoundError:
    print(f"에러: '{scaler_path}' 파일을 찾을 수 없습니다. 모델 학습을 먼저 진행해주세요.")
    exit()

# 2. 모델 가중치 로드
# 입력 차원(7개: 연속형 5개 + 범주형 원핫 2개)
INPUT_DIM = 7 
model = FaultDiagnosisMLP(INPUT_DIM).to(device)

try:
    # weights_only=True를 통해 보안 경고 방지 및 안전한 로드 수행
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval() # 필수: 추론 모드로 전환하여 BatchNorm 동작을 고정
    print(f"모델 로드 및 평가 모드(eval) 전환 완료: {model_path}")
except FileNotFoundError:
    print(f"에러: '{model_path}' 파일을 찾을 수 없습니다.")
    exit()

# 현장 설비 실시간 센서 데이터 수집 (시뮬레이션)
# 현장의 PLC나 OPC-UA 서버를 통해 실시간으로 1건의 데이터가 들어왔다고 가정
incoming_data = {
    'Type': 'H',                       # 제품 등급 (L, M, H)
    'Air temperature [K]': 302.5,
    'Process temperature [K]': 311.2,
    'Rotational speed [rpm]': 1350,    # 평소보다 속도가 비정상적으로 떨어짐
    'Torque [Nm]': 70.0,               # 평소보다 토크가 높음 (과부하 징후)
    'Tool wear [min]': 215             # 공구 마모가 꽤 진행됨
}

df_new = pd.DataFrame([incoming_data])
print("\n[수집된 실시간 센서 데이터]")
# 환경에 따라 display가 없으면 print로 대체
print(df_new) if 'display' in globals() else print(df_new)

# 추론을 위한 데이터 전처리 (파이프라인)
# 학습 모델이 기대하는 7개의 피처와 순서를 정확히 맞춰야 합니다.

# 1. 범주형 변수(Type) One-Hot Encoding 수동 처리
df_new['Type_L'] = 1 if incoming_data['Type'] == 'L' else 0
df_new['Type_M'] = 1 if incoming_data['Type'] == 'M' else 0
df_new = df_new.drop(columns=['Type'])

# 2. 컬럼 순서 재배치 (학습 데이터와 100% 동일한 순서)
expected_cols = [
    'Air temperature [K]', 'Process temperature [K]', 
    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 
    'Type_L', 'Type_M'
]
df_new = df_new[expected_cols]

# 3. 연속형 센서 데이터 스케일링
num_cols = expected_cols[:5]
df_new[num_cols] = scaler.transform(df_new[num_cols])

# 4. PyTorch 텐서 변환
X_tensor = torch.tensor(df_new.astype(np.float32).values, dtype=torch.float32).to(device)
print("\n데이터 전처리 및 텐서 변환 완료")

# AI 모델 결함 진단 수행
with torch.no_grad(): 
    output = model(X_tensor)
    prob = torch.sigmoid(output).item() # 0 ~ 1 사이 확률값으로 변환
    is_fault = prob >= 0.5              # Threshold 0.5 (현장 상황에 따라 조절 가능)

print("\n" + "="*40)
print("[AI 설비 상태 판별 결과] ")
print("="*40)
print(f"▶ 결함 발생 확률: {prob * 100:.2f}%")

if is_fault:
    print("[경고] 비정상 패턴 감지! 즉시 설비 점검이 필요합니다.")
else:
    print("[정상] 설비가 안정적으로 가동 중입니다.")
print("="*40)