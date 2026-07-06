# 라이브러리 임포트 및 저장된 객체 불러오기
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib

print("실시간 추론(Inference) 환경 준비 중...")

# 학습 장치(Device) 설정
device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

# -------------------------------------------------
# 1. 모델 아키텍처 재정의 (학습 시와 동일해야 함)
# -------------------------------------------------
class FaultDiagnosisMLP(nn.Module):
    def __init__(self, input_dim: int):
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

# -------------------------------------------------
# 2. 파일 경로 설정 (개선된 MLP 학습 코드와 일치)
# -------------------------------------------------
model_path = 'models/fault_diagnosis_mlp.pth'
scaler_path = 'models/sensor_scaler_mlp.pkl'

# 학습 시 사용한 최적 threshold를 알고 있다면 여기에 반영
INFERENCE_THRESHOLD = 0.93

# -------------------------------------------------
# 3. 스케일러 로드
# -------------------------------------------------
try:
    scaler = joblib.load(scaler_path)
    print(f"스케일러 로드 완료: {scaler_path}")
except FileNotFoundError:
    print(f"에러: '{scaler_path}' 파일을 찾을 수 없습니다. 모델 학습을 먼저 진행해주세요.")
    raise SystemExit

# -------------------------------------------------
# 4. 모델 가중치 로드
# 입력 차원: 7개 (연속형 5개 + 원핫 2개)
# -------------------------------------------------
INPUT_DIM = 7
model = FaultDiagnosisMLP(INPUT_DIM).to(device)

try:
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()  # 추론 모드 전환
    print(f"모델 로드 및 평가 모드(eval) 전환 완료: {model_path}")
except FileNotFoundError:
    print(f"에러: '{model_path}' 파일을 찾을 수 없습니다.")
    raise SystemExit

# -------------------------------------------------
# 5. 실시간 센서 데이터 입력 (예시)
# -------------------------------------------------
incoming_data = {
    'Type': 'H',                        # L, M, H
    'Air temperature [K]': 302.5,
    'Process temperature [K]': 311.2,
    'Rotational speed [rpm]': 1350,
    'Torque [Nm]': 70.0,
    'Tool wear [min]': 215
}

df_new = pd.DataFrame([incoming_data])

print("\n[수집된 실시간 센서 데이터]")
print(df_new)

# -------------------------------------------------
# 6. 추론용 데이터 전처리
# 학습 시 입력 피처와 순서를 정확히 맞춰야 함
# -------------------------------------------------

# (1) Type 원핫 인코딩 수동 처리
# 학습 코드에서 drop_first=True를 사용했으므로 H는 기준 클래스
df_new['Type_L'] = 1 if incoming_data['Type'] == 'L' else 0
df_new['Type_M'] = 1 if incoming_data['Type'] == 'M' else 0
df_new = df_new.drop(columns=['Type'])

# (2) 컬럼 순서 재배치
expected_cols = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
    'Type_L',
    'Type_M'
]

df_new = df_new[expected_cols]

# (3) 연속형 변수 스케일링
num_cols = expected_cols[:5]
df_new[num_cols] = scaler.transform(df_new[num_cols])

# (4) PyTorch Tensor 변환
X_tensor = torch.tensor(df_new.astype(np.float32).values, dtype=torch.float32).to(device)

print("\n데이터 전처리 및 텐서 변환 완료")
print(df_new)

# -------------------------------------------------
# 7. AI 모델 추론 수행
# -------------------------------------------------
with torch.no_grad():
    output = model(X_tensor)
    prob = torch.sigmoid(output).item()
    is_fault = prob >= INFERENCE_THRESHOLD

print("\n" + "=" * 45)
print("[AI 설비 상태 판별 결과]")
print("=" * 45)
print(f"▶ 결함 발생 확률: {prob * 100:.2f}%")
print(f"▶ 적용 threshold: {INFERENCE_THRESHOLD:.2f}")

if is_fault:
    print("[경고] 비정상 패턴 감지! 즉시 설비 점검이 필요합니다.")
else:
    print("[정상] 설비가 안정적으로 가동 중입니다.")
print("=" * 45)