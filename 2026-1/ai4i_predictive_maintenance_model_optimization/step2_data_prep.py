# 라이브러리 임포트 및 데이터 로드
from pyexpat import features

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

print("라이브러리 로드 완료")

# 데이터 로드 (EDA 단계와 동일한 원본 데이터)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
df = pd.read_csv(url)
print(f"원본 데이터 로드 완료: {df.shape}")

# 데이터 전처리 (식별자 제거, 인코딩, 데이터 누수 방지)
print("\n데이터 전처리 시작...")

# 1. 불필요한 식별자 컬럼 제거
df_processed = df.drop(columns=['UDI', 'Product ID'])

# 2. 범주형 변수(Type) One-Hot Encoding
df_processed = pd.get_dummies(df_processed, columns=['Type'], drop_first=True)

# 3. 피처(X)와 타겟(y) 분리 및 데이터 누수(Leakage) 방지
target_col = 'Machine failure'
leakage_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF'] # 세부 고장 모드 제거 필수

X = df_processed.drop(columns=[target_col] + leakage_cols)
y = df_processed[target_col]

# 4. Train / Validation / Test 분할 (stratify 적용하여 클래스 비율 유지)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

# 5. 연속형 센서 변수 스케일링 (Train 데이터 기준으로만 fit 적용)
scaler = StandardScaler()
num_cols = [
    'Air temperature [K]', 'Process temperature [K]', 
    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
]

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

print(f"Train 피처 형태: {X_train.shape}")
print(f"Validation 피처 형태: {X_val.shape}")
print(f"Test 피처 형태: {X_test.shape}")

# PyTorch Dataset 정의
class ManufacturingDataset(Dataset):
    def __init__(self, features, labels):
        # DataFrame을 PyTorch FloatTensor로 변환
        self.X = torch.tensor(features.astype(np.float32).values, dtype=torch.float32)
        # 손실 함수(BCE Loss) 계산을 위해 타겟 형태를 [batch_size, 1]로 맞춤
        self.y = torch.tensor(labels.astype(np.float32).values, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = ManufacturingDataset(X_train, y_train)
val_dataset = ManufacturingDataset(X_val, y_val)
test_dataset = ManufacturingDataset(X_test, y_test)

print("\nPyTorch Dataset 생성 완료")

# 클래스 불균형 처리를 위한 DataLoader 구축
batch_size = 64

# 1. 클래스별 가중치 계산 (정상 vs 고장)
class_counts = y_train.value_counts().sort_index()
class_weights = 1.0 / class_counts.values

# 2. 각 샘플별 가중치 할당
sample_weights = [class_weights[int(label)] for label in y_train]
sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

# 3. 오버샘플링을 수행하는 WeightedRandomSampler 정의
sampler = WeightedRandomSampler(
    weights=sample_weights, 
    num_samples=len(sample_weights), 
    replacement=True
)

# 4. DataLoader 생성
# Train Loader: sampler를 통해 고장 데이터를 균형 있게 추출
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, sampler=sampler
)

# Validation/Test Loader: 실제 분포를 평가하기 위해 섞거나 샘플링을 조작하지 않음
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)

test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)

print("DataLoader 구축 완료 (WeightedRandomSampler 적용)")

# XGBoost용 DataFrame export
__all__ = ['train_loader', 'val_loader', 'test_loader', 'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test', 'scaler']

# 배치 데이터 추출 테스트 및 오버샘플링 효과 확인
for batch_X, batch_y in train_loader:
    print(f"\n[Train DataLoader 배치 확인]")
    print(f"- X 텐서 형태: {batch_X.shape}")
    print(f"- y 텐서 형태: {batch_y.shape}")
    print(f"- 1개 배치({batch_size}개) 내 고장(1) 데이터 개수: {int(batch_y.sum().item())}개")
    break