# 라이브러리 임포트
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import time


# 데이터 불러오기
df = pd.read_csv("AirQualityUCI.csv", sep=';', decimal=',')

# 마지막 빈 열 제거 + 결측 행 제거
df = df.dropna(how='all', axis=1)
df = df.dropna()

# 날짜-시간 합치기
df["Time"] = df["Time"].astype(str).str.replace(".", ":", regex=False)
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors='coerce')
df = df.dropna(subset=["Datetime"])  # 날짜 파싱 실패한 행 제거
df = df.set_index("Datetime")
df = df.drop(columns=["Date", "Time"])

# 정수 변환 및 결측치(-200) 제거
df = df.apply(pd.to_numeric, errors='coerce')
df[df == -200] = pd.NA
df = df.dropna()

plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지

# 주요 변수들 간의 상관관계 행렬
features_of_interest = [
    "CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)",
    "PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)",
    "T", "RH", "AH"
]

corr = df[features_of_interest].corr()

# 히트맵 그리기 (matplotlib만 사용)
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(corr, cmap="coolwarm")
plt.title("주요 변수 간 상관관계", y=1.15)
fig.colorbar(cax)

# x축 라벨을 아래로 이동
ax.xaxis.set_ticks_position('bottom')
ax.xaxis.set_label_position('bottom')

# 축 레이블 설정
ax.set_xticks(range(len(features_of_interest)))
ax.set_yticks(range(len(features_of_interest)))
ax.set_xticklabels(features_of_interest, rotation=90)
ax.set_yticklabels(features_of_interest)

# 상관계수 숫자 표시
# 절댓값이 0.7 이상인 상관계수는 흰색 텍스트로 출력
for (i, j), val in np.ndenumerate(corr.values):
    color = 'white' if abs(val) >= 0.7 else 'black'
    ax.text(j, i, f"{val:.2f}", ha='center', va='center', color=color)

plt.tight_layout()


target_var = "C6H6(GT)"

# 상관관계 행렬
corr = df.corr()

# 타깃 변수인 'C6H6(GT)' 기준 상관계수 추출
target_corr = corr['C6H6(GT)'].drop('C6H6(GT)')

# 기준 설정
# 상관관계 높은 변수 그룹 (절대값 0.9 이상)
high_corr_vars = target_corr[abs(target_corr) >= 0.9].index.tolist()
# 상관관계 낮은 변수 그룹 (절대값 0.5 이하)
low_corr_vars = target_corr[abs(target_corr) <= 0.5].index.tolist()

print("상관관계 높은 변수 그룹:", high_corr_vars)
print("상관관계 낮은 변수 그룹:", low_corr_vars)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_data(df, feature_vars, target_var, seq_length=24):
    # 타깃 변수를 포함한 전체 입력 변수 구성
    feature_vars_with_target = feature_vars.copy()
    if target_var not in feature_vars_with_target:
        feature_vars_with_target.append(target_var)

    data = df[feature_vars_with_target].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # 타깃 변수의 위치 인덱스 (스케일된 배열 기준)
    target_idx = feature_vars_with_target.index(target_var)

    X, y = [], []
    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i:i+seq_length, :len(feature_vars)])  # 입력은 원래 feature만 사용
        y.append(data_scaled[i+seq_length, target_idx])  # 타깃은 타깃 변수 위치에서 추출
    X = np.array(X)
    y = np.array(y)
    return X, y, scaler


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class CNN1DModel(nn.Module):
    def __init__(self, input_size, num_filters=64, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features) -> conv1d expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x


def train_model(model, criterion, optimizer, train_loader, device, epochs=100):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # 10 에포크 단위로 출력
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")


def evaluate_and_plot(model, X_test, y_test, device, model_name="Model"):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
        outputs = model(inputs).squeeze().cpu().numpy()
    mse = np.mean((outputs - y_test)**2)
    mae = np.mean(np.abs(outputs - y_test))
    print(f"{model_name} - Test MSE: {mse:.4f}, MAE: {mae:.4f}")
    return mse, mae, outputs


def run_model(df, feature_vars, target_var, model_type="LSTM", seq_len=24, epochs=100):
    feature_vars_wo_target = [f for f in feature_vars if f != target_var]

    X, y, scaler = prepare_data(df, feature_vars_wo_target, target_var, seq_len)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                            torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    input_size = len(feature_vars_wo_target)

    if model_type == "RNN":
        model = RNNModel(input_size).to(device)
    elif model_type == "LSTM":
        model = LSTMModel(input_size).to(device)
    elif model_type == "GRU":
        model = GRUModel(input_size).to(device)
    elif model_type == "CNN1D":
        model = CNN1DModel(input_size).to(device)
    else:
        raise ValueError("Unsupported model_type")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()  # 학습 시작 시간
    train_model(model, criterion, optimizer, train_loader, device, epochs)
    end_time = time.time()    # 학습 종료 시간

    print(f"{model_type} 학습 시간: {end_time - start_time:.2f}초")

    mse, mae, outputs = evaluate_and_plot(model, X_test, y_test, device, model_name=model_type)

    return model, scaler, outputs, y_test


# --- 여러 모델 예측 결과 한 그래프에 비교 시각화 ---
def plot_all_models_predictions(y_true, preds_dict, title="Prediction vs Actual"):
    plt.figure(figsize=(14,6))
    plt.plot(y_true, label="Actual", color='black', linewidth=2)
    for model_name, preds in preds_dict.items():
        plt.plot(preds, label=model_name)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("C6H6(GT)")
    plt.legend()


# --- 높은/낮은 상관관계 변수 그룹 MSE 비교 바 그래프 ---
def plot_model_mse_combined(mse_high, mse_low, title="Model MSE Comparison (High vs Low Correlation)"):
    models = list(mse_high.keys())
    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width/2, [mse_high[m] for m in models], width, label='High Corr Vars', color='skyblue')
    bars2 = plt.bar(x + width/2, [mse_low[m] for m in models], width, label='Low Corr Vars', color='lightcoral')

    plt.xticks(x, models)
    plt.ylabel("MSE")
    plt.title(title)
    plt.ylim(0, max(max(mse_high.values()), max(mse_low.values())) * 1.2)
    plt.legend()

    # 숫자 표시
    for bar in bars1 + bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval * 1.01, f"{yval:.4f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()


# --- 상관관계 높은 변수 그룹 모델 학습/예측/평가 및 시각화 예시 ---
results_high = {}
mse_high = {}

print("=== 상관관계 높은 변수 그룹 ===")
for model_type in ["RNN", "LSTM", "GRU", "CNN1D"]:
    print(f"Training & Evaluating {model_type}...")
    model, scaler, preds, y_test = run_model(df, high_corr_vars, target_var, model_type=model_type, epochs=100)
    mse = np.mean((preds - y_test)**2)
    results_high[model_type] = preds
    mse_high[model_type] = mse

plot_all_models_predictions(y_test, results_high, title="High Correlation Variables - Model Predictions")

# --- 상관관계 낮은 변수 그룹 모델 학습/예측/평가 및 시각화 예시 ---
results_low = {}
mse_low = {}

print("=== 상관관계 낮은 변수 그룹 ===")
for model_type in ["RNN", "LSTM", "GRU", "CNN1D"]:
    print(f"Training & Evaluating {model_type}...")
    model, scaler, preds, y_test = run_model(df, low_corr_vars, target_var, model_type=model_type, epochs=100)
    mse = np.mean((preds - y_test)**2)
    results_low[model_type] = preds
    mse_low[model_type] = mse

plot_all_models_predictions(y_test, results_low, title="Low Correlation Variables - Model Predictions")

# 통합 바 그래프 출력
plot_model_mse_combined(mse_high, mse_low)