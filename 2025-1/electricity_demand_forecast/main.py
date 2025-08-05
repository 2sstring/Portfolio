import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# --- 데이터 전처리 함수 ---
def load_and_preprocess(file_path):
    df = pd.read_csv(
        file_path,
        sep=';',
        parse_dates={'Datetime': ['Date', 'Time']},
        dayfirst=True,
        na_values='?',
        low_memory=False
    )
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    df = df.dropna(subset=['Global_active_power'])
    df.set_index('Datetime', inplace=True)
    daily_power = df['Global_active_power'].resample('D').mean()
    daily_power = daily_power.dropna()
    return daily_power

def split_train_test(series, train_ratio=0.8):
    size = int(len(series) * train_ratio)
    train = series[:size]
    test = series[size:]
    return train, test

# --- 시계열 데이터 RNN/LSTM용 시퀀스 생성 함수 ---
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# --- PyTorch 모델 정의 ---
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

class CNN1DModel(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=32, kernel_size=3):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_channels, 1)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        out = self.fc(x[:, :, -1])
        return out

# --- 학습 및 예측 함수 ---
def train_model(model, train_loader, epochs=200, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)
    model.train()
    for epoch in range(1, epochs+1):
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch}/{epochs}], Loss: {epoch_loss/len(train_loader):.6f}")

def predict_model(model, data_loader, device='cpu'):
    model.eval()
    predictions = []
    with torch.no_grad():
        for x_batch, _ in data_loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch)
            predictions.extend(preds.cpu().numpy())
    return np.array(predictions).flatten()

def predict_statsmodel(model_fit, start, end, **kwargs):
    return model_fit.predict(start=start, end=end, **kwargs)

# --- 메인 함수 ---
def main():
    file_path = 'household_power_consumption.txt'
    print("Loading and preprocessing data...")
    data = load_and_preprocess(file_path)
    train_series, test_series = split_train_test(data)
    print(f"Data size: total={len(data)}, train={len(train_series)}, test={len(test_series)}")

    print("Training ARMA model...")
    arma_model = ARIMA(train_series, order=(2,0,2)).fit()
    arma_pred = predict_statsmodel(arma_model, start=len(train_series), end=len(data)-1)

    print("Training ARIMA model...")
    arima_model = ARIMA(train_series, order=(2,1,2)).fit()
    arima_pred = predict_statsmodel(arima_model, start=len(train_series), end=len(data)-1, typ='levels')

    seq_length = 30
    train_np = train_series.values
    test_np = test_series.values
    train_x, train_y = create_sequences(train_np, seq_length)
    test_x, test_y = create_sequences(np.concatenate([train_np[-seq_length:], test_np]), seq_length)

    train_x_tensor = torch.tensor(train_x, dtype=torch.float32).unsqueeze(-1)
    train_y_tensor = torch.tensor(train_y, dtype=torch.float32).unsqueeze(-1)
    test_x_tensor = torch.tensor(test_x, dtype=torch.float32).unsqueeze(-1)
    test_y_tensor = torch.tensor(test_y, dtype=torch.float32).unsqueeze(-1)

    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = TensorDataset(test_x_tensor, test_y_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    print("Training RNN model...")
    rnn_model = RNNModel()
    train_model(rnn_model, train_loader, epochs=200, device=device)
    rnn_pred = predict_model(rnn_model, test_loader, device=device)

    print("Training LSTM model...")
    lstm_model = LSTMModel()
    train_model(lstm_model, train_loader, epochs=200, device=device)
    lstm_pred = predict_model(lstm_model, test_loader, device=device)

    print("Training GRU model...")
    gru_model = GRUModel()
    train_model(gru_model, train_loader, epochs=200, device=device)
    gru_pred = predict_model(gru_model, test_loader, device=device)

    print("Training 1D CNN model...")
    cnn_model = CNN1DModel()
    train_model(cnn_model, train_loader, epochs=200, device=device)
    cnn_pred = predict_model(cnn_model, test_loader, device=device)

    test_true = test_y

    def mse(true, pred):
        return mean_squared_error(true, pred)

    results = {
        "ARMA": mse(test_series[-len(arma_pred):], arma_pred),
        "ARIMA": mse(test_series[-len(arima_pred):], arima_pred),
        "RNN": mse(test_true, rnn_pred),
        "LSTM": mse(test_true, lstm_pred),
        "GRU": mse(test_true, gru_pred),
        "1D CNN": mse(test_true, cnn_pred),
    }

    print("\n===== MSE Results =====")
    for model_name, mse_val in results.items():
        print(f"{model_name}: {mse_val:.6f}")

    # --- 예측 결과 시각화 ---
    aligned_index = test_series.index[-len(test_true):]

    plt.figure(figsize=(14, 6))
    plt.plot(aligned_index, test_true, label='True', linewidth=2)
    plt.plot(aligned_index, rnn_pred, label='RNN', alpha=0.7)
    plt.plot(aligned_index, lstm_pred, label='LSTM', alpha=0.7)
    plt.plot(aligned_index, gru_pred, label='GRU', alpha=0.7)
    plt.plot(aligned_index, cnn_pred, label='1D CNN', alpha=0.7)
    plt.plot(aligned_index, arma_pred, label='ARMA', linestyle='--', alpha=0.8)
    plt.plot(aligned_index, arima_pred, label='ARIMA', linestyle='--', alpha=0.8)
    plt.title("All Models - Prediction vs True")
    plt.xlabel("Date")
    plt.ylabel("Global Active Power")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    model_names = list(results.keys())
    mse_values = list(results.values())
    plt.bar(model_names, mse_values, color='skyblue')
    plt.title("Model Comparison (MSE)")
    plt.ylabel("Mean Squared Error")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
