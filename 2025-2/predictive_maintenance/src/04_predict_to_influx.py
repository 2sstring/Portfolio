# 04_predict_to_influx.py

import pandas as pd
from xgboost import XGBClassifier
import lightgbm as lgb

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# ===== InfluxDB 설정 (docker-compose와 동일하게) =====
URL = "http://localhost:8086"
ORG = "my-org"
BUCKET = "ai4i"
TOKEN = "change-me"

# ===== 1. 전처리된 CSV 불러오기 =====
df = pd.read_csv("../data/ai4i_preprocessed.csv")

# timestamp를 실제 시간 타입으로 변환
df["timestamp"] = pd.to_datetime(df["timestamp"])

print("===== 데이터 미리보기 =====")
print(df.head())

# 사용할 특징(feature) 컬럼
feature_cols = ["air_temp", "process_temp", "rotational_speed", "torque", "tool_wear"]
X = df[feature_cols]
y_fail = df["machine_failure"]

# ===== 2. 고장 유형 라벨 준비 (TWF/HDF/PWF만 사용) =====
df_fail = df[df["machine_failure"] == 1].copy()
df_fail = df_fail[df_fail["failure_type"].isin([1, 2, 3])].copy()

X_type = df_fail[feature_cols]
y_type = df_fail["failure_type"]

# failure_type: 1(TWF), 2(HDF), 3(PWF) → 0,1,2 로 인코딩
label_map = {1: 0, 2: 1, 3: 2}
label_map_inv = {v: k for k, v in label_map.items()}

y_type_enc = y_type.map(label_map)

print("\n===== 고장 유형 라벨 분포 (0,1,2) =====")
print(y_type_enc.value_counts())

# ===== 3. 모델 학습 (이번에는 전체 데이터 기준으로 학습해서 시각화용 예측 생성) =====

# 3-1. XGBoost: Machine failure(0/1) 분류
pos = y_fail.sum()
neg = len(y_fail) - pos
scale_pos_weight = neg / pos
print("\nscale_pos_weight:", scale_pos_weight)

xgb_fail = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

xgb_fail.fit(X, y_fail)

# 3-2. LightGBM: 고장 유형 다중 분류 (고장 샘플 중 1,2,3만 사용)
lgb_type = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    objective="multiclass",
    num_class=3,
    random_state=42
)
lgb_type.fit(X_type, y_type_enc)

# ===== 4. 전체 시간 구간에 대해 예측 생성 =====

# 4-1. 전체 구간에 대해 고장 확률 / 고장 여부 예측
proba_fail = xgb_fail.predict_proba(X)[:, 1]       # 고장(1)일 확률
pred_fail = (proba_fail >= 0.5).astype(int)        # 0.5 기준 이진 예측

# 4-2. 고장으로 예측된 시점들에 대해 고장 유형 예측
pred_type_all = [0] * len(df)   # 0 = 예측된 고장 없음 or 미정

# pred_fail == 1 인 인덱스들만 골라서 LightGBM에 넣어서 유형 예측
import numpy as np

fail_indices = np.where(pred_fail == 1)[0]
if len(fail_indices) > 0:
    X_fail_pred = X.iloc[fail_indices]
    y_type_pred_enc = lgb_type.predict(X_fail_pred)     # 0,1,2 값
    y_type_pred = [label_map_inv[int(v)] for v in y_type_pred_enc]  # 다시 1,2,3으로 복원

    for idx, t_label in zip(fail_indices, y_type_pred):
        pred_type_all[idx] = t_label

# 리스트 → 시리즈로 변환
df["pred_failure"] = pred_fail
df["prob_failure"] = proba_fail
df["pred_failure_type"] = pred_type_all

print("\n===== 예측 결과 미리보기 =====")
print(df[["timestamp", "machine_failure", "pred_failure", "prob_failure", "failure_type", "pred_failure_type"]].head(20))

# ===== 5. 예측 결과를 InfluxDB에 쓰기 =====

client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)

measurement_name = "prediction"

count = 0
for _, row in df.iterrows():
    p = (
        Point(measurement_name)
        .tag("model", "xgb_lgbm")                         # 어떤 모델인지 태그
        .field("prob_failure", float(row["prob_failure"]))
        .field("pred_failure", int(row["pred_failure"]))
        .field("pred_failure_type", int(row["pred_failure_type"]))
        .time(row["timestamp"], WritePrecision.NS)
    )
    write_api.write(bucket=BUCKET, org=ORG, record=p)
    count += 1

print(f"\n총 {count}개 예측 레코드를 InfluxDB 'prediction' measurement에 업로드 완료!")

client.close()
