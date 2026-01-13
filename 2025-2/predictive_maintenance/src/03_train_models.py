# 03_train_models.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import lightgbm as lgb

# ===== 1. 데이터 불러오기 =====
df = pd.read_csv("../data/ai4i_preprocessed.csv")

print("===== 데이터 미리보기 =====")
print(df.head())

feature_cols = ["air_temp", "process_temp", "rotational_speed", "torque", "tool_wear"]
X = df[feature_cols]
y_fail = df["machine_failure"]   # 0/1 이진 분류

print("\n===== Machine failure 분포 =====")
print(y_fail.value_counts())

# ===== 2. 고장 발생한 샘플만 따로 뽑아서 고장 유형 분류용 데이터셋 구성 =====
df_fail = df[df["machine_failure"] == 1].copy()

print("\n===== 고장 샘플의 failure_type 분포 (원본) =====")
print(df_fail["failure_type"].value_counts())

# 4, 5 같이 복합 고장은 제거하고 1(TWF), 2(HDF), 3(PWF)만 사용
df_fail = df_fail[df_fail["failure_type"].isin([1, 2, 3])].copy()

print("\n===== 고장 샘플의 failure_type 분포 (1,2,3만 사용) =====")
print(df_fail["failure_type"].value_counts())

X_type = df_fail[feature_cols]
y_type = df_fail["failure_type"]    # 값: 1, 2, 3

# LightGBM은 보통 라벨을 0,1,2,... 형태로 쓰는 걸 선호하므로 매핑
label_map = {1: 0, 2: 1, 3: 2}
label_map_inv = {v: k for k, v in label_map.items()}  # 나중에 해석용(선택)

y_type_enc = y_type.map(label_map)   # 1,2,3 -> 0,1,2

print("\n===== 인코딩된 failure_type 분포 (0,1,2) =====")
print(y_type_enc.value_counts())

# ===== 3. Train/Test split =====
# 3-1. 고장 여부 이진 분류용
X_train, X_test, y_train_fail, y_test_fail = train_test_split(
    X, y_fail, test_size=0.2, random_state=42, stratify=y_fail
)

# 3-2. 고장 유형 다중 분류용 (stratify는 y_type_enc 사용)
X_train_type, X_test_type, y_train_type, y_test_type = train_test_split(
    X_type, y_type_enc, test_size=0.2, random_state=42, stratify=y_type_enc
)

print("\n===== Train/Test 크기 (failure) =====")
print("X_train:", X_train.shape, " / X_test:", X_test.shape)
print("y_train_fail 분포:")
print(y_train_fail.value_counts())
print("y_test_fail 분포:")
print(y_test_fail.value_counts())

print("\n===== Train/Test 크기 (failure_type) =====")
print("X_train_type:", X_train_type.shape, " / X_test_type:", X_test_type.shape)
print("y_train_type 분포:")
print(y_train_type.value_counts())
print("y_test_type 분포:")
print(y_test_type.value_counts())

# ===== 4. XGBoost - Machine failure (0/1) 이진 분류 =====
pos = y_train_fail.sum()
neg = len(y_train_fail) - pos
scale_pos_weight = neg / pos
print("\nscale_pos_weight (for XGBoost):", scale_pos_weight)

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

xgb_fail.fit(X_train, y_train_fail)
y_pred_fail = xgb_fail.predict(X_test)

print("\n=== Machine Failure (XGBoost) ===")
print(confusion_matrix(y_test_fail, y_pred_fail))
print(classification_report(y_test_fail, y_pred_fail))

# ===== 5. LightGBM - 고장 유형 (TWF/HDF/PWF) 다중 분류 =====
lgb_type = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    objective="multiclass",
    num_class=3,   # 클래스: 0(TWF), 1(HDF), 2(PWF)
    random_state=42
)

lgb_type.fit(X_train_type, y_train_type)
y_pred_type = lgb_type.predict(X_test_type)

print("\n=== Failure Type (LightGBM) ===")
print(confusion_matrix(y_test_type, y_pred_type))
print(
    classification_report(
        y_test_type,
        y_pred_type,
        target_names=["TWF(0)", "HDF(1)", "PWF(2)"]
    )
)
