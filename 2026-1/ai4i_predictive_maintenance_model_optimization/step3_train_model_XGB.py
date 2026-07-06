# 라이브러리 임포트 및 전처리 데이터 불러오기
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

from torch.utils.tensorboard import SummaryWriter  # TensorBoard 기록용

from xgboost import XGBClassifier
import xgboost as xgb

print("XGBoost 로드 성공")

# 전처리 모듈 임포트
try:
    from step2_data_prep import X_train, X_val, X_test, y_train, y_val, y_test, scaler
    print("데이터 전처리 모듈(step2_data_prep) 로드 성공")
except ModuleNotFoundError:
    print("에러: 'step2_data_prep.py' 파일을 찾을 수 없습니다. 같은 폴더에 있는지 확인해주세요.")
    exit()

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# 피처 이름에서 특수 문자 제거 (XGBoost 호환성)
X_train = X_train.copy()
X_val = X_val.copy()
X_test = X_test.copy()

X_train.columns = [col.replace('[', '').replace(']', '').replace('<', '').replace('>', '') for col in X_train.columns]
X_val.columns = [col.replace('[', '').replace(']', '').replace('<', '').replace('>', '') for col in X_val.columns]
X_test.columns = [col.replace('[', '').replace(']', '').replace('<', '').replace('>', '') for col in X_test.columns]

# -------------------------------------------------
# 1. Grid Search
# -------------------------------------------------
param_grid = {
    'max_depth': [5],
    'learning_rate': [0.05],
    'n_estimators': [900],
    'subsample': [1.0],
    'colsample_bytree': [0.9],
    'min_child_weight': [2],
    'scale_pos_weight': [28.5],
    'gamma': [0.03],
    'reg_alpha': [0],
    'reg_lambda': [1.0]
}

xgb_model = XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    eval_metric='logloss'
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=cv,
    scoring='f1',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV F1: {grid_search.best_score_:.4f}")

best_params = grid_search.best_params_

# -------------------------------------------------
# 2. DMatrix 생성
# -------------------------------------------------
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# -------------------------------------------------
# 3. TensorBoard 설정
# -------------------------------------------------
log_dir = "runs/fault_diagnosis_experiment_xgb"
writer = SummaryWriter(log_dir)
print(f"TensorBoard 로그 디렉토리 설정: {log_dir}")

# -------------------------------------------------
# 4. 사용자 정의 Callback
#    매 round마다 train/val loss + F1 기록
# -------------------------------------------------
class TensorBoardXGBoostCallback(xgb.callback.TrainingCallback):
    def __init__(self, writer, dtrain, dval, y_train, y_val, threshold=0.5):
        self.writer = writer
        self.dtrain = dtrain
        self.dval = dval
        self.y_train = np.array(y_train)
        self.y_val = np.array(y_val)
        self.threshold = threshold

        self.train_losses = []
        self.val_losses = []
        self.train_f1s = []
        self.val_f1s = []

    def after_iteration(self, model, epoch, evals_log):
        # logloss 가져오기
        train_loss = evals_log['train']['logloss'][-1]
        val_loss = evals_log['validation']['logloss'][-1]

        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        # 현재 round까지의 모델로 예측
        train_probs = model.predict(self.dtrain)
        val_probs = model.predict(self.dval)

        train_preds = (train_probs >= self.threshold).astype(int)
        val_preds = (val_probs >= self.threshold).astype(int)

        train_f1 = f1_score(self.y_train, train_preds, zero_division=0)
        val_f1 = f1_score(self.y_val, val_preds, zero_division=0)

        self.train_f1s.append(train_f1)
        self.val_f1s.append(val_f1)

        # TensorBoard 기록
        self.writer.add_scalars('Loss', {'Train': train_loss, 'Validation': val_loss}, epoch)
        self.writer.add_scalar('Metrics/Validation_F1', val_f1, epoch)

        # 콘솔 출력
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Round [{epoch+1:4d}/{best_params['n_estimators']}] | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}"
            )

        return False  # 학습 계속

# callback 생성
tb_callback = TensorBoardXGBoostCallback(
    writer=writer,
    dtrain=dtrain,
    dval=dval,
    y_train=y_train,
    y_val=y_val,
    threshold=0.5   # 일단 기본 threshold로 기록
)

# -------------------------------------------------
# 5. 학습
# -------------------------------------------------
evals_result = {}

bst = xgb.train(
    params={
        'objective': 'binary:logistic',
        'scale_pos_weight': best_params['scale_pos_weight'],
        'max_depth': best_params['max_depth'],
        'learning_rate': best_params['learning_rate'],
        'subsample': best_params.get('subsample', 1.0),
        'colsample_bytree': best_params.get('colsample_bytree', 1.0),
        'gamma': best_params.get('gamma', 0),
        'min_child_weight': best_params.get('min_child_weight', 1),
        'reg_alpha': best_params.get('reg_alpha', 0),
        'reg_lambda': best_params.get('reg_lambda', 1.0),
        'random_state': 42,
        'eval_metric': 'logloss'
    },
    dtrain=dtrain,
    num_boost_round=best_params['n_estimators'],   # 반복횟수 고정
    evals=[(dtrain, 'train'), (dval, 'validation')],
    evals_result=evals_result,
    callbacks=[tb_callback],
    verbose_eval=False
)

writer.close()
print("학습 및 TensorBoard 기록 완료!")

# -------------------------------------------------
# 6. TensorBoard 보고 threshold 선택
#    학습 후 별도로 validation threshold 최적화
# -------------------------------------------------
val_probs = bst.predict(dval)
thresholds = np.linspace(0.0, 1.0, 101)
f1_scores = [f1_score(y_val, (val_probs >= t).astype(int), zero_division=0) for t in thresholds]
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"Best validation threshold: {best_threshold:.2f}, Validation F1: {f1_scores[best_idx]:.4f}")

# -------------------------------------------------
# 7. Test 평가
# -------------------------------------------------
test_probs = bst.predict(dtest)
test_preds = (test_probs >= best_threshold).astype(int)

acc = accuracy_score(y_test, test_preds)
prec = precision_score(y_test, test_preds, zero_division=0)
rec = recall_score(y_test, test_preds, zero_division=0)
f1 = f1_score(y_test, test_preds, zero_division=0)
auc = roc_auc_score(y_test, test_probs)

print("\n[최종 테스트 데이터셋 평가 결과]")
print(f"Accuracy (정확도):  {acc:.4f}")
print(f"Precision (정밀도): {prec:.4f}")
print(f"Recall (재현율):    {rec:.4f}")
print(f"F1-Score:           {f1:.4f}")
print(f"ROC-AUC:            {auc:.4f}")

# -------------------------------------------------
# 8. 모델 저장
# -------------------------------------------------
os.makedirs('models', exist_ok=True)
model_path = 'models/fault_diagnosis_xgb.json'
scaler_path = 'models/sensor_scaler_xgb.pkl'

bst.save_model(model_path)
joblib.dump(scaler, scaler_path)

print(f"\n[저장 완료] 모델('{model_path}')와 스케일러('{scaler_path}')가 저장되었습니다.")

# -------------------------------------------------
# 9. 평가 결과 시각화
# -------------------------------------------------
print("\n평가 결과 시각화 그래프를 생성합니다...")

plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1) Confusion Matrix
cm = confusion_matrix(y_test, test_preds)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
    xticklabels=['Normal (0)', 'Fault (1)'],
    yticklabels=['Normal (0)', 'Fault (1)'],
    cbar=False, annot_kws={"size": 14}
)
axes[0].set_title('Confusion Matrix', fontsize=14, pad=10)
axes[0].set_ylabel('Actual Status', fontsize=12)
axes[0].set_xlabel('Predicted Status', fontsize=12)

# 2) Evaluation Metrics Summary
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
metrics_values = [acc, prec, rec, f1, auc]

sns.barplot(x=metrics_names, y=metrics_values, ax=axes[1], palette='viridis')
axes[1].set_title('Evaluation Metrics Summary', fontsize=14, pad=10)
axes[1].set_ylim(0, 1.1)

for i, v in enumerate(metrics_values):
    axes[1].text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# 3) ROC Curve
fpr, tpr, _ = roc_curve(y_test, test_probs)
axes[2].plot(fpr, tpr, color='crimson', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
axes[2].plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', alpha=0.7)
axes[2].set_xlim([-0.02, 1.0])
axes[2].set_ylim([0.0, 1.05])
axes[2].set_xlabel('False Positive Rate (FPR)', fontsize=12)
axes[2].set_ylabel('True Positive Rate (TPR)', fontsize=12)
axes[2].set_title('Receiver Operating Characteristic (ROC)', fontsize=14, pad=10)
axes[2].legend(loc="lower right", fontsize=11)

plt.tight_layout()
plt.show()