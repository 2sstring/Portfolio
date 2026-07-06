# 라이브러리 임포트 및 저장된 객체 불러오기
import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

print("실시간 추론(Inference) 환경 준비 중...")

# -------------------------------------------------
# 1. 파일 경로 설정
# -------------------------------------------------
model_path = 'models/fault_diagnosis_xgb.json'
scaler_path = 'models/sensor_scaler_xgb.pkl'

# 학습 시 찾은 최적 threshold를 직접 반영
INFERENCE_THRESHOLD = 0.74

# -------------------------------------------------
# 2. 스케일러 로드
# -------------------------------------------------
try:
    scaler = joblib.load(scaler_path)
    print(f"스케일러 로드 완료: {scaler_path}")
except FileNotFoundError:
    print(f"에러: '{scaler_path}' 파일을 찾을 수 없습니다. 모델 학습을 먼저 진행해주세요.")
    raise SystemExit

# -------------------------------------------------
# 3. XGBoost 모델 로드
# -------------------------------------------------
try:
    model = xgb.Booster()
    model.load_model(model_path)
    print(f"모델 로드 완료: {model_path}")
except FileNotFoundError:
    print(f"에러: '{model_path}' 파일을 찾을 수 없습니다.")
    raise SystemExit

# -------------------------------------------------
# 4. 실시간 센서 데이터 입력 (예시)
# -------------------------------------------------
incoming_data = {
    'Type': 'H',                       # 제품 등급 (L, M, H)
    'Air temperature [K]': 302.5,
    'Process temperature [K]': 311.2,
    'Rotational speed [rpm]': 1350,    # 평소보다 속도가 비정상적으로 떨어짐
    'Torque [Nm]': 70.0,               # 평소보다 토크가 높음
    'Tool wear [min]': 215             # 공구 마모 진행
}

df_new = pd.DataFrame([incoming_data])

print("\n[수집된 실시간 센서 데이터]")
print(df_new)

# -------------------------------------------------
# 5. 추론용 데이터 전처리
# -------------------------------------------------

# (1) Type 원핫 인코딩 수동 처리
# 학습 코드에서 drop_first=True를 사용했으므로 H는 기준 클래스
df_new['Type_L'] = 1 if incoming_data['Type'] == 'L' else 0
df_new['Type_M'] = 1 if incoming_data['Type'] == 'M' else 0
df_new = df_new.drop(columns=['Type'])

# (2) 학습 시 사용한 컬럼 순서와 동일하게 정렬
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

# (4) XGBoost 호환을 위해 컬럼명 특수문자 제거
df_new.columns = [
    col.replace('[', '').replace(']', '').replace('<', '').replace('>', '')
    for col in df_new.columns
]

print("\n데이터 전처리 완료")
print(df_new)

# -------------------------------------------------
# 6. DMatrix 변환
# -------------------------------------------------
dnew = xgb.DMatrix(df_new)

# -------------------------------------------------
# 7. AI 모델 추론 수행
# -------------------------------------------------
prob = model.predict(dnew)[0]
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