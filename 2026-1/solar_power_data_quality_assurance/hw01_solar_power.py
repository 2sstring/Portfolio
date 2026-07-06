import pandas as pd
import numpy as np

# =========================================================
# 0. 데이터 로드
# =========================================================
file_path = "01. 전국 태양광 발전량 예측값 학습데이터.csv"
df = pd.read_csv(file_path, encoding="cp949")

print("=" * 70)
print("[0] 원본 데이터")
print("원본 shape:", df.shape)
print("원본 컬럼:", list(df.columns))
print("=" * 70)


# =========================================================
# 1. 유일성 (Uniqueness)
#    - (시도명, 발전일자) 기준 중복 제거
# =========================================================
before_rows = len(df)

df = df.drop_duplicates(subset=["시도명", "발전일자"]).copy()

after_rows = len(df)
print("\n[1] 유일성")
print("중복 제거 전 행 수:", before_rows)
print("중복 제거 후 행 수:", after_rows)
print("제거된 중복 행 수:", before_rows - after_rows)


# =========================================================
# 2. 완전성 (Completeness)
#    2-1) 결측률 80% 이상 feature 제거
#    2-2) 대기권밖일사량계산값 == 0 인 경우,
#         일조(hr), 일사량 결측값만 0으로 대체
#    2-3) 이후 남아 있는 결측 행 제거
# =========================================================
print("\n[2] 완전성")

missing_ratio = df.isna().mean()
drop_cols = missing_ratio[missing_ratio >= 0.8].index.tolist()

print("결측률 80% 이상 feature:", drop_cols)

df = df.drop(columns=drop_cols).copy()

# 2-2. 조건부 결측 대체
# 야간 조건
mask_night = (df["대기권밖일사량계산값"] == 0)

# 대체 대상 행 기준 마스크
mask_sunshine = mask_night & df["일조(hr)"].isna()
mask_radiation = mask_night & df["일사량"].isna()

# 행 기준 통합 (OR)
mask_imputed_row = mask_sunshine | mask_radiation

# 행 개수
row_imputed = mask_imputed_row.sum()

print("[완전성 - 0 대체 행 기준]")
print(f" - 대체된 행 수: {row_imputed}")

# 대기권밖일사량계산값 == 0 이고, 해당 컬럼이 결측인 경우만 0 대체
if "일조(hr)" in df.columns:
    mask_sunshine_fill = (df["대기권밖일사량계산값"] == 0) & (df["일조(hr)"].isna())
    df.loc[mask_sunshine_fill, "일조(hr)"] = 0

if "일사량" in df.columns:
    mask_radiation_fill = (df["대기권밖일사량계산값"] == 0) & (df["일사량"].isna())
    df.loc[mask_radiation_fill, "일사량"] = 0

print("조건부 0 대체 후 결측 개수:")
print(df.isna().sum())

# 2-3. 잔여 결측 행 제거
before_rows = len(df)
df = df.dropna().copy()
after_rows = len(df)

print("잔여 결측 행 제거 전 행 수:", before_rows)
print("잔여 결측 행 제거 후 행 수:", after_rows)
print("제거된 행 수:", before_rows - after_rows)


# =========================================================
# 3. 유효성 (Validity)
#    - 형식, 자료형, 물리적 허용 범위 검증
# =========================================================
print("\n[3] 유효성")

# 3-1. 발전일자 datetime 변환
before_rows = len(df)
df["발전일자"] = pd.to_datetime(df["발전일자"], errors="coerce")
df = df.dropna(subset=["발전일자"]).copy()
after_rows = len(df)

print("발전일자 datetime 변환 불가 제거:", before_rows - after_rows)

# 3-2. 문자열 컬럼 정리
df["시도명"] = df["시도명"].astype(str).str.strip()
df["윤년여부"] = df["윤년여부"].astype(str).str.strip()

# 3-3. 수치형 컬럼 강제 변환
numeric_cols = [
    "설비용량(MW)", "기온", "습도", "풍속",
    "적운량(10분위)", "적운량(3분위)",
    "일조(hr)", "대기권밖일사량계산값", "일사량", "발전량(MWh)"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 변환 과정에서 생긴 결측 제거
before_rows = len(df)
df = df.dropna(subset=[col for col in numeric_cols if col in df.columns]).copy()
after_rows = len(df)

print("수치형 변환 불가 제거:", before_rows - after_rows)

# 3-4. 물리적 허용 범위 적용
valid_mask = pd.Series(True, index=df.index)

# 설비용량 > 0
valid_mask &= df["설비용량(MW)"] > 0

# 발전량 >= 0
valid_mask &= df["발전량(MWh)"] >= 0

# 기온: -30 ~ 50
valid_mask &= df["기온"].between(-30, 50)

# 습도: 0 ~ 100
valid_mask &= df["습도"].between(0, 100)

# 풍속 >= 0
valid_mask &= df["풍속"] >= 0

# 적운량(10분위): 0 ~ 10
if "적운량(10분위)" in df.columns:
    valid_mask &= df["적운량(10분위)"].between(0, 10)

# 적운량(3분위): 0 ~ 3
if "적운량(3분위)" in df.columns:
    valid_mask &= df["적운량(3분위)"].between(0, 3)

# 일조(hr): 0 ~ 1
if "일조(hr)" in df.columns:
    valid_mask &= df["일조(hr)"].between(0, 1)

# 대기권밖일사량계산값 >= 0
valid_mask &= df["대기권밖일사량계산값"] >= 0

# 일사량 >= 0
if "일사량" in df.columns:
    valid_mask &= df["일사량"] >= 0

before_rows = len(df)
df = df[valid_mask].copy()
after_rows = len(df)

print("유효성 범위 위반 제거:", before_rows - after_rows)


# =========================================================
# 4. 일관성 (Consistency)
#    - 대기권밖일사량계산값 == 0 AND 일사량 == 0 이면
#      발전량(MWh) = 0 으로 보정
# =========================================================
print("\n[4] 일관성")

mask_night = (
    (df["대기권밖일사량계산값"] == 0) &
    (df["일사량"] == 0)
)

count_before_nonzero = (df.loc[mask_night, "발전량(MWh)"] != 0).sum()
df.loc[mask_night, "발전량(MWh)"] = 0

print("야간 조건에서 0으로 보정된 발전량 건수:", count_before_nonzero)


# =========================================================
# 5. 정확성 (Accuracy)
#    - 설비용량 대비 과도한 발전량 이상치 제거
#    - 기준: 발전량(MWh) <= 설비용량(MW)
#      (과제 보고서에 임계값 근거를 같이 설명하는 것을 권장)
# =========================================================
print("\n[5] 정확성")

before_rows = len(df)

# 1배 기준 사용
accuracy_mask = df["발전량(MWh)"] <= (df["설비용량(MW)"])
removed_accuracy = (~accuracy_mask).sum()

df = df[accuracy_mask].copy()
after_rows = len(df)

print("정확성 기준으로 제거된 이상치 수:", removed_accuracy)
print("정확성 처리 후 행 수:", after_rows)


# =========================================================
# 6. 최종 결과 저장
# =========================================================
output_path = "solar_preprocessed.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print("\n" + "=" * 70)
print("[6] 최종 결과")
print("최종 shape:", df.shape)
print("저장 파일:", output_path)
print("=" * 70)