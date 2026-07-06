import pandas as pd
import numpy as np

# =========================
# 0. 데이터 로드
# =========================
df = pd.read_csv("uci-secom.csv")

print("=" * 60)
print("[0] 원본 데이터")
print("원본 데이터 shape:", df.shape)
print("분석 feature 수:", len([col for col in df.columns if col not in ["Time", "Pass/Fail"]]))
print(df["Pass/Fail"].value_counts())
print("=" * 60)

# =========================
# 1. 유일성 (Uniqueness)
#    - 완전 중복 row 제거
# =========================
df_uni = df.drop_duplicates()

print("\n[1] 유일성 처리")
print("중복 제거 전 shape:", df.shape)
print("중복 제거 후 shape:", df_uni.shape)
print("분석 feature 수:", len([col for col in df_uni.columns if col not in ["Time", "Pass/Fail"]]))
print("제거된 중복 row 수:", len(df) - len(df_uni))

# =========================
# 2. 완전성 (Completeness)
#    - 결측률 0.1 이상 feature 제거
#    - 남은 결측 row 제거
# =========================
meta_cols = ["Time", "Pass/Fail"]
feature_cols = [col for col in df_uni.columns if col not in meta_cols]

missing_ratio = df_uni[feature_cols].isnull().mean()

# 결측률 0.1 미만 feature만 유지
kept_features = missing_ratio[missing_ratio < 0.1].index.tolist()

cols_to_keep = ["Time"] + kept_features + ["Pass/Fail"]
df_com_step1 = df_uni[cols_to_keep].copy()

# 남은 결측 row 제거
df_com = df_com_step1.dropna().copy()

print("\n[2] 완전성 처리")
print("결측률 0.1 이상으로 제거된 feature 수:", len(feature_cols) - len(kept_features))
print("완전성 1단계 후 shape:", df_com_step1.shape)
print("분석 feature 수:", len([col for col in df_com_step1.columns if col not in ["Time", "Pass/Fail"]]))
print("결측 row 제거 후 shape:", df_com.shape)
print("분석 feature 수:", len([col for col in df_com.columns if col not in ["Time", "Pass/Fail"]]))
print("완전성 단계에서 제거된 row 수:", len(df_com_step1) - len(df_com))
print(df_com["Pass/Fail"].value_counts())

# =========================
# 3. 유효성 (Validity)
#    - IQR 기반 이상치 판정
#    - row별 이상치 비율 0.04 이상 제거
# =========================
validity_feature_cols = [col for col in df_com.columns if col not in meta_cols]

Q1 = df_com[validity_feature_cols].quantile(0.25)
Q3 = df_com[validity_feature_cols].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 각 값이 이상치인지 여부
outlier_mask = (df_com[validity_feature_cols] < lower_bound) | (df_com[validity_feature_cols] > upper_bound)

# row별 이상치 개수 / 비율
outlier_count = outlier_mask.sum(axis=1)
outlier_ratio = outlier_count / len(validity_feature_cols)

# 최종 유효성 기준
OUTLIER_RATIO_THRESHOLD = 0.04

df_valid = df_com[outlier_ratio < OUTLIER_RATIO_THRESHOLD].copy()

print("\n[3] 유효성 처리 (IQR)")
print("이상치 비율 기준:", OUTLIER_RATIO_THRESHOLD)
print("유효성 처리 후 shape:", df_valid.shape)
print("분석 feature 수:", len([col for col in df_valid.columns if col not in ["Time", "Pass/Fail"]]))
print("유효성 단계에서 제거된 row 수:", len(df_com) - len(df_valid))
print(df_valid["Pass/Fail"].value_counts())

# =========================
# 4. 일관성 (Consistency)
#    - 데이터 타입 검사 숫자가 아닌 컬럼 제거
#    - 분산이 0인 상수 feature 제거
# =========================

print("\n[4] 일관성 처리")
print("\n[4-1] 데이터 타입 검사")

meta_cols = ["Time", "Pass/Fail"]

# feature만 대상으로 검사
feature_cols = [col for col in df_valid.columns if col not in meta_cols]

non_numeric_cols = df_valid[feature_cols].select_dtypes(exclude=[np.number]).columns

print("숫자가 아닌 feature 컬럼:")
print(non_numeric_cols)

df_consistency_step = df_valid.copy()

if len(non_numeric_cols) > 0:
    df_consistency_step = df_consistency_step.drop(columns=non_numeric_cols.tolist())
    print(f"{len(non_numeric_cols)}개의 비수치 feature 제거")

# =========================
# 상수 feature 제거
# =========================
feature_cols = [col for col in df_consistency_step.columns if col not in ["Time", "Pass/Fail"]]

variance = df_consistency_step[feature_cols].var()
zero_variance_features = variance[variance == 0].index.tolist()

df_consistent = df_consistency_step.drop(columns=zero_variance_features)

print("\n[4-2] 뷴산 0인 상수")
print("제거된 상수 feature 수:", len(zero_variance_features))
print("일관성 처리 후 shape:", df_consistent.shape)
print("분석 feature 수:", len([col for col in df_consistent.columns if col not in ["Time", "Pass/Fail"]]))

# =========================
# 5. 최종 결과 저장
# =========================
df_consistent.to_csv("uci-secom_cleaned_until_consistency.csv", index=False)

print("\n최종 정제 데이터가 'uci-secom_cleaned_until_consistency.csv'로 저장되었습니다.")
print("=" * 60)

# =========================
# 6. 정확성 검증 (전/후 비교)
# =========================

# =========================
# 이상치 감소 확인
# =========================

feature_cols_before = [col for col in df.columns if col not in ["Time", "Pass/Fail"]]
feature_cols_after = [col for col in df_consistent.columns if col not in ["Time", "Pass/Fail"]]

# 공통 feature만 비교
common_cols = list(set(feature_cols_before).intersection(set(feature_cols_after)))

# 전처리 전 IQR 기준
Q1_before = df[common_cols].quantile(0.25)
Q3_before = df[common_cols].quantile(0.75)
IQR_before = Q3_before - Q1_before

lower_before = Q1_before - 1.5 * IQR_before
upper_before = Q3_before + 1.5 * IQR_before

# 이상치 개수 (전)
outliers_before = ((df[common_cols] < lower_before) | (df[common_cols] > upper_before)).sum().sum()

# 전처리 후 IQR 기준
Q1_after = df_consistent[common_cols].quantile(0.25)
Q3_after = df_consistent[common_cols].quantile(0.75)
IQR_after = Q3_after - Q1_after

lower_after = Q1_after - 1.5 * IQR_after
upper_after = Q3_after + 1.5 * IQR_after

# 이상치 개수 (후)
outliers_after = ((df_consistent[common_cols] < lower_after) | (df_consistent[common_cols] > upper_after)).sum().sum()

print("\n[정확성] IQR 기준 이상치 판단 결과")
print("전처리 전:", outliers_before)
print("전처리 후:", outliers_after)
print("감소량:", outliers_before - outliers_after)

# =========================
# 표준편차 감소 확인
# =========================

std_before = df[common_cols].std()
std_after = df_consistent[common_cols].std()

std_diff = std_after - std_before

decreased = (std_diff < 0).sum()
total = len(std_diff)

print("\n[정확성] 표준편차 감소")
print(f"표준편차 감소 feature 수: {decreased}/{total} ({decreased/total:.2%})")
