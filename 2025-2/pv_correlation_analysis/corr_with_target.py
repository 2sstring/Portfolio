# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 23:42:34 2025

@author: bigbell
"""

# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# 1. 전처리된 데이터 불러오기
# =====================================
DATA_DIR = r"C:\Uni_Project\SolarDat\min"
DATA_PATH = os.path.join(DATA_DIR, "SolarData_60min_for_train.csv")

df = pd.read_csv(DATA_PATH)

print("데이터 shape:", df.shape)
print("컬럼 일부:", df.columns[:20].tolist())

# =====================================
# 2. 숫자형 컬럼만 골라서 상관계수 계산
# =====================================
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("숫자형 컬럼 개수:", len(numeric_cols))

corr = df[numeric_cols].corr(method="pearson")

# =====================================
# 2-1. 상관계수 결과 CSV로 저장 (전체 행렬)
# =====================================
OUT_CORR_FULL_CSV = os.path.join(DATA_DIR, "corr_matrix_60min.csv")
corr.to_csv(OUT_CORR_FULL_CSV, encoding="utf-8-sig")
print("전체 상관행렬 CSV 저장 완료:", OUT_CORR_FULL_CSV)

# =====================================
# 3. 상관계수 히트맵 그리기 + 이미지 저장
# =====================================
plt.figure(figsize=(14, 12))
sns.heatmap(
    corr,
    cmap="coolwarm",
    vmin=-1, vmax=1,
    square=True,
    cbar=True,
    xticklabels=True,
    yticklabels=True
)
plt.title("Correlation Matrix (SolarData_features_60min)", fontsize=16)

OUT_IMG_PATH = os.path.join(DATA_DIR, "corr_heatmap_60min.png")
plt.tight_layout()
plt.savefig(OUT_IMG_PATH, dpi=300)
plt.close()

print("상관계수 히트맵 PNG 저장 완료:", OUT_IMG_PATH)

# =====================================
# 4. Target과 각 피처의 상관계수만 따로 보기 + CSV 저장
# =====================================
if "Target" in numeric_cols:
    target_corr = corr["Target"].sort_values(ascending=False)
    print("\n[Target과 각 피처의 Pearson 상관계수 (내림차순)]")
    print(target_corr)

    # Series를 DataFrame으로 바꿔서 CSV 저장
    target_corr_df = target_corr.to_frame(name="corr_to_Target")
    OUT_CORR_TARGET_CSV = os.path.join(DATA_DIR, "corr_with_Target_60min.csv")
    target_corr_df.to_csv(OUT_CORR_TARGET_CSV, encoding="utf-8-sig")
    print("Target 기준 상관계수 CSV 저장 완료:", OUT_CORR_TARGET_CSV)

else:
    print("경고: 'Target' 컬럼을 찾지 못했습니다.")
