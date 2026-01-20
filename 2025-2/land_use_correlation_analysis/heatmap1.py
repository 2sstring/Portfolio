# -*- coding: utf-8 -*-
"""
충북 토지이용 기초 통계 분석
 - 데이터: chungbuk_landuse_composition_2015_2025_detail.csv
 - 1) 용도별 평균/중앙값 (면적 + 비율)
 - 2) 연도별 통계
 - 3) 2025년 지역별 순위 분석
 - 4) 분포(히스토그램/박스플롯) 시각화
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== 한글 폰트 설정 =====
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ===== 0) 경로 설정 =====
base_dir = r"data"
detail_csv_path = os.path.join(base_dir, "chungbuk_landuse_composition_2015_2025_detail.csv")

df = pd.read_csv(detail_csv_path, encoding="cp949")

# 혹시 합계 행이 섞여 있으면 제거 (없으면 영향 없음)
df = df[df["토지소재명"] != "충청북도 합계"].copy().reset_index(drop=True)

col_region  = "토지소재명"
col_year    = "year"

# 면적 컬럼
col_forest  = "임야 면적(㎡)"
col_agri    = "농경지 면적(㎡)"
col_dae     = "대 면적(㎡)"
col_factory = "공장용지 면적(㎡)"

# 비율 컬럼 (detail CSV에서 이미 있는 걸 전제로)
ratio_forest  = "임야 비율"
ratio_agri    = "농경지 비율"
ratio_dae     = "대지 비율"
ratio_factory = "공장용지 비율"

area_cols = {
    "임야": col_forest,
    "농경지": col_agri,
    "대지": col_dae,
    "공장용지": col_factory,
}

ratio_cols = {
    "임야": ratio_forest,
    "농경지": ratio_agri,
    "대지": ratio_dae,
    "공장용지": ratio_factory,
}

# 숫자형으로 확실히
for c in list(area_cols.values()):
    df[c] = pd.to_numeric(df[c], errors="coerce")

for c in list(ratio_cols.values()):
    df[c] = pd.to_numeric(df[c], errors="coerce")

df[col_year] = df[col_year].astype(int)

# ======================================================================
# 1) 전체(2015~2025) 기준: 용도별 평균/중앙값 (면적 + 비율)
# ======================================================================

area_stats = []
for name, c in area_cols.items():
    area_stats.append({
        "용도": name,
        "평균 면적(㎡)": df[c].mean(),
        "중앙값 면적(㎡)": df[c].median()
    })
area_stats_df = pd.DataFrame(area_stats)

ratio_stats = []
for name, c in ratio_cols.items():
    ratio_stats.append({
        "용도": name,
        "평균 비율": df[c].mean(),
        "중앙값 비율": df[c].median()
    })
ratio_stats_df = pd.DataFrame(ratio_stats)

print("\n=== [전체 2015~2025] 용도별 면적 평균/중앙값 ===")
print(area_stats_df)

print("\n=== [전체 2015~2025] 용도별 비율 평균/중앙값 ===")
print(ratio_stats_df)

# CSV로도 저장
area_stats_path = os.path.join(base_dir, "basic_stats_area_2015_2025.csv")
ratio_stats_path = os.path.join(base_dir, "basic_stats_ratio_2015_2025.csv")
area_stats_df.to_csv(area_stats_path, index=False, encoding="cp949")
ratio_stats_df.to_csv(ratio_stats_path, index=False, encoding="cp949")
print("\n면적/비율 기초 통계 CSV 저장:")
print(" -", area_stats_path)
print(" -", ratio_stats_path)

# ======================================================================
# 2) 연도별 평균/중앙값 (비율 위주)
# ======================================================================

group = df.groupby(col_year)

year_stats_rows = []
for year, g in group:
    row = {"year": year}
    for name, c in ratio_cols.items():
        row[f"{name}_평균비율"]   = g[c].mean()
        row[f"{name}_중앙값비율"] = g[c].median()
    year_stats_rows.append(row)

year_stats_df = pd.DataFrame(year_stats_rows).sort_values("year")

print("\n=== 연도별(2015~2025) 용도별 비율 평균/중앙값 ===")
print(year_stats_df.head())

year_stats_path = os.path.join(base_dir, "yearly_ratio_stats_2015_2025.csv")
year_stats_df.to_csv(year_stats_path, index=False, encoding="cp949")
print("연도별 비율 통계 CSV 저장:", year_stats_path)

# ======================================================================
# 3) 2025년 기준: 지역별 비교 및 순위 분석
#    - 예시: 대지/공장용지 면적 및 비율 순위
# ======================================================================

TARGET_YEAR = 2025
df_2025 = df[df[col_year] == TARGET_YEAR].copy()

# 순위 컬럼 (내림차순: 값이 클수록 1위)
df_2025["대지면적_순위"]        = df_2025[col_dae].rank(ascending=False, method="min")
df_2025["공장용지면적_순위"]    = df_2025[col_factory].rank(ascending=False, method="min")
df_2025["대지비율_순위"]        = df_2025[ratio_dae].rank(ascending=False, method="min")
df_2025["공장용지비율_순위"]    = df_2025[ratio_factory].rank(ascending=False, method="min")

rank_cols = [
    col_year, col_region,
    col_dae, col_factory, ratio_dae, ratio_factory,
    "대지면적_순위", "공장용지면적_순위",
    "대지비율_순위", "공장용지비율_순위",
]

df_2025_rank = df_2025[rank_cols].sort_values("대지비율_순위")

print(f"\n=== {TARGET_YEAR}년 지역별 대지/공장용지 순위 (일부) ===")
print(df_2025_rank.head())

rank_path = os.path.join(base_dir, f"region_rank_{TARGET_YEAR}.csv")
df_2025_rank.to_csv(rank_path, index=False, encoding="cp949")
print("지역별 순위 CSV 저장:", rank_path)

# ======================================================================
# 4) 분포 특성 파악: 히스토그램 + 박스플롯 (비율 기준)
# ======================================================================

# (1) 비율 히스토그램 (전체 2015~2025)
plt.figure(figsize=(10, 8))

for i, (name, c) in enumerate(ratio_cols.items(), start=1):
    plt.subplot(2, 2, i)
    plt.hist(df[c].dropna(), bins=15, alpha=0.8)
    plt.title(f"{name} 비율 분포 (2015~2025)")
    plt.xlabel("비율")
    plt.ylabel("빈도")

plt.tight_layout()
hist_path = os.path.join(base_dir, "hist_ratio_2015_2025.png")
plt.savefig(hist_path, dpi=200)
plt.close()
print("비율 히스토그램 이미지 저장:", hist_path)

# (2) 비율 박스플롯 (전체 2015~2025)
plt.figure(figsize=(7, 6))
data_for_box = [df[c].dropna() for c in ratio_cols.values()]
plt.boxplot(data_for_box, labels=list(ratio_cols.keys()))
plt.ylabel("비율")
plt.title("임야/농경지/대지/공장용지 비율 박스플롯 (2015~2025)")
plt.tight_layout()

box_path = os.path.join(base_dir, "boxplot_ratio_2015_2025.png")
plt.savefig(box_path, dpi=200)
plt.close()
print("비율 박스플롯 이미지 저장:", box_path)

print("\n=== 기초 통계 분석 완료 ===")

