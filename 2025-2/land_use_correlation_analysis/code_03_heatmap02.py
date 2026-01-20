# -*- coding: utf-8 -*-
"""
충북 토지이용 상관분석 히트맵 (2015~2025년 전체)
 - 데이터: chungbuk_landuse_composition_2015_2025_detail.csv
 - 변수:
    [면적]  임야 면적(㎡), 농경지 면적(㎡), 대 면적(㎡), 공장용지 면적(㎡)
    [비율]  임야 비율, 농경지 비율, 대지 비율, 공장용지 비율
 - 산출물:
    corr_heatmap_area_2015_2025.png
    corr_heatmap_ratio_2015_2025.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== 한글 폰트 설정 =====
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ===== 1) 데이터 로드 =====
base_dir = r"C:/Users/leebi/OneDrive/바탕 화면/team_project"
detail_csv_path = os.path.join(base_dir, "chungbuk_landuse_composition_2015_2025_detail.csv")

df = pd.read_csv(detail_csv_path, encoding="cp949")

# '충청북도 합계'는 빼고 시군구만 사용
df = df[df["토지소재명"] != "충청북도 합계"].copy().reset_index(drop=True)

col_region  = "토지소재명"
col_year    = "year"
col_forest  = "임야 면적(㎡)"
col_agri    = "농경지 면적(㎡)"      # 이미 전+답 합친 컬럼
col_dae     = "대 면적(㎡)"
col_factory = "공장용지 면적(㎡)"

ratio_forest  = "임야 비율"
ratio_agri    = "농경지 비율"
ratio_dae     = "대지 비율"
ratio_factory = "공장용지 비율"

# 숫자형으로 확실히 변환 (혹시 문자열 섞여 있어도 처리)
num_cols = [col_forest, col_agri, col_dae, col_factory,
            ratio_forest, ratio_agri, ratio_dae, ratio_factory]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df[col_year] = df[col_year].astype(int)

print("연도 범위:", df[col_year].min(), "~", df[col_year].max())
print("전체 행 개수:", len(df))
print(df[[col_year, col_region, col_forest, col_agri, col_dae, col_factory]].head())

# ===== 2) 상관계수 행렬 계산 (Pearson, 2015~2025 전체) =====

# (1) 면적 변수들
area_vars = [col_forest, col_agri, col_dae, col_factory]
corr_area = df[area_vars].corr(method="pearson")

# (2) 비율 변수들
ratio_vars = [ratio_forest, ratio_agri, ratio_dae, ratio_factory]
corr_ratio = df[ratio_vars].corr(method="pearson")

print("\n=== [2015~2025] 면적 상관계수 (Pearson) ===")
print(corr_area)

print("\n=== [2015~2025] 비율 상관계수 (Pearson) ===")
print(corr_ratio)

# ===== 3) 히트맵 그리는 함수 =====
def draw_corr_heatmap(corr_df, title, outfile):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    im = ax.imshow(corr_df.values, vmin=-1, vmax=1, cmap="coolwarm")

    # 축 라벨
    ax.set_xticks(np.arange(len(corr_df.columns)))
    ax.set_yticks(np.arange(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_df.index)

    # 각 셀에 상관계수 숫자 표시
    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            val = corr_df.iloc[i, j]
            text_color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center", color=text_color, fontsize=9)

    plt.title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Pearson 상관계수")

    plt.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("저장:", outfile)

# ===== 4) 면적 상관 히트맵 =====
title_area = "2015~2025년 토지용도별 면적 상관계수 히트맵"
outfile_area = os.path.join(base_dir, "corr_heatmap_area_2015_2025.png")
draw_corr_heatmap(corr_area, title_area, outfile_area)

# ===== 5) 비율 상관 히트맵 =====
title_ratio = "2015~2025년 토지용도별 비율 상관계수 히트맵"
outfile_ratio = os.path.join(base_dir, "corr_heatmap_ratio_2015_2025.png")
draw_corr_heatmap(corr_ratio, title_ratio, outfile_ratio)

print("\n=== 2015~2025년 전체 기준 상관 히트맵 생성 완료 ===")
