# -*- coding: utf-8 -*-
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
col_agri    = "농경지 면적(㎡)"
col_dae     = "대 면적(㎡)"
col_factory = "공장용지 면적(㎡)"

# ===== 2) 2025년 데이터만 사용 =====
TARGET_YEAR = 2025
df_2025 = df[df[col_year] == TARGET_YEAR].copy()

if df_2025.empty:
    raise ValueError(f"{TARGET_YEAR}년 데이터가 없습니다.")

# ===== 3) 2025년 기준 비율 재계산 (임야+농경지+대지+공장용지 합 기준) =====
col_total = "합계 면적(㎡)"

denom_2025 = df[col_total]  # 지역·연도별 전체 면적

#denom_2025 = df_2025[col_forest] + df_2025[col_agri] + df_2025[col_dae] + df_2025[col_factory]

df_2025["임야 비율"]      = df_2025[col_forest]  / denom_2025
df_2025["농경지 비율"]    = df_2025[col_agri]    / denom_2025
df_2025["대지 비율"]      = df_2025[col_dae]     / denom_2025
df_2025["공장용지 비율"]  = df_2025[col_factory] / denom_2025

print(f"{TARGET_YEAR}년 행 개수:", len(df_2025))
print(df_2025[[col_region, "임야 비율", "농경지 비율", "대지 비율", "공장용지 비율"]].head())

# ===== 4) 행정구역별 색상 매핑 =====
regions = sorted(df_2025[col_region].unique())
cmap = plt.cm.get_cmap("tab20")
region_color_map = {region: cmap(i % 20) for i, region in enumerate(regions)}

# ===== 5) 상관 산점도: 비율 변수들끼리 모든 조합 =====
var_cols = ["임야 비율", "농경지 비율", "대지 비율", "공장용지 비율"]

for x_col, y_col in itertools.combinations(var_cols, 2):
    x = df_2025[x_col]
    y = df_2025[y_col]

    # 🔹 Pearson 상관계수 (method='pearson' 명시)
    r = x.corr(y, method="pearson")

    # 🔹 단순선형회귀 (y = a x + b)
    a, b = np.polyfit(x, y, 1)

    fig, ax = plt.subplots(figsize=(7.5, 6))

    # 행정구역별 색상
    for region in regions:
        sub = df_2025[df_2025[col_region] == region]
        ax.scatter(
            sub[x_col],
            sub[y_col],
            color=region_color_map[region],
            marker='o',
            alpha=0.9,
        )

    # 회귀선
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = a * x_line + b
    ax.plot(x_line, y_line, color="red", linestyle="--", label="회귀선")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{TARGET_YEAR}년 {x_col} vs {y_col} (Pearson r={r:.3f})")

    # 행정구역 범례
    region_handles = [
        Line2D(
            [0], [0],
            marker='o',
            color=color,
            linestyle='',
            markersize=7,
            label=region
        )
        for region, color in region_color_map.items()
    ]
    ax.legend(
        handles=region_handles,
        title="행정구역",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )

    plt.tight_layout()
    fig.subplots_adjust(right=0.8)

    # 파일 이름 저장
    outfile = os.path.join(
        base_dir,
        f"scatter_{TARGET_YEAR}_{x_col.replace(' ', '')}_vs_{y_col.replace(' ', '')}.png"
    )
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("저장:", outfile)

print("\n=== 2025년 기준 변수쌍별 상관 산점도(Pearson) 생성 완료 ===")

