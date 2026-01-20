# -*- coding: utf-8 -*-
"""
충북 토지이용 상관 산점도
 - X축: 대지 면적(㎡)
 - Y축: 공장용지 면적(㎡)
 - 데이터: chungbuk_landuse_composition_2015_2025_detail.csv

B) 2015~2025 전체를 한 그림에:
    - 색상: 행정구역별
    - 마커: 연도별

C) 연도별로 그림 분리:
    - 색상: 행정구역별
    - 마커: 동일('o')
"""

import os
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

# 혹시 '충청북도 합계' 같은 합계 행이 섞여 있으면 제거 (안 섞여 있으면 이 줄은 영향 없음)
df = df[df["토지소재명"] != "충청북도 합계"].copy().reset_index(drop=True)

# 사용 컬럼 이름
col_region  = "토지소재명"
col_year    = "year"
col_dae     = "대 면적(㎡)"
col_factory = "공장용지 면적(㎡)"

print("데이터 행 수 :", len(df))
print("연도 범위   :", df[col_year].min(), "~", df[col_year].max())
print("행정구역 수 :", df[col_region].nunique())

# ===== 2) 색상(행정구역별), 마커(연도별) 설정 =====
regions = sorted(df[col_region].unique())
years   = sorted(df[col_year].unique())

# 색상: 행정구역별 (tab20 팔레트 반복 사용)
cmap = plt.cm.get_cmap("tab20")
region_color_map = {
    region: cmap(i % 20) for i, region in enumerate(regions)
}

# 마커: 연도별
marker_list = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>', 'h', '*']
year_marker_map = {
    year: marker_list[i % len(marker_list)] for i, year in enumerate(years)
}

print("행정구역 색상 매핑:", region_color_map)
print("연도 마커 매핑:", year_marker_map)

# ===== 공통: 전체 상관계수, 회귀식 계산 (2015~2025 전체) =====
x_all = df[col_dae].values
y_all = df[col_factory].values

r_all = float(np.corrcoef(x_all, y_all)[0, 1])
coef_all = np.polyfit(x_all, y_all, 1)
slope_all, intercept_all = float(coef_all[0]), float(coef_all[1])

print(f"\n[전체] 상관계수 r = {r_all:.3f}")
print(f"[전체] 회귀식: 공장용지면적 ≈ {slope_all:.3f} * 대지면적 + {intercept_all:.1f}")


# ======================================================================
# B) 2015~2025 전체를 한 그림에: 색상=행정구역, 마커=연도
# ======================================================================

fig, ax = plt.subplots(figsize=(8, 7))

for region in regions:
    for year in years:
        sub = df[(df[col_region] == region) & (df[col_year] == year)]
        if sub.empty:
            continue
        ax.scatter(
            sub[col_dae],
            sub[col_factory],
            color=region_color_map[region],
            marker='o',
            alpha=0.8,
        )

# 회귀선 (전체 데이터 기준)
x_line = np.linspace(x_all.min(), x_all.max(), 200)
y_line = slope_all * x_line + intercept_all
ax.plot(x_line, y_line, color="red", linestyle="--", label="회귀선(전체)")

ax.set_xlabel("대지 면적(㎡)")
ax.set_ylabel("공장용지 면적(㎡)")
ax.set_title(f"대지 면적 vs 공장용지 면적 (2015~{max(years)}, r={r_all:.3f})")

# --- 범례 1: 행정구역 (색상) ---
region_handles = [
    Line2D([0], [0],
           marker='o',
           color=color,
           linestyle='',
           markersize=7,
           label=region)
    for region, color in region_color_map.items()
]

legend1 = ax.legend(
    handles=region_handles,
    title="행정구역",
    fontsize=8,
    title_fontsize=9,
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
)
ax.add_artist(legend1)

# --- 범례 2: 연도 (마커) ---
year_handles = [
    Line2D([0], [0],
           marker='o',
           color='black',
           linestyle='',
           markersize=7,
           label=str(year))
    for year, marker in year_marker_map.items()
]

"""legend2 = ax.legend(
    handles=year_handles,
    title="연도",
    fontsize=8,
    title_fontsize=9,
    loc="lower left",
    bbox_to_anchor=(1.02, 0.0),
)"""

plt.tight_layout()
fig.subplots_adjust(right=0.83)
outfile_B = os.path.join(base_dir, "scatter_dae_vs_factory_all_years_regioncolor_yearmarker.png")
fig.savefig(outfile_B, dpi=200)
plt.close(fig)

print("\n[B] 전체 연도+지역 산점도 저장:", outfile_B)


# ======================================================================
# C) 연도별로 파일 분리: 색상=행정구역, 마커=o (고정)
# ======================================================================

for year in years:
    df_y = df[df[col_year] == year].copy()
    if df_y.empty:
        continue

    x_y = df_y[col_dae].values
    y_y = df_y[col_factory].values

    r_y = float(np.corrcoef(x_y, y_y)[0, 1])
    coef_y = np.polyfit(x_y, y_y, 1)
    slope_y, intercept_y = float(coef_y[0]), float(coef_y[1])

    fig, ax = plt.subplots(figsize=(7, 6))

    # 행정구역별 색상으로 표시 (마커는 고정 'o')
    for region in regions:
        sub = df_y[df_y[col_region] == region]
        if sub.empty:
            continue
        ax.scatter(
            sub[col_dae],
            sub[col_factory],
            color=region_color_map[region],
            marker='o',
            alpha=0.8,
        )

    # 해당 연도 회귀선
    x_line_y = np.linspace(x_y.min(), x_y.max(), 200)
    y_line_y = slope_y * x_line_y + intercept_y
    ax.plot(x_line_y, y_line_y, color="red", linestyle="--", label="회귀선")

    ax.set_xlabel("대지 면적(㎡)")
    ax.set_ylabel("공장용지 면적(㎡)")
    ax.set_title(f"{year}년 대지 면적 vs 공장용지 면적 (r={r_y:.3f})")

    # 행정구역 범례
    region_handles_y = [
        Line2D([0], [0],
               marker='o',
               color=color,
               linestyle='',
               markersize=7,
               label=region)
        for region, color in region_color_map.items()
    ]
    ax.legend(
        handles=region_handles_y,
        title="행정구역",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )

    plt.tight_layout()
    outfile_C = os.path.join(base_dir, f"scatter_dae_vs_factory_{year}.png")
    fig.savefig(outfile_C, dpi=200)
    plt.close(fig)

    print(f"[C] {year}년 산점도 저장:", outfile_C)

print("\n=== 상관 산점도 생성 완료: B(전체 1장) + C(연도별 여러 장) ===")
