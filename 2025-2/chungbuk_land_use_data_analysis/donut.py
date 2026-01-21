# -*- coding: utf-8 -*-
import os
import math
import pandas as pd
import matplotlib.pyplot as plt

# ===== 한글 폰트 설정 (Windows) =====
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ===== 1) 데이터 불러오기 =====
base_dir = r"C:/Users/leebi/OneDrive/바탕 화면/team_project"
detail_csv_path = os.path.join(base_dir, "chungbuk_landuse_composition_2015_2025_detail.csv")

df = pd.read_csv(detail_csv_path, encoding="cp949")

# ===== 2) 분석 연도 선택 =====
TARGET_YEAR = 2025
df_year = df[df["year"] == TARGET_YEAR].copy().reset_index(drop=True)

# ===== 3) 비율 재계산 (임야+농경지+대지+공장용지 합 기준 구성비) =====
df_year["denom"] = (
    df_year["임야 면적(㎡)"]
    + df_year["농경지 면적(㎡)"]
    + df_year["대 면적(㎡)"]
    + df_year["공장용지 면적(㎡)"]
)

df_year["임야 비율"]      = df_year["임야 면적(㎡)"]   / df_year["denom"]
df_year["농경지 비율"]    = df_year["농경지 면적(㎡)"] / df_year["denom"]
df_year["대지 비율"]      = df_year["대 면적(㎡)"]     / df_year["denom"]
df_year["공장용지 비율"]  = df_year["공장용지 면적(㎡)"] / df_year["denom"]

print(f"{TARGET_YEAR}년 데이터 {len(df_year)}개 지역 로드 완료")
print(df_year[["토지소재명", "임야 비율", "농경지 비율", "대지 비율", "공장용지 비율"]].head())

labels = ["임야", "농경지", "대지", "공장용지"]


def draw_donut(values, title, outfile=None, ax=None, show_title=True):
    """
    임야/농경지/대지/공장용지 도넛차트 하나 그리는 함수
    - values: [임야, 농경지, 대지, 공장용지] 비율 (합 ~ 1.0)
    - ax가 None이면 새 figure를 만들고, 아니면 해당 axes 위에 그림
    - 각 조각 비율이 9% 미만이면 텍스트(라벨+퍼센트) 숨김
    """
    
    # 용도별 기본 색 (카테고리별 색 유지)
    base_colors = {
        "임야": "#75CD97",  # 임야: 녹색
        "농경지": "#E6C950",  # 농경지: 노란색
        "대지": "#EA9358",  # 대지: 주황색
        "공장용지": "#8FA1BB",  # 공장용지: 회색
    }

    # 전역 labels = ["임야", "농경지", "대지", "공장용지"] 를 사용
    # values, labels, colors를 (값 기준 내림차순)으로 정렬
    triplets = []
    for v, lab in zip(values, labels):
        triplets.append((v, lab, base_colors[lab]))

    # 비율 큰 순서대로 정렬
    triplets.sort(key=lambda x: x[0], reverse=True)

    values_sorted  = [t[0] for t in triplets]
    labels_sorted  = [t[1] for t in triplets]
    colors_sorted  = [t[2] for t in triplets]
    
    own_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
        own_fig = True

    wedges, texts, autotexts = ax.pie(
        values_sorted,
        labels=labels_sorted,
        autopct="%.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.4),
        pctdistance=0.75,
        colors=colors_sorted,
    )

    # 텍스트 숨김: 비율 5% 미만은 라벨/퍼센트 제거
    for v, txt, autotxt in zip(values_sorted, texts, autotexts):
        if v < 0.05:
            txt.set_text("")
            autotxt.set_text("")

    if show_title:
        ax.set_title(title, fontsize=10)

    if own_fig:
        plt.tight_layout()
        if outfile is not None:
            fig.savefig(outfile, dpi=200)
            print("저장:", outfile)
        plt.close(fig)


# ===== 4) 충청북도 전체 도넛차트 (1개) =====
total_forest  = df_year["임야 면적(㎡)"].sum()
total_agri    = df_year["농경지 면적(㎡)"].sum()
total_dae     = df_year["대 면적(㎡)"].sum()
total_factory = df_year["공장용지 면적(㎡)"].sum()

denom_total = total_forest + total_agri + total_dae + total_factory

total_values = [
    total_forest / denom_total,
    total_agri   / denom_total,
    total_dae    / denom_total,
    total_factory/ denom_total,
]

total_title = f"충청북도 전체 토지 이용 구성비 ({TARGET_YEAR}년)"
total_outfile = os.path.join(base_dir, f"donut_total_{TARGET_YEAR}.png")
draw_donut(total_values, total_title, total_outfile)

# ===== 5) 각 지역별 도넛차트 개별 파일 (14개) =====
for _, row in df_year.iterrows():
    region = row["토지소재명"]
    values = [
        row["임야 비율"],
        row["농경지 비율"],
        row["대지 비율"],
        row["공장용지 비율"],
    ]

    title = f"{region} 토지 이용 구성비 ({TARGET_YEAR}년)"
    safe_region = region.replace(" ", "_")
    outfile = os.path.join(base_dir, f"donut_{TARGET_YEAR}_{safe_region}.png")
    draw_donut(values, title, outfile)

print("=== 개별 도넛차트 저장 완료 (지역별 14개) ===")

# ===== 6) 14개 지역 도넛차트를 한 장에 모은 이미지 =====
n_regions = len(df_year)   # 보통 14개
# 적당한 그리드(예: 4x4) 계산
cols = 4
rows = math.ceil(n_regions / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
axes = axes.flatten()

for ax in axes[n_regions:]:
    ax.axis("off")  # 남는 칸은 비우기

for idx, (_, row) in enumerate(df_year.iterrows()):
    ax = axes[idx]
    region = row["토지소재명"]
    values = [
        row["임야 비율"],
        row["농경지 비율"],
        row["대지 비율"],
        row["공장용지 비율"],
    ]
    # 제목은 지역 이름만 간단히
    draw_donut(values, region, ax=ax, show_title=True)

plt.tight_layout()
grid_outfile = os.path.join(base_dir, f"donut_grid_{TARGET_YEAR}.png")
fig.savefig(grid_outfile, dpi=200)
plt.close(fig)
print("=== 전체 지역 도넛 모음 이미지 저장 완료:", grid_outfile)

print("=== 도넛차트 생성 최종 완료 (총 16개: 전체 1 + 지역별 14 + 모음 1) ===")
