# -*- coding: utf-8 -*-
"""
충북 토지이용 유형을 지도에 표시 (2025년 기준)
 - 입력 1: chungbuk_landuse_clusters_pca_kmeans_2015_2025_with_road.csv
    * year, 토지소재명, 유형(도시/산업형, 균형형, 농업/산림형)
 - 입력 2: BND_SIGUNGU_PG.shp
    * 컬럼: BASE_DATE, SIGUNGU_CD, SIGUNGU_NM, geometry
"""

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ===== 경로 설정 =====
base_dir = r"C:/Users/leebi/OneDrive/바탕 화면/team_project"

# (1) 클러스터 결과 CSV
cluster_csv_path = os.path.join(base_dir, "chungbuk_landuse_clusters_pca_kmeans_2015_2025_with_road.csv")

# (2) 시군구 경계 Shapefile
shp_path = os.path.join(base_dir, "BND_SIGUNGU_PG.shp")

# ===== 0) Shapefile 미리 확인 =====
gdf = gpd.read_file(shp_path)
print("Shapefile 컬럼 목록:", gdf.columns)
print(gdf.head())

# ===== 1) 클러스터 결과 불러오기 (2025년 기준만 사용) =====
df = pd.read_csv(cluster_csv_path, encoding="cp949")

TARGET_YEAR = 2025
df_2025 = df[df["year"] == TARGET_YEAR].copy()

print(f"{TARGET_YEAR}년 데이터 행 수:", len(df_2025))
print(df_2025[["토지소재명", "유형"]])

# ----- 이름 매핑: 토지소재명 → Shapefile의 SIGUNGU_NM 형태 -----
name_map = {
    "청주 상당구": "청주시 상당구",
    "청주 서원구": "청주시 서원구",
    "청주 흥덕구": "청주시 흥덕구",
    "청주 청원구": "청주시 청원구",
    "충주시": "충주시",
    "제천시": "제천시",
    "보은군": "보은군",
    "옥천군": "옥천군",
    "영동군": "영동군",
    "증평군": "증평군",
    "진천군": "진천군",
    "괴산군": "괴산군",
    "음성군": "음성군",
    "단양군": "단양군",
}

df_2025["SIGUNGU_NM"] = df_2025["토지소재명"].replace(name_map)

print("\n=== 2025년 이름 매핑 결과 ===")
print(df_2025[["토지소재명", "SIGUNGU_NM", "유형"]])

# ===== 2) 시군구 경계 Shapefile 다시 로드 (gdf 그대로 써도 됨) =====
# 이미 위에서 gdf = gpd.read_file(shp_path) 했으니 재사용
# gdf 컬럼: BASE_DATE, SIGUNGU_CD, SIGUNGU_NM, geometry

# ===== 3) 지도 데이터와 클러스터 결과 merge =====
gdf_merged = gdf.merge(
    df_2025[["SIGUNGU_NM", "유형"]],
    on="SIGUNGU_NM",
    how="left"
)

print("\n=== merge 결과 일부 확인 ===")
print(gdf_merged[["SIGUNGU_NM", "유형"]].head(20))

# ===== 4) 유형별 색상 매핑 =====
type_color = {
    "도시/산업형": "#EA9358",   # 주황
    "균형형":     "#589AEA",   # 파랑
    "농업/산림형": "#75CD97",  # 초록
}
default_color = "#DDDDDD"      # 유형 없는 지역(충북 외 시군구 등)

def map_color(t):
    if t in type_color:
        return type_color[t]
    else:
        return default_color

gdf_merged["color"] = gdf_merged["유형"].apply(map_color)

# ===== 5) 충청북도(유형 있는 시군구만)만 플롯 =====

# 유형 값이 있는 행만 → 우리가 분석한 충북 14개 시군구
gdf_cb = gdf_merged.dropna(subset=["유형"]).copy()

fig, ax = plt.subplots(figsize=(7, 8))

# 충북 14개 시군구만 색칠
gdf_cb.plot(
    ax=ax,
    color=gdf_cb["color"],
    edgecolor="black",
    linewidth=0.5
)

# 시군구 이름 라벨
for _, row in gdf_cb.iterrows():
    centroid = row["geometry"].centroid
    ax.text(
        centroid.x,
        centroid.y,
        row["SIGUNGU_NM"],
        fontsize=8,
        ha="center",
        va="center"
    )

ax.set_axis_off()
ax.set_title("충청북도 토지이용 유형 분류", fontsize=14)

# 범례 수동 생성
from matplotlib.patches import Patch
type_color = {
    "도시/산업형": "#FF8C00",   # 주황
    "균형형":     "#1E90FF",   # 파랑
    "농업/산림형": "#2E8B57",  # 초록
}
legend_handles = [
    Patch(facecolor=type_color["도시/산업형"], edgecolor="black", label="도시/산업형"),
    Patch(facecolor=type_color["균형형"],     edgecolor="black", label="균형형"),
    Patch(facecolor=type_color["농업/산림형"], edgecolor="black", label="농업/산림형"),
]
ax.legend(handles=legend_handles, title="유형", loc="lower left")

out_map = os.path.join(base_dir, "chungbuk_landuse_type_map_2025_cb_only.png")
plt.tight_layout()
plt.savefig(out_map, dpi=200, bbox_inches="tight")
plt.close()

print("충북만 확대된 지도 이미지 저장:", out_map)

