# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 03:29:57 2025

@author: bigbell
"""

# -*- coding: utf-8 -*-
"""
충청북도 토지이용 유형 분류 (PCA + KMeans, 2015~2025)
 - 입력 1: chungbuk_landuse_composition_2015_2025_detail.csv
 - 입력 2: chungbuk_population.xlsx
 - 입력 3: chungbuk_road_ratio_2015_2025.xlsx

 - 특징(Feature):
    1) 임야/농경지/대지/공장용지 비율 (4개 합 = 1)  ←★ KMeans는 이 4개만 사용
    2) 인구밀도 (인구 / 전체면적[km²])              ← 해석용
    3) 도로율 (도로면적 / 전체면적)                 ← 해석용
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ===== 한글 폰트 설정 (Windows 기준) =====
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ===== 1) 경로 및 파일 설정 =====
base_dir = r"C:/Users/leebi/OneDrive/바탕 화면/team_project"
detail_csv_path = os.path.join(base_dir, "chungbuk_landuse_composition_2015_2025_detail.csv")
pop_xlsx_path   = os.path.join(base_dir, "chungbuk_population.xlsx")
road_xlsx_path  = os.path.join(base_dir, "chungbuk_road_ratio_2015_2025.xlsx")

# ===== 2) 토지이용 데이터 로드 =====
detail = pd.read_csv(detail_csv_path, encoding="cp949")

# 혹시 합계 행이 섞여 있으면 제거
detail = detail[detail["토지소재명"] != "충청북도 합계"].copy().reset_index(drop=True)

print("토지 데이터 행 수:", len(detail))
print("연도 범위:", detail["year"].min(), "~", detail["year"].max())
print("시군구 목록:", sorted(detail["토지소재명"].unique()))

# ===== 3) 인구 데이터 로드 및 Long 형태로 변환 =====
pop_raw = pd.read_excel(pop_xlsx_path)

pop_long = pop_raw.melt(
    id_vars=["연도"],
    var_name="year",
    value_name="인구"
)
pop_long["year"] = pop_long["year"].astype(int)
pop_long = pop_long.rename(columns={"연도": "토지소재명"})

print("\n인구 데이터 예시:")
print(pop_long.head())

# ===== 4) 도로율 엑셀 로드 (sheet='detail') =====
road_detail = pd.read_excel(road_xlsx_path, sheet_name="detail")

if "year" not in road_detail.columns:
    raise ValueError("도로율 엑셀에 'year' 컬럼이 없습니다. 컬럼명을 확인하세요.")
if "토지소재명" not in road_detail.columns:
    raise ValueError("도로율 엑셀에 '토지소재명' 컬럼이 없습니다.")

print("\n도로율 데이터 예시:")
print(road_detail[["year", "토지소재명", "도로율"]].head())

# ===== 5) 토지 + 인구 + 도로율 데이터 병합 =====
df = detail.merge(pop_long, on=["토지소재명", "year"], how="left")
df = df.merge(
    road_detail[["year", "토지소재명", "도로율"]],
    on=["year", "토지소재명"],
    how="left"
)

missing_pop = df["인구"].isna().sum()
missing_road = df["도로율"].isna().sum()
if missing_pop > 0:
    print("\n⚠ 경고: 인구 정보가 없는 행 개수:", missing_pop)
if missing_road > 0:
    print("⚠ 경고: 도로율 정보가 없는 행 개수:", missing_road)

# ===== 6) 4개 용도 기준 비율 재계산 =====
col_forest  = "임야 면적(㎡)"
col_agri    = "농경지 면적(㎡)"
col_dae     = "대 면적(㎡)"
col_factory = "공장용지 면적(㎡)"
col_total   = "합계 면적(㎡)"

col_total = "합계 면적(㎡)"

denom_4 = df[col_total]  # 지역·연도별 전체 면적

#denom_4 = df[[col_forest, col_agri, col_dae, col_factory]].sum(axis=1)

df["임야비율4"]   = df[col_forest]  / denom_4
df["농경지비율4"] = df[col_agri]    / denom_4
df["대지비율4"]   = df[col_dae]     / denom_4
df["공장비율4"]   = df[col_factory] / denom_4

# 인구밀도 (명 / km²) – 군집에는 사용하지 않지만 정보는 유지
df["pop_density"] = df["인구"] / (df[col_total] / 1_000_000.0)

print("\n=== 특징(예시 5행) ===")
print(df[[
    "year", "토지소재명",
    "임야비율4", "농경지비율4", "대지비율4", "공장비율4",
    "도로율",
    "인구", "pop_density"
]].head())

# ===== 7) 특징 행렬 구성 & 표준화 (★ KMeans 피처 설정) =====
feature_cols = [
    "임야비율4", "농경지비율4", "대지비율4", "공장비율4",
    "도로율", "pop_density"   # ← 여기 두 개를 추가
]

X = df[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===== 8) PCA (2차원, 시각화용) =====
pca = PCA(n_components=2, random_state=0)
X_pca = pca.fit_transform(X_scaled)

df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

# ===== 9) KMeans (k=3) – 이제 6차원(4비율+도로율+인구밀도)에 대해 수행 =====
kmeans = KMeans(n_clusters=3, random_state=0, n_init=50)
clusters = kmeans.fit_predict(X_scaled)
df["cluster"] = clusters

# ----- 클러스터별 평균 프로파일 (비율 + 도로율 + 인구밀도 평균 확인용) -----
cluster_profile = df.groupby("cluster")[[
    "임야비율4", "농경지비율4", "대지비율4", "공장비율4",
    "도로율",
    "pop_density"
]].mean()

print("\n=== 클러스터별 평균 프로파일 ===")
print(cluster_profile)

# ===== 10) 클러스터 → 유형 라벨 자동 매핑 =====
cp = cluster_profile.copy()
pop_min = cp["pop_density"].min()
pop_max = cp["pop_density"].max()
cp["pop_norm"] = (cp["pop_density"] - pop_min) / (pop_max - pop_min + 1e-9)

cp["urban_score"] = (
    cp["대지비율4"] +
    cp["공장비율4"] +
    cp["도로율"] +
    cp["pop_norm"]
)

order = cp["urban_score"].sort_values().index.tolist()
type_names = ["농업/산림형", "균형형", "도시/산업형"]
cluster_to_type = {cl: t for cl, t in zip(order, type_names)}

print("\n=== 클러스터 → 유형 매핑 ===")
for cl, t in cluster_to_type.items():
    print(f" cluster {cl} → {t}")

df["유형"] = df["cluster"].map(cluster_to_type)

# ===== 10-1) Softmax 기반 유형별 확률 계산 =====
#  - KMeans는 X_scaled(4D 비율)에 대해 학습되었으므로
#    kmeans.transform(X_scaled)로 각 클러스터까지 거리(N x 3)를 구하고
#    softmax(-거리)로 3개 유형 확률을 계산

distances = kmeans.transform(X_scaled)  # shape (N, 3)

def softmax_neg_dist(d):
    s = -d
    s = s - np.max(s)      # overflow 방지
    exp_s = np.exp(s)
    return exp_s / np.sum(exp_s)

probs = np.apply_along_axis(softmax_neg_dist, 1, distances)  # (N, 3)

# cluster → 유형 이 이미 있으므로, 역매핑으로 유형별 인덱스 구하기
type_to_cluster = {v: k for k, v in cluster_to_type.items()}

idx_u = type_to_cluster["도시/산업형"]
idx_g = type_to_cluster["농업/산림형"]
idx_b = type_to_cluster["균형형"]

df["P_도시산업형"] = probs[:, idx_u]
df["P_농업산림형"] = probs[:, idx_g]
df["P_균형형"]     = probs[:, idx_b]

print("\n=== Softmax 기반 유형별 확률 예시(앞 5행) ===")
print(df[[
    "year", "토지소재명", "유형",
    "P_도시산업형", "P_농업산림형", "P_균형형"
]].head())

# ===== 11) 시군구별 대표 유형 (2015~2025 전체 기준) =====
region_type = df.groupby("토지소재명")["유형"].agg(
    lambda s: s.value_counts().idxmax()
)

print("\n=== 시군구별 대표 유형 (2015~2025 전체 기준 최빈값) ===")
for region, t in region_type.items():
    print(f" - {region}: {t}")

# ===== 12) 2025년 기준 유형 정리 =====
TARGET_YEAR = 2025
df_2025 = df[df["year"] == TARGET_YEAR].copy()

print(f"\n=== {TARGET_YEAR}년 기준 시군구별 토지구성 + 인구 + 도로율 + 유형 ===")
print(df_2025[[
    "토지소재명",
    "임야비율4", "농경지비율4", "대지비율4", "공장비율4",
    "도로율",
    "pop_density",
    "유형",
    "P_도시산업형", "P_농업산림형", "P_균형형"
]].sort_values("토지소재명"))

# ===== 13) 결과 CSV 저장 =====
out_csv_path = os.path.join(
    base_dir,
    "chungbuk_landuse_clusters_kmeans_2015_2025_ratio_only_softmax.csv"
)
df.to_csv(out_csv_path, index=False, encoding="cp949")
print("\n저장 완료: ", out_csv_path)

# ===== 14) PCA 평면 시각화 (2025년만, 유형별 색상) =====
plt.figure(figsize=(9, 7))

type_list = ["농업/산림형", "균형형", "도시/산업형"]
color_map = {
    "농업/산림형": "#75CD97",   # 초록
    "균형형": "#589AEA",       # 파랑
    "도시/산업형": "#EA9358",  # 주황
}

# ===== 15) 2015~2025 전체 PCA + KMeans 결과 시각화 (텍스트는 2025년만) =====
plt.figure(figsize=(7.5, 6))

for t in type_list:
    sub_all = df[df["유형"] == t]
    plt.scatter(
        sub_all["PC1"], sub_all["PC2"],
        color=color_map[t],
        label=f"{t}",
        alpha=0.9,
        s=60,
    )

for _, row in df_2025.iterrows():
    plt.text(
        row["PC1"] + 0.02,
        row["PC2"] + 0.02,
        row["토지소재명"],
        fontsize=8,
    )

plt.axhline(0, color="gray", linewidth=0.5)
plt.axvline(0, color="gray", linewidth=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"충청북도 토지이용 군집 (KMeans, 2015~2025년,\n임야/농경지/대지/공장 비율 + 도로율 + 인구밀도)")
plt.legend(title="유형")
plt.tight_layout()

out_png_all = os.path.join(
    base_dir,
    "pca_kmeans_clusters_2015_2025_ratio_only_allyears_with_2025_labels.png"
)
plt.savefig(out_png_all, dpi=200)
plt.close()

print("전체 연도 PCA 클러스터 그림 저장:", out_png_all)

print("\n=== 전체 작업 완료 (임야/농경지/대지/공장 비율 기반 KMeans + Softmax 버전) ===")
