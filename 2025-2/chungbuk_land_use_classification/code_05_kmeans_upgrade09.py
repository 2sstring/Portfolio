# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 00:17:57 2025

@author: bigbell
"""

# -*- coding: utf-8 -*-
"""
충북 토지이용 군집 분석 (KMeans + PCA + Softmax 확률)
 - 대상: chungbuk_data_2015~2024 (연도별 CSV, 연말 기준)
 - 특징: 지정한 19개 '... 면적(㎡)' 비율 + 인구밀도 (도로율은 제외)
 - 군집: KMeans(k=3)
 - 시각화: PCA(2D) + 클러스터 색상
 - 추가: 각 유형(도시/산업형, 농업/산림형, 균형형)에 대한 softmax 확률 계산
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ===== 한글 폰트 설정 =====
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# (선택) KMeans 메모리 경고 완화
os.environ["OMP_NUM_THREADS"] = "1"

# ===== 0) 경로 설정 =====
base_dir = r"C:/Users/leebi/OneDrive/바탕 화면/team_project"

# 연도별 토지이용 원본 CSV
landuse_pattern = os.path.join(base_dir, "chungbuk_data_*.csv")

# 인구 엑셀
pop_xlsx_path = os.path.join(base_dir, "chungbuk_population.xlsx")

col_region = "토지소재명"
col_year   = "year"
col_total  = "합계 면적(㎡)"

# ===== 1) 연도별 CSV 읽어서 하나로 결합 =====
landuse_files = sorted(glob.glob(landuse_pattern))
if not landuse_files:
    raise FileNotFoundError(f"패턴에 맞는 파일이 없습니다: {landuse_pattern}")

df_list = []
for path in landuse_files:
    name = os.path.basename(path)
    # 파일명 예: chungbuk_data_20151231.csv → year = 2015
    year = int(name[14:18])

    tmp = pd.read_csv(path, encoding="utf-8-sig")
    tmp[col_year] = year
    df_list.append(tmp)

df = pd.concat(df_list, ignore_index=True)

print("원본 결합 데이터 크기:", df.shape)
print("연도 목록:", sorted(df[col_year].unique()))

# ===== 2) 숫자 컬럼 정리 (문자열 → float) =====
num_cols = [c for c in df.columns if c != col_region]

for c in num_cols:
    df[c] = (
        df[c]
        .astype(str)
        .str.replace(",", "", regex=False)  # 천단위 콤마 제거
        .str.strip()
    )
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ===== 3) 인구 데이터 로드 및 Long 형태 변환 후 merge =====
pop_raw = pd.read_excel(pop_xlsx_path)

# pop_raw 예시:
#   연도   2015   2016 ... 2025
# 0 청주 상당구 ...
pop_long = pop_raw.melt(
    id_vars=["연도"],
    var_name="year",
    value_name="인구"
)
pop_long["year"] = pop_long["year"].astype(int)
pop_long = pop_long.rename(columns={"연도": col_region})

# 토지 + 인구 병합
df = df.merge(pop_long, on=[col_region, col_year], how="left")

missing_pop = df["인구"].isna().sum()
if missing_pop > 0:
    print(f"\n⚠ 경고: 인구 정보가 없는 행 개수 = {missing_pop}")

# ===== 4) “지정한 19개” 면적(㎡) 컬럼만 → 비율 컬럼으로 변환 =====
# 사용할 면적 컬럼 (질문에서 주신 리스트)
TARGET_AREA_COLS = [
    "대 면적(㎡)",
    "공장용지 면적(㎡)",
    "도로 면적(㎡)",
    "철도용지 면적(㎡)",
    "주차장 면적(㎡)",
    "주유소용지 면적(㎡)",
    "창고용지 면적(㎡)",
    "학교용지 면적(㎡)",
    "수도용지 면적(㎡)",
    "공원 면적(㎡)",
    "체육용지 면적(㎡)",
    "유원지 면적(㎡)",
    "종교용지 면적(㎡)",
    "잡종지 면적(㎡)",
    "답 면적(㎡)",
    "과수원 면적(㎡)",
    "목장용지 면적(㎡)",
    "임야 면적(㎡)",
    "전 면적(㎡)",
]

# 실제 df에 존재하는 컬럼만 사용 (오타/누락 방지)
area_cols_sel = [c for c in TARGET_AREA_COLS if c in df.columns]
missing_targets = set(TARGET_AREA_COLS) - set(area_cols_sel)
if missing_targets:
    print("\n⚠ 경고: 다음 면적 컬럼이 데이터에 없습니다:", missing_targets)

# 분모: 각 지역별 전체 면적(합계 면적)을 사용
denom_area = df[col_total].replace(0, np.nan)  # col_total = "합계 면적(㎡)"

# 분모: 선택된 19개(또는 존재하는 부분집합)의 합
#denom_area = df[area_cols_sel].sum(axis=1)
#denom_area = denom_area.replace(0, np.nan)

ratio_cols = []
for c in area_cols_sel:
    new_name = c.replace(" 면적(㎡)", " 비율")
    df[new_name] = df[c] / denom_area
    ratio_cols.append(new_name)

# 전부 NaN인 비율 컬럼은 제거
ratio_cols = [c for c in ratio_cols if df[c].notna().any()]

# 비율 합 확인 (디버깅용)
print("\n선택된 면적 비율 합(앞 5행):")
print(df[ratio_cols].sum(axis=1).head())

# ===== 5) 인구밀도 계산 (명 / km²) =====
df["pop_density"] = df["인구"] / (df[col_total] / 1_000_000.0)

print("\n예시 행:")
sample_cols = [col_year, col_region] + ratio_cols[:8] + ["pop_density"]
print(df[sample_cols].head())

# ===== 6) KMeans 군집 (k=3) – 선택 변수만 사용 =====
# 특징: 지정 19개 면적 비율만 사용 + (원하시면 pop_density 포함)
# → 질문은 "이 변수만"이었지만, 인구밀도는 그대로 유지해둘게요.
#   만약 진짜로 면적비율만 쓰고 싶으면 feature_cols = ratio_cols 로 바꾸면 됨.
feature_cols = ratio_cols + ["pop_density"]

# NaN 체크
nan_mask = df[feature_cols].isna().any(axis=1)
print("\nNaN이 포함된 행 수:", nan_mask.sum())
if nan_mask.sum() > 0:
    print(df.loc[nan_mask, [col_year, col_region]].head(10))

# NaN 있는 행은 군집 분석에서 제외
df = df[~nan_mask].copy()

X = df[feature_cols].values

# 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans (한 번만)
kmeans = KMeans(n_clusters=3, random_state=0, n_init=50)
clusters = kmeans.fit_predict(X_scaled)

df["cluster"] = clusters

# 클러스터 중심(centroid)을 원래 스케일로 복원
centers_scaled = kmeans.cluster_centers_
centers = scaler.inverse_transform(centers_scaled)

centers_df = pd.DataFrame(centers, columns=feature_cols)
centers_df["cluster"] = range(3)

# ===== 6-1) U(도시/산업), G(농업/산림) 지표 계산 =====
def safe_cols(cols):
    return [c for c in cols if c in centers_df.columns]

# 도시/산업 관련 (질문에서 주신 것 중 도시/비농림 성격)
urban_ratio_cols = safe_cols([
    "대 비율",
    "공장용지 비율",
    "도로 비율",
    "철도용지 비율",
    "주차장 비율",
    "주유소용지 비율",
    "창고용지 비율",
    "학교용지 비율",
    "수도용지 비율",
    "공원 비율",
    "체육용지 비율",
    "유원지 비율",
    "종교용지 비율",
    "잡종지 비율",
])

# 농업/산림 관련
agri_forest_ratio_cols = safe_cols([
    "임야 비율",
    "전 비율",
    "답 비율",
    "과수원 비율",
    "목장용지 비율",
])

centers_df["U_도시산업"] = centers_df[urban_ratio_cols].sum(axis=1) if urban_ratio_cols else 0.0
centers_df["G_농업산림"] = centers_df[agri_forest_ratio_cols].sum(axis=1) if agri_forest_ratio_cols else 0.0

print("\n=== 클러스터별 U/G 지표 ===")
print(centers_df[["cluster", "U_도시산업", "G_농업산림"]])

# ===== 7) 클러스터 라벨링 (도시/산업형, 농업/산림형, 균형형) =====
idx_urban = centers_df["U_도시산업"].idxmax()
cluster_urban = int(centers_df.loc[idx_urban, "cluster"])

idx_agri  = centers_df["G_농업산림"].idxmax()
cluster_agri = int(centers_df.loc[idx_agri, "cluster"])

all_clusters = set(centers_df["cluster"].tolist())
cluster_balanced = list(all_clusters - {cluster_urban, cluster_agri})[0]

cluster_label_map = {
    cluster_urban:    "도시/산업형",
    cluster_agri:     "농업/산림형",
    cluster_balanced: "균형형",
}

print("\n클러스터 라벨 매핑:", cluster_label_map)

df["유형"] = df["cluster"].map(cluster_label_map)

# =====================================================================
# 7-1) Softmax 기반 유형별 확률 계산 (KMeans 한 번 기준)
# =====================================================================
distances = kmeans.transform(X_scaled)   # 유클리드 거리

def softmax_neg_dist(d):
    s = -d
    s = s - np.max(s)   # overflow 방지
    exp_s = np.exp(s)
    return exp_s / np.sum(exp_s)

probs = np.apply_along_axis(softmax_neg_dist, 1, distances)  # shape (N, 3)

type_to_cluster = {v: k for k, v in cluster_label_map.items()}

idx_u = type_to_cluster["도시/산업형"]
idx_g = type_to_cluster["농업/산림형"]
idx_b = type_to_cluster["균형형"]

df["P_도시산업형"] = probs[:, idx_u]
df["P_농업산림형"] = probs[:, idx_g]
df["P_균형형"]     = probs[:, idx_b]

# =====================================================================
# 8) PCA 2D 시각화 (전체, 라벨은 "최신 연도"만 표시 / KMeans는 1개)
# =====================================================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

plt.figure(figsize=(9, 7))

type_colors = {
    "도시/산업형": "#EA9358",  # 주황
    "농업/산림형": "#75CD97",  # 초록
    "균형형":     "#589AEA",  # 파랑
}

for t, sub in df.groupby("유형"):
    plt.scatter(
        sub["PC1"],
        sub["PC2"],
        label=t,
        color=type_colors.get(t, "gray"),
        alpha=0.9,
        s=60,
    )
    latest_year = df[col_year].max()
    sub_latest = sub[sub[col_year] == latest_year]
    for _, row in sub_latest.iterrows():
        label_txt = row[col_region]
        plt.text(
            row["PC1"] + 0.02,
            row["PC2"] + 0.02,
            label_txt,
            fontsize=8
        )

plt.axhline(0, color="gray", linewidth=0.5)
plt.axvline(0, color="gray", linewidth=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"충북 토지이용 군집 (KMeans, {df[col_year].min()}~{df[col_year].max()}년,\n선택 19개 면적 비율 + 인구밀도)")
plt.legend(title="지역 유형")
plt.tight_layout()

pca_outfile = os.path.join(base_dir, "cluster_pca_selected19_area_ratio_pop.png")
plt.savefig(pca_outfile, dpi=200)
plt.close()

print("\nPCA 시각화 이미지 저장:", pca_outfile)

# =====================================================================
# 9) 결과 CSV 저장
# =====================================================================
out_csv = os.path.join(base_dir, "chungbuk_clusters_selected19_area_ratio_softmax_pop.csv")

save_cols = (
    [col_year, col_region] +
    [col_total] +
    area_cols_sel +                # 선택된 원래 면적
    ratio_cols +                   # 선택된 면적 비율
    ["인구", "pop_density"] +
    ["cluster", "유형",
     "P_도시산업형", "P_농업산림형", "P_균형형",
     "PC1", "PC2"]
)

df[save_cols].to_csv(out_csv, index=False, encoding="cp949")
print("군집 + Softmax + 선택 19개 면적비율 + 인구밀도 결과 CSV 저장:", out_csv)

print("\n=== 군집 분석(KMeans+PCA+Softmax, 선택 19개 면적 변수 + 인구밀도) 완료 ===")
