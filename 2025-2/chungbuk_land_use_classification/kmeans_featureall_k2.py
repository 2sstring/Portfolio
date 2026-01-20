# -*- coding: utf-8 -*-
"""
충북 토지이용 군집 분석 (KMeans + PCA + Softmax 확률, k=2)
 - 대상: chungbuk_data_2015~2025 (연도별 CSV, 연말 기준)
 - 특징: 모든 '... 면적(㎡)' 비율 + 인구밀도 (도로율은 제외)
 - 군집: KMeans(k=2)
 - 시각화: PCA(2D) + 클러스터 색상
 - 추가: 각 유형(도시/산업형, 농업/산림형)에 대한 softmax 확률 계산
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
base_dir = r"data"

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

# ===== 4) 모든 면적(㎡) 컬럼 → 비율 컬럼으로 변환 =====
area_cols = [c for c in df.columns if c.endswith("면적(㎡)")]
if col_total in area_cols:
    area_cols_no_total = [c for c in area_cols if c != col_total]
else:
    area_cols_no_total = area_cols.copy()

denom_area = df[area_cols_no_total].sum(axis=1)
denom_area = denom_area.replace(0, np.nan)

ratio_cols = []
for c in area_cols_no_total:
    new_name = c.replace(" 면적(㎡)", " 비율")
    df[new_name] = df[c] / denom_area
    ratio_cols.append(new_name)

# 🔹 전부 NaN인 비율 컬럼은 제거
ratio_cols = [c for c in ratio_cols if df[c].notna().any()]

# 비율 합 확인 (디버깅용)
print("\n비율 합(앞 5행):")
print(df[ratio_cols].sum(axis=1).head())

# ===== 5) 인구밀도 계산 (명 / km²) =====
df["pop_density"] = df["인구"] / (df[col_total] / 1_000_000.0)

# ===== 6) KMeans 군집 (k=2) – 모든 연도 =====
# 특징: 모든 면적 비율 + 인구밀도
feature_cols = ratio_cols + ["pop_density"]

# 👉 군집에서 행이 날아가지 않도록, 비율/인구밀도의 NaN은 0으로 채운다
df[feature_cols] = df[feature_cols].fillna(0)

print("\n예시 행:")
sample_cols = [col_year, col_region] + ratio_cols[:8] + ["pop_density"]
print(df[sample_cols].head())

X = df[feature_cols].values

# 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- KMeans (k=2, n_init=50) ---
kmeans = KMeans(n_clusters=2, random_state=0, n_init=50)
clusters = kmeans.fit_predict(X_scaled)

df["cluster"] = clusters

# 클러스터 중심(centroid)을 원래 스케일로 복원
centers_scaled = kmeans.cluster_centers_
centers = scaler.inverse_transform(centers_scaled)

centers_df = pd.DataFrame(centers, columns=feature_cols)
centers_df["cluster"] = range(2)

# ===== 6-1) U(도시/산업), G(농업/산림) 지표 계산 =====
def safe_cols(cols):
    return [c for c in cols if c in centers_df.columns]

urban_ratio_cols = safe_cols([
    "대 비율",
    "공장용지 비율",
    "주차장 비율",
    "주유소용지 비율",
    "창고용지 비율",
    "상업용지 비율",   # 있으면 사용
    "주거용지 비율",   # 있으면 사용
])

agri_forest_ratio_cols = safe_cols([
    "임야 비율",
    "전 비율",
    "답 비율",
    "과수원 비율",
    "목장용지 비율",
    "광천지 비율",
    "염전 비율",
])

centers_df["U_도시산업"] = centers_df[urban_ratio_cols].sum(axis=1) if urban_ratio_cols else 0.0
centers_df["G_농업산림"] = centers_df[agri_forest_ratio_cols].sum(axis=1) if agri_forest_ratio_cols else 0.0

print("\n=== 클러스터별 U/G 지표 ===")
print(centers_df[["cluster", "U_도시산업", "G_농업산림"]])

# ===== 7) 클러스터 라벨링 (k=2: 도시/산업형, 농업/산림형) =====
# U_도시산업이 큰 쪽 → 도시/산업형, 나머지 → 농업/산림형
idx_urban = centers_df["U_도시산업"].idxmax()
cluster_urban = int(centers_df.loc[idx_urban, "cluster"])
all_clusters = set(centers_df["cluster"].tolist())
cluster_agri = list(all_clusters - {cluster_urban})[0]

cluster_label_map = {
    cluster_urban: "도시/산업형",
    cluster_agri:  "농업/산림형",
}

print("\n클러스터 라벨 매핑 (k=2):", cluster_label_map)

df["유형"] = df["cluster"].map(cluster_label_map)

# =====================================================================
# 7-1) Softmax 기반 유형별 확률 계산 (2개 클러스터)
# =====================================================================
# (1) 각 샘플의 클러스터까지의 거리 행렬 (N x 2)
distances = kmeans.transform(X_scaled)   # 유클리드 거리

# (2) softmax 계산 함수
def softmax_neg_dist(d):
    s = -d
    s = s - np.max(s)   # overflow 방지
    exp_s = np.exp(s)
    return exp_s / np.sum(exp_s)

# (3) 모든 샘플에 대해 softmax 확률 계산
probs = np.apply_along_axis(softmax_neg_dist, 1, distances)  # shape (N, 2)

# (4) cluster 인덱스 → 유형 이름 역매핑
type_to_cluster = {v: k for k, v in cluster_label_map.items()}

idx_u = type_to_cluster["도시/산업형"]
idx_g = type_to_cluster["농업/산림형"]

# (5) 유형별 softmax 확률 컬럼 추가
df["P_도시산업형"] = probs[:, idx_u]
df["P_농업산림형"] = probs[:, idx_g]

# =====================================================================
# 8) PCA 2D 시각화 (전체, 라벨은 "최신 연도"만 표시)
# =====================================================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

plt.figure(figsize=(9, 7))

type_colors = {
    "도시/산업형": "#EA9358",  # 주황
    "농업/산림형": "#75CD97",  # 초록
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
    # 텍스트는 "가장 최근 연도" 데이터에 대해서만
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
plt.title(
    f"충북 토지이용 군집 (KMeans k=2, {df[col_year].min()}~{df[col_year].max()}년,\n"
    "모든 토지이용 면적 비율 + 인구밀도)"
)
plt.legend(title="지역 유형")
plt.tight_layout()

pca_outfile = os.path.join(base_dir, "clusters_featureall_k2.png")
plt.savefig(pca_outfile, dpi=200)
plt.close()

print("\nPCA 시각화 이미지 저장:", pca_outfile)

# =====================================================================
# 9) 결과 CSV 저장
# =====================================================================
out_csv = os.path.join(base_dir, "clusters_featureall_k2_softmax.csv")

save_cols = (
    [col_year, col_region] +
    [col_total] +
    area_cols_no_total +           # 원래 면적
    ratio_cols +                   # 면적 비율
    ["인구", "pop_density"] +
    ["cluster", "유형",
     "P_도시산업형", "P_농업산림형",
     "PC1", "PC2"]
)

df[save_cols].to_csv(out_csv, index=False, encoding="cp949")
print("군집 + Softmax + 전체 면적비율 + 인구밀도 결과 CSV 저장:", out_csv)

print("\n=== 군집 분석(KMeans k=2 + PCA + Softmax, chungbuk_data_2015~2024, 모든 면적 비율 + 인구밀도, 도로율 제외) 완료 ===")

