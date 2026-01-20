# -*- coding: utf-8 -*-
"""
충북 토지이용 군집 분석 (KMeans + PCA + Softmax 확률)
 - 대상: 2015~2025년 전체, (연도, 행정구역) 단위
 - 특징: 임야/농경지/대지/공장용지 비율 (4차원)
 - 군집: KMeans(k=2)
 - 시각화: PCA(2D) + 클러스터 색상
 - 각 유형(도시/산업형, 농업/산림형)에 대한 softmax 확률 계산
"""

import os
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

# ===== 1) 데이터 로드 =====
base_dir = r"C:/Users/leebi/OneDrive/바탕 화면/team_project"
detail_csv_path = os.path.join(base_dir, "chungbuk_landuse_composition_2015_2025_detail.csv")

df = pd.read_csv(detail_csv_path, encoding="cp949")

# '충청북도 합계' 제외
df = df[df["토지소재명"] != "충청북도 합계"].copy().reset_index(drop=True)

col_region  = "토지소재명"
col_year    = "year"
col_forest  = "임야 면적(㎡)"
col_agri    = "농경지 면적(㎡)"
col_dae     = "대 면적(㎡)"
col_factory = "공장용지 면적(㎡)"

df[col_year] = df[col_year].astype(int)

# ===== 2) 2015~2025 전체에 대해 비율 계산 =====
col_total = "합계 면적(㎡)"

denom = df[col_total]  # 지역·연도별 전체 면적

#denom = df[col_forest] + df[col_agri] + df[col_dae] + df[col_factory]

df["임야 비율"]      = df[col_forest]  / denom
df["농경지 비율"]    = df[col_agri]    / denom
df["대지 비율"]      = df[col_dae]     / denom
df["공장용지 비율"]  = df[col_factory] / denom

print("전체 행 수:", len(df))
print(df[[col_year, col_region, "임야 비율", "농경지 비율", "대지 비율", "공장용지 비율"]].head())

# ===== 3) KMeans 군집 (k=2) – 2015~2025 전체 =====
feature_cols = ["임야 비율", "농경지 비율", "대지 비율", "공장용지 비율"]
X = df[feature_cols].values

# 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans (k=2)
kmeans = KMeans(n_clusters=2, random_state=0, n_init=50)
clusters = kmeans.fit_predict(X_scaled)

df["cluster"] = clusters

# 클러스터 중심(centroid)을 원래 비율 스케일로 복원
centers_scaled = kmeans.cluster_centers_
centers = scaler.inverse_transform(centers_scaled)

centers_df = pd.DataFrame(centers, columns=feature_cols)
centers_df["cluster"] = range(2)

# U, G 지표 계산
centers_df["U_도시산업"] = centers_df["대지 비율"] + centers_df["공장용지 비율"]
centers_df["G_농업산림"] = centers_df["임야 비율"] + centers_df["농경지 비율"]

print("\n=== (2015~2025 전체 기준) 클러스터별 평균 비율 ===")
print(centers_df)

# ===== 4) 클러스터 라벨링 (도시/산업형, 농업/산림형) =====
idx_urban = centers_df["U_도시산업"].idxmax()
cluster_urban = int(centers_df.loc[idx_urban, "cluster"])

idx_agri = centers_df["G_농업산림"].idxmax()
cluster_agri = int(centers_df.loc[idx_agri, "cluster"])

# 혹시 둘이 같은 클러스터로 잡히면(아주 드문 경우) 나머지 하나를 농업형으로 지정
if cluster_urban == cluster_agri:
    other = list(set(centers_df["cluster"].tolist()) - {cluster_urban})[0]
    cluster_agri = other

cluster_label_map = {
    cluster_urban: "도시/산업형",
    cluster_agri:  "농업/산림형",
}

print("\n클러스터 라벨 매핑:", cluster_label_map)

df["유형"] = df["cluster"].map(cluster_label_map)

# =====================================================================
# 4-1) Softmax 기반 유형별 확률 계산 (2개 군집)
# =====================================================================

# (1) 각 샘플의 클러스터까지의 거리 행렬 (N x 2)
distances = kmeans.transform(X_scaled)   # 유클리드 거리

# (2) softmax 계산 함수 (길이가 2인 벡터에도 그대로 동작)
def softmax_neg_dist(d):
    # d: shape (2,) – 각 클러스터까지의 거리
    s = -d
    s = s - np.max(s)   # overflow 방지용
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
# 5) PCA 2D 시각화 (2015~2025 전체, 라벨은 2025년만)
# =====================================================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

plt.figure(figsize=(9, 7))

type_colors = {
    "도시/산업형": "#EA9358",  # red-ish
    "농업/산림형": "#75CD97",  # green-ish
}

for t, sub in df.groupby("유형"):
    # 점은 모든 연도(2015~2025)를 그림
    plt.scatter(
        sub["PC1"],
        sub["PC2"],
        label=t,
        color=type_colors.get(t, "gray"),
        alpha=0.9,
        s=60,
    )
    # 텍스트는 2025년 데이터에 대해서만, 연도 없이 지역명만 표시
    sub_2025 = sub[sub[col_year] == 2025]
    for _, row in sub_2025.iterrows():
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
plt.title("충북 토지이용 군집 (KMeans k=2, 2015~2025년, 임야/농경지/대지/공장 비율)")
plt.legend(title="지역 유형")
plt.tight_layout()

pca_outfile = os.path.join(base_dir, "cluster_pca_2015_2025_k2.png")
plt.savefig(pca_outfile, dpi=200)
plt.close()

print("\nPCA 시각화 이미지 저장:", pca_outfile)

# =====================================================================
# 6) 결과 CSV 저장 (모든 연도 + softmax 확률)
# =====================================================================
out_csv = os.path.join(base_dir, "chungbuk_clusters_2015_2025_softmax_k2.csv")
save_cols = [
    col_year, col_region,
    col_forest, col_agri, col_dae, col_factory,
    "임야 비율", "농경지 비율", "대지 비율", "공장용지 비율",
    "cluster", "유형",
    "P_도시산업형", "P_농업산림형",
    "PC1", "PC2",
]
df[save_cols].to_csv(out_csv, index=False, encoding="cp949")
print("군집 + Softmax 결과 CSV 저장:", out_csv)

print("\n=== 군집 분석(KMeans+PCA+Softmax, 2015~2025 전체, k=2) 완료 ===")

