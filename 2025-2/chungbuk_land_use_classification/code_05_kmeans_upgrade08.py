# -*- coding: utf-8 -*-
"""
충북 토지이용 군집 분석 (KMeans + PCA + Softmax 확률)

- 대상: chungbuk_data_2015~2025 (연도별 CSV, 연말 기준)
- 특징:
    · 모든 '... 면적(㎡)' 비율
    · 인구밀도(pop_density)
- 군집: KMeans(k=3)  ※ 전체 연도 통합 1회
- 시각화: PCA(2D)
- 추가: Softmax 기반 유형별 확률
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# =========================================================
# 0. 기본 설정
# =========================================================
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
os.environ["OMP_NUM_THREADS"] = "1"

base_dir = r"C:/Users/leebi/OneDrive/바탕 화면/team_project"
landuse_pattern = os.path.join(base_dir, "chungbuk_data_*.csv")
pop_xlsx_path = os.path.join(base_dir, "chungbuk_population.xlsx")

col_region = "토지소재명"
col_year   = "year"
col_total  = "합계 면적(㎡)"

# =========================================================
# 1. 토지이용 CSV 전체 로드 (연도 자동 추출)
# =========================================================
landuse_files = sorted(glob.glob(landuse_pattern))
if not landuse_files:
    raise FileNotFoundError("토지이용 CSV 파일을 찾지 못했습니다.")

df_list = []

for path in landuse_files:
    name = os.path.basename(path)

    # 🔹 정규식으로 연도 추출 (2015~2025 안전)
    m = re.search(r"(20\d{2})", name)
    if not m:
        print(f"⚠ 연도 추출 실패, 스킵: {name}")
        continue
    year = int(m.group(1))

    # 인코딩 안전 처리
    try:
        tmp = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            tmp = pd.read_csv(path, encoding="cp949")
        except UnicodeDecodeError:
            tmp = pd.read_csv(path, encoding="euc-kr")

    tmp[col_year] = year
    df_list.append(tmp)

df = pd.concat(df_list, ignore_index=True)

print("📌 연도 목록:", sorted(df[col_year].unique()))
print("📌 전체 데이터 크기:", df.shape)

# =========================================================
# 2. 숫자 컬럼 정리
# =========================================================
num_cols = [c for c in df.columns if c != col_region]

for c in num_cols:
    df[c] = (
        df[c]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df[c] = pd.to_numeric(df[c], errors="coerce")

# =========================================================
# 3. 인구 데이터 병합
# =========================================================
pop_raw = pd.read_excel(pop_xlsx_path)

pop_long = pop_raw.melt(
    id_vars=["연도"],
    var_name="year",
    value_name="인구"
)
pop_long["year"] = pop_long["year"].astype(int)
pop_long = pop_long.rename(columns={"연도": col_region})

df = df.merge(pop_long, on=[col_region, col_year], how="left")

# =========================================================
# 4. 모든 면적 비율 계산 (분모 = 합계 면적)
# =========================================================
area_cols = [c for c in df.columns if c.endswith("면적(㎡)") and c != col_total]

denom = df[col_total].replace(0, np.nan)

ratio_cols = []
for c in area_cols:
    new_c = c.replace(" 면적(㎡)", " 비율")
    df[new_c] = df[c] / denom
    ratio_cols.append(new_c)

# =========================================================
# 5. 인구밀도 계산
# =========================================================
df["pop_density"] = df["인구"] / (df[col_total] / 1_000_000)

# =========================================================
# 6. 군집용 데이터 정리 (NaN 행 제거)
# =========================================================
feature_cols = ratio_cols + ["pop_density"]

nan_mask = df[feature_cols].isna().any(axis=1)
print("⚠ NaN 포함 행 제거:", nan_mask.sum())

df = df[~nan_mask].copy()

X = df[feature_cols].values

# =========================================================
# 7. 스케일링 + KMeans (1회)
# =========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=0, n_init=50)
df["cluster"] = kmeans.fit_predict(X_scaled)

# =========================================================
# 8. 클러스터 유형 정의 (U/G 지표)
# =========================================================
centers = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(centers, columns=feature_cols)
centers_df["cluster"] = range(3)

urban_cols = [c for c in centers_df.columns if any(
    k in c for k in ["대 비율", "공장", "주차장", "주유소", "창고", "도로", "철도"]
)]

agri_cols = [c for c in centers_df.columns if any(
    k in c for k in ["임야", "전 비율", "답 비율", "과수원", "목장"]
)]

centers_df["U"] = centers_df[urban_cols].sum(axis=1)
centers_df["G"] = centers_df[agri_cols].sum(axis=1)

cluster_urban = centers_df.loc[centers_df["U"].idxmax(), "cluster"]
cluster_agri  = centers_df.loc[centers_df["G"].idxmax(), "cluster"]
cluster_bal   = list(set(range(3)) - {cluster_urban, cluster_agri})[0]

label_map = {
    cluster_urban: "도시/산업형",
    cluster_agri: "농업/산림형",
    cluster_bal: "균형형"
}

df["유형"] = df["cluster"].map(label_map)

# =========================================================
# 9. Softmax 확률 계산
# =========================================================
dist = kmeans.transform(X_scaled)

def softmax_neg(d):
    s = -d
    s -= np.max(s)
    e = np.exp(s)
    return e / e.sum()

probs = np.apply_along_axis(softmax_neg, 1, dist)

rev = {v: k for k, v in label_map.items()}

df["P_도시산업형"] = probs[:, rev["도시/산업형"]]
df["P_농업산림형"] = probs[:, rev["농업/산림형"]]
df["P_균형형"]     = probs[:, rev["균형형"]]

# =========================================================
# 10. PCA 시각화
# =========================================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

plt.figure(figsize=(9, 7))
colors = {
    "도시/산업형": "#EA9358",
    "농업/산림형": "#75CD97",
    "균형형": "#589AEA",
}

for t, sub in df.groupby("유형"):
    plt.scatter(sub["PC1"], sub["PC2"], label=t, s=60, alpha=0.9, color=colors[t])

latest = df[col_year].max()
for _, r in df[df[col_year] == latest].iterrows():
    plt.text(r["PC1"]+0.02, r["PC2"]+0.02, r[col_region], fontsize=8)

plt.legend()
plt.title(f"충북 토지이용 군집 ({df[col_year].min()}~{latest})")
plt.tight_layout()

plt.savefig(os.path.join(base_dir, "cluster_pca_all_area_ratio_pop.png"), dpi=200)
plt.close()

# ===== 저장 전 year 컬럼 존재 확인/강제 =====
print("저장 직전 컬럼:", df.columns.tolist()[:20], "...")  # 일부만 보기
if col_year not in df.columns:
    raise RuntimeError(f"❌ '{col_year}' 컬럼이 df에 없습니다. (연도 부여 단계가 누락됨)")

# year를 첫 열로 보내기(엑셀에서 안 보이는 문제 방지)
df = df[[col_year] + [c for c in df.columns if c != col_year]]


# =========================================================
# 11. 결과 저장
# =========================================================
out_csv = os.path.join(base_dir, "chungbuk_clusters_all_area_ratio_softmax_pop.csv")
df.to_csv(out_csv, index=False, encoding="cp949")

print("✅ 분석 완료")
print("📁 결과 CSV:", out_csv)
