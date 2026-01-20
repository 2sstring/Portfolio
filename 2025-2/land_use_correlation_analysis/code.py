# -*- coding: utf-8 -*-
"""
충청북도 토지이용 분석 (2015–2024)
 - 임야 / 농경지(전+답) / 대지 / 공장용지 비율
 - 결과 컬럼: 한글 이름 (ratio_~~ 안 나옴)
 - CSV: cp949 인코딩 (엑셀 한글 안 깨지게)
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# (선택) KMeans 메모리 누수 경고 완화
os.environ["OMP_NUM_THREADS"] = "1"

# ===== 한글 폰트 설정 (Windows 기준) =====
plt.rcParams['font.family'] = 'Malgun Gothic'   # 맑은 고딕
plt.rcParams['axes.unicode_minus'] = False      # 마이너스 깨짐 방지

# ===== 1. 데이터 파일 로드 =====
# 예: chungbuk_data_20151231.csv ~ chungbuk_data_20241231.csv
files = sorted(glob.glob("chungbuk_data_*.csv"))

col_region  = "토지소재명"
col_total   = "합계 면적(㎡)"
col_forest  = "임야 면적(㎡)"
col_jeon    = "전 면적(㎡)"
col_dap     = "답 면적(㎡)"
col_dae     = "대 면적(㎡)"           # 대지
col_factory = "공장용지 면적(㎡)"

records = []

for f in files:
    # 파일명에서 연도 추출 (예: chungbuk_data_20151231.csv → 2015)
    basename = os.path.basename(f)
    year = int(basename[-12:-8])

    df = pd.read_csv(f)

    # 숫자에서 콤마 제거 후 float 변환
    for c in [col_total, col_forest, col_jeon, col_dap, col_dae, col_factory]:
        df[c] = df[c].astype(str).str.replace(",", "").astype(float)

    tmp = df[[col_region, col_total, col_forest, col_jeon, col_dap, col_dae, col_factory]].copy()
    tmp["year"] = year

    # ---- 여기서부터는 "비율" 계산 (총면적 대비)
    agri_area = tmp[col_jeon] + tmp[col_dap]   # 농경지 = 전 + 답

    tmp["forest_ratio"]  = tmp[col_forest]  / tmp[col_total]    # 임야 비율
    tmp["agri_ratio"]    = agri_area        / tmp[col_total]    # 농경지 비율(전+답)
    tmp["dae_ratio"]     = tmp[col_dae]     / tmp[col_total]    # 대지 비율
    tmp["factory_ratio"] = tmp[col_factory] / tmp[col_total]    # 공장용지 비율

    records.append(tmp)

all_data = pd.concat(records, ignore_index=True)

# 분석에 쓸 내부용 컬럼(영문) –> 나중에 한글 이름으로 바꿔서 출력
ratio_cols_internal = ["forest_ratio", "agri_ratio", "dae_ratio", "factory_ratio"]

# ===== 2. 2024년 데이터 기준 정리 =====
data_2024 = all_data[all_data["year"] == 2024].copy()

# 2024년 기준 "한글 컬럼명"으로 매핑 (비율 값이지만 컬럼 이름을 한글로)
data_2024["임야비율"]          = data_2024["forest_ratio"]
data_2024["농경지비율(전+답)"] = data_2024["agri_ratio"]
data_2024["대지비율"]          = data_2024["dae_ratio"]
data_2024["공장용지비율"]      = data_2024["factory_ratio"]

korean_ratio_cols = ["임야비율", "농경지비율(전+답)", "대지비율", "공장용지비율"]

# 순위는 대지/공장 비율 기준 (내림차순)
data_2024["대지비율_순위"]     = data_2024["대지비율"].rank(ascending=False, method="min")
data_2024["공장용지비율_순위"] = data_2024["공장용지비율"].rank(ascending=False, method="min")

# ===== 3. 상관 / 회귀 (2015–2024 전체) =====
X = all_data[["dae_ratio"]].values           # 대지 비율
y = all_data["factory_ratio"].values         # 공장용지 비율

corr = np.corrcoef(all_data["dae_ratio"], all_data["factory_ratio"])[0, 1]

linreg = LinearRegression()
linreg.fit(X, y)
slope = float(linreg.coef_[0])
intercept = float(linreg.intercept_)

# ===== 4. 군집분석 (2024년 기준, 임야/농경지/대지/공장 비율) =====
features_2024 = data_2024[["forest_ratio", "agri_ratio", "dae_ratio", "factory_ratio"]].values

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_2024)

kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
clusters = kmeans.fit_predict(features_scaled)
data_2024["cluster"] = clusters

# PCA 2D
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(features_scaled)
data_2024["PC1"] = pca_coords[:, 0]
data_2024["PC2"] = pca_coords[:, 1]

# ===== 5. PDF 리포트 저장 경로 (현재 폴더에 저장) =====
pdf_path = "chungbuk_landuse_report.pdf"
pp = PdfPages(pdf_path)

# 5-1. 2024년 용도별 구성비 (임야/농경지/대지/공장) – 누적 막대그래프
plt.figure(figsize=(10, 5))
regions = data_2024[col_region].tolist()
bottom = np.zeros(len(regions))

for col in korean_ratio_cols:
    plt.bar(regions, data_2024[col].values, bottom=bottom, label=col)
    bottom += data_2024[col].values

plt.xticks(rotation=45, ha="right")
plt.ylabel("비율")
plt.title("충북 14개 행정구역 토지이용 구성비 (2024년)")
plt.legend()
plt.tight_layout()
pp.savefig()
plt.close()

# 5-2. 대지비율 vs 공장용지비율 산점도 + 회귀선
plt.figure(figsize=(6, 6))
plt.scatter(all_data["dae_ratio"], all_data["factory_ratio"], alpha=0.7)
x_line = np.linspace(all_data["dae_ratio"].min(), all_data["dae_ratio"].max(), 100)
y_line = slope * x_line + intercept
plt.plot(x_line, y_line, color="red")

plt.xlabel("대지비율")
plt.ylabel("공장용지비율")
plt.title(f"대지비율–공장용지비율 상관관계 (2015–2024, r={corr:.3f})")
plt.tight_layout()
pp.savefig()
plt.close()

# 5-3. PCA 군집 시각화
plt.figure(figsize=(6, 6))
for cl in sorted(data_2024["cluster"].unique()):
    sub = data_2024[data_2024["cluster"] == cl]
    plt.scatter(sub["PC1"], sub["PC2"], label=f"Cluster {cl}")
    for _, row in sub.iterrows():
        plt.text(row["PC1"], row["PC2"], row[col_region], fontsize=7)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("충북 14개 행정구역 토지이용 군집 (임야/농경지/대지/공장 비율, 2024년)")
plt.legend()
plt.tight_layout()
pp.savefig()
plt.close()

pp.close()

# ===== 6. CSV 저장 (cp949 인코딩, Excel에서 한글 안 깨지게) =====
summary_cols = [col_region] + korean_ratio_cols + ["cluster", "대지비율_순위", "공장용지비율_순위"]

summary_csv_path = "chungbuk_landuse_summary_2024.csv"
data_2024[summary_cols].to_csv(summary_csv_path, index=False, encoding="cp949")

print("=== 분석 완료 ===")
print("PDF 리포트 :", os.path.abspath(pdf_path))
print("요약 CSV   :", os.path.abspath(summary_csv_path))
print(f"상관계수 r = {corr:.3f}, 회귀식: 공장용지비율 = {slope:.3f} * 대지비율 + {intercept:.4f}")
