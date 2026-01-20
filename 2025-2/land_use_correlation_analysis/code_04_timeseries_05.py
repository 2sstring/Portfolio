# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 02:51:16 2025

@author: bigbell
"""

# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt

# ===== 한글 폰트 설정 =====
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 공통 색상 지정
color_map = {
    "임야 비율": "#75CD97",     # 초록
    "농경지 비율": "#E6C950",   # 노랑
    "대지 비율": "#EA9358",     # 주황
    "공장용지 비율": "#8FA1BB", # 회색
}
ROAD_COLOR = "#589AEA"          # 도로율: 파랑 계열

# ===== 1) 데이터 로드 =====
base_dir = r"C:/Users/leebi/OneDrive/바탕 화면/team_project"
detail_csv_path = os.path.join(base_dir, "chungbuk_landuse_composition_2015_2025_detail.csv")
road_xlsx_path  = os.path.join(base_dir, "chungbuk_road_ratio_2015_2025.xlsx")

df = pd.read_csv(detail_csv_path, encoding="cp949")

# '충청북도 합계'는 일단 빼고, 시·군·구 데이터만 사용
df = df[df["토지소재명"] != "충청북도 합계"].copy().reset_index(drop=True)

# 도로율 엑셀 로드 (sheet='detail')
road_detail = pd.read_excel(road_xlsx_path, sheet_name="detail")

# 필요 컬럼만 사용 (year, 토지소재명, 도로율)
if "year" not in road_detail.columns:
    raise ValueError("도로율 엑셀에 'year' 컬럼이 없습니다.")
if "토지소재명" not in road_detail.columns:
    raise ValueError("도로율 엑셀에 '토지소재명' 컬럼이 없습니다.")
if "도로율" not in road_detail.columns:
    raise ValueError("도로율 엑셀에 '도로율' 컬럼이 없습니다.")

# 연도/행정구역 기준으로 도로율 merge
df = df.merge(
    road_detail[["year", "토지소재명", "도로율"]],
    on=["year", "토지소재명"],
    how="left"
)

# 컬럼 이름
col_region  = "토지소재명"
col_year    = "year"
col_forest  = "임야 면적(㎡)"
col_agri    = "농경지 면적(㎡)"
col_dae     = "대 면적(㎡)"
col_factory = "공장용지 면적(㎡)"

df[col_year] = df[col_year].astype(int)

print("데이터 미리보기:")
print(df[[col_region, col_year, col_forest, col_agri, col_dae, col_factory, "도로율"]].head())

# ===== 2) 비율 재계산 (임야+농경지+대지+공장용지 합 기준) =====
col_total = "합계 면적(㎡)"
denom = df[col_total]

#denom = df[col_forest] + df[col_agri] + df[col_dae] + df[col_factory]

df["임야 비율"]      = df[col_forest]  / denom
df["농경지 비율"]    = df[col_agri]    / denom
df["대지 비율"]      = df[col_dae]     / denom
df["공장용지 비율"]  = df[col_factory] / denom

print("비율 데이터 미리보기:")
print(df[[col_region, col_year, "임야 비율", "농경지 비율", "대지 비율", "공장용지 비율", "도로율"]].head())

# ===== 3) 지역 목록 =====
regions = sorted(df[col_region].unique())
print("지역 개수:", len(regions))
print("지역 목록:", regions)

# ===== 4) 지역별 2015~2025 비율 + 도로율 변화 그래프 =====
for region in regions:
    sub = df[df[col_region] == region].copy()
    sub = sub.sort_values(col_year)

    years         = sub[col_year].values
    forest_ratio  = sub["임야 비율"].values
    agri_ratio    = sub["농경지 비율"].values
    dae_ratio     = sub["대지 비율"].values
    factory_ratio = sub["공장용지 비율"].values
    road_ratio    = sub["도로율"].values  # 도로율

    plt.figure(figsize=(9, 5))

    # 토지 비율
    plt.plot(
        years, forest_ratio,
        marker="o", label="임야 비율",
        color=color_map["임야 비율"]
    )
    plt.plot(
        years, agri_ratio,
        marker="o", label="농경지 비율",
        color=color_map["농경지 비율"]
    )
    plt.plot(
        years, dae_ratio,
        marker="o", label="대지 비율",
        color=color_map["대지 비율"]
    )
    plt.plot(
        years, factory_ratio,
        marker="o", label="공장용지 비율",
        color=color_map["공장용지 비율"]
    )

    # 도로율 (같은 축에 비율로 표시)
    plt.plot(
        years, road_ratio,
        marker="s", linestyle="--",
        label="도로율",
        color=ROAD_COLOR
    )

    plt.xticks(years)
    plt.ylim(0, 1)  # 비율 + 도로율 모두 0~1 구간
    plt.grid(alpha=0.3)

    plt.xlabel("연도")
    plt.ylabel("비율 / 도로율")
    plt.title(f"{region} 토지 이용 비율 및 도로율 변화 (2015~{years.max()}년)")
    plt.legend(loc="best")

    plt.tight_layout()

    safe_region = region.replace(" ", "_")
    outfile = os.path.join(base_dir, f"ratio_road_timeseries_{safe_region}.png")
    plt.savefig(outfile, dpi=200)
    plt.close()

    print("저장:", outfile)

# ===== 5) (추가) 충청북도 전체 합계 비율 + 도로율 변화 =====
# 연도별로 면적 합산 → 비율 계산
# 도로율은 시군구 단순 평균으로 계산(원하면 가중평균으로 바꿀 수 있음)
total_group = df.groupby(col_year)[[col_forest, col_agri, col_dae, col_factory, "도로율"]].agg({
    col_forest: "sum",
    col_agri:   "sum",
    col_dae:    "sum",
    col_factory:"sum",
    "도로율":   "mean"   # 연도별 시군구 평균 도로율
}).reset_index()

total_group["denom"] = (
    total_group[col_forest] +
    total_group[col_agri] +
    total_group[col_dae] +
    total_group[col_factory]
)

total_group["임야 비율"]      = total_group[col_forest]  / total_group["denom"]
total_group["농경지 비율"]    = total_group[col_agri]    / total_group["denom"]
total_group["대지 비율"]      = total_group[col_dae]     / total_group["denom"]
total_group["공장용지 비율"]  = total_group[col_factory] / total_group["denom"]

years = total_group[col_year].values

plt.figure(figsize=(9, 5))
plt.plot(
    years, total_group["임야 비율"],
    marker="o", label="임야 비율",
    color=color_map["임야 비율"]
)
plt.plot(
    years, total_group["농경지 비율"],
    marker="o", label="농경지 비율",
    color=color_map["농경지 비율"]
)
plt.plot(
    years, total_group["대지 비율"],
    marker="o", label="대지 비율",
    color=color_map["대지 비율"]
)
plt.plot(
    years, total_group["공장용지 비율"],
    marker="o", label="공장용지 비율",
    color=color_map["공장용지 비율"]
)

# 전체 도로율
plt.plot(
    years, total_group["도로율"],
    marker="s", linestyle="--",
    label="도로율(평균)",
    color=ROAD_COLOR
)

plt.xticks(years)
plt.ylim(0, 1)
plt.grid(alpha=0.3)

plt.xlabel("연도")
plt.ylabel("비율 / 도로율")
plt.title(f"충청북도 전체 토지 이용 비율 및 도로율 변화 (2015~{years.max()}년)")
plt.legend(loc="best")
plt.tight_layout()

outfile_total = os.path.join(base_dir, "ratio_road_timeseries_충청북도_전체.png")
plt.savefig(outfile_total, dpi=200)
plt.close()
print("저장:", outfile_total)

print("\n=== 지역별 + 충북전체 토지 이용 비율 + 도로율 변화 그래프 생성 완료 ===")
