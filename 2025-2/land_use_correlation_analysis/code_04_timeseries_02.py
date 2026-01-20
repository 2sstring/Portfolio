# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 23:56:03 2025

@author: bigbell
"""

# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt

# ===== 한글 폰트 설정 =====
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 공통 색상 지정 (그대로 사용)
color_map = {
    "임야": "#75CD97",     # 초록
    "농경지": "#E6C950",   # 노랑
    "대지": "#EA9358",     # 주황
    "공장용지": "#8FA1BB", # 회색
}

# ===== 1) 데이터 로드 =====
base_dir = r"C:/Users/leebi/OneDrive/바탕 화면/team_project"
detail_csv_path = os.path.join(base_dir, "chungbuk_landuse_composition_2015_2025_detail.csv")

df = pd.read_csv(detail_csv_path, encoding="cp949")

# '충청북도 합계'는 일단 빼고, 시·군·구 데이터만 사용
df = df[df["토지소재명"] != "충청북도 합계"].copy().reset_index(drop=True)

# 컬럼 이름
col_region  = "토지소재명"
col_year    = "year"
col_forest  = "임야 면적(㎡)"
col_agri    = "농경지 면적(㎡)"
col_dae     = "대 면적(㎡)"
col_factory = "공장용지 면적(㎡)"

df[col_year] = df[col_year].astype(int)

print("데이터 미리보기 (면적):")
print(df[[col_region, col_year, col_forest, col_agri, col_dae, col_factory]].head())

# ===== 2) 지역 목록 =====
regions = sorted(df[col_region].unique())
print("지역 개수:", len(regions))
print("지역 목록:", regions)

# ===== 3) 지역별 2015~2025 '면적' 변화 그래프 =====
for region in regions:
    sub = df[df[col_region] == region].copy()
    sub = sub.sort_values(col_year)

    years = sub[col_year].values
    forest_area   = sub[col_forest].values
    agri_area     = sub[col_agri].values
    dae_area      = sub[col_dae].values
    factory_area  = sub[col_factory].values

    plt.figure(figsize=(9, 5))

    plt.plot(years, forest_area,  marker="o", label="임야 면적",
             color=color_map["임야"])
    plt.plot(years, agri_area,    marker="o", label="농경지 면적",
             color=color_map["농경지"])
    plt.plot(years, dae_area,     marker="o", label="대지 면적",
             color=color_map["대지"])
    plt.plot(years, factory_area, marker="o", label="공장용지 면적",
             color=color_map["공장용지"])

    plt.xticks(years)
    # plt.ylim(0, 1)  # 비율 아니므로 삭제
    plt.grid(alpha=0.3)

    plt.xlabel("연도")
    plt.ylabel("면적(㎡)")
    plt.title(f"{region} 토지 이용 면적 변화 (2015~{years.max()}년)")
    plt.legend(
        loc="lower left",
        bbox_to_anchor=(1.02, 0.5)
    )

    plt.tight_layout()

    safe_region = region.replace(" ", "_")
    outfile = os.path.join(base_dir, f"area_timeseries_{safe_region}.png")
    plt.savefig(outfile, dpi=200)
    plt.close()

    print("저장:", outfile)


# ===== 4) (추가) 충청북도 전체 합계 '면적' 변화 =====
# 연도별로 면적 합산 → 그대로 시계열
total_group = df.groupby(col_year)[[col_forest, col_agri, col_dae, col_factory]].sum().reset_index()

years = total_group[col_year].values

plt.figure(figsize=(9, 5))
plt.plot(years, total_group[col_forest],   marker="o", label="임야 면적",
         color=color_map["임야"])
plt.plot(years, total_group[col_agri],     marker="o", label="농경지 면적",
         color=color_map["농경지"])
plt.plot(years, total_group[col_dae],      marker="o", label="대지 면적",
         color=color_map["대지"])
plt.plot(years, total_group[col_factory],  marker="o", label="공장용지 면적",
         color=color_map["공장용지"])

plt.xticks(years)
plt.grid(alpha=0.3)

plt.xlabel("연도")
plt.ylabel("면적(㎡)")
plt.title(f"충청북도 전체 토지 이용 면적 변화 (2015~{years.max()}년)")
plt.legend(
    loc="lower left",
    bbox_to_anchor=(1.02, 0.5)
)
plt.tight_layout()


outfile_total = os.path.join(base_dir, "area_timeseries_충청북도_전체.png")
plt.savefig(outfile_total, dpi=200)
plt.close()
print("저장:", outfile_total)

print("\n=== 지역별 + 충북 전체 토지 이용 '면적' 변화 그래프 생성 완료 ===")
