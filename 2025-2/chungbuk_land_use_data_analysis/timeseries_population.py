# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt

# ===== 한글 폰트 설정 =====
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 공통 색상 (인구밀도용 한 색만 사용)
POP_COLOR = "#EA9358"   # 주황 계열

# ===== 1) 데이터 로드 =====
base_dir = r"data"
detail_csv_path = os.path.join(base_dir, "chungbuk_landuse_composition_2015_2025_detail.csv")
pop_xlsx_path   = os.path.join(base_dir, "chungbuk_population.xlsx")

# 토지 이용(면적) 데이터
df = pd.read_csv(detail_csv_path, encoding="cp949")

# '충청북도 합계'는 일단 빼고, 시·군·구 데이터만 사용
df = df[df["토지소재명"] != "충청북도 합계"].copy().reset_index(drop=True)

# 컬럼 이름
col_region  = "토지소재명"
col_year    = "year"
col_total   = "합계 면적(㎡)"

df[col_year] = df[col_year].astype(int)

print("토지 데이터 미리보기:")
print(df[[col_region, col_year, col_total]].head())

# ===== 2) 인구 데이터 로드 및 Long 형태 변환 =====
pop_raw = pd.read_excel(pop_xlsx_path)

# pop_raw 예시:
#   연도      2015    2016  ...  2025
#  청주 상당구   ...
# melt → (토지소재명, year, 인구)
pop_long = pop_raw.melt(
    id_vars=["연도"],
    var_name="year",
    value_name="인구"
)

pop_long["year"] = pop_long["year"].astype(int)
pop_long = pop_long.rename(columns={"연도": "토지소재명"})

print("\n인구 데이터 미리보기:")
print(pop_long.head())

# ===== 3) 토지 데이터 + 인구 데이터 병합 =====
df = df.merge(pop_long, on=["토지소재명", "year"], how="left")

missing_pop = df["인구"].isna().sum()
if missing_pop > 0:
    print(f"\n⚠ 경고: 인구 정보가 없는 행 개수 = {missing_pop}")

# ===== 4) 인구밀도 계산 (명 / km²) =====
# 합계 면적(㎡) → km² 로 변환 후 인구 / 면적
df["pop_density"] = df["인구"] / (df[col_total] / 1_000_000.0)

print("\n인구 + 인구밀도 미리보기:")
print(df[[col_region, col_year, "인구", "pop_density"]].head())

# ===== 5) 지역 목록 =====
regions = sorted(df[col_region].unique())
print("\n지역 개수:", len(regions))
print("지역 목록:", regions)

# ===== 6) 지역별 2015~2025 인구밀도 변화 그래프 =====
for region in regions:
    sub = df[df[col_region] == region].copy()
    sub = sub.sort_values(col_year)

    years = sub[col_year].values
    density = sub["pop_density"].values

    plt.figure(figsize=(9, 5))

    plt.plot(
        years,
        density,
        marker="o",
        label="인구밀도 (명/km²)",
        color=POP_COLOR
    )

    plt.xticks(years)
    plt.grid(alpha=0.3)

    plt.xlabel("연도")
    plt.ylabel("인구밀도 (명/km²)")
    plt.title(f"{region} 인구밀도 변화 (2015~{years.max()}년)")
    plt.legend(loc="best")

    plt.tight_layout()

    safe_region = region.replace(" ", "_")
    outfile = os.path.join(base_dir, f"pop_density_timeseries_{safe_region}.png")
    plt.savefig(outfile, dpi=200)
    plt.close()

    print("저장:", outfile)

# ===== 7) (추가) 충청북도 전체 인구밀도 변화 =====
# 연도별로 인구 합 + 면적 합 → 인구밀도 계산
total_group = df.groupby(col_year)[[col_total, "인구"]].sum().reset_index()

total_group["pop_density"] = (
    total_group["인구"] / (total_group[col_total] / 1_000_000.0)
)

years = total_group[col_year].values
density_total = total_group["pop_density"].values

plt.figure(figsize=(9, 5))
plt.plot(
    years,
    density_total,
    marker="o",
    label="인구밀도 (명/km²)",
    color=POP_COLOR
)

plt.xticks(years)
plt.grid(alpha=0.3)

plt.xlabel("연도")
plt.ylabel("인구밀도 (명/km²)")
plt.title(f"충청북도 전체 인구밀도 변화 (2015~{years.max()}년)")
plt.legend(loc="best")
plt.tight_layout()

outfile_total = os.path.join(base_dir, "pop_density_timeseries_충청북도_전체.png")
plt.savefig(outfile_total, dpi=200)
plt.close()
print("저장:", outfile_total)

print("\n=== 지역별 + 충북 전체 인구밀도 변화 그래프 생성 완료 ===")

