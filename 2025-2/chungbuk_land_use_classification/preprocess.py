# -*- coding: utf-8 -*-
"""
충청북도 토지이용 구성 분석 (2015~2025)
 - 연도별 / 지역별:
    임야, 농경지(전+답), 대지, 공장용지 면적 + 비율 계산
 - 연도·지역을 모두 합친 통합 CSV 저장
"""

import os
import glob
import numpy as np
import pandas as pd

# === 0) 데이터 폴더 설정 ===
base_dir = r"data"
os.chdir(base_dir)

# === 1) 2015~2025 CSV 파일 리스트 ===
file_list = sorted(glob.glob("chungbuk_data_*.csv"))
print("찾은 파일들:", file_list)

# === 2) 컬럼 이름 정의 ===
col_region  = "토지소재명"
col_total   = "합계 면적(㎡)"
col_forest  = "임야 면적(㎡)"
col_jeon    = "전 면적(㎡)"
col_dap     = "답 면적(㎡)"
col_dae     = "대 면적(㎡)"           # 대지
col_factory = "공장용지 면적(㎡)"

records = []
records_total = []

for path in file_list:
    fname = os.path.basename(path)

    # 🔥 연도 자동 추출 (YYYY 또는 YYYYMMDD 모두 대응)
    digits = "".join(filter(str.isdigit, fname))     # 숫자만 추출
    year = int(digits[:4])
    print(f"연도 {year} 처리 중...")

    df = pd.read_csv(path, encoding="utf-8-sig")

    cols = [col_region, col_total, col_forest, col_jeon, col_dap, col_dae, col_factory]
    df = df[cols].copy()

    num_cols = cols[1:]
    for c in num_cols:
        df[c] = df[c].astype(str).str.replace(",", "").astype(float)

    df["농경지 면적(㎡)"] = df[col_jeon] + df[col_dap]

    # 비율 계산 (임야+농경지+대지+공장용지만을 모수로 하는 구성비)
    
    col_total = "합계 면적(㎡)"
    denom = df[col_total]
    """denom = (
        df[col_forest] +
        df["농경지 면적(㎡)"] +
        df[col_dae] +
        df[col_factory]
    )"""

    df["임야 비율"]      = df[col_forest]       / denom
    df["농경지 비율"]    = df["농경지 면적(㎡)"] / denom
    df["대지 비율"]      = df[col_dae]          / denom
    df["공장용지 비율"]  = df[col_factory]      / denom

    df["year"] = year

    records.append(df[[ 
        "year",
        col_region,
        col_total,
        col_forest,
        "농경지 면적(㎡)",
        col_dae,
        col_factory,
        "임야 비율",
        "농경지 비율",
        "대지 비율",
        "공장용지 비율",
    ]])

    total_area    = df[col_total].sum()
    total_forest  = df[col_forest].sum()
    total_agri    = df["농경지 면적(㎡)"].sum()
    total_dae     = df[col_dae].sum()
    total_factory = df[col_factory].sum()

    denom_total = df[col_total]  # 지역·연도별 전체 면적
    
    #denom_total = total_forest + total_agri + total_dae + total_factory

    total_row = {
        "year": year,
        "토지소재명": "충청북도 합계",
        "합계 면적(㎡)": total_area,
        "임야 면적(㎡)": total_forest,
        "농경지 면적(㎡)": total_agri,
        "대 면적(㎡)": total_dae,
        "공장용지 면적(㎡)": total_factory,
        "임야 비율":      total_forest  / denom_total,
        "농경지 비율":    total_agri    / denom_total,
        "대지 비율":      total_dae     / denom_total,
        "공장용지 비율":  total_factory / denom_total,
    }
    records_total.append(total_row)

detail_all = pd.concat(records, ignore_index=True)
total_all = pd.DataFrame(records_total)

# === 4) CSV로 저장 ===
detail_csv_path = os.path.join(base_dir, "chungbuk_landuse_composition_2015_2025_detail.csv")
total_csv_path  = os.path.join(base_dir, "chungbuk_landuse_composition_2015_2025_total.csv")

detail_all.to_csv(detail_csv_path, index=False, encoding="cp949")
total_all.to_csv(total_csv_path,  index=False, encoding="cp949")

print("\n저장 완료:")
print(" - 연도·지역별 상세:", detail_csv_path)
print(" - 연도별 충청북도 합계:", total_csv_path)

