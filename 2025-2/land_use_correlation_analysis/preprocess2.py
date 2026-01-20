# -*- coding: utf-8 -*-
"""
충북 시군구별 도로율(도로면적/전체면적) 계산 후 엑셀로 저장

입력: chungbuk_data_*.csv (연도별 원시 토지이용 데이터)
출력: chungbuk_road_ratio_2015_2025.xlsx
    - sheet "detail": 연도·시군구별 도로율
    - sheet "total" : 연도별 충청북도 전체 도로율
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

if not file_list:
    raise FileNotFoundError("chungbuk_data_*.csv 파일을 찾을 수 없습니다.")

# === 2) 공통 컬럼 이름 ===
col_region = "토지소재명"
col_total  = "합계 면적(㎡)"

detail_records = []
total_records  = []

for path in file_list:
    fname = os.path.basename(path)

    # --- 연도 자동 추출 (파일명에서 숫자만 뽑아서 앞 4자리 사용) ---
    digits = "".join(filter(str.isdigit, fname))
    if len(digits) < 4:
        raise ValueError(f"파일명에서 연도를 추출할 수 없습니다: {fname}")
    year = int(digits[:4])
    print(f"\n===== 연도 {year} 처리 중... =====")

    # --- CSV 로드 ---
    df = pd.read_csv(path, encoding="utf-8-sig")

    # --- 도로 면적 컬럼 자동 탐색 ---
    # 1순위: '도로 면적(㎡)'가 있으면 사용
    if "도로 면적(㎡)" in df.columns:
        col_road = "도로 면적(㎡)"
    else:
        # 2순위: '도로'와 '면적' 둘 다 포함하는 컬럼
        candidates = [c for c in df.columns if ("도로" in c) and ("면적" in c)]
        if candidates:
            col_road = candidates[0]
        else:
            # 3순위(옵션): '도로'만 포함하는 컬럼
            candidates = [c for c in df.columns if "도로" in c]
            if candidates:
                col_road = candidates[0]
            else:
                raise ValueError(
                    f"{fname}에서 도로 면적 컬럼을 찾을 수 없습니다. "
                    f"컬럼명 목록: {df.columns.tolist()}"
                )

    print("도로 면적 컬럼 사용:", col_road)

    # --- 필요한 컬럼만 뽑아서 복사 ---
    use_cols = [col_region, col_total, col_road]
    df = df[use_cols].copy()

    # --- 숫자형 컬럼 처리 (콤마 제거 + float 변환) ---
    for c in [col_total, col_road]:
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace("", np.nan)
            .astype(float)
        )

    # --- 도로율 계산: 도로 면적 / 전체 면적 ---
    df["도로 면적(㎡)"] = df[col_road]  # 표준화된 이름으로 하나 더
    df["도로율"] = df["도로 면적(㎡)"] / df[col_total]

    # --- 연도 정보 추가 ---
    df["year"] = year

    # ===== (1) detail 레코드: 연도·시군구별 도로율 =====
    detail_records.append(
        df[["year", col_region, "합계 면적(㎡)", "도로 면적(㎡)", "도로율"]]
        .rename(columns={col_total: "합계 면적(㎡)"})
    )

    # ===== (2) total 레코드: 연도별 충청북도 전체 도로율 =====
    total_area = df[col_total].sum()
    total_road = df["도로 면적(㎡)"].sum()

    total_row = {
        "year": year,
        "토지소재명": "충청북도 합계",
        "합계 면적(㎡)": total_area,
        "도로 면적(㎡)": total_road,
        "도로율": total_road / total_area if total_area > 0 else np.nan,
    }
    total_records.append(total_row)

# === 3) 통합 DataFrame 생성 ===
detail_all = pd.DataFrame(detail_records[0]).iloc[0:0].copy()  # 구조만 복사
detail_all = pd.concat(detail_records, ignore_index=True)
total_all  = pd.DataFrame(total_records)

# === 4) 엑셀로 저장 ===
out_xlsx_path = os.path.join(base_dir, "chungbuk_road_ratio_2015_2025.xlsx")

with pd.ExcelWriter(out_xlsx_path, engine="openpyxl") as writer:
    detail_all.to_excel(writer, sheet_name="detail", index=False)
    total_all.to_excel(writer, sheet_name="total", index=False)

print("\n저장 완료:", out_xlsx_path)
print(" - sheet 'detail': 연도·시군구별 도로율")
print(" - sheet 'total' : 연도별 충청북도 합계 도로율")

