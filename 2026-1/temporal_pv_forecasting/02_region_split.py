# -*- coding: utf-8 -*-
import os
import re
import pandas as pd

# ===== 입력/출력 경로 설정 =====
IN_PATH  = r"solar_preprocessed.csv"
OUT_DIR  = r"한국동서발전_지역별"   # 저장 폴더

REGION_COL = "시도명"  # 지역 구분 컬럼
ENC_OUT = "utf-8-sig"  # 엑셀 호환

os.makedirs(OUT_DIR, exist_ok=True)

def safe_filename(name: str) -> str:
    """파일명에 쓰기 어려운 문자 제거/치환"""
    name = str(name).strip()
    # Windows 금지 문자: \ / : * ? " < > |
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    # 공백/연속 언더스코어 정리
    name = re.sub(r"\s+", "", name)
    name = re.sub(r"_+", "_", name)
    return name

# ===== 로드 =====
df = pd.read_csv(IN_PATH, encoding="utf-8")

if REGION_COL not in df.columns:
    raise ValueError(f"지역 컬럼 '{REGION_COL}' 이(가) 없습니다. 실제 컬럼: {list(df.columns)}")

# ===== 저장 =====
regions = sorted(df[REGION_COL].dropna().unique())
print(f"[INFO] regions: {len(regions)}")

for r in regions:
    r_safe = safe_filename(r)
    out_path = os.path.join(OUT_DIR, f"PV_Dataset_{r_safe}.csv")

    sub = df[df[REGION_COL] == r].copy()
    sub.to_csv(out_path, index=False, encoding=ENC_OUT)

    print(f"[SAVED] {r} -> {out_path} (rows={len(sub)})")

print("[DONE]")
