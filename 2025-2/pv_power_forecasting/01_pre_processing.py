# -*- coding: utf-8 -*-
"""
태양광 발전량 고장진단용 전처리 + 피처 생성 스크립트 (10분 리샘플링 포함)

필요 패키지:
    pip install pandas numpy pvlib
"""

import os
import numpy as np
import pandas as pd
import pvlib


# ==========================
# 설정 (사용자 수정 구간)
# ==========================

CSV_PATH = r"SolarData_merged.csv"
OUTPUT_CSV_PATH = r"SolarData_60min_for_train.csv"

# 발전소 위치 (예시: 대전 근처 → 필요시 수정)
LATITUDE = 36.969965    # 위도 (deg, 북위 +)
LONGITUDE = 127.871715  # 경도 (deg, 동경 +)
TIMEZONE = "Asia/Seoul"

# 리샘플링 시간 간격 (예: "10min"=10분, "5min", "15min", "30min" 등)
RESAMPLE_RULE = "60min"


# ==========================
# 0. Datetime 파싱 + 정렬 + 중복 제거 + 쓸모없는 컬럼 제거
# ==========================

def load_and_basic_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "Datetime" not in df.columns:
        raise ValueError("CSV에 'Datetime' 컬럼이 없습니다.")

    df["Datetime"] = pd.to_datetime(df["Datetime"])

    df = df.sort_values("Datetime").drop_duplicates(subset=["Datetime"])
    df = df.reset_index(drop=True)

    # 타임존 설정 (naive → Asia/Seoul 가정)
    df["Datetime"] = df["Datetime"].dt.tz_localize(
        TIMEZONE, nonexistent="shift_forward", ambiguous="NaT"
    )

    # Voltage / Current / Frequency는 이번 프로젝트에서 사용하지 않으므로 삭제
    df = df.drop(columns=["Voltage", "Current", "Frequency"], errors="ignore")

    return df


# ==========================
# (추가) Datetime 기준 리샘플링 (ex. 10분 평균)
# ==========================

def resample_time(df: pd.DataFrame, rule: str = RESAMPLE_RULE) -> pd.DataFrame:
    # Datetime을 인덱스로
    df = df.set_index("Datetime")

    # 숫자형 컬럼만 뽑아서 리샘플링 (문자열, 카테고리 등은 제외)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    df_resampled = df[numeric_cols].resample(rule).mean()

    # Datetime 인덱스를 컬럼으로 복구
    df_resampled = df_resampled.reset_index()

    return df_resampled


# ==========================
# 1. 핵심 컬럼 NaN 행 삭제
# 2. 물리 범위 기반 에러값 NaN 후 제거
# ==========================

def remove_nans_and_physical_errors(df: pd.DataFrame) -> pd.DataFrame:
    core_cols = [
        "DHI", "DNI",
        "Temperature", "Module_Temperature",
        "WS", "RH",
        "Target"
    ]
    missing_core = [c for c in core_cols if c not in df.columns]
    if missing_core:
        raise ValueError(f"다음 핵심 컬럼이 CSV에 없습니다: {missing_core}")

    # 1차: 핵심 컬럼 NaN 행 제거
    df = df.dropna(subset=core_cols).copy()

    # 2차: 물리 범위 벗어나면 NaN 처리
    df.loc[(df["DHI"] < 0), "DHI"] = np.nan
    df.loc[(df["DNI"] < 0), "DNI"] = np.nan
    df.loc[(df["Module_Temperature"] < -20) | (df["Module_Temperature"] > 90), "Module_Temperature"] = np.nan
    df.loc[(df["Temperature"] < -30) | (df["Temperature"] > 50), "Temperature"] = np.nan
    df.loc[(df["WS"] < 0) | (df["WS"] > 25), "WS"] = np.nan
    df.loc[(df["RH"] < 0) | (df["RH"] > 100), "RH"] = np.nan
    df.loc[df["Target"] < 0, "Target"] = np.nan

    # 2차 후 NaN 행 제거
    df = df.dropna(subset=core_cols).copy()

    return df


# ==========================
# 3. 일출/일몰 시간 계산 후 낮 시간만 남기기
# ==========================

def filter_daytime_by_solar_position(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Datetime").copy()
    df = df.set_index("Datetime")

    solpos = pvlib.solarposition.get_solarposition(
        time=df.index,
        latitude=LATITUDE,
        longitude=LONGITUDE
    )

    df["solar_zenith"] = solpos["apparent_zenith"]
    df["solar_altitude"] = 90.0 - df["solar_zenith"]

    sun_up = df["solar_zenith"] < 90.0
    df = df[sun_up].copy()

    df = df.reset_index()
    return df


# ==========================
# 4. 시간·계절·래그/롤링 + Hour+Minute → float, sin/cos
# ==========================

def add_time_and_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Datetime").copy()

    df["Year"] = df["Datetime"].dt.year
    df["Month"] = df["Datetime"].dt.month
    df["Day"] = df["Datetime"].dt.day
    df["Hour"] = df["Datetime"].dt.hour
    df["Minute"] = df["Datetime"].dt.minute
    df["doy"] = df["Datetime"].dt.dayofyear

    df["hour_float"] = df["Hour"] + df["Minute"] / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_float"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_float"] / 24.0)

    df = df.sort_values("Datetime")
    df.set_index("Datetime", inplace=True)

    for col in ["Target", "DHI", "DNI", "Module_Temperature"]:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag3"] = df[col].shift(3)
        df[f"{col}_roll_mean3"] = df[col].shift(1).rolling(3).mean()
        df[f"{col}_roll_std3"] = df[col].shift(1).rolling(3).std()

    df = df.reset_index()
    return df


# ==========================
# 5. 마지막 3일/5일 동시간 Target 평균
# ==========================

def add_same_time_past_days_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Datetime").copy()
    df = df.set_index("Datetime")

    # 샘플 간격(분) 추정
    diffs = df.index.to_series().diff().dropna().dt.total_seconds() / 60.0
    freq_min = diffs.mode()[0]
    steps_per_day = int(round(24 * 60 / freq_min))

    for d in range(1, 6):  # 1~5일
        df[f"Target_lag_{d}d"] = df["Target"].shift(d * steps_per_day)

    df["Target_mean_3d_same_time"] = df[["Target_lag_1d", "Target_lag_2d", "Target_lag_3d"]].mean(axis=1)
    df["Target_mean_5d_same_time"] = df[
        ["Target_lag_1d", "Target_lag_2d", "Target_lag_3d", "Target_lag_4d", "Target_lag_5d"]
    ].mean(axis=1)

    df = df.reset_index()
    return df


# ==========================
# 6. 기온 + RH → 이슬점
# ==========================

def add_dew_point(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    T = df["Temperature"].astype(float)
    RH = df["RH"].astype(float)

    RH_safe = RH.clip(lower=1.0, upper=100.0)

    a, b = 17.27, 237.7
    alpha = (a * T / (b + T)) + np.log(RH_safe / 100.0)
    dew_point = (b * alpha) / (a - alpha)

    df["DewPoint"] = dew_point
    return df


# ==========================
# 7. Zenith + DNI/DHI → GHI + 효율
# ==========================

def add_ghi_and_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Datetime").copy()
    df = df.set_index("Datetime")

    if "solar_zenith" not in df.columns:
        solpos = pvlib.solarposition.get_solarposition(
            time=df.index,
            latitude=LATITUDE,
            longitude=LONGITUDE
        )
        df["solar_zenith"] = solpos["apparent_zenith"]
        df["solar_altitude"] = 90.0 - df["solar_zenith"]

    zenith_rad = np.deg2rad(df["solar_zenith"])
    cos_z = np.cos(zenith_rad).clip(lower=0.0)

    df["GHI_est"] = df["DNI"] * cos_z + df["DHI"]
    df["GHI_est"] = df["GHI_est"].clip(lower=0.0)

    df["PV_efficiency"] = np.where(
        df["GHI_est"] > 0,
        df["Target"] / df["GHI_est"],
        np.nan
    )

    df = df.reset_index()
    return df


# ==========================
# 전체 파이프라인
# ==========================

def preprocess_and_make_features(csv_path: str) -> pd.DataFrame:
    df = load_and_basic_clean(csv_path)       # 0
    df = resample_time(df, RESAMPLE_RULE)     # 0.5 리샘플링
    df = remove_nans_and_physical_errors(df)  # 1,2
    df = filter_daytime_by_solar_position(df) # 3
    df = add_time_and_lag_features(df)        # 4
    df = add_same_time_past_days_target(df)   # 5
    df = add_dew_point(df)                    # 6
    df = add_ghi_and_efficiency(df)           # 7

    # 래그/롤링/과거일 평균으로 생긴 NaN 제거
    feature_cols_to_check = [
        c for c in df.columns
        if ("lag_" in c) or ("roll_" in c) or ("Target_mean_" in c)
    ]
    df = df.dropna(subset=feature_cols_to_check).reset_index(drop=True)

    return df


# ==========================
# 실행 예시
# ==========================

if __name__ == "__main__":
    df_feat = preprocess_and_make_features(CSV_PATH)

    print("전처리 및 피처 생성 완료!")
    print("생성된 데이터 shape:", df_feat.shape)
    print("시작/끝 시각:", df_feat["Datetime"].min(), "→", df_feat["Datetime"].max())
    print("컬럼 예시:", df_feat.columns[:25].tolist())

    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    df_feat.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    print("저장 완료:", OUTPUT_CSV_PATH)

