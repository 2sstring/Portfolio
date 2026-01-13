# -*- coding: utf-8 -*-
"""
태양광 발전량 예측: LGBM + SARIMAX 앙상블 전용 스크립트

- 입력: 전처리된 SolarData_features_xxmin.csv (Datetime, Target, 피처 포함)
- Train : 2016, 2017년
- Test  : 2018년

모델:
- LightGBM
- SARIMAX
- LGBM + SARIMAX 성능 기반 가중 앙상블

출력:
- 각 모델 성능 지표 CSV
- LGBM / SARIMAX / 앙상블 가중치 pickle 파일 (ROS2 실시간 예측용)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import json
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from statsmodels.tsa.statespace.sarimax import SARIMAX

# ======================
# 0. 설정
# ======================

DATA_PATH = r"C:\Uni_Project\SolarData_60min_for_train.csv"
OUT_CSV  = r"C:\Uni_Project\SolarData_60min_for_train_lgbm_sarimax.csv"

MODEL_DIR = r"C:\Uni_Project\models"
TOLERANCE = 0.08
TARGET_COL = "Target"
TIME_COL = "Datetime"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

core_feats = [
    "Target_lag1",
    "Target_roll_mean3",
    "Target_lag3",
    "DNI", "DHI", "GHI_est",
    "DNI_lag1", "DNI_lag3", "DNI_roll_mean3",
    "DHI_lag1", "DHI_lag3", "DHI_roll_mean3",
    "Module_Temperature",
    "Module_Temperature_lag1",
    "Module_Temperature_lag3",
    "Module_Temperature_roll_mean3",
    "solar_altitude", "solar_zenith",
    "hour_cos",
]
optional_feats = [
    "WS", "RH",
    "Module_Temperature_roll_std3",
    "DHI_roll_std3",
    "Hour",
]
ALL_TABULAR_FEATURES = core_feats + optional_feats

def valid_features(candidate_list, df_columns):
    return [c for c in candidate_list if c in df_columns]


# ======================
# 메트릭
# ======================

def calc_metrics(y_true, y_pred, tol=TOLERANCE, kpx_threshold=0.1):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    eps = 1e-8
    y_min = float(np.min(y_true))
    y_max = float(np.max(y_true))
    y_range = y_max - y_min
    nmae = np.nan if y_range < eps else mae / y_range

    abs_error = np.abs(y_true - y_pred)
    acc = (abs_error <= tol).mean()

    mask = y_true >= kpx_threshold
    if np.sum(mask) == 0:
        kpx_nmae = np.nan
        fer = np.nan
    else:
        kpx_nmae = 100.0 * np.mean(np.abs(y_true[mask] - y_pred[mask]))
        fer = 100.0 * np.sum(np.abs(y_true[mask] - y_pred[mask])) / (np.sum(y_true[mask]) + eps)

    return mae, rmse, r2, nmae, acc, kpx_nmae, fer, mse


# ======================
# 데이터 로드/분리
# ======================

def load_and_split_data(path):
    df = pd.read_csv(path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    if TARGET_COL not in df.columns:
        raise ValueError("Target 컬럼이 없습니다.")

    feature_cols = valid_features(ALL_TABULAR_FEATURES, df.columns)
    cols_needed = [TARGET_COL] + feature_cols
    df_model = df.dropna(subset=cols_needed).copy()

    years = df_model[TIME_COL].dt.year
    df_year = df_model[years.isin([2016, 2017, 2018])].copy()
    df_year = df_year.sort_values(TIME_COL).reset_index(drop=True)

    years = df_year[TIME_COL].dt.year
    train_df = df_year[years.isin([2016, 2017])].copy()
    test_df  = df_year[years == 2018].copy()

    print("[INFO] Train:", train_df[TIME_COL].min(), "→", train_df[TIME_COL].max())
    print("[INFO] Test :", test_df[TIME_COL].min(), "→", test_df[TIME_COL].max())
    print("[INFO] feature_cols:", feature_cols)

    return df_year, train_df, test_df, feature_cols


# ======================
# 단일 모델 학습
# ======================

def train_lgbm(X_train, y_train, X_test):
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred


def train_sarimax(train_df, test_df, target_col, time_col):
    train_series = train_df.set_index(time_col)[target_col].astype(float)
    test_series  = test_df.set_index(time_col)[target_col].astype(float)

    model = SARIMAX(
        train_series,
        order=(1, 0, 1),
        #seasonal_order=(2, 1, 2, 24),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res = model.fit(disp=False)

    n_steps = len(test_series)
    forecast = res.forecast(steps=n_steps)

    y_true = test_series.values
    y_pred = np.asarray(forecast)

    return res, y_true, y_pred


# ======================
# 앙상블
# ======================

def performance_based_weights(mae_dict):
    inv = {}
    for k, v in mae_dict.items():
        if v is None or np.isnan(v) or v <= 0:
            continue
        inv[k] = 1.0 / v
    if not inv:
        return {}
    total = sum(inv.values())
    return {k: v / total for k, v in inv.items()}


# ======================
# 메인
# ======================

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df_year, train_df, test_df, feature_cols = load_and_split_data(DATA_PATH)

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_test  = test_df[feature_cols]
    y_test  = test_df[TARGET_COL]

    results = []

    # 1) LGBM
    print("\n[STEP] LightGBM")
    lgbm_model, y_pred_lgbm = train_lgbm(X_train, y_train, X_test)
    mae_lgbm, rmse, r2, nmae_lgbm, acc, kpx_nmae_lgbm, fer_lgbm, mse_lgbm = calc_metrics(y_test, y_pred_lgbm)
    results.append({
        "model": "lgbm",
        "type": "single",
        "base_models": "lgbm",
        "MAE": mae_lgbm, "RMSE": rmse, "MSE": mse_lgbm, "R2": r2,
        "NMAE": nmae_lgbm, "KPX_NMAE": kpx_nmae_lgbm, "FER": fer_lgbm,
        "ACC_within_tol": acc,
        "tolerance": TOLERANCE,
    })

    # 2) SARIMAX
    print("\n[STEP] SARIMAX")
    sarimax_model, y_true_sarimax, y_pred_sarimax = train_sarimax(train_df, test_df, TARGET_COL, TIME_COL)
    mae_sarimax, rmse, r2, nmae_sarimax, acc, kpx_nmae_sarimax, fer_sarimax, mse_sarimax = calc_metrics(y_true_sarimax, y_pred_sarimax)
    results.append({
        "model": "sarimax",
        "type": "single",
        "base_models": "sarimax",
        "MAE": mae_sarimax, "RMSE": rmse, "MSE": mse_sarimax, "R2": r2,
        "NMAE": nmae_sarimax, "KPX_NMAE": kpx_nmae_sarimax, "FER": fer_sarimax,
        "ACC_within_tol": acc,
        "tolerance": TOLERANCE,
    })

    # 3) LGBM + SARIMAX 앙상블
    print("\n[STEP] LGBM + SARIMAX 앙상블")
    base_mae = {"lgbm": mae_lgbm, "sarimax": mae_sarimax}
    base_pred = {"lgbm": y_pred_lgbm, "sarimax": y_pred_sarimax}
    weights = performance_based_weights(base_mae)

    # 혹시라도 키가 빠지면 안전하게 필터링
    weights = {k: v for k, v in weights.items() if k in base_pred}

    if len(weights) >= 2:
        ens = 0
        for m, w in weights.items():
            ens += base_pred[m] * w
        mae, rmse, r2, nmae, acc, kpx_nmae, fer, mse = calc_metrics(y_test, ens)
        results.append({
            "model": "ens_lgbm_sarimax",
            "type": "ensemble",
            "base_models": "+".join(weights.keys()),
            "MAE": mae, "RMSE": rmse, "MSE": mse, "R2": r2,
            "NMAE": nmae, "KPX_NMAE": kpx_nmae, "FER": fer,
            "ACC_within_tol": acc,
            "tolerance": TOLERANCE,
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("\n=== 결과 (NMAE 기준) ===")
    print(res_df.sort_values("NMAE"))

    # ----- 모델 / 메타 정보 저장 (ROS2용) -----
    joblib.dump(lgbm_model,  f"{MODEL_DIR}/lgbm_model.pkl")
    # SARIMAX는 statsmodels results 객체를 저장
    sarimax_model.save(f"{MODEL_DIR}/sarimax_model.pkl")

    with open(f"{MODEL_DIR}/ensemble_weights.json", "w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)

    with open(f"{MODEL_DIR}/feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    print("\n[INFO] 모델/가중치/피처 목록 저장 완료:", MODEL_DIR)


if __name__ == "__main__":
    main()
