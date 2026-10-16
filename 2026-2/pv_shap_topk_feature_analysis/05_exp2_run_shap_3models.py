# -*- coding: utf-8 -*-
"""LightGBM, Random Forest, LSTM saved-model SHAP (NO RETRAINING).

Edit CONFIG below and press Run in the IDE. The three model families are the
only models used. PatchTST is not imported or loaded. The analysis reproduces
the original 2023 Validation daytime / common target-time protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import torch

from shap_common import (
    DATE_COL, TARGET_COL, POWER_COL, CAPACITY_COL, DAYTIME_COL,
    check_feature_sets, choose_common_timestamps, compare_model_importance,
    ensure_dir, inspect_torch_bundle, inspect_tree_bundle, read_feature_csv,
    resolve_device, save_json, sample_times, sequence_available_times,
    set_seed, split_by_year, tree_available_times,
)
from shap_lightgbm import run_lightgbm_shap
from shap_random_forest import run_random_forest_shap
from shap_lstm import run_lstm_shap

# ====================== IDE SETTINGS ======================
CONFIG = {
    "project_dir": None,   # None: auto-detect Solar folder; or r"C:\...\Solar"
    "csv": "chungbuk_pv_features.csv",
    "output_dir": "pv_shap_lgbm_rf_lstm_outputs",
    "lightgbm_model": "pv_training_outputs/models/lightgbm/C6_Full/model.pkl",
    "rf_model": "pv_training_outputs_rf/models/random_forest/C6_Full/model.pkl",
    "lstm_model": "pv_training_outputs/models/lstm/C6_Full/model.pt",
    "lstm_scaler": "pv_training_outputs/models/lstm/C6_Full/scaler.pkl",
    "shap_sample_size": 0,
    "background_size": 100,
    "gradient_nsamples": 200,
    "deep_chunk_size": 20,
    "top_n_plot": 20,
    "top_k_compare": 10,
    "expected_feature_count": 69,
    "seed": 42,
    "device": "auto",
    "quick_test": False,
    "check_only": False,  # True: inspect saved models/data without calculating SHAP
}
# ==========================================================

MODEL_ORDER = ["RandomForest", "LightGBM", "LSTM"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for key, value in CONFIG.items():
        if isinstance(value, bool):
            p.add_argument("--"+key, action="store_true", default=value)
        else:
            p.add_argument("--"+key, type=int if isinstance(value, int) else str,
                           default=value)
    return p.parse_args()


def project_root(value=None):
    if value:
        return Path(value).expanduser().resolve()
    root = Path(__file__).resolve().parent
    if root.name == "shap_tcn_4models":
        root = root.parent
    return root


def resolve_path(root, value):
    p = Path(value).expanduser()
    return (p if p.is_absolute() else root/p).resolve()


def assert_exists(root, value, label):
    p = resolve_path(root, value)
    if not p.is_file():
        raise FileNotFoundError(
            f"{label} 파일이 없습니다: {p}\n"
            "CONFIG의 경로를 확인하세요. 모델 학습이 완료되었는지도 확인하세요."
        )
    return p


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_bundle(bundle, name, expected_count):
    features = list(bundle["feature_cols"])
    if bundle.get("case_name") != "C6_Full":
        raise ValueError(f"{name}: C6_Full 체크포인트가 아닙니다.")
    if len(features) != expected_count:
        raise ValueError(f"{name}: feature_count={len(features)}, expected={expected_count}")
    if len(set(features)) != len(features):
        raise ValueError(f"{name}: 특징 이름이 중복되었습니다.")
    if bundle.get("target_col", TARGET_COL) != TARGET_COL:
        raise ValueError(f"{name}: Target이 다릅니다.")
    return features


def check_scaler(path, features, name):
    scaler = joblib.load(path)
    if not hasattr(scaler, "n_features_in_") or scaler.n_features_in_ != len(features):
        raise ValueError(f"{name}: scaler 입력 차원이 다릅니다.")
    if hasattr(scaler, "feature_names_in_") and list(scaler.feature_names_in_) != features:
        raise ValueError(f"{name}: scaler 특징 순서가 다릅니다.")
    if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all():
        raise ValueError(f"{name}: scaler에 비정상 값이 있습니다.")
    return scaler


def validate_data(df, features):
    required = list(features) + [TARGET_COL, POWER_COL, CAPACITY_COL, DAYTIME_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"데이터에 필요한 컬럼이 없습니다: {missing}")
    numeric = df[required].apply(pd.to_numeric, errors="coerce")
    invalid_capacity = numeric[CAPACITY_COL].notna() & (numeric[CAPACITY_COL] <= 0)
    if invalid_capacity.any():
        raise ValueError("0 이하 설비용량이 있습니다.")
    # The original SHAP study uses the full feature file, not a new preprocessing fit.
    return numeric


def save_times(times, path):
    pd.DataFrame({DATE_COL: pd.DatetimeIndex(times)}).to_csv(
        path, index=False, encoding="utf-8-sig"
    )


def main():
    args = parse_args()
    if args.quick_test:
        args.shap_sample_size = min(args.shap_sample_size, 20) if args.shap_sample_size > 0 else 20
        args.background_size = min(args.background_size, 20) if args.background_size > 0 else 20
        args.gradient_nsamples = min(args.gradient_nsamples, 30)
        args.deep_chunk_size = min(args.deep_chunk_size, 5)
        args.output_dir = str(args.output_dir) + "_SMOKE_TEST"
    if args.background_size < 1 or args.gradient_nsamples < 1 or args.deep_chunk_size < 1:
        raise ValueError("background_size, gradient_nsamples, deep_chunk_size는 1 이상이어야 합니다.")
    if args.shap_sample_size < 0 or args.top_k_compare < 1 or args.top_n_plot < 1:
        raise ValueError("SHAP 표본 및 Top-K 설정을 확인하세요.")
    set_seed(args.seed)
    device = resolve_device(args.device)
    root = project_root(args.project_dir)
    paths = {
        "LightGBM": assert_exists(root, args.lightgbm_model, "LightGBM model"),
        "RandomForest": assert_exists(root, args.rf_model, "Random Forest model"),
        "LSTM": assert_exists(root, args.lstm_model, "LSTM model"),
        "LSTM_scaler": assert_exists(root, args.lstm_scaler, "LSTM scaler"),
    }
    csv_path = assert_exists(root, args.csv, "feature CSV")
    # Never write model artifacts. Output must be separate from every input tree.
    output_dir = resolve_path(root, args.output_dir)
    for p in paths.values():
        if output_dir == p or output_dir in p.parents:
            raise ValueError("SHAP 출력 폴더가 모델 입력 경로의 상위 폴더입니다.")
    if output_dir == csv_path.parent or output_dir in csv_path.parents:
        raise ValueError("SHAP 출력 폴더가 데이터 입력 경로의 상위 폴더입니다.")

    lgb_bundle = inspect_tree_bundle(paths["LightGBM"], expected_type="lightgbm")
    rf_bundle = inspect_tree_bundle(paths["RandomForest"], expected_type="random_forest")
    lstm_bundle = inspect_torch_bundle(paths["LSTM"], expected_type="lstm")
    bundles = {"LightGBM":lgb_bundle, "RandomForest":rf_bundle,
               "LSTM":lstm_bundle}
    model_features = {n:validate_bundle(b,n,args.expected_feature_count)
                      for n,b in bundles.items()}
    check_feature_sets(model_features, args.expected_feature_count)
    check_scaler(paths["LSTM_scaler"], model_features["LSTM"], "LSTM")
    lstm_seq_len = int(lstm_bundle["best_params"]["seq_len"])
    if int(lstm_bundle["architecture"]["input_dim"]) != len(model_features["LSTM"]):
        raise ValueError("LSTM input_dim이 다릅니다.")

    df = read_feature_csv(csv_path)
    numeric = validate_data(df, model_features["LightGBM"])
    train_df, valid_df, _test_df = split_by_year(df)
    availability = {
        "LightGBM": tree_available_times(valid_df, model_features["LightGBM"]),
        "RandomForest": tree_available_times(valid_df, model_features["RandomForest"]),
        "LSTM": sequence_available_times(valid_df, model_features["LSTM"], lstm_seq_len),
    }
    common_all, selected_times = choose_common_timestamps(
        valid_df, availability, args.shap_sample_size, args.seed
    )
    # Use TRAIN target moments for the LSTM reference background.
    lstm_train_available = sequence_available_times(train_df, model_features["LSTM"], lstm_seq_len)
    common_bg_candidates = lstm_train_available
    common_background_times = sample_times(
        common_bg_candidates, args.background_size, args.seed + 500
    )
    # LSTM still uses its own saved scaler and selected lookback.
    common_dir = output_dir/"common"
    comparison_dir = output_dir/"comparison"
    ensure_dir(common_dir)
    ensure_dir(comparison_dir)
    save_times(common_all, common_dir/"common_validation_daytime_timestamps_all.csv")
    save_times(selected_times, common_dir/"common_shap_sample_timestamps.csv")
    save_times(common_background_times, common_dir/"common_deep_background_timestamps.csv")
    save_json({
        "protocol": "saved-model SHAP only; no retraining",
        "models": MODEL_ORDER, "csv": str(csv_path), "csv_sha256": sha256(csv_path),
        "model_paths": {k:str(v) for k,v in paths.items()},
        "model_sha256": {k:sha256(v) for k,v in paths.items()},
        "split": {"train":"2020-2022", "validation":"2023", "test":"2024 (not used)"},
        "shap_dataset": "intersection of three-model Validation target timestamps, daytime only",
        "common_candidate_count":len(common_all), "selected_shap_sample_count":len(selected_times),
        "background_size":len(common_background_times),
        "deep_background":"training target timestamps for LSTM; saved input window/scaler",
        "sequence_length":{"LSTM":lstm_seq_len},
        "seed":args.seed,"device":device,"gradient_nsamples":args.gradient_nsamples,
        "deep_chunk_size":args.deep_chunk_size,"expected_feature_count":args.expected_feature_count,
        "feature_cols_by_model":model_features,
        "explainers":{"LightGBM":"TreeExplainer","RandomForest":"TreeExplainer",
                      "LSTM":"GradientExplainer / Expected Gradients"},
        "deep_feature_aggregation":"I_j = mean_i sum_t |SHAP_i,t,j|",
        "cross_model_comparison":"normalized importance (%) and ranks; raw SHAP magnitudes are not compared directly",
        "output_unit":"unclipped normalized PV power",
        "python":sys.version,"torch":torch.__version__,"shap":shap.__version__,
        "numpy":np.__version__,"pandas":pd.__version__,"platform":platform.platform(),
        "quick_test":args.quick_test,"check_only":args.check_only,
    },common_dir/"shap_analysis_config.json")
    print("="*85, flush=True)
    print("[3-model SHAP / NO RETRAINING] LightGBM, RF, LSTM", flush=True)
    print("Common 2023 daytime candidates:",len(common_all),flush=True)
    print("Selected SHAP timestamps:",len(selected_times),flush=True)
    print("LSTM seq_len:",lstm_seq_len,flush=True)
    print("Device:",device,"Output:",output_dir,flush=True)
    if args.check_only:
        print("[CHECK ONLY] 모델 및 데이터 검증 완료. SHAP 계산은 실행하지 않았습니다.",flush=True)
        return

    importance_by_model = {}
    print("\n[1/3] Random Forest - TreeExplainer",flush=True)
    importance_by_model["RandomForest"] = run_random_forest_shap(
        paths["RandomForest"],valid_df,selected_times,output_dir/"random_forest",
        top_n=args.top_n_plot,
    )
    print("\n[2/3] LightGBM - TreeExplainer",flush=True)
    importance_by_model["LightGBM"] = run_lightgbm_shap(
        paths["LightGBM"],valid_df,selected_times,output_dir/"lightgbm",
        top_n=args.top_n_plot,
    )
    print("\n[3/3] LSTM - GradientExplainer",flush=True)
    importance_by_model["LSTM"] = run_lstm_shap(
        paths["LSTM"],paths["LSTM_scaler"],train_df,valid_df,selected_times,
        output_dir/"lstm",device=device,background_size=args.background_size,
        background_times=common_background_times,gradient_nsamples=args.gradient_nsamples,
        deep_chunk_size=args.deep_chunk_size,seed=args.seed,top_n=args.top_n_plot,
    )
    compare_paths = compare_model_importance(
        importance_by_model,comparison_dir,top_k=args.top_k_compare
    )
    print("\n[COMPLETE]",flush=True)
    print("Output:",output_dir,flush=True)
    print("Model importance: <output>/<model>/shap_feature_importance.csv",flush=True)
    print("Spearman:",compare_paths["spearman"],flush=True)
    print("Top-K:",compare_paths["topk"],flush=True)
    print("Combined:",compare_paths["wide"],flush=True)


if __name__ == "__main__":
    main()
