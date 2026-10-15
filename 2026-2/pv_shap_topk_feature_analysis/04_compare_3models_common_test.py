# -*- coding: utf-8 -*-
"""
Final fair Test-set comparison for the PV forecasting study.

Purpose
-------
Recompute the final performance of three already-trained models on EXACTLY the
same Test timestamps. No model retraining and no Optuna rerun are performed.

Models / default result folders
-------------------------------
- LightGBM     : pv_training_outputs
- LSTM         : pv_training_outputs_lstm_rerun
- RandomForest : pv_training_outputs_rf

Default comparison case
-----------------------
C6_Full

Method
------
1. Load each model's saved Test prediction CSV.
2. Compute the timestamp intersection across all three models.
3. Verify that the ground truth is identical at the common timestamps.
4. Recalculate MAE, RMSE, NMAE, NRMSE, MBE, and R2 on the common timestamps.
5. Save both ALL-time and daytime-only results.

This is intentionally a post-processing script. It does NOT modify or retrain
any saved model.
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


# -----------------------------------------------------------------------------
# Column names: kept identical to the training scripts
# -----------------------------------------------------------------------------
DATE_COL = "발전일자"
CAPACITY_COL = "설비용량(MW)"
DAYTIME_COL = "대기권밖일사량계산값"
TRUE_COL = "true_MWh"
PRED_COL = "pred_MWh"

MODEL_ORDER = ["RandomForest", "LightGBM", "LSTM"]

MODEL_SLUG = {
    "RandomForest": "randomforest",
    "LightGBM": "lightgbm",
    "LSTM": "lstm",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Recalculate three-model PV Test metrics on common timestamps only."
    )
    p.add_argument(
        "--case",
        type=str,
        default="C6_Full",
        help="Feature case to compare (default: C6_Full)",
    )
    p.add_argument(
        "--base_output_dir",
        type=str,
        default="pv_training_outputs",
        help="Folder containing saved LightGBM/LSTM predictions",
    )
    p.add_argument(
        "--lstm_output_dir",
        type=str,
        default="pv_training_outputs_lstm_rerun",
        help="Folder containing rerun LSTM predictions",
    )
    p.add_argument(
        "--rf_output_dir",
        type=str,
        default="pv_training_outputs_rf",
        help="Folder containing saved Random Forest predictions",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="pv_training_outputs_final_comparison",
        help="Output folder for final common-Test comparison",
    )
    return p.parse_args()


def prediction_path(model_name: str, case_name: str, args) -> Path:
    if model_name == "LightGBM":
        root = Path(args.base_output_dir)
    elif model_name == "LSTM":
        root = Path(args.lstm_output_dir)
    elif model_name == "RandomForest":
        root = Path(args.rf_output_dir)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    slug = MODEL_SLUG[model_name]
    return root / "predictions" / f"test_predictions_{slug}_{case_name}.csv"


def load_prediction(path: Path, model_name: str, case_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"[{model_name}] Test prediction file not found:\n  {path}"
        )

    df = pd.read_csv(path, encoding="utf-8-sig")

    required = [DATE_COL, CAPACITY_COL, DAYTIME_COL, TRUE_COL, PRED_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{model_name}] Missing required columns in {path.name}: {missing}"
        )

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    if df[DATE_COL].duplicated().any():
        n_dup = int(df[DATE_COL].duplicated().sum())
        raise ValueError(
            f"[{model_name}] {path.name} contains {n_dup} duplicated timestamps."
        )

    df["model"] = model_name
    df["case"] = case_name
    return df


def metrics_from_df(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "n_samples": 0,
            "MAE_MWh": np.nan,
            "RMSE_MWh": np.nan,
            "NMAE_pct": np.nan,
            "NRMSE_pct": np.nan,
            "MBE_MWh": np.nan,
            "R2": np.nan,
        }

    true_mwh = df[TRUE_COL].to_numpy(dtype=float)
    pred_mwh = df[PRED_COL].to_numpy(dtype=float)
    capacity_mw = df[CAPACITY_COL].to_numpy(dtype=float)

    error = pred_mwh - true_mwh
    abs_error = np.abs(error)
    safe_capacity = np.where(capacity_mw == 0, np.nan, capacity_mw)

    try:
        r2 = float(r2_score(true_mwh, pred_mwh))
    except Exception:
        r2 = np.nan

    return {
        "n_samples": int(len(df)),
        "MAE_MWh": float(np.mean(abs_error)),
        "RMSE_MWh": float(np.sqrt(np.mean(error ** 2))),
        "NMAE_pct": float(np.nanmean(abs_error / safe_capacity) * 100.0),
        "NRMSE_pct": float(
            np.sqrt(np.nanmean((error / safe_capacity) ** 2)) * 100.0
        ),
        "MBE_MWh": float(np.mean(error)),
        "R2": r2,
    }


def evaluate_all_and_daytime(df: pd.DataFrame) -> Dict[str, float]:
    all_metrics = metrics_from_df(df)
    day = df[df[DAYTIME_COL] > 0].copy()
    day_metrics = metrics_from_df(day)

    return {
        **{f"ALL_{k}": v for k, v in all_metrics.items()},
        **{f"DAY_{k}": v for k, v in day_metrics.items()},
    }


def verify_same_ground_truth(
    predictions: Dict[str, pd.DataFrame], common_times: pd.DatetimeIndex
) -> None:
    """Ensure all three models are evaluated against the same true values/capacity."""
    reference_name = MODEL_ORDER[0]
    ref = (
        predictions[reference_name]
        .set_index(DATE_COL)
        .loc[common_times, [TRUE_COL, CAPACITY_COL, DAYTIME_COL]]
        .sort_index()
    )

    for model_name in MODEL_ORDER[1:]:
        cur = (
            predictions[model_name]
            .set_index(DATE_COL)
            .loc[common_times, [TRUE_COL, CAPACITY_COL, DAYTIME_COL]]
            .sort_index()
        )

        if not np.allclose(
            ref[TRUE_COL].to_numpy(dtype=float),
            cur[TRUE_COL].to_numpy(dtype=float),
            equal_nan=True,
        ):
            raise ValueError(
                f"Ground-truth {TRUE_COL} differs between {reference_name} and {model_name}."
            )

        if not np.allclose(
            ref[CAPACITY_COL].to_numpy(dtype=float),
            cur[CAPACITY_COL].to_numpy(dtype=float),
            equal_nan=True,
        ):
            raise ValueError(
                f"{CAPACITY_COL} differs between {reference_name} and {model_name}."
            )

        if not np.allclose(
            ref[DAYTIME_COL].to_numpy(dtype=float),
            cur[DAYTIME_COL].to_numpy(dtype=float),
            equal_nan=True,
        ):
            raise ValueError(
                f"{DAYTIME_COL} differs between {reference_name} and {model_name}."
            )


def main():
    args = parse_args()
    case_name = args.case
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load already-saved Test predictions.
    predictions: Dict[str, pd.DataFrame] = {}
    source_rows = []

    print("=" * 88)
    print(f"[Common Test comparison] case = {case_name}")
    print("No training / No Optuna rerun")
    print("=" * 88)

    for model_name in MODEL_ORDER:
        path = prediction_path(model_name, case_name, args)
        df = load_prediction(path, model_name, case_name)
        predictions[model_name] = df

        source_rows.append(
            {
                "model": model_name,
                "case": case_name,
                "source_file": str(path),
                "original_test_timestamps": int(df[DATE_COL].nunique()),
                "original_daytime_samples": int((df[DAYTIME_COL] > 0).sum()),
                "first_timestamp": df[DATE_COL].min(),
                "last_timestamp": df[DATE_COL].max(),
            }
        )
        print(
            f"{model_name:12s}: {len(df):6d} Test timestamps | "
            f"{df[DATE_COL].min()} ~ {df[DATE_COL].max()}"
        )

    # 2) Timestamp intersection across all three models.
    time_sets = {
        model_name: set(pd.to_datetime(df[DATE_COL]))
        for model_name, df in predictions.items()
    }
    common_set = set.intersection(*(time_sets[m] for m in MODEL_ORDER))
    if not common_set:
        raise ValueError("There is no common Test timestamp across all three models.")

    common_times = pd.DatetimeIndex(sorted(common_set))

    print("-" * 88)
    print(f"Common Test timestamps : {len(common_times)}")

    # 3) Ground-truth consistency check.
    verify_same_ground_truth(predictions, common_times)

    # 4) Recalculate metrics on exactly the same timestamps.
    metric_rows = []
    common_pred_frames = []

    for model_name in MODEL_ORDER:
        df = predictions[model_name]
        common = (
            df[df[DATE_COL].isin(common_times)]
            .copy()
            .sort_values(DATE_COL)
            .reset_index(drop=True)
        )

        if len(common) != len(common_times):
            raise RuntimeError(
                f"[{model_name}] Expected {len(common_times)} common rows, got {len(common)}."
            )

        metrics = evaluate_all_and_daytime(common)
        metric_rows.append(
            {
                "model": model_name,
                "case": case_name,
                "common_test_timestamps": len(common_times),
                **metrics,
            }
        )
        common_pred_frames.append(common)

    metric_df = pd.DataFrame(metric_rows)
    all_common_predictions = pd.concat(common_pred_frames, ignore_index=True)
    source_df = pd.DataFrame(source_rows)
    source_df["common_test_timestamps"] = len(common_times)
    source_df["retained_pct"] = (
        source_df["common_test_timestamps"]
        / source_df["original_test_timestamps"]
        * 100.0
    )

    # 5) Save outputs.
    source_df.to_csv(
        out_dir / f"test_timestamp_counts_{case_name}_3models.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metric_df.to_csv(
        out_dir / f"common_test_metrics_{case_name}_3models.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_common_predictions.to_csv(
        out_dir / f"common_test_predictions_{case_name}_3models.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame({DATE_COL: common_times}).to_csv(
        out_dir / f"common_test_timestamps_{case_name}_3models.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Main paper-style daytime table.
    main_cols = [
        "model",
        "case",
        "DAY_n_samples",
        "DAY_MAE_MWh",
        "DAY_RMSE_MWh",
        "DAY_NMAE_pct",
        "DAY_NRMSE_pct",
        "DAY_MBE_MWh",
        "DAY_R2",
    ]
    main_df = metric_df[main_cols].copy()
    main_df.to_csv(
        out_dir / f"MAIN_{case_name}_common_test_daytime_3models.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Small diagnostic text file for reproducibility.
    with open(
        out_dir / f"comparison_info_{case_name}.txt", "w", encoding="utf-8"
    ) as f:
        f.write(f"case={case_name}\n")
        f.write(f"common_test_timestamps={len(common_times)}\n")
        f.write("comparison_rule=intersection of Test timestamps across all three models\n")
        f.write("training=not performed\n")
        f.write("optuna=not performed\n")

    print("-" * 88)
    print("[Final daytime comparison]")
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(main_df.to_string(index=False))

    print("-" * 88)
    print("[Saved]")
    print(out_dir / f"MAIN_{case_name}_common_test_daytime_3models.csv")
    print(out_dir / f"common_test_metrics_{case_name}_3models.csv")
    print(out_dir / f"common_test_predictions_{case_name}_3models.csv")
    print(out_dir / f"common_test_timestamps_{case_name}_3models.csv")
    print(out_dir / f"test_timestamp_counts_{case_name}_3models.csv")
    print("=" * 88)


if __name__ == "__main__":
    main()
