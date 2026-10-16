# -*- coding: utf-8 -*-
"""
Compare C6_Full (69 features) vs SHAP Best Top-K on EXACTLY the same Test timestamps.
NO RETRAINING.

Usage
-----
Place this script in the same project folder as the other experiment scripts and press Run.

Expected project folders (defaults):
  pv_training_outputs_final_comparison/
      common_test_timestamps_C6_Full_3models.csv
      common_test_predictions_C6_Full_3models.csv

  pv_exp3_1_modelwise_shap_topk/
      final_selected/
          random_forest/predictions/test_predictions_randomforest_FINAL_SHAP_*.csv
          lightgbm/predictions/test_predictions_lightgbm_FINAL_SHAP_*.csv
          lstm/predictions/test_predictions_lstm_FINAL_SHAP_*.csv

The script automatically uses only models whose SHAP Top-K final prediction file exists.
Therefore it can be run now with only RandomForest and LightGBM completed, and run again
later after LSTM finishes. Existing source files are never modified.
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


# =============================================================================
# 0. USER SETTINGS
# =============================================================================
PROJECT_DIR = None  # None -> folder containing this .py file

BASELINE_DIR = "pv_training_outputs_final_comparison"
BASELINE_TIMESTAMPS_FILE = "common_test_timestamps_C6_Full_3models.csv"
BASELINE_PREDICTIONS_FILE = "common_test_predictions_C6_Full_3models.csv"

TOPK_ROOT = "pv_exp3_1_modelwise_shap_topk"
OUTPUT_DIR = "pv_exp3_1_modelwise_shap_topk/results_same_test_as_C6_Full"

# Model order requested in the project.
MODEL_ORDER = ["RandomForest", "LightGBM", "LSTM"]

# If True, a completed model must contain every baseline Test timestamp.
# This prevents accidental comparison on a smaller/different Test set.
STRICT_SAME_TEST = True

DATE_COL = "발전일자"
CAPACITY_COL = "설비용량(MW)"
DAYTIME_COL = "대기권밖일사량계산값"
TRUE_MWH_COL = "true_MWh"
PRED_MWH_COL = "pred_MWh"
MODEL_COL = "model"


MODEL_INFO = {
    "RandomForest": {
        "key": "random_forest",
        "folder": "random_forest",
        "glob": "test_predictions_randomforest_FINAL_SHAP_RandomForest_Top*.csv",
    },
    "LightGBM": {
        "key": "lightgbm",
        "folder": "lightgbm",
        "glob": "test_predictions_lightgbm_FINAL_SHAP_LightGBM_Top*.csv",
    },
    "LSTM": {
        "key": "lstm",
        "folder": "lstm",
        "glob": "test_predictions_lstm_FINAL_SHAP_LSTM_Top*.csv",
    },
}


def root_dir() -> Path:
    if PROJECT_DIR:
        return Path(PROJECT_DIR).expanduser().resolve()
    return Path(__file__).resolve().parent


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def parse_topk_from_name(path: Path) -> int | None:
    stem = path.stem
    marker = "_Top"
    if marker not in stem:
        return None
    tail = stem.rsplit(marker, 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return None


def choose_topk_prediction_file(project: Path, model: str) -> Path | None:
    """Find the final selected SHAP Top-K prediction file for one model."""
    info = MODEL_INFO[model]
    pred_dir = project / TOPK_ROOT / "final_selected" / info["folder"] / "predictions"
    if not pred_dir.exists():
        return None

    matches = list(pred_dir.glob(info["glob"]))
    if not matches:
        return None

    # Normally there is one final-selected file. If more than one exists, prefer newest.
    matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    if DATE_COL not in df.columns:
        raise ValueError(f"'{DATE_COL}' column not found")
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    return out


def validate_prediction_columns(df: pd.DataFrame, label: str) -> None:
    required = [DATE_COL, CAPACITY_COL, DAYTIME_COL, TRUE_MWH_COL, PRED_MWH_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing columns: {missing}")


def calc_metrics(df: pd.DataFrame) -> dict:
    true = pd.to_numeric(df[TRUE_MWH_COL], errors="coerce").to_numpy(float)
    pred = pd.to_numeric(df[PRED_MWH_COL], errors="coerce").to_numpy(float)
    cap = pd.to_numeric(df[CAPACITY_COL], errors="coerce").to_numpy(float)
    daytime = pd.to_numeric(df[DAYTIME_COL], errors="coerce").to_numpy(float)

    err = pred - true
    safe_cap = np.where(cap == 0, np.nan, cap)

    def one(mask: np.ndarray) -> dict:
        mask = mask & np.isfinite(true) & np.isfinite(pred) & np.isfinite(safe_cap)
        n = int(mask.sum())
        if n == 0:
            return {
                "n_samples": 0,
                "MAE_MWh": np.nan,
                "RMSE_MWh": np.nan,
                "NMAE_pct": np.nan,
                "NRMSE_pct": np.nan,
                "MBE_MWh": np.nan,
                "R2": np.nan,
            }

        y = true[mask]
        p = pred[mask]
        e = p - y
        c = safe_cap[mask]

        ss_res = float(np.sum((y - p) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

        return {
            "n_samples": n,
            "MAE_MWh": float(np.mean(np.abs(e))),
            "RMSE_MWh": float(np.sqrt(np.mean(e ** 2))),
            "NMAE_pct": float(np.nanmean(np.abs(e) / c) * 100.0),
            "NRMSE_pct": float(np.sqrt(np.nanmean((e / c) ** 2)) * 100.0),
            "MBE_MWh": float(np.mean(e)),
            "R2": float(r2),
        }

    all_mask = np.ones(len(df), dtype=bool)
    day_mask = daytime > 0
    all_m = one(all_mask)
    day_m = one(day_mask)

    row = {}
    for k, v in all_m.items():
        row[f"ALL_{k}"] = v
    for k, v in day_m.items():
        row[f"DAY_{k}"] = v
    return row


def align_to_reference(
    pred_df: pd.DataFrame,
    reference_times: pd.DatetimeIndex,
    label: str,
) -> tuple[pd.DataFrame, int]:
    pred_df = normalize_dates(pred_df)
    validate_prediction_columns(pred_df, label)

    # One prediction per timestamp is required.
    duplicated = pred_df[DATE_COL].duplicated(keep=False)
    if duplicated.any():
        dup_times = pred_df.loc[duplicated, DATE_COL].astype(str).head(10).tolist()
        raise ValueError(f"{label}: duplicate timestamps found, examples={dup_times}")

    available = set(pred_df[DATE_COL])
    reference_set = set(reference_times)
    missing = sorted(reference_set - available)

    if missing and STRICT_SAME_TEST:
        raise ValueError(
            f"{label}: {len(missing)} baseline Test timestamps are missing.\n"
            f"First missing examples: {[str(x) for x in missing[:10]]}\n"
            "STRICT_SAME_TEST=True, so the script stops rather than silently changing the Test set."
        )

    use_times = reference_times if not missing else pd.DatetimeIndex(
        [t for t in reference_times if t in available]
    )

    aligned = (
        pred_df[pred_df[DATE_COL].isin(use_times)]
        .set_index(DATE_COL)
        .loc[use_times]
        .reset_index()
    )
    return aligned, len(missing)


def main() -> None:
    project = root_dir()
    baseline_dir = project / BASELINE_DIR
    topk_root = project / TOPK_ROOT
    output_dir = project / OUTPUT_DIR
    ensure_dir(output_dir)

    baseline_times_path = baseline_dir / BASELINE_TIMESTAMPS_FILE
    baseline_pred_path = baseline_dir / BASELINE_PREDICTIONS_FILE

    if not baseline_times_path.is_file():
        raise FileNotFoundError(f"Baseline timestamp file not found: {baseline_times_path}")
    if not baseline_pred_path.is_file():
        raise FileNotFoundError(f"Baseline prediction file not found: {baseline_pred_path}")

    # -------------------------------------------------------------------------
    # A. Fixed Test reference = the exact common Test timestamps used by C6_Full
    # -------------------------------------------------------------------------
    ref_df = read_csv(baseline_times_path)
    if DATE_COL not in ref_df.columns:
        raise ValueError(f"Baseline timestamp file does not contain '{DATE_COL}'")
    reference_times = pd.DatetimeIndex(pd.to_datetime(ref_df[DATE_COL])).sort_values()

    if reference_times.duplicated().any():
        raise ValueError("Baseline timestamp file contains duplicate timestamps.")

    baseline_all = normalize_dates(read_csv(baseline_pred_path))
    if MODEL_COL not in baseline_all.columns:
        raise ValueError(f"Baseline prediction file does not contain '{MODEL_COL}'")

    print("=" * 100)
    print("[NO RETRAINING] C6_Full 69 features vs SHAP Best Top-K")
    print("Project              :", project)
    print("Fixed Test reference :", baseline_times_path)
    print("Reference timestamps :", len(reference_times))
    print("Period               :", reference_times.min(), "~", reference_times.max())
    print("Top-K root           :", topk_root)
    print("Output               :", output_dir)
    print("STRICT_SAME_TEST     :", STRICT_SAME_TEST)
    print("=" * 100)

    metric_rows = []
    comparison_rows = []
    aligned_long = []
    status_rows = []

    completed_models = []

    # -------------------------------------------------------------------------
    # B. Process only models whose Top-K final prediction already exists
    # -------------------------------------------------------------------------
    for model in MODEL_ORDER:
        topk_file = choose_topk_prediction_file(project, model)
        if topk_file is None:
            print(f"[SKIP] {model:12s}: final SHAP Top-K prediction not found yet")
            status_rows.append({
                "model": model,
                "status": "not_finished_or_prediction_missing",
                "topk_prediction_file": "",
            })
            continue

        completed_models.append(model)
        topk = parse_topk_from_name(topk_file)
        print(f"[FOUND] {model:12s}: {topk_file.name}")

        # C6_Full baseline prediction for this same model
        base_model = baseline_all[baseline_all[MODEL_COL] == model].copy()
        if base_model.empty:
            raise ValueError(f"Baseline C6_Full predictions do not contain model={model}")

        base_aligned, base_missing = align_to_reference(
            base_model, reference_times, f"C6_Full/{model}"
        )

        # SHAP Best Top-K final prediction
        topk_df = read_csv(topk_file)
        topk_aligned, topk_missing = align_to_reference(
            topk_df, reference_times, f"SHAP_TopK/{model}"
        )

        # Verify target values are identical timestamp by timestamp.
        check = base_aligned[[DATE_COL, TRUE_MWH_COL]].merge(
            topk_aligned[[DATE_COL, TRUE_MWH_COL]],
            on=DATE_COL,
            how="inner",
            suffixes=("_full69", "_topk"),
        )
        diff = np.abs(
            pd.to_numeric(check[f"{TRUE_MWH_COL}_full69"], errors="coerce") -
            pd.to_numeric(check[f"{TRUE_MWH_COL}_topk"], errors="coerce")
        )
        if np.nanmax(diff.to_numpy(float)) > 1e-9:
            raise ValueError(f"{model}: true_MWh differs between Full69 and Top-K on the same timestamps.")

        # Metrics on EXACTLY the baseline Test timestamps
        base_metrics = calc_metrics(base_aligned)
        topk_metrics = calc_metrics(topk_aligned)

        base_row = {
            "model": model,
            "feature_version": "C6_Full_69",
            "feature_count": 69,
            "selected_top_k": np.nan,
            "reference_test_timestamps": len(reference_times),
            "missing_reference_timestamps": base_missing,
            "prediction_file": str(baseline_pred_path),
            **base_metrics,
        }
        topk_row = {
            "model": model,
            "feature_version": "SHAP_Best_TopK",
            "feature_count": topk,
            "selected_top_k": topk,
            "reference_test_timestamps": len(reference_times),
            "missing_reference_timestamps": topk_missing,
            "prediction_file": str(topk_file),
            **topk_metrics,
        }
        metric_rows.extend([base_row, topk_row])

        comparison_rows.append({
            "model": model,
            "Full69_DAY_NMAE_pct": base_metrics["DAY_NMAE_pct"],
            "TopK": topk,
            "TopK_DAY_NMAE_pct": topk_metrics["DAY_NMAE_pct"],
            "Delta_DAY_NMAE_pct_TopK_minus_Full69": topk_metrics["DAY_NMAE_pct"] - base_metrics["DAY_NMAE_pct"],
            "Improvement_DAY_NMAE_pct_Full69_minus_TopK": base_metrics["DAY_NMAE_pct"] - topk_metrics["DAY_NMAE_pct"],
            "Full69_DAY_RMSE_MWh": base_metrics["DAY_RMSE_MWh"],
            "TopK_DAY_RMSE_MWh": topk_metrics["DAY_RMSE_MWh"],
            "Full69_DAY_R2": base_metrics["DAY_R2"],
            "TopK_DAY_R2": topk_metrics["DAY_R2"],
            "same_test_timestamp_count": len(reference_times) - topk_missing,
        })

        # Save model-level aligned predictions side-by-side.
        side = base_aligned[[
            DATE_COL, CAPACITY_COL, DAYTIME_COL, TRUE_MWH_COL, PRED_MWH_COL
        ]].copy()
        side = side.rename(columns={PRED_MWH_COL: "pred_MWh_C6_Full_69"})
        side["pred_MWh_SHAP_TopK"] = topk_aligned[PRED_MWH_COL].to_numpy()
        side["model"] = model
        side["selected_top_k"] = topk
        side["error_MWh_C6_Full_69"] = side["pred_MWh_C6_Full_69"] - side[TRUE_MWH_COL]
        side["error_MWh_SHAP_TopK"] = side["pred_MWh_SHAP_TopK"] - side[TRUE_MWH_COL]
        side.to_csv(
            output_dir / f"same_test_predictions_{model}_Full69_vs_TopK.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # Long format for later plotting/analysis.
        b_long = base_aligned.copy()
        b_long["model"] = model
        b_long["feature_version"] = "C6_Full_69"
        b_long["selected_top_k"] = np.nan
        t_long = topk_aligned.copy()
        t_long["model"] = model
        t_long["feature_version"] = "SHAP_Best_TopK"
        t_long["selected_top_k"] = topk
        aligned_long.extend([b_long, t_long])

        status_rows.append({
            "model": model,
            "status": "completed_and_aligned",
            "topk_prediction_file": str(topk_file),
            "selected_top_k": topk,
            "baseline_reference_timestamps": len(reference_times),
            "missing_reference_timestamps": topk_missing,
        })

    # -------------------------------------------------------------------------
    # C. Save results available at this moment
    # -------------------------------------------------------------------------
    status_df = pd.DataFrame(status_rows)
    status_df.to_csv(output_dir / "model_availability_status.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({DATE_COL: reference_times}).to_csv(
        output_dir / "FIXED_reference_test_timestamps_from_C6_Full.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not metric_rows:
        print("\nNo completed SHAP Top-K model predictions were found yet.")
        print("Status file saved:", output_dir / "model_availability_status.csv")
        return

    metrics_df = pd.DataFrame(metric_rows)
    model_cat = pd.Categorical(metrics_df["model"], categories=MODEL_ORDER, ordered=True)
    metrics_df = metrics_df.assign(_model_order=model_cat)
    version_order = pd.Categorical(
        metrics_df["feature_version"],
        categories=["C6_Full_69", "SHAP_Best_TopK"],
        ordered=True,
    )
    metrics_df = metrics_df.assign(_version_order=version_order)
    metrics_df = metrics_df.sort_values(["_model_order", "_version_order"]).drop(
        columns=["_model_order", "_version_order"]
    )
    metrics_df.to_csv(
        output_dir / "METRICS_same_test_Full69_vs_SHAP_TopK_available_models.csv",
        index=False,
        encoding="utf-8-sig",
    )

    comp_df = pd.DataFrame(comparison_rows)
    comp_df["model"] = pd.Categorical(comp_df["model"], categories=MODEL_ORDER, ordered=True)
    comp_df = comp_df.sort_values("model").reset_index(drop=True)
    comp_df.to_csv(
        output_dir / "MAIN_DAY_NMAE_Full69_vs_SHAP_TopK_same_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if aligned_long:
        long_df = pd.concat(aligned_long, ignore_index=True)
        long_df["model"] = pd.Categorical(long_df["model"], categories=MODEL_ORDER, ordered=True)
        long_df = long_df.sort_values(["model", "feature_version", DATE_COL])
        long_df.to_csv(
            output_dir / "PREDICTIONS_same_test_Full69_vs_SHAP_TopK_available_models.csv",
            index=False,
            encoding="utf-8-sig",
        )

    run_info = {
        "protocol": "NO RETRAINING; SHAP Best Top-K reevaluated on exact C6_Full common Test timestamps",
        "baseline_timestamp_file": str(baseline_times_path),
        "baseline_prediction_file": str(baseline_pred_path),
        "reference_timestamp_count": int(len(reference_times)),
        "strict_same_test": bool(STRICT_SAME_TEST),
        "completed_models_this_run": completed_models,
        "model_order": MODEL_ORDER,
    }
    with open(output_dir / "comparison_config.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print("[COMPLETE - NO RETRAINING]")
    print("Models processed now :", completed_models)
    print("Main comparison      :", output_dir / "MAIN_DAY_NMAE_Full69_vs_SHAP_TopK_same_test.csv")
    print("All metrics          :", output_dir / "METRICS_same_test_Full69_vs_SHAP_TopK_available_models.csv")
    print("Aligned predictions  :", output_dir / "PREDICTIONS_same_test_Full69_vs_SHAP_TopK_available_models.csv")
    print("Status               :", output_dir / "model_availability_status.csv")
    print("\nYou can rerun the SAME script after LSTM finishes; newly available models will be included automatically.")
    print("=" * 100)


if __name__ == "__main__":
    main()
