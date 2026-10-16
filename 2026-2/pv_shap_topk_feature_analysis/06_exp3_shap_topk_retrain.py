# -*- coding: utf-8 -*-
"""
Experiment 3: model-specific SHAP Top-K exhaustive retraining
===============================================================

Research protocol implemented in this script
--------------------------------------------
1) Use each model's own SHAP ranking (no correlation pre-filtering).
2) Exhaustively evaluate Top1 ... Top69 (or all available SHAP-ranked features).
3) Models: LightGBM, Random Forest, LSTM; PatchTST is optional.
4) Random Forest uses the supplied RF training code/protocol.
5) PatchTST uses the supplied official Hugging Face PatchTSTForRegression code,
   NOT the old SimplePatchTSTRegressor.
6) Select the best K ONLY by 2023 Validation DAY_NMAE.
7) 2024 Test is blocked during Top-K screening and is used ONLY after K selection
   for one final evaluation per selected model.

Important
---------
- The supplied 04 training scripts are imported dynamically, so their original
  hyperparameter spaces, training logic, model saving, and metric definitions
  are reused directly.
- During Top-K screening, Validation is passed as the third dataframe only as a
  placeholder because the legacy training functions require a test_df argument.
  Real 2024 Test data are NOT passed into those calls.
- After the best K is selected, the selected configuration is trained once more
  with the real 2024 Test dataframe for final evaluation.

PyCharm / IDE default
---------------------
Just press Run after placing this file in the same folder as:
  - chungbuk_pv_features.csv
  - 04_exp1_train.py   (or set BASE_TRAIN_SCRIPT below)
  - 04_exp1_train_random_forest.py (or set RF_TRAIN_SCRIPT below)
  - 04_exp1_train_patchtst_official.py (only needed if USE_PATCHTST=True)
  - pv_shap_lgbm_rf_lstm_outputs/<model>/shap_feature_importance.csv (3-model default)

PatchTST is OFF by default. The current 3-model SHAP runner does NOT create a
PatchTST SHAP ranking. Only set USE_PATCHTST = True after
<SHAP_ROOT>/patchtst/shap_feature_importance.csv has been generated separately.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# =============================================================================
# 0. USER-EDITABLE DEFAULTS (PyCharm Run button friendly)
# =============================================================================
USE_PATCHTST = False

CSV_PATH = "chungbuk_pv_features.csv"
SHAP_ROOT = "pv_shap_lgbm_rf_lstm_outputs"
OUTPUT_DIR = "pv_exp3_1_modelwise_shap_topk"

# Change these names only if your local 04 files use different filenames.
BASE_TRAIN_SCRIPT = "04_exp1_train_v1.py"                 # LightGBM + LSTM
RF_TRAIN_SCRIPT = "04_exp1_train_random_forest.py"     # Random Forest
PATCHTST_TRAIN_SCRIPT = "04_exp1_train_patchtst.py"  # official HF PatchTST

DEFAULT_MODELS = "lightgbm,random_forest,lstm"
DEFAULT_TOPK_MAX = 69
DEFAULT_N_TRIALS_LGB = 50
DEFAULT_N_TRIALS_RF = 50
DEFAULT_N_TRIALS_SEQ = 50
DEFAULT_EPOCHS = 200
DEFAULT_PATIENCE = 25
DEFAULT_SEED = 42
DEFAULT_RF_N_JOBS = -1


# =============================================================================
# 1. Utilities
# =============================================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)




def resolve_training_script(requested: str, kind: str) -> Path:
    """Resolve conventional names and downloaded duplicate-name variants."""
    p = Path(requested)
    if p.exists():
        return p

    cwd = Path.cwd()
    if kind == "base":
        preferred = [
            "04_exp1_train_v1(10).py",
            "04_exp1_train_v1.py",
            "04_exp1_train(20260911-114943).py",
            "04_exp1_train.py",
        ]
        matches = [x for x in cwd.glob("04_exp1_train*.py")
                   if "random_forest" not in x.name.lower() and "patchtst" not in x.name.lower()]
    elif kind == "rf":
        preferred = [
            "04_exp1_train_random_forest(8).py",
            "04_exp1_train_random_forest(7).py",
            "04_exp1_train_random_forest.py",
        ]
        matches = list(cwd.glob("04_exp1_train_random_forest*.py"))
    elif kind == "patchtst":
        preferred = [
            "04_exp1_train_patchtst.py",
            "04_exp1_train_patchtst_official(5).py",
            "04_exp1_train_patchtst_official.py",
        ]
        matches = list(cwd.glob("04_exp1_train_patchtst*.py"))
    else:
        raise ValueError(kind)

    for name in preferred:
        q = cwd / name
        if q.exists():
            print(f"[학습 코드 자동탐색] {kind}: {q.name}")
            return q

    if matches:
        matches = sorted(matches, key=lambda x: x.stat().st_mtime, reverse=True)
        print(f"[학습 코드 자동탐색] {kind}: {matches[0].name}")
        return matches[0]

    raise FileNotFoundError(
        f"{kind} 학습 스크립트를 찾을 수 없습니다. 요청 경로={requested}"
    )


def load_python_module(module_name: str, script_path: Path):
    if not script_path.exists():
        raise FileNotFoundError(
            f"학습 스크립트를 찾을 수 없습니다: {script_path}\n"
            f"이 파일 상단의 경로 설정 또는 실행 옵션을 확인하세요."
        )
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Python 모듈로 불러올 수 없습니다: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def normalize_model_name(name: str) -> str:
    x = name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "lgb": "lightgbm",
        "lgbm": "lightgbm",
        "light_gbm": "lightgbm",
        "lightgbm": "lightgbm",
        "rf": "random_forest",
        "randomforest": "random_forest",
        "random_forest": "random_forest",
        "lstm": "lstm",
        "patch": "patchtst",
        "patch_tst": "patchtst",
        "patchtst": "patchtst",
    }
    if x not in aliases:
        raise ValueError(f"지원하지 않는 모델명입니다: {name}")
    return aliases[x]


def parse_models(text: str, use_patchtst: bool) -> List[str]:
    out: List[str] = []
    for token in text.split(","):
        if token.strip():
            m = normalize_model_name(token)
            if m not in out:
                out.append(m)
    if use_patchtst and "patchtst" not in out:
        out.append("patchtst")
    if not use_patchtst and "patchtst" in out:
        out.remove("patchtst")
    if not out:
        raise ValueError("실행할 모델이 없습니다.")
    return out


def model_display_name(model: str) -> str:
    return {
        "lightgbm": "LightGBM",
        "random_forest": "RandomForest",
        "lstm": "LSTM",
        "patchtst": "PatchTST",
    }[model]


def default_shap_path(shap_root: Path, model: str) -> Path:
    folder = {
        "lightgbm": "lightgbm",
        "random_forest": "random_forest",
        "lstm": "lstm",
        "patchtst": "patchtst",
    }[model]
    return shap_root / folder / "shap_feature_importance.csv"


def load_model_shap_rank(path: Path, df_columns: List[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"SHAP 중요도 파일이 없습니다: {path}")
    rank = pd.read_csv(path, encoding="utf-8-sig")
    if "feature" not in rank.columns:
        raise ValueError(f"{path}: 'feature' 컬럼이 없습니다.")

    # Respect explicit rank if present; otherwise sort by mean_abs_shap.
    if "rank" in rank.columns:
        rank = rank.sort_values("rank", ascending=True)
    elif "mean_abs_shap" in rank.columns:
        rank = rank.sort_values("mean_abs_shap", ascending=False)
    elif "normalized_importance_pct" in rank.columns:
        rank = rank.sort_values("normalized_importance_pct", ascending=False)
    else:
        raise ValueError(
            f"{path}: rank / mean_abs_shap / normalized_importance_pct 중 하나가 필요합니다."
        )

    rank = rank.drop_duplicates(subset=["feature"], keep="first").copy()
    rank = rank[rank["feature"].isin(df_columns)].reset_index(drop=True)
    if rank.empty:
        raise ValueError(f"{path}: 입력 CSV와 일치하는 SHAP 특징이 없습니다.")
    rank["rank"] = np.arange(1, len(rank) + 1)
    return rank


def topk_features(rank: pd.DataFrame, k: int) -> List[str]:
    return rank.head(k)["feature"].tolist()


def clean_screening_summary(summary: Dict, model: str, k: int) -> Dict:
    """Keep validation-only fields; strip placeholder TEST fields."""
    out = {
        k0: v for k0, v in summary.items()
        if not str(k0).startswith("TEST_")
    }
    out["model_key"] = model
    out["shap_top_k"] = int(k)
    out["selection_metric"] = "VALID_DAY_NMAE_pct"
    out["real_test_used"] = False
    return out


def get_validation_selection_score(summary: Dict) -> float:
    # Prefer the final model's validation metric. Fallback to Optuna objective.
    value = summary.get("VALID_DAY_NMAE_pct", np.nan)
    try:
        value = float(value)
    except Exception:
        value = np.nan
    if np.isnan(value):
        value = summary.get("valid_objective_DAY_NMAE_pct", np.nan)
        try:
            value = float(value)
        except Exception:
            value = np.nan
    return float(value)


def save_model_screening_table(rows: List[Dict], out_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty:
        preferred = [
            "model", "model_key", "case", "shap_top_k", "feature_count", "seq_len",
            "VALID_DAY_n_samples", "VALID_DAY_MAE_MWh", "VALID_DAY_RMSE_MWh",
            "VALID_DAY_NMAE_pct", "VALID_DAY_NRMSE_pct", "VALID_DAY_MBE_MWh",
            "VALID_DAY_R2", "valid_objective_DAY_NMAE_pct",
            "selection_metric", "real_test_used", "model_path", "scaler_path",
        ]
        cols = [c for c in preferred if c in df.columns] + [
            c for c in df.columns if c not in preferred
        ]
        df = df[cols]
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


def select_best_k(screening_df: pd.DataFrame) -> pd.Series:
    if screening_df.empty:
        raise ValueError("Top-K screening 결과가 비어 있습니다.")
    if "VALID_DAY_NMAE_pct" in screening_df.columns:
        score = pd.to_numeric(screening_df["VALID_DAY_NMAE_pct"], errors="coerce")
    else:
        score = pd.Series(np.nan, index=screening_df.index)
    if score.isna().all() and "valid_objective_DAY_NMAE_pct" in screening_df.columns:
        score = pd.to_numeric(screening_df["valid_objective_DAY_NMAE_pct"], errors="coerce")
    valid = screening_df.loc[score.notna()].copy()
    valid["_selection_score"] = score.loc[valid.index]
    if valid.empty:
        raise ValueError("Validation DAY_NMAE가 유효한 Top-K 결과가 없습니다.")
    # Primary: minimum Validation DAY_NMAE. Tie-break: fewer features.
    valid = valid.sort_values(["_selection_score", "shap_top_k"], ascending=[True, True])
    return valid.iloc[0]


def remove_placeholder_test_prediction(output_root: Path, model: str, case_name: str) -> None:
    """Delete screening files named test_predictions_* because they are Validation placeholders."""
    pred_dir = output_root / "predictions"
    if not pred_dir.exists():
        return
    patterns = {
        "lightgbm": [f"test_predictions_lightgbm_{case_name}.csv"],
        "random_forest": [f"test_predictions_randomforest_{case_name}.csv"],
        "lstm": [f"test_predictions_lstm_{case_name}.csv"],
        "patchtst": [f"test_predictions_patchtst_{case_name}.csv"],
    }[model]
    for name in patterns:
        p = pred_dir / name
        if p.exists():
            p.unlink()


def add_screening_note(output_root: Path, model: str, case_name: str) -> None:
    model_folder = {
        "lightgbm": "lightgbm",
        "random_forest": "random_forest",
        "lstm": "lstm",
        "patchtst": "patchtst",
    }[model]
    note_path = output_root / "models" / model_folder / case_name / "SCREENING_NOTE.json"
    save_json(
        {
            "stage": "Top-K screening",
            "real_2024_test_used": False,
            "selection_data": "2023 Validation",
            "selection_metric": "Validation daytime NMAE",
            "note": (
                "The legacy training function requires test_df. During screening, a copy of "
                "Validation was supplied only as a placeholder. Any internal TEST fields in "
                "metadata are NOT real 2024 Test results and must not be used."
            ),
        },
        note_path,
    )


def build_final_common_timestamp_metrics(
    pred_by_model: Dict[str, pd.DataFrame],
    output_dir: Path,
    date_col: str,
    capacity_col: str,
    power_col: str,
    daytime_col: str,
) -> Optional[pd.DataFrame]:
    if len(pred_by_model) < 2:
        return None

    common = None
    for pred in pred_by_model.values():
        times = set(pd.to_datetime(pred[date_col]))
        common = times if common is None else common & times
    common_times = sorted(common or [])
    if not common_times:
        return None

    rows = []
    for model, pred in pred_by_model.items():
        work = pred.copy()
        work[date_col] = pd.to_datetime(work[date_col])
        work = work[work[date_col].isin(common_times)].sort_values(date_col)
        true_mwh = work["true_MWh"].to_numpy(float)
        pred_mwh = work["pred_MWh"].to_numpy(float)
        capacity = work[capacity_col].to_numpy(float)
        error = pred_mwh - true_mwh
        safe_capacity = np.where(capacity == 0, np.nan, capacity)
        day_mask = work[daytime_col].to_numpy(float) > 0

        def calc(mask):
            if int(np.sum(mask)) == 0:
                return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "NMAE": np.nan, "NRMSE": np.nan}
            e = error[mask]
            c = safe_capacity[mask]
            return {
                "n": int(np.sum(mask)),
                "MAE": float(np.mean(np.abs(e))),
                "RMSE": float(np.sqrt(np.mean(e ** 2))),
                "NMAE": float(np.nanmean(np.abs(e) / c) * 100.0),
                "NRMSE": float(np.sqrt(np.nanmean((e / c) ** 2)) * 100.0),
            }

        all_m = calc(np.ones(len(work), dtype=bool))
        day_m = calc(day_mask)
        rows.append({
            "model": model_display_name(model),
            "COMMON_TEST_n_samples": all_m["n"],
            "COMMON_TEST_MAE_MWh": all_m["MAE"],
            "COMMON_TEST_RMSE_MWh": all_m["RMSE"],
            "COMMON_TEST_NMAE_pct": all_m["NMAE"],
            "COMMON_TEST_NRMSE_pct": all_m["NRMSE"],
            "COMMON_TEST_DAY_n_samples": day_m["n"],
            "COMMON_TEST_DAY_MAE_MWh": day_m["MAE"],
            "COMMON_TEST_DAY_RMSE_MWh": day_m["RMSE"],
            "COMMON_TEST_DAY_NMAE_pct": day_m["NMAE"],
            "COMMON_TEST_DAY_NRMSE_pct": day_m["NRMSE"],
        })

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "results" / "FINAL_common_timestamp_test_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({date_col: common_times}).to_csv(
        output_dir / "results" / "FINAL_common_test_timestamps.csv", index=False, encoding="utf-8-sig"
    )
    return out


# =============================================================================
# 2. CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Model-specific SHAP Top-K exhaustive retraining")
    p.add_argument("--csv", type=str, default=CSV_PATH)
    p.add_argument("--shap_root", type=str, default=SHAP_ROOT)
    p.add_argument("--output_dir", type=str, default=OUTPUT_DIR)

    p.add_argument("--base_train_script", type=str, default=BASE_TRAIN_SCRIPT)
    p.add_argument("--rf_train_script", type=str, default=RF_TRAIN_SCRIPT)
    p.add_argument("--patchtst_train_script", type=str, default=PATCHTST_TRAIN_SCRIPT)

    p.add_argument("--models", type=str, default=DEFAULT_MODELS)
    p.add_argument("--use_patchtst", action="store_true", default=USE_PATCHTST)
    p.add_argument("--no_patchtst", action="store_true")

    p.add_argument("--topk_max", type=int, default=DEFAULT_TOPK_MAX)
    p.add_argument(
        "--topk_list", type=str, default="",
        help="선택 입력. 예: 1,2,3,5,10. 비우면 1~topk_max 전수조사"
    )

    p.add_argument("--n_trials_lgb", type=int, default=DEFAULT_N_TRIALS_LGB)
    p.add_argument("--n_trials_rf", type=int, default=DEFAULT_N_TRIALS_RF)
    p.add_argument("--n_trials_seq", type=int, default=DEFAULT_N_TRIALS_SEQ)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--rf_n_jobs", type=int, default=DEFAULT_RF_N_JOBS)
    p.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda")
    p.add_argument("--no_clip", action="store_true")

    p.add_argument(
        "--quick_test", action="store_true",
        help="코드 점검용: Top-K=1,2 / trial=1 / epoch=2 / patience=1"
    )
    p.add_argument(
        "--skip_existing", action="store_true",
        help="screening 결과 CSV에 이미 해당 K가 있으면 건너뜀"
    )
    return p.parse_args()


def parse_topk_list(text: str, max_k: int) -> List[int]:
    if text.strip():
        values = []
        for token in text.split(","):
            token = token.strip()
            if token:
                values.append(int(token))
        values = sorted(set(k for k in values if 1 <= k <= max_k))
    else:
        values = list(range(1, max_k + 1))
    if not values:
        raise ValueError("유효한 Top-K 후보가 없습니다.")
    return values


# =============================================================================
# 3. Model training dispatcher
# =============================================================================
def run_one_training(
    model: str,
    case_name: str,
    feature_cols: List[str],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_root: Path,
    base_mod,
    rf_mod,
    patch_mod,
    args,
    device: str,
) -> Tuple[Dict, pd.DataFrame]:
    clip_predictions = not args.no_clip

    if model == "lightgbm":
        return base_mod.train_lightgbm_case(
            case_name=case_name,
            feature_cols=feature_cols,
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            output_dir=output_root,
            n_trials=args.n_trials_lgb,
            seed=args.seed,
            clip_predictions=clip_predictions,
        )

    if model == "random_forest":
        return rf_mod.train_random_forest_case(
            case_name=case_name,
            feature_cols=feature_cols,
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            output_dir=output_root,
            n_trials=args.n_trials_rf,
            seed=args.seed,
            n_jobs=args.rf_n_jobs,
            clip_predictions=clip_predictions,
        )

    if model == "lstm":
        return base_mod.train_sequence_case(
            model_name="LSTM",
            case_name=case_name,
            feature_cols=feature_cols,
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            output_dir=output_root,
            n_trials=args.n_trials_seq,
            seed=args.seed,
            max_epochs=args.epochs,
            patience=args.patience,
            device=device,
            clip_predictions=clip_predictions,
        )

    if model == "patchtst":
        if patch_mod is None:
            raise RuntimeError("PatchTST가 요청되었지만 공식 PatchTST 학습 모듈이 로드되지 않았습니다.")
        return patch_mod.train_patchtst_case(
            case_name=case_name,
            feature_cols=feature_cols,
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            output_dir=output_root,
            n_trials=args.n_trials_seq,
            seed=args.seed,
            max_epochs=args.epochs,
            patience=args.patience,
            device=device,
            clip_predictions=clip_predictions,
        )

    raise ValueError(model)


# =============================================================================
# 4. Main
# =============================================================================
def main() -> None:
    args = parse_args()

    if args.no_patchtst:
        args.use_patchtst = False

    if args.quick_test:
        args.topk_list = "1,2"
        args.topk_max = 2
        args.n_trials_lgb = 1
        args.n_trials_rf = 1
        args.n_trials_seq = 1
        args.epochs = 2
        args.patience = 1
        print("[Quick test] Top-K=1,2 / trials=1 / epochs=2 / patience=1")

    models = parse_models(args.models, use_patchtst=args.use_patchtst)
    csv_path = Path(args.csv)
    shap_root = Path(args.shap_root)
    output_dir = Path(args.output_dir)
    screening_root = output_dir / "screening_validation_only"
    final_root = output_dir / "final_selected"
    results_dir = output_dir / "results"
    ensure_dir(screening_root)
    ensure_dir(final_root)
    ensure_dir(results_dir)

    # Load only the training modules that are actually needed.
    need_base = any(m in models for m in ["lightgbm", "lstm"])
    need_rf = "random_forest" in models
    need_patch = "patchtst" in models

    base_script = resolve_training_script(args.base_train_script, "base") if need_base else None
    rf_script = resolve_training_script(args.rf_train_script, "rf") if need_rf else None
    patch_script = resolve_training_script(args.patchtst_train_script, "patchtst") if need_patch else None

    base_mod = load_python_module("exp04_base_train", base_script) if need_base else None
    rf_mod = load_python_module("exp04_rf_train", rf_script) if need_rf else None
    patch_mod = load_python_module("exp04_patchtst_train", patch_script) if need_patch else None

    # Read data using the base module if available, otherwise another supplied module.
    data_mod = base_mod or rf_mod or patch_mod
    if data_mod is None:
        raise RuntimeError("학습 모듈을 하나도 불러오지 못했습니다.")
    df = data_mod.read_feature_csv(str(csv_path))
    train_df, valid_df, test_df = data_mod.split_by_year(df)

    date_col = data_mod.DATE_COL
    capacity_col = data_mod.CAPACITY_COL
    power_col = data_mod.POWER_COL
    daytime_col = data_mod.DAYTIME_COL

    # Resolve device.
    if args.device == "auto":
        torch_obj = getattr(base_mod, "torch", None) if base_mod is not None else None
        if torch_obj is None and patch_mod is not None:
            torch_obj = getattr(patch_mod, "torch", None)
        device = "cuda" if torch_obj is not None and torch_obj.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 100)
    print("[Experiment 3-1] Model-specific SHAP Top-K exhaustive retraining")
    print("Models          :", models)
    print("PatchTST        :", "ON (official PatchTSTForRegression)" if "patchtst" in models else "OFF")
    print("Train           : 2020-2022 /", len(train_df))
    print("Validation      : 2023 /", len(valid_df))
    print("Test            : 2024 /", len(test_df), "(BLOCKED during screening)")
    print("Device          :", device)
    print("Selection rule  : minimum 2023 Validation DAY_NMAE")
    print("Correlation prefilter: NONE")
    print("=" * 100)

    # Load model-specific SHAP ranks.
    shap_ranks: Dict[str, pd.DataFrame] = {}
    shap_paths: Dict[str, str] = {}
    for model in models:
        path = default_shap_path(shap_root, model)
        rank = load_model_shap_rank(path, list(df.columns))
        shap_ranks[model] = rank
        shap_paths[model] = str(path)
        rank.to_csv(results_dir / f"SHAP_rank_used_{model}.csv", index=False, encoding="utf-8-sig")
        print(f"[{model_display_name(model)}] SHAP features={len(rank)} | {path}")

    # Optional consistency check: all requested models should rank the same feature identities.
    feature_sets = {m: set(r["feature"].tolist()) for m, r in shap_ranks.items()}
    first_model = models[0]
    first_set = feature_sets[first_model]
    for model in models[1:]:
        if feature_sets[model] != first_set:
            missing = sorted(first_set - feature_sets[model])
            extra = sorted(feature_sets[model] - first_set)
            raise ValueError(
                f"모델별 SHAP feature identity가 다릅니다: {first_model} vs {model}\n"
                f"missing={missing[:10]} / extra={extra[:10]}"
            )

    max_available_k = min(len(shap_ranks[m]) for m in models)
    max_k = min(int(args.topk_max), max_available_k)
    topk_candidates = parse_topk_list(args.topk_list, max_k)

    config = {
        "experiment": "Experiment 3-1 - model-specific SHAP Top-K exhaustive retraining",
        "models": models,
        "patchtst_enabled": "patchtst" in models,
        "patchtst_implementation": "transformers.PatchTSTForRegression" if "patchtst" in models else None,
        "csv": str(csv_path),
        "shap_paths": shap_paths,
        "shap_selection": "each model uses its own SHAP rank",
        "correlation_prefilter": False,
        "topk_candidates": topk_candidates,
        "split": {"train": "2020-2022", "validation": "2023", "test": "2024"},
        "selection_metric": "VALID_DAY_NMAE_pct",
        "best_k_selection": "validation only; tie -> smaller K",
        "screening_real_test_access": False,
        "final_test_usage": "only selected best-K configuration per model",
        "n_trials": {
            "lightgbm": args.n_trials_lgb,
            "random_forest": args.n_trials_rf,
            "sequence": args.n_trials_seq,
        },
        "epochs": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
        "device": device,
        "clip_predictions": not args.no_clip,
        "training_scripts": {
            "base": str(base_script) if need_base else None,
            "random_forest": str(rf_script) if need_rf else None,
            "patchtst": str(patch_script) if need_patch else None,
        },
    }
    save_json(config, output_dir / "experiment_config.json")

    best_rows: List[Dict] = []
    final_rows: List[Dict] = []
    final_pred_by_model: Dict[str, pd.DataFrame] = {}

    # -------------------------------------------------------------------------
    # A. Top-K screening: 2024 Test is NOT passed into training.
    # -------------------------------------------------------------------------
    for model in models:
        display = model_display_name(model)
        model_screen_root = screening_root / model
        ensure_dir(model_screen_root)
        screening_csv = results_dir / f"SCREENING_{model}_topk_validation.csv"

        existing_rows: List[Dict] = []
        completed_k = set()
        if args.skip_existing and screening_csv.exists():
            old = pd.read_csv(screening_csv, encoding="utf-8-sig")
            existing_rows = old.to_dict("records")
            if "shap_top_k" in old.columns:
                completed_k = set(pd.to_numeric(old["shap_top_k"], errors="coerce").dropna().astype(int).tolist())

        rows: List[Dict] = existing_rows
        rank = shap_ranks[model]

        print("\n" + "#" * 100)
        print(f"[{display}] Top-K screening 시작 - REAL 2024 TEST BLOCKED")
        print("#" * 100)

        for k in topk_candidates:
            if k in completed_k:
                print(f"[{display}] Top{k}: 기존 결과 사용 (--skip_existing)")
                continue

            features = topk_features(rank, k)
            case_name = f"SHAP_{display}_Top{k}"
            pd.DataFrame({
                "rank": np.arange(1, len(features) + 1),
                "feature": features,
            }).to_csv(
                results_dir / f"features_{model}_Top{k}.csv",
                index=False,
                encoding="utf-8-sig",
            )

            print("\n" + "=" * 92)
            print(f"[{display}] Top{k}/{max_k} | n_features={len(features)}")
            print("Selection data: Train 2020-2022 + Validation 2023")
            print("2024 Test: NOT PROVIDED")

            # Critical leakage guard:
            # The legacy 04 function requires test_df, therefore a COPY of Validation
            # is supplied as a placeholder. The actual 2024 test_df object is never
            # passed to this screening call.
            validation_placeholder = valid_df.copy()
            summary, _placeholder_pred = run_one_training(
                model=model,
                case_name=case_name,
                feature_cols=features,
                train_df=train_df,
                valid_df=valid_df,
                test_df=validation_placeholder,
                output_root=model_screen_root,
                base_mod=base_mod,
                rf_mod=rf_mod,
                patch_mod=patch_mod,
                args=args,
                device=device,
            )

            row = clean_screening_summary(summary, model=model, k=k)
            row["selection_score_VALID_DAY_NMAE_pct"] = get_validation_selection_score(summary)
            row["shap_rank_file"] = shap_paths[model]
            rows.append(row)

            remove_placeholder_test_prediction(model_screen_root, model, case_name)
            add_screening_note(model_screen_root, model, case_name)
            save_model_screening_table(rows, screening_csv)

        screening_df = save_model_screening_table(rows, screening_csv)
        best = select_best_k(screening_df)
        best_k = int(best["shap_top_k"])
        best_score = float(best["_selection_score"] if "_selection_score" in best else get_validation_selection_score(best.to_dict()))
        best_features = topk_features(rank, best_k)

        best_row = best.to_dict()
        best_row.pop("_selection_score", None)
        best_row["BEST_K"] = best_k
        best_row["BEST_VALID_DAY_NMAE_pct"] = best_score
        best_row["selection_based_on_test"] = False
        best_rows.append(best_row)
        pd.DataFrame([best_row]).to_csv(
            results_dir / f"BEST_K_by_validation_{model}.csv", index=False, encoding="utf-8-sig"
        )

        print("\n" + "*" * 92)
        print(f"[{display}] BEST K = {best_k} | Validation DAY_NMAE = {best_score:.6f}%")
        print("이제 선택된 K 한 개만 실제 2024 Test로 최종 평가합니다.")
        print("*" * 92)

        # ---------------------------------------------------------------------
        # B. Final selected model: real 2024 Test is used exactly here.
        # ---------------------------------------------------------------------
        final_case_name = f"FINAL_SHAP_{display}_Top{best_k}"
        model_final_root = final_root / model
        summary_final, pred_final = run_one_training(
            model=model,
            case_name=final_case_name,
            feature_cols=best_features,
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,  # <-- the ONLY stage where real 2024 Test is supplied
            output_root=model_final_root,
            base_mod=base_mod,
            rf_mod=rf_mod,
            patch_mod=patch_mod,
            args=args,
            device=device,
        )
        summary_final = dict(summary_final)
        summary_final["model_key"] = model
        summary_final["selected_shap_top_k"] = best_k
        summary_final["selected_by"] = "2023 Validation DAY_NMAE only"
        summary_final["selection_validation_DAY_NMAE_pct"] = best_score
        summary_final["real_2024_test_used"] = True
        summary_final["shap_rank_file"] = shap_paths[model]
        final_rows.append(summary_final)
        final_pred_by_model[model] = pred_final.copy()

        pd.DataFrame([summary_final]).to_csv(
            results_dir / f"FINAL_TEST_{model}.csv", index=False, encoding="utf-8-sig"
        )

    # -------------------------------------------------------------------------
    # C. Global result tables
    # -------------------------------------------------------------------------
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(results_dir / "BEST_K_all_models_by_validation.csv", index=False, encoding="utf-8-sig")

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(results_dir / "FINAL_TEST_all_models.csv", index=False, encoding="utf-8-sig")

    build_final_common_timestamp_metrics(
        final_pred_by_model,
        output_dir,
        date_col=date_col,
        capacity_col=capacity_col,
        power_col=power_col,
        daytime_col=daytime_col,
    )

    print("\n" + "=" * 100)
    print("[완료]")
    print("모델별 Top-K Validation 탐색 :", results_dir / "SCREENING_<model>_topk_validation.csv")
    print("Validation 기준 최적 K       :", results_dir / "BEST_K_all_models_by_validation.csv")
    print("최종 2024 Test 결과          :", results_dir / "FINAL_TEST_all_models.csv")
    print("공통 timestamp 최종 비교     :", results_dir / "FINAL_common_timestamp_test_metrics.csv")
    print("PatchTST                     :", "included" if "patchtst" in models else "not included")
    print("=" * 100)


if __name__ == "__main__":
    main()
