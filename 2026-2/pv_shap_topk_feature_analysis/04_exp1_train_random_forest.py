# -*- coding: utf-8 -*-
"""
Random Forest-only experiment for the PV forecasting study.

This file intentionally leaves the original LightGBM/LSTM/PatchTST training
script and its output directory untouched.

Protocol kept identical to 04_exp1_train.py:
    - target: 정규화발전량
    - feature cases: C1-C6
    - split: train 2020-2022 / validation 2023 / test 2024
    - case-wise dropna using only the selected features and required columns
    - hyperparameter selection: validation daytime NMAE
    - test set: final evaluation only
    - negative predictions: clipped to zero by default

After RF training, this script can read:
  - LightGBM/LSTM predictions from pv_training_outputs
  - the retrained official PatchTST predictions from pv_training_outputs_patchtst
and recompute fair four-model comparisons on common Test timestamps.

Two comparison tables are produced:
  1) C6_Full fixed comparison (same 69 feature identities; PRIMARY for SHAP linkage)
  2) Validation-best-case comparison (SECONDARY model-wise optimum comparison)

Recommended run:
    python 04_exp1_train_random_forest.py \
        --csv chungbuk_pv_features.csv \
        --output_dir pv_training_outputs_rf \
        --existing_output_dir pv_training_outputs \
        --patchtst_output_dir pv_training_outputs_patchtst \
        --n_trials 50 --seed 42

If RF is already trained and only the four-model tables must be rebuilt:
    python 04_exp1_train_random_forest.py --comparison_only
"""

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

try:
    import optuna
except Exception:
    optuna = None


# ============================================================
# 0. Global column names (same as the original experiment)
# ============================================================
DATE_COL = "발전일자"
SEGMENT_COL = "segment_id"
TARGET_COL = "정규화발전량"
CAPACITY_COL = "설비용량(MW)"
POWER_COL = "발전량(MWh)"
DAYTIME_COL = "대기권밖일사량계산값"
YEAR_COL = "year"

EXCLUDE_FEATURE_COLS = [
    DATE_COL,
    SEGMENT_COL,
    CAPACITY_COL,
    POWER_COL,
    TARGET_COL,
    YEAR_COL,
]

G1_TIME_BASE = ["month", "day", "hour", "doy", "sin_hour", "cos_hour"]
G2_CURRENT_BASE = [
    "기온",
    "습도",
    "풍속",
    "적운량(10분위)",
    "일조(hr)",
    "대기권밖일사량계산값",
    "일사량",
]

EXPECTED_BASE_MODELS = ["LightGBM", "LSTM"]
PATCHTST_MODEL_NAME = "PatchTST"
RF_MODEL_NAME = "RandomForest"


# ============================================================
# 1. Utilities and experiment protocol
# ============================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_feature_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = [DATE_COL, TARGET_COL, CAPACITY_COL, POWER_COL, DAYTIME_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    if df[DATE_COL].duplicated().any():
        n_dup = int(df[DATE_COL].duplicated().sum())
        raise ValueError(f"'{DATE_COL}'에 중복 시점이 {n_dup}개 있습니다.")
    return df.sort_values(DATE_COL).reset_index(drop=True)


def build_feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    groups = {
        "G1_Time": [c for c in G1_TIME_BASE if c in df.columns],
        "G2_Current": [c for c in G2_CURRENT_BASE if c in df.columns],
        "G3_Lag": [c for c in df.columns if "_lag_" in c],
        "G4_Rolling": [c for c in df.columns if "_rollmean_" in c],
        "G5_Diff": [c for c in df.columns if "_diff_" in c],
    }
    for name, cols in groups.items():
        if not cols:
            print(f"[경고] {name} 컬럼이 비어 있습니다.")
    return groups


def build_feature_cases(groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
    g1 = groups["G1_Time"]
    g2 = groups["G2_Current"]
    g3 = groups["G3_Lag"]
    g4 = groups["G4_Rolling"]
    g5 = groups["G5_Diff"]

    raw_cases = {
        "C1_Time_Current": g1 + g2,
        "C2_Time_Current_Lag": g1 + g2 + g3,
        "C3_Time_Current_Rolling": g1 + g2 + g4,
        "C4_Time_Current_Diff": g1 + g2 + g5,
        "C5_Time_Current_Lag_Rolling": g1 + g2 + g3 + g4,
        "C6_Full": g1 + g2 + g3 + g4 + g5,
    }

    cases: Dict[str, List[str]] = {}
    for case_name, cols in raw_cases.items():
        seen = set()
        cases[case_name] = []
        for col in cols:
            if col not in seen and col not in EXCLUDE_FEATURE_COLS:
                cases[case_name].append(col)
                seen.add(col)
    return cases


def split_by_year(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    year = df[DATE_COL].dt.year
    train_df = df[(year >= 2020) & (year <= 2022)].copy()
    valid_df = df[year == 2023].copy()
    test_df = df[year == 2024].copy()
    for name, part in [("Train", train_df), ("Validation", valid_df), ("Test", test_df)]:
        if part.empty:
            raise ValueError(f"{name} 데이터가 비어 있습니다. 연도 분할을 확인하세요.")
    return train_df, valid_df, test_df


def clip_pred(pred: np.ndarray, do_clip: bool = True) -> np.ndarray:
    pred = np.asarray(pred, dtype=float).reshape(-1)
    return np.clip(pred, 0.0, None) if do_clip else pred


def metrics_from_arrays(
    true_mwh: np.ndarray,
    pred_mwh: np.ndarray,
    capacity_mw: np.ndarray,
) -> Dict[str, float]:
    true_mwh = np.asarray(true_mwh, dtype=float)
    pred_mwh = np.asarray(pred_mwh, dtype=float)
    capacity_mw = np.asarray(capacity_mw, dtype=float)

    if len(true_mwh) == 0:
        return {
            "n_samples": 0,
            "MAE_MWh": np.nan,
            "RMSE_MWh": np.nan,
            "NMAE_pct": np.nan,
            "NRMSE_pct": np.nan,
            "MBE_MWh": np.nan,
            "R2": np.nan,
        }

    error = pred_mwh - true_mwh
    abs_error = np.abs(error)
    safe_capacity = np.where(capacity_mw == 0, np.nan, capacity_mw)

    try:
        r2 = r2_score(true_mwh, pred_mwh)
    except Exception:
        r2 = np.nan

    return {
        "n_samples": int(len(true_mwh)),
        "MAE_MWh": float(np.mean(abs_error)),
        "RMSE_MWh": float(np.sqrt(np.mean(error ** 2))),
        "NMAE_pct": float(np.nanmean(abs_error / safe_capacity) * 100.0),
        "NRMSE_pct": float(np.sqrt(np.nanmean((error / safe_capacity) ** 2)) * 100.0),
        "MBE_MWh": float(np.mean(error)),
        "R2": float(r2),
    }


def evaluate_prediction_df(pred_df: pd.DataFrame) -> Dict[str, float]:
    all_metrics = metrics_from_arrays(
        pred_df["true_MWh"].values,
        pred_df["pred_MWh"].values,
        pred_df[CAPACITY_COL].values,
    )
    day_df = pred_df[pred_df[DAYTIME_COL] > 0].copy()
    day_metrics = metrics_from_arrays(
        day_df["true_MWh"].values,
        day_df["pred_MWh"].values,
        day_df[CAPACITY_COL].values,
    )
    return {
        **{f"ALL_{k}": v for k, v in all_metrics.items()},
        **{f"DAY_{k}": v for k, v in day_metrics.items()},
    }


def build_prediction_df(
    base_df: pd.DataFrame,
    pred_norm: np.ndarray,
    model_name: str,
    case_name: str,
    clip_predictions: bool,
) -> pd.DataFrame:
    pred_norm = clip_pred(pred_norm, clip_predictions)
    if len(base_df) != len(pred_norm):
        raise ValueError("예측값 개수와 평가 데이터 행 수가 다릅니다.")

    cols = [DATE_COL, CAPACITY_COL, POWER_COL, TARGET_COL, DAYTIME_COL]
    if SEGMENT_COL in base_df.columns:
        cols.append(SEGMENT_COL)
    out = base_df[cols].copy().reset_index(drop=True)
    out["pred_norm"] = pred_norm
    out["true_norm"] = out[TARGET_COL]
    out["true_MWh"] = out[POWER_COL]
    out["pred_MWh"] = out["pred_norm"] * out[CAPACITY_COL]
    out["error_MWh"] = out["pred_MWh"] - out["true_MWh"]
    out["model"] = model_name
    out["case"] = case_name
    return out


# ============================================================
# 2. Random Forest training
# ============================================================
def random_forest_params(trial, seed: int, n_jobs: int) -> Dict:
    """Search space fixed in this file for reproducibility."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
        "max_depth": trial.suggest_categorical(
            "max_depth", [None, 8, 12, 16, 20, 30]
        ),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.5, 0.7, 1.0]
        ),
        "bootstrap": True,
        "max_samples": trial.suggest_categorical("max_samples", [0.7, 0.85, 1.0]),
        "criterion": "squared_error",
        "random_state": seed,
        "n_jobs": n_jobs,
    }


def train_random_forest_case(
    case_name: str,
    feature_cols: List[str],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    n_trials: int,
    seed: int,
    n_jobs: int,
    clip_predictions: bool,
) -> Tuple[Dict, pd.DataFrame]:
    if optuna is None:
        raise ImportError("optuna가 설치되어 있지 않습니다: pip install optuna")
    if not feature_cols:
        raise ValueError(f"{case_name}의 입력 특징이 비어 있습니다.")

    # Same case-wise dropna protocol as the existing LightGBM experiment.
    use_cols = feature_cols + [
        TARGET_COL,
        DATE_COL,
        CAPACITY_COL,
        POWER_COL,
        DAYTIME_COL,
    ]
    if SEGMENT_COL in train_df.columns:
        use_cols.append(SEGMENT_COL)
    use_cols = list(dict.fromkeys(use_cols))

    tr = train_df[use_cols].dropna().copy()
    va = valid_df[use_cols].dropna().copy()
    te = test_df[use_cols].dropna().copy()
    if tr.empty or va.empty or te.empty:
        raise ValueError(
            f"{case_name}: dropna 후 데이터가 비어 있습니다 "
            f"(train={len(tr)}, valid={len(va)}, test={len(te)})."
        )

    X_tr = tr[feature_cols]
    y_tr = tr[TARGET_COL]
    X_va = va[feature_cols]

    def objective(trial) -> float:
        params = random_forest_params(trial, seed=seed, n_jobs=n_jobs)
        model = RandomForestRegressor(**params)
        model.fit(X_tr, y_tr)
        pred_df = build_prediction_df(
            va,
            model.predict(X_va),
            RF_MODEL_NAME,
            case_name,
            clip_predictions,
        )
        metrics = evaluate_prediction_df(pred_df)
        score = metrics["DAY_NMAE_pct"]
        return metrics["ALL_NMAE_pct"] if np.isnan(score) else score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = {
        **study.best_params,
        "bootstrap": True,
        "criterion": "squared_error",
        "random_state": seed,
        "n_jobs": n_jobs,
    }

    # Like the existing LightGBM code, fit the selected model on Train only.
    # Validation has been used only for model selection; Test remains untouched.
    final_model = RandomForestRegressor(**best_params)
    final_model.fit(X_tr, y_tr)

    valid_pred_df = build_prediction_df(
        va,
        final_model.predict(X_va),
        RF_MODEL_NAME,
        case_name,
        clip_predictions,
    )
    test_pred_df = build_prediction_df(
        te,
        final_model.predict(te[feature_cols]),
        RF_MODEL_NAME,
        case_name,
        clip_predictions,
    )
    valid_metrics = evaluate_prediction_df(valid_pred_df)
    test_metrics = evaluate_prediction_df(test_pred_df)

    model_dir = output_dir / "models" / "random_forest" / case_name
    pred_dir = output_dir / "predictions"
    ensure_dir(model_dir)
    ensure_dir(pred_dir)

    model_path = model_dir / "model.pkl"
    joblib.dump(
        {
            "model_type": "random_forest",
            "model": final_model,
            "case_name": case_name,
            "feature_cols": feature_cols,
            "target_col": TARGET_COL,
            "date_col": DATE_COL,
            "capacity_col": CAPACITY_COL,
            "power_col": POWER_COL,
            "daytime_col": DAYTIME_COL,
            "exclude_feature_cols": EXCLUDE_FEATURE_COLS,
            "best_params": best_params,
            "clip_predictions": clip_predictions,
        },
        model_path,
    )

    save_json(
        {
            "best_params": best_params,
            "valid_objective_DAY_NMAE_pct": float(study.best_value),
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
        },
        model_dir / "metadata.json",
    )
    study.trials_dataframe().to_csv(
        model_dir / "optuna_trials.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        {
            "model": RF_MODEL_NAME,
            "case": case_name,
            "feature": feature_cols,
            "importance": final_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).to_csv(
        model_dir / "feature_importance.csv", index=False, encoding="utf-8-sig"
    )

    valid_pred_df.to_csv(
        pred_dir / f"valid_predictions_randomforest_{case_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_pred_df.to_csv(
        pred_dir / f"test_predictions_randomforest_{case_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "model": RF_MODEL_NAME,
        "case": case_name,
        "feature_count": len(feature_cols),
        "train_rows": len(tr),
        "valid_rows": len(va),
        "test_rows": len(te),
        "valid_objective_DAY_NMAE_pct": float(study.best_value),
        **{f"VALID_{k}": v for k, v in valid_metrics.items()},
        **{f"TEST_{k}": v for k, v in test_metrics.items()},
        "model_path": str(model_path),
    }
    return summary, test_pred_df


# ============================================================
# 3. Protocol validation and saved-prediction comparison
# ============================================================
def _compare_protocol_config(
    config_path: Path,
    current_config: Dict,
    source_label: str,
    compare_n_trials: bool = True,
) -> Dict:
    """Validate that a saved experiment uses the same core data protocol."""
    if not config_path.exists():
        raise FileNotFoundError(f"{source_label} 실험 설정 파일이 없습니다: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        old = json.load(f)

    fields = ["target", "split", "feature_cases", "clip_predictions"]
    if compare_n_trials:
        fields.append("n_trials")

    mismatches = []
    for key in fields:
        if old.get(key) != current_config.get(key):
            mismatches.append(
                {"field": key, "existing": old.get(key), "random_forest": current_config.get(key)}
            )

    old_csv = Path(str(old.get("input_csv", ""))).name
    new_csv = Path(str(current_config.get("input_csv", ""))).name
    if old_csv and old_csv != new_csv:
        mismatches.append(
            {"field": "input_csv filename", "existing": old_csv, "random_forest": new_csv}
        )

    # Newer scripts save a SHA256; compare it whenever both sides have one.
    old_hash = old.get("input_csv_sha256")
    new_hash = current_config.get("input_csv_sha256")
    if old_hash and new_hash and old_hash != new_hash:
        mismatches.append(
            {"field": "input_csv_sha256", "existing": old_hash, "random_forest": new_hash}
        )

    if mismatches:
        details = "\n".join(
            f"- {m['field']}: 기존={m['existing']} / RF={m['random_forest']}"
            for m in mismatches
        )
        raise RuntimeError(
            f"{source_label} 실험과 Random Forest 실험의 핵심 조건이 다릅니다.\n" + details
        )

    return {
        "status": "passed",
        "source": source_label,
        "config_path": str(config_path),
        "checked_fields": fields + ["input_csv filename"] + (
            ["input_csv_sha256"] if old_hash and new_hash else []
        ),
    }


def validate_existing_protocol(
    existing_output_dir: Path,
    patchtst_output_dir: Path,
    current_config: Dict,
) -> Dict:
    """Validate LightGBM/LSTM and the NEW official PatchTST experiment separately."""
    base_check = _compare_protocol_config(
        existing_output_dir / "experiment_config.json",
        current_config,
        source_label="기존 LightGBM/LSTM",
        compare_n_trials=True,
    )
    patch_check = _compare_protocol_config(
        patchtst_output_dir / "experiment_config_patchtst.json",
        current_config,
        source_label="공식 PatchTST",
        compare_n_trials=True,
    )

    # Explicit architecture guard: do not accidentally merge the old PatchTST-style run.
    with open(patchtst_output_dir / "experiment_config_patchtst.json", "r", encoding="utf-8") as f:
        patch_cfg = json.load(f)
    if patch_cfg.get("implementation") != "transformers.PatchTSTForRegression":
        raise RuntimeError(
            "PatchTST 결과가 공식 PatchTSTForRegression 실험이 아닙니다: "
            f"{patch_cfg.get('implementation')}"
        )
    arch = patch_cfg.get("patchtst_architecture", {})
    if arch.get("share_embedding") is not True or arch.get("channel_attention") is not False:
        raise RuntimeError(
            "PatchTST channel-independence 설정이 기대값과 다릅니다 "
            "(share_embedding=True, channel_attention=False 필요)."
        )

    return {
        "status": "passed",
        "base_models": base_check,
        "official_patchtst": patch_check,
        "note": "LightGBM/LSTM과 새 공식 PatchTST를 서로 다른 출력 폴더에서 검증했습니다.",
    }


def prediction_filename(model_name: str, case_name: str) -> str:
    slug = {
        "LightGBM": "lightgbm",
        "LSTM": "lstm",
        "PatchTST": "patchtst",
        RF_MODEL_NAME: "randomforest",
    }.get(model_name)
    if slug is None:
        raise ValueError(f"지원하지 않는 모델명입니다: {model_name}")
    return f"test_predictions_{slug}_{case_name}.csv"


def read_prediction_csv(path: Path, model_name: str, case_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"예측 파일이 없습니다: {path}")
    pred = pd.read_csv(path, encoding="utf-8-sig")
    required = [
        DATE_COL,
        CAPACITY_COL,
        DAYTIME_COL,
        "true_MWh",
        "pred_MWh",
    ]
    missing = [c for c in required if c not in pred.columns]
    if missing:
        raise ValueError(f"{path.name}에 필수 컬럼이 없습니다: {missing}")
    pred[DATE_COL] = pd.to_datetime(pred[DATE_COL])
    if pred[DATE_COL].duplicated().any():
        raise ValueError(f"{path.name}에 중복 timestamp가 있습니다.")
    pred["model"] = model_name
    pred["case"] = case_name
    return pred


def select_best_case_rows(summary_df: pd.DataFrame) -> pd.DataFrame:
    required = ["model", "case", "valid_objective_DAY_NMAE_pct"]
    missing = [c for c in required if c not in summary_df.columns]
    if missing:
        raise ValueError(f"요약 결과에 필수 컬럼이 없습니다: {missing}")

    work = summary_df.dropna(subset=["valid_objective_DAY_NMAE_pct"]).copy()
    if work.empty:
        raise ValueError("Validation DAY_NMAE가 기록된 결과가 없습니다.")
    idx = work.groupby("model")["valid_objective_DAY_NMAE_pct"].idxmin()
    return work.loc[idx].sort_values("model").reset_index(drop=True)


def assert_same_ground_truth(
    selected_predictions: Dict[Tuple[str, str], pd.DataFrame],
    common_times: List[pd.Timestamp],
) -> None:
    reference_key: Optional[Tuple[str, str]] = None
    reference: Optional[pd.DataFrame] = None
    check_cols = ["true_MWh", CAPACITY_COL, DAYTIME_COL]

    for key, pred in selected_predictions.items():
        current = (
            pred[pred[DATE_COL].isin(common_times)]
            .sort_values(DATE_COL)
            .set_index(DATE_COL)[check_cols]
        )
        if reference is None:
            reference_key = key
            reference = current
            continue
        if not current.index.equals(reference.index):
            raise ValueError(f"공통 timestamp 정렬이 일치하지 않습니다: {key}")
        if not np.allclose(
            current.values.astype(float),
            reference.values.astype(float),
            rtol=1e-9,
            atol=1e-9,
            equal_nan=True,
        ):
            raise ValueError(
                f"동일 timestamp의 실제값/설비용량/주간판정 값이 다릅니다: "
                f"{reference_key} vs {key}. 동일한 원본 데이터를 사용했는지 확인하세요."
            )


def _load_model_summaries(
    existing_output_dir: Path,
    patchtst_output_dir: Path,
    rf_summaries: List[Dict],
) -> pd.DataFrame:
    """Load only the intended model results; explicitly discard the OLD PatchTST run."""
    base_summary_path = existing_output_dir / "results" / "summary_results.csv"
    patch_summary_path = patchtst_output_dir / "results" / "summary_results_patchtst.csv"

    if not base_summary_path.exists():
        raise FileNotFoundError(f"LightGBM/LSTM 요약 결과가 없습니다: {base_summary_path}")
    if not patch_summary_path.exists():
        raise FileNotFoundError(f"새 공식 PatchTST 요약 결과가 없습니다: {patch_summary_path}")

    base = pd.read_csv(base_summary_path, encoding="utf-8-sig")
    base = base[base["model"].isin(EXPECTED_BASE_MODELS)].copy()
    missing_base = sorted(set(EXPECTED_BASE_MODELS) - set(base["model"]))
    if missing_base:
        raise ValueError(f"기존 요약 결과에 모델이 없습니다: {missing_base}")

    patch = pd.read_csv(patch_summary_path, encoding="utf-8-sig")
    patch = patch[patch["model"] == PATCHTST_MODEL_NAME].copy()
    if patch.empty:
        raise ValueError("새 PatchTST 요약 결과에 PatchTST 행이 없습니다.")

    rf_df = pd.DataFrame(rf_summaries)
    if rf_df.empty:
        raise ValueError("Random Forest 요약 결과가 비어 있습니다.")

    combined = pd.concat([base, patch, rf_df], ignore_index=True, sort=False)
    expected = set(EXPECTED_BASE_MODELS + [PATCHTST_MODEL_NAME, RF_MODEL_NAME])
    found = set(combined["model"])
    missing = expected - found
    if missing:
        raise ValueError(f"4개 모델 요약 결과가 완전하지 않습니다. 누락={sorted(missing)}")
    return combined


def _load_prediction_for_row(
    model_name: str,
    case_name: str,
    rf_predictions: Dict[Tuple[str, str], pd.DataFrame],
    existing_output_dir: Path,
    patchtst_output_dir: Path,
) -> pd.DataFrame:
    key = (model_name, case_name)
    if model_name == RF_MODEL_NAME:
        if key in rf_predictions:
            return rf_predictions[key].copy()
        path = (
            Path("pv_training_outputs_rf")
            / "predictions"
            / prediction_filename(model_name, case_name)
        )
        # The caller normally supplies in-memory RF predictions. This fallback is
        # overwritten by build_saved_four_model_comparisons in comparison-only mode.
        return read_prediction_csv(path, model_name, case_name)

    if model_name == PATCHTST_MODEL_NAME:
        root = patchtst_output_dir
    elif model_name in EXPECTED_BASE_MODELS:
        root = existing_output_dir
    else:
        raise ValueError(f"지원하지 않는 모델명입니다: {model_name}")

    path = root / "predictions" / prediction_filename(model_name, case_name)
    return read_prediction_csv(path, model_name, case_name)


def _evaluate_selected_rows(
    selected_rows: pd.DataFrame,
    rf_predictions: Dict[Tuple[str, str], pd.DataFrame],
    existing_output_dir: Path,
    patchtst_output_dir: Path,
    output_dir: Path,
    tag: str,
) -> None:
    expected_all = set(EXPECTED_BASE_MODELS + [PATCHTST_MODEL_NAME, RF_MODEL_NAME])
    found_all = set(selected_rows["model"])
    if found_all != expected_all:
        raise ValueError(
            f"4개 모델 선택에 실패했습니다. 누락={sorted(expected_all - found_all)}"
        )

    selected_predictions: Dict[Tuple[str, str], pd.DataFrame] = {}
    for _, row in selected_rows.iterrows():
        model_name = str(row["model"])
        case_name = str(row["case"])
        key = (model_name, case_name)

        if model_name == RF_MODEL_NAME:
            if key not in rf_predictions:
                raise KeyError(f"RF 예측값을 찾을 수 없습니다: {key}")
            selected_predictions[key] = rf_predictions[key].copy()
        else:
            root = patchtst_output_dir if model_name == PATCHTST_MODEL_NAME else existing_output_dir
            path = root / "predictions" / prediction_filename(model_name, case_name)
            selected_predictions[key] = read_prediction_csv(path, model_name, case_name)

    time_sets = [set(pd.to_datetime(p[DATE_COL])) for p in selected_predictions.values()]
    common_times = sorted(set.intersection(*time_sets))
    if not common_times:
        raise ValueError(f"{tag}: 4개 모델에 공통으로 존재하는 Test timestamp가 없습니다.")

    assert_same_ground_truth(selected_predictions, common_times)

    metric_rows = []
    common_predictions = []
    for (model_name, case_name), pred in selected_predictions.items():
        common = pred[pred[DATE_COL].isin(common_times)].copy().sort_values(DATE_COL)
        metrics = evaluate_prediction_df(common)
        metric_rows.append(
            {
                "model": model_name,
                "case": case_name,
                "common_test_timestamps": len(common_times),
                **metrics,
            }
        )
        common_predictions.append(common)

    results_dir = output_dir / "results"
    ensure_dir(results_dir)

    selected_rows.to_csv(
        results_dir / f"selected_cases_{tag}_4models.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(
        results_dir / f"common_timestamp_metrics_{tag}_4models.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(common_predictions, ignore_index=True).to_csv(
        results_dir / f"common_timestamp_predictions_{tag}_4models.csv",
        index=False,
        encoding="utf-8-sig",
    )

    main_cols = [
        "model", "case", "DAY_n_samples", "DAY_MAE_MWh", "DAY_RMSE_MWh",
        "DAY_NMAE_pct", "DAY_NRMSE_pct", "DAY_MBE_MWh", "DAY_R2",
    ]
    metric_df[main_cols].to_csv(
        results_dir / f"MAIN_{tag}_common_timestamp_daytime_comparison_4models.csv",
        index=False,
        encoding="utf-8-sig",
    )


def build_saved_four_model_comparisons(
    rf_summaries: List[Dict],
    rf_predictions: Dict[Tuple[str, str], pd.DataFrame],
    existing_output_dir: Path,
    patchtst_output_dir: Path,
    output_dir: Path,
) -> None:
    """Build PRIMARY C6_Full and SECONDARY validation-best four-model comparisons."""
    combined = _load_model_summaries(
        existing_output_dir=existing_output_dir,
        patchtst_output_dir=patchtst_output_dir,
        rf_summaries=rf_summaries,
    )

    # PRIMARY: same C6_Full / 69 feature identities across all four models.
    c6 = combined[combined["case"] == "C6_Full"].copy()
    expected = set(EXPECTED_BASE_MODELS + [PATCHTST_MODEL_NAME, RF_MODEL_NAME])
    if set(c6["model"]) != expected or c6["model"].duplicated().any():
        raise ValueError(
            "C6_Full 4모델 비교에 필요한 행이 정확히 1개씩 존재하지 않습니다. "
            "각 모델의 C6_Full 학습 결과를 확인하세요."
        )
    c6 = c6.sort_values("model").reset_index(drop=True)
    _evaluate_selected_rows(
        c6, rf_predictions, existing_output_dir, patchtst_output_dir, output_dir,
        tag="C6_Full",
    )

    # SECONDARY: each model's best case chosen by Validation DAY_NMAE only.
    best = select_best_case_rows(combined)
    _evaluate_selected_rows(
        best, rf_predictions, existing_output_dir, patchtst_output_dir, output_dir,
        tag="BestCase",
    )

    # Backward-compatible copy of the former main table name = BestCase comparison.
    src = output_dir / "results" / "MAIN_BestCase_common_timestamp_daytime_comparison_4models.csv"
    dst = output_dir / "results" / "MAIN_common_timestamp_daytime_comparison_4models.csv"
    if src.exists():
        shutil.copyfile(src, dst)


# ============================================================
# 4. Main
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Random Forest-only PV forecasting experiment"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="chungbuk_pv_features.csv",
        help="Gap-safe feature CSV path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pv_training_outputs_rf",
        help="RF 전용 출력 폴더(기존 출력 폴더와 다르게 지정)",
    )
    parser.add_argument(
        "--existing_output_dir",
        type=str,
        default="pv_training_outputs",
        help="기존 LightGBM/LSTM 결과 폴더(구형 PatchTST 결과는 사용하지 않음)",
    )
    parser.add_argument(
        "--patchtst_output_dir",
        type=str,
        default="pv_training_outputs_patchtst",
        help="새 공식 PatchTSTForRegression 결과 폴더",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Optuna trials per case; 기존 실험과 동일하게 지정",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Random Forest CPU 병렬 작업 수(-1: 전체 코어)",
    )
    parser.add_argument(
        "--no_clip", action="store_true", help="음수 예측값을 0으로 보정하지 않음"
    )
    parser.add_argument(
        "--no_merge_existing",
        action="store_true",
        default=True,  # IDE에서 실행할 때 기존 모델 비교 건너뛰기
        help="기존 3개 모델과의 공통 timestamp 비교를 수행하지 않음",
    )
    parser.add_argument(
        "--comparison_only",
        action="store_true",
        help="RF를 재학습하지 않고 저장된 RF + LightGBM/LSTM + 새 PatchTST 결과로 비교표만 재생성",
    )
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="코드 점검용으로 case별 trial을 1회만 수행",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_trials < 1:
        raise ValueError("--n_trials은 1 이상이어야 합니다.")

    set_seed(args.seed)
    if optuna is None:
        raise ImportError("optuna가 설치되어 있지 않습니다: pip install optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"입력 CSV가 없습니다: {csv_path}")

    output_dir = Path(args.output_dir)
    existing_output_dir = Path(args.existing_output_dir)
    patchtst_output_dir = Path(args.patchtst_output_dir)
    if output_dir.resolve() == existing_output_dir.resolve():
        raise ValueError(
            "--output_dir은 기존 결과 보호를 위해 --existing_output_dir과 달라야 합니다."
        )
    if patchtst_output_dir.resolve() == existing_output_dir.resolve():
        raise ValueError(
            "--patchtst_output_dir은 구형 PatchTST 오사용 방지를 위해 --existing_output_dir과 달라야 합니다."
        )
    ensure_dir(output_dir / "results")

    if args.quick_test and not args.comparison_only:
        args.n_trials = 1
        if not args.no_merge_existing:
            args.no_merge_existing = True
            print("[Quick test] 기존 결과와의 병합은 건너뜁니다.")
        print("[Quick test] case별 Optuna trial을 1회만 수행합니다.")

    clip_predictions = not args.no_clip
    df = read_feature_csv(str(csv_path))
    groups = build_feature_groups(df)
    cases = build_feature_cases(groups)
    train_df, valid_df, test_df = split_by_year(df)

    config = {
        "script": "04_exp1_train_random_forest.py",
        "input_csv": str(csv_path),
        "input_csv_sha256": file_sha256(csv_path),
        "output_dir": str(output_dir),
        "model": RF_MODEL_NAME,
        "target": TARGET_COL,
        "excluded_features": EXCLUDE_FEATURE_COLS,
        "feature_groups": groups,
        "feature_cases": cases,
        "split": {"train": "2020-2022", "valid": "2023", "test": "2024"},
        "main_result": f"daytime where {DAYTIME_COL} > 0",
        "hyperparameter_selection": "minimum validation DAY_NMAE_pct",
        "test_usage": "final evaluation only",
        "n_trials": args.n_trials,
        "seed": args.seed,
        "n_jobs": args.n_jobs,
        "clip_predictions": clip_predictions,
        "rf_search_space": {
            "n_estimators": "300..1500, step 100",
            "max_depth": [None, 8, 12, 16, 20, 30],
            "min_samples_split": "2..20",
            "min_samples_leaf": "1..10",
            "max_features": ["sqrt", "log2", 0.5, 0.7, 1.0],
            "bootstrap": True,
            "max_samples": [0.7, 0.85, 1.0],
            "criterion": "squared_error",
        },
    }
    save_json(config, output_dir / "experiment_config_random_forest.json")

    if not args.no_merge_existing:
        protocol_check = validate_existing_protocol(
            existing_output_dir, patchtst_output_dir, config
        )
        save_json(protocol_check, output_dir / "protocol_check.json")
        print("[조건 검증] LightGBM/LSTM 및 새 공식 PatchTST와 핵심 프로토콜이 일치합니다.")

    print("=" * 80)
    print("[Random Forest 전용 실험]")
    print("Train:", train_df[DATE_COL].min(), "~", train_df[DATE_COL].max(), len(train_df))
    print("Valid:", valid_df[DATE_COL].min(), "~", valid_df[DATE_COL].max(), len(valid_df))
    print("Test :", test_df[DATE_COL].min(), "~", test_df[DATE_COL].max(), len(test_df))
    print("Optuna trials per case:", args.n_trials)
    print("Seed:", args.seed)

    summaries: List[Dict] = []
    rf_predictions: Dict[Tuple[str, str], pd.DataFrame] = {}

    if args.comparison_only:
        summary_path = output_dir / "results" / "summary_results_random_forest.csv"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"--comparison_only에 필요한 RF 요약 결과가 없습니다: {summary_path}"
            )
        summary_df = pd.read_csv(summary_path, encoding="utf-8-sig")
        summaries = summary_df.to_dict("records")
        for case_name in summary_df["case"].astype(str):
            path = output_dir / "predictions" / prediction_filename(RF_MODEL_NAME, case_name)
            rf_predictions[(RF_MODEL_NAME, case_name)] = read_prediction_csv(
                path, RF_MODEL_NAME, case_name
            )
        print("[Comparison only] 저장된 Random Forest 결과를 불러왔습니다.")
    else:
        for case_name, feature_cols in cases.items():
            print("\n" + "=" * 80)
            print(f"[RandomForest] {case_name} / feature_count={len(feature_cols)}")
            summary, pred_df = train_random_forest_case(
                case_name=case_name,
                feature_cols=feature_cols,
                train_df=train_df,
                valid_df=valid_df,
                test_df=test_df,
                output_dir=output_dir,
                n_trials=args.n_trials,
                seed=args.seed,
                n_jobs=args.n_jobs,
                clip_predictions=clip_predictions,
            )
            summaries.append(summary)
            rf_predictions[(RF_MODEL_NAME, case_name)] = pred_df
            pd.DataFrame(summaries).to_csv(
                output_dir / "results" / "summary_results_random_forest.csv",
                index=False,
                encoding="utf-8-sig",
            )

        summary_df = pd.DataFrame(summaries)
        best_rf = select_best_case_rows(summary_df)
        best_rf.to_csv(
            output_dir / "results" / "best_case_random_forest_validation.csv",
            index=False,
            encoding="utf-8-sig",
        )
        main_cols = [
            "model", "case", "feature_count",
            "TEST_DAY_n_samples", "TEST_DAY_MAE_MWh", "TEST_DAY_RMSE_MWh",
            "TEST_DAY_NMAE_pct", "TEST_DAY_NRMSE_pct", "TEST_DAY_MBE_MWh",
            "TEST_DAY_R2", "model_path",
        ]
        summary_df[main_cols].to_csv(
            output_dir / "results" / "MAIN_individual_daytime_results_random_forest.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if not args.no_merge_existing:
        build_saved_four_model_comparisons(
            rf_summaries=summaries,
            rf_predictions=rf_predictions,
            existing_output_dir=existing_output_dir,
            patchtst_output_dir=patchtst_output_dir,
            output_dir=output_dir,
        )
        print("[4모델 비교] C6_Full 고정 비교와 Validation Best Case 비교를 모두 재생성했습니다.")

    print("\n" + "=" * 80)
    print("[완료]")
    print("RF 전용 결과 폴더:", output_dir)
    print("RF 모델:", output_dir / "models" / "random_forest")
    print("성능표:", output_dir / "results")


if __name__ == "__main__":
    main()
