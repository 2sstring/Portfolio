# -*- coding: utf-8 -*-
"""
Experiment 4: Cross-model transfer of each model's SHAP-selected best feature subset
==================================================================================

목적
----
06_exp3_shap_topk_retrain.py에서 각 모델이 2023 Validation DAY_NMAE 기준으로 고른
최적 SHAP Top-K 특징 조합을 다른 모델에도 교차 적용하여, 특징 선택이 정말
모델별로 달라지는지 확인한다.

예:
    RF-Best features       -> RandomForest / LightGBM / LSTM
    LightGBM-Best features -> RandomForest / LightGBM / LSTM
    LSTM-Best features     -> RandomForest / LightGBM / LSTM

중요한 공정성 규칙
------------------
1) 특징 조합은 06_exp3의 Validation 결과로 이미 결정된 것만 사용한다.
2) 각 교차 조합은 기존 04 학습 함수와 동일한 Optuna/학습 프로토콜로 다시 학습한다.
3) 최종 비교는 기존 69-feature(C6_Full) 비교에서 사용한 고정 Test timestamp를 사용한다.
4) 따라서 Full69 vs SHAP-Best 및 모델 간 비교 모두 동일한 Test 시점에서 계산된다.
5) Test는 하이퍼파라미터나 특징 조합 선택에는 사용하지 않는다.

현재 LSTM Top-K가 아직 끝나지 않은 경우
----------------------------------------
- RF와 LightGBM의 BEST_K 결과만 존재해도 실행 가능하다.
- 기본 설정 ONLY_READY_TARGETS=True 이므로 자기 Best-K가 아직 없는 LSTM은 target에서도
  자동 skip된다. 따라서 현재는 RF/LGBM 2x2 교차실험만 수행할 수 있다.
- 나중에 LSTM 결과가 생성된 후 같은 코드를 다시 실행하면, 이미 끝난 조합은 skip하고
  LSTM 관련 조합만 추가하여 최종 3x3 표를 완성한다.

예상 프로젝트 구조
------------------
project/
├─ chungbuk_pv_features.csv
├─ 04_exp1_train_v1.py
├─ 04_exp1_train_random_forest.py
├─ 06_exp3_shap_topk_retrain.py
├─ 09_exp4_cross_model_best_features.py
├─ pv_exp3_1_modelwise_shap_topk/
│  └─ results/
│     ├─ BEST_K_by_validation_random_forest.csv
│     ├─ BEST_K_by_validation_lightgbm.csv
│     ├─ BEST_K_by_validation_lstm.csv              (나중에 생성 가능)
│     ├─ features_random_forest_TopK.csv
│     ├─ features_lightgbm_TopK.csv
│     └─ features_lstm_TopK.csv
└─ pv_training_outputs_final_comparison/
   ├─ common_test_timestamps_C6_Full_4models.csv
   └─ common_test_predictions_C6_Full_4models.csv

실행
----
IDE에서 그냥 Run 또는:
    python 09_exp4_cross_model_best_features.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# 0. USER SETTINGS
# =============================================================================
CONFIG = {
    # None이면 이 .py 파일이 있는 폴더를 프로젝트 루트로 사용
    "project_dir": None,

    "csv": "chungbuk_pv_features.csv",

    # 기존 학습 코드
    "base_train_script": "04_exp1_train_v1.py",                 # LightGBM + LSTM
    "rf_train_script": "04_exp1_train_random_forest.py",       # Random Forest

    # 06_exp3 결과
    "exp3_output_dir": "pv_exp3_1_modelwise_shap_topk",

    # 기존 69-feature 공통 Test 결과
    "baseline_comparison_dir": "pv_training_outputs_final_comparison",
    "reference_timestamps_file": "common_test_timestamps_C6_Full_4models.csv",
    "baseline_predictions_file": "common_test_predictions_C6_Full_4models.csv",

    # 이번 실험 출력
    "output_dir": "pv_exp4_cross_model_best_features",

    # 실험 대상 순서
    "source_models": ["random_forest", "lightgbm", "lstm"],
    "target_models": ["random_forest", "lightgbm", "lstm"],

    # 현재 LSTM Top-K가 아직 안 끝났다면:
    # True = 자기 BEST_K가 존재하는 target만 실행 -> 지금 RF/LGBM만 먼저 실행 가능
    # 나중에 LSTM BEST_K가 생기면 동일 코드 재실행으로 자동 포함
    "only_ready_targets": True,

    # 일부 source 모델 BEST_K가 아직 없어도 에러 없이 skip
    "allow_missing_source": True,

    # 이미 완료한 source->target 조합은 재학습하지 않음
    "skip_existing": True,

    # 기존 C6_Full 공통 Test timestamp가 하나라도 prediction에 없으면 오류
    # 논문용 동일 Test 보장을 위해 True 권장
    "strict_same_test": True,

    # 원래 학습 코드와 동일한 수준으로 설정
    "n_trials_lgb": 50,
    "n_trials_rf": 50,
    "n_trials_lstm": 50,
    "epochs": 200,
    "patience": 25,
    "seed": 42,
    "rf_n_jobs": -1,
    "device": "auto",
    "clip_predictions": True,
}

MODEL_ORDER = ["random_forest", "lightgbm", "lstm"]
MODEL_LABEL = {
    "random_forest": "RandomForest",
    "lightgbm": "LightGBM",
    "lstm": "LSTM",
}


# =============================================================================
# 1. Utilities
# =============================================================================
def project_root() -> Path:
    value = CONFIG["project_dir"]
    if value:
        return Path(value).expanduser().resolve()
    return Path(__file__).resolve().parent


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def resolve_training_script(root: Path, requested: str, kind: str) -> Path:
    p = resolve_path(root, requested)
    if p.is_file():
        return p

    if kind == "base":
        preferred = [
            "04_exp1_train_v1.py",
            "04_exp1_train_v1(10).py",
            "04_exp1_train(20260911-114943).py",
            "04_exp1_train.py",
        ]
        patterns = ["04_exp1_train_v1*.py", "04_exp1_train*.py"]
    elif kind == "rf":
        preferred = [
            "04_exp1_train_random_forest.py",
            "04_exp1_train_random_forest(8).py",
            "04_exp1_train_random_forest(7).py",
        ]
        patterns = ["04_exp1_train_random_forest*.py"]
    else:
        raise ValueError(kind)

    for name in preferred:
        q = root / name
        if q.is_file():
            print(f"[학습 코드 자동탐색] {kind}: {q.name}")
            return q

    matches = []
    for pattern in patterns:
        matches.extend(root.glob(pattern))
    if kind == "base":
        matches = [
            x for x in matches
            if "random_forest" not in x.name.lower()
            and "patchtst" not in x.name.lower()
        ]
    matches = sorted(set(matches), key=lambda x: x.stat().st_mtime, reverse=True)
    if matches:
        print(f"[학습 코드 자동탐색] {kind}: {matches[0].name}")
        return matches[0]

    raise FileNotFoundError(f"{kind} 학습 스크립트를 찾을 수 없습니다: {requested}")


def load_python_module(module_name: str, script_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Python 모듈로 불러올 수 없습니다: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def normalize_model_list(values: List[str]) -> List[str]:
    aliases = {
        "rf": "random_forest",
        "randomforest": "random_forest",
        "random_forest": "random_forest",
        "lgb": "lightgbm",
        "lgbm": "lightgbm",
        "lightgbm": "lightgbm",
        "lstm": "lstm",
    }
    out = []
    for v in values:
        key = str(v).strip().lower().replace("-", "_").replace(" ", "_")
        if key not in aliases:
            raise ValueError(f"지원하지 않는 모델명: {v}")
        m = aliases[key]
        if m not in out:
            out.append(m)
    return out


def slug(source: str, target: str) -> str:
    return f"src_{source}__to__{target}"


# =============================================================================
# 2. Load best feature sets from Experiment 3
# =============================================================================
def read_best_k(exp3_results: Path, model: str) -> Optional[int]:
    individual = exp3_results / f"BEST_K_by_validation_{model}.csv"
    global_file = exp3_results / "BEST_K_all_models_by_validation.csv"

    if individual.is_file():
        df = pd.read_csv(individual, encoding="utf-8-sig")
    elif global_file.is_file():
        df = pd.read_csv(global_file, encoding="utf-8-sig")
        if "model_key" in df.columns:
            df = df[df["model_key"].astype(str).str.lower() == model]
        elif "model" in df.columns:
            label = MODEL_LABEL[model].lower()
            df = df[df["model"].astype(str).str.lower() == label]
        else:
            return None
    else:
        return None

    if df.empty:
        return None

    row = df.iloc[0]
    for col in ["BEST_K", "shap_top_k", "selected_shap_top_k"]:
        if col in df.columns and pd.notna(row[col]):
            return int(row[col])
    return None


def load_best_feature_set(exp3_results: Path, model: str) -> Optional[Dict]:
    k = read_best_k(exp3_results, model)
    if k is None:
        return None

    feature_file = exp3_results / f"features_{model}_Top{k}.csv"
    if not feature_file.is_file():
        raise FileNotFoundError(
            f"{MODEL_LABEL[model]} Best-K={k}는 찾았지만 특징 파일이 없습니다: {feature_file}"
        )

    df = pd.read_csv(feature_file, encoding="utf-8-sig")
    if "feature" not in df.columns:
        raise ValueError(f"{feature_file}: 'feature' 컬럼이 없습니다.")
    features = df["feature"].dropna().astype(str).tolist()
    if len(features) != k:
        raise ValueError(
            f"{feature_file}: Best K={k}인데 feature 수={len(features)} 입니다."
        )

    return {
        "source_model": model,
        "source_label": MODEL_LABEL[model],
        "best_k": k,
        "features": features,
        "feature_file": str(feature_file),
    }


# =============================================================================
# 3. Training dispatch - reuse original 04 protocols
# =============================================================================
def resolve_device(base_mod, requested: str) -> str:
    if requested != "auto":
        return requested
    torch_obj = getattr(base_mod, "torch", None)
    if torch_obj is not None and torch_obj.cuda.is_available():
        return "cuda"
    return "cpu"


def run_target_training(
    target_model: str,
    case_name: str,
    feature_cols: List[str],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_root: Path,
    base_mod,
    rf_mod,
    device: str,
) -> Tuple[Dict, pd.DataFrame]:
    clip_predictions = bool(CONFIG["clip_predictions"])

    if target_model == "lightgbm":
        return base_mod.train_lightgbm_case(
            case_name=case_name,
            feature_cols=feature_cols,
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            output_dir=output_root,
            n_trials=int(CONFIG["n_trials_lgb"]),
            seed=int(CONFIG["seed"]),
            clip_predictions=clip_predictions,
        )

    if target_model == "random_forest":
        return rf_mod.train_random_forest_case(
            case_name=case_name,
            feature_cols=feature_cols,
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            output_dir=output_root,
            n_trials=int(CONFIG["n_trials_rf"]),
            seed=int(CONFIG["seed"]),
            n_jobs=int(CONFIG["rf_n_jobs"]),
            clip_predictions=clip_predictions,
        )

    if target_model == "lstm":
        return base_mod.train_sequence_case(
            model_name="LSTM",
            case_name=case_name,
            feature_cols=feature_cols,
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            output_dir=output_root,
            n_trials=int(CONFIG["n_trials_lstm"]),
            seed=int(CONFIG["seed"]),
            max_epochs=int(CONFIG["epochs"]),
            patience=int(CONFIG["patience"]),
            device=device,
            clip_predictions=clip_predictions,
        )

    raise ValueError(target_model)


# =============================================================================
# 4. Fixed-Test evaluation
# =============================================================================
def load_reference_times(path: Path, date_col: str) -> pd.DatetimeIndex:
    if not path.is_file():
        raise FileNotFoundError(f"기준 Test timestamp 파일이 없습니다: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if date_col not in df.columns:
        # robust fallback: use first column if the known Korean column name changed
        if len(df.columns) == 1:
            col = df.columns[0]
        else:
            raise ValueError(f"{path}: 날짜 컬럼 '{date_col}'을 찾지 못했습니다.")
    else:
        col = date_col
    times = pd.DatetimeIndex(pd.to_datetime(df[col])).drop_duplicates().sort_values()
    if len(times) == 0:
        raise ValueError("기준 Test timestamp가 비어 있습니다.")
    return times


def standardize_prediction_df(
    pred: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str,
    capacity_col: str,
    daytime_col: str,
) -> pd.DataFrame:
    work = pred.copy()
    if date_col not in work.columns:
        raise ValueError(f"prediction에 날짜 컬럼이 없습니다: {date_col}")
    work[date_col] = pd.to_datetime(work[date_col])

    required_pred = ["true_MWh", "pred_MWh"]
    missing = [c for c in required_pred if c not in work.columns]
    if missing:
        raise ValueError(f"prediction에 필요한 컬럼이 없습니다: {missing}")

    # If capacity/daytime columns are absent in a returned prediction table,
    # recover them from the original 2024 Test dataframe by timestamp.
    need_merge = [c for c in [capacity_col, daytime_col] if c not in work.columns]
    if need_merge:
        merge_cols = [date_col] + need_merge
        aux = test_df[merge_cols].copy()
        aux[date_col] = pd.to_datetime(aux[date_col])
        aux = aux.drop_duplicates(subset=[date_col])
        work = work.merge(aux, on=date_col, how="left")

    return work.sort_values(date_col).reset_index(drop=True)


def evaluate_on_reference_times(
    pred: pd.DataFrame,
    reference_times: pd.DatetimeIndex,
    test_df: pd.DataFrame,
    date_col: str,
    capacity_col: str,
    daytime_col: str,
    strict: bool,
) -> Tuple[Dict, pd.DataFrame]:
    work = standardize_prediction_df(
        pred, test_df, date_col=date_col,
        capacity_col=capacity_col, daytime_col=daytime_col,
    )

    pred_times = set(work[date_col])
    ref_times = set(reference_times)
    missing = sorted(ref_times - pred_times)

    if strict and missing:
        sample = missing[:10]
        raise ValueError(
            f"기준 Test timestamp {len(reference_times)}개 중 {len(missing)}개가 prediction에 없습니다.\n"
            f"예: {sample}\n"
            "논문용 동일 Test 조건을 유지하려면 이 조합의 prediction 생성 범위를 확인하세요."
        )

    matched_times = [t for t in reference_times if t in pred_times]
    fixed = work[work[date_col].isin(matched_times)].copy().sort_values(date_col)

    # exactly one row per timestamp expected
    fixed = fixed.drop_duplicates(subset=[date_col], keep="first").reset_index(drop=True)

    true = fixed["true_MWh"].to_numpy(float)
    pred_y = fixed["pred_MWh"].to_numpy(float)
    cap = fixed[capacity_col].to_numpy(float)
    day = pd.to_numeric(fixed[daytime_col], errors="coerce").to_numpy(float) > 0

    error = pred_y - true
    safe_cap = np.where(cap == 0, np.nan, cap)

    def calc(mask: np.ndarray, prefix: str) -> Dict:
        n = int(np.sum(mask))
        if n == 0:
            return {
                f"{prefix}_n_samples": 0,
                f"{prefix}_MAE_MWh": np.nan,
                f"{prefix}_RMSE_MWh": np.nan,
                f"{prefix}_NMAE_pct": np.nan,
                f"{prefix}_NRMSE_pct": np.nan,
                f"{prefix}_MBE_MWh": np.nan,
                f"{prefix}_R2": np.nan,
            }
        e = error[mask]
        y = true[mask]
        c = safe_cap[mask]
        ss_res = float(np.sum(e ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot
        return {
            f"{prefix}_n_samples": n,
            f"{prefix}_MAE_MWh": float(np.mean(np.abs(e))),
            f"{prefix}_RMSE_MWh": float(np.sqrt(np.mean(e ** 2))),
            f"{prefix}_NMAE_pct": float(np.nanmean(np.abs(e) / c) * 100.0),
            f"{prefix}_NRMSE_pct": float(np.sqrt(np.nanmean((e / c) ** 2)) * 100.0),
            f"{prefix}_MBE_MWh": float(np.mean(e)),
            f"{prefix}_R2": float(r2),
        }

    metrics = {
        "reference_test_timestamps": int(len(reference_times)),
        "prediction_original_timestamps": int(work[date_col].nunique()),
        "matched_test_timestamps": int(len(fixed)),
        "missing_reference_timestamps": int(len(missing)),
    }
    metrics.update(calc(np.ones(len(fixed), dtype=bool), "ALL"))
    metrics.update(calc(day, "DAY"))
    return metrics, fixed


# =============================================================================
# 5. Existing result reuse
# =============================================================================
def combo_result_paths(output_dir: Path, source: str, target: str) -> Dict[str, Path]:
    key = slug(source, target)
    return {
        "summary": output_dir / "results" / f"SUMMARY_{key}.csv",
        "prediction_raw": output_dir / "predictions_raw" / f"TEST_{key}.csv",
        "prediction_fixed": output_dir / "predictions_fixed_test" / f"TEST_FIXED_{key}.csv",
        "metrics_fixed": output_dir / "results" / f"METRICS_FIXED_{key}.csv",
    }


def combo_is_complete(paths: Dict[str, Path]) -> bool:
    return paths["summary"].is_file() and paths["prediction_raw"].is_file()


# =============================================================================
# 6. Baseline Full69 evaluation on exact same reference Test
# =============================================================================
def load_baseline_predictions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"기존 C6_Full prediction 파일이 없습니다: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "model" not in df.columns:
        raise ValueError(f"{path}: 'model' 컬럼이 없습니다.")
    return df


def baseline_model_mask(series: pd.Series, model: str) -> pd.Series:
    normalized = series.astype(str).str.lower().str.replace("_", "", regex=False).str.replace(" ", "", regex=False)
    wanted = {
        "random_forest": "randomforest",
        "lightgbm": "lightgbm",
        "lstm": "lstm",
    }[model]
    return normalized == wanted


# =============================================================================
# 7. Main
# =============================================================================
def main() -> None:
    root = project_root()
    csv_path = resolve_path(root, CONFIG["csv"])
    exp3_results = resolve_path(root, CONFIG["exp3_output_dir"]) / "results"
    baseline_dir = resolve_path(root, CONFIG["baseline_comparison_dir"])
    reference_path = baseline_dir / CONFIG["reference_timestamps_file"]
    baseline_pred_path = baseline_dir / CONFIG["baseline_predictions_file"]
    output_dir = resolve_path(root, CONFIG["output_dir"])

    ensure_dir(output_dir / "results")
    ensure_dir(output_dir / "predictions_raw")
    ensure_dir(output_dir / "predictions_fixed_test")
    ensure_dir(output_dir / "feature_sets")

    if not csv_path.is_file():
        raise FileNotFoundError(f"feature CSV가 없습니다: {csv_path}")
    if not exp3_results.is_dir():
        raise FileNotFoundError(f"06_exp3 results 폴더가 없습니다: {exp3_results}")

    source_models = normalize_model_list(CONFIG["source_models"])
    target_models_requested = normalize_model_list(CONFIG["target_models"])

    # -------------------------------------------------------------------------
    # Load available model-specific Best feature sets
    # -------------------------------------------------------------------------
    feature_sets: Dict[str, Dict] = {}
    source_status = []
    for source in source_models:
        info = load_best_feature_set(exp3_results, source)
        if info is None:
            source_status.append({
                "source_model": source,
                "source_label": MODEL_LABEL[source],
                "status": "BEST_K_not_ready",
            })
            if not CONFIG["allow_missing_source"]:
                raise FileNotFoundError(f"{MODEL_LABEL[source]} BEST_K 결과가 아직 없습니다.")
            continue
        feature_sets[source] = info
        source_status.append({
            "source_model": source,
            "source_label": MODEL_LABEL[source],
            "status": "ready",
            "best_k": info["best_k"],
            "feature_file": info["feature_file"],
        })
        pd.DataFrame({
            "rank": np.arange(1, info["best_k"] + 1),
            "feature": info["features"],
        }).to_csv(
            output_dir / "feature_sets" / f"BEST_FEATURES_{source}_Top{info['best_k']}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    pd.DataFrame(source_status).to_csv(
        output_dir / "results" / "SOURCE_BEST_FEATURE_STATUS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not feature_sets:
        raise RuntimeError("사용 가능한 모델별 Best-K 특징 조합이 하나도 없습니다.")

    # A target is considered 'ready' when its own Exp3 best feature set exists.
    # This lets the script run now with RF/LGBM only, then expand automatically later.
    if CONFIG["only_ready_targets"]:
        target_models = [m for m in target_models_requested if m in feature_sets]
    else:
        target_models = target_models_requested

    if not target_models:
        raise RuntimeError("실행 가능한 target model이 없습니다.")

    # -------------------------------------------------------------------------
    # Load training modules and data
    # -------------------------------------------------------------------------
    need_base = any(m in target_models for m in ["lightgbm", "lstm"])
    need_rf = "random_forest" in target_models

    base_script = resolve_training_script(root, CONFIG["base_train_script"], "base") if need_base else None
    rf_script = resolve_training_script(root, CONFIG["rf_train_script"], "rf") if need_rf else None

    base_mod = load_python_module("exp4_base_train", base_script) if base_script else None
    rf_mod = load_python_module("exp4_rf_train", rf_script) if rf_script else None
    data_mod = base_mod or rf_mod
    if data_mod is None:
        raise RuntimeError("학습 모듈을 불러오지 못했습니다.")

    df = data_mod.read_feature_csv(str(csv_path))
    train_df, valid_df, test_df = data_mod.split_by_year(df)

    date_col = data_mod.DATE_COL
    capacity_col = data_mod.CAPACITY_COL
    daytime_col = data_mod.DAYTIME_COL

    # Validate selected feature names against the current feature CSV.
    for source, info in feature_sets.items():
        missing_cols = [c for c in info["features"] if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"{MODEL_LABEL[source]} Best feature 중 현재 CSV에 없는 컬럼이 있습니다: {missing_cols[:10]}"
            )

    device = resolve_device(base_mod, CONFIG["device"]) if base_mod is not None else "cpu"
    reference_times = load_reference_times(reference_path, date_col)
    baseline_pred_all = load_baseline_predictions(baseline_pred_path)

    # -------------------------------------------------------------------------
    # Experiment configuration log
    # -------------------------------------------------------------------------
    config_log = {
        "experiment": "Cross-model transfer of model-specific SHAP Best-K feature subsets",
        "source_models_requested": source_models,
        "source_models_available": list(feature_sets.keys()),
        "target_models_requested": target_models_requested,
        "target_models_run_now": target_models,
        "only_ready_targets": CONFIG["only_ready_targets"],
        "reference_test_timestamp_file": str(reference_path),
        "reference_test_timestamp_count": len(reference_times),
        "baseline_predictions_file": str(baseline_pred_path),
        "fixed_test_rule": "use exact C6_Full common test timestamps; no test-based selection",
        "split": {"train": "2020-2022", "validation": "2023", "test": "2024"},
        "selection_origin": "each source feature subset was selected previously by source-model Validation DAY_NMAE",
        "n_trials": {
            "random_forest": CONFIG["n_trials_rf"],
            "lightgbm": CONFIG["n_trials_lgb"],
            "lstm": CONFIG["n_trials_lstm"],
        },
        "epochs": CONFIG["epochs"],
        "patience": CONFIG["patience"],
        "seed": CONFIG["seed"],
        "device": device,
        "strict_same_test": CONFIG["strict_same_test"],
        "skip_existing": CONFIG["skip_existing"],
        "feature_sets": {
            source: {
                "best_k": info["best_k"],
                "feature_file": info["feature_file"],
                "features": info["features"],
            }
            for source, info in feature_sets.items()
        },
    }
    save_json(config_log, output_dir / "experiment_config.json")

    print("=" * 110)
    print("[Experiment 4] Cross-model transfer of SHAP Best-K feature subsets")
    print("Available source feature sets:")
    for source in MODEL_ORDER:
        if source in feature_sets:
            print(f"  - {MODEL_LABEL[source]}: Top{feature_sets[source]['best_k']}")
        else:
            print(f"  - {MODEL_LABEL[source]}: NOT READY -> skip")
    print("Target models run now:", [MODEL_LABEL[m] for m in target_models])
    print("Fixed reference Test timestamps:", len(reference_times), "|", reference_path)
    print("Device:", device)
    print("=" * 110)

    # -------------------------------------------------------------------------
    # First, evaluate Full69 baseline on the SAME fixed Test timestamps.
    # Only current ready targets are included.
    # -------------------------------------------------------------------------
    metric_rows: List[Dict] = []
    audit_rows: List[Dict] = []

    for target in target_models:
        base_pred = baseline_pred_all[baseline_model_mask(baseline_pred_all["model"], target)].copy()
        if base_pred.empty:
            raise ValueError(f"C6_Full baseline prediction에 {MODEL_LABEL[target]}이 없습니다.")

        metrics, fixed = evaluate_on_reference_times(
            base_pred,
            reference_times,
            test_df,
            date_col,
            capacity_col,
            daytime_col,
            strict=bool(CONFIG["strict_same_test"]),
        )
        row = {
            "target_model_key": target,
            "target_model": MODEL_LABEL[target],
            "feature_source_key": "full69",
            "feature_source": "Full69",
            "feature_count": 69,
            "is_own_feature_set": False,
            "is_full69_baseline": True,
            **metrics,
        }
        metric_rows.append(row)
        audit_rows.append(row.copy())

    # -------------------------------------------------------------------------
    # Cross training: each available source Best-K -> each ready target model
    # -------------------------------------------------------------------------
    for source in source_models:
        if source not in feature_sets:
            continue

        info = feature_sets[source]
        k = int(info["best_k"])
        features = list(info["features"])

        for target in target_models:
            key = slug(source, target)
            case_name = f"XFER_{MODEL_LABEL[source]}_Top{k}_TO_{MODEL_LABEL[target]}"
            paths = combo_result_paths(output_dir, source, target)

            print("\n" + "#" * 105)
            print(
                f"[{MODEL_LABEL[source]} Best Top{k} -> {MODEL_LABEL[target]}] "
                f"features={k}"
            )
            print("#" * 105)

            if CONFIG["skip_existing"] and combo_is_complete(paths):
                print("[SKIP EXISTING] 기존 학습 결과를 다시 사용합니다.")
                summary_df = pd.read_csv(paths["summary"], encoding="utf-8-sig")
                summary = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
                pred_raw = pd.read_csv(paths["prediction_raw"], encoding="utf-8-sig")
            else:
                combo_root = output_dir / "training_runs" / key
                ensure_dir(combo_root)

                summary, pred_raw = run_target_training(
                    target_model=target,
                    case_name=case_name,
                    feature_cols=features,
                    train_df=train_df,
                    valid_df=valid_df,
                    test_df=test_df,
                    output_root=combo_root,
                    base_mod=base_mod,
                    rf_mod=rf_mod,
                    device=device,
                )

                summary = dict(summary)
                summary.update({
                    "experiment": "cross_model_best_features",
                    "feature_source_key": source,
                    "feature_source": MODEL_LABEL[source],
                    "source_best_k": k,
                    "target_model_key": target,
                    "target_model": MODEL_LABEL[target],
                    "is_own_feature_set": bool(source == target),
                    "feature_file": info["feature_file"],
                    "test_selection_used": False,
                })
                pd.DataFrame([summary]).to_csv(paths["summary"], index=False, encoding="utf-8-sig")
                pred_raw.to_csv(paths["prediction_raw"], index=False, encoding="utf-8-sig")

            # Always recompute fixed-Test metrics from saved/raw predictions.
            metrics, fixed = evaluate_on_reference_times(
                pred_raw,
                reference_times,
                test_df,
                date_col,
                capacity_col,
                daytime_col,
                strict=bool(CONFIG["strict_same_test"]),
            )

            fixed = fixed.copy()
            fixed["target_model"] = MODEL_LABEL[target]
            fixed["feature_source"] = MODEL_LABEL[source]
            fixed["source_best_k"] = k
            fixed.to_csv(paths["prediction_fixed"], index=False, encoding="utf-8-sig")

            metric_row = {
                "target_model_key": target,
                "target_model": MODEL_LABEL[target],
                "feature_source_key": source,
                "feature_source": MODEL_LABEL[source],
                "feature_count": k,
                "source_best_k": k,
                "is_own_feature_set": bool(source == target),
                "is_full69_baseline": False,
                **metrics,
            }
            pd.DataFrame([metric_row]).to_csv(paths["metrics_fixed"], index=False, encoding="utf-8-sig")
            metric_rows.append(metric_row)
            audit_rows.append(metric_row.copy())

            print(
                f"[Fixed Test] n={metrics['ALL_n_samples']} / DAY n={metrics['DAY_n_samples']} / "
                f"DAY_NMAE={metrics['DAY_NMAE_pct']:.6f}%"
            )

    # -------------------------------------------------------------------------
    # Global result tables
    # -------------------------------------------------------------------------
    metrics_df = pd.DataFrame(metric_rows)

    # Stable ordering: targets RF -> LGBM -> LSTM;
    # feature sources Full69 -> RF -> LGBM -> LSTM
    target_order = {m: i for i, m in enumerate(MODEL_ORDER)}
    source_order_list = ["full69"] + MODEL_ORDER
    source_order = {m: i for i, m in enumerate(source_order_list)}
    metrics_df["_target_order"] = metrics_df["target_model_key"].map(target_order)
    metrics_df["_source_order"] = metrics_df["feature_source_key"].map(source_order)
    metrics_df = metrics_df.sort_values(["_target_order", "_source_order"]).drop(
        columns=["_target_order", "_source_order"]
    )

    all_metrics_path = output_dir / "results" / "FINAL_FIXED_TEST_all_cross_combinations.csv"
    metrics_df.to_csv(all_metrics_path, index=False, encoding="utf-8-sig")

    # DAY_NMAE matrix
    nmae_matrix = metrics_df.pivot_table(
        index="feature_source",
        columns="target_model",
        values="DAY_NMAE_pct",
        aggfunc="first",
    )
    desired_rows = ["Full69", "RandomForest", "LightGBM", "LSTM"]
    desired_cols = ["RandomForest", "LightGBM", "LSTM"]
    nmae_matrix = nmae_matrix.reindex(
        index=[x for x in desired_rows if x in nmae_matrix.index],
        columns=[x for x in desired_cols if x in nmae_matrix.columns],
    )
    nmae_matrix.to_csv(
        output_dir / "results" / "MATRIX_DAY_NMAE_feature_source_x_target_model.csv",
        encoding="utf-8-sig",
    )

    # Improvement versus Full69 for each target: positive = lower/better NMAE than Full69
    delta_matrix = nmae_matrix.copy()
    if "Full69" in nmae_matrix.index:
        baseline_row = nmae_matrix.loc["Full69"]
        for row_name in delta_matrix.index:
            delta_matrix.loc[row_name] = baseline_row - nmae_matrix.loc[row_name]
    delta_matrix.to_csv(
        output_dir / "results" / "MATRIX_DAY_NMAE_improvement_vs_Full69_pctpoint.csv",
        encoding="utf-8-sig",
    )

    # Own vs transferred feature-set comparison table (one row per target)
    compare_rows = []
    for target in target_models:
        target_label = MODEL_LABEL[target]
        sub = metrics_df[metrics_df["target_model_key"] == target].copy()
        row = {"target_model": target_label}
        for _, r in sub.iterrows():
            src = r["feature_source"]
            row[f"DAY_NMAE__{src}"] = r["DAY_NMAE_pct"]
            row[f"feature_count__{src}"] = r["feature_count"]
        compare_rows.append(row)
    pd.DataFrame(compare_rows).to_csv(
        output_dir / "results" / "COMPARE_own_vs_transferred_feature_sets.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Test sample audit - very important for paper fairness
    audit_cols = [
        "target_model", "feature_source", "feature_count",
        "reference_test_timestamps", "prediction_original_timestamps",
        "matched_test_timestamps", "missing_reference_timestamps",
        "ALL_n_samples", "DAY_n_samples",
    ]
    pd.DataFrame(audit_rows)[audit_cols].to_csv(
        output_dir / "results" / "TEST_SAMPLE_AUDIT_same_fixed_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Long-form feature list used in the experiment
    feature_long = []
    for source, info in feature_sets.items():
        for rank, feat in enumerate(info["features"], start=1):
            feature_long.append({
                "feature_source_key": source,
                "feature_source": MODEL_LABEL[source],
                "best_k": info["best_k"],
                "rank": rank,
                "feature": feat,
            })
    pd.DataFrame(feature_long).to_csv(
        output_dir / "results" / "FEATURE_SETS_USED_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 110)
    print("[완료]")
    print("이번 실행 target models:", [MODEL_LABEL[m] for m in target_models])
    print("이번 실행 source feature sets:", [MODEL_LABEL[m] for m in feature_sets])
    print("고정 Test timestamp 수:", len(reference_times))
    print("\n주요 결과:")
    print("  -", all_metrics_path)
    print("  -", output_dir / "results" / "MATRIX_DAY_NMAE_feature_source_x_target_model.csv")
    print("  -", output_dir / "results" / "MATRIX_DAY_NMAE_improvement_vs_Full69_pctpoint.csv")
    print("  -", output_dir / "results" / "COMPARE_own_vs_transferred_feature_sets.csv")
    print("  -", output_dir / "results" / "TEST_SAMPLE_AUDIT_same_fixed_test.csv")
    print("\n현재 LSTM이 미완료라면 RF/LGBM 결과만 먼저 생성됩니다.")
    print("LSTM BEST_K가 생성된 뒤 같은 코드를 다시 실행하면 기존 조합은 skip하고 LSTM 관련 조합이 추가됩니다.")
    print("=" * 110)


if __name__ == "__main__":
    main()
