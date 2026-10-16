# -*- coding: utf-8 -*-
"""
Experiment 5: Model-specific SHAP-guided SFFS feature selection
================================================================

연구 목적
---------
각 모델(Random Forest, LightGBM, LSTM)에 대해

    Full69 모델의 model-specific SHAP ranking
        -> SHAP Top-N 후보군(기본 Top20)
        -> SFFS(Sequential Floating Forward Selection)
        -> 2023 Validation DAY_NMAE 기준 최종 특징부분집합 선택
        -> 선택된 subset만 기존 Optuna search space로 50 trials 재튜닝
        -> 2024 Test 최종평가
        -> 기존 C6_Full과 동일한 고정 Test timestamp에서 성능 재계산

핵심 방법론
-----------
1) 후보군은 각 모델 자신의 Full69 SHAP Top20을 사용합니다.
2) SFFS 중에는 Full69에서 얻은 해당 모델의 best hyperparameters를 고정합니다.
3) SFFS 후보 subset은 모두 같은 Train/Validation sample을 사용하도록,
   각 모델의 Top20 전체 후보 특징이 모두 유효한 행만 screening에 사용합니다.
   (subset마다 결측치 때문에 평가 표본이 달라지는 것을 방지)
4) SFFS의 평가 기준은 2023 Validation DAY_NMAE이며 2024 Test는 사용하지 않습니다.
5) 최종 특징 수는 미리 고정하지 않습니다. 1~TopN 범위에서 SFFS가 찾은
   cardinality별 best subset 중 Validation DAY_NMAE가 가장 낮은 subset을 선택합니다.
6) SFFS는 exhaustive search가 아닙니다. Forward inclusion + conditional exclusion을
   반복하는 floating search입니다.
7) SFFS 최종 subset 결정 후에만 기존 04 학습 함수를 이용하여 Optuna 50 trials를
   다시 수행하고, 실제 2024 Test를 최종 평가합니다.
8) 최종 성능은 기존 C6_Full 공통 Test timestamp와 정확히 맞춥니다.

실행 순서
---------
RandomForest COMPLETE -> LightGBM COMPLETE -> LSTM COMPLETE
(LSTM이 오래 걸리므로 앞 모델 결과를 먼저 확인할 수 있도록 고정)

중단/재실행
-----------
- SFFS에서 평가한 모든 feature subset은 cache CSV에 즉시 저장합니다.
- 같은 subset은 다시 학습하지 않습니다.
- 중간에 프로그램이 중단되어도 다시 실행하면 cache를 재사용하여 이어갑니다.

필요 파일/폴더 (기본값)
-----------------------
- chungbuk_pv_features.csv
- 04_exp1_train_v1.py
- 04_exp1_train_random_forest.py
- pv_shap_lgbm_rf_lstm_outputs/
    random_forest/shap_feature_importance.csv
    lightgbm/shap_feature_importance.csv
    lstm/shap_feature_importance.csv
- Full69 saved models (best_params 읽기용)
    pv_training_outputs_rf/models/random_forest/C6_Full/model.pkl
    pv_training_outputs/models/lightgbm/C6_Full/model.pkl
    pv_training_outputs/models/lstm/C6_Full/model.pt
- pv_training_outputs_final_comparison/
    common_test_timestamps_C6_Full_3models.csv

주요 출력
---------
pv_exp5_shap_sffs/
  results/
    features_sffs_random_forest.csv
    features_sffs_lightgbm.csv
    features_sffs_lstm.csv
    SFFS_SELECTED_summary_<model>.csv
    SFFS_SELECTED_all_models.csv
    FINAL_TEST_SFFS_<model>.csv
    FINAL_TEST_SFFS_all_models.csv
  sffs/<model>/
    candidate_SHAP_Top20.csv
    subset_score_cache.csv
    sffs_trace.csv
    best_subset_by_size.csv
    full69_fixed_params.json
    screening_sample_audit.csv
  final_tuned/
    ... 기존 04 학습 함수 출력 ...
  fixed_test_comparison/
    MAIN_SHAP_SFFS_same_C6Full_test_metrics.csv
    PREDICTIONS_SHAP_SFFS_same_C6Full_test.csv
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, FrozenSet

import joblib
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import torch
except Exception:
    torch = None

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 0. USER SETTINGS
# =============================================================================
CSV_PATH = "chungbuk_pv_features.csv"
SHAP_ROOT = "pv_shap_lgbm_rf_lstm_outputs"
OUTPUT_DIR = "pv_exp5_shap_sffs"

BASE_TRAIN_SCRIPT = "04_exp1_train_v1.py"
RF_TRAIN_SCRIPT = "04_exp1_train_random_forest.py"

# Full69 saved models: SFFS 중 고정할 best hyperparameters를 읽습니다.
FULL_LGB_MODEL = "pv_training_outputs/models/lightgbm/C6_Full/model.pkl"
FULL_LSTM_MODEL = "pv_training_outputs/models/lstm/C6_Full/model.pt"
FULL_RF_MODEL = "pv_training_outputs_rf/models/random_forest/C6_Full/model.pkl"

FIXED_TEST_TIMESTAMP_FILE = (
    "pv_training_outputs_final_comparison/"
    "common_test_timestamps_C6_Full_3models.csv"
)

# 연구 기본 조건
DEFAULT_CANDIDATE_TOP_N = 20
DEFAULT_N_TRIALS_LGB = 50
DEFAULT_N_TRIALS_RF = 50
DEFAULT_N_TRIALS_LSTM = 50
DEFAULT_EPOCHS = 200
DEFAULT_PATIENCE = 25
DEFAULT_SEED = 42
DEFAULT_RF_N_JOBS = -1
DEFAULT_MODELS = "random_forest,lightgbm,lstm"

# SFFS 조건
SFFS_SCORE_TOL = 1e-12
MAX_UNIQUE_SUBSET_EVALS = 10000  # 안전장치. 정상적인 Top20 SFFS에서는 보통 훨씬 적음.
STRICT_SAME_TEST = True
DEFAULT_SKIP_EXISTING = True

MODEL_ORDER = ["random_forest", "lightgbm", "lstm"]
DISPLAY = {
    "random_forest": "RandomForest",
    "lightgbm": "LightGBM",
    "lstm": "LSTM",
}
PRED_TOKEN = {
    "random_forest": "randomforest",
    "lightgbm": "lightgbm",
    "lstm": "lstm",
}


# =============================================================================
# 1. Generic utilities
# =============================================================================
def project_root() -> Path:
    return Path(__file__).resolve().parent


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def load_python_module(module_name: str, script_path: Path):
    if not script_path.is_file():
        raise FileNotFoundError(f"학습 코드를 찾을 수 없습니다: {script_path}")
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_training_script(root: Path, requested: str, kind: str) -> Path:
    p = root / requested
    if p.is_file():
        return p

    if kind == "base":
        matches = [
            x for x in root.glob("04_exp1_train*.py")
            if "random_forest" not in x.name.lower()
            and "patchtst" not in x.name.lower()
        ]
    elif kind == "rf":
        matches = list(root.glob("04_exp1_train_random_forest*.py"))
    else:
        raise ValueError(kind)

    if not matches:
        raise FileNotFoundError(f"{kind} 학습 스크립트를 찾을 수 없습니다.")
    matches = sorted(matches, key=lambda x: x.stat().st_mtime, reverse=True)
    print(f"[학습 코드 자동탐색] {kind}: {matches[0].name}")
    return matches[0]


def resolve_device(requested: str, base_mod) -> str:
    if requested != "auto":
        return requested
    torch_obj = getattr(base_mod, "torch", None)
    if torch_obj is not None and torch_obj.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_models(text: str) -> List[str]:
    aliases = {
        "rf": "random_forest",
        "randomforest": "random_forest",
        "random_forest": "random_forest",
        "lgb": "lightgbm",
        "lgbm": "lightgbm",
        "lightgbm": "lightgbm",
        "lstm": "lstm",
    }
    requested = []
    for token in text.split(","):
        x = token.strip().lower().replace("-", "_").replace(" ", "_")
        if not x:
            continue
        if x not in aliases:
            raise ValueError(f"지원하지 않는 모델명: {token}")
        m = aliases[x]
        if m not in requested:
            requested.append(m)
    return [m for m in MODEL_ORDER if m in requested]


def prediction_path(output_root: Path, model: str, case_name: str) -> Path:
    return (
        output_root / "predictions"
        / f"test_predictions_{PRED_TOKEN[model]}_{case_name}.csv"
    )


def stable_hash(obj, length: int = 12) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:length]


def feature_signature(features: List[str], length: int = 10) -> str:
    return stable_hash({"features": list(features)}, length=length)


def safe_torch_load(path: Path):
    if torch is None:
        raise ImportError("PyTorch가 필요합니다.")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def ensure_shap_root(root: Path, shap_root: Path) -> Path:
    """폴더가 없고 같은 이름의 zip이 있으면 자동 압축해제."""
    if shap_root.is_dir():
        return shap_root
    zip_path = root / f"{shap_root.name}.zip"
    if zip_path.is_file():
        print(f"[SHAP] 폴더가 없어 ZIP 자동 압축해제: {zip_path.name}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)
    if not shap_root.is_dir():
        raise FileNotFoundError(
            f"SHAP 결과 폴더가 없습니다: {shap_root}\n"
            f"또는 {zip_path.name}을 프로젝트 폴더에 두세요."
        )
    return shap_root


# =============================================================================
# 2. SHAP candidates / Full69 fixed hyperparameters
# =============================================================================
def load_shap_candidates(
    shap_root: Path,
    model: str,
    top_n: int,
    allowed_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    path = shap_root / model / "shap_feature_importance.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "feature" not in df.columns:
        raise ValueError(f"{path}: feature 컬럼 없음")

    if "rank" in df.columns:
        df = df.sort_values("rank", ascending=True)
    elif "mean_abs_shap" in df.columns:
        df = df.sort_values("mean_abs_shap", ascending=False)
    elif "normalized_importance_pct" in df.columns:
        df = df.sort_values("normalized_importance_pct", ascending=False)
    else:
        raise ValueError(f"{path}: SHAP ranking 컬럼 없음")

    df = df.drop_duplicates("feature", keep="first").copy()
    if allowed_features is not None:
        allowed = set(allowed_features)
        df = df[df["feature"].astype(str).isin(allowed)].copy()

    if len(df) < top_n:
        raise ValueError(
            f"{DISPLAY[model]} SHAP 특징이 {len(df)}개뿐이라 Top{top_n}을 만들 수 없습니다."
        )

    out = df.head(top_n).reset_index(drop=True).copy()
    out["sffs_candidate_rank"] = np.arange(1, len(out) + 1)
    return out


def load_full69_bundle_and_params(root: Path, model: str, args) -> Tuple[Dict, Dict, Path]:
    if model == "random_forest":
        path = root / args.full_rf_model
        if not path.is_file():
            raise FileNotFoundError(f"RF Full69 model 없음: {path}")
        bundle = joblib.load(path)
    elif model == "lightgbm":
        path = root / args.full_lgb_model
        if not path.is_file():
            raise FileNotFoundError(f"LightGBM Full69 model 없음: {path}")
        bundle = joblib.load(path)
    elif model == "lstm":
        path = root / args.full_lstm_model
        if not path.is_file():
            raise FileNotFoundError(f"LSTM Full69 model 없음: {path}")
        bundle = safe_torch_load(path)
    else:
        raise ValueError(model)

    if not isinstance(bundle, dict):
        raise ValueError(f"{path}: 저장 모델 bundle 형식이 dict가 아닙니다.")
    if "best_params" not in bundle:
        meta_path = path.parent / "metadata.json"
        if meta_path.is_file():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if "best_params" in meta:
                bundle["best_params"] = meta["best_params"]
    if "best_params" not in bundle:
        raise ValueError(f"{path}: best_params를 찾을 수 없습니다.")

    params = dict(bundle["best_params"])
    return bundle, params, path


# =============================================================================
# 3. Fixed-hyperparameter SFFS evaluator
# =============================================================================
def prepare_screening_frames(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    candidates: List[str],
    base_mod,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    모든 후보 subset이 같은 rows/windows를 사용하도록 TopN 전체 후보가 유효한
    Train/Validation 행만 SFFS screening에 사용.
    """
    required = list(dict.fromkeys(
        candidates + [
            base_mod.TARGET_COL,
            base_mod.DATE_COL,
            base_mod.CAPACITY_COL,
            base_mod.POWER_COL,
            base_mod.DAYTIME_COL,
        ]
    ))
    if getattr(base_mod, "SEGMENT_COL", None) in train_df.columns:
        required.append(base_mod.SEGMENT_COL)
    required = list(dict.fromkeys(required))

    missing = [c for c in required if c not in train_df.columns or c not in valid_df.columns]
    if missing:
        raise ValueError(f"SFFS screening 필수 컬럼 누락: {missing}")

    tr = train_df.dropna(subset=required).copy()
    va = valid_df.dropna(subset=required).copy()

    audit = {
        "candidate_feature_count": len(candidates),
        "train_rows_original": len(train_df),
        "train_rows_candidate_complete": len(tr),
        "valid_rows_original": len(valid_df),
        "valid_rows_candidate_complete": len(va),
        "sample_policy": "all SHAP candidate features complete; same rows/windows for every SFFS subset",
    }
    return tr, va, audit


def score_from_metrics(metrics: Dict) -> float:
    s = metrics.get("DAY_NMAE_pct", np.nan)
    try:
        s = float(s)
    except Exception:
        s = np.nan
    if not np.isfinite(s):
        s = metrics.get("ALL_NMAE_pct", np.nan)
        try:
            s = float(s)
        except Exception:
            s = np.nan
    if not np.isfinite(s):
        raise ValueError("Validation NMAE를 계산할 수 없습니다.")
    return float(s)


def evaluate_fixed_subset(
    model: str,
    features: List[str],
    fixed_params: Dict,
    screen_train_df: pd.DataFrame,
    screen_valid_df: pd.DataFrame,
    base_mod,
    rf_mod,
    seed: int,
    rf_n_jobs: int,
    epochs: int,
    patience: int,
    device: str,
    clip_predictions: bool,
) -> Tuple[float, Dict]:
    """Test를 전혀 사용하지 않는 SFFS 1개 subset 평가."""
    start = time.time()

    if model == "random_forest":
        params = dict(fixed_params)
        params["random_state"] = seed
        params["n_jobs"] = rf_n_jobs
        params.setdefault("bootstrap", True)
        params.setdefault("criterion", "squared_error")

        tr = screen_train_df
        va = screen_valid_df
        mdl = RandomForestRegressor(**params)
        mdl.fit(tr[features], tr[base_mod.TARGET_COL])
        pred = mdl.predict(va[features])
        pred_df = rf_mod.build_prediction_df(
            va,
            pred,
            "RandomForest",
            "SFFS_SCREEN",
            clip_predictions,
        )
        metrics = rf_mod.evaluate_prediction_df(pred_df)
        score = score_from_metrics(metrics)
        extra = {
            "train_samples": len(tr),
            "valid_samples": len(va),
            "valid_day_samples": metrics.get("DAY_n_samples", np.nan),
        }
        del mdl, pred, pred_df

    elif model == "lightgbm":
        if lgb is None:
            raise ImportError("lightgbm 설치가 필요합니다.")
        params = {
            "objective": "regression",
            "metric": "rmse",
            "random_state": seed,
            "verbosity": -1,
            **dict(fixed_params),
        }
        tr = screen_train_df
        va = screen_valid_df
        mdl = lgb.LGBMRegressor(**params)
        mdl.fit(
            tr[features],
            tr[base_mod.TARGET_COL],
            eval_set=[(va[features], va[base_mod.TARGET_COL])],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        pred = mdl.predict(va[features], num_iteration=mdl.best_iteration_)
        pred_df = base_mod.build_prediction_df(
            va,
            va[base_mod.DATE_COL].values,
            pred,
            "LightGBM",
            "SFFS_SCREEN",
            clip_predictions,
        )
        metrics = base_mod.evaluate_prediction_df(pred_df)
        score = score_from_metrics(metrics)
        extra = {
            "train_samples": len(tr),
            "valid_samples": len(va),
            "valid_day_samples": metrics.get("DAY_n_samples", np.nan),
            "best_iteration": int(mdl.best_iteration_ or 0),
        }
        del mdl, pred, pred_df

    elif model == "lstm":
        if torch is None:
            raise ImportError("PyTorch 설치가 필요합니다.")
        if hasattr(base_mod, "set_seed"):
            # 기존 04 코드의 seed 정책을 그대로 사용.
            base_mod.set_seed(seed)

        p = dict(fixed_params)
        required = [
            "seq_len", "hidden_size", "num_layers", "dropout",
            "learning_rate", "batch_size",
        ]
        miss = [x for x in required if x not in p]
        if miss:
            raise ValueError(f"LSTM Full69 best_params 누락: {miss}")

        scaler = StandardScaler()
        scaler.fit(screen_train_df[features].astype(float).values)
        seq_len = int(p["seq_len"])
        X_tr, y_tr, t_tr = base_mod.make_sequences_gap_safe(
            screen_train_df, features, seq_len, scaler=scaler
        )
        X_va, y_va, t_va = base_mod.make_sequences_gap_safe(
            screen_valid_df, features, seq_len, scaler=scaler
        )
        if len(X_tr) == 0 or len(X_va) == 0:
            raise ValueError(
                f"LSTM SFFS subset에서 sequence가 없습니다: {features}"
            )

        mdl = base_mod.LSTMRegressor(
            input_dim=len(features),
            hidden_size=int(p["hidden_size"]),
            num_layers=int(p["num_layers"]),
            dropout=float(p["dropout"]),
        )
        mdl, best_score, valid_metrics = base_mod.train_torch_one_run(
            mdl,
            X_tr,
            y_tr,
            X_va,
            y_va,
            screen_valid_df,
            t_va,
            lr=float(p["learning_rate"]),
            batch_size=int(p["batch_size"]),
            max_epochs=epochs,
            patience=patience,
            device=device,
            clip_predictions=clip_predictions,
        )
        score = float(best_score)
        if not np.isfinite(score):
            score = score_from_metrics(valid_metrics)
        extra = {
            "train_samples": len(X_tr),
            "valid_samples": len(X_va),
            "valid_day_samples": valid_metrics.get("DAY_n_samples", np.nan),
            "seq_len": seq_len,
        }
        del mdl, X_tr, y_tr, t_tr, X_va, y_va, t_va, scaler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        raise ValueError(model)

    gc.collect()
    extra["elapsed_seconds"] = float(time.time() - start)
    return float(score), extra


# =============================================================================
# 4. SFFS cache + algorithm
# =============================================================================
def canonical_subset(features: List[str], rank_map: Dict[str, int]) -> Tuple[str, ...]:
    return tuple(sorted(set(features), key=lambda f: rank_map[f]))


def subset_key(features: List[str], rank_map: Dict[str, int]) -> str:
    return json.dumps(list(canonical_subset(features, rank_map)), ensure_ascii=False)


def load_cache(path: Path) -> pd.DataFrame:
    if path.is_file():
        df = pd.read_csv(path, encoding="utf-8-sig")
        if "subset_key" not in df.columns or "score_VALID_DAY_NMAE_pct" not in df.columns:
            raise ValueError(f"{path}: cache 컬럼 구조가 예상과 다릅니다.")
        return df
    return pd.DataFrame()


def cache_to_map(df: pd.DataFrame) -> Dict[str, Dict]:
    out = {}
    if df.empty:
        return out
    for _, r in df.iterrows():
        key = str(r["subset_key"])
        out[key] = r.to_dict()
    return out


def run_sffs(
    model: str,
    candidates: List[str],
    rank_map: Dict[str, int],
    evaluator,
    model_dir: Path,
    max_unique_evals: int,
    experiment_signature: str,
) -> Tuple[List[str], float, pd.DataFrame, pd.DataFrame, int]:
    """
    Standard-style SFFS for minimization.

    Forward:
      현재 subset에 후보 1개를 추가한 모든 경우 중 Validation DAY_NMAE 최소 선택.

    Floating backward:
      방금 inclusion 후, 현재 subset에서 1개를 제거한 후보 중 가장 좋은 subset이
      그 cardinality에서 지금까지 기록된 best subset을 개선하면 제거를 채택.
      개선되는 동안 conditional exclusion을 반복.

    최종 선택:
      cardinality 1~N에서 발견된 best subset 중 Validation DAY_NMAE 최소.
      score 동률(수치 tolerance 내)이면 특징 수가 작은 subset을 선택.
    """
    ensure_dir(model_dir)
    cache_path = model_dir / "subset_score_cache.csv"
    trace_path = model_dir / "sffs_trace.csv"
    best_size_path = model_dir / "best_subset_by_size.csv"

    cache_df = load_cache(cache_path)
    if not cache_df.empty:
        if "experiment_signature" not in cache_df.columns:
            raise ValueError(
                f"{cache_path}: 구버전 cache입니다. 다른 폴더로 실행하거나 cache를 삭제하세요."
            )
        sigs = set(cache_df["experiment_signature"].dropna().astype(str))
        if sigs != {experiment_signature}:
            raise ValueError(
                "기존 SFFS cache의 실험조건이 현재 조건과 다릅니다. "
                "잘못된 결과 재사용을 막기 위해 중단합니다. "
                "새 --output_dir을 사용하거나 해당 모델의 subset_score_cache.csv를 삭제하세요."
            )
    cache_map = cache_to_map(cache_df)
    new_fit_count = 0

    def eval_subset(feats: List[str]) -> Tuple[float, Dict]:
        nonlocal cache_df, cache_map, new_fit_count
        ordered = list(canonical_subset(feats, rank_map))
        key = subset_key(ordered, rank_map)
        if key in cache_map:
            row = cache_map[key]
            return float(row["score_VALID_DAY_NMAE_pct"]), row

        if len(cache_map) >= max_unique_evals:
            raise RuntimeError(
                f"SFFS unique subset 평가가 안전한도({max_unique_evals})에 도달했습니다."
            )

        score, extra = evaluator(ordered)
        new_fit_count += 1
        row = {
            "evaluation_id": len(cache_map) + 1,
            "experiment_signature": experiment_signature,
            "model": DISPLAY[model],
            "feature_count": len(ordered),
            "subset_key": key,
            "features": ";".join(ordered),
            "sum_shap_candidate_rank": int(sum(rank_map[f] for f in ordered)),
            "score_VALID_DAY_NMAE_pct": float(score),
            "new_fit_in_this_run": True,
            **extra,
        }
        cache_map[key] = row
        cache_df = pd.concat([cache_df, pd.DataFrame([row])], ignore_index=True)
        cache_df.to_csv(cache_path, index=False, encoding="utf-8-sig")
        print(
            f"[{DISPLAY[model]}][SFFS fit {new_fit_count}] "
            f"k={len(ordered):2d}, DAY_NMAE={score:.6f}%"
        )
        return float(score), row

    def better(score_a: float, feats_a: List[str], score_b: float, feats_b: List[str]) -> bool:
        """a가 b보다 좋은지. score -> 같은 k에서는 SHAP rank sum -> lexical."""
        if score_a < score_b - SFFS_SCORE_TOL:
            return True
        if abs(score_a - score_b) <= SFFS_SCORE_TOL:
            sa = sum(rank_map[f] for f in feats_a)
            sb = sum(rank_map[f] for f in feats_b)
            if sa < sb:
                return True
            if sa == sb:
                return tuple(feats_a) < tuple(feats_b)
        return False

    current: List[str] = []
    best_by_size: Dict[int, Dict] = {}
    trace_rows = []
    step = 0

    while True:
        # ---------------- Forward inclusion ----------------
        remaining = [f for f in candidates if f not in current]
        if not remaining:
            break

        forward_best = None
        for f in remaining:
            trial_set = list(canonical_subset(current + [f], rank_map))
            score, _ = eval_subset(trial_set)
            item = (score, trial_set, f)
            if forward_best is None or better(
                item[0], item[1], forward_best[0], forward_best[1]
            ):
                forward_best = item

        assert forward_best is not None
        f_score, f_set, added = forward_best
        current = f_set
        k = len(current)
        step += 1

        prev = best_by_size.get(k)
        if prev is None or better(f_score, current, prev["score"], prev["features"]):
            best_by_size[k] = {"score": f_score, "features": list(current)}

        trace_rows.append({
            "step": step,
            "action": "INCLUDE",
            "feature_changed": added,
            "current_feature_count": k,
            "current_score_VALID_DAY_NMAE_pct": f_score,
            "current_features": ";".join(current),
        })
        pd.DataFrame(trace_rows).to_csv(trace_path, index=False, encoding="utf-8-sig")

        # ---------------- Conditional exclusion / floating ----------------
        while len(current) > 2:
            backward_best = None
            for f in list(current):
                trial_set = [x for x in current if x != f]
                trial_set = list(canonical_subset(trial_set, rank_map))
                score, _ = eval_subset(trial_set)
                item = (score, trial_set, f)
                if backward_best is None or better(
                    item[0], item[1], backward_best[0], backward_best[1]
                ):
                    backward_best = item

            assert backward_best is not None
            b_score, b_set, removed = backward_best
            target_k = len(b_set)
            incumbent = best_by_size.get(target_k)

            # classic floating condition: lower-cardinality best를 개선할 때만 exclusion 채택
            accept = (
                incumbent is None
                or better(b_score, b_set, incumbent["score"], incumbent["features"])
            )
            if not accept:
                break

            current = b_set
            best_by_size[target_k] = {
                "score": b_score,
                "features": list(current),
            }
            step += 1
            trace_rows.append({
                "step": step,
                "action": "EXCLUDE",
                "feature_changed": removed,
                "current_feature_count": target_k,
                "current_score_VALID_DAY_NMAE_pct": b_score,
                "current_features": ";".join(current),
            })
            pd.DataFrame(trace_rows).to_csv(trace_path, index=False, encoding="utf-8-sig")

        # current가 전체 후보에 도달했고 exclusion도 더 이상 채택되지 않으면 종료
        if len(current) == len(candidates):
            break

    if not best_by_size:
        raise RuntimeError("SFFS가 유효한 subset을 생성하지 못했습니다.")

    best_rows = []
    for k in sorted(best_by_size):
        info = best_by_size[k]
        best_rows.append({
            "feature_count": k,
            "score_VALID_DAY_NMAE_pct": float(info["score"]),
            "sum_shap_candidate_rank": int(sum(rank_map[f] for f in info["features"])),
            "features": ";".join(info["features"]),
        })
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(best_size_path, index=False, encoding="utf-8-sig")

    # 최종 cardinality 선택: score 최소, 동률이면 더 적은 특징 수
    best_df2 = best_df.sort_values(
        ["score_VALID_DAY_NMAE_pct", "feature_count", "sum_shap_candidate_rank"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    winner = best_df2.iloc[0]
    selected = [x for x in str(winner["features"]).split(";") if x]
    selected_score = float(winner["score_VALID_DAY_NMAE_pct"])

    return selected, selected_score, pd.DataFrame(trace_rows), best_df, new_fit_count


# =============================================================================
# 5. Final selected subset: original Optuna functions / fixed Test
# =============================================================================
def run_final_tuned_training(
    model: str,
    case_name: str,
    feature_cols: List[str],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_root: Path,
    base_mod,
    rf_mod,
    args,
    device: str,
) -> Tuple[Dict, pd.DataFrame]:
    clip_predictions = not args.no_clip

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
    if model == "lstm":
        return base_mod.train_sequence_case(
            model_name="LSTM",
            case_name=case_name,
            feature_cols=feature_cols,
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            output_dir=output_root,
            n_trials=args.n_trials_lstm,
            seed=args.seed,
            max_epochs=args.epochs,
            patience=args.patience,
            device=device,
            clip_predictions=clip_predictions,
        )
    raise ValueError(model)


def calc_metrics(df: pd.DataFrame, capacity_col: str, daytime_col: str) -> Dict[str, float]:
    true = pd.to_numeric(df["true_MWh"], errors="coerce").to_numpy(float)
    pred = pd.to_numeric(df["pred_MWh"], errors="coerce").to_numpy(float)
    cap = pd.to_numeric(df[capacity_col], errors="coerce").to_numpy(float)
    daytime = pd.to_numeric(df[daytime_col], errors="coerce").to_numpy(float)
    safe_cap = np.where(cap == 0, np.nan, cap)

    def one(mask):
        valid = mask & np.isfinite(true) & np.isfinite(pred) & np.isfinite(safe_cap)
        n = int(valid.sum())
        if n == 0:
            return {"n_samples": 0, "MAE_MWh": np.nan, "RMSE_MWh": np.nan,
                    "NMAE_pct": np.nan, "NRMSE_pct": np.nan,
                    "MBE_MWh": np.nan, "R2": np.nan}
        y, p, c = true[valid], pred[valid], safe_cap[valid]
        e = p - y
        ss_res = float(np.sum((y - p) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        return {
            "n_samples": n,
            "MAE_MWh": float(np.mean(np.abs(e))),
            "RMSE_MWh": float(np.sqrt(np.mean(e ** 2))),
            "NMAE_pct": float(np.nanmean(np.abs(e) / c) * 100.0),
            "NRMSE_pct": float(np.sqrt(np.nanmean((e / c) ** 2)) * 100.0),
            "MBE_MWh": float(np.mean(e)),
            "R2": np.nan if ss_tot == 0 else float(1.0 - ss_res / ss_tot),
        }

    all_m = one(np.ones(len(df), dtype=bool))
    day_m = one(daytime > 0)
    return {
        **{f"ALL_{k}": v for k, v in all_m.items()},
        **{f"DAY_{k}": v for k, v in day_m.items()},
    }


def align_to_fixed_test(
    pred_df: pd.DataFrame,
    reference_times: pd.DatetimeIndex,
    date_col: str,
    label: str,
    strict: bool,
) -> Tuple[pd.DataFrame, int]:
    work = pred_df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    if work[date_col].duplicated().any():
        raise ValueError(f"{label}: prediction timestamp 중복")
    available = set(work[date_col])
    missing = [t for t in reference_times if t not in available]
    if missing and strict:
        raise ValueError(
            f"{label}: 기준 Test timestamp {len(missing)}개 누락. "
            f"예={missing[:5]}"
        )
    use_times = reference_times if not missing else pd.DatetimeIndex(
        [t for t in reference_times if t in available]
    )
    aligned = (
        work[work[date_col].isin(use_times)]
        .set_index(date_col).loc[use_times].reset_index()
    )
    return aligned, len(missing)


# =============================================================================
# 6. CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Model-specific SHAP Top-N + SFFS + final Optuna retuning"
    )
    p.add_argument("--csv", default=CSV_PATH)
    p.add_argument("--shap_root", default=SHAP_ROOT)
    p.add_argument("--output_dir", default=OUTPUT_DIR)
    p.add_argument("--base_train_script", default=BASE_TRAIN_SCRIPT)
    p.add_argument("--rf_train_script", default=RF_TRAIN_SCRIPT)
    p.add_argument("--full_lgb_model", default=FULL_LGB_MODEL)
    p.add_argument("--full_lstm_model", default=FULL_LSTM_MODEL)
    p.add_argument("--full_rf_model", default=FULL_RF_MODEL)
    p.add_argument("--fixed_test_timestamp_file", default=FIXED_TEST_TIMESTAMP_FILE)

    p.add_argument("--models", default=DEFAULT_MODELS,
                   help="포함할 모델. 실행 순서는 항상 RF -> LightGBM -> LSTM")
    p.add_argument("--candidate_top_n", type=int, default=DEFAULT_CANDIDATE_TOP_N)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--rf_n_jobs", type=int, default=DEFAULT_RF_N_JOBS)
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument("--n_trials_rf", type=int, default=DEFAULT_N_TRIALS_RF)
    p.add_argument("--n_trials_lgb", type=int, default=DEFAULT_N_TRIALS_LGB)
    p.add_argument("--n_trials_lstm", type=int, default=DEFAULT_N_TRIALS_LSTM)
    p.add_argument("--max_unique_subset_evals", type=int, default=MAX_UNIQUE_SUBSET_EVALS)
    p.add_argument("--no_clip", action="store_true")
    p.add_argument("--no_skip_existing", action="store_true")
    p.add_argument("--skip_final_tuning", action="store_true",
                   help="SFFS selection까지만 수행하고 최종 Optuna/Test는 생략")
    p.add_argument("--non_strict_test", action="store_true")
    p.add_argument("--quick_test", action="store_true")
    return p.parse_args()


# =============================================================================
# 7. Main
# =============================================================================
def main():
    args = parse_args()
    skip_existing = not args.no_skip_existing
    strict_test = not args.non_strict_test

    if args.quick_test:
        args.candidate_top_n = min(args.candidate_top_n, 3)
        args.n_trials_rf = 1
        args.n_trials_lgb = 1
        args.n_trials_lstm = 1
        args.epochs = 2
        args.patience = 1

    root = project_root()
    csv_path = root / args.csv
    shap_root = ensure_shap_root(root, root / args.shap_root)
    output_dir = root / args.output_dir
    results_dir = output_dir / "results"
    sffs_root = output_dir / "sffs"
    final_root = output_dir / "final_tuned"
    fixed_dir = output_dir / "fixed_test_comparison"
    for d in [results_dir, sffs_root, final_root, fixed_dir]:
        ensure_dir(d)

    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    base_script = resolve_training_script(root, args.base_train_script, "base")
    rf_script = resolve_training_script(root, args.rf_train_script, "rf")
    base_mod = load_python_module("exp04_base_sffs", base_script)
    rf_mod = load_python_module("exp04_rf_sffs", rf_script)
    device = resolve_device(args.device, base_mod)
    models = parse_models(args.models)
    if not models:
        raise ValueError("실행할 모델이 없습니다.")

    df = base_mod.read_feature_csv(str(csv_path))
    train_df, valid_df, test_df = base_mod.split_by_year(df)
    groups = base_mod.build_feature_groups(df)
    cases = base_mod.build_feature_cases(groups)
    if "C6_Full" not in cases:
        raise KeyError("C6_Full case 없음")
    full69_features = list(cases["C6_Full"])

    ref_path = root / args.fixed_test_timestamp_file
    reference_times = None
    if not args.skip_final_tuning:
        if not ref_path.is_file():
            raise FileNotFoundError(ref_path)
        ref_df = pd.read_csv(ref_path, encoding="utf-8-sig")
        if base_mod.DATE_COL not in ref_df.columns:
            raise ValueError(f"{ref_path}: {base_mod.DATE_COL} 컬럼 없음")
        reference_times = pd.DatetimeIndex(
            pd.to_datetime(ref_df[base_mod.DATE_COL])
        ).sort_values()
        if reference_times.duplicated().any():
            raise ValueError(
                f"{ref_path}: 기준 Test timestamp에 중복이 있습니다."
            )
        # Exp3-1의 수정된 same-test 비교와 동일한 기준 timestamp를 audit용으로 저장
        pd.DataFrame({base_mod.DATE_COL: reference_times}).to_csv(
            fixed_dir / "FIXED_reference_test_timestamps_from_C6_Full.csv",
            index=False,
            encoding="utf-8-sig",
        )

    config = {
        "method": "model-specific SHAP Top-N preselection + SFFS wrapper",
        "candidate_top_n": args.candidate_top_n,
        "sffs_final_feature_count_fixed": False,
        "sffs_selection_metric": "2023 Validation DAY_NMAE",
        "sffs_hyperparameters": "fixed at each model's Full69 best_params",
        "sffs_sample_policy": "candidate-complete fixed Train/Validation sample within each model",
        "floating_rule": "forward inclusion + conditional exclusion if lower-cardinality best improves",
        "test_used_in_sffs": False,
        "final_selected_subset_retuned": not args.skip_final_tuning,
        "final_optuna_trials": {
            "random_forest": args.n_trials_rf,
            "lightgbm": args.n_trials_lgb,
            "lstm": args.n_trials_lstm,
        },
        "split": {"train": "2020-2022", "validation": "2023", "test": "2024"},
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "model_order": [DISPLAY[m] for m in models],
    }
    save_json(config, output_dir / "experiment_config.json")

    selected_rows = []
    final_rows = []
    fixed_rows = []
    fixed_predictions = []

    print("=" * 112)
    print("[Experiment 5] Model-specific SHAP + SFFS")
    print("Models       :", " -> ".join(DISPLAY[m] for m in models))
    print("Candidates   : model-specific SHAP Top", args.candidate_top_n, sep="")
    print("SFFS score   : 2023 Validation DAY_NMAE")
    print("SFFS HP      : Full69 best_params fixed")
    print("Final K      : NOT fixed in advance")
    print("Final retune :", not args.skip_final_tuning)
    print("=" * 112)

    for model in models:
        display = DISPLAY[model]
        model_dir = sffs_root / model
        ensure_dir(model_dir)

        print("\n" + "#" * 112)
        print(f"[MODEL START] {display}")
        print("#" * 112)

        # Full69 bundle / fixed params
        full_bundle, fixed_params, full_model_path = load_full69_bundle_and_params(
            root, model, args
        )
        bundle_features = list(full_bundle.get("feature_cols", full69_features))
        save_json(fixed_params, model_dir / "full69_fixed_params.json")

        # model-specific SHAP TopN
        cand_df = load_shap_candidates(
            shap_root,
            model,
            args.candidate_top_n,
            allowed_features=bundle_features,
        )
        candidates = cand_df["feature"].astype(str).tolist()
        rank_map = {f: i + 1 for i, f in enumerate(candidates)}
        cand_df.to_csv(
            model_dir / f"candidate_SHAP_Top{args.candidate_top_n}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # fixed screening sample within model
        screen_train, screen_valid, sample_audit = prepare_screening_frames(
            train_df, valid_df, candidates, base_mod
        )

        # LSTM sequence audit under fixed Full69 seq_len
        if model == "lstm":
            p = fixed_params
            seq_len = int(p["seq_len"])
            scaler0 = StandardScaler().fit(screen_train[candidates].astype(float).values)
            X0tr, _, _ = base_mod.make_sequences_gap_safe(
                screen_train, candidates, seq_len, scaler=scaler0
            )
            X0va, _, _ = base_mod.make_sequences_gap_safe(
                screen_valid, candidates, seq_len, scaler=scaler0
            )
            sample_audit.update({
                "fixed_seq_len": seq_len,
                "train_sequences_candidate_complete": len(X0tr),
                "valid_sequences_candidate_complete": len(X0va),
            })
            del X0tr, X0va, scaler0
            gc.collect()

        sample_audit.update({
            "model": display,
            "candidate_top_n": args.candidate_top_n,
            "full69_model_path": str(full_model_path),
        })
        pd.DataFrame([sample_audit]).to_csv(
            model_dir / "screening_sample_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )

        clip_predictions = not args.no_clip

        screen_signature = stable_hash({
            "model": model,
            "candidate_top_n": args.candidate_top_n,
            "candidates": candidates,
            "fixed_params": fixed_params,
            "seed": args.seed,
            "epochs": args.epochs,
            "patience": args.patience,
            "clip_predictions": clip_predictions,
            "train_rows": len(screen_train),
            "valid_rows": len(screen_valid),
        })
        save_json({
            "experiment_signature": screen_signature,
            "model": display,
            "candidate_top_n": args.candidate_top_n,
            "candidates": candidates,
            "fixed_params": fixed_params,
            "seed": args.seed,
            "epochs": args.epochs,
            "patience": args.patience,
            "screening_sample_audit": sample_audit,
        }, model_dir / "sffs_screening_signature.json")

        def evaluator(feats: List[str]):
            return evaluate_fixed_subset(
                model=model,
                features=feats,
                fixed_params=fixed_params,
                screen_train_df=screen_train,
                screen_valid_df=screen_valid,
                base_mod=base_mod,
                rf_mod=rf_mod,
                seed=args.seed,
                rf_n_jobs=args.rf_n_jobs,
                epochs=args.epochs,
                patience=args.patience,
                device=device,
                clip_predictions=clip_predictions,
            )

        selected_features, sffs_score, trace_df, best_by_size_df, new_fit_count = run_sffs(
            model=model,
            candidates=candidates,
            rank_map=rank_map,
            evaluator=evaluator,
            model_dir=model_dir,
            max_unique_evals=args.max_unique_subset_evals,
            experiment_signature=screen_signature,
        )

        selected_feature_file = results_dir / f"features_sffs_{model}.csv"
        out_feat = cand_df[cand_df["feature"].isin(selected_features)].copy()
        out_feat["selected_by_sffs"] = True
        # SFFS subset 내에서는 SHAP rank 순으로 저장 (학습 input 순서 안정성)
        out_feat = out_feat.sort_values("sffs_candidate_rank")
        out_feat.to_csv(selected_feature_file, index=False, encoding="utf-8-sig")

        selected_sig = feature_signature(selected_features)
        selected_row = {
            "model": display,
            "model_key": model,
            "method": "SHAP Top-N + SFFS",
            "candidate_top_n": args.candidate_top_n,
            "selected_feature_count": len(selected_features),
            "selected_feature_signature": selected_sig,
            "SFFS_VALID_DAY_NMAE_pct": sffs_score,
            "selected_features": ";".join(selected_features),
            "feature_file": str(selected_feature_file),
            "full69_model_path_for_fixed_params": str(full_model_path),
            "new_subset_fits_this_run": new_fit_count,
            "cached_unique_subset_count": len(load_cache(model_dir / "subset_score_cache.csv")),
            "test_used_in_selection": False,
        }
        selected_rows.append(selected_row)
        pd.DataFrame([selected_row]).to_csv(
            results_dir / f"SFFS_SELECTED_summary_{model}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(selected_rows).to_csv(
            results_dir / "SFFS_SELECTED_all_models.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("\n" + "=" * 112)
        print(f"[{display} SFFS SELECTED]")
        print("Candidate TopN       :", args.candidate_top_n)
        print("Selected feature cnt :", len(selected_features))
        print(f"Validation DAY_NMAE  : {sffs_score:.6f}%")
        print("Selected features    :", selected_features)
        print("=" * 112)

        if args.skip_final_tuning:
            continue

        # ------------------------------------------------------------------
        # Final subset only: original Optuna 50 trials + REAL 2024 Test
        # ------------------------------------------------------------------
        final_case = f"SHAP_SFFS_{display}_K{len(selected_features)}_{selected_sig}"
        final_summary_path = results_dir / f"FINAL_TEST_SFFS_{model}.csv"
        final_pred_path = prediction_path(final_root, model, final_case)

        if skip_existing and final_summary_path.is_file() and final_pred_path.is_file():
            print(f"[{display}] Final tuned model: 기존 결과 재사용")
            final_summary = pd.read_csv(final_summary_path, encoding="utf-8-sig").iloc[0].to_dict()
            pred_df = pd.read_csv(final_pred_path, encoding="utf-8-sig")
        else:
            print(f"[{display}] Final subset Optuna retuning + 2024 Test 시작")
            final_summary, pred_df = run_final_tuned_training(
                model=model,
                case_name=final_case,
                feature_cols=selected_features,
                train_df=train_df,
                valid_df=valid_df,
                test_df=test_df,
                output_root=final_root,
                base_mod=base_mod,
                rf_mod=rf_mod,
                args=args,
                device=device,
            )
            final_summary = dict(final_summary)
            final_summary.update({
                "model_key": model,
                "method": "SHAP Top-N + SFFS -> final Optuna retuning",
                "candidate_top_n": args.candidate_top_n,
                "selected_feature_count": len(selected_features),
                "SFFS_VALID_DAY_NMAE_pct": sffs_score,
                "selected_feature_file": str(selected_feature_file),
                "test_used_in_sffs": False,
                "real_2024_test_used": True,
            })
            pd.DataFrame([final_summary]).to_csv(
                final_summary_path, index=False, encoding="utf-8-sig"
            )

        final_rows.append(final_summary)
        pd.DataFrame(final_rows).to_csv(
            results_dir / "FINAL_TEST_SFFS_all_models.csv",
            index=False,
            encoding="utf-8-sig",
        )

        assert reference_times is not None
        aligned, missing = align_to_fixed_test(
            pred_df,
            reference_times,
            base_mod.DATE_COL,
            f"SHAP+SFFS/{display}",
            strict_test,
        )
        metrics = calc_metrics(
            aligned,
            capacity_col=base_mod.CAPACITY_COL,
            daytime_col=base_mod.DAYTIME_COL,
        )
        fixed_row = {
            "model": display,
            "model_key": model,
            "method": "SHAP Top-N + SFFS",
            "candidate_top_n": args.candidate_top_n,
            "selected_feature_count": len(selected_features),
            "SFFS_VALID_DAY_NMAE_pct": sffs_score,
            "reference_test_timestamps": len(reference_times),
            "missing_reference_timestamps": missing,
            **metrics,
        }
        fixed_rows.append(fixed_row)
        pd.DataFrame(fixed_rows).to_csv(
            fixed_dir / "MAIN_SHAP_SFFS_same_C6Full_test_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        aligned2 = aligned.copy()
        aligned2["model"] = display
        aligned2["model_key"] = model
        aligned2["method"] = "SHAP_SFFS"
        aligned2["selected_feature_count"] = len(selected_features)
        fixed_predictions.append(aligned2)
        pd.concat(fixed_predictions, ignore_index=True).to_csv(
            fixed_dir / "PREDICTIONS_SHAP_SFFS_same_C6Full_test.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(f"[{display}] Fixed-Test DAY_NMAE = {metrics['DAY_NMAE_pct']:.6f}%")

    print("\n" + "#" * 112)
    print("[완료] Model-specific SHAP + SFFS")
    print("선택 특징:", results_dir / "features_sffs_<model>.csv")
    print("SFFS 요약:", results_dir / "SFFS_SELECTED_all_models.csv")
    if not args.skip_final_tuning:
        print("최종 Test:", results_dir / "FINAL_TEST_SFFS_all_models.csv")
        print("동일 Test 비교:", fixed_dir / "MAIN_SHAP_SFFS_same_C6Full_test_metrics.csv")
    print("다음 단계: 10_exp6_cross_model_sffs.py")
    print("#" * 112)


if __name__ == "__main__":
    main()
