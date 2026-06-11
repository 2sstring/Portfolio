# -*- coding: utf-8 -*-
"""
PV forecasting full training pipeline
- Target: 정규화발전량
- Final evaluation: restored generation MWh = predicted normalized power × 설비용량(MW)
- Feature cases: C1~C6
- Models: LightGBM, LSTM, PatchTST
- Saves trained models for later prediction

Recommended input:
    chungbuk_pv_features.csv
created by the gap-safe feature generation script.
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import optuna
except Exception:
    optuna = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except Exception:
    torch = None
    nn = None
    Dataset = object
    DataLoader = None


# ============================================================
# 0. Global column names
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


# ============================================================
# 1. Utilities
# ============================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_feature_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if DATE_COL not in df.columns:
        raise ValueError(f"'{DATE_COL}' 컬럼이 없습니다.")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


def build_feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    g1 = [c for c in G1_TIME_BASE if c in df.columns]
    g2 = [c for c in G2_CURRENT_BASE if c in df.columns]
    g3 = [c for c in df.columns if "_lag_" in c]
    g4 = [c for c in df.columns if "_rollmean_" in c]
    g5 = [c for c in df.columns if "_diff_" in c]

    groups = {
        "G1_Time": g1,
        "G2_Current": g2,
        "G3_Lag": g3,
        "G4_Rolling": g4,
        "G5_Diff": g5,
    }

    # Basic validation
    for name, cols in groups.items():
        if len(cols) == 0:
            print(f"[경고] {name} 컬럼이 비어 있습니다.")
    return groups


def build_feature_cases(groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
    g1 = groups["G1_Time"]
    g2 = groups["G2_Current"]
    g3 = groups["G3_Lag"]
    g4 = groups["G4_Rolling"]
    g5 = groups["G5_Diff"]

    cases = {
        "C1_Time_Current": g1 + g2,
        "C2_Time_Current_Lag": g1 + g2 + g3,
        "C3_Time_Current_Rolling": g1 + g2 + g4,
        "C4_Time_Current_Diff": g1 + g2 + g5,
        "C5_Time_Current_Lag_Rolling": g1 + g2 + g3 + g4,
        "C6_Full": g1 + g2 + g3 + g4 + g5,
    }

    # Remove duplicates while preserving order
    deduped = {}
    for case, cols in cases.items():
        seen = set()
        out = []
        for c in cols:
            if c not in seen and c not in EXCLUDE_FEATURE_COLS:
                out.append(c)
                seen.add(c)
        deduped[case] = out
    return deduped


def split_by_year(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df[DATE_COL].dt.year
    train_df = df[(y >= 2020) & (y <= 2022)].copy()
    valid_df = df[y == 2023].copy()
    test_df = df[y == 2024].copy()
    return train_df, valid_df, test_df


def clip_pred(pred: np.ndarray, do_clip: bool = True) -> np.ndarray:
    if do_clip:
        return np.clip(pred, 0.0, None)
    return pred


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

    err = pred_mwh - true_mwh
    abs_err = np.abs(err)
    mae = np.mean(abs_err)
    rmse = np.sqrt(np.mean(err ** 2))

    # Capacity-normalized metrics, calculated per timestamp.
    safe_capacity = np.where(capacity_mw == 0, np.nan, capacity_mw)
    nmae = np.nanmean(abs_err / safe_capacity) * 100.0
    nrmse = np.sqrt(np.nanmean((err / safe_capacity) ** 2)) * 100.0
    mbe = np.mean(err)

    try:
        r2 = r2_score(true_mwh, pred_mwh)
    except Exception:
        r2 = np.nan

    return {
        "n_samples": int(len(true_mwh)),
        "MAE_MWh": float(mae),
        "RMSE_MWh": float(rmse),
        "NMAE_pct": float(nmae),
        "NRMSE_pct": float(nrmse),
        "MBE_MWh": float(mbe),
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

    out = {}
    for k, v in all_metrics.items():
        out[f"ALL_{k}"] = v
    for k, v in day_metrics.items():
        out[f"DAY_{k}"] = v
    return out


def build_prediction_df(
    base_df: pd.DataFrame,
    target_times: np.ndarray,
    pred_norm: np.ndarray,
    model_name: str,
    case_name: str,
    clip_predictions: bool = True,
) -> pd.DataFrame:
    pred_norm = clip_pred(np.asarray(pred_norm).reshape(-1), clip_predictions)
    pred_key = pd.DataFrame({
        DATE_COL: pd.to_datetime(target_times),
        "pred_norm": pred_norm,
    })

    cols = [DATE_COL, CAPACITY_COL, POWER_COL, TARGET_COL, DAYTIME_COL]
    if SEGMENT_COL in base_df.columns:
        cols.append(SEGMENT_COL)
    base = base_df[cols].copy()
    out = base.merge(pred_key, on=DATE_COL, how="inner")
    out["true_norm"] = out[TARGET_COL]
    out["true_MWh"] = out[POWER_COL]
    out["pred_MWh"] = out["pred_norm"] * out[CAPACITY_COL]
    out["error_MWh"] = out["pred_MWh"] - out["true_MWh"]
    out["model"] = model_name
    out["case"] = case_name
    return out


# ============================================================
# 2. Sequence utilities
# ============================================================
def recompute_seq_segment_id(df: pd.DataFrame) -> pd.Series:
    """
    Recompute segment after train/valid/test split and after any deletion.
    This prevents sequence construction across newly created gaps.
    """
    time_diff = df[DATE_COL].diff()
    new_segment = time_diff != pd.Timedelta(hours=1)
    new_segment.iloc[0] = True

    if SEGMENT_COL in df.columns:
        original_change = df[SEGMENT_COL].diff().fillna(0) != 0
        new_segment = new_segment | original_change

    return new_segment.cumsum()


def make_sequences_gap_safe(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int,
    pred_horizon: int = 1,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sequence samples only within continuous hourly segments.
    X: t-seq_len ... t-1
    y: t + pred_horizon - 1
    """
    work = df.copy()
    work[DATE_COL] = pd.to_datetime(work[DATE_COL])
    work = work.sort_values(DATE_COL).reset_index(drop=True)
    work["seq_segment_id"] = recompute_seq_segment_id(work)

    X_list, y_list, time_list = [], [], []

    for _, seg in work.groupby("seq_segment_id"):
        seg = seg.sort_values(DATE_COL).reset_index(drop=True)
        if len(seg) < seq_len + pred_horizon:
            continue

        X_raw = seg[feature_cols].astype(float).values
        if scaler is not None:
            X_raw = scaler.transform(X_raw)
        y_raw = seg[TARGET_COL].astype(float).values
        t_raw = seg[DATE_COL].values

        for end_idx in range(seq_len, len(seg) - pred_horizon + 1):
            start_idx = end_idx - seq_len
            target_idx = end_idx + pred_horizon - 1
            X_seq = X_raw[start_idx:end_idx]
            y_target = y_raw[target_idx]

            if np.isnan(X_seq).any() or np.isnan(y_target):
                continue

            X_list.append(X_seq.astype(np.float32))
            y_list.append(np.float32(y_target))
            time_list.append(t_raw[target_idx])

    if len(X_list) == 0:
        return (
            np.empty((0, seq_len, len(feature_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype="datetime64[ns]"),
        )

    return (
        np.asarray(X_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.float32),
        np.asarray(time_list),
    )


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


# ============================================================
# 3. Models
# ============================================================
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class SimplePatchTSTRegressor(nn.Module):
    """
    Lightweight PatchTST-style model for multivariate forecasting.
    It creates temporal patches, embeds each patch, applies TransformerEncoder,
    then predicts one scalar target.
    """
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        patch_len: int,
        stride: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        if patch_len > seq_len:
            raise ValueError("patch_len must be <= seq_len")
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = 1 + (seq_len - patch_len) // stride

        self.patch_embed = nn.Linear(input_dim * patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        # x: [B, L, C]
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # unfold output: [B, n_patches, C, patch_len]
        patches = patches.contiguous().reshape(x.size(0), self.n_patches, self.input_dim * self.patch_len)
        z = self.patch_embed(patches) + self.pos_embed
        z = self.encoder(z)
        pooled = z.mean(dim=1)
        return self.head(pooled).squeeze(-1)


# ============================================================
# 4. Training helpers
# ============================================================
def train_torch_one_run(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    valid_eval_base_df: pd.DataFrame,
    valid_times: np.ndarray,
    lr: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    device: str,
    clip_predictions: bool,
) -> Tuple[nn.Module, float, Dict[str, float]]:
    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(SeqDataset(X_valid, y_valid), batch_size=batch_size, shuffle=False)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_score = float("inf")
    best_state = None
    best_metrics = {}
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation prediction
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in valid_loader:
                xb = xb.to(device)
                pred = model(xb).detach().cpu().numpy()
                preds.append(pred)
        pred_norm = np.concatenate(preds) if preds else np.array([])
        pred_df = build_prediction_df(
            valid_eval_base_df,
            valid_times,
            pred_norm,
            model_name="torch_model",
            case_name="valid",
            clip_predictions=clip_predictions,
        )
        metrics = evaluate_prediction_df(pred_df)
        score = metrics["DAY_NMAE_pct"]
        if np.isnan(score):
            score = metrics["ALL_NMAE_pct"]

        if score < best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = metrics
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_score, best_metrics


def predict_torch(model: nn.Module, X: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    loader = DataLoader(SeqDataset(X, np.zeros(len(X), dtype=np.float32)), batch_size=batch_size, shuffle=False)
    model = model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            pred = model(xb).detach().cpu().numpy()
            preds.append(pred)
    if not preds:
        return np.array([])
    return np.concatenate(preds)


# ============================================================
# 5. LightGBM training
# ============================================================
def train_lightgbm_case(
    case_name: str,
    feature_cols: List[str],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    n_trials: int,
    seed: int,
    clip_predictions: bool,
) -> Tuple[Dict, pd.DataFrame]:
    if lgb is None:
        raise ImportError("lightgbm이 설치되어 있지 않습니다. pip install lightgbm")
    if optuna is None:
        raise ImportError("optuna가 설치되어 있지 않습니다. pip install optuna")

    use_cols = feature_cols + [TARGET_COL, DATE_COL, CAPACITY_COL, POWER_COL, DAYTIME_COL]
    if SEGMENT_COL in train_df.columns:
        use_cols.append(SEGMENT_COL)
    use_cols = list(dict.fromkeys(use_cols))

    tr = train_df[use_cols].dropna().copy()
    va = valid_df[use_cols].dropna().copy()
    te = test_df[use_cols].dropna().copy()

    X_tr = tr[feature_cols]
    y_tr = tr[TARGET_COL]
    X_va = va[feature_cols]
    y_va = va[TARGET_COL]

    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "random_state": seed,
            "verbosity": -1,
            "n_estimators": trial.suggest_int("n_estimators", 300, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        pred_norm = model.predict(X_va, num_iteration=model.best_iteration_)
        pred_df = build_prediction_df(va, va[DATE_COL].values, pred_norm, "LightGBM", case_name, clip_predictions)
        metrics = evaluate_prediction_df(pred_df)
        score = metrics["DAY_NMAE_pct"]
        if np.isnan(score):
            score = metrics["ALL_NMAE_pct"]
        return score

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    final_params = {
        "objective": "regression",
        "metric": "rmse",
        "random_state": seed,
        "verbosity": -1,
        **best_params,
    }
    final_model = lgb.LGBMRegressor(**final_params)
    final_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    # Validation metrics with final model
    valid_pred_norm = final_model.predict(X_va, num_iteration=final_model.best_iteration_)
    valid_pred_df = build_prediction_df(va, va[DATE_COL].values, valid_pred_norm, "LightGBM", case_name, clip_predictions)
    valid_metrics = evaluate_prediction_df(valid_pred_df)

    # Test prediction
    X_te = te[feature_cols]
    test_pred_norm = final_model.predict(X_te, num_iteration=final_model.best_iteration_)
    test_pred_df = build_prediction_df(te, te[DATE_COL].values, test_pred_norm, "LightGBM", case_name, clip_predictions)
    test_metrics = evaluate_prediction_df(test_pred_df)

    model_dir = output_dir / "models" / "lightgbm" / case_name
    ensure_dir(model_dir)
    joblib.dump({
        "model_type": "lightgbm",
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
        "best_iteration": int(final_model.best_iteration_ or 0),
        "clip_predictions": clip_predictions,
    }, model_dir / "model.pkl")

    save_json({"best_params": best_params, "valid_metrics": valid_metrics, "test_metrics": test_metrics}, model_dir / "metadata.json")

    pred_dir = output_dir / "predictions"
    ensure_dir(pred_dir)
    test_pred_df.to_csv(pred_dir / f"test_predictions_lightgbm_{case_name}.csv", index=False, encoding="utf-8-sig")
    valid_pred_df.to_csv(pred_dir / f"valid_predictions_lightgbm_{case_name}.csv", index=False, encoding="utf-8-sig")

    # Feature importance
    imp = pd.DataFrame({
        "model": "LightGBM",
        "case": case_name,
        "feature": feature_cols,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    imp.to_csv(model_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    summary = {
        "model": "LightGBM",
        "case": case_name,
        "feature_count": len(feature_cols),
        "train_rows": len(tr),
        "valid_rows": len(va),
        "test_rows": len(te),
        "valid_objective_DAY_NMAE_pct": float(study.best_value),
        **{f"VALID_{k}": v for k, v in valid_metrics.items()},
        **{f"TEST_{k}": v for k, v in test_metrics.items()},
        "model_path": str(model_dir / "model.pkl"),
    }
    return summary, test_pred_df


# ============================================================
# 6. LSTM / PatchTST training
# ============================================================
def train_sequence_case(
    model_name: str,
    case_name: str,
    feature_cols: List[str],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    n_trials: int,
    seed: int,
    max_epochs: int,
    patience: int,
    device: str,
    clip_predictions: bool,
) -> Tuple[Dict, pd.DataFrame]:
    if torch is None or optuna is None:
        raise ImportError("torch와 optuna가 필요합니다. pip install torch optuna")

    # Remove rows with NaN in selected features/target only for scaler fit.
    # Sequence constructor also skips windows with NaN.
    scaler_fit_df = train_df[feature_cols].dropna().copy()
    scaler = StandardScaler()
    scaler.fit(scaler_fit_df.astype(float).values)

    def build_data(seq_len: int):
        X_tr, y_tr, t_tr = make_sequences_gap_safe(train_df, feature_cols, seq_len, scaler=scaler)
        X_va, y_va, t_va = make_sequences_gap_safe(valid_df, feature_cols, seq_len, scaler=scaler)
        X_te, y_te, t_te = make_sequences_gap_safe(test_df, feature_cols, seq_len, scaler=scaler)
        return X_tr, y_tr, t_tr, X_va, y_va, t_va, X_te, y_te, t_te

    def objective(trial):
        if model_name == "LSTM":
            seq_len = trial.suggest_categorical("seq_len", [24, 48])
            hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
            num_layers = trial.suggest_int("num_layers", 1, 3)
            dropout = trial.suggest_float("dropout", 0.0, 0.5)
            lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            X_tr, y_tr, _, X_va, y_va, t_va, _, _, _ = build_data(seq_len)
            if len(X_tr) == 0 or len(X_va) == 0:
                return float("inf")
            model = LSTMRegressor(len(feature_cols), hidden_size, num_layers, dropout)
            params = {
                "seq_len": seq_len,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "learning_rate": lr,
                "batch_size": batch_size,
            }
        else:
            seq_len = trial.suggest_categorical("seq_len", [96, 168])
            patch_len = trial.suggest_categorical("patch_len", [8, 12, 16, 24])
            stride = trial.suggest_categorical("stride", [4, 8, 12])
            if patch_len > seq_len:
                return float("inf")
            d_model = trial.suggest_categorical("d_model", [64, 128, 256])
            valid_heads = [h for h in [2, 4, 8] if d_model % h == 0]
            n_heads = trial.suggest_categorical("n_heads", valid_heads)
            num_layers = trial.suggest_int("num_layers", 1, 4)
            dropout = trial.suggest_float("dropout", 0.0, 0.5)
            lr = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            X_tr, y_tr, _, X_va, y_va, t_va, _, _, _ = build_data(seq_len)
            if len(X_tr) == 0 or len(X_va) == 0:
                return float("inf")
            model = SimplePatchTSTRegressor(
                input_dim=len(feature_cols),
                seq_len=seq_len,
                patch_len=patch_len,
                stride=stride,
                d_model=d_model,
                n_heads=n_heads,
                num_layers=num_layers,
                dropout=dropout,
            )
            params = {
                "seq_len": seq_len,
                "patch_len": patch_len,
                "stride": stride,
                "d_model": d_model,
                "n_heads": n_heads,
                "num_layers": num_layers,
                "dropout": dropout,
                "learning_rate": lr,
                "batch_size": batch_size,
            }

        model, best_score, _ = train_torch_one_run(
            model,
            X_tr,
            y_tr,
            X_va,
            y_va,
            valid_df,
            t_va,
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            device=device,
            clip_predictions=clip_predictions,
        )
        trial.set_user_attr("params", params)
        return best_score

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_trial.user_attrs.get("params", study.best_params)

    # Rebuild data and train final model using best params
    seq_len = int(best_params["seq_len"])
    X_tr, y_tr, t_tr, X_va, y_va, t_va, X_te, y_te, t_te = build_data(seq_len)

    if model_name == "LSTM":
        final_model = LSTMRegressor(
            input_dim=len(feature_cols),
            hidden_size=int(best_params["hidden_size"]),
            num_layers=int(best_params["num_layers"]),
            dropout=float(best_params["dropout"]),
        )
        arch_params = {
            "input_dim": len(feature_cols),
            "hidden_size": int(best_params["hidden_size"]),
            "num_layers": int(best_params["num_layers"]),
            "dropout": float(best_params["dropout"]),
        }
    else:
        final_model = SimplePatchTSTRegressor(
            input_dim=len(feature_cols),
            seq_len=seq_len,
            patch_len=int(best_params["patch_len"]),
            stride=int(best_params["stride"]),
            d_model=int(best_params["d_model"]),
            n_heads=int(best_params["n_heads"]),
            num_layers=int(best_params["num_layers"]),
            dropout=float(best_params["dropout"]),
        )
        arch_params = {
            "input_dim": len(feature_cols),
            "seq_len": seq_len,
            "patch_len": int(best_params["patch_len"]),
            "stride": int(best_params["stride"]),
            "d_model": int(best_params["d_model"]),
            "n_heads": int(best_params["n_heads"]),
            "num_layers": int(best_params["num_layers"]),
            "dropout": float(best_params["dropout"]),
        }

    final_model, final_valid_score, final_valid_metrics = train_torch_one_run(
        final_model,
        X_tr,
        y_tr,
        X_va,
        y_va,
        valid_df,
        t_va,
        lr=float(best_params["learning_rate"]),
        batch_size=int(best_params["batch_size"]),
        max_epochs=max_epochs,
        patience=patience,
        device=device,
        clip_predictions=clip_predictions,
    )

    # Validation and test predictions
    valid_pred_norm = predict_torch(final_model, X_va, int(best_params["batch_size"]), device)
    valid_pred_df = build_prediction_df(valid_df, t_va, valid_pred_norm, model_name, case_name, clip_predictions)
    valid_metrics = evaluate_prediction_df(valid_pred_df)

    test_pred_norm = predict_torch(final_model, X_te, int(best_params["batch_size"]), device)
    test_pred_df = build_prediction_df(test_df, t_te, test_pred_norm, model_name, case_name, clip_predictions)
    test_metrics = evaluate_prediction_df(test_pred_df)

    lower_model_name = model_name.lower()
    model_dir = output_dir / "models" / lower_model_name / case_name
    ensure_dir(model_dir)

    torch.save({
        "model_type": lower_model_name,
        "state_dict": final_model.cpu().state_dict(),
        "architecture": arch_params,
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
    }, model_dir / "model.pt")
    joblib.dump(scaler, model_dir / "scaler.pkl")
    save_json({"best_params": best_params, "valid_metrics": valid_metrics, "test_metrics": test_metrics}, model_dir / "metadata.json")

    pred_dir = output_dir / "predictions"
    ensure_dir(pred_dir)
    test_pred_df.to_csv(pred_dir / f"test_predictions_{lower_model_name}_{case_name}.csv", index=False, encoding="utf-8-sig")
    valid_pred_df.to_csv(pred_dir / f"valid_predictions_{lower_model_name}_{case_name}.csv", index=False, encoding="utf-8-sig")

    summary = {
        "model": model_name,
        "case": case_name,
        "feature_count": len(feature_cols),
        "seq_len": seq_len,
        "train_sequences": int(len(X_tr)),
        "valid_sequences": int(len(X_va)),
        "test_sequences": int(len(X_te)),
        "valid_objective_DAY_NMAE_pct": float(study.best_value),
        **{f"VALID_{k}": v for k, v in valid_metrics.items()},
        **{f"TEST_{k}": v for k, v in test_metrics.items()},
        "model_path": str(model_dir / "model.pt"),
        "scaler_path": str(model_dir / "scaler.pkl"),
    }
    return summary, test_pred_df


# ============================================================
# 7. Common timestamp comparison
# ============================================================
def build_common_timestamp_comparison(
    summaries: List[Dict],
    pred_dfs: Dict[Tuple[str, str], pd.DataFrame],
    output_dir: Path,
) -> None:
    if not summaries:
        return
    summary_df = pd.DataFrame(summaries)
    results_dir = output_dir / "results"
    ensure_dir(results_dir)

    # Best case per model selected by validation daytime NMAE.
    best_rows = []
    for model_name, g in summary_df.groupby("model"):
        g2 = g.copy()
        g2 = g2.sort_values("valid_objective_DAY_NMAE_pct")
        best_rows.append(g2.iloc[0].to_dict())
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(results_dir / "best_case_by_model_validation.csv", index=False, encoding="utf-8-sig")

    # Common timestamp among best cases only.
    time_sets = []
    selected = []
    for _, row in best_df.iterrows():
        key = (row["model"], row["case"])
        if key in pred_dfs:
            selected.append(key)
            time_sets.append(set(pd.to_datetime(pred_dfs[key][DATE_COL])))
    if len(time_sets) < 2:
        return

    common_times = set.intersection(*time_sets)
    common_times = sorted(common_times)
    if not common_times:
        print("[경고] 모델 간 공통 test timestamp가 없습니다.")
        return

    common_metrics = []
    common_pred_all = []
    for key in selected:
        model_name, case_name = key
        p = pred_dfs[key].copy()
        p[DATE_COL] = pd.to_datetime(p[DATE_COL])
        p_common = p[p[DATE_COL].isin(common_times)].copy()
        metrics = evaluate_prediction_df(p_common)
        row = {
            "model": model_name,
            "case": case_name,
            "common_test_timestamps": len(common_times),
            **metrics,
        }
        common_metrics.append(row)
        common_pred_all.append(p_common)

    pd.DataFrame(common_metrics).to_csv(
        results_dir / "common_timestamp_best_models_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(common_pred_all, axis=0).to_csv(
        results_dir / "common_timestamp_best_models_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Daytime-only main comparison table
    main_cols = [
        "model", "case", "DAY_n_samples", "DAY_MAE_MWh", "DAY_RMSE_MWh",
        "DAY_NMAE_pct", "DAY_NRMSE_pct", "DAY_MBE_MWh", "DAY_R2",
    ]
    cm_df = pd.DataFrame(common_metrics)
    cm_df[[c for c in main_cols if c in cm_df.columns]].to_csv(
        results_dir / "MAIN_common_timestamp_daytime_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 8. Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="chungbuk_pv_features.csv", help="Gap-safe feature CSV path")
    p.add_argument("--output_dir", type=str, default="pv_training_outputs")
    p.add_argument("--models", type=str, default="lightgbm,lstm,patchtst", help="Comma-separated: lightgbm,lstm,patchtst")
    p.add_argument("--n_trials", type=int, default=50, help="Optuna trials per model/case")
    p.add_argument("--epochs", type=int, default=200, help="Max epochs for LSTM/PatchTST per trial")
    p.add_argument("--patience", type=int, default=25, help="Early stopping patience for neural models")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda")
    p.add_argument("--no_clip", action="store_true", help="Do not clip negative predictions")
    p.add_argument("--quick_test", action="store_true", help="Run very small trials/epochs for code check")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    ensure_dir(output_dir / "results")

    if args.quick_test:
        args.n_trials = min(args.n_trials, 2)
        args.epochs = min(args.epochs, 3)
        args.patience = min(args.patience, 2)

    if args.device == "auto":
        device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    clip_predictions = not args.no_clip
    requested_models = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    df = read_feature_csv(args.csv)
    groups = build_feature_groups(df)
    cases = build_feature_cases(groups)
    train_df, valid_df, test_df = split_by_year(df)

    # Save experiment config
    save_json({
        "input_csv": args.csv,
        "output_dir": str(output_dir),
        "target": TARGET_COL,
        "excluded_features": EXCLUDE_FEATURE_COLS,
        "feature_groups": groups,
        "feature_cases": cases,
        "split": {"train": "2020-2022", "valid": "2023", "test": "2024"},
        "main_result": "daytime where 대기권밖일사량계산값 > 0",
        "common_timestamp_comparison": "best case per model selected by validation daytime NMAE",
        "n_trials": args.n_trials,
        "epochs": args.epochs,
        "patience": args.patience,
        "device": device,
        "clip_predictions": clip_predictions,
    }, output_dir / "experiment_config.json")

    print("=" * 80)
    print("[데이터 분할]")
    print("Train:", train_df[DATE_COL].min(), "~", train_df[DATE_COL].max(), len(train_df))
    print("Valid:", valid_df[DATE_COL].min(), "~", valid_df[DATE_COL].max(), len(valid_df))
    print("Test :", test_df[DATE_COL].min(), "~", test_df[DATE_COL].max(), len(test_df))
    print("Device:", device)
    print("Requested models:", requested_models)

    summaries: List[Dict] = []
    pred_dfs: Dict[Tuple[str, str], pd.DataFrame] = {}

    for case_name, feature_cols in cases.items():
        print("\n" + "=" * 80)
        print(f"[Case] {case_name} / feature_count={len(feature_cols)}")

        if "lightgbm" in requested_models:
            print(f"[LightGBM] {case_name} training...")
            summary, pred_df = train_lightgbm_case(
                case_name,
                feature_cols,
                train_df,
                valid_df,
                test_df,
                output_dir,
                n_trials=args.n_trials,
                seed=args.seed,
                clip_predictions=clip_predictions,
            )
            summaries.append(summary)
            pred_dfs[("LightGBM", case_name)] = pred_df
            pd.DataFrame(summaries).to_csv(output_dir / "results" / "summary_results.csv", index=False, encoding="utf-8-sig")

        if "lstm" in requested_models:
            print(f"[LSTM] {case_name} training...")
            summary, pred_df = train_sequence_case(
                "LSTM",
                case_name,
                feature_cols,
                train_df,
                valid_df,
                test_df,
                output_dir,
                n_trials=args.n_trials,
                seed=args.seed,
                max_epochs=args.epochs,
                patience=args.patience,
                device=device,
                clip_predictions=clip_predictions,
            )
            summaries.append(summary)
            pred_dfs[("LSTM", case_name)] = pred_df
            pd.DataFrame(summaries).to_csv(output_dir / "results" / "summary_results.csv", index=False, encoding="utf-8-sig")

        if "patchtst" in requested_models:
            print(f"[PatchTST] {case_name} training...")
            summary, pred_df = train_sequence_case(
                "PatchTST",
                case_name,
                feature_cols,
                train_df,
                valid_df,
                test_df,
                output_dir,
                n_trials=args.n_trials,
                seed=args.seed,
                max_epochs=args.epochs,
                patience=args.patience,
                device=device,
                clip_predictions=clip_predictions,
            )
            summaries.append(summary)
            pred_dfs[("PatchTST", case_name)] = pred_df
            pd.DataFrame(summaries).to_csv(output_dir / "results" / "summary_results.csv", index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "results" / "summary_results.csv", index=False, encoding="utf-8-sig")

    # Main result tables
    if len(summary_df) > 0:
        day_cols = [
            "model", "case", "feature_count", "seq_len",
            "TEST_DAY_n_samples", "TEST_DAY_MAE_MWh", "TEST_DAY_RMSE_MWh",
            "TEST_DAY_NMAE_pct", "TEST_DAY_NRMSE_pct", "TEST_DAY_MBE_MWh", "TEST_DAY_R2",
            "model_path",
        ]
        existing = [c for c in day_cols if c in summary_df.columns]
        summary_df[existing].to_csv(output_dir / "results" / "MAIN_individual_daytime_results.csv", index=False, encoding="utf-8-sig")

    build_common_timestamp_comparison(summaries, pred_dfs, output_dir)

    print("\n" + "=" * 80)
    print("[완료]")
    print("결과 폴더:", output_dir)
    print("모델 저장 폴더:", output_dir / "models")
    print("성능표:", output_dir / "results")


if __name__ == "__main__":
    main()
