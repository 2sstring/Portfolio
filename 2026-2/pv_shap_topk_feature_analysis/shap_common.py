# -*- coding: utf-8 -*-
"""Common utilities for LightGBM, Random Forest, LSTM and TCN SHAP.

Research protocol
-----------------
- No model retraining is performed here.
- Existing C6_Full (69-feature) models are loaded.
- SHAP is computed on common 2023 validation daytime target timestamps.
- Tree models: TreeExplainer.
- LSTM/TCN: GradientExplainer (Expected Gradients).
- Deep-model SHAP values [N, T, F] are aggregated to feature level by
  averaging, over samples, the temporal sum of absolute SHAP values:
      I_j = (1/N) * sum_i sum_t |phi_i,t,j|
- Cross-model comparison uses normalized feature importance (%).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


DATE_COL = "발전일자"
SEGMENT_COL = "segment_id"
TARGET_COL = "정규화발전량"
CAPACITY_COL = "설비용량(MW)"
POWER_COL = "발전량(MWh)"
DAYTIME_COL = "대기권밖일사량계산값"
YEAR_COL = "year"

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


def setup_plot_font() -> None:
    """Use an installed Korean-capable font when available."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]:
        if candidate in installed:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams["axes.unicode_minus"] = False


setup_plot_font()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    if torch is None:
        return "cpu"
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda를 지정했지만 CUDA를 사용할 수 없습니다.")
    if device not in {"cpu", "cuda"}:
        raise ValueError("--device는 auto, cpu, cuda 중 하나여야 합니다.")
    return device


def read_feature_csv(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = [DATE_COL, TARGET_COL, DAYTIME_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"입력 CSV에 필수 컬럼이 없습니다: {missing}")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    if df[DATE_COL].duplicated().any():
        n_dup = int(df[DATE_COL].duplicated().sum())
        raise ValueError(f"'{DATE_COL}'에 중복 시점이 {n_dup}개 있습니다.")
    return df.sort_values(DATE_COL).reset_index(drop=True)


def split_by_year(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    year = df[DATE_COL].dt.year
    train_df = df[(year >= 2020) & (year <= 2022)].copy()
    valid_df = df[year == 2023].copy()
    test_df = df[year == 2024].copy()
    for name, part in [("Train", train_df), ("Validation", valid_df), ("Test", test_df)]:
        if part.empty:
            raise ValueError(f"{name} 데이터가 비어 있습니다. 연도 분할을 확인하세요.")
    return train_df, valid_df, test_df


def feature_group(feature: str) -> str:
    if feature in G1_TIME_BASE:
        return "G1_Time"
    if feature in G2_CURRENT_BASE:
        return "G2_Current"
    if "_lag_" in feature:
        return "G3_Lag"
    if "_rollmean_" in feature:
        return "G4_Rolling"
    if "_diff_" in feature:
        return "G5_Diff"
    return "Other"


def recompute_seq_segment_id(df: pd.DataFrame) -> pd.Series:
    """Same gap-safe segmentation rule as the training code."""
    work = df.sort_values(DATE_COL).reset_index(drop=True)
    time_diff = work[DATE_COL].diff()
    new_segment = time_diff != pd.Timedelta(hours=1)
    if len(new_segment):
        new_segment.iloc[0] = True

    if SEGMENT_COL in work.columns:
        original_change = work[SEGMENT_COL].diff().fillna(0) != 0
        new_segment = new_segment | original_change

    return new_segment.cumsum()


def _segmented_work(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work[DATE_COL] = pd.to_datetime(work[DATE_COL])
    work = work.sort_values(DATE_COL).reset_index(drop=True)
    work["seq_segment_id"] = recompute_seq_segment_id(work)
    return work


def tree_available_times(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DatetimeIndex:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV에 모델 입력 특징이 없습니다: {missing[:10]}")
    mask = _finite_rows(df, list(feature_cols) + [TARGET_COL])
    return pd.DatetimeIndex(pd.to_datetime(df.loc[mask, DATE_COL]))


def sequence_available_times(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    seq_len: int,
) -> pd.DatetimeIndex:
    """Return target timestamps having a valid preceding seq_len-hour input window.

    Training convention:
      X = rows [target_idx-seq_len, ..., target_idx-1]
      y = row target_idx
    """
    if seq_len <= 0:
        raise ValueError("seq_len은 1 이상이어야 합니다.")

    work = _segmented_work(df)
    out: List[pd.Timestamp] = []

    for _, seg in work.groupby("seq_segment_id", sort=False):
        seg = seg.sort_values(DATE_COL).reset_index(drop=True)
        if len(seg) < seq_len + 1:
            continue

        feature_ok = pd.Series(_finite_rows(seg, feature_cols).astype(int))
        # At target row t, check preceding seq_len rows only (exclude target row).
        prev_window_count = feature_ok.rolling(seq_len).sum().shift(1)
        target_ok = pd.Series(_finite_rows(seg, [TARGET_COL]))
        ok = prev_window_count.eq(seq_len) & target_ok
        if ok.any():
            out.extend(pd.to_datetime(seg.loc[ok, DATE_COL]).tolist())

    return pd.DatetimeIndex(out)



def _finite_rows(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")
    return np.isfinite(df[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)).all(axis=1)


def tcn_available_times(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    seq_len: int,
    max_context: int,
) -> pd.DatetimeIndex:
    """Reproduce the TCN training eligibility rule (not just its lookback).

    Requires an unbroken max_context-hour history of finite feature values,
    and a finite target, observed MWh, positive capacity and daytime column.
    The model still receives only the last seq_len rows of that history.
    """
    if seq_len <= 0 or max_context < seq_len:
        raise ValueError("TCN sequence length/max_context가 올바르지 않습니다.")
    work = _segmented_work(df)
    required = [TARGET_COL, POWER_COL, CAPACITY_COL, DAYTIME_COL]
    input_ok = _finite_rows(work, feature_cols)
    target_ok = _finite_rows(work, required) & (work[CAPACITY_COL].to_numpy(float) > 0)
    result = []
    for _, seg in work.groupby("seq_segment_id", sort=False):
        positions = seg.index.to_numpy()
        if len(positions) <= max_context:
            continue
        bad = (~input_ok[positions]).astype(np.int64)
        prefix = np.concatenate(([0], np.cumsum(bad)))
        local = np.arange(max_context, len(positions))
        valid = ((prefix[local] - prefix[local-max_context]) == 0) & target_ok[positions[local]]
        result.extend(work.loc[positions[local[valid]], DATE_COL].tolist())
    return pd.DatetimeIndex(result)


def build_sequences_for_times(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    seq_len: int,
    scaler,
    target_times: Sequence[pd.Timestamp],
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Build only the requested gap-safe sequences, preserving target_times order."""
    requested = [pd.Timestamp(t) for t in target_times]
    requested_set = set(requested)
    work = _segmented_work(df)
    seq_by_time: Dict[pd.Timestamp, np.ndarray] = {}

    for _, seg in work.groupby("seq_segment_id", sort=False):
        seg = seg.sort_values(DATE_COL).reset_index(drop=True)
        seg_times = pd.to_datetime(seg[DATE_COL])
        candidate_positions = [i for i, t in enumerate(seg_times) if pd.Timestamp(t) in requested_set]
        if not candidate_positions:
            continue

        raw = seg[list(feature_cols)].astype(float).values
        for idx in candidate_positions:
            if idx < seq_len:
                continue
            x_raw = raw[idx - seq_len: idx]
            y_val = seg.iloc[idx][TARGET_COL]
            if not np.isfinite(x_raw).all() or not np.isfinite(float(y_val)):
                continue
            if scaler is not None:
                x_raw = scaler.transform(x_raw)
            if not np.isfinite(x_raw).all():
                raise ValueError("스케일링 후 NaN/Inf가 발생했습니다.")
            seq_by_time[pd.Timestamp(seg_times.iloc[idx])] = x_raw.astype(np.float32)

    missing_times = [t for t in requested if t not in seq_by_time]
    if missing_times:
        preview = ", ".join(str(t) for t in missing_times[:5])
        raise ValueError(
            f"요청한 sequence target timestamp 중 {len(missing_times)}개를 만들 수 없습니다. "
            f"예: {preview}"
        )

    X = np.stack([seq_by_time[t] for t in requested], axis=0).astype(np.float32)
    return X, pd.DatetimeIndex(requested)


def rows_for_times(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_times: Sequence[pd.Timestamp],
) -> pd.DataFrame:
    indexed = df.copy()
    indexed[DATE_COL] = pd.to_datetime(indexed[DATE_COL])
    indexed = indexed.set_index(DATE_COL)
    times = pd.DatetimeIndex([pd.Timestamp(t) for t in target_times])
    missing = times.difference(indexed.index)
    if len(missing):
        raise ValueError(f"CSV에 없는 timestamp가 있습니다: {list(missing[:5])}")
    X = indexed.loc[times, list(feature_cols)].copy()
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("선택된 공통 timestamp의 tree 입력에 NaN이 있습니다.")
    return X


def choose_common_timestamps(
    valid_df: pd.DataFrame,
    availability: Dict[str, Iterable[pd.Timestamp]],
    sample_size: int,
    seed: int,
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    if not availability:
        raise ValueError("availability가 비어 있습니다.")

    sets = [set(pd.Timestamp(t) for t in times) for times in availability.values()]
    common = set.intersection(*sets)
    daytime = set(
        pd.Timestamp(t)
        for t in pd.to_datetime(valid_df.loc[valid_df[DAYTIME_COL] > 0, DATE_COL])
    )
    common &= daytime
    common_sorted = pd.DatetimeIndex(sorted(common))
    if len(common_sorted) == 0:
        raise ValueError("4개 모델에 공통으로 존재하는 2023년 주간 SHAP timestamp가 없습니다.")

    if sample_size <= 0 or sample_size >= len(common_sorted):
        selected = common_sorted
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(common_sorted), size=sample_size, replace=False))
        selected = common_sorted[idx]
    return common_sorted, pd.DatetimeIndex(selected)


def sample_times(times: Sequence[pd.Timestamp], n: int, seed: int) -> pd.DatetimeIndex:
    times = pd.DatetimeIndex(sorted(set(pd.Timestamp(t) for t in times)))
    if len(times) == 0:
        raise ValueError("샘플링할 timestamp가 없습니다.")
    if n <= 0 or n >= len(times):
        return times
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(times), size=n, replace=False))
    return pd.DatetimeIndex(times[idx])


def normalize_shap_array(values, expected_ndim: int) -> np.ndarray:
    """Normalize SHAP return types to ndarray and remove singleton output dimension."""
    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError(f"단일 회귀 출력인데 SHAP list 길이가 {len(values)}입니다.")
        values = values[0]
    arr = np.asarray(values)
    if arr.ndim == expected_ndim + 1 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != expected_ndim:
        raise ValueError(f"예상 SHAP 차원={expected_ndim}, 실제 shape={arr.shape}")
    return arr


def importance_from_tree_shap(
    shap_values: np.ndarray,
    feature_cols: Sequence[str],
    model_name: str,
    explainer_name: str = "TreeExplainer",
) -> pd.DataFrame:
    arr = normalize_shap_array(shap_values, expected_ndim=2)
    if arr.shape[1] != len(feature_cols):
        raise ValueError("Tree SHAP feature 수가 feature_cols와 다릅니다.")
    mean_abs = np.mean(np.abs(arr), axis=0)
    mean_signed = np.mean(arr, axis=0)
    return build_importance_df(
        feature_cols=feature_cols,
        raw_importance=mean_abs,
        mean_signed=mean_signed,
        model_name=model_name,
        explainer_name=explainer_name,
        aggregation="mean over samples of |SHAP|",
    )


def importance_from_temporal_shap(
    shap_values: np.ndarray,
    feature_cols: Sequence[str],
    model_name: str,
    explainer_name: str = "GradientExplainer",
) -> Tuple[pd.DataFrame, np.ndarray]:
    arr = normalize_shap_array(shap_values, expected_ndim=3)
    if arr.shape[2] != len(feature_cols):
        raise ValueError("Temporal SHAP feature 수가 feature_cols와 다릅니다.")

    # Primary feature-level aggregation for paper comparison:
    # I_j = mean_i sum_t |phi_i,t,j|
    temporal_sum_abs_per_sample = np.sum(np.abs(arr), axis=1)  # [N, F]
    raw_importance = np.mean(temporal_sum_abs_per_sample, axis=0)

    # Signed temporal sum is saved only as a directional descriptive statistic.
    mean_signed = np.mean(np.sum(arr, axis=1), axis=0)

    # Heatmap matrix: mean across samples at each lag and feature [T, F].
    mean_abs_time_feature = np.mean(np.abs(arr), axis=0)

    imp = build_importance_df(
        feature_cols=feature_cols,
        raw_importance=raw_importance,
        mean_signed=mean_signed,
        model_name=model_name,
        explainer_name=explainer_name,
        aggregation="mean over samples of temporal sum |SHAP|",
    )
    return imp, mean_abs_time_feature


def build_importance_df(
    feature_cols: Sequence[str],
    raw_importance: np.ndarray,
    mean_signed: np.ndarray,
    model_name: str,
    explainer_name: str,
    aggregation: str,
) -> pd.DataFrame:
    raw_importance = np.asarray(raw_importance, dtype=float)
    mean_signed = np.asarray(mean_signed, dtype=float)
    if not np.isfinite(raw_importance).all() or (raw_importance < 0).any():
        raise ValueError("SHAP 중요도에 비정상 값이 있습니다.")
    total = float(np.sum(raw_importance))
    if total <= 0 or not np.isfinite(total):
        raise ValueError("SHAP 중요도 합계가 0 또는 비정상 값입니다.")
    normalized = raw_importance / total * 100.0

    out = pd.DataFrame({
        "model": model_name,
        "feature": list(feature_cols),
        "feature_group": [feature_group(f) for f in feature_cols],
        "mean_abs_shap": raw_importance,
        "mean_signed_shap": mean_signed,
        "normalized_importance_pct": normalized,
        "explainer": explainer_name,
        "aggregation": aggregation,
    })
    out = out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def save_group_importance(importance_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    group = (
        importance_df.groupby("feature_group", as_index=False)
        .agg(
            normalized_importance_pct=("normalized_importance_pct", "sum"),
            mean_abs_shap=("mean_abs_shap", "sum"),
            feature_count=("feature", "count"),
        )
        .sort_values("normalized_importance_pct", ascending=False)
        .reset_index(drop=True)
    )
    group.insert(0, "rank", np.arange(1, len(group) + 1))
    group.to_csv(out_path, index=False, encoding="utf-8-sig")
    return group


def save_top_bar(importance_df: pd.DataFrame, out_path: Path, top_n: int = 20) -> None:
    top = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(5, 0.34 * len(top) + 1.5)))
    ax.barh(top["feature"], top["normalized_importance_pct"])
    ax.set_xlabel("Normalized SHAP importance (%)")
    ax.set_ylabel("Feature")
    ax.set_title(f"{importance_df['model'].iloc[0]} SHAP Top {len(top)}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_temporal_heatmap(
    mean_abs_time_feature: np.ndarray,
    feature_cols: Sequence[str],
    importance_df: pd.DataFrame,
    out_csv: Path,
    out_png: Path,
    top_n: int = 15,
) -> None:
    seq_len = mean_abs_time_feature.shape[0]
    lags = np.arange(seq_len, 0, -1)  # first sequence row = t-seq_len; last = t-1
    matrix_df = pd.DataFrame(mean_abs_time_feature, columns=list(feature_cols))
    matrix_df.insert(0, "lag_hours", lags)
    matrix_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    top_features = importance_df.head(top_n)["feature"].tolist()
    feature_index = {f: i for i, f in enumerate(feature_cols)}
    cols = [feature_index[f] for f in top_features]
    heat = mean_abs_time_feature[:, cols].T

    fig, ax = plt.subplots(figsize=(12, max(5, 0.42 * len(top_features) + 2)))
    im = ax.imshow(heat, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(top_features)))
    ax.set_yticklabels(top_features)

    # Avoid an unreadable x-axis for long lookback windows.
    n_ticks = min(10, seq_len)
    tick_idx = np.unique(np.linspace(0, seq_len - 1, n_ticks, dtype=int))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"t-{lags[i]}" for i in tick_idx], rotation=45, ha="right")
    ax.set_xlabel("Input lag")
    ax.set_ylabel("Feature")
    ax.set_title("Mean |SHAP| by feature and time lag")
    fig.colorbar(im, ax=ax, label="Mean |SHAP|")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# PyTorch model definitions copied exactly from the supplied training logic.
# ---------------------------------------------------------------------------
if nn is not None:
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


    class SingleOutput2DWrapper(nn.Module):
        """SHAP GradientExplainer expects PyTorch outputs shaped [B, outputs].

        The trained regressors intentionally return [B]. This wrapper changes only
        the SHAP-facing output shape to [B, 1]; prediction values are unchanged.
        """
        def __init__(self, base_model: nn.Module):
            super().__init__()
            self.base_model = base_model

        def forward(self, x):
            y = self.base_model(x)
            if y.ndim == 1:
                y = y.unsqueeze(-1)
            return y
else:
    LSTMRegressor = None
    SingleOutput2DWrapper = None


def load_torch_checkpoint(path: str | Path):
    if torch is None:
        raise ImportError("PyTorch가 필요합니다: pip install torch")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PyTorch 모델 파일이 없습니다: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def inspect_tree_bundle(path: str | Path, expected_type: str | None = None) -> Dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tree 모델 파일이 없습니다: {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle or "feature_cols" not in bundle:
        raise ValueError(f"학습 코드에서 저장한 bundle 형식이 아닙니다: {path}")
    if expected_type and str(bundle.get("model_type", "")).lower() != expected_type.lower():
        raise ValueError(
            f"모델 형식 불일치: 기대={expected_type}, 실제={bundle.get('model_type')} ({path})"
        )
    return bundle


def inspect_torch_bundle(path: str | Path, expected_type: str | None = None) -> Dict:
    bundle = load_torch_checkpoint(path)
    if not isinstance(bundle, dict) or "state_dict" not in bundle or "feature_cols" not in bundle:
        raise ValueError(f"학습 코드에서 저장한 PyTorch bundle 형식이 아닙니다: {path}")
    if expected_type and str(bundle.get("model_type", "")).lower() != expected_type.lower():
        raise ValueError(
            f"모델 형식 불일치: 기대={expected_type}, 실제={bundle.get('model_type')} ({path})"
        )
    return bundle


def build_torch_model_from_bundle(bundle: Dict):
    if torch is None or nn is None:
        raise ImportError("PyTorch가 필요합니다: pip install torch")
    model_type = str(bundle.get("model_type", "")).lower()
    arch = bundle.get("architecture", {})

    if model_type == "lstm":
        model = LSTMRegressor(
            input_dim=int(arch["input_dim"]),
            hidden_size=int(arch["hidden_size"]),
            num_layers=int(arch["num_layers"]),
            dropout=float(arch["dropout"]),
        )
    elif model_type == "tcn":
        from tcn_model import build_tcn_model_from_bundle
        return build_tcn_model_from_bundle(bundle)
    else:
        raise ValueError(f"지원하지 않는 PyTorch model_type: {model_type}")

    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model


def check_feature_sets(
    model_features: Dict[str, Sequence[str]],
    expected_feature_count: int = 69,
) -> List[str]:
    """Require the four models to use the same feature names (order may differ)."""
    if not model_features:
        raise ValueError("model_features가 비어 있습니다.")

    details = {name: list(cols) for name, cols in model_features.items()}
    for name, cols in details.items():
        if expected_feature_count > 0 and len(cols) != expected_feature_count:
            raise ValueError(
                f"{name} feature_count={len(cols)}입니다. 예상={expected_feature_count}. "
                "69개 C6_Full 모델 파일인지 확인하세요."
            )
        if len(cols) != len(set(cols)):
            raise ValueError(f"{name} feature_cols에 중복 특징이 있습니다.")

    first_name = next(iter(details))
    first_set = set(details[first_name])
    mismatches = {}
    for name, cols in details.items():
        s = set(cols)
        if s != first_set:
            mismatches[name] = {
                "missing_vs_reference": sorted(first_set - s),
                "extra_vs_reference": sorted(s - first_set),
            }
    if mismatches:
        raise ValueError(
            "4개 모델의 feature 이름 집합이 동일하지 않습니다. "
            + json.dumps(mismatches, ensure_ascii=False)
        )
    return details[first_name]


def save_model_outputs_common(
    importance_df: pd.DataFrame,
    model_dir: Path,
    top_n: int = 20,
) -> None:
    ensure_dir(model_dir)
    importance_df.to_csv(
        model_dir / "shap_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_group_importance(importance_df, model_dir / "shap_group_importance.csv")
    save_top_bar(importance_df, model_dir / "shap_top20_bar.png", top_n=top_n)


def compare_model_importance(
    importance_by_model: Dict[str, pd.DataFrame],
    output_dir: Path,
    top_k: int = 10,
) -> Dict[str, Path]:
    ensure_dir(output_dir)
    model_order = ["RandomForest", "LightGBM", "LSTM", "TCN"]
    model_order = [m for m in model_order if m in importance_by_model]
    if top_k < 1:
        raise ValueError("top_k는 1 이상이어야 합니다.")

    reference_set = set(importance_by_model[model_order[0]]["feature"])
    for name in model_order:
        df = importance_by_model[name]
        if set(df["feature"]) != reference_set or df["feature"].duplicated().any():
            raise ValueError(f"{name} 특징 집합이 다릅니다.")
    # Long format
    long_df = pd.concat(
        [importance_by_model[m].copy() for m in model_order],
        ignore_index=True,
    )
    long_path = output_dir / "all_models_feature_importance_long.csv"
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")

    # Wide format
    base_features = importance_by_model[model_order[0]][["feature", "feature_group"]].copy()
    wide = base_features.copy()
    rank_only = pd.DataFrame({"feature": base_features["feature"]})
    norm_only = pd.DataFrame({"feature": base_features["feature"]})

    for model_name in model_order:
        part = importance_by_model[model_name][
            ["feature", "rank", "mean_abs_shap", "normalized_importance_pct"]
        ].copy()
        part = part.rename(columns={
            "rank": f"{model_name}_rank",
            "mean_abs_shap": f"{model_name}_mean_abs_shap",
            "normalized_importance_pct": f"{model_name}_normalized_importance_pct",
        })
        wide = wide.merge(part, on="feature", how="inner")
        rank_only = rank_only.merge(
            part[["feature", f"{model_name}_rank"]], on="feature", how="inner"
        )
        norm_only = norm_only.merge(
            part[["feature", f"{model_name}_normalized_importance_pct"]],
            on="feature",
            how="inner",
        )

    wide_path = output_dir / "all_models_feature_importance_wide.csv"
    rank_path = output_dir / "all_models_feature_rank.csv"
    norm_path = output_dir / "all_models_normalized_importance.csv"
    wide.to_csv(wide_path, index=False, encoding="utf-8-sig")
    rank_only.to_csv(rank_path, index=False, encoding="utf-8-sig")
    norm_only.to_csv(norm_path, index=False, encoding="utf-8-sig")

    # Spearman uses average ranks for ties, not arbitrary sequential ranks.
    # The displayed rank is a stable ordinal ordering for Top-K selection.
    score_cols = [f"{m}_normalized_importance_pct" for m in model_order]
    spearman = norm_only.set_index("feature")[score_cols].corr(method="spearman")
    spearman.index = model_order
    spearman.columns = model_order
    spearman_path = output_dir / "spearman_rank_correlation.csv"
    spearman.to_csv(spearman_path, encoding="utf-8-sig")

    # Top-k side-by-side table.
    topk_data = {"rank": np.arange(1, top_k + 1)}
    top_sets = {}
    for model_name in model_order:
        top = importance_by_model[model_name].head(top_k)["feature"].tolist()
        if len(top) < top_k:
            top += [None] * (top_k - len(top))
        topk_data[model_name] = top
        top_sets[model_name] = set(x for x in top if x is not None)
    topk_df = pd.DataFrame(topk_data)
    topk_path = output_dir / f"top{top_k}_features_by_model.csv"
    topk_df.to_csv(topk_path, index=False, encoding="utf-8-sig")

    # Pairwise Top-k overlap count and Jaccard.
    overlap_count = pd.DataFrame(index=model_order, columns=model_order, dtype=float)
    overlap_jaccard = pd.DataFrame(index=model_order, columns=model_order, dtype=float)
    for a in model_order:
        for b in model_order:
            inter = top_sets[a] & top_sets[b]
            union = top_sets[a] | top_sets[b]
            overlap_count.loc[a, b] = len(inter)
            overlap_jaccard.loc[a, b] = len(inter) / len(union) if union else np.nan
    overlap_count_path = output_dir / f"top{top_k}_overlap_count.csv"
    overlap_jaccard_path = output_dir / f"top{top_k}_overlap_jaccard.csv"
    overlap_count.to_csv(overlap_count_path, encoding="utf-8-sig")
    overlap_jaccard.to_csv(overlap_jaccard_path, encoding="utf-8-sig")

    # Consensus descriptive table (not used as feature selection unless researcher chooses so later).
    consensus = base_features.copy()
    rank_matrix = rank_only.set_index("feature")
    norm_matrix = norm_only.set_index("feature")
    consensus["mean_rank"] = consensus["feature"].map(rank_matrix.mean(axis=1))
    consensus["mean_normalized_importance_pct"] = consensus["feature"].map(norm_matrix.mean(axis=1))
    consensus[f"models_in_top{top_k}"] = consensus["feature"].map(
        lambda f: sum(f in top_sets[m] for m in model_order)
    )
    consensus = consensus.sort_values(
        [f"models_in_top{top_k}", "mean_rank", "mean_normalized_importance_pct"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    consensus.insert(0, "consensus_order", np.arange(1, len(consensus) + 1))
    consensus_path = output_dir / "consensus_descriptive_summary.csv"
    consensus.to_csv(consensus_path, index=False, encoding="utf-8-sig")

    # Comparison plot: top 20 by average normalized importance.
    plot_df = norm_only.set_index("feature")
    plot_df.columns = model_order
    plot_df["mean"] = plot_df.mean(axis=1)
    selected = plot_df.sort_values("mean", ascending=False).head(20).drop(columns="mean")
    selected = selected.iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 9))
    selected.plot(kind="barh", ax=ax)
    ax.set_xlabel("Normalized SHAP importance (%)")
    ax.set_ylabel("Feature")
    ax.set_title(f"{len(model_order)}-model normalized SHAP importance comparison (Top 20 by mean)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    comparison_png = output_dir / "normalized_importance_comparison_top20.png"
    fig.savefig(comparison_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "long": long_path,
        "wide": wide_path,
        "rank": rank_path,
        "normalized": norm_path,
        "spearman": spearman_path,
        "topk": topk_path,
        "overlap_count": overlap_count_path,
        "overlap_jaccard": overlap_jaccard_path,
        "consensus": consensus_path,
        "comparison_png": comparison_png,
    }
