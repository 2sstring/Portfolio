# -*- coding: utf-8 -*-
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import torch

from shap_common import (
    DATE_COL,
    build_sequences_for_times,
    build_torch_model_from_bundle,
    ensure_dir,
    importance_from_temporal_shap,
    inspect_torch_bundle,
    sample_times,
    save_model_outputs_common,
    save_temporal_heatmap,
    sequence_available_times,
    SingleOutput2DWrapper,
)


def _gradient_shap_in_chunks(explainer, X_tensor, nsamples: int, chunk_size: int, seed: int):
    chunks = []
    for start in range(0, len(X_tensor), chunk_size):
        end = min(start + chunk_size, len(X_tensor))
        vals = explainer.shap_values(
            X_tensor[start:end],
            nsamples=nsamples,
            rseed=seed + start,
        )
        if isinstance(vals, list):
            vals = vals[0]
        vals = np.asarray(vals)
        if vals.ndim == 4 and vals.shape[-1] == 1:
            vals = vals[..., 0]
        chunks.append(vals.astype(np.float32))
    return np.concatenate(chunks, axis=0)


def run_lstm_shap(
    model_path: str | Path,
    scaler_path: str | Path,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target_times,
    output_dir: str | Path,
    device: str,
    background_size: int = 100,
    background_times=None,
    gradient_nsamples: int = 200,
    deep_chunk_size: int = 20,
    seed: int = 42,
    top_n: int = 20,
):
    out = Path(output_dir)
    ensure_dir(out)

    bundle = inspect_torch_bundle(model_path, expected_type="lstm")
    feature_cols = list(bundle["feature_cols"])
    seq_len = int(bundle["best_params"]["seq_len"])
    scaler = joblib.load(scaler_path)

    base_model = build_torch_model_from_bundle(bundle)
    model = SingleOutput2DWrapper(base_model).to(device)
    model.eval()

    X_valid, used_times = build_sequences_for_times(
        valid_df, feature_cols, seq_len, scaler, target_times
    )

    if background_times is None:
        train_available = sequence_available_times(train_df, feature_cols, seq_len)
        bg_times = sample_times(train_available, background_size, seed + 101)
    else:
        bg_times = pd.DatetimeIndex([pd.Timestamp(t) for t in background_times])
    X_bg, bg_times = build_sequences_for_times(
        train_df, feature_cols, seq_len, scaler, bg_times
    )

    background = torch.tensor(X_bg, dtype=torch.float32, device=device)
    explain_x = torch.tensor(X_valid, dtype=torch.float32, device=device)
    with torch.backends.cudnn.flags(enabled=False):
        explainer = shap.GradientExplainer(model, background)
        values = _gradient_shap_in_chunks(
            explainer,
            explain_x,
            nsamples=gradient_nsamples,
            chunk_size=deep_chunk_size,
            seed=seed,
        )

    importance, time_feature = importance_from_temporal_shap(
        values, feature_cols, "LSTM"
    )

    np.save(out / "shap_values_raw.npy", values.astype(np.float32))
    np.save(out / "shap_input_sequences.npy", X_valid.astype(np.float32))
    pd.DataFrame({DATE_COL: used_times}).to_csv(
        out / "shap_sample_timestamps.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({DATE_COL: bg_times}).to_csv(
        out / "shap_background_timestamps.csv", index=False, encoding="utf-8-sig"
    )
    save_model_outputs_common(importance, out, top_n=top_n)
    save_temporal_heatmap(
        time_feature,
        feature_cols,
        importance,
        out / "shap_time_feature_mean_abs.csv",
        out / "shap_top15_time_heatmap.png",
        top_n=min(15, top_n),
    )
    return importance
