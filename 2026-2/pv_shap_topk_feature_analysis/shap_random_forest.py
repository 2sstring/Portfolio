# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from shap_common import (
    DATE_COL,
    ensure_dir,
    importance_from_tree_shap,
    inspect_tree_bundle,
    rows_for_times,
    save_model_outputs_common,
)


def run_random_forest_shap(
    model_path: str | Path,
    valid_df: pd.DataFrame,
    target_times,
    output_dir: str | Path,
    top_n: int = 20,
):
    out = Path(output_dir)
    ensure_dir(out)

    bundle = inspect_tree_bundle(model_path, expected_type="random_forest")
    model = bundle["model"]
    feature_cols = list(bundle["feature_cols"])
    X = rows_for_times(valid_df, feature_cols, target_times)

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]

    importance = importance_from_tree_shap(values, feature_cols, "RandomForest")

    np.save(out / "shap_values_raw.npy", values.astype(np.float32))
    input_out = X.copy()
    input_out.insert(0, DATE_COL, pd.DatetimeIndex(target_times))
    input_out.to_csv(out / "shap_input.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({DATE_COL: pd.DatetimeIndex(target_times)}).to_csv(
        out / "shap_sample_timestamps.csv", index=False, encoding="utf-8-sig"
    )
    save_model_outputs_common(importance, out, top_n=top_n)
    return importance
