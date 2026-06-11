# -*- coding: utf-8 -*-
"""
05_exp2_shap_analysis.py

실험 2. SHAP 기반 중요 feature 분석
------------------------------------------------------------
목적
- 이미 선정된 LightGBM 최적 파라미터를 사용하여 C6_Full 모델을 학습하고,
- validation 구간을 기준으로 SHAP 중요도를 분석한다.
- 중요한 개별 변수와 feature group 확인

입력
- gap-safe feature CSV: chungbuk_pv_features.csv

출력 예시
pv_exp2_outputs/
├─ experiment_config.json
├─ experiment_2_shap/
│  ├─ shap_feature_importance.csv
│  ├─ shap_group_importance.csv
│  ├─ shap_values_valid_sample.npy
│  ├─ shap_input_valid_sample.csv
│  ├─ shap_top30_bar.png
│  └─ shap_beeswarm_top30.png
├─ models/lightgbm/C6_Full_SHAP_Base/
├─ predictions/
└─ results/
   ├─ exp2_c6_lightgbm_summary.csv
   ├─ exp2_c6_features.csv
   └─ exp2_top_features_preview.csv

실행 예시
python 05_exp2_shap_analysis.py ^
  --csv chungbuk_pv_features.csv ^
  --output_dir pv_exp2_outputs ^
  --shap_sample_size 5000

빠른 코드 점검
python 05_exp2_shap_analysis.py --csv chungbuk_pv_features.csv --quick_test
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pv_exp_common import (  # noqa: E402
    DATE_COL,
    TARGET_COL,
    ensure_dir,
    save_json,
    set_seed,
    read_feature_csv,
    build_feature_groups,
    build_feature_cases,
    split_by_year,
    train_lightgbm_config,
    compute_shap_importance,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="chungbuk_pv_features.csv", help="gap-safe feature CSV path")
    p.add_argument("--output_dir", type=str, default="pv_exp2_outputs")
    p.add_argument("--n_trials_lgb", type=int, default=50)
    p.add_argument("--shap_sample_size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_clip", action="store_true", help="negative prediction clipping 해제")
    p.add_argument("--quick_test", action="store_true", help="trial/sample 수를 줄여 코드만 빠르게 점검")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.quick_test:
        args.n_trials_lgb = min(args.n_trials_lgb, 2)
        args.shap_sample_size = min(args.shap_sample_size, 300)

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    ensure_dir(output_dir / "results")

    df = read_feature_csv(args.csv)
    groups = build_feature_groups(df)
    cases = build_feature_cases(groups)
    c6_features = cases["C6_Full"]
    if len(c6_features) == 0:
        raise ValueError("C6_Full feature가 비어 있습니다. feature 생성 CSV를 확인하세요.")

    train_df, valid_df, test_df = split_by_year(df)
    clip_predictions = not args.no_clip

    best_params = {
        "n_estimators": 2318,
        "learning_rate": 0.005388666432283696,
        "num_leaves": 156,
        "max_depth": 9,
        "min_child_samples": 18,
        "subsample": 0.9627296416052418,
        "colsample_bytree": 0.8569045259108246,
        "reg_alpha": 0.0002194392663239669,
        "reg_lambda": 2.1125325708158145e-07,
    }

    save_json(
        {
            "experiment": "Experiment 2 - C6_Full LightGBM + SHAP",
            "input_csv": args.csv,
            "output_dir": str(output_dir),
            "target": TARGET_COL,
            "split": {"train": "2020-2022", "valid": "2023", "test": "2024"},
            "main_metric": "DAY_NMAE_pct",
            "shap_period": "validation period, sampled if too large",
            "n_trials_lgb": args.n_trials_lgb,
            "shap_sample_size": args.shap_sample_size,
            "seed": args.seed,
            "clip_predictions": clip_predictions,
            "feature_groups": groups,
            "c6_feature_count": len(c6_features),
        },
        output_dir / "experiment_config.json",
    )

    pd.DataFrame({"feature": c6_features}).to_csv(
        output_dir / "results" / "exp2_c6_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 90)
    print("[실험 2] C6_Full LightGBM 학습 및 SHAP 분석")
    print("Train:", train_df[DATE_COL].min(), "~", train_df[DATE_COL].max(), len(train_df))
    print("Valid:", valid_df[DATE_COL].min(), "~", valid_df[DATE_COL].max(), len(valid_df))
    print("Test :", test_df[DATE_COL].min(), "~", test_df[DATE_COL].max(), len(test_df))
    print("C6_Full feature count:", len(c6_features))

    summary, _test_pred_df, model = train_lightgbm_config(
        config_name="C6_Full_SHAP_Base",
        feature_cols=c6_features,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        output_dir=output_dir,
        n_trials=args.n_trials_lgb,
        seed=args.seed,
        clip_predictions=clip_predictions,
        fixed_params=best_params, 
    )
    pd.DataFrame([summary]).to_csv(
        output_dir / "results" / "exp2_c6_lightgbm_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    shap_rank = compute_shap_importance(
        model=model,
        feature_cols=c6_features,
        train_df=train_df,
        valid_df=valid_df,
        output_dir=output_dir,
        sample_size=args.shap_sample_size,
        seed=args.seed,
    )

    preview = shap_rank.head(30)[["rank", "feature", "feature_group", "mean_abs_shap", "mean_signed_shap"]]
    preview.to_csv(output_dir / "results" / "exp2_top_features_preview.csv", index=False, encoding="utf-8-sig")

    print("\n[SHAP Top 30]")
    print(preview.to_string(index=False))
    print("\n[완료]")
    print("SHAP 중요도:", output_dir / "experiment_2_shap" / "shap_feature_importance.csv")
    print("그룹 중요도:", output_dir / "experiment_2_shap" / "shap_group_importance.csv")
    print("요약 결과:", output_dir / "results" / "exp2_c6_lightgbm_summary.csv")


if __name__ == "__main__":
    main()
