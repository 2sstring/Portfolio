# -*- coding: utf-8 -*-
"""
06_exp3_shap_corr_retrain.py

실험 3. SHAP + 상관관계 필터링 Top-K feature set별 전체 모델 재학습
------------------------------------------------------------------------
목적
- 실험 2에서 생성한 SHAP 중요도 순위를 사용한다.
- train 구간(2020~2022년)에서 feature-feature 상관관계를 계산한다.
- SHAP 순위가 높은 feature를 우선 보존하고, 상관관계가 높은 후순위 feature를 제외한다.
- 상관관계 필터링 후 남은 후보군에서 Top-K feature set을 구성한다.
- 기본 설정에서는 LightGBM만 Top1~Top58 feature set별로 재학습한다.
- LSTM, PatchTST도 함께 학습하려면 --models lightgbm,lstm,patchtst로 지정한다.

기본 조건
- corr_threshold = 0.95
- corr_method = pearson
- n_trials_lgb = 50
- n_trials_seq = 50
- epochs = 200
- patience = 25
- objective = validation DAY_NMAE_pct 최소화

PyCharm 실행 전제
- 이 파일, pv_exp_common.py, chungbuk_pv_features.csv가 같은 폴더에 있으면 Run 버튼만 눌러도 실행된다.
- 단, 실험 2 결과인 pv_exp2_outputs/experiment_2_shap/shap_feature_importance.csv가 먼저 존재해야 한다.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pv_exp_common import (  # noqa: E402
    DATE_COL,
    TARGET_COL,
    ensure_dir,
    save_json,
    parse_int_list,
    set_seed,
    read_feature_csv,
    split_by_year,
    load_shap_rank,
    train_lightgbm_config,
    train_sequence_config,
    build_common_timestamp_metrics,
    save_summary_csv,
    validate_features_exist,
    torch,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="chungbuk_pv_features.csv", help="gap-safe feature CSV path")
    p.add_argument("--shap_importance", type=str, default="pv_exp2_outputs/experiment_2_shap/shap_feature_importance.csv")
    p.add_argument("--output_dir", type=str, default="pv_exp3_outputs_LGM")
    p.add_argument("--models", type=str, default="lightgbm", help="학습할 모델 목록")
    p.add_argument("--topk_list", type=str, default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58")
    p.add_argument("--corr_threshold", type=float, default=0.95)
    p.add_argument("--corr_method", type=str, default="pearson", choices=["pearson", "spearman"])
    p.add_argument("--n_trials_lgb", type=int, default=50)
    p.add_argument("--n_trials_seq", type=int, default=50)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda")
    p.add_argument("--no_clip", action="store_true")
    p.add_argument("--quick_test", action="store_true", help="논문용 아님. 코드 동작 확인용으로 trial/epoch 축소")
    return p.parse_args()


def make_corr_filtered_shap_candidates(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    shap_rank: pd.DataFrame,
    corr_threshold: float,
    corr_method: str,
    output_dir: Path,
) -> pd.DataFrame:
    results_dir = output_dir / "results"
    ensure_dir(results_dir)

    available = [f for f in shap_rank["feature"].tolist() if f in df.columns]
    work = train_df[available + [TARGET_COL]].copy()
    work = work.apply(pd.to_numeric, errors="coerce")

    target_corr_rows = []
    for f in available:
        corr_val = work[[f, TARGET_COL]].corr(method=corr_method).iloc[0, 1]
        target_corr_rows.append(
            {
                "feature": f,
                "corr_with_target": corr_val,
                "abs_corr_with_target": abs(corr_val) if pd.notna(corr_val) else np.nan,
            }
        )
    target_corr_df = pd.DataFrame(target_corr_rows)
    target_corr_df = target_corr_df.merge(shap_rank[["rank", "feature", "feature_group", "mean_abs_shap"]], on="feature", how="left")
    target_corr_df = target_corr_df.sort_values("rank")
    target_corr_df.to_csv(results_dir / "exp3_feature_target_correlation.csv", index=False, encoding="utf-8-sig")

    corr_matrix = work[available].corr(method=corr_method).abs()
    corr_matrix.to_csv(results_dir / "exp3_abs_feature_correlation_matrix.csv", encoding="utf-8-sig")

    selected = []
    excluded_rows = []
    rank_info = shap_rank.set_index("feature").to_dict("index")

    for f in available:
        if not selected:
            selected.append(f)
            continue

        corr_to_selected = corr_matrix.loc[f, selected].dropna()
        if len(corr_to_selected) == 0:
            selected.append(f)
            continue

        max_corr = float(corr_to_selected.max())
        max_corr_feature = str(corr_to_selected.idxmax())
        if max_corr >= corr_threshold:
            excluded_rows.append(
                {
                    "excluded_feature": f,
                    "excluded_rank": rank_info.get(f, {}).get("rank", np.nan),
                    "excluded_feature_group": rank_info.get(f, {}).get("feature_group", "Unknown"),
                    "excluded_mean_abs_shap": rank_info.get(f, {}).get("mean_abs_shap", np.nan),
                    "kept_feature_with_highest_corr": max_corr_feature,
                    "kept_feature_rank": rank_info.get(max_corr_feature, {}).get("rank", np.nan),
                    "abs_corr": max_corr,
                    "corr_threshold": corr_threshold,
                    "corr_method": corr_method,
                }
            )
        else:
            selected.append(f)

    candidate_df = shap_rank[shap_rank["feature"].isin(selected)].copy()
    candidate_df["corr_filtered_rank"] = range(1, len(candidate_df) + 1)
    keep_cols = [c for c in ["corr_filtered_rank", "rank", "feature", "feature_group", "mean_abs_shap", "mean_signed_shap"] if c in candidate_df.columns]
    candidate_df = candidate_df[keep_cols]
    candidate_df.to_csv(results_dir / "exp3_shap_corr_candidate_features.csv", index=False, encoding="utf-8-sig")

    excluded_df = pd.DataFrame(excluded_rows)
    excluded_df.to_csv(results_dir / "exp3_high_corr_excluded_features.csv", index=False, encoding="utf-8-sig")

    return candidate_df


def topk_features_from_corr_candidates(candidate_df: pd.DataFrame, k: int) -> List[str]:
    k = min(int(k), len(candidate_df))
    return candidate_df.sort_values("corr_filtered_rank").head(k)["feature"].tolist()


def save_main_daytime_summary(summaries: List[Dict], output_dir: Path, prefix: str) -> None:
    df = pd.DataFrame(summaries)
    if len(df) == 0:
        return
    day_cols = [
        "model", "case", "feature_config", "feature_count", "seq_len",
        "TEST_DAY_n_samples", "TEST_DAY_MAE_MWh", "TEST_DAY_RMSE_MWh",
        "TEST_DAY_NMAE_pct", "TEST_DAY_NRMSE_pct", "TEST_DAY_MBE_MWh", "TEST_DAY_R2",
        "VALID_DAY_NMAE_pct", "valid_objective_DAY_NMAE_pct",
        "model_path", "scaler_path",
    ]
    existing = [c for c in day_cols if c in df.columns]
    df[existing].to_csv(output_dir / "results" / f"MAIN_{prefix}_individual_daytime_results.csv", index=False, encoding="utf-8-sig")


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.quick_test:
        args.topk_list = "5,10"
        args.n_trials_lgb = min(args.n_trials_lgb, 2)
        args.n_trials_seq = min(args.n_trials_seq, 2)
        args.epochs = min(args.epochs, 3)
        args.patience = min(args.patience, 2)

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    ensure_dir(output_dir / "results")

    if args.device == "auto":
        device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    requested_models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    clip_predictions = not args.no_clip
    topk_list = parse_int_list(args.topk_list)

    df = read_feature_csv(args.csv)
    shap_rank = load_shap_rank(args.shap_importance)
    shap_rank = shap_rank[shap_rank["feature"].isin(df.columns)].copy().reset_index(drop=True)
    shap_rank["rank"] = range(1, len(shap_rank) + 1)
    if len(shap_rank) == 0:
        raise ValueError("CSV 컬럼과 일치하는 SHAP feature가 없습니다.")

    train_df, valid_df, test_df = split_by_year(df)

    output_dir = Path(args.output_dir)
    candidate_df = make_corr_filtered_shap_candidates(
        df=df,
        train_df=train_df,
        shap_rank=shap_rank,
        corr_threshold=args.corr_threshold,
        corr_method=args.corr_method,
        output_dir=output_dir,
    )
    if len(candidate_df) == 0:
        raise ValueError("상관관계 필터링 후 남은 feature가 없습니다. corr_threshold를 높여보세요.")

    topk_list = sorted(set(min(k, len(candidate_df)) for k in topk_list if k > 0))
    prefix_name = f"SHAP_Corr{str(args.corr_threshold).replace('.', 'p')}"

    save_json(
        {
            "experiment": "Experiment 3 - SHAP + correlation-filtered Top-K all-candidate retraining",
            "input_csv": args.csv,
            "shap_importance": args.shap_importance,
            "output_dir": str(output_dir),
            "target": TARGET_COL,
            "split": {"train": "2020-2022", "valid": "2023", "test": "2024"},
            "selection_rule": "SHAP rank priority, remove later feature if abs(feature-feature corr) >= corr_threshold, then train every Top-K candidate with every requested model.",
            "corr_data_period": "train only",
            "corr_threshold": args.corr_threshold,
            "corr_method": args.corr_method,
            "experiment_4": "보류. 이 파일에서는 수행하지 않음.",
            "requested_models": requested_models,
            "topk_list": topk_list,
            "n_trials_lgb": args.n_trials_lgb,
            "n_trials_seq": args.n_trials_seq,
            "epochs": args.epochs,
            "patience": args.patience,
            "same_training_condition_as_04_train_v1": True,
            "device": device,
            "clip_predictions": clip_predictions,
        },
        output_dir / "experiment_config.json",
    )

    print("=" * 90)
    print("[실험 3] SHAP+상관관계 Top-K별 선택 모델 재학습")
    print("Train:", train_df[DATE_COL].min(), "~", train_df[DATE_COL].max(), len(train_df))
    print("Valid:", valid_df[DATE_COL].min(), "~", valid_df[DATE_COL].max(), len(valid_df))
    print("Test :", test_df[DATE_COL].min(), "~", test_df[DATE_COL].max(), len(test_df))
    print("Correlation-filtered candidate count:", len(candidate_df))
    print("Top-K candidates:", topk_list)
    print("Models:", requested_models)
    print("Trials: LightGBM=", args.n_trials_lgb, ", Seq=", args.n_trials_seq)
    print("Epochs/Patience:", args.epochs, "/", args.patience)
    print("Device:", device)

    summaries: List[Dict] = []
    pred_dfs: Dict[Tuple[str, str], pd.DataFrame] = {}

    for k in topk_list:
        config_name = f"{prefix_name}_Top{k}"
        feature_cols = validate_features_exist(df, topk_features_from_corr_candidates(candidate_df, k))
        pd.DataFrame({"rank_in_config": range(1, len(feature_cols) + 1), "feature": feature_cols}).to_csv(
            output_dir / "results" / f"features_{config_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("\n" + "=" * 90)
        print(f"[Feature Set] {config_name} / feature_count={len(feature_cols)}")

        if "lightgbm" in requested_models:
            print(f"[LightGBM] {config_name}")
            summary, pred_df, _ = train_lightgbm_config(
                config_name=config_name,
                feature_cols=feature_cols,
                train_df=train_df,
                valid_df=valid_df,
                test_df=test_df,
                output_dir=output_dir,
                n_trials=args.n_trials_lgb,
                seed=args.seed,
                clip_predictions=clip_predictions,
            )
            summaries.append(summary)
            pred_dfs[("LightGBM", config_name)] = pred_df
            save_summary_csv(summaries, output_dir / "results" / "exp3_all_topk_all_models_summary.csv")
            save_main_daytime_summary(summaries, output_dir, "exp3")

        if "lstm" in requested_models:
            print(f"[LSTM] {config_name}")
            summary, pred_df = train_sequence_config(
                model_name="LSTM",
                config_name=config_name,
                feature_cols=feature_cols,
                train_df=train_df,
                valid_df=valid_df,
                test_df=test_df,
                output_dir=output_dir,
                n_trials=args.n_trials_seq,
                seed=args.seed,
                max_epochs=args.epochs,
                patience=args.patience,
                device=device,
                clip_predictions=clip_predictions,
            )
            summaries.append(summary)
            pred_dfs[("LSTM", config_name)] = pred_df
            save_summary_csv(summaries, output_dir / "results" / "exp3_all_topk_all_models_summary.csv")
            save_main_daytime_summary(summaries, output_dir, "exp3")

        if "patchtst" in requested_models:
            print(f"[PatchTST] {config_name}")
            summary, pred_df = train_sequence_config(
                model_name="PatchTST",
                config_name=config_name,
                feature_cols=feature_cols,
                train_df=train_df,
                valid_df=valid_df,
                test_df=test_df,
                output_dir=output_dir,
                n_trials=args.n_trials_seq,
                seed=args.seed,
                max_epochs=args.epochs,
                patience=args.patience,
                device=device,
                clip_predictions=clip_predictions,
            )
            summaries.append(summary)
            pred_dfs[("PatchTST", config_name)] = pred_df
            save_summary_csv(summaries, output_dir / "results" / "exp3_all_topk_all_models_summary.csv")
            save_main_daytime_summary(summaries, output_dir, "exp3")

    build_common_timestamp_metrics(pred_dfs, output_dir, prefix="exp3")

    print("\n[완료]")
    print("상관관계 필터 후보:", output_dir / "results" / "exp3_shap_corr_candidate_features.csv")
    print("제외 feature 목록:", output_dir / "results" / "exp3_high_corr_excluded_features.csv")
    print("전체 요약:", output_dir / "results" / "exp3_all_topk_all_models_summary.csv")
    print("주요 DAY 성능표:", output_dir / "results" / "MAIN_exp3_individual_daytime_results.csv")
    print("모델 저장 폴더:", output_dir / "models")
    print("실험 4는 보류 상태이며, 이 코드에서는 수행하지 않았습니다.")


if __name__ == "__main__":
    main()
