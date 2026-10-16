# -*- coding: utf-8 -*-
"""
07_compare_full69_3models_same_test.py

[Full69의 3개 모델 예측 성능 비교 - 재학습 없음]

목적
----
1. RF, LightGBM, LSTM의 Full69(69개 특징) 성능 비교만 수행

평가 조건
---------
- Full69 3개 모델의 공통 Test timestamp 사용
- ALL(전체) 및 DAY(주간) 구간별 성능 평가
- MAE, RMSE, NMAE, NRMSE, MBE, R2 계산
- 기존 예측 CSV만 읽으며 모델을 다시 학습하지 않음
- SHAP Top20, SHAP-SFFS, Cross-SFFS 결과를 사용하지 않음
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

DATE = '발전일자'
CAPACITY = '설비용량(MW)'
DAYTIME = '대기권밖일사량계산값'
TRUE = 'true_MWh'
PRED = 'pred_MWh'
MODELS = {
    'random_forest': ('RandomForest', 'randomforest'),
    'lightgbm': ('LightGBM', 'lightgbm'),
    'lstm': ('LSTM', 'lstm'),
}


def arguments():
    p = argparse.ArgumentParser(
        description='Compare 3 Full69 models on the same Test timestamps; no retraining'
    )
    p.add_argument('--comparison_dir', default='pv_training_outputs_final_comparison')
    p.add_argument('--rf_full_dir', default='pv_training_outputs_rf')
    p.add_argument('--lgb_full_dir', default='pv_training_outputs')
    p.add_argument('--lstm_full_dir', default='pv_training_outputs')
    # 기존 실험 결과와 섞이지 않도록 별도 출력 폴더 사용
    p.add_argument('--output_dir', default='pv_final_full69_3models_same_test')
    return p.parse_args()


def read_reference(path):
    if not path.is_file():
        raise FileNotFoundError(
            f'3모델 공통 Test 기준 파일이 없습니다: {path}\n'
            '먼저 python 04_compare_3models_common_test.py 실행'
        )
    ref = pd.read_csv(path, encoding='utf-8-sig')
    if DATE not in ref:
        raise ValueError(f'{path}: {DATE} 컬럼이 없습니다')
    times = pd.DatetimeIndex(pd.to_datetime(ref[DATE], errors='raise')).sort_values()
    if times.empty or times.has_duplicates or times.isna().any():
        raise ValueError('공통 Test timestamp가 비어 있거나 중복/NaT를 포함합니다')
    return times


def read_raw(path, reference, label):
    if not path.is_file():
        raise FileNotFoundError(f'{label}: 원본 Test 예측 파일 없음: {path}')
    df = pd.read_csv(path, encoding='utf-8-sig')
    required = [DATE, TRUE, PRED, CAPACITY, DAYTIME]
    missing_cols = [c for c in required if c not in df]
    if missing_cols:
        raise ValueError(f'{label}: 필수 컬럼 누락 {missing_cols}: {path}')
    df[DATE] = pd.to_datetime(df[DATE], errors='raise')
    if df[DATE].isna().any() or df[DATE].duplicated().any():
        raise ValueError(f'{label}: Test timestamp 중복 또는 NaT: {path}')
    indexed = df.set_index(DATE)
    missing = reference.difference(pd.DatetimeIndex(indexed.index))
    if len(missing):
        raise ValueError(
            f'{label}: 3모델 공통 Test 시점 {len(missing)}개가 원본 예측에 없습니다.\n'
            f'예시: {list(missing[:5])}\n파일: {path}\n'
            '기준 Test를 임의로 줄이지 않습니다. 누락 원인을 확인하세요.'
        )
    aligned = indexed.loc[reference].reset_index()
    for c in [TRUE, PRED, CAPACITY, DAYTIME]:
        aligned[c] = pd.to_numeric(aligned[c], errors='raise')
        if not np.isfinite(aligned[c].to_numpy(dtype=float)).all():
            raise ValueError(f'{label}: {c}에 비정상 값(NaN/inf)이 있습니다: {path}')
    if (aligned[CAPACITY] <= 0).any():
        raise ValueError(f'{label}: 설비용량이 0 이하인 표본이 있습니다: {path}')
    return aligned


def metric(sub):
    y = sub[TRUE].to_numpy(dtype=float)
    pred = sub[PRED].to_numpy(dtype=float)
    capacity = sub[CAPACITY].to_numpy(dtype=float)
    err = pred - y
    if len(y) == 0:
        raise ValueError('평가 표본이 0개입니다')
    return dict(
        n_samples=len(y),
        MAE_MWh=float(np.mean(np.abs(err))),
        RMSE_MWh=float(np.sqrt(np.mean(err ** 2))),
        NMAE_pct=float(np.mean(np.abs(err) / capacity) * 100),
        NRMSE_pct=float(np.sqrt(np.mean((err / capacity) ** 2)) * 100),
        MBE_MWh=float(np.mean(err)),
        R2=float(r2_score(y, pred)) if len(y) >= 2 else np.nan,
    )


def jobs(args):
    roots = {
        'random_forest': args.rf_full_dir,
        'lightgbm': args.lgb_full_dir,
        'lstm': args.lstm_full_dir,
    }
    for model, (_, token) in MODELS.items():
        yield (
            model,
            Path(roots[model]) / 'predictions'
            / f'test_predictions_{token}_C6_Full.csv',
        )


def main():
    args = arguments()
    reference_path = (
        Path(args.comparison_dir) / 'common_test_timestamps_C6_Full_3models.csv'
    )
    ref = read_reference(reference_path)
    print(f'[Full69 / 3-model fixed Test] {len(ref)} timestamps | {reference_path}')

    results = []
    audits = []
    reference_truth = None
    for model, path in jobs(args):
        label = f'Full69 / {model}'
        aligned = read_raw(path, ref, label)
        truth = aligned[[TRUE, CAPACITY, DAYTIME]].to_numpy(dtype=float)
        if reference_truth is None:
            reference_truth = truth
        elif not np.allclose(
            reference_truth, truth, rtol=1e-7, atol=1e-8, equal_nan=False
        ):
            raise ValueError(
                f'{label}: 같은 timestamp의 실제값/설비용량/주간 판정값이 다릅니다: {path}'
            )

        all_stats = metric(aligned)
        daytime = aligned.loc[aligned[DAYTIME] > 0]
        day_stats = metric(daytime)
        row = dict(
            experiment='Full69',
            feature_source=MODELS[model][0],
            target_model=MODELS[model][0],
            source_key=model,
            target_key=model,
            test_reference_count=len(ref),
            **{f'ALL_{k}': v for k, v in all_stats.items()},
            **{f'DAY_{k}': v for k, v in day_stats.items()},
        )
        results.append(row)
        audits.append(dict(
            experiment='Full69',
            feature_source=MODELS[model][0],
            target_model=MODELS[model][0],
            raw_prediction_file=str(path),
            reference_file=str(reference_path),
            matched_timestamps=len(ref),
            missing_timestamps=0,
            ground_truth_consistent=True,
        ))
        print(
            f'  {label:<32} DAY_NMAE={day_stats["NMAE_pct"]:.5f}%  '
            f'DAY_RMSE={day_stats["RMSE_MWh"]:.5f}  '
            f'DAY_R2={day_stats["R2"]:.5f}'
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(
        out / 'FINAL_Full69_3models.csv', index=False, encoding='utf-8-sig'
    )
    pd.DataFrame(audits).to_csv(
        out / 'FINAL_INPUT_AUDIT_Full69_3models.csv', index=False, encoding='utf-8-sig'
    )
    print(f'\n[완료] Full69 3개 모델 결과 저장: {out.resolve()}')
    print('공통 Test 기준 변경 없음 / RF·LightGBM·LSTM 재학습 없음')


if __name__ == '__main__':
    main()
