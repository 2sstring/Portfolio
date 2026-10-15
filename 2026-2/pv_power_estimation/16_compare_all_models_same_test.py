# -*- coding: utf-8 -*-
"""
16_compare_previous_experiments_same_test.py

[기존 실험 결과의 동일 Test 기준 성능 비교]

목적
----
1. RF, LightGBM, LSTM의 Full69 성능 비교
2. SHAP Top20 재학습 결과 비교
3. SHAP-SFFS 특징 선택 결과 비교
4. Cross-SFFS 교차 적용 결과 비교

평가 조건
---------
- 3개 모델의 공통 Test 시점 사용
- ALL 및 DAY 구간별 성능 평가
- MAE, RMSE, NMAE, NRMSE, MBE, R2 계산
- 기존 예측 CSV를 사용하며 모델을 재학습하지 않음

주의
----
본 코드는 기존 실험 결과를 비교하는 코드이다.
17~19번의 Full69 고정 하이퍼파라미터
재학습 결과를 종합하는 코드는 아니다.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

DATE = '발전일자'
CAPACITY = '설비용량(MW)'
DAYTIME = '대기권밖일사량계산값'
TRUE = 'true_MWh'
PRED = 'pred_MWh'
MODELS = {'random_forest': ('RandomForest', 'randomforest'),
          'lightgbm': ('LightGBM', 'lightgbm'),
          'lstm': ('LSTM', 'lstm')}


def arguments():
    p = argparse.ArgumentParser(description='3-model fixed-Test evaluation, all experiments; no retraining')
    p.add_argument('--comparison_dir', default='pv_training_outputs_final_comparison')
    p.add_argument('--rf_full_dir', default='pv_training_outputs_rf')
    p.add_argument('--lgb_full_dir', default='pv_training_outputs')
    p.add_argument('--lstm_full_dir', default='pv_training_outputs_lstm_rerun')
    p.add_argument('--top20_dir', default='pv_exp_shap_top20_same_test')
    p.add_argument('--sffs_dir', default='pv_exp5_shap_sffs')
    p.add_argument('--cross_dir', default='pv_exp6_cross_model_sffs')
    p.add_argument('--output_dir', default='pv_final_3models_same_test')
    return p.parse_args()


def read_reference(path):
    if not path.is_file():
        raise FileNotFoundError(f'3모델 공통 Test 기준 파일이 없습니다: {path}\n먼저 python 04_compare_3models_common_test.py 실행')
    ref = pd.read_csv(path, encoding='utf-8-sig')
    if DATE not in ref:
        raise ValueError(f'{path}: {DATE} 컬럼이 없습니다')
    times = pd.DatetimeIndex(pd.to_datetime(ref[DATE], errors='raise')).sort_values()
    if times.empty or times.has_duplicates or times.isna().any():
        raise ValueError('공통 Test timestamp가 비어 있거나 중복/NaT를 포함합니다')
    return times


def single_match(folder, pattern, label):
    found = sorted(folder.glob(pattern))
    if len(found) != 1:
        raise FileNotFoundError(f'{label}: 예측 파일이 정확히 1개여야 합니다 (현재 {len(found)}개).\n'
                                f'검색: {folder / pattern}\n후보: {[str(x) for x in found]}')
    return found[0]


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
        raise ValueError(f'{label}: 3모델 공통 Test 시점 {len(missing)}개가 원본 예측에 없습니다.\n'
                         f'예시: {list(missing[:5])}\n파일: {path}\n'
                         '기준 Test를 임의로 줄이지 않습니다. 누락 원인을 확인하세요.')
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
    return dict(n_samples=len(y), MAE_MWh=float(np.mean(np.abs(err))),
                RMSE_MWh=float(np.sqrt(np.mean(err ** 2))),
                NMAE_pct=float(np.mean(np.abs(err) / capacity) * 100),
                NRMSE_pct=float(np.sqrt(np.mean((err / capacity) ** 2)) * 100),
                MBE_MWh=float(np.mean(err)),
                R2=float(r2_score(y, pred)) if len(y) >= 2 else np.nan)


def sffs_raw(folder, model):
    """Use exact case signature from per-model summary if available; otherwise require unique raw file."""
    display, token = MODELS[model]
    summary = folder / 'results' / f'FINAL_TEST_SFFS_{model}.csv'
    prediction_folder = folder / 'final_tuned' / 'predictions'
    if summary.is_file():
        row = pd.read_csv(summary, encoding='utf-8-sig').iloc[0]
        if 'selected_feature_signature' in row and pd.notna(row['selected_feature_signature']):
            sig = str(row['selected_feature_signature'])
            return single_match(prediction_folder,
                                f'test_predictions_{token}_SHAP_SFFS_{display}_K*_{sig}.csv',
                                f'SFFS/{display}')
    return single_match(prediction_folder,
                        f'test_predictions_{token}_SHAP_SFFS_{display}_K*.csv',
                        f'SFFS/{display}')


def cross_raw(folder, source, target):
    """Select the source/target raw prediction, validating the saved signature when available."""
    display_src, _ = MODELS[source]
    display_target, token = MODELS[target]
    pred_folder = folder / 'training_outputs' / 'predictions'
    summary = folder / 'results' / f'SUMMARY_{source}_to_{target}.csv'
    if summary.is_file():
        row = pd.read_csv(summary, encoding='utf-8-sig').iloc[0]
        sig = row.get('source_feature_signature')
        if pd.notna(sig) and str(sig).strip():
            return single_match(pred_folder,
                                f'test_predictions_{token}_SFFS_{display_src}Set_K*_{sig}_TO_{display_target}.csv',
                                f'Cross/{source}->{target}')
    return single_match(pred_folder,
                        f'test_predictions_{token}_SFFS_{display_src}Set_K*_TO_{display_target}.csv',
                        f'Cross/{source}->{target}')


def jobs(args):
    for model, (display, token) in MODELS.items():
        full_root = {'random_forest': args.rf_full_dir, 'lightgbm': args.lgb_full_dir,
                     'lstm': args.lstm_full_dir}[model]
        yield ('Full69', model, model, Path(full_root) / 'predictions' /
               f'test_predictions_{token}_C6_Full.csv')
    for model in MODELS:
        yield ('SHAP_Top20', model, model, Path(args.top20_dir) / 'raw_predictions' /
               f'raw_test_predictions_SHAP_Top20_{model}.csv')
    for model in MODELS:
        yield ('SHAP_SFFS', model, model, sffs_raw(Path(args.sffs_dir), model))
    for source in MODELS:
        for target in MODELS:
            yield ('Cross_SFFS', source, target, cross_raw(Path(args.cross_dir), source, target))


def main():
    args = arguments()
    reference_path = Path(args.comparison_dir) / 'common_test_timestamps_C6_Full_3models.csv'
    ref = read_reference(reference_path)
    print(f'[3-model fixed Test] {len(ref)} timestamps | {reference_path}')
    results = {'Full69': [], 'SHAP_Top20': [], 'SHAP_SFFS': [], 'Cross_SFFS': []}
    audits = []
    reference_truth = None
    for experiment, source, target, path in jobs(args):
        label = f'{experiment} / {source} -> {target}'
        aligned = read_raw(path, ref, label)
        truth = aligned[[TRUE, CAPACITY, DAYTIME]].to_numpy(dtype=float)
        if reference_truth is None:
            reference_truth = truth
        elif not np.allclose(reference_truth, truth, rtol=1e-7, atol=1e-8, equal_nan=False):
            raise ValueError(f'{label}: 같은 timestamp의 실제값/설비용량/주간 판정값이 다릅니다: {path}')
        all_stats = metric(aligned)
        daytime = aligned.loc[aligned[DAYTIME] > 0]
        day_stats = metric(daytime)
        row = dict(experiment=experiment, feature_source=MODELS[source][0],
                   target_model=MODELS[target][0], source_key=source, target_key=target,
                   test_reference_count=len(ref),
                   **{f'ALL_{k}': v for k, v in all_stats.items()},
                   **{f'DAY_{k}': v for k, v in day_stats.items()})
        results[experiment].append(row)
        audits.append(dict(experiment=experiment, feature_source=MODELS[source][0],
                           target_model=MODELS[target][0], raw_prediction_file=str(path),
                           reference_file=str(reference_path), matched_timestamps=len(ref),
                           missing_timestamps=0, ground_truth_consistent=True))
        print(f'  {label:<52} DAY_NMAE={day_stats["NMAE_pct"]:.5f}%  '
              f'DAY_RMSE={day_stats["RMSE_MWh"]:.5f}  DAY_R2={day_stats["R2"]:.5f}')

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    filenames = {'Full69': 'FINAL_Full69_3models.csv',
                 'SHAP_Top20': 'FINAL_SHAP_Top20_3models.csv',
                 'SHAP_SFFS': 'FINAL_SHAP_SFFS_3models.csv',
                 'Cross_SFFS': 'FINAL_Cross_SFFS_9combinations.csv'}
    for experiment, filename in filenames.items():
        pd.DataFrame(results[experiment]).to_csv(out / filename, index=False, encoding='utf-8-sig')
    pd.DataFrame([row for experiment in filenames for row in results[experiment]]).to_csv(
        out / 'FINAL_ALL_EXPERIMENTS_3models.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(audits).to_csv(out / 'FINAL_INPUT_AUDIT_3models.csv', index=False, encoding='utf-8-sig')
    print(f'\n[완료] {sum(map(len, results.values()))}개 결과 저장: {out.resolve()}')
    print('공통 Test 기준 변경 없음 / RF·LightGBM·LSTM 학습 없음')


if __name__ == '__main__':
    main()
