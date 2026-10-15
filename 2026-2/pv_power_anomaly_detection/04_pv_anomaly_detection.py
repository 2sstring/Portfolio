# -*- coding: utf-8 -*-
"""Step 2: Detect potential anomalies from the reference patterns (no model training).

Input: results/pattern_data.csv from 01_pv_pattern_analysis.py
Outputs: results/anomaly_events.csv, anomaly_samples.csv, detection_config.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATE = '발전일자'
POWER = '발전량(MWh)'
CAP = '설비용량(MW)'
IRR = '일사량'
EVENT_COLS = ['event_id', 'anomaly_type', 'start_time', 'end_time', 'duration_hours',
              'sample_count', 'severity', 'description']
SAMPLE_COLS = ['event_id', 'anomaly_type', 'source_row_id', 'timestamp', POWER, CAP,
               IRR, 'normalized_power', 'ref_median', 'ref_p25', 'reference_level']


def true_runs(flags: np.ndarray, timestamps: pd.Series) -> list[list[int]]:
    """Contiguous True rows with EXACTLY a 1-hour spacing; no bridging date gaps."""
    runs: list[list[int]] = []
    current: list[int] = []
    for i, flag in enumerate(flags):
        if not flag:
            if current:
                runs.append(current)
                current = []
            continue
        if current and (timestamps.iloc[i] - timestamps.iloc[current[-1]] != pd.Timedelta(hours=1)):
            runs.append(current)
            current = []
        current.append(i)
    if current:
        runs.append(current)
    return runs


def main() -> None:
    p = argparse.ArgumentParser(description='저출력·출력 정체·출력 급변 후보 탐지')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    p.add_argument('--input', type=Path, default=None, help='기본값: --output-dir/pattern_data.csv')
    p.add_argument('--low-output-ratio', type=float, default=.40,
                   help='참고 중앙값의 몇 배 이하일 때 저출력 후보로 판단할지')
    p.add_argument('--low-min-hours', type=int, default=2)
    p.add_argument('--flat-min-hours', type=int, default=4)
    p.add_argument('--flat-tolerance', type=float, default=.003,
                   help='시간 간 정규화발전량 차이가 이 값 이하면 정체로 간주')
    p.add_argument('--flat-irr-variation', type=float, default=.22,
                   help='정체 구간 내 일사량의 (최대-최소)/중앙값 기준')
    p.add_argument('--jump-min', type=float, default=.15,
                   help='정규화발전량의 최소 급변량')
    p.add_argument('--jump-relative-min', type=float, default=.50,
                   help='급변량/직전 정규화발전량 최소값')
    p.add_argument('--irr-relative-max', type=float, default=.17,
                   help='급변 후보 구간의 일사량 상대 변동 상한')
    args = p.parse_args()
    if not 0 < args.low_output_ratio < 1 or args.low_min_hours < 1 or args.flat_min_hours < 3:
        p.error('저출력 임계값과 연속 시간 기준이 올바르지 않습니다.')
    if args.flat_tolerance < 0 or args.flat_irr_variation < 0 or args.jump_min <= 0 or args.jump_relative_min <= 0:
        p.error('정체 또는 급변 탐지 임계값은 양수여야 합니다.')
    if not 0 <= args.irr_relative_max < 1:
        p.error('--irr-relative-max는 0 이상 1 미만이어야 합니다.')

    out = args.output_dir.resolve()
    src = (args.input or out / 'pattern_data.csv').resolve()
    if not src.is_file():
        raise FileNotFoundError(f'1단계 출력이 없습니다: {src}\n먼저 01_pv_pattern_analysis.py를 실행하세요.')
    df = pd.read_csv(src, encoding='utf-8-sig', parse_dates=[DATE], low_memory=False)
    need = [DATE, POWER, CAP, IRR, 'normalized_power', 'analysis_eligible',
            'is_daylight', 'ref_median', 'ref_p25', 'reference_level', 'source_row_id']
    missing = [c for c in need if c not in df]
    if missing:
        raise ValueError(f'1단계 출력 파일에서 필요한 컬럼을 찾을 수 없습니다: {missing}')
    df = df.sort_values(DATE).reset_index(drop=True)
    for c in [POWER, CAP, IRR, 'normalized_power', 'ref_median', 'ref_p25']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in ['analysis_eligible', 'is_daylight']:
        # pandas CSV will contain True/False as strings or booleans depending on version
        df[c] = df[c].astype(str).str.lower().isin(['true', '1'])
    times = df[DATE]
    eligible = df['analysis_eligible'].to_numpy(dtype=bool)
    norm = df['normalized_power'].to_numpy(dtype=float)
    irr = df[IRR].to_numpy(dtype=float)
    ref = df['ref_median'].to_numpy(dtype=float)
    events: list[dict] = []
    samples: list[dict] = []

    def register(kind: str, idx: list[int], detail: str) -> None:
        if not idx:
            return
        subset = df.iloc[idx]
        event_id = f'EVT-{len(events) + 1:05d}'
        events.append({'event_id': event_id, 'anomaly_type': kind,
                       'start_time': subset[DATE].iloc[0], 'end_time': subset[DATE].iloc[-1],
                       'duration_hours': int(len(idx)), 'sample_count': int(len(idx)),
                       'severity': 'review', 'description': detail})
        for _, row in subset.iterrows():
            samples.append({'event_id': event_id, 'anomaly_type': kind,
                            'source_row_id': int(row['source_row_id']),
                            'timestamp': row[DATE], POWER: row[POWER], CAP: row[CAP],
                            IRR: row[IRR], 'normalized_power': row['normalized_power'],
                            'ref_median': row['ref_median'], 'ref_p25': row['ref_p25'],
                            'reference_level': row['reference_level']})

    # Rule 1: persistent low output compared against season/hour/irradiance reference.
    # Additionally require an absolute difference, not only a percentage difference.
    is_low = (eligible & (ref >= .04) & (norm <= args.low_output_ratio * ref)
              & ((ref - norm) >= .07))
    for run in true_runs(is_low, times):
        if len(run) >= args.low_min_hours:
            register('low_output_relative_to_reference', run,
                     f'정규화발전량 <= 기준 중앙값의 {args.low_output_ratio:.0%}; '
                     '절대 차이 >= 0.07; 연속 발생')

    # Rule 2: daytime output stays (nearly) identical while irradiance is varying.
    # Require contiguous daylight AND eligible timestamps; midnight/gaps split a run.
    flat_runs: list[list[int]] = []
    current: list[int] = []
    for i in range(len(df)):
        if not eligible[i] or not np.isfinite(norm[i]):
            if current:
                flat_runs.append(current)
                current = []
            continue
        is_continuation = bool(current and
            times.iloc[i] - times.iloc[current[-1]] == pd.Timedelta(hours=1) and
            abs(norm[i] - norm[current[-1]]) <= args.flat_tolerance)
        if current and not is_continuation:
            flat_runs.append(current)
            current = []
        current.append(i)
    if current:
        flat_runs.append(current)
    for run in flat_runs:
        if len(run) < args.flat_min_hours:
            continue
        local_irr = irr[run]
        med_irr = float(np.median(local_irr))
        variation = float((local_irr.max() - local_irr.min()) / max(med_irr, 1e-9))
        if variation >= args.flat_irr_variation:
            register('flat_output_with_variable_irradiance', run,
                     f'{len(run)}시간 출력 거의 동일; 일사량 변동비 {variation:.1%}')

    # Rule 3: large output change despite fairly stable irradiance over adjacent hours.
    for i in range(1, len(df)):
        if not eligible[i] or not eligible[i - 1]:
            continue
        if times.iloc[i] - times.iloc[i - 1] != pd.Timedelta(hours=1):
            continue
        irr_ratio = abs(irr[i] - irr[i - 1]) / max(irr[i], irr[i - 1], 1e-9)
        abs_jump = abs(norm[i] - norm[i - 1])
        rel_jump = abs_jump / max(norm[i - 1], .02)
        if (irr_ratio <= args.irr_relative_max and abs_jump >= args.jump_min
                and rel_jump >= args.jump_relative_min):
            register('abrupt_output_change', [i],
                     f'직전 시간 대비 정규화발전량 변화 {abs_jump:.3f}; '
                     f'일사량 변화비 {irr_ratio:.1%}')

    events_df = pd.DataFrame(events, columns=EVENT_COLS)
    samples_df = pd.DataFrame(samples, columns=SAMPLE_COLS)
    out.mkdir(parents=True, exist_ok=True)
    events_df.to_csv(out / 'anomaly_events.csv', index=False, encoding='utf-8-sig')
    samples_df.to_csv(out / 'anomaly_samples.csv', index=False, encoding='utf-8-sig')
    config = {'input': str(src), 'number_of_events': int(len(events_df)),
              'number_of_flagged_unique_rows': int(samples_df['source_row_id'].nunique()) if len(samples_df) else 0,
              'parameters': vars(args) | {'output_dir': str(out), 'input': str(src)},
              'warning': '비지도 규칙 탐지 후보. 실제 설비 고장, 성능 저하를 확정할 수 없습니다. '
                         '일사량-발전량 관계는 시간대/계절/설비운전/출력제한의 영향을 받습니다.'}
    config['parameters'] = {k: str(v) if isinstance(v, Path) else v for k, v in config['parameters'].items()}
    with (out / 'detection_config.json').open('w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f'[02] 분석 행 {len(df):,}개 | 이상 후보 이벤트 {len(events_df):,}건')
    if not events_df.empty:
        print(events_df['anomaly_type'].value_counts().to_string())
    print(f'[02] 저장: {out / "anomaly_events.csv"}')


if __name__ == '__main__':
    main()
