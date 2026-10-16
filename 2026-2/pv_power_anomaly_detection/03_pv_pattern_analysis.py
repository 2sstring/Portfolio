# -*- coding: utf-8 -*-
"""Step 1: Build irradiance/season/hour reference patterns (no ML training).

Input: 한국동서발전_지역별/PV_Dataset_충북.csv (02_region_split_v1.py output).
Outputs: results/pattern_data.csv, reference_patterns.csv, pattern_summary.json
Run demo: python 01_pv_pattern_analysis.py --demo --output-dir results_demo
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
EXT = '대기권밖일사량계산값'
REQUIRED = [DATE, POWER, CAP, IRR, EXT]


def make_demo_data(path: Path) -> None:
    """Creates explicit synthetic data; it is NOT Chungbuk's measured data."""
    rng = np.random.default_rng(20261016)
    times = pd.date_range('2024-01-01', periods=190 * 24, freq='h')
    hrs = times.hour.to_numpy()
    day = times.dayofyear.to_numpy()
    sunlight = np.maximum(np.sin(np.pi * (hrs - 6) / 12), 0)
    seasonal = 0.8 + 0.18 * np.sin((day - 70) / 365 * 2 * np.pi)
    daily_cloud = rng.uniform(0.55, 1, len(times) // 24)
    cloud = np.repeat(daily_cloud, 24)
    irr = np.maximum(850 * sunlight * seasonal * cloud + rng.normal(0, 15, len(times)), 0)
    irr[sunlight <= 0] = 0
    ext = 1150 * sunlight * np.maximum(seasonal, 0.1)
    norm = np.clip(0.00070 * irr * (0.92 + 0.05 * rng.normal(size=len(times))), 0, 0.92)
    cap = np.full(len(times), 100.0)
    # Intentionally injected low-output, flatline, and abrupt-drop examples.
    for d in ['2024-03-18', '2024-05-12']:
        sel = (times.date == pd.Timestamp(d).date()) & (hrs >= 10) & (hrs <= 15)
        norm[sel] *= 0.05
    flat = (times.date == pd.Timestamp('2024-04-20').date()) & (hrs >= 9) & (hrs <= 15)
    norm[flat] = 0.18
    abrupt = (times.date == pd.Timestamp('2024-06-15').date()) & (hrs == 12)
    norm[abrupt] *= 0.04
    df = pd.DataFrame({DATE: times, '시도명': '충북', POWER: norm * cap, CAP: cap,
                       IRR: irr, EXT: ext, '기온': 18.0, '습도': 55.0})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8-sig')


def group_reference(df: pd.DataFrame, keys: list[str], level: str, min_n: int) -> pd.DataFrame:
    valid = df.loc[df['reference_candidate'], keys + ['normalized_power']]
    tab = valid.groupby(keys, observed=True)['normalized_power'].agg(
        ref_n='size', ref_p25=lambda x: x.quantile(.25), ref_median='median'
    ).reset_index()
    tab = tab.loc[tab['ref_n'] >= min_n].copy()
    tab['reference_level'] = level
    return tab


def main() -> None:
    p = argparse.ArgumentParser(description='일사량/계절/시간대별 태양광 발전 기준 패턴 계산')
    p.add_argument('--input', type=Path, default=ROOT / '한국동서발전_지역별' / 'PV_Dataset_충북.csv')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    p.add_argument('--demo', action='store_true', help='실제 자료 대신 인공 예제 생성 후 분석')
    p.add_argument('--bins', type=int, default=6, help='일사량 분위 구간 수')
    p.add_argument('--min-reference-samples', type=int, default=15)
    p.add_argument('--analysis-irr-quantile', type=float, default=.45,
                   help='주간 양의 일사량 중 이상 탐지 대상의 최소 분위수')
    args = p.parse_args()
    if not 2 <= args.bins <= 20 or args.min_reference_samples < 5 or not 0 < args.analysis_irr_quantile < 1:
        p.error('--bins, --min-reference-samples, --analysis-irr-quantile 설정을 확인하세요.')

    source = ROOT / '한국동서발전_지역별' / 'demo_PV_Dataset_충북.csv' if args.demo else args.input
    source = source.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.demo:
        make_demo_data(source)
    if not source.is_file():
        raise FileNotFoundError(f'데이터 파일을 찾을 수 없습니다: {source}\n'
                                '02번 지역 분리 출력인 PV_Dataset_충북.csv를 한국동서발전_지역별/ 폴더에 복사하세요.')

    df = pd.read_csv(source, encoding='utf-8-sig', low_memory=False)
    missing = [c for c in REQUIRED if c not in df]
    if missing:
        raise ValueError(f'필요한 컬럼이 없습니다: {missing}. 실제 컬럼: {df.columns.tolist()}')
    if '시도명' in df and df['시도명'].dropna().astype(str).nunique() > 1:
        raise ValueError('여러 지역이 섞여 있습니다. 먼저 충북 지역 데이터만 추출하세요.')

    n_raw = len(df)
    df.insert(0, 'source_row_id', np.arange(2, n_raw + 2, dtype=np.int64))
    df[DATE] = pd.to_datetime(df[DATE], errors='coerce')
    for c in [POWER, CAP, IRR, EXT]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    duplicate = df[DATE].notna() & df[DATE].duplicated(keep=False)
    valid = (df[DATE].notna() & ~duplicate & (df[CAP] > 0) & (df[POWER] >= 0)
             & df[IRR].notna() & (df[IRR] >= 0) & df[EXT].notna() & (df[EXT] >= 0))
    rejected = int((~valid).sum())
    df = df.loc[valid].copy().sort_values(DATE).reset_index(drop=True)
    if df.empty:
        raise ValueError('유효한 시간/발전량/설비용량/일사량 데이터가 없습니다.')

    df['normalized_power'] = df[POWER] / df[CAP]
    # Do not silently remove normalized >1; flag as a potential input issue.
    df['physically_suspicious'] = df['normalized_power'] > 1 + 1e-6
    # Daylight must be positive extraterrestrial irradiance AND positive measured irradiance.
    df['is_daylight'] = (df[EXT] > 0) & (df[IRR] > 0)
    df['season'] = (df[DATE].dt.month % 12 // 3 + 1).astype(int)  # 1=winter ... 4=fall
    df['hour'] = df[DATE].dt.hour.astype(int)
    daytime = df.loc[df['is_daylight'] & ~df['physically_suspicious'], IRR]
    if len(daytime) < max(30, args.min_reference_samples * 2):
        raise ValueError('주간 일사량 표본이 부족하여 기준 패턴을 만들 수 없습니다.')

    edges = np.unique(np.quantile(daytime.to_numpy(), np.linspace(0, 1, args.bins + 1)))
    if len(edges) < 3:
        raise ValueError('일사량에 서로 다른 값이 너무 적어서 구간을 만들 수 없습니다.')
    # np.searchsorted works for values equal to the maximum boundary as well.
    df['irr_bin'] = np.searchsorted(edges[1:-1], df[IRR].to_numpy(), side='right').astype(int)
    irradiance_threshold = float(daytime.quantile(args.analysis_irr_quantile))
    df['reference_candidate'] = df['is_daylight'] & ~df['physically_suspicious']

    # Hierarchical fallback prevents tiny season-hour-bin groups from setting unstable baselines.
    stages = [(['season', 'hour', 'irr_bin'], 'season_hour_irr', args.min_reference_samples),
              (['season', 'irr_bin'], 'season_irr', args.min_reference_samples),
              (['irr_bin'], 'irr_only', args.min_reference_samples)]
    ref_cols = ['ref_n', 'ref_p25', 'ref_median', 'reference_level']
    for c in ref_cols:
        df[c] = pd.Series([np.nan] * len(df), dtype='object' if c == 'reference_level' else 'float64')
    references = []
    for keys, label, minimum in stages:
        tab = group_reference(df, keys, label, minimum)
        if tab.empty:
            continue
        references.append(tab.assign(group_keys=','.join(keys)))
        temp = df[keys].merge(tab[keys + ref_cols], how='left', on=keys, sort=False)
        fill = df['ref_n'].isna() & temp['ref_n'].notna() & df['is_daylight']
        for c in ref_cols:
            df.loc[fill, c] = temp.loc[fill, c].to_numpy()

    df['analysis_eligible'] = (df['is_daylight'] & (df[IRR] >= irradiance_threshold)
                               & ~df['physically_suspicious'] & df['ref_n'].notna()
                               & (df['ref_median'] > 0))
    gap_count = int((df[DATE].diff().dropna() != pd.Timedelta(hours=1)).sum())
    df = df.drop(columns=['reference_candidate'])
    # Numeric columns must remain numeric after fallback assignments.
    for c in ['ref_n', 'ref_p25', 'ref_median']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.to_csv(out / 'pattern_data.csv', index=False, encoding='utf-8-sig')
    if references:
        pd.concat(references, ignore_index=True).to_csv(out / 'reference_patterns.csv', index=False,
                                                         encoding='utf-8-sig')
    summary = {
        'source': str(source), 'demo_data': bool(args.demo), 'raw_rows': n_raw,
        'excluded_invalid_or_duplicate_rows': rejected, 'rows_used': int(len(df)),
        'suspicious_power_over_capacity_rows': int(df['physically_suspicious'].sum()),
        'remaining_time_gaps': gap_count, 'daylight_rows': int(df['is_daylight'].sum()),
        'eligible_rows': int(df['analysis_eligible'].sum()),
        'irradiance_threshold': irradiance_threshold,
        'irradiance_edges': edges.tolist(), 'reference_group_min_count': args.min_reference_samples,
        'reference_levels': df['reference_level'].value_counts(dropna=False).to_dict(),
        'note': '기존 전처리를 반복하지 않음. 전체 관측치의 참고 분포에서 후보를 찾는 기술적 탐색이며 실제 고장 판정 아님.'
    }
    with open(out / 'pattern_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f'[01] 입력: {source.name} | 원본 {n_raw:,}행 → 분석용 {len(df):,}행')
    print(f'[01] 유효한 탐지 대상: {summary["eligible_rows"]:,}행 | 기준 일사량: {irradiance_threshold:.4g}')
    print(f'[01] 저장: {out / "pattern_data.csv"}')


if __name__ == '__main__':
    main()
