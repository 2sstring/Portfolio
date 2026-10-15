# -*- coding: utf-8 -*-
"""Step 3: Generate anomaly CSV summaries, figures and a Markdown report.

Input: results/pattern_data.csv, anomaly_events.csv, anomaly_samples.csv
Outputs: results/report.md, anomaly_type_summary.csv, monthly_summary.csv, figures/*.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # render reliably on servers / in IDEs without GUI
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATE = '발전일자'
POWER = '발전량(MWh)'
IRR = '일사량'
LABELS = {'low_output_relative_to_reference': '저출력 후보',
          'flat_output_with_variable_irradiance': '출력 정체 후보',
          'abrupt_output_change': '급격한 출력 변화 후보'}


def set_fonts() -> None:
    from matplotlib import font_manager
    candidates = ['Malgun Gothic', 'NanumGothic', 'Noto Sans CJK KR', 'AppleGothic']
    available = {x.name for x in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.family'] = name
            break
    else:
        # English plot labels below when no Korean font is available.
        pass
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 125
    plt.rcParams['savefig.dpi'] = 170


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description='태양광 발전 이상 후보 통계 및 시각화 보고서 생성')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    p.add_argument('--max-scatter-points', type=int, default=9000)
    args = p.parse_args()
    out = args.output_dir.resolve()
    fig_dir = out / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_file = out / 'pattern_data.csv'
    event_file = out / 'anomaly_events.csv'
    sample_file = out / 'anomaly_samples.csv'
    for path in [data_file, event_file, sample_file]:
        if not path.exists():
            raise FileNotFoundError(f'전 단계 분석 결과가 없습니다: {path}')
    df = pd.read_csv(data_file, encoding='utf-8-sig', parse_dates=[DATE], low_memory=False)
    events = pd.read_csv(event_file, encoding='utf-8-sig', parse_dates=['start_time', 'end_time'])
    samples = pd.read_csv(sample_file, encoding='utf-8-sig', parse_dates=['timestamp'])
    df = df.sort_values(DATE).reset_index(drop=True)
    set_fonts()

    # Summary CSV: each event is a contiguous window, not one hourly row.
    type_summary = (events.groupby('anomaly_type', as_index=False).agg(
        event_count=('event_id', 'count'), flagged_hours=('duration_hours', 'sum'))
        if len(events) else pd.DataFrame(columns=['anomaly_type', 'event_count', 'flagged_hours']))
    type_summary.to_csv(out / 'anomaly_type_summary.csv', index=False, encoding='utf-8-sig')

    df['month_key'] = df[DATE].dt.to_period('M').astype(str)
    monthly = df.groupby('month_key').agg(
        observed_hours=(DATE, 'size'), daytime_hours=('is_daylight', lambda x: x.astype(str).str.lower().eq('true').sum()),
        eligible_hours=('analysis_eligible', lambda x: x.astype(str).str.lower().eq('true').sum()),
    ).reset_index()
    if len(samples):
        marked = df.loc[df['source_row_id'].isin(samples['source_row_id']), ['month_key', 'source_row_id']]
        unique_counts = marked.groupby('month_key')['source_row_id'].nunique().rename('flagged_unique_hours')
        monthly = monthly.merge(unique_counts, on='month_key', how='left')
    else:
        monthly['flagged_unique_hours'] = 0
    monthly['flagged_unique_hours'] = monthly['flagged_unique_hours'].fillna(0).astype(int)
    monthly['flagged_pct_of_eligible'] = np.where(monthly['eligible_hours'] > 0,
        100 * monthly['flagged_unique_hours'] / monthly['eligible_hours'], np.nan)
    monthly.to_csv(out / 'monthly_summary.csv', index=False, encoding='utf-8-sig')

    # 01: daily energy/capacity ratios (for trend comparison, NOT a capacity factor estimate).
    d = df.copy()
    d['date'] = d[DATE].dt.normalize()
    daily = d.groupby('date').agg(
        generation_MWh=(POWER, 'sum'), capacity_MW=('설비용량(MW)', 'median'), hours=(DATE, 'size'))
    daily['daily_normalized_energy'] = daily['generation_MWh'] / daily['capacity_MW']
    fig, ax = plt.subplots(figsize=(12, 4.1))
    ax.plot(daily.index, daily['daily_normalized_energy'], linewidth=1.1, label='MWh / median MW')
    ax.set(title='Daily normalized PV generation', ylabel='Daily MWh / MW', xlabel='Date')
    ax.grid(alpha=.22)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    save(fig, fig_dir / '01_daily_normalized_generation.png')

    # 02: irradiance vs normalized power, mark flagged hours.
    daytime = df.loc[df['is_daylight'].astype(str).str.lower().eq('true')].copy()
    if len(daytime) > args.max_scatter_points:
        daytime = daytime.sample(args.max_scatter_points, random_state=42)
    marked_ids = set(samples['source_row_id'].tolist()) if len(samples) else set()
    all_marked = df.loc[df['source_row_id'].isin(marked_ids)]
    if len(all_marked) > args.max_scatter_points:
        all_marked = all_marked.sample(args.max_scatter_points, random_state=42)
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.scatter(daytime[IRR], daytime['normalized_power'], s=7, alpha=.28, label='Daytime observations')
    if len(all_marked):
        ax.scatter(all_marked[IRR], all_marked['normalized_power'], s=15, alpha=.75,
                   marker='x', label='Flagged candidates')
    ax.set(title='Irradiance vs. normalized PV generation', xlabel='Irradiance (original dataset units)',
           ylabel='MWh / MW')
    ax.grid(alpha=.18)
    ax.legend(loc='best')
    save(fig, fig_dir / '02_irradiance_vs_generation.png')

    # 03: anomaly events by type.
    fig, ax = plt.subplots(figsize=(9, 4.3))
    if len(type_summary):
        types = type_summary['anomaly_type'].replace({
            'low_output_relative_to_reference': 'Low output',
            'flat_output_with_variable_irradiance': 'Flat output',
            'abrupt_output_change': 'Abrupt change'})
        ax.bar(types, type_summary['event_count'].to_numpy())
        ax.set(ylabel='Number of events')
        ax.tick_params(axis='x', labelrotation=10)
    else:
        ax.text(.5, .5, 'No events under current thresholds', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_xticks([])
    ax.set(title='Potential anomaly events by type')
    ax.grid(axis='y', alpha=.2)
    save(fig, fig_dir / '03_anomaly_events_by_type.png')

    # 04: monthly rates; denominator = rows eligible for event screening.
    fig, ax = plt.subplots(figsize=(max(9, min(len(monthly) * .45, 15)), 4.2))
    ax.bar(monthly['month_key'], monthly['flagged_pct_of_eligible'].fillna(0))
    ax.set(title='Monthly flagged-hour ratio (within eligible hours)', ylabel='Flagged / eligible (%)',
           xlabel='Month')
    stride = max(1, int(np.ceil(len(monthly) / 16)))
    ax.set_xticks(monthly['month_key'].iloc[::stride].tolist())
    ax.tick_params(axis='x', rotation=65)
    ax.grid(axis='y', alpha=.2)
    save(fig, fig_dir / '04_monthly_flagged_ratio.png')

    # 05: most flagged day if available; else first suitable daytime day.
    if len(samples):
        day_counts = samples.assign(day=samples['timestamp'].dt.normalize()).groupby('day')['source_row_id'].nunique()
        selected_day = day_counts.sort_values(ascending=False).index[0]
    else:
        selected_day = df[DATE].dt.normalize().iloc[min(len(df) - 1, 48)]
    one_day = df.loc[df[DATE].dt.normalize() == selected_day].copy()
    if len(one_day):
        fig, ax = plt.subplots(figsize=(10.5, 4.3))
        ax.plot(one_day[DATE].dt.hour, one_day['normalized_power'], '-o', markersize=3,
                label='Observed normalized output')
        ax.plot(one_day[DATE].dt.hour, one_day['ref_median'], '--', alpha=.8,
                label='Reference median (available hours)')
        flagged_today = one_day.loc[one_day['source_row_id'].isin(marked_ids)]
        if len(flagged_today):
            ax.scatter(flagged_today[DATE].dt.hour, flagged_today['normalized_power'],
                       marker='x', s=70, label='Flagged hours')
        ax.set(title=f'Example day: {selected_day.date()}', xlabel='Hour', ylabel='MWh / MW',
               xticks=range(0, 24, 2), xlim=(-.3, 23.3))
        ax.grid(alpha=.2)
        ax.legend(loc='best')
        save(fig, fig_dir / '05_example_day.png')

    pattern_summary_path = out / 'pattern_summary.json'
    source_info = json.loads(pattern_summary_path.read_text(encoding='utf-8')) if pattern_summary_path.exists() else {}
    warnings = [
        '탐지값은 추가 확인이 필요한 **후보**이며 실제 고장 진단 결과가 아닙니다.',
        '참고 분포는 전체 관측기간으로 산출한 기술적 기준으로 미래 예측 모델/독립 테스트가 아닙니다.',
        '상대적 저출력은 계절·시간대·일사량 이외의 기상/그늘/출력 제한 요인으로도 발생할 수 있습니다.',
        '충북 집계 데이터로 개별 발전소의 고장 위치나 원인을 알 수 없습니다.',
        '전처리 단계에서 제거된 원본 오류 및 날짜는 이 프로그램에서 복원되지 않습니다.',
        '연속 검사는 정확한 1시간 간격의 데이터에서만 수행합니다.',
    ]
    report = [
        '# 태양광 발전량 이상 패턴 자동 분석 보고서', '',
        f'- 데이터 파일: `{source_info.get("source", "미기록")}`',
        f'- 가상 예제 여부: **{"예" if source_info.get("demo_data") else "아니오"}**',
        f'- 유효 분석 행: **{len(df):,}행**',
        f'- 일사량 기준 탐지 대상: **{int(df["analysis_eligible"].astype(str).str.lower().eq("true").sum()):,}행**',
        f'- 이상 후보 이벤트: **{len(events):,}건**',
        f'- 이상 후보로 표시된 고유 시점: **{len(marked_ids):,}개**',
        '', '## 이벤트 유형별 결과', '',
        '| 유형 | 이벤트 건수 | 표시된 시간 합계 |', '|---|---:|---:|',
    ]
    if len(type_summary):
        for row in type_summary.itertuples(index=False):
            report.append(f'| {LABELS.get(row.anomaly_type, row.anomaly_type)} | {row.event_count} | {row.flagged_hours} |')
    else:
        report.append('| 현재 기준에서 탐지된 후보 없음 | 0 | 0 |')
    report.extend(['', '## 결과 파일', '',
                   '- `pattern_data.csv`: 정규화발전량, 기준 패턴, 분석 가능 여부',
                   '- `reference_patterns.csv`: 기준 패턴별 표본수·중앙값·하위 사분위',
                   '- `anomaly_events.csv`: 연속 이상 후보 이벤트 단위',
                   '- `anomaly_samples.csv`: 탐지된 개별 시점',
                   '- `monthly_summary.csv`: 월별 탐지 후보 시간 및 비율',
                   '- `figures/`: 5개 분석 그림',
                   '', '## 분석상의 제한', ''])
    report.extend([f'- {w}' for w in warnings])
    report.extend(['', '## 그래프', '',
                   '![일별 정규화 발전량](figures/01_daily_normalized_generation.png)', '',
                   '![일사량과 발전량](figures/02_irradiance_vs_generation.png)', '',
                   '![유형별 후보](figures/03_anomaly_events_by_type.png)', '',
                   '![월별 후보 비율](figures/04_monthly_flagged_ratio.png)', '',
                   '![예시 일자](figures/05_example_day.png)', ''])
    (out / 'report.md').write_text('\n'.join(report), encoding='utf-8')
    print(f'[03] 이벤트 {len(events):,}건 | 고유 탐지 시점 {len(marked_ids):,}개')
    print(f'[03] 보고서: {out / "report.md"} | 그래프 5개: {fig_dir}')


if __name__ == '__main__':
    main()
