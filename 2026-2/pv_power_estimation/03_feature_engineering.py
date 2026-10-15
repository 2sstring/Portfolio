import pandas as pd
import numpy as np


# ============================================================
# 0. 파일 경로 설정
# ============================================================
INPUT_FILE = "chungbuk_pv_dataset.csv"
OUTPUT_FILE = "chungbuk_pv_features.csv"

TIME_GAP_REPORT_FILE = "time_gap_report.csv"


# ============================================================
# 1. 데이터 불러오기
# ============================================================
df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

print("=" * 80)
print("[1] 원본 데이터 확인")
print("shape:", df.shape)
print("columns:", df.columns.tolist())


# ============================================================
# 2. 발전일자 datetime 변환 및 정렬
# ============================================================
df["발전일자"] = pd.to_datetime(df["발전일자"])

df = df.sort_values("발전일자").reset_index(drop=True)


# ============================================================
# 2-1. 시간 공백 확인 및 연속 구간 segment_id 생성
# ============================================================
time_diff = df["발전일자"].diff()

is_new_segment = time_diff != pd.Timedelta(hours=1)
is_new_segment.iloc[0] = True

df["segment_id"] = is_new_segment.cumsum()

gap_df = df.loc[
    time_diff.notna() & (time_diff != pd.Timedelta(hours=1)),
    ["발전일자"]
].copy()

gap_df["previous_time"] = df["발전일자"].shift(1).loc[gap_df.index]
gap_df["time_diff"] = time_diff.loc[gap_df.index]
gap_df["segment_id"] = df["segment_id"].loc[gap_df.index]

gap_df.to_csv(TIME_GAP_REPORT_FILE, index=False, encoding="utf-8-sig")


# ============================================================
# 3. Target 생성
#    정규화 발전량 = 발전량(MWh) / 설비용량(MW)
# ============================================================
df["정규화발전량"] = df["발전량(MWh)"] / df["설비용량(MW)"]


# ============================================================
# 4. 시간 변수 생성
# ============================================================
df["year"] = df["발전일자"].dt.year
df["month"] = df["발전일자"].dt.month
df["day"] = df["발전일자"].dt.day
df["hour"] = df["발전일자"].dt.hour
df["doy"] = df["발전일자"].dt.dayofyear

df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)


# ============================================================
# 5. 파생변수 생성 대상
# ============================================================
feature_base_cols = [
    "기온",
    "습도",
    "풍속",
    "적운량(10분위)",
    "일조(hr)",
    "대기권밖일사량계산값",
    "일사량",
    "정규화발전량"
]


# ============================================================
# 6. 파생변수 이름을 위한 영문명 설정
# ============================================================
name_map = {
    "기온": "temperature",
    "습도": "humidity",
    "풍속": "wind_speed",
    "적운량(10분위)": "cloud_amount_10",
    "일조(hr)": "sunshine",
    "대기권밖일사량계산값": "extraterrestrial_irradiance",
    "일사량": "irradiance",
    "정규화발전량": "normalized_power"
}


# ============================================================
# 7. Lag 변수 생성
#    1시간, 3시간, 24시간
# ============================================================
lag_list = [1, 3, 24]

for col in feature_base_cols:
    base_name = name_map[col]

    for lag in lag_list:
        df[f"{base_name}_lag_{lag}"] = (
            df.groupby("segment_id")[col].shift(lag)
        )


# ============================================================
# 8. Rolling mean 변수 생성
#    3시간, 24시간
#
#    현재 시점 값을 포함하지 않도록 shift(1) 후 rolling 적용
# ============================================================
rolling_list = [3, 24]

for col in feature_base_cols:
    base_name = name_map[col]

    for window in rolling_list:
        df[f"{base_name}_rollmean_{window}"] = (
            df.groupby("segment_id", group_keys=False)[col]
            .apply(
                lambda s, w=window: s.shift(1).rolling(
                    window=w,
                    min_periods=w
                ).mean()
            )
        )


# ============================================================
# 9. Diff 변수 생성
#    1시간, 24시간
#
#    현재 시점 값을 포함하지 않도록 과거 값 기준으로 계산
# ============================================================
diff_list = [1, 24]

for col in feature_base_cols:
    base_name = name_map[col]

    for diff in diff_list:
        df[f"{base_name}_diff_{diff}"] = (
            df.groupby("segment_id", group_keys=False)[col]
            .apply(
                lambda s, d=diff: s.shift(1) - s.shift(1 + d)
            )
        )


# ============================================================
# 10. 저장할 컬럼만 선택
#
#    제외:
#    - 시도명
#    - 윤년여부
#    - 적운량(3분위)
#
#    설비용량(MW)은 학습 feature는 아니지만
#    정규화 발전량 복원용으로 남김
# ============================================================
basic_cols = [
    "발전일자",
    "segment_id",
    "설비용량(MW)",
    "발전량(MWh)",
    "정규화발전량",
    "year",
    "month",
    "day",
    "hour",
    "doy",
    "sin_hour",
    "cos_hour",
    "기온",
    "습도",
    "풍속",
    "적운량(10분위)",
    "일조(hr)",
    "대기권밖일사량계산값",
    "일사량"
]

derived_cols = []

for eng_name in name_map.values():
    derived_cols += [
        col for col in df.columns
        if col.startswith(eng_name + "_lag_")
        or col.startswith(eng_name + "_rollmean_")
        or col.startswith(eng_name + "_diff_")
    ]

save_cols = basic_cols + derived_cols

df_save = df[save_cols].copy()


# ============================================================
# 11. 날짜 단위 결측 제거
#
#    기준:
#    하루 안에 결측이 하나라도 있으면
#    해당 날짜 전체를 삭제
# ============================================================

# 날짜만 추출
df_save["date"] = df_save["발전일자"].dt.date

# 발전일자를 제외한 나머지 컬럼에서 결측 확인
check_cols = [col for col in df_save.columns if col not in ["발전일자", "date", "segment_id"]]

# 결측이 하나라도 있는 날짜 찾기
missing_dates = df_save.loc[
    df_save[check_cols].isna().any(axis=1),
    "date"
].unique()

print("\n[삭제 대상 날짜]")
for d in missing_dates:
    print(d)

print("\n삭제 대상 날짜 수:", len(missing_dates))

# 해당 날짜 전체 삭제
df_save = df_save[~df_save["date"].isin(missing_dates)].copy()

# 임시 date 컬럼 삭제
df_save = df_save.drop(columns=["date"]).reset_index(drop=True)

# 저장
df_save.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")


# ============================================================
# 12. 결과 확인
# ============================================================
print("\n" + "=" * 80)
print("[2] 피처 생성 완료")
print("저장 데이터 shape:", df_save.shape)
print("저장 컬럼 수:", len(df_save.columns))
print("저장 파일:", OUTPUT_FILE)

print("\n[생성된 주요 컬럼 예시]")
print(df_save.columns.tolist()[:30])

print("\n[결측 개수 확인]")
print(df_save.isna().sum()[df_save.isna().sum() > 0])