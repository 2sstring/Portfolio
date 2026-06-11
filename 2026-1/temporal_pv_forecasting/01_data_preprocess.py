import pandas as pd
import numpy as np

# =========================================================
# 사용자 설정
# =========================================================
INPUT_PATH = "01. 전국 태양광 발전량 예측값 학습데이터.csv"
OUTPUT_PATH = "solar_preprocessed.csv"

ENCODING = "cp949"

# 단시간 보간 허용 길이
# - 기상 변수: 최대 6시간 연속 결측까지 보간
# - 적운량: 최대 3시간 연속 결측까지 보간
# - 주간 일조/일사량: 최대 2시간 연속 결측까지 보간
MAX_GAP_WEATHER = 6
MAX_GAP_CLOUD = 3
MAX_GAP_SOLAR = 2


# =========================================================
# 보조 함수
# =========================================================
def fill_short_gaps_by_group(df, group_col, target_col, max_gap):
    """
    지역별 시간 순서 기준으로 '짧은 연속 결측'만 선형보간한다.
    max_gap보다 긴 연속 결측은 보간하지 않고 NaN으로 남긴다.
    """
    def _fill_short_gaps(s):
        original = s.copy()
        is_na = original.isna()

        if is_na.sum() == 0:
            return original

        # 전체 선형보간값 계산
        interpolated = original.interpolate(
            method="linear",
            limit_area="inside"
        )

        # 연속 결측 구간 길이 계산
        block_id = is_na.ne(is_na.shift()).cumsum()
        gap_size = is_na.groupby(block_id).transform("sum")

        # 짧은 결측 구간만 보간값으로 대체
        fill_mask = is_na & (gap_size <= max_gap)

        result = original.copy()
        result.loc[fill_mask] = interpolated.loc[fill_mask]

        return result

    before_missing = df[target_col].isna().sum()

    df[target_col] = (
        df.groupby(group_col, group_keys=False)[target_col]
          .apply(_fill_short_gaps)
    )

    after_missing = df[target_col].isna().sum()
    print(f" - {target_col}: 보간 {before_missing - after_missing}건, 잔여 결측 {after_missing}건")

    return df


def safe_numeric_convert(df, cols):
    """존재하는 컬럼만 수치형으로 변환한다."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def print_missing_summary(df, title):
    print(f"\n[{title}] 결측 개수")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) == 0:
        print("결측 없음")
    else:
        print(missing)


# =========================================================
# 0. 데이터 로드
# =========================================================
df = pd.read_csv(INPUT_PATH, encoding=ENCODING)

print("=" * 70)
print("[0] 원본 데이터")
print("원본 shape:", df.shape)
print("원본 컬럼:", list(df.columns))
print("=" * 70)


# =========================================================
# 1. 기본 형식 정리
#    - 날짜 변환
#    - 문자열 공백 제거
#    - 수치형 컬럼 변환
# =========================================================
print("\n[1] 기본 형식 정리")

# 1-1. 발전일자 datetime 변환
before_rows = len(df)
df["발전일자"] = pd.to_datetime(df["발전일자"], errors="coerce")
df = df.dropna(subset=["발전일자"]).copy()
after_rows = len(df)

print("발전일자 datetime 변환 불가 제거:", before_rows - after_rows)

# 1-2. 문자열 컬럼 정리
if "시도명" in df.columns:
    df["시도명"] = df["시도명"].astype(str).str.strip()

if "윤년여부" in df.columns:
    df["윤년여부"] = df["윤년여부"].astype(str).str.strip()

# 1-3. 수치형 컬럼 강제 변환
numeric_cols = [
    "설비용량(MW)", "기온", "습도", "풍속",
    "적운량(10분위)", "적운량(3분위)",
    "일조(hr)", "대기권밖일사량계산값", "일사량", "발전량(MWh)"
]

df = safe_numeric_convert(df, numeric_cols)

print_missing_summary(df, "수치형 변환 후")


# =========================================================
# 2. 유일성
#    - 전국 데이터이므로 반드시 (시도명, 발전일자) 기준
#    - 중복 행 중 결측 개수가 가장 적은 행을 우선 유지
# =========================================================
print("\n[2] 유일성")

before_rows = len(df)

# 중복 행 중 더 완전한 행을 남기기 위한 결측 개수 계산
df["_missing_count"] = df.isna().sum(axis=1)

df = (
    df.sort_values(["시도명", "발전일자", "_missing_count"])
      .drop_duplicates(subset=["시도명", "발전일자"], keep="first")
      .drop(columns="_missing_count")
      .copy()
)

after_rows = len(df)

print("중복 제거 전 행 수:", before_rows)
print("중복 제거 후 행 수:", after_rows)
print("제거된 중복 행 수:", before_rows - after_rows)


# =========================================================
# 3. 완전성
#    3-1) 결측률 80% 이상 feature 제거
#    3-2) 야간 결측 0 대체
#    3-3) 단시간 기상 변수 결측 보간
#    3-4) 적운량 단시간 결측 보간
#    3-5) 주간 일조/일사량 단시간 결측 보간
#    3-6) 주간 일조/일사량 잔여 결측 제거
#    3-7) 발전량 잔여 결측 제거
#    3-8) 나머지 잔여 결측 제거
# =========================================================
print("\n[3] 완전성")

# 3-1. 결측률 80% 이상 feature 제거
missing_ratio = df.isna().mean()
drop_cols = missing_ratio[missing_ratio >= 0.8].index.tolist()

print("결측률 80% 이상 feature:", drop_cols)

df = df.drop(columns=drop_cols).copy()

# 이후 처리를 위해 시간 순서 정렬
df = df.sort_values(["시도명", "발전일자"]).copy()

# 3-2. 야간 조건
# 대기권밖일사량계산값이 0이면 물리적으로 태양광 발전과 일사량이 0에 가까운 시간으로 판단
if "대기권밖일사량계산값" not in df.columns:
    raise ValueError("대기권밖일사량계산값 컬럼이 없습니다. 야간 조건부 처리를 수행할 수 없습니다.")

mask_night = df["대기권밖일사량계산값"] == 0

# 야간 일조, 일사량, 발전량 결측 0 대체
zero_fill_cols = ["일조(hr)", "일사량", "발전량(MWh)"]

print("[야간 조건부 0 대체]")
for col in zero_fill_cols:
    if col in df.columns:
        fill_mask = mask_night & df[col].isna()
        print(f" - {col}: 0 대체 {fill_mask.sum()}건")
        df.loc[fill_mask, col] = 0

print_missing_summary(df, "야간 0 대체 후")

# 3-3. 기상 변수 단시간 결측 보간
print("\n[기상 변수 단시간 보간]")
weather_cols = ["기온", "습도", "풍속"]

for col in weather_cols:
    if col in df.columns:
        df = fill_short_gaps_by_group(
            df=df,
            group_col="시도명",
            target_col=col,
            max_gap=MAX_GAP_WEATHER
        )

# 3-4. 적운량 단시간 결측 보간
print("\n[적운량 단시간 보간]")
cloud_cols = ["적운량(10분위)", "적운량(3분위)"]

for col in cloud_cols:
    if col in df.columns:
        df = fill_short_gaps_by_group(
            df=df,
            group_col="시도명",
            target_col=col,
            max_gap=MAX_GAP_CLOUD
        )

# 3-5. 주간 일조/일사량 단시간 결측 보간
# 일조/일사량 결측 중 짧은 연속 결측만 선형보간한다.
# 특히 주간(대기권밖일사량계산값 > 0) 잔여 결측을 줄이기 위한 처리이다.
print("\n[주간 일조/일사량 단시간 보간]")
solar_cols = ["일조(hr)", "일사량"]

for col in solar_cols:
    if col in df.columns:
        before_day_missing = (
            (df["대기권밖일사량계산값"] > 0) &
            (df[col].isna())
        ).sum()

        df = fill_short_gaps_by_group(
            df=df,
            group_col="시도명",
            target_col=col,
            max_gap=MAX_GAP_SOLAR
        )

        after_day_missing = (
            (df["대기권밖일사량계산값"] > 0) &
            (df[col].isna())
        ).sum()

        print(f"   주간 {col} 결측: {before_day_missing}건 → {after_day_missing}건")

# 3-6. 주간 일조/일사량 잔여 결측 제거
day_solar_missing = pd.Series(False, index=df.index)

for col in solar_cols:
    if col in df.columns:
        day_solar_missing |= (
            (df["대기권밖일사량계산값"] > 0) &
            (df[col].isna())
        )

before_rows = len(df)
df = df[~day_solar_missing].copy()
after_rows = len(df)

print("\n주간 일조/일사량 잔여 결측 제거 수:", before_rows - after_rows)

# 3-7. 발전량 잔여 결측 제거
# 야간 발전량 결측은 이미 0으로 대체했으므로,
# 여기서 남은 발전량 결측은 학습 target 결측으로 보고 제거한다.
if "발전량(MWh)" in df.columns:
    before_rows = len(df)
    df = df.dropna(subset=["발전량(MWh)"]).copy()
    after_rows = len(df)
    print("발전량 잔여 결측 제거 수:", before_rows - after_rows)

# 3-8. 나머지 잔여 결측 제거
before_rows = len(df)
df = df.dropna().copy()
after_rows = len(df)

print("최종 잔여 결측 행 제거 전 행 수:", before_rows)
print("최종 잔여 결측 행 제거 후 행 수:", after_rows)
print("최종 잔여 결측 제거 수:", before_rows - after_rows)

print_missing_summary(df, "완전성 처리 후")

# =========================================================
# 4. 유효성
#    - 물리적 허용 범위 검증
# =========================================================
print("\n[4] 유효성")

valid_mask = pd.Series(True, index=df.index)

# 설비용량 > 0
if "설비용량(MW)" in df.columns:
    valid_mask &= df["설비용량(MW)"] > 0

# 발전량 >= 0
if "발전량(MWh)" in df.columns:
    valid_mask &= df["발전량(MWh)"] >= 0

# 기온: -30 ~ 50
if "기온" in df.columns:
    valid_mask &= df["기온"].between(-30, 50)

# 습도: 0 ~ 100
if "습도" in df.columns:
    valid_mask &= df["습도"].between(0, 100)

# 풍속 >= 0
if "풍속" in df.columns:
    valid_mask &= df["풍속"] >= 0

# 적운량(10분위): 0 ~ 10
if "적운량(10분위)" in df.columns:
    valid_mask &= df["적운량(10분위)"].between(0, 10)

# 적운량(3분위): 0 ~ 3
if "적운량(3분위)" in df.columns:
    valid_mask &= df["적운량(3분위)"].between(0, 3)

# 일조(hr): 0 ~ 1
if "일조(hr)" in df.columns:
    valid_mask &= df["일조(hr)"].between(0, 1)

# 대기권밖일사량계산값 >= 0
if "대기권밖일사량계산값" in df.columns:
    valid_mask &= df["대기권밖일사량계산값"] >= 0

# 일사량 >= 0
if "일사량" in df.columns:
    valid_mask &= df["일사량"] >= 0

before_rows = len(df)
df = df[valid_mask].copy()
after_rows = len(df)

print("유효성 범위 위반 제거:", before_rows - after_rows)


# =========================================================
# 5. 일관성
#    - 야간 조건에서 발전량 보정
# =========================================================
print("\n[5] 일관성")

if all(col in df.columns for col in ["대기권밖일사량계산값", "일사량", "발전량(MWh)"]):
    mask_night_zero = (
        (df["대기권밖일사량계산값"] == 0) &
        (df["일사량"] == 0)
    )

    count_before_nonzero = (df.loc[mask_night_zero, "발전량(MWh)"] != 0).sum()
    df.loc[mask_night_zero, "발전량(MWh)"] = 0

    print("야간 조건에서 0으로 보정된 발전량 건수:", count_before_nonzero)
else:
    print("필요 컬럼이 없어 야간 발전량 보정을 건너뜁니다.")


# =========================================================
# 6. 정확성
#    - 1시간 단위 자료이므로 발전량(MWh) <= 설비용량(MW) 조건 적용
# =========================================================
print("\n[6] 정확성")

if all(col in df.columns for col in ["발전량(MWh)", "설비용량(MW)"]):
    before_rows = len(df)

    accuracy_mask = df["발전량(MWh)"] <= df["설비용량(MW)"]
    removed_accuracy = (~accuracy_mask).sum()

    df = df[accuracy_mask].copy()
    after_rows = len(df)

    print("정확성 기준으로 제거된 이상치 수:", removed_accuracy)
    print("정확성 처리 후 행 수:", after_rows)
else:
    print("필요 컬럼이 없어 정확성 검사를 건너뜁니다.")

# =========================================================
# 7. 날짜 단위 정리
#    - 앞 단계에서 일부 시간만 삭제되어 24시간이 안 되는 날짜 전체 삭제
#    - 시도명 + 날짜 기준으로 하루 24개 행이 모두 있어야 유지
# =========================================================
print("\n[7] 날짜 단위 정리")

before_rows = len(df)

# 날짜와 시간 분리
df["_date"] = df["발전일자"].dt.date
df["_hour"] = df["발전일자"].dt.hour

# 시도명 + 날짜별 남아 있는 행 수 확인
# 1시간 단위 자료이므로 하루는 24개 행이어야 정상
day_count = (
    df.groupby(["시도명", "_date"])["발전일자"]
      .transform("count")
)

# 24시간이 모두 남아 있지 않은 날짜 찾기
incomplete_day_mask = day_count != 24

# 삭제 대상 날짜 확인
drop_dates = (
    df.loc[incomplete_day_mask, ["시도명", "_date"]]
      .drop_duplicates()
      .sort_values(["시도명", "_date"])
)

print("24시간 미만으로 남아 삭제되는 시도명-날짜 수:", len(drop_dates))

if len(drop_dates) > 0:
    print(drop_dates.to_string(index=False))

# 해당 날짜 전체 삭제
df = df.loc[~incomplete_day_mask].copy()

# 보조 컬럼 삭제
df = df.drop(columns=["_date", "_hour"], errors="ignore")

after_rows = len(df)

print("날짜 단위 정리 전 행 수:", before_rows)
print("날짜 단위 정리 후 행 수:", after_rows)
print("날짜 단위 정리로 삭제된 행 수:", before_rows - after_rows)


# =========================================================
# 8. 최종 정렬 및 저장
# =========================================================
df = df.sort_values(["시도명", "발전일자"]).copy()

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("\n" + "=" * 70)
print("[8] 최종 결과")
print("최종 shape:", df.shape)
print("저장 파일:", OUTPUT_PATH)
print("=" * 70)
