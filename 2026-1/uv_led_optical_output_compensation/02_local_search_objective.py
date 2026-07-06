# 02_local_search_objective.py
# ------------------------------------------------------------
# UV LED 국소 탐색 기반 목적함수 ADC 보상 코드
#
# 목적:
#   1) 01_train_predict_model.py에서 저장한 best_model.pkl 불러오기
#   2) led_dataset.xlsx, adc_dataset.xlsx 데이터 읽기
#   3) 각 기준 ADC Level의 25℃ 광출력을 목표 광출력으로 설정
#   4) 현재 온도/기준 ADC에서 예측 광출력 저하량 계산
#   5) 25℃ ADC-광출력 관계를 이용해 예상 보상 ADC 산출
#   6) 예상 보상 ADC 주변 후보만 국소 탐색
#   7) 목표 광출력 오차 + λ × ADC 증가량을 최소화하는 ADC Level 선택
#   8) 보상 전·후 결과 및 그래프 저장
#
# 핵심 아이디어:
#   전체 탐색 기반 목적함수 방식:
#       기준 ADC ~ 255 전체 후보를 대상으로 목적함수 평가
#
#   국소 탐색 기반 목적함수 방식:
#       부족 광출력 계산
#       → 25℃ ADC-광출력 관계로 예상 보상 ADC 계산
#       → 예상 ADC 주변만 탐색
#       → 목표 오차와 ADC 증가량 벌점을 함께 고려해 목적함수가 가장 작은 ADC 선택
#
# 실행 전:
#   01_train_predict_model.py를 먼저 실행해서
#   outputs_uv_led_ai/best_model.pkl 파일이 있어야 합니다.
#
# 필요 라이브러리:
#   pip install pandas numpy scikit-learn matplotlib openpyxl joblib
#
# 실행:
#   python 05_local_search_objective.py
# ------------------------------------------------------------

from pathlib import Path
import re
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore")


# ============================================================
# 0. 기본 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

LED_FILE = BASE_DIR / "led_dataset.xlsx"
ADC_FILE = BASE_DIR / "adc_dataset.xlsx"

AI_OUTPUT_DIR = BASE_DIR / "outputs_uv_led_ai"
MODEL_FILE = AI_OUTPUT_DIR / "best_model.pkl"
MODEL_INFO_FILE = AI_OUTPUT_DIR / "best_model_info.json"

OUTPUT_DIR = BASE_DIR / "outputs_uv_led_05_local_search_objective"
GRAPH_DIR = OUTPUT_DIR / "graphs"

OUTPUT_DIR.mkdir(exist_ok=True)
GRAPH_DIR.mkdir(exist_ok=True)

# ADC 제어 범위
ADC_MIN = 0
ADC_MAX = 255

# 예상 보상 ADC 주변 탐색 폭
# 예: LOCAL_WINDOW=5이면 예상 ADC ±5 범위에서 후보 탐색
LOCAL_WINDOW = 5

# 후보 탐색 시 기준 ADC보다 낮게 내리지 않음
ALLOW_ADC_DECREASE = False

# 목표 광출력 허용오차. 결과 분석용으로 사용
TOLERANCE = 0.10

# 목적함수에서 ADC 증가량 벌점 강도
LAMBDA_ADC = 0.01

# 25℃ ADC-광출력 관계에서 기울기 계산 시 사용할 주변 폭
# base_adc 주변 ±SLOPE_WINDOW 범위로 국소 기울기를 계산
SLOPE_WINDOW = 10


# ============================================================
# 1. 데이터 로드
# ============================================================

def parse_adc_level(column_name):
    match = re.search(r"(\d+)", str(column_name))
    if match:
        return int(match.group(1))
    return None


def load_led_dataset(file_path: Path) -> pd.DataFrame:
    """
    led_dataset.xlsx를 읽어 long-format으로 변환한다.

    변환 후:
        Temperature | ADC_Level | Optical_Power
    """
    if not file_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    df_wide = pd.read_excel(file_path, sheet_name=0)

    temp_col = None
    for col in df_wide.columns:
        if "온도" in str(col) or "temp" in str(col).lower():
            temp_col = col
            break

    if temp_col is None:
        temp_col = df_wide.columns[0]

    value_cols = [
        col for col in df_wide.columns
        if col != temp_col and parse_adc_level(col) is not None
    ]

    df_long = df_wide.melt(
        id_vars=[temp_col],
        value_vars=value_cols,
        var_name="ADC_Column",
        value_name="Optical_Power"
    )

    df_long["Temperature"] = pd.to_numeric(df_long[temp_col], errors="coerce")
    df_long["ADC_Level"] = df_long["ADC_Column"].apply(parse_adc_level)
    df_long["Optical_Power"] = pd.to_numeric(df_long["Optical_Power"], errors="coerce")

    df_long = df_long[["Temperature", "ADC_Level", "Optical_Power"]].dropna()
    df_long["Temperature"] = df_long["Temperature"].astype(float)
    df_long["ADC_Level"] = df_long["ADC_Level"].astype(int)
    df_long["Optical_Power"] = df_long["Optical_Power"].astype(float)

    return df_long.sort_values(["ADC_Level", "Temperature"]).reset_index(drop=True)


def load_adc_dataset(file_path: Path) -> pd.DataFrame:
    """
    adc_dataset.xlsx를 읽어 ADC Level-25℃ 광출력 기준 데이터로 정리한다.

    이 파일은 국소 탐색 기반 목적함수 방식에서 매우 중요하다.
    부족 광출력을 ADC 증가량으로 환산하고, 국소 목적함수 탐색의 기준 후보를 설정하기 위해 사용한다.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path.name} 파일이 없습니다. 국소 탐색 기반 목적함수 방식은 adc_dataset.xlsx가 필요합니다."
        )

    df = pd.read_excel(file_path, sheet_name=0)

    adc_col = None
    power_col = None

    for col in df.columns:
        col_str = str(col)
        if "ADC" in col_str.upper() or "레벨" in col_str:
            adc_col = col
        if "광" in col_str or "출력" in col_str or "POWER" in col_str.upper():
            power_col = col

    if adc_col is None:
        adc_col = df.columns[0]
    if power_col is None:
        power_col = df.columns[1]

    out = pd.DataFrame({
        "ADC_Level": pd.to_numeric(df[adc_col], errors="coerce"),
        "Optical_Power_25C": pd.to_numeric(df[power_col], errors="coerce")
    })

    out = out.dropna().reset_index(drop=True)
    out["ADC_Level"] = out["ADC_Level"].astype(int)
    out["Optical_Power_25C"] = out["Optical_Power_25C"].astype(float)

    out = out.sort_values("ADC_Level").reset_index(drop=True)

    return out


def load_model_and_info():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"최종 모델 파일이 없습니다: {MODEL_FILE}\n"
            "먼저 01_train_predict_model.py를 실행하세요."
        )

    model = joblib.load(MODEL_FILE)

    model_info = {}
    if MODEL_INFO_FILE.exists():
        with open(MODEL_INFO_FILE, "r", encoding="utf-8") as f:
            model_info = json.load(f)

    return model, model_info


# ============================================================
# 2. 기본 함수
# ============================================================

def predict_power(model, temperature: float, adc_level: int) -> float:
    X = pd.DataFrame({
        "Temperature": [temperature],
        "ADC_Level": [adc_level]
    })
    return float(model.predict(X)[0])


def get_target_power_25c(led_df: pd.DataFrame, base_adc: int) -> float:
    """
    기준 ADC의 25℃ 광출력을 목표 광출력으로 설정한다.
    """
    rows = led_df[
        (led_df["ADC_Level"] == base_adc) &
        (led_df["Temperature"] == 25)
    ]

    if len(rows) == 0:
        raise ValueError(f"25℃ 기준 광출력을 찾을 수 없습니다. ADC={base_adc}")

    return float(rows["Optical_Power"].iloc[0])


def build_target_table(led_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for adc in sorted(led_df["ADC_Level"].unique()):
        rows.append({
            "Base_ADC": int(adc),
            "Target_Power_25C": get_target_power_25c(led_df, int(adc))
        })

    return pd.DataFrame(rows)


# ============================================================
# 3. 25℃ ADC-광출력 관계 기반 보상량 계산
# ============================================================

def estimate_local_slope(adc_df: pd.DataFrame, base_adc: int, slope_window: int = SLOPE_WINDOW) -> float:
    """
    25℃ ADC-광출력 데이터에서 base_adc 주변의 국소 기울기를 계산한다.

    기울기 의미:
        ADC Level 1 증가당 광출력 증가량

    계산 방식:
        base_adc ± slope_window 구간 데이터를 사용해 1차 회귀 기울기 계산
    """
    local = adc_df[
        (adc_df["ADC_Level"] >= base_adc - slope_window) &
        (adc_df["ADC_Level"] <= base_adc + slope_window)
    ].copy()

    # 주변 데이터가 너무 적으면 전체 데이터 사용
    if len(local) < 3:
        local = adc_df.copy()

    x = local["ADC_Level"].values.astype(float)
    y = local["Optical_Power_25C"].values.astype(float)

    if len(np.unique(x)) < 2:
        return np.nan

    slope, intercept = np.polyfit(x, y, 1)

    return float(slope)


def estimate_compensated_adc(
    adc_df: pd.DataFrame,
    base_adc: int,
    shortage_power: float
) -> dict:
    """
    부족 광출력을 25℃ ADC-광출력 관계로 ADC 증가량으로 환산한다.

    shortage_power:
        목표 광출력 - 현재 예측 광출력
        0보다 작거나 같으면 보상 증가량은 0으로 둔다.
    """
    slope = estimate_local_slope(adc_df, base_adc)

    if np.isnan(slope) or slope <= 0:
        adc_delta_est = 0
        method = "slope_invalid_delta_0"
    else:
        adc_delta_est = max(0, shortage_power / slope)
        method = "local_slope"

    estimated_adc_float = base_adc + adc_delta_est
    estimated_adc_int = int(round(estimated_adc_float))

    estimated_adc_int = min(max(estimated_adc_int, ADC_MIN), ADC_MAX)

    if not ALLOW_ADC_DECREASE:
        estimated_adc_int = max(estimated_adc_int, base_adc)

    return {
        "Local_Slope": slope,
        "Shortage_Power": shortage_power,
        "Estimated_ADC_Delta": adc_delta_est,
        "Estimated_ADC_Float": estimated_adc_float,
        "Estimated_ADC": estimated_adc_int,
        "Estimation_Method": method
    }


# ============================================================
# 4. 국소 목적함수 탐색 최적화
# ============================================================

def optimize_adc_local_objective(
    model,
    adc_df: pd.DataFrame,
    temperature: float,
    base_adc: int,
    target_power: float,
    local_window: int = LOCAL_WINDOW,
    tolerance: float = TOLERANCE,
    lambda_adc: float = LAMBDA_ADC
):
    """
    예측 기반 보상량 산출 + 국소 목적함수 탐색 기반 목적함수 최적화.

    절차:
        1) 현재 온도, 기준 ADC에서 광출력 예측
        2) 목표 광출력 대비 부족량 계산
        3) 25℃ ADC-광출력 관계로 예상 보상 ADC 산출
        4) 예상 ADC 주변 ±local_window 후보 탐색
        5) 목적함수가 가장 작은 ADC 선택

    최적화 수식:
        minimize |P_target - P_pred(T, A_comp)| + lambda * ADC_Increase

    제약:
        A_comp는 정수
        base_adc <= A_comp <= 255
        A_comp는 estimated_adc 주변 국소 후보
    """
    before_pred = predict_power(model, temperature, base_adc)
    shortage_power = max(target_power - before_pred, 0)

    est = estimate_compensated_adc(
        adc_df=adc_df,
        base_adc=base_adc,
        shortage_power=shortage_power
    )

    estimated_adc = int(est["Estimated_ADC"])

    search_start = estimated_adc - local_window
    search_end = estimated_adc + local_window

    if not ALLOW_ADC_DECREASE:
        search_start = max(search_start, base_adc)
    else:
        search_start = max(search_start, ADC_MIN)

    search_end = min(search_end, ADC_MAX)

    # 만약 범위가 비정상적으로 되면 기준 ADC~255 전체 중 최소 보정 범위로 복구
    if search_start > search_end:
        search_start = max(base_adc, ADC_MIN)
        search_end = ADC_MAX

    candidates = []

    for cand_adc in range(int(search_start), int(search_end) + 1):
        pred_power = predict_power(model, temperature, cand_adc)

        abs_error = abs(target_power - pred_power)
        rel_error = abs_error / target_power * 100 if target_power != 0 else np.nan
        adc_increase = cand_adc - base_adc
        shortage = max(target_power - pred_power, 0)
        excess = max(pred_power - target_power, 0)
        within_tolerance = abs_error <= tolerance
        objective = abs_error + lambda_adc * adc_increase

        candidates.append({
            "Temperature": temperature,
            "Base_ADC": base_adc,
            "Candidate_ADC": cand_adc,
            "Target_Power_25C": target_power,
            "Predicted_Power": pred_power,
            "Abs_Error": abs_error,
            "Rel_Error(%)": rel_error,
            "ADC_Increase": adc_increase,
            "Objective": objective,
            "Shortage": shortage,
            "Excess": excess,
            "Within_Tolerance": within_tolerance,
            "Before_Predicted_Power": before_pred,
            "Shortage_Power_Before": shortage_power,
            "Local_Slope": est["Local_Slope"],
            "Estimated_ADC_Delta": est["Estimated_ADC_Delta"],
            "Estimated_ADC_Float": est["Estimated_ADC_Float"],
            "Estimated_ADC": estimated_adc,
            "Search_Start": search_start,
            "Search_End": search_end,
            "Estimation_Method": est["Estimation_Method"]
        })

    cand_df = pd.DataFrame(candidates)

    # 1순위: 목적함수가 가장 작은 후보
    # 2순위: 목표 광출력과 절대오차가 가장 작은 후보
    # 3순위: ADC 증가량이 작은 후보
    best = cand_df.sort_values(
        by=["Objective", "Abs_Error", "ADC_Increase"],
        ascending=[True, True, True]
    ).iloc[0]

    best_dict = best.to_dict()

    if bool(best_dict["Within_Tolerance"]):
        selection_rule = "Local_Objective_Within_Tolerance"
    else:
        selection_rule = "Local_Objective_Min_Objective"

    best_dict["Selection_Rule"] = selection_rule

    return best_dict, cand_df


# ============================================================
# 5. 보상 실험
# ============================================================

def run_local_compensation_experiment(
    model,
    led_df: pd.DataFrame,
    adc_df: pd.DataFrame
):
    result_rows = []
    candidate_frames = []

    target_table = build_target_table(led_df)

    for _, target_row in target_table.iterrows():
        base_adc = int(target_row["Base_ADC"])
        target_power = float(target_row["Target_Power_25C"])

        sub = led_df[led_df["ADC_Level"] == base_adc].sort_values("Temperature")

        for _, row in sub.iterrows():
            temperature = float(row["Temperature"])
            measured_power = float(row["Optical_Power"])

            before_pred_power = predict_power(model, temperature, base_adc)
            before_abs_error = abs(target_power - before_pred_power)
            before_rel_error = before_abs_error / target_power * 100 if target_power != 0 else np.nan

            measured_before_abs_error = abs(target_power - measured_power)
            measured_before_rel_error = measured_before_abs_error / target_power * 100 if target_power != 0 else np.nan

            best, cand_df = optimize_adc_local_objective(
                model=model,
                adc_df=adc_df,
                temperature=temperature,
                base_adc=base_adc,
                target_power=target_power
            )

            candidate_frames.append(cand_df)

            optimized_adc = int(best["Candidate_ADC"])
            after_pred_power = float(best["Predicted_Power"])
            after_abs_error = abs(target_power - after_pred_power)
            after_rel_error = after_abs_error / target_power * 100 if target_power != 0 else np.nan

            result_rows.append({
                "Temperature": temperature,
                "Base_ADC": base_adc,
                "Target_Power_25C": target_power,

                "Measured_Power_Before": measured_power,
                "Measured_Before_Abs_Error": measured_before_abs_error,
                "Measured_Before_Rel_Error(%)": measured_before_rel_error,

                "Before_ADC": base_adc,
                "Before_Predicted_Power": before_pred_power,
                "Before_Abs_Error": before_abs_error,
                "Before_Rel_Error(%)": before_rel_error,

                "Estimated_ADC": int(best["Estimated_ADC"]),
                "Estimated_ADC_Float": float(best["Estimated_ADC_Float"]),
                "Estimated_ADC_Delta": float(best["Estimated_ADC_Delta"]),
                "Local_Slope": float(best["Local_Slope"]) if not pd.isna(best["Local_Slope"]) else np.nan,
                "Search_Start": int(best["Search_Start"]),
                "Search_End": int(best["Search_End"]),

                "Optimized_ADC": optimized_adc,
                "ADC_Increase": optimized_adc - base_adc,
                "After_Predicted_Power": after_pred_power,
                "After_Abs_Error": after_abs_error,
                "After_Rel_Error(%)": after_rel_error,
                "Within_Tolerance": bool(best["Within_Tolerance"]),
                "Selection_Rule": best["Selection_Rule"],
                "Shortage_After": float(best["Shortage"]),
                "Excess_After": float(best["Excess"]),
                "Candidate_Count": len(cand_df),
            })

    result_df = pd.DataFrame(result_rows)
    candidates_df = pd.concat(candidate_frames, ignore_index=True)

    return result_df, candidates_df


# ============================================================
# 6. 보상 지표 요약
# ============================================================

def summarize_compensation(result_df: pd.DataFrame) -> pd.DataFrame:
    before_mean_rel = result_df["Before_Rel_Error(%)"].mean()
    after_mean_rel = result_df["After_Rel_Error(%)"].mean()

    before_mean_abs = result_df["Before_Abs_Error"].mean()
    after_mean_abs = result_df["After_Abs_Error"].mean()

    before_max_rel = result_df["Before_Rel_Error(%)"].max()
    after_max_rel = result_df["After_Rel_Error(%)"].max()

    by_adc_rows = []

    for base_adc, adc_sub in result_df.groupby("Base_ADC"):
        before_range = adc_sub["Before_Predicted_Power"].max() - adc_sub["Before_Predicted_Power"].min()
        after_range = adc_sub["After_Predicted_Power"].max() - adc_sub["After_Predicted_Power"].min()

        before_std = adc_sub["Before_Predicted_Power"].std()
        after_std = adc_sub["After_Predicted_Power"].std()

        by_adc_rows.append({
            "Base_ADC": base_adc,
            "Before_Range": before_range,
            "After_Range": after_range,
            "Before_Std": before_std,
            "After_Std": after_std
        })

    by_adc_df = pd.DataFrame(by_adc_rows)

    total_candidate_evaluations = result_df["Candidate_Count"].sum()
    mean_candidates_per_condition = result_df["Candidate_Count"].mean()
    max_candidates_per_condition = result_df["Candidate_Count"].max()
    min_candidates_per_condition = result_df["Candidate_Count"].min()

    summary = pd.DataFrame([{
        "Method": "Prediction_Based_Local_Objective_Search",
        "Before_Mean_Abs_Error": before_mean_abs,
        "After_Mean_Abs_Error": after_mean_abs,
        "Abs_Error_Reduction(%)": (
            (before_mean_abs - after_mean_abs) / before_mean_abs * 100
            if before_mean_abs != 0 else np.nan
        ),

        "Before_Mean_Rel_Error(%)": before_mean_rel,
        "After_Mean_Rel_Error(%)": after_mean_rel,
        "Rel_Error_Reduction(%)": (
            (before_mean_rel - after_mean_rel) / before_mean_rel * 100
            if before_mean_rel != 0 else np.nan
        ),

        "Before_Max_Rel_Error(%)": before_max_rel,
        "After_Max_Rel_Error(%)": after_max_rel,

        "Before_Mean_Output_Range": by_adc_df["Before_Range"].mean(),
        "After_Mean_Output_Range": by_adc_df["After_Range"].mean(),
        "Output_Range_Reduction(%)": (
            (by_adc_df["Before_Range"].mean() - by_adc_df["After_Range"].mean())
            / by_adc_df["Before_Range"].mean() * 100
            if by_adc_df["Before_Range"].mean() != 0 else np.nan
        ),

        "Before_Mean_Output_Std": by_adc_df["Before_Std"].mean(),
        "After_Mean_Output_Std": by_adc_df["After_Std"].mean(),

        "Mean_ADC_Increase": result_df["ADC_Increase"].mean(),
        "Max_ADC_Increase": result_df["ADC_Increase"].max(),
        "Within_Tolerance_Rate(%)": result_df["Within_Tolerance"].mean() * 100,
        "Total_Candidate_Evaluations": total_candidate_evaluations,
        "Mean_Candidates_Per_Condition": mean_candidates_per_condition,
        "Max_Candidates_Per_Condition": max_candidates_per_condition,
        "Min_Candidates_Per_Condition": min_candidates_per_condition,
        "N_Conditions": len(result_df)
    }])

    return summary


def summarize_by_adc(result_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for base_adc, sub in result_df.groupby("Base_ADC"):
        rows.append({
            "Base_ADC": base_adc,
            "Target_Power_25C": sub["Target_Power_25C"].iloc[0],

            "Before_Mean_Rel_Error(%)": sub["Before_Rel_Error(%)"].mean(),
            "After_Mean_Rel_Error(%)": sub["After_Rel_Error(%)"].mean(),

            "Before_Max_Rel_Error(%)": sub["Before_Rel_Error(%)"].max(),
            "After_Max_Rel_Error(%)": sub["After_Rel_Error(%)"].max(),

            "Before_Output_Range": sub["Before_Predicted_Power"].max() - sub["Before_Predicted_Power"].min(),
            "After_Output_Range": sub["After_Predicted_Power"].max() - sub["After_Predicted_Power"].min(),

            "Before_Output_Std": sub["Before_Predicted_Power"].std(),
            "After_Output_Std": sub["After_Predicted_Power"].std(),

            "Mean_ADC_Increase": sub["ADC_Increase"].mean(),
            "Max_ADC_Increase": sub["ADC_Increase"].max(),

            "Within_Tolerance_Rate(%)": sub["Within_Tolerance"].mean() * 100,
            "Total_Candidate_Evaluations": sub["Candidate_Count"].sum(),
            "Mean_Candidates_Per_Condition": sub["Candidate_Count"].mean(),
            "Mean_Estimated_ADC_Delta": sub["Estimated_ADC_Delta"].mean()
        })

    return pd.DataFrame(rows).sort_values("Base_ADC").reset_index(drop=True)


# ============================================================
# 7. 그래프 저장
# ============================================================

def save_before_after_error_plot(result_df: pd.DataFrame):
    summary = (
        result_df
        .groupby("Temperature", as_index=False)
        .agg({
            "Before_Rel_Error(%)": "mean",
            "After_Rel_Error(%)": "mean"
        })
    )

    plt.figure(figsize=(9, 5))
    plt.plot(summary["Temperature"], summary["Before_Rel_Error(%)"], marker="o", label="Before Compensation")
    plt.plot(summary["Temperature"], summary["After_Rel_Error(%)"], marker="o", label="After Compensation")
    plt.xlabel("Temperature (℃)")
    plt.ylabel("Mean Relative Error (%)")
    plt.title("Before vs After Compensation Error: Local Objective Search")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "01_before_after_rel_error_local_objective.png", dpi=200)
    plt.close()


def save_optimized_adc_plot(result_df: pd.DataFrame):
    plt.figure(figsize=(10, 6))

    for base_adc in sorted(result_df["Base_ADC"].unique()):
        sub = result_df[result_df["Base_ADC"] == base_adc].sort_values("Temperature")
        plt.plot(
            sub["Temperature"],
            sub["Optimized_ADC"],
            marker="o",
            linewidth=1.5,
            label=f"Base ADC {base_adc}"
        )

    plt.xlabel("Temperature (℃)")
    plt.ylabel("Optimized ADC Level")
    plt.title("Optimized ADC by Temperature: Local Objective Search")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "02_optimized_adc_by_temperature_local_objective.png", dpi=200)
    plt.close()


def save_output_stability_plot(result_df: pd.DataFrame):
    # 모든 ADC Level 각각에 대해 그래프 저장
    available = sorted(result_df["Base_ADC"].unique())
    plot_adcs = available

    if not plot_adcs:
        plot_adcs = available[:5]

    for base_adc in plot_adcs:
        sub = result_df[result_df["Base_ADC"] == base_adc].sort_values("Temperature")

        plt.figure(figsize=(9, 5))
        plt.plot(sub["Temperature"], sub["Before_Predicted_Power"], marker="o", label="Before Compensation")
        plt.plot(sub["Temperature"], sub["After_Predicted_Power"], marker="o", label="After Compensation")
        plt.axhline(sub["Target_Power_25C"].iloc[0], linestyle="--", label="Target Power")
        plt.xlabel("Temperature (℃)")
        plt.ylabel("Predicted Optical Power")
        plt.title(f"Output Stability: Base ADC {base_adc} Local Objective Search")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(GRAPH_DIR / f"03_output_stability_adc_{base_adc}_local_objective.png", dpi=200)
        plt.close()


def save_estimated_vs_optimized_adc_plot(result_df: pd.DataFrame):
    plt.figure(figsize=(7, 7))
    plt.scatter(result_df["Estimated_ADC"], result_df["Optimized_ADC"])
    min_v = min(result_df["Estimated_ADC"].min(), result_df["Optimized_ADC"].min())
    max_v = max(result_df["Estimated_ADC"].max(), result_df["Optimized_ADC"].max())
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    plt.xlabel("Estimated ADC")
    plt.ylabel("Optimized ADC")
    plt.title("Estimated ADC vs Optimized ADC")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "04_estimated_vs_optimized_adc.png", dpi=200)
    plt.close()


def save_power_by_temperature_plot(result_df: pd.DataFrame):
    """
    온도에 따른 보상 전·후 광출력 비교 그래프.

    저장 이미지:
      1) 기존 대표 ADC 그래프
      2) 모든 ADC Level 포함 그래프
      3) ADC Level 170, 230만 포함한 그래프
    """
    available = sorted(result_df["Base_ADC"].unique())

    def _save_power_plot(plot_adcs, filename, title):
        if not plot_adcs:
            return

        plt.figure(figsize=(10, 6))

        for base_adc in plot_adcs:
            sub = result_df[result_df["Base_ADC"] == base_adc].sort_values("Temperature")

            plt.plot(
                sub["Temperature"],
                sub["Before_Predicted_Power"],
                marker="o",
                linestyle="--",
                label=f"ADC {base_adc} Before"
            )

            plt.plot(
                sub["Temperature"],
                sub["After_Predicted_Power"],
                marker="o",
                label=f"ADC {base_adc} After"
            )

        plt.xlabel("Temperature (℃)")
        plt.ylabel("Optical Power")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(GRAPH_DIR / filename, dpi=200)
        plt.close()

    # 1) 기존 대표 ADC 그래프 유지
    preferred_adcs = [120, 150, 180, 210]
    plot_adcs = [adc for adc in preferred_adcs if adc in available]

    if not plot_adcs:
        plot_adcs = available[:4]

    _save_power_plot(
        plot_adcs=plot_adcs,
        filename="05_power_by_temperature_local_objective.png",
        title="Optical Power Before and After Compensation: Local Objective Search"
    )

    # 2) 모든 ADC Level 포함 그래프 추가
    _save_power_plot(
        plot_adcs=available,
        filename="05_power_by_temperature_all_adc_local_objective.png",
        title="Optical Power Before and After Compensation: Local Objective Search - All ADC"
    )

    # 3) ADC Level 170, 230만 포함 그래프 추가
    selected_adcs = [adc for adc in [170, 230] if adc in available]

    _save_power_plot(
        plot_adcs=selected_adcs,
        filename="05_power_by_temperature_adc_170_230_local_objective.png",
        title="Optical Power Before and After Compensation: Local Objective Search - ADC 170, 230"
    )


# ============================================================
# 8. 메인 실행
# ============================================================

def main():
    print("=== 05. 국소 목적함수 탐색 기반 목적함수 최적화 시작 ===")

    model, model_info = load_model_and_info()
    led_df = load_led_dataset(LED_FILE)
    adc_df = load_adc_dataset(ADC_FILE)

    print(f"[모델] {MODEL_FILE}")
    if model_info:
        print(f"[모델 정보] best_model = {model_info.get('best_model')}, params = {model_info.get('best_params')}")

    print(f"[데이터] LED 데이터 수: {len(led_df)}")
    print(f"[데이터] ADC 기준 데이터 수: {len(adc_df)}")
    print(f"[설정] LOCAL_WINDOW = ±{LOCAL_WINDOW}, SLOPE_WINDOW = ±{SLOPE_WINDOW}")

    # 정리 데이터 저장
    led_df.to_csv(OUTPUT_DIR / "led_dataset_long_format.csv", index=False, encoding="utf-8-sig")
    adc_df.to_csv(OUTPUT_DIR / "adc_dataset_clean.csv", index=False, encoding="utf-8-sig")

    target_table = build_target_table(led_df)
    target_table.to_csv(OUTPUT_DIR / "target_power_25c_by_adc.csv", index=False, encoding="utf-8-sig")

    # 최적화 수행
    result_df, candidates_df = run_local_compensation_experiment(model, led_df, adc_df)

    result_df.to_csv(OUTPUT_DIR / "local_compensation_results.csv", index=False, encoding="utf-8-sig")
    candidates_df.to_csv(OUTPUT_DIR / "local_adc_optimization_candidates.csv", index=False, encoding="utf-8-sig")

    summary_df = summarize_compensation(result_df)
    summary_by_adc = summarize_by_adc(result_df)

    summary_df.to_csv(OUTPUT_DIR / "local_compensation_summary.csv", index=False, encoding="utf-8-sig")
    summary_by_adc.to_csv(OUTPUT_DIR / "local_compensation_summary_by_adc.csv", index=False, encoding="utf-8-sig")

    optimization_info = {
        "method": "Prediction-based compensation amount estimation + local objective-based discrete search",
        "target_definition": "Target optical power is the optical power at 25C for each base ADC level.",
        "decision_variable": "Optimized ADC Level",
        "objective_function": "Minimize |Target Power - Predicted Power| + lambda * ADC_Increase within local ADC candidates.",
        "constraints": [
            "ADC is integer",
            "0 <= ADC <= 255",
            "Optimized ADC >= Base ADC when ALLOW_ADC_DECREASE is False"
        ],
        "local_window": LOCAL_WINDOW,
        "slope_window": SLOPE_WINDOW,
        "tolerance": TOLERANCE,
        "lambda_adc": LAMBDA_ADC,
        "allow_adc_decrease": ALLOW_ADC_DECREASE,
        "model_file": str(MODEL_FILE),
        "model_info": model_info
    }

    with open(OUTPUT_DIR / "local_optimization_info.json", "w", encoding="utf-8") as f:
        json.dump(optimization_info, f, ensure_ascii=False, indent=2)

    # 그래프
    save_before_after_error_plot(result_df)
    save_optimized_adc_plot(result_df)
    save_output_stability_plot(result_df)
    save_estimated_vs_optimized_adc_plot(result_df)
    save_power_by_temperature_plot(result_df)

    print("\n=== 완료 ===")
    print(f"결과 폴더: {OUTPUT_DIR}")

    print("\n[보상 요약]")
    print(summary_df.to_string(index=False))

    print("\n[생성 파일]")
    for path in sorted(OUTPUT_DIR.glob("*")):
        if path.is_file():
            print("-", path.name)
    print("- graphs/")

    print("\n발표에서 주로 볼 파일:")
    print("  - local_compensation_summary.csv")
    print("  - local_compensation_summary_by_adc.csv")
    print("  - local_compensation_results.csv")
    print("  - graphs/01_before_after_rel_error_local_objective.png")
    print("  - graphs/02_optimized_adc_by_temperature_local_objective.png")


if __name__ == "__main__":
    main()
