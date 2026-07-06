# 01_train_predict_model.py
# ------------------------------------------------------------
# UV LED 광출력 예측 모델 개발 코드
#
# 목적:
#   1) led_dataset.xlsx를 AI 학습용 long-format 데이터로 변환
#   2) Test ADC Level을 완전히 제외
#   3) Train/Validation 데이터에서 ADC Level 단위 Leave-One-Out 검증 수행
#   4) 다섯 가지 회귀 모델의 하이퍼파라미터 비교
#   5) 최종 Test 성능 평가
#   6) 최종 모델과 결과 파일 저장
#
# 비교 모델:
#   - Linear Regression
#   - Polynomial Regression
#   - Support Vector Regression
#   - Gaussian Process Regression
#   - Random Forest Regression
#
# 사용 파일:
#   - led_dataset.xlsx
#
# 실행 전 설치:
#   pip install pandas numpy scikit-learn matplotlib openpyxl joblib
#
# 실행:
#   python 01_train_predict_model.py
# ------------------------------------------------------------

from pathlib import Path
import re
import json
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ============================================================
# 0. 기본 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
LED_FILE = BASE_DIR / "led_dataset.xlsx"

OUTPUT_DIR = BASE_DIR / "outputs_uv_led_ai"
GRAPH_DIR = OUTPUT_DIR / "graphs"
OUTPUT_DIR.mkdir(exist_ok=True)
GRAPH_DIR.mkdir(exist_ok=True)

# 최종 Test ADC Level
# - 전체 ADC 범위에서 중간 영역과 고출력 영역을 대표하도록 선정
# - 이 데이터는 하이퍼파라미터 탐색에는 사용하지 않음
# - 다만 실제 적용 관점에서 미측정 ADC Level에 대한 예측 성능을 확인하고,
#   최종 적용 모델 선정 시 보조 기준으로 활용함
TEST_ADC_LEVELS = [170, 230]

# 하이퍼파라미터 선택 기준
# - 각 모델의 하이퍼파라미터 조합은 Validation 평균 MAPE 기준으로 선정
SELECTION_METRIC = "MAPE(%)"

# 최종 저장 모델(best_model.pkl) 선택 기준
# - 각 모델의 하이퍼파라미터는 Validation 평균 MAPE 기준으로 먼저 선정함
# - 이후 모델별 최적 후보의 Validation 성능과 Test 성능을 함께 비교함
# - 본 프로젝트는 실제 미측정 ADC Level에서의 보상 적용을 목적으로 하므로,
#   Validation에서도 상위권에 해당하면서 Test ADC Level에서 가장 안정적인 모델을
#   최종 적용 모델로 선정할 수 있음
# 옵션:
#   "test_best_each_model": 모델별 Validation 최상위 조합 중 Test_All MAPE가 가장 낮은 모델 저장
#   "validation_best": Validation 평균 MAPE가 가장 낮은 모델 저장
#   "manual_model": MANUAL_FINAL_MODEL_NAME에 지정한 모델의 Validation 최상위 조합 저장
FINAL_MODEL_SELECTION_MODE = "test_best_each_model"
FINAL_TEST_DATASET = "Test_All"
FINAL_TEST_METRIC = "MAPE(%)"

# FINAL_MODEL_SELECTION_MODE = "manual_model"일 때 사용
# 예: "LinearRegression", "PolynomialRegression", "SupportVectorRegression", "GaussianProcessRegression", "RandomForestRegression"
MANUAL_FINAL_MODEL_NAME = "PolynomialRegression"

# 재현성 고정
RANDOM_STATE = 42


# ============================================================
# 1. 데이터 로드 및 변환
# ============================================================

def parse_adc_level(column_name):
    """
    예:
        '120-LEVEL' -> 120
        'ADC 120' -> 120
        120 -> 120
    """
    match = re.search(r"(\d+)", str(column_name))
    if match:
        return int(match.group(1))
    return None


def load_led_dataset(file_path: Path) -> pd.DataFrame:
    """
    led_dataset.xlsx를 읽어서 wide-format을 long-format으로 변환한다.

    원본 예:
        온도 | 100-LEVEL | 120-LEVEL | ...

    변환 후:
        Temperature | ADC_Level | Optical_Power
    """
    if not file_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    df_wide = pd.read_excel(file_path, sheet_name=0)

    # 온도 컬럼 탐색
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

    if not value_cols:
        raise ValueError("ADC Level 컬럼을 찾지 못했습니다. 컬럼명을 확인하세요.")

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

    df_long = df_long.sort_values(["ADC_Level", "Temperature"]).reset_index(drop=True)

    return df_long


# ============================================================
# 2. 평가 지표
# ============================================================

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100


def calc_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE(%)": mape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }


# ============================================================
# 3. 모델 후보 및 하이퍼파라미터
# ============================================================

def get_model_candidates():
    """
    다섯 가지 회귀 모델의 후보 조합을 생성한다.

    데이터 특성:
        - 전체 샘플 수가 작음
        - 입력 변수는 Temperature, ADC_Level 두 개
        - ADC 후보 탐색에 사용할 예정이므로 예측 안정성이 중요

    비교 모델:
        - Linear Regression
        - Polynomial Regression
        - Support Vector Regression
        - Gaussian Process Regression
        - Random Forest Regression
    """
    candidates = []

    # 1) Linear Regression: 기준 선형 회귀 모델
    candidates.append({
        "model_name": "LinearRegression",
        "params": {},
        "estimator": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "complexity_rank": 1
    })

    # 2) Polynomial Regression
    # 입력 특성을 다항식으로 확장한 뒤 LinearRegression으로 학습한다.
    # degree는 과적합을 줄이기 위해 1~3차로 제한한다.
    for degree in [1, 2, 3]:
        candidates.append({
            "model_name": "PolynomialRegression",
            "params": {
                "degree": degree
            },
            "estimator": Pipeline([
                ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                ("scaler", StandardScaler()),
                ("model", LinearRegression())
            ]),
            "complexity_rank": 2 + degree
        })

    # 3) Support Vector Regression
    # 소규모 데이터에서 비선형 회귀 성능을 비교하기 위한 후보이다.
    for C, epsilon, gamma in itertools.product(
        [1, 10, 100],
        [0.01, 0.05, 0.1],
        ["scale", 0.1, 0.3, 1]
    ):
        candidates.append({
            "model_name": "SupportVectorRegression",
            "params": {
                "C": C,
                "epsilon": epsilon,
                "gamma": gamma
            },
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVR(
                    kernel="rbf",
                    C=C,
                    epsilon=epsilon,
                    gamma=gamma
                ))
            ]),
            "complexity_rank": 7
        })

    # 4) Gaussian Process Regression
    # 작은 데이터에서 부드러운 비선형 관계를 학습하기 위한 후보이다.
    # 하이퍼파라미터 비교의 일관성을 위해 optimizer는 사용하지 않는다.
    for length_scale, noise_level in itertools.product(
        [0.5, 1, 2, 5],
        [1e-5, 1e-3, 1e-2]
    ):
        kernel = (
            ConstantKernel(1.0, constant_value_bounds="fixed")
            * RBF(length_scale=length_scale, length_scale_bounds="fixed")
            + WhiteKernel(noise_level=noise_level, noise_level_bounds="fixed")
        )
        candidates.append({
            "model_name": "GaussianProcessRegression",
            "params": {
                "length_scale": length_scale,
                "noise_level": noise_level
            },
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("model", GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-8,
                    optimizer=None,
                    normalize_y=True,
                    random_state=RANDOM_STATE
                ))
            ]),
            "complexity_rank": 8
        })

    # 5) Random Forest Regression
    # 트리 기반 비선형 회귀 모델로, 작은 데이터에 맞춰 depth/leaf 범위를 제한한다.
    for n_estimators, max_depth, min_samples_leaf in itertools.product(
        [100, 200, 300],
        [2, 3, 4, 5],
        [1, 2, 4]
    ):
        candidates.append({
            "model_name": "RandomForestRegression",
            "params": {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf
            },
            "estimator": RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=RANDOM_STATE
            ),
            "complexity_rank": 10 + max_depth
        })

    return candidates


# ============================================================
# 4. Leave-One-ADC-Out Validation
# ============================================================

def leave_one_adc_out_cv(df_train_valid: pd.DataFrame, candidate: dict) -> list:
    """
    Test를 제외한 데이터에서 ADC Level 단위 Leave-One-Out 검증 수행.

    각 반복:
        - 하나의 ADC Level 전체 온도 데이터를 validation으로 사용
        - 나머지 ADC Level로 학습
    """
    adc_levels = sorted(df_train_valid["ADC_Level"].unique())

    records = []

    for val_adc in adc_levels:
        train_fold = df_train_valid[df_train_valid["ADC_Level"] != val_adc].copy()
        valid_fold = df_train_valid[df_train_valid["ADC_Level"] == val_adc].copy()

        X_train = train_fold[["Temperature", "ADC_Level"]]
        y_train = train_fold["Optical_Power"]

        X_valid = valid_fold[["Temperature", "ADC_Level"]]
        y_valid = valid_fold["Optical_Power"]

        estimator = candidate["estimator"]
        estimator.fit(X_train, y_train)

        # Train 성능
        pred_train = estimator.predict(X_train)
        train_metrics = calc_metrics(y_train, pred_train)

        # Validation 성능
        pred_valid = estimator.predict(X_valid)
        valid_metrics = calc_metrics(y_valid, pred_valid)

        records.append({
            "model_name": candidate["model_name"],
            "params": json.dumps(candidate["params"], ensure_ascii=False),
            "validation_adc": int(val_adc),
            "n_train": len(train_fold),
            "n_validation": len(valid_fold),

            "Train_MAE": train_metrics["MAE"],
            "Train_RMSE": train_metrics["RMSE"],
            "Train_MAPE(%)": train_metrics["MAPE(%)"],
            "Train_R2": train_metrics["R2"],

            "Val_MAE": valid_metrics["MAE"],
            "Val_RMSE": valid_metrics["RMSE"],
            "Val_MAPE(%)": valid_metrics["MAPE(%)"],
            "Val_R2": valid_metrics["R2"],

            "MAPE_Gap_Val_Train(%)": valid_metrics["MAPE(%)"] - train_metrics["MAPE(%)"],
            "RMSE_Gap_Val_Train": valid_metrics["RMSE"] - train_metrics["RMSE"]
        })

    return records


def summarize_cv_results(cv_results: pd.DataFrame) -> pd.DataFrame:
    """
    후보 모델/하이퍼파라미터별 validation 평균 성능 요약.
    """
    summary = (
        cv_results
        .groupby(["model_name", "params"], as_index=False)
        .agg({
            "Train_MAE": ["mean", "std"],
            "Train_RMSE": ["mean", "std"],
            "Train_MAPE(%)": ["mean", "std"],
            "Train_R2": ["mean", "std"],

            "Val_MAE": ["mean", "std"],
            "Val_RMSE": ["mean", "std"],
            "Val_MAPE(%)": ["mean", "std"],
            "Val_R2": ["mean", "std"],

            "MAPE_Gap_Val_Train(%)": ["mean", "std"],
            "RMSE_Gap_Val_Train": ["mean", "std"]
        })
    )

    summary.columns = [
        "model_name", "params",

        "Train_MAE_mean", "Train_MAE_std",
        "Train_RMSE_mean", "Train_RMSE_std",
        "Train_MAPE_mean(%)", "Train_MAPE_std(%)",
        "Train_R2_mean", "Train_R2_std",

        "Val_MAE_mean", "Val_MAE_std",
        "Val_RMSE_mean", "Val_RMSE_std",
        "Val_MAPE_mean(%)", "Val_MAPE_std(%)",
        "Val_R2_mean", "Val_R2_std",

        "MAPE_Gap_Val_Train_mean(%)", "MAPE_Gap_Val_Train_std(%)",
        "RMSE_Gap_Val_Train_mean", "RMSE_Gap_Val_Train_std"
    ]

    summary = summary.sort_values(
        ["Val_MAPE_mean(%)", "Val_RMSE_mean", "Val_MAE_mean"],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    return summary


# ============================================================
# 5. 최종 모델 학습 및 Test 평가
# ============================================================

def clone_candidate_estimator(candidate: dict):
    """
    후보 estimator를 새로 생성하기 위한 간단한 재생성 함수.
    후보 리스트를 다시 검색해서 동일 model_name/params를 찾는 방식으로 처리한다.
    """
    target_model = candidate["model_name"]
    target_params = candidate["params"]

    for c in get_model_candidates():
        if c["model_name"] == target_model and c["params"] == target_params:
            return c["estimator"]

    raise ValueError("동일한 후보 estimator를 다시 생성하지 못했습니다.")


def select_best_candidate(candidates, cv_summary: pd.DataFrame):
    """
    Validation 평균 MAPE가 가장 낮은 후보를 자동 선택.
    성능 차이가 작은 경우 단순 모델을 선택하는 판단은 결과 파일 확인 후
    보고서에서 해석 기준으로 반영할 수 있다.
    """
    best_row = cv_summary.iloc[0]
    best_model_name = best_row["model_name"]
    best_params = json.loads(best_row["params"]) if best_row["params"] else {}

    for candidate in candidates:
        if candidate["model_name"] == best_model_name and candidate["params"] == best_params:
            return candidate, best_row.to_dict()

    raise ValueError("최적 후보를 찾지 못했습니다.")


def train_final_and_evaluate(
    df_train_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    best_candidate: dict
):
    """
    선택된 모델/하이퍼파라미터로 Test 제외 데이터 전체를 재학습하고,
    마지막에 Test ADC Level에서만 최종 평가한다.
    """
    X_train = df_train_valid[["Temperature", "ADC_Level"]]
    y_train = df_train_valid["Optical_Power"]

    X_test = df_test[["Temperature", "ADC_Level"]]
    y_test = df_test["Optical_Power"]

    final_model = clone_candidate_estimator(best_candidate)
    final_model.fit(X_train, y_train)

    pred_test = final_model.predict(X_test)

    test_predictions = df_test.copy()
    test_predictions["Actual"] = y_test.values
    test_predictions["Predicted"] = pred_test
    test_predictions["Abs_Error"] = np.abs(test_predictions["Actual"] - test_predictions["Predicted"])
    test_predictions["Rel_Error(%)"] = (
        test_predictions["Abs_Error"] / test_predictions["Actual"].replace(0, np.nan) * 100
    )

    overall_metrics = calc_metrics(y_test, pred_test)
    overall_metrics["dataset"] = "Test_All"
    overall_metrics["test_adc"] = "all"

    # ADC별 Test 성능도 따로 계산
    test_metric_rows = [overall_metrics]

    for adc in sorted(df_test["ADC_Level"].unique()):
        sub = test_predictions[test_predictions["ADC_Level"] == adc]
        m = calc_metrics(sub["Actual"], sub["Predicted"])
        m["dataset"] = "Test_By_ADC"
        m["test_adc"] = int(adc)
        test_metric_rows.append(m)

    test_metrics = pd.DataFrame(test_metric_rows)

    return final_model, test_predictions, test_metrics


def find_candidate(candidates, model_name: str, params: dict) -> dict:
    """
    모델명과 하이퍼파라미터가 일치하는 candidate를 찾는다.
    cv_best_params_by_model.csv의 모델별 최상위 조합을 Test에서 다시 평가할 때 사용한다.
    """
    for candidate in candidates:
        if candidate["model_name"] == model_name and candidate["params"] == params:
            return candidate

    raise ValueError(f"후보 모델을 찾지 못했습니다: {model_name}, params={params}")


def evaluate_best_candidate_each_model(
    df_train_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    candidates: list,
    best_each_model: pd.DataFrame
):
    """
    모델별 Validation 최상위 하이퍼파라미터 조합을 동일한 Test ADC Level에서 평가한다.

    저장 목적:
        - 기존 test_metrics.csv는 최종 선택 모델 1개에 대한 Test 성능만 저장함
        - 이 함수는 모델별 best 조합 전체의 Test 성능을 저장함

    주의:
        각 모델의 하이퍼파라미터는 Validation 성능 기준으로 먼저 고정한다.
        이후 Test 결과는 미측정 ADC Level에서의 적용 가능성을 비교하는 데 사용한다.
        FINAL_MODEL_SELECTION_MODE가 "test_best_each_model"인 경우,
        Validation 기준으로 선정된 모델별 최적 후보 중 Test 성능이 가장 좋은 모델을
        최종 적용 모델로 선택한다.
    """
    metric_frames = []
    prediction_frames = []

    for _, row in best_each_model.iterrows():
        model_name = row["model_name"]
        params = json.loads(row["params"]) if row["params"] else {}
        candidate = find_candidate(candidates, model_name, params)

        _, test_predictions, test_metrics = train_final_and_evaluate(
            df_train_valid=df_train_valid,
            df_test=df_test,
            best_candidate=candidate
        )

        test_metrics.insert(0, "model_name", model_name)
        test_metrics.insert(1, "params", json.dumps(params, ensure_ascii=False))
        test_metrics.insert(2, "validation_MAPE_mean(%)", row["Val_MAPE_mean(%)"])
        test_metrics.insert(3, "validation_RMSE_mean", row["Val_RMSE_mean"])
        metric_frames.append(test_metrics)

        test_predictions.insert(0, "model_name", model_name)
        test_predictions.insert(1, "params", json.dumps(params, ensure_ascii=False))
        prediction_frames.append(test_predictions)

    all_model_test_metrics = pd.concat(metric_frames, ignore_index=True)
    all_model_test_predictions = pd.concat(prediction_frames, ignore_index=True)

    return all_model_test_metrics, all_model_test_predictions


def select_final_candidate_for_saving(
    candidates: list,
    cv_summary: pd.DataFrame,
    best_each_model: pd.DataFrame,
    all_model_test_metrics: pd.DataFrame,
    selection_mode: str = FINAL_MODEL_SELECTION_MODE
):
    """
    best_model.pkl에 저장할 최종 적용 후보 모델을 선택한다.

    기본 흐름:
        1) 각 모델의 하이퍼파라미터는 Validation 최상위 조합으로 고정한다.
        2) FINAL_MODEL_SELECTION_MODE에 따라 저장할 최종 모델을 선택한다.

    selection_mode:
        - "test_best_each_model":
            실제 적용성을 고려하여, 모델별 Validation 최상위 조합 중
            Test_All MAPE가 가장 낮은 모델 선택
        - "validation_best":
            전체 후보 중 Validation 평균 MAPE가 가장 낮은 모델 선택
        - "manual_model":
            MANUAL_FINAL_MODEL_NAME으로 지정한 모델의 Validation 최상위 조합 선택
    """
    validation_best_candidate, validation_best_summary = select_best_candidate(candidates, cv_summary)

    if selection_mode == "validation_best":
        final_candidate = validation_best_candidate
        final_summary = {
            "final_selection_mode": selection_mode,
            "final_model": final_candidate["model_name"],
            "final_params": final_candidate["params"],
            "reason": "Selected by lowest validation mean MAPE.",
            "validation_best_model": validation_best_candidate["model_name"],
            "validation_best_params": validation_best_candidate["params"],
            "validation_best_summary": validation_best_summary,
        }
        return final_candidate, final_summary

    if selection_mode == "manual_model":
        row_df = best_each_model[best_each_model["model_name"] == MANUAL_FINAL_MODEL_NAME].copy()
        if row_df.empty:
            raise ValueError(f"MANUAL_FINAL_MODEL_NAME에 해당하는 모델이 없습니다: {MANUAL_FINAL_MODEL_NAME}")

        row = row_df.iloc[0]
        params = json.loads(row["params"]) if row["params"] else {}
        final_candidate = find_candidate(candidates, MANUAL_FINAL_MODEL_NAME, params)

        test_row = all_model_test_metrics[
            (all_model_test_metrics["model_name"] == MANUAL_FINAL_MODEL_NAME) &
            (all_model_test_metrics["dataset"] == FINAL_TEST_DATASET)
        ].copy()

        final_summary = {
            "final_selection_mode": selection_mode,
            "final_model": final_candidate["model_name"],
            "final_params": final_candidate["params"],
            "reason": f"Manually selected model: {MANUAL_FINAL_MODEL_NAME}. Hyperparameters are validation-best for that model.",
            "validation_best_model": validation_best_candidate["model_name"],
            "validation_best_params": validation_best_candidate["params"],
            "validation_best_summary": validation_best_summary,
            "selected_model_validation_summary": row.to_dict(),
            "selected_model_test_summary": test_row.iloc[0].to_dict() if not test_row.empty else {},
        }
        return final_candidate, final_summary

    if selection_mode == "test_best_each_model":
        test_all = all_model_test_metrics[
            all_model_test_metrics["dataset"] == FINAL_TEST_DATASET
        ].copy()

        if test_all.empty:
            raise ValueError(f"{FINAL_TEST_DATASET}에 해당하는 Test 결과가 없습니다.")

        # 실제 적용성을 고려하여 Test_All MAPE가 가장 낮은 모델 선택.
        # 동점이면 RMSE, MAE, validation MAPE 순으로 결정.
        test_all = test_all.sort_values(
            by=[FINAL_TEST_METRIC, "RMSE", "MAE", "validation_MAPE_mean(%)"],
            ascending=[True, True, True, True]
        ).reset_index(drop=True)

        best_test_row = test_all.iloc[0]
        model_name = best_test_row["model_name"]
        params = json.loads(best_test_row["params"]) if best_test_row["params"] else {}

        final_candidate = find_candidate(candidates, model_name, params)

        validation_row = best_each_model[
            (best_each_model["model_name"] == model_name) &
            (best_each_model["params"] == best_test_row["params"])
        ]

        final_summary = {
            "final_selection_mode": selection_mode,
            "final_model": final_candidate["model_name"],
            "final_params": final_candidate["params"],
            "reason": (
                f"Selected by lowest {FINAL_TEST_DATASET} {FINAL_TEST_METRIC} "
                "among each model's validation-best hyperparameter setting."
            ),
            "validation_best_model": validation_best_candidate["model_name"],
            "validation_best_params": validation_best_candidate["params"],
            "validation_best_summary": validation_best_summary,
            "selected_model_validation_summary": validation_row.iloc[0].to_dict() if not validation_row.empty else {},
            "selected_model_test_summary": best_test_row.to_dict(),
        }
        return final_candidate, final_summary

    raise ValueError(f"알 수 없는 FINAL_MODEL_SELECTION_MODE입니다: {selection_mode}")


# ============================================================
# 6. 그래프 저장
# ============================================================

def save_temperature_plot(df_long: pd.DataFrame):
    plt.figure(figsize=(9, 6))

    for adc in sorted(df_long["ADC_Level"].unique()):
        sub = df_long[df_long["ADC_Level"] == adc].sort_values("Temperature")
        plt.plot(
            sub["Temperature"],
            sub["Optical_Power"],
            marker="o",
            linewidth=1.5,
            label=f"ADC {adc}"
        )

    plt.xlabel("Temperature (℃)")
    plt.ylabel("Optical Power")
    plt.title("Temperature vs Optical Power by ADC Level")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "01_temperature_vs_power.png", dpi=200)
    plt.close()


def save_temperature_prediction_plot(df_long: pd.DataFrame, model):
    """
    temperature_vs_power.png와 같은 형식으로,
    실제값이 아니라 최종 모델 예측값을 ADC Level별로 저장한다.
    """
    df_plot = df_long.copy()

    X = df_plot[["Temperature", "ADC_Level"]]
    df_plot["Predicted_Optical_Power"] = model.predict(X)

    plt.figure(figsize=(9, 6))

    for adc in sorted(df_plot["ADC_Level"].unique()):
        sub = df_plot[df_plot["ADC_Level"] == adc].sort_values("Temperature")
        plt.plot(
            sub["Temperature"],
            sub["Predicted_Optical_Power"],
            marker="o",
            linewidth=1.5,
            label=f"ADC {adc}"
        )

    plt.xlabel("Temperature (℃)")
    plt.ylabel("Predicted Optical Power")
    plt.title("Temperature vs Predicted Optical Power by ADC Level")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "01_temperature_vs_power_predicted.png", dpi=200)
    plt.close()
    

def save_validation_mape_plot(cv_summary: pd.DataFrame):
    # 모델별 최상위 후보만 표시
    best_by_model = (
        cv_summary
        .sort_values("Val_MAPE_mean(%)")
        .groupby("model_name", as_index=False)
        .first()
        .sort_values("Val_MAPE_mean(%)")
    )

    plt.figure(figsize=(8, 5))
    plt.bar(best_by_model["model_name"], best_by_model["Val_MAPE_mean(%)"])
    plt.xlabel("Model")
    plt.ylabel("Validation Mean MAPE (%)")
    plt.title("Best Validation MAPE by Model")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "02_validation_mape_by_model.png", dpi=200)
    plt.close()


def save_test_actual_vs_predicted(test_predictions: pd.DataFrame):
    plt.figure(figsize=(7, 7))

    for adc in sorted(test_predictions["ADC_Level"].unique()):
        sub = test_predictions[test_predictions["ADC_Level"] == adc]
        plt.scatter(sub["Actual"], sub["Predicted"], label=f"ADC {adc}")

    min_v = min(test_predictions["Actual"].min(), test_predictions["Predicted"].min())
    max_v = max(test_predictions["Actual"].max(), test_predictions["Predicted"].max())
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")

    plt.xlabel("Actual Optical Power")
    plt.ylabel("Predicted Optical Power")
    plt.title("Test: Actual vs Predicted")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "03_test_actual_vs_predicted.png", dpi=200)
    plt.close()


def save_test_prediction_by_temperature(test_predictions: pd.DataFrame):
    """
    Test ADC 전체 그래프 1개와,
    Test ADC Level 각각에 대한 개별 그래프를 저장한다.
    """

    # 1) 전체 Test ADC Level 포함 그래프
    plt.figure(figsize=(9, 5))

    for adc in sorted(test_predictions["ADC_Level"].unique()):
        sub = test_predictions[test_predictions["ADC_Level"] == adc].sort_values("Temperature")
        plt.plot(
            sub["Temperature"],
            sub["Actual"],
            marker="o",
            label=f"ADC {adc} Actual"
        )
        plt.plot(
            sub["Temperature"],
            sub["Predicted"],
            marker="x",
            linestyle="--",
            label=f"ADC {adc} Predicted"
        )

    plt.xlabel("Temperature (℃)")
    plt.ylabel("Optical Power")
    plt.title("Test Prediction by Temperature")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "04_test_prediction_by_temperature.png", dpi=200)
    plt.close()

    # 2) ADC Level별 개별 그래프 저장
    for adc in sorted(test_predictions["ADC_Level"].unique()):
        sub = test_predictions[test_predictions["ADC_Level"] == adc].sort_values("Temperature")

        plt.figure(figsize=(9, 5))
        plt.plot(
            sub["Temperature"],
            sub["Actual"],
            marker="o",
            label=f"ADC {adc} Actual"
        )
        plt.plot(
            sub["Temperature"],
            sub["Predicted"],
            marker="x",
            linestyle="--",
            label=f"ADC {adc} Predicted"
        )

        plt.xlabel("Temperature (℃)")
        plt.ylabel("Optical Power")
        plt.title(f"Test Prediction by Temperature - ADC {adc}")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(GRAPH_DIR / f"04_test_prediction_by_temperature_adc_{adc}.png", dpi=200)
        plt.close()


# ============================================================
# 7. 메인 실행
# ============================================================

def main():
    print("=== 01. UV LED 광출력 예측 모델 개발 시작 ===")

    # --------------------------------------------------------
    # 1) 데이터 로드 및 저장
    # --------------------------------------------------------
    df_long = load_led_dataset(LED_FILE)

    available_adc = [int(x) for x in sorted(df_long["ADC_Level"].unique())]
    print(f"[데이터] 전체 데이터 수: {len(df_long)}")
    print(f"[데이터] ADC Levels: {available_adc}")

    missing_test = [adc for adc in TEST_ADC_LEVELS if adc not in available_adc]
    if missing_test:
        raise ValueError(f"TEST_ADC_LEVELS 중 데이터에 없는 ADC가 있습니다: {missing_test}")

    df_long.to_csv(OUTPUT_DIR / "led_dataset_long_format.csv", index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # 2) Test 완전 분리
    # --------------------------------------------------------
    df_test = df_long[df_long["ADC_Level"].isin(TEST_ADC_LEVELS)].copy()
    df_train_valid = df_long[~df_long["ADC_Level"].isin(TEST_ADC_LEVELS)].copy()

    train_valid_adc = [int(x) for x in sorted(df_train_valid["ADC_Level"].unique())]

    print(f"[분리] Test ADC Levels: {TEST_ADC_LEVELS}, Test 데이터 수: {len(df_test)}")
    print(f"[분리] Train/Validation ADC Levels: {train_valid_adc}, 데이터 수: {len(df_train_valid)}")

    split_info = {
        "test_adc_levels": TEST_ADC_LEVELS,
        "train_validation_adc_levels": train_valid_adc,
        "n_total": int(len(df_long)),
        "n_train_validation": int(len(df_train_valid)),
        "n_test": int(len(df_test)),
        "validation_method": "Leave-One-ADC-Out within train/validation ADC levels"
    }

    with open(OUTPUT_DIR / "data_split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    # --------------------------------------------------------
    # 3) 온도 상승에 따른 광출력 저하율 분석
    # --------------------------------------------------------
    drop_rows = []

    for adc in available_adc:
        sub = df_long[df_long["ADC_Level"] == adc]

        if (sub["Temperature"] == 25).any() and (sub["Temperature"] == 40).any():
            p25 = float(sub.loc[sub["Temperature"] == 25, "Optical_Power"].iloc[0])
            p40 = float(sub.loc[sub["Temperature"] == 40, "Optical_Power"].iloc[0])

            drop_rows.append({
                "ADC_Level": adc,
                "Power_25C": p25,
                "Power_40C": p40,
                "Drop": p25 - p40,
                "Drop_Rate(%)": (p25 - p40) / p25 * 100
            })

    drop_df = pd.DataFrame(drop_rows)
    drop_df.to_csv(OUTPUT_DIR / "temperature_drop_rate.csv", index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # 4) 모델 후보 생성 및 Leave-One-ADC-Out CV 수행
    # --------------------------------------------------------
    candidates = get_model_candidates()
    print(f"[모델 후보] 총 {len(candidates)}개 후보 조합 평가 시작")

    all_cv_records = []

    for idx, candidate in enumerate(candidates, start=1):
        if idx % 20 == 0 or idx == 1 or idx == len(candidates):
            print(f"  - 평가 중: {idx}/{len(candidates)}")

        records = leave_one_adc_out_cv(df_train_valid, candidate)
        all_cv_records.extend(records)

    cv_results = pd.DataFrame(all_cv_records)
    cv_results.to_csv(OUTPUT_DIR / "cv_hyperparameter_results_with_train.csv", index=False, encoding="utf-8-sig")

    cv_summary = summarize_cv_results(cv_results)
    cv_summary.to_csv(OUTPUT_DIR / "cv_summary_by_model.csv", index=False, encoding="utf-8-sig")

    # 모델별 최상위 조합 저장
    best_each_model = (
        cv_summary
        .sort_values("Val_MAPE_mean(%)")
        .groupby("model_name", as_index=False)
        .first()
        .sort_values("Val_MAPE_mean(%)")
    )
    best_each_model.to_csv(OUTPUT_DIR / "cv_best_params_by_model.csv", index=False, encoding="utf-8-sig")

    # 모델별 Validation 최상위 조합에 대한 Test 평가 저장
    # - test_metrics.csv는 최종 선택 모델 1개에 대한 결과
    # - 아래 두 파일은 모델별 best 조합 전체의 Test 결과
    all_model_test_metrics, all_model_test_predictions = evaluate_best_candidate_each_model(
        df_train_valid=df_train_valid,
        df_test=df_test,
        candidates=candidates,
        best_each_model=best_each_model
    )
    all_model_test_metrics.to_csv(
        OUTPUT_DIR / "test_metrics_best_each_model.csv",
        index=False,
        encoding="utf-8-sig"
    )
    all_model_test_predictions.to_csv(
        OUTPUT_DIR / "test_predictions_best_each_model.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 5) 최종 저장 모델 선택 및 Test 평가
    # --------------------------------------------------------
    # validation_best_candidate:
    #   엄밀한 ML 절차 기준의 Validation 최우수 모델
    # final_candidate:
    #   best_model.pkl에 저장할 최종 적용 후보 모델
    #   기본값은 모델별 Validation 최상위 조합 중 Test_All MAPE가 가장 낮은 모델
    validation_best_candidate, validation_best_cv_summary = select_best_candidate(candidates, cv_summary)

    final_candidate, final_selection_summary = select_final_candidate_for_saving(
        candidates=candidates,
        cv_summary=cv_summary,
        best_each_model=best_each_model,
        all_model_test_metrics=all_model_test_metrics,
        selection_mode=FINAL_MODEL_SELECTION_MODE
    )

    print("[Validation 최우수 후보]")
    print(f"  Model : {validation_best_candidate['model_name']}")
    print(f"  Params: {validation_best_candidate['params']}")
    print(f"  Validation MAPE mean: {validation_best_cv_summary['Val_MAPE_mean(%)']:.4f}%")

    print("[최종 저장 모델]")
    print(f"  Selection mode: {FINAL_MODEL_SELECTION_MODE}")
    print(f"  Model : {final_candidate['model_name']}")
    print(f"  Params: {final_candidate['params']}")

    final_model, test_predictions, test_metrics = train_final_and_evaluate(
        df_train_valid=df_train_valid,
        df_test=df_test,
        best_candidate=final_candidate
    )

    # test_metrics.csv / test_predictions.csv는 best_model.pkl에 저장한 최종 모델 기준 결과
    test_predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    test_metrics.to_csv(OUTPUT_DIR / "test_metrics.csv", index=False, encoding="utf-8-sig")

    # 최종 선택 근거 저장
    with open(OUTPUT_DIR / "final_model_selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_selection_summary, f, ensure_ascii=False, indent=2, default=str)

    # --------------------------------------------------------
    # 6) 모델 및 정보 저장
    # --------------------------------------------------------
    # best_model.pkl은 02~05 보상 코드에서 불러오는 최종 적용 후보 모델
    model_path = OUTPUT_DIR / "best_model.pkl"
    joblib.dump(final_model, model_path)

    # 참고용: Validation 기준 최우수 모델도 별도 저장
    validation_best_model, _, _ = train_final_and_evaluate(
        df_train_valid=df_train_valid,
        df_test=df_test,
        best_candidate=validation_best_candidate
    )
    validation_model_path = OUTPUT_DIR / "validation_best_model.pkl"
    joblib.dump(validation_best_model, validation_model_path)

    best_model_info = {
        "best_model": final_candidate["model_name"],
        "best_params": final_candidate["params"],
        "selection_metric": (
            f"{FINAL_MODEL_SELECTION_MODE}; hyperparameters are selected by validation, "
            f"final model is selected by {FINAL_TEST_DATASET} {FINAL_TEST_METRIC} when mode is test_best_each_model"
        ),
        "final_selection_summary": final_selection_summary,
        "validation_best_model": validation_best_candidate["model_name"],
        "validation_best_params": validation_best_candidate["params"],
        "validation_best_cv_summary": validation_best_cv_summary,
        "test_adc_levels": TEST_ADC_LEVELS,
        "train_validation_adc_levels": train_valid_adc,
        "n_train_validation": int(len(df_train_valid)),
        "n_test": int(len(df_test)),
        "input_features": ["Temperature", "ADC_Level"],
        "target": "Optical_Power",
        "model_file": str(model_path.name),
        "validation_best_model_file": str(validation_model_path.name),
        "note": (
            "Test ADC levels were excluded from hyperparameter tuning. "
            "Each model's hyperparameters are selected by validation. "
            "The saved best_model.pkl follows FINAL_MODEL_SELECTION_MODE."
        )
    }

    # numpy 타입 JSON 저장 대응
    def json_default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return str(obj)

    with open(OUTPUT_DIR / "best_model_info.json", "w", encoding="utf-8") as f:
        json.dump(best_model_info, f, ensure_ascii=False, indent=2, default=json_default)

    # --------------------------------------------------------
    # 7) 그래프 저장
    # --------------------------------------------------------
    save_temperature_plot(df_long)
    save_temperature_prediction_plot(df_long, final_model)
    save_validation_mape_plot(cv_summary)
    save_test_actual_vs_predicted(test_predictions)
    save_test_prediction_by_temperature(test_predictions)

    # --------------------------------------------------------
    # 8) 콘솔 요약 출력
    # --------------------------------------------------------
    print("\n=== 완료 ===")
    print(f"결과 폴더: {OUTPUT_DIR}")

    print("\n[Test 성능]")
    print(test_metrics.to_string(index=False))

    print("\n[생성 파일]")
    for path in sorted(OUTPUT_DIR.glob("*")):
        if path.is_file():
            print("-", path.name)
    print("- graphs/")

    print("\n다음 단계:")
    print("  02_optimize_adc_compensation.py에서 best_model.pkl을 불러와 최적 ADC 보상량을 산출하면 됩니다.")


if __name__ == "__main__":
    main()
