# pv_ai/pv_forecast_node.py

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32

from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import pvlib

# InfluxDB 2.x client
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ===== 태양광 발전소 위치 (오프라인 전처리 때 사용한 값과 동일하게 맞추는 게 좋음) =====
LATITUDE = 36.969965
LONGITUDE = 127.871715
TIMEZONE = "Asia/Seoul"


class FeatureBuffer:
    """
    최근 /pv/sample JSON 데이터들을 모아서
    feature_cols.json에 맞는 1개 feature vector를 만들어주는 버퍼.

    /pv/sample에서 들어오는 ac_power는 W 단위이고,
    Target은 모델 학습 스케일에 맞게 kW로 변환하여 사용한다.
    """

    def __init__(self, max_len=5000):
        self.max_len = max_len
        self.rows = []
        # 마지막 시점의 부가정보 (고장 판정/로그용)
        self.last_context = None

    def append_from_json(self, json_dict: dict):
        """
        pv_simulator_node.py와 동일한 JSON을 입력으로 받아 내부 버퍼에 1행 추가.
        json 예:
            {
              "ts": "2025-11-30T12:00:00",
              "inverter_id": "INV1",
              "ac_power": 1234.5,         # W 단위
              "dni": 800.0,
              "dhi": 100.0,
              "ws": 2.3,
              "rh": 45.6,
              "status": "OK",
              "ghi_est": 900.0,            # (선택)
              "module_temp": 35.0,         # (필수로 넣어주면 좋음)
              "hour": 13                   # (publisher에서 넘겨주는 정수 시각)
            }
        """
        ts = json_dict.get("ts")
        try:
            ts_dt = pd.to_datetime(ts)
        except Exception:
            ts_dt = datetime.now()

        # Target = 출력 (모델 학습 스케일 = kW)
        p = json_dict.get("ac_power")
        if p is None:
            p = json_dict.get("power")
        if p is None:
            return

        try:
            p = float(p)
        except Exception:
            return

        # 지금 들어오는 값은 W이므로 kW로 변환해서 Target에 저장
        p_kw = p / 1000.0

        # 기상/운전 피처들
        def _to_float(key):
            v = json_dict.get(key)
            try:
                return float(v) if v is not None else np.nan
            except Exception:
                return np.nan

        dni = _to_float("dni")
        dhi = _to_float("dhi")
        ws = _to_float("ws")
        rh = _to_float("rh")

        # GHI_est는 JSON에 들어오면 사용, 없으면 나중에 계산
        ghi_est = _to_float("ghi_est")

        # 모듈 온도 (학습 때 사용했으므로 JSON에 넣어주는 게 가장 좋음)
        module_temp = _to_float("module_temp")
        if np.isnan(module_temp):
            module_temp = _to_float("Module_Temperature")
        if np.isnan(module_temp):
            module_temp = _to_float("ambient_temp")

        # hour (publisher에서 int로 넣어주는 값)
        hour_val = json_dict.get("hour")
        try:
            hour_val = int(hour_val) if hour_val is not None else None
        except Exception:
            hour_val = None

        row = {
            "Datetime": ts_dt,
            "Target": p_kw,  # kW 단위
            "DNI": dni,
            "DHI": dhi,
            "WS": ws,
            "RH": rh,
            "GHI_est": ghi_est,
            "Module_Temperature": module_temp,
            "hour": hour_val,
        }

        self.rows.append(row)
        if len(self.rows) > self.max_len:
            self.rows = self.rows[-self.max_len:]

    def make_feature_row(self, feature_cols, core_feats=None, optional_feats=None):
        """
        현재 버퍼로부터 feature_cols 순서에 맞는 1행 feature vector 생성.

        feature_cols는 파일에서 로드한 최종 피처 목록:
          ["Target_lag1","Target_roll_mean3","Target_lag3",
           "DNI","DHI","GHI_est", ... , "hour_cos"]

        core_feats / optional_feats:
          - core_feats: 반드시 NaN 없이 있어야 하는 피처
          - optional_feats: 없거나 NaN이면 0.0으로 채워서 사용

        낮 구간(alt>0)에서는 학습 전처리와 동일하게,
        밤 구간(alt<=0) 마지막 시점일 때는 기존 방식으로 lag/rolling 계산.
        """
        # lag3 + shift(1).rolling(3)까지 고려하면 최소 4개 정도는 있어야 안정적
        if len(self.rows) < 4:
            return None

        df = pd.DataFrame(self.rows)
        df = df.sort_values("Datetime").reset_index(drop=True)

        # ----- 0) 물리 범위 기반 클리닝 (학습 전처리와 유사) -----
        if "DHI" in df.columns:
            df.loc[df["DHI"] < 0, "DHI"] = np.nan
        if "DNI" in df.columns:
            df.loc[df["DNI"] < 0, "DNI"] = np.nan
        if "Module_Temperature" in df.columns:
            df.loc[
                (df["Module_Temperature"] < -20) |
                (df["Module_Temperature"] > 90),
                "Module_Temperature"
            ] = np.nan
        if "WS" in df.columns:
            df.loc[(df["WS"] < 0) | (df["WS"] > 25), "WS"] = np.nan
        if "RH" in df.columns:
            df.loc[(df["RH"] < 0) | (df["RH"] > 100), "RH"] = np.nan
        if "Target" in df.columns:
            df.loc[df["Target"] < 0, "Target"] = np.nan

        # 센서 데이터 짧은 구멍은 전방 채우기로 메꿈
        for col in ["DHI", "DNI", "Module_Temperature", "WS", "RH", "Target"]:
            if col in df.columns:
                df[col] = df[col].ffill()

        # ----- 1) 시간 기반 -----
        df["Datetime"] = pd.to_datetime(df["Datetime"])

        # JSON에서 넘어온 hour가 있으면 우선 사용, 없으면 Datetime에서 추출
        if "hour" in df.columns:
            df["Hour"] = df["hour"].astype("Int64")
            mask_na = df["Hour"].isna()
            if mask_na.any():
                df.loc[mask_na, "Hour"] = df.loc[mask_na, "Datetime"].dt.hour
            df["Hour"] = df["Hour"].astype(int)
        else:
            df["Hour"] = df["Datetime"].dt.hour

        df["hour_float"] = df["Hour"].astype(float)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_float"] / 24.0)

        # ----- 2) 태양고도/천정각 (pvlib 사용) -----
        try:
            times = df["Datetime"].dt.tz_localize(
                TIMEZONE,
                nonexistent="shift_forward",
                ambiguous="NaT"
            )
        except TypeError:
            # 이미 tz-aware이면 tz_convert
            times = df["Datetime"].dt.tz_convert(TIMEZONE)

        sp = pvlib.solarposition.get_solarposition(
            time=times,
            latitude=LATITUDE,
            longitude=LONGITUDE
        )
        df["solar_altitude"] = sp["apparent_elevation"].values
        df["solar_zenith"] = sp["apparent_zenith"].values

        # ----- 3) GHI_est 없으면 DNI/DHI + 태양고도로 근사 -----
        if "GHI_est" not in df.columns:
            df["GHI_est"] = np.nan

        missing_ghi = df["GHI_est"].isna()
        if missing_ghi.any():
            cos_z = np.cos(np.radians(df["solar_zenith"]))
            df.loc[missing_ghi, "GHI_est"] = (
                df.loc[missing_ghi, "DNI"] * cos_z[missing_ghi] +
                df.loc[missing_ghi, "DHI"]
            )
        df["GHI_est"] = df["GHI_est"].clip(lower=0.0)

        # ===== 4) lag / rolling / std 계산 =====
        # 마지막 시점이 '낮'이면 → 학습 전처리와 동일하게
        # solar_altitude > 0 인 행만 따로 떼서 lag/rolling 계산
        is_day_last = False
        try:
            alt_last = df["solar_altitude"].iloc[-1]
            if not pd.isna(alt_last) and alt_last > 0.0:
                is_day_last = True
        except Exception:
            pass

        if is_day_last:
            # 낮/밤 플래그
            df["is_day"] = df["solar_altitude"] > 0.0
            df_day = df[df["is_day"]].copy()

            # 학습 전처리(add_time_and_lag_features)와 동일 구조
            for col in ["Target", "DNI", "DHI", "Module_Temperature"]:
                df_day[f"{col}_lag1"] = df_day[col].shift(1)
                df_day[f"{col}_lag3"] = df_day[col].shift(3)
                df_day[f"{col}_roll_mean3"] = df_day[col].shift(1).rolling(3).mean()
                if col in ["DHI", "Module_Temperature"]:
                    df_day[f"{col}_roll_std3"] = df_day[col].shift(1).rolling(3).std()

            # day에서 계산된 결과를 원본 df에 인덱스 맞춰 붙여넣기
            for c in [
                "Target_lag1", "Target_lag3", "Target_roll_mean3",
                "DNI_lag1", "DNI_lag3", "DNI_roll_mean3",
                "DHI_lag1", "DHI_lag3", "DHI_roll_mean3", "DHI_roll_std3",
                "Module_Temperature_lag1", "Module_Temperature_lag3",
                "Module_Temperature_roll_mean3", "Module_Temperature_roll_std3",
            ]:
                if c in df_day.columns:
                    df[c] = df_day[c]

        else:
            # 마지막 시점이 밤이면 → 기존 방식(낮+밤 모두 포함) 유지
            df["Target_lag1"] = df["Target"].shift(1)
            df["Target_lag3"] = df["Target"].shift(3)
            df["Target_roll_mean3"] = df["Target"].shift(1).rolling(3).mean()

            df["DNI_lag1"] = df["DNI"].shift(1)
            df["DNI_lag3"] = df["DNI"].shift(3)
            df["DNI_roll_mean3"] = df["DNI"].shift(1).rolling(3).mean()

            df["DHI_lag1"] = df["DHI"].shift(1)
            df["DHI_lag3"] = df["DHI"].shift(3)
            df["DHI_roll_mean3"] = df["DHI"].shift(1).rolling(3).mean()
            df["DHI_roll_std3"] = df["DHI"].shift(1).rolling(3).std()

            df["Module_Temperature_lag1"] = df["Module_Temperature"].shift(1)
            df["Module_Temperature_lag3"] = df["Module_Temperature"].shift(3)
            df["Module_Temperature_roll_mean3"] = df["Module_Temperature"].shift(1).rolling(3).mean()
            df["Module_Temperature_roll_std3"] = df["Module_Temperature"].shift(1).rolling(3).std()

        # ----- 마지막 시점 1개 선택 + last_context 저장 -----
        last = df.iloc[-1]

        try:
            self.last_context = {
                "Datetime": last.get("Datetime"),
                "solar_altitude": float(last.get("solar_altitude", np.nan)),
                "solar_zenith": float(last.get("solar_zenith", np.nan)),
                "GHI_est": float(last.get("GHI_est", np.nan)),
                "Target_kw": float(last.get("Target", np.nan)),
                "DNI": float(last.get("DNI", np.nan)),
                "DHI": float(last.get("DHI", np.nan)),
            }
        except Exception:
            self.last_context = None

        # core/optional 구분 리스트가 없으면 전체를 core로 취급
        if core_feats is None:
            core_feats = list(feature_cols)
        if optional_feats is None:
            optional_feats = []

        feat_values = []

        for c in feature_cols:
            if c not in last.index:
                if c in optional_feats:
                    # optional 피처는 없으면 0.0으로
                    val = 0.0
                    print(f"[DEBUG] optional feature {c} missing → filled 0.0")
                else:
                    # core 피처가 아예 없으면 예측 skip
                    print(f"[DEBUG] core feature {c} is missing → skip forecast")
                    return None
            else:
                val = last[c]
                if pd.isna(val):
                    if c in optional_feats:
                        print(f"[DEBUG] optional feature {c} is NaN → filled 0.0")
                        val = 0.0
                    else:
                        print(f"[DEBUG] core feature {c} is NaN → skip forecast")
                        return None

            feat_values.append(val)

        return np.array(feat_values, dtype=float)


class PVForecastNode(Node):
    def __init__(self):
        super().__init__('pv_forecast_node')

        # ===== 모델 / 가중치 / 피처 목록 로딩 =====
        model_dir = "/root/pv_models"  # docker run에서 -v ~/pv_models:/root/pv_models 로 마운트했다고 가정

        self.get_logger().info(f"Loading models from {model_dir} ...")

        # LightGBM (sklearn wrapper)
        self.lgbm = joblib.load(f"{model_dir}/lgbm_model.pkl")

        # SARIMAX 결과 (wrapper 또는 statsmodels results)
        try:
            self.sarimax = joblib.load(f"{model_dir}/sarimax_model.pkl")
        except Exception:
            self.sarimax = SARIMAXResults.load(f"{model_dir}/sarimax_model.pkl")

        self.get_logger().info(f"SARIMAX type: {type(self.sarimax)}")
        self.get_logger().info(f"LGBM model type: {type(self.lgbm)}")
        self.get_logger().info(f"LGBM.predict attr type: {type(self.lgbm.predict)}")

        # 앙상블 가중치 / 피처 목록
        with open(f"{model_dir}/ensemble_weights.json", "r", encoding="utf-8") as f:
            self.weights = json.load(f)
        with open(f"{model_dir}/feature_cols.json", "r", encoding="utf-8") as f:
            self.feature_cols = json.load(f)

        self.get_logger().info(f"Loaded feature columns: {self.feature_cols}")
        self.get_logger().info(f"Loaded ensemble weights: {self.weights}")

        # core / optional 피처 분리 (훈련 코드의 core_feats / optional_feats와 일치)
        self.core_feats = [
            "Target_lag1",
            "Target_roll_mean3",
            "Target_lag3",
            "DNI", "DHI", "GHI_est",
            "DNI_lag1", "DNI_lag3", "DNI_roll_mean3",
            "DHI_lag1", "DHI_lag3", "DHI_roll_mean3",
            "Module_Temperature",
            "Module_Temperature_lag1",
            "Module_Temperature_lag3",
            "Module_Temperature_roll_mean3",
            "solar_altitude", "solar_zenith",
            "hour_cos",
        ]
        self.optional_feats = [
            "WS", "RH",
            "Module_Temperature_roll_std3",
            "DHI_roll_std3",
            "Hour",
        ]

        self.buffer = FeatureBuffer()

        # ===== 고장 판정 파라미터 (하이브리드: 절대 + 상대) =====
        # 최소 절대 오차 허용 (kW) → 0.03 kW = 30 W
        self.abs_tol_kw = 0.03

        # 상대 오차 허용 비율 (30%)
        self.rel_tol = 0.30

        self.min_power_for_eval = 20.0    # W, 이보다 작으면 "거의 0W"

        # === 학습데이터와 동일한 낮/밤 기준: solar_altitude > 0° 만 낮으로 사용 ===
        self.day_altitude_deg = 0.0       # 고도 0도 초과일 때만 "낮"
        self.min_ghi_for_fault = 50.0     # irradiance 기반 UNDER fault용 기준

        # ===== ROS2 통신 설정 =====
        self.sub = self.create_subscription(String, '/pv/sample', self.cb_sample, 10)
        self.pub = self.create_publisher(Float32, '/pv/forecast', 10)

        # ===== InfluxDB 2.x 클라이언트 초기화 =====
        self.influx_client = None
        self.influx_write_api = None
        self.influx_bucket = None
        try:
            influx_url = os.getenv("INFLUX_URL", "http://localhost:8086")
            influx_token = os.getenv("INFLUX_TOKEN", "")
            influx_org = os.getenv("INFLUX_ORG", "my-org")
            influx_bucket = os.getenv("INFLUX_BUCKET", "pv")

            if influx_token:
                self.influx_client = InfluxDBClient(
                    url=influx_url,
                    token=influx_token,
                    org=influx_org,
                )
                self.influx_write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
                self.influx_bucket = influx_bucket
                self.get_logger().info(
                    f"InfluxDB enabled: url={influx_url}, org={influx_org}, bucket={influx_bucket}"
                )
            else:
                self.get_logger().warning(
                    "INFLUX_TOKEN not set → InfluxDB write disabled."
                )
        except Exception as e:
            self.get_logger().warning(
                f"Failed to initialize InfluxDB client: {repr(e)}"
            )
            self.influx_client = None
            self.influx_write_api = None

        self.get_logger().info("PVForecastNode (LGBM + SARIMAX ensemble + fault detection) started.")

    # ===== InfluxDB write helper =====
    def write_to_influx(
        self,
        raw_json: dict,
        ctx: dict,
        status: str,
        p_actual_w,
        p_pred_w,
        err_w,
        rel_err_percent,
        threshold_w,
    ):
        if self.influx_write_api is None or self.influx_bucket is None:
            return

        # timestamp
        ts = raw_json.get("ts")
        try:
            ts_dt = pd.to_datetime(ts)
        except Exception:
            ts_dt = datetime.utcnow()

        # inverter_id tag
        inverter_id = raw_json.get("inverter_id", "UNKNOWN")

        # context (태양/irradiance)
        altitude = None
        solar_zenith = None
        ghi_est = None
        dni = None
        dhi = None

        if ctx is not None:
            altitude = ctx.get("solar_altitude")
            solar_zenith = ctx.get("solar_zenith")
            ghi_est = ctx.get("GHI_est")
            dni = ctx.get("DNI")
            dhi = ctx.get("DHI")

        # raw 환경값도 있으면 사용
        def _to_float(key):
            v = raw_json.get(key)
            try:
                return float(v) if v is not None else None
            except Exception:
                return None

        ws = _to_float("ws")
        rh = _to_float("rh")
        module_temp = (
            _to_float("module_temp")
            or _to_float("Module_Temperature")
            or _to_float("ambient_temp")
        )

        p = Point("pv_forecast") \
            .tag("inverter_id", str(inverter_id)) \
            .tag("status", str(status)) \
            .time(ts_dt, WritePrecision.NS)

        # fields (None이면 스킵)
        def field_if(name, value):
            nonlocal p
            if value is not None:
                try:
                    p = p.field(name, float(value))
                except Exception:
                    pass

        field_if("p_actual_w", p_actual_w)
        field_if("p_pred_w", p_pred_w)
        if p_pred_w is not None:
            field_if("p_pred_kw", p_pred_w / 1000.0)
        field_if("err_w", err_w)
        field_if("rel_err_percent", rel_err_percent)
        field_if("threshold_w", threshold_w)

        # ★★★ 여기부터 추가 ★★★
        # status 문자열을 보고 고장 여부를 0/1로 계산
        status_str = str(status) if status is not None else ""
        is_fault = 1.0 if status_str.startswith("FAULT") else 0.0
        field_if("is_fault", is_fault)
        # ★★★ 추가 끝 ★★★

        field_if("dni", dni)
        field_if("dhi", dhi)
        field_if("ws", ws)
        field_if("rh", rh)
        field_if("module_temp", module_temp)

        field_if("ghi_est", ghi_est)
        field_if("solar_altitude", altitude)
        field_if("solar_zenith", solar_zenith)

        try:
            self.influx_write_api.write(bucket=self.influx_bucket, record=p)
        except Exception as e:
            self.get_logger().warning(f"Failed to write to InfluxDB: {repr(e)}")


    def cb_sample(self, msg: String):
        # 1) JSON 파싱
        try:
            d = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warning(f"JSON parse fail: {e}")
            return

        # 2) 버퍼에 raw 데이터 추가 (여기서 Target은 kW로 변환됨)
        self.buffer.append_from_json(d)

        # 3) feature vector 생성 (학습 전처리와 최대한 유사)
        X = self.buffer.make_feature_row(
            self.feature_cols,
            core_feats=self.core_feats,
            optional_feats=self.optional_feats,
        )
        if X is None:
            # 아직 lag/rolling 계산에 필요한 데이터 부족 or 필수 피처 NaN
            return

        # 4) LGBM 예측 (booster_.predict로 안정적으로 우회) — kW 단위
        booster = getattr(self.lgbm, "booster_", None)
        if booster is None:
            self.get_logger().error("[CB] lgbm.booster_ is None - cannot predict")
            return

        try:
            y_lgbm_kw = float(booster.predict(X.reshape(1, -1))[0])  # kW
        except Exception as e:
            self.get_logger().warning(f"LGBM booster_.predict error: {repr(e)}")
            return

        # 5) SARIMAX 예측 (kW 단위)
        try:
            try:
                y_sarimax_kw = float(self.sarimax.forecast(steps=1)[0])
            except Exception:
                fc = self.sarimax.get_forecast(steps=1)
                pm = getattr(fc, "predicted_mean", None)
                if pm is not None:
                    y_sarimax_kw = float(pm.iloc[0])
                else:
                    y_sarimax_kw = float(fc[0])
        except Exception as e:
            self.get_logger().warning(f"SARIMAX forecast error: {repr(e)}")
            y_sarimax_kw = y_lgbm_kw

        # 6) 앙상블 가중치 적용 (kW 단위)
        w_lgbm = self.weights.get("lgbm", 0.5)
        w_sarimax = self.weights.get("sarimax", 0.5)
        y_hat_kw = w_lgbm * y_lgbm_kw + w_sarimax * y_sarimax_kw  # kW

        # 7) publish (W 단위 예측 발전량로 변환 후 publish)
        y_hat_w = y_hat_kw * 1000.0
        out = Float32()
        out.data = float(y_hat_w)
        self.pub.publish(out)

        # ===== last_context (태양고도, GHI 등) =====
        ctx = getattr(self.buffer, "last_context", None)
        if ctx is not None:
            altitude = ctx.get("solar_altitude", None)
            ghi_est = ctx.get("GHI_est", None)
        else:
            altitude = None
            ghi_est = None

        # ===== 실제 값 (W 단위) =====
        if 'ac_power' in d:
            p = d.get("ac_power")
        else:
            p = d.get("power")

        try:
            p_w = float(p)
        except Exception:
            p_w = None

        # ★ 낮/밤 상관없이, 실제값이 완전히 없으면 먼저 NO_ACTUAL로 처리 ★
        if p_w is None:
            status = "NO_ACTUAL"
            self.get_logger().info(
                f"[NO_ACTUAL] P_actual=NA W → P_pred={y_hat_w:.1f} W"
            )
            self.write_to_influx(
                raw_json=d,
                ctx=ctx,
                status=status,
                p_actual_w=None,
                p_pred_w=y_hat_w,
                err_w=None,
                rel_err_percent=None,
                threshold_w=None,
            )
            return

        # ===== 낮/밤 판정 (학습데이터와 동일: solar_altitude > 0° 만 낮) =====
        is_night = False
        if altitude is not None and altitude <= self.day_altitude_deg:  # 0도 이하면 밤
            is_night = True

        # 공통적으로 쓸 변수 기본값
        err_w = None
        rel_err_percent = None
        threshold_w = None
        status = "NORMAL"

        # ===== 밤 처리: 고장 판정 안 하고 [NIGHT] + err/rel_err/thr 로그만 =====
        if is_night:
            # 예측이 음수면 0으로 클리핑 (물리적으로 의미 없으니까)
            if y_hat_w < 0.0:
                y_hat_w = 0.0
                y_hat_kw = 0.0

            # 평가용 실제값 (없으면 0으로 보고 err 계산)
            p_eval = p_w

            err_w = abs(y_hat_w - p_eval)

            # kW 기준 threshold 계산 (낮과 동일한 방식으로 계산해서 thr 로그만)
            p_kw = p_eval / 1000.0
            rel_base_kw = self.rel_tol * max(p_kw, 1e-6)
            threshold_kw = max(self.abs_tol_kw, rel_base_kw)
            threshold_w = threshold_kw * 1000.0

            status = "NIGHT"

            # 로그 메시지 구성
            if p_eval >= 1e-3:
                rel_err_percent = (err_w / p_eval) * 100.0
                info_msg = (
                    f"[NORMAL] P_actual={p_eval:.1f} W → "
                    f"P_pred={y_hat_w:.1f} W "
                    f"(night, err={err_w:.1f} W, "
                    f"rel_err={rel_err_percent:.1f}%, "
                    f"thr={threshold_w:.1f} W)"
                )
            else:
                # 실제값 거의 0W이면 rel_err는 생략
                info_msg = (
                    f"[NORMAL] P_actual={p_eval:.1f} W → "
                    f"P_pred={y_hat_w:.1f} W "
                    f"(night, err={err_w:.1f} W, "
                    f"thr={threshold_w:.1f} W)"
                )

            self.get_logger().info(info_msg)

            # InfluxDB 기록
            self.write_to_influx(
                raw_json=d,
                ctx=ctx,
                status=status,
                p_actual_w=p_eval,
                p_pred_w=y_hat_w,
                err_w=err_w,
                rel_err_percent=rel_err_percent,
                threshold_w=threshold_w,
            )
            return

        # ===== 여기까지 왔으면 '낮'이고, p_w가 None이면 NO_ACTUAL =====
        if p_w is None:
            status = "NO_ACTUAL"
            self.get_logger().info(
                f"[NO_ACTUAL] P_actual=NA W → P_pred={y_hat_w:.1f} W"
            )
            # Influx 기록 (실제값 없음)
            self.write_to_influx(
                raw_json=d,
                ctx=ctx,
                status=status,
                p_actual_w=None,
                p_pred_w=y_hat_w,
                err_w=None,
                rel_err_percent=None,
                threshold_w=None,
            )
            return

        # 이 아래부터는 낮 + 실제값 존재
        # ===== 낮인데 irradiation 충분 + 출력 거의 0W → irradiance 기반 UNDER fault =====
        is_sunny = False
        if altitude is not None and altitude > self.day_altitude_deg:
            if (ghi_est is not None) and (ghi_est >= self.min_ghi_for_fault):
                is_sunny = True

        if is_sunny and (p_w < self.min_power_for_eval):
            # 일사량은 충분한데 발전이 거의 없음 → irradiance 기반 UNDER fault
            err_w = abs(y_hat_w - p_w)

            status = "FAULT_UNDER_IRR"

            self.get_logger().info(
                f"[FAULT_UNDER_IRR] P_actual={p_w:.1f} W → P_pred={y_hat_w:.1f} W "
                f"(err={err_w:.1f} W, alt={altitude:.1f} deg, "
                f"GHI_est={ghi_est:.1f} W/m²)"
            )

            # threshold는 일반 기준으로 계산해두고 저장
            p_kw = p_w / 1000.0
            rel_base_kw = self.rel_tol * max(p_kw, 1e-6)
            threshold_kw = max(self.abs_tol_kw, rel_base_kw)
            threshold_w = threshold_kw * 1000.0

            self.write_to_influx(
                raw_json=d,
                ctx=ctx,
                status=status,
                p_actual_w=p_w,
                p_pred_w=y_hat_w,
                err_w=err_w,
                rel_err_percent=None,  # 고장 레이블은 irradiance 기반이라 상대오차는 생략
                threshold_w=threshold_w,
            )
            return

        # ===== 일반 낮 구간: 모델 오차 기반 고장 판정 =====
        err_w = abs(y_hat_w - p_w)

        # kW 기준 threshold 계산
        p_kw = p_w / 1000.0
        rel_base_kw = self.rel_tol * max(p_kw, 1e-6)
        threshold_kw = max(self.abs_tol_kw, rel_base_kw)
        threshold_w = threshold_kw * 1000.0

        # 상대 오차(%)
        if p_w >= 1e-3:
            rel_err_percent = (err_w / p_w) * 100.0
        else:
            rel_err_percent = None

        if err_w <= threshold_w:
            status = "NORMAL"
        else:
            status = "FAULT_OVER" if (y_hat_w > p_w) else "FAULT_UNDER"

        # ===== 로그 출력 =====
        if rel_err_percent is not None:
            info_msg = (
                f"[{status}] P_actual={p_w:.1f} W → "
                f"P_pred={y_hat_w:.1f} W "
                f"(err={err_w:.1f} W, rel_err={rel_err_percent:.1f}%, "
                f"thr={threshold_w:.1f} W)"
            )
        else:
            info_msg = (
                f"[{status}] P_actual={p_w:.1f} W → "
                f"P_pred={y_hat_w:.1f} W "
                f"(err={err_w:.1f} W, thr={threshold_w:.1f} W)"
            )

        self.get_logger().info(info_msg)

        # ===== InfluxDB 기록 =====
        self.write_to_influx(
            raw_json=d,
            ctx=ctx,
            status=status,
            p_actual_w=p_w,
            p_pred_w=y_hat_w,
            err_w=err_w,
            rel_err_percent=rel_err_percent,
            threshold_w=threshold_w,
        )


def main(args=None):
    rclpy.init(args=args)
    node = PVForecastNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
