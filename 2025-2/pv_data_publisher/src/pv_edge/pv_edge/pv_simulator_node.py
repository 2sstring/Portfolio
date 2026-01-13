#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, json, random
from pathlib import Path
from typing import Dict, List

import pandas as pd
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory

# ---- 열 이름 자동 매핑 (대소문자/한글 대응) ----
COLMAP: Dict[str, str] = {
    # 시간/ID
    'ts': 'ts', 'datetime': 'ts', '일시': 'ts', 'timestamp': 'ts', 'hour': 'hour',
    'inverter_id': 'inverter_id', 'inv': 'inverter_id', '인버터': 'inverter_id',
    'plant_id': 'plant_id', '플랜트': 'plant_id',

    # 일사/기상
    'dhi': 'dhi', '수평일사량': 'dhi',
    'dni': 'dni', '경사일사량': 'dni',
    'ambient_temp': 'ambient_temp', 'temperature': 'ambient_temp', '대기온도': 'ambient_temp',
    'module_temp': 'module_temp', 'module_temperature': 'module_temp', '모듈온도': 'module_temp',
    'ws': 'ws', '풍속(m/s)': 'ws',
    'rh': 'rh', '습도(%)': 'rh',

    # 전기/출력 (전압/전류 항목은 제거)
    'ac_power': 'ac_power', 'target': 'ac_power', '현재발전량': 'ac_power',

    # 상태
    'status': 'status'
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    m = {}
    for c in df.columns:
        key = str(c).strip().lstrip('\ufeff').lower()
        m[c] = COLMAP.get(key, key)
    return df.rename(columns=m)


def _num(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def fmt(x, n=2):
    return f"{x:.{n}f}" if x is not None else "NA"
    

class PvSimulatorFromExcel(Node):
    def __init__(self):
        super().__init__('pv_simulator_node')
        self.pub = self.create_publisher(String, '/pv/sample', 10)

        # ---- 기본값(코드 내 고정값) ----
        DEFAULT_FILE = r"/root/pv_ws_pub/src/pv_edge/data/SolarData_Sample_60min.csv"  # 필요 시 변경
        DEFAULT_RATE = 1.0
        DEFAULT_LOOP = True
        DEFAULT_RELOAD = 5.0
        DEFAULT_PLANT = "PLANT_A"

        # ---- 파라미터 선언 (실행 시 덮어쓰기 가능) ----
        self.declare_parameter('file_path', DEFAULT_FILE)
        self.declare_parameter('sheet', None)                 # 엑셀 시트명
        self.declare_parameter('rate_hz', float(DEFAULT_RATE))
        self.declare_parameter('loop', bool(DEFAULT_LOOP))
        self.declare_parameter('reload_sec', float(DEFAULT_RELOAD))
        self.declare_parameter('plant_id', DEFAULT_PLANT)

        # 이상 주입 관련
        self.declare_parameter('anomaly_mode', 'none')        # none|random|window
        self.declare_parameter('fault_prob', 0.0)             # random 0.2 => 20%
        self.declare_parameter('scale_low_power', 0.3)
        self.declare_parameter('scale_trip', 0.0)

        # 상태 필드 관련
        self.declare_parameter('include_status', True)
        self.declare_parameter('status_field', 'STATUS')
        
        gp = self.get_parameter
        self.anomaly_mode = gp('anomaly_mode').get_parameter_value().string_value
        self.fault_prob = float(gp('fault_prob').get_parameter_value().double_value)
        
        self.scale_low_power = float(gp('scale_low_power').get_parameter_value().double_value)
        self.scale_trip = float(gp('scale_trip').get_parameter_value().double_value)
        
        self.include_status = gp('include_status').get_parameter_value().bool_value
        self.status_field = gp('status_field').get_parameter_value().string_value or 'STATUS'

        # ---- 파라미터 읽기 ----
        self.file_path = Path(self.get_parameter('file_path').get_parameter_value().string_value)
        self.sheet = self.get_parameter('sheet').get_parameter_value().string_value or None
        self.rate_hz = float(self.get_parameter('rate_hz').get_parameter_value().double_value)
        self.loop = bool(self.get_parameter('loop').get_parameter_value().bool_value)
        self.reload_sec = float(self.get_parameter('reload_sec').get_parameter_value().double_value)
        self.plant_id = self.get_parameter('plant_id').get_parameter_value().string_value

        # ---- 상태 ----
        self._mtime = None
        self.df = None
        self.idx = 0

        # ---- 초기 로드 & 타이머 ----
        self._load_file()
        self.timer_pub = self.create_timer(1.0 / max(1e-6, self.rate_hz), self._tick)
        self.timer_reload = self.create_timer(self.reload_sec, self._hot_reload)

        self.get_logger().info(
            f"PV CSV/Excel simulator started\n"
            f"  file_path={self.file_path}\n"
            f"  rate_hz={self.rate_hz}, loop={self.loop}, reload_sec={self.reload_sec}\n"
            f"  anomaly_mode={self.anomaly_mode}, fault_prob={self.fault_prob}\n"
            f"  NOTE: ac_power is read in kW from file and converted to W before publish."
        )

    # ---------- 파일 로드 ----------
    def _load_file(self):
        file_path_param = self.get_parameter('file_path').get_parameter_value().string_value
        p = Path(file_path_param)
        if not p.exists():
            raise FileNotFoundError(f"데이터 파일이 없습니다: {p}")
        self._mtime = p.stat().st_mtime

        # 확장자에 따라 읽기
        try:
            if p.suffix.lower() in ('.xls', '.xlsx', '.xlsm'):
                df = pd.read_excel(p, sheet_name=self.sheet)
            else:
                try:
                    df = pd.read_csv(p, encoding='utf-8-sig')
                except Exception:
                    df = pd.read_csv(p, encoding='utf-8')
        except Exception:
            # CP949 대응
            df = pd.read_csv(p, encoding='cp949')

        df = _normalize_columns(df)

        if 'ts' not in df.columns:
            raise ValueError("시간 열(ts/Datetime/일시/…)이 필요합니다. (매핑 결과 'ts')")

        df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
        df = df.dropna(subset=['ts']).sort_values('ts').reset_index(drop=True)

        # 숫자화 (ac_power는 여기서는 kW로 유지)
        for c in ['hour','dhi','dni','ambient_temp','module_temp','ws','rh','ac_power']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # 기본 보정
        if 'inverter_id' not in df.columns:
            df['inverter_id'] = 'INV1'
        if 'plant_id' not in df.columns:
            df['plant_id'] = self.plant_id
        if 'status' not in df.columns:
            df['status'] = 'OK'

        self.df = df
        self.idx = 0
        self.get_logger().info(f"[load] rows={len(df):,} cols={list(df.columns)}")

    # ---------- 핫리로드 ----------
    def _hot_reload(self):
        try:
            m = self.file_path.stat().st_mtime
            if self._mtime is None or m > self._mtime:
                self.get_logger().info("[reload] 파일 변경 감지 → 재로딩")
                self._load_file()
        except Exception as e:
            self.get_logger().warn(f"[reload] 확인 실패: {e}")

    # ---------- 퍼블리시 ----------
    def _tick(self):
        if self.df is None or self.df.empty:
            return
        if self.idx >= len(self.df):
            if self.loop:
                self.idx = 0
            else:
                return

        row = self.df.iloc[self.idx]
        self.idx += 1

        hour_raw = row.get('hour')
        try:
            if pd.isna(hour_raw):
                hour_val = None
            else:
                hour_val = int(hour_raw)
        except Exception:
            hour_val = None

        # sample에서 ac_power는 우선 kW 단위로 사용
        sample = {
            'ts': row['ts'].strftime('%Y-%m-%dT%H:%M:%S'),
            'plant_id': str(row.get('plant_id', self.plant_id)),
            'inverter_id': str(row.get('inverter_id', 'INV1')),

            # 시간
            'hour': hour_val,

            # 센서/기상
            'dhi': _num(row.get('dhi')),
            'dni': _num(row.get('dni')),
            'ambient_temp': _num(row.get('ambient_temp')),
            'module_temp': _num(row.get('module_temp')),
            'ws': _num(row.get('ws')),
            'rh': _num(row.get('rh')),

            # 전기 (kW)
            'ac_power': _num(row.get('ac_power')),

            # 상태
            'status': str(row.get('status', 'OK')),
        }

        # 기본 상태: 원래 데이터의 status를 존중 (없으면 OK)
        st = str(row.get('status', 'OK'))

        # 랜덤 이상 주입 (kW 기준)
        if self.anomaly_mode == 'random' and self.fault_prob > 0.0:
            p = sample.get('ac_power')
            dni = sample.get('dni')

            try:
                p_val = float(p) if p is not None else None
            except Exception:
                p_val = None

            try:
                dni_val = float(dni) if dni is not None else None
            except Exception:
                dni_val = None

            # === 고장 주입도 "의미 있는 구간"에서만 ===
            #  - 발전량이 충분히 크고 (0.2kW 이상)
            #  - DNI도 어느 정도 이상일 때 (200 W/m² 이상)
            if (
                (p_val is not None and p_val > 0.20) and
                (dni_val is not None and dni_val > 200.0) and
                (random.random() < self.fault_prob)
            ):
                st = random.choice(['TRIP', 'LOW_AC_POWER'])
                try:
                    if st == 'TRIP':
                        sample['ac_power'] = float(p_val) * self.scale_trip
                    else:
                        sample['ac_power'] = float(p_val) * self.scale_low_power
                except Exception:
                    pass

        # 최종 status 반영
        sample['status'] = st
        if self.include_status:
            sample[self.status_field] = st

        # === kW → W 단위 변환 (퍼블리시 직전) ===
        if sample.get('ac_power') is not None:
            sample['ac_power'] = sample['ac_power'] * 1000.0

        msg = String()
        msg.data = json.dumps(sample, ensure_ascii=False)
        self.pub.publish(msg)

        hour_str = str(sample['hour']) if sample['hour'] is not None else "NA"
        ts = row['ts']
        ts_str = ts.strftime('%Y-%m-%d %H:%M:%S')

        self.get_logger().info(
            f"{ts_str} "
            f"{sample['inverter_id']} "
            f"Hour={hour_str} "
            f"DHI={fmt(sample['dhi'])} DNI={fmt(sample['dni'])} "
            f"T={fmt(sample['ambient_temp'])} MT={fmt(sample['module_temp'])} "
            f"WS={fmt(sample['ws'])} RH={fmt(sample['rh'])} "
            f"P(W)={fmt(sample['ac_power'])} status={sample['status']}"
        )


def main():
    rclpy.init()
    node = PvSimulatorFromExcel()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
