# 02_fetch_and_preprocess.py
import pandas as pd
from influxdb_client import InfluxDBClient

# ===== InfluxDB 설정 =====
URL = "http://localhost:8086"
ORG = "my-org"
BUCKET = "ai4i"
TOKEN = "change-me"

# ===== InfluxDB 클라이언트 =====
client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
query_api = client.query_api()

# ===== Flux 쿼리: 여기서 pivot까지 한 번에! =====
query = f'''
from(bucket: "{BUCKET}")
  |> range(start: 2020-01-01T00:00:00Z)
  |> filter(fn: (r) => r["_measurement"] == "machine_status")
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
'''

# 이 함수는 결과가 DataFrame 하나일 수도 있고, 여러 개(DataFrame 리스트)일 수도 있음
tables = query_api.query_data_frame(org=ORG, query=query)

# 리스트면 concat, 아니면 그대로 사용
if isinstance(tables, list):
    df = pd.concat(tables, ignore_index=True)
else:
    df = tables

print("===== 원본 df.head() =====")
print(df.head())
print(df.columns)

# ===== timestamp 컬럼 이름 정리 =====
df.rename(columns={"_time": "timestamp"}, inplace=True)

# 우리가 넣었던 field 이름들:
# "air_temp", "process_temp", "rotational_speed", "torque", "tool_wear",
# "machine_failure", "twf", "hdf", "pwf"
# 필요 컬럼만 골라서 사용
df = df[[
    "timestamp",
    "air_temp",
    "process_temp",
    "rotational_speed",
    "torque",
    "tool_wear",
    "machine_failure",
    "twf",
    "hdf",
    "pwf"
]]

print("===== 필요한 컬럼만 남긴 df.head() =====")
print(df.head())

# ===== 이상치 제거 (예시) =====
df = df[(df["air_temp"] >= 250) & (df["air_temp"] <= 350)]
df = df[(df["process_temp"] >= 250) & (df["process_temp"] <= 400)]
df = df[df["rotational_speed"] > 0]
df = df[df["torque"] > 0]
df = df[df["tool_wear"] >= 0]

# ===== 결측값 처리 =====
df = df.dropna()

# ===== 고장 유형 라벨 만들기 =====
# 0: no failure, 1: TWF, 2: HDF, 3: PWF
df["failure_type"] = (
    1 * df["twf"] +
    2 * df["hdf"] +
    3 * df["pwf"]
)

print("===== 라벨 분포 =====")
print("Machine failure:")
print(df["machine_failure"].value_counts())
print("\nFailure type:")
print(df["failure_type"].value_counts())

# ===== 전처리 결과 저장 =====
out_path = "../data/ai4i_preprocessed.csv"
df.to_csv(out_path, index=False)
print(f"\n전처리된 데이터 저장 완료: {out_path}")

client.close()
