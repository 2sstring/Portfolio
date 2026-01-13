# 01_csv_to_influx.py
import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# ===== InfluxDB 설정 (docker-compose와 동일하게) =====
URL = "http://localhost:8086"
ORG = "my-org"
BUCKET = "ai4i"
TOKEN = "change-me"  # docker-compose에 넣었던 토큰 그대로

# ===== CSV 불러오기 =====
csv_path = "../data/ai4i2020.csv"  # 리눅스에서도 상대경로는 동일
df = pd.read_csv(csv_path)

print(df.head())
print(df.columns)

# ===== 가짜 타임스탬프 생성 =====
start_time = pd.to_datetime("2020-01-01 00:00:00")
df = df.sort_index()
df["timestamp"] = start_time + pd.to_timedelta(df.index, unit="min")

# ===== InfluxDB 클라이언트 생성 =====
client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)

measurement_name = "machine_status"

count = 0
for _, row in df.iterrows():
    p = (
        Point(measurement_name)
        .tag("type", row["Type"])
        .field("air_temp", float(row["Air temperature [K]"]))
        .field("process_temp", float(row["Process temperature [K]"]))
        .field("rotational_speed", float(row["Rotational speed [rpm]"]))
        .field("torque", float(row["Torque [Nm]"]))
        .field("tool_wear", float(row["Tool wear [min]"]))
        .field("machine_failure", int(row["Machine failure"]))
        .field("twf", int(row["TWF"]))
        .field("hdf", int(row["HDF"]))
        .field("pwf", int(row["PWF"]))
        .time(row["timestamp"], WritePrecision.NS)
    )
    write_api.write(bucket=BUCKET, org=ORG, record=p)
    count += 1

print(f"총 {count}개 레코드를 InfluxDB에 업로드 완료!")
client.close()
