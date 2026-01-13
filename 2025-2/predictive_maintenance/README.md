### AI4I Predictive Maintenance – Fault Prediction System

### 환경
- Ubuntu (Docker)
- InfluxDB 2.7.12
- Grafana 12.2.1
- Python 3.12.3

### 사전 준비
본 프로젝트의 소스코드에는 토큰 및 비밀번호가 실제 값이 아닌 change-me로 설정되어 있다.
따라서 실제 실행을 위해서는 각 Python 파일 및 docker-compose.yml에서 change-me를 본인 환경의 InfluxDB 토큰/비밀번호로 변경해야 한다.

### 🔗 데이터 링크
https://www.unb.ca/cic/datasets/vpn.html

### 📥 다운로드 및 준비 방법
1. 이 프로젝트 폴더 내에 data 폴더를 만듭니다.
2. 위 링크에서 파일을 다운로드합니다.
3. 다운받은 파일을 압축해제하여 1에서 만든 data 폴더 안에 위치시켜 주세요.

### 실행 순서
1) docker compose up -d
2) python3 -m venv venv
3) source venv/bin/activate
4) pip install -r requirements.txt
5) cd src
6) python 01_csv_to_influx.py
7) python 02_fetch_and_preprocess.py
8) python 03_train_models.py
9) python 04_predict_to_influx.py
10) InfluxDB 접속: http://localhost:8086
11) Grafana 접속: http://localhost:3000

