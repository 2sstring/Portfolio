# 📡 traffic_classification

## 📌 프로젝트 소개

**traffic_classification**은 **ISCXVPN2016 VPN/Non-VPN 네트워크 트래픽 데이터셋**을 이용해 네트워크 서비스 유형을 분류하는 머신러닝·딥러닝 실습 프로젝트입니다.

패킷 캡처 파일(`.pcap`, `.pcapng`)에서 통신 흐름(Flow)별 특징을 추출하고, **Random Forest** 및 **ResMLP·FT-Transformer 명칭의 모델·PlainMLP로 구성된 딥러닝 앙상블**을 학습하여 트래픽 유형별 분류 성능을 분석합니다.

> **분류 목표:** VPN 사용 여부의 이진 분류가 아니라, VPN과 Non-VPN 폴더의 트래픽을 함께 이용해 **Chat, Email, Audio, Video, File Transfer, Other** 등의 서비스 유형을 구분합니다. 실제 라벨은 PCAP 파일명에 포함된 키워드를 기준으로 생성합니다.

---

## 📁 프로젝트 폴더 구조

```text
traffic_classification/
├── iscx_vpn_multiclass.ipynb     # 데이터 처리, 모델 학습 및 성능 평가
├── README.md                     # 프로젝트 설명
├── dataset/                      # 원본 데이터 다운로드 후 생성
│   ├── NonVPN-PCAPs-01/          # Non-VPN PCAP 파일
│   ├── VPN-PCAPS-01/             # VPN PCAP 파일
│   └── output_csvs/              # Flow 특징 추출 결과 (자동 생성)
└── output_multiple.csv           # 개별 특징 CSV를 병합한 학습용 파일 (실행 후 생성)
```

`dataset/`, `dataset/output_csvs/`, `output_multiple.csv`는 데이터 준비 및 노트북 실행 과정에서 구성되며, 최초 저장소에 포함되어 있지 않을 수 있습니다.

## 🗂️ 주요 파일

| 파일 | 설명 |
|---|---|
| `iscx_vpn_multiclass.ipynb` | PCAP 특징 추출부터 머신러닝·딥러닝 학습, 성능 평가, 시각화까지 수행하는 Jupyter Notebook |
| `README.md` | 데이터 준비 및 프로젝트 실행 방법 안내 |
| `dataset/NonVPN-PCAPs-01/` | Non-VPN 트래픽 원본 파일 배치 폴더 |
| `dataset/VPN-PCAPS-01/` | VPN 트래픽 원본 파일 배치 폴더 |
| `dataset/output_csvs/` | PCAP별 특징 추출 CSV 출력 폴더 |
| `output_multiple.csv` | 모든 특징 CSV를 결합하고 무작위로 섞어 생성한 데이터 |

---

## 🔗 데이터셋 및 다운로드 안내

본 프로젝트는 University of New Brunswick(UNB) 산하 Canadian Institute for Cybersecurity에서 공개한 **ISCXVPN2016 (VPN-nonVPN) 데이터셋**을 사용합니다.

- **공식 데이터 페이지:** [UNB — VPN-nonVPN dataset (ISCXVPN2016)](https://www.unb.ca/cic/datasets/vpn.html)
- **데이터 형태:** 네트워크 패킷 캡처 파일(`.pcap` 등)
- **활용 범위:** VPN/Non-VPN 환경에서 수집된 통신 데이터를 서비스 유형 분류에 사용

### 📥 다운로드 및 준비 방법

1. 위 공식 데이터 페이지로 이동해 데이터셋을 다운로드합니다.
2. `traffic_classification` 프로젝트 최상위 폴더에 `dataset` 폴더를 만듭니다.
3. 다운로드한 파일의 압축을 해제한 후, 해당 PCAP 파일들을 다음 폴더에 배치합니다.

   - `dataset/NonVPN-PCAPs-01/`
   - `dataset/VPN-PCAPS-01/`

4. 원본 압축 해제 폴더명이 다르면 위 이름으로 맞추거나, 노트북의 `NONVPN_DIR`, `VPN_DIR` 값을 실제 경로에 맞게 변경합니다.

> ⚠️ 원본 데이터 파일은 이 저장소에 포함되어 있지 않습니다. 데이터를 별도로 내려받아야 PCAP 변환 및 학습을 실행할 수 있습니다. 다운로드한 파일에 따라 확장자와 하위 폴더 구성이 다를 수 있으므로 실제 경로를 확인하세요.

---

## ⚙️ 주요 처리 과정

| 단계 | 처리 내용 | 사용 기술 |
|---|---|---|
| 1. PCAP 읽기 | 패킷 상대 시간, 길이, 출발지·목적지 IP 추출 | Wireshark `tshark` |
| 2. 트래픽 라벨링 | 파일명의 키워드로 서비스 유형 지정 | Python |
| 3. 특징 생성 | Flow별 패킷 크기·도착 간격(IAT) 및 파생 특징 계산 | pandas, NumPy |
| 4. 데이터 통합 | 개별 CSV 병합 및 무작위 셔플 | pandas, scikit-learn |
| 5. 머신러닝 학습 | Random Forest 다중 클래스 분류 | scikit-learn |
| 6. 딥러닝 학습 | ResMLP, `FTTransformerStable`, PlainMLP | PyTorch |
| 7. 앙상블 평가 | 모델 확률 출력에 가중치를 적용한 Soft Voting | PyTorch, scikit-learn |
| 8. 결과 시각화 | 혼동행렬, 특징 중요도, 결정트리 | Matplotlib, Seaborn |

### 🔍 주요 입력 특징

노트북은 패킷별 정보를 Flow 단위로 집계하여 다음 유형의 특징을 생성합니다.

| 특징 그룹 | 예시 |
|---|---|
| 패킷 규모 | 패킷 수, 전체 바이트 수, 평균·최대·최소 패킷 길이 |
| 시간 간격(IAT) | 평균·표준편차·분위수·왜도·첨도 |
| 분포 특성 | 패킷 길이 및 IAT 구간별 빈도·비율, 엔트로피 |
| 통신 속도 | Flow 지속 시간, 초당 패킷 수, 초당 바이트 수 |
| 파생 특징 | 변동계수, 범위, 사분위 범위(IQR), 버스트 지표, 로그·제곱근 변환 |

**라벨 생성 방식**은 파일명에서 `chat`, `email`, `audio`, `video`, `ftp`/`file`/`bittorrent`를 순서대로 확인하며, 어느 항목에도 해당하지 않으면 `other`로 분류합니다.

---

## 🧠 분류 모델 및 실험 설정

### 1. Random Forest

- 모델: `RandomForestClassifier`
- 결정트리 수: **100개**
- 데이터 분할: **학습 80% / 테스트 20%** (계층적 무작위 분할)
- 난수 시드: **42**
- 짧은 Flow 제거: `packet_count < 3`인 데이터 제외
- 평가: 클래스별 Precision·Recall·F1-score, 혼동행렬

### 2. 딥러닝 모델과 앙상블

| 모델 | 코드상 구성 | 앙상블 가중치 |
|---|---|---:|
| ResMLP | 잔차 연결을 갖는 다층 퍼셉트론 | 0.45 |
| `FTTransformerStable` | 특징 벡터 임베딩과 잔차형 피드포워드 블록 | 0.45 |
| PlainMLP | 일반적인 다층 퍼셉트론 | 0.10 |

딥러닝 공통 설정은 **250 Epochs**, 학습 배치 크기 **256**, 테스트 배치 크기 **512**, 초기 학습률 **0.0005**, AdamW 옵티마이저입니다. CUDA 사용이 가능하면 GPU로 실행하고 그렇지 않으면 CPU로 실행합니다.

각 모델의 Softmax 확률을 **0.45 : 0.45 : 0.10**으로 가중 평균해 최종 클래스를 결정합니다.

> 참고: `FTTransformerStable`은 코드에 사용된 클래스명입니다. 현재 구현에는 Self-Attention 계층이 없으므로, 일반적인 FT-Transformer와 동일한 구조라고 해석하면 안 됩니다.

---

## 📊 성능 평가 및 시각화

- **Random Forest:** Classification Report 및 Confusion Matrix를 출력합니다.
- **딥러닝 및 앙상블:** 모델별 Accuracy와 Macro-F1, 앙상블의 Classification Report 및 Confusion Matrix를 출력합니다.
- **특징 분석:** Random Forest의 Feature Importance 막대그래프를 표시합니다.
- **모델 해석:** 학습된 Random Forest의 결정트리 일부를 제한된 깊이로 시각화합니다.

그래프는 노트북에서 표시되며, 현재 코드에는 그래프 이미지 또는 학습된 모델을 자동으로 별도 저장하는 기능이 없습니다. **성능 수치는 실제 데이터로 코드를 실행한 결과에 따라 달라지므로 README에 임의로 기입하지 않았습니다.**

---

## 💻 실행 환경 및 방법

### 1. 실행 환경

- Python과 Jupyter Notebook
- Wireshark의 **TShark** (PCAP 읽기용)
- pandas, NumPy, tqdm, scikit-learn, Matplotlib, Seaborn
- PyTorch (딥러닝 학습용)

Python 라이브러리 설치 예시:

```bash
pip install jupyter pandas numpy tqdm scikit-learn matplotlib seaborn torch
```

**TShark는 위 `pip` 명령으로 설치되지 않습니다.** Wireshark 설치 과정에서 TShark 구성요소를 설치하거나, 운영체제의 별도 패키지 관리 도구로 설치해야 합니다.

### 2. TShark 경로 설정

현재 노트북의 `extract_and_save_csv` 함수에는 Windows 기준 경로가 지정되어 있습니다.

```python
tshark_path = r"C:\Program Files\Wireshark\tshark.exe"
```

본인 PC에 설치된 실행 파일 경로에 맞게 수정하세요. TShark가 시스템 PATH에 등록되어 있다면 `tshark_path = "tshark"`로 설정할 수도 있습니다.

### 3. 노트북 실행

프로젝트 최상위 폴더에서 다음 명령을 실행합니다.

```bash
jupyter notebook
```

브라우저에서 `iscx_vpn_multiclass.ipynb`를 열고 셀을 위에서부터 순서대로 실행합니다.

1. 라이브러리 불러오기 및 데이터 경로 지정
2. 파일명 기반 트래픽 라벨 생성
3. TShark를 이용한 PCAP별 특징 CSV 추출
4. CSV 병합 및 `output_multiple.csv` 생성
5. Random Forest 학습·평가
6. 딥러닝 모델 3종 및 Soft Voting 앙상블 평가
7. 특징 중요도 및 결정트리 시각화

---

## ⚠️ 실행 및 해석 시 주의사항

1. **데이터 경로:** `dataset/NonVPN-PCAPs-01`, `dataset/VPN-PCAPS-01` 폴더명이 실제 데이터 구성과 같아야 합니다. 노트북은 각 폴더의 최상위 `.pcap`·`.pcapng` 파일을 검색합니다.
2. **TShark 설치:** 프로그램이 설치되지 않았거나 경로가 다르면 PCAP에서 특징을 추출할 수 없습니다.
3. **라벨 품질:** 서비스 유형은 원본 PCAP의 실제 정답 열이 아니라 **파일명 키워드 규칙**으로 생성됩니다. 따라서 파일명에 따라 `other` 분류가 증가하거나 의도와 다른 라벨이 붙을 수 있습니다.
4. **데이터 누수 가능성:** Random Forest 부분은 현재 전체 데이터에 `StandardScaler`를 먼저 적용한 뒤 학습·테스트를 분할합니다. 엄밀한 평가를 위해서는 **분할 후 학습 데이터에만 `fit`** 해야 합니다. 딥러닝 부분은 분할 후 스케일러를 학습합니다.
5. **성능 해석:** 무작위 Flow 분할에서는 동일 원본 PCAP의 Flow가 학습·테스트에 동시에 포함될 수 있습니다. 새로운 캡처 파일에 대한 일반화 성능을 검증하려면 **PCAP 파일 단위 분리** 등을 추가로 고려해야 합니다.
6. **실행 자원:** 딥러닝 모델 3종을 각각 250 Epochs 학습하므로 GPU·메모리 상황에 따라 실행 부담이 클 수 있습니다.

---

## 🎯 프로젝트 목적

이 프로젝트는 네트워크 패킷을 **통계적 Flow 특징**으로 변환하고, 머신러닝 모델과 딥러닝 앙상블의 분류 결과를 비교하는 과정을 학습하기 위한 프로젝트입니다. PCAP 전처리, 특징 공학, 다중 클래스 분류, 성능 평가 및 시각화를 하나의 노트북에서 확인할 수 있습니다.

### 📚 데이터 출처

University of New Brunswick, Canadian Institute for Cybersecurity, **ISCXVPN2016 (VPN-nonVPN dataset)**  
https://www.unb.ca/cic/datasets/vpn.html

공식 데이터 제공 페이지의 사용·인용 안내를 확인한 후 활용해 주세요.
