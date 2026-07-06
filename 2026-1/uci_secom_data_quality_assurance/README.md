# UCI SECOM 데이터셋 품질 확보

## 프로젝트 개요

본 프로젝트는 UCI SECOM 반도체 제조 공정 데이터셋을 대상으로 데이터 품질 확보 과정을 수행하는 프로젝트입니다.

반도체 제조 데이터는 센서 feature 수가 많고 결측값과 이상치가 포함될 가능성이 높습니다. 따라서 AI 모델 학습 전에 데이터의 유일성, 완전성, 유효성, 일관성, 정확성을 점검하는 과정이 필요합니다.

본 프로젝트에서는 데이터 정제 과정을 통해 이후 품질 예측 또는 불량 분류 모델에 활용할 수 있는 데이터셋을 구성합니다.

## 데이터셋 정보

본 저장소에는 원본 데이터셋을 포함하지 않습니다.

* 데이터셋명: UCI SECOM Dataset
* 다운로드 링크: https://www.kaggle.com/datasets/paresh2047/uci-semcom?resource=download
* 사용 파일명: `uci-secom.csv`

데이터셋을 다운로드한 후 아래와 같이 프로젝트 폴더에 배치합니다.

```text
01-uci-secom-data-quality-assurance/
├── README.md
├── uci_secom.py
└── uci-secom.csv        # GitHub 업로드 제외
```

## 주요 처리 내용

### 1. 유일성 확보

* 완전 중복 row를 제거합니다.
* 중복 제거 전후 데이터 크기를 비교합니다.

### 2. 완전성 확보

* 결측률이 10% 이상인 feature를 제거합니다.
* 남아 있는 결측 row를 제거합니다.
* 정제 후 클래스 분포를 확인합니다.

### 3. 유효성 확보

* IQR 기반으로 이상치를 판정합니다.
* row별 이상치 비율을 계산합니다.
* 이상치 비율이 0.04 이상인 row를 제거합니다.

### 4. 일관성 확보

* 비수치형 feature를 확인하고 제거합니다.
* 분산이 0인 상수 feature를 제거합니다.

### 5. 정확성 검증

* 전처리 전후 IQR 기준 이상치 개수를 비교합니다.
* 전처리 전후 표준편차 감소 feature 수를 확인합니다.

## 실행 방법

필요 라이브러리를 설치한 후 Python 파일을 실행합니다.

```bash
pip install pandas numpy
python hw01_uci_secom.py
```

## 결과 파일

실행 후 다음 파일이 생성됩니다.

```text
uci-secom_cleaned_until_consistency.csv
```

## 사용 기술

* Python
* pandas
* NumPy

## 정리

본 프로젝트는 반도체 제조 데이터의 품질 확보를 목적으로 합니다. 결측값, 이상치, 비수치 feature, 상수 feature 등을 단계적으로 제거하여 AI 모델 학습에 적합한 데이터셋을 구성합니다.
