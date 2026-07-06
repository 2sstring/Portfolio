# UCI SECOM 데이터셋 품질 확보

## 프로젝트 개요

본 프로젝트는 UCI SECOM 반도체 제조 공정 데이터셋을 대상으로 데이터 품질 확보 과정을 수행한 프로젝트이다.

반도체 제조 데이터는 센서 feature 수가 많고 결측값과 이상치가 포함될 가능성이 높기 때문에, AI 모델 학습 전에 데이터의 유일성, 완전성, 유효성, 일관성, 정확성을 점검하는 과정이 필요하다.

본 프로젝트에서는 데이터 정제 과정을 통해 이후 품질 예측 또는 불량 분류 모델에 활용 가능한 형태로 데이터를 정리하였다.

## 데이터셋 정보

본 저장소에는 원본 데이터셋을 포함하지 않는다.

- 데이터셋명: UCI SECOM Dataset
- 다운로드 링크: https://www.kaggle.com/datasets/paresh2047/uci-semcom?resource=download
- 사용 파일명: `uci-secom.csv`

데이터셋을 다운로드한 후, 아래와 같이 프로젝트 폴더에 배치한다.

```text
01-uci-secom-data-quality-assurance/
├── README.md
├── hw01_uci_secom.py
└── uci-secom.csv        # GitHub 업로드 제외
