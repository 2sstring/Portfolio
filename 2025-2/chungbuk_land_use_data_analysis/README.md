# 🗺️ chungbuk_land_use_data_analysis

**충청북도 토지이용·인구·도로 데이터를 활용한 지역별 통계 분석 및 시각화 프로젝트**

2015~2025년 충청북도 시·군·구의 지적통계와 주민등록인구통계를 바탕으로 **토지이용 면적과 비율, 인구·인구밀도, 도로율**을 분석합니다. 연도별 변화와 지역 간 차이를 비교하고, **기초 통계, 상관관계 히트맵, 산점도, 도넛 차트, 시계열 그래프**로 결과를 시각화합니다.

> 이 프로젝트는 **탐색적 데이터 분석(EDA)과 시각화**에 초점을 둡니다. K-Means 기반 토지이용 유형 분류는 별도 프로젝트인 `chungbuk_land_use_classification`에서 다룹니다.

---

## 📌 프로젝트 개요

주요 작업은 다음과 같습니다.

- 연도별 지적통계 CSV를 통합하여 **지역·연도별 토지이용 데이터** 구성
- 임야, 농경지(전+답), 대지, 공장용지의 **면적 및 전체 토지 대비 비율** 계산
- 지역별 **도로율**과 **인구밀도** 산출 및 연도별 변화 분석
- 평균·중앙값, 지역별 순위, 히스토그램 및 박스플롯을 활용한 **기초 통계 분석**
- Pearson 상관계수와 산점도를 활용한 **토지이용 변수 간 관계 분석**
- 도넛 차트와 시계열 선 그래프를 이용한 **지역별·연도별 시각화**

## 📂 프로젝트 폴더 구조

```text
chungbuk_land_use_data_analysis/
├── preprocess1.py                # 연도별 토지이용 데이터 통합 및 구성비 산출
├── preprocess2.py                # 지역별·연도별 도로율 산출
├── stats_overview.py             # 기초 통계·지역 순위·분포 시각화
├── corr_heatmap_feature4.py      # 4개 주요 토지용도 면적/비율 상관 히트맵
├── corr_heatmap_feature6.py      # 도로율·인구밀도를 포함한 상관 분석 등
├── scatter_feature2.py          # 대지 면적과 공장용지 면적의 관계
├── scatter_feature4.py          # 주요 4개 토지이용 비율의 쌍별 산점도
├── donut.py                     # 지역별 토지이용 구성 도넛 차트
├── timeseries_landuse.py        # 토지이용 면적의 연도별 변화
├── timeseries_population.py     # 인구밀도의 연도별 변화
├── timeseries_population2.py    # 인구수의 연도별 변화
├── timeseries_road.py           # 토지이용 비율 및 도로율의 연도별 변화
├── README.md                    # 프로젝트 설명 문서
└── data/
    ├── chungbuk_data_20151231.csv
    ├── chungbuk_data_20161231.csv
    ├── chungbuk_data_20171231.csv
    ├── chungbuk_data_20181231.csv
    ├── chungbuk_data_20191231.csv
    ├── chungbuk_data_20201231.csv
    ├── chungbuk_data_20211231.csv
    ├── chungbuk_data_20221231.csv
    ├── chungbuk_data_20231231.csv
    ├── chungbuk_data_20241231.csv
    ├── chungbuk_data_20251231.csv
    └── chungbuk_population.xlsx
```

> ⚠️ **데이터 파일 안내:** `data/`의 연도별 CSV 11개와 `chungbuk_population.xlsx`는 **컬럼 구조와 지역명만 남겨둔 예시 파일**이며, 실제 통계 수치가 제거되어 있습니다. 원본 자료에서 해당 값을 준비해야 분석할 수 있습니다. 아래에 소개하는 전처리 산출물과 분석 결과 이미지·CSV는 **현재 제공된 원본 파일 목록이 아니라 실행 후 생성될 파일**입니다.

## 🗂️ 데이터셋 및 원본 데이터 출처

| 데이터 | 내용 | 활용 목적 |
| --- | --- | --- |
| 충청북도 지적통계 | 시·군·구별 토지 지목별 면적 및 지번 현황 | 토지이용 비율, 도로율, 지역별 변화 분석 |
| 충청북도 주민등록인구통계 | 시·군·구별 연도별 주민등록인구 | 인구수 및 인구밀도 시계열 분석 |

**🔗 공개자료 출처**

- **[충청북도 지적통계](https://www.chungbuk.go.kr/www/selectBbsNttList.do?bbsNo=100&key=658)** — 기존 데이터 안내에 기재된 수집 범위: **2015.12.31 ~ 2025.09.30**
- **[충청북도 주민등록인구통계](https://www.chungbuk.go.kr/stat/selectBbsNttList.do?bbsNo=181&key=1406)** — 기존 데이터 안내에 기재된 수집 범위: **2015.12 ~ 2025.09**

### 📥 데이터 다운로드 및 준비 방법

1. 위 공개자료 페이지에서 분석에 필요한 연도별 **지적통계**와 **주민등록인구통계**를 확보합니다.
2. 지적통계를 `data/chungbuk_data_YYYY1231.csv` 형태의 예시 파일과 동일한 **컬럼명·지역명 구조**로 정리합니다.
3. 인구통계를 `data/chungbuk_population.xlsx` 형식에 맞추어 입력합니다. 첫 번째 열은 지역명(`연도`라는 컬럼명), 이후 열은 **2015~2025 연도별 인구값**입니다.
4. 각 예시 파일의 비어 있는 수치 항목을 실제 데이터로 채우고, 분석하려는 기간과 기준일을 확인합니다.
5. 전처리 코드를 실행한 뒤 나머지 통계 분석 및 시각화 스크립트를 실행합니다.

> **주의:** `chungbuk_data_20251231.csv`는 저장소에서 사용하는 파일명 예시입니다. 기존 데이터 출처 안내의 2025년 자료 기준일은 **9월**이므로, 해당 파일이 실제 **2025년 12월 31일 기준 통계**라는 의미는 아닙니다. 실제 원본 자료의 기준일을 확인해 사용해야 합니다.

## ⚙️ 데이터 전처리

| 파일 | 주요 처리 | 생성 대상 파일 |
| --- | --- | --- |
| `preprocess1.py` | 연도별 CSV 통합, 농경지(전+답) 면적 계산, 임야·농경지·대지·공장용지 비율 산출 | `chungbuk_landuse_composition_2015_2025_detail.csv`, `chungbuk_landuse_composition_2015_2025_total.csv` |
| `preprocess2.py` | 각 지역의 도로면적을 전체 면적으로 나누어 도로율 산출 | `chungbuk_road_ratio_2015_2025.xlsx` (`detail`, `total` 시트) |

주요 계산 기준은 다음과 같습니다.

| 분석 지표 | 계산 방식 |
| --- | --- |
| 농경지 면적 | 전 면적 + 답 면적 |
| 지목별 토지이용 비율 | 해당 지목 면적 ÷ 지역의 합계 면적 |
| 도로율 | 도로 면적 ÷ 지역의 합계 면적 |
| 인구밀도 | 인구수 ÷ 지역 면적(km²) |

> 네 가지 지목(임야·농경지·대지·공장용지)은 **전체 지목 중 일부**입니다. 전처리에서 각 지목의 비율은 지역의 **합계 면적**을 분모로 계산하므로, 네 비율의 합이 반드시 100%가 되는 것은 아닙니다. 다만 `donut.py`는 네 가지 지목의 면적 합계를 기준으로 별도의 구성비를 계산합니다.

## 📊 주요 분석 기능

| 분석 구분 | 파일 | 주요 내용 |
| --- | --- | --- |
| **기초 통계** | `stats_overview.py` | 면적·비율 평균과 중앙값, 연도별 통계, 2025년 지역별 순위, 히스토그램·박스플롯 |
| **4개 변수 상관 분석** | `corr_heatmap_feature4.py` | 주요 4개 토지용도 면적 간 및 비율 간 Pearson 상관관계 히트맵 |
| **확장 상관 분석** | `corr_heatmap_feature6.py` | 기초 통계 결과와 함께 4개 토지용도 비율, 도로율, 인구밀도 간 상관관계 분석 |
| **2개 변수 산점도** | `scatter_feature2.py` | 대지 면적과 공장용지 면적의 상관계수·회귀선 및 연도·지역별 산점도 |
| **4개 변수 산점도** | `scatter_feature4.py` | 2025년 주요 4개 지목 비율의 모든 두 변수 조합에 대한 산점도 |
| **도넛 차트** | `donut.py` | 2025년 충청북도 전체 및 지역별 주요 4개 지목의 구성비 |
| **토지이용 시계열** | `timeseries_landuse.py` | 지역별·충북 전체의 토지이용 면적 변화 |
| **인구밀도 시계열** | `timeseries_population.py` | 지역별·충북 전체 인구밀도 변화 |
| **인구수 시계열** | `timeseries_population2.py` | 지역별·충북 전체 인구수 변화 |
| **토지이용·도로율 시계열** | `timeseries_road.py` | 지역별·충북 전체의 토지이용 비율과 도로율 변화 |

### 🔎 상관관계 분석

`corr_heatmap_feature4.py`는 **임야·농경지·대지·공장용지**의 면적 변수와 비율 변수를 각각 분석합니다. `corr_heatmap_feature6.py`는 여기에 **도로율과 인구밀도**까지 포함한 6개 비율·밀도 지표의 관계를 살펴봅니다.

상관계수는 Pearson 방법으로 계산하며, 히트맵의 색상과 셀 안의 수치로 변수 간 관계를 표현합니다. **상관관계가 있다고 해서 인과관계가 입증되는 것은 아닙니다.**

### 📈 시계열 및 지역별 시각화

2015~2025년의 연도별 자료를 사용하여 시·군·구별 추세와 충청북도 전체의 합계 또는 집계값을 비교합니다. `donut.py`와 `scatter_feature4.py`에는 **2025년을 선택하는 `TARGET_YEAR` 설정**이 있어 원하는 분석 연도에 맞게 변경할 수 있습니다.

## 🚀 설치 및 실행 방법

### 1. 프로젝트 폴더로 이동

```bash
cd chungbuk_land_use_data_analysis
```

### 2. 필요한 라이브러리 설치

```bash
pip install numpy pandas matplotlib openpyxl
```

Python과 해당 라이브러리들이 설치되어 있어야 합니다. 코드의 한글 글꼴은 기본적으로 **Windows의 `Malgun Gothic`**을 기준으로 설정되어 있어 다른 운영체제에서는 글꼴 변경이 필요할 수 있습니다.

### 3. 실제 데이터 준비

`data/` 아래에 연도별 토지이용 CSV와 인구 Excel 파일을 **실제 값이 포함된 상태**로 준비합니다. 현재의 빈 예시 자료만으로는 분석 그래프와 통계량을 정상적으로 구할 수 없습니다.

### 4. 데이터 전처리

```bash
python preprocess1.py
python preprocess2.py
```

이 단계에서 통합 토지이용 CSV 및 도로율 Excel 파일을 만드는 것을 목표로 합니다. **실제 실행 전에는 아래 '실행 시 확인 사항'의 경로 및 계산 설정을 먼저 확인해 주세요.**

### 5. 분석 및 시각화 실행

```bash
# 기초 통계
python stats_overview.py

# 상관관계 히트맵
python corr_heatmap_feature4.py
python corr_heatmap_feature6.py

# 산점도
python scatter_feature2.py
python scatter_feature4.py

# 도넛 차트
python donut.py

# 연도별 추세 분석
python timeseries_landuse.py
python timeseries_population.py
python timeseries_population2.py
python timeseries_road.py
```

> 모든 분석을 한 번에 실행하는 단일 진입점은 없으며, **독립적인 Python 스크립트를 목적에 맞게 선택해 실행**합니다. 다수의 그래프와 분석 CSV는 코드 설정에 따라 `data/`에 저장됩니다.

## 🖼️ 주요 출력 결과 예시

| 결과 유형 | 예시 파일명 / 출력 |
| --- | --- |
| 토지이용 기초 통계 | `basic_stats_area_2015_2025.csv`, `basic_stats_ratio_2015_2025.csv` |
| 4개 토지이용 상관 히트맵 | `corr_heatmap_area_2015_2025.png`, `corr_heatmap_ratio_2015_2025.png` |
| 확장 상관관계 | `correlation_ratio_road_pop_2015_2025.csv`, `correlation_heatmap_2015_2025.png` |
| 변수 간 산점도 | `scatter_dae_vs_factory_2025.png` 등 |
| 도넛 차트 | `donut_total_2025.png`, `donut_grid_2025.png` |
| 토지이용 변화 | `area_timeseries_충청북도_전체.png` |
| 인구수·인구밀도 변화 | `population_timeseries_충청북도_전체.png`, `pop_density_timeseries_충청북도_전체.png` |
| 토지이용 비율·도로율 변화 | `ratio_road_timeseries_충청북도_전체.png` |

위 파일들은 **스크립트 실행 시 생성되는 결과의 예시**입니다. 원본 파일에 수치가 없거나 코드의 경로 설정이 맞지 않으면 생성되지 않을 수 있습니다.

## ⚠️ 실행 시 확인 사항

1. **전처리 스크립트의 상대경로:** `preprocess1.py`와 `preprocess2.py`는 `os.chdir(base_dir)`로 `data/`에 이동한 뒤 출력 경로에 다시 `data/`를 붙이므로, 현재 코드 그대로는 `data/data/...`를 참조할 수 있습니다. `os.chdir(base_dir)`를 제거하고 입력 패턴을 `glob.glob(os.path.join(base_dir, "chungbuk_data_*.csv"))`로 변경하는 등 **입력·출력 경로를 일관되게 수정**한 후 실행해야 합니다.
2. **`preprocess1.py`의 합계 비율:** 충청북도 전체 합계 행을 계산하는 `denom_total`에 개별 지역의 면적 열이 할당되어 있습니다. 전체 합계 비율을 구하려면 `denom_total = total_area`와 같이 **전체 면적 합계를 분모로 사용하도록 확인**해야 합니다.
3. **`donut.py`의 로컬 경로:** 원본 코드에는 작성자 컴퓨터의 절대경로가 지정되어 있습니다. 저장소에서 실행할 때는 데이터 위치에 맞게 `base_dir = "data"` 등으로 변경해야 합니다.
4. **데이터 파일명과 실제 기준일:** `20251231`이라는 파일명만으로 2025년 말 기준 통계라고 판단할 수 없습니다. 사용한 원본 통계의 기준일 및 기간을 확인해야 합니다.
5. **인구 데이터 형식:** 기존 README에는 `chungbuk_population.csv`라고 기재되어 있었지만, 제공된 파일과 분석 코드가 사용하는 인구 데이터는 **`chungbuk_population.xlsx`**입니다.
6. **실행 검증 범위:** 본 문서는 업로드된 스크립트의 기능과 입력·출력 경로를 분석해 작성했습니다. 빈 값이 있는 예시 데이터로 실제 수치 분석을 재현하거나 그래프 출력을 검증한 것은 아닙니다.

---

## 🛠️ 사용 기술

| 구분 | 기술 |
| --- | --- |
| 프로그래밍 언어 | Python |
| 데이터 처리 | pandas, NumPy |
| 통계 분석 | 기초 통계, Pearson 상관계수, 단순 선형회귀 |
| 데이터 시각화 | Matplotlib |
| 데이터 파일 처리 | CSV, Excel (openpyxl) |

## 📝 참고

본 프로젝트는 공개 통계자료를 바탕으로 충청북도 지역의 토지이용 특성을 **탐색적으로 분석하고 시각화하는 용도**로 구성되었습니다. 실제 데이터 사용 시에는 원본 자료의 기준일·행정구역 구분·단위 등을 확인해 주세요.
