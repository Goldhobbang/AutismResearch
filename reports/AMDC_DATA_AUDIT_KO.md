# AMDC (Health-Spa Dataset) 데이터 무결성 및 적합성 감사 보고서 (AMDC_DATA_AUDIT_KO.md)

- **연구 과제**: 연구 C. AMDC 조도·소음과 생체 반응 연관성 및 예측 기여도 분석 — 데이터 감사
- **작성 일자**: 2026-10-25
- **감사 스크립트**: [`experiments/amdc/audit_and_preprocess.py`](file:///D:/Autism/experiments/amdc/audit_and_preprocess.py)
- **등록 매니페스트**: [`data/registry/amdc/amdc_files_manifest.csv`](file:///D:/Autism/data/registry/amdc/amdc_files_manifest.csv)
- **정제 통합 데이터**: [`data/processed/amdc/amdc_all_clean.csv`](file:///D:/Autism/data/processed/amdc/amdc_all_clean.csv)

---

## 1. 데이터셋 식별 및 출처 (Dataset Identity & Provenance)

### 1.1 데이터셋 공식 명칭 및 출처
- **공식 명칭**: A Multimodal Dataset of Cardiac, Electrodermal, and Environmental Signals (통칭 Health-Spa Dataset)
- **발행 학술지**: *Scientific Data* (Nature Portfolio), 2022년 9권 550호 (DOI: 10.1038/s41597-022-01662-8)
- **연구 주관**: Cezar Anicai, Muhammad Zeeshan Shakir (University of the West of Scotland, Artificial Intelligence and Data Analytics Lab)
- **실제 로컬 저장 절대 경로**: `D:\Autism\data\raw\AMDC\Health-spa Dataset\Health-spa Dataset`
- **데이터 보존 원칙**: 원본 파일은 읽기 전용(Read-Only)으로 취급되었으며, 원시 파일 일체의 수정이나 이동 없이 감사 및 특성 추출을 수행함.

---

## 2. 데이터 구성 및 파일 매니페스트 (File Inventory & Hashes)

### 2.1 파일 구조 및 원시-통합 관계
데이터셋 디렉토리는 14명 피험자(`P1` ~ `P14`)에 대한 1Hz 통합 데이터프레임(`Final_Dataframe_Pn.csv`) 14개, 메타데이터 문서(`ReadMe.txt`), 그리고 각 피험자 하위 폴더의 원시 센서 데이터로 구성되어 있다:
- `Final_Dataframe_Pn.csv`: 1Hz 동기화 통합 파일 (11개 컬럼, 연구 C의 주 분석 대상)
- `P1` ~ `P14` 원시 하위 폴더:
  - `ambient/`: Bosch BME680 및 주변 조도/공기질 센서 (온도, 습도, 기압, 조도, IAQ, PM, eCO2)
  - `sound/`: 100Hz 마이크로폰 음압 레벨 센서 (Sound dB)
  - `ecg/`: 250Hz 단일 유도 ECG 파형 신호
  - `rtor/`: 250Hz R-peak 검출 및 R-R 간격(R-to-R) 타임스탬프

### 2.2 14개 통합 데이터프레임 SHA-256 무결성 검증

| 피험자 ID | 파일명 | 크기 (Bytes) | 행 수 (1Hz) | 수집 시간 (분) | 공식 SHA-256 해시 |
|:---|:---|:---:|:---:|:---:|:---|
| **P1** | Final_Dataframe_P1.csv | 209,010 | 1,972 | 32.87 | `d75317b1f1cc53e6cdf959941bb77622235565850561961af3a336a8de140f72` |
| **P2** | Final_Dataframe_P2.csv | 255,418 | 2,420 | 40.33 | `e06170f8b4c90a352fb83427ac34e394bdef21a92adc72885b972cf7512daaa7` |
| **P3** | Final_Dataframe_P3.csv | 276,710 | 2,621 | 43.68 | `83b25cab4c166156fb15ad624e58021b3d7a20cf6f3e97b02f8ffab58229ab5a` |
| **P4** | Final_Dataframe_P4.csv | 284,308 | 2,720 | 45.33 | `f789080d123df7a6582b08acc4c3175c8020cc0ad0bd261bca270c11ed3f8775` |
| **P5** | Final_Dataframe_P5.csv | 201,415 | 1,941 | 32.35 | `f7646108e03d37335173c66970e27de6e8aa087017428daf15d5af1f0a006816` |
| **P6** | Final_Dataframe_P6.csv | 220,433 | 1,949 | 32.48 | `995092dac2e2dec42cd6619231b03d4300f34385b5fbf29df1708cc944fc2b61` |
| **P7** | Final_Dataframe_P7.csv | 206,881 | 1,978 | 32.97 | `dbb7e96e83e33bdabbf46b1d7dd7f6c92bea7002b88d5fd248f4b834280c057a` |
| **P8** | Final_Dataframe_P8.csv | 281,485 | 2,728 | 45.47 | `fb0655f1a223f5cd7736c832ee1fe439f7502a43f48ba0e154eb3857d41dd35d` |
| **P9** | Final_Dataframe_P9.csv | 291,914 | 2,809 | 46.82 | `9b18d4d5ed334b410723e44a931a0cc5cb93d145eac342023e2fd9c715e0a5bc` |
| **P10** | Final_Dataframe_P10.csv | 271,328 | 2,642 | 44.03 | `a4db0f34cc0b53b81527b04f6e4dd953a6cbe7003c4941b87dffee2ad8bd86e6` |
| **P11** | Final_Dataframe_P11.csv | 273,055 | 2,653 | 44.22 | `4595cfc577a53a6a04fa9a2a93c1b62bbb3d51123de8b1b545f3fc6c71f745c7` |
| **P12** | Final_Dataframe_P12.csv | 296,930 | 2,660 | 44.33 | `a8bdf347e0b4f37d460273dcdaf2b0b70a94ea3fde9d5a4ae53a040473e7b18a` |
| **P13** | Final_Dataframe_P13.csv | 262,529 | 2,552 | 42.53 | `1d2cc07e2951a9b5c6ec735f4182c11c5c69a39cbb3ead6fc778f30733e48478` |
| **P14** | Final_Dataframe_P14.csv | 235,881 | 2,348 | 39.13 | `2ee72ed3e9f62cc3bf77d6ebe70cb34afdd28104672ec62b3b47af022f30894d` |
| **전체 합계** | **14명** | **3,567,290** | **33,993** | **566.55 분** | *전원 결측치 없음 (NaN 0건)* |

---

## 3. 센서 채널, 물리량 단위 및 변환 수식 (Physical Units & Conversions)

### 3.1 원시 측정 채널 명세
`Final_Dataframe_Pn.csv`의 11개 채널은 다음과 같은 물리적 의미를 갖는다:
1. `Skin Resistance`: 피험자 손가락에 부착된 생체 센서로부터 측정된 피부 저항 (단위: $\text{k}\Omega$)
2. `Lux`: 환경 조도 센서 측정값 (단위: $\text{lx}$)
3. `PM4`, `PM5`, `PM6`: 환경 미세먼지 입자 농도 채널
4. `Temperature`: 실내 환경 온도 (단위: $^\circ\text{C}$)
5. `Humidity`: 실내 환경 상대 습도 (단위: $\%$)
6. `Pressure`: 대기압 (단위: $\text{hPa}$)
7. `IAQ`: 실내 공기질 지수 (Indoor Air Quality index)
8. `Sound`: 실내 환경 소음 음압 레벨 (단위: $\text{dB}$)
9. `Heart Rate`: ECG 기반 산출된 실시간 심박수 (단위: $\text{bpm}$)

### 3.2 피부저항($R$)의 피부전도도($G$) 물리적 변환
기존 WESAD 및 PhysioNet 데이터셋은 피부 전기 활동을 피부전도도(EDA/GSR, $\mu\text{S}$) 단위로 다룬다. 본 데이터셋의 `Skin Resistance`($R$, 단위: $\text{k}\Omega$)는 다음과 같은 물리 법칙에 따라 표준 피부전도도($G$, 단위: $\mu\text{S}$)로 정밀 변환된다:

$$G (\mu\text{S}) = \frac{1}{R (\Omega)} \times 10^6 = \frac{1}{R (\text{k}\Omega) \times 10^3} \times 10^6 = \frac{1000}{R (\text{k}\Omega)}$$

- 단위 검증: $R = 100\,\text{k}\Omega \implies G = 10\,\mu\text{S}$, $R = 50\,\text{k}\Omega \implies G = 20\,\mu\text{S}$. 인간의 일반적인 피부전도도 범위($2 \sim 30\,\mu\text{S}$)와 정확히 부합함을 검증함.
- 역수 변환 시 0으로 나누기 방지를 위해 $\epsilon = 1.0\times 10^{-6}$ 보호 적용.
- 해당 변환 공식은 [`tests/test_followup_studies.py`](file:///D:/Autism/tests/test_followup_studies.py)의 `test_amdc_conductance_conversion` 단위 테스트를 통해 100% 통과 검증됨.

---

## 4. 데이터셋 라벨 부재 감사 및 정식화 (Absence of Stress Labels & Task Formulation)

### 4.1 중대 사실 감사: 스트레스 및 멜트다운 라벨 부재
- **사실 확인**: AMDC(Health-Spa) 데이터셋에는 **자폐 진단(ASD Diagnosis), 멜트다운 발생 시점, 또는 심리학적 스트레스/불안에 대한 어떠한 정답 라벨(Ground Truth Label)도 존재하지 않는다**.
- 수집 환경은 통제된 리빙랩/스파형 사무 환경에서 건강한 성인 14명이 환경 변화에 노출되는 동안의 자율신경 반응을 기록한 것이다.
- **연구 윤리적 원칙 준수**: 존재하지 않는 스트레스 라벨을 임의로 생성하거나 휴리스틱 규칙으로 조작하여 "자폐 아동 스트레스/멜트다운 탐지 성능"으로 포장하는 행위를 절대 금지함.

### 4.2 연구 C의 과학적 과제 정식화
라벨이 부재하므로, 연구 C는 다음과 같은 객관적이고 반증 가능한 시계열 인과성 및 예측 회귀(Predictive Regression) 과제로 엄격하게 정식화되었다:
> **"웨어러블 폼팩터에 조도(Lux) 및 소음(Sound dB) 환경 센서를 추가 탑재하는 것이 착용자의 심박수(HR) 및 피부전도도(EDA)의 미래 변화를 예측하는 데 통계적·실용적으로 기여하는가?"**

- **입력 시계열 창**: 과거 60초간의 생체 및 환경 관측 신호 $[t-59, t]$
- **예측 대상 타겟**: 미래 30초간의 자율신경 반응 평균치 $[t+1, t+30]$ (심박수 `HR`, 피부전도도 `Conductance`)
- **시간 누수 원천 차단**: 입력 창과 타겟 구간 사이에 1초의 시간적 중첩도 허용하지 않음 ($t < t+1$).
- **평가 프로토콜**: 14명 피험자 완전 분리 Leave-One-Subject-Out (LOSO) 교차검증.

---

## 5. 데이터 품질 및 최종 적격성 평가 결론

1. **무결성 합격**: 14개 통합 CSV 파일 모두 공식 SHA-256 해시가 검증되었으며 결측치(NaN) 0건, 비정상적 타임스탬프 단절 없음.
2. **생체-환경 정렬 완성**: 총 33,993초(약 9.44시간)의 고품질 1Hz 멀티모달 동기화 데이터 확보 완료.
3. **적격 창 수**: 60초 입력 창 / 15초 이동 간격 적용 시 14명 전원에서 총 2,189개의 독립된 예측 샘플 창 생성.
4. **결론**: 데이터셋은 환경 요인과 생체 신호 간의 상관성 분석 및 미래 생체 반응 예측 모델 비교(연구 C)를 수행하기에 완벽한 무결성과 적합성을 갖춤.
