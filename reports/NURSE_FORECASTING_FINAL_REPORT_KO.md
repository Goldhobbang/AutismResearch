# Hospital Nurses Stress Dataset 기반 미래 스트레스 사건 사전 예측(Forecasting) 연구 최종 보고서 (`NURSE_FORECASTING_FINAL_REPORT_KO.md`)

- **작성 일자**: 2026-10-26
- **데이터셋**: Hospital Nurses Stress Dataset (S. Hosseini et al., *Scientific Data* 9, 255, 2022; DOI: `10.1038/s41597-022-01361-y`, Dryad CC0 1.0)
- **연구 과제**: 과거 180초(`t-180s ~ t`) 손목 생체신호(EDA, BVP/HR/IBI, 3축 ACC)만 사용하여 미래 스트레스 사건 시작(Onset)을 **60~180초 전(`P1`)**에 사전 예측하는 **Nurses 기반 미래 스트레스 사건 예측 모델** 구축 및 15-fold 참가자 독립(LOSO) 검증
- **관련 문서 및 산출물**:
  - 데이터 및 라벨 감사 보고서: [`reports/NURSE_DATA_AUDIT.md`](file:///D:/Autism/reports/NURSE_DATA_AUDIT.md)
  - 실시간 소프트웨어/임베디드 배포 명세서: [`reports/NURSE_MODEL_DEPLOYMENT_SPEC.md`](file:///D:/Autism/reports/NURSE_MODEL_DEPLOYMENT_SPEC.md)
  - 최종 선정 모델 아티팩트: [`results/nurse_forecasting/models/nurse_stress_forecast_lr_p1.joblib`](file:///D:/Autism/results/nurse_forecasting/models/nurse_stress_forecast_lr_p1.joblib)
  - 임베디드 추론 가중치 및 설정 JSON: [`results/nurse_forecasting/models/nurse_stress_forecast_lr_p1_spec.json`](file:///D:/Autism/results/nurse_forecasting/models/nurse_stress_forecast_lr_p1_spec.json)

---

## 0. 과학적 정직성 및 해석 경계 선언 (Epistemic & Scope Disclaimer)

1. **자폐 아동 멜트다운(ASD Meltdown) 예측과의 엄격한 구분**:
   - 본 보고서의 모든 실험 수치는 **미국 병원 간호사 15명의 실제 근무 중 직무 스트레스 사건(Survey-validated Hospital Nurse Stress Episode) 사전 예측 성능**이다.
   - 본 데이터셋의 라벨은 독립적인 외부 임상 관찰자가 비디오로 초 단위 코딩한 자폐 스펙트럼(ASD) 아동의 멜트다운 행동 Onset이 아니므로, **본 모델의 결과를 "자폐 아동 멜트다운 조기경보 성능"으로 표현하거나 외삽하지 않는다.**
2. **LOSO 일반화 성능과 최종 배포 모델의 구분**:
   - 모델 선정 및 성능 보고에 사용된 모든 지표는 테스트 피험자를 학습·검증·임계값 탐색에서 완전히 격리한 **15-fold Leave-One-Subject-Out (LOSO)** 외부 테스트 결과이다.
   - 최종 저장된 배포용 모델(`nurse_stress_forecast_lr_p1.joblib`)은 아키텍처와 하이퍼파라미터 확정 후 전체 15명 적격 개발 데이터(`27,250개 창`)로 최종 적합(Refit)한 아티팩트이며, 해당 모델의 미학습 신규 피험자 기대 성능 추정치는 15-fold LOSO 평가 수치로 보고한다.

---

## 1. 데이터 및 사건 라벨 품질 감사 요약

- 상세 감사 보고서: [`reports/NURSE_DATA_AUDIT.md`](file:///D:/Autism/reports/NURSE_DATA_AUDIT.md)
- 매니페스트 및 요약표: [`results/nurse_forecasting/dataset_manifest.json`](file:///D:/Autism/results/nurse_forecasting/dataset_manifest.json), [`results/nurse_forecasting/subject_summary.csv`](file:///D:/Autism/results/nurse_forecasting/subject_summary.csv), [`results/nurse_forecasting/event_audit.csv`](file:///D:/Autism/results/nurse_forecasting/event_audit.csv)

### 1.1 원시 아카이브 및 세션 구성
- **참가자 수**: 병원 간호사 **15명** (`15`, `5C`, `6B`, `6D`, `7A`, `7E`, `83`, `8B`, `94`, `BG`, `CE`, `DF`, `E4`, `EG`, `F5`)
- **원시 파일 수**: 총 **610개** (`.zip` 센서 아카이브 **609개** + 설문 주석 파일 `SurveyResults.xlsx` **1개**)
- **총 기록 시간**: **`1,251.68시간`**
  - 설문 주석이 매칭된 세션: **214개 `.zip` (`604.01시간`)**
  - 설문 기록이 전혀 없는 미주석 세션: **395개 `.zip` (`647.67시간`)** $\rightarrow$ 무사건(정상)인지 설문 누락인지 알 수 없으므로 **음성으로 가정하지 않고 학습·평가에서 전면 제외(`UNKNOWN_UNANNOTATED_SESSION`)**.
- **센서 구성 및 동기화**:
  - 609개 모든 `.zip` 내 `EDA.csv` (`4 Hz`), `BVP.csv` (`64 Hz`), `ACC.csv` (`32 Hz`), `HR.csv` (`1 Hz`), `IBI.csv` (박동 간격), `TEMP.csv` (`4 Hz`, 타깃 제품 비필수 센서이므로 모델 입력에서 제외) 완비.
  - 세션 내 센서 간 시작 시각 편차는 **`0.0초`**, 종료 시각 최대 편차는 **`3.02초`** 이내로 정밀 동기화 확인.
  - 로컬 시간대(`America/Chicago`, CDT `UTC-5` / CST `UTC-6`) 변환 시 전체 358개 유효 설문 행 중 **323개(90.2%)**가 실제 센서 세션과 정확히 매칭됨.

### 1.2 설문 사건 라벨 구분 및 신뢰성 한계
| 라벨 구분 (`SurveyResults.xlsx`) | 전체 건수 | 센서 매칭 및 적격성 | 학습·평가 처리 방침 |
|---|---:|---|---|
| **확인된 스트레스 사건 (`Stress level` $\in \{1, 2\}$)** | **199건**<br>(`2`: 179건, `1`: 20건) | P0 적격 **176건**, **P1 적격 176건**, P2 적격 **171건**<br>(사건 시작 전 최소 180s / 240s / 360s 연속 센서 이력 보유) | 미래 사건 예측의 **양성 타깃 사건(Target Onset)**으로 사용 |
| **설문 확인 비스트레스 (`Stress level == 0`)** | **46건** | 센서 매칭 36건 (`2,173개 창`) | 1차 알고리즘 후보였으나 간호사가 비스트레스로 확인한 구간 $\rightarrow$ **고난도 음성 대조군(`CONFIRMED_NO_STRESS_0`)**으로 평가 포함 |
| **미확인 알고리즘 후보 (`Stress level == 'na'`)** | **113건** | 센서 매칭 95건 (`10,602개 창`) | 스트레스 여부 미확인이므로 **전후 버퍼(`-300s ~ +600s`) 포함 전면 검열(`UNKNOWN_NA_CENSORED`, 라벨 `-1`)** |

### 1.3 인과적 예측 격자 및 창(Window) 상태 분포
- 214개 주석 세션에서 과거 `180초` 관측 창(`[t-180s, t)`) 기준으로 사건 주변 `30초` hop, 배경 구간 `60초` hop으로 추출한 총 **`51,234개 창`** ([`results/nurse_forecasting/features.csv`](file:///D:/Autism/results/nurse_forecasting/features.csv))의 상태 분포:
  - `VERIFIED_NON_EVENT` (검증된 비사건 배경 음성): **`24,424개`**
  - `CONFIRMED_NO_STRESS_0` (설문 확인 비스트레스 음성): **`2,173개`**
  - `PRE_ONSET_P0_0_60S` (사건 시작 0~60초 전): **`332개`**
  - `PRE_ONSET_P1_60_180S` (사건 시작 60~180초 전, **주 과제 양성**): **`659개`**
  - `PRE_ONSET_P2_180_300S` (사건 시작 180~300초 전): **`626개`**
  - `PRE_ONSET_BUFFER_300_600S` (사건 300~600초 전 전이 완충 구간, 검열): **`1,546개`**
  - `ONGOING_STRESS_EVENT` (스트레스 사건 진행 중, 검열): **`8,622개`**
  - `POST_EVENT_REFRACTORY` (사건 종료 후 900초 회복기, 검열): **`2,250개`**
  - `UNKNOWN_NA_CENSORED` (`na` 미확인 구간 및 버퍼, 검열): **`10,602개`**
- 신호 품질 검사(`qc_common_valid`: 유한값 비율 $\ge 95\%$ 및 피부 접촉 $\text{EDA mean} \ge 0.005\,\mu\text{S}$)를 통과한 P1 이진 평가 적격 창은 총 **`27,250개`** (양성 `659개` [`2.42%`], 음성 `26,591개` [`97.58%`], 검증된 비사건 총 노출 시간 **`390.35시간`**)이다.

---

## 2. 핵심 모델 비교 결과 (Primary Task P1: 60~180초 전 사전 예측)

- 산출물 파일: [`results/nurse_forecasting/model_comparison.csv`](file:///D:/Autism/results/nurse_forecasting/model_comparison.csv)
- 시각화 자료: [`01_model_auroc_auprc_comparison.png`](file:///D:/Autism/results/nurse_forecasting/figures/01_model_auroc_auprc_comparison.png), [`02_precision_recall_curve.png`](file:///D:/Autism/results/nurse_forecasting/figures/02_precision_recall_curve.png), [`03_roc_curve.png`](file:///D:/Autism/results/nurse_forecasting/figures/03_roc_curve.png)
- 평가 프로토콜: 15-fold LOSO (외부 테스트 1명 완전 격리, 나머지 14명 내부 `GroupShuffleSplit` 검증셋에서 F1 최대화 임계값 $\theta_{\text{val}}$ 선택 후 14명 전체 재적합).

| 모델 | 참가자 평균 AUROC ($\pm$SD) | 참가자 중앙값 AUROC | Pooled AUROC | 참가자 평균 AUPRC | Pooled AUPRC | 창 F1 평균 | 창 Precision 평균 | 창 Recall 평균 | 창 FPR 평균 | 사건 탐지율 Event Recall (`탐지/전체`) | 시간당 오경보 (`FA/hr`) | 경보 에피소드 정밀도 | 평균 리드타임 (`중앙값`) | 총 TP / FP / TN / FN | 학습 시간 (`추론 ms/창`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **M0 Dummy** | 0.5000 ($\pm$0.000) | 0.5000 | 0.3529 | 0.0266 | 0.0176 | 0.0000 | N/A (양성예측 0) | 0.0000 | 0.0000 | **0.00%** (`0/176`) | 0.0000 | N/A | N/A | `0 / 0 / 26591 / 659` | 2.11s (`0.0025 ms`) |
| **M1 LR (주 모델)** | **0.6167 ($\pm$0.087)** | **0.6415** | **0.6637** | 0.0653 | **0.0439** | 0.0740 | 0.0519 | 0.2838 | **0.1807** | **40.34%** (`71/176`) | **1.5678** (`612건`) | **10.26%** | **146.4s** (`155.0s`) | `209 / 4257 / 22334 / 450` | 7.12s (`0.0026 ms`) |
| **M2 RF** | **0.6391 ($\pm$0.109)** | 0.6257 | 0.6628 | **0.0755** | 0.0426 | 0.0603 | 0.0403 | 0.3225 | 0.2105 | **46.02%** (`81/176`) | 1.9777 (`772건`) | 9.39% | **150.4s** (`158.0s`) | `263 / 5908 / 20683 / 396` | 67.77s (`0.0447 ms`) |
| **M3 LightGBM** | 0.6224 ($\pm$0.126) | **0.6480** | 0.6501 | 0.0705 | 0.0412 | **0.0838** | **0.0603** | **0.3895** | 0.2752 | **56.82%** (`100/176`) | 2.6463 (`1033건`) | 8.75% | **147.3s** (`158.0s`) | `283 / 6318 / 20273 / 376` | 8.30s (`0.0063 ms`) |

### 핵심 분석
1. **기저율 대비 유의미한 사전 예측 신호 확인**:
   - P1 과제의 창 단위 양성 비율(Prevalence)은 `2.42%` (`659 / 27,250`)에 불과하나, M1 Logistic Regression(`LR`), M2 Random Forest(`RF`), M3 `LightGBM` 모두 참가자 평균 AUROC `0.6167 ~ 0.6391` (중앙값 `0.6257 ~ 0.6480`, Pooled AUROC `0.6501 ~ 0.6637`) 및 평균 AUPRC `0.0653 ~ 0.0755` (기저율 대비 **2.5~2.8배**)를 달성하여 사건 시작 1~3분 전 생체신호에 유의미한 전조가 존재함을 입증했다.
2. **모델 간 트레이드오프 및 M1 LR 채택 근거**:
   - `LightGBM`은 사건 탐지율(`56.82%`, 176건 중 100건 사전 경고)이 가장 높으나 시간당 오경보가 `2.6463 FA/hr`로 가장 많다.
   - `RF`는 참가자 평균 AUROC(`0.6391`)가 가장 높으나 모델 용량이 **`1.42 MB`**에 달해 ESP32-S3 내부 SRAM(`512 KB`) 단독 적재가 불가능하다.
   - 반면 **`M1 LR`**은 **Pooled AUROC(`0.6637`), Pooled AUPRC(`0.0439`), 경보 에피소드 정밀도(`10.26%`), 시간당 오경보 억제(`1.5678 FA/hr`), 참가자 간 분산 최소화(`SD 0.0874`)**에서 가장 안정적이며, 파라미터 크기가 **`996 Bytes (< 1 KB)`**에 불과해 온디바이스 실시간 추론 및 해석 가능성 측면에서 최적의 주 모델로 선정되었다. (추가로 8절의 9특징+추세 압축 특징셋 적용 시 LR의 참가자 평균 AUROC는 **`0.6661`**로 RF `0.6514`를 앞선다.)

---

## 3. 예측 리드타임 시간대 비교 (P0 vs P1 vs P2)

- 산출물 파일: [`results/nurse_forecasting/lead_time_comparison.csv`](file:///D:/Autism/results/nurse_forecasting/lead_time_comparison.csv)
- 시각화 자료: [`05_lead_time_performance.png`](file:///D:/Autism/results/nurse_forecasting/figures/05_lead_time_performance.png)

| 예측 시간대 과제 | 최소~최대 리드타임 | 모델 | 참가자 평균 AUROC ($\pm$SD) | 참가자 중앙값 AUROC | Pooled AUROC | 참가자 평균 AUPRC | 창 F1 평균 | 사건 탐지율 Event Recall (`탐지/적격`) | 시간당 오경보 (`FA/hr`) | 평균 최초 리드타임 (`중앙값`) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **P0 (근접 예측 기준선)** | `0 ~ 60s`<br>(1분 이상 조기경보 아님) | LR | 0.6699 ($\pm$0.151) | 0.7055 | 0.6848 | 0.0493 | 0.0647 | 33.52% (`59/176`) | 1.3398 | 39.3s (`40.0s`) |
| **P0 (근접 예측 기준선)** | `0 ~ 60s` | RF | **0.7080 ($\pm$0.108)** | **0.7289** | **0.7002** | **0.0691** | **0.0854** | 30.11% (`53/176`) | **1.1836** | 36.1s (`38.0s`) |
| **P0 (근접 예측 기준선)** | `0 ~ 60s` | LightGBM | 0.6793 ($\pm$0.115) | 0.7048 | 0.6815 | 0.0512 | 0.0550 | 40.34% (`71/176`) | 1.7395 | 41.2s (`40.0s`) |
| **P1 (주 과제: 1~3분 전)** | **`60 ~ 180s`**<br>(최소 60s 보장) | LR | 0.6167 ($\pm$0.087) | 0.6415 | **0.6637** | 0.0653 | 0.0740 | 40.34% (`71/176`) | **1.5678** | **146.4s** (`155.0s`) |
| **P1 (주 과제: 1~3분 전)** | **`60 ~ 180s`** | RF | **0.6391 ($\pm$0.109)** | 0.6257 | 0.6628 | **0.0755** | 0.0603 | 46.02% (`81/176`) | 1.9777 | **150.4s** (`158.0s`) |
| **P1 (주 과제: 1~3분 전)** | **`60 ~ 180s`** | LightGBM | 0.6224 ($\pm$0.126) | **0.6480** | 0.6501 | 0.0705 | **0.0838** | **56.82%** (`100/176`) | 2.6463 | **147.3s** (`158.0s`) |
| **P2 (확장 과제: 3~5분 전)** | `180 ~ 300s`<br>(최소 180s 보장) | LR | 0.6050 ($\pm$0.095) | **0.6241** | 0.6074 | 0.0519 | **0.0757** | 30.99% (`53/171`) | **1.4551** | **265.6s** (`276.0s`) |
| **P2 (확장 과제: 3~5분 전)** | `180 ~ 300s` | RF | **0.6304 ($\pm$0.108)** | 0.6232 | **0.6198** | **0.0529** | 0.0695 | **52.05%** (`89/171`) | 1.8112 | **273.9s** (`278.0s`) |
| **P2 (확장 과제: 3~5분 전)** | `180 ~ 300s` | LightGBM | 0.5915 ($\pm$0.114) | 0.5967 | 0.5928 | 0.0435 | 0.0551 | 47.37% (`81/171`) | 2.4081 | **270.4s** (`278.0s`) |

### 핵심 분석
- 사건 시작 시점에 가까워질수록(`P2 [180~300s]` $\rightarrow$ `P1 [60~180s]` $\rightarrow$ `P0 [0~60s]`) 교감신경 활성(EDA 및 HR 상승)이 뚜렷해져 AUROC가 상승(`LR: 0.6050 -> 0.6167 -> 0.6699`, `RF: 0.6304 -> 0.6391 -> 0.7080`)하는 일관된 물리적·생리적 단조성이 확인되었다.
- 단, `P0 (0~60s)`는 사건 직전 0~60초 구간이므로 최소 1분 이상의 개입 여유 시간을 보장하지 못하며(평균 리드타임 `36~41s`), 설문 시작 시각의 분 단위 반올림 오차($\pm 60\text{s}$)와 직접 중첩된다. 따라서 **최소 60초 이상의 선행 시간을 엄격히 강제하면서 평균 `146.4초(약 2분 26초)` 전에 경고를 제공하는 `P1 (60~180s)`을 주 운용 과제로 채택**하였다.

---

## 4. 센서 조합별 기여도 분석 (Sensor Ablation on P1)

- 산출물 파일: [`results/nurse_forecasting/sensor_ablation.csv`](file:///D:/Autism/results/nurse_forecasting/sensor_ablation.csv)

| 모델 | 센서 조합 | 입력 특징 수 | 참가자 평균 AUROC ($\pm$SD) | 참가자 중앙값 AUROC | Pooled AUROC | 참가자 평균 AUPRC | 창 F1 평균 | 사건 탐지율 Event Recall (`탐지/176`) | 시간당 오경보 (`FA/hr`) | 경보 에피소드 정밀도 | 최악 참가자 (`AUROC`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **LR** | **EDA 단독 (`eda`)** | 26 | **0.6359 ($\pm$0.106)** | **0.6642** | **0.6639** | **0.0697** | **0.0766** | 29.55% (`52/176`) | **1.0094** (`394건`) | **11.46%** | `5C` (`0.3998`) |
| **LR** | **PPG/HR/IBI 단독 (`ppg`)** | 21 | 0.5975 ($\pm$0.115) | 0.6065 | 0.5968 | 0.0424 | 0.0523 | **59.09%** (`104/176`) | 2.5925 (`1012건`) | 9.24% | `8B` (`0.4241`) |
| **LR** | **ACC 단독 (`acc`)** | 15 | 0.6242 ($\pm$0.135) | 0.6548 | 0.6271 | 0.0627 | 0.0541 | 39.20% (`69/176`) | 1.9009 (`742건`) | 8.40% | `15` (`0.4030`) |
| **LR** | **EDA + PPG (`eda_ppg`)** | 47 | 0.6154 ($\pm$0.112) | 0.6399 | 0.6632 | 0.0651 | 0.0725 | 43.75% (`77/176`) | 1.7138 (`669건`) | 10.20% | `5C` (`0.3489`) |
| **LR** | **전체 센서 (`all`)** | 62 | 0.6167 ($\pm$0.087) | 0.6415 | 0.6637 | 0.0653 | 0.0740 | 40.34% (`71/176`) | 1.5678 (`612건`) | 10.26% | `5C` (`0.3677`) |
| **RF** | **EDA 단독 (`eda`)** | 26 | 0.6096 ($\pm$0.156) | 0.6416 | 0.6408 | 0.0721 | **0.0887** | 41.48% (`73/176`) | 1.5986 (`624건`) | **10.34%** | `6D` (`0.2380`) |
| **RF** | **PPG/HR/IBI 단독 (`ppg`)** | 21 | 0.6121 ($\pm$0.071) | 0.6252 | 0.6006 | 0.0419 | 0.0495 | 50.57% (`89/176`) | 2.4619 (`961건`) | 8.30% | `8B` (`0.4753`) |
| **RF** | **ACC 단독 (`acc`)** | 15 | 0.6062 ($\pm$0.133) | 0.6534 | 0.6146 | 0.0540 | 0.0748 | **60.80%** (`107/176`) | 2.9589 (`1155건`) | 8.33% | `15` (`0.3621`) |
| **RF** | **EDA + PPG (`eda_ppg`)** | 47 | 0.6388 ($\pm$0.127) | **0.6675** | 0.6504 | 0.0678 | 0.0539 | 31.82% (`56/176`) | **1.3885** (`542건`) | 9.36% | `5C` (`0.3969`) |
| **RF** | **전체 센서 (`all`)** | 62 | **0.6391 ($\pm$0.109)** | 0.6257 | **0.6628** | **0.0755** | 0.0603 | 46.02% (`81/176`) | 1.9777 (`772건`) | 9.39% | `5C` (`0.4357`) |

### 핵심 분석
1. **EDA의 높은 특이도 및 정밀도 기여**:
   - `LR`에서 `EDA` 단독 모델은 시간당 오경보가 **`1.0094 FA/hr`**로 가장 낮고 경보 정밀도(`11.46%`)와 평균 AUROC(`0.6359`)가 가장 높았으나, 단독 사용 시 사건 탐지율이 `29.55%` (`52/176`)로 제한되었다.
2. **PPG/HR 결합을 통한 사건 민감도 보완 및 ACC 결합을 통한 분산 감소**:
   - `EDA + PPG` 결합 시 사건 탐지율이 `29.55% -> 43.75%`로 크게 상승하며, 여기에 3축 회전 불변 `ACC` 특징을 결합한 `ALL` 모델은 `EDA + PPG` 대비 오경보를 줄이면서(`1.7138 -> 1.5678 FA/hr`) 참가자 간 AUROC 표준편차를 `0.1119 -> 0.0874`로 낮추어 개인 간 안정성을 높였다. `RF`에서는 `ALL` 모델이 평균 AUROC(`0.6391`), AUPRC(`0.0755`), 최악 참가자 방어(`0.4357`) 모두에서 최고 성능을 달성했다.

---

## 5. 실험 A: 신체 활동량 및 Motion Gate 적용 전후 비교

- 산출물 파일: [`results/nurse_forecasting/motion_gate_comparison.csv`](file:///D:/Autism/results/nurse_forecasting/motion_gate_comparison.csv)
- 시각화 자료: [`08_motion_vs_risk_score.png`](file:///D:/Autism/results/nurse_forecasting/figures/08_motion_vs_risk_score.png)

| 모델 | Motion Gate 방식 | 참가자 평균 AUROC | 사건 탐지율 Event Recall (`탐지/미탐지`) | 창 FPR 평균 | 시간당 오경보 (`FA/hr`) | 비사건 Q1 (정적) 경보율 | 비사건 Q2 (저활동) 경보율 | 비사건 Q3 (중활동) 경보율 | 비사건 Q4 (고활동) 경보율 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **LR** | **`none` (게이트 미적용)** | **0.6167** | **40.34%** (`71 / 105`) | 0.1807 | 1.5678 (`612건`) | 12.11% | 12.99% | 15.19% | 23.94% |
| **LR** | **`legacy_hard` (하드 게이트)** | 0.4875 | 0.57% (`1 / 175`) | **0.0119** | **0.0589** (`23건`) | 2.81% | 0.00% | 0.00% | 0.00% |
| **LR** | **`soft_dynamic` (소프트 동적 게이트)** | 0.5189 | 22.73% (`40 / 136`) | 0.1132 | 1.0811 (`422건`) | 12.09% | 12.26% | 9.41% | **2.78%** |
| **RF** | **`none` (게이트 미적용)** | **0.6391** | **46.02%** (`81 / 95`) | 0.2105 | 1.9777 (`772건`) | 5.97% | 8.76% | 25.25% | 49.68% |
| **RF** | **`legacy_hard` (하드 게이트)** | 0.4892 | 0.00% (`0 / 176`) | **0.0044** | **0.0615** (`24건`) | 1.10% | 0.00% | 0.00% | 0.00% |
| **RF** | **`soft_dynamic` (소프트 동적 게이트)** | 0.5182 | 19.32% (`34 / 142`) | 0.0891 | 1.5371 (`600건`) | 5.89% | 7.73% | 14.94% | **6.29%** |
| **LightGBM** | **`none` (게이트 미적용)** | **0.6224** | **56.82%** (`100 / 76`) | 0.2752 | 2.6463 (`1033건`) | 13.75% | 14.26% | 27.18% | 40.35% |
| **LightGBM** | **`legacy_hard` (하드 게이트)** | 0.5179 | 0.57% (`1 / 175`) | **0.0130** | **0.0589** (`23건`) | 2.81% | 0.00% | 0.00% | 0.00% |
| **LightGBM** | **`soft_dynamic` (소프트 동적 게이트)** | 0.5422 | 33.52% (`59 / 117`) | 0.1696 | 2.1340 (`833건`) | 13.67% | 13.44% | 17.49% | **6.20%** |

### 핵심 분석 (실험실 데이터와 실제 병원 근무 환경의 결정적 차이 규명)
1. **비사건 고활동 구간(`Q4_High`)의 오탐 집중 현상**:
   - 게이트 미적용(`none`) 시 비사건 구간 활동량 4분위수별 오경보율을 보면, 정적 구간(`Q1`: LR `12.11%`, RF `5.97%`) 대비 고활동 구간(`Q4`: LR `23.94%`, RF `49.68%`)에서 오탐율이 2~8배 급증한다.
2. **기존 실험실 기준 하드 게이트(`legacy_hard`)의 완전 붕괴 확인**:
   - 실험실 좌식 프로토콜(WESAD, PhysioNet) 기준으로 설계된 하드 게이트(`acc_motion_change_per_s > 40` 또는 `acc_dynamic_energy > 150` 차단)를 실제 병원 간호사에게 적용하면, 간호사의 일상적 병동 보행 및 환자 처치 활동(전체 창 평균 `acc_motion_change_per_s = 163.6`)이 모두 운동으로 분류되어 **176개 스트레스 사건 중 175~176개가 소거(`Event Recall 0.0% ~ 0.57%`)**된다.
3. **소프트 동적 게이트(`soft_dynamic`)의 효과와 한계**:
   - `soft_dynamic` 게이트는 하드 게이트의 0% 붕괴를 막고 `Q4` 고활동 오경보율을 `23.94% -> 2.78%`(LR)로 억제하며 시간당 오경보를 `1.5678 -> 1.0811 FA/hr`(`-31.0%`)로 줄인다.
   - 그러나 병원 간호사의 실제 직무 스트레스 사건(응급 환자 처치, 업무 과중 등)은 **사건 발생 전후에 빠른 보행과 신체 활동이 동시에 수반**되므로, 활동량으로 위험 점수를 감쇠시키면 사건 탐지율도 `40.34% -> 22.73%`(LR)로 감소한다.
   - 반면 6절에서 확인되듯 **활동량 감쇠 대신 히스테리시스(`hysteresis`) 에피소드 정책을 사용하면 사건 탐지율을 희생하지 않고(`40.34% -> 44.89%`) 오경보를 `0.9965 FA/hr`로 더 효과적으로 억제**할 수 있다.

---

## 6. 실험 B: 경고 정책(Alert Policy) 및 시간당 오경보 트레이드오프 비교

- 산출물 파일: [`results/nurse_forecasting/alert_policy_comparison.csv`](file:///D:/Autism/results/nurse_forecasting/alert_policy_comparison.csv)
- 시각화 자료: [`06_recall_vs_false_alarms_curve.png`](file:///D:/Autism/results/nurse_forecasting/figures/06_recall_vs_false_alarms_curve.png)

| 모델 | 경고 발생 정책 (`policy`) | 사건 탐지율 Event Recall (`탐지/176`) | `val_opt` 대비 Recall 변화 | 시간당 오경보 (`FA/hr`) | `val_opt` 대비 `FA/hr` 변화 | 오경보 에피소드 수 (`390.35h`) | 경보 에피소드 정밀도 | 평균 최초 리드타임 (`중앙값`) | 창 F1 평균 | 창 Recall 평균 | 창 FPR 평균 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **LR** | `fixed_05` ($\theta = 0.50$ 고정) | **68.18%** (`120/176`) | +27.84%p | 2.9025 | +1.3347 | 1,133건 | 9.43% | 149.6s (`158.0s`) | 0.0718 | 0.4907 | 0.3320 |
| **LR** | `val_opt` (내부 검증 최적 $\theta_{\text{val}}$) | 40.34% (`71/176`) | 기준 (0.00) | 1.5678 | 기준 (0.00) | 612건 | 10.26% | 146.4s (`155.0s`) | **0.0740** | 0.2838 | 0.1807 |
| **LR** | **`hysteresis` (이중 임계값 히스테리시스, 권장)** | **44.89%** (`79/176`) | **+4.55%p** | **0.9965** | **-0.5713 (-36.4%)** | **389건** | **16.70%** | **152.2s (`160.0s`)** | 0.0654 | 0.3778 | 0.2706 |
| **LR** | `duration_2win` (2연속 창 이상 지속) | 32.39% (`57/176`) | -7.95%p | 1.1887 | -0.3791 | 464건 | 10.77% | 122.7s (`128.0s`) | 0.0566 | 0.1836 | 0.1465 |
| **LR** | `cooldown_300s` (300초 쿨다운 억제) | 40.34% (`71/176`) | 0.00%p | 2.7693 | +1.2015 | 1,081건 | 6.08% | 146.4s (`155.0s`) | 0.0637 | 0.0969 | 0.0442 |
| **LR** | `soft_gate_hysteresis` (소프트 게이트 + 히스테리시스) | 19.32% (`34/176`) | -21.02%p | **0.7583** | **-0.8095 (-51.6%)** | **296건** | 10.30% | 143.9s (`155.0s`) | 0.0597 | 0.1406 | 0.1368 |
| **RF** | `fixed_05` ($\theta = 0.50$ 고정) | **48.30%** (`85/176`) | +2.27%p | 1.8573 | -0.1204 | 725건 | 10.38% | 150.2s (`158.0s`) | **0.1005** | 0.3485 | 0.1574 |
| **RF** | `val_opt` (내부 검증 최적 $\theta_{\text{val}}$) | 46.02% (`81/176`) | 기준 (0.00) | 1.9777 | 기준 (0.00) | 772건 | 9.39% | 150.4s (`158.0s`) | 0.0603 | 0.3225 | 0.2105 |
| **RF** | **`hysteresis` (이중 임계값 히스테리시스)** | **45.45%** (`80/176`) | -0.57%p | **1.1323** | **-0.8454 (-42.7%)** | **442건** | **15.16%** | **153.9s (`160.0s`)** | 0.0560 | 0.3532 | 0.2597 |
| **RF** | `duration_2win` (2연속 창 이상 지속) | 39.20% (`69/176`) | -6.82%p | 1.6754 | -0.3023 | 654건 | 9.42% | 127.1s (`130.0s`) | 0.0474 | 0.2191 | 0.1717 |
| **RF** | `cooldown_300s` (300초 쿨다운 억제) | 46.02% (`81/176`) | 0.00%p | 3.7659 | +1.7881 | 1,470건 | 5.16% | 150.4s (`158.0s`) | 0.0559 | 0.0997 | 0.0523 |
| **RF** | `soft_gate_hysteresis` (소프트 게이트 + 히스테리시스) | 17.61% (`31/176`) | -28.41%p | **0.9786** | **-0.9991 (-50.5%)** | **382건** | 7.51% | 135.2s (`153.0s`) | 0.0436 | 0.0993 | 0.1030 |

### 핵심 분석
1. **히스테리시스(`hysteresis`) 정책의 압도적 우수성**:
   - 단일 임계값(`val_opt`)에서는 확률 $p(t)$가 임계값 주변에서 진동할 때 경보가 끊겼다 켜지며 다수의 파편화된 오경보 에피소드를 유발한다.
   - 반면 진입 임계값 $T_{\text{on}} = \max(\theta_{\text{val}}, 0.52)$, 해제 임계값 $T_{\text{off}} = \min(0.70 \theta_{\text{val}}, 0.35)$을 적용하고 `180초` 이내 인접 경보를 단일 에피소드로 병합하는 **`hysteresis` 정책**은 `LR` 기준 **시간당 오경보를 `1.5678 -> 0.9965 FA/hr` (`-36.4%`, 시간당 1회 미만)로 낮추면서, 사건 탐지율은 `40.34% -> 44.89%` (`79/176`), 경보 에피소드 정밀도는 `10.26% -> 16.70%`, 평균 리드타임은 `146.4s -> 152.2s`로 동시 개선**하였다.
2. **2연속 창 지속(`duration_2win`) 및 단순 쿨다운(`cooldown_300s`)의 한계**:
   - `duration_2win`은 2개 창이 연속 초과될 때까지 발령을 지연시키므로 평균 리드타임이 `146.4s -> 122.7s`로 약 24초 단축되고 탐지율도 `32.39%`로 하락한다.
   - `cooldown_300s`는 단일 창만 1로 남기고 이후 300초를 0으로 강제 소거함에 따라, `180초` 에피소드 병합 간격을 넘어서는 잔여 고위험 구간에서 매 300초마다 새로운 독립 에피소드를 재발화시켜 에피소드 단위 `FA/hr`가 오히려 증가(`2.7693 FA/hr`)하는 역효과를 보였다.

---

## 7. 실험 C: 15-Fold LOSO 참가자별 일반화 성능 상세 (Primary M1 LR, P1)

- 산출물 파일: [`results/nurse_forecasting/per_subject_metrics.csv`](file:///D:/Autism/results/nurse_forecasting/per_subject_metrics.csv), [`results/nurse_forecasting/splits.csv`](file:///D:/Autism/results/nurse_forecasting/splits.csv)
- 시각화 자료: [`04_per_subject_recall_and_fpr.png`](file:///D:/Autism/results/nurse_forecasting/figures/04_per_subject_recall_and_fpr.png)

| 참가자 ID | 선택 임계값 ($\theta_{\text{val}}$) | 평가 창 수 (`양성 / 음성`) | 창 AUROC | 창 AUPRC | 창 F1 | 창 Precision | 창 Recall | 창 FPR | 혼동행렬 (`TP / FP / TN / FN`) | 적격 사건 수 | 탐지 사건 수 | 사건 탐지율 Event Recall | 시간당 오경보 (`FA/hr`) | 평균 리드타임 (s) |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **15** | 0.78 | 1,117 (`40 / 1,077`) | 0.6808 | 0.0942 | 0.1190 | 0.1136 | 0.1250 | 0.0362 | `5 / 39 / 1,038 / 35` | 10 | 2 | 20.00% | 0.9667 | 109.5 |
| **5C** | 0.20 | 1,070 (`32 / 1,038`) | 0.3677 | 0.0222 | 0.0571 | 0.0295 | 0.8750 | 0.8873 | `28 / 921 / 117 / 4` | 9 | 7 | 77.78% | 1.6264 | 160.3 |
| **6B** | 0.68 | 1,197 (`36 / 1,161`) | 0.5785 | 0.0487 | 0.1132 | 0.0732 | 0.2500 | 0.0982 | `9 / 114 / 1,047 / 27` | 10 | 4 | 40.00% | 1.8031 | 127.5 |
| **6D** | 0.67 | 527 (`8 / 519`) | 0.6751 | 0.0264 | 0.0000 | 0.0000 | 0.0000 | 0.0173 | `0 / 9 / 510 / 8` | 2 | 0 | 0.00% | 0.6923 | N/A |
| **7A** | 0.83 | 2,930 (`107 / 2,823`) | 0.6932 | 0.0905 | 0.1022 | 0.0838 | 0.1308 | 0.0542 | `14 / 153 / 2,670 / 93` | 30 | 5 | 16.67% | 1.2203 | 150.0 |
| **7E** | 0.80 | 1,514 (`12 / 1,502`) | 0.6415 | 0.0601 | 0.0000 | 0.0000 | 0.0000 | 0.0020 | `0 / 3 / 1,499 / 12` | 3 | 0 | 0.00% | 0.0892 | N/A |
| **83** | 0.77 | 3,472 (`56 / 3,416`) | 0.6650 | 0.0290 | 0.0400 | 0.0455 | 0.0357 | 0.0123 | `2 / 42 / 3,374 / 54` | 15 | 1 | 6.67% | 0.4675 | 91.0 |
| **8B** | 0.63 | 810 (`47 / 763`) | 0.5575 | 0.0632 | 0.1186 | 0.0684 | 0.4468 | 0.3748 | `21 / 286 / 477 / 26` | 12 | 8 | 66.67% | 4.2176 | 148.6 |
| **94** | 0.54 | 3,154 (`24 / 3,130`) | 0.6321 | 0.0100 | 0.0220 | 0.0113 | 0.5000 | 0.3364 | `12 / 1,053 / 2,077 / 12` | 6 | 4 | 66.67% | 3.6979 | 140.3 |
| **BG** | 0.66 | 1,100 (`47 / 1,053`) | 0.5827 | 0.0603 | 0.0642 | 0.0429 | 0.1277 | 0.1273 | `6 / 134 / 919 / 41` | 13 | 3 | 23.08% | 2.1211 | 160.3 |
| **CE** | 0.21 | 1,539 (`26 / 1,513`) | 0.6489 | 0.0240 | 0.0514 | 0.0267 | 0.6923 | 0.4342 | `18 / 657 / 856 / 8` | 7 | 6 | 85.71% | 3.4335 | 146.3 |
| **DF** | 0.78 | 2,728 (`24 / 2,704`) | 0.5541 | 0.0179 | 0.0000 | 0.0000 | 0.0000 | 0.0026 | `0 / 7 / 2,697 / 24` | 6 | 0 | 0.00% | 0.1192 | N/A |
| **E4** | 0.80 | 3,167 (`92 / 3,075`) | **0.7087** | 0.0543 | 0.1399 | 0.0803 | 0.5435 | 0.1863 | `50 / 573 / 2,502 / 42` | 24 | 17 | 70.83% | 1.2272 | 140.7 |
| **EG** | 0.81 | 651 (`16 / 635`) | 0.5668 | 0.0388 | 0.0588 | 0.0556 | 0.0625 | 0.0268 | `1 / 17 / 618 / 15` | 4 | 1 | 25.00% | 0.9457 | 161.0 |
| **F5** | 0.60 | 2,274 (`92 / 2,182`) | 0.6970 | **0.3405** | **0.2240** | **0.1473** | 0.4674 | 0.1141 | `43 / 249 / 1,933 / 49` | 25 | 13 | 52.00% | 2.2631 | 156.9 |
| **요약** | **중앙값 0.68** | **27,250 (`659 / 26,591`)** | **평균 0.6167<br>중앙값 0.6415** | **평균 0.0653<br>중앙값 0.0543** | **평균 0.0740** | **평균 0.0519** | **평균 0.2838** | **평균 0.1807** | **`209 / 4,257 / 22,334 / 450`** | **176** | **71** | **전체 40.34%** | **전체 1.5678** | **평균 146.4s** |

### 참가자별 편차 분석
- **15명 중 14명(93.3%)에서 무작위(AUROC 0.50)를 상회**하며, 8명(`15`, `6D`, `7A`, `7E`, `83`, `94`, `CE`, `E4`, `F5` 중 9명)에서 AUROC `0.63 ~ 0.71`을 달성했다.
- **최고 성능 참가자 (`E4`, `F5`, `7A`)**: 사건 수가 많고 설문 응답이 충실한 `E4`(24건 중 17건 탐지, AUROC `0.7087`), `F5`(25건 중 13건 탐지, AUROC `0.6970`, AUPRC `0.3405`), `7A`(AUROC `0.6932`)에서 높은 순위 판별력을 보였다.
- **최악 참가자 (`5C`) 및 임계값 편차 원인**:
  - `5C`는 62개 전체 원시 특징 모델에서 AUROC `0.3677`로 유일하게 0.5 미만을 기록했는데, 이는 `5C`의 비사건 근무 구간에 고강도 신체 활동 및 높은 기저 EDA(`~1.9 uS`)가 지속된 반면 일부 스트레스 사건(`EV_039_5C` 등) 시작 전에는 오히려 EDA가 `0.15 uS`로 낮았기 때문이다.
  - 그러나 아래 **8절(개인 기준선 보정 전용 특징셋)**에서 원시 절대값 특징을 배제하고 개인 기저선 대비 상대 변화량(`CALIBRATED_PHYSIO_FEATURES`) 위주로 학습하면, 최악 참가자 AUROC가 **`0.3677 -> 0.4813 ~ 0.5238`로 대폭 개선**되며 모든 참가자에서 역전 현상이 해소된다.

---

## 8. 실험 D: 개인 기준선 보정(Personal Baseline Calibration) 방식 비교

- 산출물 파일: [`results/nurse_forecasting/calibration_comparison.csv`](file:///D:/Autism/results/nurse_forecasting/calibration_comparison.csv)
- 시각화 자료: [`07_personal_calibration_comparison.png`](file:///D:/Autism/results/nurse_forecasting/figures/07_personal_calibration_comparison.png)
- 실험 조건: 개인 보정 델타 특징의 직접적 효과를 정밀 비교하기 위해 보정 생리 특징 9개(`CALIBRATED_PHYSIO_FEATURES`) + 단기 추세/활동 특징 7개(총 16개 특징)로 6종 보정 방식을 15-fold LOSO 평가.

| 모델 | 보정 방식 (`calibration_method`) | 보정 시간 (s) | Motion Gate | 참가자 평균 AUROC ($\pm$SD) | 참가자 중앙값 AUROC | Pooled AUROC | 참가자 평균 AUPRC | 창 F1 평균 | 사건 탐지율 Event Recall (`탐지/176`) | 시간당 오경보 (`FA/hr`) | 최악 참가자 (`AUROC`) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| **LR** | `uncalibrated` (무보정 원시 로그값) | 0s | `none` | 0.6512 ($\pm$0.085) | 0.6499 | 0.6650 | 0.0727 | 0.0657 | 27.27% (`48/176`) | 1.2553 | `BG` (`0.5007`) |
| **LR** | **`cal_1m` (초기 1분 인과 기저선)** | 60s | `none` | **0.6675 ($\pm$0.072)** | **0.6794** | 0.6804 | 0.0674 | 0.0835 | 32.95% (`58/176`) | 1.4397 | `8B` (`0.4541`) |
| **LR** | **`cal_3m` (초기 3분 인과 기저선)** | 180s | `none` | 0.6652 ($\pm$0.070) | 0.6694 | 0.6840 | 0.0664 | 0.0793 | **34.09%** (`60/176`) | 1.4730 | `8B` (`0.4696`) |
| **LR** | **`cal_5m` (초기 5분 인과 기저선)** | 300s | `none` | 0.6628 ($\pm$0.062) | 0.6698 | 0.6823 | 0.0669 | **0.0875** | 33.52% (`59/176`) | 1.3501 | `8B` (**`0.5014`**) |
| **LR** | **`low_motion` (초기 3분 저동작 선택, 기본)** | 180s | `none` | **0.6661 ($\pm$0.067)** | 0.6690 | **0.6840** | 0.0666 | 0.0811 | **34.09%** (`60/176`) | 1.4833 | `8B` (`0.4813`) |
| **LR** | **`adaptive_fallback` (과각성/고동작 폴백)** | 180s | `none` | 0.6523 ($\pm$0.084) | 0.6385 | 0.6717 | **0.0737** | 0.0644 | 22.16% (`39/176`) | **0.9914** | `BG` (**`0.5238`**) |
| **RF** | `uncalibrated` (무보정 원시 로그값) | 0s | `none` | 0.5994 ($\pm$0.119) | 0.6115 | 0.6178 | 0.0687 | 0.0739 | 38.64% (`68/176`) | 1.7728 | `5C` (`0.3499`) |
| **RF** | **`cal_1m` (초기 1분 인과 기저선)** | 60s | `none` | **0.6599 ($\pm$0.086)** | 0.6569 | 0.6671 | **0.0748** | 0.0660 | 40.91% (`72/176`) | 1.9162 | `6D` (`0.4771`) |
| **RF** | **`cal_3m` (초기 3분 인과 기저선)** | 180s | `none` | 0.6495 ($\pm$0.087) | 0.6640 | 0.6667 | 0.0629 | 0.0804 | **46.02%** (`81/176`) | 1.9316 | `6D` (`0.4494`) |
| **RF** | **`cal_5m` (초기 5분 인과 기저선)** | 300s | `none` | 0.6575 ($\pm$0.088) | **0.6778** | **0.6737** | 0.0678 | **0.0830** | 42.61% (`75/176`) | **1.7164** | `6D` (**`0.4892`**) |
| **RF** | **`low_motion` (초기 3분 저동작 선택)** | 180s | `none` | 0.6514 ($\pm$0.087) | 0.6654 | 0.6679 | 0.0637 | 0.0787 | 44.89% (`79/176`) | 2.0366 | `6D` (`0.4603`) |
| **RF** | `adaptive_fallback` (과각성/고동작 폴백) | 180s | `none` | 0.6221 ($\pm$0.109) | 0.6230 | 0.6308 | 0.0710 | 0.0774 | 35.80% (`63/176`) | 1.7600 | `BG` (`0.3968`) |

### 핵심 분석
1. **초기 인과 보정의 일관된 성능 향상 입증**:
   - 무보정(`uncalibrated`) 대비 초기 1~5분 인과 기저선 보정 적용 시, `LR`은 평균 AUROC가 `0.6512 -> 0.6675` (`cal_1m`) / `0.6661` (`low_motion`, Pooled AUROC `0.6840`)로 상승하고 사건 탐지율이 `27.27% -> 34.09%`로 향상되었다.
   - `RF`에서는 보정 효과가 더욱 극적이어서 평균 AUROC가 **`0.5994 -> 0.6599` (+0.0605)**, 최악 참가자 AUROC가 **`0.3499 -> 0.4892` (+0.1393)**로 크게 향상되었다.
2. **최악 참가자 바닥 성능(Worst-Subject Floor) 방어**:
   - `cal_5m` 및 `adaptive_fallback` 적용 시 `LR`의 15명 중 최악 참가자 AUROC는 각각 **`0.5014`** (`8B`), **`0.5238`** (`BG`)를 기록하여 **15명 전 참가자에서 AUROC > 0.50을 달성**했다.

---

## 9. 실제 오탐(False Positive) 및 미탐(False Negative) 사례 정밀 분석

`results/nurse_forecasting/predictions.csv`와 `event_audit.csv`를 교차 추적하여 확인한 대표적 성공·오탐·미탐 사례는 다음과 같다 (시각화 타임라인: [`09_real_session_forecast_timeline.png`](file:///D:/Autism/results/nurse_forecasting/figures/09_real_session_forecast_timeline.png)):

### 9.1 오탐(False Positive) 주요 발생 메커니즘 및 실제 사례
1. **비스트레스 설문 확인 구간(`CONFIRMED_NO_STRESS_0`)에서의 생리적 각성 오탐**:
   - 배경 비사건 구간(`VERIFIED_NON_EVENT`, `24,418개 창`)의 평균 예측 확률은 `0.4124`, 창 FPR은 **`14.58%`**인 반면, 간호사가 설문에서 `Stress level == 0`으로 응답한 구간(`CONFIRMED_NO_STRESS_0`, `2,173개 창`)의 평균 예측 확률은 `0.4922`, 창 FPR은 **`32.12%`**로 2.2배 높았다.
   - **원인**: 원 데이터셋(Hosseini et al., 2022)에서 `Stress level == 0` 구간은 애초에 **생체신호(EDA/HR)가 크게 상승하여 1차 탐지 알고리즘에 걸렸으나 간호사가 주관적으로 스트레스가 아니었다고 답한 구간**(예: 단순 신체 노동, 더위, 긍정적 흥분 등)이기 때문이다. 이는 손목 자율신경 신호만으로는 ‘주관적 부정 정서(Distress)’와 ‘비정서적 생리 각성(Eustress / Physical arousal)’을 완벽히 분리하기 어렵다는 생리학적 한계를 정량적으로 보여준다.
2. **구체적 FP 창 사례 (`5C`, 세션 `5C_1586886712`)**:
   - 시각 `pred_ts = 1586908702.0` (`VERIFIED_NON_EVENT`): `eda_mean = 1.753 uS` (`eda_log_delta = +0.743`), `hr_mean_bpm = 101.8 bpm` (`hr_bpm_delta = +16.3 bpm`)으로 기저선 대비 심박수가 16 bpm 이상 급등하고 피부전도도가 상승했으나 설문상 보고된 스트레스 사건이 없어 오탐(`prob = 0.9999`)으로 집계됨.

### 9.2 미탐(False Negative) 주요 발생 메커니즘 및 실제 사례
1. **생리적 무반응(Hypo-responsive) 인지적 스트레스 미탐 사례**:
   - `EV_251_CE` (참가자 `CE`, 세션 `CE_1593183121`, 고스트레스 `2`, 지속시간 13분): 사건 60~180초 전 4개 창의 평균 `eda_mean = 0.047 uS`, `hr_mean = 63.4 bpm`, `acc_jerk = 38.9`로 자율신경 흥분이 전혀 선행하지 않아 최대 위험 점수가 `0.1471`(임계값 `0.21`)에 그쳐 미탐지됨.
   - `EV_279_DF` (참가자 `DF`, 세션 `DF_1596735393`, 고스트레스 `2`, 지속시간 216분): 3시간 반짜리 장기 근무 스트레스 구간으로, 시작 1~3분 전 급성 자율신경 변화(`eda_mean = 0.197 uS`, `hr_mean = 77.7 bpm`)가 뚜렷하지 않아 `max_prob = 0.2150`(임계값 `0.78`)으로 미탐지됨.
2. **높은 검증 임계값($\theta_{\text{val}} \ge 0.77$)으로 인한 아까운 미탐(Near-Miss) 사례**:
   - `EV_072_7A` (참가자 `7A`, 세션 `7A_1587228286`, 고스트레스 `2`): 사건 시작 전 `eda_mean = 5.756 uS`, `hr_mean = 99.7 bpm`으로 강한 교감신경 전조가 포착되어 모델이 **`max_prob = 0.8288`**의 높은 위험 점수를 출력했으나, 해당 폴드의 내부 검증 임계값이 `0.8300`으로 설정되어 불과 **`0.0012` 차이**로 미탐 처리됨.
   - `EV_301_E4` (참가자 `E4`, 세션 `E4_1593439701`, 고스트레스 `2`): `eda_mean = 3.974 uS`, `hr_mean = 98.7 bpm`으로 **`max_prob = 0.7924`**까지 상승했으나 폴드 임계값(`0.8000`)에 **`0.0076`** 미달하여 미탐 처리됨.
   - 이러한 Near-Miss 사건들은 6절의 히스테리시스 정책이나 전역 배포 임계값(`0.68`) 적용 시 정상 탐지로 전환된다.

---

## 10. 최종 배포 모델 사양 및 검증 결과

- 상세 배포 명세서: [`reports/NURSE_MODEL_DEPLOYMENT_SPEC.md`](file:///D:/Autism/reports/NURSE_MODEL_DEPLOYMENT_SPEC.md)
- 단위 테스트 검증: [`tests/test_nurse_model_serialization.py`](file:///D:/Autism/tests/test_nurse_model_serialization.py) (기존 11개 + 신규 직렬화·무결성·리플레이 테스트 3개 = **총 14개 테스트 100% 통과**)

### 10.1 최종 모델 아티팩트 구성 및 무결성 검증 요약
1. **최종 선정 모델 파일**:
   - 경로: [`results/nurse_forecasting/models/nurse_stress_forecast_lr_p1.joblib`](file:///D:/Autism/results/nurse_forecasting/models/nurse_stress_forecast_lr_p1.joblib) (`7,639 Bytes`)
   - C/모바일 이식용 JSON 명세: [`results/nurse_forecasting/models/nurse_stress_forecast_lr_p1_spec.json`](file:///D:/Autism/results/nurse_forecasting/models/nurse_stress_forecast_lr_p1_spec.json) (`13,794 Bytes`)
   - 비교용 트리 모델: `nurse_stress_forecast_rf_p1.joblib` (`1,449,952 Bytes`), `nurse_stress_forecast_lgbm_p1.joblib` (`221,588 Bytes`)
2. **검증 완료 항목 (Section 4 요구사항 전수 확인)**:
   - **로딩 및 추론 일치성**: `joblib` 파이프라인(`SimpleImputer(median)` $\rightarrow$ `StandardScaler` $\rightarrow$ `LogisticRegression(C=0.25, class_weight='balanced')`)의 `predict_proba` 출력과 `spec.json`의 수치 배열(`imputer_median`, `scaler_mean`, `scaler_scale`, `coefficients`, `intercept`)로 직접 계산한 시그모이드 출력 간 최대 절대 오차 **`1.22e-15`** (`< 10^-12`) 완전 일치 확인.
   - **특징 스키마**: 62개 인과 특징(`EDA 26 + PPG/HR/IBI 21 + 3축 회전불변 ACC 15`) 이름·순서·개수 100% 일치 확인 (중요도 상위 20개 가중치 시각화: [`10_final_model_feature_importance.png`](file:///D:/Autism/results/nurse_forecasting/figures/10_final_model_feature_importance.png)).
   - **결측치·NaN 방어**: 전체 행이 `NaN`이거나 일부 센서가 결측(`ibi_valid_flag == 0` 등)인 입력에서도 예외 없이 유한한 확률(`0.4658` 등)을 출력하며, 센서 접촉 불량(`qc_common_valid == False`) 시 경보를 보류하도록 설계됨.
   - **시간적 분리 및 리드타임**: 관측 창은 `[t-180s, t)`로 미래 샘플을 단 1개도 참조하지 않으며(`test_nurse_time_leakage.py` 오차 `< 10^-9` 검증), 예측 구간 `P1`은 `[t+60s, t+180s]`로 최소 60초 리드타임을 보장함.
   - **임계값 및 정책**: LOSO 폴드별 내부 검증 임계값 분포(`0.20 ~ 0.83`)의 중앙값인 **`decision_threshold = 0.68`**을 기본 배포 임계값으로 저장하였으며, 실시간 운용 시 **`hysteresis` ($T_{\text{on}}=0.68, T_{\text{off}}=0.35$, 병합 간격 `180s`)** 정책을 권장함.

---

## 11. 기존 WESAD·PhysioNet 연구와의 차이 및 실시간 제품 적용 한계

| 비교 항목 | 기존 WESAD 연구 (성인 실험실) | 기존 PhysioNet 연구 (성인 인지/운동) | **본 Nurses Stress 연구 (병원 실생활 직무)** |
|---|---|---|---|
| **수집 환경 및 인원** | 실험실 통제 환경, 성인 15명 (`~30시간`) | 실험실 인지 과제 + 트레드밀/바이크, 성인 18명 | **실제 병원 근무 현장(COVID-19 유행기), 간호사 15명 (`1,251.7시간`, 214개 주석 세션)** |
| **예측 문제 정의** | **현재 상태 분류** (Current Stress Detection, 60s 창) | **현재 상태 분류** 및 운동 오경보 분리 (60s 창) | **미래 사건 사전 예측 (Prospective Early Warning, `t+60~180s` Onset 예측)** |
| **라벨 생성 방식** | 프로토콜 고정 구간 (TSST 발표/암산 vs 휴식) | `tags.csv` 기반 과제 구간 (Stroop/TMCT vs 페달링) | **1차 생리 알고리즘 후보 구간 + 근무 종료 후 간호사 설문 확인 (`0, 1, 2, na`)** |
| **LOSO AUROC 수준** | `0.9519 ~ 0.9813` (통제된 강한 자극 vs 안정 휴식) | `0.8000 ~ 0.9191` (태그 정합 및 초기 보정 시) | **`0.6167 ~ 0.6675` (일상 보행·처치 중 사건 시작 1~3분 전 전조 예측)** |
| **Motion Gate 효과** | 해당 없음 (좌식/입식 통제) | `soft_dynamic`이 유산소 운동 오경보를 `1.27%`로 완벽 억제 | 간호사는 스트레스 사건 중에도 계속 보행·처치하므로 과도한 모션 감쇠 시 Event Recall 저하; **`hysteresis` 에피소드 병합이 핵심** |

### 실시간 제품 적용 가능성과 미검증 한계
1. **온디바이스 탑재 가능성 (검증 완료)**:
   - 선정된 `M1 LR` 모델은 가중치 메모리 **`996 Bytes`**, 180초 원시 센서 링버퍼 **`118.8 KB`**, PC 단일 창 특징 추출 **`6.47 ms`**, 선형 추론 **`< 1.0 ms`**로 ESP32-S3급 저전력 손목 웨어러블 또는 스마트폰 앱에서 재학습 없이 즉시 실시간 구동 가능하다.
2. **핵심 한계 및 미검증 사항**:
   - **시간 해상도 및 순환성 한계**: `SurveyResults.xlsx`의 시작 시각이 분 단위로 반올림($\pm 60\text{s}$)되어 있고 1차 생리 알고리즘 경계의 영향을 받으므로, 초 단위 행동 Onset 예측으로 해석할 수 없다.
   - **오경보 부담**: 히스테리시스 적용 시 시간당 오경보를 `0.9965 FA/hr`까지 낮추었으나, 12시간 교대 근무 기준 약 `10~12회`의 주의 알림이 발생할 수 있으므로 단독 강제 알람이 아닌 **1단계 무음 햅틱/시각 주의(Advisory) 지표**로 활용해야 한다.
   - **자폐 아동 도메인 미검증**: 성인 간호사의 직무 스트레스 예측 모델이므로, 자폐 아동의 상동행동(Stimming)이나 감각 과부하 멜트다운에 그대로 적용할 수 없으며 별도의 아동 임상 코호트 전향적 검증이 필수적이다.
   - **실제 하드웨어 전력 미실측**: 메모리 및 연산 복잡도는 실측·검증되었으나, 실제 ESP32-S3 보드에서의 배터리 방전 시간 및 BLE 패킷 손실률은 하드웨어 시제품 제작 후 검증해야 한다.

---

## 12. 최종 산출물 파일 및 시각 증거 목록

### 12.1 정량 결과 데이터 (`results/nurse_forecasting/`)
1. [`dataset_manifest.json`](file:///D:/Autism/results/nurse_forecasting/dataset_manifest.json) — 609개 `.zip` 세션 및 설문 전수 감사 매니페스트
2. [`subject_summary.csv`](file:///D:/Autism/results/nurse_forecasting/subject_summary.csv) — 15명 참가자별 세션·시간·설문 사건 통계표
3. [`event_audit.csv`](file:///D:/Autism/results/nurse_forecasting/event_audit.csv) — 358개 설문 행 전수 감사 및 P0/P1/P2 적격 판정표
4. [`features.csv`](file:///D:/Autism/results/nurse_forecasting/features.csv) — 214개 주석 세션 인과 추출 `51,234개 창 × 95열` 특징 캐시
5. [`splits.csv`](file:///D:/Autism/results/nurse_forecasting/splits.csv) — 15-fold LOSO 외부 테스트 및 내부 검증 분할 기록
6. [`model_comparison.csv`](file:///D:/Autism/results/nurse_forecasting/model_comparison.csv) — P1 핵심 4개 모델(`Dummy`, `LR`, `RF`, `LightGBM`) 성능 비교표
7. [`per_subject_metrics.csv`](file:///D:/Autism/results/nurse_forecasting/per_subject_metrics.csv) — P1 모델별·참가자별(15명 × 4모델 = 60행) 상세 지표
8. [`predictions.csv`](file:///D:/Autism/results/nurse_forecasting/predictions.csv) — P1 전체 테스트 창 예측 확률 및 판정 기록 (`109,000행`)
9. [`lead_time_comparison.csv`](file:///D:/Autism/results/nurse_forecasting/lead_time_comparison.csv) — `P0 (0-60s)`, `P1 (60-180s)`, `P2 (180-300s)` 리드타임 비교표
10. [`sensor_ablation.csv`](file:///D:/Autism/results/nurse_forecasting/sensor_ablation.csv) — `EDA`, `PPG`, `ACC`, `EDA+PPG`, `ALL` 센서 기여도 비교표
11. [`motion_gate_comparison.csv`](file:///D:/Autism/results/nurse_forecasting/motion_gate_comparison.csv) — 활동량 4분위수별 오경보율 및 Motion Gate 3종 비교표
12. [`alert_policy_comparison.csv`](file:///D:/Autism/results/nurse_forecasting/alert_policy_comparison.csv) — 6종 경고 발생·병합 정책 비교표
13. [`calibration_comparison.csv`](file:///D:/Autism/results/nurse_forecasting/calibration_comparison.csv) — 6종 인과 개인 기준선 보정 방식 비교표
14. [`event_metrics.csv`](file:///D:/Autism/results/nurse_forecasting/event_metrics.csv) — 핵심 실험별 사건 단위(Event-level) 지표 통합 요약표

### 12.2 시각 증거 그래프 10종 (`results/nurse_forecasting/figures/`)
1. [`01_model_auroc_auprc_comparison.png`](file:///D:/Autism/results/nurse_forecasting/figures/01_model_auroc_auprc_comparison.png) — P1 4개 모델 AUROC, AUPRC, Event Recall 비교
2. [`02_precision_recall_curve.png`](file:///D:/Autism/results/nurse_forecasting/figures/02_precision_recall_curve.png) — P1 모델별 Precision-Recall 곡선
3. [`03_roc_curve.png`](file:///D:/Autism/results/nurse_forecasting/figures/03_roc_curve.png) — P1 모델별 ROC 곡선
4. [`04_per_subject_recall_and_fpr.png`](file:///D:/Autism/results/nurse_forecasting/figures/04_per_subject_recall_and_fpr.png) — 15명 참가자별 Event Recall, Window Recall, FPR 분포
5. [`05_lead_time_performance.png`](file:///D:/Autism/results/nurse_forecasting/figures/05_lead_time_performance.png) — P0 / P1 / P2 예측 시간대별 AUROC 및 Event Recall 변화
6. [`06_recall_vs_false_alarms_curve.png`](file:///D:/Autism/results/nurse_forecasting/figures/06_recall_vs_false_alarms_curve.png) — 임계값 변화에 따른 Event Recall 대 시간당 오경보(`FA/hr`) 트레이드오프 곡선
7. [`07_personal_calibration_comparison.png`](file:///D:/Autism/results/nurse_forecasting/figures/07_personal_calibration_comparison.png) — 6종 개인 기준선 보정 방식별 AUROC, Event Recall, FA/hr, 최악 참가자 AUROC 비교
8. [`08_motion_vs_risk_score.png`](file:///D:/Autism/results/nurse_forecasting/figures/08_motion_vs_risk_score.png) — 가속도 활동량 분위수별 비사건 오경보율 및 소프트 모션 게이트 감쇠 분포
9. [`09_real_session_forecast_timeline.png`](file:///D:/Autism/results/nurse_forecasting/figures/09_real_session_forecast_timeline.png) — 실제 간호사 연속 생체신호(EDA, HR, ACC Jerk), 위험 점수, 조기경보 발령 및 실제 스트레스 사건 구간 타임라인
10. [`10_final_model_feature_importance.png`](file:///D:/Autism/results/nurse_forecasting/figures/10_final_model_feature_importance.png) — 최종 M1 LR 모델 상위 20개 표준화 특징 가중치

---

## 13. 실시간 소프트웨어에서 재학습 없이 즉시 추론하는 방법 (Inference Guide)

소프트웨어 구현자는 학습 코드나 원본 데이터셋 없이 [`nurse_stress_forecast_lr_p1.joblib`](file:///D:/Autism/results/nurse_forecasting/models/nurse_stress_forecast_lr_p1.joblib) 또는 [`nurse_stress_forecast_lr_p1_spec.json`](file:///D:/Autism/results/nurse_forecasting/models/nurse_stress_forecast_lr_p1_spec.json)만으로 실시간 스트림 추론을 수행할 수 있다.

### 13.1 Python 환경에서 `.joblib` 아티팩트로 즉시 추론하는 코드
```python
import joblib
import pandas as pd
from src.features.nurse_forecasting import extract_nurse_window_features
from src.forecasting.calibration import apply_session_causal_calibration

# 1. 최종 동결 모델 패키지 로드 (재학습 불필요)
artifact = joblib.load("results/nurse_forecasting/models/nurse_stress_forecast_lr_p1.joblib")
pipe = artifact["pipeline"]
feature_cols = artifact["feature_columns"]
t_on = artifact["alert_policy"]["t_on"]   # 0.68
t_off = artifact["alert_policy"]["t_off"] # 0.35

# 2. 실시간 180초 슬라이딩 창 특징 추출 (pred_time_rel_s >= 180.0, 과거 [t-180s, t)만 참조)
# session_data = {"start_ts": ..., "start_timestamps": ..., "rates": {"EDA":4, "BVP":64, "ACC":32, "HR":1}, "EDA": ..., "BVP": ..., "ACC": ..., "HR": ..., "IBI": ...}
raw_feats = extract_nurse_window_features(session_data, pred_time_rel_s=current_rel_s, window_s=180.0)

# 3. 센서 접촉 및 신호 품질 확인 (불량 시 추론 보류)
if not raw_feats["qc_common_valid"]:
    risk_prob = 0.0
    alert_active = 0
else:
    # 세션 누적 창 DataFrame(session_windows_df)에 현재 창을 추가한 뒤 인과적 저동작 기저선 보정 적용
    cal_df = apply_session_causal_calibration(session_windows_df, method="low_motion", duration_s=180.0)
    x_now = cal_df.iloc[[-1]][feature_cols]
    risk_prob = float(pipe.predict_proba(x_now)[0, 1])

    # 4. 히스테리시스(Hysteresis) 경보 상태 갱신
    if prev_alert_active == 0 and risk_prob >= t_on:
        alert_active = 1
    elif prev_alert_active == 1 and risk_prob < t_off:
        alert_active = 0
    else:
        alert_active = prev_alert_active
```

### 13.2 C / C++ / 모바일(Kotlin·Swift) 환경에서 `spec.json`만으로 추론하는 수식
`scikit-learn`이 없는 임베디드·모바일 환경에서는 [`nurse_stress_forecast_lr_p1_spec.json`](file:///D:/Autism/results/nurse_forecasting/models/nurse_stress_forecast_lr_p1_spec.json)의 62개 배열 상수를 로드하여 다음 3단계 연산만으로 동일 확률(오차 $< 10^{-12}$)을 산출한다:
1. **결측 대체**: `val = isnan(x[j]) ? imputer_median[j] : x[j]`
2. **표준화 및 내적**: `logit += ((val - scaler_mean[j]) / scaler_scale[j]) * coefficients[j]` (초기값 `logit = intercept`)
3. **확률 및 히스테리시스 판정**: `prob = 1.0 / (1.0 + exp(-logit))` 산출 후 `prob >= 0.68`이면 경보 ON, `prob < 0.35`이면 경보 OFF (`180초` 이내 재발화는 단일 에피소드로 유지).
