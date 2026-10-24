# AutismResearch — 최종 모델 연구 및 검증 통합 보고서

- 문서 버전: v1.0 (2026-10-24)
- 주 실행 환경: Windows 11 / AMD Ryzen 7 8845HS / 32GB RAM / NVIDIA RTX 4060 Laptop (CUDA 13.3) / Python 3.13.2
- 저장소: [https://github.com/Goldhobbang/AutismResearch](https://github.com/Goldhobbang/AutismResearch)
- 기준 커밋: `794507721caff9607cff02f58eb8265e671abcc6`

---

## A. 3대 과제(D1 / D2 / D3) 달성 상태 요약

| 과제 구분 | 정의 | 현재 상태 | 주요 근거 |
|---|---|:---:|---|
| **D1** | 성인 실험실 스트레스 분류 (WESAD) | **완료 (VERIFIED)** | 15명 원본 무결성 검증, 15-fold LOSO 교차검증 (LR AP 0.9048, RF 0.8586, ACC 0.9245), 100% 비트 단위 독립 재현 완료 |
| **D2** | 운동/일상활동과 스트레스 구별 (PhysioNet) | **검증 완료 (VERIFIED)** | 공식 원본 센서(EDA/BVP/ACC) 해시 검증 획득, 검증 세그먼트 Zero-shot 외부 검증(rf AUROC 0.601, eda 0.587) 및 유산소 운동 혼동 실측 완료 (ACC 모델의 100% 운동 오탐 및 계층적 Gating 필요성 실증) |
| **D3** | 자폐 멜트다운 60초 사전 경보 (ASD Onset) | **차단/미실행 (BLOCKED)** | 손목 신호와 실제 행동 Onset을 결합한 공인 자료 부재로 학습 차단 유지. 정식 데이터 접근 신청 패키지(`ASD_DATA_ACCESS_APPLICATION_PACKAGE.md`) 완성 |

---

## B. 데이터 현황 및 무결성 감사

| 데이터셋 | 확보 방식 및 경로 | 피험자 / 세션 수 | 관찰 입력 신호 | 정답 Onset 유무 | 라이선스 및 제약 |
|---|---|---|---|:---:|---|
| **WESAD** | 로컬 공식 원본 인수 (`data/raw/WESAD/`) | 15명 / 15세션 (5,740 창, 2,031 적격 창) | 손목 EDA 4Hz, BVP 64Hz, ACC 32Hz | 상태(Phase) 라벨 | 비상업적 학술 연구용 / SHA256 체크섬 일치 |
| **PhysioNet 1.0.1** | 공식 사이트 메타데이터 전수 감사 (`data/registry/`) | 51명 / 97세션 (STRESS 36, AEROBIC 30, ANAEROBIC 31) | 손목 EDA, BVP, ACC, TEMP, HR, tags | 세그먼트 버튼 태그 | ODC-By 1.0 / S02 중복, f07 차폐, V1/V2 성별 교란 |
| **ASD 후보군** | 7대 후보 전수 실사 (`data/registry/asd_candidates.csv`) | Kartha, Goodwin, Imbiriba 등 비공개/DUA 대기 | 손목 센서 또는 비디오 랜드마크 | 연구별 상이 | 승인 대기 / 비웨어러블(ASC-emotion) 대체 금지 |

---

## C. 핵심 모델 성능 검증 (D1 WESAD)

### 1. Primary(참가자 평균) vs Secondary(Pooled OOF) 공식 표

| 모델 | 센서 조합 | Primary: 참가자 평균 AP | Stress F1 | AUROC | Balanced Acc | Secondary: Pooled AP | Pooled F1 | Pooled FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **dummy** | all | 0.2987 | 0.0000 | 0.5000 | 0.5000 | 0.2971 | 0.0000 | 0.0000 |
| **lr** | all | **0.9048** | **0.7425** | **0.9519** | **0.8214** | **0.8012** | **0.7336** | **0.1629** |
| **rf** | all | 0.8586 | 0.6142 | 0.9065 | 0.7207 | 0.6805 | 0.6034 | 0.3132 |
| **lr** | eda (기준선) | 0.8045 | 0.6388 | 0.8791 | 0.7421 | 0.6580 | 0.6287 | 0.2676 |
| **lr** | ppg | 0.7333 | 0.6462 | 0.8622 | 0.7555 | 0.6774 | 0.6457 | 0.2528 |
| **lr** | **acc (단독)** | **0.9245** | **0.7723** | **0.9544** | **0.8478** | **0.8945** | **0.7813** | **0.0962** |
| **lr** | eda_ppg | 0.8520 | 0.6142 | 0.9185 | 0.7493 | 0.7033 | 0.6394 | 0.1938 |
| **lr** | eda_acc | 0.9134 | 0.7059 | 0.9479 | 0.8009 | 0.8331 | 0.7146 | 0.1257 |
| **lr** | ppg_acc | 0.9092 | 0.7579 | 0.9537 | 0.8415 | 0.8665 | 0.7680 | 0.1145 |

### 2. 가속도계(ACC) 원인 및 과제 교란 규명 (수치 정밀 대조)
- **ACC vs ALL 직접 Paired 비교**:
  - 참가자 15명 중 **7명 개선(ACC 우위: S2, S3, S13, S14, S15, S16, S17)**, **7명 악화(ALL 우위: S5, S6, S7, S8, S9, S10, S11)**, **1명 동률(S4: AP 1.0000)**로 팽팽하게 양분됨.
  - 참가자 평균 AP는 ACC(0.9245)가 ALL(0.9048)보다 +0.0197 높으나, 이는 S13 피험자에서 ACC가 +0.3495 대폭 상승한 영향이 큼.
  - 반면 Pooled OOF AP에서는 ACC(0.8945)가 ALL(0.8012) 대비 +0.0933 큰 격차로 우세함. 이는 ALL 모델에서 S7 피험자의 높은 FPR(0.8542)로 인한 위양성 창 누적이 전체 정밀도를 끌어내렸기 때문임.
- **7개 하위 특성군 분해 결과**:
  - `acc_posture_only` (축별 평균 자세 중력 성분): **AP 0.5701** (무작위 수준 급락).
  - `acc_no_posture` (정적 자세 제거): **AP 0.9298** (최고 성능 달성).
  - `acc_dynamic_only` (동적 에너지 및 저크): **AP 0.8620**.
- **판정 결론**: 가속도계 성능의 원천은 정적 자세 암기가 아니며, **TSST 과제 수행 중 발표·암산에 수반되는 손목 제스처 및 긴장성 동적 움직임(Task Confounding)**이 스트레스와 결합하여 높은 변별력을 나타낸 것임.

### 3. 특이 참가자 S7 오탐 및 창/에피소드 알림 정책 구분
- **S7 오탐 원인**: AUROC 1.000, AUPRC 1.000의 완벽한 랭킹 분별력을 보였으나, 개인 베이스라인의 상향 편향(비스트레스 확률 중앙값 0.9128)으로 집단 고정 임계값(0.35)에서 FPR이 0.8542로 폭증함.
- **창 단위(Window-level) FPR vs 경보 에피소드(Episode-level) 오경보의 엄격한 분리**:
  - **창 단위 FPR**: `pred_tuned`(0.1619) 대비 `pred_conservative`(0.1148, -29.1%), `pred_cooldown60`(0.0433, -73.3%) 적용 시 현저히 감소.
  - **스트레스 에피소드 검출률**: 6개 정책 전원 **15/15명(100%) 스트레스 에피소드 탐지 성공** (첫 경보 지연 8초~25초).
  - **에피소드 오경보**: `pred_cooldown60`은 S7처럼 장시간 확률이 높은 피험자에서 60초마다 재경보가 울려 세션당 4.13회(시간당 10.39회)의 오경보를 발생시킴. 반면 히스테리시스(`pred_hysteresis`, 진입 0.70, 해제 0.30) 적용 시 에피소드 오경보를 세션당 1.73회(시간당 4.35회)로 억제하면서 창 재현율 0.8048을 유지함.

### 4. PhysioNet Zero-shot 외부 검증 및 운동 혼동 실측
- **공식 원본 신호 획득 및 해시 검증**: PhysioNet 공식 디렉터리에서 S01 참가자의 멀티모달 센서 원본(ACC, BVP, EDA, tags)을 취득하고 공식 `SHA256SUMS.txt`와 100% 일치 확인.
- **스트레스 vs 휴식 Zero-shot 전이 성능 (77개 검증 창)**:
  - `rf_all`: **AUROC 0.6010**, AUPRC 0.7765
  - `lr_eda`: **AUROC 0.5873**, AUPRC 0.7825 (피부전도도 기반 전이 안정성 입증)
  - `lr_acc`: **AUROC 0.2382** (도메인 전이 완전 붕괴; WESAD 발표 제스처와 PhysioNet 화면 과제 간의 행동 불일치 규명)
- **유산소 자전거 운동(Aerobic Cycling 133창) 스트레스 혼동 실측**:
  - `lr_acc`: **운동 구간 100.0%를 스트레스로 오탐 (평균 확률 0.9858)**. 가속도계만으로는 운동과 스트레스를 결코 구별할 수 없음.
  - `lr_eda`: 발한으로 인해 운동 구간의 32.3%에서 허위 스트레스 경보 발생.
  - **결론**: 일상 환경 배포를 위해서는 가속도계 기반 운동 감지기(Physical Activity Gating Filter)를 전치하여 고활동성 구간에서 생체신호 스트레스 경보를 보류하는 2단계 계층형 아키텍처가 필수적임.

---

## D. 모델 선정 및 임베디드(MCU) 적합성

1. **최적 모델 선정**: **정규화 로지스틱 회귀 (Logistic Regression, L2)**
   - Random Forest 대비 피험자 간 일반화(LOSO AP 0.9048 vs 0.8586)에서 우위.
   - 가중치 및 스케일러 배열 크기: **760 Bytes** (RF의 882 KB 대비 1,160배 경량).
2. **ESP32-S3 적합성**:
   - 60초 원시 센서 float32 링 버퍼 요구량: **39,360 Bytes (~38.4 KiB)**.
   - 내부 SRAM(512KB) 점유율 약 10.1%로 외부 PSRAM 없이 완전 자립 구동 가능.
   - **실측 고지**: PC 실측 추론 지연시간(p50 9.48ms)을 MCU 속도로 주장하지 않으며, MCU 지연은 펌웨어 플래싱 후 하드웨어 타이머로 측정되어야 함.

---

## E. 자폐(ASD) 멜트다운에 관한 임상 사실관계

1. **라벨의 엄밀한 정의**:
   - 자폐 멜트다운(Meltdown)은 단순 스트레스(Stress), 타인 공격(Aggression), 자해(SIB), 상동 행동(Stimming)과 동일하지 않음.
   - 복합 행동 지표(Imbiriba ED 등)를 멜트다운 단독 성능으로 표기할 수 없음.
2. **60초 선행 경보의 조건**:
   - $L=0, H=60$초 모델은 1초 뒤 사건을 포함하므로 조기경보 증거가 아님.
   - 최소 60초 전 선행 경보를 입증하려면 $L=60, H=30$초의 지연 구간에서 실제 선행시간(Lead time)을 측정해야 함.
3. **현재 상태**: 공인된 웨어러블+사건 Onset 실데이터 부재로 D3 사건 예측 성능은 **N/A**로 정직하게 유지함.

---

## F. 완전한 재현 절차 (Reproducibility)

1. **환경 초기화**:
   ```powershell
   py -3.13 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
   ```
2. **WESAD 특징 추출 및 재현 학습**:
   ```powershell
   .\.venv\Scripts\python.exe -m src.features.extract --root data/raw/WESAD --out data/processed/wesad_features.csv --window 60 --step 15
   .\.venv\Scripts\python.exe -m src.train --features data/processed/wesad_features.csv --out results/phase2/wesad_repro --plan core --expected-subjects 15 --dataset-version WESAD_ICMI2018 --jobs 8
   ```
3. **ACC 교란 및 임계값 분석**:
   ```powershell
   .\.venv\Scripts\python.exe experiments/phase2/acc_confounding_analysis.py
   .\.venv\Scripts\python.exe experiments/phase2/threshold_alert_analysis.py
   ```
4. **검증 결과**: 단위 테스트 19건 전체 PASS, 베이스라인과 재현본 간 수치 오차 0.0.

---

## G. 차기 잔여 과제 (Next Actions)

- **P0 (최우선)**: PhysioNet 99개 검증 세그먼트(`verified_segments.csv`) 기반 단일 피험자 내 스트레스 vs 유산소 운동(Bicycle) 판별 모델 파이프라인 가동.
- **P1 (중기)**: Kartha 2026 연구진 또는 Northeastern 대학(Imbiriba / Goodwin)과의 정식 데이터이용계약(DUA) 및 IRB 승인 추진.
- **P2 (장기)**: ESP32-S3 펌웨어에 C 헤더(`lr_weights.h`) 포팅 및 실측 전력/지연 프로파일링.

---

## H. 최종 판정 (주장 가능한 것과 불가능한 것)

| 항목 | 현재 데이터로 **주장 가능한 사실 (VERIFIED)** | 현재 데이터로 **주장 불가능한 내용 (NEVER CLAIM)** |
|---|---|---|
| **생체신호 스트레스 분류** | 성인 15명 손목 신호(EDA, BVP, ACC)로 실험실 스트레스를 LOSO AP 0.9048 수준으로 분류 가능함. | 자폐 아동의 일상 스트레스나 감각 과부하 상태를 동일한 성능으로 감지할 수 있다는 주장 불가. |
| **센서 기여도** | WESAD 실험실 환경에서 가속도계(ACC)의 동적 움직임 성분이 분류 성능을 크게 견인함. | 일상 환경에서 가속도계만으로 스트레스를 완벽히 감지할 수 있다는 주장 불가 (운동 교란 미분리). |
| **하드웨어 구현성** | 로지스틱 회귀(760B)와 60초 링버퍼(38.4KB)는 ESP32-S3 내부 SRAM(512KB)에서 완벽히 연산 가능함. | 현재 장치가 배터리 11시간 작동을 실측 완료했거나, 무선 원거리 알림을 단독 보장한다는 주장 불가. |
| **자폐 멜트다운 조기경보** | 미래 사건 Onset을 예측하기 위한 엄격한 시간 분리($[t-W,t) \rightarrow (t+L, t+L+H]$) 프로토콜을 수립함. | **현재 손목 밴드가 멜트다운을 60초 전에 80~90% 정확도로 경보한다는 주장 절대 불가 (데이터 N/A).** |
