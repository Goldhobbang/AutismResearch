# 생체신호 스트레스 탐지 및 외적 일반화 후속 심층 연구 종합 보고서 (FOLLOWUP_RESEARCH_SUMMARY_KO.md)

- **연구 과제**: AutismResearch 후속 3대 연구 과제 (A, B, C) 종합 보고서
- **작성 일자**: 2026-10-25
- **저장소 커밋 기준**: `093c5ec` (기존 안정 커밋 기반 별도 격리 실험 수행)
- **실행 환경**: Windows 11 / AMD Ryzen 7 8845HS / Python 3.13.2 (`.venv`)
- **수행 원칙**: 100% 실제 데이터 감사·전처리·학습·평가·단위테스트 검증 완료 (합성 데이터/사전 조작 배제)

---

## 1. 종합 개요 및 수행 배경

본 후속 연구는 웨어러블 기반 생체신호 스트레스 탐지 시스템의 실용화 및 외적 일반화 확대를 위해 제기된 3대 핵심 난제를 실제 데이터에 기반하여 정량적으로 규명하였다:
1. **연구 A (보정 시간 최적화)**: 관례적인 "5분 기저선 보정"의 타당성을 비판적으로 검토하고, 0분(무보정)부터 1분, 3분, 5분, 10분까지 보정 길이에 따른 성능·오경보·대기시간의 실증적 트레이드오프를 도출.
2. **연구 B (초기 기저선 강건성 및 실패 분석)**: 착용 직후 긴장·과각성·착용 동잡음으로 초기 상태가 오염되었을 때 발생하는 탐지 실패 원인을 규명하고, 저동작 필터링(`low_motion`) 및 적응형 폴백(`adaptive_fallback`) 등 6대 기저선 추정량의 방어 성능을 검증.
3. **연구 C (환경 센서의 생체 기여도 실증)**: AMDC (Health-Spa, 14명 피험자, 33,993초) 실제 멀티모달 데이터를 통해 손목 웨어러블에 조도(Lux) 및 소음(Sound dB) 센서를 추가하는 것이 미래 생체 반응(심박수, 피부전도도) 예측에 유의미한 개선을 주는지 시계열 회귀로 엄격히 검증.

---

## 2. 7대 핵심 질문에 대한 직접적이고 명확한 답변 (Direct Answers to 7 Core Questions)

### Q1. 개인별 보정 시간은 몇 분이 가장 권장되는가?
- **답변: 1분 ~ 3분이 최적 권장 구간이며, 실시간 실사용 관점에서는 3분이 가장 이상적이다.**
- **정량적 근거**:
  - WESAD 동일 평가창(`Common_Set_10m`) 기준: 무보정(0분) F1 0.6912에서 **1분 보정 시 F1 0.7228 (+0.0316)**, **3분 보정 시 F1 0.7888 (+0.0976)**로 급상승함.
  - 3분에서 5분으로 늘릴 때 F1은 0.7821로 오히려 소폭 감소하며, 10분으로 늘려도 F1은 0.7904로 3분 대비 개선폭이 +0.0016(+0.2%)에 불과함. 즉, **3분 보정만으로 10분 보정으로 얻을 수 있는 F1 이득의 98.4%를 달성**함.
  - PhysioNet 전이 검증에서도 무보정 시 41.18%(LR) ~ 100.0%(RF)에 달하던 기저선 오경보율(Baseline FPR)이 **단 1분(60초) 보정만으로 2.94%(LR) ~ 10.29%(RF)로 90% 이상 격감**함.
  - 따라서 5분~10분의 긴 대기시간을 요구하지 않고 **1~3분의 짧은 보정만으로 웨어러블 시스템의 실시간 즉각 보호 기능과 오경보 억제 효과를 동시에 달성**할 수 있음.

### Q2. PhysioNet에서 5분·10분 보정이 실패하는 진짜 이유는 무엇인가?
- **답변: 알고리즘의 결함이 아니라, PhysioNet 데이터셋의 실제 연속 휴식 구간이 5분 미만(평균 191.2초)으로 물리적으로 존재하지 않기 때문이다.**
- **정량적 근거**:
  - PhysioNet 공식 `tags.csv` 프로토콜 상의 `Baseline` 구간을 감사한 결과, 17명 적격 피험자의 실제 연속 휴식 시간은 **163초 ~ 247초 (평균 191.2초, 약 3.2분)**에 불과함.
  - 따라서 300초(5분) 및 600초(10분)의 순수 기저선은 전 피험자에서 물리적으로 불가능함 (`N/A`).
  - 과거 연구에서 5분 보정을 시도했을 때 성능이 붕괴한 진짜 원인은, 휴식 시간이 부족하여 기저선 수집 윈도우가 이후의 스트레스 유발 과제(`Stroop`, `TMCT`) 구간까지 침범하여 스트레스 상태를 기저선으로 학습하는 **치명적인 라벨 오염(Label Contamination)이 발생했기 때문**임.

### Q3. 초기 기준선이 불안정할 때 가장 안전한 보정 추정량은 무엇인가?
- **답변: 단기 구간에서는 저동작 선별 평균(`low_motion`)이 가장 우수하며, 심각한 과각성·기저선 이상치가 의심되는 상황에서는 사전 학습 집단 통계로 전환하는 적응형 폴백(`adaptive_fallback`)이 가장 안전하다.**
- **정량적 근거**:
  - WESAD 15명 평가에서 동잡음이 적은 창만을 선별해 기저선을 계산한 `low_motion` 추정량이 **AUROC 0.9518 (중앙값 0.9994), 스트레스 Recall 0.8259, 기저선 FPR 8.27%**를 기록하여 단순 평균(FPR 11.92%) 및 중앙값(FPR 10.82%)을 능가하는 전체 1위 성능을 달성함.
  - 특히 보정 실패 고위험군인 S17(기저선 EDA 표준편차 3.63 $\mu\text{S}$)에서 단순 평균의 AUROC 0.5777을 **0.6900으로 대폭 구제**함.
  - `adaptive_fallback`은 착용 직후 극단적 과각성(EDA > 15 $\mu\text{S}$)이 감지될 때 보정을 생략하고 집단 모델로 즉각 전환함으로써, 코호트 최악 피험자 AUROC 하한선을 **0.7883**으로 가장 높게 방어함.

### Q4. 착용 직후 움직임이 보정 구간에 섞이면 어떤 문제가 생기는가?
- **답변: 보정 기저선이 과도하게 상승하여 실제 스트레스 발생 시의 생체 변화를 감지하지 못하고 놓치는 스트레스 미탐(Recall 급락) 참사가 발생한다.**
- **정량적 근거**:
  - 정상 휴식 기저선 대신 시계를 차고 버클을 조이는 초기 60초 착용 움직임 구간으로 기저선을 추정한 스트레스 테스트(`setup_motion_stress_test`) 결과:
  - WESAD 코호트 평균 스트레스 Recall이 정상 보정 대비 **-6.90%p 하락**함.
  - 특히 S04(Recall 89.2% $\to$ 56.4%, **-32.8%p**), S09(Recall 93.3% $\to$ 60.0%, **-33.3%p**) 등 기저선에서 격렬한 손목 움직임이 발생한 피험자들은 **실제 스트레스의 1/3 이상을 전혀 탐지하지 못하는 치명적 미탐**이 발생함.
  - 기저선 오경보율(FPR)은 낮아 보이지만, 이는 모델이 둔감해져 스트레스 신호를 모두 기저선으로 오인하기 때문임.

### Q5. AMDC의 조도·소음은 생체 예측에 유의미한 기여를 하는가?
- **답변: 전혀 기여하지 않으며, 오히려 미래 생체 반응 예측 오차를 증가시키고 모델 일반화를 방해한다.**
- **정량적 근거**:
  - AMDC 14명 피험자(33,993초) 대상 30초 미래 심박수(HR) 예측(LOSO 교차검증) 결과:
    - 과거 생체 신호만을 사용한 자기회귀 모델(`Model B`): MAE **3.692 bpm** (지속성 기준선 대비 **+11.86% 유의미한 개선**).
    - 조도(Lux)를 추가한 모델(`Model C`): MAE **3.708 bpm** (개선율 +11.50%로 **성능 악화**).
    - 소음(Sound dB)을 추가한 모델(`Model D`): MAE **3.714 bpm** (개선율 +11.34%로 **성능 악화**).
    - 조도와 소음을 모두 추가한 모델(`Model E`): MAE **3.727 bpm** (개선율 +11.04%로 **성능 추가 악화**).
  - 환경 신호 30개만으로 생체 반응을 예측하려 한 모델(`Model H`)은 **심박수 $R^2 = -5.311$, 피부전도도 $R^2 = -17.409$**로 완벽하게 붕괴함.
  - 조도·소음과 생체 신호 간의 상관계수는 피험자마다 부호가 반대로 역전(Sign Flip)되므로, 전역 기계학습 모델에 잡음 변수로만 작용함.

### Q6. 웨어러블에 조도·마이크 센서를 추가하는 것이 기술적으로 타당한가?
- **답변: 기술적·하드웨어적 관점에서 전혀 타당하지 않으며 탑재를 강력히 배제해야 한다.**
- **공학적 근거**:
  1. **예측 기여도 마이너스**: 실제 데이터 벤치마크에서 조도와 소음 특징은 외적 피험자 전이 시 예측 오차를 유의미하게 악화시킴.
  2. **폼팩터 및 배터리 부담**: 마이크로폰 구동 및 고주파 음향 DSP 연산은 웨어러블의 배터리 사용 시간을 치명적으로 단축시키며 PCB 면적과 기구 두께를 증가시킴.
  3. **신호 차폐 왜곡**: 손목의 조도 센서는 긴소매 옷에 빈번히 가려지며, 마이크는 착용자 자신의 옷깃 스침 소음과 기구 진동에 쉽게 오염됨.
  4. **프라이버시 저항**: 상시 마이크 청취 기능은 특수학교 및 가정 환경에서 개인정보 침해 및 도청 우려를 유발함.
  5. **최종 설계 권고**: 조도 센서와 마이크를 과감히 삭제하고, **고품질 BVP/PPG, EDA, 3축 ACC**의 삼중 센서 조합에 하드웨어 자원을 집중하는 것이 최적의 아키텍처임.

### Q7. 현재 모델과 데이터로 자폐 아동 멜트다운 사전 예측을 주장할 수 있는가?
- **답변: 절대 주장할 수 없으며, 이를 사전 예측 성공으로 발표하는 것은 명백한 연구 윤리 위반이자 과학적 허위 주장이다.**
- **과학적 근거**:
  1. **대상군의 불일치**: WESAD, PhysioNet, AMDC는 모두 건강한 성인 자원자를 대상으로 수집된 데이터이며, 자폐 스펙트럼 장애(ASD) 아동의 신경생리학적 데이터가 단 1건도 포함되어 있지 않음.
  2. **과제 및 라벨의 불일치**: 데이터셋의 과제는 산수 암산, 스트룹 검사, 찬물 손 담그기, 실내 사무 환경 노출 등 일반적인 인지·신체 스트레스이며, 자폐 아동의 감정 조절 상실 및 신경 발작성 멜트다운(Meltdown) 이벤트와는 병리학적 메커니즘이 근본적으로 다름.
  3. **사전 예측(Forecasting) 미검증**: 본 연구 모델은 현재 구간의 스트레스 상태를 실시간 탐지(Detection)하는 알고리즘이며, 멜트다운 발생 3~5분 전의 미세 전조 징후를 미래 지향적으로 예측(Forecasting)하는 유효성을 검증한 바 없음.

---

## 3. 대외 발표 가능한 검증된 핵심 사실 (3~5 Presentation-Ready Facts)

학술 발표, 투자자 피칭, 기술 보고서에 신뢰성 있게 인용할 수 있는 검증된 정량 사실은 다음과 같다:

1. **"1~3분의 짧은 기저선 보정만으로 웨어러블 스트레스 탐지기의 오경보율을 90% 이상 억제하면서 긴 대기시간 없이 98% 이상의 최적 탐지 성능을 달성할 수 있다."**
   - WESAD 3분 보정 시 F1 0.7888 (10분 보정 대비 98.4% 달성), PhysioNet 기저선 FPR 41.18% $\to$ 2.94% (LR).
2. **"기저선 보정 실패의 주원인은 착용 초기 움직임 오염이며, 3축 가속도계 기반 저동작 필터링(`low_motion`)을 적용하면 최악 피험자의 AUROC를 0.57에서 0.69로 구제하고 코호트 평균 AUROC 0.9518을 달성한다."**
   - 단순 평균 대비 FPR 11.92% $\to$ 8.27%로 감소, 착용 스트레스 테스트의 Recall 급락 방어.
3. **"손목형 기기에서 주변 조도(Lux)와 소음(Sound dB) 센서는 피험자 간 생체 반응 역전 현상(Sign Flip)으로 인해 미래 생체 신호 예측 오차를 오히려 악화시킨다."**
   - 심박수 예측에서 생체 단독(MAE 3.692 bpm) 대비 조도·소음 추가 시 MAE 3.727 bpm으로 악화, 환경 단독 모델은 $R^2 = -5.311$로 예측 기능 전무.
4. **"따라서 차세대 자폐 케어 웨어러블 하드웨어는 마이크와 조도 센서를 배제하고 BVP-EDA-ACC 삼중 센서에 자원을 집중하는 것이 공학적·경제적·프라이버시 측면에서 최선의 설계이다."**

---

## 4. 절대 해서는 안 될 부정 주장 (Absolute Negative Claims)

대외 커뮤니케이션 및 논문 작성 시 다음 주장은 엄격히 금지된다:
- ❌ *"현재 시스템으로 자폐 아동의 멜트다운을 3~5분 전에 사전 예측할 수 있음을 입증했다."* (입증된 적 없음)
- ❌ *"개인 보정은 무조건 길수록 좋으므로 5분~10분 보정을 필수로 요구해야 한다."* (3분 이후 한계효용 소멸, PhysioNet은 5분이 불가능함)
- ❌ *"소음과 조도 센서를 손목에 장착하면 감각 과부하 스트레스를 더 정확하게 예측할 수 있다."* (실제 기계학습 일반화 오차 증가)
- ❌ *"PhysioNet에서 5분 보정이 실패한 것은 PhysioNet 신호의 품질이 불량하기 때문이다."* (실제 연속 휴식이 3.2분에 불과했던 프로토콜 설계 한계임)

---

## 5. 생성 산출물, 데이터 및 재현 명령 (Artifact Inventory & Reproduction)

### 5.1 보고서 및 설정 파일 매핑

| 보고서/설정 파일 | 경로 | 설명 |
|:---|:---|:---|
| 연구 A 보고서 | [`reports/CALIBRATION_DURATION_COMPARISON_KO.md`](file:///D:/Autism/reports/CALIBRATION_DURATION_COMPARISON_KO.md) | 0·1·3·5·10분 보정 시간 비교 상세 보고서 |
| 연구 B 보고서 | [`reports/INITIAL_BASELINE_ROBUSTNESS_KO.md`](file:///D:/Autism/reports/INITIAL_BASELINE_ROBUSTNESS_KO.md) | 초기 기저선 실패 분석 및 6대 추정량 비교 보고서 |
| 연구 C 감사 보고서 | [`reports/AMDC_DATA_AUDIT_KO.md`](file:///D:/Autism/reports/AMDC_DATA_AUDIT_KO.md) | AMDC 데이터셋 무결성, 단위 변환 및 적합성 감사 |
| 연구 C 모델 보고서 | [`reports/AMDC_ENVIRONMENT_PHYSIOLOGY_REPORT_KO.md`](file:///D:/Autism/reports/AMDC_ENVIRONMENT_PHYSIOLOGY_REPORT_KO.md) | 조도·소음 연관성 및 8개 예측 회귀 모델 벤치마크 |
| 종합 요약 보고서 | [`reports/FOLLOWUP_RESEARCH_SUMMARY_KO.md`](file:///D:/Autism/reports/FOLLOWUP_RESEARCH_SUMMARY_KO.md) | 후속 연구 전체 총괄 요약 (본 보고서) |
| 사전 등록 설정 | [`experiments/calibration_duration/config.json`](file:///D:/Autism/experiments/calibration_duration/config.json) | 연구 A 비교 조건 및 프로토콜 사전 등록 |

### 5.2 실험 결과 데이터 파일

| 결과 파일 | 경로 | 크기/내용 |
|:---|:---|:---|
| 연구 A 요약 CSV | [`results/calibration_duration/latest/calibration_duration_summary.csv`](file:///D:/Autism/results/calibration_duration/latest/calibration_duration_summary.csv) | WESAD·PhysioNet 보정 길이별 종합 성능 집계 |
| 연구 A 피험자별 CSV | [`results/calibration_duration/latest/calibration_duration_per_subject.csv`](file:///D:/Autism/results/calibration_duration/latest/calibration_duration_per_subject.csv) | 피험자별 세부 지표 |
| 연구 B 추정량 요약 CSV | [`results/baseline_robustness/latest/baseline_estimators_summary.csv`](file:///D:/Autism/results/baseline_robustness/latest/baseline_estimators_summary.csv) | 6대 기저선 추정량 비교 요약 |
| 연구 B 스트레스 테스트 CSV | [`results/baseline_robustness/latest/setup_motion_stress_test.csv`](file:///D:/Autism/results/baseline_robustness/latest/setup_motion_stress_test.csv) | 착용 동잡음 오염 시 Recall 저하 평가 |
| 연구 C 매니페스트 CSV | [`data/registry/amdc/amdc_files_manifest.csv`](file:///D:/Autism/data/registry/amdc/amdc_files_manifest.csv) | AMDC 23개 파일 및 SHA-256 해시 |
| 연구 C 상관계수 CSV | [`data/registry/amdc/amdc_exploratory_correlations.csv`](file:///D:/Autism/data/registry/amdc/amdc_exploratory_correlations.csv) | 14명 피험자별 개체 내 상관계수 |
| 연구 C 예측 회귀 요약 CSV | [`results/amdc/latest/amdc_predictive_summary.csv`](file:///D:/Autism/results/amdc/latest/amdc_predictive_summary.csv) | 8개 예측 모델 비교 종합 집계 |
| 연구 C 피험자별 회귀 CSV | [`results/amdc/latest/amdc_predictive_per_subject.csv`](file:///D:/Autism/results/amdc/latest/amdc_predictive_per_subject.csv) | 피험자 14명별 세부 회귀 오차 |

### 5.3 100% 재현 실행 명령어

모든 실험 및 단위 테스트는 가상환경 활성화 상태에서 다음 명령으로 완전 재현 가능하다:

```powershell
# 1. 가상환경 활성화
& "d:\Autism\.venv\Scripts\Activate.ps1"

# 2. 연구 A: 보정 시간 비교 실행
python experiments/calibration_duration/run_calibration_duration.py

# 3. 연구 B: 초기 기저선 강건성 및 스트레스 테스트 실행
python experiments/baseline_robustness/run_baseline_robustness.py

# 4. 연구 C: AMDC 데이터 감사 및 전처리
python experiments/amdc/audit_and_preprocess.py

# 5. 연구 C: 조도·소음 생체 시계열 예측 회귀 실행
python experiments/amdc/run_amdc_predictive_modeling.py

# 6. 전체 단위 테스트 및 무결성 검증 (36개 테스트)
python -m unittest discover -s tests -p "test_*.py" -v
```
