# 검증용 테스트케이스 요구사항 및 설계 가이드라인 (VERIFICATION_TEST_SPEC.md)

본 문서는 연구 실행 원칙 제9조(*"검증용 테스트케이스는 요구사항·설계 가이드라인을 작성하고, 별도의 검증 담당 에이전트가 구현하도록 하라"*)에 따라, 모델 개선 파이프라인의 무결성·무누수(Zero-Leakage)·일반화 안정성을 독립적으로 검증하기 위해 작성된 테스트 명세서이다.

검증 담당 에이전트(`verification-engineer`)는 본 명세서의 요구사항을 충족하는 단위/회귀 테스트 스크립트를 `tests/test_phase4_advanced_pipeline.py`에 구현하고 전체 테스트 스위트(`unittest`)를 실행해야 한다.

---

## 1. 검증 목표 및 범위

1. **인과성 및 미래 정보 누수 차단 (Strict Causality & Zero Future Leakage)**
   - 초기 기준선 보정(Personal Baseline Calibration), 온라인 기준선 갱신(Online Baseline Update), 시간적 추세 특징(Temporal EMA/Lag Features) 추출 시 추론 시점 $t$ 이후($\tau > t$)의 신호나 라벨이 단 1비트도 참조되지 않음을 보장한다.
2. **Motion Gate의 스트레스 Recall 보존 및 운동 오경보 억제 동시 검증 (Motion Gate Dual Objective)**
   - 정적 스트레스 및 일반 손동작(제스처)이 수반된 스트레스 구간에서는 스트레스 확률이 강제 소거되지 않아야 한다(Stress Recall 보존).
   - 반면 지속적·주기적 유산소 운동(낮은 정지 비율 `acc_unchanged_fraction` + 높은 연속 가속도 변화율 `acc_motion_change_per_s`) 구간에서는 허위 스트레스 경보가 억제되어야 한다.
   - 동적 임계값(Dynamic Thresholding) 및 소프트 게이팅(Soft Gating) 적용 시 창 간 상대적 우선순위(Ranking)가 상수 클램핑으로 인해 완전히 붕괴되지 않음을 확인한다.
3. **고전도도(High-Tonic) 및 저반응(Low-Responder) 피험자 보정 불변성 검증**
   - 기저 피부전도도(EDA)가 매우 낮은 피험자(예: $0.2\,\mu\text{S}$)와 매우 높은 피험자(예: $20.0\,\mu\text{S}$) 모두에서, 초기 기저치 대비 상대적 교감신경 상승이 발생했을 때 이차항(`eda_energy`) 폭주로 인한 확률 역전(AUROC $< 0.5$) 없이 단조 증가(Monotonicity)함을 검증한다.
4. **ACC 방향 불변성 (3D Orientation Invariance)**
   - 3축 가속도계 신호 $\mathbf{a}(t) \in \mathbb{R}^{N \times 3}$에 임의의 3차원 직교 회전 행렬 $R \in SO(3)$ ($R^\top R = I, \det R = 1$)을 곱하더라도 방향 불변 ACC 특징(`acc_mag_*`, `acc_motion_change_per_s`, `acc_unchanged_fraction`, `acc_peak_hz`, `acc_band_fraction`)이 수치 오차($10^{-9}$) 이내로 완전히 동일함을 검증한다.
5. **참가자 분할 무결성 및 평가 지표 정확성 (Group Split & Metric Integrity)**
   - LOSO 교차검증 및 외부 데이터 평가 시 동일 참가자(`subject`)의 데이터가 훈련/검증/테스트에 중복 포함되지 않으며, 임계값은 테스트 라벨을 참조하지 않고 결정됨을 검증한다.
   - 사건 단위(Event-level) 평가 및 시간당 오경보율(FAR/hr) 계산 시 시간 단절(Time gap) 구간에서 경보 상태가 올바르게 리셋되는지 확인한다.

---

## 2. 필수 테스트케이스 설계 가이드라인

검증 담당 에이전트는 `tests/test_phase4_advanced_pipeline.py` 내에 다음 `unittest.TestCase` 클래스 및 메서드들을 구현해야 한다:

### TC-01: `TestOrientationInvariance`
- **입력**: 무작위 중력 벡터 + 동적 정현파/가우시안 진동이 포함된 가상 3축 ACC 배열 $(1920, 3)$ ($32\text{ Hz}, 60\text{s}$).
- **조작**: 3D 회전 행렬 $R = R_z(\psi) R_y(\theta) R_x(\phi)$를 생성하여 $\mathbf{a}_{\text{rot}} = \mathbf{a} R^\top$ 생성.
- **검증 조건**:
  - 방향 불변 특징(`acc_mag_mean`, `acc_mag_std`, `acc_mag_range`, `acc_mag_absdiff_per_s`, `acc_motion_change_per_s`, `acc_unchanged_fraction`, `acc_peak_hz`, `acc_band_fraction`)이 회전 전후 `np.testing.assert_allclose(..., rtol=1e-6, atol=1e-6)`를 통과해야 한다.

### TC-02: `TestCausalCalibrationAndTemporalNoFutureLeakage`
- **입력**: 한 피험자의 시계열 윈도우 데이터프레임 ($t \in [0, 1200]\text{s}$).
- **조작**: $t > 600\text{s}$ 이후 미래 구간의 모든 특징값을 극단값($+10^6$)으로 오염시킨 데이터프레임 생성.
- **검증 조건**:
  - 초기 기저 보정 및 인과적 시간 특징(Causal EMA / 온라인 기준선 갱신) 함수를 적용했을 때, $t \le 600\text{s}$ 구간의 모든 보정된 특징값 및 예측 확률이 오염 전과 정확히 일치해야 한다 (`assert_allclose`).
  - 보정 구간에 미래 스트레스 라벨 정보가 사용되지 않음을 검증한다.

### TC-03: `TestHighTonicAndLowResponderMonotonicity`
- **입력**:
  - 저반응군 피험자 데이터(기저 EDA $0.20\,\mu\text{S} \rightarrow$ 스트레스 EDA $0.35\,\mu\text{S}$)
  - 고기저군 피험자 데이터(기저 EDA $20.0\,\mu\text{S} \rightarrow$ 스트레스 EDA $28.0\,\mu\text{S}$)
- **검증 조건**:
  - 개인별 기저 보정 및 로그/상대 특징 변환 후, 두 피험자 모두에서 스트레스 구간의 예측 점수 평균이 기저 구간의 예측 점수 평균보다 엄격히 높고(`p_stress > p_base`), AUROC $\ge 0.85$를 달성해야 한다.

### TC-04: `TestMotionGateRecallPreservationAndExerciseSuppression`
- **입력**:
  - 시나리오 1 (가벼운 손동작 동반 실제 스트레스): 높은 교감신경 특징 + 간헐적 제스처(`acc_unchanged_fraction = 0.25`, `acc_motion_change_per_s = 65.0`).
  - 시나리오 2 (지속적 유산소 사이클링 운동): 발한으로 인한 높은 원시 EDA + 연속 페달링 진동(`acc_unchanged_fraction = 0.005`, `acc_motion_change_per_s = 165.0`).
- **검증 조건**:
  - 기존 하드 게이트(`b46a2e9`)와 달리, 개선된 동적 모션 게이트는 시나리오 1(제스처 동반 스트레스)에서 양성 판정($P \ge \theta$, Recall 보존)을 유지해야 하며, 시나리오 2(지속적 유산소 운동)에서는 오경보를 억제($P_{\text{gated}} < \theta$, Exercise FPR 억제)해야 한다.

### TC-05: `TestTagAlignedManifestIntegrity`
- **입력**: `data/processed/physionet/verified_segments_tag_aligned.csv` 및 `data/processed/physionet/verified_segments.csv`.
- **검증 조건**:
  - 공식 제외 대상인 `S02` STRESS, `S03`/`S07`/`S11`/`S12` AEROBIC이 적격(`verified == True`) 목록에 포함되지 않았음을 확인한다.
  - 모든 적격 세그먼트가 `end_s > start_s` 및 최소 창 길이($60\text{s}$) 이상임을 확인한다.

---

## 3. 실행 및 보고 절차

1. 기존 테스트(`tests/test_*.py`)와 신규 테스트(`tests/test_phase4_advanced_pipeline.py`)를 모두 실행한다:
   ```powershell
   .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
   ```
2. 모든 테스트가 `OK` (실패 0건)임을 확인하고 결과를 메인 에이전트에게 보고한다.
