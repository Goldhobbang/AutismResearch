# Nurses 기반 미래 스트레스 사건 예측 모델 배포 명세서 (`NURSE_MODEL_DEPLOYMENT_SPEC.md`)

- **모델 공식 명칭**: **Nurses 기반 미래 스트레스 사건 예측 모델** (`nurse_stress_forecast_lr_p1`)
- **모델 버전**: `1.0.0-nurse-p1`
- **예측 과제**: `P1` — 과거 180초(`t-180s ~ t`) 생체신호만 사용하여 향후 **60~180초(`t+60s ~ t+180s`) 내 스트레스 사건 시작(Onset)** 여부 예측
- **학습 데이터셋**: Hospital Nurses Stress Dataset (15명 참가자, 설문 매칭 214개 세션, 평가 적격 27,250개 창, P1 적격 사건 176건)
- **학습 데이터 해시 (SHA-256)**: `54e9cf4f34ad325e6926d1d251f3aa387e361b33c553be3d0d529b5f3fcf9603`
- **엄격한 명칭 및 용도 제한**:
  - 본 모델은 병원 간호사의 직무 스트레스 사건 시작 경계를 사전 예측하도록 학습된 **Nurses 기반 미래 스트레스 사건 예측 모델**이다.
  - **실제 자폐 스펙트럼(ASD) 아동의 멜트다운(Meltdown) 예측 모델이라고 표현하거나 주장할 수 없다.**

---

## 1. 모델 파일 및 산출물 위치

| 구분 | 파일 경로 | 크기 (Bytes) | 설명 |
|---|---|---:|---|
| **주 선정 모델 (M1 LR)** | [`results/nurse_forecasting/models/nurse_stress_forecast_lr_p1.joblib`](file:///D:/Autism/results/nurse_forecasting/models/nurse_stress_forecast_lr_p1.joblib) | `7,639 B` (`7.46 KB`) | `SimpleImputer(median)` + `StandardScaler` + `LogisticRegression(class_weight='balanced', C=0.2)` 파이프라인 |
| **MCU/임베디드 배포 명세 JSON** | [`results/nurse_forecasting/models/nurse_stress_forecast_lr_p1_spec.json`](file:///D:/Autism/results/nurse_forecasting/models/nurse_stress_forecast_lr_p1_spec.json) | `13,794 B` (`13.47 KB`) | 62개 특징 순서, 결측 대체 중앙값, 스케일러 평균·표준편차, 회귀 계수(`w`), 절편(`b`), 게이트·히스테리시스 설정 |
| **비교 모델 (M2 RF)** | [`results/nurse_forecasting/models/nurse_stress_forecast_rf_p1.joblib`](file:///D:/Autism/results/nurse_forecasting/models/nurse_stress_forecast_rf_p1.joblib) | `1,449,952 B` (`1.42 MB`) | `RandomForestClassifier(n_estimators=200, max_depth=8)` 비교용 |
| **비교 모델 (M3 LightGBM)** | [`results/nurse_forecasting/models/nurse_stress_forecast_lgbm_p1.joblib`](file:///D:/Autism/results/nurse_forecasting/models/nurse_stress_forecast_lgbm_p1.joblib) | `221,588 B` (`216.39 KB`) | `LGBMClassifier(n_estimators=120, max_depth=4)` 비교용 |

---

## 2. 입력 센서 사양 및 주파수·단위

| 센서 채널 | 샘플링 주파수 | 물리 단위 | 180초 관측 창 샘플 수 | 링버퍼 메모리 (`float32`) | 필수 여부 |
|---|---:|---|---:|---:|:---:|
| **EDA (피부전도도)** | `4.0 Hz` | $\mu\text{S}$ (microSiemens) | `720 samples` | `2,880 B` | 필수 (주 센서) |
| **BVP / PPG (광혈류)** | `64.0 Hz` | Raw PPG | `11,520 samples` | `46,080 B` | 필수 (주 센서) |
| **ACC (3축 가속도계)** | `32.0 Hz` | $\frac{1}{64}g$ ($x, y, z$ 3축) | `5,760 × 3 samples` | `69,120 B` | 필수 (주 센서) |
| **HR (심박수)** | `1.0 Hz` | $\text{bpm}$ (BVP 파생) | `180 samples` | `720 B` | 보조 (BVP에서 실시간 산출 가능) |
| **IBI (박동간 간격)** | Event-based | $\text{s}$ / $\text{ms}$ | 가변 (`<= 300 peaks`) | `1,200 B` | 보조 (`ibi_valid_flag`로 결측 방어) |
| **TEMP (피부온도)** | `4.0 Hz` | $^\circ\text{C}$ | 미사용 | `0 B` | **제외** (타깃 제품 비필수 센서) |
| **합계** | — | — | — | **`120,000 B` (`117.19 KB`)** | — |

---

## 3. 입력 특징 명세 (총 62개 인과 특징)

모든 특징은 예측 시각 $t$ 기준 **과거 관측 구간 $[t-180\text{s}, t)$** 및 인과 누적 이력 $[t-300\text{s}, t)$ 내부 샘플만으로 계산되며, $t$ 이후의 미래 샘플은 단 1개도 참조하지 않는다.

### 3.1 EDA 특징 (26개)
- **원시 통계 및 추세 (16개)**: `eda_mean`, `eda_median`, `eda_std`, `eda_range`, `eda_q10`, `eda_q90`, `eda_slope`, `eda_absdiff_per_s`, `eda_positive_diff_sum`, `eda_scr_candidate_count`, `eda_log_mean`, `eda_log_std`, `eda_log_slope`, `eda_rise_fraction`, `eda_recovery_ratio`, `eda_recent60_minus_prev120`
- **다중 시간창 요약 (4개)**: `eda_60s_mean`, `eda_60s_slope`, `eda_300s_mean`, `eda_300s_slope`
- **개인 기저선 대비 인과 변화량 (6개)**: `eda_log_delta`, `eda_log_std_delta`, `eda_pos_diff_log_delta`, `eda_scr_log_delta`, `eda_slope_norm`, `eda_iqr_log_delta`

### 3.2 PPG / HR / HRV 특징 (21개)
- **PPG 파형 및 주파수 (8개)**: `ppg_mean`, `ppg_std`, `ppg_range`, `ppg_q10`, `ppg_q90`, `ppg_absdiff_per_s`, `ppg_peak_hz`, `ppg_band_fraction`
- **심박수 및 다중 시간창 추세 (7개)**: `hr_mean_bpm`, `hr_std_bpm`, `hr_range_bpm`, `hr_slope_bpm_per_s`, `hr_delta_recent60_bpm`, `hr_60s_mean_bpm`, `hr_300s_mean_bpm`
- **심박변이도 및 품질 플래그 (3개)**: `ibi_rmssd_ms`, `ibi_sdnn_ms`, `ibi_valid_flag`
- **개인 기저선 대비 인과 변화량 (3개)**: `hr_bpm_delta`, `ppg_band_delta`, `ppg_absdiff_log_ratio`

### 3.3 ACC 3축 회전 불변 및 활동량 특징 (15개)
- **3축 회전 불변($SO(3)$) 크기 및 주파수 (13개)**: `acc_mag_mean`, `acc_mag_median`, `acc_mag_std`, `acc_mag_range`, `acc_mag_q10`, `acc_mag_q90`, `acc_mag_slope`, `acc_mag_absdiff_per_s`, `acc_dynamic_energy`, `acc_motion_change_per_s`, `acc_unchanged_fraction`, `acc_peak_hz`, `acc_band_fraction`
- **다중 시간창 활동 누적 (2개)**: `acc_60s_motion_change`, `acc_300s_motion_change`

---

## 4. 전처리, 개인화 보정, 운동 게이트 및 경고 정책

### 4.1 결측 대체 및 표준화 (Inference Equations)
각 특징 $x_j$ ($j = 1, \dots, 62$)에 대해:
1. 결측(`NaN`) 발생 시 JSON 명세의 `imputer_median[j]` 값으로 대체한다.
2. 표준화:
   $$z_j = \frac{x_j - \mu_j}{\sigma_j}$$
   여기서 $\mu_j = \text{scaler\_mean}[j]$, $\sigma_j = \text{scaler\_scale}[j]$이다.
3. 로지스틱 회귀 선형 결합 및 확률 산출:
   $$p(t) = \frac{1}{1 + \exp\left(-\left(b + \sum_{j=1}^{62} w_j z_j\right)\right)}$$

### 4.2 인과적 개인 기준선 보정 (`personal_calibration`)
- **기본 설정**: `low_motion` (초기 180초 구간 내 저동작 창 우선 선택)
  - 세션 시작 후 $[t_0, t_0 + 180\text{s}]$ 내의 창 중 `acc_motion_change_per_s <= 38.0` (또는 하위 60% 저동작 창)의 중앙값으로 개인 기저선 벡터 $(\text{EDA}_0, \text{HR}_0, \text{PPG}_0)$를 추정한다.
  - 보정 구간에 유효 창이 부족할 경우 현재 시점 $t$ 이전의 인과 누적 중앙값으로 안전 폴백(`adaptive_fallback`)한다.
  - **미래 누수 차단**: 어떠한 경우에도 $t$ 이후의 창이나 테스트 참가자의 정답 라벨을 보정에 사용하지 않는다.

### 4.3 운동·활동량 게이트 (`motion_gate`)
- **운용 모드 선택**:
  1. `none` (무보정 기본): Event Recall 최우선 운용 시 사용 (P1 LR Event Recall `40.34%`, FA/hr `1.5678`).
  2. `soft_dynamic` (기존 연구 이식 게이트): 고활동 구간(`Q4`) 경보율을 `23.94% -> 2.78%`로 억제하고 시간당 오경보를 `1.5678 -> 1.0811/hr`로 낮추나, 상시 보행 중인 병원 간호사 특성상 Event Recall이 `40.34% -> 22.73%`로 감소함.
  3. `nurse_adapted_soft` (병원 직무 활동 분포 맞춤 게이트): 정지 비율 `acc_unchanged_fraction < 0.06` 및 저크 `acc_motion_change_per_s > 220` 기준으로 활동 점수 $M_{\text{nurse}}(t)$를 산출하여 시간당 오경보를 `1.2604/hr`로 억제하면서 Event Recall `28.98%` (RF는 `32.39%`, LightGBM은 `43.18%`)를 유지.

### 4.4 경고 발생 및 병합 정책 (`alert_policy`)
- **단일 임계값 정책 (`val_opt`)**: 내부 검증셋 최적 임계값(전체 모델 기준 `decision_threshold = 0.68`).
- **히스테리시스 정책 (`hysteresis`, 권장)**:
  - 진입 임계값 $T_{\text{on}} = \theta_{\text{val}}$, 해제 임계값 $T_{\text{off}} = \max(0.15, \theta_{\text{val}} - 0.15)$.
  - **에피소드 병합 규칙 (`episode_merge_gap_s = 180.0`)**: 30초 간격으로 중첩 생성되는 연속 경고 창들을 독립 경보로 중복 집계하지 않고, `180초` 이내 인접 경고를 단일 경보 에피소드로 병합한다.
  - 효과: LR 기준 단일 임계값 대비 시간당 오경보를 **`1.5678/hr -> 0.9965/hr` (`-36.4%`)**로 줄이면서 사건 탐지율(Event Recall)은 **`40.34% -> 44.89%`**로 오히려 향상됨.

---

## 5. 하드웨어(ESP32-S3) 탑재 가능성 및 공학적 제약

1. **메모리 요구량 (이론적 계산치 — 검증 완료)**:
   - **모델 파라미터**: 62개 특징 × 4개 배열(`imputer_median`, `scaler_mean`, `scaler_scale`, `coefficients`) + `intercept` = `249 × 4 bytes` = **`996 Bytes (< 1 KB)`**.
   - **원시 센서 180초 링버퍼**: EDA(720) + BVP(11,520) + ACC(17,280) + HR(180) = **`118,800 Bytes` (`~116.0 KB`)**.
   - **판정**: ESP32-S3 내부 SRAM(`512 KB`)의 약 `23%`만 점유하므로 외부 PSRAM 없이 온디바이스 상주 가능함. (반면 `1.42 MB` 크기인 Random Forest는 내부 SRAM 단독 적재 불가).
2. **추론 연산 지연 시간 (PC CPU 실측치)**:
   - 180초 단일 창 특징 추출 시간: 평균 **`6.47 ms`** (중앙값 `6.12 ms`).
   - 로지스틱 회귀 파이프라인 단일 창 추론 시간: **`< 1.0 ms`** (62차원 내적 1회).
3. **미검증 사항 (엄격한 한계 명시)**:
   - 본 실험은 PC(Windows 11, Python 3.13) 환경에서 수행되었으며, **실제 ESP32-S3 하드웨어 보드에 펌웨어를 플래싱하여 측정한 전력 소모량, 배터리 지속시간, 블루투스(BLE) 패킷 손실률은 미검증(UNVERIFIED)** 상태이다.
