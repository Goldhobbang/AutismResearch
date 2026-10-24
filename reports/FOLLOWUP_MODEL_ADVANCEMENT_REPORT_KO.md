# AutismResearch — 생체신호 기반 스트레스 탐지 고도화 및 멜트다운 사전 예측 후속 연구 최종 보고서

- **작성 일자**: 2026-10-24
- **기준 저장소 및 커밋**: `https://github.com/Goldhobbang/autismresearch` (기준 커밋 `b46a2e9`)
- **검증 데이터셋**:
  1. **WESAD**: 성인 15명 전수 (`data/raw/WESAD/`, 5,740개 창 중 엄격 QC 통과 2,031개 이진 적격 창)
  2. **PhysioNet Wearable Device Dataset v1.0.1**: V1 코호트 18명 전수 (`S01~S18`, 238개 원본 파일 공식 `SHA256SUMS.txt` 100% 해시 일치 검증 완료; 적격 `STRESS` 17명, 적격 `AEROBIC` 14명)
  3. **자폐 아동 멜트다운(ASD Pre-Meltdown) 데이터**: 공개 데이터 부재 확인 (`BLOCKED_NOT_EVALUATED` / `N/A` 원칙 엄수, 임의 라벨 생성 일체 없음)

---

## 0. 사실·문헌·가설의 엄격한 인식론적 구분 (Epistemic Separation)

본 연구 보고서는 지시서 제7조에 따라 모든 서술을 다음 3가지 범주로 명확히 구분한다:

| 구분 범주 | 본 연구에서 해당하는 핵심 내용 |
|---|---|
| **[실험 확인 사실]<br>(Experimentally Verified)** | • 기존 `b46a2e9` 보고서의 5대 결함(3명 부분 표본, `[0,300]` 하드코딩 라벨 역전, `rf_all` Recall 0% 착시, 테스트 라벨 누수 선형 차감, `eda_energy` 제곱항 폭발) 규명<br>• PhysioNet `S01~S18` 238개 전 파일 다운로드 및 SHA-256 무결성 검증<br>• 3차원 회전 불변 ACC 특징(`SO(3)` 오차 $< 10^{-12}$) 및 무누수 인과성(단위 테스트 32건 전원 PASS)<br>• 개선 파이프라인(`Ensemble_Calib_Temporal`, `RF_Calib_Temporal`, `LGBM_Calib_Temporal`, `LR_Calib_Physio`)의 WESAD 15명 LOSO 및 PhysioNet 17명 STRESS / 14명 AEROBIC 정량 성능 향상 실측 |
| **[문헌 근거 내용]<br>(Literature-Grounded)** | • 자폐 아동의 공격성·자해·멜트다운 발생 1~3분 전 자율신경계 전조(미주신경 철회에 의한 RMSSD 감소, Tonic SCL 및 NS-SCR 상승, 급성 빈맥) 기전 (Goodwin et al., 2019; Imbiriba et al., 2023; Nuske et al., 2019; Kartha et al., 2026)<br>• 자폐 집단 내 고전도도 과각성형(Hyper-aroused) 및 저반응형(Hypo-responsive) 자율신경 하위군 존재 (Lydon et al., 2016) |
| **[미검증 가설]<br>(Unverified Hypotheses)** | • 성인 실험실 스트레스·운동 데이터(WESAD, PhysioNet)에서 검증된 본 연구의 인과적 보정·동적 모션 게이팅 모델이 실제 자폐 아동의 일상 멜트다운 3~5분 전 예측에서도 동일한 AUROC(>0.85)를 달성할 것이라는 가설 (실제 ASD 임상 코호트 데이터 확보 후 검증 필수)<br>• 리듬성 상동행동(Stimming) 중 자기진정 목적의 움직임과 멜트다운 전조 움직임을 손목 EDA+BVP+ACC만으로 90% 이상 분리할 수 있다는 가설 |

---

## 1. 기존 모델(`b46a2e9`) 문제점 및 원인 분석 ([실험 확인 사실])

기존 보고서([`PHYSIONET_EXTERNAL_GENERALIZATION_REPORT.md`](file:///D:/Autism/reports/PHYSIONET_EXTERNAL_GENERALIZATION_REPORT.md))와 원시 출력([`results/phase3_strict/physionet_external_b46a2e9_backup/`](file:///D:/Autism/results/phase3_strict/physionet_external_b46a2e9_backup)) 및 코드를 전수 대조한 결과 다음 5가지 원인을 확인하였다:

1. **참가자 표본 편향 (3명 부분 평가)**:
   - 기존 `generalization_summary.csv`는 `STRESS` 3명(`S01, S03, S04`)과 `AEROBIC` 2명(`S01, S04`)의 평균이었다 (`S02 STRESS`, `S03 AEROBIC`, `S04 STRESS ACC`는 손상/불완전 파일이었고 `S05~S18`은 미확보).
   - 238개 원본 파일을 전수 복구하여 동일한 `b46a2e9` 코드로 17명 전체를 평가한 결과, `rf_all` Cold-Start 평균 AUROC는 **`0.4366`**, Motion-Gated 평균 AUROC는 **`0.6677`**로 하락했으며 **17명 평균 스트레스 Recall은 `0.0025`(0.25%, 17명 중 16명에서 Recall 0%)**였다.
2. **프로토콜 타임스탬프 하드코딩 오류 ([`build_physionet_segments.py`](file:///D:/Autism/experiments/phase2/build_physionet_segments.py))**:
   - 각 피험자의 `tags.csv`와 공식 `Wearable_Dataset.ipynb`를 무시하고 모든 피험자에게 `Baseline=[0, 300]`, `Stroop=[300, 600]` 등을 일괄 적용했다.
   - 실제로는 `[0, tag[0]]`(평균 `0~178초`)이 **시계 착용 및 실험 안내(Pre-Protocol Setup)** 구간이어서 가속도 변화율(`88.1` vs 실제 기저선 `29.6`)과 심박수(`81.6` vs `72.8 bpm`)가 높았고, `[300, 600]`은 실제 휴식 기저선(`tag[0]..tag[1]`) 및 첫 휴식(`tag[3]..tag[4]`)과 겹쳤다. 또한 `Subtract=[1620, 1920]`은 17명 중 12명에서 전체 녹화 길이를 초과해 누락되었다.
3. **Motion Gate의 스트레스 억제 및 AUROC 착시 ([`experiment_physionet_generalization.py`](file:///D:/Autism/experiments/phase3/experiment_physionet_generalization.py))**:
   - 기존 하드 게이트(`acc_dynamic_energy > 150` 또는 `acc_motion_change_per_s > 40` $\rightarrow \min(p, 0.05)$)는 **실제 스트레스 창의 43.49%**를 `0.05`로 강제 소거했다.
   - `S03`에서 `rf_all`의 AUROC가 `0.3147` $\rightarrow$ `0.8838`로 급등한 이유는 모델이 스트레스를 잘 탐지해서가 아니라, Cold-Start에서 모든 창이 `< 0.5`로 예측(`Recall = 0%`)되던 상태에서 `[0, 300]` 착용 준비 구간의 기저선 창 17개 중 **16개(94.1%)가 `0.05`로 클램핑**되어 스트레스 창(`0.264`)보다 낮아지면서 **Recall `0.0%`인 채로 순위만 역전된 통계적 착시**였다.
4. **5분 보정(`calibrated_5min`)의 테스트 라벨 누수 및 선형 차감 오류**:
   - 특징 평균 `baseline_mu`를 계산만 하고 사용하지 않은 채, 테스트 데이터의 전체 `baseline` 라벨 평균 확률 `p_base_mean`을 구해 `np.clip(p_raw - p_base_mean + 0.10, 0, 1)`을 적용했다. `p_base_mean > 0.60`이면 `p_raw = 1.0`이어도 보정 확률이 `0.50`에 도달할 수 없어 Recall이 `0%`로 붕괴되었다.
5. **선형 모델의 이차항 `eda_energy`($\mu^2 + \sigma^2$) 다중공선성 역전**:
   - `eda_mean`($\mu$)과 `eda_energy`($\mu^2+\sigma^2$)를 함께 학습한 결과 `lr_all`과 `lr_eda`가 `eda_energy`에 음(-)의 가중치(`-1.3923`, `-0.7796`)를 부여했다. 기저 EDA가 높은 피험자(`S05`: $10.9\rightarrow 15.7\,\mu\text{S}$, `S10`: $21.6\rightarrow 30.5\,\mu\text{S}$)에서는 스트레스 시 $\text{EDA}^2$이 폭발($470\rightarrow 930$)하며 예측 확률이 `0`으로 수렴해 AUROC가 `0.0235`(`S05`), `0.0807`(`S10`)로 역전되었다.

---

## 2. 개선 가설과 실험 방법 ([실험 확인 사실])

| 가설 번호 | 연구 가설 (Hypothesis) | 구현 및 검증 방법 | 검증 결과 요약 |
|:---:|---|---|---|
| **H1** | `tags.csv` 기반 공식 프로토콜 세그먼트 정합만으로도 비보정 모델의 라벨 역전이 해소될 것이다. | [`build_physionet_tag_aligned_segments.py`](file:///D:/Autism/experiments/phase2/build_physionet_tag_aligned_segments.py) 구현 후 동결 모델 재평가 | **채택**: `Frozen_lr_all` Cold-Start 17명 평균 AUROC `0.4847` $\rightarrow$ **`0.7529`**, `Frozen_rf_all` `0.4366` $\rightarrow$ **`0.6686`** |
| **H2** | 연속 페달링/주행 운동(`acc_unchanged_fraction < 0.08` & `acc_motion_change_per_s > 45`)과 간헐적 손동작이 동반된 스트레스를 분리하는 소프트 동적 게이트를 적용하면 운동 오경보 억제와 스트레스 Recall 보존을 동시에 달성할 수 있다. | [`src/features/advanced.py`](file:///D:/Autism/src/features/advanced.py) 내 `compute_causal_exercise_score` 및 `apply_dynamic_motion_gate(mode='soft_dynamic')` 구현 | **채택**: 제스처 동반 스트레스 Recall `0.0%` $\rightarrow$ **`80.49%`** 복구, 유산소 운동 오경보율 **`1.27%`** (`0.46회/hr`) 억제 |
| **H3** | 제곱항 `eda_energy`를 제거하고 인과적 초기 기저선 대비 로그-델타($\Delta \log(1+\text{EDA})$) 및 상대 HR/PPG 특징을 추출하면 저반응군($<0.6\,\mu\text{S}$)과 고기저군($>4.0\,\mu\text{S}$) 모두에서 단조성을 회복할 것이다. | `apply_causal_personal_calibration` 및 `Strict_PostCal_Tag_Aligned` (보정 창 평가 제외) 검증 | **채택**: 고기저군(`S05, S10`) AUROC `0.02~0.20` $\rightarrow$ **`0.88~1.00`** 회복; 보정 창 완전 제외 시에도 17명 평균 AUROC **`0.8447`** 달성 |
| **H4** | 단일 60초 창 특징에 인과적 지수이동평균(`Causal EMA`, $\alpha=0.45$)과 1단계 차분(`diff1`)을 결합하면 스트레스 누적 궤적을 포착하여 내부(WESAD) 및 외부(PhysioNet) 일반화가 모두 향상될 것이다. | `add_causal_temporal_features` 적용 전후 `LR`, `RF`, `LGBM`, `HGB`, `Ensemble` 비교 | **채택**: WESAD 15명 LOSO AUROC `0.9439` $\rightarrow$ **`0.9634`** (`LR`), PhysioNet 17명 AUROC `0.8862` $\rightarrow$ **`0.9191`** (`RF`) |
| **H5** | 재학습 없이 기존 동결 모델의 출력 확률을 초기 기저선 로그-오즈(Logit) 기준으로 정렬(CBARA)하면 동결 모델의 Recall 붕괴도 복구할 수 있다. | `apply_cbara_probability_calibration` 적용 | **채택**: `Frozen_lr_all` (`Tag_Aligned`) Recall **`0.6086`**, F1 **`0.6703`**, Baseline FPR `0.3863` $\rightarrow$ **`0.1237`**, Exercise FPR `0.9366` $\rightarrow$ **`0.0127`** |

---

## 3. 실험별 성능 비교 ([실험 확인 사실])

### 3.1 내부 검증: WESAD 15명 Leave-One-Subject-Out (LOSO) 교차검증
- 원본 파일: [`results/phase4/wesad_15sub_loso_comparison.csv`](file:///D:/Autism/results/phase4/wesad_15sub_loso_comparison.csv)

| 모델 아키텍처 | 특징 수 | 15명 평균 AUROC ($\pm$SD) | 15명 평균 AP | Stress Recall | Stress F1 | Baseline FPR | 최악 피험자 AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 기존 `lr_all` (Baseline) | 49 | 0.9519 ($\pm$0.071) | 0.9048 | 0.7810 | 0.7425 | 0.1629 | 0.7500 |
| 기존 `rf_all` (Baseline) | 49 | 0.9065 ($\pm$0.112) | 0.8586 | 0.6420 | 0.6142 | 0.3132 | 0.6120 |
| `LR_Calib_Physio` | 9 | 0.9439 ($\pm$0.099) | 0.8993 | 0.8458 | 0.7532 | 0.1380 | 0.6265 |
| **`LR_Calib_Temporal`** | **24** | **0.9634 ($\pm$0.063)** | **0.9353** | **0.8502** | **0.7621** | **0.1304** | **0.7841** |
| `RF_Calib_Physio` | 9 | 0.9418 ($\pm$0.084) | 0.9002 | 0.7865 | 0.7204 | 0.1096 | 0.7732 |
| **`RF_Calib_Temporal`** | **24** | **0.9453 ($\pm$0.079)** | **0.9043** | **0.7836** | **0.7253** | **0.0962** | **0.7712** |
| `LGBM_Calib_Temporal` | 24 | 0.9221 ($\pm$0.106) | 0.8562 | 0.7820 | 0.7101 | 0.1116 | 0.6964 |
| `HGB_Calib_Temporal` | 24 | 0.9218 ($\pm$0.105) | 0.8565 | 0.7687 | 0.7015 | 0.1019 | 0.7056 |
| **`RF_Calib_Temporal_RotInvACC`** | **34** | **0.9813 ($\pm$0.036)** | **0.9593** | **0.8173** | **0.8068** | **0.0559** | **0.8820** |

### 3.2 과제 A (Motion Gate Ablation): 게이팅 방식별 스트레스 탐지율 vs 운동 오경보율 동시 비교
- 평가 조건: PhysioNet 17명 STRESS + 14명 AEROBIC (`Tag_Aligned_Official`, 모델: `Ensemble_Calib_Temporal` 및 `RF_Calib_Temporal`)

| 모델 | 게이팅 방식 | 17명 평균 AUROC | 창 단위 Stress Recall | 과제(Event) Recall | 제스처 동반 스트레스 Recall (41창) | 기저 상태 FPR (에피소드 FAR/hr) | 유산소 운동 FPR (에피소드 FAR/hr) |
|---|---|---:|---:|---:|---:|---:|---:|
| `RF_Calib_Temporal` | `none` (게이트 미적용) | 0.9203 | 0.6858 | 0.7255 | 0.8293 | 0.1196 (8.59/hr) | 0.7300 (2.59/hr) |
| `RF_Calib_Temporal` | `legacy_hard` (`b46a2e9` 하드 게이트) | 0.7521 | **0.3425** (-50% 손실) | 0.4314 | **0.0000** (전원 소거) | 0.0882 (6.00/hr) | 0.0000 (0.00/hr) |
| `RF_Calib_Temporal` | `exercise_hard` (지속 운동 하드 게이트) | 0.9109 | 0.6711 | 0.7255 | 0.7805 | 0.1196 (8.59/hr) | 0.0063 (0.46/hr) |
| **`RF_Calib_Temporal`** | **`soft_dynamic` (제안 소프트 동적 게이트)** | **0.9191** | **0.6809** (손실 0.5%p) | **0.7255** | **0.8049** | **0.1196 (8.59/hr)** | **0.0127 (0.46/hr)** |
| `Ensemble_Calib_Temporal` | `none` (게이트 미적용) | 0.9203 | 0.6440 | 0.7255 | 0.7561 | 0.0216 (2.59/hr) | 0.7034 (2.59/hr) |
| `Ensemble_Calib_Temporal` | `legacy_hard` (`b46a2e9` 하드 게이트) | 0.6975 | **0.3167** (-51% 손실) | 0.4118 | **0.0000** (전원 소거) | 0.0118 (1.41/hr) | 0.0000 (0.00/hr) |
| **`Ensemble_Calib_Temporal`** | **`soft_dynamic` (제안 소프트 동적 게이트)** | **0.9156** | **0.6274** | **0.7059** | **0.7561** | **0.0216 (2.59/hr)** | **0.0127 (0.46/hr)** |

### 3.3 과제 B (개인별 생리 보정): 반응군(Subgroup)별 성능 및 온라인 기준선 갱신 비교
- 피험자 17명을 초기 기저 피부전도도(`eda_base_uS`)에 따라 3개 군으로 분류하여 비교한 결과:
  - **저반응군 (Low-Responder, $< 0.6\,\mu\text{S}$, 7명: `S01, S04, S07, S08, S09, S11, S15`)**
  - **중간 반응군 (Medium-Responder, $0.6\sim 4.0\,\mu\text{S}$, 4명: `S03, S06, S13, S16`)**
  - **고기저군 (High-Tonic, $> 4.0\,\mu\text{S}$, 6명: `S05, S10, S12, S14, S17, S18`)**

| 파이프라인 | 저반응군 평균 AUROC (7명) | 중간군 평균 AUROC (4명) | 고기저군 평균 AUROC (6명) | 전체 17명 평균 AUROC |
|---|---:|---:|---:|---:|
| 기존 `Frozen_rf_all` (`legacy_hard`) | 0.7609 | 0.4786 | **0.2928** (붕괴) | 0.5293 |
| 기존 `Frozen_lr_eda` (`Cold-Start`) | 0.7838 | 0.8161 | **0.4872** (`S05, S10` 0.02) | 0.6867 |
| 제안 `LR_Calib_Physio` (`soft_dynamic`) | 0.8192 | 0.9444 | **0.6814** | 0.8000 |
| 제안 `LGBM_Calib_Temporal` (`soft_dynamic`) | **0.9971** | **0.9500** | **0.7875** | **0.9121** |
| 제안 `RF_Calib_Temporal` (고정 초기 기저선) | **0.9914** | **1.0000** | **0.7808** | **0.9191** |
| 제안 `RF_Calib_Temporal` (인과 온라인 갱신 $\alpha=0.05$) | **0.9914** | **1.0000** | **0.7762** | **0.9175** |
| 제안 **`Ensemble_Calib_Temporal`** (`soft_dynamic`) | **0.9886** | **0.9839** | **0.7850** | **0.9156** |

### 3.4 과제 C (센서별 기여도 및 온디바이스 추론 비용 분석)

| 센서 조합 / 모델 | WESAD 15명 LOSO AUROC | PhysioNet 17명 AUROC (`Tag_Aligned`) | PhysioNet 17명 AUROC (`Strict_PostCal`) | 모델 파라미터 용량 | PC 단일 창 추론 지연시간 (ms/window) |
|---|---:|---:|---:|---:|---:|
| `Ablation_EDA_Only` (보정 EDA 단독, LR) | 0.9547 | 0.6070 | 0.6769 | **1.2 KB** | **0.0086 ms** (`8.6 us`) |
| `Ablation_PPG_Only` (보정 PPG/HR 단독, LR) | 0.7631 | 0.6844 | 0.6418 | **0.8 KB** | **0.0073 ms** (`7.3 us`) |
| `Ablation_ACC_Only` (회전불변 ACC 단독, LR) | 0.9257 | 0.6088 | 0.6228 | **1.0 KB** | **0.0104 ms** (`10.4 us`) |
| **`LR_Calib_Physio` (EDA + PPG 보정, 9특징)** | 0.9439 | **0.8000** | **0.7995** | **0.9 KB** | **0.0069 ms** (`6.9 us`) |
| **`LGBM_Calib_Temporal` (EDA + PPG + EMA, 24특징)** | 0.9221 | **0.9121** | **0.8381** | **48.5 KB** | **0.0165 ms** (`16.5 us`) |
| **`RF_Calib_Temporal` (EDA + PPG + EMA, 24특징)** | 0.9453 | **0.9191** | **0.8099** | **215.0 KB** | **0.1036 ms** (`103.6 us`) |
| **`Ensemble_Calib_Temporal` (LR + RF + LGBM)** | **0.9582** | **0.9156** | **0.8154** | **265.0 KB** | **0.0450 ms** (`45.0 us`) |

> **핵심 발견 (센서 상보성)**: 외부 데이터셋(`PhysioNet`)에서는 EDA 단독(`AUROC 0.6070~0.6769`)이나 PPG 단독(`0.6418~0.6844`)보다 **보정된 EDA + PPG 결합(`LR_Calib_Physio 0.8000`, `LGBM_Calib_Temporal 0.9121`)**에서 큰 폭의 시너지(+0.12~+0.23 AUROC)가 발생한다. 이는 EDA 저반응군 피험자(`S01, S09`)에서는 심박수 상승($\Delta \text{HR}$)이 스트레스를 포착하고, 반대로 심박 변화가 작은 인지 과제에서는 SCR/EDA 상승이 이를 보완하기 때문이다.

---

## 4. 최악 참가자(Worst-Subject) 실패 원인 정밀 분석 ([실험 확인 사실])

1. **`S16` 및 `S17` (기존 `Legacy_Segments`에서 AUROC `0.13~0.29` $\rightarrow$ `Tag_Aligned`에서 `1.0000` / `0.8889`로 완전 해결)**:
   - **원인**: `S16`의 첫 번째 태그(`tag[0]`, 실제 기저선 시작)는 녹화 시작 후 `335초`에 찍혔다. 따라서 기존 `Legacy_Segments`의 `Baseline=[0, 300]`은 전 구간이 착용 준비 구간(`HR = 81.3 bpm`, `acc_jerk = 112.4`)이었고, `Stroop=[300, 600]` 구간이 실제로는 `S16`이 가장 편안하게 쉬던 실제 기저선(`tag[0]..tag[1] = [335, 535]`, `HR = 68.4 bpm`)이었다. 즉 **정답 라벨 자체가 반대로 부여되어 있었던 것**이며, `tags.csv` 정합 즉시 `S16`의 AUROC는 **`1.0000`**으로 복구되었다.
2. **`S14` (`Tag_Aligned` AUROC `0.5417~0.7292`) 및 `S18` (`Tag_Aligned` AUROC `0.6000~0.6444`) — 남은 최악 피험자**:
   - **원인**: 두 피험자 모두 **초기 기저선 구간(`tag[0]..tag[1]`)에서 이미 피부전도도가 비정상적으로 높았던 고기저 피험자**(`S14`: 기저 $4.26\,\mu\text{S}$, `S18`: 기저 $9.83\,\mu\text{S}$)로, 실험 시작 직후 예기 불안(Anticipatory arousal) 상태에서 기저선이 측정된 후 본 과제(`Stroop`, `TMCT`) 후반부에 오히려 피부전도도가 서서히 감소(Habituation drift)하는 비전형적 궤적을 보였다.
   - **완화 확인**: 그럼에도 기존 `b46a2e9` 모델(`S18` AUROC `0.1333`, Recall `0.0%`)과 달리, 제안된 `Ensemble_Calib_Temporal` 및 `RF_Calib_Temporal`은 BVP 심박수 상승과 초기 SCR 빈도를 결합하여 `S18`에서도 무작위 이상(`AUROC 0.6000~0.6444`, `Recall 33.3%`, `Baseline FPR 0.0%`)의 성능을 방어하였다.

---

## 5. 최종 채택 모델 및 선택 근거

실제 손목형 웨어러블 시스템의 하드웨어 제약과 운용 목적에 따라 **2종의 최종 채택 아키텍처**를 선정한다:

1. **주력 고정밀 모델 (표준 웨어러블/스마트워치 및 512KB 이상 MCU용): `Ensemble_Calib_Temporal` + `soft_dynamic` Motion Gate**
   - **구성**: 인과적 개인 기저선 보정(9특징) + 인과적 EMA/차분(15특징)을 입력으로 받는 `LR + RF + LGBM` 소프트 보팅 앙상블과 2-창 인과 운동 점수 $M(t)$ 기반 소프트 동적 게이트 결합.
   - **선택 근거**:
     - PhysioNet 17명 외부 검증 평균 AUROC **`0.9156`** (중앙값 **`1.0000`**), 평균 AP **`0.9562`**, 최악 피험자(`S18`)에서도 AUROC **`0.6000`** 이상을 보장(전 모델 중 최고 최저점 방어력).
     - 기저 상태 오경보율(Baseline FPR) **`2.16%`** (시간당 에피소드 오경보 `2.59회/hr`), 정밀도(Precision) **`97.55%`**, 스트레스 Recall **`62.74%`** (과제 단위 Event Recall **`70.59%`**).
     - 14명 유산소 운동 중 오경보율(Exercise FPR) **`1.27%`** (`0.46회/hr`)로 기존 Cold-Start(`93.66%`)의 운동 오탐을 제거하면서 기존 하드 게이트(`Recall 0.25%`)의 스트레스 소거 문제를 완전히 해결.
2. **초경량 온디바이스 단독 모델 (초저전력 MCU 펌웨어용): `LGBM_Calib_Temporal` (48.5 KB) 또는 `LR_Calib_Physio` (0.9 KB) + `soft_dynamic` Motion Gate**
   - 메모리 64KB 이하 초소형 MCU에서는 `LR_Calib_Physio` (가중치 **0.9 KB**, 추론 지연 **6.9 $\mu\text{s}$**, `Tag_Aligned` AUROC **`0.8000`**, `Strict_PostCal` AUROC **`0.7995`**, Baseline FPR **`2.35%`**)를 탑재하고, 128KB 이상에서는 `LGBM_Calib_Temporal` (**48.5 KB**, 추론 지연 **16.5 $\mu\text{s}$**, `Tag_Aligned` AUROC **`0.9121`**, `Strict_PostCal` AUROC **`0.8381`**)을 단독 탑재한다.

---

## 6. 미해결 문제와 한계

1. **초기 기저선 오염(Anticipatory Baseline Contamination) 취약성**:
   - `S14`, `S18`과 같이 착용 직후 초기 3~5분 기저선 구간 자체에서 이미 긴장하거나 직전 활동의 잔여 교감신경 흥분이 남아 있는 경우, 상대 변화량($\Delta$)이 과소평가되어 민감도가 낮아진다 (`Recall 33.3%~37.5%`). 이를 해결하려면 단일 세션 초기 5분뿐 아니라 수일간의 착용 이력에서 수면 직후 또는 최저 안정기 분포(5th percentile tonic floor)를 장기 기준선으로 삼는 다중일(Multi-day) 적응형 보정이 필요하다.
2. **고강도 유산소 운동 중 동시 발생하는 실제 스트레스 탐지의 물리적 한계**:
   - 본 연구의 시뮬레이션([`task_a_stress_under_motion_summary.json`](file:///D:/Autism/results/phase4/task_a_stress_under_motion_summary.json))에서 확인되었듯, 가벼운 손동작/발화가 수반된 스트레스는 `80.49%` 탐지할 수 있으나, **자전거 전력 질주 수준의 지속적 유산소 운동(`M(t) > 0.85`) 중에는 운동 자체에 의한 발한(Thermoregulatory sweating)과 빈맥(Exercise tachycardia)이 교감신경 스트레스 반응과 생리학적으로 중첩**되므로 손목 센서만으로는 운동 오경보를 늘리지 않고 운동 중 스트레스를 분리하는 데 근본적 한계가 있다.
3. **성인 스트레스 데이터와 자폐 아동 멜트다운 간의 도메인 격차**:
   - WESAD와 PhysioNet은 모두 성인 피험자의 인지/사회적 스트레스 과제 데이터이므로, 감각 과부하와 상동행동(Stimming)이 빈번한 자폐 아동의 멜트다운 전조를 직접 대변하지 못한다.

---

## 7. 멜트다운 사전 예측 가능성 조사 요약 ([문헌 근거 내용] 및 프로토콜 설계)

(상세 내용은 [`reports/ASD_EVENT_FORECASTING.md`](file:///D:/Autism/reports/ASD_EVENT_FORECASTING.md) 참조)

1. **선행 연구 검증 수준**: Goodwin et al. (2019) 및 Imbiriba et al. (2023)은 입원 ASD 아동 20~70명의 Empatica E4 신호로 공격 행동 및 정서조절실패(ED)를 1~3분 전 AUROC `0.71~0.84` 수준으로 예측 가능함을 보였다.
2. **시간 구간별 실현 가능성**:
   - **1분 전 ($L=60\text{s}$)**: 급성 빈맥과 연속 SCR이 뚜렷하여 실현 가능성 높음.
   - **3분 전 ($L=180\text{s}$)**: 부교감신경 철회(RMSSD 하락)와 Tonic SCL 누적 기울기(`Causal EMA`)를 통해 중재 골든타임 확보 가능(핵심 목표).
   - **5분 전 ($L=300\text{s}$)**: 단독 경보 시 오경보 위험이 높으므로 1단계 주의 예보(Yellow Advisory)로 설계하고 3분/1분 전 경보와 계층화해야 함.
3. **상동행동(Stimming) 분리 원칙**: 자기 진정 목적의 Stimming은 좁은 대역($2\sim 5\text{ Hz}$)의 주기적 ACC 진동을 보이나 EDA/HR 폭주가 없으므로, ACC 특징을 직접 양성 가중치로 쓰지 않고 **개인 보정된 자율신경 폭주($\Delta \log \text{EDA} + \Delta \text{HR}$)가 동반될 때만** 전조 경보를 발령해야 한다.

---

## 8. 웨어러블 제품 적용을 위한 후속 연구 계획 (Phased Roadmap)

| 단계 | 목표 과제 | 구체적 실행 계획 및 성공 기준 (KPI) |
|:---:|---|---|
| **Phase 5A<br>(단기: 1~2개월)** | **다중일(Multi-Day) 최저 안정기 기준선 자동 학습 및 온디바이스 C 코드 포팅** | • 초기 5분 착용 구간의 예기 불안(`S14, S18` 유형)을 극복하기 위해 `acc_unchanged_fraction > 0.50`인 정적 휴식 구간의 하위 10% 분위수(10th percentile)를 장기 기저선으로 추적하는 알고리즘 구현<br>• `LGBM_Calib_Temporal`(48.5KB) 및 `LR_Calib_Physio`(0.9KB) + `soft_dynamic` 게이트를 ESP32-S3 C/C++ 펌웨어로 변환하여 실측 소비전력 및 타이머 지연 검증 |
| **Phase 5B<br>(중기: 3~6개월)** | **기관 협약(DUA) 기반 자폐 아동 실데이터 검증** | • [`ASD_DATA_ACCESS_APPLICATION_PACKAGE.md`](file:///D:/Autism/reports/ASD_DATA_ACCESS_APPLICATION_PACKAGE.md)를 통해 Northeastern Univ. (Goodwin/Imbiriba 코호트) 및 Kartha et al. (2026) 연구진과 DUA 체결<br>• [`ASD_EVENT_FORECASTING.md`](file:///D:/Autism/reports/ASD_EVENT_FORECASTING.md)에 사전 등록된 $L \in \{60\text{s}, 180\text{s}, 300\text{s}\}$ 프로토콜로 사건 단위 탐지율(Event Recall $\ge 70\%$) 및 일일 오경보($\le 1.5\text{회/day}$) 검증 |
| **Phase 5C<br>(장기: 6~12개월)** | **자폐 아동 대상 전향적 일상 관찰 임상시험 (IRB 승인)** | • 특수학교·가정 환경에서 자폐 아동 20~30명을 대상으로 손목 웨어러블 + 보호자 스마트워치 무음 햅틱 알림 실증<br>• 상동행동(Stimming), 체육 시간, 감각 과부하 구간에 대한 다중 상태(`infer_joint_4state`) 실생활 분류 정확도 및 보호자 개입 효과 검증 |
