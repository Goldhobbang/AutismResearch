# Nurses 고정 30초 전체 시간축 조기경고 재평가

작성일: 2026-10-26. 원본 센서·설문·기존 특징 캐시·모델·감사 산출물은 수정하지 않았다. 신규 산출물은 `results/nurse_forecasting/fixed_grid_30s/`에만 저장했다.

## 0. 결론 요약

- 기존 라벨 의존 30/60초 격자를 **전체 세션 고정 30초 격자**로 바꾸고, 동일한 62개 특징·과거 180/60/300초 관측 창·동일 LR LOSO 절차로 다시 평가했다.
- **경고 활성(active)** 과 **사건 60~180초 전 신규 발령(issued)** 을 분리했다. 두 값은 같은 모델에서도 크게 다르며, 활성 값을 신규 조기경고로 인용하면 안 된다.
- 운동 게이트는 이전 감사와 같은 방향으로 **고활동 오경보를 줄이면서 진양성 경고도 함께 제거**했다. 게이트 후 성능 개선으로 해석하지 않는다.
- 검증된 음성(`CONFIRMED_NO_STRESS_0`)과 미검증 배경(`VERIFIED_NON_EVENT`)을 분리했다. 미검증 배경이 평가 음성의 대부분을 차지하므로 FA/h와 AUROC는 음성 출처별로 함께 읽어야 한다.
- **고정 30초 격자는 사건 주변 창을 바꾸지 않았다.** 기존 격자도 사건 주변은 이미 30초였고, 신규 캐시는 배경만 60초에서 30초로 촘촘하게 추가했다. 따라서 event-level 변화는 격자 자체가 아니라 재적합 점수·임계값 변화에서 왔다.
- 고정 격자 재평가 결과는 **성능 개선을 보장하지 않는다.** 아래 표의 실제 측정값만 보고한다.

## 1. 신규 고정 30초 특징 캐시

- 세션: 526/609 처리, 오류 0건, 185초 미만 제외 83건. 평가는 설문 주석 214세션만 사용.
- 전체 창: 147,015; 설문 주석 세션 창: 71,274.
- P1 양성: 659; 검증 음성: 2,173; 미검증 배경: 44,519.
- 고정 간격: 30초, 주 관측 창: 180초, 특징 수: 62.
- 개인 보정은 state를 숨긴 label-blind 방식으로 적용했다. 기존 캐시의 보정 참조와 동일한 low-motion/첫 창 규칙을 사용했다.
- 공유 timestamp 51,234개에서 **원시 53개 특징은 최대 절대차 0으로 동일**했다. 9개 개인보정 특징은 기존 격자의 보정 참조 두 번째 창이 배경에서 60초 간격(180/240초)이던 것이 고정 30초(180/210초)로 바뀌어 차이가 생겼다. 이는 원시 특징 추출 차이가 아니라 보정 참조 창의 격자 차이다.
- 캐시 SHA256: `7aedd60ae57b17e656fff112cda7ab1c52f63c1f3194496d082f4a9cc59498d1`.

## 2. LOSO 재학습 프로토콜

- 15-fold LOSO, 각 fold 내부 train/validation 참가자 분리. 저장된 full-refit joblib 모델은 held-out 평가에 사용하지 않았다.
- 기존과 동일한 `LogisticRegression(C=0.25, class_weight='balanced')` + median imputer + StandardScaler. 새 알고리즘·하이퍼파라미터 탐색은 하지 않았다.
- 임계값은 각 fold의 내부 validation 참가자 F1 절차로만 선택했다. 테스트 참가자 라벨은 임계값·전처리 적합에 사용하지 않았다.
- fold 수: 15, 평균 학습 창: 34960, 평균 validation F1: 0.058.

새 캐시에서 기존 LOSO 점수를 그대로 적용할 수는 없었다. 기존 저장 점수는 라벨 의존 격자에서만 존재하고, fold별 scaler/imputer 파라미터도 저장되어 있지 않기 때문이다. 따라서 **동일 모델 명세를 재적용**했으며, 이는 새 모델 탐색이 아니다.

## 3. 동일 평가 정의

- P1 양성: 사건 시작 (60s, 180s] before survey-confirmed onset.
- Active endpoint: any warning-active timestamp inside (onset-180s, onset-60s].
- Issued endpoint: first timestamp of a merged alert episode inside the same horizon.
- 오경보: unassigned episode with at least one evaluated negative window (active) or a negative first issuance (issued).
- Event Recall은 원 적격 사건 분모를 유지하고, 관측 가능 분모도 함께 표기했다. Precision은 사건-에피소드 일대일 배정 후 계산했다.
- FA/h 분모는 실제 창 노출 시간 `[t, next_t)`을 세션 종료로 자른 값이다.

## 4. 창 수준 AUROC/AUPRC

| 스트림 | 게이트 | 창 | 양성 | 음성 | 양성률 | macro AUROC | macro AUPRC | pooled AUROC | pooled AUPRC |
|---|---|---|---|---|---|---|---|---|---|
| old_labelblind | none | 27250 | 659 | 26591 | 0.024 | 0.617 | 0.065 | 0.664 | 0.044 |
| old_labelblind | legacy_hard | 27250 | 659 | 26591 | 0.024 | 0.488 | 0.027 | 0.490 | 0.024 |
| old_labelblind | soft_dynamic | 27250 | 659 | 26591 | 0.024 | 0.518 | 0.048 | 0.559 | 0.033 |
| old_legacy | none | 27250 | 659 | 26591 | 0.024 | 0.617 | 0.066 | 0.664 | 0.044 |
| old_legacy | legacy_hard | 27250 | 659 | 26591 | 0.024 | 0.488 | 0.027 | 0.490 | 0.024 |
| old_legacy | soft_dynamic | 27250 | 659 | 26591 | 0.024 | 0.518 | 0.048 | 0.559 | 0.033 |
| new_fixed30s | none | 47236 | 659 | 46577 | 0.014 | 0.621 | 0.046 | 0.672 | 0.026 |
| new_fixed30s | legacy_hard | 47236 | 659 | 46577 | 0.014 | 0.489 | 0.016 | 0.491 | 0.014 |
| new_fixed30s | soft_dynamic | 47236 | 659 | 46577 | 0.014 | 0.526 | 0.033 | 0.569 | 0.020 |

## 5. 경고 활성 vs 신규 발령 (hysteresis)

| 스트림 | 게이트 | Event Recall(active) | Event Recall(issued) | Precision(active) | Precision(issued) | FA/h(active) | FA/h(issued) |
|---|---|---|---|---|---|---|---|
| new_fixed30s | legacy_hard | 0.006 | 0.000 | 0.053 | 0.000 | 0.046 | 0.049 |
| new_fixed30s | none | 0.472 | 0.068 | 0.168 | 0.031 | 0.970 | 0.957 |
| new_fixed30s | soft_dynamic | 0.216 | 0.051 | 0.116 | 0.032 | 0.727 | 0.691 |
| old_labelblind | legacy_hard | 0.006 | 0.000 | 0.056 | 0.000 | 0.044 | 0.046 |
| old_labelblind | none | 0.534 | 0.085 | 0.197 | 0.040 | 0.926 | 0.918 |
| old_labelblind | soft_dynamic | 0.239 | 0.051 | 0.124 | 0.032 | 0.748 | 0.712 |
| old_legacy | legacy_hard | 0.006 | 0.000 | 0.056 | 0.000 | 0.044 | 0.046 |
| old_legacy | none | 0.523 | 0.080 | 0.199 | 0.039 | 0.895 | 0.887 |
| old_legacy | soft_dynamic | 0.239 | 0.057 | 0.127 | 0.036 | 0.730 | 0.696 |

공유 정의로 다시 계산한 기존 labelblind 스트림의 active는 94/176, issued는 15/176이다. 기존 감사 보고서의 full-stream hysteresis 90/176은 그 구현의 episode/state 처리 차이이며, 본 표의 old/new 비교는 **동일한 함수**로 계산한 값이다.

### 검증 음성 / 미검증 배경 분리

| 스트림 | 게이트 | FA/h 검증(active) | FA/h 미검증(active) | FA/h 검증(issued) | FA/h 미검증(issued) |
|---|---|---|---|---|---|
| new_fixed30s | legacy_hard | 0.055 | 0.046 | 0.055 | 0.049 |
| new_fixed30s | none | 1.768 | 0.990 | 1.050 | 0.952 |
| new_fixed30s | soft_dynamic | 1.436 | 0.712 | 1.270 | 0.663 |
| old_labelblind | legacy_hard | 0.055 | 0.043 | 0.055 | 0.046 |
| old_labelblind | none | 2.099 | 0.933 | 1.270 | 0.901 |
| old_labelblind | soft_dynamic | 1.823 | 0.722 | 1.547 | 0.671 |
| old_legacy | legacy_hard | 0.055 | 0.043 | 0.055 | 0.046 |
| old_legacy | none | 1.933 | 0.904 | 1.160 | 0.874 |
| old_legacy | soft_dynamic | 1.657 | 0.706 | 1.436 | 0.660 |

### 음성 출처별 창 수준 구분력

| 스트림 | 게이트 | macro AUROC 검증 | macro AUROC 미검증 | pooled AUROC 검증 | pooled AUROC 미검증 | macro AUPRC 검증 | macro AUPRC 미검증 |
|---|---|---|---|---|---|---|---|
| old_labelblind | none | 0.503 | 0.642 | 0.577 | 0.672 | 0.380 | 0.138 |
| old_labelblind | legacy_hard | 0.498 | 0.488 | 0.488 | 0.490 | 0.354 | 0.064 |
| old_labelblind | soft_dynamic | 0.469 | 0.539 | 0.486 | 0.565 | 0.379 | 0.117 |
| old_legacy | none | 0.502 | 0.642 | 0.577 | 0.671 | 0.380 | 0.139 |
| old_legacy | legacy_hard | 0.498 | 0.488 | 0.488 | 0.490 | 0.354 | 0.064 |
| old_legacy | soft_dynamic | 0.469 | 0.539 | 0.486 | 0.565 | 0.379 | 0.117 |
| new_fixed30s | none | 0.493 | 0.647 | 0.580 | 0.676 | 0.382 | 0.115 |
| new_fixed30s | legacy_hard | 0.499 | 0.490 | 0.490 | 0.491 | 0.354 | 0.052 |
| new_fixed30s | soft_dynamic | 0.468 | 0.548 | 0.491 | 0.573 | 0.375 | 0.100 |

검증 음성만 쓰면 참가자별 양성/음성 양쪽이 정의되는 참가자가 줄어든다. 정의 불가 값은 0으로 채우지 않았다.

## 6. 운동 게이트 전후

동일 확률·동일 fold 임계값·동일 대상 창에서 게이트만 바꿨다. `legacy_hard`는 고활동에서 확률을 0.05로 누르고, `soft_dynamic`은 연속 운동 점수로 확률과 임계값을 동시에 조정한다.

| 게이트 | endpoint | 탐지 사건 | Event Recall | Precision | FA/h 전체 | FA/h 검증 | FA/h 미검증 |
|---|---|---|---|---|---|---|---|
| none | active | 83 | 0.472 | 0.168 | 0.970 | 1.768 | 0.990 |
| none | issued | 12 | 0.068 | 0.031 | 0.957 | 1.050 | 0.952 |
| legacy_hard | active | 1 | 0.006 | 0.053 | 0.046 | 0.055 | 0.046 |
| legacy_hard | issued | 0 | 0.000 | 0.000 | 0.049 | 0.055 | 0.049 |
| soft_dynamic | active | 38 | 0.216 | 0.116 | 0.727 | 1.436 | 0.712 |
| soft_dynamic | issued | 9 | 0.051 | 0.032 | 0.691 | 1.270 | 0.663 |

게이트가 오경보를 줄여도 동시에 진양성 경고를 제거하면 조기경고 성능 개선이 아니다. 이번 결과도 그 방향을 재현했다.

## 7. 기존 결과와 차이의 원인

### 7.1 격자 구성 차이

| 스트림 | 창 | P1 양성 | 검증 음성 | 미검증 음성 | 검열/기타 |
|---|---|---|---|---|---|
| old_grid | 51234 | 659 | 2173 | 24479 | 23923 |
| new_grid | 71274 | 659 | 2173 | 44519 | 23923 |
| old_grid_subset_of_new | 51234 | 659 | 2173 | 24479 | 23923 |

기존 키 51,234개는 모두 신규 30초 격자에 포함된다. 차이는 **미검증 배경 24,479→44,519창(+20,040)** 이며 P1 양성·검증 음성·검열 창 수는 동일하다.

### 7.2 점수 효과와 격자 효과 분리

`old_grid_new_score`는 신규 재적합 점수·임계값을 기존 51,234창에 적용한 것이다. `new_grid_new_score`는 여기에 추가된 30초 배경까지 포함한다.

| 스트림 | 게이트 | 창 | 양성 | 음성 | 양성률 | macro AUROC | macro AUPRC | pooled AUROC | pooled AUPRC |
|---|---|---|---|---|---|---|---|---|---|
| old_grid_old_score | none | 27250 | 659 | 26591 | 0.024 | 0.617 | 0.065 | 0.664 | 0.044 |
| old_grid_old_score | soft_dynamic | 27250 | 659 | 26591 | 0.024 | 0.518 | 0.048 | 0.559 | 0.033 |
| old_grid_new_score | none | 27250 | 659 | 26591 | 0.024 | 0.616 | 0.066 | 0.664 | 0.044 |
| old_grid_new_score | soft_dynamic | 27250 | 659 | 26591 | 0.024 | 0.521 | 0.048 | 0.562 | 0.033 |
| new_grid_new_score | none | 47236 | 659 | 46577 | 0.014 | 0.621 | 0.046 | 0.672 | 0.026 |
| new_grid_new_score | soft_dynamic | 47236 | 659 | 46577 | 0.014 | 0.526 | 0.033 | 0.569 | 0.020 |

| 스트림 | 게이트 | endpoint | 탐지 사건 | Event Recall | Precision | FA/h 전체 | FA/h 검증 | FA/h 미검증 |
|---|---|---|---|---|---|---|---|---|
| old_grid_old_score | none | active | 94 | 0.534 | 0.197 | 0.926 | 2.099 | 0.933 |
| old_grid_old_score | none | issued | 15 | 0.085 | 0.040 | 0.918 | 1.270 | 0.901 |
| old_grid_old_score | soft_dynamic | active | 42 | 0.239 | 0.124 | 0.748 | 1.823 | 0.722 |
| old_grid_old_score | soft_dynamic | issued | 9 | 0.051 | 0.032 | 0.712 | 1.547 | 0.671 |
| old_grid_new_score | none | active | 83 | 0.472 | 0.180 | 0.895 | 1.768 | 0.912 |
| old_grid_new_score | none | issued | 13 | 0.074 | 0.037 | 0.879 | 1.050 | 0.871 |
| old_grid_new_score | soft_dynamic | active | 38 | 0.216 | 0.125 | 0.665 | 1.436 | 0.647 |
| old_grid_new_score | soft_dynamic | issued | 10 | 0.057 | 0.040 | 0.627 | 1.270 | 0.595 |
| new_grid_new_score | none | active | 83 | 0.472 | 0.168 | 0.970 | 1.768 | 0.990 |
| new_grid_new_score | none | issued | 12 | 0.068 | 0.031 | 0.957 | 1.050 | 0.952 |
| new_grid_new_score | soft_dynamic | active | 38 | 0.216 | 0.116 | 0.727 | 1.436 | 0.712 |
| new_grid_new_score | soft_dynamic | issued | 9 | 0.051 | 0.032 | 0.691 | 1.270 | 0.663 |

핵심 분해:

- 창 macro AUROC는 old 0.6168 → 신규 점수/기존 격자 0.6161 → 신규 전체 0.6214였다. **+0.0053의 AUROC 상승은 모델 개선이 아니라 음성 배경이 늘어난 격자 효과**다.
- Active Event Recall은 old 94/176 → 신규 점수/기존 격자 83/176 → 신규 전체 83/176이었다. **-11사건은 재적합 점수·임계값 변화에서 발생했고, 고정 30초 격자 추가는 event-level 탐지를 바꾸지 않았다** (사건 주변 창이 이미 30초였기 때문).
- Issued Event Recall은 old 15 → 13 → 12로, 점수 효과 -2, 격자 효과 -1이었다.
- FA/h(전체)는 old 0.926 → 0.895 → 0.970으로, 점수 효과는 소폭 감소, 격자 효과는 미검증 배경 증가로 +0.075였다.

### 7.3 참가자별 임계값·탐지 변화

| 참가자 | 기존 임계값 | 신규 임계값 | 차이 | 기존 평균확률 | 신규 평균확률 | 동일창 확률상관 |
|---|---|---|---|---|---|---|
| 15 | 0.780 | 0.770 | -0.010 | 0.438 | 0.427 | 0.986 |
| 5C | 0.200 | 0.200 | 0.000 | 0.480 | 0.520 | 0.998 |
| 6B | 0.680 | 0.680 | 0.000 | 0.483 | 0.477 | 0.990 |
| 6D | 0.670 | 0.690 | 0.020 | 0.410 | 0.425 | 0.996 |
| 7A | 0.830 | 0.850 | 0.020 | 0.533 | 0.523 | 0.986 |
| 7E | 0.670 | 0.850 | 0.180 | 0.315 | 0.279 | 0.994 |
| 83 | 0.770 | 0.750 | -0.020 | 0.378 | 0.394 | 0.968 |
| 8B | 0.630 | 0.730 | 0.100 | 0.581 | 0.589 | 0.985 |
| 94 | 0.540 | 0.560 | 0.020 | 0.476 | 0.464 | 0.983 |
| BG | 0.660 | 0.700 | 0.040 | 0.416 | 0.413 | 0.982 |
| CE | 0.210 | 0.220 | 0.010 | 0.258 | 0.232 | 0.987 |
| DF | 0.780 | 0.780 | 0.000 | 0.353 | 0.363 | 0.988 |
| E4 | 0.800 | 0.840 | 0.040 | 0.605 | 0.582 | 0.990 |
| EG | 0.810 | 0.820 | 0.010 | 0.512 | 0.515 | 0.993 |
| F5 | 0.600 | 0.620 | 0.020 | 0.451 | 0.425 | 0.992 |

| 참가자 | 사건 | 기존 탐지 | 신규 탐지 | 차이 | 기존 임계값 | 신규 임계값 |
|---|---|---|---|---|---|---|
| 15 | 10 | 5 | 5 | 0 | 0.780 | 0.770 |
| 5C | 9 | 7 | 6 | -1 | 0.200 | 0.200 |
| 6B | 10 | 5 | 5 | 0 | 0.680 | 0.680 |
| 6D | 2 | 0 | 0 | 0 | 0.670 | 0.690 |
| 7A | 30 | 14 | 11 | -3 | 0.830 | 0.850 |
| 7E | 3 | 2 | 0 | -2 | 0.670 | 0.850 |
| 83 | 15 | 1 | 4 | 3 | 0.770 | 0.750 |
| 8B | 12 | 11 | 7 | -4 | 0.630 | 0.730 |
| 94 | 6 | 5 | 4 | -1 | 0.540 | 0.560 |
| BG | 13 | 4 | 3 | -1 | 0.660 | 0.700 |
| CE | 7 | 3 | 3 | 0 | 0.210 | 0.220 |
| DF | 6 | 1 | 1 | 0 | 0.780 | 0.780 |
| E4 | 24 | 19 | 17 | -2 | 0.800 | 0.840 |
| EG | 4 | 2 | 3 | 1 | 0.810 | 0.820 |
| F5 | 25 | 15 | 14 | -1 | 0.600 | 0.620 |

동일 timestamp에서 기존·신규 확률 상관은 대부분 0.98 이상이었다. 탐지 변화는 주로 내부 validation F1 임계값이 함께 움직인 결과다. 예: 7E 0.67→0.85, 8B 0.63→0.73, 7A 0.83→0.85에서 탐지가 줄고, 83은 0.77→0.75에서 늘었다.

- **표집 격자**: 기존 캐시는 사건 주변 30초, 배경 60초로 라벨에 의존해 표집됐다. 신규 캐시는 모든 세션을 30초로 균일 표집하므로 양성률, 배경 노출, 에피소드 병합 기회가 달라진다.
- **음성 구성**: 기존과 신규 모두 미검증 배경이 대부분이다. 미검증 배경만으로 계산한 AUROC/AUPRC와 검증 음성만으로 계산한 값은 같은 모델에서도 다르다.
- **재학습 차이**: 저장된 OOF 확률은 새 고정 격자에 존재하지 않는다. 동일 명세 재적합을 사용했으므로 기존 수치와의 차이는 격자·재적합·임계값 재선택이 함께 작용한 결과다.
- **정책 replay**: 두 스트림 모두 전체 시간 순서로 정책을 재생하고 QC 불량 창 입력을 0으로 두었다. 기존 filtered 평가와 직접 같은 값이 아니다.
- **사건 시각**: 설문 시각은 분 단위이며 독립 onset이 아니다. 따라서 이 비교는 실제 1~3분 전 예측 성능의 인과적 개선을 입증하지 않는다.

## 8. 검증

- 자동 테스트: `results/nurse_forecasting/fixed_grid_30s/test_results.txt` (27개 통과: 기존 20 + 신규 7).
- 원본/기존 산출물 SHA256: `protected_hashes_before.csv` vs `protected_hashes_after.csv`. 불변 여부: **True**.
- 15-fold LOSO 참가자 분리: 각 fold에서 test 참가자가 train/validation에 없고 train/validation 교집합이 0임을 스크립트에서 assert했다.
- 공유 51,234 timestamp에서 원시 53개 특징 최대 절대차 0.000 (`feature_equivalence.csv`).
- 신규 파일은 모두 `results/nurse_forecasting/fixed_grid_30s/` 아래에만 있다. 상세 검사: `verification.json`.

## 9. 재현

```powershell
.venv\Scripts\python.exe experiments/nurse_forecasting/fixed_grid_30s_replay.py snapshot
.venv\Scripts\python.exe experiments/nurse_forecasting/fixed_grid_30s_replay.py build
.venv\Scripts\python.exe experiments/nurse_forecasting/fixed_grid_30s_replay.py train
.venv\Scripts\python.exe experiments/nurse_forecasting/fixed_grid_30s_replay.py evaluate
.venv\Scripts\python.exe experiments/nurse_forecasting/fixed_grid_30s_replay.py analyze
.venv\Scripts\python.exe experiments/nurse_forecasting/fixed_grid_30s_replay.py report
.venv\Scripts\python.exe experiments/nurse_forecasting/fixed_grid_30s_replay.py verify
```

## 10. 한계

- 설문 라벨은 독립 onset이 아니며 분 단위다. 후보 생성과 사건 표집이 센서 알고리즘에 의존한다.
- 미검증 배경의 실제 무스트레스 여부는 확인되지 않았다. 검증 음성만으로는 참가자별 AUROC가 정의되지 않는 경우가 있다.
- 고정 30초 격자는 라벨 의존 표집을 제거하지만, 실제 온라인 배포에서의 지연·센서 드리프트·결측은 별도 검증이 필요하다.
- 참가자가 15명이므로 참가자 재표집 불확실성은 여전히 크다.
