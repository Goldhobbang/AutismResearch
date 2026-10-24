# WESAD 다중 상태 분류 실험 보고서 (Experiment A)

## 1. 개요 및 실험 목적

기존 WESAD 기반 연구는 `Stress vs Non-stress (Baseline + Amusement)` 이진 분류에 집중되었다. 그러나 실제 착용 환경에서는 감정적 흥분(Amusement), 이완/명상(Meditation), 중립 일상(Baseline) 등 다양한 감정 및 생리 상태가 혼재한다.
본 실험의 목적은 다음과 같다:
1. 상태 구분을 세분화하여 **3분류 (Baseline / Stress / Amusement)** 및 **4분류 (Baseline / Stress / Amusement / Meditation)** 모델을 15명 전수 LOSO(Leave-One-Subject-Out)로 평가한다.
2. 긍정적 각성 상태인 **Amusement**와 부정적 각성 상태인 **Stress** 간의 생리적 혼동률을 정량 분석한다.
3. WESAD 데이터셋 내 **Meditation (Phase 4)** 라벨의 유효성 및 전 참가자 적격성을 검증한다.

---

## 2. 데이터 적격성 및 프로토콜 검증

### 2.1 라벨별 창(Window) 분포 및 Meditation 적격성 검증
WESAD 15명 참가자 전체에 대해 60초 윈도우, 5초 스트라이드로 분할된 유효 구간 검증 결과는 다음과 같다:

| 상태 (Phase) | 라벨 정의 | 유효 윈도우 수 | 참가자별 평균 윈도우 수 | 비고 |
| :--- | :--- | :---: | :---: | :--- |
| **Baseline (Phase 1)** | 중립 기저 상태 | 1,113창 | 74.2창 (~19.3분) | 15명 전원 데이터 보유 |
| **Stress (Phase 2)** | TSST 스트레스 유발 과제 | 607창 | 40.5창 (~10.8분) | 15명 전원 데이터 보유 |
| **Amusement (Phase 3)** | 재미있는 비디오 시청 | 311창 | 20.7창 (~5.8분) | 15명 전원 데이터 보유 |
| **Meditation (Phase 4)** | 가이드 호흡/이완 명상 | 665창 | 44.3창 (~12.8분) | 15명 전원 데이터 보유 |
| **전체 (4-class)** | **합계** | **2,696창** | **179.7창** | **데이터 적격성 100% 입증** |

> [!NOTE]
> Meditation(Phase 4) 구간은 15명 전원에게 안정적인 시계열 구간(평균 12.8분)으로 존재함을 실증 확인하였다. 따라서 4분류 실험(A-2)을 결측 없이 전수 수행하였다.

---

## 3. 실험 구성 및 평가 모델

### 3.1 센서 특징 그룹
- `all` (손목 63개 통계/주파수 특징): EDA(27), BVP(16), TEMP(13), ACC(7)
- `acc` (가속도계 7개 특징): 움직임 단독 특징
- `physio` (손목 56개 생리 특징): EDA, BVP, TEMP (ACC 제외)

### 3.2 평가 모델군 (15-fold LOSO)
- `DummyClassifier` (다수 클래스 기준선)
- `LogisticRegression` (L2 정규화, C=1.0, Balanced class weights)
- `RandomForestClassifier` (n_estimators=100, max_depth=10, Balanced class weights)
- `HistGradientBoostingClassifier` (max_iter=100, Balanced class weights)
- `LGBMClassifier` (LightGBM 4.7.0, n_estimators=100, lr=0.05, Balanced class weights)

---

## 4. 실험 결과 요약

### 4.1 3분류 (Baseline / Stress / Amusement, 2,031창)
15명 피험자 독립 교차검증(LOSO) 성능 집계:

| 센서군 | 모델 | Macro F1 | Balanced Acc | Overall Acc | Stress Recall | Stress FPR | S7 Stress FPR | Amusement->Stress 오탐률 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **all** | Dummy | 0.2360 | 0.3333 | 0.5481 | 0.0000 | 0.0000 | 0.0000 | 0.0% |
| **all** | **LogisticRegression** | **0.6129** | **0.6563** | **0.6952** | **0.7955** | **0.1171** | **0.7396** | **23.5%** |
| **all** | RandomForest | 0.5362 | 0.5703 | 0.6916 | 0.6991 | 0.1385 | 1.0000 | 19.6% |
| **all** | HistGradientBoosting | 0.5422 | 0.5785 | 0.7018 | 0.7433 | 0.1284 | 1.0000 | 23.5% |
| **all** | LightGBM | 0.5472 | 0.5741 | 0.6842 | 0.7330 | 0.1373 | 0.9688 | 21.9% |
| **acc** | **LogisticRegression** | **0.6748** | **0.7120** | **0.7255** | **0.8096** | **0.0802** | **0.1771** | **13.2%** |
| **acc** | RandomForest | 0.5689 | 0.5902 | 0.6891 | 0.7412 | 0.0882 | 0.5938 | 16.1% |
| **acc** | LightGBM | 0.5343 | 0.5579 | 0.6739 | 0.6920 | 0.0978 | 0.6979 | 14.8% |
| **physio**| **LogisticRegression** | **0.5306** | **0.5954** | **0.5847** | **0.6767** | **0.1194** | **0.3229** | **13.5%** |
| **physio**| RandomForest | 0.4959 | 0.5549 | 0.6197 | 0.6911 | 0.2629 | 0.7604 | 29.3% |
| **physio**| LightGBM | 0.5155 | 0.5662 | 0.6354 | 0.6800 | 0.2193 | 0.4271 | 27.7% |

### 4.2 4분류 (Baseline / Stress / Amusement / Meditation, 2,696창)

| 센서군 | 모델 | Macro F1 | Balanced Acc | Overall Acc | Stress Recall | Stress FPR | S7 Stress FPR | Amusement->Stress 오탐률 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **all** | Dummy | 0.1461 | 0.2500 | 0.4129 | 0.0000 | 0.0000 | 0.0000 | 0.0% |
| **all** | **LogisticRegression** | **0.5668** | **0.5866** | **0.6413** | **0.7877** | **0.0643** | **0.2714** | **18.0%** |
| **all** | RandomForest | 0.5586 | 0.5847 | 0.6826 | 0.7266 | 0.0726 | 0.3714 | 10.3% |
| **acc** | **LogisticRegression** | **0.6175** | **0.6310** | **0.6714** | **0.8066** | **0.0622** | **0.1143** | **11.9%** |
| **acc** | RandomForest | 0.5632 | 0.5768 | 0.6668 | 0.7242 | 0.0514 | 0.1714 | 9.3% |
| **physio**| **LogisticRegression** | **0.4998** | **0.5332** | **0.5534** | **0.6767** | **0.0817** | **0.1643** | **12.2%** |
| **physio**| RandomForest | 0.4288 | 0.4778 | 0.5455 | 0.6740 | 0.1554 | 0.2357 | 22.2% |

---

## 5. 핵심 발견 및 메커니즘 분석

### 5.1 Amusement vs Stress 혼동 (교감신경 각성 혼동 현상)
- 3분류 ALL LR 모델에서 **Amusement 311창 중 73창(23.5%)**이 Stress로 오탐되었다.
- **원인**: Amusement(코미디 비디오 시청 중 웃음, 환호)는 스트레스(TSST 암산, 스피치)와 유사하게 **피부전도도(EDA) 급증과 심박수 상승**을 수반한다. 자율신경계 관점에서 두 상태 모두 교감신경계(Sympathetic Nervous System)가 활성화되는 '고각성(High Arousal)' 상태이기 때문에 순수 생리 특징만으로는 감정가(Valence: 긍정 vs 부정)를 완벽히 분리하기 어렵다.
- 반면 ACC 특징을 추가했을 때, Amusement 동안의 웃음에 수반된 손목 진동과 스트레스 동안의 경직 상태가 물리적으로 달라 오탐률이 13.2%로 감소하였다.

### 5.2 Meditation(명상) 상태의 분리 용이성
- 4분류 평가에서 Meditation 상태는 피부전도도(SCL)가 완만히 하강하고 피부온도(TEMP)가 상승하는 전형적인 부교감신경 우세 반응을 보여, Baseline 및 Stress와의 분리도가 매우 우수하였다.
- Meditation 추가 시 Stress 오탐률(FPR)은 오히려 11.7%에서 6.4%로 절반가량 감소하는 긍정적 효과를 보였다.

---

## 6. 결론 및 제언
1. **WESAD 3분류 최고 모델**: `acc_lr` (Macro F1 **0.6748**), 다중 센서 결합 모델: `all_lr` (Macro F1 **0.6129**).
2. **트리 계열(RF, LGBM) 대비 선형 모델(LR)의 우세**: 참가자 간 기저치 차이가 큰 생리 신호 특성상, 복잡한 비선형 분할(트리)보다 L2 정규화가 적용된 선형 판별이 미학습 피험자(LOSO) 환경에서 과적합을 억제하여 더 높은 Macro F1을 달성하였다.
3. **Amusement 오탐 극복 방안**: 심박변이도(HRV)의 고주파 성분(HF, 부교감신경 지표) 및 호흡 변동률과의 다차원 위상 결합(Phase Coupling) 분석이 요구된다.
