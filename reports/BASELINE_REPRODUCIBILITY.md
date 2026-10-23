# WESAD 베이스라인 재현성 및 감사 보고서 (BASELINE_REPRODUCIBILITY.md)

- 작성 시각: 2026-10-24 03:55 (UTC+9)
- 게이트 판정: **G1 PASS (100% 완전 재현)**
- 기준 커밋: `794507721caff9607cff02f58eb8265e671abcc6`
- 실행 환경: Windows 11 / AMD Ryzen 7 8845HS (8C/16T) / 32GB RAM / NVIDIA RTX 4060 Laptop (CUDA 13.3) / Python 3.13.2

---

## 1. WESAD 원본 데이터 무결성 및 적격 창 집계

- **원시 데이터 경로**: `data/raw/WESAD/` (S2–S11, S13–S17, 총 15명).
- **무결성 감사**:
  - 15명 전원 원본 pickle 파일 SHA256 체크섬 확인 완료 (`data/registry/raw_sha256.csv` 참조).
  - 센서 duration spread: 0.0s (EDA 4Hz, BVP 64Hz, ACC 32Hz, Label 700Hz 간 동기화 오차 없음).
- **창 추출 및 라벨링**:
  - 과거 창 $W = 60$초, 이동 간격 $hop = 15$초, 반개구간 $[t-60, t)$ 적용.
  - 총 윈도우 수: 5,740개.
  - 라벨 적격 창 ($y \in \{0, 1\}$ 및 `qc_common_valid = True`): **2,031개**.
    - 스트레스 ($y = 1$, WESAD phase 2): **607개** (29.89%).
    - 비스트레스 ($y = 0$, baseline 1 및 amusement 3): **1,424개** (70.11%).
    - 평가 제외 창 ($y = -1$, 과도기/명상/혼합 라벨): 3,709개.

---

## 2. 참가자 평균(Primary) vs 풀링 OOF(Secondary) 성능 비교

15명 피험자 독립 Leave-One-Subject-Out(LOSO) 15-fold 교차검증 결과:

| 모델 | 센서 조합 | Primary: 참가자 평균 AP | Primary: Stress F1 | Primary: AUROC | Primary: Balanced Acc | Secondary: Pooled AP | Secondary: Pooled F1 | Secondary: Pooled FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **dummy** | all | 0.2987 | 0.0000 | 0.5000 | 0.5000 | 0.2989 | 0.0000 | 0.0000 |
| **lr** | all | 0.9048 | 0.7425 | 0.9519 | 0.8214 | 0.9015 | 0.7407 | 0.1419 |
| **rf** | all | 0.8586 | 0.6142 | 0.9065 | 0.7207 | 0.8586 | 0.6698 | 0.0885 |
| **lr** | eda (기준선) | 0.8045 | 0.6388 | 0.8791 | 0.7421 | 0.8115 | 0.6558 | 0.1770 |
| **lr** | ppg | 0.7333 | 0.6462 | 0.8622 | 0.7555 | 0.7439 | 0.6481 | 0.2317 |
| **lr** | **acc (단독)** | **0.9245** | **0.7723** | **0.9544** | **0.8478** | **0.9221** | **0.7719** | **0.1292** |
| **lr** | eda_ppg | 0.8520 | 0.6142 | 0.9185 | 0.7493 | 0.8576 | 0.6441 | 0.1173 |
| **lr** | eda_acc | 0.9134 | 0.7059 | 0.9479 | 0.8009 | 0.9068 | 0.7303 | 0.1131 |
| **lr** | ppg_acc | 0.9092 | 0.7579 | 0.9537 | 0.8415 | 0.9045 | 0.7548 | 0.1369 |

> **통계적 주의**:
> 1. Primary(참가자 평균)는 각 참가자의 독립적인 AP 및 F1의 평균이며, 양성 비율 차이와 개인차를 반영한 주 지표임.
> 2. Secondary(Pooled OOF)는 폴드별로 서로 다른 최적 임계값이 적용된 이진 결정을 단순히 합산한 참조 지표임.
> 3. RF는 트리 앙상블 특성상 훈련 데이터 피험자별 아티팩트에 과적합하여, 정규화된 LR(C=1 L2) 대비 LOSO 일반화 성능(AP 0.8586 vs 0.9048)이 낮음.

---

## 3. 독립 재현성(Reproduction) 검산 검증

- **비교 대상**: 기존 베이스라인 동결본(`results/baselines/wesad_20261024/`) vs 신규 독립 재현본(`results/phase2/wesad_repro/`).
- **검산 결과**:
  - `max(|baseline.metrics - repro.metrics|) = 0.0` (단 한 건의 부동소수점 오차 없이 100% 비트 단위 일치).
  - 15개 outer fold 분할 구조(`splits.json`) 및 폴드별 검증셋 임계값 탐색 일치.
  - 전처리(SimpleImputer 중앙값 대체, StandardScaler)의 훈련 폴드 국한(train-only) 및 시간 누출 0건 재확인.
