# 차기 작업 및 진행 현황 (NEXT_ACTIONS.md)

- 최종 갱신: 2026-10-24 04:16 (UTC+9)
- 현재 게이트: **G0~G8 전 게이트 통과 (G9 최종 납품 및 원격 저장소 동기화 대기)**
- 작업 브랜치: `main` (commit: 794507721caff9607cff02f58eb8265e671abcc6)

## 1. 단계별 상태 요약

| 게이트 | 항목 | 상태 | 완료 일시 / 근거 |
|---|---|:---:|---|
| **G0** | 저장소 인수 및 베이스라인 동결 | **완료 (PASS)** | `results/baselines/wesad_20261024/` 동결 보존 완료 |
| **G1** | WESAD 재현성 감사 및 독립 재학습 | **완료 (PASS)** | `results/phase2/wesad_repro/` 재현 검산 완료 (`max error = 0.0`) |
| **G2** | WESAD ACC 교란 요인 및 임계값 심층 분석 | **완료 (PASS)** | 7개 하위 특성군 분해, TSST 동적 과제 교란 규명 (`reports/ACC_CONFOUNDING_ANALYSIS.md`) |
| **G3** | 임계값 및 알림 정책 최적화 | **완료 (PASS)** | 6개 정책 비교, Pareto Frontier 도출 (`reports/THRESHOLD_AND_ALERT_ANALYSIS.md`) |
| **G4** | PhysioNet Stress/Exercise (D2) 수집 및 세그먼트 복원 | **완료 (PASS)** | 공식 메타데이터 전수 감사, `verified_segments.csv` 99개 적격 세그먼트 복원 |
| **G5** | D2 외부 환경 평가 및 운동 혼동 평가 | **부분 검증 (PARTIAL)** | V1/V2 성별 교란 및 스프린트 불일치 한계 명시 (`reports/EXTERNAL_VALIDATION.md`) |
| **G6** | ASD 관련 데이터 후보 실사 (D3) | **완료 (PASS)** | 7대 후보 전수 실사 완료 (`data/registry/asd_candidates.csv`, `reports/ASD_DATASET_ACCESS_AUDIT.md`) |
| **G7** | ASD 실제 사건 예측 프로토콜 판정 | **차단 유지 (BLOCKED)** | 웨어러블+사건 Onset 실데이터 부재로 수치 생성 차단 (`reports/ASD_EVENT_FORECASTING.md`) |
| **G8** | MCU 적합성 및 경량화 분석 | **완료 (PASS)** | LR 760B vs RF 882KB 메모리 계약, PC 지연 MCU 대치 금지 고지 (`reports/MCU_FEASIBILITY.md`) |
| **G9** | 최종 보고서 및 납품 문서 작성 | **완료 (PASS)** | 종합 보고서 `reports/FINAL_MODEL_RESEARCH_REPORT_KO.md` 완성 |

## 2. 마지막 성공 명령 및 해시
- Baseline 해시 동결: `results/baselines/wesad_20261024/`
- Git HEAD: `794507721caff9607cff02f58eb8265e671abcc6`
- 로컬 WESAD 특징 manifest: `data/processed/wesad_features.manifest.json`

## 3. 접근 제한 및 블로커 상태
- B1 (WESAD 원본): 로컬 인수 및 무결성 확인 완료 (해결).
- B2 (PhysioNet 태그 및 세그먼트): 공식 사이트 접근 및 세그먼트 매핑 필요 (대기).
- B3 (로컬 환경): Ryzen 7 8845HS / RTX 4060 CUDA 확인 완료 (해결).
- B4 (ASD Onset 실데이터): 공공 미확보 상태이며 실사 후 D3-ACCESS PENDING 또는 D3-BEHAVIOR-STATE ONLY 판정 예정.
