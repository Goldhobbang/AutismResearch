# 실행 로그

## 2026-10-23 / 마스터 명세 인수

- 붙여넣은 텍스트를 사용자의 실행 요청으로 적용했다. 이전 첨부 명세는 보관하고 최신 붙여넣은 요청을 `RESEARCH_MASTER_SPEC.md`로 저장했다.
- 기존 `/workspace/AutismResearch` 및 초기 ZIP을 확인했다. ZIP CRC 검사를 통과했다. 초기 문서 사본은 `archive/initial_kit/`에 보존했다. 기존 코드·결과는 삭제하지 않았다.
- 사용자 요청에 명시된 A–E 역할을 병렬 배정했다. 데이터→특징→평가 의존성은 유지한다.
- 사전 `EXPERIMENT_REGISTRY.md` 작성: 60초/15초, LOSO, 9개 핵심 설정, 공통 QC 창, 7종 LR 센서 비교.
- 현재 환경은 Linux 클라우드다. 사용자 Windows 노트북을 원격으로 점검하거나 CUDA를 확인했다고 주장하지 않는다. 환경 기록: `results/logs/environment.json`.
- 공식 WESAD/PhysioNet 다운로드에서 프록시 HTTP403 발생. 환경 도구에서 HTTP restricted/enforced 정책 확인. 사용자에게 환경 설정 변경을 요청했고 사용자는 인터넷 접근을 허용하겠다고 답했다. 설정 반영과 다운로드 성공은 별도로 확인한다.

후속 실행/검사 결과와 실제 데이터 유무는 이 파일에 계속 기록한다.

- 문헌 재감사: REACT Assessment2는 기록 스트림 모사이며 실제 보호자 경고 현장시험으로 읽을 수 없음을 확인했다. 라벨 band 내 refinement도 표준 window F1과 구분한다. 현행 문헌보고서를 정정했고 원본은 archive에 보존했다.

## 통합 검사와 산출물

- 새 src/ unit tests 19개 통과. 합성6명의9설정54 fit,7종LR센서비교,모든outerfold,결과CSV/JSON/그림,원기록재생 통과. 로그는 results/logs/에 보존.
- precision 분모0을 NaN+이유로 수정하고 지표 의미 검사를 추가했다.
- 공식 다운로드 스크립트를 실제 실행: WESAD와PhysioNet 모두proxy403. 사용자에게 공식 설정 경로를 안내했다. 실제원본·성능값은 여전히0/N/A.
- BOM·시스템블록도·앱시안·에너지시나리오·11장발표문구·통합보고서를 작성했다. MCU와배터리는실측하지 않았다.
- 병렬 역할의 일부 작업이 중단되어 통합책임자가 누락 파일·계약·테스트·문서를 이어서 완성했다.

## 사용자 설계 변경 반영

- LTE를 현행 설계에서 제외하고 ESP32-S3 BLE-only로 개정했다. EDA만 AFE/ADC, PPG·IMU·Mic·Light의 각 인터페이스, DRV2605L와 local feature logging을 명시했다.
- 상면/하면/측면 개념도를 추가했다. 배터리의 RF keep-out·PPG 경로 침범, 모듈형 Flash/PSRAM 중복, EDA AFE 위치와 Mic/Light 포트를 검토했다.
- BOM·전류 가정·에너지 스크립트·보고서·11장 발표를 함께 갱신했다. 과거 설계는 archive/design_before_esp32_revision에 보존했다.

## 2026-10-24 / 로컬 WESAD 15명 실데이터 학습 및 LOSO 평가 완료

- **실행 환경 확인**: Windows 11, ASUS TUF Gaming A14 FA401UV, AMD Ryzen 7 8845HS (8C/16T, 32GB RAM), NVIDIA GeForce RTX 4060 Laptop (8GB VRAM, 드라이버 610.62, CUDA 13.3). Python 3.13.2 가상환경(`.venv`) 구축 및 `numpy==2.5.4`, `pandas==2.3.3`, `scipy==1.18.1`, `scikit-learn==1.9.1`, `matplotlib==3.11.2`, `joblib==1.6.0`, `python-pptx==1.0.2` 설치 완료. `pip check` 정상 통과.
- **단위 테스트**: `python -m unittest discover -s tests -p "test_*.py" -v` 실행 (19개 테스트 전부 PASS).
- **WESAD 원본 무결성 검증**: `data/raw/WESAD` 내 공식 15명(S2–S11, S13–S17) 원본 pickle 파일 인수. 15명 전원 SHA256 체크섬 계산 및 샘플레이트(EDA 4Hz, BVP 64Hz, ACC 32Hz, Label 700Hz), 센서 duration spread 0.0s 무결성 검증 완료.
- **특징 추출**: `python -m src.features.extract --root data/raw/WESAD --out data/processed/wesad_features.csv --window 60 --step 15`. 총 5,740 윈도우(86개 특징, y=1: 607, y=0: 1,424, y=-1: 3,709, qc_common_valid: 5,740) 추출 완료 (Ryzen 7 8845HS 기준 15명 총 추출 시간 약 118초, 윈도우당 연산 약 9ms).
- **LOSO 15-fold 교차검증**: `python -m src.train --features data/processed/wesad_features.csv --out results/wesad --plan core --expected-subjects 15 --dataset-version WESAD_ICMI2018 --jobs 8`.
  - 모델: Dummy prior, Logistic Regression(LR), Random Forest(RF, 200 trees).
  - 센서 조합: 7종(EDA, PPG, ACC, EDA+PPG, EDA+ACC, PPG+ACC, ALL).
  - 총 135개 모델 fit(15 folds × 9 설정) 약 25초에 정상 완료.
  - 데이터 누출 방지: 외부 15-fold 참가자 완전 분리(LeaveOneGroupOut), 내부 GroupShuffleSplit 검증셋으로만 F1 임계값 탐색, StandardScaler/SimpleImputer는 train fold에서만 fit.
- **실제 성능 메트릭 (15명 참가자 평균)**:
  - `dummy / all`: AUPRC 0.2987, AUROC 0.5000, F1 0.0000, Balanced Acc 0.5000.
  - `lr / all`: AUPRC 0.9048, AUROC 0.9519, F1 0.7425, Precision 0.7699, Recall 0.8047, Balanced Acc 0.8214.
  - `rf / all`: AUPRC 0.8586, AUROC 0.9065, F1 0.6142, Precision 0.6829, Recall 0.7537, Balanced Acc 0.7207.
  - `lr / eda` (기준선): AUPRC 0.8045, AUROC 0.8791, F1 0.6388, Precision 0.6985, Recall 0.7507, Balanced Acc 0.7421.
  - `lr / ppg`: AUPRC 0.7333, AUROC 0.8622, F1 0.6462, Precision 0.6130, Recall 0.7632, Balanced Acc 0.7555 (EDA 대비 -0.0711).
  - `lr / acc`: AUPRC 0.9245, AUROC 0.9544, F1 0.7723, Precision 0.8267, Recall 0.7915, Balanced Acc 0.8478 (EDA 대비 +0.1200, 10명 개선).
  - `lr / eda_ppg`: AUPRC 0.8520, AUROC 0.9185, F1 0.6142, Precision 0.7501, Recall 0.6909, Balanced Acc 0.7493 (EDA 대비 +0.0475).
  - `lr / eda_acc`: AUPRC 0.9134, AUROC 0.9479, F1 0.7059, Precision 0.8305, Recall 0.7267, Balanced Acc 0.8009 (EDA 대비 +0.1089).
  - `lr / ppg_acc`: AUPRC 0.9092, AUROC 0.9537, F1 0.7579, Precision 0.8016, Recall 0.7965, Balanced Acc 0.8415 (EDA 대비 +0.1047).
- **원시 신호 리플레이 검증**: `python -m src.replay --out results/wesad --subject S2 --model lr --sensors all --raw-root data/raw/WESAD`. S2 윈도우별 실시간 특징 재계산 및 추론 확률과 OOF 예측치 일치율 100%(atol=1e-7 통과). 추론+특징 지연시간 실측: p50 9.48ms, p95 13.15ms.
- **결과 산출 및 문서 갱신**: `scripts/publish_results.py` 실행 완료. `reports/RESULTS_CURRENT.md`, `reports/FINAL_REPORT_KO.md`, `QUALITY_REPORT.md`, `BLOCKERS.md`, 11장 발표자료(`.pptx`) 갱신.
- **주의 및 경계**: 성인 실험실 스트레스 분류 결과이며 아동 멜트다운 조기경보 및 ASD 사건 단위 성능(Precision, Recall, False Alarms/day, Lead Time)은 N/A로 엄격히 유지.

## 2026-10-24 / 단일 에이전트 통합 실행 명세 v1.0 이행

- **G0 인수 및 베이스라인 동결**:
  - `results/wesad/` 원형을 `results/baselines/wesad_20261024/`로 100% 동결 복제하여 훼손 방지 조치.
  - 디렉터리 체계 구축: `data/registry/`, `data/processed/{wesad,physionet,asd}/`, `experiments/phase2/`, `results/phase2/{acc,threshold,physionet,wesad_repro}/`.
- **G1 WESAD 재현성 감사 (독립 재학습)**:
  - `results/phase2/wesad_repro/` 경로로 15-fold LOSO 교차검증 독립 재실행.
  - 베이스라인 동결본과 신규 재현본 간 수치 오차 검산: `max(|baseline - repro|) = 0.0` (100% 비트 단위 일치 확인).
  - `reports/BASELINE_REPRODUCIBILITY.md` 작성.
- **G2 ACC 교란 요인 심층 분해**:
  - `experiments/phase2/acc_confounding_analysis.py` 개발 및 실행.
  - 41개 ACC 특성을 7개 하위군으로 15-fold LOSO 분해:
    - 정적 자세(평균/중앙값 중력성분) 단독: AP 0.5701 (무작위 추측 수준).
    - 정적 자세 제거 모델(`acc_no_posture`): AP 0.9298 (최고치).
    - 동적 에너지/저크(`acc_dynamic_only`): AP 0.8620.
  - 결론: 가속도계 성능은 정적 자세가 아닌 TSST 과제 발표 중 손목 제스처 및 긴장성 동적 움직임(Task Confounding)에 기인함.
  - 특이 피험자 S7 분석: AUROC/AUPRC 1.0000임에도 FPR 0.8542 발생 원인은 개인 베이스라인 확률의 상향 편향(중앙값 0.9128)과 집단 고정 임계값(0.35) 불일치 때문임을 규명.
  - `reports/ACC_CONFOUNDING_ANALYSIS.md` 작성.
- **G3 임계값 및 알림 정책 최적화**:
  - `experiments/phase2/threshold_alert_analysis.py` 개발 및 실행.
  - 6개 정책 비교: Fixed 0.5 (F1 0.7670, FPR 0.1415), Conservative 0.70 (F1 0.7354, FPR 0.1148), Cooldown 60s (실질 비스트레스 오탐율 0.0433 최저).
  - 0.05~0.95 전체 임계값 격자 탐색으로 Recall vs FPR Pareto Frontier 도출.
  - `reports/THRESHOLD_AND_ALERT_ANALYSIS.md` 작성.
- **G4/G5 PhysioNet v1.0.1 메타데이터 전수 감사 및 세그먼트 복원**:
  - `Data_Dictionary.csv`, `physionet_data_constraints.txt`, `physionet_subject_info.csv`, `Stress_Level_v1/v2.csv`, `SHA256SUMS.txt` 공식 원본 획득 및 무결성 확인.
  - 성별-버전 교란 규명: Protocol V1은 100% 남성(18명), Protocol V2는 100% 여성(33명). 혼합 학습 시 성별 편향 학습 위험 경고.
  - 30초/45초 무산소 스프린트는 60초 창 규약과 길이 불일치로 엄격 배제.
  - 단일 상태 적격 세그먼트 99개 복원 완료 (`data/processed/physionet/verified_segments.csv`).
  - `reports/EXTERNAL_VALIDATION.md` 작성.
- **G6/G7 자폐(ASD) 7대 후보 데이터셋 실사 및 사전 프로토콜 수립**:
  - `scripts/build_asd_audit.py` 작성 및 `data/registry/asd_candidates.csv` 구축.
  - 7개 연구(Kartha 2026, Goodwin 2019, Imbiriba 2023, ASC-emotion, CALMED, PreMAC 2021, REACT 2024) 전수 실사.
  - 판정: 웨어러블+사건 Onset을 동시 만족하는 공공 자료 0건 (D3-WEARABLE-EVENT READY 없음). Kartha/Goodwin/Imbiriba는 D3-ACCESS PENDING (DUA/IRB 필요).
  - ASC-emotion(비디오 3D 안면/자세 랜드마크) 등 비웨어러블 데이터를 손목 조기경보 수치로 위조하지 않음.
  - $L=60, H=30$초 지연 분리, 사건 단위 평가 지표(Event Recall, Alert Precision, False Alarms/day) 규약 수립.
  - `reports/ASD_DATASET_ACCESS_AUDIT.md`, `reports/ASD_EVENT_FORECASTING.md` 작성.
- **G8 MCU 배포 적합성 분석**:
  - LR 모델 순수 C 가중치 배열 크기(760B) vs RF 트리 크기(882KB) 실측.
  - 60초 float32 링버퍼(38.4KB, ESP32-S3 SRAM의 10.1% 점유) 분석.
  - Windows CPU 지연(9.48ms)의 MCU 속도 대치 금지 고지.
  - `reports/MCU_FEASIBILITY.md` 작성.
- **G9 최종 통합 보고서 작성**:
  - 8대 영역(A~H) 종합 검증 보고서 `reports/FINAL_MODEL_RESEARCH_REPORT_KO.md` 작성 완료.
  - `reports/NEXT_ACTIONS.md` 최신 동기화 완료.
