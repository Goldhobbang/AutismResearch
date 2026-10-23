# 막힌 작업과 재개 조건

## B1 — WESAD 원본 데이터 인수 및 학습 / RESOLVED (2026-10-24)

로컬 `data/raw/WESAD`에 공식 15명 참가자(S2–S11, S13–S17) 원본 pickle 파일이 성공적으로 확보되었다.
- 15명 전원 파일 무결성(SHA256 해시, 샘플레이트, 동기화 편차 0.0s) 확인 완료.
- 60초 창·15초 이동 특징 추출(총 5,740 윈도우) 완료.
- Dummy / Logistic Regression / Random Forest 15-fold LOSO 교차검증 및 센서 7종 ablation 완료.
- `results/wesad/` 및 `results/`에 실데이터 메트릭, CSV, 그래프 저장 완료.

## B2 — PhysioNet 구간 라벨 복원 및 운동 혼동 / PARTIALLY RESOLVED (2026-10-24)

- 공식 `subject-info.csv`, `data_constraints.txt`, `Stress_Level_v1/v2.csv` 전수 감사 완료.
- Protocol V1(100% 남성)과 Protocol V2(100% 여성)의 성별-버전 교란을 공식 규명.
- 179개 전체 프로토콜 세그먼트 중 단일 상태 적격 세그먼트 99개를 복원하여 `data/processed/physionet/verified_segments.csv` 구축 완료.
- 30초/45초 무산소 스프린트는 60초 창과의 길이 불일치로 엄격 제외.
- 세션 전체 일괄 라벨링 금지 원칙 수립. 대용량 zip 다운로드는 서버 스로틀링(4KB/s) 기록 및 메타데이터 직접 취득으로 전환.

## B3 — 실행환경과 사용자 장비 차이 / RESOLVED (2026-10-24)

사용자 노트북(Windows 11, ASUS TUF Gaming A14 FA401UV, Ryzen 7 8845HS, RAM 32GB, RTX 4060 Laptop 8GB CUDA 13.3)에서 실제 실행 및 측정 완료.
- Ryzen 7 8845HS CPU: 15-fold LOSO 학습(135 fit) 약 25초 소요.
- S2 원시 신호 리플레이 지연시간: p50 9.48ms, p95 13.15ms 실측 완료.
- NVIDIA GeForce RTX 4060 Laptop GPU: 드라이버(610.62) 및 CUDA 13.3 환경 정상 확인.

## B4 — ASD Onset 실데이터 부재 및 조기경보 / AUDITED & BLOCKED_NOT_EVALUATED (2026-10-24)

- 7대 후보(Kartha 2026, Goodwin 2019, Imbiriba 2023, ASC-emotion, CALMED, PreMAC 2021, REACT 2024) 전수 실사 완료 (`data/registry/asd_candidates.csv`).
- 손목 웨어러블 신호와 실제 행동 Onset을 동시에 갖추고 재사용이 승인된 공공 자료가 현재 0건임.
- ASC-emotion(비디오 3D 안면/자세 랜드마크) 등 비웨어러블 데이터를 손목 조기경보 수치로 위조하지 않음.
- 정식 DUA/IRB 승인 자료 확보 전까지 D3 성능 수치는 `BLOCKED_NOT_EVALUATED` (N/A)로 유지하며, $L=60, H=30$초 기준 엄격한 사건 단위 예측 프로토콜(`reports/ASD_EVENT_FORECASTING.md`)을 수립 완료함.
