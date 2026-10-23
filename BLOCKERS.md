# 막힌 작업과 재개 조건

## B1 — WESAD 원본 데이터 인수 및 학습 / RESOLVED (2026-10-24)

로컬 `data/raw/WESAD`에 공식 15명 참가자(S2–S11, S13–S17) 원본 pickle 파일이 성공적으로 확보되었다.
- 15명 전원 파일 무결성(SHA256 해시, 샘플레이트, 동기화 편차 0.0s) 확인 완료.
- 60초 창·15초 이동 특징 추출(총 5,740 윈도우) 완료.
- Dummy / Logistic Regression / Random Forest 15-fold LOSO 교차검증 및 센서 7종 ablation 완료.
- `results/wesad/` 및 `results/`에 실데이터 메트릭, CSV, 그래프 저장 완료.

## B2 — PhysioNet 구간 라벨 복원 / OPEN

공식 문서·참가자표·제약은 검토했지만 원자료·전체 tags·공식 notebook의 대응을 실행 검증하지 못했다. `segment_manifest.csv`는 header-only다. 세션 전체를 stress/exercise로 라벨링하지 않는다. D2 지도학습은 중지 상태다. 60초 창이 30/45초 sprint보다 길 수 있으므로 anaerobic 유효창 0의 가능성을 숨기지 않는다. 30초 변경은 별도 사전등록 후 수행해야 한다.

## B3 — 실행환경과 사용자 장비 차이 / RESOLVED (2026-10-24)

사용자 노트북(Windows 11, ASUS TUF Gaming A14 FA401UV, Ryzen 7 8845HS, RAM 32GB, RTX 4060 Laptop 8GB CUDA 13.3)에서 실제 실행 및 측정 완료.
- Ryzen 7 8845HS CPU: 15-fold LOSO 학습(135 fit) 약 25초 소요.
- S2 원시 신호 리플레이 지연시간: p50 9.48ms, p95 13.15ms 실측 완료.
- NVIDIA GeForce RTX 4060 Laptop GPU: 드라이버(610.62) 및 CUDA 13.3 환경 정상 확인.

## B4 — ASD onset 자료 / 이번 범위에서 미확보

멜트다운 사건 단위 precision/recall, false alarms/day, 실제 lead time 및 개입 효과는 N/A다. 문헌 수치나 WESAD 성인 실험실 스트레스 점수로 대체할 수 없다.
