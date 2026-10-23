# 조기 위험 알림 연구 패키지 — 최신 실행·설계 안내

기준: 2026-10-23. 마스터 명세와 이후 사용자 변경 지시를 반영했습니다.

**현재 제품 설계: ESP32-S3 · BLE-only · EDA/PPG/IMU + Mic/Light 맥락 보조 · DRV2605L 진동 · 로컬 특징로그.** EDA만 AFE/ADC를 거칩니다. 상면·하면·측면 배치와 배터리/RF/광학 경로를 함께 검토했습니다. 실제 제작·회로·치수·성능 실측은 아직 없습니다.

**현재 AI 연구: 성인 실험실 스트레스 분류입니다. 아동 멜트다운 조기예측으로 해석하지 않습니다.** 기본은60초 창·15초 hop·15명 LOSO입니다. 초기 kit의120초/5fold 계획은 최신 설정이 아닙니다.

## 먼저 볼 문서

- `reports/FINAL_REPORT_KO.html` 또는 `.md`: 통합 보고서11항목, 실제 완료/보류 상태.
- `design/PRODUCT_DESIGN.html`, `design/PCB_LAYOUT.html`: 최신 제품·상하면·측면 설계.
- `design/SYSTEM_BLOCK.png`, `design/PCB_LAYOUT.png`: 발표용 개념도, SVG 원본 포함.
- `design/BOM.csv`, `ASSUMPTIONS.md`, `energy_scenarios.csv`: 가격 출처·제외항목·이론 계산.
- `design/APP_MOCKUP.html`: 연구모드/측정불가/연결끊김 화면안.
- `reports/PRESENTATION_11_KO.pptx`, `.pdf`:11장 발표 초안. 실제 성능은 N/A로 표시.
- `EXPERIMENT_REGISTRY.md`, `EVIDENCE_LEDGER.md`, `DATASET_AUDIT.md`, `BLOCKERS.md`, `reports/QUALITY_REPORT.md`.

## 실제 실행 상태

- 새 소스의 단위 검사 19개와 합성 6명×9설정 파이프라인 검사를 통과했습니다.
- **WESAD 15명 실데이터 학습 완료**: 공식 15명(S2–S11, S13–S17) 원본 무결성 검증 후 60초 창·15초 이동 특징(5,740 윈도우) 추출 및 15-fold LOSO 교차검증을 완료했습니다.
  - Logistic Regression (전체 센서): 참가자 평균 AUPRC 0.9048, Stress F1 0.7425, Balanced Acc 0.8214.
  - Random Forest (전체 센서): 참가자 평균 AUPRC 0.8586, Stress F1 0.6142, Balanced Acc 0.7207.
  - 가속도계(ACC 단독): AUPRC 0.9245 (EDA 단독 대비 +0.1200 향상, 10명 개선).
- **리플레이 실측**: S2 원시 스트림 리플레이 OOF 점수 일치율 100%, 지연시간 p50 9.48ms 실측 (Ryzen 7 8845HS).
- **주의**: 성인 실험실 스트레스 분류 결과이며 아동 멜트다운 사건 조기경보 지표(N/A)로 확대 해석하지 않습니다.

## Windows PowerShell 빠른 시작

ZIP을 푼 `AutismResearch` 폴더에서 실행합니다. Python3.11 64bit가 필요합니다. 기본 모델은 CPU에서 실행되며 PyTorch는 필요하지 않습니다.

```powershell
py -3.11 -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install -r requirements.txt
& $py -m pip check
& $py scripts/capture_environment.py
& $py -m unittest discover -s tests -p "test_*.py" -v
# 선택: 소프트웨어 전체 동작 검사, 연구 성능 아님
& $py scripts/synthetic_test_pipeline.py
```

공식 [WESAD](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html) 이용 조건을 확인한 후:

```powershell
& $py scripts/download_datasets.py --dataset wesad --extract
Get-ChildItem data/raw -Recurse -Filter S2.pkl
& $py -m src.features.extract --root data/raw/WESAD --out data/processed/wesad_features.csv --window 60 --step 15
& $py -m src.train --features data/processed/wesad_features.csv --out results/wesad --plan core --expected-subjects 15 --dataset-version WESAD_ICMI2018 --jobs 4
& $py -m src.replay --out results/wesad --subject S2 --model lr --sensors all --raw-root data/raw/WESAD
& $py scripts/publish_results.py --source results/wesad
```

압축해제 폴더가 중첩되면 `--root`를 실제 `S*/S*.pkl` 부모로 바꿉니다. 공식 원본 외 pickle은 열지 않습니다. 다운로드 스크립트는 원본을 덮어쓰지 않으며 이미 ZIP이 있으면 보존·감사하도록 중단합니다.

통합 스크립트는 `run_wesad.ps1`입니다. 이미 원본이 있으면 `-Download`를 빼고 `-WesadRoot`를 지정할 수 있습니다. PowerShell 실행정책을 바꾸고 싶지 않으면 위 개별 Python 명령을 사용하세요. 스크립트의 Windows 실행은 아직 미검증입니다.

실제 성공 후 `results/wesad/`에 참가자별 지표·7센서 비교·분할·음성조건 오류·품질/라벨별 제외율·그림·모델이 생깁니다. `publish_results.py`는 완성된 실제 결과를 계약 경로로 복사하고 `reports/RESULTS_CURRENT.md`를 만듭니다. 날짜가 고정된 기존 보고서의 N/A 문구는 실제 결과 검토 후 갱신해야 합니다. 합성 결과의 게시를 거부합니다.

## PhysioNet 게이트

`py scripts/download_datasets.py --dataset physionet --extract`로 공식 자료를 받습니다. 전체 tags·버전별 프로토콜·파일·동일 참가자 그룹을 검증하기 전 지도학습을 실행하지 않습니다. 현재 `segment_manifest.csv`는 검증된 행이 없습니다. 제공한 loader가 자동으로 태스크 라벨을 복원하는 것은 아닙니다. 세션 전체를 하나의 상태로 라벨링하지 마세요.

## CUDA와 선택 도구

`nvidia-smi`와 `check_cuda.py`는 전체 보고서6절의 PyTorch 설치 후 실행합니다. NVIDIA RTX4060을 사용하며 AMD NPU/Radeon은 CUDA 장치가 아닙니다. CUDA 문제가 생겨도 CPU 기준 실험은 계속할 수 있습니다.

발표 재생성에는 선택 의존성 `python-pptx`가 필요하며 `py scripts/build_presentation.py`로 실행합니다. 핵심 학습 의존성과 분리했습니다. 에너지 계산은 `py design/energy_budget.py`입니다.

원자료·합성 raw/모델/점수그림은 새 배포 ZIP에 포함하지 않습니다. 원본 마스터는 `RESEARCH_MASTER_SPEC.md`, 변경 결정은 `DECISIONS.md`, 실행 기록은 `EXECUTION_LOG.md`, 소스/산출물 해시는 `DELIVERY_SHA256.json`을 확인하세요.
