자폐 아동 생체신호 기반 조기 위험 알림 연구 — 마스터 실행 명세서
버전: 1.0 / 기준일: 2026-10-23 / 언어: 한국어 / 목표: 48시간 내 재현 가능한 공개 데이터 실험 + 실제 제작 없는 손목형 웨어러블 설계안 + 과장 없는 대회 발표 결과
연구 판단 완료 범위: 이 문서는 연구 목표의 분리, 검증 가능한 가설, 공개 데이터의 선택과 접근 조건, 성능 평가 원칙, 리스크·우선순위·중단 조건, 작업 분배, 발표 표현까지 상위 결정을 확정한다. 실제 WESAD 실험 수치, 코드 감사, 하드웨어 성능 실측, 아동 대상 행동 예측 성능은 완료됐다고 주장하지 않는다. 이전 작업 환경의 /workspace/AutismResearch/... 파일 내용은 본 작업 환경에서 직접 검증하지 못했으므로, 하위 에이전트가 원본을 확인해야 한다.

0. 프로젝트 결정문 (변경하려면 근거와 승인 기록 필요)
- 제품의 장기 비전: EDA·PPG(BVP)·IMU를 수집하는 손목형 기기에서 개인별 생리·움직임 패턴을 분석하고, 향후 위험 행동이나 감각 과부하 관련 상황을 보호자가 조기에 인지하도록 지원. 약 60초 사전경보는 미입증 연구 목표, 검증된 기능이 아님.
- 이번 프로젝트의 실제 산출 범위: ① 동료심사 근거 및 데이터 특성 정리, ② 성인 공개 데이터로 스트레스 분류와 운동 오탐을 탐색하는 AI 모델 실험, ③ 하드웨어 제작 없는 제품 설계도·BOM·BLE/LTE 아키텍처·앱 화면안, ④ 근거 수준을 구분한 발표 자료용 표·그래프·한계.
- 직접 검증 대상 아님: 자폐 아동의 멜트다운 onset 예측, 60초 실제 lead time, 실제 보호자 개입 효과, 장시간 착용성, 제품 정확도, 배터리 실측, 임상적 위험 감소.
- 운영 원칙: 과학적 정직성 > 피험자 독립 평가 > 데이터 무결성 > 재현성 > 결과 수치 > 모델 복잡도. 80% 점수 보장·성과 꾸미기 금지.
- 노트북: Windows 11, ASUS TUF Gaming A14 FA401UV, Ryzen 7 8845HS, RAM 32GB, RTX 4060 Laptop 8GB. Python 호출 기본은 py. 일차 모델(Dummy/LR/RF)은 CPU 우선, GPU·LSTM은 P2 옵션.
- 시간 제한: 48시간 동안 연구·실험·보고서 정리. 수면 6–8시간을 포함할 것. 데이터 받기 전 모델 성능을 예측해서 쓰지 말 것.
1. 연구 현황 및 문헌 근거: 어떤 사실이 확인됐나
연구	모집단·주요 행동	방법·성능	가능한 주장	금지할 비약
Goodwin 등 (2019)	ASD 청소년 20명, 타인 공격(ATO)	이전 180초 신호로 향후 60초 발생 예측. 집단 AUROC 0.71, 개인화 평균 0.84	센서 신호에 일정 예측정보, 개인화 가능성	84% 정확도/정밀도, 모든 멜트다운에 대한 60초 lead time 보장
Imbiriba 등 (2023, JAMA Netw Open)	ASD 정신과 입원 청소년 70명, 자해(SIB)·정서조절곤란(ED)·타인 공격(ATO), 497시간·6,665 행동	선택된 3분 예측 조건에서 LR AUROC 0.80. 최근 행동 이력을 넣는 설정과 센서 단독 설정을 구분해야 함	보다 큰 표본에서 짧은 시간 위험행동 forecasting 연구 존재	가정/학교로 성능 직접 외삽, AUROC=실제 경보 성공률
Kim 등 (2026, Scientific Reports)	ASD 아동·청소년 9명, SIB 관련 행동 정의에 공격·반복행동도 일부 포함	예측 구간 3–120초, 행동 발생 중인 입력 제외, LOSO. 60초 이상 일부 AUPRC가 무작위보다 높았으나 개인차·오경보 큼	미래 사건 예측 과제는 탐색 가치가 있지만 어려움	무전조 사건 비율을 계산, 손목 단독 범용 상용 모델 검증


절대 구분: 감각 과부하(개인 경험/상태) / 생리적 각성 / 스트레스 / meltdowns / agitation / aggression / self-injury / repetitive behaviors는 서로 동일한 레이블이 아니다. 문헌별 행동 정의, 예측 horizon, 입력 기간, 평가 분할, 모델 입력(행동 이력 포함 여부), 결과 지표를 반드시 표로 분리한다. 감각 과부하가 항상 멜트다운이나 위험 행동을 유발한다고 설명하지 않는다.

문헌상 미해결 문제: ① 손목 센서 기반 조기경보의 실제 사건단위 정밀도·재현율·일일 오경보·첫 알림 선행시간, ② 집단 모델의 일반화, ③ 현장에서의 놀이·운동 혼동, ④ 전조 신호가 약한 사건의 인구집단 비율, ⑤ 알림 개입이 실제 위험 행동을 줄이는 인과적 효과, ⑥ 보호자·아동의 기기 수용성. 모델 미탐지율 ≠ 생물학적 전조 부재율.
2. 공식 데이터셋 — 확정된 사용 계획
D1. WESAD (P0)
- 1차 출처: https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html ; UCI: https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection
- 분포: 성인 15명, 실험실 baseline(레이블 1)·stress(2)·amusement(3); 다른 레이블/전환·명상 구간은 원본 문서 확인 후 별도 처리. 손목 BVP 64Hz·EDA 4Hz·ACC 32Hz·TEMP 4Hz. 가슴 센서 신호를 손목형 제품 실험에 섞지 말 것.
- 다운로드: 원본 아카이브 약 2.5GB 압축(연구자 제공 사이트 기준). 데이터 가용량·압축 해제 용량을 확인하고 raw 압축·불필요한 복제 최소화. 위 사이트의 원본 이용 조건과 포함된 readme를 확인한다. UCI 표기 라이선스와 원 소스의 비상업 조건이 다르면 더 엄격한 원 출처 조건에 따르고 상업적 재배포 하지 않는다.
- 최우선 과제: 손목 기반 실험실 스트레스 상태 분류. Primary: stress vs baseline+amusement(클래스 2 대 1/3), 단 class 3을 '놀이/운동'이라고 부르지 않음. Secondary: 3-class baseline/stress/amusement.
- 원본 피험자 ID: S2~S11, S13~S17 (총 15). 전체 15명 이외 피험자 임의 생성 금지.
- 파일 형식 주의: 공식 제공 .pkl은 원본 구조 확인 후 로딩하며, pickle은 임의 코드 실행 위험이 있으므로 공식 출처와 무결성이 확인된 파일만 로드. 라벨의 시간해상도가 손목 신호와 다를 수 있어 매핑 전 원본 구조·길이를 반드시 검사.
D2. PhysioNet Wearable Device Dataset from Induced Stress and Structured Exercise v1.0.1 (P1, 강력 권장)
- 공식: https://physionet.org/content/wearable-device-dataset/1.0.1/ ; DOI 10.13026/he0v-tf17.
- 연구 조건: 스트레스 세션 36명, 유산소 운동 30명, 무산소 운동 31명. 서로 완전히 독립적인 97명이라고 가정 금지. 실험 대상은 18–30세 성인. 손목 Empatica E4 BVP 64Hz, ACC 32Hz, EDA 4Hz, 피부온도 4Hz; HR·IBI 파일도 있음.
- 접근: 공개 이용 조건을 준수하면 다운로드 가능; ZIP 69.7MB, 비압축 247.4MB(2025-06-24 버전 페이지), ODC-By-1.0 라이선스.
- 용도: stress vs structured exercise 구별, 활동량 ACC의 추가 효과, 운동 중 오경보 분석(이 연구 데이터에서만).
- 특별 주의: 전체 세션을 하나의 상태로 라벨링 금지. 스트레스 실험 중 휴식, 운동 전후 휴식·워밍업·쿨다운 존재. tags.csv, 공식 노트북·프로토콜 문서로 타임라인과 유효한 태스크 구간을 복원할 수 있을 때에만 supervised labels 부여. 불확실한 구간은 ambiguous로 제외. 서로 다른 프로토콜 버전 v1/v2의 단계 길이가 다름.
- 데이터 품질 관련 공식 제한: stress S02 중복, stress f07 센서 보호필름 미제거, stress f14 두 파일로 분할; aerobic S03·S07 미완주, S11 분할, S12 미수행; anaerobic S06·S16 미완주. data_constraints.txt를 직접 읽어 적용하고 제외·포함 로그 보존.
- 분할 단위: 동일 참가자의 STRESS/AEROBIC/ANAEROBIC 세션 및 분할 파일(f14_a/b)을 같은 group으로 취급. 익명 ID가 다른 버전 간 동일인인지 공문서로 확인 불가능하면 동일인 보장은 미확인 위험으로 명시.
- 교란: 스트레스는 정신과제·발성, 운동은 고정식 자전거. 데이터의 세션 종류/장소/프로토콜 차이로 분류기가 상태가 아니라 세션 차이를 학습할 수 있음. 생리적 감각 과부하·놀이·멜트다운과 동치화 금지.
D3. PPG-DaLiA (P2 선택)
- 목적: 운동 artifact가 PPG·심박 품질에 미치는 영향 정량화. 일정 미달이면 생략. 원 출처·라이선스·라벨 확인 후 이용.
D4. ASD event-onset 공개 데이터 (현시점 미확보)
- 사용자가 제공한 이전 조사에서 즉시 이용 가능한 손목 신호+행동 시작 시각+충분한 사건 전후 기록을 가진 공개 원자료를 확인하지 못했음. 저자 요청·기관 심의 필요 후보는 별도 조사 대상으로 남긴다.
- 공개 WESAD/PhysioNet으로 실제 ASD 멜트다운 forecast/event recall/lead time/day 산출 금지.
3. 확정된 연구 가설 및 반증 조건
ID	가설	실험	결과 판정	불가능/제약
H1 [P0]	손목 신호로 WESAD 실험실 스트레스 상태를 chance보다 잘 분류	D1, Dummy vs LR/RF, LOSO	피험자별 F1·Balanced Acc·AUPRC/AUROC 및 CI/범위	ASD 행동 예측으로 해석 불가
H2 [P0]	EDA·BVP·ACC 결합이 단독 센서보다 도움이 되는가	D1 센서 ablation	피험자별 paired metric 차이와 개선/악화 참가자 수	개선을 사전에 전제하지 말 것
H3 [P1]	ACC를 결합하면 운동을 스트레스로 오분류하는 비율을 줄일 수 있는가	D2 동일한 label/eval 분할에서 모델 A vs B	exercise false-stress rate, macro F1, confusion matrix	유효한 시간 라벨 복원 실패 시 결과 미제시
H4 [P1]	새 참가자로 일반화되는가	D1/D2 완전 분리된 subject groups	LOSO/group holdout 성능과 분포	random window split 금지
H5 [P2]	비지도 개인 기준선으로 참가자 간 차이를 일부 보정할 수 있는가	사용 가능한 선행 baseline만 이용해 보정 vs 미보정	동등한 LOSO 기준 비교	테스트 피험자의 평가 정답 label 활용 금지
H6 [설계]	제안한 AI가 MCU에서 실시간 실행 가능할 수 있는가	후보 MCU RAM/flash/연산량·모델 크기 종이 설계	요구량/가용량·미확인 사항 표	실제 기기에서 속도·소모전력 실측 주장 금지
H7 [미검증]	ASD 멜트다운 60초 전 실제 조기경보 가능	이번 48h 데이터로는 실험 불가	현장 연구용 프로토콜만 작성	기존 WESAD 결과를 대용 금지


성공의 정의: H1/H2/H3에서 부정적 결과가 나와도 데이터 무결성·누출 방지·정확한 해석을 충족하면 연구 성공. 명시된 사전 판정 기준을 결과에 맞춰 바꿔선 안 됨.
4. 데이터 처리 프로토콜 — 필수 세부 사항
1. 인수 검사: 라이선스·버전·원본 경로·파일 목록·크기·해시 가능 여부·데이터 열/행·환산 단위·유효 피험자 수 확인, DATASET_AUDIT.md 작성. 원데이터는 data/raw/ 읽기 전용, Git 및 ZIP 산출물에서 제외.
2. 시간 정렬: 각 센서 원본 레이트 유지하고 t_start, t_end 정의; 손목 WESAD는 원본의 동기화 메타데이터를 근거로 라벨 매핑. PhysioNet은 시작 타임스탬프, 2행째 Hz 및 tags.csv의 UTC 기준 확인. 원시 시계열 전체를 임의로 초고속 보간하거나 중복 업샘플해 허위 정보를 만들지 않는다.
3. 윈도 정의: 기본 60초(선행 자료만), hop 15초. 민감도 실험 30초·120초 중 하나(자원이 허용할 때). t시각 예측은 (t−60초, t] 자료만 사용. 끝점·중첩·경계 정렬을 테스트 코드로 입증. 180초 긴 윈도는 1개 이상 구간이 충분히 길고 조건 경계를 넘지 않을 때만 시도.
4. 레이블 규칙: WESAD 1/2/3만 포함; 한 창에서 동일 상태 비율 100%를 primary로 채택(엄격한 단일 상태), 혼합·전환창 제외. 레이블 stress=1, baseline/amusement=0; negative class 구성별 성능 별도 보고. PhysioNet은 공식 event marks를 재구성하지 못하는 구간은 unknown 처리.
5. EDA 특징: mean/median/std, slope, range, 차분 양의 합, SCR 후보 수(적절한 신호품질·peak 방법 명시); 작은 샘플·측정 artifact 시 결측 처리. tonic/phasic 분해는 P1 옵션, 외부 라이브러리 환경 명시.
6. BVP 특징: 평균/표준편차/신호 에너지 및 스펙트럼 보조 특징, 유효 맥박 추정이 가능한 창에만 HR/HRV 계산. 60초 짧은 창의 HRV 품질·IBI artifact 검사. BVP amplitude를 임상 지표로 단정하지 않음.
7. ACC 특징: magnitude mean/std, 동적 에너지, 움직임 강도, 변화량, 주요 주파수/주기성, 정지 비율(휴리스틱이면 휴리스틱 표기). 중력 성분 제거/단위 환산을 문서화.
8. 선택 TEMP 특징: 4Hz mean/std/slope; 모델에 사용했다면 실제 하드웨어 설계에 온도 센서를 추가할지 별도 결정. Sensor fusion 비교에서 TEMP의 유무를 분리.
9. 품질 제어: NaN/inf/flatline, 센서 dropout, 접촉 불량, BVP artifact, 이상치, 비율 임계값을 사전 설정. 센서별 유효 창 비율과 제외 이유를 데이터셋/피험자별 보고. 결측 대체는 training fold에서 fit.
10. 금지: 정답 라벨/미래 시점/주관적 프로토콜 위치/시각/피험자 ID를 모델 특징으로 투입 금지. 세션 파일명도 feature 금지. test 통계 전체로 정규화·특징선택 금지.
5. 모델·평가·누출 방지 프로토콜
5.1 모델 순서와 사전 설정
- B0: DummyClassifier(prior 또는 most_frequent; label imbalance 기준 baseline을 병기).
- M1: StandardScaler + LogisticRegression(class_weight='balanced', max_iter 충분히, seed=42).
- M2: RandomForestClassifier(n_estimators 200 또는 자원 기준 100, class_weight='balanced_subsample', random_state=42, n_jobs 제한).
- M3 [선택]: XGBoost; 환경 추가 의존성·튜닝 비용이 의미 있을 때만.
- 시계열 RNN/TCN, Transformer는 핵심 작업 금지. 48h 제한에서 성능향상 미확정이므로 모든 P0/P1 성공 후 선택.
- 각 모델은 같은 피험자 분할, 같은 허용 입력 특징/레이블, 같은 기준으로 비교. 비선형 모델도 fold별 결측 처리 적용.
5.2 분할
- Primary WESAD: 15-fold LOSO: 바깥 테스트 피험자 1명, 나머지 14명으로 훈련. 튜닝/threshold는 안쪽 GroupKFold 또는 지정 train 피험자 그룹으로 분리해 완료 후 바깥 test는 딱 한 번 평가. test subject의 데이터가 fitted preprocess에 들어가지 않게 sklearn Pipeline 사용.
- PhysioNet: 외부 평가가 동일한 명의 피험자 단위 group holdout이어야 한다. 모든 데이터 세션, 분할 파일 그룹 처리. 훈련/평가 겹치는 subject 없음을 assertion으로 검증. 라벨 복원 불완전하면 다중 클래스 실험 중지.
- 같은 인물의 인접·겹치는 창을 train/test에 무작위 배분하면 안 됨. Feature scaling, feature selection, hyperparameter search, threshold tuning 등은 train fold만.
- 개인화 연구는 테스트 참가자 기준 사전 지정된 비지도 baseline을 이용한 변화량 보정 정도만 1차로 허용. 실제 그 사람의 라벨을 사용한 개인별 supervised fine-tuning을 한다면, 별도 과거 학습 세션 / 미래 독립 세션 구성이 확보됐을 때만 가능. WESAD의 순차 단일 상태 세션을 임의로 쪼개면 개인화 일반화 증거가 되지 않음.
5.3 지표
- Primary D1: 피험자별 balanced accuracy, stress F1, macro-F1, stress precision/recall, stress AUPRC(양성 비율 병기). AUROC는 양성·음성 모두 있을 때만 계산. 피험자당 평균과 분산/범위/95% CI(가능한 경우, 피험자 단위 부트스트랩; n=15 작음 명시), pooled out-of-fold 보조 수치.
- Primary D2: 세션 구간 복원이 신뢰될 때 confusion matrix(stress, aerobic, anaerobic, rest 조건을 어떻게 구성했는지 명시), 운동 상태에서 stress로 잘못 예측된 비율, macro-F1, group holdout. 운동 유형별 혼동 분석.
- Ablation: EDA / BVP / ACC / EDA+BVP / EDA+ACC / BVP+ACC / EDA+BVP+ACC; TEMP는 선택·분리. 각 실험에서 동일한 유효 창 집합 사용 여부 명시하고 공정 비교 세트를 별도로 생성.
- Runtime: 피험자 단위 특성 추출 시간/추론 시간(Windows CPU; 환경과 단위, 1창당)을 실제 측정. 이는 MCU latency나 배터리 성능이 아니다.
- ASD-only metrics: event-level recall/precision, false alarms/day, lead time은 N/A: 관련 onset 주석 데이터 없음으로 표기. WESAD classification precision을 'ASD alert precision'으로 명명 금지.
- Threshold: 모델 점수 threshold는 outer train에서만 정하고 outer test는 고정. 0.5를 기준선으로 쓰되 클래스 불균형/확률 calibration 주의.
5.4 통계적 정직성
- 중첩 창 수를 독립 표본수처럼 부풀리지 않음; 독립 분석단위는 참가자. n=15에서는 p-value 남발 금지.
- 각 실험의 가설·primary metric·분할·센서 조합을 EXPERIMENT_REGISTRY.md에 실행 전 기록. 실험 후 가설 변경 시 exploratory 표시.
- 우수한 모델 한 개의 최고 성능만 보고하지 말고 피험자별 하위 25%/최악 결과와 baseline 대비 차이 공개.
- 실패 실험 로그도 보존하고 개선/불개선 모두 기술.
6. 48시간 실행 순서 (의존성 및 강제 게이트)
단계	시간 예산(예시)	수행	종료 게이트 / 중단 규칙
P0-0 인수·정책·환경	0–3h	기존 kit 유무 검사, requirements, py/CUDA, 폴더/시드/라이선스	기존 코드 미존재 시 새 최소 pipeline로 대체; 링크 접근 실패 기록
P0-1 D1 데이터	3–8h	WESAD 공식 출처 다운로드/해시/피험자 수/손목 신호 검수	15명 및 라벨·시간 정렬 검증. 다운로드 불가 시 D2 선착수
P0-2 특징 추출	8–13h	strict 60s/15s window, QC, 특징 CSV/Parquet	센서 특징 NaN 비율·시간범위·라벨 규칙 unit tests 통과
P0-3 LOSO + ablation	13–20h	Dummy/LR/RF, nested group-safe selection, 피험자별 결과	누출 검사 통과, 모든 외부 fold 산출; 실패하면 단순 LR 우선
필수 휴식	20–27h	수면 7h	야간 무리/24h 무휴식 요구 금지
P1-0 D2 데이터	27–32h	PhysioNet 공식 다운로드, 세션/버전/태그 확인	tags.csv로 유효 구간 재구성이 안되면 supervised D2는 보류
P1-1 운동 혼동 실험	32–39h	group holdout, ACC ablation, confusion matrix	운동 라벨 타당성 + 참가자 분할 테스트 통과
P1-2 재생/설계	39–43h	원기록 replay 결과(있으면), MCU/BLE/LTE/BOM/배터리 이론 계산	출력 출처·모델 목적·설계 가정 명시
P0-4 통합 보고	43–48h	산출물 품질 게이트, 결과표, 11장 발표용 핵심문구	보고서·코드·로그·근거 출처와 N/A 표시 완료


시간표는 작업 예산이지 실제 실행시간을 보장하는 값이 아니다. P0 최종 연구 결과가 없는데 P2 고급 모델에 시간을 쓰지 말 것. 실제 데이터는 URL/라이선스 때문에 다운로드 실패할 수 있으므로 BLOCKERS.md와 fallback 기록 필수.

7. 담당 에이전트 역할 (병렬 가능한 영역만 병렬)
AGENT A — Data Steward
- 원본 출처·접근·라이선스·해시·폴더·데이터 제약, WESAD 및 PhysioNet 각 참가자/세션 인덱싱.
- 파일: DATASET_AUDIT.md, DATA_CARD_WESAD.md, DATA_CARD_PHYSIONET.md, subject_manifest.csv, segment_manifest.csv.
- 최우선 게이트: 교차 세션 참가자 동일성·UTC/tags·레이블 타당성. 불확실하면 미확인으로 막는다.
AGENT B — Signal/Feature Engineer
- 원시 신호 읽기, 시간 정렬, causal window/센서별 특징/품질제어, 전처리 테스트.
- 파일: src/loaders/, src/features/, src/quality/, tests/test_time_alignment.py, tests/test_no_future.py.
- 금지: 가슴 센서 묵시적 포함, 임의 중첩 보간, 정보 누출.
AGENT C — Evaluation Scientist
- Dummy/LR/RF, LOSO, 그룹 검증, 누출/훈련-테스트 assertion, 센서 ablation 및 D2 운동 혼동.
- 파일: src/train.py, src/evaluate.py, configs/, results/per_subject.csv, results/ablation.csv, results/metrics.json, tests/test_group_split.py.
- 조건: A/B가 만든 정당한 자료만 사용. 점수 80% 미달이라고 결과 변경 금지.
AGENT D — Systems/Design Analyst
- 제품 미제작 전제 BOM·회로/블록다이어그램·개인정보 보호·BLE/LTE 옵션·에너지 예산·MCU 후보 비교.
- 파일: PRODUCT_DESIGN.md, BOM.csv, ASSUMPTIONS.md, SYSTEM_FLOW.md, ALERT_UX.md.
- 구분: 가격·배터리·통신거리·MCU 추론시간은 실측이 아니라 견적/이론/미검증 항목으로 표기. 오디오 선택사항이며 raw 오디오 저장 금지 제안.
AGENT E — QA/Evidence & Presentation
- 모든 주장에 proof level 라벨: L1 직접 실험 / L2 동료심사 선행 연구 / L3 계산 기반 설계 추정 / L4 가설·미검증.
- 실험 로그·논문 링크 검증, 그림/슬라이드 목차 11장용 메시지, 실패도 보고.
- 파일: EVIDENCE_LEDGER.md, QUALITY_REPORT.md, PRESENTATION_CLAIMS.md, FINAL_REPORT_KO.md.
통합책임자: 작업 블로커 처리, 의존성 관리, 학습 실행 여부 확인, 결과 병합, 증거 기반 완료 판정. 타 에이전트가 불완전한 값을 '추정'으로 채울 수 없게 한다.
8. 제품/시스템 설계에 대한 상위 결정
8.1 최소 제품 구조
- 기본 센서: EDA(피부 전도), PPG(BVP/HR; HRV는 신호품질 확인 필수), IMU(3축 움직임). TEMP는 센서융합 확장 옵션, Audio는 프라이버시 부담 때문에 옵션이며 WESAD·PhysioNet 실험으로 검증할 수 없음.
- MCU: BLE 지원 MCU + 전처리/특징 추출 + 경량 모델(초기엔 LR/작은 RF 또는 작은 정수화 NN 비교). 칩 품명·메모리·ADC·PPG/EDA 인터페이스·연산량·BLE·저전력 모드를 사양서로 확인한 뒤 후보 2~3개 비교. 오디오/ LTE는 소비전력·복잡도 별도 시나리오.
- BLE 기본: 아동 손목 → 보호자·교사 앱(휴대전화·태블릿 가까이) → 앱 알림; 스마트폰 없는 독립 원거리 연결을 기본 기능으로 주장 금지.
- LTE 확장: 모뎀·SIM/eSIM·데이터 요금·안테나·전류 피크·인증이 모두 추가. 하드웨어 제조 단가/BOM과 운영비 월요금 별도.
- 알림 정책: 신호 불량은 위험 경보 대신 '측정 불가/재착용' 상태, 위험도는 불확실성 표시, 알림 피로 줄일 재알림 억제·cooldown. 의료적 확정 진단·자동 강제 개입 금지.
- 보안: 최소 신호 수집·가능하면 로컬 추론, 원음 비저장, 사건 라벨 접근통제·암호화·보존기간 계획. 아동·보호자 동의, 임상연구 윤리/추가 승인 없이는 실제 아동 데이터 수집 및 유발 실험 금지.
8.2 BOM 산출 방법
1. 센서 패키지별 모듈/칩 공급처·가격·통화·조회일·MOQ·출처 링크 기입.
2. 두 시나리오 BLE-only vs BLE+LTE 모두 전원·충전·PCB·외장·스트랩·조립/인증·개발비의 포함/제외 범위를 명시.
3. 배터리 추정은 활성/슬립/BLE/LTE 이벤트 duty cycle × 전류 × 시간, 배터리 효율·온도·열화 시나리오를 명시한 계산으로 제공. 실제 수명 보장 표현 금지.
4. MCU RAM/Flash/실시간 성능은 모델 파일 크기뿐 아니라 runtime buffer·센서 버퍼·통신 stack·메모리 여유 포함. 하드웨어를 제작하지 않으므로 계산을 검증 결과와 분리.
9. 결과 파일 계약 / 재현성
필수 프로젝트 구조
AutismResearch/
  README_KO.md
  RESEARCH_MASTER_SPEC.md
  EXECUTION_LOG.md
  BLOCKERS.md
  EVIDENCE_LEDGER.md
  EXPERIMENT_REGISTRY.md
  DATASET_AUDIT.md
  src/{loaders,features,quality,train.py,evaluate.py,replay.py}
  tests/{test_no_future.py,test_group_split.py,test_label_bounds.py,test_sensor_ablation.py}
  configs/{wesad.yaml,physionet.yaml}
  data/{raw,processed}/                  # raw는 배포·커밋 제외
  results/{metrics.json,per_subject.csv,ablation.csv,figures/}
  design/{PRODUCT_DESIGN.md,BOM.csv,SYSTEM_FLOW.md,ASSUMPTIONS.md}
  reports/{FINAL_REPORT_KO.md,PRESENTATION_CLAIMS.md,QUALITY_REPORT.md}
  requirements.txt
파일 내 형식 요구
- metrics.json: dataset_version, participants, subject_ids, task_definition, labels, window_s, hop_s, sensor_set, outer_split, inner_split, model, seed, prevalence, precision, recall, macro_f1, balanced_accuracy, auprc, auroc, latency_ms, notes.
- per_subject.csv: dataset, subject_id, n_windows, positive_windows, negative_windows, model, sensors, precision, recall, f1, auprc, auroc, balanced_acc, excluded_reason. undefined metric은 빈 값/NaN + 이유.
- ablation.csv: task, split, sensor_combination, n_subjects, n_windows, macro_f1_mean, auprc_mean, balanced_accuracy_mean, delta_vs_baseline, validity_notes.
- figures/: confusion_matrix.png, subject_performance.png, sensor_ablation.png, model_comparison.png, quality_flow.png; 그림마다 dataset/label/split를 캡션에 표기.
- 재현 명령·로그·환경 freeze(py --version, 라이브러리 버전, CPU/GPU, seed, git commit) 기록.
- 합성 자료 결과는 반드시 경로와 파일명에 synthetic_test를 표기하며 실제 평가 결과와 한 표에 섞지 말 것.
- 자료가 접근 불가하면 원본을 대체 생성하지 말고 BLOCKERS.md에 download error, 대체 원천·다음 조치, P0 결과의 제한을 기록.
10. 품질 게이트 및 중단 규칙
G0 — 출처: 각 외부 데이터 출처/라이선스/버전 기록. 공식 파일 확인 실패 시 '확인' 표기 금지.

G1 — 원본 무결성: 센서 Hz·피험자 수·레이블·세션 타임라인·유효 창 수 점검. 누락 데이터 신뢰 불가 시 해당 피험자/세션 제외 로그.

G2 — 시간적 정직성: t에서 사용하는 모든 특징 원시 샘플 시간 ≤t. 조건 전환 창 제외. replay에서도 look-ahead 금지.

G3 — 그룹 분리: train과 outer test subject ID 교집합 0. 개인 baseline 계산은 사전 선언한 과거 비지도 부분만.

G4 — 성능 정직성: 다중 랜덤 시드보다 outer subject fold 안정성을 우선, Dummy·LR·RF 모두 결과, AUPRC base rate 병기. 합성 테스트 결과 분리.

G5 — 커뮤니케이션: 멜트다운 직접 예측, 60초 실제 lead, 오경보/day, 배터리 실측은 ‘미검증’ 처리. 실험이 성인 스트레스 분류임을 제목·그래프·결론 모두에 기재.

G6 — 납품: 파일 존재, 실행 예제 성공 여부, metric 생성 확인, 미실행 이유, 발표 메시지 근거 연결. 코드 테스트가 일부 실패했다면 실패 상태 그대로 보고.
금지사항: test set 탐색 반복으로 리더보드 최적화, 한 명 피험자의 윈도를 랜덤 분할, 결과 변조, 실측하지 않은 MCU 값 표기, 전조 없는 사건 비율 임의 추정, 데이터 전용 라이선스 무시, 아동 스트레스 유발 실험, 모든 파일 삭제/리팩터링으로 기존 kit 파괴.

11. 프레젠테이션 11장 구조 매핑 (실제 결과만 삽입)
1. 도입·문제: 감각 과부하와 멜트다운·위험행동의 관계 및 필요성 (과도한 인과 단정 금지).
2. 선행 연구: Goodwin, Imbiriba, Kim; 각각 행동 정의·horizon·AUROC/AUPRC·한계 구분.
3. 목표: 개인별 위험 패턴 감지 시스템 설계, 약 60초 실제 경보는 장기 가설.
4. 하드웨어 설계: EDA/BVP/IMU, optional TEMP/Audio, BLE 기본, LTE 옵션, BOM 추정.
5. 동작 시나리오: signal quality → baseline → movement compensation → risk score → alert → caregiver review.
6. AI 연구 설계: D1/D2, preprocessing, sensor ablation, group CV, 공정 모델 비교.
7. 직접 수행 실험: 모델·센서별 raw 결과/환경과 재현 명령. 제품 하드웨어 시연 아님.
8. 성과와 검증 범위: 직접 측정된 성인 stress/exercise classification; ASD alert metrics는 N/A.
9. 기대효과: 선제적 대응의 가능성; 멜트다운 감소는 가설.
10. 제약 및 연구 로드맵: 실제 아동 onset 데이터·오경보/lead time·현장 연구·윤리.
11. 결론: 공개자료 기반 센서융합 가능성 검토와 기술 설계, 향후 실증 연구를 위한 기반 확립.
12. 실제 ASD 조기경보 연구로 확장할 때의 명세 (이번 48h 미실행)
1. 동의·아동 assent·기관 윤리 승인·개인정보 최소수집, 감각 과부하/멜트다운을 의도적으로 유발하지 않음.
2. 학교/가정 등 자발적 일상생활에서 다일자·다장소 측정, 착용 편의와 데이터 소실 보고.
3. 사건 정의: 자해/공격/정서조절곤란/멜트다운의 행동 운영 정의와 onset/offset, 관찰자 일치도, context annotation.
4. 모델 입력 과거 180s, 실시간 추론 hop 10–15s; 미래 예측 horizon 예컨대 0–60s; 별도 실제 첫 알림 lead time 기록.
5. 실제 alert episode 병합/재알림 cooldown, 단위 사람-일 기준 오경보율. 시간창 분류 accuracy와 실제 사건단위 recall을 분리.
6. 사건 미검출 원인 분류: weak precursor, sensor failure, annotation uncertain, model false negative. weak-signal %는 실제 분모와 독립 임계값 정의가 확보된 뒤에만 산출.
7. 피험자 및 날짜 holdout, 사전 지정 threshold, calibration, 아동별 subgroup 분석, 비개입 silent-mode 현장 검증 후 필요시 중재 효과 실험.
13. 이 명세서의 중요한 미확정 사항
- 기존 ZIP의 코드 품질: /workspace/AutismResearch/AutismResearch_48h_Kit.zip을 실제로 제공받지 못해 미감사. 먼저 추출·구조 확인, 수정이 필요한 곳만 수정.
- 데이터 다운로드 성공 여부: 현재 사용자의 로컬 저장소·네트워크 확인 불가. 상위 연구는 데이터의 공식 위치/형식 확인까지 완료, 실제 로컬 수집은 실행 에이전트 책임.
- PhysioNet tags.csv 구간의 정확한 프로토콜 맵: 원문에 태그/프로토콜과 두 버전이 제공되나 각 세션의 상태 레이블 복원 코드는 직접 실행해 확인해야 함. 복원 실패 시 D2 supervised 결과 발표 금지.
- 센서 보드·제조 BOM 실제 단가: 부품 모델과 조회일이 정해지기 전 확정 금지. 실물 제작을 하지 않으므로 견적표만 작성.
- ASD 소아 리드타임/오경보 검증: 공개 데이터의 정보 부족으로 이번 범위에서 불가. 별도 임상 설계로 유지.
14. 1차 출처와 검증 URL
- Goodwin 2019 (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC6988899/
- Imbiriba 2023 (JAMA Netw Open/PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC10739066/
- Kim 2026 (Scientific Reports): https://www.nature.com/articles/s41598-026-50079-z
- WESAD 원 제공 사이트: https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html
- WESAD UCI: https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection
- PhysioNet 데이터셋 v1.0.1: https://physionet.org/content/wearable-device-dataset/1.0.1/
- PhysioNet 참고 논문: https://doi.org/10.1038/s41597-025-04845-9
출처 규칙: 각 그래프/수치는 논문/데이터 원문에서 검증 후 EVIDENCE_LEDGER에 DOI, 페이지/섹션, 정확한 표본, 과제, 분할, 단위를 기록할 것. 이 문서의 '연구 권고 기준'과 '논문 검증 수치'를 혼동하지 말 것.
최종 지시: 문서에 명시된 P0/P1을 끝낼 때까지 모델 복잡도를 불필요하게 늘리지 않는다. ‘연구 완료’는 실험 결과를 꾸미지 않고 재현 가능한 데이터·코드·검증·한계를 갖춘 상태를 의미한다. 실험 실패를 숨기거나 학습 결과를 예측해서 보고하는 것은 모든 품질 게이트 위반이다.