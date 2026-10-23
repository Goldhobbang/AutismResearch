# 문헌·공개 데이터 상세 부록

문헌·데이터 검토만 발췌했다. 최신 실행 설정과 ESP32-S3 BLE-only 설계는 FINAL_REPORT_KO.md 및 design/를 따른다. 실제 원자료 재현 성능은 제공하지 않는다.

## 2. 기존 논문 근거표

표의 ‘미보고’는 확인한 원문 범위에서 해당 운영 지표를 제시하지 않았다는 뜻이다. ‘미확인’은 접근 또는 세부 검증 한계로 수치를 확정하지 못했다는 뜻이다. 서로 다른 논문의 수치를 순위표로 비교하지 않는다.

| 연구·실제 목표 | 센서·환경 | 참가자·사건·시간 | 예측 시간의 뜻 | 확인한 성능 | 분할·실용성·공개성 |
|---|---|---|---|---|---|
| [Goodwin et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6988899/) — **타인에 대한 공격** | E4의 EDA, BVP/IBI, ACC; 전문 입원 병동 자연 관찰 | 20명, 6–17세; 69회 관찰, 87시간; 공격 548 episode | 과거 180초 → 향후 60초 내 공격 여부. 실제 보호자 경고 선행 시간 분포는 미보고 | 평균 AUROC: global **0.71**, 개인별 **0.84**. 정밀도·재현율·F1·오경보/일 미보고 | 표본 5-fold CV, 5회 반복. global이라는 이름을 미학습 아동 LOSO로 읽으면 안 됨. 발생 이력 특징을 포함한 설정 확인 필요. 미래 중재 시스템의 기반이라는 결론. 공개 원자료·공식 재현 코드 링크 미확인 |
| [Imbiriba et al., 2023](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2813185) — **자해(SIB), 정서조절 곤란(ED), 타인 공격(ATO)** | E4 생리·움직임 + 동기화된 관찰 라벨; 4개 입원 병원 | 분석 70명; 429회, 497시간; 6,665개 행동: SIB 3,983, ED 2,063, ATO 619 | 과거·미래 각각 60/120/180초, 15초마다 갱신. 미래 구간 내 발생을 분류 | 대표 LR AUROC **0.80 [0.79–0.81]**: 180초 미래, PM/session split/AFV-onset. 같은 미래·AFV-onset의 참가자 제외 LR **0.77 [0.73–0.81]**. 정밀도·재현율·F1·실제 오경보/일 미보고 | 세션 앞80%/뒤20%, 참가자 제외, 세션 제외를 별도 평가. **AFV에는 최근 행동 관찰·경과시간이 포함**되어 순수 손목 모델과 다름. 외래·다양한 집단 검증 필요. 보충2는 승인된 연구자의 비상업적 접근 경로 제시; 즉시 익명 다운로드와 다름 |
| [Zheng et al., PreMAC, 2021](https://doi.org/10.3390/s21020370) — **문제 행동의 관찰 가능한 전조** | E4 + WINGS 상체 움직임 + Kinect; 임상 전문가가 진행한 구조화 평가 | ASD 7명, 4–15세; 1인 평균 54.2분; 평균 전조 25.9회. 전체 약6.32시간은 평균×7의 계산값 | 전조 발생 전 90초 이내 샘플에 양성 라벨. 30/60/90/120초 비교. 실제 위험 행동까지의 시간은 불명 | RF accuracy 개인별 **98.51%**, 참가자 제외 group **82.36%**. AUROC 미보고. 집단 평균 P/R/F1 및 오경보/시간 미보고 | 개인별 무작위80:20+CV; group은 참가자 제외. 시간적으로 비슷한 샘플 분리의 낙관 편향 위험. 오프라인 파일럿이며 원자료는 개인정보·IRB 때문에 요청제 |
| [Khan et al., REACT, 2024](https://journals.sagepub.com/doi/10.1177/20552076241287884) — **agitation, 초기 불편·동요 징후** | E4 + WINGS + 마이크; IDD 아동 대상 임상 평가 | 3명, 각2회. 표2 합계 **361분, agitation 100회**(직접 합산); 첫 방문52회, 둘째48회 | 첫 방문으로 개인별 학습, 둘째 방문 기록을 스트림처럼 공급한 모사 평가. 0–30초 전 비교 | AdaBoost: 0초 F1 **0.7869**, recall **0.6997**, accuracy **0.8817**; 20초 전 F1 **0.6820**, recall **0.6375**, accuracy **0.8389**; 30초 F1 **0.5538**. AUROC·독립 정밀도 요약·오경보/일 미확인 | 새로운 아동 평가가 아니라 같은 아동의 다음 방문. 라벨 주변 시간 폭 최적화가 있어 exact onset 예측과 구분. 라벨 band 내 첫 정답 뒤 나머지 예측을 보정하는 refinement가 있어 표준 raw-window F1과도 구분. 실시간 처리 가능성 모사; 공개 원자료·공식 코드 링크 미확인 |
| [Kim et al., 2026](https://www.nature.com/articles/s41598-026-50079-z) — **SIB로 명명한 복합 행동 라벨** | E4 + 허리 ActiGraph 60Hz, 비디오 라벨; 임상 자료 재분석 | 원수집11명 중 일관된 센서 구성9명. 이번 분석의 독립 사건 총수·총시간 미확인 | 사건 없는 과거10초, 1초 이동. latency+prediction window 분리. 아래 시간표 참조 | AUROC/AUC-PR은 조건·참가자별 분포. 모델·센서 주효과 유의하지 않음. 짧은 horizon은 chance 근처; 긴 구간에서 PR 개선. 긴 설정의 false:true 비율 Double-LSTM **4.4:1**, RF **4.6:1**. P/R/F1 세부값은 부록E, 이번 조사에서 수치 미확인 | LOSO. 라벨에 자해·공격뿐 아니라 팔 흔들기·회전 등 반복 행동도 포함. 개인차·낮은 F1/MCC를 저자도 지적. 개인정보·미성년자 윤리 제한으로 요청+기관 승인 필요 |

Goodwin 원문 일부는 PMC 직접 열람이 차단되어 PubMed와 PMC의 색인된 원문 Results/Methods로 교차 확인했다. PreMAC은 [원문 PDF](https://pdfs.semanticscholar.org/d46a/79f69d008f6ab876f7d72c081f0b14945e46.pdf)를 확인했다. Imbiriba 본문과 REACT 출판사 전문, Kim 출판사 전문을 확인했지만 일부 보충파일은 접근하지 못했다. **접근하지 못한 부록 값을 ‘미보고’로 바꾸지 않았다.**

PreMAC의 혼동행렬은 행·열 제목과 FP/FN 기호가 일치하지 않는 부분이 있고 개인표는 예시 모델이다. 따라서 이를 임의로 고쳐 집단 평균 정밀도·재현율로 제시하지 않았다. 보고된 accuracy와 전조 목표 자체를 우선 인용한다.

Kim의 설정은 다음과 같다. 관찰 종료를 t라 할 때 최소 대기시간과 양성 판정 구간을 분리해야 한다. [원문 Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC13323699/).

| 설정 | latency | 미래 판정 구간 폭 | 사건을 찾는 시각 |
|---|---:|---:|---|
| H1 | 3초 | 7초 | t+3 ~ t+10초 |
| H2 | 15초 | 15초 | t+15 ~ t+30초 |
| H3 | 30초 | 30초 | t+30 ~ t+60초 |
| H4 | 60초 | 30초 | t+60 ~ t+90초 |
| H5 | 60초 | 60초 | t+60 ~ t+120초 |

H5는 최소120초 전 경고가 아니다. 구간 폭이 늘면 양성 비율도 달라지므로 PR 성능 향상이 ‘더 먼 미래가 본질적으로 더 쉽다’는 증거가 되지는 않는다. 각 조건의 양성 비율을 함께 비교해야 한다. 논문이 제시한 4.4:1도 윈도 기반 false:true 비율이며, 독립 경고 4.4회/일이라는 뜻이 아니다.

추가 최신 근거: [Kartha et al., 2026 프리프린트](https://arxiv.org/html/2605.17618v1)는 9명의 아동·청년, 교실 자료 약110.7시간, EDA·ACC·체온으로 미래 행동 분석을 보고한다. 현장 수집의 진전이지만, 이 보고서는 프리프린트임을 명시하며 상용 제품 검증으로 취급하지 않는다.

**AUROC 0.84의 해석:** 해당 평가 분포에서 양성 표본의 점수가 음성보다 높게 매겨지는 순위 구분 능력이다(동점 처리 포함). ‘84%를 맞혔다’, ‘경고의84%가 맞다’, ‘사건84%를 예측했다’ 중 어느 것도 의미하지 않는다. 정밀도는 임계값·사건 빈도에 의존한다. [지표 정의](https://scikit-learn.org/stable/modules/model_evaluation.html).

## 3. 실제 이용 가능한 데이터셋 비교표

접근 확인은 공식 배포 페이지와 조건 확인을 뜻한다. 이번 작업에서 모든 대용량 파일을 실제로 내려받아 무결성 검사한 것은 아니다. 데이터셋 페이지의 표시 크기와 디스크 여유 공간 추정은 구분한다.

| 데이터·공식 URL | 접근·인원 | 신호·주파수 | 라벨 의미 | 크기·이용 조건 | 이 프로젝트에서의 용도·노트북 제약 |
|---|---|---|---|---|---|
| **WESAD**: [UCI](https://archive.ics.uci.edu/dataset/465/wesad), [원배포기관](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html) | 공개 다운로드, 15명 | 손목 BVP64Hz, EDA4Hz, TEMP4Hz, ACC32Hz. 가슴 ECG/EDA/EMG/호흡/체온/ACC700Hz | 실험 프로토콜의 baseline/stress/amusement, 명상·전환도 별도 존재. 멜트다운 라벨 없음 | ZIP **2.5GB**. 원배포처: 과학적·비상업적 사용, 출처 표기. UCI가 원출처 조건 참조 | **1순위: 스트레스 분류·센서 비교**. 멜트다운 예측 직접 학습 불가. 1명씩 pkl 로드; 원자료·압축해제·환경 포함 디스크20GB 여유는 계획 추정 |
| **Wearable Exam Stress**: [PhysioNet 1.0.0](https://physionet.org/content/wearable-exam-stress/1.0.0/) | 공개, 10명, 시험3회 | E4 EDA/TEMP4Hz, BVP64Hz, ACC32Hz, HR1Hz, IBI 비정기; 파일 두번째 행/info 확인 | 시험 종류·시각·점수. 연속 시점별 스트레스 정답이 아님 | 비압축 **82.1MB**, ZIP82.0MB; ODC Attribution1.0 | 신호처리·현장 잡음·시험 성적 연구. 시험 전체를 양성 스트레스로 단정 금지. RAM 부담 작음; 직접 멜트다운 예측 불가 |
| **Induced Stress & Structured Exercise**: [PhysioNet 1.0.1](https://physionet.org/content/wearable-device-dataset/1.0.1/), [Zenodo 원기록](https://zenodo.org/records/13993658) | 공개. 스트레스36명, 유산소30명, 무산소31명; 합계를 독립97명으로 세지 않음 | E4 BVP64, ACC32, EDA/TEMP4Hz, HR/IBI | 과제·휴식·운동 구간 tags, 자기보고 스트레스 | PhysioNet 비압축 **247.4MB**, ZIP69.7MB; ODC Attribution1.0. Zenodo는 같은 자료 계열 | **예비 데이터 및 운동 오경보 분석**. S02중복, f07착용문제 등 공식 제약 확인. v1/v2 프로토콜 차이. 멜트다운 예측 불가. 현재 WESAD 로더와 형식 달라 변환 필요 |
| **PPG-DaLiA**: [UCI](https://archive.ics.uci.edu/dataset/495/ppg%2Bdalia), [기관](https://ubi29.informatik.uni-siegen.de/usi/data_ppgdalia.html) | 공개, 15명 | 손목 BVP64, ACC32, EDA/TEMP4Hz; 가슴 ECG/호흡/ACC700Hz | 일상 활동, ECG 기준 심박. 스트레스·위험 사건 라벨 아님 | ZIP **2.7GB**. UCI는 CC BY4.0, 기관 페이지는 비상업적 조건: 배포본 조건 차이를 기록하고 상업 이용 전 확인 | 움직임에 강한 PPG/심박 추정 검증용. 직접 스트레스 또는 멜트다운 분류 정답 없음. 1명씩 처리 권장 |
| **Imbiriba/AIC 연구자료**: [논문·보충2](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2813185) | 승인된 비상업적 연구자 접근 안내; 즉시 공개 ZIP 미확인. 분석70명 | E4 생리·ACC와 관찰 시간표. 실제 배포 파일의 주파수·스키마 재확인 | SIB/ED/ATO onset·offset | 용량 미확인; 연구계획 승인·Simons 정책. [SFARI 자료 접근 안내](https://www.sfari.org/resource/autism-inpatient-collection/)에서 해당 시계열 포함 여부 확인 필요 | **위험 행동 예측 재현 후보**. AIC의 일반 유전·표현형 자료 공개가 E4 전체 시계열 공개를 뜻하지 않는다. 전체 E4·사건시간이 함께 제공되는지 먼저 확인. 멜트다운 단일 정답은 아님. 48시간 확보를 전제하지 않음 |
| **PreMAC / Kim 원자료**: [PreMAC](https://doi.org/10.3390/s21020370), [Kim](https://www.nature.com/articles/s41598-026-50079-z) | 저자 요청제; 각각7명/분석9명 | PreMAC E4+WINGS+Kinect; Kim E4+허리ACC | 각각 전조/혼합SIB 계열 라벨 | 크기·재사용 상세 조건 미확인. 개인정보·기관 승인 제약 | 승인·스키마 확인 후 제한적 재현 가능. 48시간 주 데이터로 부적합. 원자료가 없으면 정확한 재현이라고 주장하지 않음 |
| **Heart Rate as a Predictor… 데이터 후보**: [IEEE DataPort DOI](https://doi.org/10.21227/5B7R-8J60), [관련 원논문](https://www.mdpi.com/2218-6581/12/2/55) | DOI 등록 확인. 이번 조사에서 DataPort 원페이지·파일 접근 조건 미확인. 관련 논문은5명; 배포본과 동일 범위인지 미확인 | 관련 연구 E4 생리·움직임. 배포본 주파수 미확인 | 사회적 로봇·장난감 상호작용 중 challenging behavior **탐지** 연구 | 파일 크기·라이선스·가격/기관구독 필요 여부 미확인 | 정밀 onset과 충분한 사건 전·비사건 자료 확인 전 조기예측용으로 확정 불가. 48시간 일정에서 제외 |
| **ASC-Emotion**: [저자 Kaggle](https://www.kaggle.com/datasets/zakiaturabee/asc-emotion), [2026 원논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC13450243/) | 익명화 자료 공개를 논문에서 명시. Kaggle 본문·파일 상세가 도구에 표시되지 않음 | 영상에서 추출한 얼굴·몸 point-cloud 계열. 손목 EDA/PPG 자료가 아님. 정확한 fps·인원 이번 조사 미확인 | 학습 중 정서·참여·각성 관련 상태 | 용량·배포 라이선스·로그인 조건 미확인 | ‘autism/meltdown’이라는 소개만으로 60초 사전 생체신호 학습자료라 판단 금지. 영상 연구 후보이며 본 코드에 직접 입력 불가 |

**공개 위험 행동 데이터에 대한 정확한 결론:** 이번 조사에서 **즉시 내려받아 사용할 수 있고, 손목 생체신호·위험 행동 onset·충분한 사건 전 구간·긴 비사건 관찰을 모두 확인한 멜트다운 전용 데이터셋은 확보하지 못했다.** 이것은 전 세계에 관련 데이터가 전혀 없다는 선언이 아니다. 승인제 임상 자료와 DataPort/Kaggle 후보는 존재한다. 48시간에는 WESAD로 분석 방법을 검증하고, 후속 연구에서 보호자·연구기관과 정답 정의 및 접근 절차를 진행하는 것이 현실적이다.

