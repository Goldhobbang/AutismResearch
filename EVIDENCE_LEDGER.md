# 근거 원장

확인일: 2026-10-23. 이 원장은 외부 문헌의 수치, 직접 실행한 검사, 계산에 의한 설계, 미검증 가설을 구분한다. 기존 보고서의 내용을 자동으로 확인된 사실로 승격하지 않는다.

## 근거 등급과 읽는 법

| 등급 | 의미 | 이 프로젝트에서의 적용 |
|---|---|---|
| L1 | 직접 실행·관찰한 결과 | 코드 검사, 데이터 감사, 실제 학습 결과. 합성 입력 검사는 반드시 L1-synthetic으로 표시 |
| L2 | 동료심사 논문에서 보고된 근거 | 원논문의 모집단·과제·분할·지표 범위를 함께 기입. 우리 모델의 성능 아님 |
| L3 | 명시한 가정으로 계산한 설계 추정 | 센서 버퍼·배터리 시나리오·BOM 소계. 실물 검증 아님 |
| L4 | 가설·미검증·현재 확인 불가 | 아동 멜트다운 60초 조기경보, 임상 효과, 제품 착용성 |

공식 데이터 문서와 API 문서는 **DOC**로 별도 표시한다. 동료심사 논문이 아니라는 이유로 거짓인 것도 아니고, L2로 바꿔 표시할 근거도 없다. 가격은 공급처 문서상의 관찰값이고 제품 전체 비용 추정은 L3이다.

## 핵심 논문 수치 추적

| ID/등급 | 확인한 주장·정확한 범위 | 원문 위치·단위 | 출처와 확인 한계 |
|---|---|---|---|
| LIT-01 / L2 | Goodwin: ASD 20명, 69회·87시간, 타인 공격 548 episode. 과거180초로 향후60초 구간을 예측한 ridge LR의 global AUROC .71, person-dependent .84 | Abstract; Results/Data Collected; Prediction Performance; Table3. AUROC 단위0–1 | [Goodwin 2019, DOI10.1002/aur.2151](https://pmc.ncbi.nlm.nih.gov/articles/PMC6988899/). PMC 색인 원문 재확인. 전체 특징에는 행동 발생 이력도 포함. global을 미학습 참가자 LOSO로 부르지 않음 |
| LIT-02 / L2 | Imbiriba: 분석70명, 4병원, 429회·497시간; SIB3,983/ED2,063/ATO619=6,665개 행동. 미래180초 AFV-onset 조건에서 LR AUROC .80[.79–.81]은 PM/session split, .77[.73–.81]은 PM/leave-individuals-out | Results/Table2; Experiment Outcomes. AUROC와95%CI | [Imbiriba 2023, DOI10.1001/jamanetworkopen.2023.48898](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2813185). 본문 재확인. AFV는 관찰 행동 이력을 포함. 해당 조건 최고 SVM은 각각 .85/.82이므로 LR이 모든 조건에서 최고라고 주장하지 않음 |
| LIT-03 / L2 | PreMAC: ASD7명, 1인 평균54.2분·전조25.9회. RF의 개인별 accuracy98.51%, group82.36%. 목표는 관찰 가능한 전조이며 실제 멜트다운 onset과 다름 | §4.3; §5.2–5.3 및 모델별 accuracy 표. 시간은분, accuracy는% | [Zheng 2021, DOI10.3390/s21020370](https://doi.org/10.3390/s21020370), [확인한 원문PDF](https://pdfs.semanticscholar.org/d46a/79f69d008f6ab876f7d72c081f0b14945e46.pdf). 개인별 무작위 분할의 시간 상관 문제를 주의. 평균×7의 총시간은 논문 원수치가 아닌 계산 |
| LIT-04 / L2 | REACT: IDD3명, 같은 사람의 Assessment1으로 학습하고 Assessment2 기록을 실시간 스트림처럼 모사. agitation F1은0초 .7869, 20초 .6820 | Abstract; Analysis/ML model and real-time prediction; Model analysis. F1단위0–1 | [Khan 2024, DOI10.1177/20552076241287884](https://journals.sagepub.com/doi/10.1177/20552076241287884). 실제 보호자 알림을 제공한 장기 전향 시험으로 표현하지 않음. 라벨 주변 band 내 정답 뒤 나머지 band도 정답으로 처리하는 refinement가 있어 표준 raw-window F1과 동일시 금지 |
| LIT-05 / L2 | Kim: 분석9명, 손목E4+허리ActiGraph. 사건 없는10초 창,1초 hop; LOSO. SIB 이름의 표적에 공격·반복행동도 포함. 모델·센서 주효과는 유의하지 않았고 개인차가 큼 | Methods/Dataset description; Experimental design; Results; Discussion | [Kim 2026, DOI10.1038/s41598-026-50079-z](https://www.nature.com/articles/s41598-026-50079-z). H5는60초 latency+60초 폭으로 t+60~120초를 다룸. Discussion의 최장조건 false:true4.4:1(Double-LSTM)/4.6:1(RF)은 하루 오경보 수가 아님. AppendixE 세부 P/R/F1은 이번 감사에서 확인 못함 |

Imbiriba의 ED 정의에는 tantrums와 meltdowns가 포함된다고 본문 Introduction에서 설명한다. 따라서 “기존 연구가 멜트다운과 전혀 관련 없다”는 문장도 부정확하다. 다만 ED 전체·복합 행동 지표를 명확히 분리된 멜트다운 단독 성능이나 가정용60초 경보 성능으로 바꿀 수 없다.

## 과장 방지 규칙

| ID | 허용되는 해석 | 금지되는 해석 |
|---|---|---|
| INTERP-01 | AUROC는 양성·음성의 점수 순위 구분 능력 | AUROC .84 = 정확도84% 또는 경고정밀도84% |
| INTERP-02 | 미래구간에 사건이 있는지 예측 | 해당 구간 길이만큼 실제 경고 선행시간을 보장 |
| INTERP-03 | ED가 meltdown을 포함할 수 있음 | 스트레스/각성/반복행동/감각과부하를 모두 meltdown으로 치환 |
| INTERP-04 | 예측 실패는 센서·모델·라벨·전조 강도 등 복합 원인 | 미탐지율 = 생물학적 전조가 없는 아동·사건의 비율 |
| INTERP-05 | 참가자별 외부평가의 분포·AP 양성비율을 함께 제시 | 수천 중첩창을 수천 독립 피험자로 간주 |

지표 정의 근거: [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html). 구현의 `auprc`가 average precision이면 AP라고 명시하고 사다리꼴 PR-AUC와 구별한다.

## 데이터·코드·설계 연결

| ID/등급 | 근거/산출물 | 주장 가능한 범위 |
|---|---|---|
| DATA-01 / DOC | [WESAD 공식배포](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html), DATASET_AUDIT.md | 문서상 성인 스트레스 자료. 실제 로컬 파일 여부·해시는 감사 파일에서 별도 확인 |
| DATA-02 / DOC | [PhysioNet1.0.1](https://physionet.org/content/wearable-device-dataset/1.0.1/), DATA_CARD_PHYSIONET.md | stress/exercise 자료. 같은 사람 세션을 묶고 tags 기반 구간 검증 전 supervised 결과 금지 |
| CODE-01 / L1-synthetic | tests/ 및 실행 로그 | 시간 경계·미래샘플 비사용·참가자 분리·파일 출력의 소프트웨어 동작만 검증 |
| EXP-01 / 실행 대기 | 실제 D1 LOSO 결과 | 원자료 인수·훈련·평가 로그와 metrics가 생성된 뒤에만 L1 성능으로 승격 |
| EXP-02 / 실행 대기 | 실제 D2 운동 혼동 결과 | 검증된 segment_manifest와 참가자 그룹이 선행되어야 함 |
| DESIGN-01 / L3 | design/ASSUMPTIONS.md, BOM.csv | 가정된 전류·duty cycle·부품가격에 따른 설계 계산; 배터리 실측 아님 |
| GOAL-01 / L4 | 실제 ASD60초 조기경보 | 현재 onset 데이터와 전향평가 없음. 사건정밀도·재현율·오경보/일·lead time은 N/A |

## 이번 재감사의 수정 사항

1. REACT를 “둘째 방문에서 실제 보호자 경고를 검증”한 연구로 읽을 여지를 제거했다. 원문은 다음 방문 기록을 stream처럼 공급하는 모사를 설명한다.
2. Imbiriba의 ED가 meltdown을 포함한다는 점을 유지하되 복합목표와 별도 멜트다운 지표를 구분했다.
3. 저자 결론의 가능성 표현과 상용제품 검증을 분리했다. 원자료를 확보하지 못한 기존 임상논문 수치를 재현했다고 주장하지 않는다.
4. 과거120초/GroupKFold 기준 kit와 새60초/LOSO 명세를 혼합하지 않는다. 과거 합성 테스트의 통과가 새 구현의 통과를 대신하지 않는다.

## 사용자 설계 개정

DECISIONS.md에 따라 ESP32-S3 BLE-only, Mic/Light 맥락 보조, DRV2605L와 특징로그, 상하면·측면 배치로 개정했다. ESP32-S3 N8R8의 Flash8MB/PSRAM8MB는 공식개발보드문서(DOC)이며 실제 사용가능 heap/DMA RAM은 별도다. 최신 L3 에너지 예는 임의 전류에서150mAh×80%/10.8272mA≈11.08h다. 실측수명·추론시간이 아니다. 현행 소계41.80USD는 제조사sample참고와 소매일부를 합친 값이며 완제품 원가가 아니다. 출처와 제외항목은 design/BOM.csv에 있다.
