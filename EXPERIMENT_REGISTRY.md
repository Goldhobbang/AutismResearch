# 사전 실험 등록

등록: 2026-10-23. 실제 WESAD/PhysioNet 원자료 다운로드 및 새 파이프라인 성능 관찰 **이전**에 고정한 계획. 기존 합성 검사 성능은 선택 근거로 사용하지 않는다. 사용자 마스터 명세가 기존 120초/GroupKFold 계획을 대체한다.

## P0 / D1 / H1, H2, H4

- 과제: 성인 WESAD 실험실 stress(2) 대 baseline(1)+amusement(3)의 현재 상태 분류. 가슴 신호, TEMP, 마이크는 입력하지 않는다.
- 예상 참가자: S2–S11, S13–S17의 15명. 실제 파일과 검사 결과가 일치할 때만 15명 결과로 표기한다.
- 60초 과거 창, 15초 이동, 반개구간 `[t-60,t)`로 샘플을 선택한다. t는 사용한 마지막 샘플 다음 경계다. 모든 입력 시각은 t 이하이며 t 이후 샘플을 사용하지 않는다. 명세의 `(t-60,t]`와 표기 방식이 다르므로 이 규약을 코드/테스트/도표에 일관되게 사용한다.
- 창 전체가 동일한 1/2/3 상태일 때만 정답을 부여한다. 그 외 구간은 unknown으로 보존하되 주평가에서 제외한다.
- 공정 비교: 모든 센서 조합에 동일한 `qc_common_valid` 창 집합을 적용한다. 센서별 실패율·전체 평가 제외율을 별도 공개한다. 공통 품질 통과 창에서만 평가하므로 전체 착용시간 성능과 같지 않다.
- 특징과 QC는 `src/features/` 및 `src/quality/` 문서에 사전 고정한다. 식별자·시각·상태·품질 플래그는 모델 입력이 아니다.
- 외부 15-fold LOSO, 내부는 외부 훈련 참가자만의 고정 group holdout. median imputer와 scaler는 내부 train에서만 fit한다. 내부 validation에서 F1 최대 임계값을 선택하며 동점은 높은 임계값을 택한다. validation을 합쳐 다시 fit하지 않는다. .5 임계값 결과도 보조 보고한다.
- 기본 모델: Dummy prior; LR(C=1, L2, balanced, seed42); RF(200 trees, depth8, min leaf10, balanced_subsample, seed42, 병렬8 이하). 튜닝 탐색 없이 고정한다.
- 모델 비교: 전체 센서로 Dummy/LR/RF. 센서 기여: LR로 EDA/BVP/ACC/EDA+BVP/EDA+ACC/BVP+ACC/전체의 7종. 동일 LR 전체 조합은 중복 실행하지 않아 총9개 설정이다.
- H1 판정: 동일 참가자에서 LR/RF와 Dummy의 balanced accuracy·stress F1·AP 차이와 분포를 보고한다. AUROC/AP .5가 아닌 **AP 기준선은 해당 양성비율**이다. AP는 average precision이며 사다리꼴 PR 면적과 구분한다. 모든 참가자에 일률적인 최소 성능 보장을 하지 않는다.
- H2 판정: 같은 참가자에서 전체 센서−EDA의 AP 차이를 주 비교로 삼고 7종 모두 보고한다. 개선/동률/악화 인원, paired 차이와 탐색적 참가자 bootstrap95%CI를 보고한다. 다른 센서 대비 비교는 보조 분석이다.
- H4: 외부 LOSO 참가자별 지표, 평균/표준편차/범위/하위25%/최악값을 공개한다. 검출되지 않은 임상 사건 비율은 계산하지 않는다.
- 지표: balanced accuracy, stress F1, macro-F1, stress precision/recall, AP, AUROC, prevalence, confusion counts; baseline/amusement 별 false-stress rate. pooled OOF는 보조다. undefined 지표는 이유와 함께 null/NaN 처리한다.
- 추론/특징 시간은 실행한 환경을 명시한다. 이 클라우드 측정치를 Windows 노트북 또는 MCU 실측으로 바꾸어 쓰지 않는다.

## P1 / D2 / H3, H4

- 공식 tags·프로토콜·파일·참가자 그룹을 감사한 **verified task segments**에서만 분석한다. 전체 세션을 stress 또는 exercise라고 일괄 라벨링하지 않는다.
- 분석 후보: task stress 대 aerobic/anaerobic exercise의 이진 분류. rest는 이 이진 주과제에서 제외한다. 운동 유형별 false-stress rate를 분리한다. 이 구성은 4-class 분류 결과가 아니다.
- 비교: LR EDA+BVP 대 LR EDA+BVP+ACC. 동일 창·동일 그룹·동일 threshold 선택 규칙. 운동 오분류 감소를 사전에 전제하지 않는다.
- 모든 세션/분할 파일은 같은 실제 참가자 group에 묶는다. 동일성·유효 태스크 시간·라벨 근거가 불명확하면 D2 supervised 실행을 중지하고 N/A로 보고한다.
- 모델은 조건/장소/발성/프로토콜의 차이를 학습할 수 있다. 감각 과부하·놀이·멜트다운 일반화 주장은 금지한다.

## 설계·선택 실험·중단

H6은 사양서와 계산을 이용한 L3 설계 분석, H7은 L4 장기 가설이다. ASD event recall/precision, false alarms/day, 실제 lead time은 이번 데이터로 N/A다. H5 개인 기준선 보정, 30/120초 민감도, 3-class, XGBoost, 신경망은 P2이며 P0/P1이 막히거나 미완료이면 실행하지 않는다. 사후 변경은 날짜·이유와 함께 탐색으로 기록한다.
