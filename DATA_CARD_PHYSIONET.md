# PhysioNet Stress/Exercise 데이터 카드

상태: **공식 문서·constraints·참가자표 웹열람 / 원본 CSV·notebook 미확보 / supervised 구간 복원 보류**.

출처: [v1.0.1 공식 페이지](https://physionet.org/content/wearable-device-dataset/1.0.1/), DOI [10.13026/he0v-tf17](https://doi.org/10.13026/he0v-tf17), [참고 논문](https://doi.org/10.1038/s41597-025-04845-9). 공개 ODC-By1.0. 2025-06-24 버전은 출판 인용 추가이며 데이터 내용 변경은 없다는 설명이다.

공식 설명상 stress36, aerobic30, anaerobic31명이다. 97명의 독립 참가자가 아니다. E4의 EDA/BVP/ACC/TEMP 원자료를 사용하며 HR/IBI는 선택한다. 기록 날짜는 익명화를 위해 이동됐지만 같은 세션의 신호와 태그는 함께 이동했다. 상세 서식은 [공식 README](https://physionet.org/files/wearable-device-dataset/1.0.1/README.txt)에 있다.

## 그룹·라벨 게이트

[subject-info.csv](https://physionet.org/content/wearable-device-dataset/1.0.1/subject-info.csv)는 S01–S18/V1, f01–f18/V2의 36개 ID와 세션 참여를 연결한다. `_a/_b`는 같은 ID에 묶는다. V1/V2는 별도 모집 집단으로 설명된다. 프로토콜 버전과 성별 구성이 강하게 얽혀 있으므로 버전별 성능과 누락을 분리해 보고해야 한다. 나이 소개문과 f11 행의31세가 불일치한다.

`tags.csv`는 시각만 제공하므로 행번호→조건 맵을 버전별 공식 notebook과 확인해야 한다. S01 stress/aerobic 태그 파일은 웹에서 확인했지만 전체 기록과의 대응 및 notebook 코드는 확보하지 못했다. 따라서 태그의 의미를 임의 할당하지 않았고 segment manifest는 비어 있다. 전체 세션을 stress/exercise로 지정하는 shortcut은 금지한다.

60초의 엄격한 단일상태 창은 30초/45초 sprint보다 길다. 따라서 **anaerobic 최대부하 구간은 기본60초창으로 유효창이 없을 수 있다.** 이를 여러 sprint+cooldown을 하나의 anaerobic 연속 상태로 합쳐 해결하지 않는다. 원형 프로토콜을 확인한 뒤 30초 민감도 실험을 별도 사전등록하거나 해당 클래스 N/A를 명시해야 한다.

## 공식 제약과 사전 처리 정책

[constraints 직접 파일](https://physionet.org/content/wearable-device-dataset/1.0.1/data_constraints.txt)을 확인했다.

| 세션·ID | 원문 제약 요약 | 처리 원칙 |
|---|---|---|
| stress S02 | 네 원시 신호에 중복된 뒷부분 | 일차 common-set에서 전체 세션 제외. 복구는 원본 행위치·시간을 확인한 별도 분석 |
| stress f07 | 보호덮개로 BVP/TEMP 가림, EDA/ACC 유효 | 공정한 EDA/BVP/ACC 비교에서 해당 세션 공통 제외. EDA/ACC 별도 탐색은 표시 |
| stress f14 | a=baseline, b=나머지 | 동일 subject로 묶고, 파일 사이를 이어붙여 창을 만들지 않음 |
| aerobic S03/S07 | 각각90/95rpm에서 조기 종료 | 실제 확인된 구간만, 예상 태그 개수를 강제로 채우지 않음 |
| aerobic S11 | a는105rpm까지, b는110rpm | 동일 subject, 각 파일 경계 준수 |
| aerobic S12 | 세션 미실시 | 결측 세션 생성 금지 |
| anaerobic S01 | IBI 비어 있음 | IBI 미사용 모델에는 자동 전체 제외하지 않음 |
| anaerobic S06 | 마지막 sprint 없음 | 실제 태그만 |
| anaerobic S16 | a두 sprint, b두 sprint | 동일 subject, 분리 파일. 단순 미완주라는 명세 표현 정정 |

단위: EDA µS, TEMP°C, ACC 파일값은1/64g. BVP는 임상 절대진폭으로 해석하지 않는다. CSV 첫 행 UTC 문자열, 두번째 Hz를 읽고 실제 샘플 수로 시각을 구성한다.

## 실험이 허용되는 조건

필요한 모든 조건: 원본 확보·공식 SHA256 비교(가능한 파일)·버전별 태그 맵 확인·각 구간 범위 검증·same-ID group assertion·조건 전환창 제외·QC 및 공정 공통창. 하나라도 미충족이면 D2 지도학습 결과는 N/A로 둔다. 실험이 진행되어도 정신과제와 고정식 자전거를 구분한 결과이며 놀이·감각 과부하·자폐 멜트다운의 경고 유효성이 아니다.
