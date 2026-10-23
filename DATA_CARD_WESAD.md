# WESAD 데이터 카드

상태: **공식 메타데이터 확인 / 로컬 원본 미확보 / 실제 학습 미실행**.

- 출처: [연구기관](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html), [UCI](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection).
- 인용: Schmidt et al., *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection*, ICMI2018, DOI [10.1145/3242969.3242985](https://doi.org/10.1145/3242969.3242985). 데이터 UCI DOI [10.24432/C57K5T](https://doi.org/10.24432/C57K5T).
- 공개 설명: 성인 실험실 스트레스·중립·amusement 15명. 원제공처 ZIP2.5GB. 원제공처는 과학적 비상업적 이용과 출처 표기를 요구하고, UCI는 원출처 라이선스로 연결한다.
- 손목 E4: BVP64Hz, EDA4Hz, ACC32Hz, TEMP4Hz. 가슴700Hz 신호는 이번 모델에서 제외한다.
- 예상 ID: S2–S11, S13–S17. 이는 사용자 명세에 따라 설정한 인수검사 기대값이며 실제 파일검사 통과로 표시하지 않는다.
- primary label: 2→stress1, 1/3→nonstress0. 3은 amusement이며 운동 label이 아니다. 1/2/3 이외 및 혼합창은 primary 평가 제외. 원본 README를 확보해 시간 정렬·단위·라벨 규칙을 다시 확인해야 한다.
- 계획: strict60초 창, hop15초, 과거만, LOSO15명, 전체 센서 공통 유효창에서 7개 조합 비교. 모델 입력에 label·phase·ID·시각은 넣지 않는다.
- 한계: 성인 실험실 프로토콜 분류. 멜트다운 onset·실제 경고 선행 시간·ASD 사건 단위 정밀도·일일 오경보의 근거가 아니다. 피험자15명이 독립 단위다.
- 실제 로컬 상태: archive bytes N/A; SHA256 N/A; 원본 참가자검증0; 학습창0. 다운로드 실패 사유는 `DATASET_AUDIT.md`와 `reports/evidence/download_blocker.json` 참조.

**필수 인수검사:** 신뢰된 공식 ZIP을 확보하고 README→파일목록→SHA256→pkl구조→sensor length/dtype/unit→label rate→duration mismatch→NaN/inf→유효창 수 순으로 확인한다. pickle은 신뢰된 원본만 읽는다. UCI의 일반 `fetch_ucirepo` 코드가 외부2.5GB 원자료를 제공한다고 가정하지 않는다.
