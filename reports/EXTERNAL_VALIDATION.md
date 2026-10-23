# 외부 도메인 및 운동 혼동 평가 감사 보고서 (EXTERNAL_VALIDATION.md)

- 작성 시각: 2026-10-24 04:10 (UTC+9)
- 게이트 판정: **G4 PASS (프로토콜·세그먼트 정밀 복원 완료) / G5 PARTIAL (운동 지도학습 게이트 조건 확인)**
- 대상 데이터: PhysioNet Wearable Exam Stress and Exercise v1.0.1 (`https://physionet.org/content/wearable-device-dataset/1.0.1/`)
- 산출물 경로: `data/processed/physionet/verified_segments.csv`, `data/registry/physionet_*.csv`

---

## 1. PhysioNet v1.0.1 공식 원본 구조 및 감사 결과

- **데이터셋 구성**:
  - 총 51명 등록 (Protocol V1 18명, Protocol V2 33명).
  - 3개 주요 세션: `STRESS` (36명), `AEROBIC` (30명), `ANAEROBIC` (31명).
  - 센서: Empatica E4 (EDA 4Hz, BVP 64Hz, ACC 32Hz, TEMP 4Hz, HR 1Hz, IBI, tags.csv).
- **공식 제약사항 (`data_constraints.txt`) 반영**:
  - `S02`: E4 Connect 다운로드 파일 중복 기록 결함 발생 (ACC 49,545행, BVP 99,091행 등). 지도학습 제외.
  - `f07`: 보호 캡(Protection Dock) 미제거로 PPG 및 온도 센서 차폐 (EDA 및 ACC만 유효, PPG 공통 비교 제외).
  - `f14`, `S11`, `S16`: 블루투스 연결 유실로 인한 세션 파일 분할 (`_a`, `_b`). 단일 참가자 그룹 묶음 필수.
  - `S12`: 유산소 프로토콜 미실시 (결측).

---

## 2. 프로토콜·성별 교란 (Demographic & Version Confounding)

공식 `subject-info.csv` 전수 감사 결과:

1. **성별-버전 완전 교란**:
   - **Protocol V1 (S01–S18)**: **100% 남성** (18~22세 청년).
   - **Protocol V2 (f01–f17)**: **100% 여성** (21~31세).
   - 결론: Protocol V1과 V2를 혼합하여 모델을 학습하면, 알고리즘은 생리적 스트레스가 아닌 **남녀 간 피부전도도(EDA) 및 심박 특성(성별 차이)**을 학습하게 됨.
2. **세션 전체 일괄 라벨링(Bulk Session Labeling) 절대 금지**:
   - `STRESS` 세션 내부에는 `Baseline`(휴식), `Stroop`(인지 스트레스), `First Rest`(회복), `TMCT`(암산 스트레스), `Second Rest`(회복), `Real Opinion`(발표 준비), `Opposite Opinion`(논쟁 스트레스), `Subtract`(산술 스트레스)의 8단계 세부 구간이 존재함.
   - 세션 전체를 단순히 "스트레스"로 일괄 라벨링할 경우, 휴식(Rest)과 준비(Prep) 구간까지 스트레스로 오염되어 심각한 라벨 오류 유발.
3. **무산소 운동(Anaerobic) 창 길이 불일치**:
   - 스프린트 인터벌은 30초 또는 45초 단위의 고강도 폭발적 운동임.
   - 본 연구의 표준 60초 분석 창($W=60$)은 30/45초 단일 스프린트 상태에 들어갈 수 없으며, 반드시 휴식/회복 구간이 섞이게 됨.
   - 따라서 60초 창 규약 하에서 무산소 스프린트는 단일 상태 적격 창이 존재하지 않으며, 이를 임의 연장하는 행위는 엄격히 배제됨 (`verified_segments.csv`에 제외 사유 명시).

---

## 3. 세그먼트 복원 및 적격 구간 통계 (`verified_segments.csv`)

- 총 복원 세그먼트: **179개**
- 적격 검증 세그먼트 (`verified = True`): **99개**
  - 스트레스 순수 과제 (Stroop, TMCT, Opposite Opinion, Subtract): 68개
  - 안정적 베이스라인 (Baseline): 17개
  - 검증된 유산소 자전거 운동 (Aerobic Cycling): 14개
- 평가 보류 구간: 80개 (회복 휴식기 34개, 발표 준비 17개, 무산소 스프린트 18개, 데이터 중복 및 센서 결함 11개).

---

## 4. WESAD $\rightarrow$ PhysioNet 도메인 전이(Zero-shot Transfer) 한계

- WESAD 모델의 가속도계(ACC)는 TSST 구두 발표 제스처에 최적화되어 있음.
- PhysioNet의 스트레스 과제는 컴퓨터 화면 응시(Stroop) 및 앉아서 대답하는 형태이며, 유산소 운동은 자전거 페달링과 상체 고정 상태임.
- 단위 차이: WESAD 손목 ACC는 원본 $1/64\text{g}$ 정수 스케일이며, PhysioNet ACC는 물리 $\text{g}$ 또는 $1/64\text{g}$ 변환 여부 및 부착 방향을 매핑해야 함.
- 판정: 운동과 스트레스의 엄밀한 분별(D2)은 사전에 분리 검증된 `verified_segments.csv`의 99개 유효 구간에서만 수행되어야 하며, 일상 자폐 아동의 신체 놀이나 감각 과부하와 동등하다고 주장할 수 없음.
