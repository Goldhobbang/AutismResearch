# 자폐(ASD) 관련 데이터셋 실사 및 접근성 감사 보고서 (ASD_DATASET_ACCESS_AUDIT.md)

- 작성 시각: 2026-10-24 04:12 (UTC+9)
- 게이트 판정: **G6 PASS (후보 7건 전수 실사 완료) / D3 진입 판정: BLOCKED_NOT_EVALUATED (N/A 고정)**
- 산출물 경로: `data/registry/asd_candidates.csv`

---

## 1. 7대 후보 데이터셋 종합 실사 현황표

| 후보 ID | 연구명 / 출처 | 대상 및 규모 | 수집 신호 및 위치 | 정답 라벨 | 사건 Onset 시각 유무 | 공개 다운로드 | 접근성 판정 |
|---|---|---|---|---|---|---|---|
| **ASD-01** | Kartha et al. (arXiv:2605.17618, 2026) | 학령기 자폐 아동 (교실) | 손목 EDA, ACC, 체온 | 공격·파괴적 행동 | 있음 (초 단위) | 불가 (비공개) | **D3-ACCESS PENDING** |
| **ASD-02** | Goodwin et al. (Autism Research 2019) | 자폐 입원 청소년 20명, 87시간 | 손목 E4 (EDA, BVP, ACC) | 타인 공격 548건 | 있음 (비디오 검증) | 불가 (IRB 제한) | **D3-ACCESS PENDING** |
| **ASD-03** | Imbiriba et al. (JAMA Netw Open 2023) | 자폐 입원 청소년 70명, 497시간 | 손목 E4 (EDA, BVP, ACC, TEMP) | 자해(SIB), 정서조절곤란(ED), 공격(ATO) 6,665건 | 있음 (정밀 동기화) | 불가 (DUA 필수) | **D3-ACCESS PENDING** |
| **ASD-04** | ASC-emotion (PMC13450243, 2026) | 자폐 아동 비디오 영상 | 비디오 3D 얼굴/자세/손 랜드마크 (카메라) | 몰입, 지루함, 좌절 (Engaged, Bored, Frustrated) | **없음 (클립 범주형)** | **가능 (Kaggle)** | **D3-BEHAVIOR-STATE ONLY (비웨어러블)** |
| **ASD-05** | CALMED (Interspeech 2023) | 자폐 아동 19명 대화 | 마이크 음성, 대화 전사, 얼굴 표정 | Arousal, Valence, 대화 참여도 | 대화 턴(Turn) 단위 | 불가 (승인제) | **D3-SEPARATE-TRACK (음성/영상)** |
| **ASD-06** | PreMAC (Sensors 2021) | 자폐 아동 7명 (실험실) | 손목 E4 + WINGS + 키넥트 | 전조 행동 (발성, 모터 동요) | 있음 (관찰자 태그) | 불가 (저자 요청제) | **D3-ACCESS PENDING** |
| **ASD-07** | REACT (J Med Internet Res 2024) | 발달장애 아동 3명 (학교 2회 방문) | 손목 E4 + WINGS + 실내 소음 | 동요(Agitation) 100건 | 있음 | 불가 (공동연구 전용) | **D3-ACCESS PENDING** |

---

## 2. 모달리티 및 정답 불일치 분석 (엄격 분리)

1. **ASC-emotion의 본질적 한계 (웨어러블 대체 불가)**:
   - Kaggle에 공개된 ASC-emotion 데이터는 컴퓨터 비전 알고리즘으로 추출한 **3D 얼굴 좌표 및 골격 랜드마크(Numeric Coordinates)**임.
   - 손목 피부전도도(EDA), 광용적맥파(PPG), 가속도(ACC) 신호가 **0건**이며, 손목형 웨어러블 디바이스 연구에 입력으로 사용할 수 없음.
   - 라벨 또한 'Frustrated(좌절)'라는 정서적 상태 분류일 뿐, 자폐 멜트다운의 물리적 발생 시각(Onset timestamp)이 존재하지 않음.
   - **원칙 준수**: ASC-emotion의 88~92% 정확도를 웨어러블 멜트다운 60초 예측 성능으로 둔갑시키는 행위를 일체 배제함.
2. **Goodwin 2019 / Imbiriba 2023의 임상적 차이**:
   - Goodwin의 라벨은 '타인 신체 공격(Aggression to Others)'이며, Imbiriba의 라벨은 자해(SIB), 정서조절곤란(ED), 공격(ATO)의 복합 행동임.
   - Imbiriba의 ED 범주에 멜트다운이 일부 포함될 수 있으나, 전체 입원 환자 복합 이상행동 예측 모델의 성능(AUROC 0.80)을 "멜트다운 단독 60초 선행 예측 80%"로 해석할 수 없음.
   - 두 데이터셋 모두 보호된 건강 정보(PHI)로서 정식 데이터이용계약(DUA) 및 기관생명윤리위원회(IRB) 승인 절차 없이 무단 취득 불가.
3. **Kartha 2026 (교실 환경 웨어러블)**:
   - 교실 환경에서 E4 웨어러블 신호와 실제 행동 사건 Onset을 다룬 가장 최신의 연구이나, 원자료가 공공 저장소에 배포되지 않음 (Data available upon reasonable request).

---

## 3. 최종 게이트 판정 및 실행 방침

- **판정 요약**:
  - `D3-WEARABLE-EVENT READY`: **0건 (만족하는 공개 자료 없음)**.
  - `D3-ACCESS PENDING`: 5건 (정식 연구 협약 또는 DUA 신청 필요).
  - `D3-BEHAVIOR-STATE ONLY`: 1건 (ASC-emotion, 영상 랜드마크 상태분류 전용).
- **결정 규칙 이행**:
  - 연구 원칙 제2조(데이터 없으면 숫자 없다) 및 제7조(멜트다운과 복합 행동 혼동 금지)에 의거하여, **D3(자폐 멜트다운 60초 조기예측)의 실데이터 학습 및 성능 수치는 `BLOCKED_NOT_EVALUATED` (N/A)로 유지**함.
  - 외부 공인 승인 자료가 확보되기 전까지 가상의 수치나 합성 데이터를 학습 증거로 생성하지 않음.
