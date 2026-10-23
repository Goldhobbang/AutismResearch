# 데이터셋 종합 레지스트리 보고서 (DATASET_REGISTRY.md)

- 작성 시각: 2026-10-24 04:16 (UTC+9)
- 버전: 1.0
- 데이터 메타데이터 위치: `data/registry/`

---

## 1. 프로젝트 공식 데이터셋 인벤토리

| ID | 데이터셋 명칭 | 배포 기관 | 버전 | 라이선스 | 주 연구 목적 | 상태 |
|---|---|---|---|---|---|:---:|
| **D1-WESAD** | Wearable Stress and Affect Detection | University of Siegen / UCI | ICMI 2018 | 학술 비상업적 이용 | 성인 실험실 스트레스 분류 (15명) | **로컬 원본 확보 및 15-fold LOSO 완료** |
| **D2-PhysioNet** | Wearable Exam Stress & Exercise | PhysioNet / UT Arlington | 1.0.1 | ODC-By 1.0 | 스트레스 vs 유산소/무산소 운동 구별 | **메타데이터 전수 감사 및 세그먼트 복원 완료** |
| **D2-PPG-DaLiA** | Daily Life Activities PPG Benchmark | UCI / Univ. of Freiburg | 1.0 | CC BY 4.0 | PPG 움직임 노이즈 강건성 (보조) | 공식 이용조건 확인 완료 (스트레스 정답 아님) |
| **D2-Exam-Stress** | Wearable Exam Stress Dataset | PhysioNet / AUB | 1.0.0 | ODC-By 1.0 | 실제 시험 환경 도메인 전이 (보조) | 시험 세션 단위 라벨 (연속 스트레스 아님) |
| **D3-Kartha-2026** | Wearable Classroom ASD Behavior | arXiv:2605.17618 | Preprint | 비공개 (승인제) | 자폐 아동 교실 행동 Onset 조기예측 | 원문 데이터 접근 신청 대상 (공개 다운로드 없음) |
| **D3-Goodwin-2019** | Severe Problem Behavior in ASD | Simons Foundation / Northeastern | 2019 | IRB 제한 (DUA) | 자폐 입원 청소년 공격 행동 예측 | 비공개 건강정보 (DUA 필수, 멜트다운 단독 아님) |
| **D3-Imbiriba-2023** | Multi-site Inpatient Problem Behavior | JAMA Netw Open / 4개 병원 | 2023 | 기관 승인제 (DUA) | SIB, ED, ATO 복합 행동 Onset 예측 | 비공개 건강정보 (DUA 필수) |
| **D3-ASC-Emotion** | ASC-emotion 3D Vision Landmarks | Kaggle / PMC13450243 | 2024/2026 | Open Database | 비디오 기반 3D 얼굴/자세 랜드마크 감정 | **별도 트랙 (손목 웨어러블 신호 ZERO, 대체 금지)** |
| **D3-CALMED** | Multimodal Emotion Dataset | Interspeech / USC | 2023 | 학술 연구 승인제 | 대화 음성 음향/전사 감정 상태 | **별도 트랙 (웨어러블 생체신호 없음)** |

---

## 2. 세션 및 참가자 무결성 요약

1. **WESAD (D1)**:
   - 15명 참가자 (S2–S11, S13–S17, S12는 원배포 당시 센서 결함으로 공식 제외).
   - 원본 pickle SHA256 체크섬 `data/registry/raw_sha256.csv` 영구 보존.
   - 60초 창 5,740개, 적격 유효창 2,031개(스트레스 607, 비스트레스 1,424).
2. **PhysioNet (D2)**:
   - 51명 등록 세션 (STRESS 36, AEROBIC 30, ANAEROBIC 31).
   - 성별-버전 교란: V1(남성 18명), V2(여성 33명).
   - `data_constraints.txt` 센서 중복(S02), 캡 미제거(f07), 연결 단절(f14, S11, S16) 반영 완료.
   - 복원 세그먼트: 179개 중 단일 과제 적격 99개 확인.
3. **ASD 사건 데이터 (D3)**:
   - 승인된 웨어러블+사건 Onset 데이터가 없으므로 D3 조기경보 학습은 차단 상태를 명시하고, 비웨어러블(ASC-emotion) 데이터로 위조하지 않음.
