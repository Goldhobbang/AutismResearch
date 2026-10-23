# 데이터셋 접근 및 무결성 감사 기록 (access_log.md)

기준 시각: 2026-10-24

## 1. D1: WESAD (Wearable Stress and Affect Detection)
- **출처**: University of Siegen 공식 배포 (`https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html`)
- **로컬 경로**: `data/raw/WESAD/S{2..11,13..17}/S{2..11,13..17}.pkl`
- **검증 상태**: 15명 참가자 원본 pickle 전원 확보. SHA256 체크섬 확인 완료. 센서 신호(EDA, BVP, ACC)와 라벨 간 동기화 편차 0.0s 검증.
- **제약**: 성인 15명 실험실 환경 스트레스 분류용. 가슴 신호(RespiBAN)는 웨어러블 제품 범위 밖이므로 제외하고 손목 신호만 사용.

## 2. D2: PhysioNet Wearable Exam Stress and Exercise v1.0.1
- **출처**: PhysioNet (`https://physionet.org/content/wearable-device-dataset/1.0.1/`)
- **라이선스**: ODC-By 1.0 (공개)
- **접근 방식**: 공식 HTTP curl / python urllib 다운로드 시도.
- **감사 사항**:
  - `data_constraints.txt`, `subject-info.csv`, `tags.csv` 확보 및 검증 필요.
  - 시간 표기: 1행 UTC 타임스탬프 문자열, 2행 샘플레이트.
  - 운동 태스크: 30초·45초 sprint 구간은 60초 창 단일 상태 조건 불가 (길이 임의 연장 금지).
  - 참가자 식별: f14_a, f14_b 및 동일인 중복 태스크(STRESS, AEROBIC, ANAEROBIC) 참가자 분할 격리 필수.

## 3. D3: ASD 관련 데이터 후보 실사
- **Kartha et al. (arXiv:2605.17618, 2026)**:
  - 연구 내용: 실제 자폐 아동 교실 내 웨어러블(EDA/ACC/체온) 센서 기반 위험 행동 선행 예측.
  - 접근 현황: 원본 데이터셋 공식 다운로드 링크 미공개 (Restricted / Data on Request).
  - 판정: `D3-ACCESS PENDING`. 승인 없이 합성 생성 금지.
- **Goodwin 2019 / Imbiriba 2023**:
  - 연구 내용: 입원 환자 대상 공격/자해/정서조절곤란(ED) 예측.
  - 접근 현황: 환자 보호 및 IRB 제한으로 기관 간 데이터이용계약(DUA) 필요.
  - 판정: `D3-ACCESS PENDING`.
- **ASC-emotion (Kaggle / PMC13450243)**:
  - 데이터 형태: 비디오 기반 3D 얼굴/자세 랜드마크 (Engaged/Bored/Frustrated).
  - 판정: `D3-BEHAVIOR-STATE ONLY` (손목 웨어러블 센서 아님. 60초 조기경보 검증에 사용 금지).
- **CALMED (Interspeech 2023)**:
  - 데이터 형태: 음성 대화 오디오/스크립트 감정 상태.
  - 판정: `D3-SEPARATE-TRACK` (웨어러블 생체신호 없음).
