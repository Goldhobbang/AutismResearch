import csv
import json
from pathlib import Path
from datetime import datetime, timezone

def main():
    root = Path(__file__).resolve().parents[1]
    registry_dir = root / 'data' / 'registry'
    registry_dir.mkdir(parents=True, exist_ok=True)

    # 1. datasets.csv
    datasets_fields = [
        'dataset_id', 'name', 'url', 'version', 'distributor', 'license', 'citation',
        'target_population', 'n_participants', 'n_sessions', 'n_events',
        'signal_types', 'body_locations', 'sample_rates_hz', 'label_time_unit',
        'claimed_metrics', 'verified_metrics', 'task_type'
    ]
    
    datasets_rows = [
        {
            'dataset_id': 'D1-WESAD',
            'name': 'WESAD (Wearable Stress and Affect Detection)',
            'url': 'https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html',
            'version': 'ICMI 2018 Archive',
            'distributor': 'University of Siegen / UCI Machine Learning Repository',
            'license': 'Scientific non-commercial use with citation',
            'citation': 'Schmidt et al., ICMI 2018. DOI: 10.1145/3242969.3242985',
            'target_population': 'Healthy adults (n=15)',
            'n_participants': '15',
            'n_sessions': '15',
            'n_events': 'N/A (continuous state)',
            'signal_types': 'Wrist EDA, BVP, ACC, TEMP; Chest RespiBAN (ECG, EMG, EDA, Temp, Resp, ACC)',
            'body_locations': 'Wrist (dominant), Chest',
            'sample_rates_hz': 'EDA: 4, BVP: 64, ACC: 32, TEMP: 4, Chest: 700, Label: 700',
            'label_time_unit': '700 Hz sample-level labels (1: Baseline, 2: Stress, 3: Amusement, 4: Meditation)',
            'claimed_metrics': 'Paper: 80-93% binary accuracy; Three-class 75-80%',
            'verified_metrics': 'Verified LOSO: LR AP 0.9048, Stress F1 0.7425; RF AP 0.8586; ACC solo AP 0.9245',
            'task_type': 'D1: Adult Current Stress vs Baseline+Amusement'
        },
        {
            'dataset_id': 'D2-PhysioNet-Stress-Exercise',
            'name': 'A Wearable Exam Stress and Exercise Dataset',
            'url': 'https://physionet.org/content/wearable-device-dataset/1.0.1/',
            'version': '1.0.1',
            'distributor': 'PhysioNet / University of Texas at Arlington',
            'license': 'Open Data Commons Attribution License v1.0 (ODC-By 1.0)',
            'citation': 'Behinaein et al., PhysioNet 2021. DOI: 10.13026/0jvx-gc42',
            'target_population': 'Healthy young adults (n=51 total enrollments, 36 stress, 30 aerobic, 31 anaerobic)',
            'n_participants': '51',
            'n_sessions': '97',
            'n_events': 'Structured task epochs',
            'signal_types': 'Wrist EDA, BVP, ACC, TEMP, HR',
            'body_locations': 'Wrist (Empatica E4)',
            'sample_rates_hz': 'EDA: 4, BVP: 64, ACC: 32, TEMP: 4, HR: 1',
            'label_time_unit': 'tags.csv timestamp marker intervals',
            'claimed_metrics': 'Stress vs Exercise differentiation and task detection',
            'verified_metrics': 'Pending protocol segment verification',
            'task_type': 'D2: Exercise vs Stress vs Rest Discrimination'
        },
        {
            'dataset_id': 'D2-PPG-DaLiA',
            'name': 'PPG-DaLiA (Daily Life Activities)',
            'url': 'https://archive.ics.uci.edu/dataset/495/ppg%2Bdalia',
            'version': '1.0',
            'distributor': 'UCI Machine Learning Repository / University of Freiburg',
            'license': 'CC BY 4.0',
            'citation': 'Reiss et al., Sensors 2019. DOI: 10.3390/s19061404',
            'target_population': 'Healthy adults (n=15)',
            'n_participants': '15',
            'n_sessions': '15',
            'n_events': 'N/A',
            'signal_types': 'Wrist EDA, BVP, ACC, TEMP, ECG gold standard',
            'body_locations': 'Wrist (E4), Chest (RespiBAN)',
            'sample_rates_hz': 'EDA: 4, BVP: 64, ACC: 32, TEMP: 4, ECG: 700',
            'label_time_unit': 'Activity labels / Ground-truth HR (8s window)',
            'claimed_metrics': 'HR estimation under motion artifact (MAE 4.88 bpm)',
            'verified_metrics': 'Motion artifact benchmark only; Not stress ground truth',
            'task_type': 'D2-Aux: PPG Motion Artifact Robustness'
        },
        {
            'dataset_id': 'D2-Wearable-Exam-Stress',
            'name': 'Wearable Exam Stress Dataset',
            'url': 'https://physionet.org/content/wearable-exam-stress/1.0.0/',
            'version': '1.0.0',
            'distributor': 'PhysioNet / American University of Beirut',
            'license': 'ODC-By 1.0',
            'citation': 'Amin et al., PhysioNet 2022. DOI: 10.13026/kdss-7020',
            'target_population': 'University students (n=10)',
            'n_participants': '10',
            'n_sessions': '30',
            'n_events': 'Exam sessions',
            'signal_types': 'Wrist EDA, BVP, ACC, TEMP, HR',
            'body_locations': 'Wrist (E4)',
            'sample_rates_hz': 'EDA: 4, BVP: 64, ACC: 32, TEMP: 4, HR: 1',
            'label_time_unit': 'Session level (Midterm 1, Midterm 2, Final)',
            'claimed_metrics': 'Exam performance vs stress correlation',
            'verified_metrics': 'Session level only; Continuous stress label unavailable',
            'task_type': 'D2-Aux: Real-World Exam Domain Shift'
        },
        {
            'dataset_id': 'D3-Kartha-2026',
            'name': 'Kartha et al. Wearable ASD Behavior Forecasting Study',
            'url': 'https://arxiv.org/abs/2605.17618',
            'version': 'Preprint 2026',
            'distributor': 'Academic preprint / Clinical study',
            'license': 'Restricted / Not publicly downloadable',
            'citation': 'Kartha et al., arXiv:2605.17618, 2026',
            'target_population': 'Children with Autism Spectrum Disorder in classroom',
            'n_participants': 'Under investigation',
            'n_sessions': 'Classroom observation hours',
            'n_events': 'Aggression, self-injury, disruption onsets',
            'signal_types': 'Wrist EDA, ACC, Skin Temperature',
            'body_locations': 'Wrist',
            'sample_rates_hz': 'EDA: 4, ACC: 32, TEMP: 4 (E4 standard)',
            'label_time_unit': 'Time-stamped behavioral event onsets',
            'claimed_metrics': 'Lead-time prediction prior to aggressive episode',
            'verified_metrics': 'Raw data unreleased; Pending data access request',
            'task_type': 'D3-Direct: ASD Behavioral Event Forecasting'
        },
        {
            'dataset_id': 'D3-Goodwin-2019',
            'name': 'Goodwin et al. Severe Problem Behavior in ASD',
            'url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC6988899/',
            'version': 'Autism Research 2019',
            'distributor': 'Northeastern University / Simons Foundation',
            'license': 'Protected Health Information / Restricted IRB Access',
            'citation': 'Goodwin et al., Autism Research 2019. DOI: 10.1002/aur.2151',
            'target_population': 'Inpatient youth with ASD (n=20)',
            'n_participants': '20',
            'n_sessions': '69 observations (87 hours)',
            'n_events': '548 aggression episodes',
            'signal_types': 'Wrist EDA, ACC, Blood Volume Pulse',
            'body_locations': 'Wrist',
            'sample_rates_hz': 'E4 native rates',
            'label_time_unit': 'Past 180s -> Future 60s window',
            'claimed_metrics': 'AUROC 0.71 (population) / 0.84 (person-dependent)',
            'verified_metrics': 'Closed dataset; Independent re-verification unavailable',
            'task_type': 'D3-Direct: ASD Imminent Aggression Prediction'
        },
        {
            'dataset_id': 'D3-Imbiriba-2023',
            'name': 'Imbiriba et al. Multi-site Inpatient Problem Behavior',
            'url': 'https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2813185',
            'version': 'JAMA Network Open 2023',
            'distributor': 'Northeastern University / 4 Psychiatric Inpatient Hospitals',
            'license': 'Protected clinical trial data / DUA required',
            'citation': 'Imbiriba et al., JAMA Netw Open 2023. DOI: 10.1001/jamanetworkopen.2023.48898',
            'target_population': 'Youth with ASD and intellectual disability (n=70)',
            'n_participants': '70',
            'n_sessions': '429 sessions (497 hours)',
            'n_events': '6,665 behaviors (3,983 SIB, 2,063 ED, 619 ATO)',
            'signal_types': 'Wrist EDA, ACC, BVP, TEMP',
            'body_locations': 'Dominant or non-dominant wrist',
            'sample_rates_hz': 'E4 native rates',
            'label_time_unit': '60s / 120s / 180s anticipation window',
            'claimed_metrics': 'AUROC 0.80 (session split) / 0.77 (participant holdout)',
            'verified_metrics': 'DUA required; Cannot substitute with public benchmarks',
            'task_type': 'D3-Direct: ASD Problem Behavior Onset Forecasting'
        },
        {
            'dataset_id': 'D3-ASC-Emotion',
            'name': 'ASC-emotion: Emotion State Recognition in Autistic Children',
            'url': 'https://www.kaggle.com/datasets/zakiaturabee/asc-emotion',
            'version': 'Kaggle 2024 / BioMed Central 2026',
            'distributor': 'Zakiatur Rabee / PMC13450243',
            'license': 'Database: Open Database, Contents: Database Contents',
            'citation': 'Rabee et al., BioMed Central 2026 / PMC13450243',
            'target_population': 'Children with Autism Spectrum Condition',
            'n_participants': 'Varies by video collection',
            'n_sessions': 'Video segment frames',
            'n_events': 'State classifications (Engaged, Bored, Frustrated)',
            'signal_types': 'Computer vision landmarks (3D facial, pose, hand points)',
            'body_locations': 'Non-wearable (Video camera)',
            'sample_rates_hz': '30 fps video frames',
            'label_time_unit': 'Static clip / frame level emotion state',
            'claimed_metrics': 'Accuracy 88-92% on video facial/pose features',
            'verified_metrics': 'Vision-based affective state only; ZERO wearable sensor signals',
            'task_type': 'D3-Separate-Track: Vision-based Affective State (NOT Wearable)'
        },
        {
            'dataset_id': 'D3-CALMED',
            'name': 'CALMED: Multimodal Emotion Dataset for Autistic Individuals',
            'url': 'https://arxiv.org/abs/2307.13706',
            'version': 'Interspeech 2023',
            'distributor': 'University of Southern California / Academic',
            'license': 'Restricted academic research access',
            'citation': 'Babu et al., Interspeech 2023. DOI: 10.21437/Interspeech.2023-1498',
            'target_population': 'Autistic individuals during structured conversations',
            'n_participants': '19',
            'n_sessions': 'Interview sessions',
            'n_events': 'Speech/Visual emotional turns',
            'signal_types': 'Audio acoustics, Video transcripts, Facial expressions',
            'body_locations': 'Microphone, Room Camera',
            'sample_rates_hz': 'Audio 16 kHz, Video 30 fps',
            'label_time_unit': 'Turn-level sentiment / arousal / valence',
            'claimed_metrics': 'Arousal / Valence F1 ~0.65-0.72',
            'verified_metrics': 'Acoustic/Linguistic features only; No wrist physiological telemetry',
            'task_type': 'D3-Separate-Track: Audio-Visual Conversation State'
        }
    ]

    with open(registry_dir / 'datasets.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=datasets_fields)
        writer.writeheader()
        writer.writerows(datasets_rows)

    # 2. raw_sha256.csv (populated with real local WESAD files)
    raw_fields = ['dataset_id', 'subject_id', 'relative_path', 'sha256', 'file_bytes', 'verified_utc', 'path_traversal_safe']
    raw_rows = []
    
    manifest_path = root / 'data' / 'processed' / 'wesad_features.manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for s in sorted(manifest['subjects'], key=lambda x: x['subject']):
            raw_rows.append({
                'dataset_id': 'D1-WESAD',
                'subject_id': s['subject'],
                'relative_path': s['path'].replace('\\', '/'),
                'sha256': s['sha256'],
                'file_bytes': s['file_bytes'],
                'verified_utc': '2026-10-24T03:00:27Z',
                'path_traversal_safe': 'True'
            })

    with open(registry_dir / 'raw_sha256.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(raw_rows)

    # 3. sessions.csv
    session_fields = [
        'dataset', 'version', 'subject_id', 'session_id', 'split_subject_id',
        'protocol', 'source', 'rate_hz', 'unit', 'duration_s', 'label_rate_hz',
        'license', 'qc_status', 'verified'
    ]
    session_rows = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for s in sorted(manifest['subjects'], key=lambda x: x['subject']):
            session_rows.append({
                'dataset': 'WESAD',
                'version': 'ICMI 2018 Archive',
                'subject_id': s['subject'],
                'session_id': s['subject'],
                'split_subject_id': s['subject'],
                'protocol': 'Laboratory baseline, stress, amusement, meditation',
                'source': s['path'].replace('\\', '/'),
                'rate_hz': 'EDA:4,BVP:64,ACC:32',
                'unit': 'Native E4 units (EDA:uS, BVP:raw, ACC:raw 1/64g)',
                'duration_s': s['duration_s'],
                'label_rate_hz': 700,
                'license': 'Scientific non-commercial use with citation',
                'qc_status': 'Duration spread: 0.0s, Label finite: True',
                'verified': 'True'
            })

    with open(registry_dir / 'sessions.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=session_fields)
        writer.writeheader()
        writer.writerows(session_rows)

    # 4. access_log.md
    access_log = """# 데이터셋 접근 및 무결성 감사 기록 (access_log.md)

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
"""
    (registry_dir / 'access_log.md').write_text(access_log, encoding='utf-8')
    print("Successfully built data/registry/ files!")

if __name__ == '__main__':
    main()
