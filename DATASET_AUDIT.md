# 데이터 인수 감사 — 2026-10-23

## 판정

**G0 출처/이용 조건: 문서 확인. G1 원본 무결성: BLOCKED. 실제 원자료 다운로드·해시·센서 길이 검증·학습은 완료되지 않았다.**

클라우드 실행 환경의 정책은 `restricted`, `enforced`, `observations_current=true`였다. 공식 데이터 호스트가 허용 목록에 없어 Python HTTPS 요청은 `Tunnel connection failed: 403 Forbidden`으로 종료했다. sandbox 승격 요청 후에도 동일했다. 이는 연구기관이 데이터를 비공개로 전환했다는 증거가 아니다. 공식 웹페이지는 연결된 웹 도구로 열람할 수 있었지만, 그 사실을 로컬 ZIP 확보로 간주하지 않았다. proxy를 우회하거나 제3자 원자료를 대체 생성하지 않았다.

| 자료 | 원출처·버전 | 문서에서 확인 | 로컬 원본 상태 |
|---|---|---|---|
| WESAD | [연구기관](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html), ICMI2018 배포 | 15명, ZIP 약2.5GB, 과학적 비상업적 이용·출처표기 | ZIP/README/pkl 미확보, SHA256 N/A, 분석 참가자 0 |
| Stress/Exercise | [PhysioNet](https://physionet.org/content/wearable-device-dataset/1.0.1/), v1.0.1 | 공개 ODC-By1.0, ZIP69.7MB/비압축247.4MB; 공식 README·constraints·subject-info 웹열람 | ZIP/센서 CSV 미확보, SHA256 N/A, 검증된 supervised segment 0 |

실패한 주소·방법은 `reports/evidence/download_blocker.json`에 있다. `scripts/download_datasets.py`는 공식 페이지의 링크를 찾아 다운로드하고 byte count, SHA256, ZIP 검사 및 안전한 압축해제를 수행한다. 이 스크립트의 다운로드 성공은 아직 검증하지 못했다.

## 명세서와 원출처 사이에서 발견한 차이

1. PhysioNet의 ANAEROBIC S16은 단순 미완주가 아니라 연결 손실로 두 파일에 분리됐다. S06은 마지막 sprint가 없다. STRESS f07은 EDA/ACC만 유효하고 BVP/TEMP가 가려졌다. [공식 constraints](https://physionet.org/content/wearable-device-dataset/1.0.1/data_constraints.txt)
2. 소개 페이지는 18–30세라고 설명하지만 `subject-info.csv`의 f11은 31세다. 그대로 기록하고 일치한다고 가정하지 않는다. [공식 참가자표](https://physionet.org/content/wearable-device-dataset/1.0.1/subject-info.csv)
3. 같은 참가자표에서 V1 S01–S18은 남성, V2는 대부분 여성이다. 프로토콜 버전·환경·성별이 얽혀 있어 버전 차이를 생리 상태 차이로 오해할 위험이 있다. 성별은 모델 특징으로 넣지 않는다.
4. PhysioNet의 첫 행은 이 배포본에서 UTC **문자열**이다. 원래 Empatica의 Unix 숫자 형식을 그대로 가정하지 않는다. [공식 README](https://physionet.org/files/wearable-device-dataset/1.0.1/README.txt)

## 파일 계약과 해석

- `data/processed/subject_manifest.csv`: WESAD의 명세상 예상 ID와 PhysioNet 공식표의 ID 목록이다. **로컬 관찰 참가자 목록이 아니다.** `raw_files_verified=false`와 `source_status`를 보존한다.
- `data/processed/segment_manifest.csv`: 검증된 행이 없는 header-only 파일이다. 공식 notebook과 전체 tags/raw의 대응을 확인하기 전 시작·끝 시각이나 label을 채우지 않는다.
- `data/raw/`는 원자료 확보 후 보존하는 폴더다. 배포 ZIP/커밋에서 제외한다. 다운로드 성공 이후 로컬 원본 해시를 별도로 기록한다. 자체 계산 해시는 공급자 진위 인증과 다르다.

## 재개 절차

환경 설정에서 공식 호스트를 허용하거나, 사용자의 Windows에서 아래를 실행한다. 허용 대상은 `ubi29.informatik.uni-siegen.de`, `uni-siegen.sciebo.de`, `physionet.org`이며 UCI 메타데이터에는 `archive.ics.uci.edu`가 필요하다. 공식 S3 대안을 이용할 때만 `physionet-open.s3.amazonaws.com`이 추가로 필요하다.

```powershell
py scripts/download_datasets.py --dataset wesad --extract
py scripts/download_datasets.py --dataset physionet --extract
```

다음으로 WESAD README, pkl 경로·15개 ID·주파수·라벨700Hz 동기화·센서 길이를 실제 파일에서 검증한다. D2는 전체 태그와 notebook의 구간 맵을 검증하고, constraint 적용 및 모든 세션의 subject group을 확인한 뒤 segment manifest를 승인한다. H1/H2/H3의 성능은 그 이후에만 계산한다.
