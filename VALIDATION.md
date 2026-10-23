# 최신 실행 검증 기록

검증일: 2026-10-23. Linux 클라우드, Python3.12.14. 라이브러리·환경 기록은 results/logs/environment.json에 있습니다. 사용자 Windows 노트북의 측정이 아닙니다.

- 새 src/ 단위검사19개: OK. native-rate 정렬, 끝점/혼합창, 미래 및 창 이전 샘플 비사용, 품질 cutoff, 그룹 분리, 전처리 fit 범위, 센서 공통창, undefined 지표와 AUROC 해석을 확인했습니다.
- 새 전체 실행: 합성6명×9설정=54 fit,7종LR센서비교, 결과CSV/JSON/그림, 분리assertion,원기록재생 PASS. 원자료 재계산 점수와 저장된 OOF 점수 일치를 검사했습니다.
- 합성 결과는 results/synthetic_test/ 안에만 생성됩니다. 실제 데이터 결과와 혼합하지 않으며 점수 자체는 발표용 성능이 아닙니다. 배포에는 검사 로그만 포함합니다.
- 공식 다운로드 스크립트를 실제 실행했지만 WESAD·PhysioNet 모두 proxy403으로 실패했습니다. 원본 해시·실제 학습·실제 성능은 N/A입니다.
- 설계SVG를PNG로 렌더해 확인했고11장PPTX를PDF로 변환해 페이지 수·텍스트 경계·미리보기를 확인했습니다. 제품을 제작한 검사는 아닙니다.
- ESP32-S3 BLE-only 에너지 스크립트는 가정값의 계산을 재현했습니다.150mAh×80%와 기본10.8272mA에서 약11.08h이며 실제 배터리 수명은 아닙니다.

미검증: 실제WESAD15명인수·LOSO,PhysioNet태그복원·운동혼동,Windows설치/CUDA,ESP32-S3의C모델export·지연·전력·BLE·착용·아동예측·임상효과. P0/P1의 실제 연구 평가가 완료됐다는 주장은 하지 않습니다.

로그: results/logs/unit_tests.log, synthetic_test_pipeline.log, download_attempts.jsonl.
