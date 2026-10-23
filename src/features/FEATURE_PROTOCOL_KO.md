# 특징·시간·품질 프로토콜 (결과 확인 전 고정)

이 모듈은 성인의 실험실 스트레스 상태 분류를 위한 기준 구현이다. 기존 임상 논문 코드의 재현본이나 멜트다운 예측 모델이 아니다.

## 시간과 레이블

- 기본 window=60초, hop=15초. 결정 시각 t=end에서 원시 샘플 timestamp가 `[t−60,t)`에 속할 때만 사용한다. 명세의 `(t−60,t]`와 끝점 관례를 다르게 구현한 이유는 Python 반개구간 인덱싱과 WESAD 동기화 시각을 일관되게 다루기 위해서다. 양쪽 모두 미래 샘플을 쓰지 않는다. 한 샘플의 관례 차이를 숨기지 않는다.
- EDA 4Hz, BVP 64Hz, ACC 32Hz를 각 레이트로 자른다. 원본 라벨은 700Hz로 매핑한다. 보간·업샘플·양방향 필터를 사용하지 않는다.
- 전체 창의 라벨이 정확히 하나의 1/2/3일 때만 평가한다. stress2→1, baseline1/amusement3→0. 혼합·명상·다른 라벨은 y=-1로 보존한다.
- 가슴 신호와 TEMP는 읽어 특징에 넣지 않는다. 데이터 끝은 가장 짧은 필수 채널·라벨에 맞춘다. 센서 지속시간 차이가 1초를 넘으면 extraction CLI가 중단하고 감사하도록 한다.
- `label_window`, `native_slice`의 끝점 및 혼합창은 unit test로 확인한다. `window_features`는 미래 및 입력창 이전의 값 변경에 불변이다.

## 특징

공통: 평균, 중앙값, 표준편차, 범위, 10/90분위수, 시간에 대한 선형 기울기, 평균 절대 차분×Hz, 평균 제곱 에너지. 유한한 관측만 통계에 사용하고 결측을 가로질러 차분하지 않는다.

EDA: 양의 차분 합, `scipy.signal.find_peaks`의 prominence 0.05 원자료 단위·최소 거리 1초인 후보 수. 이 값은 tonic/phasic 분해나 검증된 SCR 검출이 아니다. 원본 단위를 공식 readme로 확인해야 하며 임상적 사건 수로 부르지 않는다. 기록의 과거창 내부 이웃만 이용한다.

BVP: Welch 주파수 스펙트럼에서 0.7–3.5Hz 최대점 및 대역 에너지 비. BVP peak frequency를 검증된 HR, HRV 또는 IBI로 보고하지 않는다. 부분 결측인 창에서는 스펙트럼 값은 NaN이다.

ACC: 3축과 크기에 공통 통계, 축별 창 평균 제거 후 평균 제곱합(dynamic energy), 연속 벡터 변화량×Hz, 동일한 연속 값 비율, 크기 신호의 0.1–10Hz 최대점·대역 에너지 비. 축 평균 제거는 자세를 반영한 정확한 중력 분리가 아니다. '동일 값 비율'은 정지의 휴리스틱이며 임상 상태가 아니다. 단위는 원본 값을 유지하며 검증 전 g 또는 m/s²라고 단정하지 않는다. 에너지는 원본 단위의 제곱이다.

## 품질과 공정한 비교

- 비유한값(NaN/inf) 비율이 5% 초과하면 해당 센서의 특징을 모두 NaN으로 한다. 5% 이하는 유한값 통계만 계산한다. 필수 3개 샘플 미만은 invalid다.
- EDA/BVP의 유한값 표준편차 ≤1e-8은 거의 완전한 flatline 휴리스틱으로 invalid 처리한다. 이는 접촉불량을 임상적으로 검증한 판별기가 아니다.
- ACC는 각 축이 일정하면 flatline flag를 남기되 immobility와 dropout을 구분할 증거가 없으므로 이것만으로 invalid로 하지 않는다.
- BVP motion artifact/접촉불량/포화의 전체 탐지기를 구현한 것은 아니다. 움직임·스펙트럼·분산을 후속 점검용으로 사용한다. 생리적 정상 범위를 임의 cutoff로 만들어 적용하지 않는다.
- `qc_common_valid`는 세 센서가 모두 valid일 때만 True다. 주평가는 `y>=0 & qc_common_valid`라는 같은 창 집합을 모든 센서 조합에 사용한다. 센서별 별도 창 집합으로 성능을 비교하지 않는다.
- `qc_*`는 감사용이며 모델 특징에 넣지 않는다. 남는 NaN은 training fold에서만 학습한 median imputer로 처리한다. 별도 test 통계는 사용하지 않는다.
- 출력 `.quality.csv`는 피험자별 창 수, 라벨 유효 수, 공통 유효 수, 센서별 flatline/valid 수, 특징 NaN 비율, 특징 계산 p50/p95를 기록한다. 이 시간은 실행한 호스트의 측정이며 Windows/MCU 실측으로 바꾸어 쓰지 않는다.

## 실제 실행

```powershell
py -m src.features.extract --root data/raw/WESAD --out data/processed/wesad_features.csv --window 60 --step 15
py -m unittest discover -s tests -p "test_*.py"
```

기본 CLI는 공식 15명 ID가 정확히 존재해야 진행한다. `--allow-partial`은 명시적인 탐색/합성 테스트용이며 불완전한 자료를 완전한 WESAD 실험으로 보고할 수 없다. pickle은 원본 출처·무결성을 확인한 파일만 연다.

PhysioNet CSV loader는 Unix 숫자 또는 UTC 문자열 시작시각을 읽는다. 공식 tags·프로토콜로 확인된 segment manifest 없이 supervised label을 만들지 않는다. 현재 loader는 파일 구조를 읽는 도구일 뿐, 세션 전체에 상태 라벨을 부여하지 않는다.
