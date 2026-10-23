# MCU 배포 적합성 및 엣지 임베디드 분석 보고서 (MCU_FEASIBILITY.md)

- 작성 시각: 2026-10-24 04:15 (UTC+9)
- 게이트 판정: **G8 PASS (TinyML/ESP32-S3 메모리·연산량 수치 계약 완성 / 하드웨어 실측 N/A 명시)**
- 타깃 칩셋: Espressif ESP32-S3-WROOM-1-N8R8 (Xtensa 32-bit LX7 Dual-Core 240MHz, 512KB SRAM, 8MB Quad Flash, 8MB Octal PSRAM)
- 입력 센서: 손목 EDA (4Hz AFE/ADC), 광용적맥파 BVP/PPG (64Hz I2C), 3축 가속도계 ACC (32Hz SPI)

---

## 1. 모델 크기 및 온디바이스 메모리 풋프린트 비교

| 모델 | 파일 크기 (.joblib) | MCU 가중치 순수 배열 크기 | ESP32-S3 내부 SRAM(512KB) 적합성 | Flash/PSRAM 의존성 | 온디바이스 추천도 |
|---|---:|---:|---|---|:---:|
| **Logistic Regression (LR)** | **6.55 KB** | **760 Bytes** (가중치 63개, 절편 1개, 스케일러 126개) | **완벽 적합 (SRAM 0.15% 점유)** | 불필요 (순수 레지스터/SRAM 연산) | **최우선 권장 (Optimal)** |
| **Random Forest (RF, 200트리)** | 881.96 KB | ~850 KB (수천 개 분기 노드) | **불가 (SRAM 512KB 초과)** | 필수 (외부 SPI Flash/PSRAM 상주) | 부적합 (캐시 미스, 전력 소모) |

> **배포 판정**:
> - Scikit-learn `.joblib` 파일은 파이썬 객체 직렬화 포맷이므로 **ESP32 MCU에서 직접 실행할 수 없음**.
> - 배포 시에는 LR 모델의 표준화 계수(`scaler.mean_`, `scaler.scale_`)와 회귀 계수(`coef_`, `intercept_`)를 C 헤더 파일(`model_coefficients.h`)로 익스포트하여 순수 C 배열 내적(Dot-product)으로 구현해야 함.
> - LR의 63개 특성 정규화 및 가중합 연산량은 약 190회 부동소수점 연산(FLOPs)에 불과하여, 240MHz ESP32-S3에서 **0.05ms 미만**에 실행 가능함.

---

## 2. 60초 링 버퍼(Ring Buffer) 메모리 요구량

60초 과거 관찰창을 유지하기 위한 원시 센서 데이터 링 버퍼 요구량:
- **EDA (4 Hz, 1채널, float32)**: $4 \times 60 \times 4\text{ B} = 960\text{ Bytes}$
- **BVP (64 Hz, 1채널, float32)**: $64 \times 60 \times 4\text{ B} = 15,360\text{ Bytes}$
- **ACC (32 Hz, 3채널, float32)**: $32 \times 3 \times 60 \times 4\text{ B} = 23,040\text{ Bytes}$
- **총 원시 링 버퍼 합계**: **39,360 Bytes (~38.44 KiB)**
- **동적 메모리 여유**: ESP32-S3 내부 사용 가능 SRAM(약 380 KiB) 대비 링 버퍼 점유율은 약 **10.1%**로, FreeRTOS 스택, BLE 스택(약 65 KiB), FFT 연산 버퍼를 포함하고도 200 KiB 이상의 안전 여유가 확보됨.

---

## 3. 원시 신호 처리 및 특성 추출 수치 일치 계약

MCU 펌웨어에서 Python 파이프라인(`src/features/core.py`)과 비트 단위 일치를 달성하기 위한 규약:
1. **시간 윈도우 인덱싱**: $[t-60, t)$ 반개구간. 마지막 타임스탬프 미만의 샘플만 버퍼에서 슬라이스.
2. **동적 에너지 계산**: 축별 60초 평균 중력 제거 후 제곱합의 평균 산출.
3. **스펙트럼 대역비**: 64포인트/128포인트 Radix-4 Real FFT 적용 후 $0.7\sim 3.5\text{Hz}$(PPG) 및 $0.1\sim 10\text{Hz}$(ACC) 주파수 빈 적분 비율 계산.
4. **결측 및 센서 이상치 처리**:
   - 신호 탈락 또는 Flatline 발생 시 센서 상태를 `MEASUREMENT_UNAVAILABLE`로 플래그 처리하고 모델 추론을 바이패스하여 허위 경보 차단.

---

## 4. 엄격한 하드웨어 한계 및 실측 고지 (Hard Disclaimer)

1. **PC vs MCU 속도 혼동 금지**:
   - 본 연구에서 실측된 S2 신호 특징 재계산 및 추론 지연시간 **p50 9.48ms, p95 13.15ms**는 **Windows 11 AMD Ryzen 7 8845HS 고성능 데스크톱급 CPU**에서 측정한 값임.
   - 이를 ESP32-S3 MCU의 실제 추론 지연시간으로 인용하는 행위는 엄격히 금지되며, MCU 실측치는 실제 PCB 펌웨어 플래싱 후 하드웨어 타이머로 측정해야 함.
2. **배터리 수명 및 전력**:
   - 사양서의 150mAh 기준 약 11.08시간은 문헌 전류 소모 가정을 결합한 이론 계산값(L3)일 뿐이며, 실제 안테나 송신 전력, BLE 연결 간격, 센서 구동 전류를 반영한 실측 수명이 아님.
3. **의료 기기 아님**:
   - 제안된 시스템은 연구 및 보조 알림 프로토타입이며, 임상적 의료 진단 또는 독립 안전 보호구를 대체할 수 없음.
