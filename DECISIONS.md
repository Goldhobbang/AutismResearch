# 설계 변경 기록

## 2026-10-23 / 사용자 변경 지시

사용자가 LTE를 제품 설계에서 제외하고 ESP32-S3, EDA/PPG/IMU/Mic/Light → 특징 → TinyML → BLE/진동/로컬로그 구조를 제시했다. 상면 안테나·MCU·전원·IMU·Mic/Light·DRV2605L, 하면 EDA 전극·PPG 창의 배치 구상도 제공했다.

이에 따라 **현행 제품은 BLE-only, ESP32-S3 우선**으로 변경한다. 이 지시는 원본 마스터 명세의 MCU 후보·LTE 옵션 결정을 대체한다. 이전 설계는 archive에 보존하고 현행 BOM·블록도·메모리·에너지·배치·발표를 갱신한다.

설계 정정: EDA 전극만 EDA AFE→ADC를 통과한다. PPG/IMU/Light는 각 디지털 인터페이스, 디지털 Mic는 I2S/PDM을 통해 MCU에 연결한다. MCU의 PDM/I2S 지원·핀·클록 조건은 선택한 마이크와 ESP-IDF 구성으로 확인한다. 모든 센서를 EDA AFE에 연결하지 않는다.

기본 연구 센서는 여전히 EDA/BVP/ACC다. Mic/Light의 추가 예측력은 WESAD·현재 PhysioNet 실험에서 검증할 수 없으며 L4 가설이다. CNN/RNN으로 범위를 확대하지 않고 LR부터 TinyML 이식 후보로 둔다. 원음은 저장하지 않으며 아동 손목 진동은 기본 비활성 상태로 제안한다.

상하면 배치는 전기적 연결만큼 중요하다. 배터리·금속·구리면의 안테나 keep-out 준수, PPG 광학 경로·차광·피부 접촉, EDA AFE의 고임피던스 배선, 진동의 광학·음향·전극 artifact를 검토한다. 실제 치수·회로·Gerber·실측은 이번 산출물에 포함하지 않는다.
