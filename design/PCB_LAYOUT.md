# 상면·하면·적층 배치 수정안

기준: 2026-10-23 사용자 구상. **배치 개념도이며 실제 PCB·기구 CAD, 확정 치수, Gerber, DFM 또는 RF 검증 결과가 아니다.** 현재 제품 범위는 BLE-only, ESP32-S3다.

## 1. 상면 (바깥쪽)

| 영역 | 배치 제안 | 이유·검증할 점 |
|---|---|---|
| 가장자리 안테나 | PCB 안테나가 있는 ESP32-S3 모듈의 안테나 끝을 외곽으로 배치 | 선택 모듈의 제조사 keep-out을 따른다. 해당 영역의 모든 PCB층·주변 기구·금속·배터리·나사·케이블까지 검토한다. 단순히 상면만 비우면 충분하다고 가정하지 않는다 |
| ESP32-S3 | ESP32-S3-WROOM-1-N8R8 계열 모듈을 개발 후보로 검토 | 모듈형이면 Flash/PSRAM이 모듈 안에 포함된다. 별도 Flash/PSRAM 두 블록을 다시 올릴 필요가 없다. SoC 단독 설계는 별도 선택이다 |
| EDA AFE | 하면 전극 접점/연결부 가까운 조용한 영역 | 입력 배선은 짧게 하고, 충전·스위칭 전원·RF·모터 배선에서 분리한다. 누설·오염·습기·전극 접촉을 고려한다. guard의 필요성과 구현은 AFE 확정 후 검토한다 |
| PMIC/충전 | 배터리 커넥터 가까이, 광학·피부 접촉부에서 열 영향을 줄이는 위치 | PMIC/LDO/charge IC의 기능 중복을 피한다. 실제 rail·충전전류·보호·power-path·온도감시를 회로로 확인한다 |
| IMU | 기구에 단단히 고정된 위치, 모터 진동이 과도하게 전달되지 않는 곳 | 중심에 무조건 놓는 것이 정답은 아니다. 축 방향·기구 결합·진동 artifact를 실측한다 |
| Mic/Light | 바깥쪽 전용 음향 포트·조도 개구부 근처 | 마이크를 밀폐 배터리 뒤에 넣거나 조도센서를 불투명 케이스 안에 넣지 않는다. 방수막/메시·차광 구조·자체 LED/화면광·손목마찰 영향을 검토한다 |
| DRV2605L/진동 | 모터와 연결이 짧고 PPG/전극부와 기계적으로 간섭이 적은 영역 | MCU GPIO 직접 구동 금지. 선택 LRA/ERM의 정격·구동·보정·전원 피크를 맞춘다. 진동 구간을 로그에 기록한다 |

## 2. 하면 (피부 접촉면)

- **EDA 전극 A/B:** 피부에 안정적으로 접촉하고 PPG 창과 전기·광학·기구 간섭이 적도록 배치한다. 간격·면적·재료·압력은 미확정이며 착용·피부·교정 검토가 필요하다. 손목 EDA를 손가락 EDA와 동일하게 취급하지 않는다.
- **PPG optical window:** LED/광검출기의 광학 경로가 피부로 열려 있어야 한다. 불필요한 주변광·내부 광 누설을 줄이는 차광 가스켓/분리 구조를 검토한다. 투명 플라스틱을 단순히 덮는 것만으로 검증된 광학부가 완성되지는 않는다.
- **접촉과 기구:** 돌출 높이·스트랩 압력·피부색·털·땀·움직임·세척·재료·열을 평가한다. 아동 착용 안전성은 부품 판매 사양으로 보장되지 않는다.

## 3. 배터리 적층 — 반드시 측면에서 확인

`Battery Below`를 PCB 전체 밑에 채우면 안테나 금속 keep-out과 PPG 광학 경로를 침범할 수 있다. 제안은 배터리를 **안테나와 PPG 개구부를 피한 인접 영역**에 두거나 분리 기판/플렉스·기구 구조를 검토하는 것이다. EDA 접점까지의 배선도 동시에 확인한다.

측면 개념: 상부 케이스(음향·조도 개구부) → PCB/센서 → 배터리가 차지하는 제한된 구획 + 별도의 PPG/전극 피부 경로 → 피부 접촉면. 실제로 이 구성이 들어가는지 판단할 외형 치수·셀 치수·광학 높이·스트랩 구조는 아직 없다.

## 4. 데이터 품질과 배치의 연결

- 진동 중/직후에는 PPG·EDA·IMU·마이크가 영향을 받을 수 있다. haptic event timestamp를 남기고 해당 구간의 품질 보류·재안정화 시간을 실험으로 정한다. 고정 회복시간을 근거 없이 보장하지 않는다.
- 충전 상태·RF 송신·LED 전류 변경·센서 재시작·패킷 유실을 로그에 남긴다. 가능하면 충전 중 착용하지 않는 연구 흐름을 검토한다.
- Light/Mic는 감각 과부하 정답 센서가 아니다. 환경 맥락·품질 보조로부터 검증하며 추가 예측력을 별도 실험한다.

도면: [PCB_LAYOUT.svg](PCB_LAYOUT.svg), 블록도: [SYSTEM_BLOCK.svg](SYSTEM_BLOCK.svg).

공식 근거: [ESP32-S3-DevKitC-1 v1.1](https://docs.espressif.com/projects/esp-dev-kits/en/latest/esp32s3/esp32-s3-devkitc-1/user_guide_v1.1.html), [ESP32-S3-WROOM-1 데이터시트](https://www.espressif.com/sites/default/files/documentation/esp32-s3-wroom-1_wroom-1u_datasheet_en.pdf), [Espressif 하드웨어 설계 지침](https://docs.espressif.com/projects/esp-hardware-design-guidelines/en/latest/esp32s3/pcb-layout-design.html), [DRV2605L](https://www.ti.com/product/DRV2605L), [MAX30102](https://www.analog.com/en/products/max30102.html). 개념도에 임의의 keep-out mm를 적지 않았으며 최종 모듈 도면의 수치를 적용해야 한다.
