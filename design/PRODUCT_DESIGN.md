# ESP32-S3 손목형 시스템 설계안 (BLE-only)

개정: 2026-10-23 사용자 구상 반영. **실물 제작·배터리·무선·MCU 추론·아동 착용을 시험하지 않은 개념 설계**다. 현행 범위에서 LTE 모뎀·SIM·안테나·요금·전력 시나리오는 제외했다. 이전 버전은 archive에 보존했다.

## 1. 수정한 연결 구조

![센서별 인터페이스와 출력](SYSTEM_BLOCK.svg)

사용자 그림에서 바꾼 핵심은 **EDA 전극만 EDA Analog Front-End를 거친다는 점**이다. PPG·IMU·Mic·Light를 EDA AFE에 연결하지 않는다.

| 입력/출력 | 연결 제안 | 확인해야 할 것 |
|---|---|---|
| EDA 전극 A/B | 전류 제한·교정·필터를 포함한 EDA AFE → ADC → MCU | 전극 재료·면적·접촉·입력 누설·포화·µS 교정. MCU ADC 또는 외부 ADC 중 선택은 AFE와 잡음 예산으로 결정 |
| PPG | LED/photodiode/AFE/ADC/FIFO가 포함된 광학 센서 → I2C + interrupt | 원시 BVP 접근, 전압/논리레벨, 광학 차광·압력·피부/움직임 artifact. MAX30102는 예시 후보 |
| IMU | I2C 또는 SPI + data-ready/FIFO | 가속도3축 우선. range·축·실제 ODR을 기록. LSM6DS3TR-C 후보 |
| 디지털 Mic | 선택 마이크의 I2S 또는 지원되는 PDM 경로 → 짧은 DMA 프레임 | 마이크/ESP-IDF의 수신 모드·클록·핀 호환. 원음은 프레임 계산 후 폐기하고 저장·전송하지 않음 |
| Light | 조도 센서 I2C → MCU | 바깥쪽 개구부, 자체 LED/손/케이스 차광, 실제 integration time. VEML7700 계열은 검토 후보 |
| BLE Alert | MCU BLE → 가까운 보호자·교사 phone/tablet | 연결·시각·유효기간·수신확인·백그라운드 앱. 휴대폰이 없을 때 독립 원거리 알림을 보장하지 않음 |
| Haptic | MCU I2C → DRV2605L → 정격에 맞는 LRA/ERM | GPIO 직접 구동 금지. 모터 선택·구동 보정·전원 피크·광학/음향 artifact. 아동 손목 진동은 기본 비활성 제안 |
| Local Logging | flash의 순환 버퍼에 특징·품질·점수·event timestamp | 쓰기 수명·용량·암호화·보존기간·전원차단 복구. raw audio 비저장 |

Mic와 Light는 구상에 포함하되 **환경 맥락·품질 보조의 추가 센서**로 다룬다. 소리 크기나 조도가 곧 감각 과부하 또는 멜트다운 라벨은 아니다. 현재 WESAD/PhysioNet 입력에는 두 센서가 없으므로 추가 예측력을 검증했다고 주장하지 않는다. 알고리즘 입력은 우선 EDA/BVP/ACC이며, Mic/Light 투입은 별도 검증 이후다.

## 2. ESP32-S3 후보와 메모리

첫 개발 후보는 ESP32-S3-DevKitC-1-N8R8 / ESP32-S3-WROOM-1-N8R8 계열이다. 공식 개발보드 안내에서 이 변형의 Flash8MB/PSRAM8MB를 확인했다. SoC 내부 SRAM512KB와 외부 PSRAM은 구분한다. PSRAM 용량 전체를 DMA·실시간 내부 SRAM처럼 자유롭게 쓸 수 있다고 가정하지 않는다. 개발보드는 책상 위 실험 후보이며 최종 손목 크기 PCB가 아니다.

**모듈형이면 Flash/PSRAM은 모듈에 포함될 수 있다.** 사용자 상면의 별도 메모리 블록은 논리 블록으로 표시하고 실제 추가 IC가 필요한지는 SKU로 결정한다. 보드 변형마다 사용할 수 없는 GPIO·메모리 예약 핀이 있으므로 핀 번호는 아직 확정하지 않는다. Wi-Fi는 이번 기본 운용에서 사용하지 않는다.

| 메모리 항목 | 계산·설계 가정 | 한계 |
|---|---:|---|
| E4형60초 raw ring | (4+64+3×32)×60×4=39,360byte | 연구자료 주파수의 계산 |
| 새 장치 예: EDA4/PPG100/ACC52Hz | (4+100+3×52)×60×4=62,400byte | 후보 ODR로 다시 특징/모델 검증 필요 |
| Light10Hz float32 예 | 2,400byte/60초 | 실제 integration time에 맞출 것 |
| Mic16kHz/16bit/20ms | 640byte/프레임, 이중버퍼1,280byte | 파형은 짧은 휘발성 버퍼만. 전처리 scratch 별도 |
| Mic 원음60초 | 1,920,000byte | 저장하지 않는 설계. 작은 특징/요약만 유지 |
| 100특징 LR 수치배열 | 계수404+scaler800+imputer400=1,604byte | 전체 C runtime·stack·전처리 비용 아님 |

계획 예산은 센서 ring 약64KiB, Mic DMA/scratch16KiB, 특징·FFT32KiB, BLE/RTOS128KiB, 앱 stack/heap64KiB, 여유96KiB로 **400KiB 예시**다. ESP32-S3 내부512KiB 이내의 단순 합계일 뿐 실행 보장은 아니다. 예약 영역·링커·stack·DMA·할당 단편화를 확인하고 필요하면 PSRAM으로 로그·비핵심 버퍼를 옮긴다. TFLM 같은 별도 NN runtime을 택하면 tensor arena를 추가로 재산정한다.

## 3. Feature Extraction → TinyML

TinyML은 큰 신경망을 반드시 뜻하지 않는다. 첫 이식 후보는 실험에서 검증된 특징과 LR의 C 구현이다. 작은 RF 또는 int8 NN은 그다음 비교 대상이다. Python joblib를 MCU가 그대로 실행하지 않으므로 imputer/scaler/특징 순서·단위·계수를 export하고 PC 기준 출력과 golden-vector로 대조해야 한다.

15초마다 과거60초만 계산하고 측정 품질이 나쁘면 점수를 보류한다. Mic는 짧은 프레임의 RMS·클리핑·대역 에너지, Light는 평균·변화량 같은 후보부터 검토한다. 검증 없이 두 입력을 기존3센서 모델에 추가할 수는 없다. dB SPL 교정 전 마이크 값을 절대 소음 수준이라고 표시하지 않는다.

진동·LED 전류 변경·충전·센서 재시작은 artifact를 만들 수 있다. 해당 event를 기록하고 품질 보류/재안정화 정책을 실험으로 정한다. 비동기 취득·버퍼/시간 동기화·BLE와 로깅 부하를 함께 평가한다. 실제 MCU 지연은 아직 N/A다.

## 4. 상하면 배치와 배터리

![상면·하면·측면 개념도](PCB_LAYOUT.svg)

상면은 안테나 외곽, ESP32-S3 모듈, 전원/충전, IMU, Mic/Light의 바깥쪽 포트, DRV2605L를 배치한다. **EDA AFE를 전극 가까운 저잡음 영역에 추가**한다. 하면은 EDA 전극 A/B와 PPG 광학 창을 피부에 접촉시킨다.

배터리를 PCB 전체 밑에 깔면 안테나 keep-out과 PPG 광학 경로를 막을 수 있다. 배터리는 그 영역을 피한 구획 또는 별도 기구/플렉스 구조로 검토한다. 안테나 keep-out은 모든 PCB층과 기구 금속·배터리·나사·케이블에 대해 선택 모듈 도면을 적용한다. 임의의 거리 mm는 제시하지 않는다. 상세 검토는 [PCB_LAYOUT.md](PCB_LAYOUT.md)에 있다.

## 5. 전원·충전·BOM

150mAh 셀은 계산용 예일 뿐 확정 부품이 아니다. 셀의 허용 충전·방전·보호·온도·커넥터 극성과 충전 IC의 설정을 맞춘다. PMIC/LDO/충전기 기능이 겹칠 수 있으므로 세 IC를 무조건 따로 넣지 않는다. 배터리 측 평균전류·효율·피크·발열을 구분해 시험해야 한다.

[BOM.csv](BOM.csv)는 ESP32-S3 우선으로 개정했다. 공식 개발보드의 $15는 **제조사 표의 sample reference price**이지 계약 견적·배송가가 아니다. GSR $10.90/IMU $9.95/150mAh 셀 $5.95는 소매 참고다. 이들을 합친 $41.80 역시 **PPG·Mic·Light·진동/driver·실제EDA·전원·PCB·외장·조립·검사·배송·세금·인증·앱개발을 제외한 부분 참고 소계**다. 완성 제품 가격으로 쓰지 않는다. 기존 nRF 보드가 포함된 소계는 현행 설계에 적용하지 않는다.

[ASSUMPTIONS.md](ASSUMPTIONS.md)의 전류는 ESP32-S3를 포함한 새 구조의 임의 시나리오다. 이전 저전력MCU 시나리오를 ESP32-S3 수명으로 가져오지 않는다. 실제 전류·배터리 수명·온도·충전 호환성은 미검증이다.

## 6. 앱·로깅·검증 범위

앱은 측정불가, 연결끊김, 오래된 데이터, 연구 점수를 구분한다. 알림 수신 대상·담당자 인계·접근통제·암호화·삭제·보존기간을 정한다. local log는 raw audio 없이 특징·품질·모델 버전·시각·진동/상태 event를 제한된 순환 버퍼에 저장한다. 예컨대 100float특징400byte+메타데이터64byte를15초마다 기록하면 하루약2.67MB(십진수)라는 **계산 예**이며 실제 기록 포맷·압축·flash 예약에 따라 달라진다.

현재 모델은 성인 스트레스 분류용이다. 아동 위험 확률·멜트다운60초 경고로 바꾸어 부르지 않는다. 보호자 동의·가능한 아동 assent·철회·기관 윤리·착용/피부 검토와 별도 onset 자료, 독립 전향검증이 필요하다.

공식 근거: [ESP32-S3 데이터시트](https://www.espressif.com/sites/default/files/documentation/esp32-s3_datasheet_en.pdf), [개발보드 변형](https://docs.espressif.com/projects/esp-dev-kits/en/latest/esp32s3/esp32-s3-devkitc-1/user_guide_v1.1.html), [제조사 sample참고가격](https://www.espressif.com/en/products/devkits/esp32-s3-series/esp32-s3-devkitc-1), [하드웨어 배치 지침](https://docs.espressif.com/projects/esp-hardware-design-guidelines/en/latest/esp32s3/pcb-layout-design.html), [MAX30102](https://www.analog.com/en/products/max30102.html), [LSM6DS3TR-C](https://www.st.com/resource/en/datasheet/lsm6ds3tr-c.pdf), [DRV2605L](https://www.ti.com/product/DRV2605L), [Grove GSR 교정 주의](https://wiki.seeedstudio.com/Grove-GSR_Sensor/). 조회일2026-10-23.
