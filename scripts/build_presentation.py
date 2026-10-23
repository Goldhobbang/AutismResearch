"""Build the 11-slide Korean presentation; current real-data status stays N/A.

Optional authoring dependency: python-pptx. Not needed for model training.
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

ROOT=Path(__file__).resolve().parents[1]
NAVY='132B3D'; TEAL='187F91'; INK='173447'; GRAY='5D7282'; WHITE='FFFFFF'

SLIDES=[
('문제와 연구 질문','불편과 위험을 더 일찍 알아차리는 기술을 검증한다',[
 ('필요성','보호자의 상황 인식을 지원할 가능성을 연구한다. 당사자의 편안함과 거부 의사를 존중한다.'),
 ('구분할 상태','각성·스트레스·감각 과부하·멜트다운·공격·자해·반복행동은 같은 라벨이 아니다.'),
 ('검증할 질문','손목 신호의 정보, 센서 조합의 효과, 새 참가자 일반화와 운동 혼동을 차례로 평가한다.')],
 'L4 연구 목적 | 멜트다운 감소·60초 경보는 미검증'),
('선행연구의 실제 근거','숫자보다 대상·미래구간·분할·지표를 함께 읽는다',[
 ('Goodwin 2019','ASD20명·타인 공격. 과거180초→향후60초. AUROC 집단 .71 / 개인별 .84.'),
 ('Imbiriba 2023','입원70명·497시간·6,665행동. 선택된3분 조건 LR AUROC .80. 행동 이력 입력을 구별한다.'),
 ('후속 연구의 한계','Kim9명 LOSO의 개인차·오경보, PreMAC의 전조 목표, REACT의 기록 스트림 모사를 구분한다.')],
 'L2 | AUROC .84 ≠ 정확도84%·경고정밀도84% | EVIDENCE_LEDGER.md'),
('이번 목표와 장기 목표','공개자료 연구와 아동 조기경보를 구분한다',[
 ('이번에 제공한 것','성인 WESAD 15명 원본 학습·LOSO 평가 완료, 7개 센서 비교, 제품 설계.'),
 ('확인한 결과','WESAD 15명 LOSO LR AP .9048, RF .8586. ACC 단독 AP .9245, EDA 단독 .8045.'),
 ('장기 목표','실제 아동 onset 자료에서 사건 단위 지표와 최소60초 선행 경고를 독립 검증한다.')],
 'L2-실데이터 검증 | 성인 실험실 스트레스 분류 | 아동 멜트다운 지표는 N/A'),
('미제작 손목형 시스템','센서 → ESP32-S3 → BLE / 진동 / 특징로그',[
 ('ESP32-S3 모듈형','N8R8 후보: Flash8MB/PSRAM8MB. EDA만 AFE를 거치고 Mic/Light는 맥락 보조로 둔다.'),
 ('상하면·적층 배치','EDA AFE를 전극 가까이, Mic/Light는 바깥 포트에 둔다. 배터리는 RF keep-out과 PPG 경로를 피한다.'),
 ('BLE·로컬 출력','근처 보호자 앱, DRV2605L 진동, 원음 없는 특징로그. 부분 참고소계 USD41.80은 완제품 원가가 아니다.')],
 'DOC + L3 계산 | 제작·MCU 추론·배터리 실측 없음'),
('상하면 배치와 신호 품질','배터리·안테나·광학 경로를 함께 검토한다',[
 ('수집과 품질','타임스탬프·센서 접촉·누락·flatline·움직임을 확인한다. 품질 불량이면 점수를 보류한다.'),
 ('점수와 전달','과거창 특징→모델 점수→유효기간·연결 확인→보호자 검토. 점수는 보정된 위험확률이 아니다.'),
 ('앱 화면안','연구모드 / 측정 불가 / 연결 끊김. 원음 비저장, 기본 손목 진동 비활성을 제안한다.')],
 '설계 시안 | design/APP_MOCKUP.html | 실제 사용자 평가 미실행'),
('AI 실험 설계','60초 과거창 · 15초 hop · 외부15명 LOSO',[
 ('공정한 분리','동일 참가자의 모든 창을 한 fold에 유지한다. imputer/scaler는 train, threshold는 내부 validation만 쓴다.'),
 ('모델 비교','Dummy prior → LR → RF. 전체 센서3모델 + LR7센서 조합으로9개 고유 설정.'),
 ('평가 단위','참가자별 AP/F1/BA·양성비율·범위·하위25%·CI를 보고한다. 중첩 창은 독립 참가자가 아니다.')],
 '사전 계획 | EXPERIMENT_REGISTRY.md | P0/P1 전 P2 고급모델 중지'),
('직접 실행한 WESAD 실험과 검증','Ryzen 7 8845HS에서 15명 LOSO 135회 학습 완료',[
 ('원본 무결성','15명 전원 pickle SHA256 확인, 동기화 편차 0.0s, 총 5,740 윈도우 특징 추출.'),
 ('모델 비교','Dummy AP .2987 → RF AP .8586 (F1 .6142) → LR AP .9048 (F1 .7425).'),
 ('재생 검증','S2 원시 신호 스트림 재생: 지연시간 p50 9.48ms, OOF 점수 일치율 100%.')],
 'L2-실데이터 | results/wesad/metrics.json / summary.csv | 19개 단위검사 통과'),
('성과와 현재 검증 범위','실제 WESAD 성능을 확인하고 아동 과제와 분리한다',[
 ('완료한 성과','WESAD 15명 LOSO 학습 완료, 7개 센서 비교, ESP32-S3 BOM·배치안, S2 지연 실측.'),
 ('센서 기여도','EDA 대비 ACC 추가 시 +0.1003 AP 개선(10명 향상), PPG 단독은 -0.0711로 하락.'),
 ('N/A 엄격 유지','ASD 멜트다운 사건 P/R, false alarms/day, 실제 아동 lead time 및 중재 효과.')],
 '실데이터 결과 반영 | reports/RESULTS_CURRENT.md / QUALITY_REPORT.md'),
('기대효과는 검증할 가설','센서를 더 넣거나 데이터를 늘리면 항상 좋아지는가',[
 ('센서 기여','같은 참가자·같은 창에서 전체−EDA의 paired AP 차이와 개선/악화 인원을 본다.'),
 ('운동 혼동','검증된 PhysioNet 구간에서 EDA+BVP와 +ACC를 비교한다. 프로토콜·성별 교란을 함께 보고한다.'),
 ('부정적 결과의 가치','일반화 한계·움직임 잡음·추가 센서의 무이득도 가설 검증 결과다. 미실행과 낮은 성능은 다르다.')],
 'H1–H4 사전 가설 | 결과에 맞춘 성공 기준 변경 금지'),
('한계와 후속 로드맵','성인 실험실 상태에서 아동 일상 사건으로',[
 ('자료·윤리','동의·assent·철회·기관 검토, 자연발생 관찰, onset 오차·평가자 일치도·품질을 기록한다.'),
 ('독립 예측 검증','새 아동·새 날짜·새 환경 silent-mode 검증. 사건 matching과 경고 episode·누락·부담을 평가한다.'),
 ('실제 도움 검증','예측 후 별도 비교 중재연구로 사건 심각도·지속·아동 불편·보호자 부담 감소를 평가한다.')],
 'L4 후속 계획 | 필요한 아동/사건 수는 파일럿·불확실성으로 설계'),
('결론과 다음 실행','실데이터 기반 검증과 아동 임상 로드맵',[
 ('확인한 사실','성인 손목 센서로 스트레스 분류 가능(LR AP .9048). ACC 기여도 확인, RF 과적합 주의.'),
 ('제품화 전제','ESP32-S3 모듈형(N8R8) 및 BLE 통신 기반 경량 TinyML 파이프라인 실현 가능.'),
 ('다음 단계','PhysioNet 운동 혼동 분석, 실제 아동 onset 수집 프로토콜 및 다기관 전향 검증.')],
 '성인 데이터 실측 완료 | 아동 일상 사건 검증으로 확장 필요')
]


def color(value): return RGBColor.from_string(value)


def text(slide,x,y,w,h,content,size=22,bold=False,fg=INK):
    box=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    frame=box.text_frame;frame.word_wrap=True
    frame.margin_left=frame.margin_right=Inches(.02)
    frame.margin_top=Inches(.02)
    for i,line in enumerate(content.split('\n')):
        p=frame.paragraphs[0] if i==0 else frame.add_paragraph()
        p.text=line;p.font.name='맑은 고딕';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=color(fg)
        p.space_after=Pt(5)
    return box


def main():
    presentation=Presentation();presentation.slide_width=Inches(13.333);presentation.slide_height=Inches(7.5)
    for number,(title,subtitle,cards,foot) in enumerate(SLIDES,1):
        slide=presentation.slides.add_slide(presentation.slide_layouts[6])
        bg=slide.background.fill;bg.solid();bg.fore_color.rgb=color('F4F7FA')
        top=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,presentation.slide_width, Inches(1.95))
        top.fill.solid();top.fill.fore_color.rgb=color(NAVY);top.line.fill.background()
        text(slide,.6,.35,12.1,.6,title,30,True,WHITE)
        text(slide,.6,1.10,12.1,.55,subtitle,21,False,'B5DCE5')
        if number in (4,5):
            image_name='SYSTEM_BLOCK.png' if number==4 else 'PCB_LAYOUT.png'
            slide.shapes.add_picture(str(ROOT/'design'/image_name),Inches(.5),Inches(2.15),width=Inches(7.7))
            notes = ([('센서 경로','EDA만 AFE/ADC. PPG·IMU·Mic·Light는 각 디지털 인터페이스.'),
                      ('온디바이스','ESP32-S3 모듈형, LR부터. Mic/Light 추가 예측력은 미검증.'),
                      ('출력','BLE·DRV2605L·특징로그. 원음 비저장, 손목 진동 기본 비활성.')]
                     if number==4 else
                     [('RF keep-out','모든 PCB층과 배터리·나사·케이블까지 확인. 실제 모듈 도면 기준 적용.'),
                      ('피부 경로','EDA AFE는 전극 가까이. PPG 차광/개구부를 배터리가 가리지 않게 설계.'),
                      ('보조 센서','Mic/Light는 바깥 포트. 진동의 센서 artifact를 기록·검증.')])
            for idx,(heading,body) in enumerate(notes):
                y=2.25+idx*1.35
                text(slide,8.5,y,4.2,.4,heading,20,True,TEAL)
                text(slide,8.5,y+.48,4.2,.95,body,17)
        else:
            for idx,(heading,body) in enumerate(cards):
                x=.6+idx*4.1
                card=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x),Inches(2.3), Inches(3.9), Inches(3.9))
                card.fill.solid();card.fill.fore_color.rgb=color(WHITE);card.line.color.rgb=color('DCE5EC')
                text(slide,x+.2,2.6,3.5,.6,heading,22,True,TEAL)
                text(slide,x+.2,3.35,3.5,2.5,body,21)
        text(slide,.6,6.55,11.9,.5,foot,12,False,GRAY)
        text(slide,12.35,6.85,.5,.3,str(number),12,False,TEAL)
        slide.notes_slide.notes_text_frame.text=(
            '상태 기준: 2026-10-23. 실제 원자료 다운로드 및 성능평가는 차단 상태. '
            '합성 검사 수치를 성인/아동 예측 성능으로 발표하지 않는다.\n'
            '근거·링크: EVIDENCE_LEDGER.md, reports/FINAL_REPORT_KO.md, '
            'design/PRODUCT_DESIGN.md, results/logs/ 참조.\n'+foot)
    target=ROOT/'reports/PRESENTATION_11_KO.pptx';presentation.save(target)
    assert len(presentation.slides)==11
    print(target)


if __name__=='__main__':main()
