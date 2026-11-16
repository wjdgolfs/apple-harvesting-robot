from pathlib import Path
import sys,json,hashlib
from docx import Document
from docx.shared import Cm,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[3]; BUILD=Path(__file__).parent; OUT=BUILD.parent/'output'
DOC=Document(); sec=DOC.sections[0]
sec.page_width=Cm(21);sec.page_height=Cm(29.7)
sec.top_margin=Cm(1.65);sec.bottom_margin=Cm(1.5);sec.left_margin=Cm(1.55);sec.right_margin=Cm(1.55)
sec.header_distance=Cm(.6);sec.footer_distance=Cm(.6)
for name in ['Normal','Title','Heading 1','Heading 2','Caption']:
 s=DOC.styles[name];s.font.name='맑은 고딕';s._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'맑은 고딕');s.font.color.rgb=RGBColor(0,0,0)
 s.font.size=Pt(10)
 s.paragraph_format.space_after=Pt(5);s.paragraph_format.line_spacing=1.12
DOC.styles['Title'].font.size=Pt(13);DOC.styles['Title'].font.bold=True
DOC.styles['Heading 1'].font.size=Pt(11);DOC.styles['Heading 1'].font.bold=True
DOC.styles['Heading 2'].font.size=Pt(10);DOC.styles['Heading 2'].font.bold=True
DOC.styles['Caption'].font.size=Pt(8);DOC.styles['Caption'].font.color.rgb=RGBColor.from_string('444444')
def shade(c,color):
 pr=c._tc.get_or_add_tcPr();e=OxmlElement('w:shd');e.set(qn('w:fill'),color);pr.append(e)
def margins(c,v=85):
 pr=c._tc.get_or_add_tcPr();m=OxmlElement('w:tcMar')
 for k in ['top','left','bottom','right']:
  n=OxmlElement('w:'+k);n.set(qn('w:w'),str(v));n.set(qn('w:type'),'dxa');m.append(n)
 pr.append(m)
def borders(t,color='555555',sz='5'):
 pr=t._tbl.tblPr;e=OxmlElement('w:tblBorders')
 for k in ['top','left','bottom','right','insideH','insideV']:
  x=OxmlElement('w:'+k);x.set(qn('w:val'),'single');x.set(qn('w:sz'),sz);x.set(qn('w:color'),color);e.append(x)
 pr.append(e)
def clean_cell(c):
 c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER;margins(c)
 for p in c.paragraphs:p.paragraph_format.space_after=Pt(3)
def celltext(c,txt,bold=False,size=9,center=False):
 c.text='';p=c.paragraphs[0];p.alignment=WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT
 r=p.add_run(txt);r.bold=bold;r.font.size=Pt(size);clean_cell(c);return p
def p(c,text,style=None,bold=False):
 if len(c.paragraphs)==1 and not c.paragraphs[0].text and not len(c.paragraphs[0]._p):x=c.paragraphs[0]
 else:x=c.add_paragraph()
 if style:x.style=style
 r=x.add_run(text);r.bold=bold
 return x
def head(c,text):return p(c,text,'Heading 1')
def sub(c,text):return p(c,text,'Heading 2')
def table(c,rows,widths=None,size=9):
 t=c.add_table(rows=len(rows),cols=len(rows[0]));t.autofit=False;t.alignment=WD_TABLE_ALIGNMENT.CENTER
 if widths:
  for col,w in zip(t.columns,widths):col.width=Cm(w)
 for i,row in enumerate(rows):
  for j,val in enumerate(row):
   cc=t.cell(i,j)
   if widths:cc.width=Cm(widths[j])
   celltext(cc,str(val),i==0,size)
   if i==0:shade(cc,'E7E7E7')
  trpr=t.rows[i]._tr.get_or_add_trPr();ns=OxmlElement('w:cantSplit');trpr.append(ns)
 borders(t,'888888','4')
 c.add_paragraph().paragraph_format.space_after=Pt(1)
 return t
def pic(c,path,width,caption):
 pp=c.add_paragraph();pp.alignment=WD_ALIGN_PARAGRAPH.CENTER;pp.paragraph_format.space_after=Pt(2)
 pp.add_run().add_picture(str(ROOT/path),width=Cm(width))
 cp=c.add_paragraph(caption,style='Caption');cp.alignment=WD_ALIGN_PARAGRAPH.CENTER
def page(num,label,first=False):
 if not first:DOC.add_page_break()
 # Original HWP header: report title on the left, original ERICA logo on the right.
 ht=DOC.add_table(rows=1,cols=2);ht.autofit=False;ht.columns[0].width=Cm(13.3);ht.columns[1].width=Cm(4.6)
 a=ht.cell(0,0);b=ht.cell(0,1);a.width=Cm(13.3);b.width=Cm(4.6)
 pp=a.paragraphs[0];pp.style='Title';pp.add_run('팀 활동 보고서  팀명 Team.D_24')
 pp=b.paragraphs[0];pp.alignment=WD_ALIGN_PARAGRAPH.RIGHT;pp.add_run().add_picture(str(BUILD/'BIN0001.png'),width=Cm(4.0))
 pr=ht._tbl.tblPr;bd=OxmlElement('w:tblBorders');bb=OxmlElement('w:bottom');bb.set(qn('w:val'),'single');bb.set(qn('w:sz'),'10');bb.set(qn('w:color'),'1B5884');bd.append(bb);pr.append(bd)
 cp=DOC.add_paragraph(f'사과 수확 로봇 비전 프로젝트  /  2026년 11월 16일 기준  /  {num}');cp.paragraph_format.space_after=Pt(6)
 for r in cp.runs:r.font.size=Pt(8)
 if first:
  t=DOC.add_table(rows=3,cols=3);t.autofit=False
  for col,w in zip(t.columns,[1.65,2.5,13.75]):col.width=Cm(w)
  for row in t.rows:
   for cc,w in zip(row.cells,[1.65,2.5,13.75]):cc.width=Cm(w);clean_cell(cc)
  t.cell(0,0).merge(t.cell(2,0));celltext(t.cell(0,0),'미팅\n개요',True,10,True);shade(t.cell(0,0),'D0D0D0')
  for i,(k,v) in enumerate([('일시','2026.11.16 기준 작업 현황 정리'),('참석자','윤종호, 정해린'),('본 미팅의\n주요활동','검출·분할 데이터 준비와 학습 실행 점검\n익음도·무게 전이학습 및 합성 가림 평가\n개별 기능 평가와 임시 통합 방향 정리')]):
   celltext(t.cell(i,1),k,True,9,True);shade(t.cell(i,1),'D0D0D0');celltext(t.cell(i,2),v,False,10)
  borders(t);DOC.add_paragraph().paragraph_format.space_after=Pt(1)
 t=DOC.add_table(rows=2,cols=3);t.autofit=False
 for col,w in zip(t.columns,[1.65,2.5,13.75]):col.width=Cm(w)
 for row in t.rows:
  for cc,w in zip(row.cells,[1.65,2.5,13.75]):cc.width=Cm(w);clean_cell(cc)
 t.cell(0,0).merge(t.cell(1,0));celltext(t.cell(0,0),'추후\n계획' if num==8 else '진행\n사항',True,10,True);shade(t.cell(0,0),'D0D0D0')
 celltext(t.cell(0,1),'구분',True,9,True);shade(t.cell(0,1),'D0D0D0');celltext(t.cell(0,2),'활동 내용',True,9,True);shade(t.cell(0,2),'D0D0D0')
 celltext(t.cell(1,1),label,True,9,True);shade(t.cell(1,1),'D0D0D0');borders(t)
 content=t.cell(1,2);content.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.TOP
 return content

c=page(1,'이번\n미팅에서\n한 일',True)
head(c,'1 프로젝트 목표와 현재 도달 단계')
p(c,'우리 팀은 공개 영상과 사진에서 사과의 위치와 개별 영역을 찾고, 익음도와 예상 무게를 분석하는 비전 기능을 개발하고 있다. 이번 정리는 기존 주제 선정·자료 조사 이후 수행한 데이터 준비, 학습 실행 및 기능별 실험 결과를 취합한 것이다.')
p(c,'현재 검출·분할 데이터 준비와 소량 학습 동작 확인을 완료했으며, 익음도·무게는 ResNet18 전이학습과 합성 가림 증강의 비교 결과를 확보했다. 전체 검출·분할 학습과 실제 검출 결과를 이용한 통합 평가는 아직 완료하지 않았다.',bold=True)
table(c,[['담당','수행 내용','현재 상태'],['윤종호','MinneApple 검수·분할·라벨 변환, YOLO 학습 실행','데이터 준비·짧은 학습 완료\n전체 학습 중단, Colab 준비'],['정해린','익음도·무게 기준 모델, ResNet18 및 가림 비교','개별 기능의 예비 평가 완료'],['공통','결과 취합, 입력·출력 형식 및 통합 순서 검토','독립 평가와 임시 연결 방안 정리']],[1.65,7.15,4.45])
sub(c,'이번 활동의 핵심 결과')
p(c,'검출·분할은 670쌍의 이미지·마스크와 28,183개 주석을 학습 입력으로 준비했다. 익음도 증강 모델은 추가 가림 50%에서도 익음 20건 중 18건을 찾았지만, 원본 조건의 익음 정밀도는 35.2%로 오선택 개선이 필요하다. 무게 증강 모델의 MAE는 원본 13.41g, 추가 가림 50%에서 18.48g이었다.')
p(c,'위 수치는 각 기능의 개별 평가 결과다. 실제 수확 동작, 과수원 현장 무게 추정, 움직임 실험 및 통합 시스템 성능을 의미하지 않는다.')

c=page(2,'윤종호\n작업 내용')
head(c,'2 검출 및 개별 분할 데이터 준비')
p(c,'MinneApple 이미지와 개별 객체 마스크 670쌍의 대응·읽기·크기·형식을 검사하고 샘플을 시각 검수했다. 파일명 접두사의 촬영 시퀀스를 기준으로 분할해 인접 프레임의 학습·검증 혼입을 줄였다. 검증 데이터는 모델 선택용이며 독립 최종 시험 자료와 구분한다. [1]')
table(c,[['구분','이미지 수','사과 주석 수','촬영 시퀀스'],['학습','518장','22,082개','8개'],['검증','152장','6,101개','2개'],['합계','670장','28,183개','10개']],[2.2,2.9,4.1,4.05])
sub(c,'마스크를 YOLO 다각형 라벨로 변환')
p(c,'객체 ID별 영역을 한 사과당 한 줄의 YOLO 분할 라벨로 변환했다. 떨어진 조각·내부 구멍·점 또는 선처럼 작은 윤곽을 처리하고, 원본 ID 대응표와 변환 통계를 저장했다. 변환 제외 객체는 0개다. 데이터 YAML과 실제 데이터 로더에서 670개 파일·28,183개 객체를 읽는지 확인했다.')
pic(c,Path('outputs/sample_inspection/20150919_174151_image1/comparison.png'),12.8,'그림 1  원본 이미지와 개별 사과 마스크의 대응 검수 예시  출처 MinneApple 및 윤종호 검수 결과')
p(c,'MinneApple은 전경 나무의 사과를 중심으로 주석한다. 영상 속 배경 사과 전체를 정답 대상으로 보아서는 안 된다. 서로 다른 시퀀스에 같은 나무·배경이 남을 가능성도 분할의 한계로 기록했다.')

c=page(3,'윤종호\n작업 내용')
head(c,'3 학습 동작 확인과 메모리 문제 대응')
p(c,'YOLOv8n-seg 사전학습 모델에서 시작해 학습 32장·검증 8장으로 3 epoch의 짧은 학습을 실행했다. GPU 학습·검증, 손실 기록, best.pt와 last.pt 저장 및 모델 재로딩을 확인했다. 분할 손실은 4.43778에서 4.98167로 증가했으므로, 이 실험은 학습 동작 확인으로 해석한다. [2]')
table(c,[['실행','설정 또는 결과','상태'],['짧은 학습','32장 / 8장, 3 epoch, RTX 5060 Laptop','정상 종료'],['마스크 축소 검사','1,797개 중 mask_ratio=4에서 2개가 비어 있음','작은 객체 표현 문제 확인'],['고해상도 마스크 검사','전체 28,183개, mask_ratio=1에서 빈 마스크 0개','전처리 검사 통과'],['전체 학습 batch 2','첫 epoch 도중 CUDA 메모리 부족','미완료'],['전체 학습 batch 1','1 epoch 학습·검증 후 2 epoch 도중 종료','미완료'],['Colab 전환','노트북 저장, 데이터 세 파일 Drive 업로드','GPU·환경 확인 전, 학습 미시작']],[3.0,7.0,3.25],8.7)
sub(c,'원인과 대응 방향')
p(c,'작은 사과 마스크를 유지하려고 mask_ratio=1을 적용했으나 분할 손실 계산의 GPU 메모리 부담이 커졌다. 전처리 검사 통과와 전체 학습 성공은 별개의 확인 항목이다. 로컬 실패 시도의 체크포인트·results.csv·설정·오류 로그를 보존했다. [3]')
p(c,'Colab에서는 고정 분할과 라벨의 무결성을 확인하고 할당 GPU·VRAM을 기록한 뒤 기본 학습을 수행할 예정이다. 현재는 런타임 연결과 환경 설치 확인이 남아 있으며 학습을 시작하지 않았다. 이 보고서 작성 과정에서는 학습을 재개하지 않았다. [4]')
sub(c,'현재 성능 수치의 사용 범위')
p(c,'1 epoch 시점의 검증 mAP50은 박스 0.6173, 마스크 0.5652였다. 중단된 학습의 진단 기록이며 완성 모델이나 독립 시험 성능으로 보고하지 않는다. 전체 기본 학습 완료 후 고정 검증 152장에서 정식 비교 결과를 정리할 예정이다.')

c=page(4,'정해린\n작업 내용')
head(c,'4 익음도 기준 모델과 전이학습 비교')
p(c,'기존 색 기반 모델은 사과 박스 중심부의 빨강·초록 차이를 임계값으로 분류했다. 추가 작업에서는 ImageNet 사전학습 ResNet18 본체를 고정하고 512차원 특징을 추출한 뒤 익음도 선형 분류층을 학습했다. 원본만 학습한 모델과 원본·합성 가림을 함께 학습한 모델을 비교했다. [5][6]')
table(c,[['구분','학습','검증','평가'],['관측 영역 수','895','338','202'],['서로 다른 사과 ID','271','134','115'],['평가 라벨 구성','학습 입력 기준','임계값 선택','익음 20 / 미숙 182']],[3.4,3.25,3.25,3.35])
p(c,'같은 사과 ID와 원본 촬영 프레임의 분할 간 중복을 검사했다. 평가 202영역에는 같은 과실의 다른 관측이 포함된다. 익음 20건은 13개 과실 ID에서 나온 결과이므로 20개의 독립 과실로 해석하지 않는다.')
table(c,[['추가 가림','색 기준','ResNet 원본 학습','ResNet 가림 증강'],['0%','17 / 20','14 / 20','19 / 20'],['15%','8 / 20','17 / 20','19 / 20'],['30%','0 / 20','16 / 20','18 / 20'],['50%','0 / 20','16 / 20','18 / 20']],[2.2,3.0,4.0,4.05])
p(c,'표의 값은 익음 20건 중 찾아낸 수다. 증강 모델은 가림 50%에서 익음 재현율 90%를 유지했다. 반면 원본에서 익음으로 선택한 54건 중 실제 익음은 19건으로 정밀도는 35.2%다. 색 기준의 원본 정밀도 36.2%보다 높아지지 않았으므로, 익음 누락 감소와 오선택 문제를 함께 보고한다.',bold=True)
p(c,'기존 색 기준은 학습 자료에서 임계값을 선택했고 ResNet 분류층은 원본 검증 자료에서 선택했다. 비교에는 이 절차 차이가 포함된다. 현재 결과는 정답 사과 영역을 사용한 개별 기능 평가다.')

c=page(5,'정해린\n작업 내용')
head(c,'5 무게 추정과 촬영 조건의 한계')
p(c,'기존 면적 기반 선형 회귀와 ResNet18 특징 기반 회귀를 비교했다. 학습 과실 48개, 평가 과실 12개를 사용했으며 가림 증강은 같은 학습 과실에서 만든 192개 입력을 사용했다. 무게 모델은 개별 검출 크롭이 아니라 실험실 촬영의 전체 프레임을 입력으로 사용한다. [5][6]')
table(c,[['추가 가림','면적 기준 MAE','ResNet 원본 학습','ResNet 가림 증강'],['0%','36.52g','14.26g','13.41g'],['15%','39.15g','30.30g','14.52g'],['30%','42.15g','38.26g','16.17g'],['50%','48.66g','49.96g','18.48g']],[2.2,3.3,3.85,3.9])
p(c,'MAE는 실제 무게와 예측 무게의 절대 차이를 평균한 값이며 작을수록 좋다. 가림 증강 모델은 추가 가림 50%에서도 원본만 학습한 모델보다 오차가 작았다. 같은 과실의 원본 대비 50% 가림 예측 변화량도 평균 43.21g에서 14.37g으로 줄었다.')
pic(c,Path('haerin/2026-11-16/전이학습/figures/무게_가림별_오차.png'),12.8,'그림 2  같은 평가 과실에서 추가 합성 가림별 무게 MAE 비교  출처 정해린 전이학습 결과')
p(c,'자료에서 카메라별 촬영 과실의 무게 범위가 겹치지 않는 편향이 있다. 카메라별 평균 무게만 사용하는 기준도 MAE 12.92g이므로, 증강 모델의 원본 13.41g을 새로운 카메라·과수원에서의 정확도로 일반화할 수 없다. 현장 적용에는 촬영 거리·배율 보정과 별도 검증이 필요하다.',bold=True)

c=page(6,'정해린\n작업 내용')
head(c,'6 자연 가림 분석과 합성 잎 실험')
p(c,'자연 가림은 데이터가 제공하는 보이는 영역과 추정 전체 윤곽의 면적 차이로 분석했다. 합성 가림은 원래 보이던 사과 픽셀 중 15·30·50%를 녹색 잎 도형으로 추가 덮어 구성했다. 원본도 자연적으로 가려져 있을 수 있어 추가 가림 0%가 전체 가림 0%를 뜻하지는 않는다. [5][6]')
pic(c,Path('haerin/2026-11-16/결과사진/04_자연가림과_합성잎.png'),12.45,'그림 3  자연 가림의 가시·전체 윤곽과 추가 합성 잎 예시  출처 정해린 실험 결과 및 원자료 표기')
sub(c,'확인한 결과와 남은 범위')
p(c,'색 기준 모델은 녹색 잎의 비중이 늘면 익음 누락이 증가했다. 전이학습과 가림 증강은 이 합성 조건에서 익음 재현율과 무게 예측 안정성을 개선했다. 다만 실제 잎 모양·조명 변화·연속 프레임의 움직임에 대한 성능까지 검증한 것은 아니다.')
p(c,'잎 이동량에 따른 움직임 실험과 판단 보류 기준은 후속 작업이다. 합성한 잎의 픽셀 이동량은 실제 풍속이나 풍량으로 해석하지 않는다. 실제 검출기의 누락·박스 오차까지 포함한 가림 평가도 아직 남아 있다.')

c=page(7,'공통\n통합 현황')
head(c,'7 기능 통합의 현재 상태와 연결 방안')
p(c,'현재 검출·분할과 익음도·무게 분석은 각각 실행한 상태이며 전체 처리 흐름을 연결하지 않았다. 전체 학습과 별개로 정답 영역 또는 저장된 예측을 이용해 임시 연결을 진행하는 방안을 검토했다. 이 방안은 통합 관리 대화의 제안 단계이며 구현 완료로 집계하지 않는다. [7]')
table(c,[['공통 전달 항목','목적 및 합의할 내용'],['이미지 ID / 객체 번호','동일 사진과 동일 사과의 결과 연결'],['원본 이미지 / 박스 / 마스크','분류용 사과 영역 추출, 좌표·크기 규칙 통일'],['입력 출처 / 검출 신뢰도','정답 영역과 모델 예측 구분, 신뢰도 기록'],['익음도 점수 / 예상 무게 / 상태','분류 임계값과 판단 보류, 무게 적용 조건 표시']],[4.3,8.95])
sub(c,'독립 평가와 임시 연결을 병행하는 순서')
p(c,'① 각 기능의 기존 평가 자료와 지표를 정리한다. ② 정답 박스·마스크로 이미지 입력부터 객체별 결과 출력까지 임시 연결한다. ③ 검출·분할 기본 학습 완료 후 저장된 YOLO 예측으로 입력을 교체한다. ④ 검출 누락과 잘못된 영역까지 포함해 통합 평가한다.')
table(c,[['핵심 제약','현재 확인한 내용','다음 조치'],['무게 입력 조건','전체 실험실 프레임 학습','사과 크롭에 바로 적용하지 않고 촬영 조건·보정 검토'],['모델 파일','익음도 head와 ResNet 가중치 일부 미포함','팀원에게 파일 확보 또는 재학습'],['평가 기준','정답 영역 기반 개별 평가','검출 누락을 포함한 통합 평가 별도 수행'],['깊이·실제 지름','자료·피드백 검토 단계','보정 정보와 구현 범위·담당 추가 조율']],[2.6,4.55,6.1],8.7)
p(c,'새 사진 추론·오류 분석은 후속 통합 검증으로 미루는 방안을 검토하되, 기존 고정 검증 자료에서의 기능별 평가는 유지한다. 프로젝트 전체 완료율은 작업 비중과 완료 기준을 확정하기 전 임의의 퍼센트로 표시하지 않는다.')

c=page(8,'다음\n미팅에서\n해야 할 일')
head(c,'8 다음 활동과 역할 분담')
table(c,[['활동 내용','역할 분담','추진 순서 및 완료 기준'],['Colab 환경 점검·기본 학습','윤종호','GPU·설치·무결성 확인 후 학습 완료와 모델 저장 확인'],['익음도 오선택 분석','정해린','정밀도·재현율·혼동행렬로 오류 유형 정리'],['무게 조건·모델 파일 점검','정해린','입력 촬영 조건, 파일 확보와 재현 실행 확인'],['입력·출력 규격과 임시 연결','공동','정답 영역에서 객체별 결과 출력까지 실행'],['예측 영역 교체·통합 평가','공동','검출 누락을 포함한 평가와 새 사진 오류 분석'],['움직임·시연·최종 문서','공동','합성 조건 정의, 시연 영상과 최종 결과 정리']],[5.25,2.0,6.0],8.7)
p(c,'기존 일정은 11월 말 기본 통합, 12월 평가·시연, 12월 14~15일 최종 제출을 목표로 한다. 전체 학습과 촬영 조건 보정의 진행 상황에 따라 조정할 수 있다. 새 성능 목표 수치와 세부 일정은 다음 활동에서 확정한다.')
head(c,'9 근거 자료')
sources=[
'[1] splits/minneapple_v1/summary.json 및 데이터설정_실행방법.md  데이터 수·분할·검증 용도',
'[2] 짧은학습_실행결과.md  소량 학습 결과와 작은 마스크 검사',
'[3] outputs/training/baseline_20261116T123500109975Z/baseline_summary.json  전체 학습 실패와 1 epoch 진단 지표',
'[4] colab/실행방법.md 및 윤종호 작업 대화  Colab 준비·업로드·학습 미시작',
'[5] haerin/2026-11-16/실험기록.md 및 docs/commit-1fdae1c-analysis-ko.md  기준 실험·카메라 편향',
'[6] haerin/2026-11-16/전이학습/README.md 및 ripeness/results/metrics.json, weight/results/weight_transfer_metrics.json  전이학습 결과',
'[7] 프로젝트 진행률 통합 관리 대화 및 각 담당 작업 대화  통합 제안과 최신 상태',
'양식 원본  3조 팀활동보고서_수정본.hwp  미팅 개요·진행 사항·추후 계획의 표 구조와 ERICA 로고를 반영',
]
for s in sources:
 pp=p(c,s);pp.paragraph_format.space_after=Pt(4)
 for r in pp.runs:r.font.size=Pt(8)
p(c,'기준 시점 이후의 작업은 포함하지 않았다. 이 보고서의 일시는 현황 정리 기준이며 별도의 회의 개최 시각을 뜻하지 않는다.')

footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=footer.add_run('Team.D_24  ·  ');r.font.size=Pt(8)
fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');footer._p.append(fld)
DOC.core_properties.title='사과 수확 로봇 비전 프로젝트 팀 활동 보고서'
DOC.core_properties.author='Team.D_24';DOC.core_properties.subject='2026년 11월 16일 기준 활동 정리'
OUT.mkdir(parents=True,exist_ok=True);target=OUT/'3조_팀활동보고서_2026-11-16.docx';DOC.save(target)
(BUILD/'artifact.md').write_text('''# 보고서 양식 재현 기준\n원본 HWP는 수정하지 않음. HWP BodyText와 PrvImage 및 내장 ERICA 로고를 읽고 DOCX로 재구성. A4 세로, 본문 폭 17.9cm. 원본의 미팅 개요·진행 사항·추후 계획 구조, 회색 구분 셀, 검정 표 테두리와 ERICA 로고 유지. 구체적 폰트 계열과 줄바꿈은 편집 가능한 DOCX에 맞게 맑은 고딕 10pt로 설정. 일시는 실제 회의 개최일을 만들지 않고 현황 정리 기준일로 기입. 총 8개 구성 페이지. 원본의 계획과 시연 예시는 현재 완료 결과로 전용하지 않음.\n''',encoding='utf-8')
print(target)
