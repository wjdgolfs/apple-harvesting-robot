from pathlib import Path
import struct,zlib,json,runpy,io,contextlib,sys,math,copy
from PIL import Image,ImageOps
sys.stdout.reconfigure(encoding='utf-8')
B=Path(__file__).parent;ROOT=B.parents[2];OUT=B.parent/'output'
with contextlib.redirect_stdout(io.StringIO()):m=runpy.run_path(str(B/'inspect_hwp.py'))
records=json.loads((B/'hwp-records.json').read_text(encoding='utf-8'))
R=[dict(tag=r['tag'],level=r['level'],data=bytes.fromhex(r['hex'])) for r in records]
# Update text in the source's existing cells and paragraphs; retain all tables,
# page definitions, character/paragraph styles, ERICA logo and drawing anchors.
M={
46:'2026.11.16 기준 현황 정리',70:'검출·분할 데이터 준비 및 학습 실행 점검',74:'익음도·무게 전이학습과 합성 가림 평가',78:'개별 기능 평가 및 임시 통합 방향 정리',
115:'1. 담당별 활동 요약과 현재 상태',126:'담당·영역',131:'완료한 작업',136:'현재 상태와 후속 과제',
141:'윤종호',146:'MinneApple 670쌍 검수, 분할·YOLO 라벨 변환',151:'짧은 학습 완료. 전체 학습은 메모리 부족으로 중단',
156:'정해린 익음도',161:'색 기준과 ResNet18 분류층 학습·가림 비교',166:'가림에서 익음 누락 감소. 오선택 개선 필요',
171:'정해린 무게',176:'면적 회귀와 ResNet18 회귀의 조건별 비교',181:'가림 증강 효과 확인. 촬영 편향 검토 필요',
186:'정해린 가림',191:'자연 가림 분석 및 합성 잎 15·30·50% 평가',196:'정적 합성 평가 완료. 움직임 실험은 미진행',
201:'공통 통합',206:'결과 취합과 입력·출력 연결 방식 검토',211:'정답 영역 임시 연결 제안. 통합 구현은 미완료',
215:'현재는 데이터 준비와 개별 기능의 예비 평가를 수행한 단계이다. 전체 검출·분할 학습, 실제 예측 영역을 사용하는 통합 평가, 현장 시연은 아직 완료하지 않았다. 본 보고서는 별도 회의 개최 시각이 아닌 2026년 11월 16일의 작업 기록을 기준으로 작성하였다.',
222:'2. 검출·분할 데이터 검수와 가림 실험 예시',
236:'그림 1. MinneApple 이미지·객체 마스크 검수 예시 · 출처: 윤종호 검수 결과',
240:'그림 2. 같은 사과에 추가 가림을 적용한 익음도 비교 · 출처: 정해린 실험 결과',
244:'MinneApple은 전경 나무의 사과를 중심으로 주석한다. 배경의 모든 사과를 누락 정답으로 해석하지 않는다. 가림 실험은 원래 보이던 사과 픽셀에 합성 잎을 추가한 조건이다.',
251:'3. 윤종호 검출·분할 작업',255:'가. 데이터 검사와 고정 분할',
259:'이미지·마스크 670쌍의 대응·읽기·크기·형식을 검사하였다. 촬영 시퀀스 단위로 학습 518장과 검증 152장을 분리하였다. 주석은 학습 22,082개, 검증 6,101개로 총 28,183개다. 검증은 모델 선택용이며 독립 최종 시험 자료와 구분한다. [1]',
270:'나. 마스크 변환과 입력 검증',
274:'객체 ID별 마스크를 한 사과당 한 줄의 YOLO 분할 다각형으로 변환하였다. 떨어진 조각·내부 구멍·작은 윤곽을 처리하고 원본 ID 대응표를 저장하였다. 제외 객체는 0개이며 670개 라벨 파일과 데이터 YAML의 실제 로딩을 확인하였다.',
285:'다. 짧은 학습과 전체 학습 중단',
289:'32장 학습·8장 검증으로 3 epoch를 정상 실행하고 가중치 저장·재로딩을 확인하였다. 전체 학습은 batch 2에서 첫 epoch 도중, batch 1에서는 1 epoch 완료 후 2 epoch 도중 CUDA 메모리 부족으로 종료되었다. 전체 기본 학습은 미완료다. [2][3]',
300:'라. 작은 객체 보존과 Colab 전환',
304:'1,797개 객체의 전처리에서 mask_ratio=4는 빈 마스크 2개를 만들었고, mask_ratio=1은 0개였다. 전체 28,183개에서도 mask_ratio=1 전처리를 통과했지만 학습 메모리가 부족했다. Colab 노트북 저장과 데이터 세 파일 업로드를 완료했으며 GPU·환경 확인과 학습 시작은 남아 있다. [4]',
311:'4. 정해린 전이학습 방법과 평가 자료',
315:'ImageNet 사전학습 ResNet18 본체를 고정하고 512차원 이미지 특징을 추출한 뒤, 익음도 분류층과 무게 회귀층을 각각 학습하였다. 원본 학습과 원본·합성 가림 증강 학습을 같은 평가 대상으로 비교하였다. ResNet18 전체를 미세조정한 실험은 아니다. [5][6]',
322:'5. 기능별 구현과 확인한 내용',326:'기준 모델과 전이학습 결과를 함께 정리하고, 실제 검출 결과로 교체할 입력 규격과 적용 조건을 검토하였다.',
337:'기능',342:'적용 방법',347:'실행 결과 및 해석',
352:'검출·분할',357:'YOLOv8n-seg',362:'라벨 670개·주석 28,183개 준비, 짧은 학습 완료',
367:'익음도 기준',372:'색 차이 임계값',377:'원본 익음 17/20 검출, 익음 정밀도 36.2%',
382:'익음도 전이',387:'고정 ResNet18 + 선형 분류',392:'학습 895영역, 검증 338영역, 평가 202영역',
397:'무게 기준',402:'면적 기반 선형 회귀',407:'평가 과실 12개, 원본 MAE 36.52g',
412:'무게 전이',417:'고정 ResNet18 + 회귀',422:'학습 과실 48개, 평가 과실 12개, 전체 프레임 입력',
427:'가림 증강',432:'합성 잎 0·15·30·50%',437:'같은 학습 사과에 잎을 추가한다. 새 과실을 확보한 것이 아니다.',
442:'중복 점검',447:'사과 ID·원본 프레임 검사',452:'학습·검증·평가에 동일 ID나 원본 프레임이 겹치는지 검사',
457:'평가 지표',462:'정밀도·재현율·MAE',467:'익음 누락과 오선택을 구분하고 무게 오차는 g 단위로 기록',
472:'통합 입력',477:'이미지·객체 ID·박스·마스크',482:'정답 영역과 예측 영역의 출처 구분. 전체 연결은 미구현',
486:'익음도 증강 모델의 조건별 결과 요약',493:'추가 가림',498:'익음 검출',503:'미숙 오선택',508:'정밀도',513:'해석',
518:'0%',523:'19/20',528:'35/182',533:'35.2%',538:'누락 감소, 오선택 남음',
543:'30%',548:'18/20',553:'45/182',558:'28.6%',563:'재현율 90%',
568:'50%',573:'18/20',578:'41/182',583:'30.5%',588:'재현율 90%',
592:'익음 평가 202영역은 익음 20·미숙 182건이며 115개 사과 ID에서 나왔다. 익음 20건은 13개 ID의 관측으로, 20개의 독립 과실은 아니다. 원본 정밀도는 색 기준 36.2%보다 높아지지 않았으므로 수확 선별 성능이 해결되었다고 볼 수 없다. 기존 색 기준과 전이 모델은 임계값 선택 자료도 다르다.',
599:'6. 사용 데이터와 용도',610:'자료',615:'이번 활동에서 사용한 정보',620:'해석과 후속 적용 범위',
625:'MinneApple [1]',630:'이미지·개별 사과 마스크 670쌍',635:'검출·분할 학습과 고정 검증 152장',
640:'Fuji-Ripeness-Size [5]',645:'사과 영역과 익음·미숙 라벨',650:'정답 영역 기반 익음도 비교 평가',
655:'AmodalAppleSize [5]',660:'보이는 영역과 추정 전체 윤곽',665:'자연 가림 분석과 합성 잎 추가',
670:'IntegrALIMENTA [5]',675:'실험실 과실 사진과 실측 무게',680:'학습 48·평가 12과실의 무게 MAE',
685:'실행 로그·예측표 [2][6]',690:'학습 설정, 예측 CSV·JSON, 결과 그림',695:'수치 검산과 반복 실행 근거 보존',
699:'검출 데이터와 익음도·무게 자료는 동일한 표본이 아니다. 무게는 실험실 전체 프레임 입력으로 학습했으므로 과수원에서 검출한 사과 크롭에 바로 적용하지 않는다. 카메라별 무게 범위 편향과 촬영 거리·배율 영향을 별도로 검토해야 한다.',
706:'7. 무게 추정 결과와 가림의 영향',
710:'추가 가림 0·15·30·50%에서 면적 기준 MAE는 36.52·39.15·42.15·48.66g, ResNet 원본 학습은 14.26·30.30·38.26·49.96g, 가림 증강은 13.41·14.52·16.17·18.48g이었다. 같은 평가 과실 12개를 조건별로 비교하였다. [6]',
714:'원본 대비 50% 가림의 예상 무게 변화량은 원본 학습 43.21g에서 증강 학습 14.37g으로 줄었다. 다만 카메라별 평균 무게만 사용한 기준도 MAE 12.92g이었다. 증강 모델의 13.41g을 새 카메라·과수원에서의 정확도로 일반화하지 않는다.',
721:'8. 통합 현황과 임시 연결 방안',
725:'개별 평가와 연결 작업을 병행하는 방안을 검토하였다. 먼저 정답 영역으로 객체별 출력 흐름을 확인하고 학습 완료 후 YOLO 예측으로 교체한다. 이는 제안 단계이며 아직 실행된 통합 결과가 아니다. [7]',
736:'단계',741:'다음 활동',746:'완료 판단 기준',751:'독립 평가',756:'고정된 자료로 각 기능 평가',761:'검출·분할 지표, 정밀도·재현율·MAE 확보',
766:'임시 연결',771:'정답 박스·마스크로 연결',776:'이미지 입력부터 객체별 분석 출력까지 실행',
781:'예측 교체',786:'YOLO 박스·마스크를 입력',791:'누락·박스 오차까지 포함한 통합 평가',
795:'공통 규격에는 이미지 ID, 객체 번호, 원본 이미지, 박스, 마스크, 입력 출처, 검출 신뢰도를 포함한다. 익음도는 사과 영역을 사용하지만 무게는 전체 프레임 학습 조건과 차이가 있어 별도 보정이 필요하다. 누락된 익음도 head와 ResNet 가중치의 확보 또는 재학습도 남아 있다.',
802:'9. 한계와 다음 평가 항목',809:'항목',814:'현재 결과의 범위와 남은 확인',
819:'검출·분할',824:'전체 학습 미완료. 1 epoch 진단 지표를 최종 성능으로 보고하지 않음',
829:'익음도',834:'가림에서 재현율은 개선했으나 원본 정밀도 35.2%로 오선택이 많음',
839:'무게',844:'평가 12과실, 카메라별 무게 분포 편향. 촬영 조건 외삽 검증 필요',
849:'가림',854:'추가 합성 가림 조건의 결과. 실제 잎·조명·검출 오차까지 일반화하지 않음',
859:'움직임·깊이',864:'연속 잎 움직임과 깊이 기반 실제 지름은 미구현. 범위·담당 추가 조율',
869:'통합·속도',874:'실제 검출을 포함한 전체 평가·시연·FPS 측정은 후속 수행',
878:'합성 가림률은 원래 보이던 사과 픽셀 중 추가로 덮은 비율이다. 자연 가림과 합성 가림을 구분하며, 정적 잎 가림 평가를 움직임·실제 풍속 실험으로 표현하지 않는다.',
882:'검출 기본 학습과 개별 기능 평가를 계속하면서 입력 규격·임시 연결을 진행한다. 새 사진 오류 분석은 후속 통합 검증으로 미루는 방안을 검토하되 고정 검증 자료에서의 평가는 유지한다.',
889:'10. 추진 일정',893:'기존 목표는 11월 말 기본 기능 통합, 12월 평가·시연, 12월 14~15일 최종 제출이다. 다음 순서는 Colab 환경 확인·기본 학습, 기능별 평가, 정답 영역 임시 연결, YOLO 예측 교체·통합 평가, 시연 준비로 정리하였다.',
897:'무게 입력 조건 보정과 학습 진행에 따라 세부 일정은 조정할 수 있다. 전체 완료율은 작업 비중과 완료 기준을 확정하기 전 임의의 퍼센트로 표시하지 않는다.',
904:'11. 근거 자료',908:'[1] 데이터 분할과 라벨 준비',912:'splits/minneapple_v1/summary.json 및 데이터설정_실행방법.md',916:'MinneApple 원본과 고정 분할의 이미지 수·주석 수·사용 범위',
920:'[2] 소량 학습 동작 확인',924:'짧은학습_실행결과.md',928:'32장 학습·8장 검증, 3 epoch와 작은 마스크 축소 검사',
932:'[3] 전체 학습 중단 기록',936:'outputs/training/baseline_20261116T123500109975Z/baseline_summary.json',940:'1 epoch 완료, CUDA 메모리 부족 종료, 체크포인트·로그 보존',
944:'[4] Colab 전환 준비',948:'colab/실행방법.md 및 윤종호 작업 대화의 마지막 확인 상태',952:'노트북·데이터 업로드 완료, GPU·환경 확인 미완료, 학습 미시작',
956:'[5] 기준 모델과 데이터 분석',960:'haerin/2026-11-16/실험기록.md',964:'docs/commit-1fdae1c-analysis-ko.md',
968:'[6] ResNet18 전이학습과 가림 비교',972:'haerin/2026-11-16/전이학습/README.md',976:'익음도 예측·지표: ripeness/results/predictions.csv 및 metrics.json',980:'무게 예측·지표: weight/results/weight_transfer_predictions.csv 및 weight_transfer_metrics.json',
984:'[7] 통합 방향과 최신 작업 대화',988:'프로젝트 진행률 통합 관리, 윤종호 작업 내용, 정해린 작업 내용',992:'각 기능 독립 평가와 정답 영역 임시 연결 방안',996:'정답 영역 사용 결과와 실제 검출을 포함한 결과를 구분',
1000:'[8] 그림과 데이터 출처',1004:'MinneApple 및 정해린 저장 결과 그림의 원자료·라이선스 표기를 보존',
1008:'[9] 보고서 양식',1012:'3조 팀활동보고서_수정본.hwp',1016:'원본의 미팅 개요·진행 사항·추후 계획, 표 구조와 ERICA 로고 유지',
1020:'[10] 작성 기준',1024:'2026년 11월 16일 확인한 실행 기록과 저장된 평가 결과 기준',1028:'예정·제안·미완료 작업을 완료한 성과와 구분하여 기록',
1075:'기본 학습과 임시 기능 통합',1079:'Colab GPU·환경 확인 후 기본 학습 완료. 익음도 오선택과 무게 촬영 조건을 점검하고 정답 영역에서 임시 연결을 시작한다.',
1090:'검출·분할 및 Colab 학습 - 윤종호',1094:'익음도·무게·가림 및 모델 파일 점검 - 정해린',1098:'입력·출력 규격, 임시 연결, 통합 평가 - 공동',1102:'활동 보고서·발표 자료 검토 - 공동',1114:'기존 일정에 따라 11월 말 통합, 12월 평가·시연을 목표로 하되 학습과 보정 진행에 따라 조정한다. 판단 보류 기준, 움직임 실험과 깊이·실제 지름의 구현 범위는 추가 협의한다.'}

def set_text(idx,text):
 old=R[idx]['data'];prefix=b''
 # Inline controls can follow leading spaces or ordinary caption text.
 # Preserve every complete control, not only a control at byte offset zero.
 vals=struct.unpack('<%dH'%(len(old)//2),old);q=0
 while q<len(vals):
  if vals[q] in (1,2,3,11,12,14,15,16,17,18,21,22,23):
   prefix+=old[q*2:(q+8)*2];q+=8
  elif vals[q]==9:q+=8
  else:q+=1
 data=prefix+(text+'\r').encode('utf-16le');R[idx]['data']=data
 ph=idx-1
 while ph>=0 and R[ph]['tag']!=66:ph-=1
 h=bytearray(R[ph]['data']);orig=struct.unpack_from('<I',h)[0]
 struct.pack_into('<I',h,0,(orig&0x80000000)|(len(data)//2))
 # Preserve a valid layout-cache record. A zero-line paragraph is not a
 # supported substitute for an invalidated cache in Hancom's native reader.
 struct.pack_into('<HHH',h,12,1,0,1);R[ph]['data']=bytes(h)
 j=idx+1
 while j<len(R) and R[j]['level']>R[ph]['level']:
  if R[j]['level']==R[ph]['level']+1:
   if R[j]['tag']==68:R[j]['data']=R[j]['data'][:8]
   if R[j]['tag']==69:
    R[j]['data']=struct.pack('<I',0)+R[j]['data'][4:36]
   if R[j]['tag']==70:R[j]['skip']=True
  j+=1
for i,t in M.items():set_text(i,t)
# Remove the old schedule screenshot while retaining its now-empty paragraph.
set_text(1107,'')
R[1107]['data']='\r'.encode('utf-16le')
for i in range(1110,1113):R[i]['skip']=True
# Identify exact drawing descendants for the removed inline image.
for i in range(1108,1113):
 if R[i]['tag'] not in (66,67,68,69):R[i]['skip']=True
h=bytearray(R[1106]['data']);struct.pack_into('<I',h,0,1);struct.pack_into('<I',h,4,0);R[1106]['data']=bytes(h)
def pack_rec(r):
 d=r['data'];n=len(d);h=r['tag']|(r['level']<<10)|(min(n,4095)<<20)
 return struct.pack('<I',h)+(struct.pack('<I',n) if n>=4095 else b'')+d
section=b''.join(pack_rec(r) for r in R if not r.get('skip'))
def deflate(data):
 z=zlib.compressobj(9,zlib.DEFLATED,-15);return z.compress(data)+z.flush()
mods={'Section0':deflate(section)}
# Picture formats and original frames are retained. Pad evidence to the source
# image aspect ratio, with no cropping or invented data.
assets={
'BIN0002.bmp':'outputs/sample_inspection/20150919_174151_image1/comparison.png',
'BIN0003.bmp':'haerin/2026-11-16/전이학습/figures/가림_익음도_전이학습.png',
'BIN0004.bmp':'outputs/training/smoke_20261116T120517979228Z/results.png',
'BIN0005.bmp':'haerin/2026-11-16/전이학습/figures/익음도_가림별_성능.png',
'BIN0006.jpg':'haerin/2026-11-16/전이학습/figures/무게_가림별_오차.png'}
for name,asset in assets.items():
 original=Image.open(B/name);im=Image.open(ROOT/asset).convert('RGB')
 im=ImageOps.pad(im,original.size,color='white',method=Image.Resampling.LANCZOS)
 f=io.BytesIO();im.save(f,format='JPEG' if name.endswith('jpg') else 'BMP',quality=95)
 # The source DocInfo marks JPEG BinData 6 as explicitly uncompressed.
 # Its bytes must match that storage flag, independently of FileHeader flags.
 mods[name]=f.getvalue() if name.endswith('.jpg') else deflate(f.getvalue())
preview=('팀 활동 보고서 Team.D_24\r2026.11.16 기준\r윤종호, 정해린\r'+''.join(t+'\r' for i,t in M.items() if i>=115)).encode('utf-16le')[:2046]
mods['PrvText']=preview
# Reuse the CFB sector inventory. Rewrite streams within their allocation,
# appending sectors only when a replacement grows. Preserve directory tree.
data=bytearray(m['b']);ss=m['ss'];fat=list(m['fat']);dirchain=[];s=m['u32'](48)
while s<0xfffffffa:dirchain.append(s);s=fat[s]
directory=bytearray(m['dirs'])
def ids(start):
 out=[]
 while start<0xfffffffa:out.append(start);start=fat[start]
 return out
def write_sector(i,chunk):
 end=(i+2)*ss
 if end>len(data):data.extend(b'\0'*(end-len(data)))
 data[(i+1)*ss:end]=chunk.ljust(ss,b'\0')
for ordinal in range(0,len(directory),128):
 d=directory[ordinal:ordinal+128];n=struct.unpack_from('<H',d,64)[0]
 if n<2:continue
 name=d[:n-2].decode('utf-16le')
 if name not in mods:continue
 entry=next(e for e in m['entries'] if e['name']==name);payload=mods[name]
 if entry['size']<4096:
  if len(payload)>entry['size']:raise ValueError('Mini stream growth')
  minis=ids(entry['start']) if False else []
  mini_idx=entry['start'];offset=0;rootsecs=ids(next(e for e in m['entries'] if e['type']==5)['start'])
  while mini_idx<0xfffffffa and offset<len(payload):
   logical=mini_idx*m['ms'];sect=rootsecs[logical//ss];pos=(sect+1)*ss+logical%ss
   ch=payload[offset:offset+m['ms']];data[pos:pos+len(ch)]=ch;offset+=m['ms'];mini_idx=m['mt'][mini_idx]
 else:
  chain=ids(entry['start']);need=math.ceil(len(payload)/ss)
  while len(chain)<need:
   new=len(data)//ss-1
   if new>=len(fat):raise ValueError('FAT capacity exceeded')
   data.extend(b'\0'*ss);fat[chain[-1]]=new;fat[new]=0xfffffffe;chain.append(new)
  if need<len(chain):
   # CFB chain length must agree with the new stream's declared size.
   fat[chain[need-1]]=0xfffffffe
   for orphan in chain[need:]:fat[orphan]=0xffffffff
   chain=chain[:need]
  for k,sec in enumerate(chain):write_sector(sec,payload[k*ss:(k+1)*ss])
 struct.pack_into('<Q',directory,ordinal+120,len(payload))
for k,sec in enumerate(dirchain):write_sector(sec,directory[k*ss:(k+1)*ss])
fatids=[i for i in m['fatids'] if i<0xfffffffa]
for k,sec in enumerate(fatids):write_sector(sec,struct.pack('<%dI'%(ss//4),*fat[k*(ss//4):(k+1)*(ss//4)]))
OUT.mkdir(exist_ok=True);dest=OUT/'3조_팀활동보고서_2026-11-16_복구본.hwp';dest.write_bytes(data)
(B/'hwp-updated-text.txt').write_text('\n'.join(M.values()),encoding='utf-8')
print(json.dumps({'path':str(dest),'bytes':len(data),'updated_paragraphs':len(M),'section_raw':len(section),'section_compressed':len(mods['Section0'])},ensure_ascii=False))
