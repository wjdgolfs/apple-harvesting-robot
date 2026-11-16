import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const BUILD=path.dirname(fileURLToPath(import.meta.url));
const ROOT=path.resolve(BUILD,'../../..');
const WORK=path.dirname(BUILD), OUT=path.join(WORK,'output');
const SKILL='C:/Users/YJH/.codex/plugins/cache/openai-primary-runtime/presentations/26.1112.11217/skills/presentations';
const PY='C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES='C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt/dependencies/node/node_modules';
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const P=Presentation.create({slideSize:{width:1280,height:720}});
const FONT='Malgun Gothic';
const C={ink:'#17352D',green:'#16654C',muted:'#5D6B66',red:'#A94A32',paper:'#FAFAF7',gray:'#DDE3DF',blue:'#456C9C'};
const preview=path.join(BUILD,'slides');await fs.mkdir(preview,{recursive:true});await fs.mkdir(OUT,{recursive:true});
let n=0;const timings=[25,40,45,45,40,50,45,50,65,60,45,65,55,45,45,65,60,55];
function text(s,str,x,y,w,h,size=26,color=C.ink,bold=false){
 const t=s.shapes.add({geometry:'textbox',name:str.slice(0,35),position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 t.text=str;t.text.style={fontSize:size,fontFamily:FONT,typeface:FONT,color,bold,verticalAlignment:'top',alignment:'left',insets:0,wrap:'word'};return t;
}
function slide(title,notes,sources=[]){
 const s=P.slides.add();n++;s.background.fill=C.paper;
 text(s,title,64,45,1152,95,39,C.ink,true);
 text(s,`${String(n).padStart(2,'0')} / 18`,1140,677,90,22,15,C.muted);
 s.speakerNotes.text=`발표 시간 약 ${timings[n-1]}초\n\n${notes}\n\n근거 자료\n${sources.join('\n')}\n작성 기준 2026년 11월 16일. 예정·제안·미완료 작업은 수행 완료 결과와 구분한다.`;
 return s;
}
function line(s,heading,body,y=185,w=1100){text(s,heading,68,y,w,45,29,C.green,true);text(s,body,68,y+58,w,92,25);}
function tab(s,rows,widths,x=64,y=170,w=1152,h=350,size=23){
 const t=s.tables.add({rows:rows.length,columns:rows[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values:rows});
 t.borders.assign({style:'solid',fill:C.gray,width:1});
 t.cells.block({row:0,column:0,rowCount:rows.length,columnCount:rows[0].length}).assign({textStyle:{fontFamily:FONT,typeface:FONT,fontSize:size,color:C.ink},margins:{left:14,right:14,top:10,bottom:10},anchor:'center'});
 for(let r=0;r<rows.length;r++){
  t.rows[r].height=h/rows.length;
  for(let c=0;c<rows[0].length;c++){
   t.getCell(r,c).fill=r===0?'#E2EBE5':C.paper;
   t.getCell(r,c).text.style={fontFamily:FONT,typeface:FONT,fontSize:size,color:C.ink,bold:r===0};
  }
 }
 return t;
}
async function image(s,rel,x,y,w,h){
 const bytes=await fs.readFile(path.resolve(ROOT,rel));
 s.images.add({blob:bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),contentType:rel.toLowerCase().endsWith('jpg')?'image/jpeg':'image/png',alt:rel,fit:'contain',position:{left:x,top:y,width:w,height:h}});
}
function chart(s,series,yTitle,max,step){
 const cfg={position:{left:60,top:160,width:865,height:430},categories:['0%','15%','30%','50%'],series:series.map(([name,values,color])=>({name,values,line:{style:'solid',fill:color,width:3},marker:{symbol:'circle',size:8,fill:color}})),hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:{typeface:FONT,fontSize:19,fill:C.ink}},chartFill:C.paper,plotAreaFill:C.paper,chartLine:{fill:'none',width:0},xAxis:{title:{text:'추가 합성 가림률',textStyle:{typeface:FONT,fontSize:18}},textStyle:{typeface:FONT,fontSize:19,fill:C.muted},majorGridlines:null},yAxis:{title:{text:yTitle,textStyle:{typeface:FONT,fontSize:18}},min:0,max,majorUnit:step,numberFormatCode:yTitle.includes('%')?'0"%"':'0',textStyle:{typeface:FONT,fontSize:18,fill:C.muted},majorGridlines:{fill:'#DDE3DF',width:1}},lineOptions:{smooth:false},dataLabels:{showValue:false}};
 const ch=s.charts.add('line',cfg);applyPresentationChartFont(ch,{fontFamily:FONT});return ch;
}
const src={
 split:'splits/minneapple_v1/summary.json',
 smoke:'짧은학습_실행결과.md',
 baseline:'outputs/training/baseline_20261116T123500109975Z/baseline_summary.json',
 colab:'colab/실행방법.md',
 transfer:'haerin/2026-11-16/전이학습/README.md',
 ripeness:'haerin/2026-11-16/전이학습/ripeness/results/metrics.json',
 weight:'haerin/2026-11-16/전이학습/weight/results/weight_transfer_metrics.json',
 base:'docs/commit-1fdae1c-analysis-ko.md',
 team:'codex://threads/01a1e42c-6384-7a33-9182-b9e4899ba47b',
 yoon:'codex://threads/01a1c9f2-0b8b-71f3-8271-c4f6d3b56053',
 jeong:'codex://threads/01a1e41a-8b1a-7c70-b379-e2e0dae49685',
};
{
const s=slide('사과 수확 로봇을 위한\n영상 인식 기능 개발','우리 팀은 사과 수확 로봇에 필요한 영상 인식 기능을 개발하고 있습니다. 오늘은 실제로 수행한 데이터 준비와 기능별 실험 결과, 아직 남아 있는 통합 과제를 구분해 설명하겠습니다. 하드웨어 수확 동작은 이번 개발 범위에 포함하지 않습니다.',[src.split,src.transfer,'표지 이미지: outputs/sample_inspection/20150919_174151_image1/overlay.png, MinneApple 원본 위 정답 마스크 표시, 모델 예측 아님']);
 // Remove the ordinary title and use a spacious two-column cover.
 s.shapes.items[0].delete();
 text(s,'사과 수확 로봇을 위한\n영상 인식 기능 개발',64,115,675,210,52,C.ink,true);
 text(s,'검출·분할과 익음도·무게 추정\n합성 잎 가림에 따른 성능 비교',68,365,650,100,29,C.muted);
 text(s,'Team.D_24\n윤종호 · 정해린\n2026년 11월 16일 진행 보고',68,525,620,100,23,C.ink);
 await image(s,'outputs/sample_inspection/20150919_174151_image1/overlay.png',790,0,490,650);
 text(s,'정답 마스크 검수 예시  /  MinneApple',790,654,390,23,15,C.muted);
}
{
const s=slide('프로젝트 범위와 역할 분담','최종 목표는 사진 또는 영상에서 사과를 찾고 각 사과의 상태를 표시하는 것입니다. 윤종호는 위치와 개별 영역을 찾는 검출·분할을 맡고, 정해린은 익음도·무게·가림 분석을 맡았습니다. 현재는 각 기능을 따로 평가한 단계이며 수확 로봇 하드웨어나 완성된 현장 시스템을 구현했다고 보지 않습니다.',[src.team,src.yoon,src.jeong]);
 text(s,'사진 속 사과의 위치와 상태를 분석하는 비전 기능',68,150,1120,55,30,C.green,true);
 tab(s,[['담당','주요 작업','현재 도달 단계'],['윤종호','사과 검출·개별 분할','데이터 준비와 짧은 학습'],['정해린','익음도·무게·가림 분석','기준 모델 및 전이학습 비교'],['공통','기능 연결과 전체 평가','입력 규격·임시 연결 방안 검토']],[175,475,502],64,240,1152,290,25);
 text(s,'현재 개발 범위는 영상 인식이며, 실제 로봇의 수확 동작은 포함하지 않음',68,577,1120,65,25,C.muted);
}
{
const s=slide('사용 데이터와 평가 목적','기능별로 정답 정보가 달라 서로 다른 데이터를 사용합니다. 검출은 MinneApple의 객체 마스크, 익음도는 Fuji 라벨, 가림은 보이는 영역과 전체 윤곽, 무게는 실측 무게가 있는 실험실 자료입니다. 이 자료를 하나의 통합 성능처럼 합산하지 않습니다. 무게 평가 과실은 12개로 작고 실제 과수원과 촬영 조건도 다릅니다.',[src.split,src.transfer,src.base]);
 tab(s,[['자료','사용 정보','실험 목적'],['MinneApple','670쌍 이미지·개별 마스크','검출·분할 학습과 검증'],['Fuji 익음도 자료','익음·미숙 라벨','익음도 분류 비교'],['AmodalAppleSize','가시 영역·추정 전체 윤곽','자연·합성 가림 분석'],['IntegrALIMENTA','과실 사진·실측 무게','학습 48과실, 평가 12과실']],[330,430,392],64,170,1152,365,23);
 text(s,'기능별 자료와 촬영 조건이 달라, 개별 결과와 통합 평가를 구분',68,578,1120,65,27,C.green,true);
}
{
const s=slide('검출·분할 데이터 준비','윤종호는 데이터 파일을 확인한 뒤 촬영 시퀀스 단위로 분할했습니다. 학습은 518장 22,082개 주석, 검증은 152장 6,101개 주석입니다. 객체별 마스크를 YOLO 분할 다각형으로 변환하고 원본 ID 대응표를 남겼습니다. 검증은 모델 선택용이며 독립 최종 시험 성능과 구분합니다. 다른 시퀀스에 같은 나무나 배경이 남을 가능성은 있습니다.',[src.split,'convert_masks_to_yolo.py','데이터설정_실행방법.md']);
 text(s,'670쌍의 이미지·마스크  /  28,183개 객체 주석',68,155,1120,60,32,C.green,true);
 tab(s,[['구분','이미지','사과 주석','촬영 시퀀스'],['학습','518장','22,082개','8개'],['검증','152장','6,101개','2개']],[280,250,350,272],64,250,1152,215,27);
 text(s,'파일 검사  ·  시퀀스 분할  ·  라벨 변환  ·  실제 YOLO 로딩 확인',68,512,1120,80,27);
 text(s,'검증 자료는 모델 선택용이며 독립 최종 시험 자료와 구분',68,620,1120,38,21,C.muted);
}
{
const s=slide('개별 사과 마스크와 주석 범위 검수','이 그림은 실제 샘플의 원본과 정답 마스크를 비교한 검수 결과입니다. 객체별 ID와 경계를 확인하고 같은 ID가 여러 조각으로 나뉘는 사례, 내부 구멍과 작은 객체를 처리했습니다. 변환한 주석은 28,183개이고 제외 객체는 없습니다. 그림은 모델의 예측 성공 사례가 아니라 정답 데이터의 품질을 점검한 자료입니다.',[src.split,'outputs/data_audit/20261116T113330617064Z/samples/20150919_174151_image166/comparison.png','추가데이터_검수결과.md']);
 await image(s,'outputs/data_audit/20261116T113330617064Z/samples/20150919_174151_image166/comparison.png',64,142,1152,435);
 text(s,'떨어진 조각·내부 구멍·작은 윤곽을 포함해 변환 주석 28,183개 유지',68,589,1120,42,25,C.green,true);
 text(s,'정답 검수 그림  /  MinneApple은 전경 나무의 사과를 중심으로 주석',68,638,1120,25,18,C.muted);
}
{
const s=slide('짧은 학습과 작은 마스크 문제','학습을 바로 오래 돌리기 전에 32장과 검증 8장으로 3 epoch를 실행했습니다. 학습과 검증, 가중치 저장과 재로딩은 정상 동작했습니다. 다만 분할 손실은 증가했으므로 성능 개선으로 해석하지 않습니다. 기본 축소 설정에서는 1,797개 중 두 마스크가 비었고, 고해상도 설정에서는 같은 검사에서 모두 남았습니다. 작은 객체를 보존하는 설정이 이후 메모리 문제와 연결됩니다.',[src.smoke,src.baseline]);
 tab(s,[['확인 항목','실행 결과'],['GPU 학습·검증','32장 학습 / 8장 검증, 3 epoch 정상 종료'],['모델 저장·재로딩','best.pt와 last.pt 저장, 재로딩 확인'],['분할 손실','4.4378에서 4.9817로 증가'],['mask_ratio=4','1,797개 중 2개 마스크가 비어 있음'],['mask_ratio=1','같은 검사에서 빈 마스크 0개']],[355,797],64,155,1152,380,23);
 text(s,'이번 완료의 의미는 학습 동작 확인',68,581,1120,44,31,C.green,true);
 text(s,'3 epoch 소량 학습으로 수렴이나 최종 분할 성능을 주장하지 않음',68,634,1120,30,21,C.muted);
}
{
const s=slide('전체 학습 중단과 Colab 전환','전체 데이터를 사용한 학습은 메모리 부족으로 중단됐습니다. 배치 2는 첫 epoch에서 종료됐고 배치 1은 첫 epoch와 검증을 마친 후 두 번째 epoch에서 종료됐습니다. 체크포인트와 로그를 보존했습니다. Colab 노트북과 데이터 세 파일을 Drive에 업로드했지만 런타임 GPU와 환경 확인은 남아 있습니다. 아직 Colab에서 기본 학습을 시작하지 않았습니다.',[src.baseline,src.colab,src.yoon]);
 tab(s,[['단계','현재 상태'],['로컬 batch 2','첫 epoch 도중 CUDA 메모리 부족'],['로컬 batch 1','1 epoch 학습·검증 후 2 epoch 도중 종료'],['보존한 결과','체크포인트, results.csv, 설정과 오류 로그'],['Colab 준비','노트북 저장과 데이터 세 파일 업로드 완료'],['남은 실행','GPU·환경 확인 후 기본 학습 시작']],[335,817],64,155,1152,365,24);
 text(s,'작은 마스크 보존과 GPU 메모리 사용의 균형이 핵심 과제',68,562,1120,52,29,C.red,true);
 text(s,'전체 기본 학습과 학습 후 평가는 미완료',68,627,1120,30,23,C.muted);
}
{
const s=slide('익음도·무게의 ResNet18 전이학습','정해린의 추가 작업은 고정된 ResNet18 특징을 활용한 전이학습입니다. 본체를 고정하고 512차원 특징에서 마지막 분류층 또는 회귀층만 학습했습니다. 원본 학습과 합성 가림 증강을 비교했습니다. 익음도 입력은 개별 사과 영역이고 무게는 실험실 전체 프레임입니다. 증강 입력 수가 늘어도 새로운 사과를 추가로 수집한 것은 아닙니다.',[src.transfer,src.ripeness,src.weight]);
 text(s,'사전학습 ResNet18 본체 고정  +  512차원 특징  +  예측층 학습',68,160,1120,80,29,C.green,true);
 tab(s,[['구분','익음도 분류','무게 회귀'],['입력','정답 사과 박스 영역','실험실 전체 프레임'],['학습 자료','895영역 / 271개 사과 ID','48과실'],['증강 입력','같은 영역에서 3,580개','같은 과실에서 192개'],['평가 자료','202영역 / 115개 사과 ID','12과실']],[245,470,437],64,280,1152,300,23);
 text(s,'ResNet18 전체 미세조정은 수행하지 않음',68,622,1120,34,22,C.muted);
}
{
const s=slide('가림 증강 후 익음 누락 감소','같은 평가 202영역을 사용했고 그중 익음은 20건입니다. 색 기준은 추가 가림 30%와 50%에서 익음 20건을 모두 놓쳤습니다. 원본만 학습한 ResNet도 개선이 있지만 증강 모델은 0·15%에서 19건, 30·50%에서 18건을 찾았습니다. 따라서 이 합성 가림 조건에서는 익음 재현율이 유지됩니다. 다만 미숙을 잘못 익음으로 고르는 문제는 다음 슬라이드에서 함께 봐야 합니다.',[src.ripeness,src.transfer,'haerin/2026-11-16/occlusion/results/ripeness_stress_summary.json']);
 chart(s,[['색 기준',[85,40,0,0],C.red],['ResNet 원본',[70,85,80,80],C.blue],['ResNet 증강',[95,95,90,90],C.green]],'익음 재현율 (%)',100,20);
 text(s,'18 / 20',946,233,275,65,49,C.green,true);
 text(s,'추가 가림 50%에서\n찾아낸 익음 관측',946,314,275,105,25);
 text(s,'평가 202영역\n익음 20 / 미숙 182',946,464,275,85,23,C.muted);
 text(s,'같은 과실의 반복 관측 포함  /  정답 사과 영역 사용',68,632,1120,30,21,C.muted);
}
{
const s=slide('익음 정밀도와 오선택 문제','누락을 줄인 결과만으로 수확 선별에 쓸 수 있다고 볼 수는 없습니다. 원본 조건에서 증강 모델이 익음이라고 선택한 것은 54건이고 그중 실제 익음은 19건입니다. 정밀도는 35.2%이며 색 기준의 36.2%보다 높아지지 않았습니다. 혼동행렬을 보면 미숙 182건 중 35건을 익음으로 잘못 고릅니다. 클래스 비율이 불균형하므로 전체 정확도만으로 판단하지 않고 정밀도와 재현율을 함께 제시합니다.',[src.ripeness,src.transfer,src.base]);
 text(s,'35.2%',68,154,410,105,76,C.red,true);
 text(s,'익음으로 선택한 54건 중 실제 익음 19건',68,280,1120,50,30);
 tab(s,[['원본 조건의 증강 모델','익음 예측','미숙 예측'],['실제 익음 20건','19','1'],['실제 미숙 182건','35','147']],[510,321,321],64,385,1152,190,26);
 text(s,'익음 재현율 95%와 함께 오선택 감소가 필요',68,616,1120,40,27,C.green,true);
}
{
const s=slide('같은 사과의 가림별 익음도 예측','이 예시는 실제 같은 사과 영역에 합성 잎을 더한 결과입니다. 기존 색 기준은 녹색 잎이 늘어나자 미숙으로 바뀌었습니다. ResNet 모델은 이 사례에서 익음 판정을 유지했습니다. 한 장의 성공 사례와 전체 평가를 구분해야 하며 앞의 조건별 지표가 전체 비교의 근거입니다. 사진과 원자료 표기는 그림에 보존했습니다.',[src.transfer,'haerin/2026-11-16/전이학습/figures/가림_익음도_전이학습.png']);
 await image(s,'haerin/2026-11-16/전이학습/figures/가림_익음도_전이학습.png',64,145,1152,480);
 text(s,'예시 관측의 결과이며 전체 조건별 지표와 함께 해석',68,638,1120,27,21,C.muted);
}
{
const s=slide('무게 추정에서 가림 증강 효과','무게는 같은 평가 과실 12개를 사용했습니다. 원본 조건의 MAE는 면적 기준 36.52g, 원본 ResNet 14.26g, 증강 ResNet 13.41g입니다. 가림 50%에서는 원본 ResNet 오차가 49.96g으로 커지지만 증강 모델은 18.48g입니다. 이 데이터에서는 가림 증강이 무게 예측을 안정화했습니다. MAE는 실제 무게와 예측의 차이를 절댓값으로 평균한 값이며 작을수록 좋습니다.',[src.weight,src.transfer,'haerin/2026-11-16/weight/results/weight_metrics.json']);
 chart(s,[['면적 기준',[36.52,39.15,42.15,48.66],C.red],['ResNet 원본',[14.26,30.30,38.26,49.96],C.blue],['ResNet 증강',[13.41,14.52,16.17,18.48],C.green]],'무게 MAE (g)',60,10);
 text(s,'18.48g',946,230,275,70,48,C.green,true);
 text(s,'추가 가림 50%의\n증강 모델 MAE',946,312,275,100,25);
 text(s,'학습 48과실\n평가 12과실',946,463,275,75,23,C.muted);
 text(s,'고정 실험실 촬영 조건의 개별 기능 평가  /  현장 성능은 별도 검증 필요',68,632,1120,30,21,C.muted);
}
{
const s=slide('무게 모델의 촬영 조건과 적용 한계','무게 결과는 촬영 조건 편향을 함께 봐야 합니다. 자료에서 카메라별 무게 범위가 서로 다르며 카메라별 평균만 답해도 MAE는 12.92g입니다. 따라서 증강 모델의 원본 13.41g이 물리적인 크기와 무게 관계를 충분히 학습했다는 증거는 아닙니다. 또한 전체 실험실 프레임과 과수원의 사과 크롭은 입력 조건이 달라 바로 연결하면 안 됩니다.',[src.base,src.transfer,src.weight,'haerin/2026-11-16/결과사진/06_무게평가_사과12개_가림비교.jpg']);
 tab(s,[['비교 또는 조건','확인한 내용'],['카메라별 평균 기준','MAE 12.92g'],['증강 ResNet 원본 조건','MAE 13.41g'],['현재 입력','실험실 전체 프레임'],['통합 시 필요한 확인','과수원 촬영 거리·배율과 입력 영역 보정']],[430,722],64,170,1152,300,26);
 text(s,'새 카메라·새 촬영 조건·과수원에서 별도 검증 필요',68,527,1120,70,32,C.red,true);
 text(s,'현재 결과를 과수원 사과 크롭의 무게 추정 성능으로 일반화하지 않음',68,625,1120,35,22,C.muted);
}
{
const s=slide('자연 가림과 추가 합성 가림','자연 가림은 데이터의 보이는 윤곽과 추정 전체 윤곽을 비교한 것입니다. 합성 가림률은 원래 보이던 사과 픽셀 중 추가로 덮은 비율입니다. 그러므로 추가 가림 0%라도 원래 사과가 가려져 있을 수 있습니다. 이번 비교는 정적인 녹색 잎 도형 조건이며 실제 잎 움직임이나 조명 변화까지 검증한 결과는 아닙니다.',[src.transfer,'haerin/2026-11-16/결과사진/04_자연가림과_합성잎.png']);
 await image(s,'haerin/2026-11-16/결과사진/04_자연가림과_합성잎.png',64,143,680,500);
 text(s,'자연 가림',800,197,410,43,30,C.green,true);
 text(s,'보이는 영역과\n추정 전체 윤곽 비교',800,259,410,110,27);
 text(s,'추가 합성 가림',800,410,410,43,30,C.green,true);
 text(s,'원래 보이던 픽셀의\n15·30·50%를 추가 덮음',800,472,410,115,27);
 text(s,'움직임·실제 풍속 실험은 아직 수행하지 않음',800,616,410,48,20,C.muted);
}
{
const s=slide('프로젝트 통합 진행 상황','지금까지의 성과는 데이터 준비, 짧은 학습, 개별 상태 분석 모델과 가림 비교입니다. 검출 전체 학습은 완료하지 못했고 통합 입력과 실제 출력 연결도 남아 있습니다. 움직임, 깊이 기반 실제 지름, 처리 속도와 최종 시연은 별도 범위 조율과 구현이 필요합니다. 작업별 상태로 관리하며 전체 퍼센트는 임의로 계산하지 않습니다.',[src.team,src.yoon,src.jeong]);
 tab(s,[['영역','현재 상태','다음 과제'],['검출·분할','데이터·짧은 학습 완료','Colab 기본 학습과 검증'],['익음도','전이학습·가림 평가 완료','오선택 분석과 임계값 검토'],['무게','조건별 회귀 평가 완료','촬영 조건 보정·적용 범위 확인'],['기능 연결','연결 방식 검토','입력 규격과 임시 출력 구현'],['전체 평가·시연','미진행','검출 누락 포함 평가와 시연']],[250,450,452],64,155,1152,415,23);
 text(s,'개별 기능의 평가 완료와 전체 시스템 완성을 구분',68,620,1120,40,29,C.green,true);
}
{
const s=slide('임시 연결을 위한 공통 입력 규격','전체 학습을 기다리는 동안 각 기능 평가와 임시 연결을 병행할 수 있습니다. 먼저 정답 박스와 마스크로 사진 입력부터 객체별 결과 출력까지 연결하고 이후 저장된 YOLO 예측으로 바꾸는 방식입니다. 이미지와 객체 ID, 좌표·마스크, 입력 출처와 검출 신뢰도를 맞춰야 합니다. 정답 영역으로 연결한 결과는 검출 성능을 포함하지 않습니다. 익음도 예측층과 ResNet 가중치 일부도 확보하거나 재학습해야 합니다.',[src.team,src.transfer,src.jeong]);
 tab(s,[['전달 항목','연결 목적'],['이미지 ID·객체 번호','같은 사진과 같은 사과의 결과 대응'],['원본 이미지·박스·마스크','사과 영역 추출과 좌표·크기 규칙 통일'],['입력 출처·검출 신뢰도','정답과 예측을 구분해 기록'],['익음 점수·무게·판단 상태','임계값, 적용 조건, 판단 보류 표시']],[440,712],64,165,1152,330,25);
 text(s,'정답 영역으로 임시 연결 후 저장된 YOLO 예측으로 교체',68,547,1120,65,30,C.green,true);
 text(s,'제안 단계  /  무게 입력 조건 보정과 누락 모델 파일 확보가 선행 과제',68,631,1120,30,21,C.muted);
}
{
const s=slide('다음 작업과 기존 목표 일정','다음 순서는 학습 실행과 기능 평가, 임시 연결, 실제 예측 영역 적용, 통합 평가입니다. 기존 계획은 11월 말 기본 통합과 12월 평가·시연, 12월 14~15일 최종 제출을 목표로 합니다. 학습과 촬영 조건 문제에 따라 조정할 수 있습니다. 새 사진 오류 분석을 뒤로 미루더라도 기존 고정 검증 자료 평가는 유지합니다. 움직임과 깊이 기능은 우선 구현 범위와 담당을 확정해야 합니다.',[src.team,src.yoon,'원본 팀 활동 보고서의 프로젝트 추진 일정']);
 tab(s,[['우선순위','다음 작업','담당'],['1','Colab GPU·환경 확인, 기본 학습 완료','윤종호'],['2','익음도 오선택·무게 조건·모델 파일 점검','정해린'],['3','정답 영역 임시 연결과 결과 화면','공동'],['4','YOLO 예측 교체, 통합 평가·오류 분석','공동'],['5','시연 영상과 최종 보고 자료 정리','공동']],[180,780,192],64,160,1152,365,23);
 text(s,'기존 목표  11월 말 통합 / 12월 평가·시연 / 12월 14~15일 제출',68,565,1120,62,27,C.green,true);
 text(s,'학습·보정 진행에 따라 조정 가능  /  움직임·깊이의 구현 범위는 추가 협의',68,636,1120,30,20,C.muted);
}
{
const s=slide('현재 성과와 다음 확인 과제','우리 팀은 검출 데이터를 학습 가능한 형태로 준비하고 실제 GPU의 짧은 학습을 확인했습니다. 정해린의 전이학습과 가림 증강은 합성 가림에서 익음 누락과 무게 예측 변화를 줄였습니다. 남은 핵심은 기본 검출 모델 학습을 마치고 익음 오선택과 촬영 조건 차이를 해결하면서 기능을 연결하는 것입니다. 다음 보고에서는 실제 검출을 포함한 실행 화면과 평가 결과를 제시하는 것을 목표로 하겠습니다.',[src.split,src.smoke,src.transfer,src.team]);
 line(s,'확보한 결과','670쌍·28,183개 주석 준비와 짧은 학습\n익음도·무게 전이학습 및 합성 가림 비교',170);
 line(s,'다음 확인','전체 기본 학습, 익음 오선택, 무게 촬영 조건\n임시 연결 후 실제 예측을 포함한 통합 평가',395);
 text(s,'질의응답',68,624,1000,45,32,C.green,true);
}
const notes=[];
for(let i=0;i<P.slides.items.length;i++){
 const s=P.slides.items[i];notes.push({slide:i+1,seconds:timings[i],notes:s.speakerNotes.text});
 const blob=await P.export({slide:s,format:'png',scale:1});await fs.writeFile(path.join(preview,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
 console.log(`rendered ${i+1}/18`);
}
await fs.writeFile(path.join(BUILD,'speaker-notes.json'),JSON.stringify(notes,null,2));
const staging=path.join(BUILD,'finalizer');await fs.mkdir(staging,{recursive:true});
const candidate=path.join(staging,'candidate.pptx');await(await PresentationFile.exportPptx(P)).save(candidate);
await finalizePresentation({workspaceDir:WORK,candidatePath:candidate,finalPath:path.join(OUT,'사과수확로봇_진행발표_18장_2026-11-16.pptx'),pythonExecutable:PY,integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[2,3,4,6,7,8,10,13,15,16,17].flatMap(n=>['--require-native-table-slide',String(n)])],explicitTotalSlideCount:18,requiredNativeTableOwnerSlides:[2,3,4,6,7,8,10,13,15,16,17],requiredNativeChartOwnerSlides:[9,12],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,receiptPath:path.join(staging,'validation.json')});
console.log(`Completed 18 slides, ${timings.reduce((a,b)=>a+b,0)} seconds of speaker notes`);
