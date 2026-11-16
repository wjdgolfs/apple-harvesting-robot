import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const B=path.dirname(fileURLToPath(import.meta.url)),WORK=path.resolve(B,'../..'),O=path.join(WORK,'output');
const SOURCE=path.join(O,'사과수확로봇_진행발표_18장_2026-11-16.pptx');
const FINAL=path.join(O,'사과수확로봇_주간활동발표_보고서반영_18장.pptx');
const REPORT='C:/26_2/컴퓨터비전/팀플/11주차/팀활동보고서_Team.D_24_양식적용.docx';
const SKILL='C:/Users/YJH/.codex/plugins/cache/openai-primary-runtime/presentations/26.1112.11217/skills/presentations';
const PY='C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES='C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt/dependencies/node/node_modules';
const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const P=await PresentationFile.importPptx(await FileBlob.load(SOURCE));
const tableInspection=await P.inspect({kind:'table',maxChars:60000});
const tableAnchors=tableInspection.ndjson.split('\n').filter(Boolean).map(x=>JSON.parse(x)).filter(x=>x.kind==='table');
const FONT='Malgun Gothic',C={ink:'#17352D',green:'#16654C',muted:'#5D6B66',red:'#A94A32',paper:'#FAFAF7',gray:'#DDE3DF'};
const s=n=>P.slides.items[n-1];
function match(n,old){const a=s(n).shapes.items.filter(x=>String(x.text)===old);if(a.length!==1)throw new Error(`Expected unique text on slide ${n}: ${old}`);return a[0];}
function change(n,old,value){const sh=match(n,old);sh.text=value;return sh;}
function add(n,value,x,y,w,h,size=26,color=C.ink,bold=false){
 const sh=s(n).shapes.add({name:value.slice(0,50),geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 sh.text=value;sh.text.style={typeface:FONT,fontSize:size,color,bold,insets:0,verticalAlignment:'top',alignment:'left'};return sh;
}
function removeTables(n){for(const t of tableAnchors.filter(x=>x.slideIndex===n-1))P.delete(t.id);if(s(n).tables.items.length)throw new Error(`Unremoved source table on slide ${n}`);}
function table(n,rows,widths,y=170,h=350,size=24){
 const t=s(n).tables.add({rows:rows.length,columns:rows[0].length,left:64,top:y,width:1152,height:h,columnWidths:widths,values:rows});
 t.borders.assign({fill:C.gray,width:1,style:'solid'});
 t.cells.block({row:0,column:0,rowCount:rows.length,columnCount:rows[0].length}).assign({textStyle:{typeface:FONT,fontSize:size,color:C.ink},margins:{left:14,right:14,top:10,bottom:10},anchor:'center'});
 for(let i=0;i<rows.length;i++){
  t.rows[i].height=h/rows.length;
  for(let j=0;j<rows[0].length;j++){
   t.getCell(i,j).fill=i===0?'#E2EBE5':C.paper;
   t.getCell(i,j).text.style={typeface:FONT,fontSize:size,color:C.ink,bold:i===0};
  }
 }
 return t;
}
async function replaceImage(n,filename){
 if(s(n).images.items.length!==1)throw new Error('Expected one evidence image');
 const i=s(n).images.items[0];const bytes=await fs.readFile(path.join(B,filename));
 const frame=i.frame,crop=i.crop,fit=i.fit;
 i.replace({blob:bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),contentType:'image/jpeg',alt:`정리된 팀 활동 보고서의 그림 ${filename}`,fit});
 i.frame=frame;i.crop=crop;i.fit=fit;
}
// Preserve the source design and slide count; change the weekly report scope.
change(1,'Team.D_24\n윤종호 · 정해린\n2026년 11월 16일 진행 보고','Team.D_24\n윤종호 · 정해린\n활동 기간 2026년 11월 10~16일');
change(1,'검출·분할과 익음도·무게 추정\n합성 잎 가림에 따른 성능 비교','11주차 팀 활동 보고\n검출·분할과 익음도·무게의 개별 평가');
change(2,'프로젝트 범위와 역할 분담','프로젝트 목표와 현재 단계');
change(2,'현재 개발 범위는 영상 인식이며, 실제 로봇의 수확 동작은 포함하지 않음','데이터 준비와 개별 기능 평가 완료 / 전체 학습·통합 평가는 남아 있음');
change(3,'사용 데이터와 평가 목적','이번 활동의 핵심 결과');removeTables(3);
table(3,[['항목','결과','평가 조건'],['검출·분할 데이터','670쌍 / 28,183개 주석','데이터 준비·짧은 학습 완료'],['익음 재현율','90%  (18 / 20)','추가 가림 50%'],['익음 정밀도','35.2%','원본 조건  (추가 가림 0%)'],['무게 MAE','13.41g / 18.48g','추가 가림 0% / 50%']],[310,400,442],170,365,24);
change(3,'기능별 자료와 촬영 조건이 달라, 개별 결과와 통합 평가를 구분','모두 기능별 개별 평가 결과이며 통합·현장 성능과 구분');
change(4,'파일 검사  ·  시퀀스 분할  ·  라벨 변환  ·  실제 YOLO 로딩 확인','촬영 시퀀스 단위 분할과 객체별 YOLO 라벨 변환 / 누락 0개');
change(5,'떨어진 조각·내부 구멍·작은 윤곽을 포함해 변환 주석 28,183개 유지','보고서 그림 1  원본 이미지와 개별 사과 마스크 검수');
await replaceImage(5,'2e2589b5fcb6b32e9342238834d488ba5117190d.jpg');
const t6=s(6).tables.items[0];t6.cells.set(3,0,'실행 환경');t6.cells.set(3,1,'YOLOv8n-seg / RTX 5060 Laptop GPU');
change(7,'전체 기본 학습과 학습 후 평가는 미완료','전체 학습 완료 후 검증 152장에서 정식 결과 정리');
const t7=s(7).tables.items[0];t7.cells.set(3,0,'1 epoch의 mAP50');t7.cells.set(3,1,'박스 0.617 / 마스크 0.565  (중간 기록)');
change(8,'익음도·무게의 ResNet18 전이학습','익음도 모델 비교 조건');
change(8,'사전학습 ResNet18 본체 고정  +  512차원 특징  +  예측층 학습','색 기준과 ResNet18의 원본·가림 증강 학습 비교');removeTables(8);
table(8,[['모델','학습·판별 방법','비교 목적'],['색 기준','사과 영역의 색 차이 이용','기존 기준 성능'],['ResNet 원본 학습','원본 사과 이미지로 학습','전이학습 효과'],['ResNet 가림 증강','원본과 합성 잎 가림으로 학습','가림 대응 효과']],[330,510,312],265,280,24);
change(8,'ResNet18 전체 미세조정은 수행하지 않음','평가 202영역 / 익음 20건은 실제 13개 사과의 관측');
change(9,'같은 과실의 반복 관측 포함  /  정답 사과 영역 사용','익음 20건은 13개 과실의 관측 / 정답 영역 기반 평가');
change(10,'익음 정밀도와 오선택 문제','원본 조건의 익음 정밀도');
change(10,'익음 재현율 95%와 함께 오선택 감소가 필요','색 기준 36.2%와 비슷한 정밀도 / 오선택 문제는 남아 있음');
change(11,'같은 사과의 가림별 익음도 예측','무게 추정 비교 조건');
for(const image of [...s(11).images.items])image.delete();
change(11,'예시 관측의 결과이며 전체 조건별 지표와 함께 해석','실험실 전체 사진 입력 / 같은 평가 과실 12개를 조건별 비교');
add(11,'학습 48과실 / 평가 12과실',68,160,1120,60,32,C.green,true);
table(11,[['모델','사용한 정보와 학습 조건'],['면적 기반 회귀','이미지에서 보이는 사과 면적'],['ResNet 원본 학습','원본 이미지의 특징을 이용한 회귀'],['ResNet 가림 증강','원본·합성 가림 이미지의 특징을 이용한 회귀']],[420,732],270,285,25);
const stability=change(12,'학습 48과실\n평가 12과실','예측 변화량\n43.21g에서\n14.37g으로 감소');stability.position={left:946,top:444,width:275,height:130};stability.text.style={typeface:FONT,fontSize:23,color:C.muted,insets:0};
change(12,'고정 실험실 촬영 조건의 개별 기능 평가  /  현장 성능은 별도 검증 필요','원본 대비 50% 가림의 예측 변화량 비교 / 현장 성능은 별도 검증 필요');
await replaceImage(14,'ad396831fd0e3f799810e73c3301d638914d87ba.jpg');
change(14,'움직임·실제 풍속 실험은 아직 수행하지 않음','실제 잎·조명·움직임 조건은 아직 검증하지 않음');
change(15,'프로젝트 통합 진행 상황','기능 통합 현황과 연결 순서');removeTables(15);
add(15,'검출·분할과 익음도·무게는 현재 따로 실행 중',68,155,1120,70,30,C.green,true);
table(15,[['순서','계획한 연결 단계'],['1','기능별 평가 결과 정리'],['2','정답 박스·마스크로 임시 연결'],['3','YOLO 예측으로 입력 교체'],['4','검출 누락까지 포함한 통합 평가']],[180,972],255,310,25);
change(15,'개별 기능의 평가 완료와 전체 시스템 완성을 구분','이미지 ID·객체 번호·박스·마스크 형식을 맞춰 연결할 계획');
change(16,'임시 연결을 위한 공통 입력 규격','통합의 핵심 제약과 다음 조치');removeTables(16);
table(16,[['핵심 제약','현재 상태','다음 조치'],['무게 입력 조건','실험실 전체 사진으로 학습','사과 크롭 적용 전 촬영 조건·보정 검토'],['모델 파일','익음도 head 등 일부 누락','파일 확보 또는 재학습'],['평가 기준','정답 영역 기반 개별 평가','검출 누락 포함 통합 평가'],['깊이·실제 지름','자료 검토 단계','보정 방식·담당 조율']],[245,385,522],165,350,22);
change(16,'정답 영역으로 임시 연결 후 저장된 YOLO 예측으로 교체','기능 연결 전에 입력 조건과 평가 기준을 함께 정리');
change(16,'제안 단계  /  무게 입력 조건 보정과 누락 모델 파일 확보가 선행 과제','정답 영역을 이용한 결과와 실제 검출을 포함한 결과를 구분');
change(17,'다음 작업과 기존 목표 일정','다음 활동 계획과 역할 분담');removeTables(17);
change(17,'기존 목표  11월 말 통합 / 12월 평가·시연 / 12월 14~15일 제출','추진 기간  2026년 11월 17일~11월 23일');
const period=match(17,'추진 기간  2026년 11월 17일~11월 23일');period.position={left:68,top:158,width:1120,height:60};
table(17,[['담당','다음 미팅까지 할 일'],['윤종호','기본 모델 검출 결과 확인·오류 분석'],['정해린','익음도 판별·이미지 기반 무게 추정 방법 검토'],['공동','기능 통합용 입력·출력 형식 정리']],[240,912],270,280,26);
change(17,'학습·보정 진행에 따라 조정 가능  /  움직임·깊이의 구현 범위는 추가 협의','보고서의 다음 활동 기간과 담당별 계획을 기준으로 정리');
change(18,'전체 기본 학습, 익음 오선택, 무게 촬영 조건\n임시 연결 후 실제 예측을 포함한 통합 평가','기본 모델 결과 확인·오류 분석\n익음도·무게 방법 검토와 통합 입출력 형식 정리');

// Update the existing full script and embed matching presenter notes.
const script=JSON.parse(await fs.readFile(path.join(B,'../script-content.json'),'utf-8'));
const revisions={
1:'안녕하세요. 사과 수확 로봇을 위한 영상 인식 기능 개발을 진행한 Team.D_24의 윤종호, 정해린입니다.\n\n이번 발표는 11월 10일부터 16일까지 정리한 팀 활동 보고서를 기준으로 합니다. 검출·분할 데이터와 학습 점검, 익음도·무게의 개별 평가 결과, 기능 통합 현황과 다음 주 계획을 설명드리겠습니다.',
3:'이번 활동의 핵심 결과를 먼저 말씀드리겠습니다. 검출·분할에서는 이미지와 마스크 670쌍, 사과 주석 2만 8,183개를 학습 가능한 형태로 준비하고 짧은 학습을 확인했습니다.\n\n익음도 증강 모델은 추가 가림 50퍼센트에서도 익음 20건 중 18건을 찾아 재현율 90퍼센트를 유지했습니다. 다만 표의 정밀도 35.2퍼센트는 원본, 즉 추가 가림 0퍼센트 조건의 결과입니다. 두 수치는 평가 조건이 다릅니다.\n\n무게 증강 모델의 평균 절대오차는 원본 13.41그램, 추가 가림 50퍼센트에서 18.48그램이었습니다. 이 결과들은 각 기능을 따로 평가한 수치이며, 통합 시스템이나 과수원 현장 성능은 아닙니다.',
6:'전체 학습에 앞서 YOLOv8n-seg로 학습 32장과 검증 8장을 사용해 3회 반복 학습을 실행했습니다. RTX 5060 Laptop GPU에서 학습과 검증, 모델 저장과 재로딩이 정상 동작하는지 확인했습니다.\n\n이 과정에서 작은 사과 마스크의 축소 문제를 확인했습니다. 기본 설정인 mask_ratio 4에서는 검사한 1,797개 중 두 마스크가 비어 있었습니다. 축소를 줄인 mask_ratio 1에서는 같은 검사에서 빈 마스크가 없었습니다.\n\n따라서 작은 사과를 유지하려고 마스크 해상도를 높이는 방향을 선택했습니다. 다만 이 검사는 전처리 결과를 확인한 것이고, 짧은 학습도 실행 흐름 확인이 목적입니다. 전체 학습이나 최종 모델 성능의 완료를 의미하지는 않습니다.',
7:'마스크 해상도를 유지한 전체 학습에서는 GPU 메모리 문제가 발생했습니다. 배치 2에서는 첫 번째 학습 도중 종료됐고, 배치 1로 줄여도 한 번의 학습과 검증을 마친 뒤 두 번째 학습 도중 종료됐습니다.\n\n한 번의 학습 시점에서 기록한 mAP50은 박스가 0.617, 마스크가 0.565입니다. 이 값은 중단된 학습의 중간 기록이며, 완성 모델의 최종 성능으로 해석하지 않습니다.\n\n현재 Colab 노트북 저장과 데이터 Drive 업로드를 마쳤고, 학습은 아직 시작하지 않았습니다. Colab 환경을 확인한 뒤 전체 학습을 진행하고, 정식 결과는 고정된 검증 152장에서 정리할 계획입니다.',
8:'익음도는 세 가지 방법을 비교했습니다. 첫 번째는 사과 영역의 색 차이를 사용하는 기존 기준 모델입니다. 두 번째는 원본 사과 이미지로 학습한 ResNet18 전이학습 모델이고, 세 번째는 원본에 합성 잎 가림을 함께 넣어 학습한 모델입니다.\n\n이렇게 나눈 이유는 모델을 바꾸는 효과와 가림 증강의 효과를 구분하기 위해서입니다. 색 기준과 원본 ResNet을 비교해 전이학습의 변화를 보고, 원본 ResNet과 증강 ResNet을 비교해 가림 대응의 차이를 살펴봤습니다.\n\n평가 자료는 202개 관측 영역입니다. 그중 익음 20건은 실제로는 13개 사과에서 나온 관측이므로, 20개의 독립된 과실을 평가한 것으로 설명하지 않습니다.',
11:'무게 추정도 세 가지 모델을 비교했습니다. 기존 방법은 이미지에서 보이는 사과 면적을 이용하는 회귀 모델입니다. 여기에 원본 이미지의 ResNet18 특징을 이용하는 회귀 모델과, 합성 가림을 함께 학습한 회귀 모델을 비교했습니다.\n\n학습 과실은 48개이고 평가 과실은 12개입니다. 같은 평가 과실 12개에 가림 조건을 바꾸어 적용했기 때문에, 가림이 늘어날 때 각 모델의 오차가 어떻게 달라지는지 비교할 수 있습니다.\n\n중요한 입력 조건은 실험실의 전체 사진을 사용했다는 점입니다. 검출한 사과 부분만 잘라 입력한 모델이 아니므로, 이 조건은 이후 기능 통합 때에도 확인해야 합니다.',
12:'무게 결과입니다. 가로축은 추가 합성 가림률, 세로축은 평균 절대오차인 MAE입니다. 실제 무게와 예측의 차이를 절댓값으로 평균한 값으로, 작을수록 좋습니다.\n\n원본 조건에서 면적 기준은 36.52그램, 원본 ResNet은 14.26그램, 증강 ResNet은 13.41그램이었습니다. 추가 가림 50퍼센트에서는 면적 기준과 원본 ResNet의 오차가 각각 48.66그램, 49.96그램까지 증가했습니다. 반면 증강 모델은 18.48그램으로, 비교한 세 모델 중 가장 작은 오차를 보였습니다.\n\n같은 과실의 원본 대비 50퍼센트 가림 예측 변화량도 원본 학습 43.21그램에서 증강 학습 14.37그램으로 줄었습니다. 이 합성 실험에서는 가림 증강이 무게 예측의 안정성을 높였습니다. 다만 현장 성능은 별도 검증이 필요합니다.',
15:'기능 통합 현황입니다. 현재 검출·분할과 익음도·무게 분석은 아직 따로 실행하고 있습니다. 보고서에서는 이를 연결하는 순서를 네 단계로 정리했습니다.\n\n먼저 각 기능의 평가 결과를 정리하고, 두 번째로 정답 박스와 마스크를 이용해 임시로 연결합니다. 세 번째로 입력을 YOLO가 예측한 박스와 마스크로 바꾼 뒤, 마지막으로 검출 누락까지 포함한 통합 평가를 진행하는 순서입니다.\n\n이를 위해 이미지 ID와 객체 번호로 같은 사과의 결과를 대응시키고, 박스 좌표와 마스크의 형식을 맞춰야 합니다. 현재는 연결 계획을 정리한 상태이며, 전체 처리 흐름이 구현 완료된 것은 아닙니다.',
16:'연결 전에 해결할 핵심 제약은 네 가지입니다. 먼저 무게 모델은 실험실 전체 사진으로 학습했으므로 사과 크롭에 바로 적용하기 전에 촬영 조건과 보정을 검토해야 합니다.\n\n두 번째로 익음도 예측층을 비롯한 일부 모델 파일이 누락되어 있어, 파일을 확보하거나 재학습해야 합니다. 세 번째로 현재 개별 평가는 정답 영역을 사용했으므로 실제 검출의 누락과 영역 오차를 포함한 평가를 따로 진행해야 합니다.\n\n마지막으로 깊이와 실제 지름은 자료를 검토한 단계입니다. 구체적인 보정 방식과 담당을 더 조율해야 합니다. 따라서 단순히 실행 코드를 연결하는 것에 더해 입력 조건, 필요한 모델 파일, 평가 기준과 아직 정하지 않은 범위를 함께 정리하는 것이 통합 준비의 과제입니다.',
17:'보고서에 정리한 다음 활동 기간은 11월 17일부터 23일까지입니다. 역할은 다음과 같이 나누었습니다.\n\n윤종호는 기본 모델의 검출 결과를 확인하고 오류를 분석합니다. 현재 전체 학습이 남아 있으므로, 실행 환경과 모델 상태를 확인한 뒤 결과를 정리해야 합니다.\n\n정해린은 익음도 판별과 이미지 기반 무게 추정 방법을 검토합니다. 익음을 잘못 고르는 문제와 촬영 조건의 영향을 살펴보는 작업이 여기에 포함됩니다.\n\n공동으로는 기능 통합용 입력·출력 형식을 정리합니다. 이미지와 객체를 식별하는 방식, 박스와 마스크를 전달하는 방식부터 맞추어 각 기능을 연결할 준비를 하겠습니다. 이번 발표의 일정은 보고서에 명시된 다음 주 활동 기간을 기준으로 정리했습니다.',
18:'마지막으로 이번 주 성과와 다음 과제입니다. 검출·분할에서는 670쌍의 데이터와 2만 8,183개 주석을 준비하고 짧은 학습을 확인했습니다. 익음도와 무게에서는 전이학습과 합성 가림 증강을 비교해 가림 조건에서의 개선과 남은 한계를 확인했습니다.\n\n다음 활동은 기본 모델의 결과 확인과 오류 분석, 익음도·무게 추정 방법 검토, 기능 통합용 입력·출력 형식 정리입니다. 현재 개별 평가로 얻은 결과를 실제 검출과 연결하고, 이후 검출 누락까지 포함해 평가할 수 있도록 준비하겠습니다.\n\n이상으로 11월 10일부터 16일까지의 팀 활동 보고를 마치겠습니다. 감사합니다. 질문해 주시면 답변드리겠습니다.'
};
const cues={3:'핵심 결과의 수치와 평가 조건을 한 행씩 짚기',6:'정상 실행과 작은 마스크의 전처리 결과를 구분',7:'mAP50은 중간 기록임을 말하고 Colab 학습 미시작 상태 설명',8:'세 모델의 비교 목적을 순서대로 짚기',11:'학습 48과실·평가 12과실과 전체 사진 입력 조건 강조',12:'50% 지점의 오차 차이와 오른쪽 예측 변화량 설명',15:'네 단계의 계획을 위에서 아래로 짚기',16:'네 가지 제약과 각각의 다음 조치를 연결해 설명',17:'11월 17~23일 기간과 세 담당의 다음 활동 설명'};
const sections={1:'미팅 개요',2:'1 프로젝트 목표와 현재 단계',3:'1 핵심 결과',4:'2 검출·분할 데이터 준비',5:'2 그림 1',6:'3 학습 동작 확인과 메모리 문제',7:'3 학습 동작 확인과 메모리 문제',8:'4 익음도 모델 비교',9:'4 익음도 모델 비교',10:'4 익음도 모델 비교',11:'5 무게 추정 비교',12:'5 무게 추정 비교 및 그림 2',13:'5 무게 추정 비교의 한계',14:'6 가림 분석과 합성 잎 실험 및 그림 3',15:'7 기능 통합 현황',16:'7 핵심 제약',17:'추후 계획',18:'현재 단계 및 추후 계획'};
let elapsed=0;
for(let i=0;i<18;i++){
 const n=i+1,r=script[i];if(revisions[n])r.spoken=revisions[n];if(cues[n])r.cue=cues[n];
 r.title=String(s(n).shapes.items.find(x=>x.position.top<130&&!/^\d+ \/ 18$/.test(String(x.text))).text).replace(/\n/g,' ');
 r.start=elapsed;elapsed+=r.seconds;r.end=elapsed;
 s(n).speakerNotes.text=`발표 목표 ${r.seconds}초\n활동 기간 2026년 11월 10~16일\n\n${r.spoken}\n\n화면 안내\n${r.cue}\n\n근거\n${REPORT}\n해당 절 ${sections[n]}\n원본 조건의 익음 정밀도와 추가 가림 조건의 재현율을 구분한다.`;
}
await fs.writeFile(path.join(B,'revised-script.json'),JSON.stringify(script,null,2));
const tm=x=>`${String(Math.floor(x/60)).padStart(2,'0')}:${String(x%60).padStart(2,'0')}`;
const md=['# 사과 수확 로봇 주간 활동 발표 대본','','정리된 팀 활동 보고서를 반영한 PPT 18장에 대응합니다. 활동 기간은 2026년 11월 10~16일, 다음 활동 기간은 11월 17~23일입니다. 목표 발표 시간은 15분이며 실제 속도는 리허설로 조정합니다. 화면 안내는 읽지 않는 동작 설명입니다.',''];
for(const r of script)md.push(`## ${r.slide} ${r.title}`,'',`목표 ${r.seconds}초  /  ${tm(r.start)}–${tm(r.end)}`,'',`화면 안내: ${r.cue}`,'',r.spoken,'');
await fs.writeFile(path.join(O,'발표대본_보고서반영_18장_15분.md'),md.join('\n'));
await fs.mkdir(path.join(B,'after'),{recursive:true});
for(let i=0;i<18;i++){
 const png=await P.export({slide:s(i+1),format:'png',scale:1});
 await fs.writeFile(path.join(B,'after',`${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
const raw=path.join(B,'candidate-export.pptx'),candidate=path.join(B,'candidate-preserved.pptx');
await(await PresentationFile.exportPptx(P)).save(raw);
console.log(execFileSync(PY,[path.join(B,'preserve_chart_parts.py'),SOURCE,raw,candidate],{encoding:'utf-8'}));
const owners=[2,3,4,6,7,8,10,11,13,15,16,17];
await finalizePresentation({workspaceDir:WORK,candidatePath:candidate,finalPath:FINAL,pythonExecutable:PY,integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...owners.flatMap(n=>['--require-native-table-slide',String(n)])],explicitTotalSlideCount:18,requiredNativeTableOwnerSlides:owners,requiredNativeChartOwnerSlides:[9,12],requiredEmbeddedWorkbookChartOwnerSlides:[9,12],fontPolicy:{basis:'reference',families:[FONT],referencePath:SOURCE,referenceSha256:createHash('sha256').update(await fs.readFile(SOURCE)).digest('hex')},verifyArtifactToolImport:true,receiptPath:path.join(B,'validation.json')});
const finalP=await PresentationFile.importPptx(await FileBlob.load(FINAL));await fs.mkdir(path.join(B,'final-render'),{recursive:true});
for(let i=0;i<18;i++){
 const png=await finalP.export({slide:finalP.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(B,'final-render',`${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
console.log(`Completed report revision: ${FINAL}`);
