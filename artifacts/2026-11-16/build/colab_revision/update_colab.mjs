import fs from 'node:fs/promises';import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';import {execFileSync} from 'node:child_process';import {createHash} from 'node:crypto';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const B=path.dirname(fileURLToPath(import.meta.url)),W=path.resolve(B,'../..'),O=path.join(W,'output');
const SOURCE=path.join(O,'사과수확로봇_주간활동발표_보고서반영_18장.pptx');
const FINAL=path.join(O,'사과수확로봇_주간발표_Colab결과반영_18장.pptx');
const ROOT=path.resolve(W,'../..'),RESULTS=path.join(ROOT,'output/pdf/MinneApple_기본학습_20261116');
const SKILL='C:/Users/YJH/.codex/plugins/cache/openai-primary-runtime/presentations/26.1112.11217/skills/presentations';
const PY='C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES='C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt/dependencies/node/node_modules';
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const P=await PresentationFile.importPptx(await FileBlob.load(SOURCE));
const data=JSON.parse(await fs.readFile(path.join(B,'metrics.json'),'utf-8')),m=data.summary.current_metrics,rows=data.rows;
const ins=await P.inspect({kind:'table',maxChars:60000});const anchors=ins.ndjson.split('\n').filter(Boolean).map(x=>JSON.parse(x)).filter(x=>x.kind==='table');
const FONT='Malgun Gothic',C={ink:'#17352D',green:'#16654C',muted:'#5D6B66',red:'#A94A32',blue:'#456C9C',paper:'#FAFAF7',gray:'#DDE3DF'};
const s=n=>P.slides.items[n-1];
function find(n,txt){const a=s(n).shapes.items.filter(x=>String(x.text)===txt);if(a.length!==1)throw new Error('Text match failed: '+n+' '+txt);return a[0];}
function edit(n,old,txt){const sh=find(n,old);sh.text=txt;return sh;}
function text(n,txt,x,y,w,h,sz=26,color=C.ink,bold=false){const sh=s(n).shapes.add({geometry:'textbox',name:txt.slice(0,45),position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});sh.text=txt;sh.text.style={typeface:FONT,fontSize:sz,color,bold,insets:0,verticalAlignment:'top'};return sh;}
function deltables(n){for(const x of anchors.filter(x=>x.slideIndex===n-1))P.delete(x.id);}
function table(n,v,widths,y,h,size=25){const t=s(n).tables.add({rows:v.length,columns:v[0].length,left:64,top:y,width:1152,height:h,columnWidths:widths,values:v});t.borders.assign({fill:C.gray,width:1,style:'solid'});t.cells.block({row:0,column:0,rowCount:v.length,columnCount:v[0].length}).assign({textStyle:{typeface:FONT,fontSize:size,color:C.ink},margins:{left:14,right:14,top:10,bottom:10},anchor:'center'});for(let r=0;r<v.length;r++){t.rows[r].height=h/v.length;for(let c=0;c<v[0].length;c++){t.getCell(r,c).fill=r===0?'#E2EBE5':C.paper;t.getCell(r,c).text.style={typeface:FONT,fontSize:size,color:C.ink,bold:r===0};}}return t;}
const percent=k=>(m[k]*100).toFixed(2)+'%';
edit(2,'데이터 준비와 개별 기능 평가 완료 / 전체 학습·통합 평가는 남아 있음','Colab 기본 학습과 개별 평가 완료 / 독립 시험·통합 평가는 남아 있음');
s(2).tables.items[0].cells.set(1,2,'데이터 준비·Colab 기본 학습 완료');
deltables(3);table(3,[['항목','결과','평가 조건'],['검출·분할 데이터','670쌍 / 28,183개 주석','학습 518장 / 검증 152장'],['Colab 기본 학습','18 epoch 완료 / 최적 8','정상 조기 종료'],['익음 재현율','90%  (18 / 20)','추가 가림 50%'],['익음 정밀도','35.2%','원본 조건  (추가 가림 0%)'],['무게 MAE','13.41g / 18.48g','추가 가림 0% / 50%']],[310,400,442],160,390,23);
// Replace the diagnostic-only stage with the completed training and its trend.
edit(6,'짧은 학습과 작은 마스크 문제','Colab 기본 학습 완료와 학습 추이');
deltables(6);
edit(6,'이번 완료의 의미는 학습 동작 확인','학습 손실은 감소했지만 검증 성능은 계속 개선되지 않음');
edit(6,'3 epoch 소량 학습으로 수렴이나 최종 분할 성능을 주장하지 않음','최대 50 epoch / patience 10 / Tesla T4 / 전체 처리 약 36분 47초');
text(6,'18 epoch에서 정상 조기 종료 / 종합 기준 최적 모델은 8 epoch',68,146,1120,70,28,C.green,true);
const ch=s(6).charts.add('line',{position:{left:60,top:230,width:865,height:345},categories:rows.map(r=>r.epoch),series:[{name:'검출 mAP50',values:rows.map(r=>Number((Number(r['metrics/mAP50(B)'])*100).toFixed(3))),line:{fill:C.green,width:3},marker:{symbol:'circle',size:6,fill:C.green}},{name:'분할 mAP50',values:rows.map(r=>Number((Number(r['metrics/mAP50(M)'])*100).toFixed(3))),line:{fill:C.red,width:3},marker:{symbol:'circle',size:6,fill:C.red}}],chartFill:C.paper,plotAreaFill:C.paper,hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:{typeface:FONT,fontSize:18}},xAxis:{title:{text:'Epoch',textStyle:{typeface:FONT,fontSize:17}},textStyle:{typeface:FONT,fontSize:14}},yAxis:{min:0,max:100,majorUnit:20,numberFormatCode:'0"%"',title:{text:'검증 mAP50',textStyle:{typeface:FONT,fontSize:18}},textStyle:{typeface:FONT,fontSize:16},majorGridlines:{fill:C.gray,width:1}},dataLabels:{showValue:false}});applyPresentationChartFont(ch,{fontFamily:FONT});
text(6,'학습 박스 손실\n2.243에서 1.328',952,277,260,90,24,C.ink);
text(6,'학습 분할 손실\n2.986에서 1.892',952,413,260,90,24,C.ink);
edit(7,'전체 학습 중단과 Colab 전환','선택된 best.pt의 최종 검증 결과');
edit(7,'작은 마스크 보존과 GPU 메모리 사용의 균형이 핵심 과제','검출보다 분할 지표가 낮아 누락·오검출·경계 오류 분석 필요');
edit(7,'전체 학습 완료 후 검증 152장에서 정식 결과 정리','검증 152장·6,101개 주석 / 독립 시험 성능이 아님');
deltables(7);text(7,'8 epoch 모델 선택 후 학습 종료 시 재검증한 결과',68,153,1120,60,28,C.green,true);
table(7,[['지표','검출  박스','분할  마스크'],['정밀도 Precision',percent('metrics/precision(B)'),percent('metrics/precision(M)')],['재현율 Recall',percent('metrics/recall(B)'),percent('metrics/recall(M)')],['mAP50',percent('metrics/mAP50(B)'),percent('metrics/mAP50(M)')],['mAP50–95',percent('metrics/mAP50-95(B)'),percent('metrics/mAP50-95(M)')]],[480,336,336],240,290,27);
edit(18,'670쌍·28,183개 주석 준비와 짧은 학습\n익음도·무게 전이학습 및 합성 가림 비교','670쌍·28,183개 주석 준비와 Colab 기본 학습 완료\n익음도·무게 전이학습 및 합성 가림 비교');
// Refresh the matching script, without changing the agreed total time.
const script=JSON.parse(await fs.readFile(path.join(B,'../report_revision/revised-script.json'),'utf-8'));
const spoken={
2:'프로젝트의 목표는 이미지와 영상에서 사과의 위치와 영역을 찾고 익음도와 예상 무게를 분석하는 것입니다. 윤종호는 검출·분할을, 정해린은 익음도·무게·가림 분석을 담당했습니다.\n\n이번에 추가된 변화는 Colab에서 검출·분할 기본 학습을 정상 완료했다는 점입니다. 따라서 현재 상태는 학습 준비 단계에서 실제 기본 모델과 검증 성능을 확보한 단계로 바뀌었습니다.\n\n다만 익음도·무게 분석과 연결한 전체 시스템은 아직 완성되지 않았고, 독립 시험과 통합 평가도 남아 있습니다. 개별 기능 결과와 실제 수확 시스템의 완성은 구분해서 설명드리겠습니다.',
3:'핵심 결과를 먼저 정리하겠습니다. 검출·분할 데이터 670쌍과 사과 주석 2만 8,183개를 준비했고, Colab 기본 학습은 18회 반복 후 정상 조기 종료했습니다. 검출·분할 종합 기준으로 선택된 모델은 8회 반복 시점의 모델입니다.\n\n익음도 증강 모델은 추가 가림 50퍼센트에서 익음 20건 중 18건을 찾아 재현율 90퍼센트를 유지했습니다. 정밀도 35.2퍼센트는 원본 조건의 값이므로 서로 다른 조건을 섞어 해석하지 않습니다.\n\n무게 평균 절대오차는 원본 13.41그램, 추가 가림 50퍼센트에서 18.48그램입니다. 모두 기능별 평가 결과이며, 전체 통합이나 현장 성능을 의미하지 않습니다.',
6:'로컬에서는 작은 사과 마스크를 유지한 전체 학습 중 GPU 메모리가 부족했습니다. 이후 Tesla T4가 할당된 Colab에서 같은 고정 분할을 사용해 기본 학습을 완료했습니다. 최대 50회 학습과 개선 대기 10회를 설정했고, 실제로는 18회에서 오류 없이 조기 종료됐습니다. 종합 기준 최적 모델은 8회 시점에서 선택했습니다.\n\n그래프는 매 학습 회차의 검증 mAP50입니다. 박스와 분할의 학습 손실은 각각 약 2.24에서 1.33, 2.99에서 1.89로 감소했지만 검증 성능은 계속 좋아지지 않았습니다. 따라서 학습을 더 오래 하는 것만으로 개선된다고 판단하지 않았습니다. 다음 장은 선택된 모델을 마지막에 다시 검증한 결과입니다.',
7:'선택된 최적 모델을 학습 종료 후 검증 152장에서 다시 평가했습니다. 검출 박스는 정밀도 75.75퍼센트, 재현율 67.14퍼센트, mAP50 72.85퍼센트였습니다. 분할 마스크는 각각 67.88퍼센트, 59.48퍼센트, 61.81퍼센트였습니다.\n\n위치와 경계가 더 정확히 일치해야 하는 mAP50에서 95까지의 평균은 박스 33.01퍼센트, 마스크 21.53퍼센트로 낮았습니다. 이 지표는 단순 정확도와는 다릅니다.\n\n기본 학습과 모델 확보는 완료했지만, 검출 누락과 오검출, 마스크 경계가 어긋나는 사례를 이미지에서 확인해야 합니다. 현재 수치는 모델 선택에 사용한 검증 자료의 결과이며 독립 시험 성능은 아닙니다.',
17:'다음 활동 기간은 보고서의 11월 17일부터 23일까지입니다. 윤종호는 이번에 확보한 best.pt를 이용해 기본 모델의 검출 결과를 확인하고 누락·오검출·분할 경계 오류를 분석합니다.\n\n정해린은 익음도 판별과 이미지 기반 무게 추정 방법을 검토합니다. 익음 오선택과 촬영 조건 영향을 확인하는 작업이 포함됩니다.\n\n공통으로는 기능 통합용 입력·출력 형식을 정리합니다. 이미지와 객체를 식별하는 방식, 박스와 마스크의 좌표 규칙을 맞춰 연결을 준비하겠습니다. Colab 학습 완료로 검출 모델을 확보했으므로, 이제는 모델 결과 확인과 기능 연결 준비에 집중할 수 있습니다.',
18:'이번에 검출·분할은 데이터 준비뿐 아니라 Colab 기본 학습까지 완료했습니다. 선택된 모델의 검증 결과도 확보했고, 익음도와 무게는 전이학습·합성 가림 비교 결과를 확보한 상태입니다.\n\n남은 핵심은 best.pt의 누락·오검출·분할 경계 오류를 확인하고, 익음도·무게의 입력 조건을 맞춰 기능을 연결하는 것입니다. 검증 수치만으로 현장 성능을 단정하지 않고, 독립 입력과 통합 흐름에서 결과를 확인하겠습니다.\n\n이상으로 최신 Colab 결과를 포함한 팀 활동 보고를 마치겠습니다. 감사합니다. 질문해 주시면 답변드리겠습니다.'
};
const cues={6:'mAP50의 개선 정체와 오른쪽 손실 감소를 비교',7:'박스와 마스크의 mAP50 및 mAP50–95를 구분'};
for(let i=0;i<18;i++){
 const n=i+1,r=script[i];if(spoken[n])r.spoken=spoken[n];if(cues[n])r.cue=cues[n];
 r.title=String(s(n).shapes.items.find(x=>x.position.top<130&&!/^\d+ \/ 18$/.test(String(x.text))).text).replace(/\n/g,' ');
 s(n).speakerNotes.text=`발표 목표 ${r.seconds}초\n\n${r.spoken}\n\n화면 안내\n${r.cue}\n\n근거\n정리된 팀 활동 보고서와 최신 Colab 결과\n${RESULTS}/baseline_summary.json\n${RESULTS}/results.csv\n${RESULTS}/run.log\n확인 사항: 최적 모델은 검출·분할 종합 기준으로 선택했으며, 분할 지표 하나의 최고 epoch를 뜻하지 않는다. 매 epoch 곡선과 종료 후 best.pt 재검증 수치는 구분한다. Blur·MedianBlur·ToGray·CLAHE 각각 p=0.01 설정이 로그에 있어 완전한 무증강으로 표현하지 않는다. 독립 시험과 통합 평가는 아직 남아 있다.`;
}
const tm=x=>`${String(Math.floor(x/60)).padStart(2,'0')}:${String(x%60).padStart(2,'0')}`;
const md=['# 사과 수확 로봇 Colab 결과 반영 발표 대본','','기존 보고서 기준 18장 구성에 최신 Colab 기본 학습 결과를 반영했습니다. 목표 발표 시간은 15분이며 실제 속도는 리허설로 조정합니다. 화면 안내는 읽지 않는 동작 설명입니다.',''];for(const r of script)md.push(`## ${r.slide} ${r.title}`,'',`목표 ${r.seconds}초 / ${tm(r.start)}–${tm(r.end)}`,'',`화면 안내: ${r.cue}`,'',r.spoken,'');
md.push('## 질문에 대비할 실험 조건','','최대 50 epoch, patience 10, 실제 18 epoch 완료, 종합 기준 최적 모델 8 epoch. 결과는 best.pt를 종료 후 검증 152장에서 다시 검증한 값입니다. 로그에 소확률 영상 증강이 포함되어 완전 무증강 실험으로 설명하지 않습니다. 독립 시험과 통합 평가는 후속 작업입니다.','');
await fs.writeFile(path.join(O,'발표대본_Colab결과반영_18장_15분.md'),md.join('\n'));await fs.writeFile(path.join(B,'script.json'),JSON.stringify(script,null,2));
await fs.mkdir(path.join(B,'previews'),{recursive:true});for(let i=0;i<18;i++){const b=await P.export({slide:s(i+1),format:'png',scale:1});await fs.writeFile(path.join(B,'previews',`${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await b.arrayBuffer()));}
const raw=path.join(B,'candidate-export.pptx'),candidate=path.join(B,'candidate-preserved.pptx');await(await PresentationFile.exportPptx(P)).save(raw);
console.log(execFileSync(PY,[path.join(B,'preserve_existing_charts.py'),SOURCE,raw,candidate],{encoding:'utf-8'}));
const owners=[2,3,4,7,8,10,11,13,15,16,17];
await finalizePresentation({workspaceDir:W,candidatePath:candidate,finalPath:FINAL,pythonExecutable:PY,integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...owners.flatMap(n=>['--require-native-table-slide',String(n)])],explicitTotalSlideCount:18,requiredNativeTableOwnerSlides:owners,requiredNativeChartOwnerSlides:[6,9,12],requiredEmbeddedWorkbookChartOwnerSlides:[9,12],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'reference',families:[FONT],referencePath:SOURCE,referenceSha256:createHash('sha256').update(await fs.readFile(SOURCE)).digest('hex')},verifyArtifactToolImport:true,receiptPath:path.join(B,'validation.json')});
const F=await PresentationFile.importPptx(await FileBlob.load(FINAL));await fs.mkdir(path.join(B,'final-render'),{recursive:true});for(let i=0;i<18;i++){const b=await F.export({slide:F.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(B,'final-render',`${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await b.arrayBuffer()));}
console.log(FINAL);
