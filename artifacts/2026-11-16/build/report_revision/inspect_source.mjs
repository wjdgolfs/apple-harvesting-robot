import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const B=path.dirname(fileURLToPath(import.meta.url));
const p=await PresentationFile.importPptx(await FileBlob.load(path.join(B,'../../output/사과수확로봇_진행발표_18장_2026-11-16.pptx')));
await fs.mkdir(path.join(B,'before'),{recursive:true});
const content=[];
for(let i=0;i<p.slides.items.length;i++){
 const s=p.slides.items[i];
 const png=await p.export({slide:s,format:'png',scale:1});
 await fs.writeFile(path.join(B,'before',`${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
 content.push({slide:i+1,id:s.id,shapes:s.shapes.items.map(o=>({id:o.id,name:o.name,text:String(o.text),position:o.position})),tables:s.tables.items.map(t=>({id:t.id,rows:t.rows.length,columns:t.columns.length})),images:s.images.items.map(i=>({id:i.id,alt:i.alt}))});
}
await fs.writeFile(path.join(B,'source-content.json'),JSON.stringify(content,null,2));
console.log(JSON.stringify(content.map(s=>({slide:s.slide,shapes:s.shapes.map(x=>({name:x.name,text:x.text}))})),null,2));
