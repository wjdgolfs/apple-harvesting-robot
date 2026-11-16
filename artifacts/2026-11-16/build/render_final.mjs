import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const B=path.dirname(fileURLToPath(import.meta.url));
const file=path.join(B,'../output/사과수확로봇_진행발표_18장_2026-11-16.pptx');
const p=await PresentationFile.importPptx(await FileBlob.load(file));
const out=path.join(B,'final-slides');await fs.mkdir(out,{recursive:true});
for(let i=0;i<p.slides.items.length;i++){
 const png=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(out,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
console.log(`Rendered ${p.slides.items.length} final slides`);
