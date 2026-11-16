from pathlib import Path
import runpy,sys,contextlib,io,struct,zlib,json,math
from PIL import Image
B=Path(__file__).parent
src=B.parent/'output/3조_팀활동보고서_2026-11-16_복구본.hwp'
sys.argv=['inspect',str(src),str(B/'hwp-recovery-check')]
with contextlib.redirect_stdout(io.StringIO()):d=runpy.run_path(str(B/'inspect_hwp.py'))
issues=[]
def chain_ids(start,table):
 out=[]
 while start<0xfffffffa:
  if start in out or start>=len(table):issues.append('Invalid allocation chain');break
  out.append(start);start=table[start]
 return out
for e in d['entries']:
 if e['type']!=2 or not e['size']:continue
 small=e['size']<4096;ids=chain_ids(e['start'],d['mt'] if small else d['fat'])
 if len(ids)!=math.ceil(e['size']/(d['ms'] if small else d['ss'])):issues.append('Stream allocation length mismatch '+e['name'])
def parse(data):
 r=[];p=0
 while p<len(data):
  h=struct.unpack_from('<I',data,p)[0];p+=4;tag=h&1023;level=(h>>10)&1023;size=h>>20
  if size==4095:size=struct.unpack_from('<I',data,p)[0];p+=4
  v=data[p:p+size];p+=size
  if len(v)!=size:issues.append('Truncated record')
  r.append((tag,level,v))
 return r
section=parse(zlib.decompress(d['stream'](next(e for e in d['entries'] if e['name']=='Section0')),-15))
for i,(tag,level,v) in enumerate(section):
 if tag!=66:continue
 direct=[]
 for t,l,x in section[i+1:]:
  if l<=level:break
  if l==level+1:direct.append((t,x))
 text=next((x for t,x in direct if t==67),b'')
 count=struct.unpack_from('<I',v)[0]&0x7fffffff
 # Empty source paragraphs can omit PARA_TEXT but still count the terminator.
 if text and len(text)//2!=count:issues.append(f'Paragraph {i} character count')
 expected=struct.unpack_from('<HHH',v,12)
 actual=(sum(len(x)//8 for t,x in direct if t==68),sum(len(x)//12 for t,x in direct if t==70),sum(len(x)//36 for t,x in direct if t==69))
 if expected!=actual:issues.append(f'Paragraph {i} child counts {expected} != {actual}')
 # Each native table/drawing controller must have its inline 8-WCHAR token.
 values=struct.unpack('<%dH'%(len(text)//2),text) if text else []
 controls=0;q=0
 while q<len(values):
  c=values[q]
  if c in (1,2,3,11,12,14,15,16,17,18,21,22,23):controls+=1;q+=8
  elif c==9:q+=8
  else:q+=1
 controller_count=sum(t==71 for t,x in direct)
 if controls!=controller_count:issues.append(f'Paragraph {i} inline controls {controls} != headers {controller_count}')
docinfo=parse(zlib.decompress(d['stream'](next(e for e in d['entries'] if e['name']=='DocInfo')),-15))
images=0
for tag,level,v in docinfo:
 if tag!=18:continue
 flags=struct.unpack_from('<H',v)[0]
 if flags&0xf!=1:continue
 id=struct.unpack_from('<H',v,2)[0];e=next(e for e in d['entries'] if e['name'].startswith(f'BIN{id:04X}.'))
 payload=d['stream'](e)
 if (flags>>4)&3!=2:payload=zlib.decompress(payload,-15)
 try:Image.open(io.BytesIO(payload)).verify();images+=1
 except Exception as ex:issues.append(f'Image {e["name"]}: {ex}')
receipt={'scope':'CFB allocation, paragraph record counts, inline controller consistency and encoded image data; native Hancom opening and visual layout are not verified','records':len(section),'images':images,'issues':issues,'passed':not issues}
(B/'hwp-recovery-check.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False));sys.exit(1 if issues else 0)
