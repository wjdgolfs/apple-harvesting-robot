from pathlib import Path
import struct,zlib,re,json,sys
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
SRC=Path(sys.argv[1]) if len(sys.argv)>1 else Path(r'C:\Users\YJH\OneDrive\문서\카카오톡 받은 파일\3조 팀활동보고서_수정본.hwp')
OUT=Path(sys.argv[2]) if len(sys.argv)>2 else Path(__file__).parent
OUT.mkdir(parents=True,exist_ok=True)
b=SRC.read_bytes(); u32=lambda p:struct.unpack_from('<I',b,p)[0]
ss=1<<struct.unpack_from('<H',b,30)[0]; ms=1<<struct.unpack_from('<H',b,32)[0]
def sector(i):return b[(i+1)*ss:(i+2)*ss]
fatids=list(struct.unpack_from('<109I',b,76)); nxt=u32(68)
for _ in range(u32(72)):
 d=struct.unpack('<%dI'%(ss//4),sector(nxt));fatids.extend(d[:-1]);nxt=d[-1]
fat=[]
for i in fatids:
 if i<0xfffffffa:fat.extend(struct.unpack('<%dI'%(ss//4),sector(i)))
def chain(start,table,read):
 chunks=[];seen=set()
 while start<0xfffffffa and start not in seen:
  seen.add(start);chunks.append(read(start));start=table[start]
 return b''.join(chunks)
dirs=chain(u32(48),fat,sector);entries=[]
for i in range(0,len(dirs),128):
 d=dirs[i:i+128];n=struct.unpack_from('<H',d,64)[0]
 if not n:continue
 entries.append(dict(name=d[:n-2].decode('utf-16le'),type=d[66],start=struct.unpack_from('<I',d,116)[0],size=struct.unpack_from('<Q',d,120)[0]))
root=next(x for x in entries if x['type']==5);mini=chain(root['start'],fat,sector)[:root['size']]
mf=chain(u32(60),fat,sector);mt=list(struct.unpack('<%dI'%(len(mf)//4),mf))
def stream(e):
 if e['size']<u32(56):return chain(e['start'],mt,lambda i:mini[i*ms:(i+1)*ms])[:e['size']]
 return chain(e['start'],fat,sector)[:e['size']]
print(json.dumps(entries,ensure_ascii=False))
for e in entries:
 if e['name'].startswith('BIN'):
  data=stream(e)
  try:data=zlib.decompress(data,-15)
  except zlib.error:pass
  (OUT/e['name']).write_bytes(data)
 if e['name']=='PrvText':
  t=stream(e).decode('utf-16le',errors='replace');(OUT/'hwp_preview.txt').write_text(t,encoding='utf-8');print(t)
 if e['name']=='PrvImage':(OUT/'hwp_preview.bmp').write_bytes(stream(e))
header=stream(next(e for e in entries if e['name']=='FileHeader'));compressed=bool(struct.unpack_from('<I',header,36)[0]&1)
texts=[]
for e in entries:
 if re.fullmatch(r'Section\d+',e['name']):
  data=stream(e)
  if compressed:data=zlib.decompress(data,-15)
  pos=0;parts=[]
  while pos+4<=len(data):
   h=struct.unpack_from('<I',data,pos)[0];pos+=4;tag=h&1023;size=h>>20
   if size==4095:size=struct.unpack_from('<I',data,pos)[0];pos+=4
   payload=data[pos:pos+size];pos+=size
   if tag==67:
    vals=list(struct.unpack('<%dH'%(len(payload)//2),payload[:len(payload)//2*2]));out=[];i=0
    while i<len(vals):
     c=vals[i]
     if c in (1,2,3,11,12,14,15,16,17,18,21,22,23):i+=8;continue
     if c==9:out.append('\t');i+=8;continue
     if c in (10,13):out.append('\n')
     elif c>=32:out.append(chr(c))
     i+=1
    parts.append(''.join(out))
  texts.append('\n'.join(parts))
(OUT/'hwp_fulltext.txt').write_text('\n\n'.join(texts),encoding='utf-8')
print('\nFULL TEXT\n'+'\n'.join(texts))
