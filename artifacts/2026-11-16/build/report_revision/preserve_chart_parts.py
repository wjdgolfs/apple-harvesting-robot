from pathlib import Path
import sys,zipfile,xml.etree.ElementTree as E,hashlib,json
source,candidate,out=map(Path,sys.argv[1:4])
with zipfile.ZipFile(source) as a,zipfile.ZipFile(candidate) as b:
 keep={n:a.read(n) for n in a.namelist() if n.startswith(('ppt/charts/','ppt/slides/charts/','ppt/embeddings/'))}
 # Both unchanged charts stay on the same slides and retain the same part paths.
 ns={'p':'http://schemas.openxmlformats.org/package/2006/relationships'}
 for slide in [9,12]:
  n=f'ppt/slides/_rels/slide{slide}.xml.rels'
  targets=lambda z:[r.attrib['Target'] for r in E.fromstring(z.read(n)) if r.attrib.get('Type','').endswith('/chart')]
  assert targets(a)==targets(b),f'Chart owner changed on slide {slide}'
 root=E.fromstring(b.read('[Content_Types].xml'));seen={(x.tag,x.attrib.get('Extension'),x.attrib.get('PartName')) for x in root}
 for x in E.fromstring(a.read('[Content_Types].xml')):
  if x.attrib.get('Extension')=='xlsx' or x.attrib.get('PartName','').startswith(('/ppt/charts/','/ppt/slides/charts/')):
   key=(x.tag,x.attrib.get('Extension'),x.attrib.get('PartName'))
   if key not in seen:root.append(x);seen.add(key)
 E.register_namespace('','http://schemas.openxmlformats.org/package/2006/content-types')
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
  for n in b.namelist():
   if n in keep:continue
   z.writestr(n,E.tostring(root,encoding='utf-8',xml_declaration=True) if n=='[Content_Types].xml' else b.read(n))
  for n,payload in keep.items():z.writestr(n,payload)
with zipfile.ZipFile(out) as z:
 assert all(z.read(n)==payload for n,payload in keep.items())
print(json.dumps({'preserved_unchanged_chart_parts':len(keep),'path':str(out)}))
