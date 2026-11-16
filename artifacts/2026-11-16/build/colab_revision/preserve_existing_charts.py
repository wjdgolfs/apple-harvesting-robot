import zipfile,posixpath,sys,xml.etree.ElementTree as E
source,candidate,out=sys.argv[1:4]
def chart(z,owner):
 rel=f'ppt/slides/_rels/slide{owner}.xml.rels'
 target=next(x.attrib['Target'] for x in E.fromstring(z.read(rel)) if x.attrib.get('Type','').endswith('/chart'))
 return target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join('ppt/slides',target))
with zipfile.ZipFile(source) as a,zipfile.ZipFile(candidate) as b:
 keep={n:a.read(n) for n in a.namelist() if n.startswith('ppt/embeddings/')}
 for owner in [9,12]:
  src,dst=chart(a,owner),chart(b,owner);keep[dst]=a.read(src)
  src_rel=posixpath.join(posixpath.dirname(src),'_rels',posixpath.basename(src)+'.rels')
  dst_rel=posixpath.join(posixpath.dirname(dst),'_rels',posixpath.basename(dst)+'.rels')
  keep[dst_rel]=a.read(src_rel)
 root=E.fromstring(b.read('[Content_Types].xml'))
 if not any(x.attrib.get('Extension')=='xlsx' for x in root):
  root.append(next(x for x in E.fromstring(a.read('[Content_Types].xml')) if x.attrib.get('Extension')=='xlsx'))
 E.register_namespace('','http://schemas.openxmlformats.org/package/2006/content-types')
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
  for n in b.namelist():
   if n not in keep:z.writestr(n,E.tostring(root,encoding='utf-8',xml_declaration=True) if n=='[Content_Types].xml' else b.read(n))
  for n,v in keep.items():z.writestr(n,v)
print('Preserved original ripeness and weight chart XML, relationships and workbooks')
