"""Convert fixed MinneApple split masks to YOLO polygons, with ID mapping and raster audit."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

REPO=Path(__file__).resolve().parent


def contour_loop(contour, binary):
    points=contour.reshape(-1,2).astype(np.float64)
    if len(points)>=3 and cv2.contourArea(points.astype(np.float32))>0:
        return points,False
    # Preserve degenerate components using a half-pixel outline. Rasterizing
    # via int32 maps it back to the original pixel/line, with nonzero polygon area.
    component=np.zeros_like(binary)
    cv2.drawContours(component,[contour],-1,1,cv2.FILLED)
    enlarged=cv2.resize(component,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
    found,_=cv2.findContours(enlarged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(found)!=1:raise ValueError('Unexpected degenerate contour topology')
    return found[0].reshape(-1,2).astype(np.float64)/2,True


def polygonize(binary):
    contours,hierarchy=cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:raise ValueError('Empty object')
    hierarchy=hierarchy[0]
    loops=[];repaired=False
    for contour in contours:
        loop,repair=contour_loop(contour,binary)
        loops.append(loop);repaired|=repair
    # Start with the largest outer loop. Insert remaining closed contour walks
    # at nearest vertices. The bridge is walked forward and backward, so one
    # YOLO row retains component identity and the even-odd fill handles holes.
    start=max(range(len(loops)),key=lambda i:abs(cv2.contourArea(loops[i].astype(np.float32))))
    path=loops.pop(start)
    while loops:
        candidates=[]
        for j,loop in enumerate(loops):
            distances=((path[:,None,:]-loop[None,:,:])**2).sum(axis=2)
            i,k=np.unravel_index(distances.argmin(),distances.shape)
            candidates.append((float(distances[i,k]),j,int(i),int(k)))
        _,j,i,k=min(candidates)
        loop=loops.pop(j);loop=np.roll(loop,-k,axis=0)
        path=np.concatenate([path[:i+1],loop,loop[:1],path[i:i+1],path[i+1:]])
    return path,dict(components=int(np.count_nonzero(hierarchy[:,3]==-1)),
                     has_holes=bool(np.any(hierarchy[:,3]!=-1)),degenerate_repaired=repaired)


def roundtrip(poly, origin, width, height, binary):
    normalized=(poly + 0.001)/np.array([width,height])  # stabilize float32-to-int boundary conversion
    if len(poly)<3 or not np.isfinite(normalized).all() or normalized.min()<0 or normalized.max()>1:
        raise ValueError('Invalid normalized polygon')
    # The persisted decimal coordinates must be checked, not just in-memory points.
    tokens=[format(float(x),'.16g') for x in normalized.reshape(-1)]
    loaded=np.array(tokens,dtype=np.float32).reshape(-1,2)*np.array([width,height],dtype=np.float32)
    reconstructed=np.zeros_like(binary)
    coords=loaded.astype(np.int32)-np.array(origin,dtype=np.int32)
    cv2.fillPoly(reconstructed,[coords],1)
    union=np.count_nonzero(reconstructed|binary);intersection=np.count_nonzero(reconstructed&binary)
    return tokens,float(intersection/union),int(np.count_nonzero(reconstructed & (1-binary))),int(np.count_nonzero(binary & (1-reconstructed)))


def selfcheck():
    for kind in ['solid','disconnected','hole','point','line']:
        mask=np.zeros((20,20),dtype=np.uint8)
        if kind=='solid':mask[2:12,2:12]=1
        elif kind=='disconnected':mask[2:7,2:7]=1;mask[12:17,12:17]=1
        elif kind=='hole':mask[2:17,2:17]=1;mask[6:12,6:12]=0
        elif kind=='point':mask[5,5]=1
        elif kind=='line':mask[5,5:10]=1
        poly,_=polygonize(mask)
        tokens,iou,extra,missing=roundtrip(poly,[0,0],20,20,mask)
        if iou<.75 or (kind in ['solid','point','line'] and missing):raise AssertionError((kind,iou,extra,missing))
    print('Synthetic topology checks passed',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=r'C:\26_2\컴퓨터비전\팀플\detection\train')
    parser.add_argument('--manifest',default=str(REPO/'splits'/'minneapple_v1'/'manifest.csv'))
    parser.add_argument('--output',default=r'C:\26_2\컴퓨터비전\팀플\minneapple_yolo_v1')
    parser.add_argument('--rewrite-labels',action='store_true',help='Regenerate only a previously generated conversion with the same split manifest')
    args=parser.parse_args();selfcheck()
    root=Path(args.root);out=Path(args.output)
    if out.exists():
        previous=out/'conversion'/'summary.json'
        if not args.rewrite_labels or not previous.is_file():raise FileExistsError(f'Output exists: {out}')
        old=json.loads(previous.read_text(encoding='utf-8'))
        if old['split_manifest_sha256']!=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest():raise ValueError('Cannot rewrite a different split')
    manifest=Path(args.manifest)
    with manifest.open(encoding='utf-8-sig') as f:files=list(csv.DictReader(f))
    if len(files)!=670 or len({r['filename'] for r in files})!=670:raise ValueError('Unexpected split manifest')
    for split in ['train','val']:(out/'labels'/split).mkdir(parents=True,exist_ok=True)
    report=out/'conversion';report.mkdir(exist_ok=args.rewrite_labels)
    object_rows=[];image_rows=[]
    for index,row in enumerate(files,1):
        name=row['filename'];split=row['split']
        if split not in ['train','val']:raise ValueError(split)
        mask_path=root/row['mask'];raw=mask_path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['mask_sha256']:raise ValueError(f'Mask changed: {name}')
        with Image.open(mask_path) as im:mask=np.asarray(im)
        h,w=mask.shape
        if (w,h)!=(int(row['width']),int(row['height'])):raise ValueError(name)
        ids=np.unique(mask);ids=ids[ids!=0];lines=[]
        for line_index,oid in enumerate(ids):
            ys,xs=np.where(mask==oid);x0,y0=int(xs.min()),int(ys.min());x1,y1=int(xs.max())+1,int(ys.max())+1
            binary=(mask[y0:y1,x0:x1]==oid).astype(np.uint8)
            local,info=polygonize(binary);poly=local+np.array([x0,y0])
            tokens,iou,extra,missing=roundtrip(poly,[x0,y0],w,h,binary)
            lines.append('0 '+' '.join(tokens))
            object_rows.append(dict(filename=name,split=split,label_line_1based=line_index+1,source_instance_id=int(oid),class_id=0,source_area_px=int(binary.sum()),vertices=len(poly),**info,roundtrip_iou=iou,added_pixels=extra,missing_pixels=missing))
        if len(lines)!=int(row['instance_id_count']):raise ValueError(f'ID count mismatch {name}')
        target=out/'labels'/split/(Path(name).stem+'.txt')
        target.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        image_rows.append(dict(filename=name,split=split,source_image=row['image'],source_mask=row['mask'],label=str(target.relative_to(out)).replace('\\','/'),source_ids=len(ids),label_rows=len(lines),label_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        if index%100==0:print(f'Converted {index}/670 pairs',flush=True)
    for name,rows in [('object_mapping.csv',object_rows),('label_manifest.csv',image_rows)]:
        with (report/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    difficult=sorted(object_rows,key=lambda r:(r['roundtrip_iou'],r['filename'],r['source_instance_id']))[:30]
    (report/'lowest_iou_objects.json').write_text(json.dumps(difficult,indent=2),encoding='utf-8')
    values=np.array([r['roundtrip_iou'] for r in object_rows])
    summary={'source_root':str(root),'output':str(out),'split_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
             'label_files':len(image_rows),'train_images':sum(r['split']=='train' for r in image_rows),'val_images':sum(r['split']=='val' for r in image_rows),
             'normalization_pixel_offset':0.001,'source_object_ids':sum(r['source_ids'] for r in image_rows),'label_rows':len(object_rows),'dropped_objects':0,'class_names':{'0':'apple'},
             'multi_component_objects':sum(r['components']>1 for r in object_rows),'objects_with_holes':sum(r['has_holes'] for r in object_rows),
             'degenerate_repaired_objects':sum(r['degenerate_repaired'] for r in object_rows),'roundtrip_iou_min':float(values.min()),'roundtrip_iou_mean':float(values.mean()),
             'objects_below_iou_0_95':int((values<.95).sum()),'added_pixels_total':sum(r['added_pixels'] for r in object_rows),'missing_pixels_total':sum(r['missing_pixels'] for r in object_rows),
             'notes':['Class 0 means apple; original instance IDs are kept in object_mapping.csv.',
                      'A +0.001 pixel coordinate offset prevents float32 roundoff from truncating integer boundaries to the previous pixel.',
                      'Disconnected components/hole loops are joined by double-traversed thin bridges, which can add pixels.',
                      'Degenerate components receive half-pixel outlines; no original IDs are dropped.',
                      'Roundtrip uses persisted float32 coordinates and OpenCV int32 fill at original resolution, before YOLO resampling/downsampling/augmentation.',
                      'Images are not copied or linked yet. No YAML or training is performed in this step. Visual validation is next.']}
    (report/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    for image in image_rows:
        lines=(out/image['label']).read_text().splitlines()
        if len(lines)!=image['source_ids']:raise AssertionError(image)
        for line in lines:
            parts=line.split();xy=np.array(parts[1:],dtype=float)
            if parts[0]!='0' or len(xy)<6 or len(xy)%2 or not np.isfinite(xy).all() or xy.min()<0 or xy.max()>1:raise AssertionError(image)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
