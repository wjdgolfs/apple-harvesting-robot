"""Validate YOLO polygons with installed Ultralytics parser/resampling and render diagnostics."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ultralytics
from ultralytics.data.utils import verify_image_label, polygon2mask
from ultralytics.utils.ops import resample_segments

REPO=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',default=r'C:\26_2\컴퓨터비전\팀플\minneapple_yolo_v1')
    parser.add_argument('--source',default=r'C:\26_2\컴퓨터비전\팀플\detection\train')
    args=parser.parse_args();data=Path(args.dataset);source=Path(args.source)
    out=data/'conversion'/'visual_validation'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    with (data/'conversion'/'label_manifest.csv').open(encoding='utf-8-sig') as f:files=list(csv.DictReader(f))
    with (data/'conversion'/'object_mapping.csv').open(encoding='utf-8-sig') as f:objects=list(csv.DictReader(f))
    mapping=defaultdict(list)
    for r in objects:mapping[r['filename']].append(r)
    for v in mapping.values():v.sort(key=lambda r:int(r['label_line_1based']))
    # Original overview sample plus the 10 filename-group middle samples.
    samples=json.loads((REPO/'outputs'/'data_audit'/'20261116T112854775844Z'/'selected_samples.json').read_text(encoding='utf-8'))
    chosen=set(samples+['20150919_174151_image1.png'])
    old_low=json.loads((data/'conversion'/'lowest_iou_objects.json').read_text(encoding='utf-8'))
    detail_pairs={(r['filename'],int(r['source_instance_id'])) for r in old_low if r['roundtrip_iou']<.95}
    # Include hole and multi-component examples even when conversion was almost exact.
    for condition in ['has_holes','components']:
        selected=next(r for r in objects if (r[condition]=='True' if condition=='has_holes' else int(r[condition])>1))
        detail_pairs.add((selected['filename'],int(selected['source_instance_id'])))
    chosen.update(name for name,_ in detail_pairs)
    metrics=[];parser_messages=[];errors=[];detail_saved=[];sample_thumbs=[]
    for number,r in enumerate(files,1):
        name=r['filename'];expected=int(r['source_ids'])
        parsed=verify_image_label((str(source/r['source_image']),str(data/r['label']),'',False,1,0,0,False))
        im_file,lb,shape,segments,_,nm,nf,ne,nc,message=parsed
        if message:parser_messages.append(dict(filename=name,message=message))
        if nc or nm or ne or im_file is None or len(segments)!=expected:
            errors.append(dict(filename=name,message=message,missing=nm,empty=ne,corrupt=nc));continue
        n=max(1000,max(len(s) for s in segments)+1)
        sampled=resample_segments([s.copy() for s in segments],n=n)
        with Image.open(source/r['source_mask']) as im:original=np.asarray(im)
        h,w=original.shape
        original_colors=np.zeros((h,w,3),dtype=np.uint8) if name in chosen else None
        converted_colors=np.zeros((h,w,3),dtype=np.uint8) if name in chosen else None
        difference=np.zeros((h,w,3),dtype=np.uint8) if name in chosen else None
        with Image.open(source/r['source_image']) as im:rgb=np.array(im.convert('RGB')) if name in chosen else None
        for m,poly in zip(mapping[name],sampled):
            oid=int(m['source_instance_id']);ys,xs=np.where(original==oid)
            x0,y0=int(xs.min()),int(ys.min());x1,y1=int(xs.max())+1,int(ys.max())+1
            gt=(original[y0:y1,x0:x1]==oid).astype(np.uint8)
            full=poly*np.array([w,h],dtype=np.float32)
            # Cast in image coordinates exactly as polygon2mask, then translate the crop.
            local=full.astype(np.int32)-np.array([x0,y0],dtype=np.int32)
            pred=polygon2mask(gt.shape,[local.reshape(-1)],downsample_ratio=1)
            union=np.count_nonzero(gt|pred);iou=np.count_nonzero(gt&pred)/union
            added=bool(0)
            extra=(pred!=0)&(gt==0);missing=(pred==0)&(gt!=0)
            metrics.append(dict(filename=name,split=r['split'],source_instance_id=oid,
                                source_area_px=int(gt.sum()),loader_iou=float(iou),added_pixels=int(extra.sum()),missing_pixels=int(missing.sum()),resample_points=n))
            if name in chosen:
                color=np.random.default_rng(oid).integers(60,256,3)
                a=original_colors[y0:y1,x0:x1];b=converted_colors[y0:y1,x0:x1];d=difference[y0:y1,x0:x1]
                a[gt!=0]=color;b[pred!=0]=color;d[extra]=[255,0,0];d[missing]=[0,100,255]
                if (name,oid) in detail_pairs:
                    pad=8;left=max(0,x0-pad);top=max(0,y0-pad);right=min(w,x1+pad);bottom=min(h,y1+pad)
                    gtc=np.zeros((bottom-top,right-left),dtype=np.uint8);prc=gtc.copy()
                    gtc[y0-top:y1-top,x0-left:x1-left]=gt;prc[y0-top:y1-top,x0-left:x1-left]=pred
                    dif=np.zeros((*gtc.shape,3),dtype=np.uint8);dif[(prc!=0)&(gtc==0)]=[255,0,0];dif[(prc==0)&(gtc!=0)]=[0,100,255]
                    fig,axes=plt.subplots(1,4,figsize=(12,4),constrained_layout=True)
                    for ax,img,title in zip(axes,[rgb[top:bottom,left:right],gtc,prc,dif],['RGB crop','Original ID mask','YOLO parser + resample','Red: added / Blue: missing']):
                        ax.imshow(img,cmap='gray' if img.ndim==2 else None,interpolation='nearest');ax.set_title(title,fontsize=9);ax.axis('off')
                    fig.suptitle(f'{Path(name).stem} | ID {oid} | IoU {iou:.4f} | added {int(extra.sum())} / missing {int(missing.sum())}',fontsize=10)
                    dest=out/f'{Path(name).stem}_id{oid}_detail.png';fig.savefig(dest,dpi=160);plt.close(fig);detail_saved.append(dest.name)
        if name in chosen:
            gt_overlay=rgb.copy();pred_overlay=rgb.copy()
            sel=original!=0;gt_overlay[sel]=np.rint(.5*rgb[sel]+.5*original_colors[sel]).astype(np.uint8)
            sel=np.any(converted_colors,axis=2);pred_overlay[sel]=np.rint(.5*rgb[sel]+.5*converted_colors[sel]).astype(np.uint8)
            fig,axes=plt.subplots(1,4,figsize=(15,8),constrained_layout=True)
            for ax,img,title in zip(axes,[rgb,gt_overlay,pred_overlay,difference],['Original RGB','Original GT overlay','YOLO reconstructed overlay','Red: added / Blue: missing']):
                ax.imshow(img);ax.set_title(title,fontsize=10);ax.axis('off')
            fig.suptitle(name);dest=out/f'{Path(name).stem}_comparison.png';fig.savefig(dest,dpi=150);plt.close(fig)
            thumb=Image.fromarray(pred_overlay);thumb.thumbnail((240,450));sample_thumbs.append((name,thumb))
        if number%100==0:print(f'Checked {number}/670 files with Ultralytics',flush=True)
    with (out/'loader_object_audit.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    worst=sorted(metrics,key=lambda r:r['loader_iou'])[:30]
    (out/'lowest_loader_iou.json').write_text(json.dumps(worst,indent=2),encoding='utf-8')
    values=np.array([r['loader_iou'] for r in metrics])
    summary=dict(ultralytics=ultralytics.__version__,files_checked=len(files),files_accepted=len(files)-len(errors),parser_errors=errors,parser_messages=parser_messages,
                 expected_objects=sum(int(r['source_ids']) for r in files),objects_checked=len(metrics),loader_iou_min=float(values.min()),loader_iou_mean=float(values.mean()),
                 objects_below_iou_0_95=int((values<.95).sum()),added_pixels=sum(r['added_pixels'] for r in metrics),missing_pixels=sum(r['missing_pixels'] for r in metrics),
                 selected_comparison_images=len(chosen),detail_images=detail_saved,
                 scope='Installed Ultralytics verify_image_label, 1000+ point normalized resampling, original-resolution polygon2mask. Does NOT certify resized/augmented/downsampled training masks, overlap sorting, or model accuracy.')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    report=['# YOLO 라벨 시각 검수 및 로더 확인','','## 자동 검사',f'- Ultralytics: {ultralytics.__version__}',f'- 파일 수용: {summary["files_accepted"]}/{len(files)}',
            f'- 객체 수: {len(metrics)} / 예상 {summary["expected_objects"]}',f'- 원본 해상도 재구성 평균 IoU: {values.mean():.8f}',f'- 최소 IoU: {values.min():.6f}',
            f'- IoU 0.95 미만: {int((values<.95).sum())}',f'- 추가 / 누락 픽셀 합계: {summary["added_pixels"]} / {summary["missing_pixels"]}',
            '', '## 그림 해석','comparison 그림: 원본 / 원본 정답 / YOLO 좌표 재구성 / 픽셀 차이 순서다.',
            'detail 그림은 작은 객체·여러 조각·내부 구멍 사례를 최근접 보간으로 확대했다. 빨강은 추가 픽셀, 파랑은 누락 픽셀이다.',
            '', '## 범위와 주의','모델 예측 결과가 아니다. 라벨 변환·읽기 경로를 점검했다.',
            '실제 학습의 640 리사이즈, 증강, 마스크 다운샘플링, 객체 겹침 정렬은 다음 학습 동작 확인에서 추가 확인해야 한다.',
            '평균 IoU가 높아도 작은 객체의 개별 차이를 무시하지 말고 detail 그림으로 검토한다.',
            '원본과 라벨을 변경하지 않았고, YAML 작성·학습도 아직 하지 않았다.']
    (out/'시각검수_기록.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True);print('Output:',out)
    if errors or len(metrics)!=summary['expected_objects']:raise RuntimeError('Loader audit failed; see summary')


if __name__=='__main__':main()
