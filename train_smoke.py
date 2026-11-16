"""Reproducible 32/8-image, 3-epoch YOLOv8 segmentation smoke training."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import YAML, LOGGER

REPO=Path(__file__).resolve().parent


def build_subset():
    full_spec=YAML.load(str(REPO/'configs'/'minneapple.yaml'))
    data_root=Path(full_spec['path'])
    with (REPO/'splits'/'minneapple_v1'/'manifest.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    grouped=defaultdict(list)
    for r in rows:grouped[r['sequence']].append(r)
    force={'20150921_131346':'20150921_131346_image151.png',
           '20150921_131833':'20150921_131833_image181.png',
           '20150921_132038':'20150921_132038_image361.png'}
    selected=[]
    for sequence,items in sorted(grouped.items()):
        items.sort(key=lambda r:int(r['frame_index']))
        indices=np.linspace(0,len(items)-1,4,dtype=int)
        part=[items[i] for i in indices]
        if sequence in force and force[sequence] not in {r['filename'] for r in part}:
            part[1]=next(r for r in items if r['filename']==force[sequence])
        selected.extend(part)
    if len(selected)!=40 or len({r['filename'] for r in selected})!=40:raise ValueError('Subset selection error')
    split_dir=REPO/'splits'/'minneapple_smoke_v1';split_dir.mkdir(exist_ok=True)
    for split,count in [('train',32),('val',8)]:
        items=[r for r in selected if r['split']==split]
        if len(items)!=count:raise ValueError(split)
        paths=[(data_root/'images'/split/r['filename']).as_posix() for r in items]
        for item,path in zip(items,paths):
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=item['image_sha256']:raise ValueError(path)
        target=split_dir/f'{split}_images.txt';content='\n'.join(paths)+'\n'
        if target.exists() and target.read_text(encoding='utf-8')!=content:raise ValueError(f'Subset already differs: {target}')
        target.write_text(content,encoding='utf-8')
    target=split_dir/'manifest.csv'
    if not target.exists():
        with target.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(selected[0]));writer.writeheader();writer.writerows(selected)
    spec={'path':data_root.as_posix(),'train':(split_dir/'train_images.txt').as_posix(),
          'val':(split_dir/'val_images.txt').as_posix(),'names':{0:'apple'}}
    yaml_path=REPO/'configs'/'minneapple_smoke.yaml'
    if yaml_path.exists() and YAML.load(str(yaml_path))!=spec:raise ValueError('Smoke YAML already differs')
    YAML.save(str(yaml_path),spec,header='# Diagnostic subset only; never use smoke results as final accuracy.\n')
    return yaml_path,dict(spec,nc=1),selected


def audit_masks(data,run):
    rows=[]
    for ratio in [4,1]:
        hyp=get_cfg(overrides={'task':'segment','imgsz':640,'mask_ratio':ratio,'overlap_mask':True})
        for split in ['train','val']:
            ds=YOLODataset(img_path=data[split],data=data,task='segment',imgsz=640,augment=False,
                           cache=False,batch_size=2,hyp=hyp,prefix=f'audit {split} ratio={ratio}: ')
            for i in range(len(ds)):
                item=ds[i];n=len(item['cls']);ids=item['masks'].unique()
                present=int(((ids>0)&(ids<=n)).sum().item())
                rows.append({'split':split,'filename':Path(item['im_file']).name,'mask_ratio':ratio,
                             'target_objects':n,'nonempty_masks':present,'empty_masks':n-present,
                             'image_shape':str(tuple(item['img'].shape)),'mask_shape':str(tuple(item['masks'].shape))})
    with (run/'mask_resolution_audit.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={str(ratio):{'target_objects':sum(r['target_objects'] for r in rows if r['mask_ratio']==ratio),
                        'empty_masks':sum(r['empty_masks'] for r in rows if r['mask_ratio']==ratio),
                        'affected_images':sum(r['empty_masks']>0 for r in rows if r['mask_ratio']==ratio)} for ratio in [4,1]}
    (run/'mask_resolution_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('Mask resolution audit:',json.dumps(summary),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--device',default='0')
    args=parser.parse_args()
    if args.device!='cpu' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    yaml_path,data,selected=build_subset()
    name='smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run=REPO/'outputs'/'training'/name;run.mkdir(parents=True,exist_ok=False)
    handler=logging.FileHandler(run/'run.log',encoding='utf-8');LOGGER.addHandler(handler)
    try:
        mask_audit=audit_masks(data,run)
        # Keep input transforms deterministic here to isolate data/GPU/loss/save behavior.
        # Training and validation remain distinct original sequence groups.
        settings=dict(data=str(yaml_path),epochs=args.epochs,imgsz=640,batch=2,device=args.device,workers=0,
                      cache=False,seed=42,deterministic=True,amp=False,optimizer='AdamW',lr0=.001,warmup_epochs=0,
                      mosaic=0.,mixup=0.,cutmix=0.,copy_paste=0.,degrees=0.,translate=0.,scale=0.,shear=0.,
                      perspective=0.,flipud=0.,fliplr=0.,hsv_h=0.,hsv_s=0.,hsv_v=0.,bgr=0.,
                      close_mosaic=0,mask_ratio=4,overlap_mask=True,save=True,val=True,plots=True,
                      project=str(run.parent),name=run.name,exist_ok=True)
        record={'status':'started','purpose':'Pipeline smoke check, not final model performance',
                'subset_train_images':32,'subset_val_images':8,'mask_resolution_audit':mask_audit,
                'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'ultralytics':ultralytics.__version__,
                'device_name':torch.cuda.get_device_name(int(args.device)) if args.device!='cpu' else 'CPU','settings':settings}
        summary_file=run/'smoke_summary.json';summary_file.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True)
        (run/'environment.txt').write_text(freeze.stdout,encoding='utf-8')
        model=YOLO(str(REPO/'weights'/'yolov8n-seg.pt'))
        model.train(**settings)
        with (run/'results.csv').open(encoding='utf-8-sig') as f:
            results=[{k.strip():v.strip() for k,v in r.items()} for r in csv.DictReader(f)]
        loss_keys=[k for k in results[0] if k.startswith('train/') and 'loss' in k]
        finite=bool(loss_keys) and all(math.isfinite(float(r[k])) for r in results for k in loss_keys)
        if len(results)!=args.epochs or not finite:raise RuntimeError('Epoch count or training-loss check failed')
        for weight in ['best.pt','last.pt']:
            if not (run/'weights'/weight).is_file():raise RuntimeError(f'Missing checkpoint {weight}')
        record.update(status='completed',epochs_completed=len(results),training_losses_finite=finite,
                      first_epoch_losses={k:float(results[0][k]) for k in loss_keys},
                      last_epoch_losses={k:float(results[-1][k]) for k in loss_keys},
                      note='Mask targets can disappear at 640/mask_ratio=4. Audit records the issue; no source labels removed. Full training settings need review before the next stage.')
        summary_file.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(record,ensure_ascii=False,indent=2));print('Run directory:',run,flush=True)
    except Exception as error:
        (run/'failure.json').write_text(json.dumps({'type':type(error).__name__,'message':str(error)},ensure_ascii=False,indent=2),encoding='utf-8')
        raise
    finally:
        LOGGER.removeHandler(handler);handler.close()


if __name__=='__main__':main()
