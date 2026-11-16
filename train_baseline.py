"""Train the first full MinneApple baseline with fixed split and explicit settings."""
import csv
from datetime import datetime,timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import YAML,LOGGER
import ultralytics

REPO=Path(__file__).resolve().parent


def audit_masks(data,run):
    rows=[];checks={}
    for split,expected in [('train',518),('val',152)]:
        hyp=get_cfg(overrides={'task':'segment','imgsz':640,'mask_ratio':1,'overlap_mask':True})
        ds=YOLODataset(img_path=str(Path(data['path'])/data[split]),data=dict(data,nc=1),task='segment',
                       imgsz=640,augment=False,cache=False,batch_size=2,hyp=hyp,prefix=f'preflight {split}: ')
        if len(ds)!=expected:raise ValueError(f'Split count mismatch: {split}')
        for i in range(len(ds)):
            item=ds[i];n=len(item['cls']);ids=item['masks'].unique();present=int(((ids>0)&(ids<=n)).sum())
            rows.append(dict(split=split,filename=Path(item['im_file']).name,target_objects=n,
                             nonempty_masks=present,empty_masks=n-present))
        sub=[r for r in rows if r['split']==split]
        checks[split]=dict(images=len(ds),objects=sum(r['target_objects'] for r in sub),
                           empty_masks=sum(r['empty_masks'] for r in sub),affected_images=sum(r['empty_masks']>0 for r in sub))
        print('PREFLIGHT',split,checks[split],flush=True)
    with (run/'mask_preflight.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return checks


def train_attempt(settings,run):
    run.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    record=dict(status='preflight',purpose='First full-data baseline; validation used for model selection, not independent final test',
                started_utc=datetime.now(timezone.utc).isoformat(),settings=settings,
                python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,ultralytics=ultralytics.__version__,
                device_name=torch.cuda.get_device_name(0))
    summary=run/'baseline_summary.json'
    def save():summary.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    save()
    handler=logging.FileHandler(run/'run.log',encoding='utf-8');LOGGER.addHandler(handler)
    try:
        data=YAML.load(str(REPO/'configs'/'minneapple.yaml'))
        record['mask_preflight']=audit_masks(data,run);save()
        total=sum(r['objects'] for r in record['mask_preflight'].values())
        if total!=28183:raise ValueError('Object count changed')
        # Do not silently drop any small target; log all empty representations.
        record['mask_preflight_note']='At original input setting 640, mask_ratio=1, overlap=True, no augmentation; not a claim that every tiny target receives a positive training assignment.'
        freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True)
        (run/'environment.txt').write_text(freeze.stdout,encoding='utf-8')
        model=YOLO(str(REPO/'weights'/'yolov8n-seg.pt'))
        record['weights_start']='COCO pretrained yolov8n-seg.pt, not the smoke checkpoint'
        record['status']='training';save()
        def progress(trainer):
            record.update(epochs_completed=trainer.epoch+1,elapsed_seconds=time.monotonic()-start,
                          current_metrics={str(k):float(v) for k,v in trainer.metrics.items() if isinstance(v,(int,float,np.number))})
            save();print('BASELINE_PROGRESS',json.dumps({'epoch':trainer.epoch+1,'elapsed_seconds':round(record['elapsed_seconds'],1),'metrics':record['current_metrics']}),flush=True)
        model.add_callback('on_fit_epoch_end',progress)
        model.train(**settings)
        with (run/'results.csv').open(encoding='utf-8-sig') as f:results=[{k.strip():v.strip() for k,v in r.items()} for r in csv.DictReader(f)]
        keys=[k for k in results[0] if k.startswith('train/') and 'loss' in k]
        finite=bool(keys) and all(math.isfinite(float(r[k])) for r in results for k in keys)
        if not finite:raise RuntimeError('Non-finite training loss')
        checkpoints={}
        for name in ['best.pt','last.pt']:
            weight=run/'weights'/name
            if not weight.is_file():raise FileNotFoundError(weight)
            loaded=YOLO(str(weight))
            if loaded.names!={0:'apple'}:raise ValueError('Checkpoint class mapping changed')
            checkpoints[name]=hashlib.sha256(weight.read_bytes()).hexdigest()
        record.update(status='completed',epochs_completed=len(results),training_losses_finite=finite,
                      elapsed_seconds=time.monotonic()-start,checkpoint_sha256=checkpoints,
                      first_epoch={k:float(results[0][k]) for k in keys},last_epoch={k:float(results[-1][k]) for k in keys},
                      note='Per-epoch validation values belong to the fixed validation split; independent test inference is a separate next stage.')
        save();print('BASELINE_COMPLETE',str(run),flush=True)
        return record
    except Exception as error:
        record.update(status='failed',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-start);save()
        raise
    finally:LOGGER.removeHandler(handler);handler.close()


def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    spec_path=REPO/'configs'/'train_baseline.yaml'
    settings=dict(data=str(REPO/'configs'/'minneapple.yaml'),epochs=50,patience=10,imgsz=640,batch=1,device=0,workers=0,
                  cache=False,seed=42,deterministic=True,amp=True,optimizer='AdamW',lr0=.001,warmup_epochs=3.,
                  mosaic=0.,mixup=0.,cutmix=0.,copy_paste=0.,degrees=0.,translate=0.,scale=0.,shear=0.,perspective=0.,
                  flipud=0.,fliplr=0.,hsv_h=0.,hsv_s=0.,hsv_v=0.,bgr=0.,close_mosaic=0,mask_ratio=1,overlap_mask=True,
                  save=True,val=True,plots=True)
    if spec_path.exists() and YAML.load(str(spec_path))!=settings:raise ValueError('Baseline config differs; review before changing it')
    YAML.save(str(spec_path),settings,header='# First baseline: fixed split, no augmentation, full-resolution target masks.\n')
    name='baseline_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run=REPO/'outputs'/'training'/name
    actual=dict(settings,project=str(run.parent),name=run.name,exist_ok=True)
    try:record=train_attempt(actual,run)
    except torch.OutOfMemoryError:
        # Preserve the failed attempt and restart from the SAME pretrained weights,
        # adjusting only batch size; no validation result is used to choose this.
        torch.cuda.empty_cache();run=run.with_name(run.name+'_batch1')
        actual.update(batch=1,name=run.name)
        record=train_attempt(actual,run)
    print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
