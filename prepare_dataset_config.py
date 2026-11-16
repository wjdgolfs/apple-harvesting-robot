"""Stage original images by fixed split, create dataset YAML and verify YOLO dataset loading."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from ultralytics.utils import YAML
import ultralytics
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset

REPO=Path(__file__).resolve().parent


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default=r'C:\26_2\컴퓨터비전\팀플\detection\train')
    parser.add_argument('--dataset',default=r'C:\26_2\컴퓨터비전\팀플\minneapple_yolo_v1')
    parser.add_argument('--manifest',default=str(REPO/'splits'/'minneapple_v1'/'manifest.csv'))
    parser.add_argument('--yaml',default=str(REPO/'configs'/'minneapple.yaml'))
    args=parser.parse_args();source=Path(args.source).resolve();data_root=Path(args.dataset).resolve();manifest=Path(args.manifest)
    with manifest.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    conversion=data_root/'conversion'
    summary=json.loads((conversion/'summary.json').read_text(encoding='utf-8'))
    if summary['split_manifest_sha256']!=sha256(manifest):raise ValueError('Split manifest changed since conversion')
    with (conversion/'label_manifest.csv').open(encoding='utf-8-sig') as f:label_rows={r['filename']:r for r in csv.DictReader(f)}
    if len(rows)!=670 or len(label_rows)!=670:raise ValueError('Unexpected manifest counts')
    expected={'train':{'images':518,'objects':22082},'val':{'images':152,'objects':6101}}
    for split in expected:(data_root/'images'/split).mkdir(parents=True,exist_ok=True)
    copied=reused=0
    for i,r in enumerate(rows,1):
        split=r['split'];name=r['filename']
        if split not in expected:raise ValueError(split)
        src=source/r['image'];dest=data_root/'images'/split/name
        label=data_root/'labels'/split/(Path(name).stem+'.txt')
        lm=label_rows[name]
        if lm['split']!=split or sha256(label)!=lm['label_sha256']:raise ValueError(f'Changed label {name}')
        if sha256(src)!=r['image_sha256']:raise ValueError(f'Changed source image {name}')
        if dest.exists():
            if sha256(dest)!=r['image_sha256']:raise ValueError(f'Refusing to replace different staged image: {dest}')
            reused+=1
        else:
            shutil.copy2(src,dest)
            if sha256(dest)!=r['image_sha256']:raise ValueError(f'Copy verification failed: {dest}')
            copied+=1
        if i%100==0:print(f'Staged and hash-verified {i}/670 images',flush=True)
    for split,e in expected.items():
        staged=set(p.name for p in (data_root/'images'/split).glob('*.png'))
        wanted={r['filename'] for r in rows if r['split']==split}
        if staged!=wanted or len(wanted)!=e['images']:raise ValueError(f'Unexpected staged files in {split}')
    spec={'path':data_root.as_posix(),'train':'images/train','val':'images/val','names':{0:'apple'}}
    target=Path(args.yaml).resolve();target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and YAML.load(str(target))!=spec:raise ValueError(f'Refusing to overwrite different YAML: {target}')
    YAML.save(str(target),spec,header='# MinneApple fixed sequence split v1. Official test set intentionally omitted.\n')
    resolved=check_det_dataset(str(target),autodownload=False)
    if resolved['names']!={0:'apple'}:raise ValueError('Unexpected class mapping')
    hyp=get_cfg(overrides={'task':'segment','imgsz':640,'cache':False,'workers':0})
    checks={}
    for split,e in expected.items():
        ds=YOLODataset(img_path=resolved[split],data=resolved,task='segment',imgsz=640,augment=False,cache=False,
                       batch_size=2,hyp=hyp,prefix=f'{split}: ')
        files={Path(p).name for p in ds.im_files}
        wanted={r['filename'] for r in rows if r['split']==split}
        objects=sum(len(label['cls']) for label in ds.labels)
        segments=sum(len(label['segments']) for label in ds.labels)
        if files!=wanted or len(ds)!=e['images'] or objects!=e['objects'] or segments!=objects:
            raise ValueError(f'Loader count mismatch: {split}, images={len(ds)}, objects={objects}, segments={segments}')
        checks[split]={'images':len(ds),'objects':objects,'segments':segments,'path':resolved[split]}
        print(f'{split}: images={len(ds)}, objects={objects}, segments={segments}',flush=True)
    record={'yaml':str(target),'yaml_sha256':sha256(target),'dataset_root':str(data_root),'ultralytics':ultralytics.__version__,
            'copied_images':copied,'reused_hash_verified_images':reused,'dataset_loader_checks':checks,'names':resolved['names'],
            'official_test_included':False,'training_executed':False,
            'notes':['Images are independent copies, not hard links. Source data unchanged.',
                     'YOLO parser/dataset construction verified without training, resizing samples, augmentation, or gradient updates.',
                     'Absolute path is machine-specific; edit only path when relocating the dataset.',
                     'Small-mask resizing/downsampling behavior remains to be checked in the short training stage.']}
    (conversion/'dataset_config_check.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
