"""Build fixed group-disjoint MinneApple train/validation manifests (no copies)."""
import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
import numpy as np
from PIL import Image

REPO=Path(__file__).resolve().parent


def dhash(rgb):
    small=np.asarray(rgb.convert('L').resize((9,8),Image.Resampling.LANCZOS))
    bits=small[:,1:]>small[:,:-1]
    return int.from_bytes(np.packbits(bits).tobytes(),'big')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=r'C:\26_2\컴퓨터비전\팀플\detection\train')
    parser.add_argument('--config',default=str(REPO/'configs'/'minneapple_split_v1.json'))
    parser.add_argument('--output',default=str(REPO/'splits'/'minneapple_v1'))
    args=parser.parse_args()
    root=Path(args.root).resolve();config_path=Path(args.config)
    config=json.loads(config_path.read_text(encoding='utf-8'))
    out=Path(args.output).resolve()
    if out.exists():raise FileExistsError(f'Output exists: {out}. Choose a new output directory to preserve manifests.')
    images={p.name:p for p in (root/'images').glob('*.png')}
    masks={p.name:p for p in (root/'masks').glob('*.png')}
    if not images or images.keys()!=masks.keys():raise ValueError('Missing/mismatched image-mask files')
    val_groups=set(config['validation_sequences']);train_groups=set(config['training_sequences'])
    if train_groups & val_groups:raise ValueError('Sequence overlap')
    rows=[];hashes={};seen_groups=set()
    for index,name in enumerate(sorted(images),1):
        match=re.fullmatch(r'(.+)_image(\d+)\.png',name)
        if match is None:raise ValueError(f'Unexpected filename {name}')
        group=match.group(1);seen_groups.add(group)
        if group not in train_groups|val_groups:raise ValueError(f'Unassigned group {group}')
        split='val' if group in val_groups else 'train'
        raw=images[name].read_bytes()
        image_hash=hashlib.sha256(raw).hexdigest()
        with Image.open(images[name]) as im:
            im.load();size=im.size;hashes[name]=dhash(im)
        mask_raw=masks[name].read_bytes()
        with Image.open(masks[name]) as im:
            im.load();mask=np.asarray(im)
            if im.size!=size or mask.ndim!=2 or not np.issubdtype(mask.dtype,np.integer):raise ValueError(name)
        count=int(np.count_nonzero(np.unique(mask)))
        rows.append(dict(split=split,sequence=group,frame_index=int(match.group(2)),filename=name,
                         image='images/'+name,mask='masks/'+name,width=size[0],height=size[1],
                         instance_id_count=count,image_sha256=image_hash,mask_sha256=hashlib.sha256(mask_raw).hexdigest(),
                         image_dhash64=f'{hashes[name]:016x}'))
        if index%100==0:print(f'Validated and hashed {index} pairs',flush=True)
    if seen_groups!=train_groups|val_groups:raise ValueError('Configured group absent from data')
    trains=[r for r in rows if r['split']=='train'];vals=[r for r in rows if r['split']=='val']
    overlap=set(r['image_sha256'] for r in trains)&set(r['image_sha256'] for r in vals)
    if overlap:raise ValueError(f'Exact image duplicates across splits: {len(overlap)}; revise group assignment')
    if not trains or not vals:raise ValueError('Empty split')
    if len(rows)!=config['expected_pairs']:raise ValueError('Pair count changed; review dataset version')
    nearest=[]
    for v in vals:
        t=min(trains,key=lambda r:(hashes[v['filename']]^hashes[r['filename']]).bit_count())
        distance=(hashes[v['filename']]^hashes[t['filename']]).bit_count()
        nearest.append(dict(val=v['filename'],train=t['filename'],hamming_distance=distance))
    nearest.sort(key=lambda r:(r['hamming_distance'],r['val'],r['train']))
    out.mkdir(parents=True,exist_ok=False)
    with (out/'manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for split,items in [('train',trains),('val',vals)]:
        (out/f'{split}_images.txt').write_text(''.join(r['image']+'\n' for r in items),encoding='utf-8')
        (out/f'{split}_masks.txt').write_text(''.join(r['mask']+'\n' for r in items),encoding='utf-8')
    (out/'nearest_cross_split_pairs.json').write_text(json.dumps(nearest[:20],indent=2),encoding='utf-8')
    group_counts=Counter(r['sequence'] for r in rows)
    stats={}
    for split,items in [('train',trains),('val',vals)]:
        stats[split]={'images':len(items),'image_ratio':len(items)/len(rows),'annotation_ids':sum(r['instance_id_count'] for r in items),
                      'sequences':sorted({r['sequence'] for r in items}),
                      'dates':dict(sorted(Counter(r['sequence'][:8] for r in items).items()))}
    summary={'version':config['version'],'root_at_creation':str(root),'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
             'total_pairs':len(rows),'split_stats':stats,'group_counts':dict(sorted(group_counts.items())),
             'sequence_overlap':[],'cross_split_exact_image_hash_overlap':len(overlap),
             'nearest_cross_split_dhash_distance':nearest[0]['hamming_distance'],
             'dhash_pairs_distance_le_6':sum(r['hamming_distance']<=6 for r in nearest),
             'policy':config['policy'],'limitations':config['limitations']}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    report=['# MinneApple 学習・検証分割'.replace('学習・検証分割','학습·검증 데이터 분할'),'','## 분할 결과',
            f'- 학습: {len(trains)}장 ({len(trains)/len(rows):.1%}), 검증: {len(vals)}장 ({len(vals)/len(rows):.1%})',
            '- 같은 파일명 접두사(촬영 시퀀스)는 한쪽에만 배정했다.',
            '- 공식 test 331장은 이번 분할·설정 조정에 사용하지 않았다.',
            '- 원본 이동·복사·변경 없이 상대 경로 목록을 저장했다.','',
            '## 시퀀스 배정','| 시퀀스 | 분할 | 이미지 수 |','|---|---|---:|']
    for group,count in sorted(group_counts.items()):report.append(f'| {group} | {"val" if group in val_groups else "train"} | {count} |')
    report.extend(['','## 검사',f'- 시퀀스 중복: 0 / 파일명 대응: {len(rows)}쌍 / 분할 간 동일 이미지 SHA-256 중복: {len(overlap)}',
                   f"- 검증 이미지별 가장 가까운 학습 이미지 dHash 중 최소 거리: {nearest[0]['hamming_distance']}",
                   '- dHash는 장면 유사성의 보조 검사다. 같은 나무·배경의 누출이 없다고 보증하지 않는다.',
                   '', '## 생성 파일', '- manifest.csv: 이미지·마스크 대응, 배정, 크기, 주석 수, 원본 해시',
                   '- train_images.txt / val_images.txt: train 데이터 루트를 기준으로 한 이미지 상대 경로',
                   '- train_masks.txt / val_masks.txt: 대응하는 원본 마스크 상대 경로',
                   '- summary.json: 배정 통계 및 검사 결과', '- nearest_cross_split_pairs.json: 수동 확인용 유사 이미지 후보',
                   '', '## 다음 단계', '원본 마스크를 YOLO 다각형 라벨로 변환한다. 이번 목록은 아직 YOLO 라벨·YAML이 아니며 학습을 바로 시작할 수 없다.',
                   '', '## 한계', *['- '+x for x in config['limitations']]])
    (out/'분할결과.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2));print(f'Output: {out}',flush=True)


if __name__=='__main__':main()
