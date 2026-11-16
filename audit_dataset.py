"""Audit MinneApple file pairs and inspect one sample per filename sequence."""
import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=r'C:\26_2\컴퓨터비전\팀플\detection\train')
    parser.add_argument('--output', default=str(REPO / 'outputs' / 'data_audit'))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    images = {p.name:p for p in (root/'images').glob('*.png')}
    masks = {p.name:p for p in (root/'masks').glob('*.png')}
    if not images or not masks:
        raise ValueError('Image/mask PNG directories are empty or missing')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = Path(args.output).resolve()/stamp
    out.mkdir(parents=True, exist_ok=False)
    missing_masks = sorted(images.keys()-masks.keys())
    missing_images = sorted(masks.keys()-images.keys())
    rows = []
    groups = {}
    for number,name in enumerate(sorted(images.keys() & masks.keys()),1):
        match = re.fullmatch(r'(.+)_image(\d+)\.png', name)
        sequence = match.group(1) if match else Path(name).stem
        frame = int(match.group(2)) if match else 0
        row = dict(filename=name, sequence=sequence, frame_index=frame, status='ok', error='')
        try:
            with Image.open(images[name]) as im:
                im.load()
                row['image_width'],row['image_height'] = im.size
                row['image_mode'] = im.mode
            with Image.open(masks[name]) as im:
                im.load()
                row['mask_width'],row['mask_height'] = im.size
                row['mask_mode'] = im.mode
                mask = np.asarray(im)
            if (row['image_width'],row['image_height']) != (row['mask_width'],row['mask_height']):
                raise ValueError('Dimension mismatch')
            if mask.ndim != 2 or not np.issubdtype(mask.dtype,np.integer):
                raise ValueError('Expected single-channel integer instance mask')
            ids,counts=np.unique(mask,return_counts=True)
            row['mask_dtype']=str(mask.dtype)
            row['instance_id_count']=int(np.count_nonzero(ids))
            row['has_background_zero']=bool(np.any(ids==0))
            row['foreground_ratio']=float(np.count_nonzero(mask)/mask.size)
            areas=counts[ids!=0]
            row['min_instance_area_px']=int(areas.min()) if len(areas) else 0
            row['max_instance_area_px']=int(areas.max()) if len(areas) else 0
            groups.setdefault(sequence,[]).append((frame,name))
        except Exception as error:
            row['status']='error';row['error']=str(error)
        rows.append(row)
        if number % 100 == 0: print(f'Checked {number} pairs',flush=True)
    fields=['filename','sequence','frame_index','status','error','image_width','image_height','image_mode','mask_width','mask_height','mask_mode','mask_dtype','instance_id_count','has_background_zero','foreground_ratio','min_instance_area_px','max_instance_area_px']
    with (out/'file_audit.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    selected=[]
    for sequence,items in sorted(groups.items()):
        items.sort()
        selected.append(items[len(items)//2][1])
    (out/'selected_samples.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    for name in selected:
        print(f'Inspecting {name}',flush=True)
        subprocess.run([sys.executable,str(REPO/'inspect_sample.py'),'--root',str(root),'--name',name,'--output',str(out/'samples')],check=True)
    # Contact sheet is a diagnostic display of real images and masks.
    if selected:
        tile_w,tile_h=260,500
        sheet=Image.new('RGB',(tile_w*5,tile_h*((len(selected)+4)//5)),'white')
        draw=ImageDraw.Draw(sheet)
        for i,name in enumerate(selected):
            x=(i%5)*tile_w;y=(i//5)*tile_h
            with Image.open(out/'samples'/Path(name).stem/'overlay.png') as im:
                im=im.convert('RGB');im.thumbnail((250,450))
                sheet.paste(im,(x+(tile_w-im.width)//2,y+38))
            draw.text((x+5,y+4),name.replace('_image','\nimage'),fill='black')
        sheet.save(out/'contact_sheet.png')
    errors=[r for r in rows if r['status']!='ok']
    okay=[r for r in rows if r['status']=='ok']
    summary=dict(created_utc=stamp,root=str(root),image_count=len(images),mask_count=len(masks),matched_pairs=len(rows),missing_masks=missing_masks,missing_images=missing_images,error_pairs=errors,readable_and_structurally_valid_pairs=len(okay),image_sizes=dict(Counter(f"{r['image_width']}x{r['image_height']}" for r in okay)),sequence_counts=dict(sorted(Counter(r['sequence'] for r in rows).items())),sample_selection='Numeric middle frame from each filename-prefix group; grouping does not prove scene independence.',selected_samples=selected,instance_id_count_range=[min(r['instance_id_count'] for r in okay),max(r['instance_id_count'] for r in okay)] if okay else [],no_zero_background=[r['filename'] for r in okay if not r['has_background_zero']],empty_masks=[r['filename'] for r in okay if not r['instance_id_count']],scope='File readability, dimensions and annotation-ID structure only. No certification of semantic accuracy or completeness; no training/validation split or model training performed.')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    report=['# MinneApple 追加サンプル検査'.replace('追加サンプル検査','추가 샘플 검수'),'','## 자동 확인 결과',f"- 이미지: {len(images)}, 마스크: {len(masks)}, 파일명 대응: {len(rows)}쌍",f"- 구조·읽기 오류: {len(errors)}쌍",f"- 마스크 누락: {len(missing_masks)}, 이미지 누락: {len(missing_images)}",f"- 파일명 기반 그룹: {len(groups)}, 선택 샘플: {len(selected)}장",'', '## 샘플 선택','파일명 접두사별로 프레임 번호를 숫자 정렬하여 가운데 샘플을 선택했다. 장면 독립성·대표성을 보장하는 선택은 아니다.','', '## 육안 검수 체크리스트','- [ ] 색상 영역과 사과 경계가 대응하는지 확인','- [ ] 붙은 사과가 서로 다른 ID로 나뉘는지 확인','- [ ] 잎 가림·작은 사과·어두운 영역의 누락 의심 사례 기록','- [ ] 같은 ID의 떨어진 조각 등 변환 시 주의할 사례 확대 확인','','## 결과 위치','- file_audit.csv: 전체 쌍의 형식·크기·ID 통계','- selected_samples.json: 재현 가능한 샘플 목록','- contact_sheet.png: 선택 샘플 오버레이 모음','- samples/: 샘플별 비교 그림 및 통계','', '## 한계','자동 검사를 통과해도 라벨 의미·경계·누락의 정확성이 보장되는 것은 아니다. 학습·검증 분할과 학습은 아직 하지 않았다.']
    (out/'검수기록.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print(f'Output: {out}',flush=True)


if __name__=='__main__':
    main()