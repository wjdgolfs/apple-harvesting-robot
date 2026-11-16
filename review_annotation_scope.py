"""Compare nearby MinneApple frames to review foreground annotation scope."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=r'C:\26_2\컴퓨터비전\팀플\detection\train')
    parser.add_argument('--output',default=str(ROOT/'outputs'/'data_audit'/'20261116T112854775844Z'/'annotation_scope'))
    args=parser.parse_args()
    data=Path(args.root); out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    frames=[221,236,246,251,256,276]
    rows=[];panels=[]
    for frame in frames:
        name=f'20150919_174730_image{frame}.png'
        with Image.open(data/'images'/name) as im: rgb=np.array(im.convert('RGB'))
        with Image.open(data/'masks'/name) as im: mask=np.array(im)
        if mask.ndim!=2 or mask.shape!=rgb.shape[:2]: raise ValueError(name)
        ids=np.unique(mask);ids=ids[ids!=0]
        colors=np.zeros_like(rgb);overlay=rgb.copy()
        for oid in ids:colors[mask==oid]=np.random.default_rng(int(oid)).integers(60,256,3)
        fg=mask!=0
        overlay[fg]=np.rint(.5*rgb[fg]+.5*colors[fg]).astype(np.uint8)
        overlay=Image.fromarray(overlay);draw=ImageDraw.Draw(overlay)
        for oid in ids:
            ys,xs=np.where(mask==oid)
            bbox=[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())]
            draw.rectangle(bbox,outline='magenta',width=3)
            draw.text((max(0,bbox[0]-20),max(0,bbox[1]-12)),str(int(oid)),fill='white',stroke_width=1,stroke_fill='black')
            rows.append(dict(filename=name,id=int(oid),area_px=int(len(xs)),xmin=bbox[0],ymin=bbox[1],xmax_inclusive=bbox[2],ymax_inclusive=bbox[3]))
        overlay.save(out/f'image{frame}_overlay.png')
        a=Image.fromarray(rgb).resize((360,640));b=overlay.resize((360,640))
        panel=Image.new('RGB',(720,675),'white');d=ImageDraw.Draw(panel)
        d.text((8,8),f'frame {frame}: original | GT overlay + boxes ({len(ids)} IDs)',fill='black')
        panel.paste(a,(0,35));panel.paste(b,(360,35));panels.append(panel)
    sheet=Image.new('RGB',(720*3,675*2),'white')
    for i,panel in enumerate(panels):sheet.paste(panel,((i%3)*720,(i//3)*675))
    sheet.save(out/'neighbor_comparison.png')
    with (out/'objects.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['filename','id','area_px','xmin','ymin','xmax_inclusive','ymax_inclusive']);w.writeheader();w.writerows(rows)
    notes=['# 주석 범위 확인: 인접 프레임 비교','','- 검토 시퀀스: 20150919_174730','- 프레임: 221, 236, 246, 251, 256, 276','- 마젠타 박스와 번호는 정답 영역의 위치를 찾기 위한 보조 표시이며 모델 예측이 아니다.','','## 프레임별 정답 ID 수']
    for frame in frames:
        name=f'20150919_174730_image{frame}.png'
        count=sum(r['filename']==name for r in rows)
        notes.append(f'- image{frame}: {count}개')
    notes.extend(['','## 원 논문의 주석 정책','MinneApple 원 논문 IV Image Annotation은 전경 나무의 과일에 다각형 주석을 하고, 배경 나무와 땅의 과일은 주석하지 않았다고 설명한다.','출처: https://arxiv.org/html/1909.06441','','## 해석 시 주의','사진 전체에 보이는 모든 사과 수와 정답 ID 수는 같지 않을 수 있다. 주석이 적다는 이유만으로 오류로 판정하거나 이미지를 버리지 않는다.','사진에서 어느 나무가 전경인지 확인하고, 정책상 제외와 실제 주석 누락 가능성을 구분해야 한다. 모든 경계·가림 영역을 검증한 결과는 아니다.','','## 이번 단계에서 하지 않은 작업','주석 수정, 원본 삭제, 데이터 분할, YOLO 라벨 변환, 학습은 하지 않았다.'])
    (out/'주석범위_검토.md').write_text('\n'.join(notes),encoding='utf-8')
    print('\n'.join(notes));print(f'Output: {out.resolve()}')


if __name__=='__main__':main()