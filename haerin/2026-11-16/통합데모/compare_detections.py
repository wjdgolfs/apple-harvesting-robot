"""Compare detector masks with the dataset's visible-apple polygons on the demo images.

For every image: match detections to ground-truth apples by mask IoU (greedy, highest IoU first, IoU >= 0.5),
then run the demo's diameter and ripeness steps on both masks of each matched apple. Reports
  detection: ground-truth apples, detections, matched, missed, detections without an outline, mean IoU of matches
  effect  : diameter change (detector mask - ground-truth mask), held/used changes, ripeness decision changes
Detections are <stem>.json files in the demo hand-off format (see demo.py).
"""

import argparse
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

import demo

locate = demo.module_from_file("locate", demo.BASE / "익음도보강/locate.py")


def rasterize(polygon, shape):
    mask, (left, top) = demo.measure.rasterize([tuple(p) for p in polygon], shape)
    full = np.zeros(shape, bool)
    full[top:top + mask.shape[0], left:left + mask.shape[1]] = mask
    return full


def match(gt_masks, det_masks, threshold=0.5):
    pairs = []
    for i, g in enumerate(gt_masks):
        for j, d in enumerate(det_masks):
            inter = np.logical_and(g, d).sum()
            if inter:
                pairs.append((inter / np.logical_or(g, d).sum(), i, j))
    used_g, used_d, matches = set(), set(), []
    for iou, i, j in sorted(pairs, reverse=True):
        if iou < threshold:
            break
        if i not in used_g and j not in used_d:
            used_g.add(i)
            used_d.add(j)
            matches.append((i, j, float(iou)))
    return matches


def draw_overlay(rgb, gt, det, matches, title, path):
    """Ground-truth outlines (cyan) and detector outlines (magenta); unmatched ones on both sides are labelled
    (an unmatched detection may be a real apple that the dataset did not annotate)."""
    from PIL import ImageDraw, ImageFont
    box = Image.eval(rgb, lambda v: 255 if v > 0 else 0).convert("L").getbbox() or (0, 0, *rgb.size)
    scale = 900 / max(box[2] - box[0], box[3] - box[1])
    view = rgb.crop(box).resize((round((box[2] - box[0]) * scale), round((box[3] - box[1]) * scale)))
    canvas = Image.new("RGB", (view.width + 40, view.height + 110), "#fcfcfb")
    canvas.paste(view, (20, 70))
    d = ImageDraw.Draw(canvas)
    font, small = ImageFont.truetype(demo.FONT, 22), ImageFont.truetype(demo.FONT, 15)
    d.text((20, 18), title, font=font, fill="#0b0b0b")
    to = lambda p: ((p[0] - box[0]) * scale + 20, (p[1] - box[1]) * scale + 70)
    matched_g, matched_d = {i for i, _, _ in matches}, {j for _, j, _ in matches}
    for i, g in enumerate(gt):
        pts = [to(p) for p in g["polygon"]]
        d.line(pts + pts[:1], fill="#16c8d8", width=3)
        if i not in matched_g:
            d.text((min(p[0] for p in pts), min(p[1] for p in pts) - 18), "놓침", font=small, fill="#16c8d8")
    for j, t in enumerate(det):
        pts = [to(p) for p in t["polygon"]]
        d.line(pts + pts[:1], fill="#e040c0", width=2)
        if j not in matched_d:
            d.text((min(p[0] for p in pts), max(p[1] for p in pts) + 2), "윤곽 없는 검출", font=small, fill="#e040c0")
    d.text((20, canvas.height - 32), "청록 = 데이터셋의 보이는 사과 윤곽, 자홍 = 검출기 윤곽 · 이미지 Gené-Mola et al., AmodalAppleSize_RGB-D (CC BY 4.0)",
           font=small, fill="#52514e")
    canvas.save(path, optimize=True)


def case_sheet(rows, crops, path):
    """Per matched apple: crop from the dataset outline vs the detector outline, true label and both predictions.
    Changed decisions first, then ripe-called-unripe, unripe-called-ripe and correct unripe cases (AGENTS.md)."""
    from PIL import ImageDraw, ImageFont
    def kind(r):
        if r["gt_ripeness"] != r["det_ripeness"]:
            return 0, "윤곽에 따라 판정이 바뀜"
        if r["truth_label"] == "ripe" and r["det_ripeness"] == "unripe":
            return 1, "익음을 미숙으로 놓침"
        if r["truth_label"] == "unripe" and r["det_ripeness"] == "ripe":
            return 2, "미숙을 익음으로 판정"
        if r["truth_label"] == "unripe" and r["det_ripeness"] == "unripe":
            return 3, "미숙을 미숙으로"
        if r["truth_label"] == "ripe" and r["det_ripeness"] == "ripe":
            return 4, "익음을 익음으로"
        return 5, "그 밖(보류 또는 라벨 없음)"
    chosen = sorted(rows, key=lambda r: (kind(r)[0], r["image"], r["gt_id"]))[:12]
    tile = 130
    font, small = ImageFont.truetype(demo.FONT, 22), ImageFont.truetype(demo.FONT, 14)
    cols = 3
    w, h = 2 * tile + 230, 2 * tile + 56
    sheet = Image.new("RGB", (cols * w + 30, 70 + ((len(chosen) + cols - 1) // cols) * h + 40), "#fcfcfb")
    d = ImageDraw.Draw(sheet)
    d.text((20, 18), "같은 사과: 데이터셋 윤곽으로 자른 입력(왼쪽) 대 검출기 윤곽으로 자른 입력(오른쪽)", font=font, fill="#0b0b0b")
    for n, r in enumerate(chosen):
        x, y = 20 + (n % cols) * w, 66 + (n // cols) * h
        a, b, where = crops[(r["image"], r["gt_id"])]
        sheet.paste(where, (x + 3, y + tile + 18))
        d.text((x + tile + 14, y + tile + 22), "← 원본 속 위치", font=small, fill="#52514e")
        for k, (img, pred) in enumerate(((a, r["gt_ripeness"]), (b, r["det_ripeness"]))):
            frame = Image.new("RGB", (tile + 6, tile + 6), demo.COLOR[pred])
            frame.paste(img.resize((tile, tile)), (3, 3))
            sheet.paste(frame, (x + k * (tile + 10), y))
        tx = x + 2 * (tile + 10) + 4
        truth = demo.KO.get(r["truth_label"], "라벨 없음")
        seen = " (학습 사과)" if "train" in r["ripeness_split_of_apple"] else " (검증 사과)" if "val" in r["ripeness_split_of_apple"] else ""
        for m, text in enumerate((kind(r)[1], f"정답 {truth}{seen}", f"데이터셋 윤곽 → {demo.KO[r['gt_ripeness']]}",
                                  f"검출 윤곽 → {demo.KO[r['det_ripeness']]}", f"IoU {r['iou']:.2f}",
                                  f"{r['image'].replace('.png', '')}", f"사과 {r['gt_id']}")):
            d.text((tx, y + 2 + 19 * m), text, font=small, fill="#0b0b0b" if m < 4 else "#52514e")
    d.text((20, sheet.height - 30), "테두리 색 = 미세조정 모델 예측(빨강 익음, 초록 미숙, 회색 보류) · 이미지 Gené-Mola et al., AmodalAppleSize_RGB-D (CC BY 4.0) · 라벨 Zhu et al.",
           font=small, fill="#52514e")
    sheet.save(path, optimize=True)


def crop_of(rgb, polygon):
    xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
    box = (max(0, min(xs)), max(0, min(ys)), min(rgb.width, max(xs) + 1), min(rgb.height, max(ys) + 1))
    return demo.prepare.fit_square(rgb.crop(box))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", nargs="+", required=True)
    parser.add_argument("--detections-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("D:/Apple-harvest-data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if demo.torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    settings = demo.load_settings()
    ripeness = demo.Ripeness("finetuned", args.device, None, None)
    rows, per_image, crops = [], [], {}
    with ZipFile(args.data_root / "02-annotated_data_fuji.zip") as archive:
        focal_table = demo.measure.table(archive, "focal_length.txt")
        for name in args.image:
            depth = np.load(io.BytesIO(archive.read(f"{demo.ZIP_ROOT}/depth_maps/{Path(name).stem}.npy")))
            gt = demo.dataset_detections(archive, name)
            _, labels = demo.truth_tables(archive, name)
            det = json.loads((args.detections_dir / f"{Path(name).stem}.json").read_text(encoding="utf-8"))["instances"]
            with Image.open(demo.find_image(args.data_root, name)) as source:
                rgb = source.convert("RGB")
            gt_masks = [rasterize(g["polygon"], depth.shape) for g in gt]
            det_masks = [rasterize(d["polygon"], depth.shape) for d in det]
            matches = match(gt_masks, det_masks, args.iou)
            per_image.append(dict(image=name, gt=len(gt), detections=len(det), matched=len(matches),
                                  missed=len(gt) - len(matches), unmatched_detections=len(det) - len(matches),
                                  mean_iou=float(np.mean([m[2] for m in matches])) if matches else None))
            draw_overlay(rgb, gt, det, matches,
                         f"{name} · 정답 {len(gt)} · 검출 {len(det)} · 일치 {len(matches)} · 놓침 {len(gt) - len(matches)} · 윤곽 없는 검출 {len(det) - len(matches)}",
                         args.out / f"{Path(name).stem}_검출비교.png")
            if not matches:
                continue
            gt_crops = [crop_of(rgb, gt[i]["polygon"]) for i, _, _ in matches]
            det_crops = [crop_of(rgb, det[j]["polygon"]) for _, j, _ in matches]
            scores = ripeness.scores(gt_crops + det_crops)
            half = len(matches)
            for k, (i, j, iou) in enumerate(matches):
                a = demo.diameter_of(gt[i]["polygon"], depth, focal_table[name], settings)
                b = demo.diameter_of(det[j]["polygon"], depth, focal_table[name], settings)
                xs, ys = zip(*gt[i]["polygon"])
                crops[(name, gt[i]["id"])] = (gt_crops[k], det_crops[k],
                                               locate.locator(rgb, (min(xs), min(ys), max(xs), max(ys)), 130))
                rows.append(dict(image=name, gt_id=gt[i]["id"], det_id=det[j]["id"], iou=iou,
                                 truth_label=labels.get(gt[i]["id"]),
                                 ripeness_split_of_apple=",".join(sorted(ripeness.splits.get(gt[i]["id"], ()))),
                                 gt_ripeness=ripeness.decide(scores[k]), det_ripeness=ripeness.decide(scores[half + k]),
                                 gt_diameter_mm=a["diameter_mm"], det_diameter_mm=b["diameter_mm"],
                                 gt_hold=a["hold_reason"], det_hold=b["hold_reason"]))
    both = [r for r in rows if r["gt_diameter_mm"] and r["det_diameter_mm"]]
    total = {k: sum(p[k] for p in per_image) for k in ("gt", "detections", "matched", "missed", "unmatched_detections")}
    summary = dict(
        iou_threshold=args.iou, images=per_image, total=total,
        recall=total["matched"] / max(total["gt"], 1), precision=total["matched"] / max(total["detections"], 1),
        matched_mean_iou=float(np.mean([r["iou"] for r in rows])) if rows else None,
        ripeness_decision_changed=sum(r["gt_ripeness"] != r["det_ripeness"] for r in rows),
        diameter_used_with_gt_mask=sum(bool(r["gt_diameter_mm"]) for r in rows),
        diameter_used_with_detector_mask=sum(bool(r["det_diameter_mm"]) for r in rows),
        diameter_change_mm=dict(n=len(both),
                                mean_abs=float(np.mean([abs(r["det_diameter_mm"] - r["gt_diameter_mm"]) for r in both])) if both else None,
                                mean=float(np.mean([r["det_diameter_mm"] - r["gt_diameter_mm"] for r in both])) if both else None))
    case_sheet(rows, crops, args.out / "익음도_사례_정답윤곽_대_검출윤곽.png")
    (args.out / "detection_comparison.json").write_text(json.dumps(dict(summary=summary, apples=rows), ensure_ascii=False, indent=2),
                                                         encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
