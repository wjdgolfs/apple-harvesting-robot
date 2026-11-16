"""Check example labels and tabulate held-out predictions by natural occlusion."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def draw_examples(root, links, destination):
    records = json.loads((root / "train/train_via_final.json").read_bytes())
    photos = {r["filename"]: r for r in records.values()}
    selected = [("_MG_2902_21.png", 0), ("_MG_6876_18.png", 6)]
    ids = {(r["filename"], int(r["region_index"])): r["apple_id"] for r in links if r["split"] == "train"}
    canvas = Image.new("RGB", (870, 522), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        title, body, small = (ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", size) for size in (30, 23, 17))
    except OSError:
        title, body, small = (ImageFont.load_default(size) for size in (30, 23, 17))
    draw.text((30, 20), "같은 지름에서 확인한 두 익음도 라벨", font=title, fill="#183d2a")
    details = []
    for i, (name, index) in enumerate(selected):
        region = photos[name]["regions"][index]
        attrs = region["region_attributes"]
        shape = region["shape_attributes"]
        points = list(zip(shape["all_points_x"], shape["all_points_y"]))
        x, y = zip(*points)
        cx, cy = (min(x) + max(x)) / 2, (min(y) + max(y)) / 2
        side = 1.65 * max(max(x) - min(x), max(y) - min(y))
        crop_bounds = tuple(round(v) for v in (cx-side/2, cy-side/2, cx+side/2, cy+side/2))
        with Image.open(root / "train/images" / name) as im:
            frame = im.convert("RGB")
        ImageDraw.Draw(frame).line(points + points[:1], fill="#79f7f1", width=2)
        tile = frame.crop(crop_bounds).resize((312, 312), Image.Resampling.LANCZOS)
        start = 30 + i * 425
        canvas.paste(tile, (start, 77))
        number, label = float(attrs["Diameter"]), attrs["Ripeness"]
        apple_id = ids[(name, index)]
        ko = {"ripe": "익음", "unripe": "미숙"}[label]
        draw.text((start, 397), f"{number:.1f} mm   ·   {ko} ({label})", font=body, fill="#183d2a")
        draw.text((start, 436), f"{name}   |   ID {apple_id}", font=small, fill="#36433d")
        details.append({"filename": name, "region_index": index, "apple_id": apple_id,
                        "diameter_mm": number, "label": label})
    draw.text((30, 487), "Fuji-Ripeness-Size 라벨 · AmodalAppleSize_RGB-D 공유 사진 · 윤곽을 표시하고 확대함",
              font=small, fill="#526158")
    canvas.save(destination)
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ripeness-root", type=Path, required=True)
    parser.add_argument("--links", type=Path, required=True)
    parser.add_argument("--natural", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    links = read_csv(args.links)
    per_id = defaultdict(set)
    for row in links:
        per_id[row["apple_id"]].add(row["label"])
    natural = {(r["split"], r["filename"], r["apple_id"]): r for r in read_csv(args.natural)}
    brackets = ("0–<15%", "15–<30%", "30–<50%", "50–100%")
    by_bracket = {b: {"n": 0, "correct": 0, "ripe": 0, "ripe_correct": 0} for b in brackets}
    excluded = 0
    for prediction in read_csv(args.predictions):
        if prediction["split"] != "test":
            continue
        row = natural[("test", prediction["filename"], prediction["apple_id"])]
        if row["eligible"] != "1":
            excluded += 1
            continue
        level = float(row["occlusion_fraction"])
        group = by_bracket[brackets[sum(level >= t for t in (.15, .30, .50))]]
        correct = prediction["prediction"] == prediction["label"]
        group["n"] += 1
        group["correct"] += correct
        if prediction["label"] == "ripe":
            group["ripe"] += 1
            group["ripe_correct"] += correct
    report = {"linked_distinct_id_strings":len(per_id),
              "id_strings_with_both_labels":sum(len(classes) == 2 for classes in per_id.values()),
              "examples":draw_examples(args.ripeness_root, links, args.out / "ripeness_examples.png"),
              "natural_occlusion_on_test":by_bracket, "test_excluded_by_natural_mask_qc":excluded}
    (args.out / "case_checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
