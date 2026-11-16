"""Audit paired VIA masks directly inside the official Fuji AmodalAppleSize ZIP."""

import argparse
import csv
import hashlib
import io
import json
import statistics
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageChops, ImageDraw, ImageFont
from synthetic_leaf import apply_leaf


ROOT = "02-annotated_data_fuji"
PADDING = 100
FIELDS = ("split", "filename", "apple_id", "visible_px", "amodal_px", "intersection_px",
          "outside_visible_fraction", "raw_occlusion_fraction", "occlusion_fraction",
          "outside_canvas", "interior", "eligible")


def read_annotations(archive, split, kind):
    path = f"{ROOT}/gt_json/{split}/via_region_data_{kind}.json"
    records = json.loads(archive.read(path))
    polygons = {}
    for row in records.values():
        for region in row["regions"].values():
            key = row["filename"], region["region_attributes"]["apple_ID"]
            shape = region["shape_attributes"]
            xs, ys = shape["all_points_x"], shape["all_points_y"]
            if key in polygons or shape["name"] != "polygon" or len(xs) != len(ys) or len(xs) < 3:
                raise ValueError(f"Duplicate key or invalid polygon: {key}")
            polygons[key] = list(zip(xs, ys))
    return polygons, {r["filename"] for r in records.values()}


def render_mask(size, points, offset=(0, 0)):
    mask = Image.new("L", size)
    ImageDraw.Draw(mask).polygon([(x - offset[0], y - offset[1]) for x, y in points], fill=255)
    return mask


def measure(modal, amodal, size):
    all_points = modal + amodal
    outside_canvas = any(not (0 <= x < size[0] and 0 <= y < size[1]) for x, y in all_points)
    left = max(0, min(x for x, _ in all_points))
    top = max(0, min(y for _, y in all_points))
    width = min(size[0] - 1, max(x for x, _ in all_points)) - left + 1
    height = min(size[1] - 1, max(y for _, y in all_points)) - top + 1
    vis = render_mask((width, height), modal, (left, top))
    full = render_mask((width, height), amodal, (left, top))
    v, a = vis.histogram()[255], full.histogram()[255]
    common = ImageChops.darker(vis, full).histogram()[255]
    interior = all(PADDING < x < size[0] - 1 - PADDING and PADDING < y < size[1] - 1 - PADDING
                   for x, y in amodal)
    outside = 1 - common / v
    return dict(visible_px=v, amodal_px=a, intersection_px=common, outside_visible_fraction=outside,
                raw_occlusion_fraction=1 - v / a, occlusion_fraction=1 - common / a,
                outside_canvas=int(outside_canvas), interior=int(interior), eligible=int(interior and outside <= 0.10))


def make_figure(archive, pairs, output, seed):
    # Compare an almost fully visible apple with a substantially occluded one.
    selected = []
    for lower, upper in ((0.00, 0.10), (0.55, 0.70)):
        candidates = [(row, v, a) for row, v, a in pairs if row["eligible"]
                      and lower <= row["occlusion_fraction"] < upper and row["amodal_px"] >= 5000]
        selected.append(min(candidates, key=lambda item: hashlib.sha256(
            f"{seed}|{item[0]['filename']}|{item[0]['apple_id']}".encode()).digest()))
    crops = []
    for row, visible, whole in selected:
        name = row["filename"]
        image = Image.open(io.BytesIO(archive.read(f"{ROOT}/images/test/{name}"))).convert("RGB")
        xmin, xmax = min(x for x, y in whole), max(x for x, y in whole)
        ymin, ymax = min(y for x, y in whole), max(y for x, y in whole)
        side = round(1.6 * max(xmax - xmin, ymax - ymin))
        left, top = (xmin + xmax - side) // 2, (ymin + ymax - side) // 2
        left = max(PADDING, min(left, image.width - PADDING - side))
        top = max(PADDING, min(top, image.height - PADDING - side))
        crop = image.crop((left, top, left + side, top + side))
        contour = crop.copy()
        for points, color in ((whole, "#ffc439"), (visible, "#36eff7")):
            local = [(x - left, y - top) for x, y in points]
            ImageDraw.Draw(contour).line(local + local[:1], fill=color, width=max(1, side // 105))
        crops.append((row, crop, contour, render_mask(crop.size, visible, (left, top))))

    font_path = Path("C:/Windows/Fonts/malgun.ttf")
    try:
        title = ImageFont.truetype(str(font_path), 30)
        normal = ImageFont.truetype(str(font_path), 21)
        small = ImageFont.truetype(str(font_path), 16)
    except OSError:
        title = normal = small = ImageFont.load_default()
    canvas = Image.new("RGB", (1100, 830), "#f7f6f1")
    draw = ImageDraw.Draw(canvas)
    draw.text((32, 19), "가려진 사과의 전체 윤곽과 보이는 영역", font=title, fill="#17382a")
    draw.text((32, 70), "청록선은 보이는 면, 노란선은 데이터가 추정한 전체 윤곽", font=normal, fill="#303e38")
    tile = 244
    for i, (row, crop, contour, _) in enumerate(crops):
        x = 32 + 530 * i
        for j, frame in enumerate((crop, contour)):
            canvas.paste(frame.resize((tile, tile), Image.Resampling.LANCZOS), (x + 252 * j, 113))
        draw.text((x, 365), f"자연 가림 추정 {row['occlusion_fraction']:.1%}  ·  사과 {row['apple_id']}",
                  font=normal, fill="#17382a")
    draw.text((32, 420), "같은 사과 위에 합성 잎을 얹었을 때", font=title, fill="#17382a")
    draw.text((32, 468), "아래 비율은 원래 보이던 사과 픽셀 중 합성 잎으로 덮인 비중", font=normal, fill="#303e38")
    row, crop, _, mask = crops[0]
    measured = []
    for i, target in enumerate((0.0, 0.15, 0.30, 0.50)):
        frame, fraction = apply_leaf(crop, mask, target, f"{row['filename']}/{row['apple_id']}", seed)
        x = 32 + i * 263
        canvas.paste(frame.resize((tile, tile), Image.Resampling.LANCZOS), (x, 514))
        draw.text((x, 762), f"{fraction:.1%}", font=normal, fill="#17382a")
        measured.append(fraction)
    draw.text((32, 802), "이미지 원자료  Gené-Mola et al., AmodalAppleSize_RGB-D (CC BY 4.0)  ·  크롭, 윤곽 표시 및 합성 잎 추가",
              font=small, fill="#54625b")
    canvas.save(output, optimize=True)
    return {"examples": [item[0] for item in crops], "synthetic_measured": measured}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path("D:/Apple-harvest-data/02-annotated_data_fuji.zip"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261116)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows, test_pairs, split_counts, ids_by_split = [], [], {}, {}
    with ZipFile(args.zip) as archive:
        for split in ("train", "val", "test"):
            v, v_names = read_annotations(archive, split, "modal")
            a, a_names = read_annotations(archive, split, "amodal")
            image_paths = {Path(name).name for name in archive.namelist()
                           if name.startswith(f"{ROOT}/images/{split}/") and name.lower().endswith(".png")}
            if not v_names == a_names == image_paths or v.keys() != a.keys():
                raise ValueError(f"Unmatched annotations or image names in {split}")
            dimensions = {}
            size_counts = Counter()
            # Each entry's PNG header gives the actual size without decoding the whole archive.
            for filename in sorted(image_paths):
                with archive.open(f"{ROOT}/images/{split}/{filename}") as fp:
                    header = fp.read(24)
                if header[:8] != b"\x89PNG\r\n\x1a\n":
                    raise ValueError(f"Unexpected PNG header: {filename}")
                size = int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")
                dimensions[filename] = size
                size_counts[size] += 1
            for key in sorted(v):
                filename, apple_id = key
                row = {"split": split, "filename": filename, "apple_id": apple_id}
                row.update(measure(v[key], a[key], dimensions[filename]))
                rows.append(row)
                if split == "test":
                    test_pairs.append((row, v[key], a[key]))
            ids_by_split[split] = {key[1] for key in v}
            split_counts[split] = {"images": len(image_paths), "paired_instances": len(v),
                                   "unique_label_ids": len(ids_by_split[split]),
                                   "image_sizes": {str(k): n for k, n in size_counts.items()}}
        figure = make_figure(archive, test_pairs, args.out / "occlusion_examples.png", args.seed)

    with (args.out / "natural_occlusion.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    valid = [r["occlusion_fraction"] for r in rows if r["eligible"]]
    summary = {
        "data": "https://doi.org/10.34810/data916", "data_license": "CC BY 4.0 (archive README)",
        "splits": split_counts, "all_pairs": len(rows),
        "image_size_rng": "individual PNG headers checked",
        "apple_ids_overlap_train_test": len(ids_by_split["train"] & ids_by_split["test"]),
        "apple_ids_overlap_val_test": len(ids_by_split["val"] & ids_by_split["test"]),
        "eligible_rule": "Amodal polygon strictly inside inner unpadded frame (100 px per side) and at most 10% of visible pixels outside amodal",
        "edge_or_pad": sum(not r["interior"] for r in rows),
        "coordinates_clipped_to_canvas": sum(r["outside_canvas"] for r in rows),
        "outside_visible_over_10pct": sum(r["outside_visible_fraction"] > .10 for r in rows),
        "raw_negative": sum(r["raw_occlusion_fraction"] < 0 for r in rows),
        "eligible_pairs": len(valid), "eligible_unique_ids": len({r["apple_id"] for r in rows if r["eligible"]}),
        "eligible_occlusion_median": statistics.median(valid),
        "eligible_occlusion_quartiles": statistics.quantiles(valid, method="inclusive"),
        "eligible_occlusion_bins": {"0-.15": sum(x < .15 for x in valid),
                                    ".15-.30": sum(.15 <= x < .30 for x in valid),
                                    ".30-.50": sum(.30 <= x < .50 for x in valid),
                                    ".50-1": sum(x >= .50 for x in valid)},
        "seed": args.seed, **figure,
    }
    (args.out / "natural_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=2))


if __name__ == "__main__":
    main()
