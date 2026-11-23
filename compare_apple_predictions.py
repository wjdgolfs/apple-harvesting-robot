"""Compare original and fine-tuned apple segmentation on the fixed validation split."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import ultralytics
from ultralytics import YOLO


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def match(iou, confidence):
    """Match by descending confidence, then highest available IoU."""
    matched = {}
    for pred in np.argsort(-confidence, kind="stable"):
        for obj in np.argsort(-iou[pred], kind="stable"):
            if iou[pred, obj] < 0.5:
                break
            if int(obj) not in matched:
                matched[int(obj)] = int(pred)
                break
    return matched


def load_ground_truth(row, dataset):
    path = dataset / row["mask"]
    if sha256(path) != row["mask_sha256"]:
        raise ValueError(f"Ground truth changed: {path}")
    labels = np.asarray(Image.open(path))
    if labels.ndim != 2 or labels.shape != (int(row["height"]), int(row["width"])):
        raise ValueError(f"Unexpected mask shape: {path}")
    y, x = np.nonzero(labels)
    ids = labels[y, x]
    areas = np.bincount(ids, minlength=256)
    objects = np.flatnonzero(areas)
    if len(objects) != int(row["instance_id_count"]):
        raise ValueError(f"Unexpected instance count: {path}")
    x1, y1 = np.full(256, labels.shape[1]), np.full(256, labels.shape[0])
    x2, y2 = np.zeros(256, dtype=int), np.zeros(256, dtype=int)
    np.minimum.at(x1, ids, x)
    np.minimum.at(y1, ids, y)
    np.maximum.at(x2, ids, x + 1)
    np.maximum.at(y2, ids, y + 1)
    boxes = np.column_stack([x1[objects], y1[objects], x2[objects], y2[objects]])
    return labels, objects, areas[objects], boxes


def infer(model, classes, image, labels, objects, areas, boxes, device, conf, *, imgsz=640):
    result = model.predict(
        source=str(image), classes=classes, imgsz=imgsz, conf=conf, iou=0.7,
        max_det=300, retina_masks=True, device=device, verbose=False, save=False,
    )[0]
    if result.orig_shape != labels.shape:
        raise ValueError(f"Prediction and ground truth are misaligned: {image}")
    predictions = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    if result.masks is None:
        if len(scores):
            raise ValueError(f"Predicted boxes have no masks: {image}")
        masks = np.empty((0, *labels.shape), dtype=bool)
    else:
        masks = result.masks.data.cpu().numpy().astype(bool, copy=False)
    if masks.shape != (len(scores), *labels.shape):
        raise ValueError(f"Unexpected predicted mask shape: {image}")

    top_left = np.maximum(predictions[:, None, :2], boxes[:, :2])
    bottom_right = np.minimum(predictions[:, None, 2:], boxes[:, 2:])
    intersection = np.maximum(bottom_right - top_left, 0).prod(axis=2)
    union = (
        (predictions[:, 2:] - predictions[:, :2]).prod(axis=1)[:, None]
        + (boxes[:, 2:] - boxes[:, :2]).prod(axis=1) - intersection
    )
    box_iou = intersection / union
    mask_iou = np.zeros_like(box_iou)
    for idx, mask in enumerate(masks):
        pixels = labels[mask]
        overlap = np.bincount(pixels, minlength=256)[objects]
        mask_iou[idx] = overlap / (len(pixels) + areas - overlap)
    return {
        "masks": masks, "boxes": predictions, "scores": scores,
        "box": match(box_iou, scores), "mask": match(mask_iou, scores),
        "box_iou": box_iou, "mask_iou": mask_iou, "speed": result.speed,
    }


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_examples(objects):
    roles = [
        ("01_small_mask_recovered", lambda o: o["area"] < 1024 and not o["before_mask"] and o["after_mask"],
         lambda o: o["after_best_mask_iou"] - o["before_best_mask_iou"]),
        ("02_still_missed", lambda o: o["area"] < 1024 and not o["before_box"] and not o["after_box"],
         lambda o: o["area"]),
        ("03_box_without_matching_mask", lambda o: o["after_box"] and not o["after_mask"]
         and o["after_box_mask_iou"] < 0.35,
         lambda o: o["after_best_box_iou"] - o["after_box_mask_iou"]),
        ("04_mask_regression", lambda o: o["before_mask"] and not o["after_mask"],
         lambda o: o["before_best_mask_iou"] - o["after_best_mask_iou"]),
    ]
    examples = []
    used = set()
    for role, condition, score in roles:
        eligible = [o for o in objects if o["image"] not in used and condition(o)]
        if eligible:
            chosen = max(eligible, key=score)
            examples.append({"role": role, **chosen})
            used.add(chosen["image"])
    return examples


def tint(image, mask, color):
    image[mask] = np.rint(image[mask] * 0.59 + np.asarray(color) * 0.41)


def mark_focus(image, rect, stroke):
    ImageDraw.Draw(image).rectangle(rect, outline="#ffdf41", width=stroke)


def comparison(image, labels, objects, boxes, predictions, example, output, *, prediction_labels=("\ud559\uc2b5 \uc804", "\ud559\uc2b5 \ud6c4")):
    raw = np.asarray(Image.open(image).convert("RGB"))
    gt = raw.copy()
    colors = [(54, 207, 242), (196, 113, 249), (127, 238, 85)]
    for idx, object_id in enumerate(objects):
        tint(gt, labels == object_id, colors[idx % len(colors)])
    panels = [Image.fromarray(gt)]
    headings = [f"\uc815\ub2f5  |  \uc0ac\uacfc {len(objects)}\uac1c"]
    for label, result in zip(prediction_labels, predictions):
        overlay = raw.copy()
        matched = set(result["mask"].values())
        for idx, mask in enumerate(result["masks"]):
            tint(overlay, mask, (33, 221, 127) if idx in matched else (255, 114, 48))
        view = Image.fromarray(overlay)
        draw = ImageDraw.Draw(view)
        for idx, bounds in enumerate(result["boxes"]):
            draw.rectangle(tuple(bounds), outline="#35d98e" if idx in matched else "#ff843d", width=1)
        panels.append(view)
        headings.append(f"{label}  |  \uc608\uce21 {len(result['scores'])}\uac1c / \ub9c8\uc2a4\ud06c \uc77c\uce58 {len(matched)}\uac1c")

    x1, y1, x2, y2 = boxes[np.where(objects == example["gt_id"])[0][0]]
    focus = (max(0, x1-3), max(0, y1-3), min(raw.shape[1]-1, x2+3), min(raw.shape[0]-1, y2+3))
    size = max(145, x2-x1+64, y2-y1+64)
    left = int(np.clip((x1+x2-size)//2, 0, raw.shape[1]-size))
    top = int(np.clip((y1+y2-size)//2, 0, raw.shape[0]-size))
    crop = (left, top, left+size, top+size)
    width, gap = 334, 16
    scaled_height = round(raw.shape[0] * width / raw.shape[1])
    font_root = Path("C:/Windows/Fonts")
    title_font = ImageFont.truetype(str(font_root / "malgunbd.ttf"), 21)
    body_font = ImageFont.truetype(str(font_root / "malgun.ttf"), 17)
    canvas = Image.new("RGB", (width*3+gap*4, 84+scaled_height+width+gap*3), "#151e25")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 14), f"{image.name}   |   \ud655\ub300 \ub300\uc0c1 GT #{example['gt_id']}", font=title_font, fill="white")
    for idx, (panel, heading) in enumerate(zip(panels, headings)):
        left_panel = gap+(width+gap)*idx
        draw.text((left_panel, 47), heading, font=body_font, fill="white")
        zoom = panel.crop(crop)
        mark_focus(zoom, (focus[0]-left, focus[1]-top, focus[2]-left, focus[3]-top), 2)
        mark_focus(panel, focus, 3)
        canvas.paste(panel.resize((width, scaled_height)), (left_panel, 75))
        canvas.paste(zoom.resize((width, width)), (left_panel, 75+scaled_height+gap))
    draw.text(
        (gap, 80+scaled_height+gap+width),
        "\ub178\ub791  \ud655\ub300 \ub300\uc0c1          \ucd08\ub85d  \uc815\ub2f5\uacfc \uc77c\uce58\ud55c \uc608\uce21 \ub9c8\uc2a4\ud06c          \uc8fc\ud669  \uc77c\uce58\ud558\uc9c0 \uc54a\uc740 \uc608\uce21 \ub9c8\uc2a4\ud06c",
        font=body_font, fill="#e2eced",
    )
    canvas.save(output / f"{example['role']}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("splits/minneapple_v1/manifest.csv"))
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8-sig", newline="") as source:
        by_name = {row["filename"]: row for row in csv.DictReader(source) if row["split"] == "val"}
    if len(by_name) != 152 or sum(int(row["instance_id_count"]) for row in by_name.values()) != 6101:
        raise ValueError("Expected the fixed MinneApple v1 validation split")
    args.output.mkdir(parents=True, exist_ok=True)
    checksums = {tag: sha256(path) for tag, path in zip(("before", "after"), (args.before, args.after))}
    if checksums["before"] == checksums["after"]:
        raise ValueError("Both checkpoints have the same hash")
    models = [YOLO(str(path), task="segment") for path in (args.before, args.after)]
    apple_classes = [[i for i, name in model.names.items() if name == "apple"] for model in models]
    if not all(len(classes) == 1 for classes in apple_classes):
        raise ValueError("Each checkpoint must have one apple class")

    image_rows, object_rows = [], []
    for index, (name, row) in enumerate(by_name.items(), 1):
        source = args.dataset / row["image"]
        if sha256(source) != row["image_sha256"]:
            raise ValueError(f"Image changed: {source}")
        labels, objects, areas, boxes = load_ground_truth(row, args.dataset)
        stats = {"image": name, "sequence": row["sequence"], "ground_truth": len(objects)}
        details = [
            {"image": name, "sequence": row["sequence"], "gt_id": int(obj), "area": int(area)}
            for obj, area in zip(objects, areas)
        ]
        for tag, model, classes in zip(("before", "after"), models, apple_classes):
            result = infer(model, classes, source, labels, objects, areas, boxes, args.device, args.conf)
            stats[f"{tag}_predictions"] = count = len(result["scores"])
            for mode in ("box", "mask"):
                mapping = result[mode]
                stats[f"{tag}_{mode}_matched"] = len(mapping)
                stats[f"{tag}_{mode}_missed"] = len(objects) - len(mapping)
                stats[f"{tag}_{mode}_unmatched_predictions"] = count - len(mapping)
                best = result[f"{mode}_iou"].max(axis=0, initial=0)
                for obj_index, details_row in enumerate(details):
                    details_row[f"{tag}_{mode}"] = int(obj_index in mapping)
                    details_row[f"{tag}_best_{mode}_iou"] = round(float(best[obj_index]), 4)
            for obj_index, details_row in enumerate(details):
                pred = result["box"].get(obj_index)
                details_row[f"{tag}_box_mask_iou"] = (
                    round(float(result["mask_iou"][pred, obj_index]), 4) if pred is not None else None
                )
        image_rows.append(stats)
        object_rows.extend(details)
        if index % 20 == 0 or index == len(by_name):
            print(f"{index}/{len(by_name)} validation images compared", flush=True)

    summary = {
        "images": len(by_name), "ground_truth": len(object_rows),
        "checkpoint_sha256": checksums, "apple_classes": apple_classes,
        "device": args.device, "ultralytics": ultralytics.__version__, "torch": torch.__version__,
        "settings": {"imgsz": 640, "conf": args.conf, "nms_iou": 0.7, "match_iou": 0.5, "max_det": 300},
        "counts": {
            key: sum(row[key] for row in image_rows)
            for key in image_rows[0] if key not in ("sequence", "image")
        },
        "small_area_px": 1024,
        "transitions": {},
    }
    for key, group in (
        ("all", object_rows), ("small", [row for row in object_rows if row["area"] < 1024]),
    ):
        metrics = {"ground_truth": len(group)}
        for mode in ("box", "mask"):
            before, after = f"before_{mode}", f"after_{mode}"
            metrics[f"{mode}_recovered"] = sum(not o[before] and o[after] for o in group)
            metrics[f"{mode}_lost"] = sum(o[before] and not o[after] for o in group)
            metrics[f"{mode}_still_missed"] = sum(not o[before] and not o[after] for o in group)
        summary["transitions"][key] = metrics

    examples = read_examples(object_rows)
    for example in examples:
        row = by_name[example["image"]]
        source = args.dataset / row["image"]
        labels, objects, areas, boxes = load_ground_truth(row, args.dataset)
        predictions = [
            infer(model, classes, source, labels, objects, areas, boxes, args.device, args.conf)
            for model, classes in zip(models, apple_classes)
        ]
        comparison(source, labels, objects, boxes, predictions, example, args.output)
    summary["examples"] = examples
    write_csv(args.output / "per_image.csv", image_rows)
    write_csv(args.output / "per_apple.csv", object_rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
