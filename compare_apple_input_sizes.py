"""Compare 640 and 960 pixel inputs on the fixed MinneApple validation split."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

from compare_apple_predictions import comparison, infer, load_ground_truth, sha256, write_csv


SIZES = (640, 960)
SMALL_AREA = 1024


def load_validation(manifest):
    with manifest.open(encoding="utf-8-sig", newline="") as source:
        rows = [row for row in csv.DictReader(source) if row["split"] == "val"]
    if len(rows) != 152 or len({row["filename"] for row in rows}) != 152:
        raise ValueError("Expected 152 distinct images in the MinneApple v1 validation split")
    if sum(int(row["instance_id_count"]) for row in rows) != 6101:
        raise ValueError("Expected 6,101 instances in the fixed validation split")
    return rows


def record_result(size, result, image_stats, details):
    count = len(result["scores"])
    image_stats[f"predictions_{size}"] = count
    for mode in ("box", "mask"):
        matched = result[mode]
        image_stats[f"{mode}_matched_{size}"] = len(matched)
        image_stats[f"{mode}_unmatched_predictions_{size}"] = count - len(matched)
        ious = result[f"{mode}_iou"]
        best = ious.max(axis=0, initial=0)
        for index, detail in enumerate(details):
            detail[f"{mode}_matched_{size}"] = int(index in matched)
            detail[f"{mode}_best_iou_{size}"] = round(float(best[index]), 4)
    for stage, ms in result["speed"].items():
        image_stats[f"{stage}_ms_{size}"] = round(ms, 4) if ms is not None else 0.0


def summarize(images, apples):
    groups = {
        "all": apples,
        "small": [item for item in apples if item["area_px"] < SMALL_AREA],
        "other": [item for item in apples if item["area_px"] >= SMALL_AREA],
    }
    totals = {}
    for size in SIZES:
        result = {"predictions": sum(row[f"predictions_{size}"] for row in images)}
        for mode in ("box", "mask"):
            result[f"{mode}_unmatched_predictions"] = sum(
                row[f"{mode}_unmatched_predictions_{size}"] for row in images
            )
        for label, group in groups.items():
            result[label] = {"ground_truth": len(group)}
            for mode in ("box", "mask"):
                count = sum(row[f"{mode}_matched_{size}"] for row in group)
                result[label][f"{mode}_matched"] = count
                result[label][f"{mode}_missed"] = len(group) - count
        for mode in ("box", "mask"):
            result[f"{mode}_precision"] = result["all"][f"{mode}_matched"] / result["predictions"]
        timings = [sum(row[f"{stage}_ms_{size}"] for stage in ("preprocess", "inference", "postprocess"))
                   for row in images]
        result["model_pipeline_ms"] = {"median": float(np.median(timings)), "p95": float(np.percentile(timings, 95))}
        totals[str(size)] = result

    changes = {}
    for label, group in groups.items():
        metrics = {}
        for mode in ("box", "mask"):
            base, large = f"{mode}_matched_640", f"{mode}_matched_960"
            metrics[f"{mode}_recovered"] = sum(not row[base] and row[large] for row in group)
            metrics[f"{mode}_lost"] = sum(row[base] and not row[large] for row in group)
        changes[label] = metrics
    return totals, changes


def save_examples(apples, frames, models, classes, dataset, args):
    small = [row for row in apples if row["area_px"] < SMALL_AREA]
    roles = (
        ("01_small_apple_recovered", lambda row: not row["box_matched_640"] and row["box_matched_960"],
         lambda row: row["mask_matched_960"] + row["box_best_iou_960"] - row["box_best_iou_640"]),
        ("02_small_apple_lost", lambda row: row["box_matched_640"] and not row["box_matched_960"],
         lambda row: row["mask_matched_640"] + row["box_best_iou_640"] - row["box_best_iou_960"]),
    )
    examples = []
    used_images = set()
    for role, condition, rank in roles:
        candidates = [row for row in small if row["image"] not in used_images and condition(row)]
        if not candidates:
            continue
        example = {"role": role, **max(candidates, key=rank)}
        used_images.add(example["image"])
        row = frames[example["image"]]
        source = dataset / row["image"]
        labels, objects, areas, boxes = load_ground_truth(row, dataset)
        predictions = [
            infer(models[size], classes, source, labels, objects, areas, boxes, args.device, args.conf, imgsz=size)
            for size in SIZES
        ]
        comparison(source, labels, objects, boxes, predictions, example, args.output,
                   prediction_labels=tuple(f"\uc785\ub825 {size}" for size in SIZES))
        examples.append(example)
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Directory containing images/ and masks/")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--manifest", type=Path, default=Path("splits/minneapple_v1/manifest.csv"))
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    if not 0 < args.conf <= 1:
        parser.error("--conf must be greater than 0 and at most 1")

    rows = load_validation(args.manifest)
    checkpoint_hash = sha256(args.weights)
    models = {size: YOLO(str(args.weights), task="segment") for size in SIZES}
    classes = [index for index, label in models[SIZES[0]].names.items() if label == "apple"]
    if len(classes) != 1:
        raise ValueError("Expected one apple class in the segmentation checkpoint")

    args.output.mkdir(parents=True, exist_ok=False)
    image_rows, apples = [], []
    for index, row in enumerate(rows):
        source = args.dataset / row["image"]
        if sha256(source) != row["image_sha256"]:
            raise ValueError(f"Image changed: {source}")
        labels, objects, areas, boxes = load_ground_truth(row, args.dataset)
        image_stats = {"image": row["filename"], "sequence": row["sequence"], "ground_truth": len(objects)}
        details = [
            {"image": row["filename"], "sequence": row["sequence"], "gt_id": int(object_id), "area_px": int(area)}
            for object_id, area in zip(objects, areas)
        ]
        if index == 0:
            for size in SIZES:
                for _ in range(3):
                    infer(models[size], classes, source, labels, objects, areas, boxes, args.device, args.conf, imgsz=size)
        # Alternate evaluation order to reduce timing bias between the two sizes.
        for size in (SIZES if index % 2 == 0 else SIZES[::-1]):
            result = infer(models[size], classes, source, labels, objects, areas, boxes, args.device, args.conf, imgsz=size)
            record_result(size, result, image_stats, details)
            del result
        image_rows.append(image_stats)
        apples.extend(details)
        if (index + 1) % 25 == 0 or index + 1 == len(rows):
            print(f"{index + 1}/{len(rows)} image pairs compared", flush=True)

    metrics, changes = summarize(image_rows, apples)
    summary = {
        "images": len(image_rows), "ground_truth": len(apples),
        "model": args.weights.name, "model_sha256": checkpoint_hash,
        "manifest_sha256": sha256(args.manifest),
        "device": args.device, "ultralytics": ultralytics.__version__, "torch": torch.__version__,
        "settings": {"conf": args.conf, "nms_iou": 0.7, "match_iou": 0.5, "max_det": 300,
                     "retina_masks": True, "small_area_px_exclusive": SMALL_AREA},
        "metrics": metrics, "changes_640_to_960": changes,
        "examples": save_examples(apples, {row["filename"]: row for row in rows}, models, classes, args.dataset, args),
    }
    write_csv(args.output / "per_image.csv", image_rows)
    write_csv(args.output / "per_apple.csv", apples)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved evaluation to {args.output}", flush=True)


if __name__ == "__main__":
    main()
