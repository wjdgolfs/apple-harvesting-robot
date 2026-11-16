"""Audit Fuji VIA annotations and evaluate a train-fitted color threshold on RGB crops."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, __version__ as pillow_version


SPLITS = ("train", "val", "test")
LABELS = ("ripe", "unripe")


def group_key(filename):
    """Group tiles of one camera frame, or all snapshots from one ISO-style date."""
    tile = re.fullmatch(r"(_MG_\d+)_\d+\.png", filename)
    if tile:
        return tile[1]
    dated = re.fullmatch(r"(\d{4}_\d{2}_\d{2})_.+\.png", filename)
    if dated:
        return dated[1]
    raise ValueError(f"Unknown acquisition naming scheme: {filename}")


def stable_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def region_box(shape):
    """Return Pillow crop bounds. VIA polygon vertices are inclusive pixel coordinates."""
    if shape["name"] == "rect":
        x, y = shape["x"], shape["y"]
        return x, y, x + shape["width"], y + shape["height"]
    if shape["name"] == "polygon":
        x, y = shape["all_points_x"], shape["all_points_y"]
        if len(x) != len(y) or len(x) < 3:
            raise ValueError("Invalid polygon vertices")
        return min(x), min(y), max(x) + 1, max(y) + 1
    raise ValueError(f"Unknown VIA region shape: {shape['name']}")


def score_crop(crop):
    """Mean (red - green)/(red + green + 1) in a 24x24 center crop, range [-1, 1]."""
    width, height = crop.size
    margin_x, margin_y = width // 8, height // 8
    center = crop.crop((margin_x, margin_y, width - margin_x, height - margin_y))
    small = center.convert("RGB").resize((24, 24), Image.Resampling.BOX)
    pixels = small.get_flattened_data() if hasattr(small, "get_flattened_data") else small.getdata()
    return sum((r - g) / (r + g + 1) for r, g, _ in pixels) / (24 * 24)


def fit_threshold(rows):
    """Maximize training balanced accuracy. For ties use the higher threshold."""
    rows = sorted(rows, reverse=True)
    counts = Counter(label for _, label in rows)
    if any(counts[label] == 0 for label in LABELS):
        raise ValueError("Training requires both ripe and unripe examples")
    true_ripe = false_ripe = 0
    best_gain = 0
    threshold = math.nextafter(rows[0][0], math.inf)
    for score, batch in itertools.groupby(rows, key=lambda row: row[0]):
        for _, label in batch:
            true_ripe += label == "ripe"
            false_ripe += label == "unripe"
        gain = true_ripe * counts["unripe"] - false_ripe * counts["ripe"]
        if gain > best_gain:
            best_gain, threshold = gain, score
    return threshold


def metrics(rows, threshold, majority):
    counts = Counter(row["label"] for row in rows)
    confusion = {label: {p: 0 for p in LABELS} for label in LABELS}
    for row in rows:
        row["prediction"] = "ripe" if row["score"] >= threshold else "unripe"
        confusion[row["label"]][row["prediction"]] += 1
    recalls = {k: confusion[k][k] / counts[k] if counts[k] else None for k in LABELS}
    total = len(rows)
    return {
        "images": len({r["filename"] for r in rows}),
        "groups": len({r["group"] for r in rows}),
        "objects": total,
        "confusion_true_then_predicted": confusion,
        "accuracy": sum(confusion[k][k] for k in LABELS) / total if total else None,
        "recall": recalls,
        "balanced_accuracy": sum(recalls.values()) / 2 if all(v is not None for v in recalls.values()) else None,
        "train_majority_accuracy": counts[majority] / total if total else None,
    }


def audit_split(root, split):
    folder = root / split
    metadata = json.loads((folder / f"{split}_via_final.json").read_bytes())
    records = list(metadata.get("_via_img_metadata", metadata).values())
    filenames = [r["filename"] for r in records]
    actual = {p.name for p in (folder / "images").iterdir() if p.is_file()}
    depth_stems = {p.stem for p in (folder / "depths").glob("*.npy")}
    rgb_stems = {Path(name).stem for name in actual}
    expected = set(filenames)
    if expected - actual:
        raise ValueError(f"Missing RGB images: {sorted(expected - actual)[:5]}")
    if len(expected) != len(filenames):
        raise ValueError("Repeated filename in VIA metadata")
    counts, shapes, sizes = Counter(), Counter(), Counter()
    groups = defaultdict(list)
    missing_diameter = 0
    for record in records:
        filename = record["filename"]
        with Image.open(folder / "images" / filename) as im:
            width, height = im.size
            sizes[f"{im.format} {im.mode} {width}x{height}"] += 1
        groups[group_key(filename)].append(record)
        for reg in record["regions"]:
            label = reg["region_attributes"]["Ripeness"]
            if label not in LABELS:
                raise ValueError(f"Unknown label {label!r}")
            counts[label] += 1
            shape = reg["shape_attributes"]
            shapes[shape["name"]] += 1
            x1, y1, x2, y2 = region_box(shape)
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"Invalid region bounds in {filename}")
            missing_diameter += reg["region_attributes"]["Diameter"] == -1
    report = {
        "images": len(actual), "annotations": len(records), "objects": sum(counts.values()),
        "depth_files": len(depth_stems),
        "rgb_without_depth": sorted(rgb_stems - depth_stems),
        "depth_without_rgb": sorted(depth_stems - rgb_stems),
        "labels": counts, "shapes": shapes, "image_headers": sizes,
        "images_without_annotations_entry": sorted(actual - expected),
        "annotations_without_image": sorted(expected - actual),
        "images_with_zero_regions": sum(not r["regions"] for r in records),
        "diameter_minus_one": missing_diameter, "groups": len(groups),
    }
    return groups, report


def run(root, output):
    output.mkdir(parents=True, exist_ok=True)
    groups = {}
    summary = {"pillow_version": pillow_version, "source_audit": {}}
    for split in SPLITS:
        groups[split], summary["source_audit"][split] = audit_split(root, split)
    summary["official_group_overlap"] = {
        f"{a}-{b}": len(groups[a].keys() & groups[b].keys())
        for a, b in itertools.combinations(SPLITS, 2)
    }
    # Keep train intact. If a remaining group spans val and test, assign it to just one.
    # Keep one labeled photo per group in its original split for a bounded pilot.
    experiment = {split: [] for split in SPLITS}
    for group in sorted(set().union(*(g.keys() for g in groups.values()))):
        splits = [s for s in SPLITS if group in groups[s]]
        if "train" in splits:
            split = "train"
        else:
            split = splits[int(stable_hash(group)[:8], 16) % len(splits)]
        candidates = [r for r in groups[split][group] if r["regions"]]
        if not candidates:
            continue
        record = min(candidates, key=lambda r: stable_hash(r["filename"]))
        filename = record["filename"]
        with Image.open(root / split / "images" / filename) as image:
            image.load()
            for idx, region in enumerate(record["regions"]):
                shape = region["shape_attributes"]
                attrs = region["region_attributes"]
                experiment[split].append({
                    "split": split, "filename": filename, "region_index": idx, "group": group,
                    "shape": shape["name"], "label": attrs["Ripeness"],
                    "diameter": attrs["Diameter"], "score": score_crop(image.crop(region_box(shape))),
                })
    train = experiment["train"]
    threshold = fit_threshold((r["score"], r["label"]) for r in train)
    counts = Counter(r["label"] for r in train)
    majority = counts.most_common(1)[0][0]
    summary["threshold"] = threshold
    summary["training_majority_label"] = majority
    summary["pilot"] = {s: metrics(experiment[s], threshold, majority) for s in SPLITS}
    summary["pilot_group_overlap"] = {
        f"{a}-{b}": len({r["group"] for r in experiment[a]} & {r["group"] for r in experiment[b]})
        for a, b in itertools.combinations(SPLITS, 2)
    }
    with (output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(train[0]))
        writer.writeheader()
        for split in SPLITS:
            writer.writerows(experiment[split])
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["pilot"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(r"D:\Apple-harvest-data\ripeness"))
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    run(options.data, options.out)
