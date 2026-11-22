"""Red-pixel baseline on the same 50 % synthetic-leaf images the networks were scored on."""
import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


leaf = load("leaf", "occlusion/synthetic_leaf.py")
ex = load("ex", "익음도보강/label_examples.py")
base = Path("D:/Apple-harvest-data/ripeness")
split_of = {(r["filename"], r["region_index"]): r["split"]
            for r in csv.DictReader(open("integration/results/linked_instances.csv", encoding="utf-8-sig"))}
rows = [r for r in csv.DictReader(open("익음도보강/results/test_scores.csv", encoding="utf-8-sig")) if r["cover_percent"] == "50"]
via = {}
for sp in ("train", "val", "test"):
    meta = json.loads((base / sp / f"{sp}_via_final.json").read_bytes())
    for rec in meta.get("_via_img_metadata", meta).values():
        via[sp, rec["filename"]] = rec["regions"]
by = defaultdict(list)
for i, r in enumerate(rows):
    by[split_of[r["filename"], r["region_index"]], r["filename"]].append(i)
updated, unchanged = np.zeros(len(rows)), np.zeros(len(rows))
for (sp, name), items in by.items():
    photo = Image.open(base / sp / "images" / name).convert("RGB")
    for i in items:
        r = rows[i]
        s = via[sp, name][int(r["region_index"])]["shape_attributes"]
        xs, ys = s["all_points_x"], s["all_points_y"]
        box = (min(xs), min(ys), max(xs) + 1, max(ys) + 1)
        crop = photo.crop(box)
        mask = Image.new("L", crop.size)
        ImageDraw.Draw(mask).polygon([(x - box[0], y - box[1]) for x, y in zip(xs, ys)], fill=255)
        occluded, _ = leaf.apply_leaf(crop, mask, .5, f"test/{name}/{r['apple_id']}", 20261116)
        kept = (np.asarray(mask) > 0) & ~np.any(np.asarray(crop) != np.asarray(occluded), axis=2)
        unchanged[i] = np.nan_to_num(ex.red_share(occluded, mask))
        updated[i] = np.nan_to_num(ex.red_share(occluded, kept)) if kept.any() else 0.0
y = np.array([r["label"] == "ripe" for r in rows])


def ap(s):
    h = y[np.argsort(-s, kind="stable")]
    return round(float((np.cumsum(h) / np.arange(1, len(h) + 1) * h).sum() / h.sum()), 3)


print("red share at 50% leaf: mask updated", ap(updated), "| mask unchanged", ap(unchanged),
      "| finetune seed1", ap(np.array([float(r["finetune_seed1"]) for r in rows])))
