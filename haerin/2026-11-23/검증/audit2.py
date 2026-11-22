"""Second pass: identical-pixel label conflicts, YOLO matching by an independent rasteriser, weight sensitivity."""
import csv
import hashlib
import io
import json
import math
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

L = lambda p: list(csv.DictReader(open(p, encoding="utf-8-sig")))
DATA = Path("D:/Apple-harvest-data")

# 1. Same apple, same camera frame: are the cropped pixels identical while labels differ?
linked = L("integration/results/linked_instances.csv")
groups = defaultdict(list)
for r in linked:
    groups[r["apple_id"], re.match(r"(_MG_\d+)_", r["filename"])[1]].append(r)
conflict = {k: v for k, v in groups.items() if len({r["label"] for r in v}) > 1}
via = {}
for sp in ("train", "val", "test"):
    meta = json.loads((DATA / "ripeness" / sp / f"{sp}_via_final.json").read_bytes())
    for rec in meta.get("_via_img_metadata", meta).values():
        via[sp, rec["filename"]] = rec["regions"]


def key(r):
    s = via[r["split"], r["filename"]][int(r["region_index"])]["shape_attributes"]
    xs, ys = s["all_points_x"], s["all_points_y"]
    with Image.open(DATA / "ripeness" / r["split"] / "images" / r["filename"]) as im:
        crop = np.asarray(im.convert("RGB").crop((min(xs), min(ys), max(xs) + 1, max(ys) + 1)))
    return crop.shape, hashlib.sha1(crop.tobytes()).hexdigest()


identical = same_size = 0
examples = []
for k, v in sorted(conflict.items()):
    seen = defaultdict(set)
    sizes = defaultdict(set)
    for r in v:
        shape, digest = key(r)
        seen[digest].add(r["label"])
        sizes[shape].add(r["label"])
    if any(len(x) > 1 for x in seen.values()):
        identical += 1
        if len(examples) < 5:
            examples.append((k, [(r["filename"], r["label"]) for r in v]))
    if any(len(x) > 1 for x in sizes.values()):
        same_size += 1
with_ripe = sum(len(v) > 1 and any(r["label"] == "ripe" for r in v) for v in groups.values())
print("conflict groups", len(conflict), "of", with_ripe, "| identical pixels", identical, "| same crop size", same_size)
print(examples)

# 2. YOLO matching recomputed with OpenCV rasterisation and optimal assignment.
from scipy.optimize import linear_sum_assignment

with zipfile.ZipFile(DATA / "02-annotated_data_fuji.zip") as z:
    gt = {}
    for sp in ("train", "val", "test"):
        for rec in json.loads(z.read(f"02-annotated_data_fuji/gt_json/{sp}/via_region_data_modal.json")).values():
            gt[rec["filename"]] = list(rec["regions"].values())


def mask(points):
    m = np.zeros((1300, 1300), np.uint8)
    cv2.fillPoly(m, [np.array(points, np.int32)], 1)
    return m.astype(bool)


for model in ("n-seg", "x-seg"):
    total = [0, 0, 0]
    ious = []
    for f in sorted(Path(f"통합데모/detections_coco_{model}").glob("_MG*.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        g = [mask(list(zip(r["shape_attributes"]["all_points_x"], r["shape_attributes"]["all_points_y"]))) for r in gt[j["image"]]]
        d = [mask(i["polygon"]) for i in j["instances"]]
        m = np.array([[(a & b).sum() / (a | b).sum() for b in d] for a in g]) if d else np.zeros((len(g), 0))
        good = [m[a, b] for a, b in zip(*linear_sum_assignment(-(m * (m >= .5)))) if m[a, b] >= .5] if d else []
        total = [total[0] + len(g), total[1] + len(d), total[2] + len(good)]
        ious += good
    print("yolo", model, total, "mean IoU", round(float(np.mean(ious)), 3))

# 3. Weight sensitivity to the visibly wrong apple mask that passed the 55-130 mm filter (141 g, al view).
A = L("무게추정/results/apple_diameters.csv")
V = ("spm_int_al_ver", "spm_int_me_ver", "spm_int_ce_ver")
w = np.array([float(r["weight_g"]) for r in A])
b = np.array([r["weight_batch"] for r in A])
D = np.array([[float(r[v]) if r[v] not in ("", "nan") else np.nan for v in V] for r in A])
bad = int(np.argmin(abs(w - 141.14)))
print("bad mask apple", A[bad]["group"], D[bad])


def run(D):
    d = np.nanmedian(D, 1)
    e = []
    for rep in range(20):
        rng = np.random.default_rng([20261116, rep])
        f = np.empty(len(w), int)
        for u in sorted(set(b)):
            f[rng.permutation(np.flatnonzero(b == u))] = np.arange(20) % 5
        for k in range(5):
            s, i = np.polyfit(np.log(d[f != k]), np.log(w[f != k]), 1)
            e += list(abs(np.exp(i) * d[f == k] ** s - w[f == k]))
    return round(float(np.mean(e)), 2)


fixed = D.copy()
fixed[bad, 0] = np.nan
print("weight MAE as reported", run(D), "| without the wrong mask", run(fixed))
