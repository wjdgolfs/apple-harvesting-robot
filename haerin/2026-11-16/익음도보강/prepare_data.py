"""Build a larger, apple-ID-disjoint ripeness dataset from every linked Fuji-Ripeness-Size region.

Split rule
- test apples = the 115 apple IDs of the earlier ResNet18 test cohort (kept fixed for comparison).
- other apples: validation if a preset hash of the ID falls in 1/5, otherwise training.
- a tile (one PNG) belongs to the strictest split among its apples (test > val > train); regions of other
  splits' apples inside that tile are dropped, so no apple and no tile is shared between splits.
- tiles used by the earlier cohort's training or validation never enter the test split.

Crops follow the earlier preprocessing exactly: tight VIA box, longest side 160 px, mean-colour square.
Leaf occlusion reuses the earlier generator: one random level per training crop, 15/30/50% per test crop.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SEED = 20261116
TEST_LEVELS = (15, 30, 50)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load("prepare_ripeness", BASE / "전이학습/ripeness/prepare_ripeness.py")
baseline = load("ripeness_baseline", BASE / "ripeness/ripeness_baseline.py")
leaf = load("synthetic_leaf", BASE / "occlusion/synthetic_leaf.py")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def bucket(text, modulo):
    return int(hashlib.sha256(f"{SEED}|{text}".encode()).hexdigest()[:8], 16) % modulo


def assign(linked, cohort):
    old_split = {r["apple_id"]: r["split"] for r in cohort}
    old_tiles = {r["filename"] for r in cohort if r["split"] in ("train", "val")}
    id_split = {}
    for r in linked:
        a = r["apple_id"]
        if old_split.get(a) == "test":
            id_split[a] = "test"
        else:
            id_split[a] = "val" if bucket(f"val:{a}", 5) == 0 else "train"
    tile_ids = defaultdict(set)
    for r in linked:
        tile_ids[r["filename"]].add(id_split[r["apple_id"]])
    tile_split = {t: next(s for s in ("test", "val", "train") if s in ids) for t, ids in tile_ids.items()}
    kept = []
    for r in linked:
        split = tile_split[r["filename"]]
        if id_split[r["apple_id"]] != split:
            continue
        if split == "test" and r["filename"] in old_tiles:
            continue
        kept.append(dict(r, new_split=split))
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("D:/Apple-harvest-data/ripeness"))
    parser.add_argument("--cache", type=Path, default=HERE / "cache")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    linked = read_csv(BASE / "integration/results/linked_instances.csv")
    cohort = read_csv(BASE / "ripeness/results/linked_predictions.csv")
    old_test = {(r["filename"], r["region_index"]) for r in cohort if r["split"] == "test"}
    kept = assign(linked, cohort)
    if sum((r["filename"], r["region_index"]) in old_test for r in kept) != len(old_test):
        raise ValueError("The earlier test cohort is not fully contained in the new test split")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        for field in ("apple_id", "filename"):
            if {r[field] for r in kept if r["new_split"] == a} & {r[field] for r in kept if r["new_split"] == b}:
                raise ValueError(f"{field} shared between {a} and {b}")

    via = {}
    for split in ("train", "val", "test"):
        meta = json.loads((args.data / split / f"{split}_via_final.json").read_bytes())
        for rec in meta.get("_via_img_metadata", meta).values():
            via[split, rec["filename"]] = rec["regions"]

    by_tile = defaultdict(list)
    for r in kept:
        by_tile[r["split"], r["filename"]].append(r)
    total = sum(1 + (1 if r["new_split"] == "train" else len(TEST_LEVELS) if r["new_split"] == "test" else 0) for r in kept)
    crops = np.lib.format.open_memmap(args.cache / "crops.npy", mode="w+", dtype=np.uint8, shape=(total, 160, 160, 3))
    index, n = [], 0
    for t, ((split, name), rows) in enumerate(sorted(by_tile.items())):
        with Image.open(args.data / split / "images" / name) as photo:
            photo = photo.convert("RGB")
        for r in sorted(rows, key=lambda r: int(r["region_index"])):
            region = via[split, name][int(r["region_index"])]
            if region["region_attributes"]["Ripeness"] != r["label"]:
                raise ValueError(f"Label mismatch {name}/{r['region_index']}")
            shape = region["shape_attributes"]
            box = baseline.region_box(shape)
            crop = photo.crop(box)
            mask = Image.new("L", crop.size, 0 if shape["name"] == "polygon" else 255)
            if shape["name"] == "polygon":
                ImageDraw.Draw(mask).polygon([(x - box[0], y - box[1]) for x, y in
                                              zip(shape["all_points_x"], shape["all_points_y"])], fill=255)
            common = dict(split=r["new_split"], filename=name, region_index=r["region_index"], apple_id=r["apple_id"],
                          label=r["label"], in_old_test=int((name, r["region_index"]) in old_test))
            variants = [(0, crop)]
            if r["new_split"] == "train":
                level = (15, 30, 50)[bucket(f"level:{name}:{r['region_index']}", 3)]
                variants.append((level, leaf.apply_leaf(crop, mask, level / 100, f"train/{name}/{r['region_index']}", SEED)[0]))
            elif r["new_split"] == "test":
                # Same key as the earlier cohort so its paired leaf images are reproduced exactly.
                variants += [(lv, leaf.apply_leaf(crop, mask, lv / 100, f"test/{name}/{r['apple_id']}", SEED)[0])
                             for lv in TEST_LEVELS]
            for level, image in variants:
                crops[n] = np.asarray(prepare.fit_square(image))
                index.append(dict(row=n, cover_percent=level, **common))
                n += 1
        if (t + 1) % 300 == 0:
            print(f"{t + 1}/{len(by_tile)} tiles, {n} crops", flush=True)
    crops.flush()
    if n != total:
        raise ValueError(f"Wrote {n} crops, expected {total}")
    with (args.cache / "index.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)
    counts = {s: dict(regions=sum(r["new_split"] == s for r in kept),
                      ripe=sum(r["new_split"] == s and r["label"] == "ripe" for r in kept),
                      apple_ids=len({r["apple_id"] for r in kept if r["new_split"] == s}),
                      tiles=len({r["filename"] for r in kept if r["new_split"] == s})) for s in ("train", "val", "test")}
    summary = dict(linked_regions=len(linked), kept_regions=len(kept), dropped_regions=len(linked) - len(kept),
                   splits=counts, old_test_regions=len(old_test), crops=n, seed=SEED)
    (args.cache / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
