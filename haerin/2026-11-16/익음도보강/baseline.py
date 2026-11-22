"""No-training baseline for ripeness: share of red pixels, on the same test crops and leaves as the networks.

Three inputs, from most to least information:
  red_share      : inside the visible polygon, minus the synthetic leaf (needs a mask and knows the leaf)
  red_share_mask : inside the visible polygon, leaf included (needs a mask)
  red_share_box  : whole box crop (exactly what the networks see)
Also records how far label conflicts between overlapping tiles of one raw frame depend on apples cut by the tile.
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import label_examples
import locate
from multiview import ap
from prepare_data import SEED, leaf

HERE = Path(__file__).resolve().parent
LEVELS = (0, 15, 30, 50)
NAMES = ("red_share", "red_share_mask", "red_share_box")
EDGE_LOW, EDGE_HIGH = 137 + 10, 1160 - 10  # photo area 137..1160 and the same 10 px margin as 통합데모/demo.py


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def three_inputs(crop, mask, key, level):
    shown, kept = crop, mask
    if level:
        shown, _ = leaf.apply_leaf(crop, mask, level / 100, key, SEED)
        changed = np.any(np.asarray(crop) != np.asarray(shown), axis=2)
        kept = (np.asarray(mask) > 0) & ~changed
    whole = np.ones((crop.height, crop.width), bool)
    return [np.nan_to_num(label_examples.red_share(shown, m)) for m in (kept, mask, whole)]


def label_conflicts():
    groups = defaultdict(list)
    for r in read(HERE.parent / "integration/results/linked_instances.csv"):
        shape = locate.regions(r["split"])[r["filename"]][int(r["region_index"])]["shape_attributes"]
        xs, ys = shape["all_points_x"], shape["all_points_y"]
        cut = min(xs) <= EDGE_LOW or min(ys) <= EDGE_LOW or max(xs) >= EDGE_HIGH or max(ys) >= EDGE_HIGH
        groups[r["apple_id"], re.match(r"(_MG_\d+)_", r["filename"])[1]].append(dict(label=r["label"], cut=cut))

    def count(sets):
        chosen = [v for v in sets if len(v) > 1 and any(x["label"] == "ripe" for x in v)]
        return dict(groups=len(chosen), conflicting=sum(len({x["label"] for x in v}) > 1 for v in chosen))

    return dict(unit="same apple in the same raw frame, two or more overlapping tiles, at least one labelled ripe",
                all_tiles=count(groups.values()),
                uncut_tiles_only=count([[x for x in v if not x["cut"]] for v in groups.values()]))


def main():
    rows = read(HERE / "results/test_scores.csv")
    out = []
    for r in sorted(rows, key=lambda r: (r["filename"], int(r["region_index"]), int(r["cover_percent"]))):
        photo, box, points = locate.region(r["filename"], r["region_index"])
        crop = photo.crop(box)
        mask = Image.new("L", crop.size)
        ImageDraw.Draw(mask).polygon([(x - box[0], y - box[1]) for x, y in points], fill=255)
        values = three_inputs(crop, mask, f"test/{r['filename']}/{r['apple_id']}", int(r["cover_percent"]))
        out.append(dict({k: r[k] for k in ("filename", "region_index", "apple_id", "label", "cover_percent")},
                        **dict(zip(NAMES, values))))
    with (HERE / "results/baseline_scores.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    report = dict(ap={}, label_conflicts=label_conflicts())
    for level in LEVELS:
        subset = [r for r in out if int(r["cover_percent"]) == level]
        y = np.array([r["label"] == "ripe" for r in subset])
        report["ap"][str(level)] = {n: ap(y, np.array([r[n] for r in subset])) for n in NAMES}
    (HERE / "results/baseline.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
