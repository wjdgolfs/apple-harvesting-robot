"""Score the diameter rule with apples separated between fitting and evaluation (5 folds by apple ID).

Inside every fold the fill-ratio threshold (smallest with MAE <= target) and the constant scale are fitted on the
other apples only. Two cohorts are reported:
  visible_edge : outlines away from the photo edge, judged from the visible outline alone, as demo.py does
  full_outline : apples whose full (amodal) outline lies inside the photo; this needs the ground truth
'share' always has all 15,335 annotations as its denominator. 'demo_rule' applies the fixed settings the demo
uses to every annotation; those settings were fitted on part of the same data, so it is not an independent score.
The choices of the longer side and of the depth statistic were made earlier on this dataset and are not cross-validated.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np

import evaluate_diameter as ev

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "통합데모"))
import demo  # noqa: E402

FOLDS = 5


def fold_of(apple_id):
    return int(hashlib.sha256(apple_id.encode()).hexdigest(), 16) % FOLDS


def visible_boxes(archive):
    out = {}
    for split in ("train", "val", "test"):
        for row in json.loads(archive.read(f"{demo.ZIP_ROOT}/gt_json/{split}/via_region_data_modal.json")).values():
            for region in row["regions"].values():
                s = region["shape_attributes"]
                xs, ys = s["all_points_x"], s["all_points_y"]
                out[row["filename"], region["region_attributes"]["apple_ID"]] = min(xs), min(ys), max(xs), max(ys)
    return out


def summary(est, gt, total):
    err = np.asarray(est) - np.asarray(gt)
    return dict(n=int(len(err)), share=float(len(err) / total), mae_mm=float(np.abs(err).mean()), bias_mm=float(err.mean()))


def cross_validate(views, target, total):
    raw, scaled, gt, thresholds, scales, apples = [], [], [], [], [], set()
    for fold in range(FOLDS):
        fit = [v for v in views if fold_of(v["id"]) != fold]
        chosen = None
        for t in ev.THRESHOLDS:
            kept = [v for v in fit if v["fill"] >= t]
            if kept and np.mean([abs(v["est"] - v["gt"]) for v in kept]) <= target:
                chosen = t
                break
        if chosen is None:
            raise ValueError("No threshold meets the target in a training fold")
        s = sum(v["est"] * v["gt"] for v in kept) / sum(v["est"] ** 2 for v in kept)
        thresholds.append(chosen)
        scales.append(float(s))
        for v in views:
            if fold_of(v["id"]) == fold and v["fill"] >= chosen:
                raw.append(v["est"])
                scaled.append(v["est"] * s)
                gt.append(v["gt"])
                apples.add(v["id"])
    return dict(candidates=len(views), thresholds=thresholds, scales=scales, apples=len(apples),
                raw=summary(raw, gt, total), calibrated=summary(scaled, gt, total))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path("D:/Apple-harvest-data/02-annotated_data_fuji.zip"))
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--target-mae", type=float, default=5.0)
    args = parser.parse_args()
    rows = ev.load(args.results / "instance_measurements.csv")
    with ZipFile(args.zip) as archive:
        boxes = visible_boxes(archive)
    cohorts, holds = {"visible_edge": [], "full_outline": []}, {}
    demo_views = []
    settings = demo.load_settings()
    for r in rows:
        z = r["depth_eroded_median_m"]
        edge = demo.touches_edge(*boxes[r["filename"], r["apple_id"]], (1300, 1300))
        depth_ok = r["depth_valid_fraction"] >= ev.MIN_DEPTH_VALID and not math.isnan(z)
        fill = ev.fill_ratio(r)
        reason = ("edge" if edge else "depth" if not depth_ok else
                  "fill" if fill < settings["fill_threshold"] else "used")
        holds[reason] = holds.get(reason, 0) + 1
        if not depth_ok or math.isnan(r["gt_diameter_mm"]):
            continue
        view = dict(id=r["apple_id"], fill=fill, gt=r["gt_diameter_mm"],
                    est=ev.diameter(max(r["vis_width_px"], r["vis_height_px"]), z, r["focal_px"], False))
        if not edge:
            cohorts["visible_edge"].append(view)
        if r["eligible"]:
            cohorts["full_outline"].append(view)
        if reason == "used":
            demo_views.append(view)
    report = dict(annotations=len(rows), target_mae_mm=args.target_mae, folds=FOLDS,
                  edge_margin_px=demo.EDGE_MARGIN_PX,
                  cross_validation={k: cross_validate(v, args.target_mae, len(rows)) for k, v in cohorts.items()},
                  demo_rule=dict(threshold=settings["fill_threshold"], scale=settings["diameter_scale"], decisions=holds,
                                 **summary([v["est"] * settings["diameter_scale"] for v in demo_views],
                                           [v["gt"] for v in demo_views], len(rows))))
    (args.results / "grouped_cv.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
