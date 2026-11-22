"""Turn mask widths and depths into apple diameters (pinhole model) and score them against calliper values.

D [mm] = w [px] * Z [m] / f [px] * 1000                  (presentation slide 9)
D [mm] = w * Z / (f - w / 3) * 1000                       (same, with the visible-surface depth moved to the sphere centre)

Depth statistic, centre correction and hold threshold are computed on train+val and reported on test. This test
split is NOT independent: most of its apples are also in train/val and it was inspected while the method was
designed. See grouped_cv.py for the apple-separated score. The amodal masks were generated from the calliper diameter, so the amodal-width estimate is a
reference for depth error only, not a method that could run on new images.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEPTHS = ("depth_median_m", "depth_eroded_median_m", "depth_eroded_p25_m")
BINS = ((0.0, 0.15), (0.15, 0.30), (0.30, 0.50), (0.50, 1.01))
THRESHOLDS = tuple(round(0.50 + 0.05 * i, 2) for i in range(9))
MIN_DEPTH_VALID = 0.5


def load(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for r in rows:
        for k, v in r.items():
            if k not in ("split", "filename", "apple_id"):
                r[k] = float(v)
    return rows


def widths(r):
    vis_max = max(r["vis_width_px"], r["vis_height_px"])
    return {"vis_width": r["vis_width_px"], "vis_max": vis_max,
            "vis_area_eq": 2 * math.sqrt(r["vis_px"] / math.pi),
            "amodal_width": r["amodal_width_px"]}


def diameter(w, z, f, centre):
    return 1000 * w * z / (f - w / 3 if centre else f)


def fill_ratio(r):
    side = max(r["vis_width_px"], r["vis_height_px"])
    return r["vis_px"] / (math.pi / 4 * side ** 2)


def scores(est, gt):
    est, gt = np.asarray(est), np.asarray(gt)
    err = est - gt
    if not len(err):
        return dict(n=0)
    return dict(n=int(len(err)), mae_mm=float(np.abs(err).mean()), median_abs_mm=float(np.median(np.abs(err))),
                bias_mm=float(err.mean()), mape_pct=float(100 * np.mean(np.abs(err) / gt)),
                within_5mm_pct=float(100 * np.mean(np.abs(err) <= 5)),
                implied_weight_error_median_pct=float(100 * np.median(np.abs((est / gt) ** 3 - 1))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--target-mae", type=float, default=5.0,
                        help="Team assumption: acceptable mean diameter error (mm) of the views that are not held")
    args = parser.parse_args()
    rows = load(args.results / "instance_measurements.csv")
    usable = [r for r in rows if r["eligible"] and not math.isnan(r["gt_diameter_mm"])]
    report = dict(counts=dict(all_pairs=len(rows), eligible=sum(r["eligible"] == 1 for r in rows),
                              eligible_with_gt=len(usable),
                              by_split={s: sum(r["split"] == s for r in usable) for s in ("train", "val", "test")}))
    fit = [r for r in usable if r["split"] in ("train", "val")]
    test = [r for r in usable if r["split"] == "test"]

    # 1. Depth statistic and centre correction, chosen on train+val with the amodal width (depth error only).
    choices = {}
    for depth in DEPTHS:
        for centre in (False, True):
            ok = [r for r in fit if not math.isnan(r[depth]) and r["depth_valid_fraction"] >= MIN_DEPTH_VALID]
            est = [diameter(r["amodal_width_px"], r[depth], r["focal_px"], centre) for r in ok]
            choices[f"{depth}|centre={centre}"] = scores(est, [r["gt_diameter_mm"] for r in ok])
    best = min(choices, key=lambda k: choices[k]["mae_mm"])
    depth_key, centre = best.split("|")[0], best.endswith("True")
    report["depth_choice_on_train_val_amodal_width"] = dict(candidates=choices, chosen=best)

    def estimate(r):
        w = widths(r)
        return {k: diameter(v, r[depth_key], r["focal_px"], centre) for k, v in w.items()}

    def has_depth(r):
        return not math.isnan(r[depth_key]) and r["depth_valid_fraction"] >= MIN_DEPTH_VALID

    # 2. Test errors per width rule, overall and by natural occlusion.
    tested = [r for r in test if has_depth(r)]
    est = [estimate(r) for r in tested]
    gt = [r["gt_diameter_mm"] for r in tested]
    report["test"] = dict(
        depth_missing_hold=len(test) - len(tested),
        overall={k: scores([e[k] for e in est], gt) for k in est[0]},
        by_natural_occlusion={f"{lo:.2f}-{min(hi, 1):.2f}": {
            k: scores([e[k] for e, r in zip(est, tested) if lo <= r["natural_occlusion"] < hi],
                      [r["gt_diameter_mm"] for r in tested if lo <= r["natural_occlusion"] < hi]) for k in est[0]}
            for lo, hi in BINS})

    # 3. Hold rule from the visible mask alone: fill ratio = visible area / circle on the longer side.
    method = "vis_max"
    fit_ok = [r for r in fit if has_depth(r)]
    target = args.target_mae
    tradeoff = []
    for split_name, subset in (("train_val", fit_ok), ("test", tested)):
        e = np.array([estimate(r)[method] for r in subset])
        g = np.array([r["gt_diameter_mm"] for r in subset])
        fr = np.array([fill_ratio(r) for r in subset])
        for t in THRESHOLDS:
            keep = fr >= t
            tradeoff.append(dict(split=split_name, threshold=t, kept_pct=100 * keep.mean(),
                                 mae_kept_mm=float(np.abs(e - g)[keep].mean()) if keep.any() else float("nan"),
                                 mae_held_mm=float(np.abs(e - g)[~keep].mean()) if (~keep).any() else float("nan")))
    passing = [row for row in tradeoff if row["split"] == "train_val" and row["mae_kept_mm"] <= target]
    if not passing:
        raise ValueError(f"No fill-ratio threshold keeps train+val MAE <= {target} mm")
    chosen_t = min(row["threshold"] for row in passing)
    keep = np.array([fill_ratio(r) >= chosen_t for r in tested])
    e_m = np.array([e[method] for e in est])
    g_a = np.array(gt)
    report["hold_rule"] = dict(
        method=method, rule=f"hold if depth valid fraction < {MIN_DEPTH_VALID} or fill ratio < threshold",
        target=f"smallest threshold whose kept train+val views have MAE <= {target} mm (team assumption)",
        target_mae_mm=target, chosen_threshold=chosen_t,
        test_kept=scores(e_m[keep], g_a[keep]), test_held=scores(e_m[~keep], g_a[~keep]),
        test_kept_pct_of_eligible=float(100 * keep.sum() / len(test)),
        test_natural_occlusion_median_kept=float(np.median([r["natural_occlusion"] for r, k in zip(tested, keep) if k])),
        test_natural_occlusion_median_held=float(np.median([r["natural_occlusion"] for r, k in zip(tested, keep) if not k])))
    with (args.results / "hold_tradeoff.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tradeoff[0]))
        writer.writeheader()
        writer.writerows(tradeoff)

    # 4. One value per apple: median over its kept test views.
    per_apple = defaultdict(list)
    for r, e, k in zip(tested, est, keep):
        if k:
            per_apple[r["apple_id"]].append((e[method], r["gt_diameter_mm"]))
    apple_rows = [dict(apple_id=a, views=len(v), gt_diameter_mm=v[0][1], estimate_mm=float(np.median([x for x, _ in v])))
                  for a, v in sorted(per_apple.items())]
    for a in apple_rows:
        a["abs_error_mm"] = abs(a["estimate_mm"] - a["gt_diameter_mm"])
    report["per_apple_test"] = dict(
        apples=len(apple_rows), views_median=float(np.median([a["views"] for a in apple_rows])),
        **scores([a["estimate_mm"] for a in apple_rows], [a["gt_diameter_mm"] for a in apple_rows]))
    with (args.results / "per_apple_test.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(apple_rows[0]))
        writer.writeheader()
        writer.writerows(apple_rows)

    # Per-view test predictions for figures.
    out_rows = []
    for r, e, k in zip(tested, est, keep):
        out_rows.append(dict(filename=r["filename"], apple_id=r["apple_id"], gt_diameter_mm=r["gt_diameter_mm"],
                             natural_occlusion=r["natural_occlusion"], fill_ratio=fill_ratio(r),
                             depth_m=r[depth_key], focal_px=r["focal_px"], vis_width_px=r["vis_width_px"],
                             vis_height_px=r["vis_height_px"], amodal_width_px=r["amodal_width_px"],
                             kept=int(k), **{f"{m}_mm": v for m, v in e.items()}))
    with (args.results / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    (args.results / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
