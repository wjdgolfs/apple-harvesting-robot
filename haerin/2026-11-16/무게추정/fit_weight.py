"""Fit weight from coin-scaled apple diameter and test it with apple-level cross-validation.

Diameter correction: the coin lies on the table, the apple's widest section is about D/2 higher, so the coin
scale enlarges the apple by Z / (Z - D/2). The camera height Z comes from the coin size and the EXIF focal
length:  Z = f_px * 23.25 / coin_px,  D_true = D_coin / (1 + D_coin / (2 Z)).

Measurements outside 55-130 mm are coin-detection failures and are dropped (weight is not looked at).
Models are fitted on training folds only; folds keep 4 apples of each 20-apple weight batch, like 무게검증.
The orchard part converts the RGB-D diameter estimates of 지름추정 into grams with the fitted formula.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
COIN_MM = 23.25
VALID_MM = (55.0, 130.0)
VIEWS = ("spm_int_al_ver", "spm_int_me_ver", "spm_int_ce_ver")
SEED, FOLDS, REPEATS, BOOTSTRAP = 20261116, 5, 20, 4000


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def corrected_diameter(r):
    d = float(r["d_eq_mm"])
    z = float(r["focal_px"]) * COIN_MM / (2 * float(r["coin_r_px"]))
    return d / (1 + d / (2 * z)), z


def fold_table(batch):
    table = np.empty((REPEATS, len(batch)), dtype=int)
    for rep in range(REPEATS):
        rng = np.random.default_rng([SEED, rep])
        for b in sorted(set(batch)):
            members = np.flatnonzero(batch == b)
            table[rep, rng.permutation(members)] = np.arange(len(members)) % FOLDS
    return table


def fit_power(d, w):
    b, log_a = np.polyfit(np.log(d), np.log(w), 1)
    return math.exp(log_a), b


def fit_cube(d, w):
    return float(np.sum(w * d ** 3) / np.sum(d ** 6)), 3.0


def cross_validate(d, w, batch, folds, fitter):
    preds = np.full(folds.shape, np.nan)
    for rep, fold in enumerate(folds):
        for f in range(FOLDS):
            test = fold == f
            train = ~test & ~np.isnan(d)
            if fitter == "mean":
                preds[rep, test] = w[train].mean()
            elif fitter == "batch_mean":
                means = {b: w[train & (batch == b)].mean() for b in set(batch)}
                preds[rep, test] = [means[b] for b in batch[test]]
            else:
                a, b = fitter(d[train], w[train])
                preds[rep, test] = a * d[test] ** b
    return preds


def rank_corr(x, y):
    rx, ry = rankdata(x), rankdata(y)  # average ranks for ties
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def summary(preds, w, batch):
    ok = ~np.isnan(preds[0])
    err = np.abs(preds[:, ok] - w[ok])
    avg = preds[:, ok].mean(axis=0)
    within = {b: rank_corr(avg[batch[ok] == b], w[ok][batch[ok] == b]) for b in sorted(set(batch))}
    return dict(apples=int(ok.sum()), mae_g=float(err.mean()), mae_repeat_range_g=[float(err.mean(1).min()), float(err.mean(1).max())],
                mape_pct=float(100 * (err / w[ok]).mean()), within_batch_rank_corr=within,
                within_batch_rank_corr_mean=float(np.mean(list(within.values()))))


def paired(preds_a, preds_b, w, batch):
    ok = ~np.isnan(preds_a[0]) & ~np.isnan(preds_b[0])
    ea, eb = np.abs(preds_a[:, ok] - w[ok]).mean(0), np.abs(preds_b[:, ok] - w[ok]).mean(0)
    rng = np.random.default_rng(SEED)
    pools = [np.flatnonzero(batch[ok] == b) for b in sorted(set(batch))]
    draws = [np.mean(np.concatenate([ea[s] - eb[s] for s in [rng.choice(p, len(p)) for p in pools]])) for _ in range(BOOTSTRAP)]
    return dict(difference_g=float((ea - eb).mean()), ci95_g=[float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    rows = read_csv(args.results / "size_measurements.csv")
    groups = sorted({r["group"] for r in rows})
    info = {r["group"]: (float(r["weight_g"]), r["weight_batch"]) for r in rows}
    w = np.array([info[g][0] for g in groups])
    batch = np.array([info[g][1] for g in groups])
    per_view = {v: np.full(len(groups), np.nan) for v in VIEWS}
    heights, dropped = defaultdict(list), defaultdict(int)
    for r in rows:
        if not r.get("d_eq_mm") or not VALID_MM[0] <= float(r["d_eq_mm"]) <= VALID_MM[1]:
            dropped[r["view"]] += 1
            continue
        d, z = corrected_diameter(r)
        per_view[r["view"]][groups.index(r["group"])] = d
        heights[r["view"]].append(z)
    stacked = np.vstack([per_view[v] for v in VIEWS])
    multi = np.nanmedian(stacked, axis=0)
    raw_gap = []
    for r_ce in (r for r in rows if r["view"] == "spm_int_ce_ver" and r.get("d_eq_mm")):
        r_al = next(r for r in rows if r["view"] == "spm_int_al_ver" and r["group"] == r_ce["group"])
        if r_al.get("d_eq_mm") and all(VALID_MM[0] <= float(x["d_eq_mm"]) <= VALID_MM[1] for x in (r_ce, r_al)):
            raw_gap.append(float(r_ce["d_eq_mm"]) - float(r_al["d_eq_mm"]))
    both = ~np.isnan(per_view["spm_int_ce_ver"]) & ~np.isnan(per_view["spm_int_al_ver"])
    report = dict(
        measurement=dict(views=VIEWS, dropped_as_coin_failure=dict(dropped),
                         camera_height_median_mm={v: float(np.median(h)) for v, h in heights.items()},
                         near_minus_far_diameter_median_mm=dict(
                             before_correction=float(np.median(raw_gap)),
                             after_correction=float(np.median(per_view["spm_int_ce_ver"][both] - per_view["spm_int_al_ver"][both]))),
                         diameter_range_mm=[float(np.nanmin(multi)), float(np.nanmax(multi))],
                         diameter_weight_rank_corr_within_batch={b: rank_corr(multi[batch == b], w[batch == b]) for b in sorted(set(batch))}),
        settings=dict(folds=FOLDS, repeats=REPEATS, seed=SEED, valid_range_mm=VALID_MM, coin_mm=COIN_MM))

    folds = fold_table(batch)
    preds = {"global_mean": cross_validate(multi, w, batch, folds, "mean"),
             "batch_mean_reference": cross_validate(multi, w, batch, folds, "batch_mean"),
             "power_me_view": cross_validate(per_view["spm_int_me_ver"], w, batch, folds, fit_power),
             "power_three_views": cross_validate(multi, w, batch, folds, fit_power),
             "cube_three_views": cross_validate(multi, w, batch, folds, fit_cube)}
    report["cross_validation"] = {k: summary(p, w, batch) for k, p in preds.items()}
    report["paired_differences"] = dict(
        three_views_minus_global_mean=paired(preds["power_three_views"], preds["global_mean"], w, batch),
        three_views_minus_batch_mean=paired(preds["power_three_views"], preds["batch_mean_reference"], w, batch))
    a, b = fit_power(multi, w)
    k, _ = fit_cube(multi, w)
    report["formula_all_60"] = dict(power=dict(a=a, b=b, text=f"weight_g = {a:.3e} * D_mm^{b:.3f}"),
                                    cube=dict(k=k, text=f"weight_g = {k:.4e} * D_mm^3"),
                                    cube_density_times_shape_g_per_cm3=k / (math.pi / 6) * 1000)

    # Orchard: RGB-D diameters (지름추정) -> grams with the all-60 formula.
    orchard = read_csv(BASE / "지름추정/results/per_apple_test.csv")
    d_gt = np.array([float(r["gt_diameter_mm"]) for r in orchard])
    d_est = np.array([float(r["estimate_mm"]) for r in orchard])
    g_gt, g_est = a * d_gt ** b, a * d_est ** b
    inside = (d_gt >= multi.min()) & (d_gt <= multi.max())
    report["orchard_propagation"] = dict(
        apples=len(orchard), inside_lab_diameter_range=int(inside.sum()),
        weight_from_calliper_diameter_g=dict(median=float(np.median(g_gt)), range=[float(g_gt.min()), float(g_gt.max())]),
        weight_change_from_diameter_error_g=dict(mae=float(np.mean(np.abs(g_est - g_gt))), median=float(np.median(np.abs(g_est - g_gt))),
                                                 bias=float(np.mean(g_est - g_gt)),
                                                 mae_inside_range=float(np.mean(np.abs(g_est - g_gt)[inside]))),
        note="No orchard apple was weighed: this is the extra error caused by the diameter estimate, on top of the formula error measured in the lab.")

    with (args.results / "apple_diameters.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["group", "weight_batch", "weight_g", *VIEWS, "diameter_mm_median",
                                                    "cv_estimate_g"])
        writer.writeheader()
        cv = preds["power_three_views"].mean(axis=0)
        for i, g in enumerate(groups):
            writer.writerow(dict(group=g, weight_batch=batch[i], weight_g=w[i], **{v: per_view[v][i] for v in VIEWS},
                                 diameter_mm_median=multi[i], cv_estimate_g=cv[i]))
    (args.results / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
