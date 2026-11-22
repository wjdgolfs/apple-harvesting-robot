"""Correct the systematic under-estimate of the visible-mask diameter.

Only views that pass the hold rule (depth valid, fill ratio >= chosen threshold) are used, as in the demo.
Candidates (ratio = calliper / estimate):
  none        D
  scale       D * s                      (one constant)
  fill        D * (a + b * (1 - fill))   (more correction when more of the circle is missing)
Fit on train, choose on val, refit the chosen rule on train+val, then report test. The splits share apples and test
was seen during design, so this is not an independent score (see grouped_cv.py).
"""

import argparse
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ev = load_module("evaluate_diameter", HERE / "evaluate_diameter.py")


def kept_views(rows, threshold):
    out = []
    for r in rows:
        if not r["eligible"] or math.isnan(r["gt_diameter_mm"]) or math.isnan(r["depth_eroded_median_m"]):
            continue
        if r["depth_valid_fraction"] < ev.MIN_DEPTH_VALID or ev.fill_ratio(r) < threshold:
            continue
        d = ev.diameter(max(r["vis_width_px"], r["vis_height_px"]), r["depth_eroded_median_m"], r["focal_px"], False)
        out.append(dict(split=r["split"], apple_id=r["apple_id"], gt=r["gt_diameter_mm"], est=d,
                        missing=1 - min(ev.fill_ratio(r), 1.0)))
    return out


def fit(rule, views):
    est = np.array([v["est"] for v in views])
    gt = np.array([v["gt"] for v in views])
    if rule == "none":
        return {}
    if rule == "scale":
        return {"s": float(np.sum(gt * est) / np.sum(est ** 2))}
    x = np.column_stack([est, est * np.array([v["missing"] for v in views])])
    a, b = np.linalg.lstsq(x, gt, rcond=None)[0]
    return {"a": float(a), "b": float(b)}


def apply(rule, params, v):
    if rule == "none":
        return v["est"]
    if rule == "scale":
        return v["est"] * params["s"]
    return v["est"] * (params["a"] + params["b"] * v["missing"])


def score(rule, params, views):
    est = [apply(rule, params, v) for v in views]
    return ev.scores(est, [v["gt"] for v in views])


def per_apple(rule, params, views):
    by = defaultdict(list)
    for v in views:
        by[v["apple_id"]].append((apply(rule, params, v), v["gt"]))
    est = [float(np.median([e for e, _ in x])) for x in by.values()]
    return ev.scores(est, [x[0][1] for x in by.values()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    metrics = json.loads((args.results / "metrics.json").read_text(encoding="utf-8"))
    if metrics["depth_choice_on_train_val_amodal_width"]["chosen"] != "depth_eroded_median_m|centre=False":
        raise ValueError("Calibration assumes the eroded-median depth without centre correction")
    threshold = metrics["hold_rule"]["chosen_threshold"]
    views = kept_views(ev.load(args.results / "instance_measurements.csv"), threshold)
    split = {s: [v for v in views if v["split"] == s] for s in ("train", "val", "test")}
    rules = ("none", "scale", "fill")
    on_val = {}
    for rule in rules:
        params = fit(rule, split["train"])
        on_val[rule] = dict(params=params, val=score(rule, params, split["val"]))
    chosen = min(rules, key=lambda r: on_val[r]["val"]["mae_mm"])
    final = fit(chosen, split["train"] + split["val"])
    report = dict(hold_threshold=threshold, views={s: len(v) for s, v in split.items()},
                  selection_on_val=on_val, chosen=chosen, params_train_val=final,
                  test={rule: dict(per_view=score(rule, fit(rule, split["train"] + split["val"]), split["test"]),
                                   per_apple=per_apple(rule, fit(rule, split["train"] + split["val"]), split["test"]))
                        for rule in rules})
    (args.results / "bias_calibration.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("views", "chosen", "params_train_val")}, ensure_ascii=False))
    for rule in rules:
        t = report["test"][rule]
        print(f"{rule:6} val MAE {on_val[rule]['val']['mae_mm']:.2f} | test view MAE {t['per_view']['mae_mm']:.2f} "
              f"bias {t['per_view']['bias_mm']:+.2f} | apple MAE {t['per_apple']['mae_mm']:.2f} bias {t['per_apple']['bias_mm']:+.2f} "
              f"within5 {t['per_apple']['within_5mm_pct']:.0f}%")


if __name__ == "__main__":
    main()
