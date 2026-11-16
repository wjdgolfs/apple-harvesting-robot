"""Train two binary linear heads on frozen ResNet18 features and evaluate fixed test crops."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F


LEVELS = (0, 15, 30, 50)
L2 = 0.1


def fit_head(x, y):
    """Equal class weights, an unpenalized bias and L2-penalized standardized weights."""
    x = x.double()
    y = y.double()
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(0.05)
    z = (x - mean) / scale
    w = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([w, b], lr=1, max_iter=120, line_search_fn="strong_wolfe",
                                  tolerance_grad=1e-7, tolerance_change=1e-12)
    sample_weights = torch.where(y == 1, 0.5 / (y == 1).sum(), 0.5 / (y == 0).sum())

    def objective():
        optimizer.zero_grad()
        logits = z.mv(w) + b
        loss = (sample_weights * F.binary_cross_entropy_with_logits(logits, y, reduction="none")).sum()
        loss = loss + (L2 / 2) * w.square().sum()
        loss.backward()
        return loss

    optimizer.step(objective)
    return dict(weight=w.detach(), bias=b.detach(), mean=mean, scale=scale)


def score(head, x):
    return ((x.double() - head["mean"]) / head["scale"]).mv(head["weight"]) + head["bias"]


def val_threshold(scores, y):
    """Maximize validation balanced accuracy; equivalent thresholds nearest zero take priority."""
    s = scores.tolist()
    truth = y.bool().tolist()
    pos, neg = sum(truth), len(truth) - sum(truth)
    candidates = sorted(set(s))
    midpoints = [(a + b) / 2 for a, b in zip(candidates, candidates[1:])]
    thresholds = [math.nextafter(candidates[0], -math.inf), *midpoints, math.nextafter(candidates[-1], math.inf), 0.0]

    def key(t):
        tp = sum(v >= t and actual for v, actual in zip(s, truth))
        tn = sum(v < t and not actual for v, actual in zip(s, truth))
        return tp * neg + tn * pos, -abs(t), t

    return max(thresholds, key=key)


def metrics(rows, field):
    # The first letter of each key is truth and the second is prediction.
    counts = Counter((r["label"], r[field]) for r in rows)
    tp = counts["ripe", "ripe"]
    fn = counts["ripe", "unripe"]
    tn = counts["unripe", "unripe"]
    fp = counts["unripe", "ripe"]
    return {
        "n": len(rows), "n_ids": len({r["apple_id"] for r in rows}),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "accuracy": (tp + tn) / len(rows),
        "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2,
        "ripe_recall": tp / (tp + fn), "unripe_recall": tn / (tn + fp),
        "ripe_precision": tp / (tp + fp) if tp + fp else None,
    }


def evaluate(rows, field, level):
    level_rows = [r for r in rows if r["cover_percent"] == level]
    original = {(r["filename"], r["apple_id"]): r[field] for r in rows if r["cover_percent"] == 0}
    stats = metrics(level_rows, field)
    stats["changed_vs_original"] = sum(r[field] != original[r["filename"], r["apple_id"]] for r in level_rows)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    cached = torch.load(args.cache / "features.pt", map_location="cpu", weights_only=True)
    x = cached["features"]
    with (args.cache / "index.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["cover_percent"] = int(row["cover_percent"])
    if len(rows) != len(x) or not torch.isfinite(x).all():
        raise ValueError("Feature matrix does not match cohort or contains nonfinite values")
    index_digest = hashlib.sha256((args.cache / "index.csv").read_bytes()).hexdigest()
    y = torch.tensor([r["label"] == "ripe" for r in rows], dtype=torch.float64)
    val_idx = [i for i, r in enumerate(rows) if r["split"] == "val" and r["cover_percent"] == 0]
    test_idx = [i for i, r in enumerate(rows) if r["split"] == "test"]
    test_rows = [rows[i] for i in test_idx]
    val_rows = [rows[i] for i in val_idx]
    results = {"encoder": cached["metadata"], "index_sha256": index_digest, "l2": L2,
               "head": "Class-balanced logistic regression on train-standardized frozen 512-dimensional features",
               "threshold_selection": "Maximize balanced accuracy on 338 unmodified validation observations; closest to zero in ties",
               "split_units": {}, "fits": {}, "test": {}}
    for split in ("train", "val", "test"):
        originals = [r for r in rows if r["split"] == split and r["cover_percent"] == 0]
        results["split_units"][split] = dict(objects=len(originals), fruit_ids=len({r["apple_id"] for r in originals}),
                                              frames=len({r["source_frame"] for r in originals}),
                                              counts=dict(Counter(r["label"] for r in originals)))
    for name in ("clean", "aug"):
        train_idx = [i for i, r in enumerate(rows) if r["split"] == "train" and (name == "aug" or r["cover_percent"] == 0)]
        head = fit_head(x[train_idx], y[train_idx])
        val_scores = score(head, x[val_idx])
        threshold = val_threshold(val_scores, y[val_idx])
        head["threshold"] = threshold
        head["positive_label"] = "ripe"
        head["preprocessing"] = cached["metadata"]
        torch.save(head, args.out / f"head_{name}.pt")
        for subset, subset_scores in [(val_rows, val_scores), (test_rows, score(head, x[test_idx]))]:
            for row, value in zip(subset, subset_scores.tolist()):
                row[name + "_score"] = value
                row[name + "_prediction"] = "ripe" if value >= threshold else "unripe"
        results["fits"][name] = {"training_views": len(train_idx), "threshold_logit": threshold,
                                  "validation": metrics(val_rows, name + "_prediction")}
    for name in ("clean", "aug", "color_clean", "color_aug"):
        results["test"][name] = {str(level): evaluate(test_rows, name + "_prediction", level) for level in LEVELS}
    for file, subset in (("predictions.csv", test_rows), ("validation_predictions.csv", val_rows)):
        with (args.out / file).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(subset[0]))
            writer.writeheader()
            writer.writerows(subset)
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    for name, scores in results["test"].items():
        print(name, [(level, m["tp"], m["fp"], round(m["balanced_accuracy"], 4)) for level, m in scores.items()])


if __name__ == "__main__":
    main()
