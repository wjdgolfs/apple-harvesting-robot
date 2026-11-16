"""Paired synthetic-leaf evaluation for the frozen ripeness rule and its augmented fit."""

import argparse
import csv
import importlib.util
import itertools
import json
import statistics
from pathlib import Path

from PIL import Image, ImageDraw
from synthetic_leaf import apply_leaf


LEVELS = (0.00, 0.15, 0.30, 0.50)


def load_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def summarize(rows, threshold):
    stats = {}
    for level in LEVELS:
        group = [row for row in rows if row["target_fraction"] == level]
        if not group:
            continue
        correct = {"ripe": 0, "unripe": 0}
        total = {"ripe": 0, "unripe": 0}
        flips = 0
        for row in group:
            pred = "ripe" if row["score"] >= threshold else "unripe"
            clean = "ripe" if row["clean_score"] >= threshold else "unripe"
            truth = row["label"]
            total[truth] += 1
            correct[truth] += pred == truth
            flips += pred != clean
        recall = {label: correct[label] / count if count else None for label, count in total.items()}
        stats[str(level)] = {
            "n_instances": len(group), "n_fruit_ids": len({r["apple_id"] for r in group}),
            "labels": total, "correct_by_label": correct, "accuracy": sum(correct.values()) / len(group),
            "balanced_accuracy": statistics.mean(recall.values()) if all(total.values()) else None,
            "recall": recall, "changed_prediction": flips, "changed_fraction": flips / len(group),
            "mean_actual_fraction": statistics.mean(r["actual_fraction"] for r in group),
            "largest_abs_target_error": max(abs(r["actual_fraction"] - level) for r in group),
        }
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("D:/Apple-harvest-data/ripeness"))
    parser.add_argument("--cohort", type=Path, required=True, help="linked_predictions.csv")
    parser.add_argument("--method", type=Path, required=True, help="ripeness_baseline.py")
    parser.add_argument("--natural", type=Path, required=True, help="natural_occlusion.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261116)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("ripeness_baseline", args.method)
    method = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(method)

    cohort = load_rows(args.cohort)
    natural = {(r["filename"], r["apple_id"]): r for r in load_rows(args.natural)}
    train_clean, train_augmented, test = [], [], []
    for split in ("train", "test"):
        metadata = json.loads((args.data / split / f"{split}_via_final.json").read_bytes())
        lookup = {r["filename"]: r["regions"] for r in metadata.get("_via_img_metadata", metadata).values()}
        subset = sorted((r for r in cohort if r["split"] == split), key=lambda r: r["filename"])
        for name, group in itertools.groupby(subset, key=lambda r: r["filename"]):
            with Image.open(args.data / split / "images" / name) as source:
                source.load()
                for r in group:
                    index, apple_id = int(r["region_index"]), r["apple_id"]
                    shape = lookup[name][index]["shape_attributes"]
                    if shape["name"] != "polygon":
                        raise ValueError("The experiment requires a visible polygon")
                    box = method.region_box(shape)
                    crop = source.crop(box)
                    vis = Image.new("L", crop.size)
                    points = zip(shape["all_points_x"], shape["all_points_y"])
                    ImageDraw.Draw(vis).polygon([(x - box[0], y - box[1]) for x, y in points], fill=255)
                    clean_score = method.score_crop(crop)
                    if abs(clean_score - float(r["score"])) > 1e-10:
                        raise ValueError(f"Clean-score mismatch in {name}/{apple_id}")
                    natural_row = natural[(name, apple_id)]
                    for target in LEVELS:
                        changed, actual = apply_leaf(crop, vis, target, f"{split}/{name}/{apple_id}", args.seed)
                        score = clean_score if target == 0 else method.score_crop(changed)
                        if split == "train":
                            train_augmented.append((score, r["label"]))
                            if target == 0:
                                train_clean.append((score, r["label"]))
                        else:
                            test.append({
                                "filename": name, "region_index": index, "apple_id": apple_id,
                                "label": r["label"], "target_fraction": target, "actual_fraction": actual,
                                "natural_occlusion": float(natural_row["occlusion_fraction"]),
                                "eligible_natural": int(natural_row["eligible"]),
                                "clean_score": clean_score, "score": score,
                            })
    thresholds = {"clean_fit": method.fit_threshold(train_clean),
                  "augmented_fit": method.fit_threshold(train_augmented)}
    for r in test:
        for model, threshold in thresholds.items():
            r[model + "_prediction"] = "ripe" if r["score"] >= threshold else "unripe"
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "ripeness_paired.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(test[0]))
        writer.writeheader()
        writer.writerows(test)
    low = [r for r in test if r["eligible_natural"] and r["natural_occlusion"] < 0.15]
    summary = {"seed": args.seed, "thresholds": thresholds, "train_clean_instances": len(train_clean),
               "train_augmented_views": len(train_augmented),
               "all_test": {model: summarize(test, th) for model, th in thresholds.items()},
               "interior_natural_under_15pct": {model: summarize(low, th) for model, th in thresholds.items()}}
    (args.out / "ripeness_stress_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for model, group in summary["all_test"].items():
        print(model, [(level, round(s["balanced_accuracy"], 4), s["correct_by_label"], s["changed_prediction"])
                      for level, s in group.items()])


if __name__ == "__main__":
    main()
