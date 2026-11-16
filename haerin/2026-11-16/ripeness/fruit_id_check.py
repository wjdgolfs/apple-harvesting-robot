"""Re-fit and evaluate the color rule on linked, split-disjoint fruit and frame groups."""

import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

from ripeness_baseline import SPLITS, fit_threshold, metrics, stable_hash


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def row_key(row):
    return row["split"], row["filename"], row["region_index"]


def run(predictions, links, output):
    id_by_key = {row_key(r): (r["apple_id"], r["label"]) for r in read_csv(links)}
    by_fruit = defaultdict(list)
    prefilter_counts = Counter()
    for row in read_csv(predictions):
        link = id_by_key.get(row_key(row))
        if link is None:
            continue
        fruit, label = link
        if label != row["label"]:
            raise ValueError(f"Label mismatch for {row_key(row)}")
        row["score"] = float(row["score"])
        row["apple_id"] = fruit
        prefilter_counts[row["split"]] += 1
        by_fruit[fruit].append(row)

    experiment = {split: [] for split in SPLITS}
    for fruit, rows in by_fruit.items():
        available = [s for s in SPLITS if any(r["split"] == s for r in rows)]
        # A pre-set hash selects one of the original splits containing this ID.
        # Discard its observations in the other splits.
        split = available[int(stable_hash("apple:" + fruit)[:8], 16) % len(available)]
        experiment[split].extend(r for r in rows if r["split"] == split)
    train = experiment["train"]
    threshold = fit_threshold((r["score"], r["label"]) for r in train)
    majority = Counter(r["label"] for r in train).most_common(1)[0][0]
    summary = {"threshold": threshold, "training_majority_label": majority,
               "matched_before_id_filter": prefilter_counts, "pilot": {}, "overlap": {}}
    for split, rows in experiment.items():
        rows.sort(key=lambda r: (r["filename"], int(r["region_index"])))
        summary["pilot"][split] = metrics(rows, threshold, majority)
        summary["pilot"][split]["apple_ids"] = len({r["apple_id"] for r in rows})
    for a, b in itertools.combinations(SPLITS, 2):
        summary["overlap"][f"{a}-{b}"] = {}
        for field in ("group", "apple_id", "filename"):
            common = {r[field] for r in experiment[a]} & {r[field] for r in experiment[b]}
            if common:
                raise ValueError(f"Overlapping {field}: {common}")
            summary["overlap"][f"{a}-{b}"][field] = len(common)

    output.mkdir(parents=True, exist_ok=True)
    with (output / "linked_predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(train[0]))
        writer.writeheader()
        for split in SPLITS:
            writer.writerows(experiment[split])
    (output / "linked_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--links", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.predictions, args.links, args.out)
