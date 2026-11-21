"""Compare an exploratory trailing-three-frame median with raw ripeness decisions."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    info = json.loads((args.results/"metrics.json").read_bytes())
    lo = info["settings"]["ripeness_unripe_threshold"]
    hi = info["settings"]["ripeness_ripe_threshold"]
    with (args.results/"frames.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = defaultdict(list)
    for row in rows:
        groups[row["filename"],row["apple_id"],row["condition"]].append(row)
    summary = {}
    for (_,_,condition), sequence in groups.items():
        sequence.sort(key=lambda r:int(r["frame"]))
        history = []
        for row in sequence:
            history.append(float(row["ripeness_logit"]))
            value = statistics.median(history[-3:])
            row["median3_logit"] = value
            row["median3_ripeness"] = "ripe" if value >= hi else "unripe" if value < lo else "hold"
        totals = summary.setdefault(condition,dict(sequences=0,frames=0,raw_transitions=0,median3_transitions=0,
                                                   raw_hold_frames=0,median3_hold_frames=0))
        totals["sequences"] += 1
        totals["frames"] += len(sequence)
        totals["raw_hold_frames"] += sum(r["ripeness"]=="hold" for r in sequence)
        totals["median3_hold_frames"] += sum(r["median3_ripeness"]=="hold" for r in sequence)
        for a,b in zip(sequence,sequence[1:]):
            totals["raw_transitions"] += a["ripeness"] != b["ripeness"]
            totals["median3_transitions"] += a["median3_ripeness"] != b["median3_ripeness"]
    with (args.results/"median3_frames.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(method="Causal median of last up to three frame logits, current thresholds retained",
                  status="Exploratory analysis on these synthetic sequences; no independent video accuracy validation",
                  latency="Uses only current and up to two previous frames; can delay a genuine change",
                  conditions=summary)
    (args.results/"median3_metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
