"""Reproduce both transfer-learning tasks and the paired synthetic-leaf evaluation."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("D:/Apple-harvest-data"))
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path(__file__).resolve().parent / "weights/resnet18-f37072fd.pth")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    cache, out = args.cache.resolve(), args.out.resolve()
    ripe, weight = out / "ripeness/results", out / "weight/results"
    cache.mkdir(parents=True, exist_ok=True)

    def run(script, *arguments):
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, str(base / script), *map(str, arguments)], check=True)

    run("ripeness/prepare_ripeness.py", "--data", args.data / "ripeness",
        "--baseline-root", args.baseline_root, "--cache", cache)
    run("ripeness/encode_ripeness.py", "--cache", cache, "--weights", args.weights, "--device", args.device)
    run("ripeness/fit_ripeness.py", "--cache", cache, "--out", ripe)
    run("weight/train_weight.py", "--data-root", args.data / "weight", "--baseline-root", args.baseline_root,
        "--weights", args.weights, "--device", args.device, "--out", weight)
    run("make_figures.py", "--baseline-root", args.baseline_root, "--data", args.data,
        "--results-root", out, "--out", out / "figures")
    print(f"Finished. Results in {out}")


if __name__ == "__main__":
    main()
