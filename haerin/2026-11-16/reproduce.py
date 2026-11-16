"""Rebuild the week's audit tables and paired experiments from the three local datasets."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("D:/Apple-harvest-data"))
    parser.add_argument("--out", type=Path, required=True, help="Output folder for a new run")
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    data, out = args.data.resolve(), args.out.resolve()
    ripeness = data / "ripeness"
    orchard_zip = data / "02-annotated_data_fuji.zip"
    ri, oc, wt, join = (out / n / "results" for n in ("ripeness", "occlusion", "weight", "integration"))
    method = base / "ripeness/ripeness_baseline.py"

    def run(script, *arguments):
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, str(base / script), *map(str, arguments)], check=True)

    run("integration/link_annotations.py", "--ripeness-zip", data / "archive.zip",
        "--amodal-zip", orchard_zip, "--out", join)
    run("ripeness/ripeness_baseline.py", "--data", ripeness, "--out", ri)
    run("ripeness/fruit_id_check.py", "--predictions", ri / "predictions.csv",
        "--links", join / "linked_instances.csv", "--out", ri)
    run("occlusion/audit_occlusion.py", "--zip", orchard_zip, "--out", oc)
    run("occlusion/ripeness_stress.py", "--data", ripeness, "--cohort", ri / "linked_predictions.csv",
        "--method", method, "--natural", oc / "natural_occlusion.csv", "--out", oc)
    run("weight/weight_baseline.py", "--root", data / "weight",
        "--leaf-module", base / "occlusion/synthetic_leaf.py", "--out", wt)
    run("weight/verify_weight_jpegs.py", "--root", data / "weight", "--manifest", wt / "weight_manifest.csv",
        "--output", wt / "weight_image_audit.json")
    run("integration/check_cases.py", "--ripeness-root", ripeness, "--links", join / "linked_instances.csv",
        "--natural", oc / "natural_occlusion.csv", "--predictions", ri / "linked_predictions.csv", "--out", join)
    print(f"Results saved in {out}")


if __name__ == "__main__":
    main()
