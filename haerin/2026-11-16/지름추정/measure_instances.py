"""Measure every paired apple annotation in the Fuji AmodalAppleSize ZIP: mask widths and depth statistics.

Reads images' VIA polygons, depth maps (metres, aligned to the padded 1300x1300 PNG frame), the per-image focal
length (pixels) and the calliper diameter (mm) straight from the official ZIP. Nothing is estimated here;
evaluate_diameter.py turns these measurements into diameters.
"""

import argparse
import csv
import io
import json
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = "02-annotated_data_fuji"
EROSION_PX = 3
FIELDS = ("split", "filename", "apple_id", "gt_diameter_mm", "focal_px",
          "vis_px", "vis_width_px", "vis_height_px", "vis_eroded_px",
          "depth_valid_fraction", "depth_median_m", "depth_eroded_median_m", "depth_eroded_p25_m",
          "amodal_px", "amodal_width_px", "amodal_height_px",
          "natural_occlusion", "eligible")


def read_pairs(archive, split):
    out = {}
    for kind in ("modal", "amodal"):
        records = json.loads(archive.read(f"{ROOT}/gt_json/{split}/via_region_data_{kind}.json"))
        for row in records.values():
            for region in row["regions"].values():
                shape = region["shape_attributes"]
                key = row["filename"], region["region_attributes"]["apple_ID"]
                out.setdefault(key, {})[kind] = list(zip(shape["all_points_x"], shape["all_points_y"]))
    if any(len(v) != 2 for v in out.values()):
        raise ValueError(f"Unpaired modal/amodal polygon in {split}")
    return out


def table(archive, name):
    return {k: float(v) for k, v in (line.split(",") for line in archive.read(f"{ROOT}/{name}").decode().split())}


def rasterize(points, shape):
    """Render a polygon on its own bounding box; return (mask, top-left) clipped to the frame."""
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    left, top = max(0, int(np.floor(min(xs)))), max(0, int(np.floor(min(ys))))
    right, bottom = min(shape[1], int(np.ceil(max(xs))) + 1), min(shape[0], int(np.ceil(max(ys))) + 1)
    canvas = Image.new("L", (right - left, bottom - top))
    ImageDraw.Draw(canvas).polygon([(x - left, y - top) for x, y in points], fill=255)
    return np.asarray(canvas) > 0, (left, top)


def extent(mask):
    ys, xs = np.nonzero(mask)
    return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def measure(depth, visible, amodal):
    vis, (vl, vt) = rasterize(visible, depth.shape)
    if not vis.any():
        raise ValueError("Empty visible mask")
    z = depth[vt:vt + vis.shape[0], vl:vl + vis.shape[1]]
    inside = z[vis]
    valid = inside[inside > 0]
    eroded = ndimage.binary_erosion(vis, iterations=EROSION_PX)
    core = z[eroded]
    core = core[core > 0]
    full, _ = rasterize(amodal, depth.shape)
    vw, vh = extent(vis)
    aw, ah = extent(full) if full.any() else (0, 0)
    nan = float("nan")
    return dict(vis_px=int(vis.sum()), vis_width_px=vw, vis_height_px=vh, vis_eroded_px=int(eroded.sum()),
                depth_valid_fraction=valid.size / inside.size,
                depth_median_m=float(np.median(valid)) if valid.size else nan,
                depth_eroded_median_m=float(np.median(core)) if core.size else nan,
                depth_eroded_p25_m=float(np.percentile(core, 25)) if core.size else nan,
                amodal_px=int(full.sum()), amodal_width_px=aw, amodal_height_px=ah)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path("D:/Apple-harvest-data/02-annotated_data_fuji.zip"))
    parser.add_argument("--occlusion", type=Path,
                        default=Path(__file__).resolve().parents[1] / "occlusion/results/natural_occlusion.csv")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with args.occlusion.open(encoding="utf-8", newline="") as stream:
        occlusion = {(r["split"], r["filename"], r["apple_id"]): r for r in csv.DictReader(stream)}
    rows = []
    with ZipFile(args.zip) as archive:
        diameters = table(archive, "GT_diameter.txt")
        focal = table(archive, "focal_length.txt")
        for split in ("train", "val", "test"):
            pairs = read_pairs(archive, split)
            by_image = defaultdict(list)
            for (filename, apple_id), polys in pairs.items():
                by_image[filename].append((apple_id, polys))
            for i, (filename, items) in enumerate(sorted(by_image.items())):
                depth = np.load(io.BytesIO(archive.read(f"{ROOT}/depth_maps/{Path(filename).stem}.npy")))
                if depth.shape != (1300, 1300):
                    raise ValueError(f"Unexpected depth shape {depth.shape} for {filename}")
                for apple_id, polys in sorted(items):
                    link = occlusion[(split, filename, apple_id)]
                    rows.append(dict(split=split, filename=filename, apple_id=apple_id,
                                     gt_diameter_mm=diameters.get(apple_id, float("nan")), focal_px=focal[filename],
                                     **measure(depth, polys["modal"], polys["amodal"]),
                                     natural_occlusion=float(link["occlusion_fraction"]), eligible=int(link["eligible"])))
                if (i + 1) % 200 == 0:
                    print(f"{split}: {i + 1}/{len(by_image)} images", flush=True)
    if len(rows) != len(occlusion):
        raise ValueError(f"Measured {len(rows)} pairs but the occlusion audit has {len(occlusion)}")
    with (args.out / "instance_measurements.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} paired annotations")


if __name__ == "__main__":
    main()
