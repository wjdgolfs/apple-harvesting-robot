"""Build the fixed-cohort ResNet inputs from the previous ripeness baseline."""

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

from PIL import Image, ImageDraw


SIZE = 160
LEVELS = (0, 15, 30, 50)
SEED = 20261116
MEAN_RGB = (124, 116, 104)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fit_square(image):
    width, height = image.size
    ratio = SIZE / max(width, height)
    fitted = image.resize((max(1, round(width * ratio)), max(1, round(height * ratio))),
                          Image.Resampling.BILINEAR)
    output = Image.new("RGB", (SIZE, SIZE), MEAN_RGB)
    output.paste(fitted, ((SIZE - fitted.width) // 2, (SIZE - fitted.height) // 2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    method = load_module("color_method", args.baseline_root / "ripeness/ripeness_baseline.py")
    leaf = load_module("leaf", args.baseline_root / "occlusion/synthetic_leaf.py")
    cohort_path = args.baseline_root / "ripeness/results/linked_predictions.csv"
    cohort = read_csv(cohort_path)
    baseline = {(r["filename"], r["apple_id"], int(100 * float(r["target_fraction"]))): r
                for r in read_csv(args.baseline_root / "occlusion/results/ripeness_paired.csv")}
    seen_ids, seen_frames = set(), set()
    index = []
    pixels_path = args.cache / "pixels.rgb"
    temp_path = args.cache / "pixels.rgb.tmp"
    file_hash = hashlib.sha256()
    with temp_path.open("wb") as pixels:
        for split in ("train", "val", "test"):
            split_rows = sorted((r for r in cohort if r["split"] == split), key=lambda r: r["filename"])
            fruit_ids = {r["apple_id"] for r in split_rows}
            frames = {r["group"] for r in split_rows}
            if seen_ids & fruit_ids or seen_frames & frames:
                raise ValueError("Fruit IDs or source frames overlap across splits")
            seen_ids |= fruit_ids
            seen_frames |= frames
            metadata = json.loads((args.data / split / (split + "_via_final.json")).read_bytes())
            shapes = {r["filename"]: r["regions"] for r in metadata.get("_via_img_metadata", metadata).values()}
            for name, instance_rows in itertools.groupby(split_rows, key=lambda r: r["filename"]):
                with Image.open(args.data / split / "images" / name) as photo:
                    photo.load()
                    for row in instance_rows:
                        apple_id = row["apple_id"]
                        idx = int(row["region_index"])
                        region = shapes[name][idx]
                        if region["region_attributes"]["Ripeness"] != row["label"]:
                            raise ValueError(f"Label mismatch in {name}/{idx}")
                        shape = region["shape_attributes"]
                        box = method.region_box(shape)
                        crop = photo.crop(box).convert("RGB")
                        if abs(method.score_crop(crop) - float(row["score"])) > 1e-10:
                            raise ValueError(f"Previous clean crop differs in {name}/{idx}")
                        visible = Image.new("L", crop.size)
                        points = zip(shape["all_points_x"], shape["all_points_y"])
                        ImageDraw.Draw(visible).polygon([(x - box[0], y - box[1]) for x, y in points], fill=255)
                        for level in (LEVELS if split != "val" else (0,)):
                            image, coverage = leaf.apply_leaf(crop, visible, level / 100,
                                                               f"{split}/{name}/{apple_id}", SEED)
                            record = {"split": split, "filename": name, "region_index": idx, "apple_id": apple_id,
                                      "source_frame": row["group"], "label": row["label"], "cover_percent": level,
                                      "actual_fraction": coverage, "color_clean_prediction": "",
                                      "color_aug_prediction": ""}
                            if split == "test":
                                previous = baseline[(name, apple_id, level)]
                                if (abs(float(previous["actual_fraction"]) - coverage) > 1e-12 or
                                        abs(float(previous["score"]) - method.score_crop(image)) > 1e-10):
                                    raise ValueError(f"Paired stress image differs in {name}/{idx}/{level}")
                                record["color_clean_prediction"] = previous["clean_fit_prediction"]
                                record["color_aug_prediction"] = previous["augmented_fit_prediction"]
                            data = fit_square(image).tobytes()
                            pixels.write(data)
                            file_hash.update(data)
                            index.append(record)
            print(f"{split}: {len(split_rows)} regions from {len(fruit_ids)} IDs", flush=True)
    temp_path.replace(pixels_path)
    with (args.cache / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)
    info = {"size": SIZE, "pixel_sha256": file_hash.hexdigest(), "n_images": len(index),
            "cohort_sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(), "leaf_seed": SEED,
            "levels_percent": LEVELS, "letterbox_color": MEAN_RGB,
            "preprocessing": "Tight RGB VIA bounding box, scaled longest side bilinear to 160 px; centered on mean-colored square"}
    (args.cache / "inputs.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
