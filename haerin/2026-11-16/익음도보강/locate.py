"""Where a labelled apple sits in its original photo: polygons, crops and small locator thumbnails for case sheets."""

import csv
import json
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent.parent
DATA = Path("D:/Apple-harvest-data/ripeness")
MARK = "#00c8ff"


@lru_cache(maxsize=None)
def split_table():
    with (BASE / "integration/results/linked_instances.csv").open(encoding="utf-8-sig", newline="") as stream:
        return {(r["filename"], str(r["region_index"])): r["split"] for r in csv.DictReader(stream)}


@lru_cache(maxsize=None)
def regions(split):
    meta = json.loads((DATA / split / f"{split}_via_final.json").read_bytes())
    return {r["filename"]: r["regions"] for r in meta.get("_via_img_metadata", meta).values()}


@lru_cache(maxsize=64)
def photo(split, filename):
    with Image.open(DATA / split / "images" / filename) as source:
        return source.convert("RGB")


def region(filename, region_index):
    """Return (photo, inclusive-exclusive crop box, visible polygon) of one labelled apple."""
    split = split_table()[filename, str(region_index)]
    shape = regions(split)[filename][int(region_index)]["shape_attributes"]
    xs, ys = shape["all_points_x"], shape["all_points_y"]
    return photo(split, filename), (min(xs), min(ys), max(xs) + 1, max(ys) + 1), list(zip(xs, ys))


def locator(image, box, size):
    """The whole photo without its black padding, with the apple's box marked."""
    inner = image.getbbox() or (0, 0, *image.size)
    view = image.crop(inner)
    scale = size / max(view.size)
    view = view.resize((round(view.width * scale), round(view.height * scale)), Image.Resampling.LANCZOS)
    left, top, right, bottom = ((v - o) * scale for v, o in zip(box, (inner[0], inner[1], inner[0], inner[1])))
    draw = ImageDraw.Draw(view)
    draw.rectangle((left, top, max(right, left + 3), max(bottom, top + 3)), outline=MARK, width=max(2, size // 55))
    return view
