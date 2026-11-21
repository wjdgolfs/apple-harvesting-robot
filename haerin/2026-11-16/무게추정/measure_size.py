"""Measure apple diameter in millimetres from IntegrALIMENTA lab photos, using the 1-euro coin as a scale.

Only top-down ('ver') views photographed with one camera for all 60 apples are used, so the camera cannot
stand in for the weight batch (see 무게검증). For each photo:
  coin  : Hough circle search outside the apple, radius limited to a plausible range, scored by edge support
  apple : warm-skin colour mask (earlier experiment) -> largest blob -> holes filled
  size  : apple diameter px / coin diameter px * 23.25 mm
Nothing here uses the weight. A review sheet per view is written for visual checking.
"""

import argparse
import csv
import importlib.util
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
COIN_MM = 23.25  # 1 euro coin
WORK = 1000      # long side in pixels for measurement


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = load("weight_baseline", BASE / "weight/weight_baseline.py")


def load_frame(path):
    with Image.open(path) as im:
        im.draft("RGB", (WORK, WORK))
        rgb = ImageOps.exif_transpose(im).convert("RGB")
    rgb.thumbnail((WORK, WORK), Image.Resampling.LANCZOS)
    return rgb


def apple_mask(rgb):
    """Warm-skin mask over the whole frame, largest blob, holes filled, small gaps closed."""
    r, g, b = [np.asarray(c, dtype=np.int16) for c in rgb.split()]
    warm = ((r - b) >= 30) & ((g - r) <= 9)
    warm = ndimage.binary_opening(warm, iterations=2)
    labels, n = ndimage.label(warm)
    if n == 0:
        return None
    blob = labels == (np.argmax(np.bincount(labels.ravel())[1:]) + 1)
    blob = ndimage.binary_closing(blob, iterations=6)
    return ndimage.binary_fill_holes(blob)


def find_coin(rgb, apple):
    gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.5)
    h, w = gray.shape
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=120, param2=28,
                               minRadius=int(0.008 * max(h, w)), maxRadius=int(0.07 * max(h, w)))
    if circles is None:
        return None
    edges = cv2.Canny(gray, 60, 150)
    near_apple = ndimage.binary_dilation(apple, iterations=8) if apple is not None else np.zeros_like(gray, bool)
    best = None
    for x, y, rad in circles[0]:
        if not (rad < x < w - rad and rad < y < h - rad):
            continue
        if near_apple[int(y), int(x)]:
            continue
        theta = np.linspace(0, 2 * np.pi, 90, endpoint=False)
        support = 0
        for dr in (-1, 0, 1):
            xs = np.clip((x + (rad + dr) * np.cos(theta)).astype(int), 0, w - 1)
            ys = np.clip((y + (rad + dr) * np.sin(theta)).astype(int), 0, h - 1)
            support = max(support, (edges[ys, xs] > 0).mean())
        # bimetallic coin: the disk is not uniform and not too dark or bright compared with its ring
        if best is None or support > best[3]:
            best = (float(x), float(y), float(rad), float(support))
    return best


def focal_px(path, frame_w, frame_h):
    """Focal length in pixels of the working frame from the EXIF 35 mm-equivalent focal length."""
    with Image.open(path) as im:
        f35 = im.getexif().get_ifd(0x8769).get(41989)
    if not f35:
        return None
    return float(f35) * math.hypot(frame_w, frame_h) / math.hypot(36, 24)


def measure(path):
    rgb = load_frame(path)
    apple = apple_mask(rgb)
    out = dict(frame_w=rgb.width, frame_h=rgb.height, focal_px=focal_px(path, rgb.width, rgb.height))
    if apple is None:
        return out, rgb, None, None
    ys, xs = np.nonzero(apple)
    out.update(apple_px=int(apple.sum()), apple_w_px=int(np.ptp(xs) + 1), apple_h_px=int(np.ptp(ys) + 1),
               apple_touches_edge=int(xs.min() == 0 or ys.min() == 0 or xs.max() == rgb.width - 1 or ys.max() == rgb.height - 1),
               apple_eq_px=float(2 * math.sqrt(apple.sum() / math.pi)))
    coin = find_coin(rgb, apple)
    if coin:
        out.update(coin_x=coin[0], coin_y=coin[1], coin_r_px=coin[2], coin_support=coin[3])
        scale = COIN_MM / (2 * coin[2])
        out.update(d_eq_mm=out["apple_eq_px"] * scale, d_max_mm=max(out["apple_w_px"], out["apple_h_px"]) * scale)
    return out, rgb, apple, coin


def review_tile(rgb, apple, coin, label, size=220):
    view = rgb.copy()
    if apple is not None:
        edge = apple ^ ndimage.binary_erosion(apple, iterations=3)
        arr = np.asarray(view).copy()
        arr[edge] = (22, 222, 182)
        view = Image.fromarray(arr)
    draw = ImageDraw.Draw(view)
    if coin:
        x, y, r, _ = coin
        draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 40, 40), width=4)
    view.thumbnail((size, size * 4 // 3))
    tile = Image.new("RGB", (size, size * 4 // 3 + 16), "white")
    tile.paste(view, (0, 0))
    ImageDraw.Draw(tile).text((2, size * 4 // 3 + 1), label, fill="black")
    return tile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("D:/Apple-harvest-data/weight"))
    parser.add_argument("--views", nargs="+", required=True, help="e.g. spb_ext_me_ver")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--review", type=Path, default=HERE / "review")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.review.mkdir(parents=True, exist_ok=True)
    features = list(csv.DictReader((BASE / "weight/results/weight_features.csv").open(encoding="utf-8-sig")))
    apples = sorted({(r["group"], float(r["weight_g"]), r["camera"]) for r in features}, key=lambda a: (a[2], a[1]))
    rows = []
    for view in args.views:
        tiles = []
        for group, weight, batch in apples:
            path = next(p for p in (args.root / group).iterdir() if p.stem.lower().endswith(view))
            m, rgb, apple, coin = measure(path)
            rows.append(dict(view=view, group=group, weight_batch=batch, weight_g=weight, path=path.relative_to(args.root).as_posix(), **m))
            d = m.get("d_eq_mm")
            tiles.append(review_tile(rgb, apple, coin, f"{weight:.0f} g  {d:.1f} mm" if d else f"{weight:.0f} g  no coin"))
        sheet = Image.new("RGB", (220 * 10, tiles[0].height * 6), "white")
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % 10) * 220, (i // 10) * t.height))
        sheet.save(args.review / f"{view}.jpg", quality=85)
        print(f"{view}: done", flush=True)
    fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with (args.out / "size_measurements.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
