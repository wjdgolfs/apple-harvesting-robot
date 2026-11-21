"""Show what the 'ripe' label means: example crops and the share of red skin pixels per labelled apple view.

The dataset authors labelled each apple in each photo as ripe if it 'appeared mostly red' (arXiv 2502.01850,
Sec. 2.1); no numeric threshold is given. Here 'red share' = visible apple pixels whose hue is red
(within 20 degrees of 0) with saturation >= 0.25, inside the VIA polygon. It is only an illustration of the rule.
"""

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SEED = 20261116
FONT = "C:/Windows/Fonts/malgun.ttf"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"


def red_share(crop, mask):
    hsv = np.asarray(crop.convert("HSV")).astype(float)
    hue = hsv[..., 0] * 360 / 255
    red = ((hue <= 20) | (hue >= 340)) & (hsv[..., 1] / 255 >= 0.25)
    inside = np.asarray(mask) > 0
    return float(red[inside].mean()) if inside.any() else float("nan")


def period(filename):
    n = int(re.match(r"_MG_(\d+)_", filename)[1])
    return "2018-10" if n < 5000 else "2020-07"


def h(text):
    return hashlib.sha256(f"{SEED}|{text}".encode()).digest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("D:/Apple-harvest-data/ripeness"))
    args = parser.parse_args()
    linked = list(csv.DictReader((BASE / "integration/results/linked_instances.csv").open(encoding="utf-8-sig")))
    via = {}
    for split in ("train", "val", "test"):
        meta = json.loads((args.data / split / f"{split}_via_final.json").read_bytes())
        for rec in meta.get("_via_img_metadata", meta).values():
            via[split, rec["filename"]] = rec["regions"]
    by_tile = defaultdict(list)
    for r in linked:
        by_tile[r["split"], r["filename"]].append(r)
    rows, crops = [], {}
    for i, ((split, name), items) in enumerate(sorted(by_tile.items())):
        with Image.open(args.data / split / "images" / name) as photo:
            photo = photo.convert("RGB")
        for r in items:
            shape = via[split, name][int(r["region_index"])]["shape_attributes"]
            xs, ys = shape["all_points_x"], shape["all_points_y"]
            box = (min(xs), min(ys), max(xs) + 1, max(ys) + 1)
            crop = photo.crop(box)
            mask = Image.new("L", crop.size)
            ImageDraw.Draw(mask).polygon([(x - box[0], y - box[1]) for x, y in zip(xs, ys)], fill=255)
            share = red_share(crop, mask)
            key = (name, r["region_index"])
            rows.append(dict(split=split, filename=name, region_index=r["region_index"], apple_id=r["apple_id"],
                             label=r["label"], period=period(name), red_share=share, box_px=max(crop.size)))
            if max(crop.size) >= 90:  # keep reasonably large crops for display
                crops[key] = crop
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(by_tile)} tiles", flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    with (HERE / "results/red_share.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # ---------- photo sheet
    shown = [r for r in rows if (r["filename"], r["region_index"]) in crops]
    pick = lambda pool, n: sorted(pool, key=lambda r: h(r["filename"] + r["region_index"]))[:n]
    ripe = pick([r for r in shown if r["label"] == "ripe"], 6)
    unripe_oct = pick([r for r in shown if r["label"] == "unripe" and r["period"] == "2018-10"], 6)
    unripe_jul = pick([r for r in shown if r["label"] == "unripe" and r["period"] == "2020-07"], 6)
    views = defaultdict(list)
    for r in shown:
        views[r["apple_id"]].append(r)
    pairs = []
    for apple in sorted((a for a, v in views.items() if {x["label"] for x in v} == {"ripe", "unripe"}), key=lambda a: h(a)):
        v = views[apple]
        pairs.append((pick([x for x in v if x["label"] == "ripe"], 1)[0], pick([x for x in v if x["label"] == "unripe"], 1)[0]))
        if len(pairs) == 3:
            break
    tile, gap = 170, 14
    title, body, small = (ImageFont.truetype(FONT, s) for s in (30, 19, 15))
    sections = [("① '익음' 라벨 (2018년 10월 3일 촬영)", ripe), ("② '미숙' 라벨, 같은 날 촬영", unripe_oct),
                ("③ '미숙' 라벨 (2020년 7월 16일 촬영)", unripe_jul)]
    width = 40 + 6 * (tile + gap)
    canvas = Image.new("RGB", (width, 150 + 4 * (tile + 80) + 60), SURFACE)
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 20), "'익음'은 사과 겉면이 대부분 빨갛게 보이면 붙인 라벨이다", font=title, fill=INK)
    draw.text((30, 66), "연구진이 사진마다 사과 하나씩 눈으로 판정했다(숫자 기준 없음). 아래 %는 보이는 면 중 빨간 픽셀 비율.",
              font=body, fill=INK2)
    y = 110
    for label, items in sections:
        draw.text((30, y), label, font=body, fill=INK)
        for j, r in enumerate(items):
            img = crops[(r["filename"], r["region_index"])].copy()
            img.thumbnail((tile, tile))
            x = 30 + j * (tile + gap)
            canvas.paste(img, (x + (tile - img.width) // 2, y + 30 + (tile - img.height) // 2))
            draw.text((x, y + 34 + tile), f"빨강 {r['red_share']:.0%}", font=small, fill=INK2)
        y += tile + 80
    draw.text((30, y), "④ 같은 사과(같은 ID)인데 사진에 따라 라벨이 다르다 · 왼쪽 '익음', 오른쪽 '미숙'", font=body, fill=INK)
    for j, (a, b) in enumerate(pairs):
        for k, r in enumerate((a, b)):
            img = crops[(r["filename"], r["region_index"])].copy()
            img.thumbnail((tile, tile))
            x = 30 + (2 * j + k) * (tile + gap)
            canvas.paste(img, (x + (tile - img.width) // 2, y + 30 + (tile - img.height) // 2))
            draw.text((x, y + 34 + tile), f"{'익음' if k == 0 else '미숙'} · 빨강 {r['red_share']:.0%}", font=small, fill=INK2)
            draw.text((x, y + 54 + tile), f"사과 {r['apple_id']}", font=small, fill=INK2)
    draw.text((30, canvas.height - 34), "이미지 Gené-Mola et al., AmodalAppleSize_RGB-D (CC BY 4.0) · 라벨 Zhu et al., Fuji-Ripeness-Size · 잘라서 확대",
              font=small, fill=INK2)
    (HERE / "figures").mkdir(exist_ok=True)
    canvas.save(HERE / "figures/익음_라벨_사진.png", optimize=True)

    # ---------- red share distribution
    font_manager.fontManager.addfont(FONT)
    font_manager.fontManager.addfont("C:/Windows/Fonts/malgunbd.ttf")
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "font.size": 11})
    fig, ax = plt.subplots(figsize=(10, 4.4))
    bins = np.linspace(0, 1, 41)
    for label, period_name, color, text in (("unripe", "2020-07", "#0ca30c", "미숙 · 2020년 7월"),
                                            ("unripe", "2018-10", "#a9a8a2", "미숙 · 2018년 10월"),
                                            ("ripe", "2018-10", "#e34948", "익음 · 2018년 10월")):
        v = [r["red_share"] for r in rows if r["label"] == label and r["period"] == period_name and not np.isnan(r["red_share"])]
        ax.hist(v, bins=bins, density=True, histtype="step", linewidth=2, color=color, label=f"{text} ({len(v):,}건)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("보이는 사과 면 중 빨간 픽셀 비율")
    ax.set_ylabel("비율별 밀도")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False)
    ax.set_title("'익음' 라벨은 빨간 면이 많은 사진에 붙어 있지만, 경계는 겹친다", loc="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "figures/빨강비율_라벨별.png", dpi=160)
    stats = {}
    for label in ("ripe", "unripe"):
        for p in ("2018-10", "2020-07"):
            v = np.array([r["red_share"] for r in rows if r["label"] == label and r["period"] == p and not np.isnan(r["red_share"])])
            if len(v):
                stats[f"{label}_{p}"] = dict(n=int(len(v)), median=float(np.median(v)), q25=float(np.percentile(v, 25)),
                                             q75=float(np.percentile(v, 75)))
    (HERE / "results/red_share_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
