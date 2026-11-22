"""Figures for the diameter experiment: a worked example on real test apples and the error charts."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = "02-annotated_data_fuji"
SEED = 20261116
SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#a9a8a2"
SERIES = {"vis_width": ("보이는 가로 폭 (발표 9쪽 식 그대로)", "#2a78d6"),
          "vis_max": ("보이는 영역의 긴 변", "#eb6834"),
          "vis_area_eq": ("보이는 면적을 원으로 환산", "#1baf7a")}
FONT = "C:/Windows/Fonts/malgun.ttf"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def setup():
    font_manager.fontManager.addfont(FONT)
    font_manager.fontManager.addfont("C:/Windows/Fonts/malgunbd.ttf")
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 11})


def quiet(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def polygons(archive):
    out = {}
    for kind in ("modal", "amodal"):
        for row in json.loads(archive.read(f"{ROOT}/gt_json/test/via_region_data_{kind}.json")).values():
            for region in row["regions"].values():
                s = region["shape_attributes"]
                out.setdefault((row["filename"], region["region_attributes"]["apple_ID"]), {})[kind] = \
                    list(zip(s["all_points_x"], s["all_points_y"]))
    return out


def worked_example(zip_path, results, metrics, out):
    rows = read_csv(results / "test_predictions.csv")
    threshold = metrics["hold_rule"]["chosen_threshold"]
    picks = []
    # Neutral pick: hash order among large apples in a low and a high natural-occlusion band.
    for lo, hi in ((0.0, 0.10), (0.50, 0.65)):
        pool = [r for r in rows if lo <= float(r["natural_occlusion"]) < hi and float(r["amodal_width_px"]) >= 150]
        picks.append(min(pool, key=lambda r: hashlib.sha256(f"{SEED}|{r['filename']}|{r['apple_id']}".encode()).digest()))
    title, body, small = (ImageFont.truetype(FONT, s) for s in (32, 19, 15))
    canvas = Image.new("RGB", (1640, 600), SURFACE)
    draw = ImageDraw.Draw(canvas)
    draw.text((36, 22), "지름(근사) = 보이는 폭(px) × 깊이(m) ÷ 초점거리(px) · 평가 사진의 실제 계산", font=title, fill=INK)
    draw.text((36, 72), "청록선 = 사진에서 보이는 사과 윤곽, 노란선 = 실측 지름으로 만든 전체 윤곽(정답 확인용)", font=body, fill=INK2)
    with ZipFile(zip_path) as archive:
        polys = polygons(archive)
        for i, r in enumerate(picks):
            poly = polys[(r["filename"], r["apple_id"])]
            image = Image.open(io.BytesIO(archive.read(f"{ROOT}/images/test/{r['filename']}"))).convert("RGB")
            xs = [x for x, _ in poly["amodal"]]
            ys = [y for _, y in poly["amodal"]]
            side = round(1.7 * max(max(xs) - min(xs), max(ys) - min(ys)))
            left, top = (min(xs) + max(xs) - side) // 2, (min(ys) + max(ys) - side) // 2
            # Stay inside the 1024 x 1024 photo at pixels 137..1160 of the padded PNG.
            left = max(137, min(left, 1161 - side))
            top = max(137, min(top, 1161 - side))
            crop = image.crop((left, top, left + side, top + side))
            pen = ImageDraw.Draw(crop)
            for points, color in ((poly["amodal"], "#ffc439"), (poly["modal"], "#36eff7")):
                local = [(x - left, y - top) for x, y in points]
                pen.line(local + local[:1], fill=color, width=max(2, side // 110))
            x0 = 36 + 800 * i
            canvas.paste(crop.resize((340, 340), Image.Resampling.LANCZOS), (x0, 118))
            w, h, z, f = (float(r[k]) for k in ("vis_width_px", "vis_height_px", "depth_m", "focal_px"))
            kept = r["kept"] == "1"
            lines = [(f"자연 가림 {float(r['natural_occlusion']):.0%} · 사과 {r['apple_id']}", body, INK),
                     (f"보이는 가로 폭 {w:.0f} px, 세로 {h:.0f} px", body, INK),
                     (f"깊이(가운데 중앙값) {z:.2f} m", body, INK),
                     (f"초점거리 {f:,.0f} px", body, INK),
                     (f"가로 폭: {w:.0f} × {z:.2f} ÷ {f:,.0f} = {float(r['vis_width_mm']):.1f} mm", body, INK),
                     (f"긴 변:  {max(w, h):.0f} px → {float(r['vis_max_mm']):.1f} mm", body, INK),
                     (f"실측 지름 {float(r['gt_diameter_mm']):.1f} mm", body, INK),
                     (f"채움 비율 {float(r['fill_ratio']):.2f} → " + ("사용" if kept else f"판단 보류 (< {threshold:.2f})"),
                      body, INK if kept else "#b8431a")]
            for j, (text, font, color) in enumerate(lines):
                draw.text((x0 + 360, 124 + 41 * j), text, font=font, fill=color)
                right = draw.textbbox((x0 + 360, 0), text, font=font)[2]
                if right > x0 + 780:
                    raise ValueError(f"Example text overflows its panel: {text}")
    draw.text((36, 482), "채움 비율 = 보이는 면적 ÷ (긴 변을 지름으로 하는 원의 면적). 잎에 가리면 작아진다.", font=body, fill=INK2)
    draw.text((36, 514), "두 사과는 평가 분할에서 가림 0~10%, 50~65% 구간의 큰 사과 중 고정 해시 순서로 골랐다(오차를 보고 고르지 않음).",
              font=body, fill=INK2)
    draw.text((36, 566), "이미지·윤곽·깊이·실측 지름  Gené-Mola et al., AmodalAppleSize_RGB-D (CC BY 4.0) · 크롭, 윤곽 표시, 계산값 추가",
              font=small, fill=INK2)
    canvas.save(out / "계산예시.png", optimize=True)


def charts(results, metrics, out):
    bins = metrics["test"]["by_natural_occlusion"]
    labels = ["0~15%", "15~30%", "30~50%", "50% 이상"]
    keys = list(bins)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
    quiet(ax)
    xs = range(len(keys))
    for name, (label, color) in SERIES.items():
        ys = [bins[k][name]["mae_mm"] for k in keys]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(10, 0), textcoords="offset points", va="center", fontsize=9.5, color=INK)
    ref = [bins[k]["amodal_width"]["mae_mm"] for k in keys]
    ax.plot(xs, ref, color=INK2, linewidth=1.5, linestyle=(0, (4, 3)), marker="o", markersize=6)
    ax.annotate("전체 윤곽 폭 (정답에서 만든 참고값)", (xs[-1], ref[-1]), xytext=(10, 0), textcoords="offset points",
                va="center", fontsize=9.5, color=INK2)
    ax.set_xticks(list(xs), [f"{lab}\n{bins[k]['vis_max']['n']}장" for lab, k in zip(labels, keys)])
    ax.set_xlim(-0.3, 5.2)
    ax.set_xlabel("자연 가림 (전체 윤곽 중 가려진 비율) · 평가 분할 사과 사진 수")
    ax.set_ylabel("지름 평균 절대오차 (mm)")
    ax.set_title("가림이 커질수록 보이는 폭은 지름을 작게 잰다", loc="left", fontsize=12.5, pad=10)

    quiet(bx)
    trade = read_csv(results / "hold_tradeoff.csv")
    chosen = metrics["hold_rule"]["chosen_threshold"]
    for split, label, color in (("train_val", "학습+검증 (기준을 정한 자료)", "#2a78d6"), ("test", "평가", "#eb6834")):
        pts = [(float(r["kept_pct"]), float(r["mae_kept_mm"]), float(r["threshold"])) for r in trade if r["split"] == split]
        bx.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        pick = next(p for p in pts if abs(p[2] - chosen) < 1e-9)
        if split == "test":
            bx.annotate(f"문턱 {chosen:.2f}: {pick[0]:.0f}% 사용, {pick[1]:.1f} mm", pick[:2], xytext=(14, -26),
                        textcoords="offset points", fontsize=9.5, color=INK,
                        arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
    bx.axhline(metrics["hold_rule"]["target_mae_mm"], color=INK2, linewidth=1.1, linestyle=(0, (3, 3)))
    bx.text(98, metrics["hold_rule"]["target_mae_mm"] + 0.15, f"목표 {metrics['hold_rule']['target_mae_mm']:.0f} mm (팀 가정)",
            ha="right", va="bottom", fontsize=9, color=INK2)
    bx.set_xlim(0, 100)
    bx.set_xlabel("판단하지 않고 사용한 사진 비율 (%) · 채움 비율 문턱 0.50~0.90")
    bx.set_ylabel("사용한 사진의 평균 절대오차 (mm)")
    bx.set_title("판단 보류를 늘리면 남은 사과의 오차가 줄어든다", loc="left", fontsize=12.5, pad=10)
    bx.legend(frameon=False, fontsize=9.5, loc="upper left")
    fig.suptitle("RGB-D 지름 추정: 가림이 클수록 오차가 커진다", x=0.02, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.02, 0.885, "AmodalAppleSize 후지 평가 분할, 전체 윤곽이 사진 안에 들어온 관측 {:,}건.".format(sum(v['n'] for v in (b['vis_max'] for b in metrics['test']['by_natural_occlusion'].values()))) + " 깊이는 보이는 영역 가운데의 중앙값.",
             fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(out / "가림별_지름오차.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path("D:/Apple-harvest-data/02-annotated_data_fuji.zip"))
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--out", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    setup()
    metrics = json.loads((args.results / "metrics.json").read_text(encoding="utf-8"))
    worked_example(args.zip, args.results, metrics, args.out)
    charts(args.results, metrics, args.out)
    print(f"Figures written to {args.out}")


if __name__ == "__main__":
    main()
