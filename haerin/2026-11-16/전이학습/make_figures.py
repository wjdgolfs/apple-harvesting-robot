"""Render actual paired ripeness photos and the held-out experiment comparison."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    root = args.results_root
    data = json.loads((root / "ripeness/results/metrics.json").read_bytes())
    predictions = read_csv(root / "ripeness/results/predictions.csv")
    old_module = args.baseline_root / "occlusion/synthetic_leaf.py"
    spec = importlib.util.spec_from_file_location("leaf", old_module)
    leaf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(leaf)
    name, fruit = "_MG_2765_14.png", "0214"
    rows = sorted((r for r in predictions if r["filename"] == name and r["apple_id"] == fruit),
                  key=lambda r: int(r["cover_percent"]))
    via = json.loads((args.data / "ripeness/test/test_via_final.json").read_bytes())
    photo = next(r for r in via.values() if r["filename"] == name)
    shape = photo["regions"][int(rows[0]["region_index"])]["shape_attributes"]
    xx, yy = shape["all_points_x"], shape["all_points_y"]
    box = min(xx), min(yy), max(xx)+1, max(yy)+1
    with Image.open(args.data / "ripeness/test/images" / name) as im:
        crop = im.crop(box).convert("RGB")
    mask = Image.new("L", crop.size)
    ImageDraw.Draw(mask).polygon([(x-box[0], y-box[1]) for x,y in zip(xx,yy)], fill=255)
    canvas = Image.new("RGB", (1560, 790), "#f9f9f5")
    draw = ImageDraw.Draw(canvas)
    fontfile = "C:/Windows/Fonts/malgun.ttf"
    title, body, small = (ImageFont.truetype(fontfile, size) for size in (38, 24, 18))
    draw.text((40, 24), "같은 사과의 가림 실험 · 색 기준과 ResNet18", font=title, fill="#183c2e")
    draw.text((40, 85), "정답은 익음 · ResNet18은 ImageNet 특징을 고정하고 마지막 분류층을 학습함", font=body, fill="#354b40")
    ko = {"ripe": "익음", "unripe": "미숙"}
    for i, row in enumerate(rows):
        level = int(row["cover_percent"])
        image, actual = leaf.apply_leaf(crop, mask, level/100, f"test/{name}/{fruit}", 20261116)
        assert abs(actual-float(row["actual_fraction"])) < 1e-12
        x = 40 + 380*i
        draw.text((x, 138), f"추가 가림 {level}%", font=body, fill="#183c2e")
        ratio = 345 / max(image.size)
        image = image.resize((round(image.width*ratio), round(image.height*ratio)), Image.Resampling.LANCZOS)
        canvas.paste(image, (x+(345-image.width)//2, 184+(345-image.height)//2))
        for j, (label, key) in enumerate((("색 기준", "color_clean_prediction"),
                                         ("ResNet 원본학습", "clean_prediction"),
                                         ("ResNet 가림증강", "aug_prediction"))):
            draw.text((x, 550+40*j), f"{label}  {ko[row[key]]}", font=body,
                      fill="#1d624a" if row[key] == row["label"] else "#a34227")
    draw.text((40, 690), f"평가 사진 {name} · 사과 ID {fruit} · 원래 보이는 사과 픽셀을 기준으로 합성 잎을 추가함",
              font=small, fill="#354b40")
    draw.text((40, 727), "사진 Gené-Mola 외, AmodalAppleSize_RGB-D (CC BY 4.0) · 익음도 라벨 Zhu 외, Fuji-Ripeness-Size",
              font=small, fill="#53665c")
    draw.text((40, 754), "사진 확대 및 합성 잎 추가 · 표시한 예측은 저장된 평가 결과와 일치", font=small, fill="#53665c")
    canvas.save(args.out / "가림_익음도_전이학습.png")

    font_manager.fontManager.addfont(fontfile)
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 12})
    fig, axes = plt.subplots(1, 2, figsize=(12.5,4.8), layout="constrained")
    for model, label, color in (("color_clean","색 기준","#ad6c44"),("clean","ResNet 원본학습","#567b92"),
                                ("aug","ResNet 가림증강","#1c6a4b")):
        levels = [0,15,30,50]
        scores = [data["test"][model][str(k)] for k in levels]
        axes[0].plot(levels,[r["ripe_recall"]*100 for r in scores],"o-",label=label,color=color,lw=2)
        axes[1].plot(levels,[r["balanced_accuracy"]*100 for r in scores],"o-",label=label,color=color,lw=2)
    for ax in axes:
        ax.set_xticks(levels)
        ax.set_ylim(0,105)
        ax.set_xlabel("추가 합성 가림 (%)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=10,loc="lower left")
    axes[0].set_title("익은 사과를 찾은 비율 · 익음 20건")
    axes[1].set_title("두 클래스 재현율의 평균 · 전체 202건")
    fig.suptitle("학습과 겹치지 않는 사과 ID 115개의 같은 평가 영역 비교",fontsize=16)
    fig.savefig(args.out / "익음도_가림별_성능.png",dpi=160)
    plt.close(fig)
    weight = json.loads((root / "weight/results/weight_transfer_metrics.json").read_bytes())
    baseline_weight = json.loads((args.baseline_root / "weight/results/weight_metrics.json").read_bytes())
    fig, ax = plt.subplots(figsize=(9.5,5), layout="constrained")
    for label, levels, errors, color in (
        ("면적 기준", [0,15,30,50], [r["mae_g"] for r in baseline_weight["by_level"]["pooled_clean"]], "#ad6c44"),
        ("ResNet 원본학습", [0,15,30,50], [r["mae_g"] for r in weight["by_level"]["clean"]], "#567b92"),
        ("ResNet 가림증강", [0,15,30,50], [r["mae_g"] for r in weight["by_level"]["aug"]], "#1c6a4b")):
        ax.plot(levels, errors, "o-", label=label, color=color, lw=2)
        for x,y in zip(levels,errors):
            offset = -16 if label == "ResNet 가림증강" or (label == "면적 기준" and x == 50) else 10
            ax.annotate(f"{y:.2f}",(x,y),xytext=(0,offset),textcoords="offset points",ha="center",fontsize=10)
    ax.set_xticks([0,15,30,50])
    ax.set_ylim(0,60)
    ax.set_xlabel("추가 합성 가림 (%)")
    ax.set_ylabel("평균 절대오차 (g)")
    ax.set_title("동일한 실험실 후지 사과 12개에서 무게 오차 비교")
    ax.grid(alpha=.2)
    ax.legend(loc="upper left")
    fig.savefig(args.out / "무게_가림별_오차.png",dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
