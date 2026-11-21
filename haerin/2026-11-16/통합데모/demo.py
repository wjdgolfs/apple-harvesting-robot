"""One orchard image in, one result screen out: ripeness, diameter and hold reasons per apple.

Input masks come from detection JSON files (the hand-off format for the YOLO segmentation side), one per image
named <image stem>.json. Without them, the demo uses the dataset's own visible-apple polygons as stand-in detections.

    <stem>.json = {"image": "<file name>", "instances": [{"id": "<str>", "polygon": [[x, y], ...], "score": <float>}]}

Coordinates are pixels of the original PNG (1300 x 1300 including the dataset's 100 px black border).
Ripeness uses the fine-tuned ResNet18 of 익음도보강 with a three-way decision (ripe / unripe / hold), or the
earlier frozen-feature head with --ripeness-model linear. Diameter uses the pinhole rule and hold settings
chosen in 지름추정; weight uses the diameter formula of 무게추정.
"""

import argparse
import csv
import importlib.util
import io
import json
import sys
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import torch
from torch import nn
from torchvision.models import resnet18

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE / "전이학습"))
from imagenet_encoder import build_encoder, encode_batch  # noqa: E402

ZIP_ROOT = "02-annotated_data_fuji"
PADDING = 100
FONT = "C:/Windows/Fonts/malgun.ttf"
RIPE, UNRIPE, HOLD = "#e34948", "#0ca30c", "#8a8984"
KO = {"ripe": "익음", "unripe": "미숙", "hold": "보류"}
COLOR = {"ripe": RIPE, "unripe": UNRIPE, "hold": HOLD}


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = module_from_file("prepare_ripeness", BASE / "전이학습/ripeness/prepare_ripeness.py")
measure = module_from_file("measure_instances", BASE / "지름추정/measure_instances.py")


def find_image(data_root, name):
    for split in ("test", "val", "train"):
        path = data_root / "ripeness" / split / "images" / name
        if path.exists():
            return path
    raise FileNotFoundError(f"{name} not found under {data_root / 'ripeness'}")


class Ripeness:
    """Ripeness scorer with its decision thresholds and the train/val/test split of the apples it learned from."""

    def __init__(self, kind, device, weights, head_path):
        self.device = device
        splits = {}
        if kind == "finetuned":
            meta = json.loads((BASE / "익음도보강/results/metrics.json").read_text(encoding="utf-8"))
            th = meta["models"][meta["kept_finetune"]]["thresholds"]
            self.t_ripe, self.t_unripe = th["ripe"], th["unripe"]
            checkpoint = torch.load(BASE / "익음도보강/weights/ripeness_finetuned.pt", map_location="cpu", weights_only=True)
            model = resnet18(weights=None)
            model.fc = nn.Linear(512, 1)
            model.load_state_dict(checkpoint["state_dict"])
            self.model, self.head = model.to(device).eval(), None
            split_file = BASE / "익음도보강/results/apple_splits.csv"
            self.label = (f"익음도: ResNet18 미세조정, 익음 ≥ {self.t_ripe:.2f} · 미숙 < {self.t_unripe:.2f} · 사이는 보류")
        else:
            self.head = torch.load(head_path, map_location="cpu", weights_only=True)
            self.t_ripe = self.t_unripe = float(self.head["threshold"])
            self.model = build_encoder(weights, device)
            split_file = BASE / "ripeness/results/linked_predictions.csv"
            self.label = f"익음도: ImageNet ResNet18 고정 + 가림증강 분류층, 문턱 로짓 {self.t_ripe:.2f}"
        with split_file.open(encoding="utf-8-sig", newline="") as stream:
            for r in csv.DictReader(stream):
                splits.setdefault(r["apple_id"], set()).add(r["split"])
        self.splits = splits
        for n in (1, 4, 8, 12, 16):  # warm-up for typical batch sizes, excluded from timing
            encode_batch(self.model, [Image.new("RGB", (160, 160))] * n, device)

    def scores(self, crops):
        out = encode_batch(self.model, crops, self.device).double()
        if self.head is None:
            return out[:, 0].tolist()
        return (((out - self.head["mean"]) / self.head["scale"]).mv(self.head["weight"]) + self.head["bias"]).tolist()

    def decide(self, score):
        return "ripe" if score >= self.t_ripe else "unripe" if score < self.t_unripe else "hold"


def truth_tables(archive, name):
    """Calliper diameters and linked ripeness labels, only for the comparison lines."""
    diameters = measure.table(archive, "GT_diameter.txt")
    with (BASE / "integration/results/linked_instances.csv").open(encoding="utf-8-sig", newline="") as stream:
        labels = {r["apple_id"]: r["label"] for r in csv.DictReader(stream) if r["filename"] == name}
    return diameters, labels


def dataset_detections(archive, name):
    for split in ("test", "val", "train"):
        for row in json.loads(archive.read(f"{ZIP_ROOT}/gt_json/{split}/via_region_data_modal.json")).values():
            if row["filename"] == name:
                return [dict(id=r["region_attributes"]["apple_ID"], score=None,
                             polygon=[list(p) for p in zip(r["shape_attributes"]["all_points_x"],
                                                           r["shape_attributes"]["all_points_y"])])
                        for r in row["regions"].values()]
    raise KeyError(f"No visible-apple polygons for {name}")


def diameter_of(polygon, depth, focal, settings):
    """Same measurement and hold rule as 지름추정, from one visible mask."""
    mask, (left, top) = measure.rasterize([tuple(p) for p in polygon], depth.shape)
    ys, xs = np.nonzero(mask)
    width, height = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    side = max(width, height)
    fill = mask.sum() / (np.pi / 4 * side ** 2)
    z = depth[top:top + mask.shape[0], left:left + mask.shape[1]]
    inside = z[mask]
    valid_fraction = float((inside > 0).mean())
    core = z[ndimage.binary_erosion(mask, iterations=measure.EROSION_PX)]
    core = core[core > 0]
    touches = (left + xs.min() <= PADDING or top + ys.min() <= PADDING or
               left + xs.max() >= depth.shape[1] - 1 - PADDING or top + ys.max() >= depth.shape[0] - 1 - PADDING)
    out = dict(longer_side_px=int(side), fill_ratio=float(fill), depth_valid_fraction=valid_fraction,
               depth_m=float(np.median(core)) if core.size else None, diameter_mm=None, hold_reason="")
    if touches:
        out["hold_reason"] = "화면 경계"
    elif valid_fraction < settings["min_depth_valid"] or not core.size:
        out["hold_reason"] = "깊이 부족"
    elif fill < settings["fill_threshold"]:
        out["hold_reason"] = f"가림 의심 (채움 {fill:.2f})"
    if out["depth_m"] is not None:
        out["diameter_estimate_mm"] = float(1000 * side * out["depth_m"] / focal)
        if not out["hold_reason"]:
            out["diameter_mm"] = out["diameter_estimate_mm"]
    return out


def draw_screen(rgb, results, settings, timing, path):
    """Photo with coloured outlines and numbered tags, plus a per-apple table on the right."""
    box = Image.eval(rgb, lambda v: 255 if v > 0 else 0).convert("L").getbbox() or (0, 0, *rgb.size)
    photo = rgb.crop(box)
    scale = 860 / max(photo.size)
    photo = photo.resize((round(photo.width * scale), round(photo.height * scale)), Image.Resampling.LANCZOS)
    title, body, small, tag_font = (ImageFont.truetype(FONT, s) for s in (28, 18, 15, 17))
    canvas = Image.new("RGB", (photo.width + 640, 84 + max(photo.height, 860) + 52), "#fcfcfb")
    canvas.paste(photo, (24, 84))
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 22), "사과별 결과 화면 · 익음도 · 지름 · 무게 · 판단 보류", font=title, fill="#0b0b0b")
    draw.text((photo.width + 56, 52), "윤곽 색 = 익음도 예측 (빨강 익음, 초록 미숙, 회색 보류)", font=small, fill="#52514e")
    for r in results:
        pts = [((x - box[0]) * scale + 24, (y - box[1]) * scale + 84) for x, y in r["polygon"]]
        color = COLOR[r["ripeness"]]
        draw.line(pts + pts[:1], fill="white", width=5)
        draw.line(pts + pts[:1], fill=color, width=3)
        cx, cy = min(p[0] for p in pts), min(p[1] for p in pts)
        tag = str(r["number"])
        tw = draw.textbbox((0, 0), tag, font=tag_font)[2] + 12
        draw.rounded_rectangle((cx - 4, cy - 26, cx - 4 + tw, cy - 2), radius=6, fill=color)
        draw.text((cx + 2, cy - 25), tag, font=tag_font, fill="white")
    x0, y = photo.width + 56, 90
    head = ("번호", "익음도", "지름", "무게", "판단")
    cols = (x0, x0 + 56, x0 + 140, x0 + 225, x0 + 315)
    for c, h in zip(cols, head):
        draw.text((c, y), h, font=body, fill="#52514e")
    y += 34
    for r in results:
        color = COLOR[r["ripeness"]]
        draw.ellipse((cols[0], y + 5, cols[0] + 14, y + 19), fill=color)
        draw.text((cols[0] + 22, y), str(r["number"]), font=body, fill="#0b0b0b")
        draw.text((cols[1], y), KO[r["ripeness"]], font=body, fill="#0b0b0b")
        draw.text((cols[2], y), f"{r['diameter_mm']:.0f} mm" if r["diameter_mm"] else "—", font=body, fill="#0b0b0b")
        weight = f"{r['weight_g']:.0f} g" + ("*" if r["weight_note"] else "") if r["weight_g"] else "—"
        draw.text((cols[3], y), weight, font=body, fill="#0b0b0b")
        draw.text((cols[4], y), r["hold_reason"] or "사용", font=body, fill=HOLD if r["hold_reason"] else "#0b0b0b")
        y += 32
        if r.get("truth_ripeness") or r.get("truth_diameter_mm"):
            truth = f"정답 {KO.get(r.get('truth_ripeness'), '라벨 없음')}"
            if r.get("truth_diameter_mm"):
                truth += f" · 실측 {r['truth_diameter_mm']:.1f} mm"
            seen = r.get("ripeness_split_of_apple", "")
            if "train" in seen:
                truth += " · 익음도 학습 사과"
            elif "val" in seen:
                truth += " · 익음도 검증 사과"
            draw.text((cols[1], y - 4), truth, font=small, fill="#52514e")
            y += 26
    y += 18
    notes = [settings["ripeness_label"],
             f"지름: 긴 변 × 깊이 ÷ 초점거리, 채움 비율 < {settings['fill_threshold']:.2f} 이면 보류",
             f"무게: {settings['weight_a']:.4f} × 지름^{settings['weight_b']:.2f} (실험실 후지 60개, 교차검증 오차 "
             f"{settings['weight_cv_mae_g']:.0f} g)",
             f"* 지름이 식을 맞춘 범위({settings['weight_lab_range_mm'][0]:.0f}~{settings['weight_lab_range_mm'][1]:.0f} mm) 밖",
             f"처리 시간 {timing['per_image_s']:.2f} s (모델 로드 제외, {timing['device']})",
             f"입력 마스크: {settings['mask_source']}"]
    for n in notes:
        draw.text((x0, y), n, font=small, fill="#52514e")
        y += 24
    draw.text((24, canvas.height - 34), "이미지·깊이  Gené-Mola et al., AmodalAppleSize_RGB-D (CC BY 4.0) · 크기 조정, 윤곽·결과 표시",
              font=small, fill="#52514e")
    canvas.save(path, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, nargs="+", help="PNG file name(s), e.g. _MG_2669_18.png")
    parser.add_argument("--data-root", type=Path, default=Path("D:/Apple-harvest-data"))
    parser.add_argument("--detections-dir", type=Path,
                        help="Folder of <image stem>.json detections; default uses the dataset's visible polygons")
    parser.add_argument("--weights", type=Path, default=BASE / "전이학습/weights/resnet18-f37072fd.pth")
    parser.add_argument("--head", type=Path, default=BASE / "전이학습/ripeness/results/head_aug.pt",
                        help="Frozen-feature head, used only with --ripeness-model linear")
    parser.add_argument("--ripeness-model", choices=("finetuned", "linear"), default="finetuned")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--no-truth", action="store_true", help="Leave out the ground-truth comparison columns")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)

    diameter_metrics = json.loads((BASE / "지름추정/results/metrics.json").read_text(encoding="utf-8"))
    chosen_depth = diameter_metrics["depth_choice_on_train_val_amodal_width"]["chosen"]
    if chosen_depth != "depth_eroded_median_m|centre=False":
        raise ValueError(f"Demo implements the eroded-median depth rule, metrics chose {chosen_depth}")
    ripeness = Ripeness(args.ripeness_model, args.device, args.weights, args.head)
    weight_metrics = json.loads((BASE / "무게추정/results/metrics.json").read_text(encoding="utf-8"))
    power = weight_metrics["formula_all_60"]["power"]
    settings = dict(fill_threshold=diameter_metrics["hold_rule"]["chosen_threshold"], min_depth_valid=0.5,
                    ripeness_model=args.ripeness_model, ripeness_label=ripeness.label,
                    ripeness_ripe_threshold=ripeness.t_ripe, ripeness_unripe_threshold=ripeness.t_unripe,
                    weight_a=power["a"], weight_b=power["b"],
                    weight_lab_range_mm=weight_metrics["measurement"]["diameter_range_mm"],
                    weight_cv_mae_g=weight_metrics["cross_validation"]["power_three_views"]["mae_g"],
                    mask_source="검출 JSON" if args.detections_dir else "데이터셋의 보이는 사과 윤곽(검출기 대신)")

    summary = []
    with ZipFile(args.data_root / "02-annotated_data_fuji.zip") as archive:
        focal_table = measure.table(archive, "focal_length.txt")
        for name in args.image:
            summary.append(run_one(args, archive, focal_table, name, ripeness, settings))
    times = [s["per_image_s"] for s in summary]
    report = dict(images=summary, mean_s=float(np.mean(times)), apples=int(sum(s["apples"] for s in summary)),
                  device=args.device, note="Model loading and file reading excluded; RGB+depth already in memory")
    (args.out / "처리시간.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


def run_one(args, archive, focal_table, name, ripeness, settings):
    focal = focal_table[name]
    depth = np.load(io.BytesIO(archive.read(f"{ZIP_ROOT}/depth_maps/{Path(name).stem}.npy")))
    if args.detections_dir:
        detections = json.loads((args.detections_dir / f"{Path(name).stem}.json").read_text(encoding="utf-8"))["instances"]
    else:
        detections = dataset_detections(archive, name)
    diameters, labels = truth_tables(archive, name)
    with Image.open(find_image(args.data_root, name)) as source:
        rgb = source.convert("RGB")
    if rgb.size[::-1] != depth.shape:
        raise ValueError("RGB and depth sizes differ")

    start = time.perf_counter()
    crops = []
    for det in detections:
        xs, ys = [p[0] for p in det["polygon"]], [p[1] for p in det["polygon"]]
        crop_box = (max(0, min(xs)), max(0, min(ys)), min(rgb.width, max(xs) + 1), min(rgb.height, max(ys) + 1))
        crops.append(prepare.fit_square(rgb.crop(crop_box)))
    logits = ripeness.scores(crops)
    results = []
    for det, logit in zip(detections, logits):
        size = diameter_of(det["polygon"], depth, focal, settings)
        weight, weight_note = None, ""
        if size["diameter_mm"]:
            weight = settings["weight_a"] * size["diameter_mm"] ** settings["weight_b"]
            low, high = settings["weight_lab_range_mm"]
            if not low <= size["diameter_mm"] <= high:
                weight_note = "식 범위 밖"
        results.append(dict(id=str(det["id"]), score=det.get("score"), polygon=det["polygon"],
                            ripeness=ripeness.decide(logit),
                            ripeness_logit=logit, **size, weight_g=weight, weight_note=weight_note))
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    results.sort(key=lambda r: (min(p[1] for p in r["polygon"]), min(p[0] for p in r["polygon"])))
    for i, r in enumerate(results, 1):
        r["number"] = i
        if not args.no_truth and not args.detections_dir:
            r["truth_ripeness"] = labels.get(r["id"])
            r["truth_diameter_mm"] = diameters.get(r["id"])
            r["ripeness_split_of_apple"] = ",".join(sorted(ripeness.splits.get(r["id"], ())))
    timing = dict(per_image_s=elapsed, apples=len(results), device=args.device)

    stem = Path(name).stem
    draw_screen(rgb, results, settings, timing, args.out / f"{stem}_결과화면.png")
    payload = dict(image=name, focal_px=focal, settings=settings, timing=timing,
                   apples=[{k: v for k, v in r.items() if k != "polygon"} for r in results])
    (args.out / f"{stem}_결과.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.detections_dir:
        hand_off = dict(image=name, instances=[dict(id=r["id"], score=None, polygon=r["polygon"]) for r in results])
        (args.out / f"{stem}_검출입력예시.json").write_text(json.dumps(hand_off), encoding="utf-8")
    return dict(image=name, **timing)


if __name__ == "__main__":
    main()
