"""Evaluate the existing integrated demo on deterministic moving synthetic leaves."""

import argparse
import csv
import hashlib
import importlib.util
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from moving_leaf import generate_frames

DEMO = None  # the integrated demo module, set in main()


NAMES = ("_MG_2669_18.png", "_MG_2724_16.png", "_MG_2756_08.png",
         "_MG_2765_14.png", "_MG_2800_19.png")
KO = {"ripe": "익음", "unripe": "미숙", "hold": "보류",
      "stationary": "잎 고정", "weak": "작은 이동", "strong": "큰 이동"}


def load_module(path):
    spec = importlib.util.spec_from_file_location("integrated_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measure_mask(mask_image, depth, offset, source_shape, focal, settings, erosion, touches_edge):
    """Binary equivalent of demo.diameter_of, retaining holes and disconnected fragments."""
    return DEMO.measure_region(np.asarray(mask_image) > 0, offset[0], offset[1], depth, source_shape, focal, settings)


def _unused_old_rule(mask_image, depth, offset, source_shape, focal, settings, erosion, touches_edge):
    mask = np.asarray(mask_image) > 0
    ys, xs = np.nonzero(mask)
    if not xs.size:
        return dict(diameter_mm=None, diameter_estimate_mm=None, fill_ratio=0.0,
                    depth_valid_fraction=0.0, hold_reason="보이는 영역 없음")
    side = max(int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1))
    fill = float(mask.sum() / (np.pi/4 * side**2))
    inside = depth[mask]
    fraction = float((inside > 0).mean())
    core = depth[ndimage.binary_erosion(mask, iterations=erosion)]
    core = core[core > 0]
    left, top = offset
    edge = touches_edge(left+xs.min(), top+ys.min(), left+xs.max(), top+ys.max(), source_shape)
    reason = ""
    if edge:
        reason = "화면 경계"
    elif fraction < settings["min_depth_valid"] or not core.size:
        reason = "깊이 부족"
    elif fill < settings["fill_threshold"]:
        reason = "가림 의심"
    estimate = float(1000 * side * np.median(core) / focal) * settings["diameter_scale"] if core.size else None
    return dict(diameter_mm=estimate if not reason else None, diameter_estimate_mm=estimate,
                fill_ratio=fill, depth_valid_fraction=fraction, hold_reason=reason)


def visual(frames, row_lookup, case, destination):
    title, body, small = (ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", n) for n in (28,21,16))
    views = []
    for index in range(12):
        canvas = Image.new("RGB", (1080,500), "#fafaf5")
        draw = ImageDraw.Draw(canvas)
        draw.text((22,16), f"같은 사과에서 잎 위치를 바꾼 합성 실험 · ID {case[1]}", font=title, fill="#183d2c")
        draw.text((878,24),"라벨 익음",font=small,fill="#465348")
        for col, condition in enumerate(("stationary","weak","strong")):
            frame = next(f for f in frames if f["condition"] == condition and f["index"] == index)
            row = row_lookup[condition,index]
            image = frame["rgb"].copy()
            ratio = 290/max(image.size)
            image = image.resize((round(image.width*ratio),round(image.height*ratio)),Image.Resampling.LANCZOS)
            x = 22+col*356
            draw.text((x,62),KO[condition],font=body,fill="#183d2c")
            canvas.paste(image,(x+(310-image.width)//2,98+(290-image.height)//2))
            draw.text((x,395),f"추가 가림 {row['coverage']:.1%} · {KO[row['ripeness']]}",font=body,fill="#183d2c")
            weight = f"예상 {row['weight_g']:.0f} g" if row["weight_g"] is not None else "무게 보류"
            if row["weight_formula_extrapolated"]:
                weight += "*"
            draw.text((x,426),weight,font=body,fill="#465348")
        draw.text((22,470),"Gené-Mola 외, AmodalAppleSize_RGB-D (CC BY 4.0) · 영역 확대와 잎 합성 · 이동 폭은 시연 설정",
                  font=small,fill="#536158")
        views.append(canvas)
    views[0].save(destination/"잎_움직임_비교.gif",save_all=True,append_images=views[1:],duration=180,loop=0)
    views[0].save(destination/"잎_움직임_첫화면.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,required=True,help="Parent containing 통합데모, 익음도보강, 지름추정, 무게추정")
    parser.add_argument("--data",type=Path,default=Path("D:/Apple-harvest-data"))
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    demo_path = args.project_root/"통합데모/demo.py"
    demo = load_module(demo_path)
    global DEMO
    DEMO = demo
    settings = json.loads((args.project_root/"통합데모/results/_MG_2765_14_결과.json").read_bytes())["settings"]
    scorer = demo.Ripeness(settings["ripeness_model"],args.device,args.project_root/"전이학습/weights/resnet18-f37072fd.pth",
                           args.project_root/"전이학습/ripeness/results/head_aug.pt")
    if settings["ripeness_ripe_threshold"] != scorer.t_ripe or settings["ripeness_unripe_threshold"] != scorer.t_unripe:
        raise ValueError("Saved demo thresholds differ from the current model")
    all_rows, summaries, previews = [], [], {}
    with ZipFile(args.data/"02-annotated_data_fuji.zip") as archive:
        focal_table = demo.measure.table(archive,"focal_length.txt")
        for name in NAMES:
            focal = focal_table[name]
            depth = np.load(io.BytesIO(archive.read(f"{demo.ZIP_ROOT}/depth_maps/{Path(name).stem}.npy")))
            with Image.open(demo.find_image(args.data,name)) as source:
                rgb = source.convert("RGB")
            for detection in demo.dataset_detections(archive,name):
                fruit = detection["id"]
                if scorer.splits.get(fruit) != {"test"}:
                    continue
                original = demo.diameter_of(detection["polygon"],depth,focal,settings)
                if original["diameter_mm"] is None:
                    continue
                poly = detection["polygon"]
                xs, ys = zip(*poly)
                margin = round(.3*max(max(xs)-min(xs),max(ys)-min(ys)))
                bounds = max(0,min(xs)-margin),max(0,min(ys)-margin),min(rgb.width,max(xs)+1+margin),min(rgb.height,max(ys)+1+margin)
                left,top,right,bottom = bounds
                crop = rgb.crop(bounds)
                mask = Image.new("L",crop.size)
                ImageDraw.Draw(mask).polygon([(x-left,y-top) for x,y in poly],fill=255)
                cropped_depth = depth[top:bottom,left:right]
                check = measure_mask(mask,cropped_depth,(left,top),depth.shape,focal,settings,demo.measure.EROSION_PX,demo.touches_edge)
                if abs(check["diameter_mm"]-original["diameter_mm"]) > 1e-8:
                    raise ValueError("Binary mask geometry differs from original demo")
                original_logit = scorer.region_scores([(crop,np.asarray(mask) > 0)])[0]
                original_weight = settings["weight_a"]*original["diameter_mm"]**settings["weight_b"]
                frames = list(generate_frames(crop,mask,f"{name}/{fruit}",frames=12,target_fraction=.15))
                scores = scorer.region_scores([(f["rgb"],np.asarray(f["visible_mask"]) > 0) for f in frames])
                case_rows = []
                for frame,score in zip(frames,scores):
                    updated_depth = cropped_depth.copy()
                    updated_depth[np.asarray(frame["leaf_mask"]) > 0] = 0
                    size = measure_mask(frame["visible_mask"],updated_depth,(left,top),depth.shape,focal,settings,demo.measure.EROSION_PX,demo.touches_edge)
                    weight = settings["weight_a"]*size["diameter_mm"]**settings["weight_b"] if size["diameter_mm"] is not None else None
                    low,high = settings["weight_lab_range_mm"]
                    extrapolated = size["diameter_mm"] is not None and not low <= size["diameter_mm"] <= high
                    row = dict(filename=name,apple_id=fruit,condition=frame["condition"],frame=frame["index"],
                               dx_px=frame["offset_x_px"],dy_px=frame["offset_y_px"],coverage=frame["coverage"],
                               ripeness_logit=score,ripeness=scorer.decide(score),original_ripeness=scorer.decide(original_logit),
                               original_diameter_mm=original["diameter_mm"],original_weight_g=original_weight,
                               **size,weight_g=weight,
                               weight_formula_extrapolated=extrapolated,
                               weight_change_g=weight-original_weight if weight is not None else None)
                    case_rows.append(row)
                all_rows.extend(case_rows)
                if (name,fruit)==("_MG_2765_14.png","0214"):
                    previews[(name,fruit)]=(frames,case_rows)
                for condition in ("stationary","weak","strong"):
                    subset = [r for r in case_rows if r["condition"]==condition]
                    diameters = [r["diameter_mm"] for r in subset if r["diameter_mm"] is not None]
                    weights = [r["weight_g"] for r in subset if r["weight_g"] is not None]
                    transitions = sum(a["ripeness"]!=b["ripeness"] for a,b in zip(subset,subset[1:]))
                    summaries.append(dict(filename=name,apple_id=fruit,condition=condition,frames=len(subset),
                                          coverage_min=min(r["coverage"] for r in subset),coverage_max=max(r["coverage"] for r in subset),
                                          ripeness_transitions=transitions,
                                          ripeness_changed_from_original=sum(r["ripeness"]!=r["original_ripeness"] for r in subset),
                                          ripeness_hold=sum(r["ripeness"]=="hold" for r in subset),
                                          geometry_hold=len(subset)-len(diameters),
                                          geometry_hold_transitions=sum((a["diameter_mm"] is None)!=(b["diameter_mm"] is None) for a,b in zip(subset,subset[1:])),
                                          diameter_range_mm=max(diameters)-min(diameters) if diameters else None,
                                          weight_range_g=max(weights)-min(weights) if weights else None))
                print(name,fruit,"evaluated 36 moving-leaf frames",flush=True)
    if not all_rows:
        raise ValueError("No held-out test instances with usable original diameter")
    write_csv(args.out/"frames.csv",all_rows)
    write_csv(args.out/"sequences.csv",summaries)
    totals = {}
    for condition in ("stationary","weak","strong"):
        seq = [r for r in summaries if r["condition"]==condition]
        fr = [r for r in all_rows if r["condition"]==condition]
        totals[condition]=dict(apples=len(seq),frames=len(fr),
                              ripeness_transitions=sum(r["ripeness_transitions"] for r in seq),
                              ripeness_hold_frames=sum(r["ripeness"]=="hold" for r in fr),
                              geometry_hold_frames=sum(r["diameter_mm"] is None for r in fr),
                              diameter_range_mm_median=float(np.median([r["diameter_range_mm"] for r in seq if r["diameter_range_mm"] is not None])) if any(r["diameter_range_mm"] is not None for r in seq) else None,
                              weight_range_g_median=float(np.median([r["weight_range_g"] for r in seq if r["weight_range_g"] is not None])) if any(r["weight_range_g"] is not None for r in seq) else None,
                              weight_range_g_max=max((r["weight_range_g"] for r in seq if r["weight_range_g"] is not None), default=None),
                              confidence_changes=sum(a["confidence"]!=b["confidence"] for a,b in zip(fr,fr[1:]) if (a["filename"],a["apple_id"])==(b["filename"],b["apple_id"])),
                              geometry_hold_transitions=sum(r["geometry_hold_transitions"] for r in seq))
    report = dict(date="2026-11-23",case_count=len(summaries)//3,unique_apple_ids=len({r["apple_id"] for r in all_rows}),
                  selection="All held-out-test IDs with usable original geometry in five pre-existing demo images",
                  conditions=totals,settings=settings,
                  method="One initial 15% added leaf, fixed/5%/20% fruit-width displacement; updated RGB, visible binary mask, and depth under leaf; GT visible masks simulate ideal segmentation",
                  numeric_settings="cuDNN deterministic, benchmark false, TF32 false",
                  weight_interpretation="Changes in diameter-based expected grams; orchard ground-truth grams unavailable",
                  source_files={str(p.relative_to(args.project_root)):hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (demo_path,args.project_root/"익음도보강/weights/ripeness_finetuned.pt",
                                          args.project_root/"익음도보강/weights/ripeness_color_boosting.pkl",
                                          args.project_root/"익음도보강/results/model_selection.json",
                                          args.project_root/"익음도보강/results/apple_splits.csv",
                                          args.project_root/"지름추정/measure_instances.py",
                                          args.project_root/"지름추정/weights/shape_model.pkl",
                                          args.project_root/"지름추정/shape_features.py",
                                          args.project_root/"지름추정/results/metrics.json",
                                          args.project_root/"무게추정/results/metrics.json")},
                  decision_counts=dict(Counter(r["ripeness"] for r in all_rows)))
    (args.out/"metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    for case,(frames,case_rows) in previews.items():
        visual(frames,{(r["condition"],r["frame"]):r for r in case_rows},case,args.out)
    print(json.dumps(report["conditions"],ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
