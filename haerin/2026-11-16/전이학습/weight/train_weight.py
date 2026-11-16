"""Fit image-only weight regressors on frozen ImageNet ResNet18 features."""

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imagenet_encoder import build_encoder, encode_batch, WEIGHTS_SHA256


def load_frame(image_path):
    with Image.open(image_path) as im:
        im.draft("RGB", (384, 384))
        rgb = ImageOps.exif_transpose(im).convert("RGB")
    rgb.thumbnail((320, 320), Image.Resampling.LANCZOS)
    return rgb


def preprocess_image(rgb):
    return ImageOps.pad(rgb, (224, 224), method=Image.Resampling.LANCZOS, color=(124, 116, 104))


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fit_head(x, y):
    """Minimize mean squared error + 0.1 * sum(coef^2); leave bias unpenalized."""
    x_mean, y_mean = x.mean(axis=0), y.mean()
    centered = x - x_mean
    gram = centered @ centered.T
    gram.flat[::len(x) + 1] += 0.1 * len(x)
    coef = centered.T @ np.linalg.solve(gram, y - y_mean)
    return coef, y_mean - x_mean @ coef


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("D:/Apple-harvest-data/weight"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    baseline = module_from_file("area_method", args.baseline_root / "weight/weight_baseline.py")
    leaf = module_from_file("leaf_method", args.baseline_root / "occlusion/synthetic_leaf.py")
    encoder = build_encoder(args.weights, args.device)
    with (args.baseline_root / "weight/results/weight_features.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    counts = {s: sum(all(r["split"]==s for r in g) for g in grouped.values()) for s in ("train","test")}
    if counts != {"train":48, "test":12} or len(grouped) != 60:
        raise ValueError(f"Unexpected fruit split {counts}")
    images, embeds, ordered = [], [], []
    for group, group_rows in grouped.items():
        base = group_rows[0]
        if len({r["path"] for r in group_rows}) != 1 or sorted(float(r["target_fraction"]) for r in group_rows) != [0,.15,.30,.50]:
            raise ValueError(f"Invalid repeated-image conditions for {group}")
        rgb = load_frame(args.data_root/base["path"])
        mask = baseline.fruit_pixels(rgb)
        for r in group_rows:
            level = float(r["target_fraction"])
            frame, coverage = leaf.apply_leaf(rgb, mask, level, group, seed=baseline.SEED)
            if abs(coverage-float(r["measured_fraction"]))>1e-8:
                raise ValueError(f"Coverage mismatch for {group} at {level}: {coverage}")
            images.append(preprocess_image(frame))
            ordered.append(r)
            if len(images) == 16:
                embeds.append(encode_batch(encoder, images, args.device).numpy())
                images.clear()
    if images:
        embeds.append(encode_batch(encoder, images, args.device).numpy())
    x = np.concatenate(embeds).astype(np.float64)
    y = np.array([float(r["weight_g"]) for r in ordered])
    train = np.array([r["split"]=="train" for r in ordered])
    levels = np.array([float(r["target_fraction"]) for r in ordered])
    fitted, preds, metrics = {}, [], {}
    for name, train_ix in (("clean",train & (levels==0)),("aug",train)):
        coef,bias = fit_head(x[train_ix],y[train_ix])
        fitted.update({name+"_coef":coef,name+"_bias":np.asarray(bias)})
        est = x @ coef + bias
        clean_lookup = {ordered[i]["group"]:est[i] for i in np.flatnonzero(~train & (levels==0))}
        metrics[name] = []
        for level in (0,.15,.3,.5):
            ix = np.flatnonzero(~train & (levels==level))
            drift = [abs(est[i]-clean_lookup[ordered[i]["group"]]) for i in ix]
            metrics[name].append(dict(target_fraction=level, n=len(ix), mae_g=float(np.mean(abs(y[ix]-est[ix]))),
                                      mean_abs_change_g=float(np.mean(drift))))
            for i,d in zip(ix,drift):
                r=ordered[i]
                preds.append(dict(group=r["group"], path=r["path"], split=r["split"], target_fraction=level,
                                  measured_fraction=float(r["measured_fraction"]), weight_g=y[i], model=name,
                                  estimate_g=est[i],abs_error_g=abs(y[i]-est[i]),abs_change_from_clean_g=d))
    np.savez(args.out/"weight_transfer_heads.npz", **fitted)
    with (args.out/"weight_transfer_predictions.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(preds[0]))
        writer.writeheader()
        writer.writerows(preds)
    report=dict(model="TorchVision frozen ResNet18 IMAGENET1K_V1, 512 raw pooled features + ridge",
                weights_sha256=WEIGHTS_SHA256, train_fruit=counts["train"], test_fruit=counts["test"],
                input="Full oriented RGB, 320 maximum side then centered 224x224 letterbox, Lanczos, fill 124/116/104",
                train_views={"clean":48,"aug":192}, fixed_lambda=0.1,
                selection="Preset regularization, no validation or test-based model selection.",
                mask_seed=baseline.SEED,device=args.device,torch=torch.__version__,
                global_train_mean_g=float(y[train & (levels==0)].mean()),by_level=metrics)
    (args.out/"weight_transfer_metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(metrics,indent=2))


if __name__ == "__main__":
    main()
