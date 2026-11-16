"""Predict ripeness or grams with a saved frozen ResNet18 encoder and trained head."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from imagenet_encoder import build_encoder, encode_batch
from ripeness.prepare_ripeness import fit_square


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("ripeness", "weight"), required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--variant", choices=("clean", "aug"), default="aug",
                        help="Weight regression head stored in the NPZ file")
    parser.add_argument("--box", type=int, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                        help="Optional ripeness crop in pixel coordinates, right and bottom exclusive")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.task == "weight" and args.box:
        parser.error("Weight prediction uses the complete source frame")
    if args.task == "ripeness":
        head = torch.load(args.head, map_location="cpu", weights_only=True)
        with Image.open(args.image) as source:
            rgb = source.convert("RGB")
        if args.box:
            left, top, right, bottom = args.box
            if not (0 <= left < right <= rgb.width and 0 <= top < bottom <= rgb.height):
                parser.error("Crop coordinates must stay inside the source image")
            rgb = rgb.crop(tuple(args.box))
        image = fit_square(rgb)
    else:
        from weight.train_weight import load_frame, preprocess_image
        image = preprocess_image(load_frame(args.image))
    encoder = build_encoder(args.weights, args.device)
    feature = encode_batch(encoder, [image], args.device)
    if args.task == "ripeness":
        logits = ((feature.double() - head["mean"]) / head["scale"]).mv(head["weight"]) + head["bias"]
        value = logits.item()
        result = {"task": "ripeness", "prediction": "ripe" if value >= head["threshold"] else "unripe",
                  "logit": value, "decision_threshold": head["threshold"]}
    else:
        with np.load(args.head, allow_pickle=False) as params:
            estimate = feature.numpy().astype(np.float64)[0] @ params[args.variant + "_coef"]
            estimate += params[args.variant + "_bias"].item()
        result = {"task": "weight", "variant": args.variant, "estimate_g": float(estimate)}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
