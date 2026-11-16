"""Extract the frozen ImageNet ResNet18 penultimate features of the cached crops."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imagenet_encoder import WEIGHTS_SHA256, build_encoder, encode_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=40)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    inputs = json.loads((args.cache / "inputs.json").read_text(encoding="utf-8"))
    count, size = inputs["n_images"], inputs["size"]
    encoder = build_encoder(args.weights, args.device)
    features = torch.empty((count, 512), dtype=torch.float32)
    digest = hashlib.sha256()
    with (args.cache / "pixels.rgb").open("rb") as reader:
        for start in range(0, count, args.batch_size):
            stop = min(start + args.batch_size, count)
            pixels = bytearray(reader.read((stop - start) * size * size * 3))
            digest.update(pixels)
            tensor = torch.frombuffer(pixels, dtype=torch.uint8).view(stop - start, size, size, 3)
            features[start:stop].copy_(encode_batch(encoder, tensor, args.device))
            if start == 0 or stop % (args.batch_size * 20) == 0:
                print(f"Encoded {stop} / {count}", flush=True)
        if reader.read(1) or digest.hexdigest() != inputs["pixel_sha256"]:
            raise ValueError("Unexpected crop cache contents")
    metadata = dict(inputs, weights_sha256=WEIGHTS_SHA256, device=args.device, torch=str(torch.__version__),
                    gpu=torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None)
    torch.save({"features": features, "metadata": metadata}, args.cache / "features.pt")
    print(f"Encoded all {count} images using {args.device}", flush=True)


if __name__ == "__main__":
    main()
