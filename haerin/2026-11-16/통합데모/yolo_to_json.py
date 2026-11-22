"""Run a YOLOv8-seg model on the demo images and save the hand-off JSON that demo.py and compare_detections.py read.

    python yolo_to_json.py --weights best.pt --image _MG_2669_18.png ... --out detections

Polygons come from Ultralytics `Results.masks.xy` (pixel coordinates of the original image) with retina masks
on, so outlines are not limited to the network input resolution. `--classes 0` keeps the single apple class of
a MinneApple model; a COCO model needs `--classes 47` (apple).
Requires the `ultralytics` package. Written against the Ultralytics 8.x Results API; not yet run in this project.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def find_image(data_root, name):
    for split in ("test", "val", "train"):
        path = data_root / "ripeness" / split / "images" / name
        if path.exists():
            return path
    raise FileNotFoundError(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image", nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("D:/Apple-harvest-data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--classes", type=int, nargs="*", default=[0])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    from ultralytics import YOLO  # imported here so the other demo scripts do not need it

    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    timing = []
    for name in args.image:
        path = find_image(args.data_root, name)
        start = time.perf_counter()
        result = model.predict(source=str(path), imgsz=args.imgsz, conf=args.conf, classes=args.classes,
                               retina_masks=True, device=args.device, verbose=False)[0]
        timing.append(time.perf_counter() - start)
        instances = []
        if result.masks is not None:
            confs = result.boxes.conf.cpu().numpy()
            for k, (poly, conf) in enumerate(zip(result.masks.xy, confs)):
                if len(poly) < 3:
                    continue
                instances.append(dict(id=str(k), score=float(conf), polygon=np.round(poly).astype(int).tolist()))
        (args.out / f"{Path(name).stem}.json").write_text(json.dumps(dict(image=name, instances=instances)), encoding="utf-8")
        print(f"{name}: {len(instances)} apples, {timing[-1]:.3f} s")
    (args.out / "yolo_timing.json").write_text(json.dumps(dict(
        weights=str(args.weights), imgsz=args.imgsz, conf=args.conf, classes=args.classes,
        per_image_s=dict(zip(args.image, timing)),
        mean_s_excluding_first=float(np.mean(timing[1:])) if len(timing) > 1 else timing[0]), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
