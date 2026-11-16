"""Portable MinneApple baseline runner for a Colab GPU runtime."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER, YAML


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle(root):
    root = Path(root)
    manifest = json.loads((root / "bundle_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files_sha256"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"Invalid bundle member: {relative}")
        if sha256(path) != expected:
            raise ValueError(f"Checksum mismatch: {relative}")
    with (root / "splits/manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    counts = {}
    for split in ("train", "val"):
        selected = [r for r in rows if r["split"] == split]
        image_files = set((root / "images" / split).glob("*.png"))
        label_files = set((root / "labels" / split).glob("*.txt"))
        if len(image_files) != len(selected) or len(label_files) != len(selected):
            raise ValueError(f"Unexpected files in {split}")
        object_count = 0
        for row in selected:
            image = root / "images" / split / row["filename"]
            label = root / "labels" / split / (Path(row["filename"]).stem + ".txt")
            if sha256(image) != row["image_sha256"]:
                raise ValueError(f"Original image mismatch: {image.name}")
            lines = [s for s in label.read_text(encoding="utf-8").splitlines() if s.strip()]
            if len(lines) != int(row["instance_id_count"]):
                raise ValueError(f"Annotation count mismatch: {label.name}")
            object_count += len(lines)
        counts[split] = {"images": len(selected), "objects": object_count}
    if counts != {"train": {"images": 518, "objects": 22082},
                  "val": {"images": 152, "objects": 6101}}:
        raise ValueError(f"Unexpected fixed split: {counts}")
    print("BUNDLE_VERIFIED", json.dumps(counts), flush=True)
    return counts


def audit_masks(data, run):
    rows = []
    checks = {}
    for split in ("train", "val"):
        hyp = get_cfg(overrides={"task": "segment", "imgsz": 640,
                                 "mask_ratio": 1, "overlap_mask": True})
        ds = YOLODataset(img_path=str(Path(data["path"]) / data[split]),
                         data=dict(data, nc=1), task="segment", imgsz=640,
                         augment=False, cache=False, batch_size=1, hyp=hyp,
                         prefix=f"preflight {split}: ")
        for index in range(len(ds)):
            item = ds[index]
            count = len(item["cls"])
            ids = item["masks"].unique()
            present = int(((ids > 0) & (ids <= count)).sum())
            rows.append(dict(split=split, filename=Path(item["im_file"]).name,
                             target_objects=count, nonempty_masks=present,
                             empty_masks=count - present))
        selected = [r for r in rows if r["split"] == split]
        checks[split] = dict(images=len(ds),
                            objects=sum(r["target_objects"] for r in selected),
                            empty_masks=sum(r["empty_masks"] for r in selected))
        print("MASK_PREFLIGHT", split, checks[split], flush=True)
    with (run / "mask_preflight.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    expected = {"train": {"images": 518, "objects": 22082, "empty_masks": 0},
                "val": {"images": 152, "objects": 6101, "empty_masks": 0}}
    if checks != expected:
        raise ValueError(f"Preflight changed: {checks}")
    return checks


def main(root, drive_outputs):
    root = Path(root).resolve()
    drive_outputs = Path(drive_outputs)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable. Select a GPU runtime in Colab first.")
    if ultralytics.__version__ != "8.4.215":
        raise RuntimeError(f"Unexpected ultralytics version: {ultralytics.__version__}")
    counts = verify_bundle(root)
    name = "baseline_colab_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = root.parent / "runs" / name
    run.mkdir(parents=True, exist_ok=False)
    backup = drive_outputs / name
    backup.mkdir(parents=True, exist_ok=False)
    settings = json.loads((root / "configs/train_baseline.json").read_text(encoding="utf-8"))
    data = dict(path=str(root), train="images/train", val="images/val", names={0: "apple"})
    YAML.save(str(root / "dataset.yaml"), data)
    settings.update(data=str(root / "dataset.yaml"), project=str(run.parent),
                    name=run.name, exist_ok=True)
    started = time.monotonic()
    record = dict(status="preflight", started_utc=datetime.now(timezone.utc).isoformat(),
                  run=str(run), drive_backup=str(backup), settings=settings, split=counts,
                  python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                  ultralytics=ultralytics.__version__, gpu=torch.cuda.get_device_name(0),
                  gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                  start_weights="Original COCO pretrained yolov8n-seg.pt; fresh run, not resume",
                  note="Validation is for model selection, not an independent final test.")
    summary = run / "baseline_summary.json"
    backup_errors = []

    def save_record():
        record["elapsed_seconds"] = round(time.monotonic() - started, 2)
        summary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    def sync_to_drive():
        # Copy only the small run directory, never images or dataset caches.
        save_record()
        for source in run.rglob("*"):
            if not source.is_file():
                continue
            destination = backup / source.relative_to(run)
            if destination.exists() and destination.stat().st_size == source.stat().st_size and (
                    destination.stat().st_mtime_ns == source.stat().st_mtime_ns):
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".uploading")
            shutil.copy2(source, temporary)
            temporary.replace(destination)
        print("DRIVE_BACKUP", str(backup), flush=True)

    handler = logging.FileHandler(run / "run.log", encoding="utf-8")
    LOGGER.addHandler(handler)
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                capture_output=True, text=True, check=True)
        (run / "environment.txt").write_text(freeze.stdout, encoding="utf-8")
        record["mask_preflight"] = audit_masks(data, run)
        sync_to_drive()
        model = YOLO(str(root / "weights/yolov8n-seg.pt"))
        record["status"] = "training"
        sync_to_drive()

        def progress(trainer):
            record.update(epochs_completed=trainer.epoch + 1,
                          current_metrics={str(k): float(v) for k, v in trainer.metrics.items()
                                           if isinstance(v, (int, float, np.number))})
            handler.flush()
            try:
                sync_to_drive()
            except Exception as error:
                # The local VM checkpoint remains usable; the next epoch retries.
                backup_errors.append(f"{type(error).__name__}: {error}")
                record["backup_errors"] = list(backup_errors)
                save_record()
                print("DRIVE_BACKUP_FAILED", backup_errors[-1], flush=True)
            print("BASELINE_PROGRESS", json.dumps(
                {"epoch": trainer.epoch + 1, "elapsed_seconds": record["elapsed_seconds"],
                 "metrics": record["current_metrics"]}), flush=True)

        model.add_callback("on_fit_epoch_end", progress)
        model.train(**settings)
        with (run / "results.csv").open(encoding="utf-8-sig") as stream:
            results = [{k.strip(): v.strip() for k, v in r.items()}
                       for r in csv.DictReader(stream)]
        keys = [k for k in results[0] if k.startswith("train/") and "loss" in k]
        if not keys or not all(math.isfinite(float(r[k])) for r in results for k in keys):
            raise RuntimeError("Non-finite training loss")
        checkpoints = {}
        for filename in ("best.pt", "last.pt"):
            weight = run / "weights" / filename
            if YOLO(str(weight)).names != {0: "apple"}:
                raise ValueError("Checkpoint class mapping changed")
            checkpoints[filename] = sha256(weight)
        record.update(status="completed", epochs_completed=len(results),
                      training_losses_finite=True, checkpoint_sha256=checkpoints,
                      last_validation={k: v for k, v in results[-1].items()
                                       if k.startswith("metrics/")})
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error_type=type(error).__name__, error=str(error))
        (run / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        handler.flush()
        try:
            sync_to_drive()
        finally:
            LOGGER.removeHandler(handler)
            handler.close()
    print("BASELINE_COMPLETE", str(backup), flush=True)
    return record

