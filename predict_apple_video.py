"""Overlay YOLO apple instances on local video and export frame coordinates."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Local video file")
    parser.add_argument("--weights", type=Path, required=True, help="Apple segmentation checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="New directory for results")
    parser.add_argument("--seconds", type=float, help="Process at most this many seconds")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not args.video.is_file() or not args.weights.is_file():
        parser.error("Both --video and --weights must point to local files")
    if not 0 < args.conf <= 1:
        parser.error("--conf must be greater than 0 and at most 1")
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds must be finite and positive")
    return args


def apple_positions(result):
    boxes = result.boxes.xyxy.cpu().tolist()
    if not boxes:
        return []
    if result.masks is None:
        raise RuntimeError("Detected apples have no instance masks")
    polygons = result.masks.xy
    if len(polygons) != len(boxes):
        raise RuntimeError("Box and mask counts differ")
    positions = []
    for index, (box, score, polygon) in enumerate(zip(boxes, result.boxes.conf.cpu().tolist(), polygons), 1):
        x1, y1, x2, y2 = box
        positions.append({
            "detection": index,
            "confidence": round(score, 5),
            "box_xyxy": [round(value, 2) for value in box],
            "box_center_xy": [round((x1+x2)/2, 2), round((y1+y2)/2, 2)],
            "mask_polygon_xy": polygon.round(1).tolist(),
        })
    return positions


def make_overlay(result, time_s, count):
    frame = result.plot(conf=False, labels=False, line_width=2, color_mode="class")
    text = f"APPLE DETECTION   |   {time_s:5.2f} s   |   {count} predictions"
    scale = max(0.45, min(0.85, frame.shape[1] / 1400))
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), baseline = cv2.getTextSize(text, font, scale, 2)
    cv2.rectangle(frame, (10, 10), (width+30, height+baseline+27), (29, 36, 28), -1)
    cv2.putText(frame, text, (20, 20+height), font, scale, (240, 251, 238), 2, cv2.LINE_AA)
    return frame


def main():
    args = parse_args()
    model = YOLO(str(args.weights), task="segment")
    classes = [class_id for class_id, name in model.names.items() if name == "apple"]
    if len(classes) != 1:
        raise ValueError("The checkpoint must have exactly one apple class")

    capture = cv2.VideoCapture(str(args.video))
    writer = None
    try:
        if not capture.isOpened():
            raise OSError(f"Could not open {args.video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        ok, frame = capture.read()
        if not ok or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Expected a video with readable frames and a positive FPS")
        height, width = frame.shape[:2]
        # OpenCV's MP4 writer needs even dimensions. Keep coordinate origins unchanged.
        video_size = (width + width % 2, height + height % 2)
        limit = max(1, math.ceil(fps * args.seconds)) if args.seconds is not None else math.inf
        args.output.mkdir(parents=True, exist_ok=False)
        draft_video = args.output / "apple_predictions_partial.mp4"
        writer = cv2.VideoWriter(str(draft_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, video_size)
        if not writer.isOpened():
            raise OSError("Could not open an MP4 writer")

        counts = []
        with (args.output / "frame_counts.csv").open("w", newline="", encoding="utf-8") as table, (
            args.output / "apple_positions.jsonl"
        ).open("w", encoding="utf-8") as positions:
            count_writer = csv.writer(table)
            count_writer.writerow(["frame", "time_seconds", "predictions"])
            while ok and len(counts) < limit:
                index = len(counts)
                time_s = round(index / fps, 6)
                result = model.predict(
                    source=frame, classes=classes, device=args.device,
                    imgsz=640, conf=args.conf, iou=0.7, max_det=300,
                    retina_masks=True, verbose=False, save=False,
                )[0]
                detections = apple_positions(result)
                count = len(detections)
                counts.append(count)
                count_writer.writerow([index, time_s, count])
                positions.write(json.dumps(
                    {"frame": index, "time_seconds": time_s, "apples": detections},
                    separators=(",", ":"),
                ) + "\n")
                overlay = make_overlay(result, time_s, count)
                if video_size != (width, height):
                    overlay = cv2.copyMakeBorder(overlay, 0, height % 2, 0, width % 2, cv2.BORDER_REPLICATE)
                writer.write(overlay)
                if (index + 1) % 50 == 0:
                    print(f"{index + 1} frames processed", flush=True)
                ok, frame = capture.read()
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    draft_video.rename(args.output / "apple_predictions.mp4")
    metadata = {
        "source_file": args.video.name,
        "model": args.weights.name,
        "model_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "source_frame_size": [width, height],
        "fps": fps,
        "frames": len(counts),
        "duration_seconds": round(len(counts) / fps, 3),
        "predictions_per_frame": {"min": min(counts), "max": max(counts)},
        "settings": {"conf": args.conf, "imgsz": 640, "nms_iou": 0.7, "device": args.device},
        "coordinates": "Source pixels, origin at top-left. Frame and time start at zero.",
        "identity": "Detection numbers are local to each frame.",
    }
    (args.output / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"{len(counts)} frames saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
