"""Run pretrained YOLOv8 instance segmentation on one orchard image."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default=r'C:\26_2\컴퓨터비전\팀플\detection\train\images\20150919_174151_image1.png')
    parser.add_argument('--mask', default=None, help='Optional GT mask for visualization only; never passed to model')
    parser.add_argument('--device', default='auto', help='auto, cpu, or CUDA index such as 0')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--output', default=str(ROOT / 'outputs' / 'pretrained'))
    args = parser.parse_args()
    source = Path(args.image).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    device = ('0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use --device cpu or install compatible PyTorch.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = Path(args.output).resolve() / f'{source.stem}_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    weights = ROOT / 'weights' / 'yolov8n-seg.pt'
    weights.parent.mkdir(exist_ok=True)
    model = YOLO(str(weights))
    apple_ids = [i for i, name in model.names.items() if name == 'apple']
    if not apple_ids:
        raise RuntimeError('No apple class in the selected pretrained weights')
    result = model.predict(source=str(source), classes=apple_ids, imgsz=args.imgsz,
                           conf=args.conf, device=device, save=False, verbose=True)[0]
    prediction = result.plot()  # BGR uint8
    Image.fromarray(prediction[..., ::-1]).save(output / 'prediction.png')
    boxes = result.boxes
    rows = []
    for index in range(len(boxes)):
        xyxy = boxes.xyxy[index].cpu().tolist()
        rows.append({'prediction_index': index + 1, 'class_id': int(boxes.cls[index].item()),
                     'class_name': model.names[int(boxes.cls[index].item())],
                     'confidence': float(boxes.conf[index].item()),
                     **dict(zip(['xmin','ymin','xmax','ymax'], xyxy))})
    with (output / 'predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['prediction_index','class_id','class_name','confidence','xmin','ymin','xmax','ymax'])
        writer.writeheader(); writer.writerows(rows)
    polygons = [] if result.masks is None else [segment.tolist() for segment in result.masks.xy]
    (output / 'predicted_polygons.json').write_text(json.dumps(polygons), encoding='utf-8')
    # Read optional ground truth only AFTER inference, for independent display.
    gt_count = None
    with Image.open(source) as im:
        rgb = np.array(im.convert('RGB'))
    panels = [(rgb, 'Original RGB')]
    if args.mask:
        with Image.open(args.mask) as im:
            gt = np.array(im)
        if gt.ndim != 2 or gt.shape != rgb.shape[:2]:
            raise ValueError('Ground truth must be a single-channel mask aligned with RGB')
        gt_ids = np.unique(gt); gt_ids = gt_ids[gt_ids != 0]
        gt_count = len(gt_ids)
        colored = np.zeros_like(rgb)
        for instance_id in gt_ids:
            colored[gt == instance_id] = np.random.default_rng(int(instance_id)).integers(60,256,3)
        overlay = rgb.copy(); foreground = gt != 0
        overlay[foreground] = np.rint(0.5*rgb[foreground] + 0.5*colored[foreground]).astype(np.uint8)
        panels.append((overlay, f'Ground truth: {gt_count} IDs'))
    panels.append((prediction[..., ::-1], f'Pretrained prediction: {len(rows)} apples'))
    fig, axes = plt.subplots(1, len(panels), figsize=(5*len(panels),9), constrained_layout=True)
    for ax, (data,title) in zip(axes, panels):
        ax.imshow(data); ax.set_title(title); ax.axis('off')
    fig.savefig(output / 'comparison.png', dpi=180); plt.close(fig)
    summary = {
        'created_utc': stamp, 'source': str(source),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'weights': str(weights), 'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
        'model': 'yolov8n-seg', 'training': 'No MinneApple fine-tuning',
        'device_requested': args.device, 'device_selected': device,
        'cuda_available': torch.cuda.is_available(),
        'device_name': torch.cuda.get_device_name(int(device)) if device != 'cpu' else 'CPU',
        'python': sys.version, 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
        'ultralytics': ultralytics.__version__,
        'imgsz': args.imgsz, 'confidence_threshold': args.conf,
        'apple_class_ids': apple_ids, 'predicted_count': len(rows),
        'ground_truth_id_count': gt_count,
        'speed_ms_reported_by_library': result.speed,
        'note': 'Counts are not recall: no GT matching or accuracy evaluation performed. Single-image runtime includes no controlled benchmarking. GT IDs and prediction indices are unrelated.'
    }
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    freeze = subprocess.run([sys.executable,'-m','pip','freeze'], capture_output=True, text=True, check=True)
    (output / 'environment.txt').write_text(freeze.stdout, encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Results saved to: {output}')


if __name__ == '__main__':
    main()