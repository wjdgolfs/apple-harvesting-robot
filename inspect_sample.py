"""MinneApple: inspect an RGB image and its instance-ID mask."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', default=r'C:\26_2\컴퓨터비전\팀플\detection\train')
parser.add_argument('--name', default='20150919_174151_image1.png')
parser.add_argument('--output', default='outputs/sample_inspection')
args = parser.parse_args()
root = Path(args.root)
image_path = root / 'images' / args.name
mask_path = root / 'masks' / args.name
with Image.open(image_path) as im:
    rgb = np.array(im.convert('RGB'))
with Image.open(mask_path) as im:
    mask_mode = im.mode
    mask = np.array(im)
if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
    raise ValueError(f'Expected a single-channel integer mask: {mask.shape}, {mask.dtype}')
if rgb.shape[:2] != mask.shape:
    raise ValueError('Image and mask dimensions differ')
ids, counts = np.unique(mask, return_counts=True)
object_ids = ids[ids != 0]
colors = np.zeros_like(rgb)
rows = []
for object_id in object_ids:
    region = mask == object_id
    ys, xs = np.where(region)
    colors[region] = np.random.default_rng(int(object_id)).integers(60, 256, size=3)
    rows.append({'id': int(object_id), 'area_px': int(region.sum()),
                 'xmin': int(xs.min()), 'ymin': int(ys.min()),
                 'xmax_inclusive': int(xs.max()), 'ymax_inclusive': int(ys.max()),
                 'mean_x': float(xs.mean()), 'mean_y': float(ys.mean())})
foreground = mask != 0
blend = rgb.copy()
blend[foreground] = np.rint(0.5 * rgb[foreground] + 0.5 * colors[foreground]).astype(np.uint8)
out = Path(args.output) / Path(args.name).stem
out.mkdir(parents=True, exist_ok=True)
Image.fromarray(blend).save(out / 'overlay.png')
Image.fromarray(colors).save(out / 'instance_colors.png')
fig, axes = plt.subplots(1, 4, figsize=(16, 8), constrained_layout=True)
for ax, data, title in zip(axes, [rgb, mask, colors, blend],
                          ['Original RGB', 'Raw mask (0-255)', 'Instance colors', '50% overlay with IDs']):
    if data.ndim == 2:
        ax.imshow(data, cmap='gray', vmin=0, vmax=255)
    else:
        ax.imshow(data)
    ax.set_title(title)
    ax.axis('off')
for row in rows:
    axes[3].text(row['mean_x'], row['mean_y'], str(row['id']), fontsize=5,
                 color='white', ha='center', va='center',
                 bbox=dict(facecolor='black', alpha=0.5, pad=0.3, edgecolor='none'))
fig.savefig(out / 'comparison.png', dpi=180)
plt.close(fig)
with (out / 'instances.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['id','area_px','xmin','ymin','xmax_inclusive','ymax_inclusive','mean_x','mean_y'])
    writer.writeheader()
    writer.writerows(rows)
summary = {'image': str(image_path), 'mask': str(mask_path),
           'width': int(mask.shape[1]), 'height': int(mask.shape[0]),
           'mask_mode': mask_mode, 'mask_dtype': str(mask.dtype),
           'instance_id_count': len(rows),
           'background_pixels': int((mask == 0).sum()),
           'pixel_counts_by_id': {str(int(i)): int(c) for i,c in zip(ids,counts)},
           'note': '0 is treated as background. Nonzero IDs are annotation instances, not model detections or a manual count of every visible apple.'}
(out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'Image and mask: {mask.shape[1]} x {mask.shape[0]}')
print(f'Mask: {mask.dtype}; instance IDs: {len(rows)}')
print(f'Results: {out.resolve()}')