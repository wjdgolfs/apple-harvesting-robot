"""Audit IntegrALIMENTA Fuji labels; fit and evaluate fixed-view, camera-specific mass regressions."""

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import platform
import statistics as stats
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import PIL
from PIL import Image, ImageChops, ImageDraw, ImageOps


SEED = 20261116
LEVELS = (0.0, 0.15, 0.30, 0.50)
CONDITION = ('spb', 'int', 'me', 'ver')
EXPECTED = set(itertools.product(('pgb', 'ppb', 'spb', 'spm'), ('int', 'ext'),
                                 ('al', 'ce', 'me'), ('cen', 'sup', 'ver')))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def index_images(root):
    rows, groups = [], defaultdict(list)
    for pack in sorted(root.glob('Fuji_*')):
        if not pack.is_dir():
            continue
        for path in sorted(pack.rglob('*')):
            if path.suffix.lower() not in ('.jpg', '.jpeg'):
                continue
            site, kind, variety, grams, *condition = path.stem.lower().split('_')
            condition = tuple(condition)
            weight = Decimal(grams.replace(',', '.'))
            if weight != Decimal(path.parent.name) or condition not in EXPECTED:
                raise ValueError(f'Unmatched weight or condition: {path}')
            if variety != 'fuji' or kind != 'm' or path.parent.parent != pack:
                raise ValueError(f'Unexpected folder or filename: {path}')
            # The two packs have distinct weight folders; a collision is checked below.
            group = path.parent.relative_to(root).as_posix()
            row = dict(path=path.relative_to(root).as_posix(), group=group, weight_g=float(weight),
                       site=site, variety=variety, support=condition[0], location=condition[1],
                       distance=condition[2], angle=condition[3])
            rows.append(row)
            groups[group].append((condition, row))
    if not rows:
        raise ValueError(f'No Fuji JPEGs in {root}')
    anchors = []
    seen = {}
    for group, pictures in groups.items():
        counts = Counter(condition for condition, _ in pictures)
        if set(counts) != EXPECTED or any(n != 1 for n in counts.values()):
            raise ValueError(f'Missing/duplicate condition in {group}: {counts}')
        sample = next(row for condition, row in pictures if condition == CONDITION).copy()
        key = (sample['site'], sample['variety'], sample['weight_g'])
        if key in seen:
            raise ValueError(f'Weight identifier is ambiguous for {group} and {seen[key]}')
        seen[key] = group
        with Image.open(root / sample['path']) as im:
            sample['camera'] = str(im.getexif().get(272, '')).strip('\x00 ')
        if not sample['camera']:
            raise ValueError(f'Missing camera EXIF in {sample["path"]}')
        anchors.append(sample)
    return rows, anchors


def assign_splits(anchors):
    # Stratify by EXIF camera to fit/evaluate each camera on distinct apples.
    cohorts = defaultdict(list)
    for row in anchors:
        cohorts[row['camera']].append(row)
    for camera, cohort in cohorts.items():
        if len(cohort) < 5:
            raise ValueError(f'Too few apples for {camera}')
        cohort.sort(key=lambda row: hashlib.sha256(f'{SEED}:{row["group"]}'.encode()).digest())
        n_test = max(1, round(0.2 * len(cohort)))
        for i, row in enumerate(cohort):
            row['split'] = 'test' if i < n_test else 'train'


def load_frame(path):
    with Image.open(path) as im:
        im.draft('RGB', (384, 384))
        rgb = ImageOps.exif_transpose(im).convert('RGB')
    rgb.thumbnail((320, 320), Image.Resampling.LANCZOS)
    return rgb


def fruit_pixels(image):
    # The plain-background, overhead setup puts the coin near an upper corner.
    # Use the warm fruit skin signal in the central ROI; do not use weight or filename.
    width, height = image.size
    r, g, b = image.split()
    red_blue = ImageChops.subtract(r, b).point(lambda value: 255 if value >= 30 else 0)
    green_red = ImageChops.subtract(g, r).point(lambda value: 255 if value <= 9 else 0)
    mask = ImageChops.darker(red_blue, green_red)
    roi = (round(0.1 * width), round(0.18 * height), round(0.9 * width), round(0.94 * height))
    central_mask = Image.new('L', image.size)
    central_mask.paste(mask.crop(roi), roi[:2])
    return central_mask


def feature(mask):
    area = mask.histogram()[255] / (mask.width * mask.height)
    return area ** 1.5


def fit(rows):
    # Separate camera regressions prevent a change of focal length from changing scale across devices.
    models = {}
    for camera in sorted({r['camera'] for r in rows}):
        data = [r for r in rows if r['camera'] == camera]
        models[camera] = fit_line(data)
    return models


def fit_line(rows):
    xy = [(r['feature'], r['weight_g']) for r in rows]
    mean_x, mean_y = map(stats.mean, zip(*xy))
    denominator = sum((x - mean_x) ** 2 for x, y in xy)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in xy) / denominator
    return dict(slope=slope, intercept=mean_y - slope * mean_x)


def coverage_overlay(rgb, mask):
    return Image.composite(Image.blend(rgb, Image.new('RGB', rgb.size, '#16deb6'), 0.45), rgb, mask)


def measure(root, anchors, apply_leaf, preview_path):
    rows = []
    examples = [r['group'] for r in anchors if r['split'] == 'test']
    footer_y = len(examples) * 155 + 30
    preview = Image.new('RGB', (5 * 130, footer_y + 57), 'white')
    draw = ImageDraw.Draw(preview)
    for i, label in enumerate(['Original', 'Fruit mask', 'Leaf 15%', 'Leaf 30%', 'Leaf 50%']):
        draw.text((130*i+3, 4), label, fill='black')
    for sample in anchors:
        rgb = load_frame(root / sample['path'])
        mask = fruit_pixels(rgb)
        if mask.getbbox() is None:
            raise ValueError(f'Empty source mask in {sample["path"]}')
        frames = [rgb, coverage_overlay(rgb, mask)]
        for level in LEVELS:
            modified, coverage = apply_leaf(rgb, mask, level, sample['group'], seed=SEED)
            measure_mask = fruit_pixels(modified) if level else mask
            rows.append({**sample, 'target_fraction': level, 'measured_fraction': coverage,
                         'feature': feature(measure_mask)})
            if level:
                frames.append(modified)
        if sample['split'] == 'test':
            y = 30 + 155 * examples.index(sample['group'])
            for i, frame in enumerate(frames):
                tile = frame.copy()
                tile.thumbnail((126, 130))
                preview.paste(tile, (130*i, y))
            draw.text((2, y+132), f'{sample["group"]} | {sample["camera"]} | {sample["weight_g"]:.2f} g', fill='black')
    draw.text((3, footer_y), 'Images: Motilva et al. (2025), IntegrALIMENTA Apple fruit laboratory images, DIGITAL.CSIC.', fill='black')
    draw.text((3, footer_y + 17), 'https://hdl.handle.net/10261/397150 | CC BY-NC-SA 4.0', fill='black')
    draw.text((3, footer_y + 34), 'Adaptation: resized images, colour masks and simulated leaf overlays. CC BY-NC-SA 4.0.', fill='black')
    preview.save(preview_path, quality=88)
    return rows


def evaluate(rows):
    train = [r for r in rows if r['split'] == 'train']
    clean_train = [r for r in train if r['target_fraction'] == 0]
    models = {'clean': fit(clean_train), 'aug': fit(train)}
    pooled = fit_line(clean_train)
    models.update(pooled_clean={'all': pooled}, pooled_aug={'all': fit_line(train)})
    global_mean = stats.mean(r['weight_g'] for r in clean_train)
    camera_means = {camera: stats.mean(r['weight_g'] for r in clean_train if r['camera'] == camera)
                    for camera in models['clean']}
    predictions, clean_pred = [], {}
    for row in rows:
        if row['split'] != 'test':
            continue
        for variant, model in models.items():
            equation = model.get(row['camera']) or model['all']
            estimate = equation['intercept'] + equation['slope'] * row['feature']
            key = (row['group'], variant)
            if row['target_fraction'] == 0:
                clean_pred[key] = estimate
            predictions.append({**row, 'model': variant, 'estimate_g': round(estimate, 6),
                                'abs_error_g': abs(estimate - row['weight_g']),
                                'abs_change_from_clean_g': abs(estimate - clean_pred[key]),
                                'pooled_area_estimate_g': pooled['intercept'] + pooled['slope'] * row['feature'],
                                'global_train_mean_g': global_mean,
                                'camera_train_mean_g': camera_means[row['camera']]})
    clean_test = [r for r in predictions if r['model'] == 'clean' and r['target_fraction'] == 0]
    summary = {'global_mean_mae_g': stats.mean(abs(r['weight_g'] - global_mean) for r in clean_test),
               'area_only_mae_g': stats.mean(abs(r['weight_g'] - r['pooled_area_estimate_g']) for r in clean_test),
               'camera_mean_mae_g': stats.mean(abs(r['weight_g'] - r['camera_train_mean_g']) for r in clean_test),
               'pooled_area_coefficients': pooled, 'coefficients': models, 'by_level': {}}
    for name in models:
        summaries = []
        for level in LEVELS:
            results = [r for r in predictions if r['model'] == name and r['target_fraction'] == level]
            summaries.append(dict(target=level, n=len(results), mae_g=stats.mean(r['abs_error_g'] for r in results),
                                  mean_abs_change_g=stats.mean(r['abs_change_from_clean_g'] for r in results),
                                  actual_coverage=stats.mean(r['measured_fraction'] for r in results)))
        summary['by_level'][name] = summaries
    return predictions, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Folder containing extracted Fuji_1 and Fuji_2')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--leaf-module', type=Path, required=True, help='Path to synthetic_leaf.py')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('synthetic_leaf', args.leaf_module)
    leaf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(leaf)
    index, anchors = index_images(args.root)
    assign_splits(anchors)
    split_map = {r['group']: r['split'] for r in anchors}
    for r in index:
        r['split'] = split_map[r['group']]
    write_csv(args.out / 'weight_manifest.csv', index)
    measurements = measure(args.root, anchors, leaf.apply_leaf, args.out / 'weight_mask_review.jpg')
    predictions, scores = evaluate(measurements)
    write_csv(args.out / 'weight_predictions.csv', predictions)
    write_csv(args.out / 'weight_features.csv', measurements)
    tested = [r for r in anchors if r['split'] == 'test']
    scores.update(dict(images_indexed=len(index), fruit_groups=len(anchors),
                       groups_with_all_72_conditions=len(anchors), train_fruit=len(anchors)-len(tested),
                       test_fruit=len(tested), min_g=min(r['weight_g'] for r in anchors),
                       max_g=max(r['weight_g'] for r in anchors), fixed_condition=CONDITION,
                       camera_fruit_counts=dict(Counter(r['camera'] for r in anchors)),
                       camera_mass_ranges_g={c: [min(r['weight_g'] for r in anchors if r['camera']==c),
                                                  max(r['weight_g'] for r in anchors if r['camera']==c)]
                                             for c in sorted({r['camera'] for r in anchors})},
                       camera_test_counts=dict(Counter(r['camera'] for r in tested)),
                       split_seed=SEED, python=platform.python_version(), pillow=PIL.__version__))
    (args.out / 'weight_metrics.json').write_text(json.dumps(scores, indent=2), encoding='utf-8')
    print(json.dumps(scores, indent=2))


if __name__ == '__main__':
    main()
