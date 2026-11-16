"""Decode all indexed JPEGs at 1/8 resolution and check for identical decoded frames across fruit."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


def check(root, row):
    with Image.open(root / row['path']) as im:
        source_size = im.size
        if im.format != 'JPEG':
            raise ValueError(f'Unexpected format in {row["path"]}')
        im.draft('RGB', (64, 64))
        im.load()
        digest = hashlib.sha256(im.tobytes()).hexdigest()
    return source_size, digest, row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(encoding='utf-8-sig') as manifest:
        rows = list(csv.DictReader(manifest))
    groups = defaultdict(list)
    sizes = Counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for size, digest, row in pool.map(lambda row: check(args.root, row), rows):
            sizes[str(size)] += 1
            groups[digest].append(row)
    duplicates = [group for group in groups.values() if len(group) > 1]
    cross_group = [g for g in duplicates if len({r['group'] for r in g}) > 1]
    cross_split = [g for g in duplicates if len({r['split'] for r in g}) > 1]
    summary = dict(decoded=len(rows), original_frame_sizes=dict(sizes),
                   decoded_content_duplicate_sets=len(duplicates),
                   duplicate_sets_crossing_fruit=len(cross_group),
                   duplicate_sets_crossing_split=len(cross_split),
                   duplicates=[[r['path'] for r in group] for group in duplicates])
    args.output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print({k: v for k, v in summary.items() if k != 'duplicates'})


if __name__ == '__main__':
    main()
