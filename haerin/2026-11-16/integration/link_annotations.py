"""Find shared RGB patches and exact visible contours in two Fuji releases."""

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


SPLITS = ("train", "val", "test")


def image_entries(archive):
    entries = {}
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if "images" not in path.parts or path.suffix.lower() != ".png":
            continue
        split = next((part for part in path.parts if part in SPLITS), None)
        if split is None or path.name in entries:
            raise ValueError(f"Unexpected or duplicate image path: {path}")
        entries[path.name] = (split, info)
    return entries


def via_records(archive, path):
    entries = json.loads(archive.read(path))
    return {record["filename"]: record for record in entries.values()}


def regions(record):
    values = record["regions"]
    return list(values.values()) if isinstance(values, dict) else values


def shape_key(region):
    return json.dumps(region["shape_attributes"], sort_keys=True, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ripeness-zip", type=Path, required=True)
    parser.add_argument("--amodal-zip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fields = ["split", "filename", "region_index", "apple_id", "label", "amodal_split"]
    counts = collections.Counter()
    split_pairs = collections.Counter()
    id_sets = {split: set() for split in SPLITS}
    samples = []
    with (ZipFile(args.ripeness_zip) as ripe, ZipFile(args.amodal_zip) as amodal,
          (args.out / "linked_instances.csv").open("w", newline="", encoding="utf-8") as out):
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        first, second = image_entries(ripe), image_entries(amodal)
        counts.update(ripeness_images=len(first), amodal_images=len(second),
                      same_name=len(first.keys() & second.keys()))
        for split in SPLITS:
            labels = via_records(ripe, f"{split}/{split}_via_final.json")
            masks = via_records(amodal, f"02-annotated_data_fuji/gt_json/{split}/via_region_data_modal.json")
            checked_sample = False
            for filename, record in sorted(masks.items()):
                if filename not in first:
                    continue
                r_split, ri = first[filename]
                a_split, ai = second[filename]
                split_pairs[(r_split, a_split)] += 1
                if (ri.file_size, ri.CRC) != (ai.file_size, ai.CRC):
                    counts["different_size_or_crc32"] += 1
                    continue
                counts["same_name_size_crc32"] += 1
                if not checked_sample:
                    sha = hashlib.sha256(ripe.read(ri)).hexdigest()
                    if sha != hashlib.sha256(amodal.read(ai)).hexdigest():
                        raise ValueError(f"Image checksum mismatch for {filename}")
                    samples.append({"split": split, "filename": filename, "sha256": sha})
                    checked_sample = True
                if r_split != split:
                    # Do not accidentally look in the wrong ripeness annotation file.
                    counts["crossed_splits_skipped"] += 1
                    continue
                ripe_regions = regions(labels[filename])
                candidates = collections.defaultdict(list)
                for index, region in enumerate(ripe_regions):
                    candidates[shape_key(region)].append((index, region))
                modal_regions = regions(record)
                modal_counts = collections.Counter(shape_key(region) for region in modal_regions)
                for region in modal_regions:
                    key = shape_key(region)
                    matches = candidates[key]
                    if len(matches) != 1 or modal_counts[key] != 1:
                        counts["ambiguous_or_missing_contour"] += 1
                        continue
                    region_index, ripe_region = matches[0]
                    apple_id = str(region["region_attributes"]["apple_ID"])
                    id_sets[split].add(apple_id)
                    writer.writerow(dict(split=split, filename=filename, region_index=region_index,
                                         apple_id=apple_id, label=ripe_region["region_attributes"]["Ripeness"],
                                         amodal_split=split))
                    counts["unique_matched_contours"] += 1
    report = {"counts": counts, "split_pairs": {"/".join(k):v for k,v in split_pairs.items()},
              "sha256_spot_checks": samples,
              "distinct_apple_ids": {split: len(ids) for split, ids in id_sets.items()},
              "shared_id_strings": {f"{a}/{b}":len(id_sets[a] & id_sets[b])
                                    for a,b in (("train","val"),("train","test"),("val","test"))}}
    (args.out / "overlap.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
