"""Check whether image-only apple weight estimates rely on camera identity.

Part A  original fixed view (spb_int_me_ver). Each weight batch was shot with a different phone there.
Part B  every view in which all 60 apples were shot with the same phone.
Part C  the paired synthetic-leaf comparison, repeated on the original view and on a single-camera view
        with the same location, distance and angle (spm_int_me_ver).

All models reuse the earlier feature extraction, colour mask, leaf generator and ridge head unchanged.
Evaluation is repeated 5-fold cross-validation over apples, stratified by weight batch.
"""

import argparse
import csv
import importlib.util
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageDraw
from scipy import ndimage
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE / "전이학습"))
from imagenet_encoder import WEIGHTS_SHA256, build_encoder, encode_batch  # noqa: E402

SEED = 20261116
ORIGINAL_VIEW = "spb_int_me_ver"
MATCHED_VIEW = "spm_int_me_ver"
LEVELS = (0.0, 0.15, 0.30, 0.50)
LEAF_NAMES = {0.15: "leaf15", 0.30: "leaf30", 0.50: "leaf50"}
FOLDS, REPEATS = 5, 20
FILL = (124, 116, 104)
CORNER = 0.22
BOOTSTRAP = 4000


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = module_from_file("weight_baseline", BASE / "weight/weight_baseline.py")
leaf = module_from_file("synthetic_leaf", BASE / "occlusion/synthetic_leaf.py")
transfer = module_from_file("train_weight", BASE / "전이학습/weight/train_weight.py")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------- data

def index_images(root):
    rows = []
    for path in sorted(root.glob("Fuji_*/*/*")):
        if path.suffix.lower() not in (".jpg", ".jpeg"):
            continue
        *_, support, location, distance, angle = path.stem.lower().split("_")
        with Image.open(path) as im:
            camera = str(im.getexif().get(272, "")).strip("\x00 ")
        rows.append(dict(path=path.relative_to(root).as_posix(), group=path.parent.relative_to(root).as_posix(),
                         weight_g=float(path.parent.name), view=f"{support}_{location}_{distance}_{angle}",
                         location=location, distance=distance, camera=camera))
    per_group = Counter(r["group"] for r in rows)
    per_view = Counter(r["view"] for r in rows)
    if len(per_group) != 60 or set(per_group.values()) != {72} or len(per_view) != 72 or set(per_view.values()) != {60}:
        raise ValueError(f"Expected 60 apples x 72 views, got {len(per_group)} apples and {len(per_view)} views")
    if any(not r["camera"] for r in rows):
        raise ValueError("Missing EXIF camera model")
    return rows


def view_rows(rows, view, groups):
    lookup = {r["group"]: r for r in rows if r["view"] == view}
    return [lookup[g] for g in groups]


# ---------------------------------------------------------------- image variants

def corners(rgb):
    """Four background corners only; the fruit sits in the centre of every frame."""
    w, h = rgb.size
    cw, ch = round(CORNER * w), round(CORNER * h)
    tile = Image.new("RGB", (2 * cw, 2 * ch))
    for i, (x, y) in enumerate(((0, 0), (w - cw, 0), (0, h - ch), (w - cw, h - ch))):
        tile.paste(rgb.crop((x, y, x + cw, y + ch)), ((i % 2) * cw, (i // 2) * ch))
    return tile.resize((224, 224), Image.Resampling.LANCZOS)


def mask_reach(mask):
    """Centre of the largest mask blob's box and the farthest blob pixel from it.

    The colour mask occasionally adds small skin-coloured specks elsewhere in the frame;
    using the largest connected blob keeps one stray speck from setting the disk size.
    """
    labels, count = ndimage.label(np.asarray(mask) > 0)
    if count == 0:
        raise ValueError("Empty fruit mask")
    blob = labels == (np.argmax(np.bincount(labels.ravel())[1:]) + 1)
    yy, xx = np.nonzero(blob)
    centre = ((xx.min() + xx.max()) / 2, (yy.min() + yy.max()) / 2)
    return centre, float(np.sqrt((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2).max())


def erase(rgb, mask, radius):
    """Cover the fruit with one disk whose size is the same in every image, so size is not leaked."""
    (cx, cy), reach = mask_reach(mask)
    if reach > radius:
        raise ValueError("Disk does not cover the fruit blob")
    out = rgb.copy()
    ImageDraw.Draw(out).ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=FILL)
    return out


def encode_rows(encoder, device, rows, data_root, make_variants, cache, signature):
    """Encode named image variants for each row; scalar features are stored alongside.

    The cache is reused only when the image list and the variant settings (signature) both match.
    """
    keys = np.array([r["path"] for r in rows])
    if cache.exists():
        saved = np.load(cache, allow_pickle=False)
        if np.array_equal(saved["keys"], keys) and "signature" in saved.files and str(saved["signature"]) == signature:
            print(f"  cache hit: {cache.name}", flush=True)
            return {k: saved[k] for k in saved.files if k not in ("keys", "signature")}
    pending, feats, scalars = defaultdict(list), defaultdict(list), defaultdict(list)

    def flush(name):
        if pending[name]:
            feats[name].append(encode_batch(encoder, pending[name], device).numpy())
            pending[name].clear()

    for i, row in enumerate(rows):
        rgb = transfer.load_frame(data_root / row["path"])
        images, values = make_variants(rgb, row)
        for name, image in images.items():
            pending[name].append(np.asarray(image))
            if len(pending[name]) == 32:
                flush(name)
        for name, value in values.items():
            scalars[name].append(value)
        if (i + 1) % 300 == 0:
            print(f"  {cache.stem}: {i + 1}/{len(rows)} images", flush=True)
    for name in list(pending):
        flush(name)
    out = {f"x_{name}": np.concatenate(parts).astype(np.float64) for name, parts in feats.items()}
    out.update({f"s_{name}": np.asarray(values, dtype=np.float64) for name, values in scalars.items()})
    np.savez(cache, keys=keys, signature=np.array(signature), **out)
    return out


# ---------------------------------------------------------------- models and evaluation

def fold_table(groups, batch):
    """(REPEATS, n) fold ids. Every fold holds 4 apples from each 20-apple weight batch."""
    table = np.empty((REPEATS, len(groups)), dtype=int)
    for r in range(REPEATS):
        rng = np.random.default_rng([SEED, r])
        for b in sorted(set(batch)):
            members = np.flatnonzero(np.asarray(batch) == b)
            table[r, rng.permutation(members)] = np.arange(len(members)) % FOLDS
    return table


def cross_predict(folds, model):
    """folds: (R, n) with fold ids >= 0 for evaluated apples and -1 for train-only apples."""
    preds = np.full(folds.shape, np.nan)
    for r, fold in enumerate(folds):
        for f in np.unique(fold[fold >= 0]):
            test = fold == f
            preds[r, test] = model(~test, test)
    return preds


def ridge(x, y):
    def model(train, test):
        coef, bias = transfer.fit_head(x[train], y[train])
        return x[test] @ coef + bias
    return model


def line(feature, y):
    def model(train, test):
        fit = baseline.fit_line([dict(feature=f, weight_g=w) for f, w in zip(feature[train], y[train])])
        return fit["intercept"] + fit["slope"] * feature[test]
    return model


def line_by_group(feature, y, label):
    def model(train, test):
        out = np.empty(test.sum())
        for b in np.unique(label[test]):
            fit = baseline.fit_line([dict(feature=f, weight_g=w)
                                     for f, w in zip(feature[train & (label == b)], y[train & (label == b)])])
            sel = label[test] == b
            out[sel] = fit["intercept"] + fit["slope"] * feature[test][sel]
        return out
    return model


def global_mean(y):
    return lambda train, test: np.full(test.sum(), y[train].mean())


def group_mean(y, label):
    def model(train, test):
        means = {b: y[train & (label == b)].mean() for b in np.unique(label[train])}
        return np.array([means[b] for b in label[test]])
    return model


def ranks(values):
    order = np.argsort(values, kind="stable")
    out = np.empty(len(values))
    out[order] = np.arange(len(values))
    return out


def spearman(a, b):
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def summarize(preds, y, batch):
    errors = np.abs(preds - y)
    per_repeat = np.nanmean(errors, axis=1)
    out = dict(mae_g=float(per_repeat.mean()),
               mae_repeat_min_g=float(per_repeat.min()), mae_repeat_max_g=float(per_repeat.max()))
    if np.isnan(preds).any():
        return out
    avg = preds.mean(axis=0)
    within = {b: spearman(avg[batch == b], y[batch == b]) for b in np.unique(batch)}
    out.update(spearman_all=spearman(avg, y), spearman_within_batch=within,
               spearman_within_batch_mean=float(np.mean(list(within.values()))))
    return out


def paired_difference(preds_a, preds_b, y, batch):
    """Mean |error| of A minus B with a bootstrap 95% interval over apples, stratified by batch."""
    err_a = np.abs(preds_a - y).mean(axis=0)
    err_b = np.abs(preds_b - y).mean(axis=0)
    rng = np.random.default_rng(SEED)
    pools = [np.flatnonzero(batch == b) for b in np.unique(batch)]
    draws = np.empty(BOOTSTRAP)
    for i in range(BOOTSTRAP):
        sample = np.concatenate([rng.choice(pool, len(pool)) for pool in pools])
        draws[i] = err_a[sample].mean() - err_b[sample].mean()
    low, high = np.percentile(draws, [2.5, 97.5])
    return dict(difference_g=float(err_a.mean() - err_b.mean()), ci95_g=[float(low), float(high)])


def camera_accuracy(x, label, folds):
    classes = np.unique(label)
    target = (label[:, None] == classes[None, :]).astype(float)
    hits = []
    for fold in folds:
        correct = 0
        for f in np.unique(fold):
            test = fold == f
            scores = np.column_stack([x[test] @ c + b for c, b in
                                      (transfer.fit_head(x[~test], target[~test, k]) for k in range(len(classes)))])
            correct += int((classes[scores.argmax(axis=1)] == label[test]).sum())
        hits.append(correct / len(label))
    return float(np.mean(hits))


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("D:/Apple-harvest-data/weight"))
    parser.add_argument("--weights", type=Path, default=BASE / "전이학습/weights/resnet18-f37072fd.pth")
    parser.add_argument("--cache", type=Path, default=HERE / "cache")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)

    rows = index_images(args.data_root)
    # The earlier experiment's camera column and 48/12 split, keyed by apple.
    earlier = {r["group"]: r for r in read_csv(BASE / "weight/results/weight_features.csv") if float(r["target_fraction"]) == 0}
    groups = sorted(earlier)
    batch = np.array([earlier[g]["camera"] for g in groups])
    y = np.array([float(earlier[g]["weight_g"]) for g in groups])
    for r in view_rows(rows, ORIGINAL_VIEW, groups):
        if r["camera"] != earlier[r["group"]]["camera"]:
            raise ValueError(f"Camera mismatch for {r['group']}")
    folds = fold_table(groups, batch)
    fixed = np.array([[0 if earlier[g]["split"] == "test" else -1 for g in groups]])

    cameras_by_view = defaultdict(Counter)
    for r in rows:
        cameras_by_view[r["view"]][r["camera"]] += 1
    single_views = sorted(v for v, c in cameras_by_view.items() if len(c) == 1)
    write_csv(args.out / "camera_by_view.csv",
              [dict(view=v, single_camera=len(c) == 1, **{cam: c.get(cam, 0) for cam in ("iPhone 8", "iPhone 11", "LM-X430")})
               for v, c in sorted(cameras_by_view.items())])
    print(f"Views with one camera for all 60 apples: {len(single_views)} / 72", flush=True)

    encoder = build_encoder(args.weights, args.device)

    # Fixed erase radius: farthest fruit pixel from its box centre over all original-view images, plus 10%.
    original_rows = view_rows(rows, ORIGINAL_VIEW, groups)
    reach = [mask_reach(baseline.fruit_pixels(transfer.load_frame(args.data_root / r["path"])))[1] for r in original_rows]
    radius = 1.10 * max(reach)

    def leaf_variants(rgb, row, extra=()):
        mask = baseline.fruit_pixels(rgb)
        images = {"full": transfer.preprocess_image(rgb)}
        values = {"area": baseline.feature(mask)}
        for level, name in LEAF_NAMES.items():
            frame, coverage = leaf.apply_leaf(rgb, mask, level, row["group"], seed=baseline.SEED)
            images[name] = transfer.preprocess_image(frame)
            values[f"area_{name}"] = baseline.feature(baseline.fruit_pixels(frame))
            values[f"coverage_{name}"] = coverage
        if "erased" in extra:
            images["erased"] = transfer.preprocess_image(erase(rgb, mask, radius))
        if "corners" in extra:
            images["corners"] = corners(rgb)
        return images, values

    def single_variants(rgb, row):
        return ({"full": transfer.preprocess_image(rgb), "corners": corners(rgb)},
                {"area": baseline.feature(baseline.fruit_pixels(rgb))})

    leaf_sig = f"leaf_seed={baseline.SEED};levels={sorted(LEAF_NAMES)};weights={WEIGHTS_SHA256[:12]}"
    print("Encoding original view", flush=True)
    orig = encode_rows(encoder, args.device, original_rows, args.data_root,
                       lambda rgb, row: leaf_variants(rgb, row, ("erased", "corners")), args.cache / "original_view.npz",
                       f"{leaf_sig};erase_radius={radius:.6f};corner={CORNER}")
    print("Encoding matched single-camera view with leaves", flush=True)
    matched = encode_rows(encoder, args.device, view_rows(rows, MATCHED_VIEW, groups), args.data_root,
                          leaf_variants, args.cache / "matched_view_leaf.npz", leaf_sig)
    print("Encoding all single-camera views", flush=True)
    single_rows = [r for v in single_views for r in view_rows(rows, v, groups)]
    single = encode_rows(encoder, args.device, single_rows, args.data_root, single_variants,
                         args.cache / "single_camera_views.npz", f"corner={CORNER};weights={WEIGHTS_SHA256[:12]}")

    # Sanity: the colour-mask feature must reproduce the earlier experiment exactly.
    earlier_area = {(r["group"], float(r["target_fraction"])): float(r["feature"])
                    for r in read_csv(BASE / "weight/results/weight_features.csv")}
    for level in LEVELS:
        name = "s_area" if level == 0 else f"s_area_{LEAF_NAMES[level]}"
        expected = np.array([earlier_area[(g, level)] for g in groups])
        if not np.allclose(orig[name], expected, rtol=0, atol=1e-12):
            raise ValueError(f"Area feature differs from the earlier run at level {level}")

    report = dict(
        purpose="Does the image-only weight estimate measure the apple, or the camera that photographed each weight batch?",
        settings=dict(folds=FOLDS, repeats=REPEATS, seed=SEED, ridge="earlier head: lambda 0.1, unpenalised bias",
                      encoder="TorchVision ResNet18 IMAGENET1K_V1, frozen", weights_sha256=WEIGHTS_SHA256,
                      erase_radius_px_on_320_frame=radius, corner_fraction=CORNER, bootstrap=BOOTSTRAP,
                      device=args.device, python=platform.python_version(), torch=torch.__version__,
                      numpy=np.__version__, pillow=PIL.__version__),
        data=dict(apples=len(groups), views=72, single_camera_views=len(single_views),
                  single_camera_images=len(single_rows),
                  batch_weight_ranges_g={b: [float(y[batch == b].min()), float(y[batch == b].max())] for b in np.unique(batch)},
                  batch_weight_sd_g={b: float(y[batch == b].std(ddof=1)) for b in np.unique(batch)}),
    )

    # ------------------------------------------------ Part A
    print("Part A", flush=True)
    models_a = {
        "global_mean": global_mean(y),
        "camera_mean": group_mean(y, batch),
        "area_pooled": line(orig["s_area"], y),
        "area_by_camera": line_by_group(orig["s_area"], y, batch),
        "resnet_full": ridge(orig["x_full"], y),
        "resnet_fruit_erased": ridge(orig["x_erased"], y),
        "resnet_corners_only": ridge(orig["x_corners"], y),
    }
    preds_a = {name: cross_predict(folds, m) for name, m in models_a.items()}
    fixed_a = {name: cross_predict(fixed, m) for name, m in models_a.items()}
    report["part_a_original_view"] = dict(
        view=ORIGINAL_VIEW,
        camera_identification_accuracy={k: camera_accuracy(orig[f"x_{k}"], batch, folds) for k in ("full", "erased", "corners")},
        cross_validation={name: summarize(p, y, batch) for name, p in preds_a.items()},
        earlier_fixed_split_12_apples={name: float(np.nanmean(np.abs(p - y))) for name, p in fixed_a.items()},
        paired_differences=dict(
            resnet_full_minus_camera_mean=paired_difference(preds_a["resnet_full"], preds_a["camera_mean"], y, batch),
            resnet_erased_minus_resnet_full=paired_difference(preds_a["resnet_fruit_erased"], preds_a["resnet_full"], y, batch),
            resnet_full_minus_global_mean=paired_difference(preds_a["resnet_full"], preds_a["global_mean"], y, batch)),
    )

    # ------------------------------------------------ Part B
    print("Part B", flush=True)
    view_of = np.array([r["view"] for r in single_rows])
    per_view, preds_b = [], defaultdict(dict)
    for v in single_views:
        sel = view_of == v
        models_b = {"global_mean": global_mean(y), "area_pooled": line(single["s_area"][sel], y),
                    "resnet_full": ridge(single["x_full"][sel], y),
                    "resnet_corners_only": ridge(single["x_corners"][sel], y)}
        location, distance = v.split("_")[1:3]
        for name, m in models_b.items():
            p = cross_predict(folds, m)
            preds_b[name][v] = p
            s = summarize(p, y, batch)
            per_view.append(dict(view=v, location=location, distance=distance, model=name, mae_g=s["mae_g"],
                                 spearman_all=s["spearman_all"], spearman_within_batch_mean=s["spearman_within_batch_mean"],
                                 **{f"spearman_within_{b}": s["spearman_within_batch"][b] for b in np.unique(batch)}))
    write_csv(args.out / "single_camera_views.csv", per_view)

    def view_stats(model, keep=lambda row: True):
        chosen = [r for r in per_view if r["model"] == model and keep(r)]
        mae = np.array([r["mae_g"] for r in chosen])
        within = np.array([r["spearman_within_batch_mean"] for r in chosen])
        return dict(views=len(chosen), mae_median_g=float(np.median(mae)), mae_min_g=float(mae.min()),
                    mae_max_g=float(mae.max()), spearman_within_batch_median=float(np.median(within)))

    multi = {name: np.mean([preds_b[name][v] for v in single_views], axis=0) for name in preds_b}
    report["part_b_single_camera_views"] = dict(
        views=single_views,
        all_views={m: view_stats(m) for m in preds_b},
        by_distance={d: {m: view_stats(m, lambda r, d=d: r["distance"] == d) for m in preds_b} for d in ("al", "me", "ce")},
        by_location={loc: {m: view_stats(m, lambda r, loc=loc: r["location"] == loc) for m in preds_b} for loc in ("int", "ext")},
        matched_view=dict(view=MATCHED_VIEW, cross_validation={m: summarize(preds_b[m][MATCHED_VIEW], y, batch) for m in preds_b},
                          earlier_fixed_split_12_apples={
                              "global_mean": float(np.nanmean(np.abs(cross_predict(fixed, global_mean(y)) - y))),
                              "resnet_full": float(np.nanmean(np.abs(cross_predict(fixed, ridge(matched["x_full"], y)) - y)))},
                          paired_differences=dict(
                              resnet_full_minus_global_mean=paired_difference(preds_b["resnet_full"][MATCHED_VIEW], preds_b["global_mean"][MATCHED_VIEW], y, batch),
                              resnet_corners_minus_global_mean=paired_difference(preds_b["resnet_corners_only"][MATCHED_VIEW], preds_b["global_mean"][MATCHED_VIEW], y, batch))),
        average_over_all_single_camera_views=dict(
            cross_validation={m: summarize(p, y, batch) for m, p in multi.items()},
            paired_differences=dict(
                resnet_full_minus_global_mean=paired_difference(multi["resnet_full"], multi["global_mean"], y, batch),
                resnet_corners_minus_global_mean=paired_difference(multi["resnet_corners_only"], multi["global_mean"], y, batch),
                resnet_full_minus_resnet_corners=paired_difference(multi["resnet_full"], multi["resnet_corners_only"], y, batch))),
    )
    # Same pixels, different GPU batch composition: only float rounding may differ.
    drift = float(np.abs(matched["x_full"] - single["x_full"][view_of == MATCHED_VIEW]).max())
    if drift > 1e-2:
        raise ValueError(f"Matched-view features differ between caches by {drift}")
    report["settings"]["max_feature_difference_from_gpu_batching"] = drift

    # ------------------------------------------------ Part C
    print("Part C", flush=True)
    occlusion_rows, report["part_c_leaf_occlusion"] = [], {}
    for label, feats in ((ORIGINAL_VIEW, orig), (MATCHED_VIEW, matched)):
        x_level = {0.0: feats["x_full"], **{lv: feats[f"x_{n}"] for lv, n in LEAF_NAMES.items()}}
        a_level = {0.0: feats["s_area"], **{lv: feats[f"s_area_{n}"] for lv, n in LEAF_NAMES.items()}}
        result = {}
        for split_name, table in (("cross_validation", folds), ("earlier_fixed_split_12_apples", fixed)):
            result[split_name] = {}
            for model_name in ("resnet_clean", "resnet_aug", "area_clean", "area_aug"):
                kind = "resnet" if model_name.startswith("resnet") else "area"
                source = x_level if kind == "resnet" else a_level
                preds = {lv: np.full(table.shape, np.nan) for lv in LEVELS}
                for r, fold in enumerate(table):
                    for f in np.unique(fold[fold >= 0]):
                        test, train = fold == f, fold != f
                        levels = LEVELS if model_name.endswith("aug") else (0.0,)
                        xs = np.concatenate([source[lv][train] for lv in levels])
                        ys = np.concatenate([y[train] for _ in levels])
                        if kind == "resnet":
                            coef, bias = transfer.fit_head(xs, ys)
                            for lv in LEVELS:
                                preds[lv][r, test] = source[lv][test] @ coef + bias
                        else:
                            fit = baseline.fit_line([dict(feature=a, weight_g=w) for a, w in zip(xs, ys)])
                            for lv in LEVELS:
                                preds[lv][r, test] = fit["intercept"] + fit["slope"] * source[lv][test]
                entries = []
                for lv in LEVELS:
                    entry = dict(level=lv, mae_g=float(np.nanmean(np.nanmean(np.abs(preds[lv] - y), axis=1))),
                                 mean_abs_change_from_clean_g=float(np.nanmean(np.abs(preds[lv] - preds[0.0]))))
                    entries.append(entry)
                    occlusion_rows.append(dict(view=label, evaluation=split_name, model=model_name, **entry))
                result[split_name][model_name] = entries
        report["part_c_leaf_occlusion"][label] = result
    write_csv(args.out / "leaf_occlusion.csv", occlusion_rows)

    # ------------------------------------------------ per-apple predictions for figures
    apple_rows = []
    for label, preds in (("original_view", preds_a),
                         ("matched_view", {m: preds_b[m][MATCHED_VIEW] for m in preds_b}),
                         ("all_single_camera_views_average", multi)):
        for name, p in preds.items():
            avg = p.mean(axis=0)
            for g, b, w, est in zip(groups, batch, y, avg):
                apple_rows.append(dict(setting=label, model=name, group=g, weight_batch=b, weight_g=w,
                                       estimate_g=float(est), abs_error_g=float(abs(est - w))))
    write_csv(args.out / "apple_predictions.csv", apple_rows)
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("data", "part_a_original_view")}, indent=2, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
