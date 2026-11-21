"""Compare the earlier ripeness head, the same head on more data, and a fine-tuned ResNet18 on one fixed test set.

Everything that is chosen (best epoch, thresholds, which model to keep) uses the validation apples only.
Three-way decision: ripe if score >= t_ripe, unripe if score < t_unripe, otherwise hold for another view.
"""

import argparse
import copy
import csv
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
import torch
from torch import nn
from torchvision.models import resnet18

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE / "전이학습"))
from imagenet_encoder import RGB_MEAN, RGB_STD, WEIGHTS_SHA256, build_encoder, encode_batch  # noqa: E402

LEVELS = (0, 15, 30, 50)
BOOTSTRAP = 2000


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fit = load("fit_ripeness", BASE / "전이학습/ripeness/fit_ripeness.py")


# ---------------------------------------------------------------- metrics

def auc(y, s):
    ranks = rankdata(s)  # average ranks for ties
    pos = y.sum()
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * (len(y) - pos)))


def average_precision(y, s):
    order = np.argsort(-s, kind="stable")
    hits = y[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / hits.sum())


def balanced_threshold(y, s):
    return float(fit.val_threshold(torch.tensor(s), torch.tensor(y.astype(float))))


def three_way_thresholds(y, s, precision_target, recall_floor):
    """t_ripe: lowest score whose picks reach the precision target. t_unripe: keeps the recall floor of ripe."""
    order = np.sort(np.unique(s))
    t_ripe = None
    for t in order:
        picked = s >= t
        if picked.any() and y[picked].mean() >= precision_target:
            t_ripe = float(t)
            break
    if t_ripe is None:
        t_ripe = float(order[-1]) + 1e-9
    ripe_scores = np.sort(s[y])
    k = int(math.floor((1 - recall_floor) * len(ripe_scores)))
    t_unripe = float(ripe_scores[k]) if len(ripe_scores) else t_ripe
    return t_ripe, min(t_unripe, t_ripe)


def decide(s, t_ripe, t_unripe):
    return np.where(s >= t_ripe, "ripe", np.where(s < t_unripe, "unripe", "hold"))


def summarize(y, s, thresholds):
    t_bal, t_ripe, t_unripe = thresholds
    picked = s >= t_bal
    tp = int((picked & y).sum())
    out = dict(n=int(len(y)), ripe=int(y.sum()), auc=auc(y, s), ap=average_precision(y, s),
               at_balanced_threshold=dict(picked=int(picked.sum()), tp=tp, precision=tp / max(picked.sum(), 1),
                                          recall=tp / y.sum(), unripe_recall=float((~picked & ~y).sum() / (~y).sum())))
    d = decide(s, t_ripe, t_unripe)
    ripe_calls = d == "ripe"
    out["three_way"] = dict(ripe_calls=int(ripe_calls.sum()),
                            ripe_call_precision=float(y[ripe_calls].mean()) if ripe_calls.any() else None,
                            ripe_found=float((ripe_calls & y).sum() / y.sum()),
                            ripe_called_unripe=float(((d == "unripe") & y).sum() / y.sum()),
                            unripe_called_ripe=int((ripe_calls & ~y).sum()),
                            hold_fraction=float((d == "hold").mean()),
                            hold_ripe=int(((d == "hold") & y).sum()), hold_unripe=int(((d == "hold") & ~y).sum()))
    return out


def bootstrap_by_apple(y, s, ids, seed=20261116):
    """95% intervals for AP and AUC, resampling apples (all their crops together)."""
    groups = defaultdict(list)
    for i, a in enumerate(ids):
        groups[a].append(i)
    keys = list(groups)
    rng = np.random.default_rng(seed)
    aps, aucs = [], []
    for _ in range(BOOTSTRAP):
        pick = np.concatenate([groups[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
        if 0 < y[pick].sum() < len(pick):
            aps.append(average_precision(y[pick], s[pick]))
            aucs.append(auc(y[pick], s[pick]))
    return dict(ap_ci95=[float(np.percentile(aps, 2.5)), float(np.percentile(aps, 97.5))],
                auc_ci95=[float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))])


# ---------------------------------------------------------------- fine-tuning

def make_model(weights):
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.fc = nn.Linear(512, 1)
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(("layer3", "layer4", "fc"))
    return model


def set_train_mode(model):
    model.train()
    for name in ("conv1", "bn1", "layer1", "layer2"):
        getattr(model, name).eval()  # frozen stem keeps ImageNet BN statistics


def normalize(batch_u8, device):
    x = torch.as_tensor(batch_u8).to(device).permute(0, 3, 1, 2).float().div_(255)
    mean = x.new_tensor(RGB_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(RGB_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def augment(x, gen):
    """Flips, 90-degree turns, mild brightness/contrast. No hue change: colour is the label cue."""
    n = x.shape[0]
    flip_h = torch.rand(n, generator=gen) < 0.5
    flip_v = torch.rand(n, generator=gen) < 0.5
    x[flip_h] = x[flip_h].flip(3)
    x[flip_v] = x[flip_v].flip(2)
    k = int(torch.randint(0, 4, (1,), generator=gen))
    x = torch.rot90(x, k, dims=(2, 3))
    gain = (1 + 0.1 * (2 * torch.rand(n, 1, 1, 1, generator=gen) - 1)).to(x.device)
    bias = (0.1 * (2 * torch.rand(n, 1, 1, 1, generator=gen) - 1)).to(x.device)
    return x * gain + bias


@torch.inference_mode()
def predict(model, crops, rows, device, batch=256):
    model.eval()
    out = []
    for i in range(0, len(rows), batch):
        idx = [r["row"] for r in rows[i:i + batch]]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out.append(model(normalize(crops[idx], device)).float().squeeze(1).cpu())
    return torch.cat(out).numpy().astype(np.float64)


def finetune(crops, train_items, val_rows, weights, device, seed, epochs, log):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = make_model(weights).to(device)
    labels = np.array([it["label"] == "ripe" for it in train_items])
    pos_weight = torch.tensor([(~labels).sum() / labels.sum()], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    params = [{"params": [p for n, p in model.named_parameters() if n.startswith("fc")], "lr": 1e-3},
              {"params": [p for n, p in model.named_parameters() if n.startswith(("layer3", "layer4"))], "lr": 3e-4}]
    optimizer = torch.optim.AdamW(params, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(train_items) / 64)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[1e-3, 3e-4], total_steps=epochs * steps_per_epoch,
                                                    pct_start=0.15)
    y_val = np.array([r["label"] == "ripe" for r in val_rows])
    best = (-1, None, -1)
    history = []
    for epoch in range(epochs):
        set_train_mode(model)
        order = torch.randperm(len(train_items), generator=gen).tolist()
        total = 0.0
        for i in range(0, len(order), 64):
            chosen = [train_items[j] for j in order[i:i + 64]]
            # each region has a clean and a leaf crop; pick one at random every time it is drawn
            use_leaf = (torch.rand(len(chosen), generator=gen) < 0.5).tolist()
            idx = [it["leaf_row"] if leaf_flag else it["row"] for it, leaf_flag in zip(chosen, use_leaf)]
            x = augment(normalize(crops[idx], device), gen)
            target = torch.tensor([it["label"] == "ripe" for it in chosen], dtype=torch.float32, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = loss_fn(model(x).float().squeeze(1), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(chosen)
        val_ap = average_precision(y_val, predict(model, crops, val_rows, device))
        history.append(dict(epoch=epoch + 1, train_loss=total / len(order), val_ap=val_ap))
        log(f"  seed {seed} epoch {epoch + 1}/{epochs}  loss {total / len(order):.4f}  val AP {val_ap:.3f}")
        if val_ap > best[0]:
            best = (val_ap, copy.deepcopy(model.state_dict()), epoch + 1)
    model.load_state_dict(best[1])
    return model, dict(seed=seed, best_epoch=best[2], best_val_ap=best[0], history=history)


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=HERE / "cache")
    parser.add_argument("--weights", type=Path, default=BASE / "전이학습/weights/resnet18-f37072fd.pth")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--model-out", type=Path, default=HERE / "weights")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--precision-target", type=float, default=0.80,
                        help="Team assumption: share of 'ripe' calls that must be ripe on validation")
    parser.add_argument("--recall-floor", type=float, default=0.95,
                        help="Team assumption: share of ripe apples that must not be called unripe on validation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.model_out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    started = time.time()
    log_lines = []

    def log(text):
        print(text, flush=True)
        log_lines.append(text)

    crops = np.load(args.cache / "crops.npy", mmap_mode="r")
    with (args.cache / "index.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for r in rows:
        r["row"] = int(r["row"])
        r["cover_percent"] = int(r["cover_percent"])
    split_rows = lambda s, lv=0: [r for r in rows if r["split"] == s and r["cover_percent"] == lv]
    train_clean = split_rows("train")
    leaf_of = {(r["filename"], r["region_index"]): r["row"] for r in rows if r["split"] == "train" and r["cover_percent"]}
    train_items = [dict(r, leaf_row=leaf_of[(r["filename"], r["region_index"])]) for r in train_clean]
    val_rows = split_rows("val")
    test_rows = {lv: split_rows("test", lv) for lv in LEVELS}
    y_val = np.array([r["label"] == "ripe" for r in val_rows])

    # Frozen ImageNet features for the earlier heads and the expanded linear probe.
    feature_file = args.cache / "frozen_features.npy"
    if feature_file.exists() and np.load(feature_file, mmap_mode="r").shape[0] == len(crops):
        feats = np.load(feature_file)
    else:
        encoder = build_encoder(args.weights, args.device)
        feats = np.concatenate([encode_batch(encoder, crops[i:i + 512], args.device).numpy()
                                for i in range(0, len(crops), 512)]).astype(np.float64)
        np.save(feature_file, feats)
    x = torch.from_numpy(feats)

    scores = {}  # model -> {"val": array, level: array}
    thresholds = {}
    for name in ("clean", "aug"):
        head = torch.load(BASE / f"전이학습/ripeness/results/head_{name}.pt", map_location="cpu", weights_only=True)
        get = lambda rs: fit.score(head, x[[r["row"] for r in rs]]).numpy()
        scores[f"earlier_{name}"] = {"val": get(val_rows), **{lv: get(test_rows[lv]) for lv in LEVELS}}
        thresholds[f"earlier_{name}"] = float(head["threshold"])  # chosen on the earlier validation set
    train_rows_aug = train_clean + [r for r in rows if r["split"] == "train" and r["cover_percent"]]
    y_train_aug = torch.tensor([r["label"] == "ripe" for r in train_rows_aug], dtype=torch.float64)
    probe = fit.fit_head(x[[r["row"] for r in train_rows_aug]], y_train_aug)
    get = lambda rs: fit.score(probe, x[[r["row"] for r in rs]]).numpy()
    scores["expanded_probe_aug"] = {"val": get(val_rows), **{lv: get(test_rows[lv]) for lv in LEVELS}}

    runs = []
    for seed in args.seeds:
        log(f"Fine-tuning seed {seed}")
        model, info = finetune(crops, train_items, val_rows, args.weights, args.device, seed, args.epochs, log)
        name = f"finetune_seed{seed}"
        scores[name] = {"val": predict(model, crops, val_rows, args.device),
                        **{lv: predict(model, crops, test_rows[lv], args.device) for lv in LEVELS}}
        runs.append((info, name, model.state_dict()))

    # Selection on validation only.
    keep = max(runs, key=lambda item: item[0]["best_val_ap"])
    torch.save(dict(state_dict=keep[2], arch="resnet18, fc=Linear(512,1), layer3-4 fine-tuned",
                    input="tight box, longest side 160 px, mean-colour square, ImageNet normalisation",
                    imagenet_weights_sha256=WEIGHTS_SHA256), args.model_out / "ripeness_finetuned.pt")

    report = dict(settings=dict(epochs=args.epochs, seeds=args.seeds, precision_target=args.precision_target,
                                recall_floor=args.recall_floor, device=args.device, torch=torch.__version__,
                                frozen_layers="conv1, bn1, layer1, layer2", trained_layers="layer3, layer4, fc",
                                optimizer="AdamW one-cycle, lr fc 1e-3 / layer3-4 3e-4, wd 1e-4, batch 64",
                                loss="BCE with pos_weight = unripe/ripe in training regions",
                                augmentation="clean or leaf crop (p=0.5), flips, 90-degree turns, brightness/contrast +-10%"),
                  data=json.loads((args.cache / "split_summary.json").read_text(encoding="utf-8")),
                  finetune_runs=[r[0] for r in runs], kept_finetune=keep[1], models={})
    old_mask = np.array([r["in_old_test"] == "1" for r in test_rows[0]])
    for name, s in scores.items():
        t_bal = thresholds[name] if name in thresholds else balanced_threshold(y_val, s["val"])
        t_ripe, t_unripe = three_way_thresholds(y_val, s["val"], args.precision_target, args.recall_floor)
        th = (t_bal, t_ripe, t_unripe)
        entry = dict(thresholds=dict(balanced=t_bal, ripe=t_ripe, unripe=t_unripe),
                     validation=summarize(y_val, s["val"], th), test={}, earlier_test_202={})
        for lv in LEVELS:
            y = np.array([r["label"] == "ripe" for r in test_rows[lv]])
            entry["test"][lv] = summarize(y, s[lv], th)
            entry["earlier_test_202"][lv] = summarize(y[old_mask], s[lv][old_mask], th)
        y0 = np.array([r["label"] == "ripe" for r in test_rows[0]])
        entry["test_bootstrap_by_apple"] = bootstrap_by_apple(y0, s[0], [r["apple_id"] for r in test_rows[0]])
        report["models"][name] = entry
        log(f"{name:20s} val AP {entry['validation']['ap']:.3f} | test AP {entry['test'][0]['ap']:.3f} "
            f"AUC {entry['test'][0]['auc']:.3f} | earlier-202 AP {entry['earlier_test_202'][0]['ap']:.3f}")

    with (args.out / "test_scores.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["filename", "region_index", "apple_id", "label", "cover_percent", "in_old_test", *scores]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for lv in LEVELS:
            for i, r in enumerate(test_rows[lv]):
                writer.writerow({**{k: r[k] for k in fields[:6]}, **{m: float(scores[m][lv][i]) for m in scores}})
    report["elapsed_s"] = time.time() - started
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "train_log.txt").write_text("\n".join(log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
