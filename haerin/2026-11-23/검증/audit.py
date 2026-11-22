"""Recompute, from per-item raw files, every number planned for the weekly report. Reads only; writes to scratch."""
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import spearmanr

L = lambda p: list(csv.DictReader(open(p, encoding="utf-8-sig")))
J = lambda p: json.load(open(p, encoding="utf-8"))
O = {}


def show(k, v):
    O[k] = v
    print(f"## {k}\n{json.dumps(v, ensure_ascii=False, default=float)}")


def ap(y, s):
    h = y[np.argsort(-s, kind="stable")]
    return float((np.cumsum(h) / np.arange(1, len(h) + 1) * h).sum() / h.sum())


def pr(y, s, q):
    h = y[np.argsort(-s, kind="stable")]
    t = np.cumsum(h)
    i = np.searchsorted(t, math.ceil(q * y.sum()))
    return float(t[i] / (i + 1))


frame = lambda f: re.match(r"(_MG_\d+)_", f)[1]
P73 = lambda a, b: pr(a, b, .73)
R3 = lambda v: round(float(v), 3)
B = "익음도보강/"
M = ["earlier_aug", "expanded_probe_aug", "finetune_seed0", "finetune_seed1", "finetune_seed2"]

# ---------- detector training
R = [{k.strip(): v.strip() for k, v in r.items()} for r in L("../../output/pdf/MinneApple_기본학습_20261116/results.csv")]
F = lambda r, k: float(r["metrics/" + k])
fit = [.1 * (F(r, "mAP50(B)") + F(r, "mAP50(M)")) + .9 * (F(r, "mAP50-95(B)") + F(r, "mAP50-95(M)")) for r in R]
show("yolo_train", dict(rows=len(R), best=int(np.argmax(fit)) + 1,
                        b50=[F(r, "mAP50(B)") for r in R], m50=[F(r, "mAP50(M)") for r in R]))

# ---------- ripeness
rows = L(B + "results/test_scores.csv")
red = {(r["filename"], r["region_index"]): float(r["red_share"]) for r in L(B + "results/red_share.csv")}


def arrays(level, with_red=False):
    rr = [r for r in rows if r["cover_percent"] == level]
    s = {m: np.array([float(r[m]) for r in rr]) for m in M}
    if with_red:
        s["red_share"] = np.array([red[r["filename"], r["region_index"]] for r in rr])
    return rr, np.array([r["label"] == "ripe" for r in rr]), s, np.array([r["apple_id"] for r in rr])


def boot(y, s, ids, a, b, fn):
    g = [np.flatnonzero(ids == u) for u in np.unique(ids)]
    rng = np.random.default_rng(1)
    out = []
    for _ in range(2000):
        i = np.concatenate([g[j] for j in rng.integers(0, len(g), len(g))])
        if y[i].any():
            out.append(fn(y[i], s[a][i]) - fn(y[i], s[b][i]))
    return [R3(fn(y, s[a]) - fn(y, s[b])), *map(R3, np.percentile(out, [2.5, 97.5]))]


lv0, y, s, ids = arrays("0", True)
_, y5, s5, ids5 = arrays("50")
show("ripe_clean", {m: [R3(ap(y, v)), R3(pr(y, v, .73)), R3(pr(y, v, .9))] for m, v in s.items()})
show("ripe_leaf50", {m: [R3(ap(y5, v)), R3(pr(y5, v, .73))] for m, v in s5.items()})
pairs = (("expanded_probe_aug", "earlier_aug"), ("finetune_seed1", "expanded_probe_aug"),
         ("finetune_seed1", "earlier_aug"), ("expanded_probe_aug", "red_share"), ("earlier_aug", "red_share"))
show("ripe_diff", {f"{a}-{b}": dict(ap=boot(y, s, ids, a, b, ap), p73=boot(y, s, ids, a, b, P73)) for a, b in pairs})
show("ripe_diff50", {f"{a}-{b}": boot(y5, s5, ids5, a, b, ap) for a, b in pairs[:3]})

used = {frame(r["filename"]) for r in L(B + "cache/index.csv") if r["split"] != "test"}
k = np.array([frame(r["filename"]) not in used for r in lv0])
show("ripe_unseen_frames", dict(
    n=int(k.sum()), ripe=int(y[k].sum()), apples=len(set(ids[k])),
    m={m: [R3(ap(y[k], v[k])), R3(pr(y[k], v[k], .73))] for m, v in s.items()} if y[k].any() else None))

th = J(B + "results/metrics.json")["models"]["finetune_seed1"]["thresholds"]
v1 = s["finetune_seed1"]
call = v1 >= th["ripe"]
show("three_way", dict(prec=R3(y[call].mean()), found=R3((call & y).sum() / y.sum()),
                       hold=R3(((v1 < th["ripe"]) & (v1 >= th["unripe"])).mean())))

years, labels, same = defaultdict(set), defaultdict(set), defaultdict(list)
for r in L("integration/results/linked_instances.csv"):
    years[r["apple_id"]].add("2018" if int(frame(r["filename"])[4:]) < 5000 else "2020")
    labels[r["apple_id"]].add(r["label"])
    same[r["apple_id"], frame(r["filename"])].append(r["label"])
multi = [v for v in same.values() if len(v) > 1]
conflict = [v for v in multi if len(set(v)) > 1]
with_ripe = [v for v in multi if "ripe" in v]
show("labels", dict(ids=len(labels), ids_in_both_years=sum(len(v) > 1 for v in years.values()),
                    mixed_ids=sum(len(v) > 1 for v in labels.values()), same_photo_groups=len(multi),
                    same_photo_conflict=len(conflict), same_photo_groups_with_ripe=len(with_ripe)))

g = defaultdict(list)
for i, a in enumerate(ids):
    g[a].append(i)
A = sorted(a for a in g if len(g[a]) >= 20)
Y = np.array([y[g[a]].mean() >= .5 for a in A])
rng = np.random.default_rng(20261122)
D3 = lambda x: 2 if x >= th["ripe"] else 0 if x < th["unripe"] else 1
out = {}
for kk in (1, 3, 5, 10):
    o = defaultdict(list)
    for _ in range(1000):
        p = [rng.permutation(g[a]) for a in A]
        for m in ("earlier_aug", "expanded_probe_aug", "finetune_seed1", "red_share"):
            o[m].append(ap(Y, np.array([s[m][q[:kk]].mean() for q in p])))
        o["agree"].append(np.mean([D3(v1[q[:kk]].mean()) == D3(v1[q[kk:2 * kk]].mean()) for q in p]))
    out[kk] = {m: R3(np.mean(x)) for m, x in o.items()}
show("multi", dict(apples=len(A), positives=int(Y.sum()), k=out))
show("light", {k: [R3(x["finetuned"]["ap"]), R3(x["finetuned"]["changed_from_original"])]
               for k, x in J(B + "results/lighting.json")["results"].items()})

# ---------- diameter
m = [r for r in L("지름추정/results/instance_measurements.csv") if float(r["eligible"])]
for r in m:
    for key in r:
        if key not in ("split", "filename", "apple_id"):
            r[key] = float(r[key])
    z, f = r["depth_eroded_median_m"], r["focal_px"]
    side = max(r["vis_width_px"], r["vis_height_px"])
    r["ok"] = r["depth_valid_fraction"] >= .5 and not math.isnan(z) and not math.isnan(r["gt_diameter_mm"])
    r["fill"] = r["vis_px"] / (math.pi / 4 * side * side)
    r["e"] = 1000 * side * z / f
    r["eh"] = 1000 * r["vis_width_px"] * z / f
    r["ea"] = 1000 * r["amodal_width_px"] * z / f
mae = lambda R_, key="e": float(np.mean([abs(r[key] - r["gt_diameter_mm"]) for r in R_]))
te = [r for r in m if r["split"] == "test" and r["ok"]]
bins = {}
for lo, hi in ((0, .15), (.15, .3), (.3, .5), (.5, 1.01)):
    R_ = [r for r in te if lo <= r["natural_occlusion"] < hi]
    bins[f"{lo}-{hi}"] = [len(R_), R3(mae(R_)), R3(mae(R_, "eh"))]
show("diam_test", dict(n=len(te), longer=R3(mae(te)), horizontal=R3(mae(te, "eh")), amodal_reference=R3(mae(te, "ea")), bins=bins))

pool = [r for r in m if r["ok"]]
fold = {a: int(hashlib.sha256(a.encode()).hexdigest(), 16) % 5 for a in {r["apple_id"] for r in m}}
res, ths, scales = [], [], []
for f in range(5):
    tr = [r for r in pool if fold[r["apple_id"]] != f]
    t = next(t for t in (.5, .55, .6, .65, .7, .75, .8, .85, .9) if mae([r for r in tr if r["fill"] >= t]) <= 5)
    K = [r for r in tr if r["fill"] >= t]
    c = sum(r["e"] * r["gt_diameter_mm"] for r in K) / sum(r["e"] ** 2 for r in K)
    ths.append(t)
    scales.append(R3(c))
    res += [(r["e"], r["e"] * c, r["gt_diameter_mm"], r["apple_id"]) for r in pool
            if fold[r["apple_id"]] == f and r["fill"] >= t]
q = np.array([x[:3] for x in res])
show("diam_grouped_cv", dict(
    thresholds=ths, scales=scales, used=len(res), of=len(m), apples=len({x[3] for x in res}), apples_all=len(fold),
    share=R3(len(res) / len(m)), mae_raw=R3(abs(q[:, 0] - q[:, 2]).mean()), mae_cal=R3(abs(q[:, 1] - q[:, 2]).mean()),
    bias_raw=R3((q[:, 0] - q[:, 2]).mean()), bias_cal=R3((q[:, 1] - q[:, 2]).mean())))

# ---------- weight
S = L("무게추정/results/size_measurements.csv")
VIEWS = ("spm_int_al_ver", "spm_int_me_ver", "spm_int_ce_ver")
G = sorted({r["group"] for r in S})
n = len(G)
w, b = np.zeros(n), np.empty(n, object)
D, Z, PX = (np.full((3, n), np.nan) for _ in range(3))
for r in S:
    i, j = G.index(r["group"]), VIEWS.index(r["view"])
    w[i], b[i] = float(r["weight_g"]), r["weight_batch"]
    if r.get("d_eq_mm") and 55 <= float(r["d_eq_mm"]) <= 130:
        d0 = float(r["d_eq_mm"])
        z0 = float(r["focal_px"]) * 23.25 / (2 * float(r["coin_r_px"]))
        D[j, i], Z[j, i], PX[j, i] = d0 / (1 + d0 / (2 * z0)), z0, float(r["apple_eq_px"])
d = np.nanmedian(D, 0)
batches = sorted(set(b))


def cv(x, kind):
    err = []
    for rep in range(20):
        rng_ = np.random.default_rng([20261116, rep])
        fd = np.empty(n, int)
        for u in batches:
            fd[rng_.permutation(np.flatnonzero(b == u))] = np.arange(20) % 5
        for f in range(5):
            T = fd == f
            A_ = ~T
            if kind == "mean":
                p = w[A_].mean()
            elif kind == "batch":
                p = np.array([w[A_ & (b == u)].mean() for u in b[T]])
            elif kind == "cube":
                p = np.sum(w[A_] * x[A_] ** 3) / np.sum(x[A_] ** 6) * x[T] ** 3
            else:
                slope, inter = np.polyfit(np.log(x[A_]), np.log(w[A_]), 1)
                p = np.exp(inter) * x[T] ** slope
            err += list(abs(p - w[T]))
    return R3(np.mean(err))


show("weight_cv", dict(mean=cv(d, "mean"), batch=cv(d, "batch"), power=cv(d, "power"), cube=cv(d, "cube"),
                       one_view=cv(D[1], "power"), camera_height_only=cv(np.nanmedian(Z, 0), "power"),
                       pixels_only=cv(np.nanmedian(PX, 0), "power")))
o = {}
for u in batches:
    T, A_ = b == u, b != u
    slope, inter = np.polyfit(np.log(d[A_]), np.log(w[A_]), 1)
    cube = np.sum(w[A_] * d[A_] ** 3) / np.sum(d[A_] ** 6)
    rho = spearmanr(d[T], w[T])
    o[u] = dict(power=R3(abs(np.exp(inter) * d[T] ** slope - w[T]).mean()), cube=R3(abs(cube * d[T] ** 3 - w[T]).mean()),
                mean_of_others=R3(abs(w[A_].mean() - w[T]).mean()), rho=R3(rho.statistic), p=float(f"{rho.pvalue:.2g}"))
show("weight_leave_batch_out", o)
slope, inter = np.polyfit(np.log(d), np.log(w), 1)
show("weight_formula", [math.exp(inter), slope, float(d.min()), float(d.max())])
v = J("무게검증/results/metrics.json")["part_a_original_view"]["cross_validation"]
show("confound", {key: R3(x["mae_g"]) for key, x in v.items()})
show("single_cam", sum(r["single_camera"] == "True" for r in L("무게검증/results/camera_by_view.csv")))

json.dump(O, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, default=float, indent=1)
