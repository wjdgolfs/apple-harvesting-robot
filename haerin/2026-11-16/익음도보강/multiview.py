"""Decide ripeness per apple from several photos instead of one.

The same apples are used for every k: those with at least 20 test photos, so two draws of up to 10 photos never
share a photo. In each of 1,000 draws the photos of an apple are shuffled once; the first k and the next k form
the two looks. The apple score is the mean of its k photo scores; thresholds are applied unchanged. 'ripe share'
(fraction of the apple's photos labelled ripe) is the apple-level reference. Run baseline.py first.

Reports per k: rank correlation with ripe share, AP for 'majority of photos ripe', agreement of the two looks
and hold rate. Also writes a case sheet with the location in the original photo, per-photo labels and
predictions (AGENTS.md).
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import spearmanr

import locate

HERE = Path(__file__).resolve().parent
KS = (1, 3, 5, 10)
DRAWS = 1000
MIN_PHOTOS = 2 * max(KS)
SEED = 20261122
FONT = "C:/Windows/Fonts/malgun.ttf"
KO = {"ripe": "익음", "unripe": "미숙", "hold": "보류"}
COLOR = {"ripe": "#e34948", "unripe": "#0ca30c", "hold": "#8a8984"}


def ap(y, s):
    order = np.argsort(-s, kind="stable")
    hits = y[order]
    return float((np.cumsum(hits) / np.arange(1, len(hits) + 1) * hits).sum() / hits.sum())


def main():
    metrics = json.loads((HERE / "results/metrics.json").read_text(encoding="utf-8"))
    model = metrics["kept_finetune"]
    th = metrics["models"][model]["thresholds"]
    decide = lambda s: "ripe" if s >= th["ripe"] else "unripe" if s < th["unripe"] else "hold"
    rows = [r for r in csv.DictReader((HERE / "results/test_scores.csv").open(encoding="utf-8-sig")) if r["cover_percent"] == "0"]
    apples = defaultdict(list)
    for r in rows:
        apples[r["apple_id"]].append(r)
    ids = sorted(apples)
    share = np.array([np.mean([v["label"] == "ripe" for v in apples[a]]) for a in ids])
    with (HERE / "results/baseline_scores.csv").open(encoding="utf-8-sig") as stream:
        red = {(r["filename"], r["region_index"]): r["red_share"] for r in csv.DictReader(stream) if r["cover_percent"] == "0"}
    for r in rows:
        r["red_share"] = red[r["filename"], r["region_index"]]
    names = (model, "expanded_probe_aug", "earlier_aug", "red_share")
    cohort = [i for i, a in enumerate(ids) if len(apples[a]) >= MIN_PHOTOS]
    values = [np.array([[float(v[n]) for n in names] for v in apples[ids[i]]]) for i in cohort]
    target, y = share[cohort], share[cohort] >= 0.5
    rng = np.random.default_rng(SEED)
    found = {(n, k): defaultdict(list) for n in names for k in KS}
    for _ in range(DRAWS):
        shuffled = [v[rng.permutation(len(v))] for v in values]
        for k in KS:
            first = np.array([v[:k].mean(0) for v in shuffled])
            second = np.array([v[k:2 * k].mean(0) for v in shuffled])
            for j, name in enumerate(names):
                cell = found[name, k]
                cell["rank_corr_with_ripe_share"].append(spearmanr(first[:, j], target).statistic)
                cell["ap_majority_ripe"].append(ap(y, first[:, j]))
                if name == model:
                    a, b = ([decide(s) for s in look[:, j]] for look in (first, second))
                    cell["decision_same_in_two_draws"].append(np.mean([p == q for p, q in zip(a, b)]))
                    cell["hold_rate"].append(np.mean([p == "hold" for p in a]))
    report = dict(model=model, thresholds=th, test_apples=len(ids), photos=len(rows), draws=DRAWS, seed=SEED,
                  cohort=dict(min_photos=MIN_PHOTOS, apples=len(cohort), majority_ripe_apples=int(y.sum())),
                  by_k={n: {k: {key: float(np.mean(v)) for key, v in found[n, k].items()} for k in KS} for n in names})
    (HERE / "results/multiview.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------- chart
    font_manager.fontManager.addfont(FONT)
    font_manager.fontManager.addfont("C:/Windows/Fonts/malgunbd.ttf")
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "font.size": 11})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    colours = [(model, "미세조정", "#eb6834"), ("expanded_probe_aug", "데이터 확장", "#2a78d6"),
               ("earlier_aug", "기존 분류층", "#a9a8a2"), ("red_share", "빨간 픽셀 비율(학습 없음)", "#1baf7a")]
    panels = [("사과 점수와 '익음 사진 비율'의 순위 상관",
               [(n, "rank_corr_with_ripe_share", label, c) for n, label, c in colours]),
              ("다수 사진이 익음인 사과 찾기 (AP)", [(n, "ap_majority_ripe", label, c) for n, label, c in colours]),
              ("미세조정 3단 판정의 안정성과 보류",
               [(model, "decision_same_in_two_draws", "겹치지 않는 사진끼리 같은 판정", "#2a78d6"),
                (model, "hold_rate", "보류 비율", "#8a8984")])]
    for ax, (title, lines) in zip(axes, panels):
        ends = []
        for name, key, label, color in lines:
            ys = [report["by_k"][name][k][key] for k in KS]
            ax.plot(KS, ys, marker="o", color=color, linewidth=2, label=label)
            ends.append(ys[-1])
        last = -1.0
        for value in sorted(ends):
            last = max(value, last + 0.055)
            ax.annotate(f"{value:.2f}", (KS[-1], last), xytext=(7, 0), textcoords="offset points", va="center", fontsize=9.5)
        ax.legend(frameon=False, loc="center right" if title.startswith("미세조정 3단") else "lower right", fontsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="#e4e3df")
        ax.set_xticks(KS)
        ax.set_xlabel("사과 하나당 모은 사진 수 k")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_xlim(0.5, 11.6)
    fig.suptitle("여러 장을 모으면 판정이 안정되고 익음 사과를 찾는 AP가 오른다", x=0.02, ha="left", fontsize=13.5, fontweight="bold")
    fig.text(0.02, 0.88, f"사진 {MIN_PHOTOS}장 이상인 사과 {len(cohort)}개(다수 익음 {int(y.sum())}개)를 모든 k에 똑같이 사용. 양성 사과가 적어 AP 차이는 불확실.",
             fontsize=9.5, color="#52514e")
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures/여러사진_모아판단.png", dpi=160)

    # ---------- case sheet (per-photo label vs prediction, then the apple-level decision from all photos)
    crops = np.load(HERE / "cache/crops.npy", mmap_mode="r")
    index = {(r["filename"], r["region_index"]): int(r["row"]) for r in csv.DictReader((HERE / "cache/index.csv").open(encoding="utf-8"))
             if r["split"] == "test" and r["cover_percent"] == "0"}
    apple_score = {a: np.mean([float(v[model]) for v in apples[a]]) for a in ids}
    with_many = [a for a in ids if len(apples[a]) >= 6]
    pick = lambda pool: max(pool, key=lambda a: len(apples[a])) if pool else None
    cases = [("익음 사과를 익음으로 (다수 사진이 익음 라벨)", pick([a for a in with_many if share[ids.index(a)] >= 0.5 and decide(apple_score[a]) == "ripe"])),
             ("익음 사진이 섞인 사과를 보류", pick([a for a in with_many if 0 < share[ids.index(a)] < 0.5 and decide(apple_score[a]) == "hold"])),
             ("미숙 사과(익음 라벨 0장)를 익음으로 잘못 판정", pick([a for a in with_many if share[ids.index(a)] == 0 and decide(apple_score[a]) == "ripe"])),
             ("익음 라벨이 있는 사과를 미숙으로 놓침", pick([a for a in with_many if share[ids.index(a)] > 0 and decide(apple_score[a]) == "unripe"])),
             ("미숙 사과를 미숙으로", pick([a for a in with_many if share[ids.index(a)] == 0 and decide(apple_score[a]) == "unripe"]))]
    if not cases[2][1]:
        # No apple without a ripe label is called ripe from all its photos: show the photo-level false calls instead.
        wrong = {a: sum(decide(float(v[model])) == "ripe" for v in apples[a]) for a in with_many if share[ids.index(a)] == 0}
        a = max(wrong, key=lambda k: (wrong[k], len(apples[k]))) if wrong else None
        if a and wrong[a]:
            cases[2] = (f"사진 단위 오탐 {wrong[a]}장: 미숙 라벨을 익음으로", a)
        else:
            cases[2] = ("", None)
    cases = [(t, a) for t, a in cases if a]
    tile, n_show = 120, 8
    title_f, body_f, small_f = (ImageFont.truetype(FONT, s) for s in (26, 17, 13))
    sheet = Image.new("RGB", (40 + n_show * (tile + 8) + 250, 90 + len(cases) * (2 * tile + 104) + 40), "#fcfcfb")
    d = ImageDraw.Draw(sheet)
    d.text((24, 18), "같은 사과의 여러 사진: 사진별 라벨·예측과 사과 단위 판정", font=title_f, fill="#0b0b0b")
    d.text((24, 56), "테두리 색 = 사진별 예측(빨강 익음, 초록 미숙, 회색 보류). 아래 글자 = 연구진 라벨 / 예측. 작은 사진 = 원본 속 위치(하늘색 상자). 평가 분할, 미세조정 모델.",
           font=small_f, fill="#52514e")
    y = 90
    for title, a in cases:
        views = sorted(apples[a], key=lambda v: v["filename"])
        shown = [views[int(i)] for i in np.linspace(0, len(views) - 1, min(n_show, len(views)))]
        false_calls = [v for v in views if v["label"] == "unripe" and decide(float(v[model])) == "ripe"]
        if title.startswith("사진 단위 오탐"):
            shown = (false_calls + [v for v in shown if v not in false_calls])[:n_show]
        d.text((24, y), f"{title} · 사과 {a} · 사진 {len(views)}장 중 익음 라벨 {share[ids.index(a)]:.0%}", font=body_f, fill="#0b0b0b")
        for j, v in enumerate(shown):
            x = 24 + j * (tile + 8)
            img = Image.fromarray(np.asarray(crops[index[(v["filename"], v["region_index"])]])).resize((tile, tile))
            pred = decide(float(v[model]))
            framed = Image.new("RGB", (tile + 6, tile + 6), COLOR[pred])
            framed.paste(img, (3, 3))
            sheet.paste(framed, (x, y + 26))
            d.text((x, y + 34 + tile), f"{KO[v['label']]} / {KO[pred]}", font=small_f, fill="#0b0b0b")
            d.text((x, y + 50 + tile), v["filename"].replace(".png", ""), font=small_f, fill="#8a8984")
            photo, box, _ = locate.region(v["filename"], v["region_index"])
            sheet.paste(locate.locator(photo, box, tile), (x + 3, y + 72 + tile))
        final = decide(apple_score[a])
        d.text((24 + n_show * (tile + 8) + 10, y + 60), f"사진 {len(views)}장 평균\n→ {KO[final]}", font=body_f, fill=COLOR[final])
        y += 2 * tile + 104
    d.text((24, sheet.height - 28), "이미지 Gené-Mola et al., AmodalAppleSize_RGB-D (CC BY 4.0) · 라벨 Zhu et al., Fuji-Ripeness-Size · 잘라서 확대",
           font=small_f, fill="#52514e")
    sheet.save(HERE / "figures/여러사진_사례.png", optimize=True)
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
