"""Ripeness under lighting changes (presentation slide 11): brightness, contrast and white balance.

The 3,023 unoccluded test crops are changed in pixel space and scored again by the kept fine-tuned model
(three-way thresholds) and the earlier frozen-feature head (its own threshold). Training used only
brightness/contrast +-10%, so most conditions here are outside the training augmentation.
Reports per condition: AP, ripe recall/precision of the decisions, and the share of photos whose decision
changed from the original. Writes a case sheet of the same apples under every condition (AGENTS.md).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE / "전이학습"))
from imagenet_encoder import build_encoder, encode_batch  # noqa: E402
import locate  # noqa: E402

FONT = "C:/Windows/Fonts/malgun.ttf"
KO = {"ripe": "익음", "unripe": "미숙", "hold": "보류"}
COLOR = {"ripe": "#e34948", "unripe": "#0ca30c", "hold": "#8a8984"}
MEAN_RGB = np.array([124, 116, 104], dtype=np.float32)
CONDITIONS = {
    "원본": dict(gain=1.0),
    "어둡게 0.6배": dict(gain=0.6),
    "어둡게 0.8배": dict(gain=0.8),
    "밝게 1.25배": dict(gain=1.25),
    "밝게 1.5배": dict(gain=1.5),
    "대비 70%": dict(contrast=0.7),
    "따뜻한 색온도": dict(wb=(1.12, 1.0, 0.85)),
    "차가운 색온도": dict(wb=(0.88, 1.0, 1.15)),
}


def transform(crops, gain=1.0, contrast=1.0, wb=(1.0, 1.0, 1.0)):
    """Change only the apple crop; the mean-colour letterbox padding stays as in preprocessing."""
    x = crops.astype(np.float32)
    pad = np.all(np.abs(x - MEAN_RGB) < 0.5, axis=-1, keepdims=True)
    mean = x.reshape(len(x), -1, 3).mean(axis=1)[:, None, None, :]
    y = (mean + (x - mean) * contrast) * gain * np.array(wb, dtype=np.float32)
    y = np.where(pad, x, y)
    return np.clip(np.round(y), 0, 255).astype(np.uint8)


def ap(y, s):
    order = np.argsort(-s, kind="stable")
    hits = y[order]
    return float((np.cumsum(hits) / np.arange(1, len(hits) + 1) * hits).sum() / hits.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    metrics = json.loads((HERE / "results/metrics.json").read_text(encoding="utf-8"))
    kept = metrics["kept_finetune"]
    th = metrics["models"][kept]["thresholds"]
    rows = [r for r in csv.DictReader((HERE / "cache/index.csv").open(encoding="utf-8"))
            if r["split"] == "test" and r["cover_percent"] == "0"]
    crops = np.load(HERE / "cache/crops.npy", mmap_mode="r")[[int(r["row"]) for r in rows]]
    y = np.array([r["label"] == "ripe" for r in rows])

    ckpt = torch.load(HERE / "weights/ripeness_finetuned.pt", map_location="cpu", weights_only=True)
    fine = resnet18(weights=None)
    fine.fc = nn.Linear(512, 1)
    fine.load_state_dict(ckpt["state_dict"])
    fine = fine.to(args.device).eval()
    encoder = build_encoder(BASE / "전이학습/weights/resnet18-f37072fd.pth", args.device)
    head = torch.load(BASE / "전이학습/ripeness/results/head_aug.pt", map_location="cpu", weights_only=True)

    def score(model, batch):
        out = np.concatenate([encode_batch(model, batch[i:i + 256], args.device).numpy() for i in range(0, len(batch), 256)])
        return out.astype(np.float64)

    def fine_decide(s):
        return np.where(s >= th["ripe"], "ripe", np.where(s < th["unripe"], "unripe", "hold"))

    results, decisions, images = {}, {}, {}
    for name, params in CONDITIONS.items():
        batch = transform(np.asarray(crops), **params)
        images[name] = batch
        s_fine = score(fine, batch)[:, 0]
        feats = torch.from_numpy(score(encoder, batch))
        s_lin = (((feats - head["mean"]) / head["scale"]).mv(head["weight"]) + head["bias"]).numpy()
        d_fine, d_lin = fine_decide(s_fine), np.where(s_lin >= head["threshold"], "ripe", "unripe")
        decisions[name] = d_fine
        if name == "원본":
            base_fine, base_lin = d_fine, d_lin
        called = d_fine == "ripe"
        results[name] = dict(
            finetuned=dict(ap=ap(y, s_fine), ripe_calls=int(called.sum()),
                           ripe_call_precision=float(y[called].mean()) if called.any() else None,
                           ripe_found=float((called & y).sum() / y.sum()), hold_rate=float((d_fine == "hold").mean()),
                           changed_from_original=float((d_fine != base_fine).mean())),
            earlier_aug=dict(ap=ap(y, s_lin), ripe_calls=int((d_lin == "ripe").sum()),
                             ripe_call_precision=float(y[d_lin == "ripe"].mean()) if (d_lin == "ripe").any() else None,
                             ripe_found=float(((d_lin == "ripe") & y).sum() / y.sum()),
                             changed_from_original=float((d_lin != base_lin).mean())))
        f, e = results[name]["finetuned"], results[name]["earlier_aug"]
        print(f"{name:10} fine AP {f['ap']:.3f} prec {f['ripe_call_precision']} found {f['ripe_found']:.2f} hold {f['hold_rate']:.2f} "
              f"changed {f['changed_from_original']:.3f} | earlier AP {e['ap']:.3f} found {e['ripe_found']:.2f} changed {e['changed_from_original']:.3f}",
              flush=True)
    (HERE / "results/lighting.json").write_text(json.dumps(dict(model=kept, thresholds=th, photos=len(rows), ripe=int(y.sum()),
                                                             conditions=CONDITIONS, results=results), indent=2, ensure_ascii=False),
                                               encoding="utf-8")

    # Case sheet: same photos under every condition. Rows: ripe found, unripe called ripe somewhere, ripe missed somewhere.
    names = list(CONDITIONS)
    flips = np.array([[decisions[n][i] != decisions["원본"][i] for n in names] for i in range(len(rows))]).sum(1)
    def first(mask):
        idx = np.flatnonzero(mask)
        return int(idx[np.argmax(flips[idx])]) if len(idx) else None
    picks = [("익음 사진을 원본에서 익음으로", first(y & (decisions["원본"] == "ripe"))),
             ("미숙 사진이 조명에 따라 익음이 됨", first(~y & np.any([decisions[n] == "ripe" for n in names], axis=0))),
             ("익음 사진을 조명에 따라 놓침", first(y & np.any([decisions[n] == "unripe" for n in names], axis=0))),
             ("미숙 사진을 미숙으로", first(~y & np.all([decisions[n] == "unripe" for n in names], axis=0)))]
    picks = [(t, i) for t, i in picks if i is not None]
    tile = 118
    title_f, body_f, small_f = (ImageFont.truetype(FONT, s) for s in (24, 16, 13))
    sheet = Image.new("RGB", (24 + (len(names) + 1) * (tile + 10) + 10, 134 + len(picks) * (tile + 84) + 30), "#fcfcfb")
    d = ImageDraw.Draw(sheet)
    d.text((20, 16), "같은 사진의 조명만 바꿨을 때 익음도 판정 (미세조정 모델)", font=title_f, fill="#0b0b0b")
    d.text((20, 52), "테두리 색 = 예측(빨강 익음, 초록 미숙, 회색 보류). 평가 분할 사진.", font=small_f, fill="#52514e")
    for j, n in enumerate(names):
        d.text((20 + j * (tile + 10), 84), n, font=small_f, fill="#0b0b0b")
    d.text((20 + len(names) * (tile + 10), 84), "원본 속 위치", font=small_f, fill="#0b0b0b")
    yy = 118
    for title, i in picks:
        r = rows[i]
        d.text((20, yy), f"{title} · 정답 {KO[r['label']]} · {r['filename'].replace('.png', '')} · 사과 {r['apple_id']}", font=body_f, fill="#0b0b0b")
        for j, n in enumerate(names):
            pred = decisions[n][i]
            framed = Image.new("RGB", (tile + 6, tile + 6), COLOR[pred])
            framed.paste(Image.fromarray(images[n][i]).resize((tile, tile)), (3, 3))
            x = 20 + j * (tile + 10)
            sheet.paste(framed, (x, yy + 24))
            d.text((x, yy + 32 + tile), KO[pred], font=small_f, fill="#0b0b0b")
        photo, box, _ = locate.region(r["filename"], r["region_index"])
        sheet.paste(locate.locator(photo, box, tile + 6), (20 + len(names) * (tile + 10), yy + 24))
        yy += tile + 84
    d.text((20, sheet.height - 26), "이미지 Gené-Mola et al., AmodalAppleSize_RGB-D (CC BY 4.0) · 라벨 Zhu et al., Fuji-Ripeness-Size · 밝기·색 변환",
           font=small_f, fill="#52514e")
    sheet.save(HERE / "figures/조명별_익음도_사례.png", optimize=True)


if __name__ == "__main__":
    main()
