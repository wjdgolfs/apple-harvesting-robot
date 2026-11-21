"""Figure: coin-scaled diameter vs measured weight, and cross-validated weight estimates."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE = Path(__file__).resolve().parent
SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#a9a8a2"
BATCH = {"iPhone 11": ("가벼운 묶음 20개", "#2a78d6"), "iPhone 8": ("중간 묶음 20개", "#eb6834"),
         "LM-X430": ("무거운 묶음 20개", "#1baf7a")}


def main():
    font_manager.fontManager.addfont("C:/Windows/Fonts/malgun.ttf")
    font_manager.fontManager.addfont("C:/Windows/Fonts/malgunbd.ttf")
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 11})
    m = json.loads((HERE / "results/metrics.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((HERE / "results/apple_diameters.csv").open(encoding="utf-8-sig")))
    a, b = m["formula_all_60"]["power"]["a"], m["formula_all_60"]["power"]["b"]
    cv = m["cross_validation"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5.2))
    for axis in (ax, bx):
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.grid(color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
    d_line = np.linspace(57, 95, 100)
    ax.plot(d_line, a * d_line ** b, color=INK2, linewidth=1.5, linestyle=(0, (4, 3)))
    ax.text(94, a * 94 ** b - 22, f"무게 = {a:.4f} × 지름^{b:.2f}", ha="right", fontsize=10, color=INK2)
    for key, (label, color) in BATCH.items():
        pts = [r for r in rows if r["weight_batch"] == key]
        ax.scatter([float(r["diameter_mm_median"]) for r in pts], [float(r["weight_g"]) for r in pts], s=44,
                   color=color, edgecolor=SURFACE, linewidth=1.5, label=label, zorder=3)
        bx.scatter([float(r["weight_g"]) for r in pts], [float(r["cv_estimate_g"]) for r in pts], s=44,
                   color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_xlabel("동전으로 보정한 지름 (mm, 세 높이 중앙값)")
    ax.set_ylabel("실측 무게 (g)")
    ax.set_title("지름이 클수록 무겁다 (사과 60개)", loc="left", fontsize=12, pad=8)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    bx.plot([100, 330], [100, 330], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    s = cv["power_three_views"]
    bx.text(0.03, 0.97, f"평균 절대오차 {s['mae_g']:.1f} g ({s['mape_pct']:.1f}%)\n같은 묶음 안 순위 상관 {s['within_batch_rank_corr_mean']:.2f}\n"
            f"전체 평균만 답하면 {cv['global_mean']['mae_g']:.1f} g", transform=bx.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=GRID))
    bx.set_xlabel("실측 무게 (g)")
    bx.set_ylabel("교차검증 예측 무게 (g)")
    bx.set_title("학습에 안 쓴 사과의 무게 예측", loc="left", fontsize=12, pad=8)
    fig.suptitle("동전으로 잰 지름에서 무게를 추정한다: 카메라가 아니라 크기를 본다", x=0.02, ha="left", fontsize=14,
                 fontweight="bold")
    fig.text(0.02, 0.885, "IntegrALIMENTA 후지 60개, 모두 iPhone 8로 찍은 하트 무늬 실내 위에서 본 사진 3장. 5겹 교차검증 20회 반복.",
             fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures/지름_무게.png", dpi=160)


if __name__ == "__main__":
    main()
