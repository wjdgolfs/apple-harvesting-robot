"""Chart: per-view ripeness AP and precision under added leaf occlusion, earlier head vs expanded data vs fine-tuning."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#a9a8a2"
LEVELS = ("0", "15", "30", "50")


def main():
    font_manager.fontManager.addfont("C:/Windows/Fonts/malgun.ttf")
    font_manager.fontManager.addfont("C:/Windows/Fonts/malgunbd.ttf")
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 11})
    m = json.loads((HERE / "results/metrics.json").read_text(encoding="utf-8"))["models"]
    seeds = [k for k in m if k.startswith("finetune")]
    series = [("기존 분류층 (가림증강)", "#a9a8a2", ["earlier_aug"]),
              ("같은 방식 + 데이터 확장", "#2a78d6", ["expanded_probe_aug"]),
              ("미세조정 (시드 3개 평균, 범위)", "#eb6834", seeds)]
    panels = [("사진 한 장 단위 평균 정밀도(AP)", lambda e, lv: e["test"][lv]["ap"]),
              ("균형 문턱에서 '익음' 판정의 정밀도", lambda e, lv: e["test"][lv]["at_balanced_threshold"]["precision"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    xs = [int(v) for v in LEVELS]
    for ax, (title, get) in zip(axes, panels):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for label, color, names in series:
            vals = [[get(m[n], lv) for n in names] for lv in LEVELS]
            mean = [sum(v) / len(v) for v in vals]
            if len(names) > 1:
                ax.fill_between(xs, [min(v) for v in vals], [max(v) for v in vals], color=color, alpha=0.18, linewidth=0)
            ax.plot(xs, mean, color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=label)
            ax.annotate(f"{mean[-1]:.2f}", (xs[-1], mean[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9.5, color=INK)
        ax.set_title(title, loc="left", fontsize=12, pad=8)
        ax.set_xticks(xs)
        ax.set_xlabel("원래 보이던 사과 면적 중 합성 잎으로 가린 비율 (%)")
        ax.set_xlim(-3, 58)
        ax.set_ylim(0, 1)
    axes[0].legend(frameon=False, fontsize=9.5, loc="lower left")
    fig.suptitle("미세조정은 가림에 강해졌지만, 가림 없는 사진의 구분력은 거의 그대로다", x=0.02, ha="left",
                 fontsize=14, fontweight="bold")
    fig.text(0.02, 0.885, "평가 사과 115개의 사진 3,023장(익음 460). 학습·검증과 사과 ID·사진이 겹치지 않음. 문턱은 검증 자료에서만 정함.",
             fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures/가림별_익음도_비교.png", dpi=160)


if __name__ == "__main__":
    main()
