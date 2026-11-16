"""Draw the camera-confound check figures from results/ (no data access needed)."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#a9a8a2"
# Categorical slots 1-3 of the validated reference palette (all-pairs safe for scatter).
BATCH_COLOR = {"iPhone 11": "#2a78d6", "iPhone 8": "#eb6834", "LM-X430": "#1baf7a"}
BATCH_LABEL = {"iPhone 11": "가벼운 묶음 20개 (원래 조건 카메라: iPhone 11)",
               "iPhone 8": "중간 묶음 20개 (iPhone 8)",
               "LM-X430": "무거운 묶음 20개 (LM-X430)"}
HIGHLIGHT, SECOND = "#2a78d6", "#eb6834"


def setup():
    font_manager.fontManager.addfont("C:/Windows/Fonts/malgun.ttf")
    font_manager.fontManager.addfont("C:/Windows/Fonts/malgunbd.ttf")
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 11})


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def quiet(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def scatter_pair(rows, cv, setting, panels, title, subtitle, path):
    """Two panels of measured vs predicted weight, one per input variant."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.1), sharex=True, sharey=True)
    batch_mean = {b: sum(float(r["weight_g"]) for r in rows if r["setting"] == setting and r["model"] == "global_mean"
                         and r["weight_batch"] == b) / 20 for b in BATCH_COLOR}
    for ax, (model, panel_title) in zip(axes, panels):
        quiet(ax, "both")
        ax.plot([95, 335], [95, 335], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
        ax.text(328, 318, "정확히 맞힌 선", color=INK2, fontsize=9, ha="right", va="top")
        for batch, color in BATCH_COLOR.items():
            pts = [r for r in rows if r["setting"] == setting and r["model"] == model and r["weight_batch"] == batch]
            xs = [float(r["weight_g"]) for r in pts]
            ax.plot([min(xs) - 3, max(xs) + 3], [batch_mean[batch]] * 2, color=color, linewidth=2.2, alpha=0.4, zorder=2)
            ax.scatter(xs, [float(r["estimate_g"]) for r in pts], s=46, color=color, edgecolor=SURFACE,
                       linewidth=1.6, zorder=3, label=BATCH_LABEL[batch])
        s = cv[model]
        ax.set_title(panel_title, fontsize=12, color=INK, loc="left", pad=10)
        ax.text(0.03, 0.97, f"평균 절대오차 {s['mae_g']:.1f} g\n같은 묶음 안 순위 상관 {s['spearman_within_batch_mean']:.2f}",
                transform=ax.transAxes, va="top", fontsize=10.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=GRID))
        ax.set_xlabel("실측 무게 (g)")
        ax.set_xlim(95, 335)
        ax.set_ylim(95, 335)
    axes[0].set_ylabel("ResNet18 예측 무게 (g)")
    axes[1].text(0.97, 0.03, "옅은 가로선 = 그 묶음의 평균 무게", transform=axes[1].transAxes, ha="right",
                 fontsize=9, color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, x=0.06, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.06, 0.905, subtitle, fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def bars(metrics, out):
    a = metrics["part_a_original_view"]["cross_validation"]
    b = metrics["part_b_single_camera_views"]["average_over_all_single_camera_views"]["cross_validation"]
    batch_mean = a["camera_mean"]
    panels = [("원래 조건 · 무게 묶음마다 카메라가 다름", [
                  ("전체 평균만 답함", a["global_mean"], "base"),
                  ("사과 면적 회귀", a["area_pooled"], "base"),
                  ("ResNet18 · 원본 사진", a["resnet_full"], "full"),
                  ("ResNet18 · 사과를 지운 사진", a["resnet_fruit_erased"], "background"),
                  ("ResNet18 · 네 모서리 배경만", a["resnet_corners_only"], "background")]),
              ("카메라 1대 조건 54개 · 사과마다 예측 평균", [
                  ("전체 평균만 답함", b["global_mean"], "base"),
                  ("사과 면적 회귀", b["area_pooled"], "base"),
                  ("ResNet18 · 원본 사진", b["resnet_full"], "full"),
                  ("ResNet18 · 네 모서리 배경만", b["resnet_corners_only"], "background")])]
    color = {"base": MUTED, "full": HIGHLIGHT, "background": SECOND}
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharex=True)
    top = max(s["mae_g"] for _, items in panels for _, s, _ in items)
    for ax, (title, items) in zip(axes, panels):
        quiet(ax, "x")
        labels = [label for label, _, _ in items][::-1]
        values = [s["mae_g"] for _, s, _ in items][::-1]
        ax.barh(labels, values, color=[color[k] for _, _, k in items][::-1], height=0.56, edgecolor=SURFACE, linewidth=2)
        for i, v in enumerate(values):
            ax.text(v + top * 0.015, i, f"{v:.1f} g", va="center", fontsize=10, color=INK)
        ax.axvline(batch_mean["mae_g"], color=INK2, linewidth=1.2, linestyle=(0, (3, 3)))
        ax.set_ylim(-1.15, len(items) - 0.5)
        ax.text(batch_mean["mae_g"] + top * 0.012, -0.85, f"무게 묶음 평균만 답해도 {batch_mean['mae_g']:.1f} g",
                fontsize=9, color=INK2, va="center")
        ax.set_title(title, loc="left", fontsize=12, pad=8)
        ax.set_xlim(0, top * 1.18)
        ax.set_xlabel("평균 절대오차 (g) · 낮을수록 좋음")
        ax.tick_params(axis="y", length=0)
    fig.suptitle("사과 정보 없이 배경만으로도 오차 15~19 g까지 내려간다", x=0.02, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.02, 0.885, "사과 60개, 5겹 교차검증 20회 반복. 파랑 = 원본 사진, 주황 = 사과 정보를 뺀 입력, 회색 = 단순 기준. "
             "고정 ResNet18 특징 + 릿지 회귀(이전 설정 그대로).", fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(out / "무게오차_비교.png", dpi=160)
    plt.close(fig)


def occlusion(results, metrics, out):
    rows = [r for r in read_csv(results / "leaf_occlusion.csv") if r["evaluation"] == "cross_validation"]
    batch_mean = metrics["part_a_original_view"]["cross_validation"]["camera_mean"]["mae_g"]
    views = [("spb_int_me_ver", "원래 조건 (spb_int_me_ver, 카메라 3종)"),
             ("spm_int_me_ver", "카메라 1대 조건 (spm_int_me_ver, 같은 거리·각도)")]
    styles = {"resnet_clean": ("ResNet18 원본학습", HIGHLIGHT), "resnet_aug": ("ResNet18 가림증강", SECOND)}
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), sharey=True)
    for ax, (view, title) in zip(axes, views):
        quiet(ax)
        ax.axhline(batch_mean, color=INK2, linewidth=1.1, linestyle=(0, (3, 3)))
        ax.text(59, batch_mean - 1.2, f"묶음 평균 {batch_mean:.1f} g", ha="right", va="top", fontsize=9, color=INK2)
        for model, (label, color) in styles.items():
            pts = sorted((float(r["level"]), float(r["mae_g"])) for r in rows if r["view"] == view and r["model"] == model)
            ax.plot([100 * x for x, _ in pts], [v for _, v in pts], color=color, linewidth=2, marker="o",
                    markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, label=label)
            ax.annotate(f"{pts[-1][1]:.1f} g", (100 * pts[-1][0], pts[-1][1]), xytext=(9, 0), textcoords="offset points",
                        va="center", fontsize=9.5, color=INK)
        ax.set_title(title, loc="left", fontsize=11.5)
        ax.set_xticks([0, 15, 30, 50])
        ax.set_xlabel("원래 보이던 사과 면적 중 추가로 가린 비율 (%)")
        ax.set_xlim(-3, 60)
        ax.set_ylim(0, 60)
    axes[0].set_ylabel("평균 절대오차 (g)")
    axes[0].legend(frameon=False, fontsize=10, loc="upper left")
    fig.suptitle("원래 조건에서 잎에 끄떡없던 가림증강 모델도 카메라 1대 조건에서는 오차가 커진다", x=0.02, ha="left",
                 fontsize=14, fontweight="bold")
    fig.text(0.02, 0.885, "같은 사과 60개, 5겹 교차검증 20회 반복. 합성 잎 생성기·학습 설정은 이전 실험 그대로.",
             fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(out / "가림별_무게오차_카메라비교.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--out", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    setup()
    metrics = json.loads((args.results / "metrics.json").read_text(encoding="utf-8"))
    rows = read_csv(args.results / "apple_predictions.csv")
    scatter_pair(rows, metrics["part_a_original_view"]["cross_validation"], "original_view",
                 [("resnet_full", "원본 사진"), ("resnet_fruit_erased", "사과를 같은 크기 원으로 지운 사진")],
                 "원래 조건: 사과를 지워도 ResNet18의 무게 예측은 그대로다",
                 "spb_int_me_ver 사진, 사과 60개. 5겹 교차검증 20회 반복 평균 예측. 점 = 사과 1개.",
                 args.out / "원래조건_사과지우기.png")
    scatter_pair(rows, metrics["part_b_single_camera_views"]["average_over_all_single_camera_views"]["cross_validation"],
                 "all_single_camera_views_average",
                 [("resnet_full", "원본 사진"), ("resnet_corners_only", "네 모서리 배경만")],
                 "카메라 1대 조건: 사과가 보여야 같은 묶음 안의 무게 차이를 따라간다",
                 "60개 모두 iPhone 8로 찍은 촬영 조건 54개에서 각각 학습·평가한 뒤 사과마다 예측을 평균.",
                 args.out / "카메라1대_다시점평균.png")
    bars(metrics, args.out)
    occlusion(args.results, metrics, args.out)
    print(f"Figures written to {args.out}")


if __name__ == "__main__":
    main()
