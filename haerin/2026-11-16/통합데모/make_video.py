"""Build the demo video from the demo's input and result screens: input photo -> cross-fade -> result screen."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FONT = "C:/Windows/Fonts/malgun.ttf"


def as_bgr(img, size):
    canvas = Image.new("RGB", size, "#fcfcfb")
    img = img.copy()
    img.thumbnail(size)
    canvas.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def card(lines, size):
    img = Image.new("RGB", size, "#fcfcfb")
    d = ImageDraw.Draw(img)
    big, small = ImageFont.truetype(FONT, 46), ImageFont.truetype(FONT, 26)
    y = size[1] // 2 - 120
    for i, text in enumerate(lines):
        d.text((90, y), text, font=big if i == 0 else small, fill="#0b0b0b" if i == 0 else "#52514e")
        y += 80 if i == 0 else 44
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--out", type=Path, default=HERE / "results/시연영상.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--input-s", type=float, default=2.5)
    parser.add_argument("--result-s", type=float, default=6.0)
    parser.add_argument("--fade-s", type=float, default=0.8)
    args = parser.parse_args()
    timing = json.loads((args.results / "처리시간.json").read_text(encoding="utf-8"))
    stems = [Path(i["image"]).stem for i in timing["images"]]
    threshold = json.loads((args.results / f"{stems[0]}_결과.json").read_text(encoding="utf-8"))["settings"]["fill_threshold"]
    multi = json.loads((HERE.parent / "익음도보강/results/multiview.json").read_text(encoding="utf-8"))
    agree = {k: v["decision_same_in_two_draws"] for k, v in multi["by_k"][multi["model"]].items()}
    first = Image.open(args.results / f"{stems[0]}_결과화면.png")
    size = (first.width // 2 * 2, first.height // 2 * 2)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)
    hold = lambda frame, s: [writer.write(frame) for _ in range(int(s * args.fps))]
    hold(card(["사과 수확 로봇 영상 인식 · 통합 데모",
               "사진 한 장 → 사과별 익음도(익음/미숙/보류) · 지름 · 예상 무게 · 판단 보류",
               f"골라 둔 과수원 사진 {len(stems)}장 · 윤곽은 정답 사용 · GPU 처리 평균 {timing['mean_s']:.2f} s/장 (검출 제외)",
               "Team.D_24 · 정해린"], size), 3.5)
    for stem in stems:
        a = as_bgr(Image.open(args.results / f"{stem}_입력화면.png").convert("RGB"), size)
        b = as_bgr(Image.open(args.results / f"{stem}_결과화면.png").convert("RGB"), size)
        hold(a, args.input_s)
        steps = int(args.fade_s * args.fps)
        for k in range(steps):
            writer.write(cv2.addWeighted(a, 1 - k / steps, b, k / steps, 0))
        hold(b, args.result_s)
    hold(card(["판단 보류가 있는 이유",
               "· 사과가 사진 경계에서 잘리면 지름·무게를 내지 않는다",
               f"· 잎에 가려 보이는 면적이 작으면 지름·무게를 내지 않는다 (채움 비율 {threshold:.2f} 미만)",
               "· 익음도 점수가 애매하면 '보류' → 다른 각도의 사진으로 다시 판단",
               f"· 사과 {multi['cohort']['apples']}개 시험: 사진 1장 → 10장 평균 때 판정 일치 {agree['1']:.2f} → {agree['10']:.2f}",
               "· 과수원 무게는 실측과 비교하지 못한 예상값"], size), 5)
    writer.release()
    seconds = 3.5 + len(stems) * (args.input_s + args.fade_s + args.result_s) + 5
    print(f"Wrote {args.out} ({seconds:.0f} s, {size[0]}x{size[1]})")


if __name__ == "__main__":
    main()
