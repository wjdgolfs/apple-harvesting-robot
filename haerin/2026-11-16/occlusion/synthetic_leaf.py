"""Deterministic synthetic leaf. Fractions refer to the original visible fruit pixels."""

import hashlib
import math
import random

from PIL import Image, ImageChops, ImageDraw


def apply_leaf(rgb, visible_mask, target_fraction, sample_key, seed=20261116):
    """Overlay one opaque leaf; return (RGB image, achieved intersection fraction)."""
    if not 0 <= target_fraction <= 1 or rgb.size != visible_mask.size:
        raise ValueError("Invalid target fraction or mask shape")
    if target_fraction == 0:
        return rgb.convert("RGB"), 0.0
    mask = visible_mask.convert("L").point(lambda v: 255 if v else 0)
    box = mask.getbbox()
    if box is None:
        raise ValueError("No visible fruit pixels")
    fruit = mask.crop(box)
    area = fruit.histogram()[255]
    width, height = fruit.size
    rng = random.Random(hashlib.sha256(f"{seed}|{sample_key}".encode()).digest())
    angle = rng.uniform(0, 2 * math.pi)
    center = (width * rng.uniform(0.26, 0.74), height * rng.uniform(0.26, 0.74))
    aspect = rng.uniform(0.35, 0.51)
    direction = math.cos(angle), math.sin(angle)
    # Biconvex outline with pointed tips. All levels share orientation, color and anchor.
    profile = [(t / 16, side * aspect * (1 - (t / 16) ** 2))
               for side, steps in ((1, range(-16, 17)), (-1, range(15, -16, -1)))
               for t in steps]
    outline = [(direction[0] * x - direction[1] * y, direction[1] * x + direction[0] * y)
               for x, y in profile]

    def polygon(radius, offset=(0, 0)):
        return [(offset[0] + round(center[0] + radius * x),
                 offset[1] + round(center[1] + radius * y)) for x, y in outline]

    low, high = 0.0, 5.0 * max(width, height)
    best_error, best_radius, covered = 2.0, 0.0, 0
    for _ in range(14):
        radius = (low + high) / 2
        silhouette = Image.new("L", fruit.size)
        ImageDraw.Draw(silhouette).polygon(polygon(radius), fill=255)
        count = ImageChops.darker(fruit, silhouette).histogram()[255]
        error = abs(count / area - target_fraction)
        if error < best_error:
            best_error, best_radius, covered = error, radius, count
        if count / area < target_fraction:
            low = radius
        else:
            high = radius

    output = rgb.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    green = rng.randrange(64, 110)
    color = (round(green * 0.56), green, round(green * 0.37))
    absolute = polygon(best_radius, box[:2])
    draw.polygon(absolute, fill=color)
    # Thin midrib is entirely inside the silhouette.
    midrib = (round(green * 0.71), green + 9, round(green * 0.43))
    endpoints = [(round(center[0] + box[0] + sign * best_radius * direction[0] * 0.91),
                  round(center[1] + box[1] + sign * best_radius * direction[1] * 0.91))
                 for sign in (-1, 1)]
    draw.line(endpoints, fill=midrib, width=1)
    return output, covered / area
