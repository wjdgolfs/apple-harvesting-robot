"""Translate one synthetic leaf over a fixed apple image and visible mask.

The displacement is a screen-space simulation. Its amplitude is a fraction of the
visible fruit's bounding-box width. Coverage always uses the original visible
fruit area as denominator. The helper leaves RGB camera geometry unchanged.
"""

import hashlib
import math
import random

from PIL import Image, ImageChops, ImageDraw


AMPLITUDES = {"stationary": 0.0, "weak": 0.05, "strong": 0.20}


def _leaf_template(size, fruit, target_fraction, sample_key, seed):
    """Return an RGB leaf layer and its exact binary silhouette on the full frame."""
    box = fruit.getbbox()
    local = fruit.crop(box)
    area = local.histogram()[255]
    width, height = local.size
    rng = random.Random(hashlib.sha256(f"{seed}|{sample_key}".encode()).digest())
    angle = rng.uniform(0, 2 * math.pi)
    center = width * rng.uniform(.26, .74), height * rng.uniform(.26, .74)
    aspect = rng.uniform(.35, .51)
    direction = math.cos(angle), math.sin(angle)
    profile = [(t / 16, side * aspect * (1 - (t / 16) ** 2))
               for side, steps in ((1, range(-16, 17)), (-1, range(15, -16, -1)))
               for t in steps]
    outline = [(direction[0]*x-direction[1]*y, direction[1]*x+direction[0]*y) for x,y in profile]

    def polygon(radius, offset=(0, 0)):
        return [(offset[0]+round(center[0]+radius*x), offset[1]+round(center[1]+radius*y))
                for x,y in outline]

    low, high = 0.0, 5.0 * max(width, height)
    best_radius, best_error = 0.0, 2.0
    for _ in range(16):
        radius = (low + high) / 2
        silhouette = Image.new("L", local.size)
        ImageDraw.Draw(silhouette).polygon(polygon(radius), fill=255)
        fraction = ImageChops.darker(local, silhouette).histogram()[255] / area
        error = abs(fraction - target_fraction)
        if error < best_error:
            best_radius, best_error = radius, error
        if fraction < target_fraction:
            low = radius
        else:
            high = radius
    mask = Image.new("L", size)
    ImageDraw.Draw(mask).polygon(polygon(best_radius, box[:2]), fill=255)
    green = rng.randrange(64, 110)
    layer = Image.new("RGB", size, (round(green*.56), green, round(green*.37)))
    ends = [(round(center[0]+box[0]+sign*best_radius*direction[0]*.91),
             round(center[1]+box[1]+sign*best_radius*direction[1]*.91)) for sign in (-1, 1)]
    ImageDraw.Draw(layer).line(ends, fill=(round(green*.71),green+9,round(green*.43)), width=1)
    return layer, mask


def _translate(image, dx, dy):
    """Shift without wrapping pixels across a frame boundary."""
    shifted = Image.new(image.mode, image.size)
    shifted.paste(image, (dx, dy))
    return shifted


def generate_frames(rgb, visible_mask, sample_key, frames=12, seed=20261117, target_fraction=.15):
    """Yield stationary/weak/strong frame dictionaries for one fixed source image.

    Each condition starts at zero displacement. Weak and strong translations have
    peak magnitudes 5% and 20% of the original visible bounding-box width, sampled
    once around a sinusoid. A shared seed fixes leaf shape, position and motion
    direction. ``visible_mask`` in each output is the original binary mask with
    the shifted leaf removed. The added ``leaf_mask`` covers its whole silhouette,
    including background pixels and any original natural occluder.
    """
    if frames < 4 or not isinstance(frames, int):
        raise ValueError("frames must be an integer of at least four")
    if rgb.size != visible_mask.size or not 0 < target_fraction < 1:
        raise ValueError("RGB/mask size mismatch or invalid target fraction")
    source = rgb.convert("RGB")
    original = visible_mask.convert("L").point(lambda value: 255 if value else 0)
    box = original.getbbox()
    if box is None:
        raise ValueError("Visible fruit mask must contain pixels")
    original_area = original.histogram()[255]
    fruit_width = box[2] - box[0]
    layer, initial_leaf = _leaf_template(source.size, original, target_fraction, sample_key, seed)
    motion_rng = random.Random(hashlib.sha256(f"motion|{seed}|{sample_key}".encode()).digest())
    direction = motion_rng.uniform(0, 2 * math.pi)
    for condition, amplitude in AMPLITUDES.items():
        for index in range(frames):
            displacement = amplitude * fruit_width * math.sin(2 * math.pi * index / frames)
            dx = round(displacement * math.cos(direction))
            dy = round(displacement * math.sin(direction))
            leaf = _translate(initial_leaf, dx, dy)
            shifted_layer = _translate(layer, dx, dy)
            visible = ImageChops.subtract(original, leaf)
            covered = ImageChops.darker(original, leaf).histogram()[255]
            yield {"condition":condition, "index":index,
                   "rgb":Image.composite(shifted_layer, source, leaf),
                   "leaf_mask":leaf, "visible_mask":visible,
                   "coverage":covered/original_area,
                   "covered_pixels":covered, "original_visible_pixels":original_area,
                   "offset_x_px":dx, "offset_y_px":dy,
                   "motion_amplitude_fraction":amplitude, "target_fraction":target_fraction}
