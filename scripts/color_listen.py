"""Listen to the per-pixel color -> spectrum model (decision [C]).

Renders clean color swatches (so the mapping is unambiguous) plus, if given a
Blender capture dir and/or an image, real features.

    python scripts/color_listen.py [CAPTURE_DIR] [IMAGE.jpg x0 y0 x1 y1]
"""

import glob
import os
import sys

import numpy as np

from music_making import audio
from music_making.color_spectrum import feature_spectrum, synthesize

OUT = "demos/color"
os.makedirs(OUT, exist_ok=True)


def render(name, pixels, dur=3.0):
    freqs, E, band = feature_spectrum(pixels)
    audio.save_wav(f"{OUT}/{name}.wav", synthesize(E, freqs, dur=dur))
    print(f"  {name:16s} {len(pixels):7d} px   band {band[0]:5.0f}-{band[1]:5.0f} Hz")


def swatch(rgb, n=3000):
    return np.tile(np.array(rgb, float), (n, 1))


print("swatches (pure color -> the mapping, audibly):")
for name, c in {
    "swatch_red": (1, 0, 0), "swatch_green": (0, 1, 0), "swatch_blue": (0, 0, 1),
    "swatch_magenta": (1, 0, 1), "swatch_gray": (0.5, 0.5, 0.5), "swatch_orange": (1, 0.5, 0),
}.items():
    render(name, swatch(c), dur=2.5)

# real features
import cv2  # noqa: E402


def rgb01(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(float) / 255.0


print("real features:")
if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
    from music_making.capture import load_capture

    cap = load_capture(sys.argv[1])
    i = len(cap.frames) // 2
    png = sorted(glob.glob(os.path.join(sys.argv[1], "frame_*.png")))[i]
    rgb = rgb01(cv2.imread(png, cv2.IMREAD_COLOR))
    rock = rgb[np.isfinite(cap.frames[i].depth)]      # rock = where the depth pass hit geometry
    render("rock", rock, dur=3.5)

if len(sys.argv) > 6 and os.path.exists(sys.argv[2]):
    img = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in sys.argv[3:7])
    crop = img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    render(os.path.splitext(os.path.basename(sys.argv[2]))[0], rgb01(crop).reshape(-1, 3), dur=3.5)
