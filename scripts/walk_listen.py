"""Walk through a Blender capture and listen: per-frame color->spectrum stacked
into a spectrogram, then resynthesized. Also saves a spectrogram image.

    python scripts/walk_listen.py CAPTURE_DIR [DURATION_S]
"""

import glob
import os
import sys

import cv2
import numpy as np

from music_making import audio
from music_making.color_spectrum import FREQS, feature_spectrum, synthesize_spectrogram

cap_dir = sys.argv[1]
dur = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
OUT = "demos/walk"
os.makedirs(OUT, exist_ok=True)

pngs = sorted(glob.glob(os.path.join(cap_dir, "frame_*.png")))
cols = []
for i, p in enumerate(pngs):
    rgb = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(float) / 255.0
    _, E, band = feature_spectrum(rgb.reshape(-1, 3), seed=i)
    cols.append(E)
    if i % 6 == 0 or i == len(pngs) - 1:
        print(f"  frame {i:2d}  band {band[0]:5.0f}-{band[1]:5.0f} Hz")
E_t = np.array(cols)

audio.save_wav(f"{OUT}/walk.wav", synthesize_spectrogram(E_t, dur=dur))

# spectrogram image: freq (low at bottom) x time
img = (E_t.T / (E_t.max() + 1e-9) * 255).astype(np.uint8)        # (NG, T)
img = cv2.resize(np.flipud(img), (max(480, len(pngs) * 12), len(FREQS)),
                 interpolation=cv2.INTER_NEAREST)
cv2.imwrite(f"{OUT}/walk_spectrogram.png", cv2.applyColorMap(img, cv2.COLORMAP_MAGMA))
print(f"saved {OUT}/walk.wav ({dur:.0f}s) and walk_spectrogram.png")
