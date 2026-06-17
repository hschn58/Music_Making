"""Extract a tabulation from a real still image and render it.

Feature identification (the regions below) is the one perceptual step — authored by
looking at the photo. Everything else is measured. Regions here are for the
campfire scene (scenes/IMG_2054.jpeg); swap in your own image + regions.

    /opt/venv/bin/python scripts/extract_demo.py [image] [out_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

from music_making import audio
from music_making.extract import FeatureRegion, extract_tabulation
from music_making.tabulation import (
    describe_grid,
    describe_perspective,
    render_tabulation,
    save_tabulation,
)

# Campfire scene: features I identified by eye (box = x0,y0,x1,y1 normalized).
CAMPFIRE = [
    FeatureRegion("night", (0.00, 0.00, 1.00, 0.12)),   # dark surround, top
    FeatureRegion("logs",  (0.38, 0.22, 0.60, 0.50)),   # the crossed wood
    FeatureRegion("fire",  (0.28, 0.42, 0.52, 0.72)),   # bright flames
    FeatureRegion("sand",  (0.62, 0.58, 1.00, 1.00)),   # ground, lower-right
]


def main(argv: list[str]) -> int:
    image = argv[1] if len(argv) > 1 else "scenes/IMG_2054.jpeg"
    out = Path(argv[2]) if len(argv) > 2 else Path("demos/extract")
    out.mkdir(parents=True, exist_ok=True)

    tab = extract_tabulation(image, CAMPFIRE, duration=14.0, title="campfire (measured)")
    yaml_path = save_tabulation(tab, "tabulations/campfire.yaml")
    print(f"\nfeature-ID: {len(CAMPFIRE)} regions (mine) | everything below measured\n")
    print(open(yaml_path).read())
    print(describe_perspective(tab) + "\n")
    print(describe_grid(tab) + "\n")

    x = render_tabulation(tab, seed=0)
    wav = str(out / "campfire.wav")
    audio.save_wav(wav, x)
    try:
        audio.to_mp3(wav, str(out / "campfire.mp3"))
    except Exception as e:
        print(f"(mp3 skipped: {e})")
    print(f"wrote {wav}  ({tab.duration:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
