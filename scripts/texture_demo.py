"""Render the gold-standard feature textures so they can be heard, and print the
rigor table for each. Pure additive — no SoundFont needed.

    /opt/venv/bin/python scripts/texture_demo.py [out_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

from music_making import audio
from music_making.texture import (
    describe_table,
    fire_texture,
    render_feature,
    rock_texture,
    tree_texture,
)

FEATURES = [
    ("rock", rock_texture(), 55.0),
    ("fire", fire_texture(), 70.0),
    ("tree", tree_texture(), 70.0),
]


def main(argv: list[str]) -> int:
    out = Path(argv[1]) if len(argv) > 1 else Path("demos/textures")
    out.mkdir(parents=True, exist_ok=True)
    for name, ft, f0 in FEATURES:
        print("\n" + describe_table(ft) + "\n")
        x = render_feature(ft, f0=f0, dur=4.0, seed=0)
        wav = str(out / f"{name}.wav")
        audio.save_wav(wav, x)
        try:
            audio.to_mp3(wav, str(out / f"{name}.mp3"))
        except Exception as e:  # ffmpeg optional for the demo
            print(f"(mp3 skipped: {e})")
        print(f"wrote {wav}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
