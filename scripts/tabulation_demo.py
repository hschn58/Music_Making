"""Render a tabulation to audio and print its perspective table.

    /opt/venv/bin/python scripts/tabulation_demo.py [out_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

from music_making import audio
from music_making.tabulation import describe_perspective, forest_fire_walk, render_tabulation


def main(argv: list[str]) -> int:
    out = Path(argv[1]) if len(argv) > 1 else Path("demos/tabulation")
    out.mkdir(parents=True, exist_ok=True)
    tab = forest_fire_walk()
    print("\n" + describe_perspective(tab) + "\n")
    x = render_tabulation(tab, seed=0)
    wav = str(out / "walk.wav")
    audio.save_wav(wav, x)
    try:
        audio.to_mp3(wav, str(out / "walk.mp3"))
    except Exception as e:
        print(f"(mp3 skipped: {e})")
    print(f"wrote {wav}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
