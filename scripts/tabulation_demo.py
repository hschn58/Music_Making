"""Render a tabulation to audio and show it as both tables (perspective + grid).

    /opt/venv/bin/python scripts/tabulation_demo.py [tabulation.yaml] [out_dir]

With no YAML path it falls back to the built-in forest_fire_walk(). Edit the
YAML file and re-run to hear the change — the file is the authoring surface.
"""

from __future__ import annotations

import sys
from pathlib import Path

from music_making import audio
from music_making.tabulation import (
    describe_grid,
    describe_perspective,
    forest_fire_walk,
    load_tabulation,
    render_tabulation,
)

DEFAULT_YAML = "tabulations/forest_fire_walk.yaml"


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:]]
    yaml_path = next((a for a in args if a.endswith((".yaml", ".yml"))), None)
    out_args = [a for a in args if a != yaml_path]
    out = Path(out_args[0]) if out_args else Path("demos/tabulation")
    out.mkdir(parents=True, exist_ok=True)

    if yaml_path:
        tab = load_tabulation(yaml_path)
    elif Path(DEFAULT_YAML).is_file():
        tab = load_tabulation(DEFAULT_YAML)
        yaml_path = DEFAULT_YAML
    else:
        tab = forest_fire_walk()

    print(f"\nsource: {yaml_path or 'built-in forest_fire_walk()'}\n")
    print(describe_perspective(tab) + "\n")
    print(describe_grid(tab) + "\n")

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
