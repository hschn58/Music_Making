# Music_Making

Compose original music from a **scene**. You describe a situation (or hand it a
video); a *storyboard* of that scene drives four parallel workflows — lyrics,
composition, beats, and sung vocals — that are mixed into one track and checked
by an autonomous, scene-aware quality gate.

The guiding idea: **good music is the frequency representation of the scene it is
about.** A scene splits into three layers, each mapped to a frequency band — the
ground/structure (low), the moving "characters" (mid: lead + vocals), and the
heat/atmosphere (high). See [DESIGN.md](DESIGN.md).

## Install

System tools (free): `ffmpeg`, `fluidsynth` + a General MIDI SoundFont, `espeak-ng`.

```bash
sudo apt-get install -y ffmpeg fluidsynth fluid-soundfont-gm espeak-ng libsndfile1
pip install -e ".[dev]"
```

The SoundFont is auto-discovered at `/usr/share/sounds/sf2/FluidR3_GM.sf2`; override
with `MUSIC_MAKING_SOUNDFONT=/path/to.sf2`.

## Usage

```bash
# from a described scene
make-song --situation "hopping across lava rock while fire enemies jump in the heat" \
          --duration 30 --out ./out

# from a real video (the music mirrors what's on screen)
make-song --video clip.mp4 --out ./out
```

Outputs `out/track.wav`, `out/track.mp3`, and `out/metadata.json` (storyboard +
QC report). The process exits non-zero if the track fails the QC gate.

```python
from music_making import from_text, produce
track = produce(from_text("a calm night drive through neon streets"), "out")
print(track.qc.passed, track.qc.mean_correlation)
```

## Lyrics: free by default

Lyrics use an LLM when one is available (`ANTHROPIC_API_KEY`), and a deterministic
free generator otherwise. Set `MUSIC_MAKING_OFFLINE=1` to force the free, offline,
reproducible path (this is what tests and CI use).

## How it fits together

```
storyboard ─► lyrics  ─┐
            ► compose ─┼─► vocals ─► mix ─► QC gate
            ► beats   ─┘
```

Each instrument is generated so its frequency band tracks its scene layer, and the
QC gate passes a track only when the rendered mix's per-band energy envelope
follows the storyboard — the theory, made into a measurable pass/fail.

## Development

```bash
MUSIC_MAKING_OFFLINE=1 pytest -q
```

## License

MIT
