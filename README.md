# Music_Making

Turn a **scene** into music. The guiding idea: **good music is the frequency
representation of the scene it is about.**

The project has two layers, at different stages of maturity:

1. **The storyboard pipeline (working).** Describe a situation — or hand it photos or
   a video — and a *storyboard* drives four parallel workflows (lyrics, composition,
   beats, sung vocals) that are mixed into one track and checked by a scene-aware QC
   gate. This is the `make-song` CLI below.
2. **The literal scene→frequency model (active research — current focus).** A more
   direct realization of the thesis: read each visual *feature* straight into sound,
   with no musical middleman. This is where current work is — see
   **[docs/design_state.md](docs/design_state.md)** for the live design.

A scene splits into three layers, each mapped to a frequency band — ground/structure
(low), the moving "characters" (mid), heat/atmosphere (high). See [DESIGN.md](DESIGN.md).

## The literal scene→frequency model (current direction)

The newer line drops the lyrics/beats/vocals scaffolding and asks: what does a scene
*sound like* if you read it straight off the pixels?

- **Extraction — the "sample space."** Every pixel's color becomes a little spectrum
  — hue → which frequencies (red low … violet high), saturation → tone-vs-noise,
  brightness → weight — and a feature's pixels sum into a frequency distribution.
  That distribution is the *complete* sound of the feature: beauty and ugliness
  together, never edited. (`color_spectrum.py`; a Blender scene-capture path feeds it
  synthetic walks via `capture.py` / `blender/`.)
- **Choosing the beauty.** Beauty isn't installed into the extraction — it's the act
  of *choosing* what to attend to within it: the structure latent in the sample space
  that is low-roughness, has a coherent center, and is resolvable. **Harmonicity is
  optional** — bells, bowls, and gongs are inharmonic and beautiful. Sensory scoring
  lives in `consonance.py` (Sethares roughness + relative period); the per-feature
  "beauty-optimizer" that uses it is still being designed.
- **Safety.** Every render passes a mastering guard (`safety.py`): infrasound
  high-pass and a peak limit, so nothing reaches biologically harmful levels.
- **Status & docs.** The extraction, capture, safety stage, and two-axis consonance
  analysis exist and are tested. The chooser/optimizer is in design. Full decisions
  and open questions: **[docs/design_state.md](docs/design_state.md)**; the math in
  [docs/relative_period.md](docs/relative_period.md).

## Install

System tools (free): `ffmpeg`, `fluidsynth` + a General MIDI SoundFont, `espeak-ng`.

```bash
sudo apt-get install -y ffmpeg fluidsynth fluid-soundfont-gm espeak-ng libsndfile1
pip install -e ".[dev]"
```

The SoundFont is auto-discovered at `/usr/share/sounds/sf2/FluidR3_GM.sf2`; override
with `MUSIC_MAKING_SOUNDFONT=/path/to.sf2`.

## Usage (storyboard pipeline)

```bash
# from a described scene
make-song --situation "hopping across lava rock while fire enemies jump in the heat" \
          --duration 30 --out ./out

# from a sequence of photos (a literal storyboard; order = the story)
make-song --images trail.jpg teepee.jpg fire.jpg --seconds-per-scene 9 --out ./out

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

### Lyrics: free by default

Lyrics use an LLM when one is available (`ANTHROPIC_API_KEY`), and a deterministic
free generator otherwise. Set `MUSIC_MAKING_OFFLINE=1` to force the free, offline,
reproducible path (this is what tests and CI use).

### How it fits together

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
