# walk_complex — a video → sound example

A self-contained example of the current pipeline: a synthetic Blender walk through a
rich scene (14 object-index features, multi-hue textured materials) turned into sound
by the viewer-coupled dynamics model `G = A(F)·A(B)·σ`.

- `scene.mp4` — the input "video" (the rendered RGB walk).
- `features.png` — per-feature spectrograms, each with its feature-attention `A(F)` profile above.
- `features_mix.wav` — the 45 s output mix.

## What it shows (and the limitation it exposed)

The world is deliberately rich, yet most features come out as **broadband washes**, not
chords. The color→frequency channel can only ever make a single tone (uniform hue) or a
wash (wide/gradient hue) — never several discrete *related* partials. This motivated the
pivot to a **physical-resonator** model (mode frequencies from size+shape, color demoted
to timbre); see `docs/viewer_coupled_dynamics.md`.

## Regenerate the full capture + output

The full capture (frame/depth/index passes, ~116 MB) is gitignored; regenerate it:

```bash
blender --background --python blender/walk_complex.py -- demos/walk_complex 72 200 24
python scripts/blender_feature_test.py demos/walk_complex 45 --force
# -> demos/walk_complex/{features.png, features_mix.wav}
```
