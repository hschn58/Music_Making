# Design

## The idea

Good music is the **frequency representation of the scene it is about**. A song
isn't decorated by its setting — the setting *is* the spec the music solves for.
Two anchoring examples:

- Michael Jackson made the video first; the scene informed the musical decisions.
- Super Mario Bros' lava world: low-frequency beats are the lava rock you hop
  across, high-frequency lingering tones are the heat and danger, and the little
  vocal motifs ("ba ba ba… Haaa") are the anthropomorphized enemies.

So a scene decomposes into **three layers, each mapped to a frequency band**:

| Scene layer        | Meaning                         | Band | Instruments        |
|--------------------|---------------------------------|------|--------------------|
| `terrain`          | ground / structure              | low  | bass, kick         |
| `entity_activity`  | anthropomorphized moving agents | mid  | keys, lead, vocals |
| `atmosphere`       | heat / danger / brightness      | high | pad, hats          |

## Architecture

A single **Storyboard** (the scene, as layered timelines + discrete entity
events) is the root contract. Four workflows condition on it and run as a DAG:

```
lyrics   ─┐
compose  ─┼─► vocals ─► mix ─► QC gate
beats    ─┘
```

`lyrics`, `compose`, and `beats` are independent and run concurrently. `vocals`
needs the lyrics (syllables) and the lead melody (note pitches). `mix` needs all
stems. Every arrow is a typed Pydantic contract (`contracts.py`), which is what
lets the workflows run in parallel yet still cohere into one song.

Each instrument is generated so its band energy *follows* its scene layer (e.g.
the bass plays louder/denser where `terrain` is high), and is rendered to an
isolated stem so the mix and the QC gate can reason per-band.

## The autonomous quality gate

There is no human in the loop. A track passes QC (`qc.py`) when:

1. it is non-silent, the right length, and not clipping, **and**
2. its measured per-band energy envelope **tracks the scene**: the low/mid/high
   envelopes of the final mix correlate with `terrain`/`entity_activity`/
   `atmosphere` above a threshold.

(2) is the theory made measurable — the lava-world intuition as a pass/fail test.
Mastering uses a single constant gain rather than dynamic loudness normalization
precisely so this temporal envelope is preserved.

## Inputs

- **Text** (`from_text`): a situation in words → keyword + arc heuristics build
  the layered timelines.
- **Video** (`from_video`): a real clip → ffmpeg/opencv extract brightness
  (→ atmosphere), motion (→ entities), and contrast (→ terrain), with motion
  spikes becoming entity events.

## Lyrics: hybrid, free-first

Lyrics come from an LLM when one is available (`ANTHROPIC_API_KEY`, online), and
from a deterministic free generator otherwise. Set `MUSIC_MAKING_OFFLINE=1` to
force the free path — the whole pipeline then runs free, offline, and
reproducibly (this is what CI uses).
