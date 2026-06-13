# Design

## The idea

Good music is the **frequency representation of the scene it is about**. A song
isn't decorated by its setting — the setting *is* the spec the music solves for.
Two anchoring examples:

- Michael Jackson made the video first; the scene informed the musical decisions.
- Super Mario Bros' lava world: low-frequency beats are the lava rock you hop
  across, high-frequency lingering tones are the heat and danger, and the little
  vocal motifs ("ba ba ba… Haaa") are the anthropomorphized enemies.

A scene decomposes into **three layers / frequency streams**:

| Scene layer        | Meaning                         | Band | Instruments        |
|--------------------|---------------------------------|------|--------------------|
| `terrain`          | ground / structure              | low  | bass, kick         |
| `entity_activity`  | anthropomorphized moving agents | mid  | keys, lead, vocals, snare |
| `atmosphere`       | heat / danger / brightness      | high | pad, shimmer, hats |

## Story: shifting dominance

The crucial part of story-depth is that **which stream is in the foreground
shifts as the story unfolds** — an all-rock stretch is bass-forward; running
through fire is highs-forward; facing moving enemies is mid-forward.

So a situation is first parsed into a **Story**: an ordered list of segments,
each scored for all three streams (with one dominant). Text is split on
connectives (`then`, `while`, `and`, punctuation) with light stemming; video is
split into windows. The scene timeline is then built from the segments so the
dominant stream changes section to section. That single decision drives three
things downstream:

- **Arrangement** — instruments are gated by their stream, so they thin and
  thicken with the story (bass drops out where there's no terrain, hats only on
  atmosphere, the lead motif foregrounds on entity moments).
- **Mix** — each stem's level rides its stream's scene layer, so the dominant
  stream literally comes forward in its segment.
- **A recurring motif** — the lead is a short theme that is transposed/retrograded
  per segment, so the piece feels composed rather than random.

## Timbre: a voice per stream

Each stream has its own **TimbreKit** (in the genre preset, swappable), and the
timbre is also modulated by the scene over time:

- `drive` (saturation) follows **tension**
- a time-varying low-pass opens with **brightness**
- per-kit `reverb` (space) and `tremolo` (movement)

Cutoffs are chosen so the streams stay spectrally separated (the mid stream sits
below the high band), which keeps both the mix and the QC gate legible.

## Architecture

A single **Storyboard** (scene timelines + Story + entity events) is the root
contract. Four workflows condition on it and run as a DAG:

```
lyrics   ─┐
compose  ─┼─► vocals ─► mix ─► QC gate
beats    ─┘
```

`lyrics`/`compose`/`beats` are independent and run concurrently; `vocals` needs
the lyrics (syllables) and lead melody (note pitches); `mix` needs all stems.
Every arrow is a typed Pydantic contract (`contracts.py`). Each instrument is
rendered to an isolated stem (`fluidsynth` + GM SoundFont), stamped with its
stream's timbre, then mixed.

## The autonomous quality gate

No human in the loop. A track passes QC (`qc.py`) when it is non-silent, the
right length, not clipping, **and** the scene is tracked — by either of two
measures:

1. **Envelope correlation** — the low/mid/high energy envelopes of the final mix
   correlate with `terrain`/`entity_activity`/`atmosphere`.
2. **Dominance accuracy** — for each story segment, its dominant stream's band is
   louder than that band's own average (i.e. the foreground actually shifts with
   the story).

(2) is the more robust, concept-aligned measure: it tests the lava-world
intuition directly. Mastering uses a single constant gain (not dynamic loudness
normalization) so the scene's energy arc — the thing being measured — survives.

## Inputs

- **Text** (`from_text`): a situation → story segments → scene timelines.
- **Video** (`from_video`): a clip → ffmpeg/opencv brightness (atmosphere),
  motion (entities), contrast (terrain), windowed into story segments.

## Lyrics: hybrid, free-first

Lyrics come from an LLM when available (`ANTHROPIC_API_KEY`), and a deterministic
free generator otherwise. `MUSIC_MAKING_OFFLINE=1` forces the free path — the
whole pipeline then runs free, offline, and reproducibly (this is what CI uses).
