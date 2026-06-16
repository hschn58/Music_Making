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

## Timbre: describing each element's texture

Timbre is not a coating on a note — it is the **sonification of the real
element**. Each stream has a **TextureProfile** (in the genre preset, swappable)
that describes its element as faithfully as possible:

- **spectral homogeneity** (`bandwidth`): homogeneous/tonal ↔ broadband/noisy
- **multi-timescale amplitude modulation**: a `slow` drift + a `fast` flicker,
  where `chaos` makes the flicker noise-driven rather than a clean LFO
- **medium / residue**: a viscous, dark decay tail — the substance the element
  sits in
- saturation (`drive`, follows **tension**) and a low-pass that opens with
  **brightness**

The three archetypes fall straight out of this:

| Stream | Element | Profile |
|---|---|---|
| terrain | rock | homogeneous, narrow; gentle slow in-and-out; a **viscous lava residue** tail |
| atmosphere | fire | broadband; a **fast chaotic flicker** under a **slow drift** (the flame's wandering centre of mass) |
| entity | agents | kept clean — conscious beings are described by *feeling*, carried by the sparse unison motif, not by physical texture |

A conscious agent can't be captured by texture, so the entity stream is
deliberately untextured; its character comes from the composition (sparse,
collective-unison motif). Cutoffs keep the streams spectrally separated (mid sits
below the high band) so the mix and QC gate stay legible. Note these profiles are
the *same object* a per-feature scalogram would measure — the bridge to driving
timbre from real video texture later.

### Multiscale texture → pure additive synthesis (`texture.py`)

The `TextureProfile` above *post-processes* a fluidsynth note, so it can shape but
not create spectral depth. A feature, though, is itself **multiscale** — a tree is
a trunk, then branches, then twigs, then leaf texture — and depth comes from
sounding those scales directly. So a `FeatureTexture` describes a feature as a
stack of spatial **scale bands** (coarse → fine), and the synthesizer makes *the
feature the spectrum*: each scale becomes a partial layer, coarse scales low, fine
scales high.

The description is the artifact; if it is rigorous enough, sonification is
mechanical. **The table is the rigor gate** — every cell must be filled, and each
maps to an exact sonic consequence (`describe_table()` renders it for any feature;
`missing_rigor()` flags empty cells):

| Dimension | Describes | → Sound |
|---|---|---|
| `scales` | spatial structure, coarse→fine | one partial layer per band |
| · `energy` | structural energy at the scale | partial loudness |
| · `homogeneity` | regular/tonal (1) vs noisy (0) | sine stack vs band noise |
| · `density` | space-filling (1) vs sparse (0) | sustained vs granular twinkle |
| `fractal_slope` | self-similarity (1/fᵝ) | power-law partial rolloff |
| `scale_ratio` | branching ratio across scales | audio octave step per band |
| `bloom` | vertical order, bottom→top | coarse-first onset stagger |
| `medium` | substance it emerges from | dark residue tail |
| `modulation` | temporal life (drift, flicker) | multi-rate amplitude mod |
| `conscious` | a being, not a material | no texture (carried by motif) |

The three archetypes, encoded as gold-standard descriptions: **rock** = energy in
coarse scales, high homogeneity, lava residue → darkest, solid; **fire** =
broadband at every scale + a fast chaotic flicker under a slow drift; **tree** =
homogeneous trunk + a self-similar (fractal) branch rolloff + a sparse, noisy leaf
canopy that blooms up from the trunk. `scripts/texture_demo.py` renders all three.

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
- **Images** (`from_images`): a sequence of photos (a literal storyboard). Each
  image's pixels give its stream levels (warmth+contrast → atmosphere, edges →
  terrain) *and* a TextureProfile (2D spatial-frequency → timbral homogeneity);
  structural features are normalized across the set. The order is the authored
  story and may repeat scenes (theme + recapitulation). This is the spatial-domain
  realization of the scalogram idea: timbre is sourced from real pixels.
- **Video** (`from_video`): a clip → ffmpeg/opencv brightness (atmosphere),
  motion (entities), contrast (terrain), windowed into story segments.

## Lyrics: hybrid, free-first

Lyrics come from an LLM when available (`ANTHROPIC_API_KEY`), and a deterministic
free generator otherwise. `MUSIC_MAKING_OFFLINE=1` forces the free path — the
whole pipeline then runs free, offline, and reproducibly (this is what CI uses).
