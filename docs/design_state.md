# Beauty-perspective design — current state (2026-06-19)

Authoritative snapshot of decisions and open questions for the per-feature
"beauty perspective" work. Supersedes the relevant parts of `decision_map.md`
(which still marks older items as resolved). Companion math: `relative_period.md`.

Status of the work: **design / thinking stage.** The modules and scripts that
exist (`consonance.py`, `scripts/consonance_study.py`, `periodicity_study.py`,
`inharmonic_good.py`) are **analysis tools and probes only** — not the chosen
architecture. Nothing here is committed as the implementation.

---

## 0. Foundation (carried in, still in force)

- **Thesis:** music = the frequency representation of the scene it is about.
- **Meaning is listener-emergent**, not authored onto elements; our job is faithful
  per-element acoustic description.
- The hypothesis is **mode-independent** (visual is just the default input).

---

## 1. The sample-space frame (settled this session)

- The literal color→spectrum extraction **stays, untouched.** It is **not the music**
  — it is the **sample space**: the complete sound of the scene, beauty and ugliness
  tangled together. (Analogy: a charming town still has a hundred describable defects;
  it is *all* the light hitting the retina.)
- **Beauty is not installed into the base layer.** It is the **act of choosing** what
  to attend to within it. We never edit / retune / quantize the base.
- Corollary: building the **chooser** is the work — not "fixing" the extraction.

### The beauty hypothesis (Henry's, the working aesthetic)

- Beauty ≈ **the most resolvable detail the ear can bind to the fewest underlying
  rules** — i.e. *ordered + detailed + resolvable*, maximized together.
- Delivered across three axes: **frequency** (harmony), **space** (the scene),
  **time** (evolution).
- Refinements: "resolvable" must mean resolvable **into structure** (two ceilings:
  cochlear = critical band, and cognitive = parseable); richness is best delivered
  **over time** (the ear's resolution renews each instant).
- **Known gap:** this is a theory of *sensory* beauty. It under-explains
  expectation / tension-and-release / surprise (a deceptive cadence is beautiful by
  *violating* a set-up pattern). Treat resolvable-ordered-detail as the **floor**,
  structured violation as a higher layer for later.

---

## 2. The extraction = the sample space (technical, as implemented)

Source: `src/music_making/color_spectrum.py`. Per pixel, color → `(H, S, V)`:

```
Tonal peak at frequency       f(H) = 110 * 32^(H / 0.75)   Hz        (110 = red, 3520 = violet, 32 = 3520/110)
  - spectral hues (H <= 0.75): one peak at f(H),  weight V*S
  - purples    (H >  0.75):    TWO peaks, at 110 and 3520 Hz,
                               weights V*S*p and V*S*(1-p),  p = (H-0.75)/0.25
Broadband floor:              weight V*(1-S), spread UNIFORMLY over [110, 3520] Hz

Per-pixel spectrum  =  V * [ S * tone(H)  +  (1 - S) * flat ]
```

Per feature (sum over its pixels):
- tonal peaks accumulate on a **900-bin log grid, 60–5000 Hz**, then smoothed by a
  Gaussian ~**0.06 octaves** wide (each line → a small bump);
- floor = total `Σ V*(1-S)` spread evenly over the band;
- `E = tone + flat`, normalized so `max(E) = 1`.
- Band edges `[f_lo, f_hi]` = saturation-weighted 5th / 95th percentile of the
  tonal-peak frequencies.

`E` over the 900 bins is the coarse per-feature distribution we call the sample space.

**Empirical finding (3 local walk frames, whole-frame pixels):** the literal map is
**smooth but inharmonic** — roughness ~0.01 (milder than a major triad), but the
prominent partials form no simple ratios (relative period ~40–105 cycles). It is a
*pitchless wash*. Under the new frame this is fine: it's the sample space, not the
output.

---

## 3. What "good sound" is (settled understanding)

Two perceptual axes for combinations (from the discrete-tone study + `relative_period.md`):

1. **Roughness** (Sethares / Plomp-Levelt): beating between partials inside a
   **critical band**. A hard perceptual ceiling — not a stylistic choice. Catches
   beating, semitone clusters, high dense screech.
2. **Relative period**: cycles of the lowest tone before `p(t)` repeats =
   `lcm` of the rationalized frequency ratios. Short = "locks in." Catches the
   tritone and inharmonic clang (which roughness rates as fine).

**Key result: harmonicity (short relative period) is NOT required for beauty.**
Demonstrated with tuned bell / singing bowl / gamelan — all low-roughness, all
*long / inharmonic* relative period (15–240), all beautiful. So relative period is
**demoted from a necessary condition to one route in.**

What good (incl. inharmonic) sounds share instead:
- a **definite, stable set of partials** (order without harmonicity — the line
  between a bell and a smear of hiss);
- **roughness avoided** (partials spaced beyond the critical band);
- a **perceptual center** (a few strong, near-harmonic low partials let the ear
  infer one pitch; genuinely inharmonic partials ride on top as *color*);
- it **unfolds in time** (differential decay).

Two enlarging nuances:
- **Beating is rate-dependent**, not binary-bad: slow ~1–5 Hz detune = beautiful
  *shimmer*; fast 20–40 Hz = roughness.
- **A "dissonant" interval baked into a timbre is charm, not tension** (the bell's
  minor-third tierce). Consonance is relative to the spectrum (Sethares).

"Good sounding noise/sound" therefore has **two poles, both inharmonic-friendly**:
- **Bell pole:** stable discrete partials + a center, sparse enough to be clear.
- **Noise pole:** no discrete pitch, but a smooth **downward-tilted** envelope
  (pink-ish), energy kept **off the harsh ~2–5 kHz** band, a gentle resonant center,
  under the roughness ceiling. (e.g. ocean / wind / rain / fire.)

---

## 4. Architecture decisions (current)

### 4.1 Levels of attention (the chooser is hierarchical)
Attention is a **finite, contested budget**, allocated **scene → feature → frequency**:
- **Across features** (how much budget each feature gets): the **gaze/path is just a
  gain modulator**, plus `1/√r` distance falloff and size / energy share.
  **Already built.** (Settled — leave it.)
- **Within a feature** (how the budget is spent across its frequencies): the **new
  work**, split into two orthogonal problems.

### 4.2 Structure vs dynamics (the orthogonal split)
A feature's sound = **invariant structure** + **variable, structure-preserving movement**:
- **Problem A — structure (invariant):** the stable skeleton of where energy lives
  (band centers, the center, overall organization). Organized, *not necessarily
  harmonic*. This is "the broader organizational structure."
- **Problem B — dynamics (variable):** within-band change over time in amplitude and
  relative distribution (decay, slow oscillation/shimmer, energy sloshing between the
  existing bands).
- **Honesty rule:** movement may **modulate what's there in time**, may **not
  restructure it.** This makes A and B orthogonal — build A now, layer B later.
- **Focus order: A first** (Henry's call).

### 4.3 Softmax: rejected as the finder, kept as a local masker
- **Rejected:** softmax-as-the-beauty-finder. Softmax is **pointwise** (weights each
  frequency by its own salience), so it follows **naturally present attention /
  salience — which is not beauty.** Beauty is **configurational** (a property of the
  whole set: spacing, roughness, a center). Softmax cannot see configurations, so it
  can only amplify what's already loud.
- **Kept (demoted):** softmax as a **per-band, per-feature, per-time-step local
  masker.** Within one band the frequencies are mutually close → mutually rough; a
  sharp per-band softmax concentrates the band onto a representative and lets the rest
  recede = **masking / lateral inhibition that kills intra-band roughness.**

### 4.4 Problem A = a constrained optimization
> Choose the spectral structure (**band centers + their loudness**) that
> **maximizes beauty** (configurational — low inter-band roughness, a coherent
> center, resolvable) **subject to a faithfulness constraint.**

The optimization, **not** softmax, is what finds the structure — because beauty is
configurational.

### 4.5 Faithfulness attaches to FREQUENCIES, not loudness (latest, important)
- The faithfulness constraint binds **which frequencies may exist** — those come from
  color; we **cannot invent frequencies or retune.**
- It does **not** bind **how loud** each frequency is. **Loudness is the free
  variable the beauty-optimizer sets** — freely, down to silence.
- **The brightness/value loudness envelope is dropped (for now)** from the structure.
  Rationale ("child-president"): the brightness histogram is the *current standing,
  not the potential*; anchoring loudness to it forecloses the beautiful realization
  before we look for it. We seek **best-case beauty, not current state.**
- **Image:** color = the **palette** (identity); the optimizer **paints** the
  beautiful composition with it.
- **Brightness relocated (proposed, not decided):** value is the quantity that varies
  every time step — the signature of a **dynamics** quantity. Its natural home is
  likely **Problem B** (brightness-over-time → the movement / life), not static
  loudness. So value isn't discarded, it moves.

---

## 5. Open questions (current)

**Problem A (the focus):**
1. **The beauty objective.** What exactly does "beauty" score over a configuration?
   Components in play: a roughness term (inter-band), center/coherence, resolvability/
   detail, the bell↔noise pole position. How are they combined? *Undefined.*
2. **What the band centers are chosen from.** Candidate center frequencies — and how
   the optimizer selects/places them — is the heart of A. *Undefined.*
3. **What we keep from color: band edges vs the support.** Just `[f_lo, f_hi]`, or the
   actual **set of frequencies present** (stripped of loudness)? If only the edges,
   features with the same range collapse to identical inputs → scene variety lost.
   Leaning toward keeping the support. *Open.*
4. **How free is the loudness, really?** If loudness is fully free over the available
   frequencies, does feature diversity collapse (different features → similar
   beautiful sounds)? Does the frequency **set** carry enough identity on its own?
   *Open.*
5. **Within-band masker specifics.** Band width (= a critical band?), how bands form
   around centers, the per-band softmax sharpness. "Makes sense, specifics TBD."

**Problem B (parked, dynamics):**
6. **How are within-band dynamics derived from the scene?** Fork: **scene-sourced**
   (brightness flicker, motion as you walk) vs **freely-added** aesthetic — probably
   both. Does brightness-over-time drive it (per 4.5)? *Open.*
7. **Sub-feature link (noted, undeveloped).** Parent and sub-feature (e.g. a glass vs
   the liquid in it) may have *vastly different* spectra yet share a **physical link**,
   reflected as **coupled in-band dynamics** (coupled oscillators) — i.e. the link
   lives in **Problem B**, not in shared structure (A). Connects the old [D]
   sub-feature-connectivity problem.

**Cross-cutting:**
8. **Role of the cents/tolerance dial.** From relative-period: how much mistuning the
   ear forgives (loose = lush, tight = austere). Still meaningful if harmonicity is
   part of the beauty objective — but harmonicity is now optional, so its role is TBD.
9. **Combining features + the time axis.** How per-feature structures sum into the full
   scene sound, and how the frame sequence (the macro walk) interacts with the new
   per-feature optimizer. Largely the existing across-feature budget + frame sequence,
   but the interaction with A is unspecified.
10. **The beauty-hypothesis gap.** Expectation / tension-release / surprise — the layer
    above the sensory floor — is not modeled.

---

## 6. One-line summary of the current direction

Color picks the **available frequencies** of a feature (its identity); a
**beauty-optimizer** freely sets their **loudness** to find the most beautiful
*configuration* (Problem A — constrained optimization, with a per-band softmax doing
local roughness-masking); and **brightness-over-time** will likely drive the
**movement** (Problem B). The literal extraction is never edited — it is the sample
space we *choose* beauty out of.
