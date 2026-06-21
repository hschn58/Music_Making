# Viewer-coupled dynamics — Problem B architecture (draft for review)

**Status: draft.** Mathematical write-up of Henry's proposal for Problem B (dynamics),
in Henry's own notation (from the handwritten sheet, `IMG_4097.heic`), so it can be
checked against intent *before* anything is built. Nothing here is committed.
Companion: `design_state.md` (§4.2 structure vs dynamics, §4.3 softmax),
`relative_period.md` (Δf / critical-band math).

The core idea: a feature's sound **evolves in time because the viewer moves**. Each
band of a feature has its own location in the frame, so as the gaze/path drifts, bands
rise and fall *independently* — giving instrument-like timbral morphing, and (with
strong suppression) the bell/gong character where only a few partials survive.

---

## 1. Notation (from the sheet)

For features indexed $i = 1, \dots, n(t)$:

| symbol | meaning |
|--------|---------|
| $F(i,t)$ | **feature** $i$ at time $t$ |
| $f$ | a **frequency** |
| $B\big(F(i,t),\, f\big)$ | the **band** (an *array of frequencies*) that $f$ resides in, for feature $i$ at $t$ |
| $\mathrm{COM}\big(\langle \text{array},\, \text{dist}\rangle\big)$ | pixel **center-of-mass** of a pixel set; **requires 3D vector input** $(x, y, \text{depth})$, so distance is built in |
| $P\big(f, F(i,t)\big)$ | the **pixels responsible for frequency** $f$ of feature $i$ at $t$ |
| $c(f) = \mathrm{COM}\big(P(f,F(i,t))\big)$ | per-**frequency** center of mass (3D, with depth) |
| $r(f) = \lVert \mathrm{view}(t) - c(f)\rVert$ | viewer distance to frequency $f$'s pixels |
| $N(\cdot)$ | **pixel count** of a band ($N_b$) or a frequency group ($N(f)$) |
| $E(f)$ | the initial **value envelope** (color→freq amplitude) — the within-band prior |
| $A(\cdot)$ | **attention** of an input (see §2) |
| $G\big(F(i,t),\, f\big)$ | **gain** of frequency $f$ of feature $F(i,t)$ at $t$ — the value sent to the synth |
| $\Phi\big(F(i,t),\, f\big)$ | **phase** of frequency $f$ — per-bin sound-travel delay from $c(f)$ (see §3) |

### The partition invariant (the clarification on the sheet)

> **Each frequency $f$ belongs to exactly one band of exactly one feature at time $t$.**

So "the feature of $f$" and "the band of $f$" are unambiguous, and no frequency's gain
is ever assembled from two places. The set of all bands across all features therefore
**partitions** the frequency axis. This is what makes $B(F(i,t),f)$ well-defined and the
budget below telescope cleanly (§3).

---

## 2. Attention — one weighted-softmax operator, two type settings

$A$ is a **count-weighted softmax** over a competing set $S$, scoring each member $X$ by
a **proximity $\times$ gaze** score $s_X$ and weighting by its pixel count:

$$
A(X) \;=\; \frac{N_X\,\exp\!\big(s_X/T\big)}{\sum_{X' \in S} N_{X'}\,\exp\!\big(s_{X'}/T\big)},
\qquad
s_X = \frac{k_{\text{gaze}}(X)}{r_X},
\qquad
r_X = \lVert \mathrm{view}(t) - \mathrm{COM}(X)\rVert .
$$

The gaze weight is a von Mises (foveal) kernel on the angle $\theta_X$ between the gaze
direction $\hat g(t)$ and the viewer→$X$ direction:

$$
k_{\text{gaze}}(X) = \exp\!\big(\kappa_g\,(\cos\theta_X - 1)\big),
\qquad
\cos\theta_X = \hat g(t)\cdot \frac{\mathrm{COM}(X) - \mathrm{view}(t)}{\lVert \mathrm{COM}(X) - \mathrm{view}(t)\rVert}.
$$

- **Proximity $1/r_X$** = the physical pressure-amplitude falloff of a point source.
- **Gaze $k_{\text{gaze}}\in(0,1]$** = 1 when looked straight at, decaying off-axis;
  $\kappa_g$ is foveal sharpness ($\kappa_g=0$ → gaze off; large → tight spotlight, so
  looking away collapses a band to its count-baseline = the **gong** effect). Applied at
  the feature/band level only, **not** inside $\sigma$ (a band's per-frequency COMs are
  nearly co-located, so gaze is ~uniform across them).
- **Count enters as $\ln N_X$, not $N_X$** — "more pixels draw more attention" without
  the exponential exploding (a $10\times$ band adds only $\ln 10 \approx 2.3$ to its logit).
- **$T$** = temperature = sharpness of the competition: $T\to 0$ winner-take-all
  (sparse → **gong/bell**), $T\to\infty$ uniform (all survive → **chord**). Independent
  of the count weight.

Two **settings**, distinguished only by the competing set $S$:

1. **type = FEATURE** ($S$ = all features): $\;\sum_{i=1}^{n(t)} A\big(F(i,t)\big) = 1$ — budget across features.
2. **type = BAND** ($S$ = the bands of feature $i$, using each band's **band COM**):
   $\;\sum_{b\,\in\,\text{bands}(i)} A\big(B_b(F(i,t))\big) = 1$ — budget across a feature's bands.

> **Decision #1 — resolved:** band-attention normalizes **within a feature** (across its
> bands), using the **band COM** for $r_b$ and $N_b = \sum_{f\in b} N(f)$.

One operator, two granularities — unifying what my earlier draft split into "feature
gain" and "per-band gain."

---

## 3. The gain, assembled

The per-frequency gain is the product of the two attentions and a within-band share:

$$
\boxed{\;
G\big(F(i,t),\, f\big) \;=\;
\underbrace{A\big(F(i,t)\big)}_{\text{feature budget}}\;\cdot\;
\underbrace{A\big(B(F(i,t),f)\big)}_{\text{band budget}}\;\cdot\;
\underbrace{\sigma\big(f;\, B(F(i,t),f)\big)}_{\text{within-band share}}
\;}
$$

and the scene signal is the sum over every present frequency (each owned by one
feature/band, per the invariant):

$$
x(t) = \sum_{i}\;\sum_{f \,\in\, F(i,t)} G\big(F(i,t),f\big)\,\sin\!\big(2\pi f t + \Phi(F(i,t),f)\big).
$$

**The within-band share $\sigma$** (the §4.3 masker) distributes a band's budget across
its frequencies. Its **logits are the initial value envelope $E(f)$** (the color→freq
amplitude, restored here as the within-band *prior* — see note), and its **temperature
is the per-frequency distance** $r(f) = \lVert\mathrm{view}(t) - c(f)\rVert$:

$$
\sigma\big(f;\, B\big) = \frac{\exp\!\big(k\,E(f)/r(f)\big)}
{\sum_{f' \in B}\exp\!\big(k\,E(f')/r(f')\big)},
\qquad \sum_{f \in B}\sigma(f;B) = 1 .
$$

**Distance as inverse-sharpness:** a near frequency (small $r$) → $E/r$ large →
sharpened, so its loud partials win; a far frequency (large $r$) → flattened toward
$e^0$ → washes into a low floor. As the viewer moves, $r(f)$ shifts and the dominant
partial **drifts** — within-band viewer-coupling. The global constant $k$ sets *how
strongly* distance sharpens (the one free knob here).

> **Value envelope, scope (reconciles `design_state.md` §4.5).** §4.5 dropped $E$ as the
> *loudness source*. It returns only as the **within-band prior** (which frequencies in a
> band start out loud — the shape $\sigma$ sharpens). Across-feature / across-band
> loudness is still free, set by attention ($A(F), A(B)$). $E$ is a prior, not the level.

**Composition with the across-band level:** a **near band** is loud ($A(B)$ high) *and*
sharp ($\sigma$ peaked) → a clear **tone**; a **far band** is quiet *and* spread → a
**hazy wash**. One quantity (distance) unifies loudness, sharpness, and the tone↔noise
character — near = present/detailed, far = blended.

**Why the budget is clean (the payoff of the partition invariant).** Because each level
is a normalized distribution over a *different* domain, the total scene gain telescopes:

$$
\sum_i A(F_i)\!\!\sum_{b \in \text{bands}(i)}\!\! A(B_b)\!\!\sum_{f \in b}\!\sigma(f;b)
\;=\; \sum_i A(F_i)\cdot 1 \;=\; 1 .
$$

No frequency is double-counted (the invariant) and no distance is double-applied
(normalization at each level cancels the common magnitude, leaving only the *relative*
geometry — gross distance decides $A(F)$ across features, the within-feature spatial
offset decides $A(B)$ across bands). This is exactly the "distance once, then relative
offsets" accounting from the earlier draft — but it now falls out of your two
normalizations instead of being imposed.

### Phase — propagation delay from COM

Each frequency also carries a **phase**: the sound-travel delay from where it lives in
the scene to the viewer,

$$
\Phi\big(F(i,t),f\big) = 2\pi f \cdot \frac{r(f)}{c_{\text{snd}}},
\qquad
r(f) = \big\lVert \mathrm{view}(t) - c(f) \big\rVert,
\qquad c_{\text{snd}} \approx 343\ \text{m/s}.
$$

"Count how many cycles fit in the travel distance, keep the leftover." It uses the
**same per-frequency $c(f)$** that tempers $\sigma$, so depth drives the gain falloff,
the within-band sharpness, *and* the phase — one geometry, three jobs.

Phase is near-inaudible for an *isolated* steady tone, but **audible in within-band
interference** (close partials beat, and their relative phases shape the fringe
pattern) — which is exactly where **within-feature depth variation** places its phases.
So phase is computed **per-bin** (each bin's phase from the depth of *its* contributing
pixels), not just per-band: a feature spanning a depth range then gets a spread of
phases across its bins → a richer, physically-grounded shimmer. **Depth source =
Blender Z-pass (accurate per-pixel).**

Bonus that falls out: same-color pixels at different depths share a frequency but differ
in phase, so they combine by **phasor addition** — a subtle, honest interference
attenuation of that bin. (Fuller version — amplitude *and* phase both emerging from a
per-pixel phasor sum — is available later; phase-per-bin captures most of it without
disturbing the clean $A\cdot A\cdot\sigma$ gain.)

Also gives **Doppler** when the viewer moves ($d\Phi/dt = 2\pi f\,\dot r/c$) and kills
the all-aligned crest-factor spike. Not a roughness fix — phase organizes beating,
never removes it.

### The equal-close-pair ringing (noted, rare)

Softmax is **energy** masking, not roughness removal: it cleans a beat only when one
partial dominates. **Two equal-energy close partials** (yin/yang colors; many slightly
varied bright marbles in one band) defeat it. **Noted as rare, not solved here** —
identical colors don't beat (same $f$); dark / desaturated fields fall to the soft
floor; textured fields legitimately land at the **noise pole** (honest granular shimmer
via wide $\sigma$). Only a narrow zone actually rings: bright + saturated + *slight*
variation landing in the 20–40 Hz beat range at roughly equal energy. Character is set
by $\Delta f$ (slow = shimmer/pretty, fast = rough), gated by the critical band / cents
dial (`relative_period.md` §5).

---

## 4. What this produces

- **Varying frequency domination:** different bands lead at different moments as the
  gaze moves (via $A(B)$), and within a band the lead partial drifts (via $\sigma$).
- **Varying gain envelope per band:** each band breathes on its own COM distance.
- **Instrument-like timbre:** a held gaze = a sustained note with a fixed rich internal
  balance; a drifting gaze = an evolving timbre.
- **Bell / gong:** when the gaze strongly favors part of a feature, the within-feature
  budget conservation drives the unfavored bands toward silence, leaving a sparse set
  of strong partials (sharp $\sigma$) — the bell/gong skeleton.

---

## 5. Consequences for Problem A (structure)

A no longer outputs a fixed per-band amplitude or a Gaussian width, but it still
supplies the raw material the dynamics layer consumes:

- **Band amplitude becomes $A(B)$** — set by attention, time-varying.
- **Band width becomes emergent** from $\sigma$, now driven by **distance**: a near band
  sharpens (tone), a far band spreads (wash); a uniform-color patch shares a COM and
  moves together (honestly less rich). The **data-driven answer to "discrete tones vs
  wide bands"** — it falls out of the geometry, no imposed Gaussian.
- **A's outputs:** (1) roughness-free **band centers**; (2) the **chosen range** around
  each center (the basin / critical-band range — $\sigma$'s domain); (3) pixel→frequency
  **memberships** $P(f,F(i,t))$ realizing the partition invariant; (4) the initial
  **value envelope $E(f)$** as the within-band prior.

> **Decision #2 — resolved:** `structure.py` drops its Gaussian `sigma` and final `amp`,
> but **keeps** centers, ranges, memberships, and $E$. (The "chosen range" optimization —
> how far each range extends around its local max — is still to be specified.)

> **Decision #4 — band lifetime: per-frame scaffold, bin-grid continuity.** A feature's
> visible pixels change every frame (parallax, gaze, occlusion), so $E(f)$ and its
> roughness-free basins change too. Bands are therefore **recomputed every frame** — but
> they are *scaffolding*, not persistent objects. The band layer's output is a **per-bin
> gain field** $G(f,t)$; the persistent unit of continuity is the **frequency bin** (the
> fixed `FREQS` grid), not the band. So band identity is never tracked across frames —
> amplitudes glide per-bin, exactly as the current synth already glides $E$. The one seam
> (a basin splitting/merging between frames → a jump in the $A(B)$ denominator) is
> suppressed by a **prominence / hysteresis floor** so only genuinely deep minima create
> boundaries; at a real split the boundary bins carry $E\approx 0$, so $G(f,t)$ passes
> through continuously. Same category as the equal-close-pair ringing: a narrow, mitigated
> artifact, not a structural flaw.

---

## 6. Boundary: viewer motion vs scene motion

The only source of time-variation above is **viewer motion** (gaze/path through
$\mathrm{view}(t)$ inside $A$). For a held gaze on a static scene, nothing evolves —
musically fine (held gaze = sustained note). **Scene-intrinsic motion** (objects
moving, brightness flicker — `design_state.md` Q6 / row 7) is a **separate, additive**
dynamics source not covered here:

- **viewer-coupling** (this doc) = how *you* play the instrument;
- **scene-motion** = the instrument's own vibration.

Both are legal under the honesty rule (neither restructures A).

> **Confirm (Decision #3).** Keep the two as separate additive contributions for now?

---

## 7. Status

### Resolved
- **Gain decomposition** $G = A(F)\cdot A(B)\cdot \sigma$, with the partition invariant
  making the budget telescope. (§3)
- **Decision #1 — band-attention normalization:** within a feature's bands, via the
  **band COM** and a count-weighted softmax ($\ln N$ weight, temperature $T$). (§2)
- **Within-band $\sigma$:** softmax of the value envelope $E(f)$, temperature = the
  per-frequency distance $r(f)$; global sharpness $k$. (§3)
- **Value envelope** demoted to within-band prior (reconciles §4.5). (§3)
- **Decision #2 — Problem A:** drops `sigma`/`amp`; keeps centers, ranges, memberships,
  $E$. (§5)
- **Decision #4 — band lifetime:** bands are recomputed per frame as scaffolding; the
  persistent unit is the frequency bin. Output is a per-bin gain field $G(f,t)$; split/merge
  seam mitigated by a prominence/hysteresis floor. (§5)
- **COM 3D input = scene depth**, source = **Blender Z-pass** (per-pixel). Drives both
  gain falloff and phase.
- **Phase** = per-bin propagation delay from $c(f)$ (§3).
- **COM weighting:** contribution-weighted, **static per frame** (viewer enters only via
  $r$); the band COM does not drift recursively with attention.
- **Gaze in the score:** $s_X = k_{\text{gaze}}(X)/r_X$, von Mises kernel with foveal
  sharpness $\kappa_g$, at feature/band level only (§2).

### Still open
1. **The "chosen range"** optimization (§5) — how far each $\sigma$-range extends around
   its local max (basin vs critical-band cap).
2. **Free knobs to tune:** within-band sharpness $k$, band temperature $T$, feature
   temperature $T_F$, gaze concentration $\kappa_g$.
3. **Decision #3** — viewer-motion vs scene-motion kept separate and additive (§6).
   *(recommended, unconfirmed)*
4. **Depth acquisition** for non-Blender inputs (RGB-D / monocular) — a later swap.
5. Fuller **per-pixel phasor** synthesis (amplitude + phase together) — deferred.

---

## 8. Reference — all variables and all equations

### 8.1 Variables, in words

**Scene / inputs**
- $t$ — time (the current frame / moment along the walk).
- $i$ — feature index; features run $i = 1,\dots,n(t)$.
- $n(t)$ — number of features present at time $t$.
- $F(i,t)$ — feature $i$ at time $t$: a set of pixels, each with a color and a 3D position.
- $f$ — a frequency (Hz).
- $\mathrm{view}(t)$ — the viewer's 3D position at time $t$.
- $\hat g(t)$ — the viewer's gaze direction at time $t$ (a unit vector pointing where they look).
- $c_{\text{snd}}$ — speed of sound, $\approx 343$ m/s.

**Pixel groupings and geometry**
- $P(f, F(i,t))$ — the pixels of feature $i$ responsible for frequency $f$.
- $B(F(i,t),f)$ — the band (an array of frequencies) that $f$ belongs to in feature $i$;
  bands **partition** the frequency axis (each $f$ is in exactly one band of one feature).
- $\mathrm{COM}(\cdot)$ — 3D pixel center-of-mass of a pixel set (uses depth).
- $c(f) = \mathrm{COM}(P(f,F(i,t)))$ — the per-**frequency** center of mass.
- $X$ — a generic competing element: either a feature $F(i,t)$ or a band $B_b$.
- $\mathrm{COM}(X)$ — center of mass of element $X$ (feature COM $c_F$, or band COM $c_b$).
- $r(f) = \lVert \mathrm{view}(t) - c(f)\rVert$ — viewer distance to frequency $f$'s pixels.
- $r_X = \lVert \mathrm{view}(t) - \mathrm{COM}(X)\rVert$ — viewer distance to element $X$.
- $\theta_X$ — angle between the gaze direction $\hat g(t)$ and the viewer$\to X$ direction.

**Counts and envelope**
- $N(f)$ — number of pixels responsible for frequency $f$.
- $N_b = \sum_{f\in b} N(f)$ — pixel count of band $b$.
- $N_X$ — pixel count of element $X$ (feature or band).
- $E(f)$ — the initial **value envelope**: the color→frequency amplitude; the within-band prior.

**Attention, gain, output**
- $k_{\text{gaze}}(X)$ — gaze-alignment weight of $X$ (1 looked-at, decays off-axis).
- $s_X$ — attention score of $X$ = proximity $\times$ gaze.
- $A(X)$ — attention of $X$: a count-weighted softmax over its competing set.
- $A(F(i,t))$ — feature attention (competing set = all features).
- $A(B(F(i,t),f))$ — band attention (competing set = the feature's bands).
- $\sigma(f;B)$ — within-band share of frequency $f$ among its band's frequencies.
- $G(F(i,t),f)$ — final gain of frequency $f$ (sent to the synth).
- $\Phi(F(i,t),f)$ — phase of frequency $f$.
- $x(t)$ — the output audio signal.

**Parameters (knobs)**
- $T$ — attention temperature (softmax sharpness: sparse↔chord). $T_F$ if the feature
  level uses a separate value.
- $\kappa_g$ — gaze concentration / foveal sharpness ($\kappa_g=0$ = gaze off).
- $k$ — within-band sharpness constant (how strongly distance sharpens $\sigma$).

### 8.2 Equations

Gaze cosine and weight:
$$
\cos\theta_X = \hat g(t)\cdot \frac{\mathrm{COM}(X) - \mathrm{view}(t)}{\lVert \mathrm{COM}(X) - \mathrm{view}(t)\rVert},
\qquad
k_{\text{gaze}}(X) = \exp\!\big(\kappa_g\,(\cos\theta_X - 1)\big).
$$

Attention score and attention (count-weighted softmax over competing set $S$):
$$
s_X = \frac{k_{\text{gaze}}(X)}{r_X},
\qquad
A(X) = \frac{N_X\,\exp(s_X/T)}{\sum_{X'\in S} N_{X'}\,\exp(s_{X'}/T)} .
$$

Feature attention (set $S=\{F(j,t)\}_{j=1}^{n(t)}$) and band attention (set $S=$ bands of feature $i$):
$$
\sum_{i=1}^{n(t)} A\big(F(i,t)\big) = 1,
\qquad
\sum_{b\,\in\,\text{bands}(i)} A\big(B_b(F(i,t))\big) = 1,
\qquad N_b = \sum_{f\in b} N(f).
$$

Within-band share (value envelope as logits, per-frequency distance as temperature):
$$
\sigma(f;B) = \frac{\exp\!\big(k\,E(f)/r(f)\big)}{\sum_{f'\in B}\exp\!\big(k\,E(f')/r(f')\big)},
\qquad
\sum_{f\in B}\sigma(f;B) = 1 .
$$

Gain:
$$
G\big(F(i,t),f\big) = A\big(F(i,t)\big)\cdot A\big(B(F(i,t),f)\big)\cdot \sigma\big(f;B(F(i,t),f)\big).
$$

Phase (per-bin propagation delay):
$$
\Phi\big(F(i,t),f\big) = 2\pi f\,\frac{r(f)}{c_{\text{snd}}} .
$$

Output signal:
$$
x(t) = \sum_{i}\;\sum_{f\,\in\,F(i,t)} G\big(F(i,t),f\big)\,\sin\!\big(2\pi f t + \Phi(F(i,t),f)\big).
$$

Budget telescoping (a property, from the partition invariant + the three normalizations):
$$
\sum_i A(F_i)\sum_{b\in\text{bands}(i)} A(B_b)\sum_{f\in b}\sigma(f;b) = 1 .
$$

---

## 9. Implementation map (planned — not yet built)

The model gets **one home**, `src/music_making/dynamics.py`, computing $G(f,t)$ and
$\Phi(f,t)$ for a frame. The two scripts (`blender_feature_test.py`, `image_listen.py`)
become thin drivers — Blender supplies depth + object-index; stills fall back to flat
depth + k-means features. This removes the duplicated $A(F)$ logic now copied in both.

**Current state:** only $A(F)$ is wired (duplicated in both scripts); $A(B)$, $\sigma$,
the per-bin attribution $P(f)$, and per-bin $\Phi$ are unbuilt. The whole missing layer
is gated on **per-bin pixel attribution**, which `feature_spectrum` currently discards.

Traceability — each feature, its planned home in `dynamics.py`, and where it lives
**today** (file:line, snapshot of the current tree; ✅ built · ⚠️ partial · ❌ unbuilt):

| feature / equation | planned home | implemented at (file:line) | status |
|---|---|---|---|
| $E_i(f)$ value envelope (hue→pitch, sat→tone/noise, value→gain) | `color_spectrum` | `src/music_making/color_spectrum.py:57–60, 63–100` | ✅ |
| band ends $[f_{lo},f_{hi}]$ | `color_spectrum` | `src/music_making/color_spectrum.py:103–112` | ✅ |
| critical-band (Bark) ruler | `structure` | `src/music_making/structure.py:22–26` | ✅ |
| bands = roughness-free basins (per-frame) | `structure.bands(E)` | `src/music_making/structure.py:29–40, 73–78` | ⚠️ emits Gaussian, not wired; needs prominence floor |
| per-pixel range $r$ / eccentricity | `capture` | `src/music_making/capture.py:29–36, 39–45` | ✅ |
| **$P(f)\to c(f), N(f), r(f)$ per-bin attribution** | `dynamics.attribution` | — | ❌ gating piece |
| gaze $k_{\text{gaze}}$ (von Mises) | `dynamics.attention` | `scripts/blender_feature_test.py:97–98`; `scripts/image_listen.py:69–70` | ⚠️ duplicated; gaze axis = camera forward |
| $A(F)$ feature attention (count·gaze/$r$ softmax) | `dynamics.attention(features)` | `scripts/blender_feature_test.py:99–107`; `scripts/image_listen.py:71–75` | ⚠️ duplicated |
| $A(B)$ band attention (band COM) | `dynamics.attention(bands)` | — | ❌ new |
| $\sigma(f;B)$ within-band share | `dynamics.within_band(E, r)` | — | ❌ new |
| $\Phi$ phase (per-bin propagation delay) | `dynamics.gain` | `scripts/blender_feature_test.py:159` | ⚠️ per-feature mean only |
| $G=A(F)\,A(B)\,\sigma$ assembly | `dynamics.gain` | — | ❌ new |
| additive synth, gliding per-bin env → $x(t)$ | synth | `src/music_making/color_spectrum.py:159–189`; `scripts/blender_feature_test.py:111–126`; `scripts/image_listen.py:79–98` | ✅ |

**Per-bin attribution, cheaply** (avoids a dense pixel×bin matrix):
- *Tonal part* — each pixel maps to a single hue-bin (two for purples), so scatter the
  pixel's $V\!\cdot\!S$ weight **and its 3D position** into that bin: gives the tonal
  $\mathrm{COM}(f)$ and $N(f)$ by `np.add.at`, like the existing `tone` accumulation.
- *Flat floor* — every desaturated pixel spreads uniformly across the chromatic band, so
  there is **one** $V\!\cdot\!(1\!-\!S)$-weighted COM, broadcast to all band bins.
- $c(f)$ = the tone/floor-magnitude-weighted blend of the two; $r(f)=\lVert\mathrm{view}-c(f)\rVert$.

**Band COM** for $A(B)$ = the $N(f)$-weighted aggregate of $c(f)$ over $f\in b$
(equivalently the COM of the band's contributing pixels); $N_b=\sum_{f\in b}N(f)$.

---

## 10. Spectral-density structure + the lateral-inhibition second pass

**The band-count question dissolves.** A feature's structure is a measured spectral
*density*, not a chosen count of bands. It lives on a continuum between two poles:
**peaky** (a few separated maxima → discrete tones, the bell pole) and **power-law /
fractal** (a smooth tilted continuum → ordered noise, the noise pole). The "number of
bands" is the shadow the density casts when it is peaky; in the fractal regime there is no
integer to choose, and that continuum *is* the wanted multiband low-frequency noise.
Selecting maxima of a *thin* signal (the color histogram) felt like "not extracting
beauty" because the source was thin — beauty is latent order in a *rich* source.

**Geometry of a 2D feature.** A 2D shape has at most **two independent spatial axes of
variation** (it is a function of two coordinates). A fractal does not add axes — it adds
an unbounded *mode series along* those ≤2 axes (the drum picture: modes labelled by two
indices, infinitely many, the **boundary** setting their spectrum). So **boundary
character → band structure**: smooth boundary → sparse clean modes; rough/fractal boundary
→ dense power-law spectrum; constant cross-section (rectangle) → sharp line, varying
cross-section (ellipse) → a band whose curvature is sourced from the axis lengths.
Color stays the **catalogue** (which frequencies / spectral identity); shape sets the
**structure** (count, character). *(Open: the exact boundary→density map, and whether band
centers come from color peaks or shape dimensions — parked.)*

**Roughness is a smoothness property, not a counting one.** Plomp–Levelt roughness is
spectral *contrast at the critical-band scale* — a few near-equal partials within ~1 Bark
at a rough $\Delta f$ (≈20–40 Hz). Both poles are safe (separated peaks; smooth noise);
only the bumpy middle rings. The density approach does **not** structurally prevent two
near-equal maxima from landing within a Bark — that guarantee was exactly what the old
discrete ≥1-Bark selector bought, at the cost of richness. No free lunch: the guarantee
is paid for somewhere.

> **Decision #5 — density + a lateral-inhibition second pass.** Use the spectral-density
> structure (richness), then correct roughness in a second pass. Where two near-equal
> local maxima fall within ~1 Bark at a rough $\Delta f$ (not slow shimmer), **bring the
> lesser one down — harder the closer the pair is to equal.** Because nothing in measured
> data is *exactly* equal, there is always a slightly-lesser peak; the pass amplifies that
> existing asymmetry into clear dominance, so masking restores a single clean lead tone —
> no tie-break rule needed (exact equality is measure-zero). This is the **frequency-local
> softmax / lateral inhibition** idea (the global within-band $\sigma$ can't do it: equal
> logits → equal share). $\sigma$'s per-band renormalization then redistributes the freed
> share, so loudness is preserved (the pair fuses into one stronger tone). Untouched:
> separated peaks (>1 Bark), smooth continua / noise, and **slow shimmer** ($\Delta f
> \lesssim$ shimmer cutoff — kept, it's pretty).

Suppression of the lesser maximum, by amplitude ratio $\rho = a_{\text{lo}}/a_{\text{hi}}\in(0,1]$:
$$
a_{\text{lo}} \;\leftarrow\; a_{\text{lo}}\cdot 10^{-D\,\rho/20},
$$
so the drop (in dB) grows with $\rho$: hardest at near-equality, none when already masked.
Knobs: $D$ = max suppression (dB), the shimmer cutoff (Hz), and the window (≈1 Bark).

**Build status.** Implemented as `structure.resolve_roughness` and wired into
`dynamics.measure_frame` (runs on the current **color** density `E` before bands/$\sigma$).
It is forward-compatible: when the shape→density source lands, the same pass cleans it.
Caveat: on the current rail-polarized hue map the within-Bark near-equal case is rare
(the map polarizes energy onto fixed rails far apart), so the audible effect on this scene
is small — the pass is infrastructure for the shape-density richness still to come, and
de-polarizing the hue map removes an artificial source of equal pairs.
