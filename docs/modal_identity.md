# Modal identity — the spectrum source (Problem A, reworked)

**Status: specced + built (v1).** Replaces the color→frequency channel as the source of
each feature's spectrum, per the empirically proven color ceiling (a color histogram can
make a tone or a wash, never a chord — `examples/walk_complex/README.md`). The
viewer-coupled dynamics scaffolding (`docs/viewer_coupled_dynamics.md`) is unchanged and
consumes this layer's output.

---

## 1. Decision #6 — the identity / encounter split (option b)

> **What a feature IS is static; how it is ENCOUNTERED is dynamic.**

- **Identity** — the feature's mode spectrum (its chord, or its noise character) is
  computed **once per feature** from its geometry + material, and never changes during
  the piece. A bell's pitch does not change as you circle it.
- **Encounter** — *all* time-variation comes from the existing dynamics scaffolding:
  attention `A(F)·A(B)`, within-band sharpness `σ` (distance still sharpens near /
  washes far), and phase `Φ`. The bands are fixed; **perspective owns the gains.**

The rejected alternative (recompute modes from the visible patch every frame) was more
radically perspective-faithful but destroys chord identity and adds occlusion seams.

## 2. Identity = the shape's eigen-ladder

The physical question: *what would this thing sound like if the scene excited it?*
The answer for any bounded shape is its Laplacian eigenvalue ladder.

**v1 approximation of the 3D profile:** the feature's **world-scaled silhouette**
(object-index mask, metric pixel pitch `h` measured from the Z-pass point cloud), taken
from the **reference frame** = the frame where the feature is largest. Modes = Dirichlet
eigenvalues of the 5-point Laplacian on the mask (the drum). Roadmap, not built:
Laplace–Beltrami on the depth-relief shell, then a closed-volume estimate.

$$ f_k = \frac{c_{\text{mat}}\sqrt{\lambda_k}}{2\pi}, \qquad
   c_{\text{mat}} = C_{\text{WAVE}} \cdot (\text{material wave factor}) $$

What falls out with no extra rules:

- **Size → register**: $\lambda \propto 1/L^2$, so the same shape twice as big rings an
  octave lower. (The locked size→pitch decision, now a theorem.)
- **Boundary → density**: Weyl's law $N(\lambda) \approx A\lambda/4\pi$ plus a boundary
  correction — rough/fractal boundaries genuinely densify the mode tail (§10 of the
  dynamics doc, now literal).
- **Relatedness**: one shape determines the whole ladder — the discrete, ordered,
  inharmonic-but-related partials color could never produce.
- **Multi-part features** (six conifers, scattered blossoms share one index): the
  union's spectrum is the union of the parts' spectra — near-identical parts give
  near-degenerate mode clusters = a natural **chorus**, and each mode *localizes on one
  part*, so mode COMs genuinely differ in space.

## 3. Two regimes, one measured criterion

The audible-window mode count $N_{\text{aud}} = A\,\lambda(f_{\text{ceil}})/4\pi$
(Weyl) decides the rendering — **measured, not chosen**:

- **Enumerated (peaky) regime** — $N_{\text{aud}} \le N_{\text{ENUM}}$: solve the first
  $K$ eigenpairs (`scipy.sparse.linalg.eigsh`, shift-invert). Each mode becomes a
  log-frequency Gaussian of width set by the material's **Q**: high Q (crystal, stone)
  → ringing partials, a **chord**; low Q → the same ladder overdamps into wide bands.
  Amplitude prior $a_k = (f_k/f_1)^{-p}$ with a comfort shelf.
- **Continuum (dense) regime** — huge features (ground, water): thousands of audible
  modes; enumerating is meaningless, so render the **envelope of the mode density**: a
  power law $E(f) \propto f^{-\alpha}$ whose slope comes from the boundary's
  **compactness** $c_4 = 4\pi A/P^2$ — smooth boundary (water) → steep → dark rumble;
  fractal boundary (foliage) → flat → bright hiss. This *is* the noise pole, and the
  multiband low-frequency noise the density framing promised.

Both regimes emit the same object: a static envelope `E(f)` on the `FREQS` grid + a
static band partition (`structure.bands` on `E`, so overlapping low-Q modes fuse into
one band exactly as the ear fuses them). The Decision-#5 lateral-inhibition pass runs on
the modal `E` — near-equal within-Bark mode pairs (degenerate modes!) are its real
customer, as predicted.

## 4. Color's role (Decision #7)

Color **exits the pitch role entirely**. Its remaining jobs:

1. **Segmentation evidence** (the ink-in-water principle): color variation inside a
   region is either *individuating* (a distinct substance → split into two voices, each
   getting the full modal treatment on its own geometry) or *shading* (wood grain → one
   voice). Deciding which is a **recognition** judgment — the future perception layer
   (AI-assisted feature extraction). In Blender the object-index pass already is that
   layer. Red ink and black ink with identical plumes sound **identical** (confirmed).
2. **Residual timbre tilt** (deferred, small): at most a brightness tilt on the mode
   amplitudes. Not built in v1.

## 5. The gauge ledger

The irreducible free choices, all **global** — the rule:

> **No per-feature choices. All freedom lives in this table; everything else is measured.**

| constant | value | meaning / status |
|---|---|---|
| `C_WAVE` | 287 m/s | wave-speed anchor: a 1 m disk of reference material rings at ≈220 Hz. The "A=440" of the system. |
| `SLOPE` | 1.0 | octaves-per-size-octave compression (1.0 = pure physics; <1 crowds registers). The one character-laden knob — flagged. |
| comfort shelf | `1/(1+(f/2800 Hz)^4)` | rendering constraint (like a display gamut): rolls off the harsh 2–5 kHz band; extends `safety.py` (which still floors at 30 Hz). |
| `P_DECAY` | 0.5 | mode amplitude prior exponent $a_k=(f_k/f_1)^{-p}$ (excitation-neutral strike). |
| `N_ENUM` | 600 | Weyl-count threshold: enumerated ladder vs continuum envelope. Counted below the comfort shelf (modes above it are rolled off, so they must not push a ringable object into noise); eigensolve capped at `K_MAX=96` (the decay prior quiets the crowded top). |
| `ALPHA_RANGE` | 0.3 – 1.5 | continuum slope range, mapped from boundary compactness $c_4\in(0,1]$. |
| material table | see `modes.MATERIALS` | per **class** (not per feature): wave factor + Q. Physics constants, shared by every feature of the class. In Blender the class comes from the material; elsewhere, the perception layer. |
| `T_F, T_B, κ_g, k` | dynamics doc | **listener** parameters (how attention behaves), not scene interpretation. |

Perceptual note: hearing is relative (almost nobody has absolute pitch), so the global
anchor/slope choices are near-inaudible *as choices* — all relational structure
(intervals, contours, who is above whom) is measured.

## 6. Wiring

| piece | status |
|---|---|
| `modes.identity(mask, pos, material)` → `{E, bands, f_k, contrib, ψ-maps, regime}` | **new** (`src/music_making/modes.py`) |
| color→spectrum `E` as the source | **replaced** (kept for the legacy driver) |
| `A(F)`, `A(B)`, `σ`, `Φ`, roughness pass | **kept**, now fed by modal `E` (`dynamics.measure_frame_modal`) |
| per-bin geometry `r(f)` | **upgraded**: mode shapes $|\psi_k|^2$ (sampled into each frame via normalized-bbox coords) give per-mode COMs → per-bin `r` as the contribution-weighted blend. A mode living on rock #3 is *located* at rock #3, so `A(B)` swings as the gaze passes it. |
| driver | `scripts/modal_walk.py` (thin; outputs `modal_features.png`, `modal_mix.wav`) |

## 7. Open

1. Curved-shell Laplace–Beltrami (use the depth relief, not just the silhouette), then
   closed-volume approximation.
2. The perception layer: material class + individuation-vs-shading from real video.
3. Scene-intrinsic excitation (fire flicker, water ripple → striking/bowing the modes) —
   Decision #3's second dynamics source, natural next.
4. Residual color→timbre tilt.
