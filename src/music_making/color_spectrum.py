"""The per-pixel color -> spectrum model (decision [C]).

Each pixel is one color that decomposes into MULTIPLE frequencies — a little
light-spectrum — so one pixel contributes to several frequency bins. HSV splits
the jobs:

  * **Hue -> pitch** of a tone (red -> low ... violet -> high, the EM/light-
    frequency analogy). Purples are non-spectral (= red + blue), so they become
    TWO peaks, not a fake midpoint.
  * **Saturation -> tone vs noise.** A saturated color is near-monochromatic (a
    sharp line); a washed-out color is white-ish (a broadband floor). This is
    also the old surface-roughness axis, for free.
  * **Value (brightness) -> the gain / weight** of that pixel's contribution.

Per-pixel spectrum  =  V * [ S * tone(hue)  +  (1 - S) * flat ].
The feature spectrum is the brightness-weighted sum over all its pixels; the band
[f_lo, f_hi] is the robust (saturation-weighted percentile) extent of the support.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import audio

RED_HZ = 110.0          # where pure red lands
VIOLET_HZ = 3520.0      # where pure violet lands (5 octaves above red)
H_VIOLET = 0.75         # hue fraction at the violet end; beyond it = non-spectral purples
F_LO_GRID, F_HI_GRID, NG = 60.0, 5000.0, 900   # spectrum grid (log-spaced)
TONE_SIGMA_OCT = 0.06   # tonal-peak width, in octaves
MIN_PIX = 200           # below this, percentile band is unreliable -> fall back

FREQS = np.geomspace(F_LO_GRID, F_HI_GRID, NG)


def _rgb_to_hsv(rgb: np.ndarray):
    """Vectorized RGB[0,1] -> (H[0,1), S[0,1], V[0,1])."""
    rgb = np.asarray(rgb, dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = mx - mn
    v = mx
    s = np.where(mx > 1e-12, d / np.where(mx > 1e-12, mx, 1.0), 0.0)
    nz = d > 1e-12
    dd = np.where(nz, d, 1.0)
    h = np.zeros_like(mx)
    ir = (mx == r) & nz
    ig = (mx == g) & nz & ~ir
    ib = (mx == b) & nz & ~ir & ~ig
    h = np.where(ir, ((g - b) / dd) % 6.0, h)
    h = np.where(ig, ((b - r) / dd) + 2.0, h)
    h = np.where(ib, ((r - g) / dd) + 4.0, h)
    return h / 6.0, s, v


def _hue_freq(h: np.ndarray) -> np.ndarray:
    """Spectral-arc hue -> tone frequency (red->RED_HZ, violet->VIOLET_HZ)."""
    frac = np.clip(h / H_VIOLET, 0.0, 1.0)
    return RED_HZ * (VIOLET_HZ / RED_HZ) ** frac


def _decompose(rgb: np.ndarray, pos: np.ndarray | None = None, seed: int = 0):
    """Shared core: pixels -> (E, (f_lo, f_hi), attribution).

    ``attribution`` retains the per-pixel -> per-bin mapping that ``feature_spectrum``
    otherwise discards, so the dynamics layer can build the per-frequency 3D
    center-of-mass c(f). If ``pos`` (N,3 positions, same order as ``rgb``) is given it is
    subsampled in lockstep and returned in the attribution. Keys:
      tone_pix  - pixel index of each tonal contribution (purples appear twice)
      tone_bin  - its FREQS bin;   tone_w - its weight (V*S share)
      flat_w    - per-pixel V*(1-S) floor weight
      pos       - the aligned positions (or None);  band_mask - chromatic-band bins
    """
    rgb = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    if pos is not None:
        pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    if len(rgb) > 20000:                       # subsample huge regions for speed
        sel = np.random.default_rng(seed).choice(len(rgb), 20000, replace=False)
        rgb = rgb[sel]
        if pos is not None:
            pos = pos[sel]
    h, s, v = _rgb_to_hsv(rgb)

    # tonal peaks: spectral pixels -> one peak; purples (h > H_VIOLET) -> red + blue
    spectral = h <= H_VIOLET
    sp, pu = np.where(spectral)[0], np.where(~spectral)[0]
    pk_f = [_hue_freq(h[spectral])]
    pk_w = [(v * s)[spectral]]
    pk_pix = [sp]
    if pu.size:
        p = (h[pu] - H_VIOLET) / (1.0 - H_VIOLET)               # 0 at violet -> 1 back to red
        vs = (v * s)[pu]
        pk_f += [np.full(pu.size, RED_HZ), np.full(pu.size, VIOLET_HZ)]
        pk_w += [vs * p, vs * (1.0 - p)]
        pk_pix += [pu, pu]
    pk_f = np.concatenate(pk_f)
    pk_w = np.concatenate(pk_w)
    pk_pix = np.concatenate(pk_pix)

    # accumulate peaks onto the log grid, then smooth into tonal bumps
    tone_bin = np.clip(np.searchsorted(FREQS, pk_f), 0, NG - 1)
    tone = np.zeros(NG)
    np.add.at(tone, tone_bin, pk_w)
    bins_per_oct = NG / np.log2(F_HI_GRID / F_LO_GRID)
    tone = gaussian_filter1d(tone, TONE_SIGMA_OCT * bins_per_oct)

    # broadband floor from desaturated pixels, spread over the chromatic band
    band = (FREQS >= RED_HZ) & (FREQS <= VIOLET_HZ)
    flat_w = v * (1.0 - s)
    flat = np.zeros(NG)
    flat[band] = flat_w.sum() / max(band.sum(), 1)

    E = tone + flat
    if E.max() > 0:
        E = E / E.max()

    f_lo, f_hi = _robust_band(pk_f, pk_w)
    attr = {"tone_pix": pk_pix, "tone_bin": tone_bin, "tone_w": pk_w,
            "flat_w": flat_w, "pos": pos, "band_mask": band}
    return E, (f_lo, f_hi), attr


def feature_spectrum(rgb: np.ndarray, seed: int = 0):
    """Pixels (N,3 in [0,1]) -> (FREQS, spectrum E, (f_lo, f_hi))."""
    E, fband, _ = _decompose(rgb, seed=seed)
    return FREQS, E, fband


def _robust_band(freqs: np.ndarray, weights: np.ndarray):
    """Saturation-weighted 5th/95th percentile extent of the tonal support — the
    outlier-aware band ends (no clamping to the rails). Falls back if too sparse."""
    w = weights.copy()
    if w.sum() <= 0 or (w > 0).sum() < MIN_PIX:
        return RED_HZ, VIOLET_HZ
    order = np.argsort(freqs)
    f, cw = freqs[order], np.cumsum(w[order])
    cw = cw / cw[-1]
    return float(np.interp(0.05, cw, f)), float(np.interp(0.95, cw, f))


def synthesize(E: np.ndarray, freqs: np.ndarray = FREQS, dur: float = 3.0,
               sr: int = audio.SR, seed: int = 0, phase: str = "aligned") -> np.ndarray:
    """Additive resynthesis of a spectrum: sum sinusoids weighted by E. Sharp
    peaks read as tones; broad regions become dense -> noise. 'The feature IS the
    spectrum.'

    phase: "aligned" (all partials in-phase, phi=0) or "random" (per-seed). Phase
    is a FUTURE KNOB -- it shapes the attack transient and the within-band
    interference pattern, but not whether distinct frequencies beat. Default
    in-phase for now."""
    keep = E > 0.01 * E.max() if E.max() > 0 else np.zeros(len(E), bool)
    f, a = freqs[keep], E[keep]
    n = int(dur * sr)
    if not len(f):
        return np.zeros(n, np.float32)
    phi = (np.zeros(len(f)) if phase == "aligned"
           else np.random.default_rng(seed).uniform(0, 2 * np.pi, len(f)))
    x = np.zeros(n, dtype=np.float64)
    for s0 in range(0, n, 8192):                                # chunk to bound memory
        e0 = min(n, s0 + 8192)
        t = np.arange(s0, e0) / sr
        x[s0:e0] = (a[:, None] * np.sin(2 * np.pi * f[:, None] * t[None, :] + phi[:, None])).sum(0)
    m = np.max(np.abs(x))
    if m > 0:
        x *= 0.6 / m
    fade = int(0.02 * sr)
    env = np.ones(n)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return (x * env).astype(np.float32)


def render_pixels(rgb: np.ndarray, dur: float = 3.0, seed: int = 0) -> np.ndarray:
    """Convenience: pixels -> audio (the full [C] pipeline)."""
    freqs, E, _ = feature_spectrum(rgb, seed=seed)
    return synthesize(E, freqs, dur=dur, seed=seed)


def spectrogram(frame_pixels: list, seed: int = 0) -> np.ndarray:
    """Per-frame color spectra stacked into a spectrogram (T, NG) — decision [E]:
    the time axis is the frame sequence, each frame is one column via [C]."""
    return np.array([feature_spectrum(px, seed=seed + i)[1] for i, px in enumerate(frame_pixels)])


def synthesize_spectrogram(E_t: np.ndarray, freqs: np.ndarray = FREQS, dur: float = 8.0,
                           sr: int = audio.SR, seed: int = 0, phase: str = "aligned") -> np.ndarray:
    """Additive resynthesis of a spectrogram: each bin's amplitude glides between
    the per-frame columns. The walk becomes the music.

    phase: "aligned" (in-phase) or "random" -- a future knob (see synthesize)."""
    E_t = np.asarray(E_t, dtype=np.float64)
    T = E_t.shape[0]
    n = int(dur * sr)
    peak = E_t.max(axis=0)
    keep = peak > 0.01 * peak.max() if peak.max() > 0 else np.zeros(len(freqs), bool)
    f, Ek = freqs[keep], E_t[:, keep]
    if not f.size:
        return np.zeros(n, np.float32)
    phi = (np.zeros(len(f)) if phase == "aligned"
           else np.random.default_rng(seed).uniform(0, 2 * np.pi, len(f)))
    col_pos = np.linspace(0, n - 1, T)
    x = np.zeros(n, dtype=np.float64)
    for s0 in range(0, n, 8192):                         # chunk to bound memory
        e0 = min(n, s0 + 8192)
        idx = np.arange(s0, e0)
        env = np.stack([np.interp(idx, col_pos, Ek[:, k]) for k in range(len(f))])
        x[s0:e0] = (env * np.sin(2 * np.pi * f[:, None] * (idx / sr)[None, :] + phi[:, None])).sum(0)
    m = np.max(np.abs(x))
    if m > 0:
        x *= 0.6 / m
    fade = int(0.03 * sr)
    win = np.ones(n)
    win[:fade] = np.linspace(0, 1, fade)
    win[-fade:] = np.linspace(1, 0, fade)
    return (x * win).astype(np.float32)
