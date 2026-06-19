"""Consonance scoring: how good a combination of frequencies sounds, on two axes.

axis 1  ROUGHNESS         - beating between partials inside a critical band
                            (Sethares / Plomp-Levelt). Low = smooth.
axis 2  RELATIVE PERIOD   - how many cycles of the lowest tone pass before the whole
                            waveform p(t) repeats (= lcm of the rationalized
                            frequency ratios). Short = deeply periodic, "locks in".

Good-sounding = low roughness AND short relative period. This is the computable
core of the "perspective = filter for the good" idea: score the spectrum a
perspective produces, and prefer perspectives that stay in the lower-left.

Math write-up: docs/relative_period.md.  Discrete-sound dataset: scripts/.
"""

import math

import numpy as np
from scipy.signal import find_peaks

from .color_spectrum import FREQS

# Sethares dissonance-curve constants (Sethares, 1993).
_B1, _B2 = 3.5, 5.75
_DSTAR, _S1, _S2 = 0.24, 0.0207, 18.96


def roughness(freqs, amps) -> float:
    """Sethares sensory dissonance summed over all partial pairs, amplitude-
    normalized so loudness doesn't dominate. Higher = rougher."""
    f = np.asarray(freqs, float)
    a = np.asarray(amps, float)
    if f.size < 2:
        return 0.0
    order = np.argsort(f)
    f, a = f[order], a[order]
    total = 0.0
    for i in range(f.size - 1):
        d = f[i + 1:] - f[i]                       # gaps to all higher partials
        s = _DSTAR / (_S1 * f[i] + _S2)            # critical-band scaling at lower freq
        total += float(np.sum(a[i] * a[i + 1:]
                              * (np.exp(-_B1 * s * d) - np.exp(-_B2 * s * d))))
    return total / (a.sum() ** 2 + 1e-12)


def _simplest_ratio(r: float, cents_tol: float, qcap: int):
    """Smallest-denominator fraction p/q within cents_tol of r (the ear rounding a
    near-miss to a simple ratio). Returns q, or None if nothing within tolerance."""
    for q in range(1, qcap + 1):
        p = round(r * q)
        if p <= 0:
            continue
        if abs(1200.0 * math.log2(r / (p / q))) <= cents_tol:
            return q
    return None


def relative_period(freqs, cents_tol: float = 15.0, qcap: int = 1000) -> float:
    """Cycles of the lowest tone before p(t) repeats = lcm of the rationalized
    denominators. `cents_tol` is how much mistuning the ear forgives -- the
    aesthetic dial (loose -> lush/fusing, tight -> austere). Returns inf if any
    ratio has no simple rational within tolerance (effectively aperiodic / dead)."""
    f = np.sort(np.asarray(freqs, float))
    f = f[f > 0]
    if f.size == 0:
        return math.inf
    dens = []
    for fi in f:
        q = _simplest_ratio(fi / f[0], cents_tol, qcap)
        if q is None:
            return math.inf
        dens.append(q)
    return float(math.lcm(*dens))


def peak_partials(E, freqs=FREQS, rel_height: float = 0.05, max_peaks: int = 24):
    """Reduce a dense spectrum (e.g. a color->spectrum histogram) to discrete
    partials: prominent local maxima, keeping the top `max_peaks` by amplitude."""
    E = np.asarray(E, float)
    if E.max() <= 0:
        return np.array([]), np.array([])
    idx, _ = find_peaks(E, height=rel_height * E.max())
    if idx.size == 0:
        idx = np.array([int(np.argmax(E))])
    if idx.size > max_peaks:
        idx = idx[np.argsort(E[idx])[::-1][:max_peaks]]
    idx = np.sort(idx)
    return np.asarray(freqs)[idx], E[idx]


def consonance_axes(E, freqs=FREQS, cents_tol: float = 15.0) -> dict:
    """Score a dense spectrum on both axes via its prominent partials."""
    pf, pa = peak_partials(E, freqs)
    rp = relative_period(pf, cents_tol)
    return {
        "n_partials": int(pf.size),
        "roughness": roughness(pf, pa),
        "relative_period": rp,
        "log_relative_period": math.log10(rp) if math.isfinite(rp) else math.inf,
    }
