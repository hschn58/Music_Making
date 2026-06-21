"""Problem A — the static spectral STRUCTURE of a feature in one frame.

From a feature's value-weighted amplitude envelope E(f) (the spline-smoothed
color->frequency extraction), keep the **maximum number** of local maxima that are
mutually roughness-free (>= ~1 critical band apart), and rebuild each as a Gaussian
whose width is the spread of its basin. Everything is measured on the critical-band
(Bark) ruler, so the frequency-dependence of the critical band is automatic.

This is the "broader organizational structure": color sets which frequencies and the
value envelope sets their loudness; this picks the roughness-free skeleton of peaks
and re-expresses each as a clean bump. Dynamics (Problem B) are layered on later.

See docs/design_state.md.
"""

import numpy as np
from scipy.signal import find_peaks

from .color_spectrum import FREQS


def to_bark(f):
    """Hz -> Bark (Traunmuller 1990). One Bark ~ one critical band, so a fixed
    Bark spacing is a fixed 'how close is too close' at every frequency."""
    f = np.asarray(f, dtype=float)
    return 26.81 * f / (1960.0 + f) - 0.53


def _basins(E, peaks):
    """For each peak bin, the (lo, hi) bin span of its basin = between the flanking
    local minima (or the array ends)."""
    minima, _ = find_peaks(-E)
    bounds = np.concatenate(([0], minima, [len(E) - 1]))
    los, his = [], []
    for p in peaks:
        left = bounds[bounds < p]
        right = bounds[bounds > p]
        los.append(int(left.max()) if left.size else 0)
        his.append(int(right.min()) if right.size else len(E) - 1)
    return los, his


def structure(E, freqs=FREQS, spacing_bark=1.0, prominence=1e-3,
              sigma_floor_bark=0.1, sigma_max_bark=None, confine_to_basin=True):
    """Reduce a dense amplitude envelope to its roughness-free structure.

    Returns (E_struct, peaks):
      * E_struct - the reconstructed spectrum, a sum of Gaussians (on the Bark
        axis) at the surviving pitches, same shape as E.
      * peaks - list of {f_hz, amp, sigma_bark, bin}, low frequency first.

    spacing_bark    : min center-to-center distance kept (~1 critical band).
    prominence      : tiny floor (x max) to ignore numerical ripples; the input
                      envelope's smoothness already handles real peak definition.
    sigma_floor_bark: minimum Gaussian width, to avoid degenerate spikes.
    confine_to_basin: zero each Gaussian outside its own basin. Basins are
                      disjoint, so the bands cannot overlap -- this prevents the
                      wide Gaussian tails of neighbouring bands from bleeding into
                      each other (which would re-introduce cross-band beating and
                      undo the center-spacing's roughness guarantee).
    """
    E = np.asarray(E, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    bark = to_bark(freqs)

    cand, _ = find_peaks(E, prominence=prominence * (E.max() + 1e-12))
    if cand.size == 0:
        return np.zeros_like(E), []

    # Greedy max-count selection on the Bark ruler: sweep low->high, keep a peak
    # whenever it is >= spacing_bark from the last one kept. Raw count, amplitude-
    # blind. (cand is already ascending in frequency, hence in Bark.)
    kept = []
    last = -np.inf
    for p in cand:
        if bark[p] - last >= spacing_bark:
            kept.append(p)
            last = bark[p]

    los, his = _basins(E, kept)
    peaks = []
    E_struct = np.zeros_like(E)
    for p, lo, hi in zip(kept, los, his):
        seg = slice(lo, hi + 1)
        w = E[seg]
        wsum = w.sum() + 1e-12
        center = bark[p]
        sigma = np.sqrt((w * (bark[seg] - center) ** 2).sum() / wsum)
        sigma = max(float(sigma), sigma_floor_bark)
        if sigma_max_bark is not None:
            sigma = min(sigma, float(sigma_max_bark))     # sharpen the drop-off
        amp = float(E[p])
        g = amp * np.exp(-0.5 * ((bark - center) / sigma) ** 2)
        if confine_to_basin:
            g[:lo] = 0.0
            g[hi + 1:] = 0.0
        E_struct += g
        peaks.append({"f_hz": float(freqs[p]), "amp": amp,
                      "sigma_bark": sigma, "bin": int(p)})
    return E_struct, peaks


def bands(E, freqs=FREQS, spacing_bark=1.0, prominence=0.02):
    """The per-frame band partition for the dynamics layer (docs/viewer_coupled_dynamics.md
    §5, Decision #4): the roughness-free basins of E, recomputed each frame. Returns a list
    of (lo_bin, hi_bin, peak_bin), low frequency first. Unlike ``structure`` it emits no
    Gaussian/amplitude (those become A(B) and sigma downstream); it only carves the axis
    into basins. The ``prominence`` floor is the hysteresis that stops shallow wobbles from
    spawning ephemeral bands (which would warble A(B) frame-to-frame)."""
    E = np.asarray(E, dtype=float)
    bark = to_bark(freqs)
    cand, _ = find_peaks(E, prominence=prominence * (E.max() + 1e-12))
    if cand.size == 0:
        return []
    kept, last = [], -np.inf
    for p in cand:
        if bark[p] - last >= spacing_bark:
            kept.append(p)
            last = bark[p]
    los, his = _basins(E, kept)
    return [(int(lo), int(hi), int(p)) for p, lo, hi in zip(kept, los, his)]


def resolve_roughness(E, freqs=FREQS, max_supp_db=15.0, shimmer_hz=10.0,
                      within_bark=1.0, prominence=0.02):
    """Second-pass lateral inhibition (docs/viewer_coupled_dynamics.md §10, Decision #5).

    Where two near-equal local maxima of the density fall within ~1 critical band at a
    ROUGH delta-f (not slow shimmer), push the lesser one DOWN -- harder the closer the
    pair is to equal -- so masking takes over and the equal-energy beat collapses to one
    clean lead tone. Because measured data is never *exactly* equal there is always a
    slightly-lesser peak; we just amplify that existing asymmetry (no tie-break needed).

    Untouched: separated peaks (>1 Bark, already clean), smooth/noise regions (no isolated
    maxima), and slow shimmer (gap < ``shimmer_hz``, kept because it's pretty). Energy is
    not moved here -- the per-band ``sigma`` renormalization downstream redistributes the
    freed share to the survivors, so loudness is preserved (the pair fuses).

    The lesser peak's basin is scaled by 10**(-max_supp_db*rho/20), rho = a_lo/a_hi.
    """
    E = np.asarray(E, dtype=float).copy()
    if E.max() <= 0:
        return E
    bark = to_bark(freqs)
    peaks, _ = find_peaks(E, prominence=prominence * E.max())
    if peaks.size < 2:
        return E
    los, his = _basins(E, list(peaks))
    amp = E[peaks].copy()                                   # winners judged on original amps
    for a, p in enumerate(peaks):
        near = (np.abs(bark[peaks] - bark[p]) <= within_bark)
        near[a] = False
        if not near.any():
            continue
        b = int(np.argmax(np.where(near, amp, -np.inf)))    # loudest competitor within a Bark
        if amp[b] <= amp[a]:
            continue                                        # this peak is the local winner
        if abs(freqs[p] - freqs[peaks[b]]) < shimmer_hz:
            continue                                        # slow beat = shimmer, keep it
        rho = amp[a] / amp[b]                               # in (0,1]; ~1 = the rough case
        supp = 10.0 ** (-max_supp_db * rho / 20.0)
        E[los[a]:his[a] + 1] *= supp
    return E
