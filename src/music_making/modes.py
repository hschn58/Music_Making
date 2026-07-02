"""Modal identity — the spectrum source (docs/modal_identity.md, Decision #6).

A feature's spectrum is its shape's eigen-ladder: what the thing would sound like
if the scene excited it. Computed ONCE per feature from a reference frame (option
b — identity is static; the dynamics scaffolding owns all time variation).

v1 geometry: the world-scaled silhouette (object-index mask, metric pixel pitch
from the Z-pass point cloud). Modes = Dirichlet eigenvalues of the 5-point
Laplacian on the mask,   f_k = c_mat sqrt(lambda_k) / 2 pi.

Two regimes, decided by the measured Weyl mode count in the audible window:
  * enumerated (peaky) — solve the first K eigenpairs; each mode is a log-f
    Gaussian of width 1/Q (material): high Q = chord, low Q = wide bands.
  * continuum (dense)  — huge features: render the mode-density envelope, a
    power law whose slope comes from boundary compactness (smooth = dark rumble,
    fractal = bright hiss).

All free constants live in the gauge ledger (docs/modal_identity.md §5) — no
per-feature choices.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from . import structure
from .color_spectrum import F_HI_GRID, F_LO_GRID, FREQS, NG

# --- gauge ledger (docs/modal_identity.md §5) ---
C_WAVE = 287.0          # m/s: a 1 m disk of reference material rings at ~220 Hz
F_SHELF = 2800.0        # comfort shelf corner (rolls off the harsh 2-5 kHz band)
P_DECAY = 0.5           # mode amplitude prior a_k = (f_k/f_1)^-P_DECAY
N_ENUM = 600            # Weyl-count threshold (below the shelf): enumerate vs continuum
K_MAX = 96              # eigensolve cap; the decay prior quiets the crowded top anyway
ALPHA_LO, ALPHA_HI = 0.3, 1.5   # continuum slope range, mapped from compactness

# material class -> (wave-speed factor, Q). Physics per CLASS, shared by every
# feature of the class (Blender material / future perception layer supplies it).
MATERIALS = {
    "crystal": (1.8, 90.0), "ceramic": (1.6, 60.0), "stone": (1.5, 40.0),
    "wood": (1.0, 20.0), "petal": (0.9, 8.0), "soil": (0.8, 6.0),
    "grass": (0.9, 5.0), "foliage": (0.9, 5.0), "water": (0.6, 2.5),
    "fire": (0.8, 1.8),
}

EPS = 1e-9
_LNF = np.log(FREQS)


def _shelf(f):
    """Comfort shelf: unity below the corner, rolling energy off the harsh band."""
    return 1.0 / (1.0 + (f / F_SHELF) ** 4)


def _pixel_pitch(mask, pos):
    """Metric size of one mask pixel: median 3D distance between horizontally
    adjacent in-mask pixels (the Z-pass point cloud carries the world scale)."""
    a = mask[:, :-1] & mask[:, 1:]
    d = np.linalg.norm(pos[:, 1:][a] - pos[:, :-1][a], axis=-1)
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if d.size else 0.0


def _perimeter(mask, h):
    """Boundary length: count of exposed pixel edges * pitch."""
    up = np.diff(mask.astype(np.int8), axis=0) != 0
    lr = np.diff(mask.astype(np.int8), axis=1) != 0
    edge = up.sum() + lr.sum() + mask[0].sum() + mask[-1].sum() \
        + mask[:, 0].sum() + mask[:, -1].sum()
    return float(edge) * h


def _eigenmodes(mask, h, k):
    """First k Dirichlet eigenpairs of the 5-point Laplacian on the mask.
    Returns (lambdas [1/m^2], psi2 (k, H, W) mode energy maps |psi|^2)."""
    H, W = mask.shape
    ids = -np.ones((H, W), dtype=int)
    ids[mask] = np.arange(int(mask.sum()))
    n = int(mask.sum())
    rows, cols, vals = [np.arange(n)], [np.arange(n)], [np.full(n, 4.0)]
    for dr, dc in ((0, 1), (1, 0)):
        a = mask & np.roll(mask, (-dr, -dc), (0, 1))
        if dr:
            a[-1, :] = False
        else:
            a[:, -1] = False
        i, j = ids[a], ids[np.roll(a, (dr, dc), (0, 1))]
        rows += [i, j]
        cols += [j, i]
        vals += [np.full(i.size, -1.0)] * 2
    L = sparse.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, n)) / h ** 2
    k = max(1, min(k, n - 2))
    lam, vec = eigsh(L, k=k, sigma=0, which="LM")
    order = np.argsort(lam)
    lam, vec = lam[order], vec[:, order]
    psi2 = np.zeros((k, H, W))
    psi2[:, mask] = (vec.T ** 2)
    return np.maximum(lam, EPS), psi2


def identity(mask, pos, material, f_ceil=F_HI_GRID):
    """One feature's static modal identity (docs/modal_identity.md §2-3).

    mask (H,W bool) + pos (H,W,3 camera-space) from the REFERENCE frame (where the
    feature is largest); material = a MATERIALS class name. Returns a dict:
      E (NG)       static spectral envelope on the FREQS grid (roughness-resolved)
      bands        static (lo, hi, peak) partition of E
      fk (K)       mode frequencies (continuum regime: one pseudo-mode)
      contrib      (K, NG) per-mode contribution to each bin (for r(f) blending)
      psi          (K, hb, wb) mode energy maps on the mask bbox (for per-frame COMs)
      bbox         (r0, r1, c0, c1) reference bbox, regime, h, area
    """
    wave, Q = MATERIALS[material]
    c_mat = C_WAVE * wave
    h = _pixel_pitch(mask, pos)
    npix = int(mask.sum())
    if h <= 0 or npix < 16:
        return None
    area = npix * h ** 2
    per = _perimeter(mask, h)
    compact = min(1.0, 4.0 * np.pi * area / max(per ** 2, EPS))
    # regime decision: mode count below the comfort SHELF (modes above it are
    # rolled off anyway, so they shouldn't push a ringable object into noise)
    n_weyl = area * (2 * np.pi * F_SHELF / c_mat) ** 2 / (4 * np.pi)

    rs, cs = np.where(mask)
    r0, r1, c0, c1 = rs.min(), rs.max() + 1, cs.min(), cs.max() + 1
    sub = mask[r0:r1, c0:c1]

    if n_weyl <= N_ENUM:                                   # --- enumerated / peaky ---
        lam, psi2 = _eigenmodes(sub, h, k=min(int(np.ceil(n_weyl)) + 8, K_MAX))
        fk = c_mat * np.sqrt(lam) / (2 * np.pi)
        keep = (fk >= F_LO_GRID) & (fk <= f_ceil)
        if not keep.any():                                 # everything infra/ultra: keep the
            keep = np.zeros(fk.size, bool)                 # closest mode, honest and faint
            keep[int(np.argmin(np.abs(np.log(fk / np.sqrt(F_LO_GRID * f_ceil)))))] = True
        fk, psi2 = fk[keep], psi2[keep]
        amp = (fk / fk[0]) ** (-P_DECAY) * _shelf(fk)
        sig_ln = 1.0 / (2.0 * Q)                           # fractional bandwidth 1/Q
        contrib = amp[:, None] * np.exp(-(_LNF[None, :] - np.log(fk)[:, None]) ** 2
                                        / (2 * sig_ln ** 2))
        regime = "modes"
    else:                                                  # --- continuum / dense ---
        alpha = ALPHA_LO + (ALPHA_HI - ALPHA_LO) * compact
        env = (FREQS / F_LO_GRID) ** (-alpha) * _shelf(FREQS)
        fk = np.array([np.exp((_LNF * env).sum() / env.sum())])   # spectral centroid label
        contrib = env[None, :]
        psi2 = (sub / max(npix, 1)).astype(float)[None]
        regime = "continuum"

    E = contrib.sum(0)
    if E.max() > 0:
        E = E / E.max()
    E = structure.resolve_roughness(E)                     # Decision #5, once, static
    bands = structure.bands(E)
    if not bands and E.max() > 0:
        sup = np.where(E > 0.01 * E.max())[0]
        bands = [(int(sup[0]), int(sup[-1]), int(sup[E[sup].argmax()]))]
    return dict(E=E, bands=bands, fk=fk, contrib=contrib, psi=psi2,
                bbox=(int(r0), int(r1), int(c0), int(c1)), regime=regime,
                h=h, area=area, compact=compact, n_weyl=n_weyl, Q=Q)
