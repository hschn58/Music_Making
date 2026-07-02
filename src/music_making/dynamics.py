"""Viewer-coupled dynamics — the gain model in ONE place (docs/viewer_coupled_dynamics.md).

For each frequency f (which belongs to exactly one band of one feature — the partition
invariant), the gain sent to the synth is

    G(F(i,t), f) = A(F) . A(B) . sigma(f)            (feature x band x within-band share)

with a per-bin propagation-delay phase  Phi(f) = 2 pi f r(f) / c_snd.

The geometry that drives all three comes from one quantity: the per-frequency 3D
center-of-mass c(f) and its viewer distance r(f). Working in CAMERA space, the viewer
sits at the origin and looks down +z, so for a pixel of camera-axis depth z at normalized
offset (dx, dy) its 3D position is z*(dx, dy, 1), r = ||pos||, and the gaze cosine of any
COM is simply c_z / ||c||.

Public entry point: ``measure_frame`` -> per feature {G(f), Phi(f), A_F} for one frame.
The driver glides G(f, t) across frames per-bin (continuity rides the fixed FREQS grid;
bands are per-frame scaffolding, never tracked across frames — Decision #4).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates

from . import structure
from .color_spectrum import F_HI_GRID, F_LO_GRID, FREQS, NG, TONE_SIGMA_OCT, _decompose

C_SND = 343.0
EPS = 1e-9


def _com(w, pts):
    """Weighted 3D center of mass, or None if there is no weight."""
    wt = float(w.sum())
    if wt <= EPS:
        return None
    return (w[:, None] * pts).sum(0) / wt


def _attention(scores, counts, T):
    """Count-weighted softmax  A_X = N_X exp(s_X/T) / sum N exp(s/T)  (the N prefactor is
    the ln N count weight). Stable: shift logits by their max before exponentiating."""
    if scores.size == 0:
        return scores
    logits = scores / max(T, EPS)
    logits = logits - logits.max()
    w = counts * np.exp(logits)
    Z = w.sum()
    return w / Z if Z > 0 else np.zeros_like(w)


def _flat_com(attr):
    u = attr["flat_w"]
    return (u[:, None] * attr["pos"]).sum(0) / max(float(u.sum()), EPS)


def _bin_geometry(attr):
    """Per-bin contribution-weighted 3D COM -> viewer range r(f) over the whole FREQS grid.

    Tonal part: scatter each pixel's (weight, weight*position) into its hue-bin and smooth
    exactly like E (so c(f) tracks the smoothed envelope). Flat floor: ONE V*(1-S)-weighted
    COM, broadcast across the chromatic band. r(f)=inf where a bin has no support."""
    pos = attr["pos"]
    tp, tb, tw = attr["tone_pix"], attr["tone_bin"], attr["tone_w"]
    W = np.zeros(NG)
    Px = np.zeros(NG)
    Py = np.zeros(NG)
    Pz = np.zeros(NG)
    np.add.at(W, tb, tw)
    np.add.at(Px, tb, tw * pos[tp, 0])
    np.add.at(Py, tb, tw * pos[tp, 1])
    np.add.at(Pz, tb, tw * pos[tp, 2])
    sig = TONE_SIGMA_OCT * NG / np.log2(F_HI_GRID / F_LO_GRID)
    W, Px, Py, Pz = (gaussian_filter1d(a, sig) for a in (W, Px, Py, Pz))

    u = attr["flat_w"]
    U = float(u.sum())
    band = attr["band_mask"]
    nb = max(int(band.sum()), 1)
    fc = _flat_com(attr)
    wflat = np.where(band, U / nb, 0.0)

    Wt = W + wflat
    cx = (Px + wflat * fc[0])
    cy = (Py + wflat * fc[1])
    cz = (Pz + wflat * fc[2])
    r = np.full(NG, np.inf)
    nz = Wt > EPS
    r[nz] = np.sqrt(cx[nz] ** 2 + cy[nz] ** 2 + cz[nz] ** 2) / Wt[nz]
    return r


def _gaze_score(com, kappa_g):
    """proximity x gaze  s = k_gaze / r,  k_gaze = exp(kappa_g (cos theta - 1)), cos = c_z/||c||."""
    r = float(np.linalg.norm(com))
    cos_t = com[2] / max(r, EPS)
    return np.exp(kappa_g * (cos_t - 1.0)) / max(r, EPS)


def measure_frame(rgb, pos, idx, feature_ids, T_F=0.20, T_B=0.20,
                  KAPPA_G=2.0, K_SIGMA=20.0, min_pix=8, seed=0, roughness_pass=True):
    """One frame -> list of {fid, G(f), Phi(f), A_F}.

    rgb (H,W,3 in [0,1]), pos (H,W,3 camera-space positions; non-finite z = background),
    idx (H,W int feature labels). Knobs: T_F/T_B attention temperatures, KAPPA_G gaze
    sharpness, K_SIGMA within-band sharpness (how hard distance sharpens sigma).
    roughness_pass: run the lateral-inhibition second pass on E (Decision #5)."""
    feats = []
    for fid in feature_ids:
        m = (idx == fid) & np.isfinite(pos[..., 2])
        if int(m.sum()) < min_pix:
            continue
        E, _, attr = _decompose(rgb[m], pos[m], seed=seed + int(fid))
        if roughness_pass:                              # bring near-equal within-Bark peaks apart
            E = structure.resolve_roughness(E)
        posA = attr["pos"]
        # per-pixel total energy e_p (= V): floor weight plus its tonal contributions
        e_p = attr["flat_w"].astype(float).copy()
        np.add.at(e_p, attr["tone_pix"], attr["tone_w"])
        c_F = _com(e_p, posA)
        if c_F is None:
            continue
        bands_i = structure.bands(E)
        if not bands_i and E.max() > 0:                 # featureless wash: one wide band
            sup = np.where(E > 0.01 * E.max())[0]
            if sup.size:
                bands_i = [(int(sup[0]), int(sup[-1]), int(sup[E[sup].argmax()]))]
        if not bands_i:
            continue
        feats.append(dict(fid=fid, E=E, attr=attr, posA=posA, bands=bands_i,
                          r_bin=_bin_geometry(attr), N=int(m.sum()),
                          s_F=_gaze_score(c_F, KAPPA_G)))
    if not feats:
        return []

    A_F = _attention(np.array([f["s_F"] for f in feats]),
                     np.array([f["N"] for f in feats], float), T_F)

    out = []
    for f, a_f in zip(feats, A_F):
        E, attr, posA, r_bin = f["E"], f["attr"], f["posA"], f["r_bin"]
        tb, tw, tp = attr["tone_bin"], attr["tone_w"], attr["tone_pix"]
        fc = _flat_com(attr)
        # --- band attention A(B): one softmax over the feature's bands, by band COM ---
        s_b, n_b, spans = [], [], []
        for lo, hi, _pk in f["bands"]:
            sel = (tb >= lo) & (tb <= hi)
            if sel.any() and tw[sel].sum() > EPS:
                c_b = _com(tw[sel], posA[tp[sel]])
                nb = int(np.unique(tp[sel]).size)
            else:                                       # flat-only band -> the floor wash
                c_b, nb = fc, 1
            s_b.append(_gaze_score(c_b, KAPPA_G))
            n_b.append(max(nb, 1))
            spans.append((lo, hi))
        A_B = _attention(np.array(s_b), np.array(n_b, float), T_B)
        # --- within-band share sigma, then assemble G and Phi per bin ---
        G = np.zeros(NG)
        Phi = np.zeros(NG)
        for (lo, hi), a_b in zip(spans, A_B):
            bins = np.arange(lo, hi + 1)
            rb = np.where(np.isfinite(r_bin[bins]), r_bin[bins], 1e6)
            logits = K_SIGMA * E[bins] / np.maximum(rb, EPS)
            logits -= logits.max()
            sig = np.exp(logits)
            ssum = sig.sum()
            if ssum <= 0:
                continue
            G[bins] = a_f * a_b * (sig / ssum)
            Phi[bins] = 2 * np.pi * FREQS[bins] * np.where(np.isfinite(r_bin[bins]),
                                                           r_bin[bins], 0.0) / C_SND
        out.append(dict(fid=f["fid"], G=G, Phi=Phi, A_F=float(a_f)))
    return out


def _mode_coms(pts, mask_sub, ident, c_F):
    """Per-mode 3D COMs for THIS frame: sample each reference mode-energy map
    |psi_k|^2 at the current pixels' normalized-bbox coordinates (the identity is
    static — option b — but WHERE each mode sits is re-located every frame, so a
    mode living on rock #3 is at rock #3). Falls back to the feature COM where a
    mode has no support in view."""
    K, hb, wb = ident["psi"].shape
    rs, cs = np.where(mask_sub)
    r0, r1 = rs.min(), rs.max() + 1
    c0, c1 = cs.min(), cs.max() + 1
    u = (rs - r0) / max(r1 - r0 - 1, 1) * (hb - 1)
    v = (cs - c0) / max(c1 - c0 - 1, 1) * (wb - 1)
    coms = np.empty((K, 3))
    for k in range(K):
        w = np.maximum(map_coordinates(ident["psi"][k], [u, v], order=1), 0.0)
        c = _com(w, pts)
        coms[k] = c if c is not None else c_F
    return coms


def measure_frame_modal(pos, idx, idents, T_F=0.20, T_B=0.20,
                        KAPPA_G=2.0, K_SIGMA=20.0, min_pix=8):
    """One frame -> list of {fid, G(f), Phi(f), A_F}, with each feature's spectrum
    sourced from its STATIC modal identity (docs/modal_identity.md) instead of
    color. Same scaffolding as ``measure_frame`` — A(F).A(B).sigma over the fixed
    bands of the modal envelope E; perspective owns the gains.

    pos (H,W,3 camera-space), idx (H,W int labels), idents = {fid: modes.identity(...)}.
    """
    feats = []
    for fid, ident in idents.items():
        if ident is None:
            continue
        m = (idx == fid) & np.isfinite(pos[..., 2])
        n = int(m.sum())
        if n < min_pix:
            continue
        pts = pos[m]
        c_F = pts.mean(0)
        coms = _mode_coms(pts, m, ident, c_F)
        r_k = np.linalg.norm(coms, axis=1)
        # per-bin geometry: blend the mode COM ranges by each mode's contribution
        contrib = ident["contrib"]                          # (K, NG), static
        wsum = contrib.sum(0)
        r_bin = np.full(NG, np.inf)
        nz = wsum > EPS
        r_bin[nz] = (contrib * r_k[:, None]).sum(0)[nz] / wsum[nz]
        feats.append(dict(fid=fid, ident=ident, coms=coms, r_bin=r_bin, N=n,
                          contrib=contrib, s_F=_gaze_score(c_F, KAPPA_G)))
    if not feats:
        return []

    A_F = _attention(np.array([f["s_F"] for f in feats]),
                     np.array([f["N"] for f in feats], float), T_F)

    out = []
    for f, a_f in zip(feats, A_F):
        ident, contrib, coms, r_bin = f["ident"], f["contrib"], f["coms"], f["r_bin"]
        E = ident["E"]
        Etot = max(E.sum(), EPS)
        # --- band attention A(B) over the STATIC bands: band COM = contribution-
        # weighted blend of the (per-frame) mode COMs over the band's bins ---
        s_b, n_b, spans = [], [], []
        for lo, hi, _pk in ident["bands"]:
            wk = contrib[:, lo:hi + 1].sum(1)
            c_b = (wk[:, None] * coms).sum(0) / max(wk.sum(), EPS)
            s_b.append(_gaze_score(c_b, KAPPA_G))
            n_b.append(max(1.0, f["N"] * E[lo:hi + 1].sum() / Etot))
            spans.append((lo, hi))
        A_B = _attention(np.array(s_b), np.array(n_b), T_B)
        # --- within-band share sigma, then assemble G and Phi per bin ---
        G = np.zeros(NG)
        Phi = np.zeros(NG)
        for (lo, hi), a_b in zip(spans, A_B):
            bins = np.arange(lo, hi + 1)
            rb = np.where(np.isfinite(r_bin[bins]), r_bin[bins], 1e6)
            logits = K_SIGMA * E[bins] / np.maximum(rb, EPS)
            logits -= logits.max()
            sig = np.exp(logits)
            ssum = sig.sum()
            if ssum <= 0:
                continue
            G[bins] = a_f * a_b * (sig / ssum)
            Phi[bins] = 2 * np.pi * FREQS[bins] * np.where(np.isfinite(r_bin[bins]),
                                                           r_bin[bins], 0.0) / C_SND
        out.append(dict(fid=f["fid"], G=G, Phi=Phi, A_F=float(a_f)))
    return out
