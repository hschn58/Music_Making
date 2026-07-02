"""Walk a feature-segmented Blender capture and synthesize with MODAL IDENTITIES
(docs/modal_identity.md): each feature's spectrum is its shape's eigen-ladder,
computed once from the frame where the feature is largest (option b — identity is
static); the viewer-coupled dynamics scaffolding then focuses the gains by
perspective (A(F).A(B).sigma over the fixed bands).

    python scripts/modal_walk.py [CAPTURE_DIR] [DURATION_S] [--force]

Outputs modal_mix.wav + modal_features.png alongside the capture.
"""

import json
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import spectrogram as stft

from music_making import audio, dynamics, modes
from music_making.color_spectrum import FREQS, NG

CAP = sys.argv[1] if len(sys.argv) > 1 else "demos/walk_complex"
DUR = float(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 45.0
SR = audio.SR
OUT = CAP
MAXIDX = 16

# listener knobs (gauge ledger: these parameterize attention, not the scene)
T_F = 0.20
T_B = 0.20
KAPPA_G = 2.0
K_SIGMA = 20.0
TIME_SMOOTH = 1.5

# feature id -> (name, plot color, MATERIAL CLASS). The class is the Blender
# material's class (the perception layer's job on real video) — per docs/
# modal_identity.md §5 it indexes the shared physics table, no per-feature knobs.
FEATURES = {1: ("ground", "#5a4a2a", "soil"), 2: ("water", "#1f6fb0", "water"),
            3: ("conifers", "#1f7a2e", "foliage"), 4: ("fire", "#ff7a18", "fire"),
            5: ("red", "#c0142a", "crystal"), 6: ("yellow", "#d8c020", "crystal"),
            7: ("purple", "#8a2da0", "crystal"), 8: ("cyan", "#16b0b0", "crystal"),
            9: ("orange", "#e07a18", "crystal"), 10: ("meadow", "#d040a0", "grass"),
            11: ("magenta", "#c01080", "ceramic"), 12: ("foliage", "#5a9a20", "foliage"),
            13: ("rocks", "#888890", "stone"), 14: ("blossoms", "#e89ac8", "petal")}


def load(cap_dir):
    """Capture -> per-frame (pos, idx). Camera space: viewer at origin, +z forward."""
    meta = json.load(open(os.path.join(cap_dir, "camera.json")))
    w, h = meta["resolution"]
    fov_x, clip0, clip1 = meta["fov_x"], float(meta["clip_start"]), float(meta["clip_end"])
    fx = (w / 2.0) / np.tan(fov_x / 2.0)
    ys, xs = np.mgrid[0:h, 0:w]
    base = np.dstack([(xs - w / 2.0) / fx, (ys - h / 2.0) / fx, np.ones((h, w))])
    frames = []
    for k in range(len(meta["frames"])):
        d = cv2.imread(f"{cap_dir}/depth_{k:04d}.png", cv2.IMREAD_UNCHANGED).astype(np.float64)
        z = clip0 + d / np.iinfo(np.uint16).max * (clip1 - clip0)
        z[d / np.iinfo(np.uint16).max >= 1.0 - 1e-6] = np.inf
        with np.errstate(invalid="ignore"):
            pos = z[..., None] * base
        ix = cv2.imread(f"{cap_dir}/index_{k:04d}.png", cv2.IMREAD_UNCHANGED).astype(np.float64)
        idx = np.round(ix / np.iinfo(np.uint16).max * MAXIDX).astype(int)
        frames.append((pos, idx))
    return frames


def build_identities(frames):
    """Reference frame per feature = where it is largest; identity computed once."""
    counts = {fid: np.zeros(len(frames), int) for fid in FEATURES}
    for t, (pos, idx) in enumerate(frames):
        fin = np.isfinite(pos[..., 2])
        for fid in FEATURES:
            counts[fid][t] = int(((idx == fid) & fin).sum())
    idents = {}
    for fid, (name, _c, mat) in FEATURES.items():
        t = int(counts[fid].argmax())
        if counts[fid][t] < 16:
            continue
        pos, idx = frames[t]
        m = (idx == fid) & np.isfinite(pos[..., 2])
        ident = modes.identity(m, pos, mat)
        if ident is None:
            continue
        idents[fid] = ident
        fk = ident["fk"]
        lead = ", ".join(f"{f:.0f}" for f in fk[:6]) + ("…" if fk.size > 6 else "")
        print(f"  {name:9s} [{mat:8s}] {ident['regime']:9s} ref@{t:02d} "
              f"{ident['area']:7.1f} m²  compact={ident['compact']:.2f}  "
              f"N_weyl={ident['n_weyl']:8.0f}  bands={len(ident['bands'])}  f_k = {lead} Hz")
    return idents


def measure(frames, idents):
    nT = len(frames)
    spec = {fid: np.zeros((nT, NG)) for fid in idents}
    phase = {fid: np.zeros((nT, NG)) for fid in idents}
    att = {fid: np.zeros(nT) for fid in idents}
    for t, (pos, idx) in enumerate(frames):
        for r in dynamics.measure_frame_modal(pos, idx, idents, T_F=T_F, T_B=T_B,
                                              KAPPA_G=KAPPA_G, K_SIGMA=K_SIGMA):
            spec[r["fid"]][t] = r["G"]
            phase[r["fid"]][t] = r["Phi"]
            att[r["fid"]][t] = r["A_F"]
    return spec, phase, att, nT


def synth(E_t, dur, phi):
    """Additive resynthesis, amplitudes gliding between frames; per-freq phase phi."""
    n = int(dur * SR)
    peak = E_t.max(axis=0)
    keep = peak > 0.01 * peak.max() if peak.max() > 0 else np.zeros(len(FREQS), bool)
    f, Ek, ph = FREQS[keep], E_t[:, keep], phi[keep]
    x = np.zeros(n)
    if not f.size:
        return x.astype(np.float32)
    col_pos = np.linspace(0, n - 1, E_t.shape[0])
    for s0 in range(0, n, 8192):
        e0 = min(n, s0 + 8192)
        idx = np.arange(s0, e0)
        env = np.stack([np.interp(idx, col_pos, Ek[:, j]) for j in range(len(f))])
        x[s0:e0] = (env * np.sin(2 * np.pi * f[:, None] * (idx / SR)[None, :] + ph[:, None])).sum(0)
    return x.astype(np.float32)


def spectro(ax, x, title, color):
    fr, tt, S = stft(x, fs=SR, nperseg=2048, noverlap=1024)
    band = fr <= 4200
    fr, S = fr[band], S[band]
    Sdb = 10 * np.log10(S + 1e-12)
    ax.pcolormesh(tt, fr, Sdb, shading="auto", cmap="magma",
                  vmin=Sdb.max() - 70, vmax=Sdb.max(), rasterized=True)
    ax.set_ylim(0, 4000)
    ax.set_yscale("symlog", linthresh=200)
    ax.set_ylabel(f"{title}\nHz", color=color, fontsize=8, fontweight="bold")
    ax.tick_params(labelsize=7)


def compute():
    cache = f"{OUT}/_modal_cache.npz"
    if os.path.exists(cache) and "--force" not in sys.argv:
        z = np.load(cache, allow_pickle=True)
        return list(z["ids"]), z["att"].item(), z["stems"].item(), int(z["nT"])
    frames = load(CAP)
    print(f"loaded {len(frames)} frames; building modal identities:")
    idents = build_identities(frames)
    spec, phase, att, nT = measure(frames, idents)
    active = [fid for fid in idents if att[fid].max() > 0]

    stems, n = {}, int(DUR * SR)
    for fid in active:
        spec[fid] = gaussian_filter1d(spec[fid], TIME_SMOOTH, axis=0, mode="nearest")
        att[fid] = gaussian_filter1d(att[fid], TIME_SMOOTH, mode="nearest")
        a = spec[fid] > 0
        denom = a.sum(0)
        phi = np.zeros(NG)
        nz = denom > 0
        phi[nz] = (phase[fid] * a).sum(0)[nz] / denom[nz]
        stems[fid] = synth(spec[fid], DUR, phi % (2 * np.pi))
    mix = np.zeros(n)
    for s in stems.values():
        mix[:len(s)] += s[:n]
    m = np.max(np.abs(mix))
    if m > 0:
        mix *= 0.6 / m
    fade = int(0.05 * SR)
    mix[:fade] *= np.linspace(0, 1, fade)
    mix[-fade:] *= np.linspace(1, 0, fade)
    audio.save_wav(f"{OUT}/modal_mix.wav", mix.astype(np.float32))
    np.savez(cache, ids=np.array(active),
             att={fid: att[fid] for fid in active}, stems=stems, nT=nT)
    return active, {fid: att[fid] for fid in active}, stems, nT


def main():
    os.makedirs(OUT, exist_ok=True)
    active, att, stems, nT = compute()
    tfr = np.linspace(0, DUR, nT)
    fig = plt.figure(figsize=(13, 2.6 * len(active)))
    gs = fig.add_gridspec(len(active) * 2, 1,
                          height_ratios=[1, 3] * len(active), hspace=0.08)
    for row, fid in enumerate(active):
        name, color, mat = FEATURES[fid]
        ap = fig.add_subplot(gs[row * 2, 0])
        ap.fill_between(tfr, att[fid], color=color, alpha=0.35)
        ap.plot(tfr, att[fid], color=color, lw=1.4)
        ap.set_xlim(0, DUR)
        ap.set_ylim(0, max(0.01, att[fid].max() * 1.15))
        ap.set_ylabel("A(F)", fontsize=7)
        ap.set_xticks([])
        ap.tick_params(labelsize=7)
        ap.set_title(f"{name} ({mat})  —  modal identity x perspective gains",
                     fontsize=9, color=color, fontweight="bold", loc="left")
        sx = fig.add_subplot(gs[row * 2 + 1, 0])
        spectro(sx, stems[fid], name, color)
        sx.set_xlim(0, DUR)
        if row == len(active) - 1:
            sx.set_xlabel("time (s)")
        else:
            sx.set_xticklabels([])
    fig.suptitle("Blender walk — modal identities (shape eigen-ladders), gains by perspective",
                 fontsize=12, y=0.997)
    fig.savefig(f"{OUT}/modal_features.png", dpi=110, bbox_inches="tight")
    print(f"saved {OUT}/modal_mix.wav ({DUR:.0f}s) and {OUT}/modal_features.png")


if __name__ == "__main__":
    main()
