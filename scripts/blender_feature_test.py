"""Walk a feature-segmented Blender capture and synthesize, per the viewer-coupled
dynamics model (docs/viewer_coupled_dynamics.md).

Each object-index region is a feature F(i,t). Per frame, per feature we measure:
  * E(f)      — colour->spectrum of the feature's pixels (the within-band prior)
  * N         — pixel count
  * r_F       — mean eye->pixel range (metres) to the feature
  * theta     — eccentricity of the feature centroid from the gaze axis (screen centre)
and form the FEATURE ATTENTION as a count-weighted, gaze-and-distance softmax
across the features present that frame:

    s   = k_gaze / r_F,   k_gaze = exp(KAPPA_G (cos theta - 1))
    A_i = N_i exp(s_i / T) / sum_j N_j exp(s_j / T)            (sum_i A_i = 1)

The feature's spectrogram column is A_i * E(f); each feature is resynthesized
(amplitudes gliding between frames) with a depth-derived phase, and the features
are mixed to the test sound. Output: a 45 s wav + a figure with every feature's
spectrogram and its attention profile above it.

    python scripts/blender_feature_test.py [CAPTURE_DIR] [DURATION_S]
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

from music_making import audio, dynamics
from music_making.color_spectrum import FREQS, NG

CAP = sys.argv[1] if len(sys.argv) > 1 else "demos/walk_features"
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
SR = audio.SR
OUT = CAP                  # write outputs/cache alongside whatever capture is passed in
MAXIDX = 16
C_SND = 343.0

# model knobs (docs/viewer_coupled_dynamics.md §2-3) — sniff-test these on the spectrogram
T_F = 0.20        # feature-attention temperature  A(F)
T_B = 0.20        # band-attention temperature      A(B)
KAPPA_G = 2.0     # gaze concentration (foveal sharpness)
K_SIGMA = 20.0    # within-band sharpness: how hard distance sharpens sigma
TIME_SMOOTH = 1.5  # temporal smoothing of G(f,t), in frames (kills the per-frame step)

FEATURES = {1: ("ground", "#5a4a2a"), 2: ("water", "#1f6fb0"), 3: ("conifers", "#1f7a2e"),
            4: ("fire", "#ff7a18"), 5: ("red", "#c0142a"), 6: ("yellow", "#d8c020"),
            7: ("purple", "#8a2da0"), 8: ("cyan", "#16b0b0"), 9: ("orange", "#e07a18"),
            10: ("meadow", "#d040a0"), 11: ("magenta", "#c01080"), 12: ("foliage", "#5a9a20"),
            13: ("rocks", "#888890"), 14: ("blossoms", "#e89ac8")}


def load(cap_dir):
    """Read the capture into per-frame (rgb, camera-space 3D positions, feature idx).
    Camera space: viewer at origin, looking down +z; pos = z*(dx, dy, 1)."""
    meta = json.load(open(os.path.join(cap_dir, "camera.json")))
    w, h = meta["resolution"]
    fov_x, clip0, clip1 = meta["fov_x"], float(meta["clip_start"]), float(meta["clip_end"])
    fx = (w / 2.0) / np.tan(fov_x / 2.0)
    ys, xs = np.mgrid[0:h, 0:w]
    base = np.dstack([(xs - w / 2.0) / fx, (ys - h / 2.0) / fx, np.ones((h, w))])
    frames = []
    for k in range(len(meta["frames"])):
        rgb = cv2.cvtColor(cv2.imread(f"{cap_dir}/frame_{k:04d}.png"), cv2.COLOR_BGR2RGB) / 255.0
        d = cv2.imread(f"{cap_dir}/depth_{k:04d}.png", cv2.IMREAD_UNCHANGED).astype(np.float64)
        z = clip0 + d / np.iinfo(np.uint16).max * (clip1 - clip0)
        z[d / np.iinfo(np.uint16).max >= 1.0 - 1e-6] = np.inf
        with np.errstate(invalid="ignore"):              # inf*0 at the dead-centre bg pixel
            pos = z[..., None] * base                     # (H,W,3); +inf z = background
        ix = cv2.imread(f"{cap_dir}/index_{k:04d}.png", cv2.IMREAD_UNCHANGED).astype(np.float64)
        idx = np.round(ix / np.iinfo(np.uint16).max * MAXIDX).astype(int)
        frames.append((rgb, pos, idx))
    return w, h, frames


def measure(frames):
    """Per feature, per frame: the full gain G(f,t), phase Phi(f,t), and A(F)(t), via
    dynamics.measure_frame (G = A(F).A(B).sigma)."""
    nT = len(frames)
    spec = {fid: np.zeros((nT, NG)) for fid in FEATURES}
    phase = {fid: np.zeros((nT, NG)) for fid in FEATURES}
    att = {fid: np.zeros(nT) for fid in FEATURES}
    for t, (rgb, pos, idx) in enumerate(frames):
        for r in dynamics.measure_frame(rgb, pos, idx, list(FEATURES), T_F=T_F, T_B=T_B,
                                        KAPPA_G=KAPPA_G, K_SIGMA=K_SIGMA, seed=t):
            spec[r["fid"]][t] = r["G"]
            phase[r["fid"]][t] = r["Phi"]
            att[r["fid"]][t] = r["A_F"]
    return spec, phase, att, nT


def synth(E_t, dur, phi):
    """Additive resynthesis with amplitudes gliding between frames; per-freq phase phi."""
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
    band = fr <= 4200                                   # crop before drawing (memory)
    fr, S = fr[band], S[band]
    Sdb = 10 * np.log10(S + 1e-12)
    ax.pcolormesh(tt, fr, Sdb, shading="auto", cmap="magma",
                  vmin=Sdb.max() - 70, vmax=Sdb.max(), rasterized=True)
    ax.set_ylim(0, 4000)
    ax.set_ylabel(f"{title}\nHz", color=color, fontsize=8, fontweight="bold")
    ax.tick_params(labelsize=7)


def compute():
    """Slow path: measure + synth per feature. Cached to _cache.npz so the figure
    can be re-rendered fast. Pass --force to recompute."""
    cache = f"{OUT}/_cache.npz"
    if os.path.exists(cache) and "--force" not in sys.argv:
        z = np.load(cache, allow_pickle=True)
        return list(z["ids"]), z["att"].item(), z["stems"].item(), int(z["nT"])
    os.makedirs(OUT, exist_ok=True)
    w, h, frames = load(CAP)
    spec, phase, att, nT = measure(frames)
    active = [fid for fid in FEATURES if att[fid].max() > 0]
    print(f"loaded {nT} frames, {len(active)} active features: "
          + ", ".join(FEATURES[f][0] for f in active))

    stems, n = {}, int(DUR * SR)
    for fid in active:
        # smooth gain transitions across frames so the 0.6s-per-frame steps become swells
        spec[fid] = gaussian_filter1d(spec[fid], TIME_SMOOTH, axis=0, mode="nearest")
        att[fid] = gaussian_filter1d(att[fid], TIME_SMOOTH, mode="nearest")
        G_t = spec[fid]                                # already G = A(F).A(B).sigma
        a = G_t > 0                                    # per-bin propagation phase, averaged
        denom = a.sum(0)
        phi = np.zeros(NG)
        nz = denom > 0
        phi[nz] = (phase[fid] * a).sum(0)[nz] / denom[nz]
        stems[fid] = synth(G_t, DUR, phi % (2 * np.pi))
    mix = np.zeros(n)
    for s in stems.values():
        mix[:len(s)] += s[:n]
    m = np.max(np.abs(mix))
    if m > 0:
        mix *= 0.6 / m
    fade = int(0.05 * SR)
    mix[:fade] *= np.linspace(0, 1, fade)
    mix[-fade:] *= np.linspace(1, 0, fade)
    audio.save_wav(f"{OUT}/features_mix.wav", mix.astype(np.float32))
    np.savez(cache, ids=np.array(active),
             att={fid: att[fid] for fid in active}, stems=stems, nT=nT)
    return active, {fid: att[fid] for fid in active}, stems, nT


def main():
    os.makedirs(OUT, exist_ok=True)
    active, att, stems, nT = compute()

    # figure: per feature, attention profile (top) over spectrogram (bottom)
    tfr = np.linspace(0, DUR, nT)
    fig = plt.figure(figsize=(13, 2.6 * len(active)))
    gs = fig.add_gridspec(len(active) * 2, 1,
                          height_ratios=[1, 3] * len(active), hspace=0.08)
    for row, fid in enumerate(active):
        name, color = FEATURES[fid]
        ap = fig.add_subplot(gs[row * 2, 0])
        ap.fill_between(tfr, att[fid], color=color, alpha=0.35)
        ap.plot(tfr, att[fid], color=color, lw=1.4)
        ap.set_xlim(0, DUR)
        ap.set_ylim(0, max(0.01, att[fid].max() * 1.15))
        ap.set_ylabel("A(F)", fontsize=7)
        ap.set_xticks([])
        ap.tick_params(labelsize=7)
        ap.set_title(f"{name}  —  feature attention A(F(i,t))  +  spectrogram",
                     fontsize=9, color=color, fontweight="bold", loc="left")
        sx = fig.add_subplot(gs[row * 2 + 1, 0])
        spectro(sx, stems[fid], name, color)
        sx.set_xlim(0, DUR)
        if row == len(active) - 1:
            sx.set_xlabel("time (s)")
        else:
            sx.set_xticklabels([])
    fig.suptitle("Blender walk — per-feature spectrograms with feature-attention profiles",
                 fontsize=12, y=0.997)
    fig.savefig(f"{OUT}/features.png", dpi=110, bbox_inches="tight")
    print(f"saved {OUT}/features_mix.wav ({DUR:.0f}s) and {OUT}/features.png")


if __name__ == "__main__":
    main()
