"""Labeled consonance dataset: 5 sounds humans reliably rate PLEASANT, 5 UNPLEASANT.

The point is to expose the mathematical pattern behind "good sounding" so we can
build a perspective that filters for it. Each sound is built from constant-amplitude
partials (NO decay / dissipation, as requested). For each we save:
  - the audio  (<name>.wav)
  - the spectrogram             (top panel)
  - the pressure waveform p(t)  (bottom panel, zoomed so periodicity is visible)
plus one overview grid (overview.png) and a summary table (summary.csv).

The 10 are designed to vary along the two axes that the consonance literature says
matter:  (1) HARMONICITY  -- are the partials integer multiples of a common f0?
         (2) ROUGHNESS     -- are any partials spaced *within* a critical band
                              (~ a few percent / < ~ a minor third in the mid range),
                              which produces beating the ear hears as harsh?
Good = harmonic + partials spaced beyond the critical band.
Bad  = inharmonic and/or partials packed inside the critical band.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram

SR = 44100
DUR = 2.5
OUT = "demos/consonance"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- synthesis ---
def partials_tone(specs, dur=DUR, sr=SR):
    """specs: list of (freq_hz, amp). Constant amplitude, random phase, no decay."""
    t = np.arange(int(dur * sr)) / sr
    x = np.zeros_like(t)
    for f, a in specs:
        x += a * np.sin(2 * np.pi * f * t + RNG.uniform(0, 2 * np.pi))
    return x


def sethares_dissonance(specs):
    """Plomp-Levelt / Sethares sensory dissonance summed over all partial pairs.
    Each pair (f1,a1),(f2,a2) contributes a1*a2 * [exp(-b1*s*d) - exp(-b2*s*d)]
    where d = |f2-f1| and s scales by the critical bandwidth at the lower freq.
    Dips at low-integer frequency ratios -> this IS the consonance pattern."""
    b1, b2, dstar = 3.5, 5.75, 0.24
    s1, s2 = 0.0207, 18.96
    fa = sorted(specs)
    total = 0.0
    for i in range(len(fa)):
        f1, a1 = fa[i]
        for j in range(i + 1, len(fa)):
            f2, a2 = fa[j]
            d = f2 - f1
            s = dstar / (s1 * f1 + s2)
            total += a1 * a2 * (np.exp(-b1 * s * d) - np.exp(-b2 * s * d))
    norm = sum(a for _, a in specs) ** 2 + 1e-12
    return float(total / norm)  # amplitude-normalized so loudness doesn't dominate


def harmonic_stack(f0, n=8, rolloff=1.0):
    """f0 with harmonics 1..n at amplitude 1/k**rolloff -- a natural pitched tone."""
    return [(f0 * k, 1.0 / k**rolloff) for k in range(1, n + 1)]


def chord(f0s, n=6, rolloff=1.0):
    specs = []
    for f0 in f0s:
        specs += harmonic_stack(f0, n=n, rolloff=rolloff)
    return specs


def arpeggio(f0s, n=6, dur=DUR, sr=SR):
    """Sequence of harmonic notes, equal slices, tiny crossfade -- evolving spectrogram."""
    seg = int(dur / len(f0s) * sr)
    fade = int(0.01 * sr)
    env = np.ones(seg)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    out = []
    for f0 in f0s:
        out.append(partials_tone(harmonic_stack(f0, n=n), dur=seg / sr) * env)
    return np.concatenate(out)


# ------------------------------------------------------------- the dataset ---
# Equal-tempered helper (A=220 reference, ratios are what matter).
def et(semitones, base=220.0):
    return base * 2 ** (semitones / 12)


# Each entry: (name, description, spec).  spec is a list of (freq, amp) partials,
# OR a list of such lists for a time-varying (arpeggio) sound.
GOOD = [
    ("good_1_harmonic_tone",
     "Single pitched tone: f0=220 with 8 integer harmonics. Perfectly harmonic, "
     "one periodic waveform.",
     harmonic_stack(220, n=8)),

    ("good_2_octave",
     "Two tones an octave apart (2:1). Partials interleave exactly -> no new beating.",
     chord([220, 440], n=6)),

    ("good_3_perfect_fifth",
     "Perfect fifth (3:2, 220 & 330). Low-integer ratio, partials align on a sub-harmonic.",
     chord([220, 330], n=6)),

    ("good_4_major_triad",
     "Major triad 4:5:6 (220, 275, 330). The canonical consonant chord.",
     chord([220, 275, 330], n=6)),

    ("good_5_major_arpeggio",
     "Time-varying: C-E-G-C major arpeggio (each note harmonic). Pleasant evolving spectrogram.",
     [harmonic_stack(f, n=6) for f in [220, et(4), et(7), et(12)]]),
]

BAD = [
    ("bad_1_critical_band_beat",
     "Two pure tones 23 Hz apart at 440/463 -- spacing sits in the critical band -> "
     "maximal roughness / beating.",
     [(440, 1.0), (463, 1.0)]),

    ("bad_2_semitone_cluster",
     "Five adjacent semitones (440,466,494,523,554) all sustained -> dense beating.",
     [(et(s + 12), 1.0) for s in range(5)]),

    ("bad_3_inharmonic_clang",
     "Partials at NON-integer ratios 1,1.41,1.73,2.13,2.78,3.33 of 233 Hz -> metallic, "
     "pitchless clang.",
     [(233 * r, 1.0 / (i + 1))
      for i, r in enumerate([1, 1.41, 1.73, 2.13, 2.78, 3.33])]),

    ("bad_4_tritone_rich",
     "Tritone 440 & 622 (~sqrt2, 45:32) with full harmonic stacks -> harmonics collide "
     "inside critical bands.",
     chord([440, 622], n=6)),

    ("bad_5_high_dense_screech",
     "Eight closely packed inharmonic partials 2.0-3.1 kHz (spacing << critical band) "
     "-> harsh screech.",
     [(float(f), 1.0) for f in np.linspace(2000, 3100, 8)
      * (1 + 0.01 * RNG.standard_normal(8))]),
]

ALL = [(n, d, s, "good") for n, d, s in GOOD] + [(n, d, s, "bad") for n, d, s in BAD]


def synth_from_spec(spec):
    """spec = flat partial list -> sustained tone; list-of-lists -> arpeggio."""
    if spec and isinstance(spec[0], list):
        seg = DUR / len(spec)
        fade = int(0.01 * SR)
        env = np.ones(int(seg * SR))
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
        return np.concatenate([partials_tone(s, dur=seg) * env for s in spec])
    return partials_tone(spec)


def diss_from_spec(spec):
    """Average Sethares dissonance (per simultaneous sound for arpeggios)."""
    if spec and isinstance(spec[0], list):
        return float(np.mean([sethares_dissonance(s) for s in spec]))
    return sethares_dissonance(spec)


# ---------------------------------------------------------------- analysis ---
def rms_normalize(x, target=0.12):
    x = np.nan_to_num(x)
    r = np.sqrt(np.mean(x**2)) + 1e-12
    x = x * (target / r)
    return np.clip(x, -1.0, 1.0)


# ---------------------------------------------------------------- plotting ---
def plot_one(name, desc, x):
    fig, (axs, axw) = plt.subplots(2, 1, figsize=(9, 6),
                                   gridspec_kw={"height_ratios": [2, 1]})
    f, t, Sxx = scipy_spectrogram(x, fs=SR, nperseg=4096, noverlap=3072)
    Sdb = 10 * np.log10(Sxx + 1e-12)
    axs.pcolormesh(t, f, Sdb, shading="gouraud", cmap="magma",
                   vmin=Sdb.max() - 70, vmax=Sdb.max())
    axs.set_ylim(0, 4000)
    axs.set_ylabel("frequency (Hz)")
    axs.set_title(f"{name}\n{desc}", fontsize=9, loc="left", wrap=True)

    # zoomed pressure waveform p(t): 150 ms from the middle, shows periodicity/beating
    n0 = len(x) // 2
    w = int(0.15 * SR)
    seg = x[n0:n0 + w]
    tt = np.arange(len(seg)) / SR * 1000
    axw.plot(tt, seg, lw=0.7, color="#1b6")
    axw.set_xlabel("time (ms)  --  p(t), 150 ms window")
    axw.set_ylabel("pressure")
    axw.margins(x=0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- run ---------
import soundfile as sf  # noqa: E402

rows = []
fig_grid, axes = plt.subplots(2, 5, figsize=(20, 6))
for col, (name, desc, spec, label) in enumerate(ALL):
    x = rms_normalize(synth_from_spec(spec))
    diss = diss_from_spec(spec)
    sf.write(f"{OUT}/{name}.wav", x.astype(np.float32), SR)
    plot_one(name, desc, x)

    rows.append({"name": name, "label": label, "dissonance": round(diss, 4),
                 "description": desc})

    # overview grid: spectrogram thumbnails
    ax = axes[0 if label == "good" else 1][col % 5]
    f, t, Sxx = scipy_spectrogram(x, fs=SR, nperseg=2048, noverlap=1024)
    Sdb = 10 * np.log10(Sxx + 1e-12)
    ax.pcolormesh(t, f, Sdb, shading="gouraud", cmap="magma",
                  vmin=Sdb.max() - 70, vmax=Sdb.max())
    ax.set_ylim(0, 4000)
    ax.set_title(f"{label.upper()}  D={diss:.3f}\n{name[6:]}", fontsize=8)
    ax.set_xticks([])
    print(f"{label:4s}  dissonance={diss:7.4f}  {name}")

axes[0][0].set_ylabel("GOOD\nfreq (Hz)")
axes[1][0].set_ylabel("BAD\nfreq (Hz)")
fig_grid.suptitle("Consonance study -- spectrograms (D = Sethares dissonance, higher = worse)")
fig_grid.tight_layout()
fig_grid.savefig(f"{OUT}/overview.png", dpi=110)
plt.close(fig_grid)

with open(f"{OUT}/summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["name", "label", "dissonance", "description"])
    w.writeheader()
    w.writerows(rows)

print(f"\nWrote {len(ALL)} wavs + per-sound PNGs + overview.png + summary.csv to {OUT}/")
