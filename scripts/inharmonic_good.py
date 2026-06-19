"""Inharmonic-but-GOOD sounds, to complete the picture of what sounds good.

Everything we catalogued before was harmonic (chords/intervals) or noise. Bells,
singing bowls, and gamelan metallophones are inharmonic -- their partials are NOT
integer multiples of an f0 -- yet they sound beautiful. They show harmonicity is
not required for beauty. Unlike the earlier consonance study, these INCLUDE decay,
because differential decay (high partials dying faster than low) is a big part of
why they sound good -- the clangorous attack settles into a pure hum.

For each: wav + (spectrogram, zoomed p(t)). Also prints the two-axis scores so we
can see: roughness LOW (good), relative period LONG/inf (inharmonic) -- i.e. they
are good while failing the integer-ratio test.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram as stft

from music_making.consonance import relative_period, roughness

SR = 44100
OUT = "demos/consonance"
os.makedirs(OUT, exist_ok=True)


def struck(f0, ratios, amps, taus, dur, sr=SR):
    """Sum of partials a_i at f0*ratio_i, each with exponential decay exp(-t/tau_i)
    and a short attack. ratios may carry a detuned twin for slow beating."""
    t = np.arange(int(dur * sr)) / sr
    x = np.zeros_like(t)
    atk = np.clip(t / 0.004, 0, 1)               # 4 ms attack, no click
    for r, a, tau in zip(ratios, amps, taus):
        x += a * np.exp(-t / tau) * np.sin(2 * np.pi * f0 * r * t)
    return x * atk


def rms_norm(x, target=0.12):
    x = np.nan_to_num(x)
    return np.clip(x * (target / (np.sqrt(np.mean(x**2)) + 1e-12)), -1, 1)


# name, f0, ratios, amps, taus(s), dur, p(t)-window(ms), description, tone-set-for-scoring
SOUNDS = [
    ("good_inh_1_tuned_bell", 370.0,
     [0.5, 1.0, 1.2, 1.5, 2.0, 2.5, 2.67, 3.0, 4.0],
     [0.8, 1.0, 0.7, 0.5, 0.6, 0.3, 0.25, 0.3, 0.2],
     [3.5, 3.0, 1.6, 1.4, 1.2, 0.7, 0.6, 0.7, 0.4],
     4.0, 120,
     "Tuned church bell: hum(0.5), prime(1), minor-3rd tierce(1.2), 5th(1.5), "
     "octave(2)... Inharmonic but gives a clear strike pitch; the minor third is "
     "what makes it sound 'bell'."),

    ("good_inh_2_singing_bowl", 280.0,
     [1.0, 1.0008, 2.75, 5.2],                 # detuned twin on the fundamental -> ~1 Hz beat
     [1.0, 1.0, 0.35, 0.12],
     [7.0, 7.0, 3.0, 1.5],
     6.0, 2500,
     "Singing bowl: near-pure low modes with a ~1 Hz detune -> slow shimmer (beating "
     "used as BEAUTY, not roughness). Long smooth decay, serene."),

    ("good_inh_3_gamelan", 330.0,
     [1.0, 1.012, 2.8, 5.4],                   # paired bars ~5 Hz apart (gamelan 'ombak')
     [1.0, 1.0, 0.5, 0.25],
     [2.2, 2.2, 1.0, 0.6],
     3.0, 600,
     "Gamelan-style metallophone: inharmonic bar modes (1, 2.8, 5.4) plus a paired "
     "bar tuned ~5 Hz off for the characteristic shimmer. Bright, alive, not harsh."),
]


def plot(name, desc, x, win_ms):
    fig, (axs, axw) = plt.subplots(2, 1, figsize=(9, 6),
                                   gridspec_kw={"height_ratios": [2, 1]})
    f, t, S = stft(x, fs=SR, nperseg=4096, noverlap=3072)
    Sdb = 10 * np.log10(S + 1e-12)
    axs.pcolormesh(t, f, Sdb, shading="gouraud", cmap="magma",
                   vmin=Sdb.max() - 70, vmax=Sdb.max())
    axs.set_ylim(0, 4000)
    axs.set_ylabel("frequency (Hz)")
    axs.set_title(f"{name}\n{desc}", fontsize=9, loc="left", wrap=True)
    n0 = int(0.05 * SR)
    w = int(win_ms / 1000 * SR)
    seg = x[n0:n0 + w]
    axw.plot(np.arange(len(seg)) / SR * 1000, seg, lw=0.6, color="#1b6")
    axw.set_xlabel(f"time (ms)  --  p(t), {win_ms} ms window")
    axw.set_ylabel("pressure")
    axw.margins(x=0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=110)
    plt.close(fig)


print(f"{'name':26s} {'rough':>8s} {'rel_period':>11s}")
for name, f0, ratios, amps, taus, dur, win, desc in SOUNDS:
    x = rms_norm(struck(f0, ratios, amps, taus, dur))
    sf.write(f"{OUT}/{name}.wav", x.astype(np.float32), SR)
    plot(name, desc, x, win)
    # score on the distinct partials (drop the detuned twin for the ratio test)
    tones = sorted(set(round(f0 * r, 1) for r in ratios))
    rp = relative_period(tones)
    print(f"{name:26s} {roughness(tones, [1]*len(tones)):>8.4f} "
          f"{('inf' if np.isinf(rp) else f'{rp:.0f}'):>11s}")

print(f"\nWrote 3 inharmonic-good wavs + PNGs to {OUT}/")
