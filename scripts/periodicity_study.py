"""Axis 2 of consonance: the RELATIVE PERIOD of p(t).

A combined waveform repeats only after the least common period of its tones.
Express every tone as a ratio to the lowest, approximate each ratio by p/q
(bounded denominator), and the waveform repeats after LCM(q) cycles of the
lowest tone. Small LCM = deep, frequent reinforcement (consonant/"alive");
large LCM = the whole pattern repeats only after a long time (the tritone's
"spooky but sort of works"); astronomically large = never repeats (dead/hollow).

This pairs with the Sethares roughness from consonance_study.py to give the two
axes that, together, separate good from bad. Reads that script's summary.csv for
the dissonance column and writes a 2-axis scatter.
"""

import csv
import math
import os
from fractions import Fraction

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "demos/consonance"
QMAX = 64  # how complex a ratio we still call "periodic" before it's effectively aperiodic


def et(semitones, base=220.0):
    return base * 2 ** (semitones / 12)


# The TONE fundamentals of each sound (harmonics don't change the period, so for
# the chords we list note f0s; for the inharmonic/cluster sounds the partials ARE
# the tones).
TONES = {
    "good_1_harmonic_tone": [220.0],
    "good_2_octave": [220.0, 440.0],
    "good_3_perfect_fifth": [220.0, 330.0],
    "good_4_major_triad": [220.0, 275.0, 330.0],
    "good_5_major_arpeggio": [220.0],  # one note sounds at a time
    "bad_1_critical_band_beat": [440.0, 463.0],
    "bad_2_semitone_cluster": [et(12 + s) for s in range(5)],
    "bad_3_inharmonic_clang": [233.0 * r for r in [1, 1.41, 1.73, 2.13, 2.78, 3.33]],
    "bad_4_tritone_rich": [440.0, 622.0],
    "bad_5_high_dense_screech": [2000.0, 2157.0, 2314.0, 2471.0, 2629.0, 2786.0, 2943.0, 3100.0],
}


def relative_period(freqs, qmax=QMAX):
    """Return (relperiod_cycles, max_relative_error, period_ms).
    relperiod = LCM of denominators of the rationalized ratios to the lowest tone."""
    f0 = min(freqs)
    dens, max_err = [], 0.0
    for f in freqs:
        r = f / f0
        frac = Fraction(r).limit_denominator(qmax)
        dens.append(frac.denominator)
        max_err = max(max_err, abs(r - float(frac)) / r)
    relperiod = math.lcm(*dens)
    period_ms = relperiod / f0 * 1000
    return relperiod, max_err, period_ms


# pull dissonance from the roughness study
diss = {}
with open(f"{OUT}/summary.csv") as fh:
    for row in csv.DictReader(fh):
        diss[row["name"]] = float(row["dissonance"])

rows = []
for name, freqs in TONES.items():
    rp, err, pms = relative_period(freqs)
    rows.append({
        "name": name,
        "label": "good" if name.startswith("good") else "bad",
        "dissonance": diss.get(name, float("nan")),
        "rel_period_cycles": rp,
        "period_ms": round(pms, 1),
        "ratio_error_pct": round(err * 100, 2),
    })

# table
print(f"{'name':28s} {'lbl':4s} {'D(rough)':>9s} {'relperiod':>10s} {'period_ms':>10s} {'err%':>6s}")
for r in rows:
    rp = r["rel_period_cycles"]
    rp_s = f"{rp:.2e}" if rp > 1e5 else str(rp)
    print(f"{r['name']:28s} {r['label']:4s} {r['dissonance']:9.4f} "
          f"{rp_s:>10s} {r['period_ms']:10.1f} {r['ratio_error_pct']:6.2f}")

with open(f"{OUT}/periodicity.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# 2-axis scatter: roughness (x) vs log relative period (y)
fig, ax = plt.subplots(figsize=(9, 7))
for r in rows:
    color = "#2a8" if r["label"] == "good" else "#d33"
    y = math.log10(max(r["rel_period_cycles"], 1))
    ax.scatter(r["dissonance"], y, c=color, s=90, edgecolor="k", zorder=3)
    ax.annotate(r["name"].split("_", 2)[-1], (r["dissonance"], y),
                fontsize=7, xytext=(5, 4), textcoords="offset points")
ax.set_xlabel("Axis 1: Sethares roughness  (beating within critical band) ->")
ax.set_ylabel("Axis 2: log10 relative period of p(t)  (how long until it repeats) ->")
ax.set_title("Two axes of consonance\ngreen = pleasant, red = unpleasant; "
             "good sits in the lower-left (smooth AND deeply periodic)")
ax.scatter([], [], c="#2a8", label="good", edgecolor="k")
ax.scatter([], [], c="#d33", label="bad", edgecolor="k")
ax.legend(loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/two_axes.png", dpi=120)
print(f"\nWrote {OUT}/periodicity.csv and {OUT}/two_axes.png")
