"""Multiscale textural description of a feature, and its pure-additive sonification.

The principle (Henry's): *good music is the frequency representation of the scene
it is about*, and a single feature (a rock, a flame, a tree) is itself a
**multiscale** object. If we describe that multiscale texture rigorously enough —
the way one can describe a rock verbally (homogeneous macro and micro texture,
emerging from a viscous lava it leaves a half-second residue in) — then sonifying
it is a mechanical mapping.

So the value lives in the *description*. A :class:`FeatureTexture` captures a
feature as a stack of spatial **scale bands** (coarse -> fine), each with its own
energy, homogeneity and spatial density, plus the medium it sits in and its
temporal life. The synthesizer is the dual: *the feature IS the spectrum*.

Mapping (spatial -> audio):
  coarse spatial scale (the trunk, the bulk)   -> low partials, the fundamental
  fine   spatial scale (twigs, leaf texture)   -> high partials, sparse shimmer
  self-similar branching (a tree's fractal)    -> a power-law partial rolloff
  spatial homogeneity (rock) vs heterogeneity  -> tonal sines vs broadband noise
  spatial sparsity at a scale                  -> temporal sparsity of that partial
  vertical extent (bottom -> top)              -> onset bloom (bulk first)
  the medium it emerges from                   -> a dark residue tail
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from . import audio


@dataclass(frozen=True)
class ScaleBand:
    """One octave of spatial scale, coarse bands first.

    ``energy`` is the structural energy at this scale (-> partial loudness);
    ``homogeneity`` is 0 = noisy/broadband .. 1 = regular/tonal (-> noise vs
    sine); ``density`` is 0 = sparse .. 1 = space-filling (-> how continuous vs
    granular this partial is in time).
    """
    energy: float
    homogeneity: float = 1.0
    density: float = 1.0


@dataclass(frozen=True)
class Medium:
    """The substance the feature emerges from / sits in (e.g. lava under rock)."""
    amount: float = 0.0          # 0 = none, 1 = strongly present
    persistence_s: float = 0.5   # how long the residue lingers after each hit
    darkness: float = 0.85       # 0 = bright, 1 = heavily low-passed


@dataclass(frozen=True)
class ModBand:
    """A temporal modulation of the whole feature (its 'life').

    Fire = a fast chaotic flicker under a slow drift -> two ModBands.
    """
    rate: float                  # Hz
    depth: float = 0.0           # 0..1 amount of amplitude modulation
    chaos: float = 0.0           # 0 = clean LFO, 1 = noise-driven flicker


@dataclass(frozen=True)
class FeatureTexture:
    """A rigorous multiscale description of one feature, ready to sonify."""
    name: str
    scales: list[ScaleBand]            # coarse -> fine
    fractal_slope: float = 1.0         # 1/f^beta slope of the spatial spectrum
    scale_ratio: float = 2.0           # spatial->audio octave ratio between bands
    bloom: float = 0.0                 # 0 = all at once, 1 = coarse-to-fine bloom
    medium: Medium = field(default_factory=Medium)
    modulation: list[ModBand] = field(default_factory=list)
    conscious: bool = False            # if True, described by feeling, not texture


# --------------------------------------------------------------------------- #
# Synthesis: the feature IS the spectrum.
# --------------------------------------------------------------------------- #

def _norm_rms(x: np.ndarray) -> np.ndarray:
    r = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
    return x / r if r > 1e-9 else x


def _bandnoise(n: int, lo: float, hi: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """White noise band-passed to [lo, hi] Hz — the broadband content of a scale."""
    nyq = sr / 2.0
    lo = max(20.0, min(lo, nyq * 0.9))
    hi = max(lo * 1.05, min(hi, nyq * 0.98))
    w = rng.standard_normal(n).astype(np.float32)
    sos = signal.butter(2, [lo / nyq, hi / nyq], btype="band", output="sos")
    return _norm_rms(signal.sosfilt(sos, w).astype(np.float32))


def _tonal(n: int, fc: float, sr: int) -> np.ndarray:
    """A small harmonic stack on ``fc`` — the tonal content of a homogeneous scale."""
    t = np.arange(n) / sr
    nyq = sr / 2.0
    out = np.zeros(n, dtype=np.float32)
    for m in (1, 2, 3):
        f = fc * m
        if f >= nyq * 0.95:
            break
        out += (1.0 / m) * np.sin(2 * np.pi * f * t).astype(np.float32)
    return _norm_rms(out)


def _scale_env(n: int, onset_s: float, density: float, sr: int,
               rng: np.random.Generator) -> np.ndarray:
    """Onset bloom + density-driven granularity for one scale band."""
    env = np.ones(n, dtype=np.float32)
    a = max(1, int((0.02 + onset_s) * sr))          # attack incl. bloom delay
    env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    r = max(1, int(0.05 * sr))                       # short release
    env[-r:] *= np.linspace(1.0, 0.0, r, dtype=np.float32)
    if density < 1.0:
        # sparse scales (twigs, sparks) twinkle rather than sustain
        g = rng.standard_normal(n).astype(np.float32)
        sos = signal.butter(2, 18 / (sr / 2), btype="low", output="sos")
        g = signal.sosfilt(sos, g).astype(np.float32)
        g = (g - g.min()) / (float(np.ptp(g)) or 1.0)
        env = env * (density + (1.0 - density) * g)
    return env


def _modulate(x: np.ndarray, sr: int, mods: list[ModBand], seed: int) -> np.ndarray:
    if not mods:
        return x
    t = np.arange(len(x)) / sr
    env = np.ones(len(x), dtype=np.float32)
    for i, m in enumerate(mods):
        if m.depth <= 0 or m.rate <= 0:
            continue
        clean = 0.5 * (1.0 + np.sin(2 * np.pi * m.rate * t)).astype(np.float32)
        if m.chaos > 0:
            rng = np.random.default_rng(seed + i)
            w = rng.standard_normal(len(x)).astype(np.float32)
            sos = signal.butter(2, min(0.99, max(m.rate, 0.5) / (sr / 2)),
                                btype="low", output="sos")
            f = signal.sosfilt(sos, w).astype(np.float32)
            f = (f - f.min()) / (float(np.ptp(f)) or 1.0)
            lfo = m.chaos * f + (1.0 - m.chaos) * clean
        else:
            lfo = clean
        env *= 1.0 - m.depth * (1.0 - lfo)
    return (x * env).astype(np.float32)


def _residue(x: np.ndarray, sr: int, med: Medium) -> np.ndarray:
    """A dark, viscous tail — the medium the feature emerges from."""
    if med.amount <= 0:
        return x
    length = max(1, int(med.persistence_s * sr))
    t = np.arange(length) / sr
    ir = np.exp(-t / (med.persistence_s / 3.0)).astype(np.float32)
    tail = signal.fftconvolve(x, ir)[: len(x)].astype(np.float32)
    cutoff = (1.0 - med.darkness) * 4000.0 + 250.0
    sos = signal.butter(2, cutoff / (sr / 2), btype="low", output="sos")
    tail = signal.sosfilt(sos, tail).astype(np.float32)
    tail *= (np.max(np.abs(x)) + 1e-9) / (np.max(np.abs(tail)) + 1e-9)
    return ((1.0 - med.amount * 0.5) * x + med.amount * tail).astype(np.float32)


# --------------------------------------------------------------------------- #
# The rigor table: a description is adequate when every cell is filled.
# Each row pairs a textural dimension with the exact sonic consequence, so a
# description can be read (or audited) and heard. `describe_table` renders any
# FeatureTexture; `missing_rigor` flags under-specified cells before synthesis.
# --------------------------------------------------------------------------- #

# dimension -> (what it describes, how it is sonified)
RIGOR_ROWS: list[tuple[str, str, str]] = [
    ("scales",        "spatial structure, coarse->fine",      "one partial layer per band"),
    ("  energy",      "structural energy at the scale",        "partial loudness"),
    ("  homogeneity", "regular/tonal (1) vs noisy (0)",        "sine stack vs band noise"),
    ("  density",     "space-filling (1) vs sparse (0)",       "sustained vs granular twinkle"),
    ("fractal_slope", "self-similarity (1/f^beta)",            "power-law partial rolloff"),
    ("scale_ratio",   "branching ratio across scales",         "audio octave step per band"),
    ("bloom",         "vertical order, bottom->top",           "coarse-first onset stagger"),
    ("medium",        "substance it emerges from",             "dark residue tail"),
    ("modulation",    "temporal life (drift, flicker)",        "multi-rate amplitude mod"),
    ("conscious",     "a being, not a material",               "no texture (carried by motif)"),
]


def missing_rigor(ft: FeatureTexture) -> list[str]:
    """Cells that are under-specified for adequate sonification (empty rigor)."""
    gaps: list[str] = []
    if not ft.scales:
        gaps.append("scales: no spatial structure described")
    for i, s in enumerate(ft.scales):
        if s.energy <= 0:
            gaps.append(f"scales[{i}].energy: zero (band is silent)")
    if not ft.conscious and not ft.modulation and ft.medium.amount <= 0:
        gaps.append("modulation/medium: a material with neither flicker nor medium reads inert")
    return gaps


def describe_table(ft: FeatureTexture) -> str:
    """Render the rigor table for one feature: dimension | value | -> sound."""
    def fmt(v: float) -> str:
        return f"{v:.2f}"

    val = {
        "scales": f"{len(ft.scales)} bands",
        "fractal_slope": fmt(ft.fractal_slope),
        "scale_ratio": fmt(ft.scale_ratio),
        "bloom": fmt(ft.bloom),
        "medium": (f"amt {fmt(ft.medium.amount)}, {fmt(ft.medium.persistence_s)}s, "
                   f"dark {fmt(ft.medium.darkness)}") if ft.medium.amount > 0 else "none",
        "modulation": ", ".join(f"{fmt(m.rate)}Hz x{fmt(m.depth)}"
                                f"{' chaos' if m.chaos > 0 else ''}"
                                for m in ft.modulation) or "none",
        "conscious": "yes" if ft.conscious else "no",
    }
    w = 13
    lines = [f"FeatureTexture: {ft.name}",
             f"{'dimension':<{w}} {'value':<26} -> sound"]
    for key, _what, sound in RIGOR_ROWS:
        if key.startswith("  "):
            continue  # per-scale rows shown in the scale sub-table below
        lines.append(f"{key:<{w}} {val[key]:<26} -> {sound}")
    lines.append(f"{'scale':>5} {'energy':>7} {'homog':>7} {'density':>8}   (coarse -> fine)")
    for i, s in enumerate(ft.scales):
        lines.append(f"{i:>5} {s.energy:>7.2f} {s.homogeneity:>7.2f} {s.density:>8.2f}")
    gaps = missing_rigor(ft)
    if gaps:
        lines.append("UNDER-SPECIFIED:")
        lines.extend(f"  - {g}" for g in gaps)
    return "\n".join(lines)


def render_feature(ft: FeatureTexture, f0: float, dur: float,
                   sr: int = audio.SR, seed: int = 0) -> np.ndarray:
    """Synthesize a feature additively: each scale band becomes a partial layer.

    ``f0`` is the fundamental (the coarsest scale); finer scales sit ``scale_ratio``
    octaves above. Returns a mono float32 buffer of length ``dur*sr``.
    """
    n = int(dur * sr)
    if n <= 0 or not ft.scales:
        return np.zeros(max(0, n), dtype=np.float32)
    rng = np.random.default_rng(seed)
    out = np.zeros(n, dtype=np.float32)
    k_max = max(1, len(ft.scales) - 1)
    for k, band in enumerate(ft.scales):           # coarse -> fine
        if band.energy <= 0:
            continue
        fc = f0 * (ft.scale_ratio ** k)
        if fc >= sr / 2 * 0.95:
            break
        tonal = _tonal(n, fc, sr)
        noise = _bandnoise(n, fc * 0.7, fc * 1.7, sr, rng)
        content = band.homogeneity * tonal + (1.0 - band.homogeneity) * noise
        onset = ft.bloom * (k / k_max) * dur * 0.5
        env = _scale_env(n, onset, band.density, sr, rng)
        out += band.energy * content * env
    out = _modulate(out, sr, ft.modulation, seed)
    out = _residue(out, sr, ft.medium)
    peak = float(np.max(np.abs(out))) or 1.0
    return (out / peak * 0.99).astype(np.float32)


# --------------------------------------------------------------------------- #
# Gold-standard hand-authored descriptions (Henry's verbal rigor, encoded).
# These are the reference the auto-extractor is checked against.
# --------------------------------------------------------------------------- #

def rock_texture() -> FeatureTexture:
    """Rock: homogeneous macro AND micro texture, emerging from viscous lava that
    leaves a ~half-second dark residue. Energy concentrated in coarse scales."""
    return FeatureTexture(
        name="rock",
        scales=[
            ScaleBand(energy=1.00, homogeneity=0.95, density=1.0),
            ScaleBand(energy=0.45, homogeneity=0.90, density=1.0),
            ScaleBand(energy=0.18, homogeneity=0.85, density=1.0),
        ],
        fractal_slope=1.8,                 # steep rolloff -> dark, solid
        bloom=0.0,                         # a rock arrives as one unit
        medium=Medium(amount=0.35, persistence_s=0.5, darkness=0.9),
        modulation=[ModBand(rate=0.2, depth=0.22, chaos=0.0)],  # gentle slow in/out
    )


def fire_texture() -> FeatureTexture:
    """Fire: broadband at every scale, a fast chaotic flicker under a slow drift
    (the flame's wandering centre of mass)."""
    return FeatureTexture(
        name="fire",
        scales=[
            ScaleBand(energy=0.7, homogeneity=0.25, density=1.0),
            ScaleBand(energy=0.9, homogeneity=0.15, density=0.9),
            ScaleBand(energy=1.0, homogeneity=0.10, density=0.8),
            ScaleBand(energy=0.8, homogeneity=0.05, density=0.7),
        ],
        fractal_slope=0.8,                 # flat-ish -> energy across the spectrum
        bloom=0.0,
        medium=Medium(amount=0.0),
        modulation=[
            ModBand(rate=0.3, depth=0.35, chaos=0.0),    # slow drift
            ModBand(rate=11.0, depth=0.6, chaos=0.85),   # fast chaotic flicker
        ],
    )


def tree_texture() -> FeatureTexture:
    """Tree: a trunk (coarse, homogeneous, rock-like but longer-sustaining) with
    fractal branches (self-similar across scales) and a sparse, noisy leaf canopy
    up top. Blooms bottom (trunk) to top (canopy)."""
    return FeatureTexture(
        name="tree",
        scales=[
            ScaleBand(energy=1.00, homogeneity=0.92, density=1.0),   # trunk
            ScaleBand(energy=0.62, homogeneity=0.72, density=0.9),   # boughs
            ScaleBand(energy=0.40, homogeneity=0.55, density=0.7),   # branches
            ScaleBand(energy=0.26, homogeneity=0.38, density=0.5),   # twigs
            ScaleBand(energy=0.17, homogeneity=0.22, density=0.35),  # leaf texture
        ],
        fractal_slope=1.3,                 # self-similar power law -> rich top
        bloom=0.6,                         # trunk first, canopy blooms up
        medium=Medium(amount=0.12, persistence_s=0.35, darkness=0.8),
        modulation=[ModBand(rate=0.7, depth=0.18, chaos=0.3)],  # gentle leaf sway
    )
