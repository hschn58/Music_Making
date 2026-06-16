"""The tabulation: the mode-agnostic spine of the system.

    Source (video / images / text)  ->  TABULATION  ->  Music

A *tabulation* is a complete, time-resolved, rigorous description of a scene **as
experienced by an observer moving through it** (the Michael-Jackson point: the
music is the frequency representation of the video, and the video is the path).
It has two tables:

1. **Feature catalog** — what each feature *is*: a set of spatial ``Component``s,
   each placed by the three orthogonal axes
   (size -> ``freq`` / pitch, surface -> ``roughness`` / tone-vs-noise,
   occurrence -> ``grain_rate`` / density), plus the medium it sits in.
2. **Perspective track** — how it is *experienced*: the observer's path tabulated
   as ``time x feature.component -> gain``. A static feature the gaze sweeps
   (a tree, top->bottom) and a dynamic one met on the ground (a fire, swelling as
   you near it) are the *same* mechanism — a gain envelope per component over time.

The renderer is the dual: each component is synthesized additively for the whole
piece, scaled by its perspective-gain envelope, and summed. Because the table is
the contract, anything that can fill it (a video now, plain text later) makes
music — that is what enables text -> tabulation -> music.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import audio
from .texture import Medium, ModBand, _bandnoise, _modulate, _norm_rms, _residue, _tonal

FUSION_RATE = 25.0  # events/sec ceiling: above this, grains smear into noise


@dataclass(frozen=True)
class Component:
    """One spatial component of a feature, placed by the three orthogonal axes."""
    name: str
    freq: float                              # size  -> pitch (center Hz)
    roughness: float = 0.0                   # surface: 0 = pure tone, 1 = noise
    mode: str = "sustained"                  # "sustained" | "granular"
    grain_rate: float = 0.0                  # occurrence: events/sec (granular)
    grain_dur: float = 0.12                  # seconds per grain
    grain_jitter: float = 0.4                # 0 = metronomic, 1 = clustered/natural
    modulation: list[ModBand] = field(default_factory=list)


@dataclass(frozen=True)
class Feature:
    name: str
    components: list[Component]
    medium: Medium = field(default_factory=Medium)
    dynamic: bool = False                    # static (scanned) vs dynamic (process)


@dataclass(frozen=True)
class PerspectiveKey:
    """A keyframe of the observer's experience: target's gain ramps g0->g1 over
    [t0, t1]. ``target`` is "feature.component" or "feature.*". ``g0=None`` means
    'continue from the held value' (the '-> x' rows)."""
    t0: float
    t1: float
    target: str
    g1: float
    g0: float | None = None


@dataclass(frozen=True)
class Tabulation:
    title: str
    duration: float
    features: list[Feature]
    perspective: list[PerspectiveKey]


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _grain(freq: float, roughness: float, glen: int, sr: int,
           rng: np.random.Generator) -> np.ndarray:
    t = np.arange(glen) / sr
    tone = _norm_rms(np.sin(2 * np.pi * freq * t).astype(np.float32)
                     + 0.5 * np.sin(2 * np.pi * 2 * freq * t).astype(np.float32))
    noise = _bandnoise(glen, freq * 0.6, freq * 1.9, sr, rng)
    x = (1.0 - roughness) * tone + roughness * noise
    env = np.exp(-t / max(t[-1] / 3.0, 1e-4)).astype(np.float32)
    a = max(1, int(0.005 * sr))
    env[:a] *= np.linspace(0.0, 1.0, a, dtype=np.float32)
    return (x * env).astype(np.float32)


def _render_component(comp: Component, dur: float, sr: int, seed: int) -> np.ndarray:
    n = int(dur * sr)
    rng = np.random.default_rng(seed)
    if comp.mode == "granular":
        x = np.zeros(n, dtype=np.float32)
        rate = min(comp.grain_rate, FUSION_RATE)        # the fusion cap
        glen = max(1, int(comp.grain_dur * sr))
        t = 0.0
        while t < dur:
            start = int(t * sr)
            end = min(n, start + glen)
            if end > start:
                detune = comp.freq * float(np.exp(rng.normal(0, 0.03)))  # needles vary
                x[start:end] += _grain(detune, comp.roughness, end - start, sr, rng)
            mean = 1.0 / rate
            # lognormal intervals: jitter -> natural clustering; never below fusion
            interval = mean * float(np.exp(rng.normal(0, comp.grain_jitter)))
            t += max(interval, 1.0 / FUSION_RATE)
    else:
        tonal = _tonal(n, comp.freq, sr)
        noise = _bandnoise(n, comp.freq * 0.6, comp.freq * 1.8, sr, rng)
        x = (1.0 - comp.roughness) * tonal + comp.roughness * noise
    x = _modulate(x, sr, comp.modulation, seed)
    return _norm_rms(x).astype(np.float32) * 0.5


def _gain_env(target_feat: str, target_comp: str, keys: list[PerspectiveKey],
              dur: float, n: int, control: int = 1024) -> np.ndarray:
    """Piecewise-linear gain for one component: ramp inside keyed spans, hold the
    last value between them, 0 before the first key."""
    matches = sorted(
        [k for k in keys
         if k.target == f"{target_feat}.{target_comp}" or k.target == f"{target_feat}.*"],
        key=lambda k: k.t0,
    )
    if not matches:
        return np.zeros(n, dtype=np.float32)
    cn = max(2, control)
    ts = np.linspace(0.0, dur, cn)
    vals = np.zeros(cn, dtype=np.float32)
    held = 0.0
    ki = 0
    for i, t in enumerate(ts):
        while ki < len(matches) and t > matches[ki].t1:
            held = matches[ki].g1
            ki += 1
        k = matches[ki] if ki < len(matches) else None
        if k is not None and k.t0 <= t <= k.t1:
            g0 = held if k.g0 is None else k.g0
            frac = (t - k.t0) / max(k.t1 - k.t0, 1e-6)
            vals[i] = g0 + (k.g1 - g0) * frac
        else:
            vals[i] = held
    return np.interp(np.linspace(0, cn - 1, n), np.arange(cn), vals).astype(np.float32)


def render_tabulation(tab: Tabulation, sr: int = audio.SR, seed: int = 0) -> np.ndarray:
    n = int(tab.duration * sr)
    out = np.zeros(n, dtype=np.float32)
    for fi, feat in enumerate(tab.features):
        fbuf = np.zeros(n, dtype=np.float32)
        for ci, comp in enumerate(feat.components):
            buf = _render_component(comp, tab.duration, sr, seed + fi * 17 + ci)
            buf *= _gain_env(feat.name, comp.name, tab.perspective, tab.duration, n)
            fbuf += buf
        fbuf = _residue(fbuf, sr, feat.medium)
        out += fbuf
    peak = float(np.max(np.abs(out))) or 1.0
    return (out / peak * 0.99).astype(np.float32)


def describe_perspective(tab: Tabulation) -> str:
    """Render the perspective track as the table a text prompt would target."""
    lines = [f"Tabulation: {tab.title}  ({tab.duration:.0f}s)",
             f"{'time':>10}  {'target':<16} {'gain':>12}"]
    for k in tab.perspective:
        g0 = "·" if k.g0 is None else f"{k.g0:.2f}"
        lines.append(f"{k.t0:>4.0f}-{k.t1:<4.0f}s  {k.target:<16} {g0:>5} -> {k.g1:<.2f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Authored example: "a walk through a forest to a campfire".
# This IS the tabulation of a video; later, text generates these same rows.
# --------------------------------------------------------------------------- #

def forest_fire_walk() -> Tabulation:
    tree = Feature(
        name="tree",
        dynamic=False,
        medium=Medium(amount=0.10, persistence_s=0.3, darkness=0.85),
        components=[
            # size -> pitch | surface: bark gritty low, needles clean high (inverted!)
            Component("trunk", freq=60.0, roughness=0.25, mode="sustained",
                      modulation=[ModBand(rate=0.2, depth=0.20)]),
            Component("branches", freq=280.0, roughness=0.30, mode="granular",
                      grain_rate=8.0, grain_dur=0.12, grain_jitter=0.5),
            Component("canopy", freq=1800.0, roughness=0.08, mode="granular",
                      grain_rate=14.0, grain_dur=0.07, grain_jitter=0.45),
        ],
    )
    fire = Feature(
        name="fire",
        dynamic=True,
        components=[
            # turbulent at every scale -> broadband everywhere (noise is faithful)
            Component("body", freq=110.0, roughness=0.85, mode="sustained",
                      modulation=[ModBand(rate=0.3, depth=0.35)]),
            Component("tongues", freq=600.0, roughness=0.90, mode="sustained",
                      modulation=[ModBand(rate=0.3, depth=0.3),
                                  ModBand(rate=11.0, depth=0.6, chaos=0.85)]),
            Component("sparks", freq=3000.0, roughness=0.70, mode="granular",
                      grain_rate=10.0, grain_dur=0.05, grain_jitter=0.8),
        ],
    )
    perspective = [
        # 0-4s: enter the forest, look up (canopy) then down the trunk
        PerspectiveKey(0, 2, "tree.canopy", g1=0.6, g0=0.9),
        PerspectiveKey(0, 4, "tree.branches", g1=0.5, g0=0.5),
        PerspectiveKey(2, 4, "tree.trunk", g1=0.8, g0=0.3),
        # 4-8s: walk on; the whole tree recedes as a fire appears on the ground
        PerspectiveKey(4, 8, "tree.*", g1=0.2),
        PerspectiveKey(4, 8, "fire.body", g1=0.7, g0=0.0),
        # 8-12s: stand by the fire — tongues and sparks swell, body steady
        PerspectiveKey(8, 12, "fire.tongues", g1=0.9, g0=0.4),
        PerspectiveKey(8, 12, "fire.sparks", g1=0.8, g0=0.2),
        PerspectiveKey(8, 12, "fire.body", g1=0.7),
    ]
    return Tabulation(title="forest -> campfire walk", duration=12.0,
                      features=[tree, fire], perspective=perspective)
