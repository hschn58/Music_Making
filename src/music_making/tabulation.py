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
import yaml

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
    gain: float = 1.0                        # static mix weight (band energy)
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
    a = min(glen, max(1, int(0.005 * sr)))
    env[:a] *= np.linspace(0.0, 1.0, a, dtype=np.float32)
    return (x * env).astype(np.float32)


def _render_component(comp: Component, dur: float, sr: int, seed: int) -> np.ndarray:
    n = int(dur * sr)
    rng = np.random.default_rng(seed)
    if comp.mode == "granular" and comp.grain_rate > 0:
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
    return _norm_rms(x).astype(np.float32) * 0.5 * comp.gain


def _gain_env(target_feat: str, target_comp: str, keys: list[PerspectiveKey],
              dur: float, n: int, control: int = 1024) -> np.ndarray:
    """Piecewise-linear gain for one component: ramp inside keyed spans, hold the
    last value between them, 0 before the first key."""
    matches = [k for k in keys
               if k.target == f"{target_feat}.{target_comp}" or k.target == f"{target_feat}.*"]
    if not matches:
        return np.zeros(n, dtype=np.float32)
    cn = max(2, control)
    ts = np.linspace(0.0, dur, cn)
    vals = np.zeros(cn, dtype=np.float32)

    def held_at(t: float) -> float:
        """Value held outside any span: the most recently ended key, else 0."""
        ended = [k for k in matches if k.t1 <= t]
        return max(ended, key=lambda k: k.t1).g1 if ended else 0.0

    for i, t in enumerate(ts):
        # innermost open span (largest t0) wins, so a key nested inside a wider
        # span (e.g. a per-component dip inside a feature.* sweep) is honored.
        open_keys = [k for k in matches if k.t0 <= t <= k.t1]
        if open_keys:
            k = max(open_keys, key=lambda k: (k.t0, k.t1))
            g0 = held_at(k.t0) if k.g0 is None else k.g0
            frac = (t - k.t0) / max(k.t1 - k.t0, 1e-6)
            vals[i] = g0 + (k.g1 - g0) * frac
        else:
            vals[i] = held_at(t)
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


def describe_grid(tab: Tabulation, frame: float = 2.0) -> str:
    """Render the perspective as a time x component gain grid — see it at a glance."""
    pts = max(2, int(tab.duration * 50))
    cols, envs = [], []
    for f in tab.features:
        for c in f.components:
            cols.append(f"{f.name[:4]}.{c.name[:6]}")
            envs.append(_gain_env(f.name, c.name, tab.perspective, tab.duration, pts))
    lines = [f"grid: {tab.title}",
             f"{'time':>8} " + " ".join(f"{c:>11}" for c in cols)]
    n_fr = int((tab.duration - 1e-9) // frame) + 1
    for i in range(n_fr):
        t0 = i * frame
        t1 = min(tab.duration, t0 + frame)
        idx = min(pts - 1, int((t0 + t1) / 2 / tab.duration * (pts - 1)))
        row = " ".join(f"{envs[j][idx]:>11.2f}" for j in range(len(cols)))
        lines.append(f"{t0:>3.0f}-{t1:<3.0f}s {row}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# YAML I/O — the editable source of truth. Compact, hand-formatted on save so the
# file reads like the catalog/perspective it is; parsed with safe_load on read.
# --------------------------------------------------------------------------- #

def _n(x: float) -> str:
    return f"{x:g}"


def dumps(tab: Tabulation) -> str:
    """Serialize to the compact, human-editable YAML form."""
    out = [f"title: {tab.title}", f"duration: {_n(tab.duration)}", "features:"]
    for feat in tab.features:
        out.append(f"  {feat.name}:")
        if feat.dynamic:
            out.append("    dynamic: true")
        m = feat.medium
        if m.amount > 0:
            out.append(f"    medium: {{amount: {_n(m.amount)}, "
                       f"persistence_s: {_n(m.persistence_s)}, darkness: {_n(m.darkness)}}}")
        out.append("    components:")
        for c in feat.components:
            fields = [f"freq: {_n(c.freq)}", f"roughness: {_n(c.roughness)}", f"mode: {c.mode}"]
            if c.mode == "granular":
                fields.append(f"grain_rate: {_n(c.grain_rate)}")
                if c.grain_dur != 0.12:
                    fields.append(f"grain_dur: {_n(c.grain_dur)}")
                if c.grain_jitter != 0.4:
                    fields.append(f"grain_jitter: {_n(c.grain_jitter)}")
            if c.gain != 1.0:
                fields.append(f"gain: {_n(c.gain)}")
            if c.modulation:
                mods = ", ".join(
                    "{" + f"rate: {_n(mm.rate)}, depth: {_n(mm.depth)}"
                    + (f", chaos: {_n(mm.chaos)}" if mm.chaos else "") + "}"
                    for mm in c.modulation)
                fields.append(f"mod: [{mods}]")
            out.append(f"      {c.name}: {{{', '.join(fields)}}}")
    out.append("perspective:")
    for k in tab.perspective:
        g0 = "" if k.g0 is None else _n(k.g0)
        out.append(f"  - {{t: {_n(k.t0)}-{_n(k.t1)}, target: {k.target}, "
                   f"gain: {g0}->{_n(k.g1)}}}")
    return "\n".join(out) + "\n"


def _span(s: str) -> tuple[float, float]:
    a, b = str(s).split("-")
    return float(a), float(b)


def _gain(s: str) -> tuple[float | None, float]:
    a, b = str(s).split("->")
    return (None if a.strip() == "" else float(a)), float(b)


def loads(text: str) -> Tabulation:
    """Parse the YAML form back into a Tabulation."""
    d = yaml.safe_load(text)
    features = []
    for fname, fd in d["features"].items():
        comps = []
        for cname, cd in fd["components"].items():
            mods = [ModBand(rate=float(mm["rate"]), depth=float(mm.get("depth", 0.0)),
                            chaos=float(mm.get("chaos", 0.0))) for mm in cd.get("mod", [])]
            comps.append(Component(
                name=cname, freq=float(cd["freq"]), roughness=float(cd.get("roughness", 0.0)),
                mode=cd.get("mode", "sustained"), grain_rate=float(cd.get("grain_rate", 0.0)),
                grain_dur=float(cd.get("grain_dur", 0.12)),
                grain_jitter=float(cd.get("grain_jitter", 0.4)),
                gain=float(cd.get("gain", 1.0)), modulation=mods))
        md = fd.get("medium")
        medium = Medium(amount=float(md["amount"]), persistence_s=float(md["persistence_s"]),
                        darkness=float(md["darkness"])) if md else Medium()
        features.append(Feature(name=fname, components=comps, medium=medium,
                                dynamic=bool(fd.get("dynamic", False))))
    persp = []
    for row in d["perspective"]:
        t0, t1 = _span(row["t"])
        g0, g1 = _gain(row["gain"])
        persp.append(PerspectiveKey(t0=t0, t1=t1, target=row["target"], g1=g1, g0=g0))
    return Tabulation(title=d["title"], duration=float(d["duration"]),
                      features=features, perspective=persp)


def save_tabulation(tab: Tabulation, path: str) -> str:
    with open(path, "w") as fh:
        fh.write(dumps(tab))
    return path


def load_tabulation(path: str) -> Tabulation:
    with open(path) as fh:
        return loads(fh.read())


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
