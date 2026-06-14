"""Per-stream timbre as a sonification of physical texture.

The principle (see DESIGN.md): a stream's timbre should describe the *real
element* as faithfully as possible — materials and processes by their **texture**
(spectral homogeneity + multi-timescale modulation + the medium they sit in),
conscious agents by **feeling** (kept clean here; their character is carried by
the sparse unison motif in the composition).

A ``TextureProfile`` captures:
  * spectral placement / homogeneity   (narrow tonal  <->  broadband noisy)
  * multi-timescale amplitude modulation (slow drift  +  fast/chaotic flicker)
  * a medium/residue tail               (the substance the element sits in)
  * saturation + space

Scene-modulated: drive follows `tension`, the low-pass opens with `brightness`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from . import audio
from .contracts import Storyboard


@dataclass(frozen=True)
class TextureProfile:
    name: str
    # spectral placement / homogeneity
    cutoff_base: float          # Hz low-pass baseline
    brightness_depth: float     # how much `brightness` opens the filter
    bandwidth: float = 0.0      # 0 = homogeneous/tonal, 1 = broadband/noisy
    # saturation
    drive_base: float = 0.0
    drive_depth: float = 0.0    # `tension` adds saturation
    # multi-timescale amplitude modulation
    slow_rate: float = 0.0      # Hz, slow drift (e.g. fire's centre-of-mass wander)
    slow_depth: float = 0.0
    fast_rate: float = 0.0      # Hz, fast flicker (e.g. flames)
    fast_depth: float = 0.0
    chaos: float = 0.0          # 0 = periodic LFO, 1 = noise-driven chaotic flicker
    # medium / residue (the substance the element emerges from)
    residue: float = 0.0        # amount of viscous decay tail
    residue_decay: float = 0.5  # seconds
    # space
    reverb: float = 0.0
    program: int | None = None  # optional GM program override


# Backwards-compatible alias (the concept used to be called a "kit").
TimbreKit = TextureProfile


def _layer_env(sb: Storyboard, layer: str, n: int, control: int = 1024) -> np.ndarray:
    cn = max(2, min(control, n))
    cvals = np.array([sb.layer_at(layer, i / (cn - 1)) for i in range(cn)], dtype=np.float32)
    return np.interp(np.linspace(0, cn - 1, n), np.arange(cn), cvals).astype(np.float32)


def _saturate(x: np.ndarray, drive_env: np.ndarray) -> np.ndarray:
    k = 1.0 + drive_env * 5.0
    return (np.tanh(x * k) / np.tanh(np.maximum(k, 1e-6))).astype(np.float32)


def _sine_lfo(n: int, rate: float, sr: int) -> np.ndarray:
    t = np.arange(n) / sr
    return (0.5 * (1.0 + np.sin(2 * np.pi * rate * t))).astype(np.float32)


def _noise_lfo(n: int, rate: float, sr: int, seed: int) -> np.ndarray:
    """Chaotic flicker: white noise band-limited to ~`rate`, normalized 0..1."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n).astype(np.float32)
    cutoff = min(0.99, max(rate, 0.5) / (sr / 2))
    sos = signal.butter(2, cutoff, btype="low", output="sos")
    f = signal.sosfilt(sos, w)
    f -= f.min()
    return (f / (f.max() or 1.0)).astype(np.float32)


def _modulate(x: np.ndarray, sr: int, p: TextureProfile) -> np.ndarray:
    env = np.ones(len(x), dtype=np.float32)
    if p.slow_depth > 0 and p.slow_rate > 0:
        env *= 1.0 - p.slow_depth * (1.0 - _sine_lfo(len(x), p.slow_rate, sr))
    if p.fast_depth > 0 and p.fast_rate > 0:
        per = _sine_lfo(len(x), p.fast_rate, sr)
        if p.chaos > 0:
            fast = p.chaos * _noise_lfo(len(x), p.fast_rate, sr, seed=1234) + (1 - p.chaos) * per
        else:
            fast = per
        env *= 1.0 - p.fast_depth * (1.0 - fast)
    return (x * env).astype(np.float32)


def _broadband(x: np.ndarray, sr: int, p: TextureProfile) -> np.ndarray:
    """Heterogeneity: add high-frequency crackle that follows the signal's
    amplitude (e.g. the broadband texture of fire)."""
    if p.bandwidth <= 0:
        return x
    amp_sos = signal.butter(2, 20 / (sr / 2), btype="low", output="sos")
    amp = signal.sosfilt(amp_sos, np.abs(x)).astype(np.float32)
    rng = np.random.default_rng(777)
    noise = rng.standard_normal(len(x)).astype(np.float32)
    hp = signal.butter(2, 2000 / (sr / 2), btype="high", output="sos")
    noise = signal.sosfilt(hp, noise).astype(np.float32)
    return (x + p.bandwidth * 0.5 * noise * amp).astype(np.float32)


def _lp_timevarying(x: np.ndarray, sb: Storyboard, cutoff_base: float, depth: float,
                    block: int = 8192) -> np.ndarray:
    n = len(x)
    if n == 0:
        return x
    nyq = audio.SR / 2.0
    out = np.empty_like(x)
    zi = None
    pos = 0
    while pos < n:
        end = min(n, pos + block)
        br = sb.layer_at("brightness", ((pos + end) / 2.0) / n)
        cutoff = cutoff_base * (0.4 + 1.3 * depth * br) + cutoff_base * 0.2
        cutoff = max(200.0, min(nyq * 0.95, cutoff))
        sos = signal.butter(2, cutoff / nyq, btype="low", output="sos")
        if zi is None:
            zi = signal.sosfilt_zi(sos) * x[pos]
        out[pos:end], zi = signal.sosfilt(sos, x[pos:end], zi=zi)
        pos = end
    return out


def _residue(x: np.ndarray, sr: int, p: TextureProfile) -> np.ndarray:
    """Viscous medium tail: a dark, smeared decay after each hit — e.g. the lava
    the rock emerges from, persisting for ~`residue_decay` seconds."""
    if p.residue <= 0:
        return x
    length = max(1, int(p.residue_decay * sr))
    t = np.arange(length) / sr
    ir = np.exp(-t / (p.residue_decay / 3.0)).astype(np.float32)
    tail = signal.fftconvolve(x, ir)[: len(x)].astype(np.float32)
    dark = signal.butter(2, 600 / (sr / 2), btype="low", output="sos")
    tail = signal.sosfilt(dark, tail).astype(np.float32)
    tail *= (np.max(np.abs(x)) + 1e-9) / (np.max(np.abs(tail)) + 1e-9)  # match level
    return ((1.0 - p.residue * 0.5) * x + p.residue * tail).astype(np.float32)


_IR: np.ndarray | None = None


def _impulse_response() -> np.ndarray:
    global _IR
    if _IR is None:
        rng = np.random.default_rng(0)
        length = int(0.45 * audio.SR)
        t = np.arange(length) / audio.SR
        ir = (rng.standard_normal(length) * np.exp(-t / 0.12)).astype(np.float32)
        _IR = ir / (np.sqrt(np.sum(ir ** 2)) + 1e-9)
    return _IR


def _reverb(x: np.ndarray, wet: float) -> np.ndarray:
    if wet <= 0:
        return x
    tail = signal.fftconvolve(x, _impulse_response())[: len(x)].astype(np.float32)
    return ((1.0 - wet) * x + wet * 0.6 * tail).astype(np.float32)


def apply(samples: np.ndarray, sb: Storyboard, p: TextureProfile) -> np.ndarray:
    x = samples.astype(np.float32)
    if len(x) == 0:
        return x
    drive_env = p.drive_base + p.drive_depth * _layer_env(sb, "tension", len(x))
    x = _saturate(x, drive_env)
    x = _modulate(x, audio.SR, p)        # slow drift + fast/chaotic flicker
    x = _broadband(x, audio.SR, p)       # heterogeneity / crackle
    x = _lp_timevarying(x, sb, p.cutoff_base, p.brightness_depth)
    x = _residue(x, audio.SR, p)         # viscous medium tail
    x = _reverb(x, p.reverb)
    peak = float(np.max(np.abs(x))) or 1.0
    if peak > 1.0:
        x = x / peak * 0.99
    return x


def render_stem(midi_path: str, wav_path: str, sb: Storyboard, profile: TextureProfile,
                soundfont: str | None = None) -> str:
    """Render a MIDI stem and stamp the stream's texture onto it."""
    audio.render_midi(midi_path, wav_path, soundfont=soundfont)
    audio.save_wav(wav_path, apply(audio.load_wav(wav_path), sb, profile))
    return wav_path
