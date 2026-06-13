"""Per-stream timbre.

Each frequency stream (terrain/low, entity/mid, atmosphere/high) has its own
spectral character, described by a configurable ``TimbreKit`` and modulated by
the scene over time:

  drive      <- tension     (saturation / grit)
  low-pass   <- brightness  (the filter opens as the scene brightens)
  reverb     -- space (per-kit)
  tremolo    -- movement (per-kit)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from . import audio
from .contracts import Storyboard


@dataclass(frozen=True)
class TimbreKit:
    name: str
    cutoff_base: float        # Hz, low-pass baseline
    brightness_depth: float   # how much `brightness` opens the filter
    drive_base: float         # static saturation
    drive_depth: float        # `tension` adds saturation
    reverb: float             # wet mix 0..1
    tremolo_rate: float = 0.0
    tremolo_depth: float = 0.0
    program: int | None = None  # optional GM program override for this stream


def _layer_env(sb: Storyboard, layer: str, n: int, control: int = 1024) -> np.ndarray:
    """A control-rate scene-layer envelope upsampled to ``n`` samples."""
    cn = max(2, min(control, n))
    cvals = np.array([sb.layer_at(layer, i / (cn - 1)) for i in range(cn)], dtype=np.float32)
    return np.interp(np.linspace(0, cn - 1, n), np.arange(cn), cvals).astype(np.float32)


def _saturate(x: np.ndarray, drive_env: np.ndarray) -> np.ndarray:
    k = 1.0 + drive_env * 5.0
    return (np.tanh(x * k) / np.tanh(np.maximum(k, 1e-6))).astype(np.float32)


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
        t_norm = ((pos + end) / 2.0) / n
        br = sb.layer_at("brightness", t_norm)
        cutoff = cutoff_base * (0.4 + 1.3 * depth * br) + cutoff_base * 0.2
        cutoff = max(200.0, min(nyq * 0.95, cutoff))
        sos = signal.butter(2, cutoff / nyq, btype="low", output="sos")
        if zi is None:
            zi = signal.sosfilt_zi(sos) * x[pos]
        out[pos:end], zi = signal.sosfilt(sos, x[pos:end], zi=zi)
        pos = end
    return out


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


def _tremolo(x: np.ndarray, rate: float, depth: float) -> np.ndarray:
    if rate <= 0 or depth <= 0:
        return x
    t = np.arange(len(x)) / audio.SR
    lfo = 1.0 - depth * 0.5 * (1.0 + np.sin(2 * np.pi * rate * t))
    return (x * lfo.astype(np.float32)).astype(np.float32)


def apply(samples: np.ndarray, sb: Storyboard, kit: TimbreKit) -> np.ndarray:
    x = samples.astype(np.float32)
    if len(x) == 0:
        return x
    drive_env = kit.drive_base + kit.drive_depth * _layer_env(sb, "tension", len(x))
    x = _saturate(x, drive_env)
    x = _lp_timevarying(x, sb, kit.cutoff_base, kit.brightness_depth)
    x = _tremolo(x, kit.tremolo_rate, kit.tremolo_depth)
    x = _reverb(x, kit.reverb)
    peak = float(np.max(np.abs(x))) or 1.0
    if peak > 1.0:
        x = x / peak * 0.99
    return x


def render_stem(midi_path: str, wav_path: str, sb: Storyboard, kit: TimbreKit,
                soundfont: str | None = None) -> str:
    """Render a MIDI stem and stamp the stream's timbre onto it."""
    audio.render_midi(midi_path, wav_path, soundfont=soundfont)
    audio.save_wav(wav_path, apply(audio.load_wav(wav_path), sb, kit))
    return wav_path
