"""Audio rendering and analysis utilities.

Render MIDI -> WAV via fluidsynth, mix stems, master loudness via ffmpeg, and
compute the per-band energy envelopes the QC gate compares against the scene.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100

SOUNDFONT_CANDIDATES = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/default-GM.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
]

# Frequency bands paired with scene layers (see contracts.LAYER_BANDS).
# The high band starts well above the snare body so it is hats/sparkle-dominated
# (atmosphere), keeping the streams spectrally separated for the QC gate.
BANDS = {"low": (0.0, 220.0), "mid": (220.0, 2000.0), "high": (4500.0, SR / 2)}


def find_soundfont() -> str:
    env = os.environ.get("MUSIC_MAKING_SOUNDFONT")
    if env and Path(env).is_file():
        return env
    for c in SOUNDFONT_CANDIDATES:
        if Path(c).is_file():
            return c
    raise FileNotFoundError(
        "No SoundFont found. Install fluid-soundfont-gm or set MUSIC_MAKING_SOUNDFONT."
    )


def _require(tool: str) -> None:
    if shutil.which(tool) is None:
        raise RuntimeError(f"Required tool '{tool}' not found on PATH")


def render_midi(midi_path: str, out_wav: str, soundfont: str | None = None, gain: float = 0.8) -> str:
    _require("fluidsynth")
    sf2 = soundfont or find_soundfont()
    subprocess.run(
        ["fluidsynth", "-ni", "-g", str(gain), "-F", str(out_wav), "-r", str(SR), sf2, str(midi_path)],
        check=True,
        capture_output=True,
    )
    return out_wav


def load_wav(path: str) -> np.ndarray:
    data, sr = sf.read(str(path), always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR:
        import librosa

        data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=SR)
    return data.astype(np.float32)


def save_wav(path: str, samples: np.ndarray, sr: int = SR) -> str:
    sf.write(str(path), np.clip(samples, -1.0, 1.0), sr)
    return path


def pad_to(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


def mix_stems(stems: dict[str, np.ndarray], gains: dict[str, float]) -> np.ndarray:
    n = max((len(s) for s in stems.values()), default=0)
    out = np.zeros(n, dtype=np.float32)
    for name, s in stems.items():
        out += pad_to(s, n) * gains.get(name, 1.0)
    peak = float(np.max(np.abs(out))) if n else 0.0
    if peak > 1.0:
        out = out / peak * 0.99
    return out


def to_mp3(in_wav: str, out_mp3: str) -> str:
    _require("ffmpeg")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(in_wav), "-codec:a", "libmp3lame", "-q:a", "2", str(out_mp3)],
        check=True,
        capture_output=True,
    )
    return out_mp3


def measure_lufs(wav_path: str) -> float:
    _require("ffmpeg")
    p = subprocess.run(
        ["ffmpeg", "-i", str(wav_path), "-af", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    lufs = -70.0
    for line in p.stderr.splitlines():
        s = line.strip()
        if s.startswith("I:") and "LUFS" in s:
            try:
                lufs = float(s.split()[1])
            except ValueError:
                pass
    return lufs


def true_peak_db(samples: np.ndarray) -> float:
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    return 20.0 * np.log10(peak) if peak > 0 else -120.0


def _downsample(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == 0:
        return np.zeros(n)
    idx = np.linspace(0, len(x) - 1, n)
    return np.interp(idx, np.arange(len(x)), x)


def _smooth(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Light moving-average so per-onset spikes don't mask the macro arc."""
    if window <= 1 or len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def normalize_env(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-9:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def band_envelopes(samples: np.ndarray, sr: int = SR, n_points: int = 64) -> dict[str, np.ndarray]:
    """Normalized low/mid/high energy envelopes (each length ``n_points``)."""
    import librosa

    if len(samples) < sr // 10:
        samples = pad_to(samples, sr // 10)
    spec = np.abs(librosa.stft(samples.astype(np.float32), n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    out: dict[str, np.ndarray] = {}
    for band, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        energy = spec[mask, :].sum(axis=0)
        out[band] = normalize_env(_smooth(_downsample(energy, n_points)))
    return out


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a, b = normalize_env(a), normalize_env(b)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
