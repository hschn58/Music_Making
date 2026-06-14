import numpy as np

from music_making import audio, from_text, timbre
from music_making.timbre import TextureProfile


def _high_energy(x: np.ndarray) -> float:
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1 / audio.SR)
    return float((spec[freqs > 4500] ** 2).sum())


def _shorttime_rms_std(x: np.ndarray, win: int = 2048) -> float:
    n = (len(x) // win) * win
    frames = x[:n].reshape(-1, win)
    return float(np.std(np.sqrt((frames ** 2).mean(axis=1))))


def test_lowpass_profile_darkens_signal():
    sb = from_text("a calm night", seed=0, duration_sec=4.0)
    x = (np.random.default_rng(0).standard_normal(int(4 * audio.SR)) * 0.2).astype(np.float32)
    dark = TextureProfile("dark", cutoff_base=800, brightness_depth=0.0)
    assert _high_energy(timbre.apply(x, sb, dark)) < _high_energy(x) * 0.5


def test_apply_preserves_length_and_bounds():
    sb = from_text("bright fire and heat", seed=0, duration_sec=3.0)
    x = (np.random.default_rng(1).standard_normal(int(3 * audio.SR)) * 0.3).astype(np.float32)
    fire = TextureProfile("fire", cutoff_base=9000, brightness_depth=0.7, bandwidth=0.5,
                          drive_base=0.1, drive_depth=0.3, slow_rate=0.2, slow_depth=0.3,
                          fast_rate=11.0, fast_depth=0.6, chaos=0.85, reverb=0.2)
    y = timbre.apply(x, sb, fire)
    assert len(y) == len(x)
    assert np.max(np.abs(y)) <= 1.0


def test_residue_smears_a_burst_forward():
    sb = from_text("solid rock", seed=0, duration_sec=2.0)
    x = np.zeros(int(1.0 * audio.SR), dtype=np.float32)
    burst = int(0.05 * audio.SR)
    x[:burst] = np.random.default_rng(2).standard_normal(burst).astype(np.float32) * 0.5
    late = slice(int(0.2 * audio.SR), int(0.7 * audio.SR))

    plain = TextureProfile("plain", cutoff_base=12000, brightness_depth=0.0)
    viscous = TextureProfile("viscous", cutoff_base=12000, brightness_depth=0.0,
                             residue=0.6, residue_decay=0.5)
    e_plain = float(np.sum(timbre.apply(x, sb, plain)[late] ** 2))
    e_visc = float(np.sum(timbre.apply(x, sb, viscous)[late] ** 2))
    assert e_visc > e_plain * 2


def test_fast_flicker_adds_amplitude_movement():
    sb = from_text("bright fire", seed=0, duration_sec=2.0)
    t = np.arange(int(2.0 * audio.SR)) / audio.SR
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    still = TextureProfile("still", cutoff_base=12000, brightness_depth=0.0)
    flicker = TextureProfile("flicker", cutoff_base=12000, brightness_depth=0.0,
                             fast_rate=11.0, fast_depth=0.8, chaos=0.0)
    assert _shorttime_rms_std(timbre.apply(tone, sb, flicker)) > \
        _shorttime_rms_std(timbre.apply(tone, sb, still))
