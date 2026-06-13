import numpy as np

from music_making import audio, timbre
from music_making import from_text
from music_making.timbre import TimbreKit


def _high_energy(x: np.ndarray) -> float:
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1 / audio.SR)
    return float((spec[freqs > 4500] ** 2).sum())


def test_lowpass_kit_darkens_signal():
    sb = from_text("a calm night", seed=0, duration_sec=4.0)
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(int(4 * audio.SR)) * 0.2).astype(np.float32)
    dark = TimbreKit("dark", cutoff_base=800, brightness_depth=0.0,
                     drive_base=0.0, drive_depth=0.0, reverb=0.0)
    y = timbre.apply(x, sb, dark)
    assert _high_energy(y) < _high_energy(x) * 0.5


def test_apply_preserves_length_and_bounds():
    sb = from_text("bright fire and heat", seed=0, duration_sec=3.0)
    x = (np.random.default_rng(1).standard_normal(int(3 * audio.SR)) * 0.3).astype(np.float32)
    kit = TimbreKit("k", cutoff_base=6000, brightness_depth=0.5, drive_base=0.2,
                    drive_depth=0.3, reverb=0.2, tremolo_rate=5.0, tremolo_depth=0.2)
    y = timbre.apply(x, sb, kit)
    assert len(y) == len(x)
    assert np.max(np.abs(y)) <= 1.0
