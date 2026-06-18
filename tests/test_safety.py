import numpy as np

from music_making.audio import SR
from music_making.safety import PEAK_CEILING, master


def test_scrubs_nan_and_inf():
    x = np.array([0.0, np.nan, np.inf, -np.inf, 0.5], np.float32)
    assert np.all(np.isfinite(master(x, SR)))


def test_peak_is_capped():
    x = np.full(SR, 5.0, np.float32)            # absurd level
    assert np.max(np.abs(master(x, SR))) <= PEAK_CEILING + 1e-6


def test_infrasound_removed_audible_kept():
    """A resonance-band tone (8 Hz) is suppressed; an audible tone (220 Hz) survives."""
    t = np.arange(SR) / SR
    infra = np.sin(2 * np.pi * 8 * t)
    audible = np.sin(2 * np.pi * 220 * t)
    e_infra = np.sqrt(np.mean(master(infra, SR) ** 2))
    e_aud = np.sqrt(np.mean(master(audible, SR) ** 2))
    assert e_infra < 0.15 * e_aud
