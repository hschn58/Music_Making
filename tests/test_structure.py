import numpy as np

from music_making.color_spectrum import FREQS
from music_making.structure import structure, to_bark


def _bump(f0, amp=1.0, rel_width=0.02):
    return amp * np.exp(-0.5 * ((FREQS - f0) / (rel_width * f0)) ** 2)


def test_to_bark_is_monotonic():
    b = to_bark(FREQS)
    assert np.all(np.diff(b) > 0)


def test_well_separated_peaks_all_survive():
    E = _bump(200) + _bump(800) + _bump(3000)        # all > 1 Bark apart
    _, peaks = structure(E)
    assert len(peaks) == 3
    fs = sorted(p["f_hz"] for p in peaks)
    assert fs[0] < 300 and 700 < fs[1] < 900 and fs[2] > 2800


def test_too_close_peaks_collapse_to_one():
    E = _bump(1000) + _bump(1120)                    # ~0.7 Bark apart -> rough
    _, peaks = structure(E)
    assert len(peaks) == 1                            # raw-count greedy keeps one


def test_just_far_enough_peaks_both_survive():
    E = _bump(1000) + _bump(1300)                    # ~1.6 Bark apart -> clean
    _, peaks = structure(E)
    assert len(peaks) == 2


def test_output_is_finite_and_widths_floored():
    E = _bump(500) + _bump(2000)
    E_struct, peaks = structure(E, sigma_floor_bark=0.1)
    assert E_struct.shape == E.shape and np.all(np.isfinite(E_struct))
    assert all(p["amp"] > 0 and p["sigma_bark"] >= 0.1 for p in peaks)


def test_spacing_knob_controls_count():
    E = _bump(1000) + _bump(1300)
    loose = structure(E, spacing_bark=3.0)[1]        # demand more separation
    tight = structure(E, spacing_bark=0.5)[1]
    assert len(loose) <= len(tight)
