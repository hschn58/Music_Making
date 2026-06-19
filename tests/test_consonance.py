import math

import numpy as np

from music_making.consonance import (
    consonance_axes,
    peak_partials,
    relative_period,
    roughness,
)


def _harmonic(f0, n=6):
    f = np.array([f0 * k for k in range(1, n + 1)], float)
    return f, np.ones(n)


def test_roughness_beating_pair_is_rougher_than_a_fifth():
    beat = roughness([440.0, 463.0], [1.0, 1.0])      # ~23 Hz apart, in critical band
    fifth = roughness([440.0, 660.0], [1.0, 1.0])     # 3:2, partials well separated
    assert beat > fifth > 0.0


def test_relative_period_of_simple_intervals():
    assert relative_period([220.0, 440.0]) == 1        # octave 2:1
    assert relative_period([220.0, 330.0]) == 2        # fifth 3:2
    assert relative_period([220.0, 275.0, 330.0]) == 4  # major triad 4:5:6


def test_tritone_is_intermediate_and_inharmonic_is_aperiodic():
    fifth = relative_period([220.0, 330.0])
    tritone = relative_period([440.0, 622.0])
    inharm = relative_period([233.0 * r for r in (1, 1.41, 1.73, 2.13, 2.78, 3.33)],
                             cents_tol=15.0, qcap=200)
    assert tritone > fifth                              # tritone doesn't lock in as fast
    assert inharm > tritone                             # inharmonic is far worse / aperiodic


def test_cents_tolerance_is_a_dial():
    # a loose tolerance snaps the tritone to a simpler ratio -> shorter period
    tight = relative_period([440.0, 622.0], cents_tol=5.0)
    loose = relative_period([440.0, 622.0], cents_tol=25.0)
    assert loose <= tight


def test_harmonic_tone_is_smooth_and_periodic():
    f, a = _harmonic(220.0)
    assert relative_period(f) == 1                      # integer harmonics -> period 1
    assert roughness(f, a) < roughness([440.0, 463.0], [1.0, 1.0])


def test_peak_partials_extracts_bumps():
    E = np.zeros_like(np.geomspace(60, 5000, 900))
    for c in (100, 400, 700):                           # three planted peaks
        E[c] = 1.0
    pf, pa = peak_partials(E)
    assert 1 <= pf.size <= 5 and np.all(pa > 0)


def test_consonance_axes_on_dense_spectrum_is_finite_typed():
    from music_making.color_spectrum import feature_spectrum

    rgb = np.tile([1.0, 0.0, 0.0], (2000, 1))          # a red swatch
    _, E, _ = feature_spectrum(rgb)
    out = consonance_axes(E)
    assert out["n_partials"] >= 1 and math.isfinite(out["roughness"])
