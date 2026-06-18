import numpy as np

from music_making.color_spectrum import (
    FREQS,
    RED_HZ,
    VIOLET_HZ,
    feature_spectrum,
    synthesize,
)


def _swatch(rgb, n=2000):
    return np.tile(np.array(rgb, float), (n, 1))


def _centroid(E):
    return float((FREQS * E).sum() / (E.sum() + 1e-9))


def _spread(E):
    c = _centroid(E)
    return float(np.sqrt(((FREQS - c) ** 2 * E).sum() / (E.sum() + 1e-9)))


def test_red_is_lower_pitched_than_blue():
    _, er, _ = feature_spectrum(_swatch((1, 0, 0)))
    _, eb, _ = feature_spectrum(_swatch((0, 0, 1)))
    assert _centroid(er) < _centroid(eb)


def test_gray_is_broadband_and_red_is_peaked():
    """Desaturated -> wide spectrum (noise); saturated -> narrow (tone)."""
    _, er, _ = feature_spectrum(_swatch((1, 0, 0)))
    _, eg, _ = feature_spectrum(_swatch((0.5, 0.5, 0.5)))
    assert _spread(eg) > _spread(er)


def test_magenta_has_energy_at_both_ends():
    """A non-spectral purple = red + blue -> peaks low AND high, not in the middle."""
    _, em, _ = feature_spectrum(_swatch((1, 0, 1)))
    lo = em[np.argmin(np.abs(FREQS - RED_HZ))]
    hi = em[np.argmin(np.abs(FREQS - VIOLET_HZ))]
    mid = em[np.argmin(np.abs(FREQS - np.sqrt(RED_HZ * VIOLET_HZ)))]
    assert lo > mid and hi > mid


def test_render_is_finite_and_audible():
    freqs, e, _ = feature_spectrum(_swatch((0, 1, 0)))
    x = synthesize(e, freqs, dur=0.5)
    assert len(x) > 0 and np.all(np.isfinite(x)) and np.max(np.abs(x)) > 0.05


def test_spectrogram_resynth_is_finite_and_audible():
    from music_making.color_spectrum import spectrogram, synthesize_spectrogram

    # a walk from red to blue -> the column spectra evolve over time
    frames = [_swatch((1, 0, 0)), _swatch((0, 1, 0)), _swatch((0, 0, 1))]
    e_t = spectrogram(frames)
    x = synthesize_spectrogram(e_t, dur=1.0)
    assert e_t.shape[0] == 3 and np.all(np.isfinite(x)) and np.max(np.abs(x)) > 0.05
