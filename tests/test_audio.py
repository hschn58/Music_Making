import numpy as np

from music_making import audio


def test_correlation_identity_and_inverse():
    x = np.linspace(0, 1, 64)
    assert audio.correlation(x, x) > 0.99
    assert audio.correlation(x, x[::-1]) < -0.99
    assert audio.correlation(x, np.ones(64)) == 0.0  # constant -> no correlation


def test_band_envelopes_shape_and_range():
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(audio.SR).astype(np.float32) * 0.1
    env = audio.band_envelopes(sig, n_points=32)
    assert set(env) == {"low", "mid", "high"}
    for band in env.values():
        assert len(band) == 32
        assert band.min() >= 0.0 and band.max() <= 1.0


def test_mix_stems_prevents_clipping():
    a = np.ones(100, dtype=np.float32)
    b = np.ones(100, dtype=np.float32)
    out = audio.mix_stems({"a": a, "b": b}, {"a": 1.0, "b": 1.0})
    assert np.max(np.abs(out)) <= 1.0
