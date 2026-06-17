import numpy as np

from music_making import audio
from music_making.texture import (
    FeatureTexture,
    ScaleBand,
    fire_texture,
    render_feature,
    rock_texture,
    tree_texture,
)


def _centroid(x: np.ndarray, sr: int = audio.SR) -> float:
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    return float((freqs * spec).sum() / (spec.sum() + 1e-9))


def _spread(x: np.ndarray, sr: int = audio.SR) -> float:
    """Spectral bandwidth: how widely energy is spread across the spectrum."""
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    c = (freqs * spec).sum() / (spec.sum() + 1e-9)
    return float(np.sqrt(((freqs - c) ** 2 * spec).sum() / (spec.sum() + 1e-9)))


def test_features_render_non_silent():
    for ft, f0 in [(rock_texture(), 55.0), (fire_texture(), 70.0), (tree_texture(), 70.0)]:
        x = render_feature(ft, f0, dur=2.0, seed=0)
        assert len(x) == int(2.0 * audio.SR)
        assert np.max(np.abs(x)) > 0.1, ft.name


def test_rock_is_darkest():
    """Rock is homogeneous and coarse -> the lowest spectral centroid of the set."""
    feats = {n: render_feature(f(), 70.0, 2.0, seed=0)
             for n, f in [("rock", rock_texture), ("tree", tree_texture), ("fire", fire_texture)]}
    cen = {n: _centroid(x) for n, x in feats.items()}
    assert cen["rock"] == min(cen.values())


def test_textured_features_are_wider_than_solid_rock():
    """Both the broadband fire and the fractal tree spread energy across more of
    the spectrum than the solid, homogeneous rock — that wider spread is 'depth'."""
    rock = _spread(render_feature(rock_texture(), 70.0, 2.0, seed=0))
    tree = _spread(render_feature(tree_texture(), 70.0, 2.0, seed=0))
    fire = _spread(render_feature(fire_texture(), 70.0, 2.0, seed=0))
    assert tree > rock
    assert fire > rock


def test_fractal_top_end_is_richer_than_a_single_scale():
    """A multiscale (fractal) description has more high-frequency energy than a
    one-scale feature at the same fundamental — that is the 'depth'."""
    flat = FeatureTexture(name="single", scales=[ScaleBand(energy=1.0, homogeneity=0.95)])
    one = _centroid(render_feature(flat, 70.0, 2.0, seed=0))
    many = _centroid(render_feature(tree_texture(), 70.0, 2.0, seed=0))
    assert many > one


def test_short_duration_and_degenerate_medium_are_safe():
    """Durations shorter than the attack/release windows, and a zero-persistence /
    out-of-range-darkness medium, render finite instead of crashing or NaN."""
    from music_making.texture import FeatureTexture, Medium, ScaleBand

    x = render_feature(rock_texture(), 110.0, 0.03, seed=0)   # < attack+release
    assert len(x) == int(0.03 * audio.SR) and np.all(np.isfinite(x))

    ft = FeatureTexture("x", scales=[ScaleBand(1.0)],
                        medium=Medium(amount=0.5, persistence_s=0.0, darkness=2.0))
    y = render_feature(ft, 110.0, 1.0, seed=0)
    assert np.all(np.isfinite(y))


def test_bloom_delays_fine_scales():
    """With bloom, the fine (high) scales enter later: the back half is brighter
    than the front half."""
    x = render_feature(tree_texture(), 70.0, 3.0, seed=0)
    half = len(x) // 2
    assert _centroid(x[half:]) > _centroid(x[:half])
