import numpy as np

from music_making import audio
from music_making.tabulation import (
    Component,
    Feature,
    PerspectiveKey,
    Tabulation,
    forest_fire_walk,
    render_tabulation,
)


def _energy(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def test_renders_correct_length_and_non_silent():
    tab = forest_fire_walk()
    x = render_tabulation(tab, seed=0)
    assert len(x) == int(tab.duration * audio.SR)
    assert np.max(np.abs(x)) > 0.1


def test_gain_envelope_is_zero_before_first_key():
    """fire.body is keyed from 4s on; the first 3s should carry ~no fire body."""
    tab = forest_fire_walk()
    x = render_tabulation(tab, seed=0)
    sr = audio.SR
    # isolate fire by rendering a fire-only tabulation and comparing early vs late
    fire = [f for f in tab.features if f.name == "fire"][0]
    fire_only = Tabulation("fire", tab.duration, [fire], tab.perspective)
    fx = render_tabulation(fire_only, seed=0)
    early = _energy(fx[: 3 * sr])
    late = _energy(fx[8 * sr : 12 * sr])
    assert late > early * 3  # fire arrives only as you near it


def test_perspective_moves_the_foreground_over_time():
    """The tree dominates early; the fire dominates late — the walk is audible."""
    tab = forest_fire_walk()
    sr = audio.SR
    tree = [f for f in tab.features if f.name == "tree"][0]
    fire = [f for f in tab.features if f.name == "fire"][0]
    tx = render_tabulation(Tabulation("t", tab.duration, [tree], tab.perspective), seed=0)
    fx = render_tabulation(Tabulation("f", tab.duration, [fire], tab.perspective), seed=0)
    tree_early, tree_late = _energy(tx[: 3 * sr]), _energy(tx[9 * sr :])
    fire_early, fire_late = _energy(fx[: 3 * sr]), _energy(fx[9 * sr :])
    assert tree_early > fire_early   # forest first
    assert fire_late > tree_late     # fire last


def test_fusion_cap_limits_grain_rate():
    """A granular component asked for an absurd rate is capped (stays renderable,
    not a hard-clipped wall)."""
    comp = Component("x", freq=2000.0, mode="granular", grain_rate=10_000.0)
    feat = Feature("f", [comp])
    tab = Tabulation("t", 1.0, [feat],
                     [PerspectiveKey(0, 1, "f.x", g1=1.0, g0=1.0)])
    x = render_tabulation(tab, seed=0)
    assert np.max(np.abs(x)) <= 1.0
