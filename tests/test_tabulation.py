import numpy as np

from music_making import audio
from music_making.tabulation import (
    Component,
    Feature,
    PerspectiveKey,
    Tabulation,
    _gain_env,
    dumps,
    forest_fire_walk,
    loads,
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


def test_yaml_round_trip_is_lossless():
    """The editable YAML form parses back to the identical tabulation."""
    tab = forest_fire_walk()
    assert loads(dumps(tab)) == tab


def test_absurd_grain_rate_stays_finite_and_capped():
    """A granular component asked for an absurd rate renders finite (capped),
    not NaN/inf."""
    comp = Component("x", freq=2000.0, mode="granular", grain_rate=10_000.0)
    tab = Tabulation("t", 1.0, [Feature("f", [comp])],
                     [PerspectiveKey(0, 1, "f.*", g1=1.0, g0=1.0)])
    x = render_tabulation(tab, seed=0)
    assert np.all(np.isfinite(x)) and np.max(np.abs(x)) <= 1.0


def test_granular_zero_rate_does_not_crash():
    """A granular component with grain_rate=0 falls back to sustained instead of
    dividing by zero (regression)."""
    comp = Component("x", freq=300.0, mode="granular", grain_rate=0.0)
    tab = Tabulation("t", 1.0, [Feature("f", [comp])],
                     [PerspectiveKey(0, 1, "f.*", g1=1.0, g0=1.0)])
    x = render_tabulation(tab, seed=0)
    assert np.all(np.isfinite(x))


def test_gain_env_honors_span_nested_in_a_wider_one():
    """A per-component dip nested inside a feature.* sweep is applied, not dropped
    (regression for the single-pointer scan)."""
    keys = [PerspectiveKey(0, 10, "f.*", g1=1.0, g0=1.0),     # full-piece sweep
            PerspectiveKey(4, 6, "f.x", g1=0.0, g0=0.0)]      # nested dip to 0
    env = _gain_env("f", "x", keys, 10.0, 1000)
    assert env[500] < 0.2     # t=5: inside the dip
    assert env[200] > 0.8     # t=2: wildcard only
    assert env[800] > 0.8     # t=8: wildcard only
