from music_making import from_text
from music_making.contracts import LAYER_BANDS


def test_from_text_builds_valid_storyboard():
    sb = from_text("running fast through fire and heat", seed=3, duration_sec=16.0)
    assert sb.duration_sec > 0
    assert sb.sections and sum(s.bars for s in sb.sections) > 0
    assert len(sb.frames) == 64
    for f in sb.frames:
        for layer in LAYER_BANDS:
            assert 0.0 <= getattr(f, layer) <= 1.0


def test_layer_at_interpolates_and_clamps():
    sb = from_text("calm quiet night", seed=0, duration_sec=12.0)
    a = sb.layer_at("terrain", 0.0)
    b = sb.layer_at("terrain", 1.0)
    assert 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0
    # out-of-range times clamp to the endpoints
    assert sb.layer_at("terrain", -5.0) == sb.frames[0].terrain
    assert sb.layer_at("terrain", 5.0) == sb.frames[-1].terrain


def test_seed_is_deterministic():
    a = from_text("fire on the mountain", seed=7, duration_sec=12.0)
    b = from_text("fire on the mountain", seed=7, duration_sec=12.0)
    assert a.model_dump() == b.model_dump()


def test_heat_words_raise_atmosphere_vs_calm():
    hot = from_text("fire lava heat burn flame blaze danger", seed=1, duration_sec=12.0)
    calm = from_text("calm quiet still night rest mellow gentle", seed=1, duration_sec=12.0)
    hot_atm = sum(f.atmosphere for f in hot.frames)
    calm_atm = sum(f.atmosphere for f in calm.frames)
    assert hot_atm > calm_atm
