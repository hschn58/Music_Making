from music_making import from_text
from music_making import lyrics as lyrics_mod


def test_syllables_split_on_vowel_groups():
    assert lyrics_mod.syllables("soul") == ["soul"]  # one vowel group
    assert len(lyrics_mod.syllables("running")) == 2  # run-ning
    assert lyrics_mod.syllables("") == []


def test_generate_offline_uses_local_source():
    sb = from_text("moving slow through the city lights", seed=4, duration_sec=16.0)
    res = lyrics_mod.generate(sb)
    assert res.source == "local"
    assert res.lines, "expected at least one singable line"
    assert all(line.syllables for line in res.lines)
    assert res.hook  # the anthropomorphized 'call'


def test_lyrics_deterministic():
    sb = from_text("dancing in the dark", seed=9, duration_sec=16.0)
    a = lyrics_mod.generate(sb)
    b = lyrics_mod.generate(sb)
    assert a.model_dump() == b.model_dump()
