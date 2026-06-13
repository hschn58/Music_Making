from music_making import from_text
from music_making.storyboard import ENTITY_W, _in_lex


def test_story_segments_shift_dominance():
    sb = from_text("solid rock ground, then fire and heat everywhere, then jumping enemies running",
                   seed=1, duration_sec=24.0)
    doms = [s.dominant for s in sb.story.segments]
    assert "terrain" in doms
    assert "atmosphere" in doms
    assert "entity_activity" in doms
    # sections are built 1:1 from story segments
    assert len(sb.sections) == len(sb.story.segments)


def test_stemming_matches_inflections():
    assert _in_lex("running", ENTITY_W)
    assert _in_lex("jumping", ENTITY_W)
    assert _in_lex("hopping", ENTITY_W)


def test_single_clause_still_makes_one_segment():
    sb = from_text("a quiet room", seed=0, duration_sec=12.0)
    assert len(sb.story.segments) >= 1
    assert len(sb.sections) == len(sb.story.segments)
