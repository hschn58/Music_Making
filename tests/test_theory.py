from music_making import theory


def test_parse_key():
    assert theory.parse_key("A dorian") == 9
    assert theory.parse_key("C minor") == 0
    assert theory.parse_key("Bb dorian") == 10


def test_scale_pitches_dorian():
    scale = theory.scale_pitches(9, "dorian", octave=4)  # A dorian
    assert len(scale) == 7
    # A dorian intervals from A
    assert [p - scale[0] for p in scale] == [0, 2, 3, 5, 7, 9, 10]


def test_diatonic_triad_stacks_thirds():
    scale = theory.scale_pitches(0, "ionian", octave=4)  # C major
    triad = theory.diatonic_triad(scale, 0)
    # C major triad: root, major third, perfect fifth
    assert [t - triad[0] for t in triad] == [0, 4, 7]
