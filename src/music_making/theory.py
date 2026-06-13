"""Minimal music theory: parse a key, build a modal scale, build diatonic chords."""

NOTE_TO_PC = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
    "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

MODES = {
    "ionian": [0, 2, 4, 5, 7, 9, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "aeolian": [0, 2, 3, 5, 7, 8, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
}


def parse_key(key: str) -> int:
    """'A minor' / 'Bb dorian' -> tonic pitch class (0-11)."""
    root = key.strip().split()[0]
    return NOTE_TO_PC.get(root, 9)  # default A


def scale_pitches(tonic_pc: int, mode: str, octave: int) -> list[int]:
    """Return the 7 MIDI pitches of one octave of the mode starting at `octave`."""
    steps = MODES.get(mode, MODES["dorian"])
    base = 12 * (octave + 1) + tonic_pc  # MIDI: octave -1 == midi 0..11
    return [base + s for s in steps]


def diatonic_triad(scale: list[int], degree: int) -> list[int]:
    """Build a 1-3-5 triad on a scale degree (0-indexed), wrapping octaves."""
    ext = scale + [p + 12 for p in scale] + [p + 24 for p in scale]
    return [ext[degree], ext[degree + 2], ext[degree + 4]]
