"""Genre presets.

Default targets early-'80s smooth/minimal funk-soul — the 'Cool Cat' (Queen,
*Hot Space*) sound: bass-forward, sparse, mellow, with a falsetto lead. GM
program numbers are 0-indexed (mido convention).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GenrePreset:
    name: str
    tempo_bpm: int
    mode: str  # scale mode key into music_making.theory.MODES
    bass_program: int
    keys_program: int
    pad_program: int
    lead_program: int
    swing: float  # 0..1 swing pushed onto off-beats
    falsetto: bool  # shift sung vocals up an octave


SMOOTH_FUNK = GenrePreset(
    name="smooth-funk",
    tempo_bpm=104,
    mode="dorian",
    bass_program=33,  # Electric Bass (finger)
    keys_program=4,   # Electric Piano 1 (Rhodes)
    pad_program=89,   # Pad 2 (warm)
    lead_program=54,  # Synth Voice
    swing=0.12,
    falsetto=True,
)

PRESETS = {"smooth-funk": SMOOTH_FUNK}


def get_preset(name: str) -> GenrePreset:
    """Return a preset by name, defaulting to smooth-funk."""
    return PRESETS.get(name, SMOOTH_FUNK)
