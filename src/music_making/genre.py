"""Genre presets.

Default targets early-'80s smooth/minimal funk-soul — the 'Cool Cat' (Queen,
*Hot Space*) sound: bass-forward, sparse, mellow, with a falsetto lead. GM
program numbers are 0-indexed (mido convention). Each preset also carries a
per-stream :class:`~music_making.timbre.TimbreKit` (terrain/entity/atmosphere),
which is configurable and scene-modulated.
"""

from dataclasses import dataclass, field

from .timbre import TimbreKit


def _smooth_funk_timbres() -> dict[str, TimbreKit]:
    return {
        # warm, rounded low end
        "terrain": TimbreKit("warm-low", cutoff_base=2200, brightness_depth=0.4,
                             drive_base=0.18, drive_depth=0.25, reverb=0.06),
        # present, slightly gritty mid — the 'voice' of the agents. Cutoff kept
        # below the high band so the mid stream stays out of atmosphere's territory.
        "entity_activity": TimbreKit("present-mid", cutoff_base=3500, brightness_depth=0.4,
                                     drive_base=0.10, drive_depth=0.35, reverb=0.16),
        # airy, shimmering, spacious top
        "atmosphere": TimbreKit("airy-high", cutoff_base=11000, brightness_depth=0.9,
                                drive_base=0.04, drive_depth=0.10, reverb=0.22,
                                tremolo_rate=5.5, tremolo_depth=0.16),
    }


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
    timbres: dict[str, TimbreKit] = field(default_factory=_smooth_funk_timbres)


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
