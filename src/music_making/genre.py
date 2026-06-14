"""Genre presets.

Default targets early-'80s smooth/minimal funk-soul — the 'Cool Cat' (Queen,
*Hot Space*) sound: bass-forward, sparse, mellow, with a falsetto lead. GM
program numbers are 0-indexed (mido convention). Each preset also carries a
per-stream :class:`~music_making.timbre.TimbreKit` (terrain/entity/atmosphere),
which is configurable and scene-modulated.
"""

from dataclasses import dataclass, field

from .timbre import TextureProfile


def _smooth_funk_timbres() -> dict[str, TextureProfile]:
    return {
        # ROCK: homogeneous at macro + micro scale -> narrow/tonal, a gentle slow
        # in-and-out, no fast flicker; emerging from lava -> a viscous residue tail.
        "terrain": TextureProfile(
            "rock", cutoff_base=2000, brightness_depth=0.3, bandwidth=0.05,
            drive_base=0.18, drive_depth=0.20,
            slow_rate=0.3, slow_depth=0.30, fast_depth=0.0,
            residue=0.35, residue_decay=0.5, reverb=0.05),
        # AGENTS: conscious -> described by feeling, kept clean (the sparse unison
        # motif in the composition carries their character, not DSP texture).
        "entity_activity": TextureProfile(
            "agents", cutoff_base=3500, brightness_depth=0.4, bandwidth=0.05,
            drive_base=0.10, drive_depth=0.30, reverb=0.16),
        # FIRE: broadband + a fast, chaotic flicker (flames) modulated by a slow
        # drift (the centre of mass wandering over long periods).
        "atmosphere": TextureProfile(
            "fire", cutoff_base=11000, brightness_depth=0.9, bandwidth=0.55,
            drive_base=0.04, drive_depth=0.10,
            slow_rate=0.2, slow_depth=0.35, fast_rate=11.0, fast_depth=0.6, chaos=0.85,
            reverb=0.22),
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
    timbres: dict[str, TextureProfile] = field(default_factory=_smooth_funk_timbres)


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
