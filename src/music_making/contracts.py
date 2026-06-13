"""Pydantic IO contracts shared across the parallel workflows.

The root contract is the ``Storyboard``: a scene decomposed into layers that map
to frequency registers (the project's core idea — see DESIGN.md). Every workflow
reads the Storyboard; each downstream stage reads the upstream stage's typed
result. Keeping these contracts explicit is what lets the four workflows run in
parallel yet still cohere into one song.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Scene layers, in the order terrain -> entities -> atmosphere, paired with the
# audio frequency band each one drives.
LAYER_BANDS = {
    "terrain": "low",          # ground/structure -> bass + harmonic bed
    "entity_activity": "mid",  # anthropomorphized agents -> lead melody + vocals
    "atmosphere": "high",      # heat/danger/brightness -> hats, sustained highs
}


class Section(BaseModel):
    """A structural region of the song (intro/verse/chorus/...)."""

    name: str
    bars: int = Field(gt=0)


class SceneFrame(BaseModel):
    """One time-sampled point on the scene timeline. All values normalized 0..1."""

    t: float = Field(ge=0, le=1)  # normalized position in the song
    terrain: float = Field(ge=0, le=1)
    entity_activity: float = Field(ge=0, le=1)
    atmosphere: float = Field(ge=0, le=1)
    tension: float = Field(ge=0, le=1)
    brightness: float = Field(ge=0, le=1)


class EntityEvent(BaseModel):
    """A discrete agent 'call' — the anthropomorphized 'ba ba ba... Haaa'."""

    t: float = Field(ge=0, le=1)  # normalized time
    intensity: float = Field(ge=0, le=1)
    label: str


class Storyboard(BaseModel):
    """Root contract: the scene every workflow conditions on."""

    title: str
    situation: str
    genre: str
    seed: int
    tempo_bpm: int = Field(gt=0)
    key: str
    beats_per_bar: int = 4
    duration_sec: float = Field(gt=0)
    sections: list[Section]
    frames: list[SceneFrame]
    entity_events: list[EntityEvent]

    def layer_at(self, layer: str, t_norm: float) -> float:
        """Linearly interpolate a scene layer at a normalized time (0..1)."""
        frames = self.frames
        if not frames:
            return 0.0
        t = min(max(t_norm, 0.0), 1.0)
        if t <= frames[0].t:
            return getattr(frames[0], layer)
        for a, b in zip(frames, frames[1:]):
            if a.t <= t <= b.t:
                span = (b.t - a.t) or 1e-9
                w = (t - a.t) / span
                return getattr(a, layer) * (1 - w) + getattr(b, layer) * w
        return getattr(frames[-1], layer)


class Note(BaseModel):
    """A single pitched note in seconds-domain (consumed by render + vocals)."""

    midi: int
    start: float
    dur: float
    vel: int = 80


class Stem(BaseModel):
    name: str
    path: str


class LyricLine(BaseModel):
    section: str
    text: str
    syllables: list[str]


class LyricsResult(BaseModel):
    title: str
    lines: list[LyricLine]
    hook: list[str]  # the entity 'calls', e.g. ["ba", "ba", "ba", "ba", "haa"]
    source: str  # "llm" or "local"


class CompositionResult(BaseModel):
    midi_path: str
    bass_stem: Stem
    harmony_stem: Stem
    pad_stem: Stem
    lead_stem: Stem
    melody: list[Note]  # lead line, consumed by the vocals workflow


class BeatResult(BaseModel):
    drum_stem: Stem


class VocalResult(BaseModel):
    vocal_stem: Stem
    aligned_syllables: int


class QCBandScore(BaseModel):
    band: str
    layer: str
    correlation: float


class QCReport(BaseModel):
    passed: bool
    duration_sec: float
    integrated_lufs: float
    true_peak_db: float
    non_silent: bool
    mean_correlation: float
    band_scores: list[QCBandScore]
    notes: list[str] = Field(default_factory=list)


class Track(BaseModel):
    wav_path: str
    mp3_path: str
    metadata_path: str
    storyboard: Storyboard
    qc: QCReport
