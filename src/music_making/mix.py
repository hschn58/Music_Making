"""Mix and master the stems into a single track.

Sum the stems with genre-appropriate gains (bass-forward, vocals upfront for the
smooth-funk target), then master to a target integrated loudness with a single
constant gain. Constant (not dynamic) gain is deliberate: the scene's energy arc
*is* the music, so we must preserve the temporal envelope the QC gate checks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import audio
from .contracts import Stem

GAINS = {
    "bass": 1.0,
    "drums": 0.8,
    "harmony": 0.6,
    "lead": 0.5,
    "pad": 0.45,
    "vocals": 0.95,
}

TAIL_SEC = 0.5  # allow a short natural release past the scene length
MAX_GAIN_DB = 12.0


def mix(stems: dict[str, Stem], workdir: str, duration_sec: float,
        target_lufs: float = -14.0) -> tuple[str, str]:
    wd = Path(workdir)
    loaded = {name: audio.load_wav(stem.path) for name, stem in stems.items()}
    bed = audio.mix_stems(loaded, GAINS)

    n = int((duration_sec + TAIL_SEC) * audio.SR)
    bed = audio.pad_to(bed, n)

    # Measure loudness on the trimmed bed, then apply one constant gain.
    raw = str(wd / "mix_raw.wav")
    audio.save_wav(raw, bed)
    measured = audio.measure_lufs(raw)
    gain_db = max(-MAX_GAIN_DB, min(MAX_GAIN_DB, target_lufs - measured))
    mastered = bed * (10.0 ** (gain_db / 20.0))
    peak = float(np.max(np.abs(mastered))) if len(mastered) else 0.0
    if peak > 0.99:
        mastered = mastered / peak * 0.99

    wav_path = str(wd / "track.wav")
    audio.save_wav(wav_path, mastered)
    mp3_path = str(wd / "track.mp3")
    audio.to_mp3(wav_path, mp3_path)
    return wav_path, mp3_path
