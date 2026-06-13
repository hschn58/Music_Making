"""Mix and master the stems into a single track.

Two scene-driven things happen here:
  1. story automation — each stem's level rides its stream's scene layer, so the
     dominant stream comes forward in its segment (more bass on rock, more highs
     on fire, more mid when facing the moving 'agents').
  2. mastering — a single constant gain to a target loudness. Constant (not
     dynamic) gain is deliberate: the scene's energy arc is the music, so the
     temporal envelope the QC gate checks must be preserved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import audio
from .contracts import STEM_STREAM, Stem, Storyboard

GAINS = {
    "bass": 1.0,
    "kick": 0.95,
    "harmony": 0.6,
    "lead": 0.6,
    "snare": 0.7,
    "vocals": 0.95,
    "pad": 0.5,
    "hats": 0.45,
}

TAIL_SEC = 0.5
MAX_GAIN_DB = 12.0
AUTOMATION_FLOOR = 0.5   # a stream never fully disappears
AUTOMATION_RANGE = 0.85  # how much the scene layer lifts a stream


def _automation_env(sb: Storyboard, layer: str, n: int, control: int = 1024) -> np.ndarray:
    cn = max(2, min(control, n))
    cv = np.array([sb.layer_at(layer, i / (cn - 1)) for i in range(cn)], dtype=np.float32)
    env = np.interp(np.linspace(0, cn - 1, n), np.arange(cn), cv)
    return (AUTOMATION_FLOOR + AUTOMATION_RANGE * env).astype(np.float32)


def mix(stems: dict[str, Stem], workdir: str, sb: Storyboard,
        target_lufs: float = -14.0) -> tuple[str, str]:
    wd = Path(workdir)
    n = int((sb.duration_sec + TAIL_SEC) * audio.SR)
    bed = np.zeros(n, dtype=np.float32)
    for name, stem in stems.items():
        x = audio.pad_to(audio.load_wav(stem.path), n)
        layer = STEM_STREAM.get(name)
        if layer:
            x = x * _automation_env(sb, layer, n)
        bed += x * GAINS.get(name, 0.7)

    peak = float(np.max(np.abs(bed))) if n else 0.0
    if peak > 1.0:
        bed = bed / peak * 0.99

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
