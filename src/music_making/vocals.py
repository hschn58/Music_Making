"""Vocals workflow: free-CPU singing synthesis.

espeak-ng speaks each syllable; we estimate its base pitch, pitch-shift it to the
melody note (an octave up for falsetto), time-stretch it to the note length, and
place it on the timeline. Robotic but genuinely pitched. The 'ba...haa' hook is
sung on notes that land on entity events (the anthropomorphized 'calls').
"""

from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

from . import audio
from .contracts import CompositionResult, LyricsResult, Stem, Storyboard, VocalResult
from .genre import get_preset

MAX_NOTES = 220  # keep synthesis bounded
EVENT_WINDOW = 0.18  # seconds: a note this close to an event sings the hook


def _espeak(text: str, wd: Path) -> np.ndarray:
    if shutil.which("espeak-ng") is None:
        raise RuntimeError("espeak-ng not found on PATH")
    out = wd / f"sy_{hashlib.md5(text.encode()).hexdigest()[:10]}.wav"
    subprocess.run(["espeak-ng", "-v", "en+f3", "-s", "150", "-p", "60", "-w", str(out), text],
                   check=True, capture_output=True)
    return audio.load_wav(str(out))


def _base_f0(y: np.ndarray) -> float:
    import warnings

    import librosa

    try:
        f0, _, _ = librosa.pyin(y, fmin=80, fmax=500, sr=audio.SR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            med = float(np.nanmedian(f0))
        if med > 0 and not math.isnan(med):
            return med
    except Exception:
        pass
    return 150.0


def _midi_to_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12.0)


def _fade(seg: np.ndarray, n: int = 256) -> np.ndarray:
    if len(seg) < 2 * n:
        return seg
    seg = seg.copy()
    ramp = np.linspace(0, 1, n, dtype=np.float32)
    seg[:n] *= ramp
    seg[-n:] *= ramp[::-1]
    return seg


def sing(sb: Storyboard, lyrics: LyricsResult, comp: CompositionResult, workdir: str,
         soundfont: str | None = None) -> VocalResult:
    import librosa

    preset = get_preset(sb.genre)
    wd = Path(workdir)
    total_samples = int(sb.duration_sec * audio.SR) + audio.SR
    buf = np.zeros(total_samples, dtype=np.float32)

    syl_pool = [s for ln in lyrics.lines for s in ln.syllables] or ["la"]
    notes = sorted(comp.melody, key=lambda n: n.start)[:MAX_NOTES]
    event_times = [ev.t * sb.duration_sec for ev in sb.entity_events]

    cache: dict[str, tuple[np.ndarray, float]] = {}
    si = hi = aligned = 0
    for note in notes:
        near_event = any(abs(note.start - et) <= EVENT_WINDOW for et in event_times)
        if near_event:
            text = lyrics.hook[hi % len(lyrics.hook)]
            hi += 1
        else:
            text = syl_pool[si % len(syl_pool)]
            si += 1

        if text not in cache:
            y = _espeak(text, wd)
            cache[text] = (y, _base_f0(y))
        y, base = cache[text]
        if len(y) < 64:
            continue

        target_m = note.midi + (12 if preset.falsetto else 0)
        while target_m > 76:
            target_m -= 12
        while target_m < 52:
            target_m += 12
        n_steps = 12 * math.log2(_midi_to_hz(target_m) / base)
        n_steps = max(-12.0, min(18.0, n_steps))
        shifted = librosa.effects.pitch_shift(y, sr=audio.SR, n_steps=n_steps)

        target_len = max(1, int(note.dur * audio.SR))
        if len(shifted) > 1:
            rate = min(4.0, max(0.25, len(shifted) / target_len))
            shifted = librosa.effects.time_stretch(shifted, rate=rate)
        seg = _fade(audio.pad_to(shifted, target_len)) * (note.vel / 127.0)

        start = int(note.start * audio.SR)
        end = min(len(buf), start + len(seg))
        buf[start:end] += seg[: end - start]
        aligned += 1

    peak = float(np.max(np.abs(buf))) or 1.0
    if peak > 1.0:
        buf = buf / peak * 0.99
    wav_path = str(wd / "vocals.wav")
    audio.save_wav(wav_path, buf)
    return VocalResult(vocal_stem=Stem(name="vocals", path=wav_path), aligned_syllables=aligned)
