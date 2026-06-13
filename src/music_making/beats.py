"""Beats workflow: a sparse funk/boogie kit driven by the scene.

  kick  -> low  band, driven by `terrain`     (the ground you move across)
  hats  -> high band, driven by `atmosphere`  (heat/sparkle/danger)
  snare -> mid  band, steady backbeat
"""

from __future__ import annotations

import random
from pathlib import Path

from . import audio, midi
from .contracts import BeatResult, Note, Stem, Storyboard
from .genre import get_preset

KICK, SNARE, CHAT, OHAT = 36, 38, 42, 46

KICK_STEPS = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]
SNARE_STEPS = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]


def make_beats(sb: Storyboard, workdir: str, soundfont: str | None = None) -> BeatResult:
    preset = get_preset(sb.genre)
    rng = random.Random(sb.seed + 2)
    sec_per_beat = 60.0 / sb.tempo_bpm
    step_sec = sec_per_beat / 4.0
    total_bars = sum(s.bars for s in sb.sections)
    swing = preset.swing * step_sec

    notes: list[Note] = []
    for bar in range(total_bars):
        bar_start = bar * (sb.beats_per_bar * sec_per_beat)
        for step in range(16):
            t = bar_start + step * step_sec
            if step % 2 == 1:
                t += swing
            tn = t / sb.duration_sec
            terr = sb.layer_at("terrain", tn)
            atm = sb.layer_at("atmosphere", tn)

            if KICK_STEPS[step] and terr > 0.1:
                notes.append(Note(midi=KICK, start=t, dur=step_sec, vel=int(55 + 60 * terr)))
            if SNARE_STEPS[step]:
                notes.append(Note(midi=SNARE, start=t, dur=step_sec, vel=80))

            # Hats: every 8th always, upgrade to 16th when atmosphere is hot.
            on_eighth = step % 2 == 0
            on_sixteenth = atm > 0.5
            if (on_eighth or on_sixteenth) and atm > 0.08:
                drum = OHAT if (rng.random() < 0.12 and atm > 0.6) else CHAT
                notes.append(Note(midi=drum, start=t, dur=step_sec * 0.8,
                                  vel=int(30 + 55 * atm)))

    wd = Path(workdir)
    mid_path = str(wd / "drums.mid")
    wav_path = str(wd / "drums.wav")
    midi.write_instrument_midi(mid_path, notes, program=0, tempo_bpm=sb.tempo_bpm, is_drum=True)
    audio.render_midi(mid_path, wav_path, soundfont=soundfont)
    return BeatResult(drum_stem=Stem(name="drums", path=wav_path))
