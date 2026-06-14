"""Beats workflow: a sparse funk/boogie kit, split into per-stream stems.

  kick  -> low  band (terrain)     -- the ground you move across
  snare -> mid  band (entity)      -- steady backbeat
  hats  -> high band (atmosphere)  -- heat / sparkle / danger

Splitting the kit lets each piece carry its stream's timbre and ride the
story-driven mix automation independently.
"""

from __future__ import annotations

from pathlib import Path

from . import midi, timbre
from .contracts import BeatResult, Note, Stem, Storyboard
from .genre import get_preset

KICK, SNARE, CHAT, OHAT = 36, 38, 42, 46
KICK_STEPS = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]
SNARE_STEPS = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]


def make_beats(sb: Storyboard, workdir: str, soundfont: str | None = None,
               timbres: dict | None = None) -> BeatResult:
    preset = get_preset(sb.genre)
    kits = timbres or preset.timbres
    sec_per_beat = 60.0 / sb.tempo_bpm
    step_sec = sec_per_beat / 4.0
    total_bars = sum(s.bars for s in sb.sections)
    swing = preset.swing * step_sec

    kick: list[Note] = []
    snare: list[Note] = []
    hats: list[Note] = []
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
                kick.append(Note(midi=KICK, start=t, dur=step_sec, vel=int(55 + 60 * terr)))
            if SNARE_STEPS[step]:
                snare.append(Note(midi=SNARE, start=t, dur=step_sec, vel=80))
            # Hats only carry the atmosphere band, so drop them when it's low and
            # thicken to 16ths when it's hot — density tracks the scene, not just level.
            if atm > 0.2 and (step % 2 == 0 or atm > 0.5):
                drum = OHAT if (step == 14 and atm > 0.6) else CHAT  # deterministic open hat
                hats.append(Note(midi=drum, start=t, dur=step_sec * 0.8, vel=int(30 + 55 * atm)))

    wd = Path(workdir)
    spec = [("kick", kick, "terrain"), ("snare", snare, "entity_activity"),
            ("hats", hats, "atmosphere")]
    rendered: dict[str, Stem] = {}
    for name, notes, stream in spec:
        kit = kits[stream]
        mid_path = str(wd / f"{name}.mid")
        wav_path = str(wd / f"{name}.wav")
        midi.write_instrument_midi(mid_path, notes, program=0, tempo_bpm=sb.tempo_bpm, is_drum=True)
        timbre.render_stem(mid_path, wav_path, sb, kit, soundfont=soundfont)
        rendered[name] = Stem(name=name, path=wav_path)

    return BeatResult(kick_stem=rendered["kick"], snare_stem=rendered["snare"],
                      hats_stem=rendered["hats"])
