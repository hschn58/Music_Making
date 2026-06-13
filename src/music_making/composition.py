"""Composition workflow: harmony, bass, pad, and lead melody.

Each instrument is rendered to its own stem and deliberately mapped to a
frequency band that follows a scene layer (see DESIGN.md):
  bass  -> low  band, driven by `terrain`
  keys  -> mid  band, driven by `entity_activity`
  pad   -> high band, driven by `atmosphere`
  lead  -> mid  band, driven by `entity_activity` (also consumed by vocals)
"""

from __future__ import annotations

import random
from pathlib import Path

from . import audio, midi, theory
from .contracts import CompositionResult, Note, Stem, Storyboard
from .genre import get_preset

# 16-step (one bar of 16th notes) rhythmic templates.
BASS_STEPS = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0]
KEYS_STEPS = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]
# i - IV - i - V vamp (degrees into the mode), classic for a dorian groove
CHORD_LOOP = [0, 3, 0, 4]


def _layer(sb: Storyboard, layer: str, t: float) -> float:
    return sb.layer_at(layer, t / sb.duration_sec)


def compose(sb: Storyboard, workdir: str, soundfont: str | None = None) -> CompositionResult:
    preset = get_preset(sb.genre)
    rng = random.Random(sb.seed + 1)
    tonic = theory.parse_key(sb.key)
    mode = sb.key.split()[-1]

    scale_low = theory.scale_pitches(tonic, mode, octave=2)
    scale_mid = theory.scale_pitches(tonic, mode, octave=4)
    scale_high = theory.scale_pitches(tonic, mode, octave=5)

    sec_per_beat = 60.0 / sb.tempo_bpm
    step_sec = sec_per_beat / 4.0  # 16th note
    total_bars = sum(s.bars for s in sb.sections)
    swing = preset.swing * step_sec

    bass: list[Note] = []
    keys: list[Note] = []
    pad: list[Note] = []
    lead: list[Note] = []

    prev_lead_idx = rng.randrange(len(scale_mid))

    for bar in range(total_bars):
        degree = CHORD_LOOP[bar % len(CHORD_LOOP)]
        bar_start = bar * (sb.beats_per_bar * sec_per_beat)
        root = scale_low[degree]
        chord_mid = theory.diatonic_triad(scale_mid, degree)
        chord_high = theory.diatonic_triad(scale_high, degree)

        # Pad: one sustained high chord per bar, driven by atmosphere (HIGH band)
        atm = _layer(sb, "atmosphere", bar_start)
        if atm > 0.12:
            for p in chord_high:
                pad.append(Note(midi=p, start=bar_start, dur=4 * sec_per_beat * 0.98,
                                vel=int(30 + 55 * atm)))

        for step in range(16):
            t = bar_start + step * step_sec
            if step % 2 == 1:
                t += swing  # push off-beats for groove
            terr = _layer(sb, "terrain", t)
            ent = _layer(sb, "entity_activity", t)

            # Bass: LOW band, driven by terrain
            if BASS_STEPS[step] and terr > 0.12:
                pitch = root if rng.random() > 0.25 else root + 7  # occasional fifth
                bass.append(Note(midi=pitch, start=t, dur=step_sec * 1.6,
                                 vel=int(45 + 70 * terr)))

            # Keys: MID band, comping driven by entity activity
            if KEYS_STEPS[step]:
                vel = int(35 + 55 * (0.4 + 0.6 * ent))
                for p in chord_mid:
                    keys.append(Note(midi=p, start=t, dur=step_sec * 1.2, vel=vel))

            # Lead melody: MID band, denser/louder when entities are active
            play_lead = (step % 2 == 0) and (rng.random() < 0.35 + 0.5 * ent)
            if play_lead:
                prev_lead_idx = max(0, min(len(scale_mid) - 1,
                                           prev_lead_idx + rng.choice([-2, -1, 0, 1, 2])))
                pitch = scale_mid[prev_lead_idx]
                lead.append(Note(midi=pitch, start=t, dur=step_sec * 1.8,
                                 vel=int(45 + 60 * (0.4 + 0.6 * ent))))

    # Accent a lead note near each entity event (the anthropomorphized 'call')
    for ev in sb.entity_events:
        t = ev.t * sb.duration_sec
        idx = (prev_lead_idx + 2) % len(scale_mid)
        lead.append(Note(midi=scale_mid[idx], start=t, dur=step_sec * 2.5,
                         vel=int(70 + 50 * ev.intensity)))
    lead.sort(key=lambda n: n.start)

    wd = Path(workdir)
    stems = {}
    for name, notes, program, drum in [
        ("bass", bass, preset.bass_program, False),
        ("harmony", keys, preset.keys_program, False),
        ("pad", pad, preset.pad_program, False),
        ("lead", lead, preset.lead_program, False),
    ]:
        mid_path = str(wd / f"{name}.mid")
        wav_path = str(wd / f"{name}.wav")
        midi.write_instrument_midi(mid_path, notes, program, sb.tempo_bpm, channel=0, is_drum=drum)
        audio.render_midi(mid_path, wav_path, soundfont=soundfont)
        stems[name] = Stem(name=name, path=wav_path)

    return CompositionResult(
        midi_path=str(wd / "lead.mid"),
        bass_stem=stems["bass"],
        harmony_stem=stems["harmony"],
        pad_stem=stems["pad"],
        lead_stem=stems["lead"],
        melody=lead,
    )
