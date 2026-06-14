"""Composition workflow: harmony, bass, pad, and a motif-driven lead.

Each instrument is rendered to its own stem, mapped to a frequency band, and
gated by its scene layer so the arrangement *thins and thickens with the story*:
  bass  -> low  band, driven by `terrain`
  keys  -> mid  band, driven by `entity_activity`
  pad   -> high band, driven by `atmosphere`
  lead  -> mid  band: a recurring motif that develops across story segments
The stream's TimbreKit is stamped onto each stem as it is rendered.
"""

from __future__ import annotations

import random
from pathlib import Path

from . import audio, midi, theory, timbre
from .contracts import CompositionResult, Note, Stem, Storyboard
from .genre import get_preset

BASS_STEPS = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0]
KEYS_STEPS = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]
CHORD_LOOP = [0, 3, 0, 4]  # i - IV - i - V vamp


def _layer(sb: Storyboard, layer: str, t: float) -> float:
    return sb.layer_at(layer, t / sb.duration_sec)


def _bar_sections(sb: Storyboard) -> list[int]:
    out: list[int] = []
    for i, s in enumerate(sb.sections):
        out.extend([i] * s.bars)
    return out


def _make_motif(rng: random.Random, length: int = 5) -> list[tuple[int, int]]:
    """A short theme: (scale-degree offset, duration in 16th steps)."""
    offs = [0]
    for _ in range(length - 1):
        offs.append(offs[-1] + rng.choice([-2, -1, 1, 2, 1]))
    durs = [rng.choice([2, 2, 4]) for _ in range(length)]
    return list(zip(offs, durs))


def _transform(motif, section_idx, dominant_entity):
    """Develop the motif per story segment (retrograde, thin out)."""
    m = list(reversed(motif)) if section_idx % 2 == 1 else list(motif)
    if not dominant_entity:
        m = m[::2]  # thinner where entities aren't in the foreground
    return m


def compose(sb: Storyboard, workdir: str, soundfont: str | None = None,
            timbres: dict | None = None) -> CompositionResult:
    preset = get_preset(sb.genre)
    kits = timbres or preset.timbres
    rng = random.Random(sb.seed + 1)
    tonic = theory.parse_key(sb.key)
    mode = sb.key.split()[-1]

    scale_low = theory.scale_pitches(tonic, mode, octave=2)
    scale_mid = theory.scale_pitches(tonic, mode, octave=4)
    scale_high = theory.scale_pitches(tonic, mode, octave=5)
    scale_shimmer = theory.scale_pitches(tonic, mode, octave=6)
    ext_mid = scale_mid + [p + 12 for p in scale_mid]

    sec_per_beat = 60.0 / sb.tempo_bpm
    step_sec = sec_per_beat / 4.0
    total_bars = sum(s.bars for s in sb.sections)
    swing = preset.swing * step_sec
    bar_sec = sb.beats_per_bar * sec_per_beat
    bar_sec_map = _bar_sections(sb)
    seg_dominant = [s.dominant for s in sb.story.segments]
    motif = _make_motif(rng)

    bass: list[Note] = []
    keys: list[Note] = []
    pad: list[Note] = []
    lead: list[Note] = []

    for bar in range(total_bars):
        degree = CHORD_LOOP[bar % len(CHORD_LOOP)]
        bar_start = bar * bar_sec
        root = scale_low[degree]
        chord_mid = theory.diatonic_triad(scale_mid, degree)
        chord_high = theory.diatonic_triad(scale_high, degree)
        chord_shimmer = theory.diatonic_triad(scale_shimmer, degree)
        sec_idx = bar_sec_map[bar] if bar < len(bar_sec_map) else 0
        dom_entity = seg_dominant[sec_idx] == "entity_activity" if sec_idx < len(seg_dominant) else False

        # Pad + airy shimmer: HIGH band, sustained, swelling with atmosphere
        atm = _layer(sb, "atmosphere", bar_start)
        if atm > 0.18:
            for p in chord_high:
                pad.append(Note(midi=p, start=bar_start, dur=bar_sec * 0.98,
                                vel=int(28 + 55 * atm)))
            for p in chord_shimmer[:2]:  # high, harmonic-rich shimmer for the top band
                pad.append(Note(midi=p, start=bar_start, dur=bar_sec * 0.98,
                                vel=int(18 + 45 * atm)))

        for step in range(16):
            t = bar_start + step * step_sec
            if step % 2 == 1:
                t += swing
            terr = _layer(sb, "terrain", t)
            ent = _layer(sb, "entity_activity", t)

            if BASS_STEPS[step] and terr > 0.12:
                pitch = root if rng.random() > 0.25 else root + 7
                bass.append(Note(midi=pitch, start=t, dur=step_sec * 1.6,
                                 vel=int(45 + 70 * terr)))

            if KEYS_STEPS[step] and ent > 0.2:
                vel = int(35 + 55 * ent)
                for p in chord_mid:
                    keys.append(Note(midi=p, start=t, dur=step_sec * 1.2, vel=vel))

        # Lead: the motif, transformed per segment, laid over the bar
        theme = _transform(motif, sec_idx, dom_entity)
        step = 0
        for off, dur in theme:
            if step >= 16:
                break
            t = bar_start + step * step_sec
            ent = _layer(sb, "entity_activity", t)
            if ent > 0.22:
                idx = max(0, min(len(ext_mid) - 1, degree + off))
                lead.append(Note(midi=ext_mid[idx], start=t, dur=step_sec * dur * 0.95,
                                 vel=int(45 + 60 * ent)))
            step += dur

    # Accent the motif's head on each entity event (the 'ba...haa' call)
    for ev in sb.entity_events:
        t = ev.t * sb.duration_sec
        idx = max(0, min(len(ext_mid) - 1, motif[0][0] + 4))
        lead.append(Note(midi=ext_mid[idx], start=t, dur=step_sec * 3,
                         vel=int(70 + 50 * ev.intensity)))
    lead.sort(key=lambda n: n.start)

    wd = Path(workdir)
    spec = [
        ("bass", bass, preset.bass_program, "terrain"),
        ("harmony", keys, preset.keys_program, "entity_activity"),
        ("pad", pad, preset.pad_program, "atmosphere"),
        ("lead", lead, preset.lead_program, "entity_activity"),
    ]
    stems: dict[str, Stem] = {}
    for name, notes, default_prog, stream in spec:
        kit = kits[stream]
        program = kit.program if kit.program is not None else default_prog
        mid_path = str(wd / f"{name}.mid")
        wav_path = str(wd / f"{name}.wav")
        midi.write_instrument_midi(mid_path, notes, program, sb.tempo_bpm, channel=0)
        timbre.render_stem(mid_path, wav_path, sb, kit, soundfont=soundfont)
        stems[name] = Stem(name=name, path=wav_path)

    return CompositionResult(
        midi_path=str(wd / "lead.mid"),
        bass_stem=stems["bass"],
        harmony_stem=stems["harmony"],
        pad_stem=stems["pad"],
        lead_stem=stems["lead"],
        melody=lead,
    )
