"""Write single-instrument MIDI files with mido (one stem per file).

Rendering each instrument to its own MIDI/WAV gives true isolated stems, which
the mixer and the per-band QC gate both rely on.
"""

from __future__ import annotations

import mido

from .contracts import Note

TICKS = 480


def write_instrument_midi(
    path: str,
    notes: list[Note],
    program: int,
    tempo_bpm: int,
    channel: int = 0,
    is_drum: bool = False,
) -> str:
    mid = mido.MidiFile(ticks_per_beat=TICKS)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))

    ch = 9 if is_drum else channel
    if not is_drum:
        track.append(mido.Message("program_change", program=program, channel=ch, time=0))

    sec_per_beat = 60.0 / tempo_bpm
    events: list[tuple[int, int, int, int]] = []  # (tick, on?, midi, vel)
    for n in notes:
        start = int(round(n.start / sec_per_beat * TICKS))
        end = max(int(round((n.start + n.dur) / sec_per_beat * TICKS)), start + 1)
        events.append((start, 1, n.midi, max(1, min(127, n.vel))))
        events.append((end, 0, n.midi, 0))

    # off (0) before on (1) when ticks tie, so repeated pitches don't get cut
    events.sort(key=lambda e: (e[0], e[1]))

    prev = 0
    for tick, on, midi, vel in events:
        delta = tick - prev
        prev = tick
        msg = "note_on" if on else "note_off"
        track.append(mido.Message(msg, note=midi, velocity=vel, channel=ch, time=delta))

    mid.save(path)
    return path
