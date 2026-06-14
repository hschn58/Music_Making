"""Orchestrate the parallel workflows into one track.

DAG (the 'parallel agent workflows' idea):

    lyrics  ─┐
    compose ─┼─► vocals ─► mix ─► QC
    beats   ─┘

lyrics / compose / beats are independent and run concurrently; vocals depends on
lyrics+compose; mix depends on every stem; QC is the autonomous gate. On failure
the gate re-runs with a fresh seed up to ``max_attempts``.
"""

from __future__ import annotations

import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import beats as beats_mod
from . import composition, lyrics as lyrics_mod, mix as mix_mod, qc, vocals
from .contracts import Storyboard, Track


def _run_once(sb: Storyboard, workdir: str, soundfont: str | None,
              timbres: dict | None = None) -> Track:
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_lyrics = ex.submit(lyrics_mod.generate, sb)
        f_comp = ex.submit(composition.compose, sb, workdir, soundfont, timbres)
        f_beats = ex.submit(beats_mod.make_beats, sb, workdir, soundfont, timbres)
        lyr = f_lyrics.result()
        comp = f_comp.result()
        beat = f_beats.result()

    voc = vocals.sing(sb, lyr, comp, workdir, soundfont)

    stems = {
        "bass": comp.bass_stem,
        "harmony": comp.harmony_stem,
        "pad": comp.pad_stem,
        "lead": comp.lead_stem,
        "kick": beat.kick_stem,
        "snare": beat.snare_stem,
        "hats": beat.hats_stem,
        "vocals": voc.vocal_stem,
    }
    wav_path, mp3_path = mix_mod.mix(stems, workdir, sb)
    report = qc.evaluate(wav_path, sb)

    meta_path = str(Path(workdir) / "metadata.json")
    metadata = {
        "title": sb.title,
        "genre": sb.genre,
        "tempo_bpm": sb.tempo_bpm,
        "key": sb.key,
        "duration_sec": sb.duration_sec,
        "lyrics_source": lyr.source,
        "aligned_syllables": voc.aligned_syllables,
        "stems": {k: v.path for k, v in stems.items()},
        "qc": report.model_dump(),
    }
    Path(meta_path).write_text(json.dumps(metadata, indent=2))

    return Track(wav_path=wav_path, mp3_path=mp3_path, metadata_path=meta_path,
                 storyboard=sb, qc=report)


def produce(sb: Storyboard, out_dir: str, *, soundfont: str | None = None,
            max_attempts: int = 1, timbres: dict | None = None) -> Track:
    """Run the pipeline, retrying with a fresh seed until QC passes."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    track: Track | None = None
    current = sb
    for attempt in range(max_attempts):
        workdir = tempfile.mkdtemp(prefix=f"mm_{attempt}_", dir=str(out))
        track = _run_once(current, workdir, soundfont, timbres)
        if track.qc.passed:
            break
        current = current.model_copy(update={"seed": current.seed + 1})

    assert track is not None
    # Promote the chosen track's files to stable names in out_dir.
    final_wav = out / "track.wav"
    final_mp3 = out / "track.mp3"
    final_meta = out / "metadata.json"
    for src, dst in [(track.wav_path, final_wav), (track.mp3_path, final_mp3),
                     (track.metadata_path, final_meta)]:
        Path(dst).write_bytes(Path(src).read_bytes())
    return track.model_copy(update={
        "wav_path": str(final_wav),
        "mp3_path": str(final_mp3),
        "metadata_path": str(final_meta),
    })
