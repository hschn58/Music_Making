"""The project's core claim, tested deterministically: a track whose per-band
energy follows the scene scores high; the QC gate measures exactly that.
"""

import numpy as np

from music_making import audio, qc
from music_making.contracts import SceneFrame, Section, Storyboard

N = 64


def _ramp_storyboard(duration_sec, terrain, atmosphere, entity=0.5):
    frames = []
    for i in range(N):
        t = i / (N - 1)
        frames.append(SceneFrame(
            t=t, terrain=terrain(t), entity_activity=entity,
            atmosphere=atmosphere(t), tension=0.5, brightness=atmosphere(t),
        ))
    return Storyboard(
        title="synthetic", situation="test", genre="smooth-funk", seed=0,
        tempo_bpm=100, key="A dorian", duration_sec=duration_sec,
        sections=[Section(name="verse", bars=4)], frames=frames, entity_events=[],
    )


def _am_tone(freq, env, n, sr):
    t = np.arange(n) / sr
    amp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(env)), env)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


def test_energy_envelope_tracks_scene(tmp_path):
    dur = 4.0
    n = int(dur * audio.SR)
    # low energy falls over time, high energy rises over time
    sig = 0.5 * _am_tone(80, np.linspace(1, 0, 64), n, audio.SR)
    sig += 0.5 * _am_tone(3500, np.linspace(0, 1, 64), n, audio.SR)
    wav = str(tmp_path / "synthetic.wav")
    audio.save_wav(wav, sig)

    sb = _ramp_storyboard(dur, terrain=lambda t: 1 - t, atmosphere=lambda t: t)
    report = qc.evaluate(wav, sb)

    assert report.non_silent
    assert len(report.band_scores) == 3
    by_band = {s.band: s.correlation for s in report.band_scores}
    # low band falls with terrain (which also falls) -> positive correlation
    assert by_band["low"] > 0.5
    # high band rises with atmosphere (which also rises) -> positive correlation
    assert by_band["high"] > 0.5


def test_silent_track_fails_qc(tmp_path):
    wav = str(tmp_path / "silence.wav")
    audio.save_wav(wav, np.zeros(audio.SR, dtype=np.float32))
    sb = _ramp_storyboard(1.0, terrain=lambda t: t, atmosphere=lambda t: t)
    report = qc.evaluate(wav, sb)
    assert not report.non_silent
    assert not report.passed
