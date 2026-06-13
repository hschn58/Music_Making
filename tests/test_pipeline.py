import os

import numpy as np

from music_making import from_text, produce
from music_making.audio import SR, load_wav


def test_full_pipeline_produces_passing_track(tmp_path):
    sb = from_text(
        "hopping across lava rock while fire enemies jump in the heat, then a calm night falls",
        seed=2, duration_sec=16.0,
    )
    track = produce(sb, str(tmp_path), max_attempts=2)

    assert os.path.getsize(track.wav_path) > 10_000
    assert os.path.getsize(track.mp3_path) > 1_000
    assert os.path.exists(track.metadata_path)

    samples = load_wav(track.wav_path)
    rms = float(np.sqrt(np.mean(samples ** 2)))
    assert rms > 1e-3  # non-silent
    assert abs(len(samples) / SR - sb.duration_sec) <= 0.2 * sb.duration_sec

    qc = track.qc
    assert len(qc.band_scores) == 3
    assert qc.non_silent
    assert qc.passed
