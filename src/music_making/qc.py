"""Autonomous quality gate.

The project's success metric (see DESIGN.md): the rendered track is "good" when
its per-band energy envelope tracks the scene. We measure the low/mid/high
envelopes of the final mix and correlate each against its scene layer
(low<-terrain, mid<-entity_activity, high<-atmosphere), alongside basic technical
checks. No human in the loop.
"""

from __future__ import annotations

import numpy as np

from . import audio
from .contracts import LAYER_BANDS, QCBandScore, QCReport, Storyboard

N_POINTS = 64
MIN_CORRELATION = 0.15  # modest: synthesis envelopes are noisy but should track
DURATION_TOLERANCE = 0.20  # fraction
SILENCE_RMS = 1e-3


def _target_envelope(sb: Storyboard, layer: str, n: int) -> np.ndarray:
    return np.array([sb.layer_at(layer, i / (n - 1)) for i in range(n)])


def evaluate(wav_path: str, sb: Storyboard) -> QCReport:
    samples = audio.load_wav(wav_path)
    duration = len(samples) / audio.SR
    rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0
    non_silent = rms > SILENCE_RMS

    band_env = audio.band_envelopes(samples, n_points=N_POINTS)
    scores: list[QCBandScore] = []
    for layer, band in LAYER_BANDS.items():
        target = _target_envelope(sb, layer, N_POINTS)
        corr = audio.correlation(band_env[band], target)
        scores.append(QCBandScore(band=band, layer=layer, correlation=corr))
    mean_corr = float(np.mean([s.correlation for s in scores])) if scores else 0.0

    lufs = audio.measure_lufs(wav_path)
    tp = audio.true_peak_db(samples)

    notes: list[str] = []
    dur_ok = abs(duration - sb.duration_sec) <= DURATION_TOLERANCE * sb.duration_sec
    if not non_silent:
        notes.append("track is silent")
    if not dur_ok:
        notes.append(f"duration {duration:.1f}s off target {sb.duration_sec:.1f}s")
    if tp > 0.0:
        notes.append(f"true peak clipping ({tp:.1f} dB)")
    if mean_corr < MIN_CORRELATION:
        notes.append(f"weak scene tracking (mean corr {mean_corr:.2f})")

    passed = non_silent and dur_ok and tp <= 0.0 and mean_corr >= MIN_CORRELATION
    return QCReport(
        passed=passed,
        duration_sec=duration,
        integrated_lufs=lufs,
        true_peak_db=tp,
        non_silent=non_silent,
        mean_correlation=mean_corr,
        band_scores=scores,
        notes=notes,
    )
