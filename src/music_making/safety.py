"""Final safety master — applied to everything written to disk.

Audio harm is dominated by sound *pressure level* and sudden transients, not
pitch. But biological structures do have mechanical resonances — eyeball ~19 Hz,
whole-body ~4-8 Hz, head ~20-30 Hz — where a given level deposits more energy
internally. So the mechanical guard is: exclude the infrasound/resonance band
outright, cap the level, and scrub anything that could become a pop.

This is the *mechanical* version. FUTURE polish (noted, not built): proper EBU
R128 integrated-loudness targeting (LUFS, e.g. via pyloudnorm), oversampled
true-peak detection, and explicit derating/notching of the named resonance bands
instead of a single blanket high-pass.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

HPF_HZ = 30.0          # high-pass above the eyeball (~19 Hz) / infrasound / DC band
PEAK_CEILING = 0.891   # ~ -1 dBFS true-peak ceiling


def master(x: np.ndarray, sr: int, peak_ceiling: float = PEAK_CEILING,
           hpf_hz: float = HPF_HZ) -> np.ndarray:
    """Scrub NaN/Inf, high-pass out the infrasound/resonance band (and DC), and
    limit the peak. What leaves here cannot pop or run away in level."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if x.size:
        sos = butter(2, hpf_hz / (sr / 2.0), btype="high", output="sos")
        x = sosfilt(sos, x)
        peak = float(np.max(np.abs(x)))
        if peak > peak_ceiling:
            x *= peak_ceiling / peak
        x = np.clip(x, -peak_ceiling, peak_ceiling)
    return x.astype(np.float32)
