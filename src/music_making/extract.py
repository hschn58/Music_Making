"""Extract a Tabulation from a single still image — measurement, not authorship.

The one perceptual step is **feature identification** (which regions exist); it is
supplied as input (currently authored by the operator/AI looking at the image).
Everything else — pitch, roughness, grain density, loudness — is *measured* from
the pixels inside each region by the transfer law below. The transfer law is the
"instrument": uniform and content-independent, the same for every image. No
per-feature meaning is authored (no "this is fire so make it warm").

Single-image limitation (honest): temporal texture — flicker/drift *rates* — cannot
be measured from one frame; that needs video. So ``modulation`` is left empty here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .tabulation import FUSION_RATE, Component, Feature, Medium, PerspectiveKey, Tabulation

# --- the transfer law (the instrument: uniform, content-independent) -------- #
F_REF = 55.0              # Hz: pitch of a structure spanning the whole image height
F_MIN, F_MAX = 40.0, 8000.0
N_BANDS = 3               # spatial scale bands per region (coarse -> fine)
GRAIN_DWELL = 3.0         # s: converts element count -> grains/sec
GRAIN_MIN_ELEMENTS = 6.0  # above this an element-dense band sounds granular
GAIN_DROP = 0.12          # drop components quieter than this (normalized)
SCAN_WINDOW = 0.4         # fraction of image height the gaze sees at once (#6, placeholder)
DEFAULT_DURATION = 12.0
IMG_H = 256               # working height (aspect preserved)


@dataclass
class FeatureRegion:
    """A feature's identity + location. The perceptual input; everything else is
    measured. ``box`` is (x0, y0, x1, y1) normalized to [0, 1]."""
    name: str
    box: tuple[float, float, float, float]


def _load_gray(path: str) -> np.ndarray:
    import cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    img = cv2.resize(img, (max(1, int(w / h * IMG_H)), IMG_H))
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def _radial_power(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2D spatial power spectrum + radial frequency map (cycles/pixel)."""
    hp, wp = patch.shape
    win = np.outer(np.hanning(hp), np.hanning(wp)).astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2((patch - patch.mean()) * win))
    P = (np.abs(f) ** 2).astype(np.float64)
    fy = np.fft.fftshift(np.fft.fftfreq(hp))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(wp))[None, :]
    return np.sqrt(fy ** 2 + fx ** 2), P


def _measure(gray: np.ndarray, region: FeatureRegion):
    """Measure one region -> dict with raw per-band measurements (loudness as raw
    amplitude, normalized GLOBALLY by the caller), medium, and scan geometry."""
    h, w = gray.shape
    x0, y0, x1, y1 = region.box
    px0, py0 = int(max(0, x0) * w), int(max(0, y0) * h)
    px1, py1 = int(min(1, x1) * w), int(min(1, y1) * h)
    patch = gray[py0:py1, px0:px1]
    if patch.size < 16 or min(patch.shape) < 4:
        return None

    r, P = _radial_power(patch)
    # Bands relative to THIS patch's resolvable range [~1/N, Nyquist], not the whole
    # image: a patch of N px cannot hold a wave longer than N px, so image-relative
    # edges leave the coarse bands empty for small regions (the band-edge collapse).
    # Pitch stays absolute below, so a region's size sets its register (option A).
    region_px = max(patch.shape)
    edges = np.geomspace(0.9 / region_px, 0.5, N_BANDS + 1)
    bands = []
    for b in range(N_BANDS):
        mask = (r > edges[b]) & (r <= edges[b + 1])
        Pe, rr = P[mask], r[mask]
        if Pe.size < 4 or Pe.sum() < 1e-9:
            continue
        r_mean = float((rr * Pe).sum() / Pe.sum())                 # size  -> pitch
        freq = float(np.clip(F_REF * IMG_H * r_mean, F_MIN, F_MAX))
        flat = float(np.exp(np.mean(np.log(Pe + 1e-12))) / (Pe.mean() + 1e-12))  # surface
        amp = float(np.sqrt(Pe.sum()))    # loudness: amplitude = sqrt(AC power) (DC dropped)
        elements = r_mean * region_px                              # occurrence -> density
        granular = b == N_BANDS - 1 and elements > GRAIN_MIN_ELEMENTS
        bands.append({
            "name": f"s{b}", "freq": round(freq, 2),
            "roughness": round(float(np.clip(flat, 0, 1)), 3),
            "mode": "granular" if granular else "sustained",
            "grain_rate": round(float(np.clip(elements / GRAIN_DWELL, 0, FUSION_RATE)), 2)
            if granular else 0.0,
            "amp": amp})
    if not bands:
        return None

    bright = float(patch.mean())
    medium = Medium(amount=round(float(np.clip((1 - bright) * 0.4, 0, 0.4)), 3),
                    persistence_s=0.4, darkness=0.85)
    return {"name": region.name, "bands": bands, "medium": medium,
            "yc": (y0 + y1) / 2, "extent": y1 - y0}


def extract_tabulation(image_path: str, regions: list[FeatureRegion],
                       duration: float = DEFAULT_DURATION, title: str | None = None) -> Tabulation:
    """Measure each identified region into a Tabulation (literal pitch, no authored meaning).

    Loudness and on-screen timing are kept separate: a component's ``gain`` is its
    measured energy share (amplitude normalized across ALL features), and the
    perspective track is a pure 0->1 visibility gate as the gaze crosses each feature.
    """
    gray = _load_gray(image_path)
    measured = [m for reg in regions if (m := _measure(gray, reg)) is not None]
    if not measured:
        return Tabulation(title=title or os.path.splitext(os.path.basename(image_path))[0],
                          duration=float(duration), features=[], perspective=[])

    global_amp = max(b["amp"] for m in measured for b in m["bands"]) or 1.0
    feats, meta = [], []
    for m in measured:
        loudest = max(b["amp"] for b in m["bands"])
        comps = []
        for b in m["bands"]:
            gain = round(b["amp"] / global_amp, 3)
            # keep a band above the floor, or the feature's own loudest (so an
            # identified feature is never fully silent)
            if gain < GAIN_DROP and b["amp"] < loudest:
                continue
            comps.append(Component(name=b["name"], freq=b["freq"], roughness=b["roughness"],
                                   mode=b["mode"], grain_rate=b["grain_rate"], gain=gain))
        if not comps:
            continue
        feats.append(Feature(name=m["name"], components=comps, medium=m["medium"], dynamic=False))
        meta.append((m["name"], m["yc"], m["extent"]))

    # Perspective: pure visibility gate (peak 1.0). Loudness lives in the gains above;
    # this only says WHEN each feature is in view as the gaze scans top->bottom (#6).
    persp = []
    half = SCAN_WINDOW / 2
    for name, yc, extent in meta:
        peak = round(yc * duration, 2)
        t0 = round(max(0.0, (yc - half - extent / 2) * duration), 2)
        t1 = round(min(duration, (yc + half + extent / 2) * duration), 2)
        persp.append(PerspectiveKey(t0=t0, t1=peak, target=f"{name}.*", g1=1.0, g0=0.0))
        persp.append(PerspectiveKey(t0=peak, t1=t1, target=f"{name}.*", g1=0.0, g0=1.0))

    return Tabulation(title=title or os.path.splitext(os.path.basename(image_path))[0],
                      duration=float(duration), features=feats, perspective=persp)
