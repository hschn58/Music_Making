"""Analyze still photos into scene-feature scores + texture descriptors.

A photo is an information-rich, *specific* scene. We read per-stream levels and a
``TextureProfile`` straight from the pixels — the spatial-domain analog of a
scalogram:

  2D spatial-frequency content  ->  timbral homogeneity / bandwidth
  warmth + contrast             ->  atmosphere (heat / danger)
  structure + edges             ->  terrain (ground / form)

Structural features (spatial frequency, edges) are normalized *across the set* of
scenes, because a story is relative — "this scene is busier/calmer than that one."
Global statistics are used, so EXIF rotation doesn't matter.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .timbre import TextureProfile

STREAMS = ("terrain", "entity_activity", "atmosphere")


@dataclass
class SceneFeatures:
    name: str
    brightness: float
    warmth: float
    contrast: float
    high_freq: float  # spatial-frequency content, normalized within the set
    edges: float      # edge density, normalized within the set
    scores: dict[str, float]
    intensity: float


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _raw(path: str) -> dict:
    import cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    img = cv2.resize(img, (256, 256))
    b = img[..., 0].astype(np.float32)
    r = img[..., 2].astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # radial spectral centroid: smooth image -> energy near DC -> low; busy -> high
    spec = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean())))
    cy, cx = np.array(spec.shape) // 2
    yy, xx = np.ogrid[: spec.shape[0], : spec.shape[1]]
    rad = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    centroid = float((rad * spec).sum() / (spec.sum() + 1e-9)) / float(rad.max())

    edges = cv2.Canny((gray * 255).astype(np.uint8), 80, 160)
    return {
        "name": os.path.splitext(os.path.basename(path))[0],
        "brightness": _clamp(gray.mean()),
        "warmth": _clamp(0.5 + 1.2 * float((r - b).mean()) / 255.0),
        "contrast": _clamp(gray.std() * 2.0),
        "hf_raw": centroid,
        "edge_raw": float((edges > 0).mean()),
    }


def _minmax(vals: list[float]) -> list[float]:
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-6:
        return [0.5] * len(vals)
    return [(v - lo) / (hi - lo) for v in vals]


def analyze_scenes(paths: list[str]) -> list[SceneFeatures]:
    raws = [_raw(p) for p in paths]
    hf = _minmax([r["hf_raw"] for r in raws])
    eg = _minmax([r["edge_raw"] for r in raws])
    scenes: list[SceneFeatures] = []
    for r, h, e in zip(raws, hf, eg):
        # atmosphere = heat/light (warmth + contrast); terrain = structure (edges +
        # coolness). Busyness (high_freq) is texture, not a stream level.
        atmosphere = _clamp(0.6 * r["warmth"] + 0.4 * r["contrast"])
        terrain = _clamp(0.6 * e + 0.4 * (1 - r["warmth"]))
        entity = _clamp(0.10 + 0.2 * e)  # weak proxy; usually low without agents
        intensity = _clamp(0.4 + 0.4 * r["contrast"] + 0.2 * r["brightness"])
        scenes.append(SceneFeatures(
            name=r["name"], brightness=r["brightness"], warmth=r["warmth"],
            contrast=r["contrast"], high_freq=h, edges=e, intensity=intensity,
            scores={"terrain": terrain, "entity_activity": entity, "atmosphere": atmosphere},
        ))
    return scenes


def analyze_image(path: str) -> SceneFeatures:
    """Single image (set-relative features are degenerate -> 0.5)."""
    return analyze_scenes([path])[0]


# Per-stream cutoff ranges keep streams spectrally separated (mid < high band).
_CUTOFF = {"terrain": (1200, 3000), "entity_activity": (2500, 4000), "atmosphere": (6000, 12000)}
_REVERB = {"terrain": 0.06, "entity_activity": 0.14, "atmosphere": 0.20}


def derive_timbres(scenes: list[SceneFeatures]) -> dict[str, TextureProfile]:
    """Source each stream's timbre from the scene where that stream is strongest."""
    profiles: dict[str, TextureProfile] = {}
    for stream in STREAMS:
        ex = max(scenes, key=lambda s: s.scores[stream])
        lo, hi = _CUTOFF[stream]
        cutoff = lo + (hi - lo) * ex.brightness
        firey = ex.warmth > 0.65 and ex.contrast > 0.4  # genuinely hot + high-contrast
        profiles[stream] = TextureProfile(
            name=f"{stream}<-{ex.name}",
            cutoff_base=cutoff,
            brightness_depth=0.5,
            bandwidth=ex.high_freq,
            drive_base=0.06 + 0.2 * ex.high_freq,
            drive_depth=0.2,
            slow_rate=0.2 if (firey or stream == "terrain") else 0.0,
            slow_depth=0.3 if firey else (0.25 if stream == "terrain" else 0.0),
            fast_rate=11.0 if firey else 0.0,
            fast_depth=0.6 if firey else 0.0,
            chaos=0.85 if firey else 0.0,
            residue=0.3 if stream == "terrain" else 0.0,
            residue_decay=0.5,
            reverb=_REVERB[stream],
        )
    return profiles
