"""Read a Blender *capture* (see ``blender/render_scene.py``) into the pipeline.

A capture is a directory of ``frame_####.png`` (sRGB), ``depth_####.png`` (16-bit
Z-depth, normalized over the camera clip range) and ``camera.json`` (intrinsics +
per-frame camera-to-world). This module turns that into per-frame arrays plus the
two geometry quantities the synthesis needs — both *measured*, never assumed:

  * **eccentricity** — angle of a pixel from the optical axis (screen center).
    Drives the foveal window: central -> wide band, peripheral -> narrowed.
  * **range r** — Euclidean eye-to-pixel distance. The Z-depth pass is distance
    *along* the camera axis; true range is ``Z / cos(eccentricity)``. Drives the
    1/sqrt(r) loudness falloff.

Pure geometry (``pixel_eccentricity``, ``z_to_range``) needs no image libraries,
so it is unit-testable without Blender.
"""

from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass

import numpy as np


def pixel_eccentricity(px: float, py: float, width: int, height: int, fov_x: float) -> float:
    """Angle (radians) of pixel (px, py) from the optical axis, via the pinhole
    model. We only build first-person sims where the center of vision is the
    center of every frame, so the principal point = image center IS the fovea."""
    fx = (width / 2.0) / math.tan(fov_x / 2.0)        # focal length in pixels
    dx = (px - width / 2.0) / fx
    dy = (py - height / 2.0) / fx
    return math.atan(math.hypot(dx, dy))


def z_to_range(z: np.ndarray, width: int, height: int, fov_x: float) -> np.ndarray:
    """Convert the camera-axis Z-depth pass to Euclidean eye-to-pixel range r."""
    fx = (width / 2.0) / math.tan(fov_x / 2.0)
    ys, xs = np.mgrid[0:height, 0:width]
    dx = (xs - width / 2.0) / fx
    dy = (ys - height / 2.0) / fx
    return z * np.sqrt(1.0 + dx * dx + dy * dy)


@dataclass
class Frame:
    index: int
    gray: np.ndarray            # (H, W) float in [0, 1]
    depth: np.ndarray           # (H, W) camera-axis Z-depth (metres); +inf where no hit
    cam_to_world: np.ndarray    # (4, 4)
    forward: np.ndarray         # (3,) world-space gaze direction


@dataclass
class Capture:
    width: int
    height: int
    fov_x: float
    fps: float
    frames: list[Frame]

    def range_map(self, frame: Frame) -> np.ndarray:
        """Per-pixel Euclidean range r for one frame (for the 1/sqrt(r) law)."""
        return z_to_range(frame.depth, self.width, self.height, self.fov_x)


def _read_gray(path: str) -> np.ndarray:
    import cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"could not read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def _read_depth(path: str, clip_start: float, clip_end: float) -> np.ndarray:
    """16-bit PNG normalized over [clip_start, clip_end] -> metric Z. Pixels at
    the far clip are background (no geometry) and become +inf."""
    import cv2

    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise RuntimeError(f"could not read depth {path}")
    if d.ndim == 3:
        d = d[..., 0]
    full = np.iinfo(d.dtype).max if np.issubdtype(d.dtype, np.integer) else 1.0
    norm = d.astype(np.float32) / full
    z = clip_start + norm * (clip_end - clip_start)
    z[norm >= 1.0 - 1e-6] = np.inf
    return z


def load_capture(capture_dir: str) -> Capture:
    with open(os.path.join(capture_dir, "camera.json")) as fh:
        meta = json.load(fh)
    w, h = meta["resolution"]
    clip_start, clip_end = float(meta["clip_start"]), float(meta["clip_end"])
    rgb_paths = sorted(glob.glob(os.path.join(capture_dir, "frame_*.png")))
    depth_paths = sorted(glob.glob(os.path.join(capture_dir, "depth_*.png")))
    frames = []
    for fmeta, rgb_p, d_p in zip(meta["frames"], rgb_paths, depth_paths):
        frames.append(Frame(
            index=fmeta["index"],
            gray=_read_gray(rgb_p),
            depth=_read_depth(d_p, clip_start, clip_end),
            cam_to_world=np.array(fmeta["matrix_world"], dtype=np.float64),
            forward=np.array(fmeta["forward"], dtype=np.float64),
        ))
    return Capture(width=w, height=h, fov_x=float(meta["fov_x"]),
                   fps=float(meta["fps"]), frames=frames)
