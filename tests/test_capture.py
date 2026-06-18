"""Capture loader + geometry. The pure-geometry tests need nothing; the loader
test writes a tiny synthetic capture (no Blender) and reads it back."""
import json
import math

import numpy as np
import pytest

from music_making.capture import pixel_eccentricity, z_to_range

cv2 = pytest.importorskip("cv2")

from music_making.capture import load_capture  # noqa: E402


def test_eccentricity_zero_at_center_and_grows_outward():
    fov = math.radians(90)
    assert pixel_eccentricity(64, 64, 128, 128, fov) == pytest.approx(0.0, abs=1e-9)
    assert pixel_eccentricity(0, 0, 128, 128, fov) > pixel_eccentricity(32, 32, 128, 128, fov) > 0


def test_z_to_range_equals_z_on_axis_and_exceeds_it_off_axis():
    z = np.full((8, 8), 5.0, np.float32)
    r = z_to_range(z, 8, 8, math.radians(90))
    assert r[4, 4] == pytest.approx(5.0, abs=1e-6)   # on the optical axis r == Z
    assert r[0, 0] > 5.0                              # off-axis Euclidean range is longer


def _write_capture(d, clip_start=0.1, clip_end=10.1):
    w = h = 8
    gray = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    cv2.imwrite(str(d / "frame_0000.png"), cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    depth = np.full((h, w), 65535, np.uint16)        # background -> +inf
    depth[:, :4] = 32768                             # geometry at Z ~ midpoint
    cv2.imwrite(str(d / "depth_0000.png"), depth)
    meta = {"resolution": [w, h], "fov_x": math.radians(90), "fov_y": math.radians(90),
            "clip_start": clip_start, "clip_end": clip_end, "fps": 50,
            "frames": [{"index": 0, "matrix_world": np.eye(4).tolist(),
                        "location": [0, 0, 0], "forward": [0, 0, -1]}]}
    (d / "camera.json").write_text(json.dumps(meta))


def test_loader_recovers_metric_depth_and_marks_background(tmp_path):
    _write_capture(tmp_path)
    cap = load_capture(str(tmp_path))
    assert (cap.width, cap.height) == (8, 8) and len(cap.frames) == 1
    f = cap.frames[0]
    assert not np.isfinite(f.depth[:, 4:]).any()                 # far clip -> inf
    assert f.depth[:, :4] == pytest.approx(0.1 + 0.5 * 10.0, abs=0.01)  # 32768/65535 ~ 0.5
    r = cap.range_map(f)
    fin = np.isfinite(f.depth)
    assert np.all(r[fin] >= f.depth[fin] - 1e-4)
