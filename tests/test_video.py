import shutil
import subprocess

import pytest

from music_making import from_video
from music_making.contracts import LAYER_BANDS

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg required to synthesize a test video"
)


@pytest.fixture
def sample_video(tmp_path):
    path = tmp_path / "scene.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=3:size=160x90:rate=15",
         "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return str(path)


def test_from_video_builds_storyboard(sample_video):
    sb = from_video(sample_video, seed=0, samples=40)
    assert len(sb.frames) == 64
    assert sb.duration_sec > 0
    for f in sb.frames:
        for layer in LAYER_BANDS:
            assert 0.0 <= getattr(f, layer) <= 1.0
