import numpy as np
import pytest

from music_making.contracts import LAYER_BANDS
from music_making.images import analyze_scenes
from music_making.storyboard import from_images

cv2 = pytest.importorskip("cv2")


@pytest.fixture
def scene_imgs(tmp_path):
    rng = np.random.default_rng(0)
    # cool, structured (forest-like): green vertical stripes -> edges, not warm
    cool = np.zeros((128, 128, 3), np.uint8)
    cool[:, ::6] = (40, 120, 40)  # BGR
    # calm: smooth gray gradient -> few edges, neutral
    grad = np.tile(np.linspace(40, 170, 128).astype(np.uint8), (128, 1))
    calm = np.dstack([grad, grad, grad])
    # warm, busy (fire-like): high red, low blue, noisy -> warm + high contrast
    warm = np.zeros((128, 128, 3), np.uint8)
    warm[..., 2] = rng.integers(120, 255, (128, 128))  # red
    warm[..., 0] = rng.integers(0, 40, (128, 128))     # blue
    paths = []
    for name, img in [("cool", cool), ("calm", calm), ("warm", warm)]:
        p = str(tmp_path / f"{name}.png")
        cv2.imwrite(p, img)
        paths.append(p)
    return paths


def test_analyze_scenes_differentiates_streams(scene_imgs):
    cool, calm, warm = analyze_scenes(scene_imgs)
    # the warm noisy image is the most atmospheric of the set
    assert warm.scores["atmosphere"] == max(s.scores["atmosphere"] for s in (cool, calm, warm))
    # the structured cool image is more terrain than the warm one
    assert cool.scores["terrain"] > warm.scores["terrain"]


def test_from_images_builds_storyboard_and_timbres(scene_imgs):
    sb, timbres = from_images(scene_imgs, seed=0, seconds_per_scene=4.0)
    assert len(sb.story.segments) == 3
    assert len(sb.sections) == len(sb.story.segments)
    assert set(timbres) == set(LAYER_BANDS)
    for f in sb.frames:
        for layer in LAYER_BANDS:
            assert 0.0 <= getattr(f, layer) <= 1.0


def test_order_can_repeat_scenes(scene_imgs):
    # multiple orderings in one song (theme + recapitulation)
    sb, _ = from_images(scene_imgs, order=[0, 1, 2, 2, 1, 0], seed=0, seconds_per_scene=2.0)
    assert len(sb.story.segments) == 6
