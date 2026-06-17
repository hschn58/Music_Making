import numpy as np
import pytest

from music_making.tabulation import dumps, loads, render_tabulation

cv2 = pytest.importorskip("cv2")

from music_making.extract import FeatureRegion, extract_tabulation  # noqa: E402


@pytest.fixture
def split_image(tmp_path):
    """Top half: coarse low-frequency stripes (tonal). Bottom half: white noise
    (rough, fine)."""
    x = np.arange(256)
    stripe = (0.5 + 0.5 * np.sin(2 * np.pi * x / 64)) * 255.0   # wavelength 64px
    top = np.tile(stripe, (128, 1)).astype(np.uint8)
    rng = np.random.default_rng(0)
    bottom = rng.integers(0, 256, (128, 256)).astype(np.uint8)
    img = np.zeros((256, 256, 3), np.uint8)
    img[:128] = np.dstack([top] * 3)
    img[128:] = np.dstack([bottom] * 3)
    p = str(tmp_path / "split.png")
    cv2.imwrite(p, img)
    return p


def _max(feat, attr):
    return max(getattr(c, attr) for c in feat.components)


def test_measures_roughness_and_pitch_from_pixels(split_image):
    regions = [FeatureRegion("top", (0, 0, 1, 0.5)), FeatureRegion("bottom", (0, 0.5, 1, 1))]
    tab = extract_tabulation(split_image, regions, duration=8.0)
    feats = {f.name: f for f in tab.features}
    assert set(feats) == {"top", "bottom"}
    # noise reads rougher than the tonal stripes
    assert _max(feats["bottom"], "roughness") > _max(feats["top"], "roughness")
    # noise carries higher spatial frequency -> higher pitch than the coarse stripes
    assert _max(feats["bottom"], "freq") > _max(feats["top"], "freq")


def test_loudness_is_global_energy_share_and_perspective_is_a_gate(tmp_path):
    """A high-contrast region is louder than a near-flat one (AC energy share),
    gains are normalized globally (loudest component == 1.0), and the perspective
    is a pure 0..1 visibility gate (peaks at 1.0, not area-prominence)."""
    rng = np.random.default_rng(1)
    img = np.zeros((256, 256, 3), np.uint8)
    img[:128] = 128 + rng.integers(-4, 4, (128, 256, 1))   # near-flat: low AC energy
    img[128:] = rng.integers(0, 256, (128, 256, 1))         # busy: high AC energy
    p = str(tmp_path / "energy.png")
    cv2.imwrite(p, img)
    regions = [FeatureRegion("flat", (0, 0, 1, 0.5)), FeatureRegion("busy", (0, 0.5, 1, 1))]
    tab = extract_tabulation(p, regions, duration=6.0)
    feats = {f.name: f for f in tab.features}
    assert _max(feats["busy"], "gain") > _max(feats["flat"], "gain")
    assert max(c.gain for f in tab.features for c in f.components) == 1.0
    assert max(k.g1 for k in tab.perspective) == 1.0


def test_extracted_tabulation_renders_and_round_trips(split_image):
    regions = [FeatureRegion("top", (0, 0, 1, 0.5)), FeatureRegion("bottom", (0, 0.5, 1, 1))]
    tab = extract_tabulation(split_image, regions, duration=6.0)
    assert loads(dumps(tab)) == tab            # measured numbers survive the YAML
    x = render_tabulation(tab, seed=0)
    assert np.max(np.abs(x)) > 0.05
