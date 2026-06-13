"""Build the root Storyboard from a text situation or from a real video.

The Storyboard is the shared scene contract. It decomposes a scene into three
layers that map to frequency bands (see DESIGN.md):
  terrain (low)  -  structure/ground
  entity (mid)   -  anthropomorphized agents
  atmosphere(hi) -  heat/danger/brightness
"""

from __future__ import annotations

import math
import random
import re
import subprocess

import numpy as np

from .contracts import EntityEvent, SceneFrame, Section, Storyboard
from .genre import get_preset

N_FRAMES = 64

# Keyword lexicons that nudge the scene layers from a text situation.
HEAT = {"fire", "lava", "heat", "burn", "hot", "sun", "flame", "danger", "fight",
        "battle", "explode", "storm", "bright", "blaze", "inferno"}
MOTION = {"run", "jump", "chase", "fly", "dance", "move", "fast", "leap", "hop",
          "race", "rush", "spin", "drive", "swing"}
CALM = {"night", "calm", "slow", "rest", "quiet", "still", "dark", "cool",
        "mellow", "smooth", "float", "drift", "gentle"}
WEIGHT = {"rock", "stone", "ground", "deep", "heavy", "mountain", "castle",
          "earth", "low", "underground", "cave", "weight"}

# Per-section base levels: (terrain, entity, atmosphere, tension)
SECTION_BASE = {
    "intro": (0.45, 0.20, 0.30, 0.20),
    "verse": (0.60, 0.50, 0.40, 0.40),
    "chorus": (0.80, 0.80, 0.70, 0.75),
    "bridge": (0.50, 0.60, 0.80, 0.60),
    "outro": (0.40, 0.20, 0.30, 0.20),
}

SECTION_PATTERN = [
    ("intro", 2), ("verse", 4), ("chorus", 4), ("verse", 4),
    ("chorus", 4), ("bridge", 2), ("chorus", 4), ("outro", 2),
]


def _frac(words: list[str], lexicon: set[str]) -> float:
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in lexicon)
    return min(1.0, hits / max(4, len(words) / 6))


def _plan_sections(total_bars: int) -> list[Section]:
    out: list[Section] = []
    used = 0
    for name, bars in SECTION_PATTERN:
        if used >= total_bars:
            break
        b = min(bars, total_bars - used)
        out.append(Section(name=name, bars=b))
        used += b
    while used < total_bars:
        b = min(4, total_bars - used)
        out.append(Section(name="chorus", bars=b))
        used += b
    return out or [Section(name="verse", bars=max(1, total_bars))]


def _section_at_bar(sections: list[Section], bar: float) -> str:
    acc = 0
    for s in sections:
        if bar < acc + s.bars:
            return s.name
        acc += s.bars
    return sections[-1].name


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def from_text(
    situation: str,
    *,
    genre: str = "smooth-funk",
    seed: int = 0,
    duration_sec: float = 30.0,
    title: str | None = None,
) -> Storyboard:
    preset = get_preset(genre)
    rng = random.Random(seed)
    words = re.findall(r"[a-z']+", situation.lower())

    heat = _frac(words, HEAT)
    motion = _frac(words, MOTION)
    calm = _frac(words, CALM)
    weight = _frac(words, WEIGHT)

    tempo = preset.tempo_bpm + int(round(12 * (motion - calm)))
    tempo = max(70, min(140, tempo))
    beats_per_bar = 4
    bar_sec = beats_per_bar * 60.0 / tempo
    total_bars = max(4, round(duration_sec / bar_sec))
    sections = _plan_sections(total_bars)
    duration_sec = total_bars * bar_sec

    frames: list[SceneFrame] = []
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        bar = t * total_bars
        base = SECTION_BASE[_section_at_bar(sections, bar)]
        wobble = 0.06 * math.sin(2 * math.pi * (t * 3 + rng.random()))
        terrain = _clamp(base[0] + 0.25 * weight + wobble)
        entity = _clamp(base[1] + 0.30 * motion + wobble)
        atmosphere = _clamp(base[2] + 0.35 * heat - 0.20 * calm + wobble)
        tension = _clamp(base[3] + 0.30 * heat + 0.15 * motion - 0.20 * calm)
        brightness = _clamp(0.3 + 0.6 * atmosphere)
        frames.append(SceneFrame(t=t, terrain=terrain, entity_activity=entity,
                                 atmosphere=atmosphere, tension=tension, brightness=brightness))

    events = _entity_events(sections, frames, total_bars, bar_sec, duration_sec, motion, rng)

    return Storyboard(
        title=title or _title_from(words, seed),
        situation=situation,
        genre=preset.name,
        seed=seed,
        tempo_bpm=tempo,
        key=_key_for(preset, calm),
        beats_per_bar=beats_per_bar,
        duration_sec=duration_sec,
        sections=sections,
        frames=frames,
        entity_events=events,
    )


def _entity_events(sections, frames, total_bars, bar_sec, duration_sec, motion, rng) -> list[EntityEvent]:
    """Place the anthropomorphized 'calls' on hook moments + motion spikes."""
    events: list[EntityEvent] = []
    acc = 0
    for s in sections:
        if s.name in ("chorus", "bridge"):
            # a 'call' on the last bar of the section
            t_norm = min(1.0, (acc + s.bars - 0.5) / total_bars)
            ent = _layer_value(frames, "entity_activity", t_norm)
            events.append(EntityEvent(t=t_norm, intensity=_clamp(0.5 + 0.5 * ent), label="agent"))
        acc += s.bars
    n_extra = int(round(3 * motion))
    for _ in range(n_extra):
        t_norm = rng.random()
        events.append(EntityEvent(t=t_norm, intensity=_clamp(0.4 + 0.5 * rng.random()), label="agent"))
    events.sort(key=lambda e: e.t)
    return events


def _layer_value(frames, layer, t_norm) -> float:
    i = min(len(frames) - 1, max(0, int(round(t_norm * (len(frames) - 1)))))
    return getattr(frames[i], layer)


def _key_for(preset, calm: float) -> str:
    # mellower scenes get a flatter, calmer tonic; deterministic, not random
    tonic = "A" if calm < 0.5 else "D"
    return f"{tonic} {preset.mode}"


def _title_from(words: list[str], seed: int) -> str:
    content = [w for w in words if len(w) > 3 and w not in CALM]
    if not content:
        return f"untitled-{seed}"
    pick = content[seed % len(content)]
    return pick.capitalize()


# --------------------------------------------------------------------------- #
# Video ingestion: extract brightness / motion / contrast timelines.
# --------------------------------------------------------------------------- #

def _video_duration(path: str) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, check=True,
        )
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def from_video(
    video_path: str,
    *,
    genre: str = "smooth-funk",
    seed: int = 0,
    max_duration_sec: float = 60.0,
    title: str | None = None,
    samples: int = 120,
) -> Storyboard:
    """Derive a storyboard from a real video (MJ 'make the video first' mode)."""
    import cv2  # imported lazily so text mode needs no opencv

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or samples
    step = max(1, total // samples)

    brightness, motion, contrast = [], [], []
    prev = None
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            small = cv2.resize(gray, (96, 54))
            brightness.append(float(small.mean()))
            contrast.append(float(small.std()))
            motion.append(0.0 if prev is None else float(np.abs(small - prev).mean()))
            prev = small
        idx += 1
    cap.release()

    if len(brightness) < 2:
        raise RuntimeError(f"Video had too few frames to analyze: {video_path}")

    dur = _video_duration(video_path)
    duration_sec = min(max_duration_sec, dur) if dur > 0 else max_duration_sec

    bri = _resample_series(np.array(brightness), N_FRAMES)
    mot = _resample_series(np.array(motion), N_FRAMES)
    con = _resample_series(np.array(contrast), N_FRAMES)
    bri_n, mot_n, con_n = _norm(bri), _norm(mot), _norm(con)

    preset = get_preset(genre)
    beats_per_bar = 4
    bar_sec = beats_per_bar * 60.0 / preset.tempo_bpm
    total_bars = max(4, round(duration_sec / bar_sec))
    sections = _plan_sections(total_bars)
    duration_sec = total_bars * bar_sec

    frames: list[SceneFrame] = []
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        terrain = _clamp(0.35 + 0.55 * con_n[i])      # structure/contrast -> ground
        entity = _clamp(0.20 + 0.70 * mot_n[i])        # motion -> moving agents
        atmosphere = _clamp(0.20 + 0.70 * bri_n[i])    # brightness/heat -> highs
        tension = _clamp(0.5 * mot_n[i] + 0.5 * bri_n[i])
        brightness_v = _clamp(bri_n[i])
        frames.append(SceneFrame(t=t, terrain=terrain, entity_activity=entity,
                                 atmosphere=atmosphere, tension=tension, brightness=brightness_v))

    # entity events at motion spikes (scene cuts / moving objects)
    events: list[EntityEvent] = []
    thresh = float(mot_n.mean() + mot_n.std())
    for i in range(1, N_FRAMES - 1):
        if mot_n[i] >= thresh and mot_n[i] >= mot_n[i - 1] and mot_n[i] >= mot_n[i + 1]:
            events.append(EntityEvent(t=i / (N_FRAMES - 1), intensity=_clamp(mot_n[i]), label="agent"))

    return Storyboard(
        title=title or "video-scene",
        situation=f"video:{video_path}",
        genre=preset.name,
        seed=seed,
        tempo_bpm=preset.tempo_bpm,
        key=f"A {preset.mode}",
        beats_per_bar=beats_per_bar,
        duration_sec=duration_sec,
        sections=sections,
        frames=frames,
        entity_events=events,
    )


def _resample_series(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == n:
        return x
    idx = np.linspace(0, len(x) - 1, n)
    return np.interp(idx, np.arange(len(x)), x)


def _norm(x: np.ndarray) -> np.ndarray:
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-9:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)
