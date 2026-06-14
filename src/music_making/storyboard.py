"""Build the root Storyboard from a text situation or a real video.

A situation is first parsed into a **Story**: an ordered list of segments, each
scored for the three frequency streams (terrain/low, entity/mid, atmosphere/high)
with one dominant. The scene timeline is then built so the dominant stream shifts
across the song (an all-rock stretch -> bass forward; running through fire ->
highs forward; facing enemies -> mid forward). That shifting dominance is what
gives the music story-depth, and it drives arrangement, mix, and timbre.
"""

from __future__ import annotations

import math
import random
import re
import subprocess

import numpy as np

from .contracts import (EntityEvent, SceneFrame, Section, Story, StorySegment,
                        Storyboard)
from .genre import get_preset

N_FRAMES = 64

# Per-stream lexicons. A word can score more than one stream.
TERRAIN_W = {"rock", "stone", "ground", "deep", "heavy", "mountain", "castle",
             "earth", "low", "underground", "cave", "weight", "floor", "path",
             "platform", "brick", "solid", "road", "terrain", "land"}
ENTITY_W = {"run", "jump", "chase", "fly", "dance", "move", "fast", "leap", "hop",
            "race", "rush", "spin", "drive", "swing", "enemy", "enemies", "turtle",
            "turtles", "goomba", "koopa", "people", "crowd", "dancer", "dancers",
            "creature", "monster", "foe", "foes", "them", "they", "runner", "walk"}
ATMOS_W = {"fire", "lava", "heat", "burn", "hot", "sun", "flame", "danger", "fight",
           "battle", "explode", "storm", "bright", "blaze", "inferno", "light",
           "lights", "neon", "glow", "glowing", "spark", "shine", "sky", "air",
           "mist", "smoke", "steam", "electric"}
CALM = {"night", "calm", "slow", "rest", "quiet", "still", "dark", "cool",
        "mellow", "smooth", "float", "drift", "gentle"}

STREAM_LEX = {"terrain": TERRAIN_W, "entity_activity": ENTITY_W, "atmosphere": ATMOS_W}
DOMINANT_LABEL = {"terrain": "ground", "entity_activity": "agents", "atmosphere": "heat"}

CLAUSE_SPLIT = re.compile(r"\s*(?:[,;.!]|\bthen\b|\bwhile\b|\band\b|\bas\b|\bbefore\b|\bafter\b)\s*")


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def _stems(word: str) -> set[str]:
    """Crude lemmatization so 'running'/'hopping' match 'run'/'hop'."""
    cands = {word}
    for suf in ("ing", "ed", "es", "s"):
        if word.endswith(suf) and len(word) > len(suf) + 1:
            base = word[: -len(suf)]
            cands.add(base)
            cands.add(base + "e")
            if len(base) >= 2 and base[-1] == base[-2]:
                cands.add(base[:-1])  # running -> runn -> run
    return cands


def _count(words: list[str], lex: set[str]) -> int:
    return sum(1 for w in words if _stems(w) & lex)


def _in_lex(word: str, lex: set[str]) -> bool:
    return bool(_stems(word) & lex)


# --------------------------------------------------------------------------- #
# Story segmentation
# --------------------------------------------------------------------------- #

def _segment_text(situation: str) -> list[StorySegment]:
    clauses = [c.strip() for c in CLAUSE_SPLIT.split(situation) if c.strip()]
    if not clauses:
        clauses = [situation.strip() or "a scene"]

    segments: list[StorySegment] = []
    for clause in clauses:
        words = _words(clause)
        if not words:
            continue
        counts = {s: _count(words, lex) for s, lex in STREAM_LEX.items()}
        total = sum(counts.values())
        m = max(counts.values()) or 1
        if total == 0:
            scores = {s: 0.5 for s in STREAM_LEX}
        else:
            scores = {s: counts[s] / m for s in STREAM_LEX}
        calm_frac = _count(words, CALM) / len(words)
        intensity = _clamp(0.4 + 0.12 * total - 0.3 * calm_frac)

        dominant = max(scores, key=lambda k: scores[k])
        label = next((w for w in words if _in_lex(w, STREAM_LEX[dominant])), DOMINANT_LABEL[dominant])
        segments.append(StorySegment(label=label, text=clause, scores=scores,
                                     intensity=intensity, weight=float(len(words))))
    return segments or [StorySegment(label="scene", text=situation,
                                     scores={s: 0.5 for s in STREAM_LEX},
                                     intensity=0.5, weight=1.0)]


def _segment_video(brightness, motion, contrast, n_windows: int) -> list[StorySegment]:
    n = len(brightness)
    n_windows = max(1, min(n_windows, n))
    bounds = np.linspace(0, n, n_windows + 1).astype(int)
    segments: list[StorySegment] = []
    for i in range(n_windows):
        a, b = bounds[i], max(bounds[i] + 1, bounds[i + 1])
        terr = float(np.mean(contrast[a:b]))
        ent = float(np.mean(motion[a:b]))
        atm = float(np.mean(brightness[a:b]))
        raw = {"terrain": terr, "entity_activity": ent, "atmosphere": atm}
        m = max(raw.values()) or 1.0
        scores = {s: raw[s] / m for s in raw}
        intensity = _clamp(0.3 + 0.7 * float(np.mean([terr, ent, atm])))
        dominant = max(scores, key=lambda k: scores[k])
        segments.append(StorySegment(label=DOMINANT_LABEL[dominant], text=f"shot {i + 1}",
                                     scores=scores, intensity=intensity, weight=float(b - a)))
    return segments


# --------------------------------------------------------------------------- #
# Story -> sections + frames
# --------------------------------------------------------------------------- #

def _sections_from_story(story: Story, total_bars: int) -> tuple[list[Section], list[tuple[int, int, StorySegment]]]:
    total_w = sum(s.weight for s in story.segments) or 1.0
    sections: list[Section] = []
    spans: list[tuple[int, int, StorySegment]] = []
    used = 0
    for i, seg in enumerate(story.segments):
        remaining = len(story.segments) - i - 1
        bars = max(1, round(seg.weight / total_w * total_bars))
        bars = min(bars, total_bars - used - remaining)
        bars = max(1, bars)
        sections.append(Section(name=seg.label, bars=bars))
        spans.append((used, used + bars, seg))
        used += bars
    if used < total_bars and spans:
        s, e, seg = spans[-1]
        spans[-1] = (s, total_bars, seg)
        sections[-1] = Section(name=sections[-1].name, bars=total_bars - s)
    return sections, spans


def _frames_from_story(spans, total_bars, rng) -> list[SceneFrame]:
    frames: list[SceneFrame] = []
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        bar = t * total_bars
        seg = spans[-1][2]
        p = 0.0
        for a, b, s in spans:
            if a <= bar < b:
                seg = s
                p = (bar - a) / max(1e-9, (b - a))
                break
        build = 0.85 + 0.25 * p  # gentle rise across each segment
        wob = 0.05 * math.sin(2 * math.pi * (t * 4 + rng.random()))
        vals = {}
        for layer in ("terrain", "entity_activity", "atmosphere"):
            base = seg.scores[layer] * (0.55 + 0.45 * seg.intensity)
            vals[layer] = _clamp(0.12 + 0.8 * base * build + wob)
        brightness = _clamp(0.25 + 0.6 * vals["atmosphere"])
        tension = _clamp(0.45 * vals["atmosphere"] + 0.4 * vals["entity_activity"]
                         + 0.2 * seg.intensity)
        frames.append(SceneFrame(t=t, terrain=vals["terrain"],
                                 entity_activity=vals["entity_activity"],
                                 atmosphere=vals["atmosphere"], tension=tension,
                                 brightness=brightness))
    return frames


def _entity_events(spans, total_bars, bar_sec, duration_sec, rng) -> list[EntityEvent]:
    events: list[EntityEvent] = []
    for a, b, seg in spans:
        if seg.dominant == "entity_activity" or seg.scores["entity_activity"] > 0.6:
            # a 'call' near the start of each entity-forward moment
            t_norm = _clamp((a + 0.5) / total_bars)
            events.append(EntityEvent(t=t_norm, intensity=_clamp(0.5 + 0.5 * seg.intensity),
                                      label=seg.label))
            if (b - a) >= 3:
                events.append(EntityEvent(t=_clamp((a + (b - a) * 0.6) / total_bars),
                                          intensity=_clamp(0.4 + 0.4 * seg.intensity),
                                          label=seg.label))
    events.sort(key=lambda e: e.t)
    return events


# --------------------------------------------------------------------------- #
# Public builders
# --------------------------------------------------------------------------- #

def _tempo_key(words, preset, seed):
    motion = _count(words, ENTITY_W) / max(4, len(words) / 6)
    calm = _count(words, CALM) / max(4, len(words) / 6)
    tempo = max(70, min(140, preset.tempo_bpm + int(round(12 * (motion - calm)))))
    tonic = "A" if calm < 0.5 else "D"
    return tempo, f"{tonic} {preset.mode}"


def _title_from(words, seed) -> str:
    content = [w for w in words if len(w) > 3 and w not in CALM]
    return content[seed % len(content)].capitalize() if content else f"untitled-{seed}"


def from_text(situation, *, genre="smooth-funk", seed=0, duration_sec=30.0, title=None) -> Storyboard:
    preset = get_preset(genre)
    rng = random.Random(seed)
    words = _words(situation)

    segments = _segment_text(situation)
    story = Story(title=title or _title_from(words, seed), source=situation, segments=segments)

    tempo, key = _tempo_key(words, preset, seed)
    beats_per_bar = 4
    bar_sec = beats_per_bar * 60.0 / tempo
    total_bars = max(len(segments), round(duration_sec / bar_sec))
    sections, spans = _sections_from_story(story, total_bars)
    duration_sec = sum(s.bars for s in sections) * bar_sec

    frames = _frames_from_story(spans, total_bars, rng)
    events = _entity_events(spans, total_bars, bar_sec, duration_sec, rng)

    return Storyboard(title=story.title, situation=situation, genre=preset.name, seed=seed,
                      tempo_bpm=tempo, key=key, beats_per_bar=beats_per_bar,
                      duration_sec=duration_sec, story=story, sections=sections,
                      frames=frames, entity_events=events)


def from_images(paths, *, order=None, genre="smooth-funk", seed=0,
                seconds_per_scene=10.0, title=None):
    """Build a story from a sequence of photos (a literal storyboard).

    Each image becomes a Story segment whose stream levels are read from its
    pixels; `order` (names/substrings or indices) chooses the narrative sequence,
    and may repeat scenes for theme-and-recapitulation forms. Returns
    ``(storyboard, timbres)`` where timbres are sourced from the exemplar scenes.
    """
    from . import images as imgmod

    scenes_all = imgmod.analyze_scenes(list(paths))
    if order is None:
        seq = scenes_all
    else:
        seq = []
        for o in order:
            if isinstance(o, int):
                seq.append(scenes_all[o])
            else:
                match = next((s for s in scenes_all if o.lower() in s.name.lower()), None)
                seq.append(match or scenes_all[0])

    preset = get_preset(genre)
    rng = random.Random(seed)
    beats_per_bar = 4
    bar_sec = beats_per_bar * 60.0 / preset.tempo_bpm
    bars_per_scene = max(1, round(seconds_per_scene / bar_sec))

    segments = [StorySegment(label=sc.name, text=sc.name, scores=sc.scores,
                             intensity=sc.intensity, weight=float(bars_per_scene))
                for sc in seq]
    story = Story(title=title or "image-story",
                  source="images:" + ",".join(s.name for s in seq), segments=segments)
    total_bars = bars_per_scene * len(seq)
    sections, spans = _sections_from_story(story, total_bars)
    duration_sec = sum(s.bars for s in sections) * bar_sec
    frames = _frames_from_story(spans, total_bars, rng)
    events = _entity_events(spans, total_bars, bar_sec, duration_sec, rng)

    sb = Storyboard(title=story.title, situation=story.source, genre=preset.name, seed=seed,
                    tempo_bpm=preset.tempo_bpm, key=f"A {preset.mode}",
                    beats_per_bar=beats_per_bar, duration_sec=duration_sec, story=story,
                    sections=sections, frames=frames, entity_events=events)
    timbres = imgmod.derive_timbres(seq)
    return sb, timbres


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


def _norm(x: np.ndarray) -> np.ndarray:
    mn, mx = float(x.min()), float(x.max())
    return (x - mn) / (mx - mn) if mx - mn > 1e-9 else np.zeros_like(x)


def _resample_series(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == n:
        return x
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)


def from_video(video_path, *, genre="smooth-funk", seed=0, max_duration_sec=60.0,
               title=None, samples=120, n_windows=6) -> Storyboard:
    """Derive a storyboard from a real video (MJ 'make the video first' mode)."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or samples
    step = max(1, total // samples)

    brightness, motion, contrast = [], [], []
    prev = None
    idx = 0
    while cap.grab():
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

    bri, mot, con = (_norm(np.array(brightness)), _norm(np.array(motion)),
                     _norm(np.array(contrast)))

    dur = _video_duration(video_path)
    duration_sec = min(max_duration_sec, dur) if dur > 0 else max_duration_sec
    preset = get_preset(genre)
    rng = random.Random(seed)
    beats_per_bar = 4
    bar_sec = beats_per_bar * 60.0 / preset.tempo_bpm
    total_bars = max(n_windows, round(duration_sec / bar_sec))

    segments = _segment_video(bri, mot, con, n_windows)
    story = Story(title=title or "video-scene", source=f"video:{video_path}", segments=segments)
    sections, spans = _sections_from_story(story, total_bars)
    duration_sec = sum(s.bars for s in sections) * bar_sec

    # Detailed per-frame curves from the actual video features.
    bri_f, mot_f, con_f = (_resample_series(bri, N_FRAMES), _resample_series(mot, N_FRAMES),
                           _resample_series(con, N_FRAMES))
    frames = []
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        terrain = _clamp(0.15 + 0.8 * con_f[i])
        entity = _clamp(0.15 + 0.8 * mot_f[i])
        atmosphere = _clamp(0.15 + 0.8 * bri_f[i])
        frames.append(SceneFrame(t=t, terrain=terrain, entity_activity=entity,
                                 atmosphere=atmosphere,
                                 tension=_clamp(0.5 * mot_f[i] + 0.5 * bri_f[i]),
                                 brightness=_clamp(bri_f[i])))

    events = []
    thresh = float(mot_f.mean() + mot_f.std())
    for i in range(1, N_FRAMES - 1):
        if mot_f[i] >= thresh and mot_f[i] >= mot_f[i - 1] and mot_f[i] >= mot_f[i + 1]:
            events.append(EntityEvent(t=i / (N_FRAMES - 1), intensity=_clamp(mot_f[i]),
                                      label="agent"))

    return Storyboard(title=story.title, situation=story.source, genre=preset.name, seed=seed,
                      tempo_bpm=preset.tempo_bpm, key=f"A {preset.mode}",
                      beats_per_bar=beats_per_bar, duration_sec=duration_sec, story=story,
                      sections=sections, frames=frames, entity_events=events)
