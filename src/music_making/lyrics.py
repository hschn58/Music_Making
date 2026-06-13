"""Lyrics workflow (hybrid).

If an LLM is available (anthropic SDK + API key, and not MUSIC_MAKING_OFFLINE),
use it for the words; otherwise fall back to a deterministic free generator so
the pipeline stays free, offline, and reproducible in CI. Either way the output
is the same typed contract, including per-line syllables for vocal alignment and
the anthropomorphized 'call' hook.
"""

from __future__ import annotations

import os
import random
import re

from .contracts import LyricLine, LyricsResult, Storyboard

FILLER = ["baby", "tonight", "moving", "slow", "feel", "the", "groove", "now",
          "shadow", "light", "we", "keep", "on", "running", "easy", "low"]
HOOK = ["ba", "ba", "ba", "ba", "haa"]


def syllables(word: str) -> list[str]:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return []
    cuts = [m.end() for m in re.finditer(r"[aeiouy]+", word)]
    if len(cuts) <= 1:
        return [word]
    chunks, prev = [], 0
    for cut in cuts[:-1]:
        chunks.append(word[prev:cut])
        prev = cut
    chunks.append(word[prev:])
    return [c for c in chunks if c]


def _content_words(situation: str) -> list[str]:
    words = re.findall(r"[a-z']+", situation.lower())
    seen, out = set(), []
    for w in words:
        if len(w) > 3 and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _local(sb: Storyboard) -> LyricsResult:
    rng = random.Random(sb.seed + 3)
    pool = _content_words(sb.situation) + FILLER
    lines: list[LyricLine] = []
    # One+ line per story segment; entity-forward moments get more words to sing.
    for seg in sb.story.segments:
        n_lines = 2 if seg.dominant == "entity_activity" else 1
        for _ in range(n_lines):
            words = [rng.choice(pool) for _ in range(rng.randint(5, 7))]
            syl: list[str] = []
            for w in words:
                syl.extend(syllables(w))
            lines.append(LyricLine(section=seg.label, text=" ".join(words), syllables=syl))
    return LyricsResult(title=sb.title, lines=lines, hook=HOOK, source="local")


def _llm(sb: Storyboard) -> LyricsResult | None:
    if os.environ.get("MUSIC_MAKING_OFFLINE") == "1":
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic  # optional; not a hard dependency
    except ImportError:
        return None
    try:
        client = anthropic.Anthropic()
        prompt = (
            f"Write short song lyrics for a {sb.genre} track titled '{sb.title}'. "
            f"Scene: {sb.situation}. Give 4-6 short lines, lowercase, one per line, "
            f"no section labels, no commentary."
        )
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        raw = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not raw:
            return None
        labels = [seg.label for seg in sb.story.segments] or ["verse"]
        lines = []
        for i, ln in enumerate(raw):
            syl: list[str] = []
            for w in re.findall(r"[a-z']+", ln.lower()):
                syl.extend(syllables(w))
            if syl:
                lines.append(LyricLine(section=labels[i % len(labels)], text=ln, syllables=syl))
        if not lines:
            return None
        return LyricsResult(title=sb.title, lines=lines, hook=HOOK, source="llm")
    except Exception:
        return None


def generate(sb: Storyboard) -> LyricsResult:
    return _llm(sb) or _local(sb)
