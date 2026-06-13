import os

import pytest


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Keep the whole suite free/offline/deterministic (no LLM calls)."""
    monkeypatch.setenv("MUSIC_MAKING_OFFLINE", "1")
