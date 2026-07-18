import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def _reset_focused_window_ttl_cache(monkeypatch):
    """The hyprctl-fallback TTL cache in ``emitter.focused_window`` is a
    production optimization; between tests it must not leak results."""
    import wldictate.emitter as em

    monkeypatch.setattr(em, "_fallback_at", 0.0)
    monkeypatch.setattr(em, "_fallback_result", ("", ""))
