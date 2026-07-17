"""Contextual transform: output cleaning, context capture, key resolution,
prompt assembly — no network, no audio."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from wldictate.config import ContextualConfig, ContextualProfile
from wldictate.transform import (
    ScreenContext,
    Transformer,
    TransformError,
    TransformUnavailable,
    _clean_output,
    build_user_message,
    capture_context,
    make_backend,
    resolve_api_key,
)
import wldictate.transform as transform_mod


# ── _clean_output ────────────────────────────────────────────────────────────


def test_clean_strips_think_blocks():
    assert _clean_output("<think>hmm\nokay</think>Hello there.") == "Hello there."
    assert _clean_output("a<think>x</think>b<think>y</think>c") == "abc"


def test_clean_strips_orphan_leading_think_close():
    # Reasoning models sometimes emit only the closing tag mid-stream.
    assert _clean_output("some reasoning...</think>\nFinal text.") == "Final text."


def test_clean_strips_code_fence():
    assert _clean_output("```\nhello world\n```") == "hello world"
    assert _clean_output("```text\nhello\n```") == "hello"


def test_clean_strips_wrapping_quotes():
    assert _clean_output('"Hello there."') == "Hello there."
    assert _clean_output("“Hello.”") == "Hello."
    # Interior quotes are untouched.
    assert _clean_output('He said "hi" to me') == 'He said "hi" to me'


def test_clean_plain_text_passthrough():
    assert _clean_output("  Fix the bug.  ") == "Fix the bug."


# ── capture_context ──────────────────────────────────────────────────────────


class _Capture:
    """subprocess.run stand-in: per-argv canned results."""

    def __init__(self, results):
        self.results = results  # first argv token -> (returncode, stdout)
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        rc, out = self.results.get(cmd[0], (1, ""))
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(returncode=rc, stdout=out, stderr="")


def test_capture_context_collects_all_sources(monkeypatch):
    cap = _Capture(
        {
            "hyprctl": (0, '{"class": "kitty", "title": "~/src — zsh"}'),
            "wl-paste": (0, "selected words"),
        }
    )
    monkeypatch.setattr(subprocess, "run", cap)
    ctx = capture_context(max_chars=100, env={})
    assert ctx.window_class == "kitty"
    assert ctx.window_title == "~/src — zsh"
    assert ctx.selection == "selected words"
    assert ctx.clipboard == "selected words"
    # Selection uses --primary; clipboard doesn't.
    pastes = [c for c, _ in cap.calls if c[0] == "wl-paste"]
    assert any("--primary" in c for c in pastes)
    assert any("--primary" not in c for c in pastes)


def test_capture_context_degrades_on_failure(monkeypatch):
    cap = _Capture(
        {
            "hyprctl": (1, ""),
            "wl-paste": (0, TimeoutError("slow")),
        }
    )
    monkeypatch.setattr(subprocess, "run", cap)
    ctx = capture_context(max_chars=100, env={})
    assert ctx == ScreenContext()  # everything empty, nothing raised


def test_capture_context_truncates_and_drops_binary(monkeypatch):
    cap = _Capture(
        {
            "hyprctl": (0, "{}"),
            "wl-paste": (0, "x" * 50),
        }
    )
    monkeypatch.setattr(subprocess, "run", cap)
    ctx = capture_context(max_chars=10, env={})
    assert ctx.selection.startswith("x" * 10)
    assert ctx.selection.endswith("…[truncated]")

    cap = _Capture({"hyprctl": (0, "{}"), "wl-paste": (0, "PNG\x00binary")})
    monkeypatch.setattr(subprocess, "run", cap)
    assert capture_context(max_chars=10, env={}).clipboard == ""


# ── key resolution / backend selection ───────────────────────────────────────


def test_key_file_wins_over_env(tmp_path, monkeypatch):
    key_file = tmp_path / "k"
    key_file.write_text("sk-file\n")
    monkeypatch.setenv("TEST_LLM_KEY", "sk-env")
    profile = ContextualProfile(api_key_env="TEST_LLM_KEY", api_key_file=str(key_file))
    assert resolve_api_key(profile) == "sk-file"


def test_key_falls_back_to_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "sk-env")
    profile = ContextualProfile(
        api_key_env="TEST_LLM_KEY", api_key_file=str(tmp_path / "missing")
    )
    assert resolve_api_key(profile) == "sk-env"


def test_remote_endpoint_without_key_is_unavailable(monkeypatch):
    monkeypatch.delenv("TEST_LLM_KEY", raising=False)
    profile = ContextualProfile(
        backend="openai",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="TEST_LLM_KEY",
    )
    with pytest.raises(TransformUnavailable):
        make_backend(profile, timeout_s=5.0)


def test_local_endpoint_needs_no_key():
    profile = ContextualProfile(backend="openai", base_url="http://127.0.0.1:8890/v1")
    backend = make_backend(profile, timeout_s=5.0)
    assert backend is not None


def test_anthropic_without_key_is_unavailable(monkeypatch):
    monkeypatch.delenv("NOPE_KEY", raising=False)
    profile = ContextualProfile(backend="anthropic", api_key_env="NOPE_KEY")
    with pytest.raises(TransformUnavailable):
        make_backend(profile, timeout_s=5.0)


# ── prompt assembly / Transformer ────────────────────────────────────────────


def test_build_user_message_placeholders():
    msg = build_user_message(ScreenContext(), "hello world")
    assert "class=(unknown)" in msg
    assert "PRIMARY SELECTION:\n(empty)" in msg
    assert msg.rstrip().endswith("hello world")


class FakeBackend:
    def __init__(self, reply="OK"):
        self.reply = reply
        self.calls = []

    def complete(self, system, messages, *, model, max_tokens):
        self.calls.append(
            {
                "system": system,
                "messages": messages,
                "user": messages[-1]["content"],
                "model": model,
            }
        )
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _make_transformer(monkeypatch, backend, ctx=None):
    cfg = ContextualConfig()
    monkeypatch.setattr(transform_mod, "make_backend", lambda *a, **k: backend)
    monkeypatch.setattr(
        transform_mod, "capture_context", lambda **k: ctx or ScreenContext()
    )
    return Transformer(cfg)


def test_transformer_threads_context_and_cleans(monkeypatch):
    backend = FakeBackend(reply="<think>x</think>Polished text.")
    ctx = ScreenContext(window_class="vesktop", window_title="#general")
    tr = _make_transformer(monkeypatch, backend, ctx)
    out = tr.transform("polish text")
    assert out == "Polished text."
    call = backend.calls[0]
    assert "dictation post-processor" in call["system"]
    assert "class=vesktop" in call["user"]
    assert call["model"] == "qwen3.5-9b"  # default local profile


def test_transformer_wraps_backend_errors(monkeypatch):
    tr = _make_transformer(monkeypatch, FakeBackend(reply=RuntimeError("boom")))
    with pytest.raises(TransformError):
        tr.transform("hi")


def test_transformer_empty_output_is_error(monkeypatch):
    tr = _make_transformer(monkeypatch, FakeBackend(reply="<think>only</think>"))
    with pytest.raises(TransformError):
        tr.transform("hi")


def test_transformer_rejects_unknown_profile():
    cfg = ContextualConfig(profile="ghost")
    with pytest.raises(TransformUnavailable):
        Transformer(cfg)


# ── OpenRouter reasoning exclusion ───────────────────────────────────────────


def test_openrouter_gets_reasoning_exclude(monkeypatch):
    from wldictate.transform import OpenAICompatBackend

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            msg = SimpleNamespace(content="out")
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    def fake_client(self, base_url, api_key, timeout_s):
        self._is_openrouter = "openrouter.ai" in base_url
        self._client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )

    monkeypatch.setattr(OpenAICompatBackend, "__init__", fake_client)
    b = OpenAICompatBackend("https://openrouter.ai/api/v1", "k", 5.0)
    assert b.complete("s", "u", model="m", max_tokens=10) == "out"
    assert captured["extra_body"] == {"reasoning": {"exclude": True}}

    captured.clear()
    b = OpenAICompatBackend("http://127.0.0.1:8890/v1", "", 5.0)
    b.complete("s", "u", model="m", max_tokens=10)
    assert captured["extra_body"] is None


# ── preamble/label stripping ─────────────────────────────────────────────────


def test_clean_strips_announcement_preambles():
    # The exact in-the-wild failure: Chinese translation announced with a label.
    assert (
        _clean_output("这是中文翻译：它最终并五十年前已经分裂的China.")
        == "它最终并五十年前已经分裂的China."
    )
    assert _clean_output("Here is the translation: Bonjour tout le monde.") == (
        "Bonjour tout le monde."
    )
    assert _clean_output("Translation: Hola.") == "Hola."
    assert _clean_output("Sure, here you go: The fixed sentence.") == (
        "The fixed sentence."
    )
    assert _clean_output("以下是翻译：你好。") == "你好。"
    assert _clean_output("Rewritten version: Better words.") == "Better words."


def test_clean_preserves_legitimate_colons():
    # Dictated content that happens to contain colons must survive.
    assert _clean_output("Note: buy milk tomorrow") == "Note: buy milk tomorrow"
    assert _clean_output("TODO: fix the login bug") == "TODO: fix the login bug"
    assert _clean_output("Meeting at 3:30 pm works for me") == (
        "Meeting at 3:30 pm works for me"
    )
    assert _clean_output("Dear team: the release slips a week.") == (
        "Dear team: the release slips a week."
    )


def test_clean_never_strips_to_nothing():
    # An output that IS just a label-looking string stays rather than vanishing.
    assert _clean_output("Translation:") == "Translation:"
