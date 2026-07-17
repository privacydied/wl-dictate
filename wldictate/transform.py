"""Contextual dictation: LLM transform of finalized utterances.

The contextual mode (Ctrl+Alt+D) runs the normal correcting engine — raw words
appear instantly and self-correct — and, when an utterance finalizes, sends the
transcript plus lightweight screen context (focused window class/title, primary
selection, clipboard) to an LLM. The LLM's output replaces the utterance in
place via the CorrectingEmitter.

Two backends:

- ``openai``   — any OpenAI-compatible server (local llama.cpp ``llama-server``,
                 OpenRouter, ...), via the official ``openai`` SDK.
- ``anthropic`` — the Anthropic API via the official ``anthropic`` SDK (never
                 through an OpenAI-compatible shim).

Threading model: the LLM call (and context capture) run on a private
single-worker executor; only strings cross the boundary. All emitter/formatter
access happens on the session-loop thread via ``TransformCoordinator.poll()`` /
``drain()``, and a pending transform is cancelled before the next utterance
starts — so the emitter is only ever touched between ``finalize()`` and the
next ``begin_utterance()``. No locks needed.
"""

from __future__ import annotations

import os
import re
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Protocol

from .config import ContextualConfig, ContextualProfile
from .emitter import CorrectingEmitter, _guess_wayland_env, focused_window
from .notify import notify
from .textproc import TextFormatter

#: Backspace budget for a full-utterance replacement (the live decode path
#: keeps its own tighter cap; a replacement is one-shot and must not truncate).
_TRANSFORM_MAX_BACKSPACES = 4000

_RE_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_RE_LEADING_ORPHAN_THINK = re.compile(r"\A.*?</think>", re.DOTALL)
_RE_FENCE = re.compile(r"\A```[a-zA-Z0-9_+-]*\n(.*?)\n?```\Z", re.DOTALL)
#: Announcement labels some models prefix despite instructions ("Here is the
#: translation:", "这是中文翻译：", "Output:"). Stripped once from the start;
#: keyword-gated so legitimate colon-bearing dictation survives.
_RE_PREAMBLE = re.compile(
    r"\A(?:"
    r"(?:sure|okay|of course)[^:：\n]{0,40}"
    r"|here(?:'s| is)[^:：\n]{0,50}"
    r"|[^:：\n]{0,30}(?:translation|translated|rewritten|corrected|cleaned|"
    r"polished|output|result|reply|response|version)[^:：\n]{0,20}"
    r"|[^:：\n]{0,20}(?:翻译|译文|重写|输出|结果|回复)[^:：\n]{0,12}"
    r"|(?:这是|以下是|以下为)[^:：\n]{0,24}"
    r")[:：]\s*",
    re.IGNORECASE,
)


class TransformError(Exception):
    """The transform failed (network, timeout, empty output, ...)."""


class TransformUnavailable(Exception):
    """The transform cannot be constructed (missing key, bad profile, ...)."""


# ── Context capture ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScreenContext:
    window_class: str = ""
    window_title: str = ""
    selection: str = ""
    clipboard: str = ""


def _wl_paste(args: list[str], env: dict[str, str]) -> str:
    try:
        result = subprocess.run(
            ["wl-paste", *args, "--no-newline"],
            env=env,
            timeout=0.5,
            capture_output=True,
            text=True,
            errors="replace",
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    text = result.stdout
    if "\x00" in text:  # binary clipboard content (image, ...)
        return ""
    return text


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…[truncated]"


def capture_context(
    *, max_chars: int = 4000, env: dict[str, str] | None = None
) -> ScreenContext:
    """Best-effort screen context; every source degrades to "" on failure.

    Runs on the transform worker thread — never the session loop.
    """
    if env is None:
        env = os.environ.copy()
        _guess_wayland_env(env)
    window_class, window_title = focused_window(env)
    selection = _truncate(_wl_paste(["--primary"], env), max_chars)
    clipboard = _truncate(_wl_paste([], env), max_chars)
    return ScreenContext(
        window_class=window_class,
        window_title=_truncate(window_title, 300),
        selection=selection,
        clipboard=clipboard,
    )


# ── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a dictation post-processor. The user spoke into a microphone; you receive
the raw speech-to-text transcript plus context about their screen. Produce the
exact text that should be typed at their cursor, and nothing else.

Rules:
- Output ONLY the text to type. No preamble, no explanations, no quotes around
  the output, no markdown fences, no <think> blocks.
- Never announce or label the output. Wrong: "Here is the translation: ...",
  "这是中文翻译：...", "Rewritten: ...". Start directly with the content itself.
- By default treat the transcript as dictation: fix recognition errors, grammar,
  punctuation, and capitalization, but preserve the speaker's meaning, wording,
  and voice. Do not expand, summarize, or embellish plain dictation.
- If the transcript is an instruction about text (e.g. "reply to this saying I
  can't make it", "rewrite that more formally", "translate this to Spanish",
  "make it a bullet list"), execute the instruction and output only the result —
  never type the instruction itself.
- Match the register of the focused application: terse and technical in a
  terminal or editor; casual in chat apps like Discord; clear prose in
  documents, email, and browsers. Infer this from the window class and title.
- The PRIMARY SELECTION and CLIPBOARD are reference material the user may mean
  by "this" or "that". Use them to resolve references; never copy them into the
  output unless explicitly asked.
- The output is typed at the current cursor position, possibly continuing
  existing text. No leading or trailing blank lines. Prefer a single line
  unless the content clearly requires line breaks.
- If the transcript is empty or pure noise, output nothing.

Deciding dictation vs instruction: ask "is the user telling ME to produce
something, or are these words the message itself?" Requests aimed at you —
"give me…", "write…", "reply to this…", "translate…", "say X in Spanish",
"make that shorter" — are instructions: output the requested artifact, never
the request. Everything else is dictation to clean up.

Examples (transcript -> output):
- "i recieve teh package yesterday" -> "I received the package yesterday."
- "give me the command to find what's using that port"
  (clipboard: "could not bind to 0.0.0.0:8080")
  -> "lsof -i :8080"
- "say good morning everyone the build is fixed in spanish"
  -> "Buenos días a todos, la build está arreglada."
- (selection: "can you review my PR today?")
  "reply to this say i can do it after lunch"
  -> "I can do it after lunch."
- "um so tell them uh the meeting moved to thursday 3 pm"
  -> "The meeting has been moved to Thursday at 3 p.m."
"""

_USER_TEMPLATE = """\
FOCUSED WINDOW: class={window_class} title={window_title}

PRIMARY SELECTION:
{selection}

CLIPBOARD:
{clipboard}

TRANSCRIPT:
{transcript}
"""


def build_user_message(context: ScreenContext, transcript: str) -> str:
    return _USER_TEMPLATE.format(
        window_class=context.window_class or "(unknown)",
        window_title=context.window_title or "(unknown)",
        selection=context.selection or "(empty)",
        clipboard=context.clipboard or "(empty)",
        transcript=transcript,
    )


def _clean_output(text: str) -> str:
    """Defensively normalize LLM output into plain typeable text."""
    text = _RE_THINK.sub("", text)
    if "</think>" in text:  # unmatched leading reasoning block
        text = _RE_LEADING_ORPHAN_THINK.sub("", text, count=1)
    text = text.strip()
    fence = _RE_FENCE.match(text)
    if fence:
        text = fence.group(1).strip()
    stripped = _RE_PREAMBLE.sub("", text, count=1).strip()
    if stripped:  # never strip the whole output down to nothing
        text = stripped
    for opening, closing in (('"', '"'), ("'", "'"), ("“", "”")):
        if len(text) >= 2 and text[0] == opening and text[-1] == closing:
            text = text[1:-1].strip()
            break
    return text


# ── Backends ─────────────────────────────────────────────────────────────────


class TransformBackend(Protocol):
    def complete(
        self, system: str, user: str, *, model: str, max_tokens: int
    ) -> str: ...


class OpenAICompatBackend:
    """OpenAI-compatible chat completions: llama.cpp llama-server, OpenRouter."""

    def __init__(self, base_url: str, api_key: str, timeout_s: float) -> None:
        from openai import OpenAI  # lazy: standard mode never imports this

        self._is_openrouter = "openrouter.ai" in base_url
        # Prewarm only makes sense (and is free) against a local server.
        self._is_local = base_url.startswith(("http://127.", "http://localhost"))
        # The SDK requires a non-empty key; local llama-server ignores it.
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key or "none",
            timeout=timeout_s,
            max_retries=0,
        )

    def prewarm(self, system: str, user_prefix: str, *, model: str) -> None:
        """Speculative prefill: process system+context into the server's KV
        cache (llama.cpp ``--cache-prompt``/``--cache-reuse``) while the user
        is still speaking, so the real request only pays for the transcript.
        Local servers only — remote calls would cost money per utterance.
        """
        if not self._is_local:
            return
        self._client.chat.completions.create(
            model=model,
            max_tokens=1,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prefix},
            ],
        )

    def complete(self, system: str, user: str, *, model: str, max_tokens: int) -> str:
        extra_body = {}
        if self._is_openrouter:
            # Reasoning models (e.g. nemotron ...:reasoning) must not leak
            # their chain of thought into the typed text.
            extra_body["reasoning"] = {"exclude": True}
        response = self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            extra_body=extra_body or None,
        )
        choice = response.choices[0] if response.choices else None
        return (choice.message.content or "") if choice else ""


class AnthropicBackend:
    """Anthropic API via the official SDK."""

    def __init__(self, api_key: str, timeout_s: float) -> None:
        import anthropic  # lazy

        self._client = anthropic.Anthropic(
            api_key=api_key, timeout=timeout_s, max_retries=0
        )

    def complete(self, system: str, user: str, *, model: str, max_tokens: int) -> str:
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in response.content if b.type == "text")


def resolve_api_key(profile: ContextualProfile) -> str:
    """Key file wins over env; "" when neither is configured."""
    if profile.api_key_file:
        path = os.path.expanduser(profile.api_key_file)
        try:
            with open(path) as f:
                key = f.read().strip()
            if key:
                return key
        except OSError:
            pass  # fall through to env
    if profile.api_key_env:
        return os.environ.get(profile.api_key_env, "").strip()
    return ""


def make_backend(profile: ContextualProfile, timeout_s: float) -> TransformBackend:
    key = resolve_api_key(profile)
    if profile.backend == "anthropic":
        if not key:
            raise TransformUnavailable(
                f"no Anthropic API key: set ${profile.api_key_env or 'ANTHROPIC_API_KEY'}"
                f" or write it to {profile.api_key_file or 'an api_key_file'}"
            )
        return AnthropicBackend(key, timeout_s)
    key_source_configured = bool(profile.api_key_env or profile.api_key_file)
    local = profile.base_url.startswith(("http://127.", "http://localhost"))
    if not key and key_source_configured and not local:
        # A key source is configured but resolves empty and the endpoint is
        # remote — fail up-front with a clear message instead of a 401 later.
        # Local servers need no key.
        raise TransformUnavailable(
            f"no API key for endpoint {profile.base_url}: set"
            f" ${profile.api_key_env or '<api_key_env>'} or write it to"
            f" {profile.api_key_file or 'an api_key_file'}"
        )
    return OpenAICompatBackend(profile.base_url, key, timeout_s)


# ── Transformer ──────────────────────────────────────────────────────────────


class Transformer:
    """Resolves the active profile and turns transcripts into final text."""

    def __init__(self, cfg: ContextualConfig) -> None:
        self._cfg = cfg
        profile = cfg.profiles.get(cfg.profile)
        if profile is None:
            raise TransformUnavailable(f"unknown contextual profile '{cfg.profile}'")
        if not profile.model:
            raise TransformUnavailable(
                f"contextual profile '{cfg.profile}' has no model configured"
            )
        self._profile = profile
        self._backend = make_backend(profile, cfg.timeout_s)

    @property
    def profile_name(self) -> str:
        return self._cfg.profile

    def prefetch_context(self) -> ScreenContext:
        """Capture screen context now (utterance start) for reuse at submit."""
        return capture_context(max_chars=self._cfg.context_max_chars)

    def prewarm(self, context: ScreenContext) -> None:
        """Best-effort speculative prefill of the LLM with system+context."""
        prewarm = getattr(self._backend, "prewarm", None)
        if prewarm is None:
            return
        # The prefix shared with the eventual real request: everything up to
        # (and including) the "TRANSCRIPT:" header.
        user_prefix = build_user_message(context, "")
        try:
            prewarm(SYSTEM_PROMPT, user_prefix, model=self._profile.model)
        except Exception:
            pass  # purely opportunistic

    def transform(self, transcript: str, context: ScreenContext | None = None) -> str:
        if context is None:
            context = capture_context(max_chars=self._cfg.context_max_chars)
        user = build_user_message(context, transcript)
        try:
            raw = self._backend.complete(
                SYSTEM_PROMPT,
                user,
                model=self._profile.model,
                max_tokens=self._cfg.max_output_tokens,
            )
        except Exception as e:  # SDK errors, timeouts, network
            raise TransformError(f"{type(e).__name__}: {e}") from e
        text = _clean_output(raw)
        if not text:
            raise TransformError("empty transform output")
        return text


# ── Coordinator ──────────────────────────────────────────────────────────────


class TransformCoordinator:
    """Bridges the session loop and the LLM thread.

    Invariants:
    - ``submit``/``cancel_pending``/``poll``/``drain`` are called only from the
      session-loop thread.
    - The LLM thread only runs ``Transformer.transform`` (pure I/O → string).
    - The emitter/formatter are touched only between an utterance's
      ``finalize()`` and the next ``begin_utterance()`` — guaranteed because
      the loop calls ``cancel_pending()`` before ``session.start_utterance()``.
    """

    def __init__(
        self,
        transformer: Transformer,
        emitter: CorrectingEmitter,
        formatter: TextFormatter,
        *,
        timeout_s: float,
        notify_enabled: bool = True,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self._transformer = transformer
        self._emitter = emitter
        self._formatter = formatter
        self._timeout_s = timeout_s
        self._notify_enabled = notify_enabled
        self._on_error = on_error
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="transform")
        self._pending: tuple[Future, int, str] | None = None
        self._generation = 0
        # (generation, context) captured at utterance start; written and read
        # on the single transform-pool thread (ordering guarantees safety).
        self._prefetched: tuple[int, ScreenContext] | None = None
        self._notified_failure = False

    # ── Session-loop API ────────────────────────────────────────────────

    def prefetch(self) -> None:
        """Utterance started: capture context + prewarm the LLM *now*, hiding
        both costs under the user's own speech time."""
        generation = self._generation

        def run() -> None:
            context = self._transformer.prefetch_context()
            self._prefetched = (generation, context)
            self._transformer.prewarm(context)

        self._pool.submit(run)

    def submit(self, final_text: str) -> None:
        """Queue the finalized utterance text for transformation."""
        if not final_text.strip():
            return
        generation = self._generation
        transcript = final_text.strip()

        def run() -> str:
            # Runs after any prefetch task (same single-worker pool), so the
            # prefetched context is either ready or from a stale generation.
            pre = self._prefetched
            context = pre[1] if pre is not None and pre[0] == generation else None
            return self._transformer.transform(transcript, context=context)

        future = self._pool.submit(run)
        self._pending = (future, generation, final_text)
        if self._notify_enabled:
            notify("Transforming…", timeout_ms=1500)

    def cancel_pending(self) -> None:
        """Discard any in-flight transform (a new utterance is starting)."""
        self._generation += 1
        self._pending = None

    def poll(self) -> None:
        """Apply a completed transform, if any. Session-loop thread only."""
        if self._pending is None:
            return
        future, generation, original = self._pending
        if not future.done():
            return
        self._pending = None
        if generation != self._generation:
            return  # stale — cancelled between completion and poll
        self._finish(future, original)

    def drain(self, timeout_s: float | None = None) -> None:
        """Session stop: wait (bounded) for an in-flight transform and apply."""
        if self._pending is None:
            return
        future, generation, original = self._pending
        self._pending = None
        if generation != self._generation:
            return
        try:
            future.result(timeout=self._timeout_s if timeout_s is None else timeout_s)
        except Exception:
            pass  # _finish re-reads the (now settled or timed-out) future
        if future.done():
            self._finish(future, original)
        else:
            self._fail("transform timed out")

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ── Internals (session-loop thread) ─────────────────────────────────

    def _finish(self, future: Future, original: str) -> None:
        try:
            transformed = future.result(timeout=0)
        except Exception as e:
            self._fail(str(e))
            return
        self._apply(original, transformed)

    def _apply(self, original: str, transformed: str) -> None:
        # Preserve the inter-utterance separator the formatter emitted (the
        # on-screen text may start with a space joining it to prior text).
        prefix = original[: len(original) - len(original.lstrip())]
        body = prefix + transformed.strip()
        if body == original.rstrip() or body == original:
            return  # no-op: keep the Whisper text, zero keystrokes
        self._formatter.reseed(body)
        trailer = self._formatter.end_utterance()
        if not self._emitter.sync(
            body + trailer, max_backspaces=_TRANSFORM_MAX_BACKSPACES, bulk=True
        ):
            self._error("emitter failed to apply transform")

    def _fail(self, msg: str) -> None:
        self._error(f"transform failed (keeping dictated text): {msg}")
        if self._notify_enabled and not self._notified_failure:
            self._notified_failure = True
            notify("Contextual transform failed — keeping dictated text")

    def _error(self, msg: str) -> None:
        if self._on_error is not None:
            self._on_error(msg)
