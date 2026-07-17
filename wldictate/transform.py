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
import base64
import json
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Empty, SimpleQueue
from dataclasses import dataclass
from dataclasses import field as dataclasses_field
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
    screenshot: bytes | None = dataclasses_field(default=None, repr=False)


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


#: Cap the screenshot's long edge (vision models resample anyway; smaller
#: images prefill faster and cost fewer tokens).
_SCREENSHOT_MAX_EDGE = 1120


def capture_screenshot(env: dict[str, str]) -> bytes | None:
    """PNG of the focused window via grim (Hyprland geometry); None on any
    failure (grim missing, no focused window, X11, ...)."""
    try:
        result = subprocess.run(
            ["hyprctl", "-j", "activewindow"],
            env=env,
            timeout=1.0,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        x, y = data["at"]
        w, h = data["size"]
        if w <= 0 or h <= 0:
            return None
        scale = min(1.0, _SCREENSHOT_MAX_EDGE / max(w, h))
        cmd = ["grim"]
        if scale < 1.0:
            cmd += ["-s", f"{scale:.3f}"]
        cmd += ["-g", f"{x},{y} {w}x{h}", "-"]
        shot = subprocess.run(cmd, env=env, timeout=2.0, capture_output=True)
        if shot.returncode != 0 or not shot.stdout:
            return None
        return shot.stdout
    except Exception:
        return None


def capture_context(
    *,
    max_chars: int = 4000,
    env: dict[str, str] | None = None,
    include_screenshot: bool = False,
) -> ScreenContext:
    """Best-effort screen context; every source degrades to ""/None on
    failure. Runs on the transform worker thread — never the session loop.
    """
    if env is None:
        env = os.environ.copy()
        _guess_wayland_env(env)
    window_class, window_title = focused_window(env)
    selection = _truncate(_wl_paste(["--primary"], env), max_chars)
    clipboard = _truncate(_wl_paste([], env), max_chars)
    screenshot = capture_screenshot(env) if include_screenshot else None
    return ScreenContext(
        window_class=window_class,
        window_title=_truncate(window_title, 300),
        selection=selection,
        clipboard=clipboard,
        screenshot=screenshot,
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
- When executing an instruction to reply to or continue text that is in a
  different language than the transcript, write the output in THAT language
  (replying to a German thread -> German), unless told otherwise. Plain
  dictation always stays in the language the user spoke.
- The PRIMARY SELECTION and CLIPBOARD are reference material the user may mean
  by "this" or "that". Use them to resolve references; never copy them into the
  output unless explicitly asked.
- A SCREENSHOT of the focused window may be attached: read it to understand
  what is on screen (the conversation being replied to, the document, the
  page). Prefer the selection when both could answer a reference.
- The output is typed at the current cursor position, possibly continuing
  existing text. No leading or trailing blank lines. Prefer a single line
  unless the content clearly requires line breaks.
- If the transcript is empty or pure noise, output nothing.
- If the transcript asks you to change YOUR PREVIOUS OUTPUT (shown as your
  last reply in this conversation) — "make it shorter", "actually sign it
  'cheers'", "add an emoji to that" — begin your reply with the exact marker
  @@REVISE@@ followed by the full replacement text. The previous output will
  be replaced on screen. Only use @@REVISE@@ for revisions of your own
  previous output, never for new content, and never on the first exchange of
  a conversation.

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
- (your previous output: "I can do it after lunch.")
  "make that sound more enthusiastic"
  -> "@@REVISE@@Absolutely — I'll get on it right after lunch!"
"""

#: Output prefix marking a revision of the previous output (see SYSTEM_PROMPT).
REVISE_MARKER = "@@REVISE@@"

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


def _clean_partial(raw: str) -> tuple[str, bool]:
    """Best-effort clean of an IN-PROGRESS streamed output.

    Returns (text_safe_to_show_so_far, is_revise). Conservative where the
    stream could still change meaning (partial marker, unclosed think block,
    unterminated fence header); the final authoritative sync with
    ``_clean_output`` fixes anything this lets through — the correcting
    emitter absorbs the difference with backspaces.
    """
    text = raw.lstrip()
    revise = False
    if len(text) < len(REVISE_MARKER) and REVISE_MARKER.startswith(text):
        return "", False  # could still become the marker: hold
    if text.startswith(REVISE_MARKER):
        revise = True
        text = text[len(REVISE_MARKER) :].lstrip()
    text = _RE_THINK.sub("", text)
    if "<think>" in text:
        text = text.split("<think>", 1)[0]  # unclosed block: hold its content
    elif "</think>" in text:
        text = text.split("</think>", 1)[1]  # orphan close
    text = text.lstrip()
    if text.startswith("`"):
        newline = text.find("\n")
        if text.startswith("```") and newline != -1:
            text = text[newline + 1 :]
        elif len(text) <= 3:
            return "", revise  # could be a fence opening: hold
    stripped = _RE_PREAMBLE.sub("", text, count=1)
    if stripped != text:
        text = stripped.lstrip()
    return text.rstrip("`").rstrip(), revise


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
        self,
        system: str,
        messages: list[dict],
        *,
        model: str,
        max_tokens: int,
    ) -> str: ...


class OpenAICompatBackend:
    """OpenAI-compatible chat completions: llama.cpp llama-server, OpenRouter."""

    def __init__(self, base_url: str, api_key: str, timeout_s: float) -> None:
        from openai import OpenAI  # lazy: standard mode never imports this

        self._is_openrouter = "openrouter.ai" in base_url
        # Prewarm only makes sense (and is free) against a local server; the
        # screenshot privacy default ("local") also keys off this.
        self.is_local = base_url.startswith(("http://127.", "http://localhost"))
        # The SDK requires a non-empty key; local llama-server ignores it.
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key or "none",
            timeout=timeout_s,
            max_retries=0,
        )

    def prewarm(self, system: str, messages: list[dict], *, model: str) -> None:
        """Speculative prefill: process system+history+context into the
        server's KV cache (llama.cpp ``--cache-prompt``/``--cache-reuse``)
        while the user is still speaking, so the real request only pays for
        the transcript. Local servers only — remote calls would cost money.
        """
        if not self.is_local:
            return
        self._client.chat.completions.create(
            model=model,
            max_tokens=1,
            messages=[{"role": "system", "content": system}, *messages],
        )

    def complete(
        self, system: str, messages: list[dict], *, model: str, max_tokens: int
    ) -> str:
        extra_body = {}
        if self._is_openrouter:
            # Reasoning models (e.g. nemotron ...:reasoning) must not leak
            # their chain of thought into the typed text.
            extra_body["reasoning"] = {"exclude": True}
        response = self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, *messages],
            extra_body=extra_body or None,
        )
        choice = response.choices[0] if response.choices else None
        return (choice.message.content or "") if choice else ""

    def complete_stream(
        self, system: str, messages: list[dict], *, model: str, max_tokens: int
    ):
        extra_body = {}
        if self._is_openrouter:
            extra_body["reasoning"] = {"exclude": True}
        stream = self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, *messages],
            extra_body=extra_body or None,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

    @staticmethod
    def user_content(text: str, screenshot: bytes | None):
        if screenshot is None:
            return text
        b64 = base64.standard_b64encode(screenshot).decode("ascii")
        return [
            # Image first: the text template stays a stable suffix, and the
            # prewarm request's KV prefix matches the real request's.
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": text},
        ]


class AnthropicBackend:
    """Anthropic API via the official SDK."""

    is_local = False

    def __init__(self, api_key: str, timeout_s: float) -> None:
        import anthropic  # lazy

        self._client = anthropic.Anthropic(
            api_key=api_key, timeout=timeout_s, max_retries=0
        )

    def complete(
        self, system: str, messages: list[dict], *, model: str, max_tokens: int
    ) -> str:
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return "".join(b.text for b in response.content if b.type == "text")

    def complete_stream(
        self, system: str, messages: list[dict], *, model: str, max_tokens: int
    ):
        with self._client.messages.stream(
            model=model, max_tokens=max_tokens, system=system, messages=messages
        ) as stream:
            for text in stream.text_stream:
                yield text

    @staticmethod
    def user_content(text: str, screenshot: bytes | None):
        if screenshot is None:
            return text
        b64 = base64.standard_b64encode(screenshot).decode("ascii")
        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": b64,
                },
            },
            {"type": "text", "text": text},
        ]


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
        # Per-session system prompt: static base + who the speaker is + their
        # vocabulary. Stable for the session, so the KV-cache prefix holds.
        self._system = SYSTEM_PROMPT
        if cfg.persona.strip():
            self._system += f"\nABOUT THE SPEAKER:\n{cfg.persona.strip()}\n"
        if cfg.vocabulary:
            self._system += (
                "\nSPEAKER VOCABULARY (names/terms that may be misheard — "
                "prefer these spellings): " + ", ".join(cfg.vocabulary) + "\n"
            )
        self._app_hints = {k.lower(): v for k, v in cfg.app_hints.items()}
        # Screenshot policy: "local" keeps images on-machine (privacy
        # default); "always" also sends them to cloud profiles; "off" never.
        self._include_screenshot = cfg.screenshot == "always" or (
            cfg.screenshot == "local" and getattr(self._backend, "is_local", False)
        )

    @property
    def profile_name(self) -> str:
        return self._cfg.profile

    def prefetch_context(self) -> ScreenContext:
        """Capture screen context now (utterance start) for reuse at submit."""
        return capture_context(
            max_chars=self._cfg.context_max_chars,
            include_screenshot=self._include_screenshot,
        )

    @staticmethod
    def _history_messages(history: tuple[tuple[str, str], ...]) -> list[dict]:
        """Prior (transcript -> output) exchanges as chat turns, so follow-up
        commands ("make that shorter") can resolve what "that" is."""
        messages: list[dict] = []
        for transcript, output in history:
            messages.append({"role": "user", "content": transcript})
            messages.append({"role": "assistant", "content": output})
        return messages

    def prewarm(
        self,
        context: ScreenContext,
        history: tuple[tuple[str, str], ...] = (),
    ) -> None:
        """Best-effort speculative prefill of the LLM with system+history+
        context — everything the real request will contain except the
        transcript."""
        prewarm = getattr(self._backend, "prewarm", None)
        if prewarm is None:
            return
        messages = self._history_messages(history)
        messages.append(
            {"role": "user", "content": self._user_content(context, "")}
        )
        try:
            prewarm(self._system, messages, model=self._profile.model)
        except Exception:
            pass  # purely opportunistic

    def _user_content(self, context: ScreenContext, transcript: str):
        text = build_user_message(context, transcript)
        hint = self._app_hint(context.window_class)
        if hint:
            text += f"\nAPP GUIDANCE: {hint}\n"
        user_content = getattr(self._backend, "user_content", None)
        if user_content is None:
            return text
        return user_content(text, context.screenshot)

    def _app_hint(self, window_class: str) -> str:
        cls = window_class.lower()
        if not cls:
            return ""
        for needle, hint in self._app_hints.items():
            if needle in cls:
                return hint
        return ""

    def transform(
        self,
        transcript: str,
        context: ScreenContext | None = None,
        history: tuple[tuple[str, str], ...] = (),
    ) -> str:
        if context is None:
            context = capture_context(
                max_chars=self._cfg.context_max_chars,
                include_screenshot=self._include_screenshot,
            )
        messages = self._history_messages(history)
        messages.append({"role": "user", "content": self._user_content(context, transcript)})
        try:
            raw = self._backend.complete(
                self._system,
                messages,
                model=self._profile.model,
                max_tokens=self._cfg.max_output_tokens,
            )
        except Exception as e:  # SDK errors, timeouts, network
            raise TransformError(f"{type(e).__name__}: {e}") from e
        text = _clean_output(raw)
        if not text:
            raise TransformError("empty transform output")
        return text

    def transform_stream(
        self,
        transcript: str,
        context: ScreenContext | None = None,
        history: tuple[tuple[str, str], ...] = (),
    ):
        """Yield RAW output deltas as they generate (caller cleans/applies
        incrementally). Falls back to one chunk when the backend can't stream.
        """
        if context is None:
            context = capture_context(
                max_chars=self._cfg.context_max_chars,
                include_screenshot=self._include_screenshot,
            )
        messages = self._history_messages(history)
        messages.append({"role": "user", "content": self._user_content(context, transcript)})
        stream_fn = getattr(self._backend, "complete_stream", None)
        try:
            if stream_fn is None:
                yield self._backend.complete(
                    self._system,
                    messages,
                    model=self._profile.model,
                    max_tokens=self._cfg.max_output_tokens,
                )
                return
            yield from stream_fn(
                self._system,
                messages,
                model=self._profile.model,
                max_tokens=self._cfg.max_output_tokens,
            )
        except Exception as e:  # SDK errors, timeouts, network
            raise TransformError(f"{type(e).__name__}: {e}") from e


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
        stream_enabled: bool = True,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self._transformer = transformer
        self._emitter = emitter
        self._formatter = formatter
        self._timeout_s = timeout_s
        self._notify_enabled = notify_enabled
        self._on_error = on_error
        self._stream_enabled = stream_enabled and hasattr(
            transformer, "transform_stream"
        )
        # Live stream state (session-loop thread only): events arrive on the
        # queue from the transform thread; poll() applies them.
        self._stream: dict | None = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="transform")
        self._pending: tuple[Future, int, str] | None = None
        self._generation = 0
        # Last few (transcript -> model reply) exchanges: lets follow-up
        # commands resolve "that"/"it" and powers revise-in-place.
        self._history: deque[tuple[str, str]] = deque(maxlen=4)
        # (generation, context) captured at utterance start; written and read
        # on the single transform-pool thread (ordering guarantees safety).
        self._prefetched: tuple[int, ScreenContext] | None = None
        self._notified_failure = False

    # ── Session-loop API ────────────────────────────────────────────────

    def prefetch(self) -> None:
        """Utterance started: capture context + prewarm the LLM *now*, hiding
        both costs under the user's own speech time."""
        generation = self._generation
        history = tuple(self._history)

        def run() -> None:
            context = self._transformer.prefetch_context()
            self._prefetched = (generation, context)
            self._transformer.prewarm(context, history)

        self._pool.submit(run)

    def submit(self, final_text: str, merge_all: bool = False) -> None:
        """Queue the finalized utterance text for transformation.

        ``merge_all``: the text spans a long-speech rollover chain — the
        emitter's accumulated previous region is part of this message, so the
        replacement rewrites the whole chain."""
        if not final_text.strip():
            return
        generation = self._generation
        transcript = final_text.strip()
        history = tuple(self._history)

        if self._stream_enabled:
            events: SimpleQueue = SimpleQueue()

            def run_stream() -> None:
                pre = self._prefetched
                context = pre[1] if pre is not None and pre[0] == generation else None
                try:
                    for delta in self._transformer.transform_stream(
                        transcript, context=context, history=history
                    ):
                        events.put(("delta", delta))
                    events.put(("done", None))
                except Exception as e:
                    events.put(("error", e))

            self._pool.submit(run_stream)
            self._stream = {
                "events": events,
                "generation": generation,
                "original": final_text,
                "raw": "",  # accumulated raw model output
                "prefix": None,  # separator prefix, decided on first visible text
                "merged": False,  # previous region folded in (revise/merge_all)
                "merge_all": merge_all,  # rollover chain: rewrite it all
                "typed": False,  # anything synced yet
            }
        else:

            def run() -> str:
                # Runs after any prefetch task (same single-worker pool), so
                # the prefetched context is ready or from a stale generation.
                pre = self._prefetched
                context = pre[1] if pre is not None and pre[0] == generation else None
                return self._transformer.transform(
                    transcript, context=context, history=history
                )

            future = self._pool.submit(run)
            self._pending = (future, generation, final_text, merge_all)
        if self._notify_enabled:
            notify("Transforming…", timeout_ms=1500)

    def cancel_pending(self) -> None:
        """Discard any in-flight transform (a new utterance is starting)."""
        self._generation += 1
        self._pending = None
        self._stream = None  # partial replacement (if any) simply stays

    def poll(self) -> None:
        """Apply completed/streaming transform work. Session-loop thread only."""
        if self._stream is not None:
            self._poll_stream(block=False)
            return
        if self._pending is None:
            return
        future, generation, original, merge_all = self._pending
        if not future.done():
            return
        self._pending = None
        if generation != self._generation:
            return  # stale — cancelled between completion and poll
        self._finish(future, original, merge_all)

    def drain(self, timeout_s: float | None = None) -> None:
        """Session stop: wait (bounded) for an in-flight transform and apply."""
        budget = self._timeout_s if timeout_s is None else timeout_s
        if self._stream is not None:
            deadline = time.monotonic() + budget
            while self._stream is not None and time.monotonic() < deadline:
                self._poll_stream(block=True, block_s=0.05)
            if self._stream is not None:
                self._stream = None
                self._fail("transform timed out")
            return
        if self._pending is None:
            return
        future, generation, original, merge_all = self._pending
        self._pending = None
        if generation != self._generation:
            return
        try:
            future.result(timeout=budget)
        except Exception:
            pass  # _finish re-reads the (now settled or timed-out) future
        if future.done():
            self._finish(future, original, merge_all)
        else:
            self._fail("transform timed out")

    # ── Streaming apply (session-loop thread) ───────────────────────────

    def _poll_stream(self, *, block: bool, block_s: float = 0.0) -> None:
        state = self._stream
        if state is None:
            return
        if state["generation"] != self._generation:
            self._stream = None
            return
        progressed = False
        while True:
            try:
                kind, payload = state["events"].get(
                    block=block and not progressed, timeout=block_s or None
                )
            except Empty:
                break
            if kind == "delta":
                state["raw"] += payload
                progressed = True
            elif kind == "error":
                self._stream = None
                if state["typed"]:
                    # Partial replacement is on screen — keep it, fix state.
                    self._formatter.reseed(self._emitter.logical)
                    self._error(f"transform stream failed mid-apply: {payload}")
                else:
                    self._fail(str(payload))
                return
            else:  # done
                self._stream = None
                self._stream_finish(state)
                return
        if progressed:
            self._stream_progress(state)

    def _stream_progress(self, state: dict) -> None:
        partial, revise = _clean_partial(state["raw"])
        if not partial:
            return
        if state["merge_all"] and not state["merged"]:
            state["merged"] = self._emitter.merge_previous()
            state["prefix"] = None
        elif revise and not state["merged"] and self._history:
            state["merged"] = self._emitter.merge_previous()
            state["prefix"] = None  # re-derive against the merged region
        if state["prefix"] is None:
            base = self._emitter.logical if state["merged"] else state["original"]
            state["prefix"] = base[: len(base) - len(base.lstrip())]
        if not self._emitter.sync(
            state["prefix"] + partial, max_backspaces=_TRANSFORM_MAX_BACKSPACES
        ):
            self._stream = None
            self._error("emitter failed during streamed transform")
            return
        state["typed"] = True

    def _stream_finish(self, state: dict) -> None:
        cleaned = _clean_output(state["raw"])
        if not cleaned:
            if state["typed"]:
                self._formatter.reseed(self._emitter.logical)
            else:
                self._fail("empty transform output")
            return
        reply = cleaned
        revise = cleaned.startswith(REVISE_MARKER)
        if state["merge_all"] and not state["merged"]:
            state["merged"] = self._emitter.merge_previous()
            state["prefix"] = None
        if revise:
            cleaned = cleaned[len(REVISE_MARKER) :].strip()
            if not cleaned:
                return
            if not state["merged"] and self._history:
                state["merged"] = self._emitter.merge_previous()
        if state["prefix"] is None:
            base = self._emitter.logical if state["merged"] else state["original"]
            state["prefix"] = base[: len(base) - len(base.lstrip())]
        body = state["prefix"] + cleaned
        if not state["typed"] and body in (
            state["original"],
            state["original"].rstrip(),
        ):
            self._history.append((state["original"].strip(), state["original"].strip()))
            return  # no-op
        self._finish_apply(state["original"], reply, body)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ── Internals (session-loop thread) ─────────────────────────────────

    def _finish(self, future: Future, original: str, merge_all: bool = False) -> None:
        try:
            transformed = future.result(timeout=0)
        except Exception as e:
            self._fail(str(e))
            return
        self._apply(original, transformed, merge_all)

    def _apply(self, original: str, transformed: str, merge_all: bool = False) -> None:
        reply = transformed  # recorded in history verbatim (marker included)
        if merge_all and self._emitter.merge_previous():
            # Rollover chain: the accumulated previous region + the current
            # one are ONE message — replace the whole thing.
            if transformed.startswith(REVISE_MARKER):
                transformed = transformed[len(REVISE_MARKER) :].strip() or transformed
            logical = self._emitter.logical
            prefix = logical[: len(logical) - len(logical.lstrip())]
            self._finish_apply(original, reply, prefix + transformed.strip())
            return
        if transformed.startswith(REVISE_MARKER):
            replacement = transformed[len(REVISE_MARKER) :].strip()
            # Revise the PREVIOUS output: fold its region into the mutable
            # region (the spoken revision command is consumed with it).
            # Guard: small models sometimes emit the marker spuriously —
            # honor it only when a previous exchange actually exists.
            if replacement and self._history and self._emitter.merge_previous():
                logical = self._emitter.logical
                prefix = logical[: len(logical) - len(logical.lstrip())]
                self._finish_apply(original, reply, prefix + replacement)
                return
            if not replacement:
                return
            transformed = replacement  # no previous region: plain replacement

        # Preserve the inter-utterance separator the formatter emitted (the
        # on-screen text may start with a space joining it to prior text).
        prefix = original[: len(original) - len(original.lstrip())]
        body = prefix + transformed.strip()
        if body == original.rstrip() or body == original:
            self._history.append((original.strip(), original.strip()))
            return  # no-op: keep the Whisper text, zero keystrokes
        self._finish_apply(original, reply, body)

    def _finish_apply(self, original: str, reply: str, body: str) -> None:
        self._formatter.reseed(body)
        trailer = self._formatter.end_utterance()
        if not self._emitter.sync(
            body + trailer, max_backspaces=_TRANSFORM_MAX_BACKSPACES, bulk=True
        ):
            self._error("emitter failed to apply transform")
            return
        self._history.append((original.strip(), reply))

    def _fail(self, msg: str) -> None:
        self._error(f"transform failed (keeping dictated text): {msg}")
        if self._notify_enabled and not self._notified_failure:
            self._notified_failure = True
            notify("Contextual transform failed — keeping dictated text")

    def _error(self, msg: str) -> None:
        if self._on_error is not None:
            self._on_error(msg)
