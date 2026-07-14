"""Incremental transcript cleanup and spacing.

Ported from the batch regex cleanup in the old ``whisper_dictate.py``, reworked
to operate on per-commit deltas: the formatter owns all spacing state so text
joins correctly across commits and across utterances (the old implementation
used module globals for this).
"""

from __future__ import annotations

import re

# Pre-compiled cleanup regexes (hot path).
_RE_PARENS = re.compile(r"\([^)]*\)\s*")
_RE_BRACKETS = re.compile(r"\[[^\]]*\]\s*")
_RE_DOTS = re.compile(r"(?:\s*\.\s*){2,}")
_RE_SENTENCE_PERIOD = re.compile(r"(?<!\.)\.(\s)?(?!\.)")
_RE_SENTENCE_PUNCT = re.compile(r"([!?;:])(\s)?")
_RE_COMMA_SPACE = re.compile(r",(\s)?")
_RE_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,!?;:])(?!\.)")
_RE_WHITESPACE = re.compile(r"\s+")
_RE_LEADING_PUNCT_SPACE = re.compile(r"^[.,!?;:][\s\u00A0\u200B\u200C\u200D\u2060]*")
_RE_STARTS_WITH_WORD = re.compile(r"^[A-Za-z0-9]")
_RE_STARTS_WITH_PUNCT = re.compile(r"^[.,!?;:'\")\]]")


def clean_text(text: str, *, utterance_start: bool) -> str:
    """Normalize one chunk of raw transcript text."""
    text = _RE_PARENS.sub("", text)
    text = _RE_BRACKETS.sub("", text)
    # "word ," -> "word," (also puts stripped-annotation leftovers in reach of
    # the utterance-start leading-punct rule below).
    text = _RE_SPACE_BEFORE_PUNCT.sub(r"\1", text).strip()
    text = _RE_COMMA_SPACE.sub(", ", text)
    # Order matters: sentence periods BEFORE ellipsis normalization so a "."
    # is never inserted inside "...".
    text = _RE_SENTENCE_PERIOD.sub(". ", text)
    text = _RE_DOTS.sub("... ", text)
    text = _RE_SENTENCE_PUNCT.sub(r"\1 ", text)
    if utterance_start:
        text = _RE_LEADING_PUNCT_SPACE.sub("", text)
    return _RE_WHITESPACE.sub(" ", text).strip()


class TextFormatter:
    """Stateful formatter turning commit deltas into typeable text.

    Guarantees:
    - single spaces between words across commit and utterance boundaries
    - no space before punctuation that continues the previous commit
    - annotations like ``(coughs)`` / ``[music]`` are stripped
    """

    def __init__(self) -> None:
        self._emitted_any = False
        self._utterance_has_output = False

    def on_utterance_start(self) -> None:
        self._utterance_has_output = False

    def format_delta(self, raw: str) -> str:
        """Format one commit delta; returns "" if nothing remains after cleanup."""
        text = clean_text(raw, utterance_start=not self._utterance_has_output)
        if not text:
            return ""
        if self._emitted_any and _RE_STARTS_WITH_WORD.match(text):
            text = f" {text}"
        elif self._emitted_any and not _RE_STARTS_WITH_PUNCT.match(text):
            # Non-alphanumeric, non-punctuation start (e.g. unicode word): still
            # separate it from the previous commit.
            text = f" {text}"
        self._emitted_any = True
        self._utterance_has_output = True
        return text
