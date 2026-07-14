"""Incremental transcript cleanup and spacing.

Ported from the batch regex cleanup in the old ``whisper_dictate.py``, reworked
to operate on per-commit deltas: the formatter owns all spacing state so text
joins correctly across commits and across utterances.

Spacing is decided from *evidence*, not guessed from character classes:
Whisper's word tokens carry their own leading space when they start a new word
(" hello") and don't when they continue one ("times" after " some"). The
formatter honors that, tracks the last emitted character, and never emits two
adjacent spaces or a space at the start of a session.
"""

from __future__ import annotations

import re

# Pre-compiled cleanup regexes (hot path).
_RE_PARENS = re.compile(r"\([^)]*\)\s*")
_RE_BRACKETS = re.compile(r"\[[^\]]*\]\s*")
_RE_DOTS = re.compile(r"(?:\s*\.\s*){2,}")
_RE_SENTENCE_PERIOD = re.compile(r"(?<!\.)\.(\s)?(?![.\"'’)\]}])")
_RE_SENTENCE_PUNCT = re.compile(r"([!?;:])(\s)?(?![\"'’)\]}])")
_RE_COMMA_SPACE = re.compile(r",(\s)?")
_RE_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,!?;:])(?!\.)")
_RE_WHITESPACE = re.compile(r"\s+")
_RE_LEADING_PUNCT_SPACE = re.compile(
    r"^[.,!?;:](?![.,!?;:])[\s\u00A0\u200B\u200C\u200D\u2060]*"
)

# Characters that attach to the preceding text (never preceded by a space).
_BINDS_LEFT = set(".,!?;:%'’\")]}…")
# The delta ends a sentence (possibly inside closing quotes/brackets).
_RE_ENDS_SENTENCE = re.compile(r"[.!?][\"'’)\]}]*$")


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
    - never a leading space on the first output of a session
    - exactly one space between words, across commit and utterance boundaries
    - no space before punctuation or word continuations; never a double space
    - a single trailing space after sentence-final punctuation when an
      utterance ends (so manual typing can continue naturally)
    - annotations like ``(coughs)`` / ``[music]`` are stripped

    Create one instance per dictation session: spacing context must not leak
    across toggles (the cursor may have moved anywhere in between).
    """

    def __init__(self, *, sentence_trailing_space: bool = True) -> None:
        self._sentence_trailing_space = sentence_trailing_space
        self._tail: str = ""  # last few emitted chars; "" = nothing emitted yet
        self._utterance_has_output = False

    def on_utterance_start(self) -> None:
        self._utterance_has_output = False

    def format_delta(self, raw: str) -> str:
        """Format one commit delta; returns "" if nothing remains after cleanup."""
        # Whisper convention: a token starting a new word carries leading
        # whitespace; a continuation token does not. Capture before cleanup.
        starts_new_word = raw[:1].isspace()
        first_of_utterance = not self._utterance_has_output

        text = clean_text(raw, utterance_start=first_of_utterance)
        if not text:
            return ""

        if not self._tail or self._tail[-1].isspace():
            needs_separator = False  # nothing emitted yet, or already spaced
        elif starts_new_word:
            # The token carried its own leading space: it starts a new word
            # (or an opening quote) — always separate.
            needs_separator = True
        elif first_of_utterance:
            # New utterance whose first token lacks a leading space: separate
            # unless it begins with punctuation that attaches left.
            needs_separator = text[0] not in _BINDS_LEFT
        else:
            # Mid-utterance continuation token ("times" after " some") or
            # attached punctuation (", world"): glue it on.
            needs_separator = False
        if needs_separator:
            text = f" {text}"

        self._utterance_has_output = True
        self._tail = (self._tail + text)[-8:]
        return text

    def end_utterance(self) -> str:
        """Trailing output for a finished utterance ("" or a single space).

        Emitted after sentence-final punctuation so the text field is left
        ready for continued typing; the tracked state guarantees the next
        utterance never double-spaces after it.
        """
        if (
            self._sentence_trailing_space
            and self._utterance_has_output
            and self._tail
            and not self._tail[-1].isspace()
            and _RE_ENDS_SENTENCE.search(self._tail)
        ):
            self._tail = (self._tail + " ")[-8:]
            return " "
        return ""
