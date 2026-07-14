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

# Punctuation after which a following letter/digit must be spaced off (a word
# glued directly onto sentence-final punctuation, e.g. "talk.And").
_SPACE_AFTER = set(".,!?;:…")


def _capitalize_first_alpha(text: str) -> str:
    """Upper-case the first alphabetic character, leaving leading spaces/quotes.

    " and then" -> " And then";  '"stop' -> '"Stop';  "42 apples" left as-is
    (leading char is a digit, not alphabetic — nothing to capitalize until a
    letter, so we uppercase the first letter we find).
    """
    for i, ch in enumerate(text):
        if ch.isalpha():
            if ch.upper() == ch:
                return text  # already capital (or non-cased script)
            return text[:i] + ch.upper() + text[i + 1 :]
    return text


def _joins_words(prev: str, curr: str) -> bool:
    """Would placing ``curr`` right after ``prev`` fuse two separate words?

    True when both sides are word-forming (letter/digit) — "talkAnd" — or when
    ``prev`` is sentence/clause punctuation immediately followed by a
    word-forming ``curr`` — "talk.And". Used as a defensive separator when
    Whisper's leading-space hint on a delta was lost upstream.
    """
    if not curr.isalnum():
        return False
    return prev.isalnum() or prev in _SPACE_AFTER


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

    def __init__(
        self,
        *,
        sentence_trailing_space: bool = True,
        capitalize_sentences: bool = True,
    ) -> None:
        self._sentence_trailing_space = sentence_trailing_space
        self._capitalize = capitalize_sentences
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
        elif not text[0].isspace() and text[0] not in _BINDS_LEFT and _joins_words(
            self._tail[-1], text[0]
        ):
            # Safety net: the delta lacks a leading space (Whisper's hint was
            # lost — common on fast speech / distil models, and across commit
            # boundaries), but gluing here would fuse two words like
            # "talk.And" or "thissucks". If the boundary sits between two
            # word-forming characters (or right after sentence punctuation),
            # force a separator regardless of the missing hint. Genuine
            # continuation tokens ("times" after " some") don't cross a commit
            # boundary in LocalAgreement streaming, so this can't wrongly split
            # a real word.
            needs_separator = True
        else:
            # Mid-utterance continuation token ("times" after " some") or
            # attached punctuation (", world"): glue it on.
            needs_separator = False
        if needs_separator:
            text = f" {text}"

        if self._capitalize and self._at_sentence_start():
            text = _capitalize_first_alpha(text)

        self._utterance_has_output = True
        self._tail = (self._tail + text)[-8:]
        return text

    def _at_sentence_start(self) -> bool:
        """True when the next word begins a sentence.

        That's either the very first output of the session, or a point where the
        emitted tail ends with sentence-final punctuation (optionally followed by
        whitespace / closing quotes). Whisper often lowercases the word after a
        pause ("... talk. and then ..."), so we fix the casing here.
        """
        tail = self._tail.rstrip()
        if not tail:
            return True
        return bool(_RE_ENDS_SENTENCE.search(tail))

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
