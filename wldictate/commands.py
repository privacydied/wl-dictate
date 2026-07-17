"""LLM-free voice edit commands.

When a finalized utterance is EXACTLY one of these phrases (after
normalization), it is executed as an edit instead of being typed/transformed:

- "scratch that" — delete the previous utterance (and the spoken command)
- "new line"     — replace the spoken command with a line break
- "press enter"  — remove the spoken command and hit Return

Deliberately exact-match: fuzzy matching would eat real dictation.
"""

from __future__ import annotations

import re

_RE_NORMALIZE = re.compile(r"[^a-z ]+")

#: normalized phrase -> action id
_PHRASES = {
    "scratch that": "scratch",
    "delete that": "scratch",
    "undo that": "scratch",
    "new line": "newline",
    "newline": "newline",
    "press enter": "enter",
    "hit enter": "enter",
    "press tab": "tab",
    "press escape": "escape",
    "copy that": "copy",
}


def match_command(text: str) -> str | None:
    """Action id when ``text`` is exactly a voice command, else None."""
    normalized = _RE_NORMALIZE.sub("", text.lower())
    normalized = " ".join(normalized.split())
    return _PHRASES.get(normalized)


_RE_LITERAL = re.compile(r"^(\s*)[Ll]iterally[,.]?\s*(\S.*)$", re.DOTALL)


def strip_literal(text: str) -> str | None:
    """Verbatim escape: an utterance starting with "literally" bypasses the
    contextual transform — return the text with the guard word removed
    (leading separator preserved), or None when the guard isn't present.
    """
    m = _RE_LITERAL.match(text)
    if not m or not re.search(r"\w", m.group(2)):
        return None  # no real content after the guard word
    return m.group(1) + m.group(2)
