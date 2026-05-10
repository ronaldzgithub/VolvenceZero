"""Whitespace normalisation cleaner.

Collapses redundant intra-line whitespace and excessive blank lines.

* Multiple spaces / tabs collapse to a single space (preserving leading
  paragraph indentation rules is the paragraph cleaner's job).
* Three or more consecutive newlines collapse to two (paragraph
  separator).
* Trailing whitespace per line is stripped.
* Leading / trailing whitespace on the whole document is stripped.

The cleaner is monotonically non-expanding (the orchestrator's
:class:`CleaningOpRecord` invariant).
"""

from __future__ import annotations

import re

OP_VERSION = "1"

_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_INTRA_LINE_WS_RE = re.compile(r"[ \t]{2,}")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def normalise_whitespace(text: str) -> str:
    text = _TRAILING_WS_RE.sub("", text)
    text = _INTRA_LINE_WS_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


__all__ = ["OP_VERSION", "normalise_whitespace"]
