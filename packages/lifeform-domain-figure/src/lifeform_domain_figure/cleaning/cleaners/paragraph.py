"""Paragraph normalisation cleaner.

After whitespace + boilerplate strip, paragraph boundaries can still
be lossy in OCR'd documents:

* Lines wrapped at fixed widths (a single paragraph spans 6 lines, no
  blank line between them) are re-flowed into one line per paragraph.
  We use the heuristic: if line N ends with a sentence terminator
  (``. ! ? : ;``) or a closing quote, the next line starts a new
  intra-paragraph segment; otherwise the next line is a continuation
  and joins with a space.
* Empty paragraphs are dropped.
* Paragraphs are joined with ``\n\n`` (single blank line separator).

The cleaner is monotonically non-expanding: collapsing intra-paragraph
newlines to spaces is length-preserving, and dropping empty paragraphs
is length-reducing.
"""

from __future__ import annotations

import re

OP_VERSION = "1"

_PARAGRAPH_SPLIT_RE = re.compile(r"\n{2,}")
_LINE_END_TERMINATORS = (".", "!", "?", ":", ";", '"', "'", ")", "]", "}")


def _reflow_paragraph(paragraph: str) -> str:
    lines = [line.rstrip() for line in paragraph.split("\n")]
    lines = [line for line in lines if line.strip()]
    if not lines:
        return ""
    rebuilt: list[str] = []
    buffer = lines[0]
    for line in lines[1:]:
        if buffer.endswith(_LINE_END_TERMINATORS):
            rebuilt.append(buffer)
            buffer = line
        else:
            buffer = f"{buffer} {line.lstrip()}"
    rebuilt.append(buffer)
    return "\n".join(rebuilt)


def normalise_paragraphs(text: str) -> str:
    paragraphs = _PARAGRAPH_SPLIT_RE.split(text)
    rebuilt = [_reflow_paragraph(p) for p in paragraphs]
    rebuilt = [p for p in rebuilt if p]
    return "\n\n".join(rebuilt)


__all__ = ["OP_VERSION", "normalise_paragraphs"]
