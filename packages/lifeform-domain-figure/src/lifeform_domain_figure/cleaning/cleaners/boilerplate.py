"""Boilerplate stripping cleaner.

Removes structural noise that survives parsing but does not belong to
the document body:

* Bare page-number lines (``"   12  "``) common in OCR'd facsimiles
* Repeated running heads of the form ``"Vol. <N>, p. <M>"`` or
  ``"<TITLE> -- Page <N>"``
* Form-feed characters (used by parsers as page separators)
* CPAE editorial brackets like ``"[p. 23]"`` left dangling on their
  own line

The cleaner returns the cleaned text; the orchestrator
(:mod:`lifeform_domain_figure.cleaning.cleaners`) records the
:class:`CleaningOpRecord`.
"""

from __future__ import annotations

import re

OP_VERSION = "1"

_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")
_RUNNING_HEAD_RE = re.compile(
    r"^\s*(?:Vol\.\s*\d+,\s*p\.\s*\d+|Page\s+\d+|p\.\s*\d+)\s*$",
    re.IGNORECASE,
)
_INLINE_PAGE_MARKER_RE = re.compile(r"^\s*\[\s*p\.\s*\d+\s*\]\s*$", re.IGNORECASE)


def strip_boilerplate(text: str) -> str:
    """Strip page numbers / running heads / page-marker lines / form-feeds."""

    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        plain = line.replace("\f", "")
        if _PAGE_NUMBER_RE.match(plain):
            continue
        if _RUNNING_HEAD_RE.match(plain):
            continue
        if _INLINE_PAGE_MARKER_RE.match(plain):
            continue
        cleaned_lines.append(plain)
    return "\n".join(cleaned_lines)


__all__ = ["OP_VERSION", "strip_boilerplate"]
