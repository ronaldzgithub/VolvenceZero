"""Intra-document dedupe cleaner.

OCR'd documents commonly emit the same paragraph twice (a
double-scanned page, a header repeated on every page sneaking past
the boilerplate stripper, etc.). This cleaner deduplicates at the
paragraph level (paragraphs separated by blank lines), keeping the
first occurrence and dropping subsequent verbatim repeats.

Short paragraphs (< ``MIN_PARAGRAPH_LEN`` chars) are kept as-is to
avoid clobbering legitimately-recurring single-line markers like
chapter headings. The dedupe is exact-match on stripped text.
"""

from __future__ import annotations

OP_VERSION = "1"

MIN_PARAGRAPH_LEN = 80


def dedupe_paragraphs(text: str) -> str:
    paragraphs = text.split("\n\n")
    seen: set[str] = set()
    kept: list[str] = []
    for paragraph in paragraphs:
        candidate = paragraph.strip()
        if not candidate:
            continue
        if len(candidate) < MIN_PARAGRAPH_LEN:
            kept.append(paragraph)
            continue
        key = candidate
        if key in seen:
            continue
        seen.add(key)
        kept.append(paragraph)
    return "\n\n".join(kept)


__all__ = ["MIN_PARAGRAPH_LEN", "OP_VERSION", "dedupe_paragraphs"]
