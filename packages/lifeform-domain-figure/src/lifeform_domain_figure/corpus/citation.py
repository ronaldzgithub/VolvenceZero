"""Typed parser for the four citation-quality locator string formats.

Each :mod:`lifeform_domain_figure.corpus.ingest_*` adapter writes a
locator string of the form::

    paper:{paper_id}:lang={lang}:para={i}:offset={s}-{e}
    letter:{sender_id}-to-{recipient_id}:date={date}:lang={lang}:para={i}:offset={s}-{e}
    lecture:{lecture_id}:venue={venue_id}:date={date}:lang={lang}:para={i}:offset={s}-{e}
    notebook:{notebook_id}:vol={volume}:page={page}:lang={lang}:para={i}:offset={s}-{e}

The runtime ``GroundedDecoder`` (P3.1) only needs the locator as an
opaque string — it surfaces the locator verbatim in evidence
pointers and never branches on its parts. **Reviewer-facing**
tooling, however, wants to extract structured fields ("show me every
locator from a 1935-04-12 letter") without re-parsing the format
inline at every call site.

This module gives them one typed parser. The parser is **strict** —
unknown prefix or missing required key fails loudly per
``no-swallow-errors-no-hasattr-abuse.mdc``; there is no silent
"return None" path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class LocatorKind(str, Enum):
    """The four citation locator kinds the figure vertical produces."""

    PAPER = "paper"
    LETTER = "letter"
    LECTURE = "lecture"
    NOTEBOOK = "notebook"


@dataclass(frozen=True)
class LocatorOffset:
    """The ``offset=<start>-<end>`` segment of every locator."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(
                f"LocatorOffset.start must be >= 0, got {self.start!r}"
            )
        if self.end < self.start:
            raise ValueError(
                f"LocatorOffset.end ({self.end!r}) must be >= start "
                f"({self.start!r})"
            )


@dataclass(frozen=True)
class ParsedLocator:
    """Typed view onto a parsed citation locator.

    ``raw`` keeps the original string so a consumer that wants to
    surface the citation verbatim does not need to re-render it.

    ``extras`` carries any kind-specific keys not exposed as named
    fields here; that keeps the parser future-proof against new
    locator components landing in later packets without breaking
    today's call sites.
    """

    raw: str
    kind: LocatorKind
    document_id: str
    paragraph_index: int
    offset: LocatorOffset
    language: str = ""
    sender_id: str = ""
    recipient_id: str = ""
    date_iso: str = ""
    venue_id: str = ""
    volume: str = ""
    page: int = -1
    extras: tuple[tuple[str, str], ...] = ()


_PAIR_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*=.+$")


def _split_kv(part: str) -> tuple[str, str]:
    if "=" not in part:
        raise ValueError(
            f"parse_locator: expected 'key=value' segment but got {part!r}"
        )
    key, value = part.split("=", 1)
    if not key or not value:
        raise ValueError(
            f"parse_locator: empty key or value in segment {part!r}"
        )
    return key, value


def _parse_offset(value: str) -> LocatorOffset:
    if "-" not in value:
        raise ValueError(
            f"parse_locator: offset must look like 'start-end', got {value!r}"
        )
    start_str, end_str = value.split("-", 1)
    return LocatorOffset(start=int(start_str), end=int(end_str))


def _parse_letter_thread(thread: str) -> tuple[str, str]:
    """Split ``einstein-to-bohr`` into ``("einstein", "bohr")``."""
    if "-to-" not in thread:
        raise ValueError(
            f"parse_locator: letter thread must contain '-to-', got "
            f"{thread!r}"
        )
    sender, recipient = thread.split("-to-", 1)
    if not sender or not recipient:
        raise ValueError(
            f"parse_locator: letter thread has empty sender or recipient: "
            f"{thread!r}"
        )
    return sender, recipient


def parse_locator(locator: str) -> ParsedLocator:
    """Parse a citation locator into a typed :class:`ParsedLocator` record.

    Strict by design: unknown prefix or missing required field raises
    ``ValueError`` with the bad locator quoted in the message.
    """

    if not locator or not locator.strip():
        raise ValueError("parse_locator: empty locator")
    parts = locator.split(":")
    if len(parts) < 3:
        raise ValueError(
            f"parse_locator: locator must have at least 3 colon-separated "
            f"segments, got {locator!r}"
        )
    head = parts[0]
    if head not in {kind.value for kind in LocatorKind}:
        raise ValueError(
            f"parse_locator: unknown locator kind {head!r} in {locator!r}; "
            f"expected one of {[k.value for k in LocatorKind]}"
        )
    kind = LocatorKind(head)
    document_id_or_thread = parts[1]
    if not document_id_or_thread:
        raise ValueError(
            f"parse_locator: empty document id segment in {locator!r}"
        )
    kv_segments = parts[2:]
    kv_map: dict[str, str] = {}
    extras: list[tuple[str, str]] = []
    for segment in kv_segments:
        if not _PAIR_RE.match(segment):
            raise ValueError(
                f"parse_locator: malformed segment {segment!r} in {locator!r}; "
                f"expected 'key=value'"
            )
        key, value = _split_kv(segment)
        if key in kv_map:
            raise ValueError(
                f"parse_locator: duplicate key {key!r} in {locator!r}"
            )
        kv_map[key] = value
    for required in ("para", "offset"):
        if required not in kv_map:
            raise ValueError(
                f"parse_locator: missing required key {required!r} in "
                f"{locator!r}"
            )
    paragraph_index = int(kv_map.pop("para"))
    offset = _parse_offset(kv_map.pop("offset"))
    language = kv_map.pop("lang", "")
    sender_id = ""
    recipient_id = ""
    date_iso = ""
    venue_id = ""
    volume = ""
    page = -1
    document_id = document_id_or_thread
    if kind is LocatorKind.LETTER:
        sender_id, recipient_id = _parse_letter_thread(document_id_or_thread)
        date_iso = kv_map.pop("date", "")
        if not date_iso:
            raise ValueError(
                f"parse_locator: letter locator missing 'date' key in "
                f"{locator!r}"
            )
        document_id = document_id_or_thread
    elif kind is LocatorKind.LECTURE:
        venue_id = kv_map.pop("venue", "")
        date_iso = kv_map.pop("date", "")
        if not venue_id:
            raise ValueError(
                f"parse_locator: lecture locator missing 'venue' key in "
                f"{locator!r}"
            )
    elif kind is LocatorKind.NOTEBOOK:
        volume = kv_map.pop("vol", "")
        page_str = kv_map.pop("page", "")
        if not volume or not page_str:
            raise ValueError(
                f"parse_locator: notebook locator must declare both 'vol' "
                f"and 'page' in {locator!r}"
            )
        page = int(page_str)
    for leftover_key, leftover_value in sorted(kv_map.items()):
        extras.append((leftover_key, leftover_value))
    return ParsedLocator(
        raw=locator,
        kind=kind,
        document_id=document_id,
        paragraph_index=paragraph_index,
        offset=offset,
        language=language,
        sender_id=sender_id,
        recipient_id=recipient_id,
        date_iso=date_iso,
        venue_id=venue_id,
        volume=volume,
        page=page,
        extras=tuple(extras),
    )


__all__ = [
    "LocatorKind",
    "LocatorOffset",
    "ParsedLocator",
    "parse_locator",
]
