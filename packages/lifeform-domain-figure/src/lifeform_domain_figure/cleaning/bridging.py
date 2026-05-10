"""Bridge cleaned text into the existing archive ``*Payload`` schemas.

The L1 cleaning pipeline produces a neutral
:class:`CleanedDocument`. Crossing into a figure-vertical typed
source record (``FigurePaperSource`` / ``FigureLetterSource`` / etc.)
is intentionally a TWO-step process to keep cleaner / parser
decoupled from typed source choice (R8):

1. ``cleaned_to_*_payload(...)`` — pure function in this module — wraps
   the cleaned text in the existing archive payload (``CPAEPayload``
   / ``WikisourcePayload`` / ``GutenbergPayload`` /
   ``InternetArchivePayload``). The curator / caller still supplies
   archive-specific metadata (``volume`` / ``document_number`` /
   ``page_title`` / etc.) that the parser cannot know.
2. ``<archive>_to_<source_kind>_source(...)`` — already existing in
   :mod:`lifeform_domain_figure.corpus.archives` — translates the
   payload into the typed ``Figure*Source``.

The bridging functions are pure and own no state. They never touch
the filesystem and never touch the cleaning store; the call site is
expected to read the cleaned text from the store first (via
``CleaningStore.get_cleaned``) when they need persistence.
"""

from __future__ import annotations

from lifeform_domain_figure.cleaning.raw_document import CleanedDocument
from lifeform_domain_figure.corpus.archives.cpae import (
    CPAEDocumentKind,
    CPAEPayload,
)
from lifeform_domain_figure.corpus.archives.gutenberg import GutenbergPayload
from lifeform_domain_figure.corpus.archives.internet_archive import (
    InternetArchivePayload,
)
from lifeform_domain_figure.corpus.archives.wikisource import WikisourcePayload


def cleaned_to_cpae_payload(
    cleaned: CleanedDocument,
    *,
    document_id: str,
    document_kind: CPAEDocumentKind,
    volume: int,
    document_number: int,
    title: str,
    year: int,
    language: str,
    source_url: str,
    sender_id: str = "",
    recipient_id: str = "",
    date_iso: str = "",
) -> CPAEPayload:
    """Wrap a :class:`CleanedDocument` as a :class:`CPAEPayload`.

    The cleaned ``text`` is the payload's ``body``. All
    archive-specific structural metadata (volume / document_number /
    title / year / language / sender / recipient / date) must be
    supplied by the curator: the cleaning pipeline does not infer
    them and the parser only sees raw bytes.
    """

    return CPAEPayload(
        document_id=document_id,
        document_kind=document_kind,
        volume=volume,
        document_number=document_number,
        title=title,
        year=year,
        language=language,
        body=cleaned.text,
        source_url=source_url,
        sender_id=sender_id,
        recipient_id=recipient_id,
        date_iso=date_iso,
    )


def cleaned_to_wikisource_payload(
    cleaned: CleanedDocument,
    *,
    page_title: str,
    language: str,
    source_url: str,
    year: int | None = None,
    author_id: str = "",
    venue_id: str = "",
    date_iso: str = "",
    audience: str = "",
) -> WikisourcePayload:
    """Wrap a :class:`CleanedDocument` as a :class:`WikisourcePayload`."""

    return WikisourcePayload(
        page_title=page_title,
        language=language,
        source_url=source_url,
        body=cleaned.text,
        year=year,
        author_id=author_id,
        venue_id=venue_id,
        date_iso=date_iso,
        audience=audience,
    )


def cleaned_to_gutenberg_payload(
    cleaned: CleanedDocument,
    *,
    ebook_id: int,
    title: str,
    language: str,
    source_url: str,
    section_label: str = "",
    year: int | None = None,
    author_id: str = "",
) -> GutenbergPayload:
    """Wrap a :class:`CleanedDocument` as a :class:`GutenbergPayload`."""

    return GutenbergPayload(
        ebook_id=ebook_id,
        title=title,
        language=language,
        body=cleaned.text,
        source_url=source_url,
        section_label=section_label,
        year=year,
        author_id=author_id,
    )


def cleaned_to_internet_archive_payload(
    cleaned: CleanedDocument,
    *,
    identifier: str,
    title: str,
    language: str,
    source_url: str,
    creator_id: str = "",
    year: int | None = None,
    venue_id: str = "",
    date_iso: str = "",
    audience: str = "",
) -> InternetArchivePayload:
    """Wrap a :class:`CleanedDocument` as an :class:`InternetArchivePayload`."""

    return InternetArchivePayload(
        identifier=identifier,
        title=title,
        language=language,
        body=cleaned.text,
        source_url=source_url,
        creator_id=creator_id,
        year=year,
        venue_id=venue_id,
        date_iso=date_iso,
        audience=audience,
    )


__all__ = [
    "cleaned_to_cpae_payload",
    "cleaned_to_gutenberg_payload",
    "cleaned_to_internet_archive_payload",
    "cleaned_to_wikisource_payload",
]
