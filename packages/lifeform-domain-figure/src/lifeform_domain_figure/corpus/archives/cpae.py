"""Princeton Collected Papers of Albert Einstein (CPAE) archive adapter.

CPAE is the canonical source for Einstein's writings. The Princeton
URL pattern is::

    https://einsteinpapers.press.princeton.edu/vol{N}-doc/{ID}

Documents come in several kinds (papers, letters, notes, articles).
This adapter normalises the typed CPAE payload into a
:class:`FigurePaperSource` (for scientific papers / articles) or a
:class:`FigureLetterSource` (for correspondence).

V1 takes a pre-downloaded :class:`CPAEPayload` and returns the typed
source record. V2 will add an HTTP fetcher behind the
:class:`ArchiveFetcher` Protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from lifeform_domain_figure.corpus.ingest_letters import FigureLetterSource
from lifeform_domain_figure.corpus.ingest_papers import FigurePaperSource


class CPAEDocumentKind(str, Enum):
    """The CPAE document classification.

    ``ARTICLE``  — published paper / book chapter / public article.
    ``LETTER``   — correspondence; sender / recipient mandatory.
    ``NOTE``     — short editorial note, treated as a paper for
                   ingestion purposes (mid-density text).
    """

    ARTICLE = "article"
    LETTER = "letter"
    NOTE = "note"


@dataclass(frozen=True)
class CPAEPayload:
    """Typed pre-downloaded payload for one CPAE document.

    All fields are reviewer-supplied. ``body`` is the cleaned full
    text (no facsimile boilerplate). ``volume`` / ``document_number``
    / ``language`` / ``year`` come from the CPAE table of contents
    and are mandatory: the citation locator format requires them.
    """

    document_id: str  # canonical CPAE id, e.g., "cpae-vol2-doc24"
    document_kind: CPAEDocumentKind
    volume: int
    document_number: int
    title: str
    year: int
    language: str
    body: str
    source_url: str
    sender_id: str = ""  # required iff kind is LETTER
    recipient_id: str = ""  # required iff kind is LETTER
    date_iso: str = ""  # used by LETTER

    def __post_init__(self) -> None:
        for name in (
            "document_id",
            "title",
            "language",
            "body",
            "source_url",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"CPAEPayload.{name} must be non-empty for "
                    f"document_id={self.document_id!r}"
                )
        if self.volume <= 0:
            raise ValueError(
                f"CPAEPayload.volume must be > 0 for {self.document_id!r}"
            )
        if self.document_number <= 0:
            raise ValueError(
                f"CPAEPayload.document_number must be > 0 for "
                f"{self.document_id!r}"
            )
        if self.document_kind is CPAEDocumentKind.LETTER:
            if not self.sender_id.strip() or not self.recipient_id.strip():
                raise ValueError(
                    f"CPAEPayload.kind=LETTER requires sender_id and "
                    f"recipient_id (document_id={self.document_id!r})"
                )


def cpae_to_paper_source(
    payload: CPAEPayload,
    *,
    figure_id: str,
) -> FigurePaperSource:
    """Translate a CPAE article / note payload into a :class:`FigurePaperSource`.

    Refuses LETTER payloads — those should go through
    :func:`cpae_to_letter_source` so the citation locator carries
    the ``letter:sender-to-recipient:date=...`` shape.
    """

    if payload.document_kind is CPAEDocumentKind.LETTER:
        raise ValueError(
            f"cpae_to_paper_source: refuse to coerce a LETTER payload into "
            f"a paper source (document_id={payload.document_id!r}); use "
            f"cpae_to_letter_source instead."
        )
    publication_locator = (
        f"cpae:vol={payload.volume}:doc={payload.document_number}"
    )
    return FigurePaperSource(
        paper_id=payload.document_id,
        title=payload.title,
        year=payload.year,
        language=payload.language,
        body=payload.body,
        publication_locator=publication_locator,
        figure_id=figure_id,
    )


def cpae_to_letter_source(
    payload: CPAEPayload,
    *,
    figure_id: str,
    in_reply_to: str = "",
) -> FigureLetterSource:
    """Translate a CPAE letter payload into a :class:`FigureLetterSource`."""

    if payload.document_kind is not CPAEDocumentKind.LETTER:
        raise ValueError(
            f"cpae_to_letter_source: payload kind must be LETTER, got "
            f"{payload.document_kind.value!r} for "
            f"document_id={payload.document_id!r}"
        )
    return FigureLetterSource(
        letter_id=payload.document_id,
        sender_id=payload.sender_id,
        recipient_id=payload.recipient_id,
        date_iso=payload.date_iso,
        language=payload.language,
        body=payload.body,
        in_reply_to=in_reply_to,
        figure_id=figure_id,
    )


__all__ = [
    "CPAEDocumentKind",
    "CPAEPayload",
    "cpae_to_letter_source",
    "cpae_to_paper_source",
]
