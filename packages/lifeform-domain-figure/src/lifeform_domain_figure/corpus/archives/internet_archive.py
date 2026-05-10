"""Internet Archive adapter.

Internet Archive (archive.org) hosts scanned books, journals,
ephemera, and audio recordings of lectures. The URL pattern is::

    https://archive.org/details/{identifier}

This adapter handles the two most common kinds the figure vertical
ingests from IA:

* Scanned papers / books → :class:`FigurePaperSource`
* Recorded lectures (transcript form) → :class:`FigureLectureSource`

The IA OCR text quality varies; the adapter does **not** attempt
post-OCR cleaning (that is the curator's job, recorded in
:class:`SourceProvenance.capture_method`).
"""

from __future__ import annotations

from dataclasses import dataclass

from lifeform_domain_figure.corpus.ingest_lectures import FigureLectureSource
from lifeform_domain_figure.corpus.ingest_papers import FigurePaperSource


@dataclass(frozen=True)
class InternetArchivePayload:
    """Pre-downloaded Internet Archive item payload."""

    identifier: str
    title: str
    language: str
    body: str
    source_url: str
    creator_id: str = ""
    year: int | None = None
    venue_id: str = ""
    date_iso: str = ""
    audience: str = ""

    def __post_init__(self) -> None:
        for name in ("identifier", "title", "language", "body", "source_url"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"InternetArchivePayload.{name} must be non-empty for "
                    f"identifier={self.identifier!r}"
                )
        if self.year is not None and not (-3000 <= self.year <= 9999):
            raise ValueError(
                f"InternetArchivePayload.year out of plausible range: "
                f"{self.year!r}"
            )


def internet_archive_to_paper_source(
    payload: InternetArchivePayload,
    *,
    figure_id: str,
) -> FigurePaperSource:
    """Translate an IA item into a paper source."""

    publication_locator = f"internet-archive:identifier={payload.identifier}"
    return FigurePaperSource(
        paper_id=f"ia:{payload.identifier}",
        title=payload.title,
        year=payload.year if payload.year is not None else 0,
        language=payload.language,
        body=payload.body,
        publication_locator=publication_locator,
        figure_id=figure_id,
    )


def internet_archive_to_lecture_source(
    payload: InternetArchivePayload,
    *,
    figure_id: str,
) -> FigureLectureSource:
    """Translate an IA item into a lecture source.

    Refuses payloads missing ``venue_id`` or ``date_iso``: lectures
    must declare both so the citation locator carries traceable
    venue / date metadata.
    """

    if not payload.venue_id.strip():
        raise ValueError(
            f"internet_archive_to_lecture_source: payload.venue_id must be "
            f"set for lecture items (identifier={payload.identifier!r})"
        )
    return FigureLectureSource(
        lecture_id=f"ia:{payload.identifier}",
        venue_id=payload.venue_id,
        date_iso=payload.date_iso,
        audience=payload.audience or "general",
        language=payload.language,
        body=payload.body,
        figure_id=figure_id,
    )


__all__ = [
    "InternetArchivePayload",
    "internet_archive_to_lecture_source",
    "internet_archive_to_paper_source",
]
