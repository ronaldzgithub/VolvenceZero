"""Wikisource archive adapter.

Wikisource is a multilingual (en / de / fr / zh / ...) project
hosting public-domain primary sources. The URL pattern is::

    https://{lang}.wikisource.org/wiki/{Title}

The cleaned body is plain text (the curator strips MediaWiki
template artifacts). Pages can be papers, lectures, books, or
anything in between; this adapter exposes both
``wikisource_to_paper_source`` (for written works) and
``wikisource_to_lecture_source`` (for transcribed addresses).
"""

from __future__ import annotations

from dataclasses import dataclass

from lifeform_domain_figure.corpus.ingest_lectures import FigureLectureSource
from lifeform_domain_figure.corpus.ingest_papers import FigurePaperSource


@dataclass(frozen=True)
class WikisourcePayload:
    """Pre-downloaded Wikisource page payload."""

    page_title: str
    language: str
    source_url: str
    body: str
    year: int | None = None
    author_id: str = ""
    venue_id: str = ""
    date_iso: str = ""
    audience: str = ""

    def __post_init__(self) -> None:
        for name in ("page_title", "language", "source_url", "body"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"WikisourcePayload.{name} must be non-empty for "
                    f"page_title={self.page_title!r}"
                )
        if self.year is not None and not (-3000 <= self.year <= 9999):
            raise ValueError(
                f"WikisourcePayload.year out of plausible range: {self.year!r}"
            )


def _stable_id(prefix: str, payload: WikisourcePayload) -> str:
    slug = (
        payload.page_title.strip()
        .replace(" ", "_")
        .replace("/", "-")
        .lower()
    )
    return f"{prefix}:wikisource:{payload.language}:{slug}"


def wikisource_to_paper_source(
    payload: WikisourcePayload,
    *,
    figure_id: str,
) -> FigurePaperSource:
    """Translate a Wikisource page into a :class:`FigurePaperSource`."""

    publication_locator = f"wikisource:{payload.language}:{payload.page_title}"
    return FigurePaperSource(
        paper_id=_stable_id("paper", payload),
        title=payload.page_title,
        year=payload.year if payload.year is not None else 0,
        language=payload.language,
        body=payload.body,
        publication_locator=publication_locator,
        figure_id=figure_id,
    )


def wikisource_to_lecture_source(
    payload: WikisourcePayload,
    *,
    figure_id: str,
) -> FigureLectureSource:
    """Translate a Wikisource page into a :class:`FigureLectureSource`.

    Used when the Wikisource page is a transcribed lecture / address.
    The ``venue_id`` and ``date_iso`` fields are mandatory for
    lectures; the helper refuses payloads missing either.
    """

    if not payload.venue_id.strip():
        raise ValueError(
            f"wikisource_to_lecture_source: payload.venue_id must be set "
            f"for lecture pages (page_title={payload.page_title!r})"
        )
    return FigureLectureSource(
        lecture_id=_stable_id("lecture", payload),
        venue_id=payload.venue_id,
        date_iso=payload.date_iso,
        audience=payload.audience or "general",
        language=payload.language,
        body=payload.body,
        figure_id=figure_id,
    )


__all__ = [
    "WikisourcePayload",
    "wikisource_to_lecture_source",
    "wikisource_to_paper_source",
]
