"""Chinese Text Project (CTP) archive adapter.

CTP (https://ctext.org/) hosts pre-Qin / classical Chinese texts (the
Analects, the Daodejing, etc.) in collated, page-addressable form.
The figure vertical uses CTP as the canonical source for classical
Chinese figures (Confucius, Mencius, Laozi, ...). The URL pattern
is::

    https://ctext.org/{collection}/{chapter}#{section_id}

Each CTP page is one chapter; the curator pre-downloads the cleaned
text and constructs a :class:`CTPPayload`. The adapter normalises
the payload into a :class:`FigurePaperSource` (CTP texts are
written prose / aphoristic compilations — not letters or lectures).

This adapter is the V1 entry point for **classical Chinese
figures**; modern Chinese figures (e.g., 鲁迅, born 1881) are
better served by the existing
:mod:`lifeform_domain_figure.corpus.archives.wikisource` adapter
pointing at ``zh.wikisource.org``.
"""

from __future__ import annotations

from dataclasses import dataclass

from lifeform_domain_figure.corpus.ingest_papers import FigurePaperSource


@dataclass(frozen=True)
class CTPPayload:
    """Pre-downloaded Chinese Text Project chapter payload."""

    collection: str  # e.g., "analects"
    chapter_id: str  # e.g., "xueer"
    title: str  # e.g., "論語 · 學而"
    body: str
    source_url: str
    section_id: str = ""  # optional section id within the chapter
    estimated_year: int | None = None  # rough composition year (BCE / CE)
    language: str = "zh-Hant"  # default: traditional classical Chinese

    def __post_init__(self) -> None:
        for name in ("collection", "chapter_id", "title", "body", "source_url", "language"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"CTPPayload.{name} must be non-empty for "
                    f"chapter_id={self.chapter_id!r}"
                )
        if self.estimated_year is not None and not (
            -3000 <= self.estimated_year <= 9999
        ):
            raise ValueError(
                f"CTPPayload.estimated_year out of plausible range: "
                f"{self.estimated_year!r}"
            )


def ctp_to_paper_source(
    payload: CTPPayload,
    *,
    figure_id: str,
) -> FigurePaperSource:
    """Translate a CTP chapter payload into a :class:`FigurePaperSource`."""

    section_suffix = f":section={payload.section_id}" if payload.section_id else ""
    publication_locator = (
        f"ctext:{payload.collection}:{payload.chapter_id}{section_suffix}"
    )
    section_paper_suffix = (
        f":{payload.section_id}" if payload.section_id else ""
    )
    paper_id = (
        f"ctext:{payload.collection}:{payload.chapter_id}{section_paper_suffix}"
    )
    return FigurePaperSource(
        paper_id=paper_id,
        title=payload.title,
        year=payload.estimated_year if payload.estimated_year is not None else 0,
        language=payload.language,
        body=payload.body,
        publication_locator=publication_locator,
        figure_id=figure_id,
    )


__all__ = [
    "CTPPayload",
    "ctp_to_paper_source",
]
