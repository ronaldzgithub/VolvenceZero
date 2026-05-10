"""Project Gutenberg archive adapter.

Project Gutenberg hosts public-domain books in plain-text and EPUB
form. The URL pattern is::

    https://www.gutenberg.org/ebooks/{ID}
    https://www.gutenberg.org/files/{ID}/{ID}-0.txt

Books on Gutenberg are full-length, so the curator typically chunks
them at chapter or section boundaries before constructing one
payload per chunk. This adapter treats each payload as one
:class:`FigurePaperSource` (Gutenberg's lecture / letter offerings
are rare; the few that exist can be re-cast through CPAE or
Internet Archive instead).
"""

from __future__ import annotations

from dataclasses import dataclass

from lifeform_domain_figure.corpus.ingest_papers import FigurePaperSource


@dataclass(frozen=True)
class GutenbergPayload:
    """Pre-downloaded Gutenberg book / section payload."""

    ebook_id: int
    title: str
    language: str
    body: str
    source_url: str
    section_label: str = ""
    year: int | None = None
    author_id: str = ""

    def __post_init__(self) -> None:
        for name in ("title", "language", "body", "source_url"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"GutenbergPayload.{name} must be non-empty for "
                    f"ebook_id={self.ebook_id!r}"
                )
        if self.ebook_id <= 0:
            raise ValueError(
                f"GutenbergPayload.ebook_id must be > 0, got "
                f"{self.ebook_id!r}"
            )
        if self.year is not None and not (-3000 <= self.year <= 9999):
            raise ValueError(
                f"GutenbergPayload.year out of plausible range: "
                f"{self.year!r}"
            )


def gutenberg_to_paper_source(
    payload: GutenbergPayload,
    *,
    figure_id: str,
) -> FigurePaperSource:
    """Translate a Gutenberg ebook / section payload into a paper source."""

    section_suffix = f":section={payload.section_label}" if payload.section_label else ""
    publication_locator = f"gutenberg:ebook={payload.ebook_id}{section_suffix}"
    paper_id_suffix = (
        f":{payload.section_label}" if payload.section_label else ""
    )
    paper_id = f"gutenberg:{payload.ebook_id}{paper_id_suffix}"
    return FigurePaperSource(
        paper_id=paper_id,
        title=(
            f"{payload.title}"
            if not payload.section_label
            else f"{payload.title} — {payload.section_label}"
        ),
        year=payload.year if payload.year is not None else 0,
        language=payload.language,
        body=payload.body,
        publication_locator=publication_locator,
        figure_id=figure_id,
    )


__all__ = [
    "GutenbergPayload",
    "gutenberg_to_paper_source",
]
