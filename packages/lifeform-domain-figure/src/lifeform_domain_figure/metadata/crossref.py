"""Crossref metadata adapter.

Crossref (https://api.crossref.org/) provides DOI-resolved publication
metadata. The figure vertical uses Crossref records to enrich
authored-work metadata that complements OpenAlex (Crossref tends to
have more complete venue / volume / issue strings).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from lifeform_domain_figure.metadata.records import (
    AuthoredWorkSummary,
    MetadataSource,
)


@dataclass(frozen=True)
class CrossrefWorkPayload:
    """Pre-downloaded Crossref ``Work`` record (one DOI)."""

    doi: str  # canonical DOI, e.g., "10.1002/andp.19053221004"
    title: str
    publication_year: int | None
    container_title: str
    language: str
    subject_tags: tuple[str, ...] = ()
    issue: str = ""
    volume: str = ""

    def __post_init__(self) -> None:
        for name in ("doi", "title", "container_title", "language"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"CrossrefWorkPayload.{name} must be non-empty for "
                    f"doi={self.doi!r}"
                )
        if self.publication_year is not None and not (
            -3000 <= self.publication_year <= 9999
        ):
            raise ValueError(
                f"CrossrefWorkPayload.publication_year out of plausible "
                f"range: {self.publication_year!r}"
            )


def crossref_to_authored_work(
    payload: CrossrefWorkPayload,
    *,
    figure_id: str,
) -> AuthoredWorkSummary:
    """Translate a Crossref work payload into an :class:`AuthoredWorkSummary`."""

    venue = payload.container_title
    if payload.volume:
        venue = f"{venue} (vol={payload.volume})"
    if payload.issue:
        venue = f"{venue} (issue={payload.issue})"
    return AuthoredWorkSummary(
        work_id=f"crossref:{payload.doi}",
        figure_id=figure_id,
        title=payload.title,
        year=payload.publication_year,
        venue=venue,
        language=payload.language,
        topic_tags=payload.subject_tags,
        source=MetadataSource.CROSSREF,
        source_id=payload.doi,
    )


class CrossrefClient(Protocol):
    """Forward-declared Protocol for a live Crossref HTTP client."""

    def fetch_work(self, *, doi: str) -> CrossrefWorkPayload: ...


class _OfflineCrossrefClient:
    """V1 stub: every fetch raises ``NotImplementedError``."""

    def fetch_work(self, *, doi: str) -> CrossrefWorkPayload:
        raise NotImplementedError(
            "V1 of the figure vertical has no live Crossref client. "
            "Construct CrossrefWorkPayload instances directly from "
            f"pre-fetched JSON. Refused fetch for doi={doi!r}."
        )


def offline_crossref_client() -> CrossrefClient:
    """Return the V1 offline stub Crossref client."""
    return _OfflineCrossrefClient()


__all__ = [
    "CrossrefClient",
    "CrossrefWorkPayload",
    "crossref_to_authored_work",
    "offline_crossref_client",
]
