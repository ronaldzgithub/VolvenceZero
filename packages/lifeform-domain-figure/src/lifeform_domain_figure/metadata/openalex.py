"""OpenAlex metadata adapter.

OpenAlex (https://api.openalex.org/) is a public scholarly graph
covering papers, authors, venues, and topics. The figure vertical
uses two slices of OpenAlex:

* Author works listing → :class:`AuthoredWorkSummary` records that
  widen the profile's coverage seed before centroid building.
* Concept / topic tags → :class:`DomainCoverageHint` records that
  surface in-domain topic labels for the L4 coverage map.

V1 takes a pre-downloaded :class:`OpenAlexWorkPayload`. V2 will add
an HTTP client behind the :class:`OpenAlexClient` Protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from lifeform_domain_figure.metadata.records import (
    AuthoredWorkSummary,
    DomainCoverageHint,
    MetadataSource,
)


@dataclass(frozen=True)
class OpenAlexWorkPayload:
    """Pre-downloaded OpenAlex ``Work`` record (a single paper / book)."""

    openalex_id: str  # canonical OpenAlex id, e.g., "W4205692301"
    title: str
    publication_year: int | None
    venue: str
    language: str
    concept_labels: tuple[str, ...]
    primary_topic: str = ""
    cited_by_count: int = 0

    def __post_init__(self) -> None:
        for name in ("openalex_id", "title", "venue", "language"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"OpenAlexWorkPayload.{name} must be non-empty for "
                    f"openalex_id={self.openalex_id!r}"
                )
        if self.publication_year is not None and not (
            -3000 <= self.publication_year <= 9999
        ):
            raise ValueError(
                f"OpenAlexWorkPayload.publication_year out of plausible "
                f"range: {self.publication_year!r}"
            )
        if self.cited_by_count < 0:
            raise ValueError(
                f"OpenAlexWorkPayload.cited_by_count must be >= 0, got "
                f"{self.cited_by_count!r}"
            )


def openalex_to_authored_work(
    payload: OpenAlexWorkPayload,
    *,
    figure_id: str,
) -> AuthoredWorkSummary:
    """Translate an OpenAlex work payload into an :class:`AuthoredWorkSummary`."""

    return AuthoredWorkSummary(
        work_id=f"openalex:{payload.openalex_id}",
        figure_id=figure_id,
        title=payload.title,
        year=payload.publication_year,
        venue=payload.venue,
        language=payload.language,
        topic_tags=payload.concept_labels,
        source=MetadataSource.OPENALEX,
        source_id=payload.openalex_id,
    )


def openalex_to_domain_hints(
    payload: OpenAlexWorkPayload,
    *,
    confidence: float = 0.6,
) -> tuple[DomainCoverageHint, ...]:
    """Lift OpenAlex concept labels into :class:`DomainCoverageHint` records.

    Each concept label becomes one in-domain coverage hint
    (``is_out_of_scope=False``). Reviewers may downstream invert
    individual labels to out-of-scope before feeding them into the
    coverage map.
    """

    hints: list[DomainCoverageHint] = []
    seen: set[str] = set()
    for label in payload.concept_labels:
        norm = label.strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        hints.append(
            DomainCoverageHint(
                label=norm,
                description=f"OpenAlex concept tag from work {payload.openalex_id}",
                is_out_of_scope=False,
                source=MetadataSource.OPENALEX,
                source_id=payload.openalex_id,
                confidence=confidence,
            )
        )
    return tuple(hints)


class OpenAlexClient(Protocol):
    """Forward-declared Protocol for a live OpenAlex HTTP client."""

    def fetch_author_works(
        self, *, openalex_author_id: str
    ) -> tuple[OpenAlexWorkPayload, ...]: ...


class _OfflineOpenAlexClient:
    """V1 stub: every fetch raises ``NotImplementedError``."""

    def fetch_author_works(
        self, *, openalex_author_id: str
    ) -> tuple[OpenAlexWorkPayload, ...]:
        raise NotImplementedError(
            "V1 of the figure vertical has no live OpenAlex client. "
            "Construct OpenAlexWorkPayload instances directly from "
            f"pre-fetched JSON. Refused fetch for "
            f"openalex_author_id={openalex_author_id!r}."
        )


def offline_openalex_client() -> OpenAlexClient:
    """Return the V1 offline stub OpenAlex client."""
    return _OfflineOpenAlexClient()


__all__ = [
    "OpenAlexClient",
    "OpenAlexWorkPayload",
    "offline_openalex_client",
    "openalex_to_authored_work",
    "openalex_to_domain_hints",
]
