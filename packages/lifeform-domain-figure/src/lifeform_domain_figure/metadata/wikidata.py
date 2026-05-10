"""Wikidata metadata adapter.

Wikidata (https://www.wikidata.org/) holds biographical metadata for
historical figures. The figure vertical uses three slices:

* ``date_of_birth`` / ``date_of_death`` → :class:`FigureLifespan`.
  The lifespan feeds the L4 not-known refusal contract: any query
  about events after ``death_year`` is automatically out-of-scope.
* ``occupation`` / ``field_of_work`` labels → optional in-domain
  coverage hints (low confidence; reviewer review required).
* Reviewer-declared time-window hints (e.g., "early-career-1900-1925")
  derived from the lifespan plus reviewer policy → produced by
  :func:`wikidata_to_time_window_hints`.

V1 takes a pre-downloaded :class:`WikidataPersonPayload`. V2 will
add the live SPARQL / WDQS Protocol implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from lifeform_domain_figure.metadata.records import (
    FigureLifespan,
    MetadataSource,
    TimeWindowHint,
)
from lifeform_domain_figure.metadata.time_window_builder import (
    build_time_window_hints_from_lifespan,
)


@dataclass(frozen=True)
class WikidataPersonPayload:
    """Pre-downloaded Wikidata person record (a single Q-id)."""

    qid: str  # canonical Wikidata id, e.g., "Q937"
    label: str  # "Albert Einstein"
    birth_year: int
    death_year: int | None
    occupation_labels: tuple[str, ...] = ()
    field_of_work_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("qid", "label"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"WikidataPersonPayload.{name} must be non-empty for "
                    f"qid={self.qid!r}"
                )
        if self.death_year is not None and self.death_year < self.birth_year:
            raise ValueError(
                f"WikidataPersonPayload: death_year ({self.death_year}) "
                f"must be >= birth_year ({self.birth_year}) for "
                f"qid={self.qid!r}"
            )


def wikidata_to_lifespan(
    payload: WikidataPersonPayload,
    *,
    figure_id: str,
    confidence: float = 0.95,
) -> FigureLifespan:
    """Translate a Wikidata person payload into a :class:`FigureLifespan`."""

    return FigureLifespan(
        figure_id=figure_id,
        birth_year=payload.birth_year,
        death_year=payload.death_year,
        source=MetadataSource.WIKIDATA,
        source_id=payload.qid,
        confidence=confidence,
    )


def wikidata_to_time_window_hints(
    payload: WikidataPersonPayload,
    *,
    figure_id: str,
    splits_at_years: tuple[int, ...] = (),
) -> tuple[TimeWindowHint, ...]:
    """Build a tuple of :class:`TimeWindowHint` records from a Wikidata payload.

    ``splits_at_years`` lets the curator declare reviewed inflection
    years (e.g., ``(1925,)`` for an early-vs-late Einstein split). If
    empty, the helper returns one window covering the full lifespan.
    """

    lifespan = wikidata_to_lifespan(payload, figure_id=figure_id)
    return build_time_window_hints_from_lifespan(
        lifespan,
        splits_at_years=splits_at_years,
        source=MetadataSource.WIKIDATA,
        source_id=payload.qid,
    )


class WikidataClient(Protocol):
    """Forward-declared Protocol for a live Wikidata SPARQL / WDQS client."""

    def fetch_person(self, *, qid: str) -> WikidataPersonPayload: ...


class _OfflineWikidataClient:
    """V1 stub: every fetch raises ``NotImplementedError``."""

    def fetch_person(self, *, qid: str) -> WikidataPersonPayload:
        raise NotImplementedError(
            "V1 of the figure vertical has no live Wikidata client. "
            "Construct WikidataPersonPayload instances directly from "
            f"pre-fetched SPARQL JSON. Refused fetch for qid={qid!r}."
        )


def offline_wikidata_client() -> WikidataClient:
    """Return the V1 offline stub Wikidata client."""
    return _OfflineWikidataClient()


__all__ = [
    "WikidataClient",
    "WikidataPersonPayload",
    "offline_wikidata_client",
    "wikidata_to_lifespan",
    "wikidata_to_time_window_hints",
]
