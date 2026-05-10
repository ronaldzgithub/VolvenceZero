"""Neutral typed metadata records used by all metadata adapters.

Every metadata source (OpenAlex / Wikidata / Crossref / SEP) emits
records in the schemas below; this layer is **the canonical surface**
that the coverage-enrichment + time-window builders consume.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum


class MetadataSource(str, Enum):
    """Where a metadata record came from. Carried on every record."""

    OPENALEX = "openalex"
    WIKIDATA = "wikidata"
    CROSSREF = "crossref"
    SEP = "sep"


@dataclass(frozen=True)
class FigureLifespan:
    """Reviewer-confirmed lifespan record.

    Aggregated from Wikidata or other biographical sources. The L4
    not-known refusal contract uses ``death_year`` to refuse queries
    about events the figure cannot have observed.
    """

    figure_id: str
    birth_year: int
    death_year: int | None
    source: MetadataSource
    source_id: str
    confidence: float

    def __post_init__(self) -> None:
        if not self.figure_id.strip():
            raise ValueError("FigureLifespan.figure_id must be non-empty")
        if self.death_year is not None and self.death_year < self.birth_year:
            raise ValueError(
                f"FigureLifespan: death_year ({self.death_year}) must be "
                f">= birth_year ({self.birth_year}) for {self.figure_id!r}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"FigureLifespan.confidence must be in [0,1], "
                f"got {self.confidence!r}"
            )
        if not self.source_id.strip():
            raise ValueError("FigureLifespan.source_id must be non-empty")


@dataclass(frozen=True)
class AuthoredWorkSummary:
    """One authored work the figure produced (paper / book / lecture / etc.).

    Used to widen the profile's ``domain_coverage_seed`` /
    ``knowledge_seeds`` corpus before the L4 in-domain centroids are
    built. Matches OpenAlex / Crossref work records.
    """

    work_id: str
    figure_id: str
    title: str
    year: int | None
    venue: str
    language: str
    topic_tags: tuple[str, ...]
    source: MetadataSource
    source_id: str

    def __post_init__(self) -> None:
        for name in ("work_id", "figure_id", "title", "source_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"AuthoredWorkSummary.{name} must be non-empty for "
                    f"work_id={self.work_id!r}"
                )
        if self.year is not None and not (-3000 <= self.year <= 9999):
            raise ValueError(
                f"AuthoredWorkSummary.year out of plausible range: "
                f"{self.year!r}"
            )


@dataclass(frozen=True)
class DomainCoverageHint:
    """Reviewer-confirmed in-domain (or out-of-scope) topic hint."""

    label: str
    description: str
    is_out_of_scope: bool
    source: MetadataSource
    source_id: str
    confidence: float

    def __post_init__(self) -> None:
        for name in ("label", "description", "source_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"DomainCoverageHint.{name} must be non-empty for "
                    f"label={self.label!r}"
                )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"DomainCoverageHint.confidence must be in [0,1], got "
                f"{self.confidence!r}"
            )


@dataclass(frozen=True)
class TimeWindowHint:
    """One reviewer-confirmed time-window hint.

    Translated by :func:`lifeform_domain_figure.metadata.time_window_builder
    .build_time_window_hints_from_lifespan` into the bundle's
    ``time_windows`` payload.
    """

    window_id: str
    year_start: int
    year_end: int
    description: str
    source: MetadataSource
    source_id: str

    def __post_init__(self) -> None:
        if not self.window_id.strip():
            raise ValueError("TimeWindowHint.window_id must be non-empty")
        if not self.description.strip():
            raise ValueError("TimeWindowHint.description must be non-empty")
        if self.year_end < self.year_start:
            raise ValueError(
                f"TimeWindowHint.year_end ({self.year_end}) must be >= "
                f"year_start ({self.year_start}) for {self.window_id!r}"
            )
        if not self.source_id.strip():
            raise ValueError("TimeWindowHint.source_id must be non-empty")


@dataclass(frozen=True)
class MetadataDigest:
    """Aggregated metadata for a single figure, fingerprinted for audit."""

    figure_id: str
    lifespan: FigureLifespan | None
    authored_works: tuple[AuthoredWorkSummary, ...]
    coverage_hints: tuple[DomainCoverageHint, ...]
    time_window_hints: tuple[TimeWindowHint, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        if not self.figure_id.strip():
            raise ValueError("MetadataDigest.figure_id must be non-empty")
        if len(self.fingerprint) != 64:
            raise ValueError(
                f"MetadataDigest.fingerprint must be a 64-char hex digest, "
                f"got {self.fingerprint!r}"
            )
        if self.lifespan is not None and self.lifespan.figure_id != self.figure_id:
            raise ValueError(
                f"MetadataDigest: lifespan.figure_id "
                f"{self.lifespan.figure_id!r} != digest.figure_id "
                f"{self.figure_id!r}"
            )


def _digest_payload(
    *,
    figure_id: str,
    lifespan: FigureLifespan | None,
    authored_works: tuple[AuthoredWorkSummary, ...],
    coverage_hints: tuple[DomainCoverageHint, ...],
    time_window_hints: tuple[TimeWindowHint, ...],
) -> str:
    payload = (
        figure_id,
        (
            (
                lifespan.birth_year,
                lifespan.death_year,
                lifespan.source.value,
                lifespan.source_id,
                round(lifespan.confidence, 6),
            )
            if lifespan is not None
            else None
        ),
        tuple(
            (
                w.work_id,
                w.title,
                w.year,
                w.venue,
                w.language,
                w.topic_tags,
                w.source.value,
                w.source_id,
            )
            for w in authored_works
        ),
        tuple(
            (
                h.label,
                h.description,
                h.is_out_of_scope,
                h.source.value,
                h.source_id,
                round(h.confidence, 6),
            )
            for h in coverage_hints
        ),
        tuple(
            (
                w.window_id,
                w.year_start,
                w.year_end,
                w.description,
                w.source.value,
                w.source_id,
            )
            for w in time_window_hints
        ),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def aggregate_metadata(
    *,
    figure_id: str,
    lifespan: FigureLifespan | None = None,
    authored_works: tuple[AuthoredWorkSummary, ...] = (),
    coverage_hints: tuple[DomainCoverageHint, ...] = (),
    time_window_hints: tuple[TimeWindowHint, ...] = (),
) -> MetadataDigest:
    """Combine typed metadata records into a fingerprinted digest."""

    fingerprint = _digest_payload(
        figure_id=figure_id,
        lifespan=lifespan,
        authored_works=authored_works,
        coverage_hints=coverage_hints,
        time_window_hints=time_window_hints,
    )
    return MetadataDigest(
        figure_id=figure_id,
        lifespan=lifespan,
        authored_works=authored_works,
        coverage_hints=coverage_hints,
        time_window_hints=time_window_hints,
        fingerprint=fingerprint,
    )


__all__ = [
    "AuthoredWorkSummary",
    "DomainCoverageHint",
    "FigureLifespan",
    "MetadataDigest",
    "MetadataSource",
    "TimeWindowHint",
    "aggregate_metadata",
]
