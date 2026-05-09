"""Frozen ``FigureCorpusBundle`` schema.

This is the **data subset** of the broader ``FigureArtifactBundle`` (planned
for F2.3 / F5 / F6). A corpus bundle carries everything the L1 / L3 / L4
fidelity layers need *without* requiring any GPU training:

* ``primary_corpus``        — T1: documented one-of-a-kind primary sources
* ``contrast_pairs``        — T2: reviewed positions vs named opponents
* ``coverage_map``          — T3: domain coverage map (knows / does-not-know)
* ``time_windows``          — T3: corpus-derived early/middle/late views
* ``license_summary``       — license / IP audit footprint
* ``statistics``            — token / doc / span counts

The bundle is a frozen dataclass and is sha256-addressed by ``integrity_hash``
so any byte-level change yields a fresh ``bundle_id`` (R15 rollback contract).
Later packets (F5 steering, F6 LoRA) extend this bundle into the broader
``FigureArtifactBundle`` by composition, not mutation: a steering bundle
*references* the corpus bundle by ``bundle_id``.

This module is **schema only**. The freezer / hash / builder helpers live in
``lifeform_domain_figure.curation.freeze`` so that schema imports stay
side-effect-free.

Forward-compat note:
    ``FigureCoverageMap``, ``ReviewedContrastPair``, and ``TimeWindow`` are
    declared here as minimal frozen records so the bundle schema is stable
    from V1 (D1). Later packets (D4 / D5) **add fields with defaults** to
    these records — never reorder or rename — preserving the integrity hash
    invariant for old bundles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


SCHEMA_VERSION = 1


class DocumentKind(str, Enum):
    """The kind of primary-source document.

    The kernel ingestion layer only knows ``IngestionSourceKind`` (a coarse
    label — BOOK / WEB / TASK_RESULT / CORPUS). Inside the figure vertical
    we care about a finer split because the L3 grounding contract surfaces
    different citation forms per kind (a paper cites by section; a letter
    cites by sender / recipient / date; a notebook cites by page).
    """

    PAPER = "paper"
    LETTER = "letter"
    LECTURE = "lecture"
    NOTEBOOK = "notebook"
    BOOK = "book"
    MANUSCRIPT = "manuscript"


class EvidenceStrength(str, Enum):
    """Evidentiary basis for treating this source as the figure's voice.

    ``FIRST_HAND`` — the figure wrote / signed it themselves.
    ``DICTATED``   — recorded as the figure's words by a named scribe / stenographer.
    ``CONTEMPORANEOUS_ATTRIBUTED`` — third-party record made at the time, attributed.
    ``REPORTED``   — later third-party paraphrase or recollection.
    """

    FIRST_HAND = "first_hand"
    DICTATED = "dictated"
    CONTEMPORANEOUS_ATTRIBUTED = "contemporaneous_attributed"
    REPORTED = "reported"


@dataclass(frozen=True)
class PrimarySource:
    """One primary-source document, normalised across the four ingest kinds.

    ``locator`` is the canonical citation string the kernel ingestion adapter
    already produces (``paper:einstein-1905-001:lang=de:para=...``); the
    bundle keeps it verbatim so downstream retrieval / decoding can render
    citations without re-deriving provenance.

    ``sha256`` covers ``text`` (UTF-8 encoded) so cross-source dedup (D2) is
    a one-line tuple lookup.

    ``license`` is mandatory non-empty: V1 enforces "license known or fail
    loud" — the curation flow refuses to admit a source without a declared
    license string. The exact license vocabulary is deliberately not an enum
    here so verticals can declare their own audit-time vocabularies without
    requiring schema migration.
    """

    source_id: str
    document_kind: DocumentKind
    figure_id: str
    written_lang: str
    license: str
    source_url: str
    sha256: str
    locator: str
    text: str
    written_year: int | None = None
    evidence_strength: EvidenceStrength = EvidenceStrength.FIRST_HAND

    def __post_init__(self) -> None:
        for name in (
            "source_id",
            "figure_id",
            "written_lang",
            "license",
            "source_url",
            "sha256",
            "locator",
            "text",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"PrimarySource.{name} must be a non-empty string for "
                    f"source_id={self.source_id!r}"
                )
        if len(self.sha256) != 64:
            raise ValueError(
                f"PrimarySource.sha256 must be a 64-char hex digest, "
                f"got {self.sha256!r} for source_id={self.source_id!r}"
            )
        if self.written_year is not None and not (-3000 <= self.written_year <= 9999):
            raise ValueError(
                f"PrimarySource.written_year out of plausible range "
                f"({self.written_year!r}) for source_id={self.source_id!r}"
            )


@dataclass(frozen=True)
class LicenseSummary:
    """Aggregated license footprint for a bundle.

    Inputs come from the per-source ``PrimarySource.license`` values; the
    summary is what an auditor reads to decide whether the bundle is safe
    to ship under a given regulatory posture.
    """

    distinct_licenses: tuple[str, ...]
    public_domain_source_count: int
    licensed_source_count: int
    unknown_license_source_count: int
    audit_note: str

    def __post_init__(self) -> None:
        if self.public_domain_source_count < 0:
            raise ValueError(
                "LicenseSummary.public_domain_source_count must be >= 0"
            )
        if self.licensed_source_count < 0:
            raise ValueError(
                "LicenseSummary.licensed_source_count must be >= 0"
            )
        if self.unknown_license_source_count < 0:
            raise ValueError(
                "LicenseSummary.unknown_license_source_count must be >= 0"
            )


@dataclass(frozen=True)
class CorpusStatistics:
    """Cheap-to-compute statistics over the primary corpus.

    These are reviewer-oriented surface metrics, not learning signals. The
    retrieval index will compute its own internal token statistics; this
    record exists so a curator can inspect "is this bundle big enough?"
    without loading the index.
    """

    document_count: int
    paragraph_count: int
    char_count: int
    token_estimate: int
    earliest_year: int | None
    latest_year: int | None
    languages: tuple[str, ...]
    document_kind_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        for name in (
            "document_count",
            "paragraph_count",
            "char_count",
            "token_estimate",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"CorpusStatistics.{name} must be >= 0")
        if (
            self.earliest_year is not None
            and self.latest_year is not None
            and self.latest_year < self.earliest_year
        ):
            raise ValueError(
                f"CorpusStatistics.latest_year ({self.latest_year}) must be "
                f">= earliest_year ({self.earliest_year})"
            )


# ---------------------------------------------------------------------------
# T2 (contrast) — schema declared in V1 to lock the bundle layout. The
# extractor + first reviewed pairs land in D5; until then this defaults to
# empty.
# ---------------------------------------------------------------------------


class StanceTag(str, Enum):
    """Coarse stance label used by reviewed contrast pairs.

    The label is **descriptive only**: the runtime steering layer derives
    its own embedding-space directions from the underlying texts; the tag is
    surfaced to reviewers / dashboards so an Einstein-vs-Bohr pair can be
    grouped under ``foundations`` rather than a free-form string.
    """

    FOUNDATIONS = "foundations"
    ETHICS = "ethics"
    POLITICS = "politics"
    METHODOLOGY = "methodology"
    AESTHETICS = "aesthetics"
    OTHER = "other"


@dataclass(frozen=True)
class ReviewedContrastPair:
    """A reviewed Einstein-vs-Opponent (or general figure-vs-opponent) pair.

    Both sides are carried as ``PrimarySource``-style records (text +
    citation locator) so the steering trainer can produce reviewable
    evidence pointers identical to the L3 grounding format.
    """

    pair_id: str
    stance_tag: StanceTag
    figure_excerpt_source_id: str
    figure_excerpt_locator: str
    figure_excerpt_text: str
    opponent_id: str
    opponent_excerpt_locator: str
    opponent_excerpt_text: str
    description: str
    reviewed_by: str
    confidence: float

    def __post_init__(self) -> None:
        for name in (
            "pair_id",
            "figure_excerpt_source_id",
            "figure_excerpt_locator",
            "figure_excerpt_text",
            "opponent_id",
            "opponent_excerpt_locator",
            "opponent_excerpt_text",
            "description",
            "reviewed_by",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"ReviewedContrastPair.{name} must be non-empty for "
                    f"pair_id={self.pair_id!r}"
                )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"ReviewedContrastPair.confidence must be in [0,1], "
                f"got {self.confidence!r}"
            )


# ---------------------------------------------------------------------------
# T3 (coverage / time window) — schema declared in V1, populated in D4 / D6.
# ---------------------------------------------------------------------------


class CoverageDecision(str, Enum):
    """Per-domain coverage decision the L4 ScopeRefuser will read.

    ``COVERED``        — the corpus has substantive evidence on this domain.
    ``THIN``           — token / document count below threshold; degrade to soft disclaim.
    ``OUT_OF_SCOPE``   — explicitly out of the figure's documented competence.
    ``POST_LIFESPAN``  — temporally impossible (asks about events after death).
    """

    COVERED = "covered"
    THIN = "thin"
    OUT_OF_SCOPE = "out_of_scope"
    POST_LIFESPAN = "post_lifespan"


@dataclass(frozen=True)
class CoverageEntry:
    """One per-domain coverage entry inside a ``FigureCoverageMap``."""

    domain: str
    decision: CoverageDecision
    source_count: int
    token_estimate: int
    rationale: str

    def __post_init__(self) -> None:
        if not self.domain.strip():
            raise ValueError("CoverageEntry.domain must be non-empty")
        if self.source_count < 0:
            raise ValueError("CoverageEntry.source_count must be >= 0")
        if self.token_estimate < 0:
            raise ValueError("CoverageEntry.token_estimate must be >= 0")
        if not self.rationale.strip():
            raise ValueError(
                f"CoverageEntry.rationale must be non-empty for domain "
                f"{self.domain!r}; the L4 ScopeRefuser surfaces this rationale "
                f"to the user when refusing"
            )


@dataclass(frozen=True)
class FigureCoverageMap:
    """Domain-level knows / does-not-know map driving the L4 contract.

    Entries are ordered (curation order); the runtime ``ScopeRefuser`` does
    a single typed dispatch on ``decision`` — there is **no** keyword
    matching on the user query (``no-keyword-matching-hacks.mdc``).
    """

    entries: tuple[CoverageEntry, ...]
    default_decision: CoverageDecision
    audit_note: str

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for entry in self.entries:
            if entry.domain in seen:
                raise ValueError(
                    f"FigureCoverageMap.entries duplicate domain {entry.domain!r}"
                )
            seen.add(entry.domain)


@dataclass(frozen=True)
class TimeWindow:
    """A corpus-derived time window (early / middle / late figure view).

    Distinct from ``HistoricalFigureProfile.time_windows`` (which is
    reviewer-declared at profile level). This window is **derived from the
    actual corpus** so the runtime can answer "which view of Einstein was
    this argument written under?" deterministically.
    """

    window_id: str
    year_start: int
    year_end: int
    description: str
    source_count: int
    token_estimate: int

    def __post_init__(self) -> None:
        if not self.window_id.strip():
            raise ValueError("TimeWindow.window_id must be non-empty")
        if self.year_end < self.year_start:
            raise ValueError(
                f"TimeWindow.year_end ({self.year_end}) must be >= "
                f"year_start ({self.year_start}) for window {self.window_id!r}"
            )
        if self.source_count < 0:
            raise ValueError("TimeWindow.source_count must be >= 0")
        if self.token_estimate < 0:
            raise ValueError("TimeWindow.token_estimate must be >= 0")


# ---------------------------------------------------------------------------
# Top-level bundle.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FigureCorpusBundle:
    """Frozen, sha256-addressed corpus bundle for a single figure.

    Constructed by ``lifeform_domain_figure.curation.freeze.build_figure_corpus_bundle``;
    consumed by later packets (retrieval index builder, coverage runtime,
    style prior builder, contrast steering trainer, LoRA data prep).

    Defaults are tuned so the V1 ("T1 only") happy path can produce a valid
    bundle without yet having any contrast pairs / coverage map / time
    windows — D2 / D4 / D5 / D6 progressively populate the rest. Adding
    fields later is allowed via dataclass replacement; **renaming** or
    **reordering** existing fields would change the integrity hash and is
    treated as a schema migration (bump ``SCHEMA_VERSION`` and document
    in DATA_CONTRACT 2.15).
    """

    bundle_id: str
    figure_id: str
    schema_version: int
    reviewed_by: str
    created_at_utc: str
    integrity_hash: str
    primary_corpus: tuple[PrimarySource, ...]
    license_summary: LicenseSummary
    statistics: CorpusStatistics
    contrast_pairs: tuple[ReviewedContrastPair, ...] = ()
    coverage_map: FigureCoverageMap | None = None
    time_windows: tuple[TimeWindow, ...] = ()
    audit_note: str = ""

    def __post_init__(self) -> None:
        for name in (
            "bundle_id",
            "figure_id",
            "reviewed_by",
            "created_at_utc",
            "integrity_hash",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"FigureCorpusBundle.{name} must be a non-empty string"
                )
        if not isinstance(self.schema_version, int) or self.schema_version < 1:
            raise ValueError(
                "FigureCorpusBundle.schema_version must be a positive int"
            )
        if not self.primary_corpus:
            raise ValueError(
                "FigureCorpusBundle.primary_corpus must be non-empty: a "
                "corpus bundle with zero primary sources cannot drive any "
                "fidelity layer"
            )
        seen_source_ids: set[str] = set()
        for source in self.primary_corpus:
            if source.figure_id != self.figure_id:
                raise ValueError(
                    f"FigureCorpusBundle: source_id={source.source_id!r} "
                    f"declares figure_id={source.figure_id!r} but bundle is "
                    f"for {self.figure_id!r}"
                )
            if source.source_id in seen_source_ids:
                raise ValueError(
                    f"FigureCorpusBundle: duplicate source_id "
                    f"{source.source_id!r}"
                )
            seen_source_ids.add(source.source_id)
        seen_pair_ids: set[str] = set()
        for pair in self.contrast_pairs:
            if pair.pair_id in seen_pair_ids:
                raise ValueError(
                    f"FigureCorpusBundle: duplicate contrast_pair pair_id "
                    f"{pair.pair_id!r}"
                )
            seen_pair_ids.add(pair.pair_id)
        seen_window_ids: set[str] = set()
        for window in self.time_windows:
            if window.window_id in seen_window_ids:
                raise ValueError(
                    f"FigureCorpusBundle: duplicate time_window window_id "
                    f"{window.window_id!r}"
                )
            seen_window_ids.add(window.window_id)
        if len(self.integrity_hash) != 64:
            raise ValueError(
                f"FigureCorpusBundle.integrity_hash must be a 64-char hex "
                f"digest, got {self.integrity_hash!r}"
            )

    def has_contrast(self) -> bool:
        return bool(self.contrast_pairs)

    def has_coverage_map(self) -> bool:
        return self.coverage_map is not None

    def has_time_windows(self) -> bool:
        return bool(self.time_windows)


__all__ = [
    "SCHEMA_VERSION",
    "CorpusStatistics",
    "CoverageDecision",
    "CoverageEntry",
    "DocumentKind",
    "EvidenceStrength",
    "FigureCorpusBundle",
    "FigureCoverageMap",
    "LicenseSummary",
    "PrimarySource",
    "ReviewedContrastPair",
    "StanceTag",
    "TimeWindow",
]
