"""Pure freeze pipeline for ``FigureCorpusBundle``.

Mirrors the integrity-hash discipline from
[`packages/lifeform-domain-character/src/lifeform_domain_character/template.py`]
(``compute_template_integrity_hash`` / ``_to_serializable``) so the two
verticals share the same audit shape.

Inputs are typed records produced by other parts of the wheel:

* ``primary_corpus`` — produced by ``corpus.ingest_*`` adapters (D1)
* ``contrast_pairs`` — produced by ``contrast.pair_extract`` (D5; default empty)
* ``coverage_map``   — produced by ``metadata.coverage_map`` (D4; default None)
* ``time_windows``   — produced by ``metadata.time_window`` (D4; default empty)

The freezer:

1. Computes per-source statistics (counts, langs, kind histogram, year span).
2. Aggregates the license footprint into a ``LicenseSummary``.
3. Computes a deterministic SHA-256 over the canonical serialization of
   every identity-bearing field (everything in the bundle except
   ``integrity_hash`` itself and ``created_at_utc``).
4. Builds a frozen ``FigureCorpusBundle`` carrying that hash; downstream
   consumers see one immutable object addressable by ``bundle_id``.

Why ``created_at_utc`` is excluded from the integrity hash:
    Two identical curation inputs frozen at different wall-clock times
    must produce the same ``bundle_id`` (R15: byte-equivalent reproduction
    is the rollback contract). ``created_at_utc`` is a non-load-bearing
    audit field for human reviewers.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from lifeform_ingestion.envelope import IngestionChunk

from lifeform_domain_figure.bundle import (
    SCHEMA_VERSION,
    CorpusStatistics,
    DocumentKind,
    EvidenceStrength,
    FigureCorpusBundle,
    FigureCoverageMap,
    LicenseSummary,
    PrimarySource,
    ReviewedContrastPair,
    TimeWindow,
)


_PUBLIC_DOMAIN_LABELS = frozenset(
    {
        "public-domain",
        "public_domain",
        "pd",
        "cc0",
        "cc-0",
    }
)
_UNKNOWN_LABELS = frozenset(
    {
        "",
        "unknown",
        "unspecified",
        "tbd",
    }
)


def utc_iso_now() -> str:
    """Wall-clock UTC timestamp in ``YYYY-MM-DDTHH:MM:SSZ`` form."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_serializable(value: Any) -> Any:
    """Recursively convert a typed dataclass tree to JSON-compatible primitives.

    Mirrors ``lifeform-domain-character.template._to_serializable`` and
    ``vz-memory.persistence._to_serializable``. Each wheel deliberately
    keeps its own copy: cross-wheel sharing of a serialization helper is
    not worth the dependency overhead, and integrity-hash inputs differ
    per vertical.
    """

    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (tuple, list)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_serializable(val) for key, val in value.items()}
    if is_dataclass(value):
        return {
            field.name: _to_serializable(getattr(value, field.name))
            for field in fields(value)
        }
    raise TypeError(
        f"figure-bundle freeze: refusing to serialize unknown value type "
        f"{type(value).__name__!r} — every field on the bundle must be a "
        f"frozen dataclass / enum / tuple / dict / primitive."
    )


def canonical_serialize(value: Any) -> bytes:
    """Return the canonical UTF-8 JSON bytes used as the integrity hash input.

    Sort-keys + no extra whitespace so two byte-equivalent payloads always
    serialize identically regardless of the dict insertion order Python
    happens to produce on a given platform.
    """

    import json

    return json.dumps(
        _to_serializable(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def primary_source_from_envelope_chunk(
    *,
    chunk: IngestionChunk,
    figure_id: str,
    document_kind: DocumentKind,
    written_lang: str,
    license: str,
    source_url: str,
    source_id: str | None = None,
    written_year: int | None = None,
    evidence_strength: EvidenceStrength = EvidenceStrength.FIRST_HAND,
) -> PrimarySource:
    """Lift one ``IngestionChunk`` (already locator-stamped by an adapter)
    into a ``PrimarySource`` record.

    The chunk is treated as one document; the chunk's ``locator`` becomes
    the source's citation locator. Multi-paragraph documents that have been
    split into multiple chunks should produce multiple ``PrimarySource``
    records, one per chunk, each with its own ``source_id``.

    The ``sha256`` is computed over ``chunk.text`` (UTF-8) so cross-source
    dedup (D2) can collapse byte-identical chunks deterministically.
    """

    if chunk.has_parse_error:
        raise ValueError(
            f"primary_source_from_envelope_chunk: chunk {chunk.chunk_id!r} "
            f"has parse_error={chunk.parse_error!r}; the curation flow "
            f"must not lift failed chunks into the bundle (silent failure "
            f"would let the bundle ship a placeholder)."
        )
    text_bytes = chunk.text.encode("utf-8")
    text_sha = hashlib.sha256(text_bytes).hexdigest()
    final_source_id = source_id or chunk.chunk_id
    return PrimarySource(
        source_id=final_source_id,
        document_kind=document_kind,
        figure_id=figure_id,
        written_lang=written_lang,
        license=license,
        source_url=source_url,
        sha256=text_sha,
        locator=chunk.locator,
        text=chunk.text,
        written_year=written_year,
        evidence_strength=evidence_strength,
    )


def _classify_license(label: str) -> str:
    """Return the audit category for a license label."""
    normalised = label.strip().lower()
    if normalised in _UNKNOWN_LABELS:
        return "unknown"
    if normalised in _PUBLIC_DOMAIN_LABELS or normalised.startswith("public-domain"):
        return "public_domain"
    return "licensed"


def _build_license_summary(sources: tuple[PrimarySource, ...]) -> LicenseSummary:
    pd_count = 0
    licensed_count = 0
    unknown_count = 0
    distinct: set[str] = set()
    for source in sources:
        category = _classify_license(source.license)
        distinct.add(source.license.strip())
        if category == "public_domain":
            pd_count += 1
        elif category == "licensed":
            licensed_count += 1
        else:
            unknown_count += 1
    audit_note = (
        f"public_domain={pd_count} licensed={licensed_count} "
        f"unknown={unknown_count} total={len(sources)}"
    )
    return LicenseSummary(
        distinct_licenses=tuple(sorted(distinct)),
        public_domain_source_count=pd_count,
        licensed_source_count=licensed_count,
        unknown_license_source_count=unknown_count,
        audit_note=audit_note,
    )


def _estimate_tokens(text: str) -> int:
    """Cheap whitespace-based token estimate.

    The retrieval index will compute its own real token statistics; this
    estimator exists only so :class:`CorpusStatistics` has something
    reviewer-meaningful without pulling a tokenizer dependency into the
    schema layer.
    """

    return len(text.split())


def _build_corpus_statistics(sources: tuple[PrimarySource, ...]) -> CorpusStatistics:
    paragraph_count = len(sources)
    char_count = sum(len(source.text) for source in sources)
    token_estimate = sum(_estimate_tokens(source.text) for source in sources)
    languages = tuple(sorted({source.written_lang for source in sources}))
    kind_hist: Counter[str] = Counter(
        source.document_kind.value for source in sources
    )
    document_kind_counts = tuple(sorted(kind_hist.items()))
    years = [source.written_year for source in sources if source.written_year is not None]
    earliest = min(years) if years else None
    latest = max(years) if years else None
    return CorpusStatistics(
        document_count=len({source.source_id for source in sources}),
        paragraph_count=paragraph_count,
        char_count=char_count,
        token_estimate=token_estimate,
        earliest_year=earliest,
        latest_year=latest,
        languages=languages,
        document_kind_counts=document_kind_counts,
    )


@dataclass(frozen=True)
class BuildBundleInputs:
    """Typed input bag for ``build_figure_corpus_bundle``.

    A frozen record (rather than a long kwargs list) so that adding a new
    optional input in a later packet does not break call-sites that still
    only supply the V1 fields.
    """

    figure_id: str
    reviewed_by: str
    primary_corpus: tuple[PrimarySource, ...]
    contrast_pairs: tuple[ReviewedContrastPair, ...] = ()
    coverage_map: FigureCoverageMap | None = None
    time_windows: tuple[TimeWindow, ...] = ()
    audit_note: str = ""

    def __post_init__(self) -> None:
        if not self.figure_id.strip():
            raise ValueError("BuildBundleInputs.figure_id must be non-empty")
        if not self.reviewed_by.strip():
            raise ValueError(
                "BuildBundleInputs.reviewed_by must be non-empty: every "
                "frozen bundle must declare a reviewer."
            )
        if not self.primary_corpus:
            raise ValueError(
                "BuildBundleInputs.primary_corpus must be non-empty: V1 of "
                "the figure vertical refuses to freeze a bundle with no "
                "primary sources (no L3 ground truth would exist)."
            )


def compute_corpus_bundle_integrity_hash(inputs: BuildBundleInputs) -> str:
    """Return the SHA-256 hex digest used as ``bundle.integrity_hash``.

    Identity-bearing fields only (everything ``BuildBundleInputs`` carries
    plus ``schema_version``). ``created_at_utc`` is excluded so byte-
    equivalent inputs at different wall-clock times produce the same hash
    (R15 reproducibility).
    """

    identity = {
        "schema_version": SCHEMA_VERSION,
        "figure_id": inputs.figure_id,
        "reviewed_by": inputs.reviewed_by,
        "primary_corpus": inputs.primary_corpus,
        "contrast_pairs": inputs.contrast_pairs,
        "coverage_map": inputs.coverage_map,
        "time_windows": inputs.time_windows,
        "audit_note": inputs.audit_note,
    }
    canonical = canonical_serialize(identity)
    return hashlib.sha256(canonical).hexdigest()


def bundle_id_from_hash(figure_id: str, integrity_hash: str) -> str:
    """Build the canonical ``bundle_id`` from figure id + hash prefix."""
    return f"figure-corpus:{figure_id}:{integrity_hash[:16]}"


def build_figure_corpus_bundle(
    inputs: BuildBundleInputs,
    *,
    created_at_utc: str | None = None,
) -> FigureCorpusBundle:
    """Assemble + freeze a ``FigureCorpusBundle``.

    Pure / deterministic for fixed ``created_at_utc``. Default
    ``created_at_utc=None`` stamps the wall-clock time, which is fine for
    most callers; tests that need byte-stable output should pass an
    explicit timestamp.
    """

    integrity_hash = compute_corpus_bundle_integrity_hash(inputs)
    bundle_id = bundle_id_from_hash(inputs.figure_id, integrity_hash)
    statistics = _build_corpus_statistics(inputs.primary_corpus)
    license_summary = _build_license_summary(inputs.primary_corpus)
    return FigureCorpusBundle(
        bundle_id=bundle_id,
        figure_id=inputs.figure_id,
        schema_version=SCHEMA_VERSION,
        reviewed_by=inputs.reviewed_by,
        created_at_utc=created_at_utc or utc_iso_now(),
        integrity_hash=integrity_hash,
        primary_corpus=inputs.primary_corpus,
        license_summary=license_summary,
        statistics=statistics,
        contrast_pairs=inputs.contrast_pairs,
        coverage_map=inputs.coverage_map,
        time_windows=inputs.time_windows,
        audit_note=inputs.audit_note,
    )


__all__ = [
    "BuildBundleInputs",
    "bundle_id_from_hash",
    "build_figure_corpus_bundle",
    "canonical_serialize",
    "compute_corpus_bundle_integrity_hash",
    "primary_source_from_envelope_chunk",
    "utc_iso_now",
]
