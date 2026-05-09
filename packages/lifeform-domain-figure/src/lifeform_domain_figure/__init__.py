"""Vertical: real-person digital revival from primary-source corpora.

This package is the monorepo-local application layer for "historical /
real-person figure to lifeform" work. It stands parallel to
``lifeform-domain-character`` (fictional characters) but enforces a
strictly different set of invariants:

* The source corpus is **evidence**, not narrative truth. Every claim
  the resulting lifeform produces must be traceable to a corpus
  citation (L3 grounding contract).
* Coverage is finite and known. Out-of-corpus topics must be refused
  or disclaimed (L4 not-known refusal contract).
* Style fidelity is statistical: tone / lexicon / sentence shape
  (L1 style prior contract).
* Stance fidelity is contrastive: the figure's documented positions
  versus contemporary opponents (L2 steering contract).

The wheel does NOT add a new kernel owner. Reviewed structured
artifacts compile into existing Volvence Zero owners (domain
knowledge, case memory, strategy playbook, boundary policy) just as
``lifeform-domain-character`` does, and additionally produce an
immutable :class:`FigureCorpusBundle` (V1 data subset of the broader
``FigureArtifactBundle`` planned for F2.3 / F5 / F6) consumed at
runtime by the ``lifeform-expression`` enforcement layer.

Public surface is added incrementally per the F1-F6 packet sequence
(see ``docs/specs/figure-vertical.md``). Early imports are
intentionally scoped: each packet adds its own modules and re-exports
here so consumers can pin against a stable namespace.
"""

from __future__ import annotations

from lifeform_domain_figure.bundle import (
    SCHEMA_VERSION as CORPUS_BUNDLE_SCHEMA_VERSION,
    CorpusStatistics,
    CoverageDecision,
    CoverageEntry,
    DocumentKind,
    EvidenceStrength,
    FigureCorpusBundle,
    FigureCoverageMap,
    LicenseSummary,
    PrimarySource,
    ReviewedContrastPair,
    StanceTag,
    TimeWindow,
)
from lifeform_domain_figure.corpus import (
    FigureLectureSource,
    FigureLetterSource,
    FigureNotebookSource,
    FigurePaperSource,
    ingest_lectures,
    ingest_letters,
    ingest_notebooks,
    ingest_papers,
)
from lifeform_domain_figure.curation import (
    BuildBundleInputs,
    bundle_id_from_hash,
    build_figure_corpus_bundle,
    canonical_serialize,
    compute_corpus_bundle_integrity_hash,
    primary_source_from_envelope_chunk,
    utc_iso_now,
)
from lifeform_domain_figure.profile import (
    FigureBoundaryPrior,
    FigureDrivePrior,
    FigureKnowledgeSeed,
    FigureSignatureCase,
    FigureStrategyPrior,
    HistoricalFigureProfile,
    TimeWindowedView,
)
from lifeform_domain_figure.profiles import build_einstein_profile


__all__ = [
    # Profile schema (F1.1)
    "FigureBoundaryPrior",
    "FigureDrivePrior",
    "FigureKnowledgeSeed",
    "FigureSignatureCase",
    "FigureStrategyPrior",
    "HistoricalFigureProfile",
    "TimeWindowedView",
    "build_einstein_profile",
    # Corpus ingestion adapters (F1.2)
    "FigureLectureSource",
    "FigureLetterSource",
    "FigureNotebookSource",
    "FigurePaperSource",
    "ingest_lectures",
    "ingest_letters",
    "ingest_notebooks",
    "ingest_papers",
    # Bundle schema (F1.3 / D1)
    "CORPUS_BUNDLE_SCHEMA_VERSION",
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
    # Curation / freeze (F1.3 / D1)
    "BuildBundleInputs",
    "bundle_id_from_hash",
    "build_figure_corpus_bundle",
    "canonical_serialize",
    "compute_corpus_bundle_integrity_hash",
    "primary_source_from_envelope_chunk",
    "utc_iso_now",
]
