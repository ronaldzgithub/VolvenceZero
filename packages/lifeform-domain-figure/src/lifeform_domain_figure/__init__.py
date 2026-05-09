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
immutable :class:`FigureArtifactBundle` consumed at runtime by the
``lifeform-expression`` enforcement layer.

Public surface is added incrementally per the F1-F6 packet sequence
(see ``docs/specs/figure-vertical.md``). Early imports are
intentionally scoped: each packet adds its own modules and re-exports
here so consumers can pin against a stable namespace.
"""

from __future__ import annotations

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
from lifeform_domain_figure.compiler import (
    FigureBundleInputs,
    attach_lora_to_bundle,
    attach_steering_to_bundle,
    build_figure_artifact_bundle,
    build_figure_package,
    build_figure_vitals_bootstrap,
)
from lifeform_domain_figure.coverage_map import (
    CoverageClassification,
    CoverageDecision,
    FigureCoverageMap,
    build_figure_coverage_map,
)
from lifeform_domain_figure.figure_artifact import (
    SCHEMA_VERSION as FIGURE_BUNDLE_SCHEMA_VERSION,
    FigureArtifactBundle,
    bundle_id_from_hash,
    compute_bundle_integrity_hash,
)
from lifeform_domain_figure.lifeform_builder import (
    FigureLifeformBundle,
    build_einstein_lifeform,
    build_figure_lifeform,
)
from lifeform_domain_figure.style_prior import (
    FigureStylePrior,
    build_figure_style_prior,
)
from lifeform_domain_figure.envelope_builder import (
    FigureCorpusSourceBundle,
    FigureIngestionEnvelopeSet,
    build_figure_ingestion_envelope,
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
from lifeform_domain_figure.retrieval_index import (
    FigureRetrievalIndex,
    RetrievalEvidence,
    build_figure_retrieval_index,
)
from lifeform_domain_figure.sample_corpus import synthetic_einstein_corpus


__all__ = [
    # Profile schema (P1.1)
    "FigureBoundaryPrior",
    "FigureDrivePrior",
    "FigureKnowledgeSeed",
    "FigureSignatureCase",
    "FigureStrategyPrior",
    "HistoricalFigureProfile",
    "TimeWindowedView",
    "build_einstein_profile",
    # Corpus ingestion (P1.2)
    "FigureCorpusSourceBundle",
    "FigureIngestionEnvelopeSet",
    "FigureLectureSource",
    "FigureLetterSource",
    "FigureNotebookSource",
    "FigurePaperSource",
    "build_figure_ingestion_envelope",
    "ingest_lectures",
    "ingest_letters",
    "ingest_notebooks",
    "ingest_papers",
    "synthetic_einstein_corpus",
    # Retrieval index (P2.1)
    "FigureRetrievalIndex",
    "RetrievalEvidence",
    "build_figure_retrieval_index",
    # Coverage map (P2.2)
    "CoverageClassification",
    "CoverageDecision",
    "FigureCoverageMap",
    "build_figure_coverage_map",
    # Style prior (P2.3)
    "FigureStylePrior",
    "build_figure_style_prior",
    # Bundle (P2.3)
    "FIGURE_BUNDLE_SCHEMA_VERSION",
    "FigureArtifactBundle",
    "FigureBundleInputs",
    "attach_lora_to_bundle",
    "attach_steering_to_bundle",
    "build_figure_artifact_bundle",
    "build_figure_package",
    "build_figure_vitals_bootstrap",
    "bundle_id_from_hash",
    "compute_bundle_integrity_hash",
    # Lifeform builder (P2.3 / P4.2)
    "FigureLifeformBundle",
    "build_einstein_lifeform",
    "build_figure_lifeform",
]
