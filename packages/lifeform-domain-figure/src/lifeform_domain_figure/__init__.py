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
    CaptureMethod,
    DedupReport,
    DuplicateGroup,
    FigureLectureSource,
    FigureLetterSource,
    FigureNotebookSource,
    FigurePaperSource,
    LegalClearance,
    LocatorKind,
    LocatorOffset,
    ParsedLocator,
    SourceProvenance,
    compute_dedup_report,
    fingerprint_provenance,
    ingest_lectures,
    ingest_letters,
    ingest_notebooks,
    ingest_papers,
    parse_locator,
)
from lifeform_domain_figure.corpus.archives import (
    ArchiveFetcher,
    ArchiveFetchResult,
    CPAEDocumentKind,
    CPAEPayload,
    CTPPayload,
    GutenbergPayload,
    InternetArchivePayload,
    WikisourcePayload,
    cpae_to_letter_source,
    cpae_to_paper_source,
    ctp_to_paper_source,
    gutenberg_to_paper_source,
    internet_archive_to_lecture_source,
    internet_archive_to_paper_source,
    offline_archive_fetcher,
    wikisource_to_lecture_source,
    wikisource_to_paper_source,
)
from lifeform_domain_figure.corpus.loaders import (
    CuratedCorpusBundle,
    CuratedSourceMetadata,
    load_curated_corpus_from_cleaning_store,
    load_curated_metadata_jsonl,
)
from lifeform_domain_figure.verification import (
    VerificationGateError,
    VerificationLedger,
)
from lifeform_domain_figure.metadata import (
    AuthoredWorkSummary,
    CrossrefClient,
    CrossrefWorkPayload,
    DomainCoverageHint,
    FigureLifespan,
    MetadataDigest,
    MetadataSource,
    OpenAlexClient,
    OpenAlexWorkPayload,
    SEPClient,
    SEPEntryPayload,
    TimeWindowHint,
    WikidataClient,
    WikidataPersonPayload,
    aggregate_metadata,
    build_time_window_hints_from_lifespan,
    crossref_to_authored_work,
    enrich_profile_with_metadata,
    offline_crossref_client,
    offline_openalex_client,
    offline_sep_client,
    offline_wikidata_client,
    openalex_to_authored_work,
    openalex_to_domain_hints,
    sep_to_domain_hints,
    wikidata_to_lifespan,
    wikidata_to_time_window_hints,
)
from lifeform_domain_figure.audit import (
    AUDIT_SCHEMA_VERSION,
    FigureBakeAction,
    FigureBakeAuditRecord,
    FigureGateDecisionLabel,
    build_audit_record,
    find_previous_audit_for_bundle,
    read_audit_records,
    write_audit,
)
from lifeform_domain_figure.bundle_io import (
    BundleManifest,
    list_figure_bundles,
    load_figure_bundle,
    save_figure_bundle,
)
from lifeform_domain_figure.compiler import (
    FigureBundleInputs,
    attach_lora_to_bundle,
    attach_presence_to_bundle,
    attach_steering_to_bundle,
    build_figure_artifact_bundle,
    build_figure_package,
    build_figure_vitals_bootstrap,
)
from lifeform_domain_figure.presence_affordance import (
    CONSENT_LIKENESS_RENDER,
    CONSENT_VOICE_CLONE,
    PRESENCE_AFFORDANCE_DESCRIPTORS,
    TALKING_HEAD_VIDEO_DESCRIPTOR,
)
from lifeform_domain_figure.presence_artifact import (
    SCHEMA_VERSION as FIGURE_PRESENCE_SCHEMA_VERSION,
    FigurePresenceArtifact,
    PresenceEngineId,
    build_figure_presence_artifact,
    compute_presence_integrity_hash,
    hash_consent_token,
    presence_artifact_id_from_hash,
    presence_fingerprint,
)
from lifeform_domain_figure.presence_lora_register import (
    PresenceLoraRegistration,
    build_registration_from_artifact,
    register_lora_into_presence,
    revoke_lora_from_presence,
)
from lifeform_domain_figure.contrast_set import (
    FigureContrastPair,
    FigureContrastSet,
    build_einstein_contrast_set,
)
from lifeform_domain_figure.coverage_map import (
    CoverageClassification,
    CoverageDecision,
    FigureCoverageMap,
    build_figure_coverage_map,
)
from lifeform_domain_figure.figure_artifact import (
    SCHEMA_VERSION as FIGURE_BUNDLE_SCHEMA_VERSION,
    AuthorCompanionBundle,
    FigureArtifactBundle,
    bundle_id_from_hash,
    compute_bundle_integrity_hash,
)
from lifeform_domain_figure.lifeform_builder import (
    FigureLifeformBundle,
    build_einstein_lifeform,
    build_figure_lifeform,
    build_figure_lifeform_from_bundle,
)
from lifeform_domain_figure.gate_apply import (
    GatedPersonaLoRAProposal,
    PersonaLoRAApplyResult,
    apply_persona_lora_through_gate,
)
from lifeform_domain_figure.lora_artifact import (
    SCHEMA_VERSION as LORA_ARTIFACT_SCHEMA_VERSION,
    FigureLoRAArtifact,
    LoRABakeBackend,
    compute_lora_integrity_hash,
)
from lifeform_domain_figure.lora_bake_peft import (
    PEFTLoRABakeBackend,
    PEFTLoRAConfig,
)
from lifeform_domain_figure.lora_bake_synthetic import (
    SyntheticLoRABakeBackend,
    attach_baked_lora,
)
from lifeform_domain_figure.lora_data_prep import (
    SCHEMA_VERSION as LORA_TRAINING_PLAN_SCHEMA_VERSION,
    LoRATrainingExample,
    LoRATrainingPlan,
    PersonaLoRAProposal,
    build_lora_training_plan,
    build_persona_lora_proposal,
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
from lifeform_domain_figure.profiles import (
    build_einstein_profile,
    build_family_profile_from_json,
    build_generic_profile_from_json,
    build_lu_xun_profile,
    build_myriad_profile_from_json,
    load_family_profile_file,
    load_generic_profile_file,
    load_myriad_profile_file,
    load_profile,
)
from lifeform_domain_figure.retrieval_index import (
    FigureRetrievalIndex,
    RetrievalEvidence,
    build_figure_retrieval_index,
)
from lifeform_domain_figure.sample_corpus import (
    synthetic_corpus_from_profile,
    synthetic_einstein_corpus,
)
from lifeform_domain_figure.steering_bake import (
    SCHEMA_VERSION as STEERING_SET_SCHEMA_VERSION,
    FigureSteeringSet,
    GatedSteeringProposal,
    SteeringApplyResult,
    SteeringVector,
    apply_steering_through_gate,
    attach_baked_steering,
    bake_steering_set,
)
from lifeform_domain_figure.steering_data_prep import (
    SCHEMA_VERSION as STEERING_PLAN_SCHEMA_VERSION,
    FigureSteeringTrainingPlan,
    SteeringTrainingPair,
    build_steering_training_plan,
)


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
    "build_lu_xun_profile",
    # Dynamic profile loaders (family / myriad / generic persona)
    "build_family_profile_from_json",
    "build_generic_profile_from_json",
    "build_myriad_profile_from_json",
    "load_family_profile_file",
    "load_generic_profile_file",
    "load_myriad_profile_file",
    "load_profile",
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
    "synthetic_corpus_from_profile",
    "synthetic_einstein_corpus",
    # D2 provenance / dedupe / citation parser
    "CaptureMethod",
    "LegalClearance",
    "SourceProvenance",
    "fingerprint_provenance",
    "DedupReport",
    "DuplicateGroup",
    "compute_dedup_report",
    "LocatorKind",
    "LocatorOffset",
    "ParsedLocator",
    "parse_locator",
    # D3 archive adapter facades
    "ArchiveFetcher",
    "ArchiveFetchResult",
    "CPAEDocumentKind",
    "CPAEPayload",
    "GutenbergPayload",
    "InternetArchivePayload",
    "WikisourcePayload",
    "CTPPayload",
    "cpae_to_letter_source",
    "cpae_to_paper_source",
    "ctp_to_paper_source",
    "gutenberg_to_paper_source",
    "internet_archive_to_lecture_source",
    "internet_archive_to_paper_source",
    "offline_archive_fetcher",
    # Curated corpus loader (Wave J)
    "CuratedCorpusBundle",
    "CuratedSourceMetadata",
    "load_curated_corpus_from_cleaning_store",
    "load_curated_metadata_jsonl",
    # L2 verification surfaces re-exported for CLI curated path (Wave J)
    "VerificationGateError",
    "VerificationLedger",
    "wikisource_to_lecture_source",
    "wikisource_to_paper_source",
    # D4 metadata adapters
    "AuthoredWorkSummary",
    "CrossrefClient",
    "CrossrefWorkPayload",
    "DomainCoverageHint",
    "FigureLifespan",
    "MetadataDigest",
    "MetadataSource",
    "OpenAlexClient",
    "OpenAlexWorkPayload",
    "SEPClient",
    "SEPEntryPayload",
    "TimeWindowHint",
    "WikidataClient",
    "WikidataPersonPayload",
    "aggregate_metadata",
    "build_time_window_hints_from_lifespan",
    "crossref_to_authored_work",
    "enrich_profile_with_metadata",
    "offline_crossref_client",
    "offline_openalex_client",
    "offline_sep_client",
    "offline_wikidata_client",
    "openalex_to_authored_work",
    "openalex_to_domain_hints",
    "sep_to_domain_hints",
    "wikidata_to_lifespan",
    "wikidata_to_time_window_hints",
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
    "AuthorCompanionBundle",
    "FigureArtifactBundle",
    "FigureBundleInputs",
    "attach_lora_to_bundle",
    "attach_presence_to_bundle",
    "attach_steering_to_bundle",
    "build_figure_artifact_bundle",
    "build_figure_package",
    "build_figure_vitals_bootstrap",
    "bundle_id_from_hash",
    "compute_bundle_integrity_hash",
    # L0 visual presence (talking-head metadata)
    "CONSENT_LIKENESS_RENDER",
    "CONSENT_VOICE_CLONE",
    "FIGURE_PRESENCE_SCHEMA_VERSION",
    "FigurePresenceArtifact",
    "PRESENCE_AFFORDANCE_DESCRIPTORS",
    "PresenceEngineId",
    "TALKING_HEAD_VIDEO_DESCRIPTOR",
    "build_figure_presence_artifact",
    "compute_presence_integrity_hash",
    "hash_consent_token",
    "presence_artifact_id_from_hash",
    "presence_fingerprint",
    # Presence LoRA fingerprint registration (R4-7)
    "PresenceLoraRegistration",
    "build_registration_from_artifact",
    "register_lora_into_presence",
    "revoke_lora_from_presence",
    # Lifeform builder (P2.3 / P4.2)
    "FigureLifeformBundle",
    "build_einstein_lifeform",
    "build_figure_lifeform",
    "build_figure_lifeform_from_bundle",
    # Contrast set + steering data prep (P5.1)
    "FigureContrastPair",
    "FigureContrastSet",
    "FigureSteeringTrainingPlan",
    "STEERING_PLAN_SCHEMA_VERSION",
    "SteeringTrainingPair",
    "build_einstein_contrast_set",
    "build_steering_training_plan",
    # Steering bake (P5.2)
    "FigureSteeringSet",
    "GatedSteeringProposal",
    "STEERING_SET_SCHEMA_VERSION",
    "SteeringApplyResult",
    "SteeringVector",
    "apply_steering_through_gate",
    "attach_baked_steering",
    "bake_steering_set",
    # LoRA data prep + proposal (P6.1)
    "LORA_TRAINING_PLAN_SCHEMA_VERSION",
    "LoRATrainingExample",
    "LoRATrainingPlan",
    "PersonaLoRAProposal",
    "build_lora_training_plan",
    "build_persona_lora_proposal",
    # LoRA artifact + bake backends (P6.2)
    "FigureLoRAArtifact",
    "LORA_ARTIFACT_SCHEMA_VERSION",
    "LoRABakeBackend",
    "PEFTLoRABakeBackend",
    "PEFTLoRAConfig",
    "SyntheticLoRABakeBackend",
    "attach_baked_lora",
    "compute_lora_integrity_hash",
    # Persona LoRA gate apply (P6.3)
    "GatedPersonaLoRAProposal",
    "PersonaLoRAApplyResult",
    "apply_persona_lora_through_gate",
    # Bundle persistence (#23 CLI)
    "BundleManifest",
    "list_figure_bundles",
    "load_figure_bundle",
    "save_figure_bundle",
    # Bake / gate / rollback audit log (#23 CLI)
    "AUDIT_SCHEMA_VERSION",
    "FigureBakeAction",
    "FigureBakeAuditRecord",
    "FigureGateDecisionLabel",
    "build_audit_record",
    "find_previous_audit_for_bundle",
    "read_audit_records",
    "write_audit",
]
