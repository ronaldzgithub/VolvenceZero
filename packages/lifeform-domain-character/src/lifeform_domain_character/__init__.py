"""Vertical: reviewed fictional-character bootstrap.

This package is the monorepo-local application layer for "novel character to
lifeform" work. It converts reviewed character profiles into existing Volvence
Zero inputs:

* ``DomainExperiencePackage`` for knowledge, cases, playbook, and boundaries.
* ``VitalsBootstrap`` for the character's drive profile.
* ``IngestionEnvelope`` for replaying source text through the canonical turn
  pipeline.

It does not add a new kernel owner and does not infer behavior from keywords in
novel text.
"""

from __future__ import annotations

from lifeform_domain_character.arcs import build_zhang_wuji_demo_arc
from lifeform_domain_character.behavior_fidelity import (
    BEHAVIOR_FIDELITY_DIMENSIONS,
    BEHAVIOR_FIDELITY_SCHEMA_VERSION,
    BehaviorFidelityCapture,
    BehaviorFidelityComparisonReport,
    BehaviorFidelityEvidenceSource,
    BehaviorFidelityReference,
    BehaviorFidelityReport,
    BehaviorFidelityStimulus,
    ReviewedBehaviorFidelityAssessment,
    behavior_fidelity_capture_from_dict,
    behavior_fidelity_report_from_dict,
    build_scene_behavior_fidelity_inputs,
    capture_behavior_fidelity_async,
    compare_behavior_fidelity_reports,
    review_behavior_fidelity,
    reviewed_behavior_fidelity_assessment_from_dict,
)
from lifeform_domain_character.behavior_fidelity_ablation import (
    BehaviorFidelityAblationArm,
    BehaviorFidelityArmReport,
    BehaviorFidelityCaseObservation,
    BehaviorFidelityCausalAblationReport,
    evaluate_behavior_fidelity_ablation,
)
from lifeform_domain_character.behavior_fidelity_matrix import (
    BEHAVIOR_FIDELITY_MATRIX_SCHEMA_VERSION,
    BehaviorFidelityCaseKind,
    BehaviorFidelityMatrix,
    BehaviorFidelityMatrixCase,
    BehaviorFidelityMatrixThresholds,
    PromotionExpectation,
    load_behavior_fidelity_matrix,
    load_zhang_wuji_action_applicability_matrix,
)
from lifeform_domain_character.behavior_family_portfolio import (
    ActionEvidenceOnlyTextProvider,
    BehaviorFamilyPortfolioReport,
    BehaviorFamilyPromptKind,
    BehaviorFamilyProviderTrace,
    BehaviorFamilyRoutingObservation,
    RealProviderBehaviorEvidenceReport,
    evaluate_behavior_family_portfolio,
    evaluate_real_provider_behavior_evidence,
)
from lifeform_domain_character.compiler import (
    build_character_ingestion_envelope,
    build_character_package,
    build_character_vitals_bootstrap,
)
from lifeform_domain_character.character_package import (
    CHARACTER_ARTIFACT_REF_SCHEMA_VERSION,
    CHARACTER_FIDELITY_EVIDENCE_SCHEMA_VERSION,
    CHARACTER_LORA_REF_SCHEMA_VERSION,
    CHARACTER_PACKAGE_GATE_SCHEMA_VERSION,
    CHARACTER_PACKAGE_MANIFEST_SCHEMA_VERSION,
    CharacterArtifactRef,
    CharacterFidelityEvidence,
    CharacterLoRARef,
    CharacterPackageGateRecord,
    CharacterPackageManifest,
    character_fidelity_evidence_from_json,
    character_package_gate_record_from_json,
    rebind_fidelity_only,
    resolve_artifact_path,
    sha256_path,
    verify_manifest_artifacts,
)
from lifeform_domain_character.first_person import (
    FirstPersonRewriteResult,
    to_first_person,
)
from lifeform_domain_character.lifeform_builder import (
    CharacterLifeformBundle,
    build_character_lifeform,
    build_zhang_wuji_lifeform,
)
from lifeform_domain_character.narrative import NarrativeArc, NarrativeScene
from lifeform_domain_character.chapter_experience import (
    ChapterCoverageKind,
    ChapterLiveThroughLedger,
    CharacterChapterSemanticAdapter,
    CharacterSemanticEvent,
    CharacterSemanticEventBundle,
    ReviewedChapterExperience,
)
from lifeform_domain_character.chapter_replay import (
    ChapterLiveThroughDriver,
    ChapterLiveThroughReport,
    ChapterReplayRecord,
    ChapterSceneBakeEvidence,
)
from lifeform_domain_character.chapter_artifacts import (
    SourceChapter,
    build_review_scaffold,
    read_ledger_json,
    read_text_with_detected_encoding,
    split_source_chapters,
    write_ledger_json,
)
from lifeform_domain_character.replay import (
    ExperientialReplayDriver,
    ReplayReport,
    SceneReplayRecord,
)
from lifeform_domain_character.template import (
    ApplicationOwnerState,
    IncompatibleTemplateVersion,
    LifeformTemplate,
    LifeformTemplateManifest,
    SCHEMA_VERSION as TEMPLATE_SCHEMA_VERSION,
    compute_template_integrity_hash,
    utc_iso_now,
)
from lifeform_domain_character.evolution import (
    DriveShapeEvolution,
    DriveSpecDelta,
    compute_drive_shape_evolution,
)
from lifeform_domain_character.rare_heavy_apply import (
    DriveEvolutionApplyResult,
    GatedDriveSpecDelta,
    apply_drive_evolution_through_gate,
    invert_delta,
)
from lifeform_domain_character.extraction import (
    ChapterLedgerCandidate,
    NarrativeArcCandidate,
    ReviewedProfileCandidate,
    extract_arc_candidate,
    extract_chapter_ledger_candidate,
    extract_profile_candidate,
    load_chapter_live_through_prompt,
    load_chapter_live_through_schema,
    review_arc_candidate,
    review_chapter_ledger,
    review_profile_candidate,
)
from lifeform_domain_character.template_load import (
    RebirthBundle,
    give_birth,
)
from lifeform_domain_character.template_save import (
    SaveLifeformTemplateResult,
    save_lifeform_template,
    vitals_drive_levels_from_session,
)
from lifeform_domain_character.template_adapter import (
    CharacterTemplateAdapter,
    build_character_template_adapter,
)
from lifeform_domain_character.profile import (
    CharacterBoundaryPrior,
    CharacterDrivePrior,
    CharacterKnowledgeSeed,
    CharacterSignatureCase,
    CharacterSoulProfile,
    CharacterStrategyPrior,
)
from lifeform_domain_character.profiles import build_zhang_wuji_profile
from lifeform_domain_character.sample_excerpts import zhang_wuji_long_arc_excerpt

__all__ = [
    "CHARACTER_ARTIFACT_REF_SCHEMA_VERSION",
    "CHARACTER_FIDELITY_EVIDENCE_SCHEMA_VERSION",
    "CHARACTER_LORA_REF_SCHEMA_VERSION",
    "CHARACTER_PACKAGE_GATE_SCHEMA_VERSION",
    "CHARACTER_PACKAGE_MANIFEST_SCHEMA_VERSION",
    "BEHAVIOR_FIDELITY_DIMENSIONS",
    "BEHAVIOR_FIDELITY_MATRIX_SCHEMA_VERSION",
    "BEHAVIOR_FIDELITY_SCHEMA_VERSION",
    "ActionEvidenceOnlyTextProvider",
    "BehaviorFidelityAblationArm",
    "BehaviorFidelityArmReport",
    "BehaviorFidelityCaseKind",
    "BehaviorFidelityCaseObservation",
    "BehaviorFidelityCapture",
    "BehaviorFidelityComparisonReport",
    "BehaviorFidelityCausalAblationReport",
    "BehaviorFidelityEvidenceSource",
    "BehaviorFidelityMatrix",
    "BehaviorFidelityMatrixCase",
    "BehaviorFidelityMatrixThresholds",
    "BehaviorFidelityReference",
    "BehaviorFidelityReport",
    "BehaviorFidelityStimulus",
    "BehaviorFamilyPortfolioReport",
    "BehaviorFamilyPromptKind",
    "BehaviorFamilyProviderTrace",
    "BehaviorFamilyRoutingObservation",
    "PromotionExpectation",
    "RealProviderBehaviorEvidenceReport",
    "CharacterBoundaryPrior",
    "CharacterArtifactRef",
    "CharacterFidelityEvidence",
    "CharacterLoRARef",
    "CharacterPackageGateRecord",
    "CharacterPackageManifest",
    "character_fidelity_evidence_from_json",
    "character_package_gate_record_from_json",
    "ChapterCoverageKind",
    "ChapterLiveThroughDriver",
    "ChapterLiveThroughLedger",
    "ChapterLiveThroughReport",
    "ChapterReplayRecord",
    "ChapterSceneBakeEvidence",
    "ChapterLedgerCandidate",
    "SourceChapter",
    "CharacterDrivePrior",
    "CharacterChapterSemanticAdapter",
    "CharacterKnowledgeSeed",
    "CharacterSemanticEvent",
    "CharacterSemanticEventBundle",
    "CharacterLifeformBundle",
    "CharacterSignatureCase",
    "CharacterSoulProfile",
    "CharacterStrategyPrior",
    "ExperientialReplayDriver",
    "FirstPersonRewriteResult",
    "NarrativeArc",
    "NarrativeScene",
    "ReviewedChapterExperience",
    "ReviewedBehaviorFidelityAssessment",
    "build_review_scaffold",
    "build_scene_behavior_fidelity_inputs",
    "behavior_fidelity_capture_from_dict",
    "behavior_fidelity_report_from_dict",
    "capture_behavior_fidelity_async",
    "compare_behavior_fidelity_reports",
    "evaluate_behavior_fidelity_ablation",
    "evaluate_behavior_family_portfolio",
    "evaluate_real_provider_behavior_evidence",
    "load_behavior_fidelity_matrix",
    "load_zhang_wuji_action_applicability_matrix",
    "review_behavior_fidelity",
    "reviewed_behavior_fidelity_assessment_from_dict",
    "rebind_fidelity_only",
    "resolve_artifact_path",
    "sha256_path",
    "verify_manifest_artifacts",
    "ReplayReport",
    "SceneReplayRecord",
    "to_first_person",
    "build_character_ingestion_envelope",
    "build_character_lifeform",
    "build_character_package",
    "build_character_vitals_bootstrap",
    "build_zhang_wuji_demo_arc",
    "build_zhang_wuji_lifeform",
    "build_zhang_wuji_profile",
    "read_ledger_json",
    "read_text_with_detected_encoding",
    "split_source_chapters",
    "write_ledger_json",
    "zhang_wuji_long_arc_excerpt",
    # Template (Wave T4)
    "ApplicationOwnerState",
    "IncompatibleTemplateVersion",
    "LifeformTemplate",
    "LifeformTemplateManifest",
    "TEMPLATE_SCHEMA_VERSION",
    "compute_template_integrity_hash",
    "utc_iso_now",
    # Template save (Wave T5)
    "SaveLifeformTemplateResult",
    "save_lifeform_template",
    "vitals_drive_levels_from_session",
    # Template load / give_birth (Wave T6)
    "RebirthBundle",
    "give_birth",
    # Browser-chat template adapter (chat-browser template surface)
    "CharacterTemplateAdapter",
    "build_character_template_adapter",
    # LLM extraction (Wave T7 + T8)
    "NarrativeArcCandidate",
    "ReviewedProfileCandidate",
    "extract_arc_candidate",
    "extract_chapter_ledger_candidate",
    "extract_profile_candidate",
    "load_chapter_live_through_prompt",
    "load_chapter_live_through_schema",
    "review_arc_candidate",
    "review_chapter_ledger",
    "review_profile_candidate",
    # Drive evolution (Wave T9)
    "DriveShapeEvolution",
    "DriveSpecDelta",
    "compute_drive_shape_evolution",
    # Rare-heavy apply (Wave T10)
    "DriveEvolutionApplyResult",
    "GatedDriveSpecDelta",
    "apply_drive_evolution_through_gate",
    "invert_delta",
]
