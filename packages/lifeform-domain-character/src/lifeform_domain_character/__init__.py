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
from lifeform_domain_character.compiler import (
    build_character_ingestion_envelope,
    build_character_package,
    build_character_vitals_bootstrap,
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
    NarrativeArcCandidate,
    ReviewedProfileCandidate,
    extract_arc_candidate,
    extract_profile_candidate,
    review_arc_candidate,
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
    "CharacterBoundaryPrior",
    "CharacterDrivePrior",
    "CharacterKnowledgeSeed",
    "CharacterLifeformBundle",
    "CharacterSignatureCase",
    "CharacterSoulProfile",
    "CharacterStrategyPrior",
    "ExperientialReplayDriver",
    "FirstPersonRewriteResult",
    "NarrativeArc",
    "NarrativeScene",
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
    "extract_profile_candidate",
    "review_arc_candidate",
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
