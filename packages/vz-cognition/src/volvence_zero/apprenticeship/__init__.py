"""Reliable-apprenticeship alignment package.

Owner of the ``apprenticeship_alignment`` slot: compares operator
guidance against the AI's current public cognition in real time using
reliable active apprenticeship learning (Hanneke, Yang, Wang & Song,
ALT 2025) — version space + agreement/disagreement (reliability) +
eluder-style surprise + Massart/Tsybakov noise gating, with a
CLAIRE-style mismatch taxonomy and AGM minimal-change belief revision.

See ``docs/specs/apprenticeship-alignment.md``.
"""

from __future__ import annotations

from volvence_zero.apprenticeship.contracts import (
    ApprenticeshipAlignmentSnapshot,
    ConstraintLevel,
    ContradictionFinding,
    IntentConstraint,
    MismatchRef,
    MismatchType,
    ReliabilityState,
    VersionSpaceStatus,
)
from volvence_zero.apprenticeship.core import (
    ApprenticeshipAlignmentModule,
    ApprenticeshipThresholds,
    GuidanceConstraintExtractor,
    HolisticGuidanceConstraintExtractor,
    LLMGuidanceConstraintExtractor,
    MappingConstraintExtractor,
    apply_apprenticeship_revisions,
    build_intent_constraint,
    load_apprenticeship_prompt_template,
    reconcile_guidance,
)
from volvence_zero.apprenticeship.active_learning import (
    GATE4_ACTIVE_POLICIES,
    BoundedBinaryAlignmentReadout,
    Gate4FeedbackCandidate,
    random_feedback_order,
    select_active_candidate,
)

__all__ = [
    "ApprenticeshipAlignmentModule",
    "ApprenticeshipAlignmentSnapshot",
    "ApprenticeshipThresholds",
    "BoundedBinaryAlignmentReadout",
    "ConstraintLevel",
    "ContradictionFinding",
    "GuidanceConstraintExtractor",
    "GATE4_ACTIVE_POLICIES",
    "Gate4FeedbackCandidate",
    "HolisticGuidanceConstraintExtractor",
    "IntentConstraint",
    "LLMGuidanceConstraintExtractor",
    "MappingConstraintExtractor",
    "MismatchRef",
    "MismatchType",
    "ReliabilityState",
    "VersionSpaceStatus",
    "apply_apprenticeship_revisions",
    "build_intent_constraint",
    "load_apprenticeship_prompt_template",
    "reconcile_guidance",
    "random_feedback_order",
    "select_active_candidate",
]
