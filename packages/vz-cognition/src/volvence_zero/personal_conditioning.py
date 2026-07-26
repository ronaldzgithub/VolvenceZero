from __future__ import annotations

from typing import Any, Mapping

from volvence_zero.personal_conditioning_contracts import (
    PERSONAL_CONDITIONING_SCHEMA_VERSION,
    PERSONAL_CONDITIONING_VECTOR_LABELS,
    PersonalConditioningSnapshot,
)
from volvence_zero.personal_conditioning_rendering import (
    render_personal_conditioning_statement,
)
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel, stable_value_hash
from volvence_zero.semantic_state import (
    BoundaryConsentSnapshot,
    GoalValueSnapshot,
    RelationshipStateSnapshot,
    UserModelSnapshot,
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class PersonalConditioningModule(RuntimeModule[PersonalConditioningSnapshot]):
    """Compile owner-published human state into a substrate-safe vector."""

    slot_name = "personal_conditioning"
    owner = "PersonalConditioningModule"
    value_type = PersonalConditioningSnapshot
    dependencies = (
        "user_model",
        "relationship_state",
        "goal_value",
        "boundary_consent",
    )
    default_wiring_level = WiringLevel.SHADOW

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[PersonalConditioningSnapshot]:
        user = upstream["user_model"]
        relationship = upstream["relationship_state"]
        goal = upstream["goal_value"]
        boundary = upstream["boundary_consent"]
        if not isinstance(user.value, UserModelSnapshot):
            raise TypeError("personal_conditioning requires UserModelSnapshot.")
        if not isinstance(relationship.value, RelationshipStateSnapshot):
            raise TypeError(
                "personal_conditioning requires RelationshipStateSnapshot."
            )
        if not isinstance(goal.value, GoalValueSnapshot):
            raise TypeError("personal_conditioning requires GoalValueSnapshot.")
        if not isinstance(boundary.value, BoundaryConsentSnapshot):
            raise TypeError(
                "personal_conditioning requires BoundaryConsentSnapshot."
            )

        coverage_flags = (
            bool(
                user.value.stable_preferences
                or user.value.working_style_hints
                or user.value.sensitive_boundaries
                or user.value.durable_goals
            ),
            bool(
                relationship.value.rapport_signals
                or relationship.value.relational_tensions
                or relationship.value.relationship_age_turns
            ),
            bool(
                goal.value.explicit_goals
                or goal.value.value_priorities
                or goal.value.tradeoff_notes
            ),
            bool(
                boundary.value.granted_consents
                or boundary.value.missing_consents
                or boundary.value.denied_boundaries
            ),
        )
        coverage = sum(float(flag) for flag in coverage_flags) / len(coverage_flags)
        is_cold_start = coverage == 0.0
        state_vector = (
            _clamp(user.value.stability_score),
            _clamp(user.value.overwhelm_pattern_strength),
            _clamp(user.value.control_signal),
            _clamp(relationship.value.trust_level),
            _clamp(relationship.value.continuity_level),
            _clamp(relationship.value.repair_need),
            _clamp(relationship.value.emotional_load),
            _clamp(relationship.value.attunement_gap),
            _clamp(goal.value.alignment_score),
            _clamp(goal.value.value_conflict),
            _clamp(goal.value.decision_readiness),
            _clamp(goal.value.reversibility_need),
            _clamp(boundary.value.compliance_score),
            _clamp(boundary.value.autonomy_risk),
            _clamp(boundary.value.consent_clarity),
            _clamp(boundary.value.overreach_risk),
        )
        if is_cold_start:
            state_vector = tuple(0.0 for _ in PERSONAL_CONDITIONING_VECTOR_LABELS)
        source_versions = (
            ("user_model", user.version),
            ("relationship_state", relationship.version),
            ("goal_value", goal.version),
            ("boundary_consent", boundary.version),
        )
        source_fingerprint = stable_value_hash(
            (
                source_versions,
                stable_value_hash(user.value),
                stable_value_hash(relationship.value),
                stable_value_hash(goal.value),
                stable_value_hash(boundary.value),
            )
        )
        confidence = 0.0 if is_cold_start else _clamp(
            coverage
            * (
                0.35
                + 0.25 * user.value.stability_score
                + 0.20 * relationship.value.continuity_level
                + 0.20 * boundary.value.consent_clarity
            )
        )
        rendered_statement = render_personal_conditioning_statement(
            state_vector=state_vector,
            vector_labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
            confidence=confidence,
            is_cold_start=is_cold_start,
        )
        return self.publish(
            PersonalConditioningSnapshot(
                schema_version=PERSONAL_CONDITIONING_SCHEMA_VERSION,
                state_vector=state_vector,
                vector_labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
                source_versions=source_versions,
                source_fingerprint=source_fingerprint,
                confidence=confidence,
                is_cold_start=is_cold_start,
                description=(
                    "Personal conditioning compiled from four typed semantic "
                    f"owners; coverage={coverage:.2f} confidence={confidence:.2f} "
                    f"cold_start={is_cold_start}."
                ),
                rendered_statement=rendered_statement,
            )
        )

    async def process_standalone(
        self, **kwargs: Any
    ) -> Snapshot[PersonalConditioningSnapshot]:
        raise NotImplementedError(
            "PersonalConditioningModule requires typed semantic owner snapshots."
        )


__all__ = ["PersonalConditioningModule"]
