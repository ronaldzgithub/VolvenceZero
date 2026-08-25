from __future__ import annotations

from dataclasses import dataclass


PERSONAL_CONDITIONING_SCHEMA_VERSION = "personal-conditioning.v1"
PERSONAL_CONDITIONING_VECTOR_LABELS = (
    "user_stability",
    "user_overwhelm",
    "user_control",
    "relationship_trust",
    "relationship_continuity",
    "relationship_repair_need",
    "relationship_emotional_load",
    "relationship_attunement_gap",
    "goal_alignment",
    "goal_value_conflict",
    "goal_decision_readiness",
    "goal_reversibility_need",
    "boundary_compliance",
    "boundary_autonomy_risk",
    "boundary_consent_clarity",
    "boundary_overreach_risk",
)


@dataclass(frozen=True)
class PersonalConditioningSnapshot:
    """Auditable, bounded user-state conditioning for substrate prefill.

    The vector contains only typed owner readouts. It never contains raw
    dialogue, profile prose, or memory text, so substrate consumers do not
    become second semantic owners.

    ``rendered_statement`` is the owner-produced natural-language form of
    the *same* typed readout (State KV arm B-prime and the distillation
    teacher context). It is derived exclusively from the labelled
    coordinates, confidence, and coverage of this snapshot -- never from
    semantic records or raw profile text -- so its information content
    and privacy posture match the latent vector exactly.
    """

    schema_version: str
    state_vector: tuple[float, ...]
    vector_labels: tuple[str, ...]
    source_versions: tuple[tuple[str, int], ...]
    source_fingerprint: str
    confidence: float
    is_cold_start: bool
    description: str
    rendered_statement: str = ""
    # State KV P5-c: owner-published readout of the bounded credit-driven
    # confidence adjustment currently applied (ACTIVE) or merely computed
    # (SHADOW). Separate from ``confidence`` so the evidence-derived base
    # and the credit-learned drift stay auditable independently. Additive
    # with a default, so the frozen v1 coordinate contract is unchanged.
    credit_confidence_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.schema_version != PERSONAL_CONDITIONING_SCHEMA_VERSION:
            raise ValueError(
                "PersonalConditioningSnapshot schema_version must be "
                f"{PERSONAL_CONDITIONING_SCHEMA_VERSION!r}."
            )
        if self.vector_labels != PERSONAL_CONDITIONING_VECTOR_LABELS:
            raise ValueError(
                "PersonalConditioningSnapshot vector_labels must match the "
                "frozen personal-conditioning.v1 coordinate contract."
            )
        if len(self.state_vector) != len(self.vector_labels):
            raise ValueError(
                "PersonalConditioningSnapshot state_vector length must match "
                "vector_labels."
            )
        if any(not 0.0 <= value <= 1.0 for value in self.state_vector):
            raise ValueError(
                "PersonalConditioningSnapshot state_vector values must be in [0, 1]."
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "PersonalConditioningSnapshot confidence must be in [0, 1]."
            )
        if self.is_cold_start and (
            self.confidence != 0.0 or any(value != 0.0 for value in self.state_vector)
        ):
            raise ValueError(
                "Cold-start personal conditioning must have zero confidence "
                "and an all-zero vector."
            )
        if self.is_cold_start and self.rendered_statement:
            raise ValueError(
                "Cold-start personal conditioning must not carry a rendered "
                "statement: there is no evidence to state."
            )
        if not self.source_fingerprint:
            raise ValueError(
                "PersonalConditioningSnapshot source_fingerprint must be non-empty."
            )
        if not -1.0 <= self.credit_confidence_delta <= 1.0:
            raise ValueError(
                "PersonalConditioningSnapshot credit_confidence_delta must "
                "be in [-1, 1]."
            )
        if self.is_cold_start and self.credit_confidence_delta != 0.0:
            raise ValueError(
                "Cold-start personal conditioning must not carry a credit "
                "confidence delta: no bank was live to earn credit."
            )


__all__ = [
    "PERSONAL_CONDITIONING_SCHEMA_VERSION",
    "PERSONAL_CONDITIONING_VECTOR_LABELS",
    "PersonalConditioningSnapshot",
]
