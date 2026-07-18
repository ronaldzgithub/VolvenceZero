# Copyright 2026 Companion Standard Contributors
# Licensed under the Apache License, Version 2.0.

"""Owner prediction signal — the typed pre-action prediction record.

Part of the Relationship Representation Standard: semantic / social owners
publish typed predictions about their OWN next-turn readout, embedded in
their snapshot values as ``owner_prediction_signals``.

This module owns the *representation* only: the closed ``kind`` vocabulary
and the immutable signal record. How predictions are settled and how
mismatch is computed (``OwnerPredictionSettlement``, the prediction-error
module) is runtime mechanism and deliberately does not ship in the standard.

``kind`` is a CLOSED enum. Adding a kind is a contract change that goes
through this file, never a free-form string.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

_MAX_VECTOR_DIM = 8
_ALLOWED_TRACKS = ("world", "self", "shared")


class OwnerPredictionKind(str, Enum):
    """Closed set of owner prediction classes (one per publishing owner)."""

    # First wave (publishers wired):
    RELATIONSHIP_TRUST_TRAJECTORY = "relationship_trust_trajectory"
    COMMITMENT_FOLLOW_THROUGH = "commitment_follow_through"
    BOUNDARY_CONSENT_STABILITY = "boundary_consent_stability"
    EXECUTION_RESULT_SUCCESS = "execution_result_success"
    GOAL_VALUE_ALIGNMENT = "goal_value_alignment"
    # Second wave (publishers wired, CP-12 / GAP-05):
    PLAN_INTENT_PROGRESS = "plan_intent_progress"
    OPEN_LOOP_CLOSURE = "open_loop_closure"
    BELIEF_ASSUMPTION_STABILITY = "belief_assumption_stability"
    USER_MODEL_PACING = "user_model_pacing"


@dataclass(frozen=True)
class OwnerPredictionSignal:
    """A typed, compact, owner-authored prediction about the owner's own state.

    ``predicted_vector`` is the owner's compact forecast of its next-turn
    readout (each component in [0, 1], at most ``_MAX_VECTOR_DIM`` dims).
    ``settled_vector`` stays ``None`` until the source owner observes the
    actual next-turn readout and settles.
    """

    signal_id: str
    prediction_id: str
    source_owner: str
    source_slot: str
    track: str
    kind: OwnerPredictionKind
    predicted_vector: tuple[float, ...]
    confidence: float
    description: str
    source_turn_index: int
    evidence: tuple[str, ...] = ()
    outcome_evidence: tuple[str, ...] = ()
    settled_vector: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        _require_non_empty("signal_id", self.signal_id)
        _require_non_empty("prediction_id", self.prediction_id)
        _require_non_empty("source_owner", self.source_owner)
        _require_non_empty("source_slot", self.source_slot)
        _require_non_empty("description", self.description)
        if self.track not in _ALLOWED_TRACKS:
            raise ValueError(
                f"track must be one of {_ALLOWED_TRACKS}, got {self.track!r}"
            )
        _require_unit_vector("predicted_vector", self.predicted_vector)
        _require_unit_interval("confidence", self.confidence)
        if self.source_turn_index < 0:
            raise ValueError("source_turn_index must be non-negative")
        _require_non_empty_items("evidence", self.evidence)
        _require_non_empty_items("outcome_evidence", self.outcome_evidence)
        if self.settled_vector is not None:
            _require_unit_vector("settled_vector", self.settled_vector)
            if len(self.settled_vector) != len(self.predicted_vector):
                raise ValueError(
                    "settled_vector must match predicted_vector dimensionality"
                )
            if not self.outcome_evidence:
                raise ValueError(
                    "a settled signal must carry non-empty outcome_evidence"
                )

    @property
    def settled(self) -> bool:
        return self.settled_vector is not None


def _require_non_empty(field_name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _require_non_empty_items(field_name: str, values: tuple[str, ...]) -> None:
    for value in values:
        if not value.strip():
            raise ValueError(f"{field_name} entries must be non-empty")


def _require_unit_interval(field_name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1], got {value!r}")


def _require_unit_vector(field_name: str, values: tuple[float, ...]) -> None:
    if not values:
        raise ValueError(f"{field_name} must contain at least one component")
    if len(values) > _MAX_VECTOR_DIM:
        raise ValueError(
            f"{field_name} must have at most {_MAX_VECTOR_DIM} components, "
            f"got {len(values)}"
        )
    for component in values:
        if component < 0.0 or component > 1.0:
            raise ValueError(
                f"{field_name} components must be in [0, 1], got {component!r}"
            )


__all__ = [
    "OwnerPredictionKind",
    "OwnerPredictionSignal",
]
