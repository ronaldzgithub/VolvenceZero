"""Owner prediction signal contract (CP-12, AGI-uplift Phase 3).

One shared immutable signal that lets semantic / social owners publish typed
pre-action predictions about their OWN state, and lets the single
``prediction_error`` owner settle them — without a parallel PE owner and
without consumers reading owner-internal fields or raw text.

Ownership rules (R8 / prediction-error-loop.md):

* The SOURCE owner builds the signal, authors its semantics (``description``)
  and later settles it by attaching ``settled_vector`` + ``outcome_evidence``
  (:func:`settle_owner_prediction`). Nobody else may fabricate a settlement.
* Only ``PredictionErrorModule`` computes mismatch and constructs
  :class:`OwnerPredictionSettlement`. Lifters / renderers forward; they never
  compute a second error.
* ``kind`` is a CLOSED enum. Adding a kind is a contract change that goes
  through this file, never a free-form string.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    actual next-turn readout and settles via :func:`settle_owner_prediction`.
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


def settle_owner_prediction(
    signal: OwnerPredictionSignal,
    *,
    settled_vector: tuple[float, ...],
    outcome_evidence: tuple[str, ...],
) -> OwnerPredictionSignal:
    """Source-owner-only helper: attach the observed outcome to a signal.

    Re-runs full contract validation via the frozen dataclass so an invalid
    settlement fails loudly at the owner, not downstream.
    """

    if signal.settled:
        raise ValueError(
            f"signal {signal.signal_id!r} is already settled; settlement is one-shot"
        )
    return replace(
        signal,
        settled_vector=settled_vector,
        outcome_evidence=outcome_evidence,
    )


@dataclass(frozen=True)
class OwnerPredictionSettlement:
    """Mismatch record for one settled signal.

    ONLY ``PredictionErrorModule`` constructs this. ``mismatch_magnitude``
    is the mean absolute component difference between predicted and settled
    vectors, clamped to [0, 1].
    """

    prediction_id: str
    source_owner: str
    source_slot: str
    track: str
    kind: OwnerPredictionKind
    mismatch_magnitude: float
    confidence: float
    settled_turn_index: int
    description: str

    def __post_init__(self) -> None:
        _require_non_empty("prediction_id", self.prediction_id)
        _require_non_empty("source_owner", self.source_owner)
        _require_non_empty("source_slot", self.source_slot)
        _require_non_empty("description", self.description)
        if self.track not in _ALLOWED_TRACKS:
            raise ValueError(
                f"track must be one of {_ALLOWED_TRACKS}, got {self.track!r}"
            )
        _require_unit_interval("mismatch_magnitude", self.mismatch_magnitude)
        _require_unit_interval("confidence", self.confidence)
        if self.settled_turn_index < 0:
            raise ValueError("settled_turn_index must be non-negative")


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
    "OwnerPredictionSettlement",
    "OwnerPredictionSignal",
    "settle_owner_prediction",
]
