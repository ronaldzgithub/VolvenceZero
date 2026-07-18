"""Owner prediction signal contract (CP-12, AGI-uplift Phase 3).

One shared immutable signal that lets semantic / social owners publish typed
pre-action predictions about their OWN state, and lets the single
``prediction_error`` owner settle them — without a parallel PE owner and
without consumers reading owner-internal fields or raw text.

SSOT split (oss-relationship-representation-standard.md, Phase A1):

* The *representation* — :class:`OwnerPredictionKind` and
  :class:`OwnerPredictionSignal` — lives in ``companion_standard`` (the
  public Relationship Representation Standard) and is re-exported here so
  every existing ``volvence_zero.owner_prediction`` import keeps working.
* The *mechanism* — :func:`settle_owner_prediction` and
  :class:`OwnerPredictionSettlement` — stays private in this module.

Ownership rules (R8 / prediction-error-loop.md):

* The SOURCE owner builds the signal, authors its semantics (``description``)
  and later settles it by attaching ``settled_vector`` + ``outcome_evidence``
  (:func:`settle_owner_prediction`). Nobody else may fabricate a settlement.
* Only ``PredictionErrorModule`` computes mismatch and constructs
  :class:`OwnerPredictionSettlement`. Lifters / renderers forward; they never
  compute a second error.
* ``kind`` is a CLOSED enum. Adding a kind is a contract change that goes
  through ``companion_standard.owner_prediction``, never a free-form string.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from companion_standard.owner_prediction import (
    OwnerPredictionKind,
    OwnerPredictionSignal,
)

_ALLOWED_TRACKS = ("world", "self", "shared")


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


def _require_unit_interval(field_name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1], got {value!r}")


__all__ = [
    "OwnerPredictionKind",
    "OwnerPredictionSettlement",
    "OwnerPredictionSignal",
    "settle_owner_prediction",
]
