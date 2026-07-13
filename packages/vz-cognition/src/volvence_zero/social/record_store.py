"""Session-held cross-turn store for ToM / common-ground owners (W1.C).

CP-16/17 core: the four ToM owners and the common-ground owner were
rebuilt per turn with no memory of their own records, which made
prediction settlement and PE-weighted promote/retire structurally
impossible. This store (same lifetime pattern as ``SemanticStateStore``)
holds:

- per-slot ToM ``OtherMindRecord`` windows (bounded),
- per-slot pending ``SocialPrediction`` issues awaiting settlement,
- common-ground atom windows + pending common-ground predictions.

Settlement rule (semantic, no keywords, no LLM-as-truth-owner): a
pending prediction is compared against this turn's new evidence for the
same scope via embedding similarity of the typed summaries.

- similarity >= confirm threshold -> CONFIRMED (record confidence up,
  stays ACTIVE)
- similarity <= disconfirm threshold -> DISCONFIRMED
  (ACTIVE -> CONTESTED -> RETIRED on repeat)
- in between -> stays pending; predictions older than the max pending
  age settle STALE.

Settled outcomes are lifted into the existing ``SocialPredictionError``
contract by the owner itself; ``SocialPredictionErrorModule`` forwards
them. The epistemic pressure (disconfirmation magnitude) is what drives
promote/retire — never raw text matching.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from volvence_zero.semantic_embedding import (
    semantic_embedding as _semantic_embedding,
    stub_cosine_similarity as _cosine_similarity,
)
from volvence_zero.social_cognition import (
    CommonGroundAtom,
    OtherMindRecord,
    OtherMindRecordStatus,
    SocialPrediction,
    SocialPredictionError,
    SocialPredictionOutcome,
)

SimilarityFn = Callable[[str, str], float]

TOM_SLOTS: tuple[str, ...] = (
    "belief_about_other",
    "intent_about_other",
    "feeling_about_other",
    "preference_about_other",
)

_RECORD_WINDOW = 12
_ATOM_WINDOW = 12
CONFIRM_SIMILARITY = 0.60
DISCONFIRM_SIMILARITY = 0.40
MAX_PENDING_AGE_TURNS = 6
_CONFIRM_CONFIDENCE_GAIN = 0.10
_DISCONFIRM_CONFIDENCE_LOSS = 0.20


def default_summary_similarity(left: str, right: str) -> float:
    """Unit-scaled semantic similarity via the shared embedding seam."""

    cosine = _cosine_similarity(
        _semantic_embedding(left), _semantic_embedding(right)
    )
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


@dataclass(frozen=True)
class PendingSocialPrediction:
    prediction: SocialPrediction
    source_record_id: str
    issued_turn: int


@dataclass(frozen=True)
class SocialSettlementResult:
    settled_errors: tuple[SocialPredictionError, ...]
    still_pending: tuple[PendingSocialPrediction, ...]
    # (record_id, outcome, error_id) triples for promote/retire
    # transitions and prediction_error_refs lineage.
    outcomes_by_record: tuple[tuple[str, SocialPredictionOutcome, str], ...]


def settle_pending_predictions(
    *,
    pending: tuple[PendingSocialPrediction, ...],
    new_evidence_by_scope: dict[str, tuple[tuple[str, str], ...]],
    turn_index: int,
    owner: str,
    similarity: SimilarityFn,
    confirm_threshold: float = CONFIRM_SIMILARITY,
    disconfirm_threshold: float = DISCONFIRM_SIMILARITY,
    max_pending_age: int = MAX_PENDING_AGE_TURNS,
) -> SocialSettlementResult:
    """Settle pending predictions against this turn's typed evidence.

    ``new_evidence_by_scope`` maps ``scope_id`` to tuples of
    ``(evidence_id, summary)`` produced THIS turn by the same owner.
    Pure function: transitions to the records themselves are applied by
    the caller via ``apply_outcome_to_record``.
    """

    settled: list[SocialPredictionError] = []
    still_pending: list[PendingSocialPrediction] = []
    outcomes: list[tuple[str, SocialPredictionOutcome, str]] = []
    for entry in pending:
        prediction = entry.prediction
        candidates = new_evidence_by_scope.get(prediction.scope_id, ())
        if not candidates:
            if turn_index - entry.issued_turn >= max_pending_age:
                error_id = f"{prediction.prediction_id}:settled:{turn_index}"
                settled.append(
                    SocialPredictionError(
                        error_id=error_id,
                        prediction_id=prediction.prediction_id,
                        kind=prediction.kind,
                        outcome=SocialPredictionOutcome.STALE,
                        magnitude=0.0,
                        owner=owner,
                        scope_kind=prediction.scope_kind,
                        scope_id=prediction.scope_id,
                        evidence=(
                            f"pending_age={turn_index - entry.issued_turn}",
                            f"max_pending_age={max_pending_age}",
                        ),
                    )
                )
                outcomes.append(
                    (
                        entry.source_record_id,
                        SocialPredictionOutcome.STALE,
                        error_id,
                    )
                )
            else:
                still_pending.append(entry)
            continue
        best_evidence_id, best_similarity = max(
            (
                (evidence_id, similarity(prediction.predicted_outcome, summary))
                for evidence_id, summary in candidates
            ),
            key=lambda item: item[1],
        )
        best_similarity = max(0.0, min(1.0, best_similarity))
        if best_similarity >= confirm_threshold:
            outcome = SocialPredictionOutcome.CONFIRMED
            magnitude = max(0.0, min(1.0, 1.0 - best_similarity))
        elif best_similarity <= disconfirm_threshold:
            outcome = SocialPredictionOutcome.DISCONFIRMED
            magnitude = max(0.0, min(1.0, 1.0 - best_similarity))
        else:
            # Ambiguous evidence: keep pending until confirmed,
            # disconfirmed, or stale.
            still_pending.append(entry)
            continue
        error_id = f"{prediction.prediction_id}:settled:{turn_index}"
        settled.append(
            SocialPredictionError(
                error_id=error_id,
                prediction_id=prediction.prediction_id,
                kind=prediction.kind,
                outcome=outcome,
                magnitude=magnitude,
                owner=owner,
                scope_kind=prediction.scope_kind,
                scope_id=prediction.scope_id,
                evidence=(
                    f"settled_by:{best_evidence_id}",
                    f"similarity={best_similarity:.3f}",
                ),
            )
        )
        outcomes.append((entry.source_record_id, outcome, error_id))
    return SocialSettlementResult(
        settled_errors=tuple(settled),
        still_pending=tuple(still_pending),
        outcomes_by_record=tuple(outcomes),
    )


def apply_outcome_to_record(
    record: OtherMindRecord,
    outcome: SocialPredictionOutcome,
    *,
    error_id: str = "",
) -> OtherMindRecord:
    """PE-weighted promote/retire transition for one ToM record.

    Decision table (unit-tested in
    ``tests/test_social_tom_settlement.py``):

    - CONFIRMED: confidence += gain, status stays / returns to ACTIVE
      (a contested record that is re-confirmed is promoted back).
    - DISCONFIRMED while ACTIVE: confidence -= loss, ACTIVE -> CONTESTED.
    - DISCONFIRMED while CONTESTED: CONTESTED -> RETIRED.
    - STALE / UNKNOWN: no transition.

    ``error_id`` (when provided and new) is appended to the record's
    ``prediction_error_refs`` so the promote/retire lineage is auditable
    from the record itself.
    """

    refs = record.prediction_error_refs
    if error_id and error_id not in refs:
        refs = (*refs, error_id)
    if outcome is SocialPredictionOutcome.CONFIRMED:
        return replace(
            record,
            confidence=min(1.0, record.confidence + _CONFIRM_CONFIDENCE_GAIN),
            status=OtherMindRecordStatus.ACTIVE,
            prediction_error_refs=refs,
        )
    if outcome is SocialPredictionOutcome.DISCONFIRMED:
        if record.status is OtherMindRecordStatus.CONTESTED:
            return replace(
                record,
                status=OtherMindRecordStatus.RETIRED,
                prediction_error_refs=refs,
            )
        return replace(
            record,
            confidence=max(0.0, record.confidence - _DISCONFIRM_CONFIDENCE_LOSS),
            status=OtherMindRecordStatus.CONTESTED,
            prediction_error_refs=refs,
        )
    return replace(record, prediction_error_refs=refs)


class SocialRecordStore:
    """Session-held single writer for ToM records / common-ground atoms.

    Owner modules are the only mutators (via their ``process``); the
    store never invents records itself.
    """

    def __init__(self, *, similarity: SimilarityFn | None = None) -> None:
        self._similarity = similarity or default_summary_similarity
        self._tom_records: dict[str, tuple[OtherMindRecord, ...]] = {
            slot: () for slot in TOM_SLOTS
        }
        self._tom_pending: dict[str, tuple[PendingSocialPrediction, ...]] = {
            slot: () for slot in TOM_SLOTS
        }
        self._cg_dyad_atoms: tuple[CommonGroundAtom, ...] = ()
        self._cg_group_atoms: tuple[CommonGroundAtom, ...] = ()
        self._cg_pending: tuple[PendingSocialPrediction, ...] = ()

    @property
    def similarity(self) -> SimilarityFn:
        return self._similarity

    # ----- ToM slots -----

    def tom_records(self, slot: str) -> tuple[OtherMindRecord, ...]:
        self._require_tom_slot(slot)
        return self._tom_records[slot]

    def set_tom_records(
        self, slot: str, records: tuple[OtherMindRecord, ...]
    ) -> None:
        self._require_tom_slot(slot)
        self._tom_records[slot] = records[-_RECORD_WINDOW:]

    def pending_tom_predictions(
        self, slot: str
    ) -> tuple[PendingSocialPrediction, ...]:
        self._require_tom_slot(slot)
        return self._tom_pending[slot]

    def set_pending_tom_predictions(
        self, slot: str, pending: tuple[PendingSocialPrediction, ...]
    ) -> None:
        self._require_tom_slot(slot)
        self._tom_pending[slot] = pending[-_RECORD_WINDOW:]

    def _require_tom_slot(self, slot: str) -> None:
        if slot not in TOM_SLOTS:
            raise ValueError(f"unknown ToM slot {slot!r}")

    # ----- common ground -----

    @property
    def common_ground_dyad_atoms(self) -> tuple[CommonGroundAtom, ...]:
        return self._cg_dyad_atoms

    @property
    def common_ground_group_atoms(self) -> tuple[CommonGroundAtom, ...]:
        return self._cg_group_atoms

    def set_common_ground_atoms(
        self,
        *,
        dyad_atoms: tuple[CommonGroundAtom, ...],
        group_atoms: tuple[CommonGroundAtom, ...],
    ) -> None:
        self._cg_dyad_atoms = dyad_atoms[-_ATOM_WINDOW:]
        self._cg_group_atoms = group_atoms[-_ATOM_WINDOW:]

    @property
    def pending_common_ground_predictions(
        self,
    ) -> tuple[PendingSocialPrediction, ...]:
        return self._cg_pending

    def set_pending_common_ground_predictions(
        self, pending: tuple[PendingSocialPrediction, ...]
    ) -> None:
        self._cg_pending = pending[-_ATOM_WINDOW:]


__all__ = [
    "CONFIRM_SIMILARITY",
    "DISCONFIRM_SIMILARITY",
    "MAX_PENDING_AGE_TURNS",
    "TOM_SLOTS",
    "PendingSocialPrediction",
    "SocialRecordStore",
    "SocialSettlementResult",
    "apply_outcome_to_record",
    "default_summary_similarity",
    "settle_pending_predictions",
]
