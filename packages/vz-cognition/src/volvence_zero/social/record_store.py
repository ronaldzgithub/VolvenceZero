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

from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Callable

from volvence_zero.owner_hydration import (
    HydrationOwnerMismatchError,
    HydrationPayloadInvalidError,
    HydrationVersionMismatchError,
    OwnerPersistenceSnapshot,
)
from volvence_zero.semantic_embedding import (
    semantic_embedding as _semantic_embedding,
    stub_cosine_similarity as _cosine_similarity,
)
from volvence_zero.social_cognition import (
    CommonGroundAtom,
    OtherMindRecord,
    OtherMindRecordKind,
    OtherMindRecordStatus,
    PreferenceActionOutcomeEvidence,
    PreferenceActionOutcomeMutationOperation,
    PreferenceActionOutcomeMutationReceipt,
    PreferenceActionForecast,
    PreferenceActionForecastSettlement,
    SocialPrediction,
    SocialPredictionError,
    SocialPredictionOutcome,
    SocialScopeKind,
    preference_action_outcome_evidence_sha256,
    preference_action_forecast_from_payload,
    preference_action_forecast_to_payload,
)

SimilarityFn = Callable[[str, str], float]
_SOCIAL_RECORD_OWNER_NAME = "social_record_store"
_SOCIAL_RECORD_SCHEMA_VERSION = 4
_SOCIAL_RECORD_COMPATIBLE_SCHEMA_VERSIONS = frozenset({1, 2, 3, 4})

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
_SOCIAL_RECORD_PERSISTENCE_HASH_SCHEMA_VERSION = (
    "social-record-store-persistence-hash.v1"
)
MAX_PENDING_AGE_TURNS = 6
_CONFIRM_CONFIDENCE_GAIN = 0.10
_DISCONFIRM_CONFIDENCE_LOSS = 0.20
# G1 / CP-18: uninformed prior for the learned per-group
# commitment-durability score (see ``group_durability_for``).
GROUP_DURABILITY_PRIOR = 0.5


def default_summary_similarity(left: str, right: str) -> float:
    """Unit-scaled semantic similarity via the shared embedding seam."""

    cosine = _cosine_similarity(_semantic_embedding(left), _semantic_embedding(right))
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
            ((evidence_id, similarity(prediction.predicted_outcome, summary)) for evidence_id, summary in candidates),
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
        self._tom_records: dict[str, tuple[OtherMindRecord, ...]] = {slot: () for slot in TOM_SLOTS}
        self._tom_pending: dict[str, tuple[PendingSocialPrediction, ...]] = {slot: () for slot in TOM_SLOTS}
        self._preference_action_outcomes: tuple[PreferenceActionOutcomeEvidence, ...] = ()
        self._preference_action_forecasts: tuple[PreferenceActionForecast, ...] = ()
        self._preference_forecast_settlements: tuple[PreferenceActionForecastSettlement, ...] = ()
        self._preference_action_outcome_mutation_receipts: tuple[PreferenceActionOutcomeMutationReceipt, ...] = ()
        self._cg_dyad_atoms: tuple[CommonGroundAtom, ...] = ()
        self._cg_group_atoms: tuple[CommonGroundAtom, ...] = ()
        self._cg_pending: tuple[PendingSocialPrediction, ...] = ()
        # R14 / CP-18: persistent per-group regime map (GroupModule is the
        # single writer).
        self._group_regimes: dict[str, str] = {}
        # G1 / CP-18: pending group-durability predictions + the learned
        # per-group durability score they settle into.
        self._group_pending: tuple[PendingSocialPrediction, ...] = ()
        self._group_durability: dict[str, float] = {}

    # ----- cross-session hydration -----
    #
    # Only stable owner-owned state is persisted. Pending predictions
    # are intentionally session-scoped: carrying an unsettled prediction
    # across a scene/session boundary would extend a one-turn settlement
    # contract beyond the evidence window that issued it.

    def export_persistence_snapshot(self) -> OwnerPersistenceSnapshot:
        return OwnerPersistenceSnapshot(
            owner_name=_SOCIAL_RECORD_OWNER_NAME,
            schema_version=_SOCIAL_RECORD_SCHEMA_VERSION,
            payload={
                "tom_records": {
                    slot: [_serialize_other_mind_record(record) for record in records]
                    for slot, records in self._tom_records.items()
                },
                "preference_action_outcomes": [
                    _serialize_preference_action_outcome(item) for item in self._preference_action_outcomes
                ],
                "preference_action_forecasts": [
                    _serialize_preference_action_forecast(item) for item in self._preference_action_forecasts
                ],
                "preference_forecast_settlements": [
                    _serialize_preference_forecast_settlement(item) for item in self._preference_forecast_settlements
                ],
                "preference_action_outcome_mutation_receipts": [
                    _serialize_preference_action_outcome_mutation_receipt(item)
                    for item in self._preference_action_outcome_mutation_receipts
                ],
                "common_ground": {
                    "dyad_atoms": [_serialize_common_ground_atom(atom) for atom in self._cg_dyad_atoms],
                    "group_atoms": [_serialize_common_ground_atom(atom) for atom in self._cg_group_atoms],
                },
                "group_regimes": dict(self._group_regimes),
                "group_durability": dict(self._group_durability),
            },
            description=(
                f"SocialRecordStore snapshot v{_SOCIAL_RECORD_SCHEMA_VERSION}: "
                f"{sum(len(records) for records in self._tom_records.values())} ToM records, "
                f"{len(self._cg_dyad_atoms) + len(self._cg_group_atoms)} common-ground atoms"
            ),
        )

    def hydrate_from_persistence(self, snapshot: OwnerPersistenceSnapshot) -> None:
        if snapshot.owner_name != _SOCIAL_RECORD_OWNER_NAME:
            raise HydrationOwnerMismatchError(
                f"SocialRecordStore expected owner_name={_SOCIAL_RECORD_OWNER_NAME!r}, got {snapshot.owner_name!r}"
            )
        if snapshot.schema_version not in _SOCIAL_RECORD_COMPATIBLE_SCHEMA_VERSIONS:
            raise HydrationVersionMismatchError(
                "SocialRecordStore unsupported schema_version="
                f"{snapshot.schema_version!r}; expected one of "
                f"{sorted(_SOCIAL_RECORD_COMPATIBLE_SCHEMA_VERSIONS)!r}"
            )
        payload = snapshot.payload
        try:
            tom_records_blob = _mapping_value(payload, "tom_records", default={})
            common_ground_blob = _mapping_value(payload, "common_ground", default={})
            group_regimes_blob = _mapping_value(payload, "group_regimes", default={})
            group_durability_blob = _mapping_value(payload, "group_durability", default={})
        except HydrationPayloadInvalidError:
            raise

        unknown_slots = set(tom_records_blob).difference(TOM_SLOTS)
        if unknown_slots:
            raise HydrationPayloadInvalidError(
                "SocialRecordStore payload references unknown ToM slot(s): "
                f"{sorted(unknown_slots)!r}; expected subset of {TOM_SLOTS!r}"
            )

        new_tom_records: dict[str, tuple[OtherMindRecord, ...]] = {}
        for slot in TOM_SLOTS:
            entries = tom_records_blob.get(slot, ())
            if not isinstance(entries, list | tuple):
                raise HydrationPayloadInvalidError(
                    f"SocialRecordStore payload['tom_records'][{slot!r}] must be a list; got {type(entries).__name__}"
                )
            new_tom_records[slot] = tuple(_deserialize_other_mind_record(entry) for entry in entries)[-_RECORD_WINDOW:]

        preference_action_outcomes_blob = payload.get("preference_action_outcomes", ())
        if not isinstance(preference_action_outcomes_blob, list | tuple):
            raise HydrationPayloadInvalidError(
                "SocialRecordStore preference_action_outcomes must be a list; "
                f"got {type(preference_action_outcomes_blob).__name__}"
            )
        new_preference_action_outcomes = tuple(
            _deserialize_preference_action_outcome(item) for item in preference_action_outcomes_blob
        )[-_RECORD_WINDOW:]
        preference_action_forecasts_blob = payload.get("preference_action_forecasts", ())
        if not isinstance(preference_action_forecasts_blob, list | tuple):
            raise HydrationPayloadInvalidError(
                "SocialRecordStore preference_action_forecasts must be a list; "
                f"got {type(preference_action_forecasts_blob).__name__}"
            )
        new_preference_action_forecasts = tuple(
            _deserialize_preference_action_forecast(
                item,
                schema_version=snapshot.schema_version,
            )
            for item in preference_action_forecasts_blob
        )[-_RECORD_WINDOW:]
        preference_forecast_settlements_blob = payload.get("preference_forecast_settlements", ())
        if not isinstance(preference_forecast_settlements_blob, list | tuple):
            raise HydrationPayloadInvalidError(
                "SocialRecordStore preference_forecast_settlements must be a list; "
                f"got {type(preference_forecast_settlements_blob).__name__}"
            )
        new_preference_forecast_settlements = tuple(
            _deserialize_preference_forecast_settlement(item) for item in preference_forecast_settlements_blob
        )[-_RECORD_WINDOW:]
        if snapshot.schema_version >= 3:
            try:
                mutation_receipts_blob = payload["preference_action_outcome_mutation_receipts"]
            except KeyError as exc:
                raise HydrationPayloadInvalidError(
                    "SocialRecordStore v3 payload is missing preference_action_outcome_mutation_receipts"
                ) from exc
        else:
            mutation_receipts_blob = ()
        if not isinstance(mutation_receipts_blob, list | tuple):
            raise HydrationPayloadInvalidError(
                "SocialRecordStore preference_action_outcome_mutation_receipts "
                f"must be a list; got {type(mutation_receipts_blob).__name__}"
            )
        new_mutation_receipts = tuple(
            _deserialize_preference_action_outcome_mutation_receipt(item) for item in mutation_receipts_blob
        )

        try:
            _validate_preference_action_outcome_mutation_state(
                records=new_tom_records["preference_about_other"],
                action_outcomes=new_preference_action_outcomes,
                receipts=new_mutation_receipts,
            )
        except ValueError as exc:
            raise HydrationPayloadInvalidError(
                f"SocialRecordStore preference action mutation state is invalid: {exc}"
            ) from exc

        dyad_blob = common_ground_blob.get("dyad_atoms", ())
        group_blob = common_ground_blob.get("group_atoms", ())
        if not isinstance(dyad_blob, list | tuple):
            raise HydrationPayloadInvalidError(
                f"SocialRecordStore common_ground.dyad_atoms must be a list; got {type(dyad_blob).__name__}"
            )
        if not isinstance(group_blob, list | tuple):
            raise HydrationPayloadInvalidError(
                f"SocialRecordStore common_ground.group_atoms must be a list; got {type(group_blob).__name__}"
            )

        self._tom_records = new_tom_records
        self._tom_pending = {slot: () for slot in TOM_SLOTS}
        self._preference_action_outcomes = new_preference_action_outcomes
        self._preference_action_forecasts = new_preference_action_forecasts
        self._preference_forecast_settlements = new_preference_forecast_settlements
        self._preference_action_outcome_mutation_receipts = new_mutation_receipts
        self._cg_dyad_atoms = tuple(_deserialize_common_ground_atom(entry) for entry in dyad_blob)[-_ATOM_WINDOW:]
        self._cg_group_atoms = tuple(_deserialize_common_ground_atom(entry) for entry in group_blob)[-_ATOM_WINDOW:]
        self._cg_pending = ()
        self._group_regimes = {
            str(group_id): str(regime_id)
            for group_id, regime_id in group_regimes_blob.items()
            if str(group_id).strip() and str(regime_id).strip()
        }
        self._group_pending = ()
        self._group_durability = {
            str(group_id): max(0.0, min(1.0, float(score)))
            for group_id, score in group_durability_blob.items()
            if str(group_id).strip()
        }
    @property
    def similarity(self) -> SimilarityFn:
        return self._similarity

    # ----- ToM slots -----

    def tom_records(self, slot: str) -> tuple[OtherMindRecord, ...]:
        self._require_tom_slot(slot)
        return self._tom_records[slot]

    def set_tom_records(self, slot: str, records: tuple[OtherMindRecord, ...]) -> None:
        self._require_tom_slot(slot)
        bounded_records = records[-_RECORD_WINDOW:]
        if slot == "preference_about_other":
            _validate_receipt_managed_preference_records(
                records=bounded_records,
                action_outcomes=self._preference_action_outcomes,
                receipts=self._preference_action_outcome_mutation_receipts,
            )
        self._tom_records[slot] = bounded_records

    def pending_tom_predictions(self, slot: str) -> tuple[PendingSocialPrediction, ...]:
        self._require_tom_slot(slot)
        return self._tom_pending[slot]

    def set_pending_tom_predictions(self, slot: str, pending: tuple[PendingSocialPrediction, ...]) -> None:
        self._require_tom_slot(slot)
        self._tom_pending[slot] = pending[-_RECORD_WINDOW:]

    def _require_tom_slot(self, slot: str) -> None:
        if slot not in TOM_SLOTS:
            raise ValueError(f"unknown ToM slot {slot!r}")

    # ----- typed preference action/outcome evidence -----

    @property
    def preference_action_outcomes(
        self,
    ) -> tuple[PreferenceActionOutcomeEvidence, ...]:
        return self._preference_action_outcomes

    def set_preference_action_outcomes(
        self,
        action_outcomes: tuple[PreferenceActionOutcomeEvidence, ...],
    ) -> None:
        bounded_outcomes = action_outcomes[-_RECORD_WINDOW:]
        evidence_ids = tuple(item.evidence_id for item in action_outcomes)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("preference action outcome evidence ids must be unique")
        _validate_preference_action_outcome_mutation_state(
            records=self._tom_records["preference_about_other"],
            action_outcomes=bounded_outcomes,
            receipts=self._preference_action_outcome_mutation_receipts,
        )
        self._preference_action_outcomes = bounded_outcomes

    @property
    def preference_action_forecasts(self) -> tuple[PreferenceActionForecast, ...]:
        return self._preference_action_forecasts

    def set_preference_action_forecasts(
        self,
        forecasts: tuple[PreferenceActionForecast, ...],
    ) -> None:
        forecast_ids = tuple(item.forecast_id for item in forecasts)
        if len(set(forecast_ids)) != len(forecast_ids):
            raise ValueError("preference action forecast ids must be unique")
        self._preference_action_forecasts = forecasts[-_RECORD_WINDOW:]

    @property
    def preference_forecast_settlements(
        self,
    ) -> tuple[PreferenceActionForecastSettlement, ...]:
        return self._preference_forecast_settlements

    def set_preference_forecast_settlements(
        self,
        settlements: tuple[PreferenceActionForecastSettlement, ...],
    ) -> None:
        settlement_ids = tuple(item.settlement_id for item in settlements)
        forecast_ids = tuple(item.forecast_id for item in settlements)
        if len(set(settlement_ids)) != len(settlement_ids):
            raise ValueError("preference forecast settlement ids must be unique")
        if len(set(forecast_ids)) != len(forecast_ids):
            raise ValueError("a preference action forecast may settle at most once")
        self._preference_forecast_settlements = settlements[-_RECORD_WINDOW:]

    @property
    def preference_action_outcome_mutation_receipts(
        self,
    ) -> tuple[PreferenceActionOutcomeMutationReceipt, ...]:
        return self._preference_action_outcome_mutation_receipts

    def replace_preference_action_mutation_state(
        self,
        *,
        records: tuple[OtherMindRecord, ...],
        pending_predictions: tuple[PendingSocialPrediction, ...],
        action_outcomes: tuple[PreferenceActionOutcomeEvidence, ...],
        action_forecasts: tuple[PreferenceActionForecast, ...],
        receipts: tuple[PreferenceActionOutcomeMutationReceipt, ...],
    ) -> None:
        """Atomically install owner-validated correction/redaction state.

        This method does not interpret a mutation command. It only gives the
        single preference owner one consistency boundary across the five store
        collections that a correction or redaction must change together.
        """

        bounded_records = records[-_RECORD_WINDOW:]
        bounded_pending = pending_predictions[-_RECORD_WINDOW:]
        bounded_outcomes = action_outcomes[-_RECORD_WINDOW:]
        bounded_forecasts = action_forecasts[-_RECORD_WINDOW:]
        current_receipts = self._preference_action_outcome_mutation_receipts
        if receipts[: len(current_receipts)] != current_receipts:
            raise ValueError("preference action outcome mutation receipts are append-only")
        _validate_preference_action_outcome_mutation_state(
            records=bounded_records,
            action_outcomes=bounded_outcomes,
            receipts=receipts,
        )
        record_ids = {record.record_id for record in bounded_records}
        unknown_pending_ids = {item.source_record_id for item in bounded_pending}.difference(record_ids)
        if unknown_pending_ids:
            raise ValueError(
                f"preference pending predictions must reference owner records; unknown={sorted(unknown_pending_ids)!r}"
            )
        forecast_ids = tuple(item.forecast_id for item in bounded_forecasts)
        if len(set(forecast_ids)) != len(forecast_ids):
            raise ValueError("preference action forecast ids must be unique")
        unknown_forecast_record_ids = {
            source_record_id for forecast in bounded_forecasts for source_record_id in forecast.source_record_ids
        }.difference(record_ids)
        if unknown_forecast_record_ids:
            raise ValueError(
                "preference action forecasts must reference owner records; "
                f"unknown={sorted(unknown_forecast_record_ids)!r}"
            )
        self._tom_records["preference_about_other"] = bounded_records
        self._tom_pending["preference_about_other"] = bounded_pending
        self._preference_action_outcomes = bounded_outcomes
        self._preference_action_forecasts = bounded_forecasts
        self._preference_action_outcome_mutation_receipts = receipts

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

    def set_pending_common_ground_predictions(self, pending: tuple[PendingSocialPrediction, ...]) -> None:
        self._cg_pending = pending[-_ATOM_WINDOW:]

    # ----- group regime persistence (R14 / CP-18) -----
    #
    # Group regime is runtime state owned by ``GroupModule`` (never a
    # prompt label). Because owner modules are rebuilt per turn, the
    # durable per-group regime map lives here — same lifetime pattern as
    # the ToM record windows. Single writer: only GroupModule mutates it.

    def group_regime_for(self, group_id: str) -> str | None:
        if not group_id:
            raise ValueError("group_id must be non-empty")
        return self._group_regimes.get(group_id)

    def record_group_regime(self, group_id: str, regime_id: str) -> None:
        if not group_id:
            raise ValueError("group_id must be non-empty")
        if not regime_id:
            raise ValueError("regime_id must be non-empty")
        self._group_regimes[group_id] = regime_id

    # ----- group PE settlement state (G1 / CP-18) -----
    #
    # Same settlement machinery as the ToM / common-ground owners:
    # GroupModule issues GROUP_COMMITMENT_DURABILITY predictions, parks
    # them here, and settles them against the NEXT turn's observed
    # group state. The per-group durability score is the bounded
    # learned readout of that PE stream (CONFIRMED pushes up,
    # DISCONFIRMED pushes down) — it feeds the confidence of future
    # durability predictions so the group-level PE actually shapes the
    # owner's forward model instead of dead-ending in telemetry.
    # Single writer: only GroupModule mutates these.

    @property
    def pending_group_predictions(self) -> tuple[PendingSocialPrediction, ...]:
        return self._group_pending

    def set_pending_group_predictions(self, pending: tuple[PendingSocialPrediction, ...]) -> None:
        self._group_pending = pending[-_RECORD_WINDOW:]

    def group_durability_for(self, group_id: str) -> float:
        """Learned commitment-durability score, 0.5 uninformed prior."""

        if not group_id:
            raise ValueError("group_id must be non-empty")
        return self._group_durability.get(group_id, GROUP_DURABILITY_PRIOR)

    def apply_group_settlement(self, group_id: str, outcome: SocialPredictionOutcome) -> float:
        """Bounded online update of the learned durability score.

        CONFIRMED: score += gain. DISCONFIRMED: score -= loss (the
        asymmetry mirrors the ToM confidence table: broken joint
        commitments are stronger evidence than kept ones). STALE /
        UNKNOWN: no update. Returns the post-update score.
        """

        if not group_id:
            raise ValueError("group_id must be non-empty")
        score = self.group_durability_for(group_id)
        if outcome is SocialPredictionOutcome.CONFIRMED:
            score = min(1.0, score + _CONFIRM_CONFIDENCE_GAIN)
        elif outcome is SocialPredictionOutcome.DISCONFIRMED:
            score = max(0.0, score - _DISCONFIRM_CONFIDENCE_LOSS)
        self._group_durability[group_id] = score
        return score


def social_record_store_persistence_sha256(
    snapshot: OwnerPersistenceSnapshot,
) -> str:
    """Hash one canonical owner persistence payload without exposing its shape."""

    if not isinstance(snapshot, OwnerPersistenceSnapshot):
        raise TypeError("snapshot must be OwnerPersistenceSnapshot")
    store = SocialRecordStore()
    store.hydrate_from_persistence(snapshot)
    canonical = store.export_persistence_snapshot()
    if (
        canonical.owner_name != snapshot.owner_name
        or canonical.schema_version != snapshot.schema_version
        or canonical.payload != snapshot.payload
    ):
        raise ValueError(
            "social record persistence must be a canonical current-schema export"
        )
    encoded = json.dumps(
        {
            "hash_schema_version": (
                _SOCIAL_RECORD_PERSISTENCE_HASH_SCHEMA_VERSION
            ),
            "owner_name": canonical.owner_name,
            "owner_schema_version": canonical.schema_version,
            "payload": canonical.payload,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping_value(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: Mapping[str, Any],
) -> Mapping[str, Any]:
    value = payload.get(key, default)
    if not isinstance(value, Mapping):
        raise HydrationPayloadInvalidError(
            f"SocialRecordStore payload[{key!r}] must be a mapping; got {type(value).__name__}"
        )
    return value


def _validate_preference_action_outcome_mutation_state(
    *,
    records: tuple[OtherMindRecord, ...],
    action_outcomes: tuple[PreferenceActionOutcomeEvidence, ...],
    receipts: tuple[PreferenceActionOutcomeMutationReceipt, ...],
) -> None:
    record_ids = tuple(record.record_id for record in records)
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("preference owner record ids must be unique")
    for record in records:
        if record.kind is not OtherMindRecordKind.PREFERENCE:
            raise ValueError("preference owner store cannot contain another ToM kind")

    records_by_id = {record.record_id: record for record in records}
    evidence_ids = tuple(item.evidence_id for item in action_outcomes)
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("preference action outcome evidence ids must be unique")
    for item in action_outcomes:
        record = records_by_id.get(item.evidence_id)
        if record is None:
            raise ValueError(
                f"preference action outcome evidence must reference owner records; unknown={item.evidence_id!r}"
            )
        if record.interlocutor_id != item.interlocutor_id:
            raise ValueError("preference action outcome evidence interlocutor lineage mismatch")
        if record.source_turn != item.source_turn:
            raise ValueError("preference action outcome evidence turn lineage mismatch")

    mutation_ids = tuple(receipt.mutation_id for receipt in receipts)
    if len(set(mutation_ids)) != len(mutation_ids):
        raise ValueError("preference action outcome mutation ids must be unique")
    latest_by_target: dict[str, PreferenceActionOutcomeMutationReceipt] = {}
    for receipt in receipts:
        previous = latest_by_target.get(receipt.target_evidence_id)
        if previous is not None:
            if previous.operation is PreferenceActionOutcomeMutationOperation.REDACT:
                raise ValueError("redacted preference evidence cannot be mutated again")
            if receipt.applied_turn < previous.applied_turn:
                raise ValueError("preference mutation receipt turns must be monotonic")
            if receipt.before_evidence_sha256 != previous.after_evidence_sha256:
                raise ValueError("preference mutation receipt hash chain is broken")
        latest_by_target[receipt.target_evidence_id] = receipt

    _validate_receipt_managed_preference_records(
        records=records,
        action_outcomes=action_outcomes,
        receipts=receipts,
    )


def _validate_receipt_managed_preference_records(
    *,
    records: tuple[OtherMindRecord, ...],
    action_outcomes: tuple[PreferenceActionOutcomeEvidence, ...],
    receipts: tuple[PreferenceActionOutcomeMutationReceipt, ...],
) -> None:
    records_by_id = {record.record_id: record for record in records}
    outcomes_by_id = {item.evidence_id: item for item in action_outcomes}
    latest_by_target: dict[str, PreferenceActionOutcomeMutationReceipt] = {}
    for receipt in receipts:
        latest_by_target[receipt.target_evidence_id] = receipt
    for target_evidence_id, latest in latest_by_target.items():
        current_outcome = outcomes_by_id.get(target_evidence_id)
        current_record = records_by_id.get(target_evidence_id)
        if latest.operation is PreferenceActionOutcomeMutationOperation.REDACT:
            if current_outcome is not None or current_record is not None:
                raise ValueError("redacted preference evidence or owner record was resurrected")
            continue
        if current_outcome is not None and (
            preference_action_outcome_evidence_sha256(current_outcome) != latest.after_evidence_sha256
        ):
            raise ValueError("corrected preference evidence does not match latest receipt hash")
        if (
            current_outcome is not None
            and current_record is not None
            and (
                current_record.summary != current_outcome.observation_summary
                or current_record.detail != current_outcome.reaction_summary
                or current_record.evidence != current_outcome.evidence_refs[0]
            )
        ):
            raise ValueError("corrected preference owner record does not match corrected evidence")


def _serialize_other_mind_record(record: OtherMindRecord) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "interlocutor_id": record.interlocutor_id,
        "kind": record.kind.value,
        "summary": record.summary,
        "detail": record.detail,
        "confidence": record.confidence,
        "status": record.status.value,
        "source_turn": record.source_turn,
        "prediction_error_refs": list(record.prediction_error_refs),
        "evidence": record.evidence,
    }


def _deserialize_other_mind_record(blob: Mapping[str, Any]) -> OtherMindRecord:
    try:
        refs = blob.get("prediction_error_refs", ())
        if not isinstance(refs, list | tuple):
            raise HydrationPayloadInvalidError(
                f"OtherMindRecord.prediction_error_refs must be a list; got {type(refs).__name__}"
            )
        return OtherMindRecord(
            record_id=str(blob["record_id"]),
            interlocutor_id=str(blob["interlocutor_id"]),
            kind=OtherMindRecordKind(str(blob["kind"])),
            summary=str(blob["summary"]),
            detail=str(blob["detail"]),
            confidence=float(blob["confidence"]),
            status=OtherMindRecordStatus(str(blob["status"])),
            source_turn=int(blob["source_turn"]),
            prediction_error_refs=tuple(str(item) for item in refs),
            evidence=str(blob["evidence"]),
        )
    except KeyError as exc:
        raise HydrationPayloadInvalidError(
            f"SocialRecordStore OtherMindRecord missing key {exc.args[0]!r}; blob={blob!r}"
        ) from exc
    except ValueError as exc:
        raise HydrationPayloadInvalidError(
            f"SocialRecordStore OtherMindRecord invalid enum/value: {exc}; blob={blob!r}"
        ) from exc


def _serialize_preference_action_outcome(
    item: PreferenceActionOutcomeEvidence,
) -> dict[str, Any]:
    return {
        "evidence_id": item.evidence_id,
        "interlocutor_id": item.interlocutor_id,
        "observation_summary": item.observation_summary,
        "action_id": item.action_id,
        "observed_outcome_id": item.observed_outcome_id,
        "reaction_summary": item.reaction_summary,
        "source_turn": item.source_turn,
        "evidence_refs": list(item.evidence_refs),
    }


def _deserialize_preference_action_outcome(
    blob: Mapping[str, Any],
) -> PreferenceActionOutcomeEvidence:
    try:
        evidence_refs = blob["evidence_refs"]
        if not isinstance(evidence_refs, list | tuple):
            raise HydrationPayloadInvalidError(
                f"PreferenceActionOutcomeEvidence.evidence_refs must be a list; got {type(evidence_refs).__name__}"
            )
        return PreferenceActionOutcomeEvidence(
            evidence_id=str(blob["evidence_id"]),
            interlocutor_id=str(blob["interlocutor_id"]),
            observation_summary=str(blob["observation_summary"]),
            action_id=str(blob["action_id"]),
            observed_outcome_id=str(blob["observed_outcome_id"]),
            reaction_summary=str(blob["reaction_summary"]),
            source_turn=int(blob["source_turn"]),
            evidence_refs=tuple(str(item) for item in evidence_refs),
        )
    except KeyError as exc:
        raise HydrationPayloadInvalidError(
            f"SocialRecordStore PreferenceActionOutcomeEvidence missing key {exc.args[0]!r}; blob={blob!r}"
        ) from exc
    except ValueError as exc:
        raise HydrationPayloadInvalidError(
            f"SocialRecordStore PreferenceActionOutcomeEvidence invalid value: {exc}; blob={blob!r}"
        ) from exc


def _serialize_preference_action_outcome_mutation_receipt(
    receipt: PreferenceActionOutcomeMutationReceipt,
) -> dict[str, Any]:
    return {
        "mutation_id": receipt.mutation_id,
        "command_sha256": receipt.command_sha256,
        "target_evidence_id": receipt.target_evidence_id,
        "operation": receipt.operation.value,
        "before_evidence_sha256": receipt.before_evidence_sha256,
        "after_evidence_sha256": receipt.after_evidence_sha256,
        "applied_turn": receipt.applied_turn,
        "invalidated_forecast_ids": list(receipt.invalidated_forecast_ids),
        "evidence_refs": list(receipt.evidence_refs),
    }


def _deserialize_preference_action_outcome_mutation_receipt(
    blob: Mapping[str, Any],
) -> PreferenceActionOutcomeMutationReceipt:
    if not isinstance(blob, Mapping):
        raise HydrationPayloadInvalidError("PreferenceActionOutcomeMutationReceipt must be an object")
    try:
        invalidated_forecast_ids = blob["invalidated_forecast_ids"]
        evidence_refs = blob["evidence_refs"]
        if not isinstance(invalidated_forecast_ids, list | tuple):
            raise HydrationPayloadInvalidError(
                "PreferenceActionOutcomeMutationReceipt.invalidated_forecast_ids must be a list"
            )
        if not isinstance(evidence_refs, list | tuple):
            raise HydrationPayloadInvalidError("PreferenceActionOutcomeMutationReceipt.evidence_refs must be a list")
        after_evidence_sha256 = blob["after_evidence_sha256"]
        return PreferenceActionOutcomeMutationReceipt(
            mutation_id=str(blob["mutation_id"]),
            command_sha256=str(blob["command_sha256"]),
            target_evidence_id=str(blob["target_evidence_id"]),
            operation=PreferenceActionOutcomeMutationOperation(str(blob["operation"])),
            before_evidence_sha256=str(blob["before_evidence_sha256"]),
            after_evidence_sha256=(None if after_evidence_sha256 is None else str(after_evidence_sha256)),
            applied_turn=int(blob["applied_turn"]),
            invalidated_forecast_ids=tuple(str(item) for item in invalidated_forecast_ids),
            evidence_refs=tuple(str(item) for item in evidence_refs),
        )
    except KeyError as exc:
        raise HydrationPayloadInvalidError(
            f"SocialRecordStore PreferenceActionOutcomeMutationReceipt missing key {exc.args[0]!r}; blob={blob!r}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise HydrationPayloadInvalidError(
            f"SocialRecordStore PreferenceActionOutcomeMutationReceipt invalid value: {exc}; blob={blob!r}"
        ) from exc


def _serialize_preference_action_forecast(
    forecast: PreferenceActionForecast,
) -> dict[str, Any]:
    return preference_action_forecast_to_payload(forecast)


def _deserialize_preference_action_forecast(
    blob: Mapping[str, Any],
    *,
    schema_version: int,
) -> PreferenceActionForecast:
    try:
        payload = (
            blob
            if schema_version >= 4
            else _adapt_legacy_preference_action_forecast_payload(blob)
        )
        return preference_action_forecast_from_payload(payload)
    except KeyError as exc:
        raise HydrationPayloadInvalidError(
            f"SocialRecordStore PreferenceActionForecast missing key {exc.args[0]!r}; blob={blob!r}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise HydrationPayloadInvalidError(
            f"SocialRecordStore PreferenceActionForecast invalid value: {exc}; blob={blob!r}"
        ) from exc


def _adapt_legacy_preference_action_forecast_payload(
    blob: Mapping[str, Any],
) -> dict[str, object]:
    """Normalize SocialRecordStore schemas v1-v3 into the strict v4 shape."""

    candidates_blob = blob["candidate_predictions"]
    source_record_ids = blob["source_record_ids"]
    evidence = blob["evidence"]
    if not isinstance(candidates_blob, list | tuple):
        raise TypeError("PreferenceActionForecast.candidate_predictions must be a list")
    if not isinstance(source_record_ids, list | tuple):
        raise TypeError("PreferenceActionForecast.source_record_ids must be a list")
    if not isinstance(evidence, list | tuple):
        raise TypeError("PreferenceActionForecast.evidence must be a list")
    candidates: list[dict[str, object]] = []
    for candidate_blob in candidates_blob:
        if not isinstance(candidate_blob, Mapping):
            raise TypeError("PreferenceActionForecast candidate must be an object")
        outcomes_blob = candidate_blob["outcomes"]
        if not isinstance(outcomes_blob, list | tuple):
            raise TypeError("PreferenceActionForecast candidate outcomes must be a list")
        outcomes: list[dict[str, object]] = []
        for outcome_blob in outcomes_blob:
            if not isinstance(outcome_blob, Mapping):
                raise TypeError("PreferenceActionForecast outcome must be an object")
            outcomes.append(
                {
                    "outcome_id": str(outcome_blob["outcome_id"]),
                    "probability": float(outcome_blob["probability"]),
                }
            )
        candidates.append(
            {
                "action_id": str(candidate_blob["action_id"]),
                "outcomes": outcomes,
            }
        )
    return {
        "forecast_id": str(blob["forecast_id"]),
        "decision_id": str(blob["decision_id"]),
        "interlocutor_id": str(blob["interlocutor_id"]),
        "candidate_predictions": candidates,
        "recommended_action_id": str(blob["recommended_action_id"]),
        "confidence": float(blob["confidence"]),
        "source_record_ids": [str(item) for item in source_record_ids],
        "issued_turn": int(blob["issued_turn"]),
        "evidence": [str(item) for item in evidence],
        "session_scope": str(blob.get("session_scope", "")),
        "condition_readout": None,
    }


def _serialize_preference_forecast_settlement(
    settlement: PreferenceActionForecastSettlement,
) -> dict[str, Any]:
    return {
        "settlement_id": settlement.settlement_id,
        "forecast_id": settlement.forecast_id,
        "decision_id": settlement.decision_id,
        "session_scope": settlement.session_scope,
        "interlocutor_id": settlement.interlocutor_id,
        "action_id": settlement.action_id,
        "observed_outcome_id": settlement.observed_outcome_id,
        "predicted_probability": settlement.predicted_probability,
        "negative_log_likelihood": settlement.negative_log_likelihood,
        "outcome": settlement.outcome.value,
        "magnitude": settlement.magnitude,
        "source_evidence_id": settlement.source_evidence_id,
        "forecast_issued_turn": settlement.forecast_issued_turn,
        "observed_turn": settlement.observed_turn,
        "evidence_confidence": settlement.evidence_confidence,
        "expected_utility": settlement.expected_utility,
        "observed_utility": settlement.observed_utility,
        "signed_utility_prediction_error": (settlement.signed_utility_prediction_error),
    }


def _deserialize_preference_forecast_settlement(
    blob: Mapping[str, Any],
) -> PreferenceActionForecastSettlement:
    try:
        return PreferenceActionForecastSettlement(
            settlement_id=str(blob["settlement_id"]),
            forecast_id=str(blob["forecast_id"]),
            decision_id=str(blob["decision_id"]),
            session_scope=str(blob["session_scope"]),
            interlocutor_id=str(blob["interlocutor_id"]),
            action_id=str(blob["action_id"]),
            observed_outcome_id=str(blob["observed_outcome_id"]),
            predicted_probability=float(blob["predicted_probability"]),
            negative_log_likelihood=float(blob["negative_log_likelihood"]),
            outcome=SocialPredictionOutcome(str(blob["outcome"])),
            magnitude=float(blob["magnitude"]),
            source_evidence_id=str(blob["source_evidence_id"]),
            forecast_issued_turn=int(blob["forecast_issued_turn"]),
            observed_turn=int(blob["observed_turn"]),
            evidence_confidence=float(blob.get("evidence_confidence", 1.0)),
            expected_utility=float(blob.get("expected_utility", 0.0)),
            observed_utility=float(blob.get("observed_utility", 0.0)),
            signed_utility_prediction_error=float(blob.get("signed_utility_prediction_error", 0.0)),
        )
    except KeyError as exc:
        raise HydrationPayloadInvalidError(
            f"SocialRecordStore PreferenceActionForecastSettlement missing key {exc.args[0]!r}; blob={blob!r}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise HydrationPayloadInvalidError(
            f"SocialRecordStore PreferenceActionForecastSettlement invalid value: {exc}; blob={blob!r}"
        ) from exc


def _serialize_common_ground_atom(atom: CommonGroundAtom) -> dict[str, Any]:
    return {
        "atom_id": atom.atom_id,
        "scope_id": atom.scope_id,
        "scope_kind": atom.scope_kind.value,
        "summary": atom.summary,
        "recursion_depth": atom.recursion_depth,
        "confidence": atom.confidence,
        "accepted_by_ids": list(atom.accepted_by_ids),
        "evidence": list(atom.evidence),
    }


def _deserialize_common_ground_atom(blob: Mapping[str, Any]) -> CommonGroundAtom:
    try:
        accepted = blob["accepted_by_ids"]
        evidence = blob["evidence"]
        if not isinstance(accepted, list | tuple):
            raise HydrationPayloadInvalidError(
                f"CommonGroundAtom.accepted_by_ids must be a list; got {type(accepted).__name__}"
            )
        if not isinstance(evidence, list | tuple):
            raise HydrationPayloadInvalidError(
                f"CommonGroundAtom.evidence must be a list; got {type(evidence).__name__}"
            )
        return CommonGroundAtom(
            atom_id=str(blob["atom_id"]),
            scope_id=str(blob["scope_id"]),
            scope_kind=SocialScopeKind(str(blob["scope_kind"])),
            summary=str(blob["summary"]),
            recursion_depth=int(blob["recursion_depth"]),
            confidence=float(blob["confidence"]),
            accepted_by_ids=tuple(str(item) for item in accepted),
            evidence=tuple(str(item) for item in evidence),
        )
    except KeyError as exc:
        raise HydrationPayloadInvalidError(
            f"SocialRecordStore CommonGroundAtom missing key {exc.args[0]!r}; blob={blob!r}"
        ) from exc
    except ValueError as exc:
        raise HydrationPayloadInvalidError(
            f"SocialRecordStore CommonGroundAtom invalid enum/value: {exc}; blob={blob!r}"
        ) from exc


__all__ = [
    "CONFIRM_SIMILARITY",
    "DISCONFIRM_SIMILARITY",
    "GROUP_DURABILITY_PRIOR",
    "MAX_PENDING_AGE_TURNS",
    "TOM_SLOTS",
    "PendingSocialPrediction",
    "SocialRecordStore",
    "SocialSettlementResult",
    "apply_outcome_to_record",
    "default_summary_similarity",
    "social_record_store_persistence_sha256",
    "settle_pending_predictions",
]
