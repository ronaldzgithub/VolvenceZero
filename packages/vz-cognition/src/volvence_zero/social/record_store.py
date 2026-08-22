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
    PreferenceActionForecast,
    PreferenceActionForecastSettlement,
    SocialActionCandidatePrediction,
    SocialActionOutcomeProbability,
    SocialPrediction,
    SocialPredictionError,
    SocialPredictionOutcome,
    SocialScopeKind,
)

SimilarityFn = Callable[[str, str], float]
_SOCIAL_RECORD_OWNER_NAME = "social_record_store"
_SOCIAL_RECORD_SCHEMA_VERSION = 2
_SOCIAL_RECORD_COMPATIBLE_SCHEMA_VERSIONS = frozenset({1, 2})

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
# G1 / CP-18: uninformed prior for the learned per-group
# commitment-durability score (see ``group_durability_for``).
GROUP_DURABILITY_PRIOR = 0.5


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
        self._preference_action_outcomes: tuple[
            PreferenceActionOutcomeEvidence, ...
        ] = ()
        self._preference_action_forecasts: tuple[PreferenceActionForecast, ...] = ()
        self._preference_forecast_settlements: tuple[
            PreferenceActionForecastSettlement, ...
        ] = ()
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
                    _serialize_preference_action_outcome(item)
                    for item in self._preference_action_outcomes
                ],
                "preference_action_forecasts": [
                    _serialize_preference_action_forecast(item)
                    for item in self._preference_action_forecasts
                ],
                "preference_forecast_settlements": [
                    _serialize_preference_forecast_settlement(item)
                    for item in self._preference_forecast_settlements
                ],
                "common_ground": {
                    "dyad_atoms": [
                        _serialize_common_ground_atom(atom)
                        for atom in self._cg_dyad_atoms
                    ],
                    "group_atoms": [
                        _serialize_common_ground_atom(atom)
                        for atom in self._cg_group_atoms
                    ],
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

    def hydrate_from_persistence(
        self, snapshot: OwnerPersistenceSnapshot
    ) -> None:
        if snapshot.owner_name != _SOCIAL_RECORD_OWNER_NAME:
            raise HydrationOwnerMismatchError(
                "SocialRecordStore expected owner_name="
                f"{_SOCIAL_RECORD_OWNER_NAME!r}, got {snapshot.owner_name!r}"
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
                    "SocialRecordStore payload['tom_records']"
                    f"[{slot!r}] must be a list; got {type(entries).__name__}"
                )
            new_tom_records[slot] = tuple(
                _deserialize_other_mind_record(entry) for entry in entries
            )[-_RECORD_WINDOW:]

        preference_action_outcomes_blob = payload.get(
            "preference_action_outcomes", ()
        )
        if not isinstance(preference_action_outcomes_blob, list | tuple):
            raise HydrationPayloadInvalidError(
                "SocialRecordStore preference_action_outcomes must be a list; "
                f"got {type(preference_action_outcomes_blob).__name__}"
            )
        new_preference_action_outcomes = tuple(
            _deserialize_preference_action_outcome(item)
            for item in preference_action_outcomes_blob
        )[-_RECORD_WINDOW:]
        preference_action_forecasts_blob = payload.get(
            "preference_action_forecasts", ()
        )
        if not isinstance(preference_action_forecasts_blob, list | tuple):
            raise HydrationPayloadInvalidError(
                "SocialRecordStore preference_action_forecasts must be a list; "
                f"got {type(preference_action_forecasts_blob).__name__}"
            )
        new_preference_action_forecasts = tuple(
            _deserialize_preference_action_forecast(item)
            for item in preference_action_forecasts_blob
        )[-_RECORD_WINDOW:]
        preference_forecast_settlements_blob = payload.get(
            "preference_forecast_settlements", ()
        )
        if not isinstance(preference_forecast_settlements_blob, list | tuple):
            raise HydrationPayloadInvalidError(
                "SocialRecordStore preference_forecast_settlements must be a list; "
                f"got {type(preference_forecast_settlements_blob).__name__}"
            )
        new_preference_forecast_settlements = tuple(
            _deserialize_preference_forecast_settlement(item)
            for item in preference_forecast_settlements_blob
        )[-_RECORD_WINDOW:]

        dyad_blob = common_ground_blob.get("dyad_atoms", ())
        group_blob = common_ground_blob.get("group_atoms", ())
        if not isinstance(dyad_blob, list | tuple):
            raise HydrationPayloadInvalidError(
                "SocialRecordStore common_ground.dyad_atoms must be a list; "
                f"got {type(dyad_blob).__name__}"
            )
        if not isinstance(group_blob, list | tuple):
            raise HydrationPayloadInvalidError(
                "SocialRecordStore common_ground.group_atoms must be a list; "
                f"got {type(group_blob).__name__}"
            )

        self._tom_records = new_tom_records
        self._tom_pending = {slot: () for slot in TOM_SLOTS}
        self._preference_action_outcomes = new_preference_action_outcomes
        self._preference_action_forecasts = new_preference_action_forecasts
        self._preference_forecast_settlements = (
            new_preference_forecast_settlements
        )
        self._cg_dyad_atoms = tuple(
            _deserialize_common_ground_atom(entry) for entry in dyad_blob
        )[-_ATOM_WINDOW:]
        self._cg_group_atoms = tuple(
            _deserialize_common_ground_atom(entry) for entry in group_blob
        )[-_ATOM_WINDOW:]
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
        evidence_ids = tuple(item.evidence_id for item in action_outcomes)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("preference action outcome evidence ids must be unique")
        record_ids = {
            record.record_id
            for record in self._tom_records["preference_about_other"]
        }
        unknown_ids = set(evidence_ids).difference(record_ids)
        if unknown_ids:
            raise ValueError(
                "preference action outcome evidence must reference owner records; "
                f"unknown={sorted(unknown_ids)!r}"
            )
        self._preference_action_outcomes = action_outcomes[-_RECORD_WINDOW:]

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

    def set_pending_group_predictions(
        self, pending: tuple[PendingSocialPrediction, ...]
    ) -> None:
        self._group_pending = pending[-_RECORD_WINDOW:]

    def group_durability_for(self, group_id: str) -> float:
        """Learned commitment-durability score, 0.5 uninformed prior."""

        if not group_id:
            raise ValueError("group_id must be non-empty")
        return self._group_durability.get(group_id, GROUP_DURABILITY_PRIOR)

    def apply_group_settlement(
        self, group_id: str, outcome: SocialPredictionOutcome
    ) -> float:
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


def _mapping_value(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: Mapping[str, Any],
) -> Mapping[str, Any]:
    value = payload.get(key, default)
    if not isinstance(value, Mapping):
        raise HydrationPayloadInvalidError(
            f"SocialRecordStore payload[{key!r}] must be a mapping; "
            f"got {type(value).__name__}"
        )
    return value


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
                "OtherMindRecord.prediction_error_refs must be a list; "
                f"got {type(refs).__name__}"
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
            f"SocialRecordStore OtherMindRecord missing key {exc.args[0]!r}; "
            f"blob={blob!r}"
        ) from exc
    except ValueError as exc:
        raise HydrationPayloadInvalidError(
            f"SocialRecordStore OtherMindRecord invalid enum/value: {exc}; "
            f"blob={blob!r}"
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
                "PreferenceActionOutcomeEvidence.evidence_refs must be a list; "
                f"got {type(evidence_refs).__name__}"
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
            "SocialRecordStore PreferenceActionOutcomeEvidence missing key "
            f"{exc.args[0]!r}; blob={blob!r}"
        ) from exc
    except ValueError as exc:
        raise HydrationPayloadInvalidError(
            "SocialRecordStore PreferenceActionOutcomeEvidence invalid value: "
            f"{exc}; blob={blob!r}"
        ) from exc


def _serialize_preference_action_forecast(
    forecast: PreferenceActionForecast,
) -> dict[str, Any]:
    return {
        "forecast_id": forecast.forecast_id,
        "decision_id": forecast.decision_id,
        "interlocutor_id": forecast.interlocutor_id,
        "candidate_predictions": [
            {
                "action_id": candidate.action_id,
                "outcomes": [
                    {
                        "outcome_id": outcome.outcome_id,
                        "probability": outcome.probability,
                    }
                    for outcome in candidate.outcomes
                ],
            }
            for candidate in forecast.candidate_predictions
        ],
        "recommended_action_id": forecast.recommended_action_id,
        "confidence": forecast.confidence,
        "source_record_ids": list(forecast.source_record_ids),
        "issued_turn": forecast.issued_turn,
        "evidence": list(forecast.evidence),
        "session_scope": forecast.session_scope,
    }


def _deserialize_preference_action_forecast(
    blob: Mapping[str, Any],
) -> PreferenceActionForecast:
    try:
        candidates_blob = blob["candidate_predictions"]
        source_record_ids = blob["source_record_ids"]
        evidence = blob["evidence"]
        if not isinstance(candidates_blob, list | tuple):
            raise HydrationPayloadInvalidError(
                "PreferenceActionForecast.candidate_predictions must be a list"
            )
        if not isinstance(source_record_ids, list | tuple):
            raise HydrationPayloadInvalidError(
                "PreferenceActionForecast.source_record_ids must be a list"
            )
        if not isinstance(evidence, list | tuple):
            raise HydrationPayloadInvalidError(
                "PreferenceActionForecast.evidence must be a list"
            )
        candidates: list[SocialActionCandidatePrediction] = []
        for candidate_blob in candidates_blob:
            if not isinstance(candidate_blob, Mapping):
                raise HydrationPayloadInvalidError(
                    "PreferenceActionForecast candidate must be an object"
                )
            outcomes_blob = candidate_blob["outcomes"]
            if not isinstance(outcomes_blob, list | tuple):
                raise HydrationPayloadInvalidError(
                    "PreferenceActionForecast candidate outcomes must be a list"
                )
            candidates.append(
                SocialActionCandidatePrediction(
                    action_id=str(candidate_blob["action_id"]),
                    outcomes=tuple(
                        SocialActionOutcomeProbability(
                            outcome_id=str(outcome_blob["outcome_id"]),
                            probability=float(outcome_blob["probability"]),
                        )
                        for outcome_blob in outcomes_blob
                    ),
                )
            )
        return PreferenceActionForecast(
            forecast_id=str(blob["forecast_id"]),
            decision_id=str(blob["decision_id"]),
            interlocutor_id=str(blob["interlocutor_id"]),
            candidate_predictions=tuple(candidates),
            recommended_action_id=str(blob["recommended_action_id"]),
            confidence=float(blob["confidence"]),
            source_record_ids=tuple(str(item) for item in source_record_ids),
            issued_turn=int(blob["issued_turn"]),
            evidence=tuple(str(item) for item in evidence),
            session_scope=str(blob.get("session_scope", "")),
        )
    except KeyError as exc:
        raise HydrationPayloadInvalidError(
            "SocialRecordStore PreferenceActionForecast missing key "
            f"{exc.args[0]!r}; blob={blob!r}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise HydrationPayloadInvalidError(
            "SocialRecordStore PreferenceActionForecast invalid value: "
            f"{exc}; blob={blob!r}"
        ) from exc


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
        "signed_utility_prediction_error": (
            settlement.signed_utility_prediction_error
        ),
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
            signed_utility_prediction_error=float(
                blob.get("signed_utility_prediction_error", 0.0)
            ),
        )
    except KeyError as exc:
        raise HydrationPayloadInvalidError(
            "SocialRecordStore PreferenceActionForecastSettlement missing key "
            f"{exc.args[0]!r}; blob={blob!r}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise HydrationPayloadInvalidError(
            "SocialRecordStore PreferenceActionForecastSettlement invalid value: "
            f"{exc}; blob={blob!r}"
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
                "CommonGroundAtom.accepted_by_ids must be a list; "
                f"got {type(accepted).__name__}"
            )
        if not isinstance(evidence, list | tuple):
            raise HydrationPayloadInvalidError(
                "CommonGroundAtom.evidence must be a list; "
                f"got {type(evidence).__name__}"
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
            f"SocialRecordStore CommonGroundAtom missing key {exc.args[0]!r}; "
            f"blob={blob!r}"
        ) from exc
    except ValueError as exc:
        raise HydrationPayloadInvalidError(
            f"SocialRecordStore CommonGroundAtom invalid enum/value: {exc}; "
            f"blob={blob!r}"
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
    "settle_pending_predictions",
]
