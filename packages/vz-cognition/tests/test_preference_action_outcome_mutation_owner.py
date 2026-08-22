from __future__ import annotations

import asyncio
import json

import pytest

from volvence_zero.owner_hydration import OwnerPersistenceSnapshot
from volvence_zero.runtime import WiringLevel
from volvence_zero.semantic_state import (
    SemanticProposal,
    SemanticProposalBatch,
    SemanticProposalOperation,
)
from volvence_zero.social import (
    PreferenceAboutOtherModule,
    PreferenceActionForecastProposal,
    PreferenceActionForecastRequest,
    SocialRecordStore,
)
from volvence_zero.social_cognition import (
    OtherMindRecord,
    OtherMindRecordKind,
    OtherMindRecordStatus,
    PreferenceActionForecast,
    PreferenceActionOutcomeEvidence,
    PreferenceActionOutcomeMutation,
    PreferenceActionOutcomeMutationOperation,
    SocialActionCandidatePrediction,
    SocialActionOutcomeProbability,
    preference_action_outcome_evidence_sha256,
)


_ACTIONS = ("stay_present", "respect_space")
_OUTCOMES = ("helped", "missed")


def _record(suffix: str, *, source_turn: int) -> OtherMindRecord:
    return OtherMindRecord(
        record_id=f"preference:{suffix}",
        interlocutor_id="primary",
        kind=OtherMindRecordKind.PREFERENCE,
        summary=f"observation before {suffix}",
        detail=f"reaction before {suffix}",
        confidence=0.8,
        status=OtherMindRecordStatus.ACTIVE,
        source_turn=source_turn,
        prediction_error_refs=(),
        evidence=f"event:{suffix}",
    )


def _outcome(suffix: str, *, source_turn: int) -> PreferenceActionOutcomeEvidence:
    return PreferenceActionOutcomeEvidence(
        evidence_id=f"preference:{suffix}",
        interlocutor_id="primary",
        observation_summary=f"observation before {suffix}",
        action_id="stay_present",
        observed_outcome_id="missed",
        reaction_summary=f"reaction before {suffix}",
        source_turn=source_turn,
        evidence_refs=(f"event:{suffix}",),
    )


def _candidate(action_id: str) -> SocialActionCandidatePrediction:
    return SocialActionCandidatePrediction(
        action_id=action_id,
        outcomes=(
            SocialActionOutcomeProbability("helped", 0.5),
            SocialActionOutcomeProbability("missed", 0.5),
        ),
    )


def _forecast(forecast_id: str, decision_id: str, source_id: str) -> PreferenceActionForecast:
    return PreferenceActionForecast(
        forecast_id=forecast_id,
        decision_id=decision_id,
        interlocutor_id="primary",
        candidate_predictions=tuple(_candidate(action) for action in _ACTIONS),
        recommended_action_id="stay_present",
        confidence=0.7,
        source_record_ids=(source_id,),
        issued_turn=3,
        evidence=("typed test forecast",),
    )


def _seed_store(*, include_keep: bool = False) -> SocialRecordStore:
    target_record = _record("target", source_turn=1)
    target_outcome = _outcome("target", source_turn=1)
    records = (target_record,)
    outcomes = (target_outcome,)
    forecasts = (_forecast("forecast:target", "decision:target", target_record.record_id),)
    if include_keep:
        keep_record = _record("keep", source_turn=2)
        keep_outcome = _outcome("keep", source_turn=2)
        records = (*records, keep_record)
        outcomes = (*outcomes, keep_outcome)
        forecasts = (
            *forecasts,
            _forecast("forecast:keep", "decision:keep", keep_record.record_id),
        )
    store = SocialRecordStore()
    store.set_tom_records("preference_about_other", records)
    store.set_preference_action_outcomes(outcomes)
    store.set_preference_action_forecasts(forecasts)
    return store


def _corrected_outcome() -> PreferenceActionOutcomeEvidence:
    return PreferenceActionOutcomeEvidence(
        evidence_id="preference:target",
        interlocutor_id="primary",
        observation_summary="corrected observation",
        action_id="stay_present",
        observed_outcome_id="helped",
        reaction_summary="corrected reaction",
        source_turn=1,
        evidence_refs=("user-correction:5",),
    )


def _correction(*, mutation_id: str = "mutation:correct:1") -> PreferenceActionOutcomeMutation:
    target = _outcome("target", source_turn=1)
    return PreferenceActionOutcomeMutation(
        mutation_id=mutation_id,
        target_evidence_id=target.evidence_id,
        expected_evidence_sha256=preference_action_outcome_evidence_sha256(target),
        operation=PreferenceActionOutcomeMutationOperation.CORRECT,
        requested_turn=5,
        evidence_refs=("console-command:5",),
        replacement=_corrected_outcome(),
    )


def _redaction(*, mutation_id: str = "mutation:redact:1") -> PreferenceActionOutcomeMutation:
    target = _outcome("target", source_turn=1)
    return PreferenceActionOutcomeMutation(
        mutation_id=mutation_id,
        target_evidence_id=target.evidence_id,
        expected_evidence_sha256=preference_action_outcome_evidence_sha256(target),
        operation=PreferenceActionOutcomeMutationOperation.REDACT,
        requested_turn=5,
        evidence_refs=("console-command:5",),
    )


class _CorrectionAwareForecastRuntime:
    runtime_id = "correction-aware-test-runtime"

    def __init__(self) -> None:
        self.seen_records: tuple[OtherMindRecord, ...] = ()
        self.seen_outcomes: tuple[PreferenceActionOutcomeEvidence, ...] = ()

    def propose(
        self,
        *,
        request: PreferenceActionForecastRequest,
        records: tuple[OtherMindRecord, ...],
        action_outcomes: tuple[PreferenceActionOutcomeEvidence, ...],
    ) -> PreferenceActionForecastProposal:
        del request
        self.seen_records = records
        self.seen_outcomes = action_outcomes
        return PreferenceActionForecastProposal(
            candidate_predictions=(
                SocialActionCandidatePrediction(
                    action_id="stay_present",
                    outcomes=(
                        SocialActionOutcomeProbability("helped", 0.8),
                        SocialActionOutcomeProbability("missed", 0.2),
                    ),
                ),
                SocialActionCandidatePrediction(
                    action_id="respect_space",
                    outcomes=(
                        SocialActionOutcomeProbability("helped", 0.3),
                        SocialActionOutcomeProbability("missed", 0.7),
                    ),
                ),
            ),
            recommended_action_id="stay_present",
            confidence=0.85,
            source_record_ids=("preference:target",),
            evidence=("corrected typed evidence",),
        )


def _request() -> PreferenceActionForecastRequest:
    return PreferenceActionForecastRequest(
        decision_id="decision:after-correction",
        interlocutor_id="primary",
        current_observation="new but structurally related situation",
        observation_ref="event:5",
        candidate_action_ids=_ACTIONS,
        outcome_ids=_OUTCOMES,
        turn_index=5,
        session_scope="subject:1",
    )


def test_correction_is_owner_applied_and_visible_to_same_turn_forecast() -> None:
    store = _seed_store()
    runtime = _CorrectionAwareForecastRuntime()
    owner = PreferenceAboutOtherModule(
        wiring_level=WiringLevel.SHADOW,
        record_store=store,
        turn_index=5,
        action_outcome_mutation=_correction(),
        action_forecast_runtime=runtime,
        action_forecast_request=_request(),
    )

    snapshot = asyncio.run(owner.process({})).value

    assert snapshot.action_outcome_evidence == (_corrected_outcome(),)
    assert snapshot.records[0].summary == "corrected observation"
    assert snapshot.records[0].detail == "corrected reaction"
    assert snapshot.records[0].evidence == "user-correction:5"
    assert runtime.seen_outcomes == (_corrected_outcome(),)
    assert runtime.seen_records == snapshot.records
    assert tuple(item.forecast_id for item in snapshot.action_forecasts) == (
        "preference_about_other:decision:after-correction:forecast:5",
    )
    receipt = snapshot.action_outcome_mutation_receipts[0]
    assert receipt.operation is PreferenceActionOutcomeMutationOperation.CORRECT
    assert receipt.invalidated_forecast_ids == ("forecast:target",)
    assert receipt.after_evidence_sha256 == preference_action_outcome_evidence_sha256(_corrected_outcome())
    pending = store.pending_tom_predictions("preference_about_other")
    assert len(pending) == 1
    assert pending[0].prediction.predicted_outcome == "corrected observation"
    assert pending[0].issued_turn == 5

    restored = SocialRecordStore()
    exported = store.export_persistence_snapshot()
    assert exported.schema_version == 4
    restored.hydrate_from_persistence(exported)
    assert restored.preference_action_outcomes == (_corrected_outcome(),)
    assert restored.preference_action_outcome_mutation_receipts == snapshot.action_outcome_mutation_receipts


def test_redaction_removes_content_persists_tombstone_and_blocks_resurrection() -> None:
    store = _seed_store(include_keep=True)
    owner = PreferenceAboutOtherModule(
        wiring_level=WiringLevel.SHADOW,
        record_store=store,
        turn_index=5,
        action_outcome_mutation=_redaction(),
    )

    snapshot = asyncio.run(owner.process({})).value

    assert tuple(record.record_id for record in snapshot.records) == ("preference:keep",)
    assert tuple(item.evidence_id for item in snapshot.action_outcome_evidence) == ("preference:keep",)
    assert tuple(item.forecast_id for item in snapshot.action_forecasts) == ("forecast:keep",)
    receipt = snapshot.action_outcome_mutation_receipts[0]
    assert receipt.operation is PreferenceActionOutcomeMutationOperation.REDACT
    assert receipt.after_evidence_sha256 is None
    assert receipt.invalidated_forecast_ids == ("forecast:target",)
    assert all(
        item.source_record_id != "preference:target" for item in store.pending_tom_predictions("preference_about_other")
    )

    exported = store.export_persistence_snapshot()
    persisted = json.dumps(exported.payload, ensure_ascii=False)
    assert "observation before target" not in persisted
    assert "reaction before target" not in persisted
    restored = SocialRecordStore()
    restored.hydrate_from_persistence(exported)
    assert restored.preference_action_outcomes == (_outcome("keep", source_turn=2),)
    assert restored.preference_action_outcome_mutation_receipts == (receipt,)

    stale_owner = PreferenceAboutOtherModule(
        proposal_runtime=_OneEvidenceProposalRuntime(),
        user_input="observation before target",
        turn_index=1,
        wiring_level=WiringLevel.SHADOW,
        record_store=restored,
        action_outcome_evidence=_outcome("target", source_turn=1),
    )
    with pytest.raises(ValueError, match="cannot be reintroduced"):
        asyncio.run(stale_owner.process({}))


def test_mutation_retry_is_idempotent_and_conflicting_id_fails() -> None:
    store = _seed_store(include_keep=True)
    command = _redaction()
    first = PreferenceAboutOtherModule(
        wiring_level=WiringLevel.SHADOW,
        record_store=store,
        turn_index=5,
        action_outcome_mutation=command,
    )
    first_snapshot = asyncio.run(first.process({})).value

    retry = PreferenceAboutOtherModule(
        wiring_level=WiringLevel.SHADOW,
        record_store=store,
        turn_index=5,
        action_outcome_mutation=command,
    )
    retry_snapshot = asyncio.run(retry.process({})).value
    assert retry_snapshot.action_outcome_mutation_receipts == (first_snapshot.action_outcome_mutation_receipts)

    conflicting = PreferenceActionOutcomeMutation(
        mutation_id=command.mutation_id,
        target_evidence_id="preference:keep",
        expected_evidence_sha256=preference_action_outcome_evidence_sha256(_outcome("keep", source_turn=2)),
        operation=PreferenceActionOutcomeMutationOperation.REDACT,
        requested_turn=5,
        evidence_refs=("different-command:5",),
    )
    with pytest.raises(ValueError, match="reused"):
        asyncio.run(
            PreferenceAboutOtherModule(
                wiring_level=WiringLevel.SHADOW,
                record_store=store,
                turn_index=5,
                action_outcome_mutation=conflicting,
            ).process({})
        )


def test_mutation_rejects_stale_hash_and_action_lineage_change() -> None:
    store = _seed_store()
    stale = PreferenceActionOutcomeMutation(
        mutation_id="mutation:stale",
        target_evidence_id="preference:target",
        expected_evidence_sha256="0" * 64,
        operation=PreferenceActionOutcomeMutationOperation.REDACT,
        requested_turn=5,
        evidence_refs=("console-command:5",),
    )
    with pytest.raises(ValueError, match="expected hash"):
        asyncio.run(
            PreferenceAboutOtherModule(
                wiring_level=WiringLevel.SHADOW,
                record_store=store,
                turn_index=5,
                action_outcome_mutation=stale,
            ).process({})
        )

    changed_action = PreferenceActionOutcomeEvidence(
        evidence_id="preference:target",
        interlocutor_id="primary",
        observation_summary="corrected observation",
        action_id="respect_space",
        observed_outcome_id="helped",
        reaction_summary="corrected reaction",
        source_turn=1,
        evidence_refs=("user-correction:5",),
    )
    command = PreferenceActionOutcomeMutation(
        mutation_id="mutation:changed-action",
        target_evidence_id="preference:target",
        expected_evidence_sha256=preference_action_outcome_evidence_sha256(_outcome("target", source_turn=1)),
        operation=PreferenceActionOutcomeMutationOperation.CORRECT,
        requested_turn=5,
        evidence_refs=("console-command:5",),
        replacement=changed_action,
    )
    with pytest.raises(ValueError, match="exposed action"):
        asyncio.run(
            PreferenceAboutOtherModule(
                wiring_level=WiringLevel.SHADOW,
                record_store=_seed_store(),
                turn_index=5,
                action_outcome_mutation=command,
            ).process({})
        )


def test_v2_hydration_defaults_mutation_receipts_to_empty() -> None:
    source = _seed_store()
    v4 = source.export_persistence_snapshot()
    v2_payload = dict(v4.payload)
    del v2_payload["preference_action_outcome_mutation_receipts"]
    legacy = OwnerPersistenceSnapshot(
        owner_name="social_record_store",
        schema_version=2,
        payload=v2_payload,
    )

    restored = SocialRecordStore()
    restored.hydrate_from_persistence(legacy)

    assert restored.preference_action_outcome_mutation_receipts == ()
    assert restored.export_persistence_snapshot().schema_version == 4


class _OneEvidenceProposalRuntime:
    runtime_id = "one-redacted-evidence-test-runtime"

    def propose(
        self,
        *,
        target_slot: str,
        user_input: str | None,
        substrate_snapshot: object | None,
        memory_snapshot: object | None,
        previous_snapshot: object | None,
        turn_index: int,
    ) -> SemanticProposalBatch:
        del substrate_snapshot, memory_snapshot, previous_snapshot
        assert target_slot == "preference_about_other"
        assert user_input == "observation before target"
        assert turn_index == 1
        return SemanticProposalBatch(
            proposals=(
                SemanticProposal(
                    proposal_id="preference:target",
                    target_slot="preference_about_other",
                    operation=SemanticProposalOperation.OBSERVE,
                    summary="observation before target",
                    detail="reaction before target",
                    confidence=0.8,
                    evidence="event:target",
                ),
            ),
            runtime_id=self.runtime_id,
            schema_version=1,
            description="Attempt to reintroduce redacted evidence.",
        )
