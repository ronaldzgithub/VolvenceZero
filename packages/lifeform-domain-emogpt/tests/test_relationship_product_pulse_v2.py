from __future__ import annotations

import asyncio
import hashlib
import inspect
from dataclasses import fields, replace

import pytest

import lifeform_domain_emogpt.lab.relationship_product_pulse as pulse_module
from lifeform_domain_emogpt.lab.relationship_product_pulse import (
    RelationshipProductExecutorDisposition,
    RelationshipProductOnboardingInput,
    RelationshipProductPreActionRequest,
    RelationshipProductPulseAuthorization,
    RelationshipProductSettlementInput,
    RelationshipProductV2CollectedCreditBatch,
    RelationshipProductV2CollectionSegment,
    RelationshipProductV2ExecutorCommand,
    RelationshipProductV2FederatedCollectedCreditBatch,
    RelationshipProductV2FederatedMatchedGateTransitions,
    RelationshipProductV2ForcedCollectionAuthorization,
    RelationshipProductV2ForcedCollectionReceipt,
    RelationshipProductV2FrozenPulseAuthorization,
    RelationshipProductV2GateTransition,
    RelationshipProductV2MatchedGateTransitions,
    RelationshipProductV2SegmentedCollectedCreditBatch,
    RelationshipProductV2SegmentedGateTransition,
    RelationshipProductV2SegmentedMatchedGateTransitions,
    append_relationship_product_onboarding,
    build_relationship_product_v2_collected_credit_batch,
    build_relationship_product_v2_federated_collected_credit_batch,
    build_relationship_product_v2_segmented_collected_credit_batch,
    commit_relationship_product_v2_federated_matched_gate_transitions,
    commit_relationship_product_v2_matched_gate_transitions,
    commit_relationship_product_v2_segmented_matched_gate_transitions,
    prepare_relationship_product_v2_forced_collection_preaction,
    prepare_relationship_product_v2_frozen_preaction,
    settle_relationship_product_v2_forced_collection,
    settle_relationship_product_v2_frozen_pulse,
)
from lifeform_domain_emogpt.relationship_action_contracts import (
    RELATIONSHIP_ACTIONS,
    RELATIONSHIP_OUTCOMES,
    RelationshipAction,
)
from lifeform_domain_emogpt.relationship_action_gate import (
    RelationshipActionGateBatchDisposition,
    RelationshipGateAction,
)
from lifeform_domain_emogpt.relationship_action_gate_v2 import (
    RelationshipActionGateV2,
    RelationshipActionGateV2Artifact,
    RelationshipActionGateV2AssignmentReceipt,
    RelationshipActionGateV2AssignmentRole,
    RelationshipActionGateV2AssignmentScheduleArtifact,
    RelationshipActionGateV2AssignmentScheduleEntry,
    RelationshipActionGateV2CreditBatch,
    RelationshipActionGateV2FederatedAssignmentScheduleArtifact,
    RelationshipActionGateV2FederatedCreditBatch,
    RelationshipActionGateV2FederatedScheduleSegment,
    temporal_action_advisory_from_gate_v2_decision,
)
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
)
from volvence_zero.social import (
    PreferenceActionForecastProposal,
    PreferenceActionForecastRequest,
    SocialRecordStore,
)
from volvence_zero.social_cognition import (
    PreferenceActionOutcomeEvidence,
    SocialActionCandidatePrediction,
    SocialActionOutcomeProbability,
)
from volvence_zero.substrate import SubstrateSnapshot, SurfaceKind


_ACTION_IDS = tuple(item.value for item in RELATIONSHIP_ACTIONS)
_OUTCOME_IDS = tuple(item.value for item in RELATIONSHIP_OUTCOMES)
_SCOPE = hashlib.sha256(b"relationship-product-pulse-v2-scope").hexdigest()


class _ForecastRuntime:
    runtime_id = "relationship-product-pulse-v2-test-runtime"

    def propose(self, *, request, records, action_outcomes):
        del records, action_outcomes
        return PreferenceActionForecastProposal(
            candidate_predictions=(
                _candidate(
                    RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
                    (0.1, 0.6, 0.2, 0.1),
                ),
                _candidate(
                    RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION.value,
                    (0.2, 0.2, 0.2, 0.4),
                ),
                _candidate(
                    RelationshipAction.NEUTRAL_NOOP.value,
                    (0.25, 0.25, 0.25, 0.25),
                ),
            ),
            recommended_action_id=RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
            confidence=0.8,
            source_record_ids=(),
            evidence=(f"runtime:{self.runtime_id}:{request.decision_id}",),
        )


def _candidate(
    action_id: str,
    probabilities: tuple[float, float, float, float],
) -> SocialActionCandidatePrediction:
    return SocialActionCandidatePrediction(
        action_id=action_id,
        outcomes=tuple(
            SocialActionOutcomeProbability(outcome_id, probability)
            for outcome_id, probability in zip(
                _OUTCOME_IDS,
                probabilities,
                strict=True,
            )
        ),
    )


def _seed() -> RelationshipActionGateV2Artifact:
    return RelationshipActionGateV2Artifact.create_bootstrap_seed(
        bootstrap_learning_rate=0.25,
        online_learning_rate=0.125,
        max_abs_parameter=4.0,
        bootstrap_source_artifact_id=(f"relationship-product-source-sha256:{'a' * 64}"),
    )


def _placeholder_substrate() -> SubstrateSnapshot:
    return SubstrateSnapshot(
        model_id="relationship-product-pulse-v2-test-placeholder",
        is_frozen=True,
        surface_kind=SurfaceKind.PLACEHOLDER,
        token_logits=(),
        feature_surface=(),
        residual_activations=(),
        residual_sequence=(),
        unavailable_fields=(),
        description="Typed v2 pulse test; no model or CUDA.",
    )


def _request(
    index: int,
    *,
    session_scope: str = _SCOPE,
) -> RelationshipProductPreActionRequest:
    turn = 2 * index + 1
    return RelationshipProductPreActionRequest(
        session_id=f"relationship-product-v2-session-{index}",
        forecast_request=PreferenceActionForecastRequest(
            decision_id=f"relationship-product-v2-decision-{index}",
            interlocutor_id="primary",
            current_observation=f"Public v2 relationship observation {index}.",
            observation_ref=f"public-v2-observation:{index}",
            candidate_action_ids=_ACTION_IDS,
            outcome_ids=_OUTCOME_IDS,
            turn_index=turn,
            session_scope=session_scope,
        ),
        outcome_turn_index=turn + 1,
    )


def _pulse_authorization(
    artifact: RelationshipActionGateV2Artifact,
    *,
    suffix: str,
) -> RelationshipProductPulseAuthorization:
    return RelationshipProductPulseAuthorization(
        authorization_id=hashlib.sha256(f"relationship-product-v2-authorization:{suffix}".encode("utf-8")).hexdigest(),
        allowed_policy_artifact_id=artifact.artifact_id,
        allowed_policy_artifact_version=2,
    )


def _settlement_input(preaction) -> RelationshipProductSettlementInput:
    action_id = preaction.delivered_action_id
    kind = (
        DialogueExternalOutcomeKind.MISSED
        if action_id == RelationshipAction.NEUTRAL_NOOP.value
        else DialogueExternalOutcomeKind.FELT_HEARD
    )
    evidence_ref = hashlib.sha256(
        f"{preaction.forecast.forecast_id}:{action_id}:{kind.value}".encode("utf-8")
    ).hexdigest()
    external = DialogueExternalOutcomeEvidence(
        evidence_id=f"v2-environment:{evidence_ref}",
        turn_index=preaction.request.outcome_turn_index,
        kind=kind,
        source=DialogueExternalOutcomeEvidenceSource.ENVIRONMENT,
        confidence=1.0,
        evidence_ref=evidence_ref,
        description="Typed action-conditioned v2 pulse outcome.",
        session_scope=preaction.forecast.session_scope,
        action_turn_index=preaction.forecast.issued_turn,
        forecast_id=preaction.forecast.forecast_id,
        decision_id=preaction.forecast.decision_id,
        action_id=action_id,
    )
    owner_evidence = PreferenceActionOutcomeEvidence(
        evidence_id=f"v2-owner-outcome:{preaction.forecast.decision_id}",
        interlocutor_id=preaction.forecast.interlocutor_id,
        observation_summary=preaction.request.forecast_request.current_observation,
        action_id=action_id,
        observed_outcome_id=kind.value,
        reaction_summary=f"Typed v2 reaction: {kind.value}.",
        source_turn=preaction.request.outcome_turn_index,
        evidence_refs=(evidence_ref,),
    )
    return RelationshipProductSettlementInput(
        external_outcome=external,
        owner_outcome_evidence=owner_evidence,
        credit_timestamp_ms=preaction.request.outcome_turn_index * 1000,
        apply_credit_to_gate=False,
    )


async def _collect_settlements(
    artifact: RelationshipActionGateV2Artifact,
    *,
    start_index: int,
    reset_owner_between: bool = False,
):
    requests = (_request(start_index), _request(start_index + 1))
    schedule = RelationshipActionGateV2AssignmentScheduleArtifact(
        source_artifact_id=f"relationship-product-source-sha256:{'b' * 64}",
        schedule_scope_id=f"v2-test-schedule:{start_index}",
        entries=(
            RelationshipActionGateV2AssignmentScheduleEntry(
                decision_id=requests[0].forecast_request.decision_id,
                sequence_index=0,
                assignment_role=RelationshipActionGateV2AssignmentRole.CANDIDATE,
            ),
            RelationshipActionGateV2AssignmentScheduleEntry(
                decision_id=requests[1].forecast_request.decision_id,
                sequence_index=1,
                assignment_role=RelationshipActionGateV2AssignmentRole.NEUTRAL_NOOP,
            ),
        ),
    )
    policy = RelationshipActionGateV2(artifact=artifact).freeze_for_evaluation()
    initial_persistence = SocialRecordStore().export_persistence_snapshot()
    persistence = initial_persistence
    settlements = []
    for request, entry in zip(requests, schedule.entries, strict=True):
        authorization = RelationshipProductV2ForcedCollectionAuthorization(
            pulse_authorization=_pulse_authorization(
                artifact,
                suffix=entry.decision_id,
            ),
            frozen_policy=policy,
            assignment=RelationshipActionGateV2AssignmentReceipt(
                schedule_artifact=schedule,
                schedule_entry=entry,
            ),
        )
        preaction = await prepare_relationship_product_v2_forced_collection_preaction(
            request=request,
            owner_persistence_snapshot=(initial_persistence if reset_owner_between else persistence),
            forecast_runtime=_ForecastRuntime(),
            authorization=authorization,
            substrate_snapshot=_placeholder_substrate(),
        )
        settlement = await settle_relationship_product_v2_forced_collection(
            preaction=preaction,
            settlement_input=_settlement_input(preaction),
        )
        settlements.append(settlement)
        persistence = settlement.owner_persistence_snapshot
    return tuple(settlements), persistence


async def _collect_batch(
    artifact: RelationshipActionGateV2Artifact,
    *,
    start_index: int,
):
    settlements, persistence = await _collect_settlements(
        artifact,
        start_index=start_index,
    )
    return (
        settlements,
        build_relationship_product_v2_collected_credit_batch(settlements),
        persistence,
    )


async def _collect_segmented_batch(
    artifact: RelationshipActionGateV2Artifact,
    *,
    start_index: int,
    duplicate_segment_scope: bool = False,
):
    distinct_scopes = tuple(
        hashlib.sha256(f"v2-segmented-scope:{start_index}:{segment_index}".encode("utf-8")).hexdigest()
        for segment_index in range(2)
    )
    segment_scopes = (distinct_scopes[0], distinct_scopes[0]) if duplicate_segment_scope else distinct_scopes
    requests = tuple(
        _request(
            start_index + offset,
            session_scope=segment_scopes[offset // 2],
        )
        for offset in range(4)
    )
    schedule = RelationshipActionGateV2AssignmentScheduleArtifact(
        source_artifact_id=f"relationship-product-source-sha256:{'d' * 64}",
        schedule_scope_id=f"v2-segmented-test-schedule:{start_index}",
        entries=tuple(
            RelationshipActionGateV2AssignmentScheduleEntry(
                decision_id=request.forecast_request.decision_id,
                sequence_index=offset,
                assignment_role=(
                    RelationshipActionGateV2AssignmentRole.CANDIDATE
                    if offset % 2 == 0
                    else RelationshipActionGateV2AssignmentRole.NEUTRAL_NOOP
                ),
            )
            for offset, request in enumerate(requests)
        ),
    )
    policy = RelationshipActionGateV2(artifact=artifact).freeze_for_evaluation()
    segments = []
    for segment_index, segment_requests in enumerate((requests[:2], requests[2:])):
        if segment_index == 0:
            reset = SocialRecordStore().export_persistence_snapshot()
        else:
            onboarding = await append_relationship_product_onboarding(
                owner_persistence_snapshot=None,
                onboarding=RelationshipProductOnboardingInput(
                    session_id=f"v2-segmented-onboarding:{start_index}",
                    session_index=0,
                    turn_index=0,
                    public_observation="A separate relationship root begins.",
                    action_id=RelationshipAction.NEUTRAL_NOOP.value,
                    observed_outcome_id=DialogueExternalOutcomeKind.MISSED.value,
                    reaction_summary="The separate root remained unresolved.",
                    evidence_ref=(hashlib.sha256(f"v2-segmented-reset:{start_index}".encode("utf-8")).hexdigest()),
                ),
            )
            reset = onboarding.owner_persistence_snapshot
        persistence = reset
        settlements = []
        for request in segment_requests:
            entry = schedule.entry_for_decision(request.forecast_request.decision_id)
            authorization = RelationshipProductV2ForcedCollectionAuthorization(
                pulse_authorization=_pulse_authorization(
                    artifact,
                    suffix=entry.decision_id,
                ),
                frozen_policy=policy,
                assignment=RelationshipActionGateV2AssignmentReceipt(
                    schedule_artifact=schedule,
                    schedule_entry=entry,
                ),
            )
            preaction = await prepare_relationship_product_v2_forced_collection_preaction(
                request=request,
                owner_persistence_snapshot=persistence,
                forecast_runtime=_ForecastRuntime(),
                authorization=authorization,
                substrate_snapshot=_placeholder_substrate(),
            )
            settlement = await settle_relationship_product_v2_forced_collection(
                preaction=preaction,
                settlement_input=_settlement_input(preaction),
            )
            settlements.append(settlement)
            persistence = settlement.owner_persistence_snapshot
        segments.append(
            RelationshipProductV2CollectionSegment(
                segment_scope_id=segment_scopes[segment_index],
                segment_start_owner_persistence_snapshot=reset,
                settlements=tuple(settlements),
            )
        )
    collection = build_relationship_product_v2_segmented_collected_credit_batch(tuple(segments))
    return tuple(segments), collection


def _federated_schedule(
    child_collected_batches: tuple[
        RelationshipProductV2SegmentedCollectedCreditBatch,
        ...,
    ],
    *,
    scope_suffix: str,
) -> RelationshipActionGateV2FederatedAssignmentScheduleArtifact:
    offset = 0
    segments = []
    for collection in child_collected_batches:
        child_schedule = collection.gate_batch.schedule_artifact
        segments.append(
            RelationshipActionGateV2FederatedScheduleSegment(
                global_start_index=offset,
                child_schedule_artifact=child_schedule,
            )
        )
        offset += len(child_schedule.entries)
    return RelationshipActionGateV2FederatedAssignmentScheduleArtifact(
        source_artifact_id=(child_collected_batches[0].gate_batch.schedule_artifact.source_artifact_id),
        schedule_scope_id=f"v2-pulse-federation:{scope_suffix}",
        segments=tuple(segments),
    )


async def _trained_theta_and_online_batch():
    seed = _seed()
    seed_settlements, seed_batch, _persistence = await _collect_batch(
        seed,
        start_index=0,
    )
    seed_transitions = commit_relationship_product_v2_matched_gate_transitions(
        artifact=seed,
        collected_batch=seed_batch,
    )
    theta0 = RelationshipActionGateV2Artifact.create_learned_theta0(
        parent_artifact=seed,
        source_batch=seed_batch.gate_batch,
        apply_receipt=seed_transitions.applied.gate_receipt,
    )
    online_settlements, online_batch, persistence = await _collect_batch(
        theta0,
        start_index=10,
    )
    return seed_settlements, theta0, online_settlements, online_batch, persistence


def test_v2_forced_delivery_is_derived_from_full_assignment_receipt() -> None:
    settlements, _batch, _persistence = asyncio.run(_collect_batch(_seed(), start_index=20))
    candidate, noop = settlements

    assert candidate.preaction.frozen_decision.decision.gate_action is RelationshipGateAction.NOOP
    assert candidate.preaction.delivered_action_id == (RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value)
    assert candidate.preaction.execution_receipt.temporal_delivery.active_abstract_action == (
        candidate.preaction.delivered_action_id
    )
    assert noop.preaction.delivered_action_id == RelationshipAction.NEUTRAL_NOOP.value
    assert noop.preaction.execution_receipt.temporal_delivery.active_abstract_action == (
        RelationshipAction.NEUTRAL_NOOP.value
    )
    assert candidate.preaction.execution_receipt.command.to_payload().keys() == {
        "command_id",
        "schema_version",
        "forced_exposure",
        "authorization",
        "owner_prestate_sha256",
    }

    sibling_assignment = noop.preaction.execution_receipt.command.authorization.assignment
    with pytest.raises(ValueError, match="assignment receipt drifted"):
        replace(
            candidate.preaction.execution_receipt.command,
            authorization=replace(
                candidate.preaction.execution_receipt.command.authorization,
                assignment=sibling_assignment,
            ),
        )

    command = candidate.preaction.execution_receipt.command
    assignment = command.authorization.assignment
    changed_schedule = replace(
        assignment.schedule_artifact,
        source_artifact_id=f"relationship-product-source-sha256:{'c' * 64}",
    )
    changed_assignment = RelationshipActionGateV2AssignmentReceipt(
        schedule_artifact=changed_schedule,
        schedule_entry=assignment.schedule_entry,
    )
    with pytest.raises(ValueError, match="assignment receipt drifted"):
        replace(
            command,
            authorization=replace(
                command.authorization,
                assignment=changed_assignment,
            ),
        )

    other_action = RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION.value
    receipt = candidate.preaction.execution_receipt
    with pytest.raises(ValueError, match="delivered advisory drifted"):
        replace(
            receipt,
            delivered_advisory=replace(
                receipt.delivered_advisory,
                action_id=other_action,
            ),
            temporal_delivery=replace(
                receipt.temporal_delivery,
                active_abstract_action=other_action,
            ),
        )

    assert "delivered_action_id" not in {item.name for item in fields(type(command))}
    prepare_parameters = inspect.signature(prepare_relationship_product_v2_forced_collection_preaction).parameters
    assert not {
        "delivered_action_id",
        "assignment_role",
        "schedule_artifact_id",
    }.intersection(prepare_parameters)


def test_v2_pulse_replays_policy_before_authorizing_frozen_decision() -> None:
    settlements, _batch, _persistence = asyncio.run(_collect_batch(_seed(), start_index=30))
    command = settlements[0].preaction.execution_receipt.command
    forged_decision = replace(
        command.frozen_decision.decision,
        steer_probability=0.75,
        gate_action=RelationshipGateAction.STEER,
        selected_action_id=command.forecast.recommended_action_id,
    )
    forged_frozen = replace(
        command.frozen_decision,
        decision=forged_decision,
    )

    with pytest.raises(ValueError, match="exact cold-policy replay"):
        replace(
            command,
            forced_exposure=replace(
                command.forced_exposure,
                frozen_decision=forged_frozen,
            ),
        )
    raw = temporal_action_advisory_from_gate_v2_decision(
        command.frozen_decision,
        frozen_policy=command.frozen_policy,
        forecast=command.forecast,
    )
    assert raw.active_authorized is False
    assert any("checkpoint-sha256" in item for item in raw.evidence_refs)
    assert any("preference-action-forecast-sha256" in item for item in raw.evidence_refs)
    assert "authorize_relationship_product_v2_pulse_advisory" not in pulse_module.__all__


def test_v2_forced_settlement_publishes_exact_common_baseline_credit() -> None:
    settlements, _batch, _persistence = asyncio.run(_collect_batch(_seed(), start_index=40))
    for item in settlements:
        actual = item.preaction.delivered_action_id
        assert item.settlement.action_id == actual
        assert item.credit.abstract_action_id == actual
        assert item.common_baseline_credit.external_evidence.action_id == actual
        assert item.common_baseline_credit.parent_action_credit == item.credit

    with pytest.raises(ValueError, match="common-baseline credit differs"):
        replace(
            settlements[0],
            common_baseline_credit=settlements[1].common_baseline_credit,
        )

    bad_input = replace(
        _settlement_input(settlements[0].preaction),
        apply_credit_to_gate=True,
    )
    with pytest.raises(ValueError, match="cannot apply credit online"):
        asyncio.run(
            settle_relationship_product_v2_forced_collection(
                preaction=settlements[0].preaction,
                settlement_input=bad_input,
            )
        )

    low_confidence = replace(
        _settlement_input(settlements[0].preaction),
        external_outcome=replace(
            _settlement_input(settlements[0].preaction).external_outcome,
            confidence=0.5,
        ),
    )
    with pytest.raises(ValueError, match="requires confidence 1.0"):
        asyncio.run(
            settle_relationship_product_v2_forced_collection(
                preaction=settlements[0].preaction,
                settlement_input=low_confidence,
            )
        )


def test_v2_batch_requires_complete_ordered_actual_settlements() -> None:
    settlements, collection, _persistence = asyncio.run(_collect_batch(_seed(), start_index=50))
    assert isinstance(collection, RelationshipProductV2CollectedCreditBatch)
    assert isinstance(collection.gate_batch, RelationshipActionGateV2CreditBatch)
    assert collection.gate_batch.exposures == tuple(item.forced_exposure for item in settlements)
    assert collection.gate_batch.credits == tuple(item.common_baseline_credit for item in settlements)

    with pytest.raises(ValueError, match="complete schedule in order"):
        build_relationship_product_v2_collected_credit_batch(settlements[:1])
    with pytest.raises(ValueError, match="sequence must be contiguous"):
        build_relationship_product_v2_collected_credit_batch(tuple(reversed(settlements)))

    reset_settlements, _ = asyncio.run(
        _collect_settlements(
            _seed(),
            start_index=60,
            reset_owner_between=True,
        )
    )
    with pytest.raises(ValueError, match="owner persistence handoff"):
        build_relationship_product_v2_collected_credit_batch(reset_settlements)

    mutable_settlements, mutable_collection, _ = asyncio.run(
        _collect_batch(_seed(), start_index=70)
    )
    target_payload = mutable_settlements[0].preaction.owner_input_persistence_snapshot.payload
    replacement_payload = mutable_settlements[1].preaction.owner_input_persistence_snapshot.payload
    assert isinstance(target_payload, dict)
    target_payload.clear()
    target_payload.update(dict(replacement_payload))
    with pytest.raises(ValueError):
        _ = mutable_collection.collection_id
    with pytest.raises(ValueError):
        commit_relationship_product_v2_matched_gate_transitions(
            artifact=_seed(),
            collected_batch=mutable_collection,
        )


def test_v2_segmented_batch_binds_explicit_segment_starts_and_one_schedule() -> None:
    segments, collection = asyncio.run(_collect_segmented_batch(_seed(), start_index=80))

    assert isinstance(
        collection,
        RelationshipProductV2SegmentedCollectedCreditBatch,
    )
    assert len(collection.segments) == 2
    assert len(collection.gate_batch.credits) == 4
    assert tuple(exposure.sequence_index for exposure in collection.gate_batch.exposures) == (0, 1, 2, 3)
    assert collection.to_payload().keys() == {
        "collection_id",
        "schema_version",
        "gate_batch_id",
        "segments",
    }

    matched = commit_relationship_product_v2_segmented_matched_gate_transitions(
        artifact=_seed(),
        collected_batch=collection,
    )
    assert isinstance(
        matched,
        RelationshipProductV2SegmentedMatchedGateTransitions,
    )
    assert matched.applied.gate_receipt.credit_count == 4
    assert matched.withheld.gate_receipt.update_count_delta == 0
    with pytest.raises(TypeError, match="RelationshipProductV2CollectedCreditBatch"):
        commit_relationship_product_v2_matched_gate_transitions(
            artifact=_seed(),
            collected_batch=collection,
        )
    with pytest.raises(
        TypeError,
        match="RelationshipProductV2SegmentedCollectedCreditBatch",
    ):
        commit_relationship_product_v2_segmented_matched_gate_transitions(
            artifact=_seed(),
            collected_batch=collection.gate_batch,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="explicit owner start"):
        replace(
            segments[1],
            segment_start_owner_persistence_snapshot=(segments[0].segment_start_owner_persistence_snapshot),
        )
    with pytest.raises(ValueError, match="forecast scope"):
        replace(
            segments[1],
            segment_scope_id=hashlib.sha256(b"forged-segment-scope").hexdigest(),
        )
    with pytest.raises(ValueError, match="schedule ordered"):
        build_relationship_product_v2_segmented_collected_credit_batch(tuple(reversed(segments)))
    with pytest.raises(ValueError, match="complete schedule in order"):
        build_relationship_product_v2_segmented_collected_credit_batch(segments[:1])
    with pytest.raises(ValueError, match="scope ids must be unique"):
        asyncio.run(
            _collect_segmented_batch(
                _seed(),
                start_index=90,
                duplicate_segment_scope=True,
            )
        )
    with pytest.raises(ValueError, match="persistence handoff"):
        RelationshipProductV2CollectionSegment(
            segment_scope_id=segments[0].segment_scope_id,
            segment_start_owner_persistence_snapshot=(
                segments[0].settlements[1].preaction.owner_input_persistence_snapshot
            ),
            settlements=(
                segments[0].settlements[1],
                segments[0].settlements[0],
            ),
        )

    other_segments, other_collection = asyncio.run(
        _collect_segmented_batch(_seed(), start_index=100)
    )
    other_matched = commit_relationship_product_v2_segmented_matched_gate_transitions(
        artifact=_seed(),
        collected_batch=other_collection,
    )
    with pytest.raises(ValueError, match="share one collected batch"):
        RelationshipProductV2SegmentedMatchedGateTransitions(
            applied=matched.applied,
            withheld=other_matched.withheld,
        )
    assert isinstance(other_matched.applied, RelationshipProductV2SegmentedGateTransition)

    target_payload = other_segments[1].segment_start_owner_persistence_snapshot.payload
    replacement_payload = other_segments[0].segment_start_owner_persistence_snapshot.payload
    assert isinstance(target_payload, dict)
    target_payload.clear()
    target_payload.update(dict(replacement_payload))
    with pytest.raises(ValueError):
        _ = other_collection.collection_id
    with pytest.raises(ValueError):
        commit_relationship_product_v2_segmented_matched_gate_transitions(
            artifact=_seed(),
            collected_batch=other_collection,
        )


def test_v2_federated_collection_retains_children_and_commits_only_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = _seed()
    _first_segments, first = asyncio.run(_collect_segmented_batch(seed, start_index=120))
    _second_segments, second = asyncio.run(_collect_segmented_batch(seed, start_index=130))
    children = (first, second)
    parent = _federated_schedule(children, scope_suffix="complete")
    collection = build_relationship_product_v2_federated_collected_credit_batch(
        federated_schedule_artifact=parent,
        child_collected_batches=children,
    )

    assert isinstance(
        collection,
        RelationshipProductV2FederatedCollectedCreditBatch,
    )
    assert collection.child_collected_batches == children
    assert isinstance(collection.gate_batch, RelationshipActionGateV2FederatedCreditBatch)
    assert collection.gate_batch.federated_schedule_artifact == parent
    assert collection.gate_batch.child_batches == tuple(item.gate_batch for item in children)
    assert tuple(item.global_start_index for item in parent.segments) == (0, 4)
    assert collection.gate_batch.exposures == tuple(
        exposure for child in children for exposure in child.gate_batch.exposures
    )
    assert collection.gate_batch.credits == tuple(credit for child in children for credit in child.gate_batch.credits)
    timestamps = tuple(item.parent_action_credit.timestamp_ms for item in collection.gate_batch.credits)
    assert all(left < right for left, right in zip(timestamps, timestamps[1:], strict=False))
    payload = collection.to_payload()
    assert payload.keys() == {
        "collection_id",
        "schema_version",
        "federated_schedule_artifact_id",
        "gate_batch_id",
        "child_collection_count",
        "credit_count",
        "child_collections",
    }
    assert (
        RelationshipProductV2FederatedCollectedCreditBatch.from_payload(
            payload,
            federated_schedule_artifact=parent,
            full_child_collected_batches=children,
        )
        == collection
    )
    tampered_payload = dict(payload)
    tampered_payload["credit_count"] = 7
    with pytest.raises(ValueError, match="payload mismatch"):
        RelationshipProductV2FederatedCollectedCreditBatch.from_payload(
            tampered_payload,
            federated_schedule_artifact=parent,
            full_child_collected_batches=children,
        )
    typed_drift_payload = dict(payload)
    typed_drift_payload["credit_count"] = 8.0
    with pytest.raises(ValueError, match="payload mismatch"):
        RelationshipProductV2FederatedCollectedCreditBatch.from_payload(
            typed_drift_payload,
            federated_schedule_artifact=parent,
            full_child_collected_batches=children,
        )
    container_drift_payload = dict(payload)
    child_payloads = container_drift_payload["child_collections"]
    assert isinstance(child_payloads, list)
    container_drift_payload["child_collections"] = tuple(child_payloads)
    with pytest.raises(ValueError, match="payload mismatch"):
        RelationshipProductV2FederatedCollectedCreditBatch.from_payload(
            container_drift_payload,
            federated_schedule_artifact=parent,
            full_child_collected_batches=children,
        )

    def _reject_flat_child_commit(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("pulse federation attempted a flat child transition")

    monkeypatch.setattr(
        RelationshipActionGateV2,
        "commit_credit_batch",
        _reject_flat_child_commit,
    )
    matched = commit_relationship_product_v2_federated_matched_gate_transitions(
        artifact=seed,
        collected_batch=collection,
    )
    assert isinstance(
        matched,
        RelationshipProductV2FederatedMatchedGateTransitions,
    )
    assert matched.collected_batch is collection
    assert matched.applied.batch == collection.gate_batch
    assert matched.withheld.batch == collection.gate_batch
    assert matched.applied.gate_receipt.plan_id == matched.withheld.gate_receipt.plan_id
    assert (
        matched.applied.gate_receipt.pre_checkpoint_content_sha256
        == matched.withheld.gate_receipt.pre_checkpoint_content_sha256
    )
    assert (
        matched.applied.gate_receipt.candidate_checkpoint_content_sha256
        == matched.withheld.gate_receipt.candidate_checkpoint_content_sha256
    )
    assert matched.applied.gate_receipt.atomic_commit_count == 1
    assert matched.applied.gate_receipt.update_count_delta == 8
    assert matched.applied.gate_receipt.child_batch_count == 2
    assert matched.applied.gate_receipt.child_transition_count == 0
    assert matched.withheld.gate_receipt.atomic_commit_count == 0
    assert matched.withheld.gate_receipt.update_count_delta == 0
    assert matched.withheld.gate_receipt.child_transition_count == 0
    assert matched.applied.terminal_checkpoint.update_count == 8
    assert matched.withheld.terminal_checkpoint.update_count == 0
    matched_payload = matched.to_payload()
    assert matched_payload["child_transition_count"] == 0
    assert (
        RelationshipProductV2FederatedMatchedGateTransitions.from_payload(
            matched_payload,
            collected_batch=collection,
            gate_matched_transitions=matched.gate_matched_transitions,
        )
        == matched
    )
    typed_drift_matched_payload = dict(matched_payload)
    typed_drift_matched_payload["child_transition_count"] = False
    with pytest.raises(ValueError, match="payload mismatch"):
        RelationshipProductV2FederatedMatchedGateTransitions.from_payload(
            typed_drift_matched_payload,
            collected_batch=collection,
            gate_matched_transitions=matched.gate_matched_transitions,
        )
    with pytest.raises(
        TypeError,
        match="RelationshipProductV2FederatedCollectedCreditBatch",
    ):
        commit_relationship_product_v2_federated_matched_gate_transitions(
            artifact=seed,
            collected_batch=collection.gate_batch,  # type: ignore[arg-type]
        )


def test_v2_federated_collection_rejects_partial_reorder_credit_time_and_mutation() -> None:
    seed = _seed()
    first_segments, first = asyncio.run(_collect_segmented_batch(seed, start_index=140))
    second_segments, second = asyncio.run(_collect_segmented_batch(seed, start_index=150))
    children = (first, second)
    parent = _federated_schedule(children, scope_suffix="rejection")

    with pytest.raises(ValueError, match="at least two exact child collections"):
        build_relationship_product_v2_federated_collected_credit_batch(
            federated_schedule_artifact=parent,
            child_collected_batches=(first,),
        )
    with pytest.raises(ValueError, match="parent order"):
        build_relationship_product_v2_federated_collected_credit_batch(
            federated_schedule_artifact=parent,
            child_collected_batches=tuple(reversed(children)),
        )

    _earlier_segments, earlier = asyncio.run(_collect_segmented_batch(seed, start_index=135))
    time_reversed_children = (first, earlier)
    time_reversed_parent = _federated_schedule(
        time_reversed_children,
        scope_suffix="time-reversed",
    )
    with pytest.raises(ValueError, match="globally increasing"):
        build_relationship_product_v2_federated_collected_credit_batch(
            federated_schedule_artifact=time_reversed_parent,
            child_collected_batches=time_reversed_children,
        )

    collection = build_relationship_product_v2_federated_collected_credit_batch(
        federated_schedule_artifact=parent,
        child_collected_batches=children,
    )
    target_payload = second_segments[1].segment_start_owner_persistence_snapshot.payload
    replacement_payload = first_segments[0].segment_start_owner_persistence_snapshot.payload
    assert isinstance(target_payload, dict)
    assert isinstance(replacement_payload, dict)
    target_payload.clear()
    target_payload.update(dict(replacement_payload))
    with pytest.raises(ValueError):
        _ = collection.collection_id
    with pytest.raises(ValueError):
        collection.to_payload()
    with pytest.raises(ValueError):
        commit_relationship_product_v2_federated_matched_gate_transitions(
            artifact=seed,
            collected_batch=collection,
        )


def test_v2_apply_withhold_transition_is_full_and_replayable() -> None:
    _seed_settlements, theta0, _online_settlements, collection, _persistence = asyncio.run(
        _trained_theta_and_online_batch()
    )
    matched = commit_relationship_product_v2_matched_gate_transitions(
        artifact=theta0,
        collected_batch=collection,
    )
    replayed = commit_relationship_product_v2_matched_gate_transitions(
        artifact=theta0,
        collected_batch=collection,
    )

    assert isinstance(matched, RelationshipProductV2MatchedGateTransitions)
    assert matched == replayed
    assert matched.applied.frozen_policy.checkpoint.update_count == len(collection.gate_batch.credits)
    assert matched.withheld.frozen_policy.checkpoint.update_count == 0
    assert matched.withheld.frozen_policy.transition_receipt is not None
    assert (
        RelationshipProductV2GateTransition(
            collected_batch=collection,
            gate_receipt=matched.applied.gate_receipt,
            frozen_policy=matched.applied.frozen_policy,
        )
        == matched.applied
    )
    with pytest.raises(ValueError, match="full batch/receipt"):
        RelationshipProductV2GateTransition(
            collected_batch=collection,
            gate_receipt=matched.withheld.gate_receipt,
            frozen_policy=matched.applied.frozen_policy,
        )

    with pytest.raises(TypeError, match="collected_batch"):
        commit_relationship_product_v2_matched_gate_transitions(
            artifact=theta0,
            collected_batch=collection.gate_batch,
        )

    _other_settlements, other_collection, _ = asyncio.run(_collect_batch(theta0, start_index=70))
    other_matched = commit_relationship_product_v2_matched_gate_transitions(
        artifact=theta0,
        collected_batch=other_collection,
    )
    with pytest.raises(ValueError, match="share one collected batch"):
        RelationshipProductV2MatchedGateTransitions(
            applied=matched.applied,
            withheld=other_matched.withheld,
        )


def test_v2_strict_noop_changes_only_executor_bit() -> None:
    _seed_settlements, theta0, _online_settlements, collection, persistence = asyncio.run(
        _trained_theta_and_online_batch()
    )
    matched = commit_relationship_product_v2_matched_gate_transitions(
        artifact=theta0,
        collected_batch=collection,
    )
    request = _request(100)

    async def _prepare(gate_disposition, executor_disposition):
        authorization = RelationshipProductV2FrozenPulseAuthorization(
            pulse_authorization=_pulse_authorization(
                theta0,
                suffix=f"evaluation:{gate_disposition.value}",
            ),
            matched_transitions=matched,
            gate_disposition=gate_disposition,
        )
        return await prepare_relationship_product_v2_frozen_preaction(
            request=request,
            owner_persistence_snapshot=persistence,
            forecast_runtime=_ForecastRuntime(),
            executor_disposition=executor_disposition,
            authorization=authorization,
            substrate_snapshot=_placeholder_substrate(),
        )

    prepared = {}
    for gate_disposition in (
        RelationshipActionGateBatchDisposition.APPLY,
        RelationshipActionGateBatchDisposition.WITHHOLD,
    ):
        candidate = asyncio.run(
            _prepare(
                gate_disposition,
                RelationshipProductExecutorDisposition.APPLY_CANDIDATE,
            )
        )
        strict = asyncio.run(
            _prepare(
                gate_disposition,
                RelationshipProductExecutorDisposition.FORCE_STRICT_NOOP,
            )
        )
        assert candidate.execution_receipt.command.frozen_policy == strict.execution_receipt.command.frozen_policy
        assert candidate.forecast == strict.forecast
        assert candidate.frozen_decision == strict.frozen_decision
        assert candidate.execution_receipt.candidate_advisory == strict.execution_receipt.candidate_advisory
        candidate_payload = candidate.execution_receipt.command.to_payload(include_command_id=False)
        strict_payload = strict.execution_receipt.command.to_payload(include_command_id=False)
        candidate_payload.pop("executor_disposition")
        strict_payload.pop("executor_disposition")
        assert candidate_payload == strict_payload
        assert candidate.frozen_decision.decision.gate_action is RelationshipGateAction.STEER
        assert candidate.delivered_action_id != RelationshipAction.NEUTRAL_NOOP.value
        assert candidate.delivered_action_id == candidate.frozen_decision.decision.selected_action_id
        assert strict.delivered_action_id == RelationshipAction.NEUTRAL_NOOP.value
        assert strict.execution_receipt.action_diverged is True
        prepared[gate_disposition] = (candidate, strict)

    applied, strict = prepared[RelationshipActionGateBatchDisposition.WITHHOLD]

    applied_settlement = asyncio.run(
        settle_relationship_product_v2_frozen_pulse(
            preaction=applied,
            settlement_input=_settlement_input(applied),
        )
    )
    strict_settlement = asyncio.run(
        settle_relationship_product_v2_frozen_pulse(
            preaction=strict,
            settlement_input=_settlement_input(strict),
        )
    )
    assert applied_settlement.common_baseline_credit.external_evidence.action_id == applied.delivered_action_id
    assert strict_settlement.common_baseline_credit.external_evidence.action_id == (
        RelationshipAction.NEUTRAL_NOOP.value
    )

    seed = _seed()
    _cold_settlements, cold_collection, _ = asyncio.run(_collect_batch(seed, start_index=110))
    cold_matched = commit_relationship_product_v2_matched_gate_transitions(
        artifact=seed,
        collected_batch=cold_collection,
    )
    with pytest.raises(ValueError, match="learned theta0"):
        RelationshipProductV2FrozenPulseAuthorization(
            pulse_authorization=_pulse_authorization(seed, suffix="cold"),
            matched_transitions=cold_matched,
            gate_disposition=RelationshipActionGateBatchDisposition.WITHHOLD,
        )

    assert isinstance(applied.execution_receipt.command, RelationshipProductV2ExecutorCommand)
    assert isinstance(
        _online_settlements[0].preaction.execution_receipt,
        RelationshipProductV2ForcedCollectionReceipt,
    )
