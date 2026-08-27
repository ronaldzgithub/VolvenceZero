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
    RelationshipProductV2CondensedTheta0FrozenPulseAuthorization,
    RelationshipProductV2ExecutorCommand,
    RelationshipProductV2FederatedCollectedCreditBatch,
    RelationshipProductV2FederatedMatchedGateTransitions,
    RelationshipProductV2ForcedCollectionAuthorization,
    RelationshipProductV2ForcedCollectionReceipt,
    RelationshipProductV2FrozenPulseAuthorization,
    RelationshipProductV2GateTransition,
    RelationshipProductV2MatchedGateTransitions,
    RelationshipProductV2OnlineExecutorCommand,
    RelationshipProductV2OnlinePulseAuthorization,
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
    prepare_relationship_product_v2_online_preaction,
    settle_relationship_product_v2_forced_collection,
    settle_relationship_product_v2_frozen_pulse,
    settle_relationship_product_v2_online_pulse,
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
    RelationshipActionGateV2OnlineSession,
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


def _online_settlement_input(preaction) -> RelationshipProductSettlementInput:
    settlement_input = _settlement_input(preaction)
    owner_evidence = settlement_input.owner_outcome_evidence
    external = replace(
        settlement_input.external_outcome,
        description=owner_evidence.reaction_summary,
    )
    return replace(
        settlement_input,
        external_outcome=external,
        owner_outcome_evidence=replace(
            owner_evidence,
            evidence_id=external.evidence_id,
        ),
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
            temporal_delivery_timestamp_ms=(
                request.forecast_request.turn_index * 1000
            ),
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
                temporal_delivery_timestamp_ms=(
                    request.forecast_request.turn_index * 1000
                ),
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


async def _federated_condensed_theta0(*, start_index: int):
    seed = _seed()
    _first_segments, first = await _collect_segmented_batch(
        seed,
        start_index=start_index,
    )
    _second_segments, second = await _collect_segmented_batch(
        seed,
        start_index=start_index + 10,
    )
    children = (first, second)
    parent = _federated_schedule(
        children,
        scope_suffix=f"condensed-theta0:{start_index}",
    )
    collection = build_relationship_product_v2_federated_collected_credit_batch(
        federated_schedule_artifact=parent,
        child_collected_batches=children,
    )
    matched = commit_relationship_product_v2_federated_matched_gate_transitions(
        artifact=seed,
        collected_batch=collection,
    )
    learned_theta0 = RelationshipActionGateV2Artifact.create_learned_theta0_from_federation(
        parent_artifact=matched.applied.artifact,
        source_batch=matched.applied.batch,
        apply_receipt=matched.applied.gate_receipt,
    )
    return seed, matched, learned_theta0


@pytest.fixture(scope="module")
def _online_theta0_bundle():
    return asyncio.run(_federated_condensed_theta0(start_index=300))


def _online_authorizations(bundle):
    _seed_artifact, matched, learned_theta0 = bundle
    theta0_authorization = RelationshipProductV2CondensedTheta0FrozenPulseAuthorization(
        pulse_authorization=_pulse_authorization(
            learned_theta0,
            suffix="online-evaluation",
        ),
        learned_theta0_artifact=learned_theta0,
        source_federated_matched_transitions=matched,
    )
    return tuple(
        RelationshipProductV2OnlinePulseAuthorization(
            theta0_authorization=theta0_authorization,
            gate_disposition=disposition,
            owner_session_scope=_SCOPE,
        )
        for disposition in (
            RelationshipActionGateBatchDisposition.APPLY,
            RelationshipActionGateBatchDisposition.WITHHOLD,
        )
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


def test_v2_forced_receipt_and_collection_provenance_replay_with_logical_time() -> None:
    artifact = _seed()
    request = _request(21)
    sibling_request = _request(210)
    schedule = RelationshipActionGateV2AssignmentScheduleArtifact(
        source_artifact_id=f"relationship-product-source-sha256:{'f' * 64}",
        schedule_scope_id="v2-logical-time-replay",
        entries=(
            RelationshipActionGateV2AssignmentScheduleEntry(
                decision_id=request.forecast_request.decision_id,
                sequence_index=0,
                assignment_role=RelationshipActionGateV2AssignmentRole.CANDIDATE,
            ),
            RelationshipActionGateV2AssignmentScheduleEntry(
                decision_id=sibling_request.forecast_request.decision_id,
                sequence_index=1,
                assignment_role=RelationshipActionGateV2AssignmentRole.NEUTRAL_NOOP,
            ),
        ),
    )
    authorization = RelationshipProductV2ForcedCollectionAuthorization(
        pulse_authorization=_pulse_authorization(
            artifact,
            suffix="logical-time-replay",
        ),
        frozen_policy=RelationshipActionGateV2(
            artifact=artifact
        ).freeze_for_evaluation(),
        assignment=RelationshipActionGateV2AssignmentReceipt(
            schedule_artifact=schedule,
            schedule_entry=schedule.entries[0],
        ),
    )
    owner_start = SocialRecordStore().export_persistence_snapshot()

    async def collect(timestamp_ms: int):
        preaction = await prepare_relationship_product_v2_forced_collection_preaction(
            request=request,
            owner_persistence_snapshot=owner_start,
            forecast_runtime=_ForecastRuntime(),
            authorization=authorization,
            substrate_snapshot=_placeholder_substrate(),
            temporal_delivery_timestamp_ms=timestamp_ms,
        )
        settlement = await settle_relationship_product_v2_forced_collection(
            preaction=preaction,
            settlement_input=_settlement_input(preaction),
        )
        return preaction, settlement

    first_preaction, first_settlement = asyncio.run(collect(421))
    replay_preaction, replay_settlement = asyncio.run(collect(421))
    changed_preaction, _changed_settlement = asyncio.run(collect(423))

    assert first_preaction.execution_receipt.to_payload() == (
        replay_preaction.execution_receipt.to_payload()
    )
    assert pulse_module._relationship_product_v2_forced_settlement_provenance_payload(
        first_settlement
    ) == pulse_module._relationship_product_v2_forced_settlement_provenance_payload(
        replay_settlement
    )
    assert first_preaction.execution_receipt.temporal_delivery.timestamp_ms == 421
    assert changed_preaction.execution_receipt.temporal_delivery.timestamp_ms == 423
    assert first_preaction.execution_receipt.command == (
        changed_preaction.execution_receipt.command
    )
    assert first_preaction.execution_receipt.receipt_id != (
        changed_preaction.execution_receipt.receipt_id
    )


@pytest.mark.parametrize("timestamp_ms", [True, 1.5, "421"])
def test_v2_forced_preaction_rejects_invalid_logical_time(timestamp_ms) -> None:
    artifact = _seed()
    request = _request(22)
    sibling_request = _request(220)
    schedule = RelationshipActionGateV2AssignmentScheduleArtifact(
        source_artifact_id=f"relationship-product-source-sha256:{'e' * 64}",
        schedule_scope_id="v2-invalid-logical-time",
        entries=(
            RelationshipActionGateV2AssignmentScheduleEntry(
                decision_id=request.forecast_request.decision_id,
                sequence_index=0,
                assignment_role=RelationshipActionGateV2AssignmentRole.CANDIDATE,
            ),
            RelationshipActionGateV2AssignmentScheduleEntry(
                decision_id=sibling_request.forecast_request.decision_id,
                sequence_index=1,
                assignment_role=RelationshipActionGateV2AssignmentRole.NEUTRAL_NOOP,
            ),
        ),
    )
    authorization = RelationshipProductV2ForcedCollectionAuthorization(
        pulse_authorization=_pulse_authorization(
            artifact,
            suffix="invalid-logical-time",
        ),
        frozen_policy=RelationshipActionGateV2(
            artifact=artifact
        ).freeze_for_evaluation(),
        assignment=RelationshipActionGateV2AssignmentReceipt(
            schedule_artifact=schedule,
            schedule_entry=schedule.entries[0],
        ),
    )

    with pytest.raises(TypeError, match="temporal_delivery_timestamp_ms"):
        asyncio.run(
            prepare_relationship_product_v2_forced_collection_preaction(
                request=request,
                owner_persistence_snapshot=(
                    SocialRecordStore().export_persistence_snapshot()
                ),
                forecast_runtime=_ForecastRuntime(),
                authorization=authorization,
                substrate_snapshot=_placeholder_substrate(),
                temporal_delivery_timestamp_ms=timestamp_ms,
            )
        )


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


def test_v2_condensed_theta0_authorization_replays_source_and_freezes_identity() -> None:
    seed, matched, learned_theta0 = asyncio.run(
        _federated_condensed_theta0(start_index=160)
    )
    pulse_authorization = _pulse_authorization(
        learned_theta0,
        suffix="condensed-theta0:authorization",
    )
    authorization = RelationshipProductV2CondensedTheta0FrozenPulseAuthorization(
        pulse_authorization=pulse_authorization,
        learned_theta0_artifact=learned_theta0,
        source_federated_matched_transitions=matched,
    )

    assert any(value != 0.0 for value in learned_theta0.weights)
    assert authorization.source_disposition is RelationshipActionGateBatchDisposition.APPLY
    assert authorization.frozen_policy.artifact == learned_theta0
    assert authorization.frozen_policy.transition_batch is None
    assert authorization.frozen_policy.transition_receipt is None
    assert authorization.frozen_policy.checkpoint.update_count == 0
    assert authorization.frozen_policy.checkpoint.informative_update_count == 0
    assert authorization.frozen_policy.checkpoint.processed_credit_ids == ()
    assert authorization.frozen_policy.checkpoint.weights == learned_theta0.weights
    payload = authorization.to_payload()
    assert matched.transitions_id == (
        "relationship-product-v2-federated-matched-transitions-sha256:"
        "af831926991d4c392aa3e35d613ea101dd8fdd4c8494331aed14e9ddf980a26b"
    )
    assert learned_theta0.artifact_id == (
        "relationship-action-gate-v2-artifact-sha256:"
        "6f8d51219edddc438160339324f391c05d74a67988db9aedcdee95ae3d8fb097"
    )
    assert authorization.authorization_id == (
        "relationship-product-v2-condensed-theta0-authorization-sha256:"
        "03cef9b06848809f9ecd0666a9660705b60b64163937e60c3bc8fd51b23b9a4f"
    )
    assert payload["source_transition_disposition"] == "apply"
    assert payload["evaluation_transition_disposition"] is None
    assert (
        RelationshipProductV2CondensedTheta0FrozenPulseAuthorization.from_payload(
            payload,
            pulse_authorization=pulse_authorization,
            learned_theta0_artifact=learned_theta0,
            source_federated_matched_transitions=matched,
        )
        == authorization
    )

    with pytest.raises(ValueError, match="learned theta0 artifact"):
        RelationshipProductV2CondensedTheta0FrozenPulseAuthorization(
            pulse_authorization=_pulse_authorization(
                seed,
                suffix="condensed-theta0:bootstrap",
            ),
            learned_theta0_artifact=seed,
            source_federated_matched_transitions=matched,
        )

    _flat_settlements, flat_batch, _flat_persistence = asyncio.run(
        _collect_batch(seed, start_index=180)
    )
    flat_matched = commit_relationship_product_v2_matched_gate_transitions(
        artifact=seed,
        collected_batch=flat_batch,
    )
    flat_theta0 = RelationshipActionGateV2Artifact.create_learned_theta0(
        parent_artifact=seed,
        source_batch=flat_batch.gate_batch,
        apply_receipt=flat_matched.applied.gate_receipt,
    )
    with pytest.raises(ValueError, match="canonical federated condensation"):
        RelationshipProductV2CondensedTheta0FrozenPulseAuthorization(
            pulse_authorization=_pulse_authorization(
                flat_theta0,
                suffix="condensed-theta0:flat",
            ),
            learned_theta0_artifact=flat_theta0,
            source_federated_matched_transitions=matched,
        )

    _other_seed, other_matched, other_theta0 = asyncio.run(
        _federated_condensed_theta0(start_index=200)
    )
    drifted_rate_theta0 = RelationshipActionGateV2Artifact._create(
        artifact_kind=learned_theta0.artifact_kind,
        weights=learned_theta0.weights,
        bootstrap_learning_rate=learned_theta0.bootstrap_learning_rate,
        online_learning_rate=learned_theta0.online_learning_rate / 2.0,
        max_abs_parameter=learned_theta0.max_abs_parameter,
        bootstrap_source_artifact_id=(
            learned_theta0.bootstrap_source_artifact_id
        ),
        source_parent_artifact_id=matched.applied.artifact.artifact_id,
        source_credit_batch_id=matched.applied.batch.batch_id,
        source_apply_receipt_id=matched.applied.gate_receipt.receipt_id,
        source_checkpoint_content_sha256=(
            matched.applied.terminal_checkpoint.content_sha256
        ),
        source_parent=matched.applied.artifact,
        source_batch=matched.applied.batch,
        source_apply_receipt=matched.applied.gate_receipt,
    )
    with pytest.raises(ValueError, match="canonical federated condensation"):
        RelationshipProductV2CondensedTheta0FrozenPulseAuthorization(
            pulse_authorization=_pulse_authorization(
                drifted_rate_theta0,
                suffix="condensed-theta0:drifted-rate",
            ),
            learned_theta0_artifact=drifted_rate_theta0,
            source_federated_matched_transitions=matched,
        )

    with pytest.raises(ValueError, match="canonical federated condensation"):
        RelationshipProductV2CondensedTheta0FrozenPulseAuthorization(
            pulse_authorization=pulse_authorization,
            learned_theta0_artifact=learned_theta0,
            source_federated_matched_transitions=other_matched,
        )
    with pytest.raises(ValueError, match="outside pulse authorization"):
        RelationshipProductV2CondensedTheta0FrozenPulseAuthorization(
            pulse_authorization=_pulse_authorization(
                seed,
                suffix="condensed-theta0:wrong-pulse-artifact",
            ),
            learned_theta0_artifact=learned_theta0,
            source_federated_matched_transitions=matched,
        )

    other_target = (
        other_matched.collected_batch.child_collected_batches[1]
        .segments[1]
        .segment_start_owner_persistence_snapshot.payload
    )
    other_replacement = (
        other_matched.collected_batch.child_collected_batches[1]
        .segments[0]
        .segment_start_owner_persistence_snapshot.payload
    )
    assert isinstance(other_target, dict)
    assert isinstance(other_replacement, dict)
    other_target.clear()
    other_target.update(dict(other_replacement))
    with pytest.raises(ValueError):
        RelationshipProductV2CondensedTheta0FrozenPulseAuthorization(
            pulse_authorization=_pulse_authorization(
                other_theta0,
                suffix="condensed-theta0:mutated-before-load",
            ),
            learned_theta0_artifact=other_theta0,
            source_federated_matched_transitions=other_matched,
        )

    frozen_authorization_id = authorization.authorization_id
    frozen_payload = authorization.to_payload()
    frozen_policy = authorization.frozen_policy
    target = (
        matched.collected_batch.child_collected_batches[1]
        .segments[1]
        .segment_start_owner_persistence_snapshot.payload
    )
    replacement_payload = (
        matched.collected_batch.child_collected_batches[1]
        .segments[0]
        .segment_start_owner_persistence_snapshot.payload
    )
    assert isinstance(target, dict)
    assert isinstance(replacement_payload, dict)
    target.clear()
    target.update(dict(replacement_payload))
    assert authorization.authorization_id == frozen_authorization_id
    assert authorization.to_payload() == frozen_payload
    assert authorization.frozen_policy == frozen_policy
    with pytest.raises(ValueError):
        RelationshipProductV2CondensedTheta0FrozenPulseAuthorization.from_payload(
            frozen_payload,
            pulse_authorization=pulse_authorization,
            learned_theta0_artifact=learned_theta0,
            source_federated_matched_transitions=matched,
        )


def test_v2_condensed_theta0_candidate_and_strict_share_exact_cold_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_artifact, matched, learned_theta0 = asyncio.run(
        _federated_condensed_theta0(start_index=220)
    )
    authorization = RelationshipProductV2CondensedTheta0FrozenPulseAuthorization(
        pulse_authorization=_pulse_authorization(
            learned_theta0,
            suffix="condensed-theta0:evaluation",
        ),
        learned_theta0_artifact=learned_theta0,
        source_federated_matched_transitions=matched,
    )

    def _reject_training_replay(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("cold evaluation attempted to replay training")

    monkeypatch.setattr(
        RelationshipActionGateV2,
        "from_applied_federated_credit_batch",
        _reject_training_replay,
    )
    persistence = SocialRecordStore().export_persistence_snapshot()
    request = _request(240)

    async def _prepare(executor_disposition):
        return await prepare_relationship_product_v2_frozen_preaction(
            request=request,
            owner_persistence_snapshot=persistence,
            forecast_runtime=_ForecastRuntime(),
            executor_disposition=executor_disposition,
            authorization=authorization,
            substrate_snapshot=_placeholder_substrate(),
        )

    candidate = asyncio.run(
        _prepare(RelationshipProductExecutorDisposition.APPLY_CANDIDATE)
    )
    strict = asyncio.run(
        _prepare(RelationshipProductExecutorDisposition.FORCE_STRICT_NOOP)
    )

    assert candidate.execution_receipt.command.command_id == (
        "relationship-product-v2-executor-command-sha256:"
        "1dc50005a613d39d42dd4cfc57fe255cebfb583f72266b0817cd6ed8aec441d7"
    )
    assert strict.execution_receipt.command.command_id == (
        "relationship-product-v2-executor-command-sha256:"
        "2eb9983f36add4bd8210e431429be09a394b023084a38fb5398e328b02d272c4"
    )
    assert candidate.forecast == strict.forecast
    assert candidate.frozen_decision == strict.frozen_decision
    assert candidate.execution_receipt.command.frozen_policy == authorization.frozen_policy
    assert strict.execution_receipt.command.frozen_policy == authorization.frozen_policy
    assert (
        candidate.execution_receipt.candidate_advisory
        == strict.execution_receipt.candidate_advisory
    )
    candidate_command = candidate.execution_receipt.command.to_payload(
        include_command_id=False
    )
    strict_command = strict.execution_receipt.command.to_payload(
        include_command_id=False
    )
    candidate_command.pop("executor_disposition")
    strict_command.pop("executor_disposition")
    assert candidate_command == strict_command
    assert candidate.frozen_decision.decision.gate_action is RelationshipGateAction.STEER
    assert candidate.delivered_action_id != RelationshipAction.NEUTRAL_NOOP.value
    assert candidate.delivered_action_id == candidate.frozen_decision.decision.selected_action_id
    assert strict.delivered_action_id == RelationshipAction.NEUTRAL_NOOP.value
    assert strict.execution_receipt.action_diverged is True
    for prepared in (candidate, strict):
        receipt_payload = prepared.execution_receipt.to_payload()
        assert receipt_payload["gate_transition_disposition"] is None
        assert receipt_payload["evaluation_gate_update_delta"] == 0
        assert prepared.execution_receipt.command.frozen_policy.checkpoint.update_count == 0


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
        assert (
            candidate.execution_receipt.to_payload()[
                "gate_transition_disposition"
            ]
            == gate_disposition.value
        )
        if gate_disposition is RelationshipActionGateBatchDisposition.APPLY:
            assert candidate.execution_receipt.command.authorization.authorization_id == (
                "relationship-product-v2-frozen-authorization-sha256:"
                "e2566018e8173364472f750296c83910cce10c17687531fab3ac4f4fe8a7443d"
            )
            assert candidate.execution_receipt.command.command_id == (
                "relationship-product-v2-executor-command-sha256:"
                "a5ec1ebba9771a945376f059f0399fd35c0712da4ca19127b4598ab542c30cab"
            )
            assert strict.execution_receipt.command.command_id == (
                "relationship-product-v2-executor-command-sha256:"
                "15cdbee1e7fdbf8bf24fd9024988abc13fd15cf5fde9ccb486b0a5c0fea83a20"
            )
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


def test_v2_online_pulse_prebinds_matched_apply_withhold_before_outcome(
    _online_theta0_bundle,
) -> None:
    applied_authorization, withheld_authorization = _online_authorizations(
        _online_theta0_bundle
    )
    learned_theta0 = applied_authorization.learned_theta0_artifact
    initial_owner = SocialRecordStore().export_persistence_snapshot()
    request = _request(400)

    async def _run_first(authorization):
        session = RelationshipActionGateV2OnlineSession(
            artifact=learned_theta0,
            disposition=authorization.gate_disposition,
        )
        preaction = await prepare_relationship_product_v2_online_preaction(
            request=request,
            owner_persistence_snapshot=initial_owner,
            forecast_runtime=_ForecastRuntime(),
            online_session=session,
            authorization=authorization,
            substrate_snapshot=_placeholder_substrate(),
            temporal_delivery_timestamp_ms=801_000,
        )
        settlement = await settle_relationship_product_v2_online_pulse(
            preaction=preaction,
            settlement_input=_online_settlement_input(preaction),
            online_session=session,
        )
        return session, preaction, settlement

    applied_session, applied_preaction, applied = asyncio.run(
        _run_first(applied_authorization)
    )
    withheld_session, withheld_preaction, withheld = asyncio.run(
        _run_first(withheld_authorization)
    )

    assert applied_authorization.theta0_authorization == (
        withheld_authorization.theta0_authorization
    )
    assert applied_authorization.authorization_id != (
        withheld_authorization.authorization_id
    )
    assert applied_preaction.forecast == withheld_preaction.forecast
    assert applied_preaction.online_exposure.frozen_decision == (
        withheld_preaction.online_exposure.frozen_decision
    )
    assert applied_preaction.delivered_action_id == withheld_preaction.delivered_action_id
    assert applied_preaction.execution_receipt.command.owner_prestate_sha256 == (
        withheld_preaction.execution_receipt.command.owner_prestate_sha256
    )
    assert isinstance(
        applied_preaction.execution_receipt.command,
        RelationshipProductV2OnlineExecutorCommand,
    )
    assert applied.common_baseline_credit == withheld.common_baseline_credit
    assert applied.owner_persistence_snapshot == withheld.owner_persistence_snapshot

    assert applied.gate_transition.receipt.generated_credit_count == 1
    assert applied.gate_transition.receipt.applied_credit_count == 1
    assert applied.gate_transition.receipt.update_count_delta == 1
    assert (
        applied.gate_transition.receipt.candidate_nonzero_parameter_update_count
        == 1
    )
    assert applied.gate_transition.receipt.applied_nonzero_parameter_update_count == 1
    assert applied.credit_applied_to_gate is True
    assert applied_session.export_checkpoint().update_count == 1
    assert applied_session.export_checkpoint().weights != learned_theta0.weights
    assert withheld.gate_transition.receipt.generated_credit_count == 1
    assert withheld.gate_transition.receipt.applied_credit_count == 0
    assert withheld.gate_transition.receipt.update_count_delta == 0
    assert (
        withheld.gate_transition.receipt.candidate_nonzero_parameter_update_count
        == 1
    )
    assert withheld.gate_transition.receipt.applied_nonzero_parameter_update_count == 0
    assert withheld.credit_applied_to_gate is False
    assert withheld_session.export_checkpoint().update_count == 0
    assert withheld_session.export_checkpoint().weights == learned_theta0.weights
    assert inspect.signature(settle_relationship_product_v2_online_pulse).parameters.keys() == {
        "preaction",
        "settlement_input",
        "online_session",
    }
    for authorization in (applied_authorization, withheld_authorization):
        payload = authorization.to_payload()
        assert payload["evaluation_or_judge_feedback_received"] is False
        assert payload["executor_disposition"] == "apply_candidate"


def test_v2_online_pulse_next_preaction_reads_prior_terminal_checkpoint(
    _online_theta0_bundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applied_authorization, withheld_authorization = _online_authorizations(
        _online_theta0_bundle
    )
    learned_theta0 = applied_authorization.learned_theta0_artifact

    def _forbid_hot_path_chain_export(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("online hot path exported or replayed a full chain")

    monkeypatch.setattr(
        RelationshipActionGateV2OnlineSession,
        "export_transition_chain",
        _forbid_hot_path_chain_export,
    )
    monkeypatch.setattr(
        RelationshipActionGateV2OnlineSession,
        "from_transition_chain",
        _forbid_hot_path_chain_export,
    )

    async def _two_preactions(authorization):
        session = RelationshipActionGateV2OnlineSession(
            artifact=learned_theta0,
            disposition=authorization.gate_disposition,
        )
        first = await prepare_relationship_product_v2_online_preaction(
            request=_request(410),
            owner_persistence_snapshot=SocialRecordStore().export_persistence_snapshot(),
            forecast_runtime=_ForecastRuntime(),
            online_session=session,
            authorization=authorization,
            substrate_snapshot=_placeholder_substrate(),
            temporal_delivery_timestamp_ms=821_000,
        )
        first_settlement = await settle_relationship_product_v2_online_pulse(
            preaction=first,
            settlement_input=_online_settlement_input(first),
            online_session=session,
        )
        terminal_chain_id = session.current_chain_id
        second = await prepare_relationship_product_v2_online_preaction(
            request=_request(411),
            owner_persistence_snapshot=first_settlement.owner_persistence_snapshot,
            forecast_runtime=_ForecastRuntime(),
            online_session=session,
            authorization=authorization,
            substrate_snapshot=_placeholder_substrate(),
            temporal_delivery_timestamp_ms=823_000,
        )
        return first_settlement, terminal_chain_id, second

    applied_first, applied_terminal_chain_id, applied_second = asyncio.run(
        _two_preactions(applied_authorization)
    )
    withheld_first, withheld_terminal_chain_id, withheld_second = asyncio.run(
        _two_preactions(withheld_authorization)
    )

    assert applied_second.parent_chain_id == applied_terminal_chain_id
    assert withheld_second.parent_chain_id == withheld_terminal_chain_id
    assert applied_second.gate_transition_count_before == 1
    assert withheld_second.gate_transition_count_before == 1
    assert applied_second.gate_checkpoint_content_sha256_before == (
        applied_first.terminal_checkpoint_content_sha256
    )
    assert withheld_second.gate_checkpoint_content_sha256_before == (
        withheld_first.terminal_checkpoint_content_sha256
    )
    assert applied_second.online_exposure.frozen_decision.decision.update_count == 1
    assert withheld_second.online_exposure.frozen_decision.decision.update_count == 0
    assert applied_second.forecast == withheld_second.forecast
    assert applied_second.delivered_action_id == withheld_second.delivered_action_id


def test_v2_online_pulse_rejects_mismatch_without_gate_mutation(
    _online_theta0_bundle,
) -> None:
    applied_authorization, withheld_authorization = _online_authorizations(
        _online_theta0_bundle
    )
    wrong_scope_session = RelationshipActionGateV2OnlineSession(
        artifact=applied_authorization.learned_theta0_artifact,
        disposition=applied_authorization.gate_disposition,
    )
    wrong_scope_checkpoint = wrong_scope_session.export_checkpoint()
    with pytest.raises(ValueError, match="owner scope differs"):
        asyncio.run(
            prepare_relationship_product_v2_online_preaction(
                request=_request(
                    419,
                    session_scope=hashlib.sha256(b"wrong-online-owner-scope").hexdigest(),
                ),
                owner_persistence_snapshot=SocialRecordStore().export_persistence_snapshot(),
                forecast_runtime=_ForecastRuntime(),
                online_session=wrong_scope_session,
                authorization=applied_authorization,
                substrate_snapshot=_placeholder_substrate(),
            )
        )
    assert wrong_scope_session.export_checkpoint() == wrong_scope_checkpoint
    assert wrong_scope_session.transition_count == 0
    assert wrong_scope_session.pending_exposure is None

    wrong_disposition_session = RelationshipActionGateV2OnlineSession(
        artifact=applied_authorization.learned_theta0_artifact,
        disposition=withheld_authorization.gate_disposition,
    )
    wrong_disposition_checkpoint = wrong_disposition_session.export_checkpoint()
    with pytest.raises(ValueError, match="disposition differs"):
        asyncio.run(
            prepare_relationship_product_v2_online_preaction(
                request=_request(419),
                owner_persistence_snapshot=SocialRecordStore().export_persistence_snapshot(),
                forecast_runtime=_ForecastRuntime(),
                online_session=wrong_disposition_session,
                authorization=applied_authorization,
                substrate_snapshot=_placeholder_substrate(),
            )
        )
    assert wrong_disposition_session.export_checkpoint() == wrong_disposition_checkpoint
    assert wrong_disposition_session.transition_count == 0
    assert wrong_disposition_session.pending_exposure is None

    session = RelationshipActionGateV2OnlineSession(
        artifact=applied_authorization.learned_theta0_artifact,
        disposition=applied_authorization.gate_disposition,
    )
    preaction = asyncio.run(
        prepare_relationship_product_v2_online_preaction(
            request=_request(420),
            owner_persistence_snapshot=SocialRecordStore().export_persistence_snapshot(),
            forecast_runtime=_ForecastRuntime(),
            online_session=session,
            authorization=applied_authorization,
            substrate_snapshot=_placeholder_substrate(),
            temporal_delivery_timestamp_ms=841_000,
        )
    )
    checkpoint_before = session.export_checkpoint()
    chain_before = session.current_chain_id
    pending_before = session.pending_exposure
    valid_input = _online_settlement_input(preaction)

    wrong_scope_authorization = replace(
        applied_authorization,
        owner_session_scope=hashlib.sha256(
            b"rewrapped-online-owner-scope"
        ).hexdigest(),
    )
    with pytest.raises(ValueError, match="owner scope differs"):
        replace(
            preaction.execution_receipt.command,
            authorization=wrong_scope_authorization,
        )

    def _assert_pending_unchanged() -> None:
        assert session.export_checkpoint() == checkpoint_before
        assert session.current_chain_id == chain_before
        assert session.pending_exposure == pending_before
        assert session.pending_plan is None
        assert session.transition_count == 0

    with pytest.raises(ValueError, match="cannot export while an exposure is pending"):
        session.export_transition_chain()

    with pytest.raises(ValueError, match="per-outcome apply bit is forbidden"):
        asyncio.run(
            settle_relationship_product_v2_online_pulse(
                preaction=preaction,
                settlement_input=replace(valid_input, apply_credit_to_gate=True),
                online_session=session,
            )
        )
    _assert_pending_unchanged()

    evidence_mutations = (
        replace(
            valid_input,
            owner_outcome_evidence=replace(
                valid_input.owner_outcome_evidence,
                evidence_id="injected-owner-evidence-id",
            ),
        ),
        replace(
            valid_input,
            owner_outcome_evidence=replace(
                valid_input.owner_outcome_evidence,
                reaction_summary="Injected evaluator reaction text.",
            ),
        ),
        replace(
            valid_input,
            owner_outcome_evidence=replace(
                valid_input.owner_outcome_evidence,
                evidence_refs=(hashlib.sha256(b"injected-evidence-ref").hexdigest(),),
            ),
        ),
    )
    for drifted in evidence_mutations:
        with pytest.raises(ValueError, match="exact environment reaction projection"):
            asyncio.run(
                settle_relationship_product_v2_online_pulse(
                    preaction=preaction,
                    settlement_input=drifted,
                    online_session=session,
                )
            )
        _assert_pending_unchanged()

    other_action = RelationshipAction.NEUTRAL_NOOP.value
    if other_action == preaction.delivered_action_id:
        other_action = RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value
    with pytest.raises(ValueError, match="external action lineage mismatch"):
        asyncio.run(
            settle_relationship_product_v2_online_pulse(
                preaction=preaction,
                settlement_input=replace(
                    valid_input,
                    external_outcome=replace(
                        valid_input.external_outcome,
                        action_id=other_action,
                    ),
                ),
                online_session=session,
            )
        )
    _assert_pending_unchanged()

    wrong_arm_checkpoint = wrong_disposition_session.export_checkpoint()
    with pytest.raises(ValueError, match="disposition differs"):
        asyncio.run(
            settle_relationship_product_v2_online_pulse(
                preaction=preaction,
                settlement_input=valid_input,
                online_session=wrong_disposition_session,
            )
        )
    assert wrong_disposition_session.export_checkpoint() == wrong_arm_checkpoint
    assert wrong_disposition_session.transition_count == 0
    assert wrong_disposition_session.pending_exposure is None
    _assert_pending_unchanged()

    with pytest.raises(ValueError, match="unresolved preaction"):
        asyncio.run(
            prepare_relationship_product_v2_online_preaction(
                request=_request(421),
                owner_persistence_snapshot=preaction.owner_persistence_snapshot,
                forecast_runtime=_ForecastRuntime(),
                online_session=session,
                authorization=applied_authorization,
                substrate_snapshot=_placeholder_substrate(),
            )
        )
    settled = asyncio.run(
        settle_relationship_product_v2_online_pulse(
            preaction=preaction,
            settlement_input=valid_input,
            online_session=session,
        )
    )
    assert settled.gate_transition_count_after == 1
    assert session.export_transition_chain().transitions == (
        settled.gate_transition,
    )
    with pytest.raises(ValueError, match="exact environment reaction projection"):
        replace(settled, settlement_input=evidence_mutations[0])
    with pytest.raises(ValueError, match="active pending preaction"):
        asyncio.run(
            settle_relationship_product_v2_online_pulse(
                preaction=preaction,
                settlement_input=valid_input,
                online_session=session,
            )
        )


def test_v2_online_pulse_detects_session_change_during_temporal_await(
    _online_theta0_bundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applied_authorization, _withheld_authorization = _online_authorizations(
        _online_theta0_bundle
    )
    session = RelationshipActionGateV2OnlineSession(
        artifact=applied_authorization.learned_theta0_artifact,
        disposition=applied_authorization.gate_disposition,
    )
    original_apply = pulse_module._apply_typed_environment_advisory

    async def _change_session_during_await(*args: object, **kwargs: object):
        snapshot = await original_apply(*args, **kwargs)
        session._chain_id = (
            "relationship-action-gate-v2-online-chain-sha256:"
            f"{'f' * 64}"
        )
        return snapshot

    monkeypatch.setattr(
        pulse_module,
        "_apply_typed_environment_advisory",
        _change_session_during_await,
    )
    with pytest.raises(RuntimeError, match="changed during temporal delivery"):
        asyncio.run(
            prepare_relationship_product_v2_online_preaction(
                request=_request(435),
                owner_persistence_snapshot=SocialRecordStore().export_persistence_snapshot(),
                forecast_runtime=_ForecastRuntime(),
                online_session=session,
                authorization=applied_authorization,
                substrate_snapshot=_placeholder_substrate(),
            )
        )
    assert session.pending_exposure is not None
    assert session.transition_count == 0


def test_v2_online_pulse_detects_public_plan_sealed_during_owner_await(
    _online_theta0_bundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applied_authorization, _withheld_authorization = _online_authorizations(
        _online_theta0_bundle
    )
    session = RelationshipActionGateV2OnlineSession(
        artifact=applied_authorization.learned_theta0_artifact,
        disposition=applied_authorization.gate_disposition,
    )
    preaction = asyncio.run(
        prepare_relationship_product_v2_online_preaction(
            request=_request(436),
            owner_persistence_snapshot=SocialRecordStore().export_persistence_snapshot(),
            forecast_runtime=_ForecastRuntime(),
            online_session=session,
            authorization=applied_authorization,
            substrate_snapshot=_placeholder_substrate(),
            temporal_delivery_timestamp_ms=871_000,
        )
    )
    settlement_input = _online_settlement_input(preaction)
    original_settle_owner = pulse_module._settle_relationship_product_owner_chain

    async def _seal_plan_during_owner_await(*args: object, **kwargs: object):
        owner_settlement = await original_settle_owner(*args, **kwargs)
        common_credit = pulse_module._derive_relationship_product_v2_common_credit(
            preaction=preaction,
            settlement_input=settlement_input,
            owner_settlement=owner_settlement,
        )
        session.plan_credit(preaction.online_exposure, common_credit)
        return owner_settlement

    monkeypatch.setattr(
        pulse_module,
        "_settle_relationship_product_owner_chain",
        _seal_plan_during_owner_await,
    )
    with pytest.raises(RuntimeError, match="changed while owner settlement was publishing"):
        asyncio.run(
            settle_relationship_product_v2_online_pulse(
                preaction=preaction,
                settlement_input=settlement_input,
                online_session=session,
            )
        )
    assert session.pending_exposure == preaction.online_exposure
    assert session.pending_plan is not None
    assert session.transition_count == 0


def test_v2_online_pulse_temporal_failure_leaves_fail_stop_pending_state(
    _online_theta0_bundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applied_authorization, _withheld_authorization = _online_authorizations(
        _online_theta0_bundle
    )
    session = RelationshipActionGateV2OnlineSession(
        artifact=applied_authorization.learned_theta0_artifact,
        disposition=applied_authorization.gate_disposition,
    )
    checkpoint_before = session.export_checkpoint()

    async def _fail_temporal_delivery(*args: object, **kwargs: object):
        del args, kwargs
        raise RuntimeError("injected temporal delivery failure")

    monkeypatch.setattr(
        pulse_module,
        "_apply_typed_environment_advisory",
        _fail_temporal_delivery,
    )
    with pytest.raises(RuntimeError, match="injected temporal delivery failure"):
        asyncio.run(
            prepare_relationship_product_v2_online_preaction(
                request=_request(430),
                owner_persistence_snapshot=SocialRecordStore().export_persistence_snapshot(),
                forecast_runtime=_ForecastRuntime(),
                online_session=session,
                authorization=applied_authorization,
                substrate_snapshot=_placeholder_substrate(),
            )
        )
    assert session.pending_exposure is not None
    assert session.transition_count == 0
    assert session.export_checkpoint() == checkpoint_before
    with pytest.raises(ValueError, match="unresolved preaction"):
        asyncio.run(
            prepare_relationship_product_v2_online_preaction(
                request=_request(431),
                owner_persistence_snapshot=SocialRecordStore().export_persistence_snapshot(),
                forecast_runtime=_ForecastRuntime(),
                online_session=session,
                authorization=applied_authorization,
                substrate_snapshot=_placeholder_substrate(),
            )
        )
