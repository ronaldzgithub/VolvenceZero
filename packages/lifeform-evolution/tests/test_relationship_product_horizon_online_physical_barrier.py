from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import pathlib
import threading

import pytest

import lifeform_evolution.relationship_product_horizon_development_campaign as subject
from lifeform_domain_emogpt.lab.relationship_product_pulse import (
    RelationshipProductPreActionRequest,
    RelationshipProductPulseAuthorization,
    RelationshipProductSettlementInput,
    RelationshipProductV2CollectionSegment,
    RelationshipProductV2CondensedTheta0FrozenPulseAuthorization,
    RelationshipProductV2ForcedCollectionAuthorization,
    RelationshipProductV2OnlinePulseAuthorization,
    build_relationship_product_v2_federated_collected_credit_batch,
    build_relationship_product_v2_segmented_collected_credit_batch,
    commit_relationship_product_v2_federated_matched_gate_transitions,
    prepare_relationship_product_v2_forced_collection_preaction,
    settle_relationship_product_v2_forced_collection,
)
from lifeform_domain_emogpt.relationship_action_contracts import (
    RELATIONSHIP_ACTIONS,
    RELATIONSHIP_OUTCOMES,
    RelationshipAction,
)
from lifeform_domain_emogpt.relationship_action_gate import (
    RelationshipActionGateBatchDisposition,
)
from lifeform_domain_emogpt.relationship_action_gate_v2 import (
    RelationshipActionGateV2,
    RelationshipActionGateV2Artifact,
    RelationshipActionGateV2AssignmentReceipt,
    RelationshipActionGateV2AssignmentRole,
    RelationshipActionGateV2AssignmentScheduleArtifact,
    RelationshipActionGateV2AssignmentScheduleEntry,
    RelationshipActionGateV2FederatedAssignmentScheduleArtifact,
    RelationshipActionGateV2FederatedScheduleSegment,
    RelationshipActionGateV2OnlineSession,
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
_SOURCE_ARTIFACT_ID = f"relationship-product-source-sha256:{'a' * 64}"
_SOURCE_CAPABILITY_ID = "synthetic-online-physical-source:v1"


class _ForecastRuntime:
    runtime_id = "relationship-product-online-physical-test-runtime"

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
            recommended_action_id=(
                RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value
            ),
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


def _placeholder_substrate() -> SubstrateSnapshot:
    return SubstrateSnapshot(
        model_id="relationship-product-online-physical-placeholder",
        is_frozen=True,
        surface_kind=SurfaceKind.PLACEHOLDER,
        token_logits=(),
        feature_surface=(),
        residual_activations=(),
        residual_sequence=(),
        unavailable_fields=(),
        description="Typed physical barrier unit test; no model or CUDA.",
    )


def _request(
    index: int,
    *,
    scope: str,
) -> RelationshipProductPreActionRequest:
    turn = 2 * index + 1
    return RelationshipProductPreActionRequest(
        session_id=f"online-physical-session:{scope}:{index}",
        forecast_request=PreferenceActionForecastRequest(
            decision_id=f"online-physical-decision:{index}",
            interlocutor_id="primary",
            current_observation=f"Shared public relationship observation {index}.",
            observation_ref=f"online-physical-observation:{index}",
            candidate_action_ids=_ACTION_IDS,
            outcome_ids=_OUTCOME_IDS,
            turn_index=turn,
            session_scope=scope,
        ),
        outcome_turn_index=turn + 1,
    )


def _pulse_authorization(
    artifact: RelationshipActionGateV2Artifact,
    *,
    suffix: str,
) -> RelationshipProductPulseAuthorization:
    return RelationshipProductPulseAuthorization(
        authorization_id=hashlib.sha256(
            f"online-physical-authorization:{suffix}".encode("utf-8")
        ).hexdigest(),
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
        (
            f"{preaction.forecast.session_scope}:"
            f"{preaction.forecast.forecast_id}:{action_id}:{kind.value}"
        ).encode("utf-8")
    ).hexdigest()
    reaction = f"Typed online physical reaction: {kind.value}."
    external = DialogueExternalOutcomeEvidence(
        evidence_id=f"online-physical-environment:{evidence_ref}",
        turn_index=preaction.request.outcome_turn_index,
        kind=kind,
        source=DialogueExternalOutcomeEvidenceSource.ENVIRONMENT,
        confidence=1.0,
        evidence_ref=evidence_ref,
        description=reaction,
        session_scope=preaction.forecast.session_scope,
        action_turn_index=preaction.forecast.issued_turn,
        forecast_id=preaction.forecast.forecast_id,
        decision_id=preaction.forecast.decision_id,
        action_id=action_id,
    )
    owner = PreferenceActionOutcomeEvidence(
        evidence_id=external.evidence_id,
        interlocutor_id=preaction.forecast.interlocutor_id,
        observation_summary=preaction.request.forecast_request.current_observation,
        action_id=action_id,
        observed_outcome_id=kind.value,
        reaction_summary=reaction,
        source_turn=preaction.request.outcome_turn_index,
        evidence_refs=(evidence_ref,),
    )
    return RelationshipProductSettlementInput(
        external_outcome=external,
        owner_outcome_evidence=owner,
        credit_timestamp_ms=preaction.request.outcome_turn_index * 1000,
        apply_credit_to_gate=False,
    )


async def _child_collection(
    artifact: RelationshipActionGateV2Artifact,
    *,
    start_index: int,
):
    scope = hashlib.sha256(
        f"online-physical-child:{start_index}".encode("utf-8")
    ).hexdigest()
    requests = (
        _request(start_index, scope=scope),
        _request(start_index + 1, scope=scope),
    )
    schedule = RelationshipActionGateV2AssignmentScheduleArtifact(
        source_artifact_id=_SOURCE_ARTIFACT_ID,
        schedule_scope_id=f"online-physical-child-schedule:{start_index}",
        entries=tuple(
            RelationshipActionGateV2AssignmentScheduleEntry(
                decision_id=request.forecast_request.decision_id,
                sequence_index=offset,
                assignment_role=(
                    RelationshipActionGateV2AssignmentRole.CANDIDATE
                    if offset == 0
                    else RelationshipActionGateV2AssignmentRole.NEUTRAL_NOOP
                ),
            )
            for offset, request in enumerate(requests)
        ),
    )
    policy = RelationshipActionGateV2(artifact=artifact).freeze_for_evaluation()
    owner_start = SocialRecordStore().export_persistence_snapshot()
    owner = owner_start
    settlements = []
    for request, entry in zip(requests, schedule.entries, strict=True):
        authorization = RelationshipProductV2ForcedCollectionAuthorization(
            pulse_authorization=_pulse_authorization(
                artifact,
                suffix=request.forecast_request.decision_id,
            ),
            frozen_policy=policy,
            assignment=RelationshipActionGateV2AssignmentReceipt(
                schedule_artifact=schedule,
                schedule_entry=entry,
            ),
        )
        preaction = await prepare_relationship_product_v2_forced_collection_preaction(
            request=request,
            owner_persistence_snapshot=owner,
            forecast_runtime=_ForecastRuntime(),
            authorization=authorization,
            substrate_snapshot=_placeholder_substrate(),
            temporal_delivery_timestamp_ms=request.forecast_request.turn_index * 1000,
        )
        settlement = await settle_relationship_product_v2_forced_collection(
            preaction=preaction,
            settlement_input=_settlement_input(preaction),
        )
        settlements.append(settlement)
        owner = settlement.owner_persistence_snapshot
    segment = RelationshipProductV2CollectionSegment(
        segment_scope_id=scope,
        segment_start_owner_persistence_snapshot=owner_start,
        settlements=tuple(settlements),
    )
    return build_relationship_product_v2_segmented_collected_credit_batch(
        (segment,)
    )


@pytest.fixture(scope="module")
def online_bindings():
    async def build():
        seed = RelationshipActionGateV2Artifact.create_bootstrap_seed(
            bootstrap_learning_rate=0.25,
            online_learning_rate=0.125,
            max_abs_parameter=4.0,
            bootstrap_source_artifact_id=_SOURCE_ARTIFACT_ID,
        )
        children = (
            await _child_collection(seed, start_index=100),
            await _child_collection(seed, start_index=110),
        )
        parent = RelationshipActionGateV2FederatedAssignmentScheduleArtifact(
            source_artifact_id=_SOURCE_ARTIFACT_ID,
            schedule_scope_id="online-physical-parent-schedule",
            segments=(
                RelationshipActionGateV2FederatedScheduleSegment(
                    global_start_index=0,
                    child_schedule_artifact=children[0].gate_batch.schedule_artifact,
                ),
                RelationshipActionGateV2FederatedScheduleSegment(
                    global_start_index=len(
                        children[0].gate_batch.schedule_artifact.entries
                    ),
                    child_schedule_artifact=children[1].gate_batch.schedule_artifact,
                ),
            ),
        )
        collection = build_relationship_product_v2_federated_collected_credit_batch(
            federated_schedule_artifact=parent,
            child_collected_batches=children,
        )
        matched = commit_relationship_product_v2_federated_matched_gate_transitions(
            artifact=seed,
            collected_batch=collection,
        )
        theta0 = RelationshipActionGateV2Artifact.create_learned_theta0_from_federation(
            parent_artifact=matched.applied.artifact,
            source_batch=matched.applied.batch,
            apply_receipt=matched.applied.gate_receipt,
        )
        theta0_authorization = (
            RelationshipProductV2CondensedTheta0FrozenPulseAuthorization(
                pulse_authorization=_pulse_authorization(
                    theta0,
                    suffix="online-physical-evaluation",
                ),
                learned_theta0_artifact=theta0,
                source_federated_matched_transitions=matched,
            )
        )
        owner_start = SocialRecordStore().export_persistence_snapshot()
        forecast_runtime = _ForecastRuntime()
        return tuple(
            subject.RelationshipProductHorizonOnlineArmBinding(
                arm_id=arm,
                authorization=RelationshipProductV2OnlinePulseAuthorization(
                    theta0_authorization=theta0_authorization,
                    gate_disposition=disposition,
                    owner_session_scope=hashlib.sha256(
                        f"online-physical-owner:{arm.value}".encode("utf-8")
                    ).hexdigest(),
                ),
                initial_owner_persistence_snapshot=owner_start,
                forecast_runtime=forecast_runtime,
            )
            for arm, disposition in (
                (
                    subject.RelationshipProductHorizonOnlineArm.FULL,
                    RelationshipActionGateBatchDisposition.APPLY,
                ),
                (
                    subject.RelationshipProductHorizonOnlineArm.FROZEN_THETA0,
                    RelationshipActionGateBatchDisposition.WITHHOLD,
                ),
            )
        )

    return asyncio.run(build())


class _Source:
    def __init__(
        self,
        events: list[str] | None = None,
        *,
        fail_open: bool = False,
        fail_settle: bool = False,
        duplicate_branch: bool = False,
    ):
        self.events = events if events is not None else []
        self.fail_open = fail_open
        self.fail_settle = fail_settle
        self.duplicate_branch = duplicate_branch
        self.open_count = 0
        self.call_count = 0
        self.request_payloads: list[dict[str, object]] = []

    async def open(self, capability):
        self.events.append(f"open:{capability.slot_index}")
        self.open_count += 1
        if self.fail_open:
            raise RuntimeError("injected source open failure")
        return self

    async def settle_actions(self, *, request):
        self.events.append(f"source:{request.open_capability.slot_index}")
        self.call_count += 1
        self.request_payloads.append(dict(request.to_payload()))
        if self.fail_settle:
            raise RuntimeError("injected source failure")
        branches = tuple(
            _source_branch(request=request, action=action)
            for action in request.selected_actions
        )
        if self.duplicate_branch:
            return (*branches, branches[0])
        return branches

    def descriptor(self):
        return subject.RelationshipProductHorizonOnlineSettlementSourceDescriptor(
            source_capability_id=_SOURCE_CAPABILITY_ID,
            open_source=self.open,
        )


def _source_branch(*, request, action):
    kind = (
        DialogueExternalOutcomeKind.MISSED
        if action is RelationshipAction.NEUTRAL_NOOP
        else DialogueExternalOutcomeKind.FELT_HEARD
    )
    evidence_ref = hashlib.sha256(
        f"{request.source_request_id}:{action.value}:{kind.value}".encode("utf-8")
    ).hexdigest()
    return subject.RelationshipProductHorizonOnlineSourceBranch(
        source_request_id=request.source_request_id,
        source_capability_id=_SOURCE_CAPABILITY_ID,
        selected_action=action,
        typed_outcome=kind,
        rendered_user_reaction=f"Typed online physical reaction: {kind.value}.",
        environment_evidence_ref=evidence_ref,
        environment_version="synthetic-online-physical-source.v1",
    )


def _requests(bindings, slot_index: int):
    return tuple(
        (
            binding.arm_id,
            _request(
                500 + slot_index,
                scope=binding.authorization.owner_session_scope,
            ),
        )
        for binding in bindings
    )


def test_two_slot_pair_is_durable_before_source_and_next_and_scans_exact(
    online_bindings,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    original_fsync = subject.os.fsync
    original_prepare = subject.prepare_relationship_product_v2_online_preaction
    original_commit = RelationshipActionGateV2OnlineSession.commit_credit

    def fsync(fd: int) -> None:
        events.append("fsync")
        original_fsync(fd)

    async def prepare(**kwargs):
        events.append(f"prepare:{kwargs['online_session'].transition_count}")
        return await original_prepare(**kwargs)

    def commit(session, plan):
        events.append(f"commit:{session.disposition.value}")
        return original_commit(session, plan)

    monkeypatch.setattr(subject.os, "fsync", fsync)
    monkeypatch.setattr(
        subject,
        "prepare_relationship_product_v2_online_preaction",
        prepare,
    )
    monkeypatch.setattr(RelationshipActionGateV2OnlineSession, "commit_credit", commit)

    path = tmp_path / "online-physical.jsonl"
    source = _Source(events)
    barrier = subject.RelationshipProductHorizonOnlinePhysicalBarrier(
        path=path,
        mechanism_run_id="online-physical-test-run",
        root_sequence_index=7,
        bindings=online_bindings,
        source_descriptor=source.descriptor(),
        substrate_snapshot=_placeholder_substrate(),
        expected_slot_count=2,
    )
    header_fsync_index = events.index("fsync")
    first = asyncio.run(
        barrier.execute_slot(
            requests=_requests(online_bindings, 0),
            temporal_delivery_timestamp_ms=1_001_000,
        )
    )
    second = asyncio.run(
        barrier.execute_slot(
            requests=_requests(online_bindings, 1),
            temporal_delivery_timestamp_ms=1_003_000,
        )
    )
    barrier.close()

    first_source = events.index("source:0")
    first_open = events.index("open:0")
    first_commits = [
        index
        for index, event in enumerate(events)
        if event.startswith("commit:")
    ][:2]
    second_prepare = events.index("prepare:1")
    assert header_fsync_index < events.index("prepare:0")
    assert events[:first_open].count("fsync") == 3
    assert first_open < first_source
    assert first_source < min(first_commits)
    assert events[:second_prepare].count("fsync") == 5
    assert first.next_slot_authorized is True
    assert first.ledger_complete is False
    assert first.terminal_row_id is None
    assert second.next_slot_authorized is False
    assert second.ledger_complete is True
    assert second.terminal_row_id is not None
    assert (
        subject.RelationshipProductHorizonOnlineSlotCompletion.from_payload(
            second.to_payload()
        )
        == second
    )
    assert source.call_count == 2
    assert source.open_count == 1
    assert len(source.request_payloads[0]["selected_actions"]) == 1
    assert set(source.request_payloads[0]) == {
        "schema_version",
        "open_capability",
        "decision_id",
        "interlocutor_id",
        "current_observation",
        "observation_ref",
        "candidate_action_ids",
        "outcome_ids",
        "turn_index",
        "outcome_turn_index",
        "selected_actions",
    }
    open_capability_payload = source.request_payloads[0]["open_capability"]
    assert set(open_capability_payload) == {
        "schema_version",
        "source_capability_id",
        "mechanism_run_id",
        "root_sequence_index",
        "slot_index",
    }
    assert {
        "barrier_id",
        "barrier_receipt_row_id",
        "stream_prefix_raw_sha256",
    }.isdisjoint(open_capability_payload)
    persisted_rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    for boundary_row in (persisted_rows[0], persisted_rows[-1]):
        assert (
            boundary_row[
                "forecast_runtime_object_identity_shared_in_live_constructor"
            ]
            is True
        )
        assert (
            boundary_row["forecast_runtime_arm_invariance_verified_by_mechanism"]
            is False
        )
        assert (
            boundary_row[
                "forecast_runtime_session_scope_blinding_verified_by_mechanism"
            ]
            is False
        )
        assert (
            boundary_row[
                "forecast_runtime_call_order_blinding_verified_by_mechanism"
            ]
            is False
        )
    persisted_credit_timestamps = [
        row["credit_timestamp_ms"]
        for row in persisted_rows
        if row["record_type"] == "online_postaction"
    ]
    assert persisted_credit_timestamps == [70_000, 70_000, 70_001, 70_001]

    before = path.read_bytes()
    unreceipted = subject.scan_relationship_product_horizon_online_physical_barrier(
        path=path,
        mechanism_run_id="online-physical-test-run",
        root_sequence_index=7,
        bindings=online_bindings,
        expected_source_capability_id=_SOURCE_CAPABILITY_ID,
        expected_slot_count=2,
    )
    assert unreceipted.status is (
        subject.RelationshipProductHorizonOnlineLedgerStatus
        .TERMINAL_CONTENT_VALID_DURABILITY_UNPROVEN
    )
    scan = subject.validate_relationship_product_horizon_online_physical_barrier(
        path=path,
        mechanism_run_id="online-physical-test-run",
        root_sequence_index=7,
        bindings=online_bindings,
        expected_source_capability_id=_SOURCE_CAPABILITY_ID,
        terminal_completion=second,
        expected_slot_count=2,
    )
    assert scan.status is (
        subject.RelationshipProductHorizonOnlineLedgerStatus
        .TERMINAL_CONTENT_VALID_DURABILITY_UNPROVEN
    )
    assert scan.row_count == 14
    assert scan.source_open_count == 0
    assert scan.append_count == 0
    assert scan.resume_authorized is False
    assert path.read_bytes() == before


def test_constructor_rejects_distinct_arm_runtimes_even_with_same_runtime_id(
    online_bindings,
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "runtime-drift.jsonl"
    drifted_bindings = (
        online_bindings[0],
        dataclasses.replace(
            online_bindings[1],
            forecast_runtime=_ForecastRuntime(),
        ),
    )
    with pytest.raises(ValueError, match="one exact forecast runtime"):
        subject.RelationshipProductHorizonOnlinePhysicalBarrier(
            path=path,
            mechanism_run_id="online-physical-runtime-drift",
            root_sequence_index=0,
            bindings=drifted_bindings,
            source_descriptor=_Source().descriptor(),
            substrate_snapshot=_placeholder_substrate(),
            expected_slot_count=1,
        )
    assert path.exists() is False


def test_constructor_and_scanner_reject_shared_arm_owner_scope(
    online_bindings,
    tmp_path: pathlib.Path,
) -> None:
    good_path = tmp_path / "exclusive-scope-good.jsonl"
    barrier = subject.RelationshipProductHorizonOnlinePhysicalBarrier(
        path=good_path,
        mechanism_run_id="online-physical-exclusive-scope",
        root_sequence_index=0,
        bindings=online_bindings,
        source_descriptor=_Source().descriptor(),
        substrate_snapshot=_placeholder_substrate(),
        expected_slot_count=1,
    )
    asyncio.run(
        barrier.execute_slot(
            requests=_requests(online_bindings, 0),
            temporal_delivery_timestamp_ms=1_001_000,
        )
    )
    barrier.close()

    shared_scope_bindings = (
        online_bindings[0],
        dataclasses.replace(
            online_bindings[1],
            authorization=dataclasses.replace(
                online_bindings[1].authorization,
                owner_session_scope=(
                    online_bindings[0].authorization.owner_session_scope
                ),
            ),
        ),
    )
    rejected_path = tmp_path / "shared-scope-rejected.jsonl"
    with pytest.raises(ValueError, match="exclusive owner scopes"):
        subject.RelationshipProductHorizonOnlinePhysicalBarrier(
            path=rejected_path,
            mechanism_run_id="online-physical-shared-scope-rejected",
            root_sequence_index=0,
            bindings=shared_scope_bindings,
            source_descriptor=_Source().descriptor(),
            substrate_snapshot=_placeholder_substrate(),
            expected_slot_count=1,
        )
    assert rejected_path.exists() is False

    scan = subject.scan_relationship_product_horizon_online_physical_barrier(
        path=good_path,
        mechanism_run_id="online-physical-exclusive-scope",
        root_sequence_index=0,
        bindings=shared_scope_bindings,
        expected_source_capability_id=_SOURCE_CAPABILITY_ID,
        expected_slot_count=1,
    )
    assert scan.status is (
        subject.RelationshipProductHorizonOnlineLedgerStatus.INVALID_INTERRUPTED_TAIL
    )
    assert scan.failure_message == "online scan mechanical arm binding drifted"


def test_source_request_action_order_is_public_not_arm_order() -> None:
    capability = subject.RelationshipProductHorizonOnlineSourceOpenCapability(
        source_capability_id=_SOURCE_CAPABILITY_ID,
        mechanism_run_id="online-physical-action-order",
        root_sequence_index=0,
        slot_index=0,
    )
    with pytest.raises(ValueError, match="public candidate-action order"):
        subject.RelationshipProductHorizonOnlineSourceRequest(
            open_capability=capability,
            decision_id="online-action-order-decision",
            interlocutor_id="primary",
            current_observation="Public action ordering observation.",
            observation_ref="online-action-order-observation",
            candidate_action_ids=_ACTION_IDS,
            outcome_ids=_OUTCOME_IDS,
            turn_index=1,
            outcome_turn_index=2,
            selected_actions=(
                RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
                RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
            ),
        )


@pytest.mark.parametrize("fail_at_sync", (2, 3, 4, 5, 6))
def test_any_group_or_terminal_sync_failure_permanently_poisons_owner_and_sink(
    online_bindings,
    fail_at_sync: int,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fsync = subject.os.fsync
    sync_count = 0

    def fsync(fd: int) -> None:
        nonlocal sync_count
        sync_count += 1
        if sync_count == fail_at_sync:
            raise OSError(f"injected sync failure {fail_at_sync}")
        original_fsync(fd)

    monkeypatch.setattr(subject.os, "fsync", fsync)
    source = _Source()
    path = tmp_path / f"fsync-failure-{fail_at_sync}.jsonl"
    barrier = subject.RelationshipProductHorizonOnlinePhysicalBarrier(
        path=path,
        mechanism_run_id=f"online-physical-fsync-failure:{fail_at_sync}",
        root_sequence_index=0,
        bindings=online_bindings,
        source_descriptor=source.descriptor(),
        substrate_snapshot=_placeholder_substrate(),
        expected_slot_count=1,
    )
    with pytest.raises(RuntimeError, match="failed closed at slot 0"):
        asyncio.run(
            barrier.execute_slot(
                requests=_requests(online_bindings, 0),
                temporal_delivery_timestamp_ms=1_001_000,
            )
        )
    assert barrier.failed is True
    source_count = source.call_count
    with pytest.raises(RuntimeError, match="permanently failed closed"):
        asyncio.run(
            barrier.execute_slot(
                requests=_requests(online_bindings, 0),
                temporal_delivery_timestamp_ms=1_001_000,
            )
        )
    assert source.call_count == source_count
    if fail_at_sync <= 3:
        assert source.call_count == 0
        assert source.open_count == 0
    else:
        assert source.call_count == 1
    barrier.close()
    scan = subject.scan_relationship_product_horizon_online_physical_barrier(
        path=path,
        mechanism_run_id=f"online-physical-fsync-failure:{fail_at_sync}",
        root_sequence_index=0,
        bindings=online_bindings,
        expected_source_capability_id=_SOURCE_CAPABILITY_ID,
        expected_slot_count=1,
    )
    expected_status = (
        subject.RelationshipProductHorizonOnlineLedgerStatus
        .TERMINAL_CONTENT_VALID_DURABILITY_UNPROVEN
        if fail_at_sync == 6
        else subject.RelationshipProductHorizonOnlineLedgerStatus.INVALID_INTERRUPTED_TAIL
    )
    assert scan.status is expected_status


def test_header_fsync_failure_never_opens_source(
    online_bindings,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _Source()

    def fsync(_fd: int) -> None:
        raise OSError("injected header fsync failure")

    monkeypatch.setattr(subject.os, "fsync", fsync)
    with pytest.raises(OSError, match="header fsync"):
        subject.RelationshipProductHorizonOnlinePhysicalBarrier(
            path=tmp_path / "header-fsync-failure.jsonl",
            mechanism_run_id="online-physical-header-fsync-failure",
            root_sequence_index=0,
            bindings=online_bindings,
            source_descriptor=source.descriptor(),
            substrate_snapshot=_placeholder_substrate(),
            expected_slot_count=1,
        )
    assert source.open_count == 0
    assert source.call_count == 0


def test_terminal_is_not_complete_and_close_is_rejected_until_fsync_returns(
    online_bindings,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fsync = subject.os.fsync
    terminal_sync_entered = threading.Event()
    release_terminal_sync = threading.Event()
    sync_count = 0

    def fsync(fd: int) -> None:
        nonlocal sync_count
        sync_count += 1
        if sync_count == 6:
            terminal_sync_entered.set()
            assert release_terminal_sync.wait(timeout=10)
        original_fsync(fd)

    monkeypatch.setattr(subject.os, "fsync", fsync)
    source = _Source()
    barrier = subject.RelationshipProductHorizonOnlinePhysicalBarrier(
        path=tmp_path / "terminal-blocked.jsonl",
        mechanism_run_id="online-physical-terminal-blocked",
        root_sequence_index=0,
        bindings=online_bindings,
        source_descriptor=source.descriptor(),
        substrate_snapshot=_placeholder_substrate(),
        expected_slot_count=1,
    )
    completions = []
    errors = []

    def execute() -> None:
        try:
            completions.append(
                asyncio.run(
                    barrier.execute_slot(
                        requests=_requests(online_bindings, 0),
                        temporal_delivery_timestamp_ms=1_001_000,
                    )
                )
            )
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=execute)
    worker.start()
    assert terminal_sync_entered.wait(timeout=10)
    assert barrier.completed_slot_count == 0
    assert barrier.ledger_complete is False
    assert barrier.failed is False
    with pytest.raises(RuntimeError, match="concurrent close"):
        barrier.close()
    release_terminal_sync.set()
    worker.join(timeout=10)
    assert worker.is_alive() is False
    assert errors == []
    assert len(completions) == 1
    assert completions[0].ledger_complete is True
    assert barrier.completed_slot_count == 1
    assert barrier.ledger_complete is True


def test_create_only_path_rejects_existing_bytes_without_opening_source(
    online_bindings,
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "existing.jsonl"
    path.write_bytes(b"user-owned-existing-bytes")
    source = _Source()
    with pytest.raises(FileExistsError):
        subject.RelationshipProductHorizonOnlinePhysicalBarrier(
            path=path,
            mechanism_run_id="online-physical-existing",
            root_sequence_index=0,
            bindings=online_bindings,
            source_descriptor=source.descriptor(),
            substrate_snapshot=_placeholder_substrate(),
            expected_slot_count=1,
        )
    assert path.read_bytes() == b"user-owned-existing-bytes"
    assert source.open_count == 0


def test_mismatched_public_requests_reject_before_source_and_can_retry(
    online_bindings,
    tmp_path: pathlib.Path,
) -> None:
    source = _Source()
    barrier = subject.RelationshipProductHorizonOnlinePhysicalBarrier(
        path=tmp_path / "matched-requests.jsonl",
        mechanism_run_id="online-physical-matched-requests",
        root_sequence_index=0,
        bindings=online_bindings,
        source_descriptor=source.descriptor(),
        substrate_snapshot=_placeholder_substrate(),
        expected_slot_count=1,
    )
    requests = list(_requests(online_bindings, 0))
    frozen_arm, frozen_request = requests[1]
    requests[1] = (
        frozen_arm,
        dataclasses.replace(
            frozen_request,
            forecast_request=dataclasses.replace(
                frozen_request.forecast_request,
                decision_id="arm-dependent-decision",
                current_observation="Arm-dependent observation.",
                observation_ref="arm-dependent-ref",
            ),
        ),
    )
    with pytest.raises(ValueError, match="exact public request projection"):
        asyncio.run(
            barrier.execute_slot(
                requests=tuple(requests),
                temporal_delivery_timestamp_ms=1_001_000,
            )
        )
    assert barrier.failed is False
    assert source.open_count == 0
    completion = asyncio.run(
        barrier.execute_slot(
            requests=_requests(online_bindings, 0),
            temporal_delivery_timestamp_ms=1_001_000,
        )
    )
    assert completion.ledger_complete is True


def test_duplicate_same_action_branch_fails_closed(
    online_bindings,
    tmp_path: pathlib.Path,
) -> None:
    source = _Source(duplicate_branch=True)
    barrier = subject.RelationshipProductHorizonOnlinePhysicalBarrier(
        path=tmp_path / "duplicate-branch.jsonl",
        mechanism_run_id="online-physical-duplicate-branch",
        root_sequence_index=0,
        bindings=online_bindings,
        source_descriptor=source.descriptor(),
        substrate_snapshot=_placeholder_substrate(),
        expected_slot_count=1,
    )
    with pytest.raises(RuntimeError, match="failed closed at slot 0"):
        asyncio.run(
            barrier.execute_slot(
                requests=_requests(online_bindings, 0),
                temporal_delivery_timestamp_ms=1_001_000,
            )
        )
    assert barrier.failed is True
    assert source.call_count == 1


def test_source_failure_permanently_poisons_before_any_commit(
    online_bindings,
    tmp_path: pathlib.Path,
) -> None:
    source = _Source(fail_settle=True)
    barrier = subject.RelationshipProductHorizonOnlinePhysicalBarrier(
        path=tmp_path / "source-failure.jsonl",
        mechanism_run_id="online-physical-source-failure",
        root_sequence_index=0,
        bindings=online_bindings,
        source_descriptor=source.descriptor(),
        substrate_snapshot=_placeholder_substrate(),
        expected_slot_count=1,
    )
    with pytest.raises(RuntimeError, match="failed closed at slot 0"):
        asyncio.run(
            barrier.execute_slot(
                requests=_requests(online_bindings, 0),
                temporal_delivery_timestamp_ms=1_001_000,
            )
        )
    assert barrier.failed is True
    assert source.call_count == 1


def test_second_arm_prepare_failure_never_writes_preactions_or_opens_source(
    online_bindings,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_prepare = subject.prepare_relationship_product_v2_online_preaction
    prepare_count = 0

    async def prepare(**kwargs):
        nonlocal prepare_count
        prepare_count += 1
        if prepare_count == 2:
            raise RuntimeError("injected frozen prepare failure")
        return await original_prepare(**kwargs)

    monkeypatch.setattr(
        subject,
        "prepare_relationship_product_v2_online_preaction",
        prepare,
    )
    path = tmp_path / "second-prepare-failure.jsonl"
    source = _Source()
    barrier = subject.RelationshipProductHorizonOnlinePhysicalBarrier(
        path=path,
        mechanism_run_id="online-physical-second-prepare-failure",
        root_sequence_index=0,
        bindings=online_bindings,
        source_descriptor=source.descriptor(),
        substrate_snapshot=_placeholder_substrate(),
        expected_slot_count=1,
    )
    with pytest.raises(RuntimeError, match="failed closed at slot 0"):
        asyncio.run(
            barrier.execute_slot(
                requests=_requests(online_bindings, 0),
                temporal_delivery_timestamp_ms=1_001_000,
            )
        )
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    assert [item["record_type"] for item in rows] == ["online_physical_header"]
    assert source.open_count == 0
    assert source.call_count == 0
    assert barrier.failed is True


def test_postaction_member_write_failure_permanently_poisons_after_commit(
    online_bindings,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_write = subject._CreateOnlyStreamingJsonlSink._write
    write_count = 0

    def write(sink, raw: bytes) -> None:
        nonlocal write_count
        write_count += 1
        if write_count == 6:
            raise OSError("injected second postaction write failure")
        original_write(sink, raw)

    monkeypatch.setattr(subject._CreateOnlyStreamingJsonlSink, "_write", write)
    source = _Source()
    barrier = subject.RelationshipProductHorizonOnlinePhysicalBarrier(
        path=tmp_path / "post-write-failure.jsonl",
        mechanism_run_id="online-physical-post-write-failure",
        root_sequence_index=0,
        bindings=online_bindings,
        source_descriptor=source.descriptor(),
        substrate_snapshot=_placeholder_substrate(),
        expected_slot_count=1,
    )
    with pytest.raises(RuntimeError, match="failed closed at slot 0"):
        asyncio.run(
            barrier.execute_slot(
                requests=_requests(online_bindings, 0),
                temporal_delivery_timestamp_ms=1_001_000,
            )
        )
    assert source.call_count == 1
    assert barrier.failed is True


def test_one_arm_commit_failure_poisons_pair_before_any_postaction_write(
    online_bindings,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_commit = RelationshipActionGateV2OnlineSession.commit_credit

    def commit(session, plan):
        if session.disposition is RelationshipActionGateBatchDisposition.WITHHOLD:
            raise RuntimeError("injected frozen commit failure")
        return original_commit(session, plan)

    monkeypatch.setattr(RelationshipActionGateV2OnlineSession, "commit_credit", commit)
    path = tmp_path / "one-arm-commit-failure.jsonl"
    source = _Source()
    barrier = subject.RelationshipProductHorizonOnlinePhysicalBarrier(
        path=path,
        mechanism_run_id="online-physical-one-arm-commit-failure",
        root_sequence_index=0,
        bindings=online_bindings,
        source_descriptor=source.descriptor(),
        substrate_snapshot=_placeholder_substrate(),
        expected_slot_count=1,
    )
    with pytest.raises(RuntimeError, match="failed closed at slot 0"):
        asyncio.run(
            barrier.execute_slot(
                requests=_requests(online_bindings, 0),
                temporal_delivery_timestamp_ms=1_001_000,
            )
        )
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    assert [item["record_type"] for item in rows] == [
        "online_physical_header",
        "online_preaction",
        "online_preaction",
        "online_preaction_group_fsync",
    ]
    assert source.call_count == 1
    assert barrier.completed_slot_count == 0
    assert barrier.ledger_complete is False
    assert barrier.failed is True


def test_reentrant_next_slot_is_rejected_before_a_second_source_call(
    online_bindings,
    tmp_path: pathlib.Path,
) -> None:
    class ReentrantSource(_Source):
        owner = None

        async def settle_actions(self, *, request):
            assert self.owner is not None
            with pytest.raises(RuntimeError, match="concurrent or reentrant"):
                await self.owner.execute_slot(
                    requests=_requests(online_bindings, 0),
                    temporal_delivery_timestamp_ms=1_001_000,
                )
            return await super().settle_actions(request=request)

    source = ReentrantSource()
    barrier = subject.RelationshipProductHorizonOnlinePhysicalBarrier(
        path=tmp_path / "reentrant.jsonl",
        mechanism_run_id="online-physical-reentrant",
        root_sequence_index=0,
        bindings=online_bindings,
        source_descriptor=source.descriptor(),
        substrate_snapshot=_placeholder_substrate(),
        expected_slot_count=1,
    )
    source.owner = barrier
    completion = asyncio.run(
        barrier.execute_slot(
            requests=_requests(online_bindings, 0),
            temporal_delivery_timestamp_ms=1_001_000,
        )
    )
    assert completion.ledger_complete is True
    assert source.call_count == 1


def _canonical_row(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def _resign_terminal(rows: list[dict[str, object]]) -> bytes:
    terminal = rows[-1]
    terminal_core = {
        key: value
        for key, value in terminal.items()
        if key
        not in {
            "row_id",
            "physical_sequence_index",
            "schema_version",
            "record_type",
            "terminal_id",
        }
    }
    terminal["terminal_id"] = subject.sha256_json(terminal_core)
    row_core = {key: value for key, value in terminal.items() if key != "row_id"}
    terminal["row_id"] = subject.sha256_json(row_core)
    return b"".join(_canonical_row(row) for row in rows)


def _resign_one_slot_post_group(rows: list[dict[str, object]]) -> bytes:
    assert len(rows) == 8
    prefix = b"".join(_canonical_row(row) for row in rows[:4])
    member_raw: list[bytes] = []
    for physical_index in (4, 5):
        row = rows[physical_index]
        row["physical_sequence_index"] = physical_index
        row_core = {key: value for key, value in row.items() if key != "row_id"}
        row["row_id"] = subject.sha256_json(row_core)
        raw = _canonical_row(row)
        member_raw.append(raw)
        prefix += raw

    receipt = rows[6]
    receipt["physical_sequence_index"] = 6
    receipt["postaction_row_ids"] = [rows[4]["row_id"], rows[5]["row_id"]]
    receipt["postaction_rows_start_index"] = 4
    receipt["postaction_rows_end_index"] = 5
    receipt["postaction_rows_raw_sha256"] = hashlib.sha256(
        b"".join(member_raw)
    ).hexdigest()
    receipt["durable_prefix_byte_count_before_receipt"] = len(prefix)
    receipt["durable_prefix_raw_sha256_before_receipt"] = hashlib.sha256(
        prefix
    ).hexdigest()
    receipt_core = {
        key: value
        for key, value in receipt.items()
        if key
        not in {
            "row_id",
            "physical_sequence_index",
            "schema_version",
            "record_type",
            "postaction_receipt_id",
            "next_slot_authorized",
        }
    }
    receipt["postaction_receipt_id"] = subject.sha256_json(receipt_core)
    receipt_row_core = {
        key: value for key, value in receipt.items() if key != "row_id"
    }
    receipt["row_id"] = subject.sha256_json(receipt_row_core)
    prefix += _canonical_row(receipt)

    terminal = rows[7]
    terminal["physical_sequence_index"] = 7
    terminal["postaction_receipt_row_ids"] = [receipt["row_id"]]
    terminal_core = {
        key: value
        for key, value in terminal.items()
        if key
        not in {
            "row_id",
            "physical_sequence_index",
            "schema_version",
            "record_type",
            "terminal_id",
        }
    }
    terminal["terminal_id"] = subject.sha256_json(terminal_core)
    terminal_row_core = {
        key: value for key, value in terminal.items() if key != "row_id"
    }
    terminal["row_id"] = subject.sha256_json(terminal_row_core)
    return prefix + _canonical_row(terminal)


def test_scanner_rejects_partial_torn_and_resigned_semantic_mutants_without_writes(
    online_bindings,
    tmp_path: pathlib.Path,
) -> None:
    good = tmp_path / "good.jsonl"
    source = _Source()
    barrier = subject.RelationshipProductHorizonOnlinePhysicalBarrier(
        path=good,
        mechanism_run_id="online-physical-scan-mutants",
        root_sequence_index=3,
        bindings=online_bindings,
        source_descriptor=source.descriptor(),
        substrate_snapshot=_placeholder_substrate(),
        expected_slot_count=1,
    )
    terminal_completion = asyncio.run(
        barrier.execute_slot(
            requests=_requests(online_bindings, 0),
            temporal_delivery_timestamp_ms=1_001_000,
        )
    )
    barrier.close()
    good_raw = good.read_bytes()
    rows = [json.loads(line) for line in good_raw.splitlines()]

    mutants = {
        "complete_post_prefix_without_terminal": b"\n".join(
            good_raw.splitlines()[:-1]
        )
        + b"\n",
        "torn_terminal": good_raw[:-1],
    }
    semantic_rows = json.loads(json.dumps(rows))
    semantic_rows[-1]["completed_slot_count"] = 2
    mutants["resigned_wrong_terminal_count"] = _resign_terminal(semantic_rows)

    pe_rows = json.loads(json.dumps(rows))
    description = pe_rows[4]["social_prediction_error"]["description"]
    pe_rows[4]["social_prediction_error"]["description"] = "X" * len(
        description
    )
    mutants["resigned_wrong_pe_description"] = _resign_one_slot_post_group(
        pe_rows
    )

    owner_rows = json.loads(json.dumps(rows))
    initial_owner = owner_rows[0]["arm_initializations"][0]
    owner_rows[4]["owner_postaction_persistence"] = initial_owner[
        "initial_owner_persistence"
    ]
    owner_rows[4]["owner_postaction_persistence_sha256"] = initial_owner[
        "initial_owner_persistence_sha256"
    ]
    owner_rows[7]["arm_terminals"][0]["terminal_owner_persistence"] = (
        initial_owner["initial_owner_persistence"]
    )
    owner_rows[7]["arm_terminals"][0][
        "terminal_owner_persistence_sha256"
    ] = initial_owner["initial_owner_persistence_sha256"]
    mutants["resigned_owner_writeback_erased"] = _resign_one_slot_post_group(
        owner_rows
    )

    for name, raw in mutants.items():
        path = tmp_path / f"{name}.jsonl"
        path.write_bytes(raw)
        before = path.read_bytes()
        scan = subject.scan_relationship_product_horizon_online_physical_barrier(
            path=path,
            mechanism_run_id="online-physical-scan-mutants",
            root_sequence_index=3,
            bindings=online_bindings,
            expected_source_capability_id=_SOURCE_CAPABILITY_ID,
            terminal_completion=terminal_completion,
            expected_slot_count=1,
        )
        assert scan.status is (
            subject.RelationshipProductHorizonOnlineLedgerStatus.INVALID_INTERRUPTED_TAIL
        )
        assert scan.resume_authorized is False
        assert scan.source_open_count == 0
        assert scan.append_count == 0
        assert path.read_bytes() == before

    missing = subject.scan_relationship_product_horizon_online_physical_barrier(
        path=tmp_path / "absent.jsonl",
        mechanism_run_id="online-physical-scan-mutants",
        root_sequence_index=3,
        bindings=online_bindings,
        expected_source_capability_id=_SOURCE_CAPABILITY_ID,
        expected_slot_count=1,
    )
    assert missing.status is subject.RelationshipProductHorizonOnlineLedgerStatus.FRESH
