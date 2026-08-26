from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from dataclasses import fields, replace

import pytest

from lifeform_domain_emogpt.lab.relationship_product_pulse import (
    RelationshipProductExecutorCommand,
    RelationshipProductExecutorDisposition,
    RelationshipProductExecutorReceipt,
    RelationshipProductExecutorStatus,
    RelationshipProductFrozenPulseAuthorization,
    RelationshipProductOnboardingInput,
    RelationshipProductPreActionRequest,
    RelationshipProductPulseAuthorization,
    RelationshipProductSettlementInput,
    append_relationship_product_onboarding,
    prepare_relationship_product_frozen_preaction,
    prepare_relationship_product_preaction,
    settle_relationship_product_frozen_pulse,
    settle_relationship_product_pulse,
)
from lifeform_domain_emogpt.relationship_action_contracts import (
    RELATIONSHIP_ACTIONS,
    RELATIONSHIP_OUTCOMES,
    RelationshipAction,
)
from lifeform_domain_emogpt.relationship_action_gate import (
    RelationshipActionGate,
    RelationshipActionGateCheckpoint,
    RelationshipActionGateDecision,
    RelationshipActionGateFrozenPolicy,
    RelationshipActionGateMode,
    RelationshipActionGateTheta0Artifact,
)
from lifeform_domain_emogpt.relationship_condition_reader import (
    PrototypeRelationshipPreferenceForecastRuntime,
    RelationshipConditionPrototype,
    RelationshipConditionReaderArtifact,
)
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
)
from volvence_zero.social import (
    PreferenceActionForecastRequest,
    SocialRecordStore,
)
from volvence_zero.social_cognition import PreferenceActionOutcomeEvidence
from volvence_zero.substrate import SubstrateSnapshot, SurfaceKind


_ACTION_IDS = tuple(action.value for action in RELATIONSHIP_ACTIONS)
_OUTCOME_IDS = tuple(outcome.value for outcome in RELATIONSHIP_OUTCOMES)
_SUBJECT_SCOPE = hashlib.sha256(b"relationship-product-pulse-subject").hexdigest()


class _DeterministicFakeEmbedder:
    def embed(self, text: str) -> tuple[float, ...]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return tuple((value + 1) / 256.0 for value in digest[:16])


class _FailIfCalledForecastRuntime:
    runtime_id = "relationship-product-pulse-fail-if-called"

    def __init__(self) -> None:
        self.call_count = 0

    def propose(self, *, request, records, action_outcomes):
        del request, records, action_outcomes
        self.call_count += 1
        raise AssertionError("forecast runtime must not run after invalid checkpoint")


class _DeterministicFakeWorld:
    """Typed transition test double with no preferred-action/evaluator label."""

    def settle(
        self,
        *,
        preaction,
    ) -> RelationshipProductSettlementInput:
        selected = preaction.gate_decision.selected_action_id
        return _typed_world_settlement(preaction=preaction, selected=selected)


class _FrozenDeterministicFakeWorld:
    """New consumer reads actual delivery only from the executor receipt."""

    def settle(
        self,
        *,
        preaction,
    ) -> RelationshipProductSettlementInput:
        return _typed_world_settlement(
            preaction=preaction,
            selected=preaction.execution_receipt.delivered_action_id,
        )


def _typed_world_settlement(
    *,
    preaction,
    selected: str,
) -> RelationshipProductSettlementInput:
    kind = (
        DialogueExternalOutcomeKind.MISSED
        if selected == RelationshipAction.NEUTRAL_NOOP.value
        else DialogueExternalOutcomeKind.FELT_HEARD
    )
    evidence_ref = hashlib.sha256(
        f"{preaction.forecast.decision_id}:{selected}:{kind.value}".encode("utf-8")
    ).hexdigest()
    external = DialogueExternalOutcomeEvidence(
        evidence_id=f"fake-world:{evidence_ref}",
        turn_index=preaction.request.outcome_turn_index,
        kind=kind,
        source=DialogueExternalOutcomeEvidenceSource.ENVIRONMENT,
        confidence=1.0,
        evidence_ref=evidence_ref,
        description="Deterministic typed product-pulse outcome.",
        session_scope=preaction.request.forecast_request.session_scope,
        action_turn_index=preaction.request.forecast_request.turn_index,
        forecast_id=preaction.forecast.forecast_id,
        decision_id=preaction.forecast.decision_id,
        action_id=selected,
    )
    owner_evidence = PreferenceActionOutcomeEvidence(
        evidence_id=f"owner-outcome:{preaction.forecast.decision_id}",
        interlocutor_id=preaction.forecast.interlocutor_id,
        observation_summary=preaction.request.forecast_request.current_observation,
        action_id=selected,
        observed_outcome_id=kind.value,
        reaction_summary=f"Typed fake-world reaction: {kind.value}.",
        source_turn=preaction.request.outcome_turn_index,
        evidence_refs=(evidence_ref,),
    )
    return RelationshipProductSettlementInput(
        external_outcome=external,
        owner_outcome_evidence=owner_evidence,
        credit_timestamp_ms=preaction.request.outcome_turn_index * 1000,
        apply_credit_to_gate=True,
    )


def _reader() -> PrototypeRelationshipPreferenceForecastRuntime:
    artifact = RelationshipConditionReaderArtifact(
        embedding_model_id="deterministic-fake-bge",
        embedding_weights_sha256=hashlib.sha256(b"fake-bge").hexdigest(),
        prototypes=(
            RelationshipConditionPrototype(
                label="closeness_request",
                summary="The person needs calm presence without pressure.",
            ),
            RelationshipConditionPrototype(
                label="space_request",
                summary="The person needs space with a reliable return option.",
            ),
        ),
    )
    return PrototypeRelationshipPreferenceForecastRuntime(
        artifact=artifact,
        embedder=_DeterministicFakeEmbedder(),
    )


def _placeholder_substrate() -> SubstrateSnapshot:
    return SubstrateSnapshot(
        model_id="relationship-product-pulse-test-placeholder",
        is_frozen=True,
        surface_kind=SurfaceKind.PLACEHOLDER,
        token_logits=(),
        feature_surface=(),
        residual_activations=(),
        residual_sequence=(),
        unavailable_fields=(),
        description="Typed product-pulse test; no model or CUDA.",
    )


def _request(*, decision_index: int) -> RelationshipProductPreActionRequest:
    action_turn = 1 + decision_index * 2
    return RelationshipProductPreActionRequest(
        session_id=f"product-session-{decision_index}",
        forecast_request=PreferenceActionForecastRequest(
            decision_id=f"product-decision-{decision_index}",
            interlocutor_id="primary",
            current_observation=(
                f"Public relationship observation {decision_index}; no hidden truth."
            ),
            observation_ref=f"public-observation:{decision_index}",
            candidate_action_ids=_ACTION_IDS,
            outcome_ids=_OUTCOME_IDS,
            turn_index=action_turn,
            session_scope=_SUBJECT_SCOPE,
        ),
        outcome_turn_index=action_turn + 1,
    )


def _authorization() -> RelationshipProductPulseAuthorization:
    checkpoint = RelationshipActionGate().export_checkpoint()
    return RelationshipProductPulseAuthorization(
        authorization_id=hashlib.sha256(b"pulse-authorization").hexdigest(),
        allowed_policy_artifact_id=checkpoint.artifact_id,
        allowed_policy_artifact_version=checkpoint.artifact_version,
    )


def _frozen_policy(*, bias: float = 2.0) -> RelationshipActionGateFrozenPolicy:
    source_checkpoint = RelationshipActionGateCheckpoint(
        artifact_id="product-pulse-test-calibration-gate",
        artifact_version=1,
        weights=(0.0, 0.0, 0.0, 0.0, 0.0),
        bias=bias,
        update_count=1,
        processed_credit_ids=("product-pulse-test-calibration-credit",),
        pending_decisions=(),
    )
    calibration_digest = hashlib.sha256(
        b"product-pulse-test-calibration-batch"
    ).hexdigest()
    theta0 = RelationshipActionGateTheta0Artifact.create(
        source_checkpoint=source_checkpoint,
        learning_rate=0.25,
        max_abs_parameter=4.0,
        source_batch_artifact_id=(
            f"product-pulse-test-calibration-batch-sha256:{calibration_digest}"
        ),
    )
    return RelationshipActionGate.from_theta0(
        theta0,
        random_seed="product-pulse-test-frozen-policy",
    ).freeze_for_evaluation()


def _frozen_authorization(
    policy: RelationshipActionGateFrozenPolicy,
) -> RelationshipProductFrozenPulseAuthorization:
    return RelationshipProductFrozenPulseAuthorization(
        pulse_authorization=RelationshipProductPulseAuthorization(
            authorization_id=hashlib.sha256(
                b"product-pulse-frozen-authorization"
            ).hexdigest(),
            allowed_policy_artifact_id=policy.artifact.artifact_id,
            allowed_policy_artifact_version=policy.artifact.artifact_version,
        ),
        allowed_frozen_policy_id=policy.policy_id,
        allowed_checkpoint_content_sha256=policy.checkpoint.content_sha256,
    )


async def _seed_owner_state():
    snapshot = await append_relationship_product_onboarding(
        owner_persistence_snapshot=None,
        onboarding=RelationshipProductOnboardingInput(
            session_id="seed-onboarding-session",
            session_index=0,
            turn_index=0,
            public_observation="A prior public request for calm presence.",
            action_id=RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
            observed_outcome_id=DialogueExternalOutcomeKind.FELT_HEARD.value,
            reaction_summary="The prior typed outcome was felt_heard.",
            evidence_ref="seed-owner-evidence-ref",
        ),
    )
    return snapshot.owner_persistence_snapshot


def _prepare_frozen_preaction(
    *,
    policy: RelationshipActionGateFrozenPolicy,
    owner_state,
    disposition: RelationshipProductExecutorDisposition,
    decision_index: int = 0,
):
    return asyncio.run(
        prepare_relationship_product_frozen_preaction(
            request=_request(decision_index=decision_index),
            owner_persistence_snapshot=owner_state,
            forecast_runtime=_reader(),
            frozen_policy=policy,
            executor_disposition=disposition,
            authorization=_frozen_authorization(policy),
            substrate_snapshot=_placeholder_substrate(),
        )
    )


def test_product_onboarding_is_fresh_child_safe_and_snapshot_only() -> None:
    assert {field.name for field in fields(RelationshipProductOnboardingInput)} == {
        "session_id",
        "session_index",
        "turn_index",
        "public_observation",
        "action_id",
        "observed_outcome_id",
        "reaction_summary",
        "evidence_ref",
    }
    first_input = RelationshipProductOnboardingInput(
        session_id="onboarding-session-0",
        session_index=0,
        turn_index=0,
        public_observation="First public onboarding observation.",
        action_id=RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
        observed_outcome_id=DialogueExternalOutcomeKind.FELT_HEARD.value,
        reaction_summary="First typed public reaction.",
        evidence_ref="onboarding-evidence-ref-0",
    )
    first = asyncio.run(
        append_relationship_product_onboarding(
            owner_persistence_snapshot=None,
            onboarding=first_input,
        )
    )
    second = asyncio.run(
        append_relationship_product_onboarding(
            owner_persistence_snapshot=first.owner_persistence_snapshot,
            onboarding=RelationshipProductOnboardingInput(
                session_id="onboarding-session-1",
                session_index=1,
                turn_index=1,
                public_observation="Second public onboarding observation.",
                action_id=(
                    RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION.value
                ),
                observed_outcome_id=DialogueExternalOutcomeKind.HELPED.value,
                reaction_summary="Second typed public reaction.",
                evidence_ref="onboarding-evidence-ref-1",
            ),
        )
    )

    assert first.preference_snapshot.value.action_outcome_evidence[0].evidence_id == (
        first_input.evidence_id
    )
    assert len(second.preference_snapshot.value.action_outcome_evidence) == 2
    assert second.owner_persistence_snapshot != first.owner_persistence_snapshot


def test_gate_checkpoint_codec_is_canonical_strict_and_legacy_compatible() -> None:
    owner_state = asyncio.run(_seed_owner_state())
    preaction = asyncio.run(
        prepare_relationship_product_preaction(
            request=_request(decision_index=0),
            owner_persistence_snapshot=owner_state,
            gate_checkpoint=None,
            forecast_runtime=_reader(),
            gate_mode=RelationshipActionGateMode.LEARNED,
            authorization=_authorization(),
            substrate_snapshot=_placeholder_substrate(),
        )
    )
    checkpoint = preaction.gate_checkpoint_after
    payload = checkpoint.to_payload()

    assert set(payload) == {
        "schema_version",
        "artifact_id",
        "artifact_version",
        "weights",
        "bias",
        "update_count",
        "processed_credit_ids",
        "pending_decisions",
    }
    assert payload["pending_decisions"] == [
        preaction.gate_decision.to_payload()
    ]
    assert RelationshipActionGateCheckpoint.from_payload(
        json.loads(json.dumps(payload, sort_keys=True))
    ) == checkpoint
    assert RelationshipActionGateDecision.from_payload(
        preaction.gate_decision.to_payload()
    ) == preaction.gate_decision

    legacy = dict(payload)
    legacy["content_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert RelationshipActionGateCheckpoint.from_payload(legacy) == checkpoint

    extra = {**payload, "unexpected": True}
    with pytest.raises(ValueError, match="fields do not match schema"):
        RelationshipActionGateCheckpoint.from_payload(extra)
    tampered_legacy = {**legacy, "bias": 0.25}
    with pytest.raises(ValueError, match="content hash mismatch"):
        RelationshipActionGateCheckpoint.from_payload(tampered_legacy)
    decision_payload = copy.deepcopy(preaction.gate_decision.to_payload())
    decision_payload["artifact_version"] = True
    with pytest.raises(ValueError, match="must be an integer"):
        RelationshipActionGateDecision.from_payload(decision_payload)


def test_legacy_preaction_rejects_checkpoint_before_forecast_runtime() -> None:
    owner_state = asyncio.run(_seed_owner_state())
    invalid_checkpoint = replace(
        RelationshipActionGate().export_checkpoint(),
        artifact_id="wrong-legacy-gate-artifact",
    )
    runtime = _FailIfCalledForecastRuntime()
    with pytest.raises(ValueError, match="artifact_id mismatch"):
        asyncio.run(
            prepare_relationship_product_preaction(
                request=_request(decision_index=0),
                owner_persistence_snapshot=owner_state,
                gate_checkpoint=invalid_checkpoint,
                forecast_runtime=runtime,
                gate_mode=RelationshipActionGateMode.LEARNED,
                authorization=_authorization(),
                substrate_snapshot=_placeholder_substrate(),
            )
        )
    assert runtime.call_count == 0


def test_product_pulse_freezes_preaction_then_settles_only_from_pe_credit() -> None:
    preaction_field_names = {field.name for field in fields(RelationshipProductPreActionRequest)}
    assert preaction_field_names == {
        "session_id",
        "forecast_request",
        "outcome_turn_index",
    }
    assert not preaction_field_names.intersection(
        {"preferred_action", "expected_action", "evaluator", "judge", "outcome"}
    )

    owner_state = asyncio.run(_seed_owner_state())
    preaction = asyncio.run(
        prepare_relationship_product_preaction(
            request=_request(decision_index=0),
            owner_persistence_snapshot=owner_state,
            gate_checkpoint=None,
            forecast_runtime=_reader(),
            gate_mode=RelationshipActionGateMode.LEARNED,
            authorization=_authorization(),
            substrate_snapshot=_placeholder_substrate(),
        )
    )

    assert preaction.forecast.condition_readout is not None
    assert preaction.gate_decision.selected_action_id == "neutral_noop"
    assert preaction.temporal_snapshot.value.active_abstract_action == "neutral_noop"
    assert len(preaction.gate_checkpoint_after.pending_decisions) == 1
    assert preaction.gate_checkpoint_before.update_count == 0

    world = _DeterministicFakeWorld()
    settlement_input = world.settle(preaction=preaction)
    settlement = asyncio.run(
        settle_relationship_product_pulse(
            preaction=preaction,
            settlement_input=settlement_input,
        )
    )

    assert settlement.settlement.forecast_id == preaction.forecast.forecast_id
    assert settlement.credit.prediction_id == preaction.forecast.forecast_id
    assert settlement.credit.level == "relationship_action_prediction_error"
    assert settlement.credit.source_event.startswith("social_pe:social-pe:")
    assert settlement.gate_update is not None
    assert settlement.gate_checkpoint.update_count == 1
    assert not settlement.gate_checkpoint.pending_decisions
    assert settlement.owner_persistence_snapshot != owner_state

    resumed = asyncio.run(
        prepare_relationship_product_preaction(
            request=_request(decision_index=1),
            owner_persistence_snapshot=settlement.owner_persistence_snapshot,
            gate_checkpoint=RelationshipActionGateCheckpoint.from_payload(
                settlement.gate_checkpoint.to_payload()
            ),
            forecast_runtime=_reader(),
            gate_mode=RelationshipActionGateMode.LEARNED,
            authorization=_authorization(),
            substrate_snapshot=_placeholder_substrate(),
        )
    )
    assert resumed.gate_checkpoint_before.update_count == 1
    assert resumed.gate_decision.steer_probability > 0.5
    assert resumed.gate_decision.selected_action_id != "neutral_noop"


def test_product_pulse_rejects_postaction_lineage_drift_and_non_placeholder() -> None:
    owner_state = asyncio.run(_seed_owner_state())
    arguments = {
        "request": _request(decision_index=0),
        "owner_persistence_snapshot": owner_state,
        "gate_checkpoint": None,
        "forecast_runtime": _reader(),
        "gate_mode": RelationshipActionGateMode.LEARNED,
        "authorization": _authorization(),
    }
    residual_surface = replace(
        _placeholder_substrate(),
        surface_kind=SurfaceKind.RESIDUAL_STREAM,
    )
    with pytest.raises(ValueError, match="typed simulation only"):
        asyncio.run(
            prepare_relationship_product_preaction(
                **arguments,
                substrate_snapshot=residual_surface,
            )
        )

    preaction = asyncio.run(
        prepare_relationship_product_preaction(
            **arguments,
            substrate_snapshot=_placeholder_substrate(),
        )
    )
    settlement_input = _DeterministicFakeWorld().settle(preaction=preaction)
    drifted_external = replace(
        settlement_input.external_outcome,
        action_id=RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
    )
    with pytest.raises(ValueError, match="external action lineage mismatch"):
        asyncio.run(
            settle_relationship_product_pulse(
                preaction=preaction,
                settlement_input=replace(
                    settlement_input,
                    external_outcome=drifted_external,
                ),
            )
        )


def test_frozen_preaction_apply_and_strict_noop_share_candidate_core() -> None:
    owner_state = asyncio.run(_seed_owner_state())
    policy = _frozen_policy()
    checkpoint_before = policy.checkpoint
    applied = _prepare_frozen_preaction(
        policy=policy,
        owner_state=owner_state,
        disposition=RelationshipProductExecutorDisposition.APPLY_CANDIDATE,
    )
    strict = _prepare_frozen_preaction(
        policy=policy,
        owner_state=owner_state,
        disposition=RelationshipProductExecutorDisposition.FORCE_STRICT_NOOP,
    )

    assert applied.forecast == strict.forecast
    assert (
        applied.execution_receipt.command.owner_prestate_sha256
        == strict.execution_receipt.command.owner_prestate_sha256
    )
    assert applied.frozen_decision == strict.frozen_decision
    assert applied.execution_receipt.candidate_advisory == (
        strict.execution_receipt.candidate_advisory
    )
    applied_core = applied.execution_receipt.command.to_payload(
        include_command_id=False
    )
    strict_core = strict.execution_receipt.command.to_payload(
        include_command_id=False
    )
    assert set(applied_core) == set(strict_core)
    assert {
        key: value
        for key, value in applied_core.items()
        if key != "executor_disposition"
    } == {
        key: value
        for key, value in strict_core.items()
        if key != "executor_disposition"
    }

    applied_receipt = applied.execution_receipt
    strict_receipt = strict.execution_receipt
    assert applied_receipt.command.non_noop_opportunity
    assert applied_receipt.executor_apply_bit
    assert applied_receipt.candidate_applied
    assert not applied_receipt.strict_noop_substituted
    assert (
        applied_receipt.executor_status
        is RelationshipProductExecutorStatus.APPLIED_CANDIDATE
    )
    assert applied_receipt.delivered_action_id == (
        applied_receipt.command.candidate_action_id
    )
    assert not applied_receipt.action_diverged
    assert not strict_receipt.executor_apply_bit
    assert not strict_receipt.candidate_applied
    assert strict_receipt.strict_noop_substituted
    assert (
        strict_receipt.executor_status
        is RelationshipProductExecutorStatus.STRICT_NOOP
    )
    assert strict_receipt.delivered_action_id == "neutral_noop"
    assert strict_receipt.action_diverged
    assert (
        strict_receipt.temporal_delivery.active_abstract_action
        == "neutral_noop"
    )
    assert policy.checkpoint == checkpoint_before
    assert not policy.checkpoint.pending_decisions
    assert applied_receipt.to_payload()["evaluation_gate_update_delta"] == 0
    assert strict_receipt.to_payload()["evaluation_gate_update_delta"] == 0
    assert applied_receipt.to_payload()["owner_prestate_sha256"] == (
        applied_receipt.command.owner_prestate_sha256
    )
    assert applied_receipt.command.owner_prestate_sha256 == (
        "bc351b76fe5fea473d50088186a49679fcdba559d3a31f54a36d0d74d59295f4"
    )
    assert applied_receipt.command.command_id == (
        "relationship-product-executor-command-sha256:"
        "d52e5d3d69bc8b7d90f4d382b6ed695996207687f6bdb1215162aedaee997f3d"
    )
    assert strict_receipt.command.command_id == (
        "relationship-product-executor-command-sha256:"
        "ff19892547a80d41e5b75d204132d48ebcd9fa26b4fd1a6382eefffa61005960"
    )
    fixed_applied_receipt = RelationshipProductExecutorReceipt(
        command=applied_receipt.command,
        candidate_advisory=applied_receipt.candidate_advisory,
        delivered_advisory=applied_receipt.delivered_advisory,
        temporal_delivery=replace(
            applied_receipt.temporal_delivery,
            timestamp_ms=1_700_000_000_000,
        ),
    )
    fixed_strict_receipt = RelationshipProductExecutorReceipt(
        command=strict_receipt.command,
        candidate_advisory=strict_receipt.candidate_advisory,
        delivered_advisory=strict_receipt.delivered_advisory,
        temporal_delivery=replace(
            strict_receipt.temporal_delivery,
            timestamp_ms=1_700_000_000_000,
        ),
    )
    assert fixed_applied_receipt.receipt_id == (
        "relationship-product-executor-receipt-sha256:"
        "b09c7d994da7eb928031e73a0995e41964d1fe963852f1313438de9955bf9b8f"
    )
    assert fixed_strict_receipt.receipt_id == (
        "relationship-product-executor-receipt-sha256:"
        "431b96cdb9800f37edd12cfb2fbbd2ad3a971b5bbd985b125d4b94d1ebf8691f"
    )


def test_frozen_settlement_joins_delivered_action_not_gate_candidate() -> None:
    owner_state = asyncio.run(_seed_owner_state())
    policy = _frozen_policy()
    strict = _prepare_frozen_preaction(
        policy=policy,
        owner_state=owner_state,
        disposition=RelationshipProductExecutorDisposition.FORCE_STRICT_NOOP,
    )
    assert strict.execution_receipt.command.non_noop_opportunity
    actual_input = _FrozenDeterministicFakeWorld().settle(preaction=strict)
    with pytest.raises(ValueError, match="cannot apply credit"):
        asyncio.run(
            settle_relationship_product_frozen_pulse(
                preaction=strict,
                settlement_input=actual_input,
            )
        )
    actual_input = replace(actual_input, apply_credit_to_gate=False)
    settled = asyncio.run(
        settle_relationship_product_frozen_pulse(
            preaction=strict,
            settlement_input=actual_input,
        )
    )
    assert settled.settlement.action_id == strict.delivered_action_id
    assert settled.credit.abstract_action_id == strict.delivered_action_id
    assert not settled.credit_applied_to_gate
    assert settled.evaluation_gate_update_delta == 0
    assert settled.gate_checkpoint == policy.checkpoint
    with pytest.raises(ValueError, match="PE-derived credit payload drifted"):
        replace(
            settled,
            credit=replace(
                settled.credit,
                abstract_action_id=strict.execution_receipt.command.candidate_action_id,
            ),
        )
    with pytest.raises(ValueError, match="PE-derived credit payload drifted"):
        replace(
            settled,
            credit=replace(
                settled.credit,
                record_id="forged-credit",
            ),
        )
    external_entry = settled.external_outcome_snapshot.value.entries[0]
    with pytest.raises(ValueError, match="external outcome action_id mismatch"):
        replace(
            settled,
            external_outcome_snapshot=replace(
                settled.external_outcome_snapshot,
                value=replace(
                    settled.external_outcome_snapshot.value,
                    entries=(
                        replace(
                            external_entry,
                            action_id=strict.execution_receipt.command.candidate_action_id,
                        ),
                    ),
                ),
            ),
        )
    with pytest.raises(ValueError, match="preference owner settlement payload drifted"):
        replace(
            settled,
            preference_snapshot=replace(
                settled.preference_snapshot,
                value=replace(
                    settled.preference_snapshot.value,
                    forecast_settlements=(),
                ),
            ),
        )
    social_error = next(
        item
        for item in settled.social_prediction_error_snapshot.value.errors
        if item.error_id == f"social-pe:{settled.settlement.settlement_id}"
    )
    with pytest.raises(ValueError, match="social PE owner mismatch"):
        replace(
            settled,
            social_prediction_error_snapshot=replace(
                settled.social_prediction_error_snapshot,
                value=replace(
                    settled.social_prediction_error_snapshot.value,
                    errors=tuple(
                        replace(item, owner="ForgedPEOwner")
                        if item == social_error
                        else item
                        for item in settled.social_prediction_error_snapshot.value.errors
                    ),
                ),
            ),
        )

    candidate = strict.execution_receipt.command.candidate_action_id
    candidate_input = _typed_world_settlement(
        preaction=strict,
        selected=candidate,
    )
    with pytest.raises(ValueError, match="external action lineage mismatch"):
        asyncio.run(
            settle_relationship_product_frozen_pulse(
                preaction=strict,
                settlement_input=replace(
                    candidate_input,
                    apply_credit_to_gate=False,
                ),
            )
        )


def test_frozen_evidence_and_persistence_are_exactly_bound() -> None:
    owner_state = asyncio.run(_seed_owner_state())
    policy = _frozen_policy()
    strict = _prepare_frozen_preaction(
        policy=policy,
        owner_state=owner_state,
        disposition=RelationshipProductExecutorDisposition.FORCE_STRICT_NOOP,
    )
    with pytest.raises(
        ValueError,
        match="frozen preaction owner persistence action_forecasts drifted",
    ):
        replace(strict, owner_persistence_snapshot=owner_state)
    resealed_store = SocialRecordStore()
    resealed_store.hydrate_from_persistence(strict.owner_persistence_snapshot)
    resealed_store.set_preference_action_outcomes(())
    with pytest.raises(
        ValueError,
        match="frozen preaction owner state hash drifted from executor command",
    ):
        replace(
            strict,
            preference_snapshot=replace(
                strict.preference_snapshot,
                value=replace(
                    strict.preference_snapshot.value,
                    action_outcome_evidence=(),
                ),
            ),
            owner_persistence_snapshot=resealed_store.export_persistence_snapshot(),
        )
    for envelope_drift in (
        {"slot_name": "forged_preference"},
        {"owner": "ForgedPreferenceOwner"},
        {"version": 2},
        {"timestamp_ms": -1},
    ):
        with pytest.raises(ValueError, match="snapshot envelope drifted"):
            replace(
                strict,
                preference_snapshot=replace(
                    strict.preference_snapshot,
                    **envelope_drift,
                ),
            )

    settlement_input = replace(
        _FrozenDeterministicFakeWorld().settle(preaction=strict),
        apply_credit_to_gate=False,
    )
    mutable_payload = json.loads(
        json.dumps(strict.owner_persistence_snapshot.payload)
    )
    mutable_preaction = replace(
        strict,
        owner_persistence_snapshot=replace(
            strict.owner_persistence_snapshot,
            payload=mutable_payload,
        ),
    )
    mutable_payload["preference_action_forecasts"] = []
    with pytest.raises(
        ValueError,
        match="frozen preaction owner persistence action_forecasts drifted",
    ):
        asyncio.run(
            settle_relationship_product_frozen_pulse(
                preaction=mutable_preaction,
                settlement_input=settlement_input,
            )
        )
    with pytest.raises(ValueError, match="requires ENVIRONMENT external evidence"):
        asyncio.run(
            settle_relationship_product_frozen_pulse(
                preaction=strict,
                settlement_input=replace(
                    settlement_input,
                    external_outcome=replace(
                        settlement_input.external_outcome,
                        source=DialogueExternalOutcomeEvidenceSource.HUMAN_REVIEW,
                    ),
                ),
            )
        )

    settled = asyncio.run(
        settle_relationship_product_frozen_pulse(
            preaction=strict,
            settlement_input=settlement_input,
        )
    )
    assert settled.settlement_input == settlement_input

    persisted = SocialRecordStore()
    persisted.hydrate_from_persistence(settled.owner_persistence_snapshot)
    assert all(
        item.forecast_id != strict.forecast.forecast_id
        for item in persisted.preference_action_forecasts
    )
    assert settled.settlement in persisted.preference_forecast_settlements
    assert (
        settlement_input.owner_outcome_evidence
        in persisted.preference_action_outcomes
    )

    with pytest.raises(
        ValueError,
        match="frozen external outcome does not match exact settlement input",
    ):
        replace(
            settled,
            external_outcome_snapshot=replace(
                settled.external_outcome_snapshot,
                value=replace(
                    settled.external_outcome_snapshot.value,
                    entries=(
                        replace(
                            settled.external_outcome_snapshot.value.entries[0],
                            description="forged external description",
                        ),
                    ),
                ),
            ),
        )
    with pytest.raises(ValueError, match="preference owner exact derivation"):
        replace(
            settled,
            settlement=replace(
                settled.settlement,
                predicted_probability=0.6,
            ),
        )
    with pytest.raises(
        ValueError,
        match="frozen preference owner outcome evidence drifted",
    ):
        replace(
            settled,
            settlement_input=replace(
                settlement_input,
                owner_outcome_evidence=replace(
                    settlement_input.owner_outcome_evidence,
                    reaction_summary="forged reaction",
                ),
            ),
        )
    with pytest.raises(ValueError, match="frozen settlement credit timestamp mismatch"):
        replace(
            settled,
            credit=replace(
                settled.credit,
                timestamp_ms=settled.credit.timestamp_ms + 1,
            ),
        )
    with pytest.raises(
        ValueError,
        match="frozen settlement persistence is not the exact owner transition",
    ):
        replace(
            settled,
            owner_persistence_snapshot=strict.owner_persistence_snapshot,
        )

    subset_store = SocialRecordStore()
    subset_store.hydrate_from_persistence(settled.owner_persistence_snapshot)
    subset_store.set_preference_action_outcomes(
        (settlement_input.owner_outcome_evidence,)
    )
    with pytest.raises(
        ValueError,
        match="frozen settlement persistence is not the exact owner transition",
    ):
        replace(
            settled,
            preference_snapshot=replace(
                settled.preference_snapshot,
                value=replace(
                    settled.preference_snapshot.value,
                    action_outcome_evidence=(
                        settlement_input.owner_outcome_evidence,
                    ),
                ),
            ),
            owner_persistence_snapshot=subset_store.export_persistence_snapshot(),
        )


def test_frozen_gate_noop_is_not_a_nonnoop_treatment_opportunity() -> None:
    owner_state = asyncio.run(_seed_owner_state())
    policy = _frozen_policy(bias=-2.0)
    applied = _prepare_frozen_preaction(
        policy=policy,
        owner_state=owner_state,
        disposition=RelationshipProductExecutorDisposition.APPLY_CANDIDATE,
    )
    strict = _prepare_frozen_preaction(
        policy=policy,
        owner_state=owner_state,
        disposition=RelationshipProductExecutorDisposition.FORCE_STRICT_NOOP,
    )

    assert not applied.execution_receipt.command.non_noop_opportunity
    assert (
        applied.execution_receipt.executor_status
        is RelationshipProductExecutorStatus.GATE_NOOP
    )
    assert applied.delivered_action_id == "neutral_noop"
    assert strict.delivered_action_id == "neutral_noop"
    assert not strict.execution_receipt.action_diverged
    assert strict.execution_receipt.strict_noop_substituted


def test_frozen_authorization_and_decision_replay_fail_closed() -> None:
    owner_state = asyncio.run(_seed_owner_state())
    policy = _frozen_policy()
    authorization = _frozen_authorization(policy)
    common = {
        "request": _request(decision_index=0),
        "owner_persistence_snapshot": owner_state,
        "forecast_runtime": _reader(),
        "frozen_policy": policy,
        "executor_disposition": (
            RelationshipProductExecutorDisposition.APPLY_CANDIDATE
        ),
        "substrate_snapshot": _placeholder_substrate(),
    }
    with pytest.raises(ValueError, match="policy id is outside"):
        asyncio.run(
            prepare_relationship_product_frozen_preaction(
                **common,
                authorization=replace(
                    authorization,
                    allowed_frozen_policy_id="wrong-frozen-policy",
                ),
            )
        )
    with pytest.raises(ValueError, match="checkpoint is outside"):
        asyncio.run(
            prepare_relationship_product_frozen_preaction(
                **common,
                authorization=replace(
                    authorization,
                    allowed_checkpoint_content_sha256="0" * 64,
                ),
            )
        )

    valid = _prepare_frozen_preaction(
        policy=policy,
        owner_state=owner_state,
        disposition=RelationshipProductExecutorDisposition.APPLY_CANDIDATE,
    )
    decision = valid.frozen_decision.decision
    forged_action = (
        RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION.value
        if decision.selected_action_id
        != RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION.value
        else RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value
    )
    forged_decision = replace(
        valid.frozen_decision,
        decision=replace(
            decision,
            selected_action_id=forged_action,
        ),
    )
    with pytest.raises(ValueError, match="exact frozen-policy replay"):
        RelationshipProductExecutorCommand(
            forecast=valid.forecast,
            frozen_policy=policy,
            frozen_decision=forged_decision,
            authorization=authorization,
            owner_prestate_sha256=(
                valid.execution_receipt.command.owner_prestate_sha256
            ),
            executor_disposition=(
                RelationshipProductExecutorDisposition.APPLY_CANDIDATE
            ),
        )


def test_executor_receipt_rejects_delivered_advisory_tampering() -> None:
    owner_state = asyncio.run(_seed_owner_state())
    strict = _prepare_frozen_preaction(
        policy=_frozen_policy(),
        owner_state=owner_state,
        disposition=RelationshipProductExecutorDisposition.FORCE_STRICT_NOOP,
    )
    receipt = strict.execution_receipt
    with pytest.raises(ValueError, match="delivered advisory drifted"):
        RelationshipProductExecutorReceipt(
            command=receipt.command,
            candidate_advisory=receipt.candidate_advisory,
            delivered_advisory=replace(
                receipt.delivered_advisory,
                action_id=receipt.command.candidate_action_id,
            ),
            temporal_delivery=receipt.temporal_delivery,
        )
    with pytest.raises(ValueError, match="temporal delivery owner envelope drifted"):
        RelationshipProductExecutorReceipt(
            command=receipt.command,
            candidate_advisory=receipt.candidate_advisory,
            delivered_advisory=receipt.delivered_advisory,
            temporal_delivery=replace(
                receipt.temporal_delivery,
                owner="ForgedTemporalOwner",
            ),
        )
    assert "temporal_snapshot" not in {
        field.name for field in fields(RelationshipProductExecutorReceipt)
    }
    changed_projection = replace(
        receipt,
        temporal_delivery=replace(
            receipt.temporal_delivery,
            controller_params_hash="different-controller-params",
        ),
    )
    assert changed_projection.receipt_id != receipt.receipt_id
