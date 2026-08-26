from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import pytest

from lifeform_domain_emogpt.relationship_action_contracts import RelationshipAction
from lifeform_domain_emogpt.relationship_action_gate import (
    RELATIONSHIP_ACTION_CREDIT_LEVEL,
    RelationshipActionGate,
    RelationshipActionGateArtifact,
    RelationshipActionGateBatchDisposition,
    RelationshipActionGateBatchPlan,
    RelationshipActionGateBatchReceipt,
    RelationshipActionGateCheckpoint,
    RelationshipActionGateCreditBatch,
    RelationshipActionGateDecision,
    RelationshipActionGateForcedExposure,
    RelationshipActionGateFrozenDecision,
    RelationshipActionGateFrozenPolicy,
    RelationshipActionGateMode,
    RelationshipActionGateTheta0Artifact,
    RelationshipGateAction,
    temporal_action_advisory_from_gate_decision,
)
from volvence_zero.credit import CreditRecord
from volvence_zero.memory import Track
from volvence_zero.runtime import WiringLevel
from volvence_zero.social_cognition import (
    PreferenceActionForecast,
    SocialActionCandidatePrediction,
    SocialActionOutcomeProbability,
)
from volvence_zero.substrate import SubstrateSnapshot, SurfaceKind
from volvence_zero.temporal import PlaceholderTemporalPolicy, TrackTemporalModule
from volvence_zero.temporal_types import TemporalActionAdvisoryStatus


_OUTCOMES = ("helped", "felt_heard", "missed", "over_directive")


def _candidate(
    action_id: str,
    probabilities: tuple[float, float, float, float],
) -> SocialActionCandidatePrediction:
    return SocialActionCandidatePrediction(
        action_id=action_id,
        outcomes=tuple(
            SocialActionOutcomeProbability(outcome_id, probability)
            for outcome_id, probability in zip(
                _OUTCOMES,
                probabilities,
                strict=True,
            )
        ),
    )


def _forecast(suffix: str = "1") -> PreferenceActionForecast:
    return PreferenceActionForecast(
        forecast_id=f"relationship-forecast-{suffix}",
        decision_id=f"relationship-decision-{suffix}",
        interlocutor_id="primary",
        candidate_predictions=(
            _candidate("stay_present_without_probe", (0.15, 0.55, 0.2, 0.1)),
            _candidate("respect_space_with_return_option", (0.2, 0.2, 0.2, 0.4)),
            _candidate("neutral_noop", (0.25, 0.25, 0.25, 0.25)),
        ),
        recommended_action_id="stay_present_without_probe",
        confidence=0.8,
        source_record_ids=("preference-record-1", "preference-record-2"),
        issued_turn=4,
        evidence=("runtime:bounded-owner-reader",),
        session_scope="closed-alpha-user-1",
    )


def _credit(decision, *, value: float) -> CreditRecord:
    return CreditRecord(
        record_id=f"relationship-credit:{decision.forecast_id}",
        level=RELATIONSHIP_ACTION_CREDIT_LEVEL,
        track=Track.SELF,
        source_event=f"social_pe:{decision.forecast_id}",
        credit_value=value,
        context="typed PE-derived relationship action credit",
        timestamp_ms=5000,
        prediction_id=decision.forecast_id,
        environment_outcome_id=f"external-outcome:{decision.forecast_id}",
        abstract_action_id=decision.selected_action_id,
    )


def _theta0_source_checkpoint() -> RelationshipActionGateCheckpoint:
    gate = RelationshipActionGate()
    decision = gate.decide(
        _forecast("theta0-source"),
        mode=RelationshipActionGateMode.LEARNED,
    )
    gate.observe_credit(_credit(decision, value=-0.8))
    return gate.export_checkpoint()


def _theta0() -> RelationshipActionGateTheta0Artifact:
    return RelationshipActionGateTheta0Artifact.create(
        source_checkpoint=_theta0_source_checkpoint(),
        learning_rate=0.25,
        max_abs_parameter=4.0,
        source_batch_artifact_id=f"theta0-calibration-batch-sha256:{'b' * 64}",
    )


def _theta0_batch(
    gate: RelationshipActionGate,
) -> RelationshipActionGateCreditBatch:
    first = gate.record_forced_exposure(
        _forecast("theta0-1"),
        forced_action_id=RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
        sequence_index=0,
    )
    second = gate.record_forced_exposure(
        _forecast("theta0-2"),
        forced_action_id=RelationshipAction.NEUTRAL_NOOP.value,
        sequence_index=1,
    )
    exposures = (first, second)
    credits = tuple(
        replace(
            _credit(exposure.frozen_decision.decision, value=value),
            record_id=f"relationship-action-pe-credit:theta0-settlement-{index}",
            source_event=f"social_pe:social-pe:theta0-settlement-{index}",
            timestamp_ms=6000 + index,
            abstract_action_id=exposure.forced_action_id,
        )
        for index, (exposure, value) in enumerate(
            zip(exposures, (0.8, -0.6), strict=True)
        )
    )
    return RelationshipActionGateCreditBatch(
        exposures=exposures,
        credits=credits,
    )


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _substrate() -> SubstrateSnapshot:
    return SubstrateSnapshot(
        model_id="frozen-test-substrate",
        is_frozen=True,
        surface_kind=SurfaceKind.PLACEHOLDER,
        token_logits=(),
        feature_surface=(),
        residual_activations=(),
        residual_sequence=(),
        unavailable_fields=(),
        description="gate self-temporal contract fixture",
    )


def test_legacy_gate_payloads_and_state_hashes_remain_byte_compatible() -> None:
    gate = RelationshipActionGate()
    decision = gate.decide(_forecast(), mode=RelationshipActionGateMode.LEARNED)
    cold_pending = gate.export_checkpoint()

    assert _canonical_sha256(decision.to_payload()) == (
        "1948d531f37d00cb7925b25940e60fe62ef47377a77eb94b1c2e91e7b1ca470e"
    )
    assert cold_pending.content_sha256 == (
        "99afe1d177265fe3c53275e9b9b6e9dc9a9b5f93ac8f87b94774a506067aeb80"
    )
    assert RelationshipActionGateCheckpoint.from_payload(
        cold_pending.to_payload()
    ) == cold_pending
    legacy_envelope = {
        **cold_pending.to_payload(),
        "content_sha256": cold_pending.content_sha256,
    }
    assert RelationshipActionGateCheckpoint.from_payload(legacy_envelope) == (
        cold_pending
    )

    gate.observe_credit(_credit(decision, value=-0.8))
    post = gate.export_checkpoint()
    assert post.checkpoint_sha256 == (
        "78efc585826734375dc17a2805d63b296fa3761ce10f6fead472b0bad7dbdad4"
    )
    assert post.content_sha256 == (
        "1577cffc336b8c2bf16644588b0f6a755f10495d89b9a8abfdc0835aaface83f"
    )


def test_control_modes_are_explicit_and_oracle_is_evaluator_only() -> None:
    forecast = _forecast()
    gate = RelationshipActionGate(random_seed="frozen-control-seed")

    noop = gate.decide(forecast, mode=RelationshipActionGateMode.NOOP)
    always = gate.decide(forecast, mode=RelationshipActionGateMode.ALWAYS)
    random_a = gate.decide(forecast, mode=RelationshipActionGateMode.RANDOM)
    random_b = gate.decide(forecast, mode=RelationshipActionGateMode.RANDOM)
    oracle = gate.decide(
        forecast,
        mode=RelationshipActionGateMode.ORACLE,
        oracle_action_id=RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION.value,
        evaluator_only=True,
    )

    assert noop.gate_action is RelationshipGateAction.NOOP
    assert noop.selected_action_id == RelationshipAction.NEUTRAL_NOOP.value
    assert always.gate_action is RelationshipGateAction.STEER
    assert always.selected_action_id == forecast.recommended_action_id
    assert random_a == random_b
    assert oracle.evaluator_only is True
    assert oracle.selected_action_id == "respect_space_with_return_option"
    with pytest.raises(ValueError, match="evaluator_only"):
        gate.decide(
            forecast,
            mode=RelationshipActionGateMode.ORACLE,
            oracle_action_id="stay_present_without_probe",
        )


def test_only_matching_pe_credit_updates_the_learned_gate_and_restores() -> None:
    gate = RelationshipActionGate()
    decision = gate.decide(_forecast(), mode=RelationshipActionGateMode.LEARNED)
    assert decision.gate_action is RelationshipGateAction.NOOP
    before = gate.parameter_state

    update = gate.observe_credit(_credit(decision, value=-0.8))

    assert gate.parameter_state != before
    assert update.update_count == 1
    assert update.old_state_sha256 != update.new_state_sha256
    checkpoint = gate.export_checkpoint()
    restored = RelationshipActionGate(checkpoint=checkpoint)
    assert restored.parameter_state == gate.parameter_state
    assert restored.update_count == 1
    next_decision = restored.decide(
        _forecast("2"),
        mode=RelationshipActionGateMode.LEARNED,
    )
    assert next_decision.steer_probability > 0.5
    assert next_decision.gate_action is RelationshipGateAction.STEER

    wrong_level = replace(
        _credit(next_decision, value=0.4),
        level="evaluation_score",
    )
    with pytest.raises(ValueError, match="PE-derived"):
        restored.observe_credit(wrong_level)
    with pytest.raises(ValueError, match="action lineage"):
        restored.observe_credit(
            replace(
                _credit(next_decision, value=0.4),
                abstract_action_id="neutral_noop",
            )
        )


def test_theta0_artifact_is_nonzero_content_addressed_and_lineage_bound() -> None:
    source = _theta0_source_checkpoint()
    theta0 = _theta0()

    assert theta0.artifact_id.startswith(
        "relationship-action-gate-theta0-sha256:"
    )
    assert theta0.artifact_id == (
        "relationship-action-gate-theta0-sha256:"
        "58d1b2d5d3587d5885d05715a38c292fd9f9661ab779191159c80744725dcff4"
    )
    assert theta0.source_checkpoint_content_sha256 == (
        "537d7d0ff6905e9e56af5f167ba90c3000f4a02f26ce8dc7811572f2da7508cc"
    )
    assert any(value != 0.0 for value in (*theta0.weights, theta0.bias))
    assert theta0.to_runtime_artifact() == RelationshipActionGateArtifact(
        artifact_id=theta0.artifact_id,
        artifact_version=2,
        initial_weights=theta0.weights,
        initial_bias=theta0.bias,
        learning_rate=theta0.learning_rate,
        max_abs_parameter=theta0.max_abs_parameter,
    )
    with pytest.raises(ValueError, match="non-zero"):
        RelationshipActionGateTheta0Artifact.create(
            source_checkpoint=replace(
                source,
                weights=(0.0, 0.0, 0.0, 0.0, 0.0),
                bias=0.0,
            ),
            learning_rate=0.25,
            max_abs_parameter=4.0,
            source_batch_artifact_id=f"theta0-batch-sha256:{'b' * 64}",
        )
    with pytest.raises(ValueError, match="end in a canonical SHA-256"):
        RelationshipActionGateTheta0Artifact.create(
            source_checkpoint=source,
            learning_rate=theta0.learning_rate,
            max_abs_parameter=theta0.max_abs_parameter,
            source_batch_artifact_id="mutable-training-batch:v1",
        )
    with pytest.raises(ValueError, match="canonical content"):
        replace(theta0, bias_hex=(theta0.bias + 0.1).hex())
    with pytest.raises(ValueError, match="content hash mismatch"):
        theta0.validate_source_checkpoint(replace(source, bias=source.bias + 0.01))


def test_theta0_artifact_and_transition_payloads_strictly_round_trip() -> None:
    theta0 = _theta0()
    batch = _theta0_batch(RelationshipActionGate.from_theta0(theta0))
    gate = RelationshipActionGate.from_theta0(theta0)
    plan = gate.plan_credit_batch(batch)
    applied = gate.commit_credit_batch(
        plan,
        disposition=RelationshipActionGateBatchDisposition.APPLY,
    )
    withheld_gate = RelationshipActionGate.from_theta0(theta0)
    withheld = withheld_gate.commit_credit_batch(
        withheld_gate.plan_credit_batch(batch),
        disposition=RelationshipActionGateBatchDisposition.WITHHOLD,
    )

    assert RelationshipActionGateTheta0Artifact.from_payload(
        theta0.to_payload()
    ) == theta0
    assert RelationshipActionGateFrozenDecision.from_payload(
        batch.exposures[0].frozen_decision.to_payload()
    ) == batch.exposures[0].frozen_decision
    assert RelationshipActionGateForcedExposure.from_payload(
        batch.exposures[0].to_payload()
    ) == batch.exposures[0]
    replayed_batch = RelationshipActionGateCreditBatch.from_payload(
        batch.to_payload()
    )
    assert replayed_batch == batch
    replayed_receipt = RelationshipActionGateBatchReceipt.from_payload(
        applied.to_payload()
    )
    assert replayed_receipt == applied
    assert RelationshipActionGateBatchReceipt.from_payload(
        withheld.to_payload()
    ) == withheld
    assert RelationshipActionGate.from_applied_credit_batch(
        theta0,
        batch=replayed_batch,
        receipt=replayed_receipt,
    ).export_checkpoint() == gate.export_checkpoint()

    for loader, payload in (
        (RelationshipActionGateTheta0Artifact.from_payload, theta0.to_payload()),
        (
            RelationshipActionGateFrozenDecision.from_payload,
            batch.exposures[0].frozen_decision.to_payload(),
        ),
        (
            RelationshipActionGateForcedExposure.from_payload,
            batch.exposures[0].to_payload(),
        ),
        (RelationshipActionGateCreditBatch.from_payload, batch.to_payload()),
        (RelationshipActionGateBatchReceipt.from_payload, applied.to_payload()),
    ):
        forged = copy.deepcopy(payload)
        forged["unexpected_field"] = "forbidden"
        with pytest.raises(ValueError, match="fields do not match schema"):
            loader(forged)

    forged_batch = copy.deepcopy(batch.to_payload())
    forged_batch["entries"][0]["credit"]["timestamp_ms"] = True
    with pytest.raises(ValueError, match="timestamp_ms must be an integer"):
        RelationshipActionGateCreditBatch.from_payload(forged_batch)

    forged_exposure = copy.deepcopy(batch.exposures[0].to_payload())
    forged_exposure["exposure_id"] += ":tampered"
    with pytest.raises(ValueError, match="canonical content"):
        RelationshipActionGateForcedExposure.from_payload(forged_exposure)


def test_theta0_frozen_policy_is_pure_non_noop_and_rejects_forged_cold_state() -> None:
    theta0 = _theta0()
    gate = RelationshipActionGate.from_theta0(theta0)
    cold = gate.validate_frozen_theta0()
    frozen_policy = gate.freeze_for_evaluation()

    frozen = frozen_policy.decide(_forecast("theta0-frozen"))
    assert isinstance(frozen.decision, RelationshipActionGateDecision)
    assert frozen.decision.gate_action is RelationshipGateAction.STEER
    assert frozen.decision.selected_action_id == (
        RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value
    )
    assert gate.export_checkpoint() == cold
    assert not gate.export_checkpoint().pending_decisions

    legacy_shaped_clone = RelationshipActionGate(
        artifact=theta0.to_runtime_artifact(),
        checkpoint=cold,
    )
    assert legacy_shaped_clone.decide(_forecast("theta0-frozen")) == frozen.decision
    before_exposure = gate.export_checkpoint()
    exposure = gate.record_forced_exposure(
        _forecast("theta0-exposure"),
        forced_action_id=RelationshipAction.NEUTRAL_NOOP.value,
        sequence_index=0,
    )
    assert exposure.frozen_decision.decision.forecast_id == (
        "relationship-forecast-theta0-exposure"
    )
    assert gate.export_checkpoint() == before_exposure

    forged_cold = replace(
        cold,
        weights=(cold.weights[0] + 0.01, *cold.weights[1:]),
    )
    with pytest.raises(ValueError, match="parameters differ"):
        RelationshipActionGate.from_theta0(theta0, checkpoint=forged_cold)


def test_theta0_batch_apply_and_withhold_share_one_pure_transition() -> None:
    theta0 = _theta0()
    collector = RelationshipActionGate.from_theta0(theta0)
    batch = _theta0_batch(collector)
    assert collector.update_count == 0
    assert collector.freeze_for_evaluation().policy_id == (
        "relationship-action-gate-frozen-policy-sha256:"
        "a9336ab27b8c5e941b560e2c59c6c2c6ea4dc1a7f9c9764f939a63ceec1ac328"
    )
    assert batch.batch_id == (
        "relationship-action-gate-credit-batch-sha256:"
        "31a151033328219a1dd986c7faae74e02c67c843afcf219052bc458923721761"
    )

    applied_gate = RelationshipActionGate.from_theta0(theta0)
    withheld_gate = RelationshipActionGate.from_theta0(theta0)
    applied_plan = applied_gate.plan_credit_batch(batch)
    withheld_plan = withheld_gate.plan_credit_batch(batch)
    assert applied_plan == withheld_plan
    assert applied_plan.plan_id == (
        "relationship-action-gate-batch-plan-sha256:"
        "0ff4d02ee674d7af1a8a7f0612b5cf9b69eeb54d964f963af4e70a12610502a0"
    )
    assert applied_plan.candidate_checkpoint.content_sha256 == (
        "3154072e13ac531d1f6d240419838d817f0942b3958ef65917d49fe12959a783"
    )
    assert tuple(value.hex() for value in applied_plan.candidate_checkpoint.weights) == (
        "0x1.bd93a325f0906p-3",
        "0x1.85e12ec1327e4p-3",
        "0x1.bd93a325f0906p-5",
        "0x1.62cb6724aac1cp-5",
        "0x1.167c45f7b65a2p-3",
    )
    assert applied_plan.candidate_checkpoint.bias.hex() == "0x1.167c45f7b65a2p-2"
    assert applied_plan.pre_checkpoint.content_sha256 == (
        batch.base_checkpoint_content_sha256
    )
    assert applied_plan.candidate_checkpoint != applied_plan.pre_checkpoint

    withheld = withheld_gate.commit_credit_batch(
        withheld_plan,
        disposition=RelationshipActionGateBatchDisposition.WITHHOLD,
    )
    assert withheld.post_checkpoint_content_sha256 == (
        withheld.pre_checkpoint_content_sha256
    )
    assert withheld.update_count_delta == 0
    assert withheld.atomic_commit_count == 0
    assert all(value == 0.0 for value in (*withheld.weight_delta, withheld.bias_delta))
    assert withheld_gate.export_checkpoint() == withheld_plan.pre_checkpoint
    assert withheld_gate.update_count == 0
    assert withheld.receipt_id == (
        "relationship-action-gate-batch-receipt-sha256:"
        "2d7cefe62c265101f2a3202f24e98203a90e134d06a695d2ed42eba4c79d0687"
    )

    applied = applied_gate.commit_credit_batch(
        applied_plan,
        disposition=RelationshipActionGateBatchDisposition.APPLY,
    )
    assert applied.batch_id == withheld.batch_id == batch.batch_id
    assert applied.plan_id == withheld.plan_id == applied_plan.plan_id
    assert applied.post_checkpoint_content_sha256 == (
        applied.candidate_checkpoint_content_sha256
    )
    assert applied.update_count_delta == len(batch.credits)
    assert applied.atomic_commit_count == 1
    assert applied.applied_credit_ids == tuple(
        credit.record_id for credit in batch.credits
    )
    assert applied.receipt_id == (
        "relationship-action-gate-batch-receipt-sha256:"
        "f01e9f0b10e223bebaa9aa8ba9f415e00764ed30b5a9528b7669537e79b0cd50"
    )
    assert tuple(value.hex() for value in applied.weight_delta) == (
        "0x1.19bc98e87fec8p-3",
        "0x1.ed0a0b96dfddcp-4",
        "0x1.19bc98e87fec8p-5",
        "0x1.c0abbeaead54ep-6",
        "0x1.602bbf229fe77p-4",
    )
    assert applied.bias_delta.hex() == "0x1.602bbf229fe77p-3"
    assert applied_gate.export_checkpoint() == applied_plan.candidate_checkpoint
    assert applied_gate.update_count == len(batch.credits)
    with pytest.raises(ValueError, match="cold checkpoint"):
        RelationshipActionGate.from_theta0(
            theta0,
            checkpoint=applied_gate.export_checkpoint(),
        )
    replayed_gate = RelationshipActionGate.from_applied_credit_batch(
        theta0,
        batch=batch,
        receipt=applied,
    )
    assert replayed_gate.export_checkpoint() == applied_gate.export_checkpoint()
    with pytest.raises(ValueError, match="exact batch and APPLY receipt"):
        RelationshipActionGateFrozenPolicy(
            artifact=theta0.to_runtime_artifact(),
            checkpoint=applied_gate.export_checkpoint(),
            random_seed="relationship-action-random-control-v1",
            theta0_artifact=theta0,
        )
    replay_verified_policy = RelationshipActionGateFrozenPolicy(
        artifact=theta0.to_runtime_artifact(),
        checkpoint=applied_gate.export_checkpoint(),
        random_seed="relationship-action-random-control-v1",
        theta0_artifact=theta0,
        transition_batch=batch,
        transition_receipt=applied,
    )
    assert replay_verified_policy.policy_id == (
        applied_gate.freeze_for_evaluation().policy_id
    )
    assert replay_verified_policy.policy_id == (
        "relationship-action-gate-frozen-policy-sha256:"
        "4552ee0ffd1bb2b122bafd7e2243ea8ee0d7d62291fa7d469e938298216370a8"
    )
    assert applied_gate.freeze_for_evaluation().decide(
        _forecast("post-batch")
    ).decision.gate_action is RelationshipGateAction.STEER
    assert withheld_gate.freeze_for_evaluation().decide(
        _forecast("withheld")
    ).decision.gate_action is RelationshipGateAction.STEER


def test_theta0_batch_failures_leave_gate_state_unchanged() -> None:
    theta0 = _theta0()
    collector = RelationshipActionGate.from_theta0(theta0)
    batch = _theta0_batch(collector)
    gate = RelationshipActionGate.from_theta0(theta0)
    pre = gate.export_checkpoint()

    wrong_action_credits = (
        batch.credits[0],
        replace(
            batch.credits[1],
            abstract_action_id=RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
        ),
    )
    with pytest.raises(ValueError, match="forced-action lineage"):
        RelationshipActionGateCreditBatch(
            exposures=batch.exposures,
            credits=wrong_action_credits,
        )
    duplicate_credits = (
        batch.credits[0],
        replace(
            batch.credits[1],
            record_id=batch.credits[0].record_id,
            source_event=batch.credits[0].source_event,
        ),
    )
    with pytest.raises(ValueError, match="record_id values must be unique"):
        RelationshipActionGateCreditBatch(
            exposures=batch.exposures,
            credits=duplicate_credits,
        )
    with pytest.raises(ValueError, match="social PE owner"):
        RelationshipActionGateCreditBatch(
            exposures=batch.exposures,
            credits=(
                batch.credits[0],
                replace(batch.credits[1], source_event="social_pe:"),
            ),
        )
    assert gate.export_checkpoint() == pre

    original_frozen = batch.exposures[0].frozen_decision
    forged_features = (
        original_frozen.decision.features[0] - 0.01,
        *original_frozen.decision.features[1:],
    )
    forged_exposure = replace(
        batch.exposures[0],
        frozen_decision=replace(
            original_frozen,
            decision=replace(
                original_frozen.decision,
                features=forged_features,
            ),
        ),
    )
    forged_batch = RelationshipActionGateCreditBatch(
        exposures=(forged_exposure, batch.exposures[1]),
        credits=batch.credits,
    )
    with pytest.raises(ValueError, match="forged or stale"):
        gate.plan_credit_batch(forged_batch)
    assert gate.export_checkpoint() == pre

    plan = gate.plan_credit_batch(batch)
    forged_candidate = replace(
        plan.candidate_checkpoint,
        weights=(
            plan.candidate_checkpoint.weights[0] + 0.01,
            *plan.candidate_checkpoint.weights[1:],
        ),
    )
    forged_plan = RelationshipActionGateBatchPlan(
        batch=plan.batch,
        pre_checkpoint=plan.pre_checkpoint,
        candidate_checkpoint=forged_candidate,
    )
    with pytest.raises(ValueError, match="pure transition"):
        gate.commit_credit_batch(
            forged_plan,
            disposition=RelationshipActionGateBatchDisposition.APPLY,
        )
    assert gate.export_checkpoint() == pre

    applied_receipt = gate.commit_credit_batch(
        plan,
        disposition=RelationshipActionGateBatchDisposition.APPLY,
    )
    post = gate.export_checkpoint()
    with pytest.raises(ValueError, match="cold checkpoint"):
        gate.commit_credit_batch(
            plan,
            disposition=RelationshipActionGateBatchDisposition.APPLY,
        )
    assert gate.export_checkpoint() == post

    tampered_receipt = replace(
        applied_receipt,
        plan_id=f"{applied_receipt.plan_id}:tampered",
    )
    with pytest.raises(ValueError, match="exact owner replay"):
        RelationshipActionGate.from_applied_credit_batch(
            theta0,
            batch=batch,
            receipt=tampered_receipt,
        )


def test_legacy_nonfinite_credit_fails_before_state_mutation() -> None:
    gate = RelationshipActionGate()
    decision = gate.decide(_forecast("nonfinite"))
    before = gate.export_checkpoint()

    with pytest.raises(ValueError, match="finite"):
        gate.observe_credit(_credit(decision, value=float("nan")))

    assert gate.export_checkpoint() == before


async def test_self_temporal_records_unpromoted_advisory_in_shadow_only() -> None:
    decision = RelationshipActionGate().decide(
        _forecast(),
        mode=RelationshipActionGateMode.ALWAYS,
    )
    advisory = temporal_action_advisory_from_gate_decision(decision)
    module = TrackTemporalModule(
        track=Track.SELF,
        policy=PlaceholderTemporalPolicy(),
        wiring_level=WiringLevel.ACTIVE,
        action_advisory=advisory,
        action_advisory_level=WiringLevel.SHADOW,
    )

    snapshot = await module.process_standalone(substrate_snapshot=_substrate())

    assert snapshot.value.active_abstract_action == "placeholder-controller"
    assert snapshot.value.action_advisory == advisory
    assert (
        snapshot.value.action_advisory_status
        is TemporalActionAdvisoryStatus.SHADOW_RECORDED
    )
    with pytest.raises(ValueError, match="ACTIVE authorization"):
        TrackTemporalModule(
            track=Track.SELF,
            policy=PlaceholderTemporalPolicy(),
            action_advisory=advisory,
            action_advisory_level=WiringLevel.ACTIVE,
        )
    with pytest.raises(ValueError, match="only target self_temporal"):
        TrackTemporalModule(
            track=Track.WORLD,
            policy=PlaceholderTemporalPolicy(),
            action_advisory=advisory,
            action_advisory_level=WiringLevel.SHADOW,
        )
