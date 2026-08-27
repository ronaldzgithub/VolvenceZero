from __future__ import annotations

from dataclasses import replace

import pytest

from lifeform_domain_emogpt.relationship_action_contracts import RelationshipAction
from lifeform_domain_emogpt.relationship_action_gate import (
    RelationshipActionGateBatchDisposition,
    RelationshipGateAction,
)
from lifeform_domain_emogpt.relationship_action_gate_v2 import (
    RELATIONSHIP_ACTION_GATE_V2_FEATURE_ORDER,
    RELATIONSHIP_ACTION_GATE_V2_OBJECTIVE_ID,
    RELATIONSHIP_ACTION_GATE_V2_ONLINE_OBJECTIVE_ID,
    RELATIONSHIP_ACTION_GATE_V2_ONLINE_OPERATOR_ID,
    RelationshipActionGateV2,
    RelationshipActionGateV2Artifact,
    RelationshipActionGateV2ArtifactKind,
    RelationshipActionGateV2AssignmentDesign,
    RelationshipActionGateV2AssignmentReceipt,
    RelationshipActionGateV2AssignmentRole,
    RelationshipActionGateV2AssignmentScheduleArtifact,
    RelationshipActionGateV2AssignmentScheduleEntry,
    RelationshipActionGateV2BatchReceipt,
    RelationshipActionGateV2Checkpoint,
    RelationshipActionGateV2CreditBatch,
    RelationshipActionGateV2Decision,
    RelationshipActionGateV2FederatedAssignmentScheduleArtifact,
    RelationshipActionGateV2FederatedBatchReceipt,
    RelationshipActionGateV2FederatedCreditBatch,
    RelationshipActionGateV2FederatedScheduleSegment,
    RelationshipActionGateV2OnlineExposure,
    RelationshipActionGateV2OnlinePlan,
    RelationshipActionGateV2OnlineReceipt,
    RelationshipActionGateV2OnlineSession,
    RelationshipActionGateV2OnlineTransition,
    RelationshipActionGateV2OnlineTransitionChain,
    commit_relationship_action_gate_v2_federated_matched_transitions,
    relationship_action_gate_v2_features,
    temporal_action_advisory_from_gate_v2_decision,
    temporal_action_advisory_from_gate_v2_online_exposure,
)
from volvence_zero.credit import (
    RelationshipActionCommonBaselineCredit,
    derive_preference_action_common_baseline_credit_records,
)
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
)
from volvence_zero.social import (
    settle_preference_action_forecast,
    social_prediction_error_from_preference_action_forecast_settlement,
)
from volvence_zero.social_cognition import (
    PreferenceActionForecast,
    SocialActionCandidatePrediction,
    SocialActionOutcomeProbability,
)


_OUTCOMES = (
    DialogueExternalOutcomeKind.HELPED.value,
    DialogueExternalOutcomeKind.FELT_HEARD.value,
    DialogueExternalOutcomeKind.MISSED.value,
    DialogueExternalOutcomeKind.OVER_DIRECTIVE.value,
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
                _OUTCOMES,
                probabilities,
                strict=True,
            )
        ),
    )


def _forecast(
    suffix: str,
    *,
    recommended_action_id: str = RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
) -> PreferenceActionForecast:
    return PreferenceActionForecast(
        forecast_id=f"relationship-v2-forecast-{suffix}",
        decision_id=f"relationship-v2-decision-{suffix}",
        interlocutor_id="primary",
        candidate_predictions=(
            _candidate("stay_present_without_probe", (0.15, 0.55, 0.2, 0.1)),
            _candidate("respect_space_with_return_option", (0.2, 0.2, 0.2, 0.4)),
            _candidate("neutral_noop", (0.25, 0.25, 0.25, 0.25)),
        ),
        recommended_action_id=recommended_action_id,
        confidence=0.8,
        source_record_ids=("preference-record-1", "preference-record-2"),
        issued_turn=4,
        evidence=("runtime:bounded-owner-reader",),
        session_scope=f"closed-alpha-user-v2-{suffix}",
    )


def _seed(
    *,
    bootstrap_learning_rate: float = 0.25,
    online_learning_rate: float = 0.125,
    max_abs_parameter: float = 4.0,
    source_digest: str = "a" * 64,
) -> RelationshipActionGateV2Artifact:
    return RelationshipActionGateV2Artifact.create_bootstrap_seed(
        bootstrap_learning_rate=bootstrap_learning_rate,
        online_learning_rate=online_learning_rate,
        max_abs_parameter=max_abs_parameter,
        bootstrap_source_artifact_id=(f"relationship-product-source-sha256:{source_digest}"),
    )


def _schedule(
    forecasts: tuple[PreferenceActionForecast, ...],
    roles: tuple[RelationshipActionGateV2AssignmentRole, ...],
) -> RelationshipActionGateV2AssignmentScheduleArtifact:
    return RelationshipActionGateV2AssignmentScheduleArtifact(
        source_artifact_id=f"relationship-product-source-sha256:{'c' * 64}",
        schedule_scope_id=f"test-schedule:{forecasts[0].session_scope}",
        entries=tuple(
            RelationshipActionGateV2AssignmentScheduleEntry(
                decision_id=forecast.decision_id,
                sequence_index=index,
                assignment_role=role,
            )
            for index, (forecast, role) in enumerate(zip(forecasts, roles, strict=True))
        ),
    )


def _exposures(
    gate: RelationshipActionGateV2,
    forecasts: tuple[PreferenceActionForecast, ...],
    roles: tuple[RelationshipActionGateV2AssignmentRole, ...],
) -> tuple:
    schedule = _schedule(forecasts, roles)
    exposures = []
    for forecast, entry in zip(forecasts, schedule.entries, strict=True):
        assignment = RelationshipActionGateV2AssignmentReceipt(
            schedule_artifact=schedule,
            schedule_entry=entry,
        )
        delivered_action_id = (
            forecast.recommended_action_id
            if entry.assignment_role is RelationshipActionGateV2AssignmentRole.CANDIDATE
            else RelationshipAction.NEUTRAL_NOOP.value
        )
        exposures.append(
            gate.record_forced_exposure(
                forecast,
                assignment=assignment,
                delivered_action_id=delivered_action_id,
            )
        )
    return tuple(exposures)


def _common_credit(
    exposure: object,
    *,
    outcome: DialogueExternalOutcomeKind,
    timestamp_ms: int,
) -> RelationshipActionCommonBaselineCredit:
    forecast = exposure.forecast
    external = DialogueExternalOutcomeEvidence(
        evidence_id=f"environment-outcome:{forecast.forecast_id}",
        turn_index=forecast.issued_turn + 1,
        kind=outcome,
        source=DialogueExternalOutcomeEvidenceSource.ENVIRONMENT,
        confidence=1.0,
        evidence_ref=f"environment-ref:{forecast.forecast_id}",
        description="Typed action-conditioned reactive outcome.",
        session_scope=forecast.session_scope,
        action_turn_index=forecast.issued_turn,
        forecast_id=forecast.forecast_id,
        decision_id=forecast.decision_id,
        action_id=exposure.delivered_action_id,
    )
    settlement = settle_preference_action_forecast(
        forecast=forecast,
        evidence=external,
    )
    social_error = social_prediction_error_from_preference_action_forecast_settlement(settlement)
    records = derive_preference_action_common_baseline_credit_records(
        forecasts=(forecast,),
        external_evidence=(external,),
        settlements=(settlement,),
        social_errors=(social_error,),
        settled_at_turn=settlement.observed_turn,
        timestamp_ms=timestamp_ms,
    )
    assert len(records) == 1
    return records[0]


def _batch(
    gate: RelationshipActionGateV2,
    specs: tuple[
        tuple[
            str,
            RelationshipActionGateV2AssignmentRole,
            str,
            DialogueExternalOutcomeKind,
        ],
        ...,
    ],
    *,
    timestamp_start: int = 5000,
) -> RelationshipActionGateV2CreditBatch:
    forecasts = tuple(
        _forecast(suffix, recommended_action_id=recommended) for suffix, _role, recommended, _outcome in specs
    )
    roles = tuple(role for _suffix, role, _recommended, _outcome in specs)
    exposures = _exposures(gate, forecasts, roles)
    credits = tuple(
        _common_credit(
            exposure,
            outcome=specs[index][3],
            timestamp_ms=timestamp_start + index,
        )
        for index, exposure in enumerate(exposures)
    )
    return RelationshipActionGateV2CreditBatch(
        exposures=exposures,
        credits=credits,
    )


def _informative_specs() -> tuple:
    return (
        (
            "candidate-helped",
            RelationshipActionGateV2AssignmentRole.CANDIDATE,
            RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
            DialogueExternalOutcomeKind.HELPED,
        ),
        (
            "noop-over-directive",
            RelationshipActionGateV2AssignmentRole.NEUTRAL_NOOP,
            RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
            DialogueExternalOutcomeKind.OVER_DIRECTIVE,
        ),
    )


def _federation(
    seed: RelationshipActionGateV2Artifact,
) -> RelationshipActionGateV2FederatedCreditBatch:
    first = _batch(
        RelationshipActionGateV2(artifact=seed),
        (
            (
                "federated-a-candidate",
                RelationshipActionGateV2AssignmentRole.CANDIDATE,
                RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
                DialogueExternalOutcomeKind.HELPED,
            ),
            (
                "federated-a-noop",
                RelationshipActionGateV2AssignmentRole.NEUTRAL_NOOP,
                RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
                DialogueExternalOutcomeKind.OVER_DIRECTIVE,
            ),
        ),
        timestamp_start=5000,
    )
    second = _batch(
        RelationshipActionGateV2(artifact=seed),
        (
            (
                "federated-b-candidate",
                RelationshipActionGateV2AssignmentRole.CANDIDATE,
                RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION.value,
                DialogueExternalOutcomeKind.FELT_HEARD,
            ),
            (
                "federated-b-noop",
                RelationshipActionGateV2AssignmentRole.NEUTRAL_NOOP,
                RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION.value,
                DialogueExternalOutcomeKind.MISSED,
            ),
        ),
        timestamp_start=6000,
    )
    parent = RelationshipActionGateV2FederatedAssignmentScheduleArtifact(
        source_artifact_id=first.schedule_artifact.source_artifact_id,
        schedule_scope_id="test-federated-schedule:two-roots",
        segments=(
            RelationshipActionGateV2FederatedScheduleSegment(
                global_start_index=0,
                child_schedule_artifact=first.schedule_artifact,
            ),
            RelationshipActionGateV2FederatedScheduleSegment(
                global_start_index=len(first.exposures),
                child_schedule_artifact=second.schedule_artifact,
            ),
        ),
    )
    return RelationshipActionGateV2FederatedCreditBatch(
        federated_schedule_artifact=parent,
        child_batches=(first, second),
    )


def _learned_theta0(
    *,
    bootstrap_learning_rate: float = 1.0 / 512.0,
    bootstrap_pair_count: int = 1,
    online_learning_rate: float = 0.25,
    max_abs_parameter: float = 4.0,
) -> RelationshipActionGateV2Artifact:
    seed = _seed(
        bootstrap_learning_rate=bootstrap_learning_rate,
        online_learning_rate=online_learning_rate,
        max_abs_parameter=max_abs_parameter,
    )
    specs = (
        _informative_specs()
        if bootstrap_pair_count == 1
        else tuple(
            item
            for index in range(bootstrap_pair_count)
            for item in (
                (
                    f"candidate-helped-{index}",
                    RelationshipActionGateV2AssignmentRole.CANDIDATE,
                    RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
                    DialogueExternalOutcomeKind.HELPED,
                ),
                (
                    f"noop-over-directive-{index}",
                    RelationshipActionGateV2AssignmentRole.NEUTRAL_NOOP,
                    RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
                    DialogueExternalOutcomeKind.OVER_DIRECTIVE,
                ),
            )
        )
    )
    batch = _batch(RelationshipActionGateV2(artifact=seed), specs)
    gate = RelationshipActionGateV2(artifact=seed)
    receipt = gate.commit_credit_batch(
        gate.plan_credit_batch(batch),
        disposition=RelationshipActionGateBatchDisposition.APPLY,
    )
    return RelationshipActionGateV2Artifact.create_learned_theta0(
        parent_artifact=seed,
        source_batch=batch,
        apply_receipt=receipt,
    )


def _online_step(
    session: RelationshipActionGateV2OnlineSession,
    *,
    suffix: str,
    outcome: DialogueExternalOutcomeKind,
    timestamp_ms: int,
    recommended_action_id: str = RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
) -> RelationshipActionGateV2OnlineTransition:
    forecast = _forecast(suffix, recommended_action_id=recommended_action_id)
    return _online_step_for_forecast(
        session,
        forecast=forecast,
        outcome=outcome,
        timestamp_ms=timestamp_ms,
    )


def _online_step_for_forecast(
    session: RelationshipActionGateV2OnlineSession,
    *,
    forecast: PreferenceActionForecast,
    outcome: DialogueExternalOutcomeKind,
    timestamp_ms: int,
) -> RelationshipActionGateV2OnlineTransition:
    decision = session.decide(forecast)
    exposure = session.record_exposure(
        forecast,
        delivered_action_id=decision.decision.selected_action_id,
    )
    credit = _common_credit(
        exposure,
        outcome=outcome,
        timestamp_ms=timestamp_ms,
    )
    return session.commit_credit(session.plan_credit(exposure, credit))


def test_v2_feature_threshold_and_frozen_advisory_lineage() -> None:
    gate = RelationshipActionGateV2(artifact=_seed())
    forecast = _forecast("feature")
    policy = gate.freeze_for_evaluation()
    frozen = policy.decide(forecast)

    assert RELATIONSHIP_ACTION_GATE_V2_FEATURE_ORDER == (
        "forecast_confidence_centered",
        "recommended_positive_mass_centered",
        "recommended_positive_margin_over_noop",
        "recommended_entropy_certainty_centered",
    )
    assert len(relationship_action_gate_v2_features(_forecast("feature-2"))) == 4
    assert frozen.decision.features[0] == pytest.approx(0.6)
    assert frozen.decision.gate_action is RelationshipGateAction.NOOP
    assert frozen.decision.steer_probability == 0.5
    assert policy.policy_id == (
        "relationship-action-gate-v2-frozen-policy-sha256:"
        "54bf9e617e4ea387ae2641363d0bfbb166d8c0f10f20a9afbaadc748e7c5eaf3"
    )
    assert frozen.checkpoint_content_sha256 == (
        "5656821386b4b64b5ddd9fe8fecaff20d072d8995d41efe139d85fa3e8ccf8ea"
    )
    assert RelationshipActionGateV2Decision.from_payload(frozen.decision.to_payload()) == frozen.decision

    with pytest.raises(ValueError, match="strict probability threshold"):
        replace(
            frozen.decision,
            gate_action=RelationshipGateAction.STEER,
            selected_action_id=frozen.decision.recommended_action_id,
            steer_probability=0.1,
        )

    advisory = temporal_action_advisory_from_gate_v2_decision(
        frozen,
        frozen_policy=policy,
        forecast=forecast,
    )
    assert advisory.action_id == RelationshipAction.NEUTRAL_NOOP.value
    assert advisory.active_authorized is False
    assert advisory.policy_artifact_id == frozen.decision.artifact_id
    assert advisory.advisory_id == (
        "relationship-action-advisory-v2-sha256:"
        "c8b2725f99d65d5f73f426ac447cb87029d2f47dc945d5cca6bfedc8147b3bb3"
    )
    assert frozen.frozen_policy_id in advisory.evidence_refs
    assert any("checkpoint-sha256" in item for item in advisory.evidence_refs)
    assert any("preference-action-forecast-sha256" in item for item in advisory.evidence_refs)
    with pytest.raises(TypeError, match="frozen_decision"):
        temporal_action_advisory_from_gate_v2_decision(  # type: ignore[arg-type]
            frozen.decision,
            frozen_policy=policy,
            forecast=forecast,
        )
    with pytest.raises(ValueError, match="exact policy replay"):
        temporal_action_advisory_from_gate_v2_decision(
            replace(frozen, frozen_policy_id=f"relationship-action-gate-v2-frozen-policy-sha256:{'f' * 64}"),
            frozen_policy=policy,
            forecast=forecast,
        )


def test_fixed_schedule_binds_complete_exact_membership_without_randomized_claim() -> None:
    forecasts = (_forecast("schedule-a"), _forecast("schedule-b"))
    schedule = _schedule(
        forecasts,
        (
            RelationshipActionGateV2AssignmentRole.CANDIDATE,
            RelationshipActionGateV2AssignmentRole.NEUTRAL_NOOP,
        ),
    )
    assert schedule.assignment_design is (RelationshipActionGateV2AssignmentDesign.FIXED_BALANCED_HALF)
    assert RelationshipActionGateV2AssignmentScheduleArtifact.from_payload(schedule.to_payload()) == schedule
    receipt = RelationshipActionGateV2AssignmentReceipt(
        schedule_artifact=schedule,
        schedule_entry=schedule.entries[0],
    )
    assert RelationshipActionGateV2AssignmentReceipt.from_payload(receipt.to_payload()) == receipt

    forged_role = replace(
        schedule.entries[0],
        assignment_role=RelationshipActionGateV2AssignmentRole.NEUTRAL_NOOP,
    )
    with pytest.raises(ValueError, match="exact schedule member"):
        RelationshipActionGateV2AssignmentReceipt(
            schedule_artifact=schedule,
            schedule_entry=forged_role,
        )
    with pytest.raises(ValueError, match="exactly candidate/noop balanced"):
        RelationshipActionGateV2AssignmentScheduleArtifact(
            source_artifact_id=schedule.source_artifact_id,
            schedule_scope_id="unbalanced",
            entries=(schedule.entries[0],),
        )
    with pytest.raises(ValueError):
        RelationshipActionGateV2AssignmentDesign("sealed-randomized-half.v1")


def test_artifact_uses_exact_batch_apply_lineage_and_resets_runtime_counts() -> None:
    seed = _seed(
        bootstrap_learning_rate=1.0 / 512.0,
        online_learning_rate=0.25,
    )
    assert seed.artifact_kind is RelationshipActionGateV2ArtifactKind.BOOTSTRAP_SEED
    assert seed.bootstrap_learning_rate == 1.0 / 512.0
    assert seed.online_learning_rate == 0.25
    assert seed.objective_id == RELATIONSHIP_ACTION_GATE_V2_OBJECTIVE_ID
    assert RelationshipActionGateV2Artifact.from_payload(seed.to_payload()) == seed

    collection_gate = RelationshipActionGateV2(artifact=seed)
    batch = _batch(collection_gate, _informative_specs())
    apply_gate = RelationshipActionGateV2(artifact=seed)
    receipt = apply_gate.commit_credit_batch(
        apply_gate.plan_credit_batch(batch),
        disposition=RelationshipActionGateBatchDisposition.APPLY,
    )
    terminal = apply_gate.export_checkpoint()
    theta0 = RelationshipActionGateV2Artifact.create_learned_theta0(
        parent_artifact=seed,
        source_batch=batch,
        apply_receipt=receipt,
    )

    assert receipt.atomic_commit_count == 1
    assert terminal.update_count == 2
    assert terminal.informative_update_count == 2
    assert any(value != 0.0 for value in terminal.weights)
    assert theta0.artifact_kind is RelationshipActionGateV2ArtifactKind.LEARNED_THETA0
    assert theta0.active_learning_rate == 0.25
    assert theta0.source_parent_artifact_id == seed.artifact_id
    assert theta0.source_credit_batch_id == batch.batch_id
    assert theta0.source_apply_receipt_id == receipt.receipt_id
    theta0.validate_source_transition(
        parent_artifact=seed,
        source_batch=batch,
        apply_receipt=receipt,
    )
    with pytest.raises(ValueError, match="full source transition components"):
        RelationshipActionGateV2Artifact.from_payload(theta0.to_payload())
    with pytest.raises(ValueError, match="full source transition components"):
        RelationshipActionGateV2Artifact.from_payload(
            theta0.to_payload(),
            parent_artifact=seed,
            source_batch=batch,
        )
    assert (
        RelationshipActionGateV2Artifact.from_payload(
            theta0.to_payload(),
            parent_artifact=seed,
            source_batch=batch,
            apply_receipt=receipt,
        )
        == theta0
    )
    cold = RelationshipActionGateV2(artifact=theta0).export_checkpoint()
    assert cold.weights == terminal.weights
    assert cold.update_count == 0
    assert cold.informative_update_count == 0
    assert cold.processed_credit_ids == ()

    with pytest.raises(ValueError, match="full source transition components"):
        replace(theta0, weights_hex=theta0.weights_hex)
    with pytest.raises(ValueError, match="nonzero value"):
        replace(
            theta0,
            weights_hex=((0.0).hex(), (-0.0).hex(), (0.0).hex(), (-0.0).hex()),
        )
    with pytest.raises(ValueError, match="loading cannot cite"):
        RelationshipActionGateV2Artifact.from_payload(
            seed.to_payload(),
            parent_artifact=seed,
            source_batch=batch,
            apply_receipt=receipt,
        )

    wrong_parent = _seed(source_digest="b" * 64)
    with pytest.raises(ValueError, match="source transition drifted"):
        theta0.validate_source_transition(
            parent_artifact=wrong_parent,
            source_batch=batch,
            apply_receipt=receipt,
        )

    withhold_gate = RelationshipActionGateV2(artifact=seed)
    withhold = withhold_gate.commit_credit_batch(
        withhold_gate.plan_credit_batch(batch),
        disposition=RelationshipActionGateBatchDisposition.WITHHOLD,
    )
    with pytest.raises(ValueError, match="APPLY receipt"):
        RelationshipActionGateV2Artifact.create_learned_theta0(
            parent_artifact=seed,
            source_batch=batch,
            apply_receipt=withhold,
        )


def test_learned_theta0_rejects_informative_exact_cancellation() -> None:
    seed = _seed()
    specs = (
        (
            "candidate-same-outcome",
            RelationshipActionGateV2AssignmentRole.CANDIDATE,
            RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
            DialogueExternalOutcomeKind.HELPED,
        ),
        (
            "noop-same-outcome",
            RelationshipActionGateV2AssignmentRole.NEUTRAL_NOOP,
            RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
            DialogueExternalOutcomeKind.HELPED,
        ),
    )
    batch = _batch(RelationshipActionGateV2(artifact=seed), specs)
    gate = RelationshipActionGateV2(artifact=seed)
    plan = gate.plan_credit_batch(batch)
    receipt = gate.commit_credit_batch(
        plan,
        disposition=RelationshipActionGateBatchDisposition.APPLY,
    )

    assert plan.informative_count == 2
    assert receipt.atomic_commit_count == 1
    assert gate.export_checkpoint().informative_update_count == 2
    assert all(value == 0.0 for value in gate.export_checkpoint().weights)
    with pytest.raises(ValueError, match="zero net update"):
        RelationshipActionGateV2Artifact.create_learned_theta0(
            parent_artifact=seed,
            source_batch=batch,
            apply_receipt=receipt,
        )


def test_fixed_half_common_credit_score_is_role_swap_symmetric() -> None:
    seed = _seed()
    recommended = RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value
    batch_a = _batch(
        RelationshipActionGateV2(artifact=seed),
        (
            (
                "symmetry-helped",
                RelationshipActionGateV2AssignmentRole.CANDIDATE,
                recommended,
                DialogueExternalOutcomeKind.HELPED,
            ),
            (
                "symmetry-harmed",
                RelationshipActionGateV2AssignmentRole.NEUTRAL_NOOP,
                recommended,
                DialogueExternalOutcomeKind.OVER_DIRECTIVE,
            ),
        ),
    )
    batch_b = _batch(
        RelationshipActionGateV2(artifact=seed),
        (
            (
                "symmetry-helped",
                RelationshipActionGateV2AssignmentRole.NEUTRAL_NOOP,
                recommended,
                DialogueExternalOutcomeKind.HELPED,
            ),
            (
                "symmetry-harmed",
                RelationshipActionGateV2AssignmentRole.CANDIDATE,
                recommended,
                DialogueExternalOutcomeKind.OVER_DIRECTIVE,
            ),
        ),
    )
    plan_a = RelationshipActionGateV2(artifact=seed).plan_credit_batch(batch_a)
    plan_b = RelationshipActionGateV2(artifact=seed).plan_credit_batch(batch_b)

    assert tuple(item.credit_value for item in batch_a.credits) == (0.5, -0.5)
    assert plan_a.candidate_checkpoint.weights == pytest.approx(
        tuple(-value for value in plan_b.candidate_checkpoint.weights)
    )
    assert plan_a.informative_candidate_count == 1
    assert plan_a.informative_noop_count == 1


def test_owner_recommendation_noop_is_zero_information_but_processed() -> None:
    seed = _seed()
    noop = RelationshipAction.NEUTRAL_NOOP.value
    gate = RelationshipActionGateV2(artifact=seed)
    batch = _batch(
        gate,
        (
            (
                "zero-candidate",
                RelationshipActionGateV2AssignmentRole.CANDIDATE,
                noop,
                DialogueExternalOutcomeKind.HELPED,
            ),
            (
                "zero-noop",
                RelationshipActionGateV2AssignmentRole.NEUTRAL_NOOP,
                noop,
                DialogueExternalOutcomeKind.OVER_DIRECTIVE,
            ),
        ),
    )
    before = gate.export_checkpoint()
    plan = gate.plan_credit_batch(batch)
    receipt = gate.commit_credit_batch(
        plan,
        disposition=RelationshipActionGateBatchDisposition.APPLY,
    )
    after = gate.export_checkpoint()

    assert batch.informative_count == 0
    assert batch.zero_information_count == 2
    assert plan.candidate_checkpoint.weights == before.weights
    assert after.weights == before.weights
    assert after.update_count == 2
    assert after.informative_update_count == 0
    assert receipt.zero_information_count == 2
    assert receipt.weight_delta == (0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="no information"):
        RelationshipActionGateV2Artifact.create_learned_theta0(
            parent_artifact=seed,
            source_batch=batch,
            apply_receipt=receipt,
        )


def test_apply_withhold_replay_and_single_transition_close_exactly() -> None:
    seed = _seed()
    batch = _batch(RelationshipActionGateV2(artifact=seed), _informative_specs())
    payload = batch.to_payload()
    with pytest.raises(TypeError):
        RelationshipActionGateV2CreditBatch.from_payload(payload)  # type: ignore[call-arg]
    assert (
        RelationshipActionGateV2CreditBatch.from_payload(
            payload,
            full_common_credits=batch.credits,
        )
        == batch
    )

    tampered = batch.to_payload()
    tampered_entries = tampered["entries"]
    assert isinstance(tampered_entries, list)
    tampered_credit = tampered_entries[0]["credit"]
    assert isinstance(tampered_credit, dict)
    tampered_credit["record_id"] = "relationship-action-common-baseline-credit-sha256:" + ("0" * 64)
    with pytest.raises(ValueError, match="audit projection mismatch"):
        RelationshipActionGateV2CreditBatch.from_payload(
            tampered,
            full_common_credits=batch.credits,
        )

    apply_gate = RelationshipActionGateV2(artifact=seed)
    plan = apply_gate.plan_credit_batch(batch)
    apply_receipt = apply_gate.commit_credit_batch(
        plan,
        disposition=RelationshipActionGateBatchDisposition.APPLY,
    )
    replayed = RelationshipActionGateV2.from_applied_credit_batch(
        seed,
        batch=batch,
        receipt=apply_receipt,
    )
    assert replayed.export_checkpoint() == apply_gate.export_checkpoint()
    assert RelationshipActionGateV2BatchReceipt.from_payload(apply_receipt.to_payload()) == apply_receipt
    assert (
        RelationshipActionGateV2Checkpoint.from_payload(apply_gate.export_checkpoint().to_payload())
        == apply_gate.export_checkpoint()
    )
    with pytest.raises(ValueError, match="exactly one transition"):
        apply_gate.plan_credit_batch(batch)

    withhold_gate = RelationshipActionGateV2(artifact=seed)
    before = withhold_gate.export_checkpoint()
    withhold_receipt = withhold_gate.commit_credit_batch(
        withhold_gate.plan_credit_batch(batch),
        disposition=RelationshipActionGateBatchDisposition.WITHHOLD,
    )
    assert withhold_gate.export_checkpoint() == before
    assert withhold_gate.freeze_for_evaluation().transition_receipt == withhold_receipt
    assert (
        RelationshipActionGateV2.from_credit_batch_transition(
            seed,
            batch=batch,
            receipt=withhold_receipt,
        ).export_checkpoint()
        == before
    )
    with pytest.raises(ValueError, match="exactly one transition"):
        withhold_gate.plan_credit_batch(batch)


def test_federated_batch_commits_once_and_condenses_to_cold_theta0() -> None:
    seed = _seed(
        bootstrap_learning_rate=1.0 / 512.0,
        online_learning_rate=0.25,
    )
    federation = _federation(seed)
    schedule = federation.federated_schedule_artifact

    assert len(schedule.segments) == 2
    assert len(schedule.flattened_entries) == 4
    assert RelationshipActionGateV2FederatedAssignmentScheduleArtifact.from_payload(schedule.to_payload()) == schedule
    with pytest.raises(TypeError):
        RelationshipActionGateV2FederatedCreditBatch.from_payload(  # type: ignore[call-arg]
            federation.to_payload()
        )
    assert (
        RelationshipActionGateV2FederatedCreditBatch.from_payload(
            federation.to_payload(),
            federated_schedule_artifact=schedule,
            full_common_credit_batches=tuple(child.credits for child in federation.child_batches),
        )
        == federation
    )

    matched = commit_relationship_action_gate_v2_federated_matched_transitions(
        artifact=seed,
        batch=federation,
    )
    applied = matched.applied
    withheld = matched.withheld
    assert applied.gate_receipt.plan_id == withheld.gate_receipt.plan_id
    assert (
        applied.gate_receipt.candidate_checkpoint_content_sha256
        == withheld.gate_receipt.candidate_checkpoint_content_sha256
    )
    assert applied.gate_receipt.atomic_commit_count == 1
    assert applied.gate_receipt.update_count_delta == 4
    assert applied.gate_receipt.informative_update_count_delta == 4
    assert applied.gate_receipt.child_batch_count == 2
    assert applied.gate_receipt.child_transition_count == 0
    assert withheld.gate_receipt.atomic_commit_count == 0
    assert withheld.gate_receipt.update_count_delta == 0
    assert withheld.gate_receipt.informative_update_count_delta == 0
    assert withheld.terminal_checkpoint.update_count == 0
    assert matched.to_payload()["unique_parent_plan_identity_count"] == 1
    assert matched.to_payload()["child_transition_count"] == 0
    assert (
        RelationshipActionGateV2FederatedBatchReceipt.from_payload(applied.gate_receipt.to_payload())
        == applied.gate_receipt
    )
    for field_name, wrong_identity in (
        ("batch_id", federation.child_batches[0].batch_id),
        (
            "federated_schedule_artifact_id",
            federation.child_batches[0].schedule_artifact.artifact_id,
        ),
        (
            "plan_id",
            RelationshipActionGateV2(artifact=seed).plan_credit_batch(federation.child_batches[0]).plan_id,
        ),
    ):
        with pytest.raises(ValueError, match="must use prefix"):
            replace(applied.gate_receipt, **{field_name: wrong_identity})
    for field_name in (
        "batch_id",
        "federated_schedule_artifact_id",
        "plan_id",
    ):
        valid_identity = getattr(applied.gate_receipt, field_name)
        prefix_without_colon, digest = valid_identity.rsplit(":", 1)
        nested_identity = f"{prefix_without_colon}:nested:{digest}"
        with pytest.raises(ValueError, match="one exact prefixed SHA-256"):
            replace(applied.gate_receipt, **{field_name: nested_identity})

    replayed = RelationshipActionGateV2.from_applied_federated_credit_batch(
        seed,
        batch=federation,
        receipt=applied.gate_receipt,
    )
    assert replayed.export_checkpoint() == applied.terminal_checkpoint
    with pytest.raises(ValueError, match="condensed into a learned theta0"):
        replayed.freeze_for_evaluation()

    theta0 = RelationshipActionGateV2Artifact.create_learned_theta0_from_federation(
        parent_artifact=seed,
        source_batch=federation,
        apply_receipt=applied.gate_receipt,
    )
    theta0.validate_federated_source_transition(
        parent_artifact=seed,
        source_batch=federation,
        apply_receipt=applied.gate_receipt,
    )
    assert (
        RelationshipActionGateV2Artifact.from_payload(
            theta0.to_payload(),
            parent_artifact=seed,
            source_batch=federation,
            apply_receipt=applied.gate_receipt,
        )
        == theta0
    )
    cold_gate = RelationshipActionGateV2(artifact=theta0)
    cold = cold_gate.export_checkpoint()
    assert cold.weights == applied.terminal_checkpoint.weights
    assert cold.update_count == 0
    assert cold.informative_update_count == 0
    assert cold.processed_credit_ids == ()
    assert cold_gate.freeze_for_evaluation().checkpoint == cold

    with pytest.raises(TypeError, match="RelationshipActionGateV2CreditBatch"):
        RelationshipActionGateV2(artifact=seed).plan_credit_batch(federation)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="RelationshipActionGateV2CreditBatch"):
        RelationshipActionGateV2Artifact.create_learned_theta0(  # type: ignore[arg-type]
            parent_artifact=seed,
            source_batch=federation,
            apply_receipt=applied.gate_receipt,
        )


def test_federated_candidate_checkpoint_is_byte_exact_to_flat_ordered_math() -> None:
    seed = _seed()
    federation = _federation(seed)
    forecasts = tuple(exposure.forecast for exposure in federation.exposures)
    roles = tuple(exposure.assignment_role for exposure in federation.exposures)
    flat_exposures = _exposures(
        RelationshipActionGateV2(artifact=seed),
        forecasts,
        roles,
    )
    flat_batch = RelationshipActionGateV2CreditBatch(
        exposures=flat_exposures,
        credits=federation.credits,
    )

    flat_plan = RelationshipActionGateV2(artifact=seed).plan_credit_batch(flat_batch)
    federated_plan = RelationshipActionGateV2(artifact=seed).plan_federated_credit_batch(federation)
    assert flat_plan.candidate_checkpoint == federated_plan.candidate_checkpoint
    assert flat_plan.candidate_checkpoint.content_sha256 == federated_plan.candidate_checkpoint.content_sha256
    assert flat_plan.informative_candidate_count == (federated_plan.informative_candidate_count)
    assert flat_plan.informative_noop_count == federated_plan.informative_noop_count
    assert flat_plan.cap_hit_count == federated_plan.cap_hit_count


def test_federation_rejects_order_overlap_drift_and_second_transition() -> None:
    seed = _seed()
    federation = _federation(seed)
    first, second = federation.child_batches
    parent = federation.federated_schedule_artifact

    with pytest.raises(ValueError, match="parent order"):
        replace(federation, child_batches=(second, first))
    with pytest.raises(ValueError, match="child batch ids must be unique"):
        replace(federation, child_batches=(first, first))
    with pytest.raises(ValueError, match="child artifacts must be unique"):
        RelationshipActionGateV2FederatedAssignmentScheduleArtifact(
            source_artifact_id=parent.source_artifact_id,
            schedule_scope_id="duplicate-child-schedule",
            segments=(
                parent.segments[0],
                RelationshipActionGateV2FederatedScheduleSegment(
                    global_start_index=len(first.exposures),
                    child_schedule_artifact=first.schedule_artifact,
                ),
            ),
        )

    duplicate_decision_schedule = replace(
        first.schedule_artifact,
        schedule_scope_id="duplicate-decision-other-scope",
    )
    with pytest.raises(ValueError, match="decision ids must be globally unique"):
        RelationshipActionGateV2FederatedAssignmentScheduleArtifact(
            source_artifact_id=parent.source_artifact_id,
            schedule_scope_id="duplicate-global-decisions",
            segments=(
                parent.segments[0],
                RelationshipActionGateV2FederatedScheduleSegment(
                    global_start_index=len(first.exposures),
                    child_schedule_artifact=duplicate_decision_schedule,
                ),
            ),
        )

    overlapping_second = RelationshipActionGateV2CreditBatch(
        exposures=second.exposures,
        credits=tuple(
            _common_credit(
                exposure,
                outcome=credit.external_evidence.kind,
                timestamp_ms=5001 + index,
            )
            for index, (exposure, credit) in enumerate(zip(second.exposures, second.credits, strict=True))
        ),
    )
    with pytest.raises(ValueError, match="globally increasing"):
        replace(federation, child_batches=(first, overlapping_second))

    wrong_seed = _seed(source_digest="b" * 64)
    wrong_second = _batch(
        RelationshipActionGateV2(artifact=wrong_seed),
        (
            (
                "wrong-seed-candidate",
                RelationshipActionGateV2AssignmentRole.CANDIDATE,
                RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
                DialogueExternalOutcomeKind.HELPED,
            ),
            (
                "wrong-seed-noop",
                RelationshipActionGateV2AssignmentRole.NEUTRAL_NOOP,
                RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
                DialogueExternalOutcomeKind.MISSED,
            ),
        ),
        timestamp_start=7000,
    )
    mixed_parent = RelationshipActionGateV2FederatedAssignmentScheduleArtifact(
        source_artifact_id=parent.source_artifact_id,
        schedule_scope_id="mixed-checkpoint-parent",
        segments=(
            parent.segments[0],
            RelationshipActionGateV2FederatedScheduleSegment(
                global_start_index=len(first.exposures),
                child_schedule_artifact=wrong_second.schedule_artifact,
            ),
        ),
    )
    with pytest.raises(ValueError, match="share one gate artifact"):
        RelationshipActionGateV2FederatedCreditBatch(
            federated_schedule_artifact=mixed_parent,
            child_batches=(first, wrong_second),
        )

    partial = federation.to_payload()
    partial.pop("credit_count")
    with pytest.raises(ValueError, match="fields do not match schema"):
        RelationshipActionGateV2FederatedCreditBatch.from_payload(
            partial,
            federated_schedule_artifact=parent,
            full_common_credit_batches=tuple(child.credits for child in federation.child_batches),
        )
    with pytest.raises(ValueError, match="credit groups do not match"):
        RelationshipActionGateV2FederatedCreditBatch.from_payload(
            federation.to_payload(),
            federated_schedule_artifact=parent,
            full_common_credit_batches=(first.credits,),
        )

    gate = RelationshipActionGateV2(artifact=seed)
    gate.commit_federated_credit_batch(
        gate.plan_federated_credit_batch(federation),
        disposition=RelationshipActionGateBatchDisposition.WITHHOLD,
    )
    with pytest.raises(ValueError, match="exactly one transition"):
        gate.plan_federated_credit_batch(federation)
    with pytest.raises(ValueError, match="condensed into a learned theta0"):
        gate.freeze_for_evaluation()


def test_existing_v2_golden_identities_remain_byte_exact() -> None:
    seed = _seed(
        bootstrap_learning_rate=1.0 / 512.0,
        online_learning_rate=0.25,
    )
    batch = _batch(RelationshipActionGateV2(artifact=seed), _informative_specs())
    gate = RelationshipActionGateV2(artifact=seed)
    plan = gate.plan_credit_batch(batch)
    receipt = gate.commit_credit_batch(
        plan,
        disposition=RelationshipActionGateBatchDisposition.APPLY,
    )

    assert seed.artifact_id == (
        "relationship-action-gate-v2-artifact-sha256:2bcb447911d84ae05fbd882cbab8cc8c4e779357c28a713793778df3e36df165"
    )
    assert batch.schedule_artifact.artifact_id == (
        "relationship-action-gate-v2-assignment-schedule-sha256:"
        "10776f69e148c4391c0e4e761d45a32eb9fbfeb4bf2b168c29def75d7f393492"
    )
    assert batch.batch_id == (
        "relationship-action-gate-v2-credit-batch-sha256:"
        "64b7b6065f9a14963c847ea2cc67c23b924a31f5df63f76a8906043e33988e49"
    )
    assert plan.plan_id == (
        "relationship-action-gate-v2-batch-plan-sha256:74b52522a8b53dcd47e7c74198bcc94b95972018b18ad080d3dad5fc0b114773"
    )
    assert receipt.receipt_id == (
        "relationship-action-gate-v2-batch-receipt-sha256:"
        "432efb8696ec54223969b542c92340ae1487724d242d92f39f1d00ef8a3c4ea5"
    )


def test_batch_rejects_legacy_credit_partial_schedule_and_mutable_shapes() -> None:
    seed = _seed()
    gate = RelationshipActionGateV2(artifact=seed)
    batch = _batch(gate, _informative_specs())

    with pytest.raises(TypeError, match="common-baseline credits"):
        RelationshipActionGateV2CreditBatch(
            exposures=batch.exposures,
            credits=tuple(  # type: ignore[arg-type]
                item.parent_action_credit for item in batch.credits
            ),
        )
    with pytest.raises(ValueError, match="complete schedule"):
        RelationshipActionGateV2CreditBatch(
            exposures=batch.exposures[:1],
            credits=batch.credits[:1],
        )
    with pytest.raises(ValueError, match="forecast lineage"):
        RelationshipActionGateV2CreditBatch(
            exposures=batch.exposures,
            credits=tuple(reversed(batch.credits)),
        )

    checkpoint = gate.export_checkpoint()
    with pytest.raises(ValueError, match="checkpoint weights"):
        replace(
            checkpoint,
            weights=list(checkpoint.weights),  # type: ignore[arg-type]
        )

    forecast = _forecast("mutable")
    mutable_candidate = replace(
        forecast.candidate_predictions[0],
        outcomes=list(forecast.candidate_predictions[0].outcomes),  # type: ignore[arg-type]
    )
    mutable_forecast = replace(
        forecast,
        candidate_predictions=(
            mutable_candidate,
            *forecast.candidate_predictions[1:],
        ),
    )
    with pytest.raises(TypeError, match="immutable canonical owner shape"):
        relationship_action_gate_v2_features(mutable_forecast)


def test_v1_module_contract_remains_separate() -> None:
    seed = _seed()
    assert seed.schema_version.endswith(".v2")
    assert len(seed.weights) == 4
    with pytest.raises(ValueError):
        RelationshipActionGateV2Artifact.from_payload(
            {**seed.to_payload(), "schema_version": "relationship-action-gate.v1"}
        )


def test_online_v2_two_apply_steps_chain_and_payload_replay() -> None:
    theta0 = _learned_theta0()
    session = RelationshipActionGateV2OnlineSession(
        artifact=theta0,
        disposition=RelationshipActionGateBatchDisposition.APPLY,
    )
    initial = session.export_checkpoint()

    first = _online_step(
        session,
        suffix="online-apply-0",
        outcome=DialogueExternalOutcomeKind.HELPED,
        timestamp_ms=7000,
    )
    second = _online_step(
        session,
        suffix="online-apply-1",
        outcome=DialogueExternalOutcomeKind.OVER_DIRECTIVE,
        timestamp_ms=7001,
    )

    assert first.plan.pre_checkpoint == initial
    assert second.plan.pre_checkpoint == first.terminal_checkpoint
    assert second.plan.exposure.frozen_decision.checkpoint_content_sha256 == (
        first.terminal_checkpoint.content_sha256
    )
    assert second.plan.exposure.frozen_decision.decision.update_count == 1
    assert session.export_checkpoint().update_count == 2
    assert len(session.export_checkpoint().processed_credit_ids) == 2
    assert first.plan.operator_id == RELATIONSHIP_ACTION_GATE_V2_ONLINE_OPERATOR_ID
    assert first.plan.objective_id == RELATIONSHIP_ACTION_GATE_V2_ONLINE_OBJECTIVE_ID
    assert first.receipt.credit_applied_to_gate is True
    assert first.receipt.evaluation_or_judge_feedback_received is False

    chain = session.export_transition_chain()
    restored_payload = RelationshipActionGateV2OnlineTransitionChain.from_payload(
        chain.to_payload(),
        artifact=theta0,
        full_common_credits=tuple(item.plan.credit for item in chain.transitions),
    )
    assert restored_payload == chain
    restored_session = RelationshipActionGateV2OnlineSession.from_transition_chain(chain)
    assert restored_session.export_checkpoint() == session.export_checkpoint()
    assert restored_session.export_transition_chain() == chain

    next_forecast = _forecast("online-advisory-after-two")
    next_decision = session.decide(next_forecast)
    next_exposure = session.record_exposure(
        next_forecast,
        delivered_action_id=next_decision.decision.selected_action_id,
    )
    advisory = temporal_action_advisory_from_gate_v2_online_exposure(
        next_exposure,
        session=session,
    )
    assert advisory.action_id == next_exposure.delivered_action_id
    assert advisory.active_authorized is False
    assert chain.chain_id in advisory.evidence_refs


def test_online_v2_formula_covers_steer_noop_and_parameter_cap() -> None:
    theta0 = _learned_theta0()

    for suffix, forecast, outcome, expected_indicator in (
        (
            "steer",
            _forecast("online-formula-steer"),
            DialogueExternalOutcomeKind.HELPED,
            1,
        ),
        (
            "noop",
            replace(
                _forecast("online-formula-noop"),
                confidence=0.2,
                candidate_predictions=(
                    _candidate(
                        RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
                        (0.05, 0.05, 0.45, 0.45),
                    ),
                    *_forecast(
                        "online-formula-noop"
                    ).candidate_predictions[1:],
                ),
            ),
            DialogueExternalOutcomeKind.MISSED,
            0,
        ),
    ):
        session = RelationshipActionGateV2OnlineSession(
            artifact=theta0,
            disposition=RelationshipActionGateBatchDisposition.APPLY,
        )
        decision = session.decide(forecast)
        exposure = session.record_exposure(
            forecast,
            delivered_action_id=decision.decision.selected_action_id,
        )
        credit = _common_credit(
            exposure,
            outcome=outcome,
            timestamp_ms=7500 + expected_indicator,
        )
        plan = session.plan_credit(exposure, credit)
        expected_scale = (
            theta0.online_learning_rate
            * credit.credit_value
            * (float(expected_indicator) - decision.decision.steer_probability)
        )
        expected_raw = tuple(
            weight + expected_scale * feature
            for weight, feature in zip(
                plan.pre_checkpoint.weights,
                decision.decision.features,
                strict=True,
            )
        )
        expected_weights = tuple(
            max(-theta0.max_abs_parameter, min(theta0.max_abs_parameter, value))
            for value in expected_raw
        )
        assert plan.actual_steer_indicator == expected_indicator, suffix
        assert float.fromhex(plan.gradient_scale_hex) == pytest.approx(expected_scale)
        assert plan.candidate_checkpoint.weights == pytest.approx(expected_weights)
        assert tuple(
            float.fromhex(value) for value in plan.candidate_weight_delta_hex
        ) == pytest.approx(
            tuple(
                after - before
                for before, after in zip(
                    plan.pre_checkpoint.weights,
                    expected_weights,
                    strict=True,
                )
            )
        )
        transition = session.commit_credit(plan)
        assert transition.receipt.candidate_nonzero_parameter_update_count == int(
            plan.candidate_nonzero_parameter_delta
        )
        assert transition.receipt.applied_nonzero_parameter_update_count == int(
            plan.candidate_nonzero_parameter_delta
        )

    capped = RelationshipActionGateV2OnlineSession(
        artifact=_learned_theta0(
            online_learning_rate=0.5,
            max_abs_parameter=0.5,
        ),
        disposition=RelationshipActionGateBatchDisposition.APPLY,
    )
    cap_transition = None
    for index in range(200):
        forecast = _forecast(f"online-cap-{index}")
        decision = capped.decide(forecast)
        exposure = capped.record_exposure(
            forecast,
            delivered_action_id=decision.decision.selected_action_id,
        )
        credit = _common_credit(
            exposure,
            outcome=DialogueExternalOutcomeKind.HELPED,
            timestamp_ms=7600 + index,
        )
        plan = capped.plan_credit(exposure, credit)
        cap_transition = capped.commit_credit(plan)
        if plan.candidate_cap_hit_count:
            break
    assert cap_transition is not None
    assert cap_transition.plan.candidate_cap_hit_count > 0
    assert cap_transition.receipt.candidate_cap_hit_count > 0
    assert cap_transition.receipt.applied_cap_hit_count == (
        cap_transition.receipt.candidate_cap_hit_count
    )
    assert max(abs(value) for value in cap_transition.terminal_checkpoint.weights) == 0.5

    frozen_at_cap = RelationshipActionGateV2OnlineSession(
        artifact=_learned_theta0(
            bootstrap_learning_rate=0.5,
            bootstrap_pair_count=8,
            online_learning_rate=0.5,
            max_abs_parameter=0.5,
        ),
        disposition=RelationshipActionGateBatchDisposition.WITHHOLD,
    )
    frozen_forecast = _forecast("online-withhold-cap")
    frozen_decision = frozen_at_cap.decide(frozen_forecast)
    frozen_exposure = frozen_at_cap.record_exposure(
        frozen_forecast,
        delivered_action_id=frozen_decision.decision.selected_action_id,
    )
    frozen_credit = _common_credit(
        frozen_exposure,
        outcome=DialogueExternalOutcomeKind.HELPED,
        timestamp_ms=9000,
    )
    frozen_plan = frozen_at_cap.plan_credit(frozen_exposure, frozen_credit)
    frozen_transition = frozen_at_cap.commit_credit(frozen_plan)
    assert frozen_transition.receipt.candidate_cap_hit_count > 0
    assert frozen_transition.receipt.candidate_nonzero_parameter_update_count == 1
    assert frozen_transition.receipt.applied_cap_hit_count == 0
    assert frozen_transition.receipt.applied_nonzero_parameter_update_count == 0
    assert frozen_transition.terminal_checkpoint == frozen_plan.pre_checkpoint


def test_online_v2_forty_apply_vs_withhold_transitions_close_exactly() -> None:
    theta0 = _learned_theta0()
    full = RelationshipActionGateV2OnlineSession(
        artifact=theta0,
        disposition=RelationshipActionGateBatchDisposition.APPLY,
    )
    frozen = RelationshipActionGateV2OnlineSession(
        artifact=theta0,
        disposition=RelationshipActionGateBatchDisposition.WITHHOLD,
    )
    full_initial = full.export_checkpoint()
    frozen_initial = frozen.export_checkpoint()
    assert full_initial == frozen_initial
    first_forecast = _forecast("online-matched-initial")
    assert full.decide(first_forecast) == frozen.decide(first_forecast)

    full_transitions = []
    frozen_transitions = []
    outcomes = (
        DialogueExternalOutcomeKind.HELPED,
        DialogueExternalOutcomeKind.OVER_DIRECTIVE,
        DialogueExternalOutcomeKind.FELT_HEARD,
        DialogueExternalOutcomeKind.MISSED,
    )
    for index in range(40):
        forecast = _forecast(f"online-matched-{index}")
        full_transitions.append(
            _online_step_for_forecast(
                full,
                forecast=forecast,
                outcome=outcomes[index % len(outcomes)],
                timestamp_ms=8000 + index,
            )
        )
        frozen_transitions.append(
            _online_step_for_forecast(
                frozen,
                forecast=forecast,
                outcome=outcomes[index % len(outcomes)],
                timestamp_ms=8000 + index,
            )
        )
        full_transition = full_transitions[-1]
        frozen_transition = frozen_transitions[-1]
        assert full_transition.plan.exposure.forecast is forecast
        assert frozen_transition.plan.exposure.forecast is forecast
        if (
            full_transition.plan.exposure.delivered_action_id
            == frozen_transition.plan.exposure.delivered_action_id
        ):
            assert full_transition.plan.credit == frozen_transition.plan.credit

    assert full.export_checkpoint().update_count == 40
    assert len(full.export_checkpoint().processed_credit_ids) == 40
    assert sum(item.receipt.generated_credit_count for item in full_transitions) == 40
    assert sum(item.receipt.applied_credit_count for item in full_transitions) == 40
    assert sum(item.receipt.update_count_delta for item in full_transitions) == 40
    assert all(item.receipt.credit_applied_to_gate for item in full_transitions)
    assert all(
        item.receipt.applied_nonzero_parameter_update_count
        == item.receipt.candidate_nonzero_parameter_update_count
        for item in full_transitions
    )
    assert all(
        item.receipt.applied_cap_hit_count == item.receipt.candidate_cap_hit_count
        for item in full_transitions
    )
    assert frozen.export_checkpoint() == frozen_initial
    assert sum(item.receipt.generated_credit_count for item in frozen_transitions) == 40
    assert sum(item.receipt.applied_credit_count for item in frozen_transitions) == 0
    assert sum(item.receipt.update_count_delta for item in frozen_transitions) == 0
    assert all(not item.receipt.credit_applied_to_gate for item in frozen_transitions)
    assert any(
        item.receipt.candidate_nonzero_parameter_update_count
        for item in frozen_transitions
    )
    assert all(
        item.receipt.applied_nonzero_parameter_update_count == 0
        and item.receipt.applied_cap_hit_count == 0
        for item in frozen_transitions
    )
    assert all(
        item.receipt.post_checkpoint_content_sha256
        == item.receipt.pre_checkpoint_content_sha256
        for item in frozen_transitions
    )
    assert max(abs(value) for value in full.export_checkpoint().weights) <= (
        theta0.max_abs_parameter
    )
    full_chain = full.export_transition_chain()
    frozen_chain = frozen.export_transition_chain()
    assert full_chain.generated_credit_count == 40
    assert full_chain.applied_credit_count == 40
    assert full_chain.downstream_exposed_applied_update_count == 39
    assert frozen_chain.generated_credit_count == 40
    assert frozen_chain.applied_credit_count == 0
    assert frozen_chain.downstream_exposed_applied_update_count == 0
    for session, chain in ((full, full_chain), (frozen, frozen_chain)):
        restored_chain = RelationshipActionGateV2OnlineTransitionChain.from_payload(
            chain.to_payload(),
            artifact=theta0,
            full_common_credits=tuple(item.plan.credit for item in chain.transitions),
        )
        assert restored_chain == chain
        assert RelationshipActionGateV2OnlineSession.from_transition_chain(
            restored_chain
        ).export_checkpoint() == session.export_checkpoint()
    tampered = {
        **full_chain.to_payload(),
        "terminal_checkpoint": full_initial.to_payload(),
    }
    with pytest.raises(ValueError, match="terminal checkpoint projection"):
        RelationshipActionGateV2OnlineTransitionChain.from_payload(
            tampered,
            artifact=theta0,
            full_common_credits=tuple(
                item.plan.credit for item in full_chain.transitions
            ),
        )


def test_online_v2_zero_information_processes_credit_without_parameter_signal() -> None:
    session = RelationshipActionGateV2OnlineSession(
        artifact=_learned_theta0(),
        disposition=RelationshipActionGateBatchDisposition.APPLY,
    )
    forecast = _forecast(
        "online-zero-information",
        recommended_action_id=RelationshipAction.NEUTRAL_NOOP.value,
    )
    decision = session.decide(forecast)
    exposure = session.record_exposure(
        forecast,
        delivered_action_id=decision.decision.selected_action_id,
    )
    credit = _common_credit(
        exposure,
        outcome=DialogueExternalOutcomeKind.MISSED,
        timestamp_ms=9000,
    )
    plan = session.plan_credit(exposure, credit)

    assert plan.informative is False
    assert float.fromhex(plan.gradient_scale_hex) == 0.0
    assert plan.candidate_checkpoint.weights == plan.pre_checkpoint.weights
    assert plan.candidate_checkpoint.update_count == 1
    assert plan.candidate_checkpoint.informative_update_count == 0
    transition = session.commit_credit(plan)
    assert transition.receipt.update_count_delta == 1
    assert transition.receipt.informative_update_count_delta == 0


def test_online_v2_rejects_stale_reordered_duplicate_and_feedback_shapes() -> None:
    theta0 = _learned_theta0()
    session = RelationshipActionGateV2OnlineSession(
        artifact=theta0,
        disposition=RelationshipActionGateBatchDisposition.APPLY,
    )
    forecast = _forecast("online-reject-0")
    decision = session.decide(forecast)
    exposure = session.record_exposure(
        forecast,
        delivered_action_id=decision.decision.selected_action_id,
    )
    assert session.pending_exposure == exposure
    with pytest.raises(ValueError, match="pending settlement"):
        session.decide(_forecast("online-reject-premature-decision"))
    with pytest.raises(ValueError, match="another exposure is forbidden"):
        session.record_exposure(
            _forecast("online-reject-premature-exposure"),
            delivered_action_id=decision.decision.selected_action_id,
        )
    with pytest.raises(ValueError, match="cannot export"):
        session.export_transition_chain()
    forged_parent = replace(
        exposure,
        parent_chain_id=(
            "relationship-action-gate-v2-online-chain-sha256:" + "f" * 64
        ),
    )
    with pytest.raises(ValueError, match="parent chain differs"):
        temporal_action_advisory_from_gate_v2_online_exposure(
            forged_parent,
            session=session,
        )
    credit = _common_credit(
        exposure,
        outcome=DialogueExternalOutcomeKind.HELPED,
        timestamp_ms=10000,
    )
    with pytest.raises(TypeError, match="RelationshipActionCommonBaselineCredit"):
        session.plan_credit(  # type: ignore[arg-type]
            exposure,
            credit.parent_action_credit,
        )
    plan = session.plan_credit(exposure, credit)
    with pytest.raises(ValueError, match="already has a sealed credit plan"):
        session.plan_credit(exposure, credit)
    with pytest.raises(ValueError, match="evaluation or judge feedback"):
        replace(plan, evaluation_or_judge_feedback_received=True)
    wrong_action = (
        RelationshipAction.NEUTRAL_NOOP.value
        if exposure.delivered_action_id != RelationshipAction.NEUTRAL_NOOP.value
        else RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value
    )
    with pytest.raises(ValueError, match="exact learned gate action"):
        replace(exposure, delivered_action_id=wrong_action)

    first = session.commit_credit(plan)
    assert session.pending_exposure is None
    with pytest.raises(ValueError, match="pending recorded exposure"):
        session.plan_credit(exposure, credit)
    with pytest.raises(ValueError, match="sealed pending credit plan"):
        session.commit_credit(plan)
    with pytest.raises(ValueError, match="decision id was already consumed"):
        session.record_exposure(
            replace(forecast, forecast_id="online-reject-new-forecast-id"),
            delivered_action_id=decision.decision.selected_action_id,
        )
    with pytest.raises(ValueError, match="forecast id was already consumed"):
        session.record_exposure(
            replace(forecast, decision_id="online-reject-new-decision-id"),
            delivered_action_id=decision.decision.selected_action_id,
        )

    second_forecast = _forecast("online-reject-1")
    second_decision = session.decide(second_forecast)
    second_exposure = session.record_exposure(
        second_forecast,
        delivered_action_id=second_decision.decision.selected_action_id,
    )
    with pytest.raises(ValueError, match="already processed or withheld"):
        session.plan_credit(second_exposure, credit)
    second_credit = _common_credit(
        second_exposure,
        outcome=DialogueExternalOutcomeKind.MISSED,
        timestamp_ms=10001,
    )
    second = session.commit_credit(
        session.plan_credit(second_exposure, second_credit)
    )
    with pytest.raises(ValueError, match="disposition drifted"):
        RelationshipActionGateV2OnlineTransitionChain(
            artifact=theta0,
            disposition=RelationshipActionGateBatchDisposition.WITHHOLD,
            initial_checkpoint=RelationshipActionGateV2OnlineSession(
                artifact=theta0,
                disposition=RelationshipActionGateBatchDisposition.WITHHOLD,
            ).export_checkpoint(),
            transitions=(first,),
        )
    with pytest.raises(ValueError, match="sequence must be contiguous"):
        RelationshipActionGateV2OnlineTransitionChain(
            artifact=theta0,
            disposition=RelationshipActionGateBatchDisposition.APPLY,
            initial_checkpoint=RelationshipActionGateV2OnlineSession(
                artifact=theta0,
                disposition=RelationshipActionGateBatchDisposition.APPLY,
            ).export_checkpoint(),
            transitions=(second, first),
        )
    with pytest.raises(ValueError, match="sequence must be contiguous"):
        RelationshipActionGateV2OnlineTransitionChain(
            artifact=theta0,
            disposition=RelationshipActionGateBatchDisposition.APPLY,
            initial_checkpoint=RelationshipActionGateV2OnlineSession(
                artifact=theta0,
                disposition=RelationshipActionGateBatchDisposition.APPLY,
            ).export_checkpoint(),
            transitions=(first, first),
        )
    with pytest.raises(ValueError, match="non-cold v2 checkpoint"):
        RelationshipActionGateV2(
            artifact=theta0,
            checkpoint=session.export_checkpoint(),
        )


def test_online_v2_exact_types_and_receipts_round_trip() -> None:
    session = RelationshipActionGateV2OnlineSession(
        artifact=_learned_theta0(),
        disposition=RelationshipActionGateBatchDisposition.APPLY,
    )
    transition = _online_step(
        session,
        suffix="online-golden",
        outcome=DialogueExternalOutcomeKind.HELPED,
        timestamp_ms=12000,
    )

    assert transition.plan.exposure.exposure_id == (
        "relationship-action-gate-v2-online-exposure-sha256:"
        "3c2b4ca5ed884e9019b6e0fd9f5dc446bf99a6972bc1aa30d381761772d4707a"
    )
    assert transition.plan.exposure.frozen_decision.frozen_policy_id == (
        "relationship-action-gate-v2-online-policy-sha256:"
        "c79ceb278591d5152c385d4fc4cac9e6e196ab7ec237ae242618f96ac92f8745"
    )
    assert transition.plan.plan_id == (
        "relationship-action-gate-v2-online-plan-sha256:"
        "3b1d1d18cf1e4fd6174309a11dcb4e29aca079c677be45ee0aa0534840fbb354"
    )
    assert transition.receipt.receipt_id == (
        "relationship-action-gate-v2-online-receipt-sha256:"
        "a4a92697f1b54870c5dfcc4af6c36c6952c89d1b28009d698d7547b4f46fd0cd"
    )
    assert transition.transition_id == (
        "relationship-action-gate-v2-online-transition-sha256:"
        "390ff8933acfb5e65a5f27be0e8c09c615c9d5e31bfcf53f495a1f452bdce008"
    )
    assert session.export_transition_chain().chain_id == (
        "relationship-action-gate-v2-online-chain-sha256:"
        "6ed42db5f37bff02b5402ecec42cd8aac4937f052a28e8d101da36fbd3de2d2b"
    )

    assert RelationshipActionGateV2OnlineExposure.from_payload(
        transition.plan.exposure.to_payload()
    ) == transition.plan.exposure
    assert RelationshipActionGateV2OnlinePlan.from_payload(
        transition.plan.to_payload(),
        artifact=session.artifact,
        full_common_credit=transition.plan.credit,
    ) == transition.plan
    assert RelationshipActionGateV2OnlineReceipt.from_payload(
        transition.receipt.to_payload()
    ) == transition.receipt
    assert RelationshipActionGateV2OnlineTransition.from_payload(
        transition.to_payload(),
        artifact=session.artifact,
        full_common_credit=transition.plan.credit,
    ) == transition
