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
    relationship_action_gate_v2_features,
    temporal_action_advisory_from_gate_v2_decision,
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
            timestamp_ms=5000 + index,
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
