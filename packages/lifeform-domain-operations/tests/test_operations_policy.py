from __future__ import annotations

from dataclasses import replace

import pytest

from lifeform_domain_operations import (
    OperationsBrainController,
    OperationsBenchmarkArmResult,
    OperationsContextPackSnapshot,
    OperationsContextRequest,
    OperationsCostBreakdown,
    OperationsDecisionKind,
    OperationsDependencyState,
    OperationsDivisionState,
    OperationsEvidenceClass,
    OperationsEvidenceRef,
    OperationsEvidenceRole,
    OperationsExecutionOutcome,
    OperationsGoalState,
    OperationsIncidentSeverity,
    OperationsIncidentState,
    OperationsObjectiveResult,
    OperationsOutcomeKind,
    OperationsOutcomeReport,
    OperationsOutcomeVerdict,
    OperationsPolicyCheckpoint,
    OperationsPolicyActivationReceipt,
    OperationsPolicyAction,
    OperationsPolicyBenchmarkReport,
    OperationsPolicyDecision,
    OperationsPolicyMode,
    OperationsPromotionReview,
    OperationsRecentOutcomeState,
    OperationsReversibility,
    OperationsRiskLevel,
    OperationsStateSnapshot,
    OperationsWorkItemState,
    OperationsWorkItemStatus,
    build_operations_lifeform,
    issue_operations_policy_activation,
    review_operations_policy_promotion,
    run_operations_policy_benchmark,
    validate_operations_policy_activation,
)
from lifeform_domain_operations.operations_policy import (
    OPERATIONS_POLICY_NOOP_CANDIDATE_ID,
    OperationsPolicy,
    default_operations_policy_checkpoint,
    settle_operations_policy_credit,
)
from volvence_zero.credit import CreditRecord, CreditSnapshot, GateDecision
from volvence_zero.memory import Track
from volvence_zero.prediction import (
    ActualOutcome,
    PredictedOutcome,
    PredictionActionContext,
    PredictionError,
    PredictionErrorSnapshot,
)
from volvence_zero.runtime import WiringLevel


def _evidence(ref_id: str, digest: str) -> OperationsEvidenceRef:
    return OperationsEvidenceRef(
        ref_id=ref_id,
        evidence_class=OperationsEvidenceClass.FIELD,
        role=OperationsEvidenceRole.OPERATING_SIGNAL,
        locator=f"autocompany://evidence/{ref_id}",
        content_sha256=digest * 64,
        observed_at_ms=1_000,
    )


def _request() -> OperationsContextRequest:
    evidence = (
        _evidence("division-evidence", "a"),
        _evidence("goal-evidence", "b"),
        _evidence("work-evidence", "c"),
        _evidence("dependency-evidence", "d"),
        _evidence("incident-evidence", "e"),
        _evidence("outcome-evidence", "f"),
    )
    state = OperationsStateSnapshot.create(
        as_of_ms=1_000,
        currency="USD",
        divisions=(
            OperationsDivisionState(
                division_id="engineering",
                health=0.25,
                available_human_minutes=120,
                committed_human_minutes=180,
                queue_depth=9,
                sla_breach_probability=0.85,
                budget_remaining_minor=2_000,
                cost_to_date_minor=8_000,
                evidence_ref_ids=("division-evidence",),
            ),
        ),
        goals=(
            OperationsGoalState(
                goal_id="goal-reliability",
                division_id="engineering",
                progress=0.2,
                target_progress=0.8,
                weight=1.0,
                deadline_ms=8_000,
                evidence_ref_ids=("goal-evidence",),
            ),
        ),
        work_items=(
            OperationsWorkItemState(
                work_item_id="work-upstream",
                division_id="engineering",
                action_catalog_id="catalog:operate-engineering",
                status=OperationsWorkItemStatus.IN_PROGRESS,
                progress=0.4,
                priority=0.8,
                deadline_ms=3_000,
                required_human_minutes=100,
                expected_cost_minor=1_500,
                dependency_ids=(),
                evidence_ref_ids=("work-evidence",),
            ),
            OperationsWorkItemState(
                work_item_id="work-repair",
                division_id="engineering",
                action_catalog_id="catalog:operate-engineering",
                status=OperationsWorkItemStatus.BLOCKED,
                progress=0.1,
                priority=1.0,
                deadline_ms=2_000,
                required_human_minutes=90,
                expected_cost_minor=1_000,
                dependency_ids=("dependency-1",),
                evidence_ref_ids=("work-evidence",),
            ),
        ),
        dependencies=(
            OperationsDependencyState(
                dependency_id="dependency-1",
                predecessor_work_item_id="work-upstream",
                successor_work_item_id="work-repair",
                resolved=False,
                criticality=0.95,
                evidence_ref_ids=("dependency-evidence",),
            ),
        ),
        incidents=(
            OperationsIncidentState(
                incident_id="incident-1",
                division_id="engineering",
                severity=OperationsIncidentSeverity.HIGH,
                open=True,
                started_at_ms=500,
                sla_deadline_ms=2_500,
                estimated_recovery_minutes=90,
                evidence_ref_ids=("incident-evidence",),
            ),
        ),
        recent_outcomes=(
            OperationsRecentOutcomeState(
                outcome_id="prior-outcome",
                division_id="engineering",
                candidate_id="prior-candidate",
                utility=-0.5,
                observed_at_ms=900,
                evidence_ref_ids=("outcome-evidence",),
            ),
        ),
    )
    return OperationsContextRequest.from_json(
        {
            "schema_version": "operations-context-request.v2",
            "request_id": "request-v2",
            "company_id": "company-1",
            "cycle_id": "cycle-1",
            "workstream_id": "engineering",
            "decision_id": "decision-1",
            "decision_point": "incident_recovery",
            "division_ids": ["engineering"],
            "action_catalog_ids": ["catalog:operate-engineering"],
            "confirmed_facts": [
                {
                    "fact_id": "fact-health",
                    "kind": "division_health",
                    "division_id": "engineering",
                    "statement": "Typed owner health observation.",
                    "evidence_ref_ids": ["division-evidence"],
                    "as_of_ms": 1_000,
                }
            ],
            "constraints": [
                {
                    "constraint_id": "authority",
                    "kind": "authority",
                    "division_id": "",
                    "description": "AutoCompany retains action authority.",
                    "hard": True,
                }
            ],
            "operating_window": {
                "currency": "USD",
                "maximum_external_cost_minor": 2_000,
                "maximum_human_minutes": 120,
                "starts_at_ms": 1_000,
                "ends_at_ms": 10_000,
                "maximum_work_orders": 4,
            },
            "uncertainties": [],
            "evidence_refs": [item.to_json() for item in evidence],
            "operations_state": state.to_json(),
        }
    )


def _field_report(
    *,
    pack: OperationsContextPackSnapshot,
    suffix: str,
    policy_action_applied: bool,
) -> OperationsOutcomeReport:
    decision = pack.advice.policy_decision
    assert decision is not None
    work_order_ref = f"autocompany://work-orders/{suffix}"
    work_order_evidence = OperationsEvidenceRef(
        ref_id=f"work-order-{suffix}",
        evidence_class=OperationsEvidenceClass.FIELD,
        role=OperationsEvidenceRole.WORK_ORDER,
        locator=work_order_ref,
        content_sha256="1" * 64,
        observed_at_ms=3_000,
    )
    objective_evidence = OperationsEvidenceRef(
        ref_id=f"objective-{suffix}",
        evidence_class=OperationsEvidenceClass.FIELD,
        role=OperationsEvidenceRole.OBJECTIVE_PROGRESS,
        locator=f"autocompany://objectives/{suffix}",
        content_sha256="2" * 64,
        observed_at_ms=3_000,
    )
    return OperationsOutcomeReport(
        outcome_id=f"outcome-{suffix}",
        context_pack_id=pack.context_pack_id,
        decision_id=pack.request.decision_id,
        work_order_ref=work_order_ref,
        decision=(
            OperationsDecisionKind.ACCEPT
            if policy_action_applied
            else OperationsDecisionKind.REJECT
        ),
        outcome_kind=OperationsOutcomeKind.FIELD_OPERATION_RESULT,
        evidence_class=OperationsEvidenceClass.FIELD,
        verdict=(
            OperationsOutcomeVerdict.FAVORABLE
            if policy_action_applied
            else OperationsOutcomeVerdict.MIXED
        ),
        summary="Typed field aggregate.",
        detail="AutoCompany-owned multi-objective operating result.",
        observed_at_ms=3_000,
        evidence_refs=(work_order_evidence, objective_evidence),
        execution_outcome=OperationsExecutionOutcome(
            objective_result=(
                OperationsObjectiveResult.ADVANCED
                if policy_action_applied
                else OperationsObjectiveResult.STALLED
            ),
            metrics=(),
            currency="USD",
            realized_costs=OperationsCostBreakdown(),
            elapsed_ms=1_000,
            blocker_duration_ms=0,
            rework_count=0,
            incident_count=0,
            human_minutes=0,
            risk_level=OperationsRiskLevel.LOW,
            reversibility=OperationsReversibility.REVERSIBLE,
        ),
        source_advice_id=pack.advice.advice_id,
        policy_decision_id=decision.policy_decision_id,
        selection_id=f"selection-{suffix}",
        selected_candidate_id=(
            decision.selected_candidate_id if policy_action_applied else ""
        ),
        activation_receipt_id=pack.advice.activation_receipt_id,
        selection_wiring_level=pack.advice.wiring_level,
        policy_action_applied=policy_action_applied,
        candidate_applied=policy_action_applied,
    )


def _pe_and_credit(
    *,
    environment_outcome_id: str,
    signed_reward: float,
) -> tuple[PredictionErrorSnapshot, CreditSnapshot]:
    action_context = PredictionActionContext(
        prediction_id="prediction-1",
        environment_outcome_id=environment_outcome_id,
        abstract_action_id="operations",
    )
    predicted = PredictedOutcome(
        source_turn_index=1,
        target_turn_index=2,
        predicted_task_progress=0.5,
        predicted_relationship_delta=0.5,
        predicted_regime_stability=0.5,
        predicted_action_payoff=0.0,
        confidence=1.0,
        description="owner prediction",
        action_context=action_context,
        prediction_id="prediction-1",
    )
    actual = ActualOutcome(
        observed_turn_index=2,
        task_progress=0.8,
        relationship_delta=0.5,
        regime_stability=0.6,
        action_payoff=signed_reward,
        description="owner actual",
        action_context=action_context,
    )
    error = PredictionError(
        task_error=0.3,
        relationship_error=0.0,
        regime_error=0.1,
        action_error=signed_reward,
        magnitude=abs(signed_reward),
        signed_reward=signed_reward,
        description="owner mismatch",
    )
    snapshot = PredictionErrorSnapshot(
        evaluated_prediction=predicted,
        actual_outcome=actual,
        next_prediction=replace(predicted, source_turn_index=2, target_turn_index=3),
        error=error,
        turn_index=2,
        bootstrap=False,
        description="settled",
        action_context=action_context,
    )
    credits = tuple(
        CreditRecord(
            record_id=f"credit-{source}",
            level="prediction_error",
            track=Track.SHARED,
            source_event=source,
            credit_value=0.5,
            context="owner PE credit",
            timestamp_ms=2_000 + index,
            prediction_id="prediction-1",
            environment_outcome_id=environment_outcome_id,
            abstract_action_id="operations",
        )
        for index, source in enumerate(
            ("pe:task", "pe:relationship", "pe:regime", "pe:action")
        )
    )
    return snapshot, CreditSnapshot(
        recent_credits=credits,
        recent_modifications=(),
        cumulative_credit_by_level=(),
    )


def test_rich_state_and_policy_contracts_round_trip_and_publish_nonempty_shadow() -> None:
    request = _request()
    assert request.operations_state is not None
    assert OperationsStateSnapshot.from_json(
        request.operations_state.to_json()
    ) == request.operations_state

    checkpoint = default_operations_policy_checkpoint()
    assert OperationsPolicyCheckpoint.from_json(checkpoint.to_json()) == checkpoint
    candidates, decision = OperationsPolicy().decide(
        request=request,
        recalled_experiences=(),
        source_prediction_id="prediction-round-trip",
        checkpoint=checkpoint,
    )

    assert candidates
    assert len(candidates) == request.operating_window.maximum_work_orders
    assert tuple(item.candidate_id for item in candidates) == tuple(
        item.candidate_id for item in decision.ranked_candidates
    )
    assert decision.checkpoint_id == checkpoint.checkpoint_id
    assert decision.state_snapshot_id == request.operations_state.state_snapshot_id
    assert OperationsPolicyDecision.from_json(decision.to_json()) == decision
    assert all(candidate.evidence_ref_ids for candidate in candidates)


async def test_default_controller_publishes_nonempty_v2_shadow_policy() -> None:
    controller = OperationsBrainController()
    session = build_operations_lifeform().create_session(
        session_id="operations-policy-v2",
    )
    pack = await controller.build_context_pack(
        session=session,
        request=_request(),
        generated_at_ms=1_000,
    )

    assert pack.to_json()["schema_version"] == "operations-context-pack.v2"
    assert pack.advice.to_json()["schema_version"] == "operations-advice.v2"
    assert pack.advice.candidates
    assert pack.advice.policy_decision is not None
    assert pack.advice.applied is False
    assert OperationsContextPackSnapshot.from_json(pack.to_json()) == pack


def test_policy_ranking_and_intervention_update_only_from_exact_pe_credit() -> None:
    policy = OperationsPolicy()
    request = _request()
    checkpoint = default_operations_policy_checkpoint()
    candidates, decision = policy.decide(
        request=request,
        recalled_experiences=(),
        source_prediction_id="prediction-1",
        checkpoint=checkpoint,
    )
    assert decision.selected_candidate_id
    pe, credit_snapshot = _pe_and_credit(
        environment_outcome_id="environment-outcome-1",
        signed_reward=0.8,
    )
    credit = settle_operations_policy_credit(
        prediction_error_snapshot=pe,
        credit_snapshot=credit_snapshot,
        policy_decision_id=decision.policy_decision_id,
        selection_id="selection-1",
        candidate_id=decision.selected_candidate_id,
        environment_outcome_id="environment-outcome-1",
    )
    updated, receipt = policy.observe_credit(
        checkpoint=checkpoint,
        decision=decision,
        candidates=candidates,
        credit=credit,
    )

    assert updated.update_count == 1
    assert updated.checkpoint_id != checkpoint.checkpoint_id
    assert receipt.credit_id == credit.credit_id
    assert receipt.parameter_delta_l2 > 0.0
    with pytest.raises(ValueError, match="already processed"):
        policy.observe_credit(
            checkpoint=updated,
            decision=replace(decision, checkpoint_id=updated.checkpoint_id),
            candidates=candidates,
            credit=credit,
        )

    missing_axis = replace(
        credit_snapshot,
        recent_credits=credit_snapshot.recent_credits[:-1],
    )
    with pytest.raises(ValueError, match="exactly four"):
        settle_operations_policy_credit(
            prediction_error_snapshot=pe,
            credit_snapshot=missing_axis,
            policy_decision_id=decision.policy_decision_id,
            selection_id="selection-1",
            candidate_id=decision.selected_candidate_id,
            environment_outcome_id="environment-outcome-1",
        )


def test_noop_outcome_updates_timing_without_mutating_candidate_ranking() -> None:
    policy = OperationsPolicy()
    request = _request()
    base = default_operations_policy_checkpoint()
    checkpoint = OperationsPolicyCheckpoint.create(
        artifact_id=base.artifact_id,
        action_weights=base.action_weights,
        intervention_weights=(0.0,) * len(base.intervention_weights),
        intervention_bias=-4.0,
        learning_rate=base.learning_rate,
        max_abs_parameter=base.max_abs_parameter,
    )
    candidates, decision = policy.decide(
        request=request,
        recalled_experiences=(),
        source_prediction_id="prediction-1",
        checkpoint=checkpoint,
        mode=OperationsPolicyMode.LEARNED,
    )
    assert decision.selected_candidate_id == ""
    pe, credit_snapshot = _pe_and_credit(
        environment_outcome_id="environment-noop",
        signed_reward=0.7,
    )
    credit = settle_operations_policy_credit(
        prediction_error_snapshot=pe,
        credit_snapshot=credit_snapshot,
        policy_decision_id=decision.policy_decision_id,
        selection_id="selection-noop",
        candidate_id=OPERATIONS_POLICY_NOOP_CANDIDATE_ID,
        environment_outcome_id="environment-noop",
    )
    updated, _ = policy.observe_credit(
        checkpoint=checkpoint,
        decision=decision,
        candidates=candidates,
        credit=credit,
    )
    assert updated.action_weights == checkpoint.action_weights
    assert (
        updated.intervention_weights != checkpoint.intervention_weights
        or updated.intervention_bias != checkpoint.intervention_bias
    )


async def test_multicycle_uplift_gate_and_staging_activation_are_exact() -> None:
    report, checkpoint = run_operations_policy_benchmark()
    learned = next(item for item in report.arms if item.arm_id == report.learned_arm)
    baselines = tuple(item for item in report.arms if item.arm_id != report.learned_arm)

    assert learned.training_cycles >= 120
    assert len(learned.evaluation_utilities) >= 60
    assert report.evidence_scope == "deterministic_simulation"
    assert report.validation_delta >= 0.05
    assert report.paired_delta_lower_95 > 0.0
    assert learned.mean_utility > max(item.mean_utility for item in baselines)
    assert learned.correct_action_rate >= 0.75
    assert learned.favorable_rate >= 0.75
    assert learned.policy_update_count == learned.pe_credit_count
    assert OperationsPolicyBenchmarkReport.from_json(report.to_json()) == report

    review = review_operations_policy_promotion(
        report=report,
        candidate_checkpoint=checkpoint,
    )
    assert review.decision is GateDecision.ALLOW
    assert OperationsPromotionReview.from_json(review.to_json()) == review
    receipt = issue_operations_policy_activation(
        review=review,
        report=report,
        candidate_checkpoint=checkpoint,
        issued_at_ms=2_026_083_000_000,
    )
    assert receipt.activation_scope == "autocompany_staging"
    assert receipt.authorizes(checkpoint)
    descendant = OperationsPolicyCheckpoint.create(
        artifact_id=checkpoint.artifact_id,
        action_weights=checkpoint.action_weights,
        intervention_weights=checkpoint.intervention_weights,
        intervention_bias=checkpoint.intervention_bias,
        learning_rate=checkpoint.learning_rate,
        max_abs_parameter=checkpoint.max_abs_parameter,
        update_count=checkpoint.update_count + 1,
        processed_credit_ids=(
            *checkpoint.processed_credit_ids,
            "post-promotion-credit",
        ),
    )
    assert receipt.authorizes(descendant)
    divergent = OperationsPolicyCheckpoint.create(
        artifact_id=checkpoint.artifact_id,
        action_weights=checkpoint.action_weights,
        intervention_weights=checkpoint.intervention_weights,
        intervention_bias=checkpoint.intervention_bias,
        learning_rate=checkpoint.learning_rate,
        max_abs_parameter=checkpoint.max_abs_parameter,
        update_count=checkpoint.update_count + 1,
        processed_credit_ids=(
            "divergent-credit",
            *checkpoint.processed_credit_ids[1:],
            "post-promotion-credit",
        ),
    )
    assert not receipt.authorizes(divergent)
    assert OperationsPolicyActivationReceipt.from_json(receipt.to_json()) == receipt
    validate_operations_policy_activation(
        report=report,
        review=review,
        receipt=receipt,
        candidate_checkpoint=checkpoint,
    )
    weak_learned = OperationsBenchmarkArmResult.create(
        arm_id=learned.arm_id,
        training_cycles=learned.training_cycles,
        evaluation_utilities=learned.evaluation_utilities,
        correct_action_count=0,
        intervention_count=round(
            learned.intervention_rate * len(learned.evaluation_utilities)
        ),
        policy_update_count=learned.policy_update_count,
        pe_credit_count=learned.pe_credit_count,
    )
    weak_report = OperationsPolicyBenchmarkReport.create(
        preregistration_sha256=report.preregistration_sha256,
        scenario_set_sha256=report.scenario_set_sha256,
        seed=report.seed,
        arms=tuple(
            weak_learned if item.arm_id == learned.arm_id else item
            for item in report.arms
        ),
        primary_baseline_arm=report.primary_baseline_arm,
        learned_arm=report.learned_arm,
        paired_delta_lower_95=report.paired_delta_lower_95,
        candidate_checkpoint=checkpoint,
        checkpoint_round_trip_verified=True,
        rollback_drill_verified=True,
        exact_pe_credit_lineage_verified=True,
    )
    blocked = review_operations_policy_promotion(
        report=weak_report,
        candidate_checkpoint=checkpoint,
    )
    assert blocked.decision is GateDecision.BLOCK
    assert "correct_action_rate_below_0.75" in blocked.blocking_reasons

    with pytest.raises(ValueError, match="ModificationGate activation receipt"):
        OperationsBrainController(policy_wiring_level=WiringLevel.ACTIVE)

    session = build_operations_lifeform().create_session(
        session_id="operations-policy-promoted-staging",
    )
    await OperationsBrainController().build_context_pack(
        session=session,
        request=_request(),
        generated_at_ms=1_000,
    )
    controller = OperationsBrainController(
        policy_checkpoint_seed=checkpoint,
        policy_wiring_level=WiringLevel.ACTIVE,
        activation_receipt=receipt,
    )
    pack = await controller.build_context_pack(
        session=session,
        request=replace(
            _request(),
            request_id="request-v2-promoted",
            cycle_id="cycle-v2-promoted",
            decision_id="decision-v2-promoted",
        ),
        generated_at_ms=2_000,
    )
    assert pack.advice.wiring_level is WiringLevel.ACTIVE
    assert pack.advice.activation_receipt_id == receipt.activation_receipt_id
    assert pack.advice.applied is False
    assert pack.advice.policy_decision is not None
    assert pack.advice.policy_decision.action is OperationsPolicyAction.INTERVENE

    ignored_receipt, created = await controller.publish_outcome(
        session=session,
        report=_field_report(
            pack=pack,
            suffix="ignored-policy",
            policy_action_applied=False,
        ),
    )
    assert created
    assert ignored_receipt.source_advice_applied is False
    after_ignored = await controller.build_context_pack(
        session=session,
        request=replace(
            _request(),
            request_id="request-v2-after-ignored",
            cycle_id="cycle-v2-after-ignored",
            decision_id="decision-v2-after-ignored",
        ),
        generated_at_ms=3_000,
    )
    assert after_ignored.settled_policy_credits == ()
    assert after_ignored.policy_updates == ()
    assert after_ignored.advice.policy_decision is not None
    assert after_ignored.advice.policy_decision.checkpoint_update_count == 720

    applied_receipt, created = await controller.publish_outcome(
        session=session,
        report=_field_report(
            pack=after_ignored,
            suffix="applied-policy",
            policy_action_applied=True,
        ),
    )
    assert created
    assert applied_receipt.source_advice_applied is True
    after_applied = await controller.build_context_pack(
        session=session,
        request=replace(
            _request(),
            request_id="request-v2-after-applied",
            cycle_id="cycle-v2-after-applied",
            decision_id="decision-v2-after-applied",
        ),
        generated_at_ms=4_000,
    )
    assert len(after_applied.settled_policy_credits) == 1
    assert len(after_applied.policy_updates) == 1
    assert after_applied.policy_updates[0].update_count == 721
