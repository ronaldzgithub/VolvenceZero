"""Pre-registered multi-cycle benchmark for the Operations policy.

The benchmark is an evidence harness, not a learning owner. Only the learned
arm calls :func:`settle_operations_policy_credit`, and every update is joined
to the four records emitted by the Prediction Error credit owner. Held-out
utilities are readouts only and are never written back into a checkpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from volvence_zero.credit import (
    CreditSnapshot,
    derive_prediction_error_credit_records,
)
from volvence_zero.prediction import (
    ActualOutcome,
    PredictedOutcome,
    PredictionActionContext,
    PredictionErrorModule,
    PredictionErrorSnapshot,
)

from lifeform_domain_operations.operations_brain_contracts import (
    OPERATIONS_POLICY_FEATURE_ORDER,
    OperationsAdviceCandidate,
    OperationsAdviceKind,
    OperationsConstraint,
    OperationsConstraintKind,
    OperationsContextRequest,
    OperationsDecisionPoint,
    OperationsDependencyState,
    OperationsDivisionState,
    OperationsEvidenceClass,
    OperationsEvidenceRef,
    OperationsEvidenceRole,
    OperationsFact,
    OperationsFactKind,
    OperationsGoalState,
    OperationsIncidentSeverity,
    OperationsIncidentState,
    OperationsOperatingWindow,
    OperationsPolicyCheckpoint,
    OperationsPolicyMode,
    OperationsRecentOutcomeState,
    OperationsStateSnapshot,
    OperationsWorkItemState,
    OperationsWorkItemStatus,
    stable_content_sha256,
)
from lifeform_domain_operations.operations_policy import (
    OPERATIONS_POLICY_NOOP_CANDIDATE_ID,
    OperationsPolicy,
    default_operations_policy_checkpoint,
    settle_operations_policy_credit,
)
from lifeform_domain_operations.operations_promotion import (
    OperationsBenchmarkArmResult,
    OperationsPolicyBenchmarkReport,
)


OPERATIONS_BENCHMARK_SEED = 20_260_830
OPERATIONS_BENCHMARK_TRAINING_CYCLES = 720
OPERATIONS_BENCHMARK_EVALUATION_CYCLES = 120
OPERATIONS_BENCHMARK_PRIMARY_METRIC = "normalized_multiobjective_utility"

_NOOP_ARM = "noop"
_UNIFORM_ARM = "uniform_candidate"
_FROZEN_ARM = "frozen_theta0"
_LEARNED_ARM = "pe_credit_learned"


@dataclass(frozen=True)
class _BenchmarkScenario:
    scenario_id: str
    decision_point: OperationsDecisionPoint
    features: tuple[float, ...]
    optimal_action: OperationsAdviceKind | None

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id must not be empty")
        if len(self.features) != len(OPERATIONS_POLICY_FEATURE_ORDER):
            raise ValueError("benchmark scenario feature geometry drift")
        if any(not 0.0 <= value <= 1.0 for value in self.features):
            raise ValueError("benchmark scenario features must be in [0, 1]")
        if self.features[7] not in {0.0, 0.25, 0.5, 0.75, 1.0}:
            raise ValueError("incident pressure must map to a typed severity")

    def to_json(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "decision_point": self.decision_point.value,
            "features": [
                {"name": name, "value": value}
                for name, value in zip(
                    OPERATIONS_POLICY_FEATURE_ORDER,
                    self.features,
                    strict=True,
                )
            ],
            "optimal_action": (
                self.optimal_action.value
                if self.optimal_action is not None
                else "noop"
            ),
        }


_SCENARIOS = (
    _BenchmarkScenario(
        scenario_id="launch_reversibility_constraint",
        decision_point=OperationsDecisionPoint.CYCLE_PLANNING,
        features=(0.50, 0.90, 0.80, 0.40, 0.90, 0.60, 0.10, 0.00, 0.40, 0.30),
        optimal_action=OperationsAdviceKind.PAUSE_WORK,
    ),
    _BenchmarkScenario(
        scenario_id="cross_division_authority_boundary",
        decision_point=OperationsDecisionPoint.DEPENDENCY_RESOLUTION,
        features=(0.50, 0.50, 0.60, 0.60, 0.80, 0.50, 1.00, 0.25, 0.50, 0.80),
        optimal_action=OperationsAdviceKind.REQUEST_HUMAN,
    ),
    _BenchmarkScenario(
        scenario_id="capacity_churn_control",
        decision_point=OperationsDecisionPoint.CAPACITY_REBALANCE,
        features=(0.40, 0.50, 1.00, 1.00, 0.70, 0.50, 0.20, 0.25, 0.30, 0.40),
        optimal_action=OperationsAdviceKind.PRIORITIZE_WORK,
    ),
    _BenchmarkScenario(
        scenario_id="critical_incident_authority_boundary",
        decision_point=OperationsDecisionPoint.INCIDENT_RECOVERY,
        features=(0.70, 0.40, 0.50, 0.80, 0.80, 1.00, 0.30, 1.00, 0.30, 0.80),
        optimal_action=OperationsAdviceKind.REQUEST_HUMAN,
    ),
    _BenchmarkScenario(
        scenario_id="budget_exhaustion_guard",
        decision_point=OperationsDecisionPoint.OPERATING_REVIEW,
        features=(0.40, 0.30, 0.60, 0.90, 0.60, 0.70, 0.40, 0.50, 1.00, 0.80),
        optimal_action=OperationsAdviceKind.PAUSE_WORK,
    ),
    _BenchmarkScenario(
        scenario_id="stable_no_intervention",
        decision_point=OperationsDecisionPoint.OPERATING_REVIEW,
        features=(0.05, 0.05, 0.10, 0.10, 0.10, 0.05, 0.00, 0.00, 0.10, 0.10),
        optimal_action=None,
    ),
)


def operations_policy_benchmark_preregistration() -> dict[str, object]:
    """Return the immutable benchmark plan used before any arm executes."""

    return {
        "schema_version": "operations-policy-benchmark-preregistration.v1",
        "seed": OPERATIONS_BENCHMARK_SEED,
        "training_cycles": OPERATIONS_BENCHMARK_TRAINING_CYCLES,
        "evaluation_cycles": OPERATIONS_BENCHMARK_EVALUATION_CYCLES,
        "arms": [_NOOP_ARM, _UNIFORM_ARM, _FROZEN_ARM, _LEARNED_ARM],
        "primary_baseline_rule": "maximum held-out mean among non-learning arms",
        "primary_metric": OPERATIONS_BENCHMARK_PRIMARY_METRIC,
        "paired_interval": "mean_delta_minus_1.96_standard_errors",
        "promotion_thresholds": {
            "minimum_validation_delta": 0.05,
            "paired_delta_lower_95_must_be_positive": True,
            "minimum_training_cycles": 120,
            "minimum_evaluation_cycles": 60,
            "minimum_correct_action_rate": 0.75,
            "minimum_favorable_rate": 0.75,
        },
        "learning_input": "exactly pe:task, pe:relationship, pe:regime, pe:action",
        "evaluation_writeback_allowed": False,
        "production_default_changed": False,
    }


def operations_policy_benchmark_scenario_set() -> dict[str, object]:
    return {
        "schema_version": "operations-policy-benchmark-scenarios.v1",
        "feature_order": list(OPERATIONS_POLICY_FEATURE_ORDER),
        "scenarios": [scenario.to_json() for scenario in _SCENARIOS],
        "utility_contract": {
            "correct_candidate": 0.90,
            "incorrect_candidate": -0.50,
            "required_intervention_noop": -0.65,
            "correct_noop": 0.60,
            "unnecessary_intervention": -0.55,
            "shared_noise_bounds": [-0.04, 0.04],
        },
    }


def _evidence(
    *,
    scenario: _BenchmarkScenario,
    cycle_index: int,
    name: str,
    observed_at_ms: int,
) -> OperationsEvidenceRef:
    ref_id = f"benchmark-{scenario.scenario_id}-{cycle_index}-{name}"
    return OperationsEvidenceRef(
        ref_id=ref_id,
        evidence_class=OperationsEvidenceClass.SIMULATION,
        role=OperationsEvidenceRole.OPERATING_SIGNAL,
        locator=f"benchmark://operations/{scenario.scenario_id}/{cycle_index}/{name}",
        content_sha256=stable_content_sha256(
            {
                "scenario_id": scenario.scenario_id,
                "cycle_index": cycle_index,
                "name": name,
            }
        ),
        observed_at_ms=observed_at_ms,
    )


def _severity(pressure: float) -> OperationsIncidentSeverity:
    return {
        0.0: OperationsIncidentSeverity.LOW,
        0.25: OperationsIncidentSeverity.LOW,
        0.5: OperationsIncidentSeverity.MEDIUM,
        0.75: OperationsIncidentSeverity.HIGH,
        1.0: OperationsIncidentSeverity.CRITICAL,
    }[pressure]


def _request(
    *,
    scenario: _BenchmarkScenario,
    phase: str,
    cycle_index: int,
) -> OperationsContextRequest:
    as_of_ms = 1_000_000 + cycle_index * 20_000
    horizon = 10_000
    refs = tuple(
        _evidence(
            scenario=scenario,
            cycle_index=cycle_index,
            name=name,
            observed_at_ms=as_of_ms,
        )
        for name in (
            "division",
            "goal",
            "work",
            "dependency",
            "incident",
            "outcome",
        )
    )
    ref_ids = {item.locator.rsplit("/", maxsplit=1)[-1]: item.ref_id for item in refs}
    (
        health_deficit,
        goal_gap,
        queue_pressure,
        capacity_pressure,
        deadline_pressure,
        sla_pressure,
        dependency_pressure,
        incident_pressure,
        budget_pressure,
        recent_failure_pressure,
    ) = scenario.features
    dependency = OperationsDependencyState(
        dependency_id="benchmark-dependency",
        predecessor_work_item_id="benchmark-upstream",
        successor_work_item_id="benchmark-primary",
        resolved=dependency_pressure == 0.0,
        criticality=dependency_pressure,
        evidence_ref_ids=(ref_ids["dependency"],),
    )
    state = OperationsStateSnapshot.create(
        as_of_ms=as_of_ms,
        currency="USD",
        divisions=(
            OperationsDivisionState(
                division_id="benchmark-division",
                health=1.0 - health_deficit,
                available_human_minutes=100,
                committed_human_minutes=round(capacity_pressure * 100),
                queue_depth=round(queue_pressure * 10),
                sla_breach_probability=sla_pressure,
                budget_remaining_minor=round((1.0 - budget_pressure) * 10_000),
                cost_to_date_minor=round(budget_pressure * 10_000),
                evidence_ref_ids=(ref_ids["division"],),
            ),
        ),
        goals=(
            OperationsGoalState(
                goal_id="benchmark-goal",
                division_id="benchmark-division",
                progress=0.0,
                target_progress=goal_gap,
                weight=1.0,
                deadline_ms=as_of_ms + horizon,
                evidence_ref_ids=(ref_ids["goal"],),
            ),
        ),
        work_items=(
            OperationsWorkItemState(
                work_item_id="benchmark-upstream",
                division_id="benchmark-division",
                action_catalog_id="catalog:benchmark-operations",
                status=OperationsWorkItemStatus.IN_PROGRESS,
                progress=0.5,
                priority=0.5,
                deadline_ms=as_of_ms + horizon,
                required_human_minutes=50,
                expected_cost_minor=500,
                dependency_ids=(),
                evidence_ref_ids=(ref_ids["work"],),
            ),
            OperationsWorkItemState(
                work_item_id="benchmark-primary",
                division_id="benchmark-division",
                action_catalog_id="catalog:benchmark-operations",
                status=(
                    OperationsWorkItemStatus.BLOCKED
                    if dependency_pressure > 0.0
                    else OperationsWorkItemStatus.QUEUED
                ),
                progress=0.0,
                priority=1.0,
                deadline_ms=as_of_ms + round(
                    (1.0 - deadline_pressure) * horizon
                ),
                required_human_minutes=80,
                expected_cost_minor=1_000,
                dependency_ids=(dependency.dependency_id,),
                evidence_ref_ids=(ref_ids["work"],),
            ),
        ),
        dependencies=(dependency,),
        incidents=(
            OperationsIncidentState(
                incident_id="benchmark-incident",
                division_id="benchmark-division",
                severity=_severity(incident_pressure),
                open=incident_pressure > 0.0,
                started_at_ms=as_of_ms,
                sla_deadline_ms=as_of_ms + horizon,
                estimated_recovery_minutes=60,
                evidence_ref_ids=(ref_ids["incident"],),
            ),
        ),
        recent_outcomes=(
            OperationsRecentOutcomeState(
                outcome_id="benchmark-prior-outcome",
                division_id="benchmark-division",
                candidate_id="benchmark-prior-candidate",
                utility=1.0 - 2.0 * recent_failure_pressure,
                observed_at_ms=as_of_ms,
                evidence_ref_ids=(ref_ids["outcome"],),
            ),
        ),
    )
    return OperationsContextRequest(
        request_id=f"benchmark-{phase}-request-{cycle_index}",
        company_id="benchmark-company",
        cycle_id=f"benchmark-{phase}-cycle-{cycle_index}",
        workstream_id="benchmark-division",
        decision_id=f"benchmark-{phase}-decision-{cycle_index}",
        decision_point=scenario.decision_point,
        division_ids=("benchmark-division",),
        action_catalog_ids=("catalog:benchmark-operations",),
        confirmed_facts=(
            OperationsFact(
                fact_id=f"benchmark-fact-{cycle_index}",
                kind=OperationsFactKind.DIVISION_HEALTH,
                division_id="benchmark-division",
                statement="Frozen typed benchmark operating observation.",
                evidence_ref_ids=(ref_ids["division"],),
                as_of_ms=as_of_ms,
            ),
        ),
        constraints=(
            OperationsConstraint(
                constraint_id=f"benchmark-authority-{cycle_index}",
                kind=OperationsConstraintKind.AUTHORITY,
                division_id="",
                description="The benchmark environment retains action authority.",
                hard=True,
            ),
        ),
        operating_window=OperationsOperatingWindow(
            currency="USD",
            maximum_external_cost_minor=2_000,
            maximum_human_minutes=100,
            starts_at_ms=as_of_ms,
            ends_at_ms=as_of_ms + horizon,
            maximum_work_orders=6,
        ),
        uncertainties=(),
        evidence_refs=refs,
        operations_state=state,
    )


def _schedule(*, phase: str, cycle_count: int) -> tuple[_BenchmarkScenario, ...]:
    repeated = tuple(
        _SCENARIOS[index % len(_SCENARIOS)] for index in range(cycle_count)
    )
    return tuple(
        scenario
        for _, scenario in sorted(
            (
                (
                    stable_content_sha256(
                        {
                            "seed": OPERATIONS_BENCHMARK_SEED,
                            "phase": phase,
                            "cycle_index": index,
                            "scenario_id": scenario.scenario_id,
                        }
                    ),
                    scenario,
                )
                for index, scenario in enumerate(repeated)
            ),
            key=lambda item: item[0],
        )
    )


def _shared_noise(*, phase: str, cycle_index: int, scenario_id: str) -> float:
    digest = stable_content_sha256(
        {
            "seed": OPERATIONS_BENCHMARK_SEED,
            "phase": phase,
            "cycle_index": cycle_index,
            "scenario_id": scenario_id,
        }
    )
    unit = int(digest[:8], 16) / 0xFFFFFFFF
    return -0.04 + unit * 0.08


def _utility(
    *,
    scenario: _BenchmarkScenario,
    action: OperationsAdviceKind | None,
    phase: str,
    cycle_index: int,
) -> float:
    if scenario.optimal_action is None:
        base = 0.60 if action is None else -0.55
    elif action is None:
        base = -0.65
    elif action is scenario.optimal_action:
        base = 0.90
    else:
        base = -0.50
    return max(
        -1.0,
        min(
            1.0,
            base
            + _shared_noise(
                phase=phase,
                cycle_index=cycle_index,
                scenario_id=scenario.scenario_id,
            ),
        ),
    )


def _candidate_action(
    candidates: tuple[OperationsAdviceCandidate, ...],
    candidate_id: str,
) -> OperationsAdviceKind:
    return next(item.kind for item in candidates if item.candidate_id == candidate_id)


def _prediction_error_and_credit_snapshot(
    *,
    cycle_index: int,
    utility: float,
) -> tuple[PredictionErrorSnapshot, CreditSnapshot, str]:
    environment_outcome_id = f"benchmark-environment-outcome-{cycle_index}"
    prediction_id = f"benchmark-prediction-{cycle_index}"
    action_context = PredictionActionContext(
        prediction_id=prediction_id,
        environment_outcome_id=environment_outcome_id,
        abstract_action_id="operations-policy-benchmark",
    )
    predicted = PredictedOutcome(
        source_turn_index=cycle_index,
        target_turn_index=cycle_index + 1,
        predicted_task_progress=0.0,
        predicted_relationship_delta=0.0,
        predicted_regime_stability=0.0,
        predicted_action_payoff=0.0,
        confidence=1.0,
        description="Frozen zero-utility benchmark prediction.",
        action_context=action_context,
        prediction_id=prediction_id,
    )
    actual = ActualOutcome(
        observed_turn_index=cycle_index + 1,
        task_progress=utility,
        relationship_delta=0.0,
        regime_stability=0.0,
        action_payoff=utility,
        description="Typed benchmark environment outcome.",
        action_context=action_context,
    )
    error = PredictionErrorModule().compute_prediction_error(
        predicted=predicted,
        actual_outcome=actual,
    )
    prediction_error = PredictionErrorSnapshot(
        evaluated_prediction=predicted,
        actual_outcome=actual,
        next_prediction=PredictedOutcome(
            source_turn_index=cycle_index + 1,
            target_turn_index=cycle_index + 2,
            predicted_task_progress=0.0,
            predicted_relationship_delta=0.0,
            predicted_regime_stability=0.0,
            predicted_action_payoff=0.0,
            confidence=1.0,
            description="Next frozen benchmark prediction.",
            action_context=action_context,
            prediction_id=f"benchmark-prediction-{cycle_index + 1}",
        ),
        error=error,
        turn_index=cycle_index + 1,
        bootstrap=False,
        description="Settled benchmark prediction.",
        action_context=action_context,
    )
    records = derive_prediction_error_credit_records(
        prediction_error=error,
        timestamp_ms=2_000_000 + cycle_index,
        action_context=action_context,
    )
    return (
        prediction_error,
        CreditSnapshot(
            recent_credits=records,
            recent_modifications=(),
            cumulative_credit_by_level=(),
        ),
        environment_outcome_id,
    )


def _train_learned_policy() -> OperationsPolicyCheckpoint:
    policy = OperationsPolicy()
    checkpoint = default_operations_policy_checkpoint()
    for cycle_index, scenario in enumerate(
        _schedule(
            phase="training",
            cycle_count=OPERATIONS_BENCHMARK_TRAINING_CYCLES,
        )
    ):
        request = _request(
            scenario=scenario,
            phase="training",
            cycle_index=cycle_index,
        )
        candidates, decision = policy.decide(
            request=request,
            recalled_experiences=(),
            source_prediction_id=f"benchmark-prediction-{cycle_index}",
            checkpoint=checkpoint,
            mode=OperationsPolicyMode.LEARNED,
        )
        selected_candidate_id = (
            decision.selected_candidate_id
            or OPERATIONS_POLICY_NOOP_CANDIDATE_ID
        )
        action = (
            _candidate_action(candidates, decision.selected_candidate_id)
            if decision.selected_candidate_id
            else None
        )
        utility = _utility(
            scenario=scenario,
            action=action,
            phase="training",
            cycle_index=cycle_index,
        )
        prediction_error, credit_snapshot, environment_outcome_id = (
            _prediction_error_and_credit_snapshot(
                cycle_index=cycle_index,
                utility=utility,
            )
        )
        credit = settle_operations_policy_credit(
            prediction_error_snapshot=prediction_error,
            credit_snapshot=credit_snapshot,
            policy_decision_id=decision.policy_decision_id,
            selection_id=f"benchmark-training-selection-{cycle_index}",
            candidate_id=selected_candidate_id,
            environment_outcome_id=environment_outcome_id,
        )
        checkpoint, receipt = policy.observe_credit(
            checkpoint=checkpoint,
            decision=decision,
            candidates=candidates,
            credit=credit,
        )
        if receipt.update_count != cycle_index + 1:
            raise RuntimeError("Operations benchmark update sequence drift")
    return checkpoint


def _uniform_candidate_id(
    *,
    candidates: tuple[OperationsAdviceCandidate, ...],
    cycle_index: int,
) -> str:
    digest = stable_content_sha256(
        {
            "seed": OPERATIONS_BENCHMARK_SEED,
            "arm": _UNIFORM_ARM,
            "cycle_index": cycle_index,
        }
    )
    return candidates[int(digest[:8], 16) % len(candidates)].candidate_id


def _evaluate_arm(
    *,
    arm_id: str,
    checkpoint: OperationsPolicyCheckpoint,
    schedule: tuple[_BenchmarkScenario, ...],
) -> OperationsBenchmarkArmResult:
    policy = OperationsPolicy()
    utilities: list[float] = []
    correct = 0
    interventions = 0
    for cycle_index, scenario in enumerate(schedule):
        request = _request(
            scenario=scenario,
            phase="evaluation",
            cycle_index=cycle_index,
        )
        mode = (
            OperationsPolicyMode.NOOP
            if arm_id == _NOOP_ARM
            else OperationsPolicyMode.FROZEN
        )
        candidates, decision = policy.decide(
            request=request,
            recalled_experiences=(),
            source_prediction_id=f"benchmark-owner-evaluation-{cycle_index}",
            checkpoint=checkpoint,
            mode=mode,
        )
        if arm_id == _UNIFORM_ARM:
            candidate_id = _uniform_candidate_id(
                candidates=candidates,
                cycle_index=cycle_index,
            )
        else:
            candidate_id = decision.selected_candidate_id
        action = (
            _candidate_action(candidates, candidate_id)
            if candidate_id
            else None
        )
        utilities.append(
            _utility(
                scenario=scenario,
                action=action,
                phase="evaluation",
                cycle_index=cycle_index,
            )
        )
        interventions += int(action is not None)
        correct += int(action is scenario.optimal_action)
    return OperationsBenchmarkArmResult.create(
        arm_id=arm_id,
        training_cycles=(
            OPERATIONS_BENCHMARK_TRAINING_CYCLES
            if arm_id == _LEARNED_ARM
            else 0
        ),
        evaluation_utilities=tuple(utilities),
        correct_action_count=correct,
        intervention_count=interventions,
        policy_update_count=(
            checkpoint.update_count if arm_id == _LEARNED_ARM else 0
        ),
        pe_credit_count=(
            checkpoint.update_count if arm_id == _LEARNED_ARM else 0
        ),
    )


def _paired_lower_95(
    learned: tuple[float, ...],
    baseline: tuple[float, ...],
) -> float:
    if len(learned) != len(baseline) or len(learned) < 2:
        raise ValueError("paired interval requires equal non-trivial samples")
    differences = tuple(
        left - right for left, right in zip(learned, baseline, strict=True)
    )
    mean = math.fsum(differences) / len(differences)
    variance = math.fsum((value - mean) ** 2 for value in differences) / (
        len(differences) - 1
    )
    return mean - 1.96 * math.sqrt(variance / len(differences))


def _rollback_drill_verified(
    candidate_checkpoint: OperationsPolicyCheckpoint,
) -> bool:
    from volvence_zero.runtime import WiringLevel

    from lifeform_domain_operations.operations_brain import OperationsBrainController

    OperationsBrainController(
        policy_checkpoint_seed=candidate_checkpoint,
        policy_wiring_level=WiringLevel.SHADOW,
    )
    try:
        OperationsBrainController(
            policy_checkpoint_seed=candidate_checkpoint,
            policy_wiring_level=WiringLevel.ACTIVE,
        )
    except ValueError as exc:
        return "ModificationGate activation receipt" in str(exc)
    return False


def run_operations_policy_benchmark() -> tuple[
    OperationsPolicyBenchmarkReport,
    OperationsPolicyCheckpoint,
]:
    """Run all pre-registered arms and return frozen promotion evidence."""

    preregistration = operations_policy_benchmark_preregistration()
    scenario_set = operations_policy_benchmark_scenario_set()
    learned_checkpoint = _train_learned_policy()
    evaluation_schedule = _schedule(
        phase="evaluation",
        cycle_count=OPERATIONS_BENCHMARK_EVALUATION_CYCLES,
    )
    initial_checkpoint = default_operations_policy_checkpoint()
    arms = (
        _evaluate_arm(
            arm_id=_NOOP_ARM,
            checkpoint=initial_checkpoint,
            schedule=evaluation_schedule,
        ),
        _evaluate_arm(
            arm_id=_UNIFORM_ARM,
            checkpoint=initial_checkpoint,
            schedule=evaluation_schedule,
        ),
        _evaluate_arm(
            arm_id=_FROZEN_ARM,
            checkpoint=initial_checkpoint,
            schedule=evaluation_schedule,
        ),
        _evaluate_arm(
            arm_id=_LEARNED_ARM,
            checkpoint=learned_checkpoint,
            schedule=evaluation_schedule,
        ),
    )
    baselines = tuple(item for item in arms if item.arm_id != _LEARNED_ARM)
    primary_baseline = max(
        baselines,
        key=lambda item: (item.mean_utility, item.arm_id),
    )
    learned = next(item for item in arms if item.arm_id == _LEARNED_ARM)
    checkpoint_round_trip_verified = (
        OperationsPolicyCheckpoint.from_json(learned_checkpoint.to_json())
        == learned_checkpoint
    )
    report = OperationsPolicyBenchmarkReport.create(
        preregistration_sha256=stable_content_sha256(preregistration),
        scenario_set_sha256=stable_content_sha256(scenario_set),
        seed=OPERATIONS_BENCHMARK_SEED,
        arms=arms,
        primary_baseline_arm=primary_baseline.arm_id,
        learned_arm=_LEARNED_ARM,
        paired_delta_lower_95=_paired_lower_95(
            learned.evaluation_utilities,
            primary_baseline.evaluation_utilities,
        ),
        candidate_checkpoint=learned_checkpoint,
        checkpoint_round_trip_verified=checkpoint_round_trip_verified,
        rollback_drill_verified=_rollback_drill_verified(learned_checkpoint),
        exact_pe_credit_lineage_verified=(
            learned_checkpoint.update_count
            == OPERATIONS_BENCHMARK_TRAINING_CYCLES
            == len(learned_checkpoint.processed_credit_ids)
        ),
    )
    return report, learned_checkpoint


__all__ = (
    "OPERATIONS_BENCHMARK_EVALUATION_CYCLES",
    "OPERATIONS_BENCHMARK_PRIMARY_METRIC",
    "OPERATIONS_BENCHMARK_SEED",
    "OPERATIONS_BENCHMARK_TRAINING_CYCLES",
    "operations_policy_benchmark_preregistration",
    "operations_policy_benchmark_scenario_set",
    "run_operations_policy_benchmark",
)
