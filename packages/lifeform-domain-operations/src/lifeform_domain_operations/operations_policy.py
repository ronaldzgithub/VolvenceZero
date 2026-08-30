"""Bounded Operations policy learned only from exact PE/credit lineage.

The policy owns candidate ranking and intervention timing. AutoCompany still
owns selection, authority, work orders, execution, and outcome facts. Natural
language is never parsed here: every feature is reduced from the frozen typed
``OperationsStateSnapshot`` supplied by that owner.
"""

from __future__ import annotations

import math

from lifeform_core import (
    BoundedPolicyCandidate,
    BoundedPolicyDecision,
    BoundedPolicyRankedCandidate,
    apply_bounded_policy_credit,
    rank_and_gate_bounded_policy,
)

from volvence_zero.credit import CreditRecord, CreditSnapshot
from volvence_zero.prediction import PredictionErrorSnapshot

from lifeform_domain_operations.operations_brain_contracts import (
    OPERATIONS_POLICY_FEATURE_ORDER,
    OperationsAdviceCandidate,
    OperationsAdviceKind,
    OperationsContextRequest,
    OperationsEstimateRange,
    OperationsIncidentSeverity,
    OperationsPolicyAction,
    OperationsPolicyCheckpoint,
    OperationsPolicyCredit,
    OperationsPolicyDecision,
    OperationsPolicyMode,
    OperationsPolicyUpdateReceipt,
    OperationsRankedCandidate,
    OperationsRecalledExperience,
    OperationsReversibility,
    OperationsRiskLevel,
    OperationsWorkItemStatus,
    stable_content_sha256,
)


OPERATIONS_POLICY_ARTIFACT_ID = "operations-bounded-linear-policy.v1"
OPERATIONS_POLICY_NOOP_CANDIDATE_ID = "operations-policy:no-op"

_PE_CREDIT_SOURCES = (
    "pe:task",
    "pe:relationship",
    "pe:regime",
    "pe:action",
)

_INITIAL_ACTION_WEIGHTS: dict[OperationsAdviceKind, tuple[float, ...]] = {
    OperationsAdviceKind.PRIORITIZE_WORK: (
        0.35,
        0.90,
        0.85,
        0.20,
        0.55,
        0.30,
        0.15,
        0.10,
        -0.20,
        0.25,
    ),
    OperationsAdviceKind.SEQUENCE_DEPENDENCY: (
        0.20,
        0.35,
        0.30,
        0.20,
        0.55,
        0.25,
        1.10,
        0.15,
        -0.10,
        0.20,
    ),
    OperationsAdviceKind.REBALANCE_CAPACITY: (
        0.25,
        0.30,
        0.55,
        1.10,
        0.35,
        0.35,
        0.30,
        0.20,
        -0.15,
        0.25,
    ),
    OperationsAdviceKind.RECOVER_INCIDENT: (
        0.45,
        0.25,
        0.25,
        0.35,
        0.55,
        0.85,
        0.30,
        1.20,
        -0.10,
        0.35,
    ),
    OperationsAdviceKind.PAUSE_WORK: (
        0.20,
        0.10,
        0.10,
        0.45,
        0.20,
        0.35,
        0.30,
        0.55,
        0.55,
        0.45,
    ),
    OperationsAdviceKind.REQUEST_HUMAN: (
        0.25,
        0.20,
        0.25,
        0.55,
        0.30,
        0.55,
        0.45,
        0.80,
        0.20,
        0.45,
    ),
}

_INITIAL_INTERVENTION_WEIGHTS = (
    0.35,
    0.45,
    0.35,
    0.45,
    0.35,
    0.45,
    0.35,
    0.55,
    0.20,
    0.35,
)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def default_operations_policy_checkpoint() -> OperationsPolicyCheckpoint:
    """Return the reviewed, non-zero SHADOW theta0."""

    return OperationsPolicyCheckpoint.create(
        artifact_id=OPERATIONS_POLICY_ARTIFACT_ID,
        action_weights=tuple(
            (kind.value, _INITIAL_ACTION_WEIGHTS[kind])
            for kind in OperationsAdviceKind
        ),
        intervention_weights=_INITIAL_INTERVENTION_WEIGHTS,
        intervention_bias=-1.0,
        learning_rate=0.12,
        max_abs_parameter=4.0,
    )


def _severity_pressure(severity: OperationsIncidentSeverity) -> float:
    return {
        OperationsIncidentSeverity.LOW: 0.25,
        OperationsIncidentSeverity.MEDIUM: 0.50,
        OperationsIncidentSeverity.HIGH: 0.75,
        OperationsIncidentSeverity.CRITICAL: 1.0,
    }[severity]


def _risk_level_for_division(
    request: OperationsContextRequest,
    division_id: str,
) -> OperationsRiskLevel:
    state = request.operations_state
    if state is None:
        raise ValueError("Operations policy requires operations_state")
    severities = tuple(
        incident.severity
        for incident in state.incidents
        if incident.division_id == division_id and incident.open
    )
    if not severities:
        return OperationsRiskLevel.LOW
    maximum = max(_severity_pressure(item) for item in severities)
    if maximum >= 1.0:
        return OperationsRiskLevel.CRITICAL
    if maximum >= 0.75:
        return OperationsRiskLevel.HIGH
    if maximum >= 0.5:
        return OperationsRiskLevel.MEDIUM
    return OperationsRiskLevel.LOW


def _division_features(
    request: OperationsContextRequest,
    division_id: str,
) -> tuple[float, ...]:
    state = request.operations_state
    if state is None:
        raise ValueError("Operations policy requires operations_state")
    division = next(
        item for item in state.divisions if item.division_id == division_id
    )
    goals = tuple(item for item in state.goals if item.division_id == division_id)
    if goals:
        goal_gap = math.fsum(
            max(0.0, item.target_progress - item.progress) * item.weight
            for item in goals
        ) / math.fsum(item.weight for item in goals)
    else:
        goal_gap = 0.0
    work_items = tuple(
        item
        for item in state.work_items
        if item.division_id == division_id
        and item.status
        not in {OperationsWorkItemStatus.DONE, OperationsWorkItemStatus.CANCELLED}
    )
    if work_items:
        horizon = max(1, request.operating_window.ends_at_ms - state.as_of_ms)
        deadline_pressure = max(
            _clamp(
                1.0 - (item.deadline_ms - state.as_of_ms) / horizon,
            )
            for item in work_items
        )
    else:
        deadline_pressure = 0.0
    unresolved = tuple(item for item in state.dependencies if not item.resolved)
    relevant_work_item_ids = {item.work_item_id for item in work_items}
    relevant_dependencies = tuple(
        item
        for item in unresolved
        if item.predecessor_work_item_id in relevant_work_item_ids
        or item.successor_work_item_id in relevant_work_item_ids
    )
    dependency_pressure = max(
        (item.criticality for item in relevant_dependencies),
        default=0.0,
    )
    incident_pressure = max(
        (
            _severity_pressure(item.severity)
            for item in state.incidents
            if item.division_id == division_id and item.open
        ),
        default=0.0,
    )
    recent = tuple(
        item.utility
        for item in state.recent_outcomes
        if item.division_id == division_id
    )
    recent_failure_pressure = (
        math.fsum((1.0 - value) / 2.0 for value in recent) / len(recent)
        if recent
        else 0.5
    )
    capacity_pressure = (
        _clamp(
            division.committed_human_minutes
            / max(1, division.available_human_minutes)
        )
        if division.committed_human_minutes
        else 0.0
    )
    budget_total = division.budget_remaining_minor + division.cost_to_date_minor
    budget_pressure = (
        _clamp(division.cost_to_date_minor / budget_total)
        if budget_total
        else 0.0
    )
    features = (
        1.0 - division.health,
        _clamp(goal_gap),
        _clamp(division.queue_depth / 10.0),
        capacity_pressure,
        deadline_pressure,
        division.sla_breach_probability,
        dependency_pressure,
        incident_pressure,
        budget_pressure,
        _clamp(recent_failure_pressure),
    )
    if len(features) != len(OPERATIONS_POLICY_FEATURE_ORDER):
        raise RuntimeError("Operations policy feature geometry drift")
    return features


def _evidence_for_division(
    request: OperationsContextRequest,
    division_id: str,
) -> tuple[str, ...]:
    state = request.operations_state
    if state is None:
        raise ValueError("Operations policy requires operations_state")
    references: list[str] = []
    for collection in (
        state.divisions,
        state.goals,
        state.work_items,
        state.incidents,
        state.recent_outcomes,
    ):
        references.extend(
            reference
            for item in collection
            if item.division_id == division_id
            for reference in item.evidence_ref_ids
        )
    relevant_work_ids = {
        item.work_item_id
        for item in state.work_items
        if item.division_id == division_id
    }
    references.extend(
        reference
        for dependency in state.dependencies
        if dependency.predecessor_work_item_id in relevant_work_ids
        or dependency.successor_work_item_id in relevant_work_ids
        for reference in dependency.evidence_ref_ids
    )
    if not references:
        references.extend(
            reference.ref_id
            for reference in request.evidence_refs
            if any(
                fact.division_id == division_id
                and reference.ref_id in fact.evidence_ref_ids
                for fact in request.confirmed_facts
            )
        )
    if not references:
        raise ValueError("Operations state division has no evidence lineage")
    return tuple(dict.fromkeys(references))


def _catalog_for_division(
    request: OperationsContextRequest,
    division_id: str,
) -> str:
    state = request.operations_state
    if state is None:
        raise ValueError("Operations policy requires operations_state")
    work_items = sorted(
        (
            item
            for item in state.work_items
            if item.division_id == division_id
            and item.status
            not in {OperationsWorkItemStatus.DONE, OperationsWorkItemStatus.CANCELLED}
        ),
        key=lambda item: (-item.priority, item.work_item_id),
    )
    return (
        work_items[0].action_catalog_id
        if work_items
        else request.action_catalog_ids[0]
    )


def _resource_bounds(
    request: OperationsContextRequest,
    division_id: str,
) -> tuple[int, int]:
    state = request.operations_state
    if state is None:
        raise ValueError("Operations policy requires operations_state")
    relevant = tuple(
        item
        for item in state.work_items
        if item.division_id == division_id
        and item.status
        not in {OperationsWorkItemStatus.DONE, OperationsWorkItemStatus.CANCELLED}
    )
    estimated_cost = max((item.expected_cost_minor for item in relevant), default=0)
    estimated_human = max(
        (item.required_human_minutes for item in relevant),
        default=0,
    )
    return (
        min(request.operating_window.maximum_external_cost_minor, estimated_cost),
        min(request.operating_window.maximum_human_minutes, estimated_human),
    )


def _candidate(
    *,
    request: OperationsContextRequest,
    checkpoint: OperationsPolicyCheckpoint,
    division_id: str,
    kind: OperationsAdviceKind,
    features: tuple[float, ...],
    source_entry_ids: tuple[str, ...],
) -> OperationsAdviceCandidate:
    catalog_id = _catalog_for_division(request, division_id)
    evidence_ref_ids = _evidence_for_division(request, division_id)
    candidate_digest = stable_content_sha256(
        {
            "schema_version": "operations-policy-candidate.v1",
            "request_id": request.request_id,
            "state_snapshot_id": request.operations_state.state_snapshot_id,
            "checkpoint_id": checkpoint.checkpoint_id,
            "division_id": division_id,
            "kind": kind.value,
            "action_catalog_id": catalog_id,
        }
    )
    expected_utility = _clamp(
        math.fsum(features) / len(features),
        -1.0,
        1.0,
    )
    cost, human = _resource_bounds(request, division_id)
    risk = _risk_level_for_division(request, division_id)
    approval_required = risk in {
        OperationsRiskLevel.HIGH,
        OperationsRiskLevel.CRITICAL,
    } or kind is OperationsAdviceKind.REQUEST_HUMAN
    prerequisite_fact_ids = tuple(
        item.fact_id
        for item in request.confirmed_facts
        if not item.division_id or item.division_id == division_id
    )
    return OperationsAdviceCandidate(
        candidate_id=f"operations-candidate:{candidate_digest}",
        kind=kind,
        target_division_id=division_id,
        action_catalog_id=catalog_id,
        summary=(
            f"Evaluate {kind.value} for division {division_id} within the "
            "declared operating window."
        ),
        rationale=(
            "Ranked from typed OperationsStateSnapshot features by the "
            "bounded SHADOW policy; no free-text routing or authority transfer."
        ),
        maximum_cost_minor=cost,
        maximum_human_minutes=human,
        requires_human_approval=approval_required,
        risk_level=risk,
        reversibility=OperationsReversibility.REVERSIBLE,
        prerequisite_fact_ids=prerequisite_fact_ids,
        prediction_ranges=(
            OperationsEstimateRange(
                metric="operations_utility",
                lower_bound=max(-1.0, expected_utility - 0.30),
                upper_bound=min(1.0, expected_utility + 0.30),
                unit="normalized_multiobjective_utility",
                horizon_start_ms=request.operating_window.starts_at_ms,
                horizon_end_ms=request.operating_window.ends_at_ms,
                evidence_ref_ids=evidence_ref_ids,
            ),
        ),
        falsification_conditions=(
            "Owner-measured operations_utility is below the published lower bound.",
        ),
        evidence_ref_ids=evidence_ref_ids,
        source_entry_ids=source_entry_ids,
    )


class OperationsPolicy:
    """Pure checkpoint-in/checkpoint-out domain policy owner."""

    def decide(
        self,
        *,
        request: OperationsContextRequest,
        recalled_experiences: tuple[OperationsRecalledExperience, ...],
        source_prediction_id: str,
        checkpoint: OperationsPolicyCheckpoint | None = None,
        mode: OperationsPolicyMode = OperationsPolicyMode.LEARNED,
    ) -> tuple[tuple[OperationsAdviceCandidate, ...], OperationsPolicyDecision]:
        if request.operations_state is None:
            raise ValueError("Operations policy requires a v2 operations_state")
        if not isinstance(mode, OperationsPolicyMode):
            raise TypeError("mode must be OperationsPolicyMode")
        current = checkpoint or default_operations_policy_checkpoint()
        if current.artifact_id != OPERATIONS_POLICY_ARTIFACT_ID:
            raise ValueError("Operations policy artifact mismatch")
        source_entry_ids = tuple(
            item.memory_entry_id for item in recalled_experiences[:5]
        )
        generated: list[OperationsAdviceCandidate] = []
        policy_candidates: list[BoundedPolicyCandidate] = []
        for division_id in request.division_ids:
            features = _division_features(request, division_id)
            for kind in OperationsAdviceKind:
                candidate = _candidate(
                    request=request,
                    checkpoint=current,
                    division_id=division_id,
                    kind=kind,
                    features=features,
                    source_entry_ids=source_entry_ids,
                )
                generated.append(candidate)
                policy_candidates.append(
                    BoundedPolicyCandidate(
                        candidate_id=candidate.candidate_id,
                        action_key=kind.value,
                        feature_values=features,
                    )
                )
        limit = min(
            len(generated),
            32,
            request.operating_window.maximum_work_orders,
        )
        shared_decision = rank_and_gate_bounded_policy(
            candidates=tuple(policy_candidates),
            action_weights=current.action_weights,
            intervention_weights=current.intervention_weights,
            intervention_bias=current.intervention_bias,
            maximum_candidates=limit,
            intervention_enabled=mode is not OperationsPolicyMode.NOOP,
        )
        candidates_by_id = {item.candidate_id: item for item in generated}
        ranked = tuple(
            OperationsRankedCandidate(
                candidate_id=item.candidate_id,
                rank=item.rank,
                policy_score=item.policy_score,
                selection_probability=item.selection_probability,
                feature_values=tuple(
                    zip(
                        OPERATIONS_POLICY_FEATURE_ORDER,
                        item.feature_values,
                        strict=True,
                    )
                ),
            )
            for item in shared_decision.ranked_candidates
        )
        candidates = tuple(
            candidates_by_id[item.candidate_id]
            for item in shared_decision.ranked_candidates
        )
        action = (
            OperationsPolicyAction.INTERVENE
            if shared_decision.intervenes
            else OperationsPolicyAction.NOOP
        )
        decision = OperationsPolicyDecision.create(
            checkpoint_id=current.checkpoint_id,
            checkpoint_update_count=current.update_count,
            state_snapshot_id=request.operations_state.state_snapshot_id,
            source_prediction_id=source_prediction_id,
            mode=mode,
            action=action,
            recommended_candidate_id=shared_decision.recommended_candidate_id,
            selected_candidate_id=shared_decision.selected_candidate_id,
            intervention_probability=shared_decision.intervention_probability,
            ranked_candidates=ranked,
            rationale_codes=(
                "inputs:typed-operations-state-only",
                "ranking:bounded-linear-softmax",
                "timing:bounded-logistic-gate",
                "learning:exact-pe-credit-only",
            ),
        )
        return candidates, decision

    def observe_credit(
        self,
        *,
        checkpoint: OperationsPolicyCheckpoint,
        decision: OperationsPolicyDecision,
        candidates: tuple[OperationsAdviceCandidate, ...],
        credit: OperationsPolicyCredit,
    ) -> tuple[OperationsPolicyCheckpoint, OperationsPolicyUpdateReceipt]:
        if decision.mode is not OperationsPolicyMode.LEARNED:
            raise ValueError("only learned policy decisions can consume credit")
        if decision.checkpoint_id != checkpoint.checkpoint_id:
            raise ValueError("policy credit decision/checkpoint lineage mismatch")
        if credit.policy_decision_id != decision.policy_decision_id:
            raise ValueError("policy credit references another decision")
        if credit.prediction_id != decision.source_prediction_id:
            raise ValueError("policy credit references another owner prediction")
        if credit.credit_id in checkpoint.processed_credit_ids:
            raise ValueError("Operations policy credit was already processed")
        candidates_by_id = {item.candidate_id: item for item in candidates}
        if len(candidates_by_id) != len(candidates):
            raise ValueError("candidate ids must be unique")
        ranked_ids = tuple(item.candidate_id for item in decision.ranked_candidates)
        if set(candidates_by_id) != set(ranked_ids):
            raise ValueError("policy update candidate surface drift")
        noop_credit = credit.candidate_id == OPERATIONS_POLICY_NOOP_CANDIDATE_ID
        if noop_credit:
            if decision.action is not OperationsPolicyAction.NOOP:
                raise ValueError("NOOP credit requires a NOOP policy decision")
        else:
            if decision.action is not OperationsPolicyAction.INTERVENE:
                raise ValueError("candidate credit requires an INTERVENE decision")
            if credit.candidate_id != decision.selected_candidate_id:
                raise ValueError("policy credit candidate lineage mismatch")

        shared_decision = BoundedPolicyDecision(
            ranked_candidates=tuple(
                BoundedPolicyRankedCandidate(
                    candidate_id=item.candidate_id,
                    action_key=candidates_by_id[item.candidate_id].kind.value,
                    rank=item.rank,
                    policy_score=item.policy_score,
                    selection_probability=item.selection_probability,
                    feature_values=tuple(value for _, value in item.feature_values),
                )
                for item in decision.ranked_candidates
            ),
            recommended_candidate_id=decision.recommended_candidate_id,
            selected_candidate_id=decision.selected_candidate_id,
            intervention_probability=decision.intervention_probability,
            intervenes=decision.action is OperationsPolicyAction.INTERVENE,
        )
        update = apply_bounded_policy_credit(
            action_weights=checkpoint.action_weights,
            intervention_weights=checkpoint.intervention_weights,
            intervention_bias=checkpoint.intervention_bias,
            decision=shared_decision,
            credited_candidate_id=credit.candidate_id,
            noop_candidate_id=OPERATIONS_POLICY_NOOP_CANDIDATE_ID,
            signed_credit=credit.signed_prediction_error,
            learning_rate=checkpoint.learning_rate,
            max_abs_parameter=checkpoint.max_abs_parameter,
        )
        next_checkpoint = OperationsPolicyCheckpoint.create(
            artifact_id=checkpoint.artifact_id,
            action_weights=update.action_weights,
            intervention_weights=update.intervention_weights,
            intervention_bias=update.intervention_bias,
            learning_rate=checkpoint.learning_rate,
            max_abs_parameter=checkpoint.max_abs_parameter,
            update_count=checkpoint.update_count + 1,
            processed_credit_ids=tuple(
                (*checkpoint.processed_credit_ids, credit.credit_id)
            ),
        )
        receipt = OperationsPolicyUpdateReceipt.create(
            credit_id=credit.credit_id,
            policy_decision_id=decision.policy_decision_id,
            candidate_id=credit.candidate_id,
            previous_checkpoint_id=checkpoint.checkpoint_id,
            next_checkpoint_id=next_checkpoint.checkpoint_id,
            parameter_delta_l2=update.parameter_delta_l2,
            update_count=next_checkpoint.update_count,
        )
        return next_checkpoint, receipt


def settle_operations_policy_credit(
    *,
    prediction_error_snapshot: PredictionErrorSnapshot,
    credit_snapshot: CreditSnapshot,
    policy_decision_id: str,
    selection_id: str,
    candidate_id: str,
    environment_outcome_id: str,
) -> OperationsPolicyCredit:
    """Create one policy credit only after exact PE and Credit-owner join."""

    if prediction_error_snapshot.bootstrap:
        raise ValueError("bootstrap PE cannot settle Operations policy credit")
    context = prediction_error_snapshot.actual_outcome.action_context
    if context.environment_outcome_id != environment_outcome_id:
        raise ValueError("Operations policy PE environment outcome mismatch")
    if not context.prediction_id:
        raise ValueError("Operations policy PE prediction lineage is missing")
    evaluated_prediction = prediction_error_snapshot.evaluated_prediction
    if (
        evaluated_prediction is None
        or evaluated_prediction.prediction_id != context.prediction_id
    ):
        raise ValueError("Operations policy PE did not settle the bound prediction")
    if prediction_error_snapshot.action_context != context:
        raise ValueError("Operations policy PE action-context lineage diverged")
    lineage_records = {
        record.record_id: record
        for record in (
            *credit_snapshot.recent_credits,
            *credit_snapshot.recent_action_lineage_credits,
        )
    }
    matching: list[CreditRecord] = [
        record
        for record in lineage_records.values()
        if record.level == "prediction_error"
        and record.source_event in _PE_CREDIT_SOURCES
        and record.environment_outcome_id == environment_outcome_id
        and record.prediction_id == context.prediction_id
    ]
    by_source = {record.source_event: record for record in matching}
    if len(matching) != len(_PE_CREDIT_SOURCES) or set(by_source) != set(
        _PE_CREDIT_SOURCES
    ):
        raise ValueError(
            "Operations policy requires exactly four owner-published PE credits"
        )
    ordered = tuple(by_source[source] for source in _PE_CREDIT_SOURCES)
    return OperationsPolicyCredit.create(
        policy_decision_id=policy_decision_id,
        selection_id=selection_id,
        candidate_id=candidate_id,
        environment_outcome_id=environment_outcome_id,
        prediction_id=context.prediction_id,
        signed_prediction_error=_clamp(
            prediction_error_snapshot.error.signed_reward,
            -1.0,
            1.0,
        ),
        source_credit_record_ids=tuple(item.record_id for item in ordered),
        observed_at_ms=max(item.timestamp_ms for item in ordered),
    )


__all__ = (
    "OPERATIONS_POLICY_ARTIFACT_ID",
    "OPERATIONS_POLICY_NOOP_CANDIDATE_ID",
    "OperationsPolicy",
    "default_operations_policy_checkpoint",
    "settle_operations_policy_credit",
)
