from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from lifeform_domain_operations.operations_brain_contracts import stable_content_sha256
from lifeform_domain_operations import (
    OperationsAdviceCandidate,
    OperationsAdviceKind,
    OperationsAdviceSnapshot,
    OperationsContextRequest,
    OperationsEstimateRange,
    OperationsEvidenceClass,
    OperationsOutcomeKind,
    OperationsOutcomeReport,
    OperationsReversibility,
    OperationsRiskLevel,
)
from volvence_zero.runtime import WiringLevel


_WORK_ORDER_REF = "autocompany://work-orders/work-order-1"


def test_canonical_digest_has_a_cross_runtime_fixed_vector() -> None:
    assert stable_content_sha256(
        {
            "z": [None, True, False, 1, 1.5, -0.0, "中文"],
            "a": {"b": "x", "a": 2},
        }
    ) == "b663952e1fbde628165aca891402b850f12ae1c54305609cb134c0e1a3683370"
    with pytest.raises(ValueError, match="safe range"):
        stable_content_sha256({"unsafe": float(1 << 53)})
    with pytest.raises(ValueError, match="Unicode scalar"):
        stable_content_sha256({"unsafe": "\ud800"})


def _evidence(
    *,
    ref_id: str = "operating-signal-1",
    evidence_class: str = "field",
    role: str = "operating_signal",
    digest_char: str = "a",
    locator: str | None = None,
) -> dict[str, object]:
    return {
        "ref_id": ref_id,
        "evidence_class": evidence_class,
        "role": role,
        "locator": locator or f"autocompany://evidence/{ref_id}",
        "content_sha256": digest_char * 64,
        "observed_at_ms": 100,
    }


def _context_payload() -> dict[str, object]:
    return {
        "schema_version": "operations-context-request.v1",
        "request_id": "request-1",
        "company_id": "company-1",
        "cycle_id": "cycle-1",
        "workstream_id": "reliability",
        "decision_id": "decision-1",
        "decision_point": "work_prioritization",
        "division_ids": ["division-engineering", "division-support"],
        "action_catalog_ids": ["catalog:repair-service", "catalog:pause-rollout"],
        "confirmed_facts": [
            {
                "fact_id": "fact-1",
                "kind": "division_health",
                "division_id": "division-engineering",
                "statement": "The service error budget is nearly exhausted.",
                "evidence_ref_ids": ["operating-signal-1"],
                "as_of_ms": 100,
            }
        ],
        "constraints": [
            {
                "constraint_id": "constraint-1",
                "kind": "budget",
                "division_id": "",
                "description": "Do not exceed the approved operating budget.",
                "hard": True,
            }
        ],
        "operating_window": {
            "currency": "USD",
            "maximum_external_cost_minor": 10_000,
            "maximum_human_minutes": 240,
            "starts_at_ms": 100,
            "ends_at_ms": 10_000,
            "maximum_work_orders": 2,
        },
        "uncertainties": [
            {
                "uncertainty_id": "uncertainty-1",
                "statement": "The primary failure mechanism is not yet isolated.",
                "probability_lower": 0.3,
                "probability_upper": 0.7,
                "evidence_ref_ids": ["operating-signal-1"],
            }
        ],
        "evidence_refs": [_evidence()],
    }


def _costs(**overrides: int) -> dict[str, int]:
    values = {
        "model_minor": 0,
        "data_minor": 0,
        "human_minor": 0,
        "infrastructure_minor": 0,
        "vendor_minor": 0,
        "incident_response_minor": 0,
        "other_minor": 0,
    }
    values.update(overrides)
    return values


def _execution(
    *,
    objective_result: str = "not_observed",
    metrics: list[dict[str, object]] | None = None,
    costs: dict[str, int] | None = None,
    elapsed_ms: int = 0,
    blocker_duration_ms: int = 0,
    rework_count: int = 0,
    incident_count: int = 0,
    human_minutes: int = 0,
    risk_level: str = "unassessed",
) -> dict[str, object]:
    return {
        "objective_result": objective_result,
        "metrics": metrics or [],
        "currency": "USD",
        "realized_costs": costs or _costs(),
        "elapsed_ms": elapsed_ms,
        "blocker_duration_ms": blocker_duration_ms,
        "rework_count": rework_count,
        "incident_count": incident_count,
        "human_minutes": human_minutes,
        "risk_level": risk_level,
        "reversibility": "reversible",
    }


def _outcome_payload(
    *,
    outcome_kind: str,
    evidence_class: str,
    role: str,
    execution: dict[str, object] | None = None,
) -> dict[str, object]:
    execution_payload = execution or _execution()
    evidence_refs = [
        _evidence(
            ref_id="result-1",
            evidence_class=evidence_class,
            role=role,
            digest_char="b",
        )
    ]
    if evidence_class == "field":
        evidence_refs.append(
            _evidence(
                ref_id="work-order-1",
                evidence_class="field",
                role="work_order",
                digest_char="c",
                locator=_WORK_ORDER_REF,
            )
        )
        claimed_roles = {role, "work_order"}
        role_claims = (
            (execution_payload["objective_result"] != "not_observed", "objective_progress", "d"),
            (bool(execution_payload["metrics"]), "field_observation", "e"),
            (any(execution_payload["realized_costs"].values()), "cost", "f"),  # type: ignore[union-attr]
            (execution_payload["incident_count"] != 0, "incident", "0"),
            (execution_payload["human_minutes"] != 0, "human_load", "1"),
        )
        for observed, required_role, digest_char in role_claims:
            if observed and required_role not in claimed_roles:
                evidence_refs.append(
                    _evidence(
                        ref_id=f"result-{required_role}",
                        evidence_class="field",
                        role=required_role,
                        digest_char=digest_char,
                    )
                )
                claimed_roles.add(required_role)
    return {
        "schema_version": "operations-outcome-report.v1",
        "outcome_id": "outcome-1",
        "context_pack_id": "operations-context-pack:" + "b" * 64,
        "decision_id": "decision-1",
        "work_order_ref": _WORK_ORDER_REF,
        "decision": "accept",
        "outcome_kind": outcome_kind,
        "evidence_class": evidence_class,
        "verdict": "inconclusive",
        "summary": "AutoCompany supplied a typed operational result.",
        "detail": "The evidence remains in its declared lane.",
        "observed_at_ms": 200,
        "evidence_refs": evidence_refs,
        "execution_outcome": execution_payload,
    }


def test_context_request_is_versioned_strict_frozen_and_lineage_checked() -> None:
    request = OperationsContextRequest.from_json(_context_payload())
    assert request.to_json()["schema_version"] == "operations-context-request.v1"
    with pytest.raises(FrozenInstanceError):
        request.decision_id = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="unknown fields"):
        OperationsContextRequest.from_json({**_context_payload(), "guessed_route": "field"})
    with pytest.raises(ValueError, match="schema_version"):
        OperationsContextRequest.from_json(
            {**_context_payload(), "schema_version": "operations-context-request.v2"}
        )
    with pytest.raises(ValueError, match="memory_limit"):
        OperationsContextRequest.from_json({**_context_payload(), "memory_limit": "many"})

    bad_fact = _context_payload()
    bad_fact["confirmed_facts"] = [
        {
            "fact_id": "fact-1",
            "kind": "division_health",
            "division_id": "division-engineering",
            "statement": "Untraceable assertion.",
            "evidence_ref_ids": ["missing"],
            "as_of_ms": 100,
        }
    ]
    with pytest.raises(ValueError, match="unknown evidence ids"):
        OperationsContextRequest.from_json(bad_fact)

    bad_scope = _context_payload()
    bad_scope["constraints"] = [
        {
            "constraint_id": "constraint-1",
            "kind": "capacity",
            "division_id": "division-unknown",
            "description": "Unknown scope.",
            "hard": True,
        }
    ]
    with pytest.raises(ValueError, match="unknown division ids"):
        OperationsContextRequest.from_json(bad_scope)


@pytest.mark.parametrize(
    ("outcome_kind", "evidence_class", "role"),
    (
        ("simulation_result", "field", "field_observation"),
        ("internal_review_result", "machine_check", "machine_audit"),
        ("objective_progress", "simulation", "operating_signal"),
        ("field_operation_result", "internal_review", "internal_review"),
    ),
)
def test_outcome_class_kind_pairs_fail_closed(
    outcome_kind: str,
    evidence_class: str,
    role: str,
) -> None:
    with pytest.raises(ValueError, match="not legal"):
        OperationsOutcomeReport.from_json(
            _outcome_payload(
                outcome_kind=outcome_kind,
                evidence_class=evidence_class,
                role=role,
            )
        )


@pytest.mark.parametrize(
    ("outcome_kind", "evidence_class", "role", "execution"),
    (
        ("simulation_result", "simulation", "operating_signal", None),
        ("internal_review_result", "internal_review", "internal_review", None),
        ("machine_check_result", "machine_check", "machine_audit", None),
        ("work_order_progress", "field", "work_order", _execution(elapsed_ms=100)),
        (
            "objective_progress",
            "field",
            "objective_progress",
            _execution(objective_result="advanced"),
        ),
        ("cost_recorded", "field", "cost", _execution(costs=_costs(model_minor=300))),
        ("incident_recorded", "field", "incident", _execution(incident_count=1)),
        ("human_load_recorded", "field", "human_load", _execution(human_minutes=30)),
        (
            "field_operation_result",
            "field",
            "field_observation",
            _execution(
                objective_result="mixed",
                metrics=[
                    {
                        "metric_id": "error_rate",
                        "unit": "ratio",
                        "baseline_value": 0.08,
                        "observed_value": 0.03,
                        "evidence_ref_ids": ["result-1"],
                    }
                ],
                elapsed_ms=500,
                risk_level="medium",
            ),
        ),
    ),
)
def test_legal_outcome_lanes_are_explicit_and_only_field_aggregate_is_pe_eligible(
    outcome_kind: str,
    evidence_class: str,
    role: str,
    execution: dict[str, object] | None,
) -> None:
    report = OperationsOutcomeReport.from_json(
        _outcome_payload(
            outcome_kind=outcome_kind,
            evidence_class=evidence_class,
            role=role,
            execution=execution,
        )
    )
    assert report.evidence_class is OperationsEvidenceClass(evidence_class)
    assert report.outcome_kind is OperationsOutcomeKind(outcome_kind)
    assert report.pe_eligible is (outcome_kind == "field_operation_result")


def test_non_field_lanes_cannot_claim_operational_observations() -> None:
    with pytest.raises(ValueError, match="role=internal_review"):
        OperationsOutcomeReport.from_json(
            _outcome_payload(
                outcome_kind="internal_review_result",
                evidence_class="internal_review",
                role="operating_signal",
            )
        )
    with pytest.raises(ValueError, match="cannot carry"):
        OperationsOutcomeReport.from_json(
            _outcome_payload(
                outcome_kind="machine_check_result",
                evidence_class="machine_check",
                role="machine_audit",
                execution=_execution(elapsed_ms=1_000),
            )
        )
    with pytest.raises(ValueError, match="simulation evidence"):
        OperationsOutcomeReport.from_json(
            _outcome_payload(
                outcome_kind="simulation_result",
                evidence_class="simulation",
                role="field_observation",
            )
        )


def test_field_dimensions_and_work_order_require_matching_evidence() -> None:
    payload = _outcome_payload(
        outcome_kind="field_operation_result",
        evidence_class="field",
        role="field_observation",
        execution=_execution(
            objective_result="advanced",
            costs=_costs(infrastructure_minor=100),
            incident_count=1,
            human_minutes=20,
        ),
    )
    payload["evidence_refs"] = [payload["evidence_refs"][0]]  # type: ignore[index]
    with pytest.raises(ValueError, match="role=work_order"):
        OperationsOutcomeReport.from_json(payload)

    locator_mismatch = _outcome_payload(
        outcome_kind="work_order_progress",
        evidence_class="field",
        role="work_order",
        execution=_execution(elapsed_ms=100),
    )
    for reference in locator_mismatch["evidence_refs"]:  # type: ignore[union-attr]
        if reference["role"] == "work_order":
            reference["locator"] = "autocompany://work-orders/other"
    with pytest.raises(ValueError, match="matching work_order_ref"):
        OperationsOutcomeReport.from_json(locator_mismatch)

    elapsed_only = _outcome_payload(
        outcome_kind="field_operation_result",
        evidence_class="field",
        role="work_order",
        execution=_execution(elapsed_ms=100),
    )
    with pytest.raises(ValueError, match="beyond elapsed time"):
        OperationsOutcomeReport.from_json(elapsed_only)


def test_execution_outcome_keeps_all_operational_costs_and_metric_lineage() -> None:
    costs = _costs(
        model_minor=100,
        data_minor=200,
        human_minor=300,
        infrastructure_minor=400,
        vendor_minor=500,
        incident_response_minor=600,
        other_minor=700,
    )
    payload = _outcome_payload(
        outcome_kind="field_operation_result",
        evidence_class="field",
        role="field_observation",
        execution=_execution(
            objective_result="advanced",
            metrics=[
                {
                    "metric_id": "cycle_time",
                    "unit": "minutes",
                    "baseline_value": 90,
                    "observed_value": 55,
                    "evidence_ref_ids": ["result-1"],
                }
            ],
            costs=costs,
            elapsed_ms=86_400_000,
            blocker_duration_ms=3_600_000,
            rework_count=1,
            incident_count=1,
            human_minutes=75,
            risk_level="low",
        ),
    )
    report = OperationsOutcomeReport.from_json(payload)
    assert report.execution_outcome.realized_costs.total_minor == 2_800
    assert report.execution_outcome.metrics[0].observed_value == 55

    invalid = _outcome_payload(
        outcome_kind="field_operation_result",
        evidence_class="field",
        role="field_observation",
        execution=_execution(
            metrics=[
                {
                    "metric_id": "cycle_time",
                    "unit": "minutes",
                    "baseline_value": 90,
                    "observed_value": 55,
                    "evidence_ref_ids": ["unknown"],
                }
            ]
        ),
    )
    with pytest.raises(ValueError, match="unknown evidence ids"):
        OperationsOutcomeReport.from_json(invalid)


def test_advice_is_catalog_bounded_falsifiable_and_permanently_shadow() -> None:
    prediction = OperationsEstimateRange(
        metric="error_rate",
        lower_bound=0.0,
        upper_bound=0.04,
        unit="ratio",
        horizon_start_ms=100,
        horizon_end_ms=10_000,
        evidence_ref_ids=("operating-signal-1",),
    )
    candidate = OperationsAdviceCandidate(
        candidate_id="candidate-1",
        kind=OperationsAdviceKind.PRIORITIZE_WORK,
        target_division_id="division-engineering",
        action_catalog_id="catalog:repair-service",
        summary="Prioritize the bounded reliability repair.",
        rationale="It addresses the declared health risk within the approved window.",
        maximum_cost_minor=5_000,
        maximum_human_minutes=120,
        requires_human_approval=False,
        risk_level=OperationsRiskLevel.LOW,
        reversibility=OperationsReversibility.REVERSIBLE,
        prerequisite_fact_ids=("fact-1",),
        prediction_ranges=(prediction,),
        falsification_conditions=("The error rate remains above four percent.",),
        evidence_ref_ids=("operating-signal-1",),
        source_entry_ids=(),
    )
    advice = OperationsAdviceSnapshot(
        advice_id="operations-advice:" + "d" * 64,
        source_turn_index=1,
        candidate_regime_id="problem_solving",
        candidate_abstract_action="compare_options",
        candidates=(candidate,),
        rationale="Structured projection only.",
    )
    assert advice.to_json()["schema_version"] == "operations-advice.v1"
    assert advice.wiring_level is WiringLevel.SHADOW
    assert advice.applied is False
    assert OperationsAdviceSnapshot.from_json(advice.to_json()) == advice
    with pytest.raises(ValueError, match="must be assessed"):
        replace(candidate, risk_level=OperationsRiskLevel.UNASSESSED)
    with pytest.raises(ValueError, match="unknown fields"):
        OperationsAdviceSnapshot.from_json({**advice.to_json(), "unknown": True})
    with pytest.raises(ValueError, match="SHADOW"):
        OperationsAdviceSnapshot(
            advice_id="operations-advice:" + "e" * 64,
            source_turn_index=1,
            candidate_regime_id="",
            candidate_abstract_action="",
            candidates=(candidate,),
            rationale="Invalid promotion attempt.",
            wiring_level=WiringLevel.ACTIVE,
        )
