from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from lifeform_domain_venture import (
    VentureAdviceCandidate,
    VentureAdviceKind,
    VentureAdviceSnapshot,
    VentureContextRequest,
    VentureEstimateRange,
    VentureEvidenceClass,
    VentureOutcomeKind,
    VentureOutcomeReport,
)
from volvence_zero.runtime import WiringLevel


def _evidence(
    *,
    ref_id: str = "demand-1",
    evidence_class: str = "field",
    role: str = "demand_signal",
    digest_char: str = "a",
) -> dict[str, object]:
    return {
        "ref_id": ref_id,
        "evidence_class": evidence_class,
        "role": role,
        "locator": f"foundry://evidence/{ref_id}",
        "content_sha256": digest_char * 64,
        "observed_at_ms": 100,
    }


def _context_payload() -> dict[str, object]:
    return {
        "schema_version": "venture-context-request.v1",
        "request_id": "request-1",
        "portfolio_id": "portfolio-1",
        "cycle_id": "cycle-1",
        "venture_id": "venture-1",
        "decision_id": "decision-1",
        "decision_point": "experiment_planning",
        "confirmed_facts": [
            {
                "fact_id": "fact-1",
                "kind": "demand_signal",
                "statement": "Three customers requested the workflow.",
                "evidence_ref_ids": ["demand-1"],
                "as_of_ms": 100,
            }
        ],
        "constraints": [
            {
                "constraint_id": "constraint-1",
                "kind": "budget",
                "description": "Do not exceed the approved experiment budget.",
                "hard": True,
            }
        ],
        "resource_window": {
            "currency": "USD",
            "maximum_total_cost_minor": 10_000,
            "starts_at_ms": 100,
            "ends_at_ms": 10_000,
            "maximum_experiments": 2,
        },
        "uncertainties": [
            {
                "uncertainty_id": "uncertainty-1",
                "statement": "Paid conversion remains unknown.",
                "probability_lower": 0.1,
                "probability_upper": 0.5,
                "evidence_ref_ids": ["demand-1"],
            }
        ],
        "evidence_refs": [_evidence()],
    }


def _costs(**overrides: int) -> dict[str, int]:
    values = {
        "acquisition_minor": 0,
        "model_minor": 0,
        "data_minor": 0,
        "human_review_minor": 0,
        "delivery_minor": 0,
        "support_minor": 0,
        "risk_reserve_minor": 0,
    }
    values.update(overrides)
    return values


def _commercial(
    *,
    revenue: int = 0,
    costs: dict[str, int] | None = None,
    refund: int = 0,
    customer_result: str = "not_observed",
    elapsed_ms: int = 0,
    risk_level: str = "unassessed",
) -> dict[str, object]:
    realized_costs = costs or _costs()
    return {
        "customer_result": customer_result,
        "currency": "USD",
        "realized_revenue_minor": revenue,
        "realized_costs": realized_costs,
        "refund_minor": refund,
        "realized_net_value_minor": revenue - sum(realized_costs.values()) - refund,
        "elapsed_ms": elapsed_ms,
        "risk_level": risk_level,
        "reversibility": "reversible",
    }


def _outcome_payload(
    *,
    outcome_kind: str,
    evidence_class: str,
    role: str,
    commercial: dict[str, object] | None = None,
) -> dict[str, object]:
    commercial_payload = commercial or _commercial()
    evidence_refs = [
        _evidence(
            ref_id="result-1",
            evidence_class=evidence_class,
            role=role,
            digest_char="c",
        )
    ]
    if evidence_class == "field":
        claimed_roles = {role}
        role_claims = (
            (commercial_payload["customer_result"] != "not_observed", "customer_outcome", "d"),
            (commercial_payload["realized_revenue_minor"] != 0, "payment", "e"),
            (any(commercial_payload["realized_costs"].values()), "cost", "f"),  # type: ignore[union-attr]
            (commercial_payload["refund_minor"] != 0, "refund", "0"),
        )
        for observed, required_role, digest_char in role_claims:
            if observed and required_role not in claimed_roles:
                evidence_refs.append(
                    _evidence(
                        ref_id=f"result-{required_role}",
                        evidence_class=evidence_class,
                        role=required_role,
                        digest_char=digest_char,
                    )
                )
                claimed_roles.add(required_role)
    return {
        "schema_version": "venture-outcome-report.v1",
        "outcome_id": "outcome-1",
        "context_pack_id": "venture-context-pack:" + "b" * 64,
        "decision_id": "decision-1",
        "decision": "continue",
        "outcome_kind": outcome_kind,
        "evidence_class": evidence_class,
        "verdict": "inconclusive",
        "summary": "Foundry supplied a typed result.",
        "detail": "The evidence remains in its declared lane.",
        "observed_at_ms": 200,
        "evidence_refs": evidence_refs,
        "commercial_outcome": commercial_payload,
    }


def test_context_request_is_versioned_strict_frozen_and_lineage_checked() -> None:
    request = VentureContextRequest.from_json(_context_payload())
    assert request.to_json()["schema_version"] == "venture-context-request.v1"
    with pytest.raises(FrozenInstanceError):
        request.decision_id = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="unknown fields"):
        VentureContextRequest.from_json({**_context_payload(), "guessed_route": "field"})
    with pytest.raises(ValueError, match="schema_version"):
        VentureContextRequest.from_json({**_context_payload(), "schema_version": "venture-context-request.v2"})
    with pytest.raises(ValueError, match="memory_limit"):
        VentureContextRequest.from_json({**_context_payload(), "memory_limit": "many"})
    bad_fact = _context_payload()
    bad_fact["confirmed_facts"] = [
        {
            "fact_id": "fact-1",
            "kind": "demand_signal",
            "statement": "Untraceable assertion.",
            "evidence_ref_ids": ["missing"],
            "as_of_ms": 100,
        }
    ]
    with pytest.raises(ValueError, match="unknown evidence ids"):
        VentureContextRequest.from_json(bad_fact)


@pytest.mark.parametrize(
    "decision_point",
    (
        "opportunity_brainstorm",
        "candidate_comparison",
        "experiment_planning",
        "product_design",
        "portfolio_review",
        "monitor_attribution",
        "stop_review",
    ),
)
def test_foundry_v1_decision_points_are_closed_and_complete(decision_point: str) -> None:
    payload = {**_context_payload(), "decision_point": decision_point}
    assert VentureContextRequest.from_json(payload).decision_point.value == decision_point


@pytest.mark.parametrize(
    ("outcome_kind", "evidence_class", "role"),
    (
        ("simulation_result", "field", "field_observation"),
        ("internal_review_result", "machine_check", "machine_audit"),
        ("payment_received", "simulation", "experiment"),
        ("field_experiment_result", "internal_review", "internal_review"),
    ),
)
def test_outcome_class_kind_pairs_fail_closed(
    outcome_kind: str,
    evidence_class: str,
    role: str,
) -> None:
    with pytest.raises(ValueError, match="not legal"):
        VentureOutcomeReport.from_json(
            _outcome_payload(
                outcome_kind=outcome_kind,
                evidence_class=evidence_class,
                role=role,
            )
        )


@pytest.mark.parametrize(
    ("outcome_kind", "evidence_class", "role", "commercial"),
    (
        ("simulation_result", "simulation", "experiment", None),
        ("internal_review_result", "internal_review", "internal_review", None),
        ("machine_check_result", "machine_check", "machine_audit", None),
        (
            "customer_outcome",
            "field",
            "customer_outcome",
            _commercial(customer_result="positive", elapsed_ms=100, risk_level="low"),
        ),
        ("payment_received", "field", "payment", _commercial(revenue=1_000)),
        (
            "cost_recorded",
            "field",
            "cost",
            _commercial(costs=_costs(model_minor=300)),
        ),
        ("refund_recorded", "field", "refund", _commercial(refund=100)),
        (
            "field_experiment_result",
            "field",
            "field_observation",
            _commercial(customer_result="mixed", elapsed_ms=500, risk_level="medium"),
        ),
    ),
)
def test_legal_outcome_lanes_are_explicit_and_only_field_aggregate_is_pe_eligible(
    outcome_kind: str,
    evidence_class: str,
    role: str,
    commercial: dict[str, object] | None,
) -> None:
    report = VentureOutcomeReport.from_json(
        _outcome_payload(
            outcome_kind=outcome_kind,
            evidence_class=evidence_class,
            role=role,
            commercial=commercial,
        )
    )
    assert report.evidence_class is VentureEvidenceClass(evidence_class)
    assert report.outcome_kind is VentureOutcomeKind(outcome_kind)
    assert report.pe_eligible is (outcome_kind == "field_experiment_result")


def test_review_machine_and_financial_claims_cannot_cross_lanes() -> None:
    with pytest.raises(ValueError, match="role=internal_review"):
        VentureOutcomeReport.from_json(
            _outcome_payload(
                outcome_kind="internal_review_result",
                evidence_class="internal_review",
                role="payment",
            )
        )
    with pytest.raises(ValueError, match="cannot carry"):
        VentureOutcomeReport.from_json(
            _outcome_payload(
                outcome_kind="machine_check_result",
                evidence_class="machine_check",
                role="machine_audit",
                commercial=_commercial(revenue=1_000),
            )
        )
    with pytest.raises(ValueError, match="simulation evidence"):
        VentureOutcomeReport.from_json(
            _outcome_payload(
                outcome_kind="simulation_result",
                evidence_class="simulation",
                role="payment",
            )
        )
    with pytest.raises(ValueError, match="cannot carry"):
        VentureOutcomeReport.from_json(
            _outcome_payload(
                outcome_kind="simulation_result",
                evidence_class="simulation",
                role="experiment",
                commercial=_commercial(revenue=1_000),
            )
        )


def test_field_commercial_dimensions_require_matching_evidence_roles() -> None:
    payload = _outcome_payload(
        outcome_kind="field_experiment_result",
        evidence_class="field",
        role="field_observation",
        commercial=_commercial(
            revenue=1_000,
            costs=_costs(model_minor=100),
            refund=50,
            customer_result="positive",
        ),
    )
    payload["evidence_refs"] = [payload["evidence_refs"][0]]  # type: ignore[index]
    with pytest.raises(ValueError, match="customer_result.*customer_outcome"):
        VentureOutcomeReport.from_json(payload)


def test_commercial_outcome_keeps_all_seven_costs_and_checks_net_value() -> None:
    costs = _costs(
        acquisition_minor=100,
        model_minor=200,
        data_minor=300,
        human_review_minor=400,
        delivery_minor=500,
        support_minor=600,
        risk_reserve_minor=700,
    )
    payload = _outcome_payload(
        outcome_kind="field_experiment_result",
        evidence_class="field",
        role="field_observation",
        commercial=_commercial(
            revenue=5_000,
            costs=costs,
            refund=100,
            customer_result="positive",
            elapsed_ms=86_400_000,
            risk_level="low",
        ),
    )
    report = VentureOutcomeReport.from_json(payload)
    assert report.commercial_outcome.realized_costs.total_minor == 2_800
    assert report.commercial_outcome.realized_net_value_minor == 2_100
    invalid = dict(payload)
    invalid["commercial_outcome"] = {
        **payload["commercial_outcome"],  # type: ignore[misc]
        "realized_net_value_minor": 5_000,
    }
    with pytest.raises(ValueError, match="seven costs"):
        VentureOutcomeReport.from_json(invalid)


def test_advice_is_structured_falsifiable_and_permanently_shadow() -> None:
    prediction = VentureEstimateRange(
        metric="qualified_customer_count",
        lower_bound=1,
        upper_bound=3,
        unit="customers",
        horizon_start_ms=100,
        horizon_end_ms=10_000,
        evidence_ref_ids=("demand-1",),
    )
    candidate = VentureAdviceCandidate(
        candidate_id="candidate-1",
        kind=VentureAdviceKind.EXPERIMENT,
        summary="Run a reversible paid-intent test.",
        rationale="It resolves the largest declared uncertainty at bounded cost.",
        prediction_ranges=(prediction,),
        falsification_conditions=("No qualified customer accepts the price.",),
        evidence_ref_ids=("demand-1",),
        source_entry_ids=(),
    )
    advice = VentureAdviceSnapshot(
        advice_id="venture-advice:" + "d" * 64,
        source_turn_index=1,
        candidate_regime_id="problem_solving",
        candidate_abstract_action="compare_options",
        candidates=(candidate,),
        rationale="Structured projection only.",
    )
    assert advice.to_json()["schema_version"] == "venture-advice.v1"
    assert advice.wiring_level is WiringLevel.SHADOW
    assert advice.applied is False
    assert VentureAdviceSnapshot.from_json(advice.to_json()) == advice
    with pytest.raises(ValueError, match="unknown fields"):
        VentureAdviceSnapshot.from_json({**advice.to_json(), "unknown": True})
    with pytest.raises(ValueError, match="SHADOW"):
        VentureAdviceSnapshot(
            advice_id="venture-advice:" + "e" * 64,
            source_turn_index=1,
            candidate_regime_id="",
            candidate_abstract_action="",
            candidates=(candidate,),
            rationale="Invalid promotion attempt.",
            wiring_level=WiringLevel.ACTIVE,
        )
