from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from lifeform_core import LifeformConfig
from lifeform_domain_operations import (
    OperationsAdviceCandidate,
    OperationsAdviceKind,
    OperationsBrainConflictError,
    OperationsBrainController,
    OperationsBrainLineageError,
    OperationsBrainSettlementPendingError,
    OperationsContextPackSnapshot,
    OperationsContextRequest,
    OperationsEstimateRange,
    OperationsOutcomeReceipt,
    OperationsOutcomeReport,
    OperationsOutcomeRoute,
    OperationsReversibility,
    OperationsRiskLevel,
    build_operations_lifeform,
)
from volvence_zero.brain import BrainConfig
from volvence_zero.memory import StaticIdentityProvider, UserIdentity
from volvence_zero.runtime import WiringLevel


_WORK_ORDER_REF = "autocompany://work-orders/work-order-1"


def _evidence(
    *,
    ref_id: str,
    evidence_class: str,
    role: str,
    observed_at_ms: int,
    digest_char: str,
    locator: str | None = None,
) -> dict[str, object]:
    return {
        "ref_id": ref_id,
        "evidence_class": evidence_class,
        "role": role,
        "locator": locator or f"autocompany://evidence/{ref_id}",
        "content_sha256": digest_char * 64,
        "observed_at_ms": observed_at_ms,
    }


def _request(
    *,
    request_id: str = "request-1",
    decision_id: str = "decision-1",
    cycle_id: str = "cycle-1",
    company_id: str = "company-1",
) -> OperationsContextRequest:
    return OperationsContextRequest.from_json(
        {
            "schema_version": "operations-context-request.v1",
            "request_id": request_id,
            "company_id": company_id,
            "cycle_id": cycle_id,
            "workstream_id": "reliability",
            "decision_id": decision_id,
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
                    "description": "Use only the approved bounded operating budget.",
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
            "evidence_refs": [
                _evidence(
                    ref_id="operating-signal-1",
                    evidence_class="field",
                    role="operating_signal",
                    observed_at_ms=100,
                    digest_char="a",
                )
            ],
        }
    )


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


def _report(
    *,
    context_pack_id: str,
    outcome_id: str = "outcome-1",
    decision_id: str = "decision-1",
    outcome_kind: str = "field_operation_result",
    evidence_class: str = "field",
    role: str = "field_observation",
    verdict: str = "favorable",
    detail: str = "The bounded repair reduced errors without a new incident.",
    work_order_ref: str = _WORK_ORDER_REF,
) -> OperationsOutcomeReport:
    if outcome_kind == "field_operation_result":
        objective_result = "advanced"
        metrics = [
            {
                "metric_id": "error_rate",
                "unit": "ratio",
                "baseline_value": 0.08,
                "observed_value": 0.03,
                "evidence_ref_ids": [f"evidence-{outcome_id}"],
            }
        ]
        costs = _costs(infrastructure_minor=300, human_minor=200)
        elapsed_ms = 86_400_000
        blocker_duration_ms = 3_600_000
        rework_count = 1
        human_minutes = 90
        risk_level = "low"
    elif outcome_kind == "work_order_progress":
        objective_result = "not_observed"
        metrics = []
        costs = _costs()
        elapsed_ms = 1_000
        blocker_duration_ms = 0
        rework_count = 0
        human_minutes = 0
        risk_level = "unassessed"
    else:
        objective_result = "not_observed"
        metrics = []
        costs = _costs()
        elapsed_ms = 0
        blocker_duration_ms = 0
        rework_count = 0
        human_minutes = 0
        risk_level = "unassessed"

    evidence_refs = [
        _evidence(
            ref_id=f"evidence-{outcome_id}",
            evidence_class=evidence_class,
            role=role,
            observed_at_ms=200,
            digest_char="b",
        )
    ]
    if evidence_class == "field":
        evidence_refs.append(
            _evidence(
                ref_id=f"evidence-{outcome_id}-work-order",
                evidence_class="field",
                role="work_order",
                observed_at_ms=200,
                digest_char="c",
                locator=work_order_ref,
            )
        )
    if outcome_kind == "field_operation_result":
        for suffix, dimension_role, digest_char in (
            ("objective", "objective_progress", "d"),
            ("cost", "cost", "e"),
            ("human", "human_load", "f"),
        ):
            evidence_refs.append(
                _evidence(
                    ref_id=f"evidence-{outcome_id}-{suffix}",
                    evidence_class="field",
                    role=dimension_role,
                    observed_at_ms=200,
                    digest_char=digest_char,
                )
            )
    return OperationsOutcomeReport.from_json(
        {
            "schema_version": "operations-outcome-report.v1",
            "outcome_id": outcome_id,
            "context_pack_id": context_pack_id,
            "decision_id": decision_id,
            "work_order_ref": work_order_ref,
            "decision": "accept",
            "outcome_kind": outcome_kind,
            "evidence_class": evidence_class,
            "verdict": verdict,
            "summary": "AutoCompany supplied a typed Operations result.",
            "detail": detail,
            "observed_at_ms": 200,
            "evidence_refs": evidence_refs,
            "execution_outcome": {
                "objective_result": objective_result,
                "metrics": metrics,
                "currency": "USD",
                "realized_costs": costs,
                "elapsed_ms": elapsed_ms,
                "blocker_duration_ms": blocker_duration_ms,
                "rework_count": rework_count,
                "incident_count": 0,
                "human_minutes": human_minutes,
                "risk_level": risk_level,
                "reversibility": "reversible",
            },
        }
    )


def _session(*, session_id: str = "operations-session"):
    return build_operations_lifeform().create_session(session_id=session_id)


def _candidate(
    *,
    action_catalog_id: str = "catalog:repair-service",
    maximum_cost_minor: int = 5_000,
    maximum_human_minutes: int = 120,
    risk_level: OperationsRiskLevel = OperationsRiskLevel.LOW,
    requires_human_approval: bool = False,
) -> OperationsAdviceCandidate:
    return OperationsAdviceCandidate(
        candidate_id="shadow-marker-candidate",
        kind=OperationsAdviceKind.PRIORITIZE_WORK,
        target_division_id="division-engineering",
        action_catalog_id=action_catalog_id,
        summary="SHADOW_MARKER prioritize the bounded reliability repair.",
        rationale="It targets the largest declared operating risk.",
        maximum_cost_minor=maximum_cost_minor,
        maximum_human_minutes=maximum_human_minutes,
        requires_human_approval=requires_human_approval,
        risk_level=risk_level,
        reversibility=OperationsReversibility.REVERSIBLE,
        prerequisite_fact_ids=("fact-1",),
        prediction_ranges=(
            OperationsEstimateRange(
                metric="error_rate",
                lower_bound=0.0,
                upper_bound=0.04,
                unit="ratio",
                horizon_start_ms=100,
                horizon_end_ms=10_000,
                evidence_ref_ids=("operating-signal-1",),
            ),
        ),
        falsification_conditions=("The error rate remains above four percent.",),
        evidence_ref_ids=("operating-signal-1",),
        source_entry_ids=(),
    )


class _StructuredAdviceProvider:
    def __init__(self, candidate: OperationsAdviceCandidate | None = None) -> None:
        self._candidate = candidate or _candidate()

    async def propose(
        self,
        *,
        request,
        recalled_experiences,
        source_turn_index,
        candidate_regime_id,
        candidate_abstract_action,
    ):
        del request, recalled_experiences, source_turn_index
        del candidate_regime_id, candidate_abstract_action
        return (self._candidate,)


async def test_active_context_never_injects_shadow_advice_and_request_is_idempotent() -> None:
    controller = OperationsBrainController(advice_provider=_StructuredAdviceProvider())
    session = _session()
    request = _request()
    first = await controller.build_context_pack(session=session, request=request, generated_at_ms=100)
    repeated = await controller.build_context_pack(session=session, request=request, generated_at_ms=999)

    assert repeated is first
    assert first.wiring_level is WiringLevel.ACTIVE
    assert first.advice.wiring_level is WiringLevel.SHADOW
    assert first.advice.applied is False
    assert "SHADOW_MARKER" in first.advice.candidates[0].summary
    assert "SHADOW_MARKER" not in first.rendered_context
    assert OperationsContextPackSnapshot.from_json(first.to_json()) == first
    with pytest.raises(ValueError, match="unknown fields"):
        OperationsContextPackSnapshot.from_json({**first.to_json(), "unknown": True})
    with pytest.raises(OperationsBrainConflictError, match="reused"):
        await controller.build_context_pack(
            session=session,
            request=replace(request, cycle_id="different-cycle"),
            generated_at_ms=101,
        )


@pytest.mark.parametrize(
    ("candidate", "message"),
    (
        (_candidate(action_catalog_id="catalog:unknown"), "unknown action catalog"),
        (_candidate(maximum_cost_minor=20_000), "exceeds the operating cost bound"),
        (_candidate(maximum_human_minutes=500), "exceeds the human capacity bound"),
        (_candidate(risk_level=OperationsRiskLevel.HIGH), "requires explicit human approval"),
    ),
)
async def test_shadow_advice_is_revalidated_against_autocompany_bounds(
    candidate: OperationsAdviceCandidate,
    message: str,
) -> None:
    controller = OperationsBrainController(advice_provider=_StructuredAdviceProvider(candidate))
    with pytest.raises(ValueError, match=message):
        await controller.build_context_pack(
            session=_session(session_id=f"invalid-{candidate.action_catalog_id}-{candidate.maximum_cost_minor}"),
            request=_request(),
            generated_at_ms=100,
        )


async def test_shadow_advice_prediction_must_fit_operating_window() -> None:
    candidate = _candidate()
    outside = replace(
        candidate,
        prediction_ranges=(
            replace(candidate.prediction_ranges[0], horizon_end_ms=10_001),
        ),
    )
    controller = OperationsBrainController(
        advice_provider=_StructuredAdviceProvider(outside)
    )
    with pytest.raises(ValueError, match="predicts outside"):
        await controller.build_context_pack(
            session=_session(session_id="outside-horizon"),
            request=_request(),
            generated_at_ms=100,
        )


async def test_evidence_lanes_are_isolated_and_only_field_aggregate_enters_pe() -> None:
    controller = OperationsBrainController()
    session = _session()
    context = await controller.build_context_pack(session=session, request=_request(), generated_at_ms=100)
    lanes = (
        ("simulation_result", "simulation", "operating_signal"),
        ("internal_review_result", "internal_review", "internal_review"),
        ("machine_check_result", "machine_check", "machine_audit"),
        ("work_order_progress", "field", "work_order"),
    )
    for index, (kind, evidence_class, role) in enumerate(lanes):
        receipt = await controller.record_outcome(
            session=session,
            report=_report(
                context_pack_id=context.context_pack_id,
                outcome_id=f"lane-{index}",
                outcome_kind=kind,
                evidence_class=evidence_class,
                role=role,
                verdict="inconclusive",
            ),
        )
        assert receipt.learning_route is OperationsOutcomeRoute.MEMORY_AND_EXECUTION_RESULT
        assert receipt.environment_outcome_id == ""

    aggregate = _report(context_pack_id=context.context_pack_id, outcome_id="field-aggregate")
    receipt = await controller.record_outcome(session=session, report=aggregate)
    replayed = await controller.record_outcome(session=session, report=aggregate)
    assert replayed is receipt
    assert (
        receipt.learning_route
        is OperationsOutcomeRoute.FIELD_OPERATION_PE_MEMORY_AND_EXECUTION_RESULT
    )
    assert receipt.environment_outcome_id
    assert receipt.work_order_ref == _WORK_ORDER_REF
    assert receipt.source_advice_applied is False
    assert OperationsOutcomeReceipt.from_json(receipt.to_json()) == receipt
    tampered_receipt = receipt.to_json()
    tampered_receipt["memory_persisted"] = not receipt.memory_persisted
    with pytest.raises(ValueError, match="content_sha256"):
        OperationsOutcomeReceipt.from_json(tampered_receipt)

    with pytest.raises(OperationsBrainSettlementPendingError):
        await controller.record_outcome(
            session=session,
            report=_report(
                context_pack_id=context.context_pack_id,
                outcome_id="second-field-aggregate",
            ),
        )


async def test_next_context_settles_field_outcome_with_pe_and_evidence_lineage() -> None:
    controller = OperationsBrainController()
    session = _session()
    context = await controller.build_context_pack(session=session, request=_request(), generated_at_ms=100)
    report = _report(
        context_pack_id=context.context_pack_id,
        outcome_id="settled-field-outcome",
        detail="The work order reduced the error rate and completed inside its bounded window.",
    )
    receipt = await controller.record_outcome(session=session, report=report)

    next_context = await controller.build_context_pack(
        session=session,
        request=_request(request_id="request-2", decision_id="decision-2", cycle_id="cycle-2"),
        generated_at_ms=300,
    )
    assert next_context.settled_outcome_ids == (receipt.environment_outcome_id,)
    assert next_context.settled_evidence_ref_ids == tuple(
        reference.ref_id for reference in report.evidence_refs
    )
    assert receipt.memory_entry_id in next_context.source_entry_ids
    assert "reduced the error rate" in next_context.rendered_context
    assert _WORK_ORDER_REF in next_context.rendered_context
    assert next_context.pe_bootstrap is False
    assert next_context.pe_magnitude > 0.0

    conflicting = replace(report, detail="Different immutable result under the same id.")
    with pytest.raises(OperationsBrainConflictError, match="reused"):
        await controller.record_outcome(session=session, report=conflicting)


async def test_outcome_requires_same_live_session_and_latest_pe_lineage() -> None:
    controller = OperationsBrainController()
    first = _session(session_id="operations-first")
    second = _session(session_id="operations-second")
    context = await controller.build_context_pack(session=first, request=_request(), generated_at_ms=100)
    with pytest.raises(OperationsBrainLineageError, match="this live session"):
        await controller.record_outcome(
            session=second,
            report=_report(context_pack_id=context.context_pack_id, outcome_id="cross-session"),
        )

    latest = await controller.build_context_pack(
        session=first,
        request=_request(request_id="request-latest", decision_id="decision-latest"),
        generated_at_ms=150,
    )
    assert latest.context_pack_id != context.context_pack_id
    with pytest.raises(OperationsBrainLineageError, match="latest"):
        await controller.record_outcome(
            session=first,
            report=_report(context_pack_id=context.context_pack_id, outcome_id="stale-field"),
        )


async def test_identity_scoped_outcome_is_recalled_across_sessions(tmp_path) -> None:
    identity = StaticIdentityProvider(
        identity=UserIdentity(user_id="autocompany-1", scope_key="autocompany-1")
    )
    config = LifeformConfig(brain_config=BrainConfig(memory_scope_root_dir=str(tmp_path)))
    first_lifeform = build_operations_lifeform(config=config, identity_provider=identity)
    first_session = first_lifeform.create_session(session_id="operations-session-a")
    first_controller = OperationsBrainController()
    context = await first_controller.build_context_pack(
        session=first_session,
        request=_request(),
        generated_at_ms=100,
    )
    receipt = await first_controller.record_outcome(
        session=first_session,
        report=_report(
            context_pack_id=context.context_pack_id,
            outcome_id="persistent-outcome",
            outcome_kind="work_order_progress",
            role="work_order",
            verdict="mixed",
            detail="The repair started, but a declared dependency remains blocked.",
        ),
    )
    assert receipt.memory_persisted is True

    second_lifeform = build_operations_lifeform(config=config, identity_provider=identity)
    second_session = second_lifeform.create_session(session_id="operations-session-b")
    recalled = await OperationsBrainController().build_context_pack(
        session=second_session,
        request=_request(request_id="request-b", decision_id="decision-b", cycle_id="cycle-b"),
        generated_at_ms=300,
    )
    assert receipt.memory_entry_id in recalled.source_entry_ids
    assert "declared dependency remains blocked" in recalled.rendered_context

    other_company = await OperationsBrainController().build_context_pack(
        session=second_lifeform.create_session(session_id="operations-session-c"),
        request=_request(
            request_id="request-c",
            decision_id="decision-c",
            cycle_id="cycle-c",
            company_id="company-2",
        ),
        generated_at_ms=400,
    )
    assert receipt.memory_entry_id not in other_company.source_entry_ids
    assert "declared dependency remains blocked" not in other_company.rendered_context


def test_operations_controller_uses_facades_not_owner_stores() -> None:
    package_root = Path(__file__).parents[1] / "src" / "lifeform_domain_operations"
    forbidden_attributes = {
        "runner",
        "memory_store",
        "semantic_state_store",
        "prediction_error_module",
    }
    offenders: list[str] = []
    for path in package_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in forbidden_attributes:
                offenders.append(f"{path.name}:{node.lineno}:{node.attr}")
    assert offenders == []
