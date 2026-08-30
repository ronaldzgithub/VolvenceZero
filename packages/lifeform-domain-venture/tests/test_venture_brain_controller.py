from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from lifeform_core import LifeformConfig
from lifeform_domain_venture import (
    VentureAdviceCandidate,
    VentureAdviceKind,
    VentureBrainConflictError,
    VentureBrainController,
    VentureBrainLineageError,
    VentureBrainSettlementPendingError,
    VentureContextPackSnapshot,
    VentureContextRequest,
    VentureEstimateRange,
    VentureOutcomeReport,
    VentureOutcomeReceipt,
    VentureOutcomeRoute,
    build_venture_lifeform,
)
from volvence_zero.brain import BrainConfig
from volvence_zero.memory import StaticIdentityProvider, UserIdentity
from volvence_zero.runtime import WiringLevel


def _evidence(
    *,
    ref_id: str,
    evidence_class: str,
    role: str,
    observed_at_ms: int,
    digest_char: str,
) -> dict[str, object]:
    return {
        "ref_id": ref_id,
        "evidence_class": evidence_class,
        "role": role,
        "locator": f"foundry://evidence/{ref_id}",
        "content_sha256": digest_char * 64,
        "observed_at_ms": observed_at_ms,
    }


def _request(
    *,
    request_id: str = "request-1",
    decision_id: str = "decision-1",
    cycle_id: str = "cycle-1",
) -> VentureContextRequest:
    return VentureContextRequest.from_json(
        {
            "schema_version": "venture-context-request.v1",
            "request_id": request_id,
            "portfolio_id": "portfolio-1",
            "cycle_id": cycle_id,
            "venture_id": "venture-1",
            "decision_id": decision_id,
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
                    "description": "Use only the approved reversible test budget.",
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
            "evidence_refs": [
                _evidence(
                    ref_id="demand-1",
                    evidence_class="field",
                    role="demand_signal",
                    observed_at_ms=100,
                    digest_char="a",
                )
            ],
        }
    )


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


def _report(
    *,
    context_pack_id: str,
    outcome_id: str = "outcome-1",
    decision_id: str = "decision-1",
    outcome_kind: str = "field_experiment_result",
    evidence_class: str = "field",
    role: str = "field_observation",
    verdict: str = "favorable",
    detail: str = "A customer completed the paid field workflow.",
) -> VentureOutcomeReport:
    if outcome_kind == "field_experiment_result":
        revenue = 5_000
        costs = _costs(
            acquisition_minor=100,
            model_minor=200,
            data_minor=100,
            human_review_minor=200,
            delivery_minor=300,
            support_minor=100,
            risk_reserve_minor=500,
        )
        refund = 100
        customer_result = "positive"
        elapsed_ms = 86_400_000
        risk_level = "low"
    elif outcome_kind == "customer_outcome":
        revenue = 0
        costs = _costs()
        refund = 0
        customer_result = "mixed"
        elapsed_ms = 1_000
        risk_level = "medium"
    else:
        revenue = 0
        costs = _costs()
        refund = 0
        customer_result = "not_observed"
        elapsed_ms = 0
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
    if outcome_kind == "field_experiment_result":
        for suffix, dimension_role, digest_char in (
            ("customer", "customer_outcome", "c"),
            ("payment", "payment", "d"),
            ("cost", "cost", "e"),
            ("refund", "refund", "f"),
        ):
            evidence_refs.append(
                _evidence(
                    ref_id=f"evidence-{outcome_id}-{suffix}",
                    evidence_class=evidence_class,
                    role=dimension_role,
                    observed_at_ms=200,
                    digest_char=digest_char,
                )
            )
    return VentureOutcomeReport.from_json(
        {
            "schema_version": "venture-outcome-report.v1",
            "outcome_id": outcome_id,
            "context_pack_id": context_pack_id,
            "decision_id": decision_id,
            "decision": "continue",
            "outcome_kind": outcome_kind,
            "evidence_class": evidence_class,
            "verdict": verdict,
            "summary": "Foundry supplied a typed venture result.",
            "detail": detail,
            "observed_at_ms": 200,
            "evidence_refs": evidence_refs,
            "commercial_outcome": {
                "customer_result": customer_result,
                "currency": "USD",
                "realized_revenue_minor": revenue,
                "realized_costs": costs,
                "refund_minor": refund,
                "realized_net_value_minor": revenue - sum(costs.values()) - refund,
                "elapsed_ms": elapsed_ms,
                "risk_level": risk_level,
                "reversibility": "reversible",
            },
        }
    )


def _session(*, session_id: str = "venture-session"):
    return build_venture_lifeform().create_session(session_id=session_id)


class _StructuredAdviceProvider:
    async def propose(
        self,
        *,
        request,
        recalled_experiences,
        source_turn_index,
        candidate_regime_id,
        candidate_abstract_action,
    ):
        del recalled_experiences, source_turn_index, candidate_regime_id, candidate_abstract_action
        return (
            VentureAdviceCandidate(
                candidate_id="shadow-marker-candidate",
                kind=VentureAdviceKind.EXPERIMENT,
                summary="SHADOW_MARKER run a reversible paid-intent test.",
                rationale="It targets the widest current uncertainty.",
                prediction_ranges=(
                    VentureEstimateRange(
                        metric="qualified_customer_count",
                        lower_bound=1,
                        upper_bound=3,
                        unit="customers",
                        horizon_start_ms=request.resource_window.starts_at_ms,
                        horizon_end_ms=request.resource_window.ends_at_ms,
                        evidence_ref_ids=("demand-1",),
                    ),
                ),
                falsification_conditions=("No qualified customer accepts the stated price.",),
                evidence_ref_ids=("demand-1",),
                source_entry_ids=(),
            ),
        )


async def test_active_context_never_injects_shadow_advice_and_request_is_idempotent() -> None:
    controller = VentureBrainController(advice_provider=_StructuredAdviceProvider())
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
    assert VentureContextPackSnapshot.from_json(first.to_json()) == first
    with pytest.raises(ValueError, match="unknown fields"):
        VentureContextPackSnapshot.from_json({**first.to_json(), "unknown": True})
    with pytest.raises(VentureBrainConflictError, match="reused"):
        await controller.build_context_pack(
            session=session,
            request=replace(request, cycle_id="different-cycle"),
            generated_at_ms=101,
        )


async def test_evidence_lanes_are_isolated_and_only_field_aggregate_enters_pe() -> None:
    controller = VentureBrainController()
    session = _session()
    context = await controller.build_context_pack(session=session, request=_request(), generated_at_ms=100)
    lanes = (
        ("simulation_result", "simulation", "experiment"),
        ("internal_review_result", "internal_review", "internal_review"),
        ("machine_check_result", "machine_check", "machine_audit"),
        ("customer_outcome", "field", "customer_outcome"),
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
        assert receipt.learning_route is VentureOutcomeRoute.MEMORY_AND_EXECUTION_RESULT
        assert receipt.environment_outcome_id == ""

    aggregate = _report(context_pack_id=context.context_pack_id, outcome_id="field-aggregate")
    receipt = await controller.record_outcome(session=session, report=aggregate)
    replayed = await controller.record_outcome(session=session, report=aggregate)
    assert replayed is receipt
    assert receipt.learning_route is VentureOutcomeRoute.FIELD_PE_MEMORY_AND_EXECUTION_RESULT
    assert receipt.environment_outcome_id
    assert receipt.source_advice_applied is False
    assert VentureOutcomeReceipt.from_json(receipt.to_json()) == receipt
    tampered_receipt = receipt.to_json()
    tampered_receipt["memory_persisted"] = not receipt.memory_persisted
    with pytest.raises(ValueError, match="content_sha256"):
        VentureOutcomeReceipt.from_json(tampered_receipt)

    with pytest.raises(VentureBrainSettlementPendingError):
        await controller.record_outcome(
            session=session,
            report=_report(context_pack_id=context.context_pack_id, outcome_id="second-field-aggregate"),
        )


async def test_next_context_settles_field_outcome_with_pe_and_evidence_lineage() -> None:
    controller = VentureBrainController()
    session = _session()
    context = await controller.build_context_pack(session=session, request=_request(), generated_at_ms=100)
    report = _report(
        context_pack_id=context.context_pack_id,
        outcome_id="settled-field-outcome",
        detail="Customer paid and completed the bounded field experiment.",
    )
    receipt = await controller.record_outcome(session=session, report=report)

    next_context = await controller.build_context_pack(
        session=session,
        request=_request(request_id="request-2", decision_id="decision-2", cycle_id="cycle-2"),
        generated_at_ms=300,
    )
    assert next_context.settled_outcome_ids == (receipt.environment_outcome_id,)
    assert next_context.settled_evidence_ref_ids == tuple(reference.ref_id for reference in report.evidence_refs)
    assert receipt.memory_entry_id in next_context.source_entry_ids
    assert "paid and completed" in next_context.rendered_context
    assert next_context.pe_bootstrap is False
    assert next_context.pe_magnitude > 0.0

    conflicting = replace(report, detail="Different immutable result under the same id.")
    with pytest.raises(VentureBrainConflictError, match="reused"):
        await controller.record_outcome(session=session, report=conflicting)


async def test_outcome_requires_same_live_session_and_latest_pe_lineage() -> None:
    controller = VentureBrainController()
    first = _session(session_id="venture-first")
    second = _session(session_id="venture-second")
    context = await controller.build_context_pack(session=first, request=_request(), generated_at_ms=100)
    with pytest.raises(VentureBrainLineageError, match="this live session"):
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
    with pytest.raises(VentureBrainLineageError, match="latest"):
        await controller.record_outcome(
            session=first,
            report=_report(context_pack_id=context.context_pack_id, outcome_id="stale-field"),
        )


async def test_identity_scoped_outcome_is_recalled_across_sessions(tmp_path) -> None:
    identity = StaticIdentityProvider(identity=UserIdentity(user_id="foundry-1", scope_key="foundry-1"))
    config = LifeformConfig(brain_config=BrainConfig(memory_scope_root_dir=str(tmp_path)))
    first_lifeform = build_venture_lifeform(config=config, identity_provider=identity)
    first_session = first_lifeform.create_session(session_id="venture-session-a")
    first_controller = VentureBrainController()
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
            outcome_kind="customer_outcome",
            role="customer_outcome",
            verdict="mixed",
            detail="Customer completed onboarding but rejected the renewal price.",
        ),
    )
    assert receipt.memory_persisted is True

    second_lifeform = build_venture_lifeform(config=config, identity_provider=identity)
    second_session = second_lifeform.create_session(session_id="venture-session-b")
    recalled = await VentureBrainController().build_context_pack(
        session=second_session,
        request=_request(request_id="request-b", decision_id="decision-b", cycle_id="cycle-b"),
        generated_at_ms=300,
    )
    assert receipt.memory_entry_id in recalled.source_entry_ids
    assert "rejected the renewal price" in recalled.rendered_context


async def test_content_position_policy_settles_exact_field_credit() -> None:
    controller = VentureBrainController()
    session = _session(session_id="venture-content-policy")
    first = await controller.build_context_pack(
        session=session,
        request=_request(),
        generated_at_ms=1_000,
    )
    for index in range(2):
        await controller.record_outcome(
            session=session,
            report=_report(
                context_pack_id=first.context_pack_id,
                outcome_id=f"policy-memory-{index}",
                outcome_kind="customer_outcome",
                role="customer_outcome",
                verdict="mixed",
                detail=f"Memory-only customer observation {index}",
            ),
        )

    policy_pack = await controller.build_context_pack(
        session=session,
        request=_request(
            request_id="policy-request-2",
            decision_id="policy-decision-2",
            cycle_id="policy-cycle-2",
        ),
        generated_at_ms=1_100,
    )
    decision = policy_pack.content_policy_decision
    assert decision is not None
    assert policy_pack.source_entry_ids == decision.output_entry_ids
    with pytest.raises(
        ValueError,
        match="source_entry_ids must match content policy output order",
    ):
        replace(
            policy_pack,
            recalled_experiences=tuple(
                reversed(policy_pack.recalled_experiences)
            ),
            source_entry_ids=tuple(reversed(policy_pack.source_entry_ids)),
        )
    assert VentureContextPackSnapshot.from_json(policy_pack.to_json()) == policy_pack
    receipt = await controller.record_outcome(
        session=session,
        report=_report(
            context_pack_id=policy_pack.context_pack_id,
            outcome_id="policy-field-result",
            decision_id="policy-decision-2",
            detail="The policy-positioned Context Pack preceded this field aggregate.",
        ),
    )
    assert receipt.source_content_policy_decision_id == decision.policy_decision_id
    assert receipt.content_policy_action_applied is True

    settled = await controller.build_context_pack(
        session=session,
        request=_request(
            request_id="policy-request-3",
            decision_id="policy-decision-3",
            cycle_id="policy-cycle-3",
        ),
        generated_at_ms=1_200,
    )

    assert len(settled.settled_policy_credits) == 1
    assert len(settled.policy_updates) == 1
    assert (
        settled.settled_policy_credits[0].policy_decision_id
        == decision.policy_decision_id
    )
    assert settled.policy_updates[0].update_count == 1


async def test_content_position_policy_can_be_disabled() -> None:
    controller = VentureBrainController(
        content_policy_wiring_level=WiringLevel.DISABLED
    )
    session = _session(session_id="venture-content-policy-disabled")
    first = await controller.build_context_pack(
        session=session,
        request=_request(),
        generated_at_ms=2_000,
    )
    for index in range(2):
        await controller.record_outcome(
            session=session,
            report=_report(
                context_pack_id=first.context_pack_id,
                outcome_id=f"disabled-memory-{index}",
                outcome_kind="customer_outcome",
                role="customer_outcome",
                verdict="mixed",
            ),
        )
    recalled = await controller.build_context_pack(
        session=session,
        request=_request(
            request_id="disabled-request-2",
            decision_id="disabled-decision-2",
            cycle_id="disabled-cycle-2",
        ),
        generated_at_ms=2_100,
    )

    assert recalled.content_policy_wiring_level is WiringLevel.DISABLED
    assert recalled.content_policy_decision is None
    assert len(recalled.source_entry_ids) == 2


async def test_updated_content_policy_checkpoint_restores_across_sessions(
    tmp_path,
) -> None:
    identity = StaticIdentityProvider(
        identity=UserIdentity(user_id="foundry-policy", scope_key="foundry-policy")
    )
    config = LifeformConfig(
        brain_config=BrainConfig(memory_scope_root_dir=str(tmp_path))
    )
    first_lifeform = build_venture_lifeform(
        config=config,
        identity_provider=identity,
    )
    first_session = first_lifeform.create_session(session_id="policy-session-a")
    first_controller = VentureBrainController()
    first = await first_controller.build_context_pack(
        session=first_session,
        request=_request(),
        generated_at_ms=3_000,
    )
    for index in range(2):
        await first_controller.record_outcome(
            session=first_session,
            report=_report(
                context_pack_id=first.context_pack_id,
                outcome_id=f"restored-memory-{index}",
                outcome_kind="customer_outcome",
                role="customer_outcome",
                verdict="mixed",
            ),
        )
    policy_pack = await first_controller.build_context_pack(
        session=first_session,
        request=_request(
            request_id="restore-request-2",
            decision_id="restore-decision-2",
            cycle_id="restore-cycle-2",
        ),
        generated_at_ms=3_100,
    )
    assert policy_pack.content_policy_decision is not None
    await first_controller.record_outcome(
        session=first_session,
        report=_report(
            context_pack_id=policy_pack.context_pack_id,
            outcome_id="restore-field-result",
            decision_id="restore-decision-2",
        ),
    )
    settled = await first_controller.build_context_pack(
        session=first_session,
        request=_request(
            request_id="restore-request-3",
            decision_id="restore-decision-3",
            cycle_id="restore-cycle-3",
        ),
        generated_at_ms=3_200,
    )
    assert settled.policy_updates[0].update_count == 1

    second_lifeform = build_venture_lifeform(
        config=config,
        identity_provider=identity,
    )
    second_session = second_lifeform.create_session(session_id="policy-session-b")
    restored = await VentureBrainController().build_context_pack(
        session=second_session,
        request=_request(
            request_id="restore-request-b",
            decision_id="restore-decision-b",
            cycle_id="restore-cycle-b",
        ),
        generated_at_ms=3_300,
    )

    assert restored.content_policy_decision is not None
    assert restored.content_policy_decision.checkpoint_update_count == 1


def test_venture_controller_uses_facades_not_owner_stores() -> None:
    package_root = Path(__file__).parents[1] / "src" / "lifeform_domain_venture"
    forbidden_attributes = {"runner", "memory_store", "semantic_state_store", "prediction_error_module"}
    offenders: list[str] = []
    for path in package_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in forbidden_attributes:
                offenders.append(f"{path.name}:{node.lineno}:{node.attr}")
    assert offenders == []
