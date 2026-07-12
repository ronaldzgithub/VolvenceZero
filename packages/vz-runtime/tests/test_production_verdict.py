from __future__ import annotations

import pytest

from volvence_zero.agent.production_verdict import (
    ClaimStatus,
    ComponentDecision,
    ComponentGateEvidence,
    FinalPositioning,
    ProductionEvidenceSummary,
    evaluate_production_verdict,
)


def _gate(component_id: str, **overrides) -> ComponentGateEvidence:
    payload = dict(
        component_id=component_id,
        gate_passed=True,
        rollback_drill_passed=True,
        safety_gate_passed=True,
        retain_evidence_present=True,
    )
    payload.update(overrides)
    return ComponentGateEvidence(**payload)


def _summary(**overrides) -> ProductionEvidenceSummary:
    payload = dict(
        pipeline_gt_raw=ClaimStatus.RETAIN,
        gt_standard_layers=ClaimStatus.RETAIN,
        component_causal=ClaimStatus.RETAIN,
        training_adds_value=ClaimStatus.RETAIN,
        heldout_multi_seed_stable=ClaimStatus.RETAIN,
        relationship_continuity=ClaimStatus.RETAIN,
        longitudinal_20_session=ClaimStatus.RETAIN,
        human_anchor=ClaimStatus.RETAIN,
        component_gates=(
            _gate("temporal_runtime_backend"),
            _gate("internal_rl_backend", retain_evidence_present=False),
            _gate("autonomous_loop", safety_gate_passed=False),
        ),
    )
    payload.update(overrides)
    return ProductionEvidenceSummary(**payload)


def test_first_stage_retained_requires_all_claims_retain() -> None:
    verdict = evaluate_production_verdict(_summary())
    assert verdict.positioning is FinalPositioning.FIRST_STAGE_RETAINED
    decisions = {item.component_id: item.decision for item in verdict.component_verdicts}
    assert decisions["temporal_runtime_backend"] is ComponentDecision.ACTIVE
    assert decisions["internal_rl_backend"] is ComponentDecision.SHADOW
    assert decisions["autonomous_loop"] is ComponentDecision.DISABLED


def test_core_failure_with_relationship_retain_becomes_product_companion() -> None:
    verdict = evaluate_production_verdict(
        _summary(component_causal=ClaimStatus.FAIL)
    )
    assert verdict.positioning is FinalPositioning.PRODUCT_COMPANION_RETAINED
    assert "component_causal" in verdict.missing_or_failed


def test_core_failure_without_relationship_retain_becomes_platform_only() -> None:
    verdict = evaluate_production_verdict(
        _summary(
            pipeline_gt_raw=ClaimStatus.FAIL,
            relationship_continuity=ClaimStatus.WEAK,
        )
    )
    assert verdict.positioning is FinalPositioning.ARCHITECTURE_PLATFORM_ONLY


def test_missing_claims_are_inconclusive_not_weak_positive() -> None:
    verdict = evaluate_production_verdict(
        _summary(human_anchor=ClaimStatus.INSUFFICIENT)
    )
    assert verdict.positioning is FinalPositioning.INCONCLUSIVE
    assert "human_anchor" in verdict.missing_or_failed


def test_component_gate_rejects_empty_component_id() -> None:
    with pytest.raises(ValueError, match="component_id"):
        _gate("")
