from __future__ import annotations

import pytest

from volvence_zero.agent.learned_active_gate import (
    LearnedActiveEvidence,
    LearnedBackendComponent,
    evaluate_learned_active_candidate,
)


def _evidence(component: LearnedBackendComponent, **overrides) -> LearnedActiveEvidence:
    payload = dict(
        component=component,
        real_trace_turns=500,
        validation_delta=0.03,
        strict_eta_gate_passed=True,
        pe_off_control_direction_correct=True,
        eta_off_control_direction_correct=True,
        rollback_drill_passed=True,
        latency_slo_ok=True,
        safety_gate_ok=True,
    )
    payload.update(overrides)
    return LearnedActiveEvidence(**payload)


def test_runtime_candidate_requires_real_trace_controls_and_rollback() -> None:
    verdict = evaluate_learned_active_candidate(
        _evidence(
            LearnedBackendComponent.TEMPORAL_RUNTIME,
            real_trace_turns=120,
            validation_delta=0.01,
            rollback_drill_passed=False,
        )
    )
    assert verdict.eligible is False
    assert "real_trace_turns<500" in verdict.missing_gates
    assert "validation_delta<0.02" in verdict.missing_gates
    assert "rollback_drill" in verdict.missing_gates


def test_ssl_and_internal_rl_enforce_sequential_promotion() -> None:
    ssl = evaluate_learned_active_candidate(
        _evidence(LearnedBackendComponent.TEMPORAL_SSL)
    )
    assert ssl.eligible is False
    assert "runtime_active_first" in ssl.missing_gates

    internal = evaluate_learned_active_candidate(
        _evidence(
            LearnedBackendComponent.INTERNAL_RL,
            prior_runtime_active=True,
            prior_ssl_active=False,
            internal_rl_no_reward_leakage=False,
        )
    )
    assert internal.eligible is False
    assert "ssl_active_first" in internal.missing_gates
    assert "reward_leakage" in internal.missing_gates


def test_internal_rl_candidate_can_pass_when_all_gates_close() -> None:
    verdict = evaluate_learned_active_candidate(
        _evidence(
            LearnedBackendComponent.INTERNAL_RL,
            prior_runtime_active=True,
            prior_ssl_active=True,
            internal_rl_no_reward_leakage=True,
        )
    )
    assert verdict.eligible is True
    assert verdict.missing_gates == ()


def test_cms_candidate_requires_retention_and_absorption() -> None:
    verdict = evaluate_learned_active_candidate(
        _evidence(
            LearnedBackendComponent.CMS_TORCH,
            cms_retention_non_degrading=False,
            cms_absorption_improved=False,
        )
    )
    assert verdict.eligible is False
    assert "cms_retention" in verdict.missing_gates
    assert "cms_absorption" in verdict.missing_gates


def test_evidence_rejects_negative_turn_count() -> None:
    with pytest.raises(ValueError, match="real_trace_turns"):
        _evidence(LearnedBackendComponent.TEMPORAL_RUNTIME, real_trace_turns=-1)
