from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from scripts.evaluate_learned_backend_promotion import main as evaluate_promotion_main
from volvence_zero.agent.learned_active_gate import (
    LEARNED_BACKEND_PROMOTION_ORDER,
    LearnedActiveEvidence,
    LearnedBackendComponent,
    evaluate_learned_active_candidate,
    evaluate_learned_active_chain,
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


def _terminal_evidence() -> tuple[LearnedActiveEvidence, ...]:
    return tuple(_evidence(component) for component in LEARNED_BACKEND_PROMOTION_ORDER)


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


def test_terminal_candidate_can_pass_before_staged_production_promotion() -> None:
    verdict = evaluate_learned_active_chain(_terminal_evidence())

    assert verdict.terminal_candidate_ready is True
    assert verdict.production_terminal_ready is False
    assert verdict.next_component is LearnedBackendComponent.TEMPORAL_RUNTIME
    assert verdict.next_component_eligible is True
    assert all(report.eligible for report in verdict.terminal_reports)
    assert "production_promotion_incomplete" in verdict.blocking_reasons


def test_chain_recommends_only_the_next_component_in_order() -> None:
    verdict = evaluate_learned_active_chain(
        _terminal_evidence(),
        active_components=(LearnedBackendComponent.TEMPORAL_RUNTIME,),
    )

    assert verdict.next_component is LearnedBackendComponent.TEMPORAL_SSL
    assert verdict.next_component_eligible is True
    assert verdict.production_terminal_ready is False


def test_production_terminal_requires_full_active_prefix() -> None:
    verdict = evaluate_learned_active_chain(
        _terminal_evidence(),
        active_components=LEARNED_BACKEND_PROMOTION_ORDER,
    )

    assert verdict.terminal_candidate_ready is True
    assert verdict.production_terminal_ready is True
    assert verdict.next_component is None
    assert verdict.next_component_eligible is False
    assert verdict.blocking_reasons == ()


def test_terminal_chain_blocks_failed_component_gate() -> None:
    evidence = list(_terminal_evidence())
    evidence[2] = _evidence(
        LearnedBackendComponent.INTERNAL_RL,
        internal_rl_no_reward_leakage=False,
    )
    verdict = evaluate_learned_active_chain(
        evidence,
        active_components=(
            LearnedBackendComponent.TEMPORAL_RUNTIME,
            LearnedBackendComponent.TEMPORAL_SSL,
        ),
    )

    assert verdict.terminal_candidate_ready is False
    assert verdict.production_terminal_ready is False
    assert verdict.next_component is LearnedBackendComponent.INTERNAL_RL
    assert verdict.next_component_eligible is False
    assert "internal_rl_backend:reward_leakage" in verdict.blocking_reasons


def test_failed_terminal_gate_blocks_even_an_earlier_next_component() -> None:
    evidence = list(_terminal_evidence())
    evidence[2] = _evidence(
        LearnedBackendComponent.INTERNAL_RL,
        internal_rl_no_reward_leakage=False,
    )
    verdict = evaluate_learned_active_chain(evidence)

    assert verdict.next_component is LearnedBackendComponent.TEMPORAL_RUNTIME
    assert verdict.next_component_eligible is False
    assert (
        "next_component_blocked:temporal_runtime_backend"
        in verdict.blocking_reasons
    )


def test_terminal_chain_rejects_non_prefix_active_state() -> None:
    with pytest.raises(ValueError, match="ordered promotion prefix"):
        evaluate_learned_active_chain(
            _terminal_evidence(),
            active_components=(
                LearnedBackendComponent.TEMPORAL_RUNTIME,
                LearnedBackendComponent.INTERNAL_RL,
            ),
        )


def test_terminal_chain_requires_exactly_one_row_per_component() -> None:
    evidence = _terminal_evidence()
    with pytest.raises(ValueError, match="terminal evidence incomplete"):
        evaluate_learned_active_chain(evidence[:-1])
    with pytest.raises(ValueError, match="duplicate learned backend evidence"):
        evaluate_learned_active_chain(evidence + (evidence[0],))


def test_promotion_report_recommends_only_one_production_flip(tmp_path: Path) -> None:
    artifact_path = tmp_path / "evidence.json"
    report_path = tmp_path / "report.json"
    artifact_path.write_text(
        json.dumps(
            {
                "learned_active_gate": {
                    "evidence": [
                        asdict(evidence) for evidence in _terminal_evidence()
                    ],
                    "active_components": [],
                    "validation_gate_version": "v1",
                    "validation_delta_v2": None,
                }
            }
        ),
        encoding="utf-8",
    )

    assert (
        evaluate_promotion_main(
            ["--artifact", str(artifact_path), "--output", str(report_path)]
        )
        == 0
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    recommendations = [
        row["recommended_env"] for row in report["reports"] if row["recommended_env"]
    ]

    assert report["terminal_candidate_ready"] is True
    assert report["production_terminal_ready"] is False
    assert report["staged_gate"]["next_component"] == "temporal_runtime_backend"
    assert recommendations == ["VZ_TEMPORAL_RUNTIME_BACKEND=active"]
