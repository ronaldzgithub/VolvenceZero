"""Workstream F tests: ant-active-evidence lane reuses the substrate-agnostic gate."""

from __future__ import annotations

from volvence_zero.agent.learned_active_gate import LearnedBackendComponent

from volvence_ant.evidence import collect_ant_active_evidence


async def test_evidence_lane_runs_and_blocks_at_toy_scale() -> None:
    bundle = await collect_ant_active_evidence(
        trace_turns=4, behavioral_ticks=12, seed=0, with_latent=False
    )
    # gate must be conservative: 4 real traces cannot promote (needs 500)
    assert not bundle.verdict.eligible
    assert "real_trace_turns<500" in bundle.verdict.missing_gates
    assert bundle.evidence.component is LearnedBackendComponent.TEMPORAL_RUNTIME


async def test_safety_and_latency_gates_pass() -> None:
    bundle = await collect_ant_active_evidence(
        trace_turns=4, behavioral_ticks=12, seed=0, with_latent=False
    )
    # the hardwired escape reflex + per-turn latency should always pass
    assert bundle.evidence.safety_gate_ok
    assert bundle.evidence.latency_slo_ok


async def test_rollback_drill_is_reproducible() -> None:
    bundle = await collect_ant_active_evidence(
        trace_turns=2,
        behavioral_ticks=8,
        seed=1,
        with_latent=False,
        rollback_drill_passed=True,
    )
    # Full orchestrator supplies this only after apply/rollback verification.
    assert bundle.evidence.rollback_drill_passed
