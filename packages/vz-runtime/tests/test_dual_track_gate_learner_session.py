"""W1.A runtime acceptance: the dual-track gate learner survives per-turn
module rebuilds and accumulates PE-scored updates across a session."""

from __future__ import annotations

from volvence_zero.agent.session import AgentSessionRunner
from volvence_zero.dual_track import DualTrackSnapshot


async def test_session_gate_learner_accumulates_pe_scored_updates() -> None:
    runner = AgentSessionRunner(rare_heavy_enabled=False)
    for text in (
        "Help me lay out the deployment steps for tonight.",
        "That plan slipped; I feel a bit overwhelmed now.",
        "Okay, let's refocus on the next concrete step.",
    ):
        result = await runner.run_turn(text)

    dual_track = result.active_snapshots.get("dual_track") or result.shadow_snapshots.get(
        "dual_track"
    )
    assert dual_track is not None
    assert isinstance(dual_track.value, DualTrackSnapshot)
    gate = dual_track.value.learned_gate_shadow
    assert gate is not None
    # The session-held learner (not the fixed-prior fallback) produced it.
    assert "online-SGD" in gate.description
    assert abs(gate.world_weight + gate.self_weight - 1.0) <= 1e-6
    # PE realized outcomes settled at least one prior-turn gate candidate.
    learner = runner.dual_track_gate_learner
    assert learner.update_count >= 1
    readout = learner.readout()
    assert readout.update_count == learner.update_count
    assert "report-only" in readout.description
