from __future__ import annotations

import asyncio

from volvence_zero.agent import AgentSessionRunner, default_active_runner


def test_agent_session_runner_executes_single_turn():
    runner = default_active_runner()
    result = asyncio.run(runner.run_turn("I need help organizing my plan and I also feel overwhelmed."))

    assert result.acceptance_passed is True
    assert result.wave_id == "wave-1"
    assert "evaluation" in result.active_snapshots
    assert result.active_regime is not None
    assert result.response.text
    assert result.event_count > 0


def test_agent_session_runner_reuses_session_memory_across_turns():
    runner = AgentSessionRunner(session_id="s1")
    first = asyncio.run(runner.run_turn("Remember that I prefer calm, reflective collaboration."))
    second = asyncio.run(runner.run_turn("Can you help me continue that plan from before?"))

    assert first.wave_id == "wave-1"
    assert second.wave_id == "wave-2"
    assert len(second.active_snapshots["memory"].value.retrieved_entries) >= 1


def test_agent_session_runner_exposes_temporal_and_regime_views():
    runner = default_active_runner()
    result = asyncio.run(runner.run_turn("Please guide me carefully through a difficult decision."))

    assert result.active_regime is not None
    assert result.active_abstract_action is not None
    assert result.metacontroller_state is not None
    assert result.metacontroller_state.mode == "learned-lite"
    assert isinstance(result.evaluation_alerts, tuple)
    assert result.response.regime_id == result.active_regime
    assert result.response.abstract_action == result.active_abstract_action


def test_agent_session_runner_returns_user_visible_response():
    runner = default_active_runner()
    result = asyncio.run(runner.run_turn("I feel tense and I need a careful response."))

    assert result.response.text
    assert result.response.rationale
