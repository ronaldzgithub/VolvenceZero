"""Evidence gates for companion state sensitivity and learning claims.

These tests intentionally separate what is proven today from what is
still a roadmap item:

* state sensitivity within one turn is expected to pass;
* within-session adaptation across a low-mood episode is expected to pass;
* cross-session retention is marked xfail because default sessions do not
  yet share memory / semantic state.
"""

from __future__ import annotations

import pytest


async def test_companion_state_sensitivity_changes_interlocutor_readout() -> None:
    """Same companion, different user state -> different readout axes."""

    from lifeform_domain_emogpt import build_companion_lifeform

    lifeform = build_companion_lifeform()
    task_session = lifeform.create_session(session_id="evidence-state-task")
    emotional_session = lifeform.create_session(session_id="evidence-state-emotional")

    task_result = await task_session.run_turn(
        "Can you help me draft a concise email declining a meeting invite and suggesting async instead?"
    )
    emotional_result = await emotional_session.run_turn(
        "I have been feeling really stuck and heavy lately, and I mostly need to feel heard first."
    )

    task_state = task_session.interlocutor_state
    emotional_state = emotional_session.interlocutor_state

    assert task_result.response.text.strip()
    assert emotional_result.response.text.strip()
    assert task_state.readout_confidence >= 0.6
    assert emotional_state.readout_confidence >= 0.6
    assert task_state.task_focus_level > emotional_state.task_focus_level
    assert task_state.directness > emotional_state.directness
    assert emotional_state.rapport_warmth > task_state.rapport_warmth


async def test_companion_within_session_adapts_across_low_mood_episode() -> None:
    """A single session should move as the user's state unfolds."""

    from lifeform_domain_emogpt import build_companion_lifeform, scenarios_dir
    from lifeform_evolution import load_scenarios

    scenario = next(
        sc
        for sc in load_scenarios(scenarios_dir())
        if sc.scenario_id == "low-mood-disclosure"
    )
    session = build_companion_lifeform().create_session(session_id="evidence-low-mood")

    regimes: list[str | None] = []
    intents: list[str | None] = []
    pe_values: list[float] = []
    for turn in scenario.turns:
        result = await session.run_turn(turn.user_input)
        regimes.append(result.active_regime)
        assembly = result.active_snapshots["response_assembly"].value
        intents.append(assembly.expression_intent)
        pe_values.append(result.prediction_error.magnitude if result.prediction_error else 0.0)

    assert "emotional_support" in regimes
    assert len({intent for intent in intents if intent is not None}) >= 2
    assert max(pe_values) > min(pe_values)
    assert session.interlocutor_state.readout_confidence >= 0.6


async def test_companion_cross_session_retention_with_shared_memory_store() -> None:
    """Explicit shared memory lets a later session retrieve a learned preference."""

    from lifeform_domain_emogpt import build_companion_lifeform
    from volvence_zero.memory import build_default_memory_store

    shared_memory = build_default_memory_store()
    lifeform = build_companion_lifeform(memory_store=shared_memory)
    first = lifeform.create_session(session_id="evidence-retention-a")
    await first.run_turn(
        "When I am overwhelmed, please do not jump straight into steps; "
        "help me sort the feeling first."
    )
    await first.end_scene(reason="preference-captured")

    second = lifeform.create_session(session_id="evidence-retention-b")
    result = await second.run_turn(
        "I feel overwhelmed about work; what should I do first?"
    )
    memory = result.active_snapshots["memory"].value
    retrieved_text = "\n".join(entry.content for entry in memory.retrieved_entries)

    assert first.brain_session.runner.memory_store is second.brain_session.runner.memory_store
    assert "do not jump straight into steps" in retrieved_text
    assert "help me sort the feeling first" in retrieved_text


async def test_companion_default_sessions_keep_memory_isolated() -> None:
    """Default sessions are isolated unless a shared memory store is injected."""

    from lifeform_domain_emogpt import build_companion_lifeform

    lifeform = build_companion_lifeform()
    first = lifeform.create_session(session_id="evidence-isolated-a")
    second = lifeform.create_session(session_id="evidence-isolated-b")

    assert first.brain_session.runner.memory_store is not second.brain_session.runner.memory_store
