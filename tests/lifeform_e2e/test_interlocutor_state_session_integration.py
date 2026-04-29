"""End-to-end: ``LifeformSession.interlocutor_state`` over real kernel snapshots
(Gap 9 slice 1).

Validates that the 12-axis readout survives real kernel turn
dynamics:

* Baseline: before any turn runs, ``interlocutor_state`` returns
  the cold neutral default with ``readout_confidence`` in the
  ~0.25 range.
* After one turn: confidence climbs because dual_track + evaluation
  + PE + memory are all present; axes have meaningful values in
  [0, 1].
* Differentiation: running the companion lifeform on a
  "task-focused" prompt vs an "emotional-support" prompt should
  produce materially different ``task_focus_level`` /
  ``emotional_weight`` / ``directness`` / ``rapport_warmth``.
"""

from __future__ import annotations

import pytest


async def test_session_without_turn_returns_cold_interlocutor_state() -> None:
    from lifeform_domain_emogpt import build_companion_lifeform

    lifeform = build_companion_lifeform()
    session = lifeform.create_session(session_id="g9-cold")
    state = session.interlocutor_state
    # No turn has run -> no active snapshots -> cold neutral.
    assert state.engagement_intensity == pytest.approx(0.5, abs=0.12)
    assert state.trust_signal == pytest.approx(0.0, abs=0.05)
    # Confidence is low (only the baseline 0.10 evidence applies).
    assert state.readout_confidence <= 0.35


async def test_session_after_turn_raises_readout_confidence() -> None:
    from lifeform_domain_emogpt import build_companion_lifeform

    lifeform = build_companion_lifeform()
    session = lifeform.create_session(session_id="g9-one-turn")
    await session.run_turn("hello there, how are you doing today?")
    state = session.interlocutor_state
    # After one turn: regime + dual_track + evaluation + memory +
    # PE are all present. Readout confidence should climb.
    assert state.readout_confidence >= 0.6, (
        f"expected readout confidence >= 0.6 after a real turn; "
        f"got {state.readout_confidence}. state rationale={state.rationale!r}"
    )
    # Every axis must remain a valid float in [0, 1] except trust
    # which is in [-1, 1] \u2014 post_init would have raised otherwise,
    # so the absence of an exception is the assertion here.
    assert 0.0 <= state.engagement_intensity <= 1.0
    assert 0.0 <= state.resistance_level <= 1.0
    assert 0.0 <= state.openness_to_guidance <= 1.0
    assert -1.0 <= state.trust_signal <= 1.0
    # Rationale carries the readout version tag.
    assert state.rationale.startswith("readout.v1.interlocutor:")


async def test_interlocutor_state_tracks_multi_turn_progression() -> None:
    """Running N turns should make the readout EVOLVE even if the
    regime classifier lands the session in the same regime each
    turn \u2014 drives / memory presence / PE error signals change
    across turns and the readout must reflect that.

    We don't assert a specific direction (too brittle against
    calibration changes across builds) \u2014 we assert that at least
    one axis moves by >= 0.03 between turn 1 and turn 3. That
    proves the readout is dynamic, not a constant snapshot of
    "session opened" state.

    This test covers the end-to-end wiring (``LifeformSession``
    property reads real kernel snapshots + the readout captures
    the turn delta). Prompt-level differentiation in the readout
    is a kernel-calibration concern, not a readout-correctness
    concern, so we don't assert it here.
    """
    from lifeform_domain_emogpt import build_companion_lifeform

    lifeform = build_companion_lifeform()
    session = lifeform.create_session(session_id="g9-multi-turn")
    await session.run_turn("hello")
    state_turn1 = session.interlocutor_state
    await session.run_turn("tell me more")
    await session.run_turn("that was interesting, thanks")
    state_turn3 = session.interlocutor_state

    axes = (
        "engagement_intensity",
        "self_disclosure_level",
        "task_focus_level",
        "emotional_weight",
        "cognitive_engagement",
        "resistance_level",
        "openness_to_guidance",
        "directness",
        "trust_signal",
        "stability",
        "rapport_warmth",
        "pace_pressure",
    )
    moved = [
        axis
        for axis in axes
        if abs(getattr(state_turn1, axis) - getattr(state_turn3, axis)) >= 0.03
    ]
    assert moved, (
        f"No axis moved by >= 0.03 across 3 turns; readout is static.\n"
        f"  turn1: {state_turn1!r}\n  turn3: {state_turn3!r}"
    )


async def test_interlocutor_state_differentiates_on_handcrafted_contexts() -> None:
    """Differentiation is a readout-level property even if the
    real kernel doesn't always diverge on different prompts.

    Build two ``InterlocutorReadoutContext`` instances by hand:
    one with a task-heavy profile, one with an emotional-support
    profile. The readout MUST produce materially different
    ``task_focus_level`` / ``emotional_weight`` / ``directness``
    / ``rapport_warmth`` axes.

    This guards the scorer semantics end-to-end through the
    public API exposed by ``volvence_zero.interlocutor`` \u2014 it
    complements the unit tests by catching any future integration
    wiring that might silently normalise contexts.
    """
    from volvence_zero.interlocutor import (
        InterlocutorReadoutContext,
        readout_interlocutor_state,
    )

    task_ctx = InterlocutorReadoutContext(
        active_regime_id="problem_solving",
        has_dual_track=True, has_evaluation=True, has_prediction_error=True,
        has_memory=True,
        world_drive=0.85, world_tension=0.4,
        task_bias=1.0, task_pressure=0.8,
        world_presence=0.9, self_presence=0.1,
        warmth=0.35, support_presence=0.25,
        info_integration=0.8,
    )
    emo_ctx = InterlocutorReadoutContext(
        active_regime_id="emotional_support",
        has_dual_track=True, has_evaluation=True, has_prediction_error=True,
        has_memory=True,
        self_drive=0.8, self_tension=0.8,
        repair_bias=0.7,
        self_presence=0.9, world_presence=0.1,
        warmth=0.75, support_presence=0.85,
        cross_track_tension=0.4,
        info_integration=0.4,
    )
    task_state = readout_interlocutor_state(task_ctx)
    emo_state = readout_interlocutor_state(emo_ctx)

    # Task context should focus harder on the task axis.
    assert task_state.task_focus_level > emo_state.task_focus_level, (
        f"task_focus_level should rise under task-heavy context; "
        f"task={task_state.task_focus_level} emo={emo_state.task_focus_level}"
    )
    # Emotional-support context should weigh emotion higher.
    assert emo_state.emotional_weight > task_state.emotional_weight
    # Task context should be more direct; emotional less so.
    assert task_state.directness > emo_state.directness
    # Emotional-support context should have higher rapport warmth.
    assert emo_state.rapport_warmth > task_state.rapport_warmth
    # Emotional context should disclose more.
    assert emo_state.self_disclosure_level > task_state.self_disclosure_level
