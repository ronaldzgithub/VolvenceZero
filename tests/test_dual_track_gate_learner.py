"""W1.A acceptance: dual-track gate is a genuine bounded online-SGD learner.

The session-held ``DualTrackGateLearner`` replaces the fixed
``derive_learned_gate_shadow`` formula: it predicts the SHADOW gate weight
from typed track readouts and is scored next turn against the PE owner's
realized task/relationship outcome. Report-only — the live track fusion is
untouched (CP-19 kill conditions unchanged).
"""

from __future__ import annotations

import asyncio

from volvence_zero.dual_track import (
    DualTrackGateLearner,
    DualTrackLearnedGateShadow,
    DualTrackModule,
    TrackState,
)
from volvence_zero.memory import Track
from volvence_zero.runtime import WiringLevel


def _track(track: Track, *, tension: float, code: tuple[float, float, float]) -> TrackState:
    return TrackState(
        track=track,
        active_goals=("goal-a", "goal-b"),
        recent_credits=(),
        controller_code=code,
        tension_level=tension,
    )


_WORLD = _track(Track.WORLD, tension=0.8, code=(0.7, 0.3, 0.5))
_SELF = _track(Track.SELF, tension=0.2, code=(0.2, 0.4, 0.1))


def test_gate_learner_cold_start_is_neutral_and_bounded() -> None:
    learner = DualTrackGateLearner()
    gate = learner.derive_shadow(
        world_track=_WORLD, self_track=_SELF, cross_track_tension=0.3
    )
    assert isinstance(gate, DualTrackLearnedGateShadow)
    # Cold start: bias-only weights predict the neutral 0.5/0.5 prior.
    assert abs(gate.world_weight - 0.5) <= 1e-6
    assert abs(gate.world_weight + gate.self_weight - 1.0) <= 1e-6
    assert "report-only" in gate.description
    assert "online-SGD" in gate.description
    assert learner.update_count == 0


def test_gate_learner_skips_bootstrap_settlement() -> None:
    learner = DualTrackGateLearner()
    # No derive_shadow yet -> nothing settleable.
    assert learner.observe_realized_outcome(task_progress=0.9, relationship_delta=-0.5) is False
    # First derive rotates None into the settlement window: still nothing.
    learner.derive_shadow(world_track=_WORLD, self_track=_SELF, cross_track_tension=0.3)
    assert learner.observe_realized_outcome(task_progress=0.9, relationship_delta=-0.5) is False
    assert learner.update_count == 0


def test_gate_learner_learns_toward_world_favoring_outcomes() -> None:
    learner = DualTrackGateLearner()
    for _ in range(60):
        learner.derive_shadow(
            world_track=_WORLD, self_track=_SELF, cross_track_tension=0.3
        )
        # Realized outcome consistently favors the world axis: high task
        # progress, negative relationship delta -> target near the world cap.
        learner.observe_realized_outcome(task_progress=1.0, relationship_delta=-1.0)
    assert learner.update_count >= 50
    gate = learner.derive_shadow(
        world_track=_WORLD, self_track=_SELF, cross_track_tension=0.3
    )
    assert gate.world_weight > 0.6
    # Movement stays capped around the neutral prior while SHADOW.
    assert gate.world_weight <= 0.85 + 1e-6
    assert abs(gate.world_weight + gate.self_weight - 1.0) <= 1e-6
    readout = learner.readout()
    assert readout.update_count == learner.update_count
    assert all(-2.0 <= w <= 2.0 for w in readout.weights)


def test_gate_learner_error_shrinks_on_stationary_target() -> None:
    learner = DualTrackGateLearner()
    errors: list[float] = []
    for _ in range(80):
        gate = learner.derive_shadow(
            world_track=_WORLD, self_track=_SELF, cross_track_tension=0.3
        )
        errors.append(abs(gate.world_weight - 0.85))
        learner.observe_realized_outcome(task_progress=1.0, relationship_delta=-1.0)
    early = sum(errors[:10]) / 10
    late = sum(errors[-10:]) / 10
    assert late < early


def test_module_uses_injected_session_held_learner() -> None:
    learner = DualTrackGateLearner()
    module = DualTrackModule(wiring_level=WiringLevel.ACTIVE, gate_learner=learner)
    snapshot = asyncio.run(
        module.process_standalone(world_entries=(), self_entries=())
    ).value
    gate = snapshot.learned_gate_shadow
    assert gate is not None
    assert "online-SGD" in gate.description
    # A rebuilt module (per-turn rebuild) keeps the SAME learner state.
    learner.observe_realized_outcome(task_progress=0.2, relationship_delta=0.8)
    rebuilt = DualTrackModule(wiring_level=WiringLevel.ACTIVE, gate_learner=learner)
    snapshot_2 = asyncio.run(
        rebuilt.process_standalone(world_entries=(), self_entries=())
    ).value
    assert snapshot_2.learned_gate_shadow is not None
    learner.observe_realized_outcome(task_progress=0.2, relationship_delta=0.8)
    assert learner.update_count == 1  # first observe had no settleable features


def test_module_without_learner_keeps_fixed_prior_fallback() -> None:
    module = DualTrackModule(wiring_level=WiringLevel.ACTIVE)
    snapshot = asyncio.run(
        module.process_standalone(world_entries=(), self_entries=())
    ).value
    gate = snapshot.learned_gate_shadow
    assert gate is not None
    assert "online-SGD" not in gate.description
