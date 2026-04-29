"""Unit tests for ``lifeform_affordance.scorer`` (Gap 1 slice 3).

Covers:

* ``AffordanceScoringContext`` invariants: scalar fields in [0,1],
  non-negative turns, defaults produce a low-confidence neutral.
* ``score_affordance`` happy path: TOOL + task_bias high -> score
  rises; SLOW latency + REFLEXIVE depth -> score falls;
  irreversible + cold-start -> safety_push kicks in.
* Regime block: descriptor whose ``blocked_in_regimes`` contains
  the active regime -> score=0.0 + ``blocked_reason`` populated.
* Selection threshold: top candidate below 0.5 OR margin < 0.06
  -> ``selected`` is None.
* ``build_scored_snapshot`` shape + ``selected`` policy.
* Duck-typed ``build_scoring_context_from_snapshots`` extracts
  the right fields from real kernel ``RegimeSnapshot`` /
  ``DualTrackSnapshot`` shapes + survives missing / None inputs.
* Back-compat: ``build_neutral_snapshot`` still works unchanged.
"""

from __future__ import annotations

import pytest

from lifeform_affordance import (
    AffordanceCandidate,
    AffordanceCost,
    AffordanceDescriptor,
    AffordanceKind,
    AffordanceLatencyClass,
    AffordanceMonetaryClass,
    AffordanceRegistry,
    AffordanceSafety,
    AffordanceScoringContext,
    AffordanceSnapshot,
    build_neutral_snapshot,
    build_scored_snapshot,
    build_scoring_context_from_snapshots,
    score_affordance,
)


# ---------------------------------------------------------------------------
# Descriptor fixtures
# ---------------------------------------------------------------------------


_HINT = (
    "Use when the lifeform needs to trigger a scripted test affordance "
    "to exercise the scorer under controlled feature inputs."
)


def _make_descriptor(
    *,
    name: str,
    kind: AffordanceKind = AffordanceKind.TOOL,
    tags: tuple[str, ...] = (),
    latency: AffordanceLatencyClass = AffordanceLatencyClass.FAST,
    irreversible: bool = False,
    requires_confirmation: bool = False,
    blocked_in_regimes: tuple[str, ...] = (),
    excluded: bool = False,
) -> AffordanceDescriptor:
    return AffordanceDescriptor(
        name=name,
        kind=kind,
        version="0.1.0",
        display_name=f"Test {name}",
        description=f"test descriptor {name}",
        when_to_use=_HINT + " " + name,
        when_not_to_use=_HINT + " do not use if the scorer is not under test " + name,
        parameters_schema={"type": "object"},
        output_schema={"type": "object"},
        cost_model=AffordanceCost(
            latency_class=latency,
            monetary_class=AffordanceMonetaryClass.FREE,
        ),
        safety_model=AffordanceSafety(
            irreversible=irreversible,
            requires_user_confirmation=requires_confirmation,
            blocked_in_regimes=blocked_in_regimes,
        ),
        affordance_tags=tags,
        excluded_from_runtime_selection=excluded,
    )


# ---------------------------------------------------------------------------
# Context invariants
# ---------------------------------------------------------------------------


def test_default_context_is_neutral_low_evidence() -> None:
    ctx = AffordanceScoringContext()
    assert ctx.evidence == 0.5
    assert ctx.flow_kind == "info"
    assert ctx.cognitive_depth == "focused"
    assert ctx.active_regime_id == ""


def test_context_rejects_out_of_range_scalar() -> None:
    with pytest.raises(ValueError, match="in \\[0,1\\]"):
        AffordanceScoringContext(world_drive=1.5)


def test_context_rejects_negative_turns() -> None:
    with pytest.raises(ValueError, match="turns_in_current_regime"):
        AffordanceScoringContext(turns_in_current_regime=-1)


# ---------------------------------------------------------------------------
# score_affordance: regime blocking
# ---------------------------------------------------------------------------


def test_score_affordance_blocks_on_regime_match() -> None:
    descriptor = _make_descriptor(
        name="read_file_like",
        blocked_in_regimes=("casual_social", "emotional_support"),
    )
    ctx = AffordanceScoringContext(
        active_regime_id="casual_social",
        evidence=0.8,
        task_bias=1.0,
    )
    candidate = score_affordance(descriptor, ctx)
    assert candidate.score == 0.0
    assert candidate.is_blocked
    assert "regime_blocked" in candidate.blocked_reason


def test_score_affordance_not_blocked_when_regime_unmatched() -> None:
    descriptor = _make_descriptor(
        name="read_file_like",
        blocked_in_regimes=("casual_social",),
    )
    ctx = AffordanceScoringContext(
        active_regime_id="problem_solving",
        evidence=0.8,
        task_bias=0.8,
        world_drive=0.7,
    )
    candidate = score_affordance(descriptor, ctx)
    assert not candidate.is_blocked
    assert candidate.score > 0.0


def test_score_affordance_blocks_excluded_descriptor() -> None:
    descriptor = _make_descriptor(name="excluded_op", excluded=True)
    ctx = AffordanceScoringContext(
        active_regime_id="problem_solving", evidence=0.8
    )
    candidate = score_affordance(descriptor, ctx)
    assert candidate.is_blocked
    assert "descriptor_excluded" in candidate.blocked_reason


# ---------------------------------------------------------------------------
# score_affordance: continuous-feature pulls
# ---------------------------------------------------------------------------


def test_tool_affordance_scores_higher_under_task_pressure() -> None:
    descriptor = _make_descriptor(
        name="run_test_like",
        kind=AffordanceKind.TOOL,
        tags=("execute", "test", "code"),
    )
    cold_ctx = AffordanceScoringContext(
        active_regime_id="problem_solving", evidence=0.8
    )
    task_ctx = AffordanceScoringContext(
        active_regime_id="problem_solving",
        evidence=0.8,
        task_bias=1.0,
        world_drive=0.8,
    )
    cold_score = score_affordance(descriptor, cold_ctx).score
    task_score = score_affordance(descriptor, task_ctx).score
    assert task_score > cold_score, (
        f"task-heavy context should score {descriptor.name!r} above "
        f"neutral baseline; got cold={cold_score:.3f} task={task_score:.3f}"
    )


def test_slow_latency_penalised_at_shallow_depth() -> None:
    slow = _make_descriptor(
        name="slow_tool",
        latency=AffordanceLatencyClass.SLOW,
        tags=("execute", "code"),
    )
    fast = _make_descriptor(
        name="fast_tool",
        latency=AffordanceLatencyClass.FAST,
        tags=("execute", "code"),
    )
    # SHALLOW depth should penalise slow more than fast.
    ctx = AffordanceScoringContext(
        active_regime_id="casual_social",
        cognitive_depth="shallow",
        evidence=0.8,
        task_bias=0.5,
        world_drive=0.5,
    )
    slow_score = score_affordance(slow, ctx).score
    fast_score = score_affordance(fast, ctx).score
    assert fast_score > slow_score, (
        f"SHALLOW depth should prefer FAST over SLOW; "
        f"slow={slow_score:.3f} fast={fast_score:.3f}"
    )


def test_irreversible_descriptor_penalised_cold_start() -> None:
    risky = _make_descriptor(
        name="write_file_like",
        irreversible=True,
        requires_confirmation=True,
        tags=("write", "code"),
    )
    safe = _make_descriptor(
        name="read_file_like",
        irreversible=False,
        tags=("read", "code"),
    )
    cold_ctx = AffordanceScoringContext(
        active_regime_id="problem_solving",
        turns_in_current_regime=0,
        evidence=0.4,
        task_bias=0.9,
        world_drive=0.8,
        cross_track_tension=0.5,
    )
    risky_score = score_affordance(risky, cold_ctx).score
    safe_score = score_affordance(safe, cold_ctx).score
    assert safe_score > risky_score, (
        f"Cold + tense context should prefer read-only over irreversible; "
        f"risky={risky_score:.3f} safe={safe_score:.3f}"
    )


def test_evidence_zero_keeps_score_at_base() -> None:
    """With evidence=0 the scorer should not move from the 0.40 base.

    This guarantees cold-start snapshots are close to neutral,
    matching the slice-1 ``build_neutral_snapshot`` semantics.
    """
    descriptor = _make_descriptor(
        name="any_tool",
        tags=("read", "code"),
    )
    ctx = AffordanceScoringContext(
        active_regime_id="problem_solving",
        evidence=0.0,
        task_bias=1.0,  # max pull but evidence cancels it
    )
    assert score_affordance(descriptor, ctx).score == pytest.approx(0.40)


def test_rationale_includes_scorer_version_tag() -> None:
    descriptor = _make_descriptor(name="x", tags=("code",))
    ctx = AffordanceScoringContext(active_regime_id="problem_solving")
    rationale = score_affordance(descriptor, ctx).rationale
    assert rationale.startswith("scorer.v1:")


# ---------------------------------------------------------------------------
# build_scored_snapshot
# ---------------------------------------------------------------------------


def _registry_with(descriptors: tuple[AffordanceDescriptor, ...]) -> AffordanceRegistry:
    registry = AffordanceRegistry()
    registry.register_all(descriptors)
    registry.seal()
    return registry


def test_build_scored_snapshot_includes_all_unblocked_as_candidates() -> None:
    registry = _registry_with((
        _make_descriptor(name="a", tags=("code",)),
        _make_descriptor(name="b", tags=("read", "code")),
    ))
    snapshot = build_scored_snapshot(
        registry,
        AffordanceScoringContext(active_regime_id="problem_solving", evidence=0.7),
    )
    names = {c.descriptor_name for c in snapshot.candidates_for_turn}
    assert names == {"a", "b"}
    assert len(snapshot.available) == 2


def test_build_scored_snapshot_selects_top_when_margin_met() -> None:
    # winner has strong task bias alignment; loser is generic.
    winner = _make_descriptor(
        name="winner",
        tags=("execute", "test", "code"),
    )
    loser = _make_descriptor(
        name="loser",
        kind=AffordanceKind.SHELL,
    )
    registry = _registry_with((winner, loser))
    ctx = AffordanceScoringContext(
        active_regime_id="problem_solving",
        evidence=0.9,
        task_bias=1.0,
        world_drive=0.8,
    )
    snapshot = build_scored_snapshot(registry, ctx)
    assert snapshot.selected is not None
    assert snapshot.selected.descriptor_name == "winner"


def test_build_scored_snapshot_no_selection_when_margin_too_thin() -> None:
    a = _make_descriptor(name="a", tags=("code",))
    b = _make_descriptor(name="b", tags=("code",))  # same tags -> same score
    registry = _registry_with((a, b))
    ctx = AffordanceScoringContext(
        active_regime_id="problem_solving", evidence=0.7, task_bias=0.9
    )
    snapshot = build_scored_snapshot(registry, ctx)
    # Two near-identical descriptors -> margin too thin -> no selection.
    assert snapshot.selected is None


def test_build_scored_snapshot_no_selection_when_all_low() -> None:
    descriptor = _make_descriptor(name="only")
    registry = _registry_with((descriptor,))
    ctx = AffordanceScoringContext(
        active_regime_id="casual_social",
        evidence=0.1,  # low evidence keeps score near 0.40
    )
    snapshot = build_scored_snapshot(registry, ctx)
    assert snapshot.selected is None


def test_build_scored_snapshot_blocked_candidates_stay_in_list() -> None:
    blocked = _make_descriptor(
        name="blocked_one",
        blocked_in_regimes=("casual_social",),
    )
    allowed = _make_descriptor(name="allowed_one", tags=("code",))
    registry = _registry_with((blocked, allowed))
    ctx = AffordanceScoringContext(
        active_regime_id="casual_social",
        evidence=0.7,
    )
    snapshot = build_scored_snapshot(registry, ctx)
    names = {c.descriptor_name for c in snapshot.candidates_for_turn}
    assert names == {"blocked_one", "allowed_one"}
    [blocked_candidate] = [
        c for c in snapshot.candidates_for_turn if c.descriptor_name == "blocked_one"
    ]
    assert blocked_candidate.is_blocked
    # selected must never be a blocked candidate
    if snapshot.selected is not None:
        assert not snapshot.selected.is_blocked


# ---------------------------------------------------------------------------
# build_scoring_context_from_snapshots (duck-typed)
# ---------------------------------------------------------------------------


def test_builder_returns_low_evidence_for_none_inputs() -> None:
    ctx = build_scoring_context_from_snapshots(
        regime_snapshot=None, dual_track_snapshot=None
    )
    assert ctx.evidence == pytest.approx(0.10)
    assert ctx.active_regime_id == ""


def test_builder_extracts_fields_from_real_kernel_snapshots() -> None:
    from volvence_zero.dual_track.core import DualTrackSnapshot, TrackState
    from volvence_zero.memory.store import Track
    from volvence_zero.regime import (
        CognitiveDepth,
        CognitiveDepthHint,
        ParticipationFlowKind,
        ParticipationHint,
        ParticipationLevel,
        RegimeIdentity,
        RegimeSnapshot,
    )

    regime_snapshot_value = RegimeSnapshot(
        active_regime=RegimeIdentity(
            regime_id="problem_solving",
            name="problem solving",
            embedding=(0.8, 0.2, 0.3),
            entry_conditions="",
            exit_conditions="",
            historical_effectiveness=0.6,
        ),
        previous_regime=None,
        switch_reason="test",
        candidate_regimes=(("problem_solving", 0.85), ("guided_exploration", 0.3)),
        turns_in_current_regime=3,
        description="",
        participation_hint=ParticipationHint(
            flow_kind=ParticipationFlowKind.PROBLEM,
            panorama_level=ParticipationLevel.STRUCTURED,
            method_level=ParticipationLevel.STRUCTURED,
            task_level=ParticipationLevel.STRUCTURED,
        ),
        depth_hint=CognitiveDepthHint(depth=CognitiveDepth.ALERT),
    )
    dual_track_value = DualTrackSnapshot(
        world_track=TrackState(
            track=Track.WORLD,
            active_goals=(),
            recent_credits=(),
            controller_code=(0.80, 0.40, 0.20),
            tension_level=0.50,
            abstract_action_hint="task_controller",
        ),
        self_track=TrackState(
            track=Track.SELF,
            active_goals=(),
            recent_credits=(),
            controller_code=(0.30, 0.20, 0.15),
            tension_level=0.20,
            abstract_action_hint="task_controller",
        ),
        cross_track_tension=0.25,
        description="",
    )
    ctx = build_scoring_context_from_snapshots(
        regime_snapshot=regime_snapshot_value,
        dual_track_snapshot=dual_track_value,
    )
    assert ctx.active_regime_id == "problem_solving"
    assert ctx.flow_kind == "problem"
    assert ctx.cognitive_depth == "alert"
    assert ctx.turns_in_current_regime == 3
    assert ctx.world_drive == pytest.approx(0.80)
    assert ctx.self_drive == pytest.approx(0.30)
    assert ctx.cross_track_tension == pytest.approx(0.25)
    assert ctx.task_bias == pytest.approx(1.0)
    assert ctx.repair_bias == pytest.approx(0.0)
    assert ctx.evidence == pytest.approx(0.80)


def test_builder_handles_partial_inputs_mid_evidence() -> None:
    """Only regime present -> evidence 0.45; drives stay 0."""
    from volvence_zero.regime import RegimeIdentity, RegimeSnapshot

    regime = RegimeSnapshot(
        active_regime=RegimeIdentity(
            regime_id="casual_social",
            name="casual social",
            embedding=(0.1, 0.1, 0.1),
            entry_conditions="",
            exit_conditions="",
            historical_effectiveness=0.5,
        ),
        previous_regime=None,
        switch_reason="",
        candidate_regimes=(("casual_social", 0.7),),
        turns_in_current_regime=1,
        description="",
    )
    ctx = build_scoring_context_from_snapshots(
        regime_snapshot=regime, dual_track_snapshot=None
    )
    assert ctx.evidence == pytest.approx(0.45)
    assert ctx.active_regime_id == "casual_social"
    assert ctx.world_drive == 0.0


# ---------------------------------------------------------------------------
# Back-compat: build_neutral_snapshot still works unchanged
# ---------------------------------------------------------------------------


def test_build_neutral_snapshot_still_returns_flat_snapshot() -> None:
    registry = _registry_with((
        _make_descriptor(name="a"),
        _make_descriptor(name="b"),
    ))
    snapshot = build_neutral_snapshot(registry)
    for candidate in snapshot.candidates_for_turn:
        assert candidate.score == 0.5
        assert candidate.rationale.startswith("neutral scaffold")
    assert snapshot.selected is None
