"""End-to-end: ``build_scored_snapshot`` over the coding vertical's
real 5 affordances, driven by a real ``LifeformSession`` + its
``regime`` / ``dual_track`` snapshots (Gap 1 slice 3).

Validates that slice 3 wiring is a drop-in replacement for the
slice-1 neutral scaffold:

* The coding vertical's registry still has 5 descriptors after
  slice 3 \u2014 no descriptor was accidentally excluded.
* When a real session runs a turn, the scored snapshot built
  from ``session.latest_active_snapshots`` produces per-descriptor
  scores in [0, 1] + a rationale tagged ``scorer.v1:``.
* The snapshot's ``selected`` policy respects the
  descriptor-level ``blocked_in_regimes``: invoking in
  ``casual_social`` blocks all 5 coding tools (they all list
  ``casual_social`` in their blocked_in_regimes).
* ``build_scoring_context_from_snapshots`` survives the real
  kernel snapshot shapes produced by ``build_coding_lifeform``.
"""

from __future__ import annotations

from lifeform_affordance import (
    AffordanceScoringContext,
    build_scored_snapshot,
    build_scoring_context_from_snapshots,
)


async def test_coding_vertical_scored_snapshot_with_real_session() -> None:
    """Run one turn of the coding lifeform, then build a scored
    snapshot from the live regime + dual_track snapshots.

    The scorer should produce 5 candidates (one per descriptor),
    each with a non-empty rationale tagged ``scorer.v1:`` and
    a score in [0, 1]. Selection is not asserted (depends on
    the regime the kernel lands in for this canned input).
    """
    from lifeform_domain_coding import (
        build_coding_affordance_registry,
        build_coding_lifeform,
    )

    lifeform = build_coding_lifeform()
    session = lifeform.create_session(session_id="scorer-e2e-real")
    await session.run_turn(
        "I have a bug where the thing crashes when the input is empty."
    )

    registry = build_coding_affordance_registry()
    regime_snap = session.latest_active_snapshots.get("regime")
    dual_track_snap = session.latest_active_snapshots.get("dual_track")
    assert regime_snap is not None, "kernel must publish a regime snapshot"
    assert dual_track_snap is not None, "kernel must publish a dual_track snapshot"

    ctx = build_scoring_context_from_snapshots(
        regime_snapshot=regime_snap,
        dual_track_snapshot=dual_track_snap,
    )
    # Both snapshots present -> evidence climbs to 0.80.
    assert ctx.evidence == 0.80
    assert ctx.active_regime_id  # non-empty

    snapshot = build_scored_snapshot(registry, ctx)
    assert len(snapshot.candidates_for_turn) == 5
    names = {c.descriptor_name for c in snapshot.candidates_for_turn}
    assert names == {"read_file", "list_dir", "grep", "write_file", "run_test"}
    for candidate in snapshot.candidates_for_turn:
        assert 0.0 <= candidate.score <= 1.0
        assert candidate.rationale  # non-empty
        # Either a scorer.v1 prefix (unblocked) or the blocked tag.
        assert candidate.rationale.startswith("scorer.v1:")


async def test_scored_snapshot_blocks_all_coding_tools_in_casual_social() -> None:
    """All 5 coding-vertical descriptors declare
    ``blocked_in_regimes=(casual_social, emotional_support,
    repair_and_deescalation)``. Synthesising a context with
    ``active_regime_id='casual_social'`` should mark every
    candidate blocked and leave ``selected`` as None.
    """
    from lifeform_domain_coding import build_coding_affordance_registry

    registry = build_coding_affordance_registry()
    ctx = AffordanceScoringContext(
        active_regime_id="casual_social",
        evidence=0.8,
        task_bias=1.0,  # task signal max; still must be blocked
        world_drive=0.8,
    )
    snapshot = build_scored_snapshot(registry, ctx)
    assert len(snapshot.candidates_for_turn) == 5
    assert all(c.is_blocked for c in snapshot.candidates_for_turn), (
        f"all coding tools must be blocked in casual_social; got "
        f"{[(c.descriptor_name, c.blocked_reason) for c in snapshot.candidates_for_turn]!r}"
    )
    assert snapshot.selected is None


async def test_scored_snapshot_promotes_read_over_write_cold_start() -> None:
    """Fresh scene (turns_in_current_regime=0) + no runtime signal
    should prefer read_file / grep / list_dir over write_file +
    run_test (which carry irreversible / subprocess cost). The
    ``selected`` field may still be None if the margin is thin.
    """
    from lifeform_domain_coding import build_coding_affordance_registry

    registry = build_coding_affordance_registry()
    ctx = AffordanceScoringContext(
        active_regime_id="problem_solving",
        cognitive_depth="shallow",
        turns_in_current_regime=0,
        evidence=0.6,
        task_bias=0.5,
        world_drive=0.4,
        cross_track_tension=0.4,
    )
    snapshot = build_scored_snapshot(registry, ctx)
    by_name = {c.descriptor_name: c.score for c in snapshot.candidates_for_turn}
    # Read-only tools > write_file under cold + tense conditions.
    assert by_name["read_file"] >= by_name["write_file"], (
        f"read_file should outscore write_file under cold + tense; "
        f"got {by_name!r}"
    )
    # run_test (SLOW) under SHALLOW depth should be penalised
    # relative to read_file (FAST).
    assert by_name["read_file"] >= by_name["run_test"], (
        f"read_file (FAST) should outscore run_test (SLOW) at SHALLOW depth; "
        f"got {by_name!r}"
    )
