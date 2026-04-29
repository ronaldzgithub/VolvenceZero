"""End-to-end: scene-end drives case-memory provisional reconcile (Gap 4 slice 2a).

Scope (docs/specs/thinking-loop.md §Owner-side apply path + Gap 4
slice 2a of docs/implementation/13_emogpt_prd_alignment_upgrade.md):

* ``LifeformSession.end_scene`` must drive the case-memory provisional
  reconcile pass **after** ``drain_session_post_slow_loop`` so any
  records the slow loop wrote during scene close are part of the
  decision set.
* The result must be exposed via
  ``LifeformSession.latest_case_memory_reconcile`` for observability
  without poking owner internals.
* Idle-timeout scene closures must still sweep (no
  ``drain_slow_loop`` happens, but the reconcile still fires so
  stale provisional records cannot persist across scenes).
* Pure ``VALIDATED`` fleets produce no-op reconciles \u2014 the sweep is
  cheap.

These tests reach directly into the kernel's ``AgentSessionRunner``
via ``brain_session.runner`` to seed case memory records with
non-default lifecycle states. That is the documented back-door for
tests / tooling that need to write records outside the normal
reflection-writeback path.
"""

from __future__ import annotations

import asyncio

import pytest

from volvence_zero.application import (
    CaseLifecycle,
    CaseMemoryRecord,
    ProvisionalReconcileResult,
)


def _case(
    case_id: str,
    *,
    lifecycle: CaseLifecycle,
    relevance_score: float,
    confidence: float,
    ttl_seconds: int | None = 1800,
    expires_at_tick: int | None = None,
    provisional_origin: str = "test-seed",
) -> CaseMemoryRecord:
    return CaseMemoryRecord(
        case_id=case_id,
        domain="relationship_continuity",
        problem_pattern="overload-then-decision",
        user_state_pattern="emotionally-overloaded",
        risk_markers=("risk-medium",),
        track_tags=("self",),
        regime_tags=("emotional_support",),
        intervention_ordering=("acknowledge", "slow_pace"),
        outcome_label="stable",
        delayed_signal_count=1,
        escalation_observed=False,
        repair_observed=True,
        confidence=confidence,
        relevance_score=relevance_score,
        description=f"Seeded case {case_id} for scene-reconcile test.",
        lifecycle=lifecycle,
        ttl_seconds=ttl_seconds,
        expires_at_tick=expires_at_tick,
        provisional_origin=provisional_origin,
    )


def _seed_case_records(session, records: tuple[CaseMemoryRecord, ...]) -> None:
    """Directly upsert records into the kernel's case memory store.

    This bypasses reflection writeback and is a test-only shortcut
    to put records into PROVISIONAL / CANDIDATE / RETIRED states that
    production code would normally reach only through the slow loop.
    """
    store = session.brain_session.runner._case_memory_store  # noqa: SLF001
    store.upsert_records(records)


# ---------------------------------------------------------------------------
# Happy path: end_scene promotes a strong PROVISIONAL record
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_scene_promotes_strong_provisional_record() -> None:
    from lifeform_domain_emogpt import build_companion_lifeform

    lifeform = build_companion_lifeform()
    session = lifeform.create_session(session_id="scene-reconcile-promote")
    # Run one turn to open a scene, then seed a PROVISIONAL record.
    await session.run_turn("I'm overloaded and can't decide.")
    _seed_case_records(
        session,
        (
            _case(
                "provisional-promote",
                lifecycle=CaseLifecycle.PROVISIONAL,
                relevance_score=0.80,
                confidence=0.70,
            ),
        ),
    )
    # Close the scene \u2014 end_scene must drive reconcile.
    await session.end_scene(reason="scene-complete")
    reconcile = session.latest_case_memory_reconcile
    assert isinstance(reconcile, ProvisionalReconcileResult)
    assert "provisional-promote" in reconcile.promoted
    assert "provisional-promote" not in reconcile.retired
    assert "provisional-promote" not in reconcile.expired
    # Follow-through: the store now carries the record as VALIDATED.
    store = session.brain_session.runner._case_memory_store  # noqa: SLF001
    by_id = {r.case_id: r for r in store.records}
    assert by_id["provisional-promote"].lifecycle is CaseLifecycle.VALIDATED


# ---------------------------------------------------------------------------
# Weak record is retired at scene-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_scene_retires_weak_provisional_record() -> None:
    from lifeform_domain_emogpt import build_companion_lifeform

    lifeform = build_companion_lifeform()
    session = lifeform.create_session(session_id="scene-reconcile-retire")
    await session.run_turn("Let's talk.")
    _seed_case_records(
        session,
        (
            _case(
                "provisional-weak",
                lifecycle=CaseLifecycle.PROVISIONAL,
                relevance_score=0.10,
                confidence=0.80,
            ),
        ),
    )
    await session.end_scene(reason="scene-complete")
    reconcile = session.latest_case_memory_reconcile
    assert isinstance(reconcile, ProvisionalReconcileResult)
    assert "provisional-weak" in reconcile.retired


# ---------------------------------------------------------------------------
# Expired-by-tick retires even when strong
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_scene_expires_record_whose_tick_has_passed() -> None:
    from lifeform_domain_emogpt import build_companion_lifeform

    lifeform = build_companion_lifeform()
    session = lifeform.create_session(session_id="scene-reconcile-expire")
    await session.run_turn("Hello.")
    # Advance a handful of ticks so the lifeform clock progresses; the
    # seeded record's ``expires_at_tick`` must be in the past by the
    # time end_scene fires.
    await session.advance_tick(system_ticks=5, reason="prep")
    current_tick = session.tick_engine.tick_index
    _seed_case_records(
        session,
        (
            _case(
                "provisional-expired",
                lifecycle=CaseLifecycle.PROVISIONAL,
                relevance_score=0.95,  # would otherwise promote
                confidence=0.95,
                expires_at_tick=max(0, current_tick - 1),
            ),
        ),
    )
    await session.end_scene(reason="scene-complete")
    reconcile = session.latest_case_memory_reconcile
    assert isinstance(reconcile, ProvisionalReconcileResult)
    assert "provisional-expired" in reconcile.expired
    assert "provisional-expired" not in reconcile.promoted


# ---------------------------------------------------------------------------
# Pure-VALIDATED fleet: reconcile is a no-op
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_scene_reconcile_is_noop_for_validated_fleet() -> None:
    from lifeform_domain_emogpt import build_companion_lifeform

    lifeform = build_companion_lifeform()
    session = lifeform.create_session(session_id="scene-reconcile-noop")
    await session.run_turn("All good.")
    await session.end_scene(reason="scene-complete")
    reconcile = session.latest_case_memory_reconcile
    assert isinstance(reconcile, ProvisionalReconcileResult)
    assert reconcile.promoted == ()
    assert reconcile.retired == ()
    assert reconcile.expired == ()
    assert reconcile.decisions == ()


# ---------------------------------------------------------------------------
# Idle-timeout scene close also sweeps (drain_slow_loop=False path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idle_timeout_scene_close_still_sweeps_provisional_records() -> None:
    """Idle-timeout scenes call ``end_scene(drain_slow_loop=False)``
    through ``advance_tick``. The reconcile must still fire so stale
    provisional records cannot persist across scenes just because the
    slow loop was skipped.
    """
    from lifeform_core import LifeformConfig, TickEngineConfig
    from lifeform_domain_emogpt import build_companion_lifeform

    # Aggressive idle-timeout config so a few ticks close the scene.
    config = LifeformConfig(
        tick=TickEngineConfig(system_tick_seconds=0.001),
        idle_close_after_system_ticks=2,
    )
    lifeform = build_companion_lifeform(config=config)
    session = lifeform.create_session(session_id="idle-reconcile")
    await session.run_turn("Hi.")
    _seed_case_records(
        session,
        (
            _case(
                "provisional-idle-weak",
                lifecycle=CaseLifecycle.PROVISIONAL,
                relevance_score=0.10,
                confidence=0.70,
            ),
        ),
    )
    # Tick forward enough to exceed the idle threshold, which closes
    # the scene with ``drain_slow_loop=False``.
    await session.advance_tick(system_ticks=5, reason="idle-sweep")
    reconcile = session.latest_case_memory_reconcile
    assert isinstance(reconcile, ProvisionalReconcileResult)
    assert "provisional-idle-weak" in reconcile.retired


# ---------------------------------------------------------------------------
# No scene open: end_scene returns None and does NOT fire reconcile
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_scene_without_open_scene_does_not_fire_reconcile() -> None:
    from lifeform_domain_emogpt import build_companion_lifeform

    lifeform = build_companion_lifeform()
    session = lifeform.create_session(session_id="scene-none")
    # No turns run => no open scene.
    closed = await session.end_scene(reason="noop")
    assert closed is None
    # Reconcile should not have fired either \u2014 ``latest_case_memory_reconcile``
    # stays at its initial ``None`` value.
    assert session.latest_case_memory_reconcile is None


# ---------------------------------------------------------------------------
# Multi-scene: the decision trace is reset per scene close
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconcile_trace_refreshes_per_scene_close() -> None:
    from lifeform_domain_emogpt import build_companion_lifeform

    lifeform = build_companion_lifeform()
    session = lifeform.create_session(session_id="scene-reconcile-multi")
    await session.run_turn("First scene.")
    _seed_case_records(
        session,
        (
            _case(
                "weak-in-scene-1",
                lifecycle=CaseLifecycle.PROVISIONAL,
                relevance_score=0.10,
                confidence=0.80,
            ),
        ),
    )
    await session.end_scene(reason="first-scene-done")
    first_result = session.latest_case_memory_reconcile
    assert isinstance(first_result, ProvisionalReconcileResult)
    assert "weak-in-scene-1" in first_result.retired

    # Open a fresh scene. Seed a strong provisional and confirm the
    # second reconcile reflects the new decision \u2014 not the stale
    # retire from scene 1.
    await session.run_turn("Second scene.")
    _seed_case_records(
        session,
        (
            _case(
                "strong-in-scene-2",
                lifecycle=CaseLifecycle.PROVISIONAL,
                relevance_score=0.80,
                confidence=0.70,
            ),
        ),
    )
    await session.end_scene(reason="second-scene-done")
    second_result = session.latest_case_memory_reconcile
    assert isinstance(second_result, ProvisionalReconcileResult)
    assert "strong-in-scene-2" in second_result.promoted
    # The retired set in scene-2's result must not carry over the
    # scene-1 record id \u2014 reconcile only sees lifecycles != VALIDATED
    # at THIS scene-end.
    assert "weak-in-scene-1" not in second_result.retired
