"""End-to-end: ``ThinkingAdapter`` drives a real ``LifeformSession``.

Validates Gap 4 slice 2c production wiring:

* A lifeform built with a ``thinking_adapter_factory`` gets a fresh
  adapter per session.
* Each turn submits world + self lane tasks AFTER the kernel runs.
* The next turn's ``on_turn_begin`` collects them; appliable
  artifacts appear in ``LifeformSession.latest_thinking_artifacts_by_consumer``.
* Scene close drains all in-flight tasks so no worker outlives
  the scene.
* A session WITHOUT an adapter behaves identically to before (no
  behavioural change; thinking_adapter_snapshot is None).
* A broken adapter (raises from every hook) does NOT break the
  turn \u2014 the session logs + continues.
"""

from __future__ import annotations

import asyncio
from typing import Any, Mapping

from lifeform_thinking import (
    CONSUMER_SELF_TEMPORAL,
    CONSUMER_WORLD_TEMPORAL,
    MidReflectionPayload,
    ThinkingAdapter,
    ThinkingTaskStatus,
    ThinkingWiringLevel,
    build_default_thinking_adapter,
)


async def test_lifeform_session_without_adapter_behaves_unchanged() -> None:
    """Baseline: no adapter factory -> session.thinking_adapter is None."""
    from lifeform_domain_emogpt import build_companion_lifeform

    lifeform = build_companion_lifeform()
    session = lifeform.create_session(session_id="thinking-e2e-baseline")
    await session.run_turn("hello")

    assert session.thinking_adapter is None
    assert session.thinking_adapter_snapshot is None
    assert session.latest_thinking_artifacts_by_consumer == {}


async def test_lifeform_session_with_adapter_exposes_artifacts() -> None:
    """Turn 1 -> submits tasks. Turn 2 -> collects them.

    After turn 2, the adapter's latest_artifacts_by_consumer should
    carry the world + self lane reflection payloads from the turn-1
    upstream, stamped COMPLETED.
    """
    from lifeform_domain_emogpt import build_companion_lifeform

    base_lifeform = build_companion_lifeform()
    lifeform_with_thinking = base_lifeform.with_thinking_adapter_factory(
        lambda: build_default_thinking_adapter(
            wiring_level=ThinkingWiringLevel.SHADOW,
        ),
    )
    session = lifeform_with_thinking.create_session(
        session_id="thinking-e2e-shadow"
    )

    assert session.thinking_adapter is not None
    assert session.thinking_adapter_snapshot is not None

    # Turn 1: kernel runs, adapter.on_turn_end submits 2 tasks.
    await session.run_turn("tell me about yourself")
    snap_after_t1 = session.thinking_adapter_snapshot
    assert snap_after_t1 is not None
    # Both tasks pending until next turn collects them.
    assert len(snap_after_t1.pending_task_ids) == 2

    # Turn 2: adapter.on_turn_begin collects the batch (snapshots
    # from turn 1 still match since the kernel hasn't run yet).
    # Then kernel runs + new batch is submitted.
    await session.run_turn("keep going")
    artifacts = session.latest_thinking_artifacts_by_consumer
    assert CONSUMER_WORLD_TEMPORAL in artifacts, (
        f"Expected world-lane artifact collected after turn 2; "
        f"got keys: {list(artifacts)!r}"
    )
    assert CONSUMER_SELF_TEMPORAL in artifacts
    world = artifacts[CONSUMER_WORLD_TEMPORAL]
    self_ = artifacts[CONSUMER_SELF_TEMPORAL]
    assert world.status is ThinkingTaskStatus.COMPLETED
    assert self_.status is ThinkingTaskStatus.COMPLETED
    assert isinstance(world.payload, MidReflectionPayload)
    assert isinstance(self_.payload, MidReflectionPayload)
    assert world.payload.track == "world"
    assert self_.payload.track == "self"

    # A fresh batch has also been submitted for turn 2.
    snap_after_t2 = session.thinking_adapter_snapshot
    assert snap_after_t2 is not None
    assert len(snap_after_t2.pending_task_ids) == 2
    assert snap_after_t2.scheduler_snapshot.total_submitted == 4


async def test_scene_close_drains_thinking_tasks() -> None:
    """Scene close must not leave workers running.

    After end_scene, the adapter's pending_task_ids is empty and
    any in-flight task has reached a terminal state.
    """
    from lifeform_domain_emogpt import build_companion_lifeform

    base_lifeform = build_companion_lifeform()
    lifeform = base_lifeform.with_thinking_adapter_factory(
        lambda: build_default_thinking_adapter(
            wiring_level=ThinkingWiringLevel.SHADOW,
        ),
    )
    session = lifeform.create_session(session_id="thinking-e2e-drain")
    await session.run_turn("hello")
    # Pending from turn 1.
    assert len(session.thinking_adapter_snapshot.pending_task_ids) == 2

    closed = await session.end_scene(reason="test-scene-close")
    assert closed is not None
    # After drain, no pending batches remain.
    snap = session.thinking_adapter_snapshot
    assert snap.pending_task_ids == ()
    # All submitted tasks reached terminal states.
    sched_snap = snap.scheduler_snapshot
    assert sched_snap.total_submitted == (
        sched_snap.total_completed
        + sched_snap.total_stale
        + sched_snap.total_failed
        + sched_snap.total_cancelled
    )


async def test_broken_adapter_does_not_break_turn() -> None:
    """A buggy adapter must not propagate exceptions into the turn path."""

    class _BrokenAdapter:
        """Raises on every hook \u2014 simulates a worst-case plugin bug."""

        async def on_turn_begin(
            self, *, snapshots: Mapping[str, Any], turn_index: int
        ) -> None:
            raise RuntimeError("broken on_turn_begin")

        async def on_turn_end(
            self, *, snapshots: Mapping[str, Any], turn_index: int
        ) -> None:
            raise RuntimeError("broken on_turn_end")

        async def drain(self) -> Any:
            raise RuntimeError("broken drain")

        def snapshot(self) -> None:
            return None

        @property
        def latest_artifacts_by_consumer(self) -> Mapping[str, Any]:
            return {}

    from lifeform_domain_emogpt import build_companion_lifeform

    base_lifeform = build_companion_lifeform()
    lifeform = base_lifeform.with_thinking_adapter_factory(
        lambda: _BrokenAdapter(),
    )
    session = lifeform.create_session(session_id="thinking-e2e-broken")

    # Turn still runs successfully despite every hook raising.
    result = await session.run_turn("hello")
    assert result is not None
    assert session.latest_response_text  # kernel still produced output

    # Scene close still works.
    closed = await session.end_scene()
    assert closed is not None


async def test_disabled_wiring_produces_no_live_artifacts() -> None:
    """DISABLED wiring: adapter is attached but every task is CANCELLED."""
    from lifeform_domain_emogpt import build_companion_lifeform

    base_lifeform = build_companion_lifeform()
    lifeform = base_lifeform.with_thinking_adapter_factory(
        lambda: build_default_thinking_adapter(
            wiring_level=ThinkingWiringLevel.DISABLED,
        ),
    )
    session = lifeform.create_session(session_id="thinking-e2e-disabled")
    await session.run_turn("hello")
    await session.run_turn("again")

    artifacts = session.latest_thinking_artifacts_by_consumer
    assert artifacts == {}, (
        f"DISABLED wiring should produce no appliable artifacts; "
        f"got keys: {list(artifacts)!r}"
    )
    snap = session.thinking_adapter_snapshot
    assert snap is not None
    assert snap.scheduler_snapshot.total_cancelled >= 2
    assert snap.scheduler_snapshot.total_completed == 0
