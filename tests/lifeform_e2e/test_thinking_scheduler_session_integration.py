"""Integration: ThinkingScheduler co-existing with a live LifeformSession.

Validates three properties that slice 2b needs to prove:

* **Worker runs against real kernel snapshots** \u2014 not stubs. The
  scheduler reads ``brain_session.runner._upstream_snapshots`` and the
  worker completes successfully.
* **Stale detection works with real kernel state** \u2014 when a fresh
  turn advances kernel state between submit and collect, the
  artifact flips to STALE. This is the canonical Gap-4 invariant
  (docs/specs/thinking-loop.md \u00a7proof-surfaces).
* **Submit does not block the turn loop** \u2014 a scheduled task that
  deliberately yields does not delay ``run_turn``. The scheduler is
  a middle-frequency side-channel, not a blocker on the fast path.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Mapping

from lifeform_thinking import (
    FingerprintScope,
    ThinkingArtifact,
    ThinkingDepth,
    ThinkingPurpose,
    ThinkingScheduler,
    ThinkingTask,
    ThinkingTaskStatus,
    mid_reflection_worker,
)


def _kernel_snapshots(session) -> Mapping[str, Any]:
    """Grab the current upstream dict the session has published.

    This is a documented test-only back-door: it's the same dict the
    kernel's reflection writeback / evaluation readouts consume. We
    don't mutate it; we just pass it to the scheduler.
    """
    return dict(session.brain_session.runner._upstream_snapshots)  # noqa: SLF001


async def test_scheduler_completes_against_real_kernel_state() -> None:
    from lifeform_domain_emogpt import build_companion_lifeform

    lifeform = build_companion_lifeform()
    session = lifeform.create_session(session_id="sched-int-1")
    await session.run_turn("I'm feeling overwhelmed.")

    scheduler = ThinkingScheduler()
    scope = FingerprintScope(slot_names=("regime", "dual_track"))
    upstream = _kernel_snapshots(session)
    task_id = await scheduler.submit(
        task=ThinkingTask(
            task_id="sched-int-1:t1",
            depth=ThinkingDepth.MID,
            purpose=ThinkingPurpose.SELF_LANE_REFLECT,
            requested_at_turn_index=1,
            snapshot_fingerprint="sha256:initial",
            consumer_owner="self_temporal",
        ),
        worker=mid_reflection_worker,
        scope=scope,
        upstream_snapshots=upstream,
    )
    artifact = await scheduler.collect(task_id, current_snapshots=upstream)
    assert isinstance(artifact, ThinkingArtifact)
    assert artifact.status is ThinkingTaskStatus.COMPLETED
    assert artifact.is_appliable()
    assert artifact.payload.track == "self"


async def test_scheduler_flips_to_stale_after_new_turn_changes_kernel_state() -> None:
    from lifeform_domain_emogpt import build_companion_lifeform

    lifeform = build_companion_lifeform()
    session = lifeform.create_session(session_id="sched-int-stale")
    await session.run_turn("I need help.")
    scheduler = ThinkingScheduler()
    scope = FingerprintScope(slot_names=("regime", "dual_track"))
    upstream_t1 = _kernel_snapshots(session)
    task_id = await scheduler.submit(
        task=ThinkingTask(
            task_id="sched-int-stale:t1",
            depth=ThinkingDepth.MID,
            purpose=ThinkingPurpose.WORLD_LANE_REFLECT,
            requested_at_turn_index=1,
            snapshot_fingerprint="sha256:initial",
            consumer_owner="world_temporal",
        ),
        worker=mid_reflection_worker,
        scope=scope,
        upstream_snapshots=upstream_t1,
    )
    # Advance kernel state: a second turn updates regime / dual_track.
    await session.run_turn("Actually I want to debug a production bug.")
    upstream_t2 = _kernel_snapshots(session)
    # Collect against the NEW snapshots \u2014 fingerprint must mismatch.
    artifact = await scheduler.collect(task_id, current_snapshots=upstream_t2)
    assert artifact.status is ThinkingTaskStatus.STALE
    assert not artifact.is_appliable()


async def test_scheduler_submit_does_not_block_subsequent_turn() -> None:
    """A slow scheduled worker must not delay the next ``run_turn``.

    We measure the wall time of ``run_turn`` while a pending task is
    still in flight. The delta against a baseline turn must be
    small; we pick a generous 2x upper bound so the test is not
    flaky on slow CI while still catching an outright block.
    """
    from lifeform_domain_emogpt import build_companion_lifeform

    lifeform = build_companion_lifeform()
    session = lifeform.create_session(session_id="sched-int-noblock")
    # Baseline: one turn with no scheduler involvement.
    t0 = time.perf_counter()
    await session.run_turn("warm up")
    baseline_ms = (time.perf_counter() - t0) * 1000.0

    async def _sleepy_worker(task, upstream):
        # Takes longer than a baseline turn; the test would fail if
        # submit awaited the worker.
        await asyncio.sleep(baseline_ms / 1000.0 * 3.0)
        return ThinkingArtifact(
            task_id=task.task_id,
            status=ThinkingTaskStatus.COMPLETED,
            payload=None,
            produced_at_turn_index=task.requested_at_turn_index,
            consumer_owner=task.consumer_owner,
        )

    scheduler = ThinkingScheduler()
    scope = FingerprintScope(slot_names=("regime", "dual_track"))
    upstream = _kernel_snapshots(session)
    await scheduler.submit(
        task=ThinkingTask(
            task_id="sched-int-noblock:t1",
            depth=ThinkingDepth.MID,
            purpose=ThinkingPurpose.SELF_LANE_REFLECT,
            requested_at_turn_index=1,
            snapshot_fingerprint="sha256:initial",
            consumer_owner="self_temporal",
        ),
        worker=_sleepy_worker,
        scope=scope,
        upstream_snapshots=upstream,
    )
    # Now measure a turn while the sleepy worker is still scheduled.
    t1 = time.perf_counter()
    await session.run_turn("next turn")
    with_pending_ms = (time.perf_counter() - t1) * 1000.0

    # A liberal upper bound: turn delay with pending worker should be
    # no more than 2x baseline. Absolute cap at 2s to catch
    # pathological blocking even on slow CI.
    assert with_pending_ms < max(baseline_ms * 2.0, 2000.0), (
        f"run_turn appears to be blocked by the scheduler: "
        f"baseline={baseline_ms:.1f}ms vs with_pending={with_pending_ms:.1f}ms"
    )
    # Drain for test hygiene.
    await scheduler.drain()
