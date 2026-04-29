"""Unit tests for Gap 4 slice 2b: lifeform-thinking scheduler.

Covers the five proof surfaces from ``docs/specs/thinking-loop.md``
that slice 2b can validate:

* **Lifecycle state machine** \u2014 QUEUED -> RUNNING -> COMPLETED (or
  terminal peer); terminal artifacts are immutable.
* **Fingerprint mismatch => STALE** \u2014 a worker that ran on
  snapshots A, collected against snapshots B with different hashes,
  must be rewritten to STALE before the caller ever sees the
  payload.
* **Worker isolation** \u2014 worker exceptions get captured into FAILED
  with ``error_class`` populated; the scheduler never propagates a
  worker bug into the caller's loop.
* **Non-blocking** \u2014 ``submit`` does not await the worker; the
  caller can continue without being forced to ``collect``
  immediately.
* **DISABLED wiring is a no-op** \u2014 at ``ThinkingWiringLevel.DISABLED``
  submitted tasks flip straight to CANCELLED and no worker runs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from lifeform_thinking import (
    FingerprintScope,
    ThinkingArtifact,
    ThinkingDepth,
    ThinkingPurpose,
    ThinkingScheduler,
    ThinkingTask,
    ThinkingTaskStatus,
    ThinkingWiringLevel,
    compute_fingerprint,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeSnapshotValue:
    label: str
    level: float


@dataclass(frozen=True)
class _FakeSnapshot:
    slot_name: str
    version: int
    value: _FakeSnapshotValue


def _snap(slot: str, version: int, label: str, level: float = 0.5) -> _FakeSnapshot:
    return _FakeSnapshot(
        slot_name=slot,
        version=version,
        value=_FakeSnapshotValue(label=label, level=level),
    )


def _make_task(
    task_id: str,
    *,
    purpose: ThinkingPurpose = ThinkingPurpose.PROVISIONAL_RECONCILE,
    consumer_owner: str = "case_memory",
    fingerprint: str = "sha256:placeholder",
) -> ThinkingTask:
    return ThinkingTask(
        task_id=task_id,
        depth=ThinkingDepth.MID,
        purpose=purpose,
        requested_at_turn_index=1,
        snapshot_fingerprint=fingerprint,
        consumer_owner=consumer_owner,
    )


async def _echo_worker(task: ThinkingTask, upstream) -> ThinkingArtifact:
    return ThinkingArtifact(
        task_id=task.task_id,
        status=ThinkingTaskStatus.COMPLETED,
        payload={"label": "ok", "seen_slots": sorted(upstream.keys())},
        produced_at_turn_index=task.requested_at_turn_index,
        consumer_owner=task.consumer_owner,
    )


async def _explode_worker(task: ThinkingTask, upstream) -> ThinkingArtifact:
    raise RuntimeError("simulated worker crash")


async def _slow_worker(task: ThinkingTask, upstream) -> ThinkingArtifact:
    # Yield a few ticks so ``submit`` returns before the worker finishes;
    # the scheduler semantics require the submit path to be non-blocking.
    for _ in range(10):
        await asyncio.sleep(0)
    return ThinkingArtifact(
        task_id=task.task_id,
        status=ThinkingTaskStatus.COMPLETED,
        payload={"slow": True},
        produced_at_turn_index=task.requested_at_turn_index,
        consumer_owner=task.consumer_owner,
    )


# ---------------------------------------------------------------------------
# Happy path: submit -> collect -> COMPLETED
# ---------------------------------------------------------------------------


async def test_submit_and_collect_returns_completed_artifact() -> None:
    scheduler = ThinkingScheduler()
    scope = FingerprintScope(slot_names=("regime", "dual_track"))
    upstream = {
        "regime": _snap("regime", 1, "emotional_support"),
        "dual_track": _snap("dual_track", 1, "dual_state", level=0.3),
    }
    task_id = await scheduler.submit(
        task=_make_task("t-1"),
        worker=_echo_worker,
        scope=scope,
        upstream_snapshots=upstream,
    )
    artifact = await scheduler.collect(task_id, current_snapshots=upstream)
    assert artifact.status is ThinkingTaskStatus.COMPLETED
    assert artifact.is_appliable()
    assert artifact.task_id == task_id
    assert artifact.consumer_owner == "case_memory"
    assert artifact.payload["seen_slots"] == ["dual_track", "regime"]


# ---------------------------------------------------------------------------
# Fingerprint guard: mismatch at collect => STALE
# ---------------------------------------------------------------------------


async def test_fingerprint_mismatch_rewrites_completed_to_stale() -> None:
    scheduler = ThinkingScheduler()
    scope = FingerprintScope(slot_names=("regime",))
    snapshots_a = {"regime": _snap("regime", 1, "emotional_support")}
    snapshots_b = {
        # Same slot_name, DIFFERENT version + value -> different fingerprint.
        "regime": _snap("regime", 2, "problem_solving", level=0.9),
    }
    task_id = await scheduler.submit(
        task=_make_task("t-stale"),
        worker=_echo_worker,
        scope=scope,
        upstream_snapshots=snapshots_a,
    )
    # Worker ran against snapshots_a; collect against snapshots_b.
    artifact = await scheduler.collect(task_id, current_snapshots=snapshots_b)
    assert artifact.status is ThinkingTaskStatus.STALE
    assert not artifact.is_appliable()


async def test_fingerprint_match_keeps_artifact_appliable() -> None:
    scheduler = ThinkingScheduler()
    scope = FingerprintScope(slot_names=("regime",))
    snapshots = {"regime": _snap("regime", 1, "emotional_support")}
    task_id = await scheduler.submit(
        task=_make_task("t-fresh"),
        worker=_echo_worker,
        scope=scope,
        upstream_snapshots=snapshots,
    )
    artifact = await scheduler.collect(task_id, current_snapshots=snapshots)
    assert artifact.status is ThinkingTaskStatus.COMPLETED
    assert artifact.is_appliable()


# ---------------------------------------------------------------------------
# Worker exception => FAILED with error_class
# ---------------------------------------------------------------------------


async def test_worker_exception_is_captured_as_failed_artifact() -> None:
    scheduler = ThinkingScheduler()
    scope = FingerprintScope(slot_names=("regime",))
    snapshots = {"regime": _snap("regime", 1, "emotional_support")}
    task_id = await scheduler.submit(
        task=_make_task("t-explode"),
        worker=_explode_worker,
        scope=scope,
        upstream_snapshots=snapshots,
    )
    artifact = await scheduler.collect(task_id, current_snapshots=snapshots)
    assert artifact.status is ThinkingTaskStatus.FAILED
    assert artifact.error_class == "RuntimeError"
    assert "simulated worker crash" in artifact.error_detail
    assert artifact.payload is None
    assert not artifact.is_appliable()


# ---------------------------------------------------------------------------
# Non-blocking submit: control returns before worker finishes
# ---------------------------------------------------------------------------


async def test_submit_does_not_block_on_slow_worker() -> None:
    scheduler = ThinkingScheduler()
    scope = FingerprintScope(slot_names=("regime",))
    snapshots = {"regime": _snap("regime", 1, "emotional_support")}
    task_id = await scheduler.submit(
        task=_make_task("t-slow"),
        worker=_slow_worker,
        scope=scope,
        upstream_snapshots=snapshots,
    )
    # Immediately after submit, the scheduler should NOT have a
    # terminal artifact for this task (the slow worker is still
    # yielding via asyncio.sleep(0)).
    observed = scheduler.try_get(task_id)
    assert observed is None
    snap = scheduler.snapshot()
    assert task_id in snap.in_flight_task_ids
    # Now drain.
    await scheduler.collect(task_id, current_snapshots=snapshots)
    final_snap = scheduler.snapshot()
    assert task_id in final_snap.completed_task_ids
    assert task_id not in final_snap.in_flight_task_ids


# ---------------------------------------------------------------------------
# DISABLED wiring => immediate CANCELLED
# ---------------------------------------------------------------------------


async def test_disabled_wiring_turns_submissions_into_cancelled_noops() -> None:
    scheduler = ThinkingScheduler(wiring_level=ThinkingWiringLevel.DISABLED)
    scope = FingerprintScope(slot_names=("regime",))
    snapshots = {"regime": _snap("regime", 1, "emotional_support")}
    task_id = await scheduler.submit(
        task=_make_task("t-disabled"),
        worker=_echo_worker,
        scope=scope,
        upstream_snapshots=snapshots,
    )
    observed = scheduler.try_get(task_id)
    assert observed is not None
    assert observed.status is ThinkingTaskStatus.CANCELLED
    assert not observed.is_appliable()


async def test_set_wiring_level_affects_only_future_submissions() -> None:
    scheduler = ThinkingScheduler()
    scope = FingerprintScope(slot_names=("regime",))
    snapshots = {"regime": _snap("regime", 1, "emotional_support")}
    # First submission with ACTIVE.
    active_task_id = await scheduler.submit(
        task=_make_task("t-active"),
        worker=_echo_worker,
        scope=scope,
        upstream_snapshots=snapshots,
    )
    # Flip to DISABLED.
    scheduler.set_wiring_level(ThinkingWiringLevel.DISABLED)
    disabled_task_id = await scheduler.submit(
        task=_make_task("t-cancelled"),
        worker=_echo_worker,
        scope=scope,
        upstream_snapshots=snapshots,
    )
    active_result = await scheduler.collect(
        active_task_id, current_snapshots=snapshots
    )
    disabled_result = await scheduler.collect(
        disabled_task_id, current_snapshots=snapshots
    )
    assert active_result.status is ThinkingTaskStatus.COMPLETED
    assert disabled_result.status is ThinkingTaskStatus.CANCELLED


# ---------------------------------------------------------------------------
# Observability snapshot
# ---------------------------------------------------------------------------


async def test_snapshot_exposes_totals_and_per_bucket_task_ids() -> None:
    scheduler = ThinkingScheduler()
    scope = FingerprintScope(slot_names=("regime",))
    snapshots = {"regime": _snap("regime", 1, "emotional_support")}
    ok_id = await scheduler.submit(
        task=_make_task("t-ok"),
        worker=_echo_worker,
        scope=scope,
        upstream_snapshots=snapshots,
    )
    fail_id = await scheduler.submit(
        task=_make_task("t-fail"),
        worker=_explode_worker,
        scope=scope,
        upstream_snapshots=snapshots,
    )
    await scheduler.collect(ok_id, current_snapshots=snapshots)
    await scheduler.collect(fail_id, current_snapshots=snapshots)
    snap = scheduler.snapshot()
    assert snap.total_submitted == 2
    assert snap.total_completed == 1
    assert snap.total_failed == 1
    assert ok_id in snap.completed_task_ids
    assert fail_id in snap.failed_task_ids


# ---------------------------------------------------------------------------
# Fingerprint scope invariants
# ---------------------------------------------------------------------------


def test_fingerprint_scope_rejects_empty_slot_names() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        FingerprintScope(slot_names=())


def test_fingerprint_scope_rejects_duplicate_slot_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        FingerprintScope(slot_names=("regime", "regime"))


def test_compute_fingerprint_is_order_independent() -> None:
    scope_a = FingerprintScope(slot_names=("regime", "dual_track"))
    scope_b = FingerprintScope(slot_names=("dual_track", "regime"))
    snapshots = {
        "regime": _snap("regime", 1, "x"),
        "dual_track": _snap("dual_track", 2, "y"),
    }
    assert compute_fingerprint(snapshots=snapshots, scope=scope_a) == compute_fingerprint(
        snapshots=snapshots, scope=scope_b
    )


def test_compute_fingerprint_changes_with_version_bump() -> None:
    scope = FingerprintScope(slot_names=("regime",))
    snap_v1 = {"regime": _snap("regime", 1, "emotional_support")}
    snap_v2 = {"regime": _snap("regime", 2, "emotional_support")}
    assert compute_fingerprint(snapshots=snap_v1, scope=scope) != compute_fingerprint(
        snapshots=snap_v2, scope=scope
    )


def test_compute_fingerprint_fails_loudly_on_missing_slot() -> None:
    scope = FingerprintScope(slot_names=("regime",))
    with pytest.raises(KeyError):
        compute_fingerprint(snapshots={}, scope=scope)
