"""Unit tests for ``lifeform_thinking.adapter.ThinkingAdapter`` (Gap 4 slice 2c).

Covers:

* Happy path: submit after turn, collect before next turn, artifacts
  appear in ``latest_artifacts_by_consumer``.
* DISABLED wiring short-circuits submit to CANCELLED.
* Fingerprint guard: if ``on_turn_begin`` sees snapshots that drifted,
  the artifact is filtered out (the scheduler flips it to STALE and
  the adapter does not publish it).
* Missing-scope on submit: the adapter is fail-quiet and skips the
  batch; the next ``on_turn_begin`` sees no pending tasks.
* Drain: scene-close drains in-flight tasks and still populates
  latest_artifacts_by_consumer for completed ones.
* Adapter snapshot shape is stable across reads.
"""

from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import dataclass, field
from typing import Any

import pytest

from lifeform_thinking import (
    APPLIABLE_THINKING_TASK_STATUSES,
    CONSUMER_SELF_TEMPORAL,
    CONSUMER_WORLD_TEMPORAL,
    FingerprintScope,
    MID_REFLECTION_SCOPE,
    MidReflectionPayload,
    ThinkingAdapter,
    ThinkingAdapterSnapshot,
    ThinkingArtifact,
    ThinkingScheduler,
    ThinkingTaskStatus,
    ThinkingWiringLevel,
    build_default_thinking_adapter,
)


# ---------------------------------------------------------------------------
# Fake snapshot shapes (mirror the shape the real kernel publishes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeTrack:
    tension_level: float = 0.3


@dataclass(frozen=True)
class _FakeDualTrackValue:
    cross_track_tension: float = 0.2
    world_track: _FakeTrack = field(default_factory=_FakeTrack)
    self_track: _FakeTrack = field(default_factory=_FakeTrack)


@dataclass(frozen=True)
class _FakeRegimeIdentity:
    regime_id: str = "problem_solving"


@dataclass(frozen=True)
class _FakeRegimeValue:
    active_regime: _FakeRegimeIdentity = field(default_factory=_FakeRegimeIdentity)


@dataclass(frozen=True)
class _FakeSnapshot:
    slot_name: str
    version: int
    value: Any


def _make_snapshots(
    *,
    version: int = 1,
    cross_track_tension: float = 0.2,
    regime_id: str = "problem_solving",
) -> dict[str, _FakeSnapshot]:
    dual_track = _FakeDualTrackValue(cross_track_tension=cross_track_tension)
    regime = _FakeRegimeValue(
        active_regime=_FakeRegimeIdentity(regime_id=regime_id)
    )
    return {
        "dual_track": _FakeSnapshot("dual_track", version, dual_track),
        "regime": _FakeSnapshot("regime", version, regime),
    }


# ---------------------------------------------------------------------------
# Basic construction / invariants
# ---------------------------------------------------------------------------


def test_build_default_adapter_has_shadow_wiring() -> None:
    adapter = build_default_thinking_adapter()
    assert adapter.wiring_level is ThinkingWiringLevel.SHADOW
    assert adapter.latest_artifacts_by_consumer == {}


def test_adapter_accepts_custom_wiring_level() -> None:
    adapter = build_default_thinking_adapter(
        wiring_level=ThinkingWiringLevel.DISABLED,
    )
    assert adapter.wiring_level is ThinkingWiringLevel.DISABLED


def test_adapter_requires_worker_when_scheduler_provided() -> None:
    scheduler = ThinkingScheduler()
    with pytest.raises(ValueError, match="worker"):
        ThinkingAdapter(scheduler=scheduler)


# ---------------------------------------------------------------------------
# Happy path: submit -> collect -> latest artifacts
# ---------------------------------------------------------------------------


async def test_on_turn_end_submits_world_and_self_tasks() -> None:
    adapter = build_default_thinking_adapter()
    snapshots = _make_snapshots()
    await adapter.on_turn_end(snapshots=snapshots, turn_index=0)
    snapshot = adapter.snapshot()
    # Two tasks pending after a single turn.
    assert len(snapshot.pending_task_ids) == 2
    # Scheduler saw them, but artifacts are still in flight / completing.
    sched_snap = snapshot.scheduler_snapshot
    assert sched_snap.total_submitted == 2


async def test_on_turn_begin_collects_previous_artifacts() -> None:
    adapter = build_default_thinking_adapter()
    snapshots = _make_snapshots()
    await adapter.on_turn_end(snapshots=snapshots, turn_index=0)
    # Now the next turn begins; pass the same snapshots (no drift) so
    # the fingerprint guard is satisfied.
    await adapter.on_turn_begin(snapshots=snapshots, turn_index=1)
    artifacts = adapter.latest_artifacts_by_consumer
    assert CONSUMER_WORLD_TEMPORAL in artifacts
    assert CONSUMER_SELF_TEMPORAL in artifacts
    world = artifacts[CONSUMER_WORLD_TEMPORAL]
    self_ = artifacts[CONSUMER_SELF_TEMPORAL]
    assert world.status is ThinkingTaskStatus.COMPLETED
    assert self_.status is ThinkingTaskStatus.COMPLETED
    # Payload shape matches the mid-reflection worker's contract.
    assert isinstance(world.payload, MidReflectionPayload)
    assert world.payload.track == "world"
    assert self_.payload.track == "self"


async def test_latest_world_and_self_accessors_match_dict() -> None:
    adapter = build_default_thinking_adapter()
    snapshots = _make_snapshots()
    await adapter.on_turn_end(snapshots=snapshots, turn_index=0)
    await adapter.on_turn_begin(snapshots=snapshots, turn_index=1)
    assert adapter.latest_world_artifact is not None
    assert adapter.latest_self_artifact is not None
    assert (
        adapter.latest_world_artifact
        is adapter.latest_artifacts_by_consumer[CONSUMER_WORLD_TEMPORAL]
    )


async def test_adapter_across_multiple_turns_keeps_latest() -> None:
    adapter = build_default_thinking_adapter()
    snapshots_a = _make_snapshots(version=1, regime_id="problem_solving")
    snapshots_b = _make_snapshots(version=2, regime_id="repair_and_deescalation")

    # Turn 0 -> submit with A
    await adapter.on_turn_end(snapshots=snapshots_a, turn_index=0)
    # Turn 1 -> begin with A (collect), then submit with B
    await adapter.on_turn_begin(snapshots=snapshots_a, turn_index=1)
    first = adapter.latest_world_artifact
    assert first is not None
    assert first.payload.observed_regime_id == "problem_solving"

    await adapter.on_turn_end(snapshots=snapshots_b, turn_index=1)
    # Turn 2 -> begin with B, second batch collected
    await adapter.on_turn_begin(snapshots=snapshots_b, turn_index=2)
    second = adapter.latest_world_artifact
    assert second is not None
    assert second.payload.observed_regime_id == "repair_and_deescalation"
    # The update is monotonic: the slot now holds the newer artifact.
    assert first.task_id != second.task_id


# ---------------------------------------------------------------------------
# Fingerprint guard
# ---------------------------------------------------------------------------


async def test_on_turn_begin_drops_stale_artifact_when_snapshots_drifted() -> None:
    adapter = build_default_thinking_adapter()
    submit_snapshots = _make_snapshots(version=1, regime_id="problem_solving")
    await adapter.on_turn_end(snapshots=submit_snapshots, turn_index=0)

    # Before collect, advance scheduler: let workers actually finish.
    await adapter.scheduler.drain()
    # Inject drift: version changed -> fingerprint mismatch.
    drifted = _make_snapshots(version=2, regime_id="problem_solving")
    await adapter.on_turn_begin(snapshots=drifted, turn_index=1)
    # No appliable artifacts because both got flipped to STALE.
    assert adapter.latest_world_artifact is None
    assert adapter.latest_self_artifact is None
    # Scheduler records stale count.
    sched_snap = adapter.scheduler.snapshot()
    assert sched_snap.total_stale == 2


async def test_on_turn_begin_with_empty_snapshots_cancels_pending() -> None:
    adapter = build_default_thinking_adapter()
    snapshots = _make_snapshots()
    await adapter.on_turn_end(snapshots=snapshots, turn_index=0)
    # Empty snapshots on turn begin -> adapter cancels pending tasks.
    await adapter.on_turn_begin(snapshots={}, turn_index=1)
    assert adapter.latest_world_artifact is None
    assert adapter.latest_self_artifact is None


# ---------------------------------------------------------------------------
# DISABLED wiring
# ---------------------------------------------------------------------------


async def test_disabled_wiring_short_circuits_submit() -> None:
    adapter = build_default_thinking_adapter(
        wiring_level=ThinkingWiringLevel.DISABLED,
    )
    snapshots = _make_snapshots()
    await adapter.on_turn_end(snapshots=snapshots, turn_index=0)
    sched_snap = adapter.scheduler.snapshot()
    # Scheduler recorded 2 submitted but both immediately CANCELLED.
    assert sched_snap.total_submitted == 2
    assert sched_snap.total_cancelled == 2
    assert sched_snap.total_completed == 0
    # Collection finds nothing appliable.
    await adapter.on_turn_begin(snapshots=snapshots, turn_index=1)
    assert adapter.latest_artifacts_by_consumer == {}


# ---------------------------------------------------------------------------
# Missing scope on submit
# ---------------------------------------------------------------------------


async def test_on_turn_end_skips_when_upstream_missing_slots() -> None:
    adapter = build_default_thinking_adapter()
    # Only half the scope -> adapter skips submission.
    partial = {k: v for k, v in _make_snapshots().items() if k == "dual_track"}
    await adapter.on_turn_end(snapshots=partial, turn_index=0)
    sched_snap = adapter.scheduler.snapshot()
    assert sched_snap.total_submitted == 0
    snap = adapter.snapshot()
    assert snap.pending_task_ids == ()


async def test_on_turn_begin_no_pending_is_no_op() -> None:
    adapter = build_default_thinking_adapter()
    snapshots = _make_snapshots()
    # No on_turn_end first -> no pending batch.
    await adapter.on_turn_begin(snapshots=snapshots, turn_index=0)
    assert adapter.latest_artifacts_by_consumer == {}


# ---------------------------------------------------------------------------
# Drain
# ---------------------------------------------------------------------------


async def test_drain_completes_in_flight_tasks_and_exposes_them() -> None:
    adapter = build_default_thinking_adapter()
    snapshots = _make_snapshots()
    await adapter.on_turn_end(snapshots=snapshots, turn_index=0)
    artifacts = await adapter.drain()
    assert len(artifacts) == 2
    # Both COMPLETED (drain doesn't run fingerprint guard; these
    # are the raw scheduler outputs).
    assert all(
        a.status is ThinkingTaskStatus.COMPLETED for a in artifacts
    )
    # Adapter copies them into the latest-artifacts map.
    assert CONSUMER_WORLD_TEMPORAL in adapter.latest_artifacts_by_consumer
    assert CONSUMER_SELF_TEMPORAL in adapter.latest_artifacts_by_consumer


async def test_drain_clears_pending_batch() -> None:
    adapter = build_default_thinking_adapter()
    snapshots = _make_snapshots()
    await adapter.on_turn_end(snapshots=snapshots, turn_index=0)
    await adapter.drain()
    snap = adapter.snapshot()
    assert snap.pending_task_ids == ()


# ---------------------------------------------------------------------------
# Snapshot shape
# ---------------------------------------------------------------------------


async def test_adapter_snapshot_is_stable_and_complete() -> None:
    adapter = build_default_thinking_adapter()
    snapshots = _make_snapshots()
    await adapter.on_turn_end(snapshots=snapshots, turn_index=0)
    await adapter.on_turn_begin(snapshots=snapshots, turn_index=1)
    snap = adapter.snapshot()
    assert isinstance(snap, ThinkingAdapterSnapshot)
    assert snap.wiring_level is ThinkingWiringLevel.SHADOW
    assert snap.latest_world_artifact is not None
    assert snap.latest_self_artifact is not None
    assert snap.scheduler_snapshot.total_completed == 2
    # Re-reading produces an equal-but-distinct dataclass (not shared ref).
    snap2 = adapter.snapshot()
    assert snap2 == snap
    assert snap2 is not snap


# ---------------------------------------------------------------------------
# Custom worker / scope injection
# ---------------------------------------------------------------------------


async def test_custom_scope_must_cover_worker_reads() -> None:
    calls: list[str] = []

    async def _echo_worker(task, upstream):
        calls.append(task.task_id)
        return ThinkingArtifact(
            task_id=task.task_id,
            status=ThinkingTaskStatus.COMPLETED,
            payload={"echo": task.task_id},
            produced_at_turn_index=task.requested_at_turn_index,
            consumer_owner=task.consumer_owner,
        )

    scheduler = ThinkingScheduler()
    scope = FingerprintScope(slot_names=("dual_track",))
    adapter = ThinkingAdapter(
        scheduler=scheduler,
        worker=_echo_worker,
        scope=scope,
    )
    # With custom scope covering only dual_track, missing regime
    # should no longer block (scope doesn't declare it). Supply
    # only dual_track.
    snapshots = {k: v for k, v in _make_snapshots().items() if k == "dual_track"}
    await adapter.on_turn_end(snapshots=snapshots, turn_index=0)
    await adapter.drain()
    # Both world + self tasks submitted, both ran.
    assert len(calls) == 2
