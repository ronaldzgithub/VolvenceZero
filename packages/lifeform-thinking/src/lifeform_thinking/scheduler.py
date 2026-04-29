"""ThinkingScheduler \u2014 async task-lifecycle manager with fingerprint guard.

The scheduler is *deliberately small*: it owns the lifecycle state
machine (QUEUED -> RUNNING -> terminal) and the fingerprint check at
submit + complete time. It does NOT:

* mutate kernel owners (workers return artifacts; consumers apply)
* inspect owner internals (it only sees ``Snapshot`` objects through
  the ``FingerprintScope`` protocol)
* schedule anything synchronously \u2014 all work runs on the caller's
  ``asyncio`` loop

Why a per-session scheduler rather than per-process: each
``LifeformSession`` has its own scene clock, its own regime history,
its own tick counter. Sharing a scheduler across sessions would
couple their task queues in a way that violates session isolation
(see ``docs/specs/core-package-boundary.md``). Host applications can
create their own aggregator that polls N schedulers if they need
cross-session dashboards.

R-IDs touched:

* **R8** (snapshot-first): workers consume snapshots; artifacts are
  immutable envelopes.
* **R11** (internal state publishable): current task states are
  exposed via ``ThinkingLoopSnapshot`` so observability / family
  report can describe what the scheduler is doing.
* **R15** (rollback-friendly): the scheduler has an explicit wiring
  level; at DISABLED the submit path no-ops.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable

from volvence_zero.thinking import (
    APPLIABLE_THINKING_TASK_STATUSES,
    TERMINAL_THINKING_TASK_STATUSES,
    ThinkingArtifact,
    ThinkingTask,
    ThinkingTaskStatus,
)

from lifeform_thinking.fingerprint import (
    FingerprintScope,
    compute_fingerprint,
    fingerprints_match,
)


_LOG = logging.getLogger("lifeform_thinking.scheduler")


class ThinkingWiringLevel(str, Enum):
    """Three-state rollout gate for the scheduler itself.

    Independent of per-task statuses \u2014 this gates whether the
    scheduler even accepts submissions.
    """

    DISABLED = "disabled"
    SHADOW = "shadow"
    ACTIVE = "active"


WorkerFunc = Callable[[ThinkingTask, Mapping[str, Any]], Awaitable[ThinkingArtifact]]
"""Worker contract: receive a task + the upstream snapshots it
declared as its scope, return a ``ThinkingArtifact``. The scheduler
wraps this call and handles fingerprint guards / cancellation, so
worker code can stay focused on the actual reflection logic.

Workers MUST be read-only: they cannot call any owner's mutation
API. This is enforced by convention + contract test; see
``tests/contracts/test_thinking_worker_read_only.py``.
"""


@dataclass(frozen=True)
class _TaskRecord:
    """Internal bookkeeping: task + worker + asyncio task handle."""

    task: ThinkingTask
    worker: WorkerFunc
    scope: FingerprintScope
    upstream_at_submit: Mapping[str, Any]
    runner: asyncio.Task[ThinkingArtifact]


@dataclass(frozen=True)
class ThinkingLoopSnapshot:
    """Observability snapshot of the scheduler state.

    Exposed to ``LifeformSession`` consumers so they can describe
    what middle-frequency work is in flight without poking the
    scheduler internals. Immutable by construction; recomputed on
    each read.
    """

    wiring_level: ThinkingWiringLevel
    in_flight_task_ids: tuple[str, ...]
    completed_task_ids: tuple[str, ...]
    stale_task_ids: tuple[str, ...]
    failed_task_ids: tuple[str, ...]
    cancelled_task_ids: tuple[str, ...]
    total_submitted: int
    total_completed: int
    total_stale: int
    total_failed: int
    total_cancelled: int


class ThinkingScheduler:
    """Per-session middle-frequency task scheduler.

    Usage:

        scheduler = ThinkingScheduler()
        task_id = await scheduler.submit(
            task=task,
            worker=my_worker,
            scope=my_scope,
            upstream_snapshots=session_snapshots,
        )
        # later, when you want to apply:
        artifact = await scheduler.collect(
            task_id, current_snapshots=session_snapshots,
        )
        if artifact.is_appliable():
            consumer.apply(artifact.payload)

    The ``collect`` call recomputes the fingerprint against the
    current snapshots, so a task that ran to worker completion but
    is no longer fresh is explicitly flipped to ``STALE`` before
    the caller ever sees the payload.
    """

    def __init__(
        self,
        *,
        wiring_level: ThinkingWiringLevel = ThinkingWiringLevel.ACTIVE,
        max_concurrent_tasks: int = 2,
    ) -> None:
        if max_concurrent_tasks < 1:
            raise ValueError(
                f"max_concurrent_tasks must be >= 1, got {max_concurrent_tasks!r}"
            )
        self._wiring_level = wiring_level
        self._max_concurrent = max_concurrent_tasks
        self._records: dict[str, _TaskRecord] = {}
        self._final_artifacts: dict[str, ThinkingArtifact] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._total_submitted = 0
        self._total_completed = 0
        self._total_stale = 0
        self._total_failed = 0
        self._total_cancelled = 0

    # ------------------------------------------------------------------
    # Configuration / observability
    # ------------------------------------------------------------------

    @property
    def wiring_level(self) -> ThinkingWiringLevel:
        return self._wiring_level

    def set_wiring_level(self, level: ThinkingWiringLevel) -> None:
        """Transition the scheduler rollout gate.

        Only affects future ``submit`` calls; tasks already in flight
        continue to run and their artifacts remain collectable.
        Downgrading to DISABLED is an explicit, explainable operation
        (owner decides to stop scheduling new thinking work).
        """
        self._wiring_level = level

    def snapshot(self) -> ThinkingLoopSnapshot:
        """Return an immutable observation of current scheduler state.

        Bucketing by terminal status lets the family report emit
        per-bucket metrics (stale-rate, failed-rate) without the
        consumer learning the internal dict layout.
        """
        in_flight: list[str] = []
        completed: list[str] = []
        stale: list[str] = []
        failed: list[str] = []
        cancelled: list[str] = []
        for task_id, artifact in self._final_artifacts.items():
            if artifact.status is ThinkingTaskStatus.COMPLETED:
                completed.append(task_id)
            elif artifact.status is ThinkingTaskStatus.STALE:
                stale.append(task_id)
            elif artifact.status is ThinkingTaskStatus.FAILED:
                failed.append(task_id)
            elif artifact.status is ThinkingTaskStatus.CANCELLED:
                cancelled.append(task_id)
        for task_id in self._records:
            if task_id not in self._final_artifacts:
                in_flight.append(task_id)
        return ThinkingLoopSnapshot(
            wiring_level=self._wiring_level,
            in_flight_task_ids=tuple(in_flight),
            completed_task_ids=tuple(completed),
            stale_task_ids=tuple(stale),
            failed_task_ids=tuple(failed),
            cancelled_task_ids=tuple(cancelled),
            total_submitted=self._total_submitted,
            total_completed=self._total_completed,
            total_stale=self._total_stale,
            total_failed=self._total_failed,
            total_cancelled=self._total_cancelled,
        )

    # ------------------------------------------------------------------
    # Submission / collection
    # ------------------------------------------------------------------

    async def submit(
        self,
        *,
        task: ThinkingTask,
        worker: WorkerFunc,
        scope: FingerprintScope,
        upstream_snapshots: Mapping[str, Any],
    ) -> str:
        """Enqueue a task for async execution.

        When ``wiring_level`` is ``DISABLED`` the submission is a
        no-op: the scheduler records a ``CANCELLED`` artifact
        immediately so callers see a terminal status without
        consuming any worker cycles. This makes rollout easy \u2014
        flip the wiring level and the scheduler stops doing work.

        The caller's provided ``task.snapshot_fingerprint`` is
        REPLACED by a freshly-computed fingerprint over
        ``upstream_snapshots`` restricted to ``scope``. This
        guarantees the fingerprint matches what the scheduler
        actually observed (we do not trust caller-supplied digests).
        """
        self._total_submitted += 1
        if self._wiring_level is ThinkingWiringLevel.DISABLED:
            artifact = ThinkingArtifact(
                task_id=task.task_id,
                status=ThinkingTaskStatus.CANCELLED,
                payload=None,
                produced_at_turn_index=task.requested_at_turn_index,
                consumer_owner=task.consumer_owner,
            )
            self._final_artifacts[task.task_id] = artifact
            self._total_cancelled += 1
            return task.task_id
        fresh_fingerprint = compute_fingerprint(
            snapshots=upstream_snapshots, scope=scope
        )
        stamped_task = ThinkingTask(
            task_id=task.task_id,
            depth=task.depth,
            purpose=task.purpose,
            requested_at_turn_index=task.requested_at_turn_index,
            snapshot_fingerprint=fresh_fingerprint,
            consumer_owner=task.consumer_owner,
            deadline_at_turn_index=task.deadline_at_turn_index,
        )
        # Take a frozen local copy of upstream at submit time so the
        # worker sees a stable input even if the caller subsequently
        # mutates its dict. (Snapshot values themselves are frozen by
        # contract.)
        frozen_upstream: Mapping[str, Any] = dict(upstream_snapshots)
        runner = asyncio.create_task(
            self._run_worker(
                task=stamped_task,
                worker=worker,
                frozen_upstream=frozen_upstream,
            )
        )
        self._records[stamped_task.task_id] = _TaskRecord(
            task=stamped_task,
            worker=worker,
            scope=scope,
            upstream_at_submit=frozen_upstream,
            runner=runner,
        )
        return stamped_task.task_id

    async def _run_worker(
        self,
        *,
        task: ThinkingTask,
        worker: WorkerFunc,
        frozen_upstream: Mapping[str, Any],
    ) -> ThinkingArtifact:
        """Gate-protected worker execution.

        Catches all worker exceptions and turns them into FAILED
        artifacts with the exception class name \u2014 we do NOT let a
        worker bug propagate into the scheduler's event loop.
        """
        async with self._semaphore:
            try:
                artifact = await worker(task, frozen_upstream)
            except Exception as exc:  # noqa: BLE001 \u2014 worker isolation boundary
                _LOG.exception(
                    "ThinkingScheduler worker %s for task %s raised",
                    worker.__name__ if hasattr(worker, "__name__") else worker,
                    task.task_id,
                )
                artifact = ThinkingArtifact(
                    task_id=task.task_id,
                    status=ThinkingTaskStatus.FAILED,
                    payload=None,
                    produced_at_turn_index=task.requested_at_turn_index,
                    consumer_owner=task.consumer_owner,
                    error_class=type(exc).__name__,
                    error_detail=str(exc)[:512],
                )
            # Sanity: worker is supposed to stamp its own artifact's
            # task_id and consumer_owner. If it didn't, we overwrite
            # them here so downstream lookups work.
            if (
                artifact.task_id != task.task_id
                or artifact.consumer_owner != task.consumer_owner
            ):
                artifact = ThinkingArtifact(
                    task_id=task.task_id,
                    status=artifact.status,
                    payload=artifact.payload,
                    produced_at_turn_index=artifact.produced_at_turn_index,
                    consumer_owner=task.consumer_owner,
                    error_class=artifact.error_class,
                    error_detail=artifact.error_detail,
                )
            self._final_artifacts[task.task_id] = artifact
            if artifact.status is ThinkingTaskStatus.COMPLETED:
                self._total_completed += 1
            elif artifact.status is ThinkingTaskStatus.STALE:
                self._total_stale += 1
            elif artifact.status is ThinkingTaskStatus.FAILED:
                self._total_failed += 1
            elif artifact.status is ThinkingTaskStatus.CANCELLED:
                self._total_cancelled += 1
            return artifact

    async def collect(
        self,
        task_id: str,
        *,
        current_snapshots: Mapping[str, Any],
    ) -> ThinkingArtifact:
        """Wait for ``task_id`` to terminate, then enforce fingerprint guard.

        If the worker completed successfully BUT the current
        snapshots have drifted from the submit-time snapshots within
        the declared scope, the artifact is rewritten to ``STALE``
        before the caller sees it. That way the apply path can
        always trust ``artifact.is_appliable()``.
        """
        if task_id not in self._records and task_id not in self._final_artifacts:
            raise KeyError(f"Unknown task_id {task_id!r}")
        # Already-terminal cancelled / pre-DISABLED artifacts have
        # no runner to await; return them directly.
        if task_id in self._final_artifacts and task_id not in self._records:
            return self._final_artifacts[task_id]
        record = self._records[task_id]
        artifact = await record.runner
        if artifact.status not in TERMINAL_THINKING_TASK_STATUSES:
            # Should never happen \u2014 _run_worker always produces a
            # terminal artifact. Guard rail.
            raise RuntimeError(
                f"ThinkingScheduler collected non-terminal artifact "
                f"for task {task_id!r}: status={artifact.status!r}"
            )
        # Fingerprint guard: if the worker's artifact is COMPLETED
        # but the current scope fingerprint does not match the
        # one the task was stamped with, flip to STALE.
        if artifact.status is ThinkingTaskStatus.COMPLETED:
            current_fp = compute_fingerprint(
                snapshots=current_snapshots, scope=record.scope
            )
            if not fingerprints_match(
                task_fingerprint=record.task.snapshot_fingerprint,
                current_fingerprint=current_fp,
            ):
                artifact = ThinkingArtifact(
                    task_id=artifact.task_id,
                    status=ThinkingTaskStatus.STALE,
                    payload=artifact.payload,
                    produced_at_turn_index=artifact.produced_at_turn_index,
                    consumer_owner=artifact.consumer_owner,
                )
                self._final_artifacts[task_id] = artifact
                self._total_completed -= 1
                self._total_stale += 1
        return artifact

    def try_get(self, task_id: str) -> ThinkingArtifact | None:
        """Return the terminal artifact if the task finished, else None.

        Non-blocking: for callers (e.g. prompt planner) that want to
        glance at scheduler state without awaiting a pending task.
        Does NOT run the fingerprint guard \u2014 that is the job of
        ``collect``. Use this for observability only.
        """
        return self._final_artifacts.get(task_id)

    async def cancel(self, task_id: str) -> bool:
        """Best-effort cancellation.

        If the task is still running, we request asyncio
        cancellation; the worker's post-cancel artifact is recorded
        as ``CANCELLED``. If it has already terminated, returns
        False (caller can check ``try_get`` for the final state).
        """
        if task_id not in self._records:
            return False
        if task_id in self._final_artifacts:
            return False
        record = self._records[task_id]
        record.runner.cancel()
        try:
            await record.runner
        except asyncio.CancelledError:
            pass
        # Worker might have already recorded a FAILED / COMPLETED
        # artifact before cancellation landed. Only overwrite when
        # we genuinely see no recorded artifact.
        if task_id not in self._final_artifacts:
            self._final_artifacts[task_id] = ThinkingArtifact(
                task_id=record.task.task_id,
                status=ThinkingTaskStatus.CANCELLED,
                payload=None,
                produced_at_turn_index=record.task.requested_at_turn_index,
                consumer_owner=record.task.consumer_owner,
            )
            self._total_cancelled += 1
        return True

    async def drain(self) -> tuple[ThinkingArtifact, ...]:
        """Await every in-flight task. Returns all terminal artifacts.

        Intended for test harnesses and for session teardown when
        the host wants a clean shutdown. The fingerprint guard does
        NOT run on drain \u2014 drained artifacts are the raw worker
        outputs. Use ``collect`` per-task for apply-path safety.
        """
        in_flight = [
            record.runner
            for task_id, record in self._records.items()
            if task_id not in self._final_artifacts
        ]
        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)
        return tuple(self._final_artifacts.values())


__all__ = [
    "APPLIABLE_THINKING_TASK_STATUSES",
    "TERMINAL_THINKING_TASK_STATUSES",
    "ThinkingLoopSnapshot",
    "ThinkingScheduler",
    "ThinkingWiringLevel",
    "WorkerFunc",
]
