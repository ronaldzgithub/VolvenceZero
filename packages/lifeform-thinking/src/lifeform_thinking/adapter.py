"""ThinkingAdapter \u2014 production wiring for per-session thinking loop (Gap 4 slice 2c).

The adapter bridges the lifeform-layer session loop (``LifeformSession.run_turn``
and ``LifeformSession.end_scene``) to the pure-Python ``ThinkingScheduler``.
It is deliberately the ONLY place where the session's turn lifecycle
knows about thinking tasks \u2014 ``lifeform-core`` imports nothing from
``lifeform-thinking``, and instead calls a narrow protocol.

Wiring pattern per turn:

1. ``on_turn_begin(snapshots, turn_index)`` \u2014 collect any pending
   tasks submitted after the previous turn. The fingerprint guard
   uses ``snapshots`` (the previous-turn snapshots, still current
   because the kernel has not yet run this turn). Appliable
   artifacts are exposed via ``latest_artifacts`` for downstream
   consumers (prompt planner, observability).
2. Kernel runs.
3. ``on_turn_end(snapshots, turn_index)`` \u2014 submit a world-lane
   and a self-lane mid-reflection task whose upstream scope covers
   ``dual_track`` + ``regime``. Both are submitted concurrently;
   the scheduler's ``max_concurrent_tasks`` sema governs parallelism.

Wiring at scene close:

4. ``drain()`` \u2014 await all in-flight tasks so no worker outlives
   the scene. The final artifacts are kept for observability even
   after drain.

R-ID alignment:

* **R1** (multi-timescale): the adapter delivers the "middle" band
  between per-turn (FAST) and post-scene (SLOW).
* **R8** (snapshot-first): workers receive read-only snapshots at
  submit time; artifacts are the ONLY channel back.
* **R15** (rollback-friendly): DISABLED wiring level makes submit
  a no-op (CANCELLED artifacts), so flipping the wiring level off
  is a single switch.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from volvence_zero.thinking import (
    APPLIABLE_THINKING_TASK_STATUSES,
    ThinkingArtifact,
    ThinkingDepth,
    ThinkingPurpose,
    ThinkingTask,
    ThinkingTaskStatus,
)

from lifeform_thinking.fingerprint import FingerprintScope
from lifeform_thinking.scheduler import (
    ThinkingLoopSnapshot,
    ThinkingScheduler,
    ThinkingWiringLevel,
    WorkerFunc,
)
from lifeform_thinking.workers.mid_reflection import (
    MidReflectionPayload,
    controller_pressure_advisory_from_mid_reflection,
    mid_reflection_worker,
)


_LOG = logging.getLogger("lifeform_thinking.adapter")


MID_REFLECTION_SCOPE: FingerprintScope = FingerprintScope(
    slot_names=("dual_track", "regime"),
)
"""Declared fingerprint scope for ``mid_reflection_worker``.

Lives at module scope (not per-submit) so every mid-reflection task
uses the exact same scope set, making fingerprint comparison
deterministic. Callers that want a different scope build their own
adapter; this one does the Gap 4 slice 2c default.
"""


CONSUMER_WORLD_TEMPORAL: str = "world_temporal"
CONSUMER_SELF_TEMPORAL: str = "self_temporal"
"""Consumer owner IDs used in ``ThinkingTask.consumer_owner``.

They match the kernel slot names (``world_temporal`` / ``self_temporal``).
If a future owner wants to read the same mid-reflection payload,
they just read from ``latest_artifacts`` by the same key \u2014 these
strings are the wiring lookup keys, NOT enum values.
"""


@dataclass(frozen=True)
class ThinkingAdapterSnapshot:
    """Observability view of the adapter state.

    Separate from ``ThinkingLoopSnapshot`` (which is scheduler-scoped)
    because the adapter adds one extra bit: which task_id maps to
    which consumer owner for the MOST RECENT turn. Product code that
    wants to read "what is the latest world-lane reflection?" uses
    this snapshot.
    """

    wiring_level: ThinkingWiringLevel
    latest_artifacts_by_consumer: Mapping[str, ThinkingArtifact]
    pending_task_ids: tuple[str, ...]
    scheduler_snapshot: ThinkingLoopSnapshot

    @property
    def latest_world_artifact(self) -> ThinkingArtifact | None:
        return self.latest_artifacts_by_consumer.get(CONSUMER_WORLD_TEMPORAL)

    @property
    def latest_self_artifact(self) -> ThinkingArtifact | None:
        return self.latest_artifacts_by_consumer.get(CONSUMER_SELF_TEMPORAL)


@dataclass
class _PendingBatch:
    """Bookkeeping for a batch of tasks submitted after one turn.

    A batch groups tasks that came from the same turn so the next
    ``on_turn_begin`` can collect them all and populate
    ``latest_artifacts_by_consumer`` in one pass.
    """

    turn_index: int
    task_ids_by_consumer: dict[str, str] = field(default_factory=dict)


class ThinkingAdapter:
    """Per-session adapter that drives the thinking scheduler.

    Construct one per ``LifeformSession``. The session calls
    ``on_turn_begin`` / ``on_turn_end`` / ``drain`` at the
    appropriate lifecycle points. The adapter does the rest.

    **Read-only invariant:** the adapter never mutates kernel owner
    state. It only reads snapshots (via the ``snapshots`` parameter
    passed in from the session) and exposes artifacts for
    downstream consumers to apply. A contract test enforces the
    ``lifeform-core`` side of this boundary.
    """

    def __init__(
        self,
        *,
        scheduler: ThinkingScheduler | None = None,
        wiring_level: ThinkingWiringLevel = ThinkingWiringLevel.SHADOW,
        max_concurrent_tasks: int = 2,
        worker: WorkerFunc | None = None,
        scope: FingerprintScope | None = None,
    ) -> None:
        """Construct an adapter.

        * ``scheduler`` \u2014 inject a pre-built scheduler (useful for
          tests that want a fake clock). When None, the adapter
          builds its own with the supplied wiring_level +
          max_concurrent_tasks.
        * ``worker`` \u2014 worker callable to use for each task.
          Defaults to ``mid_reflection_worker``. Custom adapters can
          inject their own (e.g. a stub in tests).
        * ``scope`` \u2014 declared fingerprint scope. Defaults to
          ``MID_REFLECTION_SCOPE``. A custom scope must include at
          least the slots the injected worker reads.
        """
        if scheduler is not None and worker is None:
            # Scheduler supplied but no worker: that's a contract
            # bug because the adapter cannot dispatch tasks without
            # knowing which worker to run.
            raise ValueError(
                "ThinkingAdapter: when passing a scheduler you must "
                "also pass a worker; the scheduler is worker-agnostic "
                "but the adapter is not."
            )
        self._scheduler = scheduler or ThinkingScheduler(
            wiring_level=wiring_level,
            max_concurrent_tasks=max_concurrent_tasks,
        )
        self._worker: WorkerFunc = worker or mid_reflection_worker
        self._scope: FingerprintScope = scope or MID_REFLECTION_SCOPE
        self._latest_artifacts_by_consumer: dict[str, ThinkingArtifact] = {}
        self._pending_batch: _PendingBatch | None = None
        # One monotonic counter per adapter so unit tests can
        # predict task_ids deterministically.
        self._task_seq: int = 0

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def wiring_level(self) -> ThinkingWiringLevel:
        return self._scheduler.wiring_level

    @property
    def scheduler(self) -> ThinkingScheduler:
        """Return the underlying scheduler.

        Exposed so tests can call ``scheduler.snapshot()`` without
        going through the adapter. Product code should prefer
        ``adapter.snapshot()`` for the full adapter-scoped view.
        """
        return self._scheduler

    @property
    def latest_artifacts_by_consumer(self) -> Mapping[str, ThinkingArtifact]:
        """Return the most recent appliable artifacts, keyed by consumer owner.

        Only artifacts with status in ``APPLIABLE_THINKING_TASK_STATUSES``
        are present here; STALE / FAILED / CANCELLED ones are
        filtered out so downstream consumers can safely iterate
        without re-checking status.
        """
        return dict(self._latest_artifacts_by_consumer)

    @property
    def latest_world_artifact(self) -> ThinkingArtifact | None:
        return self._latest_artifacts_by_consumer.get(CONSUMER_WORLD_TEMPORAL)

    @property
    def latest_self_artifact(self) -> ThinkingArtifact | None:
        return self._latest_artifacts_by_consumer.get(CONSUMER_SELF_TEMPORAL)

    @property
    def latest_advisory_artifacts_by_consumer(self) -> Mapping[str, ThinkingArtifact]:
        """Latest appliable artifacts re-wrapped in the consumer contract.

        CP-21 (GAP-06): the temporal owners accept only
        ``ControllerPressureAdvisory`` payloads. The thinking side (this
        adapter + the worker module) owns the conversion, so consumers
        receive contract-ready artifacts and never rebuild advisory
        semantics from ``MidReflectionPayload`` internals. Artifacts whose
        payload is not a ``MidReflectionPayload`` are passed through
        unchanged (a custom worker may already emit the advisory payload).
        """

        import dataclasses

        advisory: dict[str, ThinkingArtifact] = {}
        for consumer, artifact in self._latest_artifacts_by_consumer.items():
            payload = artifact.payload
            if isinstance(payload, MidReflectionPayload):
                artifact = dataclasses.replace(
                    artifact,
                    payload=controller_pressure_advisory_from_mid_reflection(payload),
                )
            advisory[consumer] = artifact
        return advisory

    def snapshot(self) -> ThinkingAdapterSnapshot:
        """Build a frozen observation of current adapter state.

        Used by the session (``LifeformSession.thinking_adapter_snapshot``)
        and by contract tests. Safe to call any time; does not
        touch scheduler internals.
        """
        pending: tuple[str, ...] = ()
        if self._pending_batch is not None:
            pending = tuple(self._pending_batch.task_ids_by_consumer.values())
        return ThinkingAdapterSnapshot(
            wiring_level=self._scheduler.wiring_level,
            latest_artifacts_by_consumer=dict(self._latest_artifacts_by_consumer),
            pending_task_ids=pending,
            scheduler_snapshot=self._scheduler.snapshot(),
        )

    # ------------------------------------------------------------------
    # Session lifecycle hooks
    # ------------------------------------------------------------------

    async def on_turn_begin(
        self,
        *,
        snapshots: Mapping[str, Any],
        turn_index: int,
    ) -> None:
        """Collect artifacts for tasks submitted after the previous turn.

        Called BEFORE the kernel runs this turn. ``snapshots`` MUST be
        the snapshots from the PREVIOUS turn (i.e. current "latest"
        from the session's perspective). This is what the fingerprint
        was computed against at submit time, so COMPLETED artifacts
        will pass the guard; STALE artifacts (scope snapshots drifted)
        and CANCELLED / FAILED artifacts do NOT populate
        ``latest_artifacts_by_consumer``.

        Pre-first-turn (when there are no prior snapshots) is a
        no-op: there's no pending batch yet.
        """
        batch = self._pending_batch
        if batch is None:
            return
        self._pending_batch = None  # drain under any exception path
        if not snapshots:
            # No current snapshots to validate against; scheduler
            # will raise on missing slots. Turn every task to STALE
            # by cancelling \u2014 the payload is unsafe to apply.
            for task_id in batch.task_ids_by_consumer.values():
                await self._scheduler.cancel(task_id)
            return
        for consumer, task_id in batch.task_ids_by_consumer.items():
            try:
                artifact = await self._scheduler.collect(
                    task_id, current_snapshots=snapshots
                )
            except KeyError:
                # Task was never recorded (scheduler raised). Not a
                # caller bug we can recover from; log and continue
                # so other consumers still get their artifacts.
                _LOG.exception(
                    "ThinkingAdapter: collect failed for task %s (consumer %s)",
                    task_id,
                    consumer,
                )
                continue
            if artifact.status in APPLIABLE_THINKING_TASK_STATUSES:
                self._latest_artifacts_by_consumer[consumer] = artifact
            else:
                # Drop non-appliable: the consumer could not safely
                # use a STALE / FAILED / CANCELLED payload, and we
                # don't want stale data lingering in the "latest"
                # slot either.
                self._latest_artifacts_by_consumer.pop(consumer, None)

    async def on_turn_end(
        self,
        *,
        snapshots: Mapping[str, Any],
        turn_index: int,
    ) -> None:
        """Submit mid-reflection tasks for this turn.

        Called AFTER the kernel runs. ``snapshots`` MUST be the
        snapshots the kernel just produced (``result.active_snapshots``).
        If the declared scope slots are missing (e.g. a synthetic
        kernel skipping dual_track), submit is skipped silently \u2014
        this is a degradation signal, not a failure: the session
        continues but the thinking loop has no usable upstream.

        Two tasks are submitted, both using the same scope + worker
        but differing in ``purpose`` / ``consumer_owner``:

        * ``WORLD_LANE_REFLECT`` -> ``world_temporal`` consumer
        * ``SELF_LANE_REFLECT`` -> ``self_temporal`` consumer
        """
        # Missing-scope guard: fail-quiet (don't raise), log once per
        # missing slot, and leave pending_batch empty. The session
        # keeps running; observability shows no new tasks.
        missing = [
            slot for slot in self._scope.slot_names if slot not in snapshots
        ]
        if missing:
            _LOG.debug(
                "ThinkingAdapter: upstream snapshots missing %r; skipping submit",
                missing,
            )
            return
        batch = _PendingBatch(turn_index=turn_index)
        for purpose, consumer in (
            (ThinkingPurpose.WORLD_LANE_REFLECT, CONSUMER_WORLD_TEMPORAL),
            (ThinkingPurpose.SELF_LANE_REFLECT, CONSUMER_SELF_TEMPORAL),
        ):
            self._task_seq += 1
            task_id = (
                f"mid-reflect-{turn_index}-{purpose.value}-{self._task_seq}-"
                f"{uuid.uuid4().hex[:8]}"
            )
            # placeholder_fingerprint: the scheduler replaces this
            # with a freshly-computed one before dispatch (see
            # ``ThinkingScheduler.submit``). Non-empty to satisfy
            # ``ThinkingTask.__post_init__``.
            task = ThinkingTask(
                task_id=task_id,
                depth=ThinkingDepth.MID,
                purpose=purpose,
                requested_at_turn_index=turn_index,
                snapshot_fingerprint="pending-scheduler-stamp",
                consumer_owner=consumer,
                deadline_at_turn_index=None,
            )
            await self._scheduler.submit(
                task=task,
                worker=self._worker,
                scope=self._scope,
                upstream_snapshots=snapshots,
            )
            batch.task_ids_by_consumer[consumer] = task_id
        self._pending_batch = batch

    async def drain(self) -> tuple[ThinkingArtifact, ...]:
        """Await every in-flight task. Called at scene close.

        After drain:

        * No worker outlives the scene.
        * Latest-artifact slots are UPDATED with any newly-completed
          artifacts (subject to fingerprint guard when a pending
          batch exists).
        * Returns all terminal artifacts for observability.
        """
        artifacts = await self._scheduler.drain()
        if self._pending_batch is not None:
            # Drain does not run the fingerprint guard; we still
            # want to populate latest_artifacts for the ones that
            # completed without drift. Re-lookup by task_id and
            # copy into latest_artifacts_by_consumer.
            by_task_id = {a.task_id: a for a in artifacts}
            for consumer, task_id in self._pending_batch.task_ids_by_consumer.items():
                artifact = by_task_id.get(task_id)
                if artifact is None:
                    continue
                if artifact.status in APPLIABLE_THINKING_TASK_STATUSES:
                    self._latest_artifacts_by_consumer[consumer] = artifact
            self._pending_batch = None
        return artifacts


# ---------------------------------------------------------------------------
# Default factory
# ---------------------------------------------------------------------------


def build_default_thinking_adapter(
    *,
    wiring_level: ThinkingWiringLevel = ThinkingWiringLevel.SHADOW,
    max_concurrent_tasks: int = 2,
) -> ThinkingAdapter:
    """Convenience factory for the standard Gap 4 slice 2c wiring.

    Returns an adapter pre-configured with:

    * ``mid_reflection_worker`` as the worker (world + self lane)
    * ``MID_REFLECTION_SCOPE`` (dual_track + regime) as scope
    * Default scheduler with the supplied wiring_level +
      max_concurrent_tasks

    Callers that need custom worker / scope / scheduler build
    ``ThinkingAdapter`` directly.
    """
    return ThinkingAdapter(
        wiring_level=wiring_level,
        max_concurrent_tasks=max_concurrent_tasks,
    )


__all__ = [
    "CONSUMER_SELF_TEMPORAL",
    "CONSUMER_WORLD_TEMPORAL",
    "MID_REFLECTION_SCOPE",
    "ThinkingAdapter",
    "ThinkingAdapterSnapshot",
    "build_default_thinking_adapter",
]
