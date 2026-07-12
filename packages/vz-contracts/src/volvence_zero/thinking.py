"""Thinking-task lifecycle envelopes (Gap 4 foundation).

Cross-wheel immutable contracts for the mid-session / async ``thinking``
loop path described in ``docs/specs/thinking-loop.md``. These envelopes
are *data only* — no scheduler, no workers, no owner state. The
``lifeform-thinking`` wheel (built in a follow-up slice) owns the
``ThinkingScheduler`` that produces and consumes them; kernel-side
owners (case_memory, world_temporal, self_temporal, boundary_consent)
each decide whether to ``apply`` an artifact based on their own
fingerprint guard.

Design rules baked into these types:

* Everything is a ``@dataclass(frozen=True)`` with stdlib types only.
  No torch, no enums that aren't string enums, no runtime state.
* ``ThinkingArtifact.status`` must be checked BEFORE looking at
  ``payload``. A ``STALE`` / ``FAILED`` / ``CANCELLED`` artifact has no
  guarantees about payload shape; owners must not apply it.
* ``snapshot_fingerprint`` is owner-scoped. The ``lifeform-thinking``
  wheel computes it from a *declared* set of upstream slots; consumers
  compare it against the current fingerprint *in the same owner's
  scope* before applying. Never trust a stale fingerprint.
* ``produced_at_turn_index`` vs ``requested_at_turn_index`` lets the
  consumer see how long a task was in flight. If the delta exceeds
  ``deadline_at_turn_index - requested_at_turn_index``, the scheduler
  should have flipped status to ``STALE`` already.

Why these live in ``vz-contracts`` (not ``lifeform-thinking``): the
artifacts flow THROUGH kernel owners. If every owner that consumed an
artifact had to import ``lifeform-thinking``, the kernel would
depend on a lifeform wheel \u2014 a reversal of the ``vz-* \u219b lifeform-*``
invariant (see ``SPLIT.md`` and ``tests/contracts/test_import_boundaries.py``).
Putting the envelopes in ``vz-contracts`` keeps the scheduler
lifeform-side while still letting kernel owners consume its outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ThinkingDepth(str, Enum):
    """Time-scale bucket a ``ThinkingTask`` operates in.

    ``FAST`` is listed for completeness \u2014 per-turn synchronous thinking
    is *not* scheduled by ``ThinkingScheduler`` (the kernel already
    handles it inline). ``MID`` is the Gap 4 contribution: scene-scoped
    asynchronous thinking that lives between the per-turn path and the
    session-post slow loop. ``SLOW`` is the existing R6 session-post
    loop, also listed for completeness.
    """

    FAST = "fast"
    MID = "mid"
    SLOW = "slow"


class ThinkingTaskStatus(str, Enum):
    """Lifecycle states of a ``ThinkingTask``.

    State transitions are linear: ``QUEUED \u2192 RUNNING \u2192 (COMPLETED |
    STALE | CANCELLED | FAILED)``. Once a terminal state is set, the
    artifact is immutable and consumers must not retry. ``STALE`` is
    the fingerprint-mismatch terminal state \u2014 a task's upstream
    snapshots changed while it was running so its output can no longer
    be safely applied.
    """

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    STALE = "stale"
    CANCELLED = "cancelled"
    FAILED = "failed"


TERMINAL_THINKING_TASK_STATUSES: frozenset[ThinkingTaskStatus] = frozenset(
    {
        ThinkingTaskStatus.COMPLETED,
        ThinkingTaskStatus.STALE,
        ThinkingTaskStatus.CANCELLED,
        ThinkingTaskStatus.FAILED,
    }
)


APPLIABLE_THINKING_TASK_STATUSES: frozenset[ThinkingTaskStatus] = frozenset(
    {ThinkingTaskStatus.COMPLETED}
)


class ThinkingPurpose(str, Enum):
    """What the task is doing. Determines which worker runs it and which
    owner is expected to consume the resulting artifact.

    Purposes are *coarse enough* to survive growing the worker set: a
    new kind of mid-reflection (e.g. a ``CASE_MEMORY_REFLECT``) should
    be expressible without adding a purpose if possible. We deliberately
    do NOT list one purpose per future worker; keep this enum narrow.
    """

    WORLD_LANE_REFLECT = "world_lane_reflect"
    SELF_LANE_REFLECT = "self_lane_reflect"
    EXPLORATION = "exploration"
    PROVISIONAL_RECONCILE = "provisional_reconcile"


@dataclass(frozen=True)
class ThinkingTask:
    """Immutable envelope describing a task the scheduler has queued.

    ``snapshot_fingerprint`` is a SHA256 / equivalent stable digest of
    the upstream snapshot set the task started from. Consumers
    recompute a fingerprint in the same scope at apply-time and
    reject any mismatch (\u2192 ``STALE``). The fingerprint MUST be a
    plain string so it round-trips through JSON logging / debug
    dumps.

    ``consumer_owner`` identifies the owner module that will apply the
    eventual artifact. Required for dispatch \u2014 the scheduler cannot
    guess which kernel owner a worker writes to.
    """

    task_id: str
    depth: ThinkingDepth
    purpose: ThinkingPurpose
    requested_at_turn_index: int
    snapshot_fingerprint: str
    consumer_owner: str
    deadline_at_turn_index: int | None = None

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("ThinkingTask.task_id must be non-empty")
        if not self.snapshot_fingerprint:
            raise ValueError(
                "ThinkingTask.snapshot_fingerprint must be non-empty; "
                "fingerprint guard is required for staleness detection."
            )
        if self.requested_at_turn_index < 0:
            raise ValueError(
                f"ThinkingTask.requested_at_turn_index must be >= 0, "
                f"got {self.requested_at_turn_index!r}"
            )
        if (
            self.deadline_at_turn_index is not None
            and self.deadline_at_turn_index < self.requested_at_turn_index
        ):
            raise ValueError(
                f"ThinkingTask.deadline_at_turn_index "
                f"({self.deadline_at_turn_index}) must be >= "
                f"requested_at_turn_index ({self.requested_at_turn_index})"
            )
        if not self.consumer_owner:
            raise ValueError("ThinkingTask.consumer_owner must be non-empty")


@dataclass(frozen=True)
class ThinkingArtifact:
    """Immutable envelope produced by a worker.

    Apply protocol (owner-side):

    1. Read ``status``. If not in ``APPLIABLE_THINKING_TASK_STATUSES``,
       archive the artifact and do NOT touch ``payload``.
    2. Recompute the fingerprint in the consumer's current scope and
       compare against the ``ThinkingTask.snapshot_fingerprint``. On
       mismatch: treat as ``STALE`` (even if the worker happened to
       deliver ``COMPLETED``) and archive.
    3. If status is appliable AND fingerprint matches, decode
       ``payload`` according to the consumer's declared payload
       schema. Payload shape is consumer-defined; this envelope does
       not constrain it beyond ``Any`` (the consumer's contract test
       enforces the shape).

    ``error_class`` / ``error_detail`` are non-empty only for
    ``FAILED`` status. Contract tests enforce this.
    """

    task_id: str
    status: ThinkingTaskStatus
    payload: Any
    produced_at_turn_index: int
    consumer_owner: str
    error_class: str = ""
    error_detail: str = ""

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("ThinkingArtifact.task_id must be non-empty")
        if not self.consumer_owner:
            raise ValueError("ThinkingArtifact.consumer_owner must be non-empty")
        if self.produced_at_turn_index < 0:
            raise ValueError(
                f"ThinkingArtifact.produced_at_turn_index must be >= 0, "
                f"got {self.produced_at_turn_index!r}"
            )
        if self.status is ThinkingTaskStatus.FAILED:
            if not self.error_class.strip():
                raise ValueError(
                    "ThinkingArtifact.error_class must be non-empty when "
                    "status is FAILED (fail-loud audit invariant)."
                )
        else:
            if self.error_class or self.error_detail:
                raise ValueError(
                    f"ThinkingArtifact.error_class/error_detail must be "
                    f"empty unless status is FAILED; status={self.status.value!r}"
                )

    def is_appliable(self) -> bool:
        """Return True iff a consumer may inspect ``payload`` safely."""
        return self.status in APPLIABLE_THINKING_TASK_STATUSES


@dataclass(frozen=True)
class ControllerPressureAdvisory:
    """Compact controller-pressure payload for temporal owners (CP-21).

    Produced by read-only thinking workers, consumed only by the intended
    temporal owner after fingerprint validation. ``pressure_delta`` is
    bounded to [-1, 1]; the temporal owner decides whether to apply it.
    """

    track: str
    pressure_delta: float
    confidence: float
    evidence: tuple[str, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if self.track not in {"world", "self"}:
            raise ValueError(f"track must be 'world' or 'self', got {self.track!r}")
        if self.pressure_delta < -1.0 or self.pressure_delta > 1.0:
            raise ValueError("pressure_delta must be in [-1, 1]")
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not self.evidence:
            raise ValueError("evidence must be non-empty")
        for item in self.evidence:
            if not item.strip():
                raise ValueError("evidence entries must be non-empty")
        if self.description is not None and not isinstance(self.description, str):
            raise TypeError("description must be a string")


__all__ = [
    "APPLIABLE_THINKING_TASK_STATUSES",
    "TERMINAL_THINKING_TASK_STATUSES",
    "ControllerPressureAdvisory",
    "ThinkingArtifact",
    "ThinkingDepth",
    "ThinkingPurpose",
    "ThinkingTask",
    "ThinkingTaskStatus",
]
