"""Contract tests for the Gap 4 thinking-loop envelopes.

These tests enforce the invariants spelled out in
``docs/specs/thinking-loop.md`` §Interface contracts. They are
data-only: no scheduler, no workers, no kernel wiring. Their purpose
is to keep the envelope shape stable so ``lifeform-thinking`` (the
scheduler wheel, built in a later slice) and any kernel owner that
consumes artifacts can rely on the same guarantees.

Enforced invariants:

1. **Enum completeness.** Every ``ThinkingTaskStatus`` /
   ``ThinkingDepth`` / ``ThinkingPurpose`` value is accounted for.
   Adding a new state MUST update this test so nothing silently
   branches on an un-observed status.
2. **Terminal / appliable sets.** Exactly one status
   (``COMPLETED``) is appliable; the other terminals
   (``STALE`` / ``CANCELLED`` / ``FAILED``) archive.
3. **Task fingerprint is required.** A task with empty
   ``snapshot_fingerprint`` raises — fingerprint guard is a hard
   invariant.
4. **Artifact error fields are FAILED-gated.** ``error_class`` and
   ``error_detail`` must be empty for any non-FAILED status, and
   ``error_class`` must be non-empty when status is FAILED.
5. **Deadline sanity.** ``deadline_at_turn_index`` < ``requested``
   is rejected (a deadline before the request is meaningless).
"""

from __future__ import annotations

import pytest

from volvence_zero.thinking import (
    APPLIABLE_THINKING_TASK_STATUSES,
    ControllerPressureAdvisory,
    TERMINAL_THINKING_TASK_STATUSES,
    ThinkingArtifact,
    ThinkingDepth,
    ThinkingPurpose,
    ThinkingTask,
    ThinkingTaskStatus,
)


# ---------------------------------------------------------------------------
# Enum completeness
# ---------------------------------------------------------------------------


def test_thinking_task_status_values_are_exhaustive() -> None:
    known = {
        ThinkingTaskStatus.QUEUED,
        ThinkingTaskStatus.RUNNING,
        ThinkingTaskStatus.COMPLETED,
        ThinkingTaskStatus.STALE,
        ThinkingTaskStatus.CANCELLED,
        ThinkingTaskStatus.FAILED,
    }
    assert set(ThinkingTaskStatus) == known, (
        "ThinkingTaskStatus set changed; update TERMINAL_THINKING_TASK_STATUSES "
        "and APPLIABLE_THINKING_TASK_STATUSES, then update this test."
    )


def test_thinking_depth_values_are_exhaustive() -> None:
    assert set(ThinkingDepth) == {
        ThinkingDepth.FAST,
        ThinkingDepth.MID,
        ThinkingDepth.SLOW,
    }


def test_thinking_purpose_values_are_exhaustive() -> None:
    assert set(ThinkingPurpose) == {
        ThinkingPurpose.WORLD_LANE_REFLECT,
        ThinkingPurpose.SELF_LANE_REFLECT,
        ThinkingPurpose.EXPLORATION,
        ThinkingPurpose.PROVISIONAL_RECONCILE,
    }


# ---------------------------------------------------------------------------
# Terminal / appliable sets
# ---------------------------------------------------------------------------


def test_terminal_status_set_is_exactly_completed_stale_cancelled_failed() -> None:
    assert TERMINAL_THINKING_TASK_STATUSES == frozenset(
        {
            ThinkingTaskStatus.COMPLETED,
            ThinkingTaskStatus.STALE,
            ThinkingTaskStatus.CANCELLED,
            ThinkingTaskStatus.FAILED,
        }
    )


def test_appliable_status_set_is_exactly_completed() -> None:
    # Exactly COMPLETED. STALE / CANCELLED / FAILED terminate the task
    # but their payloads must never be applied by a consumer.
    assert APPLIABLE_THINKING_TASK_STATUSES == frozenset(
        {ThinkingTaskStatus.COMPLETED}
    )


def test_appliable_is_subset_of_terminal() -> None:
    assert APPLIABLE_THINKING_TASK_STATUSES <= TERMINAL_THINKING_TASK_STATUSES


# ---------------------------------------------------------------------------
# ThinkingTask invariants
# ---------------------------------------------------------------------------


def test_task_accepts_valid_construction() -> None:
    task = ThinkingTask(
        task_id="t-1",
        depth=ThinkingDepth.MID,
        purpose=ThinkingPurpose.WORLD_LANE_REFLECT,
        requested_at_turn_index=3,
        snapshot_fingerprint="sha256:abc",
        consumer_owner="world_temporal",
        deadline_at_turn_index=5,
    )
    assert task.task_id == "t-1"
    assert task.deadline_at_turn_index == 5


def test_task_rejects_empty_task_id() -> None:
    with pytest.raises(ValueError, match="task_id"):
        ThinkingTask(
            task_id="",
            depth=ThinkingDepth.MID,
            purpose=ThinkingPurpose.WORLD_LANE_REFLECT,
            requested_at_turn_index=0,
            snapshot_fingerprint="sha256:abc",
            consumer_owner="world_temporal",
        )


def test_task_rejects_empty_fingerprint() -> None:
    with pytest.raises(ValueError, match="snapshot_fingerprint"):
        ThinkingTask(
            task_id="t-2",
            depth=ThinkingDepth.MID,
            purpose=ThinkingPurpose.WORLD_LANE_REFLECT,
            requested_at_turn_index=0,
            snapshot_fingerprint="",
            consumer_owner="world_temporal",
        )


def test_task_rejects_empty_consumer_owner() -> None:
    with pytest.raises(ValueError, match="consumer_owner"):
        ThinkingTask(
            task_id="t-3",
            depth=ThinkingDepth.MID,
            purpose=ThinkingPurpose.EXPLORATION,
            requested_at_turn_index=0,
            snapshot_fingerprint="sha256:abc",
            consumer_owner="",
        )


def test_task_rejects_deadline_before_requested() -> None:
    with pytest.raises(ValueError, match="deadline"):
        ThinkingTask(
            task_id="t-4",
            depth=ThinkingDepth.MID,
            purpose=ThinkingPurpose.WORLD_LANE_REFLECT,
            requested_at_turn_index=5,
            snapshot_fingerprint="sha256:abc",
            consumer_owner="world_temporal",
            deadline_at_turn_index=3,
        )


def test_task_accepts_none_deadline() -> None:
    task = ThinkingTask(
        task_id="t-5",
        depth=ThinkingDepth.MID,
        purpose=ThinkingPurpose.WORLD_LANE_REFLECT,
        requested_at_turn_index=0,
        snapshot_fingerprint="sha256:abc",
        consumer_owner="world_temporal",
        deadline_at_turn_index=None,
    )
    assert task.deadline_at_turn_index is None


# ---------------------------------------------------------------------------
# ThinkingArtifact invariants
# ---------------------------------------------------------------------------


def test_artifact_completed_requires_empty_error_fields() -> None:
    with pytest.raises(ValueError, match="error_class/error_detail"):
        ThinkingArtifact(
            task_id="t-1",
            status=ThinkingTaskStatus.COMPLETED,
            payload={"foo": 1},
            produced_at_turn_index=3,
            consumer_owner="world_temporal",
            error_class="ShouldBeEmpty",
        )


def test_artifact_failed_requires_error_class() -> None:
    with pytest.raises(ValueError, match="error_class"):
        ThinkingArtifact(
            task_id="t-1",
            status=ThinkingTaskStatus.FAILED,
            payload=None,
            produced_at_turn_index=3,
            consumer_owner="world_temporal",
            error_class="",
        )


def test_artifact_failed_accepts_error_class_and_detail() -> None:
    artifact = ThinkingArtifact(
        task_id="t-1",
        status=ThinkingTaskStatus.FAILED,
        payload=None,
        produced_at_turn_index=3,
        consumer_owner="world_temporal",
        error_class="TimeoutError",
        error_detail="scene-scoped timeout, 12s",
    )
    assert artifact.error_class == "TimeoutError"
    assert not artifact.is_appliable()


def test_artifact_stale_is_not_appliable() -> None:
    artifact = ThinkingArtifact(
        task_id="t-1",
        status=ThinkingTaskStatus.STALE,
        payload={"ignored": True},
        produced_at_turn_index=3,
        consumer_owner="world_temporal",
    )
    assert not artifact.is_appliable()


def test_artifact_completed_is_appliable() -> None:
    artifact = ThinkingArtifact(
        task_id="t-1",
        status=ThinkingTaskStatus.COMPLETED,
        payload={"track": "world", "pressure": 0.2},
        produced_at_turn_index=3,
        consumer_owner="world_temporal",
    )
    assert artifact.is_appliable()


def test_controller_pressure_advisory_validates_bounds_and_evidence() -> None:
    advisory = ControllerPressureAdvisory(
        track="world",
        pressure_delta=0.4,
        confidence=0.8,
        evidence=("pe-history-compressed",),
        description="bounded pressure",
    )
    assert advisory.track == "world"
    with pytest.raises(ValueError, match="track"):
        ControllerPressureAdvisory(
            track="shared",
            pressure_delta=0.0,
            confidence=0.5,
            evidence=("x",),
        )
    with pytest.raises(ValueError, match="pressure_delta"):
        ControllerPressureAdvisory(
            track="world",
            pressure_delta=1.4,
            confidence=0.5,
            evidence=("x",),
        )
    with pytest.raises(ValueError, match="evidence"):
        ControllerPressureAdvisory(
            track="self",
            pressure_delta=0.2,
            confidence=0.5,
            evidence=(),
        )


def test_artifact_rejects_negative_produced_at_turn_index() -> None:
    with pytest.raises(ValueError, match="produced_at_turn_index"):
        ThinkingArtifact(
            task_id="t-1",
            status=ThinkingTaskStatus.COMPLETED,
            payload={},
            produced_at_turn_index=-1,
            consumer_owner="world_temporal",
        )
