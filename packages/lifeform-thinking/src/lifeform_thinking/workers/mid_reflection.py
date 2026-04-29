"""Mid-reflection worker \u2014 scene-scoped self/world reflection.

Reads ``world_temporal`` / ``self_temporal`` / ``memory`` / ``regime``
snapshots (all taken at submit-time and passed in read-only) and
produces a ``MidReflectionPayload`` describing a scene-scoped
pressure estimate that the corresponding track-temporal owner MAY
consume as an advisory on the next turn.

Design stance:

* **Read-only.** The worker only inspects snapshot fields via public
  attributes. No store handles, no mutation calls. Violating this
  breaks the worker-cannot-mutate-owners invariant enforced by
  contract test.
* **Bounded output.** The payload is a frozen dataclass with a
  narrow shape so future refactors (e.g. richer reflection signals)
  can grow via new dataclasses rather than dict-of-arbitrary.
* **Single-purpose.** One worker -> one purpose
  (``SELF_LANE_REFLECT`` / ``WORLD_LANE_REFLECT``). The caller
  supplies the track through the task's purpose; this worker does
  not branch on behavior other than choosing which upstream track
  snapshot to look at.

Slice 2b MVP: the payload carries a scalar ``pressure`` derived from
cross-track tension + track confidence. Future slices will enrich
with advisory hints (e.g. suggested provisional cases). The slice-2b
goal is to prove the envelope + pipeline work, not to ship a
final-form reflection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from volvence_zero.thinking import (
    ThinkingArtifact,
    ThinkingPurpose,
    ThinkingTask,
    ThinkingTaskStatus,
)


@dataclass(frozen=True)
class MidReflectionPayload:
    """Immutable payload shape for mid-reflection artifacts.

    ``track`` is a plain string (``"world"`` or ``"self"``) rather
    than an enum so the payload can round-trip through JSON
    logging. ``pressure`` is signed: positive = "relax" advisory
    (things are going well), negative = "tighten" advisory (tension
    rising). Clamped to ``[-1, 1]`` by the worker so downstream
    consumers get a stable envelope.

    ``rationale`` is a short human-readable description for audit /
    family-report rendering; it is NOT parsed by downstream
    consumers and must never drive policy by keyword match (red
    line A).
    """

    track: str
    pressure: float
    rationale: str
    observed_cross_track_tension: float
    observed_regime_id: str


def _clamp_signed(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _extract_track_tension(snapshot_value: Any, track: str) -> float:
    """Pull the single-track tension from a ``DualTrackSnapshot``-shaped value.

    Works with both the real kernel snapshot and the synthetic test
    stubs; uses ``getattr`` with a safe default so missing fields
    don't crash the worker (they become a neutral-pressure
    reflection).
    """
    if track == "world":
        track_state = getattr(snapshot_value, "world_track", None)
    else:
        track_state = getattr(snapshot_value, "self_track", None)
    if track_state is None:
        return 0.5
    return float(getattr(track_state, "tension_level", 0.5))


async def mid_reflection_worker(
    task: ThinkingTask,
    upstream: Mapping[str, Any],
) -> ThinkingArtifact:
    """Scheduler-invoked worker.

    Expects the task's fingerprint scope to include at least
    ``dual_track`` and ``regime``. Returns a ``COMPLETED`` artifact
    with a ``MidReflectionPayload`` on success; raises on
    misconfiguration so the scheduler can wrap into ``FAILED``.
    """
    if task.purpose not in {
        ThinkingPurpose.SELF_LANE_REFLECT,
        ThinkingPurpose.WORLD_LANE_REFLECT,
    }:
        raise ValueError(
            f"mid_reflection_worker invoked with non-reflection purpose "
            f"{task.purpose.value!r}"
        )
    track = (
        "world"
        if task.purpose is ThinkingPurpose.WORLD_LANE_REFLECT
        else "self"
    )
    dual_track_snapshot = upstream.get("dual_track")
    regime_snapshot = upstream.get("regime")
    if dual_track_snapshot is None or regime_snapshot is None:
        raise ValueError(
            "mid_reflection_worker requires dual_track and regime snapshots "
            "in the fingerprint scope; upstream is missing one of them."
        )
    dual_track_value = getattr(dual_track_snapshot, "value", dual_track_snapshot)
    regime_value = getattr(regime_snapshot, "value", regime_snapshot)
    cross_track_tension = float(
        getattr(dual_track_value, "cross_track_tension", 0.5)
    )
    track_tension = _extract_track_tension(dual_track_value, track)
    regime_identity = getattr(regime_value, "active_regime", None)
    regime_id = str(getattr(regime_identity, "regime_id", "unknown"))
    # Scene-scoped pressure: positive when cross-track tension is
    # low AND track-specific tension is low; negative when the
    # combination indicates the track is dragging. The exact
    # formula is deliberately simple \u2014 slice 2b is about the
    # envelope, not the reflection intelligence.
    relief = (1.0 - cross_track_tension) + (1.0 - track_tension)
    pressure = _clamp_signed((relief / 2.0) * 2.0 - 1.0)
    rationale = (
        f"mid-reflection[{track}] cross_track_tension={cross_track_tension:.2f} "
        f"track_tension={track_tension:.2f} regime={regime_id} "
        f"pressure={pressure:+.2f}"
    )
    payload = MidReflectionPayload(
        track=track,
        pressure=pressure,
        rationale=rationale,
        observed_cross_track_tension=cross_track_tension,
        observed_regime_id=regime_id,
    )
    return ThinkingArtifact(
        task_id=task.task_id,
        status=ThinkingTaskStatus.COMPLETED,
        payload=payload,
        produced_at_turn_index=task.requested_at_turn_index,
        consumer_owner=task.consumer_owner,
    )


__all__ = [
    "MidReflectionPayload",
    "mid_reflection_worker",
]
