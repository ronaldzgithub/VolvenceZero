"""Unit tests for ``mid_reflection_worker`` (Gap 4 slice 2b).

The worker is read-only and purely functional over its upstream
mapping, so we feed it synthetic snapshot stubs rather than driving
a full LifeformSession. That keeps the test fast AND validates that
the worker does not depend on hidden global state.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from lifeform_thinking import (
    MidReflectionPayload,
    ThinkingArtifact,
    ThinkingDepth,
    ThinkingPurpose,
    ThinkingTask,
    ThinkingTaskStatus,
    mid_reflection_worker,
)


@dataclass(frozen=True)
class _TrackStub:
    tension_level: float


@dataclass(frozen=True)
class _DualTrackStub:
    world_track: _TrackStub
    self_track: _TrackStub
    cross_track_tension: float


@dataclass(frozen=True)
class _RegimeIdentityStub:
    regime_id: str


@dataclass(frozen=True)
class _RegimeStub:
    active_regime: _RegimeIdentityStub


@dataclass(frozen=True)
class _SnapStub:
    slot_name: str
    version: int
    value: object


def _make_upstream(
    *,
    cross_tension: float = 0.4,
    world_tension: float = 0.3,
    self_tension: float = 0.5,
    regime_id: str = "emotional_support",
) -> dict[str, _SnapStub]:
    return {
        "dual_track": _SnapStub(
            slot_name="dual_track",
            version=1,
            value=_DualTrackStub(
                world_track=_TrackStub(tension_level=world_tension),
                self_track=_TrackStub(tension_level=self_tension),
                cross_track_tension=cross_tension,
            ),
        ),
        "regime": _SnapStub(
            slot_name="regime",
            version=1,
            value=_RegimeStub(
                active_regime=_RegimeIdentityStub(regime_id=regime_id),
            ),
        ),
    }


def _task(
    *,
    purpose: ThinkingPurpose,
    task_id: str = "t-mid-1",
    consumer_owner: str = "world_temporal",
) -> ThinkingTask:
    return ThinkingTask(
        task_id=task_id,
        depth=ThinkingDepth.MID,
        purpose=purpose,
        requested_at_turn_index=2,
        snapshot_fingerprint="sha256:stub",
        consumer_owner=consumer_owner,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_world_lane_reflect_produces_typed_payload() -> None:
    upstream = _make_upstream()
    artifact = await mid_reflection_worker(
        _task(purpose=ThinkingPurpose.WORLD_LANE_REFLECT),
        upstream,
    )
    assert artifact.status is ThinkingTaskStatus.COMPLETED
    assert artifact.is_appliable()
    payload = artifact.payload
    assert isinstance(payload, MidReflectionPayload)
    assert payload.track == "world"
    assert payload.observed_regime_id == "emotional_support"
    # Pressure is clamped to [-1, 1]
    assert -1.0 <= payload.pressure <= 1.0


async def test_self_lane_reflect_targets_self_track() -> None:
    upstream = _make_upstream(world_tension=0.1, self_tension=0.9)
    artifact = await mid_reflection_worker(
        _task(
            purpose=ThinkingPurpose.SELF_LANE_REFLECT,
            consumer_owner="self_temporal",
        ),
        upstream,
    )
    payload = artifact.payload
    assert isinstance(payload, MidReflectionPayload)
    assert payload.track == "self"
    # Self-track tension is high, so the pressure should lean negative
    # (the worker infers "this track is dragging") versus a calmer
    # world-lane reflection on the same upstream.
    alt_upstream = _make_upstream(world_tension=0.9, self_tension=0.1)
    alt_artifact = await mid_reflection_worker(
        _task(
            purpose=ThinkingPurpose.SELF_LANE_REFLECT,
            consumer_owner="self_temporal",
            task_id="t-mid-alt",
        ),
        alt_upstream,
    )
    alt_payload = alt_artifact.payload
    assert isinstance(alt_payload, MidReflectionPayload)
    assert alt_payload.pressure > payload.pressure


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


async def test_worker_rejects_non_reflection_purpose() -> None:
    upstream = _make_upstream()
    with pytest.raises(ValueError, match="non-reflection purpose"):
        await mid_reflection_worker(
            _task(purpose=ThinkingPurpose.EXPLORATION),
            upstream,
        )


async def test_worker_rejects_missing_upstream_slots() -> None:
    with pytest.raises(ValueError, match="requires dual_track and regime"):
        await mid_reflection_worker(
            _task(purpose=ThinkingPurpose.WORLD_LANE_REFLECT),
            upstream={},
        )


async def test_worker_uses_safe_defaults_for_missing_track_fields() -> None:
    """If the dual_track stub lacks ``world_track`` / ``self_track``
    the worker returns a neutral pressure rather than crashing.
    """

    @dataclass(frozen=True)
    class _PartialDualTrack:
        cross_track_tension: float = 0.5
        # no world_track / self_track at all

    upstream = {
        "dual_track": _SnapStub(
            slot_name="dual_track",
            version=1,
            value=_PartialDualTrack(cross_track_tension=0.5),
        ),
        "regime": _SnapStub(
            slot_name="regime",
            version=1,
            value=_RegimeStub(active_regime=_RegimeIdentityStub("emotional_support")),
        ),
    }
    artifact = await mid_reflection_worker(
        _task(purpose=ThinkingPurpose.WORLD_LANE_REFLECT),
        upstream,
    )
    assert isinstance(artifact.payload, MidReflectionPayload)
    # Neutral inputs (cross=0.5, track defaulted to 0.5) should
    # produce a near-zero pressure.
    assert abs(artifact.payload.pressure) <= 0.05
