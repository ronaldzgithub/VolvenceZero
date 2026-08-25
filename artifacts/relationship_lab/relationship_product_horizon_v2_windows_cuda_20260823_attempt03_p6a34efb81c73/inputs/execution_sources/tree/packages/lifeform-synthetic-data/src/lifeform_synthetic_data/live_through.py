"""Run a trajectory through the public Lifeform facade and capture snapshots."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Protocol

from lifeform_core import Lifeform, LifeformConfig

from .canonical import canonical_json, stable_hash
from .contracts import (
    ExperienceSession,
    ExperienceTrajectory,
    GenerationTier,
    QualityRecord,
    QualitySeverity,
    SnapshotFrame,
    TurnRole,
)


class LifeformFactory(Protocol):
    def __call__(self) -> Lifeform: ...


def default_lifeform_factory() -> Lifeform:
    config = LifeformConfig()
    config = replace(
        config,
        brain_config=replace(config.brain_config, rare_heavy_enabled=False),
    )
    return Lifeform(config)


def live_through_trajectory(
    trajectory: ExperienceTrajectory,
    *,
    lifeform_factory: LifeformFactory = default_lifeform_factory,
) -> ExperienceTrajectory:
    return asyncio.run(
        live_through_trajectory_async(
            trajectory,
            lifeform_factory=lifeform_factory,
        )
    )


async def live_through_trajectory_async(
    trajectory: ExperienceTrajectory,
    *,
    lifeform_factory: LifeformFactory = default_lifeform_factory,
) -> ExperienceTrajectory:
    if trajectory.generation_tier is GenerationTier.LIVE_THROUGH:
        raise ValueError("trajectory is already live-through")
    lifeform = lifeform_factory()
    session = lifeform.create_session(session_id=f"synthetic-live::{trajectory.trajectory_id}")
    snapshot_frames: list[SnapshotFrame] = []
    output_sessions: list[ExperienceSession] = []
    pending_response: str | None = None

    for source_session in trajectory.sessions:
        if source_session.session_index > 0:
            await session.advance_tick(
                max(1, source_session.gap_days_before),
                reason="synthetic-inter-session-gap",
            )
        output_turns = []
        for turn in source_session.turns:
            if turn.role is TurnRole.USER:
                result = await session.run_turn(turn.text)
                turn_frames = _capture_result_snapshots(
                    trajectory_id=trajectory.trajectory_id,
                    turn_id=turn.turn_id,
                    active=result.active_snapshots,
                    shadow=result.shadow_snapshots,
                )
                snapshot_frames.extend(turn_frames)
                output_turns.append(
                    replace(
                        turn,
                        snapshot_refs=tuple(frame.snapshot_id for frame in turn_frames),
                    )
                )
                pending_response = result.response.text
            elif turn.role is TurnRole.ASSISTANT:
                if pending_response is None:
                    raise ValueError("live-through assistant turn has no preceding user result")
                output_turns.append(replace(turn, text=pending_response))
                pending_response = None
            else:
                output_turns.append(turn)
        if pending_response is not None:
            raise ValueError("live-through source session ends with an unpaired user turn")
        output_sessions.append(replace(source_session, turns=tuple(output_turns)))
        await session.end_scene(
            reason="synthetic-live-session-boundary",
            drain_slow_loop=True,
        )

    quality = trajectory.quality + (
        QualityRecord(
            quality_id=f"{trajectory.trajectory_id}:quality:live-through",
            check_kind="public_snapshot_lineage",
            passed=True,
            severity=QualitySeverity.INFO,
            score=1.0,
            evidence_refs=tuple(frame.snapshot_id for frame in snapshot_frames),
            description=(
                "Runtime observations were captured only from public turn results; "
                "no owner method or mutable internal state was accessed."
            ),
        ),
    )
    return replace(
        trajectory,
        generation_tier=GenerationTier.LIVE_THROUGH,
        sessions=tuple(output_sessions),
        snapshot_frames=tuple(snapshot_frames),
        quality=quality,
        provenance=replace(
            trajectory.provenance,
            source_kind=f"{trajectory.provenance.source_kind}_plus_live_through",
        ),
    )


def _capture_result_snapshots(
    *,
    trajectory_id: str,
    turn_id: str,
    active: Mapping[str, object],
    shadow: Mapping[str, object],
) -> tuple[SnapshotFrame, ...]:
    frames: list[SnapshotFrame] = []
    for wiring_level, snapshots in (("active", active), ("shadow", shadow)):
        for slot_name, raw_snapshot in sorted(snapshots.items()):
            snapshot = _require_snapshot_surface(raw_snapshot, slot_name=slot_name)
            payload_value = _runtime_primitive(snapshot.value)
            payload_json = json.dumps(
                payload_value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            value_hash = stable_hash(payload_value)
            snapshot_id = f"{trajectory_id}:runtime:{turn_id}:{wiring_level}:{slot_name}:{snapshot.version}"
            frames.append(
                SnapshotFrame(
                    snapshot_id=snapshot_id,
                    turn_ref=turn_id,
                    slot_name=slot_name,
                    owner=snapshot.owner,
                    version=snapshot.version,
                    timestamp_ms=snapshot.timestamp_ms,
                    value_type=(f"{type(snapshot.value).__module__}.{type(snapshot.value).__qualname__}"),
                    value_hash=value_hash,
                    payload_json=payload_json,
                    wiring_level=wiring_level,
                    description="",
                )
            )
    return tuple(frames)


class _SnapshotSurface(Protocol):
    slot_name: str
    owner: str
    version: int
    timestamp_ms: int
    value: object


def _require_snapshot_surface(
    value: object,
    *,
    slot_name: str,
) -> _SnapshotSurface:
    if not is_dataclass(value) or isinstance(value, type):
        raise TypeError(f"public snapshot {slot_name!r} must be a dataclass")
    field_names = {field.name for field in fields(value)}
    required = {"slot_name", "owner", "version", "timestamp_ms", "value"}
    if not required.issubset(field_names):
        raise TypeError(f"public snapshot {slot_name!r} lacks the canonical Snapshot fields")
    snapshot = value
    published_slot = object.__getattribute__(snapshot, "slot_name")
    owner = object.__getattribute__(snapshot, "owner")
    version = object.__getattribute__(snapshot, "version")
    timestamp_ms = object.__getattribute__(snapshot, "timestamp_ms")
    if published_slot != slot_name:
        raise ValueError(f"snapshot mapping key {slot_name!r} != published slot {published_slot!r}")
    if not isinstance(owner, str) or not owner:
        raise TypeError(f"snapshot {slot_name!r}.owner must be non-empty")
    if type(version) is not int or version < 0:
        raise TypeError(f"snapshot {slot_name!r}.version must be non-negative int")
    if type(timestamp_ms) is not int or timestamp_ms < 0:
        raise TypeError(f"snapshot {slot_name!r}.timestamp_ms must be non-negative int")
    return snapshot  # type: ignore[return-value]


def _runtime_primitive(value: object) -> object:
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _runtime_primitive(object.__getattribute__(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple | list):
        return [_runtime_primitive(item) for item in value]
    if isinstance(value, frozenset | set):
        resolved = [_runtime_primitive(item) for item in value]
        return sorted(resolved, key=canonical_json)
    if isinstance(value, Mapping):
        output: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("runtime snapshot mappings require string keys for canonical export")
            output[key] = _runtime_primitive(item)
        return output
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"unsupported public snapshot value type: {type(value).__module__}.{type(value).__qualname__}")


__all__ = [
    "LifeformFactory",
    "default_lifeform_factory",
    "live_through_trajectory",
    "live_through_trajectory_async",
]
