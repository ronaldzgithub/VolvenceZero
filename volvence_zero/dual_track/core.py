from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from volvence_zero.memory import MemoryEntry, MemorySnapshot, Track
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel


@dataclass(frozen=True)
class TrackState:
    track: Track
    active_goals: tuple[str, ...]
    recent_credits: tuple[tuple[str, float], ...]
    controller_code: tuple[float, ...]
    tension_level: float


@dataclass(frozen=True)
class DualTrackSnapshot:
    world_track: TrackState
    self_track: TrackState
    cross_track_tension: float
    description: str


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _goal_from_entry(entry: MemoryEntry) -> str:
    return entry.content


def _controller_code(entries: tuple[MemoryEntry, ...]) -> tuple[float, ...]:
    if not entries:
        return (0.0, 0.0)
    average_strength = sum(entry.strength for entry in entries) / len(entries)
    average_recency = sum(entry.last_accessed_ms - entry.created_at_ms for entry in entries) / len(entries)
    normalized_recency = 1.0 if average_recency > 0 else 0.0
    return (_clamp(average_strength), _clamp(normalized_recency))


def _tension(entries: tuple[MemoryEntry, ...]) -> float:
    if not entries:
        return 0.0
    average_strength = sum(entry.strength for entry in entries) / len(entries)
    return _clamp(average_strength)


def entries_by_track(memory_snapshot: MemorySnapshot, track: Track) -> tuple[MemoryEntry, ...]:
    return tuple(entry for entry in memory_snapshot.retrieved_entries if entry.track is track)


def derive_track_state(
    *,
    track: Track,
    memory_entries: tuple[MemoryEntry, ...],
) -> TrackState:
    goals = tuple(_goal_from_entry(entry) for entry in memory_entries[:3])
    recent_credits = tuple(
        (entry.entry_id, round(entry.strength, 3)) for entry in memory_entries[:3]
    )
    return TrackState(
        track=track,
        active_goals=goals,
        recent_credits=recent_credits,
        controller_code=_controller_code(memory_entries),
        tension_level=_tension(memory_entries),
    )


def derive_cross_track_tension(world_track: TrackState, self_track: TrackState) -> float:
    goal_overlap = len(set(world_track.active_goals) & set(self_track.active_goals))
    overlap_penalty = 0.15 if goal_overlap else 0.0
    return _clamp(abs(world_track.tension_level - self_track.tension_level) + overlap_penalty)


class DualTrackModule(RuntimeModule[DualTrackSnapshot]):
    slot_name = "dual_track"
    owner = "DualTrackModule"
    value_type = DualTrackSnapshot
    dependencies = ("memory",)
    default_wiring_level = WiringLevel.SHADOW

    async def process(self, upstream: Mapping[str, Snapshot[object]]) -> Snapshot[DualTrackSnapshot]:
        memory_snapshot = upstream["memory"]
        memory_value = memory_snapshot.value
        if not isinstance(memory_value, MemorySnapshot):
            world_track = derive_track_state(track=Track.WORLD, memory_entries=())
            self_track = derive_track_state(track=Track.SELF, memory_entries=())
        else:
            world_entries = entries_by_track(memory_value, Track.WORLD)
            self_entries = entries_by_track(memory_value, Track.SELF)
            world_track = derive_track_state(track=Track.WORLD, memory_entries=world_entries)
            self_track = derive_track_state(track=Track.SELF, memory_entries=self_entries)

        cross_track_tension = derive_cross_track_tension(world_track, self_track)
        description = (
            f"Dual-track state with world_tension={world_track.tension_level:.2f}, "
            f"self_tension={self_track.tension_level:.2f}, "
            f"cross_track_tension={cross_track_tension:.2f}."
        )
        return self.publish(
            DualTrackSnapshot(
                world_track=world_track,
                self_track=self_track,
                cross_track_tension=cross_track_tension,
                description=description,
            )
        )

    async def process_standalone(self, **kwargs: object) -> Snapshot[DualTrackSnapshot]:
        world_entries = kwargs.get("world_entries")
        self_entries = kwargs.get("self_entries")
        if world_entries is None:
            world_entries = ()
        if self_entries is None:
            self_entries = ()
        if not isinstance(world_entries, tuple):
            raise TypeError("world_entries must be a tuple when provided.")
        if not isinstance(self_entries, tuple):
            raise TypeError("self_entries must be a tuple when provided.")

        world_track = derive_track_state(track=Track.WORLD, memory_entries=world_entries)
        self_track = derive_track_state(track=Track.SELF, memory_entries=self_entries)
        cross_track_tension = derive_cross_track_tension(world_track, self_track)
        return self.publish(
            DualTrackSnapshot(
                world_track=world_track,
                self_track=self_track,
                cross_track_tension=cross_track_tension,
                description="Standalone dual-track snapshot.",
            )
        )
