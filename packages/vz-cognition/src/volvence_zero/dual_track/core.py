from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from volvence_zero.memory import MemoryEntry, MemorySnapshot, Track
from volvence_zero.runtime import RuntimeModule, RuntimePlaceholderValue, Snapshot, WiringLevel
from volvence_zero.semantic_state import (
    SELF_SEMANTIC_OWNER_SLOTS,
    SEMANTIC_OWNER_SLOTS,
    WORLD_SEMANTIC_OWNER_SLOTS,
)
from volvence_zero.substrate import SubstrateSnapshot, feature_signal_value


@dataclass(frozen=True)
class TrackState:
    track: Track
    active_goals: tuple[str, ...]
    recent_credits: tuple[tuple[str, float], ...]
    controller_code: tuple[float, ...]
    tension_level: float
    abstract_action_hint: str | None = None
    action_family_version_hint: int = 0
    controller_source: str = "memory"


@dataclass(frozen=True)
class DualTrackSnapshot:
    world_track: TrackState
    self_track: TrackState
    cross_track_tension: float
    description: str
    memory_retrieval_facets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.memory_retrieval_facets:
            return
        object.__setattr__(
            self,
            "memory_retrieval_facets",
            (
                *(f"world-goal:{goal}" for goal in self.world_track.active_goals[:2]),
                *(f"self-goal:{goal}" for goal in self.self_track.active_goals[:2]),
                f"cross-track:{self.cross_track_tension:.2f}",
            ),
        )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _goal_from_entry(entry: MemoryEntry) -> str:
    return entry.content


def _semantic_embedding(text: str, *, dim: int = 8) -> tuple[float, ...]:
    tokens = _semantic_tokens(text)
    if not tokens:
        return tuple(0.0 for _ in range(dim))
    vector = [0.0 for _ in range(dim)]
    for token in tokens:
        token_scale = max(len(token), 1)
        for index, char in enumerate(token):
            vector[(index + len(token)) % dim] += (ord(char) % 37) / 37.0 / token_scale
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-6:
        return tuple(0.0 for _ in range(dim))
    return tuple(value / norm for value in vector)


def _semantic_tokens(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    ascii_buffer: list[str] = []
    compact = "".join(char for char in text.lower() if not char.isspace())
    for char in text.lower():
        if char.isascii() and char.isalnum():
            ascii_buffer.append(char)
            continue
        if ascii_buffer:
            tokens.append("".join(ascii_buffer))
            ascii_buffer.clear()
        if not char.isspace():
            tokens.append(char)
    if ascii_buffer:
        tokens.append("".join(ascii_buffer))
    tokens.extend(compact[index : index + 2] for index in range(len(compact) - 1))
    return tuple(tokens)


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right:
        return 0.0
    return sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))


WORLD_TRACK_PROTOTYPE = _semantic_embedding(
    "decide priority execute plan concrete action task urgency next step solve "
    "明确顺序 直接执行 判断取舍 推进任务 行动步骤"
)
SELF_TRACK_PROTOTYPE = _semantic_embedding(
    "feel overwhelmed need support warmth steadiness reassurance emotional care trust repair "
    "先陪我稳住 情绪支持 别急着解决 温暖 安抚 信任 修复"
)
SEMANTIC_OWNER_GOAL_LIMIT = 3
SEMANTIC_OWNER_CONTEXT_WEIGHT = 0.35


def _shared_track_affinity(entry: MemoryEntry, *, track: Track) -> float:
    embedding = _semantic_embedding(entry.content)
    target = WORLD_TRACK_PROTOTYPE if track is Track.WORLD else SELF_TRACK_PROTOTYPE
    other = SELF_TRACK_PROTOTYPE if track is Track.WORLD else WORLD_TRACK_PROTOTYPE
    target_score = (_cosine_similarity(embedding, target) + 1.0) / 2.0
    other_score = (_cosine_similarity(embedding, other) + 1.0) / 2.0
    return _clamp(target_score - other_score * 0.55)


def _project_shared_entries(
    shared_entries: tuple[MemoryEntry, ...],
    *,
    track: Track,
) -> tuple[MemoryEntry, ...]:
    ranked = sorted(
        (
            (_shared_track_affinity(entry, track=track), entry)
            for entry in shared_entries
        ),
        key=lambda item: (-item[0], -item[1].strength, -item[1].created_at_ms),
    )
    return tuple(entry for affinity, entry in ranked if affinity > 0.12)


def _shared_track_controller_code(
    shared_entries: tuple[MemoryEntry, ...],
    *,
    track: Track,
) -> tuple[float, float]:
    scored = tuple(
        (_shared_track_affinity(entry, track=track), entry)
        for entry in shared_entries
    )
    total_affinity = sum(max(affinity, 0.0) for affinity, _ in scored)
    if total_affinity <= 1e-6:
        return (0.0, 0.0)
    weighted_strength = sum(entry.strength * max(affinity, 0.0) for affinity, entry in scored) / total_affinity
    weighted_presence = min(total_affinity / max(len(scored), 1), 1.0)
    return (_clamp(weighted_strength), _clamp(weighted_presence))


def _semantic_owner_context(
    semantic_snapshots: Mapping[str, Snapshot[object]],
    *,
    track: Track,
) -> tuple[tuple[str, ...], float]:
    from volvence_zero.semantic_state import (
        BoundaryConsentSnapshot,
        GoalValueSnapshot,
        RelationshipStateSnapshot,
        UserModelSnapshot,
    )

    slot_names = WORLD_SEMANTIC_OWNER_SLOTS if track is Track.WORLD else SELF_SEMANTIC_OWNER_SLOTS
    goals: list[str] = []
    for slot_name in slot_names:
        snapshot = semantic_snapshots.get(slot_name)
        if snapshot is None or isinstance(snapshot.value, RuntimePlaceholderValue):
            continue
        value = snapshot.value
        if isinstance(value, GoalValueSnapshot):
            if value.active_goal_id is not None:
                goals.append(f"goal_value:active_goal={value.active_goal_id}")
            if value.tradeoff_notes:
                goals.append(
                    f"goal_value:tradeoffs={value.active_tradeoff_count} "
                    f"conflict={value.value_conflict:.2f} readiness={value.decision_readiness:.2f} "
                    f"reversibility={value.reversibility_need:.2f}"
                )
        elif isinstance(value, RelationshipStateSnapshot):
            goals.append(
                "relationship_state:"
                f"trust={value.trust_level:.2f} emotional_load={value.emotional_load:.2f} "
                f"repair_need={value.repair_need:.2f} stabilization={value.stabilization_need:.2f}"
            )
        elif isinstance(value, BoundaryConsentSnapshot):
            goals.append(
                "boundary_consent:"
                f"autonomy_risk={value.autonomy_risk:.2f} consent_clarity={value.consent_clarity:.2f} "
                f"overreach={value.overreach_risk:.2f}"
            )
        elif isinstance(value, UserModelSnapshot):
            goals.append(
                "user_model:"
                f"pacing={value.preferred_support_pacing} decision_style={value.decision_style} "
                f"overwhelm={value.overwhelm_pattern_strength:.2f} durable_goals={len(value.durable_goals)}"
            )
        else:
            description = value.description
            if description:
                goals.append(f"{slot_name}:{description}")
    presence = _clamp(len(goals) / max(len(slot_names), 1))
    return (tuple(goals[:SEMANTIC_OWNER_GOAL_LIMIT]), presence)


def _memory_controller_code(entries: tuple[MemoryEntry, ...]) -> tuple[float, ...]:
    if not entries:
        return (0.0, 0.0)
    average_strength = sum(entry.strength for entry in entries) / len(entries)
    average_recency = sum(entry.last_accessed_ms - entry.created_at_ms for entry in entries) / len(entries)
    normalized_recency = 1.0 if average_recency > 0 else 0.0
    return (_clamp(average_strength), _clamp(normalized_recency))


def _tension(
    entries: tuple[MemoryEntry, ...],
    *,
    shared_entries: tuple[MemoryEntry, ...] = (),
    temporal_controller_code: tuple[float, float, float] = (0.0, 0.0, 0.0),
    semantic_primary: float = 0.0,
    semantic_repair: float = 0.0,
) -> float:
    direct_signal = 0.0
    if entries:
        direct_signal = sum(entry.strength for entry in entries) / len(entries)
    shared_signal = 0.0
    if shared_entries:
        shared_signal = sum(entry.strength for entry in shared_entries[:3]) / len(shared_entries[:3]) * 0.7
    temporal_signal = temporal_controller_code[0] * 0.7 + temporal_controller_code[2] * 0.2
    semantic_signal = semantic_primary * 0.8 + semantic_repair * 0.2
    return _clamp(max(direct_signal, shared_signal, temporal_signal, semantic_signal))


def entries_by_track(memory_snapshot: MemorySnapshot, track: Track) -> tuple[MemoryEntry, ...]:
    return tuple(entry for entry in memory_snapshot.retrieved_entries if entry.track is track)


def derive_track_state(
    *,
    track: Track,
    memory_entries: tuple[MemoryEntry, ...],
    shared_entries: tuple[MemoryEntry, ...] = (),
    temporal_snapshot: Any = None,
    track_temporal_snapshot: Any = None,
    substrate_snapshot: SubstrateSnapshot | None = None,
    semantic_owner_goals: tuple[str, ...] = (),
    semantic_owner_presence: float = 0.0,
) -> TrackState:
    projected_shared_entries = _project_shared_entries(shared_entries, track=track)
    base_entries = memory_entries if memory_entries else projected_shared_entries[:3]
    goals = list(semantic_owner_goals[:SEMANTIC_OWNER_GOAL_LIMIT])
    goals.extend(
        _goal_from_entry(entry)
        for entry in base_entries[: max(0, SEMANTIC_OWNER_GOAL_LIMIT - len(goals))]
    )
    temporal_controller_code, abstract_action_hint, action_family_version_hint, controller_source = _temporal_track_context(
        track=track,
        temporal_snapshot=temporal_snapshot,
        track_temporal_snapshot=track_temporal_snapshot,
    )
    semantic_primary, semantic_shared, semantic_repair = _substrate_track_context(
        track=track,
        substrate_snapshot=substrate_snapshot,
    )
    if not goals and semantic_primary > 0.48:
        goals.append("substrate:task-focused" if track is Track.WORLD else "substrate:support-focused")
    if len(goals) < 2 and semantic_shared > 0.45:
        goals.append("substrate:exploratory")
    if not goals and abstract_action_hint is not None:
        goals.append(f"temporal:{abstract_action_hint}")
    recent_credits = tuple(
        (entry.entry_id, round(entry.strength, 3)) for entry in base_entries[:3]
    )
    memory_controller_code = _memory_controller_code(base_entries)
    if not memory_entries and projected_shared_entries:
        memory_controller_code = _shared_track_controller_code(projected_shared_entries, track=track)
    controller_code = (
        _clamp(
            memory_controller_code[0] * 0.25
            + temporal_controller_code[0] * 0.20
            + semantic_primary * 0.20
            + semantic_owner_presence * SEMANTIC_OWNER_CONTEXT_WEIGHT
        ),
        _clamp(
            memory_controller_code[1] * 0.20
            + temporal_controller_code[1] * 0.25
            + semantic_shared * 0.35
            + semantic_owner_presence * 0.20
        ),
        temporal_controller_code[2],
    )
    if semantic_owner_goals:
        controller_source = (
            "semantic-owner"
            if controller_source == "memory"
            else f"{controller_source}+semantic-owner"
        )
    return TrackState(
        track=track,
        active_goals=tuple(goals),
        recent_credits=recent_credits,
        controller_code=controller_code,
        tension_level=_tension(
            memory_entries,
            shared_entries=projected_shared_entries,
            temporal_controller_code=temporal_controller_code,
            semantic_primary=semantic_primary,
            semantic_repair=semantic_repair,
        ),
        abstract_action_hint=abstract_action_hint,
        action_family_version_hint=action_family_version_hint,
        controller_source=controller_source,
    )


def derive_cross_track_tension(
    world_track: TrackState,
    self_track: TrackState,
    *,
    substrate_snapshot: SubstrateSnapshot | None = None,
) -> float:
    goal_overlap = len(set(world_track.active_goals) & set(self_track.active_goals))
    overlap_penalty = 0.15 if goal_overlap else 0.0
    controller_divergence = (
        abs(world_track.controller_code[0] - self_track.controller_code[0]) * 0.45
        + abs(world_track.controller_code[1] - self_track.controller_code[1]) * 0.20
    )
    semantic_support = (
        feature_signal_value(substrate_snapshot.feature_surface, name="semantic_support_pull")
        if substrate_snapshot is not None
        else 0.0
    )
    semantic_task = (
        feature_signal_value(substrate_snapshot.feature_surface, name="semantic_task_pull")
        if substrate_snapshot is not None
        else 0.0
    )
    semantic_exploration = (
        feature_signal_value(substrate_snapshot.feature_surface, name="semantic_exploration_pull")
        if substrate_snapshot is not None
        else 0.0
    )
    fallback_active = (
        feature_signal_value(substrate_snapshot.feature_surface, name="fallback_active")
        if substrate_snapshot is not None
        else 0.0
    )
    hook_coverage = (
        feature_signal_value(substrate_snapshot.feature_surface, name="hook_layer_coverage")
        if substrate_snapshot is not None
        else 0.0
    )
    real_path_bonus = (1.0 - fallback_active) * (0.05 + hook_coverage * 0.08)
    coordination_bonus = min(semantic_support, semantic_task) * 0.08 + semantic_exploration * 0.04
    return _clamp(
        abs(world_track.tension_level - self_track.tension_level)
        + overlap_penalty
        + controller_divergence
        - real_path_bonus
        - coordination_bonus
    )


class DualTrackModule(RuntimeModule[DualTrackSnapshot]):
    slot_name = "dual_track"
    owner = "DualTrackModule"
    value_type = DualTrackSnapshot
    dependencies = (
        "memory",
        "temporal_abstraction",
        "world_temporal",
        "self_temporal",
        "substrate",
        *SEMANTIC_OWNER_SLOTS,
    )
    default_wiring_level = WiringLevel.SHADOW

    async def process(self, upstream: Mapping[str, Snapshot[object]]) -> Snapshot[DualTrackSnapshot]:
        memory_snapshot = upstream["memory"]
        temporal_snapshot = upstream.get("temporal_abstraction")
        world_temporal_snapshot = upstream.get("world_temporal")
        self_temporal_snapshot = upstream.get("self_temporal")
        substrate_snapshot = upstream.get("substrate")
        semantic_snapshots = {
            slot_name: snapshot
            for slot_name in SEMANTIC_OWNER_SLOTS
            if (snapshot := upstream.get(slot_name)) is not None
        }
        world_semantic_goals, world_semantic_presence = _semantic_owner_context(
            semantic_snapshots,
            track=Track.WORLD,
        )
        self_semantic_goals, self_semantic_presence = _semantic_owner_context(
            semantic_snapshots,
            track=Track.SELF,
        )
        memory_value = memory_snapshot.value
        temporal_value = temporal_snapshot.value if temporal_snapshot is not None else None
        world_temporal_value = world_temporal_snapshot.value if world_temporal_snapshot is not None else None
        self_temporal_value = self_temporal_snapshot.value if self_temporal_snapshot is not None else None
        substrate_value = substrate_snapshot.value if substrate_snapshot is not None and isinstance(substrate_snapshot.value, SubstrateSnapshot) else None
        if not isinstance(memory_value, MemorySnapshot):
            world_track = derive_track_state(
                track=Track.WORLD,
                memory_entries=(),
                temporal_snapshot=temporal_value,
                track_temporal_snapshot=world_temporal_value,
                substrate_snapshot=substrate_value,
                semantic_owner_goals=world_semantic_goals,
                semantic_owner_presence=world_semantic_presence,
            )
            self_track = derive_track_state(
                track=Track.SELF,
                memory_entries=(),
                temporal_snapshot=temporal_value,
                track_temporal_snapshot=self_temporal_value,
                substrate_snapshot=substrate_value,
                semantic_owner_goals=self_semantic_goals,
                semantic_owner_presence=self_semantic_presence,
            )
        else:
            world_entries = entries_by_track(memory_value, Track.WORLD)
            self_entries = entries_by_track(memory_value, Track.SELF)
            shared_entries = entries_by_track(memory_value, Track.SHARED)
            world_track = derive_track_state(
                track=Track.WORLD,
                memory_entries=world_entries,
                shared_entries=shared_entries,
                temporal_snapshot=temporal_value,
                track_temporal_snapshot=world_temporal_value,
                substrate_snapshot=substrate_value,
                semantic_owner_goals=world_semantic_goals,
                semantic_owner_presence=world_semantic_presence,
            )
            self_track = derive_track_state(
                track=Track.SELF,
                memory_entries=self_entries,
                shared_entries=shared_entries,
                temporal_snapshot=temporal_value,
                track_temporal_snapshot=self_temporal_value,
                substrate_snapshot=substrate_value,
                semantic_owner_goals=self_semantic_goals,
                semantic_owner_presence=self_semantic_presence,
            )

        cross_track_tension = derive_cross_track_tension(
            world_track,
            self_track,
            substrate_snapshot=substrate_value,
        )
        description = (
            f"Dual-track state with world_tension={world_track.tension_level:.2f}, "
            f"self_tension={self_track.tension_level:.2f}, "
            f"cross_track_tension={cross_track_tension:.2f}, "
            f"world_controller_source={world_track.controller_source}, "
            f"self_controller_source={self_track.controller_source}."
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
        temporal_snapshot = kwargs.get("temporal_snapshot")
        world_temporal_snapshot = kwargs.get("world_temporal_snapshot")
        self_temporal_snapshot = kwargs.get("self_temporal_snapshot")
        substrate_snapshot = kwargs.get("substrate_snapshot")
        shared_entries = kwargs.get("shared_entries")
        semantic_snapshots = kwargs.get("semantic_snapshots")
        if world_entries is None:
            world_entries = ()
        if self_entries is None:
            self_entries = ()
        if shared_entries is None:
            shared_entries = ()
        if not isinstance(world_entries, tuple):
            raise TypeError("world_entries must be a tuple when provided.")
        if not isinstance(self_entries, tuple):
            raise TypeError("self_entries must be a tuple when provided.")
        if not isinstance(shared_entries, tuple):
            raise TypeError("shared_entries must be a tuple when provided.")
        if semantic_snapshots is None:
            semantic_snapshots = {}
        if not isinstance(semantic_snapshots, Mapping):
            raise TypeError("semantic_snapshots must be a mapping when provided.")
        world_semantic_goals, world_semantic_presence = _semantic_owner_context(
            semantic_snapshots,
            track=Track.WORLD,
        )
        self_semantic_goals, self_semantic_presence = _semantic_owner_context(
            semantic_snapshots,
            track=Track.SELF,
        )

        world_track = derive_track_state(
            track=Track.WORLD,
            memory_entries=world_entries,
            shared_entries=shared_entries,
            temporal_snapshot=temporal_snapshot,
            track_temporal_snapshot=world_temporal_snapshot,
            substrate_snapshot=substrate_snapshot if isinstance(substrate_snapshot, SubstrateSnapshot) else None,
            semantic_owner_goals=world_semantic_goals,
            semantic_owner_presence=world_semantic_presence,
        )
        self_track = derive_track_state(
            track=Track.SELF,
            memory_entries=self_entries,
            shared_entries=shared_entries,
            temporal_snapshot=temporal_snapshot,
            track_temporal_snapshot=self_temporal_snapshot,
            substrate_snapshot=substrate_snapshot if isinstance(substrate_snapshot, SubstrateSnapshot) else None,
            semantic_owner_goals=self_semantic_goals,
            semantic_owner_presence=self_semantic_presence,
        )
        cross_track_tension = derive_cross_track_tension(
            world_track,
            self_track,
            substrate_snapshot=substrate_snapshot if isinstance(substrate_snapshot, SubstrateSnapshot) else None,
        )
        return self.publish(
            DualTrackSnapshot(
                world_track=world_track,
                self_track=self_track,
                cross_track_tension=cross_track_tension,
                description="Standalone dual-track snapshot.",
            )
        )


def _temporal_track_context(
    *,
    track: Track,
    temporal_snapshot: Any,
    track_temporal_snapshot: Any = None,
) -> tuple[tuple[float, float, float], str | None, int, str]:
    from volvence_zero.temporal.interface import TemporalAbstractionSnapshot

    if isinstance(track_temporal_snapshot, TemporalAbstractionSnapshot):
        controller_code = track_temporal_snapshot.controller_state.code
        if len(controller_code) >= 3:
            return (
                (
                    _clamp(controller_code[0]),
                    _clamp(controller_code[1]),
                    _clamp(track_temporal_snapshot.controller_state.switch_gate),
                ),
                track_temporal_snapshot.active_abstract_action,
                track_temporal_snapshot.action_family_version,
                "temporal-track-owner",
            )
        return (
            (
                _clamp(controller_code[0]) if controller_code else 0.0,
                _clamp(controller_code[1]) if len(controller_code) > 1 else 0.0,
                _clamp(track_temporal_snapshot.controller_state.switch_gate),
            ),
            track_temporal_snapshot.active_abstract_action,
            track_temporal_snapshot.action_family_version,
            "temporal-track-owner",
        )
    if not isinstance(temporal_snapshot, TemporalAbstractionSnapshot):
        return ((0.0, 0.0, 0.0), None, 0, "memory")
    track_code = _extract_track_code(temporal_snapshot, track)
    if track_code is not None:
        return (
            (
                _clamp(track_code[0]) if len(track_code) > 0 else 0.0,
                _clamp(track_code[1]) if len(track_code) > 1 else 0.0,
                _clamp(temporal_snapshot.controller_state.switch_gate),
            ),
            temporal_snapshot.active_abstract_action,
            temporal_snapshot.action_family_version,
            "temporal-track-projected",
        )
    controller_code = temporal_snapshot.controller_state.code
    world_component = controller_code[0] if len(controller_code) > 0 else 0.0
    self_component = controller_code[1] if len(controller_code) > 1 else 0.0
    shared_component = controller_code[2] if len(controller_code) > 2 else 0.0
    track_component = world_component if track is Track.WORLD else self_component
    return (
        (
            _clamp(track_component),
            _clamp(shared_component),
            _clamp(temporal_snapshot.controller_state.switch_gate),
        ),
        temporal_snapshot.active_abstract_action,
        temporal_snapshot.action_family_version,
        "temporal+memory",
    )


def _extract_track_code(
    temporal_snapshot: Any,
    track: Track,
) -> tuple[float, ...] | None:
    track_codes = temporal_snapshot.controller_state.track_codes
    if not track_codes:
        return None
    for track_name, code in track_codes:
        if track_name == track.value:
            return code
    return None


def _substrate_track_context(
    *,
    track: Track,
    substrate_snapshot: SubstrateSnapshot | None,
) -> tuple[float, float, float]:
    if substrate_snapshot is None:
        return (0.0, 0.0, 0.0)
    task_pull = feature_signal_value(substrate_snapshot.feature_surface, name="semantic_task_pull")
    support_pull = feature_signal_value(substrate_snapshot.feature_surface, name="semantic_support_pull")
    repair_pull = feature_signal_value(substrate_snapshot.feature_surface, name="semantic_repair_pull")
    exploration_pull = feature_signal_value(substrate_snapshot.feature_surface, name="semantic_exploration_pull")
    directive_pull = feature_signal_value(substrate_snapshot.feature_surface, name="semantic_directive_pull")
    if track is Track.WORLD:
        primary = _clamp(task_pull * 0.60 + directive_pull * 0.30 + repair_pull * 0.10)
    else:
        primary = _clamp(support_pull * 0.72 + repair_pull * 0.20 + max(0.0, 0.12 - directive_pull) * 0.08)
    shared = _clamp(exploration_pull * 0.60 + repair_pull * 0.10 + support_pull * 0.08 + directive_pull * 0.22)
    return (primary, shared, repair_pull)
