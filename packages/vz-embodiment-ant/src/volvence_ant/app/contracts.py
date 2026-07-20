"""Immutable contracts for the digital-ant realtime experiment app.

These values are an embodiment-facing transport surface.  They are not
runtime slots and never become an input to the ant controller.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
import math
from typing import Any

from volvence_ant.env.world_objects import (
    WorldObjectKind,
    WorldObjectSnapshot,
)

APP_SCHEMA_VERSION = "digital-ant-app.v2"


class AppMode(str, Enum):
    SOLO = "solo"
    COLONY = "colony"


class AppArm(str, Enum):
    LEARNED = "learned"
    NO_OPTIMIZE = "no_optimize"
    FIXED_RULE = "fixed_rule"


class AppObjective(str, Enum):
    FORAGING = "foraging"
    HEADING_STABILITY = "heading_stability"
    ECOLOGY = "ecology"


class AppRunState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class AppCommandKind(str, Enum):
    PAUSE = "pause"
    RESUME = "resume"
    STEP = "step"
    SET_SPEED = "set_speed"
    STOP = "stop"


class AppDisturbanceKind(str, Enum):
    RELOCATE_FOOD = "relocate_food"
    TRIGGER_ALARM = "trigger_alarm"
    MOTOR_DISTORTION = "motor_distortion"
    UPSERT_WORLD_OBJECT = "upsert_world_object"
    MOVE_WORLD_OBJECT = "move_world_object"
    REMOVE_WORLD_OBJECT = "remove_world_object"


class AppEventKind(str, Enum):
    FRAME = "frame"
    STATUS = "status"
    DISTURBANCE = "disturbance"
    ERROR = "error"


@dataclass(frozen=True)
class AppExperimentConfig:
    """A frozen, reproducible run configuration."""

    mode: AppMode = AppMode.SOLO
    arm: AppArm = AppArm.LEARNED
    objective: AppObjective = AppObjective.FORAGING
    seed: int = 0
    n_ants: int = 1
    temporal_latent_dim: int = 16
    tick_interval_ms: int = 150
    max_ticks: int | None = 1000
    autostart: bool = True
    food_x: float = 6.0
    food_y: float = 0.0
    motor_turn_gain: float = 1.0
    motor_turn_bias: float = 0.0
    motor_switch_tick: int | None = None
    motor_switched_turn_gain: float = 1.0
    motor_switched_turn_bias: float = 0.0

    def __post_init__(self) -> None:
        if self.n_ants < 1:
            raise ValueError("n_ants must be >= 1")
        if self.mode is AppMode.SOLO and self.n_ants != 1:
            raise ValueError("solo mode requires n_ants=1")
        if self.temporal_latent_dim < 3:
            raise ValueError("temporal_latent_dim must be >= 3")
        if self.tick_interval_ms < 0:
            raise ValueError("tick_interval_ms must be >= 0")
        if self.max_ticks is not None and self.max_ticks < 1:
            raise ValueError("max_ticks must be >= 1 when provided")
        if self.motor_switch_tick is not None and self.motor_switch_tick < 0:
            raise ValueError("motor_switch_tick must be >= 0")
        for name, value in (
            ("food_x", self.food_x),
            ("food_y", self.food_y),
            ("motor_turn_gain", self.motor_turn_gain),
            ("motor_turn_bias", self.motor_turn_bias),
            ("motor_switched_turn_gain", self.motor_switched_turn_gain),
            ("motor_switched_turn_bias", self.motor_switched_turn_bias),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class AppAntFrame:
    body_id: int
    x: float
    y: float
    heading: float
    target_heading: float
    carrying_food: bool
    action: str
    turn_command: float
    applied_turn: float
    step_command: float
    code: tuple[float, ...] = ()
    switch_gate: float = 0.0
    pe_magnitude: float = 0.0
    cumulative_credit: float = 0.0
    heading_stability_error: float = 0.0
    motor_execution_error: float = 0.0
    heat_center: float = 0.0
    heat_harmful: bool = False


@dataclass(frozen=True)
class AppEvidenceProjection:
    """Compact public evidence; never fed back into learning."""

    backend_wiring: tuple[tuple[str, str], ...] = ()
    runtime_replay_captured: int = 0
    runtime_replay_settled: int = 0
    runtime_replay_transitions: int = 0
    runtime_replay_lineage_matches: int = 0
    runtime_replay_drop_reasons: tuple[str, ...] = ()
    verdict: str = "BLOCK"
    verdict_reason: str = "formal evidence has not passed"
    checkpoint_loaded: bool = False
    checkpoint_fingerprint: str = ""
    checkpoint_verdict: str = "UNAVAILABLE"


@dataclass(frozen=True)
class AppFrame:
    schema_version: str
    run_id: str
    sequence: int
    tick: int
    tick_latency_ms: float
    mode: AppMode
    arm: AppArm
    objective: AppObjective
    nest: tuple[float, float]
    food: tuple[tuple[float, float], ...]
    delivered: int
    pickups: int
    ants: tuple[AppAntFrame, ...]
    trail: tuple[tuple[float, ...], ...] = ()
    evidence: AppEvidenceProjection = AppEvidenceProjection()
    objects: tuple[WorldObjectSnapshot, ...] = ()


@dataclass(frozen=True)
class AppRunStatus:
    schema_version: str
    run_id: str
    state: AppRunState
    tick: int
    sequence: int
    mode: AppMode
    arm: AppArm
    objective: AppObjective
    seed: int
    n_ants: int
    tick_interval_ms: int
    pending_disturbances: int
    frames_retained: int
    frames_dropped: int
    last_error: str = ""


@dataclass(frozen=True)
class AppCommand:
    command_id: str
    kind: AppCommandKind
    value: float | None = None

    def __post_init__(self) -> None:
        if not self.command_id:
            raise ValueError("command_id must not be empty")
        if self.kind is AppCommandKind.SET_SPEED:
            if self.value is None or not math.isfinite(self.value) or self.value < 0.0:
                raise ValueError("set_speed requires a finite value >= 0")
        elif self.value is not None:
            raise ValueError(f"{self.kind.value} does not accept a value")


@dataclass(frozen=True)
class AppDisturbance:
    event_id: str
    kind: AppDisturbanceKind
    requested_tick: int | None = None
    body_id: int | None = None
    food_index: int = 0
    x: float | None = None
    y: float | None = None
    magnitude: float | None = None
    turn_gain: float | None = None
    turn_bias: float | None = None
    object_id: str | None = None
    object_kind: WorldObjectKind | None = None
    start_x: float | None = None
    start_y: float | None = None
    end_x: float | None = None
    end_y: float | None = None
    radius: float | None = None
    strength: float | None = None
    decay: float | None = None
    remaining: float | None = None
    angle: float | None = None
    length: float | None = None
    harm_threshold: float | None = None
    delta_x: float | None = None
    delta_y: float | None = None

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        if self.requested_tick is not None and self.requested_tick < 0:
            raise ValueError("requested_tick must be >= 0")
        if self.body_id is not None and self.body_id < 0:
            raise ValueError("body_id must be >= 0")
        if self.kind is AppDisturbanceKind.RELOCATE_FOOD:
            if self.x is None or self.y is None:
                raise ValueError("relocate_food requires x and y")
        elif self.kind is AppDisturbanceKind.TRIGGER_ALARM:
            if self.magnitude is None or not math.isfinite(self.magnitude):
                raise ValueError("trigger_alarm requires a finite magnitude")
        elif self.kind is AppDisturbanceKind.MOTOR_DISTORTION:
            if self.turn_gain is None or self.turn_bias is None:
                raise ValueError("motor_distortion requires turn_gain and turn_bias")
            if not math.isfinite(self.turn_gain) or not math.isfinite(self.turn_bias):
                raise ValueError("motor distortion values must be finite")
        elif self.kind is AppDisturbanceKind.UPSERT_WORLD_OBJECT:
            self._validate_world_object_upsert()
        elif self.kind is AppDisturbanceKind.MOVE_WORLD_OBJECT:
            if (
                not self.object_id
                or self.delta_x is None
                or self.delta_y is None
                or not math.isfinite(self.delta_x)
                or not math.isfinite(self.delta_y)
            ):
                raise ValueError("move_world_object requires object_id and finite deltas")
        elif self.kind is AppDisturbanceKind.REMOVE_WORLD_OBJECT:
            if not self.object_id:
                raise ValueError("remove_world_object requires object_id")

    def _validate_world_object_upsert(self) -> None:
        if not self.object_id or self.object_kind is None:
            raise ValueError("upsert_world_object requires object_id and object_kind")
        if self.object_kind is WorldObjectKind.WOOD_STICK:
            coordinates = (
                self.start_x,
                self.start_y,
                self.end_x,
                self.end_y,
            )
            if any(value is None for value in coordinates):
                raise ValueError("wood_stick requires start_x, start_y, end_x and end_y")
            values = tuple(float(value) for value in coordinates if value is not None)
        else:
            if self.x is None or self.y is None:
                raise ValueError(f"{self.object_kind.value} requires x and y")
            values = (self.x, self.y)
        optional_values = (
            self.radius,
            self.strength,
            self.decay,
            self.remaining,
            self.angle,
            self.length,
            self.harm_threshold,
        )
        if not all(math.isfinite(value) for value in (*values, *(v for v in optional_values if v is not None))):
            raise ValueError("world object values must be finite")


@dataclass(frozen=True)
class AppDisturbanceRecord:
    disturbance: AppDisturbance
    status: str
    applied_tick: int | None
    detail: str


@dataclass(frozen=True)
class AppStreamEvent:
    sequence: int
    kind: AppEventKind
    payload_json: str


def json_dict(value: Any) -> dict[str, Any]:
    """Convert one frozen app contract to a JSON-compatible object."""

    payload = json.loads(json.dumps(asdict(value), separators=(",", ":")))
    if not isinstance(payload, dict):
        raise TypeError("app contract must serialize to a JSON object")
    return payload


__all__ = [
    "APP_SCHEMA_VERSION",
    "AppAntFrame",
    "AppArm",
    "AppCommand",
    "AppCommandKind",
    "AppDisturbance",
    "AppDisturbanceKind",
    "AppDisturbanceRecord",
    "AppEventKind",
    "AppEvidenceProjection",
    "AppExperimentConfig",
    "AppFrame",
    "AppMode",
    "AppObjective",
    "AppRunState",
    "AppRunStatus",
    "AppStreamEvent",
    "WorldObjectKind",
    "WorldObjectSnapshot",
    "json_dict",
]
