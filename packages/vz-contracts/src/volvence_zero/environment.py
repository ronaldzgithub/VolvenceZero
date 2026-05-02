"""Environment interface contracts.

These frozen dataclasses are the Phase 1 runtime surface for
``docs/specs/environment-interface.md``. They are data-only contracts in
``vz-contracts`` so kernel wheels and lifeform adapters can share the same
canonical event shape without reversing package dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from volvence_zero.social_cognition import (
    PRIMARY_INTERLOCUTOR_ID,
    SELF_INTERLOCUTOR_ID,
)


class EnvironmentEventKind(str, Enum):
    """Canonical categories for facts crossing the environment boundary."""

    USER_INPUT = "user_input"
    SYSTEM_TICK = "system_tick"
    SCENE_EVENT = "scene_event"
    TOOL_RESULT = "tool_result"
    INGESTION = "ingestion"
    APPRENTICE = "apprentice"
    INTERNAL_DRIVE = "internal_drive"
    FOLLOWUP_DUE = "followup_due"


@dataclass(frozen=True)
class EnvironmentActorRef:
    """Stable reference to the actor that caused an environment event."""

    actor_id: str
    actor_kind: str = "interlocutor"
    display_name: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty("actor_id", self.actor_id)
        _require_non_empty("actor_kind", self.actor_kind)
        if self.display_name is not None:
            _require_non_empty("display_name", self.display_name)


@dataclass(frozen=True)
class EnvironmentFrame:
    """Social and operational frame attached to an environment event."""

    actor: EnvironmentActorRef
    active_speaker_id: str
    addressee_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    audience_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_non_empty("active_speaker_id", self.active_speaker_id)
        _require_non_empty_unique_tuple("addressee_ids", self.addressee_ids)
        _require_non_empty_unique_tuple("subject_ids", self.subject_ids)
        _require_non_empty_unique_tuple("audience_ids", self.audience_ids)

    @property
    def actor_id(self) -> str:
        return self.actor.actor_id


@dataclass(frozen=True)
class EnvironmentEvent:
    """Canonical event entering the cognition boundary."""

    event_id: str
    event_kind: EnvironmentEventKind
    trigger_kind: str
    frame: EnvironmentFrame
    scene_id: str
    timestamp_ms: int
    provenance: str
    consent_context: tuple[str, ...] = ()
    payload_summary: str = ""

    def __post_init__(self) -> None:
        _require_non_empty("event_id", self.event_id)
        _require_non_empty("trigger_kind", self.trigger_kind)
        _require_non_empty("scene_id", self.scene_id)
        _require_non_empty("provenance", self.provenance)
        _require_non_negative_int("timestamp_ms", self.timestamp_ms)
        _require_unique_non_empty("consent_context", self.consent_context)

    @property
    def actor_id(self) -> str:
        return self.frame.actor_id

    @property
    def active_speaker_id(self) -> str:
        return self.frame.active_speaker_id

    @property
    def addressee_ids(self) -> tuple[str, ...]:
        return self.frame.addressee_ids

    @property
    def subject_ids(self) -> tuple[str, ...]:
        return self.frame.subject_ids

    @property
    def audience_ids(self) -> tuple[str, ...]:
        return self.frame.audience_ids


@dataclass(frozen=True)
class EnvironmentOutcome:
    """Canonical evidence produced after an environment-facing action."""

    outcome_id: str
    event_id: str
    outcome_kind: EnvironmentEventKind
    action_id: str
    status: str
    summary: str
    detail: str
    confidence: float = 0.8
    prediction_id: str | None = None
    evidence: tuple[str, ...] = ()
    latency_ms: int | None = None
    monetary_cost: float = 0.0
    reversibility: str = "reversible"
    environment_state_delta_kind: str = "none"

    def __post_init__(self) -> None:
        _require_non_empty("outcome_id", self.outcome_id)
        _require_non_empty("event_id", self.event_id)
        _require_non_empty("action_id", self.action_id)
        _require_non_empty("status", self.status)
        _require_non_empty("summary", self.summary)
        _require_unit_interval("confidence", self.confidence)
        if self.prediction_id is not None:
            _require_non_empty("prediction_id", self.prediction_id)
        _require_unique_non_empty("evidence", self.evidence)
        if self.latency_ms is not None:
            _require_non_negative_int("latency_ms", self.latency_ms)
        _require_non_negative_float("monetary_cost", self.monetary_cost)
        _require_enum_value(
            "reversibility",
            self.reversibility,
            ("reversible", "costly", "irreversible"),
        )
        _require_non_empty("environment_state_delta_kind", self.environment_state_delta_kind)


def build_primary_environment_frame(
    *,
    actor_id: str = PRIMARY_INTERLOCUTOR_ID,
    actor_kind: str = "interlocutor",
    active_speaker_id: str = PRIMARY_INTERLOCUTOR_ID,
    addressee_ids: tuple[str, ...] = (SELF_INTERLOCUTOR_ID,),
    subject_ids: tuple[str, ...] = (PRIMARY_INTERLOCUTOR_ID,),
    audience_ids: tuple[str, ...] = (SELF_INTERLOCUTOR_ID,),
) -> EnvironmentFrame:
    """Return the single-party compatibility frame."""

    return EnvironmentFrame(
        actor=EnvironmentActorRef(actor_id=actor_id, actor_kind=actor_kind),
        active_speaker_id=active_speaker_id,
        addressee_ids=addressee_ids,
        subject_ids=subject_ids,
        audience_ids=audience_ids,
    )


def build_user_input_environment_event(
    *,
    event_id: str,
    user_input: str,
    scene_id: str,
    timestamp_ms: int,
    trigger_kind: str = EnvironmentEventKind.USER_INPUT.value,
    frame: EnvironmentFrame | None = None,
    provenance: str = "user_input",
    consent_context: tuple[str, ...] = (),
) -> EnvironmentEvent:
    """Build a compatibility EnvironmentEvent for the existing run_turn API."""

    return EnvironmentEvent(
        event_id=event_id,
        event_kind=EnvironmentEventKind.USER_INPUT,
        trigger_kind=trigger_kind,
        frame=frame or build_primary_environment_frame(),
        scene_id=scene_id,
        timestamp_ms=timestamp_ms,
        provenance=provenance,
        consent_context=consent_context,
        payload_summary=user_input,
    )


def _require_non_empty(field_name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _require_non_empty_items(field_name: str, values: tuple[str, ...]) -> None:
    for value in values:
        if not value.strip():
            raise ValueError(f"{field_name} entries must be non-empty")


def _require_unique_non_empty(field_name: str, values: tuple[str, ...]) -> None:
    _require_non_empty_items(field_name, values)
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} entries must be unique")


def _require_non_empty_unique_tuple(field_name: str, values: tuple[str, ...]) -> None:
    if not values:
        raise ValueError(f"{field_name} must contain at least one entry")
    _require_unique_non_empty(field_name, values)


def _require_non_negative_int(field_name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _require_non_negative_float(field_name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{field_name} must be non-negative")


def _require_unit_interval(field_name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1], got {value!r}")


def _require_enum_value(
    field_name: str,
    value: str,
    allowed: tuple[str, ...],
) -> None:
    if value not in allowed:
        raise ValueError(f"{field_name} must be one of {allowed}, got {value!r}")


__all__ = [
    "EnvironmentActorRef",
    "EnvironmentEvent",
    "EnvironmentEventKind",
    "EnvironmentFrame",
    "EnvironmentOutcome",
    "build_primary_environment_frame",
    "build_user_input_environment_event",
]
