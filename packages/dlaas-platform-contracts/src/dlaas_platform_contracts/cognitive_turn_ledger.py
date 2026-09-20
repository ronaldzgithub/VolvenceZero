"""Durable idempotency contracts for native cognitive turns.

The Registry owns execution identity and replay state.  These frozen values
only bind the caller-owned semantic input to that platform ledger; they do
not own Lifeform memory, cognition, or expression state.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from dlaas_platform_contracts.cognitive_turn import (
    cognition_task_to_json,
    expression_contract_to_json,
)
from dlaas_platform_contracts.envelope import InteractionEnvelope, InteractionType


COGNITIVE_TURN_LEDGER_SCHEMA = "dlaas.cognitive-turn-ledger"
COGNITIVE_TURN_LEDGER_SCHEMA_VERSION = 1


class CognitiveTurnLedgerStatus(str, Enum):
    RESERVED = "RESERVED"
    COMPLETED = "COMPLETED"
    OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"


class CognitiveTurnReserveOutcome(str, Enum):
    ACQUIRED = "acquired"
    REPLAY_COMPLETED = "replay_completed"
    IN_PROGRESS = "in_progress"
    CONFLICT = "conflict"
    OUTCOME_UNKNOWN = "outcome_unknown"


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _freeze_json(value: Any) -> Any:
    """Detach and recursively freeze one JSON-compatible value."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class CognitiveTurnRequest:
    """Canonical semantic input bound to one cognitive idempotency key.

    ``output_contract`` is deliberately absent because delivery channel,
    response format, and streaming are transport preferences.  Every field
    that can change the Lifeform turn is present, including the four native
    cognitive contracts.
    """

    ai_id: str
    contract_id: str
    session_id: str
    end_user_ref: str
    interaction_type: str
    protocol_version: str
    mode: str
    human_brief: str
    structured_context: Mapping[str, Any]
    feedback: Mapping[str, Any] | None
    target_person_ids: tuple[str, ...]
    lang: str
    template_binding: Mapping[str, Any] | None
    perceived_event: Mapping[str, Any]
    cognition_task: Mapping[str, Any]
    expression_contract: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.ai_id.strip():
            raise ValueError("cognitive-turn ai_id must be non-empty")
        if self.interaction_type != InteractionType.COGNITIVE_TURN.value:
            raise ValueError("cognitive-turn request requires interaction_type=cognitive_turn")
        for name in ("perceived_event", "cognition_task", "expression_contract"):
            value = getattr(self, name)
            if not isinstance(value, Mapping) or not value:
                raise ValueError(f"cognitive-turn {name} must be a non-empty mapping")
            object.__setattr__(self, name, _freeze_json(value))
        if not isinstance(self.structured_context, Mapping):
            raise ValueError("cognitive-turn structured_context must be a mapping")
        object.__setattr__(
            self, "structured_context", _freeze_json(self.structured_context)
        )
        if self.feedback is not None:
            if not isinstance(self.feedback, Mapping):
                raise ValueError("cognitive-turn feedback must be a mapping or null")
            object.__setattr__(self, "feedback", _freeze_json(self.feedback))
        if self.template_binding is not None:
            object.__setattr__(self, "template_binding", _freeze_json(self.template_binding))
        # Fail at contract construction, not during reservation, if a caller
        # bypasses InteractionEnvelope and supplies non-canonical JSON values.
        _canonical_json_bytes(self.canonical_payload())

    @classmethod
    def from_envelope(cls, *, ai_id: str, envelope: InteractionEnvelope) -> "CognitiveTurnRequest":
        if envelope.interaction_type is not InteractionType.COGNITIVE_TURN:
            raise ValueError("cognitive-turn ledger accepts cognitive_turn only")
        if (
            envelope.perceived_event is None or envelope.cognition_task is None or envelope.expression_contract is None
        ):  # pragma: no cover - InteractionEnvelope invariant
            raise ValueError("cognitive-turn envelope is missing native contracts")
        return cls(
            ai_id=ai_id,
            contract_id=envelope.contract_id,
            session_id=envelope.session_id,
            end_user_ref=envelope.end_user_ref,
            interaction_type=envelope.interaction_type.value,
            protocol_version=envelope.protocol_version,
            mode=envelope.mode.value,
            human_brief=envelope.human_brief,
            structured_context=dict(envelope.structured_context),
            feedback=(
                envelope.feedback.to_json() if envelope.feedback is not None else None
            ),
            target_person_ids=envelope.target_person_ids,
            lang=envelope.lang,
            template_binding=(envelope.template_binding.to_json() if envelope.template_binding is not None else None),
            perceived_event=envelope.perceived_event.to_json(),
            cognition_task=cognition_task_to_json(envelope.cognition_task),
            expression_contract=expression_contract_to_json(envelope.expression_contract),
        )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "ai_id": self.ai_id,
            "cognition_task": _thaw_json(self.cognition_task),
            "contract_id": self.contract_id,
            "end_user_ref": self.end_user_ref,
            "expression_contract": _thaw_json(self.expression_contract),
            "feedback": _thaw_json(self.feedback),
            "human_brief": self.human_brief,
            "interaction_type": self.interaction_type,
            "lang": self.lang,
            "mode": self.mode,
            "perceived_event": _thaw_json(self.perceived_event),
            "protocol_version": self.protocol_version,
            "session_id": self.session_id,
            "structured_context": _thaw_json(self.structured_context),
            "target_person_ids": list(self.target_person_ids),
            "template_binding": (_thaw_json(self.template_binding) if self.template_binding is not None else None),
        }

    @property
    def request_sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.canonical_payload())).hexdigest()


@dataclass(frozen=True)
class CognitiveTurnLedgerRecord:
    schema_id: str
    schema_version: int
    contract_id: str
    ai_id: str
    idempotency_key: str
    request_sha256: str
    status: CognitiveTurnLedgerStatus
    lease_token: str
    lease_expires_at_ms: int
    response_status: int | None
    response_body: Mapping[str, Any] | None
    unknown_reason: str
    created_at_ms: int
    updated_at_ms: int

    def __post_init__(self) -> None:
        if self.schema_id != COGNITIVE_TURN_LEDGER_SCHEMA:
            raise ValueError("cognitive-turn ledger schema_id is incompatible")
        if self.schema_version != COGNITIVE_TURN_LEDGER_SCHEMA_VERSION:
            raise ValueError("cognitive-turn ledger schema_version is incompatible")
        if not self.contract_id.strip() or not self.ai_id.strip():
            raise ValueError("cognitive-turn ledger contract_id and ai_id must be non-empty")
        if not self.idempotency_key.strip():
            raise ValueError("cognitive-turn ledger idempotency_key must be non-empty")
        if len(self.request_sha256) != 64:
            raise ValueError("cognitive-turn ledger request_sha256 must be SHA-256 hex")
        try:
            int(self.request_sha256, 16)
        except ValueError as exc:
            raise ValueError("cognitive-turn ledger request_sha256 must be SHA-256 hex") from exc
        if self.created_at_ms < 1 or self.updated_at_ms < self.created_at_ms:
            raise ValueError("cognitive-turn ledger timestamps are invalid")
        if self.status is CognitiveTurnLedgerStatus.COMPLETED:
            if self.response_status is None or self.response_body is None:
                raise ValueError("completed cognitive-turn ledger record requires response")
        elif self.response_status is not None or self.response_body is not None:
            raise ValueError("non-completed cognitive-turn ledger record cannot carry response")
        if self.status is CognitiveTurnLedgerStatus.OUTCOME_UNKNOWN:
            if not self.unknown_reason.strip():
                raise ValueError("unknown cognitive-turn ledger record requires reason")
        elif self.unknown_reason:
            raise ValueError("only unknown cognitive-turn ledger record can carry reason")
        if self.response_body is not None:
            object.__setattr__(self, "response_body", MappingProxyType(dict(self.response_body)))


@dataclass(frozen=True)
class CognitiveTurnReservation:
    outcome: CognitiveTurnReserveOutcome
    record: CognitiveTurnLedgerRecord

    @property
    def acquired(self) -> bool:
        return self.outcome is CognitiveTurnReserveOutcome.ACQUIRED


__all__ = [
    "COGNITIVE_TURN_LEDGER_SCHEMA",
    "COGNITIVE_TURN_LEDGER_SCHEMA_VERSION",
    "CognitiveTurnLedgerRecord",
    "CognitiveTurnLedgerStatus",
    "CognitiveTurnRequest",
    "CognitiveTurnReservation",
    "CognitiveTurnReserveOutcome",
]
