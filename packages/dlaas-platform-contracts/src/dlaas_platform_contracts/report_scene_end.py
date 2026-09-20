"""Durable idempotency contracts for destructive ``report`` scene closure.

The registry owns the ledger lifecycle.  This contracts module only defines
the frozen, transport-neutral values exchanged by the platform API and the
registry store; it contains no runtime or cognitive state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from collections.abc import Mapping
from typing import Any

from dlaas_platform_contracts.envelope import InteractionEnvelope, InteractionType


REPORT_SCENE_END_LEDGER_SCHEMA = "dlaas.report-scene-end-ledger"
REPORT_SCENE_END_LEDGER_SCHEMA_VERSION = 1


class SceneEndLedgerStatus(str, Enum):
    RESERVED = "RESERVED"
    COMPLETED = "COMPLETED"
    OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"


class SceneEndReserveOutcome(str, Enum):
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


@dataclass(frozen=True)
class ReportSceneEndRequest:
    """Canonical semantic input whose digest scopes a report idempotency key.

    ``output_contract`` is intentionally absent: JSON versus SSE is a
    transport preference and must not permit a second destructive scene end.
    """

    ai_id: str
    contract_id: str
    session_id: str
    end_user_ref: str
    interaction_type: str
    structured_context: Mapping[str, Any]
    protocol_version: str
    mode: str
    human_brief: str
    target_person_ids: tuple[str, ...]
    lang: str
    feedback: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        if not self.ai_id.strip():
            raise ValueError("report scene-end ai_id must be non-empty")
        if self.interaction_type != InteractionType.REPORT.value:
            raise ValueError("report scene-end request requires interaction_type=report")
        object.__setattr__(
            self,
            "structured_context",
            MappingProxyType(dict(self.structured_context)),
        )
        if self.feedback is not None:
            object.__setattr__(self, "feedback", MappingProxyType(dict(self.feedback)))

    @classmethod
    def from_envelope(
        cls, *, ai_id: str, envelope: InteractionEnvelope
    ) -> "ReportSceneEndRequest":
        return cls(
            ai_id=ai_id,
            contract_id=envelope.contract_id,
            session_id=envelope.session_id,
            end_user_ref=envelope.end_user_ref,
            interaction_type=envelope.interaction_type.value,
            structured_context=dict(envelope.structured_context),
            protocol_version=envelope.protocol_version,
            mode=envelope.mode.value,
            human_brief=envelope.human_brief,
            target_person_ids=envelope.target_person_ids,
            lang=envelope.lang,
            feedback=(
                envelope.feedback.to_json() if envelope.feedback is not None else None
            ),
        )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "ai_id": self.ai_id,
            "contract_id": self.contract_id,
            "end_user_ref": self.end_user_ref,
            "feedback": dict(self.feedback) if self.feedback is not None else None,
            "human_brief": self.human_brief,
            "interaction_type": self.interaction_type,
            "lang": self.lang,
            "mode": self.mode,
            "protocol_version": self.protocol_version,
            "session_id": self.session_id,
            "structured_context": dict(self.structured_context),
            "target_person_ids": list(self.target_person_ids),
        }

    @property
    def request_sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.canonical_payload())).hexdigest()


@dataclass(frozen=True)
class SceneEndLedgerRecord:
    schema_id: str
    schema_version: int
    contract_id: str
    ai_id: str
    idempotency_key: str
    request_sha256: str
    status: SceneEndLedgerStatus
    lease_token: str
    lease_expires_at_ms: int
    response_status: int | None
    response_body: Mapping[str, Any] | None
    unknown_reason: str
    created_at_ms: int
    updated_at_ms: int

    def __post_init__(self) -> None:
        if self.schema_id != REPORT_SCENE_END_LEDGER_SCHEMA:
            raise ValueError("scene-end ledger schema_id is incompatible")
        if self.schema_version != REPORT_SCENE_END_LEDGER_SCHEMA_VERSION:
            raise ValueError("scene-end ledger schema_version is incompatible")
        if not self.contract_id.strip() or not self.ai_id.strip():
            raise ValueError("scene-end ledger contract_id and ai_id must be non-empty")
        if not self.idempotency_key.strip():
            raise ValueError("scene-end ledger idempotency_key must be non-empty")
        if len(self.request_sha256) != 64:
            raise ValueError("scene-end ledger request_sha256 must be SHA-256 hex")
        try:
            int(self.request_sha256, 16)
        except ValueError as exc:
            raise ValueError(
                "scene-end ledger request_sha256 must be SHA-256 hex"
            ) from exc
        if self.created_at_ms < 1 or self.updated_at_ms < self.created_at_ms:
            raise ValueError("scene-end ledger timestamps are invalid")
        if self.status is SceneEndLedgerStatus.COMPLETED:
            if self.response_status is None or self.response_body is None:
                raise ValueError("completed scene-end ledger record requires response")
        elif self.response_status is not None or self.response_body is not None:
            raise ValueError("non-completed scene-end ledger record cannot carry response")
        if self.status is SceneEndLedgerStatus.OUTCOME_UNKNOWN:
            if not self.unknown_reason.strip():
                raise ValueError("unknown scene-end ledger record requires reason")
        elif self.unknown_reason:
            raise ValueError("only unknown scene-end ledger record can carry reason")
        if self.response_body is not None:
            object.__setattr__(
                self, "response_body", MappingProxyType(dict(self.response_body))
            )


@dataclass(frozen=True)
class SceneEndReservation:
    outcome: SceneEndReserveOutcome
    record: SceneEndLedgerRecord

    @property
    def acquired(self) -> bool:
        return self.outcome is SceneEndReserveOutcome.ACQUIRED


__all__ = [
    "REPORT_SCENE_END_LEDGER_SCHEMA",
    "REPORT_SCENE_END_LEDGER_SCHEMA_VERSION",
    "ReportSceneEndRequest",
    "SceneEndLedgerRecord",
    "SceneEndLedgerStatus",
    "SceneEndReservation",
    "SceneEndReserveOutcome",
]
