# Copyright 2026 Companion Standard Contributors
# Licensed under the Apache License, Version 2.0.

"""Theory-of-mind record types — typed state about another mind.

Part of the Relationship Representation Standard: the four distinct
ToM state kinds (belief / intent / feeling / preference), the record
lifecycle status, and the immutable inferred-state record itself.

What deliberately does NOT live here (runtime mechanism, private):
owner-published snapshots (which embed runtime proposal diagnostics),
social prediction / prediction-error records, common-ground and group
state, and all aggregation helpers.

Extracted from the upstream production runtime's social-cognition contract
module (Phase A1 of the standard split); the runtime keeps its original
import paths by re-exporting from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OtherMindRecordKind(str, Enum):
    """Four distinct Theory-of-Mind state kinds (R17)."""

    BELIEF = "belief"
    INTENT = "intent"
    FEELING = "feeling"
    PREFERENCE = "preference"


class OtherMindRecordStatus(str, Enum):
    """Lifecycle state for an inferred other-mind record."""

    ACTIVE = "active"
    CONTESTED = "contested"
    RETIRED = "retired"


@dataclass(frozen=True)
class OtherMindRecord:
    record_id: str
    interlocutor_id: str
    kind: OtherMindRecordKind
    summary: str
    detail: str
    confidence: float
    status: OtherMindRecordStatus
    source_turn: int
    prediction_error_refs: tuple[str, ...]
    evidence: str

    def __post_init__(self) -> None:
        _require_non_empty("record_id", self.record_id)
        _require_non_empty("interlocutor_id", self.interlocutor_id)
        _require_non_empty("summary", self.summary)
        _require_non_empty("detail", self.detail)
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence!r}"
            )
        if self.source_turn < 0:
            raise ValueError(f"source_turn must be >= 0, got {self.source_turn!r}")
        _require_unique_non_empty("prediction_error_refs", self.prediction_error_refs)
        _require_non_empty("evidence", self.evidence)


def _require_non_empty(field_name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _require_unique_non_empty(field_name: str, values: tuple[str, ...]) -> None:
    for value in values:
        if not value.strip():
            raise ValueError(f"{field_name} entries must be non-empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} entries must be unique")


__all__ = [
    "OtherMindRecord",
    "OtherMindRecordKind",
    "OtherMindRecordStatus",
]
