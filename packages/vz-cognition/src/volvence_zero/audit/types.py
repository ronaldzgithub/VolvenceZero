"""Audit owner frozen dataclasses (architecture-uplift A5 / T11).

Schemas defined in
[`docs/specs/audit-owner.md`](../../../../../../docs/specs/audit-owner.md) §A5.1.

All dataclasses are ``frozen=True`` and use ``tuple[...]`` for collection
fields so AuditSnapshot is immutable end-to-end (R8 SSOT).
"""

from __future__ import annotations

import dataclasses
from typing import Literal

__all__ = [
    "AuditDetectedAttackClass",
    "AuditSnapshot",
    "AuditToolTrace",
    "AuditThresholdDecision",
    "AUDIT_THRESHOLD_DECISION_VALUES",
]


# spec §A5.1 — closed enum of audit threshold decisions.
AuditThresholdDecision = Literal["pass", "soft-warn", "hard-block"]
AUDIT_THRESHOLD_DECISION_VALUES: frozenset[str] = frozenset(
    {"pass", "soft-warn", "hard-block"}
)


@dataclasses.dataclass(frozen=True)
class AuditToolTrace:
    """N8 audit-agent tool 调用 trace；具体 tool 集归 OA-4."""

    tool_name: str
    tool_args_summary: str
    tool_output_summary: str
    duration_ms: int
    succeeded: bool


@dataclasses.dataclass(frozen=True)
class AuditDetectedAttackClass:
    """N8 8 类已知 attack 检测结果。"""

    attack_class: str
    detected: bool
    confidence: float
    evidence_summary: str


@dataclasses.dataclass(frozen=True)
class AuditSnapshot:
    """A5 audit owner published snapshot (spec §A5.1).

    Fail-loudly: ``threshold_decision`` must be one of the closed enum values
    or ``__post_init__`` raises.
    """

    audit_id: str
    timestamp_ms: int
    proposal_id: str | None
    risk_score: float
    transcript: tuple[str, ...]
    tool_traces: tuple[AuditToolTrace, ...]
    detected_attack_classes: tuple[AuditDetectedAttackClass, ...]
    threshold_decision: str
    description: str

    def __post_init__(self) -> None:
        if self.threshold_decision not in AUDIT_THRESHOLD_DECISION_VALUES:
            raise ValueError(
                f"AuditSnapshot.threshold_decision must be one of "
                f"{sorted(AUDIT_THRESHOLD_DECISION_VALUES)!r}, "
                f"got {self.threshold_decision!r}"
            )
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError(
                f"AuditSnapshot.risk_score must be within [0.0, 1.0], "
                f"got {self.risk_score!r}"
            )
