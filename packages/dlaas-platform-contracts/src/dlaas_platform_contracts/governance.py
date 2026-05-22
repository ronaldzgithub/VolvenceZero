"""DLaaS governance, audit, artifact, eval, usage, and consent contracts.

These are platform-owned records. They intentionally do not import
lifeform or kernel runtime modules: they describe audit/governance
state around a hosted life, not cognitive state inside that life.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DataJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ArtifactKind(str, Enum):
    PROTOCOL_JSON = "protocol_json"
    TRAINING_OUTPUT = "training_output"
    ADAPTER_CANDIDATE = "adapter_candidate"
    EVAL_REPORT = "eval_report"
    MONTHLY_REPORT = "monthly_report"


class PromotionDecision(str, Enum):
    PENDING = "pending"
    ALLOW = "allow"
    BLOCK = "block"


class EvalRunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    APPROVED = "approved"
    FAILED = "failed"


@dataclass(frozen=True)
class DataExportJob:
    job_id: str
    ai_id: str
    contract_id: str
    end_user_ref: str = ""
    status: DataJobStatus = DataJobStatus.COMPLETED
    artifact_ref: str = ""
    created_at_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "ai_id": self.ai_id,
            "contract_id": self.contract_id,
            "end_user_ref": self.end_user_ref,
            "status": self.status.value,
            "artifact_ref": self.artifact_ref,
            "created_at_ms": self.created_at_ms,
        }


@dataclass(frozen=True)
class DeletionJob:
    job_id: str
    ai_id: str
    contract_id: str
    end_user_ref: str
    status: DataJobStatus = DataJobStatus.COMPLETED
    deleted_scopes: tuple[str, ...] = ()
    created_at_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "ai_id": self.ai_id,
            "contract_id": self.contract_id,
            "end_user_ref": self.end_user_ref,
            "status": self.status.value,
            "deleted_scopes": list(self.deleted_scopes),
            "created_at_ms": self.created_at_ms,
        }


@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    event_type: str
    ai_id: str = ""
    contract_id: str = ""
    session_id: str = ""
    actor: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)
    created_at_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "ai_id": self.ai_id,
            "contract_id": self.contract_id,
            "session_id": self.session_id,
            "actor": self.actor,
            "payload": dict(self.payload),
            "created_at_ms": self.created_at_ms,
        }


@dataclass(frozen=True)
class EvidenceTrace:
    trace_id: str
    session_id: str
    event_ids: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "event_ids": list(self.event_ids),
        }


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    artifact_kind: ArtifactKind
    ai_id: str = ""
    contract_id: str = ""
    source_ref: str = ""
    status: str = "created"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    promotion_decision: PromotionDecision = PromotionDecision.PENDING
    created_at_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind.value,
            "ai_id": self.ai_id,
            "contract_id": self.contract_id,
            "source_ref": self.source_ref,
            "status": self.status,
            "metadata": dict(self.metadata),
            "promotion_decision": self.promotion_decision.value,
            "created_at_ms": self.created_at_ms,
        }


@dataclass(frozen=True)
class EvalGateDecision:
    decision: PromotionDecision
    reasons: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class EvalRun:
    run_id: str
    gate_id: str
    ai_id: str = ""
    contract_id: str = ""
    status: EvalRunStatus = EvalRunStatus.COMPLETED
    score: float = 1.0
    decision: EvalGateDecision = field(
        default_factory=lambda: EvalGateDecision(PromotionDecision.ALLOW)
    )
    created_at_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "gate_id": self.gate_id,
            "ai_id": self.ai_id,
            "contract_id": self.contract_id,
            "status": self.status.value,
            "score": self.score,
            "decision": self.decision.to_json(),
            "created_at_ms": self.created_at_ms,
        }


@dataclass(frozen=True)
class WebhookSubscription:
    webhook_id: str
    target_url: str
    event_types: tuple[str, ...] = ()
    secret_ref: str = ""
    enabled: bool = True
    created_at_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "webhook_id": self.webhook_id,
            "target_url": self.target_url,
            "event_types": list(self.event_types),
            "secret_ref": self.secret_ref,
            "enabled": self.enabled,
            "created_at_ms": self.created_at_ms,
        }


@dataclass(frozen=True)
class EventStreamRecord:
    event_id: str
    event_type: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    created_at_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "payload": dict(self.payload),
            "created_at_ms": self.created_at_ms,
        }


@dataclass(frozen=True)
class UsageRecord:
    usage_id: str
    tenant_id: str = ""
    ai_id: str = ""
    metric: str = ""
    quantity: float = 0.0
    unit: str = "count"
    created_at_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "usage_id": self.usage_id,
            "tenant_id": self.tenant_id,
            "ai_id": self.ai_id,
            "metric": self.metric,
            "quantity": self.quantity,
            "unit": self.unit,
            "created_at_ms": self.created_at_ms,
        }


@dataclass(frozen=True)
class QuotaSnapshot:
    tenant_id: str
    limits: Mapping[str, Any] = field(default_factory=dict)
    usage: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "limits": dict(self.limits),
            "usage": dict(self.usage),
        }


@dataclass(frozen=True)
class BillingEvent:
    billing_event_id: str
    tenant_id: str = ""
    ai_id: str = ""
    amount: float = 0.0
    currency: str = "USD"
    reason: str = ""
    created_at_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "billing_event_id": self.billing_event_id,
            "tenant_id": self.tenant_id,
            "ai_id": self.ai_id,
            "amount": self.amount,
            "currency": self.currency,
            "reason": self.reason,
            "created_at_ms": self.created_at_ms,
        }


@dataclass(frozen=True)
class ConsentRecord:
    consent_id: str
    ai_id: str
    end_user_ref: str
    consent_type: str
    granted: bool
    evidence_ref: str = ""
    created_at_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "consent_id": self.consent_id,
            "ai_id": self.ai_id,
            "end_user_ref": self.end_user_ref,
            "consent_type": self.consent_type,
            "granted": self.granted,
            "evidence_ref": self.evidence_ref,
            "created_at_ms": self.created_at_ms,
        }


@dataclass(frozen=True)
class PolicySnapshot:
    policy_id: str
    policy_kind: str
    rules: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_kind": self.policy_kind,
            "rules": dict(self.rules),
        }


__all__ = [
    "ArtifactKind",
    "ArtifactRecord",
    "AuditEvent",
    "BillingEvent",
    "ConsentRecord",
    "DataExportJob",
    "DataJobStatus",
    "DeletionJob",
    "EvalGateDecision",
    "EvalRun",
    "EvalRunStatus",
    "EventStreamRecord",
    "EvidenceTrace",
    "PolicySnapshot",
    "PromotionDecision",
    "QuotaSnapshot",
    "UsageRecord",
    "WebhookSubscription",
]
