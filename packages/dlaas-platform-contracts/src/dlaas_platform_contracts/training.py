"""DLaaS protocol and training intake contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProtocolSubmissionSourceType(str, Enum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"
    JSON_PAYLOAD = "json_payload"


class TrainingJobType(str, Enum):
    PROTOCOL_EXTRACTION = "protocol_extraction"
    PROTOCOL_REVISION = "protocol_revision"
    CORPUS_INGESTION = "corpus_ingestion"
    ADAPTER_CANDIDATE = "adapter_candidate"
    EVAL_ONLY = "eval_only"


class TrainingJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PROMOTED = "promoted"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ProtocolSubmission:
    submission_id: str
    ai_id: str
    contract_id: str
    source_type: ProtocolSubmissionSourceType
    submitted_by: str = ""
    source_ref: str = ""
    target_vertical: str = ""
    notes: str = ""
    review_level_requested: str = "L3"
    candidate_protocol_id: str = ""
    review_status: str = "draft"

    @classmethod
    def from_json(
        cls,
        data: Mapping[str, Any],
        *,
        ai_id: str,
        submission_id: str,
    ) -> "ProtocolSubmission":
        if not isinstance(data, Mapping):
            raise ValueError("ProtocolSubmission payload must be a JSON object")
        contract_id = str(data.get("contract_id", "") or "")
        if not contract_id.strip():
            raise ValueError("ProtocolSubmission.contract_id must be non-empty")
        try:
            source_type = ProtocolSubmissionSourceType(
                str(data.get("source_type", "") or "")
            )
        except ValueError as exc:
            allowed = ", ".join(t.value for t in ProtocolSubmissionSourceType)
            raise ValueError(
                f"ProtocolSubmission.source_type must be one of: {allowed}"
            ) from exc
        return cls(
            submission_id=submission_id,
            ai_id=ai_id,
            contract_id=contract_id,
            source_type=source_type,
            submitted_by=str(data.get("submitted_by", "") or ""),
            source_ref=str(data.get("source_ref", "") or ""),
            target_vertical=str(data.get("target_vertical", "") or ""),
            notes=str(data.get("notes", "") or ""),
            review_level_requested=str(
                data.get("review_level_requested", "L3") or "L3"
            ),
            candidate_protocol_id=str(data.get("candidate_protocol_id", "") or ""),
            review_status=str(data.get("review_status", "draft") or "draft"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "ai_id": self.ai_id,
            "contract_id": self.contract_id,
            "source_type": self.source_type.value,
            "submitted_by": self.submitted_by,
            "source_ref": self.source_ref,
            "target_vertical": self.target_vertical,
            "notes": self.notes,
            "review_level_requested": self.review_level_requested,
            "candidate_protocol_id": self.candidate_protocol_id,
            "review_status": self.review_status,
        }


@dataclass(frozen=True)
class TrainingJob:
    job_id: str
    ai_id: str
    contract_id: str
    job_type: TrainingJobType
    status: TrainingJobStatus = TrainingJobStatus.PENDING
    created_by: str = ""
    source_ref: str = ""
    promotion_gate: str = ""
    artifact_ref: str = ""
    gate_evidence: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def from_json(
        cls,
        data: Mapping[str, Any],
        *,
        ai_id: str,
        job_id: str,
    ) -> "TrainingJob":
        if not isinstance(data, Mapping):
            raise ValueError("TrainingJob payload must be a JSON object")
        contract_id = str(data.get("contract_id", "") or "")
        if not contract_id.strip():
            raise ValueError("TrainingJob.contract_id must be non-empty")
        try:
            job_type = TrainingJobType(str(data.get("job_type", "") or ""))
        except ValueError as exc:
            allowed = ", ".join(t.value for t in TrainingJobType)
            raise ValueError(f"TrainingJob.job_type must be one of: {allowed}") from exc
        gate = data.get("gate_evidence") or {}
        if not isinstance(gate, Mapping):
            raise ValueError("TrainingJob.gate_evidence must be an object")
        return cls(
            job_id=job_id,
            ai_id=ai_id,
            contract_id=contract_id,
            job_type=job_type,
            created_by=str(data.get("created_by", "") or ""),
            source_ref=str(data.get("source_ref", "") or ""),
            promotion_gate=str(data.get("promotion_gate", "") or ""),
            artifact_ref=str(data.get("artifact_ref", "") or ""),
            gate_evidence=dict(gate),
            notes=str(data.get("notes", "") or ""),
        )

    def with_status(self, status: TrainingJobStatus) -> "TrainingJob":
        return TrainingJob(
            job_id=self.job_id,
            ai_id=self.ai_id,
            contract_id=self.contract_id,
            job_type=self.job_type,
            status=status,
            created_by=self.created_by,
            source_ref=self.source_ref,
            promotion_gate=self.promotion_gate,
            artifact_ref=self.artifact_ref,
            gate_evidence=dict(self.gate_evidence),
            notes=self.notes,
        )

    def with_artifact_ref(self, artifact_ref: str) -> "TrainingJob":
        return TrainingJob(
            job_id=self.job_id,
            ai_id=self.ai_id,
            contract_id=self.contract_id,
            job_type=self.job_type,
            status=self.status,
            created_by=self.created_by,
            source_ref=self.source_ref,
            promotion_gate=self.promotion_gate,
            artifact_ref=artifact_ref,
            gate_evidence=dict(self.gate_evidence),
            notes=self.notes,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "ai_id": self.ai_id,
            "contract_id": self.contract_id,
            "job_type": self.job_type.value,
            "status": self.status.value,
            "created_by": self.created_by,
            "source_ref": self.source_ref,
            "promotion_gate": self.promotion_gate,
            "artifact_ref": self.artifact_ref,
            "gate_evidence": dict(self.gate_evidence),
            "notes": self.notes,
        }


__all__ = [
    "ProtocolSubmission",
    "ProtocolSubmissionSourceType",
    "TrainingJob",
    "TrainingJobStatus",
    "TrainingJobType",
]
