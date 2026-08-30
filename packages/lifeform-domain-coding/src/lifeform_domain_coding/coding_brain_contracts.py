"""Frozen product contracts for the Coding Brain facade.

These objects define the narrow boundary between a coding agent host and the
Volvence coding lifeform. They deliberately carry typed task/outcome metadata:
free text is retrieval evidence, never a classifier for outcome kind or source.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from lifeform_core import (
    BoundedContentPolicyCredit,
    BoundedContentPolicyDecision,
    BoundedContentPolicyUpdateReceipt,
)
from volvence_zero.runtime import WiringLevel


CONTEXT_REQUEST_SCHEMA_VERSION = "coding-context-request.v1"
CONTEXT_PACK_SCHEMA_VERSION = "coding-context-pack.v1"
ADVICE_SCHEMA_VERSION = "coding-advice.v1"
OUTCOME_REPORT_SCHEMA_VERSION = "coding-outcome-report.v1"
OUTCOME_RECEIPT_SCHEMA_VERSION = "coding-outcome-receipt.v1"


class CodingTaskKind(str, Enum):
    FEATURE = "feature"
    BUGFIX = "bugfix"
    REFACTOR = "refactor"
    REVIEW = "review"
    TEST = "test"
    DOCS = "docs"
    RESEARCH = "research"
    MAINTENANCE = "maintenance"
    OTHER = "other"


class CodingOutcomeKind(str, Enum):
    TASK_VERIFIED = "task_verified"
    TASK_REGRESSED = "task_regressed"
    REVIEW_APPROVED = "review_approved"
    REVIEW_CHANGES_REQUESTED = "review_changes_requested"
    MERGED = "merged"
    REVERTED = "reverted"


class CodingOutcomeSource(str, Enum):
    TEST_SUITE = "test_suite"
    BUILD_GATE = "build_gate"
    CI = "ci"
    CODE_REVIEW = "code_review"
    VCS = "vcs"


class CodingOutcomeRoute(str, Enum):
    DIALOGUE_EXTERNAL_OUTCOME = "dialogue_external_outcome"
    EXECUTION_RESULT = "execution_result"


class CodingSettlementState(str, Enum):
    PENDING_NEXT_CONTEXT_TURN = "pending_next_context_turn"


_DETERMINISTIC_SOURCES = frozenset(
    {
        CodingOutcomeSource.TEST_SUITE,
        CodingOutcomeSource.BUILD_GATE,
        CodingOutcomeSource.CI,
    }
)
_REVIEW_KINDS = frozenset(
    {
        CodingOutcomeKind.REVIEW_APPROVED,
        CodingOutcomeKind.REVIEW_CHANGES_REQUESTED,
    }
)
_VCS_KINDS = frozenset({CodingOutcomeKind.MERGED, CodingOutcomeKind.REVERTED})


def _require_text(name: str, value: str, *, max_length: int) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > max_length:
        raise ValueError(f"{name} must be at most {max_length} characters")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} must not contain control characters")


def _require_optional_text(name: str, value: str, *, max_length: int) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if value:
        _require_text(name, value, max_length=max_length)


def _require_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_string_tuple(
    name: str,
    values: tuple[str, ...],
    *,
    max_items: int,
    item_max_length: int,
) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{name} must be a tuple")
    if len(values) > max_items:
        raise ValueError(f"{name} must contain at most {max_items} entries")
    for value in values:
        _require_text(name, value, max_length=item_max_length)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} entries must be unique")


def _closed_enum(enum_type: type[Enum], name: str, value: object) -> Any:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{name} must be one of: {allowed}") from exc


def _strict_payload(
    payload: Mapping[str, object],
    *,
    allowed: frozenset[str],
    required: frozenset[str],
) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a JSON object")
    unknown = set(payload) - allowed
    missing = required - set(payload)
    if unknown:
        raise ValueError(f"unknown fields: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"missing fields: {', '.join(sorted(missing))}")


def stable_content_sha256(payload: Mapping[str, object]) -> str:
    """Return the canonical content digest used by product lineage IDs."""

    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CodingContextRequest:
    request_id: str
    project_id: str
    repository_id: str
    task_id: str
    task_kind: CodingTaskKind
    task_summary: str
    repository_revision: str = ""
    target_paths: tuple[str, ...] = ()
    memory_limit: int = 8
    max_context_chars: int = 4_000

    def __post_init__(self) -> None:
        for name, value in (
            ("request_id", self.request_id),
            ("project_id", self.project_id),
            ("repository_id", self.repository_id),
            ("task_id", self.task_id),
        ):
            _require_text(name, value, max_length=256)
        if not isinstance(self.task_kind, CodingTaskKind):
            raise ValueError("task_kind must be a CodingTaskKind")
        _require_text("task_summary", self.task_summary, max_length=8_000)
        _require_optional_text(
            "repository_revision", self.repository_revision, max_length=512
        )
        _require_string_tuple(
            "target_paths",
            self.target_paths,
            max_items=128,
            item_max_length=1_024,
        )
        if isinstance(self.memory_limit, bool) or not 1 <= self.memory_limit <= 20:
            raise ValueError("memory_limit must be an integer from 1 to 20")
        if (
            isinstance(self.max_context_chars, bool)
            or not 256 <= self.max_context_chars <= 32_000
        ):
            raise ValueError(
                "max_context_chars must be an integer from 256 to 32000"
            )

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "CodingContextRequest":
        allowed = frozenset(
            {
                "schema_version",
                "request_id",
                "project_id",
                "repository_id",
                "task_id",
                "task_kind",
                "task_summary",
                "repository_revision",
                "target_paths",
                "memory_limit",
                "max_context_chars",
            }
        )
        required = frozenset(
            {
                "request_id",
                "project_id",
                "repository_id",
                "task_id",
                "task_kind",
                "task_summary",
            }
        )
        _strict_payload(payload, allowed=allowed, required=required)
        schema_version = payload.get("schema_version", CONTEXT_REQUEST_SCHEMA_VERSION)
        if schema_version != CONTEXT_REQUEST_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {CONTEXT_REQUEST_SCHEMA_VERSION!r}"
            )
        target_paths = payload.get("target_paths", ())
        if not isinstance(target_paths, (list, tuple)):
            raise ValueError("target_paths must be an array")
        return cls(
            request_id=payload["request_id"],
            project_id=payload["project_id"],
            repository_id=payload["repository_id"],
            task_id=payload["task_id"],
            task_kind=_closed_enum(
                CodingTaskKind, "task_kind", payload["task_kind"]
            ),
            task_summary=payload["task_summary"],
            repository_revision=payload.get("repository_revision", ""),
            target_paths=tuple(target_paths),
            memory_limit=payload.get("memory_limit", 8),
            max_context_chars=payload.get("max_context_chars", 4_000),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": CONTEXT_REQUEST_SCHEMA_VERSION,
            "request_id": self.request_id,
            "project_id": self.project_id,
            "repository_id": self.repository_id,
            "task_id": self.task_id,
            "task_kind": self.task_kind.value,
            "task_summary": self.task_summary,
            "repository_revision": self.repository_revision,
            "target_paths": list(self.target_paths),
            "memory_limit": self.memory_limit,
            "max_context_chars": self.max_context_chars,
        }


@dataclass(frozen=True)
class CodingOutcomeReport:
    outcome_id: str
    context_pack_id: str
    kind: CodingOutcomeKind
    source: CodingOutcomeSource
    summary: str
    detail: str
    observed_at_ms: int
    evidence_ref: str
    changed_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text("outcome_id", self.outcome_id, max_length=256)
        _require_text("context_pack_id", self.context_pack_id, max_length=256)
        if not self.context_pack_id.startswith("coding-context-pack:"):
            raise ValueError("context_pack_id must reference a Coding Context Pack")
        if not isinstance(self.kind, CodingOutcomeKind):
            raise ValueError("kind must be a CodingOutcomeKind")
        if not isinstance(self.source, CodingOutcomeSource):
            raise ValueError("source must be a CodingOutcomeSource")
        _require_text("summary", self.summary, max_length=2_000)
        _require_text("detail", self.detail, max_length=16_000)
        _require_non_negative_int("observed_at_ms", self.observed_at_ms)
        _require_text("evidence_ref", self.evidence_ref, max_length=2_048)
        _require_string_tuple(
            "changed_paths",
            self.changed_paths,
            max_items=256,
            item_max_length=1_024,
        )
        if self.kind in {
            CodingOutcomeKind.TASK_VERIFIED,
            CodingOutcomeKind.TASK_REGRESSED,
        }:
            if self.source not in _DETERMINISTIC_SOURCES:
                raise ValueError(
                    "task_verified/task_regressed require test_suite, build_gate, or ci"
                )
        elif self.kind in _REVIEW_KINDS:
            if self.source is not CodingOutcomeSource.CODE_REVIEW:
                raise ValueError("review outcomes require source=code_review")
        elif self.kind in _VCS_KINDS and self.source is not CodingOutcomeSource.VCS:
            raise ValueError("merge/revert outcomes require source=vcs")

    @property
    def deterministic_environment_outcome(self) -> bool:
        return self.kind in {
            CodingOutcomeKind.TASK_VERIFIED,
            CodingOutcomeKind.TASK_REGRESSED,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "CodingOutcomeReport":
        allowed = frozenset(
            {
                "schema_version",
                "outcome_id",
                "context_pack_id",
                "kind",
                "source",
                "summary",
                "detail",
                "observed_at_ms",
                "evidence_ref",
                "changed_paths",
            }
        )
        required = frozenset(
            {
                "outcome_id",
                "context_pack_id",
                "kind",
                "source",
                "summary",
                "detail",
                "observed_at_ms",
                "evidence_ref",
            }
        )
        _strict_payload(payload, allowed=allowed, required=required)
        schema_version = payload.get("schema_version", OUTCOME_REPORT_SCHEMA_VERSION)
        if schema_version != OUTCOME_REPORT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {OUTCOME_REPORT_SCHEMA_VERSION!r}"
            )
        changed_paths = payload.get("changed_paths", ())
        if not isinstance(changed_paths, (list, tuple)):
            raise ValueError("changed_paths must be an array")
        return cls(
            outcome_id=payload["outcome_id"],
            context_pack_id=payload["context_pack_id"],
            kind=_closed_enum(CodingOutcomeKind, "kind", payload["kind"]),
            source=_closed_enum(CodingOutcomeSource, "source", payload["source"]),
            summary=payload["summary"],
            detail=payload["detail"],
            observed_at_ms=payload["observed_at_ms"],
            evidence_ref=payload["evidence_ref"],
            changed_paths=tuple(changed_paths),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": OUTCOME_REPORT_SCHEMA_VERSION,
            "outcome_id": self.outcome_id,
            "context_pack_id": self.context_pack_id,
            "kind": self.kind.value,
            "source": self.source.value,
            "summary": self.summary,
            "detail": self.detail,
            "observed_at_ms": self.observed_at_ms,
            "evidence_ref": self.evidence_ref,
            "changed_paths": list(self.changed_paths),
        }


@dataclass(frozen=True)
class CodingAdviceSnapshot:
    advice_id: str
    source_turn_index: int
    candidate_regime_id: str
    candidate_abstract_action: str
    evidence_entry_ids: tuple[str, ...]
    rationale: str
    wiring_level: WiringLevel = WiringLevel.SHADOW
    applied: bool = False

    def __post_init__(self) -> None:
        _require_text("advice_id", self.advice_id, max_length=256)
        _require_non_negative_int("source_turn_index", self.source_turn_index)
        _require_optional_text(
            "candidate_regime_id", self.candidate_regime_id, max_length=512
        )
        _require_optional_text(
            "candidate_abstract_action",
            self.candidate_abstract_action,
            max_length=512,
        )
        _require_string_tuple(
            "evidence_entry_ids",
            self.evidence_entry_ids,
            max_items=20,
            item_max_length=256,
        )
        _require_text("rationale", self.rationale, max_length=2_000)
        if self.wiring_level is not WiringLevel.SHADOW:
            raise ValueError("Coding advice must remain WiringLevel.SHADOW")
        if self.applied:
            raise ValueError("SHADOW Coding advice cannot be applied")

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": ADVICE_SCHEMA_VERSION,
            "advice_id": self.advice_id,
            "source_turn_index": self.source_turn_index,
            "candidate_regime_id": self.candidate_regime_id,
            "candidate_abstract_action": self.candidate_abstract_action,
            "evidence_entry_ids": list(self.evidence_entry_ids),
            "rationale": self.rationale,
            "wiring_level": self.wiring_level.value,
            "applied": self.applied,
        }


@dataclass(frozen=True)
class CodingContextPackSnapshot:
    context_pack_id: str
    content_sha256: str
    request: CodingContextRequest
    generated_at_ms: int
    source_turn_index: int
    rendered_context: str
    source_entry_ids: tuple[str, ...]
    retrieval_facets: tuple[str, ...]
    memory_entry_count: int
    truncated: bool
    settled_outcome_evidence_refs: tuple[str, ...]
    pe_magnitude: float
    pe_bootstrap: bool
    advice: CodingAdviceSnapshot
    content_policy_decision: BoundedContentPolicyDecision | None = None
    settled_policy_credits: tuple[BoundedContentPolicyCredit, ...] = ()
    policy_updates: tuple[BoundedContentPolicyUpdateReceipt, ...] = ()
    content_policy_wiring_level: WiringLevel = WiringLevel.ACTIVE
    wiring_level: WiringLevel = WiringLevel.ACTIVE

    def __post_init__(self) -> None:
        _require_text("context_pack_id", self.context_pack_id, max_length=256)
        if not self.context_pack_id.startswith("coding-context-pack:"):
            raise ValueError("context_pack_id must use coding-context-pack lineage")
        if len(self.content_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.content_sha256
        ):
            raise ValueError("content_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.request, CodingContextRequest):
            raise ValueError("request must be a CodingContextRequest")
        _require_non_negative_int("generated_at_ms", self.generated_at_ms)
        _require_non_negative_int("source_turn_index", self.source_turn_index)
        if not isinstance(self.rendered_context, str):
            raise ValueError("rendered_context must be a string")
        _require_string_tuple(
            "source_entry_ids",
            self.source_entry_ids,
            max_items=20,
            item_max_length=256,
        )
        _require_string_tuple(
            "retrieval_facets",
            self.retrieval_facets,
            max_items=256,
            item_max_length=1_024,
        )
        _require_non_negative_int("memory_entry_count", self.memory_entry_count)
        if not isinstance(self.truncated, bool):
            raise ValueError("truncated must be a boolean")
        _require_string_tuple(
            "settled_outcome_evidence_refs",
            self.settled_outcome_evidence_refs,
            max_items=128,
            item_max_length=512,
        )
        if not isinstance(self.pe_magnitude, (int, float)) or isinstance(
            self.pe_magnitude, bool
        ):
            raise ValueError("pe_magnitude must be numeric")
        if not isinstance(self.pe_bootstrap, bool):
            raise ValueError("pe_bootstrap must be a boolean")
        if not isinstance(self.advice, CodingAdviceSnapshot):
            raise ValueError("advice must be a CodingAdviceSnapshot")
        if self.content_policy_decision is not None and not isinstance(
            self.content_policy_decision,
            BoundedContentPolicyDecision,
        ):
            raise ValueError(
                "content_policy_decision must be a BoundedContentPolicyDecision"
            )
        if (
            self.content_policy_decision is not None
            and self.source_entry_ids
            != self.content_policy_decision.output_entry_ids
        ):
            raise ValueError(
                "source_entry_ids must match content policy output order"
            )
        if any(
            not isinstance(item, BoundedContentPolicyCredit)
            for item in self.settled_policy_credits
        ):
            raise ValueError(
                "settled_policy_credits must contain BoundedContentPolicyCredit"
            )
        if any(
            not isinstance(item, BoundedContentPolicyUpdateReceipt)
            for item in self.policy_updates
        ):
            raise ValueError(
                "policy_updates must contain BoundedContentPolicyUpdateReceipt"
            )
        if len(self.settled_policy_credits) != len(self.policy_updates):
            raise ValueError("each settled content policy credit requires one update")
        if self.content_policy_wiring_level not in {
            WiringLevel.ACTIVE,
            WiringLevel.DISABLED,
        }:
            raise ValueError("Coding content policy must be ACTIVE or DISABLED")
        if self.content_policy_wiring_level is WiringLevel.DISABLED and any(
            (
                self.content_policy_decision is not None,
                self.settled_policy_credits,
                self.policy_updates,
            )
        ):
            raise ValueError("DISABLED content policy cannot publish policy lineage")
        if self.wiring_level is not WiringLevel.ACTIVE:
            raise ValueError("Coding Context Pack must be WiringLevel.ACTIVE")

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": CONTEXT_PACK_SCHEMA_VERSION,
            "context_pack_id": self.context_pack_id,
            "content_sha256": self.content_sha256,
            "request": self.request.to_json(),
            "generated_at_ms": self.generated_at_ms,
            "source_turn_index": self.source_turn_index,
            "rendered_context": self.rendered_context,
            "source_entry_ids": list(self.source_entry_ids),
            "retrieval_facets": list(self.retrieval_facets),
            "memory_entry_count": self.memory_entry_count,
            "truncated": self.truncated,
            "settled_outcome_evidence_refs": list(
                self.settled_outcome_evidence_refs
            ),
            "pe_magnitude": float(self.pe_magnitude),
            "pe_bootstrap": self.pe_bootstrap,
            "advice": self.advice.to_json(),
            "content_policy_decision": (
                self.content_policy_decision.to_json()
                if self.content_policy_decision is not None
                else None
            ),
            "settled_policy_credits": [
                item.to_json() for item in self.settled_policy_credits
            ],
            "policy_updates": [item.to_json() for item in self.policy_updates],
            "content_policy_wiring_level": (
                self.content_policy_wiring_level.value
            ),
            "wiring_level": self.wiring_level.value,
        }


@dataclass(frozen=True)
class CodingOutcomeReceipt:
    receipt_id: str
    content_sha256: str
    session_id: str
    project_id: str
    repository_id: str
    task_id: str
    report: CodingOutcomeReport
    action_turn_index: int
    memory_entry_id: str
    memory_persisted: bool
    task_event_ids: tuple[str, ...]
    external_outcome_evidence_id: str
    learning_route: CodingOutcomeRoute
    source_content_policy_decision_id: str = ""
    content_policy_action_applied: bool = False
    settlement_state: CodingSettlementState = (
        CodingSettlementState.PENDING_NEXT_CONTEXT_TURN
    )

    def __post_init__(self) -> None:
        for name, value in (
            ("receipt_id", self.receipt_id),
            ("session_id", self.session_id),
            ("project_id", self.project_id),
            ("repository_id", self.repository_id),
            ("task_id", self.task_id),
            ("memory_entry_id", self.memory_entry_id),
        ):
            _require_text(name, value, max_length=256)
        if len(self.content_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.content_sha256
        ):
            raise ValueError("content_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.report, CodingOutcomeReport):
            raise ValueError("report must be a CodingOutcomeReport")
        _require_non_negative_int("action_turn_index", self.action_turn_index)
        if not isinstance(self.memory_persisted, bool):
            raise ValueError("memory_persisted must be a boolean")
        _require_string_tuple(
            "task_event_ids",
            self.task_event_ids,
            max_items=16,
            item_max_length=512,
        )
        _require_optional_text(
            "external_outcome_evidence_id",
            self.external_outcome_evidence_id,
            max_length=512,
        )
        if not isinstance(self.learning_route, CodingOutcomeRoute):
            raise ValueError("learning_route must be a CodingOutcomeRoute")
        _require_optional_text(
            "source_content_policy_decision_id",
            self.source_content_policy_decision_id,
            max_length=256,
        )
        if not isinstance(self.content_policy_action_applied, bool):
            raise ValueError("content_policy_action_applied must be a boolean")
        if self.content_policy_action_applied != bool(
            self.source_content_policy_decision_id
        ):
            raise ValueError(
                "content policy application requires exact decision lineage"
            )
        if not isinstance(self.settlement_state, CodingSettlementState):
            raise ValueError("settlement_state must be a CodingSettlementState")
        if self.report.deterministic_environment_outcome:
            if self.learning_route is not CodingOutcomeRoute.DIALOGUE_EXTERNAL_OUTCOME:
                raise ValueError("deterministic outcomes must route to PE")
            if not self.external_outcome_evidence_id:
                raise ValueError(
                    "deterministic outcomes require external outcome evidence"
                )
        elif self.learning_route is not CodingOutcomeRoute.EXECUTION_RESULT:
            raise ValueError("review/VCS outcomes must route through execution_result")

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": OUTCOME_RECEIPT_SCHEMA_VERSION,
            "receipt_id": self.receipt_id,
            "content_sha256": self.content_sha256,
            "session_id": self.session_id,
            "project_id": self.project_id,
            "repository_id": self.repository_id,
            "task_id": self.task_id,
            "report": self.report.to_json(),
            "action_turn_index": self.action_turn_index,
            "memory_entry_id": self.memory_entry_id,
            "memory_persisted": self.memory_persisted,
            "task_event_ids": list(self.task_event_ids),
            "external_outcome_evidence_id": self.external_outcome_evidence_id,
            "learning_route": self.learning_route.value,
            "source_content_policy_decision_id": (
                self.source_content_policy_decision_id
            ),
            "content_policy_action_applied": self.content_policy_action_applied,
            "settlement_state": self.settlement_state.value,
        }


__all__ = (
    "ADVICE_SCHEMA_VERSION",
    "CONTEXT_PACK_SCHEMA_VERSION",
    "CONTEXT_REQUEST_SCHEMA_VERSION",
    "OUTCOME_RECEIPT_SCHEMA_VERSION",
    "OUTCOME_REPORT_SCHEMA_VERSION",
    "CodingAdviceSnapshot",
    "CodingContextPackSnapshot",
    "CodingContextRequest",
    "CodingOutcomeKind",
    "CodingOutcomeReceipt",
    "CodingOutcomeReport",
    "CodingOutcomeRoute",
    "CodingOutcomeSource",
    "CodingSettlementState",
    "CodingTaskKind",
    "stable_content_sha256",
)
