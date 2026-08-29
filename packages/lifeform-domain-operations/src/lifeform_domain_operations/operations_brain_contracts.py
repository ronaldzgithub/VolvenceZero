"""Strict frozen contracts for the AutoCompany-facing Operations Brain v1/v2.

AutoCompany supplies evidence class, operating scope, action-catalog identity,
and outcome verdict as explicit protocol fields. Free text is context only and
is never parsed to choose a route, upgrade evidence, or authorize work.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from volvence_zero.runtime import WiringLevel


CONTEXT_REQUEST_SCHEMA_VERSION = "operations-context-request.v1"
CONTEXT_PACK_SCHEMA_VERSION = "operations-context-pack.v1"
ADVICE_SCHEMA_VERSION = "operations-advice.v1"
OUTCOME_REPORT_SCHEMA_VERSION = "operations-outcome-report.v1"
OUTCOME_RECEIPT_SCHEMA_VERSION = "operations-outcome-receipt.v1"
EXPERIENCE_RECORD_SCHEMA_VERSION = "operations-experience-record.v1"
CONTEXT_REQUEST_V2_SCHEMA_VERSION = "operations-context-request.v2"
CONTEXT_PACK_V2_SCHEMA_VERSION = "operations-context-pack.v2"
ADVICE_V2_SCHEMA_VERSION = "operations-advice.v2"
OUTCOME_REPORT_V2_SCHEMA_VERSION = "operations-outcome-report.v2"
OUTCOME_RECEIPT_V2_SCHEMA_VERSION = "operations-outcome-receipt.v2"
OPERATIONS_STATE_SCHEMA_VERSION = "operations-state-snapshot.v1"
OPERATIONS_POLICY_CHECKPOINT_SCHEMA_VERSION = "operations-policy-checkpoint.v1"
OPERATIONS_POLICY_DECISION_SCHEMA_VERSION = "operations-policy-decision.v1"
OPERATIONS_POLICY_CREDIT_SCHEMA_VERSION = "operations-policy-credit.v1"
OPERATIONS_POLICY_UPDATE_SCHEMA_VERSION = "operations-policy-update.v1"
OPERATIONS_POLICY_FEATURE_ORDER = (
    "health_deficit",
    "goal_gap",
    "queue_pressure",
    "capacity_pressure",
    "deadline_pressure",
    "sla_pressure",
    "dependency_pressure",
    "incident_pressure",
    "budget_pressure",
    "recent_failure_pressure",
)


class OperationsDecisionPoint(str, Enum):
    CYCLE_PLANNING = "cycle_planning"
    WORK_PRIORITIZATION = "work_prioritization"
    CAPACITY_REBALANCE = "capacity_rebalance"
    DEPENDENCY_RESOLUTION = "dependency_resolution"
    INCIDENT_RECOVERY = "incident_recovery"
    OPERATING_REVIEW = "operating_review"


class OperationsFactKind(str, Enum):
    OKR_PROGRESS = "okr_progress"
    DIVISION_HEALTH = "division_health"
    CAPACITY = "capacity"
    DEPENDENCY = "dependency"
    WORK_ORDER_STATE = "work_order_state"
    INCIDENT = "incident"
    COST = "cost"
    HUMAN_LOAD = "human_load"
    RISK = "risk"
    OTHER = "other"


class OperationsConstraintKind(str, Enum):
    BUDGET = "budget"
    AUTHORITY = "authority"
    CAPACITY = "capacity"
    DEADLINE = "deadline"
    DEPENDENCY = "dependency"
    COMPLIANCE = "compliance"
    SAFETY = "safety"
    REVERSIBILITY = "reversibility"
    OTHER = "other"


class OperationsEvidenceClass(str, Enum):
    SIMULATION = "simulation"
    INTERNAL_REVIEW = "internal_review"
    MACHINE_CHECK = "machine_check"
    FIELD = "field"


class OperationsEvidenceRole(str, Enum):
    OPERATING_SIGNAL = "operating_signal"
    CONSTRAINT = "constraint"
    DECISION_RECORD = "decision_record"
    WORK_ORDER = "work_order"
    INTERNAL_REVIEW = "internal_review"
    MACHINE_AUDIT = "machine_audit"
    FIELD_OBSERVATION = "field_observation"
    OBJECTIVE_PROGRESS = "objective_progress"
    COST = "cost"
    INCIDENT = "incident"
    HUMAN_LOAD = "human_load"


class OperationsAdviceKind(str, Enum):
    PRIORITIZE_WORK = "prioritize_work"
    SEQUENCE_DEPENDENCY = "sequence_dependency"
    REBALANCE_CAPACITY = "rebalance_capacity"
    RECOVER_INCIDENT = "recover_incident"
    PAUSE_WORK = "pause_work"
    REQUEST_HUMAN = "request_human"


class OperationsWorkItemStatus(str, Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"
    CANCELLED = "cancelled"


class OperationsIncidentSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class OperationsPolicyMode(str, Enum):
    NOOP = "noop"
    FROZEN = "frozen"
    LEARNED = "learned"


class OperationsPolicyAction(str, Enum):
    NOOP = "noop"
    INTERVENE = "intervene"


class OperationsOutcomeKind(str, Enum):
    SIMULATION_RESULT = "simulation_result"
    INTERNAL_REVIEW_RESULT = "internal_review_result"
    MACHINE_CHECK_RESULT = "machine_check_result"
    WORK_ORDER_PROGRESS = "work_order_progress"
    OBJECTIVE_PROGRESS = "objective_progress"
    COST_RECORDED = "cost_recorded"
    INCIDENT_RECORDED = "incident_recorded"
    HUMAN_LOAD_RECORDED = "human_load_recorded"
    FIELD_OPERATION_RESULT = "field_operation_result"


class OperationsDecisionKind(str, Enum):
    ACCEPT = "accept"
    MODIFY = "modify"
    REJECT = "reject"
    DEFER = "defer"
    PAUSE = "pause"
    CANCEL = "cancel"
    NO_STATE_CHANGE = "no_state_change"


class OperationsOutcomeVerdict(str, Enum):
    FAVORABLE = "favorable"
    UNFAVORABLE = "unfavorable"
    MIXED = "mixed"
    INCONCLUSIVE = "inconclusive"


class OperationsObjectiveResult(str, Enum):
    NOT_OBSERVED = "not_observed"
    ADVANCED = "advanced"
    STALLED = "stalled"
    REGRESSED = "regressed"
    MIXED = "mixed"


class OperationsRiskLevel(str, Enum):
    UNASSESSED = "unassessed"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class OperationsReversibility(str, Enum):
    REVERSIBLE = "reversible"
    COSTLY = "costly"
    IRREVERSIBLE = "irreversible"


class OperationsOutcomeRoute(str, Enum):
    MEMORY_AND_EXECUTION_RESULT = "memory_and_execution_result"
    FIELD_OPERATION_PE_MEMORY_AND_EXECUTION_RESULT = (
        "field_operation_pe_memory_and_execution_result"
    )


class OperationsSettlementState(str, Enum):
    NOT_PE_ELIGIBLE = "not_pe_eligible"
    PENDING_NEXT_CONTEXT_TURN = "pending_next_context_turn"


_OUTCOME_CLASS_PAIRS: dict[OperationsEvidenceClass, frozenset[OperationsOutcomeKind]] = {
    OperationsEvidenceClass.SIMULATION: frozenset({OperationsOutcomeKind.SIMULATION_RESULT}),
    OperationsEvidenceClass.INTERNAL_REVIEW: frozenset({OperationsOutcomeKind.INTERNAL_REVIEW_RESULT}),
    OperationsEvidenceClass.MACHINE_CHECK: frozenset({OperationsOutcomeKind.MACHINE_CHECK_RESULT}),
    OperationsEvidenceClass.FIELD: frozenset(
        {
            OperationsOutcomeKind.WORK_ORDER_PROGRESS,
            OperationsOutcomeKind.OBJECTIVE_PROGRESS,
            OperationsOutcomeKind.COST_RECORDED,
            OperationsOutcomeKind.INCIDENT_RECORDED,
            OperationsOutcomeKind.HUMAN_LOAD_RECORDED,
            OperationsOutcomeKind.FIELD_OPERATION_RESULT,
        }
    ),
}

_SIMULATION_FORBIDDEN_ROLES = frozenset(
    {
        OperationsEvidenceRole.INTERNAL_REVIEW,
        OperationsEvidenceRole.MACHINE_AUDIT,
        OperationsEvidenceRole.FIELD_OBSERVATION,
        OperationsEvidenceRole.OBJECTIVE_PROGRESS,
        OperationsEvidenceRole.COST,
        OperationsEvidenceRole.INCIDENT,
        OperationsEvidenceRole.HUMAN_LOAD,
    }
)


def _require_text(name: str, value: object, *, max_length: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > max_length:
        raise ValueError(f"{name} must be at most {max_length} characters")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _require_optional_text(name: str, value: object, *, max_length: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if value:
        _require_text(name, value, max_length=max_length)
    return value


def _require_non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_signed_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_numeric(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _require_probability(name: str, value: object) -> float:
    numeric = _require_numeric(name, value)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return numeric


def _require_currency(value: object) -> str:
    currency = _require_text("currency", value, max_length=3)
    if len(currency) != 3 or not currency.isascii() or not currency.isalpha():
        raise ValueError("currency must be a three-letter ASCII code")
    if currency != currency.upper():
        raise ValueError("currency must be uppercase")
    return currency


def _require_sha256(name: str, value: object) -> str:
    digest = _require_text(name, value, max_length=64)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _require_content_id(name: str, value: object, *, prefix: str) -> str:
    content_id = _require_text(name, value, max_length=len(prefix) + 64)
    if not content_id.startswith(prefix):
        raise ValueError(f"{name} must start with {prefix!r}")
    _require_sha256(f"{name} digest", content_id[len(prefix) :])
    return content_id


def _require_string_tuple(
    name: str,
    values: object,
    *,
    max_items: int,
    item_max_length: int,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ValueError(f"{name} must be a tuple")
    if not allow_empty and not values:
        raise ValueError(f"{name} must not be empty")
    if len(values) > max_items:
        raise ValueError(f"{name} must contain at most {max_items} entries")
    for value in values:
        _require_text(name, value, max_length=item_max_length)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} entries must be unique")
    return values


def _require_typed_tuple(
    name: str,
    values: object,
    *,
    item_type: type,
    max_items: int,
    allow_empty: bool = True,
) -> tuple[Any, ...]:
    if not isinstance(values, tuple):
        raise ValueError(f"{name} must be a tuple")
    if not allow_empty and not values:
        raise ValueError(f"{name} must not be empty")
    if len(values) > max_items:
        raise ValueError(f"{name} must contain at most {max_items} entries")
    if any(not isinstance(value, item_type) for value in values):
        raise ValueError(f"{name} must contain only {item_type.__name__}")
    return values


def _require_unique_ids(name: str, ids: tuple[str, ...]) -> None:
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} ids must be unique")


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


def _mapping(name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _array(name: str, value: object) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be an array")
    return tuple(value)


_CANONICAL_DIGEST_DOMAIN = b"operations-canonical-value.v1\x00"
_MAX_SAFE_INTEGER = (1 << 53) - 1


def _canonical_length(value: int) -> bytes:
    if not 0 <= value <= 0xFFFF_FFFF:
        raise ValueError("canonical value length exceeds uint32")
    return value.to_bytes(4, byteorder="big", signed=False)


def _canonical_utf8(value: str) -> bytes:
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(
            "canonical strings must contain valid Unicode scalar values"
        ) from exc


def _canonical_value_bytes(value: object) -> bytes:
    """Encode JSON-shaped values without runtime-specific number spelling.

    JSON text is not a portable hash preimage: Python preserves ``0.0`` while
    JavaScript parses it as ``0``, and exponent formatting also differs. The
    Operations v1 digest therefore uses an explicit typed value encoding with
    IEEE-754 binary64 numbers, UTF-8 strings, length-prefixed collections, and
    UTF-8-byte-sorted object keys.
    """

    if value is None:
        return b"n"
    if isinstance(value, bool):
        return b"t" if value else b"f"
    if isinstance(value, str):
        encoded = _canonical_utf8(value)
        return b"s" + _canonical_length(len(encoded)) + encoded
    if isinstance(value, (int, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("canonical numbers must be finite")
        if numeric.is_integer() and abs(numeric) > _MAX_SAFE_INTEGER:
            raise ValueError("canonical integer exceeds IEEE-754 safe range")
        return b"d" + struct.pack(">d", numeric)
    if isinstance(value, (list, tuple)):
        return b"a" + _canonical_length(len(value)) + b"".join(
            _canonical_value_bytes(item) for item in value
        )
    if isinstance(value, Mapping):
        items: list[tuple[bytes, str, object]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical object keys must be strings")
            items.append((_canonical_utf8(key), key, item))
        items.sort(key=lambda entry: entry[0])
        return b"o" + _canonical_length(len(items)) + b"".join(
            _canonical_value_bytes(key) + _canonical_value_bytes(item)
            for _, key, item in items
        )
    raise TypeError(f"unsupported canonical value type: {type(value).__name__}")


def stable_content_sha256(payload: Mapping[str, object]) -> str:
    """Return the cross-runtime canonical digest for Operations lineage IDs."""

    return hashlib.sha256(
        _CANONICAL_DIGEST_DOMAIN + _canonical_value_bytes(payload)
    ).hexdigest()


@dataclass(frozen=True)
class OperationsEvidenceRef:
    ref_id: str
    evidence_class: OperationsEvidenceClass
    role: OperationsEvidenceRole
    locator: str
    content_sha256: str
    observed_at_ms: int

    def __post_init__(self) -> None:
        _require_text("ref_id", self.ref_id, max_length=256)
        if not isinstance(self.evidence_class, OperationsEvidenceClass):
            raise ValueError("evidence_class must be an OperationsEvidenceClass")
        if not isinstance(self.role, OperationsEvidenceRole):
            raise ValueError("role must be an OperationsEvidenceRole")
        _require_text("locator", self.locator, max_length=2_048)
        _require_sha256("content_sha256", self.content_sha256)
        _require_non_negative_int("observed_at_ms", self.observed_at_ms)
        if (
            self.evidence_class is OperationsEvidenceClass.INTERNAL_REVIEW
            and self.role is not OperationsEvidenceRole.INTERNAL_REVIEW
        ):
            raise ValueError("internal_review evidence requires role=internal_review")
        if (
            self.evidence_class is OperationsEvidenceClass.MACHINE_CHECK
            and self.role is not OperationsEvidenceRole.MACHINE_AUDIT
        ):
            raise ValueError("machine_check evidence requires role=machine_audit")
        if self.evidence_class is OperationsEvidenceClass.SIMULATION and self.role in _SIMULATION_FORBIDDEN_ROLES:
            raise ValueError("simulation evidence cannot use review, audit, or field-only roles")
        if self.evidence_class is OperationsEvidenceClass.FIELD and self.role in {
            OperationsEvidenceRole.INTERNAL_REVIEW,
            OperationsEvidenceRole.MACHINE_AUDIT,
        }:
            raise ValueError("field evidence cannot use internal review or machine audit roles")

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "OperationsEvidenceRef":
        fields = frozenset(
            {
                "ref_id",
                "evidence_class",
                "role",
                "locator",
                "content_sha256",
                "observed_at_ms",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        return cls(
            ref_id=payload["ref_id"],
            evidence_class=_closed_enum(
                OperationsEvidenceClass,
                "evidence_class",
                payload["evidence_class"],
            ),
            role=_closed_enum(OperationsEvidenceRole, "role", payload["role"]),
            locator=payload["locator"],
            content_sha256=payload["content_sha256"],
            observed_at_ms=payload["observed_at_ms"],
        )

    def to_json(self) -> dict[str, object]:
        return {
            "ref_id": self.ref_id,
            "evidence_class": self.evidence_class.value,
            "role": self.role.value,
            "locator": self.locator,
            "content_sha256": self.content_sha256,
            "observed_at_ms": self.observed_at_ms,
        }


@dataclass(frozen=True)
class OperationsFact:
    fact_id: str
    kind: OperationsFactKind
    division_id: str
    statement: str
    evidence_ref_ids: tuple[str, ...]
    as_of_ms: int

    def __post_init__(self) -> None:
        _require_text("fact_id", self.fact_id, max_length=256)
        if not isinstance(self.kind, OperationsFactKind):
            raise ValueError("kind must be an OperationsFactKind")
        _require_optional_text("division_id", self.division_id, max_length=256)
        _require_text("statement", self.statement, max_length=2_000)
        _require_string_tuple(
            "evidence_ref_ids",
            self.evidence_ref_ids,
            max_items=32,
            item_max_length=256,
            allow_empty=False,
        )
        _require_non_negative_int("as_of_ms", self.as_of_ms)

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "OperationsFact":
        fields = frozenset(
            {"fact_id", "kind", "division_id", "statement", "evidence_ref_ids", "as_of_ms"}
        )
        _strict_payload(payload, allowed=fields, required=fields)
        return cls(
            fact_id=payload["fact_id"],
            kind=_closed_enum(OperationsFactKind, "kind", payload["kind"]),
            division_id=payload["division_id"],
            statement=payload["statement"],
            evidence_ref_ids=tuple(_array("evidence_ref_ids", payload["evidence_ref_ids"])),
            as_of_ms=payload["as_of_ms"],
        )

    def to_json(self) -> dict[str, object]:
        return {
            "fact_id": self.fact_id,
            "kind": self.kind.value,
            "division_id": self.division_id,
            "statement": self.statement,
            "evidence_ref_ids": list(self.evidence_ref_ids),
            "as_of_ms": self.as_of_ms,
        }


@dataclass(frozen=True)
class OperationsConstraint:
    constraint_id: str
    kind: OperationsConstraintKind
    division_id: str
    description: str
    hard: bool

    def __post_init__(self) -> None:
        _require_text("constraint_id", self.constraint_id, max_length=256)
        if not isinstance(self.kind, OperationsConstraintKind):
            raise ValueError("kind must be an OperationsConstraintKind")
        _require_optional_text("division_id", self.division_id, max_length=256)
        _require_text("description", self.description, max_length=2_000)
        if not isinstance(self.hard, bool):
            raise ValueError("hard must be a boolean")

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "OperationsConstraint":
        fields = frozenset({"constraint_id", "kind", "division_id", "description", "hard"})
        _strict_payload(payload, allowed=fields, required=fields)
        return cls(
            constraint_id=payload["constraint_id"],
            kind=_closed_enum(OperationsConstraintKind, "kind", payload["kind"]),
            division_id=payload["division_id"],
            description=payload["description"],
            hard=payload["hard"],
        )

    def to_json(self) -> dict[str, object]:
        return {
            "constraint_id": self.constraint_id,
            "kind": self.kind.value,
            "division_id": self.division_id,
            "description": self.description,
            "hard": self.hard,
        }


@dataclass(frozen=True)
class OperationsUncertainty:
    uncertainty_id: str
    statement: str
    probability_lower: float
    probability_upper: float
    evidence_ref_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text("uncertainty_id", self.uncertainty_id, max_length=256)
        _require_text("statement", self.statement, max_length=2_000)
        lower = _require_probability("probability_lower", self.probability_lower)
        upper = _require_probability("probability_upper", self.probability_upper)
        if lower > upper:
            raise ValueError("probability_lower must be <= probability_upper")
        _require_string_tuple(
            "evidence_ref_ids",
            self.evidence_ref_ids,
            max_items=32,
            item_max_length=256,
        )

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "OperationsUncertainty":
        fields = frozenset(
            {
                "uncertainty_id",
                "statement",
                "probability_lower",
                "probability_upper",
                "evidence_ref_ids",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        return cls(
            uncertainty_id=payload["uncertainty_id"],
            statement=payload["statement"],
            probability_lower=payload["probability_lower"],
            probability_upper=payload["probability_upper"],
            evidence_ref_ids=tuple(_array("evidence_ref_ids", payload["evidence_ref_ids"])),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "uncertainty_id": self.uncertainty_id,
            "statement": self.statement,
            "probability_lower": float(self.probability_lower),
            "probability_upper": float(self.probability_upper),
            "evidence_ref_ids": list(self.evidence_ref_ids),
        }


@dataclass(frozen=True)
class OperationsOperatingWindow:
    currency: str
    maximum_external_cost_minor: int
    maximum_human_minutes: int
    starts_at_ms: int
    ends_at_ms: int
    maximum_work_orders: int

    def __post_init__(self) -> None:
        _require_currency(self.currency)
        _require_non_negative_int("maximum_external_cost_minor", self.maximum_external_cost_minor)
        _require_non_negative_int("maximum_human_minutes", self.maximum_human_minutes)
        start = _require_non_negative_int("starts_at_ms", self.starts_at_ms)
        end = _require_non_negative_int("ends_at_ms", self.ends_at_ms)
        if end <= start:
            raise ValueError("ends_at_ms must be greater than starts_at_ms")
        maximum = _require_non_negative_int("maximum_work_orders", self.maximum_work_orders)
        if maximum < 1:
            raise ValueError("maximum_work_orders must be positive")

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "OperationsOperatingWindow":
        fields = frozenset(
            {
                "currency",
                "maximum_external_cost_minor",
                "maximum_human_minutes",
                "starts_at_ms",
                "ends_at_ms",
                "maximum_work_orders",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        return cls(
            currency=payload["currency"],
            maximum_external_cost_minor=payload["maximum_external_cost_minor"],
            maximum_human_minutes=payload["maximum_human_minutes"],
            starts_at_ms=payload["starts_at_ms"],
            ends_at_ms=payload["ends_at_ms"],
            maximum_work_orders=payload["maximum_work_orders"],
        )

    def to_json(self) -> dict[str, object]:
        return {
            "currency": self.currency,
            "maximum_external_cost_minor": self.maximum_external_cost_minor,
            "maximum_human_minutes": self.maximum_human_minutes,
            "starts_at_ms": self.starts_at_ms,
            "ends_at_ms": self.ends_at_ms,
            "maximum_work_orders": self.maximum_work_orders,
        }


@dataclass(frozen=True)
class OperationsGoalState:
    goal_id: str
    division_id: str
    progress: float
    target_progress: float
    weight: float
    deadline_ms: int
    evidence_ref_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text("goal_id", self.goal_id, max_length=256)
        _require_text("division_id", self.division_id, max_length=256)
        _require_probability("progress", self.progress)
        _require_probability("target_progress", self.target_progress)
        weight = _require_numeric("weight", self.weight)
        if not 0.0 < weight <= 1.0:
            raise ValueError("weight must be in (0, 1]")
        _require_non_negative_int("deadline_ms", self.deadline_ms)
        _require_string_tuple(
            "evidence_ref_ids",
            self.evidence_ref_ids,
            max_items=32,
            item_max_length=256,
            allow_empty=False,
        )

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "OperationsGoalState":
        fields = frozenset(
            {
                "goal_id",
                "division_id",
                "progress",
                "target_progress",
                "weight",
                "deadline_ms",
                "evidence_ref_ids",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        return cls(
            goal_id=payload["goal_id"],
            division_id=payload["division_id"],
            progress=payload["progress"],
            target_progress=payload["target_progress"],
            weight=payload["weight"],
            deadline_ms=payload["deadline_ms"],
            evidence_ref_ids=tuple(
                _array("evidence_ref_ids", payload["evidence_ref_ids"])
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "goal_id": self.goal_id,
            "division_id": self.division_id,
            "progress": float(self.progress),
            "target_progress": float(self.target_progress),
            "weight": float(self.weight),
            "deadline_ms": self.deadline_ms,
            "evidence_ref_ids": list(self.evidence_ref_ids),
        }


@dataclass(frozen=True)
class OperationsWorkItemState:
    work_item_id: str
    division_id: str
    action_catalog_id: str
    status: OperationsWorkItemStatus
    progress: float
    priority: float
    deadline_ms: int
    required_human_minutes: int
    expected_cost_minor: int
    dependency_ids: tuple[str, ...]
    evidence_ref_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text("work_item_id", self.work_item_id, max_length=256)
        _require_text("division_id", self.division_id, max_length=256)
        _require_text("action_catalog_id", self.action_catalog_id, max_length=256)
        if not isinstance(self.status, OperationsWorkItemStatus):
            raise ValueError("status must be an OperationsWorkItemStatus")
        _require_probability("progress", self.progress)
        _require_probability("priority", self.priority)
        _require_non_negative_int("deadline_ms", self.deadline_ms)
        _require_non_negative_int(
            "required_human_minutes",
            self.required_human_minutes,
        )
        _require_non_negative_int("expected_cost_minor", self.expected_cost_minor)
        _require_string_tuple(
            "dependency_ids",
            self.dependency_ids,
            max_items=64,
            item_max_length=256,
        )
        _require_string_tuple(
            "evidence_ref_ids",
            self.evidence_ref_ids,
            max_items=32,
            item_max_length=256,
            allow_empty=False,
        )

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "OperationsWorkItemState":
        fields = frozenset(
            {
                "work_item_id",
                "division_id",
                "action_catalog_id",
                "status",
                "progress",
                "priority",
                "deadline_ms",
                "required_human_minutes",
                "expected_cost_minor",
                "dependency_ids",
                "evidence_ref_ids",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        return cls(
            work_item_id=payload["work_item_id"],
            division_id=payload["division_id"],
            action_catalog_id=payload["action_catalog_id"],
            status=_closed_enum(
                OperationsWorkItemStatus,
                "status",
                payload["status"],
            ),
            progress=payload["progress"],
            priority=payload["priority"],
            deadline_ms=payload["deadline_ms"],
            required_human_minutes=payload["required_human_minutes"],
            expected_cost_minor=payload["expected_cost_minor"],
            dependency_ids=tuple(
                _array("dependency_ids", payload["dependency_ids"])
            ),
            evidence_ref_ids=tuple(
                _array("evidence_ref_ids", payload["evidence_ref_ids"])
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "work_item_id": self.work_item_id,
            "division_id": self.division_id,
            "action_catalog_id": self.action_catalog_id,
            "status": self.status.value,
            "progress": float(self.progress),
            "priority": float(self.priority),
            "deadline_ms": self.deadline_ms,
            "required_human_minutes": self.required_human_minutes,
            "expected_cost_minor": self.expected_cost_minor,
            "dependency_ids": list(self.dependency_ids),
            "evidence_ref_ids": list(self.evidence_ref_ids),
        }


@dataclass(frozen=True)
class OperationsDependencyState:
    dependency_id: str
    predecessor_work_item_id: str
    successor_work_item_id: str
    resolved: bool
    criticality: float
    evidence_ref_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text("dependency_id", self.dependency_id, max_length=256)
        _require_text(
            "predecessor_work_item_id",
            self.predecessor_work_item_id,
            max_length=256,
        )
        _require_text(
            "successor_work_item_id",
            self.successor_work_item_id,
            max_length=256,
        )
        if self.predecessor_work_item_id == self.successor_work_item_id:
            raise ValueError("dependency endpoints must differ")
        if not isinstance(self.resolved, bool):
            raise ValueError("resolved must be a boolean")
        _require_probability("criticality", self.criticality)
        _require_string_tuple(
            "evidence_ref_ids",
            self.evidence_ref_ids,
            max_items=32,
            item_max_length=256,
            allow_empty=False,
        )

    @classmethod
    def from_json(
        cls,
        payload: Mapping[str, object],
    ) -> "OperationsDependencyState":
        fields = frozenset(
            {
                "dependency_id",
                "predecessor_work_item_id",
                "successor_work_item_id",
                "resolved",
                "criticality",
                "evidence_ref_ids",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        return cls(
            dependency_id=payload["dependency_id"],
            predecessor_work_item_id=payload["predecessor_work_item_id"],
            successor_work_item_id=payload["successor_work_item_id"],
            resolved=payload["resolved"],
            criticality=payload["criticality"],
            evidence_ref_ids=tuple(
                _array("evidence_ref_ids", payload["evidence_ref_ids"])
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "dependency_id": self.dependency_id,
            "predecessor_work_item_id": self.predecessor_work_item_id,
            "successor_work_item_id": self.successor_work_item_id,
            "resolved": self.resolved,
            "criticality": float(self.criticality),
            "evidence_ref_ids": list(self.evidence_ref_ids),
        }


@dataclass(frozen=True)
class OperationsIncidentState:
    incident_id: str
    division_id: str
    severity: OperationsIncidentSeverity
    open: bool
    started_at_ms: int
    sla_deadline_ms: int
    estimated_recovery_minutes: int
    evidence_ref_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text("incident_id", self.incident_id, max_length=256)
        _require_text("division_id", self.division_id, max_length=256)
        if not isinstance(self.severity, OperationsIncidentSeverity):
            raise ValueError("severity must be an OperationsIncidentSeverity")
        if not isinstance(self.open, bool):
            raise ValueError("open must be a boolean")
        started = _require_non_negative_int("started_at_ms", self.started_at_ms)
        deadline = _require_non_negative_int(
            "sla_deadline_ms",
            self.sla_deadline_ms,
        )
        if deadline < started:
            raise ValueError("sla_deadline_ms must be >= started_at_ms")
        _require_non_negative_int(
            "estimated_recovery_minutes",
            self.estimated_recovery_minutes,
        )
        _require_string_tuple(
            "evidence_ref_ids",
            self.evidence_ref_ids,
            max_items=32,
            item_max_length=256,
            allow_empty=False,
        )

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "OperationsIncidentState":
        fields = frozenset(
            {
                "incident_id",
                "division_id",
                "severity",
                "open",
                "started_at_ms",
                "sla_deadline_ms",
                "estimated_recovery_minutes",
                "evidence_ref_ids",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        return cls(
            incident_id=payload["incident_id"],
            division_id=payload["division_id"],
            severity=_closed_enum(
                OperationsIncidentSeverity,
                "severity",
                payload["severity"],
            ),
            open=payload["open"],
            started_at_ms=payload["started_at_ms"],
            sla_deadline_ms=payload["sla_deadline_ms"],
            estimated_recovery_minutes=payload["estimated_recovery_minutes"],
            evidence_ref_ids=tuple(
                _array("evidence_ref_ids", payload["evidence_ref_ids"])
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "incident_id": self.incident_id,
            "division_id": self.division_id,
            "severity": self.severity.value,
            "open": self.open,
            "started_at_ms": self.started_at_ms,
            "sla_deadline_ms": self.sla_deadline_ms,
            "estimated_recovery_minutes": self.estimated_recovery_minutes,
            "evidence_ref_ids": list(self.evidence_ref_ids),
        }


@dataclass(frozen=True)
class OperationsRecentOutcomeState:
    outcome_id: str
    division_id: str
    candidate_id: str
    utility: float
    observed_at_ms: int
    evidence_ref_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text("outcome_id", self.outcome_id, max_length=256)
        _require_text("division_id", self.division_id, max_length=256)
        _require_optional_text("candidate_id", self.candidate_id, max_length=256)
        utility = _require_numeric("utility", self.utility)
        if not -1.0 <= utility <= 1.0:
            raise ValueError("utility must be in [-1, 1]")
        _require_non_negative_int("observed_at_ms", self.observed_at_ms)
        _require_string_tuple(
            "evidence_ref_ids",
            self.evidence_ref_ids,
            max_items=32,
            item_max_length=256,
            allow_empty=False,
        )

    @classmethod
    def from_json(
        cls,
        payload: Mapping[str, object],
    ) -> "OperationsRecentOutcomeState":
        fields = frozenset(
            {
                "outcome_id",
                "division_id",
                "candidate_id",
                "utility",
                "observed_at_ms",
                "evidence_ref_ids",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        return cls(
            outcome_id=payload["outcome_id"],
            division_id=payload["division_id"],
            candidate_id=payload["candidate_id"],
            utility=payload["utility"],
            observed_at_ms=payload["observed_at_ms"],
            evidence_ref_ids=tuple(
                _array("evidence_ref_ids", payload["evidence_ref_ids"])
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "outcome_id": self.outcome_id,
            "division_id": self.division_id,
            "candidate_id": self.candidate_id,
            "utility": float(self.utility),
            "observed_at_ms": self.observed_at_ms,
            "evidence_ref_ids": list(self.evidence_ref_ids),
        }


@dataclass(frozen=True)
class OperationsDivisionState:
    division_id: str
    health: float
    available_human_minutes: int
    committed_human_minutes: int
    queue_depth: int
    sla_breach_probability: float
    budget_remaining_minor: int
    cost_to_date_minor: int
    evidence_ref_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text("division_id", self.division_id, max_length=256)
        _require_probability("health", self.health)
        _require_non_negative_int(
            "available_human_minutes",
            self.available_human_minutes,
        )
        _require_non_negative_int(
            "committed_human_minutes",
            self.committed_human_minutes,
        )
        _require_non_negative_int("queue_depth", self.queue_depth)
        _require_probability(
            "sla_breach_probability",
            self.sla_breach_probability,
        )
        _require_non_negative_int(
            "budget_remaining_minor",
            self.budget_remaining_minor,
        )
        _require_non_negative_int("cost_to_date_minor", self.cost_to_date_minor)
        _require_string_tuple(
            "evidence_ref_ids",
            self.evidence_ref_ids,
            max_items=64,
            item_max_length=256,
            allow_empty=False,
        )

    @classmethod
    def from_json(
        cls,
        payload: Mapping[str, object],
    ) -> "OperationsDivisionState":
        fields = frozenset(
            {
                "division_id",
                "health",
                "available_human_minutes",
                "committed_human_minutes",
                "queue_depth",
                "sla_breach_probability",
                "budget_remaining_minor",
                "cost_to_date_minor",
                "evidence_ref_ids",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        return cls(
            division_id=payload["division_id"],
            health=payload["health"],
            available_human_minutes=payload["available_human_minutes"],
            committed_human_minutes=payload["committed_human_minutes"],
            queue_depth=payload["queue_depth"],
            sla_breach_probability=payload["sla_breach_probability"],
            budget_remaining_minor=payload["budget_remaining_minor"],
            cost_to_date_minor=payload["cost_to_date_minor"],
            evidence_ref_ids=tuple(
                _array("evidence_ref_ids", payload["evidence_ref_ids"])
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "division_id": self.division_id,
            "health": float(self.health),
            "available_human_minutes": self.available_human_minutes,
            "committed_human_minutes": self.committed_human_minutes,
            "queue_depth": self.queue_depth,
            "sla_breach_probability": float(self.sla_breach_probability),
            "budget_remaining_minor": self.budget_remaining_minor,
            "cost_to_date_minor": self.cost_to_date_minor,
            "evidence_ref_ids": list(self.evidence_ref_ids),
        }


@dataclass(frozen=True)
class OperationsStateSnapshot:
    state_snapshot_id: str
    content_sha256: str
    as_of_ms: int
    currency: str
    divisions: tuple[OperationsDivisionState, ...]
    goals: tuple[OperationsGoalState, ...]
    work_items: tuple[OperationsWorkItemState, ...]
    dependencies: tuple[OperationsDependencyState, ...]
    incidents: tuple[OperationsIncidentState, ...]
    recent_outcomes: tuple[OperationsRecentOutcomeState, ...]

    def __post_init__(self) -> None:
        _require_content_id(
            "state_snapshot_id",
            self.state_snapshot_id,
            prefix="operations-state:",
        )
        _require_sha256("content_sha256", self.content_sha256)
        if self.state_snapshot_id != f"operations-state:{self.content_sha256}":
            raise ValueError("state_snapshot_id must match content_sha256")
        _require_non_negative_int("as_of_ms", self.as_of_ms)
        _require_currency(self.currency)
        for name, values, item_type, maximum, allow_empty in (
            ("divisions", self.divisions, OperationsDivisionState, 64, False),
            ("goals", self.goals, OperationsGoalState, 256, True),
            ("work_items", self.work_items, OperationsWorkItemState, 512, True),
            (
                "dependencies",
                self.dependencies,
                OperationsDependencyState,
                1_024,
                True,
            ),
            ("incidents", self.incidents, OperationsIncidentState, 256, True),
            (
                "recent_outcomes",
                self.recent_outcomes,
                OperationsRecentOutcomeState,
                256,
                True,
            ),
        ):
            _require_typed_tuple(
                name,
                values,
                item_type=item_type,
                max_items=maximum,
                allow_empty=allow_empty,
            )
        _require_unique_ids(
            "divisions",
            tuple(item.division_id for item in self.divisions),
        )
        _require_unique_ids("goals", tuple(item.goal_id for item in self.goals))
        _require_unique_ids(
            "work_items",
            tuple(item.work_item_id for item in self.work_items),
        )
        _require_unique_ids(
            "dependencies",
            tuple(item.dependency_id for item in self.dependencies),
        )
        _require_unique_ids(
            "incidents",
            tuple(item.incident_id for item in self.incidents),
        )
        _require_unique_ids(
            "recent_outcomes",
            tuple(item.outcome_id for item in self.recent_outcomes),
        )
        divisions = {item.division_id for item in self.divisions}
        scoped = {
            item.division_id
            for item in (
                *self.goals,
                *self.work_items,
                *self.incidents,
                *self.recent_outcomes,
            )
        }
        if scoped - divisions:
            raise ValueError("state children reference unknown division ids")
        work_item_ids = {item.work_item_id for item in self.work_items}
        dependency_ids = {item.dependency_id for item in self.dependencies}
        for item in self.dependencies:
            if (
                item.predecessor_work_item_id not in work_item_ids
                or item.successor_work_item_id not in work_item_ids
            ):
                raise ValueError("dependency edge references unknown work item ids")
        for item in self.work_items:
            if set(item.dependency_ids) - dependency_ids:
                raise ValueError("work item references unknown dependency ids")

    @classmethod
    def create(
        cls,
        *,
        as_of_ms: int,
        currency: str,
        divisions: tuple[OperationsDivisionState, ...],
        goals: tuple[OperationsGoalState, ...] = (),
        work_items: tuple[OperationsWorkItemState, ...] = (),
        dependencies: tuple[OperationsDependencyState, ...] = (),
        incidents: tuple[OperationsIncidentState, ...] = (),
        recent_outcomes: tuple[OperationsRecentOutcomeState, ...] = (),
    ) -> "OperationsStateSnapshot":
        core = {
            "schema_version": OPERATIONS_STATE_SCHEMA_VERSION,
            "as_of_ms": as_of_ms,
            "currency": currency,
            "divisions": [item.to_json() for item in divisions],
            "goals": [item.to_json() for item in goals],
            "work_items": [item.to_json() for item in work_items],
            "dependencies": [item.to_json() for item in dependencies],
            "incidents": [item.to_json() for item in incidents],
            "recent_outcomes": [item.to_json() for item in recent_outcomes],
        }
        digest = stable_content_sha256(core)
        return cls(
            state_snapshot_id=f"operations-state:{digest}",
            content_sha256=digest,
            as_of_ms=as_of_ms,
            currency=currency,
            divisions=divisions,
            goals=goals,
            work_items=work_items,
            dependencies=dependencies,
            incidents=incidents,
            recent_outcomes=recent_outcomes,
        )

    @classmethod
    def from_json(
        cls,
        payload: Mapping[str, object],
    ) -> "OperationsStateSnapshot":
        fields = frozenset(
            {
                "schema_version",
                "state_snapshot_id",
                "content_sha256",
                "as_of_ms",
                "currency",
                "divisions",
                "goals",
                "work_items",
                "dependencies",
                "incidents",
                "recent_outcomes",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        if payload["schema_version"] != OPERATIONS_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {OPERATIONS_STATE_SCHEMA_VERSION!r}"
            )
        snapshot = cls(
            state_snapshot_id=payload["state_snapshot_id"],
            content_sha256=payload["content_sha256"],
            as_of_ms=payload["as_of_ms"],
            currency=payload["currency"],
            divisions=tuple(
                OperationsDivisionState.from_json(_mapping("divisions[]", item))
                for item in _array("divisions", payload["divisions"])
            ),
            goals=tuple(
                OperationsGoalState.from_json(_mapping("goals[]", item))
                for item in _array("goals", payload["goals"])
            ),
            work_items=tuple(
                OperationsWorkItemState.from_json(_mapping("work_items[]", item))
                for item in _array("work_items", payload["work_items"])
            ),
            dependencies=tuple(
                OperationsDependencyState.from_json(
                    _mapping("dependencies[]", item)
                )
                for item in _array("dependencies", payload["dependencies"])
            ),
            incidents=tuple(
                OperationsIncidentState.from_json(_mapping("incidents[]", item))
                for item in _array("incidents", payload["incidents"])
            ),
            recent_outcomes=tuple(
                OperationsRecentOutcomeState.from_json(
                    _mapping("recent_outcomes[]", item)
                )
                for item in _array("recent_outcomes", payload["recent_outcomes"])
            ),
        )
        core = snapshot.to_json()
        core.pop("state_snapshot_id")
        core.pop("content_sha256")
        if stable_content_sha256(core) != snapshot.content_sha256:
            raise ValueError("Operations state content_sha256 does not match payload")
        return snapshot

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": OPERATIONS_STATE_SCHEMA_VERSION,
            "state_snapshot_id": self.state_snapshot_id,
            "content_sha256": self.content_sha256,
            "as_of_ms": self.as_of_ms,
            "currency": self.currency,
            "divisions": [item.to_json() for item in self.divisions],
            "goals": [item.to_json() for item in self.goals],
            "work_items": [item.to_json() for item in self.work_items],
            "dependencies": [item.to_json() for item in self.dependencies],
            "incidents": [item.to_json() for item in self.incidents],
            "recent_outcomes": [item.to_json() for item in self.recent_outcomes],
        }


@dataclass(frozen=True)
class OperationsContextRequest:
    request_id: str
    company_id: str
    cycle_id: str
    workstream_id: str
    decision_id: str
    decision_point: OperationsDecisionPoint
    division_ids: tuple[str, ...]
    action_catalog_ids: tuple[str, ...]
    confirmed_facts: tuple[OperationsFact, ...]
    constraints: tuple[OperationsConstraint, ...]
    operating_window: OperationsOperatingWindow
    uncertainties: tuple[OperationsUncertainty, ...]
    evidence_refs: tuple[OperationsEvidenceRef, ...]
    memory_limit: int = 8
    max_context_chars: int = 8_000
    operations_state: OperationsStateSnapshot | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("request_id", self.request_id),
            ("company_id", self.company_id),
            ("cycle_id", self.cycle_id),
            ("decision_id", self.decision_id),
        ):
            _require_text(name, value, max_length=256)
        _require_optional_text("workstream_id", self.workstream_id, max_length=256)
        if not isinstance(self.decision_point, OperationsDecisionPoint):
            raise ValueError("decision_point must be an OperationsDecisionPoint")
        _require_string_tuple(
            "division_ids",
            self.division_ids,
            max_items=64,
            item_max_length=256,
            allow_empty=False,
        )
        _require_string_tuple(
            "action_catalog_ids",
            self.action_catalog_ids,
            max_items=128,
            item_max_length=256,
            allow_empty=False,
        )
        _require_typed_tuple(
            "confirmed_facts",
            self.confirmed_facts,
            item_type=OperationsFact,
            max_items=64,
            allow_empty=False,
        )
        _require_typed_tuple(
            "constraints",
            self.constraints,
            item_type=OperationsConstraint,
            max_items=32,
            allow_empty=False,
        )
        if not isinstance(self.operating_window, OperationsOperatingWindow):
            raise ValueError("operating_window must be an OperationsOperatingWindow")
        _require_typed_tuple(
            "uncertainties",
            self.uncertainties,
            item_type=OperationsUncertainty,
            max_items=32,
        )
        _require_typed_tuple(
            "evidence_refs",
            self.evidence_refs,
            item_type=OperationsEvidenceRef,
            max_items=128,
            allow_empty=False,
        )
        _require_unique_ids("confirmed_facts", tuple(item.fact_id for item in self.confirmed_facts))
        _require_unique_ids("constraints", tuple(item.constraint_id for item in self.constraints))
        _require_unique_ids("uncertainties", tuple(item.uncertainty_id for item in self.uncertainties))
        _require_unique_ids("evidence_refs", tuple(item.ref_id for item in self.evidence_refs))
        known_refs = {reference.ref_id for reference in self.evidence_refs}
        used_refs = {reference_id for fact in self.confirmed_facts for reference_id in fact.evidence_ref_ids}
        used_refs.update(
            reference_id for uncertainty in self.uncertainties for reference_id in uncertainty.evidence_ref_ids
        )
        unknown_refs = used_refs - known_refs
        if unknown_refs:
            raise ValueError("facts/uncertainties reference unknown evidence ids: " + ", ".join(sorted(unknown_refs)))
        known_divisions = set(self.division_ids)
        scoped_divisions = {
            item.division_id
            for item in (*self.confirmed_facts, *self.constraints)
            if item.division_id
        }
        unknown_divisions = scoped_divisions - known_divisions
        if unknown_divisions:
            raise ValueError(
                "facts/constraints reference unknown division ids: "
                + ", ".join(sorted(unknown_divisions))
            )
        if (
            isinstance(self.memory_limit, bool)
            or not isinstance(self.memory_limit, int)
            or not 1 <= self.memory_limit <= 20
        ):
            raise ValueError("memory_limit must be an integer from 1 to 20")
        if (
            isinstance(self.max_context_chars, bool)
            or not isinstance(self.max_context_chars, int)
            or not 512 <= self.max_context_chars <= 32_000
        ):
            raise ValueError("max_context_chars must be an integer from 512 to 32000")
        if self.operations_state is not None:
            if not isinstance(self.operations_state, OperationsStateSnapshot):
                raise ValueError(
                    "operations_state must be an OperationsStateSnapshot"
                )
            if self.operations_state.currency != self.operating_window.currency:
                raise ValueError(
                    "operations_state currency must match operating_window"
                )
            state_divisions = {
                item.division_id for item in self.operations_state.divisions
            }
            if state_divisions != known_divisions:
                raise ValueError(
                    "operations_state divisions must exactly match division_ids"
                )
            state_catalog_ids = {
                item.action_catalog_id
                for item in self.operations_state.work_items
            }
            if state_catalog_ids - set(self.action_catalog_ids):
                raise ValueError(
                    "operations_state references unknown action catalog ids"
                )
            state_refs = {
                reference_id
                for collection in (
                    self.operations_state.divisions,
                    self.operations_state.goals,
                    self.operations_state.work_items,
                    self.operations_state.dependencies,
                    self.operations_state.incidents,
                    self.operations_state.recent_outcomes,
                )
                for item in collection
                for reference_id in item.evidence_ref_ids
            }
            if state_refs - known_refs:
                raise ValueError(
                    "operations_state references unknown evidence ids"
                )

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "OperationsContextRequest":
        allowed = frozenset(
            {
                "schema_version",
                "request_id",
                "company_id",
                "cycle_id",
                "workstream_id",
                "decision_id",
                "decision_point",
                "division_ids",
                "action_catalog_ids",
                "confirmed_facts",
                "constraints",
                "operating_window",
                "uncertainties",
                "evidence_refs",
                "memory_limit",
                "max_context_chars",
                "operations_state",
            }
        )
        required = frozenset(
            {
                "schema_version",
                "request_id",
                "company_id",
                "cycle_id",
                "workstream_id",
                "decision_id",
                "decision_point",
                "division_ids",
                "action_catalog_ids",
                "confirmed_facts",
                "constraints",
                "operating_window",
                "uncertainties",
                "evidence_refs",
            }
        )
        _strict_payload(payload, allowed=allowed, required=required)
        schema_version = payload["schema_version"]
        if schema_version not in {
            CONTEXT_REQUEST_SCHEMA_VERSION,
            CONTEXT_REQUEST_V2_SCHEMA_VERSION,
        }:
            raise ValueError(
                "schema_version must be a supported Operations Context Request version"
            )
        if schema_version == CONTEXT_REQUEST_SCHEMA_VERSION and "operations_state" in payload:
            raise ValueError("operations_state requires operations-context-request.v2")
        if schema_version == CONTEXT_REQUEST_V2_SCHEMA_VERSION and "operations_state" not in payload:
            raise ValueError(
                "schema_version operations-context-request.v2 requires operations_state"
            )
        facts = tuple(
            OperationsFact.from_json(_mapping("confirmed_facts[]", item))
            for item in _array("confirmed_facts", payload["confirmed_facts"])
        )
        constraints = tuple(
            OperationsConstraint.from_json(_mapping("constraints[]", item))
            for item in _array("constraints", payload["constraints"])
        )
        uncertainties = tuple(
            OperationsUncertainty.from_json(_mapping("uncertainties[]", item))
            for item in _array("uncertainties", payload["uncertainties"])
        )
        evidence_refs = tuple(
            OperationsEvidenceRef.from_json(_mapping("evidence_refs[]", item))
            for item in _array("evidence_refs", payload["evidence_refs"])
        )
        return cls(
            request_id=payload["request_id"],
            company_id=payload["company_id"],
            cycle_id=payload["cycle_id"],
            workstream_id=payload["workstream_id"],
            decision_id=payload["decision_id"],
            decision_point=_closed_enum(
                OperationsDecisionPoint,
                "decision_point",
                payload["decision_point"],
            ),
            division_ids=tuple(_array("division_ids", payload["division_ids"])),
            action_catalog_ids=tuple(
                _array("action_catalog_ids", payload["action_catalog_ids"])
            ),
            confirmed_facts=facts,
            constraints=constraints,
            operating_window=OperationsOperatingWindow.from_json(
                _mapping("operating_window", payload["operating_window"])
            ),
            uncertainties=uncertainties,
            evidence_refs=evidence_refs,
            memory_limit=payload.get("memory_limit", 8),
            max_context_chars=payload.get("max_context_chars", 8_000),
            operations_state=(
                OperationsStateSnapshot.from_json(
                    _mapping("operations_state", payload["operations_state"])
                )
                if "operations_state" in payload
                else None
            ),
        )

    def to_json(self) -> dict[str, object]:
        payload = {
            "schema_version": (
                CONTEXT_REQUEST_V2_SCHEMA_VERSION
                if self.operations_state is not None
                else CONTEXT_REQUEST_SCHEMA_VERSION
            ),
            "request_id": self.request_id,
            "company_id": self.company_id,
            "cycle_id": self.cycle_id,
            "workstream_id": self.workstream_id,
            "decision_id": self.decision_id,
            "decision_point": self.decision_point.value,
            "division_ids": list(self.division_ids),
            "action_catalog_ids": list(self.action_catalog_ids),
            "confirmed_facts": [fact.to_json() for fact in self.confirmed_facts],
            "constraints": [constraint.to_json() for constraint in self.constraints],
            "operating_window": self.operating_window.to_json(),
            "uncertainties": [item.to_json() for item in self.uncertainties],
            "evidence_refs": [reference.to_json() for reference in self.evidence_refs],
            "memory_limit": self.memory_limit,
            "max_context_chars": self.max_context_chars,
        }
        if self.operations_state is not None:
            payload["operations_state"] = self.operations_state.to_json()
        return payload


@dataclass(frozen=True)
class OperationsEstimateRange:
    metric: str
    lower_bound: float
    upper_bound: float
    unit: str
    horizon_start_ms: int
    horizon_end_ms: int
    evidence_ref_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text("metric", self.metric, max_length=256)
        lower = _require_numeric("lower_bound", self.lower_bound)
        upper = _require_numeric("upper_bound", self.upper_bound)
        if lower > upper:
            raise ValueError("lower_bound must be <= upper_bound")
        _require_text("unit", self.unit, max_length=128)
        start = _require_non_negative_int("horizon_start_ms", self.horizon_start_ms)
        end = _require_non_negative_int("horizon_end_ms", self.horizon_end_ms)
        if end <= start:
            raise ValueError("horizon_end_ms must be greater than horizon_start_ms")
        _require_string_tuple(
            "evidence_ref_ids",
            self.evidence_ref_ids,
            max_items=32,
            item_max_length=256,
        )

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "OperationsEstimateRange":
        fields = frozenset(
            {
                "metric",
                "lower_bound",
                "upper_bound",
                "unit",
                "horizon_start_ms",
                "horizon_end_ms",
                "evidence_ref_ids",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        return cls(
            metric=payload["metric"],
            lower_bound=payload["lower_bound"],
            upper_bound=payload["upper_bound"],
            unit=payload["unit"],
            horizon_start_ms=payload["horizon_start_ms"],
            horizon_end_ms=payload["horizon_end_ms"],
            evidence_ref_ids=tuple(_array("evidence_ref_ids", payload["evidence_ref_ids"])),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "lower_bound": float(self.lower_bound),
            "upper_bound": float(self.upper_bound),
            "unit": self.unit,
            "horizon_start_ms": self.horizon_start_ms,
            "horizon_end_ms": self.horizon_end_ms,
            "evidence_ref_ids": list(self.evidence_ref_ids),
        }


@dataclass(frozen=True)
class OperationsAdviceCandidate:
    candidate_id: str
    kind: OperationsAdviceKind
    target_division_id: str
    action_catalog_id: str
    summary: str
    rationale: str
    maximum_cost_minor: int
    maximum_human_minutes: int
    requires_human_approval: bool
    risk_level: OperationsRiskLevel
    reversibility: OperationsReversibility
    prerequisite_fact_ids: tuple[str, ...]
    prediction_ranges: tuple[OperationsEstimateRange, ...]
    falsification_conditions: tuple[str, ...]
    evidence_ref_ids: tuple[str, ...]
    source_entry_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text("candidate_id", self.candidate_id, max_length=256)
        if not isinstance(self.kind, OperationsAdviceKind):
            raise ValueError("kind must be an OperationsAdviceKind")
        _require_text("target_division_id", self.target_division_id, max_length=256)
        _require_text("action_catalog_id", self.action_catalog_id, max_length=256)
        _require_text("summary", self.summary, max_length=2_000)
        _require_text("rationale", self.rationale, max_length=4_000)
        _require_non_negative_int("maximum_cost_minor", self.maximum_cost_minor)
        _require_non_negative_int("maximum_human_minutes", self.maximum_human_minutes)
        if not isinstance(self.requires_human_approval, bool):
            raise ValueError("requires_human_approval must be a boolean")
        if not isinstance(self.risk_level, OperationsRiskLevel):
            raise ValueError("risk_level must be an OperationsRiskLevel")
        if self.risk_level is OperationsRiskLevel.UNASSESSED:
            raise ValueError("advice risk_level must be assessed")
        if not isinstance(self.reversibility, OperationsReversibility):
            raise ValueError("reversibility must be an OperationsReversibility")
        _require_string_tuple(
            "prerequisite_fact_ids",
            self.prerequisite_fact_ids,
            max_items=64,
            item_max_length=256,
        )
        _require_typed_tuple(
            "prediction_ranges",
            self.prediction_ranges,
            item_type=OperationsEstimateRange,
            max_items=16,
            allow_empty=False,
        )
        _require_string_tuple(
            "falsification_conditions",
            self.falsification_conditions,
            max_items=16,
            item_max_length=2_000,
            allow_empty=False,
        )
        _require_string_tuple(
            "evidence_ref_ids",
            self.evidence_ref_ids,
            max_items=64,
            item_max_length=256,
        )
        _require_string_tuple(
            "source_entry_ids",
            self.source_entry_ids,
            max_items=20,
            item_max_length=256,
        )
        if not self.evidence_ref_ids and not self.source_entry_ids:
            raise ValueError("advice candidates require evidence or memory lineage")

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "OperationsAdviceCandidate":
        fields = frozenset(
            {
                "candidate_id",
                "kind",
                "target_division_id",
                "action_catalog_id",
                "summary",
                "rationale",
                "maximum_cost_minor",
                "maximum_human_minutes",
                "requires_human_approval",
                "risk_level",
                "reversibility",
                "prerequisite_fact_ids",
                "prediction_ranges",
                "falsification_conditions",
                "evidence_ref_ids",
                "source_entry_ids",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        return cls(
            candidate_id=payload["candidate_id"],
            kind=_closed_enum(OperationsAdviceKind, "kind", payload["kind"]),
            target_division_id=payload["target_division_id"],
            action_catalog_id=payload["action_catalog_id"],
            summary=payload["summary"],
            rationale=payload["rationale"],
            maximum_cost_minor=payload["maximum_cost_minor"],
            maximum_human_minutes=payload["maximum_human_minutes"],
            requires_human_approval=payload["requires_human_approval"],
            risk_level=_closed_enum(
                OperationsRiskLevel,
                "risk_level",
                payload["risk_level"],
            ),
            reversibility=_closed_enum(
                OperationsReversibility,
                "reversibility",
                payload["reversibility"],
            ),
            prerequisite_fact_ids=tuple(
                _array("prerequisite_fact_ids", payload["prerequisite_fact_ids"])
            ),
            prediction_ranges=tuple(
                OperationsEstimateRange.from_json(_mapping("prediction_ranges[]", item))
                for item in _array("prediction_ranges", payload["prediction_ranges"])
            ),
            falsification_conditions=tuple(_array("falsification_conditions", payload["falsification_conditions"])),
            evidence_ref_ids=tuple(_array("evidence_ref_ids", payload["evidence_ref_ids"])),
            source_entry_ids=tuple(_array("source_entry_ids", payload["source_entry_ids"])),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "kind": self.kind.value,
            "target_division_id": self.target_division_id,
            "action_catalog_id": self.action_catalog_id,
            "summary": self.summary,
            "rationale": self.rationale,
            "maximum_cost_minor": self.maximum_cost_minor,
            "maximum_human_minutes": self.maximum_human_minutes,
            "requires_human_approval": self.requires_human_approval,
            "risk_level": self.risk_level.value,
            "reversibility": self.reversibility.value,
            "prerequisite_fact_ids": list(self.prerequisite_fact_ids),
            "prediction_ranges": [item.to_json() for item in self.prediction_ranges],
            "falsification_conditions": list(self.falsification_conditions),
            "evidence_ref_ids": list(self.evidence_ref_ids),
            "source_entry_ids": list(self.source_entry_ids),
        }


@dataclass(frozen=True)
class OperationsPolicyCheckpoint:
    checkpoint_id: str
    content_sha256: str
    artifact_id: str
    action_weights: tuple[tuple[str, tuple[float, ...]], ...]
    intervention_weights: tuple[float, ...]
    intervention_bias: float
    learning_rate: float
    max_abs_parameter: float
    update_count: int
    processed_credit_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_content_id(
            "checkpoint_id",
            self.checkpoint_id,
            prefix="operations-policy-checkpoint:",
        )
        _require_sha256("content_sha256", self.content_sha256)
        if self.checkpoint_id != f"operations-policy-checkpoint:{self.content_sha256}":
            raise ValueError("checkpoint_id must match content_sha256")
        _require_text("artifact_id", self.artifact_id, max_length=256)
        if not isinstance(self.action_weights, tuple):
            raise ValueError("action_weights must be a tuple")
        expected_actions = tuple(item.value for item in OperationsAdviceKind)
        if tuple(name for name, _ in self.action_weights) != expected_actions:
            raise ValueError("action_weights must follow OperationsAdviceKind order")
        feature_count = len(OPERATIONS_POLICY_FEATURE_ORDER)
        for name, weights in self.action_weights:
            _require_text("action_weights name", name, max_length=64)
            if not isinstance(weights, tuple) or len(weights) != feature_count:
                raise ValueError("action weight rows must match policy feature order")
            for value in weights:
                numeric = _require_numeric("action weight", value)
                if abs(numeric) > self.max_abs_parameter:
                    raise ValueError("action weight exceeds max_abs_parameter")
        if (
            not isinstance(self.intervention_weights, tuple)
            or len(self.intervention_weights) != feature_count
        ):
            raise ValueError("intervention_weights must match policy feature order")
        for value in (*self.intervention_weights, self.intervention_bias):
            numeric = _require_numeric("intervention parameter", value)
            if abs(numeric) > self.max_abs_parameter:
                raise ValueError("intervention parameter exceeds max_abs_parameter")
        learning_rate = _require_numeric("learning_rate", self.learning_rate)
        if not 0.0 < learning_rate <= 0.5:
            raise ValueError("learning_rate must be in (0, 0.5]")
        parameter_cap = _require_numeric(
            "max_abs_parameter",
            self.max_abs_parameter,
        )
        if not 0.5 <= parameter_cap <= 8.0:
            raise ValueError("max_abs_parameter must be in [0.5, 8]")
        _require_non_negative_int("update_count", self.update_count)
        _require_string_tuple(
            "processed_credit_ids",
            self.processed_credit_ids,
            max_items=4_096,
            item_max_length=256,
        )
        if len(self.processed_credit_ids) != self.update_count:
            raise ValueError("processed credit count must equal update_count")

    @classmethod
    def create(
        cls,
        *,
        artifact_id: str,
        action_weights: tuple[tuple[str, tuple[float, ...]], ...],
        intervention_weights: tuple[float, ...],
        intervention_bias: float,
        learning_rate: float,
        max_abs_parameter: float,
        update_count: int = 0,
        processed_credit_ids: tuple[str, ...] = (),
    ) -> "OperationsPolicyCheckpoint":
        core = {
            "schema_version": OPERATIONS_POLICY_CHECKPOINT_SCHEMA_VERSION,
            "artifact_id": artifact_id,
            "feature_order": list(OPERATIONS_POLICY_FEATURE_ORDER),
            "action_weights": [
                {"action": action, "weights": list(weights)}
                for action, weights in action_weights
            ],
            "intervention_weights": list(intervention_weights),
            "intervention_bias": intervention_bias,
            "learning_rate": learning_rate,
            "max_abs_parameter": max_abs_parameter,
            "update_count": update_count,
            "processed_credit_ids": list(processed_credit_ids),
        }
        digest = stable_content_sha256(core)
        return cls(
            checkpoint_id=f"operations-policy-checkpoint:{digest}",
            content_sha256=digest,
            artifact_id=artifact_id,
            action_weights=action_weights,
            intervention_weights=intervention_weights,
            intervention_bias=intervention_bias,
            learning_rate=learning_rate,
            max_abs_parameter=max_abs_parameter,
            update_count=update_count,
            processed_credit_ids=processed_credit_ids,
        )

    @classmethod
    def from_json(
        cls,
        payload: Mapping[str, object],
    ) -> "OperationsPolicyCheckpoint":
        fields = frozenset(
            {
                "schema_version",
                "checkpoint_id",
                "content_sha256",
                "artifact_id",
                "feature_order",
                "action_weights",
                "intervention_weights",
                "intervention_bias",
                "learning_rate",
                "max_abs_parameter",
                "update_count",
                "processed_credit_ids",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        if payload["schema_version"] != OPERATIONS_POLICY_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported Operations policy checkpoint schema")
        if tuple(_array("feature_order", payload["feature_order"])) != OPERATIONS_POLICY_FEATURE_ORDER:
            raise ValueError("Operations policy feature order drift")
        rows: list[tuple[str, tuple[float, ...]]] = []
        for item in _array("action_weights", payload["action_weights"]):
            row = _mapping("action_weights[]", item)
            _strict_payload(
                row,
                allowed=frozenset({"action", "weights"}),
                required=frozenset({"action", "weights"}),
            )
            rows.append(
                (
                    row["action"],
                    tuple(_array("weights", row["weights"])),
                )
            )
        checkpoint = cls(
            checkpoint_id=payload["checkpoint_id"],
            content_sha256=payload["content_sha256"],
            artifact_id=payload["artifact_id"],
            action_weights=tuple(rows),
            intervention_weights=tuple(
                _array("intervention_weights", payload["intervention_weights"])
            ),
            intervention_bias=payload["intervention_bias"],
            learning_rate=payload["learning_rate"],
            max_abs_parameter=payload["max_abs_parameter"],
            update_count=payload["update_count"],
            processed_credit_ids=tuple(
                _array("processed_credit_ids", payload["processed_credit_ids"])
            ),
        )
        core = checkpoint.to_json()
        core.pop("checkpoint_id")
        core.pop("content_sha256")
        if stable_content_sha256(core) != checkpoint.content_sha256:
            raise ValueError("Operations policy checkpoint digest mismatch")
        return checkpoint

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": OPERATIONS_POLICY_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_id": self.checkpoint_id,
            "content_sha256": self.content_sha256,
            "artifact_id": self.artifact_id,
            "feature_order": list(OPERATIONS_POLICY_FEATURE_ORDER),
            "action_weights": [
                {"action": action, "weights": [float(value) for value in weights]}
                for action, weights in self.action_weights
            ],
            "intervention_weights": [
                float(value) for value in self.intervention_weights
            ],
            "intervention_bias": float(self.intervention_bias),
            "learning_rate": float(self.learning_rate),
            "max_abs_parameter": float(self.max_abs_parameter),
            "update_count": self.update_count,
            "processed_credit_ids": list(self.processed_credit_ids),
        }


@dataclass(frozen=True)
class OperationsRankedCandidate:
    candidate_id: str
    rank: int
    policy_score: float
    selection_probability: float
    feature_values: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        _require_text("candidate_id", self.candidate_id, max_length=256)
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("rank must be a positive integer")
        _require_numeric("policy_score", self.policy_score)
        _require_probability("selection_probability", self.selection_probability)
        if not isinstance(self.feature_values, tuple):
            raise ValueError("feature_values must be a tuple")
        if tuple(name for name, _ in self.feature_values) != OPERATIONS_POLICY_FEATURE_ORDER:
            raise ValueError("feature_values must follow policy feature order")
        for _, value in self.feature_values:
            _require_probability("feature value", value)

    @classmethod
    def from_json(
        cls,
        payload: Mapping[str, object],
    ) -> "OperationsRankedCandidate":
        fields = frozenset(
            {
                "candidate_id",
                "rank",
                "policy_score",
                "selection_probability",
                "feature_values",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        feature_values = tuple(
            (
                _mapping("feature_values[]", item)["name"],
                _mapping("feature_values[]", item)["value"],
            )
            for item in _array("feature_values", payload["feature_values"])
        )
        return cls(
            candidate_id=payload["candidate_id"],
            rank=payload["rank"],
            policy_score=payload["policy_score"],
            selection_probability=payload["selection_probability"],
            feature_values=feature_values,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "rank": self.rank,
            "policy_score": float(self.policy_score),
            "selection_probability": float(self.selection_probability),
            "feature_values": [
                {"name": name, "value": float(value)}
                for name, value in self.feature_values
            ],
        }


@dataclass(frozen=True)
class OperationsPolicyDecision:
    policy_decision_id: str
    content_sha256: str
    checkpoint_id: str
    checkpoint_update_count: int
    state_snapshot_id: str
    source_prediction_id: str
    mode: OperationsPolicyMode
    action: OperationsPolicyAction
    recommended_candidate_id: str
    selected_candidate_id: str
    intervention_probability: float
    ranked_candidates: tuple[OperationsRankedCandidate, ...]
    rationale_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_content_id(
            "policy_decision_id",
            self.policy_decision_id,
            prefix="operations-policy-decision:",
        )
        _require_sha256("content_sha256", self.content_sha256)
        if self.policy_decision_id != f"operations-policy-decision:{self.content_sha256}":
            raise ValueError("policy_decision_id must match content_sha256")
        _require_content_id(
            "checkpoint_id",
            self.checkpoint_id,
            prefix="operations-policy-checkpoint:",
        )
        _require_non_negative_int(
            "checkpoint_update_count",
            self.checkpoint_update_count,
        )
        _require_content_id(
            "state_snapshot_id",
            self.state_snapshot_id,
            prefix="operations-state:",
        )
        _require_text(
            "source_prediction_id",
            self.source_prediction_id,
            max_length=512,
        )
        if not isinstance(self.mode, OperationsPolicyMode):
            raise ValueError("mode must be an OperationsPolicyMode")
        if not isinstance(self.action, OperationsPolicyAction):
            raise ValueError("action must be an OperationsPolicyAction")
        _require_text(
            "recommended_candidate_id",
            self.recommended_candidate_id,
            max_length=256,
        )
        _require_optional_text(
            "selected_candidate_id",
            self.selected_candidate_id,
            max_length=256,
        )
        _require_probability(
            "intervention_probability",
            self.intervention_probability,
        )
        _require_typed_tuple(
            "ranked_candidates",
            self.ranked_candidates,
            item_type=OperationsRankedCandidate,
            max_items=32,
            allow_empty=False,
        )
        candidate_ids = tuple(item.candidate_id for item in self.ranked_candidates)
        _require_unique_ids("ranked_candidates", candidate_ids)
        if tuple(item.rank for item in self.ranked_candidates) != tuple(
            range(1, len(self.ranked_candidates) + 1)
        ):
            raise ValueError("ranked candidate ranks must be contiguous")
        if self.recommended_candidate_id != candidate_ids[0]:
            raise ValueError("recommended candidate must be rank 1")
        if self.action is OperationsPolicyAction.NOOP and self.selected_candidate_id:
            raise ValueError("NOOP decisions cannot select a candidate")
        if self.action is OperationsPolicyAction.INTERVENE:
            if self.selected_candidate_id != self.recommended_candidate_id:
                raise ValueError("INTERVENE decisions must select the recommendation")
        _require_string_tuple(
            "rationale_codes",
            self.rationale_codes,
            max_items=16,
            item_max_length=256,
            allow_empty=False,
        )

    @classmethod
    def create(
        cls,
        *,
        checkpoint_id: str,
        checkpoint_update_count: int,
        state_snapshot_id: str,
        source_prediction_id: str,
        mode: OperationsPolicyMode,
        action: OperationsPolicyAction,
        recommended_candidate_id: str,
        selected_candidate_id: str,
        intervention_probability: float,
        ranked_candidates: tuple[OperationsRankedCandidate, ...],
        rationale_codes: tuple[str, ...],
    ) -> "OperationsPolicyDecision":
        core = {
            "schema_version": OPERATIONS_POLICY_DECISION_SCHEMA_VERSION,
            "checkpoint_id": checkpoint_id,
            "checkpoint_update_count": checkpoint_update_count,
            "state_snapshot_id": state_snapshot_id,
            "source_prediction_id": source_prediction_id,
            "mode": mode.value,
            "action": action.value,
            "recommended_candidate_id": recommended_candidate_id,
            "selected_candidate_id": selected_candidate_id,
            "intervention_probability": intervention_probability,
            "ranked_candidates": [item.to_json() for item in ranked_candidates],
            "rationale_codes": list(rationale_codes),
        }
        digest = stable_content_sha256(core)
        return cls(
            policy_decision_id=f"operations-policy-decision:{digest}",
            content_sha256=digest,
            checkpoint_id=checkpoint_id,
            checkpoint_update_count=checkpoint_update_count,
            state_snapshot_id=state_snapshot_id,
            source_prediction_id=source_prediction_id,
            mode=mode,
            action=action,
            recommended_candidate_id=recommended_candidate_id,
            selected_candidate_id=selected_candidate_id,
            intervention_probability=intervention_probability,
            ranked_candidates=ranked_candidates,
            rationale_codes=rationale_codes,
        )

    @classmethod
    def from_json(
        cls,
        payload: Mapping[str, object],
    ) -> "OperationsPolicyDecision":
        fields = frozenset(
            {
                "schema_version",
                "policy_decision_id",
                "content_sha256",
                "checkpoint_id",
                "checkpoint_update_count",
                "state_snapshot_id",
                "source_prediction_id",
                "mode",
                "action",
                "recommended_candidate_id",
                "selected_candidate_id",
                "intervention_probability",
                "ranked_candidates",
                "rationale_codes",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        if payload["schema_version"] != OPERATIONS_POLICY_DECISION_SCHEMA_VERSION:
            raise ValueError("unsupported Operations policy decision schema")
        decision = cls(
            policy_decision_id=payload["policy_decision_id"],
            content_sha256=payload["content_sha256"],
            checkpoint_id=payload["checkpoint_id"],
            checkpoint_update_count=payload["checkpoint_update_count"],
            state_snapshot_id=payload["state_snapshot_id"],
            source_prediction_id=payload["source_prediction_id"],
            mode=_closed_enum(OperationsPolicyMode, "mode", payload["mode"]),
            action=_closed_enum(
                OperationsPolicyAction,
                "action",
                payload["action"],
            ),
            recommended_candidate_id=payload["recommended_candidate_id"],
            selected_candidate_id=payload["selected_candidate_id"],
            intervention_probability=payload["intervention_probability"],
            ranked_candidates=tuple(
                OperationsRankedCandidate.from_json(
                    _mapping("ranked_candidates[]", item)
                )
                for item in _array(
                    "ranked_candidates",
                    payload["ranked_candidates"],
                )
            ),
            rationale_codes=tuple(
                _array("rationale_codes", payload["rationale_codes"])
            ),
        )
        core = decision.to_json()
        core.pop("policy_decision_id")
        core.pop("content_sha256")
        if stable_content_sha256(core) != decision.content_sha256:
            raise ValueError("Operations policy decision digest mismatch")
        return decision

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": OPERATIONS_POLICY_DECISION_SCHEMA_VERSION,
            "policy_decision_id": self.policy_decision_id,
            "content_sha256": self.content_sha256,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_update_count": self.checkpoint_update_count,
            "state_snapshot_id": self.state_snapshot_id,
            "source_prediction_id": self.source_prediction_id,
            "mode": self.mode.value,
            "action": self.action.value,
            "recommended_candidate_id": self.recommended_candidate_id,
            "selected_candidate_id": self.selected_candidate_id,
            "intervention_probability": float(self.intervention_probability),
            "ranked_candidates": [item.to_json() for item in self.ranked_candidates],
            "rationale_codes": list(self.rationale_codes),
        }


@dataclass(frozen=True)
class OperationsPolicyCredit:
    credit_id: str
    content_sha256: str
    policy_decision_id: str
    selection_id: str
    candidate_id: str
    environment_outcome_id: str
    prediction_id: str
    signed_prediction_error: float
    source_credit_record_ids: tuple[str, ...]
    observed_at_ms: int

    def __post_init__(self) -> None:
        _require_content_id(
            "credit_id",
            self.credit_id,
            prefix="operations-policy-credit:",
        )
        _require_sha256("content_sha256", self.content_sha256)
        if self.credit_id != f"operations-policy-credit:{self.content_sha256}":
            raise ValueError("credit_id must match content_sha256")
        _require_content_id(
            "policy_decision_id",
            self.policy_decision_id,
            prefix="operations-policy-decision:",
        )
        for name, value in (
            ("selection_id", self.selection_id),
            ("candidate_id", self.candidate_id),
            ("environment_outcome_id", self.environment_outcome_id),
            ("prediction_id", self.prediction_id),
        ):
            _require_text(name, value, max_length=512)
        value = _require_numeric(
            "signed_prediction_error",
            self.signed_prediction_error,
        )
        if not -1.0 <= value <= 1.0:
            raise ValueError("signed_prediction_error must be in [-1, 1]")
        _require_string_tuple(
            "source_credit_record_ids",
            self.source_credit_record_ids,
            max_items=8,
            item_max_length=256,
            allow_empty=False,
        )
        if len(self.source_credit_record_ids) != 4:
            raise ValueError("Operations policy credit requires all four PE credits")
        _require_non_negative_int("observed_at_ms", self.observed_at_ms)

    @classmethod
    def create(
        cls,
        *,
        policy_decision_id: str,
        selection_id: str,
        candidate_id: str,
        environment_outcome_id: str,
        prediction_id: str,
        signed_prediction_error: float,
        source_credit_record_ids: tuple[str, ...],
        observed_at_ms: int,
    ) -> "OperationsPolicyCredit":
        core = {
            "schema_version": OPERATIONS_POLICY_CREDIT_SCHEMA_VERSION,
            "policy_decision_id": policy_decision_id,
            "selection_id": selection_id,
            "candidate_id": candidate_id,
            "environment_outcome_id": environment_outcome_id,
            "prediction_id": prediction_id,
            "signed_prediction_error": signed_prediction_error,
            "source_credit_record_ids": list(source_credit_record_ids),
            "observed_at_ms": observed_at_ms,
        }
        digest = stable_content_sha256(core)
        return cls(
            credit_id=f"operations-policy-credit:{digest}",
            content_sha256=digest,
            policy_decision_id=policy_decision_id,
            selection_id=selection_id,
            candidate_id=candidate_id,
            environment_outcome_id=environment_outcome_id,
            prediction_id=prediction_id,
            signed_prediction_error=signed_prediction_error,
            source_credit_record_ids=source_credit_record_ids,
            observed_at_ms=observed_at_ms,
        )

    @classmethod
    def from_json(
        cls,
        payload: Mapping[str, object],
    ) -> "OperationsPolicyCredit":
        fields = frozenset(
            {
                "schema_version",
                "credit_id",
                "content_sha256",
                "policy_decision_id",
                "selection_id",
                "candidate_id",
                "environment_outcome_id",
                "prediction_id",
                "signed_prediction_error",
                "source_credit_record_ids",
                "observed_at_ms",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        if payload["schema_version"] != OPERATIONS_POLICY_CREDIT_SCHEMA_VERSION:
            raise ValueError("unsupported Operations policy credit schema")
        credit = cls(
            credit_id=payload["credit_id"],
            content_sha256=payload["content_sha256"],
            policy_decision_id=payload["policy_decision_id"],
            selection_id=payload["selection_id"],
            candidate_id=payload["candidate_id"],
            environment_outcome_id=payload["environment_outcome_id"],
            prediction_id=payload["prediction_id"],
            signed_prediction_error=payload["signed_prediction_error"],
            source_credit_record_ids=tuple(
                _array(
                    "source_credit_record_ids",
                    payload["source_credit_record_ids"],
                )
            ),
            observed_at_ms=payload["observed_at_ms"],
        )
        core = credit.to_json()
        core.pop("credit_id")
        core.pop("content_sha256")
        if stable_content_sha256(core) != credit.content_sha256:
            raise ValueError("Operations policy credit digest mismatch")
        return credit

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": OPERATIONS_POLICY_CREDIT_SCHEMA_VERSION,
            "credit_id": self.credit_id,
            "content_sha256": self.content_sha256,
            "policy_decision_id": self.policy_decision_id,
            "selection_id": self.selection_id,
            "candidate_id": self.candidate_id,
            "environment_outcome_id": self.environment_outcome_id,
            "prediction_id": self.prediction_id,
            "signed_prediction_error": float(self.signed_prediction_error),
            "source_credit_record_ids": list(self.source_credit_record_ids),
            "observed_at_ms": self.observed_at_ms,
        }


@dataclass(frozen=True)
class OperationsPolicyUpdateReceipt:
    update_id: str
    content_sha256: str
    credit_id: str
    policy_decision_id: str
    candidate_id: str
    previous_checkpoint_id: str
    next_checkpoint_id: str
    parameter_delta_l2: float
    update_count: int

    def __post_init__(self) -> None:
        _require_content_id(
            "update_id",
            self.update_id,
            prefix="operations-policy-update:",
        )
        _require_sha256("content_sha256", self.content_sha256)
        if self.update_id != f"operations-policy-update:{self.content_sha256}":
            raise ValueError("update_id must match content_sha256")
        _require_content_id(
            "credit_id",
            self.credit_id,
            prefix="operations-policy-credit:",
        )
        _require_content_id(
            "policy_decision_id",
            self.policy_decision_id,
            prefix="operations-policy-decision:",
        )
        _require_text("candidate_id", self.candidate_id, max_length=256)
        for name, value in (
            ("previous_checkpoint_id", self.previous_checkpoint_id),
            ("next_checkpoint_id", self.next_checkpoint_id),
        ):
            _require_content_id(
                name,
                value,
                prefix="operations-policy-checkpoint:",
            )
        delta = _require_numeric("parameter_delta_l2", self.parameter_delta_l2)
        if delta < 0.0:
            raise ValueError("parameter_delta_l2 must be non-negative")
        count = _require_non_negative_int("update_count", self.update_count)
        if count < 1:
            raise ValueError("update_count must be positive")

    @classmethod
    def create(
        cls,
        *,
        credit_id: str,
        policy_decision_id: str,
        candidate_id: str,
        previous_checkpoint_id: str,
        next_checkpoint_id: str,
        parameter_delta_l2: float,
        update_count: int,
    ) -> "OperationsPolicyUpdateReceipt":
        core = {
            "schema_version": OPERATIONS_POLICY_UPDATE_SCHEMA_VERSION,
            "credit_id": credit_id,
            "policy_decision_id": policy_decision_id,
            "candidate_id": candidate_id,
            "previous_checkpoint_id": previous_checkpoint_id,
            "next_checkpoint_id": next_checkpoint_id,
            "parameter_delta_l2": parameter_delta_l2,
            "update_count": update_count,
        }
        digest = stable_content_sha256(core)
        return cls(
            update_id=f"operations-policy-update:{digest}",
            content_sha256=digest,
            credit_id=credit_id,
            policy_decision_id=policy_decision_id,
            candidate_id=candidate_id,
            previous_checkpoint_id=previous_checkpoint_id,
            next_checkpoint_id=next_checkpoint_id,
            parameter_delta_l2=parameter_delta_l2,
            update_count=update_count,
        )

    @classmethod
    def from_json(
        cls,
        payload: Mapping[str, object],
    ) -> "OperationsPolicyUpdateReceipt":
        fields = frozenset(
            {
                "schema_version",
                "update_id",
                "content_sha256",
                "credit_id",
                "policy_decision_id",
                "candidate_id",
                "previous_checkpoint_id",
                "next_checkpoint_id",
                "parameter_delta_l2",
                "update_count",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        if payload["schema_version"] != OPERATIONS_POLICY_UPDATE_SCHEMA_VERSION:
            raise ValueError("unsupported Operations policy update schema")
        receipt = cls(
            update_id=payload["update_id"],
            content_sha256=payload["content_sha256"],
            credit_id=payload["credit_id"],
            policy_decision_id=payload["policy_decision_id"],
            candidate_id=payload["candidate_id"],
            previous_checkpoint_id=payload["previous_checkpoint_id"],
            next_checkpoint_id=payload["next_checkpoint_id"],
            parameter_delta_l2=payload["parameter_delta_l2"],
            update_count=payload["update_count"],
        )
        core = receipt.to_json()
        core.pop("update_id")
        core.pop("content_sha256")
        if stable_content_sha256(core) != receipt.content_sha256:
            raise ValueError("Operations policy update digest mismatch")
        return receipt

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": OPERATIONS_POLICY_UPDATE_SCHEMA_VERSION,
            "update_id": self.update_id,
            "content_sha256": self.content_sha256,
            "credit_id": self.credit_id,
            "policy_decision_id": self.policy_decision_id,
            "candidate_id": self.candidate_id,
            "previous_checkpoint_id": self.previous_checkpoint_id,
            "next_checkpoint_id": self.next_checkpoint_id,
            "parameter_delta_l2": float(self.parameter_delta_l2),
            "update_count": self.update_count,
        }


@dataclass(frozen=True)
class OperationsAdviceSnapshot:
    advice_id: str
    source_turn_index: int
    candidate_regime_id: str
    candidate_abstract_action: str
    candidates: tuple[OperationsAdviceCandidate, ...]
    rationale: str
    policy_decision: OperationsPolicyDecision | None = None
    activation_receipt_id: str = ""
    wiring_level: WiringLevel = WiringLevel.SHADOW
    applied: bool = False

    def __post_init__(self) -> None:
        _require_content_id("advice_id", self.advice_id, prefix="operations-advice:")
        _require_non_negative_int("source_turn_index", self.source_turn_index)
        _require_optional_text("candidate_regime_id", self.candidate_regime_id, max_length=512)
        _require_optional_text(
            "candidate_abstract_action",
            self.candidate_abstract_action,
            max_length=512,
        )
        _require_typed_tuple(
            "candidates",
            self.candidates,
            item_type=OperationsAdviceCandidate,
            max_items=32,
        )
        _require_unique_ids("candidates", tuple(item.candidate_id for item in self.candidates))
        _require_text("rationale", self.rationale, max_length=2_000)
        if self.policy_decision is not None:
            if not isinstance(self.policy_decision, OperationsPolicyDecision):
                raise ValueError(
                    "policy_decision must be an OperationsPolicyDecision"
                )
            if not self.candidates:
                raise ValueError("v2 policy advice must contain candidates")
            if tuple(item.candidate_id for item in self.candidates) != tuple(
                item.candidate_id
                for item in self.policy_decision.ranked_candidates
            ):
                raise ValueError(
                    "advice candidates must match policy ranking order"
                )
        if self.wiring_level not in {WiringLevel.SHADOW, WiringLevel.ACTIVE}:
            raise ValueError("Operations advice must be SHADOW or ACTIVE")
        if self.wiring_level is WiringLevel.ACTIVE:
            if self.policy_decision is None:
                raise ValueError("ACTIVE Operations advice requires a policy decision")
            _require_content_id(
                "activation_receipt_id",
                self.activation_receipt_id,
                prefix="operations-policy-activation:",
            )
        elif self.activation_receipt_id:
            raise ValueError("SHADOW advice cannot carry an activation receipt")
        if not isinstance(self.applied, bool):
            raise ValueError("applied must be a boolean")
        if self.applied:
            raise ValueError("Operations advice publication cannot claim applied")

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "OperationsAdviceSnapshot":
        fields = frozenset(
            {
                "schema_version",
                "advice_id",
                "source_turn_index",
                "candidate_regime_id",
                "candidate_abstract_action",
                "candidates",
                "rationale",
                "policy_decision",
                "activation_receipt_id",
                "wiring_level",
                "applied",
            }
        )
        _strict_payload(
            payload,
            allowed=fields,
            required=fields - {"policy_decision", "activation_receipt_id"},
        )
        schema_version = payload["schema_version"]
        if schema_version not in {ADVICE_SCHEMA_VERSION, ADVICE_V2_SCHEMA_VERSION}:
            raise ValueError("unsupported Operations advice schema")
        if schema_version == ADVICE_SCHEMA_VERSION and {
            "policy_decision",
            "activation_receipt_id",
        } & set(payload):
            raise ValueError("policy activation fields require operations-advice.v2")
        if schema_version == ADVICE_V2_SCHEMA_VERSION and not {
            "policy_decision",
            "activation_receipt_id",
        }.issubset(payload):
            raise ValueError(
                "operations-advice.v2 requires policy_decision and activation_receipt_id"
            )
        return cls(
            advice_id=payload["advice_id"],
            source_turn_index=payload["source_turn_index"],
            candidate_regime_id=payload["candidate_regime_id"],
            candidate_abstract_action=payload["candidate_abstract_action"],
            candidates=tuple(
                OperationsAdviceCandidate.from_json(_mapping("candidates[]", item))
                for item in _array("candidates", payload["candidates"])
            ),
            rationale=payload["rationale"],
            policy_decision=(
                OperationsPolicyDecision.from_json(
                    _mapping("policy_decision", payload["policy_decision"])
                )
                if "policy_decision" in payload
                else None
            ),
            activation_receipt_id=payload.get("activation_receipt_id", ""),
            wiring_level=_closed_enum(WiringLevel, "wiring_level", payload["wiring_level"]),
            applied=payload["applied"],
        )

    def to_json(self) -> dict[str, object]:
        payload = {
            "schema_version": (
                ADVICE_V2_SCHEMA_VERSION
                if self.policy_decision is not None
                else ADVICE_SCHEMA_VERSION
            ),
            "advice_id": self.advice_id,
            "source_turn_index": self.source_turn_index,
            "candidate_regime_id": self.candidate_regime_id,
            "candidate_abstract_action": self.candidate_abstract_action,
            "candidates": [candidate.to_json() for candidate in self.candidates],
            "rationale": self.rationale,
            "wiring_level": self.wiring_level.value,
            "applied": self.applied,
        }
        if self.policy_decision is not None:
            payload["policy_decision"] = self.policy_decision.to_json()
            payload["activation_receipt_id"] = self.activation_receipt_id
        return payload


@dataclass(frozen=True)
class OperationsMetricObservation:
    metric_id: str
    unit: str
    baseline_value: float
    observed_value: float
    evidence_ref_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text("metric_id", self.metric_id, max_length=256)
        _require_text("unit", self.unit, max_length=128)
        _require_numeric("baseline_value", self.baseline_value)
        _require_numeric("observed_value", self.observed_value)
        _require_string_tuple(
            "evidence_ref_ids",
            self.evidence_ref_ids,
            max_items=32,
            item_max_length=256,
            allow_empty=False,
        )

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "OperationsMetricObservation":
        fields = frozenset(
            {
                "metric_id",
                "unit",
                "baseline_value",
                "observed_value",
                "evidence_ref_ids",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        return cls(
            metric_id=payload["metric_id"],
            unit=payload["unit"],
            baseline_value=payload["baseline_value"],
            observed_value=payload["observed_value"],
            evidence_ref_ids=tuple(
                _array("evidence_ref_ids", payload["evidence_ref_ids"])
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "metric_id": self.metric_id,
            "unit": self.unit,
            "baseline_value": float(self.baseline_value),
            "observed_value": float(self.observed_value),
            "evidence_ref_ids": list(self.evidence_ref_ids),
        }


@dataclass(frozen=True)
class OperationsCostBreakdown:
    model_minor: int = 0
    data_minor: int = 0
    human_minor: int = 0
    infrastructure_minor: int = 0
    vendor_minor: int = 0
    incident_response_minor: int = 0
    other_minor: int = 0

    def __post_init__(self) -> None:
        for name, value in self.to_json().items():
            _require_non_negative_int(name, value)

    @property
    def total_minor(self) -> int:
        return sum(self.to_json().values())

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "OperationsCostBreakdown":
        fields = frozenset(
            {
                "model_minor",
                "data_minor",
                "human_minor",
                "infrastructure_minor",
                "vendor_minor",
                "incident_response_minor",
                "other_minor",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        return cls(**dict(payload))

    def to_json(self) -> dict[str, int]:
        return {
            "model_minor": self.model_minor,
            "data_minor": self.data_minor,
            "human_minor": self.human_minor,
            "infrastructure_minor": self.infrastructure_minor,
            "vendor_minor": self.vendor_minor,
            "incident_response_minor": self.incident_response_minor,
            "other_minor": self.other_minor,
        }


@dataclass(frozen=True)
class OperationsExecutionOutcome:
    objective_result: OperationsObjectiveResult
    metrics: tuple[OperationsMetricObservation, ...]
    currency: str
    realized_costs: OperationsCostBreakdown
    elapsed_ms: int
    blocker_duration_ms: int
    rework_count: int
    incident_count: int
    human_minutes: int
    risk_level: OperationsRiskLevel
    reversibility: OperationsReversibility

    def __post_init__(self) -> None:
        if not isinstance(self.objective_result, OperationsObjectiveResult):
            raise ValueError("objective_result must be an OperationsObjectiveResult")
        _require_typed_tuple(
            "metrics",
            self.metrics,
            item_type=OperationsMetricObservation,
            max_items=64,
        )
        _require_unique_ids("metrics", tuple(item.metric_id for item in self.metrics))
        _require_currency(self.currency)
        if not isinstance(self.realized_costs, OperationsCostBreakdown):
            raise ValueError("realized_costs must be an OperationsCostBreakdown")
        _require_non_negative_int("elapsed_ms", self.elapsed_ms)
        _require_non_negative_int("blocker_duration_ms", self.blocker_duration_ms)
        _require_non_negative_int("rework_count", self.rework_count)
        _require_non_negative_int("incident_count", self.incident_count)
        _require_non_negative_int("human_minutes", self.human_minutes)
        if not isinstance(self.risk_level, OperationsRiskLevel):
            raise ValueError("risk_level must be an OperationsRiskLevel")
        if not isinstance(self.reversibility, OperationsReversibility):
            raise ValueError("reversibility must be an OperationsReversibility")

    @property
    def has_operational_observation(self) -> bool:
        return bool(
            self.objective_result is not OperationsObjectiveResult.NOT_OBSERVED
            or self.metrics
            or self.realized_costs.total_minor
            or self.elapsed_ms
            or self.blocker_duration_ms
            or self.rework_count
            or self.incident_count
            or self.human_minutes
            or self.risk_level is not OperationsRiskLevel.UNASSESSED
        )

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "OperationsExecutionOutcome":
        fields = frozenset(
            {
                "objective_result",
                "metrics",
                "currency",
                "realized_costs",
                "elapsed_ms",
                "blocker_duration_ms",
                "rework_count",
                "incident_count",
                "human_minutes",
                "risk_level",
                "reversibility",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        return cls(
            objective_result=_closed_enum(
                OperationsObjectiveResult,
                "objective_result",
                payload["objective_result"],
            ),
            metrics=tuple(
                OperationsMetricObservation.from_json(_mapping("metrics[]", item))
                for item in _array("metrics", payload["metrics"])
            ),
            currency=payload["currency"],
            realized_costs=OperationsCostBreakdown.from_json(_mapping("realized_costs", payload["realized_costs"])),
            elapsed_ms=payload["elapsed_ms"],
            blocker_duration_ms=payload["blocker_duration_ms"],
            rework_count=payload["rework_count"],
            incident_count=payload["incident_count"],
            human_minutes=payload["human_minutes"],
            risk_level=_closed_enum(OperationsRiskLevel, "risk_level", payload["risk_level"]),
            reversibility=_closed_enum(
                OperationsReversibility,
                "reversibility",
                payload["reversibility"],
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "objective_result": self.objective_result.value,
            "metrics": [item.to_json() for item in self.metrics],
            "currency": self.currency,
            "realized_costs": self.realized_costs.to_json(),
            "elapsed_ms": self.elapsed_ms,
            "blocker_duration_ms": self.blocker_duration_ms,
            "rework_count": self.rework_count,
            "incident_count": self.incident_count,
            "human_minutes": self.human_minutes,
            "risk_level": self.risk_level.value,
            "reversibility": self.reversibility.value,
        }


@dataclass(frozen=True)
class OperationsOutcomeReport:
    outcome_id: str
    context_pack_id: str
    decision_id: str
    work_order_ref: str
    decision: OperationsDecisionKind
    outcome_kind: OperationsOutcomeKind
    evidence_class: OperationsEvidenceClass
    verdict: OperationsOutcomeVerdict
    summary: str
    detail: str
    observed_at_ms: int
    evidence_refs: tuple[OperationsEvidenceRef, ...]
    execution_outcome: OperationsExecutionOutcome
    source_advice_id: str = ""
    policy_decision_id: str = ""
    selection_id: str = ""
    selected_candidate_id: str = ""
    activation_receipt_id: str = ""
    selection_wiring_level: WiringLevel = WiringLevel.SHADOW
    policy_action_applied: bool = False
    candidate_applied: bool = False

    def __post_init__(self) -> None:
        _require_text("outcome_id", self.outcome_id, max_length=256)
        _require_content_id(
            "context_pack_id",
            self.context_pack_id,
            prefix="operations-context-pack:",
        )
        _require_text("decision_id", self.decision_id, max_length=256)
        _require_text("work_order_ref", self.work_order_ref, max_length=512)
        if not isinstance(self.decision, OperationsDecisionKind):
            raise ValueError("decision must be an OperationsDecisionKind")
        if not isinstance(self.outcome_kind, OperationsOutcomeKind):
            raise ValueError("outcome_kind must be an OperationsOutcomeKind")
        if not isinstance(self.evidence_class, OperationsEvidenceClass):
            raise ValueError("evidence_class must be an OperationsEvidenceClass")
        if self.outcome_kind not in _OUTCOME_CLASS_PAIRS[self.evidence_class]:
            raise ValueError(
                f"outcome_kind={self.outcome_kind.value} is not legal for evidence_class={self.evidence_class.value}"
            )
        if not isinstance(self.verdict, OperationsOutcomeVerdict):
            raise ValueError("verdict must be an OperationsOutcomeVerdict")
        _require_text("summary", self.summary, max_length=2_000)
        _require_text("detail", self.detail, max_length=16_000)
        _require_non_negative_int("observed_at_ms", self.observed_at_ms)
        _require_typed_tuple(
            "evidence_refs",
            self.evidence_refs,
            item_type=OperationsEvidenceRef,
            max_items=128,
            allow_empty=False,
        )
        _require_unique_ids("evidence_refs", tuple(item.ref_id for item in self.evidence_refs))
        if any(reference.evidence_class is not self.evidence_class for reference in self.evidence_refs):
            raise ValueError("all outcome evidence refs must match evidence_class")
        if not isinstance(self.execution_outcome, OperationsExecutionOutcome):
            raise ValueError("execution_outcome must be an OperationsExecutionOutcome")
        lineage_values = (
            self.source_advice_id,
            self.policy_decision_id,
            self.selection_id,
        )
        if any(lineage_values) and not all(lineage_values):
            raise ValueError("v2 selection lineage must be complete")
        if self.source_advice_id:
            _require_content_id(
                "source_advice_id",
                self.source_advice_id,
                prefix="operations-advice:",
            )
            _require_content_id(
                "policy_decision_id",
                self.policy_decision_id,
                prefix="operations-policy-decision:",
            )
            _require_text("selection_id", self.selection_id, max_length=512)
            _require_optional_text(
                "selected_candidate_id",
                self.selected_candidate_id,
                max_length=256,
            )
        elif (
            self.selected_candidate_id
            or self.activation_receipt_id
            or self.selection_wiring_level is WiringLevel.ACTIVE
            or self.policy_action_applied
            or self.candidate_applied
        ):
            raise ValueError("policy selection fields require v2 selection lineage")
        if self.selection_wiring_level not in {
            WiringLevel.SHADOW,
            WiringLevel.ACTIVE,
        }:
            raise ValueError("selection_wiring_level must be SHADOW or ACTIVE")
        if self.selection_wiring_level is WiringLevel.ACTIVE:
            _require_content_id(
                "activation_receipt_id",
                self.activation_receipt_id,
                prefix="operations-policy-activation:",
            )
        elif self.activation_receipt_id:
            raise ValueError("SHADOW selection cannot carry activation_receipt_id")
        if not isinstance(self.policy_action_applied, bool):
            raise ValueError("policy_action_applied must be a boolean")
        if (
            self.policy_action_applied
            and self.selection_wiring_level is not WiringLevel.ACTIVE
        ):
            raise ValueError("policy_action_applied requires ACTIVE selection wiring")
        if not isinstance(self.candidate_applied, bool):
            raise ValueError("candidate_applied must be a boolean")
        if self.selected_candidate_id and not self.candidate_applied:
            raise ValueError("selected_candidate_id requires candidate_applied")
        if self.candidate_applied:
            if not self.selected_candidate_id:
                raise ValueError("candidate_applied requires selected_candidate_id")
            if self.selection_wiring_level is not WiringLevel.ACTIVE:
                raise ValueError("candidate_applied requires ACTIVE selection wiring")
            if not self.policy_action_applied:
                raise ValueError(
                    "candidate_applied requires policy_action_applied"
                )
        known_refs = {reference.ref_id for reference in self.evidence_refs}
        metric_refs = {
            reference_id
            for metric in self.execution_outcome.metrics
            for reference_id in metric.evidence_ref_ids
        }
        unknown_metric_refs = metric_refs - known_refs
        if unknown_metric_refs:
            raise ValueError(
                "metrics reference unknown evidence ids: "
                + ", ".join(sorted(unknown_metric_refs))
            )
        self._validate_lane_payload()

    @property
    def pe_eligible(self) -> bool:
        return (
            self.evidence_class is OperationsEvidenceClass.FIELD
            and self.outcome_kind is OperationsOutcomeKind.FIELD_OPERATION_RESULT
        )

    def _validate_lane_payload(self) -> None:
        roles = {reference.role for reference in self.evidence_refs}
        execution = self.execution_outcome
        if self.evidence_class in {
            OperationsEvidenceClass.SIMULATION,
            OperationsEvidenceClass.INTERNAL_REVIEW,
            OperationsEvidenceClass.MACHINE_CHECK,
        }:
            if execution.has_operational_observation:
                raise ValueError(
                    "simulation/internal_review/machine_check outcomes cannot carry objective, "
                    "metric, cost, elapsed, blocker, rework, incident, human-load, or risk observations"
                )
            return
        if self.evidence_class is not OperationsEvidenceClass.FIELD:
            return
        if OperationsEvidenceRole.WORK_ORDER not in roles:
            raise ValueError("field outcomes require evidence role=work_order")
        if not any(
            reference.role is OperationsEvidenceRole.WORK_ORDER
            and reference.locator == self.work_order_ref
            for reference in self.evidence_refs
        ):
            raise ValueError(
                "field outcomes require a work_order evidence locator matching work_order_ref"
            )
        required_role: OperationsEvidenceRole | None = None
        if self.outcome_kind is OperationsOutcomeKind.WORK_ORDER_PROGRESS:
            required_role = OperationsEvidenceRole.WORK_ORDER
            if not execution.has_operational_observation:
                raise ValueError("work_order_progress requires an observed operating dimension")
        elif self.outcome_kind is OperationsOutcomeKind.OBJECTIVE_PROGRESS:
            required_role = OperationsEvidenceRole.OBJECTIVE_PROGRESS
            if execution.objective_result is OperationsObjectiveResult.NOT_OBSERVED:
                raise ValueError("objective_progress requires an observed objective result")
        elif self.outcome_kind is OperationsOutcomeKind.COST_RECORDED:
            required_role = OperationsEvidenceRole.COST
            if execution.realized_costs.total_minor <= 0:
                raise ValueError("cost_recorded requires positive realized cost")
        elif self.outcome_kind is OperationsOutcomeKind.INCIDENT_RECORDED:
            required_role = OperationsEvidenceRole.INCIDENT
            if execution.incident_count <= 0:
                raise ValueError("incident_recorded requires a positive incident_count")
        elif self.outcome_kind is OperationsOutcomeKind.HUMAN_LOAD_RECORDED:
            required_role = OperationsEvidenceRole.HUMAN_LOAD
            if execution.human_minutes <= 0:
                raise ValueError("human_load_recorded requires positive human_minutes")
        elif self.outcome_kind is OperationsOutcomeKind.FIELD_OPERATION_RESULT:
            has_aggregate_dimension = bool(
                execution.objective_result is not OperationsObjectiveResult.NOT_OBSERVED
                or execution.metrics
                or execution.realized_costs.total_minor
                or execution.blocker_duration_ms
                or execution.rework_count
                or execution.incident_count
                or execution.human_minutes
                or execution.risk_level is not OperationsRiskLevel.UNASSESSED
            )
            if not has_aggregate_dimension:
                raise ValueError(
                    "field_operation_result requires an observed multi-objective dimension "
                    "beyond elapsed time"
                )
        if required_role is not None and required_role not in roles:
            raise ValueError(f"{self.outcome_kind.value} requires evidence role={required_role.value}")
        dimension_roles = (
            (
                execution.objective_result is not OperationsObjectiveResult.NOT_OBSERVED,
                OperationsEvidenceRole.OBJECTIVE_PROGRESS,
                "observed objective_result",
            ),
            (
                bool(execution.metrics),
                OperationsEvidenceRole.FIELD_OBSERVATION,
                "metric observations",
            ),
            (
                execution.realized_costs.total_minor > 0,
                OperationsEvidenceRole.COST,
                "positive realized costs",
            ),
            (
                execution.incident_count > 0,
                OperationsEvidenceRole.INCIDENT,
                "positive incident_count",
            ),
            (
                execution.human_minutes > 0,
                OperationsEvidenceRole.HUMAN_LOAD,
                "positive human_minutes",
            ),
        )
        for observed, role, dimension in dimension_roles:
            if observed and role not in roles:
                raise ValueError(f"{dimension} requires evidence role={role.value}")

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "OperationsOutcomeReport":
        fields = frozenset(
            {
                "schema_version",
                "outcome_id",
                "context_pack_id",
                "decision_id",
                "work_order_ref",
                "decision",
                "outcome_kind",
                "evidence_class",
                "verdict",
                "summary",
                "detail",
                "observed_at_ms",
                "evidence_refs",
                "execution_outcome",
                "source_advice_id",
                "policy_decision_id",
                "selection_id",
                "selected_candidate_id",
                "selection_wiring_level",
                "activation_receipt_id",
                "policy_action_applied",
                "candidate_applied",
            }
        )
        v2_fields = frozenset(
            {
                "source_advice_id",
                "policy_decision_id",
                "selection_id",
                "selected_candidate_id",
                "selection_wiring_level",
                "activation_receipt_id",
                "policy_action_applied",
                "candidate_applied",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields - v2_fields)
        schema_version = payload["schema_version"]
        if schema_version not in {
            OUTCOME_REPORT_SCHEMA_VERSION,
            OUTCOME_REPORT_V2_SCHEMA_VERSION,
        }:
            raise ValueError("unsupported Operations Outcome Report schema")
        if schema_version == OUTCOME_REPORT_SCHEMA_VERSION and set(payload) & v2_fields:
            raise ValueError("selection lineage requires operations-outcome-report.v2")
        if schema_version == OUTCOME_REPORT_V2_SCHEMA_VERSION and not v2_fields.issubset(payload):
            raise ValueError("operations-outcome-report.v2 requires selection lineage")
        evidence_refs = tuple(
            OperationsEvidenceRef.from_json(_mapping("evidence_refs[]", item))
            for item in _array("evidence_refs", payload["evidence_refs"])
        )
        return cls(
            outcome_id=payload["outcome_id"],
            context_pack_id=payload["context_pack_id"],
            decision_id=payload["decision_id"],
            work_order_ref=payload["work_order_ref"],
            decision=_closed_enum(OperationsDecisionKind, "decision", payload["decision"]),
            outcome_kind=_closed_enum(OperationsOutcomeKind, "outcome_kind", payload["outcome_kind"]),
            evidence_class=_closed_enum(
                OperationsEvidenceClass,
                "evidence_class",
                payload["evidence_class"],
            ),
            verdict=_closed_enum(OperationsOutcomeVerdict, "verdict", payload["verdict"]),
            summary=payload["summary"],
            detail=payload["detail"],
            observed_at_ms=payload["observed_at_ms"],
            evidence_refs=evidence_refs,
            execution_outcome=OperationsExecutionOutcome.from_json(
                _mapping("execution_outcome", payload["execution_outcome"])
            ),
            source_advice_id=payload.get("source_advice_id", ""),
            policy_decision_id=payload.get("policy_decision_id", ""),
            selection_id=payload.get("selection_id", ""),
            selected_candidate_id=payload.get("selected_candidate_id", ""),
            selection_wiring_level=(
                _closed_enum(
                    WiringLevel,
                    "selection_wiring_level",
                    payload["selection_wiring_level"],
                )
                if "selection_wiring_level" in payload
                else WiringLevel.SHADOW
            ),
            activation_receipt_id=payload.get("activation_receipt_id", ""),
            policy_action_applied=payload.get("policy_action_applied", False),
            candidate_applied=payload.get("candidate_applied", False),
        )

    def to_json(self) -> dict[str, object]:
        payload = {
            "schema_version": (
                OUTCOME_REPORT_V2_SCHEMA_VERSION
                if self.source_advice_id
                else OUTCOME_REPORT_SCHEMA_VERSION
            ),
            "outcome_id": self.outcome_id,
            "context_pack_id": self.context_pack_id,
            "decision_id": self.decision_id,
            "work_order_ref": self.work_order_ref,
            "decision": self.decision.value,
            "outcome_kind": self.outcome_kind.value,
            "evidence_class": self.evidence_class.value,
            "verdict": self.verdict.value,
            "summary": self.summary,
            "detail": self.detail,
            "observed_at_ms": self.observed_at_ms,
            "evidence_refs": [reference.to_json() for reference in self.evidence_refs],
            "execution_outcome": self.execution_outcome.to_json(),
        }
        if self.source_advice_id:
            payload.update(
                {
                    "source_advice_id": self.source_advice_id,
                    "policy_decision_id": self.policy_decision_id,
                    "selection_id": self.selection_id,
                    "selected_candidate_id": self.selected_candidate_id,
                    "selection_wiring_level": self.selection_wiring_level.value,
                    "activation_receipt_id": self.activation_receipt_id,
                    "policy_action_applied": self.policy_action_applied,
                    "candidate_applied": self.candidate_applied,
                }
            )
        return payload


@dataclass(frozen=True)
class OperationsRecalledExperience:
    memory_entry_id: str
    company_id: str
    cycle_id: str
    workstream_id: str
    decision_id: str
    source_context_pack_id: str
    created_at_ms: int
    report: OperationsOutcomeReport

    def __post_init__(self) -> None:
        for name, value in (
            ("memory_entry_id", self.memory_entry_id),
            ("company_id", self.company_id),
            ("cycle_id", self.cycle_id),
            ("decision_id", self.decision_id),
            ("source_context_pack_id", self.source_context_pack_id),
        ):
            _require_text(name, value, max_length=256)
        _require_optional_text("workstream_id", self.workstream_id, max_length=256)
        _require_non_negative_int("created_at_ms", self.created_at_ms)
        if not isinstance(self.report, OperationsOutcomeReport):
            raise ValueError("report must be an OperationsOutcomeReport")
        if self.source_context_pack_id != self.report.context_pack_id:
            raise ValueError("source_context_pack_id must match report lineage")

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "OperationsRecalledExperience":
        fields = frozenset(
            {
                "memory_entry_id",
                "company_id",
                "cycle_id",
                "workstream_id",
                "decision_id",
                "source_context_pack_id",
                "created_at_ms",
                "report",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        return cls(
            memory_entry_id=payload["memory_entry_id"],
            company_id=payload["company_id"],
            cycle_id=payload["cycle_id"],
            workstream_id=payload["workstream_id"],
            decision_id=payload["decision_id"],
            source_context_pack_id=payload["source_context_pack_id"],
            created_at_ms=payload["created_at_ms"],
            report=OperationsOutcomeReport.from_json(_mapping("report", payload["report"])),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "memory_entry_id": self.memory_entry_id,
            "company_id": self.company_id,
            "cycle_id": self.cycle_id,
            "workstream_id": self.workstream_id,
            "decision_id": self.decision_id,
            "source_context_pack_id": self.source_context_pack_id,
            "created_at_ms": self.created_at_ms,
            "report": self.report.to_json(),
        }


@dataclass(frozen=True)
class OperationsContextPackSnapshot:
    context_pack_id: str
    content_sha256: str
    session_id: str
    session_lineage_id: str
    request: OperationsContextRequest
    generated_at_ms: int
    source_turn_index: int
    rendered_context: str
    recalled_experiences: tuple[OperationsRecalledExperience, ...]
    source_entry_ids: tuple[str, ...]
    source_evidence_ref_ids: tuple[str, ...]
    retrieval_facets: tuple[str, ...]
    memory_entry_count: int
    truncated: bool
    current_uncertainties: tuple[OperationsUncertainty, ...]
    settled_outcome_ids: tuple[str, ...]
    settled_evidence_ref_ids: tuple[str, ...]
    pe_magnitude: float
    pe_bootstrap: bool
    advice: OperationsAdviceSnapshot
    settled_policy_credits: tuple[OperationsPolicyCredit, ...] = ()
    policy_updates: tuple[OperationsPolicyUpdateReceipt, ...] = ()
    wiring_level: WiringLevel = WiringLevel.ACTIVE

    def __post_init__(self) -> None:
        _require_content_id(
            "context_pack_id",
            self.context_pack_id,
            prefix="operations-context-pack:",
        )
        _require_sha256("content_sha256", self.content_sha256)
        if self.context_pack_id != f"operations-context-pack:{self.content_sha256}":
            raise ValueError("context_pack_id must match content_sha256")
        _require_text("session_id", self.session_id, max_length=256)
        _require_content_id(
            "session_lineage_id",
            self.session_lineage_id,
            prefix="operations-live-session:",
        )
        if not isinstance(self.request, OperationsContextRequest):
            raise ValueError("request must be an OperationsContextRequest")
        _require_non_negative_int("generated_at_ms", self.generated_at_ms)
        _require_non_negative_int("source_turn_index", self.source_turn_index)
        if not isinstance(self.rendered_context, str):
            raise ValueError("rendered_context must be a string")
        if len(self.rendered_context) > self.request.max_context_chars:
            raise ValueError("rendered_context exceeds request max_context_chars")
        _require_typed_tuple(
            "recalled_experiences",
            self.recalled_experiences,
            item_type=OperationsRecalledExperience,
            max_items=20,
        )
        _require_string_tuple(
            "source_entry_ids",
            self.source_entry_ids,
            max_items=20,
            item_max_length=256,
        )
        if self.source_entry_ids != tuple(experience.memory_entry_id for experience in self.recalled_experiences):
            raise ValueError("source_entry_ids must match recalled experience order")
        for name, values, maximum in (
            ("source_evidence_ref_ids", self.source_evidence_ref_ids, 256),
            ("retrieval_facets", self.retrieval_facets, 256),
            ("settled_outcome_ids", self.settled_outcome_ids, 16),
            ("settled_evidence_ref_ids", self.settled_evidence_ref_ids, 128),
        ):
            _require_string_tuple(
                name,
                values,
                max_items=maximum,
                item_max_length=1_024,
            )
        _require_non_negative_int("memory_entry_count", self.memory_entry_count)
        if not isinstance(self.truncated, bool):
            raise ValueError("truncated must be a boolean")
        _require_typed_tuple(
            "current_uncertainties",
            self.current_uncertainties,
            item_type=OperationsUncertainty,
            max_items=32,
        )
        if self.current_uncertainties != self.request.uncertainties:
            raise ValueError("current_uncertainties must preserve request uncertainty")
        _require_numeric("pe_magnitude", self.pe_magnitude)
        if not isinstance(self.pe_bootstrap, bool):
            raise ValueError("pe_bootstrap must be a boolean")
        if not isinstance(self.advice, OperationsAdviceSnapshot):
            raise ValueError("advice must be an OperationsAdviceSnapshot")
        _require_typed_tuple(
            "settled_policy_credits",
            self.settled_policy_credits,
            item_type=OperationsPolicyCredit,
            max_items=4,
        )
        _require_typed_tuple(
            "policy_updates",
            self.policy_updates,
            item_type=OperationsPolicyUpdateReceipt,
            max_items=4,
        )
        if len(self.settled_policy_credits) != len(self.policy_updates):
            raise ValueError("policy credit/update counts must match")
        for credit, update in zip(
            self.settled_policy_credits,
            self.policy_updates,
            strict=True,
        ):
            if update.credit_id != credit.credit_id:
                raise ValueError("policy update credit lineage mismatch")
            if credit.environment_outcome_id not in self.settled_outcome_ids:
                raise ValueError("policy credit must reference a settled outcome")
        if self.request.operations_state is None:
            if self.advice.policy_decision is not None:
                raise ValueError("v1 Context Pack cannot publish policy advice")
            if self.settled_policy_credits or self.policy_updates:
                raise ValueError("v1 Context Pack cannot publish policy settlement")
        elif self.advice.policy_decision is None:
            raise ValueError("v2 Context Pack requires policy advice")
        if self.wiring_level is not WiringLevel.ACTIVE:
            raise ValueError("Operations Context Pack must be WiringLevel.ACTIVE")

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "OperationsContextPackSnapshot":
        fields = frozenset(
            {
                "schema_version",
                "context_pack_id",
                "content_sha256",
                "session_id",
                "session_lineage_id",
                "request",
                "generated_at_ms",
                "source_turn_index",
                "rendered_context",
                "recalled_experiences",
                "source_entry_ids",
                "source_evidence_ref_ids",
                "retrieval_facets",
                "memory_entry_count",
                "truncated",
                "current_uncertainties",
                "settled_outcome_ids",
                "settled_evidence_ref_ids",
                "pe_magnitude",
                "pe_bootstrap",
                "advice",
                "settled_policy_credits",
                "policy_updates",
                "wiring_level",
            }
        )
        v2_fields = frozenset({"settled_policy_credits", "policy_updates"})
        _strict_payload(payload, allowed=fields, required=fields - v2_fields)
        schema_version = payload["schema_version"]
        if schema_version not in {
            CONTEXT_PACK_SCHEMA_VERSION,
            CONTEXT_PACK_V2_SCHEMA_VERSION,
        }:
            raise ValueError("unsupported Operations Context Pack schema")
        if schema_version == CONTEXT_PACK_SCHEMA_VERSION and set(payload) & v2_fields:
            raise ValueError("policy settlement fields require Context Pack v2")
        if schema_version == CONTEXT_PACK_V2_SCHEMA_VERSION and not v2_fields.issubset(payload):
            raise ValueError("Context Pack v2 requires policy settlement fields")
        snapshot = cls(
            context_pack_id=payload["context_pack_id"],
            content_sha256=payload["content_sha256"],
            session_id=payload["session_id"],
            session_lineage_id=payload["session_lineage_id"],
            request=OperationsContextRequest.from_json(_mapping("request", payload["request"])),
            generated_at_ms=payload["generated_at_ms"],
            source_turn_index=payload["source_turn_index"],
            rendered_context=payload["rendered_context"],
            recalled_experiences=tuple(
                OperationsRecalledExperience.from_json(_mapping("recalled_experiences[]", item))
                for item in _array("recalled_experiences", payload["recalled_experiences"])
            ),
            source_entry_ids=tuple(_array("source_entry_ids", payload["source_entry_ids"])),
            source_evidence_ref_ids=tuple(_array("source_evidence_ref_ids", payload["source_evidence_ref_ids"])),
            retrieval_facets=tuple(_array("retrieval_facets", payload["retrieval_facets"])),
            memory_entry_count=payload["memory_entry_count"],
            truncated=payload["truncated"],
            current_uncertainties=tuple(
                OperationsUncertainty.from_json(_mapping("current_uncertainties[]", item))
                for item in _array("current_uncertainties", payload["current_uncertainties"])
            ),
            settled_outcome_ids=tuple(_array("settled_outcome_ids", payload["settled_outcome_ids"])),
            settled_evidence_ref_ids=tuple(_array("settled_evidence_ref_ids", payload["settled_evidence_ref_ids"])),
            pe_magnitude=payload["pe_magnitude"],
            pe_bootstrap=payload["pe_bootstrap"],
            advice=OperationsAdviceSnapshot.from_json(_mapping("advice", payload["advice"])),
            settled_policy_credits=tuple(
                OperationsPolicyCredit.from_json(
                    _mapping("settled_policy_credits[]", item)
                )
                for item in _array(
                    "settled_policy_credits",
                    payload.get("settled_policy_credits", ()),
                )
            ),
            policy_updates=tuple(
                OperationsPolicyUpdateReceipt.from_json(
                    _mapping("policy_updates[]", item)
                )
                for item in _array(
                    "policy_updates",
                    payload.get("policy_updates", ()),
                )
            ),
            wiring_level=_closed_enum(WiringLevel, "wiring_level", payload["wiring_level"]),
        )
        is_v2 = snapshot.request.operations_state is not None
        if is_v2 != (schema_version == CONTEXT_PACK_V2_SCHEMA_VERSION):
            raise ValueError("Operations Context Pack/request schema versions drift")
        if is_v2 != (snapshot.advice.policy_decision is not None):
            raise ValueError("Operations Context Pack v2 requires policy advice")
        digest_payload = snapshot.to_json()
        digest_payload.pop("context_pack_id")
        digest_payload.pop("content_sha256")
        if stable_content_sha256(digest_payload) != snapshot.content_sha256:
            raise ValueError("Operations Context Pack content_sha256 does not match its payload")
        return snapshot

    def to_json(self) -> dict[str, object]:
        payload = {
            "schema_version": (
                CONTEXT_PACK_V2_SCHEMA_VERSION
                if self.request.operations_state is not None
                else CONTEXT_PACK_SCHEMA_VERSION
            ),
            "context_pack_id": self.context_pack_id,
            "content_sha256": self.content_sha256,
            "session_id": self.session_id,
            "session_lineage_id": self.session_lineage_id,
            "request": self.request.to_json(),
            "generated_at_ms": self.generated_at_ms,
            "source_turn_index": self.source_turn_index,
            "rendered_context": self.rendered_context,
            "recalled_experiences": [item.to_json() for item in self.recalled_experiences],
            "source_entry_ids": list(self.source_entry_ids),
            "source_evidence_ref_ids": list(self.source_evidence_ref_ids),
            "retrieval_facets": list(self.retrieval_facets),
            "memory_entry_count": self.memory_entry_count,
            "truncated": self.truncated,
            "current_uncertainties": [item.to_json() for item in self.current_uncertainties],
            "settled_outcome_ids": list(self.settled_outcome_ids),
            "settled_evidence_ref_ids": list(self.settled_evidence_ref_ids),
            "pe_magnitude": float(self.pe_magnitude),
            "pe_bootstrap": self.pe_bootstrap,
            "advice": self.advice.to_json(),
            "wiring_level": self.wiring_level.value,
        }
        if self.request.operations_state is not None:
            payload["settled_policy_credits"] = [
                item.to_json() for item in self.settled_policy_credits
            ]
            payload["policy_updates"] = [
                item.to_json() for item in self.policy_updates
            ]
        return payload


@dataclass(frozen=True)
class OperationsOutcomeReceipt:
    receipt_id: str
    content_sha256: str
    session_id: str
    session_lineage_id: str
    company_id: str
    cycle_id: str
    workstream_id: str
    decision_id: str
    work_order_ref: str
    report: OperationsOutcomeReport
    action_turn_index: int
    source_advice_id: str
    source_advice_applied: bool
    memory_entry_id: str
    memory_persisted: bool
    task_event_ids: tuple[str, ...]
    environment_outcome_id: str
    learning_route: OperationsOutcomeRoute
    settlement_state: OperationsSettlementState
    source_policy_decision_id: str = ""
    selection_id: str = ""
    selected_candidate_id: str = ""

    def __post_init__(self) -> None:
        _require_content_id(
            "receipt_id",
            self.receipt_id,
            prefix="operations-outcome-receipt:",
        )
        for name, value in (
            ("session_id", self.session_id),
            ("company_id", self.company_id),
            ("cycle_id", self.cycle_id),
            ("decision_id", self.decision_id),
            ("memory_entry_id", self.memory_entry_id),
        ):
            _require_text(name, value, max_length=256)
        _require_text("work_order_ref", self.work_order_ref, max_length=512)
        _require_content_id(
            "source_advice_id",
            self.source_advice_id,
            prefix="operations-advice:",
        )
        _require_content_id(
            "session_lineage_id",
            self.session_lineage_id,
            prefix="operations-live-session:",
        )
        _require_optional_text("workstream_id", self.workstream_id, max_length=256)
        _require_sha256("content_sha256", self.content_sha256)
        if self.receipt_id != f"operations-outcome-receipt:{self.content_sha256}":
            raise ValueError("receipt_id must match content_sha256")
        if not isinstance(self.report, OperationsOutcomeReport):
            raise ValueError("report must be an OperationsOutcomeReport")
        if self.work_order_ref != self.report.work_order_ref:
            raise ValueError("work_order_ref must match report lineage")
        _require_non_negative_int("action_turn_index", self.action_turn_index)
        if not isinstance(self.source_advice_applied, bool):
            raise ValueError("source_advice_applied must be a boolean")
        if self.report.source_advice_id:
            if self.source_advice_id != self.report.source_advice_id:
                raise ValueError("receipt/report source_advice_id mismatch")
            if self.source_policy_decision_id != self.report.policy_decision_id:
                raise ValueError("receipt/report policy_decision_id mismatch")
            if self.selection_id != self.report.selection_id:
                raise ValueError("receipt/report selection_id mismatch")
            if self.selected_candidate_id != self.report.selected_candidate_id:
                raise ValueError("receipt/report selected_candidate_id mismatch")
            if (
                self.source_advice_applied
                is not self.report.policy_action_applied
            ):
                raise ValueError("receipt/report policy_action_applied mismatch")
        elif any(
            (
                self.source_policy_decision_id,
                self.selection_id,
                self.selected_candidate_id,
            )
        ) or self.source_advice_applied:
            raise ValueError("v1 receipts cannot claim policy selection lineage")
        if not isinstance(self.memory_persisted, bool):
            raise ValueError("memory_persisted must be a boolean")
        _require_string_tuple(
            "task_event_ids",
            self.task_event_ids,
            max_items=16,
            item_max_length=512,
        )
        _require_optional_text("environment_outcome_id", self.environment_outcome_id, max_length=512)
        if not isinstance(self.learning_route, OperationsOutcomeRoute):
            raise ValueError("learning_route must be an OperationsOutcomeRoute")
        if not isinstance(self.settlement_state, OperationsSettlementState):
            raise ValueError("settlement_state must be an OperationsSettlementState")
        if self.report.pe_eligible:
            if (
                self.learning_route
                is not OperationsOutcomeRoute.FIELD_OPERATION_PE_MEMORY_AND_EXECUTION_RESULT
            ):
                raise ValueError("field_operation_result must use the PE route")
            if not self.environment_outcome_id:
                raise ValueError("PE-eligible reports require environment outcome lineage")
            if self.settlement_state is not OperationsSettlementState.PENDING_NEXT_CONTEXT_TURN:
                raise ValueError("PE-eligible reports must remain pending until next turn")
        else:
            if self.learning_route is not OperationsOutcomeRoute.MEMORY_AND_EXECUTION_RESULT:
                raise ValueError("non-eligible reports cannot use the PE route")
            if self.environment_outcome_id:
                raise ValueError("non-eligible reports cannot publish environment outcomes")
            if self.settlement_state is not OperationsSettlementState.NOT_PE_ELIGIBLE:
                raise ValueError("non-eligible reports require not_pe_eligible state")

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "OperationsOutcomeReceipt":
        fields = frozenset(
            {
                "schema_version",
                "receipt_id",
                "content_sha256",
                "session_id",
                "session_lineage_id",
                "company_id",
                "cycle_id",
                "workstream_id",
                "decision_id",
                "work_order_ref",
                "report",
                "action_turn_index",
                "source_advice_id",
                "source_advice_applied",
                "memory_entry_id",
                "memory_persisted",
                "task_event_ids",
                "environment_outcome_id",
                "learning_route",
                "settlement_state",
                "source_policy_decision_id",
                "selection_id",
                "selected_candidate_id",
            }
        )
        v2_fields = frozenset(
            {
                "source_policy_decision_id",
                "selection_id",
                "selected_candidate_id",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields - v2_fields)
        schema_version = payload["schema_version"]
        if schema_version not in {
            OUTCOME_RECEIPT_SCHEMA_VERSION,
            OUTCOME_RECEIPT_V2_SCHEMA_VERSION,
        }:
            raise ValueError("unsupported Operations Outcome Receipt schema")
        if schema_version == OUTCOME_RECEIPT_SCHEMA_VERSION and set(payload) & v2_fields:
            raise ValueError("policy lineage requires operations-outcome-receipt.v2")
        if schema_version == OUTCOME_RECEIPT_V2_SCHEMA_VERSION and not v2_fields.issubset(payload):
            raise ValueError("operations-outcome-receipt.v2 requires policy lineage")
        receipt = cls(
            receipt_id=payload["receipt_id"],
            content_sha256=payload["content_sha256"],
            session_id=payload["session_id"],
            session_lineage_id=payload["session_lineage_id"],
            company_id=payload["company_id"],
            cycle_id=payload["cycle_id"],
            workstream_id=payload["workstream_id"],
            decision_id=payload["decision_id"],
            work_order_ref=payload["work_order_ref"],
            report=OperationsOutcomeReport.from_json(_mapping("report", payload["report"])),
            action_turn_index=payload["action_turn_index"],
            source_advice_id=payload["source_advice_id"],
            source_advice_applied=payload["source_advice_applied"],
            memory_entry_id=payload["memory_entry_id"],
            memory_persisted=payload["memory_persisted"],
            task_event_ids=tuple(_array("task_event_ids", payload["task_event_ids"])),
            environment_outcome_id=payload["environment_outcome_id"],
            learning_route=_closed_enum(
                OperationsOutcomeRoute,
                "learning_route",
                payload["learning_route"],
            ),
            settlement_state=_closed_enum(
                OperationsSettlementState,
                "settlement_state",
                payload["settlement_state"],
            ),
            source_policy_decision_id=payload.get(
                "source_policy_decision_id",
                "",
            ),
            selection_id=payload.get("selection_id", ""),
            selected_candidate_id=payload.get("selected_candidate_id", ""),
        )
        digest_payload = receipt.to_json()
        digest_payload.pop("receipt_id")
        digest_payload.pop("content_sha256")
        if stable_content_sha256(digest_payload) != receipt.content_sha256:
            raise ValueError("Operations Outcome Receipt content_sha256 does not match its payload")
        return receipt

    def to_json(self) -> dict[str, object]:
        payload = {
            "schema_version": (
                OUTCOME_RECEIPT_V2_SCHEMA_VERSION
                if self.source_policy_decision_id
                else OUTCOME_RECEIPT_SCHEMA_VERSION
            ),
            "receipt_id": self.receipt_id,
            "content_sha256": self.content_sha256,
            "session_id": self.session_id,
            "session_lineage_id": self.session_lineage_id,
            "company_id": self.company_id,
            "cycle_id": self.cycle_id,
            "workstream_id": self.workstream_id,
            "decision_id": self.decision_id,
            "work_order_ref": self.work_order_ref,
            "report": self.report.to_json(),
            "action_turn_index": self.action_turn_index,
            "source_advice_id": self.source_advice_id,
            "source_advice_applied": self.source_advice_applied,
            "memory_entry_id": self.memory_entry_id,
            "memory_persisted": self.memory_persisted,
            "task_event_ids": list(self.task_event_ids),
            "environment_outcome_id": self.environment_outcome_id,
            "learning_route": self.learning_route.value,
            "settlement_state": self.settlement_state.value,
        }
        if self.source_policy_decision_id:
            payload.update(
                {
                    "source_policy_decision_id": self.source_policy_decision_id,
                    "selection_id": self.selection_id,
                    "selected_candidate_id": self.selected_candidate_id,
                }
            )
        return payload


__all__ = (
    "ADVICE_SCHEMA_VERSION",
    "ADVICE_V2_SCHEMA_VERSION",
    "CONTEXT_PACK_SCHEMA_VERSION",
    "CONTEXT_PACK_V2_SCHEMA_VERSION",
    "CONTEXT_REQUEST_SCHEMA_VERSION",
    "CONTEXT_REQUEST_V2_SCHEMA_VERSION",
    "EXPERIENCE_RECORD_SCHEMA_VERSION",
    "OUTCOME_RECEIPT_SCHEMA_VERSION",
    "OUTCOME_RECEIPT_V2_SCHEMA_VERSION",
    "OUTCOME_REPORT_SCHEMA_VERSION",
    "OUTCOME_REPORT_V2_SCHEMA_VERSION",
    "OPERATIONS_POLICY_FEATURE_ORDER",
    "OperationsAdviceCandidate",
    "OperationsAdviceKind",
    "OperationsAdviceSnapshot",
    "OperationsExecutionOutcome",
    "OperationsConstraint",
    "OperationsConstraintKind",
    "OperationsContextPackSnapshot",
    "OperationsContextRequest",
    "OperationsCostBreakdown",
    "OperationsObjectiveResult",
    "OperationsDecisionKind",
    "OperationsDecisionPoint",
    "OperationsEstimateRange",
    "OperationsEvidenceClass",
    "OperationsEvidenceRef",
    "OperationsEvidenceRole",
    "OperationsFact",
    "OperationsFactKind",
    "OperationsGoalState",
    "OperationsIncidentSeverity",
    "OperationsIncidentState",
    "OperationsMetricObservation",
    "OperationsOutcomeKind",
    "OperationsOutcomeReceipt",
    "OperationsOutcomeReport",
    "OperationsOutcomeRoute",
    "OperationsOutcomeVerdict",
    "OperationsPolicyAction",
    "OperationsPolicyCheckpoint",
    "OperationsPolicyCredit",
    "OperationsPolicyDecision",
    "OperationsPolicyMode",
    "OperationsPolicyUpdateReceipt",
    "OperationsRankedCandidate",
    "OperationsRecentOutcomeState",
    "OperationsRecalledExperience",
    "OperationsOperatingWindow",
    "OperationsReversibility",
    "OperationsRiskLevel",
    "OperationsSettlementState",
    "OperationsStateSnapshot",
    "OperationsUncertainty",
    "OperationsDependencyState",
    "OperationsDivisionState",
    "OperationsWorkItemState",
    "OperationsWorkItemStatus",
    "stable_content_sha256",
)
