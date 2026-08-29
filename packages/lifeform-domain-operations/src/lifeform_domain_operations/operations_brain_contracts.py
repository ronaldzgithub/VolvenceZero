"""Strict frozen contracts for the AutoCompany-facing Operations Brain v1.

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
        encoded = value.encode("utf-8")
        return b"s" + _canonical_length(len(encoded)) + encoded
    if isinstance(value, (int, float)):
        if isinstance(value, int) and abs(value) > _MAX_SAFE_INTEGER:
            raise ValueError("canonical integer exceeds IEEE-754 safe range")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("canonical numbers must be finite")
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
            items.append((key.encode("utf-8"), key, item))
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
        if payload["schema_version"] != CONTEXT_REQUEST_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {CONTEXT_REQUEST_SCHEMA_VERSION!r}")
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
        )

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": CONTEXT_REQUEST_SCHEMA_VERSION,
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
class OperationsAdviceSnapshot:
    advice_id: str
    source_turn_index: int
    candidate_regime_id: str
    candidate_abstract_action: str
    candidates: tuple[OperationsAdviceCandidate, ...]
    rationale: str
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
        if self.wiring_level is not WiringLevel.SHADOW:
            raise ValueError("Operations advice must remain WiringLevel.SHADOW")
        if not isinstance(self.applied, bool):
            raise ValueError("applied must be a boolean")
        if self.applied:
            raise ValueError("SHADOW Operations advice cannot be applied")

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
                "wiring_level",
                "applied",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        if payload["schema_version"] != ADVICE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {ADVICE_SCHEMA_VERSION!r}")
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
            wiring_level=_closed_enum(WiringLevel, "wiring_level", payload["wiring_level"]),
            applied=payload["applied"],
        )

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": ADVICE_SCHEMA_VERSION,
            "advice_id": self.advice_id,
            "source_turn_index": self.source_turn_index,
            "candidate_regime_id": self.candidate_regime_id,
            "candidate_abstract_action": self.candidate_abstract_action,
            "candidates": [candidate.to_json() for candidate in self.candidates],
            "rationale": self.rationale,
            "wiring_level": self.wiring_level.value,
            "applied": self.applied,
        }


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
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        if payload["schema_version"] != OUTCOME_REPORT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {OUTCOME_REPORT_SCHEMA_VERSION!r}")
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
        )

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": OUTCOME_REPORT_SCHEMA_VERSION,
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
                "wiring_level",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        if payload["schema_version"] != CONTEXT_PACK_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {CONTEXT_PACK_SCHEMA_VERSION!r}")
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
            wiring_level=_closed_enum(WiringLevel, "wiring_level", payload["wiring_level"]),
        )
        digest_payload = snapshot.to_json()
        digest_payload.pop("context_pack_id")
        digest_payload.pop("content_sha256")
        if stable_content_sha256(digest_payload) != snapshot.content_sha256:
            raise ValueError("Operations Context Pack content_sha256 does not match its payload")
        return snapshot

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": CONTEXT_PACK_SCHEMA_VERSION,
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
        if self.source_advice_applied:
            raise ValueError("Operations v1 receipts cannot claim SHADOW advice was applied")
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
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        if payload["schema_version"] != OUTCOME_RECEIPT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {OUTCOME_RECEIPT_SCHEMA_VERSION!r}")
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
        )
        digest_payload = receipt.to_json()
        digest_payload.pop("receipt_id")
        digest_payload.pop("content_sha256")
        if stable_content_sha256(digest_payload) != receipt.content_sha256:
            raise ValueError("Operations Outcome Receipt content_sha256 does not match its payload")
        return receipt

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": OUTCOME_RECEIPT_SCHEMA_VERSION,
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


__all__ = (
    "ADVICE_SCHEMA_VERSION",
    "CONTEXT_PACK_SCHEMA_VERSION",
    "CONTEXT_REQUEST_SCHEMA_VERSION",
    "EXPERIENCE_RECORD_SCHEMA_VERSION",
    "OUTCOME_RECEIPT_SCHEMA_VERSION",
    "OUTCOME_REPORT_SCHEMA_VERSION",
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
    "OperationsMetricObservation",
    "OperationsOutcomeKind",
    "OperationsOutcomeReceipt",
    "OperationsOutcomeReport",
    "OperationsOutcomeRoute",
    "OperationsOutcomeVerdict",
    "OperationsRecalledExperience",
    "OperationsOperatingWindow",
    "OperationsReversibility",
    "OperationsRiskLevel",
    "OperationsSettlementState",
    "OperationsUncertainty",
    "stable_content_sha256",
)
