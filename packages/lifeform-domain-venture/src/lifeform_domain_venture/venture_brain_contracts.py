"""Strict frozen contracts for the Foundry-facing Venture Brain v1.

Evidence class, outcome route, and business verdict are explicit protocol
fields supplied by Foundry. Free text is context only and is never parsed to
upgrade evidence, choose a route, or make a commercial decision.
"""

from __future__ import annotations

import hashlib
import json
import math
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


CONTEXT_REQUEST_SCHEMA_VERSION = "venture-context-request.v1"
CONTEXT_PACK_SCHEMA_VERSION = "venture-context-pack.v1"
ADVICE_SCHEMA_VERSION = "venture-advice.v1"
OUTCOME_REPORT_SCHEMA_VERSION = "venture-outcome-report.v1"
OUTCOME_RECEIPT_SCHEMA_VERSION = "venture-outcome-receipt.v1"
EXPERIENCE_RECORD_SCHEMA_VERSION = "venture-experience-record.v1"


class VentureDecisionPoint(str, Enum):
    OPPORTUNITY_BRAINSTORM = "opportunity_brainstorm"
    CANDIDATE_COMPARISON = "candidate_comparison"
    EXPERIMENT_PLANNING = "experiment_planning"
    PORTFOLIO_REVIEW = "portfolio_review"
    MONITOR_ATTRIBUTION = "monitor_attribution"
    STOP_REVIEW = "stop_review"


class VentureFactKind(str, Enum):
    DEMAND_SIGNAL = "demand_signal"
    OPPORTUNITY = "opportunity"
    CANDIDATE = "candidate"
    PRODUCT = "product"
    EXPERIMENT = "experiment"
    PORTFOLIO = "portfolio"
    COMMERCIAL = "commercial"
    OPERATIONAL = "operational"
    RISK = "risk"
    OTHER = "other"


class VentureConstraintKind(str, Enum):
    BUDGET = "budget"
    TIME = "time"
    SAFETY = "safety"
    LEGAL = "legal"
    PORTFOLIO = "portfolio"
    PRODUCT_SHAPE = "product_shape"
    EVIDENCE = "evidence"
    REVERSIBILITY = "reversibility"
    OTHER = "other"


class VentureEvidenceClass(str, Enum):
    SIMULATION = "simulation"
    INTERNAL_REVIEW = "internal_review"
    MACHINE_CHECK = "machine_check"
    FIELD = "field"


class VentureEvidenceRole(str, Enum):
    DEMAND_SIGNAL = "demand_signal"
    CONSTRAINT = "constraint"
    DECISION_RECORD = "decision_record"
    EXPERIMENT = "experiment"
    INTERNAL_REVIEW = "internal_review"
    MACHINE_AUDIT = "machine_audit"
    FIELD_OBSERVATION = "field_observation"
    CUSTOMER_OUTCOME = "customer_outcome"
    PAYMENT = "payment"
    COST = "cost"
    REFUND = "refund"


class VentureAdviceKind(str, Enum):
    OPPORTUNITY = "opportunity"
    COMPARISON = "comparison"
    EXPERIMENT = "experiment"
    STOP = "stop"


class VentureOutcomeKind(str, Enum):
    SIMULATION_RESULT = "simulation_result"
    INTERNAL_REVIEW_RESULT = "internal_review_result"
    MACHINE_CHECK_RESULT = "machine_check_result"
    CUSTOMER_OUTCOME = "customer_outcome"
    PAYMENT_RECEIVED = "payment_received"
    COST_RECORDED = "cost_recorded"
    REFUND_RECORDED = "refund_recorded"
    FIELD_EXPERIMENT_RESULT = "field_experiment_result"


class VentureDecisionKind(str, Enum):
    RUN_EXPERIMENT = "run_experiment"
    CONTINUE = "continue"
    PAUSE = "pause"
    STOP = "stop"
    SCALE = "scale"
    KILL = "kill"
    NO_STATE_CHANGE = "no_state_change"


class VentureOutcomeVerdict(str, Enum):
    FAVORABLE = "favorable"
    UNFAVORABLE = "unfavorable"
    MIXED = "mixed"
    INCONCLUSIVE = "inconclusive"


class VentureCustomerResult(str, Enum):
    NOT_OBSERVED = "not_observed"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    MIXED = "mixed"


class VentureRiskLevel(str, Enum):
    UNASSESSED = "unassessed"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VentureReversibility(str, Enum):
    REVERSIBLE = "reversible"
    COSTLY = "costly"
    IRREVERSIBLE = "irreversible"


class VentureOutcomeRoute(str, Enum):
    MEMORY_AND_EXECUTION_RESULT = "memory_and_execution_result"
    FIELD_PE_MEMORY_AND_EXECUTION_RESULT = "field_pe_memory_and_execution_result"


class VentureSettlementState(str, Enum):
    NOT_PE_ELIGIBLE = "not_pe_eligible"
    PENDING_NEXT_CONTEXT_TURN = "pending_next_context_turn"


_OUTCOME_CLASS_PAIRS: dict[VentureEvidenceClass, frozenset[VentureOutcomeKind]] = {
    VentureEvidenceClass.SIMULATION: frozenset({VentureOutcomeKind.SIMULATION_RESULT}),
    VentureEvidenceClass.INTERNAL_REVIEW: frozenset({VentureOutcomeKind.INTERNAL_REVIEW_RESULT}),
    VentureEvidenceClass.MACHINE_CHECK: frozenset({VentureOutcomeKind.MACHINE_CHECK_RESULT}),
    VentureEvidenceClass.FIELD: frozenset(
        {
            VentureOutcomeKind.CUSTOMER_OUTCOME,
            VentureOutcomeKind.PAYMENT_RECEIVED,
            VentureOutcomeKind.COST_RECORDED,
            VentureOutcomeKind.REFUND_RECORDED,
            VentureOutcomeKind.FIELD_EXPERIMENT_RESULT,
        }
    ),
}

_SIMULATION_FORBIDDEN_ROLES = frozenset(
    {
        VentureEvidenceRole.INTERNAL_REVIEW,
        VentureEvidenceRole.MACHINE_AUDIT,
        VentureEvidenceRole.FIELD_OBSERVATION,
        VentureEvidenceRole.CUSTOMER_OUTCOME,
        VentureEvidenceRole.PAYMENT,
        VentureEvidenceRole.COST,
        VentureEvidenceRole.REFUND,
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


def stable_content_sha256(payload: Mapping[str, object]) -> str:
    """Return the canonical digest used by Venture Brain lineage IDs."""

    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class VentureEvidenceRef:
    ref_id: str
    evidence_class: VentureEvidenceClass
    role: VentureEvidenceRole
    locator: str
    content_sha256: str
    observed_at_ms: int

    def __post_init__(self) -> None:
        _require_text("ref_id", self.ref_id, max_length=256)
        if not isinstance(self.evidence_class, VentureEvidenceClass):
            raise ValueError("evidence_class must be a VentureEvidenceClass")
        if not isinstance(self.role, VentureEvidenceRole):
            raise ValueError("role must be a VentureEvidenceRole")
        _require_text("locator", self.locator, max_length=2_048)
        _require_sha256("content_sha256", self.content_sha256)
        _require_non_negative_int("observed_at_ms", self.observed_at_ms)
        if (
            self.evidence_class is VentureEvidenceClass.INTERNAL_REVIEW
            and self.role is not VentureEvidenceRole.INTERNAL_REVIEW
        ):
            raise ValueError("internal_review evidence requires role=internal_review")
        if (
            self.evidence_class is VentureEvidenceClass.MACHINE_CHECK
            and self.role is not VentureEvidenceRole.MACHINE_AUDIT
        ):
            raise ValueError("machine_check evidence requires role=machine_audit")
        if self.evidence_class is VentureEvidenceClass.SIMULATION and self.role in _SIMULATION_FORBIDDEN_ROLES:
            raise ValueError("simulation evidence cannot use review, audit, or field-only roles")
        if self.evidence_class is VentureEvidenceClass.FIELD and self.role in {
            VentureEvidenceRole.INTERNAL_REVIEW,
            VentureEvidenceRole.MACHINE_AUDIT,
        }:
            raise ValueError("field evidence cannot use internal review or machine audit roles")

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "VentureEvidenceRef":
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
                VentureEvidenceClass,
                "evidence_class",
                payload["evidence_class"],
            ),
            role=_closed_enum(VentureEvidenceRole, "role", payload["role"]),
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
class VentureFact:
    fact_id: str
    kind: VentureFactKind
    statement: str
    evidence_ref_ids: tuple[str, ...]
    as_of_ms: int

    def __post_init__(self) -> None:
        _require_text("fact_id", self.fact_id, max_length=256)
        if not isinstance(self.kind, VentureFactKind):
            raise ValueError("kind must be a VentureFactKind")
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
    def from_json(cls, payload: Mapping[str, object]) -> "VentureFact":
        fields = frozenset({"fact_id", "kind", "statement", "evidence_ref_ids", "as_of_ms"})
        _strict_payload(payload, allowed=fields, required=fields)
        return cls(
            fact_id=payload["fact_id"],
            kind=_closed_enum(VentureFactKind, "kind", payload["kind"]),
            statement=payload["statement"],
            evidence_ref_ids=tuple(_array("evidence_ref_ids", payload["evidence_ref_ids"])),
            as_of_ms=payload["as_of_ms"],
        )

    def to_json(self) -> dict[str, object]:
        return {
            "fact_id": self.fact_id,
            "kind": self.kind.value,
            "statement": self.statement,
            "evidence_ref_ids": list(self.evidence_ref_ids),
            "as_of_ms": self.as_of_ms,
        }


@dataclass(frozen=True)
class VentureConstraint:
    constraint_id: str
    kind: VentureConstraintKind
    description: str
    hard: bool

    def __post_init__(self) -> None:
        _require_text("constraint_id", self.constraint_id, max_length=256)
        if not isinstance(self.kind, VentureConstraintKind):
            raise ValueError("kind must be a VentureConstraintKind")
        _require_text("description", self.description, max_length=2_000)
        if not isinstance(self.hard, bool):
            raise ValueError("hard must be a boolean")

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "VentureConstraint":
        fields = frozenset({"constraint_id", "kind", "description", "hard"})
        _strict_payload(payload, allowed=fields, required=fields)
        return cls(
            constraint_id=payload["constraint_id"],
            kind=_closed_enum(VentureConstraintKind, "kind", payload["kind"]),
            description=payload["description"],
            hard=payload["hard"],
        )

    def to_json(self) -> dict[str, object]:
        return {
            "constraint_id": self.constraint_id,
            "kind": self.kind.value,
            "description": self.description,
            "hard": self.hard,
        }


@dataclass(frozen=True)
class VentureUncertainty:
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
    def from_json(cls, payload: Mapping[str, object]) -> "VentureUncertainty":
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
class VentureResourceWindow:
    currency: str
    maximum_total_cost_minor: int
    starts_at_ms: int
    ends_at_ms: int
    maximum_experiments: int

    def __post_init__(self) -> None:
        _require_currency(self.currency)
        _require_non_negative_int("maximum_total_cost_minor", self.maximum_total_cost_minor)
        start = _require_non_negative_int("starts_at_ms", self.starts_at_ms)
        end = _require_non_negative_int("ends_at_ms", self.ends_at_ms)
        if end <= start:
            raise ValueError("ends_at_ms must be greater than starts_at_ms")
        maximum = _require_non_negative_int("maximum_experiments", self.maximum_experiments)
        if maximum < 1:
            raise ValueError("maximum_experiments must be positive")

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "VentureResourceWindow":
        fields = frozenset(
            {
                "currency",
                "maximum_total_cost_minor",
                "starts_at_ms",
                "ends_at_ms",
                "maximum_experiments",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        return cls(
            currency=payload["currency"],
            maximum_total_cost_minor=payload["maximum_total_cost_minor"],
            starts_at_ms=payload["starts_at_ms"],
            ends_at_ms=payload["ends_at_ms"],
            maximum_experiments=payload["maximum_experiments"],
        )

    def to_json(self) -> dict[str, object]:
        return {
            "currency": self.currency,
            "maximum_total_cost_minor": self.maximum_total_cost_minor,
            "starts_at_ms": self.starts_at_ms,
            "ends_at_ms": self.ends_at_ms,
            "maximum_experiments": self.maximum_experiments,
        }


@dataclass(frozen=True)
class VentureContextRequest:
    request_id: str
    portfolio_id: str
    cycle_id: str
    venture_id: str
    decision_id: str
    decision_point: VentureDecisionPoint
    confirmed_facts: tuple[VentureFact, ...]
    constraints: tuple[VentureConstraint, ...]
    resource_window: VentureResourceWindow
    uncertainties: tuple[VentureUncertainty, ...]
    evidence_refs: tuple[VentureEvidenceRef, ...]
    memory_limit: int = 8
    max_context_chars: int = 8_000

    def __post_init__(self) -> None:
        for name, value in (
            ("request_id", self.request_id),
            ("portfolio_id", self.portfolio_id),
            ("cycle_id", self.cycle_id),
            ("decision_id", self.decision_id),
        ):
            _require_text(name, value, max_length=256)
        _require_optional_text("venture_id", self.venture_id, max_length=256)
        if not isinstance(self.decision_point, VentureDecisionPoint):
            raise ValueError("decision_point must be a VentureDecisionPoint")
        _require_typed_tuple(
            "confirmed_facts",
            self.confirmed_facts,
            item_type=VentureFact,
            max_items=64,
            allow_empty=False,
        )
        _require_typed_tuple(
            "constraints",
            self.constraints,
            item_type=VentureConstraint,
            max_items=32,
            allow_empty=False,
        )
        if not isinstance(self.resource_window, VentureResourceWindow):
            raise ValueError("resource_window must be a VentureResourceWindow")
        _require_typed_tuple(
            "uncertainties",
            self.uncertainties,
            item_type=VentureUncertainty,
            max_items=32,
        )
        _require_typed_tuple(
            "evidence_refs",
            self.evidence_refs,
            item_type=VentureEvidenceRef,
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
    def from_json(cls, payload: Mapping[str, object]) -> "VentureContextRequest":
        allowed = frozenset(
            {
                "schema_version",
                "request_id",
                "portfolio_id",
                "cycle_id",
                "venture_id",
                "decision_id",
                "decision_point",
                "confirmed_facts",
                "constraints",
                "resource_window",
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
                "portfolio_id",
                "cycle_id",
                "venture_id",
                "decision_id",
                "decision_point",
                "confirmed_facts",
                "constraints",
                "resource_window",
                "uncertainties",
                "evidence_refs",
            }
        )
        _strict_payload(payload, allowed=allowed, required=required)
        if payload["schema_version"] != CONTEXT_REQUEST_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {CONTEXT_REQUEST_SCHEMA_VERSION!r}")
        facts = tuple(
            VentureFact.from_json(_mapping("confirmed_facts[]", item))
            for item in _array("confirmed_facts", payload["confirmed_facts"])
        )
        constraints = tuple(
            VentureConstraint.from_json(_mapping("constraints[]", item))
            for item in _array("constraints", payload["constraints"])
        )
        uncertainties = tuple(
            VentureUncertainty.from_json(_mapping("uncertainties[]", item))
            for item in _array("uncertainties", payload["uncertainties"])
        )
        evidence_refs = tuple(
            VentureEvidenceRef.from_json(_mapping("evidence_refs[]", item))
            for item in _array("evidence_refs", payload["evidence_refs"])
        )
        return cls(
            request_id=payload["request_id"],
            portfolio_id=payload["portfolio_id"],
            cycle_id=payload["cycle_id"],
            venture_id=payload["venture_id"],
            decision_id=payload["decision_id"],
            decision_point=_closed_enum(
                VentureDecisionPoint,
                "decision_point",
                payload["decision_point"],
            ),
            confirmed_facts=facts,
            constraints=constraints,
            resource_window=VentureResourceWindow.from_json(_mapping("resource_window", payload["resource_window"])),
            uncertainties=uncertainties,
            evidence_refs=evidence_refs,
            memory_limit=payload.get("memory_limit", 8),
            max_context_chars=payload.get("max_context_chars", 8_000),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": CONTEXT_REQUEST_SCHEMA_VERSION,
            "request_id": self.request_id,
            "portfolio_id": self.portfolio_id,
            "cycle_id": self.cycle_id,
            "venture_id": self.venture_id,
            "decision_id": self.decision_id,
            "decision_point": self.decision_point.value,
            "confirmed_facts": [fact.to_json() for fact in self.confirmed_facts],
            "constraints": [constraint.to_json() for constraint in self.constraints],
            "resource_window": self.resource_window.to_json(),
            "uncertainties": [item.to_json() for item in self.uncertainties],
            "evidence_refs": [reference.to_json() for reference in self.evidence_refs],
            "memory_limit": self.memory_limit,
            "max_context_chars": self.max_context_chars,
        }


@dataclass(frozen=True)
class VentureEstimateRange:
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
    def from_json(cls, payload: Mapping[str, object]) -> "VentureEstimateRange":
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
class VentureAdviceCandidate:
    candidate_id: str
    kind: VentureAdviceKind
    summary: str
    rationale: str
    prediction_ranges: tuple[VentureEstimateRange, ...]
    falsification_conditions: tuple[str, ...]
    evidence_ref_ids: tuple[str, ...]
    source_entry_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text("candidate_id", self.candidate_id, max_length=256)
        if not isinstance(self.kind, VentureAdviceKind):
            raise ValueError("kind must be a VentureAdviceKind")
        _require_text("summary", self.summary, max_length=2_000)
        _require_text("rationale", self.rationale, max_length=4_000)
        _require_typed_tuple(
            "prediction_ranges",
            self.prediction_ranges,
            item_type=VentureEstimateRange,
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
    def from_json(cls, payload: Mapping[str, object]) -> "VentureAdviceCandidate":
        fields = frozenset(
            {
                "candidate_id",
                "kind",
                "summary",
                "rationale",
                "prediction_ranges",
                "falsification_conditions",
                "evidence_ref_ids",
                "source_entry_ids",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        return cls(
            candidate_id=payload["candidate_id"],
            kind=_closed_enum(VentureAdviceKind, "kind", payload["kind"]),
            summary=payload["summary"],
            rationale=payload["rationale"],
            prediction_ranges=tuple(
                VentureEstimateRange.from_json(_mapping("prediction_ranges[]", item))
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
            "summary": self.summary,
            "rationale": self.rationale,
            "prediction_ranges": [item.to_json() for item in self.prediction_ranges],
            "falsification_conditions": list(self.falsification_conditions),
            "evidence_ref_ids": list(self.evidence_ref_ids),
            "source_entry_ids": list(self.source_entry_ids),
        }


@dataclass(frozen=True)
class VentureAdviceSnapshot:
    advice_id: str
    source_turn_index: int
    candidate_regime_id: str
    candidate_abstract_action: str
    candidates: tuple[VentureAdviceCandidate, ...]
    rationale: str
    wiring_level: WiringLevel = WiringLevel.SHADOW
    applied: bool = False

    def __post_init__(self) -> None:
        _require_content_id("advice_id", self.advice_id, prefix="venture-advice:")
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
            item_type=VentureAdviceCandidate,
            max_items=32,
        )
        _require_unique_ids("candidates", tuple(item.candidate_id for item in self.candidates))
        _require_text("rationale", self.rationale, max_length=2_000)
        if self.wiring_level is not WiringLevel.SHADOW:
            raise ValueError("Venture advice must remain WiringLevel.SHADOW")
        if not isinstance(self.applied, bool):
            raise ValueError("applied must be a boolean")
        if self.applied:
            raise ValueError("SHADOW Venture advice cannot be applied")

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "VentureAdviceSnapshot":
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
                VentureAdviceCandidate.from_json(_mapping("candidates[]", item))
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
class VentureCostBreakdown:
    acquisition_minor: int = 0
    model_minor: int = 0
    data_minor: int = 0
    human_review_minor: int = 0
    delivery_minor: int = 0
    support_minor: int = 0
    risk_reserve_minor: int = 0

    def __post_init__(self) -> None:
        for name, value in self.to_json().items():
            _require_non_negative_int(name, value)

    @property
    def total_minor(self) -> int:
        return sum(self.to_json().values())

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "VentureCostBreakdown":
        fields = frozenset(
            {
                "acquisition_minor",
                "model_minor",
                "data_minor",
                "human_review_minor",
                "delivery_minor",
                "support_minor",
                "risk_reserve_minor",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        return cls(**dict(payload))

    def to_json(self) -> dict[str, int]:
        return {
            "acquisition_minor": self.acquisition_minor,
            "model_minor": self.model_minor,
            "data_minor": self.data_minor,
            "human_review_minor": self.human_review_minor,
            "delivery_minor": self.delivery_minor,
            "support_minor": self.support_minor,
            "risk_reserve_minor": self.risk_reserve_minor,
        }


@dataclass(frozen=True)
class VentureCommercialOutcome:
    customer_result: VentureCustomerResult
    currency: str
    realized_revenue_minor: int
    realized_costs: VentureCostBreakdown
    refund_minor: int
    realized_net_value_minor: int
    elapsed_ms: int
    risk_level: VentureRiskLevel
    reversibility: VentureReversibility

    def __post_init__(self) -> None:
        if not isinstance(self.customer_result, VentureCustomerResult):
            raise ValueError("customer_result must be a VentureCustomerResult")
        _require_currency(self.currency)
        revenue = _require_non_negative_int("realized_revenue_minor", self.realized_revenue_minor)
        if not isinstance(self.realized_costs, VentureCostBreakdown):
            raise ValueError("realized_costs must be a VentureCostBreakdown")
        refund = _require_non_negative_int("refund_minor", self.refund_minor)
        net_value = _require_signed_int("realized_net_value_minor", self.realized_net_value_minor)
        expected = revenue - self.realized_costs.total_minor - refund
        if net_value != expected:
            raise ValueError("realized_net_value_minor must equal revenue minus seven costs minus refund")
        _require_non_negative_int("elapsed_ms", self.elapsed_ms)
        if not isinstance(self.risk_level, VentureRiskLevel):
            raise ValueError("risk_level must be a VentureRiskLevel")
        if not isinstance(self.reversibility, VentureReversibility):
            raise ValueError("reversibility must be a VentureReversibility")

    @property
    def has_commercial_observation(self) -> bool:
        return bool(
            self.customer_result is not VentureCustomerResult.NOT_OBSERVED
            or self.realized_revenue_minor
            or self.realized_costs.total_minor
            or self.refund_minor
            or self.elapsed_ms
            or self.risk_level is not VentureRiskLevel.UNASSESSED
        )

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "VentureCommercialOutcome":
        fields = frozenset(
            {
                "customer_result",
                "currency",
                "realized_revenue_minor",
                "realized_costs",
                "refund_minor",
                "realized_net_value_minor",
                "elapsed_ms",
                "risk_level",
                "reversibility",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        return cls(
            customer_result=_closed_enum(
                VentureCustomerResult,
                "customer_result",
                payload["customer_result"],
            ),
            currency=payload["currency"],
            realized_revenue_minor=payload["realized_revenue_minor"],
            realized_costs=VentureCostBreakdown.from_json(_mapping("realized_costs", payload["realized_costs"])),
            refund_minor=payload["refund_minor"],
            realized_net_value_minor=payload["realized_net_value_minor"],
            elapsed_ms=payload["elapsed_ms"],
            risk_level=_closed_enum(VentureRiskLevel, "risk_level", payload["risk_level"]),
            reversibility=_closed_enum(
                VentureReversibility,
                "reversibility",
                payload["reversibility"],
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "customer_result": self.customer_result.value,
            "currency": self.currency,
            "realized_revenue_minor": self.realized_revenue_minor,
            "realized_costs": self.realized_costs.to_json(),
            "refund_minor": self.refund_minor,
            "realized_net_value_minor": self.realized_net_value_minor,
            "elapsed_ms": self.elapsed_ms,
            "risk_level": self.risk_level.value,
            "reversibility": self.reversibility.value,
        }


@dataclass(frozen=True)
class VentureOutcomeReport:
    outcome_id: str
    context_pack_id: str
    decision_id: str
    decision: VentureDecisionKind
    outcome_kind: VentureOutcomeKind
    evidence_class: VentureEvidenceClass
    verdict: VentureOutcomeVerdict
    summary: str
    detail: str
    observed_at_ms: int
    evidence_refs: tuple[VentureEvidenceRef, ...]
    commercial_outcome: VentureCommercialOutcome

    def __post_init__(self) -> None:
        _require_text("outcome_id", self.outcome_id, max_length=256)
        _require_content_id(
            "context_pack_id",
            self.context_pack_id,
            prefix="venture-context-pack:",
        )
        _require_text("decision_id", self.decision_id, max_length=256)
        if not isinstance(self.decision, VentureDecisionKind):
            raise ValueError("decision must be a VentureDecisionKind")
        if not isinstance(self.outcome_kind, VentureOutcomeKind):
            raise ValueError("outcome_kind must be a VentureOutcomeKind")
        if not isinstance(self.evidence_class, VentureEvidenceClass):
            raise ValueError("evidence_class must be a VentureEvidenceClass")
        if self.outcome_kind not in _OUTCOME_CLASS_PAIRS[self.evidence_class]:
            raise ValueError(
                f"outcome_kind={self.outcome_kind.value} is not legal for evidence_class={self.evidence_class.value}"
            )
        if not isinstance(self.verdict, VentureOutcomeVerdict):
            raise ValueError("verdict must be a VentureOutcomeVerdict")
        _require_text("summary", self.summary, max_length=2_000)
        _require_text("detail", self.detail, max_length=16_000)
        _require_non_negative_int("observed_at_ms", self.observed_at_ms)
        _require_typed_tuple(
            "evidence_refs",
            self.evidence_refs,
            item_type=VentureEvidenceRef,
            max_items=128,
            allow_empty=False,
        )
        _require_unique_ids("evidence_refs", tuple(item.ref_id for item in self.evidence_refs))
        if any(reference.evidence_class is not self.evidence_class for reference in self.evidence_refs):
            raise ValueError("all outcome evidence refs must match evidence_class")
        if not isinstance(self.commercial_outcome, VentureCommercialOutcome):
            raise ValueError("commercial_outcome must be a VentureCommercialOutcome")
        self._validate_lane_payload()

    @property
    def pe_eligible(self) -> bool:
        return (
            self.evidence_class is VentureEvidenceClass.FIELD
            and self.outcome_kind is VentureOutcomeKind.FIELD_EXPERIMENT_RESULT
        )

    def _validate_lane_payload(self) -> None:
        roles = {reference.role for reference in self.evidence_refs}
        commercial = self.commercial_outcome
        if self.evidence_class in {
            VentureEvidenceClass.SIMULATION,
            VentureEvidenceClass.INTERNAL_REVIEW,
            VentureEvidenceClass.MACHINE_CHECK,
        }:
            if commercial.has_commercial_observation:
                raise ValueError(
                    "simulation/internal_review/machine_check outcomes cannot carry customer, "
                    "financial, elapsed, or risk observations"
                )
            return
        if self.evidence_class is not VentureEvidenceClass.FIELD:
            return
        required_role: VentureEvidenceRole | None = None
        if self.outcome_kind is VentureOutcomeKind.CUSTOMER_OUTCOME:
            required_role = VentureEvidenceRole.CUSTOMER_OUTCOME
            if commercial.customer_result is VentureCustomerResult.NOT_OBSERVED:
                raise ValueError("customer_outcome requires an observed customer result")
        elif self.outcome_kind is VentureOutcomeKind.PAYMENT_RECEIVED:
            required_role = VentureEvidenceRole.PAYMENT
            if commercial.realized_revenue_minor <= 0:
                raise ValueError("payment_received requires positive realized revenue")
        elif self.outcome_kind is VentureOutcomeKind.COST_RECORDED:
            required_role = VentureEvidenceRole.COST
            if commercial.realized_costs.total_minor <= 0:
                raise ValueError("cost_recorded requires positive realized cost")
        elif self.outcome_kind is VentureOutcomeKind.REFUND_RECORDED:
            required_role = VentureEvidenceRole.REFUND
            if commercial.refund_minor <= 0:
                raise ValueError("refund_recorded requires a positive refund")
        elif self.outcome_kind is VentureOutcomeKind.FIELD_EXPERIMENT_RESULT:
            permitted = {
                VentureEvidenceRole.FIELD_OBSERVATION,
                VentureEvidenceRole.CUSTOMER_OUTCOME,
                VentureEvidenceRole.PAYMENT,
                VentureEvidenceRole.COST,
                VentureEvidenceRole.REFUND,
            }
            if not roles.intersection(permitted):
                raise ValueError("field_experiment_result requires field/customer/payment/cost/refund evidence")
            if not commercial.has_commercial_observation:
                raise ValueError("field_experiment_result requires at least one observed business dimension")
        if required_role is not None and required_role not in roles:
            raise ValueError(f"{self.outcome_kind.value} requires evidence role={required_role.value}")
        dimension_roles = (
            (
                commercial.customer_result is not VentureCustomerResult.NOT_OBSERVED,
                VentureEvidenceRole.CUSTOMER_OUTCOME,
                "observed customer_result",
            ),
            (
                commercial.realized_revenue_minor > 0,
                VentureEvidenceRole.PAYMENT,
                "positive realized_revenue_minor",
            ),
            (
                commercial.realized_costs.total_minor > 0,
                VentureEvidenceRole.COST,
                "positive realized costs",
            ),
            (
                commercial.refund_minor > 0,
                VentureEvidenceRole.REFUND,
                "positive refund_minor",
            ),
        )
        for observed, role, dimension in dimension_roles:
            if observed and role not in roles:
                raise ValueError(f"{dimension} requires evidence role={role.value}")

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "VentureOutcomeReport":
        fields = frozenset(
            {
                "schema_version",
                "outcome_id",
                "context_pack_id",
                "decision_id",
                "decision",
                "outcome_kind",
                "evidence_class",
                "verdict",
                "summary",
                "detail",
                "observed_at_ms",
                "evidence_refs",
                "commercial_outcome",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        if payload["schema_version"] != OUTCOME_REPORT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {OUTCOME_REPORT_SCHEMA_VERSION!r}")
        evidence_refs = tuple(
            VentureEvidenceRef.from_json(_mapping("evidence_refs[]", item))
            for item in _array("evidence_refs", payload["evidence_refs"])
        )
        return cls(
            outcome_id=payload["outcome_id"],
            context_pack_id=payload["context_pack_id"],
            decision_id=payload["decision_id"],
            decision=_closed_enum(VentureDecisionKind, "decision", payload["decision"]),
            outcome_kind=_closed_enum(VentureOutcomeKind, "outcome_kind", payload["outcome_kind"]),
            evidence_class=_closed_enum(
                VentureEvidenceClass,
                "evidence_class",
                payload["evidence_class"],
            ),
            verdict=_closed_enum(VentureOutcomeVerdict, "verdict", payload["verdict"]),
            summary=payload["summary"],
            detail=payload["detail"],
            observed_at_ms=payload["observed_at_ms"],
            evidence_refs=evidence_refs,
            commercial_outcome=VentureCommercialOutcome.from_json(
                _mapping("commercial_outcome", payload["commercial_outcome"])
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": OUTCOME_REPORT_SCHEMA_VERSION,
            "outcome_id": self.outcome_id,
            "context_pack_id": self.context_pack_id,
            "decision_id": self.decision_id,
            "decision": self.decision.value,
            "outcome_kind": self.outcome_kind.value,
            "evidence_class": self.evidence_class.value,
            "verdict": self.verdict.value,
            "summary": self.summary,
            "detail": self.detail,
            "observed_at_ms": self.observed_at_ms,
            "evidence_refs": [reference.to_json() for reference in self.evidence_refs],
            "commercial_outcome": self.commercial_outcome.to_json(),
        }


@dataclass(frozen=True)
class VentureRecalledExperience:
    memory_entry_id: str
    portfolio_id: str
    cycle_id: str
    venture_id: str
    decision_id: str
    source_context_pack_id: str
    created_at_ms: int
    report: VentureOutcomeReport

    def __post_init__(self) -> None:
        for name, value in (
            ("memory_entry_id", self.memory_entry_id),
            ("portfolio_id", self.portfolio_id),
            ("cycle_id", self.cycle_id),
            ("decision_id", self.decision_id),
            ("source_context_pack_id", self.source_context_pack_id),
        ):
            _require_text(name, value, max_length=256)
        _require_optional_text("venture_id", self.venture_id, max_length=256)
        _require_non_negative_int("created_at_ms", self.created_at_ms)
        if not isinstance(self.report, VentureOutcomeReport):
            raise ValueError("report must be a VentureOutcomeReport")
        if self.source_context_pack_id != self.report.context_pack_id:
            raise ValueError("source_context_pack_id must match report lineage")

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "VentureRecalledExperience":
        fields = frozenset(
            {
                "memory_entry_id",
                "portfolio_id",
                "cycle_id",
                "venture_id",
                "decision_id",
                "source_context_pack_id",
                "created_at_ms",
                "report",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        return cls(
            memory_entry_id=payload["memory_entry_id"],
            portfolio_id=payload["portfolio_id"],
            cycle_id=payload["cycle_id"],
            venture_id=payload["venture_id"],
            decision_id=payload["decision_id"],
            source_context_pack_id=payload["source_context_pack_id"],
            created_at_ms=payload["created_at_ms"],
            report=VentureOutcomeReport.from_json(_mapping("report", payload["report"])),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "memory_entry_id": self.memory_entry_id,
            "portfolio_id": self.portfolio_id,
            "cycle_id": self.cycle_id,
            "venture_id": self.venture_id,
            "decision_id": self.decision_id,
            "source_context_pack_id": self.source_context_pack_id,
            "created_at_ms": self.created_at_ms,
            "report": self.report.to_json(),
        }


@dataclass(frozen=True)
class VentureContextPackSnapshot:
    context_pack_id: str
    content_sha256: str
    session_id: str
    request: VentureContextRequest
    generated_at_ms: int
    source_turn_index: int
    rendered_context: str
    recalled_experiences: tuple[VentureRecalledExperience, ...]
    source_entry_ids: tuple[str, ...]
    source_evidence_ref_ids: tuple[str, ...]
    retrieval_facets: tuple[str, ...]
    memory_entry_count: int
    truncated: bool
    current_uncertainties: tuple[VentureUncertainty, ...]
    settled_outcome_ids: tuple[str, ...]
    settled_evidence_ref_ids: tuple[str, ...]
    pe_magnitude: float
    pe_bootstrap: bool
    advice: VentureAdviceSnapshot
    content_policy_decision: BoundedContentPolicyDecision | None = None
    settled_policy_credits: tuple[BoundedContentPolicyCredit, ...] = ()
    policy_updates: tuple[BoundedContentPolicyUpdateReceipt, ...] = ()
    content_policy_wiring_level: WiringLevel = WiringLevel.ACTIVE
    wiring_level: WiringLevel = WiringLevel.ACTIVE

    def __post_init__(self) -> None:
        _require_content_id(
            "context_pack_id",
            self.context_pack_id,
            prefix="venture-context-pack:",
        )
        _require_sha256("content_sha256", self.content_sha256)
        if self.context_pack_id != f"venture-context-pack:{self.content_sha256}":
            raise ValueError("context_pack_id must match content_sha256")
        _require_text("session_id", self.session_id, max_length=256)
        if not isinstance(self.request, VentureContextRequest):
            raise ValueError("request must be a VentureContextRequest")
        _require_non_negative_int("generated_at_ms", self.generated_at_ms)
        _require_non_negative_int("source_turn_index", self.source_turn_index)
        if not isinstance(self.rendered_context, str):
            raise ValueError("rendered_context must be a string")
        if len(self.rendered_context) > self.request.max_context_chars:
            raise ValueError("rendered_context exceeds request max_context_chars")
        _require_typed_tuple(
            "recalled_experiences",
            self.recalled_experiences,
            item_type=VentureRecalledExperience,
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
            item_type=VentureUncertainty,
            max_items=32,
        )
        if self.current_uncertainties != self.request.uncertainties:
            raise ValueError("current_uncertainties must preserve request uncertainty")
        _require_numeric("pe_magnitude", self.pe_magnitude)
        if not isinstance(self.pe_bootstrap, bool):
            raise ValueError("pe_bootstrap must be a boolean")
        if not isinstance(self.advice, VentureAdviceSnapshot):
            raise ValueError("advice must be a VentureAdviceSnapshot")
        if self.content_policy_decision is not None and not isinstance(
            self.content_policy_decision,
            BoundedContentPolicyDecision,
        ):
            raise ValueError(
                "content_policy_decision must be a BoundedContentPolicyDecision"
            )
        _require_typed_tuple(
            "settled_policy_credits",
            self.settled_policy_credits,
            item_type=BoundedContentPolicyCredit,
            max_items=1,
        )
        _require_typed_tuple(
            "policy_updates",
            self.policy_updates,
            item_type=BoundedContentPolicyUpdateReceipt,
            max_items=1,
        )
        if len(self.settled_policy_credits) != len(self.policy_updates):
            raise ValueError("each settled content policy credit requires one update")
        if self.content_policy_wiring_level not in {
            WiringLevel.ACTIVE,
            WiringLevel.DISABLED,
        }:
            raise ValueError("Venture content policy must be ACTIVE or DISABLED")
        if self.content_policy_wiring_level is WiringLevel.DISABLED and any(
            (
                self.content_policy_decision is not None,
                self.settled_policy_credits,
                self.policy_updates,
            )
        ):
            raise ValueError("DISABLED content policy cannot publish policy lineage")
        if self.wiring_level is not WiringLevel.ACTIVE:
            raise ValueError("Venture Context Pack must be WiringLevel.ACTIVE")

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "VentureContextPackSnapshot":
        fields = frozenset(
            {
                "schema_version",
                "context_pack_id",
                "content_sha256",
                "session_id",
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
                "content_policy_decision",
                "settled_policy_credits",
                "policy_updates",
                "content_policy_wiring_level",
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
            request=VentureContextRequest.from_json(_mapping("request", payload["request"])),
            generated_at_ms=payload["generated_at_ms"],
            source_turn_index=payload["source_turn_index"],
            rendered_context=payload["rendered_context"],
            recalled_experiences=tuple(
                VentureRecalledExperience.from_json(_mapping("recalled_experiences[]", item))
                for item in _array("recalled_experiences", payload["recalled_experiences"])
            ),
            source_entry_ids=tuple(_array("source_entry_ids", payload["source_entry_ids"])),
            source_evidence_ref_ids=tuple(_array("source_evidence_ref_ids", payload["source_evidence_ref_ids"])),
            retrieval_facets=tuple(_array("retrieval_facets", payload["retrieval_facets"])),
            memory_entry_count=payload["memory_entry_count"],
            truncated=payload["truncated"],
            current_uncertainties=tuple(
                VentureUncertainty.from_json(_mapping("current_uncertainties[]", item))
                for item in _array("current_uncertainties", payload["current_uncertainties"])
            ),
            settled_outcome_ids=tuple(_array("settled_outcome_ids", payload["settled_outcome_ids"])),
            settled_evidence_ref_ids=tuple(_array("settled_evidence_ref_ids", payload["settled_evidence_ref_ids"])),
            pe_magnitude=payload["pe_magnitude"],
            pe_bootstrap=payload["pe_bootstrap"],
            advice=VentureAdviceSnapshot.from_json(_mapping("advice", payload["advice"])),
            content_policy_decision=(
                BoundedContentPolicyDecision.from_json(
                    _mapping(
                        "content_policy_decision",
                        payload["content_policy_decision"],
                    )
                )
                if payload["content_policy_decision"] is not None
                else None
            ),
            settled_policy_credits=tuple(
                BoundedContentPolicyCredit.from_json(
                    _mapping("settled_policy_credits[]", item)
                )
                for item in _array(
                    "settled_policy_credits",
                    payload["settled_policy_credits"],
                )
            ),
            policy_updates=tuple(
                BoundedContentPolicyUpdateReceipt.from_json(
                    _mapping("policy_updates[]", item)
                )
                for item in _array("policy_updates", payload["policy_updates"])
            ),
            content_policy_wiring_level=_closed_enum(
                WiringLevel,
                "content_policy_wiring_level",
                payload["content_policy_wiring_level"],
            ),
            wiring_level=_closed_enum(WiringLevel, "wiring_level", payload["wiring_level"]),
        )
        digest_payload = snapshot.to_json()
        digest_payload.pop("context_pack_id")
        digest_payload.pop("content_sha256")
        if stable_content_sha256(digest_payload) != snapshot.content_sha256:
            raise ValueError("Venture Context Pack content_sha256 does not match its payload")
        return snapshot

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": CONTEXT_PACK_SCHEMA_VERSION,
            "context_pack_id": self.context_pack_id,
            "content_sha256": self.content_sha256,
            "session_id": self.session_id,
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
class VentureOutcomeReceipt:
    receipt_id: str
    content_sha256: str
    session_id: str
    portfolio_id: str
    cycle_id: str
    venture_id: str
    decision_id: str
    report: VentureOutcomeReport
    action_turn_index: int
    source_advice_id: str
    source_advice_applied: bool
    memory_entry_id: str
    memory_persisted: bool
    task_event_ids: tuple[str, ...]
    environment_outcome_id: str
    learning_route: VentureOutcomeRoute
    settlement_state: VentureSettlementState
    source_content_policy_decision_id: str = ""
    content_policy_action_applied: bool = False

    def __post_init__(self) -> None:
        _require_content_id(
            "receipt_id",
            self.receipt_id,
            prefix="venture-outcome-receipt:",
        )
        for name, value in (
            ("session_id", self.session_id),
            ("portfolio_id", self.portfolio_id),
            ("cycle_id", self.cycle_id),
            ("decision_id", self.decision_id),
            ("memory_entry_id", self.memory_entry_id),
        ):
            _require_text(name, value, max_length=256)
        _require_content_id(
            "source_advice_id",
            self.source_advice_id,
            prefix="venture-advice:",
        )
        _require_optional_text("venture_id", self.venture_id, max_length=256)
        _require_sha256("content_sha256", self.content_sha256)
        if self.receipt_id != f"venture-outcome-receipt:{self.content_sha256}":
            raise ValueError("receipt_id must match content_sha256")
        if not isinstance(self.report, VentureOutcomeReport):
            raise ValueError("report must be a VentureOutcomeReport")
        _require_non_negative_int("action_turn_index", self.action_turn_index)
        if not isinstance(self.source_advice_applied, bool):
            raise ValueError("source_advice_applied must be a boolean")
        if self.source_advice_applied:
            raise ValueError("Venture v1 receipts cannot claim SHADOW advice was applied")
        if not isinstance(self.memory_persisted, bool):
            raise ValueError("memory_persisted must be a boolean")
        _require_string_tuple(
            "task_event_ids",
            self.task_event_ids,
            max_items=16,
            item_max_length=512,
        )
        _require_optional_text("environment_outcome_id", self.environment_outcome_id, max_length=512)
        if not isinstance(self.learning_route, VentureOutcomeRoute):
            raise ValueError("learning_route must be a VentureOutcomeRoute")
        if not isinstance(self.settlement_state, VentureSettlementState):
            raise ValueError("settlement_state must be a VentureSettlementState")
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
        if self.report.pe_eligible:
            if self.learning_route is not VentureOutcomeRoute.FIELD_PE_MEMORY_AND_EXECUTION_RESULT:
                raise ValueError("field_experiment_result must use the PE route")
            if not self.environment_outcome_id:
                raise ValueError("PE-eligible reports require environment outcome lineage")
            if self.settlement_state is not VentureSettlementState.PENDING_NEXT_CONTEXT_TURN:
                raise ValueError("PE-eligible reports must remain pending until next turn")
        else:
            if self.learning_route is not VentureOutcomeRoute.MEMORY_AND_EXECUTION_RESULT:
                raise ValueError("non-eligible reports cannot use the PE route")
            if self.environment_outcome_id:
                raise ValueError("non-eligible reports cannot publish environment outcomes")
            if self.settlement_state is not VentureSettlementState.NOT_PE_ELIGIBLE:
                raise ValueError("non-eligible reports require not_pe_eligible state")

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "VentureOutcomeReceipt":
        fields = frozenset(
            {
                "schema_version",
                "receipt_id",
                "content_sha256",
                "session_id",
                "portfolio_id",
                "cycle_id",
                "venture_id",
                "decision_id",
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
                "source_content_policy_decision_id",
                "content_policy_action_applied",
            }
        )
        _strict_payload(payload, allowed=fields, required=fields)
        if payload["schema_version"] != OUTCOME_RECEIPT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {OUTCOME_RECEIPT_SCHEMA_VERSION!r}")
        receipt = cls(
            receipt_id=payload["receipt_id"],
            content_sha256=payload["content_sha256"],
            session_id=payload["session_id"],
            portfolio_id=payload["portfolio_id"],
            cycle_id=payload["cycle_id"],
            venture_id=payload["venture_id"],
            decision_id=payload["decision_id"],
            report=VentureOutcomeReport.from_json(_mapping("report", payload["report"])),
            action_turn_index=payload["action_turn_index"],
            source_advice_id=payload["source_advice_id"],
            source_advice_applied=payload["source_advice_applied"],
            memory_entry_id=payload["memory_entry_id"],
            memory_persisted=payload["memory_persisted"],
            task_event_ids=tuple(_array("task_event_ids", payload["task_event_ids"])),
            environment_outcome_id=payload["environment_outcome_id"],
            learning_route=_closed_enum(
                VentureOutcomeRoute,
                "learning_route",
                payload["learning_route"],
            ),
            settlement_state=_closed_enum(
                VentureSettlementState,
                "settlement_state",
                payload["settlement_state"],
            ),
            source_content_policy_decision_id=payload[
                "source_content_policy_decision_id"
            ],
            content_policy_action_applied=payload[
                "content_policy_action_applied"
            ],
        )
        digest_payload = receipt.to_json()
        digest_payload.pop("receipt_id")
        digest_payload.pop("content_sha256")
        if stable_content_sha256(digest_payload) != receipt.content_sha256:
            raise ValueError("Venture Outcome Receipt content_sha256 does not match its payload")
        return receipt

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": OUTCOME_RECEIPT_SCHEMA_VERSION,
            "receipt_id": self.receipt_id,
            "content_sha256": self.content_sha256,
            "session_id": self.session_id,
            "portfolio_id": self.portfolio_id,
            "cycle_id": self.cycle_id,
            "venture_id": self.venture_id,
            "decision_id": self.decision_id,
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
            "source_content_policy_decision_id": (
                self.source_content_policy_decision_id
            ),
            "content_policy_action_applied": self.content_policy_action_applied,
        }


__all__ = (
    "ADVICE_SCHEMA_VERSION",
    "CONTEXT_PACK_SCHEMA_VERSION",
    "CONTEXT_REQUEST_SCHEMA_VERSION",
    "EXPERIENCE_RECORD_SCHEMA_VERSION",
    "OUTCOME_RECEIPT_SCHEMA_VERSION",
    "OUTCOME_REPORT_SCHEMA_VERSION",
    "VentureAdviceCandidate",
    "VentureAdviceKind",
    "VentureAdviceSnapshot",
    "VentureCommercialOutcome",
    "VentureConstraint",
    "VentureConstraintKind",
    "VentureContextPackSnapshot",
    "VentureContextRequest",
    "VentureCostBreakdown",
    "VentureCustomerResult",
    "VentureDecisionKind",
    "VentureDecisionPoint",
    "VentureEstimateRange",
    "VentureEvidenceClass",
    "VentureEvidenceRef",
    "VentureEvidenceRole",
    "VentureFact",
    "VentureFactKind",
    "VentureOutcomeKind",
    "VentureOutcomeReceipt",
    "VentureOutcomeReport",
    "VentureOutcomeRoute",
    "VentureOutcomeVerdict",
    "VentureRecalledExperience",
    "VentureResourceWindow",
    "VentureReversibility",
    "VentureRiskLevel",
    "VentureSettlementState",
    "VentureUncertainty",
    "stable_content_sha256",
)
