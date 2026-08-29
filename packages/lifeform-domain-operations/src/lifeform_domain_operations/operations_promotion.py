"""Immutable Operations policy benchmark and ModificationGate promotion.

Benchmark readouts are validation-only. They never enter ``OperationsPolicy``
updates, whose only input is ``OperationsPolicyCredit`` derived from PE and the
Credit owner. This module can authorize a SHADOW -> ACTIVE wiring change, but
does not mutate deployment configuration itself.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from volvence_zero.credit import (
    GateDecision,
    ModificationGate,
    ModificationProposal,
    evaluate_gate_reasons,
)
from volvence_zero.evaluation import EvaluationScore, EvaluationSnapshot
from volvence_zero.runtime import WiringLevel

from lifeform_domain_operations.operations_brain_contracts import (
    OperationsPolicyCheckpoint,
    stable_content_sha256,
)


OPERATIONS_BENCHMARK_SCHEMA_VERSION = "operations-policy-benchmark.v1"
OPERATIONS_PROMOTION_REVIEW_SCHEMA_VERSION = "operations-promotion-review.v1"
OPERATIONS_ACTIVATION_SCHEMA_VERSION = "operations-policy-activation.v1"
OPERATIONS_BENCHMARK_EVIDENCE_SCOPE = "deterministic_simulation"
OPERATIONS_ACTIVATION_SCOPE = "autocompany_staging"
OPERATIONS_PROMOTION_TARGET = "operations.policy.wiring.autocompany_staging"
OPERATIONS_BENCHMARK_PROTOCOL_ID = "operations-multicycle-matched-baseline.v1"


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_sha256(name: str, value: object) -> str:
    text = _require_text(name, value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _require_content_id(name: str, value: object, prefix: str) -> str:
    text = _require_text(name, value)
    if not text.startswith(prefix):
        raise ValueError(f"{name} must start with {prefix!r}")
    _require_sha256(name, text[len(prefix) :])
    return text


def _require_probability(name: str, value: float) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def _strict_payload(
    payload: Mapping[str, object],
    *,
    fields: frozenset[str],
) -> None:
    missing = fields - set(payload)
    unknown = set(payload) - fields
    if missing:
        raise ValueError(f"missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"unknown fields: {', '.join(sorted(unknown))}")


def _mapping(name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _array(name: str, value: object) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return tuple(value)


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _boolean(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


@dataclass(frozen=True)
class OperationsBenchmarkArmResult:
    arm_id: str
    training_cycles: int
    evaluation_utilities: tuple[float, ...]
    mean_utility: float
    favorable_rate: float
    correct_action_rate: float
    intervention_rate: float
    policy_update_count: int
    pe_credit_count: int

    def __post_init__(self) -> None:
        _require_text("arm_id", self.arm_id)
        if self.training_cycles < 0 or self.policy_update_count < 0 or self.pe_credit_count < 0:
            raise ValueError("benchmark counts must be non-negative")
        if not self.evaluation_utilities:
            raise ValueError("evaluation_utilities must not be empty")
        if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in self.evaluation_utilities):
            raise ValueError("evaluation utilities must be finite and in [-1, 1]")
        expected_mean = math.fsum(self.evaluation_utilities) / len(
            self.evaluation_utilities
        )
        if not math.isclose(self.mean_utility, expected_mean, abs_tol=1e-12):
            raise ValueError("mean_utility does not match evaluation utilities")
        for name, value in (
            ("favorable_rate", self.favorable_rate),
            ("correct_action_rate", self.correct_action_rate),
            ("intervention_rate", self.intervention_rate),
        ):
            _require_probability(name, value)

    @classmethod
    def create(
        cls,
        *,
        arm_id: str,
        training_cycles: int,
        evaluation_utilities: tuple[float, ...],
        correct_action_count: int,
        intervention_count: int,
        policy_update_count: int,
        pe_credit_count: int,
    ) -> "OperationsBenchmarkArmResult":
        count = len(evaluation_utilities)
        return cls(
            arm_id=arm_id,
            training_cycles=training_cycles,
            evaluation_utilities=evaluation_utilities,
            mean_utility=math.fsum(evaluation_utilities) / count,
            favorable_rate=sum(value > 0.0 for value in evaluation_utilities)
            / count,
            correct_action_rate=correct_action_count / count,
            intervention_rate=intervention_count / count,
            policy_update_count=policy_update_count,
            pe_credit_count=pe_credit_count,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "arm_id": self.arm_id,
            "training_cycles": self.training_cycles,
            "evaluation_utilities": list(self.evaluation_utilities),
            "mean_utility": self.mean_utility,
            "favorable_rate": self.favorable_rate,
            "correct_action_rate": self.correct_action_rate,
            "intervention_rate": self.intervention_rate,
            "policy_update_count": self.policy_update_count,
            "pe_credit_count": self.pe_credit_count,
        }

    @classmethod
    def from_json(
        cls,
        payload: Mapping[str, object],
    ) -> "OperationsBenchmarkArmResult":
        fields = frozenset(
            {
                "arm_id",
                "training_cycles",
                "evaluation_utilities",
                "mean_utility",
                "favorable_rate",
                "correct_action_rate",
                "intervention_rate",
                "policy_update_count",
                "pe_credit_count",
            }
        )
        _strict_payload(payload, fields=fields)
        return cls(
            arm_id=_require_text("arm_id", payload["arm_id"]),
            training_cycles=_integer(
                "training_cycles",
                payload["training_cycles"],
            ),
            evaluation_utilities=tuple(
                _number("evaluation_utilities[]", item)
                for item in _array(
                    "evaluation_utilities",
                    payload["evaluation_utilities"],
                )
            ),
            mean_utility=_number("mean_utility", payload["mean_utility"]),
            favorable_rate=_number(
                "favorable_rate",
                payload["favorable_rate"],
            ),
            correct_action_rate=_number(
                "correct_action_rate",
                payload["correct_action_rate"],
            ),
            intervention_rate=_number(
                "intervention_rate",
                payload["intervention_rate"],
            ),
            policy_update_count=_integer(
                "policy_update_count",
                payload["policy_update_count"],
            ),
            pe_credit_count=_integer(
                "pe_credit_count",
                payload["pe_credit_count"],
            ),
        )


@dataclass(frozen=True)
class OperationsPolicyBenchmarkReport:
    report_id: str
    content_sha256: str
    protocol_id: str
    evidence_scope: str
    preregistration_sha256: str
    scenario_set_sha256: str
    seed: int
    arms: tuple[OperationsBenchmarkArmResult, ...]
    primary_baseline_arm: str
    learned_arm: str
    validation_delta: float
    paired_delta_lower_95: float
    candidate_checkpoint_id: str
    candidate_checkpoint_sha256: str
    checkpoint_round_trip_verified: bool
    rollback_drill_verified: bool
    exact_pe_credit_lineage_verified: bool
    evaluation_writeback_allowed: bool
    production_default_changed: bool

    def __post_init__(self) -> None:
        _require_content_id(
            "report_id",
            self.report_id,
            "operations-policy-benchmark:",
        )
        _require_sha256("content_sha256", self.content_sha256)
        if self.report_id != f"operations-policy-benchmark:{self.content_sha256}":
            raise ValueError("benchmark report id/content mismatch")
        if self.protocol_id != OPERATIONS_BENCHMARK_PROTOCOL_ID:
            raise ValueError("benchmark protocol id drift")
        if self.evidence_scope != OPERATIONS_BENCHMARK_EVIDENCE_SCOPE:
            raise ValueError("benchmark evidence scope drift")
        _require_sha256("preregistration_sha256", self.preregistration_sha256)
        _require_sha256("scenario_set_sha256", self.scenario_set_sha256)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if len(self.arms) < 4:
            raise ValueError("benchmark requires at least four arms")
        arms_by_id = {item.arm_id: item for item in self.arms}
        if len(arms_by_id) != len(self.arms):
            raise ValueError("benchmark arm ids must be unique")
        if self.primary_baseline_arm not in arms_by_id or self.learned_arm not in arms_by_id:
            raise ValueError("benchmark primary arm ids are missing")
        learned = arms_by_id[self.learned_arm]
        baseline = arms_by_id[self.primary_baseline_arm]
        if len(learned.evaluation_utilities) != len(baseline.evaluation_utilities):
            raise ValueError("benchmark primary arms must be paired")
        expected_delta = learned.mean_utility - baseline.mean_utility
        if not math.isclose(self.validation_delta, expected_delta, abs_tol=1e-12):
            raise ValueError("benchmark validation_delta mismatch")
        if not math.isfinite(self.paired_delta_lower_95):
            raise ValueError("paired_delta_lower_95 must be finite")
        _require_content_id(
            "candidate_checkpoint_id",
            self.candidate_checkpoint_id,
            "operations-policy-checkpoint:",
        )
        _require_sha256(
            "candidate_checkpoint_sha256",
            self.candidate_checkpoint_sha256,
        )
        if self.candidate_checkpoint_id != (
            f"operations-policy-checkpoint:{self.candidate_checkpoint_sha256}"
        ):
            raise ValueError("benchmark checkpoint id/digest mismatch")
        if self.evaluation_writeback_allowed:
            raise ValueError("benchmark evaluation writeback must remain disabled")
        if self.production_default_changed:
            raise ValueError("benchmark cannot change production defaults")

    @classmethod
    def create(
        cls,
        *,
        preregistration_sha256: str,
        scenario_set_sha256: str,
        seed: int,
        arms: tuple[OperationsBenchmarkArmResult, ...],
        primary_baseline_arm: str,
        learned_arm: str,
        paired_delta_lower_95: float,
        candidate_checkpoint: OperationsPolicyCheckpoint,
        checkpoint_round_trip_verified: bool,
        rollback_drill_verified: bool,
        exact_pe_credit_lineage_verified: bool,
    ) -> "OperationsPolicyBenchmarkReport":
        arms_by_id = {item.arm_id: item for item in arms}
        validation_delta = (
            arms_by_id[learned_arm].mean_utility
            - arms_by_id[primary_baseline_arm].mean_utility
        )
        core = {
            "schema_version": OPERATIONS_BENCHMARK_SCHEMA_VERSION,
            "protocol_id": OPERATIONS_BENCHMARK_PROTOCOL_ID,
            "evidence_scope": OPERATIONS_BENCHMARK_EVIDENCE_SCOPE,
            "preregistration_sha256": preregistration_sha256,
            "scenario_set_sha256": scenario_set_sha256,
            "seed": seed,
            "arms": [item.to_json() for item in arms],
            "primary_baseline_arm": primary_baseline_arm,
            "learned_arm": learned_arm,
            "validation_delta": validation_delta,
            "paired_delta_lower_95": paired_delta_lower_95,
            "candidate_checkpoint_id": candidate_checkpoint.checkpoint_id,
            "candidate_checkpoint_sha256": candidate_checkpoint.content_sha256,
            "checkpoint_round_trip_verified": checkpoint_round_trip_verified,
            "rollback_drill_verified": rollback_drill_verified,
            "exact_pe_credit_lineage_verified": exact_pe_credit_lineage_verified,
            "evaluation_writeback_allowed": False,
            "production_default_changed": False,
        }
        digest = stable_content_sha256(core)
        return cls(
            report_id=f"operations-policy-benchmark:{digest}",
            content_sha256=digest,
            protocol_id=OPERATIONS_BENCHMARK_PROTOCOL_ID,
            evidence_scope=OPERATIONS_BENCHMARK_EVIDENCE_SCOPE,
            preregistration_sha256=preregistration_sha256,
            scenario_set_sha256=scenario_set_sha256,
            seed=seed,
            arms=arms,
            primary_baseline_arm=primary_baseline_arm,
            learned_arm=learned_arm,
            validation_delta=validation_delta,
            paired_delta_lower_95=paired_delta_lower_95,
            candidate_checkpoint_id=candidate_checkpoint.checkpoint_id,
            candidate_checkpoint_sha256=candidate_checkpoint.content_sha256,
            checkpoint_round_trip_verified=checkpoint_round_trip_verified,
            rollback_drill_verified=rollback_drill_verified,
            exact_pe_credit_lineage_verified=exact_pe_credit_lineage_verified,
            evaluation_writeback_allowed=False,
            production_default_changed=False,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": OPERATIONS_BENCHMARK_SCHEMA_VERSION,
            "report_id": self.report_id,
            "content_sha256": self.content_sha256,
            "protocol_id": self.protocol_id,
            "evidence_scope": self.evidence_scope,
            "preregistration_sha256": self.preregistration_sha256,
            "scenario_set_sha256": self.scenario_set_sha256,
            "seed": self.seed,
            "arms": [item.to_json() for item in self.arms],
            "primary_baseline_arm": self.primary_baseline_arm,
            "learned_arm": self.learned_arm,
            "validation_delta": self.validation_delta,
            "paired_delta_lower_95": self.paired_delta_lower_95,
            "candidate_checkpoint_id": self.candidate_checkpoint_id,
            "candidate_checkpoint_sha256": self.candidate_checkpoint_sha256,
            "checkpoint_round_trip_verified": self.checkpoint_round_trip_verified,
            "rollback_drill_verified": self.rollback_drill_verified,
            "exact_pe_credit_lineage_verified": self.exact_pe_credit_lineage_verified,
            "evaluation_writeback_allowed": self.evaluation_writeback_allowed,
            "production_default_changed": self.production_default_changed,
        }

    @classmethod
    def from_json(
        cls,
        payload: Mapping[str, object],
    ) -> "OperationsPolicyBenchmarkReport":
        fields = frozenset(
            {
                "schema_version",
                "report_id",
                "content_sha256",
                "protocol_id",
                "evidence_scope",
                "preregistration_sha256",
                "scenario_set_sha256",
                "seed",
                "arms",
                "primary_baseline_arm",
                "learned_arm",
                "validation_delta",
                "paired_delta_lower_95",
                "candidate_checkpoint_id",
                "candidate_checkpoint_sha256",
                "checkpoint_round_trip_verified",
                "rollback_drill_verified",
                "exact_pe_credit_lineage_verified",
                "evaluation_writeback_allowed",
                "production_default_changed",
            }
        )
        _strict_payload(payload, fields=fields)
        if payload["schema_version"] != OPERATIONS_BENCHMARK_SCHEMA_VERSION:
            raise ValueError("unsupported Operations benchmark schema")
        report = cls(
            report_id=_require_text("report_id", payload["report_id"]),
            content_sha256=_require_text(
                "content_sha256",
                payload["content_sha256"],
            ),
            protocol_id=_require_text("protocol_id", payload["protocol_id"]),
            evidence_scope=_require_text(
                "evidence_scope",
                payload["evidence_scope"],
            ),
            preregistration_sha256=_require_text(
                "preregistration_sha256",
                payload["preregistration_sha256"],
            ),
            scenario_set_sha256=_require_text(
                "scenario_set_sha256",
                payload["scenario_set_sha256"],
            ),
            seed=_integer("seed", payload["seed"]),
            arms=tuple(
                OperationsBenchmarkArmResult.from_json(_mapping("arms[]", item))
                for item in _array("arms", payload["arms"])
            ),
            primary_baseline_arm=_require_text(
                "primary_baseline_arm",
                payload["primary_baseline_arm"],
            ),
            learned_arm=_require_text("learned_arm", payload["learned_arm"]),
            validation_delta=_number(
                "validation_delta",
                payload["validation_delta"],
            ),
            paired_delta_lower_95=_number(
                "paired_delta_lower_95",
                payload["paired_delta_lower_95"],
            ),
            candidate_checkpoint_id=_require_text(
                "candidate_checkpoint_id",
                payload["candidate_checkpoint_id"],
            ),
            candidate_checkpoint_sha256=_require_text(
                "candidate_checkpoint_sha256",
                payload["candidate_checkpoint_sha256"],
            ),
            checkpoint_round_trip_verified=_boolean(
                "checkpoint_round_trip_verified",
                payload["checkpoint_round_trip_verified"],
            ),
            rollback_drill_verified=_boolean(
                "rollback_drill_verified",
                payload["rollback_drill_verified"],
            ),
            exact_pe_credit_lineage_verified=_boolean(
                "exact_pe_credit_lineage_verified",
                payload["exact_pe_credit_lineage_verified"],
            ),
            evaluation_writeback_allowed=_boolean(
                "evaluation_writeback_allowed",
                payload["evaluation_writeback_allowed"],
            ),
            production_default_changed=_boolean(
                "production_default_changed",
                payload["production_default_changed"],
            ),
        )
        core = report.to_json()
        core.pop("report_id")
        core.pop("content_sha256")
        if stable_content_sha256(core) != report.content_sha256:
            raise ValueError("Operations benchmark digest mismatch")
        return report


@dataclass(frozen=True)
class OperationsPromotionReview:
    review_id: str
    content_sha256: str
    benchmark_report_id: str
    proposal_target: str
    desired_gate: ModificationGate
    old_value_hash: str
    new_value_hash: str
    validation_delta: float
    capacity_cost: float
    rollback_evidence: str
    decision: GateDecision
    blocking_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_content_id(
            "review_id",
            self.review_id,
            "operations-promotion-review:",
        )
        _require_sha256("content_sha256", self.content_sha256)
        if self.review_id != f"operations-promotion-review:{self.content_sha256}":
            raise ValueError("promotion review id/content mismatch")
        _require_content_id(
            "benchmark_report_id",
            self.benchmark_report_id,
            "operations-policy-benchmark:",
        )
        if self.proposal_target != OPERATIONS_PROMOTION_TARGET:
            raise ValueError("Operations promotion target drift")
        if self.desired_gate is not ModificationGate.OFFLINE:
            raise ValueError("Operations activation requires OFFLINE ModificationGate")
        _require_sha256("old_value_hash", self.old_value_hash)
        _require_sha256("new_value_hash", self.new_value_hash)
        if not math.isfinite(self.validation_delta) or not math.isfinite(self.capacity_cost):
            raise ValueError("promotion values must be finite")
        _require_text("rollback_evidence", self.rollback_evidence)
        if self.decision is GateDecision.ALLOW and self.blocking_reasons:
            raise ValueError("ALLOW review cannot have blocking reasons")
        if self.decision is GateDecision.BLOCK and not self.blocking_reasons:
            raise ValueError("BLOCK review requires blocking reasons")

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": OPERATIONS_PROMOTION_REVIEW_SCHEMA_VERSION,
            "review_id": self.review_id,
            "content_sha256": self.content_sha256,
            "benchmark_report_id": self.benchmark_report_id,
            "proposal_target": self.proposal_target,
            "desired_gate": self.desired_gate.value,
            "old_value_hash": self.old_value_hash,
            "new_value_hash": self.new_value_hash,
            "validation_delta": self.validation_delta,
            "capacity_cost": self.capacity_cost,
            "rollback_evidence": self.rollback_evidence,
            "decision": self.decision.value,
            "blocking_reasons": list(self.blocking_reasons),
        }

    @classmethod
    def from_json(
        cls,
        payload: Mapping[str, object],
    ) -> "OperationsPromotionReview":
        fields = frozenset(
            {
                "schema_version",
                "review_id",
                "content_sha256",
                "benchmark_report_id",
                "proposal_target",
                "desired_gate",
                "old_value_hash",
                "new_value_hash",
                "validation_delta",
                "capacity_cost",
                "rollback_evidence",
                "decision",
                "blocking_reasons",
            }
        )
        _strict_payload(payload, fields=fields)
        if payload["schema_version"] != OPERATIONS_PROMOTION_REVIEW_SCHEMA_VERSION:
            raise ValueError("unsupported Operations promotion review schema")
        review = cls(
            review_id=_require_text("review_id", payload["review_id"]),
            content_sha256=_require_text(
                "content_sha256",
                payload["content_sha256"],
            ),
            benchmark_report_id=_require_text(
                "benchmark_report_id",
                payload["benchmark_report_id"],
            ),
            proposal_target=_require_text(
                "proposal_target",
                payload["proposal_target"],
            ),
            desired_gate=ModificationGate(
                _require_text("desired_gate", payload["desired_gate"])
            ),
            old_value_hash=_require_text(
                "old_value_hash",
                payload["old_value_hash"],
            ),
            new_value_hash=_require_text(
                "new_value_hash",
                payload["new_value_hash"],
            ),
            validation_delta=_number(
                "validation_delta",
                payload["validation_delta"],
            ),
            capacity_cost=_number("capacity_cost", payload["capacity_cost"]),
            rollback_evidence=_require_text(
                "rollback_evidence",
                payload["rollback_evidence"],
            ),
            decision=GateDecision(
                _require_text("decision", payload["decision"])
            ),
            blocking_reasons=tuple(
                _require_text("blocking_reasons[]", item)
                for item in _array(
                    "blocking_reasons",
                    payload["blocking_reasons"],
                )
            ),
        )
        core = review.to_json()
        core.pop("review_id")
        core.pop("content_sha256")
        if stable_content_sha256(core) != review.content_sha256:
            raise ValueError("Operations promotion review digest mismatch")
        return review


@dataclass(frozen=True)
class OperationsPolicyActivationReceipt:
    activation_receipt_id: str
    content_sha256: str
    review_id: str
    benchmark_report_id: str
    proposal_target: str
    activation_scope: str
    policy_artifact_id: str
    candidate_checkpoint_id: str
    candidate_checkpoint_sha256: str
    candidate_update_count: int
    processed_credit_prefix_sha256: str
    from_wiring_level: WiringLevel
    to_wiring_level: WiringLevel
    rollback_config_field: str
    issued_at_ms: int

    def __post_init__(self) -> None:
        _require_content_id(
            "activation_receipt_id",
            self.activation_receipt_id,
            "operations-policy-activation:",
        )
        _require_sha256("content_sha256", self.content_sha256)
        if self.activation_receipt_id != f"operations-policy-activation:{self.content_sha256}":
            raise ValueError("activation receipt id/content mismatch")
        _require_content_id(
            "review_id",
            self.review_id,
            "operations-promotion-review:",
        )
        _require_content_id(
            "benchmark_report_id",
            self.benchmark_report_id,
            "operations-policy-benchmark:",
        )
        if self.proposal_target != OPERATIONS_PROMOTION_TARGET:
            raise ValueError("activation target drift")
        if self.activation_scope != OPERATIONS_ACTIVATION_SCOPE:
            raise ValueError("activation scope drift")
        _require_text("policy_artifact_id", self.policy_artifact_id)
        _require_content_id(
            "candidate_checkpoint_id",
            self.candidate_checkpoint_id,
            "operations-policy-checkpoint:",
        )
        _require_sha256("candidate_checkpoint_sha256", self.candidate_checkpoint_sha256)
        if self.candidate_checkpoint_id != f"operations-policy-checkpoint:{self.candidate_checkpoint_sha256}":
            raise ValueError("activation checkpoint id/content mismatch")
        if self.candidate_update_count < 1:
            raise ValueError("activation requires a learned checkpoint")
        _require_sha256(
            "processed_credit_prefix_sha256",
            self.processed_credit_prefix_sha256,
        )
        if self.from_wiring_level is not WiringLevel.SHADOW or self.to_wiring_level is not WiringLevel.ACTIVE:
            raise ValueError("activation must be SHADOW -> ACTIVE")
        if self.rollback_config_field != "AUTOCOMPANY_OPERATIONS_BRAIN_WIRING":
            raise ValueError("activation rollback field drift")
        if self.issued_at_ms < 0:
            raise ValueError("issued_at_ms must be non-negative")

    def authorizes(self, checkpoint: OperationsPolicyCheckpoint) -> bool:
        if checkpoint.artifact_id != self.policy_artifact_id:
            return False
        if checkpoint.update_count < self.candidate_update_count:
            return False
        if checkpoint.update_count == self.candidate_update_count:
            return checkpoint.checkpoint_id == self.candidate_checkpoint_id
        prefix = checkpoint.processed_credit_ids[: self.candidate_update_count]
        return stable_content_sha256({"processed_credit_ids": list(prefix)}) == (
            self.processed_credit_prefix_sha256
        )

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": OPERATIONS_ACTIVATION_SCHEMA_VERSION,
            "activation_receipt_id": self.activation_receipt_id,
            "content_sha256": self.content_sha256,
            "review_id": self.review_id,
            "benchmark_report_id": self.benchmark_report_id,
            "proposal_target": self.proposal_target,
            "activation_scope": self.activation_scope,
            "policy_artifact_id": self.policy_artifact_id,
            "candidate_checkpoint_id": self.candidate_checkpoint_id,
            "candidate_checkpoint_sha256": self.candidate_checkpoint_sha256,
            "candidate_update_count": self.candidate_update_count,
            "processed_credit_prefix_sha256": self.processed_credit_prefix_sha256,
            "from_wiring_level": self.from_wiring_level.value,
            "to_wiring_level": self.to_wiring_level.value,
            "rollback_config_field": self.rollback_config_field,
            "issued_at_ms": self.issued_at_ms,
        }

    @classmethod
    def from_json(
        cls,
        payload: Mapping[str, object],
    ) -> "OperationsPolicyActivationReceipt":
        fields = frozenset(
            {
                "schema_version",
                "activation_receipt_id",
                "content_sha256",
                "review_id",
                "benchmark_report_id",
                "proposal_target",
                "activation_scope",
                "policy_artifact_id",
                "candidate_checkpoint_id",
                "candidate_checkpoint_sha256",
                "candidate_update_count",
                "processed_credit_prefix_sha256",
                "from_wiring_level",
                "to_wiring_level",
                "rollback_config_field",
                "issued_at_ms",
            }
        )
        _strict_payload(payload, fields=fields)
        if payload["schema_version"] != OPERATIONS_ACTIVATION_SCHEMA_VERSION:
            raise ValueError("unsupported Operations activation schema")
        receipt = cls(
            activation_receipt_id=_require_text(
                "activation_receipt_id",
                payload["activation_receipt_id"],
            ),
            content_sha256=_require_text(
                "content_sha256",
                payload["content_sha256"],
            ),
            review_id=_require_text("review_id", payload["review_id"]),
            benchmark_report_id=_require_text(
                "benchmark_report_id",
                payload["benchmark_report_id"],
            ),
            proposal_target=_require_text(
                "proposal_target",
                payload["proposal_target"],
            ),
            activation_scope=_require_text(
                "activation_scope",
                payload["activation_scope"],
            ),
            policy_artifact_id=_require_text(
                "policy_artifact_id",
                payload["policy_artifact_id"],
            ),
            candidate_checkpoint_id=_require_text(
                "candidate_checkpoint_id",
                payload["candidate_checkpoint_id"],
            ),
            candidate_checkpoint_sha256=_require_text(
                "candidate_checkpoint_sha256",
                payload["candidate_checkpoint_sha256"],
            ),
            candidate_update_count=_integer(
                "candidate_update_count",
                payload["candidate_update_count"],
                minimum=1,
            ),
            processed_credit_prefix_sha256=_require_text(
                "processed_credit_prefix_sha256",
                payload["processed_credit_prefix_sha256"],
            ),
            from_wiring_level=WiringLevel(
                _require_text("from_wiring_level", payload["from_wiring_level"])
            ),
            to_wiring_level=WiringLevel(
                _require_text("to_wiring_level", payload["to_wiring_level"])
            ),
            rollback_config_field=_require_text(
                "rollback_config_field",
                payload["rollback_config_field"],
            ),
            issued_at_ms=_integer("issued_at_ms", payload["issued_at_ms"]),
        )
        core = receipt.to_json()
        core.pop("activation_receipt_id")
        core.pop("content_sha256")
        if stable_content_sha256(core) != receipt.content_sha256:
            raise ValueError("Operations activation receipt digest mismatch")
        return receipt


def review_operations_policy_promotion(
    *,
    report: OperationsPolicyBenchmarkReport,
    candidate_checkpoint: OperationsPolicyCheckpoint,
) -> OperationsPromotionReview:
    """Run benchmark admission and the system ModificationGate."""

    if OperationsPolicyBenchmarkReport.from_json(report.to_json()) != report:
        raise ValueError("benchmark report failed canonical round trip")
    if (
        OperationsPolicyCheckpoint.from_json(candidate_checkpoint.to_json())
        != candidate_checkpoint
    ):
        raise ValueError("candidate checkpoint failed canonical round trip")

    learned = next(item for item in report.arms if item.arm_id == report.learned_arm)
    baseline = next(
        item for item in report.arms if item.arm_id == report.primary_baseline_arm
    )
    blockers: list[str] = []
    if learned.training_cycles < 120:
        blockers.append("training_cycles_below_120")
    if len(learned.evaluation_utilities) < 60:
        blockers.append("evaluation_cycles_below_60")
    if report.validation_delta < 0.05:
        blockers.append("validation_delta_below_0.05")
    if report.paired_delta_lower_95 <= 0.0:
        blockers.append("paired_delta_lower_95_not_positive")
    if learned.mean_utility <= baseline.mean_utility:
        blockers.append("learned_policy_did_not_beat_primary_baseline")
    if learned.correct_action_rate < 0.75:
        blockers.append("correct_action_rate_below_0.75")
    if learned.favorable_rate < 0.75:
        blockers.append("favorable_rate_below_0.75")
    if not report.exact_pe_credit_lineage_verified:
        blockers.append("exact_pe_credit_lineage_not_verified")
    if learned.pe_credit_count != learned.policy_update_count:
        blockers.append("pe_credit_and_update_counts_differ")
    if not report.checkpoint_round_trip_verified:
        blockers.append("checkpoint_round_trip_not_verified")
    if not report.rollback_drill_verified:
        blockers.append("rollback_drill_not_verified")
    if report.candidate_checkpoint_id != candidate_checkpoint.checkpoint_id:
        blockers.append("candidate_checkpoint_lineage_mismatch")

    rollback_evidence = (
        f"operations-shadow-rollback:{report.report_id}:"
        "AUTOCOMPANY_OPERATIONS_BRAIN_WIRING"
        if report.rollback_drill_verified
        else ""
    )
    old_hash = stable_content_sha256(
        {"target": OPERATIONS_PROMOTION_TARGET, "wiring_level": "shadow"}
    )
    proposal = ModificationProposal(
        target=OPERATIONS_PROMOTION_TARGET,
        desired_gate=ModificationGate.OFFLINE,
        old_value_hash=old_hash,
        new_value_hash=candidate_checkpoint.content_sha256,
        justification=(
            "Promote only the benchmarked Operations policy artifact from "
            "SHADOW to the AutoCompany selection surface."
        ),
        is_reversible=True,
        validation_delta=report.validation_delta,
        capacity_cost=0.0,
        rollback_evidence=rollback_evidence,
    )
    evaluation_snapshot = EvaluationSnapshot(
        turn_scores=(
            EvaluationScore(
                family="safety",
                metric_name="contract_integrity",
                value=float(report.exact_pe_credit_lineage_verified),
                confidence=1.0,
                evidence=report.report_id,
            ),
            EvaluationScore(
                family="safety",
                metric_name="rollback_resilience",
                value=float(report.rollback_drill_verified),
                confidence=1.0,
                evidence=rollback_evidence or "rollback-evidence-missing",
            ),
            EvaluationScore(
                family="safety",
                metric_name="fallback_reliance",
                value=0.0,
                confidence=1.0,
                evidence=report.scenario_set_sha256,
            ),
        ),
        session_scores=(),
        alerts=(),
        structured_alerts=(),
        description=(
            "Operations benchmark readout for ModificationGate only; it is "
            "not a policy learning input."
        ),
    )
    blockers.extend(
        evaluate_gate_reasons(
            proposal=proposal,
            evaluation_snapshot=evaluation_snapshot,
            audit_required=False,
        )
    )
    reasons = tuple(dict.fromkeys(blockers))
    decision = GateDecision.BLOCK if reasons else GateDecision.ALLOW
    core = {
        "schema_version": OPERATIONS_PROMOTION_REVIEW_SCHEMA_VERSION,
        "benchmark_report_id": report.report_id,
        "proposal_target": proposal.target,
        "desired_gate": proposal.desired_gate.value,
        "old_value_hash": proposal.old_value_hash,
        "new_value_hash": proposal.new_value_hash,
        "validation_delta": proposal.validation_delta,
        "capacity_cost": proposal.capacity_cost,
        "rollback_evidence": proposal.rollback_evidence,
        "decision": decision.value,
        "blocking_reasons": list(reasons),
    }
    digest = stable_content_sha256(core)
    return OperationsPromotionReview(
        review_id=f"operations-promotion-review:{digest}",
        content_sha256=digest,
        benchmark_report_id=report.report_id,
        proposal_target=proposal.target,
        desired_gate=proposal.desired_gate,
        old_value_hash=proposal.old_value_hash,
        new_value_hash=proposal.new_value_hash,
        validation_delta=proposal.validation_delta,
        capacity_cost=proposal.capacity_cost,
        rollback_evidence=proposal.rollback_evidence,
        decision=decision,
        blocking_reasons=reasons,
    )


def issue_operations_policy_activation(
    *,
    review: OperationsPromotionReview,
    report: OperationsPolicyBenchmarkReport,
    candidate_checkpoint: OperationsPolicyCheckpoint,
    issued_at_ms: int,
) -> OperationsPolicyActivationReceipt:
    if OperationsPolicyBenchmarkReport.from_json(report.to_json()) != report:
        raise ValueError("benchmark report failed canonical round trip")
    if OperationsPromotionReview.from_json(review.to_json()) != review:
        raise ValueError("promotion review failed canonical round trip")
    if (
        OperationsPolicyCheckpoint.from_json(candidate_checkpoint.to_json())
        != candidate_checkpoint
    ):
        raise ValueError("candidate checkpoint failed canonical round trip")
    if review.decision is not GateDecision.ALLOW:
        raise ValueError("blocked promotion review cannot issue activation")
    if review.benchmark_report_id != report.report_id:
        raise ValueError("promotion review/report lineage mismatch")
    if review.new_value_hash != candidate_checkpoint.content_sha256:
        raise ValueError("promotion review/checkpoint lineage mismatch")
    prefix_sha256 = stable_content_sha256(
        {"processed_credit_ids": list(candidate_checkpoint.processed_credit_ids)}
    )
    core = {
        "schema_version": OPERATIONS_ACTIVATION_SCHEMA_VERSION,
        "review_id": review.review_id,
        "benchmark_report_id": report.report_id,
        "proposal_target": OPERATIONS_PROMOTION_TARGET,
        "activation_scope": OPERATIONS_ACTIVATION_SCOPE,
        "policy_artifact_id": candidate_checkpoint.artifact_id,
        "candidate_checkpoint_id": candidate_checkpoint.checkpoint_id,
        "candidate_checkpoint_sha256": candidate_checkpoint.content_sha256,
        "candidate_update_count": candidate_checkpoint.update_count,
        "processed_credit_prefix_sha256": prefix_sha256,
        "from_wiring_level": WiringLevel.SHADOW.value,
        "to_wiring_level": WiringLevel.ACTIVE.value,
        "rollback_config_field": "AUTOCOMPANY_OPERATIONS_BRAIN_WIRING",
        "issued_at_ms": issued_at_ms,
    }
    digest = stable_content_sha256(core)
    return OperationsPolicyActivationReceipt(
        activation_receipt_id=f"operations-policy-activation:{digest}",
        content_sha256=digest,
        review_id=review.review_id,
        benchmark_report_id=report.report_id,
        proposal_target=OPERATIONS_PROMOTION_TARGET,
        activation_scope=OPERATIONS_ACTIVATION_SCOPE,
        policy_artifact_id=candidate_checkpoint.artifact_id,
        candidate_checkpoint_id=candidate_checkpoint.checkpoint_id,
        candidate_checkpoint_sha256=candidate_checkpoint.content_sha256,
        candidate_update_count=candidate_checkpoint.update_count,
        processed_credit_prefix_sha256=prefix_sha256,
        from_wiring_level=WiringLevel.SHADOW,
        to_wiring_level=WiringLevel.ACTIVE,
        rollback_config_field="AUTOCOMPANY_OPERATIONS_BRAIN_WIRING",
        issued_at_ms=issued_at_ms,
    )


def validate_operations_policy_activation(
    *,
    report: OperationsPolicyBenchmarkReport,
    review: OperationsPromotionReview,
    receipt: OperationsPolicyActivationReceipt,
    candidate_checkpoint: OperationsPolicyCheckpoint,
) -> None:
    """Recompute the complete gate bundle before a deploy consumes it."""

    if report.candidate_checkpoint_id != candidate_checkpoint.checkpoint_id:
        raise ValueError("benchmark report/checkpoint lineage mismatch")
    expected_review = review_operations_policy_promotion(
        report=report,
        candidate_checkpoint=candidate_checkpoint,
    )
    if expected_review != review:
        raise ValueError("promotion review does not match recomputed ModificationGate")
    if review.decision is not GateDecision.ALLOW:
        raise ValueError("blocked promotion review cannot authorize deployment")
    expected_receipt = issue_operations_policy_activation(
        review=review,
        report=report,
        candidate_checkpoint=candidate_checkpoint,
        issued_at_ms=receipt.issued_at_ms,
    )
    if expected_receipt != receipt:
        raise ValueError("activation receipt does not match the exact gate bundle")
    if not receipt.authorizes(candidate_checkpoint):
        raise ValueError("activation receipt does not authorize candidate checkpoint")


__all__ = (
    "OPERATIONS_ACTIVATION_SCOPE",
    "OPERATIONS_BENCHMARK_EVIDENCE_SCOPE",
    "OPERATIONS_BENCHMARK_PROTOCOL_ID",
    "OPERATIONS_PROMOTION_TARGET",
    "OperationsBenchmarkArmResult",
    "OperationsPolicyActivationReceipt",
    "OperationsPolicyBenchmarkReport",
    "OperationsPromotionReview",
    "issue_operations_policy_activation",
    "review_operations_policy_promotion",
    "validate_operations_policy_activation",
)
