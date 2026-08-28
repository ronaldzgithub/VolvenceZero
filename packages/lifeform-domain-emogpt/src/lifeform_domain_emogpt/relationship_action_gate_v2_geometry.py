"""Outcome-free geometric reachability for relationship action gate v2.

The gate owner solves a four-dimensional bounded geometry problem from a
complete frozen policy and a typed owner forecast.  It never reads outcomes,
prediction errors, credit, evaluation, or judge signals and never updates a
policy.  Geometry is therefore neither PE-credit achievability nor an effect
claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from volvence_zero.social_cognition import (
    PreferenceActionForecast,
    preference_action_forecast_to_payload,
)

from lifeform_domain_emogpt import relationship_action_gate as legacy
from lifeform_domain_emogpt.relationship_action_contracts import RelationshipAction
from lifeform_domain_emogpt.relationship_action_gate_v2 import (
    RELATIONSHIP_ACTION_GATE_V2_FEATURE_ORDER,
    RELATIONSHIP_ACTION_GATE_V2_OPERATOR_ID,
    RELATIONSHIP_ACTION_GATE_V2_THRESHOLD_RULE,
    RelationshipActionGateV2FrozenPolicy,
    _RelationshipActionGateV2CounterfactualResponse,
    _relationship_action_gate_v2_counterfactual_response,
    _relationship_action_gate_v2_gate_response_from_logit,
)


RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_DOSE_SCHEMA_VERSION = "relationship-action-gate-v2-geometric-dose.v1"
RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_NORM_RESULT_SCHEMA_VERSION = (
    "relationship-action-gate-v2-geometric-norm-result.v1"
)
RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_RECEIPT_SCHEMA_VERSION = (
    "relationship-action-gate-v2-geometric-reachability-receipt.v1"
)
RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_OPERATOR_ID = "relationship-action-gate-v2-outcome-free-box-reachability.v1"
RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_L2_SOLVER_ID = "deterministic-box-l2-active-set-water-filling.v1"
RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_LINF_SOLVER_ID = "deterministic-box-linf-breakpoint-water-filling.v1"
RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_WITNESS_SCOPE = "per_forecast_individual_not_joint_or_simultaneous"
RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_CONTINUOUS_SOLUTION_SEMANTICS = (
    "deterministic_binary64_continuous_solver_candidate_not_exact_optimum_or_executable_gate_witness"
)
RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_ROBUST_WITNESS_SEMANTICS = "binary64_exact_gate_replay_distance_upper_bound"
_FEATURE_COUNT = len(RELATIONSHIP_ACTION_GATE_V2_FEATURE_ORDER)
_ZERO_HEX = (0.0).hex()
_RECEIPT_PREFIX = "relationship-action-gate-v2-geometric-receipt-sha256:"


def _float_hex(value: float) -> str:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError("geometric values must be exact finite floats")
    return _ZERO_HEX if value == 0.0 else value.hex()


def _decode_float_hex(value: object, field_name: str) -> float:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be an exact canonical float hex string")
    try:
        numeric = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an exact canonical float hex string") from exc
    if not math.isfinite(numeric) or _float_hex(numeric) != value:
        raise ValueError(f"{field_name} must be an exact canonical finite float hex string")
    return numeric


def _decode_vector(values: object, field_name: str) -> tuple[float, ...]:
    if type(values) is not tuple or len(values) != _FEATURE_COUNT:
        raise ValueError(f"{field_name} must contain {_FEATURE_COUNT} values")
    return tuple(_decode_float_hex(value, f"{field_name}[{index}]") for index, value in enumerate(values))


def _require_bool(value: object, field_name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be an exact bool")


def _require_coordinate_indices(values: object, field_name: str) -> None:
    if type(values) is not tuple or any(type(value) is not int for value in values):
        raise TypeError(f"{field_name} must be an exact tuple of integers")
    if tuple(sorted(set(values))) != values:
        raise ValueError(f"{field_name} must be sorted and unique")
    if any(value < 0 or value >= _FEATURE_COUNT for value in values):
        raise ValueError(f"{field_name} contains an invalid coordinate")


@dataclass(frozen=True)
class RelationshipActionGateV2GeometricDose:
    absolute_parameter_cap_hex: str
    coordinate_delta_cap_hex: tuple[str, ...]
    robust_logit_clearance_hex: str
    schema_version: str = RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_DOSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_DOSE_SCHEMA_VERSION:
            raise ValueError("relationship action gate v2 geometric dose schema mismatch")
        if _decode_float_hex(self.absolute_parameter_cap_hex, "absolute_parameter_cap_hex") <= 0.0:
            raise ValueError("absolute_parameter_cap_hex must be positive")
        caps = _decode_vector(self.coordinate_delta_cap_hex, "coordinate_delta_cap_hex")
        if any(value < 0.0 for value in caps) or not any(value > 0.0 for value in caps):
            raise ValueError("coordinate delta caps must be non-negative and not all zero")
        clearance = _decode_float_hex(
            self.robust_logit_clearance_hex,
            "robust_logit_clearance_hex",
        )
        if clearance <= 0.0:
            raise ValueError("robust_logit_clearance_hex must be positive")
        positive = _relationship_action_gate_v2_gate_response_from_logit(clearance)
        negative = _relationship_action_gate_v2_gate_response_from_logit(-clearance)
        if (
            positive[1] is not legacy.RelationshipGateAction.STEER
            or negative[1] is not legacy.RelationshipGateAction.NOOP
        ):
            raise ValueError("robust logit clearance must leave the production floating threshold band")

    @classmethod
    def create(
        cls,
        *,
        absolute_parameter_cap: float,
        coordinate_delta_cap: tuple[float, ...],
        robust_logit_clearance: float,
    ) -> "RelationshipActionGateV2GeometricDose":
        if type(absolute_parameter_cap) is not float:
            raise TypeError("absolute_parameter_cap must be an exact float")
        if type(coordinate_delta_cap) is not tuple or any(type(value) is not float for value in coordinate_delta_cap):
            raise TypeError("coordinate_delta_cap must be an exact tuple of floats")
        if type(robust_logit_clearance) is not float:
            raise TypeError("robust_logit_clearance must be an exact float")
        return cls(
            absolute_parameter_cap_hex=_float_hex(absolute_parameter_cap),
            coordinate_delta_cap_hex=tuple(_float_hex(value) for value in coordinate_delta_cap),
            robust_logit_clearance_hex=_float_hex(robust_logit_clearance),
        )

    @classmethod
    def from_payload(cls, payload: object) -> "RelationshipActionGateV2GeometricDose":
        raw = legacy._require_exact_mapping(
            payload,
            expected={
                "schema_version",
                "absolute_parameter_cap_hex",
                "coordinate_delta_cap_hex",
                "robust_logit_clearance_hex",
            },
            source="relationship action gate v2 geometric dose",
        )
        caps = raw["coordinate_delta_cap_hex"]
        if type(caps) is not list or any(type(value) is not str for value in caps):
            raise ValueError("coordinate_delta_cap_hex must be an array of strings")
        return cls(
            absolute_parameter_cap_hex=legacy._payload_text(
                raw,
                "absolute_parameter_cap_hex",
            ),
            coordinate_delta_cap_hex=tuple(caps),
            robust_logit_clearance_hex=legacy._payload_text(
                raw,
                "robust_logit_clearance_hex",
            ),
            schema_version=legacy._payload_text(raw, "schema_version"),
        )

    @property
    def absolute_parameter_cap(self) -> float:
        return _decode_float_hex(self.absolute_parameter_cap_hex, "absolute_parameter_cap_hex")

    @property
    def coordinate_delta_cap(self) -> tuple[float, ...]:
        return _decode_vector(self.coordinate_delta_cap_hex, "coordinate_delta_cap_hex")

    @property
    def robust_logit_clearance(self) -> float:
        return _decode_float_hex(
            self.robust_logit_clearance_hex,
            "robust_logit_clearance_hex",
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "absolute_parameter_cap_hex": self.absolute_parameter_cap_hex,
            "coordinate_delta_cap_hex": list(self.coordinate_delta_cap_hex),
            "robust_logit_clearance_hex": self.robust_logit_clearance_hex,
        }


def _validate_optional_solution(
    *,
    feasible: bool,
    distance_hex: str | None,
    delta_hex: tuple[str, ...] | None,
    prefix: str,
) -> None:
    _require_bool(feasible, f"{prefix}_feasible")
    if not feasible:
        if distance_hex is not None or delta_hex is not None:
            raise ValueError(f"infeasible {prefix} solution must be empty")
        return
    if distance_hex is None or delta_hex is None:
        raise ValueError(f"feasible {prefix} solution must be complete")
    if _decode_float_hex(distance_hex, f"{prefix}_distance_hex") < 0.0:
        raise ValueError(f"{prefix} distance must be non-negative")
    _decode_vector(delta_hex, f"{prefix}_delta_hex")


@dataclass(frozen=True)
class RelationshipActionGateV2GeometricNormResult:
    norm_id: str
    solver_id: str
    boundary_distance_kind: str
    continuous_boundary_feasible: bool
    continuous_boundary_distance_hex: str | None
    continuous_boundary_delta_hex: tuple[str, ...] | None
    representable_robust_witness_exists: bool
    robust_witness_distance_upper_bound_hex: str | None
    robust_witness_actual_delta_hex: tuple[str, ...] | None
    robust_witness_weights_hex: tuple[str, ...] | None
    robust_witness_logit_hex: str | None
    robust_witness_probability_hex: str | None
    robust_witness_gate_action: legacy.RelationshipGateAction | None
    robust_witness_delivered_action_id: str | None
    robust_witness_cap_hit_coordinate_indices: tuple[int, ...]
    schema_version: str = RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_NORM_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_NORM_RESULT_SCHEMA_VERSION:
            raise ValueError("relationship action gate v2 geometric norm result schema mismatch")
        if self.norm_id not in {"l2", "linf"}:
            raise ValueError("norm_id must be l2 or linf")
        expected_solver = (
            RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_L2_SOLVER_ID
            if self.norm_id == "l2"
            else RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_LINF_SOLVER_ID
        )
        if self.solver_id != expected_solver:
            raise ValueError("geometric norm solver_id mismatch")
        if self.boundary_distance_kind not in {"minimum", "infimum"}:
            raise ValueError("boundary_distance_kind must be minimum or infimum")
        _validate_optional_solution(
            feasible=self.continuous_boundary_feasible,
            distance_hex=self.continuous_boundary_distance_hex,
            delta_hex=self.continuous_boundary_delta_hex,
            prefix="continuous_boundary",
        )
        if self.continuous_boundary_feasible:
            boundary_delta = _decode_vector(
                self.continuous_boundary_delta_hex,
                "continuous_boundary_delta_hex",
            )
            if self.continuous_boundary_distance_hex != _float_hex(_distance(self.norm_id, boundary_delta)):
                raise ValueError("continuous boundary distance differs from its delta")
        _require_bool(
            self.representable_robust_witness_exists,
            "representable_robust_witness_exists",
        )
        _require_coordinate_indices(
            self.robust_witness_cap_hit_coordinate_indices,
            "robust_witness_cap_hit_coordinate_indices",
        )
        projection = (
            self.robust_witness_distance_upper_bound_hex,
            self.robust_witness_actual_delta_hex,
            self.robust_witness_weights_hex,
            self.robust_witness_logit_hex,
            self.robust_witness_probability_hex,
            self.robust_witness_gate_action,
            self.robust_witness_delivered_action_id,
        )
        if not self.representable_robust_witness_exists:
            if any(value is not None for value in projection):
                raise ValueError("absent robust witness must have an empty projection")
            if self.robust_witness_cap_hit_coordinate_indices:
                raise ValueError("absent robust witness cannot report cap hits")
            return
        if any(value is None for value in projection):
            raise ValueError("representable robust witness must be complete")
        if (
            _decode_float_hex(
                self.robust_witness_distance_upper_bound_hex,
                "robust_witness_distance_upper_bound_hex",
            )
            < 0.0
        ):
            raise ValueError("robust witness distance must be non-negative")
        witness_delta = _decode_vector(
            self.robust_witness_actual_delta_hex,
            "robust_witness_actual_delta_hex",
        )
        if self.robust_witness_distance_upper_bound_hex != _float_hex(_distance(self.norm_id, witness_delta)):
            raise ValueError("robust witness distance differs from its actual delta")
        _decode_vector(self.robust_witness_weights_hex, "robust_witness_weights_hex")
        _decode_float_hex(self.robust_witness_logit_hex, "robust_witness_logit_hex")
        probability = _decode_float_hex(
            self.robust_witness_probability_hex,
            "robust_witness_probability_hex",
        )
        if not 0.0 <= probability <= 1.0:
            raise ValueError("robust witness probability must be in [0, 1]")
        if type(self.robust_witness_gate_action) is not legacy.RelationshipGateAction:
            raise TypeError("robust witness gate action must be RelationshipGateAction")
        if self.robust_witness_delivered_action_id not in {action.value for action in RelationshipAction}:
            raise ValueError("robust witness delivered action is outside the action surface")

    def to_payload(self) -> dict[str, object]:
        def _vector(value: tuple[str, ...] | None) -> list[str] | None:
            return None if value is None else list(value)

        return {
            "schema_version": self.schema_version,
            "norm_id": self.norm_id,
            "solver_id": self.solver_id,
            "boundary_distance_kind": self.boundary_distance_kind,
            "continuous_solution_semantics": (RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_CONTINUOUS_SOLUTION_SEMANTICS),
            "continuous_boundary_feasible": self.continuous_boundary_feasible,
            "continuous_boundary_distance_hex": self.continuous_boundary_distance_hex,
            "continuous_boundary_delta_hex": _vector(self.continuous_boundary_delta_hex),
            "robust_witness_semantics": (RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_ROBUST_WITNESS_SEMANTICS),
            "representable_robust_witness_exists": (self.representable_robust_witness_exists),
            "robust_witness_distance_upper_bound_hex": (self.robust_witness_distance_upper_bound_hex),
            "robust_witness_actual_delta_hex": _vector(self.robust_witness_actual_delta_hex),
            "robust_witness_weights_hex": _vector(self.robust_witness_weights_hex),
            "robust_witness_logit_hex": self.robust_witness_logit_hex,
            "robust_witness_probability_hex": self.robust_witness_probability_hex,
            "robust_witness_gate_action": (
                None if self.robust_witness_gate_action is None else self.robust_witness_gate_action.value
            ),
            "robust_witness_delivered_action_id": (self.robust_witness_delivered_action_id),
            "robust_witness_cap_hit_coordinate_indices": list(self.robust_witness_cap_hit_coordinate_indices),
        }


def _validate_actual_projection(
    *,
    baseline_weights: tuple[float, ...],
    delta_hex: tuple[str, ...],
    weights_hex: tuple[str, ...],
    lower: tuple[float, ...],
    upper: tuple[float, ...],
    absolute_cap: float,
    label: str,
) -> None:
    delta = _decode_vector(delta_hex, f"{label}_delta_hex")
    weights = _decode_vector(weights_hex, f"{label}_weights_hex")
    for index, (baseline, change, weight, low, high) in enumerate(
        zip(baseline_weights, delta, weights, lower, upper, strict=True)
    ):
        if weight - baseline != change:
            raise ValueError(f"{label} coordinate {index} is not an actual delta")
        if not low <= change <= high or abs(weight) > absolute_cap:
            raise ValueError(f"{label} coordinate {index} exceeds frozen dose")


@dataclass(frozen=True)
class RelationshipActionGateV2GeometricReachabilityReceipt:
    frozen_policy_id: str
    artifact_id: str
    checkpoint_content_sha256: str
    forecast_sha256: str
    forecast_id: str
    decision_id: str
    dose: RelationshipActionGateV2GeometricDose
    baseline_weights_hex: tuple[str, ...]
    baseline_features_hex: tuple[str, ...]
    baseline_logit_hex: str
    baseline_probability_hex: str
    baseline_gate_action: legacy.RelationshipGateAction
    baseline_delivered_action_id: str
    recommended_action_id: str
    target_gate_action: legacy.RelationshipGateAction
    target_delivered_action_id: str
    effective_lower_delta_hex: tuple[str, ...]
    effective_upper_delta_hex: tuple[str, ...]
    directional_endpoint_delta_hex: tuple[str, ...]
    directional_endpoint_weights_hex: tuple[str, ...]
    directional_endpoint_logit_hex: str
    directional_endpoint_probability_hex: str
    directional_endpoint_gate_action: legacy.RelationshipGateAction
    directional_endpoint_delivered_action_id: str
    l2_result: RelationshipActionGateV2GeometricNormResult
    linf_result: RelationshipActionGateV2GeometricNormResult
    analytic_boundary_feasible: bool
    operational_gate_flip_reachable: bool
    robust_gate_flip_witness_exists: bool
    operational_delivered_action_flip_reachable: bool
    robust_delivered_action_flip_witness_exists: bool
    logit_sign_action_consistent: bool
    floating_threshold_band_detected: bool
    reason_codes: tuple[str, ...]
    schema_version: str = RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_RECEIPT_SCHEMA_VERSION:
            raise ValueError("relationship action gate v2 geometric receipt schema mismatch")
        legacy._require_content_addressed_id(self.frozen_policy_id, "frozen_policy_id")
        legacy._require_content_addressed_id(self.artifact_id, "artifact_id")
        legacy._require_sha256(self.checkpoint_content_sha256, "checkpoint_content_sha256")
        legacy._require_sha256(self.forecast_sha256, "forecast_sha256")
        legacy._require_text(self.forecast_id, "forecast_id")
        legacy._require_text(self.decision_id, "decision_id")
        if type(self.dose) is not RelationshipActionGateV2GeometricDose:
            raise TypeError("dose must be RelationshipActionGateV2GeometricDose")
        if type(self.l2_result) is not RelationshipActionGateV2GeometricNormResult:
            raise TypeError("l2_result must be RelationshipActionGateV2GeometricNormResult")
        if type(self.linf_result) is not RelationshipActionGateV2GeometricNormResult:
            raise TypeError("linf_result must be RelationshipActionGateV2GeometricNormResult")
        if self.l2_result.norm_id != "l2" or self.linf_result.norm_id != "linf":
            raise ValueError("geometric receipt norm results are misbound")

        baseline_weights = _decode_vector(self.baseline_weights_hex, "baseline_weights_hex")
        baseline_features = _decode_vector(
            self.baseline_features_hex,
            "baseline_features_hex",
        )
        baseline_logit = _decode_float_hex(self.baseline_logit_hex, "baseline_logit_hex")
        baseline_probability = _decode_float_hex(
            self.baseline_probability_hex,
            "baseline_probability_hex",
        )
        endpoint_probability = _decode_float_hex(
            self.directional_endpoint_probability_hex,
            "directional_endpoint_probability_hex",
        )
        endpoint_logit = _decode_float_hex(
            self.directional_endpoint_logit_hex,
            "directional_endpoint_logit_hex",
        )
        if not 0.0 <= baseline_probability <= 1.0 or not 0.0 <= endpoint_probability <= 1.0:
            raise ValueError("geometric probabilities must be in [0, 1]")
        for name, action in (
            ("baseline_gate_action", self.baseline_gate_action),
            ("target_gate_action", self.target_gate_action),
            ("directional_endpoint_gate_action", self.directional_endpoint_gate_action),
        ):
            if type(action) is not legacy.RelationshipGateAction:
                raise TypeError(f"{name} must be RelationshipGateAction")
        expected_baseline_probability, expected_baseline_gate = _relationship_action_gate_v2_gate_response_from_logit(
            baseline_logit
        )
        expected_endpoint_probability, expected_endpoint_gate = _relationship_action_gate_v2_gate_response_from_logit(
            endpoint_logit
        )
        if baseline_probability != expected_baseline_probability:
            raise ValueError("baseline probability differs from exact gate replay")
        if endpoint_probability != expected_endpoint_probability:
            raise ValueError("endpoint probability differs from exact gate replay")
        if self.baseline_gate_action is not expected_baseline_gate:
            raise ValueError("baseline gate differs from the strict probability threshold")
        if self.directional_endpoint_gate_action is not expected_endpoint_gate:
            raise ValueError("endpoint gate differs from the strict probability threshold")
        if self.target_gate_action is self.baseline_gate_action:
            raise ValueError("target gate action must oppose baseline")

        action_surface = {action.value for action in RelationshipAction}
        for name, action_id in (
            ("recommended_action_id", self.recommended_action_id),
            ("baseline_delivered_action_id", self.baseline_delivered_action_id),
            ("target_delivered_action_id", self.target_delivered_action_id),
            (
                "directional_endpoint_delivered_action_id",
                self.directional_endpoint_delivered_action_id,
            ),
        ):
            if action_id not in action_surface:
                raise ValueError(f"{name} is outside the action surface")
        noop = RelationshipAction.NEUTRAL_NOOP.value
        baseline_delivered = (
            self.recommended_action_id if self.baseline_gate_action is legacy.RelationshipGateAction.STEER else noop
        )
        target_delivered = (
            self.recommended_action_id if self.target_gate_action is legacy.RelationshipGateAction.STEER else noop
        )
        endpoint_delivered = (
            self.recommended_action_id
            if self.directional_endpoint_gate_action is legacy.RelationshipGateAction.STEER
            else noop
        )
        if self.baseline_delivered_action_id != baseline_delivered:
            raise ValueError("baseline delivered action differs from gate semantics")
        if self.target_delivered_action_id != target_delivered:
            raise ValueError("target delivered action differs from gate semantics")
        if self.directional_endpoint_delivered_action_id != endpoint_delivered:
            raise ValueError("endpoint delivered action differs from gate semantics")

        lower = _decode_vector(self.effective_lower_delta_hex, "effective_lower_delta_hex")
        upper = _decode_vector(self.effective_upper_delta_hex, "effective_upper_delta_hex")
        if any(low > 0.0 or high < 0.0 or low > high for low, high in zip(lower, upper, strict=True)):
            raise ValueError("effective delta bounds must contain baseline")
        _validate_actual_projection(
            baseline_weights=baseline_weights,
            delta_hex=self.directional_endpoint_delta_hex,
            weights_hex=self.directional_endpoint_weights_hex,
            lower=lower,
            upper=upper,
            absolute_cap=self.dose.absolute_parameter_cap,
            label="directional_endpoint",
        )
        endpoint_weights = _decode_vector(
            self.directional_endpoint_weights_hex,
            "directional_endpoint_weights_hex",
        )
        replayed_baseline_logit = math.fsum(
            weight * feature
            for weight, feature in zip(
                baseline_weights,
                baseline_features,
                strict=True,
            )
        )
        replayed_endpoint_logit = math.fsum(
            weight * feature
            for weight, feature in zip(
                endpoint_weights,
                baseline_features,
                strict=True,
            )
        )
        if baseline_logit != replayed_baseline_logit:
            raise ValueError("baseline logit differs from exact feature replay")
        if endpoint_logit != replayed_endpoint_logit:
            raise ValueError("endpoint logit differs from exact feature replay")

        bool_fields = (
            "analytic_boundary_feasible",
            "operational_gate_flip_reachable",
            "robust_gate_flip_witness_exists",
            "operational_delivered_action_flip_reachable",
            "robust_delivered_action_flip_witness_exists",
            "logit_sign_action_consistent",
            "floating_threshold_band_detected",
        )
        for name in bool_fields:
            _require_bool(getattr(self, name), name)
        if self.l2_result.continuous_boundary_feasible != self.linf_result.continuous_boundary_feasible:
            raise ValueError("L2/Linf boundary feasibility must agree")
        if self.analytic_boundary_feasible != self.l2_result.continuous_boundary_feasible:
            raise ValueError("analytic boundary closure differs from norm receipts")
        if self.l2_result.representable_robust_witness_exists != self.linf_result.representable_robust_witness_exists:
            raise ValueError("L2/Linf robust witness feasibility must agree")
        if self.robust_gate_flip_witness_exists != self.l2_result.representable_robust_witness_exists:
            raise ValueError("robust witness closure differs from norm receipts")
        expected_kind = "infimum" if self.target_gate_action is legacy.RelationshipGateAction.STEER else "minimum"
        if any(result.boundary_distance_kind != expected_kind for result in (self.l2_result, self.linf_result)):
            raise ValueError("boundary kind differs from strict gate target")
        target_sign = 1.0 if self.target_gate_action is legacy.RelationshipGateAction.STEER else -1.0
        boundary_direction = tuple(target_sign * feature for feature in baseline_features)
        boundary_progress = max(0.0, -(target_sign * baseline_logit))
        for result in (self.l2_result, self.linf_result):
            if not result.continuous_boundary_feasible:
                continue
            continuous_delta = _decode_vector(
                result.continuous_boundary_delta_hex,
                f"{result.norm_id}_continuous_boundary_delta_hex",
            )
            if any(not low <= change <= high for change, low, high in zip(continuous_delta, lower, upper, strict=True)):
                raise ValueError("continuous boundary candidate exceeds frozen dose")
            achieved_progress = math.fsum(
                coefficient * change for coefficient, change in zip(boundary_direction, continuous_delta, strict=True)
            )
            if achieved_progress < boundary_progress:
                raise ValueError("continuous boundary candidate does not reach its algebraic boundary")

        operational = self.directional_endpoint_gate_action is self.target_gate_action
        robust = self.robust_gate_flip_witness_exists
        if self.operational_gate_flip_reachable != operational:
            raise ValueError("operational reachability differs from endpoint replay")
        if self.operational_delivered_action_flip_reachable != (
            operational and self.baseline_delivered_action_id != self.directional_endpoint_delivered_action_id
        ):
            raise ValueError("operational delivered action closure is inconsistent")
        if self.robust_delivered_action_flip_witness_exists != (
            robust and self.baseline_delivered_action_id != self.target_delivered_action_id
        ):
            raise ValueError("robust delivered action closure is inconsistent")
        expected_sign = (baseline_logit > 0.0) is (self.baseline_gate_action is legacy.RelationshipGateAction.STEER)
        if self.logit_sign_action_consistent != expected_sign:
            raise ValueError("baseline logit sign closure is inconsistent")
        if self.floating_threshold_band_detected == expected_sign:
            raise ValueError("floating threshold band must negate sign consistency")

        if robust:
            for result in (self.l2_result, self.linf_result):
                if result.robust_witness_gate_action is not self.target_gate_action:
                    raise ValueError("robust norm witness does not flip exact gate")
                if result.robust_witness_delivered_action_id != self.target_delivered_action_id:
                    raise ValueError("robust norm witness differs from target action")
                _validate_actual_projection(
                    baseline_weights=baseline_weights,
                    delta_hex=result.robust_witness_actual_delta_hex,
                    weights_hex=result.robust_witness_weights_hex,
                    lower=lower,
                    upper=upper,
                    absolute_cap=self.dose.absolute_parameter_cap,
                    label=f"{result.norm_id}_robust_witness",
                )
                witness_delta = _decode_vector(
                    result.robust_witness_actual_delta_hex,
                    f"{result.norm_id}_robust_witness_actual_delta_hex",
                )
                if result.robust_witness_cap_hit_coordinate_indices != _cap_hits(witness_delta, lower, upper):
                    raise ValueError("robust norm witness cap-hit indices differ from its actual delta")
                witness_weights = _decode_vector(
                    result.robust_witness_weights_hex,
                    f"{result.norm_id}_robust_witness_weights_hex",
                )
                witness_logit = _decode_float_hex(
                    result.robust_witness_logit_hex,
                    f"{result.norm_id}_robust_witness_logit_hex",
                )
                witness_probability = _decode_float_hex(
                    result.robust_witness_probability_hex,
                    f"{result.norm_id}_robust_witness_probability_hex",
                )
                expected_probability, expected_gate = _relationship_action_gate_v2_gate_response_from_logit(
                    witness_logit
                )
                replayed_witness_logit = math.fsum(
                    weight * feature for weight, feature in zip(witness_weights, baseline_features, strict=True)
                )
                signed_logit = (
                    witness_logit if self.target_gate_action is legacy.RelationshipGateAction.STEER else -witness_logit
                )
                if (
                    witness_logit != replayed_witness_logit
                    or witness_probability != expected_probability
                    or result.robust_witness_gate_action is not expected_gate
                    or signed_logit < self.dose.robust_logit_clearance
                ):
                    raise ValueError("robust norm witness fails exact clearance replay")
        if type(self.reason_codes) is not tuple or not self.reason_codes:
            raise ValueError("reason_codes must be a non-empty exact tuple")
        if any(type(value) is not str or not value for value in self.reason_codes):
            raise TypeError("reason_codes must contain non-empty exact strings")
        if tuple(dict.fromkeys(self.reason_codes)) != self.reason_codes:
            raise ValueError("reason_codes must be unique and ordered")

    @property
    def receipt_id(self) -> str:
        return f"{_RECEIPT_PREFIX}{legacy._canonical_sha256(self._core_payload())}"

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "operator_id": RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_OPERATOR_ID,
            "gate_operator_id": RELATIONSHIP_ACTION_GATE_V2_OPERATOR_ID,
            "feature_order": list(RELATIONSHIP_ACTION_GATE_V2_FEATURE_ORDER),
            "threshold_rule": RELATIONSHIP_ACTION_GATE_V2_THRESHOLD_RULE,
            "witness_scope": RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_WITNESS_SCOPE,
            "frozen_policy_id": self.frozen_policy_id,
            "artifact_id": self.artifact_id,
            "checkpoint_content_sha256": self.checkpoint_content_sha256,
            "forecast_sha256": self.forecast_sha256,
            "forecast_id": self.forecast_id,
            "decision_id": self.decision_id,
            "dose": self.dose.to_payload(),
            "baseline_weights_hex": list(self.baseline_weights_hex),
            "baseline_features_hex": list(self.baseline_features_hex),
            "baseline_logit_hex": self.baseline_logit_hex,
            "baseline_probability_hex": self.baseline_probability_hex,
            "baseline_gate_action": self.baseline_gate_action.value,
            "baseline_delivered_action_id": self.baseline_delivered_action_id,
            "recommended_action_id": self.recommended_action_id,
            "target_gate_action": self.target_gate_action.value,
            "target_delivered_action_id": self.target_delivered_action_id,
            "effective_lower_delta_hex": list(self.effective_lower_delta_hex),
            "effective_upper_delta_hex": list(self.effective_upper_delta_hex),
            "directional_endpoint_delta_hex": list(self.directional_endpoint_delta_hex),
            "directional_endpoint_weights_hex": list(self.directional_endpoint_weights_hex),
            "directional_endpoint_logit_hex": self.directional_endpoint_logit_hex,
            "directional_endpoint_probability_hex": (self.directional_endpoint_probability_hex),
            "directional_endpoint_gate_action": self.directional_endpoint_gate_action.value,
            "directional_endpoint_delivered_action_id": (self.directional_endpoint_delivered_action_id),
            "l2_result": self.l2_result.to_payload(),
            "linf_result": self.linf_result.to_payload(),
            "analytic_boundary_feasible": self.analytic_boundary_feasible,
            "operational_gate_flip_reachable": self.operational_gate_flip_reachable,
            "robust_gate_flip_witness_exists": self.robust_gate_flip_witness_exists,
            "operational_delivered_action_flip_reachable": (self.operational_delivered_action_flip_reachable),
            "robust_delivered_action_flip_witness_exists": (self.robust_delivered_action_flip_witness_exists),
            "logit_sign_action_consistent": self.logit_sign_action_consistent,
            "floating_threshold_band_detected": self.floating_threshold_band_detected,
            "reason_codes": list(self.reason_codes),
            "outcome_read_count": 0,
            "prediction_error_read_count": 0,
            "credit_read_count": 0,
            "evaluation_read_count": 0,
            "gate_update_count": 0,
            "policy_apply_count": 0,
            "model_forward_count": 0,
            "cuda_call_count": 0,
            "credit_achievability_established": False,
            "effect_authorized": False,
            "learnable_authorized": False,
            "steerable_authorized": False,
            "campaign_execution_authorized": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {"receipt_id": self.receipt_id, **self._core_payload()}


@dataclass(frozen=True)
class _DirectionalBoxProblem:
    direction: tuple[float, ...]
    lower_delta: tuple[float, ...]
    upper_delta: tuple[float, ...]

    @property
    def directional_capacity(self) -> tuple[float, ...]:
        return tuple(
            upper if coefficient > 0.0 else -lower if coefficient < 0.0 else 0.0
            for coefficient, lower, upper in zip(
                self.direction,
                self.lower_delta,
                self.upper_delta,
                strict=True,
            )
        )


def _close_algebraic_progress_with_endpoint(
    problem: _DirectionalBoxProblem,
    candidate: tuple[float, ...],
    required_progress: float,
) -> tuple[float, ...] | None:
    if (
        math.fsum(coefficient * change for coefficient, change in zip(problem.direction, candidate, strict=True))
        >= required_progress
    ):
        return candidate
    endpoint = tuple(
        math.copysign(capacity, coefficient) if capacity != 0.0 else 0.0
        for coefficient, capacity in zip(
            problem.direction,
            problem.directional_capacity,
            strict=True,
        )
    )
    if (
        math.fsum(coefficient * change for coefficient, change in zip(problem.direction, endpoint, strict=True))
        < required_progress
    ):
        return None
    return endpoint


def _solve_l2(
    problem: _DirectionalBoxProblem,
    required_progress: float,
) -> tuple[float, ...] | None:
    if required_progress <= 0.0:
        return (0.0,) * _FEATURE_COUNT
    raw_magnitudes = tuple(abs(value) for value in problem.direction)
    capacities = problem.directional_capacity
    movable = tuple(
        index
        for index, (magnitude, capacity) in enumerate(zip(raw_magnitudes, capacities, strict=True))
        if magnitude > 0.0 and capacity > 0.0
    )
    scale = max((raw_magnitudes[index] for index in movable), default=0.0)
    if scale == 0.0:
        return None
    scaled_required = required_progress / scale
    if not math.isfinite(scaled_required):
        return _close_algebraic_progress_with_endpoint(
            problem,
            (0.0,) * _FEATURE_COUNT,
            required_progress,
        )
    magnitudes = tuple(value / scale if index in movable else 0.0 for index, value in enumerate(raw_magnitudes))
    active = tuple(index for index in movable if magnitudes[index] > 0.0)
    if not active or (
        math.fsum(magnitude * capacity for magnitude, capacity in zip(magnitudes, capacities, strict=True))
        < scaled_required
    ):
        return _close_algebraic_progress_with_endpoint(
            problem,
            (0.0,) * _FEATURE_COUNT,
            required_progress,
        )
    breakpoints = sorted({capacities[index] / magnitudes[index] for index in active})
    saturated: set[int] = set()
    multiplier: float | None = None
    for breakpoint in breakpoints:
        fixed = math.fsum(magnitudes[index] * capacities[index] for index in saturated)
        denominator = math.fsum(magnitudes[index] ** 2 for index in active if index not in saturated)
        if denominator == 0.0:
            return _close_algebraic_progress_with_endpoint(
                problem,
                (0.0,) * _FEATURE_COUNT,
                required_progress,
            )
        candidate = (scaled_required - fixed) / denominator
        if not math.isfinite(candidate):
            return _close_algebraic_progress_with_endpoint(
                problem,
                (0.0,) * _FEATURE_COUNT,
                required_progress,
            )
        if candidate <= breakpoint:
            multiplier = max(0.0, candidate)
            break
        saturated.update(
            index for index in active if index not in saturated and capacities[index] / magnitudes[index] == breakpoint
        )
    if multiplier is None:
        multiplier = max(breakpoints, default=0.0)
    if not math.isfinite(multiplier):
        return _close_algebraic_progress_with_endpoint(
            problem,
            (0.0,) * _FEATURE_COUNT,
            required_progress,
        )
    directed = tuple(
        min(multiplier * magnitude, capacity) for magnitude, capacity in zip(magnitudes, capacities, strict=True)
    )
    delta = tuple(
        math.copysign(value, coefficient) if value != 0.0 else 0.0
        for value, coefficient in zip(directed, problem.direction, strict=True)
    )
    return _close_algebraic_progress_with_endpoint(
        problem,
        tuple(0.0 if value == 0.0 else value for value in delta),
        required_progress,
    )


def _solve_linf(
    problem: _DirectionalBoxProblem,
    required_progress: float,
) -> tuple[float, ...] | None:
    if required_progress <= 0.0:
        return (0.0,) * _FEATURE_COUNT
    raw_magnitudes = tuple(abs(value) for value in problem.direction)
    capacities = problem.directional_capacity
    movable = tuple(
        index
        for index, (magnitude, capacity) in enumerate(zip(raw_magnitudes, capacities, strict=True))
        if magnitude > 0.0 and capacity > 0.0
    )
    scale = max((raw_magnitudes[index] for index in movable), default=0.0)
    if scale == 0.0:
        return None
    scaled_required = required_progress / scale
    if not math.isfinite(scaled_required):
        return _close_algebraic_progress_with_endpoint(
            problem,
            (0.0,) * _FEATURE_COUNT,
            required_progress,
        )
    magnitudes = tuple(value / scale if index in movable else 0.0 for index, value in enumerate(raw_magnitudes))
    active = tuple(index for index in movable if magnitudes[index] > 0.0)
    if not active or (
        math.fsum(magnitude * capacity for magnitude, capacity in zip(magnitudes, capacities, strict=True))
        < scaled_required
    ):
        return _close_algebraic_progress_with_endpoint(
            problem,
            (0.0,) * _FEATURE_COUNT,
            required_progress,
        )
    breakpoints = sorted({capacities[index] for index in active})
    saturated: set[int] = set()
    radius: float | None = None
    for breakpoint in breakpoints:
        fixed = math.fsum(magnitudes[index] * capacities[index] for index in saturated)
        denominator = math.fsum(magnitudes[index] for index in active if index not in saturated)
        if denominator == 0.0:
            return _close_algebraic_progress_with_endpoint(
                problem,
                (0.0,) * _FEATURE_COUNT,
                required_progress,
            )
        candidate = (scaled_required - fixed) / denominator
        if not math.isfinite(candidate):
            return _close_algebraic_progress_with_endpoint(
                problem,
                (0.0,) * _FEATURE_COUNT,
                required_progress,
            )
        if candidate <= breakpoint:
            radius = max(0.0, candidate)
            break
        saturated.update(index for index in active if index not in saturated and capacities[index] == breakpoint)
    if radius is None:
        radius = max(breakpoints, default=0.0)
    if not math.isfinite(radius):
        return _close_algebraic_progress_with_endpoint(
            problem,
            (0.0,) * _FEATURE_COUNT,
            required_progress,
        )
    directed = tuple(min(radius, capacity) for capacity in capacities)
    delta = tuple(
        math.copysign(value, coefficient) if value != 0.0 else 0.0
        for value, coefficient in zip(directed, problem.direction, strict=True)
    )
    return _close_algebraic_progress_with_endpoint(
        problem,
        tuple(0.0 if value == 0.0 else value for value in delta),
        required_progress,
    )


def _distance(norm_id: str, delta: tuple[float, ...]) -> float:
    if norm_id == "l2":
        return math.sqrt(math.fsum(value * value for value in delta))
    return max((abs(value) for value in delta), default=0.0)


def _cap_hits(
    delta: tuple[float, ...],
    lower: tuple[float, ...],
    upper: tuple[float, ...],
) -> tuple[int, ...]:
    return tuple(
        index
        for index, (value, low, high) in enumerate(zip(delta, lower, upper, strict=True))
        if value != 0.0 and (value == low or value == high)
    )


def _representable_weight_bounds(
    *,
    weights: tuple[float, ...],
    absolute_cap: float,
    delta_caps: tuple[float, ...],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    lower_weights: list[float] = []
    upper_weights: list[float] = []
    lower_delta: list[float] = []
    upper_delta: list[float] = []
    for index, (weight, delta_cap) in enumerate(zip(weights, delta_caps, strict=True)):
        low_weight = max(-absolute_cap, weight - delta_cap)
        high_weight = min(absolute_cap, weight + delta_cap)
        for _ in range(4):
            actual_low = low_weight - weight
            if low_weight >= -absolute_cap and actual_low >= -delta_cap and low_weight <= weight:
                break
            low_weight = math.nextafter(low_weight, weight)
        else:
            raise RuntimeError(f"cannot close lower dose bound at coordinate {index}")
        for _ in range(4):
            actual_high = high_weight - weight
            if high_weight <= absolute_cap and actual_high <= delta_cap and high_weight >= weight:
                break
            high_weight = math.nextafter(high_weight, weight)
        else:
            raise RuntimeError(f"cannot close upper dose bound at coordinate {index}")
        actual_low = low_weight - weight
        actual_high = high_weight - weight
        if actual_low > 0.0 or actual_high < 0.0 or actual_low > actual_high:
            raise RuntimeError(f"dose bounds exclude baseline coordinate {index}")
        lower_weights.append(0.0 if low_weight == 0.0 else low_weight)
        upper_weights.append(0.0 if high_weight == 0.0 else high_weight)
        lower_delta.append(0.0 if actual_low == 0.0 else actual_low)
        upper_delta.append(0.0 if actual_high == 0.0 else actual_high)
    return tuple(lower_weights), tuple(upper_weights), tuple(lower_delta), tuple(upper_delta)


def _project_representable_weights(
    *,
    baseline_weights: tuple[float, ...],
    proposed_delta: tuple[float, ...],
    lower_weights: tuple[float, ...],
    upper_weights: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple(
        min(high, max(low, baseline + delta))
        for baseline, delta, low, high in zip(
            baseline_weights,
            proposed_delta,
            lower_weights,
            upper_weights,
            strict=True,
        )
    )


def _robust_response_passes(
    *,
    response: _RelationshipActionGateV2CounterfactualResponse,
    target_gate_action: legacy.RelationshipGateAction,
    robust_clearance: float,
) -> bool:
    signed_logit = response.logit if target_gate_action is legacy.RelationshipGateAction.STEER else -response.logit
    return response.gate_action is target_gate_action and signed_logit >= robust_clearance


def _repair_representable_robust_witness(
    *,
    candidate_weights: tuple[float, ...],
    endpoint_weights: tuple[float, ...],
    forecast: PreferenceActionForecast,
    target_gate_action: legacy.RelationshipGateAction,
    robust_clearance: float,
) -> tuple[tuple[float, ...], _RelationshipActionGateV2CounterfactualResponse] | None:
    candidate_response = _relationship_action_gate_v2_counterfactual_response(
        weights=candidate_weights,
        forecast=forecast,
    )
    if _robust_response_passes(
        response=candidate_response,
        target_gate_action=target_gate_action,
        robust_clearance=robust_clearance,
    ):
        return candidate_weights, candidate_response
    endpoint_response = _relationship_action_gate_v2_counterfactual_response(
        weights=endpoint_weights,
        forecast=forecast,
    )
    if not _robust_response_passes(
        response=endpoint_response,
        target_gate_action=target_gate_action,
        robust_clearance=robust_clearance,
    ):
        return None
    failing = candidate_weights
    passing = endpoint_weights
    passing_response = endpoint_response
    for _ in range(256):
        midpoint = tuple(low + ((high - low) * 0.5) for low, high in zip(failing, passing, strict=True))
        if midpoint in {failing, passing}:
            break
        response = _relationship_action_gate_v2_counterfactual_response(
            weights=midpoint,
            forecast=forecast,
        )
        if _robust_response_passes(
            response=response,
            target_gate_action=target_gate_action,
            robust_clearance=robust_clearance,
        ):
            passing = midpoint
            passing_response = response
        else:
            failing = midpoint
    return passing, passing_response


def _norm_result(
    *,
    norm_id: str,
    problem: _DirectionalBoxProblem,
    weights: tuple[float, ...],
    forecast: PreferenceActionForecast,
    lower_weights: tuple[float, ...],
    upper_weights: tuple[float, ...],
    endpoint_weights: tuple[float, ...],
    boundary_progress: float,
    robust_progress: float,
    boundary_distance_kind: str,
    target_gate_action: legacy.RelationshipGateAction,
    robust_clearance: float,
) -> RelationshipActionGateV2GeometricNormResult:
    solver = _solve_l2 if norm_id == "l2" else _solve_linf
    solver_id = (
        RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_L2_SOLVER_ID
        if norm_id == "l2"
        else RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_LINF_SOLVER_ID
    )
    boundary_delta = solver(problem, boundary_progress)
    robust_delta = solver(problem, robust_progress)
    robust_witness = None
    actual_delta = None
    candidate_weights = (
        endpoint_weights
        if robust_delta is None
        else _project_representable_weights(
            baseline_weights=weights,
            proposed_delta=robust_delta,
            lower_weights=lower_weights,
            upper_weights=upper_weights,
        )
    )
    robust_witness = _repair_representable_robust_witness(
        candidate_weights=candidate_weights,
        endpoint_weights=endpoint_weights,
        forecast=forecast,
        target_gate_action=target_gate_action,
        robust_clearance=robust_clearance,
    )
    if robust_witness is not None:
        actual_delta = tuple(witness - baseline for witness, baseline in zip(robust_witness[0], weights, strict=True))
        if any(
            not low <= change <= high
            for change, low, high in zip(
                actual_delta,
                problem.lower_delta,
                problem.upper_delta,
                strict=True,
            )
        ):
            raise RuntimeError("representable robust witness exceeds frozen dose")
    response = None if robust_witness is None else robust_witness[1]
    return RelationshipActionGateV2GeometricNormResult(
        norm_id=norm_id,
        solver_id=solver_id,
        boundary_distance_kind=boundary_distance_kind,
        continuous_boundary_feasible=boundary_delta is not None,
        continuous_boundary_distance_hex=(
            None if boundary_delta is None else _float_hex(_distance(norm_id, boundary_delta))
        ),
        continuous_boundary_delta_hex=(
            None if boundary_delta is None else tuple(_float_hex(value) for value in boundary_delta)
        ),
        representable_robust_witness_exists=robust_witness is not None,
        robust_witness_distance_upper_bound_hex=(
            None if actual_delta is None else _float_hex(_distance(norm_id, actual_delta))
        ),
        robust_witness_actual_delta_hex=(
            None if actual_delta is None else tuple(_float_hex(value) for value in actual_delta)
        ),
        robust_witness_weights_hex=(
            None if robust_witness is None else tuple(_float_hex(value) for value in robust_witness[0])
        ),
        robust_witness_logit_hex=(None if response is None else _float_hex(response.logit)),
        robust_witness_probability_hex=(None if response is None else _float_hex(response.probability)),
        robust_witness_gate_action=None if response is None else response.gate_action,
        robust_witness_delivered_action_id=(None if response is None else response.selected_action_id),
        robust_witness_cap_hit_coordinate_indices=(
            () if actual_delta is None else _cap_hits(actual_delta, problem.lower_delta, problem.upper_delta)
        ),
    )


def analyze_relationship_action_gate_v2_geometric_reachability(
    *,
    frozen_policy: RelationshipActionGateV2FrozenPolicy,
    forecast: PreferenceActionForecast,
    dose: RelationshipActionGateV2GeometricDose,
) -> RelationshipActionGateV2GeometricReachabilityReceipt:
    """Compute one per-forecast geometry receipt without reading outcomes."""

    if type(frozen_policy) is not RelationshipActionGateV2FrozenPolicy:
        raise TypeError("frozen_policy must be RelationshipActionGateV2FrozenPolicy")
    if type(forecast) is not PreferenceActionForecast:
        raise TypeError("forecast must be an exact PreferenceActionForecast")
    if type(dose) is not RelationshipActionGateV2GeometricDose:
        raise TypeError("dose must be RelationshipActionGateV2GeometricDose")
    policy_replay = frozen_policy.decide(forecast)
    weights = frozen_policy.checkpoint.weights
    baseline = _relationship_action_gate_v2_counterfactual_response(
        weights=weights,
        forecast=forecast,
    )
    if (
        policy_replay.decision.features != baseline.features
        or policy_replay.decision.steer_probability != baseline.probability
        or policy_replay.decision.gate_action is not baseline.gate_action
        or policy_replay.decision.selected_action_id != baseline.selected_action_id
    ):
        raise ValueError("geometric baseline differs from exact frozen policy replay")
    if dose.absolute_parameter_cap_hex != frozen_policy.artifact.max_abs_parameter_hex:
        raise ValueError("geometric absolute parameter cap differs from frozen artifact")

    lower_weights, upper_weights, lower, upper = _representable_weight_bounds(
        weights=weights,
        absolute_cap=dose.absolute_parameter_cap,
        delta_caps=dose.coordinate_delta_cap,
    )
    target = (
        legacy.RelationshipGateAction.NOOP
        if baseline.gate_action is legacy.RelationshipGateAction.STEER
        else legacy.RelationshipGateAction.STEER
    )
    sign = 1.0 if target is legacy.RelationshipGateAction.STEER else -1.0
    problem = _DirectionalBoxProblem(
        direction=tuple(sign * feature for feature in baseline.features),
        lower_delta=lower,
        upper_delta=upper,
    )
    endpoint_weights = tuple(
        high if coefficient > 0.0 else low if coefficient < 0.0 else baseline_weight
        for coefficient, low, high, baseline_weight in zip(
            problem.direction,
            lower_weights,
            upper_weights,
            weights,
            strict=True,
        )
    )
    endpoint_delta = tuple(
        endpoint - baseline_weight for endpoint, baseline_weight in zip(endpoint_weights, weights, strict=True)
    )
    endpoint = _relationship_action_gate_v2_counterfactual_response(
        weights=endpoint_weights,
        forecast=forecast,
    )
    signed_baseline = sign * baseline.logit
    boundary_progress = max(0.0, -signed_baseline)
    robust_progress = max(0.0, dose.robust_logit_clearance - signed_baseline)
    boundary_kind = "infimum" if target is legacy.RelationshipGateAction.STEER else "minimum"
    common = {
        "problem": problem,
        "weights": weights,
        "forecast": forecast,
        "lower_weights": lower_weights,
        "upper_weights": upper_weights,
        "endpoint_weights": endpoint_weights,
        "boundary_progress": boundary_progress,
        "robust_progress": robust_progress,
        "boundary_distance_kind": boundary_kind,
        "target_gate_action": target,
        "robust_clearance": dose.robust_logit_clearance,
    }
    l2_result = _norm_result(norm_id="l2", **common)
    linf_result = _norm_result(norm_id="linf", **common)
    analytic_boundary = l2_result.continuous_boundary_feasible and linf_result.continuous_boundary_feasible
    robust_witness = l2_result.representable_robust_witness_exists and linf_result.representable_robust_witness_exists
    operational = endpoint.gate_action is target
    target_delivered = (
        forecast.recommended_action_id
        if target is legacy.RelationshipGateAction.STEER
        else RelationshipAction.NEUTRAL_NOOP.value
    )
    sign_consistent = (baseline.logit > 0.0) is (baseline.gate_action is legacy.RelationshipGateAction.STEER)
    reasons = [
        "scope:per-forecast-individual-witness",
        "inputs:frozen-policy-and-typed-owner-forecast-only",
        "learning-signal:none",
        ("analytic-boundary:box-feasible" if analytic_boundary else "analytic-boundary:box-infeasible"),
        (
            "operational-gate:endpoint-exact-replay-pass"
            if operational
            else "operational-gate:endpoint-exact-replay-fail"
        ),
        (
            "robust-witness:exact-gate-replay-pass"
            if robust_witness
            else "robust-witness:box-or-exact-replay-infeasible"
        ),
    ]
    if forecast.recommended_action_id == RelationshipAction.NEUTRAL_NOOP.value:
        reasons.append("delivered-action:recommended-noop-no-treatment-divergence")
    if not sign_consistent:
        reasons.append("floating-threshold-band:detected")
    return RelationshipActionGateV2GeometricReachabilityReceipt(
        frozen_policy_id=frozen_policy.policy_id,
        artifact_id=frozen_policy.artifact.artifact_id,
        checkpoint_content_sha256=frozen_policy.checkpoint.content_sha256,
        forecast_sha256=legacy._canonical_sha256(preference_action_forecast_to_payload(forecast)),
        forecast_id=forecast.forecast_id,
        decision_id=forecast.decision_id,
        dose=dose,
        baseline_weights_hex=tuple(_float_hex(value) for value in weights),
        baseline_features_hex=tuple(_float_hex(value) for value in baseline.features),
        baseline_logit_hex=_float_hex(baseline.logit),
        baseline_probability_hex=_float_hex(baseline.probability),
        baseline_gate_action=baseline.gate_action,
        baseline_delivered_action_id=baseline.selected_action_id,
        recommended_action_id=forecast.recommended_action_id,
        target_gate_action=target,
        target_delivered_action_id=target_delivered,
        effective_lower_delta_hex=tuple(_float_hex(value) for value in lower),
        effective_upper_delta_hex=tuple(_float_hex(value) for value in upper),
        directional_endpoint_delta_hex=tuple(_float_hex(value) for value in endpoint_delta),
        directional_endpoint_weights_hex=tuple(_float_hex(value) for value in endpoint_weights),
        directional_endpoint_logit_hex=_float_hex(endpoint.logit),
        directional_endpoint_probability_hex=_float_hex(endpoint.probability),
        directional_endpoint_gate_action=endpoint.gate_action,
        directional_endpoint_delivered_action_id=endpoint.selected_action_id,
        l2_result=l2_result,
        linf_result=linf_result,
        analytic_boundary_feasible=analytic_boundary,
        operational_gate_flip_reachable=operational,
        robust_gate_flip_witness_exists=robust_witness,
        operational_delivered_action_flip_reachable=(
            operational and baseline.selected_action_id != endpoint.selected_action_id
        ),
        robust_delivered_action_flip_witness_exists=(
            robust_witness and baseline.selected_action_id != target_delivered
        ),
        logit_sign_action_consistent=sign_consistent,
        floating_threshold_band_detected=not sign_consistent,
        reason_codes=tuple(reasons),
    )


def replay_relationship_action_gate_v2_geometric_reachability_receipt(
    *,
    payload: object,
    frozen_policy: RelationshipActionGateV2FrozenPolicy,
    forecast: PreferenceActionForecast,
) -> RelationshipActionGateV2GeometricReachabilityReceipt:
    """Accept a receipt only after recomputing the complete owner result."""

    if type(payload) is not dict:
        raise TypeError("geometric receipt payload must be an exact mapping")
    dose = RelationshipActionGateV2GeometricDose.from_payload(payload.get("dose"))
    replayed = analyze_relationship_action_gate_v2_geometric_reachability(
        frozen_policy=frozen_policy,
        forecast=forecast,
        dose=dose,
    )
    if payload != replayed.to_payload():
        raise ValueError("relationship action gate v2 geometric receipt replay drifted")
    return replayed


__all__ = [
    "RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_CONTINUOUS_SOLUTION_SEMANTICS",
    "RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_DOSE_SCHEMA_VERSION",
    "RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_L2_SOLVER_ID",
    "RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_LINF_SOLVER_ID",
    "RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_NORM_RESULT_SCHEMA_VERSION",
    "RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_OPERATOR_ID",
    "RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_RECEIPT_SCHEMA_VERSION",
    "RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_ROBUST_WITNESS_SEMANTICS",
    "RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_WITNESS_SCOPE",
    "RelationshipActionGateV2GeometricDose",
    "RelationshipActionGateV2GeometricNormResult",
    "RelationshipActionGateV2GeometricReachabilityReceipt",
    "analyze_relationship_action_gate_v2_geometric_reachability",
    "replay_relationship_action_gate_v2_geometric_reachability_receipt",
]
