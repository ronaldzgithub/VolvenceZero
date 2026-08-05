"""Immutable contracts for the bounded residual-steering owner chain.

The three runtime slots intentionally exchange only frozen values.  Learned
parameter artifacts are injected into their unique owners at construction and
never inferred from evaluation output or reconstructed by consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
import struct

from volvence_zero.canonical_json import (
    canonical_json_bytes,
    strict_json_loads,
    typed_from_json,
    typed_to_json,
)


STEERING_CONDITION_BELIEF_SLOT = "steering_condition_belief"
STEERING_GATE_DECISION_SLOT = "steering_gate_decision"
STEERING_INTERVENTION_SLOT = "steering_intervention"

STEERING_READER_ARTIFACT_SCHEMA_VERSION = "steering-reader-artifact.v1"
STEERING_EXECUTOR_ARTIFACT_SCHEMA_VERSION = "steering-executor-artifact.v1"
STEERING_GATE_ARTIFACT_SCHEMA_VERSION = "steering-gate-artifact.v1"
STEERING_ARTIFACT_BUNDLE_SCHEMA_VERSION = "steering-artifact-bundle.v1"
STEERING_GATE_CHECKPOINT_SCHEMA_VERSION = "steering-gate-checkpoint.v1"


class SteeringGateAction(str, Enum):
    NOOP = "noop"
    STEER = "steer"


def _require_nonempty(value: str, *, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _require_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_finite(values: tuple[float, ...], *, field_name: str) -> None:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{field_name} must contain finite values")


def _vector_sha256(values: tuple[float, ...]) -> str:
    return hashlib.sha256(struct.pack(f"!{len(values)}d", *values)).hexdigest()


def _require_matrix(
    values: tuple[tuple[float, ...], ...],
    *,
    rows: int,
    columns: int,
    field_name: str,
) -> None:
    if len(values) != rows or any(len(row) != columns for row in values):
        raise ValueError(
            f"{field_name} must have shape ({rows}, {columns}), got "
            f"({len(values)}, {tuple(len(row) for row in values)})"
        )
    if not all(math.isfinite(value) for row in values for value in row):
        raise ValueError(f"{field_name} must contain finite values")


@dataclass(frozen=True)
class SteeringReaderArtifact:
    schema_version: str
    artifact_id: str
    model_id: str
    model_weights_sha256: str
    source_preregistration_sha256: str
    layer_index: int
    residual_width: int
    class_labels: tuple[str, ...]
    weights: tuple[tuple[float, ...], ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    ridge_lambda: float
    description: str

    def __post_init__(self) -> None:
        if self.schema_version != STEERING_READER_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported SteeringReaderArtifact schema_version")
        for field_name, value in (
            ("artifact_id", self.artifact_id),
            ("model_id", self.model_id),
            ("description", self.description),
        ):
            _require_nonempty(value, field_name=field_name)
        _require_sha256(
            self.model_weights_sha256,
            field_name="model_weights_sha256",
        )
        _require_sha256(
            self.source_preregistration_sha256,
            field_name="source_preregistration_sha256",
        )
        if self.layer_index < 0 or self.residual_width < 1:
            raise ValueError("reader layer_index/residual_width are invalid")
        if len(self.class_labels) < 2 or any(not label.strip() for label in self.class_labels):
            raise ValueError("reader requires at least two non-empty class labels")
        if len(set(self.class_labels)) != len(self.class_labels):
            raise ValueError("reader class_labels must be unique")
        _require_matrix(
            self.weights,
            rows=self.residual_width,
            columns=len(self.class_labels),
            field_name="weights",
        )
        _require_finite(self.feature_mean, field_name="feature_mean")
        _require_finite(self.feature_scale, field_name="feature_scale")
        if (
            len(self.feature_mean) != self.residual_width
            or len(self.feature_scale) != self.residual_width
            or any(value <= 0.0 for value in self.feature_scale)
        ):
            raise ValueError("reader normalization vectors do not match residual_width")
        if not math.isfinite(self.ridge_lambda) or self.ridge_lambda <= 0.0:
            raise ValueError("ridge_lambda must be finite and positive")


@dataclass(frozen=True)
class SteeringExecutorArtifact:
    schema_version: str
    artifact_id: str
    model_id: str
    model_weights_sha256: str
    source_preregistration_sha256: str
    reader_artifact_id: str
    layer_index: int
    residual_width: int
    rank: int
    class_labels: tuple[str, ...]
    u_factors: tuple[tuple[float, ...], ...]
    v_factors: tuple[tuple[float, ...], ...]
    condition_codes: tuple[tuple[float, ...], ...]
    control_norm_cap_ratio: float
    free_bias_present: bool
    zero_code_strict_noop: bool
    description: str

    def __post_init__(self) -> None:
        if self.schema_version != STEERING_EXECUTOR_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported SteeringExecutorArtifact schema_version")
        for field_name, value in (
            ("artifact_id", self.artifact_id),
            ("model_id", self.model_id),
            ("reader_artifact_id", self.reader_artifact_id),
            ("description", self.description),
        ):
            _require_nonempty(value, field_name=field_name)
        _require_sha256(
            self.model_weights_sha256,
            field_name="model_weights_sha256",
        )
        _require_sha256(
            self.source_preregistration_sha256,
            field_name="source_preregistration_sha256",
        )
        if self.layer_index < 0 or self.residual_width < 1 or self.rank < 1:
            raise ValueError("executor layer_index/residual_width/rank are invalid")
        if len(self.class_labels) < 2 or len(set(self.class_labels)) != len(
            self.class_labels
        ):
            raise ValueError("executor class_labels must be unique and non-trivial")
        _require_matrix(
            self.u_factors,
            rows=self.residual_width,
            columns=self.rank,
            field_name="u_factors",
        )
        _require_matrix(
            self.v_factors,
            rows=self.residual_width,
            columns=self.rank,
            field_name="v_factors",
        )
        _require_matrix(
            self.condition_codes,
            rows=len(self.class_labels),
            columns=self.rank,
            field_name="condition_codes",
        )
        if (
            not math.isfinite(self.control_norm_cap_ratio)
            or not 0.0 < self.control_norm_cap_ratio <= 1.0
        ):
            raise ValueError("control_norm_cap_ratio must be within (0, 1]")
        if self.free_bias_present:
            raise ValueError("steering executor artifacts must not contain free bias")
        if not self.zero_code_strict_noop:
            raise ValueError("steering executor artifacts require strict zero-code no-op")


@dataclass(frozen=True)
class SteeringGateArtifact:
    schema_version: str
    artifact_id: str
    source_preregistration_sha256: str
    feature_names: tuple[str, ...]
    weights: tuple[tuple[float, float], ...]
    bias: tuple[float, float]
    policy_version: int
    description: str

    def __post_init__(self) -> None:
        if self.schema_version != STEERING_GATE_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported SteeringGateArtifact schema_version")
        _require_nonempty(self.artifact_id, field_name="artifact_id")
        _require_nonempty(self.description, field_name="description")
        _require_sha256(
            self.source_preregistration_sha256,
            field_name="source_preregistration_sha256",
        )
        if not self.feature_names or any(not name.strip() for name in self.feature_names):
            raise ValueError("gate feature_names must be non-empty")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("gate feature_names must be unique")
        _require_matrix(
            self.weights,
            rows=len(self.feature_names),
            columns=2,
            field_name="weights",
        )
        if len(self.bias) != 2 or not all(math.isfinite(value) for value in self.bias):
            raise ValueError("gate bias must contain two finite values")
        if self.policy_version < 1:
            raise ValueError("gate policy_version must be positive")


@dataclass(frozen=True)
class SteeringArtifactBundle:
    schema_version: str
    bundle_id: str
    reader: SteeringReaderArtifact
    executor: SteeringExecutorArtifact
    gate: SteeringGateArtifact
    description: str
    sensor_off_executor: SteeringExecutorArtifact | None = None

    def __post_init__(self) -> None:
        if self.schema_version != STEERING_ARTIFACT_BUNDLE_SCHEMA_VERSION:
            raise ValueError("unsupported SteeringArtifactBundle schema_version")
        _require_nonempty(self.bundle_id, field_name="bundle_id")
        _require_nonempty(self.description, field_name="description")
        if self.executor.reader_artifact_id != self.reader.artifact_id:
            raise ValueError("executor is not bound to the bundled reader")
        if self.executor.model_id != self.reader.model_id:
            raise ValueError("reader/executor model_id mismatch")
        if (
            self.executor.model_weights_sha256
            != self.reader.model_weights_sha256
        ):
            raise ValueError("reader/executor model weights lineage mismatch")
        if self.executor.layer_index != self.reader.layer_index:
            raise ValueError("reader/executor layer mismatch")
        if self.executor.residual_width != self.reader.residual_width:
            raise ValueError("reader/executor residual width mismatch")
        if self.executor.class_labels != self.reader.class_labels:
            raise ValueError("reader/executor class label mismatch")
        if self.sensor_off_executor is not None:
            control = self.sensor_off_executor
            if (
                control.model_id != self.executor.model_id
                or control.model_weights_sha256
                != self.executor.model_weights_sha256
                or control.reader_artifact_id != self.reader.artifact_id
                or control.layer_index != self.executor.layer_index
                or control.residual_width != self.executor.residual_width
                or control.rank != self.executor.rank
                or control.class_labels != self.executor.class_labels
            ):
                raise ValueError("sensor-off executor lineage/geometry mismatch")
            if len(set(control.condition_codes)) != 1:
                raise ValueError(
                    "sensor-off executor must repeat one unconditional code"
                )

    def to_json(self) -> str:
        payload = typed_to_json(self, SteeringArtifactBundle)
        return canonical_json_bytes(payload).decode("utf-8")

    @classmethod
    def from_json(cls, payload: str) -> "SteeringArtifactBundle":
        if not isinstance(payload, str):
            raise TypeError("steering artifact bundle JSON must be text")
        raw = strict_json_loads(
            payload.encode("utf-8"),
            max_bytes=64 * 1024 * 1024,
        )
        decoded = typed_from_json(raw, SteeringArtifactBundle)
        if not isinstance(decoded, SteeringArtifactBundle):
            raise TypeError("decoded steering artifact bundle has wrong type")
        return decoded


@dataclass(frozen=True)
class SteeringConditionBelief:
    belief_label: str
    belief_index: int
    belief_margin: float
    fresh_belief_label: str
    fresh_belief_index: int
    fresh_margin: float
    belief_disagrees_fresh: bool
    staleness_proxy: float
    base_action_entropy: float
    reader_artifact_id: str
    source_model_id: str
    source_layer_index: int
    source_residual_norm: float
    description: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("belief_label", self.belief_label),
            ("fresh_belief_label", self.fresh_belief_label),
            ("reader_artifact_id", self.reader_artifact_id),
            ("source_model_id", self.source_model_id),
            ("description", self.description),
        ):
            _require_nonempty(value, field_name=field_name)
        if self.belief_index < 0 or self.fresh_belief_index < 0:
            raise ValueError("belief indices must be non-negative")
        for field_name, value in (
            ("belief_margin", self.belief_margin),
            ("fresh_margin", self.fresh_margin),
            ("staleness_proxy", self.staleness_proxy),
            ("base_action_entropy", self.base_action_entropy),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be within [0, 1]")
        if self.source_layer_index < 0:
            raise ValueError("source_layer_index must be non-negative")
        if not math.isfinite(self.source_residual_norm) or self.source_residual_norm <= 0:
            raise ValueError("source_residual_norm must be finite and positive")


@dataclass(frozen=True)
class SteeringGateDecision:
    decision_id: str
    action: SteeringGateAction
    steer_probability: float
    observations: tuple[tuple[str, float], ...]
    policy_artifact_id: str
    policy_version: int
    terminal_credit_pending: bool
    decision_mode: str
    description: str

    def __post_init__(self) -> None:
        _require_nonempty(self.decision_id, field_name="decision_id")
        if not isinstance(self.action, SteeringGateAction):
            raise ValueError("action must be SteeringGateAction")
        if not math.isfinite(self.steer_probability) or not 0.0 <= self.steer_probability <= 1.0:
            raise ValueError("steer_probability must be within [0, 1]")
        if not self.observations or any(
            not name.strip() or not math.isfinite(value)
            for name, value in self.observations
        ):
            raise ValueError("gate observations must be named finite values")
        _require_nonempty(self.policy_artifact_id, field_name="policy_artifact_id")
        _require_nonempty(self.decision_mode, field_name="decision_mode")
        _require_nonempty(self.description, field_name="description")
        if self.policy_version < 1:
            raise ValueError("policy_version must be positive")


@dataclass(frozen=True)
class SteeringResidualContext:
    """Substrate-owned normalized context before/after one intervention.

    The value is intentionally text-free.  It gives PE research the exact
    matched context coordinates emitted by the residual executor without
    requiring a consumer to reconstruct a producer's mutable model state.
    """

    source_sha256: str
    layer_indices: tuple[int, ...]
    activation_widths: tuple[int, ...]
    values: tuple[float, ...]
    values_sha256: str
    conditioned: bool
    readout_kind: str = "latest-token-hooked-layer-residual-l2.v1"

    def __post_init__(self) -> None:
        _require_sha256(self.source_sha256, field_name="source_sha256")
        if self.readout_kind != "latest-token-hooked-layer-residual-l2.v1":
            raise ValueError("unsupported steering residual context readout")
        if (
            not self.layer_indices
            or tuple(sorted(self.layer_indices)) != self.layer_indices
            or len(set(self.layer_indices)) != len(self.layer_indices)
        ):
            raise ValueError("steering residual context layers must be unique/sorted")
        if (
            len(self.activation_widths) != len(self.layer_indices)
            or any(width < 1 for width in self.activation_widths)
            or sum(self.activation_widths) != len(self.values)
        ):
            raise ValueError("steering residual context geometry is inconsistent")
        _require_finite(self.values, field_name="values")
        norm = math.sqrt(sum(value * value for value in self.values))
        if not math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("steering residual context must be L2-normalized")
        if self.values_sha256 != _vector_sha256(self.values):
            raise ValueError("steering residual context values SHA-256 mismatch")


@dataclass(frozen=True)
class SteeringTerminalPredictionError:
    """PE-owned terminal comparison against a matched noop counterfactual."""

    episode_id: str
    decision_ids: tuple[str, ...]
    action_batch_id: str
    noop_batch_id: str
    sample_ids: tuple[str, ...]
    prediction_head_fingerprint: str
    target_lineage_fingerprint: str
    target_model_id: str
    target_model_weights_sha256: str
    action_mean_squared_error: float
    noop_mean_squared_error: float
    relative_mse_improvement: float
    action_mean_cosine_similarity: float
    noop_mean_cosine_similarity: float
    cosine_error_improvement: float
    terminal: bool
    description: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("episode_id", self.episode_id),
            ("action_batch_id", self.action_batch_id),
            ("noop_batch_id", self.noop_batch_id),
            ("target_model_id", self.target_model_id),
            ("description", self.description),
        ):
            _require_nonempty(value, field_name=field_name)
        if self.action_batch_id == self.noop_batch_id:
            raise ValueError("action/noop terminal batches must be distinct")
        for field_name, values in (
            ("decision_ids", self.decision_ids),
            ("sample_ids", self.sample_ids),
        ):
            if not values or any(not value.strip() for value in values):
                raise ValueError(f"{field_name} must be non-empty")
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} must be unique")
        for field_name, value in (
            ("prediction_head_fingerprint", self.prediction_head_fingerprint),
            ("target_lineage_fingerprint", self.target_lineage_fingerprint),
            ("target_model_weights_sha256", self.target_model_weights_sha256),
        ):
            _require_sha256(value, field_name=field_name)
        for field_name, value in (
            ("action_mean_squared_error", self.action_mean_squared_error),
            ("noop_mean_squared_error", self.noop_mean_squared_error),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        for field_name, value in (
            ("relative_mse_improvement", self.relative_mse_improvement),
            ("action_mean_cosine_similarity", self.action_mean_cosine_similarity),
            ("noop_mean_cosine_similarity", self.noop_mean_cosine_similarity),
            ("cosine_error_improvement", self.cosine_error_improvement),
        ):
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be within [-1, 1]")
        if not self.terminal:
            raise ValueError("steering terminal prediction error must be terminal")


@dataclass(frozen=True)
class SteeringIntervention:
    action: SteeringGateAction
    source_model_id: str
    source_model_weights_sha256: str
    layer_index: int
    residual_delta: tuple[float, ...]
    residual_norm: float
    control_norm: float
    control_norm_cap: float
    executor_artifact_id: str
    reader_artifact_id: str
    gate_policy_version: int
    zero_code_noop: bool
    application_mode: str
    shadow_hook_executed: bool
    runtime_backend: str
    downstream_effect: tuple[float, ...]
    description: str
    noop_context: SteeringResidualContext | None = None
    action_context: SteeringResidualContext | None = None
    shadow_hook_latency_ms: float = 0.0
    sensor_off_action_context: SteeringResidualContext | None = None
    sensor_off_executor_artifact_id: str = ""
    sensor_off_control_norm: float = 0.0
    sensor_off_shadow_hook_latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.action, SteeringGateAction):
            raise ValueError("action must be SteeringGateAction")
        _require_nonempty(self.source_model_id, field_name="source_model_id")
        _require_sha256(
            self.source_model_weights_sha256,
            field_name="source_model_weights_sha256",
        )
        if self.layer_index < 0:
            raise ValueError("layer_index must be non-negative")
        if not self.residual_delta or not all(
            math.isfinite(value) for value in self.residual_delta
        ):
            raise ValueError("residual_delta must contain finite values")
        for field_name, value in (
            ("residual_norm", self.residual_norm),
            ("control_norm", self.control_norm),
            ("control_norm_cap", self.control_norm_cap),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if self.control_norm > self.control_norm_cap + 1e-8:
            raise ValueError("control_norm exceeds the frozen norm cap")
        if self.action is SteeringGateAction.NOOP:
            if not self.zero_code_noop or self.control_norm > 1e-8:
                raise ValueError("NOOP interventions must be strict zero-code no-ops")
        for field_name, value in (
            ("executor_artifact_id", self.executor_artifact_id),
            ("reader_artifact_id", self.reader_artifact_id),
            ("application_mode", self.application_mode),
            ("runtime_backend", self.runtime_backend),
            ("description", self.description),
        ):
            _require_nonempty(value, field_name=field_name)
        if self.gate_policy_version < 1:
            raise ValueError("gate_policy_version must be positive")
        if not all(math.isfinite(value) for value in self.downstream_effect):
            raise ValueError("downstream_effect must contain finite values")
        if (
            not math.isfinite(self.shadow_hook_latency_ms)
            or self.shadow_hook_latency_ms < 0.0
        ):
            raise ValueError("shadow_hook_latency_ms must be finite/non-negative")
        if not self.shadow_hook_executed and self.shadow_hook_latency_ms != 0.0:
            raise ValueError("non-executed steering hook cannot report latency")
        if (
            not math.isfinite(self.sensor_off_control_norm)
            or self.sensor_off_control_norm < 0.0
            or not math.isfinite(self.sensor_off_shadow_hook_latency_ms)
            or self.sensor_off_shadow_hook_latency_ms < 0.0
        ):
            raise ValueError("sensor-off steering norm/latency is invalid")
        if self.sensor_off_action_context is None:
            if (
                self.sensor_off_executor_artifact_id
                or self.sensor_off_control_norm != 0.0
                or self.sensor_off_shadow_hook_latency_ms != 0.0
            ):
                raise ValueError("partial sensor-off steering evidence")
        else:
            _require_nonempty(
                self.sensor_off_executor_artifact_id,
                field_name="sensor_off_executor_artifact_id",
            )
            if not self.sensor_off_action_context.conditioned:
                raise ValueError("sensor-off action context must be conditioned")
            if self.sensor_off_control_norm > self.control_norm_cap + 1e-8:
                raise ValueError("sensor-off control norm exceeds cap")
        if self.noop_context is not None and self.noop_context.conditioned:
            raise ValueError("noop_context must be unconditioned")
        if self.action_context is not None:
            if self.action is SteeringGateAction.STEER and not self.action_context.conditioned:
                raise ValueError("STEER action_context must be conditioned")
            if self.action is SteeringGateAction.NOOP and self.action_context.conditioned:
                raise ValueError("NOOP action_context must be unconditioned")
        if self.noop_context is not None and self.action_context is not None:
            if (
                self.noop_context.layer_indices != self.action_context.layer_indices
                or self.noop_context.activation_widths
                != self.action_context.activation_widths
            ):
                raise ValueError("matched steering context geometry drift")


@dataclass(frozen=True)
class SteeringGateCheckpoint:
    """Complete owner-local gate state for rollback and exact continuation."""

    schema_version: str
    checkpoint_id: str
    artifact: SteeringGateArtifact
    learning_rate: float
    learning_enabled: bool
    decision_mode: str
    exploration_seed: int
    decision_count: int
    pending_decisions: tuple[SteeringGateDecision, ...]
    consumed_credit_record_ids: tuple[str, ...]
    description: str

    def __post_init__(self) -> None:
        if self.schema_version != STEERING_GATE_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported SteeringGateCheckpoint schema_version")
        _require_nonempty(self.checkpoint_id, field_name="checkpoint_id")
        _require_nonempty(self.description, field_name="description")
        if not math.isfinite(self.learning_rate) or not 0.0 < self.learning_rate <= 0.2:
            raise ValueError("gate checkpoint learning_rate must be within (0, 0.2]")
        if self.decision_mode not in {
            "frozen-policy-argmax",
            "evidence-stochastic",
        }:
            raise ValueError("gate checkpoint decision_mode is unsupported")
        if self.exploration_seed < 0 or self.decision_count < 0:
            raise ValueError("gate checkpoint counters must be non-negative")
        pending_ids = tuple(item.decision_id for item in self.pending_decisions)
        if len(set(pending_ids)) != len(pending_ids):
            raise ValueError("gate checkpoint pending decision ids must be unique")
        if len(set(self.consumed_credit_record_ids)) != len(
            self.consumed_credit_record_ids
        ):
            raise ValueError("gate checkpoint consumed record ids must be unique")

    def to_json(self) -> str:
        payload = typed_to_json(self, SteeringGateCheckpoint)
        return canonical_json_bytes(payload).decode("utf-8")

    @classmethod
    def from_json(cls, payload: str) -> "SteeringGateCheckpoint":
        if not isinstance(payload, str):
            raise TypeError("steering gate checkpoint JSON must be text")
        raw = strict_json_loads(payload.encode("utf-8"), max_bytes=64 * 1024 * 1024)
        decoded = typed_from_json(raw, SteeringGateCheckpoint)
        if not isinstance(decoded, SteeringGateCheckpoint):
            raise TypeError("decoded steering gate checkpoint has wrong type")
        return decoded


__all__ = (
    "STEERING_CONDITION_BELIEF_SLOT",
    "STEERING_ARTIFACT_BUNDLE_SCHEMA_VERSION",
    "STEERING_EXECUTOR_ARTIFACT_SCHEMA_VERSION",
    "STEERING_GATE_ARTIFACT_SCHEMA_VERSION",
    "STEERING_GATE_CHECKPOINT_SCHEMA_VERSION",
    "STEERING_GATE_DECISION_SLOT",
    "STEERING_INTERVENTION_SLOT",
    "STEERING_READER_ARTIFACT_SCHEMA_VERSION",
    "SteeringConditionBelief",
    "SteeringArtifactBundle",
    "SteeringExecutorArtifact",
    "SteeringGateAction",
    "SteeringGateArtifact",
    "SteeringGateDecision",
    "SteeringGateCheckpoint",
    "SteeringIntervention",
    "SteeringReaderArtifact",
    "SteeringResidualContext",
    "SteeringTerminalPredictionError",
)
