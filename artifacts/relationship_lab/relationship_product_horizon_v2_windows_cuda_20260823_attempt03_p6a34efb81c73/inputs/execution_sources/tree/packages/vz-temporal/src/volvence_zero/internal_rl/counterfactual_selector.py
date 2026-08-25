from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math

from volvence_zero.conditioning_bank_contracts import (
    ConditioningBankReadout,
    ConditioningBankType,
)
from volvence_zero.substrate import SubstrateSnapshot
from volvence_zero.temporal.metacontroller_components import (
    residual_sequence_from_snapshot,
    summarize_residual_activations,
)


@dataclass(frozen=True)
class CounterfactualActionExample:
    example_id: str
    group_id: str
    split: str
    state_features: tuple[float, ...]
    candidate_raw_deltas: tuple[float, ...]
    audit_candidate_raw_deltas: tuple[float, ...] = ()


@dataclass(frozen=True)
class CounterfactualActionSelection:
    example_id: str
    group_id: str
    split: str
    prediction_source: str
    selected_action_index: int
    oracle_action_index: int
    predicted_top3_action_indices: tuple[int, ...]
    selected_raw_delta: float
    oracle_raw_delta: float
    oracle_regret: float
    top1_match: bool
    top3_match: bool
    selected_positive: bool
    audit_selected_raw_delta: float | None
    audit_oracle_raw_delta: float | None
    audit_oracle_regret: float | None
    audit_selected_positive: bool | None
    model_fingerprint: str


RELATIONSHIP_CONDITIONED_SELECTOR_FEATURE_SCHEMA_VERSION = (
    "residual-state+relationship-owner-readout.v1"
)


def relationship_conditioned_selector_state_vector(
    substrate_snapshot: SubstrateSnapshot,
    relationship_readout: ConditioningBankReadout,
) -> tuple[float, ...]:
    """Append one admitted Relationship owner readout to residual state.

    The temporal selector treats the ordered owner vector as an opaque,
    bounded condition.  It does not interpret label names or reconstruct
    relationship semantics.  Cold, zero-confidence, or wrong-bank inputs
    fail closed so a purported conditioned lane cannot silently collapse to
    the historical unconditioned v35 selector.
    """

    if relationship_readout.bank_type is not ConditioningBankType.RELATIONSHIP:
        raise ValueError(
            "relationship-conditioned selector requires a RELATIONSHIP bank"
        )
    if relationship_readout.is_cold_start:
        raise ValueError(
            "relationship-conditioned selector cannot consume a cold-start "
            "Relationship readout"
        )
    if relationship_readout.confidence <= 0.0:
        raise ValueError(
            "relationship-conditioned selector requires positive owner "
            "confidence"
        )
    if not relationship_readout.readout:
        raise ValueError(
            "relationship-conditioned selector requires a non-empty owner "
            "readout"
        )
    conditioned = tuple(
        (2.0 * value - 1.0) * relationship_readout.confidence
        for value in relationship_readout.readout
    )
    if any(not -1.0 <= value <= 1.0 for value in conditioned):
        raise RuntimeError(
            "relationship-conditioned selector produced an unbounded owner "
            "condition"
        )
    return residual_action_state_vector(substrate_snapshot) + conditioned


@dataclass(frozen=True)
class ResidualActionSelectorArtifact:
    input_mean: tuple[float, ...]
    input_scale: tuple[float, ...]
    encoder_components: tuple[tuple[float, ...], ...]
    action_value_weights: tuple[tuple[float, ...], ...]
    action_value_bias: tuple[float, ...]
    input_dim: int
    latent_dim: int
    action_count: int
    explained_variance_ratio: float
    ridge_strength: float
    model_fingerprint: str

    def predict_action_values(
        self,
        state_features: tuple[float, ...],
    ) -> tuple[float, ...]:
        if len(state_features) != self.input_dim:
            raise ValueError(
                "counterfactual selector input dimension mismatch: "
                f"expected={self.input_dim}, actual={len(state_features)}"
            )
        normalized = tuple(
            (value - mean) / scale
            for value, mean, scale in zip(
                state_features,
                self.input_mean,
                self.input_scale,
                strict=True,
            )
        )
        latent = tuple(
            sum(
                component[index] * normalized[index]
                for index in range(self.input_dim)
            )
            for component in self.encoder_components
        )
        return tuple(
            bias
            + sum(
                weights[index] * latent[index]
                for index in range(self.latent_dim)
            )
            for weights, bias in zip(
                self.action_value_weights,
                self.action_value_bias,
                strict=True,
            )
        )


@dataclass(frozen=True)
class KernelResidualActionSelectorArtifact:
    input_mean: tuple[float, ...]
    input_scale: tuple[float, ...]
    normalized_training_inputs: tuple[tuple[float, ...], ...]
    dual_action_weights: tuple[tuple[float, ...], ...]
    action_value_bias: tuple[float, ...]
    input_dim: int
    action_count: int
    ridge_strength: float
    model_fingerprint: str

    def predict_action_values(
        self,
        state_features: tuple[float, ...],
    ) -> tuple[float, ...]:
        if len(state_features) != self.input_dim:
            raise ValueError(
                "kernel counterfactual selector input dimension mismatch: "
                f"expected={self.input_dim}, actual={len(state_features)}"
            )
        normalized = tuple(
            (value - mean) / scale
            for value, mean, scale in zip(
                state_features,
                self.input_mean,
                self.input_scale,
                strict=True,
            )
        )
        kernel = tuple(
            sum(
                normalized[index] * training[index]
                for index in range(self.input_dim)
            )
            / self.input_dim
            for training in self.normalized_training_inputs
        )
        return tuple(
            bias
            + sum(
                kernel[index] * dual_weights[index]
                for index in range(len(kernel))
            )
            for dual_weights, bias in zip(
                self.dual_action_weights,
                self.action_value_bias,
                strict=True,
            )
        )


ResidualActionSelectorModel = (
    ResidualActionSelectorArtifact | KernelResidualActionSelectorArtifact
)

_SELECTOR_ARTIFACT_SCHEMA_VERSION = "residual-action-selector.v1"
_LINEAR_SELECTOR_KIND = "linear-pca-ridge-v1"
_KERNEL_SELECTOR_KIND = "linear-kernel-ridge-v1"


def _selector_model_fingerprint(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _linear_selector_fingerprint_payload(
    artifact: ResidualActionSelectorArtifact,
) -> dict[str, object]:
    return {
        "input_mean": list(artifact.input_mean),
        "input_scale": list(artifact.input_scale),
        "encoder_components": [
            list(row) for row in artifact.encoder_components
        ],
        "action_value_weights": [
            list(row) for row in artifact.action_value_weights
        ],
        "action_value_bias": list(artifact.action_value_bias),
        "ridge_strength": artifact.ridge_strength,
    }


def _kernel_selector_fingerprint_payload(
    artifact: KernelResidualActionSelectorArtifact,
) -> dict[str, object]:
    return {
        "input_mean": list(artifact.input_mean),
        "input_scale": list(artifact.input_scale),
        "normalized_training_inputs": [
            list(row) for row in artifact.normalized_training_inputs
        ],
        "dual_action_weights": [
            list(row) for row in artifact.dual_action_weights
        ],
        "action_value_bias": list(artifact.action_value_bias),
        "ridge_strength": artifact.ridge_strength,
    }


def selector_artifact_to_payload(
    artifact: ResidualActionSelectorModel,
) -> dict[str, object]:
    """Serialize a frozen selector artifact to a JSON-compatible payload."""

    if isinstance(artifact, ResidualActionSelectorArtifact):
        model_kind = _LINEAR_SELECTOR_KIND
        model_payload = _linear_selector_fingerprint_payload(artifact)
        dimensions = {
            "input_dim": artifact.input_dim,
            "latent_dim": artifact.latent_dim,
            "action_count": artifact.action_count,
            "explained_variance_ratio": artifact.explained_variance_ratio,
        }
    elif isinstance(artifact, KernelResidualActionSelectorArtifact):
        model_kind = _KERNEL_SELECTOR_KIND
        model_payload = _kernel_selector_fingerprint_payload(artifact)
        dimensions = {
            "input_dim": artifact.input_dim,
            "action_count": artifact.action_count,
        }
    else:
        raise TypeError(
            "selector artifact serialization requires a supported frozen "
            f"artifact, got {type(artifact).__name__}"
        )
    return {
        "schema_version": _SELECTOR_ARTIFACT_SCHEMA_VERSION,
        "model_kind": model_kind,
        **dimensions,
        **model_payload,
        "model_fingerprint": artifact.model_fingerprint,
    }


def _payload_float(
    payload: Mapping[str, object],
    key: str,
) -> float:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"selector artifact {key} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"selector artifact {key} must be finite")
    return resolved


def _payload_positive_int(
    payload: Mapping[str, object],
    key: str,
) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"selector artifact {key} must be a positive integer"
        )
    return value


def _payload_vector(
    payload: Mapping[str, object],
    key: str,
) -> tuple[float, ...]:
    value = payload[key]
    if not isinstance(value, list):
        raise ValueError(f"selector artifact {key} must be a JSON array")
    return tuple(
        _payload_float({"value": item}, "value")
        for item in value
    )


def _payload_matrix(
    payload: Mapping[str, object],
    key: str,
) -> tuple[tuple[float, ...], ...]:
    value = payload[key]
    if not isinstance(value, list):
        raise ValueError(f"selector artifact {key} must be a JSON array")
    rows = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list):
            raise ValueError(
                f"selector artifact {key}[{row_index}] must be a JSON array"
            )
        rows.append(
            tuple(
                _payload_float({"value": item}, "value")
                for item in row
            )
        )
    return tuple(rows)


def _validate_selector_dimensions(
    artifact: ResidualActionSelectorModel,
) -> None:
    if len(artifact.input_mean) != artifact.input_dim:
        raise ValueError("selector artifact input_mean dimension mismatch")
    if len(artifact.input_scale) != artifact.input_dim:
        raise ValueError("selector artifact input_scale dimension mismatch")
    if any(scale <= 0.0 for scale in artifact.input_scale):
        raise ValueError("selector artifact input_scale must be positive")
    if len(artifact.action_value_bias) != artifact.action_count:
        raise ValueError(
            "selector artifact action_value_bias dimension mismatch"
        )
    if artifact.ridge_strength <= 0.0:
        raise ValueError(
            "selector artifact ridge_strength must be positive"
        )
    if isinstance(artifact, ResidualActionSelectorArtifact):
        if len(artifact.encoder_components) != artifact.latent_dim or any(
            len(row) != artifact.input_dim
            for row in artifact.encoder_components
        ):
            raise ValueError(
                "selector artifact encoder_components dimension mismatch"
            )
        if len(artifact.action_value_weights) != artifact.action_count or any(
            len(row) != artifact.latent_dim
            for row in artifact.action_value_weights
        ):
            raise ValueError(
                "selector artifact action_value_weights dimension mismatch"
            )
        if not 0.0 <= artifact.explained_variance_ratio <= 1.0:
            raise ValueError(
                "selector artifact explained_variance_ratio must be in [0,1]"
            )
    else:
        training_count = len(artifact.normalized_training_inputs)
        if training_count < 1 or any(
            len(row) != artifact.input_dim
            for row in artifact.normalized_training_inputs
        ):
            raise ValueError(
                "selector artifact normalized_training_inputs dimension "
                "mismatch"
            )
        if len(artifact.dual_action_weights) != artifact.action_count or any(
            len(row) != training_count
            for row in artifact.dual_action_weights
        ):
            raise ValueError(
                "selector artifact dual_action_weights dimension mismatch"
            )


def selector_artifact_from_payload(
    payload: Mapping[str, object],
) -> ResidualActionSelectorModel:
    """Restore a selector and verify its declared model fingerprint."""

    if payload.get("schema_version") != _SELECTOR_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported selector artifact schema_version "
            f"{payload.get('schema_version')!r}"
        )
    model_kind = payload.get("model_kind")
    declared_fingerprint = payload.get("model_fingerprint")
    if not isinstance(declared_fingerprint, str) or not declared_fingerprint:
        raise ValueError(
            "selector artifact model_fingerprint must be non-empty"
        )
    input_dim = _payload_positive_int(payload, "input_dim")
    action_count = _payload_positive_int(payload, "action_count")
    common = {
        "input_mean": _payload_vector(payload, "input_mean"),
        "input_scale": _payload_vector(payload, "input_scale"),
        "action_value_bias": _payload_vector(
            payload,
            "action_value_bias",
        ),
        "input_dim": input_dim,
        "action_count": action_count,
        "ridge_strength": _payload_float(payload, "ridge_strength"),
        "model_fingerprint": declared_fingerprint,
    }
    if model_kind == _LINEAR_SELECTOR_KIND:
        artifact: ResidualActionSelectorModel = (
            ResidualActionSelectorArtifact(
                **common,
                encoder_components=_payload_matrix(
                    payload,
                    "encoder_components",
                ),
                action_value_weights=_payload_matrix(
                    payload,
                    "action_value_weights",
                ),
                latent_dim=_payload_positive_int(payload, "latent_dim"),
                explained_variance_ratio=_payload_float(
                    payload,
                    "explained_variance_ratio",
                ),
            )
        )
        fingerprint_payload = _linear_selector_fingerprint_payload(
            artifact
        )
    elif model_kind == _KERNEL_SELECTOR_KIND:
        artifact = KernelResidualActionSelectorArtifact(
            **common,
            normalized_training_inputs=_payload_matrix(
                payload,
                "normalized_training_inputs",
            ),
            dual_action_weights=_payload_matrix(
                payload,
                "dual_action_weights",
            ),
        )
        fingerprint_payload = _kernel_selector_fingerprint_payload(
            artifact
        )
    else:
        raise ValueError(
            f"unsupported selector artifact model_kind {model_kind!r}"
        )
    _validate_selector_dimensions(artifact)
    recomputed_fingerprint = _selector_model_fingerprint(
        fingerprint_payload
    )
    if recomputed_fingerprint != declared_fingerprint:
        raise ValueError(
            "selector artifact fingerprint mismatch: "
            f"declared={declared_fingerprint}, "
            f"recomputed={recomputed_fingerprint}"
        )
    return artifact


def residual_action_state_sketch(
    substrate_snapshot: SubstrateSnapshot,
    *,
    bucket_width: int = 64,
) -> tuple[float, ...]:
    """Publish a label-free, layer-aware sketch for offline selector learning."""

    if isinstance(bucket_width, bool) or not isinstance(bucket_width, int):
        raise ValueError("residual action sketch bucket_width must be an integer")
    if bucket_width < 4:
        raise ValueError(
            "residual action sketch bucket_width must be at least 4, "
            f"got {bucket_width!r}"
        )
    sequence = residual_sequence_from_snapshot(substrate_snapshot)
    if not sequence:
        raise ValueError(
            "counterfactual residual selector requires a residual sequence"
        )
    layer_indices = tuple(
        sorted(
            {
                activation.layer_index
                for step in sequence
                for activation in step.residual_activations
            }
        )
    )
    if not layer_indices:
        raise ValueError(
            "counterfactual residual selector requires residual activations"
        )
    activation_by_step_layer = tuple(
        {
            activation.layer_index: activation.activation
            for activation in step.residual_activations
        }
        for step in sequence
    )
    for layer_index in layer_indices:
        widths = {
            len(step[layer_index])
            for step in activation_by_step_layer
            if layer_index in step
        }
        if len(widths) != 1 or len(widths) == 0:
            raise ValueError(
                "counterfactual residual selector requires stable activation "
                f"width for layer {layer_index}"
            )
        if sum(layer_index in step for step in activation_by_step_layer) != len(
            activation_by_step_layer
        ):
            raise ValueError(
                "counterfactual residual selector requires every layer on "
                "every sequence step"
            )

    statistic_buckets = [
        [0.0 for _ in range(bucket_width)]
        for _ in range(3)
    ]
    statistic_counts = [
        [0 for _ in range(bucket_width)]
        for _ in range(3)
    ]
    for layer_index in layer_indices:
        layer_vectors = tuple(
            step[layer_index] for step in activation_by_step_layer
        )
        width = len(layer_vectors[0])
        mean_vector = tuple(
            sum(vector[index] for vector in layer_vectors)
            / len(layer_vectors)
            for index in range(width)
        )
        latest_vector = layer_vectors[-1]
        trend_vector = tuple(
            latest_vector[index] - layer_vectors[0][index]
            for index in range(width)
        )
        for statistic_index, vector in enumerate(
            (mean_vector, latest_vector, trend_vector)
        ):
            rms = math.sqrt(
                sum(value * value for value in vector) / max(len(vector), 1)
            )
            scale = max(rms, 1e-8)
            for coordinate_index, value in enumerate(vector):
                key = (
                    (layer_index + 1) * 2_654_435_761
                    + (coordinate_index + 1) * 2_246_822_519
                    + (statistic_index + 1) * 3_266_489_917
                )
                bucket_index = key % bucket_width
                sign = -1.0 if ((key // bucket_width) & 1) else 1.0
                statistic_buckets[statistic_index][bucket_index] += (
                    sign * float(value) / scale
                )
                statistic_counts[statistic_index][bucket_index] += 1

    projected = tuple(
        max(
            -8.0,
            min(
                8.0,
                statistic_buckets[statistic_index][bucket_index]
                / math.sqrt(
                    max(
                        statistic_counts[statistic_index][bucket_index],
                        1,
                    )
                ),
            ),
        )
        for statistic_index in range(3)
        for bucket_index in range(bucket_width)
    )
    summaries = tuple(
        summarize_residual_activations(
            step.residual_activations,
            step.feature_surface,
        )
        for step in sequence
    )
    average_summary = tuple(
        sum(summary[index] for summary in summaries) / len(summaries)
        for index in range(3)
    )
    latest_summary = summaries[-1]
    trend_summary = tuple(
        latest_summary[index] - summaries[0][index]
        for index in range(3)
    )
    span_summary = tuple(
        max(summary[index] for summary in summaries)
        - min(summary[index] for summary in summaries)
        for index in range(3)
    )
    return projected + average_summary + latest_summary + trend_summary + span_summary


def residual_action_state_vector(
    substrate_snapshot: SubstrateSnapshot,
) -> tuple[float, ...]:
    """Keep complete layer coordinates for the no-compression selector test."""

    sequence = residual_sequence_from_snapshot(substrate_snapshot)
    if not sequence:
        raise ValueError(
            "full residual action state requires a residual sequence"
        )
    layer_indices = tuple(
        sorted(
            {
                activation.layer_index
                for step in sequence
                for activation in step.residual_activations
            }
        )
    )
    if not layer_indices:
        raise ValueError(
            "full residual action state requires residual activations"
        )
    activation_by_step_layer = tuple(
        {
            activation.layer_index: activation.activation
            for activation in step.residual_activations
        }
        for step in sequence
    )
    state_values: list[float] = []
    for statistic in ("mean", "latest", "trend"):
        for layer_index in layer_indices:
            layer_vectors = tuple(
                step.get(layer_index)
                for step in activation_by_step_layer
            )
            if any(vector is None for vector in layer_vectors):
                raise ValueError(
                    "full residual action state requires every layer on "
                    "every sequence step"
                )
            resolved_vectors = tuple(
                vector
                for vector in layer_vectors
                if vector is not None
            )
            widths = {len(vector) for vector in resolved_vectors}
            if len(widths) != 1 or not widths or next(iter(widths)) == 0:
                raise ValueError(
                    "full residual action state requires one positive "
                    f"activation width for layer {layer_index}"
                )
            width = len(resolved_vectors[0])
            if statistic == "mean":
                vector = tuple(
                    sum(step[index] for step in resolved_vectors)
                    / len(resolved_vectors)
                    for index in range(width)
                )
            elif statistic == "latest":
                vector = resolved_vectors[-1]
            else:
                vector = tuple(
                    resolved_vectors[-1][index]
                    - resolved_vectors[0][index]
                    for index in range(width)
                )
            rms = math.sqrt(
                sum(value * value for value in vector) / width
            )
            scale = max(rms, 1e-8)
            state_values.extend(
                max(-8.0, min(8.0, float(value) / scale))
                for value in vector
            )
    summaries = tuple(
        summarize_residual_activations(
            step.residual_activations,
            step.feature_surface,
        )
        for step in sequence
    )
    average_summary = tuple(
        sum(summary[index] for summary in summaries) / len(summaries)
        for index in range(3)
    )
    latest_summary = summaries[-1]
    trend_summary = tuple(
        latest_summary[index] - summaries[0][index]
        for index in range(3)
    )
    span_summary = tuple(
        max(summary[index] for summary in summaries)
        - min(summary[index] for summary in summaries)
        for index in range(3)
    )
    return tuple(state_values) + (
        average_summary
        + latest_summary
        + trend_summary
        + span_summary
    )


def residual_action_state_with_committed_control_summary(
    substrate_snapshot: SubstrateSnapshot,
    *,
    committed_controls: tuple[tuple[float, ...], ...],
    committed_control_window: int,
    expected_control_dim: int,
) -> tuple[float, ...]:
    """Append a bounded owner-defined summary of active committed controls."""

    if (
        isinstance(committed_control_window, bool)
        or not isinstance(committed_control_window, int)
        or committed_control_window < 1
    ):
        raise ValueError(
            "committed control summary requires a positive integer window"
        )
    if (
        isinstance(expected_control_dim, bool)
        or not isinstance(expected_control_dim, int)
        or expected_control_dim < 1
    ):
        raise ValueError(
            "committed control summary requires a positive control dimension"
        )
    for index, control in enumerate(committed_controls):
        if len(control) != expected_control_dim:
            raise ValueError(
                "committed control summary shape mismatch: "
                f"index={index}, expected={expected_control_dim}, "
                f"actual={len(control)}"
            )
        if any(not math.isfinite(float(value)) for value in control):
            raise ValueError(
                "committed control summary requires finite controls: "
                f"index={index}"
            )

    active_controls = committed_controls[-committed_control_window:]
    zero = (0.0,) * expected_control_dim
    latest = active_controls[-1] if active_controls else zero
    previous = (
        active_controls[-2] if len(active_controls) >= 2 else zero
    )
    aggregate = tuple(
        sum(control[dimension] for control in active_controls)
        for dimension in range(expected_control_dim)
    )
    trend = tuple(
        latest[dimension] - previous[dimension]
        for dimension in range(expected_control_dim)
    )
    bounded_summary = (
        tuple(math.tanh(value) for value in aggregate)
        + tuple(math.tanh(value) for value in latest)
        + tuple(math.tanh(value) for value in trend)
        + (len(active_controls) / committed_control_window,)
    )
    expected_summary_dim = expected_control_dim * 3 + 1
    if len(bounded_summary) != expected_summary_dim or any(
        not -1.0 <= value <= 1.0 for value in bounded_summary
    ):
        raise RuntimeError(
            "committed control summary violated its bounded shape contract"
        )
    return residual_action_state_vector(substrate_snapshot) + bounded_summary


def fit_residual_action_selector(
    examples: tuple[CounterfactualActionExample, ...],
    *,
    latent_dim: int = 16,
    ridge_strength: float = 1.0,
) -> ResidualActionSelectorArtifact:
    if len(examples) < 2:
        raise ValueError(
            "counterfactual residual selector requires at least two examples"
        )
    input_dim, action_count = _validate_examples(examples)
    if isinstance(latent_dim, bool) or not isinstance(latent_dim, int):
        raise ValueError("counterfactual selector latent_dim must be an integer")
    if latent_dim < 1:
        raise ValueError(
            "counterfactual selector latent_dim must be positive, "
            f"got {latent_dim!r}"
        )
    if not math.isfinite(ridge_strength) or ridge_strength <= 0.0:
        raise ValueError(
            "counterfactual selector ridge_strength must be positive and "
            f"finite, got {ridge_strength!r}"
        )
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "counterfactual residual selector requires the vz-temporal[torch] "
            "extra"
        ) from exc

    inputs = torch.tensor(
        [example.state_features for example in examples],
        dtype=torch.float64,
        device="cpu",
    )
    targets = torch.tensor(
        [example.candidate_raw_deltas for example in examples],
        dtype=torch.float64,
        device="cpu",
    )
    input_mean = inputs.mean(dim=0)
    input_scale = inputs.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized_inputs = (inputs - input_mean) / input_scale
    _left, singular_values, right = torch.linalg.svd(
        normalized_inputs,
        full_matrices=False,
    )
    resolved_latent_dim = min(
        latent_dim,
        int(right.shape[0]),
        len(examples) - 1,
    )
    if resolved_latent_dim < 1:
        raise ValueError(
            "counterfactual selector could not resolve a positive latent dim"
        )
    components = right[:resolved_latent_dim, :]
    latent = normalized_inputs @ components.transpose(0, 1)
    target_means = targets.mean(dim=1, keepdim=True)
    target_ranges = (
        targets.max(dim=1, keepdim=True).values
        - targets.min(dim=1, keepdim=True).values
    ).clamp_min(1e-8)
    normalized_targets = (targets - target_means) / target_ranges
    design = torch.cat(
        (
            latent,
            torch.ones(
                (latent.shape[0], 1),
                dtype=torch.float64,
                device="cpu",
            ),
        ),
        dim=1,
    )
    regularizer = torch.eye(
        resolved_latent_dim + 1,
        dtype=torch.float64,
        device="cpu",
    ) * float(ridge_strength)
    regularizer[-1, -1] = 0.0
    coefficients = torch.linalg.solve(
        design.transpose(0, 1) @ design + regularizer,
        design.transpose(0, 1) @ normalized_targets,
    )
    action_weights = coefficients[:-1, :].transpose(0, 1)
    action_bias = coefficients[-1, :]
    total_variance = float((singular_values * singular_values).sum().item())
    explained_variance_ratio = (
        float(
            (
                singular_values[:resolved_latent_dim]
                * singular_values[:resolved_latent_dim]
            )
            .sum()
            .item()
        )
        / total_variance
        if total_variance > 0.0
        else 0.0
    )
    payload = {
        "input_mean": input_mean.tolist(),
        "input_scale": input_scale.tolist(),
        "encoder_components": components.tolist(),
        "action_value_weights": action_weights.tolist(),
        "action_value_bias": action_bias.tolist(),
        "ridge_strength": float(ridge_strength),
    }
    fingerprint = _selector_model_fingerprint(payload)
    return ResidualActionSelectorArtifact(
        input_mean=tuple(float(value) for value in input_mean.tolist()),
        input_scale=tuple(float(value) for value in input_scale.tolist()),
        encoder_components=tuple(
            tuple(float(value) for value in row)
            for row in components.tolist()
        ),
        action_value_weights=tuple(
            tuple(float(value) for value in row)
            for row in action_weights.tolist()
        ),
        action_value_bias=tuple(
            float(value) for value in action_bias.tolist()
        ),
        input_dim=input_dim,
        latent_dim=resolved_latent_dim,
        action_count=action_count,
        explained_variance_ratio=explained_variance_ratio,
        ridge_strength=float(ridge_strength),
        model_fingerprint=fingerprint,
    )


def fit_kernel_residual_action_selector(
    examples: tuple[CounterfactualActionExample, ...],
    *,
    ridge_strength: float = 1.0,
) -> KernelResidualActionSelectorArtifact:
    if len(examples) < 2:
        raise ValueError(
            "kernel residual selector requires at least two examples"
        )
    input_dim, action_count = _validate_examples(examples)
    if not math.isfinite(ridge_strength) or ridge_strength <= 0.0:
        raise ValueError(
            "kernel selector ridge_strength must be positive and finite, "
            f"got {ridge_strength!r}"
        )
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "kernel residual selector requires the vz-temporal[torch] extra"
        ) from exc
    inputs = torch.tensor(
        [example.state_features for example in examples],
        dtype=torch.float64,
        device="cpu",
    )
    targets = torch.tensor(
        [example.candidate_raw_deltas for example in examples],
        dtype=torch.float64,
        device="cpu",
    )
    input_mean = inputs.mean(dim=0)
    input_scale = inputs.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized_inputs = (inputs - input_mean) / input_scale
    target_means = targets.mean(dim=1, keepdim=True)
    target_ranges = (
        targets.max(dim=1, keepdim=True).values
        - targets.min(dim=1, keepdim=True).values
    ).clamp_min(1e-8)
    normalized_targets = (targets - target_means) / target_ranges
    action_bias = normalized_targets.mean(dim=0)
    centered_targets = normalized_targets - action_bias
    kernel = (
        normalized_inputs @ normalized_inputs.transpose(0, 1)
    ) / input_dim
    regularized_kernel = kernel + torch.eye(
        len(examples),
        dtype=torch.float64,
        device="cpu",
    ) * float(ridge_strength)
    dual_weights = torch.linalg.solve(
        regularized_kernel,
        centered_targets,
    ).transpose(0, 1)
    fingerprint_payload = {
        "input_mean": input_mean.tolist(),
        "input_scale": input_scale.tolist(),
        "normalized_training_inputs": normalized_inputs.tolist(),
        "dual_action_weights": dual_weights.tolist(),
        "action_value_bias": action_bias.tolist(),
        "ridge_strength": float(ridge_strength),
    }
    fingerprint = _selector_model_fingerprint(fingerprint_payload)
    return KernelResidualActionSelectorArtifact(
        input_mean=tuple(float(value) for value in input_mean.tolist()),
        input_scale=tuple(float(value) for value in input_scale.tolist()),
        normalized_training_inputs=tuple(
            tuple(float(value) for value in row)
            for row in normalized_inputs.tolist()
        ),
        dual_action_weights=tuple(
            tuple(float(value) for value in row)
            for row in dual_weights.tolist()
        ),
        action_value_bias=tuple(
            float(value) for value in action_bias.tolist()
        ),
        input_dim=input_dim,
        action_count=action_count,
        ridge_strength=float(ridge_strength),
        model_fingerprint=fingerprint,
    )


def select_counterfactual_actions(
    artifact: ResidualActionSelectorModel,
    examples: tuple[CounterfactualActionExample, ...],
    *,
    prediction_source: str,
) -> tuple[CounterfactualActionSelection, ...]:
    if not prediction_source:
        raise ValueError(
            "counterfactual selector prediction_source must be non-empty"
        )
    selections = []
    for example in examples:
        if len(example.candidate_raw_deltas) != artifact.action_count:
            raise ValueError(
                "counterfactual selector action count mismatch: "
                f"expected={artifact.action_count}, "
                f"actual={len(example.candidate_raw_deltas)}"
            )
        predicted = artifact.predict_action_values(example.state_features)
        ranked = tuple(
            sorted(
                range(artifact.action_count),
                key=lambda index: (-predicted[index], index),
            )
        )
        oracle_index = max(
            range(artifact.action_count),
            key=lambda index: (
                example.candidate_raw_deltas[index],
                -index,
            ),
        )
        selected_index = ranked[0]
        selected_delta = example.candidate_raw_deltas[selected_index]
        oracle_delta = example.candidate_raw_deltas[oracle_index]
        audit_values = example.audit_candidate_raw_deltas
        audit_oracle_index = (
            max(
                range(artifact.action_count),
                key=lambda index: (
                    audit_values[index],
                    -index,
                ),
            )
            if audit_values
            else None
        )
        audit_selected_delta = (
            audit_values[selected_index] if audit_values else None
        )
        audit_oracle_delta = (
            audit_values[audit_oracle_index]
            if audit_oracle_index is not None
            else None
        )
        selections.append(
            CounterfactualActionSelection(
                example_id=example.example_id,
                group_id=example.group_id,
                split=example.split,
                prediction_source=prediction_source,
                selected_action_index=selected_index,
                oracle_action_index=oracle_index,
                predicted_top3_action_indices=ranked[:3],
                selected_raw_delta=selected_delta,
                oracle_raw_delta=oracle_delta,
                oracle_regret=max(0.0, oracle_delta - selected_delta),
                top1_match=selected_index == oracle_index,
                top3_match=oracle_index in ranked[:3],
                selected_positive=selected_delta > 0.0,
                audit_selected_raw_delta=audit_selected_delta,
                audit_oracle_raw_delta=audit_oracle_delta,
                audit_oracle_regret=(
                    max(
                        0.0,
                        audit_oracle_delta - audit_selected_delta,
                    )
                    if (
                        audit_oracle_delta is not None
                        and audit_selected_delta is not None
                    )
                    else None
                ),
                audit_selected_positive=(
                    audit_selected_delta > 0.0
                    if audit_selected_delta is not None
                    else None
                ),
                model_fingerprint=artifact.model_fingerprint,
            )
        )
    return tuple(selections)


def grouped_cross_validate_kernel_residual_action_selector(
    examples: tuple[CounterfactualActionExample, ...],
    *,
    fold_count: int = 4,
    ridge_strength: float = 1.0,
) -> tuple[CounterfactualActionSelection, ...]:
    _validate_examples(examples)
    groups = tuple(sorted({example.group_id for example in examples}))
    if len(groups) < 2:
        raise ValueError(
            "kernel selector grouped CV requires at least two groups"
        )
    if isinstance(fold_count, bool) or not isinstance(fold_count, int):
        raise ValueError("kernel selector fold_count must be an integer")
    resolved_fold_count = min(fold_count, len(groups))
    if resolved_fold_count < 2:
        raise ValueError(
            "kernel selector fold_count must resolve to at least two"
        )
    group_to_fold = {
        group_id: index % resolved_fold_count
        for index, group_id in enumerate(groups)
    }
    selections = []
    for fold_index in range(resolved_fold_count):
        training = tuple(
            example
            for example in examples
            if group_to_fold[example.group_id] != fold_index
        )
        validation = tuple(
            example
            for example in examples
            if group_to_fold[example.group_id] == fold_index
        )
        artifact = fit_kernel_residual_action_selector(
            training,
            ridge_strength=ridge_strength,
        )
        selections.extend(
            select_counterfactual_actions(
                artifact,
                validation,
                prediction_source=(
                    f"train-kernel-grouped-cv-fold-{fold_index}"
                ),
            )
        )
    return tuple(
        sorted(selections, key=lambda selection: selection.example_id)
    )


def grouped_cross_validate_residual_action_selector(
    examples: tuple[CounterfactualActionExample, ...],
    *,
    fold_count: int = 4,
    latent_dim: int = 16,
    ridge_strength: float = 1.0,
) -> tuple[CounterfactualActionSelection, ...]:
    _validate_examples(examples)
    groups = tuple(sorted({example.group_id for example in examples}))
    if len(groups) < 2:
        raise ValueError(
            "counterfactual selector grouped CV requires at least two groups"
        )
    if isinstance(fold_count, bool) or not isinstance(fold_count, int):
        raise ValueError("counterfactual selector fold_count must be an integer")
    resolved_fold_count = min(fold_count, len(groups))
    if resolved_fold_count < 2:
        raise ValueError(
            "counterfactual selector fold_count must resolve to at least two"
        )
    group_to_fold = {
        group_id: index % resolved_fold_count
        for index, group_id in enumerate(groups)
    }
    selections = []
    for fold_index in range(resolved_fold_count):
        training = tuple(
            example
            for example in examples
            if group_to_fold[example.group_id] != fold_index
        )
        validation = tuple(
            example
            for example in examples
            if group_to_fold[example.group_id] == fold_index
        )
        if not training or not validation:
            raise ValueError(
                "counterfactual selector grouped CV produced an empty fold"
            )
        artifact = fit_residual_action_selector(
            training,
            latent_dim=latent_dim,
            ridge_strength=ridge_strength,
        )
        selections.extend(
            select_counterfactual_actions(
                artifact,
                validation,
                prediction_source=f"train-grouped-cv-fold-{fold_index}",
            )
        )
    return tuple(
        sorted(selections, key=lambda selection: selection.example_id)
    )


def summarize_action_selections(
    selections: tuple[CounterfactualActionSelection, ...],
) -> tuple[tuple[str, float], ...]:
    if not selections:
        return (
            ("count", 0.0),
            ("mean_selected_raw_delta", 0.0),
            ("mean_oracle_raw_delta", 0.0),
            ("mean_oracle_regret", 0.0),
            ("top1_rate", 0.0),
            ("top3_rate", 0.0),
            ("selected_positive_rate", 0.0),
            ("audit_available_rate", 0.0),
            ("mean_audit_selected_raw_delta", 0.0),
            ("mean_audit_oracle_raw_delta", 0.0),
            ("mean_audit_oracle_regret", 0.0),
            ("audit_selected_positive_rate", 0.0),
        )
    count = len(selections)
    audit_selections = tuple(
        selection
        for selection in selections
        if selection.audit_selected_raw_delta is not None
        and selection.audit_oracle_raw_delta is not None
        and selection.audit_oracle_regret is not None
        and selection.audit_selected_positive is not None
    )
    audit_count = len(audit_selections)
    return (
        ("count", float(count)),
        (
            "mean_selected_raw_delta",
            sum(selection.selected_raw_delta for selection in selections)
            / count,
        ),
        (
            "mean_oracle_raw_delta",
            sum(selection.oracle_raw_delta for selection in selections)
            / count,
        ),
        (
            "mean_oracle_regret",
            sum(selection.oracle_regret for selection in selections) / count,
        ),
        (
            "top1_rate",
            sum(selection.top1_match for selection in selections) / count,
        ),
        (
            "top3_rate",
            sum(selection.top3_match for selection in selections) / count,
        ),
        (
            "selected_positive_rate",
            sum(selection.selected_positive for selection in selections)
            / count,
        ),
        ("audit_available_rate", audit_count / count),
        (
            "mean_audit_selected_raw_delta",
            (
                sum(
                    selection.audit_selected_raw_delta
                    for selection in audit_selections
                    if selection.audit_selected_raw_delta is not None
                )
                / audit_count
                if audit_count
                else 0.0
            ),
        ),
        (
            "mean_audit_oracle_raw_delta",
            (
                sum(
                    selection.audit_oracle_raw_delta
                    for selection in audit_selections
                    if selection.audit_oracle_raw_delta is not None
                )
                / audit_count
                if audit_count
                else 0.0
            ),
        ),
        (
            "mean_audit_oracle_regret",
            (
                sum(
                    selection.audit_oracle_regret
                    for selection in audit_selections
                    if selection.audit_oracle_regret is not None
                )
                / audit_count
                if audit_count
                else 0.0
            ),
        ),
        (
            "audit_selected_positive_rate",
            (
                sum(
                    selection.audit_selected_positive
                    for selection in audit_selections
                    if selection.audit_selected_positive is not None
                )
                / audit_count
                if audit_count
                else 0.0
            ),
        ),
    )


def _validate_examples(
    examples: tuple[CounterfactualActionExample, ...],
) -> tuple[int, int]:
    if not examples:
        raise ValueError(
            "counterfactual residual selector requires non-empty examples"
        )
    input_dims = {len(example.state_features) for example in examples}
    action_counts = {
        len(example.candidate_raw_deltas) for example in examples
    }
    if len(input_dims) != 1 or 0 in input_dims:
        raise ValueError(
            "counterfactual selector examples require one positive input dim"
        )
    if len(action_counts) != 1 or 0 in action_counts:
        raise ValueError(
            "counterfactual selector examples require one positive action count"
        )
    action_count = next(iter(action_counts))
    if any(
        example.audit_candidate_raw_deltas
        and len(example.audit_candidate_raw_deltas) != action_count
        for example in examples
    ):
        raise ValueError(
            "counterfactual selector audit action count must match target "
            "action count"
        )
    if len({example.example_id for example in examples}) != len(examples):
        raise ValueError(
            "counterfactual selector example_id values must be unique"
        )
    if any(
        not example.group_id or not example.split
        for example in examples
    ):
        raise ValueError(
            "counterfactual selector examples require group_id and split"
        )
    if any(
        not math.isfinite(value)
        for example in examples
        for value in (
            *example.state_features,
            *example.candidate_raw_deltas,
            *example.audit_candidate_raw_deltas,
        )
    ):
        raise ValueError(
            "counterfactual selector examples require finite values"
        )
    return next(iter(input_dims)), next(iter(action_counts))
