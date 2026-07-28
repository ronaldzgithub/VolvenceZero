from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

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
    model_fingerprint: str


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
    fingerprint = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
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
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
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
        )
    count = len(selections)
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
        )
    ):
        raise ValueError(
            "counterfactual selector examples require finite values"
        )
    return next(iter(input_dims)), next(iter(action_counts))
