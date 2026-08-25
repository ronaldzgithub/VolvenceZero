"""Fit and freeze the model-bound reader/executor owner artifacts.

This is the one bridge from the sealed ETA proxy instrument into the runtime
owner chain.  It reuses the exact S3 reader and rank-r executor mathematics,
then serializes their learned parameters into immutable contracts.  The gate
artifact is an explicit always-steer SHADOW collection control; C3 trains and
adjudicates a fresh gate from dialogue-domain terminal PE credit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from volvence_zero.agent.eta_conditional_steering_screen import (
    _ConditionalOperator,
    _subgoal_vocabulary,
    _train_operator,
)
from volvence_zero.agent.eta_conflict_instrument import (
    build_conflict_junction_rows,
)
from volvence_zero.agent.eta_proof_benchmark import ETAProofCorpus
from volvence_zero.agent.eta_read_steer_prereq import (
    _capture_examples,
    _labelled_rows,
    _per_row_baseline_nll,
    _per_row_controlled_nll,
    _reader_accuracy,
    fit_condition_reader,
)
from volvence_zero.steering_contracts import (
    STEERING_ARTIFACT_BUNDLE_SCHEMA_VERSION,
    STEERING_EXECUTOR_ARTIFACT_SCHEMA_VERSION,
    STEERING_GATE_ARTIFACT_SCHEMA_VERSION,
    STEERING_READER_ARTIFACT_SCHEMA_VERSION,
    SteeringArtifactBundle,
    SteeringExecutorArtifact,
    SteeringGateArtifact,
    SteeringReaderArtifact,
)
from volvence_zero.substrate import OpenWeightResidualRuntime


_DIALOGUE_GATE_FEATURES = (
    "belief_margin",
    "fresh_margin",
    "belief_disagrees_fresh",
    "base_action_entropy",
    "prediction_error_magnitude",
    "staleness_proxy",
)


@dataclass(frozen=True)
class SteeringArtifactFitReport:
    train_row_count: int
    heldout_row_count: int
    reader_heldout_accuracy: float
    heldout_noop_nll: float
    heldout_online_steer_nll: float
    heldout_sensor_off_nll: float
    heldout_gain_vs_noop_nll: float
    heldout_conditional_advantage_nll: float
    reader_ridge_lambda: float
    executor_updates: int
    executor_learning_rate: float
    steering_rank: int
    seed: int
    control_norm_cap_ratio: float
    free_bias_present: bool
    zero_code_strict_noop: bool
    substrate_trainable_parameter_count: int
    reader_executor_frozen_for_dialogue: bool
    description: str

    @property
    def prerequisite_passed(self) -> bool:
        return (
            self.reader_heldout_accuracy >= 0.80
            and self.heldout_gain_vs_noop_nll > 0.0
            and self.heldout_conditional_advantage_nll > 0.0
            and not self.free_bias_present
            and self.zero_code_strict_noop
            and self.substrate_trainable_parameter_count == 0
            and self.reader_executor_frozen_for_dialogue
        )


@dataclass(frozen=True)
class SteeringArtifactFitResult:
    bundle: SteeringArtifactBundle
    report: SteeringArtifactFitReport


def _matrix(values: object) -> tuple[tuple[float, ...], ...]:
    if isinstance(values, np.ndarray):
        raw = values.tolist()
    else:
        import torch

        if not isinstance(values, torch.Tensor):
            raise TypeError(
                "steering artifact matrix must be numpy.ndarray or torch.Tensor"
            )
        raw = values.detach().cpu().tolist()
    return tuple(tuple(float(value) for value in row) for row in raw)


def fit_steering_artifact_bundle(
    *,
    corpus: ETAProofCorpus,
    runtime: OpenWeightResidualRuntime,
    scorer: Any,
    model_weights_sha256: str,
    source_preregistration_sha256: str,
    injection_layer_index: int = 20,
    residual_width: int = 896,
    steering_rank: int = 8,
    executor_updates: int = 80,
    executor_learning_rate: float = 0.01,
    reader_ridge_lambda: float = 10.0,
    batch_size: int = 32,
    seed: int = 0,
    control_norm_cap_ratio: float = 0.25,
    progress: Any | None = None,
) -> SteeringArtifactFitResult:
    """Fit once on the sealed proxy corpus and freeze runtime artifacts."""

    import torch

    if type(seed) is not int or seed < 0:
        raise ValueError("steering artifact fit seed must be a non-negative int")
    if (
        type(control_norm_cap_ratio) is not float
        or not math.isfinite(control_norm_cap_ratio)
        or not 0.0 < control_norm_cap_ratio <= 2.0
    ):
        raise ValueError(
            "steering artifact fit control_norm_cap_ratio must be a finite "
            "float within (0, 2]"
        )
    if scorer.trainable_parameters():
        raise RuntimeError("steering artifact fit requires a frozen substrate")
    if (
        not math.isfinite(float(scorer.probe_hidden_norm))
        or float(scorer.probe_hidden_norm) <= 0.0
        or not math.isfinite(float(scorer.control_norm_cap))
        or not math.isclose(
            float(scorer.control_norm_cap) / float(scorer.probe_hidden_norm),
            control_norm_cap_ratio,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            "steering artifact fit scorer control-norm ratio differs from "
            "the frozen owner configuration"
        )
    if steering_rank < 1 or steering_rank > residual_width:
        raise ValueError("steering_rank must be within residual width")
    vocabulary = _subgoal_vocabulary(corpus)
    subgoal_index = {name: index for index, name in enumerate(vocabulary)}
    train_rows = _labelled_rows(build_conflict_junction_rows(corpus, split="train"))
    heldout_rows = _labelled_rows(
        build_conflict_junction_rows(corpus, split="heldout")
    )
    train_examples = _capture_examples(
        train_rows,
        runtime=runtime,
        scorer=scorer,
        subgoal_index=subgoal_index,
        injection_layer_index=injection_layer_index,
        residual_width=residual_width,
        progress=progress,
        split_label="artifact-train",
    )
    heldout_examples = _capture_examples(
        heldout_rows,
        runtime=runtime,
        scorer=scorer,
        subgoal_index=subgoal_index,
        injection_layer_index=injection_layer_index,
        residual_width=residual_width,
        progress=progress,
        split_label="artifact-heldout",
    )
    reader = fit_condition_reader(
        train_examples,
        class_count=len(vocabulary),
        ridge_lambda=reader_ridge_lambda,
    )
    train_residuals = torch.tensor(
        [item.action_residual for item in train_examples], dtype=torch.float32
    )
    train_subgoals = torch.tensor(
        [item.subgoal_index for item in train_examples], dtype=torch.long
    )
    operator = _ConditionalOperator(
        torch=torch,
        width=residual_width,
        rank=steering_rank,
        class_count=len(vocabulary),
        conditional=True,
        seed=seed,
    )
    _train_operator(
        torch=torch,
        operator=operator,
        residuals=train_residuals,
        subgoal_indices=train_subgoals,
        action_indices=tuple(item.action_index for item in train_examples),
        texts=tuple(item.observation_text for item in train_examples),
        scorer=scorer,
        updates=executor_updates,
        learning_rate=executor_learning_rate,
        batch_size=batch_size,
        seed=seed,
        progress=progress,
        label="artifact-executor",
    )
    sensor_off_operator = _ConditionalOperator(
        torch=torch,
        width=residual_width,
        rank=steering_rank,
        class_count=len(vocabulary),
        conditional=False,
        seed=seed,
    )
    _train_operator(
        torch=torch,
        operator=sensor_off_operator,
        residuals=train_residuals,
        subgoal_indices=train_subgoals,
        action_indices=tuple(item.action_index for item in train_examples),
        texts=tuple(item.observation_text for item in train_examples),
        scorer=scorer,
        updates=executor_updates,
        learning_rate=executor_learning_rate,
        batch_size=batch_size,
        seed=seed,
        progress=progress,
        label="artifact-sensor-off-executor",
    )
    heldout_action = torch.tensor(
        [item.action_residual for item in heldout_examples], dtype=torch.float32
    )
    heldout_context = np.asarray(
        [item.context_residual for item in heldout_examples], dtype=np.float64
    )
    belief_indices_np = reader.predict(heldout_context)
    belief_indices = torch.tensor(belief_indices_np.tolist(), dtype=torch.long)
    with torch.no_grad():
        heldout_deltas = operator.deltas(
            residuals=heldout_action,
            subgoal_indices=belief_indices,
        )
        saved_codes = operator._Z.detach().clone()
        operator._Z.zero_()
        zero_delta = operator.deltas(
            residuals=heldout_action,
            subgoal_indices=belief_indices,
        )
        zero_code_max_abs = float(zero_delta.abs().max())
        operator._Z.copy_(saved_codes)
        sensor_off_deltas = sensor_off_operator.deltas(
            residuals=heldout_action,
            subgoal_indices=belief_indices,
        )
    texts = tuple(item.observation_text for item in heldout_examples)
    action_indices = tuple(item.action_index for item in heldout_examples)
    noop_rows = _per_row_baseline_nll(
        texts=texts,
        action_indices=action_indices,
        scorer=scorer,
        batch_size=batch_size,
    )
    steer_rows = _per_row_controlled_nll(
        deltas=heldout_deltas,
        texts=texts,
        action_indices=action_indices,
        scorer=scorer,
        batch_size=batch_size,
    )
    sensor_off_rows = _per_row_controlled_nll(
        deltas=sensor_off_deltas,
        texts=texts,
        action_indices=action_indices,
        scorer=scorer,
        batch_size=batch_size,
    )
    noop_mean = sum(noop_rows) / len(noop_rows)
    steer_mean = sum(steer_rows) / len(steer_rows)
    sensor_off_mean = sum(sensor_off_rows) / len(sensor_off_rows)
    prefix = f"{source_preregistration_sha256[:12]}:{model_weights_sha256[:12]}"
    reader_artifact = SteeringReaderArtifact(
        schema_version=STEERING_READER_ARTIFACT_SCHEMA_VERSION,
        artifact_id=f"steering-reader:{prefix}",
        model_id=runtime.model_id,
        model_weights_sha256=model_weights_sha256,
        source_preregistration_sha256=source_preregistration_sha256,
        layer_index=injection_layer_index,
        residual_width=residual_width,
        class_labels=vocabulary,
        weights=_matrix(reader.weights),
        feature_mean=tuple(float(value) for value in reader.feature_mean.tolist()),
        feature_scale=tuple(float(value) for value in reader.feature_scale.tolist()),
        ridge_lambda=reader_ridge_lambda,
        description=(
            "Frozen S3 context-residual ridge reader refit on the exact bundled "
            "substrate weights."
        ),
    )
    u_factors, v_factors, condition_codes = operator.parameters()
    executor_artifact = SteeringExecutorArtifact(
        schema_version=STEERING_EXECUTOR_ARTIFACT_SCHEMA_VERSION,
        artifact_id=f"steering-executor:{prefix}",
        model_id=runtime.model_id,
        model_weights_sha256=model_weights_sha256,
        source_preregistration_sha256=source_preregistration_sha256,
        reader_artifact_id=reader_artifact.artifact_id,
        layer_index=injection_layer_index,
        residual_width=residual_width,
        rank=steering_rank,
        class_labels=vocabulary,
        u_factors=_matrix(u_factors),
        v_factors=_matrix(v_factors),
        condition_codes=_matrix(condition_codes),
        control_norm_cap_ratio=control_norm_cap_ratio,
        free_bias_present=False,
        zero_code_strict_noop=zero_code_max_abs == 0.0,
        description=(
            f"Frozen rank-{steering_rank} multiplicative executor trained "
            "once on the sealed S3 proxy instrument; no dialogue-domain "
            "executor update."
        ),
    )
    sensor_u, sensor_v, sensor_z = sensor_off_operator.parameters()
    unconditional_code = tuple(float(value) for value in sensor_z[0].detach().cpu().tolist())
    sensor_off_artifact = SteeringExecutorArtifact(
        schema_version=STEERING_EXECUTOR_ARTIFACT_SCHEMA_VERSION,
        artifact_id=f"steering-executor-sensor-off:{prefix}",
        model_id=runtime.model_id,
        model_weights_sha256=model_weights_sha256,
        source_preregistration_sha256=source_preregistration_sha256,
        reader_artifact_id=reader_artifact.artifact_id,
        layer_index=injection_layer_index,
        residual_width=residual_width,
        rank=steering_rank,
        class_labels=vocabulary,
        u_factors=_matrix(sensor_u),
        v_factors=_matrix(sensor_v),
        condition_codes=tuple(unconditional_code for _ in vocabulary),
        control_norm_cap_ratio=control_norm_cap_ratio,
        free_bias_present=False,
        zero_code_strict_noop=True,
        description=(
            "Matched-budget unconditional executor used only for the B3 "
            "sensor-off SHADOW ablation."
        ),
    )
    gate_artifact = SteeringGateArtifact(
        schema_version=STEERING_GATE_ARTIFACT_SCHEMA_VERSION,
        artifact_id=f"steering-gate-shadow-collector:{prefix}",
        source_preregistration_sha256=source_preregistration_sha256,
        feature_names=_DIALOGUE_GATE_FEATURES,
        weights=tuple((0.0, 0.0) for _ in _DIALOGUE_GATE_FEATURES),
        bias=(-4.0, 4.0),
        policy_version=1,
        description=(
            "Explicit always-steer SHADOW collection control; never an ACTIVE "
            "promotion candidate."
        ),
    )
    bundle = SteeringArtifactBundle(
        schema_version=STEERING_ARTIFACT_BUNDLE_SCHEMA_VERSION,
        bundle_id=f"steering-dialogue-shadow:{prefix}",
        reader=reader_artifact,
        executor=executor_artifact,
        gate=gate_artifact,
        sensor_off_executor=sensor_off_artifact,
        description=(
            "Model-bound frozen reader/executor with an explicit SHADOW-only "
            "collection gate."
        ),
    )
    report = SteeringArtifactFitReport(
        train_row_count=len(train_examples),
        heldout_row_count=len(heldout_examples),
        reader_heldout_accuracy=_reader_accuracy(reader, heldout_examples),
        heldout_noop_nll=noop_mean,
        heldout_online_steer_nll=steer_mean,
        heldout_sensor_off_nll=sensor_off_mean,
        heldout_gain_vs_noop_nll=noop_mean - steer_mean,
        heldout_conditional_advantage_nll=sensor_off_mean - steer_mean,
        reader_ridge_lambda=reader_ridge_lambda,
        executor_updates=executor_updates,
        executor_learning_rate=executor_learning_rate,
        steering_rank=steering_rank,
        seed=seed,
        control_norm_cap_ratio=control_norm_cap_ratio,
        free_bias_present=False,
        zero_code_strict_noop=zero_code_max_abs == 0.0,
        substrate_trainable_parameter_count=0,
        reader_executor_frozen_for_dialogue=True,
        description=(
            "Runtime bundle fit on the sealed proxy surface and frozen before "
            "any dialogue-domain terminal PE is observed."
        ),
    )
    if not all(
        math.isfinite(value)
        for value in (
            report.reader_heldout_accuracy,
            report.heldout_noop_nll,
            report.heldout_online_steer_nll,
            report.heldout_sensor_off_nll,
        )
    ):
        raise RuntimeError("steering artifact fit produced non-finite evidence")
    return SteeringArtifactFitResult(bundle=bundle, report=report)


__all__ = (
    "SteeringArtifactFitReport",
    "SteeringArtifactFitResult",
    "fit_steering_artifact_bundle",
)
