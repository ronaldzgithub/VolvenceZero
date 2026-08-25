"""Fit bounded residual artifacts from frozen named-action owner readouts.

This module is an additive fit surface for domains whose pre-action owner has
already published a typed recommendation.  It intentionally does not import a
lifeform package and it never accepts evaluator truth, observed outcomes,
rewards, or judge scores.  A domain adapter must materialize the frozen corpus
before this owner captures model residuals.

The older :mod:`steering_artifact_training` ETA fit surface remains byte-for-
byte untouched because published P4.6 evidence pins that source file.  Both
surfaces reuse the same frozen linear reader, multiplicative rank-r operator,
strict-zero code, matched unconditional operator, and immutable runtime
artifact contracts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np

from volvence_zero.agent.eta_conditional_steering_screen import (
    _ConditionalOperator,
    _train_operator,
)
from volvence_zero.agent.eta_read_steer_prereq import (
    _per_row_baseline_nll,
    _per_row_controlled_nll,
    _reader_accuracy,
    fit_condition_reader,
)
from volvence_zero.agent.steering_artifact_training import (
    _DIALOGUE_GATE_FEATURES,
    _matrix,
    SteeringArtifactFitReport,
    SteeringArtifactFitResult,
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


NAMED_ACTION_STEERING_CORPUS_SCHEMA_VERSION = "named-action-steering-corpus.v1"
_FIT_OWNER = "volvence_zero.agent.named_action_steering_artifact_training"


@dataclass(frozen=True)
class NamedActionSteeringRow:
    """One outcome-free row captured before any action is exposed."""

    row_id: str
    subject_scope: str
    action_text: str
    condition_text: str
    condition_label: str
    target_action_id: str
    source_condition_lineage_sha256: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("row_id", self.row_id),
            ("subject_scope", self.subject_scope),
            ("action_text", self.action_text),
            ("condition_text", self.condition_text),
            ("condition_label", self.condition_label),
            ("target_action_id", self.target_action_id),
        ):
            _require_text(value, field_name)
        _require_sha256(
            self.source_condition_lineage_sha256,
            "source_condition_lineage_sha256",
        )
        if self.condition_label != self.target_action_id:
            raise ValueError(
                "named-action steering requires the condition label to equal "
                "the pre-action owner recommendation"
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "row_id": self.row_id,
            "subject_scope": self.subject_scope,
            "action_text": self.action_text,
            "condition_text": self.condition_text,
            "condition_label": self.condition_label,
            "target_action_id": self.target_action_id,
            "source_condition_lineage_sha256": self.source_condition_lineage_sha256,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "NamedActionSteeringRow":
        raw = _exact_mapping(
            payload,
            expected={
                "row_id",
                "subject_scope",
                "action_text",
                "condition_text",
                "condition_label",
                "target_action_id",
                "source_condition_lineage_sha256",
            },
            label="named-action steering row",
        )
        return cls(
            row_id=_mapping_text(raw, "row_id"),
            subject_scope=_mapping_text(raw, "subject_scope"),
            action_text=_mapping_text(raw, "action_text"),
            condition_text=_mapping_text(raw, "condition_text"),
            condition_label=_mapping_text(raw, "condition_label"),
            target_action_id=_mapping_text(raw, "target_action_id"),
            source_condition_lineage_sha256=_mapping_text(
                raw,
                "source_condition_lineage_sha256",
            ),
        )


@dataclass(frozen=True)
class NamedActionSteeringCorpus:
    """Content-addressed, group-disjoint residual fit corpus."""

    source_protocol_sha256: str
    action_ids: tuple[str, ...]
    class_labels: tuple[str, ...]
    train_rows: tuple[NamedActionSteeringRow, ...]
    heldout_rows: tuple[NamedActionSteeringRow, ...]
    description: str
    schema_version: str = NAMED_ACTION_STEERING_CORPUS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != NAMED_ACTION_STEERING_CORPUS_SCHEMA_VERSION:
            raise ValueError("named-action steering corpus schema mismatch")
        _require_sha256(self.source_protocol_sha256, "source_protocol_sha256")
        _require_unique_texts(self.action_ids, "action_ids")
        _require_unique_texts(self.class_labels, "class_labels")
        _require_text(self.description, "description")
        if len(self.action_ids) < 2:
            raise ValueError("named-action steering requires at least two actions")
        if self.class_labels != self.action_ids:
            raise ValueError(
                "named-action steering v1 requires class_labels to exactly "
                "match the ordered action surface"
            )
        if not self.train_rows or not self.heldout_rows:
            raise ValueError("named-action steering requires train and heldout rows")

        rows = (*self.train_rows, *self.heldout_rows)
        row_ids = tuple(row.row_id for row in rows)
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("named-action steering row IDs must be globally unique")
        train_scopes = {row.subject_scope for row in self.train_rows}
        heldout_scopes = {row.subject_scope for row in self.heldout_rows}
        if train_scopes & heldout_scopes:
            raise ValueError(
                "named-action steering train and heldout subject scopes must be disjoint"
            )
        for split_name, split_rows in (
            ("train", self.train_rows),
            ("heldout", self.heldout_rows),
        ):
            labels = {row.condition_label for row in split_rows}
            actions = {row.target_action_id for row in split_rows}
            expected = set(self.action_ids)
            if labels != expected or actions != expected:
                raise ValueError(
                    f"named-action steering {split_name} split must cover every "
                    "action/condition class"
                )
            for row in split_rows:
                if row.condition_label not in expected:
                    raise ValueError(
                        f"named-action steering {split_name} row is outside the class surface"
                    )

    @property
    def corpus_id(self) -> str:
        return _sha256_json(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_protocol_sha256": self.source_protocol_sha256,
            "action_ids": list(self.action_ids),
            "class_labels": list(self.class_labels),
            "train_rows": [row.to_payload() for row in self.train_rows],
            "heldout_rows": [row.to_payload() for row in self.heldout_rows],
            "description": self.description,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "NamedActionSteeringCorpus":
        raw = _exact_mapping(
            payload,
            expected={
                "schema_version",
                "source_protocol_sha256",
                "action_ids",
                "class_labels",
                "train_rows",
                "heldout_rows",
                "description",
            },
            label="named-action steering corpus",
        )
        return cls(
            schema_version=_mapping_text(raw, "schema_version"),
            source_protocol_sha256=_mapping_text(
                raw,
                "source_protocol_sha256",
            ),
            action_ids=_text_tuple(raw["action_ids"], "action_ids"),
            class_labels=_text_tuple(raw["class_labels"], "class_labels"),
            train_rows=_row_tuple(raw["train_rows"], "train_rows"),
            heldout_rows=_row_tuple(raw["heldout_rows"], "heldout_rows"),
            description=_mapping_text(raw, "description"),
        )


@dataclass(frozen=True)
class _NamedActionExample:
    case_id: str
    observation_text: str
    subgoal_revealed_text: str
    subgoal_index: int
    action_index: int
    action_residual: tuple[float, ...]
    context_residual: tuple[float, ...]


def fit_named_action_steering_artifact_bundle(
    *,
    corpus: NamedActionSteeringCorpus,
    runtime: OpenWeightResidualRuntime,
    scorer: Any,
    model_weights_sha256: str,
    source_preregistration_sha256: str,
    injection_layer_index: int,
    residual_width: int,
    steering_rank: int = 8,
    executor_updates: int = 80,
    executor_learning_rate: float = 0.01,
    reader_ridge_lambda: float = 10.0,
    batch_size: int = 32,
    seed: int = 0,
    control_norm_cap_ratio: float = 0.25,
    progress: Any | None = None,
) -> SteeringArtifactFitResult:
    """Fit one frozen bundle from pre-action named owner recommendations."""

    import torch

    expected_fit_lineage = named_action_fit_lineage_sha256(corpus)
    if source_preregistration_sha256 != expected_fit_lineage:
        raise ValueError(
            "named-action source preregistration does not bind the exact "
            "protocol and corpus"
        )
    _require_fit_configuration(
        scorer=scorer,
        corpus=corpus,
        seed=seed,
        control_norm_cap_ratio=control_norm_cap_ratio,
        steering_rank=steering_rank,
        residual_width=residual_width,
    )
    class_index = {
        label: index for index, label in enumerate(corpus.class_labels)
    }
    train_examples = _capture_examples(
        corpus.train_rows,
        runtime=runtime,
        scorer=scorer,
        class_index=class_index,
        injection_layer_index=injection_layer_index,
        residual_width=residual_width,
        progress=progress,
        split_label="named-action-train",
    )
    heldout_examples = _capture_examples(
        corpus.heldout_rows,
        runtime=runtime,
        scorer=scorer,
        class_index=class_index,
        injection_layer_index=injection_layer_index,
        residual_width=residual_width,
        progress=progress,
        split_label="named-action-heldout",
    )
    reader = fit_condition_reader(
        train_examples,
        class_count=len(corpus.class_labels),
        ridge_lambda=reader_ridge_lambda,
    )
    train_residuals = torch.tensor(
        [example.action_residual for example in train_examples],
        dtype=torch.float32,
    )
    train_conditions = torch.tensor(
        [example.subgoal_index for example in train_examples],
        dtype=torch.long,
    )
    operator = _ConditionalOperator(
        torch=torch,
        width=residual_width,
        rank=steering_rank,
        class_count=len(corpus.class_labels),
        conditional=True,
        seed=seed,
    )
    _train_operator(
        torch=torch,
        operator=operator,
        residuals=train_residuals,
        subgoal_indices=train_conditions,
        action_indices=tuple(example.action_index for example in train_examples),
        texts=tuple(example.observation_text for example in train_examples),
        scorer=scorer,
        updates=executor_updates,
        learning_rate=executor_learning_rate,
        batch_size=batch_size,
        seed=seed,
        progress=progress,
        label="named-action-conditional-executor",
    )
    sensor_off_operator = _ConditionalOperator(
        torch=torch,
        width=residual_width,
        rank=steering_rank,
        class_count=len(corpus.class_labels),
        conditional=False,
        seed=seed,
    )
    _train_operator(
        torch=torch,
        operator=sensor_off_operator,
        residuals=train_residuals,
        subgoal_indices=train_conditions,
        action_indices=tuple(example.action_index for example in train_examples),
        texts=tuple(example.observation_text for example in train_examples),
        scorer=scorer,
        updates=executor_updates,
        learning_rate=executor_learning_rate,
        batch_size=batch_size,
        seed=seed,
        progress=progress,
        label="named-action-sensor-off-executor",
    )

    heldout_residuals = torch.tensor(
        [example.action_residual for example in heldout_examples],
        dtype=torch.float32,
    )
    heldout_context = np.asarray(
        [example.context_residual for example in heldout_examples],
        dtype=np.float64,
    )
    predicted_indices = reader.predict(heldout_context)
    belief_indices = torch.tensor(predicted_indices.tolist(), dtype=torch.long)
    with torch.no_grad():
        heldout_deltas = operator.deltas(
            residuals=heldout_residuals,
            subgoal_indices=belief_indices,
        )
        saved_codes = operator._Z.detach().clone()
        operator._Z.zero_()
        zero_delta = operator.deltas(
            residuals=heldout_residuals,
            subgoal_indices=belief_indices,
        )
        zero_code_max_abs = float(zero_delta.abs().max())
        operator._Z.copy_(saved_codes)
        sensor_off_deltas = sensor_off_operator.deltas(
            residuals=heldout_residuals,
            subgoal_indices=belief_indices,
        )

    texts = tuple(example.observation_text for example in heldout_examples)
    action_indices = tuple(example.action_index for example in heldout_examples)
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
        class_labels=corpus.class_labels,
        weights=_matrix(reader.weights),
        feature_mean=tuple(float(value) for value in reader.feature_mean.tolist()),
        feature_scale=tuple(float(value) for value in reader.feature_scale.tolist()),
        ridge_lambda=reader_ridge_lambda,
        description=(
            "Frozen linear reader fit only on pre-action named owner-condition "
            "residuals; no evaluator or future outcome input."
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
        class_labels=corpus.class_labels,
        u_factors=_matrix(u_factors),
        v_factors=_matrix(v_factors),
        condition_codes=_matrix(condition_codes),
        control_norm_cap_ratio=control_norm_cap_ratio,
        free_bias_present=False,
        zero_code_strict_noop=zero_code_max_abs == 0.0,
        description=(
            f"Frozen rank-{steering_rank} multiplicative named-action executor; "
            "target actions come only from pre-action owner recommendations."
        ),
    )
    sensor_u, sensor_v, sensor_z = sensor_off_operator.parameters()
    unconditional_code = tuple(
        float(value) for value in sensor_z[0].detach().cpu().tolist()
    )
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
        class_labels=corpus.class_labels,
        u_factors=_matrix(sensor_u),
        v_factors=_matrix(sensor_v),
        condition_codes=tuple(unconditional_code for _ in corpus.class_labels),
        control_norm_cap_ratio=control_norm_cap_ratio,
        free_bias_present=False,
        zero_code_strict_noop=True,
        description=(
            "Matched-budget unconditional named-action executor for the "
            "sensor-off SHADOW arm only."
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
            "Explicit always-steer SHADOW collection control. A relationship "
            "consumer must project its PE-trained typed gate separately."
        ),
    )
    bundle = SteeringArtifactBundle(
        schema_version=STEERING_ARTIFACT_BUNDLE_SCHEMA_VERSION,
        bundle_id=f"steering-named-action-shadow:{prefix}",
        reader=reader_artifact,
        executor=executor_artifact,
        gate=gate_artifact,
        sensor_off_executor=sensor_off_artifact,
        description=(
            "Model-bound named-action reader/executor bundle with a SHADOW-only "
            "collection gate; no product wiring authorization."
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
            "Named-action artifacts fit from pre-action owner readouts and frozen "
            "before any heldout product outcome is observed."
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
        raise RuntimeError("named-action steering fit produced non-finite evidence")
    return SteeringArtifactFitResult(bundle=bundle, report=report)


def named_action_fit_lineage_sha256(corpus: NamedActionSteeringCorpus) -> str:
    """Bind one artifact family to both the frozen protocol and exact corpus."""

    if not isinstance(corpus, NamedActionSteeringCorpus):
        raise TypeError("named-action fit lineage requires a typed corpus")
    return _sha256_json(
        {
            "protocol_id": corpus.source_protocol_sha256,
            "corpus_id": corpus.corpus_id,
            "owner": _FIT_OWNER,
        }
    )


def _capture_examples(
    rows: tuple[NamedActionSteeringRow, ...],
    *,
    runtime: OpenWeightResidualRuntime,
    scorer: Any,
    class_index: dict[str, int],
    injection_layer_index: int,
    residual_width: int,
    progress: Any | None,
    split_label: str,
) -> tuple[_NamedActionExample, ...]:
    examples: list[_NamedActionExample] = []
    for index, row in enumerate(rows):
        examples.append(
            _NamedActionExample(
                case_id=row.row_id,
                observation_text=row.action_text,
                subgoal_revealed_text=row.condition_text,
                subgoal_index=class_index[row.condition_label],
                action_index=scorer.action_index(row.target_action_id),
                action_residual=_capture_one(
                    runtime=runtime,
                    text=row.action_text,
                    injection_layer_index=injection_layer_index,
                    residual_width=residual_width,
                ),
                context_residual=_capture_one(
                    runtime=runtime,
                    text=row.condition_text,
                    injection_layer_index=injection_layer_index,
                    residual_width=residual_width,
                ),
            )
        )
        if progress is not None and (
            index + 1 == len(rows) or (index + 1) % 32 == 0
        ):
            progress(f"named-action capture {split_label}: {index + 1}/{len(rows)}")
    return tuple(examples)


def _capture_one(
    *,
    runtime: OpenWeightResidualRuntime,
    text: str,
    injection_layer_index: int,
    residual_width: int,
) -> tuple[float, ...]:
    capture = runtime.capture(source_text=text)
    activations = capture.residual_activations
    if (
        len(activations) != 1
        or activations[0].layer_index != injection_layer_index
        or len(activations[0].activation) != residual_width
    ):
        raise RuntimeError(
            "named-action steering requires exactly one full-width residual at "
            f"layer {injection_layer_index}"
        )
    return tuple(float(value) for value in activations[0].activation)


def _require_fit_configuration(
    *,
    scorer: Any,
    corpus: NamedActionSteeringCorpus,
    seed: int,
    control_norm_cap_ratio: float,
    steering_rank: int,
    residual_width: int,
) -> None:
    if type(seed) is not int or seed < 0:
        raise ValueError("named-action steering seed must be a non-negative int")
    if (
        type(control_norm_cap_ratio) is not float
        or not math.isfinite(control_norm_cap_ratio)
        or not 0.0 < control_norm_cap_ratio <= 2.0
    ):
        raise ValueError(
            "named-action control_norm_cap_ratio must be a finite float in (0, 2]"
        )
    if steering_rank < 1 or steering_rank > residual_width:
        raise ValueError("named-action steering_rank must be within residual width")
    if scorer.trainable_parameters():
        raise RuntimeError("named-action steering requires a frozen substrate")
    if tuple(scorer.action_option_ids) != corpus.action_ids:
        raise ValueError("named-action scorer action surface differs from corpus")
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
        raise ValueError("named-action scorer control-norm ratio drift")


def _row_tuple(value: object, label: str) -> tuple[NamedActionSteeringRow, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    return tuple(NamedActionSteeringRow.from_payload(item) for item in value)


def _text_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    result = tuple(value)
    if any(not isinstance(item, str) for item in result):
        raise TypeError(f"{label} must contain only strings")
    return result


def _exact_mapping(
    value: object,
    *,
    expected: set[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if set(value) != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )
    return value


def _mapping_text(value: Mapping[str, object], key: str) -> str:
    item = value[key]
    if not isinstance(item, str):
        raise TypeError(f"{key} must be a string")
    return item


def _require_unique_texts(values: tuple[str, ...], label: str) -> None:
    if not values:
        raise ValueError(f"{label} must be non-empty")
    for value in values:
        _require_text(value, label)
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


def _require_text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")


def _require_sha256(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = (
    "NAMED_ACTION_STEERING_CORPUS_SCHEMA_VERSION",
    "NamedActionSteeringCorpus",
    "NamedActionSteeringRow",
    "fit_named_action_steering_artifact_bundle",
    "named_action_fit_lineage_sha256",
)
