"""Synthetic open-weight residual runtime.

:class:`SyntheticOpenWeightResidualRuntime` is the deterministic,
zero-dependency runtime used for tests and for development flows that
do not need a live Hugging Face model. It synthesises residual
activations from a lightweight :class:`HashingWhitespaceTokenizer`
pipeline and implements the full :class:`OpenWeightResidualRuntime`
contract (capture / intervene / generate / checkpoint / import).

Slice S.3 (2026-05-04): extracted from the previous monolithic
``residual_backend.py``.
"""

from __future__ import annotations

import importlib

import hashlib
import math
import random
from typing import Any, Sequence
from uuid import uuid4

from volvence_zero.substrate.adapter import (
    FeatureSignal,
    ResidualActivation,
    ResidualSequenceStep,
    SubstrateSnapshot,
    SurfaceKind,
    UnavailableField,
)

from volvence_zero.substrate.residual_contracts import (
    GenerationResult,
    HashingWhitespaceTokenizer,
    LocalModelCompatibilityReport,
    LocalSubstrateRuntimeMode,
    OpenWeightRuntimeCapture,
    ResidualControlApplication,
    SubstrateDeltaAdapterLayer,
    SubstrateFallbackMode,
    SubstrateOnlineFastCheckpoint,
    SubstrateRareHeavyCheckpoint,
    TrainingTrace,
)
from volvence_zero.substrate.residual_interfaces import (
    OpenWeightResidualRuntime,
    ResidualInterventionBackend,
)
from volvence_zero.substrate.residual_intervention import (
    NoOpResidualInterventionBackend,
    TraceResidualInterventionBackend,
    apply_residual_control,
)
from volvence_zero.substrate.rare_heavy_training import (
    RareHeavyAdapterTrainingBackend,
    RareHeavyTrainingRequest,
)
from volvence_zero.substrate.residual_training import build_training_trace
from volvence_zero.substrate.residual_helpers import (
    RARE_HEAVY_ANCHOR_ORDER,
    _adapter_parameter_count,
    _anchor_profile_bank,
    _build_compatibility_fingerprint,
    _checkpoint_with_adapter_payload,
    _clamp_delta_vector,
    _clamp_signed,
    _clamp_unit,
    _cosine_similarity,
    _derive_anchor_bias,
    _derive_rare_heavy_checkpoint,
    _flatten_substrate_batches,
    _hashed_semantic_embedding,
    _mean_abs_delta,
    _mean_feature_value,
    _mean_residual_magnitude,
    _mean_sequence_length,
    _normalize_semantic_weights,
    _normalize_vector,
    _normalized_entropy,
    _semantic_tokens,
    _softmax_probabilities,
    _summarize_activations,
    _summarize_real_activations,
    resolve_local_runtime_mode,
    resolve_substrate_fallback_mode,
)


class SyntheticOpenWeightResidualRuntime(OpenWeightResidualRuntime):
    """Hook-shaped frozen runtime backed by the existing residual simulator."""

    def __init__(
        self,
        *,
        model_id: str = "synthetic-open-weight-runtime",
        allow_live_substrate_mutation: bool = False,
        allow_offline_substrate_training: bool = False,
    ) -> None:
        self.model_id = model_id
        self.is_frozen = True
        self.runtime_origin = "synthetic-open-weight"
        self.supports_live_substrate_mutation = allow_live_substrate_mutation
        self.supports_offline_substrate_training = allow_offline_substrate_training
        self._rare_heavy_control_scale = 0.12
        self._rare_heavy_semantic_text_weight = 0.85
        self._rare_heavy_semantic_residual_weight = 0.15
        self._rare_heavy_anchor_bias = tuple(0.0 for _ in RARE_HEAVY_ANCHOR_ORDER)
        self._rare_heavy_update_count = 0
        self._rare_heavy_adapter_scale = 0.0
        self._rare_heavy_adapter_layers: dict[int, tuple[float, ...]] = {}
        self._rare_heavy_activation_width = 0
        self._rare_heavy_layer_indices: tuple[int, ...] = ()
        self._online_fast_delta_scale = 0.0
        self._online_fast_update_count = 0
        self._online_fast_optimizer_state_norm = 0.0
        self._online_fast_parameter_change_rate = 0.0
        self._online_fast_adapter_layers: dict[int, tuple[float, ...]] = {}
        self._online_fast_state_hash = ""
        self._online_fast_source_state_hash = ""
        self._online_fast_signal: tuple[float, ...] = ()
        self._online_fast_optimizer_state_description = ""
        # S1: injectable real rare-heavy training backend. None -> the
        # built-in heuristic/adapter-delta path stays the documented fallback.
        self._rare_heavy_training_backend: RareHeavyAdapterTrainingBackend | None = None

    def set_rare_heavy_training_backend(
        self, backend: RareHeavyAdapterTrainingBackend | None
    ) -> None:
        """Install (or clear) the offline rare-heavy training backend.

        Mirrors the transformers runtime seam so wiring/contract tests can
        exercise the delegation path without the HF stack. An injected
        backend that fails must fail loudly; clearing the backend restores
        the built-in path (R15 rollback).
        """

        self._rare_heavy_training_backend = backend

    def capture(self, *, source_text: str) -> OpenWeightRuntimeCapture:
        trace = build_training_trace(trace_id=f"{self.model_id}:capture", source_text=source_text)
        self._remember_trace_shape(trace)
        latest_step = trace.steps[-1]
        adapted_residuals = self._apply_adapter_to_residuals(latest_step.residual_activations)
        feature_surface = latest_step.feature_surface + (
            FeatureSignal(
                name="semantic_text_weight",
                values=(self._rare_heavy_semantic_text_weight,),
                source="synthetic-open-weight-semantic",
            ),
            FeatureSignal(
                name="semantic_residual_weight",
                values=(self._rare_heavy_semantic_residual_weight,),
                source="synthetic-open-weight-semantic",
            ),
            FeatureSignal(
                name="substrate_rare_heavy_update_count",
                values=(_clamp_unit(self._rare_heavy_update_count / 10.0),),
                source="synthetic-open-weight-semantic",
            ),
            FeatureSignal(
                name="substrate_delta_parameter_count",
                values=(_clamp_unit(_adapter_parameter_count(self._export_adapter_layers()) / 64.0),),
                source="synthetic-open-weight-semantic",
            ),
            FeatureSignal(
                name="substrate_online_fast_update_count",
                values=(_clamp_unit(self._online_fast_update_count / 10.0),),
                source="synthetic-open-weight-semantic",
            ),
            FeatureSignal(
                name="substrate_online_fast_delta_parameter_count",
                values=(_clamp_unit(_adapter_parameter_count(self._export_online_fast_layers()) / 64.0),),
                source="synthetic-open-weight-semantic",
            ),
            FeatureSignal(
                name="substrate_online_fast_parameter_change_rate",
                values=(_clamp_unit(self._online_fast_parameter_change_rate),),
                source="synthetic-open-weight-semantic",
            ),
            FeatureSignal(
                name="substrate_online_fast_experimental_mode",
                values=(1.0 if self.experimental_live_mutation_enabled else 0.0,),
                source="synthetic-open-weight-semantic",
            ),
        )
        residual_sequence = tuple(
            ResidualSequenceStep(
                step=step.step,
                token=step.token,
                feature_surface=step.feature_surface,
                residual_activations=self._apply_adapter_to_residuals(step.residual_activations),
                description=f"Synthetic hook token '{step.token}' at step {step.step}.",
            )
            for step in trace.steps
        )
        return OpenWeightRuntimeCapture(
            token_logits=tuple(
                min(sum(feature.values) / max(len(feature.values), 1), 1.0)
                for feature in feature_surface
            ),
            feature_surface=feature_surface,
            residual_activations=adapted_residuals,
            residual_sequence=residual_sequence,
            description=(
                f"Synthetic frozen open-weight capture for len={len(source_text)} "
                f"rare_heavy_updates={self._rare_heavy_update_count} "
                f"adapter_params={_adapter_parameter_count(self._export_adapter_layers())} "
                f"online_fast_updates={self._online_fast_update_count} "
                f"online_fast_params={_adapter_parameter_count(self._export_online_fast_layers())} "
                f"live_mode={self.live_mutation_mode}."
            ),
        )

    def apply_control(
        self,
        *,
        source_text: str,
        substrate_snapshot: SubstrateSnapshot,
        applied_control: tuple[float, ...],
        track_scale: tuple[float, ...] = (1.0, 1.0, 1.0),
    ) -> ResidualControlApplication:
        del source_text
        scale_ratio = max(0.25, min(2.0, self._rare_heavy_control_scale / 0.12))
        scaled_control = tuple(value * scale_ratio for value in applied_control)
        result = TraceResidualInterventionBackend().apply_control(
            substrate_snapshot=substrate_snapshot,
            applied_control=scaled_control,
            track_scale=track_scale,
        )
        return ResidualControlApplication(
            applied_snapshot=result.applied_snapshot,
            downstream_effect=result.downstream_effect,
            control_energy=result.control_energy,
            backend_name=f"synthetic-open-weight:{self.model_id}",
            description=(
                f"Synthetic open-weight runtime delegated residual intervention. "
                f"{result.description}"
            ),
        )

    def export_rare_heavy_state(self, *, checkpoint_id: str | None = None) -> SubstrateRareHeavyCheckpoint:
        adapter_layers = self._export_adapter_layers()
        hidden_size = self._rare_heavy_activation_width
        layer_indices = self._rare_heavy_layer_indices
        training_mode = "adapter-delta-v2" if adapter_layers else "bounded-state-v1"
        return SubstrateRareHeavyCheckpoint(
            checkpoint_id=checkpoint_id or f"{self.model_id}:rare-heavy",
            model_id=self.model_id,
            runtime_origin=self.runtime_origin,
            control_scale=self._rare_heavy_control_scale,
            semantic_text_weight=self._rare_heavy_semantic_text_weight,
            semantic_residual_weight=self._rare_heavy_semantic_residual_weight,
            semantic_anchor_bias=self._rare_heavy_anchor_bias,
            update_count=self._rare_heavy_update_count,
            source_batch_count=0,
            mean_sequence_length=0.0,
            mean_residual_magnitude=0.0,
            description=(
                f"Synthetic rare-heavy checkpoint for {self.model_id} "
                f"updates={self._rare_heavy_update_count}."
            ),
            checkpoint_version=2 if adapter_layers else 1,
            training_mode=training_mode,
            compatibility_fingerprint=(
                _build_compatibility_fingerprint(
                    model_id=self.model_id,
                    runtime_origin=self.runtime_origin,
                    hidden_size=hidden_size,
                    layer_indices=layer_indices,
                    training_mode=training_mode,
                )
                if hidden_size > 0 and layer_indices
                else ""
            ),
            adapter_scale=self._rare_heavy_adapter_scale,
            adapter_parameter_count=_adapter_parameter_count(adapter_layers),
            adapter_training_loss=0.0,
            adapter_layers=adapter_layers,
        )

    def import_rare_heavy_state(self, checkpoint: SubstrateRareHeavyCheckpoint) -> tuple[str, ...]:
        self.require_substrate_artifact_import(operation="import_rare_heavy_state()")
        if checkpoint.model_id != self.model_id:
            raise ValueError(
                f"Synthetic runtime {self.model_id!r} cannot import checkpoint for {checkpoint.model_id!r}."
            )
        text_weight, residual_weight = _normalize_semantic_weights(
            text_weight=checkpoint.semantic_text_weight,
            residual_weight=checkpoint.semantic_residual_weight,
        )
        self._rare_heavy_control_scale = max(0.04, min(0.30, checkpoint.control_scale))
        self._rare_heavy_semantic_text_weight = text_weight
        self._rare_heavy_semantic_residual_weight = residual_weight
        self._rare_heavy_anchor_bias = tuple(
            max(-0.2, min(0.2, value))
            for value in checkpoint.semantic_anchor_bias[: len(RARE_HEAVY_ANCHOR_ORDER)]
        )
        if len(self._rare_heavy_anchor_bias) < len(RARE_HEAVY_ANCHOR_ORDER):
            self._rare_heavy_anchor_bias = self._rare_heavy_anchor_bias + tuple(
                0.0 for _ in range(len(RARE_HEAVY_ANCHOR_ORDER) - len(self._rare_heavy_anchor_bias))
            )
        self._rare_heavy_update_count = max(0, checkpoint.update_count)
        self._rare_heavy_adapter_scale = max(0.0, checkpoint.adapter_scale)
        self._rare_heavy_adapter_layers = {
            layer.layer_index: _clamp_delta_vector(layer.delta_vector)
            for layer in checkpoint.adapter_layers
        }
        if checkpoint.adapter_layers:
            self._rare_heavy_activation_width = len(checkpoint.adapter_layers[0].delta_vector)
            self._rare_heavy_layer_indices = tuple(layer.layer_index for layer in checkpoint.adapter_layers)
        return ("rare-heavy:substrate-import",)

    def restore_rare_heavy_state(self, checkpoint: SubstrateRareHeavyCheckpoint) -> tuple[str, ...]:
        self.import_rare_heavy_state(checkpoint)
        return ("rare-heavy:substrate-rollback",)

    def train_rare_heavy(
        self,
        *,
        traces: tuple[TrainingTrace, ...] = (),
        substrate_steps_per_trace: tuple[tuple[SubstrateSnapshot, ...], ...],
        checkpoint_id: str | None = None,
    ) -> SubstrateRareHeavyCheckpoint:
        self.require_offline_substrate_training(operation="train_rare_heavy()")
        checkpoint = _derive_rare_heavy_checkpoint(
            checkpoint_id=checkpoint_id or f"{self.model_id}:rare-heavy-trained",
            model_id=self.model_id,
            runtime_origin=self.runtime_origin,
            current_control_scale=self._rare_heavy_control_scale,
            default_text_weight=self._rare_heavy_semantic_text_weight,
            default_residual_weight=self._rare_heavy_semantic_residual_weight,
            previous_update_count=self._rare_heavy_update_count,
            substrate_steps_per_trace=substrate_steps_per_trace,
        )
        backend = self._rare_heavy_training_backend
        if backend is not None:
            for trace in traces:
                if self._rare_heavy_activation_width > 0 and self._rare_heavy_layer_indices:
                    break
                self._remember_trace_shape(trace)
            if self._rare_heavy_activation_width <= 0 or not self._rare_heavy_layer_indices:
                raise ValueError(
                    "Synthetic runtime cannot delegate rare-heavy training: no "
                    "residual shape observed (provide traces with residual "
                    "activations or capture at least once before training)."
                )
            result = backend.train(
                RareHeavyTrainingRequest(
                    model_id=self.model_id,
                    hidden_size=self._rare_heavy_activation_width,
                    layer_indices=self._rare_heavy_layer_indices,
                    device="cpu",
                    traces=traces,
                )
            )
            return _checkpoint_with_adapter_payload(
                checkpoint,
                training_mode=result.training_mode,
                compatibility_fingerprint=_build_compatibility_fingerprint(
                    model_id=self.model_id,
                    runtime_origin=self.runtime_origin,
                    hidden_size=self._rare_heavy_activation_width,
                    layer_indices=self._rare_heavy_layer_indices,
                    training_mode=result.training_mode,
                ),
                adapter_scale=1.0,
                adapter_training_loss=result.training_loss,
                adapter_layers=result.adapter_layers,
                description=f"{checkpoint.description} {result.description}",
            )
        adapter_layers, training_loss = self._train_adapter_deltas(
            traces=traces,
            substrate_steps_per_trace=substrate_steps_per_trace,
        )
        if not adapter_layers:
            return checkpoint
        return _checkpoint_with_adapter_payload(
            checkpoint,
            training_mode="adapter-delta-v2",
            compatibility_fingerprint=_build_compatibility_fingerprint(
                model_id=self.model_id,
                runtime_origin=self.runtime_origin,
                hidden_size=self._rare_heavy_activation_width,
                layer_indices=self._rare_heavy_layer_indices,
                training_mode="adapter-delta-v2",
            ),
            adapter_scale=1.0,
            adapter_training_loss=training_loss,
            adapter_layers=adapter_layers,
            description=(
                f"{checkpoint.description} Synthetic adapter-delta payload "
                f"layers={len(adapter_layers)} loss={training_loss:.4f}."
            ),
        )

    def clone_for_rare_heavy(self) -> "OpenWeightResidualRuntime":
        cloned = SyntheticOpenWeightResidualRuntime(
            model_id=self.model_id,
            allow_offline_substrate_training=True,
        )
        cloned.set_rare_heavy_training_backend(self._rare_heavy_training_backend)
        cloned.import_rare_heavy_state(self.export_rare_heavy_state())
        return cloned

    def export_online_fast_state(
        self,
        *,
        checkpoint_id: str | None = None,
    ) -> SubstrateOnlineFastCheckpoint:
        self.require_live_substrate_mutation(operation="export_online_fast_state()")
        adapter_layers = self._export_online_fast_layers()
        return SubstrateOnlineFastCheckpoint(
            checkpoint_id=checkpoint_id or f"{self.model_id}:online-fast",
            model_id=self.model_id,
            runtime_origin=self.runtime_origin,
            delta_scale=self._online_fast_delta_scale,
            update_count=self._online_fast_update_count,
            source_wave_id="runtime",
            source_turn_index=self._online_fast_update_count,
            gate="online",
            optimizer_state_norm=self._online_fast_optimizer_state_norm,
            parameter_change_rate=self._online_fast_parameter_change_rate,
            description=(
                f"Synthetic online-fast checkpoint for {self.model_id} "
                f"updates={self._online_fast_update_count}."
            ),
            compatibility_fingerprint=(
                _build_compatibility_fingerprint(
                    model_id=self.model_id,
                    runtime_origin=self.runtime_origin,
                    hidden_size=self._rare_heavy_activation_width,
                    layer_indices=self._rare_heavy_layer_indices,
                    training_mode="online-fast-delta-v1",
                )
                if self._rare_heavy_activation_width > 0 and self._rare_heavy_layer_indices
                else ""
            ),
            adapter_parameter_count=_adapter_parameter_count(adapter_layers),
            adapter_layers=adapter_layers,
            fast_state_hash=self._online_fast_state_hash,
            source_fast_state_hash=self._online_fast_source_state_hash,
            fast_memory_signal=self._online_fast_signal,
            optimizer_state_description=self._online_fast_optimizer_state_description,
        )

    def apply_online_fast_state(self, checkpoint: SubstrateOnlineFastCheckpoint) -> tuple[str, ...]:
        self.require_live_substrate_mutation(operation="apply_online_fast_state()")
        if checkpoint.model_id != self.model_id:
            raise ValueError(
                f"Synthetic runtime {self.model_id!r} cannot import online-fast checkpoint for "
                f"{checkpoint.model_id!r}."
            )
        self._online_fast_delta_scale = max(0.0, min(0.18, checkpoint.delta_scale))
        self._online_fast_update_count = max(0, checkpoint.update_count)
        self._online_fast_optimizer_state_norm = _clamp_unit(checkpoint.optimizer_state_norm)
        self._online_fast_parameter_change_rate = _clamp_unit(checkpoint.parameter_change_rate)
        self._online_fast_state_hash = checkpoint.fast_state_hash
        self._online_fast_source_state_hash = checkpoint.source_fast_state_hash
        self._online_fast_signal = checkpoint.fast_memory_signal
        self._online_fast_optimizer_state_description = checkpoint.optimizer_state_description
        self._online_fast_adapter_layers = {
            layer.layer_index: _clamp_delta_vector(layer.delta_vector, limit=0.12)
            for layer in checkpoint.adapter_layers
        }
        if checkpoint.adapter_layers:
            self._rare_heavy_activation_width = len(checkpoint.adapter_layers[0].delta_vector)
            self._rare_heavy_layer_indices = tuple(layer.layer_index for layer in checkpoint.adapter_layers)
        return ("online-fast:substrate-import",)

    def restore_online_fast_state(self, checkpoint: SubstrateOnlineFastCheckpoint) -> tuple[str, ...]:
        self.apply_online_fast_state(checkpoint)
        return ("online-fast:substrate-rollback",)

    def _remember_trace_shape(self, trace: TrainingTrace) -> None:
        layer_indices = tuple(
            sorted(
                {
                    activation.layer_index
                    for step in trace.steps
                    for activation in step.residual_activations
                }
            )
        )
        if layer_indices:
            self._rare_heavy_layer_indices = layer_indices
        for step in trace.steps:
            for activation in step.residual_activations:
                if activation.activation:
                    self._rare_heavy_activation_width = len(activation.activation)
                    return

    def _apply_adapter_to_residuals(
        self,
        residual_activations: tuple[ResidualActivation, ...],
    ) -> tuple[ResidualActivation, ...]:
        if (
            not self._rare_heavy_adapter_layers
            and not self._online_fast_adapter_layers
        ) or (
            self._rare_heavy_adapter_scale <= 0.0
            and self._online_fast_delta_scale <= 0.0
        ):
            return residual_activations
        adapted: list[ResidualActivation] = []
        for activation in residual_activations:
            rare_heavy_delta = self._rare_heavy_adapter_layers.get(activation.layer_index)
            online_fast_delta = self._online_fast_adapter_layers.get(activation.layer_index)
            if (rare_heavy_delta is None and online_fast_delta is None) or not activation.activation:
                adapted.append(activation)
                continue
            values = tuple(
                _clamp_signed(
                    base
                    + (
                        (rare_heavy_delta[index] * self._rare_heavy_adapter_scale)
                        if rare_heavy_delta is not None
                        else 0.0
                    )
                    + (
                        (online_fast_delta[index] * self._online_fast_delta_scale)
                        if online_fast_delta is not None
                        else 0.0
                    )
                )
                for index, base in enumerate(activation.activation)
            )
            adapted.append(
                ResidualActivation(
                    layer_index=activation.layer_index,
                    activation=values,
                    step=activation.step,
                )
            )
        return tuple(adapted)

    def _export_adapter_layers(self) -> tuple[SubstrateDeltaAdapterLayer, ...]:
        return tuple(
            SubstrateDeltaAdapterLayer(
                layer_index=layer_index,
                delta_vector=delta_vector,
                mean_abs_delta=_mean_abs_delta(delta_vector),
                description=(
                    f"Synthetic adapter delta for layer {layer_index} "
                    f"width={len(delta_vector)}."
                ),
            )
            for layer_index, delta_vector in sorted(self._rare_heavy_adapter_layers.items())
        )

    def _export_online_fast_layers(self) -> tuple[SubstrateDeltaAdapterLayer, ...]:
        return tuple(
            SubstrateDeltaAdapterLayer(
                layer_index=layer_index,
                delta_vector=delta_vector,
                mean_abs_delta=_mean_abs_delta(delta_vector),
                description=f"Synthetic online-fast delta for layer {layer_index}.",
            )
            for layer_index, delta_vector in sorted(self._online_fast_adapter_layers.items())
        )

    def _train_adapter_deltas(
        self,
        *,
        traces: tuple[TrainingTrace, ...],
        substrate_steps_per_trace: tuple[tuple[SubstrateSnapshot, ...], ...],
    ) -> tuple[tuple[SubstrateDeltaAdapterLayer, ...], float]:
        if not traces or not substrate_steps_per_trace:
            return ((), 0.0)
        torch = importlib.import_module("torch")
        target_vectors = self._target_layer_vectors(substrate_steps_per_trace=substrate_steps_per_trace)
        base_vectors = self._trace_layer_vectors(traces=traces)
        common_layers = tuple(layer for layer in self._rare_heavy_layer_indices if layer in target_vectors and layer in base_vectors)
        if not common_layers or self._rare_heavy_activation_width <= 0:
            return ((), 0.0)
        parameters = {
            layer: torch.tensor(
                self._rare_heavy_adapter_layers.get(
                    layer,
                    tuple(0.0 for _ in range(self._rare_heavy_activation_width)),
                ),
                dtype=torch.float32,
                requires_grad=True,
            )
            for layer in common_layers
        }
        optimizer = torch.optim.Adam(tuple(parameters.values()), lr=0.08)
        final_loss = 0.0
        for _ in range(12):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, dtype=torch.float32)
            for layer in common_layers:
                base = torch.tensor(base_vectors[layer], dtype=torch.float32)
                target = torch.tensor(target_vectors[layer], dtype=torch.float32)
                predicted = base + parameters[layer]
                total_loss = total_loss + torch.mean((predicted - target) ** 2)
                total_loss = total_loss + torch.mean(parameters[layer] ** 2) * 0.01
            total_loss.backward()
            optimizer.step()
            final_loss = float(total_loss.detach().item())
        adapter_layers = tuple(
            SubstrateDeltaAdapterLayer(
                layer_index=layer,
                delta_vector=_clamp_delta_vector(parameters[layer].detach().tolist(), limit=0.18),
                mean_abs_delta=_mean_abs_delta(parameters[layer].detach().tolist()),
                description=f"Synthetic trained adapter delta for layer {layer}.",
            )
            for layer in common_layers
        )
        return (adapter_layers, final_loss)

    def _target_layer_vectors(
        self,
        *,
        substrate_steps_per_trace: tuple[tuple[SubstrateSnapshot, ...], ...],
    ) -> dict[int, tuple[float, ...]]:
        values: dict[int, list[tuple[float, ...]]] = {}
        for batch in substrate_steps_per_trace:
            for snapshot in batch:
                for activation in snapshot.residual_activations:
                    if activation.activation:
                        values.setdefault(activation.layer_index, []).append(activation.activation)
        return {
            layer: tuple(
                sum(vector[index] for vector in vectors) / len(vectors)
                for index in range(len(vectors[0]))
            )
            for layer, vectors in values.items()
            if vectors
        }

    def _trace_layer_vectors(
        self,
        *,
        traces: tuple[TrainingTrace, ...],
    ) -> dict[int, tuple[float, ...]]:
        values: dict[int, list[tuple[float, ...]]] = {}
        for trace in traces:
            self._remember_trace_shape(trace)
            for step in trace.steps:
                for activation in step.residual_activations:
                    if activation.activation:
                        values.setdefault(activation.layer_index, []).append(activation.activation)
        return {
            layer: tuple(
                sum(vector[index] for vector in vectors) / len(vectors)
                for index in range(len(vectors[0]))
            )
            for layer, vectors in values.items()
            if vectors
        }

