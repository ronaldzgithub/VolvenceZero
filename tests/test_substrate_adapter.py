from __future__ import annotations

import asyncio
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from volvence_zero.runtime import Snapshot, WiringLevel, propagate
from volvence_zero.substrate import (
    apply_residual_control,
    FeatureSignal,
    FeatureSurfaceSubstrateAdapter,
    OpenWeightResidualInterventionBackend,
    OpenWeightResidualStreamSubstrateAdapter,
    PlaceholderSubstrateAdapter,
    SyntheticOpenWeightResidualRuntime,
    TraceResidualInterventionBackend,
    TransformersOpenWeightResidualRuntime,
    SimulatedResidualSubstrateAdapter,
    SubstrateModule,
    SurfaceKind,
    SubstrateFallbackMode,
    TrainingTraceDataset,
    build_builtin_transformers_runtime,
    build_training_trace,
    build_transformers_runtime_with_fallback,
)


@dataclass
class _TinyWhitespaceTokenizer:
    vocab_size: int = 64

    def __post_init__(self) -> None:
        self._token_to_id: dict[str, int] = {"<empty>": 1}
        self._id_to_token: dict[int, str] = {1: "<empty>"}

    def __call__(self, text: str, *, return_tensors: str, truncation: bool, max_length: int):
        del truncation
        if return_tensors != "pt":
            raise ValueError("Tiny test tokenizer expects return_tensors='pt'.")
        tokens = tuple(part for part in text.split() if part.strip())[:max_length] or ("<empty>",)
        token_ids = [self._resolve_token_id(token) for token in tokens]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def convert_ids_to_tokens(self, token_ids: tuple[int, ...]) -> tuple[str, ...]:
        return tuple(self._id_to_token.get(token_id, f"<tok:{token_id}>") for token_id in token_ids)

    def _resolve_token_id(self, token: str) -> int:
        existing = self._token_to_id.get(token)
        if existing is not None:
            return existing
        next_id = (len(self._token_to_id) % (self.vocab_size - 1)) + 1
        self._token_to_id[token] = next_id
        self._id_to_token[next_id] = token
        return next_id


class _BackboneContainer(nn.Module):
    def __init__(self, *, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers


class _BackboneWrappedCausalLM(nn.Module):
    def __init__(self, base: GPT2LMHeadModel) -> None:
        super().__init__()
        self._base = base
        self.backbone = _BackboneContainer(layers=base.transformer.h)
        self.config = base.config

    def forward(self, *args, **kwargs):
        return self._base(*args, **kwargs)


def _build_tiny_transformers_runtime() -> TransformersOpenWeightResidualRuntime:
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=24,
            n_layer=4,
            n_head=4,
        )
    )
    return TransformersOpenWeightResidualRuntime(
        model_id="tiny-transformers-runtime",
        model=model,
        tokenizer=_TinyWhitespaceTokenizer(),
        device="cpu",
        layer_indices=(1, 2),
        activation_width=6,
        top_k_logits=4,
    )


def _build_backbone_wrapped_runtime() -> TransformersOpenWeightResidualRuntime:
    base = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=24,
            n_layer=4,
            n_head=4,
        )
    )
    return TransformersOpenWeightResidualRuntime(
        model_id="backbone-wrapped-runtime",
        model=_BackboneWrappedCausalLM(base),
        tokenizer=_TinyWhitespaceTokenizer(),
        device="cpu",
        layer_indices=(1, 2),
        activation_width=6,
        top_k_logits=4,
    )


def test_feature_surface_adapter_publishes_current_stable_contract():
    adapter = FeatureSurfaceSubstrateAdapter(
        model_id="test-model",
        feature_surface=(
            FeatureSignal(name="semantic_state", values=(0.1, 0.9), source="adapter"),
        ),
        token_logits=(0.6, 0.4),
    )
    module = SubstrateModule(adapter=adapter, source_text="hello", wiring_level=WiringLevel.ACTIVE)

    result = asyncio.run(propagate([module], session_id="s1", wave_id="w1"))
    snapshot = result["substrate"]

    assert snapshot.value.surface_kind is SurfaceKind.FEATURE_SURFACE
    assert snapshot.value.model_id == "test-model"
    assert snapshot.value.feature_surface[0].name == "semantic_state"
    assert snapshot.value.residual_activations == ()
    assert snapshot.value.residual_sequence == ()
    assert snapshot.value.unavailable_fields[0].field_name == "residual_activations"


def test_placeholder_adapter_keeps_shape_stable_for_downstream_consumers():
    adapter = PlaceholderSubstrateAdapter(model_id="placeholder-model")
    module = SubstrateModule(adapter=adapter, wiring_level=WiringLevel.ACTIVE)

    result = asyncio.run(propagate([module], session_id="s1", wave_id="w1"))
    snapshot = result["substrate"]

    assert snapshot.value.surface_kind is SurfaceKind.PLACEHOLDER
    assert snapshot.value.token_logits == ()
    assert snapshot.value.feature_surface == ()
    assert snapshot.value.residual_sequence == ()
    assert len(snapshot.value.unavailable_fields) >= 3


def test_shadow_substrate_module_publishes_shadow_only():
    adapter = FeatureSurfaceSubstrateAdapter(
        model_id="shadow-model",
        feature_surface=(FeatureSignal(name="shadow", values=(1.0,), source="adapter"),),
    )
    module = SubstrateModule(adapter=adapter)
    shadow_snapshots: dict[str, Snapshot[object]] = {}

    result = asyncio.run(
        propagate([module], session_id="s1", wave_id="w1", shadow_snapshots=shadow_snapshots)
    )

    assert "substrate" not in result
    assert shadow_snapshots["substrate"].value.model_id == "shadow-model"


def test_simulated_residual_adapter_exposes_executable_residual_surface():
    dataset = TrainingTraceDataset()
    dataset.add_trace(build_training_trace(trace_id="trace-1", source_text="calm reflective collaboration"))
    adapter = SimulatedResidualSubstrateAdapter(trace=dataset.latest())
    snapshot = asyncio.run(adapter.capture(source_text=dataset.latest().source_text))

    assert snapshot.surface_kind is SurfaceKind.RESIDUAL_STREAM
    assert snapshot.residual_activations
    assert snapshot.feature_surface
    assert snapshot.residual_sequence
    assert snapshot.residual_sequence[-1].token


def test_residual_control_application_updates_snapshot_and_effect():
    trace = build_training_trace(trace_id="control-trace", source_text="calm reflective collaboration")
    adapter = SimulatedResidualSubstrateAdapter(trace=trace)
    snapshot = asyncio.run(adapter.capture(source_text=trace.source_text))

    applied = apply_residual_control(
        substrate_snapshot=snapshot,
        applied_control=(0.4, 0.2, 0.3),
        track_scale=(1.0, 0.5, 0.5),
    )

    assert applied.applied_snapshot.residual_sequence
    assert applied.applied_snapshot.residual_activations != snapshot.residual_activations
    assert applied.downstream_effect != (0.0, 0.0, 0.0)
    assert applied.control_energy > 0.0
    assert applied.backend_name == "trace-residual-backend"


def test_trace_residual_backend_exposes_named_intervention_contract():
    trace = build_training_trace(trace_id="backend-trace", source_text="steady guided exploration")
    adapter = SimulatedResidualSubstrateAdapter(trace=trace)
    snapshot = asyncio.run(adapter.capture(source_text=trace.source_text))
    backend = TraceResidualInterventionBackend()

    applied = backend.apply_control(
        substrate_snapshot=snapshot,
        applied_control=(0.3, 0.3, 0.2),
    )

    assert applied.backend_name == "trace-residual-backend"
    assert "trace-residual-backend" in applied.description


def test_open_weight_residual_adapter_uses_runtime_capture_contract():
    runtime = SyntheticOpenWeightResidualRuntime(model_id="synthetic-runtime")
    adapter = OpenWeightResidualStreamSubstrateAdapter(runtime=runtime)

    snapshot = asyncio.run(adapter.capture(source_text="real residual hooks later"))

    assert snapshot.model_id == "synthetic-runtime"
    assert snapshot.surface_kind is SurfaceKind.RESIDUAL_STREAM
    assert snapshot.residual_sequence


def test_open_weight_residual_backend_delegates_to_runtime():
    runtime = SyntheticOpenWeightResidualRuntime(model_id="synthetic-runtime")
    adapter = OpenWeightResidualStreamSubstrateAdapter(runtime=runtime)
    snapshot = asyncio.run(adapter.capture(source_text="real residual hooks later"))
    backend = OpenWeightResidualInterventionBackend(
        runtime=runtime,
        source_text="real residual hooks later",
    )

    applied = backend.apply_control(
        substrate_snapshot=snapshot,
        applied_control=(0.2, 0.3, 0.1),
    )

    assert applied.backend_name == "open-weight:synthetic-runtime"
    assert applied.downstream_effect != (0.0, 0.0, 0.0)


def test_transformers_open_weight_runtime_captures_real_middle_layer_hooks():
    runtime = _build_tiny_transformers_runtime()

    capture = runtime.capture(source_text="calm reflective collaboration")

    assert capture.residual_sequence
    assert capture.residual_activations
    assert len(capture.token_logits) == 4
    assert capture.feature_surface[0].source == "transformers-open-weight"
    feature_names = {signal.name for signal in capture.feature_surface}
    assert "fallback_active" in feature_names
    assert "top_logit_entropy" in feature_names
    assert "hook_layer_coverage" in feature_names
    assert "semantic_task_pull" in feature_names
    assert "semantic_support_pull" in feature_names
    assert "semantic_directive_pull" in feature_names
    assert "layers=(1, 2)" in capture.description


def test_transformers_open_weight_runtime_applies_residual_control_via_hooks():
    runtime = _build_tiny_transformers_runtime()
    adapter = OpenWeightResidualStreamSubstrateAdapter(
        runtime=runtime,
        default_source_text="calm reflective collaboration",
    )
    snapshot = asyncio.run(adapter.capture(source_text="calm reflective collaboration"))

    applied = runtime.apply_control(
        source_text="calm reflective collaboration",
        substrate_snapshot=snapshot,
        applied_control=(0.3, 0.2, 0.1),
    )

    assert applied.backend_name == "transformers-open-weight:tiny-transformers-runtime"
    assert applied.applied_snapshot.residual_sequence
    assert applied.downstream_effect != (0.0, 0.0, 0.0)
    assert applied.applied_snapshot.residual_activations != snapshot.residual_activations


def test_builtin_transformers_runtime_is_deterministic_across_instances():
    left = build_builtin_transformers_runtime(model_id="builtin-left")
    right = build_builtin_transformers_runtime(model_id="builtin-right")

    left_capture = left.capture(source_text="直接推进执行 给我明确顺序")
    right_capture = right.capture(source_text="直接推进执行 给我明确顺序")

    left_features = {signal.name: signal.values for signal in left_capture.feature_surface}
    right_features = {signal.name: signal.values for signal in right_capture.feature_surface}

    assert left_features["semantic_task_pull"] == right_features["semantic_task_pull"]
    assert left_features["semantic_support_pull"] == right_features["semantic_support_pull"]
    assert left_capture.token_logits == right_capture.token_logits


def test_transformers_runtime_resolves_backbone_layers_path():
    runtime = _build_backbone_wrapped_runtime()

    capture = runtime.capture(source_text="steady guided planning")

    assert capture.residual_sequence
    assert "family=gpt2" in capture.description


def test_build_transformers_runtime_with_fallback_honors_deny_mode():
    try:
        build_transformers_runtime_with_fallback(
            model_id="missing-local-model",
            local_files_only=True,
            fallback_mode=SubstrateFallbackMode.DENY,
        )
    except Exception as exc:
        assert type(exc).__name__ in {"OSError", "ValueError", "RuntimeError"}
    else:
        raise AssertionError("Expected missing local model to fail when fallback mode is deny.")
