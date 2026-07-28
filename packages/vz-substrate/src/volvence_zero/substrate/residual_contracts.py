"""Substrate residual-backend contract surface.

Pure data-only module. Holds the frozen dataclasses, Enums, and the
``HashingWhitespaceTokenizer`` fallback tokenizer consumed by both the
synthetic and Hugging Face runtimes plus the intervention backends.

Slice S.3 (2026-05-04): extracted from the previous monolithic
``residual_backend.py``. External consumers import these names via
``volvence_zero.substrate`` facade unchanged.
"""

from __future__ import annotations

import importlib
import math

from dataclasses import dataclass, field
from enum import Enum

from volvence_zero.substrate.adapter import (
    FeatureSignal,
    ResidualActivation,
    ResidualSequenceStep,
    SubstrateSnapshot,
)


@dataclass(frozen=True)
class ExpertActionTarget:
    """Environment-published action target for offline expert imitation.

    The environment owns ``action_id`` semantics. This passive trace contract
    carries only a bounded numeric target and provenance; beta boundaries,
    subgoal completion, reward, and evaluation labels are intentionally absent.
    """

    action_id: str
    values: tuple[float, ...]
    source: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.action_id.strip():
            raise ValueError("ExpertActionTarget.action_id must be nonempty.")
        if not self.values or not all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in self.values
        ):
            raise ValueError(
                "ExpertActionTarget.values must be nonempty finite floats "
                "within [0, 1]."
            )
        if not self.source.strip():
            raise ValueError("ExpertActionTarget.source must be nonempty.")


@dataclass(frozen=True)
class TraceStep:
    step: int
    token: str
    feature_surface: tuple[FeatureSignal, ...]
    residual_activations: tuple[ResidualActivation, ...]
    expert_action_target: ExpertActionTarget | None = None


@dataclass(frozen=True)
class TrainingTrace:
    trace_id: str
    source_text: str
    steps: tuple[TraceStep, ...]


@dataclass(frozen=True)
class ResidualControlApplication:
    applied_snapshot: SubstrateSnapshot
    downstream_effect: tuple[float, ...]
    control_energy: float
    backend_name: str
    description: str


@dataclass(frozen=True)
class ContinuationScore:
    source_text: str
    continuation_text: str
    token_count: int
    mean_negative_log_likelihood: float
    geometric_mean_probability: float
    applied_control: tuple[float, ...]
    backend_name: str
    description: str


@dataclass(frozen=True)
class OpenWeightRuntimeCapture:
    token_logits: tuple[float, ...]
    feature_surface: tuple[FeatureSignal, ...]
    residual_activations: tuple[ResidualActivation, ...]
    residual_sequence: tuple[ResidualSequenceStep, ...]
    description: str
    personal_conditioning_applied: bool = False


@dataclass(frozen=True)
class GenerationResult:
    """Result of a runtime ``generate()`` call.

    ``personal_conditioning_applied`` is the audit source of truth for
    whether the runtime actually injected the personal conditioning
    delta into the residual stream during this generation. Consumers
    must read this flag instead of inferring injection from having
    passed a snapshot: a runtime may legitimately receive a snapshot
    and not inject (e.g. trace-only synthetic runtime, or a
    zero-confidence snapshot filtered at the substrate boundary).
    """

    text: str
    token_count: int
    capture: OpenWeightRuntimeCapture | None
    description: str
    personal_conditioning_applied: bool = False


@dataclass(frozen=True)
class HookLayerCalibrationCase:
    layer_indices: tuple[int, ...]
    hook_layer_coverage: float
    residual_sequence_length: int
    semantic_separation: float
    signal_quality: float
    runtime_origin: str
    description: str


@dataclass(frozen=True)
class HookLayerCalibrationReport:
    model_id: str
    source_text: str
    cases: tuple[HookLayerCalibrationCase, ...]
    recommended_layers: tuple[int, ...]
    description: str


@dataclass(frozen=True)
class LocalModelCompatibilityReport:
    model_id: str
    local_tokenizer_available: bool
    local_model_available: bool
    strict_local_runtime_available: bool
    error_type: str | None
    error_message: str
    description: str


@dataclass(frozen=True)
class SubstrateDeltaAdapterLayer:
    layer_index: int
    delta_vector: tuple[float, ...]
    mean_abs_delta: float
    description: str


@dataclass(frozen=True)
class SubstrateRareHeavyCheckpoint:
    checkpoint_id: str
    model_id: str
    runtime_origin: str
    control_scale: float
    semantic_text_weight: float
    semantic_residual_weight: float
    semantic_anchor_bias: tuple[float, ...]
    update_count: int
    source_batch_count: int
    mean_sequence_length: float
    mean_residual_magnitude: float
    description: str
    checkpoint_version: int = 1
    training_mode: str = "bounded-state-v1"
    compatibility_fingerprint: str = ""
    adapter_scale: float = 0.0
    adapter_parameter_count: int = 0
    adapter_training_loss: float = 0.0
    adapter_layers: tuple[SubstrateDeltaAdapterLayer, ...] = ()


@dataclass(frozen=True)
class SubstrateOnlineFastCheckpoint:
    checkpoint_id: str
    model_id: str
    runtime_origin: str
    delta_scale: float
    update_count: int
    source_wave_id: str
    source_turn_index: int
    gate: str
    optimizer_state_norm: float
    parameter_change_rate: float
    description: str
    checkpoint_version: int = 1
    training_mode: str = "online-fast-delta-v1"
    compatibility_fingerprint: str = ""
    adapter_parameter_count: int = 0
    adapter_layers: tuple[SubstrateDeltaAdapterLayer, ...] = ()
    fast_state_hash: str = ""
    source_fast_state_hash: str = ""
    fast_memory_signal: tuple[float, ...] = ()
    optimizer_state_description: str = ""


class SubstrateFallbackMode(str, Enum):
    ALLOW_BUILTIN = "allow-builtin"
    DENY = "deny"


class LocalSubstrateRuntimeMode(str, Enum):
    STRICT_LOCAL = "strict-local"
    PREFER_LOCAL = "prefer-local"
    BUILTIN_ONLY = "builtin-only"


@dataclass
class HashingWhitespaceTokenizer:
    """Minimal local tokenizer for bundled tiny transformers runtimes."""

    vocab_size: int = 256
    _token_to_id: dict[str, int] = field(default_factory=lambda: {"<empty>": 1}, init=False, repr=False)
    _id_to_token: dict[int, str] = field(default_factory=lambda: {1: "<empty>"}, init=False, repr=False)

    def __call__(self, text: str, *, return_tensors: str, truncation: bool, max_length: int):
        del truncation
        if return_tensors != "pt":
            raise ValueError("HashingWhitespaceTokenizer expects return_tensors='pt'.")
        torch = importlib.import_module("torch")
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
        next_id = (len(self._token_to_id) % max(self.vocab_size - 1, 1)) + 1
        self._token_to_id[token] = next_id
        self._id_to_token[next_id] = token
        return next_id
