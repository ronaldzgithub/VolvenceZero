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
import hashlib
import json
import math

from dataclasses import dataclass, field
from enum import Enum

from volvence_zero.substrate.adapter import (
    FeatureSignal,
    ResidualActivation,
    ResidualSequenceStep,
    SubstrateSnapshot,
)


TRANSFORMERS_EXECUTION_ATTESTATION_SCHEMA_VERSION = (
    "transformers-execution-attestation.v1"
)
GENERATION_CONTEXT_BUDGET_ATTESTATION_SCHEMA_VERSION = (
    "generation-context-budget-attestation.v1"
)


@dataclass(frozen=True)
class TransformersExecutionProfile:
    """Frozen, content-addressed execution requirements for one HF lane."""

    preset_name: str
    platform_system: str
    device_type: str
    attention_implementation: str
    sdpa_backend: str
    sdpa_backend_policy: str
    sdpa_backend_exclusive: bool
    generation_use_cache: bool
    generation_capture_strategy: str
    capture_failure_mode: str
    context_window_tokens: int
    local_files_only: bool
    fallback_mode: str
    fail_on_truncation: bool
    model_dtype: str
    require_verified_model_revision: bool
    require_model_weights_sha256: bool
    require_execution_assets_sha256: bool
    require_generation_chat_template: bool

    def __post_init__(self) -> None:
        for name in (
            "preset_name",
            "platform_system",
            "device_type",
            "attention_implementation",
            "sdpa_backend",
            "sdpa_backend_policy",
            "generation_capture_strategy",
            "capture_failure_mode",
            "fallback_mode",
            "model_dtype",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"TransformersExecutionProfile.{name} must be nonempty")
        for name in (
            "sdpa_backend_exclusive",
            "generation_use_cache",
            "local_files_only",
            "fail_on_truncation",
            "require_verified_model_revision",
            "require_model_weights_sha256",
            "require_execution_assets_sha256",
            "require_generation_chat_template",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(
                    f"TransformersExecutionProfile.{name} must be an exact bool"
                )
        if (
            type(self.context_window_tokens) is not int
            or self.context_window_tokens <= 0
        ):
            raise ValueError(
                "TransformersExecutionProfile.context_window_tokens must be positive"
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "preset_name": self.preset_name,
            "platform_system": self.platform_system,
            "device_type": self.device_type,
            "attention_implementation": self.attention_implementation,
            "sdpa_backend": self.sdpa_backend,
            "sdpa_backend_policy": self.sdpa_backend_policy,
            "sdpa_backend_exclusive": self.sdpa_backend_exclusive,
            "generation_use_cache": self.generation_use_cache,
            "generation_capture_strategy": self.generation_capture_strategy,
            "capture_failure_mode": self.capture_failure_mode,
            "context_window_tokens": self.context_window_tokens,
            "local_files_only": self.local_files_only,
            "fallback_mode": self.fallback_mode,
            "fail_on_truncation": self.fail_on_truncation,
            "model_dtype": self.model_dtype,
            "require_verified_model_revision": self.require_verified_model_revision,
            "require_model_weights_sha256": self.require_model_weights_sha256,
            "require_execution_assets_sha256": (
                self.require_execution_assets_sha256
            ),
            "require_generation_chat_template": (
                self.require_generation_chat_template
            ),
        }

    @property
    def profile_id(self) -> str:
        return _content_sha256(self.to_payload())


WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1 = TransformersExecutionProfile(
    preset_name="windows-cuda-cudnn-sdpa-cached-strict.v1",
    platform_system="Windows",
    device_type="cuda",
    attention_implementation="sdpa",
    sdpa_backend="cudnn",
    sdpa_backend_policy="exclusive-cudnn",
    sdpa_backend_exclusive=True,
    generation_use_cache=True,
    generation_capture_strategy="first-full-prompt-set-once",
    capture_failure_mode="raise",
    context_window_tokens=32768,
    local_files_only=True,
    fallback_mode="deny",
    fail_on_truncation=True,
    model_dtype="bfloat16",
    require_verified_model_revision=True,
    require_model_weights_sha256=True,
    require_execution_assets_sha256=True,
    require_generation_chat_template=True,
)


@dataclass(frozen=True)
class TransformersExecutionAttestation:
    """Content-addressed facts for one loaded strict transformers runtime."""

    profile_id: str
    preset_name: str
    model_id: str
    model_revision: str
    model_weights_sha256: str
    execution_assets_sha256: str
    runtime_origin: str
    platform_system: str
    platform_release: str
    device: str
    device_name: str
    python_version: str
    torch_version: str
    transformers_version: str
    cuda_version: str
    cudnn_version: int
    device_compute_capability: tuple[int, int]
    attention_implementation: str
    sdpa_backend: str
    sdpa_backend_policy: str
    sdpa_backend_exclusive: bool
    generation_use_cache: bool
    require_generation_chat_template: bool
    generation_capture_strategy: str
    capture_failure_mode: str
    context_window_tokens: int
    local_files_only: bool
    fallback_mode: str
    fail_on_truncation: bool
    model_dtype: str
    hidden_size: int
    model_max_position_embeddings: int
    hook_layer_indices: tuple[int, ...]
    schema_version: str = TRANSFORMERS_EXECUTION_ATTESTATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version
            != TRANSFORMERS_EXECUTION_ATTESTATION_SCHEMA_VERSION
        ):
            raise ValueError("TransformersExecutionAttestation schema drift")
        _require_sha256_text(self.profile_id, "profile_id")
        _require_sha256_text(self.model_weights_sha256, "model_weights_sha256")
        _require_sha256_text(
            self.execution_assets_sha256, "execution_assets_sha256"
        )
        for name in (
            "preset_name",
            "model_id",
            "model_revision",
            "runtime_origin",
            "platform_system",
            "platform_release",
            "device",
            "device_name",
            "python_version",
            "torch_version",
            "transformers_version",
            "cuda_version",
            "attention_implementation",
            "sdpa_backend",
            "sdpa_backend_policy",
            "generation_capture_strategy",
            "capture_failure_mode",
            "fallback_mode",
            "model_dtype",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(
                    f"TransformersExecutionAttestation.{name} must be nonempty"
                )
        if not _is_model_revision(self.model_revision):
            raise ValueError(
                "TransformersExecutionAttestation.model_revision must be a "
                "40- or 64-character lowercase hexadecimal revision"
            )
        for name in (
            "sdpa_backend_exclusive",
            "generation_use_cache",
            "require_generation_chat_template",
            "local_files_only",
            "fail_on_truncation",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(
                    f"TransformersExecutionAttestation.{name} must be an exact bool"
                )
        for name in (
            "cudnn_version",
            "context_window_tokens",
            "hidden_size",
            "model_max_position_embeddings",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(
                    f"TransformersExecutionAttestation.{name} must be a positive int"
                )
        if type(self.device_compute_capability) is not tuple or (
            len(self.device_compute_capability) != 2
        ):
            raise TypeError(
                "Transformers execution compute capability must be an exact pair"
            )
        if any(
            type(value) is not int or value < 0
            for value in self.device_compute_capability
        ):
            raise ValueError("Transformers execution compute capability is invalid")
        if type(self.hook_layer_indices) is not tuple:
            raise TypeError("Transformers execution hook layers must be an exact tuple")
        if not self.hook_layer_indices or any(
            type(index) is not int or index < 0 for index in self.hook_layer_indices
        ):
            raise ValueError("Transformers execution hook layers are invalid")
        if (
            tuple(sorted(self.hook_layer_indices)) != self.hook_layer_indices
            or len(set(self.hook_layer_indices)) != len(self.hook_layer_indices)
        ):
            raise ValueError(
                "Transformers execution hook layers must be sorted and unique"
            )
        profile = WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1
        if self.profile_id != profile.profile_id:
            raise ValueError("Transformers execution profile_id is not canonical")
        repeated_profile_facts = {
            "preset_name": profile.preset_name,
            "platform_system": profile.platform_system,
            "attention_implementation": profile.attention_implementation,
            "sdpa_backend": profile.sdpa_backend,
            "sdpa_backend_policy": profile.sdpa_backend_policy,
            "sdpa_backend_exclusive": profile.sdpa_backend_exclusive,
            "generation_use_cache": profile.generation_use_cache,
            "require_generation_chat_template": (
                profile.require_generation_chat_template
            ),
            "generation_capture_strategy": profile.generation_capture_strategy,
            "capture_failure_mode": profile.capture_failure_mode,
            "context_window_tokens": profile.context_window_tokens,
            "local_files_only": profile.local_files_only,
            "fallback_mode": profile.fallback_mode,
            "fail_on_truncation": profile.fail_on_truncation,
            "model_dtype": profile.model_dtype,
        }
        drifted = tuple(
            name
            for name, expected in repeated_profile_facts.items()
            if getattr(self, name) != expected
        )
        if drifted:
            raise ValueError(
                "Transformers execution attestation profile facts drifted: "
                + ", ".join(drifted)
            )
        if self.runtime_origin != "hf-local":
            raise ValueError("Transformers execution runtime_origin must be hf-local")
        valid_device = self.device == profile.device_type or (
            self.device.startswith(f"{profile.device_type}:")
            and self.device.removeprefix(f"{profile.device_type}:").isdigit()
        )
        if not valid_device:
            raise ValueError("Transformers execution device must name a CUDA index")
        if (
            self.model_max_position_embeddings
            != self.context_window_tokens
        ):
            raise ValueError("Transformers execution CUDA/context facts are invalid")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "preset_name": self.preset_name,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_weights_sha256": self.model_weights_sha256,
            "execution_assets_sha256": self.execution_assets_sha256,
            "runtime_origin": self.runtime_origin,
            "platform_system": self.platform_system,
            "platform_release": self.platform_release,
            "device": self.device,
            "device_name": self.device_name,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "transformers_version": self.transformers_version,
            "cuda_version": self.cuda_version,
            "cudnn_version": self.cudnn_version,
            "device_compute_capability": list(self.device_compute_capability),
            "attention_implementation": self.attention_implementation,
            "sdpa_backend": self.sdpa_backend,
            "sdpa_backend_policy": self.sdpa_backend_policy,
            "sdpa_backend_exclusive": self.sdpa_backend_exclusive,
            "generation_use_cache": self.generation_use_cache,
            "require_generation_chat_template": (
                self.require_generation_chat_template
            ),
            "generation_capture_strategy": self.generation_capture_strategy,
            "capture_failure_mode": self.capture_failure_mode,
            "context_window_tokens": self.context_window_tokens,
            "local_files_only": self.local_files_only,
            "fallback_mode": self.fallback_mode,
            "fail_on_truncation": self.fail_on_truncation,
            "model_dtype": self.model_dtype,
            "hidden_size": self.hidden_size,
            "model_max_position_embeddings": self.model_max_position_embeddings,
            "hook_layer_indices": list(self.hook_layer_indices),
        }

    @property
    def attestation_id(self) -> str:
        return _content_sha256(self.to_payload())


@dataclass(frozen=True)
class GenerationContextBudgetAttestation:
    """Exact post-template, post-prefix context budget for one generation."""

    execution_attestation_id: str
    input_mode: str
    input_token_count: int
    prefix_slot_count: int
    effective_max_new_tokens: int
    combined_token_count: int
    context_window_tokens: int
    remaining_token_count: int
    schema_version: str = GENERATION_CONTEXT_BUDGET_ATTESTATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version
            != GENERATION_CONTEXT_BUDGET_ATTESTATION_SCHEMA_VERSION
        ):
            raise ValueError("GenerationContextBudgetAttestation schema drift")
        _require_sha256_text(
            self.execution_attestation_id, "execution_attestation_id"
        )
        if type(self.input_mode) is not str or self.input_mode != "chat-template":
            raise ValueError("generation context input_mode is invalid")
        counts = (
            self.input_token_count,
            self.prefix_slot_count,
            self.effective_max_new_tokens,
            self.combined_token_count,
            self.context_window_tokens,
            self.remaining_token_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("generation context token counts must be non-negative")
        if self.input_token_count <= 0 or self.effective_max_new_tokens <= 0:
            raise ValueError(
                "generation context input and max-new token counts must be positive"
            )
        if (
            self.context_window_tokens
            != WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1.context_window_tokens
        ):
            raise ValueError(
                "generation context window must equal the canonical 32768 tokens"
            )
        expected = (
            self.input_token_count
            + self.prefix_slot_count
            + self.effective_max_new_tokens
        )
        if self.combined_token_count != expected:
            raise ValueError("generation context combined count drift")
        if self.remaining_token_count != self.context_window_tokens - expected:
            raise ValueError("generation context remaining count drift")
        if self.remaining_token_count < 0:
            raise ValueError("generation context exceeds the native window")


def _content_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256_text(value: str, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _is_model_revision(value: object) -> bool:
    return (
        type(value) is str
        and len(value) in (40, 64)
        and all(character in "0123456789abcdef" for character in value)
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
    # ETA Eq.3 steered-action supervision needs the observation prompt that
    # produced this step's residual capture, so the SSL distortion can be
    # re-scored through the controlled frozen model. Empty for legacy
    # residual-proxy traces.
    observation_text: str = ""


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
    input_token_count: int = 0
    source_sha256: str = ""
    personal_conditioning_applied: bool = False
    conditioning_bank_carriers_applied: tuple[tuple[str, str], ...] = ()
    character_prefix_applied: bool = False
    character_prefix_id: str = ""
    character_id: str = ""
    character_prefix_wiring_level: str = "disabled"
    character_prefix_shadow_id: str = ""
    character_residual_applied: bool = False
    character_residual_adapter_id: str = ""
    steering_intervention_applied: bool = False
    steering_action: str = ""
    steering_executor_artifact_id: str = ""
    steering_gate_policy_version: int = 0
    execution_attestation_id: str = ""
    context_budget: GenerationContextBudgetAttestation | None = None

    def __post_init__(self) -> None:
        if type(self.execution_attestation_id) is not str:
            raise TypeError("GenerationResult.execution_attestation_id must be str")
        has_execution_id = bool(self.execution_attestation_id)
        has_context_budget = self.context_budget is not None
        if has_execution_id != has_context_budget:
            raise ValueError(
                "GenerationResult execution attestation and context budget "
                "must be present or absent together"
            )
        if not has_execution_id:
            return
        _require_sha256_text(
            self.execution_attestation_id, "execution_attestation_id"
        )
        if type(self.context_budget) is not GenerationContextBudgetAttestation:
            raise TypeError(
                "GenerationResult.context_budget must be an exact "
                "GenerationContextBudgetAttestation"
            )
        if (
            self.context_budget.execution_attestation_id
            != self.execution_attestation_id
        ):
            raise ValueError(
                "GenerationResult execution attestation lineage mismatch"
            )


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

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        truncation: bool,
        max_length: int | None = None,
    ):
        if return_tensors != "pt":
            raise ValueError("HashingWhitespaceTokenizer expects return_tensors='pt'.")
        torch = importlib.import_module("torch")
        tokens = tuple(part for part in text.split() if part.strip())
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        tokens = tokens or ("<empty>",)
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
