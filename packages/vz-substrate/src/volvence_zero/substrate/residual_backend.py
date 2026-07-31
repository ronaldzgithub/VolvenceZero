from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import importlib
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from volvence_zero.conditioning_bank_contracts import (
    ConditioningBankLatentCarrier,
    ConditioningBankType,
)
from volvence_zero.personal_conditioning_contracts import (
    PERSONAL_CONDITIONING_VECTOR_LABELS,
    PersonalConditioningSnapshot,
)

_LOG = logging.getLogger("volvence_zero.substrate.residual_backend")

from volvence_zero.substrate.adapter import (
    FeatureSignal,
    ResidualActivation,
    ResidualSequenceStep,
    ResidualStreamSubstrateAdapter,
    SubstrateSnapshot,
)

if False:
    from volvence_zero.agent.response import GenerationConstraints



# Slice S.3 (2026-05-04): pure data contracts, ABCs, statistical helpers,
# intervention backends, the synthetic runtime, and training-trace
# helpers previously inlined in this file now live in sibling modules.
# ``TransformersOpenWeightResidualRuntime`` and its factory builders
# remain here. External consumers import everything through the
# ``volvence_zero.substrate`` package facade unchanged.
from volvence_zero.substrate.residual_contracts import (  # noqa: E402,F401
    ContinuationScore,
    GenerationResult,
    HashingWhitespaceTokenizer,
    HookLayerCalibrationCase,
    HookLayerCalibrationReport,
    LocalModelCompatibilityReport,
    LocalSubstrateRuntimeMode,
    OpenWeightRuntimeCapture,
    ResidualControlApplication,
    SubstrateDeltaAdapterLayer,
    SubstrateFallbackMode,
    SubstrateOnlineFastCheckpoint,
    SubstrateRareHeavyCheckpoint,
    TraceStep,
    TrainingTrace,
)
from volvence_zero.substrate.control_basis import (  # noqa: E402,F401
    FIXED_SINUSOID_CONTROL_BASIS_PROVENANCE,
)
from volvence_zero.substrate.conditioning_bank_projector import (  # noqa: E402
    RELATIONSHIP_RESIDUAL_PROJECTOR_VERSION,
    RelationshipConditioningProjectorArtifact,
    build_conditioning_bank_residual_delta,
    load_relationship_projector_basis,
)
from volvence_zero.substrate.residual_interfaces import (  # noqa: E402,F401
    OpenWeightResidualRuntime,
    ResidualInterventionBackend,
)
from volvence_zero.substrate.residual_helpers import (  # noqa: E402,F401
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
from volvence_zero.substrate.residual_intervention import (  # noqa: E402,F401
    NoOpResidualInterventionBackend,
    OpenWeightResidualInterventionBackend,
    TraceResidualInterventionBackend,
    apply_residual_control,
)
from volvence_zero.substrate.rare_heavy_training import (  # noqa: E402
    RareHeavyAdapterTrainingBackend,
    RareHeavyTrainingRequest,
)
from volvence_zero.substrate.personal_conditioning_projector import (  # noqa: E402
    PersonalConditioningProjectorArtifact,
    load_projector_basis,
)
from volvence_zero.substrate.prefix_kv_artifact import (  # noqa: E402
    CharacterPrefixKVPackage,
    CharacterPrefixKVRegistry,
    PrefixKVArtifact,
    load_prefix_generator,
)
from volvence_zero.substrate.common_adapter_bundle import (  # noqa: E402
    CommonAdapterBundle,
    fingerprint_model_weight_files,
)
from volvence_zero.substrate.character_residual_artifact import (  # noqa: E402
    CharacterResidualAdapterPackage,
    load_character_residual_deltas,
)
from volvence_zero.substrate.relationship_prefix_kv_artifact import (  # noqa: E402
    RelationshipPrefixKVArtifact,
    load_relationship_prefix_generator,
)
from volvence_zero.substrate.residual_synthetic import (  # noqa: E402,F401
    SyntheticOpenWeightResidualRuntime,
)
from volvence_zero.substrate.residual_training import (  # noqa: E402,F401
    SimulatedResidualSubstrateAdapter,
    TrainingTraceDataset,
    build_training_trace,
)


PERSONAL_CONDITIONING_SCALE_CAP = 0.12


def _concat_prefix_pairs(
    torch_module: Any,
    left: list[tuple[Any, Any]] | None,
    right: list[tuple[Any, Any]] | None,
) -> list[tuple[Any, Any]] | None:
    """Concatenate static character and dynamic personal KV slots."""

    if left is None:
        return right
    if right is None:
        return left
    if len(left) != len(right):
        raise ValueError(
            "character and personal prefix artifacts must have the same layer count."
        )
    return [
        (
            torch_module.cat((left[index][0], right[index][0]), dim=-2),
            torch_module.cat((left[index][1], right[index][1]), dim=-2),
        )
        for index in range(len(left))
    ]


def _banned_repeated_ngram_tokens(
    token_ids: Sequence[int],
    *,
    ngram_size: int,
) -> tuple[int, ...]:
    """Return standard no-repeat-ngram candidates for the next token."""

    if ngram_size <= 1 or len(token_ids) < ngram_size:
        return ()
    prefix = tuple(token_ids[-(ngram_size - 1) :])
    banned = {
        token_ids[index + ngram_size - 1]
        for index in range(len(token_ids) - ngram_size + 1)
        if tuple(token_ids[index : index + ngram_size - 1]) == prefix
    }
    return tuple(sorted(banned))


def clamp_personal_conditioning_scale(scale: float) -> float:
    """Clamp the personal conditioning scale into ``[0, 0.12]``.

    The upper bound is the hard contract cap documented in
    ``docs/specs/personal-conditioning.md``; no configuration may raise
    the injection magnitude above it.
    """

    return max(0.0, min(float(scale), PERSONAL_CONDITIONING_SCALE_CAP))


def build_personal_conditioning_basis(
    *,
    torch_module: Any,
    hidden_size: int,
    vector_dim: int,
    device: Any | None = None,
):
    """Build the fixed sine/cosine projection basis for personal conditioning.

    Pure function of ``(hidden_size, vector_dim)``: one L2-normalised
    row per conditioning coordinate, deterministic across processes.
    Extracted from the runtime so the projection can be unit-tested
    without loading a model.
    """

    positions = torch_module.arange(hidden_size, dtype=torch_module.float32)
    rows = []
    for factor in range(1, vector_dim + 1):
        row = torch_module.sin(
            (positions + 1.0) * 0.053 * factor
        ) + torch_module.cos(
            (positions + 1.0) * 0.031 * (factor + 2.0)
        )
        row = row / row.norm().clamp_min(1e-6)
        rows.append(row)
    basis = torch_module.stack(rows, dim=0)
    return basis.to(device) if device is not None else basis


def build_personal_conditioning_delta(
    *,
    torch_module: Any,
    conditioning: PersonalConditioningSnapshot | None,
    basis: Any,
    scale: float,
    device: Any | None = None,
):
    """Project a conditioning snapshot onto the hidden-width delta.

    Returns ``None`` (no injection) for absent, cold-start, or
    zero-confidence snapshots. Otherwise the delta magnitude is
    ``scale * confidence`` after per-active-dimension normalisation.
    """

    if (
        conditioning is None
        or conditioning.is_cold_start
        or conditioning.confidence <= 0.0
    ):
        return None
    if len(conditioning.state_vector) != len(
        PERSONAL_CONDITIONING_VECTOR_LABELS
    ):
        raise ValueError(
            "personal conditioning vector width does not match the "
            "substrate projection contract."
        )
    state = torch_module.tensor(
        conditioning.state_vector,
        dtype=torch_module.float32,
        device=device,
    )
    active_dims = max(
        int((state.abs() > 1e-8).sum().item()),
        1,
    )
    delta = state @ basis
    delta = delta / math.sqrt(active_dims)
    return delta * scale * float(conditioning.confidence)


class TransformersOpenWeightResidualRuntime(OpenWeightResidualRuntime):
    """Frozen HF runtime with real middle-layer capture and intervention hooks."""

    def __init__(
        self,
        *,
        model_id: str,
        pretrained_source: str | None = None,
        device: str = "cpu",
        model: object | None = None,
        tokenizer: object | None = None,
        max_length: int = 64,
        mps_generation_max_input_tokens: int = 1024,
        top_k_logits: int = 8,
        activation_width: int = 8,
        layer_indices: tuple[int, ...] | None = None,
        hook_layer_selection: str = "middle",
        control_scale: float = 0.12,
        personal_conditioning_scale: float = 0.08,
        personal_conditioning_projector: (
            PersonalConditioningProjectorArtifact | None
        ) = None,
        relationship_conditioning_projector: (
            RelationshipConditioningProjectorArtifact | None
        ) = None,
        personal_conditioning_prefix: PrefixKVArtifact | None = None,
        relationship_conditioning_prefix: (
            RelationshipPrefixKVArtifact | None
        ) = None,
        character_prefix_package: CharacterPrefixKVPackage | None = None,
        character_prefix_registry: CharacterPrefixKVRegistry | None = None,
        common_adapter_bundle: CommonAdapterBundle | None = None,
        character_residual_package: CharacterResidualAdapterPackage | None = None,
        local_files_only: bool = False,
        runtime_origin: str = "hf-pretrained",
        allow_live_substrate_mutation: bool = False,
        allow_offline_substrate_training: bool = False,
    ) -> None:
        self._torch = importlib.import_module("torch")
        self._transformers = importlib.import_module("transformers")
        self.model_id = model_id
        self._pretrained_source = pretrained_source or model_id
        self.is_frozen = True
        self.supports_live_substrate_mutation = allow_live_substrate_mutation
        self.supports_offline_substrate_training = allow_offline_substrate_training
        self._device = self._resolve_device(device=device)
        self._max_length = max(1, max_length)
        self._mps_generation_max_input_tokens = max(1, mps_generation_max_input_tokens)
        self._top_k_logits = max(1, top_k_logits)
        if (
            isinstance(activation_width, bool)
            or not isinstance(activation_width, int)
            or activation_width < 1
        ):
            raise ValueError(
                "activation_width must be a positive integer, "
                f"got {activation_width!r}"
            )
        self._activation_width = activation_width
        self._control_scale = max(0.0, control_scale)
        self._personal_conditioning_scale = clamp_personal_conditioning_scale(
            personal_conditioning_scale
        )
        self._runtime_origin = runtime_origin
        self.runtime_origin = runtime_origin
        self._tokenizer = tokenizer or self._load_tokenizer(
            model_id=self._pretrained_source,
            local_files_only=local_files_only,
        )
        self._model = model or self._load_model(
            model_id=self._pretrained_source,
            local_files_only=local_files_only,
        )
        self._prepare_model()
        self._block_modules = self._resolve_transformer_blocks()
        self._layer_indices = self._normalize_layer_indices(
            requested=layer_indices,
            block_count=len(self._block_modules),
            hook_layer_selection=hook_layer_selection,
        )
        self._hidden_size = self._resolve_hidden_size()
        self._model_family = self._resolve_model_family()
        self._control_basis = self._build_control_basis(hidden_size=self._hidden_size)
        self._control_basis_provenance = FIXED_SINUSOID_CONTROL_BASIS_PROVENANCE
        self._control_layer_gains = {
            layer_index: 1.0 for layer_index in self._layer_indices
        }
        self._personal_conditioning_basis = (
            self._build_personal_conditioning_basis(
                hidden_size=self._hidden_size,
                vector_dim=len(PERSONAL_CONDITIONING_VECTOR_LABELS),
            )
        )
        self._personal_conditioning_layer_gains = {
            self._layer_indices[0]: 1.0
        }
        self._personal_conditioning_projector_id = "fixed-sine-cosine-v1"
        self._personal_conditioning_projector_training_mode = "fixed"
        if personal_conditioning_projector is not None:
            (
                self._personal_conditioning_basis,
                self._personal_conditioning_layer_gains,
            ) = load_projector_basis(
                torch_module=self._torch,
                artifact=personal_conditioning_projector,
                expected_model_id=self.model_id,
                expected_hidden_size=self._hidden_size,
                available_layer_indices=self._layer_indices,
                device=self._device,
            )
            self._personal_conditioning_projector_id = (
                personal_conditioning_projector.artifact_id
            )
            self._personal_conditioning_projector_training_mode = (
                personal_conditioning_projector.training_mode
            )
        self._relationship_conditioning_projector = (
            relationship_conditioning_projector
        )
        self._relationship_conditioning_basis = None
        self._relationship_conditioning_vector_labels = None
        self._relationship_conditioning_layer_gains = {
            self._layer_indices[0]: 1.0
        }
        self._relationship_conditioning_projector_id = (
            RELATIONSHIP_RESIDUAL_PROJECTOR_VERSION
        )
        self._relationship_conditioning_projector_training_mode = "fixed"
        self._relationship_conditioning_projector_version = (
            RELATIONSHIP_RESIDUAL_PROJECTOR_VERSION
        )
        if relationship_conditioning_projector is not None:
            (
                self._relationship_conditioning_basis,
                self._relationship_conditioning_layer_gains,
            ) = load_relationship_projector_basis(
                torch_module=self._torch,
                artifact=relationship_conditioning_projector,
                expected_model_id=self.model_id,
                expected_hidden_size=self._hidden_size,
                available_layer_indices=self._layer_indices,
                device=self._device,
            )
            self._relationship_conditioning_vector_labels = (
                relationship_conditioning_projector.vector_labels
            )
            self._relationship_conditioning_projector_id = (
                relationship_conditioning_projector.artifact_id
            )
            self._relationship_conditioning_projector_training_mode = (
                relationship_conditioning_projector.training_mode
            )
            self._relationship_conditioning_projector_version = (
                relationship_conditioning_projector.projector_version
            )
        if (
            common_adapter_bundle is not None
            and personal_conditioning_prefix is not None
            and personal_conditioning_prefix.artifact_id
            != common_adapter_bundle.state_kv_artifact.artifact_id
        ):
            raise ValueError(
                "personal_conditioning_prefix conflicts with the State-KV "
                "artifact in common_adapter_bundle."
            )
        if common_adapter_bundle is not None:
            personal_conditioning_prefix = common_adapter_bundle.state_kv_artifact
        self._prefix_generator = None
        self._personal_conditioning_prefix_id = ""
        if personal_conditioning_prefix is not None:
            self._prefix_generator = load_prefix_generator(
                torch_module=self._torch,
                artifact=personal_conditioning_prefix,
                expected_model_id=self.model_id,
                expected_num_layers=len(self._block_modules),
                expected_num_kv_heads=self._resolve_num_kv_heads(),
                expected_head_dim=self._resolve_head_dim(),
                device=self._device,
                dtype=self._model_dtype(),
            )
            self._personal_conditioning_prefix_id = (
                personal_conditioning_prefix.artifact_id
            )
        self._relationship_conditioning_prefix = (
            relationship_conditioning_prefix
        )
        self._relationship_prefix_generator = None
        self._relationship_conditioning_prefix_id = ""
        self._relationship_conditioning_prefix_version = ""
        self._relationship_conditioning_prefix_norm_cap = 0.0
        if relationship_conditioning_prefix is not None:
            self._relationship_prefix_generator = (
                load_relationship_prefix_generator(
                    torch_module=self._torch,
                    artifact=relationship_conditioning_prefix,
                    expected_model_id=self.model_id,
                    expected_num_layers=len(self._block_modules),
                    expected_num_kv_heads=self._resolve_num_kv_heads(),
                    expected_head_dim=self._resolve_head_dim(),
                    device=self._device,
                    dtype=self._model_dtype(),
                )
            )
            self._relationship_conditioning_prefix_id = (
                relationship_conditioning_prefix.artifact_id
            )
            self._relationship_conditioning_prefix_version = (
                relationship_conditioning_prefix.carrier_version
            )
            self._relationship_conditioning_prefix_norm_cap = (
                relationship_conditioning_prefix.prefix_artifact.norm_cap
            )
        self._character_prefix_generator = None
        self._character_prefix_id = ""
        self._character_prefix_pairs = None
        self._character_prefix_package = character_prefix_package
        self._character_prefix_registry = character_prefix_registry
        self._character_prefix_pairs_by_character: dict[str, object] = {}
        if character_prefix_package is not None:
            if character_prefix_package.model_id != self.model_id:
                raise ValueError(
                    "character prefix package model_id "
                    f"{character_prefix_package.model_id!r} does not match "
                    f"runtime {self.model_id!r}."
                )
            self._character_prefix_generator = load_prefix_generator(
                torch_module=self._torch,
                artifact=character_prefix_package.prefix_artifact,
                expected_model_id=self.model_id,
                expected_num_layers=len(self._block_modules),
                expected_num_kv_heads=self._resolve_num_kv_heads(),
                expected_head_dim=self._resolve_head_dim(),
                device=self._device,
                dtype=self._model_dtype(),
            )
            self._character_prefix_pairs = self._character_prefix_generator.build(
                character_prefix_package.state_vector
            )
            self._character_prefix_id = character_prefix_package.package_id
        if character_prefix_registry is not None:
            if common_adapter_bundle is None:
                raise ValueError(
                    "character_prefix_registry requires common_adapter_bundle "
                    "so adapter compatibility can be checked."
                )
            if character_prefix_registry.base_model_id != self.model_id:
                raise ValueError(
                    "character prefix registry base_model_id does not match "
                    f"runtime {self.model_id!r}."
                )
            if (
                character_prefix_registry.common_adapter_version
                != common_adapter_bundle.common_adapter_version
                or character_prefix_registry.compatibility_fingerprint
                != common_adapter_bundle.compatibility_fingerprint
            ):
                raise ValueError(
                    "character prefix registry does not match the loaded "
                    "common adapter version/fingerprint."
                )
            for entry in character_prefix_registry.entries:
                generator = load_prefix_generator(
                    torch_module=self._torch,
                    artifact=entry.prefix_package.prefix_artifact,
                    expected_model_id=self.model_id,
                    expected_num_layers=len(self._block_modules),
                    expected_num_kv_heads=self._resolve_num_kv_heads(),
                    expected_head_dim=self._resolve_head_dim(),
                    device=self._device,
                    dtype=self._model_dtype(),
                )
                self._character_prefix_pairs_by_character[entry.character_id] = (
                    generator.build(entry.prefix_package.state_vector)
                )
        self._character_residual_package = character_residual_package
        self._character_residual_adapter_id = ""
        self._character_residual_deltas: dict[int, object] = {}
        if character_residual_package is not None:
            self._character_residual_deltas = load_character_residual_deltas(
                torch_module=self._torch,
                package=character_residual_package,
                expected_model_id=self.model_id,
                expected_hidden_size=self._hidden_size,
                available_layer_indices=self._layer_indices,
                device=self._device,
            )
            self._character_residual_adapter_id = character_residual_package.package_id
        self._semantic_projection_dim = 24
        self._semantic_basis = self._build_semantic_basis(
            hidden_size=self._hidden_size,
            projection_dim=self._semantic_projection_dim,
        )
        self._semantic_anchor_profiles = _anchor_profile_bank(dim=self._semantic_projection_dim)
        base_text_weight, base_residual_weight = self._base_semantic_weights()
        self._rare_heavy_control_scale = self._control_scale
        self._rare_heavy_semantic_text_weight = base_text_weight
        self._rare_heavy_semantic_residual_weight = base_residual_weight
        self._rare_heavy_anchor_bias = tuple(0.0 for _ in RARE_HEAVY_ANCHOR_ORDER)
        self._rare_heavy_update_count = 0
        self._rare_heavy_adapter_scale = 0.0
        self._rare_heavy_adapter_deltas: dict[int, object] = {}
        self._online_fast_delta_scale = 0.0
        self._online_fast_update_count = 0
        self._online_fast_optimizer_state_norm = 0.0
        self._online_fast_parameter_change_rate = 0.0
        self._online_fast_adapter_deltas: dict[int, object] = {}
        self._online_fast_state_hash = ""
        self._online_fast_source_state_hash = ""
        self._online_fast_signal: tuple[float, ...] = ()
        self._online_fast_optimizer_state_description = ""
        # S1: injectable real rare-heavy training backend (e.g. PEFT LoRA).
        # None -> the built-in adapter-delta autograd loop stays in charge.
        self._rare_heavy_training_backend: RareHeavyAdapterTrainingBackend | None = None
        self._common_adapter_bundle = common_adapter_bundle
        if common_adapter_bundle is not None:
            self._install_common_adapter_bundle(common_adapter_bundle)

    @property
    def hook_layer_indices(self) -> tuple[int, ...]:
        """Frozen hook surface available to projector artifacts."""

        return self._layer_indices

    @property
    def hidden_size(self) -> int:
        """Residual width used by projector compatibility checks."""

        return self._hidden_size

    @property
    def personal_conditioning_projector_id(self) -> str:
        return self._personal_conditioning_projector_id

    @property
    def personal_conditioning_projector_training_mode(self) -> str:
        return self._personal_conditioning_projector_training_mode

    @property
    def relationship_conditioning_projector_id(self) -> str:
        return self._relationship_conditioning_projector_id

    @property
    def relationship_conditioning_projector_training_mode(self) -> str:
        return self._relationship_conditioning_projector_training_mode

    @property
    def relationship_conditioning_projector_version(self) -> str:
        return self._relationship_conditioning_projector_version

    @property
    def relationship_conditioning_prefix_id(self) -> str:
        return self._relationship_conditioning_prefix_id

    @property
    def relationship_conditioning_prefix_version(self) -> str:
        return self._relationship_conditioning_prefix_version

    @property
    def relationship_conditioning_prefix_norm_cap(self) -> float:
        return self._relationship_conditioning_prefix_norm_cap

    @property
    def personal_conditioning_prefix_id(self) -> str:
        """Artifact id of the loaded State-KV prefix, empty when unloaded."""

        return self._personal_conditioning_prefix_id

    @property
    def supports_prefix_kv(self) -> bool:
        return self._prefix_generator is not None

    @property
    def character_residual_adapter_id(self) -> str:
        """Content id of the loaded character residual adapter, if any."""

        return self._character_residual_adapter_id

    @property
    def common_adapter_version(self) -> str:
        bundle = self._common_adapter_bundle
        return bundle.common_adapter_version if bundle is not None else ""

    @property
    def common_adapter_compatibility_fingerprint(self) -> str:
        bundle = self._common_adapter_bundle
        return bundle.compatibility_fingerprint if bundle is not None else ""

    @property
    def registered_character_ids(self) -> tuple[str, ...]:
        registry = self._character_prefix_registry
        return registry.character_ids if registry is not None else ()

    def _resolve_num_kv_heads(self) -> int:
        config = getattr(self._model, "config", None)
        heads = getattr(config, "num_key_value_heads", None)
        if heads is None:
            heads = getattr(config, "num_attention_heads", None)
        if not isinstance(heads, int) or heads <= 0:
            raise ValueError(
                f"cannot resolve attention KV head count for {self.model_id!r}; "
                "a prefix artifact cannot be checked against unknown geometry."
            )
        return heads

    def _resolve_head_dim(self) -> int:
        config = getattr(self._model, "config", None)
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            heads = getattr(config, "num_attention_heads", 0)
            hidden = getattr(config, "hidden_size", 0)
            if isinstance(heads, int) and heads > 0 and isinstance(hidden, int):
                head_dim = hidden // heads
        if not isinstance(head_dim, int) or head_dim <= 0:
            raise ValueError(
                f"cannot resolve attention head dim for {self.model_id!r}; "
                "a prefix artifact cannot be checked against unknown geometry."
            )
        return head_dim

    def _model_dtype(self) -> Any:
        for parameter in self._model.parameters():
            return parameter.dtype
        return self._torch.float32

    def set_rare_heavy_training_backend(
        self, backend: RareHeavyAdapterTrainingBackend | None
    ) -> None:
        """Install (or clear) the offline rare-heavy training backend.

        When set, ``train_rare_heavy`` delegates adapter training to the
        backend instead of the built-in ``_train_adapter_deltas`` loop.
        A backend failure fails loudly — an explicitly injected backend
        never silently falls back to the built-in loop (R15: the fallback
        is the *uninjected* configuration, reachable by clearing this).
        """

        self._rare_heavy_training_backend = backend

    def capture(self, *, source_text: str) -> OpenWeightRuntimeCapture:
        # Windows CUDA hosts with the Raptor Lake Vmin Shift defect (pre-fix
        # microcode) intermittently corrupt the token-level hook capture path.
        # Default to the pooled summary there; bypass with
        # VZ_TORCH_BACKENDS_FORCE=1 on a stabilized lane (E-core pinned or
        # patched microcode) because promotion evidence requires the real
        # residual sequence. Same switch as final_wiring.py. Reversible.
        force = os.environ.get("VZ_TORCH_BACKENDS_FORCE", "").strip().lower() in (
            "1", "true", "on", "yes",
        )
        if not force and os.name == "nt" and str(self._device).startswith("cuda"):
            return self._capture_pooled_summary(source_text=source_text)
        return self._capture_with_hooks(source_text=source_text)

    def capture_conditioned(
        self,
        *,
        source_text: str,
        personal_conditioning: PersonalConditioningSnapshot,
        personal_conditioning_carrier: str,
    ) -> OpenWeightRuntimeCapture:
        """Capture the real prompt-token residuals after State-KV delivery."""

        if personal_conditioning_carrier not in ("residual", "prefix_kv"):
            raise ValueError(
                "personal_conditioning_carrier must be 'residual' or "
                f"'prefix_kv', got {personal_conditioning_carrier!r}."
            )
        if (
            personal_conditioning_carrier == "prefix_kv"
            and self._prefix_generator is None
        ):
            raise ValueError(
                "personal_conditioning_carrier='prefix_kv' requires a prefix "
                "artifact."
            )

        effective_source = source_text.strip() or "<empty>"
        model_inputs = self._tokenize(source_text=effective_source)
        input_ids = model_inputs["input_ids"]
        captured_layers: dict[int, object] = {}
        personal_delta = None
        prefix_pairs = None
        if personal_conditioning_carrier == "prefix_kv":
            prefix_pairs = self._build_personal_conditioning_prefix(
                conditioning=personal_conditioning
            )
        else:
            personal_delta = self._build_personal_conditioning_delta(
                conditioning=personal_conditioning
            )
        conditioning_applied = (
            prefix_pairs is not None
            if personal_conditioning_carrier == "prefix_kv"
            else personal_delta is not None
        )
        hooks = [
            self._block_modules[layer_index].register_forward_hook(
                self._make_capture_hook(
                    layer_index=layer_index,
                    captured_layers=captured_layers,
                    control_delta=None,
                    personal_delta=personal_delta,
                    character_residual_delta=self._character_residual_deltas.get(
                        layer_index
                    ),
                )
            )
            for layer_index in self._layer_indices
        ]
        try:
            with self._torch.no_grad():
                if personal_conditioning_carrier == "prefix_kv":
                    attention_mask = model_inputs.get("attention_mask")
                    if attention_mask is None:
                        attention_mask = self._torch.ones_like(input_ids)
                    cache = self._transformers.DynamicCache()
                    slots = 0
                    if prefix_pairs:
                        slots = int(prefix_pairs[0][0].shape[-2])
                        cache = self._transformers.DynamicCache(
                            ddp_cache_data=prefix_pairs
                        )
                    if slots:
                        attention_mask = self._torch.cat(
                            [
                                self._torch.ones(
                                    (1, slots),
                                    dtype=attention_mask.dtype,
                                    device=attention_mask.device,
                                ),
                                attention_mask,
                            ],
                            dim=-1,
                        )
                    positions = self._torch.arange(
                        int(input_ids.shape[-1]),
                        device=input_ids.device,
                    ).unsqueeze(0)
                    outputs = self._model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=positions,
                        past_key_values=cache,
                        use_cache=True,
                    )
                else:
                    outputs = self._model(**model_inputs, use_cache=False)
        finally:
            for hook in hooks:
                hook.remove()
        logits = self._extract_logits(outputs=outputs)
        return self._build_runtime_capture(
            source_text=effective_source,
            input_ids=input_ids,
            logits=logits,
            captured_layers=captured_layers,
            control_applied=conditioning_applied,
            personal_conditioning_applied=conditioning_applied,
        )

    def activate_lora(self, layers):
        """Real-forward override of :meth:`OpenWeightResidualRuntime.activate_lora`.

        Registers a forward hook on every ``_block_modules[layer_index]``
        in the supplied LoRA layer set; the hook adds the broadcast
        ``delta_vector`` to the residual stream output of that
        attention block. On context exit the hooks are removed and
        the model returns to its frozen-base behaviour. The base
        ``state_dict`` is never mutated (R2: controller-layer only).

        Layer indexing: each :class:`SubstrateDeltaAdapterLayer`
        carries ``layer_index``. Layers whose index is not in this
        runtime's hooked layer set (``self._layer_indices``) are
        applied at the closest hooked layer modulo block count;
        this keeps the activation effective even when the persona
        was baked against a different block layout (debt #20
        recommendation 6 — additive sequential, not silent drop).
        """

        return self._lora_hot_swap_context(layers)

    def activate_peft_adapter(self, checkpoint_dir):
        """Load a saved PEFT adapter onto the base model for the context.

        Debt #40 closure path. The projected ``adapter_layers``
        summary vector consumed by :meth:`activate_lora` is a
        per-layer constant that LayerNorm zeroes out — making
        ``BUNDLE`` and ``BUNDLE_LORA`` byte-identical in real Qwen
        forward. ``activate_peft_adapter`` instead loads the
        trained LoRA A/B matrices saved by
        :meth:`peft.PeftModel.save_pretrained` and wraps the base
        model so q_proj / k_proj / etc. forward calls produce the
        true ``base @ x + lora_alpha/r * (B @ (A @ x))`` output.

        ``checkpoint_dir`` MUST point at a directory containing
        ``adapter_config.json`` + ``adapter_model.safetensors``
        (the standard ``peft.save_pretrained`` layout).

        R2 contract: the underlying frozen base weights are not
        mutated. peft inserts adapter sub-modules in-place, but on
        context exit we call ``unload()`` to detach those sub-modules
        without merging their weights into the base. A pre-/post-
        context ``state_dict`` hash on the base weights stays
        byte-identical (covered by ``test_lora_activate_does_not_
        mutate_base.py``).

        Raises ``ImportError`` if ``peft`` is not importable.
        Raises ``FileNotFoundError`` if ``checkpoint_dir`` doesn't
        exist — refuses to silently fall back to a no-op (per
        ``no-swallow-errors-no-hasattr-abuse.mdc``).
        """

        return self._peft_adapter_context(checkpoint_dir)

    def _ensure_peft_cache(self):
        """Return this runtime's lazily-built PEFT adapter LRU cache.

        Keeps up to ``VZ_LORA_CACHE_MAX`` adapters resident in one
        ``peft.PeftModel`` so repeated turns for the same persona avoid
        re-running ``PeftModel.from_pretrained`` per request.
        """

        cache = getattr(self, "_peft_adapter_cache", None)
        if cache is not None:
            return cache
        try:
            peft_mod = importlib.import_module("peft")
        except ImportError as exc:
            raise ImportError(
                "activate_peft_adapter: peft is required to load a saved "
                "PEFT adapter at inference time. Install via "
                "``pip install vz-runtime[torch]`` (peft is a transitive "
                "dep)."
            ) from exc
        from volvence_zero.substrate.peft_adapter_cache import (
            build_default_peft_adapter_cache,
        )

        cache = build_default_peft_adapter_cache(peft_mod)
        object.__setattr__(self, "_peft_adapter_cache", cache)
        return cache

    @property
    def peft_cache_stats(self) -> dict[str, int]:
        """Resident-adapter cache hit/miss/size counters (diagnostics)."""

        cache = getattr(self, "_peft_adapter_cache", None)
        if cache is None:
            return {"hits": 0, "misses": 0, "resident": 0}
        return {
            "hits": cache.hits,
            "misses": cache.misses,
            "resident": cache.resident_count,
        }

    @contextlib.contextmanager
    def _peft_adapter_context(self, checkpoint_dir):
        """Context-manager body for :meth:`activate_peft_adapter`.

        Uses a bounded LRU of resident adapters (debt #40 + adapter
        VRAM cache): instead of ``from_pretrained`` + ``unload`` every
        turn, the requested adapter is loaded once and re-activated on
        subsequent turns. R2 is preserved: on context exit all adapter
        layers are disabled so the base forward path is restored.
        """

        import pathlib

        path = pathlib.Path(str(checkpoint_dir))
        if not path.is_dir():
            raise FileNotFoundError(
                f"activate_peft_adapter: checkpoint_dir {path!r} is not a "
                f"directory. The PEFTLoRABakeBackend saves the adapter at "
                f"bake time; ensure the bundle was baked with a non-empty "
                f"checkpoint_dir and that the directory has not been "
                f"deleted."
            )
        if getattr(self, "_lora_activation_in_flight", False):
            raise RuntimeError(
                "activate_peft_adapter: nested activation detected; exit "
                "the outer LoRA context before activating a different "
                "persona."
            )
        cache = self._ensure_peft_cache()
        original_model = self._model
        with cache.activate(
            base_model=original_model, checkpoint_dir=str(path)
        ) as peft_model:
            try:
                object.__setattr__(self, "_lora_activation_in_flight", True)
                object.__setattr__(
                    self, "_lora_activation_checkpoint_dir", str(path)
                )
                object.__setattr__(self, "_model", peft_model)
                yield
            finally:
                # Restore the base reference; the cache has already
                # disabled the adapter layers so this base forward is
                # the clean frozen-base path (R2). Adapter weights stay
                # resident for the next turn (no per-turn reload).
                object.__setattr__(self, "_model", original_model)
                object.__setattr__(self, "_lora_activation_in_flight", False)
                object.__setattr__(self, "_lora_activation_checkpoint_dir", "")

    @contextlib.contextmanager
    def _lora_hot_swap_context(self, layers):
        """Context-manager body for :meth:`activate_lora`.

        Defined as a separate helper so :meth:`activate_lora` can
        be a thin wrapper that returns the context manager rather
        than yielding from a contextmanager-decorated method (the
        Protocol declares the return type as a context manager).
        """

        if not layers:
            raise ValueError(
                "activate_lora: layers tuple must be non-empty"
            )
        if getattr(self, "_lora_activation_in_flight", False):
            raise RuntimeError(
                "activate_lora: nested activation detected; exit the "
                "outer context before activating a different persona."
            )
        block_count = len(self._block_modules)
        hooked = sorted(self._layer_indices)
        if not hooked:
            raise RuntimeError(
                "activate_lora: runtime has no hooked layers; cannot "
                "apply persona LoRA on top of the frozen base."
            )
        per_layer_deltas: dict[int, list[tuple[float, ...]]] = {
            layer_idx: [] for layer_idx in hooked
        }
        for layer in layers:
            if layer.layer_index in per_layer_deltas:
                target = layer.layer_index
            else:
                normalised = layer.layer_index % max(block_count, 1)
                target = min(hooked, key=lambda idx: abs(idx - normalised))
            per_layer_deltas[target].append(tuple(layer.delta_vector))
        hooks: list[Any] = []
        try:
            object.__setattr__(self, "_lora_activation_in_flight", True)
            object.__setattr__(self, "_lora_activation_layers", tuple(layers))
            for hooked_index, delta_list in per_layer_deltas.items():
                if not delta_list:
                    continue
                summed = self._sum_persona_deltas_to_tensor(delta_list)
                hook = self._block_modules[hooked_index].register_forward_hook(
                    self._make_lora_forward_hook(delta_tensor=summed)
                )
                hooks.append(hook)
            yield
        finally:
            for hook in hooks:
                hook.remove()
            object.__setattr__(self, "_lora_activation_in_flight", False)
            object.__setattr__(self, "_lora_activation_layers", ())

    def _sum_persona_deltas_to_tensor(self, deltas: list[tuple[float, ...]]):
        """Sum-pool persona LoRA deltas into one torch tensor."""

        max_width = max(len(delta) for delta in deltas)
        accumulator = [0.0] * max_width
        for delta in deltas:
            for index, value in enumerate(delta):
                accumulator[index] += float(value)
        return self._torch.tensor(
            accumulator, dtype=self._torch.float32, device=self._device
        )

    def _make_lora_forward_hook(self, *, delta_tensor):
        """Create a per-layer forward hook that adds the LoRA delta."""

        torch_mod = self._torch

        def hook(module, args, output):
            del module, args
            tensor = self._extract_hidden_tensor(output=output)
            hidden_dim = tensor.shape[-1]
            delta = delta_tensor
            if delta.shape[0] < hidden_dim:
                padded = torch_mod.zeros(
                    hidden_dim, dtype=tensor.dtype, device=tensor.device
                )
                padded[: delta.shape[0]] = delta.to(
                    dtype=tensor.dtype, device=tensor.device
                )
                delta = padded
            elif delta.shape[0] > hidden_dim:
                delta = delta[:hidden_dim].to(
                    dtype=tensor.dtype, device=tensor.device
                )
            else:
                delta = delta.to(dtype=tensor.dtype, device=tensor.device)
            broadcast = delta.view(*([1] * (tensor.dim() - 1)), hidden_dim)
            mutated = tensor + broadcast
            return self._replace_hidden_tensor(output=output, replacement=mutated)

        return hook

    def _replace_hidden_tensor(self, *, output, replacement):
        """Return ``output`` with its primary hidden tensor replaced."""

        if isinstance(output, tuple):
            return (replacement,) + tuple(output[1:])
        if isinstance(output, list):
            return [replacement, *output[1:]]
        return replacement

    def _mean_residual_at_layer(
        self,
        *,
        texts: tuple[str, ...],
        layer_index: int,
    ) -> tuple[float, ...]:
        """Batched override using the internal hidden-state means hook.

        Avoids the full-capture overhead of the public path: walks
        each text through ``_capture_hidden_state_means`` (which
        already pools per-token hidden states to a single vector
        per layer) and averages across texts at the requested
        layer. The end result is mathematically equivalent to the
        ABC default but ~2x faster on torch CPU for short texts.
        """

        if not texts:
            raise ValueError(
                "_mean_residual_at_layer: texts must be non-empty"
            )
        if layer_index not in self._layer_indices:
            raise ValueError(
                f"_mean_residual_at_layer: layer_index={layer_index!r} not "
                f"in runtime's hooked layer set {self._layer_indices!r}"
            )
        sums = None
        sample_count = 0
        for text in texts:
            means = self._capture_hidden_state_means(source_text=text)
            tensor = means.get(layer_index)
            if tensor is None:
                raise RuntimeError(
                    f"_mean_residual_at_layer: capture did not produce a "
                    f"hidden-state mean at layer_index={layer_index!r} for "
                    f"text {text[:40]!r}"
                )
            values = tuple(float(v) for v in tensor.detach().cpu().tolist())
            if sums is None:
                sums = list(values)
            else:
                if len(values) != len(sums):
                    raise RuntimeError(
                        "_mean_residual_at_layer: hidden-state width drifted "
                        f"across texts ({len(values)} vs {len(sums)})"
                    )
                for index, value in enumerate(values):
                    sums[index] += value
            sample_count += 1
        if sums is None or sample_count == 0:
            raise RuntimeError(
                "_mean_residual_at_layer: produced empty activation pool"
            )
        return tuple(value / sample_count for value in sums)

    def apply_control(
        self,
        *,
        source_text: str,
        substrate_snapshot: SubstrateSnapshot,
        applied_control: tuple[float, ...],
        track_scale: tuple[float, ...] = (1.0, 1.0, 1.0),
    ) -> ResidualControlApplication:
        after_capture = self._capture_control_summary(
            source_text=source_text,
            applied_control=applied_control,
            track_scale=track_scale,
        )
        before_summary = _summarize_real_activations(substrate_snapshot.residual_activations)
        after_summary = _summarize_real_activations(after_capture.residual_activations)
        logit_before = max(substrate_snapshot.token_logits, default=0.0)
        logit_after = max(after_capture.token_logits, default=0.0)
        downstream_effect = (
            _clamp_signed(after_summary[0] - before_summary[0]),
            _clamp_signed(after_summary[1] - before_summary[1]),
            _clamp_signed((logit_after - logit_before) + after_summary[2] - before_summary[2]),
        )
        control_energy = sum(abs(value) for value in applied_control) / max(len(applied_control), 1)
        applied_snapshot = SubstrateSnapshot(
            model_id=self.model_id,
            is_frozen=self.is_frozen,
            surface_kind=substrate_snapshot.surface_kind,
            token_logits=after_capture.token_logits,
            feature_surface=after_capture.feature_surface,
            residual_activations=after_capture.residual_activations,
            residual_sequence=after_capture.residual_sequence,
            unavailable_fields=substrate_snapshot.unavailable_fields,
            description=(
                f"{after_capture.description} Applied transformers residual control "
                f"{tuple(round(value, 3) for value in applied_control)}."
            ),
        )
        return ResidualControlApplication(
            applied_snapshot=applied_snapshot,
            downstream_effect=downstream_effect,
            control_energy=control_energy,
            backend_name=f"transformers-open-weight:{self.model_id}",
            description=(
                f"transformers-open-weight:{self.model_id} device={self._device} "
                f"layers={self._layer_indices} effect={tuple(round(value, 3) for value in downstream_effect)}."
            ),
        )

    def score_continuation(
        self,
        *,
        source_text: str,
        continuation_text: str,
        applied_control: tuple[float, ...],
        track_scale: tuple[float, ...] = (1.0, 1.0, 1.0),
    ) -> ContinuationScore:
        return self.score_continuations(
            source_text=source_text,
            continuation_texts=(continuation_text,),
            applied_control=applied_control,
            track_scale=track_scale,
        )[0]

    def score_continuations(
        self,
        *,
        source_text: str,
        continuation_texts: tuple[str, ...],
        applied_control: tuple[float, ...],
        track_scale: tuple[float, ...] = (1.0, 1.0, 1.0),
    ) -> tuple[ContinuationScore, ...]:
        effective_source = source_text.strip() or "<empty>"
        if not continuation_texts:
            raise ValueError(
                "continuation_texts must contain at least one continuation"
            )
        if any(not text.strip() for text in continuation_texts):
            raise ValueError("continuation_texts must all be nonempty")
        source_inputs = self._tokenize(source_text=effective_source)
        source_ids = source_inputs["input_ids"]
        source_length = int(source_ids.shape[-1])
        combined_ids_by_row = []
        combined_lengths = []
        for continuation_text in continuation_texts:
            combined_text = (
                f"{effective_source.rstrip()} "
                f"{continuation_text.strip()}"
            )
            combined_ids = self._tokenize(
                source_text=combined_text
            )["input_ids"]
            combined_length = int(combined_ids.shape[-1])
            if combined_length <= source_length:
                raise ValueError(
                    "continuation_text did not add any scoreable tokens"
                )
            if not self._torch.equal(
                combined_ids[:, :source_length],
                source_ids,
            ):
                raise ValueError(
                    "source tokenization is not a prefix of "
                    "source+continuation; continuation score would be "
                    "misaligned"
                )
            combined_ids_by_row.append(combined_ids[0])
            combined_lengths.append(combined_length)
        max_length = max(combined_lengths)
        pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(
                self._tokenizer,
                "eos_token_id",
                None,
            )
        if pad_token_id is None:
            pad_token_id = 0
        batched_ids = self._torch.full(
            (len(combined_ids_by_row), max_length),
            int(pad_token_id),
            dtype=combined_ids_by_row[0].dtype,
            device=self._device,
        )
        attention_mask = self._torch.zeros(
            (len(combined_ids_by_row), max_length),
            dtype=self._torch.long,
            device=self._device,
        )
        for row_index, row_ids in enumerate(combined_ids_by_row):
            row_length = combined_lengths[row_index]
            batched_ids[row_index, :row_length] = row_ids
            attention_mask[row_index, :row_length] = 1
        combined_inputs = {
            "input_ids": batched_ids,
            "attention_mask": attention_mask,
        }
        hooks = [
            self._block_modules[layer_index].register_forward_hook(
                self._make_capture_hook(
                    layer_index=layer_index,
                    captured_layers={},
                    control_delta=self._build_control_delta(
                        applied_control=applied_control,
                        track_scale=track_scale,
                        layer_index=layer_index,
                    ),
                    capture_residuals=False,
                )
            )
            for layer_index in self._layer_indices
        ]
        try:
            with self._torch.no_grad():
                outputs = self._model(
                    **combined_inputs,
                    use_cache=False,
                )
        finally:
            for hook in hooks:
                hook.remove()
        logits = self._extract_logits(outputs=outputs).to(
            dtype=self._torch.float32
        )
        scores = []
        for row_index, continuation_text in enumerate(
            continuation_texts
        ):
            combined_length = combined_lengths[row_index]
            prediction_logits = logits[
                row_index : row_index + 1,
                source_length - 1 : combined_length - 1,
                :,
            ]
            target_ids = batched_ids[
                row_index : row_index + 1,
                source_length:combined_length,
            ]
            token_log_probabilities = self._torch.log_softmax(
                prediction_logits,
                dim=-1,
            ).gather(
                dim=-1,
                index=target_ids.unsqueeze(-1),
            ).squeeze(-1)
            mean_log_probability = float(
                token_log_probabilities.mean().item()
            )
            scores.append(
                ContinuationScore(
                    source_text=effective_source,
                    continuation_text=continuation_text,
                    token_count=int(target_ids.numel()),
                    mean_negative_log_likelihood=-mean_log_probability,
                    geometric_mean_probability=float(
                        self._torch.exp(
                            self._torch.tensor(mean_log_probability)
                        ).item()
                    ),
                    applied_control=applied_control,
                    backend_name=(
                        f"transformers-open-weight:{self.model_id}"
                    ),
                    description=(
                        "Observed continuation cohort scored under "
                        "prefix-aligned frozen residual control on "
                        f"{self.model_id}."
                    ),
                )
            )
        return tuple(scores)

    def _capture_control_summary(
        self,
        *,
        source_text: str,
        applied_control: tuple[float, ...],
        track_scale: tuple[float, ...],
    ) -> OpenWeightRuntimeCapture:
        """Capture only pooled control effects for internal RL rollouts.

        ``apply_control`` is called inside the internal-RL sandbox many times
        per real turn. Building a token-by-token residual sequence there is not
        consumed by the sandbox and can crash Windows CUDA while projecting
        large activation tensors. This path still runs the model with the
        control hook installed, but stores one pooled activation per hooked
        layer and no residual sequence.
        """

        effective_source = source_text.strip() or "<empty>"
        model_inputs = self._tokenize(source_text=effective_source)
        pooled_layers: dict[int, object] = {}

        def make_hook(layer_index: int):
            control_delta = self._build_control_delta(
                applied_control=applied_control,
                track_scale=track_scale,
                layer_index=layer_index,
            )

            def hook(module, args, output):
                del module
                del args
                hidden = self._extract_hidden_tensor(output=output)
                adapter_delta = self._adapter_delta_for_layer(layer_index=layer_index)
                adjusted = hidden
                if adapter_delta is not None:
                    adjusted = adjusted + adapter_delta.view(1, 1, -1).to(dtype=hidden.dtype)
                adjusted = adjusted + control_delta.view(1, 1, -1).to(dtype=hidden.dtype)
                pooled_layers[layer_index] = (
                    self._latest_token_control_activation(adjusted)
                )
                if isinstance(output, tuple):
                    return (adjusted, *output[1:])
                return adjusted

            return hook

        hooks = [
            self._block_modules[layer_index].register_forward_hook(
                make_hook(layer_index)
            )
            for layer_index in self._layer_indices
        ]
        try:
            with self._torch.no_grad():
                outputs = self._model(**model_inputs, use_cache=False)
        finally:
            for hook in hooks:
                hook.remove()

        if not pooled_layers:
            raise RuntimeError(
                f"Transformers runtime '{self.model_id}' recorded no pooled control activations."
            )
        logits = self._extract_logits(outputs=outputs)
        last_logits = logits[0, -1]
        probabilities = self._torch.softmax(last_logits, dim=-1)
        top_k = min(self._top_k_logits, int(probabilities.shape[-1]))
        top_values, _ = self._torch.topk(probabilities, k=top_k)
        token_logits = tuple(float(value) for value in top_values.detach().cpu().tolist())
        residual_activations = tuple(
            ResidualActivation(
                layer_index=layer_index,
                activation=self._tensor_to_activation_tuple(pooled_layers[layer_index]),
                step=0,
            )
            for layer_index in self._layer_indices
            if layer_index in pooled_layers
        )
        summary = _summarize_real_activations(residual_activations)
        feature_surface = (
            FeatureSignal(
                name="control_residual_mean_abs",
                values=(summary[0],),
                source="transformers-open-weight-control-summary",
            ),
            FeatureSignal(
                name="control_residual_peak_abs",
                values=(summary[1],),
                source="transformers-open-weight-control-summary",
            ),
            FeatureSignal(
                name="control_summary_layers",
                values=(_clamp_unit(len(residual_activations) / max(len(self._layer_indices), 1)),),
                source="transformers-open-weight-control-summary",
            ),
        )
        return OpenWeightRuntimeCapture(
            token_logits=token_logits,
            feature_surface=feature_surface,
            residual_activations=residual_activations,
            residual_sequence=(),
            description=(
                f"Transformers control-summary capture model={self.model_id} "
                f"device={self._device} layers={tuple(a.layer_index for a in residual_activations)}."
            ),
        )

    def _latest_token_control_activation(self, hidden):
        if hidden.dim() != 3 or hidden.shape[1] <= 0:
            raise ValueError(
                "control-summary hidden state must have shape "
                "[batch, sequence, hidden] with a nonempty sequence"
            )
        return (
            hidden.detach()
            .to(dtype=self._torch.float32)[:, -1, :]
            .mean(dim=0)
            .cpu()
        )

    def _capture_pooled_summary(self, *, source_text: str) -> OpenWeightRuntimeCapture:
        """Capture pooled substrate features without token-level sequences."""

        effective_source = source_text.strip() or "<empty>"
        model_inputs = self._tokenize(source_text=effective_source)
        pooled_layers: dict[int, object] = {}

        def make_hook(layer_index: int):
            def hook(module, args, output):
                del module
                del args
                hidden = self._extract_hidden_tensor(output=output)
                adapter_delta = self._adapter_delta_for_layer(layer_index=layer_index)
                adjusted = hidden
                if adapter_delta is not None:
                    adjusted = adjusted + adapter_delta.view(1, 1, -1).to(dtype=hidden.dtype)
                pooled_layers[layer_index] = adjusted.detach().to(
                    dtype=self._torch.float32
                ).mean(dim=1)[0].cpu()
                if adapter_delta is None:
                    return None
                if isinstance(output, tuple):
                    return (adjusted, *output[1:])
                return adjusted

            return hook

        hooks = [
            self._block_modules[layer_index].register_forward_hook(
                make_hook(layer_index)
            )
            for layer_index in self._layer_indices
        ]
        try:
            with self._torch.no_grad():
                outputs = self._model(**model_inputs, use_cache=False)
        finally:
            for hook in hooks:
                hook.remove()

        if not pooled_layers:
            raise RuntimeError(
                f"Transformers runtime '{self.model_id}' recorded no pooled activations."
            )
        logits = self._extract_logits(outputs=outputs)
        last_logits = logits[0, -1]
        probabilities = self._torch.softmax(last_logits, dim=-1)
        top_k = min(self._top_k_logits, int(probabilities.shape[-1]))
        top_values, _ = self._torch.topk(probabilities, k=top_k)
        token_logits = tuple(float(value) for value in top_values.detach().cpu().tolist())
        residual_activations = tuple(
            ResidualActivation(
                layer_index=layer_index,
                activation=self._tensor_to_activation_tuple(pooled_layers[layer_index]),
                step=0,
            )
            for layer_index in self._layer_indices
            if layer_index in pooled_layers
        )
        summary = _summarize_real_activations(residual_activations)
        feature_surface = (
            FeatureSignal(
                name="pooled_residual_mean_abs",
                values=(summary[0],),
                source="transformers-open-weight-pooled-summary",
            ),
            FeatureSignal(
                name="pooled_residual_peak_abs",
                values=(summary[1],),
                source="transformers-open-weight-pooled-summary",
            ),
            FeatureSignal(
                name="pooled_summary_layers",
                values=(_clamp_unit(len(residual_activations) / max(len(self._layer_indices), 1)),),
                source="transformers-open-weight-pooled-summary",
            ),
            FeatureSignal(
                name="residual_sequence_present",
                values=(0.0,),
                source="transformers-open-weight-pooled-summary",
            ),
        )
        return OpenWeightRuntimeCapture(
            token_logits=token_logits,
            feature_surface=feature_surface,
            residual_activations=residual_activations,
            residual_sequence=(),
            description=(
                f"Transformers pooled-summary capture model={self.model_id} "
                f"device={self._device} layers={tuple(a.layer_index for a in residual_activations)}."
            ),
        )

    def generate(
        self,
        *,
        prompt: str,
        system_context: str = "",
        chat_messages: tuple[tuple[str, str], ...] = (),
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        control_parameters: tuple[float, ...] = (),
        control_scale: float = 0.0,
        generation_constraints: "GenerationConstraints | None" = None,
        capture_residuals: bool = True,
        personal_conditioning: PersonalConditioningSnapshot | None = None,
        personal_conditioning_carrier: str = "residual",
        conditioning_bank_carriers: tuple[
            ConditioningBankLatentCarrier, ...
        ] = (),
        sampling_seed: int | None = None,
        character_id: str = "",
    ) -> GenerationResult:
        if sampling_seed is not None:
            if isinstance(sampling_seed, bool) or not isinstance(
                sampling_seed, int
            ):
                raise ValueError(
                    "sampling_seed must be an int or None, got "
                    f"{type(sampling_seed).__name__}."
                )
            if sampling_seed < 0:
                raise ValueError(
                    f"sampling_seed must be non-negative, got {sampling_seed}."
                )
        if personal_conditioning_carrier not in ("residual", "prefix_kv"):
            raise ValueError(
                "unknown personal_conditioning_carrier "
                f"{personal_conditioning_carrier!r}; expected 'residual' or "
                "'prefix_kv'."
            )
        if (
            personal_conditioning_carrier == "prefix_kv"
            and self._prefix_generator is None
        ):
            # Silently degrading to the residual carrier would publish an arm
            # labelled "prefix-KV" whose evidence came from a different
            # channel entirely (AGENTS §6: no silent fallback).
            raise ValueError(
                "personal_conditioning_carrier='prefix_kv' requires a prefix "
                "artifact; construct the runtime with "
                "personal_conditioning_prefix=..."
            )
        bank_types = tuple(
            carrier.bank.bank_type.value
            for carrier in conditioning_bank_carriers
        )
        if len(set(bank_types)) != len(bank_types):
            raise ValueError(
                "conditioning_bank_carriers must name each bank type at most "
                f"once, got {bank_types!r}."
            )
        effective_max_new_tokens = max_new_tokens
        effective_temperature = temperature
        effective_repetition_penalty = 1.08
        effective_top_p = 1.0
        if generation_constraints is not None:
            if generation_constraints.answer_depth_limit == "high-level-only":
                effective_max_new_tokens = min(effective_max_new_tokens, 192)
            elif generation_constraints.answer_depth_limit == "support-first":
                effective_max_new_tokens = min(effective_max_new_tokens, 256)
            if generation_constraints.response_mode in {"clarify", "refer-out"}:
                effective_temperature = min(effective_temperature, 0.45)
            (
                effective_max_new_tokens,
                effective_temperature,
                effective_repetition_penalty,
                effective_top_p,
            ) = self._apply_continuum_generation_controls(
                max_new_tokens=effective_max_new_tokens,
                temperature=effective_temperature,
                repetition_penalty=effective_repetition_penalty,
                top_p=effective_top_p,
                constraints=generation_constraints,
            )
        effective_prompt, model_inputs = self._build_generation_inputs(
            prompt=prompt,
            system_context=system_context,
            chat_messages=chat_messages,
        )
        input_ids = model_inputs["input_ids"]
        prompt_length = int(input_ids.shape[-1])

        control_active = bool(control_parameters and control_scale > 0)
        requested_character_id = character_id.strip()
        character_prefix_wiring = "disabled"
        character_prefix_shadow_id = ""
        selected_character_prefix_id = ""
        selected_character_prefix_pairs = self._character_prefix_pairs
        if self._character_prefix_registry is not None and requested_character_id:
            entry = self._character_prefix_registry.get(requested_character_id)
            if entry is not None:
                character_prefix_wiring = entry.wiring_level.value
                if entry.wiring_level.value == "active":
                    selected_character_prefix_pairs = (
                        self._character_prefix_pairs_by_character[
                            requested_character_id
                        ]
                    )
                    selected_character_prefix_id = entry.prefix_package.package_id
                else:
                    selected_character_prefix_pairs = None
                    character_prefix_shadow_id = entry.prefix_package.package_id
            else:
                selected_character_prefix_pairs = None
        elif selected_character_prefix_pairs is not None:
            character_prefix_wiring = "active"
            selected_character_prefix_id = self._character_prefix_id
            if not requested_character_id and self._character_prefix_package is not None:
                requested_character_id = self._character_prefix_package.character_id

        # Prefix order is stable and owner-auditable: character, Personal,
        # then admitted conditioning banks in request order.
        prefix_pairs = selected_character_prefix_pairs
        personal_prefix_pairs = None
        personal_delta = None
        if personal_conditioning_carrier == "prefix_kv":
            personal_prefix_pairs = self._build_personal_conditioning_prefix(
                conditioning=personal_conditioning
            )
            prefix_pairs = _concat_prefix_pairs(
                self._torch,
                prefix_pairs,
                personal_prefix_pairs,
            )
        else:
            personal_delta = self._build_personal_conditioning_delta(
                conditioning=personal_conditioning
            )
        bank_prefix_pairs = tuple(
            (
                carrier,
                self._build_relationship_conditioning_prefix(carrier=carrier),
            )
            for carrier in conditioning_bank_carriers
            if carrier.carrier == "prefix_kv"
        )
        for _, relationship_prefix_pairs in bank_prefix_pairs:
            prefix_pairs = _concat_prefix_pairs(
                self._torch,
                prefix_pairs,
                relationship_prefix_pairs,
            )
        bank_delta_pairs = tuple(
            (carrier, delta)
            for carrier in conditioning_bank_carriers
            if carrier.carrier == "residual"
            if (
                delta := build_conditioning_bank_residual_delta(
                    torch_module=self._torch,
                    carrier=carrier,
                    hidden_size=self._hidden_size,
                    device=self._device,
                    basis=self._relationship_conditioning_basis,
                    vector_labels=(
                        self._relationship_conditioning_vector_labels
                    ),
                    expected_projector_version=(
                        self._relationship_conditioning_projector_version
                    ),
                )
            )
            is not None
        )
        bank_delta_by_layer = {
            layer_index: sum(
                (
                    bank_delta
                    * self._relationship_conditioning_layer_gains.get(
                        layer_index,
                        0.0,
                    )
                    for _, bank_delta in bank_delta_pairs
                ),
                start=self._torch.zeros(
                    self._hidden_size,
                    dtype=self._torch.float32,
                    device=self._device,
                ),
            )
            for layer_index in self._layer_indices
            if bank_delta_pairs
            and self._relationship_conditioning_layer_gains.get(
                layer_index,
                0.0,
            )
            > 0.0
        }
        captured_layers: dict[int, object] = {}
        # ``capture_residuals=False`` (raw pass-through path) skips both the
        # forward hooks and the post-generate full-prompt re-forward that
        # builds the runtime capture. That re-forward (line ~622 below) runs
        # a second full attention pass over the *entire* prompt with all
        # target layers' hidden states materialised — a memory spike that
        # scales with prompt length and OOM/native-crashes the process on
        # long multi-turn contexts. The raw ablation track never reads the
        # capture, so skipping it is both correct and the memory fix.
        has_runtime_deltas = bool(
            getattr(self, "_online_fast_adapter_deltas", {})
            or getattr(self, "_rare_heavy_adapter_deltas", {})
        )
        has_character_residual = bool(self._character_residual_deltas)
        hook_required = (
            capture_residuals
            or control_active
            or personal_delta is not None
            or bool(bank_delta_by_layer)
            or has_runtime_deltas
            or has_character_residual
        )
        hooks = (
            [
                self._block_modules[layer_index].register_forward_hook(
                    self._make_capture_hook(
                        layer_index=layer_index,
                        captured_layers=captured_layers,
                        control_delta=(
                            self._build_control_delta(
                                applied_control=control_parameters,
                                track_scale=(control_scale,),
                                layer_index=layer_index,
                            )
                            if control_active
                            else None
                        ),
                        capture_residuals=capture_residuals,
                        personal_delta=personal_delta,
                        conditioning_bank_delta=bank_delta_by_layer.get(
                            layer_index
                        ),
                        character_residual_delta=self._character_residual_deltas.get(
                            layer_index
                        ),
                    )
                )
                for layer_index in self._layer_indices
            ]
            if hook_required
            else []
        )
        prefix_carrier_active = prefix_pairs is not None
        try:
            with self._torch.no_grad():
                generate_kwargs: dict[str, object] = {
                    "max_new_tokens": effective_max_new_tokens,
                    "do_sample": effective_temperature > 0,
                    "pad_token_id": getattr(self._tokenizer, "eos_token_id", 0) or 0,
                    "eos_token_id": self._generation_eos_token_id(),
                    "repetition_penalty": effective_repetition_penalty,
                }
                if os.name == "nt":
                    # Transformers emits a deprecated tuple-KV-cache warning
                    # on this stack, and repeated Windows generations can
                    # terminate the process with 0xC0000005 on both CUDA and
                    # CPU before Python can raise. Disabling cache is slower
                    # but keeps the frozen substrate path stable for long arcs
                    # and bootstrap calibration.
                    generate_kwargs["use_cache"] = False
                if effective_temperature > 0:
                    generate_kwargs["temperature"] = effective_temperature
                    if effective_top_p < 0.999:
                        generate_kwargs["top_p"] = effective_top_p
                if prefix_carrier_active:
                    output_ids = self._generate_with_prefix(
                        model_inputs=model_inputs,
                        prefix_pairs=prefix_pairs,
                        max_new_tokens=effective_max_new_tokens,
                        repetition_penalty=effective_repetition_penalty,
                        temperature=effective_temperature,
                        top_p=effective_top_p,
                        sampling_seed=sampling_seed,
                        require_sampling_seed=(
                            personal_conditioning_carrier == "prefix_kv"
                            or bool(bank_prefix_pairs)
                        ),
                    )
                else:
                    with self._temporary_torch_seed(sampling_seed):
                        output_ids = self._model.generate(
                            **model_inputs, **generate_kwargs
                        )
        finally:
            for hook in hooks:
                hook.remove()

        new_token_ids = output_ids[0, prompt_length:]
        generated_text = self._decode_generated_text(token_ids=new_token_ids)
        if generation_constraints is not None:
            generated_text = self._apply_generation_constraints(
                text=generated_text,
                constraints=generation_constraints,
            )
        token_count = int(new_token_ids.shape[0])

        capture = None
        if captured_layers:
            try:
                logits_pass = self._model(
                    output_ids[:, :prompt_length],
                    use_cache=False,
                )
                logits = self._extract_logits(outputs=logits_pass)
                capture = self._build_runtime_capture(
                    source_text=effective_prompt,
                    input_ids=input_ids,
                    logits=logits,
                    captured_layers=captured_layers,
                    control_applied=(
                        control_active
                        or personal_delta is not None
                        or bool(bank_delta_by_layer)
                    ),
                )
            except (RuntimeError, ValueError, AttributeError, IndexError) as exc:
                _LOG.warning(
                    "residual capture failed (model=%s device=%s): %r",
                    self.model_id,
                    self._device,
                    exc,
                )

        if str(self._device).startswith("cuda"):
            cuda = getattr(self._torch, "cuda", None)
            if cuda is not None and cuda.is_available():
                cuda.empty_cache()

        # The delta is applied by forward hooks, so injection only actually
        # happened when both the delta was built (non-cold-start, positive
        # confidence) and the hooks were registered. The prefix carrier does
        # not use hooks: it reports injection when the state-derived key/value
        # slots were actually prepended to the attention cache.
        if personal_conditioning_carrier == "prefix_kv":
            personal_conditioning_applied = personal_prefix_pairs is not None
        else:
            personal_conditioning_applied = personal_delta is not None and bool(
                hooks
            )
        residual_bank_carriers_applied = (
            tuple(
                (
                    carrier.bank.bank_type.value,
                    carrier.projector_version,
                )
                for carrier in conditioning_bank_carriers
                if any(
                    applied_carrier is carrier
                    for applied_carrier, _ in bank_delta_pairs
                )
            )
            if bank_delta_pairs and hooks
            else ()
        )
        prefix_bank_carriers_applied = tuple(
            (
                carrier.bank.bank_type.value,
                carrier.projector_version,
            )
            for carrier, _ in bank_prefix_pairs
        )
        conditioning_bank_carriers_applied = (
            *residual_bank_carriers_applied,
            *prefix_bank_carriers_applied,
        )
        result = GenerationResult(
            text=generated_text,
            token_count=token_count,
            capture=capture,
            description=(
                f"Generated {token_count} tokens from {self.model_id} "
                f"device={self._device} temp={effective_temperature} "
                f"profile={generation_constraints.decoding_profile if generation_constraints is not None else 'balanced'} "
                f"control={'on' if control_active else 'off'} "
                "personal_conditioning="
                f"{'on' if personal_conditioning_applied else 'off'} "
                f"conditioning_banks={bank_types!r}"
            ),
            personal_conditioning_applied=personal_conditioning_applied,
            conditioning_bank_carriers_applied=(
                conditioning_bank_carriers_applied
            ),
            character_prefix_applied=selected_character_prefix_pairs is not None,
            character_prefix_id=selected_character_prefix_id,
            character_id=requested_character_id,
            character_prefix_wiring_level=character_prefix_wiring,
            character_prefix_shadow_id=character_prefix_shadow_id,
            character_residual_applied=bool(self._character_residual_deltas and hooks),
            character_residual_adapter_id=self._character_residual_adapter_id,
        )
        if str(self._device).startswith("mps"):
            # MPS uses unified memory and retains released generation buffers
            # in its allocator cache. Long multi-turn evidence runs otherwise
            # grow until macOS jetsam kills the shared substrate process.
            del new_token_ids
            del output_ids
            del input_ids
            del model_inputs
            self._release_mps_generation_cache()
        return result

    def _apply_generation_constraints(
        self,
        *,
        text: str,
        constraints: "GenerationConstraints",
    ) -> str:
        compact = text.strip()
        if not compact:
            return compact
        question_budget = (
            constraints.question_budget
            if constraints.question_budget is not None
            else constraints.max_questions
        )
        question_budget = min(constraints.max_questions, question_budget)
        if question_budget <= 0:
            compact = self._remove_question_tail(compact)
        elif question_budget < constraints.max_questions:
            compact = self._limit_questions(compact, max_questions=question_budget)
        for phrase in constraints.required_disclaimer_phrases:
            if phrase and phrase not in compact:
                compact = f"{compact} {phrase}".strip()
        if constraints.ordering_driver in {"continuum-support-first", "continuum-support-clarify"}:
            compact = self._support_first_trim(compact)
        elif constraints.ordering_driver == "continuum-structure-first":
            compact = self._structure_first_trim(compact)
        if constraints.ordering_bias and len(compact.split()) > 80:
            compact = compact[:320].rstrip()
        return compact

    def _remove_question_tail(self, text: str) -> str:
        compact = text.strip()
        if not compact:
            return compact
        question_match = re.search(r"[?？]", compact)
        if question_match is None:
            return compact
        question_sentence_start = max(
            compact.rfind(mark, 0, question_match.start()) for mark in ".。!！\n"
        )
        trim_at = question_sentence_start + 1 if question_sentence_start >= 0 else question_match.start()
        trimmed = compact[:trim_at].rstrip(" ，,;；:：")
        if trimmed:
            return trimmed.rstrip()
        return compact.replace("?", "").replace("？", "").strip()

    def _limit_questions(self, text: str, *, max_questions: int) -> str:
        question_count = 0
        truncated_chars: list[str] = []
        for char in text:
            if char in {"?", "？"}:
                question_count += 1
                if question_count > max_questions:
                    continue
            truncated_chars.append(char)
        return "".join(truncated_chars).strip()

    def _apply_continuum_generation_controls(
        self,
        *,
        max_new_tokens: int,
        temperature: float,
        repetition_penalty: float,
        top_p: float,
        constraints: "GenerationConstraints",
    ) -> tuple[int, float, float, float]:
        target = constraints.continuum_target_position
        effective_max_new_tokens = max_new_tokens
        effective_temperature = temperature
        effective_repetition_penalty = repetition_penalty
        effective_top_p = top_p
        if constraints.decoding_profile == "support-first":
            effective_max_new_tokens = min(effective_max_new_tokens, 224)
            effective_temperature = min(effective_temperature, 0.42)
            effective_repetition_penalty = max(effective_repetition_penalty, 1.04)
            effective_top_p = min(effective_top_p, 0.92)
        elif constraints.decoding_profile == "clarify-first":
            effective_max_new_tokens = min(effective_max_new_tokens, 168)
            effective_temperature = min(effective_temperature, 0.34)
            effective_repetition_penalty = max(effective_repetition_penalty, 1.06)
            effective_top_p = min(effective_top_p, 0.82)
        elif constraints.decoding_profile == "structure-first":
            effective_max_new_tokens = min(effective_max_new_tokens, 352)
            effective_temperature = min(effective_temperature, 0.28)
            effective_repetition_penalty = max(effective_repetition_penalty, 1.10)
            effective_top_p = min(effective_top_p, 0.74)

        if target >= 0.75:
            effective_temperature = min(effective_temperature, 0.40)
            effective_max_new_tokens = min(effective_max_new_tokens, 208)
        elif target < 0.42:
            effective_max_new_tokens = min(effective_max_new_tokens, 336)
            effective_repetition_penalty = max(effective_repetition_penalty, 1.09)

        return (
            effective_max_new_tokens,
            effective_temperature,
            effective_repetition_penalty,
            effective_top_p,
        )

    def _support_first_trim(self, text: str) -> str:
        sentences = [part.strip() for part in text.replace("\n", " ").split(".") if part.strip()]
        if not sentences:
            return text
        compact = ". ".join(sentences[:2]).strip()
        if not text.strip().endswith("?") and compact and not compact.endswith("."):
            compact += "."
        if len(compact) > 160:
            compact = compact[:160].rstrip(". ").rstrip()
        return compact

    def _structure_first_trim(self, text: str) -> str:
        compact = text.strip()
        if not compact:
            return compact
        return compact[:420].rstrip()

    def _build_generation_inputs(
        self,
        *,
        prompt: str,
        system_context: str,
        chat_messages: tuple[tuple[str, str], ...],
    ) -> tuple[str, dict[str, object]]:
        source_text = self._chat_messages_to_source_text(
            prompt=prompt,
            system_context=system_context,
            chat_messages=chat_messages,
        )
        if chat_messages:
            apply_chat_template = getattr(self._tokenizer, "apply_chat_template", None)
            if callable(apply_chat_template):
                chat_payload = [
                    {
                        "role": role,
                        "content": content,
                    }
                    for role, content in chat_messages
                ]
                try:
                    encoded = apply_chat_template(
                        chat_payload,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                    )
                except TypeError:
                    encoded = None
                except ValueError:
                    # Tokenizers without a configured chat_template
                    # (e.g. ``sshleifer/tiny-gpt2`` used in CI smoke
                    # tests) raise ValueError instead of returning
                    # gracefully. Drop through to the ROLE-prefix
                    # fallback below — it produces a valid
                    # text-completion prompt without needing a
                    # template.
                    encoded = None
                    apply_chat_template = None
                if encoded is not None:
                    return source_text, self._prepare_model_inputs(encoded=encoded)
                if callable(apply_chat_template):
                    try:
                        rendered = apply_chat_template(
                            chat_payload,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    except ValueError:
                        rendered = None
                    if isinstance(rendered, str) and rendered.strip():
                        return rendered.strip(), self._tokenize(source_text=rendered.strip())
            fallback_sections = [f"{role.upper()}:\n{content}" for role, content in chat_messages if content.strip()]
            fallback_sections.append("ASSISTANT:\n")
            rendered_fallback = "\n\n".join(fallback_sections).strip()
            if rendered_fallback:
                return rendered_fallback, self._tokenize(source_text=rendered_fallback)
        return source_text, self._tokenize(source_text=source_text)

    def _chat_messages_to_source_text(
        self,
        *,
        prompt: str,
        system_context: str,
        chat_messages: tuple[tuple[str, str], ...],
    ) -> str:
        if chat_messages:
            rendered_messages = [
                f"{role}: {content}"
                for role, content in chat_messages
                if content.strip()
            ]
            return "\n".join(rendered_messages).strip() or "<empty>"
        full_prompt = f"{system_context}\n{prompt}".strip() if system_context else prompt.strip()
        return full_prompt or "<empty>"

    def _prepare_model_inputs(self, *, encoded) -> dict[str, object]:
        model_inputs: dict[str, object] = {}
        for key, value in encoded.items():
            if isinstance(value, self._torch.Tensor):
                if (
                    str(self._device).startswith("mps")
                    and value.ndim >= 2
                    and value.shape[-1] > self._mps_generation_max_input_tokens
                ):
                    # Keep the most recent context. This cap is intentionally
                    # MPS-only: CUDA evidence hosts retain their existing full
                    # context behaviour and memory envelope.
                    value = value[..., -self._mps_generation_max_input_tokens :]
                model_inputs[key] = value.to(self._device)
            else:
                model_inputs[key] = value
        input_ids = model_inputs.get("input_ids")
        if not isinstance(input_ids, self._torch.Tensor):
            raise ValueError(f"Transformers runtime '{self.model_id}' chat template did not return tensor input_ids.")
        return model_inputs

    def _release_mps_generation_cache(self) -> None:
        if not str(self._device).startswith("mps"):
            return
        mps = getattr(self._torch, "mps", None)
        if mps is None or not mps.is_available():
            return
        mps.synchronize()
        mps.empty_cache()

    def _generation_eos_token_id(self) -> int | list[int]:
        token_ids: list[int] = []
        eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, int) and eos_token_id >= 0:
            token_ids.append(eos_token_id)
        elif isinstance(eos_token_id, (list, tuple)):
            token_ids.extend(token_id for token_id in eos_token_id if isinstance(token_id, int) and token_id >= 0)
        convert_tokens_to_ids = getattr(self._tokenizer, "convert_tokens_to_ids", None)
        if callable(convert_tokens_to_ids):
            for token in ("<|im_end|>", "<|eot_id|>"):
                token_id = convert_tokens_to_ids(token)
                if isinstance(token_id, int) and token_id >= 0:
                    token_ids.append(token_id)
        unique_ids = list(dict.fromkeys(token_ids))
        if not unique_ids:
            return 0
        if len(unique_ids) == 1:
            return unique_ids[0]
        return unique_ids

    def _load_tokenizer(self, *, model_id: str, local_files_only: bool):
        try:
            return self._transformers.AutoTokenizer.from_pretrained(
                model_id,
                local_files_only=local_files_only,
            )
        except Exception as first_exc:
            if not local_files_only:
                raise
            try:
                return self._transformers.AutoTokenizer.from_pretrained(
                    model_id,
                    local_files_only=True,
                    use_fast=False,
                )
            except Exception:
                raise first_exc

    def _load_model(self, *, model_id: str, local_files_only: bool):
        load_kwargs: dict[str, object] = {"local_files_only": local_files_only}
        if self._device == "mps":
            load_kwargs["torch_dtype"] = self._torch.float16
        if os.name == "nt" and str(self._device).startswith("cuda"):
            # Windows CUDA has been observed to terminate the interpreter with
            # 0xC0000005 under repeated SDPA-backed generation on long
            # OpenAI-compat arcs. Eager attention is slower but stays on the
            # conservative transformers path and lets Python own failures.
            load_kwargs["attn_implementation"] = "eager"
        return self._transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            **load_kwargs,
        )

    def _resolve_device(self, *, device: str) -> str:
        if device != "auto":
            return device
        if self._torch.cuda.is_available():
            return "cuda"
        if self._mps_is_available():
            return "mps"
        return "cpu"

    def _mps_is_available(self) -> bool:
        try:
            mps_backend = self._torch.backends.mps
        except AttributeError:
            return False
        try:
            return bool(mps_backend.is_available())
        except (AttributeError, RuntimeError):
            return False

    def _prepare_model(self) -> None:
        self._model.to(self._device)
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)

    def _resolve_transformer_blocks(self) -> tuple[object, ...]:
        candidate_paths = (
            ("model", "layers"),
            ("base_model", "model", "layers"),
            ("base_model", "layers"),
            ("language_model", "model", "layers"),
            ("language_model", "base_model", "model", "layers"),
            ("model", "decoder", "layers"),
            ("decoder", "layers"),
            ("transformer", "h"),
            ("base_model", "transformer", "h"),
            ("gpt_neox", "layers"),
            ("transformer", "blocks"),
            ("backbone", "layers"),
            ("layers",),
        )
        for path in candidate_paths:
            resolved = self._resolve_module_path(path)
            if resolved is not None:
                return resolved
        raise NotImplementedError(
            f"Transformers runtime '{self.model_id}' could not resolve transformer blocks for hook capture."
        )

    def _normalize_layer_indices(
        self,
        *,
        requested: tuple[int, ...] | None,
        block_count: int,
        hook_layer_selection: str = "middle",
    ) -> tuple[int, ...]:
        if block_count <= 0:
            raise ValueError(f"Transformers runtime '{self.model_id}' has no hookable transformer blocks.")
        if requested is not None:
            normalized = tuple(sorted({index for index in requested if 0 <= index < block_count}))
            if not normalized:
                raise ValueError(f"Transformers runtime '{self.model_id}' received no valid hook layer indices.")
            return normalized
        if hook_layer_selection == "all":
            return tuple(range(block_count))
        if hook_layer_selection != "middle":
            raise ValueError(
                f"Unsupported hook_layer_selection {hook_layer_selection!r}; expected 'middle' or 'all'."
            )
        if block_count <= 3:
            return tuple(range(block_count))
        middle = block_count // 2
        return tuple(sorted({middle - 1, middle, min(block_count - 1, middle + 1)}))

    def _resolve_hidden_size(self) -> int:
        try:
            return int(self._model.config.hidden_size)
        except AttributeError:
            pass
        try:
            return int(self._model.config.n_embd)
        except AttributeError:
            pass
        try:
            return int(self._model.config.d_model)
        except AttributeError as exc:
            raise AttributeError(
                f"Transformers runtime '{self.model_id}' could not resolve hidden size from model config."
            ) from exc

    def _resolve_model_family(self) -> str:
        model_type = getattr(self._model.config, "model_type", None)
        if isinstance(model_type, str) and model_type:
            return model_type
        return type(self._model).__name__

    def _resolve_module_path(self, path: tuple[str, ...]) -> tuple[object, ...] | None:
        current = self._model
        for segment in path:
            try:
                current = getattr(current, segment)
            except AttributeError:
                return None
        return self._as_module_tuple(current)

    def _as_module_tuple(self, container: object) -> tuple[object, ...] | None:
        try:
            resolved = tuple(container)  # type: ignore[arg-type]
        except TypeError:
            return None
        return resolved if resolved else None

    @property
    def control_basis_provenance(self) -> str:
        """Provenance tag of the active control basis (manifest surface)."""

        return self._control_basis_provenance

    @property
    def control_basis_rank(self) -> int:
        return int(self._control_basis.shape[0])

    def install_control_basis(
        self,
        *,
        basis: tuple[tuple[float, ...], ...],
        provenance: str,
        layer_indices: tuple[int, ...] | None = None,
        layer_gains: tuple[float, ...] | None = None,
    ) -> None:
        """Replace the fixed sinusoid control basis with a learned artifact.

        The basis is a rare-heavy offline artifact (e.g. fit by
        ``volvence_zero.substrate.control_basis.fit_transition_control_basis``
        from frozen-model transition captures). Installation only rotates
        the directions in which bounded ``applied_control`` vectors can
        perturb the hidden state; it does not change model weights, the
        control scale clamp, or capture semantics. Rows are re-normalized
        to unit norm so control-scale semantics stay comparable with the
        sinusoid default.
        """

        if not provenance.strip():
            raise ValueError("install_control_basis requires a non-empty provenance tag")
        if not basis:
            raise ValueError(
                "control basis must contain at least one row"
            )
        tensor = self._torch.tensor(basis, dtype=self._torch.float32)
        if tensor.ndim != 2 or int(tensor.shape[1]) != self._hidden_size:
            raise ValueError(
                "control basis rows must match the substrate hidden size "
                f"{self._hidden_size}, got shape {tuple(tensor.shape)!r}"
            )
        if not bool(self._torch.isfinite(tensor).all()):
            raise ValueError("control basis contains non-finite values")
        norms = tensor.norm(dim=1)
        if bool((norms < 1e-6).any()):
            raise ValueError("control basis contains a degenerate (near-zero) row")
        tensor = tensor / norms.unsqueeze(1)
        target_layers = (
            tuple(self._layer_indices)
            if layer_indices is None
            else tuple(layer_indices)
        )
        if not target_layers or len(set(target_layers)) != len(target_layers):
            raise ValueError(
                "control basis layer_indices must be non-empty and unique"
            )
        unavailable = sorted(set(target_layers) - set(self._layer_indices))
        if unavailable:
            raise ValueError(
                "control basis targets layers not hooked by this runtime: "
                f"{unavailable}"
            )
        gains = (
            tuple(1.0 for _ in target_layers)
            if layer_gains is None
            else tuple(float(value) for value in layer_gains)
        )
        if len(gains) != len(target_layers):
            raise ValueError(
                "control basis layer_gains must align with layer_indices"
            )
        if any(not 0.0 < gain <= 1.0 for gain in gains):
            raise ValueError("control basis layer gains must be in (0, 1]")
        self._control_basis = tensor.to(self._device)
        self._control_basis_provenance = provenance
        self._control_layer_gains = {
            layer_index: gain
            for layer_index, gain in zip(
                target_layers,
                gains,
                strict=True,
            )
        }

    def _build_control_basis(self, *, hidden_size: int):
        positions = self._torch.arange(hidden_size, dtype=self._torch.float32)
        rows = []
        for factor in (1.0, 2.0, 3.0):
            row = self._torch.sin((positions + 1.0) * 0.173 * factor) + self._torch.cos(
                (positions + 1.0) * 0.117 * (factor + 1.0)
            )
            row = row / row.norm().clamp_min(1e-6)
            rows.append(row)
        return self._torch.stack(rows, dim=0).to(self._device)

    def _build_personal_conditioning_basis(
        self, *, hidden_size: int, vector_dim: int
    ):
        return build_personal_conditioning_basis(
            torch_module=self._torch,
            hidden_size=hidden_size,
            vector_dim=vector_dim,
            device=self._device,
        )

    def _build_semantic_basis(self, *, hidden_size: int, projection_dim: int):
        positions = self._torch.arange(hidden_size, dtype=self._torch.float32)
        rows = []
        for factor in range(1, projection_dim + 1):
            row = self._torch.sin((positions + 1.0) * 0.071 * factor) + self._torch.cos(
                (positions + 1.0) * 0.043 * (factor + 1.0)
            )
            row = row / row.norm().clamp_min(1e-6)
            rows.append(row)
        return self._torch.stack(rows, dim=0).to(self._device)

    def _base_semantic_weights(self) -> tuple[float, float]:
        if self._runtime_origin == "builtin-fallback":
            return (0.9, 0.1)
        return (0.55, 0.45)

    def _semantic_profile_from_capture(self, *, source_text: str, captured_layers: dict[int, object]) -> tuple[float, ...]:
        text_profile = _hashed_semantic_embedding(source_text, dim=self._semantic_projection_dim)
        residual_profile = self._residual_semantic_profile(captured_layers=captured_layers)
        text_weight, residual_weight = _normalize_semantic_weights(
            text_weight=self._rare_heavy_semantic_text_weight,
            residual_weight=self._rare_heavy_semantic_residual_weight,
        )
        combined = tuple(
            text_value * text_weight + residual_value * residual_weight
            for text_value, residual_value in zip(text_profile, residual_profile, strict=True)
        )
        return _normalize_vector(combined)

    def _residual_semantic_profile(self, *, captured_layers: dict[int, object]) -> tuple[float, ...]:
        stacked = self._torch.stack(
            [captured_layers[layer_index][0].to(self._device, dtype=self._torch.float32) for layer_index in self._layer_indices],
            dim=0,
        )
        mean_hidden = stacked.mean(dim=(0, 1))
        tail_hidden = stacked[:, -1, :].mean(dim=0)
        dispersion_hidden = stacked.std(dim=1).mean(dim=0) if stacked.shape[1] > 1 else self._torch.zeros_like(mean_hidden)
        composite = mean_hidden * 0.55 + tail_hidden * 0.30 + dispersion_hidden * 0.15
        projected = self._semantic_basis.to(dtype=self._torch.float32) @ composite.to(dtype=self._torch.float32)
        norm = projected.norm().clamp_min(1e-6)
        normalized = (projected / norm).detach().cpu().tolist()
        return tuple(float(value) for value in normalized)

    def _semantic_feature_surface(
        self,
        *,
        source_text: str,
        captured_layers: dict[int, object],
    ) -> tuple[FeatureSignal, ...]:
        profile = self._semantic_profile_from_capture(
            source_text=source_text,
            captured_layers=captured_layers,
        )
        similarities = {
            name: _cosine_similarity(profile, anchor_profile)
            for name, anchor_profile in self._semantic_anchor_profiles.items()
        }
        centered_similarities = {
            name: similarities[name] - (sum(similarities.values()) / max(len(similarities), 1))
            for name in similarities
        }
        distribution = {
            name: probability
            for name, probability in zip(
                similarities.keys(),
                _softmax_probabilities(tuple(centered_similarities.values()), temperature=0.22),
                strict=True,
            )
        }

        def relative_pull(target_name: str) -> float:
            target = similarities[target_name]
            others = [value for name, value in similarities.items() if name != target_name]
            runner_up = max(others) if others else 0.0
            absolute = _clamp_unit((target + 1.0) / 2.0)
            margin = _clamp_unit(0.5 + (target - runner_up) * 3.2)
            return _clamp_unit(
                distribution[target_name] * 0.65
                + margin * 0.25
                + absolute * 0.10
            )

        raw_task_pull = relative_pull("task")
        raw_support_pull = relative_pull("support")
        raw_repair_pull = relative_pull("repair")
        raw_exploration_pull = relative_pull("exploration")
        raw_directive_pull = relative_pull("directive")
        raw_task_pull = _clamp_unit(raw_task_pull + self._rare_heavy_anchor_bias[0])
        raw_support_pull = _clamp_unit(raw_support_pull + self._rare_heavy_anchor_bias[1])
        raw_repair_pull = _clamp_unit(raw_repair_pull + self._rare_heavy_anchor_bias[2])
        raw_exploration_pull = _clamp_unit(raw_exploration_pull + self._rare_heavy_anchor_bias[3])
        raw_directive_pull = _clamp_unit(raw_directive_pull + self._rare_heavy_anchor_bias[4])
        semantic_task_pull = _clamp_unit(raw_task_pull * 0.35 + raw_directive_pull * 0.65)
        semantic_support_pull = _clamp_unit(raw_support_pull * 0.75 + raw_repair_pull * 0.25)
        semantic_repair_pull = _clamp_unit(raw_repair_pull * 0.80 + raw_support_pull * 0.20)
        text_weight, residual_weight = _normalize_semantic_weights(
            text_weight=self._rare_heavy_semantic_text_weight,
            residual_weight=self._rare_heavy_semantic_residual_weight,
        )

        return (
            FeatureSignal(
                name="semantic_task_pull",
                values=(semantic_task_pull,),
                source="transformers-open-weight-semantic",
            ),
            FeatureSignal(
                name="semantic_support_pull",
                values=(semantic_support_pull,),
                source="transformers-open-weight-semantic",
            ),
            FeatureSignal(
                name="semantic_repair_pull",
                values=(semantic_repair_pull,),
                source="transformers-open-weight-semantic",
            ),
            FeatureSignal(
                name="semantic_exploration_pull",
                values=(raw_exploration_pull,),
                source="transformers-open-weight-semantic",
            ),
            FeatureSignal(
                name="semantic_directive_pull",
                values=(raw_directive_pull,),
                source="transformers-open-weight-semantic",
            ),
            FeatureSignal(
                name="semantic_text_weight",
                values=(text_weight,),
                source="transformers-open-weight-semantic",
            ),
            FeatureSignal(
                name="semantic_residual_weight",
                values=(residual_weight,),
                source="transformers-open-weight-semantic",
            ),
            FeatureSignal(
                name="substrate_rare_heavy_update_count",
                values=(_clamp_unit(self._rare_heavy_update_count / 10.0),),
                source="transformers-open-weight-semantic",
            ),
            FeatureSignal(
                name="substrate_delta_parameter_count",
                values=(_clamp_unit(len(self._rare_heavy_adapter_deltas) * self._hidden_size / 512.0),),
                source="transformers-open-weight-semantic",
            ),
            FeatureSignal(
                name="substrate_online_fast_update_count",
                values=(_clamp_unit(self._online_fast_update_count / 10.0),),
                source="transformers-open-weight-semantic",
            ),
            FeatureSignal(
                name="substrate_online_fast_delta_parameter_count",
                values=(_clamp_unit(len(self._online_fast_adapter_deltas) * self._hidden_size / 512.0),),
                source="transformers-open-weight-semantic",
            ),
            FeatureSignal(
                name="substrate_online_fast_parameter_change_rate",
                values=(_clamp_unit(self._online_fast_parameter_change_rate),),
                source="transformers-open-weight-semantic",
            ),
            FeatureSignal(
                name="substrate_online_fast_experimental_mode",
                values=(1.0 if self.experimental_live_mutation_enabled else 0.0,),
                source="transformers-open-weight-semantic",
            ),
        )

    def _capture_with_hooks(
        self,
        *,
        source_text: str,
        applied_control: tuple[float, ...] | None = None,
        track_scale: tuple[float, ...] = (1.0, 1.0, 1.0),
    ) -> OpenWeightRuntimeCapture:
        effective_source = source_text.strip() or "<empty>"
        model_inputs = self._tokenize(source_text=effective_source)
        input_ids = model_inputs["input_ids"]
        captured_layers: dict[int, object] = {}
        hooks = [
            self._block_modules[layer_index].register_forward_hook(
                self._make_capture_hook(
                    layer_index=layer_index,
                    captured_layers=captured_layers,
                    control_delta=(
                        self._build_control_delta(
                            applied_control=applied_control,
                            track_scale=track_scale,
                            layer_index=layer_index,
                        )
                        if applied_control is not None
                        else None
                    ),
                )
            )
            for layer_index in self._layer_indices
        ]
        try:
            with self._torch.no_grad():
                outputs = self._model(**model_inputs, use_cache=False)
        finally:
            for hook in hooks:
                hook.remove()
        logits = self._extract_logits(outputs=outputs)
        return self._build_runtime_capture(
            source_text=effective_source,
            input_ids=input_ids,
            logits=logits,
            captured_layers=captured_layers,
            control_applied=applied_control is not None,
        )

    def _capture_hidden_state_means(
        self,
        *,
        source_text: str,
    ) -> dict[int, object]:
        effective_source = source_text.strip() or "<empty>"
        model_inputs = self._tokenize(source_text=effective_source)
        captured_layers: dict[int, object] = {}

        def make_hook(layer_index: int):
            def hook(module, args, output):
                del module
                del args
                captured_layers[layer_index] = self._extract_hidden_tensor(output=output).detach().cpu()
                return None

            return hook

        hooks = [
            self._block_modules[layer_index].register_forward_hook(make_hook(layer_index))
            for layer_index in self._layer_indices
        ]
        try:
            with self._torch.no_grad():
                self._model(**model_inputs, use_cache=False)
        finally:
            for hook in hooks:
                hook.remove()
        return {
            layer_index: captured_layers[layer_index][0].to(self._device, dtype=self._torch.float32).mean(dim=0)
            for layer_index in self._layer_indices
            if layer_index in captured_layers
        }

    def _target_semantic_profile_tensor(
        self,
        *,
        substrates: Sequence[SubstrateSnapshot],
        source_text: str,
    ):
        feature_names = {
            "task": "semantic_task_pull",
            "support": "semantic_support_pull",
            "repair": "semantic_repair_pull",
            "exploration": "semantic_exploration_pull",
            "directive": "semantic_directive_pull",
        }
        weights = tuple(_mean_feature_value(substrates, name=feature_names[anchor]) for anchor in RARE_HEAVY_ANCHOR_ORDER)
        if any(weight > 1e-6 for weight in weights):
            target = self._torch.zeros(self._semantic_projection_dim, dtype=self._torch.float32, device=self._device)
            for anchor, weight in zip(RARE_HEAVY_ANCHOR_ORDER, weights, strict=True):
                anchor_profile = self._torch.tensor(
                    self._semantic_anchor_profiles[anchor],
                    dtype=self._torch.float32,
                    device=self._device,
                )
                target = target + anchor_profile * float(weight)
            return target / target.norm().clamp_min(1e-6)
        return self._torch.tensor(
            _hashed_semantic_embedding(source_text, dim=self._semantic_projection_dim),
            dtype=self._torch.float32,
            device=self._device,
        )

    def _target_residual_tensor(
        self,
        *,
        substrates: Sequence[SubstrateSnapshot],
    ):
        return self._torch.tensor(
            max(
                _mean_residual_magnitude(substrates),
                _mean_feature_value(substrates, name="residual_mean_abs"),
            ),
            dtype=self._torch.float32,
            device=self._device,
        )

    def _train_adapter_deltas(
        self,
        *,
        traces: tuple[TrainingTrace, ...],
        substrate_steps_per_trace: tuple[tuple[SubstrateSnapshot, ...], ...],
    ) -> tuple[tuple[SubstrateDeltaAdapterLayer, ...], float]:
        if not traces or not substrate_steps_per_trace:
            return ((), 0.0)
        paired = tuple(zip(traces, substrate_steps_per_trace))
        if not paired:
            return ((), 0.0)
        parameters = {
            layer_index: self._torch.nn.Parameter(
                self._rare_heavy_adapter_deltas.get(
                    layer_index,
                    self._torch.zeros(self._hidden_size, dtype=self._torch.float32, device=self._device),
                ).detach().clone().to(self._device, dtype=self._torch.float32)
            )
            for layer_index in self._layer_indices
        }
        optimizer = self._torch.optim.Adam(tuple(parameters.values()), lr=0.03)
        final_loss = 0.0
        for _ in range(max(4, min(12, len(paired) * 2))):
            optimizer.zero_grad()
            total_loss = self._torch.tensor(0.0, dtype=self._torch.float32, device=self._device)
            for trace, batch in paired:
                base_means = self._capture_hidden_state_means(source_text=trace.source_text)
                available_layers = tuple(layer for layer in self._layer_indices if layer in base_means)
                if not available_layers:
                    continue
                predicted_layers = self._torch.stack(
                    tuple(base_means[layer] + parameters[layer] for layer in available_layers),
                    dim=0,
                )
                composite = predicted_layers.mean(dim=0)
                projected = self._semantic_basis.to(dtype=self._torch.float32) @ composite.to(dtype=self._torch.float32)
                predicted_profile = projected / projected.norm().clamp_min(1e-6)
                target_profile = self._target_semantic_profile_tensor(
                    substrates=batch,
                    source_text=trace.source_text,
                )
                predicted_residual = self._torch.tanh(predicted_layers.abs().mean())
                target_residual = self._target_residual_tensor(substrates=batch)
                total_loss = total_loss + self._torch.mean((predicted_profile - target_profile) ** 2)
                total_loss = total_loss + (predicted_residual - target_residual) ** 2 * 0.15
            total_loss = total_loss / max(len(paired), 1)
            total_loss = total_loss + sum(parameter.pow(2).mean() for parameter in parameters.values()) * 0.002
            total_loss.backward()
            optimizer.step()
            final_loss = float(total_loss.detach().item())
        adapter_layers = tuple(
            SubstrateDeltaAdapterLayer(
                layer_index=layer_index,
                delta_vector=_clamp_delta_vector(parameters[layer_index].detach().cpu().tolist(), limit=0.18),
                mean_abs_delta=_mean_abs_delta(parameters[layer_index].detach().cpu().tolist()),
                description=(
                    f"Transformers trained adapter delta for layer {layer_index} "
                    f"hidden={self._hidden_size}."
                ),
            )
            for layer_index in self._layer_indices
        )
        return (adapter_layers, final_loss)

    def _tokenize(self, *, source_text: str) -> dict[str, object]:
        encoded = self._tokenizer(
            source_text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
        )
        model_inputs: dict[str, object] = {}
        for key, value in encoded.items():
            if isinstance(value, self._torch.Tensor):
                model_inputs[key] = value.to(self._device)
            else:
                model_inputs[key] = value
        input_ids = model_inputs.get("input_ids")
        if not isinstance(input_ids, self._torch.Tensor):
            raise ValueError(f"Transformers runtime '{self.model_id}' tokenizer did not return tensor input_ids.")
        return model_inputs

    def _build_control_delta(
        self,
        *,
        applied_control: tuple[float, ...],
        track_scale: tuple[float, ...],
        layer_index: int | None = None,
    ):
        if not applied_control:
            raise ValueError("applied_control must be non-empty")
        if not track_scale:
            raise ValueError("track_scale must be non-empty")
        basis_rank = int(self._control_basis.shape[0])
        coeffs = []
        for index in range(basis_rank):
            coeffs.append(
                (
                    float(applied_control[index])
                    if index < len(applied_control)
                    else 0.0
                )
                * float(track_scale[min(index, len(track_scale) - 1)])
            )
        control_vector = self._torch.tensor(coeffs, dtype=self._torch.float32, device=self._device)
        delta = control_vector @ self._control_basis
        layer_gain = (
            self._control_layer_gains.get(layer_index, 0.0)
            if layer_index is not None
            else 1.0
        )
        return delta * self._rare_heavy_control_scale * layer_gain

    def _build_personal_conditioning_delta(
        self,
        *,
        conditioning: PersonalConditioningSnapshot | None,
    ):
        return build_personal_conditioning_delta(
            torch_module=self._torch,
            conditioning=conditioning,
            basis=self._personal_conditioning_basis,
            scale=self._personal_conditioning_scale,
            device=self._device,
        )

    def _build_personal_conditioning_prefix(
        self,
        *,
        conditioning: PersonalConditioningSnapshot | None,
    ) -> list[tuple[Any, Any]] | None:
        """Generate this turn's key/value prefix, or ``None`` for no injection.

        Gated identically to the residual carrier (absent / cold-start /
        zero-confidence snapshots inject nothing) so the two arms differ only
        in how the same admitted state reaches the substrate.
        """

        if (
            conditioning is None
            or conditioning.is_cold_start
            or conditioning.confidence <= 0.0
        ):
            return None
        assert self._prefix_generator is not None  # guarded in generate()
        return self._prefix_generator.build(conditioning.state_vector)

    def _build_relationship_conditioning_prefix(
        self,
        *,
        carrier: ConditioningBankLatentCarrier,
    ) -> list[tuple[Any, Any]]:
        """Build one admitted Relationship prefix with strict provenance."""

        artifact = self._relationship_conditioning_prefix
        generator = self._relationship_prefix_generator
        if artifact is None or generator is None:
            raise ValueError(
                "Relationship carrier='prefix_kv' requires a Relationship "
                "Prefix-KV artifact; construct the runtime with "
                "relationship_conditioning_prefix=..."
            )
        if carrier.bank.bank_type is not ConditioningBankType.RELATIONSHIP:
            raise ValueError(
                "Relationship Prefix-KV accepts only the RELATIONSHIP bank, "
                f"got {carrier.bank.bank_type.value!r}."
            )
        if carrier.projector_version != artifact.carrier_version:
            raise ValueError(
                "Relationship Prefix-KV carrier version does not match the "
                f"loaded artifact: {carrier.projector_version!r} != "
                f"{artifact.carrier_version!r}."
            )
        if carrier.bank.readout_labels != artifact.readout_labels:
            raise ValueError(
                "Relationship Prefix-KV readout_labels do not match the "
                "loaded artifact."
            )
        if not math.isclose(
            carrier.scale,
            artifact.prefix_artifact.norm_cap,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Relationship Prefix-KV carrier scale must equal the loaded "
                f"artifact norm_cap {artifact.prefix_artifact.norm_cap}."
            )
        strength = float(carrier.bank.confidence) * float(
            carrier.bank.freshness
        )
        state = tuple(
            0.5 + (float(value) - 0.5) * strength
            for value in carrier.bank.readout
        )
        return generator.build(state)

    @contextlib.contextmanager
    def _temporary_torch_seed(self, seed: int | None):
        if seed is None:
            yield
            return
        torch = self._torch
        cpu_state = torch.random.get_rng_state()
        cuda_states = None
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and cuda.is_available():
            cuda_states = cuda.get_rng_state_all()
        mps_state = None
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "get_rng_state"):
            try:
                mps_state = mps.get_rng_state()
            except RuntimeError as exc:
                _LOG.debug("MPS RNG state is not available: %r", exc)
        torch.manual_seed(int(seed))
        try:
            yield
        finally:
            torch.random.set_rng_state(cpu_state)
            if cuda is not None and cuda_states is not None:
                cuda.set_rng_state_all(cuda_states)
            if mps is not None and mps_state is not None and hasattr(
                mps, "set_rng_state"
            ):
                try:
                    mps.set_rng_state(mps_state)
                except RuntimeError as exc:
                    _LOG.debug("MPS RNG state could not be restored: %r", exc)

    def _generate_with_prefix(
        self,
        *,
        model_inputs: dict[str, Any],
        prefix_pairs: list[tuple[Any, Any]] | None,
        max_new_tokens: int,
        repetition_penalty: float,
        temperature: float,
        top_p: float,
        sampling_seed: int | None,
        require_sampling_seed: bool = True,
    ):
        """Decode over a state-derived key/value prefix.

        ``model.generate`` cannot be used here. It derives ``cache_position``
        from ``past_key_values.get_seq_length()``, so a pre-filled cache makes
        it treat the first ``num_slots`` prompt tokens as already processed and
        silently truncate the prompt; it also derives ``position_ids`` from the
        widened attention mask, which shifts every real token by ``num_slots``.
        That shift alone changes the output, which would make even a
        zero-content prefix look like a working carrier.

        This loop instead pins real tokens to positions ``0..n-1`` and feeds
        the widened mask explicitly, so the *only* difference from the
        no-prefix path is the prefix content itself. It reproduces
        ``model.generate``'s greedy output byte-for-byte when no prefix is
        supplied, which is what makes this arm comparable to the others.
        """

        torch = self._torch
        if temperature > 0:
            # Evidence runs must close C5 by aligning the RNG source across
            # the two users for the same arm/probe. Letting prefix-KV sample
            # without an explicit per-turn seed would let RNG masquerade as a
            # state carrier.
            if require_sampling_seed and sampling_seed is None:
                raise ValueError(
                    "the prefix-KV carrier requires sampling_seed when "
                    "temperature > 0."
                )
        elif require_sampling_seed and sampling_seed is not None:
            raise ValueError(
                "sampling_seed has no effect when temperature=0 on the "
                "prefix-KV carrier; omit it or open stochastic rollout."
            )
        input_ids = model_inputs["input_ids"]
        if int(input_ids.shape[0]) != 1:
            raise ValueError(
                "the prefix-KV carrier decodes one sequence at a time; got "
                f"batch {int(input_ids.shape[0])}."
            )
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        prompt_length = int(input_ids.shape[-1])
        slots = 0
        cache = self._transformers.DynamicCache()
        if prefix_pairs:
            slots = int(prefix_pairs[0][0].shape[-2])
            cache = self._transformers.DynamicCache(ddp_cache_data=prefix_pairs)
        mask = attention_mask
        if slots:
            mask = torch.cat(
                [
                    torch.ones(
                        (1, slots),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                    attention_mask,
                ],
                dim=-1,
            )
        eos_token_id = self._generation_eos_token_id()
        stop_ids = set(
            eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
        )
        penalty = float(repetition_penalty)
        seen = input_ids[0].tolist()
        generated: list[int] = []
        step_input = input_ids
        positions = torch.arange(
            prompt_length, device=input_ids.device
        ).unsqueeze(0)
        with self._temporary_torch_seed(sampling_seed):
            for step in range(max_new_tokens):
                outputs = self._model(
                    input_ids=step_input,
                    attention_mask=mask,
                    position_ids=positions,
                    past_key_values=cache,
                    use_cache=True,
                )
                logits = self._extract_logits(outputs=outputs)[0, -1].to(
                    torch.float32
                )
                if penalty and penalty != 1.0:
                    # Same transform as transformers'
                    # RepetitionPenaltyLogitsProcessor, over prompt plus
                    # generated tokens.
                    index = torch.tensor(
                        sorted(set(seen)), device=logits.device, dtype=torch.long
                    )
                    scored = logits[index]
                    logits[index] = torch.where(
                        scored < 0, scored * penalty, scored / penalty
                    )
                for token_id in _banned_repeated_ngram_tokens(
                    seen, ngram_size=3
                ):
                    logits[token_id] = float("-inf")
                if temperature > 0:
                    next_id = self._sample_next_token_id(
                        logits=logits,
                        temperature=temperature,
                        top_p=top_p,
                    )
                else:
                    next_id = int(logits.argmax())
                if next_id in stop_ids:
                    break
                generated.append(next_id)
                seen.append(next_id)
                step_input = torch.tensor(
                    [[next_id]], device=input_ids.device, dtype=input_ids.dtype
                )
                positions = torch.tensor(
                    [[prompt_length + step]], device=input_ids.device
                )
                mask = torch.cat(
                    [
                        mask,
                        torch.ones((1, 1), dtype=mask.dtype, device=mask.device),
                    ],
                    dim=-1,
                )
        if not generated:
            return input_ids
        tail = torch.tensor(
            [generated], device=input_ids.device, dtype=input_ids.dtype
        )
        return torch.cat([input_ids, tail], dim=-1)

    def _sample_next_token_id(
        self,
        *,
        logits: Any,
        temperature: float,
        top_p: float,
    ) -> int:
        torch = self._torch
        scaled = logits / max(float(temperature), 1e-6)
        if top_p < 0.999:
            sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative > float(top_p)
            remove[0] = False
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            filtered = torch.full_like(scaled, float("-inf"))
            filtered.scatter_(0, sorted_indices, sorted_logits)
            scaled = filtered
        probs = torch.softmax(scaled, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return int(next_token.item())

    def _greedy_generate_with_prefix(
        self,
        *,
        model_inputs: dict[str, Any],
        prefix_pairs: list[tuple[Any, Any]] | None,
        max_new_tokens: int,
        repetition_penalty: float,
        temperature: float,
    ):
        return self._generate_with_prefix(
            model_inputs=model_inputs,
            prefix_pairs=prefix_pairs,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=1.0,
            sampling_seed=None,
        )

    def export_rare_heavy_state(self, *, checkpoint_id: str | None = None) -> SubstrateRareHeavyCheckpoint:
        adapter_layers = self._export_adapter_layers()
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
                f"Transformers rare-heavy checkpoint for {self.model_id} "
                f"updates={self._rare_heavy_update_count}."
            ),
            checkpoint_version=2 if adapter_layers else 1,
            training_mode=training_mode,
            compatibility_fingerprint=_build_compatibility_fingerprint(
                model_id=self.model_id,
                runtime_origin=self.runtime_origin,
                hidden_size=self._hidden_size,
                layer_indices=self._layer_indices,
                training_mode=training_mode,
            ),
            adapter_scale=self._rare_heavy_adapter_scale,
            adapter_parameter_count=_adapter_parameter_count(adapter_layers),
            adapter_training_loss=0.0,
            adapter_layers=adapter_layers,
        )

    def _install_common_adapter_bundle(
        self,
        bundle: CommonAdapterBundle,
    ) -> None:
        """Install a gate-admitted process-wide bundle during startup."""

        bundle.require_active()
        if bundle.base_model_id != self.model_id:
            raise ValueError(
                "common adapter base_model_id does not match runtime: "
                f"bundle={bundle.base_model_id!r}, runtime={self.model_id!r}."
            )
        checkpoint = bundle.rare_heavy_checkpoint
        expected_checkpoint_fingerprint = _build_compatibility_fingerprint(
            model_id=self.model_id,
            runtime_origin=self.runtime_origin,
            hidden_size=self._hidden_size,
            layer_indices=self._layer_indices,
            training_mode=checkpoint.training_mode,
        )
        if (
            checkpoint.compatibility_fingerprint
            != expected_checkpoint_fingerprint
        ):
            raise ValueError(
                "common adapter rare-heavy compatibility fingerprint does "
                "not match this runtime."
            )
        unavailable_layers = sorted(
            set(layer.layer_index for layer in checkpoint.adapter_layers)
            - set(self._layer_indices)
        )
        if unavailable_layers:
            raise ValueError(
                "common adapter rare-heavy checkpoint targets unavailable "
                f"runtime hook layers: {unavailable_layers}."
            )
        for layer in checkpoint.adapter_layers:
            if len(layer.delta_vector) != self._hidden_size:
                raise ValueError(
                    "common adapter rare-heavy delta width does not match "
                    f"runtime hidden size on layer {layer.layer_index}."
                )
        self._rare_heavy_control_scale = checkpoint.control_scale
        self._rare_heavy_semantic_text_weight = checkpoint.semantic_text_weight
        self._rare_heavy_semantic_residual_weight = (
            checkpoint.semantic_residual_weight
        )
        self._rare_heavy_anchor_bias = checkpoint.semantic_anchor_bias
        self._rare_heavy_update_count = checkpoint.update_count
        self._rare_heavy_adapter_scale = checkpoint.adapter_scale
        self._rare_heavy_adapter_deltas = {
            layer.layer_index: self._torch.tensor(
                _clamp_delta_vector(layer.delta_vector),
                dtype=self._torch.float32,
                device=self._device,
            )
            for layer in checkpoint.adapter_layers
        }
        control = bundle.control_basis_artifact
        if control.hidden_size != self._hidden_size:
            raise ValueError(
                "common adapter control basis hidden_size does not match runtime."
            )
        self.install_control_basis(
            basis=control.basis,
            provenance=control.artifact_id,
            layer_indices=control.layer_indices,
            layer_gains=control.layer_gains,
        )

    def import_rare_heavy_state(self, checkpoint: SubstrateRareHeavyCheckpoint) -> tuple[str, ...]:
        self.require_substrate_artifact_import(operation="import_rare_heavy_state()")
        if checkpoint.model_id != self.model_id:
            raise ValueError(
                f"Transformers runtime {self.model_id!r} cannot import checkpoint for {checkpoint.model_id!r}."
            )
        if checkpoint.compatibility_fingerprint and checkpoint.training_mode != "bounded-state-v1":
            expected = _build_compatibility_fingerprint(
                model_id=self.model_id,
                runtime_origin=self.runtime_origin,
                hidden_size=self._hidden_size,
                layer_indices=self._layer_indices,
                training_mode=checkpoint.training_mode,
            )
            if checkpoint.compatibility_fingerprint != expected:
                raise ValueError(
                    f"Checkpoint fingerprint {checkpoint.compatibility_fingerprint!r} does not match runtime {expected!r}."
                )
        text_weight, residual_weight = _normalize_semantic_weights(
            text_weight=checkpoint.semantic_text_weight,
            residual_weight=checkpoint.semantic_residual_weight,
        )
        self._rare_heavy_control_scale = max(0.04, min(0.30, checkpoint.control_scale))
        self._rare_heavy_semantic_text_weight = text_weight
        self._rare_heavy_semantic_residual_weight = residual_weight
        anchor_bias = tuple(
            max(-0.2, min(0.2, value))
            for value in checkpoint.semantic_anchor_bias[: len(RARE_HEAVY_ANCHOR_ORDER)]
        )
        if len(anchor_bias) < len(RARE_HEAVY_ANCHOR_ORDER):
            anchor_bias = anchor_bias + tuple(
                0.0 for _ in range(len(RARE_HEAVY_ANCHOR_ORDER) - len(anchor_bias))
            )
        self._rare_heavy_anchor_bias = anchor_bias
        self._rare_heavy_update_count = max(0, checkpoint.update_count)
        self._rare_heavy_adapter_scale = max(0.0, checkpoint.adapter_scale)
        self._rare_heavy_adapter_deltas = {
            layer.layer_index: self._torch.tensor(
                _clamp_delta_vector(layer.delta_vector),
                dtype=self._torch.float32,
                device=self._device,
            )
            for layer in checkpoint.adapter_layers
        }
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
        base_text_weight, base_residual_weight = self._base_semantic_weights()
        checkpoint = _derive_rare_heavy_checkpoint(
            checkpoint_id=checkpoint_id or f"{self.model_id}:rare-heavy-trained",
            model_id=self.model_id,
            runtime_origin=self.runtime_origin,
            current_control_scale=self._rare_heavy_control_scale,
            default_text_weight=base_text_weight,
            default_residual_weight=base_residual_weight,
            previous_update_count=self._rare_heavy_update_count,
            substrate_steps_per_trace=substrate_steps_per_trace,
        )
        backend = self._rare_heavy_training_backend
        if backend is not None:
            result = backend.train(
                RareHeavyTrainingRequest(
                    model_id=self.model_id,
                    hidden_size=self._hidden_size,
                    layer_indices=self._layer_indices,
                    device=str(self._device),
                    traces=traces,
                )
            )
            return _checkpoint_with_adapter_payload(
                checkpoint,
                training_mode=result.training_mode,
                compatibility_fingerprint=_build_compatibility_fingerprint(
                    model_id=self.model_id,
                    runtime_origin=self.runtime_origin,
                    hidden_size=self._hidden_size,
                    layer_indices=self._layer_indices,
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
                hidden_size=self._hidden_size,
                layer_indices=self._layer_indices,
                training_mode="adapter-delta-v2",
            ),
            adapter_scale=1.0,
            adapter_training_loss=training_loss,
            adapter_layers=adapter_layers,
            description=(
                f"{checkpoint.description} Adapter-delta payload "
                f"layers={len(adapter_layers)} loss={training_loss:.4f}."
            ),
        )

    def clone_for_rare_heavy(self) -> "OpenWeightResidualRuntime":
        cloned = TransformersOpenWeightResidualRuntime(
            model_id=self.model_id,
            pretrained_source=self._pretrained_source,
            device=self._device,
            model=self._model,
            tokenizer=self._tokenizer,
            max_length=self._max_length,
            mps_generation_max_input_tokens=self._mps_generation_max_input_tokens,
            top_k_logits=self._top_k_logits,
            activation_width=self._activation_width,
            layer_indices=self._layer_indices,
            control_scale=self._control_scale,
            relationship_conditioning_projector=(
                self._relationship_conditioning_projector
            ),
            relationship_conditioning_prefix=(
                self._relationship_conditioning_prefix
            ),
            character_prefix_package=self._character_prefix_package,
            character_prefix_registry=self._character_prefix_registry,
            common_adapter_bundle=self._common_adapter_bundle,
            character_residual_package=self._character_residual_package,
            runtime_origin=self._runtime_origin,
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
                f"Transformers online-fast checkpoint for {self.model_id} "
                f"updates={self._online_fast_update_count}."
            ),
            compatibility_fingerprint=_build_compatibility_fingerprint(
                model_id=self.model_id,
                runtime_origin=self.runtime_origin,
                hidden_size=self._hidden_size,
                layer_indices=self._layer_indices,
                training_mode="online-fast-delta-v1",
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
                f"Transformers runtime {self.model_id!r} cannot import online-fast checkpoint for "
                f"{checkpoint.model_id!r}."
            )
        if checkpoint.compatibility_fingerprint:
            expected = _build_compatibility_fingerprint(
                model_id=self.model_id,
                runtime_origin=self.runtime_origin,
                hidden_size=self._hidden_size,
                layer_indices=self._layer_indices,
                training_mode=checkpoint.training_mode,
            )
            if checkpoint.compatibility_fingerprint != expected:
                raise ValueError(
                    f"Online-fast checkpoint fingerprint {checkpoint.compatibility_fingerprint!r} "
                    f"does not match runtime {expected!r}."
                )
        self._online_fast_delta_scale = max(0.0, min(0.18, checkpoint.delta_scale))
        self._online_fast_update_count = max(0, checkpoint.update_count)
        self._online_fast_optimizer_state_norm = _clamp_unit(checkpoint.optimizer_state_norm)
        self._online_fast_parameter_change_rate = _clamp_unit(checkpoint.parameter_change_rate)
        self._online_fast_state_hash = checkpoint.fast_state_hash
        self._online_fast_source_state_hash = checkpoint.source_fast_state_hash
        self._online_fast_signal = checkpoint.fast_memory_signal
        self._online_fast_optimizer_state_description = checkpoint.optimizer_state_description
        self._online_fast_adapter_deltas = {
            layer.layer_index: self._torch.tensor(
                _clamp_delta_vector(
                    (
                        layer.delta_vector[: self._hidden_size]
                        + tuple(0.0 for _ in range(max(self._hidden_size - len(layer.delta_vector), 0)))
                    ),
                    limit=0.12,
                ),
                dtype=self._torch.float32,
                device=self._device,
            )
            for layer in checkpoint.adapter_layers
        }
        return ("online-fast:substrate-import",)

    def restore_online_fast_state(self, checkpoint: SubstrateOnlineFastCheckpoint) -> tuple[str, ...]:
        self.apply_online_fast_state(checkpoint)
        return ("online-fast:substrate-rollback",)

    def _adapter_delta_for_layer(self, *, layer_index: int):
        rare_heavy_delta = self._rare_heavy_adapter_deltas.get(layer_index)
        online_fast_delta = self._online_fast_adapter_deltas.get(layer_index)
        if rare_heavy_delta is None and online_fast_delta is None:
            return None
        combined = None
        if rare_heavy_delta is not None and self._rare_heavy_adapter_scale > 0.0:
            combined = rare_heavy_delta * self._rare_heavy_adapter_scale
        if online_fast_delta is not None and self._online_fast_delta_scale > 0.0:
            scaled_online_fast = online_fast_delta * self._online_fast_delta_scale
            combined = scaled_online_fast if combined is None else combined + scaled_online_fast
        return combined

    def _export_adapter_layers(self) -> tuple[SubstrateDeltaAdapterLayer, ...]:
        return tuple(
            SubstrateDeltaAdapterLayer(
                layer_index=layer_index,
                delta_vector=_clamp_delta_vector(delta.detach().cpu().tolist(), limit=0.18),
                mean_abs_delta=_mean_abs_delta(delta.detach().cpu().tolist()),
                description=(
                    f"Transformers adapter delta for layer {layer_index} "
                    f"hidden={self._hidden_size}."
                ),
            )
            for layer_index, delta in sorted(self._rare_heavy_adapter_deltas.items())
        )

    def _export_online_fast_layers(self) -> tuple[SubstrateDeltaAdapterLayer, ...]:
        return tuple(
            SubstrateDeltaAdapterLayer(
                layer_index=layer_index,
                delta_vector=_clamp_delta_vector(delta.detach().cpu().tolist(), limit=0.12),
                mean_abs_delta=_mean_abs_delta(delta.detach().cpu().tolist()),
                description=(
                    f"Transformers online-fast delta for layer {layer_index} "
                    f"hidden={self._hidden_size}."
                ),
            )
            for layer_index, delta in sorted(self._online_fast_adapter_deltas.items())
        )

    def _make_capture_hook(
        self,
        *,
        layer_index: int,
        captured_layers: dict[int, object],
        control_delta,
        capture_residuals: bool = True,
        personal_delta=None,
        conditioning_bank_delta=None,
        character_residual_delta=None,
    ):
        def hook(module, args, output):
            del module
            del args
            hidden = self._extract_hidden_tensor(output=output)
            adapter_delta = self._adapter_delta_for_layer(layer_index=layer_index)
            personal_gain = self._personal_conditioning_layer_gains.get(
                layer_index,
                0.0,
            )
            applies_personal_delta = (
                personal_delta is not None and personal_gain > 0.0
            )
            if (
                adapter_delta is None
                and control_delta is None
                and not applies_personal_delta
                and conditioning_bank_delta is None
                and character_residual_delta is None
            ):
                if capture_residuals:
                    captured_layers[layer_index] = hidden.detach().cpu()
                return None
            adjusted = hidden
            if adapter_delta is not None:
                adjusted = adjusted + adapter_delta.view(1, 1, -1).to(dtype=hidden.dtype)
            if character_residual_delta is not None:
                adjusted = adjusted + character_residual_delta.view(
                    1, 1, -1
                ).to(dtype=hidden.dtype)
            if control_delta is not None:
                adjusted = adjusted + control_delta.view(1, 1, -1).to(dtype=hidden.dtype)
            if applies_personal_delta:
                adjusted = adjusted + (
                    personal_delta * personal_gain
                ).view(1, 1, -1).to(dtype=hidden.dtype)
            if conditioning_bank_delta is not None:
                adjusted = adjusted + conditioning_bank_delta.view(
                    1, 1, -1
                ).to(dtype=hidden.dtype)
            if capture_residuals:
                captured_layers[layer_index] = adjusted.detach().cpu()
            if isinstance(output, tuple):
                return (adjusted, *output[1:])
            return adjusted

        return hook

    def _extract_hidden_tensor(self, *, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        if not isinstance(hidden, self._torch.Tensor):
            raise TypeError(f"Transformers runtime '{self.model_id}' hook output was not tensor-shaped.")
        return hidden

    def _extract_logits(self, *, outputs):
        try:
            logits = outputs.logits
        except AttributeError:
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                raise TypeError(f"Transformers runtime '{self.model_id}' outputs did not expose logits.")
        if not isinstance(logits, self._torch.Tensor):
            raise TypeError(f"Transformers runtime '{self.model_id}' logits were not tensor-shaped.")
        return logits.detach().cpu()

    def _decode_tokens(self, *, input_ids) -> tuple[str, ...]:
        token_ids = tuple(int(token_id) for token_id in input_ids[0].tolist())
        try:
            raw_tokens = tuple(self._tokenizer.convert_ids_to_tokens(token_ids))
        except AttributeError:
            raw_tokens = tuple(str(token_id) for token_id in token_ids)
        normalized = []
        for index, token in enumerate(raw_tokens):
            cleaned = token.replace("Ġ", "").replace("▁", "").strip()
            normalized.append(cleaned or f"<tok:{token_ids[index]}>")
        return tuple(normalized)

    def _decode_generated_text(self, *, token_ids) -> str:
        try:
            return str(self._tokenizer.decode(token_ids, skip_special_tokens=True)).strip()
        except AttributeError:
            pass
        flattened_ids = tuple(int(token_id) for token_id in token_ids.tolist())
        try:
            tokens = tuple(self._tokenizer.convert_ids_to_tokens(flattened_ids))
        except AttributeError:
            tokens = tuple(str(token_id) for token_id in flattened_ids)
        cleaned_tokens = [
            token.replace("Ġ", " ").replace("▁", " ").strip()
            for token in tokens
            if token.strip()
        ]
        return " ".join(cleaned_tokens).strip()

    def _build_runtime_capture(
        self,
        *,
        source_text: str,
        input_ids,
        logits,
        captured_layers: dict[int, object],
        control_applied: bool,
        personal_conditioning_applied: bool = False,
    ) -> OpenWeightRuntimeCapture:
        if not captured_layers:
            raise RuntimeError(f"Transformers runtime '{self.model_id}' did not record any hooked activations.")
        tokens = self._decode_tokens(input_ids=input_ids)
        step_count = len(tokens)
        last_logits = logits[0, -1]
        probabilities = self._torch.softmax(last_logits, dim=-1)
        top_k = min(self._top_k_logits, int(probabilities.shape[-1]))
        top_values, _ = self._torch.topk(probabilities, k=top_k)
        token_logits = tuple(float(value) for value in top_values.tolist())
        top_entropy = _normalized_entropy(token_logits)
        top_margin = _clamp_unit(token_logits[0] - token_logits[1]) if len(token_logits) > 1 else _clamp_unit(
            token_logits[0] if token_logits else 0.0
        )
        captured_layer_indices = tuple(
            layer_index for layer_index in self._layer_indices if layer_index in captured_layers
        )
        if not captured_layer_indices:
            raise RuntimeError(f"Transformers runtime '{self.model_id}' did not record any requested hooked activations.")
        planned_layer_fraction = _clamp_unit(len(self._layer_indices) / max(len(self._block_modules), 1))
        hook_fire_rate = _clamp_unit(len(captured_layer_indices) / max(len(self._layer_indices), 1))
        available_step_count = min(
            step_count,
            *(
                int(captured_layers[layer_index].shape[1])
                for layer_index in captured_layer_indices
            ),
        )
        if available_step_count <= 0:
            raise RuntimeError(f"Transformers runtime '{self.model_id}' captured no token-level residual activations.")
        token_step_coverage = _clamp_unit(available_step_count / max(step_count, 1))
        residual_sequence: list[ResidualSequenceStep] = []
        for step_index, token in enumerate(tokens[:available_step_count]):
            step_residuals = tuple(
                ResidualActivation(
                    layer_index=layer_index,
                    activation=self._tensor_to_activation_tuple(captured_layers[layer_index][0, step_index, :]),
                    step=step_index,
                )
                for layer_index in captured_layer_indices
            )
            step_summary = _summarize_real_activations(step_residuals)
            step_features = (
                FeatureSignal(
                    name="residual_mean_abs",
                    values=(step_summary[0],),
                    source="transformers-open-weight",
                    layer_hint=self._layer_indices[0],
                ),
                FeatureSignal(
                    name="residual_peak_abs",
                    values=(step_summary[1],),
                    source="transformers-open-weight",
                    layer_hint=self._layer_indices[-1],
                ),
                FeatureSignal(
                    name="sequence_progress",
                    values=((step_index + 1) / max(step_count, 1),),
                    source="transformers-open-weight",
                ),
                FeatureSignal(
                    name="hook_layer_coverage",
                    values=(hook_fire_rate,),
                    source="transformers-open-weight",
                ),
                FeatureSignal(
                    name="planned_layer_fraction",
                    values=(planned_layer_fraction,),
                    source="transformers-open-weight",
                ),
                FeatureSignal(
                    name="hook_fire_rate",
                    values=(hook_fire_rate,),
                    source="transformers-open-weight",
                ),
                FeatureSignal(
                    name="captured_hook_layer_count",
                    values=(float(len(captured_layer_indices)),),
                    source="transformers-open-weight",
                ),
                FeatureSignal(
                    name="requested_hook_layer_count",
                    values=(float(len(self._layer_indices)),),
                    source="transformers-open-weight",
                ),
                FeatureSignal(
                    name="total_hook_block_count",
                    values=(float(len(self._block_modules)),),
                    source="transformers-open-weight",
                ),
                FeatureSignal(
                    name="token_step_coverage",
                    values=(token_step_coverage,),
                    source="transformers-open-weight",
                ),
                FeatureSignal(
                    name="residual_sequence_present",
                    values=(1.0,),
                    source="transformers-open-weight",
                ),
            )
            residual_sequence.append(
                ResidualSequenceStep(
                    step=step_index,
                    token=token,
                    feature_surface=step_features,
                    residual_activations=step_residuals,
                    description=(
                        f"Transformers hook capture for token '{token}' on layers {captured_layer_indices}"
                        f"{' with control' if control_applied else ''}."
                    ),
                )
            )
        latest_activations = residual_sequence[-1].residual_activations
        latest_summary = _summarize_real_activations(latest_activations)
        feature_surface = (
            FeatureSignal(
                name="residual_mean_abs",
                values=(latest_summary[0],),
                source="transformers-open-weight",
                layer_hint=self._layer_indices[0],
            ),
            FeatureSignal(
                name="residual_peak_abs",
                values=(latest_summary[1],),
                source="transformers-open-weight",
                layer_hint=self._layer_indices[-1],
            ),
            FeatureSignal(
                name="top_logit_confidence",
                values=(max(token_logits, default=0.0),),
                source="transformers-open-weight",
            ),
            FeatureSignal(
                name="top_logit_entropy",
                values=(top_entropy,),
                source="transformers-open-weight",
            ),
            FeatureSignal(
                name="top_logit_margin",
                values=(top_margin,),
                source="transformers-open-weight",
            ),
            FeatureSignal(
                name="residual_signed_mean",
                values=(latest_summary[2],),
                source="transformers-open-weight",
                layer_hint=self._layer_indices[-1],
            ),
            FeatureSignal(
                name="hook_layer_coverage",
                values=(hook_fire_rate,),
                source="transformers-open-weight",
            ),
            FeatureSignal(
                name="planned_layer_fraction",
                values=(planned_layer_fraction,),
                source="transformers-open-weight",
            ),
            FeatureSignal(
                name="hook_fire_rate",
                values=(hook_fire_rate,),
                source="transformers-open-weight",
            ),
            FeatureSignal(
                name="captured_hook_layer_count",
                values=(float(len(captured_layer_indices)),),
                source="transformers-open-weight",
            ),
            FeatureSignal(
                name="requested_hook_layer_count",
                values=(float(len(self._layer_indices)),),
                source="transformers-open-weight",
            ),
            FeatureSignal(
                name="total_hook_block_count",
                values=(float(len(self._block_modules)),),
                source="transformers-open-weight",
            ),
            FeatureSignal(
                name="token_step_coverage",
                values=(token_step_coverage,),
                source="transformers-open-weight",
            ),
            FeatureSignal(
                name="residual_sequence_present",
                values=(1.0,),
                source="transformers-open-weight",
            ),
            FeatureSignal(
                name="hidden_size_scale",
                values=(_clamp_unit(self._hidden_size / 4096.0),),
                source="transformers-open-weight",
            ),
            FeatureSignal(
                name="fallback_active",
                values=(1.0 if self._runtime_origin == "builtin-fallback" else 0.0,),
                source="transformers-open-weight",
            ),
        ) + self._semantic_feature_surface(
            source_text=source_text,
            captured_layers=captured_layers,
        )
        return OpenWeightRuntimeCapture(
            token_logits=token_logits,
            feature_surface=feature_surface,
            residual_activations=latest_activations,
            residual_sequence=tuple(residual_sequence),
            description=(
                f"Transformers open-weight capture model={self.model_id} device={self._device} "
                f"family={self._model_family} origin={self._runtime_origin} "
                f"tokens={len(tokens)} captured_tokens={available_step_count} layers={captured_layer_indices} "
                f"planned_layers={self._layer_indices} source_len={len(source_text)} "
                f"hook_fire_rate={hook_fire_rate:.3f} planned_layer_fraction={planned_layer_fraction:.3f} "
                f"live_mode={self.live_mutation_mode}."
            ),
            personal_conditioning_applied=personal_conditioning_applied,
        )

    def _tensor_to_activation_tuple(self, tensor) -> tuple[float, ...]:
        values = tensor.detach().cpu().tolist()
        if len(values) <= self._activation_width:
            return tuple(float(value) for value in values)
        chunk_size = max(len(values) // self._activation_width, 1)
        projected: list[float] = []
        for index in range(self._activation_width):
            start = index * chunk_size
            end = len(values) if index == self._activation_width - 1 else min((index + 1) * chunk_size, len(values))
            window = values[start:end]
            projected.append(sum(float(value) for value in window) / max(len(window), 1))
        return tuple(projected)


_BUILTIN_TRANSFORMERS_RUNTIME_LAYER_COUNT: int = 4


def _builtin_safe_layer_indices(
    layer_indices: tuple[int, ...] | None,
    *,
    builtin_layer_count: int = _BUILTIN_TRANSFORMERS_RUNTIME_LAYER_COUNT,
) -> tuple[int, ...] | None:
    if layer_indices is None:
        return None
    clipped = tuple(index for index in layer_indices if 0 <= index < builtin_layer_count)
    return clipped or None


def build_builtin_transformers_runtime(
    *,
    model_id: str = "builtin-transformers-runtime",
    device: str = "cpu",
    tokenizer: object | None = None,
    layer_indices: tuple[int, ...] | None = None,
    hook_layer_selection: str = "middle",
    activation_width: int = 8,
    allow_live_substrate_mutation: bool = False,
    ) -> TransformersOpenWeightResidualRuntime:
    transformers = importlib.import_module("transformers")
    torch = importlib.import_module("torch")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = transformers.GPT2LMHeadModel(
            transformers.GPT2Config(
                vocab_size=256,
                n_positions=64,
                n_ctx=64,
                n_embd=48,
                n_layer=4,
                n_head=4,
            )
        )
    
    return TransformersOpenWeightResidualRuntime(
        model_id=model_id,
        model=model,
        tokenizer=tokenizer or HashingWhitespaceTokenizer(vocab_size=256),
        device=device,
        layer_indices=layer_indices,
        hook_layer_selection=hook_layer_selection,
        activation_width=activation_width,
        top_k_logits=8,
        runtime_origin="builtin-fallback",
        allow_live_substrate_mutation=allow_live_substrate_mutation,
    )


def build_transformers_runtime_with_fallback(
    *,
    model_id: str,
    model_source: str | None = None,
    device: str = "auto",
    layer_indices: tuple[int, ...] | None = None,
    hook_layer_selection: str = "middle",
    activation_width: int = 8,
    local_files_only: bool = False,
    fallback_to_builtin: bool | None = None,
    fallback_mode: SubstrateFallbackMode | str | None = None,
    runtime_mode: LocalSubstrateRuntimeMode | str | None = None,
    builtin_model_id: str = "builtin-transformers-runtime",
    allow_live_substrate_mutation: bool = False,
    character_prefix_package: CharacterPrefixKVPackage | None = None,
    character_prefix_registry: CharacterPrefixKVRegistry | None = None,
    common_adapter_bundle: CommonAdapterBundle | None = None,
    character_residual_package: CharacterResidualAdapterPackage | None = None,
) -> TransformersOpenWeightResidualRuntime:
    resolved_runtime_mode = resolve_local_runtime_mode(
        runtime_mode=runtime_mode,
        local_files_only=local_files_only,
        fallback_mode=fallback_mode,
        fallback_to_builtin=fallback_to_builtin,
    )
    if resolved_runtime_mode is LocalSubstrateRuntimeMode.BUILTIN_ONLY:
        if (
            character_prefix_package is not None
            or character_prefix_registry is not None
            or common_adapter_bundle is not None
            or character_residual_package is not None
        ):
            raise ValueError(
                "character model-side packages require a real HF runtime; "
                "builtin fallback cannot provide model-compatible KV geometry."
            )
        return build_builtin_transformers_runtime(
            model_id=builtin_model_id,
            device=device,
            layer_indices=_builtin_safe_layer_indices(layer_indices),
            hook_layer_selection=hook_layer_selection,
            activation_width=activation_width,
            allow_live_substrate_mutation=allow_live_substrate_mutation,
        )
    resolved_mode = resolve_substrate_fallback_mode(
        fallback_mode=fallback_mode,
        fallback_to_builtin=fallback_to_builtin,
    )
    effective_local_files_only = local_files_only
    effective_runtime_origin = "hf-local" if local_files_only else "hf-pretrained"
    if resolved_runtime_mode is LocalSubstrateRuntimeMode.STRICT_LOCAL:
        effective_local_files_only = True
        resolved_mode = SubstrateFallbackMode.DENY
        effective_runtime_origin = "hf-local"
    elif resolved_runtime_mode is LocalSubstrateRuntimeMode.PREFER_LOCAL:
        effective_local_files_only = True
        resolved_mode = SubstrateFallbackMode.ALLOW_BUILTIN
        effective_runtime_origin = "hf-local"
    effective_model_source = model_source or model_id
    if common_adapter_bundle is not None:
        candidate = Path(effective_model_source).expanduser()
        if candidate.is_dir():
            weights_root = candidate.resolve()
        else:
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise RuntimeError(
                    "loading a common adapter by model id requires "
                    "huggingface_hub so the frozen weight digest can be "
                    "verified."
                ) from exc
            weights_root = Path(
                snapshot_download(
                    repo_id=effective_model_source,
                    local_files_only=effective_local_files_only,
                )
            ).resolve()
        actual_weights_sha256 = fingerprint_model_weight_files(weights_root)
        if (
            actual_weights_sha256
            != common_adapter_bundle.base_model_weights_sha256
        ):
            raise ValueError(
                "common adapter base weight digest does not match the frozen "
                f"runtime snapshot: declared="
                f"{common_adapter_bundle.base_model_weights_sha256}, "
                f"actual={actual_weights_sha256}."
            )
        effective_model_source = str(weights_root)
        effective_local_files_only = True
    try:
        return TransformersOpenWeightResidualRuntime(
            model_id=model_id,
            pretrained_source=effective_model_source,
            device=device,
            layer_indices=layer_indices,
            hook_layer_selection=hook_layer_selection,
            activation_width=activation_width,
            local_files_only=effective_local_files_only,
            runtime_origin=effective_runtime_origin,
            allow_live_substrate_mutation=allow_live_substrate_mutation,
            character_prefix_package=character_prefix_package,
            character_prefix_registry=character_prefix_registry,
            common_adapter_bundle=common_adapter_bundle,
            character_residual_package=character_residual_package,
        )
    except Exception as exc:
        if resolved_mode is not SubstrateFallbackMode.ALLOW_BUILTIN or not _is_transformers_runtime_fallback_error(exc):
            raise
        if (
            character_prefix_package is not None
            or character_prefix_registry is not None
            or common_adapter_bundle is not None
            or character_residual_package is not None
        ):
            raise RuntimeError(
                "real HF runtime failed while a character model-side package was "
                "requested; refusing to fall back to an incompatible builtin "
                "substrate."
            ) from exc
        return build_builtin_transformers_runtime(
            model_id=builtin_model_id,
            device=device,
            layer_indices=_builtin_safe_layer_indices(layer_indices),
            hook_layer_selection=hook_layer_selection,
            activation_width=activation_width,
            allow_live_substrate_mutation=allow_live_substrate_mutation,
        )


def run_hook_layer_calibration(
    *,
    model_id: str,
    source_text: str,
    runtime_builder: Callable[[tuple[int, ...]], OpenWeightResidualRuntime],
    layer_index_sets: tuple[tuple[int, ...], ...],
) -> HookLayerCalibrationReport:
    cases: list[HookLayerCalibrationCase] = []
    for layer_indices in layer_index_sets:
        runtime = runtime_builder(layer_indices)
        capture = runtime.capture(source_text=source_text)
        feature_map = {signal.name: signal.values[0] for signal in capture.feature_surface if signal.values}
        task_pull = feature_map.get("semantic_task_pull", 0.0)
        support_pull = feature_map.get("semantic_support_pull", 0.0)
        repair_pull = feature_map.get("semantic_repair_pull", 0.0)
        directive_pull = feature_map.get("semantic_directive_pull", 0.0)
        exploration_pull = feature_map.get("semantic_exploration_pull", 0.0)
        hook_coverage = feature_map.get("hook_layer_coverage", 0.0)
        fallback_active = feature_map.get("fallback_active", 0.0)
        semantic_separation = _clamp_unit(
            max(task_pull, support_pull, repair_pull, directive_pull, exploration_pull)
            - min(task_pull, support_pull, repair_pull, directive_pull, exploration_pull)
        )
        signal_quality = _clamp_unit(
            hook_coverage * 0.35
            + (1.0 - fallback_active) * 0.25
            + feature_map.get("top_logit_margin", 0.0) * 0.15
            + (1.0 - feature_map.get("top_logit_entropy", 0.0)) * 0.10
            + semantic_separation * 0.15
        )
        cases.append(
            HookLayerCalibrationCase(
                layer_indices=layer_indices,
                hook_layer_coverage=round(hook_coverage, 4),
                residual_sequence_length=len(capture.residual_sequence),
                semantic_separation=round(semantic_separation, 4),
                signal_quality=round(signal_quality, 4),
                runtime_origin=getattr(runtime, "runtime_origin", "unknown"),
                description=capture.description,
            )
        )
    ranked = sorted(
        cases,
        key=lambda item: (
            item.signal_quality,
            item.semantic_separation,
            item.hook_layer_coverage,
            item.residual_sequence_length,
        ),
        reverse=True,
    )
    recommended_layers = ranked[0].layer_indices if ranked else ()
    return HookLayerCalibrationReport(
        model_id=model_id,
        source_text=source_text,
        cases=tuple(cases),
        recommended_layers=recommended_layers,
        description=(
            f"Hook layer calibration for {model_id} over {len(cases)} cases; "
            f"recommended_layers={recommended_layers}."
        ),
    )


def _is_transformers_runtime_fallback_error(exc: Exception) -> bool:
    if isinstance(exc, (OSError, ValueError, RuntimeError, TimeoutError)):
        return True
    module_name = type(exc).__module__
    class_name = type(exc).__name__
    if module_name.startswith("httpx") and class_name.endswith("Timeout"):
        return True
    if module_name.startswith("requests") and class_name in {"ReadTimeout", "ConnectTimeout", "Timeout"}:
        return True
    if module_name.startswith("huggingface_hub.errors"):
        return True
    return False


def probe_local_model_compatibility(
    *,
    model_id: str,
    model_source: str | None = None,
    device: str = "cpu",
) -> LocalModelCompatibilityReport:
    transformers = importlib.import_module("transformers")
    load_source = model_source or model_id
    local_tokenizer_available = False
    local_model_available = False
    strict_local_runtime_available = False
    error_type: str | None = None
    error_message = ""
    try:
        try:
            transformers.AutoTokenizer.from_pretrained(
                load_source,
                local_files_only=True,
            )
            local_tokenizer_available = True
        except Exception:
            transformers.AutoTokenizer.from_pretrained(
                load_source,
                local_files_only=True,
                use_fast=False,
            )
            local_tokenizer_available = True
        transformers.AutoModelForCausalLM.from_pretrained(
            load_source,
            local_files_only=True,
        )
        local_model_available = True
        runtime = build_transformers_runtime_with_fallback(
            model_id=model_id,
            model_source=model_source,
            device=device,
            local_files_only=True,
            runtime_mode=LocalSubstrateRuntimeMode.STRICT_LOCAL,
        )
        runtime.capture(source_text="local compatibility probe")
        strict_local_runtime_available = True
        description = (
            f"Local model compatibility OK for {model_id}: tokenizer/model/runtime all available."
        )
    except Exception as exc:
        error_type = type(exc).__name__
        error_message = str(exc)
        description = (
            f"Local model compatibility probe failed for {model_id}: "
            f"{error_type}: {error_message}"
        )
    return LocalModelCompatibilityReport(
        model_id=model_id,
        local_tokenizer_available=local_tokenizer_available,
        local_model_available=local_model_available,
        strict_local_runtime_available=strict_local_runtime_available,
        error_type=error_type,
        error_message=error_message,
        description=description,
    )
