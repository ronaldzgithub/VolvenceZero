from __future__ import annotations

import pytest

from volvence_zero.personal_conditioning_contracts import (
    PERSONAL_CONDITIONING_SCHEMA_VERSION,
    PERSONAL_CONDITIONING_VECTOR_LABELS,
    PersonalConditioningSnapshot,
)
from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import (
    CHARACTER_TEACHER_FORCED_PREFIX_TRAINING_MODE,
    CharacterPrefixKVPackage,
    CharacterPrefixKVRegistry,
    CharacterPrefixKVRegistryEntry,
    CommonAdapterBundle,
    CommonAdapterGateRecord,
    ControlBasisArtifact,
    HashingWhitespaceTokenizer,
    SubstrateDeltaAdapterLayer,
    SubstrateRareHeavyCheckpoint,
    TransformersOpenWeightResidualRuntime,
    build_rare_heavy_compatibility_fingerprint,
    build_sinusoid_control_basis,
    build_teacher_distilled_prefix_artifact,
)

MODEL_ID = "Qwen/tiny-character-router"
WEIGHTS_SHA256 = "a" * 64
LAYERS = 4
KV_HEADS = 4
HEAD_DIM = 12
HIDDEN_SIZE = 48
HOOK_LAYERS = (1, 2, 3)


def _prefix(*, labels: tuple[str, ...], marker: float, mode: str):
    width = KV_HEADS * HEAD_DIM
    return build_teacher_distilled_prefix_artifact(
        model_id=MODEL_ID,
        num_layers=LAYERS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        num_slots=1,
        bottleneck_rank=1,
        encoder_rows=(tuple(0.0 for _ in labels),),
        encoder_bias=(0.0,),
        key_projection=tuple(
            tuple((0.0,) for _ in range(width)) for _ in range(LAYERS)
        ),
        key_bias=tuple(
            tuple(marker for _ in range(width)) for _ in range(LAYERS)
        ),
        value_projection=tuple(
            tuple((0.0,) for _ in range(width)) for _ in range(LAYERS)
        ),
        value_bias=tuple(
            tuple(marker for _ in range(width)) for _ in range(LAYERS)
        ),
        reference_key_norms=(1.0,) * LAYERS,
        reference_value_norms=(1.0,) * LAYERS,
        norm_cap=0.1,
        source_fingerprint=f"source:{marker}",
        sample_count=1,
        training_mode=mode,
        vector_labels=labels,
    )


def _character(character_id: str, marker: float) -> CharacterPrefixKVPackage:
    artifact = _prefix(
        labels=("character_identity",),
        marker=marker,
        mode=CHARACTER_TEACHER_FORCED_PREFIX_TRAINING_MODE,
    )
    return CharacterPrefixKVPackage.create(
        character_id=character_id,
        character_name=character_id,
        model_id=MODEL_ID,
        source_live_through_model_id="Qwen/source",
        source_template_id=f"template:{character_id}",
        source_template_integrity_hash=f"integrity:{character_id}",
        source_live_through_proof=f"proof:{character_id}",
        state_vector=(0.0,),
        prefix_artifact=artifact,
        description=f"character:{character_id}",
    )


def _common_bundle() -> CommonAdapterBundle:
    training_mode = "adapter-delta-v2"
    checkpoint = SubstrateRareHeavyCheckpoint(
        checkpoint_id="common-v1",
        model_id=MODEL_ID,
        runtime_origin="hf-local",
        control_scale=0.1,
        semantic_text_weight=0.5,
        semantic_residual_weight=0.5,
        semantic_anchor_bias=(0.0,) * 5,
        update_count=1,
        source_batch_count=1,
        mean_sequence_length=1.0,
        mean_residual_magnitude=0.001,
        description="tiny common adapter",
        checkpoint_version=2,
        training_mode=training_mode,
        compatibility_fingerprint=build_rare_heavy_compatibility_fingerprint(
            model_id=MODEL_ID,
            runtime_origin="hf-local",
            hidden_size=HIDDEN_SIZE,
            layer_indices=HOOK_LAYERS,
            training_mode=training_mode,
        ),
        adapter_scale=1.0,
        adapter_parameter_count=HIDDEN_SIZE * len(HOOK_LAYERS),
        adapter_training_loss=0.1,
        adapter_layers=tuple(
            SubstrateDeltaAdapterLayer(
                layer_index=index,
                delta_vector=(0.001,) * HIDDEN_SIZE,
                mean_abs_delta=0.001,
                description=f"layer:{index}",
            )
            for index in HOOK_LAYERS
        ),
    )
    control = ControlBasisArtifact(
        model_id=MODEL_ID,
        hidden_size=HIDDEN_SIZE,
        basis=build_sinusoid_control_basis(hidden_size=HIDDEN_SIZE, rank=1),
        layer_indices=HOOK_LAYERS,
        layer_gains=(1.0,) * len(HOOK_LAYERS),
        training_mode="full-code-sinusoid-v1",
        source_fingerprint="tiny-control",
        sample_count=1,
        description="tiny control basis",
    )
    return CommonAdapterBundle.create(
        common_adapter_version="common-v1",
        base_model_id=MODEL_ID,
        base_model_weights_sha256=WEIGHTS_SHA256,
        rare_heavy_checkpoint=checkpoint,
        state_kv_artifact=_prefix(
            labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
            marker=0.01,
            mode="state-strategy-routed-prefix-v1",
        ),
        control_basis_artifact=control,
        gate_record=CommonAdapterGateRecord(
            proposal_id="proposal:common-v1",
            decision="allow",
            desired_gate="offline",
            validation_delta=0.1,
            capacity_cost=0.1,
            rollback_evidence="restore common-v0",
            is_reversible=True,
            evaluation_ref="evaluation:common-v1",
        ),
        description="tiny common bundle",
    )


def _runtime():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(7)
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=256,
            n_positions=64,
            n_ctx=64,
            n_embd=HIDDEN_SIZE,
            n_layer=LAYERS,
            n_head=KV_HEADS,
        )
    )
    common = _common_bundle()
    registry = CharacterPrefixKVRegistry(
        base_model_id=MODEL_ID,
        common_adapter_version=common.common_adapter_version,
        compatibility_fingerprint=common.compatibility_fingerprint,
        entries=(
            CharacterPrefixKVRegistryEntry(
                manifest_package_id="manifest:active",
                common_adapter_version=common.common_adapter_version,
                compatibility_fingerprint=common.compatibility_fingerprint,
                wiring_level=WiringLevel.ACTIVE,
                prefix_package=_character("active-character", 0.02),
            ),
            CharacterPrefixKVRegistryEntry(
                manifest_package_id="manifest:shadow",
                common_adapter_version=common.common_adapter_version,
                compatibility_fingerprint=common.compatibility_fingerprint,
                wiring_level=WiringLevel.SHADOW,
                prefix_package=_character("shadow-character", 0.03),
            ),
        ),
    )
    return TransformersOpenWeightResidualRuntime(
        model_id=MODEL_ID,
        model=model,
        tokenizer=HashingWhitespaceTokenizer(vocab_size=256),
        device="cpu",
        hook_layer_selection="middle",
        runtime_origin="hf-local",
        common_adapter_bundle=common,
        loaded_base_model_weights_sha256=WEIGHTS_SHA256,
        character_prefix_registry=registry,
    )


def _generate(runtime, character_id: str):
    return runtime.generate(
        prompt="hello",
        system_context="system",
        max_new_tokens=1,
        temperature=0.0,
        capture_residuals=False,
        character_id=character_id,
    )


def _conditioning(marker: float = 0.5) -> PersonalConditioningSnapshot:
    return PersonalConditioningSnapshot(
        schema_version=PERSONAL_CONDITIONING_SCHEMA_VERSION,
        state_vector=(marker,) * len(PERSONAL_CONDITIONING_VECTOR_LABELS),
        vector_labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
        source_versions=(("held-out", 1),),
        source_fingerprint=f"held-out:{marker}",
        confidence=1.0,
        is_cold_start=False,
        description="typed held-out conditioning",
    )


def test_runtime_routes_active_and_shadow_character_per_generation() -> None:
    runtime = _runtime()

    active = _generate(runtime, "active-character")
    shadow = _generate(runtime, "shadow-character")
    absent = _generate(runtime, "")

    assert active.character_id == "active-character"
    assert active.character_prefix_applied is True
    assert active.character_prefix_wiring_level == "active"
    assert shadow.character_id == "shadow-character"
    assert shadow.character_prefix_applied is False
    assert shadow.character_prefix_wiring_level == "shadow"
    assert shadow.character_prefix_shadow_id
    assert absent.character_prefix_applied is False


def test_runtime_rejects_unverified_common_adapter_base_digest() -> None:
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=256,
            n_positions=64,
            n_ctx=64,
            n_embd=HIDDEN_SIZE,
            n_layer=LAYERS,
            n_head=KV_HEADS,
        )
    )

    with pytest.raises(ValueError, match="requires the verified"):
        TransformersOpenWeightResidualRuntime(
            model_id=MODEL_ID,
            model=model,
            tokenizer=HashingWhitespaceTokenizer(vocab_size=256),
            device="cpu",
            runtime_origin="hf-local",
            common_adapter_bundle=_common_bundle(),
        )


def test_conditioned_continuation_scores_common_and_character_arms() -> None:
    runtime = _runtime()

    common_only = runtime.score_conditioned_continuation(
        source_text="hello",
        continuation_text="world",
        personal_conditioning=_conditioning(),
        applied_control=(0.25,),
    )
    character = runtime.score_conditioned_continuation(
        source_text="hello",
        continuation_text="world",
        personal_conditioning=_conditioning(),
        applied_control=(0.25,),
        character_id="active-character",
    )

    assert common_only.token_count == character.token_count == 1
    assert (
        common_only.mean_negative_log_likelihood
        != character.mean_negative_log_likelihood
    )
    assert "prefix_slots=1" in common_only.description
    assert "prefix_slots=2" in character.description


def test_conditioned_continuation_rejects_shadow_character_arm() -> None:
    runtime = _runtime()

    with pytest.raises(ValueError, match="requires an ACTIVE"):
        runtime.score_conditioned_continuation(
            source_text="hello",
            continuation_text="world",
            personal_conditioning=_conditioning(),
            character_id="shadow-character",
        )
