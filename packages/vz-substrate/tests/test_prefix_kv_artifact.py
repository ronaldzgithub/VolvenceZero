"""State-KV prefix artifact contracts."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from volvence_zero.personal_conditioning_contracts import (
    PERSONAL_CONDITIONING_VECTOR_LABELS,
)
from volvence_zero.substrate.prefix_kv_artifact import (
    CHARACTER_TEACHER_FORCED_PREFIX_TRAINING_MODE,
    CharacterPrefixKVPackage,
    MAX_PREFIX_NORM_CAP,
    PrefixKVArtifact,
    STATE_STRATEGY_ROUTED_PREFIX_TRAINING_MODE,
    TEACHER_DISTILLED_PREFIX_TRAINING_MODE,
    build_teacher_distilled_prefix_artifact,
    load_prefix_generator,
)

LAYERS = 3
KV_HEADS = 2
HEAD_DIM = 4
SLOTS = 2
RANK = 3
WIDTH = SLOTS * KV_HEADS * HEAD_DIM
COORDINATES = len(PERSONAL_CONDITIONING_VECTOR_LABELS)


def _artifact(*, norm_cap: float = 0.2) -> PrefixKVArtifact:
    return build_teacher_distilled_prefix_artifact(
        model_id="Qwen/test",
        num_layers=LAYERS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        num_slots=SLOTS,
        bottleneck_rank=RANK,
        encoder_rows=[
            [0.1 * ((row + col) % 5) for col in range(COORDINATES)]
            for row in range(RANK)
        ],
        encoder_bias=[0.01 * row for row in range(RANK)],
        key_projection=[
            [[0.05 * ((layer + row + col) % 4) for col in range(RANK)]
             for row in range(WIDTH)]
            for layer in range(LAYERS)
        ],
        key_bias=[[0.02] * WIDTH for _ in range(LAYERS)],
        value_projection=[
            [[0.03 * ((layer + row + col) % 3) for col in range(RANK)]
             for row in range(WIDTH)]
            for layer in range(LAYERS)
        ],
        value_bias=[[0.01] * WIDTH for _ in range(LAYERS)],
        reference_key_norms=[12.0, 8.0, 5.0],
        reference_value_norms=[3.0, 2.5, 2.0],
        norm_cap=norm_cap,
        source_fingerprint="weights-and-material-fingerprint",
        sample_count=96,
    )


def _state() -> tuple[float, ...]:
    return tuple(
        0.5 + 0.4 * ((index % 3) - 1) for index in range(COORDINATES)
    )


def test_prefix_artifact_round_trips() -> None:
    artifact = _artifact()

    restored = PrefixKVArtifact.from_json(artifact.to_json())

    assert restored == artifact
    assert restored.artifact_id == artifact.artifact_id
    assert restored.output_width == WIDTH


def test_prefix_artifact_rejects_tampered_payload() -> None:
    raw = json.loads(_artifact().to_json())
    raw["norm_cap"] = 0.4

    with pytest.raises(ValueError, match="artifact_id"):
        PrefixKVArtifact.from_json(json.dumps(raw))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw.pop("artifact_id"), "frozen schema"),
        (lambda raw: raw.update({"unknown": True}), "frozen schema"),
        (
            lambda raw: raw.update({"training_mode": "unreviewed-mode"}),
            "training_mode",
        ),
    ],
)
def test_prefix_artifact_rejects_schema_drift(
    mutation: Callable[[dict[str, object]], object],
    message: str,
) -> None:
    raw = json.loads(_artifact().to_json())
    mutation(raw)

    with pytest.raises(ValueError, match=message):
        PrefixKVArtifact.from_json(json.dumps(raw))


def test_legacy_teacher_distilled_training_mode_remains_readable() -> None:
    artifact = build_teacher_distilled_prefix_artifact(
        model_id="Qwen/test",
        num_layers=LAYERS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        num_slots=SLOTS,
        bottleneck_rank=RANK,
        encoder_rows=[[0.0] * COORDINATES for _ in range(RANK)],
        encoder_bias=[0.0] * RANK,
        key_projection=[[[0.0] * RANK for _ in range(WIDTH)] for _ in range(LAYERS)],
        key_bias=[[0.0] * WIDTH for _ in range(LAYERS)],
        value_projection=[[[0.0] * RANK for _ in range(WIDTH)] for _ in range(LAYERS)],
        value_bias=[[0.0] * WIDTH for _ in range(LAYERS)],
        reference_key_norms=[12.0, 8.0, 5.0],
        reference_value_norms=[3.0, 2.5, 2.0],
        norm_cap=0.2,
        source_fingerprint="legacy-fingerprint",
        sample_count=96,
        training_mode=TEACHER_DISTILLED_PREFIX_TRAINING_MODE,
    )

    restored = PrefixKVArtifact.from_json(artifact.to_json())

    assert restored.training_mode == TEACHER_DISTILLED_PREFIX_TRAINING_MODE


def test_state_strategy_training_mode_is_supported() -> None:
    artifact = build_teacher_distilled_prefix_artifact(
        model_id="Qwen/test",
        num_layers=LAYERS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        num_slots=SLOTS,
        bottleneck_rank=RANK,
        encoder_rows=[[0.0] * COORDINATES for _ in range(RANK)],
        encoder_bias=[0.0] * RANK,
        key_projection=[[[0.0] * RANK for _ in range(WIDTH)] for _ in range(LAYERS)],
        key_bias=[[0.0] * WIDTH for _ in range(LAYERS)],
        value_projection=[[[0.0] * RANK for _ in range(WIDTH)] for _ in range(LAYERS)],
        value_bias=[[0.0] * WIDTH for _ in range(LAYERS)],
        reference_key_norms=[12.0, 8.0, 5.0],
        reference_value_norms=[3.0, 2.5, 2.0],
        norm_cap=0.2,
        source_fingerprint="state-strategy-fingerprint",
        sample_count=96,
        training_mode=STATE_STRATEGY_ROUTED_PREFIX_TRAINING_MODE,
    )

    restored = PrefixKVArtifact.from_json(artifact.to_json())

    assert restored.training_mode == STATE_STRATEGY_ROUTED_PREFIX_TRAINING_MODE


def test_norm_cap_is_bounded_above() -> None:
    with pytest.raises(ValueError, match="norm_cap"):
        _artifact(norm_cap=MAX_PREFIX_NORM_CAP + 0.01)


def test_reference_norms_must_be_positive() -> None:
    artifact = _artifact()

    # A zero reference norm would silently disable the cap for that layer
    # while the artifact still advertises itself as bounded.
    with pytest.raises(ValueError, match="reference_key_norms"):
        PrefixKVArtifact(
            **{**artifact.__dict__, "reference_key_norms": (12.0, 0.0, 5.0)}
        )


def test_geometry_mismatch_is_rejected_not_reshaped() -> None:
    artifact = _artifact()

    with pytest.raises(ValueError, match="key_projection block"):
        PrefixKVArtifact(
            **{
                **artifact.__dict__,
                "num_slots": SLOTS + 1,
            }
        )


def test_generator_respects_the_measured_norm_budget() -> None:
    torch = pytest.importorskip("torch")
    artifact = _artifact()

    generator = load_prefix_generator(
        torch_module=torch,
        artifact=artifact,
        expected_model_id="Qwen/test",
        expected_num_layers=LAYERS,
        expected_num_kv_heads=KV_HEADS,
        expected_head_dim=HEAD_DIM,
        device="cpu",
        dtype=torch.float32,
    )
    pairs = generator.build(_state())

    assert len(pairs) == LAYERS
    for index, (key, value) in enumerate(pairs):
        assert tuple(key.shape) == (1, KV_HEADS, SLOTS, HEAD_DIM)
        assert tuple(value.shape) == (1, KV_HEADS, SLOTS, HEAD_DIM)
        key_budget = artifact.reference_key_norms[index] * artifact.norm_cap
        value_budget = artifact.reference_value_norms[index] * artifact.norm_cap
        assert float(key.norm(dim=-1).max()) <= key_budget + 1e-5
        assert float(value.norm(dim=-1).max()) <= value_budget + 1e-5


def test_generator_is_deterministic_and_state_sensitive() -> None:
    torch = pytest.importorskip("torch")
    generator = load_prefix_generator(
        torch_module=torch,
        artifact=_artifact(),
        expected_model_id="Qwen/test",
        expected_num_layers=LAYERS,
        expected_num_kv_heads=KV_HEADS,
        expected_head_dim=HEAD_DIM,
        device="cpu",
        dtype=torch.float32,
    )

    first = generator.build(_state())
    again = generator.build(_state())
    other = generator.build(tuple(1.0 - value for value in _state()))

    for (key_a, value_a), (key_b, value_b) in zip(first, again, strict=True):
        assert torch.equal(key_a, key_b)
        assert torch.equal(value_a, value_b)
    # A generator that ignores its input would make the whole carrier claim
    # vacuous, so state sensitivity is a contract test, not a training metric.
    assert any(
        not torch.equal(key_a, key_c)
        for (key_a, _), (key_c, _) in zip(first, other, strict=True)
    )


def test_loading_rejects_foreign_attention_geometry() -> None:
    torch = pytest.importorskip("torch")
    artifact = _artifact()

    with pytest.raises(ValueError, match="model_id"):
        load_prefix_generator(
            torch_module=torch,
            artifact=artifact,
            expected_model_id="Other/model",
            expected_num_layers=LAYERS,
            expected_num_kv_heads=KV_HEADS,
            expected_head_dim=HEAD_DIM,
            device="cpu",
            dtype=torch.float32,
        )


def test_character_package_round_trips_without_personal_coordinate_namespace() -> None:
    artifact = build_teacher_distilled_prefix_artifact(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        num_layers=LAYERS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        num_slots=SLOTS,
        bottleneck_rank=1,
        encoder_rows=((0.0,),),
        encoder_bias=(0.0,),
        key_projection=tuple(
            tuple((0.0,) for _ in range(WIDTH)) for _ in range(LAYERS)
        ),
        key_bias=tuple(tuple(0.0 for _ in range(WIDTH)) for _ in range(LAYERS)),
        value_projection=tuple(
            tuple((0.0,) for _ in range(WIDTH)) for _ in range(LAYERS)
        ),
        value_bias=tuple(tuple(0.0 for _ in range(WIDTH)) for _ in range(LAYERS)),
        reference_key_norms=(12.0, 8.0, 5.0),
        reference_value_norms=(3.0, 2.5, 2.0),
        norm_cap=0.12,
        source_fingerprint="zhang-template-proof-ledger",
        sample_count=24,
        training_mode=CHARACTER_TEACHER_FORCED_PREFIX_TRAINING_MODE,
        vector_labels=("zhang_wuji_live_through_identity",),
        description="character smoke",
    )
    package = CharacterPrefixKVPackage.create(
        character_id="zhang-wuji",
        character_name="张无忌",
        model_id=artifact.model_id,
        source_live_through_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        source_template_id="zhang-wuji-live-through",
        source_template_integrity_hash="template-hash",
        source_live_through_proof="proof-path:proof-hash",
        state_vector=(0.0,),
        prefix_artifact=artifact,
        description="character package smoke",
    )

    restored = CharacterPrefixKVPackage.from_json(package.to_json())

    assert restored == package
    assert restored.prefix_artifact.vector_labels == (
        "zhang_wuji_live_through_identity",
    )
def test_generator_rejects_wrong_width_state_vector() -> None:
    torch = pytest.importorskip("torch")
    generator = load_prefix_generator(
        torch_module=torch,
        artifact=_artifact(),
        expected_model_id="Qwen/test",
        expected_num_layers=LAYERS,
        expected_num_kv_heads=KV_HEADS,
        expected_head_dim=HEAD_DIM,
        device="cpu",
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="coordinates"):
        generator.build((0.5, 0.5))
