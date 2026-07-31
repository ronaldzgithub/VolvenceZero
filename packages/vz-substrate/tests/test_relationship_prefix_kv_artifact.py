"""Dedicated Relationship Prefix-KV artifact contracts."""

from __future__ import annotations

import json

import pytest

from volvence_zero.substrate.prefix_kv_artifact import (
    build_teacher_distilled_prefix_artifact,
)
from volvence_zero.substrate.relationship_prefix_kv_artifact import (
    RelationshipPrefixKVArtifact,
    bind_relationship_prefix_artifact,
    load_relationship_prefix_generator,
)

LABELS = ("rel_trust", "rel_repair_pressure", "rel_consent_clarity")
LAYERS = 2
KV_HEADS = 2
HEAD_DIM = 4
SLOTS = 2
RANK = 2
WIDTH = KV_HEADS * HEAD_DIM * SLOTS


def _artifact(*, norm_cap: float = 0.12) -> RelationshipPrefixKVArtifact:
    nested = build_teacher_distilled_prefix_artifact(
        model_id="Qwen/test",
        num_layers=LAYERS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        num_slots=SLOTS,
        bottleneck_rank=RANK,
        encoder_rows=((0.2, -0.1, 0.3), (-0.2, 0.4, 0.1)),
        encoder_bias=(0.0, 0.0),
        key_projection=tuple(
            tuple((0.1, -0.1) for _ in range(WIDTH))
            for _ in range(LAYERS)
        ),
        key_bias=tuple(
            tuple(0.0 for _ in range(WIDTH)) for _ in range(LAYERS)
        ),
        value_projection=tuple(
            tuple((-0.05, 0.1) for _ in range(WIDTH))
            for _ in range(LAYERS)
        ),
        value_bias=tuple(
            tuple(0.0 for _ in range(WIDTH)) for _ in range(LAYERS)
        ),
        reference_key_norms=(10.0, 8.0),
        reference_value_norms=(3.0, 2.0),
        norm_cap=norm_cap,
        source_fingerprint="relationship-prefix-test-source",
        sample_count=12,
        vector_labels=LABELS,
    )
    return bind_relationship_prefix_artifact(
        prefix_artifact=nested,
        owner_schema_version="relationship-conditioning.v2",
        readout_labels=LABELS,
        description="Relationship Prefix-KV contract test artifact.",
    )


def test_relationship_prefix_artifact_round_trips() -> None:
    artifact = _artifact()

    restored = RelationshipPrefixKVArtifact.from_json(artifact.to_json())

    assert restored == artifact
    assert restored.artifact_id == artifact.artifact_id
    assert restored.carrier_version.endswith(artifact.artifact_id)
    assert restored.prefix_artifact.vector_labels == LABELS


def test_relationship_prefix_artifact_rejects_tampering() -> None:
    raw = json.loads(_artifact().to_json())
    raw["description"] = "tampered"

    with pytest.raises(ValueError, match="artifact_id"):
        RelationshipPrefixKVArtifact.from_json(json.dumps(raw))


def test_relationship_binding_rejects_label_drift_and_excess_norm() -> None:
    artifact = _artifact()
    with pytest.raises(ValueError, match="vector_labels"):
        bind_relationship_prefix_artifact(
            prefix_artifact=artifact.prefix_artifact,
            owner_schema_version="relationship-conditioning.v2",
            readout_labels=("wrong",),
            description="wrong labels",
        )
    with pytest.raises(ValueError, match="norm_cap"):
        _artifact(norm_cap=0.13)


def test_relationship_generator_checks_runtime_geometry() -> None:
    torch = pytest.importorskip("torch")
    artifact = _artifact()

    generator = load_relationship_prefix_generator(
        torch_module=torch,
        artifact=artifact,
        expected_model_id="Qwen/test",
        expected_num_layers=LAYERS,
        expected_num_kv_heads=KV_HEADS,
        expected_head_dim=HEAD_DIM,
        device="cpu",
        dtype=torch.float32,
    )

    assert len(generator.build((0.2, 0.8, 0.5))) == LAYERS
    with pytest.raises(ValueError, match="model_id"):
        load_relationship_prefix_generator(
            torch_module=torch,
            artifact=artifact,
            expected_model_id="Other/model",
            expected_num_layers=LAYERS,
            expected_num_kv_heads=KV_HEADS,
            expected_head_dim=HEAD_DIM,
            device="cpu",
            dtype=torch.float32,
        )
