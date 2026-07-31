from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from volvence_zero.substrate import (
    CHARACTER_TEACHER_FORCED_PREFIX_TRAINING_MODE,
    CommonAdapterBundle,
    CommonAdapterGateRecord,
    ControlBasisArtifact,
    SubstrateDeltaAdapterLayer,
    SubstrateRareHeavyCheckpoint,
    build_teacher_distilled_prefix_artifact,
    fingerprint_model_weight_files,
    install_rare_heavy_checkpoint_hooks,
    rare_heavy_checkpoint_from_json,
    rare_heavy_checkpoint_to_json,
    remove_forward_hooks,
)


MODEL_ID = "Qwen/test"


def _prefix():
    return build_teacher_distilled_prefix_artifact(
        model_id=MODEL_ID,
        num_layers=2,
        num_kv_heads=1,
        head_dim=2,
        num_slots=1,
        bottleneck_rank=1,
        encoder_rows=((1.0,),),
        encoder_bias=(0.0,),
        key_projection=(((1.0,), (0.0,)), ((0.0,), (1.0,))),
        key_bias=((0.0, 0.0), (0.0, 0.0)),
        value_projection=(((0.5,), (0.0,)), ((0.0,), (0.5,))),
        value_bias=((0.0, 0.0), (0.0, 0.0)),
        reference_key_norms=(1.0, 1.0),
        reference_value_norms=(1.0, 1.0),
        norm_cap=0.1,
        source_fingerprint="state-kv-after-common-adapter",
        sample_count=2,
        training_mode=CHARACTER_TEACHER_FORCED_PREFIX_TRAINING_MODE,
        vector_labels=("state",),
        description="test prefix",
    )


def _checkpoint():
    layer = SubstrateDeltaAdapterLayer(
        layer_index=0,
        delta_vector=(0.01, -0.01),
        mean_abs_delta=0.01,
        description="test layer",
    )
    return SubstrateRareHeavyCheckpoint(
        checkpoint_id="common-v1",
        model_id=MODEL_ID,
        runtime_origin="hf-local",
        control_scale=0.1,
        semantic_text_weight=0.5,
        semantic_residual_weight=0.5,
        semantic_anchor_bias=(0.0, 0.0, 0.0, 0.0, 0.0),
        update_count=1,
        source_batch_count=1,
        mean_sequence_length=4.0,
        mean_residual_magnitude=0.01,
        description="test checkpoint",
        checkpoint_version=2,
        training_mode="adapter-delta-v2",
        compatibility_fingerprint="runtime-fingerprint",
        adapter_scale=1.0,
        adapter_parameter_count=2,
        adapter_training_loss=0.2,
        adapter_layers=(layer,),
    )


def _control():
    return ControlBasisArtifact(
        model_id=MODEL_ID,
        hidden_size=2,
        basis=((1.0, 0.0),),
        layer_indices=(0,),
        layer_gains=(1.0,),
        training_mode="train-transition-pca-v1",
        source_fingerprint="control-source",
        sample_count=2,
        description="test control",
    )


def _bundle(*, decision: str = "allow"):
    return CommonAdapterBundle.create(
        common_adapter_version="common-v1",
        base_model_id=MODEL_ID,
        base_model_weights_sha256="a" * 64,
        rare_heavy_checkpoint=_checkpoint(),
        state_kv_artifact=_prefix(),
        control_basis_artifact=_control(),
        gate_record=CommonAdapterGateRecord(
            proposal_id="proposal-common-v1",
            decision=decision,
            desired_gate="offline",
            validation_delta=0.1,
            capacity_cost=0.2,
            rollback_evidence="rollback:common-v0",
            is_reversible=True,
            evaluation_ref="evaluation:common-v1",
        ),
        description="shared adapter test bundle",
    )


def test_common_adapter_bundle_round_trip_and_gate() -> None:
    bundle = _bundle()
    restored = CommonAdapterBundle.from_json(bundle.to_json())

    assert restored == bundle
    assert restored.active_eligible
    restored.require_active()
    assert len(restored.bundle_id) == 64


def test_common_adapter_bundle_rejects_nested_model_drift() -> None:
    with pytest.raises(ValueError, match="nested carrier model ids"):
        CommonAdapterBundle.create(
            common_adapter_version="common-v1",
            base_model_id="Qwen/other",
            base_model_weights_sha256="b" * 64,
            rare_heavy_checkpoint=_checkpoint(),
            state_kv_artifact=_prefix(),
            control_basis_artifact=_control(),
            gate_record=_bundle().gate_record,
            description="bad",
        )


def test_common_adapter_bundle_fails_closed_on_denied_gate() -> None:
    bundle = _bundle(decision="deny")
    with pytest.raises(ValueError, match="not ACTIVE-eligible"):
        bundle.require_active()


def test_common_adapter_bundle_detects_payload_tampering() -> None:
    bundle = _bundle()
    with pytest.raises(ValueError, match="bundle_id"):
        replace(bundle, description="tampered")


def test_standalone_checkpoint_round_trip() -> None:
    checkpoint = _checkpoint()

    restored = rare_heavy_checkpoint_from_json(
        rare_heavy_checkpoint_to_json(checkpoint)
    )

    assert restored == checkpoint


def test_model_weight_fingerprint_binds_relative_paths_and_bytes(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "model.safetensors").write_bytes(b"weights")
    (second / "renamed.safetensors").write_bytes(b"weights")

    first_digest = fingerprint_model_weight_files(first)
    second_digest = fingerprint_model_weight_files(second)

    assert len(first_digest) == 64
    assert first_digest != second_digest


def test_offline_hook_applies_and_removes_the_rare_heavy_delta() -> None:
    torch = pytest.importorskip("torch")

    class Block(torch.nn.Module):
        def forward(self, values):
            return values

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=2)
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Block()])

        def forward(self, values):
            return self.model.layers[0](values)

    model = ToyModel()
    values = torch.zeros((1, 1, 2), dtype=torch.float32)
    handles = install_rare_heavy_checkpoint_hooks(
        model=model,
        checkpoint=_checkpoint(),
        expected_model_id=MODEL_ID,
    )

    adjusted = model(values)
    remove_forward_hooks(handles)
    restored = model(values)

    assert adjusted.tolist()[0][0] == pytest.approx([0.01, -0.01])
    assert torch.equal(restored, values)
