"""Versioned generic-bank residual projector contract tests."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from volvence_zero.conditioning_bank_contracts import (
    CONDITIONING_BANK_LATENT_CARRIER_SCHEMA_VERSION,
    CONDITIONING_BANK_SCHEMA_VERSION,
    ConditioningBankLatentCarrier,
    ConditioningBankSnapshot,
    ConditioningBankType,
    ConditioningRevocationState,
    ConditioningScope,
)
from volvence_zero.substrate.conditioning_bank_projector import (
    RELATIONSHIP_RESIDUAL_DEFAULT_SCALE,
    RELATIONSHIP_RESIDUAL_PROJECTOR_VERSION,
    RelationshipConditioningProjectorArtifact,
    build_relationship_contrastive_projector_artifact,
    build_conditioning_bank_residual_basis,
    build_conditioning_bank_residual_delta,
    load_relationship_projector_basis,
)
from volvence_zero.substrate.residual_synthetic import (
    SyntheticOpenWeightResidualRuntime,
)
from volvence_zero.substrate.residual_backend import (
    TransformersOpenWeightResidualRuntime,
)


def _bank(
    *,
    bank_type: ConditioningBankType = ConditioningBankType.RELATIONSHIP,
    confidence: float = 0.8,
    freshness: float = 0.75,
) -> ConditioningBankSnapshot:
    return ConditioningBankSnapshot(
        schema_version=CONDITIONING_BANK_SCHEMA_VERSION,
        bank_type=bank_type,
        scope=ConditioningScope(
            tenant_scope="tenant-1",
            user_scope="user-1",
            session_scope="session-1",
        ),
        readout=(0.2, 0.7, 0.9),
        readout_labels=("rel_a", "rel_b", "rel_c"),
        source_versions=(("relationship_state", 3),),
        source_fingerprint="relationship-projector-test",
        confidence=confidence,
        freshness=freshness,
        consent_version=0,
        provenance="owner:RelationshipConditioningModule/test",
        revocation_state=ConditioningRevocationState.ACTIVE,
        is_cold_start=False,
        description="typed Relationship projector test bank",
    )


def _carrier(
    *,
    bank: ConditioningBankSnapshot | None = None,
    projector_version: str = RELATIONSHIP_RESIDUAL_PROJECTOR_VERSION,
) -> ConditioningBankLatentCarrier:
    return ConditioningBankLatentCarrier(
        schema_version=CONDITIONING_BANK_LATENT_CARRIER_SCHEMA_VERSION,
        bank=bank or _bank(),
        carrier="residual",
        projector_version=projector_version,
        scale=RELATIONSHIP_RESIDUAL_DEFAULT_SCALE,
        description="Relationship residual projector test carrier",
    )


def _artifact(
    *,
    labels: tuple[str, ...] = ("rel_a", "rel_b", "rel_c"),
    width: int = 4,
) -> RelationshipConditioningProjectorArtifact:
    return build_relationship_contrastive_projector_artifact(
        model_id="Qwen/test",
        hidden_size=width,
        vector_labels=labels,
        layer_indices=(1, 2),
        contrastive_rows={
            label: tuple(
                1.0 if index == offset % width else 0.0
                for index in range(width)
            )
            for offset, label in enumerate(labels)
        },
        source_fingerprint="weights-and-relationship-anchors",
        sample_count=2 * len(labels),
    )


def test_carrier_rejects_inert_bank_and_unbounded_scale() -> None:
    active = _bank()
    with pytest.raises(ValueError, match="injectable bank"):
        _carrier(bank=replace(active, freshness=0.0))
    with pytest.raises(ValueError, match=r"\(0, 0.12\]"):
        replace(_carrier(), scale=0.13)


def test_relationship_basis_is_deterministic_and_row_normalized() -> None:
    torch = pytest.importorskip("torch")
    left = build_conditioning_bank_residual_basis(
        torch_module=torch,
        hidden_size=32,
        vector_dim=3,
    )
    right = build_conditioning_bank_residual_basis(
        torch_module=torch,
        hidden_size=32,
        vector_dim=3,
    )
    assert torch.equal(left, right)
    assert torch.allclose(
        left.norm(dim=1),
        torch.ones(3),
        atol=1e-5,
    )


def test_relationship_delta_scales_with_confidence_and_freshness() -> None:
    torch = pytest.importorskip("torch")
    full = build_conditioning_bank_residual_delta(
        torch_module=torch,
        carrier=_carrier(bank=_bank(confidence=0.8, freshness=1.0)),
        hidden_size=32,
    )
    half = build_conditioning_bank_residual_delta(
        torch_module=torch,
        carrier=_carrier(bank=_bank(confidence=0.4, freshness=1.0)),
        hidden_size=32,
    )
    stale = build_conditioning_bank_residual_delta(
        torch_module=torch,
        carrier=_carrier(bank=_bank(confidence=0.8, freshness=0.5)),
        hidden_size=32,
    )
    assert float(full.norm()) > 0.0
    assert float(half.norm()) == pytest.approx(
        0.5 * float(full.norm()), rel=1e-5
    )
    assert float(stale.norm()) == pytest.approx(
        0.5 * float(full.norm()), rel=1e-5
    )
    assert build_conditioning_bank_residual_delta(
        torch_module=torch,
        carrier=_carrier(
            bank=replace(_bank(), readout=(0.5, 0.5, 0.5))
        ),
        hidden_size=32,
    ) is None


def test_projector_fails_loudly_on_wrong_bank_or_version() -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="RELATIONSHIP"):
        build_conditioning_bank_residual_delta(
            torch_module=torch,
            carrier=_carrier(bank=_bank(bank_type=ConditioningBankType.TASK)),
            hidden_size=32,
        )
    with pytest.raises(ValueError, match="unsupported"):
        build_conditioning_bank_residual_delta(
            torch_module=torch,
            carrier=_carrier(projector_version="relationship-residual.future"),
            hidden_size=32,
        )


def test_relationship_artifact_round_trips_and_detects_tampering() -> None:
    artifact = _artifact()

    restored = RelationshipConditioningProjectorArtifact.from_json(
        artifact.to_json()
    )

    assert restored == artifact
    assert restored.projector_version.endswith(artifact.artifact_id)
    raw = json.loads(artifact.to_json())
    raw["description"] = "tampered"
    with pytest.raises(ValueError, match="artifact_id"):
        RelationshipConditioningProjectorArtifact.from_json(json.dumps(raw))


def test_relationship_artifact_load_checks_runtime_compatibility() -> None:
    torch = pytest.importorskip("torch")
    artifact = _artifact()

    basis, gains = load_relationship_projector_basis(
        torch_module=torch,
        artifact=artifact,
        expected_model_id="Qwen/test",
        expected_hidden_size=4,
        available_layer_indices=(1, 2),
        device="cpu",
    )

    assert tuple(basis.shape) == (3, 4)
    assert gains == {1: 1.0, 2: 1.0}
    with pytest.raises(ValueError, match="model_id"):
        load_relationship_projector_basis(
            torch_module=torch,
            artifact=artifact,
            expected_model_id="Other/model",
            expected_hidden_size=4,
            available_layer_indices=(1, 2),
            device="cpu",
        )
    with pytest.raises(ValueError, match="not hooked"):
        load_relationship_projector_basis(
            torch_module=torch,
            artifact=artifact,
            expected_model_id="Qwen/test",
            expected_hidden_size=4,
            available_layer_indices=(1,),
            device="cpu",
        )


def test_relationship_artifact_delta_checks_labels_and_version() -> None:
    torch = pytest.importorskip("torch")
    artifact = _artifact()
    basis, _ = load_relationship_projector_basis(
        torch_module=torch,
        artifact=artifact,
        expected_model_id="Qwen/test",
        expected_hidden_size=4,
        available_layer_indices=(1, 2),
        device="cpu",
    )
    carrier = _carrier(
        projector_version=artifact.projector_version,
    )

    delta = build_conditioning_bank_residual_delta(
        torch_module=torch,
        carrier=carrier,
        hidden_size=4,
        basis=basis,
        vector_labels=artifact.vector_labels,
        expected_projector_version=artifact.projector_version,
    )

    assert float(delta.norm()) > 0.0
    with pytest.raises(ValueError, match="readout_labels"):
        build_conditioning_bank_residual_delta(
            torch_module=torch,
            carrier=carrier,
            hidden_size=4,
            basis=basis,
            vector_labels=("wrong_a", "wrong_b", "wrong_c"),
            expected_projector_version=artifact.projector_version,
        )


def test_relationship_hook_delta_does_not_inherit_personal_layer_gain() -> None:
    torch = pytest.importorskip("torch")
    runtime = object.__new__(TransformersOpenWeightResidualRuntime)
    runtime._torch = torch
    runtime._personal_conditioning_layer_gains = {1: 0.0}
    runtime._rare_heavy_adapter_deltas = {}
    runtime._online_fast_adapter_deltas = {}
    runtime._rare_heavy_adapter_scale = 0.0
    runtime._online_fast_delta_scale = 0.0
    captured: dict[int, object] = {}
    hook = runtime._make_capture_hook(
        layer_index=1,
        captured_layers=captured,
        control_delta=None,
        capture_residuals=True,
        personal_delta=torch.ones(4),
        conditioning_bank_delta=torch.full((4,), 2.0),
    )

    adjusted = hook(
        None,
        (),
        torch.zeros((1, 1, 4), dtype=torch.float32),
    )

    assert torch.equal(adjusted, torch.full((1, 1, 4), 2.0))
    assert torch.equal(captured[1], adjusted)


def test_synthetic_runtime_traces_carrier_without_claiming_injection() -> None:
    runtime = SyntheticOpenWeightResidualRuntime()
    carrier = _carrier()
    result = runtime.generate(
        prompt="hello",
        conditioning_bank_carriers=(carrier,),
    )
    assert runtime.conditioning_bank_carrier_trace == [carrier]
    assert result.conditioning_bank_carriers_applied == ()
    assert "trace-only" in result.description
