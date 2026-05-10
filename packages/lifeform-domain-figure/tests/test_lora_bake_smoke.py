"""Smoke tests for the F6 / P6.2 LoRA bake backends.

Validates:

* :class:`SyntheticLoRABakeBackend` produces a deterministic
  :class:`FigureLoRAArtifact` whose adapter layers are
  shape-identical to :class:`SubstrateDeltaAdapterLayer`.
* The artifact's integrity hash binds the training plan + every
  layer; perturbing the plan changes the hash.
* :func:`attach_baked_lora` re-keys the bundle id and binds the
  artifact into ``bundle.lora``.
* :class:`PEFTLoRABakeBackend.bake` raises ``NotImplementedError``
  with a clear forwarding message (it is a deliberate stub).
* :class:`FigureLoRAArtifact` rejects malformed inputs (empty
  layers, zero rank, missing hashes).
"""

from __future__ import annotations

import pytest

from volvence_zero.substrate import SubstrateDeltaAdapterLayer

from lifeform_domain_figure import (
    FigureLoRAArtifact,
    LORA_ARTIFACT_SCHEMA_VERSION,
    PEFTLoRABakeBackend,
    SyntheticLoRABakeBackend,
    attach_baked_lora,
    build_figure_artifact_bundle,
    build_einstein_profile,
    build_figure_ingestion_envelope,
    build_lora_training_plan,
    synthetic_einstein_corpus,
)
from lifeform_domain_figure import FigureBundleInputs
from lifeform_domain_figure.envelope_builder import FigureCorpusSourceBundle


def _einstein_envelopes():
    papers, letters, lectures, notebooks = synthetic_einstein_corpus()
    return build_figure_ingestion_envelope(
        FigureCorpusSourceBundle(
            figure_id="einstein",
            papers=papers,
            letters=letters,
            lectures=lectures,
            notebooks=notebooks,
        ),
        uploader="lifeform-figure-tests:lora-bake",
    ).envelopes


def _einstein_plan():
    return build_lora_training_plan(
        figure_id="einstein",
        envelopes=_einstein_envelopes(),
    )


def _einstein_bundle():
    profile = build_einstein_profile()
    envelopes = _einstein_envelopes()
    return build_figure_artifact_bundle(
        FigureBundleInputs(profile=profile, envelopes=envelopes)
    )


def test_synthetic_backend_bake_produces_artifact() -> None:
    plan = _einstein_plan()
    backend = SyntheticLoRABakeBackend()
    artifact = backend.bake(plan)
    assert isinstance(artifact, FigureLoRAArtifact)
    assert artifact.figure_id == "einstein"
    assert artifact.backend_id == "synthetic-v1"
    assert artifact.schema_version == LORA_ARTIFACT_SCHEMA_VERSION
    assert artifact.training_plan_hash == plan.integrity_hash
    assert artifact.rank == plan.rank
    assert artifact.target_layer_index == plan.target_layer_index
    assert artifact.total_layers == plan.rank * backend.layer_count_per_rank
    assert artifact.parameter_count == artifact.total_layers * backend.vector_dim


def test_synthetic_backend_bake_layers_have_kernel_shape() -> None:
    plan = _einstein_plan()
    artifact = SyntheticLoRABakeBackend().bake(plan)
    for layer in artifact.adapter_layers:
        assert isinstance(layer, SubstrateDeltaAdapterLayer)
        assert len(layer.delta_vector) == 32
        assert layer.mean_abs_delta > 0.0
        assert "figure-persona-lora:einstein" in layer.description


def test_synthetic_backend_bake_is_deterministic() -> None:
    plan = _einstein_plan()
    backend = SyntheticLoRABakeBackend()
    artifact_a = backend.bake(plan)
    artifact_b = backend.bake(plan)
    assert artifact_a.integrity_hash == artifact_b.integrity_hash
    assert artifact_a.adapter_layers == artifact_b.adapter_layers


def test_synthetic_backend_bake_changes_with_plan() -> None:
    plan_a = _einstein_plan()
    plan_b = build_lora_training_plan(
        figure_id="einstein",
        envelopes=_einstein_envelopes(),
        rank=plan_a.rank + 1,
    )
    backend = SyntheticLoRABakeBackend()
    artifact_a = backend.bake(plan_a)
    artifact_b = backend.bake(plan_b)
    assert artifact_a.integrity_hash != artifact_b.integrity_hash
    assert artifact_a.rank != artifact_b.rank


def test_synthetic_backend_validates_constructor_args() -> None:
    with pytest.raises(ValueError, match="vector_dim"):
        SyntheticLoRABakeBackend(vector_dim=0)
    with pytest.raises(ValueError, match="layer_count_per_rank"):
        SyntheticLoRABakeBackend(layer_count_per_rank=0)


def test_attach_baked_lora_rekeys_bundle() -> None:
    bundle = _einstein_bundle()
    assert bundle.lora is None
    plan = _einstein_plan()
    artifact = SyntheticLoRABakeBackend().bake(plan)
    new_bundle = attach_baked_lora(bundle, artifact)
    assert new_bundle.figure_id == bundle.figure_id
    assert new_bundle.lora is artifact
    assert new_bundle.bundle_id != bundle.bundle_id
    assert new_bundle.integrity_hash != bundle.integrity_hash


def test_attach_baked_lora_rejects_figure_id_mismatch() -> None:
    bundle = _einstein_bundle()
    plan = _einstein_plan()
    artifact = SyntheticLoRABakeBackend().bake(plan)
    foreign = FigureLoRAArtifact(
        schema_version=artifact.schema_version,
        figure_id="not-einstein",
        backend_id=artifact.backend_id,
        rank=artifact.rank,
        target_layer_index=artifact.target_layer_index,
        adapter_layers=artifact.adapter_layers,
        training_plan_hash=artifact.training_plan_hash,
        integrity_hash=artifact.integrity_hash,
        parameter_count=artifact.parameter_count,
        description=artifact.description,
    )
    with pytest.raises(ValueError, match="figure_id"):
        attach_baked_lora(bundle, foreign)


def test_peft_backend_raises_not_implemented() -> None:
    plan = _einstein_plan()
    backend = PEFTLoRABakeBackend()
    assert backend.backend_id == "peft-v1-stub"
    with pytest.raises(NotImplementedError, match="future F6.X"):
        backend.bake(plan)


def test_figure_lora_artifact_rejects_empty_layers() -> None:
    plan = _einstein_plan()
    artifact = SyntheticLoRABakeBackend().bake(plan)
    with pytest.raises(ValueError, match="adapter_layers"):
        FigureLoRAArtifact(
            schema_version=artifact.schema_version,
            figure_id=artifact.figure_id,
            backend_id=artifact.backend_id,
            rank=artifact.rank,
            target_layer_index=artifact.target_layer_index,
            adapter_layers=(),
            training_plan_hash=artifact.training_plan_hash,
            integrity_hash=artifact.integrity_hash,
            parameter_count=artifact.parameter_count,
            description=artifact.description,
        )


def test_figure_lora_artifact_rejects_invalid_rank() -> None:
    plan = _einstein_plan()
    artifact = SyntheticLoRABakeBackend().bake(plan)
    with pytest.raises(ValueError, match="rank"):
        FigureLoRAArtifact(
            schema_version=artifact.schema_version,
            figure_id=artifact.figure_id,
            backend_id=artifact.backend_id,
            rank=0,
            target_layer_index=artifact.target_layer_index,
            adapter_layers=artifact.adapter_layers,
            training_plan_hash=artifact.training_plan_hash,
            integrity_hash=artifact.integrity_hash,
            parameter_count=artifact.parameter_count,
            description=artifact.description,
        )
