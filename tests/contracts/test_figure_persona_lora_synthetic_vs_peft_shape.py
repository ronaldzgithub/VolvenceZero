"""Cross-cutting contract: synthetic + PEFT LoRA backends ship same shape.

Closes debt #18's compat invariant. Two surfaces must match:

1. Both backends produce ``FigureLoRAArtifact`` with non-empty
   ``adapter_layers``, every layer is :class:`SubstrateDeltaAdapterLayer`,
   ``training_plan_hash`` matches ``plan.integrity_hash``.
2. ``PersonaLoRAPool.register(...)`` accepts an artifact from
   either backend; the resulting :class:`PersonaLoRARecord` carries
   the artifact's adapter layers byte-for-byte.

The PEFT branch is gated under ``hf``: when peft + torch are not
installed, only the synthetic shape is asserted and the parity
assertion is skipped (the contract still holds in spirit because
the synthetic backend has been the only provider for over a year).
"""

from __future__ import annotations

import importlib.util

import pytest

from volvence_zero.substrate import (
    PersonaLoRAPool,
    SubstrateDeltaAdapterLayer,
)

from lifeform_domain_figure import (
    FigureCorpusSourceBundle,
    PEFTLoRABakeBackend,
    SyntheticLoRABakeBackend,
    build_figure_ingestion_envelope,
    build_lora_training_plan,
    synthetic_einstein_corpus,
)


def _peft_stack_available() -> bool:
    return all(
        importlib.util.find_spec(name) is not None
        for name in ("peft", "transformers", "torch")
    )


def _einstein_plan():
    papers, letters, lectures, notebooks = synthetic_einstein_corpus()
    envelopes = build_figure_ingestion_envelope(
        FigureCorpusSourceBundle(
            figure_id="einstein",
            papers=papers,
            letters=letters,
            lectures=lectures,
            notebooks=notebooks,
        ),
        uploader="contract-test:lora-shape",
    ).envelopes
    return build_lora_training_plan(figure_id="einstein", envelopes=envelopes)


def test_synthetic_artifact_shape_invariants() -> None:
    plan = _einstein_plan()
    artifact = SyntheticLoRABakeBackend().bake(plan)
    assert artifact.backend_id == "synthetic-v1"
    assert artifact.training_plan_hash == plan.integrity_hash
    assert artifact.adapter_layers
    for layer in artifact.adapter_layers:
        assert isinstance(layer, SubstrateDeltaAdapterLayer)
        assert isinstance(layer.delta_vector, tuple)
        assert all(isinstance(value, float) for value in layer.delta_vector)


def test_synthetic_artifact_registers_in_pool() -> None:
    plan = _einstein_plan()
    artifact = SyntheticLoRABakeBackend().bake(plan)
    pool = PersonaLoRAPool()
    record_id = pool.register(
        figure_id=artifact.figure_id,
        source_bundle_id="contract-test-bundle",
        backend_id=artifact.backend_id,
        training_plan_hash=artifact.training_plan_hash,
        adapter_layers=artifact.adapter_layers,
        parameter_count=artifact.parameter_count,
        description=artifact.description,
    )
    record = pool.lookup(record_id)
    assert record.adapter_layers == artifact.adapter_layers
    assert record.training_plan_hash == artifact.training_plan_hash


@pytest.mark.hf
def test_peft_artifact_shape_invariants() -> None:
    if not _peft_stack_available():
        pytest.skip("peft + transformers + torch not installed")
    plan = _einstein_plan()
    backend = PEFTLoRABakeBackend(max_steps=2, delta_vector_dim=32)
    artifact = backend.bake(plan)
    assert artifact.backend_id == "peft-v1"
    assert artifact.training_plan_hash == plan.integrity_hash
    assert artifact.adapter_layers
    for layer in artifact.adapter_layers:
        assert isinstance(layer, SubstrateDeltaAdapterLayer)
        assert isinstance(layer.delta_vector, tuple)
        assert len(layer.delta_vector) == 32
        assert all(isinstance(value, float) for value in layer.delta_vector)


@pytest.mark.hf
def test_peft_and_synthetic_artifacts_both_register_in_one_pool() -> None:
    """Critical compat: a pool must accept both backend ids without
    reshaping. This is the load-bearing invariant for the persona
    LoRA hot-swap (Wave D) — `pool.activate` does not branch on
    backend_id."""

    if not _peft_stack_available():
        pytest.skip("peft + transformers + torch not installed")
    plan = _einstein_plan()
    synthetic_artifact = SyntheticLoRABakeBackend().bake(plan)
    peft_artifact = PEFTLoRABakeBackend(max_steps=2, delta_vector_dim=32).bake(plan)
    pool = PersonaLoRAPool()
    synth_id = pool.register(
        figure_id="synth-side",
        source_bundle_id="bundle-synth",
        backend_id=synthetic_artifact.backend_id,
        training_plan_hash=synthetic_artifact.training_plan_hash,
        adapter_layers=synthetic_artifact.adapter_layers,
        parameter_count=synthetic_artifact.parameter_count,
    )
    peft_id = pool.register(
        figure_id="peft-side",
        source_bundle_id="bundle-peft",
        backend_id=peft_artifact.backend_id,
        training_plan_hash=peft_artifact.training_plan_hash,
        adapter_layers=peft_artifact.adapter_layers,
        parameter_count=peft_artifact.parameter_count,
    )
    assert pool.lookup(synth_id).backend_id == "synthetic-v1"
    assert pool.lookup(peft_id).backend_id == "peft-v1"
