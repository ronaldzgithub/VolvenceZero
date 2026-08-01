from __future__ import annotations

import dataclasses

import pytest

from volvence_zero.prediction import ForwardRepresentationBatch, PredictionErrorModule
from volvence_zero.substrate import (
    SUBSTRATE_FORWARD_READOUT_KIND,
    SUBSTRATE_FORWARD_REPRESENTATION_SCHEMA_VERSION,
    SubstrateFingerprint,
    SubstrateForwardRepresentationLineage,
)


def _target_lineage(*, representation_dim: int = 2):
    return SubstrateForwardRepresentationLineage(
        schema_version=SUBSTRATE_FORWARD_REPRESENTATION_SCHEMA_VERSION,
        snapshot_fingerprint="2" * 64,
        model_fingerprint=SubstrateFingerprint(
            model_id="fixture-substrate",
            version="fixture-v1",
            weights_sha256="1" * 64,
        ),
        runtime_origin="fixture-frozen-runtime",
        readout_kind=SUBSTRATE_FORWARD_READOUT_KIND,
        layer_indices=(0,),
        activation_widths=(representation_dim,),
        representation_dim=representation_dim,
    )


def _batch() -> ForwardRepresentationBatch:
    return ForwardRepresentationBatch(
        batch_id="fixture",
        sample_ids=("a", "b", "c", "d"),
        context_representations=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
        ),
        target_representations=(
            (0.9, 0.1),
            (0.1, 0.9),
            (0.2, 0.8),
            (0.8, 0.2),
        ),
        persistence_representations=(
            (1.0, 0.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (1.0, 0.0),
        ),
        history_turns=(1, 2, 3, 4),
        target_lineage=_target_lineage(),
    )


def test_pe_owner_trains_and_settles_exact_n_plus_one_targets() -> None:
    module = PredictionErrorModule()
    module.configure_forward_representation_head(
        input_dim=3,
        target_dim=2,
        n_z=3,
        seed=7,
        learning_rate=0.05,
        device="cpu",
    )
    first = module.process_forward_representation_batch(_batch(), update=False)
    for _ in range(100):
        module.process_forward_representation_batch(_batch(), update=True)
    final = module.process_forward_representation_batch(_batch(), update=False)
    assert final.mean_squared_error < first.mean_squared_error
    assert final.sample_count == 4
    assert final.settlements[0].actual_representation == (0.9, 0.1)
    assert final.settlements[0].signed_error == pytest.approx(
        tuple(
            actual - predicted
            for actual, predicted in zip(
                final.settlements[0].actual_representation,
                final.settlements[0].predicted_representation,
                strict=True,
            )
        )
    )


def test_forward_checkpoint_is_float_only_and_restorable() -> None:
    module = PredictionErrorModule()
    module.configure_forward_representation_head(
        input_dim=3, target_dim=2, n_z=4, seed=2, device="cpu"
    )
    module.process_forward_representation_batch(_batch(), update=True)
    checkpoint = module.export_forward_representation_checkpoint(
        checkpoint_id="rollback"
    )
    before = module.process_forward_representation_batch(_batch(), update=False)
    module.process_forward_representation_batch(_batch(), update=True)
    module.restore_forward_representation_checkpoint(checkpoint)
    restored = module.process_forward_representation_batch(_batch(), update=False)
    assert restored.parameter_fingerprint == checkpoint.parameter_fingerprint
    assert restored.settlements == before.settlements
    assert all(
        isinstance(value, float)
        for _, _, values in checkpoint.parameter_values
        for value in values
    )


def test_forward_batch_fails_loudly_on_geometry_and_schema_mismatch() -> None:
    module = PredictionErrorModule()
    with pytest.raises(RuntimeError, match="not configured"):
        module.process_forward_representation_batch(_batch(), update=False)
    module.configure_forward_representation_head(
        input_dim=3, target_dim=2, n_z=3, seed=0, device="cpu"
    )
    module.process_forward_representation_batch(_batch(), update=False)
    checkpoint = module.export_forward_representation_checkpoint(checkpoint_id="x")
    with pytest.raises(ValueError, match="schema mismatch"):
        module.restore_forward_representation_checkpoint(
            dataclasses.replace(checkpoint, schema_version="bad.v0")
        )


def test_forward_batch_rejects_non_finite_target() -> None:
    with pytest.raises(ValueError, match="finite"):
        dataclasses.replace(
            _batch(),
            target_representations=((float("nan"), 0.0),) + _batch().target_representations[1:],
        )


def test_forward_batch_rejects_target_lineage_geometry_mismatch() -> None:
    with pytest.raises(ValueError, match="target lineage dimension mismatch"):
        dataclasses.replace(_batch(), target_lineage=_target_lineage(representation_dim=3))


def test_forward_head_rejects_target_lineage_change_after_binding() -> None:
    module = PredictionErrorModule()
    module.configure_forward_representation_head(
        input_dim=3, target_dim=2, n_z=3, seed=0, device="cpu"
    )
    module.process_forward_representation_batch(_batch(), update=False)
    changed_lineage = dataclasses.replace(
        _target_lineage(), snapshot_fingerprint="3" * 64
    )
    with pytest.raises(ValueError, match="target lineage changed"):
        module.process_forward_representation_batch(
            dataclasses.replace(_batch(), target_lineage=changed_lineage),
            update=False,
        )
