from __future__ import annotations

import dataclasses

import pytest

from volvence_zero.prediction import ForwardRepresentationBatch, PredictionErrorModule


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
