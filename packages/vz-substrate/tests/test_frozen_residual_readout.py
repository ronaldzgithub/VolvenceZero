from __future__ import annotations

from dataclasses import replace
import hashlib
import math
import struct

import pytest
import torch

from volvence_zero.substrate import (
    FrozenResidualReadoutArtifact,
    SubstrateFingerprint,
    SubstrateForwardRepresentation,
    SubstrateForwardRepresentationLineage,
    SubstrateForwardRepresentationSnapshot,
    SubstrateResidualReadoutPublisher,
    fit_frozen_residual_readout,
)


def _vector_sha(values: tuple[float, ...]) -> str:
    return hashlib.sha256(struct.pack(f"!{len(values)}d", *values)).hexdigest()


def _normalized(*values: float) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in values))
    return tuple(value / norm for value in values)


def _snapshot(
    rows: tuple[tuple[str, tuple[float, ...]], ...],
    *,
    fingerprint_char: str,
) -> SubstrateForwardRepresentationSnapshot:
    representations = tuple(
        SubstrateForwardRepresentation(
            sample_id=sample_id,
            source_sha256=hashlib.sha256(sample_id.encode("utf-8")).hexdigest(),
            values=values,
            values_sha256=_vector_sha(values),
        )
        for sample_id, values in rows
    )
    return SubstrateForwardRepresentationSnapshot(
        lineage=SubstrateForwardRepresentationLineage(
            schema_version="substrate-forward-representation.v1",
            snapshot_fingerprint=fingerprint_char * 64,
            model_fingerprint=SubstrateFingerprint(
                model_id="synthetic-readout",
                version="fixture-v1",
                weights_sha256="a" * 64,
            ),
            runtime_origin="synthetic-fixture",
            readout_kind="latest-token-selected-layer-residual-l2.v1",
            layer_indices=(2,),
            activation_widths=(2,),
            representation_dim=2,
        ),
        representations=representations,
        description="Synthetic normalized residual representations.",
    )


def _training_snapshot() -> SubstrateForwardRepresentationSnapshot:
    return _snapshot(
        (
            ("a-1", _normalized(1.0, 0.1)),
            ("a-2", _normalized(1.0, -0.2)),
            ("a-3", _normalized(0.9, 0.3)),
            ("b-1", _normalized(-1.0, 0.1)),
            ("b-2", _normalized(-1.0, -0.2)),
            ("b-3", _normalized(-0.9, 0.3)),
        ),
        fingerprint_char="b",
    )


def test_frozen_readout_round_trips_and_publishes_heldout_scores() -> None:
    training = _training_snapshot()
    artifact = fit_frozen_residual_readout(
        torch_module=torch,
        snapshot=training,
        labels=(
            ("a-1", "class-a"),
            ("a-2", "class-a"),
            ("a-3", "class-a"),
            ("b-1", "class-b"),
            ("b-2", "class-b"),
            ("b-3", "class-b"),
        ),
        class_ids=("class-a", "class-b"),
        ridge_alpha=1.0,
    )

    restored = FrozenResidualReadoutArtifact.from_json(artifact.to_json())
    assert restored == artifact
    assert restored.artifact_id == artifact.artifact_id
    assert all(
        math.isclose(
            math.sqrt(sum(value * value for value in axis)),
            1.0,
            abs_tol=1e-8,
        )
        for axis in artifact.class_axes
    )
    assert artifact.axis_for("class-a") == artifact.class_axes[0]

    heldout = _snapshot(
        (
            ("eval-a", _normalized(0.8, -0.1)),
            ("eval-b", _normalized(-0.8, 0.1)),
        ),
        fingerprint_char="c",
    )
    readout = SubstrateResidualReadoutPublisher(restored).publish(heldout)

    assert tuple(row.predicted_class_id for row in readout.readouts) == (
        "class-a",
        "class-b",
    )
    assert all(row.score_margin > 0.0 for row in readout.readouts)
    assert readout.lineage.artifact_id == artifact.artifact_id
    assert readout.lineage.source_snapshot_fingerprint == "c" * 64
    assert "eval-a" not in artifact.to_json()


def test_frozen_readout_rejects_label_or_lineage_drift() -> None:
    training = _training_snapshot()
    with pytest.raises(ValueError, match="match the training snapshot exactly"):
        fit_frozen_residual_readout(
            torch_module=torch,
            snapshot=training,
            labels=(("a-1", "class-a"),),
            class_ids=("class-a", "class-b"),
        )

    artifact = fit_frozen_residual_readout(
        torch_module=torch,
        snapshot=training,
        labels=tuple(
            (
                row.sample_id,
                "class-a" if row.sample_id.startswith("a-") else "class-b",
            )
            for row in training.representations
        ),
        class_ids=("class-a", "class-b"),
    )
    drifted = replace(
        training,
        lineage=replace(training.lineage, runtime_origin="different-runtime"),
    )
    with pytest.raises(ValueError, match="representation lineage mismatch"):
        SubstrateResidualReadoutPublisher(artifact).publish(drifted)

    with pytest.raises(KeyError, match="unknown frozen residual readout class_id"):
        artifact.axis_for("missing")


def test_frozen_readout_artifact_rejects_tampered_payload() -> None:
    artifact = fit_frozen_residual_readout(
        torch_module=torch,
        snapshot=_training_snapshot(),
        labels=(
            ("a-1", "class-a"),
            ("a-2", "class-a"),
            ("a-3", "class-a"),
            ("b-1", "class-b"),
            ("b-2", "class-b"),
            ("b-3", "class-b"),
        ),
        class_ids=("class-a", "class-b"),
    )
    tampered = artifact.to_json().replace(
        '"ridge_alpha": 1.0',
        '"ridge_alpha": 2.0',
    )
    with pytest.raises(ValueError, match="artifact_id mismatch"):
        FrozenResidualReadoutArtifact.from_json(tampered)
