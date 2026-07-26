"""Learned personal-conditioning projector artifact contracts."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from volvence_zero.personal_conditioning_contracts import (
    PERSONAL_CONDITIONING_VECTOR_LABELS,
)
from volvence_zero.substrate.personal_conditioning_projector import (
    PersonalConditioningProjectorArtifact,
    build_contrastive_projector_artifact,
    load_projector_basis,
)


def _rows(*, width: int = 4) -> dict[str, tuple[float, ...]]:
    return {
        label: tuple(
            1.0 if index == offset % width else 0.0
            for index in range(width)
        )
        for offset, label in enumerate(PERSONAL_CONDITIONING_VECTOR_LABELS)
    }


def _artifact(*, width: int = 4) -> PersonalConditioningProjectorArtifact:
    return build_contrastive_projector_artifact(
        model_id="Qwen/test",
        hidden_size=width,
        layer_indices=(1, 2, 3),
        contrastive_rows=_rows(width=width),
        source_fingerprint="weights-and-anchor-fingerprint",
        sample_count=32,
    )


def test_contrastive_artifact_is_normalized_and_round_trips() -> None:
    artifact = _artifact()

    restored = PersonalConditioningProjectorArtifact.from_json(
        artifact.to_json()
    )

    assert restored == artifact
    assert restored.artifact_id == artifact.artifact_id
    assert restored.layer_gains == (1.0, 1.0, 1.0)


def test_artifact_rejects_tampered_payload() -> None:
    raw = json.loads(_artifact().to_json())
    raw["description"] = "tampered"

    with pytest.raises(ValueError, match="artifact_id"):
        PersonalConditioningProjectorArtifact.from_json(json.dumps(raw))


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
def test_artifact_rejects_schema_drift(
    mutation: Callable[[dict[str, object]], object],
    message: str,
) -> None:
    raw = json.loads(_artifact().to_json())
    mutation(raw)

    with pytest.raises(ValueError, match=message):
        PersonalConditioningProjectorArtifact.from_json(json.dumps(raw))


def test_artifact_rejects_non_normalized_rows() -> None:
    artifact = _artifact()

    with pytest.raises(ValueError, match="L2-normalized"):
        PersonalConditioningProjectorArtifact(
            **{
                **artifact.__dict__,
                "basis_rows": tuple(
                    tuple(value * 2.0 for value in row)
                    for row in artifact.basis_rows
                ),
            }
        )


def test_loading_checks_model_width_and_hook_layers() -> None:
    torch = pytest.importorskip("torch")
    artifact = _artifact()

    basis, gains = load_projector_basis(
        torch_module=torch,
        artifact=artifact,
        expected_model_id="Qwen/test",
        expected_hidden_size=4,
        available_layer_indices=(1, 2, 3),
        device="cpu",
    )

    assert tuple(basis.shape) == (
        len(PERSONAL_CONDITIONING_VECTOR_LABELS),
        4,
    )
    assert gains == {1: 1.0, 2: 1.0, 3: 1.0}
    with pytest.raises(ValueError, match="model_id"):
        load_projector_basis(
            torch_module=torch,
            artifact=artifact,
            expected_model_id="Other/model",
            expected_hidden_size=4,
            available_layer_indices=(1, 2, 3),
            device="cpu",
        )
    with pytest.raises(ValueError, match="not hooked"):
        load_projector_basis(
            torch_module=torch,
            artifact=artifact,
            expected_model_id="Qwen/test",
            expected_hidden_size=4,
            available_layer_indices=(1, 2),
            device="cpu",
        )


def test_contrastive_builder_rejects_missing_or_zero_rows() -> None:
    rows = _rows()
    rows.pop(PERSONAL_CONDITIONING_VECTOR_LABELS[0])
    with pytest.raises(ValueError, match="missing"):
        build_contrastive_projector_artifact(
            model_id="Qwen/test",
            hidden_size=4,
            layer_indices=(1,),
            contrastive_rows=rows,
            source_fingerprint="source",
            sample_count=1,
        )

    zero_rows = _rows()
    zero_rows[PERSONAL_CONDITIONING_VECTOR_LABELS[0]] = (0.0,) * 4
    with pytest.raises(ValueError, match="zero norm"):
        build_contrastive_projector_artifact(
            model_id="Qwen/test",
            hidden_size=4,
            layer_indices=(1,),
            contrastive_rows=zero_rows,
            source_fingerprint="source",
            sample_count=1,
        )
