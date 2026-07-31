from __future__ import annotations

import pytest
import torch

from volvence_zero.substrate import (
    CharacterResidualAdapterPackage,
    SubstrateDeltaAdapterLayer,
    load_character_residual_deltas,
)


def _package() -> CharacterResidualAdapterPackage:
    return CharacterResidualAdapterPackage.create(
        character_id="zhang-wuji",
        character_name="张无忌",
        model_id="Qwen/test-1.5b",
        source_live_through_model_id="Qwen/test-0.5b",
        source_template_id="zhang-wuji-live-through",
        source_template_integrity_hash="a" * 64,
        source_live_through_proof="proof-sha256",
        hidden_size=3,
        adapter_layers=(
            SubstrateDeltaAdapterLayer(
                layer_index=1,
                delta_vector=(0.01, -0.02, 0.03),
                mean_abs_delta=0.02,
                description="test layer",
            ),
        ),
        training_loss=1.2,
        sample_count=4,
        description="test character residual package",
    )


def test_character_residual_package_round_trips_and_materializes() -> None:
    package = CharacterResidualAdapterPackage.from_json(_package().to_json())

    deltas = load_character_residual_deltas(
        torch_module=torch,
        package=package,
        expected_model_id="Qwen/test-1.5b",
        expected_hidden_size=3,
        available_layer_indices=(0, 1, 2),
        device=torch.device("cpu"),
    )

    assert package.package_id == _package().package_id
    assert tuple(deltas) == (1,)
    assert torch.equal(deltas[1], torch.tensor((0.01, -0.02, 0.03)))


def test_character_residual_package_rejects_wrong_target_model() -> None:
    with pytest.raises(ValueError, match="does not match runtime"):
        load_character_residual_deltas(
            torch_module=torch,
            package=_package(),
            expected_model_id="Qwen/test-0.5b",
            expected_hidden_size=3,
            available_layer_indices=(0, 1, 2),
            device=torch.device("cpu"),
        )
