from __future__ import annotations

import pytest

from volvence_zero.substrate import (
    RareHeavyStructuralObjective,
)


def test_structural_objective_is_deterministic_bounded_and_content_aware() -> None:
    objective = RareHeavyStructuralObjective()

    first = objective.residual_delta(
        source_text="shared route alpha",
        layer_index=1,
        width=8,
    )
    repeated = objective.residual_delta(
        source_text="shared route alpha",
        layer_index=1,
        width=8,
    )
    changed = objective.residual_delta(
        source_text="different content omega",
        layer_index=1,
        width=8,
    )

    assert first == repeated
    assert first != changed
    assert sum(abs(value) for value in first) / len(first) == pytest.approx(
        objective.amplitude
    )
    assert all(abs(value) <= 0.18 for value in first)


def test_structural_objective_rejects_invalid_shape() -> None:
    objective = RareHeavyStructuralObjective()

    with pytest.raises(ValueError, match="non-empty source"):
        objective.residual_delta(
            source_text=" ",
            layer_index=0,
            width=8,
        )
    with pytest.raises(ValueError, match="width"):
        objective.residual_delta(
            source_text="shared route",
            layer_index=0,
            width=0,
        )
