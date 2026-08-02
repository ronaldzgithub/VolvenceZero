from __future__ import annotations

import pytest
import torch

from volvence_zero.substrate import fit_linear_classification_probe


def test_linear_classification_probe_reports_accuracy_and_controls() -> None:
    train_features = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
    )
    eval_features = torch.tensor([[0.8, 0.2], [0.2, 0.8]])

    fit = fit_linear_classification_probe(
        torch_module=torch,
        train_features=train_features,
        train_labels=torch.tensor([0, 0, 1, 1]),
        eval_features=eval_features,
        eval_labels=torch.tensor([0, 1]),
        layer_index=3,
        class_count=2,
    )

    assert fit.layer_index == 3
    assert fit.accuracy == 1.0
    assert fit.chance_accuracy == 0.5
    assert fit.majority_accuracy == 0.5
    assert fit.support == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        (
            {
                "train_labels": torch.tensor([], dtype=torch.long),
                "eval_labels": torch.tensor([0, 1]),
            },
            "non-empty",
        ),
        (
            {
                "train_labels": torch.tensor([0, 2]),
                "eval_labels": torch.tensor([0, 1]),
            },
            "outside",
        ),
    ),
)
def test_linear_classification_probe_rejects_invalid_labels(
    kwargs: dict[str, torch.Tensor], message: str
) -> None:
    base = {
        "torch_module": torch,
        "train_features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "train_labels": torch.tensor([0, 1]),
        "eval_features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "eval_labels": torch.tensor([0, 1]),
        "layer_index": 0,
        "class_count": 2,
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=message):
        fit_linear_classification_probe(**base)
