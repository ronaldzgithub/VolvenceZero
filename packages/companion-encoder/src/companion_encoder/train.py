# Copyright 2026 Companion Standard Contributors
# Licensed under the Apache License, Version 2.0.

"""Supervised training loop (requires the ``[train]`` extra).

Loss = cross-entropy(phase) + mse_weight * MSE(trust/continuity/repair).
The embedding head trains implicitly through the shared backbone in this
scaffold; a contrastive objective over (family, phase) positives is an
M2 decision, not wired here.

Determinism: seeding covers python/torch RNGs and batch shuffling, so a
tiny-backbone CPU run is reproducible bit-for-bit given the same data
directory (which is itself hash-manifested by companion-trajgen).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import random

import torch
from torch import nn

from companion_encoder.dataset import AnchorExample, load_dataset
from companion_encoder.model import (
    EncoderConfig,
    RelationshipEncoder,
    save_checkpoint,
)


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    epochs: int = 4
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    mse_weight: float = 1.0
    seed: int = 0
    device: str = "cpu"

    def to_jsonable(self) -> dict:
        return dataclasses.asdict(self)


def _batches(
    examples: tuple[AnchorExample, ...], *, batch_size: int, rng: random.Random
) -> list[list[AnchorExample]]:
    shuffled = list(examples)
    rng.shuffle(shuffled)
    return [
        shuffled[start : start + batch_size]
        for start in range(0, len(shuffled), batch_size)
    ]


def _step_loss(
    model: RelationshipEncoder,
    batch: list[AnchorExample],
    *,
    mse_weight: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = model([example.text for example in batch])
    phase_targets = torch.tensor(
        [example.phase_index for example in batch], dtype=torch.long, device=device
    )
    regression_targets = torch.tensor(
        [example.regression_targets for example in batch],
        dtype=torch.float32,
        device=device,
    )
    phase_loss = nn.functional.cross_entropy(outputs["phase_logits"], phase_targets)
    regression_loss = nn.functional.mse_loss(outputs["regressions"], regression_targets)
    return phase_loss + mse_weight * regression_loss, phase_loss, regression_loss


@torch.no_grad()
def _eval_loss(
    model: RelationshipEncoder,
    examples: tuple[AnchorExample, ...],
    *,
    batch_size: int,
    mse_weight: float,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for start in range(0, len(examples), batch_size):
        batch = list(examples[start : start + batch_size])
        loss, _, _ = _step_loss(model, batch, mse_weight=mse_weight, device=device)
        total += float(loss) * len(batch)
        count += len(batch)
    model.train()
    return total / max(count, 1)


@dataclasses.dataclass(frozen=True)
class TrainResult:
    checkpoint_path: pathlib.Path
    history: tuple[dict, ...]
    train_example_count: int
    val_example_count: int


def train(
    *,
    data_dir: pathlib.Path | str,
    out_dir: pathlib.Path | str,
    encoder_config: EncoderConfig | None = None,
    train_config: TrainConfig | None = None,
) -> TrainResult:
    """Train on ``<data_dir>/train``, track loss on ``<data_dir>/val``,
    and write ``encoder.pt`` + ``train-report.json`` under ``out_dir``."""

    encoder_config = encoder_config or EncoderConfig()
    train_config = train_config or TrainConfig()

    torch.manual_seed(train_config.seed)
    rng = random.Random(train_config.seed)
    device = torch.device(train_config.device)

    splits = load_dataset(data_dir)
    model = RelationshipEncoder(encoder_config).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    history: list[dict] = []
    for epoch in range(train_config.epochs):
        epoch_loss = 0.0
        seen = 0
        for batch in _batches(
            splits.train, batch_size=train_config.batch_size, rng=rng
        ):
            optimizer.zero_grad()
            loss, phase_loss, regression_loss = _step_loss(
                model, batch, mse_weight=train_config.mse_weight, device=device
            )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(batch)
            seen += len(batch)
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / max(seen, 1),
                "val_loss": _eval_loss(
                    model,
                    splits.val,
                    batch_size=train_config.batch_size,
                    mse_weight=train_config.mse_weight,
                    device=device,
                ),
            }
        )

    out_directory = pathlib.Path(out_dir)
    checkpoint_path = save_checkpoint(model, out_directory / "encoder.pt")
    report = {
        "encoder_config": encoder_config.to_jsonable(),
        "train_config": train_config.to_jsonable(),
        "history": history,
        "train_example_count": len(splits.train),
        "val_example_count": len(splits.val),
    }
    (out_directory / "train-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return TrainResult(
        checkpoint_path=checkpoint_path,
        history=tuple(history),
        train_example_count=len(splits.train),
        val_example_count=len(splits.val),
    )


__all__ = ["TrainConfig", "TrainResult", "train"]
