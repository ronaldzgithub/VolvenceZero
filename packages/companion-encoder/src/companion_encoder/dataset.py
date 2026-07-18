# Copyright 2026 Companion Standard Contributors
# Licensed under the Apache License, Version 2.0.

"""Trajectory-JSON dataset loading (torch-free).

Reads canonical trajectory documents from the ``companion-trajgen``
output layout::

    <data_dir>/train/*.trajectory.json
    <data_dir>/val/*.trajectory.json

and flattens them into per-anchor supervised examples. This module never
imports the bench or the generator — trajectory JSON on disk is the only
data interface (the standard's exchange contract), so any conformant
producer can feed the encoder.

Split hygiene is re-checked at load time: a scenario family appearing in
both splits is a hard error, not a warning. The generator already splits
by whole family; this re-check catches hand-assembled or merged datasets.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

from companion_standard import (
    InteractionTrajectory,
    RelationshipPhase,
    trajectory_from_jsonable,
)

from companion_encoder.serialization import render_label_prefix

# Fixed class order for the phase head. Enum definition order is the
# canonical order (companion_standard.trajectory.RelationshipPhase).
PHASE_VOCAB: tuple[RelationshipPhase, ...] = tuple(RelationshipPhase)
PHASE_TO_INDEX: dict[RelationshipPhase, int] = {
    phase: index for index, phase in enumerate(PHASE_VOCAB)
}

REGRESSION_TARGETS: tuple[str, ...] = (
    "trust_level",
    "continuity_level",
    "repair_pressure",
)


@dataclass(frozen=True)
class AnchorExample:
    """One supervised example: trajectory prefix -> relationship state."""

    trajectory_id: str
    family: str
    scenario_ref: str
    split: str
    anchor: tuple[int, int]
    text: str
    phase: RelationshipPhase
    trust_level: float
    continuity_level: float
    repair_pressure: float

    @property
    def phase_index(self) -> int:
        return PHASE_TO_INDEX[self.phase]

    @property
    def regression_targets(self) -> tuple[float, float, float]:
        return (self.trust_level, self.continuity_level, self.repair_pressure)


def load_trajectories(split_dir: pathlib.Path | str) -> tuple[InteractionTrajectory, ...]:
    """Load and re-validate every ``*.trajectory.json`` under one split dir."""
    directory = pathlib.Path(split_dir)
    paths = sorted(directory.glob("*.trajectory.json"))
    if not paths:
        raise FileNotFoundError(
            f"no *.trajectory.json files under {directory} — expected the "
            f"companion-trajgen output layout (<data_dir>/train, <data_dir>/val)"
        )
    return tuple(
        trajectory_from_jsonable(json.loads(path.read_text(encoding="utf-8")))
        for path in paths
    )


def examples_from_trajectory(
    trajectory: InteractionTrajectory, *, split: str
) -> tuple[AnchorExample, ...]:
    return tuple(
        AnchorExample(
            trajectory_id=trajectory.trajectory_id,
            family=trajectory.family,
            scenario_ref=trajectory.scenario_ref,
            split=split,
            anchor=(label.session_index, label.turn_index),
            text=render_label_prefix(trajectory, label),
            phase=label.phase,
            trust_level=label.trust_level,
            continuity_level=label.continuity_level,
            repair_pressure=label.repair_pressure,
        )
        for label in trajectory.labels
    )


@dataclass(frozen=True)
class DatasetSplits:
    train: tuple[AnchorExample, ...]
    val: tuple[AnchorExample, ...]
    train_trajectories: tuple[InteractionTrajectory, ...]
    val_trajectories: tuple[InteractionTrajectory, ...]


def load_dataset(data_dir: pathlib.Path | str) -> DatasetSplits:
    """Load both splits and enforce whole-family split hygiene."""
    root = pathlib.Path(data_dir)
    train_trajectories = load_trajectories(root / "train")
    val_trajectories = load_trajectories(root / "val")

    train_families = {t.family for t in train_trajectories}
    val_families = {t.family for t in val_trajectories}
    overlap = train_families & val_families
    if overlap:
        raise ValueError(
            f"invalid_dataset: families {sorted(overlap)} appear in both "
            f"train and val — splits must be by whole scenario family"
        )

    train = tuple(
        example
        for trajectory in train_trajectories
        for example in examples_from_trajectory(trajectory, split="train")
    )
    val = tuple(
        example
        for trajectory in val_trajectories
        for example in examples_from_trajectory(trajectory, split="val")
    )
    return DatasetSplits(
        train=train,
        val=val,
        train_trajectories=train_trajectories,
        val_trajectories=val_trajectories,
    )


__all__ = [
    "PHASE_TO_INDEX",
    "PHASE_VOCAB",
    "REGRESSION_TARGETS",
    "AnchorExample",
    "DatasetSplits",
    "examples_from_trajectory",
    "load_dataset",
    "load_trajectories",
]
