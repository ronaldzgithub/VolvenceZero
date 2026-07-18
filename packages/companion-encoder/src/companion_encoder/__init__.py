# Copyright 2026 Companion Standard Contributors
# Licensed under the Apache License, Version 2.0.

"""companion-encoder: open-weights relationship encoder scaffold.

Top-level namespace stays torch-free: dataset loading, serialization and
baselines import eagerly; the model / training / evaluation modules
(``companion_encoder.model`` / ``.train`` / ``.evaluate`` / ``.backend``)
require the ``[train]`` extra and are imported explicitly by consumers.
"""

from companion_encoder.baselines import (
    MajorityBaseline,
    OpenAICompatibleZeroShotLabeler,
    StatePrediction,
)
from companion_encoder.dataset import (
    PHASE_TO_INDEX,
    PHASE_VOCAB,
    REGRESSION_TARGETS,
    AnchorExample,
    DatasetSplits,
    examples_from_trajectory,
    load_dataset,
    load_trajectories,
)
from companion_encoder.serialization import (
    render_full,
    render_label_prefix,
    render_prefix,
)

__all__ = [
    "PHASE_TO_INDEX",
    "PHASE_VOCAB",
    "REGRESSION_TARGETS",
    "AnchorExample",
    "DatasetSplits",
    "MajorityBaseline",
    "OpenAICompatibleZeroShotLabeler",
    "StatePrediction",
    "examples_from_trajectory",
    "load_dataset",
    "load_trajectories",
    "render_full",
    "render_label_prefix",
    "render_prefix",
]
