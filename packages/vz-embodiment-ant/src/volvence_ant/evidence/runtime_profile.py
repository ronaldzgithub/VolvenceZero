"""Single source of truth for the digital-ant learned evidence rollout."""

from __future__ import annotations

from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.runtime import WiringLevel


ANT_RUNTIME_MODULATION_STRENGTH = 0.3
ANT_RUNTIME_EXPLORATION_STRENGTH = 1.0


def ant_runtime_replay_rollout_config(
    *, enable_sparse_exploration: bool
) -> FinalRolloutConfig:
    """Open real replay; enable posterior exploration only for sparse tasks."""

    return FinalRolloutConfig(
        internal_rl_runtime_replay=WiringLevel.ACTIVE,
        internal_rl_runtime_modulation_strength=(
            ANT_RUNTIME_MODULATION_STRENGTH
        ),
        internal_rl_runtime_exploration_strength=(
            ANT_RUNTIME_EXPLORATION_STRENGTH
            if enable_sparse_exploration
            else 0.0
        ),
    )


__all__ = [
    "ANT_RUNTIME_EXPLORATION_STRENGTH",
    "ANT_RUNTIME_MODULATION_STRENGTH",
    "ant_runtime_replay_rollout_config",
]
