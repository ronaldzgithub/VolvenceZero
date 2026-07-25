"""Single source of truth for the digital-ant learned evidence rollout."""

from __future__ import annotations

import os

from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.runtime import WiringLevel


ANT_RUNTIME_MODULATION_STRENGTH = 0.3
ANT_RUNTIME_EXPLORATION_STRENGTH = 1.0
# A 24-turn ecology episode settles only 23 real transitions because the
# first turn captures an action without a preceding outcome.  The segment
# must close early enough for a later turn in the same episode to consume the
# staged rollout; otherwise a no-milestone search segment is discarded by the
# cross-episode checkpoint before it ever reaches the optimizer.
ANT_RUNTIME_SEGMENT_MAX_STEPS = 16
ANT_RUNTIME_BATCH_TRANSITION_SIZE = 4
ANT_CAUSAL_ACTION_HEAD_STRENGTH = 0.35


def ant_runtime_replay_rollout_config(
    *,
    enable_sparse_exploration: bool,
    enable_segment_credit: bool = True,
) -> FinalRolloutConfig:
    """Open real replay; enable posterior exploration only for sparse tasks."""

    requested_device = os.environ.get("VZ_TENSOR_DEVICE", "cpu").strip().lower()
    use_accelerated_runtime = requested_device.startswith(("cuda", "mps"))
    return FinalRolloutConfig(
        # The ecological body has an accelerator-capable ndarray runtime path
        # (CUDA or Apple MPS).  Keep CPU as the reproducible default; the
        # training CLI sets VZ_TENSOR_DEVICE only for an explicit
        # --device cuda/mps run.
        temporal_runtime_backend=(
            WiringLevel.ACTIVE if use_accelerated_runtime else WiringLevel.DISABLED
        ),
        internal_rl_runtime_replay=WiringLevel.ACTIVE,
        internal_rl_runtime_segment_credit=(
            WiringLevel.ACTIVE
            if enable_segment_credit
            else WiringLevel.DISABLED
        ),
        internal_rl_runtime_segment_max_steps=(
            ANT_RUNTIME_SEGMENT_MAX_STEPS
        ),
        internal_rl_batch_accumulation_size=(
            ANT_RUNTIME_BATCH_TRANSITION_SIZE
        ),
        internal_rl_causal_action_head=WiringLevel.ACTIVE,
        internal_rl_causal_action_head_strength=(
            ANT_CAUSAL_ACTION_HEAD_STRENGTH
        ),
        internal_rl_runtime_modulation_strength=(
            ANT_RUNTIME_MODULATION_STRENGTH
        ),
        internal_rl_runtime_exploration_strength=(
            ANT_RUNTIME_EXPLORATION_STRENGTH
            if enable_sparse_exploration
            else 0.0
        ),
        prediction_error_temporal_switch=WiringLevel.ACTIVE,
        prediction_error_temporal_switch_strength=0.35,
        prediction_error_temporal_switch_floor=0.5,
    )


__all__ = [
    "ANT_CAUSAL_ACTION_HEAD_STRENGTH",
    "ANT_RUNTIME_EXPLORATION_STRENGTH",
    "ANT_RUNTIME_BATCH_TRANSITION_SIZE",
    "ANT_RUNTIME_MODULATION_STRENGTH",
    "ANT_RUNTIME_SEGMENT_MAX_STEPS",
    "ant_runtime_replay_rollout_config",
]
