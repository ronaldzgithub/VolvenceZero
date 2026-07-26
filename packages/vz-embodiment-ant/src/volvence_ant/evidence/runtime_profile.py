"""Single source of truth for the digital-ant learned evidence rollout."""

from __future__ import annotations

import os

from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.runtime import WiringLevel

from volvence_ant.substrate import AntSenseSchema, sense_mirror_transform


ANT_RUNTIME_MODULATION_STRENGTH = 0.3
ANT_RUNTIME_EXPLORATION_STRENGTH = 1.0
# A 24-turn ecology episode settles only 23 real transitions because the
# first turn captures an action without a preceding outcome.  The segment
# must close early enough for a later turn in the same episode to consume the
# staged rollout; otherwise a no-milestone search segment is discarded by the
# cross-episode checkpoint before it ever reaches the optimizer.
ANT_RUNTIME_SEGMENT_MAX_STEPS = 16
ANT_RUNTIME_BATCH_TRANSITION_SIZE = 4
ANT_CAUSAL_ACTION_HEAD_STRENGTH = 1.0
ANT_CAUSAL_ACTION_HEAD_RANK = 16
# Frozen motor_decode consumes steering in z[0:2] and speed in z[2].
ANT_CAUSAL_ACTION_HEAD_EFFECTIVE_DIMS = (0, 1, 2)
# Steering consumes only the opponent-coded z[1] - z[0] axis. Its orthogonal
# common mode is actuator-null and must not absorb policy credit.
ANT_CAUSAL_ACTION_HEAD_CONTRAST_PAIRS = ((0, 1),)
# Ownership transfer measured in v22/v22r: with the base policy free to write
# the contrast axis, credit competition always favored its degenerate
# non-directional solution (baseline turn amplified 0.083 -> ~0.147 rad under
# both fixed and randomized forced-approach geometry) while the head's
# food->turn authority stayed pinned at ~1e-3 and never grew. Exclusive
# steering removes the deterministic base contrast so the state-conditioned
# head is the only learned steering writer; the base keeps the speed/common
# mode and exploration noise still proposes turns.
ANT_CAUSAL_ACTION_HEAD_EXCLUSIVE_STEERING = True
# Frozen embodiment reflection: left/right receptors swap, oriented
# pseudoscalars (gradient, egocentric sine, prior turn) change sign. The
# temporal owner consumes this transform without learning or reconstructing
# ant sensor semantics.
(
    ANT_CAUSAL_ACTION_HEAD_INPUT_MIRROR_PERMUTATION,
    ANT_CAUSAL_ACTION_HEAD_INPUT_MIRROR_SIGNS,
) = sense_mirror_transform(AntSenseSchema.ECOLOGY_V2)


def ant_runtime_replay_rollout_config(
    *,
    enable_sparse_exploration: bool,
    enable_segment_credit: bool = True,
    enable_prediction_error_switch: bool = True,
    sense_schema: AntSenseSchema | None = None,
) -> FinalRolloutConfig:
    """Open real replay; enable posterior exploration only for sparse tasks.

    ``enable_prediction_error_switch=False`` is the PE-off matched-control
    lever: the temporal switch stops consuming external prediction-error
    pressure while every other owner, budget and layout stays matched. True is
    the exact rollback path and the only setting a formal learned arm may use.
    """

    requested_device = os.environ.get("VZ_TENSOR_DEVICE", "cpu").strip().lower()
    use_accelerated_runtime = requested_device.startswith(("cuda", "mps"))
    input_mirror = (
        sense_mirror_transform(sense_schema)
        if sense_schema is not None
        else (None, None)
    )
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
        internal_rl_causal_action_head_rank=(
            ANT_CAUSAL_ACTION_HEAD_RANK
        ),
        internal_rl_causal_action_head_effective_dims=(
            ANT_CAUSAL_ACTION_HEAD_EFFECTIVE_DIMS
        ),
        internal_rl_causal_action_head_contrast_pairs=(
            ANT_CAUSAL_ACTION_HEAD_CONTRAST_PAIRS
        ),
        internal_rl_causal_action_head_exclusive_steering=(
            ANT_CAUSAL_ACTION_HEAD_EXCLUSIVE_STEERING
        ),
        internal_rl_causal_action_head_input_mirror_permutation=(
            input_mirror[0]
        ),
        internal_rl_causal_action_head_input_mirror_signs=(
            input_mirror[1]
        ),
        internal_rl_runtime_modulation_strength=(
            ANT_RUNTIME_MODULATION_STRENGTH
        ),
        internal_rl_runtime_exploration_strength=(
            ANT_RUNTIME_EXPLORATION_STRENGTH
            if enable_sparse_exploration
            else 0.0
        ),
        prediction_error_temporal_switch=(
            WiringLevel.ACTIVE
            if enable_prediction_error_switch
            else WiringLevel.DISABLED
        ),
        prediction_error_temporal_switch_strength=0.35,
        prediction_error_temporal_switch_floor=0.5,
    )


__all__ = [
    "ANT_CAUSAL_ACTION_HEAD_STRENGTH",
    "ANT_CAUSAL_ACTION_HEAD_CONTRAST_PAIRS",
    "ANT_CAUSAL_ACTION_HEAD_EXCLUSIVE_STEERING",
    "ANT_CAUSAL_ACTION_HEAD_INPUT_MIRROR_PERMUTATION",
    "ANT_CAUSAL_ACTION_HEAD_INPUT_MIRROR_SIGNS",
    "ANT_CAUSAL_ACTION_HEAD_RANK",
    "ANT_CAUSAL_ACTION_HEAD_EFFECTIVE_DIMS",
    "ANT_RUNTIME_EXPLORATION_STRENGTH",
    "ANT_RUNTIME_BATCH_TRANSITION_SIZE",
    "ANT_RUNTIME_MODULATION_STRENGTH",
    "ANT_RUNTIME_SEGMENT_MAX_STEPS",
    "ant_runtime_replay_rollout_config",
]
