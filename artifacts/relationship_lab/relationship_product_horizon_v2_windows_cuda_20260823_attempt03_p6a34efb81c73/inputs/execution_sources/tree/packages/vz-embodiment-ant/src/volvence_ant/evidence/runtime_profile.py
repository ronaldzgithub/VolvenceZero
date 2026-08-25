"""Single source of truth for the digital-ant learned evidence rollout."""

from __future__ import annotations

import os

from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.integration.final_wiring import RuntimeReplayRewardEligibility
from volvence_zero.runtime import WiringLevel

from volvence_ant.substrate import AntSenseSchema, sense_mirror_transform


ANT_RUNTIME_MODULATION_STRENGTH = 0.3
ANT_RUNTIME_EXPLORATION_STRENGTH = 1.0
# The shortest formal mechanism-audit episode is 12 turns and settles only 10
# usable transitions after capture/bootstrap.  The segment must close early
# enough for a later scheduled step in the SAME episode to consume the staged
# rollout; cross-episode checkpoints intentionally exclude replay.  Seven is
# therefore the longest horizon that still leaves a flush turn under the
# frozen P0 budget.  Two transitions are the smallest non-degenerate centered
# covariance batch; four deferred effective updates past the P0 budget.
ANT_RUNTIME_SEGMENT_MAX_STEPS = 7
ANT_RUNTIME_BATCH_TRANSITION_SIZE = 2
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
# Archive-side enforcement of the frozen action-head envelope. This domain is
# the first adopter: it is the only one with an ACTIVE causal action head, and
# docs/specs/digital-ant-embodiment.md freezes the very bounds the validation
# checks.
#
# This was previously declared False against a MEASURED blocker in the temporal
# owner (not this package): the store's own pristine head initializer produced
# input factors outside the envelope it would then be validated against.
# `_initial_causal_action_head_parameters` fell back to `_random_mat(rank, n_z)`
# -- an unbounded Gaussian with scale 1/sqrt(rank) -- whenever rank < n_z, and
# nothing tied that draw to `factor_absolute_limit=1.5`. At this domain's own
# n_z=16 the fixed seeds made it deterministic:
#
#   world  rank=4 update_step=0 max|input_factor|=1.0749  ok
#   self   rank=4 update_step=0 max|input_factor|=1.9823  VIOLATED
#   shared rank=4 update_step=0 max|input_factor|=1.6192  VIOLATED
#
# Closed by bounding the initializer, not by narrowing the validator. The
# envelope's absolute bound is applied UNCONDITIONALLY by every owner write
# path, so exempting `update_step == 0` heads would have made the validator's
# accepted set strictly larger than the owner's own image -- and `update_step`
# is an ordinary field of a restored snapshot, so that exemption is a bypass of
# exactly its own width. `_bounded_initial_input_factors` instead rescales the
# draw by one global positive scalar, which preserves its direction and leaves
# every already-in-band configuration byte-identical. Differential evidence over
# the full reachable (n_z <= 64, rank, track) grid: 6240 configurations, 5573
# byte-identical, 667 changed -- and the 667 changed are exactly the 667 that
# were out of band, i.e. no previously-valid configuration moved. This domain's
# own WORLD track (1.0749) is among the untouched.
ANT_CAUSAL_ACTION_HEAD_ENVELOPE_ENFORCED = True
# L1-B formation window. The failed v28 body remained divergent through owner
# update step 135; 160 covers that frozen failure surface while remaining a
# finite per-head window. Within it, only transition contributions opposing
# the batch net are scaled. No body ID, task phase or food/carrying label is
# visible to the temporal owner.
ANT_CAUSAL_ACTION_HEAD_FORMATION_MAX_UPDATE_STEPS = 160
ANT_CAUSAL_ACTION_HEAD_FORMATION_CONFLICT_SCALE = 0.25
# Generic temporal option commitment. The switch action counts as action one;
# a later typed milestone can still interrupt the dwell.
ANT_TEMPORAL_POST_SWITCH_MIN_DWELL_ACTIONS = 4
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
    enable_environment_milestone_switch: bool = True,
    sense_schema: AntSenseSchema | None = None,
) -> FinalRolloutConfig:
    """Open real replay; enable posterior exploration only for sparse tasks.

    ``enable_prediction_error_switch=False`` is the PE-off matched-control
    lever: the temporal switch stops consuming external prediction-error
    pressure while every other owner, budget and layout stays matched. True is
    the exact rollback path and the only setting a formal learned arm may use.
    The lever gates only the additive PE prior; typed milestone boundaries
    (``environment_milestone_temporal_switch``) are not a PE readout and stay
    ACTIVE in both PE arms.

    ``enable_environment_milestone_switch=False`` is reserved for the
    preregistered same-physics v31 matched control.  It changes exactly the
    typed boundary wiring while leaving the environment measurements, reward,
    optimizer and physical curriculum intact.  Production/evidence callers
    retain the ACTIVE default.
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
        # Typed reward eligibility (kernel contract). The digital ant's task
        # path forbids distance/potential shaping, so a tick with no published
        # environment measurement must contribute realized reward 0 instead of
        # the PE owner's internally synthesized action axis (a monotone
        # function of the food sensors, i.e. exactly the dense shaping the
        # ablation claims to have removed). Under this declaration the
        # environment-published payoff is also independent of
        # ``external_prediction_error_drive``, so the PE-off arm keeps it.
        internal_rl_runtime_reward_eligibility=(
            RuntimeReplayRewardEligibility.ENVIRONMENT_MEASURED_ONLY
        ),
        # Latent-code bound (kernel contract). The digital ant is the only
        # domain with an ACTIVE causal action head, and the audit finding that
        # motivated this contract is its own: on a saturated contrast axis the
        # replay lane reconstructed a mean below the frozen plant's latent
        # floor, so ``(action - mean)`` was signed against the action actually
        # taken and the head gradient pointed the wrong way. Declaring the
        # owner's [0, 1] bound makes both replay lanes reconstruct only what
        # the live forward could have emitted.
        internal_rl_runtime_latent_unit_clamp=True,
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
        internal_rl_causal_action_head_envelope_enforced=(
            ANT_CAUSAL_ACTION_HEAD_ENVELOPE_ENFORCED
        ),
        internal_rl_causal_action_head_formation_protection=(
            WiringLevel.ACTIVE
        ),
        internal_rl_causal_action_head_formation_max_update_steps=(
            ANT_CAUSAL_ACTION_HEAD_FORMATION_MAX_UPDATE_STEPS
        ),
        internal_rl_causal_action_head_formation_conflict_scale=(
            ANT_CAUSAL_ACTION_HEAD_FORMATION_CONFLICT_SCALE
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
        # Additive-only PE prior at the generic default floor. The ant no
        # longer overrides the floor: the v30 frozen-replay measurement
        # (scripts/measure_ant_pe_boundary_margin.py; verdict in
        # research/ant/06_ecology_implementation_status.md) showed routine PE
        # (p50 0.508, p99 0.701) overlaps event PE (natural pickups settle at
        # ~0.32), so a "calibrated" 0.45 floor was routine-noise territory and
        # carried no event information. Boundaries come from the typed
        # milestone channel below instead.
        prediction_error_temporal_switch_strength=0.35,
        # Typed boundary events: the ant session declares pickup/delivery as
        # ``EnvironmentMeasurement.discrete_milestone``; the temporal owner
        # closes the running action segment on the next decision. This is the
        # structural replacement for the refuted PE-magnitude boundary.
        environment_milestone_temporal_switch=(
            WiringLevel.ACTIVE
            if enable_environment_milestone_switch
            else WiringLevel.DISABLED
        ),
        temporal_post_switch_min_dwell=WiringLevel.ACTIVE,
        temporal_post_switch_min_dwell_actions=(
            ANT_TEMPORAL_POST_SWITCH_MIN_DWELL_ACTIONS
        ),
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
    "ANT_TEMPORAL_POST_SWITCH_MIN_DWELL_ACTIONS",
    "ant_runtime_replay_rollout_config",
]
