"""Frozen wiring guards for the digital-ant learned evidence profile."""

from __future__ import annotations

from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.memory import Track
from volvence_zero.runtime import WiringLevel
from volvence_zero.agent.learned_active_gate import LearnedBackendComponent

from volvence_ant.evidence import (
    ANT_CAUSAL_ACTION_HEAD_CONTRAST_PAIRS,
    ANT_CAUSAL_ACTION_HEAD_EXCLUSIVE_STEERING,
    ANT_CAUSAL_ACTION_HEAD_FORMATION_CONFLICT_SCALE,
    ANT_CAUSAL_ACTION_HEAD_FORMATION_MAX_UPDATE_STEPS,
    ANT_CAUSAL_ACTION_HEAD_EFFECTIVE_DIMS,
    ANT_CAUSAL_ACTION_HEAD_INPUT_MIRROR_PERMUTATION,
    ANT_CAUSAL_ACTION_HEAD_INPUT_MIRROR_SIGNS,
    ANT_CAUSAL_ACTION_HEAD_RANK,
    ANT_CAUSAL_ACTION_HEAD_STRENGTH,
    ANT_RUNTIME_BATCH_TRANSITION_SIZE,
    ANT_RUNTIME_EXPLORATION_STRENGTH,
    ANT_RUNTIME_MODULATION_STRENGTH,
    ANT_RUNTIME_SEGMENT_MAX_STEPS,
    ANT_TEMPORAL_POST_SWITCH_MIN_DWELL_ACTIONS,
    ant_runtime_replay_rollout_config,
)
from volvence_ant.env import AntWorld, AntWorldConfig
from volvence_ant.runtime import AntSession, AntSessionConfig
from volvence_ant.substrate import (
    SENSE_CHANNELS_ECOLOGY_V2,
    AntSenseSchema,
    sense_mirror_transform,
)


def test_production_rollout_defaults_remain_disabled_and_zero() -> None:
    config = FinalRolloutConfig()
    assert config.internal_rl_runtime_replay is WiringLevel.DISABLED
    assert config.internal_rl_batch_accumulation_size == 1
    assert config.internal_rl_runtime_modulation_strength == 0.0
    assert config.internal_rl_runtime_exploration_strength == 0.0
    assert config.internal_rl_causal_action_head_effective_dims is None
    assert config.internal_rl_causal_action_head_contrast_pairs is None
    assert (
        config.internal_rl_causal_action_head_input_mirror_permutation
        is None
    )
    assert config.internal_rl_causal_action_head_input_mirror_signs is None
    assert (
        config.internal_rl_causal_action_head_formation_protection
        is WiringLevel.DISABLED
    )
    assert (
        config.internal_rl_causal_action_head_formation_max_update_steps == 0
    )
    assert (
        config.internal_rl_causal_action_head_formation_conflict_scale == 1.0
    )
    assert (
        config.temporal_post_switch_min_dwell
        is WiringLevel.DISABLED
    )
    assert config.temporal_post_switch_min_dwell_actions == 0


def test_ant_evidence_profile_opens_real_replay_without_changing_defaults() -> None:
    config = ant_runtime_replay_rollout_config(
        enable_sparse_exploration=True
    )
    assert config.internal_rl_runtime_replay is WiringLevel.ACTIVE
    assert (
        config.internal_rl_batch_accumulation_size
        == ANT_RUNTIME_BATCH_TRANSITION_SIZE
        == 2
    )
    assert (
        config.internal_rl_runtime_modulation_strength
        == ANT_RUNTIME_MODULATION_STRENGTH
        == 0.3
    )
    assert (
        config.internal_rl_runtime_exploration_strength
        == ANT_RUNTIME_EXPLORATION_STRENGTH
        == 1.0
    )
    assert (
        config.internal_rl_runtime_segment_max_steps
        == ANT_RUNTIME_SEGMENT_MAX_STEPS
        == 7
    )
    assert (
        config.temporal_post_switch_min_dwell
        is WiringLevel.ACTIVE
    )
    assert (
        config.temporal_post_switch_min_dwell_actions
        == ANT_TEMPORAL_POST_SWITCH_MIN_DWELL_ACTIONS
        == 4
    )
    assert (
        config.internal_rl_causal_action_head_rank
        == ANT_CAUSAL_ACTION_HEAD_RANK
        == 16
    )
    assert (
        config.internal_rl_causal_action_head_effective_dims
        == ANT_CAUSAL_ACTION_HEAD_EFFECTIVE_DIMS
        == (0, 1, 2)
    )
    assert (
        config.internal_rl_causal_action_head_contrast_pairs
        == ANT_CAUSAL_ACTION_HEAD_CONTRAST_PAIRS
        == ((0, 1),)
    )
    assert (
        config.internal_rl_causal_action_head_strength
        == ANT_CAUSAL_ACTION_HEAD_STRENGTH
        == 1.0
    )
    assert (
        config.internal_rl_causal_action_head_exclusive_steering
        is ANT_CAUSAL_ACTION_HEAD_EXCLUSIVE_STEERING
        is True
    )
    assert (
        config.internal_rl_causal_action_head_formation_protection
        is WiringLevel.ACTIVE
    )
    assert (
        config.internal_rl_causal_action_head_formation_max_update_steps
        == ANT_CAUSAL_ACTION_HEAD_FORMATION_MAX_UPDATE_STEPS
        == 160
    )
    assert (
        config.internal_rl_causal_action_head_formation_conflict_scale
        == ANT_CAUSAL_ACTION_HEAD_FORMATION_CONFLICT_SCALE
        == 0.25
    )
    # The shortest formal P0 episode has 10 usable settled transitions after
    # capture/bootstrap.  The segment must close early enough for a later turn
    # to consume its >=4-transition optimizer batch before episode transfer.
    assert config.internal_rl_runtime_segment_max_steps == 7
    dense_config = ant_runtime_replay_rollout_config(
        enable_sparse_exploration=False
    )
    assert dense_config.internal_rl_runtime_replay is WiringLevel.ACTIVE
    assert dense_config.internal_rl_runtime_exploration_strength == 0.0
    ecology_config = ant_runtime_replay_rollout_config(
        enable_sparse_exploration=False,
        sense_schema=AntSenseSchema.ECOLOGY_V2,
    )
    assert (
        ecology_config
        .internal_rl_causal_action_head_input_mirror_permutation
        == ANT_CAUSAL_ACTION_HEAD_INPUT_MIRROR_PERMUTATION
    )
    assert (
        ecology_config.internal_rl_causal_action_head_input_mirror_signs
        == ANT_CAUSAL_ACTION_HEAD_INPUT_MIRROR_SIGNS
    )
    assert (
        ecology_config.environment_milestone_temporal_switch
        is WiringLevel.ACTIVE
    )
    milestone_control = ant_runtime_replay_rollout_config(
        enable_sparse_exploration=False,
        enable_environment_milestone_switch=False,
        sense_schema=AntSenseSchema.ECOLOGY_V2,
    )
    assert (
        milestone_control.environment_milestone_temporal_switch
        is WiringLevel.DISABLED
    )
    assert (
        {
            key
            for key, value in ecology_config.__dict__.items()
            if value != milestone_control.__dict__[key]
        }
        == {"environment_milestone_temporal_switch"}
    )

    session = AntSession(
        AntWorld(config=AntWorldConfig(seed=3)),
        config=AntSessionConfig(
            temporal_latent_dim=16,
            rollout_config=ecology_config,
            sense_schema=AntSenseSchema.ECOLOGY_V2,
        ),
    )
    assert (
        session.runner.world_temporal_policy.parameter_store
        .causal_action_head_parameters(track=Track.WORLD)
        .rank
        == 16
    )
    assert (
        session.runner.world_temporal_policy
        .causal_action_head_effective_dims
        == (0, 1, 2)
    )
    assert (
        session.runner.world_temporal_policy
        .causal_action_head_contrast_pairs
        == ((0, 1),)
    )
    assert (
        session.runner.world_temporal_policy
        .causal_action_head_formation_protection
        is WiringLevel.ACTIVE
    )
    assert (
        session.runner.self_temporal_policy
        .causal_action_head_formation_protection
        is WiringLevel.ACTIVE
    )
    assert (
        session.runner.world_temporal_policy
        .post_switch_min_dwell_wiring
        is WiringLevel.ACTIVE
    )
    assert (
        session.runner.world_temporal_policy
        .post_switch_min_dwell_actions
        == 4
    )
    assert (
        session.runner.world_temporal_policy
        .causal_action_head_exclusive_steering
        is True
    )
    assert (
        session.runner.world_temporal_policy
        .causal_action_head_mirror_equivariance
        is True
    )
    assert (
        session.runner.joint_loop._world_sandbox.causal_policy
        .causal_action_head_effective_dims
        == (0, 1, 2)
    )
    assert (
        session.runner.joint_loop._world_sandbox.causal_policy
        .causal_action_head_contrast_pairs
        == ((0, 1),)
    )
    assert (
        session.runner.joint_loop._world_sandbox.causal_policy
        .causal_action_head_mirror_equivariance
        is True
    )


def test_ecology_sense_mirror_transform_is_complete_and_involutive() -> None:
    permutation, signs = sense_mirror_transform(
        AntSenseSchema.ECOLOGY_V2
    )
    channels = SENSE_CHANNELS_ECOLOGY_V2

    assert len(permutation) == len(signs) == len(channels) == 19
    assert channels[permutation[channels.index("food_left")]] == "food_right"
    assert channels[permutation[channels.index("heat_left")]] == "heat_right"
    assert signs[channels.index("food_diff")] == -1
    assert signs[channels.index("home_ego_sin")] == -1
    assert signs[channels.index("last_turn_command")] == -1
    for index, source in enumerate(permutation):
        assert permutation[source] == index
        assert signs[index] * signs[source] == 1


def test_ant_exploration_context_uses_seed_not_session_label() -> None:
    def context_digest(*, seed: int, session_id: str) -> str:
        session = AntSession(
            AntWorld(config=AntWorldConfig(seed=3)),
            config=AntSessionConfig(
                temporal_latent_dim=4,
                session_id=session_id,
                seed=seed,
                rollout_config=ant_runtime_replay_rollout_config(
                    enable_sparse_exploration=True
                ),
            ),
        )
        return (
            session.runner.world_temporal_policy
            .runtime_exploration_context_digest
        )

    assert context_digest(seed=17, session_id="learned") == context_digest(
        seed=17,
        session_id="no-optimize",
    )
    assert context_digest(seed=17, session_id="learned") != context_digest(
        seed=18,
        session_id="learned",
    )


def test_formal_active_arms_only_isolate_declared_causal_factor() -> None:
    from scripts.run_ant_active_evidence import _arm_configs

    arms = _arm_configs(
        seed=2,
        n_z=4,
        component=LearnedBackendComponent.TEMPORAL_RUNTIME,
    )
    learned = arms["learned"]
    no_optimize = arms["no_optimize"]
    pe_off = arms["pe_off"]

    assert learned.rollout_config == no_optimize.rollout_config
    assert learned.rollout_config == pe_off.rollout_config
    assert learned.joint_apply_writeback is True
    assert no_optimize.joint_apply_writeback is True
    assert learned.joint_apply_policy_optimization is True
    assert no_optimize.joint_apply_policy_optimization is False
    assert pe_off.external_prediction_error_drive is False


async def test_eta_off_arm_keeps_configured_latent_dimension_in_active_replay() -> None:
    from scripts.run_ant_matched_control import _schedule_gated_arms
    from volvence_ant.env import AntWorld, AntWorldConfig
    from volvence_ant.runtime import AntSession

    session = AntSession(
        AntWorld(config=AntWorldConfig(seed=0)),
        config=_schedule_gated_arms(seed=0, n_z=16)["eta_off"],
    )
    records = await session.run(4)

    assert all(len(record.code) == 16 for record in records)
    assert (
        dict(records[-1].backend_wiring)["internal_rl_runtime_replay"]
        == "active"
    )
