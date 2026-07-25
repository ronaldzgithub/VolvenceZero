"""Frozen wiring guards for the digital-ant learned evidence profile."""

from __future__ import annotations

from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.memory import Track
from volvence_zero.runtime import WiringLevel
from volvence_zero.agent.learned_active_gate import LearnedBackendComponent

from volvence_ant.evidence import (
    ANT_CAUSAL_ACTION_HEAD_EFFECTIVE_DIMS,
    ANT_CAUSAL_ACTION_HEAD_RANK,
    ANT_RUNTIME_BATCH_TRANSITION_SIZE,
    ANT_RUNTIME_EXPLORATION_STRENGTH,
    ANT_RUNTIME_MODULATION_STRENGTH,
    ANT_RUNTIME_SEGMENT_MAX_STEPS,
    ant_runtime_replay_rollout_config,
)
from volvence_ant.env import AntWorld, AntWorldConfig
from volvence_ant.runtime import AntSession, AntSessionConfig


def test_production_rollout_defaults_remain_disabled_and_zero() -> None:
    config = FinalRolloutConfig()
    assert config.internal_rl_runtime_replay is WiringLevel.DISABLED
    assert config.internal_rl_batch_accumulation_size == 1
    assert config.internal_rl_runtime_modulation_strength == 0.0
    assert config.internal_rl_runtime_exploration_strength == 0.0
    assert config.internal_rl_causal_action_head_effective_dims is None


def test_ant_evidence_profile_opens_real_replay_without_changing_defaults() -> None:
    config = ant_runtime_replay_rollout_config(
        enable_sparse_exploration=True
    )
    assert config.internal_rl_runtime_replay is WiringLevel.ACTIVE
    assert (
        config.internal_rl_batch_accumulation_size
        == ANT_RUNTIME_BATCH_TRANSITION_SIZE
        == 4
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
        == 16
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
    # A 24-turn ecology episode has 23 settled transitions.  The segment
    # must close before the final turn so a later scheduled step can optimize
    # it before the cross-episode checkpoint discards pending replay.
    assert config.internal_rl_runtime_segment_max_steps <= 22
    dense_config = ant_runtime_replay_rollout_config(
        enable_sparse_exploration=False
    )
    assert dense_config.internal_rl_runtime_replay is WiringLevel.ACTIVE
    assert dense_config.internal_rl_runtime_exploration_strength == 0.0

    session = AntSession(
        AntWorld(config=AntWorldConfig(seed=3)),
        config=AntSessionConfig(
            temporal_latent_dim=16,
            rollout_config=dense_config,
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
        session.runner.joint_loop._world_sandbox.causal_policy
        .causal_action_head_effective_dims
        == (0, 1, 2)
    )


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
