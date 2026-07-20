"""Frozen wiring guards for the digital-ant learned evidence profile."""

from __future__ import annotations

from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.runtime import WiringLevel

from volvence_ant.evidence import (
    ANT_RUNTIME_EXPLORATION_STRENGTH,
    ANT_RUNTIME_MODULATION_STRENGTH,
    ant_runtime_replay_rollout_config,
)
from volvence_zero.agent.learned_active_gate import LearnedBackendComponent


def test_production_rollout_defaults_remain_disabled_and_zero() -> None:
    config = FinalRolloutConfig()
    assert config.internal_rl_runtime_replay is WiringLevel.DISABLED
    assert config.internal_rl_runtime_modulation_strength == 0.0
    assert config.internal_rl_runtime_exploration_strength == 0.0


def test_ant_evidence_profile_opens_real_replay_without_changing_defaults() -> None:
    config = ant_runtime_replay_rollout_config()
    assert config.internal_rl_runtime_replay is WiringLevel.ACTIVE
    assert (
        config.internal_rl_runtime_modulation_strength
        == ANT_RUNTIME_MODULATION_STRENGTH
        == 0.3
    )
    assert (
        config.internal_rl_runtime_exploration_strength
        == ANT_RUNTIME_EXPLORATION_STRENGTH
        == 0.35
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
