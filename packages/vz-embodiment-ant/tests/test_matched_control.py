"""Workstream E tests: behavioural matched-control (fast) + latent proof gate."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import importlib.util
import multiprocessing

import pytest

from volvence_ant.controllers.e2e_rl_ant import E2ERLAnt, PPOConfig
from volvence_ant.proofs import (
    aggregate_matched_control_reports,
    run_behavioral_matched_control,
    run_multiseed_matched_control,
)
from volvence_ant.runtime import AntSessionConfig

_HAS_TORCH = importlib.util.find_spec("torch") is not None


async def test_behavioral_matched_control_arms_present() -> None:
    report = await run_behavioral_matched_control(ticks=20, seed=0)
    arm_names = {arm.arm for arm in report.arms}
    assert {"learned", "pe_off", "fixed_rule", "random"} <= arm_names
    for arm in report.arms:
        assert arm.ticks == 20
        assert arm.mean_food_experienced >= 0.0
        assert arm.minimum_food_distance >= 0.0
        assert arm.diagnostic_breakpoint != "not-evaluated"
    learned = next(arm for arm in report.arms if arm.arm == "learned")
    assert learned.initial_policy_fingerprint is not None
    assert learned.trained_policy_fingerprint is not None


async def test_learned_arm_beats_random_floor() -> None:
    report = await run_behavioral_matched_control(ticks=40, seed=0)
    by_arm = {arm.arm: arm for arm in report.arms}
    # a controller that senses food should not do worse than the random floor
    assert by_arm["learned"].mean_food_experienced >= by_arm["random"].mean_food_experienced


@pytest.mark.skipif(not _HAS_TORCH, reason="latent proofs require torch")
def test_latent_proofs_learning_is_real() -> None:
    from volvence_ant.proofs import run_ant_latent_proofs

    report = run_ant_latent_proofs(rl_iterations=20, eta_epochs=12)
    assert report.learning_is_real
    assert report.eta_bottleneck_holds


async def test_multiseed_matrix_reports_real_no_optimize_effect() -> None:
    from volvence_zero.joint_loop import JointLoopSchedule

    def arms(seed: int, n_z: int) -> dict[str, AntSessionConfig]:
        return {
            "no_optimize": AntSessionConfig(
                temporal_latent_dim=n_z,
                seed=seed,
                external_prediction_error_drive=True,
                joint_schedule=JointLoopSchedule(ssl_interval=1, rl_interval=3),
                joint_apply_writeback=True,
                joint_apply_policy_optimization=False,
            )
        }

    report = await run_multiseed_matched_control(
        seeds=(0, 1),
        ticks=6,
        temporal_latent_dim=4,
        kernel_arm_factory=arms,
        include_e2e_rl=False,
    )
    assert report.learned_minus_no_optimize is not None
    assert {aggregate.arm for aggregate in report.aggregates} >= {
        "learned",
        "no_optimize",
        "pe_off",
    }


def test_spawn_seed_workers_match_serial_and_preserve_seed_order() -> None:
    from scripts.run_ant_matched_control import _run_seed_worker

    seeds = (1, 0)
    serial = tuple(
        _run_seed_worker(seed, 2, 0, 16, False)
        for seed in seeds
    )
    with ProcessPoolExecutor(
        max_workers=2,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = [
            executor.submit(_run_seed_worker, seed, 2, 0, 16, False)
            for seed in seeds
        ]
        parallel = tuple(future.result() for future in reversed(futures))

    serial_aggregate = aggregate_matched_control_reports(
        serial,
        seed_order=seeds,
    )
    parallel_aggregate = aggregate_matched_control_reports(
        parallel,
        seed_order=seeds,
    )
    assert parallel_aggregate == serial_aggregate
    assert tuple(report.seed for report in parallel_aggregate.reports) == seeds


def test_formal_no_optimize_is_a_real_policy_update_ablation() -> None:
    from scripts.run_ant_matched_control import (
        _ANT_RL_RUNTIME_EXPLORATION_STRENGTH,
        _ANT_RL_RUNTIME_MODULATION_STRENGTH,
        _learned_config,
        _schedule_gated_arms,
    )

    learned = _learned_config(seed=0, n_z=16)
    no_optimize = _schedule_gated_arms(seed=0, n_z=16)["no_optimize"]

    assert learned.joint_apply_policy_optimization is True
    assert no_optimize.joint_apply_policy_optimization is False
    assert learned.joint_apply_writeback is no_optimize.joint_apply_writeback
    assert (
        learned.rollout_config.internal_rl_runtime_modulation_strength
        == no_optimize.rollout_config.internal_rl_runtime_modulation_strength
        == _ANT_RL_RUNTIME_MODULATION_STRENGTH
    )
    assert (
        learned.rollout_config.internal_rl_runtime_exploration_strength
        == no_optimize.rollout_config.internal_rl_runtime_exploration_strength
        == _ANT_RL_RUNTIME_EXPLORATION_STRENGTH
    )


@pytest.mark.skipif(not _HAS_TORCH, reason="E2E PPO baseline requires torch")
def test_e2e_ppo_changes_parameters_without_using_kernel() -> None:
    from volvence_ant.env.ant_world import AntWorld, AntWorldConfig, FoodSource

    def world_factory(seed: int) -> AntWorld:
        return AntWorld(
            config=AntWorldConfig(seed=seed),
            food_sources=(FoodSource(x=3.0, y=0.0),),
        )

    ant = E2ERLAnt(seed=0, hidden_dim=8)
    before = ant.parameter_digest()
    ant.train(
        world_factory=world_factory,
        seed=0,
        config=PPOConfig(episodes=1, ticks_per_episode=8, update_epochs=1),
    )
    assert ant.parameter_digest() != before
