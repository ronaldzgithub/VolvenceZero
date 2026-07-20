"""Contract tests for the dynamic stigmergy regime-shift benchmark."""

from __future__ import annotations

from dataclasses import replace

import pytest

from volvence_ant.experiments.dynamic_colony import (
    DynamicColonyArmKind,
    DynamicColonyConfig,
    DynamicPerturbationKind,
    RuntimeReplayCoverage,
    aggregate_dynamic_colony_reports,
    _apply_perturbation,
    run_dynamic_colony_seed,
)
from volvence_ant.env import AntWorld
from volvence_ant.runtime import AntSessionConfig
from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.joint_loop import JointLoopSchedule
from volvence_zero.runtime import WiringLevel


async def test_dynamic_colony_runs_frozen_seven_arm_matrix() -> None:
    config = DynamicColonyConfig(
        n_ants=1,
        training_rounds=0,
        pre_shift_rounds=2,
        post_shift_rounds=3,
        recovery_window=1,
        temporal_latent_dim=4,
    )
    rollout_config = FinalRolloutConfig(
        internal_rl_runtime_replay=WiringLevel.ACTIVE,
        internal_rl_runtime_modulation_strength=0.3,
    )
    schedule = JointLoopSchedule(ssl_interval=0, rl_interval=1)
    report = await run_dynamic_colony_seed(
        seed=0,
        perturbation=DynamicPerturbationKind.OBSTACLE_BLOCK,
        config=config,
        learned_config=AntSessionConfig(
            temporal_latent_dim=4,
            seed=0,
            rollout_config=rollout_config,
            joint_schedule=schedule,
        ),
        no_optimize_config=AntSessionConfig(
            temporal_latent_dim=4,
            seed=0,
            rollout_config=rollout_config,
            joint_schedule=schedule,
            joint_apply_policy_optimization=False,
        ),
        pe_off_config=AntSessionConfig(
            temporal_latent_dim=4,
            seed=0,
            rollout_config=rollout_config,
            joint_schedule=schedule,
            external_prediction_error_drive=False,
        ),
    )

    assert {arm.arm for arm in report.arms} == {
        arm.value for arm in DynamicColonyArmKind
    }
    kernel_initials = {
        arm.initial_checkpoint_fingerprints
        for arm in report.arms
        if arm.controller_kind == "kernel"
    }
    assert len(kernel_initials) == 1
    assert all(
        len(arm.delivery_curve)
        == config.pre_shift_rounds + config.post_shift_rounds
        for arm in report.arms
    )
    kernel_arms = tuple(
        arm for arm in report.arms if arm.controller_kind == "kernel"
    )
    assert all(
        coverage.captured == 8
        and coverage.settled == 6
        and coverage.transitions == 6
        and coverage.lineage_matches == 6
        for arm in kernel_arms
        for coverage in arm.runtime_replay_per_ant
    )
    assert all(
        coverage.captured == 4
        and coverage.settled == 4
        and coverage.transitions == 4
        and coverage.lineage_matches == 4
        for arm in kernel_arms
        for coverage in arm.post_shift_runtime_replay_per_ant
    )

    # Freeze all promotion thresholds independently of current ant quality.
    template_by_arm = {arm.arm: arm for arm in report.arms}
    formal_config = DynamicColonyConfig(
        n_ants=8,
        training_rounds=200,
        pre_shift_rounds=50,
        post_shift_rounds=100,
        recovery_window=20,
        temporal_latent_dim=4,
    )

    def formalize(arm):
        return replace(
            arm,
            n_ants=formal_config.n_ants,
            training_rounds=formal_config.training_rounds,
            pre_shift_rounds=formal_config.pre_shift_rounds,
            post_shift_rounds=formal_config.post_shift_rounds,
        )

    synthetic_reports = []
    for seed in range(10):
        shared_fingerprints = tuple(
            f"shared:{seed}:body:{body_id}" for body_id in range(8)
        )
        full_replay_coverage = tuple(
            RuntimeReplayCoverage(
                captured=298,
                settled=296,
                transitions=296,
                lineage_matches=296,
                settlement_coverage=296 / 298,
                lineage_coverage=1.0,
                drop_reasons=(),
            )
            for _ in range(8)
        )
        post_replay_coverage = tuple(
            RuntimeReplayCoverage(
                captured=198,
                settled=198,
                transitions=198,
                lineage_matches=198,
                settlement_coverage=1.0,
                lineage_coverage=1.0,
                drop_reasons=(),
            )
            for _ in range(8)
        )
        learned = replace(
            formalize(
                template_by_arm[DynamicColonyArmKind.LEARNED_BUS.value]
            ),
            pre_shift_pickups=1,
            pre_shift_throughput_per_1k_actions=80.0,
            post_shift_throughput_per_1k_actions=110.0,
            recovery_rounds=5,
            oracle_delivery_shortfall=5.0,
            initial_checkpoint_fingerprints=shared_fingerprints,
            trained_policy_fingerprints=tuple(
                f"trained:{seed}:body:{body_id}" for body_id in range(8)
            ),
            shift_policy_fingerprints=tuple(
                f"shift:{seed}:body:{body_id}" for body_id in range(8)
            ),
            adaptation_start_policy_fingerprints=tuple(
                f"adapt-start:{seed}:body:{body_id}" for body_id in range(8)
            ),
            final_policy_fingerprints=tuple(
                f"final:{seed}:body:{body_id}" for body_id in range(8)
            ),
            policy_parameters_changed=True,
            post_shift_policy_parameters_changed=True,
            runtime_replay_captured=100,
            runtime_replay_settled=100,
            runtime_replay_transitions=100,
            runtime_replay_lineage_matches=100,
            runtime_replay_lineage_coverage=1.0,
            runtime_replay_active=True,
            runtime_replay_per_ant=full_replay_coverage,
            post_shift_runtime_replay_per_ant=post_replay_coverage,
        )
        learned_no_bus = replace(
            formalize(
                template_by_arm[DynamicColonyArmKind.LEARNED_NO_BUS.value]
            ),
            post_shift_throughput_per_1k_actions=50.0,
            initial_checkpoint_fingerprints=shared_fingerprints,
            trained_policy_fingerprints=tuple(
                f"trained-no-bus:{seed}:body:{body_id}"
                for body_id in range(8)
            ),
            shift_policy_fingerprints=tuple(
                f"shift-no-bus:{seed}:body:{body_id}"
                for body_id in range(8)
            ),
            adaptation_start_policy_fingerprints=tuple(
                f"adapt-start-no-bus:{seed}:body:{body_id}"
                for body_id in range(8)
            ),
            final_policy_fingerprints=tuple(
                f"final-no-bus:{seed}:body:{body_id}"
                for body_id in range(8)
            ),
            runtime_replay_captured=100,
            runtime_replay_settled=100,
            runtime_replay_transitions=100,
            runtime_replay_lineage_matches=100,
            runtime_replay_lineage_coverage=1.0,
            runtime_replay_active=True,
            runtime_replay_per_ant=full_replay_coverage,
            post_shift_runtime_replay_per_ant=post_replay_coverage,
        )
        no_optimize = replace(
            formalize(
                template_by_arm[DynamicColonyArmKind.NO_OPTIMIZE_BUS.value]
            ),
            pre_shift_throughput_per_1k_actions=80.0,
            post_shift_throughput_per_1k_actions=60.0,
            initial_checkpoint_fingerprints=shared_fingerprints,
            trained_policy_fingerprints=tuple(
                f"stable-trained:{seed}:body:{body_id}"
                for body_id in range(8)
            ),
            shift_policy_fingerprints=tuple(
                f"stable:{seed}:body:{body_id}" for body_id in range(8)
            ),
            adaptation_start_policy_fingerprints=tuple(
                f"stable:{seed}:body:{body_id}" for body_id in range(8)
            ),
            final_policy_fingerprints=tuple(
                f"stable:{seed}:body:{body_id}" for body_id in range(8)
            ),
            policy_parameters_changed=False,
            post_shift_policy_parameters_changed=False,
            runtime_replay_captured=100,
            runtime_replay_settled=100,
            runtime_replay_transitions=100,
            runtime_replay_lineage_matches=100,
            runtime_replay_lineage_coverage=1.0,
            runtime_replay_active=True,
            runtime_replay_per_ant=full_replay_coverage,
            post_shift_runtime_replay_per_ant=post_replay_coverage,
        )
        pe_off = replace(
            formalize(template_by_arm[DynamicColonyArmKind.PE_OFF_BUS.value]),
            post_shift_throughput_per_1k_actions=55.0,
            initial_checkpoint_fingerprints=shared_fingerprints,
            trained_policy_fingerprints=tuple(
                f"pe-trained:{seed}:body:{body_id}"
                for body_id in range(8)
            ),
            shift_policy_fingerprints=tuple(
                f"pe-shift:{seed}:body:{body_id}" for body_id in range(8)
            ),
            adaptation_start_policy_fingerprints=tuple(
                f"pe-adapt-start:{seed}:body:{body_id}"
                for body_id in range(8)
            ),
            final_policy_fingerprints=tuple(
                f"pe-final:{seed}:body:{body_id}" for body_id in range(8)
            ),
            runtime_replay_captured=100,
            runtime_replay_settled=100,
            runtime_replay_transitions=100,
            runtime_replay_lineage_matches=100,
            runtime_replay_lineage_coverage=1.0,
            runtime_replay_active=True,
            runtime_replay_per_ant=full_replay_coverage,
            post_shift_runtime_replay_per_ant=post_replay_coverage,
        )
        fixed = replace(
            formalize(
                template_by_arm[DynamicColonyArmKind.FIXED_RULE_BUS.value]
            ),
            pre_shift_throughput_per_1k_actions=100.0,
            post_shift_throughput_per_1k_actions=100.0,
            pre_shift_pickups=1,
            post_shift_delivered=1,
            recovery_rounds=10,
            oracle_delivery_shortfall=10.0,
        )
        arms = (
            learned,
            learned_no_bus,
            no_optimize,
            pe_off,
            fixed,
            formalize(
                template_by_arm[DynamicColonyArmKind.FIXED_RULE_NO_BUS.value]
            ),
            formalize(
                template_by_arm[DynamicColonyArmKind.RANDOM_NO_BUS.value]
            ),
        )
        synthetic_reports.append(
            replace(
                report,
                seed=seed,
                training_world_seed=seed * 2 + 17,
                evaluation_world_seed=seed * 2 + 1_000_003,
                config=formal_config,
                arms=arms,
            )
        )

    aggregate = aggregate_dynamic_colony_reports(
        tuple(synthetic_reports),
        seed_order=tuple(range(10)),
    )
    assert aggregate.verdict == "PASS"
    assert all(gate.passed for gate in aggregate.gates)
    with pytest.raises(ValueError, match="distinct"):
        aggregate_dynamic_colony_reports(
            tuple(synthetic_reports),
            seed_order=(0,) * 10,
        )


async def test_real_tiny_dynamic_colony_remains_honestly_blocked() -> None:
    config = DynamicColonyConfig(
        n_ants=1,
        training_rounds=0,
        pre_shift_rounds=1,
        post_shift_rounds=1,
        recovery_window=1,
        temporal_latent_dim=4,
    )
    report = await run_dynamic_colony_seed(
        seed=7,
        perturbation=DynamicPerturbationKind.FOOD_RELOCATION,
        config=config,
        learned_config=AntSessionConfig(temporal_latent_dim=4, seed=7),
        no_optimize_config=AntSessionConfig(
            temporal_latent_dim=4,
            seed=7,
            joint_apply_policy_optimization=False,
        ),
        pe_off_config=AntSessionConfig(
            temporal_latent_dim=4,
            seed=7,
            external_prediction_error_drive=False,
        ),
    )
    aggregate = aggregate_dynamic_colony_reports((report,), seed_order=(7,))

    assert aggregate.verdict == "BLOCK"
    assert next(
        gate for gate in aggregate.gates if gate.gate_name == "formal_seed_count"
    ).passed is False


def test_food_relocation_reports_body_overlap() -> None:
    world = AntWorld()
    world.set_body_pose(x=-5.0, y=0.0, heading=0.0)

    overlap = _apply_perturbation(
        world,
        perturbation=DynamicPerturbationKind.FOOD_RELOCATION,
    )

    assert overlap == 1
