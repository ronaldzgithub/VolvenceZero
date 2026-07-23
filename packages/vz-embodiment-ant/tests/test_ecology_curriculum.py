"""Small-budget smoke for the real ecology training/evaluation path."""

from __future__ import annotations

from volvence_ant.experiments.ecology_curriculum import (
    ECOLOGY_REQUIRED_GATE_NAMES,
    EcologyCurriculumConfig,
    EcologyDataSplit,
    EcologyStage,
    EcologyTrainingTier,
    _synchronize_forced_return_navigators,
    _world,
    _session_config,
    train_and_evaluate_ecology_checkpoint,
)
from volvence_ant.runtime import KernelColonyRunner


def test_forced_return_curriculum_balances_state_without_action_labels() -> None:
    config = EcologyCurriculumConfig(
        n_ants=2,
        temporal_latent_dim=4,
        stage_rounds=1,
        stage_episodes=1,
        mastery_min_episodes=1,
        validation_rounds=1,
        validation_seeds=(13,),
        heldout_rounds=1,
        heldout_seeds=(19,),
        seed=2,
    )
    world = _world(
        config=config,
        stage=EcologyStage.BUTTER,
        seed=10,
        data_split=EcologyDataSplit.TRAIN,
        tier=EcologyTrainingTier.NEAR,
        forced_return=True,
    )
    runner = KernelColonyRunner(
        world,
        base_config=_session_config(
            config=config,
            seed=10,
            session_id="forced-return-test",
            optimize=False,
        ),
    )

    _synchronize_forced_return_navigators(runner)

    assert all(world.body(body_id).carrying_food for body_id in range(2))
    assert all(
        session.navigator.state.home_distance > world.config.nest_radius
        for session in runner.sessions
    )
    home_sides = tuple(
        session.navigator.egocentric_home()[1]
        for session in runner.sessions
    )
    assert home_sides[0] * home_sides[1] < 0.0


async def test_ecology_curriculum_exports_checkpoint_and_honest_gates() -> None:
    candidate = await train_and_evaluate_ecology_checkpoint(
        EcologyCurriculumConfig(
            n_ants=1,
            temporal_latent_dim=4,
            stage_rounds=1,
            stage_episodes=1,
            mastery_min_episodes=1,
            mastery_min_pickups=1,
            mastery_min_heat_events=1,
            validation_rounds=1,
            validation_seeds=(13,),
            heldout_rounds=3,
            heldout_seeds=(19,),
            seed=3,
        )
    )

    assert len(candidate.checkpoints) == 1
    assert len(candidate.checkpoint_archives) == 1
    assert candidate.report.verdict in {"PASS", "BLOCK"}
    assert candidate.report.gates
    assert tuple(gate.name for gate in candidate.report.gates) == ECOLOGY_REQUIRED_GATE_NAMES
    assert len(candidate.report.learned_metrics) == 5
    assert {
        item.scenario.value
        for item in candidate.report.learned_metrics
    } == {
        "butter_only",
        "butter_with_neutral_stick",
        "heat_route_avoidance",
        "heat_forced_escape",
        "composite",
    }
    assert candidate.report.training_schedule
    assert candidate.report.learned_training
    assert all(
        len(item.body_lineage) == 1
        and item.body_lineage[0].body_id == 0
        and item.body_lineage[0].episode_id
        and item.body_lineage[0].layout_seed == item.plan.seed
        for item in candidate.report.learned_training
    )
    assert candidate.report.action_probes
    assert all(
        item.policy_fingerprint_stable
        and item.temporal_learning_fingerprint_stable
        for item in candidate.report.learned_metrics
    )
    assert {
        gate.name for gate in candidate.report.gates
    } >= {
        "burning_match_route_avoidance",
        "burning_match_forced_escape",
        "checkpoint_archive_roundtrip",
    }
    assert candidate.report.learned_metrics[0].replay_captured >= 0
    assert all(
        item.replay_pending_captures >= 0 and item.replay_settlement_coverage == 1.0
        for item in candidate.report.learned_metrics
    )
    if candidate.report.verdict == "BLOCK":
        assert candidate.report.diagnostic_breakpoints
