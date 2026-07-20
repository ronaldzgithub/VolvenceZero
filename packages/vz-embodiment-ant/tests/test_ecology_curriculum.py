"""Small-budget smoke for the real ecology training/evaluation path."""

from __future__ import annotations

from volvence_ant.experiments.ecology_curriculum import (
    ECOLOGY_REQUIRED_GATE_NAMES,
    EcologyCurriculumConfig,
    train_and_evaluate_ecology_checkpoint,
)


async def test_ecology_curriculum_exports_checkpoint_and_honest_gates() -> None:
    candidate = await train_and_evaluate_ecology_checkpoint(
        EcologyCurriculumConfig(
            n_ants=1,
            temporal_latent_dim=4,
            stage_rounds=1,
            stage_episodes=1,
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
    assert len(candidate.report.learned_metrics) == 4
    assert {item.stage.value for item in candidate.report.learned_metrics} == {
        "butter",
        "wood_stick",
        "burning_match",
        "composite",
    }
    assert candidate.report.learned_metrics[0].replay_captured >= 0
    assert all(
        item.replay_pending_captures >= 0 and item.replay_settlement_coverage == 1.0
        for item in candidate.report.learned_metrics
    )
    if candidate.report.verdict == "BLOCK":
        assert candidate.report.diagnostic_breakpoints
