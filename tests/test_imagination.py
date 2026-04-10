from __future__ import annotations

import asyncio

from volvence_zero.agent.session import AgentSessionRunner
from volvence_zero.dual_track import DualTrackSnapshot, TrackState
from volvence_zero.evaluation import EvaluationScore, EvaluationSnapshot
from volvence_zero.memory import Track
from volvence_zero.planning import ImaginationResult, imagine
from volvence_zero.prediction import PredictedOutcome


def _evaluation_snapshot() -> EvaluationSnapshot:
    return EvaluationSnapshot(
        turn_scores=(
            EvaluationScore("task", "task_pressure", 0.7, 0.8, "task"),
            EvaluationScore("relationship", "relationship_continuity", 0.6, 0.8, "rel"),
            EvaluationScore("learning", "joint_learning_progress", 0.5, 0.8, "learn"),
            EvaluationScore("abstraction", "abstract_action_usefulness", 0.55, 0.8, "abs"),
            EvaluationScore("safety", "contract_integrity", 0.9, 0.9, "safe"),
        ),
        session_scores=(),
        alerts=(),
        description="eval",
    )


def _dual_track_snapshot() -> DualTrackSnapshot:
    return DualTrackSnapshot(
        world_track=TrackState(track=Track.WORLD, active_goals=("task",), recent_credits=(), controller_code=(0.3, 0.4), tension_level=0.4),
        self_track=TrackState(track=Track.SELF, active_goals=("support",), recent_credits=(), controller_code=(0.4, 0.3), tension_level=0.5),
        cross_track_tension=0.2,
        description="dual",
    )


def test_imagine_produces_k_trajectories_with_distinct_pe():
    result = imagine(
        current_substrate=None,
        current_evaluation=_evaluation_snapshot(),
        current_dual_track=_dual_track_snapshot(),
        current_regime=None,
        previous_prediction=None,
        action_family_centroids=(
            ("family_a", (0.8, 0.2, 0.5)),
            ("family_b", (0.2, 0.8, 0.5)),
            ("family_c", (0.5, 0.5, 0.8)),
        ),
        prior_mean=(0.5, 0.5, 0.5),
        prior_std=(0.1, 0.1, 0.1),
        n_candidates=3,
        horizon=3,
    )

    assert isinstance(result, ImaginationResult)
    assert len(result.trajectories) == 3
    assert result.selected_trajectory in result.trajectories

    rewards = [t.cumulative_reward for t in result.trajectories]
    assert len(set(rewards)) >= 2, f"Expected at least 2 distinct rewards, got {rewards}"

    for trajectory in result.trajectories:
        assert len(trajectory.z_sequence) == 3
        assert len(trajectory.predicted_pe_trajectory) == 3
        for pe in trajectory.predicted_pe_trajectory:
            assert -1.0 <= pe.signed_reward <= 1.0


def test_imagine_with_prior_sampling_when_no_families():
    result = imagine(
        current_substrate=None,
        current_evaluation=_evaluation_snapshot(),
        current_dual_track=_dual_track_snapshot(),
        current_regime=None,
        previous_prediction=None,
        action_family_centroids=(),
        prior_mean=(0.5, 0.5, 0.5),
        prior_std=(0.15, 0.15, 0.15),
        n_candidates=3,
        horizon=2,
    )

    assert len(result.trajectories) == 3
    assert all(t.candidate_id.startswith("prior_sample:") for t in result.trajectories)


def test_imagine_selects_best_cumulative_reward():
    result = imagine(
        current_substrate=None,
        current_evaluation=_evaluation_snapshot(),
        current_dual_track=_dual_track_snapshot(),
        current_regime=None,
        previous_prediction=None,
        action_family_centroids=(
            ("good_family", (0.9, 0.9, 0.9)),
            ("bad_family", (0.01, 0.01, 0.01)),
        ),
        prior_mean=(0.5, 0.5, 0.5),
        prior_std=(0.1, 0.1, 0.1),
        n_candidates=2,
        horizon=3,
    )

    assert result.selected_trajectory.cumulative_reward >= result.trajectories[-1].cumulative_reward


def test_imagine_uses_previous_prediction_when_available():
    prev_pred = PredictedOutcome(
        source_turn_index=1,
        target_turn_index=2,
        predicted_task_progress=0.3,
        predicted_relationship_delta=0.3,
        predicted_regime_stability=0.5,
        predicted_action_payoff=0.4,
        confidence=0.7,
        description="prev",
    )
    result = imagine(
        current_substrate=None,
        current_evaluation=_evaluation_snapshot(),
        current_dual_track=_dual_track_snapshot(),
        current_regime=None,
        previous_prediction=prev_pred,
        action_family_centroids=(("fam_x", (0.8, 0.7, 0.6)),),
        prior_mean=(0.5, 0.5, 0.5),
        prior_std=(0.1, 0.1, 0.1),
        n_candidates=2,
        horizon=2,
    )

    assert len(result.trajectories) == 2
    assert result.selected_trajectory.cumulative_reward != 0.0


def test_agent_session_runner_produces_imagination_result():
    runner = AgentSessionRunner(session_id="imagination-session")
    result = asyncio.run(runner.run_turn("Help me think through a complex decision."))

    assert result.imagination_result is not None
    assert isinstance(result.imagination_result, ImaginationResult)
    assert len(result.imagination_result.trajectories) >= 1
    assert result.imagination_result.selected_trajectory is not None


def test_imagination_is_deterministic():
    result1 = imagine(
        current_substrate=None,
        current_evaluation=_evaluation_snapshot(),
        current_dual_track=_dual_track_snapshot(),
        current_regime=None,
        previous_prediction=None,
        action_family_centroids=(),
        prior_mean=(0.5, 0.5, 0.5),
        prior_std=(0.1, 0.1, 0.1),
        n_candidates=3,
        horizon=3,
    )
    result2 = imagine(
        current_substrate=None,
        current_evaluation=_evaluation_snapshot(),
        current_dual_track=_dual_track_snapshot(),
        current_regime=None,
        previous_prediction=None,
        action_family_centroids=(),
        prior_mean=(0.5, 0.5, 0.5),
        prior_std=(0.1, 0.1, 0.1),
        n_candidates=3,
        horizon=3,
    )
    assert result1.selected_trajectory.cumulative_reward == result2.selected_trajectory.cumulative_reward
    assert result1.selected_trajectory.candidate_id == result2.selected_trajectory.candidate_id
