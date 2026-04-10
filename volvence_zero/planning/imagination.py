"""z_t space imagination rollout for multi-step planning.

Generates K candidate abstract action sequences, simulates their
prediction-error trajectories speculatively, and selects the best
one before the system commits to a real response.

This is a **pure function** module — it does not publish to the
snapshot bus and does not mutate any module state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from volvence_zero.dual_track import DualTrackSnapshot
from volvence_zero.evaluation import EvaluationSnapshot
from volvence_zero.prediction.error import (
    ActualOutcome,
    PredictedOutcome,
    PredictionError,
    PredictionErrorModule,
    _clamp_signed,
    _clamp_unit,
)
from volvence_zero.substrate import SubstrateSnapshot

if TYPE_CHECKING:
    from volvence_zero.regime import RegimeSnapshot


@dataclass(frozen=True)
class ImaginedTrajectory:
    candidate_id: str
    z_sequence: tuple[tuple[float, ...], ...]
    predicted_pe_trajectory: tuple[PredictionError, ...]
    cumulative_reward: float
    description: str


@dataclass(frozen=True)
class ImaginationResult:
    trajectories: tuple[ImaginedTrajectory, ...]
    selected_trajectory: ImaginedTrajectory
    selection_reason: str
    description: str


def imagine(
    *,
    current_substrate: SubstrateSnapshot | None,
    current_evaluation: EvaluationSnapshot,
    current_dual_track: DualTrackSnapshot,
    current_regime: "RegimeSnapshot | None",
    previous_prediction: PredictedOutcome | None,
    action_family_centroids: tuple[tuple[str, tuple[float, ...]], ...],
    prior_mean: tuple[float, ...],
    prior_std: tuple[float, ...],
    n_candidates: int = 3,
    horizon: int = 3,
    gamma: float = 0.95,
) -> ImaginationResult:
    """Generate K candidate z_t sequences and evaluate their PE trajectories."""
    candidates = _generate_candidates(
        prior_mean=prior_mean,
        prior_std=prior_std,
        action_family_centroids=action_family_centroids,
        n_candidates=n_candidates,
    )

    pe_module = PredictionErrorModule()

    base_prediction = previous_prediction
    if base_prediction is None:
        base_prediction = pe_module.compute_prediction(
            source_turn_index=0,
            substrate_snapshot=current_substrate,
            evaluation_snapshot=current_evaluation,
            dual_track_snapshot=current_dual_track,
            regime_snapshot=current_regime,
        )

    trajectories: list[ImaginedTrajectory] = []
    for candidate_id, z_t in candidates:
        pe_trajectory, cumulative = _simulate_trajectory(
            z_t=z_t,
            base_prediction=base_prediction,
            current_evaluation=current_evaluation,
            current_dual_track=current_dual_track,
            horizon=horizon,
            gamma=gamma,
        )
        z_sequence = tuple(z_t for _ in range(horizon))
        trajectories.append(
            ImaginedTrajectory(
                candidate_id=candidate_id,
                z_sequence=z_sequence,
                predicted_pe_trajectory=pe_trajectory,
                cumulative_reward=round(cumulative, 4),
                description=(
                    f"Candidate {candidate_id}: z={tuple(round(v, 3) for v in z_t)} "
                    f"cumulative_reward={cumulative:.3f} over {horizon} steps."
                ),
            )
        )

    trajectories.sort(key=lambda t: -t.cumulative_reward)
    selected = trajectories[0]
    reason = (
        f"Selected {selected.candidate_id} with cumulative_reward={selected.cumulative_reward:.3f} "
        f"(best of {len(trajectories)} candidates)."
    )

    return ImaginationResult(
        trajectories=tuple(trajectories),
        selected_trajectory=selected,
        selection_reason=reason,
        description=(
            f"Imagination rollout: {len(trajectories)} candidates, horizon={horizon}, "
            f"gamma={gamma:.2f}. Winner: {selected.candidate_id} "
            f"reward={selected.cumulative_reward:.3f}."
        ),
    )


def _generate_candidates(
    *,
    prior_mean: tuple[float, ...],
    prior_std: tuple[float, ...],
    action_family_centroids: tuple[tuple[str, tuple[float, ...]], ...],
    n_candidates: int,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    """Generate K candidate z_t vectors.

    Strategy priority:
    1. Use action family centroids if enough are available
    2. Otherwise sample from the variational prior N(mean, std)
    """
    candidates: list[tuple[str, tuple[float, ...]]] = []

    for family_id, centroid in action_family_centroids[:n_candidates]:
        candidates.append((f"family:{family_id}", centroid))

    remaining = n_candidates - len(candidates)
    if remaining > 0:
        n_z = len(prior_mean)
        for seed_index in range(remaining):
            z = _deterministic_sample(
                mean=prior_mean,
                std=prior_std,
                seed=seed_index + 42,
                n_z=n_z,
            )
            candidates.append((f"prior_sample:{seed_index}", z))

    return tuple(candidates[:n_candidates])


def _simulate_trajectory(
    *,
    z_t: tuple[float, ...],
    base_prediction: PredictedOutcome,
    current_evaluation: EvaluationSnapshot,
    current_dual_track: DualTrackSnapshot,
    horizon: int,
    gamma: float,
) -> tuple[tuple[PredictionError, ...], float]:
    """Simulate a multi-step PE trajectory for a candidate z_t.

    This is a heuristic world model: we shift the base prediction by
    the z_t's implied direction and compute speculative PE at each step.
    """
    z_influence = _z_influence(z_t)
    pe_trajectory: list[PredictionError] = []
    cumulative = 0.0
    predicted = base_prediction

    for step in range(horizon):
        speculative_actual = ActualOutcome(
            observed_turn_index=predicted.target_turn_index + step,
            task_progress=_clamp_unit(
                predicted.predicted_task_progress + z_influence["task"] * (0.8 ** step)
            ),
            relationship_delta=_clamp_unit(
                predicted.predicted_relationship_delta + z_influence["relationship"] * (0.8 ** step)
            ),
            regime_stability=_clamp_unit(
                predicted.predicted_regime_stability + z_influence["stability"] * (0.8 ** step)
            ),
            action_payoff=_clamp_unit(
                predicted.predicted_action_payoff + z_influence["action"] * (0.8 ** step)
            ),
            description=f"Speculative actual at imagination step {step}.",
        )
        pe = PredictionError(
            task_error=_clamp_signed(
                speculative_actual.task_progress - predicted.predicted_task_progress
            ),
            relationship_error=_clamp_signed(
                speculative_actual.relationship_delta - predicted.predicted_relationship_delta
            ),
            regime_error=_clamp_signed(
                speculative_actual.regime_stability - predicted.predicted_regime_stability
            ),
            action_error=_clamp_signed(
                speculative_actual.action_payoff - predicted.predicted_action_payoff
            ),
            magnitude=0.0,
            signed_reward=0.0,
            description=f"Imagination PE step {step}.",
        )
        magnitude = abs(pe.task_error) + abs(pe.relationship_error) + abs(pe.regime_error) + abs(pe.action_error)
        signed_reward = (pe.task_error + pe.relationship_error + pe.regime_error + pe.action_error) / 4.0
        pe = PredictionError(
            task_error=pe.task_error,
            relationship_error=pe.relationship_error,
            regime_error=pe.regime_error,
            action_error=pe.action_error,
            magnitude=round(magnitude, 4),
            signed_reward=round(signed_reward, 4),
            description=f"Imagination PE step {step}: reward={signed_reward:.3f}.",
        )
        pe_trajectory.append(pe)
        cumulative += (gamma ** step) * signed_reward

        predicted = PredictedOutcome(
            source_turn_index=predicted.source_turn_index + step + 1,
            target_turn_index=predicted.target_turn_index + step + 2,
            predicted_task_progress=speculative_actual.task_progress,
            predicted_relationship_delta=speculative_actual.relationship_delta,
            predicted_regime_stability=speculative_actual.regime_stability,
            predicted_action_payoff=speculative_actual.action_payoff,
            confidence=predicted.confidence * 0.9,
            description=f"Carried-forward prediction at imagination step {step + 1}.",
        )

    return tuple(pe_trajectory), round(cumulative, 4)


def _z_influence(z_t: tuple[float, ...]) -> dict[str, float]:
    """Derive directional influence from a z_t vector.

    Maps the z_t dimensions to the four PE prediction dimensions.
    Uses a simple projection: first dims → task, middle → relationship,
    later → stability/action.
    """
    n = len(z_t)
    if n == 0:
        return {"task": 0.0, "relationship": 0.0, "stability": 0.0, "action": 0.0}

    task_dims = z_t[: max(n // 4, 1)]
    rel_dims = z_t[max(n // 4, 1) : max(n // 2, 2)]
    stab_dims = z_t[max(n // 2, 2) : max(3 * n // 4, 3)]
    action_dims = z_t[max(3 * n // 4, 3) :]

    def _mean_shift(dims: tuple[float, ...] | list[float]) -> float:
        if not dims:
            return 0.0
        return _clamp_signed((sum(dims) / len(dims) - 0.5) * 0.3)

    return {
        "task": _mean_shift(task_dims),
        "relationship": _mean_shift(rel_dims),
        "stability": _mean_shift(stab_dims) if stab_dims else 0.0,
        "action": _mean_shift(action_dims) if action_dims else _mean_shift(z_t),
    }


def _deterministic_sample(
    *,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    seed: int,
    n_z: int,
) -> tuple[float, ...]:
    """Deterministic pseudo-random sample from N(mean, std)."""
    result: list[float] = []
    for i in range(n_z):
        hash_val = ((seed * 2654435761 + i * 40503) % 65537) / 32768.5 - 1.0
        m = mean[i] if i < len(mean) else 0.5
        s = std[i] if i < len(std) else 0.1
        result.append(_clamp_unit(m + s * hash_val))
    return tuple(result)
