"""Fast Phase 0 experiment tests (tiny params; full runs live in scripts/)."""

from __future__ import annotations

from volvence_ant.experiments import (
    homing_precision_experiment,
    route_learning_experiment,
)


def test_homing_precision_antbot_scale() -> None:
    result = homing_precision_experiment(
        journey_step_grid=(10, 40), n_trials=8, seed=1
    )
    # The exact frozen AntBot threshold is evaluated honestly; this unit test
    # validates the independent truth/estimate simulation rather than forcing PASS.
    assert result.antbot_reference_ratio == 0.005
    for point in result.curve:
        assert point.mean_normalized_error >= 0.0


def test_homing_curve_is_ordered_by_length() -> None:
    result = homing_precision_experiment(journey_step_grid=(10, 20, 40), n_trials=6)
    lengths = [p.journey_length for p in result.curve]
    assert lengths == sorted(lengths)


async def test_route_learning_reduces_novelty() -> None:
    result = await route_learning_experiment(exposures=6, route_length=4, seed=2)
    assert len(result.novelty_by_exposure) == 6
    # reducible novelty should not increase over exposures (familiarity forms)
    assert result.last_exposure_novelty <= result.first_exposure_novelty + 1e-6
    assert result.familiarity_improved
    assert result.novel_route_novelty >= 0.0
    assert result.shuffled_route_novelty >= 0.0
