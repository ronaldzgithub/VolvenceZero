from __future__ import annotations

import pytest

from volvence_zero.apprenticeship.active_learning import (
    BoundedLabelUtilityReadout,
    Gate4FeedbackCandidate,
    label_utility_features,
    select_active_candidate,
)


def _candidate(
    transition_id: str,
    *,
    segment: tuple[float, ...],
) -> Gate4FeedbackCandidate:
    return Gate4FeedbackCandidate(
        transition_id=transition_id,
        guidance_text="typed guidance",
        turn_index=1,
        predictor_features=(0.0,),
        pe_magnitude=0.5,
        segment_features=segment,
    )


def test_label_utility_readout_learns_observed_ranking() -> None:
    readout = BoundedLabelUtilityReadout(
        feature_count=2,
        minimum_observations=4,
    )
    observations = (
        ((1.0, 0.0), 0.50, 0.10),
        ((0.8, 0.1), 0.45, 0.15),
        ((0.0, 1.0), 0.40, 0.39),
        ((0.1, 0.8), 0.35, 0.34),
    )
    for features, loss_before, loss_after in observations:
        readout.observe_loss_delta(
            features=features,
            loss_before=loss_before,
            loss_after=loss_after,
        )

    assert readout.ready
    assert (
        readout.predict_utility((0.9, 0.0))
        > readout.predict_utility((0.0, 0.9))
    )
    assert all(abs(value) <= 4.0 for value in readout.parameters)


def test_segment_selector_exposes_uncertainty_cold_start() -> None:
    candidates = (
        _candidate("low-uncertainty", segment=(0.0, 0.0, 0.1)),
        _candidate("high-uncertainty", segment=(1.0, 1.0, 0.9)),
    )
    readout = BoundedLabelUtilityReadout(
        feature_count=8,
        minimum_observations=2,
    )

    assert not readout.ready
    assert (
        select_active_candidate(
            candidates=candidates,
            probabilities=(0.05, 0.50),
            selected_indices=frozenset(),
            policy="segment-aware-active",
            utility_readout=readout,
        )
        == 1
    )
    with pytest.raises(RuntimeError, match="cold start"):
        readout.predict_utility((0.0,) * 8)


def test_segment_coverage_reduces_novelty_feature() -> None:
    candidate = _candidate("candidate", segment=(0.2, 0.8, 1.0))
    uncovered = label_utility_features(
        candidate=candidate,
        probability=0.5,
        selected_segment_features=(),
    )
    covered = label_utility_features(
        candidate=candidate,
        probability=0.5,
        selected_segment_features=(candidate.segment_features,),
    )

    assert uncovered[2] == 1.0
    assert uncovered[4] == 0.0
    assert covered[2] == 0.0
    assert covered[4] == 1.0
