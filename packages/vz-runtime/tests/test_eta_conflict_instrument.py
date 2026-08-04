from __future__ import annotations

import pytest

from volvence_zero.agent.eta_conflict_instrument import (
    ConflictBaseUncertaintyMetrics,
    ConflictInstrumentThresholds,
    assess_conflict_instrument,
    build_conflict_junction_rows,
    compute_conflict_headroom,
)
from volvence_zero.agent.eta_proof_benchmark import generate_eta_proof_corpus


def _corpus():
    return generate_eta_proof_corpus(
        seed=20260802,
        objective_count=8,
        corridor_count=2,
        extra_edge_probability=0.35,
        train_route_count=32,
        heldout_route_count=16,
        train_lengths=(2, 3),
        heldout_lengths=(3, 4),
    )


def test_goal_stripped_text_never_leaks_the_subgoal():
    corpus = _corpus()
    rows = build_conflict_junction_rows(corpus, split="heldout")
    assert rows
    for row in rows:
        # The goal-ambiguous view must not announce the goal: no objective
        # declaration, no route plan, no completed-objective ledger. (The
        # subgoal's own node id may legitimately appear as one available
        # out-edge -- that is part of the local view, not a goal reveal.)
        assert "Objective:" not in row.observation_text
        assert "Route plan" not in row.observation_text
        assert "Next objective" not in row.observation_text
        assert "Completed" not in row.observation_text
        if row.active_subgoal is not None:
            # The paired control DOES reveal exactly the missing bit.
            assert row.subgoal_revealed_text.startswith(
                f"Objective: {row.active_subgoal}."
            )


def test_conflict_headroom_has_room_no_constant_map_can_saturate():
    corpus = _corpus()
    rows = build_conflict_junction_rows(corpus, split="heldout")
    headroom = compute_conflict_headroom(rows, split="heldout")
    # Under goal stripping the vast majority of junctions are branching
    # conflict views; a constant map cannot cover them.
    assert headroom.conflict_row_fraction >= 0.50
    # A single constant action per view is wrong on a large fraction of rows.
    assert headroom.constant_operator_error_rate >= 0.20
    # Conditioning on the subgoal makes the action deterministic: the subgoal
    # is exactly the missing bit a conditional policy must supply.
    assert headroom.view_subgoal_residual_ambiguity == 0
    assert headroom.oracle_conditional_error_rate == pytest.approx(0.0)


def test_assess_structural_only_admits_valid_instrument():
    corpus = _corpus()
    rows = build_conflict_junction_rows(corpus, split="heldout")
    headroom = compute_conflict_headroom(rows, split="heldout")
    admission = assess_conflict_instrument(
        headroom=headroom,
        thresholds=ConflictInstrumentThresholds(),
        base_uncertainty=None,
    )
    assert admission.valid
    assert admission.failed_conditions == ()
    assert admission.base_uncertainty_evaluated is False


def test_assess_fails_when_constant_operator_saturates():
    corpus = _corpus()
    rows = build_conflict_junction_rows(corpus, split="heldout")
    headroom = compute_conflict_headroom(rows, split="heldout")
    admission = assess_conflict_instrument(
        headroom=headroom,
        thresholds=ConflictInstrumentThresholds(
            min_constant_operator_error=0.99
        ),
        base_uncertainty=None,
    )
    assert not admission.valid
    assert "constant-operator-error" in admission.failed_conditions


def test_assess_requires_base_headroom_when_uncertainty_supplied():
    corpus = _corpus()
    rows = build_conflict_junction_rows(corpus, split="heldout")
    headroom = compute_conflict_headroom(rows, split="heldout")
    saturated_base = ConflictBaseUncertaintyMetrics(
        split="heldout",
        scored_row_count=100,
        goal_stripped_mean_expert_nll=0.01,
        goal_stripped_median_expert_nll=0.01,
        subgoal_revealed_mean_expert_nll=0.01,
        steerable_headroom_nll=0.0,
        fraction_base_uncertain=0.0,
        uncertain_nll_threshold=0.10,
    )
    admission = assess_conflict_instrument(
        headroom=headroom,
        thresholds=ConflictInstrumentThresholds(),
        base_uncertainty=saturated_base,
    )
    assert not admission.valid
    assert admission.base_uncertainty_evaluated is True
    assert "base-steerable-headroom" in admission.failed_conditions
    assert "base-uncertain-fraction" in admission.failed_conditions
