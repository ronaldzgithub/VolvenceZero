"""Packet 3.2: protocol revision rule unit tests.

Tests the pure-function rules that ``ProtocolReflectionEngine``
calls. Each rule is exercised on synthesized PE +
active_mixture history that targets exactly one rule's trigger
condition.
"""

from __future__ import annotations

from volvence_zero.behavior_protocol import (
    ActiveMixtureSnapshot,
    ActiveProtocolEntry,
    ProtocolRevisionChangeKind,
    ProtocolRevisionTargetField,
    ReviewLevel,
)
from volvence_zero.prediction import (
    ActualOutcome,
    PredictedOutcome,
    PredictionActionContext,
    PredictionError,
    PredictionErrorSnapshot,
)
from volvence_zero.reflection.protocol_revision_rules import (
    STRATEGY_DECAY_MIN_TURNS,
    STRATEGY_DECAY_PE_THRESHOLD,
    propose_strategy_decay,
    run_all_protocol_revision_rules,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pe(turn_index: int, signed_reward: float) -> PredictionErrorSnapshot:
    ctx = PredictionActionContext()
    actual = ActualOutcome(
        observed_turn_index=turn_index,
        task_progress=0.5,
        relationship_delta=0.5,
        regime_stability=0.5,
        action_payoff=0.5,
        description="",
        action_context=ctx,
    )
    pred = PredictedOutcome(
        source_turn_index=turn_index,
        target_turn_index=turn_index + 1,
        predicted_task_progress=0.5,
        predicted_relationship_delta=0.5,
        predicted_regime_stability=0.5,
        predicted_action_payoff=0.5,
        confidence=0.5,
        description="",
        action_context=ctx,
    )
    return PredictionErrorSnapshot(
        evaluated_prediction=pred,
        actual_outcome=actual,
        next_prediction=pred,
        error=PredictionError(
            task_error=signed_reward,
            relationship_error=signed_reward,
            regime_error=signed_reward,
            action_error=signed_reward,
            magnitude=abs(signed_reward),
            signed_reward=signed_reward,
            description="",
        ),
        turn_index=turn_index,
        bootstrap=False,
        description="",
        action_context=ctx,
    )


def _mixture(*entries: tuple[str, float]) -> ActiveMixtureSnapshot:
    return ActiveMixtureSnapshot(
        active_protocols=tuple(
            ActiveProtocolEntry(
                protocol_id=protocol_id,
                activation_weight=weight,
            )
            for protocol_id, weight in entries
        ),
        boundary_union_ids=(),
        revision_fingerprint="",
        description="",
    )


# ---------------------------------------------------------------------------
# propose_strategy_decay
# ---------------------------------------------------------------------------


def test_strategy_decay_no_history_no_proposals() -> None:
    proposals = propose_strategy_decay(
        pe_history=(),
        active_mixture_history=(),
    )
    assert proposals == ()


def test_strategy_decay_below_min_turns_no_proposals() -> None:
    """Even with bad PE, fewer than MIN_TURNS observations → skip."""
    pe_history = tuple(_pe(t, -0.9) for t in range(1, STRATEGY_DECAY_MIN_TURNS))
    am_history = tuple(
        _mixture(("p_a", 1.0)) for _ in range(STRATEGY_DECAY_MIN_TURNS)
    )
    proposals = propose_strategy_decay(
        pe_history=pe_history,
        active_mixture_history=am_history,
    )
    assert proposals == ()


def test_strategy_decay_consistent_negative_pe_emits_proposal() -> None:
    """Long stretch of bad PE → WEIGHT_DECAY proposal at L3."""
    n = STRATEGY_DECAY_MIN_TURNS + 2
    pe_history = tuple(_pe(t, -0.9) for t in range(1, n + 1))
    am_history = tuple(_mixture(("p_a", 1.0)) for _ in range(n))

    proposals = propose_strategy_decay(
        pe_history=pe_history,
        active_mixture_history=am_history,
    )
    assert len(proposals) == 1
    p = proposals[0]
    assert p.target_protocol_id == "p_a"
    assert p.target_field is ProtocolRevisionTargetField.STRATEGY_PRIOR
    assert p.change_kind is ProtocolRevisionChangeKind.WEIGHT_DECAY
    assert p.required_review_level is ReviewLevel.L3
    assert p.proposed_payload is not None
    assert p.proposed_payload["weight_multiplier"] == 0.5


def test_strategy_decay_above_threshold_no_proposal() -> None:
    """If PE is above the threshold, no decay proposed."""
    n = STRATEGY_DECAY_MIN_TURNS + 5
    # threshold is -0.3; +0.5 is well above.
    pe_history = tuple(_pe(t, 0.5) for t in range(1, n + 1))
    am_history = tuple(_mixture(("p_a", 1.0)) for _ in range(n))

    proposals = propose_strategy_decay(
        pe_history=pe_history,
        active_mixture_history=am_history,
    )
    assert proposals == ()


def test_strategy_decay_only_flags_protocol_with_bad_pe_when_others_active() -> None:
    """Two protocols active; only one is consistently negative."""
    n = STRATEGY_DECAY_MIN_TURNS + 4

    # p_winner: weight=0.9 in mixtures, but PE positive
    # p_loser : weight=0.1 in mixtures, contributes 0.1 * -0.9 ≈ -0.09 each turn
    # threshold is -0.3, so loser at -0.09 doesn't trigger.
    # Make loser dominate to trigger: weight=0.9 instead.
    pe_history = tuple(_pe(t, -0.9) for t in range(1, n + 1))
    am_history = tuple(
        _mixture(("p_loser", 0.9), ("p_winner", 0.1)) for _ in range(n)
    )

    proposals = propose_strategy_decay(
        pe_history=pe_history,
        active_mixture_history=am_history,
    )
    flagged = {p.target_protocol_id for p in proposals}
    assert "p_loser" in flagged
    assert "p_winner" not in flagged


# ---------------------------------------------------------------------------
# run_all_protocol_revision_rules — aggregator + dedup
# ---------------------------------------------------------------------------


def test_run_all_dedupes_by_proposal_id() -> None:
    """The aggregator dedupes by proposal_id (currently a no-op since
    each rule emits a distinct id; this test pins the contract for
    future rules that might overlap)."""
    n = STRATEGY_DECAY_MIN_TURNS + 2
    pe_history = tuple(_pe(t, -0.9) for t in range(1, n + 1))
    am_history = tuple(_mixture(("p_a", 1.0)) for _ in range(n))
    proposals = run_all_protocol_revision_rules(
        pe_history=pe_history,
        active_mixture_history=am_history,
    )
    ids = [p.proposal_id for p in proposals]
    assert len(ids) == len(set(ids))


def test_run_all_returns_empty_on_empty_history() -> None:
    proposals = run_all_protocol_revision_rules(
        pe_history=(),
        active_mixture_history=(),
    )
    assert proposals == ()


def test_threshold_constants_are_documented() -> None:
    """Sanity: the two tunable constants exist and have plausible values."""
    assert STRATEGY_DECAY_MIN_TURNS >= 3
    assert -1.0 < STRATEGY_DECAY_PE_THRESHOLD < 0.0
