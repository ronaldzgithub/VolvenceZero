"""Contract tests for interval valuation, option value, and VOI.

The claims under test are the ones that make a decision aid safe to
believe, and each of them has an obvious cheaper alternative that
produces confident nonsense:

* Unknowns **widen** the interval; known haircuts **shift** it. The
  cheap alternative — collapsing an unresolved question to a midpoint —
  is how "the equity is probably worth X" gets stated about documents
  nobody has read.
* Dimensions aggregate **comonotonically**. The cheap alternative
  (independence) narrows the total, which makes separation claims
  easier to reach.
* A winner is only named on **strict separation**. The cheap
  alternative — argmax of the midpoints — always names one.
* An unknown is worth asking about only if its answer could **move the
  ranking**. The cheap alternative — asking about whatever feels most
  uncertain — is interrogation.

The last block replays the fourth-act situation. It is an acceptance
sample, not the source of the design: the assertions are about which
claims the arithmetic licenses, not about reproducing a transcript.
"""

from __future__ import annotations

import pytest

from volvence_zero.decision_workspace.rendering import (
    CLAIM_COMPARATIVE,
    CLAIM_NONE,
    CLAIM_ROBUSTNESS,
    licence_for,
)
from volvence_zero.decision_workspace.valuation import (
    DimensionEstimate,
    Interval,
    evaluate_options,
)


def _estimate(
    option: str,
    dimension: str,
    low: float,
    base: float,
    high: float,
    **kwargs: object,
) -> DimensionEstimate:
    return DimensionEstimate(
        option_ref=option,
        dimension_ref=dimension,
        interval=Interval(low, base, high),
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Interval arithmetic
# ---------------------------------------------------------------------------


def test_interval_rejects_incoherent_bounds() -> None:
    with pytest.raises(ValueError, match="low <= base <= high"):
        Interval(1.0, 0.0, 2.0)


def test_known_haircut_shifts_the_interval_down() -> None:
    """A three-year lock-up is a discount: the whole range moves."""
    estimate = _estimate("a", "money", 100.0, 200.0, 300.0, haircuts=(("lockup", 0.5),))
    resolved = estimate.resolved_interval()
    assert (resolved.low, resolved.base, resolved.high) == (50.0, 100.0, 150.0)


def test_unresolved_question_widens_without_moving_the_estimate() -> None:
    """The distinction the whole model rests on.

    "Nobody has read the equity documents" is not evidence that the
    working estimate is wrong. It is evidence that it is unsupported.
    Moving ``base`` would be inventing a finding; leaving the width
    alone would be inventing certainty.
    """
    estimate = _estimate(
        "a",
        "money",
        100.0,
        200.0,
        300.0,
        unresolved_refs=("who-owns-the-equity",),
        unresolved_spread=80.0,
    )
    resolved = estimate.resolved_interval()
    assert resolved.base == 200.0
    assert resolved.low == 20.0
    assert resolved.high == 380.0


def test_resolving_an_unknown_removes_its_widening() -> None:
    estimate = _estimate(
        "a",
        "money",
        100.0,
        200.0,
        300.0,
        unresolved_refs=("u1",),
        unresolved_spread=80.0,
    )
    resolved = estimate.resolved_interval(assume_resolved=frozenset({"u1"}))
    assert (resolved.low, resolved.high) == (100.0, 300.0)


def test_dimensions_aggregate_comonotonically() -> None:
    """Extremes add. Independence would narrow the total.

    A narrower total makes it easier to claim one option clears another,
    so the width-maximising assumption is the one that errs away from
    overclaiming.
    """
    result = evaluate_options(
        (
            _estimate("a", "money", 0.0, 10.0, 20.0),
            _estimate("a", "children", 0.0, 10.0, 20.0),
        )
    )
    option = result.options[0]
    assert (option.value.low, option.value.base, option.value.high) == (0.0, 20.0, 40.0)


# ---------------------------------------------------------------------------
# Separation
# ---------------------------------------------------------------------------


def test_overlapping_intervals_name_no_winner() -> None:
    result = evaluate_options(
        (
            _estimate("a", "money", 0.0, 12.0, 30.0),
            _estimate("b", "money", 0.0, 10.0, 30.0),
        )
    )
    assert result.leader_ref is None
    assert result.separated is False
    # An honest fallback still exists — it is just a different claim.
    assert result.most_robust_ref is not None


def test_separated_intervals_do_name_a_winner() -> None:
    result = evaluate_options(
        (
            _estimate("a", "money", 40.0, 50.0, 60.0),
            _estimate("b", "money", 0.0, 10.0, 20.0),
        )
    )
    assert result.leader_ref == "a"


def test_midpoint_argmax_is_not_enough_to_win() -> None:
    """The specific overclaim this design exists to block.

    ``a`` has the higher working estimate. Its range still overlaps
    ``b``'s, so "a is highest" is not a result the arithmetic has.
    """
    result = evaluate_options(
        (
            _estimate("a", "money", 0.0, 30.0, 100.0),
            _estimate("b", "money", 20.0, 25.0, 30.0),
        )
    )
    assert max(result.options, key=lambda o: o.value.base).option_ref == "a"
    assert result.leader_ref is None


# ---------------------------------------------------------------------------
# Option value
# ---------------------------------------------------------------------------


def test_reversible_option_that_buys_information_earns_option_value() -> None:
    """The formal content of "separate for three months".

    It is cheap to undo *and* it answers the things the ranking hangs
    on. Neither property alone is worth anything.
    """
    estimates = (
        _estimate(
            "wait", "money", 0.0, 10.0, 20.0, unresolved_refs=("u1",), unresolved_spread=30.0
        ),
        _estimate(
            "commit", "money", 0.0, 12.0, 24.0, unresolved_refs=("u1",), unresolved_spread=30.0
        ),
    )
    result = evaluate_options(
        estimates,
        reversibility=(("wait", 1.0), ("commit", 0.0)),
        resolves=(("wait", ("u1",)), ("commit", ("u1",))),
    )
    valuations = {item.option_ref: item for item in result.options}
    assert valuations["wait"].option_value > 0.0
    # ``commit`` resolves the same unknown but cannot act on the answer.
    assert valuations["commit"].option_value == 0.0


def test_option_value_does_not_raise_the_floor() -> None:
    """Preserved choices are worth nothing in the branch that goes badly."""
    result = evaluate_options(
        (
            _estimate("wait", "money", 5.0, 10.0, 20.0, unresolved_refs=("u1",), unresolved_spread=30.0),
            _estimate("commit", "money", 0.0, 40.0, 41.0),
        ),
        reversibility=(("wait", 1.0),),
        resolves=(("wait", ("u1",)),),
    )
    wait = next(item for item in result.options if item.option_ref == "wait")
    assert wait.total.low == wait.value.low
    assert wait.total.base > wait.value.base


def test_irreversible_option_gets_no_credit_for_information() -> None:
    result = evaluate_options(
        (
            _estimate("a", "money", 0.0, 10.0, 20.0, unresolved_refs=("u1",), unresolved_spread=30.0),
            _estimate("b", "money", 0.0, 10.0, 20.0),
        ),
        reversibility=(("a", 0.0),),
        resolves=(("a", ("u1",)),),
    )
    assert result.options[0].option_value == 0.0


# ---------------------------------------------------------------------------
# Value of information / termination
# ---------------------------------------------------------------------------


def test_unknown_that_cannot_change_the_ranking_is_not_worth_asking() -> None:
    """The termination condition, stated as a property.

    ``u1`` widens both options symmetrically and by an amount that
    cannot reorder them. Asking about it is interrogation.
    """
    result = evaluate_options(
        (
            _estimate("a", "money", 0.0, 10.0, 20.0),
            _estimate("b", "irrelevant", 0.0, 10.0, 20.0, unresolved_refs=("u1",), unresolved_spread=0.0),
        )
    )
    values = {u.unknown_ref: u for u in result.unknowns}
    assert values["u1"].is_worth_asking is False


def test_next_question_is_the_one_that_could_flip_the_leader() -> None:
    result = evaluate_options(
        (
            _estimate("a", "money", 40.0, 50.0, 60.0, unresolved_refs=("decisive",), unresolved_spread=45.0),
            _estimate("b", "money", 0.0, 10.0, 20.0, unresolved_refs=("trivial",), unresolved_spread=0.5),
        )
    )
    assert result.next_unknown_to_resolve() == "decisive"


def test_nothing_left_to_learn_returns_no_next_question() -> None:
    result = evaluate_options(
        (
            _estimate("a", "money", 40.0, 50.0, 60.0),
            _estimate("b", "money", 0.0, 10.0, 20.0),
        )
    )
    assert result.next_unknown_to_resolve() is None


def test_empty_input_is_not_an_error() -> None:
    result = evaluate_options(())
    assert result.options == ()
    assert result.next_unknown_to_resolve() is None


# ---------------------------------------------------------------------------
# Claim licence
# ---------------------------------------------------------------------------


def test_overlap_licenses_robustness_not_a_winner() -> None:
    result = evaluate_options(
        (
            _estimate("a", "money", 0.0, 12.0, 30.0),
            _estimate("b", "money", 0.0, 10.0, 30.0),
        )
    )
    licence = licence_for(result)
    assert licence.claim_kind == CLAIM_ROBUSTNESS
    assert licence.may_state_a_winner is False


def test_separation_licenses_a_comparative_claim() -> None:
    result = evaluate_options(
        (
            _estimate("a", "money", 40.0, 50.0, 60.0),
            _estimate("b", "money", 0.0, 10.0, 20.0),
        )
    )
    licence = licence_for(result)
    assert licence.claim_kind == CLAIM_COMPARATIVE
    assert licence.subject_ref == "a"


def test_figures_without_evidence_are_not_licensed_as_fact() -> None:
    result = evaluate_options(
        (
            _estimate("a", "equity", 0.0, 500.0, 900.0),
            _estimate("a", "salary", 0.0, 20.0, 25.0, evidence_refs=("payslip",)),
        )
    )
    licence = licence_for(result)
    assert licence.permits("salary") is True
    assert licence.permits("equity") is False


def test_open_unknowns_must_stay_visible_in_the_conclusion() -> None:
    result = evaluate_options(
        (
            _estimate("a", "money", 40.0, 50.0, 60.0, unresolved_refs=("who-owns-it",), unresolved_spread=45.0),
            _estimate("b", "money", 0.0, 10.0, 20.0),
        )
    )
    assert "who-owns-it" in licence_for(result).must_surface_unknown_refs


def test_safety_hold_revokes_every_ranking_claim() -> None:
    """Separation does not buy past a safety hold."""
    result = evaluate_options(
        (
            _estimate("a", "money", 40.0, 50.0, 60.0),
            _estimate("b", "money", 0.0, 10.0, 20.0),
        )
    )
    assert licence_for(result).claim_kind == CLAIM_COMPARATIVE
    held = licence_for(result, safety_hold=True)
    assert held.claim_kind == CLAIM_NONE
    assert held.subject_ref is None


def test_weights_cannot_clear_a_safety_hold() -> None:
    """Safety is above the ranking, not a dimension inside it.

    Driving every weight to zero — the arithmetic equivalent of a user
    deprioritising the safety dimension — must not produce a licensed
    claim.
    """
    estimates = (
        _estimate("a", "safety", 0.0, 1.0, 2.0),
        _estimate("a", "money", 40.0, 50.0, 60.0),
        _estimate("b", "safety", 40.0, 50.0, 60.0),
        _estimate("b", "money", 0.0, 1.0, 2.0),
    )
    zeroed = evaluate_options(estimates, weights=(("safety", 0.0), ("money", 1.0)))
    assert licence_for(zeroed, safety_hold=True).claim_kind == CLAIM_NONE


# ---------------------------------------------------------------------------
# Fourth-act acceptance sample
#
# Structure only. The point is which claims the arithmetic licenses, not
# whether the wording matches the script — the script itself contains the
# overclaim ("现在收益最高的是...") this module exists to prevent.
# ---------------------------------------------------------------------------


def _fourth_act():
    money_unknowns = ("equity-ownership", "company-valuation")
    return evaluate_options(
        (
            # Leave now: fast emotional relief, worst financial position,
            # and it forecloses the equity question entirely.
            _estimate("leave-now", "children", 0.0, 3.0, 5.0),
            _estimate("leave-now", "money", 0.0, 3.0, 6.0,
                      unresolved_refs=money_unknowns, unresolved_spread=3.0),
            _estimate("leave-now", "emotion", 2.0, 6.0, 8.0),
            # Negotiate first: deliberately given the BEST headline
            # numbers. If the reversible option also had the top
            # midpoint the assertions below would prove nothing — the
            # scenario has to make the two criteria disagree.
            _estimate("negotiate", "children", 0.0, 5.0, 7.0),
            _estimate("negotiate", "money", 0.0, 7.0, 11.0,
                      unresolved_refs=money_unknowns, unresolved_spread=3.0),
            _estimate("negotiate", "emotion", 0.0, 3.0, 6.0),
            # Separate for three months: middling on every axis.
            _estimate("separate", "children", 1.0, 4.0, 6.0),
            _estimate("separate", "money", 0.0, 4.0, 8.0,
                      unresolved_refs=money_unknowns, unresolved_spread=3.0),
            _estimate("separate", "emotion", 1.0, 5.0, 7.0),
            # Keep repairing: the user has already ruled this out
            # emotionally; it stays on the table but scores low.
            _estimate("repair", "children", 0.0, 4.0, 7.0),
            _estimate("repair", "money", 0.0, 5.0, 9.0,
                      unresolved_refs=money_unknowns, unresolved_spread=3.0),
            _estimate("repair", "emotion", 0.0, 1.0, 3.0),
        ),
        weights=(("children", 1.0), ("money", 1.0), ("emotion", 0.6)),
        reversibility=(
            ("leave-now", 0.1),
            ("negotiate", 0.5),
            ("separate", 0.95),
            ("repair", 0.7),
        ),
        resolves=(
            ("separate", money_unknowns),
            ("negotiate", ("equity-ownership",)),
        ),
    )


def test_fourth_act_licenses_robustness_not_a_winner() -> None:
    """No option separates, so no option "has the highest EV".

    This is the correction to the script's own wording, arrived at from
    the numbers rather than imposed on them.
    """
    licence = licence_for(_fourth_act())
    assert licence.claim_kind == CLAIM_ROBUSTNESS
    assert licence.may_state_a_winner is False


def test_fourth_act_favours_the_reversible_information_buying_option() -> None:
    """"Separate for three months" wins on optionality, not on score.

    It does not have the best midpoint. It is the most reversible option
    that also resolves the unknowns the ranking hangs on — which is a
    different, and defensible, claim.
    """
    result = _fourth_act()
    assert result.most_robust_ref == "separate"
    best_midpoint = max(result.options, key=lambda o: o.value.base).option_ref
    assert best_midpoint != "separate"


def test_fourth_act_keeps_the_equity_question_open_and_next() -> None:
    result = _fourth_act()
    licence = licence_for(result)
    assert "equity-ownership" in licence.must_surface_unknown_refs
    assert result.next_unknown_to_resolve() in {
        "equity-ownership",
        "company-valuation",
    }


def test_fourth_act_will_not_state_the_valuation_as_fact() -> None:
    """Nobody has read the documents, so no figure is licensed."""
    assert licence_for(_fourth_act()).permits("money") is False
