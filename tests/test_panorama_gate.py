"""Contract tests for the panorama participation gate (v2).

What these tests are actually defending:

* The gate reads **decision structure**, not attention posture. The v1
  score opens on world presence / switch pressure / task bias, which is
  why it expands a panorama into a library-hours question.
* All four decision-structure features are load-bearing. Blinding the
  gate to any one of them must produce a ceiling breach the corpus
  catches — otherwise the four-way conjunction is decoration.
* The four are not measuring the same thing. Real decision situations
  co-vary, so some correlation is expected; near-duplication is not.
* The asymmetry holds: opening when unwanted is worse than staying
  quiet, so entry is taxed, escalation is one tier per turn, and "never
  measured" resolves to SILENT rather than to a confident zero.
* v1 behaviour is untouched. The gate is opt-in and the rollback is a
  one-field flip.
"""

from __future__ import annotations

import dataclasses

import pytest

from volvence_zero.regime import ParticipationLevel
from volvence_zero.regime.decision_structure import (
    DecisionStructureSignals,
    derive_decision_structure_signals,
)
from volvence_zero.regime.hint_readout import (
    HintReadoutContext,
    panorama_structure_score,
    readout_panorama_level,
    readout_participation_hint,
)
from volvence_zero.regime.panorama_audit import (
    DECISION_FEATURES,
    ablation_report,
    audit_gate,
    build_v2_context,
    feature_correlations,
    v1_gate,
    v2_gate,
)
from volvence_zero.regime.panorama_corpus import (
    PANORAMA_CORPUS,
    build_snapshots,
)

# A near-duplicate pair would make the geometric mean a single axis
# raised to a power. 0.85 leaves room for the genuine co-variation of
# real decision situations while still failing on duplication.
_MAX_FEATURE_CORRELATION = 0.85


def _context(**overrides: object) -> HintReadoutContext:
    base = HintReadoutContext(
        has_dual_track=True,
        has_evaluation=True,
        has_decision_structure=True,
        decision_structure_slots=("plan_intent", "goal_value"),
    )
    return dataclasses.replace(base, **overrides)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------


def test_v2_gate_has_no_ceiling_breaches() -> None:
    """Opening a panorama into a turn that did not want one.

    Reported separately from floor misses on purpose: this is the
    failure the user does not report, they just find the system cold.
    """
    report = audit_gate(v2_gate, gate_name="v2")
    offenders = [case.case_id for case in report.cases if case.verdict == "over"]
    assert offenders == [], f"gate opened where it should not: {offenders}"


def test_v2_gate_has_no_floor_misses() -> None:
    report = audit_gate(v2_gate, gate_name="v2")
    offenders = [case.case_id for case in report.cases if case.verdict == "under"]
    assert offenders == [], f"gate stayed quiet where structure was due: {offenders}"


def test_corpus_keeps_negatives_and_boundaries_in_the_majority() -> None:
    """A corpus dominated by positives cannot catch an always-open gate."""
    positives = sum(1 for case in PANORAMA_CORPUS if case.family == "positive")
    assert positives * 2 <= len(PANORAMA_CORPUS)


def test_positive_cases_span_unrelated_topics() -> None:
    """Structurally identical, topically unrelated.

    If the gate only fires on one of them it has learned the topic.
    """
    topics = {
        case.topic for case in PANORAMA_CORPUS if case.family == "positive"
    }
    assert len(topics) >= 4


def test_v1_gate_still_fails_the_corpus() -> None:
    """The baseline this work exists to replace.

    Pinned so a future change to the v1 score cannot quietly erase the
    reason the v2 gate was written.
    """
    report = audit_gate(v1_gate, gate_name="v1")
    assert report.over_count > 0
    assert report.under_count > 0


# ---------------------------------------------------------------------------
# Adversarial probes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("feature", DECISION_FEATURES)
def test_every_feature_is_load_bearing(feature: str) -> None:
    """Blind the gate to one feature; the corpus must notice.

    A feature whose removal changes nothing is not a condition of the
    conjunction, it is a comment.
    """
    report = ablation_report(feature)
    assert report.over_count + report.under_count > 0, (
        f"pinning {feature} to 1.0 changed no verdict — the corpus does "
        f"not test it, so the four-way conjunction is unverified"
    )


def test_features_are_not_duplicates() -> None:
    for left, right, correlation in feature_correlations():
        assert abs(correlation) <= _MAX_FEATURE_CORRELATION, (
            f"{left} and {right} correlate at r={correlation:+.3f}; a "
            f"geometric mean over duplicated axes is not a conjunction"
        )


def test_ranking_instability_does_not_read_option_counts() -> None:
    """The specific collinearity that r=0.96 exposed.

    ``ranking_instability`` must measure a *moving* ordering, not an
    absent one — otherwise it is ``option_multiplicity`` again.
    """
    from companion_standard.semantic_state import PlanIntentSnapshot

    def plan(candidate_count: int) -> PlanIntentSnapshot:
        from volvence_zero.regime.panorama_corpus import _records

        return PlanIntentSnapshot(
            active_plan_id=None,
            active_goal="",
            active_step="",
            active_constraints=(),
            deferred_intents=(),
            standing_plans=(),
            candidate_plans=_records("candidate", candidate_count),
            completed_plan_refs=(),
            plan_revision_count=1,
            continuity_score=0.7,
            control_signal=0.0,
            description="",
        )

    few = derive_decision_structure_signals(plan_intent=plan(1))
    many = derive_decision_structure_signals(plan_intent=plan(5))
    assert few.ranking_instability == many.ranking_instability
    assert few.option_multiplicity < many.option_multiplicity


# ---------------------------------------------------------------------------
# Combination rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("zeroed", DECISION_FEATURES)
def test_any_zero_feature_closes_the_gate(zeroed: str) -> None:
    """All four necessary, none sufficient.

    A weighted sum would let three strong axes drag a zero fourth over
    the line — which is how a gate ends up structuring someone's grief.
    """
    features: dict[str, object] = {name: 1.0 for name in DECISION_FEATURES}
    features[zeroed] = 0.0
    context = _context(**features)
    assert panorama_structure_score(context) == 0.0
    level, _ = readout_panorama_level(context)
    assert level is ParticipationLevel.SILENT


def test_unmeasured_structure_is_not_a_confident_zero() -> None:
    """"Never measured" must not read as "measured and safe to expand"."""
    context = _context(has_decision_structure=False, decision_structure_slots=())
    level, rationale = readout_panorama_level(context)
    assert level is ParticipationLevel.SILENT
    assert "no-decision-structure-observed" in rationale


def test_derive_reports_which_owners_were_observed() -> None:
    empty = derive_decision_structure_signals()
    assert empty.observed_slots == ()
    assert empty.has_observation is False
    assert isinstance(empty, DecisionStructureSignals)


# ---------------------------------------------------------------------------
# Asymmetry, hysteresis, exit
# ---------------------------------------------------------------------------


def test_escalation_is_capped_to_one_tier_per_turn() -> None:
    """A fully formed panorama on turn one talks past the person."""
    strong = dict(
        option_multiplicity=1.0,
        irreversibility=1.0,
        ranking_instability=1.0,
        unknown_dominance=1.0,
    )
    first_turn, _ = readout_panorama_level(
        _context(previous_panorama_level=None, **strong)
    )
    assert first_turn is ParticipationLevel.BRIEF
    second_turn, _ = readout_panorama_level(
        _context(previous_panorama_level=ParticipationLevel.BRIEF, **strong)
    )
    assert second_turn is ParticipationLevel.STRUCTURED


def test_de_escalation_is_not_capped() -> None:
    """Backing off is always allowed immediately."""
    level, _ = readout_panorama_level(
        _context(
            previous_panorama_level=ParticipationLevel.STRUCTURED,
            option_multiplicity=0.0,
            irreversibility=0.0,
            ranking_instability=0.0,
            unknown_dominance=0.0,
        )
    )
    assert level is ParticipationLevel.SILENT


def test_open_panorama_is_held_against_flapping() -> None:
    """Same borderline signals: closed stays closed, open stays open."""
    borderline = dict(
        option_multiplicity=0.62,
        irreversibility=0.62,
        ranking_instability=0.62,
        unknown_dominance=0.62,
    )
    from_open, _ = readout_panorama_level(
        _context(previous_panorama_level=ParticipationLevel.STRUCTURED, **borderline)
    )
    from_closed, _ = readout_panorama_level(
        _context(previous_panorama_level=ParticipationLevel.SILENT, **borderline)
    )
    assert from_open is ParticipationLevel.STRUCTURED
    assert from_closed is ParticipationLevel.BRIEF


def test_option_collapse_forces_the_panorama_down() -> None:
    """The exit condition must beat the hold bias, not lose to it."""
    level, rationale = readout_panorama_level(
        _context(
            previous_panorama_level=ParticipationLevel.STRUCTURED,
            option_multiplicity=0.2,
            irreversibility=1.0,
            ranking_instability=1.0,
            unknown_dominance=1.0,
        )
    )
    assert level is ParticipationLevel.BRIEF
    assert "exit=options-collapsed" in rationale


def test_user_override_wins_outright() -> None:
    level, rationale = readout_panorama_level(
        _context(
            panorama_override=ParticipationLevel.SILENT,
            previous_panorama_level=ParticipationLevel.STRUCTURED,
            option_multiplicity=1.0,
            irreversibility=1.0,
            ranking_instability=1.0,
            unknown_dominance=1.0,
        )
    )
    assert level is ParticipationLevel.SILENT
    assert "override" in rationale


# ---------------------------------------------------------------------------
# Rollback surface
# ---------------------------------------------------------------------------


def test_v1_is_the_default_and_is_unchanged_by_the_new_features() -> None:
    """Decision-structure features must not leak into the v1 path."""
    for scenario in PANORAMA_CORPUS:
        snapshots = build_snapshots(scenario)
        with_structure = build_v2_context(scenario, snapshots)
        default = readout_participation_hint(with_structure)
        explicit_v1 = readout_participation_hint(with_structure, panorama_gate="v1")
        assert default.panorama_level is explicit_v1.panorama_level
        assert default.panorama_level is v1_gate(scenario, snapshots)


def test_gate_mode_is_validated() -> None:
    with pytest.raises(ValueError, match="panorama_gate"):
        readout_participation_hint(_context(), panorama_gate="v3")


def test_only_panorama_moves_between_modes() -> None:
    """method_level / task_level are deliberately untouched by v2."""
    for scenario in PANORAMA_CORPUS:
        context = build_v2_context(scenario, build_snapshots(scenario))
        one = readout_participation_hint(context, panorama_gate="v1")
        two = readout_participation_hint(context, panorama_gate="v2")
        assert one.method_level is two.method_level
        assert one.task_level is two.task_level
        assert one.flow_kind is two.flow_kind


def test_gate_never_reads_topic() -> None:
    """Same structure, different topic label -> identical decision."""
    for scenario in PANORAMA_CORPUS:
        renamed = dataclasses.replace(
            scenario, topic="a completely unrelated subject"
        )
        assert v2_gate(scenario, build_snapshots(scenario)) is v2_gate(
            renamed, build_snapshots(renamed)
        )
