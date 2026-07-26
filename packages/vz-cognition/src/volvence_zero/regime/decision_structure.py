"""Decision-structure signals for the panorama participation gate.

The question this module answers is *not* "what is the user talking
about". It is:

    Does this turn have the shape of a decision that a panorama would
    actually help with?

Four signals, all derived from typed semantic-owner snapshots, none of
them a function of topic or wording:

``option_multiplicity``
    Are there at least two live, mutually exclusive paths? With one
    path there is no panorama to lay out.

``irreversibility``
    Would a wrong choice be costly to undo? Cheap reversible choices do
    not earn the interruption.

``ranking_instability``
    Is the user's own ordering missing, self-contradictory, or moving?
    Someone with a settled ranking wants execution, not a table.

``unknown_dominance``
    Is the ordering waiting on information not yet in hand? When every
    relevant fact is already known, a panorama only restates.

All four are treated as **necessary and none as sufficient** — see
``panorama_score`` in ``hint_readout``, which combines them with a
geometric mean so any one of them at zero closes the gate. That is the
whole content of the design: the emotional-support turn is held back
not by a special "if repair then be quiet" rule but because nothing is
waiting on information; the restaurant choice is held back because
nothing is costly to undo.

Red line: this module reads typed snapshot fields only. It never
inspects user text, and it must never grow a topic classifier — that
would be the scenario-enumeration approach the gate exists to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from companion_standard.semantic_state import (
        BeliefAssumptionSnapshot,
        BoundaryConsentSnapshot,
        CommitmentSnapshot,
        GoalValueSnapshot,
        OpenLoopSnapshot,
        PlanIntentSnapshot,
        UserModelSnapshot,
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _saturating(count: int, *, full_at: float) -> float:
    """Map a non-negative count onto [0,1], saturating at ``full_at``."""
    if count <= 0:
        return 0.0
    return _clamp01(count / full_at)


@dataclass(frozen=True)
class DecisionStructureSignals:
    """The four gate features plus provenance.

    ``observed_slots`` records which semantic owners actually supplied
    data. An empty tuple means the gate has *no* basis for a decision-
    structure claim — the caller must treat that as "stay quiet", not as
    "all features are zero, therefore confidently quiet".
    """

    option_multiplicity: float = 0.0
    irreversibility: float = 0.0
    ranking_instability: float = 0.0
    unknown_dominance: float = 0.0
    observed_slots: tuple[str, ...] = ()
    rationale: str = ""

    @property
    def has_observation(self) -> bool:
        return bool(self.observed_slots)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (
            self.option_multiplicity,
            self.irreversibility,
            self.ranking_instability,
            self.unknown_dominance,
        )


# ---------------------------------------------------------------------------
# Per-feature derivations
# ---------------------------------------------------------------------------


def _option_multiplicity(
    plan_intent: "PlanIntentSnapshot | None",
    goal_value: "GoalValueSnapshot | None",
) -> float:
    """How many live, mutually exclusive paths are on the table.

    Primary source is ``plan_intent``: ``candidate_plans`` are exactly
    "options under consideration" and ``deferred_intents`` are options
    kept alive but not chosen. ``goal_value.active_tradeoff_count``
    corroborates — a recorded trade-off implies at least two things
    being weighed — but it is weighted lower because a trade-off note
    is evidence of comparison, not an enumerated option.

    The map is deliberately steep between one and two: one path is not a
    decision.
    """
    if plan_intent is None and goal_value is None:
        return 0.0
    options = 0
    if plan_intent is not None:
        options = len(plan_intent.candidate_plans) + len(plan_intent.deferred_intents)
    # 0 -> 0.0, 1 -> 0.20, 2 -> 0.60, 3 -> 0.80, 4+ -> 1.0
    ladder = (0.0, 0.20, 0.60, 0.80, 1.0)
    direct = ladder[min(options, len(ladder) - 1)]
    corroboration = 0.0
    if goal_value is not None:
        corroboration = 0.75 * _saturating(goal_value.active_tradeoff_count, full_at=3.0)
    return _clamp01(max(direct, corroboration))


def _irreversibility(
    commitment: "CommitmentSnapshot | None",
    boundary_consent: "BoundaryConsentSnapshot | None",
    plan_intent: "PlanIntentSnapshot | None",
) -> float:
    """How costly a wrong choice would be to undo.

    There is no owner that measures "cost of undoing" directly, so this
    is an explicit **proxy: exposure beyond the conversation**. A choice
    that stays inside the dialogue is reversible by definition; a choice
    that creates obligations toward other people, requires acting on the
    outside world, or needs a professional in the loop is not.

    Deliberately NOT used: ``goal_value.reversibility_need``. Despite
    the name it is computed from ``value_conflict``, ``goal_shift_
    pressure`` and ``alignment_score`` — the same quantities that feed
    ``ranking_instability`` below. Reusing it would make two of the four
    "independent" features the same number under different names, and
    the geometric mean would silently square one axis instead of
    conjoining four. It measures how much the *user* wants to keep
    options open, not how costly the choice is to undo.
    """
    if commitment is None and boundary_consent is None and plan_intent is None:
        return 0.0
    obligation = 0.0
    if commitment is not None:
        obligation = max(
            _saturating(commitment.trust_obligation_count, full_at=3.0),
            _saturating(len(commitment.active_commitments), full_at=3.0),
            _saturating(len(commitment.at_risk_commitments), full_at=2.0),
        )
    external = 0.0
    scope = 0.0
    if boundary_consent is not None:
        # Consent state is a proxy for "this leaves the conversation".
        # ``required`` is the strongest: the action is consequential
        # enough that it may not proceed unasked.
        external = {
            "required": 0.85,
            "granted": 0.60,
            "denied": 0.40,
        }.get(boundary_consent.external_action_consent, 0.0)
        scope = max(
            _clamp01(boundary_consent.professional_scope_pressure),
            _clamp01(boundary_consent.autonomy_risk),
            _clamp01(boundary_consent.overreach_risk),
        )
    binding = 0.0
    if plan_intent is not None:
        binding = _saturating(len(plan_intent.active_constraints), full_at=4.0)
    return _clamp01(
        0.34 * obligation + 0.30 * external + 0.26 * scope + 0.10 * binding
    )


def _ranking_instability(
    plan_intent: "PlanIntentSnapshot | None",
    goal_value: "GoalValueSnapshot | None",
    user_model: "UserModelSnapshot | None",
) -> float:
    """How unsettled the user's own ordering is.

    Sources: ``goal_value``'s typed instability readouts
    (``value_conflict`` / ``goal_shift_pressure`` / the inverse of
    ``decision_readiness`` / ``conflicted_goal_count``) plus
    ``plan_intent``'s revision churn.

    Deliberately NOT used: "there are live options but none is chosen".
    That reads as the strongest instability signal available, and it was
    the first version of this function — but it is a function of the
    option count, which is what ``option_multiplicity`` already
    measures. Including it drove the two features to r=0.96 across the
    audit corpus, which turns the four-way geometric mean into one axis
    raised to a power wearing a conjunction's clothes. Instability here
    means the *ordering* is moving, not that an ordering is absent.
    """
    if plan_intent is None and goal_value is None and user_model is None:
        return 0.0
    churn = 0.0
    if plan_intent is not None:
        churn = max(
            _saturating(plan_intent.plan_revision_count, full_at=3.0),
            _clamp01(1.0 - plan_intent.continuity_score),
        )
    conflict = 0.0
    if goal_value is not None:
        conflict = max(
            _clamp01(goal_value.value_conflict),
            _clamp01(goal_value.goal_shift_pressure),
            _clamp01(1.0 - goal_value.decision_readiness),
            _saturating(goal_value.conflicted_goal_count, full_at=3.0),
        )
    overwhelm = 0.0
    if user_model is not None:
        overwhelm = _clamp01(user_model.overwhelm_pattern_strength)
    return _clamp01(0.62 * conflict + 0.30 * churn + 0.08 * overwhelm)


def _unknown_dominance(
    belief_assumption: "BeliefAssumptionSnapshot | None",
    open_loop: "OpenLoopSnapshot | None",
) -> float:
    """How much the ordering hinges on information not yet in hand.

    ``belief_assumption.verification_needs`` is the direct expression of
    "we do not know this yet and it matters"; low ``mean_confidence``
    says the same thing in aggregate. ``open_loop`` contributes the
    unresolved questions and the confirmation debt.

    This is the axis that separates a decision from distress. Someone
    whose ranking is unstable because they are overwhelmed has no
    pending verification; someone whose ranking is unstable because they
    do not know what the asset is worth does.
    """
    if belief_assumption is None and open_loop is None:
        return 0.0
    verification = 0.0
    if belief_assumption is not None:
        verification = max(
            _saturating(len(belief_assumption.verification_needs), full_at=3.0),
            _clamp01(1.0 - belief_assumption.mean_confidence),
        )
    pending = 0.0
    if open_loop is not None:
        pending = max(
            _saturating(len(open_loop.unresolved_loops), full_at=3.0),
            _clamp01(open_loop.closure_pressure),
            _saturating(open_loop.confirmation_debt_count, full_at=2.0),
        )
    return _clamp01(0.62 * verification + 0.38 * pending)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def derive_decision_structure_signals(
    *,
    plan_intent: "PlanIntentSnapshot | None" = None,
    goal_value: "GoalValueSnapshot | None" = None,
    open_loop: "OpenLoopSnapshot | None" = None,
    belief_assumption: "BeliefAssumptionSnapshot | None" = None,
    commitment: "CommitmentSnapshot | None" = None,
    boundary_consent: "BoundaryConsentSnapshot | None" = None,
    user_model: "UserModelSnapshot | None" = None,
) -> DecisionStructureSignals:
    """Derive the four gate features from semantic-owner snapshots.

    Every argument is optional; missing owners simply do not contribute.
    ``observed_slots`` records which ones were present so the caller can
    distinguish "measured as zero" from "not measured".
    """
    observed: list[str] = []
    for name, snapshot in (
        ("plan_intent", plan_intent),
        ("goal_value", goal_value),
        ("open_loop", open_loop),
        ("belief_assumption", belief_assumption),
        ("commitment", commitment),
        ("boundary_consent", boundary_consent),
        ("user_model", user_model),
    ):
        if snapshot is not None:
            observed.append(name)

    option = _option_multiplicity(plan_intent, goal_value)
    irreversible = _irreversibility(commitment, boundary_consent, plan_intent)
    instability = _ranking_instability(plan_intent, goal_value, user_model)
    unknown = _unknown_dominance(belief_assumption, open_loop)
    rationale = (
        f"options={option:.2f},irreversible={irreversible:.2f},"
        f"unstable={instability:.2f},unknown={unknown:.2f}"
    )
    return DecisionStructureSignals(
        option_multiplicity=option,
        irreversibility=irreversible,
        ranking_instability=instability,
        unknown_dominance=unknown,
        observed_slots=tuple(observed),
        rationale=rationale,
    )
