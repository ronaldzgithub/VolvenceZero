"""Typed corpus for auditing the panorama participation gate.

The gate this corpus exercises is
``ParticipationHint.panorama_level`` — the single decision surface that
answers "should this turn expand a full decision panorama, or stay
out of the way?". See ``docs/specs/cognitive-regime.md`` and
``.cursor/plans/panorama-decision-workspace_7b41ce02.plan.md``.

Why a corpus and not a benchmark of transcripts:

* The gate must be **topic-independent**. A corpus keyed off
  structural signals (how many live mutually-exclusive options, how
  costly a wrong choice is, how unstable the user's own ranking is,
  how much the ranking hinges on information not yet in hand) can
  state the expectation without ever naming a topic. ``topic`` is
  carried only so the audit report is readable; it is never a
  feature.
* **Negative cases carry as much weight as positive ones.** A gate
  that opens everywhere scores perfectly on positives alone. The
  asymmetry matters too: failing to open costs the user one
  structured turn and is recoverable next turn; opening when it was
  not wanted is a relational harm the user usually will not report.
  So each scenario declares a ``expected_max`` ceiling as well as an
  ``expected_min`` floor, and the audit reports ceiling breaches
  (false positives) separately from floor misses (false negatives).

Each scenario is a flat set of typed knobs with conservative
defaults; a scenario overrides only the axes it is about. Builders
turn a scenario into the runtime snapshots the readout consumes, so
the same corpus can be replayed against any gate implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from companion_standard.semantic_state import (
    BeliefAssumptionSnapshot,
    BoundaryConsentSnapshot,
    CommitmentSnapshot,
    GoalValueSnapshot,
    OpenLoopSnapshot,
    PlanIntentSnapshot,
    SemanticRecord,
)

from volvence_zero.dual_track.core import DualTrackSnapshot, TrackState
from volvence_zero.evaluation import EvaluationScore, EvaluationSnapshot
from volvence_zero.memory.store import MemoryEntry, MemorySnapshot, Track
from volvence_zero.regime.identity import ParticipationLevel

# Ordering used to compare levels; SILENT < BRIEF < STRUCTURED.
_LEVEL_ORDER: dict[ParticipationLevel, int] = {
    ParticipationLevel.SILENT: 0,
    ParticipationLevel.BRIEF: 1,
    ParticipationLevel.STRUCTURED: 2,
}


def level_rank(level: ParticipationLevel) -> int:
    """Ordinal rank of a participation level (SILENT=0 .. STRUCTURED=2)."""
    return _LEVEL_ORDER[level]


@dataclass(frozen=True)
class PanoramaScenario:
    """One auditable situation, described structurally.

    ``expected_min`` / ``expected_max`` bound the acceptable gate
    output. Most scenarios pin an exact level by setting both to the
    same value; genuinely ambiguous situations leave a one-tier band
    so the corpus does not encode false precision.
    """

    case_id: str
    # "negative" (gate must stay quiet), "boundary" (ambiguous;
    # conservatism is the point), "positive" (gate must engage).
    family: str
    # Audit-readability label ONLY. Never read by any gate.
    topic: str
    expected_min: ParticipationLevel
    expected_max: ParticipationLevel
    note: str = ""

    # --- attention / controller profile (features the v1 gate reads) ---
    world_action_hint: str | None = None
    self_action_hint: str | None = None
    cross_tension: float = 0.2
    world_tension: float = 0.2
    self_tension: float = 0.2
    world_drive: float = 0.3
    self_drive: float = 0.3
    shared_drive: float = 0.3
    switch_pressure: float = 0.2
    warmth: float = 0.5
    support_presence: float = 0.5
    task_pressure: float = 0.5
    cross_track_stability: float = 0.6
    info_integration: float = 0.5
    world_memory_entries: int = 0
    self_memory_entries: int = 0
    regime_id: str = "guided_exploration"
    turns_in_current_regime: int = 3
    # What the gate published last turn. ``None`` means "first turn of
    # the session". The gate escalates at most one tier per turn, so a
    # scenario several turns into a hard conversation must say so —
    # otherwise the corpus would be asking the gate to jump straight to
    # a full panorama from a standing start, which it refuses by design.
    previous_panorama_level: ParticipationLevel | None = None

    # --- decision structure (features the v2 gate reads) ---
    # option_multiplicity sources
    candidate_option_count: int = 0
    deferred_option_count: int = 0
    tradeoff_count: int = 0
    # ranking_instability sources
    has_chosen_option: bool = True
    plan_revision_count: int = 0
    plan_continuity: float = 0.8
    value_conflict: float = 0.0
    goal_shift_pressure: float = 0.0
    decision_readiness: float = 0.8
    conflicted_goal_count: int = 0
    # unknown_dominance sources
    verification_need_count: int = 0
    belief_mean_confidence: float = 0.8
    unresolved_loop_count: int = 0
    closure_pressure: float = 0.2
    confirmation_debt_count: int = 0
    # irreversibility sources
    active_commitment_count: int = 0
    at_risk_commitment_count: int = 0
    trust_obligation_count: int = 0
    active_constraint_count: int = 0
    external_action_consent: str = "unknown"
    professional_scope_pressure: float = 0.0
    autonomy_risk: float = 0.0
    overreach_risk: float = 0.0

    # Semantic owners absent entirely (cold start / package default).
    semantic_owners_present: bool = True
    # Dual-track / evaluation absent entirely — the genuine cold boot the
    # readout's scaffold fallback exists for.
    runtime_signals_present: bool = True


# ---------------------------------------------------------------------------
# Snapshot builders
# ---------------------------------------------------------------------------


def _records(prefix: str, count: int, *, confidence: float = 0.7) -> tuple[SemanticRecord, ...]:
    return tuple(
        SemanticRecord(
            record_id=f"{prefix}-{index}",
            summary=f"{prefix} {index}",
            detail="",
            confidence=confidence,
            status="open",
            source_turn=index,
            evidence=f"corpus:{prefix}",
        )
        for index in range(count)
    )


def build_dual_track(scenario: PanoramaScenario) -> DualTrackSnapshot:
    world_code = (scenario.world_drive, scenario.shared_drive, scenario.switch_pressure)
    self_code = (scenario.self_drive, scenario.shared_drive, scenario.switch_pressure)
    return DualTrackSnapshot(
        world_track=TrackState(
            track=Track.WORLD,
            active_goals=(),
            recent_credits=(),
            controller_code=world_code,
            tension_level=scenario.world_tension,
            abstract_action_hint=scenario.world_action_hint,
        ),
        self_track=TrackState(
            track=Track.SELF,
            active_goals=(),
            recent_credits=(),
            controller_code=self_code,
            tension_level=scenario.self_tension,
            abstract_action_hint=scenario.self_action_hint,
        ),
        cross_track_tension=scenario.cross_tension,
        description="panorama-corpus",
    )


def build_evaluation(scenario: PanoramaScenario) -> EvaluationSnapshot:
    metrics = {
        "warmth": scenario.warmth,
        "support_presence": scenario.support_presence,
        "task_pressure": scenario.task_pressure,
        "cross_track_stability": scenario.cross_track_stability,
        "info_integration": scenario.info_integration,
    }
    return EvaluationSnapshot(
        turn_scores=tuple(
            EvaluationScore(
                family="corpus",
                metric_name=name,
                value=value,
                confidence=1.0,
                evidence="panorama-corpus",
            )
            for name, value in metrics.items()
        ),
        session_scores=(),
        alerts=(),
        description="panorama-corpus",
    )


def build_memory(scenario: PanoramaScenario) -> MemorySnapshot:
    entries: list[MemoryEntry] = []
    for index in range(scenario.world_memory_entries):
        entries.append(
            MemoryEntry(
                entry_id=f"w-{index}",
                content="",
                track=Track.WORLD,
                stratum="episodic",
                created_at_ms=0,
                last_accessed_ms=0,
                strength=0.5,
                tags=(),
            )
        )
    for index in range(scenario.self_memory_entries):
        entries.append(
            MemoryEntry(
                entry_id=f"s-{index}",
                content="",
                track=Track.SELF,
                stratum="episodic",
                created_at_ms=0,
                last_accessed_ms=0,
                strength=0.5,
                tags=(),
            )
        )
    return MemorySnapshot(
        transient_summary="",
        episodic_summary="",
        durable_summary="",
        retrieved_entries=tuple(entries),
        total_entries_by_stratum=(),
        pending_promotions=0,
        pending_decays=0,
        cms_state=None,
        description="panorama-corpus",
    )


def build_plan_intent(scenario: PanoramaScenario) -> PlanIntentSnapshot:
    return PlanIntentSnapshot(
        active_plan_id="plan-active" if scenario.has_chosen_option else None,
        active_goal="",
        active_step="",
        active_constraints=tuple(
            f"constraint-{index}" for index in range(scenario.active_constraint_count)
        ),
        deferred_intents=_records("deferred", scenario.deferred_option_count),
        standing_plans=(),
        candidate_plans=_records("candidate", scenario.candidate_option_count),
        completed_plan_refs=(),
        plan_revision_count=scenario.plan_revision_count,
        continuity_score=scenario.plan_continuity,
        control_signal=0.0,
        description="panorama-corpus",
    )


def build_goal_value(scenario: PanoramaScenario) -> GoalValueSnapshot:
    return GoalValueSnapshot(
        explicit_goals=(),
        value_priorities=(),
        tradeoff_notes=_records("tradeoff", scenario.tradeoff_count),
        active_goal_id="goal-active" if scenario.has_chosen_option else None,
        alignment_score=scenario.decision_readiness,
        control_signal=0.0,
        description="panorama-corpus",
        value_conflict=scenario.value_conflict,
        decision_readiness=scenario.decision_readiness,
        active_tradeoff_count=scenario.tradeoff_count,
        goal_shift_pressure=scenario.goal_shift_pressure,
        conflicted_goal_count=scenario.conflicted_goal_count,
    )


def build_open_loop(scenario: PanoramaScenario) -> OpenLoopSnapshot:
    return OpenLoopSnapshot(
        unresolved_loops=_records("loop", scenario.unresolved_loop_count),
        pending_confirmations=_records("confirm", scenario.confirmation_debt_count),
        closure_refs=(),
        highest_priority_loop_id=None,
        closure_pressure=scenario.closure_pressure,
        control_signal=0.0,
        description="panorama-corpus",
        confirmation_debt_count=scenario.confirmation_debt_count,
    )


def build_belief_assumption(scenario: PanoramaScenario) -> BeliefAssumptionSnapshot:
    return BeliefAssumptionSnapshot(
        beliefs=(),
        assumptions=(),
        verification_needs=_records("verify", scenario.verification_need_count),
        contradiction_refs=(),
        mean_confidence=scenario.belief_mean_confidence,
        control_signal=0.0,
        description="panorama-corpus",
    )


def build_commitment(scenario: PanoramaScenario) -> CommitmentSnapshot:
    return CommitmentSnapshot(
        active_commitments=_records("commit", scenario.active_commitment_count),
        honored_commitment_refs=(),
        at_risk_commitments=_records("at-risk", scenario.at_risk_commitment_count),
        trust_obligation_count=scenario.trust_obligation_count,
        continuity_score=0.6,
        control_signal=0.0,
        description="panorama-corpus",
    )


def build_boundary_consent(scenario: PanoramaScenario) -> BoundaryConsentSnapshot:
    return BoundaryConsentSnapshot(
        granted_consents=(),
        missing_consents=(),
        denied_boundaries=(),
        memory_consent="unknown",
        external_action_consent=scenario.external_action_consent,
        compliance_score=0.8,
        control_signal=0.0,
        description="panorama-corpus",
        autonomy_risk=scenario.autonomy_risk,
        professional_scope_pressure=scenario.professional_scope_pressure,
        overreach_risk=scenario.overreach_risk,
    )


# ---------------------------------------------------------------------------
# The corpus
#
# Naming: ``neg-*`` must stay quiet, ``bnd-*`` is the conservatism band,
# ``pos-*`` must engage. Positive cases deliberately span unrelated
# topics with the SAME structural signature — a gate that only fires on
# one of them has learned the topic, not the structure.
# ---------------------------------------------------------------------------

_NEGATIVE: tuple[PanoramaScenario, ...] = (
    PanoramaScenario(
        case_id="neg-casual-chat",
        family="negative",
        topic="weekend small talk",
        expected_min=ParticipationLevel.SILENT,
        expected_max=ParticipationLevel.SILENT,
        note="No live options at all; nothing to lay out.",
        regime_id="casual_social",
        self_action_hint="stabilize_controller",
        world_action_hint="stabilize_controller",
        cross_tension=0.1,
        world_tension=0.1,
        self_tension=0.1,
        world_drive=0.15,
        self_drive=0.2,
        switch_pressure=0.05,
        warmth=0.85,
        task_pressure=0.1,
        cross_track_stability=0.85,
        decision_readiness=0.9,
        plan_continuity=0.9,
    ),
    PanoramaScenario(
        case_id="neg-emotional-support",
        previous_panorama_level=ParticipationLevel.BRIEF,
        family="negative",
        topic="venting after a hard day",
        expected_min=ParticipationLevel.SILENT,
        expected_max=ParticipationLevel.BRIEF,
        note=(
            "Options exist but the instability is emotional, not "
            "informational: no unknown gates the choice. Structuring here "
            "is the signature failure mode — turning distress into a table."
        ),
        regime_id="emotional_support",
        self_action_hint="repair_controller",
        world_action_hint="repair_controller",
        cross_tension=0.55,
        self_tension=0.75,
        world_tension=0.3,
        world_drive=0.2,
        self_drive=0.6,
        warmth=0.7,
        support_presence=0.8,
        task_pressure=0.15,
        cross_track_stability=0.45,
        self_memory_entries=3,
        candidate_option_count=2,
        has_chosen_option=False,
        value_conflict=0.5,
        goal_shift_pressure=0.5,
        decision_readiness=0.35,
        # The decisive axis: nothing is waiting on information.
        verification_need_count=0,
        belief_mean_confidence=0.8,
        unresolved_loop_count=1,
        closure_pressure=0.2,
    ),
    PanoramaScenario(
        case_id="neg-single-path-execution",
        previous_panorama_level=ParticipationLevel.BRIEF,
        family="negative",
        topic="already-decided move, executing steps",
        expected_min=ParticipationLevel.SILENT,
        expected_max=ParticipationLevel.BRIEF,
        note="One live path and a stable ranking; the user wants execution.",
        regime_id="problem_solving",
        world_action_hint="task_controller",
        self_action_hint="task_controller",
        world_drive=0.75,
        task_pressure=0.8,
        world_memory_entries=3,
        switch_pressure=0.5,
        candidate_option_count=0,
        has_chosen_option=True,
        decision_readiness=0.85,
        value_conflict=0.05,
        active_commitment_count=2,
        trust_obligation_count=2,
        external_action_consent="granted",
    ),
    PanoramaScenario(
        case_id="neg-factual-lookup",
        family="negative",
        topic="what time does the library close",
        expected_min=ParticipationLevel.SILENT,
        expected_max=ParticipationLevel.SILENT,
        note="Information request, no choice at all.",
        regime_id="problem_solving",
        world_action_hint="task_controller",
        world_drive=0.5,
        task_pressure=0.6,
        world_memory_entries=2,
        switch_pressure=0.6,
        verification_need_count=1,
        belief_mean_confidence=0.6,
    ),
    PanoramaScenario(
        case_id="neg-cold-start",
        family="negative",
        topic="first turn, no owner state yet",
        expected_min=ParticipationLevel.SILENT,
        expected_max=ParticipationLevel.BRIEF,
        note="No decision structure observed yet; must not assume one.",
        regime_id="acquaintance_building",
        turns_in_current_regime=1,
        semantic_owners_present=False,
        runtime_signals_present=False,
    ),
)

_BOUNDARY: tuple[PanoramaScenario, ...] = (
    PanoramaScenario(
        case_id="bnd-restaurant-choice",
        previous_panorama_level=ParticipationLevel.BRIEF,
        family="boundary",
        topic="picking a restaurant tonight",
        expected_min=ParticipationLevel.SILENT,
        expected_max=ParticipationLevel.BRIEF,
        note=(
            "Genuinely multiple options and an unstable ranking, but the "
            "choice is cheap and reversible — the cost side of the gate "
            "has to carry this one."
        ),
        regime_id="guided_exploration",
        world_action_hint="exploration_controller",
        world_drive=0.5,
        task_pressure=0.5,
        world_memory_entries=2,
        candidate_option_count=4,
        tradeoff_count=2,
        has_chosen_option=False,
        value_conflict=0.45,
        goal_shift_pressure=0.4,
        decision_readiness=0.4,
        verification_need_count=2,
        belief_mean_confidence=0.55,
        unresolved_loop_count=2,
        closure_pressure=0.5,
        # Nothing binding: no obligations, no external action, no scope
        # pressure. Reversible by construction.
        active_commitment_count=0,
        trust_obligation_count=0,
        active_constraint_count=0,
        external_action_consent="unknown",
        professional_scope_pressure=0.0,
    ),
    PanoramaScenario(
        case_id="bnd-consequential-but-settled",
        previous_panorama_level=ParticipationLevel.BRIEF,
        family="boundary",
        topic="signing an offer the user has already chosen",
        expected_min=ParticipationLevel.SILENT,
        expected_max=ParticipationLevel.BRIEF,
        note=(
            "High stakes and irreversible, but the ranking is stable and "
            "nothing is waiting on information. Re-opening the panorama "
            "here reads as second-guessing the user."
        ),
        regime_id="problem_solving",
        world_action_hint="task_controller",
        world_drive=0.7,
        task_pressure=0.75,
        world_memory_entries=3,
        candidate_option_count=1,
        has_chosen_option=True,
        decision_readiness=0.85,
        value_conflict=0.1,
        verification_need_count=0,
        belief_mean_confidence=0.85,
        active_commitment_count=3,
        trust_obligation_count=3,
        active_constraint_count=2,
        external_action_consent="required",
        professional_scope_pressure=0.4,
    ),
    PanoramaScenario(
        case_id="bnd-early-distress-with-stakes",
        family="boundary",
        topic="first turn of a high-stakes crisis, still raw",
        expected_min=ParticipationLevel.SILENT,
        expected_max=ParticipationLevel.BRIEF,
        note=(
            "All four structural signals are heading up, but it is turn 1 "
            "and the user has not been met yet. Conservatism means BRIEF "
            "now and STRUCTURED once the ranking work actually starts."
        ),
        regime_id="emotional_support",
        turns_in_current_regime=1,
        self_action_hint="repair_controller",
        world_action_hint="repair_controller",
        cross_tension=0.7,
        self_tension=0.85,
        world_tension=0.5,
        self_drive=0.7,
        world_drive=0.35,
        support_presence=0.75,
        task_pressure=0.2,
        cross_track_stability=0.3,
        self_memory_entries=2,
        candidate_option_count=3,
        has_chosen_option=False,
        value_conflict=0.7,
        goal_shift_pressure=0.7,
        decision_readiness=0.2,
        verification_need_count=2,
        belief_mean_confidence=0.4,
        unresolved_loop_count=3,
        closure_pressure=0.6,
        active_commitment_count=1,
        trust_obligation_count=2,
        professional_scope_pressure=0.5,
        external_action_consent="required",
    ),
    # ------------------------------------------------------------------
    # Discriminating cases: each holds three axes high and drops exactly
    # one. Without these the ablation probe passes with a feature pinned
    # to 1.0 — i.e. the corpus would not notice the gate going blind to
    # it, and the four-way conjunction would be untested decoration.
    # ------------------------------------------------------------------
    PanoramaScenario(
        case_id="bnd-isolate-ranking-settled",
        family="boundary",
        topic="several heavy options, but a firm ordering",
        expected_min=ParticipationLevel.SILENT,
        expected_max=ParticipationLevel.BRIEF,
        note=(
            "Isolates ranking_instability. Options, stakes and unknowns "
            "are all high; the user's ordering is settled. Re-opening the "
            "ranking is not help, it is second-guessing — verify the "
            "unknowns instead."
        ),
        previous_panorama_level=ParticipationLevel.BRIEF,
        regime_id="problem_solving",
        turns_in_current_regime=4,
        world_action_hint="task_controller",
        world_drive=0.6,
        task_pressure=0.6,
        world_memory_entries=3,
        candidate_option_count=4,
        tradeoff_count=3,
        has_chosen_option=True,
        plan_revision_count=0,
        plan_continuity=0.85,
        value_conflict=0.10,
        goal_shift_pressure=0.10,
        decision_readiness=0.85,
        conflicted_goal_count=0,
        verification_need_count=3,
        belief_mean_confidence=0.35,
        unresolved_loop_count=3,
        closure_pressure=0.70,
        active_commitment_count=2,
        trust_obligation_count=3,
        active_constraint_count=3,
        external_action_consent="required",
        professional_scope_pressure=0.70,
    ),
    PanoramaScenario(
        case_id="bnd-isolate-unknown-fully-informed",
        family="boundary",
        topic="heavy unsettled choice where every fact is already known",
        expected_min=ParticipationLevel.SILENT,
        expected_max=ParticipationLevel.BRIEF,
        note=(
            "Isolates unknown_dominance. Options, stakes and instability "
            "are high but nothing is waiting on information, so a "
            "panorama can only restate what the user already holds."
        ),
        previous_panorama_level=ParticipationLevel.BRIEF,
        regime_id="guided_exploration",
        turns_in_current_regime=4,
        world_action_hint="exploration_controller",
        world_drive=0.55,
        task_pressure=0.55,
        world_memory_entries=3,
        candidate_option_count=4,
        tradeoff_count=3,
        has_chosen_option=False,
        plan_revision_count=2,
        plan_continuity=0.40,
        value_conflict=0.70,
        goal_shift_pressure=0.60,
        decision_readiness=0.30,
        conflicted_goal_count=2,
        verification_need_count=0,
        belief_mean_confidence=0.90,
        unresolved_loop_count=0,
        closure_pressure=0.10,
        confirmation_debt_count=0,
        active_commitment_count=2,
        trust_obligation_count=3,
        active_constraint_count=3,
        external_action_consent="required",
        professional_scope_pressure=0.70,
    ),
    PanoramaScenario(
        case_id="bnd-isolate-single-option",
        family="boundary",
        topic="one path, high stakes, plenty still unknown",
        expected_min=ParticipationLevel.SILENT,
        expected_max=ParticipationLevel.BRIEF,
        note=(
            "Isolates option_multiplicity. Stakes, instability and "
            "unknowns are all high but there is only one live path, so "
            "there is no panorama to lay out — the work is verification."
        ),
        previous_panorama_level=ParticipationLevel.BRIEF,
        regime_id="guided_exploration",
        turns_in_current_regime=4,
        world_action_hint="exploration_controller",
        world_drive=0.55,
        task_pressure=0.55,
        world_memory_entries=3,
        candidate_option_count=1,
        tradeoff_count=0,
        has_chosen_option=False,
        plan_revision_count=2,
        plan_continuity=0.40,
        value_conflict=0.70,
        goal_shift_pressure=0.60,
        decision_readiness=0.30,
        conflicted_goal_count=2,
        verification_need_count=4,
        belief_mean_confidence=0.30,
        unresolved_loop_count=4,
        closure_pressure=0.75,
        active_commitment_count=2,
        trust_obligation_count=3,
        active_constraint_count=3,
        external_action_consent="required",
        professional_scope_pressure=0.70,
    ),
)

_POSITIVE: tuple[PanoramaScenario, ...] = (
    PanoramaScenario(
        case_id="pos-separation-decision",
        previous_panorama_level=ParticipationLevel.BRIEF,
        family="positive",
        topic="whether and how to end a marriage",
        expected_min=ParticipationLevel.STRUCTURED,
        expected_max=ParticipationLevel.STRUCTURED,
        note="The fourth-act situation, several turns in.",
        regime_id="guided_exploration",
        turns_in_current_regime=5,
        world_action_hint="exploration_controller",
        self_action_hint="repair_controller",
        cross_tension=0.6,
        self_tension=0.6,
        world_tension=0.6,
        world_drive=0.6,
        self_drive=0.55,
        switch_pressure=0.4,
        task_pressure=0.6,
        support_presence=0.55,
        cross_track_stability=0.35,
        world_memory_entries=3,
        self_memory_entries=2,
        candidate_option_count=4,
        tradeoff_count=3,
        has_chosen_option=False,
        plan_revision_count=2,
        plan_continuity=0.35,
        value_conflict=0.75,
        goal_shift_pressure=0.65,
        decision_readiness=0.25,
        conflicted_goal_count=3,
        verification_need_count=4,
        belief_mean_confidence=0.3,
        unresolved_loop_count=4,
        closure_pressure=0.75,
        confirmation_debt_count=2,
        active_commitment_count=2,
        at_risk_commitment_count=1,
        trust_obligation_count=3,
        active_constraint_count=3,
        external_action_consent="required",
        professional_scope_pressure=0.8,
        autonomy_risk=0.5,
        overreach_risk=0.4,
    ),
    PanoramaScenario(
        case_id="pos-cofounder-split",
        previous_panorama_level=ParticipationLevel.BRIEF,
        family="positive",
        topic="dissolving a founding partnership",
        expected_min=ParticipationLevel.STRUCTURED,
        expected_max=ParticipationLevel.STRUCTURED,
        note="Structurally identical to the separation case, unrelated topic.",
        regime_id="problem_solving",
        turns_in_current_regime=4,
        world_action_hint="exploration_controller",
        self_action_hint="task_controller",
        cross_tension=0.5,
        world_tension=0.6,
        self_tension=0.4,
        world_drive=0.7,
        self_drive=0.4,
        switch_pressure=0.35,
        task_pressure=0.7,
        cross_track_stability=0.4,
        world_memory_entries=3,
        candidate_option_count=3,
        tradeoff_count=3,
        has_chosen_option=False,
        plan_revision_count=2,
        plan_continuity=0.4,
        value_conflict=0.7,
        goal_shift_pressure=0.6,
        decision_readiness=0.3,
        conflicted_goal_count=2,
        verification_need_count=3,
        belief_mean_confidence=0.35,
        unresolved_loop_count=3,
        closure_pressure=0.7,
        active_commitment_count=3,
        trust_obligation_count=4,
        active_constraint_count=2,
        external_action_consent="required",
        professional_scope_pressure=0.75,
        overreach_risk=0.35,
    ),
    PanoramaScenario(
        case_id="pos-treatment-choice",
        previous_panorama_level=ParticipationLevel.BRIEF,
        family="positive",
        topic="choosing between treatment paths",
        expected_min=ParticipationLevel.STRUCTURED,
        expected_max=ParticipationLevel.STRUCTURED,
        note="Same structure again; heavy on unknowns, low on drive.",
        regime_id="guided_exploration",
        turns_in_current_regime=4,
        world_action_hint="exploration_controller",
        self_action_hint="repair_controller",
        cross_tension=0.55,
        self_tension=0.65,
        world_tension=0.5,
        # Deliberately low world_drive / task_pressure: the v1 gate leans
        # on these, so this case separates "attention posture" from
        # "decision structure".
        world_drive=0.3,
        self_drive=0.45,
        switch_pressure=0.2,
        task_pressure=0.35,
        support_presence=0.65,
        cross_track_stability=0.4,
        world_memory_entries=2,
        self_memory_entries=2,
        candidate_option_count=3,
        tradeoff_count=3,
        has_chosen_option=False,
        plan_revision_count=1,
        plan_continuity=0.4,
        value_conflict=0.7,
        goal_shift_pressure=0.55,
        decision_readiness=0.3,
        conflicted_goal_count=2,
        verification_need_count=4,
        belief_mean_confidence=0.3,
        unresolved_loop_count=4,
        closure_pressure=0.7,
        confirmation_debt_count=1,
        active_commitment_count=1,
        trust_obligation_count=2,
        active_constraint_count=3,
        external_action_consent="required",
        professional_scope_pressure=0.85,
        autonomy_risk=0.45,
    ),
    PanoramaScenario(
        case_id="pos-relocation-with-dependents",
        previous_panorama_level=ParticipationLevel.BRIEF,
        family="positive",
        topic="moving countries with a family",
        expected_min=ParticipationLevel.STRUCTURED,
        expected_max=ParticipationLevel.STRUCTURED,
        note="Same structure, warm and low-tension delivery.",
        regime_id="guided_exploration",
        turns_in_current_regime=6,
        world_action_hint="exploration_controller",
        self_action_hint="exploration_controller",
        cross_tension=0.35,
        self_tension=0.35,
        world_tension=0.45,
        world_drive=0.5,
        self_drive=0.4,
        warmth=0.65,
        task_pressure=0.5,
        cross_track_stability=0.5,
        world_memory_entries=3,
        candidate_option_count=4,
        tradeoff_count=4,
        has_chosen_option=False,
        plan_revision_count=2,
        plan_continuity=0.45,
        value_conflict=0.65,
        goal_shift_pressure=0.5,
        decision_readiness=0.3,
        conflicted_goal_count=2,
        verification_need_count=3,
        belief_mean_confidence=0.4,
        unresolved_loop_count=3,
        closure_pressure=0.65,
        active_commitment_count=2,
        trust_obligation_count=3,
        active_constraint_count=4,
        external_action_consent="required",
        professional_scope_pressure=0.6,
    ),
)

PANORAMA_CORPUS: tuple[PanoramaScenario, ...] = _NEGATIVE + _BOUNDARY + _POSITIVE


def scenarios_by_family(family: str) -> tuple[PanoramaScenario, ...]:
    return tuple(case for case in PANORAMA_CORPUS if case.family == family)


@dataclass(frozen=True)
class ScenarioSnapshots:
    """Everything a gate implementation may read for one scenario."""

    dual_track: DualTrackSnapshot | None
    evaluation: EvaluationSnapshot | None
    memory: MemorySnapshot | None
    plan_intent: PlanIntentSnapshot | None
    goal_value: GoalValueSnapshot | None
    open_loop: OpenLoopSnapshot | None
    belief_assumption: BeliefAssumptionSnapshot | None
    commitment: CommitmentSnapshot | None
    boundary_consent: BoundaryConsentSnapshot | None
    candidates: tuple[tuple[str, float], ...] = field(default=())


def build_snapshots(scenario: PanoramaScenario) -> ScenarioSnapshots:
    """Materialise every snapshot a gate may consume for ``scenario``."""
    semantic = scenario.semantic_owners_present
    runtime = scenario.runtime_signals_present
    return ScenarioSnapshots(
        dual_track=build_dual_track(scenario) if runtime else None,
        evaluation=build_evaluation(scenario) if runtime else None,
        memory=build_memory(scenario) if runtime else None,
        plan_intent=build_plan_intent(scenario) if semantic else None,
        goal_value=build_goal_value(scenario) if semantic else None,
        open_loop=build_open_loop(scenario) if semantic else None,
        belief_assumption=build_belief_assumption(scenario) if semantic else None,
        commitment=build_commitment(scenario) if semantic else None,
        boundary_consent=build_boundary_consent(scenario) if semantic else None,
        candidates=((scenario.regime_id, 0.7), ("guided_exploration", 0.5)),
    )
