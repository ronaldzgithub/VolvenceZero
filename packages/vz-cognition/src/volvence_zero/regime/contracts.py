"""Regime contract dataclasses (identity, snapshot, bootstrap, payoff)."""

from __future__ import annotations

from dataclasses import dataclass, field

from volvence_zero.regime.hints import CognitiveDepthHint, ParticipationHint


@dataclass(frozen=True)
class ExpressionBrief:
    """Per-regime expression-intent placeholders.

    Owned by the regime module (W3 of ssot-cleanup-p0-p4). The
    response synthesizer reads these short semantic placeholders
    instead of branching on ``regime_id``. Each field describes
    "what this regime wants the corresponding section to convey",
    not the final UX text - the lifeform expression layer renders
    actual prose using the placeholder as a label.

    The five ``*_hint`` fields are short string codes consumed by
    the deterministic renderer in ``lifeform-expression`` as
    template lookup keys. ``llm_guidance`` is the prose form
    consumed by the LLM expression path (``vz-runtime/agent/prompts.py``)
    so the system prompt does not have to hardcode a parallel
    ``regime_id -> guidance`` dict outside the regime owner.
    Both forms are owned by the regime module so a new regime
    only needs one change in :mod:`volvence_zero.regime.templates`.

    Adding a new field requires updating the per-regime template
    table in :mod:`volvence_zero.regime.templates` and the spec at
    ``docs/specs/expression-layer.md``.
    """

    acknowledge_hint: str = "default"
    frame_hint: str = "default"
    next_step_hint: str = "default"
    open_loop_hint: str = "default"
    continuity_hint: str = "default"
    llm_guidance: str = ""


@dataclass(frozen=True)
class ApplicationBrief:
    """Per-regime application-facing semantic readout (W4 of ssot-cleanup-p0-p4).

    Owned by the regime module. ``vz-application`` consumes these
    fields instead of branching on ``regime_id`` strings. New
    regimes only need an ``ApplicationBrief`` populated in
    :mod:`volvence_zero.regime.templates`; ``application/runtime.py``
    and ``application/retrieval_readout.py`` automatically pick up
    the new behaviour.

    Field semantics:

    * ``task_focus`` / ``support_focus`` / ``repair_focus`` /
      ``exploration_focus``: 0..1 semantic mode strengths. These
      replace one-hot ``regime_id == 'X'`` features in retrieval
      scoring. Multiple foci can co-exist.
    * ``domain_affinity``: per-knowledge-domain additive bonus.
      Replaces ``_regime_bonus(regime_id, {dom_a: x, dom_b: y, ...})``
      lookups: each regime publishes its own affinity vector. Use
      :meth:`domain_bonus` to query a single domain.
    * ``knowledge_weight_nudge``: additive nudge to the
      knowledge-vs-experience weight. Positive = pull toward
      knowledge, negative = pull toward lived experience.
    * ``continuum_target_position``: regime's preferred response
      continuum target (0 = task-first, 1 = support-first).
    * ``decision_kind_hint`` / ``ordering_driver_hint``: structured
      defaults consumed by ``_decision_kind`` /
      ``_response_ordering_plan``.
    * ``support_decision_threshold``: support-before-decision gate
      lowered for exploratory regimes.
    """

    task_focus: float = 0.0
    support_focus: float = 0.0
    repair_focus: float = 0.0
    exploration_focus: float = 0.0
    domain_affinity: tuple[tuple[str, float], ...] = ()
    knowledge_weight_nudge: float = 0.0
    continuum_target_position: float = 0.5
    decision_kind_hint: str = "default"
    ordering_driver_hint: str = "playbook-only"
    support_decision_threshold: float = 0.44

    def domain_bonus(self, domain: str) -> float:
        """Return the additive bonus this regime contributes for ``domain``.

        Returns ``0.0`` when the domain is not in ``domain_affinity``.
        Replaces ``_regime_bonus(regime_id, {domain: bonus})`` lookups
        in ``vz-application``.
        """

        for d, value in self.domain_affinity:
            if d == domain:
                return value
        return 0.0


@dataclass(frozen=True)
class DomainHintCatalog:
    """Reviewer-curated per-knowledge-domain hint data (#81).

    Replaces the ``if domain == "X": return "..."`` chains that used
    to live in ``vz-application/runtime_helpers.py``: the curated hint
    summary and topic tags are owner-published typed data next to
    :class:`ApplicationBrief` (whose ``domain_affinity`` already owns
    the closed knowledge-domain vocabulary), not branch logic in a
    consumer. Adding a new reviewed domain means adding one catalog
    entry; a contract test enforces summary/topic-tag consistency so
    a reviewer cannot update one side and forget the other.

    ``language`` keys the catalog for future i18n: a second-language
    rollout registers another catalog instance instead of doubling
    every hardcoded branch.
    """

    language: str = "en"
    summary_per_domain: tuple[tuple[str, str], ...] = ()
    topic_tags_per_domain: tuple[tuple[str, tuple[str, ...]], ...] = ()
    default_summary: str = ""
    default_topic_tags: tuple[str, ...] = ("general",)

    def summary_for(self, domain: str) -> str:
        for d, summary in self.summary_per_domain:
            if d == domain:
                return summary
        return self.default_summary

    def topic_tags_for(self, domain: str) -> tuple[str, ...]:
        for d, tags in self.topic_tags_per_domain:
            if d == domain:
                return tags
        return self.default_topic_tags


# Curated English catalog (SSOT for domain hint readouts). Text is
# byte-identical to the pre-#81 ``runtime_helpers`` branches so the
# migration is a pure ownership move, not a copy change.
DEFAULT_DOMAIN_HINT_CATALOG = DomainHintCatalog(
    language="en",
    summary_per_domain=(
        (
            "family_transition",
            "Separate emotional stabilization from legal or procedural next steps, and keep any "
            "child-safety or jurisdiction-sensitive guidance explicitly bounded.",
        ),
        (
            "professional_process",
            "Use sourced high-level process guidance first, and avoid definitive professional conclusions "
            "before local specifics are confirmed.",
        ),
        (
            "career_decision",
            "Frame trade-offs explicitly, reduce ambiguity, and prefer the smallest next step over a full "
            "life-plan answer.",
        ),
        (
            "structured_decision_support",
            "Prefer option framing, trade-off comparison, and one grounded next action instead of broad "
            "multi-branch advice.",
        ),
        (
            "relational_repair",
            "Prioritize de-escalation, acknowledgement, and safety before moving into explanation or planning.",
        ),
        (
            "emotional_support_basics",
            "Acknowledge the felt experience first, then add structure gradually so the response does not "
            "skip past distress.",
        ),
    ),
    topic_tags_per_domain=(
        ("family_transition", ("family", "transition", "procedure")),
        ("professional_process", ("professional", "process", "bounded-advice")),
        ("career_decision", ("career", "tradeoff", "next-step")),
        ("structured_decision_support", ("decision", "structure", "options")),
        ("relational_repair", ("repair", "de-escalation", "safety")),
        ("emotional_support_basics", ("support", "stabilization", "presence")),
        ("general_support_guidance", ("support", "boundedness")),
    ),
    default_summary=(
        "Keep the response grounded, bounded, and shaped by the current regime rather than defaulting to a "
        "generic information dump."
    ),
    default_topic_tags=("general",),
)


@dataclass(frozen=True)
class RegimeIdentity:
    regime_id: str
    name: str
    embedding: tuple[float, ...]
    entry_conditions: str
    exit_conditions: str
    historical_effectiveness: float
    expression_brief: ExpressionBrief = field(default_factory=ExpressionBrief)
    application_brief: ApplicationBrief = field(default_factory=ApplicationBrief)


@dataclass(frozen=True)
class RegimeSelectionWeights:
    weights: tuple[tuple[str, float], ...]
    learning_rate: float = 0.02


@dataclass(frozen=True)
class RegimeLearnedScoreShadow:
    """Report-only learned regime scorer candidate.

    The live regime choice remains ``candidate_regimes``. This readout
    lets evidence compare the learned candidate against the hand-crafted
    scorer without changing behavior.
    """

    learned_scores: tuple[tuple[str, float], ...]
    baseline_scores: tuple[tuple[str, float], ...]
    update_count: int
    running_abs_error: float
    last_target_regime_id: str
    ready: bool
    kill_recommended: bool
    blocking_reasons: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class RegimeBootstrap:
    """Pre-trained regime classifier state.

    Lets a fresh ``RegimeModule`` start from learned ``selection_weights``,
    ``historical_effectiveness``, and ``strategy_priors`` instead of the
    flat 1.0 / 0.5 / 0.0 defaults. Mirrors ``MetacontrollerParameterSnapshot``
    on the temporal axis: collected via the lifeform-evolution regime
    calibrator, persisted via ``lifeform_evolution.regime_io``, injected at
    ``Brain`` / ``Lifeform`` construction time.

    Every field is optional; missing fields fall back to the historical
    defaults so an old artifact stays loadable when the schema grows.
    """

    selection_weights: tuple[tuple[str, float], ...] = ()
    historical_effectiveness: tuple[tuple[str, float], ...] = ()
    strategy_priors: tuple[tuple[str, float], ...] = ()
    feature_weights: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()
    description: str = ""


@dataclass(frozen=True)
class RegimeSnapshot:
    active_regime: RegimeIdentity
    previous_regime: RegimeIdentity | None
    switch_reason: str
    candidate_regimes: tuple[tuple[str, float], ...]
    turns_in_current_regime: int
    description: str
    delayed_outcomes: tuple[tuple[str, float], ...] = ()
    delayed_attributions: tuple["DelayedOutcomeAttribution", ...] = ()
    delayed_attribution_ledger: tuple["DelayedOutcomeAttribution", ...] = ()
    delayed_payoffs: tuple["DelayedOutcomePayoff", ...] = ()
    sequence_payoffs: tuple["RegimeSequencePayoff", ...] = ()
    identity_hints: tuple[str, ...] = ()
    effectiveness_trend: tuple[tuple[str, float], ...] = ()
    regime_changed: bool = False
    selection_weights: RegimeSelectionWeights | None = None
    learned_score_shadow: RegimeLearnedScoreShadow | None = None
    # Gap 8: participation + cognitive-depth hints. Defaults are the
    # "neutral / fully-structured / focused" baseline so pre-Gap-8
    # consumers keep seeing the same behaviour when they ignore the
    # hints. ``RegimeModule._derive_hints_from_regime`` overrides
    # these per-regime with a scaffold derivation table; a future
    # learned metacontroller readout will replace that scaffold.
    participation_hint: ParticipationHint = ParticipationHint()
    depth_hint: CognitiveDepthHint = CognitiveDepthHint()


@dataclass(frozen=True)
class RegimeCheckpoint:
    checkpoint_id: str
    historical_effectiveness: tuple[tuple[str, float], ...]
    strategy_priors: tuple[tuple[str, float], ...]
    active_regime_id: str | None
    previous_regime_id: str | None
    turns_in_current_regime: int
    turn_index: int = 0
    pending_outcomes: tuple["PendingRegimeOutcome", ...] = ()
    last_delayed_outcomes: tuple[tuple[str, float], ...] = ()
    last_delayed_attributions: tuple["DelayedOutcomeAttribution", ...] = ()
    delayed_attribution_ledger: tuple["DelayedOutcomeAttribution", ...] = ()
    delayed_payoffs: tuple["DelayedOutcomePayoff", ...] = ()
    turn_evaluation_scores: tuple[float, ...] = ()
    regime_sequence: tuple[str, ...] = ()
    sequence_payoffs: tuple["RegimeSequencePayoff", ...] = ()
    attribution_horizons: tuple[int, ...] = (2,)
    selection_weights: tuple[tuple[str, float], ...] = ()
    feature_weights: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()
    external_outcome_scores: tuple[tuple[str, float], ...] = ()
    learned_score_weights: tuple[tuple[str, tuple[float, ...]], ...] = ()
    learned_score_update_count: int = 0
    learned_score_abs_error_sum: float = 0.0
    learned_score_baseline_abs_error_sum: float = 0.0
    learned_score_settled_count: int = 0
    learned_score_last_target_regime_id: str = ""


@dataclass(frozen=True)
class PendingRegimeOutcome:
    regime_id: str
    source_turn_index: int
    source_wave_id: str
    abstract_action: str | None = None
    action_family_version: int = 0
    resolution_horizon_turns: int = 2


@dataclass(frozen=True)
class DelayedOutcomeAttribution:
    regime_id: str
    outcome_score: float
    source_turn_index: int
    source_wave_id: str
    abstract_action: str | None = None
    action_family_version: int = 0
    resolved_turn_index: int = 0


@dataclass(frozen=True)
class DelayedOutcomePayoff:
    regime_id: str
    abstract_action: str | None
    action_family_version: int
    sample_count: int
    rolling_payoff: float
    latest_outcome: float
    last_source_wave_id: str


@dataclass(frozen=True)
class RegimeSequencePayoff:
    regime_sequence: tuple[str, ...]
    family_version: int
    sample_count: int
    rolling_payoff: float
    latest_outcome: float
    last_source_wave_id: str
