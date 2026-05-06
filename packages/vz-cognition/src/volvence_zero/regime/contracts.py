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

    Adding a new field requires updating the per-regime template
    table in :mod:`volvence_zero.regime.templates` and the spec at
    ``docs/specs/expression-layer.md``.
    """

    acknowledge_hint: str = "default"
    frame_hint: str = "default"
    next_step_hint: str = "default"
    open_loop_hint: str = "default"
    continuity_hint: str = "default"


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
