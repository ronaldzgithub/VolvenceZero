"""Regime contract dataclasses (identity, snapshot, bootstrap, payoff)."""

from __future__ import annotations

from dataclasses import dataclass

from volvence_zero.regime.hints import CognitiveDepthHint, ParticipationHint


@dataclass(frozen=True)
class RegimeIdentity:
    regime_id: str
    name: str
    embedding: tuple[float, ...]
    entry_conditions: str
    exit_conditions: str
    historical_effectiveness: float


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
