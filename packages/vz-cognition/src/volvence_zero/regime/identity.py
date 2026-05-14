"""RegimeModule and its public API.

This module keeps the runtime class itself. The contract dataclasses,
scaffold templates, scoring helpers, and hint derivations live in
sibling modules (``contracts``, ``templates``, ``scoring``, ``hints``).
``regime/__init__.py`` and external callers continue to import the
public names from this module via the re-exports below.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeKind,
    DialogueExternalOutcomeSnapshot,
)
from volvence_zero.dual_track import DualTrackSnapshot
from volvence_zero.evaluation import EvaluationSnapshot
from volvence_zero.memory import MemoryEntry, MemorySnapshot, Track
from volvence_zero.prediction.error import PredictionErrorSnapshot
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel

from volvence_zero.regime.contracts import (
    DelayedOutcomeAttribution,
    DelayedOutcomePayoff,
    PendingRegimeOutcome,
    RegimeBootstrap,
    RegimeCheckpoint,
    RegimeIdentity,
    RegimeSelectionWeights,
    RegimeSequencePayoff,
    RegimeSnapshot,
)
from volvence_zero.regime.hints import (
    CognitiveDepth,
    CognitiveDepthHint,
    ParticipationFlowKind,
    ParticipationHint,
    ParticipationLevel,
    derive_cognitive_depth_hint,
    derive_participation_hint,
)
from volvence_zero.regime.scoring import (
    _abstract_action_profile,
    _alert_pressure,
    _clamp,
    _controller_profile,
    _dominant_abstract_action_context,
    _metacontroller_action_profile,
    _track_counts,
    build_regime_identity,
    score_regimes,
)
from volvence_zero.regime.templates import REGIME_TEMPLATES

if TYPE_CHECKING:
    from volvence_zero.temporal.interface import MetacontrollerRuntimeState


# Per-kind score table for converting external outcome kinds into
# ``DelayedOutcomeAttribution.outcome_score`` values in ``[0, 1]``.
# 0.5 is neutral; positive outcomes pull toward 1.0, negative outcomes
# pull toward 0.0. Scaled by evidence confidence before being applied.
# This is a documented static mapping; learned scoring is post-v0.
_EXTERNAL_OUTCOME_REGIME_SCORE: dict[DialogueExternalOutcomeKind, float] = {
    DialogueExternalOutcomeKind.HELPED: 0.85,
    DialogueExternalOutcomeKind.FELT_HEARD: 0.85,
    DialogueExternalOutcomeKind.DECISION_CLEARER: 0.90,
    DialogueExternalOutcomeKind.COME_BACK: 0.40,
    DialogueExternalOutcomeKind.MISSED: 0.20,
    DialogueExternalOutcomeKind.OVER_DIRECTIVE: 0.15,
    DialogueExternalOutcomeKind.UNSAFE: 0.05,
    DialogueExternalOutcomeKind.ABANDONED: 0.05,
    # W3-A conversion outcomes: business-event evidence of regime
    # appropriateness. Purchase / repurchase score very high (the
    # current regime policy produced a confirmed downstream win);
    # CHURNED is comparable to ABANDONED (regime failed at long
    # horizon).
    DialogueExternalOutcomeKind.LEAD_QUALIFIED: 0.75,
    DialogueExternalOutcomeKind.RECOMMENDATION_MADE: 0.65,
    DialogueExternalOutcomeKind.PURCHASE_CONFIRMED: 0.92,
    DialogueExternalOutcomeKind.REPURCHASE: 0.95,
    DialogueExternalOutcomeKind.CHURNED: 0.05,
}


class RegimeModule(RuntimeModule[RegimeSnapshot]):
    slot_name = "regime"
    owner = "RegimeModule"
    value_type = RegimeSnapshot
    # ``dialogue_external_outcome`` (Rupture-and-Repair M2) is added so the
    # regime owner can build ``DelayedOutcomeAttribution`` rows from
    # externally-confirmed outcomes inside its own ``process``. No
    # external writer mutates ``_pending_outcomes``; the queue stays
    # single-writer (R8).
    dependencies = (
        "memory",
        "dual_track",
        "evaluation",
        "prediction_error",
        "experience_fast_prior",
        "dialogue_external_outcome",
    )
    default_wiring_level = WiringLevel.SHADOW

    def __init__(
        self,
        *,
        attribution_horizons: tuple[int, ...] = (2,),
        wiring_level: WiringLevel | None = None,
        bootstrap: "RegimeBootstrap | None" = None,
        hint_readout_mode: str = "readout",
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._attribution_horizons = tuple(min(max(h, 1), 8) for h in attribution_horizons) or (2,)
        # Gap 8 slice 2: which derivation to use for the
        # participation / depth hints.
        #
        # * ``readout`` (default) - continuous-feature readout over
        #   the dual_track / evaluation / PE / candidate signals
        #   via ``hint_readout.readout_*_hint``.
        # * ``scaffold`` - the slice-1 static ``dict[regime_id -> hint]``
        #   fallback. Kept available for rollback, for A/B comparison
        #   in the family report, and for tests that assert the
        #   pre-slice-2 behaviour.
        if hint_readout_mode not in {"readout", "scaffold"}:
            raise ValueError(
                f"RegimeModule.hint_readout_mode must be 'readout' or "
                f"'scaffold', got {hint_readout_mode!r}"
            )
        self._hint_readout_mode = hint_readout_mode
        # Historical defaults; overridden below if a bootstrap is provided.
        self._historical_effectiveness: dict[str, float] = {
            template.regime_id: 0.5 for template in REGIME_TEMPLATES
        }
        self._strategy_priors: dict[str, float] = {
            template.regime_id: 0.0 for template in REGIME_TEMPLATES
        }
        self._selection_weights: dict[str, float] = {
            template.regime_id: 1.0 for template in REGIME_TEMPLATES
        }
        self._feature_weights: dict[str, dict[str, float]] = {
            template.regime_id: {} for template in REGIME_TEMPLATES
        }
        self._selection_weight_lr = 0.02
        if bootstrap is not None:
            # Apply only the entries that name a known regime; unknown ids
            # are silently dropped so an artifact from a future build with
            # extra regimes does not blow up old runtimes.
            known_ids = {template.regime_id for template in REGIME_TEMPLATES}
            for regime_id, value in bootstrap.selection_weights:
                if regime_id in known_ids:
                    # Same clip range as the online learning rule.
                    # Selection weights are kept tight to ``[0.85, 1.15]`` so
                    # they act as a soft prior on top of the per-turn
                    # ``base_score``, not a winner-takes-all hard switch
                    # (Gap 9 phase 1.7 architectural fix). With the previous
                    # ``[0.3, 2.0]`` cap, ``base_score * weight`` clamped at
                    # 1.0 meant any weight >= ~1.5 caused that regime to
                    # dominate every prompt regardless of substrate signal,
                    # producing the monoculture observed in phase 1.5/1.6.
                    self._selection_weights[regime_id] = max(0.85, min(1.15, float(value)))
            for regime_id, value in bootstrap.historical_effectiveness:
                if regime_id in known_ids:
                    self._historical_effectiveness[regime_id] = max(0.0, min(1.0, float(value)))
            for regime_id, value in bootstrap.strategy_priors:
                if regime_id in known_ids:
                    self._strategy_priors[regime_id] = max(-0.5, min(0.5, float(value)))
            for regime_id, entries in getattr(bootstrap, "feature_weights", ()):
                if regime_id in known_ids:
                    self._feature_weights[regime_id] = {
                        feature_name: max(-0.75, min(0.75, float(value)))
                        for feature_name, value in entries
                    }
        self._active_regime_id: str | None = None
        self._previous_regime_id: str | None = None
        self._turns_in_current_regime = 0
        self._turn_index = 0
        self._pending_outcomes: list[PendingRegimeOutcome] = []
        self._last_delayed_outcomes: tuple[tuple[str, float], ...] = ()
        self._last_delayed_attributions: tuple[DelayedOutcomeAttribution, ...] = ()
        self._delayed_attribution_ledger: list[DelayedOutcomeAttribution] = []
        self._delayed_payoffs: dict[tuple[str, str | None, int], DelayedOutcomePayoff] = {}
        self._turn_evaluation_scores: list[float] = []
        self._regime_sequence: list[str] = []
        self._sequence_payoffs: dict[tuple[tuple[str, ...], int], RegimeSequencePayoff] = {}
        self._effectiveness_history: dict[str, list[float]] = {
            template.regime_id: [] for template in REGIME_TEMPLATES
        }

    async def process(self, upstream: Mapping[str, Snapshot[object]]) -> Snapshot[RegimeSnapshot]:
        from volvence_zero.application.runtime import ExperienceFastPriorSnapshot

        memory_snapshot = upstream["memory"]
        dual_track_snapshot = upstream["dual_track"]
        evaluation_snapshot = upstream["evaluation"]
        prediction_error_snapshot = upstream["prediction_error"]
        experience_fast_prior_snapshot = upstream["experience_fast_prior"]
        external_outcome_snapshot = upstream.get("dialogue_external_outcome")

        memory_value = memory_snapshot.value if isinstance(memory_snapshot.value, MemorySnapshot) else None
        dual_track_value = dual_track_snapshot.value if isinstance(dual_track_snapshot.value, DualTrackSnapshot) else None
        evaluation_value = (
            evaluation_snapshot.value if isinstance(evaluation_snapshot.value, EvaluationSnapshot) else None
        )
        pe_value = (
            prediction_error_snapshot.value
            if isinstance(prediction_error_snapshot.value, PredictionErrorSnapshot)
            else None
        )
        experience_fast_prior = (
            experience_fast_prior_snapshot.value
            if isinstance(experience_fast_prior_snapshot.value, ExperienceFastPriorSnapshot)
            else None
        )
        external_outcome_value: DialogueExternalOutcomeSnapshot | None = None
        if (
            external_outcome_snapshot is not None
            and isinstance(
                external_outcome_snapshot.value, DialogueExternalOutcomeSnapshot
            )
        ):
            external_outcome_value = external_outcome_snapshot.value
        experience_regime_biases = (
            {item.regime_id: item.bias for item in experience_fast_prior.regime_biases}
            if experience_fast_prior is not None
            else {}
        )

        self._turn_index += 1
        self._record_turn_score(evaluation_value, prediction_error_snapshot=pe_value)
        delayed_attributions_internal = self._apply_delayed_outcomes(evaluation_value)
        self._update_historical_effectiveness(evaluation_value, prediction_error_snapshot=pe_value)
        previous_active = self._active_regime_id
        candidates = score_regimes(
            memory_snapshot=memory_value,
            dual_track_snapshot=dual_track_value,
            evaluation_snapshot=evaluation_value,
            prediction_error_snapshot=pe_value,
            historical_effectiveness=self._historical_effectiveness,
            strategy_priors=self._strategy_priors,
            selection_weights=self._selection_weights,
            feature_weights=self._feature_weights,
            experience_regime_biases=experience_regime_biases,
        )
        chosen_regime_id = candidates[0][0]
        switch_reason = self._update_active_regime(chosen_regime_id=chosen_regime_id, candidates=candidates)
        regime_changed = self._active_regime_id != previous_active and previous_active is not None
        abstract_action, action_family_version = _dominant_abstract_action_context(dual_track_value)
        self._enqueue_pending_outcomes(
            regime_id=self._active_regime_id or chosen_regime_id,
            abstract_action=abstract_action,
            action_family_version=action_family_version,
        )
        # Ingest externally-confirmed outcomes *after* the active regime
        # has been chosen this turn, so attributions are attached to the
        # regime the user actually scored. This is the only path that
        # mutates _delayed_attribution_ledger / _delayed_payoffs from an
        # external source; the pending-outcome queue stays single-writer.
        external_attributions = self._ingest_external_outcome_attributions(
            external_outcome_snapshot=external_outcome_value,
            current_regime_id=self._active_regime_id or chosen_regime_id,
            abstract_action=abstract_action,
            action_family_version=action_family_version,
        )
        delayed_attributions = delayed_attributions_internal + external_attributions
        delayed_outcomes = tuple(
            (item.regime_id, item.outcome_score) for item in delayed_attributions
        )
        self._regime_sequence.append(self._active_regime_id or chosen_regime_id)
        identity_hints = self._identity_hints(memory_value)
        active_regime = build_regime_identity(
            regime_id=self._active_regime_id or chosen_regime_id,
            historical_effectiveness=self._historical_effectiveness,
        )
        previous_regime = (
            build_regime_identity(
                regime_id=self._previous_regime_id,
                historical_effectiveness=self._historical_effectiveness,
            )
            if self._previous_regime_id is not None
            else None
        )
        description = (
            f"Regime module active={active_regime.regime_id}, previous={self._previous_regime_id}, "
            f"turns_in_current_regime={self._turns_in_current_regime}, "
            f"delayed_outcomes={len(delayed_outcomes)}, identity_hints={len(identity_hints)}."
        )
        participation_hint, depth_hint = self._derive_hints(
            regime_id=active_regime.regime_id,
            candidates=candidates,
            memory=memory_value,
            dual_track=dual_track_value,
            evaluation=evaluation_value,
            prediction_error=pe_value,
        )
        return self.publish(
            RegimeSnapshot(
                active_regime=active_regime,
                previous_regime=previous_regime,
                switch_reason=switch_reason,
                candidate_regimes=candidates,
                turns_in_current_regime=self._turns_in_current_regime,
                delayed_outcomes=delayed_outcomes,
                delayed_attributions=delayed_attributions,
                delayed_attribution_ledger=tuple(self._delayed_attribution_ledger),
                delayed_payoffs=self._sorted_delayed_payoffs(),
                sequence_payoffs=self._sorted_sequence_payoffs(),
                identity_hints=identity_hints,
                effectiveness_trend=self._compute_effectiveness_trend(),
                regime_changed=regime_changed,
                selection_weights=RegimeSelectionWeights(
                    weights=tuple(sorted(self._selection_weights.items())),
                    learning_rate=self._selection_weight_lr,
                ),
                description=description,
                # Gap 8 slice 2: hint derivation routed through
                # ``_derive_hints`` (readout vs scaffold mode).
                participation_hint=participation_hint,
                depth_hint=depth_hint,
            )
        )

    async def process_standalone(self, **kwargs: object) -> Snapshot[RegimeSnapshot]:
        from volvence_zero.application.runtime import ExperienceFastPriorSnapshot

        memory_snapshot = kwargs.get("memory_snapshot")
        dual_track_snapshot = kwargs.get("dual_track_snapshot")
        evaluation_snapshot = kwargs.get("evaluation_snapshot")
        prediction_error_snapshot = kwargs.get("prediction_error_snapshot")
        experience_fast_prior_snapshot = kwargs.get("experience_fast_prior_snapshot")
        if not isinstance(memory_snapshot, MemorySnapshot):
            memory_snapshot = None
        if not isinstance(dual_track_snapshot, DualTrackSnapshot):
            dual_track_snapshot = None
        if not isinstance(evaluation_snapshot, EvaluationSnapshot):
            evaluation_snapshot = None
        if not isinstance(prediction_error_snapshot, PredictionErrorSnapshot):
            prediction_error_snapshot = None
        if not isinstance(experience_fast_prior_snapshot, ExperienceFastPriorSnapshot):
            experience_fast_prior_snapshot = None
        experience_regime_biases = (
            {item.regime_id: item.bias for item in experience_fast_prior_snapshot.regime_biases}
            if experience_fast_prior_snapshot is not None
            else {}
        )

        self._turn_index += 1
        self._record_turn_score(evaluation_snapshot, prediction_error_snapshot=prediction_error_snapshot)
        delayed_attributions = self._apply_delayed_outcomes(evaluation_snapshot)
        delayed_outcomes = tuple(
            (item.regime_id, item.outcome_score) for item in delayed_attributions
        )
        self._update_historical_effectiveness(evaluation_snapshot, prediction_error_snapshot=prediction_error_snapshot)
        previous_active = self._active_regime_id
        candidates = score_regimes(
            memory_snapshot=memory_snapshot,
            dual_track_snapshot=dual_track_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            prediction_error_snapshot=prediction_error_snapshot,
            historical_effectiveness=self._historical_effectiveness,
            strategy_priors=self._strategy_priors,
            selection_weights=self._selection_weights,
            feature_weights=self._feature_weights,
            experience_regime_biases=experience_regime_biases,
        )
        chosen_regime_id = candidates[0][0]
        switch_reason = self._update_active_regime(chosen_regime_id=chosen_regime_id, candidates=candidates)
        regime_changed = self._active_regime_id != previous_active and previous_active is not None
        abstract_action, action_family_version = _dominant_abstract_action_context(dual_track_snapshot)
        self._enqueue_pending_outcomes(
            regime_id=self._active_regime_id or chosen_regime_id,
            abstract_action=abstract_action,
            action_family_version=action_family_version,
        )
        self._regime_sequence.append(self._active_regime_id or chosen_regime_id)
        identity_hints = self._identity_hints(memory_snapshot)
        active_regime = build_regime_identity(
            regime_id=self._active_regime_id or chosen_regime_id,
            historical_effectiveness=self._historical_effectiveness,
        )
        previous_regime = (
            build_regime_identity(
                regime_id=self._previous_regime_id,
                historical_effectiveness=self._historical_effectiveness,
            )
            if self._previous_regime_id is not None
            else None
        )
        participation_hint, depth_hint = self._derive_hints(
            regime_id=active_regime.regime_id,
            candidates=candidates,
            memory=memory_snapshot,
            dual_track=dual_track_snapshot,
            evaluation=evaluation_snapshot,
            prediction_error=prediction_error_snapshot,
        )
        return self.publish(
            RegimeSnapshot(
                active_regime=active_regime,
                previous_regime=previous_regime,
                switch_reason=switch_reason,
                candidate_regimes=candidates,
                turns_in_current_regime=self._turns_in_current_regime,
                delayed_outcomes=delayed_outcomes,
                delayed_attributions=delayed_attributions,
                delayed_attribution_ledger=tuple(self._delayed_attribution_ledger),
                delayed_payoffs=self._sorted_delayed_payoffs(),
                sequence_payoffs=self._sorted_sequence_payoffs(),
                identity_hints=identity_hints,
                effectiveness_trend=self._compute_effectiveness_trend(),
                regime_changed=regime_changed,
                selection_weights=RegimeSelectionWeights(
                    weights=tuple(sorted(self._selection_weights.items())),
                    learning_rate=self._selection_weight_lr,
                ),
                description="Standalone regime snapshot.",
                # Gap 8 slice 2: routed through ``_derive_hints``.
                participation_hint=participation_hint,
                depth_hint=depth_hint,
            )
        )

    def _compute_effectiveness_trend(self) -> tuple[tuple[str, float], ...]:
        trends: list[tuple[str, float]] = []
        for regime_id, history in self._effectiveness_history.items():
            if len(history) < 2:
                trends.append((regime_id, 0.0))
                continue
            recent = history[-5:]
            if len(recent) < 2:
                trends.append((regime_id, 0.0))
                continue
            slope = (recent[-1] - recent[0]) / max(len(recent) - 1, 1)
            trends.append((regime_id, round(slope, 4)))
        return tuple(sorted(trends))

    def _update_historical_effectiveness(
        self,
        evaluation_snapshot: EvaluationSnapshot | None,
        *,
        prediction_error_snapshot: PredictionErrorSnapshot | None = None,
    ) -> None:
        # R-PE invariant: ``historical_effectiveness`` is a learning
        # signal that biases regime selection across turns; it must be
        # PE-derived. ``evaluation`` is gate / readout, never a learning
        # source. Without an actionable PE this turn, leave the
        # effectiveness EMA untouched.
        del evaluation_snapshot  # kept in signature for caller compatibility
        if self._active_regime_id is None:
            return
        if (
            prediction_error_snapshot is None
            or prediction_error_snapshot.bootstrap
        ):
            return
        blended = _clamp(0.5 + prediction_error_snapshot.error.signed_reward)
        current = self._historical_effectiveness[self._active_regime_id]
        self._historical_effectiveness[self._active_regime_id] = round(current * 0.7 + blended * 0.3, 4)
        self._effectiveness_history.setdefault(self._active_regime_id, []).append(
            self._historical_effectiveness[self._active_regime_id]
        )
        if len(self._effectiveness_history[self._active_regime_id]) > 20:
            self._effectiveness_history[self._active_regime_id] = (
                self._effectiveness_history[self._active_regime_id][-20:]
            )

    def _record_turn_score(
        self,
        evaluation_snapshot: EvaluationSnapshot | None,
        *,
        prediction_error_snapshot: PredictionErrorSnapshot | None = None,
    ) -> None:
        # R-PE invariant: turn-level reward signal must come from
        # ``prediction_error.signed_reward``; ``evaluation`` is gate /
        # readout only. PE absent or bootstrap => append neutral 0.5
        # (no learning signal this turn).
        del evaluation_snapshot  # kept in signature for caller compatibility
        if (
            prediction_error_snapshot is not None
            and not prediction_error_snapshot.bootstrap
        ):
            self._turn_evaluation_scores.append(
                _clamp(0.5 + prediction_error_snapshot.error.signed_reward)
            )
            return
        self._turn_evaluation_scores.append(0.5)

    def _nstep_blended_score(self, source_turn_index: int) -> float:
        gamma = 0.85
        horizon = self._turn_index - source_turn_index
        if horizon <= 0:
            return self._turn_evaluation_scores[-1] if self._turn_evaluation_scores else 0.5
        scores: list[float] = []
        weights: list[float] = []
        for step in range(horizon):
            idx = source_turn_index + step  # 0-based: turn (source+1+step) stored at index (source+step)
            if 0 <= idx < len(self._turn_evaluation_scores):
                scores.append(self._turn_evaluation_scores[idx])
                weights.append(gamma ** (horizon - 1 - step))
        if not scores:
            return self._turn_evaluation_scores[-1] if self._turn_evaluation_scores else 0.5
        return sum(s * w for s, w in zip(scores, weights)) / max(sum(weights), 1e-6)

    def _enqueue_pending_outcomes(
        self,
        *,
        regime_id: str,
        abstract_action: str | None,
        action_family_version: int,
    ) -> None:
        for horizon in self._attribution_horizons:
            self._pending_outcomes.append(
                PendingRegimeOutcome(
                    regime_id=regime_id,
                    source_turn_index=self._turn_index,
                    source_wave_id=f"wave-{self._turn_index}",
                    abstract_action=abstract_action,
                    action_family_version=action_family_version,
                    resolution_horizon_turns=horizon,
                )
            )

    def _ingest_external_outcome_attributions(
        self,
        *,
        external_outcome_snapshot: DialogueExternalOutcomeSnapshot | None,
        current_regime_id: str,
        abstract_action: str | None,
        action_family_version: int,
    ) -> tuple[DelayedOutcomeAttribution, ...]:
        """Build DelayedOutcomeAttribution rows from external outcome evidence.

        Owner-internal: this is the only path by which external outcome
        snapshot entries become regime attribution rows. The snapshot is
        the single legal channel; PE and Regime each decide how to
        consume it (R8).
        """

        if external_outcome_snapshot is None:
            return ()
        entries = external_outcome_snapshot.entries
        if not entries:
            return ()
        matured: list[DelayedOutcomeAttribution] = []
        for entry in entries:
            score = _EXTERNAL_OUTCOME_REGIME_SCORE.get(entry.kind)
            if score is None:
                continue
            # Confidence scales how far the score moves away from 0.5
            # (neutral). This keeps low-confidence entries from fully
            # replacing the internal trajectory.
            scale = max(0.0, min(1.0, float(entry.confidence)))
            resolved_score = 0.5 + (score - 0.5) * scale
            resolved_score = max(0.0, min(1.0, resolved_score))
            attribution = DelayedOutcomeAttribution(
                regime_id=current_regime_id,
                outcome_score=round(resolved_score, 4),
                source_turn_index=int(entry.turn_index),
                source_wave_id=f"external:{entry.evidence_id}",
                abstract_action=abstract_action,
                action_family_version=action_family_version,
                resolved_turn_index=self._turn_index,
            )
            matured.append(attribution)
            # Update internal effectiveness / priors / weights the same
            # way _apply_delayed_outcomes does, preserving audit parity.
            current = self._historical_effectiveness.get(current_regime_id, 0.5)
            self._historical_effectiveness[current_regime_id] = round(
                current * 0.8 + resolved_score * 0.2, 4
            )
            self._strategy_priors[current_regime_id] = _clamp(
                self._strategy_priors.get(current_regime_id, 0.0)
                + (resolved_score - 0.5) * 0.08
            )
            current_weight = self._selection_weights.get(current_regime_id, 1.0)
            advantage = resolved_score - 0.5
            self._selection_weights[current_regime_id] = max(
                0.85,
                min(
                    1.15,
                    current_weight
                    + self._selection_weight_lr * advantage * current_weight,
                ),
            )
            self._update_delayed_payoff(attribution)
            self._update_sequence_payoff(attribution)
        if not matured:
            return ()
        self._delayed_attribution_ledger.extend(matured)
        self._delayed_attribution_ledger = self._delayed_attribution_ledger[-24:]
        return tuple(matured)

    def _apply_delayed_outcomes(
        self,
        evaluation_snapshot: EvaluationSnapshot | None,
    ) -> tuple[DelayedOutcomeAttribution, ...]:
        if evaluation_snapshot is None or not self._pending_outcomes:
            self._last_delayed_outcomes = ()
            self._last_delayed_attributions = ()
            return ()
        matured: list[DelayedOutcomeAttribution] = []
        remaining: list[PendingRegimeOutcome] = []
        for pending in self._pending_outcomes:
            age = self._turn_index - pending.source_turn_index
            if age < pending.resolution_horizon_turns:
                remaining.append(pending)
                continue
            delayed_score = self._nstep_blended_score(pending.source_turn_index)
            current = self._historical_effectiveness.get(pending.regime_id, 0.5)
            self._historical_effectiveness[pending.regime_id] = round(current * 0.8 + delayed_score * 0.2, 4)
            self._strategy_priors[pending.regime_id] = _clamp(
                self._strategy_priors.get(pending.regime_id, 0.0) + (delayed_score - 0.5) * 0.08
            )
            current_weight = self._selection_weights.get(pending.regime_id, 1.0)
            advantage = delayed_score - 0.5
            # Same tightened cap as the bootstrap path above.
            self._selection_weights[pending.regime_id] = max(
                0.85,
                min(
                    1.15,
                    current_weight
                    + self._selection_weight_lr * advantage * current_weight,
                ),
            )
            matured.append(
                DelayedOutcomeAttribution(
                    regime_id=pending.regime_id,
                    outcome_score=round(delayed_score, 4),
                    source_turn_index=pending.source_turn_index,
                    source_wave_id=pending.source_wave_id,
                    abstract_action=pending.abstract_action,
                    action_family_version=pending.action_family_version,
                    resolved_turn_index=self._turn_index,
                )
            )
        self._pending_outcomes = remaining
        if not matured:
            self._last_delayed_outcomes = ()
            self._last_delayed_attributions = ()
            return ()
        self._delayed_attribution_ledger.extend(matured)
        self._delayed_attribution_ledger = self._delayed_attribution_ledger[-24:]
        for attribution in matured:
            self._update_delayed_payoff(attribution)
            self._update_sequence_payoff(attribution)
        self._last_delayed_outcomes = tuple(
            (item.regime_id, item.outcome_score) for item in matured
        )
        self._last_delayed_attributions = tuple(matured)
        return self._last_delayed_attributions

    def _update_delayed_payoff(self, attribution: DelayedOutcomeAttribution) -> None:
        key = (
            attribution.regime_id,
            attribution.abstract_action,
            attribution.action_family_version,
        )
        current = self._delayed_payoffs.get(key)
        if current is None:
            self._delayed_payoffs[key] = DelayedOutcomePayoff(
                regime_id=attribution.regime_id,
                abstract_action=attribution.abstract_action,
                action_family_version=attribution.action_family_version,
                sample_count=1,
                rolling_payoff=attribution.outcome_score,
                latest_outcome=attribution.outcome_score,
                last_source_wave_id=attribution.source_wave_id,
            )
            return
        sample_count = current.sample_count + 1
        rolling_payoff = round(
            current.rolling_payoff * 0.7 + attribution.outcome_score * 0.3,
            4,
        )
        self._delayed_payoffs[key] = DelayedOutcomePayoff(
            regime_id=current.regime_id,
            abstract_action=current.abstract_action,
            action_family_version=current.action_family_version,
            sample_count=sample_count,
            rolling_payoff=rolling_payoff,
            latest_outcome=attribution.outcome_score,
            last_source_wave_id=attribution.source_wave_id,
        )

    def _sorted_delayed_payoffs(self) -> tuple[DelayedOutcomePayoff, ...]:
        ranked = sorted(
            self._delayed_payoffs.values(),
            key=lambda payoff: (payoff.rolling_payoff, payoff.sample_count),
            reverse=True,
        )
        return tuple(ranked[:12])

    def _update_sequence_payoff(self, attribution: DelayedOutcomeAttribution) -> None:
        src_idx = attribution.source_turn_index - 1
        seq_start = max(0, src_idx - 1)
        regime_seq = tuple(self._regime_sequence[seq_start : src_idx + 1])
        if not regime_seq:
            regime_seq = (attribution.regime_id,)
        key = (regime_seq, attribution.action_family_version)
        current = self._sequence_payoffs.get(key)
        if current is None:
            self._sequence_payoffs[key] = RegimeSequencePayoff(
                regime_sequence=regime_seq,
                family_version=attribution.action_family_version,
                sample_count=1,
                rolling_payoff=attribution.outcome_score,
                latest_outcome=attribution.outcome_score,
                last_source_wave_id=attribution.source_wave_id,
            )
            return
        self._sequence_payoffs[key] = RegimeSequencePayoff(
            regime_sequence=regime_seq,
            family_version=current.family_version,
            sample_count=current.sample_count + 1,
            rolling_payoff=round(current.rolling_payoff * 0.7 + attribution.outcome_score * 0.3, 4),
            latest_outcome=attribution.outcome_score,
            last_source_wave_id=attribution.source_wave_id,
        )

    def _sorted_sequence_payoffs(self) -> tuple[RegimeSequencePayoff, ...]:
        ranked = sorted(
            self._sequence_payoffs.values(),
            key=lambda p: (p.rolling_payoff, p.sample_count),
            reverse=True,
        )
        return tuple(ranked[:12])

    def _identity_hints(self, memory_snapshot: MemorySnapshot | None) -> tuple[str, ...]:
        if memory_snapshot is None:
            return ()
        hints: list[str] = []
        for entry in memory_snapshot.retrieved_entries[:4]:
            if entry.track is Track.SELF:
                hints.append(f"identity:relationship:{entry.content}")
            elif entry.track is Track.SHARED and "user_input" in entry.tags:
                hints.append(f"identity:user:{entry.content}")
        return tuple(dict.fromkeys(hints))[:3]

    def _derive_hints(
        self,
        *,
        regime_id: str,
        candidates: tuple[tuple[str, float], ...],
        memory: "MemorySnapshot | None",
        dual_track: "DualTrackSnapshot | None",
        evaluation: "EvaluationSnapshot | None",
        prediction_error: "PredictionErrorSnapshot | None",
    ) -> tuple[ParticipationHint, CognitiveDepthHint]:
        """Produce the participation + depth hints for the current turn.

        Branches on ``self._hint_readout_mode``:

        * ``readout`` \u2014 Gap 8 slice 2 continuous-feature readout.
          When cold-start (no dual_track + no evaluation) the
          readout itself falls back to the scaffold internally and
          lowers its confidence.
        * ``scaffold`` \u2014 pre-slice-2 static regime_id lookup.
          Kept for rollback parity.
        """
        if self._hint_readout_mode == "scaffold":
            return (
                derive_participation_hint(regime_id),
                derive_cognitive_depth_hint(regime_id),
            )
        # Local import to avoid a top-level cycle: hint_readout
        # imports the hint dataclasses from ``regime.identity``.
        from volvence_zero.regime.hint_readout import (
            build_hint_readout_context,
            readout_cognitive_depth_hint,
            readout_participation_hint,
        )
        context = build_hint_readout_context(
            regime_id=regime_id,
            turns_in_current_regime=self._turns_in_current_regime,
            candidates=candidates,
            memory=memory,
            dual_track=dual_track,
            evaluation=evaluation,
            prediction_error=prediction_error,
        )
        return (
            readout_participation_hint(context),
            readout_cognitive_depth_hint(context),
        )

    def _update_active_regime(
        self,
        *,
        chosen_regime_id: str,
        candidates: tuple[tuple[str, float], ...],
    ) -> str:
        top_score = candidates[0][1]
        if self._active_regime_id is None:
            self._active_regime_id = chosen_regime_id
            self._turns_in_current_regime = 1
            return f"initial selection from candidate score {top_score:.2f}"
        if chosen_regime_id == self._active_regime_id:
            self._turns_in_current_regime += 1
            return f"hold current regime with candidate score {top_score:.2f}"

        self._previous_regime_id = self._active_regime_id
        self._active_regime_id = chosen_regime_id
        self._turns_in_current_regime = 1
        return f"switch to higher-scoring regime with candidate score {top_score:.2f}"

    def apply_policy_consolidation(
        self,
        *,
        strategy_updates: tuple[str, ...],
        regime_effectiveness_updates: tuple[tuple[str, float], ...],
        strategy_gain: float = 0.05,
        effectiveness_gain: float = 0.4,
    ) -> tuple[str, ...]:
        applied: list[str] = []
        for update in strategy_updates:
            if update == "increase_self_track_priority":
                self._strategy_priors["emotional_support"] = _clamp(
                    self._strategy_priors["emotional_support"] + strategy_gain * 1.35
                )
                self._strategy_priors["repair_and_deescalation"] = _clamp(
                    self._strategy_priors["repair_and_deescalation"] + strategy_gain * 0.95
                )
                self._strategy_priors["acquaintance_building"] = _clamp(
                    self._strategy_priors["acquaintance_building"] + strategy_gain * 0.7
                )
                applied.append("strategy-prior:self-track")
            elif update == "increase_world_track_priority":
                self._strategy_priors["problem_solving"] = _clamp(
                    self._strategy_priors["problem_solving"] + strategy_gain * 1.35
                )
                self._strategy_priors["guided_exploration"] = _clamp(
                    self._strategy_priors["guided_exploration"] + strategy_gain * 0.8
                )
                applied.append("strategy-prior:world-track")
        for regime_id, value in regime_effectiveness_updates:
            if regime_id not in self._historical_effectiveness:
                continue
            current = self._historical_effectiveness[regime_id]
            self._historical_effectiveness[regime_id] = round(
                current * (1.0 - effectiveness_gain) + value * effectiveness_gain,
                4,
            )
            applied.append(f"regime-effectiveness:{regime_id}")
        return tuple(applied)

    def apply_metacontroller_evidence(
        self,
        *,
        metacontroller_state: "MetacontrollerRuntimeState | None",
        rollback_reasons: tuple[str, ...],
    ) -> tuple[str, ...]:
        if metacontroller_state is None:
            return ()
        applied: list[str] = []
        world_bias, self_bias, shared_bias = _metacontroller_action_profile(metacontroller_state)
        if self_bias >= world_bias and self_bias >= shared_bias:
            for regime_id in ("repair_and_deescalation", "emotional_support"):
                self._strategy_priors[regime_id] = _clamp(self._strategy_priors[regime_id] + 0.04)
            applied.append("metacontroller:repair")
        elif world_bias >= self_bias and world_bias >= shared_bias:
            for regime_id in ("problem_solving", "guided_exploration"):
                self._strategy_priors[regime_id] = _clamp(self._strategy_priors[regime_id] + 0.04)
            applied.append("metacontroller:task")
        elif shared_bias >= max(world_bias, self_bias):
            for regime_id in ("guided_exploration", "acquaintance_building"):
                self._strategy_priors[regime_id] = _clamp(self._strategy_priors[regime_id] + 0.04)
            applied.append("metacontroller:exploration")
        else:
            self._strategy_priors["casual_social"] = _clamp(self._strategy_priors["casual_social"] + 0.02)
            applied.append("metacontroller:stabilize")
        if metacontroller_state.binary_switch_rate > 0.55:
            self._strategy_priors["guided_exploration"] = _clamp(
                self._strategy_priors["guided_exploration"] + 0.03
            )
            applied.append("metacontroller:sparse-switch")
        if metacontroller_state.posterior_drift > 0.45:
            self._strategy_priors["repair_and_deescalation"] = _clamp(
                self._strategy_priors["repair_and_deescalation"] + 0.03
            )
            applied.append("metacontroller:posterior-guard")
        if metacontroller_state.policy_replacement_score > 0.45:
            self._strategy_priors["problem_solving"] = _clamp(
                self._strategy_priors["problem_solving"] + 0.03
            )
            applied.append("metacontroller:replacement")
        if rollback_reasons:
            self._strategy_priors["repair_and_deescalation"] = _clamp(
                self._strategy_priors["repair_and_deescalation"] + 0.05
            )
            applied.append("metacontroller:guard")
        return tuple(applied)

    def create_checkpoint(self, *, checkpoint_id: str) -> RegimeCheckpoint:
        return RegimeCheckpoint(
            checkpoint_id=checkpoint_id,
            historical_effectiveness=tuple(sorted(self._historical_effectiveness.items())),
            strategy_priors=tuple(sorted(self._strategy_priors.items())),
            active_regime_id=self._active_regime_id,
            previous_regime_id=self._previous_regime_id,
            turns_in_current_regime=self._turns_in_current_regime,
            turn_index=self._turn_index,
            pending_outcomes=tuple(self._pending_outcomes),
            last_delayed_outcomes=self._last_delayed_outcomes,
            last_delayed_attributions=self._last_delayed_attributions,
            delayed_attribution_ledger=tuple(self._delayed_attribution_ledger),
            delayed_payoffs=self._sorted_delayed_payoffs(),
            turn_evaluation_scores=tuple(self._turn_evaluation_scores),
            regime_sequence=tuple(self._regime_sequence),
            sequence_payoffs=self._sorted_sequence_payoffs(),
            attribution_horizons=self._attribution_horizons,
        )

    def restore_checkpoint(self, checkpoint: RegimeCheckpoint) -> None:
        self._historical_effectiveness = dict(checkpoint.historical_effectiveness)
        self._strategy_priors = dict(checkpoint.strategy_priors)
        self._active_regime_id = checkpoint.active_regime_id
        self._previous_regime_id = checkpoint.previous_regime_id
        self._turns_in_current_regime = checkpoint.turns_in_current_regime
        self._turn_index = checkpoint.turn_index
        self._pending_outcomes = list(checkpoint.pending_outcomes)
        self._last_delayed_outcomes = checkpoint.last_delayed_outcomes
        self._last_delayed_attributions = checkpoint.last_delayed_attributions
        self._delayed_attribution_ledger = list(checkpoint.delayed_attribution_ledger)
        self._delayed_payoffs = {
            (item.regime_id, item.abstract_action, item.action_family_version): item
            for item in checkpoint.delayed_payoffs
        }
        self._turn_evaluation_scores = list(checkpoint.turn_evaluation_scores)
        self._regime_sequence = list(checkpoint.regime_sequence)
        self._sequence_payoffs = {
            (item.regime_sequence, item.family_version): item
            for item in checkpoint.sequence_payoffs
        }
        self._attribution_horizons = checkpoint.attribution_horizons

__all__ = [
    "CognitiveDepth",
    "CognitiveDepthHint",
    "DelayedOutcomeAttribution",
    "DelayedOutcomePayoff",
    "ParticipationFlowKind",
    "ParticipationHint",
    "ParticipationLevel",
    "PendingRegimeOutcome",
    "REGIME_TEMPLATES",
    "RegimeBootstrap",
    "RegimeCheckpoint",
    "RegimeIdentity",
    "RegimeModule",
    "RegimeSelectionWeights",
    "RegimeSequencePayoff",
    "RegimeSnapshot",
    "build_regime_identity",
    "derive_cognitive_depth_hint",
    "derive_participation_hint",
    "score_regimes",
]
