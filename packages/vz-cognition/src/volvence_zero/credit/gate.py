from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Mapping
from uuid import uuid4

from volvence_zero.dual_track import DualTrackSnapshot
from volvence_zero.evaluation.types import EvaluationSnapshot
from volvence_zero.memory import Track
from volvence_zero.prediction.error import PredictionActionContext, PredictionError, PredictionErrorSnapshot
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.social_cognition import SocialPredictionError, SocialPredictionOutcome
from volvence_zero.temporal_types import TemporalAbstractionSnapshot

if TYPE_CHECKING:
    from volvence_zero.temporal.interface import MetacontrollerRuntimeState


class ModificationGate(str, Enum):
    ONLINE = "online"
    BACKGROUND = "background"
    OFFLINE = "offline"
    HUMAN_REVIEW = "human-review"


@dataclass(frozen=True)
class CreditRecord:
    record_id: str
    level: str
    track: Track
    source_event: str
    credit_value: float
    context: str
    timestamp_ms: int


@dataclass(frozen=True)
class SelfModificationRecord:
    target: str
    gate: ModificationGate
    decision: GateDecision
    old_value_hash: str
    new_value_hash: str
    justification: str
    timestamp_ms: int
    is_reversible: bool
    checkpoint_id: str = ""
    lineage_hash: str = ""
    proposal_hash: str = ""


@dataclass(frozen=True)
class NStepAttributionEntry:
    action_id: str
    family_id: str
    regime_id: str
    timestamp_ms: int
    outcome_history: tuple[float, ...]


@dataclass(frozen=True)
class CreditSnapshot:
    recent_credits: tuple[CreditRecord, ...]
    recent_modifications: tuple[SelfModificationRecord, ...]
    cumulative_credit_by_level: tuple[tuple[str, float], ...]
    session_level_credits: tuple[tuple[str, float], ...] = ()
    discount_factor: float = 0.95
    delayed_ledger_size: int = 0
    horizon_depth: int = 1
    description: str = ""


@dataclass(frozen=True)
class ModificationProposal:
    target: str
    desired_gate: ModificationGate
    old_value_hash: str
    new_value_hash: str
    justification: str
    is_reversible: bool = True
    validation_delta: float = 0.0
    capacity_cost: float = 0.0
    rollback_evidence: str = ""


@dataclass(frozen=True)
class RuntimeAdaptationAudit:
    target: str
    gate: ModificationGate
    decision: GateDecision
    old_value_hash: str
    new_value_hash: str
    justification: str
    timestamp_ms: int
    is_reversible: bool = True


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


class GateDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"


def derive_credit_records(
    *,
    dual_track_snapshot: DualTrackSnapshot,
    evaluation_snapshot: EvaluationSnapshot,
    timestamp_ms: int,
) -> tuple[CreditRecord, ...]:
    records: list[CreditRecord] = []
    for score in evaluation_snapshot.turn_scores:
        if score.family == "task":
            records.append(
                CreditRecord(
                    record_id=str(uuid4()),
                    level="turn",
                    track=Track.WORLD,
                    source_event=score.metric_name,
                    credit_value=_clamp(score.value),
                    context=score.evidence,
                    timestamp_ms=timestamp_ms,
                )
            )
        elif score.family in {"interaction", "relationship"}:
            records.append(
                CreditRecord(
                    record_id=str(uuid4()),
                    level="turn",
                    track=Track.SELF,
                    source_event=score.metric_name,
                    credit_value=_clamp(score.value),
                    context=score.evidence,
                    timestamp_ms=timestamp_ms,
                )
            )
        elif score.family == "safety":
            records.append(
                CreditRecord(
                    record_id=str(uuid4()),
                    level="turn",
                    track=Track.SHARED,
                    source_event=score.metric_name,
                    credit_value=_clamp(score.value),
                    context=score.evidence,
                    timestamp_ms=timestamp_ms,
                )
            )

    records.append(
        CreditRecord(
            record_id=str(uuid4()),
            level="session",
            track=Track.SHARED,
            source_event="cross_track_tension",
            credit_value=_clamp(1.0 - dual_track_snapshot.cross_track_tension),
            context=dual_track_snapshot.description,
            timestamp_ms=timestamp_ms,
        )
    )
    return tuple(records)


def derive_credit_records_from_prediction_error_first(
    *,
    dual_track_snapshot: DualTrackSnapshot,
    evaluation_snapshot: EvaluationSnapshot,
    prediction_error_snapshot: PredictionErrorSnapshot | None,
    timestamp_ms: int,
    temporal_snapshot: TemporalAbstractionSnapshot | None = None,
) -> tuple[CreditRecord, ...]:
    """Primary credit derivation path.

    When prediction error is available, PE-derived records are the primary
    signal and evaluation-derived records are reduced to lightweight
    readout/evidence credits. Without PE, fall back to the legacy path.
    """
    if prediction_error_snapshot is None:
        return derive_credit_records(
            dual_track_snapshot=dual_track_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            timestamp_ms=timestamp_ms,
        )
    records = list(
        derive_prediction_error_credit_records(
            prediction_error=prediction_error_snapshot.error,
            timestamp_ms=timestamp_ms,
            action_context=prediction_error_snapshot.action_context,
        )
    )
    if evaluation_snapshot.turn_scores:
        for score in evaluation_snapshot.turn_scores:
            if score.family in {"task", "relationship", "learning", "abstraction", "safety"}:
                records.append(
                    CreditRecord(
                        record_id=str(uuid4()),
                        level="evaluation_readout",
                        track=(
                            Track.WORLD
                            if score.family == "task"
                            else Track.SELF
                            if score.family == "relationship"
                            else Track.SHARED
                        ),
                        source_event=f"evaluation_readout:{score.metric_name}",
                        credit_value=_clamp(score.value * 0.25),
                        context=score.evidence,
                        timestamp_ms=timestamp_ms,
                    )
                )
    records.append(
        CreditRecord(
            record_id=str(uuid4()),
            level="prediction_error",
            track=Track.SHARED,
            source_event="pe:cross_track_tension",
            credit_value=_clamp(0.5 - dual_track_snapshot.cross_track_tension),
            context=dual_track_snapshot.description,
            timestamp_ms=timestamp_ms,
        )
    )
    if temporal_snapshot is not None:
        records.extend(
            derive_segment_closure_credit_records(
                prediction_error_snapshot=prediction_error_snapshot,
                temporal_snapshot=temporal_snapshot,
                timestamp_ms=timestamp_ms,
            )
        )
    return tuple(records)


def derive_prediction_error_credit_records(
    *,
    prediction_error: PredictionError,
    timestamp_ms: int,
    action_context: PredictionActionContext | None = None,
) -> tuple[CreditRecord, ...]:
    context_suffix = _action_context_suffix(action_context)
    records = (
        CreditRecord(
            record_id=str(uuid4()),
            level="prediction_error",
            track=Track.WORLD,
            source_event="pe:task",
            credit_value=_clamp(prediction_error.task_error),
            context=f"{prediction_error.description}{context_suffix}",
            timestamp_ms=timestamp_ms,
        ),
        CreditRecord(
            record_id=str(uuid4()),
            level="prediction_error",
            track=Track.SELF,
            source_event="pe:relationship",
            credit_value=_clamp(prediction_error.relationship_error),
            context=f"{prediction_error.description}{context_suffix}",
            timestamp_ms=timestamp_ms,
        ),
        CreditRecord(
            record_id=str(uuid4()),
            level="prediction_error",
            track=Track.SHARED,
            source_event="pe:regime",
            credit_value=_clamp(prediction_error.regime_error),
            context=f"{prediction_error.description}{context_suffix}",
            timestamp_ms=timestamp_ms,
        ),
        CreditRecord(
            record_id=str(uuid4()),
            level="prediction_error",
            track=Track.SHARED,
            source_event="pe:action",
            credit_value=_clamp(prediction_error.action_error),
            context=f"{prediction_error.description}{context_suffix}",
            timestamp_ms=timestamp_ms,
        ),
    )
    return records


def derive_segment_closure_credit_records(
    *,
    prediction_error_snapshot: PredictionErrorSnapshot,
    temporal_snapshot: TemporalAbstractionSnapshot,
    timestamp_ms: int = 0,
) -> tuple[CreditRecord, ...]:
    context = prediction_error_snapshot.action_context
    if not context.segment_id or not temporal_snapshot.closed_segments:
        return ()
    if not any(
        segment.segment_id == context.segment_id
        for segment in temporal_snapshot.closed_segments
    ):
        return ()
    return (
        CreditRecord(
            record_id=str(uuid4()),
            level="abstract_action_segment",
            track=Track.SHARED,
            source_event=f"segment:{context.segment_id}",
            credit_value=_clamp(prediction_error_snapshot.error.action_error),
            context=(
                f"abstract_action={context.abstract_action_id}; "
                f"z_t_digest={context.z_t_digest}; "
                f"environment_event_id={context.environment_event_id}; "
                f"environment_outcome_id={context.environment_outcome_id}; "
                f"pe={prediction_error_snapshot.error.description}"
            ),
            timestamp_ms=timestamp_ms,
        ),
    )


def derive_social_prediction_error_credit_records(
    *,
    social_errors: tuple[SocialPredictionError, ...],
    timestamp_ms: int,
) -> tuple[CreditRecord, ...]:
    records: list[CreditRecord] = []
    for error in social_errors:
        sign = -1.0 if error.outcome is SocialPredictionOutcome.DISCONFIRMED else 1.0
        records.append(
            CreditRecord(
                record_id=str(uuid4()),
                level="social_prediction_error",
                track=Track.SHARED,
                source_event=f"social_pe:{error.kind.value}",
                credit_value=_clamp(sign * error.magnitude),
                context=(
                    f"owner={error.owner}; scope={error.scope_kind.value}:{error.scope_id}; "
                    f"outcome={error.outcome.value}; evidence={' | '.join(error.evidence)}"
                ),
                timestamp_ms=timestamp_ms,
            )
        )
    return tuple(records)


def _action_context_suffix(action_context: PredictionActionContext | None) -> str:
    if action_context is None:
        return ""
    segment_id = action_context.segment_id
    abstract_action_id = action_context.abstract_action_id
    environment_event_id = action_context.environment_event_id
    environment_outcome_id = action_context.environment_outcome_id
    if not any((segment_id, abstract_action_id, environment_event_id, environment_outcome_id)):
        return ""
    return (
        f" action_context[segment_id={segment_id}; "
        f"abstract_action={abstract_action_id}; "
        f"environment_event_id={environment_event_id}; "
        f"environment_outcome_id={environment_outcome_id}]"
    )


def derive_abstract_action_credit_records(
    *,
    temporal_snapshot: TemporalAbstractionSnapshot,
    dual_track_snapshot: DualTrackSnapshot,
    evaluation_snapshot: EvaluationSnapshot,
    timestamp_ms: int,
) -> tuple[CreditRecord, ...]:
    dominant_track = Track.SHARED
    if dual_track_snapshot.world_track.tension_level > dual_track_snapshot.self_track.tension_level:
        dominant_track = Track.WORLD
    elif dual_track_snapshot.self_track.tension_level > dual_track_snapshot.world_track.tension_level:
        dominant_track = Track.SELF
    reward_signal = 1.0 - dual_track_snapshot.cross_track_tension
    if evaluation_snapshot.turn_scores:
        reward_signal = sum(score.value for score in evaluation_snapshot.turn_scores) / len(
            evaluation_snapshot.turn_scores
        )
    reward_signal = _clamp(reward_signal + temporal_snapshot.controller_state.switch_gate * 0.1)
    return (
        CreditRecord(
            record_id=str(uuid4()),
            level="abstract_action",
            track=dominant_track,
            source_event=temporal_snapshot.active_abstract_action,
            credit_value=reward_signal,
            context=temporal_snapshot.description,
            timestamp_ms=timestamp_ms,
        ),
    )


def evaluate_gate(
    *,
    proposal: ModificationProposal,
    evaluation_snapshot: EvaluationSnapshot,
) -> GateDecision:
    if evaluate_gate_reasons(proposal=proposal, evaluation_snapshot=evaluation_snapshot):
        return GateDecision.BLOCK
    return GateDecision.ALLOW


def evaluate_gate_reasons(
    *,
    proposal: ModificationProposal,
    evaluation_snapshot: EvaluationSnapshot,
) -> tuple[str, ...]:
    """Return fail-closed blocking reasons for a self-modification proposal."""
    critical_alert = any(alert.severity == "CRITICAL" for alert in evaluation_snapshot.structured_alerts)
    high_alert = any(alert.severity == "HIGH" for alert in evaluation_snapshot.structured_alerts)
    metric_values = {
        score.metric_name: score.value
        for score in evaluation_snapshot.turn_scores + evaluation_snapshot.session_scores
    }
    reasons: list[str] = []

    if proposal.desired_gate is ModificationGate.ONLINE and (critical_alert or high_alert):
        reasons.append("online gate blocked by high-or-critical evaluation alert")
    if proposal.desired_gate is ModificationGate.BACKGROUND and critical_alert:
        reasons.append("background gate blocked by critical evaluation alert")
    if proposal.desired_gate is ModificationGate.HUMAN_REVIEW:
        reasons.append("human-review proposal cannot be auto-allowed by runtime gate")
    margin = _validation_margin_for_gate(proposal.desired_gate)
    if proposal.validation_delta < margin:
        reasons.append(
            f"validation_delta {proposal.validation_delta:.3f} below required margin {margin:.3f}"
        )
    capacity_cap = _capacity_cap_for_gate(proposal.desired_gate)
    if proposal.capacity_cost > capacity_cap:
        reasons.append(
            f"capacity_cost {proposal.capacity_cost:.3f} exceeds cap {capacity_cap:.3f}"
        )
    if not proposal.rollback_evidence:
        reasons.append("missing rollback evidence")
    if proposal.desired_gate in {ModificationGate.ONLINE, ModificationGate.BACKGROUND} and not proposal.is_reversible:
        reasons.append("online/background proposal is not reversible")
    contract_integrity = metric_values.get("contract_integrity", 1.0)
    if contract_integrity < 0.95:
        reasons.append(f"contract_integrity {contract_integrity:.3f} below 0.950")
    fallback_reliance = metric_values.get("fallback_reliance", 0.0)
    if proposal.desired_gate is ModificationGate.ONLINE and fallback_reliance > 0.5:
        reasons.append(f"fallback_reliance {fallback_reliance:.3f} above 0.500")
    rollback_resilience = metric_values.get("rollback_resilience", 1.0)
    if rollback_resilience < 0.6:
        reasons.append(f"rollback_resilience {rollback_resilience:.3f} below 0.600")
    return tuple(reasons)


def _validation_margin_for_gate(gate: ModificationGate) -> float:
    if gate is ModificationGate.ONLINE:
        return 0.0
    if gate is ModificationGate.BACKGROUND:
        return 0.02
    if gate is ModificationGate.OFFLINE:
        return 0.05
    return 0.0


def _capacity_cap_for_gate(gate: ModificationGate) -> float:
    if gate is ModificationGate.ONLINE:
        return 0.20
    if gate is ModificationGate.BACKGROUND:
        return 0.45
    if gate is ModificationGate.OFFLINE:
        return 0.75
    return 1.0


def has_blocking_writeback(credit_snapshot: CreditSnapshot, *, target_prefix: str | None = None) -> bool:
    relevant_records = credit_snapshot.recent_modifications
    if target_prefix is not None:
        relevant_records = tuple(
            record for record in relevant_records if record.target.startswith(target_prefix)
        )
    return any(record.decision is GateDecision.BLOCK for record in relevant_records)


def extend_credit_snapshot(
    *,
    credit_snapshot: CreditSnapshot,
    extra_records: tuple[CreditRecord, ...] = (),
    extra_modifications: tuple[SelfModificationRecord, ...] = (),
) -> CreditSnapshot:
    cumulative: dict[str, float] = dict(credit_snapshot.cumulative_credit_by_level)
    for record in extra_records:
        cumulative[record.level] = cumulative.get(record.level, 0.0) + record.credit_value
    recent_credits = tuple((credit_snapshot.recent_credits + extra_records)[-20:])
    recent_modifications = tuple((credit_snapshot.recent_modifications + extra_modifications)[-20:])
    return CreditSnapshot(
        recent_credits=recent_credits,
        recent_modifications=recent_modifications,
        cumulative_credit_by_level=tuple(sorted(cumulative.items())),
        session_level_credits=credit_snapshot.session_level_credits,
        discount_factor=credit_snapshot.discount_factor,
        delayed_ledger_size=credit_snapshot.delayed_ledger_size,
        horizon_depth=credit_snapshot.horizon_depth,
        description=(
            f"{credit_snapshot.description} Extended with {len(extra_records)} extra credits "
            f"and {len(extra_modifications)} extra modification audits."
        ),
    )


_DIALOGUE_OUTCOME_KIND_SIGN: dict[str, float] = {
    "continued": 1.0,
    "clarified": 1.0,
    "corrected": -1.0,
    "rejected": -1.0,
    "deferred": 0.25,
    "scene_closed": 0.0,
    "unknown": 0.0,
}


def derive_dialogue_outcome_credit_records(
    *,
    outcome_evidence: tuple[object, ...],
    timestamp_ms: int,
    track: Track = Track.SHARED,
) -> tuple[CreditRecord, ...]:
    """Derive turn-level credit records from typed dialogue outcome evidence.

    Each entry in ``outcome_evidence`` must be a frozen
    ``DialogueOutcomeEvidence`` produced by an owner or evaluation
    readout. The credit value is the signed product of evidence
    confidence and a fixed structural sign per outcome kind. The trace
    layer stays evidence-only; this helper turns owner-published
    evidence into ledger entries without inventing semantics.
    """

    records: list[CreditRecord] = []
    for evidence in outcome_evidence:
        outcome_kind = getattr(evidence, "outcome_kind", None)
        kind_value = getattr(outcome_kind, "value", None)
        sign = _DIALOGUE_OUTCOME_KIND_SIGN.get(kind_value or "", 0.0)
        if sign == 0.0:
            continue
        confidence = float(getattr(evidence, "confidence", 0.0))
        if confidence <= 0.0:
            continue
        evidence_id = getattr(evidence, "evidence_id", "")
        source_owner = getattr(evidence, "source_owner", "")
        records.append(
            CreditRecord(
                record_id=str(uuid4()),
                level="turn",
                track=track,
                source_event=f"dialogue_outcome:{kind_value}:{source_owner}",
                credit_value=_clamp(sign * confidence),
                context=evidence_id,
                timestamp_ms=timestamp_ms,
            )
        )
    return tuple(records)


def derive_learning_evidence_credit_records(
    *,
    evaluation_snapshot: EvaluationSnapshot,
    timestamp_ms: int,
) -> tuple[CreditRecord, ...]:
    turn_learning_metrics = {
        "retrieval_quality",
        "reflection_usefulness",
        "joint_learning_progress",
        "regime_sequence_payoff",
        "delayed_retrieval_mix_alignment",
        "delayed_regime_alignment",
    }
    metacontroller_metrics = {
        "adaptive_stability",
        "posterior_stability",
        "switch_sparsity",
        "binary_gate_ratio",
        "decoder_usefulness",
        "policy_replacement_quality",
        "abstract_action_usefulness",
        "temporal_action_commitment",
        "action_family_reuse",
        "action_family_stability",
        "action_family_diversity",
        "delayed_action_alignment",
        "delayed_abstract_action_alignment",
    }
    session_learning_metrics = {
        "regime_sequence_payoff",
        "delayed_retrieval_mix_alignment",
        "delayed_regime_alignment",
    }
    credit_records: list[CreditRecord] = []
    for score in evaluation_snapshot.turn_scores:
        if score.family == "learning" and score.metric_name in turn_learning_metrics:
            credit_records.append(
                CreditRecord(
                    record_id=str(uuid4()),
                    level="turn",
                    track=Track.SHARED,
                    source_event=score.metric_name,
                    credit_value=_clamp(score.value),
                    context=score.evidence,
                    timestamp_ms=timestamp_ms,
                )
            )
            continue
        if score.metric_name in metacontroller_metrics:
            credit_records.append(
                CreditRecord(
                    record_id=str(uuid4()),
                    level="abstract_action",
                    track=Track.SHARED,
                    source_event=f"evaluation:{score.metric_name}",
                    credit_value=_clamp(score.value),
                    context=score.evidence,
                    timestamp_ms=timestamp_ms,
                )
            )
    for score in evaluation_snapshot.session_scores:
        if score.family == "learning" and score.metric_name in session_learning_metrics:
            credit_records.append(
                CreditRecord(
                    record_id=str(uuid4()),
                    level="session",
                    track=Track.SHARED,
                    source_event=f"session:{score.metric_name}",
                    credit_value=_clamp(score.value),
                    context=score.evidence,
                    timestamp_ms=timestamp_ms,
                )
            )
            continue
        if score.metric_name in metacontroller_metrics:
            credit_records.append(
                CreditRecord(
                    record_id=str(uuid4()),
                    level="abstract_action",
                    track=Track.SHARED,
                    source_event=f"session-evaluation:{score.metric_name}",
                    credit_value=_clamp(score.value),
                    context=score.evidence,
                    timestamp_ms=timestamp_ms,
                )
            )
    return tuple(credit_records)


def derive_delayed_attribution_credit_records(
    *,
    regime_snapshot: object | None,
    timestamp_ms: int,
) -> tuple[CreditRecord, ...]:
    from volvence_zero.regime import RegimeSnapshot

    if regime_snapshot is None or not isinstance(regime_snapshot, RegimeSnapshot):
        return ()
    credit_records: list[CreditRecord] = []
    for payoff in regime_snapshot.delayed_payoffs:
        payoff_context = (
            f"last_source_wave_id={payoff.last_source_wave_id} "
            f"abstract_action={payoff.abstract_action or 'none'} "
            f"action_family_version={payoff.action_family_version} "
            f"sample_count={payoff.sample_count} "
            f"rolling_payoff={payoff.rolling_payoff:.3f}"
        )
        credit_records.append(
            CreditRecord(
                record_id=str(uuid4()),
                level="session",
                track=Track.SHARED,
                source_event=f"delayed_payoff:{payoff.regime_id}",
                credit_value=_clamp(payoff.rolling_payoff),
                context=payoff_context,
                timestamp_ms=timestamp_ms,
            )
        )
        if payoff.abstract_action is not None:
            credit_records.append(
                CreditRecord(
                    record_id=str(uuid4()),
                    level="abstract_action",
                    track=Track.SHARED,
                    source_event=f"delayed_payoff_action:{payoff.abstract_action}",
                    credit_value=_clamp(payoff.rolling_payoff),
                    context=payoff_context,
                    timestamp_ms=timestamp_ms,
                )
            )
    for attribution in regime_snapshot.delayed_attributions:
        context = (
            f"source_wave_id={attribution.source_wave_id} "
            f"source_turn_index={attribution.source_turn_index} "
            f"abstract_action={attribution.abstract_action or 'none'} "
            f"action_family_version={attribution.action_family_version} "
            f"outcome_score={attribution.outcome_score:.3f}"
        )
        credit_records.append(
            CreditRecord(
                record_id=str(uuid4()),
                level="session",
                track=Track.SHARED,
                source_event=f"delayed_regime:{attribution.regime_id}",
                credit_value=_clamp(attribution.outcome_score),
                context=context,
                timestamp_ms=timestamp_ms,
            )
        )
        if attribution.abstract_action is not None:
            credit_records.append(
                CreditRecord(
                    record_id=str(uuid4()),
                    level="abstract_action",
                    track=Track.SHARED,
                    source_event=f"delayed_action:{attribution.abstract_action}",
                    credit_value=_clamp(attribution.outcome_score),
                    context=context,
                    timestamp_ms=timestamp_ms,
                )
            )
    return tuple(credit_records)


def derive_counterfactual_contribution_records(
    *,
    regime_snapshot: object | None,
    temporal_snapshot: TemporalAbstractionSnapshot | None,
    prediction_error_snapshot: PredictionErrorSnapshot | None,
    timestamp_ms: int,
) -> tuple[CreditRecord, ...]:
    """Lightweight COCOA-style contribution credit (Phase 1.A).

    Inspired by COCOA (Meulemans et al., NeurIPS 2023): credit for an
    action is the difference between its observed outcome and the
    *counterfactual baseline* the policy would have produced under the
    same context. We do **not** train a rewarding-state head here
    (Phase 2.A uplift); instead we reuse already-published statistics
    from the regime and temporal owners:

    - regime owner publishes ``selection_weights`` (per regime) and
      ``delayed_payoffs`` (rolling payoff per regime / abstract-action /
      family-version);
    - the chosen regime / abstract-action is read from the
      ``PredictionErrorSnapshot.action_context`` and from the
      ``temporal_snapshot.active_abstract_action`` at segment closure;
    - the actual scalar outcome is the PE ``signed_reward`` (already
      bounded in ``[-1, 1]``).

    baseline = sum_i normalized_w_i * rolling_payoff_i over the regime
    selection distribution. contribution = actual - baseline. Returns an
    empty tuple if any required snapshot is missing or the chosen
    context lacks a known regime / segment, so the caller stays
    fail-safe in bootstrap turns.

    The new ``CreditRecord`` ships at ``level="counterfactual_contribution"``
    so existing ``recent_credits`` consumers (which filter by level)
    treat it as opt-in evidence rather than a behavioural change.
    """

    from volvence_zero.regime import RegimeSnapshot

    if (
        prediction_error_snapshot is None
        or prediction_error_snapshot.bootstrap
        or regime_snapshot is None
        or not isinstance(regime_snapshot, RegimeSnapshot)
    ):
        return ()

    action_context = prediction_error_snapshot.action_context
    chosen_regime_id = action_context.regime_id
    chosen_abstract_action = action_context.abstract_action_id
    chosen_segment_id = action_context.segment_id

    # Backfill regime_id from the active regime when the PE owner did
    # not stamp one on the action context. This is read-only and uses
    # already-published regime state.
    if not chosen_regime_id and regime_snapshot.active_regime is not None:
        chosen_regime_id = regime_snapshot.active_regime.regime_id

    if not chosen_regime_id:
        return ()

    weights_payload: tuple[tuple[str, float], ...] = ()
    if regime_snapshot.selection_weights is not None:
        weights_payload = regime_snapshot.selection_weights.weights
    if not weights_payload:
        weights_payload = regime_snapshot.candidate_regimes

    # Build a payoff lookup keyed by regime_id; if the abstract action
    # matches in the chosen regime, prefer the (regime, action) entry,
    # otherwise fall back to the regime-level rolling payoff.
    regime_payoffs: dict[str, float] = {}
    regime_action_payoffs: dict[tuple[str, str], float] = {}
    for payoff in regime_snapshot.delayed_payoffs:
        regime_payoffs.setdefault(payoff.regime_id, payoff.rolling_payoff)
        if payoff.abstract_action is not None:
            regime_action_payoffs[(payoff.regime_id, payoff.abstract_action)] = (
                payoff.rolling_payoff
            )

    # Skip when we have no historical payoff signal for any regime.
    # Without it the baseline collapses to zero and the contribution
    # equals the raw actual, which would double-count PE credit.
    if not regime_payoffs and not regime_action_payoffs:
        return ()

    weight_total = 0.0
    weight_pairs: list[tuple[str, float]] = []
    for regime_id, weight in weights_payload:
        clean_weight = max(0.0, float(weight))
        if clean_weight <= 0.0:
            continue
        weight_total += clean_weight
        weight_pairs.append((regime_id, clean_weight))
    if weight_total <= 0.0 or not weight_pairs:
        return ()

    baseline = 0.0
    contributors = 0
    for regime_id, weight in weight_pairs:
        normalized = weight / weight_total
        action_key = (regime_id, chosen_abstract_action) if chosen_abstract_action else None
        if action_key is not None and action_key in regime_action_payoffs:
            baseline += normalized * regime_action_payoffs[action_key]
            contributors += 1
            continue
        if regime_id in regime_payoffs:
            baseline += normalized * regime_payoffs[regime_id]
            contributors += 1
    if contributors == 0:
        return ()

    actual = float(prediction_error_snapshot.error.signed_reward)
    contribution = _clamp(actual - baseline)

    source_event = f"cocoa:{chosen_regime_id}"
    if chosen_segment_id:
        source_event = f"{source_event}:{chosen_segment_id}"
    if chosen_abstract_action:
        source_event = f"{source_event}:{chosen_abstract_action}"

    context_parts = [
        f"baseline={baseline:.3f}",
        f"actual={actual:.3f}",
        f"weight_total={weight_total:.3f}",
        f"contributors={contributors}",
    ]
    if temporal_snapshot is not None:
        context_parts.append(
            f"temporal_active={temporal_snapshot.active_abstract_action}"
        )
        if chosen_segment_id and any(
            segment.segment_id == chosen_segment_id
            for segment in temporal_snapshot.closed_segments
        ):
            context_parts.append(f"segment_closed={chosen_segment_id}")
    context = "; ".join(context_parts)

    return (
        CreditRecord(
            record_id=str(uuid4()),
            level="counterfactual_contribution",
            track=Track.SHARED,
            source_event=source_event,
            credit_value=contribution,
            context=context,
            timestamp_ms=timestamp_ms,
        ),
    )


def record_nstep_outcomes_from_segment_closure(
    *,
    ledger: "CreditLedger",
    prediction_error_snapshot: PredictionErrorSnapshot | None,
    temporal_snapshot: TemporalAbstractionSnapshot | None,
    regime_snapshot: object | None,
    timestamp_ms: int,
) -> int:
    """Append the chosen action's outcome to the N-step ledger.

    The COCOA path needs a per-action outcome trajectory so future
    Phase 2 uplift can compute multi-step counterfactuals. We resurrect
    the dormant ``CreditLedger.record_nstep_outcome`` API by calling it
    when ``temporal_snapshot.closed_segments`` reports that the chosen
    segment just closed. Returns the number of N-step entries that were
    created or extended (0 when no segment closed or when the snapshot
    chain is incomplete).
    """

    from volvence_zero.regime import RegimeSnapshot

    if (
        prediction_error_snapshot is None
        or prediction_error_snapshot.bootstrap
        or temporal_snapshot is None
        or not temporal_snapshot.closed_segments
    ):
        return 0

    action_context = prediction_error_snapshot.action_context
    chosen_segment_id = action_context.segment_id
    chosen_abstract_action = action_context.abstract_action_id
    chosen_regime_id = action_context.regime_id
    if not chosen_regime_id and isinstance(regime_snapshot, RegimeSnapshot):
        if regime_snapshot.active_regime is not None:
            chosen_regime_id = regime_snapshot.active_regime.regime_id
    if not chosen_segment_id and not chosen_abstract_action:
        return 0
    matched_segment = next(
        (
            segment
            for segment in temporal_snapshot.closed_segments
            if (chosen_segment_id and segment.segment_id == chosen_segment_id)
            or (
                chosen_abstract_action
                and segment.abstract_action_id == chosen_abstract_action
            )
        ),
        None,
    )
    if matched_segment is None:
        return 0
    action_id = matched_segment.segment_id or chosen_segment_id or chosen_abstract_action
    family_id = matched_segment.abstract_action_id or chosen_abstract_action or "unknown"
    regime_id = chosen_regime_id or "unknown"
    outcome = float(prediction_error_snapshot.error.signed_reward)
    ledger.record_nstep_outcome(
        action_id=action_id,
        family_id=family_id,
        regime_id=regime_id,
        outcome=outcome,
        timestamp_ms=timestamp_ms,
    )
    return 1


def derive_runtime_adaptation_audit_records(
    *,
    rollback_reasons: tuple[str, ...],
    metacontroller_state_description: str | None,
    timestamp_ms: int,
    rollback_applied: bool,
) -> tuple[SelfModificationRecord, ...]:
    if not rollback_reasons and not rollback_applied:
        return ()
    joined_reasons = ", ".join(rollback_reasons) if rollback_reasons else "none"
    state_description = metacontroller_state_description or "metacontroller state unavailable"
    decision = GateDecision.BLOCK if rollback_applied else GateDecision.ALLOW
    return (
        SelfModificationRecord(
            target="metacontroller.runtime_adaptation",
            gate=ModificationGate.BACKGROUND,
            decision=decision,
            old_value_hash=f"reasons:{joined_reasons}",
            new_value_hash=f"state:{state_description}",
            justification=(
                f"{'BLOCKED' if rollback_applied else 'ALLOWED'} runtime metacontroller adaptation; "
                f"rollback_reasons={joined_reasons}; {state_description}"
            ),
            timestamp_ms=timestamp_ms,
            is_reversible=True,
        ),
    )


def derive_metacontroller_credit_records(
    *,
    metacontroller_state: "MetacontrollerRuntimeState | None",
    policy_objective: float,
    rollback_reasons: tuple[str, ...],
    timestamp_ms: int,
) -> tuple[CreditRecord, ...]:
    if metacontroller_state is None:
        return ()
    credit_value = _clamp(
        0.55
        + policy_objective * 0.35
        - min(metacontroller_state.latest_ssl_loss * 0.1, 0.25)
        + metacontroller_state.policy_replacement_score * 0.15
        + metacontroller_state.switch_sparsity * 0.05
        - len(rollback_reasons) * 0.15
    )
    active_family = metacontroller_state.active_family_summary
    return (
        CreditRecord(
            record_id=f"metacontroller:{timestamp_ms}",
            level="abstract_action",
            track=Track.SHARED,
            source_event=metacontroller_state.active_label,
            credit_value=credit_value,
            context=(
                f"{metacontroller_state.description} "
                f"policy_objective={policy_objective:.3f} "
                f"posterior_drift={metacontroller_state.posterior_drift:.3f} "
                f"binary_ratio={metacontroller_state.binary_switch_rate:.3f} "
                f"replacement={metacontroller_state.policy_replacement_score:.3f} "
                f"family_version={metacontroller_state.action_family_version} "
                f"family_support={active_family.support if active_family is not None else 0} "
                f"family_stability={active_family.stability if active_family is not None else 0.0:.3f} "
                f"rollback={','.join(rollback_reasons) or 'none'}"
            ),
            timestamp_ms=timestamp_ms,
        ),
    )


def extract_abstract_action_credit_bonus(
    credit_snapshot: CreditSnapshot,
    *,
    bonus_weight: float = 0.1,
    pe_magnitude: float = 0.0,
) -> dict[str, float]:
    """Aggregate abstract-action credit into per-family bonus signals.

    Returns a dict mapping source_event (family/action name) to a
    weighted credit bonus suitable for injecting into the RL environment
    via ``set_evaluation_signals``.
    """
    family_credits: dict[str, list[float]] = {}
    for record in credit_snapshot.recent_credits:
        if record.level != "abstract_action":
            continue
        family_credits.setdefault(record.source_event, []).append(record.credit_value)
    result: dict[str, float] = {}
    for family, values in family_credits.items():
        mean_credit = sum(values) / len(values) if values else 0.0
        magnitude_boost = 1.0 + max(0.0, pe_magnitude)
        result[family] = _clamp(mean_credit * bonus_weight * magnitude_boost)
    return result


class CreditLedger:
    """Stores recent credit records and gate outcomes with session-level aggregation."""

    def __init__(self, *, discount_factor: float = 0.95, horizon_depth: int = 5) -> None:
        self._recent_credits: list[CreditRecord] = []
        self._recent_modifications: list[SelfModificationRecord] = []
        self._discount_factor = discount_factor
        self._session_credits: dict[str, list[CreditRecord]] = {}
        self._turn_count = 0
        self._horizon_depth = max(horizon_depth, 1)
        self._nstep_ledger: list[NStepAttributionEntry] = []
        self._max_ledger_entries = 1000

    @property
    def recent_credits(self) -> tuple[CreditRecord, ...]:
        return tuple(self._recent_credits)

    @property
    def recent_modifications(self) -> tuple[SelfModificationRecord, ...]:
        return tuple(self._recent_modifications)

    @property
    def discount_factor(self) -> float:
        return self._discount_factor

    @property
    def horizon_depth(self) -> int:
        return self._horizon_depth

    def record_credits(self, credits: tuple[CreditRecord, ...]) -> None:
        self._recent_credits.extend(credits)
        for credit in credits:
            key = f"{credit.level}:{credit.track.value}"
            self._session_credits.setdefault(key, []).append(credit)
        self._turn_count += 1

    def record_modification(self, record: SelfModificationRecord) -> None:
        self._recent_modifications.append(record)

    def record_nstep_outcome(
        self,
        *,
        action_id: str,
        family_id: str,
        regime_id: str,
        outcome: float,
        timestamp_ms: int,
    ) -> None:
        """Append an outcome to the matching N-step entry, or create a new one."""
        for i, entry in enumerate(self._nstep_ledger):
            if entry.action_id == action_id:
                new_history = (entry.outcome_history + (outcome,))[-self._horizon_depth:]
                self._nstep_ledger[i] = NStepAttributionEntry(
                    action_id=entry.action_id,
                    family_id=entry.family_id,
                    regime_id=entry.regime_id,
                    timestamp_ms=entry.timestamp_ms,
                    outcome_history=new_history,
                )
                return
        self._nstep_ledger.append(NStepAttributionEntry(
            action_id=action_id,
            family_id=family_id,
            regime_id=regime_id,
            timestamp_ms=timestamp_ms,
            outcome_history=(outcome,),
        ))
        if len(self._nstep_ledger) > self._max_ledger_entries:
            self._nstep_ledger = self._nstep_ledger[-self._max_ledger_entries:]

    def compute_nstep_return(self, *, action_id: str) -> float:
        """Compute discounted N-step return for a given action."""
        for entry in self._nstep_ledger:
            if entry.action_id == action_id:
                total = 0.0
                for k, outcome in enumerate(entry.outcome_history):
                    total += (self._discount_factor ** k) * outcome
                return total
        return 0.0

    def rolling_payoff_by_family(self) -> dict[str, float]:
        """EMA of N-step returns grouped by family_id."""
        family_returns: dict[str, list[float]] = {}
        for entry in self._nstep_ledger:
            if not entry.outcome_history:
                continue
            nstep = sum(
                (self._discount_factor ** k) * v
                for k, v in enumerate(entry.outcome_history)
            )
            family_returns.setdefault(entry.family_id, []).append(nstep)
        result: dict[str, float] = {}
        for family_id, returns in family_returns.items():
            ema = returns[0]
            for r in returns[1:]:
                ema = 0.8 * ema + 0.2 * r
            result[family_id] = round(ema, 4)
        return result

    def rolling_payoff_by_regime(self) -> dict[str, float]:
        """EMA of N-step returns grouped by regime_id."""
        regime_returns: dict[str, list[float]] = {}
        for entry in self._nstep_ledger:
            if not entry.outcome_history:
                continue
            nstep = sum(
                (self._discount_factor ** k) * v
                for k, v in enumerate(entry.outcome_history)
            )
            regime_returns.setdefault(entry.regime_id, []).append(nstep)
        result: dict[str, float] = {}
        for regime_id, returns in regime_returns.items():
            ema = returns[0]
            for r in returns[1:]:
                ema = 0.8 * ema + 0.2 * r
            result[regime_id] = round(ema, 4)
        return result

    def aggregate_session_credits(self) -> tuple[tuple[str, float], ...]:
        """Compute discounted sum of credit records per session key."""
        result: dict[str, float] = {}
        for key, records in self._session_credits.items():
            discounted_sum = 0.0
            for i, record in enumerate(reversed(records)):
                discounted_sum += record.credit_value * (self._discount_factor ** i)
            result[key] = round(discounted_sum, 4)
        return tuple(sorted(result.items()))

    def snapshot(self) -> CreditSnapshot:
        cumulative: dict[str, float] = {}
        for record in self._recent_credits:
            cumulative[record.level] = cumulative.get(record.level, 0.0) + record.credit_value
        return CreditSnapshot(
            recent_credits=tuple(self._recent_credits[-10:]),
            recent_modifications=tuple(self._recent_modifications[-10:]),
            cumulative_credit_by_level=tuple(sorted(cumulative.items())),
            session_level_credits=self.aggregate_session_credits(),
            discount_factor=self._discount_factor,
            delayed_ledger_size=len(self._nstep_ledger),
            horizon_depth=self._horizon_depth,
            description=(
                f"Credit ledger with {len(self._recent_credits)} credit records and "
                f"{len(self._recent_modifications)} modification records, "
                f"gamma={self._discount_factor}, turns={self._turn_count}, "
                f"nstep_entries={len(self._nstep_ledger)}, horizon={self._horizon_depth}."
            ),
        )


class CreditModule(RuntimeModule[CreditSnapshot]):
    slot_name = "credit"
    owner = "CreditModule"
    value_type = CreditSnapshot
    dependencies = ("dual_track", "evaluation", "prediction_error", "temporal_abstraction")
    default_wiring_level = WiringLevel.SHADOW

    def __init__(
        self,
        *,
        ledger: CreditLedger | None = None,
        pending_proposals: tuple[ModificationProposal, ...] = (),
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._ledger = ledger or CreditLedger()
        self._pending_proposals = pending_proposals

    @property
    def ledger(self) -> CreditLedger:
        return self._ledger

    async def process(self, upstream: Mapping[str, Snapshot[object]]) -> Snapshot[CreditSnapshot]:
        dual_track_snapshot = upstream["dual_track"]
        evaluation_snapshot = upstream["evaluation"]
        prediction_error_snapshot = upstream["prediction_error"]
        temporal_snapshot = upstream.get("temporal_abstraction")
        dual_track_value = (
            dual_track_snapshot.value if isinstance(dual_track_snapshot.value, DualTrackSnapshot) else None
        )
        evaluation_value = (
            evaluation_snapshot.value if isinstance(evaluation_snapshot.value, EvaluationSnapshot) else None
        )
        prediction_error_value = (
            prediction_error_snapshot.value if isinstance(prediction_error_snapshot.value, PredictionErrorSnapshot) else None
        )
        temporal_value = (
            temporal_snapshot.value
            if temporal_snapshot is not None
            and isinstance(temporal_snapshot.value, TemporalAbstractionSnapshot)
            else None
        )
        if dual_track_value is None or evaluation_value is None:
            return self.publish(self._ledger.snapshot())

        credits = derive_credit_records_from_prediction_error_first(
            dual_track_snapshot=dual_track_value,
            evaluation_snapshot=evaluation_value,
            prediction_error_snapshot=prediction_error_value,
            timestamp_ms=max(dual_track_snapshot.timestamp_ms, evaluation_snapshot.timestamp_ms),
            temporal_snapshot=temporal_value,
        )
        self._ledger.record_credits(credits)
        self._record_proposals(
            proposals=self._pending_proposals,
            evaluation_snapshot=evaluation_value,
            timestamp_ms=max(dual_track_snapshot.timestamp_ms, evaluation_snapshot.timestamp_ms),
        )
        return self.publish(self._ledger.snapshot())

    async def process_standalone(self, **kwargs: object) -> Snapshot[CreditSnapshot]:
        dual_track_snapshot = kwargs.get("dual_track_snapshot")
        evaluation_snapshot = kwargs.get("evaluation_snapshot")
        prediction_error_snapshot = kwargs.get("prediction_error_snapshot")
        temporal_snapshot = kwargs.get("temporal_snapshot")
        proposals = kwargs.get("proposals", self._pending_proposals)
        if not isinstance(proposals, tuple):
            raise TypeError("proposals must be a tuple when provided.")

        if isinstance(dual_track_snapshot, DualTrackSnapshot) and isinstance(
            evaluation_snapshot, EvaluationSnapshot
        ):
            credits = derive_credit_records_from_prediction_error_first(
                dual_track_snapshot=dual_track_snapshot,
                evaluation_snapshot=evaluation_snapshot,
                prediction_error_snapshot=prediction_error_snapshot if isinstance(prediction_error_snapshot, PredictionErrorSnapshot) else None,
                timestamp_ms=int(kwargs.get("timestamp_ms", 1)),
                temporal_snapshot=temporal_snapshot if isinstance(temporal_snapshot, TemporalAbstractionSnapshot) else None,
            )
            self._ledger.record_credits(credits)
            self._record_proposals(
                proposals=proposals,
                evaluation_snapshot=evaluation_snapshot,
                timestamp_ms=int(kwargs.get("timestamp_ms", 1)),
            )
        return self.publish(self._ledger.snapshot())

    def _record_proposals(
        self,
        *,
        proposals: tuple[ModificationProposal, ...],
        evaluation_snapshot: EvaluationSnapshot,
        timestamp_ms: int,
    ) -> None:
        for proposal in proposals:
            gate_reasons = evaluate_gate_reasons(proposal=proposal, evaluation_snapshot=evaluation_snapshot)
            decision = GateDecision.BLOCK if gate_reasons else GateDecision.ALLOW
            if decision is GateDecision.ALLOW:
                justification = f"ALLOWED: {proposal.justification}"
            else:
                justification = f"BLOCKED: {proposal.justification}; reasons={'; '.join(gate_reasons)}"
            self._ledger.record_modification(
                SelfModificationRecord(
                    target=proposal.target,
                    gate=proposal.desired_gate,
                    decision=decision,
                    old_value_hash=proposal.old_value_hash,
                    new_value_hash=proposal.new_value_hash,
                    justification=justification,
                    timestamp_ms=timestamp_ms,
                    is_reversible=proposal.is_reversible,
                )
            )
