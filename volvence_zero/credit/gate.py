from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Mapping
from uuid import uuid4

from volvence_zero.dual_track import DualTrackSnapshot
from volvence_zero.evaluation import EvaluationSnapshot
from volvence_zero.memory import Track
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel

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
    critical_alert = any(alert.startswith("CRITICAL") for alert in evaluation_snapshot.alerts)
    high_alert = any(alert.startswith("HIGH") for alert in evaluation_snapshot.alerts)

    if proposal.desired_gate is ModificationGate.ONLINE and (critical_alert or high_alert):
        return GateDecision.BLOCK
    if proposal.desired_gate is ModificationGate.BACKGROUND and critical_alert:
        return GateDecision.BLOCK
    return GateDecision.ALLOW


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
        result[family] = _clamp(mean_credit * bonus_weight)
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
    dependencies = ("dual_track", "evaluation")
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
        dual_track_value = (
            dual_track_snapshot.value if isinstance(dual_track_snapshot.value, DualTrackSnapshot) else None
        )
        evaluation_value = (
            evaluation_snapshot.value if isinstance(evaluation_snapshot.value, EvaluationSnapshot) else None
        )
        if dual_track_value is None or evaluation_value is None:
            return self.publish(self._ledger.snapshot())

        credits = derive_credit_records(
            dual_track_snapshot=dual_track_value,
            evaluation_snapshot=evaluation_value,
            timestamp_ms=max(dual_track_snapshot.timestamp_ms, evaluation_snapshot.timestamp_ms),
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
        proposals = kwargs.get("proposals", self._pending_proposals)
        if not isinstance(proposals, tuple):
            raise TypeError("proposals must be a tuple when provided.")

        if isinstance(dual_track_snapshot, DualTrackSnapshot) and isinstance(
            evaluation_snapshot, EvaluationSnapshot
        ):
            credits = derive_credit_records(
                dual_track_snapshot=dual_track_snapshot,
                evaluation_snapshot=evaluation_snapshot,
                timestamp_ms=int(kwargs.get("timestamp_ms", 1)),
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
            decision = evaluate_gate(proposal=proposal, evaluation_snapshot=evaluation_snapshot)
            if decision is GateDecision.ALLOW:
                justification = f"ALLOWED: {proposal.justification}"
            else:
                justification = f"BLOCKED: {proposal.justification}"
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
