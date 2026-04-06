from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping
from uuid import uuid4

from volvence_zero.dual_track import DualTrackSnapshot
from volvence_zero.evaluation import EvaluationSnapshot
from volvence_zero.memory import Track
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel


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
class CreditSnapshot:
    recent_credits: tuple[CreditRecord, ...]
    recent_modifications: tuple[SelfModificationRecord, ...]
    cumulative_credit_by_level: tuple[tuple[str, float], ...]
    description: str


@dataclass(frozen=True)
class ModificationProposal:
    target: str
    desired_gate: ModificationGate
    old_value_hash: str
    new_value_hash: str
    justification: str
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
    extra_records: tuple[CreditRecord, ...],
) -> CreditSnapshot:
    cumulative: dict[str, float] = dict(credit_snapshot.cumulative_credit_by_level)
    for record in extra_records:
        cumulative[record.level] = cumulative.get(record.level, 0.0) + record.credit_value
    recent_credits = tuple((credit_snapshot.recent_credits + extra_records)[-20:])
    return CreditSnapshot(
        recent_credits=recent_credits,
        recent_modifications=credit_snapshot.recent_modifications,
        cumulative_credit_by_level=tuple(sorted(cumulative.items())),
        description=(
            f"{credit_snapshot.description} Extended with {len(extra_records)} abstract-action credits."
        ),
    )


class CreditLedger:
    """Stores recent credit records and gate outcomes."""

    def __init__(self) -> None:
        self._recent_credits: list[CreditRecord] = []
        self._recent_modifications: list[SelfModificationRecord] = []

    @property
    def recent_credits(self) -> tuple[CreditRecord, ...]:
        return tuple(self._recent_credits)

    @property
    def recent_modifications(self) -> tuple[SelfModificationRecord, ...]:
        return tuple(self._recent_modifications)

    def record_credits(self, credits: tuple[CreditRecord, ...]) -> None:
        self._recent_credits.extend(credits)

    def record_modification(self, record: SelfModificationRecord) -> None:
        self._recent_modifications.append(record)

    def snapshot(self) -> CreditSnapshot:
        cumulative: dict[str, float] = {}
        for record in self._recent_credits:
            cumulative[record.level] = cumulative.get(record.level, 0.0) + record.credit_value
        return CreditSnapshot(
            recent_credits=tuple(self._recent_credits[-10:]),
            recent_modifications=tuple(self._recent_modifications[-10:]),
            cumulative_credit_by_level=tuple(sorted(cumulative.items())),
            description=(
                f"Credit ledger with {len(self._recent_credits)} credit records and "
                f"{len(self._recent_modifications)} modification records."
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
