from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Mapping
from uuid import uuid4

from volvence_zero.dual_track import DualTrackSnapshot
from volvence_zero.memory import MemorySnapshot
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel

if TYPE_CHECKING:
    from volvence_zero.temporal.interface import MetacontrollerRuntimeState


class EvaluationTrack(str, Enum):
    WORLD = "world"
    SELF = "self"
    CROSS = "cross"


@dataclass(frozen=True)
class EvaluationScore:
    family: str
    metric_name: str
    value: float
    confidence: float
    evidence: str


@dataclass(frozen=True)
class EvaluationSnapshot:
    turn_scores: tuple[EvaluationScore, ...]
    session_scores: tuple[EvaluationScore, ...]
    alerts: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class EvaluationRecord:
    record_id: str
    session_id: str
    wave_id: str | None
    timestamp_ms: int
    timescale: str
    family: str
    metric_name: str
    value: float
    confidence: float
    track: str
    evidence: str
    signal_sources: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationReport:
    report_id: str
    report_type: str
    timestamp_ms: int
    session_ids: tuple[str, ...]
    scores_by_family: tuple[tuple[str, tuple[EvaluationRecord, ...]], ...]
    alerts: tuple[tuple[str, str], ...]
    trends: tuple[tuple[str, str, float], ...]
    recommendations: tuple[str, ...]
    description: str


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


class EvaluationBackbone:
    """Minimal turn/session evaluator with evidence and alerts."""

    def __init__(self) -> None:
        self._records: list[EvaluationRecord] = []

    @property
    def records(self) -> tuple[EvaluationRecord, ...]:
        return tuple(self._records)

    def evaluate_turn(
        self,
        *,
        session_id: str,
        wave_id: str,
        timestamp_ms: int,
        memory_snapshot: MemorySnapshot | None,
        dual_track_snapshot: DualTrackSnapshot | None,
    ) -> EvaluationSnapshot:
        turn_scores = self._build_turn_scores(
            memory_snapshot=memory_snapshot,
            dual_track_snapshot=dual_track_snapshot,
        )
        alerts = self._build_alerts(turn_scores=turn_scores)
        self._append_records(
            session_id=session_id,
            wave_id=wave_id,
            timestamp_ms=timestamp_ms,
            timescale="turn",
            scores=turn_scores,
        )
        session_scores = self._session_scores_for(session_id=session_id)
        return EvaluationSnapshot(
            turn_scores=turn_scores,
            session_scores=session_scores,
            alerts=alerts,
            description=(
                f"Evaluation backbone produced {len(turn_scores)} turn scores and "
                f"{len(alerts)} alerts."
            ),
        )

    def build_session_report(self, *, session_id: str, timestamp_ms: int) -> EvaluationReport:
        session_records = tuple(record for record in self._records if record.session_id == session_id)
        scores_by_family: dict[str, list[EvaluationRecord]] = {}
        for record in session_records:
            scores_by_family.setdefault(record.family, []).append(record)

        alerts = self._alerts_from_records(session_records)
        recommendations = self._recommendations_from_alerts(alerts)
        return EvaluationReport(
            report_id=str(uuid4()),
            report_type="session",
            timestamp_ms=timestamp_ms,
            session_ids=(session_id,),
            scores_by_family=tuple(
                (family, tuple(records)) for family, records in sorted(scores_by_family.items())
            ),
            alerts=alerts,
            trends=(),
            recommendations=recommendations,
            description=f"Session evaluation report with {len(session_records)} records.",
        )

    def record_metacontroller_evidence(
        self,
        *,
        session_id: str,
        wave_id: str,
        timestamp_ms: int,
        metacontroller_state: "MetacontrollerRuntimeState | None",
        policy_objective: float,
        rollback_reasons: tuple[str, ...],
    ) -> tuple[EvaluationScore, ...]:
        if metacontroller_state is None:
            return ()
        adaptive_stability = _clamp(
            1.0
            - min(len(rollback_reasons) * 0.25, 0.75)
            - min(metacontroller_state.latest_ssl_loss * 0.15, 0.45)
        )
        posterior_stability = _clamp(1.0 - metacontroller_state.posterior_drift)
        switch_sparsity = _clamp(metacontroller_state.switch_sparsity)
        binary_gate_ratio = _clamp(metacontroller_state.binary_switch_rate)
        decoder_usefulness = _clamp(
            sum(metacontroller_state.decoder_applied_control) / max(len(metacontroller_state.decoder_applied_control), 1)
            - min(metacontroller_state.latest_ssl_loss * 0.08, 0.15)
            + 0.15
        )
        policy_replacement_quality = _clamp(
            0.5
            + metacontroller_state.policy_replacement_score * 0.35
            - min(len(rollback_reasons) * 0.08, 0.2)
        )
        abstract_action_usefulness = _clamp(
            0.5
            + policy_objective * 0.35
            - min(metacontroller_state.latest_ssl_loss * 0.1, 0.2)
        )
        scores = (
            EvaluationScore(
                family="learning",
                metric_name="adaptive_stability",
                value=adaptive_stability,
                confidence=0.6,
                evidence=(
                    f"Derived from ssl_loss={metacontroller_state.latest_ssl_loss:.3f} "
                    f"and rollback_count={len(rollback_reasons)}."
                ),
            ),
            EvaluationScore(
                family="abstraction",
                metric_name="posterior_stability",
                value=posterior_stability,
                confidence=0.57,
                evidence=(
                    f"Derived from posterior_drift={metacontroller_state.posterior_drift:.3f} "
                    f"and z_tilde={metacontroller_state.z_tilde}."
                ),
            ),
            EvaluationScore(
                family="abstraction",
                metric_name="switch_sparsity",
                value=switch_sparsity,
                confidence=0.55,
                evidence=(
                    f"Derived from latest_switch_gate={metacontroller_state.latest_switch_gate:.3f} "
                    f"and switch_sparsity={metacontroller_state.switch_sparsity:.3f}."
                ),
            ),
            EvaluationScore(
                family="abstraction",
                metric_name="binary_gate_ratio",
                value=binary_gate_ratio,
                confidence=0.55,
                evidence=(
                    f"Derived from beta_binary={metacontroller_state.beta_binary} "
                    f"and binary_switch_rate={metacontroller_state.binary_switch_rate:.3f}."
                ),
            ),
            EvaluationScore(
                family="learning",
                metric_name="decoder_usefulness",
                value=decoder_usefulness,
                confidence=0.56,
                evidence=(
                    f"Derived from decoder_applied_control={metacontroller_state.decoder_applied_control}."
                ),
            ),
            EvaluationScore(
                family="learning",
                metric_name="policy_replacement_quality",
                value=policy_replacement_quality,
                confidence=0.58,
                evidence=(
                    f"Derived from replacement_score={metacontroller_state.policy_replacement_score:.3f}."
                ),
            ),
            EvaluationScore(
                family="abstraction",
                metric_name="abstract_action_usefulness",
                value=abstract_action_usefulness,
                confidence=0.58,
                evidence=(
                    f"Derived from policy_objective={policy_objective:.3f} and "
                    f"active_label={metacontroller_state.active_label}."
                ),
            ),
        )
        self._append_records(
            session_id=session_id,
            wave_id=wave_id,
            timestamp_ms=timestamp_ms,
            timescale="session",
            scores=scores,
        )
        return scores

    def record_learning_evidence(
        self,
        *,
        session_id: str,
        wave_id: str,
        timestamp_ms: int,
        base_snapshot: EvaluationSnapshot,
        memory_snapshot: MemorySnapshot | None,
        reflection_snapshot: object | None,
        writeback_result: object | None,
        joint_loop_result: object | None,
    ) -> EvaluationSnapshot:
        scores = self._learning_evidence_scores(
            memory_snapshot=memory_snapshot,
            reflection_snapshot=reflection_snapshot,
            writeback_result=writeback_result,
            joint_loop_result=joint_loop_result,
        )
        if not scores:
            return base_snapshot
        self._append_records(
            session_id=session_id,
            wave_id=wave_id,
            timestamp_ms=timestamp_ms,
            timescale="turn",
            scores=scores,
        )
        turn_scores = base_snapshot.turn_scores + scores
        alerts = tuple(dict.fromkeys(base_snapshot.alerts + self._build_alerts(turn_scores=turn_scores)))
        return EvaluationSnapshot(
            turn_scores=turn_scores,
            session_scores=self._session_scores_for(session_id=session_id),
            alerts=alerts,
            description=f"{base_snapshot.description} Enriched with {len(scores)} learning-evidence scores.",
        )

    def _build_turn_scores(
        self,
        *,
        memory_snapshot: MemorySnapshot | None,
        dual_track_snapshot: DualTrackSnapshot | None,
    ) -> tuple[EvaluationScore, ...]:
        memory_count = len(memory_snapshot.retrieved_entries) if memory_snapshot else 0
        info_integration = _clamp(memory_count / 5.0)
        cross_tension = dual_track_snapshot.cross_track_tension if dual_track_snapshot else 0.0
        relationship_stability = _clamp(1.0 - cross_tension)
        warmth = _clamp(dual_track_snapshot.self_track.tension_level if dual_track_snapshot else 0.0)
        contract_integrity = 1.0

        return (
            EvaluationScore(
                family="task",
                metric_name="info_integration",
                value=info_integration,
                confidence=0.7,
                evidence=f"{memory_count} retrieved memory entries available to the turn.",
            ),
            EvaluationScore(
                family="interaction",
                metric_name="warmth",
                value=warmth,
                confidence=0.55,
                evidence="Derived from self-track tension as a first-pass interaction proxy.",
            ),
            EvaluationScore(
                family="relationship",
                metric_name="cross_track_stability",
                value=relationship_stability,
                confidence=0.65,
                evidence=f"Computed from cross_track_tension={cross_tension:.2f}.",
            ),
            EvaluationScore(
                family="safety",
                metric_name="contract_integrity",
                value=contract_integrity,
                confidence=0.9,
                evidence="No runtime contract violations surfaced during this turn evaluation.",
            ),
        )

    def _learning_evidence_scores(
        self,
        *,
        memory_snapshot: MemorySnapshot | None,
        reflection_snapshot: object | None,
        writeback_result: object | None,
        joint_loop_result: object | None,
    ) -> tuple[EvaluationScore, ...]:
        scores: list[EvaluationScore] = []
        if memory_snapshot is not None:
            retrieval_quality = _clamp(
                len(memory_snapshot.retrieved_entries) / 5.0 + (0.1 if memory_snapshot.cms_state is not None else 0.0)
            )
            scores.append(
                EvaluationScore(
                    family="learning",
                    metric_name="retrieval_quality",
                    value=retrieval_quality,
                    confidence=0.58,
                    evidence=(
                        f"Derived from retrieved_entries={len(memory_snapshot.retrieved_entries)} "
                        f"and cms_state={'present' if memory_snapshot.cms_state is not None else 'missing'}."
                    ),
                )
            )
        if reflection_snapshot is not None and hasattr(reflection_snapshot, "consolidation_score"):
            confidence = getattr(reflection_snapshot.consolidation_score, "confidence", 0.0)
            applied_count = len(getattr(writeback_result, "applied_operations", ())) if writeback_result is not None else 0
            blocked_count = len(getattr(writeback_result, "blocked_operations", ())) if writeback_result is not None else 0
            reflection_usefulness = _clamp(confidence + applied_count * 0.05 - blocked_count * 0.08)
            scores.append(
                EvaluationScore(
                    family="learning",
                    metric_name="reflection_usefulness",
                    value=reflection_usefulness,
                    confidence=0.57,
                    evidence=(
                        f"Derived from consolidation_confidence={confidence:.3f}, "
                        f"applied_operations={applied_count}, blocked_operations={blocked_count}."
                    ),
                )
            )
        if joint_loop_result is not None and hasattr(joint_loop_result, "schedule_action"):
            schedule_action = getattr(joint_loop_result, "schedule_action", "evidence-only")
            cycle_report = getattr(joint_loop_result, "cycle_report", None)
            if cycle_report is not None:
                joint_progress = _clamp(
                    0.5
                    + cycle_report.total_reward * 0.15
                    + cycle_report.policy_objective * 0.1
                    - len(cycle_report.rollback_reasons) * 0.08
                )
                evidence = (
                    f"Derived from action={schedule_action}, total_reward={cycle_report.total_reward:.3f}, "
                    f"policy_objective={cycle_report.policy_objective:.3f}."
                )
            elif schedule_action == "ssl-only":
                joint_progress = _clamp(
                    0.65
                    - min(getattr(joint_loop_result, "ssl_prediction_loss", 0.0) * 0.08, 0.2)
                    - min(getattr(joint_loop_result, "ssl_kl_loss", 0.0) * 0.05, 0.1)
                )
                evidence = (
                    f"Derived from action=ssl-only, pred={getattr(joint_loop_result, 'ssl_prediction_loss', 0.0):.3f}, "
                    f"kl={getattr(joint_loop_result, 'ssl_kl_loss', 0.0):.3f}."
                )
            else:
                joint_progress = 0.5
                evidence = f"Derived from action={schedule_action} without optimizer update."
            scores.append(
                EvaluationScore(
                    family="learning",
                    metric_name="joint_learning_progress",
                    value=joint_progress,
                    confidence=0.56,
                    evidence=evidence,
                )
            )
        return tuple(scores)

    def _build_alerts(self, *, turn_scores: tuple[EvaluationScore, ...]) -> tuple[str, ...]:
        alerts: list[str] = []
        for score in turn_scores:
            if score.family == "relationship" and score.value < 0.4:
                alerts.append("HIGH: cross-track stability is degraded")
            if score.family == "safety" and score.value < 0.95:
                alerts.append("HIGH: contract integrity below threshold")
        return tuple(alerts)

    def _append_records(
        self,
        *,
        session_id: str,
        wave_id: str,
        timestamp_ms: int,
        timescale: str,
        scores: tuple[EvaluationScore, ...],
    ) -> None:
        for score in scores:
            self._records.append(
                EvaluationRecord(
                    record_id=str(uuid4()),
                    session_id=session_id,
                    wave_id=wave_id,
                    timestamp_ms=timestamp_ms,
                    timescale=timescale,
                    family=score.family,
                    metric_name=score.metric_name,
                    value=score.value,
                    confidence=score.confidence,
                    track=self._track_for_family(score.family).value,
                    evidence=score.evidence,
                    signal_sources=self._signal_sources_for_metric(score.metric_name),
                )
            )

    def _session_scores_for(self, *, session_id: str) -> tuple[EvaluationScore, ...]:
        session_records = [record for record in self._records if record.session_id == session_id]
        grouped: dict[tuple[str, str], list[EvaluationRecord]] = {}
        for record in session_records:
            grouped.setdefault((record.family, record.metric_name), []).append(record)
        session_scores: list[EvaluationScore] = []
        for (family, metric_name), records in sorted(grouped.items()):
            average = sum(record.value for record in records) / len(records)
            confidence = sum(record.confidence for record in records) / len(records)
            session_scores.append(
                EvaluationScore(
                    family=family,
                    metric_name=metric_name,
                    value=round(average, 4),
                    confidence=round(confidence, 4),
                    evidence=f"Session aggregate over {len(records)} turn records.",
                )
            )
        return tuple(session_scores)

    def _track_for_family(self, family: str) -> EvaluationTrack:
        if family == "task":
            return EvaluationTrack.WORLD
        if family == "relationship":
            return EvaluationTrack.SELF
        return EvaluationTrack.CROSS

    def _signal_sources_for_metric(self, metric_name: str) -> tuple[str, ...]:
        if metric_name == "info_integration":
            return ("memory.retrieved_entries",)
        if metric_name == "warmth":
            return ("dual_track.self_track.tension_level",)
        if metric_name == "cross_track_stability":
            return ("dual_track.cross_track_tension",)
        if metric_name == "adaptive_stability":
            return ("temporal.metacontroller_state", "joint_loop.rollback_reasons")
        if metric_name == "posterior_stability":
            return ("temporal.metacontroller_state.posterior",)
        if metric_name in {"switch_sparsity", "binary_gate_ratio"}:
            return ("temporal.metacontroller_state.switch",)
        if metric_name == "decoder_usefulness":
            return ("temporal.metacontroller_state.decoder",)
        if metric_name in {"abstract_action_usefulness", "policy_replacement_quality"}:
            return ("temporal.metacontroller_state", "internal_rl.policy_objective")
        return ("runtime.contract_status",)

    def _alerts_from_records(
        self,
        records: tuple[EvaluationRecord, ...],
    ) -> tuple[tuple[str, str], ...]:
        alerts: list[tuple[str, str]] = []
        for record in records:
            if record.family == "relationship" and record.value < 0.4:
                alerts.append(("HIGH", f"{record.metric_name} below threshold"))
        return tuple(alerts)

    def _recommendations_from_alerts(
        self,
        alerts: tuple[tuple[str, str], ...],
    ) -> tuple[str, ...]:
        if not alerts:
            return ("Continue collecting turn-level evidence before widening scope.",)
        return tuple(f"Review alert: {message}" for _, message in alerts)


class EvaluationModule(RuntimeModule[EvaluationSnapshot]):
    slot_name = "evaluation"
    owner = "EvaluationModule"
    value_type = EvaluationSnapshot
    dependencies = ("memory", "dual_track")
    default_wiring_level = WiringLevel.ACTIVE

    def __init__(
        self,
        *,
        backbone: EvaluationBackbone | None = None,
        session_id: str = "runtime-session",
        wave_id: str = "wave-0",
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._backbone = backbone or EvaluationBackbone()
        self._session_id = session_id
        self._wave_id = wave_id

    @property
    def backbone(self) -> EvaluationBackbone:
        return self._backbone

    async def process(self, upstream: Mapping[str, Snapshot[object]]) -> Snapshot[EvaluationSnapshot]:
        memory_snapshot = upstream["memory"]
        dual_track_snapshot = upstream["dual_track"]
        memory_value = memory_snapshot.value if isinstance(memory_snapshot.value, MemorySnapshot) else None
        dual_track_value = (
            dual_track_snapshot.value if isinstance(dual_track_snapshot.value, DualTrackSnapshot) else None
        )
        return self.publish(
            self._backbone.evaluate_turn(
                session_id=self._session_id,
                wave_id=self._wave_id,
                timestamp_ms=max(memory_snapshot.timestamp_ms, dual_track_snapshot.timestamp_ms),
                memory_snapshot=memory_value,
                dual_track_snapshot=dual_track_value,
            )
        )

    async def process_standalone(self, **kwargs: object) -> Snapshot[EvaluationSnapshot]:
        session_id = str(kwargs.get("session_id", self._session_id))
        wave_id = str(kwargs.get("wave_id", self._wave_id))
        timestamp_ms = int(kwargs.get("timestamp_ms", 1))
        memory_snapshot = kwargs.get("memory_snapshot")
        dual_track_snapshot = kwargs.get("dual_track_snapshot")
        return self.publish(
            self._backbone.evaluate_turn(
                session_id=session_id,
                wave_id=wave_id,
                timestamp_ms=timestamp_ms,
                memory_snapshot=memory_snapshot if isinstance(memory_snapshot, MemorySnapshot) else None,
                dual_track_snapshot=(
                    dual_track_snapshot
                    if isinstance(dual_track_snapshot, DualTrackSnapshot)
                    else None
                ),
            )
        )
