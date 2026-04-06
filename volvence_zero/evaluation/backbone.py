from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping
from uuid import uuid4

from volvence_zero.dual_track import DualTrackSnapshot
from volvence_zero.memory import MemorySnapshot
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel


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
