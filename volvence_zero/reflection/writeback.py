from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping
from uuid import uuid4

from volvence_zero.credit import CreditSnapshot, GateDecision
from volvence_zero.dual_track import DualTrackSnapshot
from volvence_zero.evaluation import EvaluationSnapshot
from volvence_zero.memory import (
    MemoryEntry,
    MemorySnapshot,
    MemoryStore,
    MemoryStoreCheckpoint,
    MemoryStratum,
    Track,
)
from volvence_zero.regime import RegimeCheckpoint, RegimeModule, RegimeSnapshot
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel


class WritebackMode(str, Enum):
    DISABLED = "disabled"
    PROPOSAL_ONLY = "proposal-only"
    APPLY = "apply"


@dataclass(frozen=True)
class MemoryConsolidation:
    new_durable_entries: tuple[MemoryEntry, ...]
    promoted_entries: tuple[str, ...]
    decayed_entries: tuple[str, ...]
    beliefs_updated: tuple[str, ...]


@dataclass(frozen=True)
class PolicyConsolidation:
    controller_updates: tuple[str, ...]
    strategy_priors_updated: tuple[str, ...]
    regime_effectiveness_updated: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class ReflectionSnapshot:
    memory_consolidation: MemoryConsolidation
    policy_consolidation: PolicyConsolidation
    interaction_trace_summary: str
    tensions_identified: tuple[str, ...]
    lessons_extracted: tuple[str, ...]
    writeback_mode: str
    review_required: bool
    description: str


@dataclass(frozen=True)
class WritebackCheckpoint:
    checkpoint_id: str
    memory_checkpoint: MemoryStoreCheckpoint
    regime_checkpoint: RegimeCheckpoint | None = None


@dataclass(frozen=True)
class WritebackResult:
    applied_operations: tuple[str, ...]
    blocked_operations: tuple[str, ...]
    checkpoint: WritebackCheckpoint | None
    description: str


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


class ReflectionEngine:
    """Builds proposal-first slow reflection artifacts from current session state."""

    def __init__(self, *, writeback_mode: WritebackMode = WritebackMode.PROPOSAL_ONLY) -> None:
        self._writeback_mode = writeback_mode

    @property
    def writeback_mode(self) -> WritebackMode:
        return self._writeback_mode

    def reflect(
        self,
        *,
        timestamp_ms: int,
        memory_snapshot: MemorySnapshot | None,
        dual_track_snapshot: DualTrackSnapshot | None,
        evaluation_snapshot: EvaluationSnapshot | None,
        credit_snapshot: CreditSnapshot | None,
        regime_snapshot: RegimeSnapshot | None = None,
    ) -> ReflectionSnapshot:
        memory_consolidation = self._memory_consolidation(
            timestamp_ms=timestamp_ms,
            memory_snapshot=memory_snapshot,
            credit_snapshot=credit_snapshot,
        )
        policy_consolidation = self._policy_consolidation(
            dual_track_snapshot=dual_track_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            credit_snapshot=credit_snapshot,
            regime_snapshot=regime_snapshot,
        )
        tensions = self._tensions(dual_track_snapshot=dual_track_snapshot, evaluation_snapshot=evaluation_snapshot)
        lessons = self._lessons(
            memory_consolidation=memory_consolidation,
            policy_consolidation=policy_consolidation,
            tensions=tensions,
        )
        trace_summary = self._interaction_trace_summary(
            memory_snapshot=memory_snapshot,
            dual_track_snapshot=dual_track_snapshot,
            evaluation_snapshot=evaluation_snapshot,
        )
        review_required = self._writeback_mode is not WritebackMode.APPLY
        description = (
            f"Reflection generated {len(memory_consolidation.new_durable_entries)} durable proposals, "
            f"{len(policy_consolidation.strategy_priors_updated)} strategy updates, "
            f"mode={self._writeback_mode.value}."
        )
        return ReflectionSnapshot(
            memory_consolidation=memory_consolidation,
            policy_consolidation=policy_consolidation,
            interaction_trace_summary=trace_summary,
            tensions_identified=tensions,
            lessons_extracted=lessons,
            writeback_mode=self._writeback_mode.value,
            review_required=review_required,
            description=description,
        )

    def apply(
        self,
        *,
        memory_store: MemoryStore,
        reflection_snapshot: ReflectionSnapshot,
        credit_snapshot: CreditSnapshot | None,
        regime_module: RegimeModule | None = None,
        checkpoint_id: str | None = None,
    ) -> WritebackResult:
        if self._writeback_mode is not WritebackMode.APPLY:
            return WritebackResult(
                applied_operations=(),
                blocked_operations=("writeback-mode-not-apply",),
                checkpoint=None,
                description="Writeback blocked because reflection engine is not in APPLY mode.",
            )
        if credit_snapshot is not None and any(
            record.decision is GateDecision.BLOCK for record in credit_snapshot.recent_modifications
        ):
            return WritebackResult(
                applied_operations=(),
                blocked_operations=("credit-gate-block",),
                checkpoint=None,
                description="Writeback blocked by credit gate evidence.",
            )
        memory_checkpoint = memory_store.create_checkpoint(checkpoint_id=checkpoint_id)
        regime_checkpoint = (
            regime_module.create_checkpoint(checkpoint_id=checkpoint_id or "regime-writeback")
            if regime_module is not None
            else None
        )
        applied_operations = memory_store.apply_reflection_consolidation(
            new_durable_entries=reflection_snapshot.memory_consolidation.new_durable_entries,
            promoted_entries=reflection_snapshot.memory_consolidation.promoted_entries,
            decayed_entries=reflection_snapshot.memory_consolidation.decayed_entries,
            lesson_count=len(reflection_snapshot.lessons_extracted),
            timestamp_ms=max(
                (entry.created_at_ms for entry in reflection_snapshot.memory_consolidation.new_durable_entries),
                default=1,
            ),
        )
        threshold_delta = -0.05 if not reflection_snapshot.tensions_identified else 0.05
        applied_operations = applied_operations + (
            memory_store.apply_promotion_threshold_update(delta=threshold_delta),
        )
        if regime_module is not None:
            applied_operations = applied_operations + regime_module.apply_policy_consolidation(
                strategy_updates=reflection_snapshot.policy_consolidation.strategy_priors_updated,
                regime_effectiveness_updates=reflection_snapshot.policy_consolidation.regime_effectiveness_updated,
            )
        checkpoint = WritebackCheckpoint(
            checkpoint_id=memory_checkpoint.checkpoint_id,
            memory_checkpoint=memory_checkpoint,
            regime_checkpoint=regime_checkpoint,
        )
        return WritebackResult(
            applied_operations=applied_operations,
            blocked_operations=(),
            checkpoint=checkpoint,
            description=(
                f"Applied {len(applied_operations)} bounded reflection operations with "
                f"checkpoint {checkpoint.checkpoint_id}."
            ),
        )

    def rollback(
        self,
        *,
        memory_store: MemoryStore,
        checkpoint: WritebackCheckpoint,
        regime_module: RegimeModule | None = None,
    ) -> None:
        memory_store.restore_checkpoint(checkpoint.memory_checkpoint)
        if regime_module is not None and checkpoint.regime_checkpoint is not None:
            regime_module.restore_checkpoint(checkpoint.regime_checkpoint)

    def _memory_consolidation(
        self,
        *,
        timestamp_ms: int,
        memory_snapshot: MemorySnapshot | None,
        credit_snapshot: CreditSnapshot | None,
    ) -> MemoryConsolidation:
        if memory_snapshot is None:
            return MemoryConsolidation((), (), (), ())

        promoted_entries: list[str] = []
        new_durable_entries: list[MemoryEntry] = []
        decayed_entries: list[str] = []
        beliefs_updated: list[str] = []

        for entry in memory_snapshot.retrieved_entries[:3]:
            if entry.stratum != MemoryStratum.DURABLE.value:
                promoted_entries.append(entry.entry_id)
                new_durable_entries.append(
                    MemoryEntry(
                        entry_id=str(uuid4()),
                        content=entry.content,
                        track=entry.track,
                        stratum=MemoryStratum.DURABLE.value,
                        created_at_ms=timestamp_ms,
                        last_accessed_ms=timestamp_ms,
                        strength=_clamp(max(entry.strength, 0.7)),
                        tags=entry.tags,
                    )
                )
        if credit_snapshot is not None:
            for record in credit_snapshot.recent_credits[:3]:
                if record.credit_value > 0.6:
                    beliefs_updated.append(f"reinforce:{record.track.value}:{record.source_event}")
                if record.credit_value < 0.2:
                    decayed_entries.append(record.record_id)
        return MemoryConsolidation(
            new_durable_entries=tuple(new_durable_entries),
            promoted_entries=tuple(promoted_entries),
            decayed_entries=tuple(decayed_entries),
            beliefs_updated=tuple(beliefs_updated),
        )

    def _policy_consolidation(
        self,
        *,
        dual_track_snapshot: DualTrackSnapshot | None,
        evaluation_snapshot: EvaluationSnapshot | None,
        credit_snapshot: CreditSnapshot | None,
        regime_snapshot: RegimeSnapshot | None,
    ) -> PolicyConsolidation:
        controller_updates: list[str] = []
        strategy_priors_updated: list[str] = []
        regime_effectiveness_updated: list[tuple[str, float]] = []

        if dual_track_snapshot is not None:
            if dual_track_snapshot.cross_track_tension > 0.5:
                controller_updates.append("reduce_cross_track_tension_before_widening_scope")
            if dual_track_snapshot.self_track.tension_level > dual_track_snapshot.world_track.tension_level:
                strategy_priors_updated.append("increase_self_track_priority")
            elif dual_track_snapshot.world_track.tension_level > 0:
                strategy_priors_updated.append("increase_world_track_priority")

        if evaluation_snapshot is not None:
            for score in evaluation_snapshot.session_scores[:4]:
                if regime_snapshot is not None and score.family in {"relationship", "task"}:
                    regime_effectiveness_updated.append(
                        (regime_snapshot.active_regime.regime_id, score.value)
                    )

        if credit_snapshot is not None and credit_snapshot.recent_modifications:
            controller_updates.append("gate_audit_available_for_follow_up")

        return PolicyConsolidation(
            controller_updates=tuple(controller_updates),
            strategy_priors_updated=tuple(strategy_priors_updated),
            regime_effectiveness_updated=tuple(regime_effectiveness_updated),
        )

    def _tensions(
        self,
        *,
        dual_track_snapshot: DualTrackSnapshot | None,
        evaluation_snapshot: EvaluationSnapshot | None,
    ) -> tuple[str, ...]:
        tensions: list[str] = []
        if dual_track_snapshot is not None and dual_track_snapshot.cross_track_tension > 0.4:
            tensions.append("cross_track_tension_high")
        if evaluation_snapshot is not None:
            tensions.extend(alert for alert in evaluation_snapshot.alerts[:3])
        return tuple(tensions)

    def _lessons(
        self,
        *,
        memory_consolidation: MemoryConsolidation,
        policy_consolidation: PolicyConsolidation,
        tensions: tuple[str, ...],
    ) -> tuple[str, ...]:
        lessons: list[str] = []
        if memory_consolidation.new_durable_entries:
            lessons.append("promote_high_signal_memories")
        if policy_consolidation.strategy_priors_updated:
            lessons.append("adjust_track_priority_from_session_feedback")
        if tensions:
            lessons.append("review_tension_before_auto_writeback")
        return tuple(lessons)

    def _interaction_trace_summary(
        self,
        *,
        memory_snapshot: MemorySnapshot | None,
        dual_track_snapshot: DualTrackSnapshot | None,
        evaluation_snapshot: EvaluationSnapshot | None,
    ) -> str:
        memory_count = len(memory_snapshot.retrieved_entries) if memory_snapshot else 0
        cross_tension = dual_track_snapshot.cross_track_tension if dual_track_snapshot else 0.0
        alert_count = len(evaluation_snapshot.alerts) if evaluation_snapshot else 0
        return (
            f"Reflection input summary: retrieved_entries={memory_count}, "
            f"cross_track_tension={cross_tension:.2f}, alerts={alert_count}."
        )


class ReflectionModule(RuntimeModule[ReflectionSnapshot]):
    slot_name = "reflection"
    owner = "ReflectionModule"
    value_type = ReflectionSnapshot
    dependencies = ("memory", "dual_track", "evaluation", "regime", "credit")
    default_wiring_level = WiringLevel.DISABLED

    def __init__(
        self,
        *,
        engine: ReflectionEngine | None = None,
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._engine = engine or ReflectionEngine()

    @property
    def engine(self) -> ReflectionEngine:
        return self._engine

    async def process(self, upstream: Mapping[str, Snapshot[object]]) -> Snapshot[ReflectionSnapshot]:
        memory_snapshot = upstream["memory"]
        dual_track_snapshot = upstream["dual_track"]
        evaluation_snapshot = upstream["evaluation"]
        credit_snapshot = upstream["credit"]
        regime_snapshot = upstream["regime"]
        return self.publish(
            self._engine.reflect(
                timestamp_ms=max(
                    memory_snapshot.timestamp_ms,
                    dual_track_snapshot.timestamp_ms,
                    evaluation_snapshot.timestamp_ms,
                    credit_snapshot.timestamp_ms,
                ),
                memory_snapshot=memory_snapshot.value
                if isinstance(memory_snapshot.value, MemorySnapshot)
                else None,
                dual_track_snapshot=dual_track_snapshot.value
                if isinstance(dual_track_snapshot.value, DualTrackSnapshot)
                else None,
                evaluation_snapshot=evaluation_snapshot.value
                if isinstance(evaluation_snapshot.value, EvaluationSnapshot)
                else None,
                credit_snapshot=credit_snapshot.value
                if isinstance(credit_snapshot.value, CreditSnapshot)
                else None,
                regime_snapshot=regime_snapshot.value if isinstance(regime_snapshot.value, RegimeSnapshot) else None,
            )
        )

    async def process_standalone(self, **kwargs: object) -> Snapshot[ReflectionSnapshot]:
        timestamp_ms = int(kwargs.get("timestamp_ms", 1))
        return self.publish(
            self._engine.reflect(
                timestamp_ms=timestamp_ms,
                memory_snapshot=kwargs.get("memory_snapshot")
                if isinstance(kwargs.get("memory_snapshot"), MemorySnapshot)
                else None,
                dual_track_snapshot=kwargs.get("dual_track_snapshot")
                if isinstance(kwargs.get("dual_track_snapshot"), DualTrackSnapshot)
                else None,
                evaluation_snapshot=kwargs.get("evaluation_snapshot")
                if isinstance(kwargs.get("evaluation_snapshot"), EvaluationSnapshot)
                else None,
                credit_snapshot=kwargs.get("credit_snapshot")
                if isinstance(kwargs.get("credit_snapshot"), CreditSnapshot)
                else None,
                regime_snapshot=kwargs.get("regime_snapshot")
                if isinstance(kwargs.get("regime_snapshot"), RegimeSnapshot)
                else None,
            )
        )
