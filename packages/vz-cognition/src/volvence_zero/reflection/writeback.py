from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping
from uuid import uuid4

import json

from volvence_zero.credit.gate import CreditSnapshot, GateDecision
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeKind,
    DialogueExternalOutcomeSnapshot,
)
from volvence_zero.dual_track import DualTrackSnapshot
from volvence_zero.evaluation.types import EvaluationSnapshot
from volvence_zero.memory import (
    MemoryEntry,
    MemorySnapshot,
    MemoryStore,
    MemoryStoreCheckpoint,
    MemoryStratum,
    Track,
)
from volvence_zero.prediction.error import PredictionErrorSnapshot
from volvence_zero.regime import RegimeCheckpoint, RegimeSnapshot
from volvence_zero.rupture_state import RuptureKind, RuptureStateSnapshot
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel


class WritebackMode(str, Enum):
    DISABLED = "disabled"
    PROPOSAL_ONLY = "proposal-only"
    APPLY = "apply"


class ReflectionTensionId(str, Enum):
    """Stable identifiers for reflection-detected tensions.

    These are the canonical (single-source) string ids that reflection
    publishes in :attr:`ReflectionSnapshot.tensions_identified` and that
    downstream consumers (expression, evaluation) read by enum key
    instead of bare-string substring comparison. Adding a new tension
    requires landing a member here AND a corresponding hint entry in
    ``lifeform_expression.reflection_hints``.
    """

    PE_RELATIONSHIP_MISMATCH = "prediction_error_relationship_mismatch"
    PE_TASK_MISMATCH = "prediction_error_task_mismatch"
    PE_ACTION_INSTABILITY = "prediction_error_action_instability"
    PE_REGIME_INSTABILITY = "prediction_error_regime_instability"
    CROSS_TRACK_TENSION_HIGH = "cross_track_tension_high"
    CROSS_TRACK_ALIGNMENT_DRIFT = "cross_track_alignment_drift"
    SELF_TRACK_PRESSURE_DOMINANT = "self_track_pressure_dominant"
    WORLD_TRACK_PRESSURE_DOMINANT = "world_track_pressure_dominant"
    RELATIONSHIP_STABILITY_SOFT_DROP = "relationship_stability_soft_drop"
    WARMTH_SIGNAL_THIN = "warmth_signal_thin"
    TASK_SIGNAL_DIFFUSE = "task_signal_diffuse"


class ReflectionLessonId(str, Enum):
    """Stable identifiers for reflection-extracted lessons.

    Same role as :class:`ReflectionTensionId` for the
    :attr:`ReflectionSnapshot.lessons_extracted` channel.
    """

    RELATIONSHIP_STRATEGY_MISMATCH = "relationship_strategy_mismatch"
    TASK_FRAMING_INADEQUATE = "task_framing_inadequate"
    ABSTRACT_ACTION_INSTABILITY = "abstract_action_instability"
    REGIME_SELECTION_MISMATCH = "regime_selection_mismatch"
    PROMOTE_HIGH_SIGNAL_MEMORIES = "promote_high_signal_memories"
    REINFORCE_RECENT_HIGH_CREDIT_BELIEFS = "reinforce_recent_high_credit_beliefs"
    ADJUST_TRACK_PRIORITY_FROM_SESSION_FEEDBACK = (
        "adjust_track_priority_from_session_feedback"
    )
    REBALANCE_TEMPORAL_PRIOR_TOWARD_MEMORY = "rebalance_temporal_prior_toward_memory"
    REBALANCE_TEMPORAL_PRIOR_TOWARD_REFLECTION = (
        "rebalance_temporal_prior_toward_reflection"
    )
    REBALANCE_TEMPORAL_PRIOR_TOWARD_RESIDUAL = (
        "rebalance_temporal_prior_toward_residual"
    )
    INCREASE_CONTROLLER_PERSISTENCE_FOR_CONTINUITY = (
        "increase_controller_persistence_for_continuity"
    )
    REDUCE_CONTROLLER_PERSISTENCE_FOR_FASTER_RECOVERY = (
        "reduce_controller_persistence_for_faster_recovery"
    )
    ALLOW_CONTROLLER_SWITCH_WHEN_CONTEXT_SHIFTS = (
        "allow_controller_switch_when_context_shifts"
    )
    HOLD_CONTROLLER_BEFORE_SWITCHING = "hold_controller_before_switching"
    RESTRUCTURE_ACTION_FAMILY_BANK = "restructure_action_family_bank"
    RESPECT_METACONTROLLER_RUNTIME_GUARD = "respect_metacontroller_runtime_guard"
    KEEP_CONTROLLER_GUARD_SIGNAL_IN_BACKGROUND = (
        "keep_controller_guard_signal_in_background"
    )
    REVIEW_TENSION_BEFORE_AUTO_WRITEBACK = "review_tension_before_auto_writeback"


# Canonical tuple of the four PE-driven lessons. Used by
# ``_count_error_driven_lessons`` and (defensively) by external
# consumers that want to know which lessons are derived from
# prediction-error magnitudes vs. reflection heuristics.
PE_DERIVED_LESSON_IDS: frozenset[ReflectionLessonId] = frozenset(
    {
        ReflectionLessonId.RELATIONSHIP_STRATEGY_MISMATCH,
        ReflectionLessonId.TASK_FRAMING_INADEQUATE,
        ReflectionLessonId.ABSTRACT_ACTION_INSTABILITY,
        ReflectionLessonId.REGIME_SELECTION_MISMATCH,
    }
)


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
    temporal_prior_update: "TemporalPriorUpdate | None" = None
    controller_guard_blocked: bool = False
    controller_guard_audit_present: bool = False


@dataclass(frozen=True)
class EvidencePack:
    source_benchmark_ids: tuple[str, ...]
    delayed_credit_summary: tuple[tuple[str, float], ...]
    session_trend: tuple[tuple[str, float], ...]
    confidence: float
    supporting_cycles: int


@dataclass(frozen=True)
class TemporalStructureProposal:
    proposal_type: str
    family_id: str
    related_family_id: str | None
    confidence: float
    justification: str
    scope: str = "single_family"
    evidence: EvidencePack | None = None


@dataclass(frozen=True)
class ProposalEvidencePack:
    benchmark_passed: bool | None
    delayed_credit_summary: tuple[tuple[str, float], ...]
    session_trend_delta: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class StructuralProposalBundle:
    proposals: tuple[TemporalStructureProposal, ...]
    evidence_pack: ProposalEvidencePack
    scope: str
    bundle_confidence: float


@dataclass(frozen=True)
class ProposalOutcomeEntry:
    bundle_scope: str
    proposal_types: tuple[str, ...]
    bundle_confidence: float
    pre_metric_snapshot: tuple[tuple[str, float], ...]
    post_metric_snapshot: tuple[tuple[str, float], ...]
    metric_delta: float
    success: bool


@dataclass(frozen=True)
class TemporalPriorUpdate:
    target: str
    target_groups: tuple[str, ...]
    residual_strength: float
    memory_strength: float
    reflection_strength: float
    switch_bias_delta: float
    persistence_delta: float
    learning_rate_delta: float
    description: str
    encoder_strength_delta: float = 0.0
    decoder_strength_delta: float = 0.0
    world_track_delta: float = 0.0
    self_track_delta: float = 0.0
    shared_track_delta: float = 0.0
    beta_threshold_delta: float = 0.0
    family_stability_delta: float = 0.0
    structure_proposals: tuple[TemporalStructureProposal, ...] = ()
    structure_bundle: "StructuralProposalBundle | None" = None


@dataclass(frozen=True)
class ConsolidationScore:
    promotion_score: float
    decay_score: float
    threshold_delta: float
    strategy_gain: float
    regime_effectiveness_gain: float
    confidence: float
    description: str


@dataclass(frozen=True)
class ReflectionSnapshot:
    memory_consolidation: MemoryConsolidation
    policy_consolidation: PolicyConsolidation
    consolidation_score: ConsolidationScore
    interaction_trace_summary: str
    tensions_identified: tuple[str, ...]
    lessons_extracted: tuple[str, ...]
    writeback_mode: str
    review_required: bool
    description: str
    proposal_success_rate: float = 0.0
    primary_prediction_error_dimension: str | None = None
    repeated_prediction_failures: tuple[str, ...] = ()
    error_driven_lesson_count: int = 0


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


def _metric(evaluation_snapshot: EvaluationSnapshot | None, metric_name: str, default: float = 0.0) -> float:
    if evaluation_snapshot is None:
        return default
    for score in evaluation_snapshot.turn_scores:
        if score.metric_name == metric_name:
            return score.value
    return default


def _relationship_relevant_alerts(evaluation_snapshot: EvaluationSnapshot | None) -> tuple[str, ...]:
    if evaluation_snapshot is None:
        return ()
    return tuple(
        alert.legacy_text
        for alert in evaluation_snapshot.structured_alerts
        if alert.code in {"cross_track_stability_degraded", "rollback_pressure_elevated"}
    )


def _primary_prediction_error_dimension(
    prediction_error_snapshot: PredictionErrorSnapshot | None,
) -> str | None:
    if prediction_error_snapshot is None:
        return None
    error = prediction_error_snapshot.error
    candidates = (
        ("task", abs(error.task_error)),
        ("relationship", abs(error.relationship_error)),
        ("regime", abs(error.regime_error)),
        ("action", abs(error.action_error)),
    )
    name, magnitude = max(candidates, key=lambda item: item[1])
    return name if magnitude > 0.05 else None


def _repeated_prediction_failures(
    prediction_error_snapshot: PredictionErrorSnapshot | None,
) -> tuple[str, ...]:
    primary = _primary_prediction_error_dimension(prediction_error_snapshot)
    if primary is None:
        return ()
    error = prediction_error_snapshot.error
    failures: list[str] = []
    if primary == "relationship" and error.relationship_error < -0.05:
        failures.append("prediction_error:relationship")
    if primary == "task" and error.task_error < -0.05:
        failures.append("prediction_error:task")
    if primary == "action" and abs(error.action_error) > 0.20:
        failures.append("prediction_error:action")
    if primary == "regime" and error.regime_error < -0.05:
        failures.append("prediction_error:regime")
    return tuple(failures)


def _prediction_error_tensions(
    prediction_error_snapshot: PredictionErrorSnapshot | None,
) -> tuple[str, ...]:
    if prediction_error_snapshot is None:
        return ()
    error = prediction_error_snapshot.error
    tensions: list[str] = []
    if error.relationship_error < -0.10:
        tensions.append(ReflectionTensionId.PE_RELATIONSHIP_MISMATCH.value)
    if error.task_error < -0.10:
        tensions.append(ReflectionTensionId.PE_TASK_MISMATCH.value)
    if abs(error.action_error) > 0.20:
        tensions.append(ReflectionTensionId.PE_ACTION_INSTABILITY.value)
    if error.regime_error < -0.10:
        tensions.append(ReflectionTensionId.PE_REGIME_INSTABILITY.value)
    return tuple(tensions)


def _prediction_error_lessons(
    prediction_error_snapshot: PredictionErrorSnapshot | None,
) -> tuple[str, ...]:
    if prediction_error_snapshot is None:
        return ()
    error = prediction_error_snapshot.error
    lessons: list[str] = []
    if error.relationship_error < -0.10:
        lessons.append(ReflectionLessonId.RELATIONSHIP_STRATEGY_MISMATCH.value)
    if error.task_error < -0.10:
        lessons.append(ReflectionLessonId.TASK_FRAMING_INADEQUATE.value)
    if abs(error.action_error) > 0.20:
        lessons.append(ReflectionLessonId.ABSTRACT_ACTION_INSTABILITY.value)
    if error.regime_error < -0.10:
        lessons.append(ReflectionLessonId.REGIME_SELECTION_MISMATCH.value)
    return tuple(lessons)


def _count_error_driven_lessons(lessons: tuple[str, ...]) -> int:
    pe_lesson_values = {item.value for item in PE_DERIVED_LESSON_IDS}
    return sum(1 for lesson in lessons if lesson in pe_lesson_values)


# Positive external outcome kinds that resolve a rupture-repair as observed
# (i.e. "the repair landed"). Negative kinds keep the pair as "pending".
_REPAIR_POSITIVE_KINDS: frozenset[DialogueExternalOutcomeKind] = frozenset(
    {
        DialogueExternalOutcomeKind.HELPED,
        DialogueExternalOutcomeKind.FELT_HEARD,
        DialogueExternalOutcomeKind.DECISION_CLEARER,
    }
)


def rupture_repair_memory_entries(
    *,
    rupture_state_snapshot: RuptureStateSnapshot | None,
    external_outcome_snapshot: DialogueExternalOutcomeSnapshot | None,
    regime_snapshot: RegimeSnapshot | None,
    user_scope: str,
    timestamp_ms: int,
) -> tuple[MemoryEntry, ...]:
    """Build durable rupture-repair ``MemoryEntry`` objects for reflection.

    Returns a 0 or 1 tuple. The entry is only produced when *both* a
    ``rupture_kind`` has been resolved (i.e. externally confirmed) on
    this turn AND at least one externally-confirmed outcome contributed
    to that turn's external-outcome snapshot. This keeps the v0 rule
    that a durable rupture-repair write requires typed external
    confirmation.

    The returned entry follows the tag schema defined in
    ``docs/DATA_CONTRACT.md`` §3.3 "Rupture-repair 记忆 tag schema":

    * tags: ``rupture_repair``, ``rupture_kind:<kind>``,
      ``repair_outcome:<observed|pending>``, ``user_scope:<user>``,
      ``source_wave:<wave_id>``;
    * ``content``: structured JSON with ``rupture_kind``,
      ``repair_move``, ``source_turn_index``, ``source_wave_id``,
      ``observed_outcome_kind``, ``confidence``.

    This function is pure; it does NOT write to ``memory_store``. The
    only write path is ``ReflectionEngine.apply`` -> ``memory_store``.
    """

    if rupture_state_snapshot is None or rupture_state_snapshot.rupture_kind is None:
        return ()
    if external_outcome_snapshot is None or not external_outcome_snapshot.entries:
        return ()
    rupture_kind: RuptureKind = rupture_state_snapshot.rupture_kind

    # Identify the externally-confirmed entry that names the rupture. It
    # is the first non-LLM entry whose kind maps into this rupture
    # category; positive entries on the same turn indicate the user
    # also reported HELPED/CLEARER, so repair_outcome becomes observed.
    from volvence_zero.rupture_state.contracts import (
        EXTERNAL_OUTCOME_TO_RUPTURE_KIND,
    )

    triggering_entry = None
    positive_entry = None
    for entry in external_outcome_snapshot.entries:
        mapped = EXTERNAL_OUTCOME_TO_RUPTURE_KIND.get(entry.kind)
        if mapped is rupture_kind and triggering_entry is None:
            triggering_entry = entry
        if entry.kind in _REPAIR_POSITIVE_KINDS and positive_entry is None:
            positive_entry = entry
    if triggering_entry is None:
        # The rupture kind was resolved (e.g. via composition), but no
        # single entry directly produced it. In that case use the first
        # non-LLM external entry as provenance.
        for entry in external_outcome_snapshot.entries:
            if entry.source.value != "llm_proposal":
                triggering_entry = entry
                break
    if triggering_entry is None:
        return ()

    repair_outcome = "observed" if positive_entry is not None else "pending"
    observed_outcome_kind = (
        positive_entry.kind.value
        if positive_entry is not None
        else triggering_entry.kind.value
    )
    source_turn_index = int(triggering_entry.turn_index)
    source_wave_id = f"wave-{source_turn_index}"
    confidence = float(rupture_state_snapshot.confidence)

    regime_id = ""
    abstract_action = ""
    action_family_version = 0
    if regime_snapshot is not None:
        regime_id = str(regime_snapshot.active_regime.regime_id)
        for attribution in regime_snapshot.delayed_attributions:
            # Prefer attributions matching our source wave id so the
            # content carries a consistent abstract action context.
            if attribution.source_wave_id.endswith(triggering_entry.evidence_id):
                abstract_action = str(attribution.abstract_action or "")
                action_family_version = int(attribution.action_family_version or 0)
                break

    content_obj = {
        "rupture_kind": rupture_kind.value,
        "repair_move": "",  # v0 placeholder; filled in when response-assembly repair primitive ships (post-v0 M7).
        "source_turn_index": source_turn_index,
        "source_wave_id": source_wave_id,
        "observed_outcome_kind": observed_outcome_kind,
        "confidence": round(confidence, 4),
        "regime_id": regime_id,
        "abstract_action": abstract_action,
        "action_family_version": action_family_version,
        "user_scope": user_scope,
    }
    tags = (
        "rupture_repair",
        f"rupture_kind:{rupture_kind.value}",
        f"repair_outcome:{repair_outcome}",
        f"user_scope:{user_scope}",
        f"source_wave:{source_wave_id}",
    )
    entry_id = (
        f"rupture_repair:{user_scope}:{source_wave_id}:{rupture_kind.value}:{repair_outcome}"
    )
    entry = MemoryEntry(
        entry_id=entry_id,
        content=json.dumps(content_obj, sort_keys=True, separators=(",", ":")),
        track=Track.SELF,
        stratum=MemoryStratum.DURABLE.value,
        created_at_ms=timestamp_ms,
        last_accessed_ms=timestamp_ms,
        strength=min(1.0, 0.6 + confidence * 0.3),
        tags=tags,
    )
    return (entry,)


class ReflectionEngine:
    """Builds proposal-first slow reflection artifacts from current session state."""

    def __init__(self, *, writeback_mode: WritebackMode = WritebackMode.PROPOSAL_ONLY) -> None:
        self._writeback_mode = writeback_mode
        self._proposal_outcome_ledger: list[ProposalOutcomeEntry] = []
        self._last_bundle: StructuralProposalBundle | None = None
        self._last_metric_snapshot: tuple[tuple[str, float], ...] = ()

    @property
    def writeback_mode(self) -> WritebackMode:
        return self._writeback_mode

    @property
    def proposal_outcome_ledger(self) -> tuple[ProposalOutcomeEntry, ...]:
        return tuple(self._proposal_outcome_ledger)

    @property
    def proposal_success_rate(self) -> float:
        if not self._proposal_outcome_ledger:
            return 0.0
        successes = sum(1 for e in self._proposal_outcome_ledger if e.success)
        return successes / len(self._proposal_outcome_ledger)

    def reflect(
        self,
        *,
        timestamp_ms: int,
        memory_snapshot: MemorySnapshot | None,
        dual_track_snapshot: DualTrackSnapshot | None,
        evaluation_snapshot: EvaluationSnapshot | None,
        credit_snapshot: CreditSnapshot | None,
        prediction_error_snapshot: PredictionErrorSnapshot | None = None,
        regime_snapshot: RegimeSnapshot | None = None,
        rupture_state_snapshot: RuptureStateSnapshot | None = None,
        dialogue_external_outcome_snapshot: DialogueExternalOutcomeSnapshot | None = None,
        user_scope: str = "anonymous",
    ) -> ReflectionSnapshot:
        self._update_proposal_outcome_ledger(evaluation_snapshot)
        consolidation_score = self._consolidation_score(
            memory_snapshot=memory_snapshot,
            dual_track_snapshot=dual_track_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            credit_snapshot=credit_snapshot,
            prediction_error_snapshot=prediction_error_snapshot,
        )
        memory_consolidation = self._memory_consolidation(
            timestamp_ms=timestamp_ms,
            memory_snapshot=memory_snapshot,
            credit_snapshot=credit_snapshot,
            consolidation_score=consolidation_score,
            regime_snapshot=regime_snapshot,
            prediction_error_snapshot=prediction_error_snapshot,
        )
        # Rupture-and-Repair M3: if the rupture_state snapshot resolved a
        # kind AND at least one externally-confirmed outcome contributed
        # this turn, write a durable rupture-repair memory entry. The
        # entry is added to ``new_durable_entries`` so it flows through
        # the existing ``ReflectionEngine.apply`` -> ``memory_store``
        # path (R8; no bypass writes).
        rupture_repair_entries = rupture_repair_memory_entries(
            rupture_state_snapshot=rupture_state_snapshot,
            external_outcome_snapshot=dialogue_external_outcome_snapshot,
            regime_snapshot=regime_snapshot,
            user_scope=user_scope,
            timestamp_ms=timestamp_ms,
        )
        if rupture_repair_entries:
            memory_consolidation = MemoryConsolidation(
                new_durable_entries=memory_consolidation.new_durable_entries
                + rupture_repair_entries,
                promoted_entries=memory_consolidation.promoted_entries,
                decayed_entries=memory_consolidation.decayed_entries,
                beliefs_updated=memory_consolidation.beliefs_updated,
            )
        policy_consolidation = self._policy_consolidation(
            dual_track_snapshot=dual_track_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            credit_snapshot=credit_snapshot,
            regime_snapshot=regime_snapshot,
            consolidation_score=consolidation_score,
            prediction_error_snapshot=prediction_error_snapshot,
        )
        pe_tensions = _prediction_error_tensions(prediction_error_snapshot)
        tensions = (
            self._tensions(
                dual_track_snapshot=dual_track_snapshot,
                evaluation_snapshot=evaluation_snapshot,
            )
            + pe_tensions
        )
        lessons = self._lessons(
            memory_consolidation=memory_consolidation,
            policy_consolidation=policy_consolidation,
            tensions=tensions,
            prediction_error_snapshot=prediction_error_snapshot,
        )
        primary_error_dimension = _primary_prediction_error_dimension(prediction_error_snapshot)
        repeated_failures = _repeated_prediction_failures(prediction_error_snapshot)
        error_driven_lesson_count = _count_error_driven_lessons(lessons)
        trace_summary = self._interaction_trace_summary(
            memory_snapshot=memory_snapshot,
            dual_track_snapshot=dual_track_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            prediction_error_snapshot=prediction_error_snapshot,
        )
        review_required = self._writeback_mode is not WritebackMode.APPLY
        self._capture_metric_snapshot(evaluation_snapshot)
        description = (
            f"Reflection generated {len(memory_consolidation.new_durable_entries)} durable proposals, "
            f"{len(policy_consolidation.strategy_priors_updated)} strategy updates, "
            f"mode={self._writeback_mode.value}, score={consolidation_score.confidence:.2f}."
        )
        return ReflectionSnapshot(
            memory_consolidation=memory_consolidation,
            policy_consolidation=policy_consolidation,
            consolidation_score=consolidation_score,
            interaction_trace_summary=trace_summary,
            tensions_identified=tensions,
            lessons_extracted=lessons,
            primary_prediction_error_dimension=primary_error_dimension,
            repeated_prediction_failures=repeated_failures,
            error_driven_lesson_count=error_driven_lesson_count,
            writeback_mode=self._writeback_mode.value,
            review_required=review_required,
            description=description,
            proposal_success_rate=self.proposal_success_rate,
        )

    def apply(
        self,
        *,
        memory_store: MemoryStore,
        reflection_snapshot: ReflectionSnapshot,
        credit_snapshot: CreditSnapshot | None,
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
            blocked_targets = tuple(
                record.target
                for record in credit_snapshot.recent_modifications
                if record.decision is GateDecision.BLOCK
            )
            rupture_repair_entries = tuple(
                entry
                for entry in reflection_snapshot.memory_consolidation.new_durable_entries
                if "rupture_repair" in entry.tags
            )
            if rupture_repair_entries:
                memory_checkpoint = memory_store.create_checkpoint(
                    checkpoint_id=checkpoint_id
                )
                applied_operations = memory_store.apply_reflection_consolidation(
                    new_durable_entries=rupture_repair_entries,
                    promoted_entries=(),
                    decayed_entries=(),
                    beliefs_updated=(),
                    promotion_boost=0.0,
                    decay_scale=0.0,
                    lesson_count=0,
                    timestamp_ms=max(
                        (entry.created_at_ms for entry in rupture_repair_entries),
                        default=1,
                    ),
                )
                checkpoint = WritebackCheckpoint(
                    checkpoint_id=memory_checkpoint.checkpoint_id,
                    memory_checkpoint=memory_checkpoint,
                )
                return WritebackResult(
                    applied_operations=applied_operations,
                    blocked_operations=("credit-gate-block",),
                    checkpoint=checkpoint,
                    description=(
                        "Credit gate blocked general reflection writeback, "
                        "but externally confirmed rupture_repair memory was applied for "
                        f"{', '.join(blocked_targets) if blocked_targets else 'unknown-target'}."
                    ),
                )
            return WritebackResult(
                applied_operations=(),
                blocked_operations=("credit-gate-block",),
                checkpoint=None,
                description=(
                    "Writeback blocked by credit gate evidence for "
                    f"{', '.join(blocked_targets) if blocked_targets else 'unknown-target'}."
                ),
            )
        memory_checkpoint = memory_store.create_checkpoint(checkpoint_id=checkpoint_id)
        applied_operations = memory_store.apply_reflection_consolidation(
            new_durable_entries=reflection_snapshot.memory_consolidation.new_durable_entries,
            promoted_entries=reflection_snapshot.memory_consolidation.promoted_entries,
            decayed_entries=reflection_snapshot.memory_consolidation.decayed_entries,
            beliefs_updated=reflection_snapshot.memory_consolidation.beliefs_updated,
            promotion_boost=reflection_snapshot.consolidation_score.promotion_score,
            decay_scale=reflection_snapshot.consolidation_score.decay_score,
            lesson_count=len(reflection_snapshot.lessons_extracted),
            timestamp_ms=max(
                (entry.created_at_ms for entry in reflection_snapshot.memory_consolidation.new_durable_entries),
                default=1,
            ),
        )
        applied_operations = applied_operations + (
            memory_store.apply_promotion_threshold_update(
                delta=reflection_snapshot.consolidation_score.threshold_delta
            ),
        )
        checkpoint = WritebackCheckpoint(
            checkpoint_id=memory_checkpoint.checkpoint_id,
            memory_checkpoint=memory_checkpoint,
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
    ) -> None:
        memory_store.restore_checkpoint(checkpoint.memory_checkpoint)

    def _memory_consolidation(
        self,
        *,
        timestamp_ms: int,
        memory_snapshot: MemorySnapshot | None,
        credit_snapshot: CreditSnapshot | None,
        consolidation_score: ConsolidationScore,
        regime_snapshot: RegimeSnapshot | None,
        prediction_error_snapshot: PredictionErrorSnapshot | None = None,
    ) -> MemoryConsolidation:
        if memory_snapshot is None:
            return MemoryConsolidation((), (), (), ())

        promoted_entries: list[str] = []
        new_durable_entries: list[MemoryEntry] = []
        decayed_entries: list[str] = []
        beliefs_updated: list[str] = []
        promotion_limit = 1
        if consolidation_score.promotion_score > 0.55:
            promotion_limit = 2
        if consolidation_score.promotion_score > 0.75:
            promotion_limit = 3

        for entry in memory_snapshot.retrieved_entries[:promotion_limit]:
            if (
                entry.stratum != MemoryStratum.DURABLE.value
                and entry.strength + consolidation_score.promotion_score * 0.2 >= 0.6
            ):
                promoted_entries.append(entry.entry_id)
        if credit_snapshot is not None:
            decay_candidates = list(memory_snapshot.retrieved_entries)
            for record in credit_snapshot.recent_credits[:3]:
                if record.credit_value > 0.6:
                    beliefs_updated.append(f"reinforce:{record.track.value}:{record.source_event}")
                if record.credit_value < 0.2 and consolidation_score.decay_score > 0.25:
                    for candidate in decay_candidates:
                        if candidate.track is record.track or record.track is Track.SHARED:
                            decayed_entries.append(candidate.entry_id)
                            decay_candidates.remove(candidate)
                            break
        if (
            regime_snapshot is not None
            and regime_snapshot.delayed_outcomes
            and regime_snapshot.identity_hints
        ):
            delayed_score = sum(
                score for _, score in regime_snapshot.delayed_outcomes
            ) / len(regime_snapshot.delayed_outcomes)
            if delayed_score >= 0.55:
                for hint in regime_snapshot.identity_hints[:2]:
                    track = Track.SELF if hint.startswith("identity:relationship:") else Track.SHARED
                    new_durable_entries.append(
                        MemoryEntry(
                            entry_id=str(uuid4()),
                            content=hint,
                            track=track,
                            stratum=MemoryStratum.DURABLE.value,
                            created_at_ms=timestamp_ms,
                            last_accessed_ms=timestamp_ms,
                            strength=_clamp(0.55 + delayed_score * 0.25),
                            tags=("identity", "delayed_attribution"),
                        )
                    )
                for regime_id, score in regime_snapshot.delayed_outcomes:
                    beliefs_updated.append(f"delayed_regime:{regime_id}:{score:.2f}")
        if prediction_error_snapshot is not None and not prediction_error_snapshot.bootstrap:
            pe = prediction_error_snapshot.error
            if pe.relationship_error < -0.15:
                for entry in memory_snapshot.retrieved_entries:
                    if entry.track is Track.SELF and entry.strength < 0.5 and entry.entry_id not in decayed_entries:
                        decayed_entries.append(entry.entry_id)
                        break
            if pe.task_error < -0.15:
                for entry in memory_snapshot.retrieved_entries[:promotion_limit]:
                    if (
                        entry.track is Track.WORLD
                        and entry.stratum != MemoryStratum.DURABLE.value
                        and entry.entry_id not in promoted_entries
                    ):
                        promoted_entries.append(entry.entry_id)
                        break
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
        consolidation_score: ConsolidationScore,
        prediction_error_snapshot: PredictionErrorSnapshot | None = None,
    ) -> PolicyConsolidation:
        controller_updates: list[str] = []
        strategy_priors_updated: list[str] = []
        regime_effectiveness_updated: list[tuple[str, float]] = []
        controller_guard_blocked = False
        controller_guard_audit_present = False
        cross_tension = dual_track_snapshot.cross_track_tension if dual_track_snapshot is not None else 0.0
        world_tension = dual_track_snapshot.world_track.tension_level if dual_track_snapshot is not None else 0.0
        self_tension = dual_track_snapshot.self_track.tension_level if dual_track_snapshot is not None else 0.0
        task_pressure = world_tension
        support_presence = self_tension
        family_monopoly_pressure = _metric(
            evaluation_snapshot,
            "action_family_monopoly_pressure",
            default=0.0,
        )
        family_turnover_health = _metric(
            evaluation_snapshot,
            "action_family_turnover_health",
            default=0.5,
        )
        family_collapse_risk = _metric(
            evaluation_snapshot,
            "action_family_collapse_risk",
            default=0.0,
        )

        if dual_track_snapshot is not None:
            task_pressure = _metric(evaluation_snapshot, "task_pressure", default=world_tension)
            support_presence = _metric(evaluation_snapshot, "support_presence", default=self_tension)
            if dual_track_snapshot.cross_track_tension > 0.5:
                controller_updates.append("reduce_cross_track_tension_before_widening_scope")
            if prediction_error_snapshot is not None and not prediction_error_snapshot.bootstrap:
                pe = prediction_error_snapshot.error
                if pe.relationship_error < pe.task_error - 0.05 and consolidation_score.strategy_gain >= 0.03:
                    strategy_priors_updated.append("increase_self_track_priority")
                elif pe.task_error < pe.relationship_error - 0.05 and consolidation_score.strategy_gain >= 0.03:
                    strategy_priors_updated.append("increase_world_track_priority")
            elif support_presence > task_pressure + 0.08 and consolidation_score.strategy_gain >= 0.03:
                strategy_priors_updated.append("increase_self_track_priority")
            elif task_pressure > support_presence + 0.08 and consolidation_score.strategy_gain >= 0.03:
                strategy_priors_updated.append("increase_world_track_priority")
            elif (
                dual_track_snapshot.self_track.tension_level > dual_track_snapshot.world_track.tension_level
                and consolidation_score.strategy_gain >= 0.03
            ):
                strategy_priors_updated.append("increase_self_track_priority")
            elif dual_track_snapshot.world_track.tension_level > 0 and consolidation_score.strategy_gain >= 0.03:
                strategy_priors_updated.append("increase_world_track_priority")

        if (
            prediction_error_snapshot is not None
            and not prediction_error_snapshot.bootstrap
            and regime_snapshot is not None
        ):
            regime_effectiveness_updated.append(
                (
                    regime_snapshot.active_regime.regime_id,
                    _clamp(0.5 + prediction_error_snapshot.error.regime_error),
                )
            )
        elif evaluation_snapshot is not None:
            for score in evaluation_snapshot.session_scores[:4]:
                if regime_snapshot is not None and score.family in {"relationship", "task"}:
                    regime_effectiveness_updated.append(
                        (regime_snapshot.active_regime.regime_id, score.value)
                    )

        if credit_snapshot is not None and credit_snapshot.recent_modifications:
            controller_updates.append("gate_audit_available_for_follow_up")
            controller_guard_audit_present = True
            metacontroller_audits = tuple(
                record for record in credit_snapshot.recent_modifications if record.target.startswith("metacontroller.")
            )
            if metacontroller_audits:
                latest_audit = metacontroller_audits[-1]
                if latest_audit.decision is GateDecision.BLOCK:
                    controller_updates.append("pause_metacontroller_writeback_after_runtime_guard")
                    controller_guard_blocked = True
                else:
                    controller_updates.append("metacontroller_runtime_guard_cleared")
        if family_monopoly_pressure > 0.55:
            controller_updates.append("reduce_action_family_monopoly")
        if family_turnover_health < 0.45:
            controller_updates.append("encourage_action_family_turnover")
        if family_collapse_risk > 0.60:
            controller_updates.append("prevent_action_family_collapse")

        structure_proposals = self._temporal_structure_proposals(
            regime_snapshot=regime_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            consolidation_score=consolidation_score,
        )
        structure_bundle = self._build_structure_bundle(
            proposals=structure_proposals,
            regime_snapshot=regime_snapshot,
            evaluation_snapshot=evaluation_snapshot,
        )
        target_groups = ["base-weights", "switch", "persistence", "learning-rate"]
        if support_presence > task_pressure + 0.08:
            target_groups.extend(["track-self", "decoder"])
        elif task_pressure > support_presence + 0.08:
            target_groups.extend(["track-world", "encoder", "decoder"])
        else:
            target_groups.extend(["track-shared", "encoder"])
        if cross_tension > 0.35:
            target_groups.extend(["beta-threshold", "action-families"])
        if family_monopoly_pressure > 0.55 or family_turnover_health < 0.45 or family_collapse_risk > 0.60:
            target_groups.append("action-families")
        if structure_proposals:
            target_groups.append("action-family-structure")
        temporal_prior_update = TemporalPriorUpdate(
            target="metacontroller.temporal_prior",
            target_groups=tuple(dict.fromkeys(target_groups)),
            residual_strength=_clamp(0.45 + world_tension * 0.25 + consolidation_score.confidence * 0.10),
            memory_strength=_clamp(0.30 + self_tension * 0.20 + consolidation_score.promotion_score * 0.10),
            reflection_strength=_clamp(
                0.25 + consolidation_score.confidence * 0.20 + consolidation_score.strategy_gain * 2.0
            ),
            switch_bias_delta=max(-0.08, min(0.08, cross_tension * 0.12 - consolidation_score.decay_score * 0.04)),
            persistence_delta=max(
                -0.08,
                min(0.08, consolidation_score.confidence * 0.06 - cross_tension * 0.08),
            ),
            learning_rate_delta=max(
                -0.02,
                min(0.02, consolidation_score.strategy_gain * 0.15 - consolidation_score.decay_score * 0.03),
            ),
            encoder_strength_delta=max(
                -0.08,
                min(0.08, task_pressure * 0.08 + consolidation_score.confidence * 0.04 - cross_tension * 0.05),
            ),
            decoder_strength_delta=max(
                -0.08,
                min(
                    0.08,
                    max(task_pressure, support_presence) * 0.06
                    + consolidation_score.confidence * 0.03
                    - consolidation_score.decay_score * 0.02,
                ),
            ),
            world_track_delta=max(-0.08, min(0.08, task_pressure * 0.08 - support_presence * 0.03)),
            self_track_delta=max(-0.08, min(0.08, support_presence * 0.08 - task_pressure * 0.03)),
            shared_track_delta=max(
                -0.06,
                min(0.06, (1.0 - abs(task_pressure - support_presence)) * 0.05 + cross_tension * 0.03),
            ),
            beta_threshold_delta=max(-0.05, min(0.05, cross_tension * 0.05 - consolidation_score.confidence * 0.02)),
            family_stability_delta=max(
                -0.06,
                min(
                    0.06,
                    consolidation_score.confidence * 0.05
                    - cross_tension * 0.03
                    - family_monopoly_pressure * 0.04
                    - max(0.0, 0.5 - family_turnover_health) * 0.05,
                ),
            ),
            structure_proposals=structure_proposals,
            structure_bundle=structure_bundle,
            description=(
                f"Temporal prior update from reflection confidence={consolidation_score.confidence:.2f}, "
                f"cross_tension={cross_tension:.2f}, "
                f"world_tension={world_tension:.2f}, "
                f"self_tension={self_tension:.2f}, "
                f"groups={tuple(dict.fromkeys(target_groups))}."
            ),
        )
        return PolicyConsolidation(
            controller_updates=tuple(controller_updates),
            strategy_priors_updated=tuple(strategy_priors_updated),
            regime_effectiveness_updated=tuple(regime_effectiveness_updated),
            temporal_prior_update=temporal_prior_update,
            controller_guard_blocked=controller_guard_blocked,
            controller_guard_audit_present=controller_guard_audit_present,
        )

    def _temporal_structure_proposals(
        self,
        *,
        regime_snapshot: RegimeSnapshot | None,
        evaluation_snapshot: EvaluationSnapshot | None,
        consolidation_score: ConsolidationScore,
    ) -> tuple[TemporalStructureProposal, ...]:
        family_monopoly_pressure = _metric(
            evaluation_snapshot,
            "action_family_monopoly_pressure",
            default=0.0,
        )
        family_turnover_health = _metric(
            evaluation_snapshot,
            "action_family_turnover_health",
            default=0.5,
        )
        family_collapse_risk = _metric(
            evaluation_snapshot,
            "action_family_collapse_risk",
            default=0.0,
        )
        if regime_snapshot is None or (
            not regime_snapshot.delayed_attributions and not regime_snapshot.delayed_payoffs
        ):
            return ()
        proposals: list[TemporalStructureProposal] = []
        cross_track_stability = _metric(evaluation_snapshot, "cross_track_stability", default=0.5)
        for payoff in regime_snapshot.delayed_payoffs:
            if payoff.abstract_action is None:
                continue
            if payoff.rolling_payoff < 0.38 and payoff.sample_count >= 2:
                proposals.append(
                    TemporalStructureProposal(
                        proposal_type="prune",
                        family_id=payoff.abstract_action,
                        related_family_id=None,
                        confidence=_clamp(
                            0.46
                            + (0.4 - payoff.rolling_payoff)
                            + min(payoff.sample_count / 4.0, 1.0) * 0.12
                            + consolidation_score.confidence * 0.10
                        ),
                        justification=(
                            f"Prune persistently weak family from rolling_payoff={payoff.rolling_payoff:.3f} "
                            f"over {payoff.sample_count} samples."
                        ),
                    )
                )
            elif payoff.rolling_payoff < 0.52 and payoff.sample_count >= 2 and family_turnover_health < 0.5:
                proposals.append(
                    TemporalStructureProposal(
                        proposal_type="split",
                        family_id=payoff.abstract_action,
                        related_family_id=None,
                        confidence=_clamp(
                            0.40
                            + (0.52 - payoff.rolling_payoff)
                            + (0.5 - family_turnover_health) * 0.22
                            + consolidation_score.confidence * 0.08
                        ),
                        justification=(
                            f"Split under-performing family with rolling_payoff={payoff.rolling_payoff:.3f} "
                            f"and turnover_health={family_turnover_health:.3f}."
                        ),
                    )
                )
        for attribution in regime_snapshot.delayed_attributions:
            if attribution.abstract_action is None:
                continue
            if family_monopoly_pressure > 0.72 and family_turnover_health < 0.45:
                proposals.append(
                    TemporalStructureProposal(
                        proposal_type="split",
                        family_id=attribution.abstract_action,
                        related_family_id=None,
                        confidence=_clamp(
                            0.45
                            + family_monopoly_pressure * 0.25
                            + (0.45 - family_turnover_health) * 0.30
                            + consolidation_score.confidence * 0.10
                        ),
                        justification=(
                            f"Split dominant family due to monopoly_pressure={family_monopoly_pressure:.3f} "
                            f"and turnover_health={family_turnover_health:.3f}."
                        ),
                    )
                )
            if attribution.outcome_score < 0.35:
                proposals.append(
                    TemporalStructureProposal(
                        proposal_type="prune",
                        family_id=attribution.abstract_action,
                        related_family_id=None,
                        confidence=_clamp(
                            0.45
                            + (0.35 - attribution.outcome_score)
                            + consolidation_score.confidence * 0.15
                        ),
                        justification=(
                            f"Prune weak family after delayed outcome {attribution.outcome_score:.3f} "
                            f"from {attribution.source_wave_id}."
                        ),
                    )
                )
            elif (
                attribution.outcome_score < 0.5 and cross_track_stability < 0.72
            ) or family_collapse_risk > 0.68:
                proposals.append(
                    TemporalStructureProposal(
                        proposal_type="split",
                        family_id=attribution.abstract_action,
                        related_family_id=None,
                        confidence=_clamp(
                            0.42
                            + (0.5 - attribution.outcome_score)
                            + (0.72 - cross_track_stability)
                            + family_collapse_risk * 0.15
                        ),
                        justification=(
                            f"Split overloaded family after mixed delayed outcome {attribution.outcome_score:.3f} "
                            f"cross_track_stability={cross_track_stability:.3f}, "
                            f"collapse_risk={family_collapse_risk:.3f}."
                        ),
                    )
                )
        strong_actions = [
            item for item in regime_snapshot.delayed_attributions
            if item.abstract_action is not None and item.outcome_score > 0.72
        ]
        if len(strong_actions) >= 2:
            first = strong_actions[0]
            second = next(
                (item for item in strong_actions[1:] if item.abstract_action != first.abstract_action),
                None,
            )
            if second is not None:
                proposals.append(
                    TemporalStructureProposal(
                        proposal_type="merge",
                        family_id=first.abstract_action or "unassigned_action",
                        related_family_id=second.abstract_action,
                        confidence=_clamp(
                            0.4
                            + min(first.outcome_score, second.outcome_score) * 0.35
                            + consolidation_score.confidence * 0.1
                        ),
                        justification=(
                            f"Merge convergent strong families from {first.source_wave_id} and {second.source_wave_id} "
                            f"within regime={first.regime_id}."
                        ),
                    )
                )
        deduped: list[TemporalStructureProposal] = []
        seen: set[tuple[str, str, str | None]] = set()
        for proposal in proposals:
            key = (proposal.proposal_type, proposal.family_id, proposal.related_family_id)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(proposal)
        return tuple(deduped[:3])

    def _consolidation_score(
        self,
        *,
        memory_snapshot: MemorySnapshot | None,
        dual_track_snapshot: DualTrackSnapshot | None,
        evaluation_snapshot: EvaluationSnapshot | None,
        credit_snapshot: CreditSnapshot | None,
        prediction_error_snapshot: PredictionErrorSnapshot | None,
    ) -> ConsolidationScore:
        memory_pressure = _clamp(
            len(memory_snapshot.retrieved_entries) / 5.0 if memory_snapshot is not None else 0.0
        )
        cross_tension = dual_track_snapshot.cross_track_tension if dual_track_snapshot is not None else 0.0
        alert_pressure = min(len(_relationship_relevant_alerts(evaluation_snapshot)), 3) / 3.0
        positive_credit = 0.0
        negative_credit = 0.0
        if credit_snapshot is not None and credit_snapshot.recent_credits:
            positive_credit = sum(
                max(record.credit_value, 0.0) for record in credit_snapshot.recent_credits[:5]
            ) / max(len(credit_snapshot.recent_credits[:5]), 1)
            negative_credit = sum(
                max(0.0, 0.3 - record.credit_value) for record in credit_snapshot.recent_credits[:5]
            ) / max(len(credit_snapshot.recent_credits[:5]), 1)
        session_bonus = 0.0
        if credit_snapshot is not None and credit_snapshot.session_level_credits:
            session_values = tuple(value for _, value in credit_snapshot.session_level_credits)
            if session_values:
                session_bonus = _clamp(sum(session_values) / len(session_values) * 0.15)
        pe_penalty = 0.0
        if prediction_error_snapshot is not None:
            pe_penalty = min(prediction_error_snapshot.error.magnitude / 4.0, 1.0)
        promotion_score = _clamp(
            0.35
            + memory_pressure * 0.25
            + positive_credit * 0.40
            + session_bonus
            - cross_tension * 0.15
            - pe_penalty * 0.12
        )
        decay_score = _clamp(0.10 + negative_credit * 0.75 + alert_pressure * 0.2 + pe_penalty * 0.15)
        threshold_delta = max(-0.05, min(0.05, (cross_tension + alert_pressure - promotion_score) * 0.04))
        strategy_gain = max(0.02, min(0.08, 0.02 + promotion_score * 0.05))
        regime_effectiveness_gain = max(0.2, min(0.45, 0.2 + promotion_score * 0.25))
        confidence = _clamp((memory_pressure + positive_credit + (1.0 - cross_tension) + (1.0 - pe_penalty)) / 4.0)
        return ConsolidationScore(
            promotion_score=promotion_score,
            decay_score=decay_score,
            threshold_delta=threshold_delta,
            strategy_gain=strategy_gain,
            regime_effectiveness_gain=regime_effectiveness_gain,
            confidence=confidence,
            description=(
                f"consolidation promotion={promotion_score:.3f} decay={decay_score:.3f} "
                f"threshold_delta={threshold_delta:.3f}"
            ),
        )

    def _tensions(
        self,
        *,
        dual_track_snapshot: DualTrackSnapshot | None,
        evaluation_snapshot: EvaluationSnapshot | None,
    ) -> tuple[str, ...]:
        tensions: list[str] = []
        if dual_track_snapshot is not None:
            if dual_track_snapshot.cross_track_tension > 0.4:
                tensions.append(ReflectionTensionId.CROSS_TRACK_TENSION_HIGH.value)
            elif dual_track_snapshot.cross_track_tension > 0.2:
                tensions.append(ReflectionTensionId.CROSS_TRACK_ALIGNMENT_DRIFT.value)
            world_tension = dual_track_snapshot.world_track.tension_level
            self_tension = dual_track_snapshot.self_track.tension_level
            if self_tension > world_tension + 0.12:
                tensions.append(ReflectionTensionId.SELF_TRACK_PRESSURE_DOMINANT.value)
            elif world_tension > self_tension + 0.12:
                tensions.append(ReflectionTensionId.WORLD_TRACK_PRESSURE_DOMINANT.value)
        if evaluation_snapshot is not None:
            if _metric(evaluation_snapshot, "cross_track_stability", default=1.0) < 0.7:
                tensions.append(
                    ReflectionTensionId.RELATIONSHIP_STABILITY_SOFT_DROP.value
                )
            if _metric(evaluation_snapshot, "warmth", default=1.0) < 0.45:
                tensions.append(ReflectionTensionId.WARMTH_SIGNAL_THIN.value)
            if _metric(evaluation_snapshot, "info_integration", default=1.0) < 0.45:
                tensions.append(ReflectionTensionId.TASK_SIGNAL_DIFFUSE.value)
            tensions.extend(_relationship_relevant_alerts(evaluation_snapshot)[:3])
        return tuple(dict.fromkeys(tensions))

    def _lessons(
        self,
        *,
        memory_consolidation: MemoryConsolidation,
        policy_consolidation: PolicyConsolidation,
        tensions: tuple[str, ...],
        prediction_error_snapshot: PredictionErrorSnapshot | None,
    ) -> tuple[str, ...]:
        lessons: list[str] = []
        if memory_consolidation.new_durable_entries or memory_consolidation.promoted_entries:
            lessons.append(ReflectionLessonId.PROMOTE_HIGH_SIGNAL_MEMORIES.value)
        if memory_consolidation.beliefs_updated:
            lessons.append(
                ReflectionLessonId.REINFORCE_RECENT_HIGH_CREDIT_BELIEFS.value
            )
        if policy_consolidation.strategy_priors_updated:
            lessons.append(
                ReflectionLessonId.ADJUST_TRACK_PRIORITY_FROM_SESSION_FEEDBACK.value
            )
        if policy_consolidation.temporal_prior_update is not None:
            temporal_update = policy_consolidation.temporal_prior_update
            strongest_channel = max(
                (
                    (
                        "residual",
                        temporal_update.residual_strength,
                        ReflectionLessonId.REBALANCE_TEMPORAL_PRIOR_TOWARD_RESIDUAL,
                    ),
                    (
                        "memory",
                        temporal_update.memory_strength,
                        ReflectionLessonId.REBALANCE_TEMPORAL_PRIOR_TOWARD_MEMORY,
                    ),
                    (
                        "reflection",
                        temporal_update.reflection_strength,
                        ReflectionLessonId.REBALANCE_TEMPORAL_PRIOR_TOWARD_REFLECTION,
                    ),
                ),
                key=lambda item: item[1],
            )[2]
            lessons.append(strongest_channel.value)
            if temporal_update.persistence_delta > 0.02:
                lessons.append(
                    ReflectionLessonId.INCREASE_CONTROLLER_PERSISTENCE_FOR_CONTINUITY.value
                )
            elif temporal_update.persistence_delta < -0.02:
                lessons.append(
                    ReflectionLessonId.REDUCE_CONTROLLER_PERSISTENCE_FOR_FASTER_RECOVERY.value
                )
            if temporal_update.switch_bias_delta > 0.02:
                lessons.append(
                    ReflectionLessonId.ALLOW_CONTROLLER_SWITCH_WHEN_CONTEXT_SHIFTS.value
                )
            elif temporal_update.switch_bias_delta < -0.02:
                lessons.append(
                    ReflectionLessonId.HOLD_CONTROLLER_BEFORE_SWITCHING.value
                )
            if temporal_update.structure_proposals:
                lessons.append(
                    ReflectionLessonId.RESTRUCTURE_ACTION_FAMILY_BANK.value
                )
        if policy_consolidation.controller_guard_blocked:
            lessons.append(ReflectionLessonId.RESPECT_METACONTROLLER_RUNTIME_GUARD.value)
        elif policy_consolidation.controller_guard_audit_present:
            lessons.append(
                ReflectionLessonId.KEEP_CONTROLLER_GUARD_SIGNAL_IN_BACKGROUND.value
            )
        if tensions:
            lessons.append(
                ReflectionLessonId.REVIEW_TENSION_BEFORE_AUTO_WRITEBACK.value
            )
        lessons.extend(_prediction_error_lessons(prediction_error_snapshot))
        return tuple(dict.fromkeys(lessons))

    def _interaction_trace_summary(
        self,
        *,
        memory_snapshot: MemorySnapshot | None,
        dual_track_snapshot: DualTrackSnapshot | None,
        evaluation_snapshot: EvaluationSnapshot | None,
        prediction_error_snapshot: PredictionErrorSnapshot | None,
    ) -> str:
        memory_count = len(memory_snapshot.retrieved_entries) if memory_snapshot else 0
        cross_tension = dual_track_snapshot.cross_track_tension if dual_track_snapshot else 0.0
        alert_count = len(evaluation_snapshot.alerts) if evaluation_snapshot else 0
        pe_dimension = _primary_prediction_error_dimension(prediction_error_snapshot) or "none"
        pe_magnitude = prediction_error_snapshot.error.magnitude if prediction_error_snapshot is not None else 0.0
        return (
            f"Reflection input summary: retrieved_entries={memory_count}, "
            f"cross_track_tension={cross_tension:.2f}, alerts={alert_count}, "
            f"prediction_error={pe_dimension}:{pe_magnitude:.2f}."
        )

    def _capture_metric_snapshot(self, evaluation_snapshot: EvaluationSnapshot | None) -> None:
        if evaluation_snapshot is None:
            self._last_metric_snapshot = ()
            return
        self._last_metric_snapshot = tuple(
            (score.metric_name, score.value) for score in evaluation_snapshot.turn_scores
        )

    def _update_proposal_outcome_ledger(self, evaluation_snapshot: EvaluationSnapshot | None) -> None:
        if self._last_bundle is None or not self._last_metric_snapshot:
            return
        current_metrics: dict[str, float] = {}
        if evaluation_snapshot is not None:
            current_metrics = {s.metric_name: s.value for s in evaluation_snapshot.turn_scores}
        pre_dict = dict(self._last_metric_snapshot)
        key_metrics = (
            "cross_track_stability",
            "action_family_monopoly_pressure",
            "action_family_turnover_health",
            "action_family_collapse_risk",
        )
        pre_vals = [pre_dict.get(m, 0.5) for m in key_metrics]
        post_vals = [current_metrics.get(m, 0.5) for m in key_metrics]
        pre_mean = sum(pre_vals) / max(len(pre_vals), 1)
        post_mean = sum(post_vals) / max(len(post_vals), 1)
        delta = post_mean - pre_mean
        post_snapshot = tuple(sorted(current_metrics.items()))
        entry = ProposalOutcomeEntry(
            bundle_scope=self._last_bundle.scope,
            proposal_types=tuple(p.proposal_type for p in self._last_bundle.proposals),
            bundle_confidence=self._last_bundle.bundle_confidence,
            pre_metric_snapshot=self._last_metric_snapshot,
            post_metric_snapshot=post_snapshot,
            metric_delta=round(delta, 4),
            success=delta >= -0.02,
        )
        self._proposal_outcome_ledger.append(entry)
        self._proposal_outcome_ledger = self._proposal_outcome_ledger[-16:]
        self._last_bundle = None

    def _build_structure_bundle(
        self,
        *,
        proposals: tuple[TemporalStructureProposal, ...],
        regime_snapshot: RegimeSnapshot | None,
        evaluation_snapshot: EvaluationSnapshot | None,
    ) -> StructuralProposalBundle | None:
        if not proposals:
            return None
        delayed_credit_summary: tuple[tuple[str, float], ...] = ()
        if regime_snapshot is not None and regime_snapshot.delayed_payoffs:
            delayed_credit_summary = tuple(
                (p.abstract_action or "none", p.rolling_payoff)
                for p in regime_snapshot.delayed_payoffs[:4]
            )
        session_trend_delta: tuple[tuple[str, float], ...] = ()
        if evaluation_snapshot is not None:
            session_trend_delta = tuple(
                (s.metric_name, s.value - 0.5)
                for s in evaluation_snapshot.session_scores[:4]
            )
        unique_families = {p.family_id for p in proposals}
        unique_types = {p.proposal_type for p in proposals}
        if len(unique_families) >= 2 and len(unique_types) >= 2:
            scope = "family-cluster"
        elif len(unique_families) >= 2:
            scope = "regime-sequence"
        else:
            scope = "single-family"
        bundle_confidence = _clamp(
            sum(p.confidence for p in proposals) / len(proposals)
        )
        evidence = ProposalEvidencePack(
            benchmark_passed=None,
            delayed_credit_summary=delayed_credit_summary,
            session_trend_delta=session_trend_delta,
        )
        bundle = StructuralProposalBundle(
            proposals=proposals,
            evidence_pack=evidence,
            scope=scope,
            bundle_confidence=bundle_confidence,
        )
        self._last_bundle = bundle
        return bundle


class ReflectionModule(RuntimeModule[ReflectionSnapshot]):
    slot_name = "reflection"
    owner = "ReflectionModule"
    value_type = ReflectionSnapshot
    # Rupture-and-Repair M3: ``rupture_state`` and
    # ``dialogue_external_outcome`` are added so reflection can emit
    # rupture-repair durable proposals on the turn both conditions hold.
    # The slots are SHADOW-safe reads: reflection only consumes them
    # through snapshots.
    dependencies = (
        "memory",
        "dual_track",
        "evaluation",
        "regime",
        "credit",
        "prediction_error",
        "rupture_state",
        "dialogue_external_outcome",
    )
    default_wiring_level = WiringLevel.SHADOW

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
        prediction_error_snapshot = upstream["prediction_error"]
        rupture_state_snapshot = upstream.get("rupture_state")
        external_outcome_snapshot = upstream.get("dialogue_external_outcome")
        rupture_value = None
        if rupture_state_snapshot is not None and isinstance(
            rupture_state_snapshot.value, RuptureStateSnapshot
        ):
            rupture_value = rupture_state_snapshot.value
        external_outcome_value = None
        if external_outcome_snapshot is not None and isinstance(
            external_outcome_snapshot.value, DialogueExternalOutcomeSnapshot
        ):
            external_outcome_value = external_outcome_snapshot.value
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
                prediction_error_snapshot=prediction_error_snapshot.value
                if isinstance(prediction_error_snapshot.value, PredictionErrorSnapshot)
                else None,
                regime_snapshot=regime_snapshot.value if isinstance(regime_snapshot.value, RegimeSnapshot) else None,
                rupture_state_snapshot=rupture_value,
                dialogue_external_outcome_snapshot=external_outcome_value,
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
                prediction_error_snapshot=kwargs.get("prediction_error_snapshot")
                if isinstance(kwargs.get("prediction_error_snapshot"), PredictionErrorSnapshot)
                else None,
                regime_snapshot=kwargs.get("regime_snapshot")
                if isinstance(kwargs.get("regime_snapshot"), RegimeSnapshot)
                else None,
            )
        )


def enrich_reflection_snapshot_with_rupture_repair(
    *,
    reflection_snapshot: ReflectionSnapshot,
    rupture_state_snapshot: RuptureStateSnapshot | None,
    external_outcome_snapshot: DialogueExternalOutcomeSnapshot | None,
    regime_snapshot: RegimeSnapshot | None,
    user_scope: str,
    timestamp_ms: int,
) -> ReflectionSnapshot:
    """Append rupture-repair durable entries to a reflection snapshot.

    Used by ``run_final_wiring_turn`` to merge SHADOW rupture_state into
    the reflection snapshot that feeds the session-post slow loop. The
    underlying ``ReflectionModule`` runs during propagate and at that
    point only sees ACTIVE upstream snapshots; SHADOW rupture_state is
    not in that view. Post-propagate enrichment reads the shadow
    snapshot and splices the rupture-repair memory entry into
    ``memory_consolidation.new_durable_entries`` so
    ``ReflectionEngine.apply`` can write it durably through the usual
    checkpoint path.
    """

    from dataclasses import replace

    entries = rupture_repair_memory_entries(
        rupture_state_snapshot=rupture_state_snapshot,
        external_outcome_snapshot=external_outcome_snapshot,
        regime_snapshot=regime_snapshot,
        user_scope=user_scope,
        timestamp_ms=timestamp_ms,
    )
    if not entries:
        return reflection_snapshot
    enriched_memory = MemoryConsolidation(
        new_durable_entries=reflection_snapshot.memory_consolidation.new_durable_entries
        + entries,
        promoted_entries=reflection_snapshot.memory_consolidation.promoted_entries,
        decayed_entries=reflection_snapshot.memory_consolidation.decayed_entries,
        beliefs_updated=reflection_snapshot.memory_consolidation.beliefs_updated,
    )
    return replace(reflection_snapshot, memory_consolidation=enriched_memory)
