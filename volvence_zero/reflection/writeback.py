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
    temporal_prior_update: "TemporalPriorUpdate | None" = None
    controller_guard_blocked: bool = False
    controller_guard_audit_present: bool = False


@dataclass(frozen=True)
class TemporalStructureProposal:
    proposal_type: str
    family_id: str
    related_family_id: str | None
    confidence: float
    justification: str


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
        alert
        for alert in evaluation_snapshot.alerts
        if "cross-track stability" in alert.lower() or "rollback pressure" in alert.lower()
    )


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
        consolidation_score = self._consolidation_score(
            memory_snapshot=memory_snapshot,
            dual_track_snapshot=dual_track_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            credit_snapshot=credit_snapshot,
        )
        memory_consolidation = self._memory_consolidation(
            timestamp_ms=timestamp_ms,
            memory_snapshot=memory_snapshot,
            credit_snapshot=credit_snapshot,
            consolidation_score=consolidation_score,
            regime_snapshot=regime_snapshot,
        )
        policy_consolidation = self._policy_consolidation(
            dual_track_snapshot=dual_track_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            credit_snapshot=credit_snapshot,
            regime_snapshot=regime_snapshot,
            consolidation_score=consolidation_score,
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
            f"mode={self._writeback_mode.value}, score={consolidation_score.confidence:.2f}."
        )
        return ReflectionSnapshot(
            memory_consolidation=memory_consolidation,
            policy_consolidation=policy_consolidation,
            consolidation_score=consolidation_score,
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
            blocked_targets = tuple(
                record.target for record in credit_snapshot.recent_modifications if record.decision is GateDecision.BLOCK
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
        regime_checkpoint = (
            regime_module.create_checkpoint(checkpoint_id=checkpoint_id or "regime-writeback")
            if regime_module is not None
            else None
        )
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
        if regime_module is not None:
            applied_operations = applied_operations + regime_module.apply_policy_consolidation(
                strategy_updates=reflection_snapshot.policy_consolidation.strategy_priors_updated,
                regime_effectiveness_updates=reflection_snapshot.policy_consolidation.regime_effectiveness_updated,
                strategy_gain=reflection_snapshot.consolidation_score.strategy_gain,
                effectiveness_gain=reflection_snapshot.consolidation_score.regime_effectiveness_gain,
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
        consolidation_score: ConsolidationScore,
        regime_snapshot: RegimeSnapshot | None,
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
        if regime_snapshot is not None and regime_snapshot.delayed_outcomes and regime_snapshot.identity_hints:
            delayed_score = sum(score for _, score in regime_snapshot.delayed_outcomes) / len(regime_snapshot.delayed_outcomes)
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
            if support_presence > task_pressure + 0.08 and consolidation_score.strategy_gain >= 0.03:
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

        if evaluation_snapshot is not None:
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
            description=(
                f"Temporal prior update from reflection confidence={consolidation_score.confidence:.2f}, "
                f"cross_tension={cross_tension:.2f}, world_tension={world_tension:.2f}, self_tension={self_tension:.2f}, "
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
                        confidence=_clamp(0.45 + (0.35 - attribution.outcome_score) + consolidation_score.confidence * 0.15),
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
        promotion_score = _clamp(0.35 + memory_pressure * 0.25 + positive_credit * 0.40 + session_bonus - cross_tension * 0.15)
        decay_score = _clamp(0.10 + negative_credit * 0.75 + alert_pressure * 0.2)
        threshold_delta = max(-0.05, min(0.05, (cross_tension + alert_pressure - promotion_score) * 0.04))
        strategy_gain = max(0.02, min(0.08, 0.02 + promotion_score * 0.05))
        regime_effectiveness_gain = max(0.2, min(0.45, 0.2 + promotion_score * 0.25))
        confidence = _clamp((memory_pressure + positive_credit + (1.0 - cross_tension)) / 3.0)
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
                tensions.append("cross_track_tension_high")
            elif dual_track_snapshot.cross_track_tension > 0.2:
                tensions.append("cross_track_alignment_drift")
            world_tension = dual_track_snapshot.world_track.tension_level
            self_tension = dual_track_snapshot.self_track.tension_level
            if self_tension > world_tension + 0.12:
                tensions.append("self_track_pressure_dominant")
            elif world_tension > self_tension + 0.12:
                tensions.append("world_track_pressure_dominant")
        if evaluation_snapshot is not None:
            if _metric(evaluation_snapshot, "cross_track_stability", default=1.0) < 0.7:
                tensions.append("relationship_stability_soft_drop")
            if _metric(evaluation_snapshot, "warmth", default=1.0) < 0.45:
                tensions.append("warmth_signal_thin")
            if _metric(evaluation_snapshot, "info_integration", default=1.0) < 0.45:
                tensions.append("task_signal_diffuse")
            tensions.extend(_relationship_relevant_alerts(evaluation_snapshot)[:3])
        return tuple(dict.fromkeys(tensions))

    def _lessons(
        self,
        *,
        memory_consolidation: MemoryConsolidation,
        policy_consolidation: PolicyConsolidation,
        tensions: tuple[str, ...],
    ) -> tuple[str, ...]:
        lessons: list[str] = []
        if memory_consolidation.new_durable_entries or memory_consolidation.promoted_entries:
            lessons.append("promote_high_signal_memories")
        if memory_consolidation.beliefs_updated:
            lessons.append("reinforce_recent_high_credit_beliefs")
        if policy_consolidation.strategy_priors_updated:
            lessons.append("adjust_track_priority_from_session_feedback")
        if policy_consolidation.temporal_prior_update is not None:
            temporal_update = policy_consolidation.temporal_prior_update
            strongest_channel = max(
                (
                    ("residual", temporal_update.residual_strength),
                    ("memory", temporal_update.memory_strength),
                    ("reflection", temporal_update.reflection_strength),
                ),
                key=lambda item: item[1],
            )[0]
            lessons.append(f"rebalance_temporal_prior_toward_{strongest_channel}")
            if temporal_update.persistence_delta > 0.02:
                lessons.append("increase_controller_persistence_for_continuity")
            elif temporal_update.persistence_delta < -0.02:
                lessons.append("reduce_controller_persistence_for_faster_recovery")
            if temporal_update.switch_bias_delta > 0.02:
                lessons.append("allow_controller_switch_when_context_shifts")
            elif temporal_update.switch_bias_delta < -0.02:
                lessons.append("hold_controller_before_switching")
            if temporal_update.structure_proposals:
                lessons.append("restructure_action_family_bank")
        if policy_consolidation.controller_guard_blocked:
            lessons.append("respect_metacontroller_runtime_guard")
        elif policy_consolidation.controller_guard_audit_present:
            lessons.append("keep_controller_guard_signal_in_background")
        if tensions:
            lessons.append("review_tension_before_auto_writeback")
        return tuple(dict.fromkeys(lessons))

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
