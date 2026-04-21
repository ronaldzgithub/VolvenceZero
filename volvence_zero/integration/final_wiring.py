from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from volvence_zero.credit import (
    CreditSnapshot,
    CreditModule,
    GateDecision,
    ModificationGate,
    ModificationProposal,
    SelfModificationRecord,
    derive_delayed_attribution_credit_records,
    derive_learning_evidence_credit_records,
    derive_prediction_error_credit_records,
    has_blocking_writeback,
)
from volvence_zero.dual_track import DualTrackModule, DualTrackSnapshot
from volvence_zero.evaluation import (
    CrossSessionBenchmarkSuite,
    EvaluationBackbone,
    EvaluationReport,
    EvaluationModule,
    EvaluationSnapshot,
    EvolutionDecision,
    EvolutionJudgement,
    JudgementCategory,
)
from volvence_zero.memory import MemoryModule, MemoryStore, Track, build_default_memory_store
from volvence_zero.prediction import PredictedOutcome, PredictionErrorModule, PredictionErrorSnapshot
from volvence_zero.reflection import (
    ReflectionEngine,
    ReflectionModule,
    ReflectionSnapshot,
    WritebackMode,
    WritebackResult,
)
from volvence_zero.regime import RegimeModule, RegimeSnapshot
from volvence_zero.runtime import EventRecorder, SlotRegistry, Snapshot, WiringLevel, propagate
from volvence_zero.runtime.kernel import stable_value_hash
from volvence_zero.substrate import SubstrateAdapter, SubstrateModule
from volvence_zero.temporal import (
    FullLearnedTemporalPolicy,
    MetacontrollerRuntimeState,
    TemporalAbstractionSnapshot,
    TemporalPolicy,
    TemporalAggregateModule,
    TemporalModule,
    TrackTemporalModule,
    build_temporal_runtime_state_aggregate,
)


@dataclass(frozen=True)
class FinalRolloutConfig:
    substrate: WiringLevel = WiringLevel.ACTIVE
    memory: WiringLevel = WiringLevel.ACTIVE
    dual_track: WiringLevel = WiringLevel.ACTIVE
    evaluation: WiringLevel = WiringLevel.ACTIVE
    prediction_error: WiringLevel = WiringLevel.ACTIVE
    regime: WiringLevel = WiringLevel.ACTIVE
    credit: WiringLevel = WiringLevel.ACTIVE
    reflection: WiringLevel = WiringLevel.ACTIVE
    temporal: WiringLevel = WiringLevel.ACTIVE
    kill_switches: frozenset[str] = frozenset()

    def level_for(self, module_name: str, default: WiringLevel) -> WiringLevel:
        if module_name in self.kill_switches:
            return WiringLevel.DISABLED
        return {
            "substrate": self.substrate,
            "memory": self.memory,
            "dual_track": self.dual_track,
            "evaluation": self.evaluation,
            "prediction_error": self.prediction_error,
            "regime": self.regime,
            "credit": self.credit,
            "reflection": self.reflection,
            "temporal": self.temporal,
        }.get(module_name, default)


def reflection_promotion_eligible(
    *,
    evaluation_snapshot: EvaluationSnapshot,
    min_accuracy: float = 0.6,
    min_observations: int = 5,
    reflection_engine: ReflectionEngine | None = None,
) -> tuple[bool, str]:
    """Evaluate whether reflection should be promoted from SHADOW to ACTIVE.

    Returns (eligible, reason). Promotion requires:
    1. reflection_accuracy >= min_accuracy
    2. At least min_observations proposal outcomes tracked

    This is a read-only evaluation — actual promotion is a manual config change.
    The caller should separately verify via evolution judge that no
    UNSAFE_MUTATION verdicts have occurred in the relevant window.
    """
    accuracy = evaluation_snapshot.reflection_accuracy
    if reflection_engine is None:
        return (False, f"no reflection engine; accuracy={accuracy:.2f}")
    ledger = reflection_engine.proposal_outcome_ledger
    if len(ledger) < min_observations:
        return (False, f"insufficient observations: {len(ledger)}/{min_observations}")
    if accuracy < min_accuracy:
        return (False, f"accuracy {accuracy:.2f} < threshold {min_accuracy:.2f}")
    return (True, f"eligible: accuracy={accuracy:.2f} observations={len(ledger)}")


@dataclass(frozen=True)
class FinalAcceptanceReport:
    passed: bool
    active_slots: tuple[str, ...]
    shadow_slots: tuple[str, ...]
    disabled_slots: tuple[str, ...]
    issues: tuple[str, ...]
    recommendations: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class FinalIntegrationResult:
    active_snapshots: dict[str, Snapshot[Any]]
    shadow_snapshots: dict[str, Snapshot[Any]]
    acceptance_report: FinalAcceptanceReport
    event_count: int
    writeback_result: WritebackResult | None
    writeback_source: str | None
    temporal_runtime_state: MetacontrollerRuntimeState | None
    track_runtime_states: tuple[tuple[str, MetacontrollerRuntimeState], ...] = ()
    prediction_error_snapshot: PredictionErrorSnapshot | None = None
    reflection_promotion_eligible: bool = False
    reflection_promotion_reason: str = ""
    evolution_judgement: EvolutionJudgement | None = None
    cross_session_verdict: str = ""
    session_post_writeback_request: "SessionPostWritebackRequest | None" = None


@dataclass(frozen=True)
class SessionPostWritebackRequest:
    context_session_id: str
    source_wave_id: str
    session_report: EvaluationReport
    reflection_snapshot: ReflectionSnapshot
    credit_snapshot: CreditSnapshot | None
    evolution_judgement: EvolutionJudgement | None
    cross_session_verdict: str
    writeback_source: str | None
    reflection_apply_enabled: bool
    structural_writeback_allowed: bool
    checkpoint_id: str
    description: str


def _judge_allows_structural_writeback(evolution_judgement: EvolutionJudgement | None) -> bool:
    if evolution_judgement is None:
        return True
    return evolution_judgement.decision is EvolutionDecision.PROMOTE or (
        evolution_judgement.decision is EvolutionDecision.HOLD
        and evolution_judgement.category is not JudgementCategory.UNSAFE_MUTATION
    )


def _apply_temporal_reflection_writeback(
    *,
    temporal_module: TemporalModule | None,
    reflection_snapshot: Snapshot[Any] | None,
    credit_snapshot: CreditSnapshot | None,
    timestamp_ms: int,
    apply_enabled: bool,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[SelfModificationRecord, ...]]:
    if (
        temporal_module is None
        or reflection_snapshot is None
        or not isinstance(reflection_snapshot.value, ReflectionSnapshot)
    ):
        return ((), (), ())
    temporal_prior_update = reflection_snapshot.value.policy_consolidation.temporal_prior_update
    if temporal_prior_update is None:
        return ((), (), ())
    target_groups = temporal_prior_update.target_groups or ("base-weights",)
    before_hash = stable_value_hash(temporal_module.policy.export_parameters())
    if not apply_enabled:
        blocked_operations = ("temporal-prior:writeback-mode-not-apply",)
        return (
            (),
            blocked_operations,
            tuple(
                SelfModificationRecord(
                    target=f"{temporal_prior_update.target}.{group}",
                    gate=ModificationGate.BACKGROUND,
                    decision=GateDecision.BLOCK,
                    old_value_hash=before_hash,
                    new_value_hash=before_hash,
                    justification="Reflection-to-temporal writeback skipped because reflection apply mode is disabled.",
                    timestamp_ms=timestamp_ms,
                    is_reversible=True,
                )
                for group in target_groups
            ),
        )
    blocked_groups = (
        tuple(
            group
            for group in target_groups
            if credit_snapshot is not None
            and has_blocking_writeback(
                credit_snapshot,
                target_prefix=f"{temporal_prior_update.target}.{group}",
            )
        )
    )
    allowed_groups = tuple(group for group in target_groups if group not in blocked_groups)
    if not allowed_groups:
        blocked_operations = ("temporal-prior:credit-gate-block",)
        return (
            (),
            blocked_operations,
            tuple(
                SelfModificationRecord(
                    target=f"{temporal_prior_update.target}.{group}",
                    gate=ModificationGate.BACKGROUND,
                    decision=GateDecision.BLOCK,
                    old_value_hash=before_hash,
                    new_value_hash=before_hash,
                    justification="Reflection-to-temporal writeback blocked by target-specific credit gate.",
                    timestamp_ms=timestamp_ms,
                    is_reversible=True,
                )
                for group in blocked_groups
            ),
        )
    applied_operations = temporal_module.policy.apply_reflection_prior_update(
        update=temporal_prior_update,
        allowed_target_groups=allowed_groups,
    )
    blocked_operations = ("temporal-prior:partial-credit-gate-block",) if blocked_groups else ()
    after_hash = stable_value_hash(temporal_module.policy.export_parameters())
    audit_records = tuple(
        [
            SelfModificationRecord(
                target=f"{temporal_prior_update.target}.{group}",
                gate=ModificationGate.BACKGROUND,
                decision=GateDecision.ALLOW,
                old_value_hash=before_hash,
                new_value_hash=after_hash,
                justification=temporal_prior_update.description,
                timestamp_ms=timestamp_ms,
                is_reversible=True,
            )
            for group in allowed_groups
        ]
        + [
            SelfModificationRecord(
                target=f"{temporal_prior_update.target}.{group}",
                gate=ModificationGate.BACKGROUND,
                decision=GateDecision.BLOCK,
                old_value_hash=before_hash,
                new_value_hash=before_hash,
                justification="Reflection-to-temporal writeback blocked by target-specific credit gate.",
                timestamp_ms=timestamp_ms,
                is_reversible=True,
            )
            for group in blocked_groups
        ]
    )
    return (applied_operations, blocked_operations, audit_records)


def _build_session_post_writeback_request(
    *,
    context_session_id: str,
    source_wave_id: str,
    session_report: EvaluationReport | None,
    reflection_snapshot: Snapshot[Any] | None,
    credit_snapshot: Snapshot[Any] | None,
    evolution_judgement: EvolutionJudgement | None,
    cross_session_verdict: str,
    writeback_source: str | None,
    reflection_mode: WritebackMode,
    structural_writeback_allowed: bool,
) -> SessionPostWritebackRequest | None:
    if (
        session_report is None
        or reflection_snapshot is None
        or not isinstance(reflection_snapshot.value, ReflectionSnapshot)
    ):
        return None
    credit_value = (
        credit_snapshot.value
        if credit_snapshot is not None and isinstance(credit_snapshot.value, CreditSnapshot)
        else None
    )
    return SessionPostWritebackRequest(
        context_session_id=context_session_id,
        source_wave_id=source_wave_id,
        session_report=session_report,
        reflection_snapshot=reflection_snapshot.value,
        credit_snapshot=credit_value,
        evolution_judgement=evolution_judgement,
        cross_session_verdict=cross_session_verdict,
        writeback_source=writeback_source,
        reflection_apply_enabled=reflection_mode is WritebackMode.APPLY,
        structural_writeback_allowed=structural_writeback_allowed,
        checkpoint_id=f"{context_session_id}:{source_wave_id}:session-post",
        description=(
            f"Deferred session-post slow loop request for {context_session_id} from {source_wave_id}; "
            f"apply={'on' if reflection_mode is WritebackMode.APPLY else 'off'} "
            f"structural={'allow' if structural_writeback_allowed else 'block'}."
        ),
    )


def apply_session_post_writeback_request(
    *,
    request: SessionPostWritebackRequest,
    memory_store: MemoryStore,
    temporal_policy: TemporalPolicy | None,
    regime_module: RegimeModule | None,
) -> tuple[WritebackResult | None, tuple[SelfModificationRecord, ...]]:
    apply_enabled = request.reflection_apply_enabled and request.structural_writeback_allowed
    if apply_enabled:
        writeback_result = ReflectionEngine(writeback_mode=WritebackMode.APPLY).apply(
            memory_store=memory_store,
            reflection_snapshot=request.reflection_snapshot,
            credit_snapshot=request.credit_snapshot,
            regime_module=regime_module,
            checkpoint_id=request.checkpoint_id,
        )
    elif request.reflection_apply_enabled:
        writeback_result = WritebackResult(
            applied_operations=(),
            blocked_operations=("evolution-judge-block",),
            checkpoint=None,
            description="Session-post reflection writeback was blocked by the evolution judge.",
        )
    else:
        writeback_result = WritebackResult(
            applied_operations=(),
            blocked_operations=("writeback-mode-not-apply",),
            checkpoint=None,
            description="Session-post reflection writeback is disabled because reflection apply mode is not active.",
        )
    temporal_module = TemporalModule(policy=temporal_policy, wiring_level=WiringLevel.ACTIVE)
    reflection_snapshot = Snapshot(
        slot_name="reflection",
        owner="ReflectionModule",
        version=0,
        timestamp_ms=request.session_report.timestamp_ms,
        value=request.reflection_snapshot,
    )
    temporal_writeback_operations, temporal_writeback_blocks, temporal_audits = _apply_temporal_reflection_writeback(
        temporal_module=temporal_module,
        reflection_snapshot=reflection_snapshot,
        credit_snapshot=request.credit_snapshot,
        timestamp_ms=request.session_report.timestamp_ms + 1,
        apply_enabled=apply_enabled,
    )
    if (temporal_writeback_operations or temporal_writeback_blocks) and writeback_result is not None:
        writeback_result = replace(
            writeback_result,
            applied_operations=writeback_result.applied_operations + temporal_writeback_operations,
            blocked_operations=writeback_result.blocked_operations + temporal_writeback_blocks,
            description=(
                f"{writeback_result.description} temporal_ops={len(temporal_writeback_operations)} "
                f"temporal_blocks={len(temporal_writeback_blocks)}."
            ),
        )
    elif temporal_writeback_operations or temporal_writeback_blocks:
        writeback_result = WritebackResult(
            applied_operations=temporal_writeback_operations,
            blocked_operations=temporal_writeback_blocks,
            checkpoint=None,
            description=(
                f"Session-post temporal writeback produced {len(temporal_writeback_operations)} applied ops and "
                f"{len(temporal_writeback_blocks)} blocked ops."
            ),
        )
    return (writeback_result, temporal_audits)


def build_final_runtime_modules(
    *,
    config: FinalRolloutConfig,
    substrate_adapter: SubstrateAdapter,
    memory_store: MemoryStore | None = None,
    evaluation_backbone: EvaluationBackbone | None = None,
    credit_proposals: tuple[ModificationProposal, ...] = (),
    reflection_mode: WritebackMode = WritebackMode.PROPOSAL_ONLY,
    temporal_policy: TemporalPolicy | None = None,
    world_temporal_policy: TemporalPolicy | None = None,
    self_temporal_policy: TemporalPolicy | None = None,
    prediction_module: PredictionErrorModule | None = None,
    regime_module: RegimeModule | None = None,
    session_id: str = "runtime-session",
    wave_id: str = "wave-0",
) -> list[Any]:
    return [
        SubstrateModule(
            adapter=substrate_adapter,
            wiring_level=config.level_for("substrate", WiringLevel.ACTIVE),
        ),
        MemoryModule(
            store=memory_store or build_default_memory_store(),
            wiring_level=config.level_for("memory", WiringLevel.SHADOW),
        ),
        DualTrackModule(
            wiring_level=config.level_for("dual_track", WiringLevel.SHADOW),
        ),
        EvaluationModule(
            backbone=evaluation_backbone,
            session_id=session_id,
            wave_id=wave_id,
            wiring_level=config.level_for("evaluation", WiringLevel.ACTIVE),
        ),
        regime_module
        or RegimeModule(
            wiring_level=config.level_for("regime", WiringLevel.SHADOW),
        ),
        prediction_module
        or PredictionErrorModule(
            wiring_level=config.level_for("prediction_error", WiringLevel.ACTIVE),
        ),
        CreditModule(
            pending_proposals=credit_proposals,
            wiring_level=config.level_for("credit", WiringLevel.SHADOW),
        ),
        ReflectionModule(
            engine=ReflectionEngine(writeback_mode=reflection_mode),
            wiring_level=config.level_for("reflection", WiringLevel.SHADOW),
        ),
        TrackTemporalModule(
            track=Track.WORLD,
            policy=world_temporal_policy or temporal_policy or FullLearnedTemporalPolicy(),
            wiring_level=config.level_for("temporal", WiringLevel.SHADOW),
        ),
        TrackTemporalModule(
            track=Track.SELF,
            policy=self_temporal_policy or FullLearnedTemporalPolicy(),
            wiring_level=config.level_for("temporal", WiringLevel.SHADOW),
        ),
        TemporalAggregateModule(
            wiring_level=config.level_for("temporal", WiringLevel.SHADOW),
        ),
    ]


async def run_final_wiring_turn(
    *,
    config: FinalRolloutConfig,
    substrate_adapter: SubstrateAdapter,
    memory_store: MemoryStore | None = None,
    evaluation_backbone: EvaluationBackbone | None = None,
    prior_session_reports: tuple[EvaluationReport, ...] = (),
    upstream_snapshots: dict[str, Snapshot[Any]] | None = None,
    joint_loop_result: object | None = None,
    credit_proposals: tuple[ModificationProposal, ...] = (),
    reflection_mode: WritebackMode = WritebackMode.PROPOSAL_ONLY,
    temporal_policy: TemporalPolicy | None = None,
    world_temporal_policy: TemporalPolicy | None = None,
    self_temporal_policy: TemporalPolicy | None = None,
    prediction_module: PredictionErrorModule | None = None,
    regime_module: RegimeModule | None = None,
    session_id: str = "runtime-session",
    wave_id: str = "wave-0",
    apply_slow_writeback: bool = True,
) -> FinalIntegrationResult:
    modules = build_final_runtime_modules(
        config=config,
        substrate_adapter=substrate_adapter,
        memory_store=memory_store,
        evaluation_backbone=evaluation_backbone,
        credit_proposals=credit_proposals,
        reflection_mode=reflection_mode,
        temporal_policy=temporal_policy,
        world_temporal_policy=world_temporal_policy,
        self_temporal_policy=self_temporal_policy,
        prediction_module=prediction_module,
        regime_module=regime_module,
        session_id=session_id,
        wave_id=wave_id,
    )
    if upstream_snapshots:
        for module in modules:
            previous_snapshot = upstream_snapshots.get(module.slot_name)
            if previous_snapshot is not None:
                module.seed_version(previous_snapshot.version)
    recorder = EventRecorder()
    registry = SlotRegistry()
    if upstream_snapshots:
        registry.seed_versions(upstream_snapshots)
    shadow_snapshots: dict[str, Snapshot[Any]] = {}
    active_snapshots = await propagate(
        modules,
        upstream=upstream_snapshots,
        registry=registry,
        recorder=recorder,
        shadow_snapshots=shadow_snapshots,
        session_id=session_id,
        wave_id=wave_id,
    )
    writeback_result: WritebackResult | None = None
    writeback_source: str | None = None
    reflection_module = next((module for module in modules if isinstance(module, ReflectionModule)), None)
    evaluation_module = next((module for module in modules if isinstance(module, EvaluationModule)), None)
    credit_module = next((module for module in modules if isinstance(module, CreditModule)), None)
    regime_module = next((module for module in modules if isinstance(module, RegimeModule)), None)
    temporal_modules = tuple(
        module
        for module in modules
        if isinstance(module, TrackTemporalModule)
    )
    reflection_snapshot = active_snapshots.get("reflection")
    if reflection_snapshot is None:
        reflection_snapshot = shadow_snapshots.get("reflection")
        if reflection_snapshot is not None:
            writeback_source = "shadow"
    else:
        writeback_source = "active"
    credit_snapshot = active_snapshots.get("credit") or shadow_snapshots.get("credit")
    evolution_judgement: EvolutionJudgement | None = None
    cross_session_verdict = ""
    judge_allows_structural_writeback = reflection_mode is WritebackMode.APPLY
    session_report: EvaluationReport | None = None
    session_post_writeback_request: SessionPostWritebackRequest | None = None
    evaluation_snapshot = active_snapshots.get("evaluation")
    prediction_snapshot_value = (
        active_snapshots.get("prediction_error").value
        if active_snapshots.get("prediction_error") is not None
        and isinstance(active_snapshots.get("prediction_error").value, PredictionErrorSnapshot)
        else None
    )
    reflection_promote = False
    reflection_promote_reason = ""
    if (
        evaluation_module is not None
        and evaluation_snapshot is not None
        and isinstance(evaluation_snapshot.value, EvaluationSnapshot)
    ):
        from volvence_zero.joint_loop.runtime import ScheduledJointLoopResult

        enriched_evaluation = evaluation_snapshot.value
        temporal_snapshot = active_snapshots.get("temporal_abstraction")
        temporal_value = (
            temporal_snapshot.value
            if temporal_snapshot is not None and isinstance(temporal_snapshot.value, TemporalAbstractionSnapshot)
            else None
        )
        enriched_evaluation = evaluation_module.backbone.record_temporal_public_evidence(
            session_id=session_id,
            wave_id=wave_id,
            timestamp_ms=evaluation_snapshot.timestamp_ms + 1,
            base_snapshot=enriched_evaluation,
            temporal_snapshot=temporal_value,
        )
        joint_kernel_scores = (
            joint_loop_result.kernel_scores
            if isinstance(joint_loop_result, ScheduledJointLoopResult)
            else ()
        )
        if joint_kernel_scores:
            enriched_evaluation = evaluation_module.backbone.record_external_scores(
                session_id=session_id,
                wave_id=wave_id,
                timestamp_ms=evaluation_snapshot.timestamp_ms + 2,
                base_snapshot=enriched_evaluation,
                scores=joint_kernel_scores,
                description_suffix=f"Enriched with {len(joint_kernel_scores)} ETA kernel scores.",
            )
        enriched_evaluation = evaluation_module.backbone.record_prediction_error_evidence(
            session_id=session_id,
            wave_id=wave_id,
            timestamp_ms=evaluation_snapshot.timestamp_ms + 3,
            base_snapshot=enriched_evaluation,
            prediction_error_snapshot=prediction_snapshot_value,
        )
        enriched_evaluation = evaluation_module.backbone.record_learning_evidence(
            session_id=session_id,
            wave_id=wave_id,
            timestamp_ms=evaluation_snapshot.timestamp_ms + 4,
            base_snapshot=enriched_evaluation,
            memory_snapshot=active_snapshots.get("memory").value if active_snapshots.get("memory") is not None else None,
            reflection_snapshot=reflection_snapshot.value if reflection_snapshot is not None else None,
            writeback_result=writeback_result,
            joint_loop_result=joint_loop_result,
            regime_snapshot=active_snapshots.get("regime").value if active_snapshots.get("regime") is not None else None,
        )
        reflection_accuracy = 0.0
        if reflection_module is not None:
            reflection_accuracy = reflection_module.engine.proposal_success_rate
        session_report = evaluation_module.backbone.build_session_report(
            session_id=session_id,
            timestamp_ms=evaluation_snapshot.timestamp_ms + 5,
        )
        cross_session_report = None
        if prior_session_reports:
            cross_session_report = evaluation_module.backbone.run_cross_session_benchmark(
                suite=CrossSessionBenchmarkSuite(
                    session_reports=prior_session_reports + (session_report,),
                )
            )
            cross_session_verdict = cross_session_report.verdict
        cycle_report = (
            joint_loop_result.cycle_report
            if isinstance(joint_loop_result, ScheduledJointLoopResult)
            else None
        )
        evolution_judgement = cycle_report.evolution_judgement if cycle_report is not None else None
        if evolution_judgement is None:
            replay_result = evaluation_module.backbone.run_default_evolution_benchmark(
                timestamp_ms=evaluation_snapshot.timestamp_ms + 6,
            )
            evolution_judgement = evaluation_module.backbone.judge_evolution_candidate(
                replay_suite_result=replay_result,
                session_report=session_report,
                cross_session_report=cross_session_report,
            )
        judge_allows_structural_writeback = _judge_allows_structural_writeback(evolution_judgement) and (
            reflection_mode is WritebackMode.APPLY
        )
        enriched_evaluation = replace(
            enriched_evaluation,
            reflection_accuracy=reflection_accuracy,
            longitudinal_verdict=cross_session_verdict,
        )
        active_snapshots["evaluation"] = evaluation_module.publish(enriched_evaluation)
        reflection_promote, reflection_promote_reason = reflection_promotion_eligible(
            evaluation_snapshot=enriched_evaluation,
            reflection_engine=reflection_module.engine if reflection_module is not None else None,
        )
        if credit_module is not None:
            extra_credits = derive_learning_evidence_credit_records(
                evaluation_snapshot=enriched_evaluation,
                timestamp_ms=active_snapshots["evaluation"].timestamp_ms + 1,
            )
            delayed_credits = derive_delayed_attribution_credit_records(
                regime_snapshot=active_snapshots.get("regime").value if active_snapshots.get("regime") is not None else None,
                timestamp_ms=active_snapshots["evaluation"].timestamp_ms + 2,
            )
            extra_credits = extra_credits + delayed_credits
            if prediction_snapshot_value is not None:
                extra_credits = extra_credits + derive_prediction_error_credit_records(
                    prediction_error=prediction_snapshot_value.error,
                    timestamp_ms=active_snapshots["evaluation"].timestamp_ms + 3,
                )
            if extra_credits:
                credit_module.ledger.record_credits(extra_credits)
            active_snapshots["credit"] = credit_module.publish(credit_module.ledger.snapshot())
            credit_snapshot = active_snapshots.get("credit")
    session_post_writeback_request = _build_session_post_writeback_request(
        context_session_id=session_id,
        source_wave_id=wave_id,
        session_report=session_report,
        reflection_snapshot=reflection_snapshot,
        credit_snapshot=credit_snapshot,
        evolution_judgement=evolution_judgement,
        cross_session_verdict=cross_session_verdict,
        writeback_source=writeback_source,
        reflection_mode=reflection_mode,
        structural_writeback_allowed=judge_allows_structural_writeback,
    )
    if (
        apply_slow_writeback
        and session_post_writeback_request is not None
        and memory_store is not None
    ):
        writeback_result, temporal_audits = apply_session_post_writeback_request(
            request=session_post_writeback_request,
            memory_store=memory_store,
            temporal_policy=temporal_policy,
            regime_module=regime_module,
        )
        if credit_module is not None and temporal_audits:
            for record in temporal_audits:
                credit_module.ledger.record_modification(record)
            active_snapshots["credit"] = credit_module.publish(credit_module.ledger.snapshot())
            credit_snapshot = active_snapshots.get("credit")
    elif (
        memory_store is not None
        and reflection_module is not None
        and reflection_snapshot is not None
        and isinstance(reflection_snapshot.value, ReflectionSnapshot)
    ):
        writeback_result = None
    if credit_module is not None:
        active_snapshots["credit"] = credit_module.publish(credit_module.ledger.snapshot())
    acceptance_report = build_acceptance_report(
        config=config,
        active_snapshots=active_snapshots,
        shadow_snapshots=shadow_snapshots,
        recorder=recorder,
    )
    track_runtime_states = tuple(
        (module.track.value, runtime_state)
        for module in temporal_modules
        for runtime_state in (module.export_runtime_state(),)
        if runtime_state is not None
    )
    temporal_runtime_state = None
    if len(track_runtime_states) == 2:
        track_state_map = dict(track_runtime_states)
        temporal_runtime_state = build_temporal_runtime_state_aggregate(
            world_state=track_state_map["world"],
            self_state=track_state_map["self"],
        )
    return FinalIntegrationResult(
        active_snapshots=active_snapshots,
        shadow_snapshots=shadow_snapshots,
        acceptance_report=acceptance_report,
        event_count=len(recorder.events),
        writeback_result=writeback_result,
        writeback_source=writeback_source,
        temporal_runtime_state=temporal_runtime_state,
        track_runtime_states=track_runtime_states,
        prediction_error_snapshot=prediction_snapshot_value,
        reflection_promotion_eligible=reflection_promote,
        reflection_promotion_reason=reflection_promote_reason,
        evolution_judgement=evolution_judgement,
        cross_session_verdict=cross_session_verdict,
        session_post_writeback_request=None if apply_slow_writeback else session_post_writeback_request,
    )


def build_acceptance_report(
    *,
    config: FinalRolloutConfig,
    active_snapshots: dict[str, Snapshot[Any]],
    shadow_snapshots: dict[str, Snapshot[Any]],
    recorder: EventRecorder,
) -> FinalAcceptanceReport:
    active_slots = tuple(sorted(active_snapshots))
    shadow_slots = tuple(sorted(shadow_snapshots))
    disabled_slots = tuple(
        sorted(
            name
            for name in (
                "substrate",
                "memory",
                "dual_track",
                "evaluation",
                "prediction_error",
                "regime",
                "credit",
                "reflection",
                "temporal",
            )
            if config.level_for(name, WiringLevel.DISABLED) is WiringLevel.DISABLED
        )
    )
    issues: list[str] = []
    recommendations: list[str] = []

    expected_active = {
        "substrate",
        "memory",
        "dual_track",
        "evaluation",
        "prediction_error",
        "regime",
        "credit",
    }
    if config.reflection is WiringLevel.ACTIVE:
        expected_active.add("reflection")
    if config.temporal is WiringLevel.ACTIVE:
        expected_active.add("temporal_abstraction")
        expected_active.add("world_temporal")
        expected_active.add("self_temporal")
    missing_active = sorted(expected_active - set(active_slots))
    if missing_active:
        issues.append(f"Missing active slots: {', '.join(missing_active)}")

    violation_count = sum(1 for event in recorder.events if event.event_type == "contract.violation")
    if violation_count:
        issues.append(f"Observed {violation_count} contract violation events during final wiring.")

    if config.reflection is WiringLevel.SHADOW and "reflection" not in shadow_slots and "reflection" not in active_slots:
        issues.append("Reflection wiring configured but no reflection snapshot was produced.")
    if config.reflection is WiringLevel.ACTIVE and "reflection" not in active_slots:
        issues.append("Reflection is configured ACTIVE but did not publish into the active chain.")

    if config.temporal is WiringLevel.SHADOW and "temporal_abstraction" not in shadow_slots and "temporal_abstraction" not in active_slots:
        issues.append("Temporal wiring configured but no temporal snapshot was produced.")
    if config.temporal is WiringLevel.SHADOW and "world_temporal" not in shadow_slots and "world_temporal" not in active_slots:
        issues.append("World temporal wiring configured but no world_temporal snapshot was produced.")
    if config.temporal is WiringLevel.SHADOW and "self_temporal" not in shadow_slots and "self_temporal" not in active_slots:
        issues.append("Self temporal wiring configured but no self_temporal snapshot was produced.")
    if config.temporal is WiringLevel.ACTIVE and "temporal_abstraction" not in active_slots:
        issues.append("Temporal is configured ACTIVE but did not publish into the active chain.")
    if config.temporal is WiringLevel.ACTIVE and "world_temporal" not in active_slots:
        issues.append("Temporal is configured ACTIVE but world_temporal did not publish into the active chain.")
    if config.temporal is WiringLevel.ACTIVE and "self_temporal" not in active_slots:
        issues.append("Temporal is configured ACTIVE but self_temporal did not publish into the active chain.")

    if config.reflection is WiringLevel.ACTIVE:
        recommendations.append("Keep reflection in proposal-only mode until writeback acceptance is proven.")
    if config.temporal is WiringLevel.ACTIVE:
        recommendations.append("Validate temporal active mode against rollout evidence before widening scope.")
    if not recommendations:
        recommendations.append("Core chain is wired; next step is controlled widening via rollout evidence.")

    passed = not issues
    description = (
        f"Final wiring acceptance {'passed' if passed else 'failed'} with "
        f"{len(active_slots)} active slots, {len(shadow_slots)} shadow slots, "
        f"{len(disabled_slots)} disabled slots."
    )
    return FinalAcceptanceReport(
        passed=passed,
        active_slots=active_slots,
        shadow_slots=shadow_slots,
        disabled_slots=disabled_slots,
        issues=tuple(issues),
        recommendations=tuple(recommendations),
        description=description,
    )
