from __future__ import annotations

from dataclasses import dataclass

from volvence_zero.credit import (
    CreditModule,
    CreditSnapshot,
    GateDecision,
    ModificationGate,
    SelfModificationRecord,
    derive_metacontroller_credit_records,
    derive_runtime_adaptation_audit_records,
    extend_credit_snapshot,
    has_blocking_writeback,
)
from volvence_zero.dual_track import DualTrackModule
from volvence_zero.evaluation import EvaluationBackbone, EvaluationModule, EvaluationSnapshot
from volvence_zero.internal_rl import (
    CausalPolicyCheckpoint,
    DualTrackOptimizationReport,
    DualTrackRollout,
    InternalRLSandbox,
    derive_abstract_action_credit,
)
from volvence_zero.joint_loop.pipeline import RareHeavyArtifact
from volvence_zero.memory import CMSMemoryCore, MemoryStore, MemoryStoreCheckpoint
from volvence_zero.reflection import ReflectionEngine, ReflectionModule, WritebackMode
from volvence_zero.regime import RegimeModule
from volvence_zero.runtime import EventRecorder, SlotRegistry, Snapshot, WiringLevel, propagate
from volvence_zero.runtime.kernel import stable_value_hash
from volvence_zero.substrate import (
    ResidualSequenceStep,
    SimulatedResidualSubstrateAdapter,
    SubstrateSnapshot,
    SurfaceKind,
    TraceStep,
    TrainingTrace,
)
from volvence_zero.temporal import (
    FullLearnedTemporalPolicy,
    MetacontrollerSSLTrainer,
    MetacontrollerRuntimeState,
    TemporalModule,
)
from volvence_zero.memory import MemoryModule
from volvence_zero.substrate import SubstrateModule


@dataclass(frozen=True)
class JointCycleReport:
    cycle_index: int
    acceptance_passed: bool
    ssl_prediction_loss: float
    ssl_kl_loss: float
    ssl_posterior_drift: float
    total_reward: float
    task_reward: float
    relationship_reward: float
    ssl_rollback_applied: bool
    policy_rollback_applied: bool
    rollback_reasons: tuple[str, ...]
    optimization_summary: str
    policy_objective: float
    kernel_score_count: int
    applied_operations: tuple[str, ...]
    metacontroller_state: MetacontrollerRuntimeState | None
    cms_description: str
    owner_path: str
    schedule_telemetry: tuple[tuple[str, int], ...]
    description: str


@dataclass(frozen=True)
class JointLoopSchedule:
    ssl_interval: int = 1
    rl_interval: int = 3


@dataclass(frozen=True)
class ScheduledJointLoopResult:
    turn_index: int
    schedule_action: str
    cycle_report: JointCycleReport | None
    ssl_prediction_loss: float
    ssl_kl_loss: float
    metacontroller_state: MetacontrollerRuntimeState | None
    cms_description: str
    owner_path: str
    schedule_telemetry: tuple[tuple[str, int], ...]
    description: str


@dataclass(frozen=True)
class RareHeavyImportCheckpoint:
    artifact_id: str
    policy_checkpoint: CausalPolicyCheckpoint
    memory_checkpoint: MemoryStoreCheckpoint


@dataclass(frozen=True)
class RareHeavyImportResult:
    artifact_id: str
    applied_operations: tuple[str, ...]
    checkpoint: RareHeavyImportCheckpoint
    description: str


class ETANLJointLoop:
    """Minimal SSL-RL alternation loop over the stage-two building blocks."""

    owner_path = "online-joint-loop"

    def __init__(
        self,
        *,
        policy: FullLearnedTemporalPolicy | None = None,
        memory_store: MemoryStore | None = None,
    ) -> None:
        self._policy = policy or FullLearnedTemporalPolicy()
        self._sandbox = InternalRLSandbox(policy=self._policy)
        self._ssl_trainer = MetacontrollerSSLTrainer()
        self._memory_store = memory_store or MemoryStore(learned_core=CMSMemoryCore())
        self._evaluation_backbone = EvaluationBackbone()
        self._regime_module = RegimeModule(wiring_level=WiringLevel.ACTIVE)
        self._previous_total_reward: float | None = None
        self._previous_metacontroller_state: MetacontrollerRuntimeState | None = None
        self._previous_family_signals: dict[str, float] = {}

    def _apply_temporal_reflection_writeback(
        self,
        *,
        temporal_module: TemporalModule,
        reflection_snapshot,
        credit_snapshot: CreditSnapshot,
        timestamp_ms: int,
        apply_enabled: bool,
    ) -> tuple[tuple[str, ...], tuple[SelfModificationRecord, ...]]:
        temporal_prior_update = reflection_snapshot.policy_consolidation.temporal_prior_update
        if temporal_prior_update is None:
            return ((), ())
        before_hash = stable_value_hash(temporal_module.policy.export_parameters())
        if not apply_enabled:
            return (
                ("temporal-prior:writeback-mode-not-apply",),
                (
                    SelfModificationRecord(
                        target=temporal_prior_update.target,
                        gate=ModificationGate.BACKGROUND,
                        decision=GateDecision.BLOCK,
                        old_value_hash=before_hash,
                        new_value_hash=before_hash,
                        justification="Joint loop skipped reflection-to-temporal writeback because apply_writeback is disabled.",
                        timestamp_ms=timestamp_ms,
                        is_reversible=True,
                    ),
                ),
            )
        if has_blocking_writeback(credit_snapshot, target_prefix=temporal_prior_update.target):
            return (
                ("temporal-prior:credit-gate-block",),
                (
                    SelfModificationRecord(
                        target=temporal_prior_update.target,
                        gate=ModificationGate.BACKGROUND,
                        decision=GateDecision.BLOCK,
                        old_value_hash=before_hash,
                        new_value_hash=before_hash,
                        justification="Joint loop blocked reflection-to-temporal writeback via target-specific credit gate.",
                        timestamp_ms=timestamp_ms,
                        is_reversible=True,
                    ),
                ),
            )
        applied_operations = temporal_module.policy.apply_reflection_prior_update(update=temporal_prior_update)
        after_hash = stable_value_hash(temporal_module.policy.export_parameters())
        return (
            applied_operations,
            (
                SelfModificationRecord(
                    target=temporal_prior_update.target,
                    gate=ModificationGate.BACKGROUND,
                    decision=GateDecision.ALLOW,
                    old_value_hash=before_hash,
                    new_value_hash=after_hash,
                    justification=temporal_prior_update.description,
                    timestamp_ms=timestamp_ms,
                    is_reversible=True,
                ),
            ),
        )

    @property
    def memory_store(self) -> MemoryStore:
        return self._memory_store

    @property
    def temporal_policy(self) -> FullLearnedTemporalPolicy:
        return self._policy

    async def run_cycle(
        self,
        *,
        cycle_index: int,
        trace: TrainingTrace,
        apply_writeback: bool = True,
    ) -> JointCycleReport:
        cycle_checkpoint = self._sandbox.create_checkpoint(checkpoint_id=f"joint-cycle-{cycle_index}")
        ssl_report = self._ssl_trainer.optimize(policy=self._policy, trace=trace)
        if ssl_report.m3_slow_momentum_signal and self._memory_store is not None:
            self._memory_store.observe_encoder_feedback(
                encoder_signal=ssl_report.m3_slow_momentum_signal,
                timestamp_ms=cycle_index + 1,
            )
        substrate_snapshots = tuple(self._snapshot_from_trace_step(step, trace) for step in trace.steps)
        if self._previous_family_signals:
            self._sandbox._env.set_evaluation_signals(self._previous_family_signals)
        dual_track_rollout = self._sandbox.rollout_dual_track(
            rollout_id=f"joint-{cycle_index}",
            substrate_steps=substrate_snapshots,
        )
        optimization_report = self._sandbox.optimize(dual_track_rollout)
        if not isinstance(optimization_report, DualTrackOptimizationReport):
            raise TypeError("dual-track optimization must return DualTrackOptimizationReport.")
        session_id = f"joint-session-{cycle_index}"
        wave_id = f"joint-wave-{cycle_index}"
        modules = [
            SubstrateModule(
                adapter=SimulatedResidualSubstrateAdapter(trace=trace),
                wiring_level=WiringLevel.ACTIVE,
            ),
            MemoryModule(store=self._memory_store, wiring_level=WiringLevel.ACTIVE),
            DualTrackModule(wiring_level=WiringLevel.ACTIVE),
            EvaluationModule(
                backbone=self._evaluation_backbone,
                session_id=session_id,
                wave_id=wave_id,
                wiring_level=WiringLevel.ACTIVE,
            ),
            self._regime_module,
            CreditModule(wiring_level=WiringLevel.ACTIVE),
            ReflectionModule(
                engine=ReflectionEngine(writeback_mode=WritebackMode.PROPOSAL_ONLY),
                wiring_level=WiringLevel.ACTIVE,
            ),
            TemporalModule(policy=self._policy, wiring_level=WiringLevel.ACTIVE),
        ]
        temporal_module = modules[-1]
        recorder = EventRecorder()
        active_snapshots = await propagate(
            modules,
            registry=SlotRegistry(),
            recorder=recorder,
            shadow_snapshots={},
            session_id=session_id,
            wave_id=wave_id,
        )
        enriched_credit_snapshot = self._enrich_credit_snapshot(
            active_snapshots,
            dual_track_rollout=dual_track_rollout,
        )
        total_reward = (
            dual_track_rollout.task_rollout.total_reward
            + dual_track_rollout.relationship_rollout.total_reward
        )
        evaluation_snapshot = active_snapshots["evaluation"].value
        self._modulate_ssl_learning_rate(evaluation_snapshot)
        pre_rollback_metacontroller_state = temporal_module.export_runtime_state()
        rollback_reasons = self._rollback_reasons(
            total_reward=total_reward,
            evaluation_snapshot=evaluation_snapshot,
            optimization_report=optimization_report,
            metacontroller_state=pre_rollback_metacontroller_state,
        )
        rollback_required = bool(rollback_reasons)
        policy_rollback_applied = False
        ssl_rollback_applied = False
        if rollback_required:
            self._sandbox.restore_checkpoint(cycle_checkpoint)
            policy_rollback_applied = True
            ssl_rollback_applied = True
        metacontroller_state = self._policy.export_runtime_state()
        policy_objective = (
            optimization_report.task_report.surrogate_objective
            + optimization_report.relationship_report.surrogate_objective
        )
        kernel_scores = self._evaluation_backbone.record_metacontroller_evidence(
            session_id=session_id,
            wave_id=wave_id,
            timestamp_ms=active_snapshots["evaluation"].timestamp_ms + 1,
            metacontroller_state=metacontroller_state,
            policy_objective=policy_objective,
            rollback_reasons=rollback_reasons,
        )
        regime_operations = self._regime_module.apply_metacontroller_evidence(
            metacontroller_state=metacontroller_state,
            rollback_reasons=rollback_reasons,
        )
        enriched_credit_snapshot = extend_credit_snapshot(
            credit_snapshot=enriched_credit_snapshot,
            extra_records=derive_metacontroller_credit_records(
                metacontroller_state=metacontroller_state,
                policy_objective=policy_objective,
                rollback_reasons=rollback_reasons,
                timestamp_ms=active_snapshots["credit"].timestamp_ms + 150,
            ),
            extra_modifications=derive_runtime_adaptation_audit_records(
                rollback_reasons=rollback_reasons,
                metacontroller_state_description=(
                    metacontroller_state.description if metacontroller_state is not None else None
                ),
                timestamp_ms=active_snapshots["credit"].timestamp_ms + 200,
                rollback_applied=policy_rollback_applied,
            ),
        )
        reflection_snapshot = ReflectionEngine(writeback_mode=WritebackMode.APPLY).reflect(
            timestamp_ms=active_snapshots["reflection"].timestamp_ms + 1,
            memory_snapshot=active_snapshots["memory"].value,
            dual_track_snapshot=active_snapshots["dual_track"].value,
            evaluation_snapshot=active_snapshots["evaluation"].value,
            credit_snapshot=enriched_credit_snapshot,
            regime_snapshot=active_snapshots["regime"].value,
        )
        temporal_writeback_operations, temporal_writeback_audits = self._apply_temporal_reflection_writeback(
            temporal_module=temporal_module,
            reflection_snapshot=reflection_snapshot,
            credit_snapshot=enriched_credit_snapshot,
            timestamp_ms=active_snapshots["credit"].timestamp_ms + 175,
            apply_enabled=apply_writeback,
        )
        if temporal_writeback_audits:
            enriched_credit_snapshot = extend_credit_snapshot(
                credit_snapshot=enriched_credit_snapshot,
                extra_modifications=temporal_writeback_audits,
            )
        applied_operations: tuple[str, ...] = regime_operations
        if apply_writeback:
            applied_operations = applied_operations + ReflectionEngine(writeback_mode=WritebackMode.APPLY).apply(
                memory_store=self._memory_store,
                reflection_snapshot=reflection_snapshot,
                credit_snapshot=enriched_credit_snapshot,
                regime_module=self._regime_module,
                checkpoint_id=f"{session_id}:{wave_id}",
            ).applied_operations
        applied_operations = applied_operations + temporal_writeback_operations
        if ssl_rollback_applied:
            applied_operations = applied_operations + ("ssl-rollback",)
        if policy_rollback_applied:
            applied_operations = applied_operations + ("policy-rollback",)
        self._previous_total_reward = total_reward
        self._previous_metacontroller_state = metacontroller_state
        self._previous_family_signals = self._evaluation_backbone.family_signals(evaluation_snapshot)
        return JointCycleReport(
            cycle_index=cycle_index,
            acceptance_passed="reflection" in active_snapshots and bool(recorder.events),
            ssl_prediction_loss=ssl_report.prediction_loss,
            ssl_kl_loss=ssl_report.kl_loss,
            ssl_posterior_drift=ssl_report.posterior_drift,
            total_reward=total_reward,
            task_reward=dual_track_rollout.task_rollout.total_reward,
            relationship_reward=dual_track_rollout.relationship_rollout.total_reward,
            ssl_rollback_applied=ssl_rollback_applied,
            policy_rollback_applied=policy_rollback_applied,
            rollback_reasons=rollback_reasons,
            optimization_summary=optimization_report.description,
            policy_objective=policy_objective,
            kernel_score_count=len(kernel_scores),
            applied_operations=applied_operations,
            metacontroller_state=metacontroller_state,
            cms_description=self._memory_store.learned_core.snapshot().description
            if self._memory_store.learned_core is not None
            else "No CMS core attached.",
            owner_path=self.owner_path,
            schedule_telemetry=(
                ("cycle_index", cycle_index),
                ("ssl_interval", 1),
                ("rl_interval", 1),
                ("ssl_due", 1),
                ("rl_due", 1),
            ),
            description=(
                f"Joint ETA/NL cycle {cycle_index} owner={self.owner_path} ran ssl(pred={ssl_report.prediction_loss:.2f}, "
                f"kl={ssl_report.kl_loss:.2f}, drift={ssl_report.posterior_drift:.2f}) and dual-track rollout "
                f"task={dual_track_rollout.task_rollout.total_reward:.2f}, "
                f"relationship={dual_track_rollout.relationship_rollout.total_reward:.2f}, "
                f"rollback={'on' if policy_rollback_applied else 'off'}, "
                f"reasons={','.join(rollback_reasons) if rollback_reasons else 'none'}, "
                f"controller={metacontroller_state.description if metacontroller_state is not None else 'unavailable'}, "
                f"kernel_scores={len(kernel_scores)}, with {len(applied_operations)} bounded writeback operations."
            ),
        )

    async def run_scheduled_step(
        self,
        *,
        turn_index: int,
        trace: TrainingTrace,
        schedule: JointLoopSchedule | None = None,
        apply_writeback: bool = True,
    ) -> ScheduledJointLoopResult:
        active_schedule = schedule or JointLoopSchedule()
        schedule_telemetry = self._schedule_telemetry(
            turn_index=turn_index,
            schedule=active_schedule,
        )
        cms_description = (
            self._memory_store.learned_core.snapshot().description
            if self._memory_store.learned_core is not None
            else "No CMS core attached."
        )
        if active_schedule.rl_interval > 0 and turn_index % active_schedule.rl_interval == 0:
            cycle_report = await self.run_cycle(
                cycle_index=turn_index,
                trace=trace,
                apply_writeback=apply_writeback,
            )
            return ScheduledJointLoopResult(
                turn_index=turn_index,
                schedule_action="full-cycle",
                cycle_report=cycle_report,
                ssl_prediction_loss=cycle_report.ssl_prediction_loss,
                ssl_kl_loss=cycle_report.ssl_kl_loss,
                metacontroller_state=cycle_report.metacontroller_state,
                cms_description=cycle_report.cms_description,
                owner_path=self.owner_path,
                schedule_telemetry=schedule_telemetry,
                description=cycle_report.description,
            )
        if active_schedule.ssl_interval > 0 and turn_index % active_schedule.ssl_interval == 0:
            ssl_report = self._ssl_trainer.optimize(policy=self._policy, trace=trace)
            metacontroller_state = self._policy.export_runtime_state()
            return ScheduledJointLoopResult(
                turn_index=turn_index,
                schedule_action="ssl-only",
                cycle_report=None,
                ssl_prediction_loss=ssl_report.prediction_loss,
                ssl_kl_loss=ssl_report.kl_loss,
                metacontroller_state=metacontroller_state,
                cms_description=cms_description,
                owner_path=self.owner_path,
                schedule_telemetry=schedule_telemetry,
                description=(
                    f"Scheduled joint loop owner={self.owner_path} ran ssl-only at turn {turn_index} with "
                    f"pred={ssl_report.prediction_loss:.2f}, kl={ssl_report.kl_loss:.2f}."
                ),
            )
        metacontroller_state = self._policy.export_runtime_state()
        return ScheduledJointLoopResult(
            turn_index=turn_index,
            schedule_action="evidence-only",
            cycle_report=None,
            ssl_prediction_loss=0.0,
            ssl_kl_loss=0.0,
            metacontroller_state=metacontroller_state,
            cms_description=cms_description,
            owner_path=self.owner_path,
            schedule_telemetry=schedule_telemetry,
            description=f"Scheduled joint loop owner={self.owner_path} collected evidence only at turn {turn_index}.",
        )

    def _schedule_telemetry(
        self,
        *,
        turn_index: int,
        schedule: JointLoopSchedule,
    ) -> tuple[tuple[str, int], ...]:
        ssl_due = int(schedule.ssl_interval > 0 and turn_index % schedule.ssl_interval == 0)
        rl_due = int(schedule.rl_interval > 0 and turn_index % schedule.rl_interval == 0)
        return (
            ("turn_index", turn_index),
            ("ssl_interval", schedule.ssl_interval),
            ("rl_interval", schedule.rl_interval),
            ("ssl_due", ssl_due),
            ("rl_due", rl_due),
        )

    def apply_rare_heavy_artifact(
        self,
        artifact: RareHeavyArtifact,
        *,
        checkpoint_id: str | None = None,
    ) -> RareHeavyImportResult:
        checkpoint_label = checkpoint_id or f"rare-heavy:{artifact.artifact_id}"
        policy_checkpoint = self._sandbox.create_checkpoint(checkpoint_id=f"{checkpoint_label}:policy")
        memory_checkpoint = self._memory_store.export_rare_heavy_state(checkpoint_id=f"{checkpoint_label}:memory")
        applied_operations = self._policy.apply_rare_heavy_snapshot(artifact.temporal_snapshot)
        if artifact.memory_checkpoint is not None:
            applied_operations = applied_operations + self._memory_store.import_rare_heavy_state(
                artifact.memory_checkpoint
            )
        checkpoint = RareHeavyImportCheckpoint(
            artifact_id=artifact.artifact_id,
            policy_checkpoint=policy_checkpoint,
            memory_checkpoint=memory_checkpoint,
        )
        return RareHeavyImportResult(
            artifact_id=artifact.artifact_id,
            applied_operations=applied_operations,
            checkpoint=checkpoint,
            description=(
                f"Applied rare-heavy artifact {artifact.artifact_id} from {artifact.owner_path} "
                f"with {len(applied_operations)} owner-side imports."
            ),
        )

    def rollback_rare_heavy_import(self, checkpoint: RareHeavyImportCheckpoint) -> tuple[str, ...]:
        self._sandbox.restore_checkpoint(checkpoint.policy_checkpoint)
        self._memory_store.restore_checkpoint(checkpoint.memory_checkpoint)
        return ("rare-heavy:temporal-rollback", "rare-heavy:memory-rollback")

    def _enrich_credit_snapshot(
        self,
        active_snapshots: dict[str, Snapshot[object]],
        *,
        dual_track_rollout: DualTrackRollout,
    ) -> CreditSnapshot:
        credit_snapshot = active_snapshots["credit"].value
        if not isinstance(credit_snapshot, CreditSnapshot):
            raise TypeError("credit snapshot must be CreditSnapshot.")
        return extend_credit_snapshot(
            credit_snapshot=credit_snapshot,
            extra_records=(
                derive_abstract_action_credit(
                    rollout=dual_track_rollout.task_rollout,
                    timestamp_ms=active_snapshots["credit"].timestamp_ms,
                )
                + derive_abstract_action_credit(
                    rollout=dual_track_rollout.relationship_rollout,
                    timestamp_ms=active_snapshots["credit"].timestamp_ms + 100,
                )
            ),
        )

    def _rollback_reasons(
        self,
        *,
        total_reward: float,
        evaluation_snapshot: EvaluationSnapshot,
        optimization_report: DualTrackOptimizationReport,
        metacontroller_state: MetacontrollerRuntimeState | None,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        family_signals = self._evaluation_backbone.family_signals(evaluation_snapshot)
        if family_signals.get("safety", 1.0) < 0.85:
            reasons.append("safety-degraded")
        if family_signals.get("relationship", 1.0) < 0.3:
            reasons.append("relationship-critical")
        if any(alert.startswith("HIGH") or alert.startswith("CRITICAL") for alert in evaluation_snapshot.alerts):
            reasons.append("evaluation-alert")
        if (
            optimization_report.task_report.surrogate_objective < -0.1
            or optimization_report.relationship_report.surrogate_objective < -0.1
        ):
            reasons.append("negative-surrogate")
        if (
            optimization_report.task_report.kl_penalty > 0.4
            or optimization_report.relationship_report.kl_penalty > 0.4
        ):
            reasons.append("excessive-kl")
        if self._metacontroller_drift_exceeds_limit(metacontroller_state):
            reasons.append("metacontroller-drift")
        if self._previous_total_reward is None:
            return tuple(reasons)
        if total_reward + 0.25 < self._previous_total_reward:
            reasons.append("reward-regression")
        return tuple(reasons)

    def _modulate_ssl_learning_rate(self, evaluation_snapshot: EvaluationSnapshot) -> None:
        """Adjust SSL learning rate based on evaluation quality signals."""
        family_signals = self._evaluation_backbone.family_signals(evaluation_snapshot)
        quality_signal = (
            family_signals.get("learning", 0.5) * 0.4
            + family_signals.get("abstraction", 0.5) * 0.3
            + family_signals.get("safety", 1.0) * 0.3
        )
        base_lr = 0.08
        modulated_lr = base_lr * (0.5 + quality_signal)
        self._policy.parameter_store.learning_rate = max(0.01, min(0.15, modulated_lr))

    def _metacontroller_drift_exceeds_limit(
        self,
        metacontroller_state: MetacontrollerRuntimeState | None,
    ) -> bool:
        if metacontroller_state is None or self._previous_metacontroller_state is None:
            return False
        current_temporal = metacontroller_state.temporal_parameters
        previous_temporal = self._previous_metacontroller_state.temporal_parameters
        temporal_shift = max(
            abs(current_temporal.residual_weight - previous_temporal.residual_weight),
            abs(current_temporal.memory_weight - previous_temporal.memory_weight),
            abs(current_temporal.reflection_weight - previous_temporal.reflection_weight),
            abs(current_temporal.switch_bias - previous_temporal.switch_bias),
        )
        persistence_shift = abs(
            metacontroller_state.persistence - self._previous_metacontroller_state.persistence
        )
        current_tracks = dict(metacontroller_state.track_parameters)
        previous_tracks = dict(self._previous_metacontroller_state.track_parameters)
        track_shift = max(
            max(
                abs(current_value - previous_value)
                for current_value, previous_value in zip(current_tracks[track], previous_tracks[track], strict=True)
            )
            for track in current_tracks
        )
        return max(temporal_shift, persistence_shift, track_shift) > 0.35

    def _snapshot_from_trace_step(self, step: TraceStep, trace: TrainingTrace) -> SubstrateSnapshot:
        return SubstrateSnapshot(
            model_id=f"joint-trace:{trace.trace_id}",
            is_frozen=True,
            surface_kind=SurfaceKind.RESIDUAL_STREAM,
            token_logits=tuple(
                min(sum(feature.values) / max(len(feature.values), 1), 1.0)
                for feature in step.feature_surface
            ),
            feature_surface=step.feature_surface,
            residual_activations=step.residual_activations,
            residual_sequence=tuple(
                ResidualSequenceStep(
                    step=prefix_step.step,
                    token=prefix_step.token,
                    feature_surface=prefix_step.feature_surface,
                    residual_activations=prefix_step.residual_activations,
                    description=f"Joint trace token '{prefix_step.token}' at step {prefix_step.step}.",
                )
                for prefix_step in trace.steps[: step.step + 1]
            ),
            unavailable_fields=(),
            description=f"Trace step {step.step} for {trace.trace_id}.",
        )

