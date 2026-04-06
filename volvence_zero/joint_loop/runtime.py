from __future__ import annotations

from dataclasses import dataclass

from volvence_zero.credit import (
    CreditModule,
    CreditSnapshot,
    extend_credit_snapshot,
)
from volvence_zero.dual_track import DualTrackModule
from volvence_zero.evaluation import EvaluationModule, EvaluationSnapshot
from volvence_zero.internal_rl import (
    DualTrackOptimizationReport,
    DualTrackRollout,
    InternalRLSandbox,
    derive_abstract_action_credit,
)
from volvence_zero.memory import CMSMemoryCore, MemoryStore
from volvence_zero.reflection import ReflectionEngine, ReflectionModule, WritebackMode
from volvence_zero.regime import RegimeModule
from volvence_zero.runtime import EventRecorder, SlotRegistry, Snapshot, WiringLevel, propagate
from volvence_zero.substrate import (
    SimulatedResidualSubstrateAdapter,
    SubstrateSnapshot,
    SurfaceKind,
    TraceStep,
    TrainingTrace,
)
from volvence_zero.temporal import LearnedLiteTemporalPolicy, TemporalModule
from volvence_zero.memory import MemoryModule
from volvence_zero.substrate import SubstrateModule


@dataclass(frozen=True)
class JointCycleReport:
    cycle_index: int
    acceptance_passed: bool
    total_reward: float
    task_reward: float
    relationship_reward: float
    policy_rollback_applied: bool
    optimization_summary: str
    policy_objective: float
    applied_operations: tuple[str, ...]
    cms_description: str
    description: str


class ETANLJointLoop:
    """Minimal SSL-RL alternation loop over the stage-two building blocks."""

    def __init__(self) -> None:
        self._policy = LearnedLiteTemporalPolicy()
        self._sandbox = InternalRLSandbox(policy=self._policy)
        self._memory_store = MemoryStore(learned_core=CMSMemoryCore())
        self._regime_module = RegimeModule(wiring_level=WiringLevel.ACTIVE)
        self._previous_total_reward: float | None = None

    @property
    def memory_store(self) -> MemoryStore:
        return self._memory_store

    @property
    def temporal_policy(self) -> LearnedLiteTemporalPolicy:
        return self._policy

    async def run_cycle(self, *, cycle_index: int, trace: TrainingTrace) -> JointCycleReport:
        substrate_snapshots = tuple(self._snapshot_from_trace_step(step, trace) for step in trace.steps)
        policy_checkpoint = self._sandbox.create_checkpoint(checkpoint_id=f"joint-policy-{cycle_index}")
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
        rollback_required = self._should_rollback(
            total_reward=total_reward,
            evaluation_snapshot=evaluation_snapshot,
            optimization_report=optimization_report,
        )
        policy_rollback_applied = False
        if rollback_required:
            self._sandbox.restore_checkpoint(policy_checkpoint)
            policy_rollback_applied = True
        reflection_snapshot = active_snapshots["reflection"].value
        applied_operations = ReflectionEngine(writeback_mode=WritebackMode.APPLY).apply(
            memory_store=self._memory_store,
            reflection_snapshot=reflection_snapshot,
            credit_snapshot=enriched_credit_snapshot,
            regime_module=self._regime_module,
            checkpoint_id=f"{session_id}:{wave_id}",
        ).applied_operations
        if policy_rollback_applied:
            applied_operations = applied_operations + ("policy-rollback",)
        self._previous_total_reward = total_reward
        return JointCycleReport(
            cycle_index=cycle_index,
            acceptance_passed="reflection" in active_snapshots and bool(recorder.events),
            total_reward=total_reward,
            task_reward=dual_track_rollout.task_rollout.total_reward,
            relationship_reward=dual_track_rollout.relationship_rollout.total_reward,
            policy_rollback_applied=policy_rollback_applied,
            optimization_summary=optimization_report.description,
            policy_objective=(
                optimization_report.task_report.surrogate_objective
                + optimization_report.relationship_report.surrogate_objective
            ),
            applied_operations=applied_operations,
            cms_description=self._memory_store.learned_core.snapshot().description
            if self._memory_store.learned_core is not None
            else "No CMS core attached.",
            description=(
                f"Joint ETA/NL cycle {cycle_index} ran dual-track rollout "
                f"task={dual_track_rollout.task_rollout.total_reward:.2f}, "
                f"relationship={dual_track_rollout.relationship_rollout.total_reward:.2f}, "
                f"rollback={'on' if policy_rollback_applied else 'off'}, "
                f"with {len(applied_operations)} bounded writeback operations."
            ),
        )

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

    def _should_rollback(
        self,
        *,
        total_reward: float,
        evaluation_snapshot: EvaluationSnapshot,
        optimization_report: DualTrackOptimizationReport,
    ) -> bool:
        if any(alert.startswith("HIGH") or alert.startswith("CRITICAL") for alert in evaluation_snapshot.alerts):
            return True
        if (
            optimization_report.task_report.surrogate_objective < -0.1
            or optimization_report.relationship_report.surrogate_objective < -0.1
        ):
            return True
        if (
            optimization_report.task_report.kl_penalty > 0.4
            or optimization_report.relationship_report.kl_penalty > 0.4
        ):
            return True
        if self._previous_total_reward is None:
            return False
        return total_reward + 0.25 < self._previous_total_reward

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
            unavailable_fields=(),
            description=f"Trace step {step.step} for {trace.trace_id}.",
        )

