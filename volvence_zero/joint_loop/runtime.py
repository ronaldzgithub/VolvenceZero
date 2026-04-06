from __future__ import annotations

from dataclasses import dataclass

from volvence_zero.integration import FinalRolloutConfig, run_final_wiring_turn
from volvence_zero.internal_rl import InternalRLSandbox
from volvence_zero.memory import CMSMemoryCore, MemoryStore
from volvence_zero.reflection import WritebackMode
from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import (
    SimulatedResidualSubstrateAdapter,
    SubstrateSnapshot,
    SurfaceKind,
    TraceStep,
    TrainingTrace,
)
from volvence_zero.temporal import LearnedLiteTemporalPolicy


@dataclass(frozen=True)
class JointCycleReport:
    cycle_index: int
    acceptance_passed: bool
    total_reward: float
    applied_operations: tuple[str, ...]
    cms_description: str
    description: str


class ETANLJointLoop:
    """Minimal SSL-RL alternation loop over the stage-two building blocks."""

    def __init__(self) -> None:
        self._policy = LearnedLiteTemporalPolicy()
        self._sandbox = InternalRLSandbox(policy=self._policy)
        self._memory_store = MemoryStore(learned_core=CMSMemoryCore())

    @property
    def memory_store(self) -> MemoryStore:
        return self._memory_store

    @property
    def temporal_policy(self) -> LearnedLiteTemporalPolicy:
        return self._policy

    async def run_cycle(self, *, cycle_index: int, trace: TrainingTrace) -> JointCycleReport:
        substrate_snapshots = tuple(self._snapshot_from_trace_step(step, trace) for step in trace.steps)
        rollout = self._sandbox.rollout(
            rollout_id=f"joint-{cycle_index}",
            substrate_steps=substrate_snapshots,
        )
        self._sandbox.optimize(rollout)
        integration_result = await run_final_wiring_turn(
            config=FinalRolloutConfig(
                substrate=WiringLevel.ACTIVE,
                memory=WiringLevel.ACTIVE,
                dual_track=WiringLevel.ACTIVE,
                evaluation=WiringLevel.ACTIVE,
                regime=WiringLevel.ACTIVE,
                credit=WiringLevel.ACTIVE,
                reflection=WiringLevel.ACTIVE,
                temporal=WiringLevel.ACTIVE,
            ),
            substrate_adapter=SimulatedResidualSubstrateAdapter(trace=trace),
            memory_store=self._memory_store,
            reflection_mode=WritebackMode.APPLY,
            temporal_policy=self._policy,
            session_id=f"joint-session-{cycle_index}",
            wave_id=f"joint-wave-{cycle_index}",
        )
        applied_operations = (
            integration_result.writeback_result.applied_operations
            if integration_result.writeback_result is not None
            else ()
        )
        return JointCycleReport(
            cycle_index=cycle_index,
            acceptance_passed=integration_result.acceptance_report.passed,
            total_reward=rollout.total_reward,
            applied_operations=applied_operations,
            cms_description=self._memory_store.learned_core.snapshot().description
            if self._memory_store.learned_core is not None
            else "No CMS core attached.",
            description=(
                f"Joint ETA/NL cycle {cycle_index} ran rollout reward {rollout.total_reward:.2f} "
                f"with {len(applied_operations)} bounded writeback operations."
            ),
        )

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

