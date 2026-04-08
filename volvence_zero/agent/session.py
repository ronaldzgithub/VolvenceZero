from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from volvence_zero.agent.response import AgentResponse, ResponseContext, ResponseSynthesizer
from volvence_zero.credit import ModificationProposal
from volvence_zero.evaluation import EvaluationSnapshot
from volvence_zero.integration import (
    FinalIntegrationResult,
    FinalRolloutConfig,
    run_final_wiring_turn,
)
from volvence_zero.joint_loop import ETANLJointLoop, JointCycleReport, JointLoopSchedule, ScheduledJointLoopResult
from volvence_zero.memory import MemorySnapshot, MemoryStore
from volvence_zero.reflection import ReflectionSnapshot, WritebackMode
from volvence_zero.regime import RegimeSnapshot
from volvence_zero.runtime import Snapshot, WiringLevel
from volvence_zero.substrate import (
    build_transformers_runtime_with_fallback,
    OpenWeightResidualStreamSubstrateAdapter,
    OpenWeightResidualRuntime,
    SubstrateFallbackMode,
    SubstrateAdapter,
    build_training_trace,
)
from volvence_zero.temporal import (
    FullLearnedTemporalPolicy,
    MetacontrollerRuntimeState,
    TemporalAbstractionSnapshot,
    TemporalPolicy,
)


@dataclass(frozen=True)
class AgentTurnResult:
    session_id: str
    wave_id: str
    user_input: str
    active_snapshots: dict[str, Snapshot[Any]]
    shadow_snapshots: dict[str, Snapshot[Any]]
    acceptance_passed: bool
    acceptance_issues: tuple[str, ...]
    active_regime: str | None
    active_abstract_action: str | None
    metacontroller_state: MetacontrollerRuntimeState | None
    evaluation_alerts: tuple[str, ...]
    bounded_writeback_applied: bool
    writeback_source: str | None
    writeback_operations: tuple[str, ...]
    writeback_blocks: tuple[str, ...]
    joint_schedule_action: str
    joint_learning_summary: str
    joint_cycle_report: JointCycleReport | None
    response: AgentResponse
    event_count: int


class AgentSessionRunner:
    """Minimal session runner over the final wiring graph."""

    def __init__(
        self,
        *,
        session_id: str = "agent-session",
        config: FinalRolloutConfig | None = None,
        memory_store: MemoryStore | None = None,
        reflection_mode: WritebackMode = WritebackMode.APPLY,
        temporal_policy: TemporalPolicy | None = None,
        credit_proposals: tuple[ModificationProposal, ...] = (),
        response_synthesizer: ResponseSynthesizer | None = None,
        substrate_adapter_factory: Callable[[str, int], SubstrateAdapter] | None = None,
        default_residual_runtime: OpenWeightResidualRuntime | None = None,
        substrate_model_id: str = "distilgpt2",
        substrate_device: str = "auto",
        substrate_local_files_only: bool = False,
        substrate_fallback_to_builtin: bool | None = None,
        substrate_fallback_mode: SubstrateFallbackMode | str | None = None,
        joint_loop: ETANLJointLoop | None = None,
        joint_schedule: JointLoopSchedule | None = None,
    ) -> None:
        self._session_id = session_id
        self._config = config or FinalRolloutConfig()
        self._memory_store = memory_store or MemoryStore()
        self._reflection_mode = reflection_mode
        self._temporal_policy = temporal_policy or FullLearnedTemporalPolicy()
        self._credit_proposals = credit_proposals
        self._response_synthesizer = response_synthesizer or ResponseSynthesizer()
        self._substrate_adapter_factory = substrate_adapter_factory
        self._joint_loop = joint_loop or ETANLJointLoop(policy=self._temporal_policy)
        self._joint_schedule = joint_schedule or JointLoopSchedule()
        self._default_residual_runtime = default_residual_runtime or build_transformers_runtime_with_fallback(
            model_id=substrate_model_id,
            device=substrate_device,
            local_files_only=substrate_local_files_only,
            fallback_to_builtin=substrate_fallback_to_builtin,
            fallback_mode=substrate_fallback_mode,
            builtin_model_id="runner-transformers-runtime",
        )
        self._turn_index = 0
        self._upstream_snapshots: dict[str, Snapshot[Any]] = {}

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def turn_index(self) -> int:
        return self._turn_index

    async def run_turn(self, user_input: str) -> AgentTurnResult:
        self._turn_index += 1
        wave_id = f"wave-{self._turn_index}"
        substrate_adapter = self._build_substrate_adapter(user_input=user_input)
        joint_result = await self._joint_loop.run_scheduled_step(
            turn_index=self._turn_index,
            trace=build_training_trace(
                trace_id=f"{self._session_id}:joint:{self._turn_index}",
                source_text=user_input,
            ),
            schedule=self._joint_schedule,
            apply_writeback=self._reflection_mode is not WritebackMode.APPLY,
        )
        integration_result = await run_final_wiring_turn(
            config=self._config,
            substrate_adapter=substrate_adapter,
            memory_store=self._memory_store,
            upstream_snapshots=self._upstream_snapshots,
            joint_loop_result=joint_result,
            credit_proposals=self._credit_proposals,
            reflection_mode=self._reflection_mode,
            temporal_policy=self._temporal_policy,
            session_id=self._session_id,
            wave_id=wave_id,
        )
        self._upstream_snapshots = {
            **integration_result.active_snapshots,
            **integration_result.shadow_snapshots,
        }
        return self._to_turn_result(
            user_input=user_input,
            wave_id=wave_id,
            integration_result=integration_result,
            joint_result=joint_result,
        )

    def _build_substrate_adapter(self, *, user_input: str) -> SubstrateAdapter:
        if self._substrate_adapter_factory is not None:
            return self._substrate_adapter_factory(user_input, self._turn_index)
        return OpenWeightResidualStreamSubstrateAdapter(
            runtime=self._default_residual_runtime,
            default_source_text=user_input,
        )

    def _to_turn_result(
        self,
        *,
        user_input: str,
        wave_id: str,
        integration_result: FinalIntegrationResult,
        joint_result: ScheduledJointLoopResult,
    ) -> AgentTurnResult:
        active_regime = None
        regime_snapshot = integration_result.active_snapshots.get("regime") or integration_result.shadow_snapshots.get(
            "regime"
        )
        if regime_snapshot is not None and isinstance(regime_snapshot.value, RegimeSnapshot):
            active_regime = regime_snapshot.value.active_regime.regime_id
        regime_switched = bool(
            regime_snapshot is not None
            and isinstance(regime_snapshot.value, RegimeSnapshot)
            and regime_snapshot.value.previous_regime is not None
            and regime_snapshot.value.previous_regime.regime_id != active_regime
        )

        active_abstract_action = None
        temporal_switch_gate = 0.0
        temporal_is_switching = False
        metacontroller_state = integration_result.temporal_runtime_state
        temporal_snapshot = integration_result.active_snapshots.get(
            "temporal_abstraction"
        ) or integration_result.shadow_snapshots.get("temporal_abstraction")
        if temporal_snapshot is not None and isinstance(
            temporal_snapshot.value, TemporalAbstractionSnapshot
        ):
            active_abstract_action = temporal_snapshot.value.active_abstract_action
            temporal_switch_gate = temporal_snapshot.value.controller_state.switch_gate
            temporal_is_switching = temporal_snapshot.value.controller_state.is_switching

        evaluation_alerts: tuple[str, ...] = ()
        evaluation_snapshot = integration_result.active_snapshots.get("evaluation")
        if evaluation_snapshot is not None and isinstance(evaluation_snapshot.value, EvaluationSnapshot):
            evaluation_alerts = evaluation_snapshot.value.alerts

        memory_retrieval_count = 0
        memory_snapshot = integration_result.active_snapshots.get("memory")
        if memory_snapshot is not None and isinstance(memory_snapshot.value, MemorySnapshot):
            memory_retrieval_count = len(memory_snapshot.value.retrieved_entries)

        reflection_lesson_count = 0
        reflection_tension_count = 0
        primary_reflection_lesson = None
        primary_reflection_tension = None
        reflection_snapshot = integration_result.active_snapshots.get("reflection") or integration_result.shadow_snapshots.get(
            "reflection"
        )
        if reflection_snapshot is not None and isinstance(reflection_snapshot.value, ReflectionSnapshot):
            reflection_lesson_count = len(reflection_snapshot.value.lessons_extracted)
            reflection_tension_count = len(reflection_snapshot.value.tensions_identified)
            primary_reflection_lesson = next(iter(reflection_snapshot.value.lessons_extracted), None)
            primary_reflection_tension = next(iter(reflection_snapshot.value.tensions_identified), None)

        response = self._response_synthesizer.synthesize(
            context=ResponseContext(
                regime_id=active_regime,
                regime_name=regime_snapshot.value.active_regime.name
                if regime_snapshot is not None and isinstance(regime_snapshot.value, RegimeSnapshot)
                else "current context",
                regime_switched=regime_switched,
                abstract_action=active_abstract_action,
                alert_count=len(evaluation_alerts),
                retrieved_memory_count=memory_retrieval_count,
                temporal_switch_gate=temporal_switch_gate,
                temporal_is_switching=temporal_is_switching,
                reflection_lesson_count=reflection_lesson_count,
                reflection_tension_count=reflection_tension_count,
                reflection_writeback_applied=bool(
                    integration_result.writeback_result is not None
                    and integration_result.writeback_result.applied_operations
                ),
                primary_reflection_lesson=primary_reflection_lesson,
                primary_reflection_tension=primary_reflection_tension,
                joint_schedule_action=joint_result.schedule_action,
            )
        )

        return AgentTurnResult(
            session_id=self._session_id,
            wave_id=wave_id,
            user_input=user_input,
            active_snapshots=integration_result.active_snapshots,
            shadow_snapshots=integration_result.shadow_snapshots,
            acceptance_passed=integration_result.acceptance_report.passed,
            acceptance_issues=integration_result.acceptance_report.issues,
            active_regime=active_regime,
            active_abstract_action=active_abstract_action,
            metacontroller_state=metacontroller_state,
            evaluation_alerts=evaluation_alerts,
            bounded_writeback_applied=bool(
                integration_result.writeback_result is not None
                and integration_result.writeback_result.applied_operations
            ),
            writeback_source=integration_result.writeback_source,
            writeback_operations=integration_result.writeback_result.applied_operations
            if integration_result.writeback_result is not None
            else (),
            writeback_blocks=integration_result.writeback_result.blocked_operations
            if integration_result.writeback_result is not None
            else (),
            joint_schedule_action=joint_result.schedule_action,
            joint_learning_summary=joint_result.description,
            joint_cycle_report=joint_result.cycle_report,
            response=response,
            event_count=integration_result.event_count,
        )


def default_active_runner() -> AgentSessionRunner:
    return AgentSessionRunner(
        config=FinalRolloutConfig(
            substrate=WiringLevel.ACTIVE,
            memory=WiringLevel.ACTIVE,
            dual_track=WiringLevel.ACTIVE,
            evaluation=WiringLevel.ACTIVE,
            regime=WiringLevel.ACTIVE,
            credit=WiringLevel.ACTIVE,
            reflection=WiringLevel.ACTIVE,
            temporal=WiringLevel.ACTIVE,
        )
    )
