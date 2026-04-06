from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from volvence_zero.agent.response import AgentResponse, ResponseSynthesizer
from volvence_zero.credit import ModificationProposal
from volvence_zero.evaluation import EvaluationSnapshot
from volvence_zero.integration import (
    FinalIntegrationResult,
    FinalRolloutConfig,
    run_final_wiring_turn,
)
from volvence_zero.memory import MemoryStore
from volvence_zero.reflection import WritebackMode
from volvence_zero.regime import RegimeSnapshot
from volvence_zero.runtime import Snapshot, WiringLevel
from volvence_zero.substrate import (
    SimulatedResidualSubstrateAdapter,
    SubstrateAdapter,
    build_training_trace,
)
from volvence_zero.temporal import (
    LearnedLiteTemporalPolicy,
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
        reflection_mode: WritebackMode = WritebackMode.PROPOSAL_ONLY,
        temporal_policy: TemporalPolicy | None = None,
        credit_proposals: tuple[ModificationProposal, ...] = (),
        response_synthesizer: ResponseSynthesizer | None = None,
    ) -> None:
        self._session_id = session_id
        self._config = config or FinalRolloutConfig()
        self._memory_store = memory_store or MemoryStore()
        self._reflection_mode = reflection_mode
        self._temporal_policy = temporal_policy or LearnedLiteTemporalPolicy()
        self._credit_proposals = credit_proposals
        self._response_synthesizer = response_synthesizer or ResponseSynthesizer()
        self._turn_index = 0

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
        integration_result = await run_final_wiring_turn(
            config=self._config,
            substrate_adapter=substrate_adapter,
            memory_store=self._memory_store,
            credit_proposals=self._credit_proposals,
            reflection_mode=self._reflection_mode,
            temporal_policy=self._temporal_policy,
            session_id=self._session_id,
            wave_id=wave_id,
        )
        return self._to_turn_result(
            user_input=user_input,
            wave_id=wave_id,
            integration_result=integration_result,
        )

    def _build_substrate_adapter(self, *, user_input: str) -> SubstrateAdapter:
        trace = build_training_trace(
            trace_id=f"{self._session_id}:{self._turn_index}",
            source_text=user_input,
        )
        return SimulatedResidualSubstrateAdapter(trace=trace, model_id="runner-residual-sim")

    def _to_turn_result(
        self,
        *,
        user_input: str,
        wave_id: str,
        integration_result: FinalIntegrationResult,
    ) -> AgentTurnResult:
        active_regime = None
        regime_snapshot = integration_result.active_snapshots.get("regime") or integration_result.shadow_snapshots.get(
            "regime"
        )
        if regime_snapshot is not None and isinstance(regime_snapshot.value, RegimeSnapshot):
            active_regime = regime_snapshot.value.active_regime.regime_id

        active_abstract_action = None
        metacontroller_state = integration_result.temporal_runtime_state
        temporal_snapshot = integration_result.active_snapshots.get(
            "temporal_abstraction"
        ) or integration_result.shadow_snapshots.get("temporal_abstraction")
        if temporal_snapshot is not None and isinstance(
            temporal_snapshot.value, TemporalAbstractionSnapshot
        ):
            active_abstract_action = temporal_snapshot.value.active_abstract_action

        evaluation_alerts: tuple[str, ...] = ()
        evaluation_snapshot = integration_result.active_snapshots.get("evaluation")
        if evaluation_snapshot is not None and isinstance(evaluation_snapshot.value, EvaluationSnapshot):
            evaluation_alerts = evaluation_snapshot.value.alerts

        response = self._response_synthesizer.synthesize(
            user_input=user_input,
            active_snapshots=integration_result.active_snapshots,
            shadow_snapshots=integration_result.shadow_snapshots,
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
            reflection=WiringLevel.SHADOW,
            temporal=WiringLevel.SHADOW,
        )
    )
