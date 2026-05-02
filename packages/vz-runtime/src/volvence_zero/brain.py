from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from collections.abc import Callable
from typing import Literal

from volvence_zero.agent.response import ResponseSynthesizer
from volvence_zero.agent.session import AgentSessionRunner, AgentTurnResult
from volvence_zero.application.domain_experience import DomainExperiencePackage
from volvence_zero.application.storage import (
    ProvisionalReconcileResult,
    ProvisionalReconcileThresholds,
)
from volvence_zero.agent.dialogue_outcome_producers import (
    tool_outcome_evidence_from_environment_outcome,
)
from volvence_zero.dialogue_trace import (
    DialogueOutcomeEvidence,
    DialogueOutcomeResolution,
)
from volvence_zero.environment import (
    EnvironmentEvent,
    EnvironmentEventKind,
    EnvironmentOutcome,
)
from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.memory import MemoryStore
from volvence_zero.semantic_state import (
    ExternalSemanticEventBatch,
    SemanticProposalRuntime,
    semantic_events_from_profile,
    semantic_events_from_reviewed_knowledge,
    semantic_events_from_task_event,
    semantic_events_from_tool_result,
)
from volvence_zero.substrate import (
    OpenWeightResidualRuntime,
    SubstrateAdapter,
    SyntheticOpenWeightResidualRuntime,
    build_transformers_runtime_with_fallback,
)
from volvence_zero.regime import RegimeBootstrap
from volvence_zero.temporal import (
    FullLearnedTemporalPolicy,
    MetacontrollerParameterSnapshot,
)


SubstrateMode = Literal["synthetic", "hf", "injected"]


@dataclass(frozen=True)
class BrainConfig:
    """Stable package-facing configuration for the Volvence Zero brain.

    The default synthetic substrate keeps the core package independent from
    external model weights. Hugging Face / Qwen runtimes are only constructed
    when ``substrate_mode="hf"`` is selected explicitly.
    """

    substrate_mode: SubstrateMode = "synthetic"
    substrate_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    substrate_model_source: str | None = None
    substrate_device: str = "auto"
    substrate_local_files_only: bool = True
    substrate_fallback_mode: str | None = "deny"
    application_persistence_dir: str | None = None
    domain_experience_packages: tuple[DomainExperiencePackage, ...] = ()
    final_rollout_config: FinalRolloutConfig | None = None
    rare_heavy_enabled: bool = True


class BrainSession:
    """Synchronous and async facade over ``AgentSessionRunner``."""

    def __init__(self, *, runner: AgentSessionRunner) -> None:
        self._runner = runner

    @property
    def runner(self) -> AgentSessionRunner:
        return self._runner

    @property
    def session_id(self) -> str:
        return self._runner.session_id

    def submit_semantic_events(self, events: ExternalSemanticEventBatch) -> tuple[str, ...]:
        return self._runner.enqueue_semantic_events(events)

    def submit_tool_result(
        self,
        *,
        event_id: str,
        tool_name: str,
        action_id: str,
        status: str,
        summary: str,
        detail: str,
        confidence: float = 0.8,
        artifact_refs: tuple[str, ...] = (),
        plan_ref: str | None = None,
        latency_ms: int | None = None,
        monetary_cost: float = 0.0,
        reversibility: str = "reversible",
        environment_state_delta_kind: str = "none",
    ) -> tuple[str, ...]:
        outcome = EnvironmentOutcome(
            outcome_id=f"{event_id}:outcome",
            event_id=event_id,
            outcome_kind=EnvironmentEventKind.TOOL_RESULT,
            action_id=action_id,
            status=status,
            summary=summary,
            detail=detail,
            confidence=confidence,
            prediction_id=plan_ref,
            evidence=(f"tool:{tool_name}",),
            latency_ms=latency_ms,
            monetary_cost=monetary_cost,
            reversibility=reversibility,
            environment_state_delta_kind=environment_state_delta_kind,
        )
        tool_evidence = tool_outcome_evidence_from_environment_outcome(
            environment_outcome=outcome,
            tool_name=tool_name,
        )
        if tool_evidence:
            self._runner.attach_dialogue_outcome_evidence(tool_evidence)
        return self.submit_semantic_events(
            semantic_events_from_tool_result(
                event_id=event_id,
                tool_name=tool_name,
                action_id=action_id,
                status=status,
                summary=summary,
                detail=detail,
                confidence=confidence,
                artifact_refs=(
                    *artifact_refs,
                    f"environment_outcome:{outcome.outcome_id}",
                ),
                plan_ref=plan_ref,
            )
        )

    def submit_profile_event(
        self,
        *,
        event_id: str,
        source: str,
        preferences: tuple[str, ...] = (),
        goals: tuple[str, ...] = (),
        consent_grants: tuple[str, ...] = (),
        consent_denials: tuple[str, ...] = (),
        relationship_note: str = "",
        confidence: float = 0.75,
    ) -> tuple[str, ...]:
        return self.submit_semantic_events(
            semantic_events_from_profile(
                event_id=event_id,
                source=source,
                preferences=preferences,
                goals=goals,
                consent_grants=consent_grants,
                consent_denials=consent_denials,
                relationship_note=relationship_note,
                confidence=confidence,
            )
        )

    def submit_task_event(
        self,
        *,
        event_id: str,
        task_id: str,
        status: str,
        summary: str,
        detail: str,
        due_hint: str | None = None,
        commitment_ref: str | None = None,
        confidence: float = 0.75,
    ) -> tuple[str, ...]:
        return self.submit_semantic_events(
            semantic_events_from_task_event(
                event_id=event_id,
                task_id=task_id,
                status=status,
                summary=summary,
                detail=detail,
                due_hint=due_hint,
                commitment_ref=commitment_ref,
                confidence=confidence,
            )
        )

    def submit_reviewed_knowledge_event(
        self,
        *,
        event_id: str,
        knowledge_id: str,
        summary: str,
        detail: str,
        source_label: str,
        confidence: float,
        relevance_hint: str = "",
        needs_followup: bool = False,
    ) -> tuple[str, ...]:
        return self.submit_semantic_events(
            semantic_events_from_reviewed_knowledge(
                event_id=event_id,
                knowledge_id=knowledge_id,
                summary=summary,
                detail=detail,
                source_label=source_label,
                confidence=confidence,
                relevance_hint=relevance_hint,
                needs_followup=needs_followup,
            )
        )

    async def run_turn_async(
        self,
        user_input: str,
        *,
        environment_event: EnvironmentEvent | None = None,
    ) -> AgentTurnResult:
        return await self._runner.run_turn(
            user_input,
            environment_event=environment_event,
        )

    def run_turn(
        self,
        user_input: str,
        *,
        environment_event: EnvironmentEvent | None = None,
    ) -> AgentTurnResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.run_turn_async(
                    user_input,
                    environment_event=environment_event,
                )
            )
        raise RuntimeError("BrainSession.run_turn() cannot be used inside a running event loop; use run_turn_async().")

    def submit_dialogue_outcome_evidence(
        self,
        evidence: tuple[DialogueOutcomeEvidence, ...],
    ) -> DialogueOutcomeResolution | None:
        """Attach typed dialogue outcome evidence to the most recent trace.

        Used by lifeform-side adapters (for example scene close) that
        observe a structural outcome after the turn finished. Evidence
        must be machine-readable; raw text is never inspected.
        """

        return self._runner.attach_dialogue_outcome_evidence(evidence)

    def reconcile_case_memory_provisional(
        self,
        *,
        now_tick: int,
        thresholds: ProvisionalReconcileThresholds | None = None,
    ) -> ProvisionalReconcileResult:
        """Scene-boundary provisional case sweep (Gap 4 slice 2a).

        Thin pass-through to ``AgentSessionRunner.reconcile_case_memory_provisional``.
        Lifeform-layer callers (``LifeformSession.end_scene``) use this to
        drive the case memory lifecycle at scene close. Returns the full
        result so observers can surface the decision list.
        """
        return self._runner.reconcile_case_memory_provisional(
            now_tick=now_tick,
            thresholds=thresholds,
        )


class Brain:
    """Package-first facade for creating brain sessions.

    This class is the stable API boundary for product and service adapters.
    It accepts domain experience packages and explicit substrate choices, then
    delegates runtime execution to ``AgentSessionRunner``.

    A pre-trained metacontroller can be injected via ``temporal_bootstrap``
    (a ``MetacontrollerParameterSnapshot`` exported from a SSL training run).
    Each ``create_session`` call rebuilds the policy from the snapshot so
    sessions never share mutable controller state across tenants.
    """

    def __init__(
        self,
        config: BrainConfig | None = None,
        *,
        substrate_runtime: OpenWeightResidualRuntime | None = None,
        substrate_adapter_factory: Callable[[str, int], SubstrateAdapter] | None = None,
        response_synthesizer: ResponseSynthesizer | None = None,
        semantic_proposal_runtime: SemanticProposalRuntime | None = None,
        temporal_bootstrap: MetacontrollerParameterSnapshot | None = None,
        regime_bootstrap: RegimeBootstrap | None = None,
        memory_store: MemoryStore | None = None,
    ) -> None:
        self._config = config or BrainConfig()
        self._injected_runtime = substrate_runtime
        self._substrate_adapter_factory = substrate_adapter_factory
        self._response_synthesizer = response_synthesizer
        self._semantic_proposal_runtime = semantic_proposal_runtime
        self._temporal_bootstrap = temporal_bootstrap
        self._regime_bootstrap = regime_bootstrap
        self._memory_store = memory_store

    @property
    def config(self) -> BrainConfig:
        return self._config

    @property
    def temporal_bootstrap(self) -> MetacontrollerParameterSnapshot | None:
        """The trained metacontroller snapshot, if one was injected."""
        return self._temporal_bootstrap

    @property
    def regime_bootstrap(self) -> RegimeBootstrap | None:
        """The calibrated regime selection-weights bootstrap, if injected."""
        return self._regime_bootstrap

    def _clone_kwargs(self) -> dict[str, object]:
        return {
            "substrate_runtime": self._injected_runtime,
            "substrate_adapter_factory": self._substrate_adapter_factory,
            "response_synthesizer": self._response_synthesizer,
            "semantic_proposal_runtime": self._semantic_proposal_runtime,
            "temporal_bootstrap": self._temporal_bootstrap,
            "regime_bootstrap": self._regime_bootstrap,
            "memory_store": self._memory_store,
        }

    def with_domain_experience(
        self,
        packages: tuple[DomainExperiencePackage, ...],
    ) -> Brain:
        return Brain(
            replace(
                self._config,
                domain_experience_packages=self._config.domain_experience_packages + packages,
            ),
            **self._clone_kwargs(),
        )

    def with_temporal_bootstrap(
        self,
        snapshot: MetacontrollerParameterSnapshot | None,
    ) -> Brain:
        """Return a clone of this Brain with the given trained metacontroller.

        Pass ``None`` to drop the bootstrap and fall back to a fresh policy.
        """
        kwargs = self._clone_kwargs()
        kwargs["temporal_bootstrap"] = snapshot
        return Brain(self._config, **kwargs)

    def with_regime_bootstrap(
        self,
        bootstrap: RegimeBootstrap | None,
    ) -> Brain:
        """Return a clone of this Brain with calibrated regime weights.

        Pass ``None`` to drop the bootstrap and fall back to flat weights.
        """
        kwargs = self._clone_kwargs()
        kwargs["regime_bootstrap"] = bootstrap
        return Brain(self._config, **kwargs)

    def create_session(
        self,
        *,
        session_id: str = "brain-session",
        response_synthesizer: ResponseSynthesizer | None = None,
    ) -> BrainSession:
        """Create a new BrainSession.

        Args:
            session_id: Stable identifier for the session.
            response_synthesizer: Optional per-session synthesizer override.
                When supplied, this synthesizer is used in place of the
                Brain-level default for THIS session only \u2014 the Brain's own
                synthesizer is unchanged. Used by the lifeform layer to
                clone a synthesizer per ``LifeformSession`` so that
                per-session state (e.g. a ``VitalsModule`` reference) can
                be bound by closure without sharing mutable state across
                sessions of the same Brain.
        """
        runtime = self._resolve_substrate_runtime()
        synthesizer = response_synthesizer or self._response_synthesizer
        runner_kwargs: dict[str, object] = dict(
            session_id=session_id,
            config=self._config.final_rollout_config or FinalRolloutConfig(),
            application_persistence_dir=self._config.application_persistence_dir,
            domain_experience_packages=self._config.domain_experience_packages,
            default_residual_runtime=runtime,
            substrate_adapter_factory=self._substrate_adapter_factory,
            response_synthesizer=synthesizer,
            semantic_proposal_runtime=self._semantic_proposal_runtime,
            rare_heavy_enabled=self._config.rare_heavy_enabled,
            regime_bootstrap=self._regime_bootstrap,
            memory_store=self._memory_store,
        )
        if self._temporal_bootstrap is not None:
            # Build fresh policies per session from the trained snapshot so
            # sessions never share mutable controller state. Both world and
            # self tracks are seeded from the same bootstrap; the runner
            # will clone the world track for the self track when needed.
            runner_kwargs["world_temporal_policy"] = (
                FullLearnedTemporalPolicy.from_bootstrap_snapshot(self._temporal_bootstrap)
            )
        runner = AgentSessionRunner(**runner_kwargs)
        return BrainSession(runner=runner)

    def _resolve_substrate_runtime(self) -> OpenWeightResidualRuntime:
        if self._config.substrate_mode == "injected":
            if self._injected_runtime is None:
                raise ValueError("BrainConfig(substrate_mode='injected') requires substrate_runtime.")
            return self._injected_runtime
        if self._config.substrate_mode == "synthetic":
            return self._injected_runtime or SyntheticOpenWeightResidualRuntime(model_id="volvence-zero-core-synthetic")
        if self._config.substrate_mode == "hf":
            try:
                return build_transformers_runtime_with_fallback(
                    model_id=self._config.substrate_model_id,
                    model_source=self._config.substrate_model_source,
                    device=self._config.substrate_device,
                    local_files_only=self._config.substrate_local_files_only,
                    fallback_mode=self._config.substrate_fallback_mode,
                )
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "BrainConfig(substrate_mode='hf') requires optional dependencies; "
                    "install with volvence-zero[hf]."
                ) from exc
        raise ValueError(f"Unsupported substrate_mode: {self._config.substrate_mode}")
