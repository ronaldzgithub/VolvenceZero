from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from collections.abc import Callable
from typing import Literal

from volvence_zero.agent.response import ResponseSynthesizer
from volvence_zero.agent.session import AgentSessionRunner, AgentTurnResult
from volvence_zero.application.domain_experience import DomainExperiencePackage
from volvence_zero.integration import FinalRolloutConfig
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
    ) -> tuple[str, ...]:
        return self.submit_semantic_events(
            semantic_events_from_tool_result(
                event_id=event_id,
                tool_name=tool_name,
                action_id=action_id,
                status=status,
                summary=summary,
                detail=detail,
                confidence=confidence,
                artifact_refs=artifact_refs,
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

    async def run_turn_async(self, user_input: str) -> AgentTurnResult:
        return await self._runner.run_turn(user_input)

    def run_turn(self, user_input: str) -> AgentTurnResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_turn_async(user_input))
        raise RuntimeError("BrainSession.run_turn() cannot be used inside a running event loop; use run_turn_async().")


class Brain:
    """Package-first facade for creating brain sessions.

    This class is the stable API boundary for product and service adapters.
    It accepts domain experience packages and explicit substrate choices, then
    delegates runtime execution to ``AgentSessionRunner``.
    """

    def __init__(
        self,
        config: BrainConfig | None = None,
        *,
        substrate_runtime: OpenWeightResidualRuntime | None = None,
        substrate_adapter_factory: Callable[[str, int], SubstrateAdapter] | None = None,
        response_synthesizer: ResponseSynthesizer | None = None,
        semantic_proposal_runtime: SemanticProposalRuntime | None = None,
    ) -> None:
        self._config = config or BrainConfig()
        self._injected_runtime = substrate_runtime
        self._substrate_adapter_factory = substrate_adapter_factory
        self._response_synthesizer = response_synthesizer
        self._semantic_proposal_runtime = semantic_proposal_runtime

    @property
    def config(self) -> BrainConfig:
        return self._config

    def with_domain_experience(
        self,
        packages: tuple[DomainExperiencePackage, ...],
    ) -> Brain:
        return Brain(
            replace(
                self._config,
                domain_experience_packages=self._config.domain_experience_packages + packages,
            ),
            substrate_runtime=self._injected_runtime,
            substrate_adapter_factory=self._substrate_adapter_factory,
            response_synthesizer=self._response_synthesizer,
            semantic_proposal_runtime=self._semantic_proposal_runtime,
        )

    def create_session(self, *, session_id: str = "brain-session") -> BrainSession:
        runtime = self._resolve_substrate_runtime()
        runner = AgentSessionRunner(
            session_id=session_id,
            config=self._config.final_rollout_config or FinalRolloutConfig(),
            application_persistence_dir=self._config.application_persistence_dir,
            domain_experience_packages=self._config.domain_experience_packages,
            default_residual_runtime=runtime,
            substrate_adapter_factory=self._substrate_adapter_factory,
            response_synthesizer=self._response_synthesizer,
            semantic_proposal_runtime=self._semantic_proposal_runtime,
            rare_heavy_enabled=self._config.rare_heavy_enabled,
        )
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
