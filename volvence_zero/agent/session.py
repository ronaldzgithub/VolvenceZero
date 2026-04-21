from __future__ import annotations

from collections.abc import Callable
import asyncio
from dataclasses import dataclass, replace
from typing import Any

from volvence_zero.agent.response import AgentResponse, LLMResponseSynthesizer, ResponseContext, ResponseSynthesizer
from volvence_zero.credit import ModificationProposal
from volvence_zero.evaluation import EvaluationBackbone, EvaluationReport, EvaluationSnapshot, EvolutionJudgement
from volvence_zero.integration import (
    FinalIntegrationResult,
    FinalRolloutConfig,
    SessionPostWritebackRequest,
    apply_session_post_writeback_request,
    run_final_wiring_turn,
)
from volvence_zero.joint_loop import (
    ETANLJointLoop,
    JointCycleReport,
    JointLoopSchedule,
    PipelineConfig,
    RareHeavyArtifact,
    RareHeavyImportCheckpoint,
    RareHeavyImportResult,
    SSLRLTrainingPipeline,
    ScheduledJointLoopResult,
)
from volvence_zero.memory import MemorySnapshot, MemoryStore, build_default_memory_store
from volvence_zero.planning import ImaginationResult, imagine
from volvence_zero.prediction import ActualOutcome, PredictedOutcome, PredictionError, PredictionErrorModule
from volvence_zero.reflection import ReflectionSnapshot, WritebackMode, WritebackResult
from volvence_zero.regime import RegimeModule, RegimeSnapshot
from volvence_zero.runtime import Snapshot, WiringLevel
from volvence_zero.substrate import (
    build_transformers_runtime_with_fallback,
    LocalSubstrateRuntimeMode,
    OpenWeightResidualStreamSubstrateAdapter,
    OpenWeightResidualRuntime,
    SubstrateFallbackMode,
    SubstrateAdapter,
    SubstrateSnapshot,
    TrainingTrace,
    TraceStep,
    build_training_trace,
)
from volvence_zero.temporal import (
    FullLearnedTemporalPolicy,
    MetacontrollerParameterStore,
    MetacontrollerRuntimeState,
    TemporalAbstractionSnapshot,
    TemporalPolicy,
)
from volvence_zero.agent.session_post_slow_loop import (
    SessionPostSlowLoopJob,
    SessionPostSlowLoopQueue,
    SessionPostSlowLoopQueueState,
    SessionPostSlowLoopResult,
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
    evaluated_prediction: PredictedOutcome | None
    actual_outcome: ActualOutcome | None
    next_prediction: PredictedOutcome | None
    prediction_error: PredictionError | None
    bounded_writeback_applied: bool
    writeback_source: str | None
    writeback_operations: tuple[str, ...]
    writeback_blocks: tuple[str, ...]
    joint_schedule_action: str
    joint_learning_summary: str
    joint_cycle_report: JointCycleReport | None
    response: AgentResponse
    event_count: int
    substrate_model_id: str | None = None
    substrate_runtime_origin: str | None = None
    substrate_fallback_active: bool = False
    substrate_capture_source: str | None = None
    substrate_residual_sequence_length: int = 0
    reflection_promotion_eligible: bool = False
    reflection_promotion_reason: str = ""
    imagination_result: ImaginationResult | None = None
    rare_heavy_result: RareHeavyTurnResult | None = None
    evolution_judgement: EvolutionJudgement | None = None
    cross_session_verdict: str = ""
    nested_profile_active: bool = False
    nested_context_reset_applied: bool = False
    nested_context_reset_total_count: int = 0
    slow_to_fast_init_benefit: float = 0.0
    slow_to_fast_target_distance_before: float = 0.0
    slow_to_fast_target_distance_after: float = 0.0
    slow_to_fast_target_alignment_gain: float = 0.0
    learned_memory_primary: bool = False
    artifact_consolidation_count: int = 0
    learned_recall_count: int = 0
    learned_recall_confidence: float = 0.0
    learned_recall_core_guided: bool = False
    session_post_pending_job_count: int = 0
    session_post_completed_job_count: int = 0
    session_post_last_completed_job_id: str | None = None


@dataclass(frozen=True)
class RareHeavyTurnResult:
    recommended: bool
    applied: bool
    artifact_id: str | None
    applied_operations: tuple[str, ...]
    substrate_status: str
    substrate_training_mode: str
    description: str


@dataclass(frozen=True)
class SubstrateBenchmarkTurn:
    turn_index: int
    substrate_runtime_origin: str | None
    substrate_fallback_active: bool
    substrate_capture_source: str | None
    substrate_residual_sequence_length: int
    active_regime: str | None
    active_abstract_action: str | None
    joint_schedule_action: str
    acceptance_passed: bool
    turn_score_count: int
    evaluation_alert_count: int
    policy_objective: float = 0.0
    action_family_version: int = 0
    metrics: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class SubstrateBenchmarkReport:
    path_label: str
    turns: tuple[SubstrateBenchmarkTurn, ...]
    acceptance_rate: float
    mean_residual_sequence_length: float
    mean_turn_score_count: float
    full_cycle_count: int
    metric_means: tuple[tuple[str, float], ...]
    mean_policy_objective: float
    max_family_version: int
    description: str


@dataclass(frozen=True)
class MultiPathBenchmarkReport:
    path_reports: tuple[SubstrateBenchmarkReport, ...]
    metric_deltas_from_baseline: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    baseline_label: str
    description: str


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
        world_temporal_policy: FullLearnedTemporalPolicy | None = None,
        self_temporal_policy: FullLearnedTemporalPolicy | None = None,
        credit_proposals: tuple[ModificationProposal, ...] = (),
        response_synthesizer: ResponseSynthesizer | None = None,
        substrate_adapter_factory: Callable[[str, int], SubstrateAdapter] | None = None,
        default_residual_runtime: OpenWeightResidualRuntime | None = None,
        substrate_model_id: str = "distilgpt2",
        substrate_model_source: str | None = None,
        substrate_device: str = "auto",
        substrate_local_files_only: bool = False,
        substrate_fallback_to_builtin: bool | None = None,
        substrate_fallback_mode: SubstrateFallbackMode | str | None = None,
        substrate_runtime_mode: LocalSubstrateRuntimeMode | str | None = None,
        joint_loop: ETANLJointLoop | None = None,
        joint_schedule: JointLoopSchedule | None = None,
        rare_heavy_enabled: bool = True,
        rare_heavy_trace_window: int = 5,
        rare_heavy_min_traces: int = 4,
        rare_heavy_cooldown_turns: int = 3,
        rare_heavy_pipeline_config: PipelineConfig | None = None,
        external_prediction_error_drive: bool = True,
    ) -> None:
        self._session_id = session_id
        self._config = config or FinalRolloutConfig()
        self._reflection_mode = reflection_mode
        if world_temporal_policy is not None:
            self._world_temporal_policy = world_temporal_policy
        elif isinstance(temporal_policy, FullLearnedTemporalPolicy):
            self._world_temporal_policy = temporal_policy
        else:
            self._world_temporal_policy = FullLearnedTemporalPolicy()
        if self_temporal_policy is not None:
            self._self_temporal_policy = self_temporal_policy
        else:
            self._self_temporal_policy = FullLearnedTemporalPolicy(
                parameter_store=MetacontrollerParameterStore(
                    n_z=self._world_temporal_policy.parameter_store.n_z
                )
            )
        self._evaluation_backbone = EvaluationBackbone()
        default_latent_dim = self._world_temporal_policy.parameter_store.n_z
        self._memory_store = memory_store or build_default_memory_store(latent_dim=default_latent_dim)
        self._credit_proposals = credit_proposals
        if response_synthesizer is not None:
            self._response_synthesizer = response_synthesizer
        else:
            self._response_synthesizer = ResponseSynthesizer()
        self._substrate_adapter_factory = substrate_adapter_factory
        self._regime_module = RegimeModule(
            wiring_level=self._config.level_for("regime", WiringLevel.ACTIVE),
        )
        self._prediction_module = PredictionErrorModule(
            wiring_level=self._config.level_for("prediction_error", WiringLevel.ACTIVE),
        )
        self._substrate_runtime_mode = (
            LocalSubstrateRuntimeMode(substrate_runtime_mode)
            if substrate_runtime_mode is not None
            else None
        )
        self._default_residual_runtime = default_residual_runtime or build_transformers_runtime_with_fallback(
            model_id=substrate_model_id,
            model_source=substrate_model_source,
            device=substrate_device,
            local_files_only=substrate_local_files_only,
            fallback_to_builtin=substrate_fallback_to_builtin,
            fallback_mode=substrate_fallback_mode,
            runtime_mode=self._substrate_runtime_mode,
            builtin_model_id="runner-transformers-runtime",
        )
        self._joint_loop = joint_loop or ETANLJointLoop(
            world_policy=self._world_temporal_policy,
            self_policy=self._self_temporal_policy,
            memory_store=self._memory_store,
            residual_runtime=self._default_residual_runtime,
            evaluation_backbone=self._evaluation_backbone,
        )
        self._joint_schedule = joint_schedule or JointLoopSchedule()
        self._rare_heavy_enabled = rare_heavy_enabled
        self._rare_heavy_trace_window = max(1, rare_heavy_trace_window)
        self._rare_heavy_min_traces = max(1, min(rare_heavy_min_traces, self._rare_heavy_trace_window))
        self._rare_heavy_cooldown_turns = max(0, rare_heavy_cooldown_turns)
        self._rare_heavy_pipeline_config = rare_heavy_pipeline_config or PipelineConfig(
            ssl_min_steps=2,
            ssl_max_steps=3,
            rl_max_steps=2,
        )
        self._external_prediction_error_drive = external_prediction_error_drive
        self._turn_index = 0
        self._upstream_snapshots: dict[str, Snapshot[Any]] = {}
        self._previous_substrate_snapshot: SubstrateSnapshot | None = None
        self._previous_prediction_reward: float = 0.0
        self._previous_prediction_magnitude: float = 0.0
        self._previous_prediction_error: PredictionError | None = None
        self._recommended_z: tuple[float, ...] | None = None
        self._recent_training_traces: list[TrainingTrace] = []
        self._recent_substrate_batches: list[tuple[SubstrateSnapshot, ...]] = []
        self._last_rare_heavy_turn_index = 0
        self._context_index = 1
        self._completed_session_reports: list[EvaluationReport] = []
        self._session_post_lock = asyncio.Lock()
        self._session_post_queue = SessionPostSlowLoopQueue(worker=self._run_session_post_slow_loop_job)
        self._last_session_post_writeback_request: SessionPostWritebackRequest | None = None

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def turn_index(self) -> int:
        return self._turn_index

    @property
    def temporal_latent_dim(self) -> int:
        return self._joint_loop.temporal_policy.parameter_store.n_z

    @property
    def residual_runtime(self) -> OpenWeightResidualRuntime:
        return self._joint_loop.residual_runtime or self._default_residual_runtime

    @property
    def evaluation_backbone(self) -> EvaluationBackbone:
        return self._evaluation_backbone

    @property
    def completed_session_reports(self) -> tuple[EvaluationReport, ...]:
        return tuple(self._completed_session_reports)

    @property
    def session_post_queue_state(self) -> SessionPostSlowLoopQueueState:
        return self._session_post_queue.snapshot()

    @property
    def active_context_session_id(self) -> str:
        return f"{self._session_id}:context-{self._context_index}"

    async def drain_session_post_slow_loop(self) -> tuple[SessionPostSlowLoopResult, ...]:
        self._session_post_queue.schedule()
        await self._session_post_queue.wait_for_idle()
        return self._session_post_queue.consume_completed_results()

    def begin_new_context(self, *, reason: str = "manual") -> tuple[str, ...]:
        operations: list[str] = []
        active_report = self._maybe_build_current_session_report()
        session_post_job = self._build_session_post_slow_loop_job(
            active_report=active_report,
        )
        if active_report is not None:
            self._completed_session_reports.append(active_report)
            operations.append(f"session-report:checkpoint:{self.active_context_session_id}")
        operations.extend(
            self._memory_store.reset_nested_context(
                reason=reason,
                timestamp_ms=max(self._turn_index, 1),
            )
        )
        self._context_index += 1
        self._upstream_snapshots = {}
        self._previous_substrate_snapshot = None
        self._previous_prediction_reward = 0.0
        self._previous_prediction_magnitude = 0.0
        self._previous_prediction_error = None
        self._recommended_z = None
        self._recent_substrate_batches = []
        if session_post_job is not None:
            self._session_post_queue.enqueue(session_post_job)
            self._session_post_queue.schedule()
            operations.append(f"session-post-slow-loop:enqueued:{session_post_job.job_id}")
        return tuple(operations)

    def _maybe_build_current_session_report(self) -> EvaluationReport | None:
        records = tuple(
            record
            for record in self._evaluation_backbone.records
            if record.session_id == self.active_context_session_id
        )
        if not records:
            return None
        return self._evaluation_backbone.build_session_report(
            session_id=self.active_context_session_id,
            timestamp_ms=max(record.timestamp_ms for record in records) + 1,
        )

    def build_current_session_report(self) -> EvaluationReport | None:
        return self._maybe_build_current_session_report()

    def _build_session_post_slow_loop_job(
        self,
        *,
        active_report: EvaluationReport | None,
    ) -> SessionPostSlowLoopJob | None:
        request = self._last_session_post_writeback_request
        if (
            active_report is None
            or request is None
            or request.context_session_id != self.active_context_session_id
        ):
            return None
        prediction_error_summary: tuple[tuple[str, float], ...] = ()
        if self._previous_prediction_error is not None:
            prediction_error_summary = (
                ("task_error", self._previous_prediction_error.task_error),
                ("relationship_error", self._previous_prediction_error.relationship_error),
                ("regime_error", self._previous_prediction_error.regime_error),
                ("action_error", self._previous_prediction_error.action_error),
                ("magnitude", self._previous_prediction_error.magnitude),
                ("signed_reward", self._previous_prediction_error.signed_reward),
            )
        job = SessionPostSlowLoopJob(
            job_id=f"{request.context_session_id}:slow-loop:{self._turn_index}",
            context_session_id=request.context_session_id,
            closed_at_turn=self._turn_index,
            session_report=active_report,
            prior_session_report_count=len(self._completed_session_reports),
            trace_count=len(self._recent_training_traces),
            substrate_batch_count=len(self._recent_substrate_batches),
            prediction_error_summary=prediction_error_summary,
            writeback_request=request,
            description=(
                f"Session-post slow loop job for {request.context_session_id} with "
                f"{len(self._recent_training_traces)} traces and {len(self._recent_substrate_batches)} substrate batches."
            ),
        )
        self._last_session_post_writeback_request = None
        return job

    async def _run_session_post_slow_loop_job(
        self,
        job: SessionPostSlowLoopJob,
    ) -> SessionPostSlowLoopResult:
        async with self._session_post_lock:
            writeback_result, _ = apply_session_post_writeback_request(
                request=job.writeback_request,
                memory_store=self._memory_store,
                temporal_policy=self._world_temporal_policy,
                regime_module=self._regime_module,
            )
        return SessionPostSlowLoopResult(
            job_id=job.job_id,
            context_session_id=job.context_session_id,
            closed_at_turn=job.closed_at_turn,
            writeback_result=writeback_result,
            applied=bool(writeback_result is not None and writeback_result.applied_operations),
            blocked=bool(writeback_result is not None and writeback_result.blocked_operations),
            description=(
                f"Session-post slow loop finished for {job.context_session_id} "
                f"applied={bool(writeback_result is not None and writeback_result.applied_operations)} "
                f"blocked={bool(writeback_result is not None and writeback_result.blocked_operations)}."
            ),
        )

    def _collect_session_post_writeback_result(self) -> WritebackResult | None:
        completed = self._session_post_queue.consume_completed_results()
        if not completed:
            return None
        applied_operations: tuple[str, ...] = ()
        blocked_operations: tuple[str, ...] = ()
        checkpoint = None
        descriptions: list[str] = []
        for result in completed:
            if result.writeback_result is None:
                descriptions.append(result.description)
                continue
            applied_operations = applied_operations + result.writeback_result.applied_operations
            blocked_operations = blocked_operations + result.writeback_result.blocked_operations
            if result.writeback_result.checkpoint is not None:
                checkpoint = result.writeback_result.checkpoint
            descriptions.append(result.writeback_result.description)
        return WritebackResult(
            applied_operations=applied_operations,
            blocked_operations=blocked_operations,
            checkpoint=checkpoint,
            description=" ".join(descriptions),
        )

    def apply_rare_heavy_artifact(
        self,
        artifact: RareHeavyArtifact,
        *,
        checkpoint_id: str | None = None,
    ) -> RareHeavyImportResult:
        result = self._joint_loop.apply_rare_heavy_artifact(
            artifact,
            checkpoint_id=checkpoint_id,
        )
        reset_operations = self._memory_store.reset_nested_context(
            reason="rare-heavy-import",
            timestamp_ms=max(self._turn_index, 1),
        )
        if not reset_operations:
            return result
        return replace(
            result,
            applied_operations=result.applied_operations + reset_operations,
            description=f"{result.description} Nested context reset applied after import.",
        )

    def rollback_rare_heavy_import(
        self,
        checkpoint: RareHeavyImportCheckpoint,
    ) -> tuple[str, ...]:
        return self._joint_loop.rollback_rare_heavy_import(checkpoint)

    async def run_turn(self, user_input: str) -> AgentTurnResult:
        deferred_writeback_result = self._collect_session_post_writeback_result()
        self._session_post_queue.schedule()
        async with self._session_post_lock:
            self._turn_index += 1
            wave_id = f"wave-{self._turn_index}"
            context_session_id = self.active_context_session_id
            substrate_adapter = self._build_substrate_adapter(user_input=user_input)
            trace = self._build_training_trace_from_substrate(user_input=user_input)
            self._record_training_trace(trace)
            pe_task_error = self._previous_prediction_error.task_error if self._previous_prediction_error is not None else 0.0
            pe_relationship_error = (
                self._previous_prediction_error.relationship_error if self._previous_prediction_error is not None else 0.0
            )
            pe_regime_error = self._previous_prediction_error.regime_error if self._previous_prediction_error is not None else 0.0
            pe_action_error = self._previous_prediction_error.action_error if self._previous_prediction_error is not None else 0.0
            if self._external_prediction_error_drive:
                self._joint_loop.set_external_learning_signals(
                    {
                        "prediction_error_reward": self._previous_prediction_reward,
                        "prediction_error_magnitude": self._previous_prediction_magnitude,
                        "prediction_error_task_error": pe_task_error,
                        "prediction_error_relationship_error": pe_relationship_error,
                        "prediction_error_regime_error": pe_regime_error,
                        "prediction_error_action_error": pe_action_error,
                    }
                    if (
                        abs(self._previous_prediction_reward) > 1e-8
                        or self._previous_prediction_magnitude > 1e-8
                    )
                    else {}
                )
            else:
                self._joint_loop.set_external_learning_signals({})
            joint_result = await self._joint_loop.run_scheduled_step(
                turn_index=self._turn_index,
                trace=trace,
                session_id=context_session_id,
                wave_id=wave_id,
                prior_session_reports=self.completed_session_reports,
                schedule=self._joint_schedule,
                apply_writeback=False,
            )
            integration_result = await run_final_wiring_turn(
                config=self._config,
                substrate_adapter=substrate_adapter,
                memory_store=self._memory_store,
                evaluation_backbone=self._evaluation_backbone,
                prior_session_reports=self.completed_session_reports,
                upstream_snapshots=self._upstream_snapshots,
                joint_loop_result=joint_result,
                credit_proposals=self._credit_proposals,
                reflection_mode=self._reflection_mode,
            world_temporal_policy=self._world_temporal_policy,
            self_temporal_policy=self._self_temporal_policy,
                prediction_module=self._prediction_module,
                regime_module=self._regime_module,
                session_id=context_session_id,
                wave_id=wave_id,
                apply_slow_writeback=False,
            )
            self._last_session_post_writeback_request = integration_result.session_post_writeback_request
            self._upstream_snapshots = {
                **integration_result.active_snapshots,
                **integration_result.shadow_snapshots,
            }
            substrate_snap = integration_result.active_snapshots.get("substrate")
            if substrate_snap is not None and isinstance(substrate_snap.value, SubstrateSnapshot):
                self._previous_substrate_snapshot = substrate_snap.value
                self._record_substrate_batch(self._substrate_batch_from_snapshot(substrate_snap.value))
            if integration_result.prediction_error_snapshot is not None:
                self._previous_prediction_reward = integration_result.prediction_error_snapshot.error.signed_reward
                self._previous_prediction_magnitude = integration_result.prediction_error_snapshot.error.magnitude
                self._previous_prediction_error = integration_result.prediction_error_snapshot.error
            imagination_result = self._run_imagination(integration_result)
            if imagination_result is not None:
                self._recommended_z = imagination_result.selected_trajectory.z_sequence[0]
            else:
                self._recommended_z = None
            rare_heavy_result = self._maybe_apply_rare_heavy(
                wave_id=wave_id,
                joint_result=joint_result,
            )
        self._session_post_queue.schedule()
        return self._to_turn_result(
            user_input=user_input,
            wave_id=wave_id,
            integration_result=integration_result,
            joint_result=joint_result,
            imagination_result=imagination_result,
            rare_heavy_result=rare_heavy_result,
            deferred_writeback_result=deferred_writeback_result,
            queue_state=self.session_post_queue_state,
        )

    def _record_training_trace(self, trace: TrainingTrace) -> None:
        self._recent_training_traces.append(trace)
        if len(self._recent_training_traces) > self._rare_heavy_trace_window:
            del self._recent_training_traces[:-self._rare_heavy_trace_window]

    def _record_substrate_batch(self, batch: tuple[SubstrateSnapshot, ...]) -> None:
        if not batch:
            return
        self._recent_substrate_batches.append(batch)
        if len(self._recent_substrate_batches) > self._rare_heavy_trace_window:
            del self._recent_substrate_batches[:-self._rare_heavy_trace_window]

    def _substrate_batch_from_snapshot(self, snapshot: SubstrateSnapshot) -> tuple[SubstrateSnapshot, ...]:
        if not snapshot.residual_sequence:
            return (snapshot,)
        return tuple(
            SubstrateSnapshot(
                model_id=snapshot.model_id,
                is_frozen=snapshot.is_frozen,
                surface_kind=snapshot.surface_kind,
                token_logits=snapshot.token_logits,
                feature_surface=step.feature_surface,
                residual_activations=step.residual_activations,
                residual_sequence=(step,),
                unavailable_fields=snapshot.unavailable_fields,
                description=f"{snapshot.description} rare-heavy-step={step.step}",
            )
            for step in snapshot.residual_sequence
        )

    def _effective_rare_heavy_pipeline_config(self) -> PipelineConfig:
        policy_n_z = self._joint_loop.temporal_policy.parameter_store.n_z
        if self._rare_heavy_pipeline_config.n_z == policy_n_z:
            return self._rare_heavy_pipeline_config
        return replace(self._rare_heavy_pipeline_config, n_z=policy_n_z)

    def _clone_memory_store_for_rare_heavy(self) -> MemoryStore:
        checkpoint = self._joint_loop.memory_store.export_rare_heavy_state(
            checkpoint_id=f"{self._session_id}:rare-heavy-seed:{self._turn_index}"
        )
        source_core = self._joint_loop.memory_store.learned_core
        learned_core = source_core.clone_empty() if source_core is not None else None
        cloned_store = MemoryStore(learned_core=learned_core)
        cloned_store.import_rare_heavy_state(checkpoint)
        return cloned_store

    def _build_rare_heavy_pipeline(self) -> SSLRLTrainingPipeline:
        policy_n_z = self._joint_loop.temporal_policy.parameter_store.n_z
        cloned_policy = FullLearnedTemporalPolicy(
            parameter_store=MetacontrollerParameterStore(n_z=policy_n_z),
        )
        cloned_policy.apply_rare_heavy_snapshot(
            self._joint_loop.temporal_policy.export_rare_heavy_snapshot()
        )
        return SSLRLTrainingPipeline(
            config=self._effective_rare_heavy_pipeline_config(),
            policy=cloned_policy,
            memory_store=self._clone_memory_store_for_rare_heavy(),
            residual_runtime=(
                self._joint_loop.residual_runtime.clone_for_rare_heavy()
                if self._joint_loop.residual_runtime is not None
                else None
            ),
        )

    def _maybe_apply_rare_heavy(
        self,
        *,
        wave_id: str,
        joint_result: ScheduledJointLoopResult,
    ) -> RareHeavyTurnResult | None:
        if not joint_result.rare_heavy_review_recommended:
            return None
        if not self._rare_heavy_enabled:
            return RareHeavyTurnResult(
                recommended=True,
                applied=False,
                artifact_id=None,
                applied_operations=(),
                substrate_status="skipped",
                substrate_training_mode="not-run",
                description="Rare-heavy review was recommended, but session owner has rare-heavy execution disabled.",
            )
        turns_since_last_import = self._turn_index - self._last_rare_heavy_turn_index
        if self._last_rare_heavy_turn_index and turns_since_last_import < self._rare_heavy_cooldown_turns:
            return RareHeavyTurnResult(
                recommended=True,
                applied=False,
                artifact_id=None,
                applied_operations=(),
                substrate_status="skipped",
                substrate_training_mode="not-run",
                description=(
                    f"Rare-heavy review was recommended, but cooldown is active "
                    f"({turns_since_last_import}/{self._rare_heavy_cooldown_turns} turns since last import)."
                ),
            )
        if len(self._recent_training_traces) < self._rare_heavy_min_traces:
            return RareHeavyTurnResult(
                recommended=True,
                applied=False,
                artifact_id=None,
                applied_operations=(),
                substrate_status="skipped",
                substrate_training_mode="not-run",
                description=(
                    f"Rare-heavy review was recommended, but only {len(self._recent_training_traces)} traces are available; "
                    f"need {self._rare_heavy_min_traces}."
                ),
            )
        pipeline = self._build_rare_heavy_pipeline()
        traces = tuple(self._recent_training_traces[-self._rare_heavy_trace_window :])
        substrate_batches = tuple(self._recent_substrate_batches[-self._rare_heavy_trace_window :])
        try:
            pipeline_result = pipeline.run_pipeline(
                traces=traces,
                substrate_steps_per_trace=substrate_batches if substrate_batches else None,
            )
        except RuntimeError as exc:
            return RareHeavyTurnResult(
                recommended=True,
                applied=False,
                artifact_id=None,
                applied_operations=(),
                substrate_status="incompatible",
                substrate_training_mode="unsupported",
                description=(
                    f"Rare-heavy pipeline failed closed during substrate training/import preparation: {exc}"
                ),
            )
        artifact = pipeline.export_rare_heavy_artifact(
            artifact_id=f"{self._session_id}:{wave_id}:rare-heavy"
        )
        if pipeline_result.rl_steps_completed <= 0:
            return RareHeavyTurnResult(
                recommended=True,
                applied=False,
                artifact_id=artifact.artifact_id,
                applied_operations=(),
                substrate_status="skipped",
                substrate_training_mode=pipeline_result.substrate_training_mode,
                description=(
                    f"Rare-heavy pipeline exported {artifact.artifact_id}, but no offline RL steps completed; "
                    f"skipping import. {pipeline_result.description}"
                ),
            )
        try:
            import_result = self.apply_rare_heavy_artifact(
                artifact,
                checkpoint_id=f"{self._session_id}:{wave_id}:rare-heavy-import",
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            return RareHeavyTurnResult(
                recommended=True,
                applied=False,
                artifact_id=artifact.artifact_id,
                applied_operations=(),
                substrate_status="incompatible",
                substrate_training_mode=pipeline_result.substrate_training_mode,
                description=(
                    f"{pipeline_result.description} Rare-heavy import failed closed: {exc}"
                ),
            )
        self._last_rare_heavy_turn_index = self._turn_index
        return RareHeavyTurnResult(
            recommended=True,
            applied=True,
            artifact_id=artifact.artifact_id,
            applied_operations=import_result.applied_operations,
            substrate_status="imported",
            substrate_training_mode=pipeline_result.substrate_training_mode,
            description=f"{pipeline_result.description} {import_result.description}",
        )

    def _build_substrate_adapter(self, *, user_input: str) -> SubstrateAdapter:
        if self._substrate_adapter_factory is not None:
            return self._substrate_adapter_factory(user_input, self._turn_index)
        return OpenWeightResidualStreamSubstrateAdapter(
            runtime=self._default_residual_runtime,
            default_source_text=user_input,
        )

    def _build_training_trace_from_substrate(self, *, user_input: str) -> TrainingTrace:
        """Build a training trace from real substrate data when available.

        When a previous turn produced a real substrate snapshot, construct
        the trace from its residual sequence.  Otherwise fall back to the
        simulated ``build_training_trace``.
        """
        prev = self._previous_substrate_snapshot
        if prev is None or not prev.residual_sequence:
            return build_training_trace(
                trace_id=f"{self._session_id}:joint:{self._turn_index}",
                source_text=user_input,
            )
        steps = tuple(
            TraceStep(
                step=rs.step,
                token=rs.token,
                feature_surface=rs.feature_surface,
                residual_activations=rs.residual_activations,
            )
            for rs in prev.residual_sequence
        )
        return TrainingTrace(
            trace_id=f"{self._session_id}:real:{self._turn_index}",
            source_text=user_input,
            steps=steps,
        )

    def _to_turn_result(
        self,
        *,
        user_input: str,
        wave_id: str,
        integration_result: FinalIntegrationResult,
        joint_result: ScheduledJointLoopResult,
        imagination_result: ImaginationResult | None = None,
        rare_heavy_result: RareHeavyTurnResult | None = None,
        deferred_writeback_result: WritebackResult | None = None,
        queue_state: SessionPostSlowLoopQueueState | None = None,
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
        prediction_error = None
        evaluated_prediction = None
        actual_outcome = None
        next_prediction = None
        evaluation_snapshot = integration_result.active_snapshots.get("evaluation")
        if evaluation_snapshot is not None and isinstance(evaluation_snapshot.value, EvaluationSnapshot):
            evaluation_alerts = evaluation_snapshot.value.alerts
        if integration_result.prediction_error_snapshot is not None:
            evaluated_prediction = integration_result.prediction_error_snapshot.evaluated_prediction
            actual_outcome = integration_result.prediction_error_snapshot.actual_outcome
            next_prediction = integration_result.prediction_error_snapshot.next_prediction
            prediction_error = integration_result.prediction_error_snapshot.error

        memory_retrieval_count = 0
        nested_profile_active = False
        nested_context_reset_applied = False
        nested_context_reset_total_count = 0
        slow_to_fast_init_benefit = 0.0
        slow_to_fast_target_distance_before = 0.0
        slow_to_fast_target_distance_after = 0.0
        slow_to_fast_target_alignment_gain = 0.0
        learned_memory_primary = False
        artifact_consolidation_count = 0
        learned_recall_count = 0
        learned_recall_confidence = 0.0
        learned_recall_core_guided = False
        memory_snapshot = integration_result.active_snapshots.get("memory")
        if memory_snapshot is not None and isinstance(memory_snapshot.value, MemorySnapshot):
            memory_retrieval_count = len(memory_snapshot.value.retrieved_entries)
            lifecycle_metrics = dict(memory_snapshot.value.lifecycle_metrics)
            nested_profile_active = lifecycle_metrics.get("nested_profile_active", 0.0) > 0.0
            nested_context_reset_applied = lifecycle_metrics.get("last_nested_reset_applied", 0.0) > 0.0
            nested_context_reset_total_count = int(lifecycle_metrics.get("nested_context_reset_count", 0.0))
            slow_to_fast_init_benefit = lifecycle_metrics.get("slow_to_fast_init_benefit", 0.0)
            slow_to_fast_target_distance_before = lifecycle_metrics.get("slow_to_fast_target_distance_before", 0.0)
            slow_to_fast_target_distance_after = lifecycle_metrics.get("slow_to_fast_target_distance_after", 0.0)
            slow_to_fast_target_alignment_gain = lifecycle_metrics.get("slow_to_fast_target_alignment_gain", 0.0)
            learned_memory_primary = lifecycle_metrics.get("learned_memory_primary", 0.0) > 0.0
            artifact_consolidation_count = int(lifecycle_metrics.get("artifact_consolidation_count", 0.0))
            learned_recall_count = int(lifecycle_metrics.get("learned_recall_count", 0.0))
            learned_recall_confidence = lifecycle_metrics.get("last_learned_recall_confidence", 0.0)
            learned_recall_core_guided = lifecycle_metrics.get("last_learned_recall_driver_is_core", 0.0) > 0.0

        substrate_model_id = None
        substrate_runtime_origin = getattr(self._default_residual_runtime, "runtime_origin", None)
        substrate_fallback_active = bool(getattr(self._default_residual_runtime, "fallback_active", False))
        substrate_capture_source = getattr(self._default_residual_runtime, "capture_source", None)
        substrate_residual_sequence_length = 0
        substrate_snapshot = integration_result.active_snapshots.get("substrate")
        if substrate_snapshot is not None and isinstance(substrate_snapshot.value, SubstrateSnapshot):
            substrate_model_id = substrate_snapshot.value.model_id
            substrate_residual_sequence_length = len(substrate_snapshot.value.residual_sequence)

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

        retrieved_memories: tuple[str, ...] = ()
        if memory_snapshot is not None and isinstance(memory_snapshot.value, MemorySnapshot):
            retrieved_memories = tuple(
                entry.content for entry in memory_snapshot.value.retrieved_entries[:5]
            )
        controller_description = ""
        control_code: tuple[float, ...] = ()
        if metacontroller_state is not None:
            controller_description = metacontroller_state.description
        if temporal_snapshot is not None and isinstance(
            temporal_snapshot.value, TemporalAbstractionSnapshot
        ):
            control_code = temporal_snapshot.value.controller_state.code

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
                user_input=user_input,
                retrieved_memories=retrieved_memories,
                controller_description=controller_description,
                control_code=control_code,
            )
        )
        effective_writeback_result = deferred_writeback_result or integration_result.writeback_result
        effective_queue_state = queue_state or self.session_post_queue_state

        return AgentTurnResult(
            session_id=self.active_context_session_id,
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
            evaluated_prediction=evaluated_prediction,
            actual_outcome=actual_outcome,
            next_prediction=next_prediction,
            prediction_error=prediction_error,
            bounded_writeback_applied=bool(
                effective_writeback_result is not None
                and effective_writeback_result.applied_operations
            ),
            writeback_source=integration_result.writeback_source,
            writeback_operations=effective_writeback_result.applied_operations
            if effective_writeback_result is not None
            else (),
            writeback_blocks=effective_writeback_result.blocked_operations
            if effective_writeback_result is not None
            else (),
            joint_schedule_action=joint_result.schedule_action,
            joint_learning_summary=joint_result.description,
            joint_cycle_report=joint_result.cycle_report,
            response=response,
            event_count=integration_result.event_count,
            substrate_model_id=substrate_model_id,
            substrate_runtime_origin=substrate_runtime_origin,
            substrate_fallback_active=substrate_fallback_active,
            substrate_capture_source=substrate_capture_source,
            substrate_residual_sequence_length=substrate_residual_sequence_length,
            reflection_promotion_eligible=integration_result.reflection_promotion_eligible,
            reflection_promotion_reason=integration_result.reflection_promotion_reason,
            imagination_result=imagination_result,
            rare_heavy_result=rare_heavy_result,
            evolution_judgement=integration_result.evolution_judgement,
            cross_session_verdict=integration_result.cross_session_verdict,
            nested_profile_active=nested_profile_active,
            nested_context_reset_applied=nested_context_reset_applied,
            nested_context_reset_total_count=nested_context_reset_total_count,
            slow_to_fast_init_benefit=slow_to_fast_init_benefit,
            slow_to_fast_target_distance_before=slow_to_fast_target_distance_before,
            slow_to_fast_target_distance_after=slow_to_fast_target_distance_after,
            slow_to_fast_target_alignment_gain=slow_to_fast_target_alignment_gain,
            learned_memory_primary=learned_memory_primary,
            artifact_consolidation_count=artifact_consolidation_count,
            learned_recall_count=learned_recall_count,
            learned_recall_confidence=learned_recall_confidence,
            learned_recall_core_guided=learned_recall_core_guided,
            session_post_pending_job_count=effective_queue_state.pending_job_count,
            session_post_completed_job_count=effective_queue_state.completed_job_count,
            session_post_last_completed_job_id=effective_queue_state.last_completed_job_id,
        )

    def _run_imagination(self, integration_result: FinalIntegrationResult) -> ImaginationResult | None:
        evaluation_snapshot = integration_result.active_snapshots.get("evaluation")
        dual_track_snapshot = integration_result.active_snapshots.get("dual_track")
        regime_snapshot = integration_result.active_snapshots.get("regime")
        if (
            evaluation_snapshot is None
            or not isinstance(evaluation_snapshot.value, EvaluationSnapshot)
            or dual_track_snapshot is None
        ):
            return None
        from volvence_zero.dual_track import DualTrackSnapshot as DTS

        if not isinstance(dual_track_snapshot.value, DTS):
            return None
        regime_value = (
            regime_snapshot.value
            if regime_snapshot is not None and isinstance(regime_snapshot.value, RegimeSnapshot)
            else None
        )
        metacontroller_state = integration_result.temporal_runtime_state
        prior_mean: tuple[float, ...] = (0.5, 0.5, 0.5)
        prior_std: tuple[float, ...] = (0.1, 0.1, 0.1)
        action_family_centroids: tuple[tuple[str, tuple[float, ...]], ...] = ()
        if metacontroller_state is not None:
            prior_mean = metacontroller_state.prior_mean or prior_mean
            prior_std = metacontroller_state.prior_std or prior_std
            action_family_centroids = tuple(
                (summary.family_id, (summary.stability, summary.switch_bias, summary.competition_score))
                for summary in metacontroller_state.action_family_summaries
                if summary.support >= 2
            )
        previous_prediction = (
            integration_result.prediction_error_snapshot.next_prediction
            if integration_result.prediction_error_snapshot is not None
            else None
        )
        return imagine(
            current_substrate=self._previous_substrate_snapshot,
            current_evaluation=evaluation_snapshot.value,
            current_dual_track=dual_track_snapshot.value,
            current_regime=regime_value,
            previous_prediction=previous_prediction,
            action_family_centroids=action_family_centroids,
            prior_mean=prior_mean,
            prior_std=prior_std,
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


def llm_active_runner(
    *,
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    model_source: str | None = None,
    device: str = "auto",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    local_files_only: bool = False,
    session_id: str = "llm-session",
) -> AgentSessionRunner:
    """Create a runner that uses a real LLM for response generation."""
    runtime = build_transformers_runtime_with_fallback(
        model_id=model_id,
        model_source=model_source,
        device=device,
        local_files_only=local_files_only,
    )
    synthesizer = LLMResponseSynthesizer(
        runtime=runtime,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return AgentSessionRunner(
        session_id=session_id,
        default_residual_runtime=runtime,
        response_synthesizer=synthesizer,
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
    )


async def run_substrate_path_benchmark(
    *,
    path_label: str,
    runner: AgentSessionRunner,
    user_inputs: tuple[str, ...],
) -> SubstrateBenchmarkReport:
    turns: list[SubstrateBenchmarkTurn] = []
    metric_totals: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    for user_input in user_inputs:
        result = await runner.run_turn(user_input)
        eval_snapshot = result.active_snapshots.get("evaluation")
        turn_score_count = 0
        metric_pairs: tuple[tuple[str, float], ...] = ()
        if eval_snapshot is not None and isinstance(eval_snapshot.value, EvaluationSnapshot):
            turn_score_count = len(eval_snapshot.value.turn_scores)
            metric_pairs = tuple(
                (f"{score.family}:{score.metric_name}", score.value)
                for score in eval_snapshot.value.turn_scores
            )
            for key, value in metric_pairs:
                metric_totals[key] = metric_totals.get(key, 0.0) + value
                metric_counts[key] = metric_counts.get(key, 0) + 1
        family_version = 0
        temporal_snapshot = result.active_snapshots.get("temporal_abstraction")
        if temporal_snapshot is not None and isinstance(temporal_snapshot.value, TemporalAbstractionSnapshot):
            family_version = temporal_snapshot.value.action_family_version
        policy_objective = (
            result.joint_cycle_report.policy_objective
            if result.joint_cycle_report is not None
            else 0.0
        )
        turns.append(
            SubstrateBenchmarkTurn(
                turn_index=runner.turn_index,
                substrate_runtime_origin=result.substrate_runtime_origin,
                substrate_fallback_active=result.substrate_fallback_active,
                substrate_capture_source=result.substrate_capture_source,
                substrate_residual_sequence_length=result.substrate_residual_sequence_length,
                active_regime=result.active_regime,
                active_abstract_action=result.active_abstract_action,
                joint_schedule_action=result.joint_schedule_action,
                acceptance_passed=result.acceptance_passed,
                turn_score_count=turn_score_count,
                evaluation_alert_count=len(result.evaluation_alerts),
                policy_objective=policy_objective,
                action_family_version=family_version,
                metrics=metric_pairs,
            )
        )
    acceptance_rate = (
        sum(1 for turn in turns if turn.acceptance_passed) / max(len(turns), 1)
        if turns
        else 0.0
    )
    mean_seq_len = (
        sum(turn.substrate_residual_sequence_length for turn in turns) / max(len(turns), 1)
        if turns
        else 0.0
    )
    mean_turn_scores = (
        sum(turn.turn_score_count for turn in turns) / max(len(turns), 1)
        if turns
        else 0.0
    )
    full_cycle_count = sum(1 for turn in turns if turn.joint_schedule_action in {"full-cycle", "full-cycle-pe"})
    metric_means = tuple(
        sorted(
            (
                key,
                round(metric_totals[key] / max(metric_counts.get(key, 1), 1), 4),
            )
            for key in metric_totals
        )
    )
    mean_policy_objective = (
        sum(turn.policy_objective for turn in turns) / max(len(turns), 1)
        if turns
        else 0.0
    )
    max_family_version = max((turn.action_family_version for turn in turns), default=0)
    return SubstrateBenchmarkReport(
        path_label=path_label,
        turns=tuple(turns),
        acceptance_rate=acceptance_rate,
        mean_residual_sequence_length=mean_seq_len,
        mean_turn_score_count=mean_turn_scores,
        full_cycle_count=full_cycle_count,
        metric_means=metric_means,
        mean_policy_objective=mean_policy_objective,
        max_family_version=max_family_version,
        description=(
            f"Substrate benchmark path={path_label} turns={len(turns)} "
            f"acceptance_rate={acceptance_rate:.2f} mean_seq_len={mean_seq_len:.2f} "
            f"full_cycles={full_cycle_count} mean_policy_objective={mean_policy_objective:.3f} "
            f"max_family_version={max_family_version}."
        ),
    )


async def run_multi_path_benchmark(
    *,
    baseline_label: str,
    path_runners: tuple[tuple[str, AgentSessionRunner], ...],
    user_inputs: tuple[str, ...],
) -> MultiPathBenchmarkReport:
    reports: list[SubstrateBenchmarkReport] = []
    for label, runner in path_runners:
        report = await run_substrate_path_benchmark(
            path_label=label,
            runner=runner,
            user_inputs=user_inputs,
        )
        reports.append(report)
    baseline_report = next(report for report in reports if report.path_label == baseline_label)
    baseline_metrics = dict(baseline_report.metric_means)
    metric_deltas: list[tuple[str, tuple[tuple[str, float], ...]]] = []
    for report in reports:
        if report.path_label == baseline_label:
            continue
        report_metrics = dict(report.metric_means)
        keys = sorted(set(report_metrics) | set(baseline_metrics))
        deltas = tuple(
            (key, round(report_metrics.get(key, 0.0) - baseline_metrics.get(key, 0.0), 4))
            for key in keys
        )
        metric_deltas.append((report.path_label, deltas))
    return MultiPathBenchmarkReport(
        path_reports=tuple(reports),
        metric_deltas_from_baseline=tuple(metric_deltas),
        baseline_label=baseline_label,
        description=(
            f"Multi-path benchmark over {len(user_inputs)} turns with baseline={baseline_label} "
            f"across {len(reports)} paths."
        ),
    )
