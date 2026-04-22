from __future__ import annotations

from collections.abc import Callable
import asyncio
from dataclasses import dataclass, replace
import os
from typing import Any

from volvence_zero.application.runtime import (
    ApplicationCaseCluster,
    ApplicationRareHeavyCheckpoint,
    ApplicationRareHeavyState,
    BoundaryPolicySnapshot,
    CaseMemorySnapshot,
    DomainKnowledgeSnapshot,
    ExperienceConsolidationModule,
    ExperienceConsolidationSnapshot,
    ExperienceDelta,
    StrategyPlaybookSnapshot,
)
from volvence_zero.application.storage import (
    ApplicationCaseMemoryStore,
    ApplicationDomainKnowledgeStore,
    build_default_case_memory_store,
    build_default_domain_knowledge_store,
    build_filesystem_persistence_backend,
)
from volvence_zero.agent.response import AgentResponse, LLMResponseSynthesizer, ResponseContext, ResponseSynthesizer
from volvence_zero.credit.gate import (
    GateDecision,
    ModificationGate,
    ModificationProposal,
    SelfModificationRecord,
    extend_credit_snapshot,
    evaluate_gate,
)
from volvence_zero.evaluation.backbone import (
    EvaluationBackbone,
    EvaluationReport,
    EvaluationScore,
    EvaluationSnapshot,
    EvolutionDecision,
    EvolutionJudgement,
)
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
    OnlineFastImportResult,
)
from volvence_zero.memory import MemorySnapshot, MemoryStore, build_default_memory_store
from volvence_zero.planning import ImaginationResult, imagine
from volvence_zero.prediction.error import (
    ActualOutcome,
    PredictedOutcome,
    PredictionError,
    PredictionErrorModule,
)
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
    SubstrateSelfModSnapshot,
    SubstrateSnapshot,
    TrainingTrace,
    TraceStep,
    build_training_trace,
)
from volvence_zero.temporal import (
    DualTrackRareHeavySnapshot,
    FullLearnedTemporalPolicy,
    MetacontrollerParameterStore,
    MetacontrollerRuntimeState,
    TemporalAbstractionSnapshot,
    TemporalPolicy,
    clone_full_learned_temporal_policy,
)
from volvence_zero.agent.session_post_slow_loop import (
    SessionPostSlowLoopJob,
    SessionPostSlowLoopModule,
    SessionPostSlowLoopQueue,
    SessionPostSlowLoopQueueState,
    SessionPostSlowLoopResult,
    SessionPostSlowLoopSnapshot,
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
    tower_consolidation_count: int = 0
    learned_recall_count: int = 0
    learned_recall_confidence: float = 0.0
    learned_recall_core_guided: bool = False
    memory_tower_depth: int = 0
    memory_tower_alignment: float = 0.0
    memory_tower_profile_id: str = ""
    session_post_pending_job_count: int = 0
    session_post_completed_job_count: int = 0
    session_post_last_completed_job_id: str | None = None
    online_fast_substrate_result: "OnlineFastSubstrateTurnResult | None" = None


@dataclass(frozen=True)
class RareHeavyTurnResult:
    recommended: bool
    applied: bool
    artifact_id: str | None
    applied_operations: tuple[str, ...]
    substrate_status: str
    substrate_training_mode: str
    description: str
    import_decision: str = "not-run"
    reject_reason: str = ""
    bundle_alignment_ratio: float = 0.0
    bundle_trace_count: int = 0
    bundle_substrate_batch_count: int = 0
    bundle_mean_trace_step_count: float = 0.0
    bundle_mean_sequence_length: float = 0.0
    bundle_mean_residual_magnitude: float = 0.0
    candidate_adapter_parameter_count: int = 0
    candidate_adapter_training_loss: float = 0.0
    pre_import_case_count: int = 0
    pre_import_mean_score_delta: float = 0.0
    pre_import_worst_score_delta: float = 0.0
    pre_import_positive_fraction: float = 0.0
    pre_import_passed: bool = False
    pre_import_judgement: str = "not-run"


@dataclass(frozen=True)
class RareHeavyTrainingExample:
    turn_index: int
    wave_id: str
    source_text: str
    trace: TrainingTrace
    substrate_batch: tuple[SubstrateSnapshot, ...]
    description: str


@dataclass(frozen=True)
class RareHeavyTrainingBundle:
    examples: tuple[RareHeavyTrainingExample, ...]
    trace_count: int
    substrate_batch_count: int
    aligned_example_count: int
    alignment_ratio: float
    mean_trace_step_count: float
    mean_sequence_length: float
    mean_residual_magnitude: float
    description: str


@dataclass(frozen=True)
class RareHeavyPreImportEvaluation:
    accepted: bool
    case_count: int
    baseline_mean_score: float
    candidate_mean_score: float
    mean_score_delta: float
    worst_score_delta: float
    positive_fraction: float
    judgement: str
    reasons: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class OnlineFastSubstrateTurnResult:
    recommended: bool
    applied: bool
    gate_decision: str
    applied_operations: tuple[str, ...]
    blocked_operations: tuple[str, ...]
    parameter_change_rate: float
    optimizer_state_norm: float
    checkpoint_id: str
    fast_state_hash: str
    source_fast_state_hash: str
    optimizer_state_description: str
    description: str
    fast_memory_signal: tuple[float, ...] = ()


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
        domain_knowledge_store: ApplicationDomainKnowledgeStore | None = None,
        case_memory_store: ApplicationCaseMemoryStore | None = None,
        application_persistence_dir: str | None = None,
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
        allow_live_substrate_mutation: bool = False,
        joint_loop: ETANLJointLoop | None = None,
        joint_schedule: JointLoopSchedule | None = None,
        rare_heavy_enabled: bool = True,
        rare_heavy_trace_window: int = 5,
        rare_heavy_min_traces: int = 4,
        rare_heavy_cooldown_turns: int = 3,
        rare_heavy_pipeline_config: PipelineConfig | None = None,
        external_prediction_error_drive: bool = True,
        prediction_error_readout_only: bool = False,
        primary_prediction_error_dominance_enabled: bool = True,
    ) -> None:
        self._session_id = session_id
        self._config = config or FinalRolloutConfig()
        self._reflection_mode = reflection_mode
        if world_temporal_policy is not None:
            self._world_temporal_policy = world_temporal_policy
        elif temporal_policy is not None:
            self._world_temporal_policy = temporal_policy
        else:
            self._world_temporal_policy = FullLearnedTemporalPolicy()
        if self_temporal_policy is not None:
            self._self_temporal_policy = self_temporal_policy
        elif isinstance(self._world_temporal_policy, FullLearnedTemporalPolicy):
            self._self_temporal_policy = clone_full_learned_temporal_policy(self._world_temporal_policy)
        else:
            self._self_temporal_policy = self._world_temporal_policy
        self._temporal_policy = self._world_temporal_policy
        self._evaluation_backbone = EvaluationBackbone()
        self._application_rare_heavy_state = ApplicationRareHeavyState()
        domain_backend = None
        case_backend = None
        if application_persistence_dir is not None:
            domain_backend = build_filesystem_persistence_backend(
                base_dir=os.path.join(application_persistence_dir, "domain_knowledge")
            )
            case_backend = build_filesystem_persistence_backend(
                base_dir=os.path.join(application_persistence_dir, "case_memory")
            )
        self._domain_knowledge_store = domain_knowledge_store or build_default_domain_knowledge_store(
            persistence_backend=domain_backend
        )
        self._case_memory_store = case_memory_store or build_default_case_memory_store(
            persistence_backend=case_backend
        )
        world_parameter_store = getattr(self._world_temporal_policy, "parameter_store", None)
        default_latent_dim = world_parameter_store.n_z if world_parameter_store is not None else 16
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
            allow_live_substrate_mutation=allow_live_substrate_mutation,
        )
        self._joint_loop = joint_loop or ETANLJointLoop(
            world_policy=self._world_temporal_policy,
            self_policy=self._self_temporal_policy,
            memory_store=self._memory_store,
            residual_runtime=self._default_residual_runtime,
            evaluation_backbone=self._evaluation_backbone,
            primary_prediction_error_dominance_enabled=primary_prediction_error_dominance_enabled,
        )
        self._joint_loop.set_primary_prediction_error_dominance_enabled(primary_prediction_error_dominance_enabled)
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
        self._prediction_error_readout_only = prediction_error_readout_only
        self._primary_prediction_error_dominance_enabled = primary_prediction_error_dominance_enabled
        self._turn_index = 0
        self._upstream_snapshots: dict[str, Snapshot[Any]] = {}
        self._previous_substrate_snapshot: SubstrateSnapshot | None = None
        self._previous_prediction_reward: float = 0.0
        self._previous_prediction_magnitude: float = 0.0
        self._previous_prediction_error: PredictionError | None = None
        self._recommended_z: tuple[float, ...] | None = None
        self._recent_training_traces: list[TrainingTrace] = []
        self._recent_substrate_batches: list[tuple[SubstrateSnapshot, ...]] = []
        self._recent_rare_heavy_examples: list[RareHeavyTrainingExample] = []
        self._last_rare_heavy_turn_index = 0
        self._context_index = 1
        self._completed_session_reports: list[EvaluationReport] = []
        self._session_post_lock = asyncio.Lock()
        self._session_post_queue = SessionPostSlowLoopQueue(worker=self._run_session_post_slow_loop_job)
        self._session_post_module = SessionPostSlowLoopModule(
            wiring_level=self._config.level_for("session_post_slow_loop", WiringLevel.SHADOW),
        )
        self._session_post_snapshot: Snapshot[SessionPostSlowLoopSnapshot] | None = None
        self._experience_consolidation_module = ExperienceConsolidationModule(
            wiring_level=self._config.level_for("experience_consolidation", WiringLevel.SHADOW),
        )
        self._experience_consolidation_snapshot: Snapshot[ExperienceConsolidationSnapshot] | None = None
        self._last_session_post_writeback_request: SessionPostWritebackRequest | None = None
        self._publish_session_post_snapshot()
        self._publish_experience_consolidation_snapshot()

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
    def session_post_snapshot(self) -> Snapshot[SessionPostSlowLoopSnapshot] | None:
        return self._session_post_snapshot

    @property
    def experience_consolidation_snapshot(self) -> Snapshot[ExperienceConsolidationSnapshot] | None:
        return self._experience_consolidation_snapshot

    @property
    def active_context_session_id(self) -> str:
        return f"{self._session_id}:context-{self._context_index}"

    async def drain_session_post_slow_loop(self) -> tuple[SessionPostSlowLoopResult, ...]:
        self._session_post_queue.schedule()
        await self._session_post_queue.wait_for_idle()
        results = self._session_post_queue.consume_completed_results()
        self._publish_session_post_snapshot(completed_results=results)
        self._publish_experience_consolidation_snapshot(completed_results=results)
        return results

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
        self._recent_rare_heavy_examples = []
        if session_post_job is not None:
            self._session_post_queue.enqueue(session_post_job)
            self._session_post_queue.schedule()
            operations.append(f"session-post-slow-loop:enqueued:{session_post_job.job_id}")
        self._publish_session_post_snapshot()
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
        case_problem_patterns: tuple[str, ...] = ()
        case_risk_markers: tuple[str, ...] = ()
        knowledge_domains: tuple[str, ...] = ()
        boundary_trigger_reasons: tuple[str, ...] = ()
        case_snapshot = self._upstream_snapshots.get("case_memory")
        if case_snapshot is not None and isinstance(case_snapshot.value, CaseMemorySnapshot):
            case_problem_patterns = case_snapshot.value.active_problem_patterns
            case_risk_markers = case_snapshot.value.active_risk_markers
        knowledge_snapshot = self._upstream_snapshots.get("domain_knowledge")
        if knowledge_snapshot is not None and isinstance(knowledge_snapshot.value, DomainKnowledgeSnapshot):
            knowledge_domains = knowledge_snapshot.value.active_domains
        boundary_snapshot = self._upstream_snapshots.get("boundary_policy")
        if boundary_snapshot is not None and isinstance(boundary_snapshot.value, BoundaryPolicySnapshot):
            boundary_trigger_reasons = boundary_snapshot.value.trigger_reasons
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
            case_problem_patterns=case_problem_patterns,
            case_risk_markers=case_risk_markers,
            knowledge_domains=knowledge_domains,
            boundary_trigger_reasons=boundary_trigger_reasons,
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
        experience_deltas: list[ExperienceDelta] = []
        for pattern in job.case_problem_patterns:
            experience_deltas.append(
                ExperienceDelta(
                    delta_id=f"{job.job_id}:case:{pattern}",
                    delta_type="case-promotion",
                    target_slot="case_memory",
                    summary=f"Promote case pattern {pattern} into durable case evidence.",
                    confidence=0.62,
                    blocked=False,
                    description=f"Derived from session-post job {job.job_id} for pattern={pattern}.",
                )
            )
            experience_deltas.append(
                ExperienceDelta(
                    delta_id=f"{job.job_id}:playbook:{pattern}",
                    delta_type="playbook-delta",
                    target_slot="strategy_playbook",
                    summary=f"Extract a reusable ordering prior for pattern {pattern}.",
                    confidence=0.58,
                    blocked=False,
                    description=f"Derived from session-post job {job.job_id} for pattern={pattern}.",
                )
            )
        if job.boundary_trigger_reasons:
            experience_deltas.append(
                ExperienceDelta(
                    delta_id=f"{job.job_id}:boundary",
                    delta_type="boundary-delta",
                    target_slot="boundary_policy",
                    summary="Preserve repeated boundary triggers as future caution priors.",
                    confidence=0.57,
                    blocked=False,
                    description=(
                        f"Derived from session-post job {job.job_id} with triggers="
                        f"{', '.join(job.boundary_trigger_reasons)}."
                    ),
                )
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
            experience_deltas=tuple(experience_deltas),
        )

    def _publish_session_post_snapshot(
        self,
        *,
        completed_results: tuple[SessionPostSlowLoopResult, ...] = (),
    ) -> Snapshot[SessionPostSlowLoopSnapshot]:
        self._session_post_snapshot = self._session_post_module.publish_snapshot(
            queue_state=self.session_post_queue_state,
            completed_results=completed_results,
        )
        return self._session_post_snapshot

    def _publish_experience_consolidation_snapshot(
        self,
        *,
        completed_results: tuple[SessionPostSlowLoopResult, ...] = (),
    ) -> Snapshot[ExperienceConsolidationSnapshot]:
        self._experience_consolidation_snapshot = self._experience_consolidation_module.publish_snapshot(
            completed_results=completed_results,
        )
        return self._experience_consolidation_snapshot

    def _collect_session_post_writeback_result(self) -> WritebackResult | None:
        completed = self._session_post_queue.consume_completed_results()
        if completed:
            self._publish_session_post_snapshot(completed_results=completed)
            self._publish_experience_consolidation_snapshot(completed_results=completed)
        else:
            self._publish_session_post_snapshot()
            self._publish_experience_consolidation_snapshot()
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
        application_checkpoint = self._application_rare_heavy_state.export_rare_heavy_state(
            checkpoint_id=f"{checkpoint_id or artifact.artifact_id}:application-preimport"
        )
        result = self._joint_loop.apply_rare_heavy_artifact(
            artifact,
            checkpoint_id=checkpoint_id,
        )
        application_operations: tuple[str, ...] = ()
        if artifact.application_checkpoint is not None:
            application_operations = self._application_rare_heavy_state.import_rare_heavy_state(
                artifact.application_checkpoint
            )
        reset_operations = self._memory_store.reset_nested_context(
            reason="rare-heavy-import",
            timestamp_ms=max(self._turn_index, 1),
        )
        result = replace(
            result,
            checkpoint=replace(result.checkpoint, application_checkpoint=application_checkpoint),
        )
        if not reset_operations and not application_operations:
            return result
        return replace(
            result,
            applied_operations=result.applied_operations + application_operations + reset_operations,
            description=(
                f"{result.description} "
                f"{'Application rare-heavy state imported. ' if application_operations else ''}"
                f"{'Nested context reset applied after import.' if reset_operations else ''}"
            ).strip(),
        )

    def review_rare_heavy_artifact(
        self,
        artifact: RareHeavyArtifact,
        *,
        checkpoint_id: str | None = None,
    ) -> RareHeavyImportResult:
        return self._joint_loop.review_rare_heavy_artifact(
            artifact,
            checkpoint_id=checkpoint_id,
        )

    def rollback_rare_heavy_import(
        self,
        checkpoint: RareHeavyImportCheckpoint,
    ) -> tuple[str, ...]:
        operations = list(self._joint_loop.rollback_rare_heavy_import(checkpoint))
        if checkpoint.application_checkpoint is not None:
            operations.extend(self._application_rare_heavy_state.restore_rare_heavy_state(checkpoint.application_checkpoint))
        return tuple(operations)

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
                readout_only_signals = {}
                if (
                    self._prediction_error_readout_only
                    and (
                        abs(self._previous_prediction_reward) > 1e-8
                        or self._previous_prediction_magnitude > 1e-8
                    )
                ):
                    readout_only_signals = {
                        "prediction_error_reward_readout": self._previous_prediction_reward,
                        "prediction_error_magnitude_readout": min(self._previous_prediction_magnitude / 4.0, 1.0),
                    }
                self._joint_loop.set_external_learning_signals(readout_only_signals)
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
                application_rare_heavy_state=self._application_rare_heavy_state,
                domain_knowledge_store=self._domain_knowledge_store,
                case_memory_store=self._case_memory_store,
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
                substrate_self_mod_pe_magnitude=self._previous_prediction_magnitude,
                substrate_self_mod_pe_reward=self._previous_prediction_reward,
                substrate_self_mod_pe_threshold=self._joint_schedule.pe_substrate_online_fast_threshold,
            )
            self._last_session_post_writeback_request = integration_result.session_post_writeback_request
            self._upstream_snapshots = {
                **integration_result.active_snapshots,
                **integration_result.shadow_snapshots,
            }
            case_snapshot = integration_result.active_snapshots.get("case_memory")
            if (
                self._config.is_active("case_memory")
                and case_snapshot is not None
                and isinstance(case_snapshot.value, CaseMemorySnapshot)
            ):
                from volvence_zero.application.runtime import CaseMemoryModule

                self._case_memory_store.upsert_records(
                    CaseMemoryModule.records_from_snapshot(case_snapshot.value)
                )
                self._case_memory_store.save_to_backend()
            substrate_snap = integration_result.active_snapshots.get("substrate")
            if substrate_snap is not None and isinstance(substrate_snap.value, SubstrateSnapshot):
                self._previous_substrate_snapshot = substrate_snap.value
                substrate_batch = self._substrate_batch_from_snapshot(substrate_snap.value)
                self._record_substrate_batch(substrate_batch)
                self._record_rare_heavy_example(
                    wave_id=wave_id,
                    user_input=user_input,
                    trace=trace,
                    substrate_batch=substrate_batch,
                )
            if integration_result.prediction_error_snapshot is not None:
                self._previous_prediction_reward = integration_result.prediction_error_snapshot.error.signed_reward
                self._previous_prediction_magnitude = integration_result.prediction_error_snapshot.error.magnitude
                self._previous_prediction_error = integration_result.prediction_error_snapshot.error
            imagination_result = self._run_imagination(integration_result)
            if imagination_result is not None:
                self._recommended_z = imagination_result.selected_trajectory.z_sequence[0]
            else:
                self._recommended_z = None
            online_fast_substrate_result = self._maybe_apply_online_fast_substrate_self_mod(
                wave_id=wave_id,
                joint_result=joint_result,
                integration_result=integration_result,
            )
            rare_heavy_result = await self._maybe_apply_rare_heavy(
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
            online_fast_substrate_result=online_fast_substrate_result,
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

    def _record_rare_heavy_example(
        self,
        *,
        wave_id: str,
        user_input: str,
        trace: TrainingTrace,
        substrate_batch: tuple[SubstrateSnapshot, ...],
    ) -> None:
        if not substrate_batch:
            return
        self._recent_rare_heavy_examples.append(
            RareHeavyTrainingExample(
                turn_index=self._turn_index,
                wave_id=wave_id,
                source_text=user_input,
                trace=trace,
                substrate_batch=substrate_batch,
                description=(
                    f"Rare-heavy example turn={self._turn_index} wave={wave_id} "
                    f"trace_steps={len(trace.steps)} substrate_steps={len(substrate_batch)}."
                ),
            )
        )
        if len(self._recent_rare_heavy_examples) > self._rare_heavy_trace_window:
            del self._recent_rare_heavy_examples[:-self._rare_heavy_trace_window]

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

    def _build_rare_heavy_training_bundle(self) -> RareHeavyTrainingBundle:
        examples = tuple(self._recent_rare_heavy_examples[-self._rare_heavy_trace_window :])
        if not examples:
            return RareHeavyTrainingBundle(
                examples=(),
                trace_count=0,
                substrate_batch_count=0,
                aligned_example_count=0,
                alignment_ratio=0.0,
                mean_trace_step_count=0.0,
                mean_sequence_length=0.0,
                mean_residual_magnitude=0.0,
                description="No aligned rare-heavy training examples are available.",
            )
        trace_count = len(examples)
        substrate_batch_count = sum(1 for example in examples if example.substrate_batch)
        aligned_example_count = sum(
            1
            for example in examples
            if example.trace.steps and example.substrate_batch
        )
        alignment_ratio = aligned_example_count / max(trace_count, 1)
        mean_trace_step_count = sum(len(example.trace.steps) for example in examples) / trace_count
        flattened_substrates = tuple(
            snapshot
            for example in examples
            for snapshot in example.substrate_batch
        )
        mean_sequence_length = (
            sum(max(len(snapshot.residual_sequence), 1) for snapshot in flattened_substrates)
            / max(len(flattened_substrates), 1)
            if flattened_substrates
            else 0.0
        )
        residual_values = tuple(
            abs(value)
            for snapshot in flattened_substrates
            for activation in snapshot.residual_activations
            for value in activation.activation
        )
        mean_residual_magnitude = (
            sum(residual_values) / len(residual_values)
            if residual_values
            else 0.0
        )
        return RareHeavyTrainingBundle(
            examples=examples,
            trace_count=trace_count,
            substrate_batch_count=substrate_batch_count,
            aligned_example_count=aligned_example_count,
            alignment_ratio=alignment_ratio,
            mean_trace_step_count=mean_trace_step_count,
            mean_sequence_length=mean_sequence_length,
            mean_residual_magnitude=mean_residual_magnitude,
            description=(
                f"Rare-heavy bundle examples={trace_count} aligned={aligned_example_count} "
                f"alignment={alignment_ratio:.2f} mean_trace_steps={mean_trace_step_count:.2f} "
                f"mean_sequence_len={mean_sequence_length:.2f}."
            ),
        )

    def _clone_memory_store_for_rare_heavy(self) -> MemoryStore:
        checkpoint = self._joint_loop.memory_store.export_rare_heavy_state(
            checkpoint_id=f"{self._session_id}:rare-heavy-seed:{self._turn_index}"
        )
        source_core = self._joint_loop.memory_store.learned_core
        learned_core = source_core.clone_empty() if source_core is not None else None
        cloned_store = MemoryStore(learned_core=learned_core)
        cloned_store.import_rare_heavy_state(checkpoint)
        return cloned_store

    def _build_rare_heavy_pipeline(
        self,
        *,
        source_policy: FullLearnedTemporalPolicy,
    ) -> SSLRLTrainingPipeline:
        policy_n_z = source_policy.parameter_store.n_z
        cloned_policy = FullLearnedTemporalPolicy(
            parameter_store=MetacontrollerParameterStore(n_z=policy_n_z),
        )
        cloned_policy.apply_rare_heavy_snapshot(
            source_policy.export_rare_heavy_snapshot()
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

    def _build_rare_heavy_replay_runner(
        self,
        *,
        label: str,
        artifact: RareHeavyArtifact | None = None,
    ) -> "AgentSessionRunner":
        runner = AgentSessionRunner(
            session_id=f"{self._session_id}:rare-heavy-replay:{label}",
            config=self._config,
            memory_store=self._clone_memory_store_for_rare_heavy(),
            reflection_mode=WritebackMode.PROPOSAL_ONLY,
            world_temporal_policy=clone_full_learned_temporal_policy(self._joint_loop.world_temporal_policy),
            self_temporal_policy=clone_full_learned_temporal_policy(self._joint_loop.self_temporal_policy),
            default_residual_runtime=(
                self.residual_runtime.clone_for_rare_heavy()
                if self.residual_runtime is not None
                else self._default_residual_runtime
            ),
            joint_schedule=JointLoopSchedule(ssl_interval=0, rl_interval=0),
            rare_heavy_enabled=False,
            external_prediction_error_drive=False,
            prediction_error_readout_only=self._prediction_error_readout_only,
            primary_prediction_error_dominance_enabled=self._primary_prediction_error_dominance_enabled,
        )
        runner._application_rare_heavy_state.import_rare_heavy_state(
            self._application_rare_heavy_state.export_rare_heavy_state(
                checkpoint_id=f"{runner.session_id}:application-seed"
            )
        )
        if artifact is not None:
            runner.apply_rare_heavy_artifact(
                artifact,
                checkpoint_id=f"{runner.session_id}:candidate-import",
            )
        return runner

    def _build_application_rare_heavy_checkpoint(
        self,
        *,
        artifact_id: str,
    ) -> ApplicationRareHeavyCheckpoint:
        domain_snapshot = self._upstream_snapshots.get("domain_knowledge")
        case_snapshot = self._upstream_snapshots.get("case_memory")
        playbook_snapshot = self._upstream_snapshots.get("strategy_playbook")
        domain_biases: list[tuple[str, float]] = []
        if domain_snapshot is not None and isinstance(domain_snapshot.value, DomainKnowledgeSnapshot):
            for domain in domain_snapshot.value.active_domains:
                domain_biases.append((domain, 0.72))
        case_clusters: list[ApplicationCaseCluster] = []
        if case_snapshot is not None and isinstance(case_snapshot.value, CaseMemorySnapshot):
            for index, pattern in enumerate(case_snapshot.value.active_problem_patterns[:3], start=1):
                matched_hits = tuple(hit for hit in case_snapshot.value.hits if hit.problem_pattern == pattern)
                mean_relevance = (
                    sum(hit.relevance_score for hit in matched_hits) / max(len(matched_hits), 1)
                    if matched_hits
                    else 0.5
                )
                risk_markers = matched_hits[0].risk_markers if matched_hits else ()
                case_clusters.append(
                    ApplicationCaseCluster(
                        cluster_id=f"{artifact_id}:cluster:{index}",
                        problem_pattern=pattern,
                        exemplar_count=max(len(matched_hits), 1),
                        mean_relevance=mean_relevance,
                        risk_markers=risk_markers,
                        description=f"Rare-heavy distilled cluster for pattern={pattern}.",
                    )
                )
        distilled_playbook_rules = ()
        if playbook_snapshot is not None and isinstance(playbook_snapshot.value, StrategyPlaybookSnapshot):
            distilled_playbook_rules = playbook_snapshot.value.matched_rules
        if not domain_biases and not case_clusters and not distilled_playbook_rules:
            return self._application_rare_heavy_state.export_rare_heavy_state(
                checkpoint_id=f"{artifact_id}:application"
            )
        return ApplicationRareHeavyCheckpoint(
            checkpoint_id=f"{artifact_id}:application",
            domain_template_biases=tuple(domain_biases),
            case_clusters=tuple(case_clusters),
            distilled_playbook_rules=tuple(distilled_playbook_rules),
            description=(
                f"Application rare-heavy checkpoint distilled {len(domain_biases)} domain biases, "
                f"{len(case_clusters)} case clusters, and {len(distilled_playbook_rules)} playbook rules."
            ),
        )

    @staticmethod
    def _rare_heavy_replay_score(result: AgentTurnResult) -> float:
        evaluation_snapshot = result.active_snapshots.get("evaluation")
        if evaluation_snapshot is None or not isinstance(evaluation_snapshot.value, EvaluationSnapshot):
            return 0.0
        turn_scores = evaluation_snapshot.value.turn_scores
        if not turn_scores:
            return 0.0
        mean_score = sum(score.value for score in turn_scores) / len(turn_scores)
        alert_penalty = min(len(result.evaluation_alerts) * 0.05, 0.20)
        acceptance_bonus = 0.05 if result.acceptance_passed else -0.05
        return max(0.0, min(1.0, mean_score - alert_penalty + acceptance_bonus))

    async def _evaluate_rare_heavy_candidate(
        self,
        *,
        artifact: RareHeavyArtifact,
        bundle: RareHeavyTrainingBundle,
    ) -> RareHeavyPreImportEvaluation:
        replay_examples = bundle.examples[-min(len(bundle.examples), 3) :]
        if not replay_examples:
            return RareHeavyPreImportEvaluation(
                accepted=False,
                case_count=0,
                baseline_mean_score=0.0,
                candidate_mean_score=0.0,
                mean_score_delta=0.0,
                worst_score_delta=0.0,
                positive_fraction=0.0,
                judgement="not-run",
                reasons=("no-replay-examples",),
                description="Rare-heavy pre-import replay skipped because no aligned examples were available.",
            )
        try:
            baseline_runner = self._build_rare_heavy_replay_runner(label="baseline")
            candidate_runner = self._build_rare_heavy_replay_runner(label="candidate", artifact=artifact)
        except (TypeError, ValueError, RuntimeError) as exc:
            return RareHeavyPreImportEvaluation(
                accepted=False,
                case_count=len(replay_examples),
                baseline_mean_score=0.0,
                candidate_mean_score=0.0,
                mean_score_delta=0.0,
                worst_score_delta=0.0,
                positive_fraction=0.0,
                judgement="runner-build-failed",
                reasons=("candidate-runner-build-failed",),
                description=f"Rare-heavy pre-import replay failed while building candidate runners: {exc}",
            )
        baseline_scores: list[float] = []
        candidate_scores: list[float] = []
        for example in replay_examples:
            baseline_result = await baseline_runner.run_turn(example.source_text)
            candidate_result = await candidate_runner.run_turn(example.source_text)
            baseline_scores.append(self._rare_heavy_replay_score(baseline_result))
            candidate_scores.append(self._rare_heavy_replay_score(candidate_result))
        deltas = tuple(
            candidate - baseline
            for baseline, candidate in zip(baseline_scores, candidate_scores, strict=True)
        )
        baseline_mean_score = sum(baseline_scores) / len(baseline_scores)
        candidate_mean_score = sum(candidate_scores) / len(candidate_scores)
        mean_score_delta = sum(deltas) / len(deltas)
        worst_score_delta = min(deltas, default=0.0)
        positive_fraction = (
            sum(1 for delta in deltas if delta > 0.0) / len(deltas)
            if deltas
            else 0.0
        )
        session_report = candidate_runner.build_current_session_report()
        judgement_label = "not-run"
        reasons: list[str] = []
        if session_report is not None:
            replay_suite = candidate_runner.evaluation_backbone.run_default_evolution_benchmark(
                timestamp_ms=max(candidate_runner.turn_index, 1) + 1,
            )
            judgement = candidate_runner.evaluation_backbone.judge_evolution_candidate(
                replay_suite_result=replay_suite,
                session_report=session_report,
            )
            judgement_label = judgement.decision.value
            if judgement.decision is EvolutionDecision.ROLLBACK:
                reasons.append("evolution-judge-rollback")
        if mean_score_delta <= 0.0:
            reasons.append("pre-import-mean-score-nonpositive")
        if worst_score_delta < -0.05:
            reasons.append("pre-import-worst-case-regressed")
        if positive_fraction < 0.5:
            reasons.append("pre-import-positive-fraction-too-low")
        accepted = not reasons
        return RareHeavyPreImportEvaluation(
            accepted=accepted,
            case_count=len(replay_examples),
            baseline_mean_score=baseline_mean_score,
            candidate_mean_score=candidate_mean_score,
            mean_score_delta=mean_score_delta,
            worst_score_delta=worst_score_delta,
            positive_fraction=positive_fraction,
            judgement=judgement_label,
            reasons=tuple(reasons),
            description=(
                f"Rare-heavy pre-import replay over {len(replay_examples)} cases produced "
                f"baseline_mean={baseline_mean_score:.3f}, candidate_mean={candidate_mean_score:.3f}, "
                f"mean_delta={mean_score_delta:.3f}, worst_delta={worst_score_delta:.3f}, "
                f"positive_fraction={positive_fraction:.3f}, judgement={judgement_label}."
            ),
        )

    def _append_online_fast_credit_audit(
        self,
        *,
        integration_result: FinalIntegrationResult,
        record: SelfModificationRecord,
    ) -> None:
        credit_snapshot = integration_result.active_snapshots.get("credit")
        if credit_snapshot is None:
            return
        extended = extend_credit_snapshot(
            credit_snapshot=credit_snapshot.value,
            extra_modifications=(record,),
        )
        integration_result.active_snapshots["credit"] = Snapshot(
            slot_name="credit",
            owner="CreditModule",
            version=credit_snapshot.version + 1,
            timestamp_ms=max(credit_snapshot.timestamp_ms + 1, self._turn_index),
            value=extended,
        )

    def _append_online_fast_evaluation_evidence(
        self,
        *,
        integration_result: FinalIntegrationResult,
        wave_id: str,
        result: OnlineFastSubstrateTurnResult,
    ) -> None:
        evaluation_snapshot = integration_result.active_snapshots.get("evaluation")
        if evaluation_snapshot is None or not isinstance(evaluation_snapshot.value, EvaluationSnapshot):
            return
        enriched = self._evaluation_backbone.record_external_scores(
            session_id=self.active_context_session_id,
            wave_id=wave_id,
            timestamp_ms=max(evaluation_snapshot.timestamp_ms + 1, self._turn_index),
            base_snapshot=evaluation_snapshot.value,
            scores=(
                EvaluationScore(
                    family="learning",
                    metric_name="substrate_online_fast_proposed",
                    value=1.0 if result.recommended else 0.0,
                    confidence=0.8,
                    evidence=result.description,
                ),
                EvaluationScore(
                    family="learning",
                    metric_name="substrate_online_fast_applied",
                    value=1.0 if result.applied else 0.0,
                    confidence=0.8,
                    evidence=result.description,
                ),
                EvaluationScore(
                    family="learning",
                    metric_name="substrate_online_fast_optimizer_norm",
                    value=result.optimizer_state_norm,
                    confidence=0.7,
                    evidence=result.optimizer_state_description,
                ),
            ),
            description_suffix="Session owner appended online-fast substrate apply evidence.",
        )
        integration_result.active_snapshots["evaluation"] = Snapshot(
            slot_name="evaluation",
            owner="EvaluationModule",
            version=evaluation_snapshot.version + 1,
            timestamp_ms=max(evaluation_snapshot.timestamp_ms + 1, self._turn_index),
            value=enriched,
        )

    def _maybe_apply_online_fast_substrate_self_mod(
        self,
        *,
        wave_id: str,
        joint_result: ScheduledJointLoopResult,
        integration_result: FinalIntegrationResult,
    ) -> OnlineFastSubstrateTurnResult | None:
        snapshot = integration_result.active_snapshots.get("substrate_self_mod") or integration_result.shadow_snapshots.get(
            "substrate_self_mod"
        )
        if snapshot is None or not isinstance(snapshot.value, SubstrateSelfModSnapshot):
            return None
        substrate_self_mod = snapshot.value
        if not substrate_self_mod.recommended or substrate_self_mod.checkpoint is None:
            return None
        evaluation_snapshot = integration_result.active_snapshots.get("evaluation")
        evaluation_value = (
            evaluation_snapshot.value
            if evaluation_snapshot is not None and isinstance(evaluation_snapshot.value, EvaluationSnapshot)
            else None
        )
        gate_decision = GateDecision.BLOCK
        if evaluation_value is not None:
            gate_decision = evaluate_gate(
                proposal=ModificationProposal(
                    target=substrate_self_mod.target,
                    desired_gate=ModificationGate.ONLINE,
                    old_value_hash="substrate.online_fast:pre",
                    new_value_hash=substrate_self_mod.checkpoint_hash,
                    justification=substrate_self_mod.description,
                ),
                evaluation_snapshot=evaluation_value,
            )
        if not joint_result.substrate_online_fast_due:
            return OnlineFastSubstrateTurnResult(
                recommended=True,
                applied=False,
                gate_decision="schedule-not-due",
                applied_operations=(),
                blocked_operations=("online-fast:schedule-not-due",),
                parameter_change_rate=substrate_self_mod.parameter_change_rate,
                optimizer_state_norm=substrate_self_mod.optimizer_state_norm,
                checkpoint_id=substrate_self_mod.checkpoint.checkpoint_id,
                fast_state_hash=substrate_self_mod.checkpoint.fast_state_hash,
                source_fast_state_hash=substrate_self_mod.checkpoint.source_fast_state_hash,
                optimizer_state_description=substrate_self_mod.checkpoint.optimizer_state_description,
                fast_memory_signal=substrate_self_mod.checkpoint.fast_memory_signal,
                description="Online-fast substrate self-mod proposal was present, but schedule was not due.",
            )
        if gate_decision is GateDecision.BLOCK:
            self._append_online_fast_credit_audit(
                integration_result=integration_result,
                record=SelfModificationRecord(
                    target=substrate_self_mod.target,
                    gate=ModificationGate.ONLINE,
                    decision=GateDecision.BLOCK,
                    old_value_hash="substrate.online_fast:pre",
                    new_value_hash=substrate_self_mod.checkpoint_hash,
                    justification=substrate_self_mod.description,
                    timestamp_ms=self._turn_index,
                    is_reversible=True,
                    checkpoint_id=substrate_self_mod.checkpoint.checkpoint_id,
                    lineage_hash=substrate_self_mod.checkpoint.fast_state_hash,
                    proposal_hash=substrate_self_mod.checkpoint_hash,
                ),
            )
            blocked_result = OnlineFastSubstrateTurnResult(
                recommended=True,
                applied=False,
                gate_decision=gate_decision.value,
                applied_operations=(),
                blocked_operations=("online-fast:evaluation-gate-block",),
                parameter_change_rate=substrate_self_mod.parameter_change_rate,
                optimizer_state_norm=substrate_self_mod.optimizer_state_norm,
                checkpoint_id=substrate_self_mod.checkpoint.checkpoint_id,
                fast_state_hash=substrate_self_mod.checkpoint.fast_state_hash,
                source_fast_state_hash=substrate_self_mod.checkpoint.source_fast_state_hash,
                optimizer_state_description=substrate_self_mod.checkpoint.optimizer_state_description,
                fast_memory_signal=substrate_self_mod.checkpoint.fast_memory_signal,
                description="Online-fast substrate self-mod proposal was blocked by the ONLINE evaluation gate.",
            )
            self._append_online_fast_evaluation_evidence(
                integration_result=integration_result,
                wave_id=wave_id,
                result=blocked_result,
            )
            return blocked_result
        if not self.residual_runtime.supports_live_substrate_mutation:
            self._append_online_fast_credit_audit(
                integration_result=integration_result,
                record=SelfModificationRecord(
                    target=substrate_self_mod.target,
                    gate=ModificationGate.ONLINE,
                    decision=GateDecision.BLOCK,
                    old_value_hash="substrate.online_fast:pre",
                    new_value_hash=substrate_self_mod.checkpoint_hash,
                    justification=(
                        "Frozen-substrate doctrine kept the online-fast substrate proposal in review-only mode. "
                        f"{substrate_self_mod.description}"
                    ),
                    timestamp_ms=self._turn_index,
                    is_reversible=True,
                    checkpoint_id=substrate_self_mod.checkpoint.checkpoint_id,
                    lineage_hash=substrate_self_mod.checkpoint.fast_state_hash,
                    proposal_hash=substrate_self_mod.checkpoint_hash,
                ),
            )
            blocked_result = OnlineFastSubstrateTurnResult(
                recommended=True,
                applied=False,
                gate_decision="frozen-substrate-doctrine",
                applied_operations=(),
                blocked_operations=("online-fast:frozen-substrate-doctrine",),
                parameter_change_rate=substrate_self_mod.parameter_change_rate,
                optimizer_state_norm=substrate_self_mod.optimizer_state_norm,
                checkpoint_id=substrate_self_mod.checkpoint.checkpoint_id,
                fast_state_hash=substrate_self_mod.checkpoint.fast_state_hash,
                source_fast_state_hash=substrate_self_mod.checkpoint.source_fast_state_hash,
                optimizer_state_description=substrate_self_mod.checkpoint.optimizer_state_description,
                fast_memory_signal=substrate_self_mod.checkpoint.fast_memory_signal,
                description=(
                    "Online-fast substrate self-mod proposal stayed review-only because the live runtime "
                    "is operating under the frozen-substrate doctrine."
                ),
            )
            self._append_online_fast_evaluation_evidence(
                integration_result=integration_result,
                wave_id=wave_id,
                result=blocked_result,
            )
            return blocked_result
        import_result = self._joint_loop.apply_online_fast_substrate_checkpoint(
            substrate_self_mod.checkpoint,
            checkpoint_id=f"{self._session_id}:{wave_id}:online-fast-substrate",
        )
        prior_checkpoint = import_result.checkpoint.substrate_checkpoint
        if substrate_self_mod.checkpoint.fast_memory_signal:
            self._memory_store.observe_fast_memory_signal(
                signal=substrate_self_mod.checkpoint.fast_memory_signal,
                timestamp_ms=max(self._turn_index, 1),
            )
        self._append_online_fast_credit_audit(
            integration_result=integration_result,
            record=SelfModificationRecord(
                target=substrate_self_mod.target,
                gate=ModificationGate.ONLINE,
                decision=GateDecision.ALLOW,
                old_value_hash=prior_checkpoint.checkpoint_id if prior_checkpoint is not None else "none",
                new_value_hash=substrate_self_mod.checkpoint.checkpoint_id,
                justification=substrate_self_mod.description,
                timestamp_ms=self._turn_index,
                is_reversible=True,
                checkpoint_id=substrate_self_mod.checkpoint.checkpoint_id,
                lineage_hash=substrate_self_mod.checkpoint.fast_state_hash,
                proposal_hash=substrate_self_mod.checkpoint_hash,
            ),
        )
        applied_result = OnlineFastSubstrateTurnResult(
            recommended=True,
            applied=True,
            gate_decision=gate_decision.value,
            applied_operations=import_result.applied_operations,
            blocked_operations=(),
            parameter_change_rate=substrate_self_mod.parameter_change_rate,
            optimizer_state_norm=substrate_self_mod.optimizer_state_norm,
            checkpoint_id=substrate_self_mod.checkpoint.checkpoint_id,
            fast_state_hash=substrate_self_mod.checkpoint.fast_state_hash,
            source_fast_state_hash=substrate_self_mod.checkpoint.source_fast_state_hash,
            optimizer_state_description=substrate_self_mod.checkpoint.optimizer_state_description,
            fast_memory_signal=substrate_self_mod.checkpoint.fast_memory_signal,
            description=import_result.description,
        )
        self._append_online_fast_evaluation_evidence(
            integration_result=integration_result,
            wave_id=wave_id,
            result=applied_result,
        )
        return applied_result

    async def _maybe_apply_rare_heavy(
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
                import_decision="disabled",
                reject_reason="rare-heavy-disabled",
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
                import_decision="skipped-cooldown",
                reject_reason="cooldown-active",
            )
        bundle = self._build_rare_heavy_training_bundle()
        if bundle.aligned_example_count < self._rare_heavy_min_traces:
            return RareHeavyTurnResult(
                recommended=True,
                applied=False,
                artifact_id=None,
                applied_operations=(),
                substrate_status="skipped",
                substrate_training_mode="not-run",
                description=(
                    f"Rare-heavy review was recommended, but only {bundle.aligned_example_count} aligned examples are available; "
                    f"need {self._rare_heavy_min_traces}. {bundle.description}"
                ),
                import_decision="skipped-insufficient-alignment",
                reject_reason="insufficient-aligned-examples",
                bundle_alignment_ratio=bundle.alignment_ratio,
                bundle_trace_count=bundle.trace_count,
                bundle_substrate_batch_count=bundle.substrate_batch_count,
                bundle_mean_trace_step_count=bundle.mean_trace_step_count,
                bundle_mean_sequence_length=bundle.mean_sequence_length,
                bundle_mean_residual_magnitude=bundle.mean_residual_magnitude,
            )
        world_pipeline = self._build_rare_heavy_pipeline(
            source_policy=self._joint_loop.world_temporal_policy
        )
        self_pipeline = self._build_rare_heavy_pipeline(
            source_policy=self._joint_loop.self_temporal_policy
        )
        traces = tuple(example.trace for example in bundle.examples)
        substrate_batches = tuple(example.substrate_batch for example in bundle.examples)
        try:
            world_pipeline_result = world_pipeline.run_pipeline(
                traces=traces,
                substrate_steps_per_trace=substrate_batches if substrate_batches else None,
            )
            self_pipeline_result = self_pipeline.run_pipeline(
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
                import_decision="pipeline-failed-closed",
                reject_reason="pipeline-failed-closed",
                bundle_alignment_ratio=bundle.alignment_ratio,
                bundle_trace_count=bundle.trace_count,
                bundle_substrate_batch_count=bundle.substrate_batch_count,
                bundle_mean_trace_step_count=bundle.mean_trace_step_count,
                bundle_mean_sequence_length=bundle.mean_sequence_length,
                bundle_mean_residual_magnitude=bundle.mean_residual_magnitude,
            )
        world_artifact = world_pipeline.export_rare_heavy_artifact(
            artifact_id=f"{self._session_id}:{wave_id}:rare-heavy:world"
        )
        self_artifact = self_pipeline.export_rare_heavy_artifact(
            artifact_id=f"{self._session_id}:{wave_id}:rare-heavy:self"
        )
        artifact = RareHeavyArtifact(
            artifact_id=f"{self._session_id}:{wave_id}:rare-heavy",
            owner_path=world_artifact.owner_path,
            created_at_ms=world_artifact.created_at_ms,
            temporal_snapshot=DualTrackRareHeavySnapshot(
                world_snapshot=world_artifact.temporal_snapshot,
                self_snapshot=self_artifact.temporal_snapshot,
                description=(
                    f"Dual rare-heavy snapshot world={world_artifact.artifact_id} "
                    f"self={self_artifact.artifact_id}."
                ),
            ),
            memory_checkpoint=world_artifact.memory_checkpoint,
            substrate_checkpoint=world_artifact.substrate_checkpoint,
            transition_step=max(world_artifact.transition_step, self_artifact.transition_step),
            final_ssl_loss=(world_artifact.final_ssl_loss + self_artifact.final_ssl_loss) / 2.0,
            final_total_reward=(world_artifact.final_total_reward + self_artifact.final_total_reward) / 2.0,
            description=(
                f"Dual-track rare-heavy artifact world={world_artifact.artifact_id} "
                f"self={self_artifact.artifact_id}."
            ),
            training_evidence=world_artifact.training_evidence,
            application_checkpoint=self._build_application_rare_heavy_checkpoint(
                artifact_id=f"{self._session_id}:{wave_id}:rare-heavy"
            ),
        )
        combined_rl_steps = (
            world_pipeline_result.rl_steps_completed + self_pipeline_result.rl_steps_completed
        )
        combined_substrate_mode = (
            world_pipeline_result.substrate_training_mode
            if world_pipeline_result.substrate_training_mode == self_pipeline_result.substrate_training_mode
            else f"{world_pipeline_result.substrate_training_mode}+{self_pipeline_result.substrate_training_mode}"
        )
        if combined_rl_steps <= 0:
            return RareHeavyTurnResult(
                recommended=True,
                applied=False,
                artifact_id=artifact.artifact_id,
                applied_operations=(),
                substrate_status="skipped",
                substrate_training_mode=combined_substrate_mode,
                description=(
                    f"Rare-heavy pipeline exported {artifact.artifact_id}, but no offline RL steps completed; "
                    f"skipping import. world={world_pipeline_result.description} self={self_pipeline_result.description}"
                ),
                import_decision="skipped-no-offline-rl",
                reject_reason="no-offline-rl-steps",
                bundle_alignment_ratio=bundle.alignment_ratio,
                bundle_trace_count=bundle.trace_count,
                bundle_substrate_batch_count=bundle.substrate_batch_count,
                bundle_mean_trace_step_count=bundle.mean_trace_step_count,
                bundle_mean_sequence_length=bundle.mean_sequence_length,
                bundle_mean_residual_magnitude=bundle.mean_residual_magnitude,
                candidate_adapter_parameter_count=(
                    artifact.substrate_checkpoint.adapter_parameter_count
                    if artifact.substrate_checkpoint is not None
                    else 0
                ),
                candidate_adapter_training_loss=(
                    artifact.substrate_checkpoint.adapter_training_loss
                    if artifact.substrate_checkpoint is not None
                    else 0.0
                ),
            )
        pre_import_evaluation = await self._evaluate_rare_heavy_candidate(
            artifact=artifact,
            bundle=bundle,
        )
        if not pre_import_evaluation.accepted:
            return RareHeavyTurnResult(
                recommended=True,
                applied=False,
                artifact_id=artifact.artifact_id,
                applied_operations=(),
                substrate_status="rejected",
                substrate_training_mode=combined_substrate_mode,
                description=(
                    f"{world_pipeline_result.description} {self_pipeline_result.description} "
                    f"{pre_import_evaluation.description}"
                ),
                import_decision="rejected-pre-import",
                reject_reason=",".join(pre_import_evaluation.reasons),
                bundle_alignment_ratio=bundle.alignment_ratio,
                bundle_trace_count=bundle.trace_count,
                bundle_substrate_batch_count=bundle.substrate_batch_count,
                bundle_mean_trace_step_count=bundle.mean_trace_step_count,
                bundle_mean_sequence_length=bundle.mean_sequence_length,
                bundle_mean_residual_magnitude=bundle.mean_residual_magnitude,
                candidate_adapter_parameter_count=(
                    artifact.substrate_checkpoint.adapter_parameter_count
                    if artifact.substrate_checkpoint is not None
                    else 0
                ),
                candidate_adapter_training_loss=(
                    artifact.substrate_checkpoint.adapter_training_loss
                    if artifact.substrate_checkpoint is not None
                    else 0.0
                ),
                pre_import_case_count=pre_import_evaluation.case_count,
                pre_import_mean_score_delta=pre_import_evaluation.mean_score_delta,
                pre_import_worst_score_delta=pre_import_evaluation.worst_score_delta,
                pre_import_positive_fraction=pre_import_evaluation.positive_fraction,
                pre_import_passed=pre_import_evaluation.accepted,
                pre_import_judgement=pre_import_evaluation.judgement,
            )
        if not self.residual_runtime.supports_live_substrate_mutation:
            return RareHeavyTurnResult(
                recommended=True,
                applied=False,
                artifact_id=artifact.artifact_id,
                applied_operations=(),
                substrate_status="review-only",
                substrate_training_mode=combined_substrate_mode,
                description=(
                    f"{world_pipeline_result.description} {self_pipeline_result.description} "
                    "Rare-heavy candidate stayed review-only because the live runtime is enforcing the "
                    "frozen-substrate doctrine."
                ),
                import_decision="blocked-by-doctrine",
                reject_reason="frozen-substrate-doctrine",
                bundle_alignment_ratio=bundle.alignment_ratio,
                bundle_trace_count=bundle.trace_count,
                bundle_substrate_batch_count=bundle.substrate_batch_count,
                bundle_mean_trace_step_count=bundle.mean_trace_step_count,
                bundle_mean_sequence_length=bundle.mean_sequence_length,
                bundle_mean_residual_magnitude=bundle.mean_residual_magnitude,
                candidate_adapter_parameter_count=(
                    artifact.substrate_checkpoint.adapter_parameter_count
                    if artifact.substrate_checkpoint is not None
                    else 0
                ),
                candidate_adapter_training_loss=(
                    artifact.substrate_checkpoint.adapter_training_loss
                    if artifact.substrate_checkpoint is not None
                    else 0.0
                ),
                pre_import_case_count=pre_import_evaluation.case_count,
                pre_import_mean_score_delta=pre_import_evaluation.mean_score_delta,
                pre_import_worst_score_delta=pre_import_evaluation.worst_score_delta,
                pre_import_positive_fraction=pre_import_evaluation.positive_fraction,
                pre_import_passed=pre_import_evaluation.accepted,
                pre_import_judgement=pre_import_evaluation.judgement,
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
                substrate_training_mode=combined_substrate_mode,
                description=(
                    f"{world_pipeline_result.description} {self_pipeline_result.description} "
                    f"Rare-heavy import failed closed: {exc}"
                ),
                import_decision="import-failed-closed",
                reject_reason="import-failed-closed",
                bundle_alignment_ratio=bundle.alignment_ratio,
                bundle_trace_count=bundle.trace_count,
                bundle_substrate_batch_count=bundle.substrate_batch_count,
                bundle_mean_trace_step_count=bundle.mean_trace_step_count,
                bundle_mean_sequence_length=bundle.mean_sequence_length,
                bundle_mean_residual_magnitude=bundle.mean_residual_magnitude,
                candidate_adapter_parameter_count=(
                    artifact.substrate_checkpoint.adapter_parameter_count
                    if artifact.substrate_checkpoint is not None
                    else 0
                ),
                candidate_adapter_training_loss=(
                    artifact.substrate_checkpoint.adapter_training_loss
                    if artifact.substrate_checkpoint is not None
                    else 0.0
                ),
                pre_import_case_count=pre_import_evaluation.case_count,
                pre_import_mean_score_delta=pre_import_evaluation.mean_score_delta,
                pre_import_worst_score_delta=pre_import_evaluation.worst_score_delta,
                pre_import_positive_fraction=pre_import_evaluation.positive_fraction,
                pre_import_passed=pre_import_evaluation.accepted,
                pre_import_judgement=pre_import_evaluation.judgement,
            )
        self._last_rare_heavy_turn_index = self._turn_index
        return RareHeavyTurnResult(
            recommended=True,
            applied=True,
            artifact_id=artifact.artifact_id,
            applied_operations=import_result.applied_operations,
            substrate_status="imported",
            substrate_training_mode=combined_substrate_mode,
            description=(
                f"{world_pipeline_result.description} {self_pipeline_result.description} "
                f"{import_result.description}"
            ),
            import_decision="imported",
            bundle_alignment_ratio=bundle.alignment_ratio,
            bundle_trace_count=bundle.trace_count,
            bundle_substrate_batch_count=bundle.substrate_batch_count,
            bundle_mean_trace_step_count=bundle.mean_trace_step_count,
            bundle_mean_sequence_length=bundle.mean_sequence_length,
            bundle_mean_residual_magnitude=bundle.mean_residual_magnitude,
            candidate_adapter_parameter_count=(
                artifact.substrate_checkpoint.adapter_parameter_count
                if artifact.substrate_checkpoint is not None
                else 0
            ),
            candidate_adapter_training_loss=(
                artifact.substrate_checkpoint.adapter_training_loss
                if artifact.substrate_checkpoint is not None
                else 0.0
            ),
            pre_import_case_count=pre_import_evaluation.case_count,
            pre_import_mean_score_delta=pre_import_evaluation.mean_score_delta,
            pre_import_worst_score_delta=pre_import_evaluation.worst_score_delta,
            pre_import_positive_fraction=pre_import_evaluation.positive_fraction,
            pre_import_passed=pre_import_evaluation.accepted,
            pre_import_judgement=pre_import_evaluation.judgement,
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
        online_fast_substrate_result: OnlineFastSubstrateTurnResult | None = None,
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
        tower_consolidation_count = 0
        learned_recall_count = 0
        learned_recall_confidence = 0.0
        learned_recall_core_guided = False
        memory_tower_depth = 0
        memory_tower_alignment = 0.0
        memory_tower_profile_id = ""
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
            tower_consolidation_count = int(lifecycle_metrics.get("tower_consolidation_count", 0.0))
            learned_recall_count = int(lifecycle_metrics.get("learned_recall_count", 0.0))
            learned_recall_confidence = lifecycle_metrics.get("last_learned_recall_confidence", 0.0)
            learned_recall_core_guided = lifecycle_metrics.get("last_learned_recall_driver_is_core", 0.0) > 0.0
            memory_tower_depth = int(lifecycle_metrics.get("last_memory_tower_depth", 0.0))
            memory_tower_alignment = lifecycle_metrics.get("last_memory_tower_alignment", 0.0)
            cms_state = memory_snapshot.value.cms_state
            if cms_state is not None and cms_state.tower_profile is not None:
                memory_tower_profile_id = cms_state.tower_profile.profile_id

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
        knowledge_hit_count = 0
        knowledge_summaries: tuple[str, ...] = ()
        case_hit_count = 0
        case_patterns: tuple[str, ...] = ()
        playbook_rule_count = 0
        playbook_ordering_hints: tuple[str, ...] = ()
        citation_required = False
        boundary_risk_band = "low"
        boundary_answer_depth_limit = "standard"
        boundary_clarification_required = False
        boundary_refer_out_required = False
        boundary_required_disclaimers: tuple[str, ...] = ()
        domain_knowledge_snapshot = integration_result.active_snapshots.get("domain_knowledge")
        if domain_knowledge_snapshot is not None and isinstance(domain_knowledge_snapshot.value, DomainKnowledgeSnapshot):
            knowledge_hit_count = len(domain_knowledge_snapshot.value.hits)
            knowledge_summaries = tuple(hit.summary for hit in domain_knowledge_snapshot.value.hits[:3])
            citation_required = domain_knowledge_snapshot.value.citation_required
        case_memory_snapshot = integration_result.active_snapshots.get("case_memory")
        if case_memory_snapshot is not None and isinstance(case_memory_snapshot.value, CaseMemorySnapshot):
            case_hit_count = len(case_memory_snapshot.value.hits)
            case_patterns = tuple(case.problem_pattern for case in case_memory_snapshot.value.hits[:3])
        strategy_playbook_snapshot = integration_result.active_snapshots.get("strategy_playbook")
        if strategy_playbook_snapshot is not None and isinstance(strategy_playbook_snapshot.value, StrategyPlaybookSnapshot):
            playbook_rule_count = len(strategy_playbook_snapshot.value.matched_rules)
            playbook_ordering_hints = tuple(
                step
                for rule in strategy_playbook_snapshot.value.matched_rules[:2]
                for step in rule.recommended_ordering[:3]
            )
        boundary_policy_snapshot = integration_result.active_snapshots.get("boundary_policy")
        if boundary_policy_snapshot is not None and isinstance(boundary_policy_snapshot.value, BoundaryPolicySnapshot):
            decision = boundary_policy_snapshot.value.active_decision
            citation_required = citation_required or decision.citation_required
            boundary_risk_band = decision.risk_band.value
            boundary_answer_depth_limit = decision.answer_depth_limit
            boundary_clarification_required = decision.clarification_required
            boundary_refer_out_required = decision.refer_out_required
            boundary_required_disclaimers = decision.required_disclaimers

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
                knowledge_hit_count=knowledge_hit_count,
                knowledge_summaries=knowledge_summaries,
                case_hit_count=case_hit_count,
                case_patterns=case_patterns,
                playbook_rule_count=playbook_rule_count,
                playbook_ordering_hints=playbook_ordering_hints,
                citation_required=citation_required,
                boundary_risk_band=boundary_risk_band,
                boundary_answer_depth_limit=boundary_answer_depth_limit,
                boundary_clarification_required=boundary_clarification_required,
                boundary_refer_out_required=boundary_refer_out_required,
                boundary_required_disclaimers=boundary_required_disclaimers,
            )
        )
        effective_writeback_result = deferred_writeback_result or integration_result.writeback_result
        effective_queue_state = queue_state or self.session_post_queue_state
        session_post_snapshot = self._publish_session_post_snapshot()
        experience_consolidation_snapshot = self._publish_experience_consolidation_snapshot()
        active_snapshots = dict(integration_result.active_snapshots)
        shadow_snapshots = dict(integration_result.shadow_snapshots)
        if self._config.is_active("session_post_slow_loop"):
            active_snapshots["session_post_slow_loop"] = session_post_snapshot
        else:
            shadow_snapshots["session_post_slow_loop"] = session_post_snapshot
        if self._config.is_active("experience_consolidation"):
            active_snapshots["experience_consolidation"] = experience_consolidation_snapshot
        else:
            shadow_snapshots["experience_consolidation"] = experience_consolidation_snapshot

        return AgentTurnResult(
            session_id=self.active_context_session_id,
            wave_id=wave_id,
            user_input=user_input,
            active_snapshots=active_snapshots,
            shadow_snapshots=shadow_snapshots,
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
            tower_consolidation_count=tower_consolidation_count,
            learned_recall_count=learned_recall_count,
            learned_recall_confidence=learned_recall_confidence,
            learned_recall_core_guided=learned_recall_core_guided,
            memory_tower_depth=memory_tower_depth,
            memory_tower_alignment=memory_tower_alignment,
            memory_tower_profile_id=memory_tower_profile_id,
            session_post_pending_job_count=effective_queue_state.pending_job_count,
            session_post_completed_job_count=effective_queue_state.completed_job_count,
            session_post_last_completed_job_id=effective_queue_state.last_completed_job_id,
            online_fast_substrate_result=online_fast_substrate_result,
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
