from __future__ import annotations

from collections.abc import Callable
import asyncio
from dataclasses import dataclass, replace
import os
from typing import Any

from volvence_zero.dialogue_trace import (
    DialogueActionTrace,
    DialogueOutcomeResolution,
    DialogueTraceSnapshot,
)
from volvence_zero.application.runtime import (
    ApplicationPriorUpdate,
    ApplicationPriorWritebackReport,
    ApplicationOutcomeAttribution,
    ApplicationSequencePayoff,
    ApplicationCaseCluster,
    DelayedCreditSummary,
    ApplicationRareHeavyCheckpoint,
    ApplicationRareHeavyState,
    BoundaryPolicySnapshot,
    CaseMemorySnapshot,
    DomainKnowledgeSnapshot,
    ExperienceConsolidationModule,
    ExperienceConsolidationSnapshot,
    ExperienceDelta,
    ExperienceFastPriorModule,
    ExperienceFastPriorSnapshot,
    KnowledgeHit,
    ResponseAssemblySnapshot,
    RetrievalPolicySnapshot,
    StrategyPlaybookSnapshot,
)
from volvence_zero.application.storage import (
    ApplicationCaseMemoryStore,
    ApplicationDomainKnowledgeStore,
    ProvisionalReconcileResult,
    ProvisionalReconcileThresholds,
    build_default_case_memory_store,
    build_default_domain_knowledge_store,
    build_filesystem_persistence_backend,
)
from volvence_zero.application.domain_experience import (
    DomainExperiencePackage,
    apply_domain_experience_packages,
)
from volvence_zero.application.experience_layers import (
    ApplicationPriorProposalBuilder,
    ApplicationPriorProposalInputs,
)
from volvence_zero.application.knowledge_channels import (
    build_conversation_knowledge_candidates,
    domain_knowledge_prior_updates_from_reviewed,
)
from volvence_zero.agent.response import AgentResponse, LLMResponseSynthesizer, ResponseContext, ResponseSynthesizer
from volvence_zero.agent.dialogue_trace import DialogueTraceStore
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
from volvence_zero.environment import (
    EnvironmentEvent,
    build_user_input_environment_event,
)
from volvence_zero.integration import (
    _apply_application_prior_writeback,
    FinalIntegrationResult,
    FinalRolloutConfig,
    SessionPostWritebackRequest,
    apply_session_post_writeback_request,
    run_final_wiring_turn,
)
from volvence_zero.joint_loop import (
    DefaultContinualLearningSurface,
    ETANLJointLoop,
    JointCycleReport,
    JointLoopSchedule,
    OnlineFastImportCheckpoint,
    PipelineConfig,
    RareHeavyArtifact,
    RareHeavyImportCheckpoint,
    RareHeavyImportResult,
    SSLRLTrainingPipeline,
    ScheduledJointLoopResult,
    OnlineFastImportResult,
)
from volvence_zero.memory import MemorySnapshot, MemoryStore, Track, build_default_memory_store
from volvence_zero.planning import ImaginationResult, imagine
from volvence_zero.prediction.error import (
    ActualOutcome,
    PredictedOutcome,
    PredictionError,
    PredictionErrorModule,
)
from volvence_zero.reflection import ReflectionSnapshot, WritebackMode, WritebackResult
from volvence_zero.regime import RegimeBootstrap, RegimeModule, RegimeSnapshot
from volvence_zero.runtime import Snapshot, WiringLevel
from volvence_zero.semantic_state import (
    AdapterSemanticProposalRuntime,
    ExternalSemanticEvent,
    ExternalSemanticEventBatch,
    NoOpSemanticProposalRuntime,
    SemanticProposalRuntime,
    SemanticStateStore,
)
from volvence_zero.social_cognition import (
    PRIMARY_INTERLOCUTOR_ID,
    SELF_INTERLOCUTOR_ID,
    MultiPartyIdentitySnapshot,
)
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
    MetacontrollerParameterSnapshot,
    MetacontrollerParameterStore,
    MetacontrollerRuntimeState,
    TemporalAbstractionSnapshot,
    TemporalPolicy,
    clone_full_learned_temporal_policy,
    resolve_temporal_bootstrap_snapshot,
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
    default_continual_learning_surface: DefaultContinualLearningSurface | None
    response: AgentResponse
    event_count: int
    environment_event_id: str = ""
    environment_event_kind: str = ""
    environment_trigger_kind: str = ""
    dialogue_trace: DialogueActionTrace | None = None
    dialogue_outcome_resolution: DialogueOutcomeResolution | None = None
    dialogue_trace_snapshot: DialogueTraceSnapshot | None = None
    active_speaker_id: str = PRIMARY_INTERLOCUTOR_ID
    addressee_ids: tuple[str, ...] = (SELF_INTERLOCUTOR_ID,)
    subject_ids: tuple[str, ...] = (PRIMARY_INTERLOCUTOR_ID,)
    audience_ids: tuple[str, ...] = (SELF_INTERLOCUTOR_ID,)
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
    runtime_backbone_evidence_active: bool = False
    runtime_backbone_signal_norm: float = 0.0
    runtime_backbone_signal_quality: float = 0.0
    runtime_backbone_signal_strength: float = 0.0
    runtime_backbone_hook_coverage: float = 0.0
    fast_memory_signal_norm: float = 0.0
    fast_memory_runtime_alignment: float = 0.0
    session_post_pending_job_count: int = 0
    session_post_completed_job_count: int = 0
    session_post_last_completed_job_id: str | None = None
    online_fast_substrate_result: "OnlineFastSubstrateTurnResult | None" = None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _application_outcome_score(
    *,
    reward: float,
    magnitude: float,
    relationship_error: float,
) -> float:
    magnitude_term = 1.0 - min(max(magnitude, 0.0) / 4.0, 1.0)
    relationship_term = 1.0 - min(abs(relationship_error), 1.0)
    return _clamp(0.45 + reward * 0.30 + magnitude_term * 0.15 + relationship_term * 0.10)


def _retrieval_mix_alignment(
    *,
    regime_id: str | None,
    knowledge_weight: float,
    experience_weight: float,
) -> float:
    if regime_id == "problem_solving":
        return _clamp(0.5 + (knowledge_weight - experience_weight) * 0.5)
    if regime_id in {"emotional_support", "repair_and_deescalation"}:
        return _clamp(0.5 + (experience_weight - knowledge_weight) * 0.5)
    return _clamp(1.0 - abs(knowledge_weight - experience_weight))


def _regime_alignment(
    *,
    regime_id: str | None,
    outcome_score: float,
    relationship_error: float,
    regime_error: float,
    magnitude: float,
) -> float:
    if regime_id in {"emotional_support", "repair_and_deescalation"}:
        contextual_fit = 1.0 - min(abs(relationship_error), 1.0)
    elif regime_id == "problem_solving":
        contextual_fit = 1.0 - min(abs(regime_error), 1.0)
    else:
        contextual_fit = 1.0 - min(max(magnitude, 0.0) / 4.0, 1.0)
    return _clamp(outcome_score * 0.65 + contextual_fit * 0.35)


def _abstract_action_alignment(
    *,
    regime_id: str | None,
    abstract_action: str | None,
    action_family_version: int,
    outcome_score: float,
) -> float:
    if abstract_action is None:
        family_bonus = min(max(action_family_version, 0), 4) / 4.0
        return _clamp(outcome_score * 0.78 + family_bonus * 0.22)
    action_label = abstract_action.lower()
    if action_label.startswith("latent-family-v"):
        family_bonus = min(max(action_family_version, 0), 4) / 4.0
        return _clamp(outcome_score * 0.72 + (0.5 + family_bonus * 0.5) * 0.28)
    if regime_id == "problem_solving":
        action_bias = 1.0 if "task_controller" in action_label else 0.45
    elif regime_id == "repair_and_deescalation":
        action_bias = 1.0 if "repair_controller" in action_label else 0.45
    elif regime_id == "emotional_support":
        action_bias = 1.0 if "stabilize_controller" in action_label else 0.45
    elif regime_id == "guided_exploration":
        action_bias = 1.0 if "exploration_controller" in action_label else 0.45
    else:
        action_bias = 0.65
    return _clamp(outcome_score * 0.65 + action_bias * 0.35)


_APPLICATION_PRIOR_PROPOSAL_BUILDER = ApplicationPriorProposalBuilder()


def _experience_deltas_from_prior_update(
    *,
    prior_update: ApplicationPriorUpdate | None,
    blocked_targets: tuple[str, ...],
) -> tuple[ExperienceDelta, ...]:
    if prior_update is None:
        return ()
    blocked_target_set = set(blocked_targets)
    deltas: list[ExperienceDelta] = []
    for update in prior_update.case_memory_updates:
        deltas.append(
            ExperienceDelta(
                delta_id=update.update_id,
                delta_type="case-promotion",
                target_slot="case_memory",
                summary=update.description,
                confidence=update.confidence,
                blocked=update.target in blocked_target_set,
                description=update.record.description,
            )
        )
    for update in prior_update.strategy_playbook_updates:
        deltas.append(
            ExperienceDelta(
                delta_id=update.update_id,
                delta_type="playbook-delta",
                target_slot="strategy_playbook",
                summary=update.description,
                confidence=update.confidence,
                blocked=update.target in blocked_target_set,
                description=update.rule.description,
            )
        )
    for update in prior_update.boundary_policy_updates:
        deltas.append(
            ExperienceDelta(
                delta_id=update.update_id,
                delta_type="boundary-delta",
                target_slot="boundary_policy",
                summary=update.description,
                confidence=update.confidence,
                blocked=update.target in blocked_target_set,
                description=update.hint.description,
            )
        )
    for update in prior_update.domain_knowledge_updates:
        deltas.append(
            ExperienceDelta(
                delta_id=update.update_id,
                delta_type="knowledge-promotion",
                target_slot="domain_knowledge",
                summary=update.description,
                confidence=update.confidence,
                blocked=update.target in blocked_target_set,
                description=update.record.summary,
            )
        )
    for update in prior_update.retrieval_readout_updates:
        deltas.append(
            ExperienceDelta(
                delta_id=update.update_id,
                delta_type="retrieval-readout-delta",
                target_slot="retrieval_policy",
                summary=update.description,
                confidence=update.confidence,
                blocked=update.target in blocked_target_set,
                description=update.checkpoint.description,
            )
        )
    return tuple(deltas)


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
    rollback_operations: tuple[str, ...]
    parameter_change_rate: float
    optimizer_state_norm: float
    checkpoint_id: str
    fast_state_hash: str
    source_fast_state_hash: str
    optimizer_state_description: str
    description: str
    fast_memory_signal: tuple[float, ...] = ()
    experimental_live_mutation: bool = False
    rollback_applied: bool = False
    rollback_reason: str = ""


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
        domain_experience_packages: tuple[DomainExperiencePackage, ...] = (),
        application_persistence_dir: str | None = None,
        credit_proposals: tuple[ModificationProposal, ...] = (),
        response_synthesizer: ResponseSynthesizer | None = None,
        semantic_proposal_runtime: SemanticProposalRuntime | None = None,
        substrate_adapter_factory: Callable[[str, int], SubstrateAdapter] | None = None,
        default_residual_runtime: OpenWeightResidualRuntime | None = None,
        regime_bootstrap: RegimeBootstrap | None = None,
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
        temporal_bootstrap: MetacontrollerParameterSnapshot | DualTrackRareHeavySnapshot | None = None,
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
        world_bootstrap_snapshot = resolve_temporal_bootstrap_snapshot(
            temporal_bootstrap,
            track=Track.WORLD,
        )
        self_bootstrap_snapshot = resolve_temporal_bootstrap_snapshot(
            temporal_bootstrap,
            track=Track.SELF,
        )
        if world_temporal_policy is not None:
            self._world_temporal_policy = world_temporal_policy
        elif temporal_policy is not None:
            self._world_temporal_policy = temporal_policy
        elif world_bootstrap_snapshot is not None:
            self._world_temporal_policy = FullLearnedTemporalPolicy.from_bootstrap_snapshot(
                world_bootstrap_snapshot
            )
        else:
            self._world_temporal_policy = FullLearnedTemporalPolicy()
        if self_temporal_policy is not None:
            self._self_temporal_policy = self_temporal_policy
        elif self_bootstrap_snapshot is not None:
            self._self_temporal_policy = FullLearnedTemporalPolicy.from_bootstrap_snapshot(
                self_bootstrap_snapshot
            )
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
        self._domain_experience_application_report = None
        if domain_experience_packages:
            self._domain_experience_application_report = apply_domain_experience_packages(
                packages=domain_experience_packages,
                domain_knowledge_store=self._domain_knowledge_store,
                case_memory_store=self._case_memory_store,
                application_rare_heavy_state=self._application_rare_heavy_state,
                persist=application_persistence_dir is not None,
            )
        world_parameter_store = getattr(self._world_temporal_policy, "parameter_store", None)
        default_latent_dim = world_parameter_store.n_z if world_parameter_store is not None else 16
        self._memory_store = memory_store or build_default_memory_store(latent_dim=default_latent_dim)
        self._semantic_state_store = SemanticStateStore()
        self._semantic_proposal_runtime = semantic_proposal_runtime or NoOpSemanticProposalRuntime()
        self._pending_semantic_events: list[ExternalSemanticEvent] = []
        self._dialogue_trace_store = DialogueTraceStore()
        self._credit_proposals = credit_proposals
        if response_synthesizer is not None:
            self._response_synthesizer = response_synthesizer
        else:
            self._response_synthesizer = ResponseSynthesizer()
        self._substrate_adapter_factory = substrate_adapter_factory
        self._regime_module = RegimeModule(
            wiring_level=self._config.level_for("regime", WiringLevel.ACTIVE),
            bootstrap=regime_bootstrap,
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
        self._last_online_fast_import_checkpoint: OnlineFastImportCheckpoint | None = None
        self._last_rare_heavy_import_checkpoint: RareHeavyImportCheckpoint | None = None
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
        self._experience_fast_prior_module = ExperienceFastPriorModule(
            wiring_level=self._config.level_for("experience_fast_prior", WiringLevel.SHADOW),
        )
        self._experience_fast_prior_snapshot: Snapshot[ExperienceFastPriorSnapshot] | None = None
        self._last_session_post_writeback_request: SessionPostWritebackRequest | None = None
        self._publish_session_post_snapshot()
        self._publish_experience_consolidation_snapshot()
        self._publish_experience_fast_prior_snapshot()

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
    def memory_store(self) -> MemoryStore:
        return self._memory_store

    @property
    def completed_session_reports(self) -> tuple[EvaluationReport, ...]:
        return tuple(self._completed_session_reports)

    @property
    def dialogue_trace_snapshot(self) -> DialogueTraceSnapshot:
        return self._dialogue_trace_store.snapshot()

    def export_dialogue_trace_replay_artifact(self) -> dict[str, object]:
        return self._dialogue_trace_store.export_replay_artifact()

    def enqueue_semantic_events(
        self,
        events: ExternalSemanticEventBatch | tuple[ExternalSemanticEvent, ...],
    ) -> tuple[str, ...]:
        event_tuple = events.events if isinstance(events, ExternalSemanticEventBatch) else events
        self._pending_semantic_events.extend(event_tuple)
        if len(self._pending_semantic_events) > 64:
            del self._pending_semantic_events[:-64]
        return tuple(event.event_id for event in event_tuple)

    def _drain_pending_semantic_events(self) -> tuple[ExternalSemanticEvent, ...]:
        events = tuple(self._pending_semantic_events)
        self._pending_semantic_events.clear()
        return events

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
    def experience_fast_prior_snapshot(self) -> Snapshot[ExperienceFastPriorSnapshot] | None:
        if self._experience_fast_prior_snapshot is not None:
            return self._experience_fast_prior_snapshot
        snapshot = self._upstream_snapshots.get("experience_fast_prior")
        if snapshot is not None and isinstance(snapshot.value, ExperienceFastPriorSnapshot):
            return snapshot
        return None

    @property
    def active_context_session_id(self) -> str:
        return f"{self._session_id}:context-{self._context_index}"

    def _record_application_delayed_evidence(
        self,
        *,
        completed_results: tuple[SessionPostSlowLoopResult, ...],
    ) -> None:
        for result in completed_results:
            if (
                not result.delayed_outcome_ledger
                and not result.sequence_payoffs
                and not result.application_prior_audits
            ):
                continue
            if result.delayed_outcome_ledger or result.sequence_payoffs:
                self._evaluation_backbone.record_application_delayed_evidence(
                    session_id=result.context_session_id,
                    wave_id=result.job_id,
                    timestamp_ms=max(result.closed_at_turn, 1) + 1000,
                    base_snapshot=EvaluationSnapshot(
                        turn_scores=(),
                        session_scores=(),
                        alerts=(),
                        description="Application delayed evidence baseline.",
                    ),
                    delayed_outcome_ledger=result.delayed_outcome_ledger,
                    sequence_payoffs=result.sequence_payoffs,
                )
            if self._upstream_snapshots.get("credit") is not None:
                credit_snapshot = self._upstream_snapshots["credit"]
                if credit_snapshot is not None:
                    delayed_scores: tuple[EvaluationScore, ...] = ()
                    if result.delayed_outcome_ledger:
                        delayed_scores = delayed_scores + (
                            EvaluationScore(
                                family="learning",
                                metric_name="delayed_retrieval_mix_alignment",
                                value=_clamp(
                                    sum(item.retrieval_mix_alignment for item in result.delayed_outcome_ledger)
                                    / len(result.delayed_outcome_ledger)
                                ),
                                confidence=0.7,
                                evidence=result.delayed_outcome_ledger[-1].description,
                            ),
                            EvaluationScore(
                                family="learning",
                                metric_name="delayed_regime_alignment",
                                value=_clamp(
                                    sum(item.regime_alignment for item in result.delayed_outcome_ledger)
                                    / len(result.delayed_outcome_ledger)
                                ),
                                confidence=0.7,
                                evidence=result.delayed_outcome_ledger[-1].description,
                            ),
                            EvaluationScore(
                                family="abstraction",
                                metric_name="delayed_abstract_action_alignment",
                                value=_clamp(
                                    sum(item.abstract_action_alignment for item in result.delayed_outcome_ledger)
                                    / len(result.delayed_outcome_ledger)
                                ),
                                confidence=0.7,
                                evidence=result.delayed_outcome_ledger[-1].description,
                            ),
                        )
                    if result.sequence_payoffs:
                        delayed_scores = delayed_scores + (
                            EvaluationScore(
                                family="learning",
                                metric_name="regime_sequence_payoff",
                                value=_clamp(
                                    sum(item.rolling_payoff for item in result.sequence_payoffs)
                                    / len(result.sequence_payoffs)
                                ),
                                confidence=0.68,
                                evidence=result.sequence_payoffs[-1].description,
                            ),
                        )
                    delayed_credit_records = derive_learning_evidence_credit_records(
                        evaluation_snapshot=EvaluationSnapshot(
                            turn_scores=(),
                            session_scores=delayed_scores,
                            alerts=(),
                            description="session delayed credit snapshot",
                        ),
                        timestamp_ms=max(result.closed_at_turn, 1) + 1001,
                    )
                    if delayed_credit_records or result.application_prior_audits:
                        extended_credit = extend_credit_snapshot(
                            credit_snapshot=credit_snapshot.value,
                            extra_records=delayed_credit_records,
                            extra_modifications=result.application_prior_audits,
                        )
                        self._upstream_snapshots["credit"] = Snapshot(
                            slot_name="credit",
                            owner=credit_snapshot.owner,
                            version=credit_snapshot.version + 1,
                            timestamp_ms=credit_snapshot.timestamp_ms + 1,
                            value=extended_credit,
                        )

    def _experience_eta_signals(self) -> dict[str, float]:
        signals: dict[str, float] = {}
        fast_prior_snapshot = self.experience_fast_prior_snapshot
        if fast_prior_snapshot is not None:
            fast_prior = fast_prior_snapshot.value
            signals["experience_fast_prior_strength"] = fast_prior.prior_strength
            signals["delayed_fast_prior_available"] = 1.0 if (
                fast_prior.source_attribution_ids or fast_prior.source_sequence_ids
            ) else 0.0
            retrieval_snapshot = self._upstream_snapshots.get("retrieval_policy")
            if retrieval_snapshot is not None and isinstance(retrieval_snapshot.value, RetrievalPolicySnapshot):
                signals["experience_retrieval_mix_bias"] = _clamp(
                    0.5 + (fast_prior.experience_weight_bias - fast_prior.knowledge_weight_bias)
                )
                signals["delayed_retrieval_mix_alignment"] = _clamp(
                    0.5 + (fast_prior.experience_weight_bias - fast_prior.knowledge_weight_bias) * 0.5
                )
            regime_snapshot = self._upstream_snapshots.get("regime")
            if regime_snapshot is not None and isinstance(regime_snapshot.value, RegimeSnapshot):
                active_regime_id = regime_snapshot.value.active_regime.regime_id
                matching_regime_bias = next(
                    (
                        item.bias
                        for item in fast_prior.regime_biases
                        if item.regime_id == active_regime_id
                    ),
                    0.0,
                )
                signals["experience_regime_bias"] = _clamp(0.5 + matching_regime_bias)
                signals["delayed_regime_alignment"] = _clamp(0.5 + matching_regime_bias * 0.5)
            temporal_snapshot = self._upstream_snapshots.get("temporal_abstraction")
            if temporal_snapshot is not None and isinstance(temporal_snapshot.value, TemporalAbstractionSnapshot):
                active_action = temporal_snapshot.value.active_abstract_action
                action_bias = next(
                    (
                        item.bias
                        for item in fast_prior.action_biases
                        if item.abstract_action == active_action
                    ),
                    0.0,
                )
                signals["experience_action_bias"] = _clamp(0.5 + action_bias)
                family_version = temporal_snapshot.value.action_family_version
                family_bias = next(
                    (
                        item.continuation_bias
                        for item in fast_prior.family_biases
                        if item.action_family_version == family_version
                    ),
                    0.0,
                )
                signals["experience_action_family_bias"] = _clamp(0.5 + family_bias)
                signals["delayed_abstract_action_alignment"] = _clamp(
                    0.5 + ((action_bias + family_bias) / 2.0) * 0.5
                )
            if fast_prior.sequence_biases:
                signals["regime_sequence_payoff"] = _clamp(
                    0.5
                    + sum(item.payoff_bias for item in fast_prior.sequence_biases) / len(fast_prior.sequence_biases) * 0.5
                )
        case_snapshot = self._upstream_snapshots.get("case_memory")
        if case_snapshot is not None and isinstance(case_snapshot.value, CaseMemorySnapshot):
            if case_snapshot.value.hits:
                mean_relevance = _clamp(
                    sum(hit.relevance_score for hit in case_snapshot.value.hits) / len(case_snapshot.value.hits)
                )
                self_track_hits = sum(1.0 for hit in case_snapshot.value.hits if "self" in hit.track_tags)
                world_track_hits = sum(1.0 for hit in case_snapshot.value.hits if "world" in hit.track_tags)
                total_hits = max(len(case_snapshot.value.hits), 1)
                signals["experience_case_strength"] = mean_relevance
                signals["experience_case_support_prior"] = _clamp(self_track_hits / total_hits)
                signals["experience_case_task_prior"] = _clamp(world_track_hits / total_hits)
                signals["experience_case_continuum_position"] = _clamp(case_snapshot.value.mean_continuum_position)
                signals["experience_case_continuum_band_coverage"] = _clamp(
                    len(case_snapshot.value.active_band_ids) / 4.0
                )
        playbook_snapshot = self._upstream_snapshots.get("strategy_playbook")
        if playbook_snapshot is not None and isinstance(playbook_snapshot.value, StrategyPlaybookSnapshot):
            if playbook_snapshot.value.matched_rules:
                matched_rules = playbook_snapshot.value.matched_rules
                signals["experience_playbook_strength"] = _clamp(
                    sum(rule.confidence for rule in matched_rules) / len(matched_rules)
                )
                signals["experience_playbook_knowledge_hint"] = _clamp(
                    sum(rule.knowledge_weight_hint for rule in matched_rules) / len(matched_rules)
                )
                signals["experience_playbook_experience_hint"] = _clamp(
                    sum(rule.experience_weight_hint for rule in matched_rules) / len(matched_rules)
                )
                support_prior = sum(
                    1.0
                    for rule in matched_rules
                    if rule.recommended_regime in {"emotional_support", "repair_and_deescalation"}
                ) / len(matched_rules)
                task_prior = sum(
                    1.0
                    for rule in matched_rules
                    if rule.recommended_regime in {"problem_solving", "guided_exploration"}
                ) / len(matched_rules)
                signals["experience_playbook_support_prior"] = _clamp(support_prior)
                signals["experience_playbook_task_prior"] = _clamp(task_prior)
                signals["experience_playbook_band_coverage"] = _clamp(
                    len(playbook_snapshot.value.active_band_ids) / 4.0
                )
        prior_strength_values = tuple(
            signals[key]
            for key in (
                "experience_case_strength",
                "experience_playbook_strength",
                "delayed_retrieval_mix_alignment",
                "regime_sequence_payoff",
                "experience_fast_prior_strength",
                "experience_action_family_bias",
            )
            if key in signals
        )
        if prior_strength_values:
            signals["experience_control_prior_strength"] = _clamp(
                sum(prior_strength_values) / len(prior_strength_values)
            )
        return signals

    async def drain_session_post_slow_loop(self) -> tuple[SessionPostSlowLoopResult, ...]:
        self._session_post_queue.schedule()
        await self._session_post_queue.wait_for_idle()
        results = self._session_post_queue.consume_completed_results()
        self._record_application_delayed_evidence(completed_results=results)
        self._upstream_snapshots["session_post_slow_loop"] = self._publish_session_post_snapshot(completed_results=results)
        self._upstream_snapshots["experience_consolidation"] = self._publish_experience_consolidation_snapshot(
            completed_results=results
        )
        self._upstream_snapshots["experience_fast_prior"] = self._publish_experience_fast_prior_snapshot()
        return results

    def reconcile_case_memory_provisional(
        self,
        *,
        now_tick: int,
        thresholds: ProvisionalReconcileThresholds | None = None,
    ) -> ProvisionalReconcileResult:
        """Sweep CANDIDATE / PROVISIONAL case_memory records (Gap 4 slice 2a).

        Scene-boundary hook: typically called by the lifeform layer's
        ``end_scene`` AFTER ``drain_session_post_slow_loop`` so that any
        provisional records the slow loop wrote during scene close are
        part of the decision set. Returns the full
        ``ProvisionalReconcileResult`` (promoted / retired / expired
        case_ids + per-decision audit tuple) so callers can surface it
        for observability \u2014 the runner does NOT silently swallow the
        outcome.

        ``now_tick`` is the lifeform clock; the kernel never advances
        this itself. Records with ``expires_at_tick <= now_tick`` are
        retired; others go through the promote / retire-by-weakness
        decision table in ``ApplicationCaseMemoryStore``.

        This is a scene-level sweep \u2014 intentionally synchronous and
        bounded. Mid-turn cheap expiry and async mid-reflection
        workers are slice-2b concerns (see
        ``docs/specs/thinking-loop.md``).
        """
        return self._case_memory_store.reconcile_provisional_cases(
            now_tick=now_tick,
            thresholds=thresholds,
        )

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
        case_band_ids: tuple[str, ...] = ()
        case_mean_continuum_position = 0.0
        knowledge_domains: tuple[str, ...] = ()
        knowledge_hits: tuple[KnowledgeHit, ...] = ()
        boundary_trigger_reasons: tuple[str, ...] = ()
        regime_id: str | None = None
        abstract_action: str | None = None
        action_family_version = 0
        retrieval_policy_id: str | None = None
        knowledge_weight = 0.0
        experience_weight = 0.0
        experience_domains: tuple[str, ...] = ()
        regime_sequence: tuple[str, ...] = ()
        case_hit_count = 0
        playbook_rule_count = 0
        playbook_band_ids: tuple[str, ...] = ()
        continuum_profile_id: str | None = None
        retrieval_fast_prior_strength = 0.0
        retrieval_fast_prior_attribution_count = 0
        retrieval_fast_prior_sequence_count = 0
        retrieval_regime_bias = 0.0
        retrieval_action_bias = 0.0
        retrieval_family_bias = 0.0
        retrieval_knowledge_weight_bias = 0.0
        retrieval_experience_weight_bias = 0.0
        case_snapshot = self._upstream_snapshots.get("case_memory")
        if case_snapshot is not None and isinstance(case_snapshot.value, CaseMemorySnapshot):
            case_problem_patterns = case_snapshot.value.active_problem_patterns
            case_risk_markers = case_snapshot.value.active_risk_markers
            case_hit_count = len(case_snapshot.value.hits)
            case_band_ids = case_snapshot.value.active_band_ids
            case_mean_continuum_position = case_snapshot.value.mean_continuum_position
            continuum_profile_id = case_snapshot.value.continuum_profile_id
        knowledge_snapshot = self._upstream_snapshots.get("domain_knowledge")
        if knowledge_snapshot is not None and isinstance(knowledge_snapshot.value, DomainKnowledgeSnapshot):
            knowledge_domains = knowledge_snapshot.value.active_domains
            knowledge_hits = knowledge_snapshot.value.hits
        retrieval_policy_snapshot = self._upstream_snapshots.get("retrieval_policy")
        if retrieval_policy_snapshot is not None:
            retrieval_policy_value = retrieval_policy_snapshot.value
            retrieval_policy_id = f"policy:{hash(retrieval_policy_value.intent_description) & 0xFFFF:04x}"
            regime_id = retrieval_policy_value.regime_id
            abstract_action = retrieval_policy_value.abstract_action
            knowledge_weight = retrieval_policy_value.knowledge_weight
            experience_weight = retrieval_policy_value.experience_weight
            experience_domains = retrieval_policy_value.experience_domains
        boundary_snapshot = self._upstream_snapshots.get("boundary_policy")
        if boundary_snapshot is not None and isinstance(boundary_snapshot.value, BoundaryPolicySnapshot):
            boundary_trigger_reasons = boundary_snapshot.value.trigger_reasons
        regime_snapshot = self._upstream_snapshots.get("regime")
        if regime_snapshot is not None and isinstance(regime_snapshot.value, RegimeSnapshot):
            regime_id = regime_id or regime_snapshot.value.active_regime.regime_id
            if regime_snapshot.value.previous_regime is not None:
                regime_sequence = (
                    regime_snapshot.value.previous_regime.regime_id,
                    regime_snapshot.value.active_regime.regime_id,
                )
            else:
                regime_sequence = (regime_snapshot.value.active_regime.regime_id,)
        temporal_snapshot = self._upstream_snapshots.get("temporal_abstraction")
        if temporal_snapshot is not None and isinstance(temporal_snapshot.value, TemporalAbstractionSnapshot):
            abstract_action = abstract_action or temporal_snapshot.value.active_abstract_action
            action_family_version = temporal_snapshot.value.action_family_version
        playbook_snapshot = self._upstream_snapshots.get("strategy_playbook")
        if playbook_snapshot is not None and isinstance(playbook_snapshot.value, StrategyPlaybookSnapshot):
            playbook_rule_count = len(playbook_snapshot.value.matched_rules)
            playbook_band_ids = playbook_snapshot.value.active_band_ids
            continuum_profile_id = continuum_profile_id or playbook_snapshot.value.continuum_profile_id
        fast_prior_snapshot = self.experience_fast_prior_snapshot
        if fast_prior_snapshot is not None:
            fast_prior = fast_prior_snapshot.value
            retrieval_fast_prior_strength = fast_prior.prior_strength
            retrieval_fast_prior_attribution_count = len(fast_prior.source_attribution_ids)
            retrieval_fast_prior_sequence_count = len(fast_prior.source_sequence_ids)
            retrieval_knowledge_weight_bias = fast_prior.knowledge_weight_bias
            retrieval_experience_weight_bias = fast_prior.experience_weight_bias
            if regime_id is not None:
                retrieval_regime_bias = next(
                    (item.bias for item in fast_prior.regime_biases if item.regime_id == regime_id),
                    0.0,
                )
            if abstract_action is not None:
                retrieval_action_bias = next(
                    (item.bias for item in fast_prior.action_biases if item.abstract_action == abstract_action),
                    0.0,
                )
            if action_family_version > 0:
                retrieval_family_bias = next(
                    (
                        item.continuation_bias
                        for item in fast_prior.family_biases
                        if item.action_family_version == action_family_version
                    ),
                    0.0,
                )
        conversation_knowledge_candidates = build_conversation_knowledge_candidates(
            knowledge_hits=knowledge_hits,
            context_session_id=request.context_session_id,
            source_wave_id=request.source_wave_id,
            source_turn_index=self._turn_index,
            boundary_trigger_reasons=boundary_trigger_reasons,
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
            case_problem_patterns=case_problem_patterns,
            case_risk_markers=case_risk_markers,
            knowledge_domains=knowledge_domains,
            knowledge_hits=knowledge_hits,
            conversation_knowledge_candidates=conversation_knowledge_candidates,
            boundary_trigger_reasons=boundary_trigger_reasons,
            regime_id=regime_id,
            abstract_action=abstract_action,
            action_family_version=action_family_version,
            retrieval_policy_id=retrieval_policy_id,
            knowledge_weight=knowledge_weight,
            experience_weight=experience_weight,
            experience_domains=experience_domains,
            regime_sequence=regime_sequence,
            case_hit_count=case_hit_count,
            playbook_rule_count=playbook_rule_count,
            continuum_profile_id=continuum_profile_id,
            case_band_ids=case_band_ids,
            case_mean_continuum_position=case_mean_continuum_position,
            playbook_band_ids=playbook_band_ids,
            retrieval_fast_prior_strength=retrieval_fast_prior_strength,
            retrieval_fast_prior_attribution_count=retrieval_fast_prior_attribution_count,
            retrieval_fast_prior_sequence_count=retrieval_fast_prior_sequence_count,
            retrieval_regime_bias=retrieval_regime_bias,
            retrieval_action_bias=retrieval_action_bias,
            retrieval_family_bias=retrieval_family_bias,
            retrieval_knowledge_weight_bias=retrieval_knowledge_weight_bias,
            retrieval_experience_weight_bias=retrieval_experience_weight_bias,
            semantic_state_descriptions=request.semantic_state_descriptions,
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
        prediction_summary = dict(job.prediction_error_summary)
        reward = float(prediction_summary.get("signed_reward", 0.0))
        magnitude = float(prediction_summary.get("magnitude", 0.0))
        relationship_error = float(prediction_summary.get("relationship_error", 0.0))
        regime_error = float(prediction_summary.get("regime_error", 0.0))
        outcome_score = _application_outcome_score(
            reward=reward,
            magnitude=magnitude,
            relationship_error=relationship_error,
        )
        retrieval_mix_alignment = _retrieval_mix_alignment(
            regime_id=job.regime_id,
            knowledge_weight=job.knowledge_weight,
            experience_weight=job.experience_weight,
        )
        regime_alignment = _regime_alignment(
            regime_id=job.regime_id,
            outcome_score=outcome_score,
            relationship_error=relationship_error,
            regime_error=regime_error,
            magnitude=magnitude,
        )
        abstract_action_alignment = _abstract_action_alignment(
            regime_id=job.regime_id,
            abstract_action=job.abstract_action,
            action_family_version=job.action_family_version,
            outcome_score=outcome_score,
        )
        dominant_band_id = (
            job.case_band_ids[0]
            if job.case_band_ids
            else job.playbook_band_ids[0]
            if job.playbook_band_ids
            else None
        )
        continuum_alignment = _clamp(
            0.5
            + (
                (1.0 - abs(job.case_mean_continuum_position - 0.5))
                * 0.35
                + retrieval_mix_alignment * 0.35
                + abstract_action_alignment * 0.30
            )
            / 2.0
        )
        delayed_outcome_ledger = (
            ApplicationOutcomeAttribution(
                attribution_id=f"{job.job_id}:outcome",
                source_context_session_id=job.context_session_id,
                source_wave_id=job.writeback_request.source_wave_id,
                regime_id=job.regime_id,
                abstract_action=job.abstract_action,
                action_family_version=job.action_family_version,
                retrieval_policy_id=job.retrieval_policy_id,
                knowledge_weight=job.knowledge_weight,
                experience_weight=job.experience_weight,
                retrieval_mix_alignment=retrieval_mix_alignment,
                regime_alignment=regime_alignment,
                abstract_action_alignment=abstract_action_alignment,
                outcome_score=outcome_score,
                resolved_turn_index=job.closed_at_turn,
                continuum_profile_id=job.continuum_profile_id,
                dominant_band_id=dominant_band_id,
                mean_continuum_position=job.case_mean_continuum_position,
                continuum_alignment=continuum_alignment,
                description=(
                    f"Delayed application outcome for regime={job.regime_id} abstract_action={job.abstract_action} "
                    f"knowledge_weight={job.knowledge_weight:.2f} experience_weight={job.experience_weight:.2f} "
                    f"reward={reward:.2f} magnitude={magnitude:.2f}."
                ),
            ),
        )
        sequence_payoffs = (
            ApplicationSequencePayoff(
                sequence_id=f"{job.job_id}:sequence",
                regime_sequence=job.regime_sequence or ((job.regime_id,) if job.regime_id is not None else ()),
                action_family_version=job.action_family_version,
                sample_count=1,
                rolling_payoff=outcome_score,
                latest_outcome=outcome_score,
                continuum_profile_id=job.continuum_profile_id,
                dominant_band_id=dominant_band_id,
                mean_continuum_position=job.case_mean_continuum_position,
                description=(
                    f"Sequence payoff for regime_sequence={job.regime_sequence or ((job.regime_id,) if job.regime_id is not None else ())} "
                    f"family_version={job.action_family_version}."
                ),
            ),
        )
        delayed_credit_summary = DelayedCreditSummary(
            summary_id=f"{job.job_id}:delayed-credit-summary",
            regime_id=job.regime_id,
            abstract_action=job.abstract_action,
            action_family_version=job.action_family_version,
            retrieval_policy_id=job.retrieval_policy_id,
            knowledge_weight=job.knowledge_weight,
            experience_weight=job.experience_weight,
            retrieval_mix_alignment=retrieval_mix_alignment,
            regime_alignment=regime_alignment,
            abstract_action_alignment=abstract_action_alignment,
            outcome_score=outcome_score,
            sequence_payoff=sequence_payoffs[0].rolling_payoff,
            continuum_alignment=continuum_alignment,
            attribution_count=len(delayed_outcome_ledger),
            sequence_count=len(sequence_payoffs),
            continuum_profile_id=job.continuum_profile_id,
            dominant_band_id=dominant_band_id,
            mean_continuum_position=job.case_mean_continuum_position,
            description=(
                f"Delayed credit summary for regime={job.regime_id} abstract_action={job.abstract_action} "
                f"family_version={job.action_family_version} outcome={outcome_score:.2f} "
                f"mix_alignment={retrieval_mix_alignment:.2f} sequence_payoff={sequence_payoffs[0].rolling_payoff:.2f}."
            ),
        )
        mean_experience_quality = _clamp(
            (
                retrieval_mix_alignment
                + regime_alignment
                + abstract_action_alignment
                + outcome_score
            )
            / 4.0
        )
        application_prior_update = _APPLICATION_PRIOR_PROPOSAL_BUILDER.build(
            inputs=ApplicationPriorProposalInputs(
                job_id=job.job_id,
                closed_at_turn=job.closed_at_turn,
                regime_id=job.regime_id,
                knowledge_domains=job.knowledge_domains,
                experience_domains=job.experience_domains,
                case_problem_patterns=job.case_problem_patterns,
                case_risk_markers=job.case_risk_markers,
                boundary_trigger_reasons=job.boundary_trigger_reasons,
                knowledge_weight=job.knowledge_weight,
                experience_weight=job.experience_weight,
                case_hit_count=job.case_hit_count,
                mean_experience_quality=mean_experience_quality,
                knowledge_hits=job.knowledge_hits,
                conversation_knowledge_candidates=job.conversation_knowledge_candidates,
                retrieval_readout_checkpoint=self._application_rare_heavy_state.retrieval_readout_checkpoint,
                retrieval_fast_prior_strength=max(job.retrieval_fast_prior_strength, mean_experience_quality),
                retrieval_fast_prior_attribution_count=max(job.retrieval_fast_prior_attribution_count, 1),
                retrieval_fast_prior_sequence_count=max(job.retrieval_fast_prior_sequence_count, 1),
                retrieval_regime_bias=max(job.retrieval_regime_bias, regime_alignment - 0.5),
                retrieval_action_bias=max(job.retrieval_action_bias, abstract_action_alignment - 0.5),
                retrieval_family_bias=max(job.retrieval_family_bias, outcome_score - 0.5),
                retrieval_knowledge_weight_bias=(
                    job.retrieval_knowledge_weight_bias - max(retrieval_mix_alignment - 0.5, 0.0) * 0.5
                ),
                retrieval_experience_weight_bias=(
                    job.retrieval_experience_weight_bias + max(retrieval_mix_alignment - 0.5, 0.0) * 0.5
                ),
                retrieval_source_attribution_ids=tuple(
                    item.attribution_id for item in delayed_outcome_ledger
                ),
                retrieval_source_sequence_ids=tuple(
                    item.sequence_id for item in sequence_payoffs
                ),
                retrieval_mean_retrieval_mix_alignment=retrieval_mix_alignment,
                retrieval_mean_regime_alignment=regime_alignment,
                retrieval_mean_action_alignment=abstract_action_alignment,
                retrieval_mean_sequence_payoff=(
                    sum(item.rolling_payoff for item in sequence_payoffs) / len(sequence_payoffs)
                    if sequence_payoffs
                    else 0.0
                ),
            )
        )
        application_apply_enabled = (
            job.writeback_request.reflection_apply_enabled
            and job.writeback_request.structural_writeback_allowed
            and mean_experience_quality >= 0.52
        )
        retrieval_checkpoint_apply_enabled = (
            job.writeback_request.reflection_apply_enabled
            and mean_experience_quality >= 0.45
        )
        if not job.writeback_request.reflection_apply_enabled:
            application_block_reason = "writeback-mode-not-apply"
        elif not job.writeback_request.structural_writeback_allowed:
            application_block_reason = "evolution-judge-block"
        elif mean_experience_quality < 0.52:
            application_block_reason = "experience-quality-below-threshold"
        else:
            application_block_reason = "allow"
        async with self._session_post_lock:
            (
                application_prior_ops,
                application_prior_blocks,
                application_prior_audits,
                application_prior_writeback_report,
            ) = _apply_application_prior_writeback(
                prior_update=application_prior_update,
                domain_knowledge_store=self._domain_knowledge_store,
                case_memory_store=self._case_memory_store,
                application_rare_heavy_state=self._application_rare_heavy_state,
                credit_snapshot=job.writeback_request.credit_snapshot,
                timestamp_ms=max(job.closed_at_turn, 1) + 2,
                checkpoint_id=job.writeback_request.checkpoint_id,
                apply_enabled=application_apply_enabled,
                retrieval_apply_enabled=retrieval_checkpoint_apply_enabled,
                blocked_reason=application_block_reason,
            )
        if application_prior_writeback_report is None:
            application_prior_writeback_report = ApplicationPriorWritebackReport(
                proposed_target_count=0,
                applied_targets=(),
                blocked_targets=(),
                audit_record_count=0,
                description="No application prior update was proposed for this slow-loop result.",
            )
        if writeback_result is not None and (application_prior_ops or application_prior_blocks):
            writeback_result = replace(
                writeback_result,
                applied_operations=writeback_result.applied_operations + application_prior_ops,
                blocked_operations=writeback_result.blocked_operations + application_prior_blocks,
                description=(
                    f"{writeback_result.description} application_prior_ops={len(application_prior_ops)} "
                    f"application_prior_blocks={len(application_prior_blocks)}."
                ),
            )
        elif application_prior_ops or application_prior_blocks:
            writeback_result = WritebackResult(
                applied_operations=application_prior_ops,
                blocked_operations=application_prior_blocks,
                checkpoint=None,
                description=(
                    f"Session-post application prior writeback produced {len(application_prior_ops)} applied ops "
                    f"and {len(application_prior_blocks)} blocked ops."
                ),
            )
        experience_deltas = _experience_deltas_from_prior_update(
            prior_update=application_prior_update,
            blocked_targets=application_prior_writeback_report.blocked_targets,
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
                f"blocked={bool(writeback_result is not None and writeback_result.blocked_operations)} "
                f"application_promotion={'allow' if application_apply_enabled else 'blocked'} "
                f"experience_quality={mean_experience_quality:.2f} "
                f"semantic_state={len(job.semantic_state_descriptions)}."
            ),
            experience_deltas=experience_deltas,
            delayed_outcome_ledger=delayed_outcome_ledger,
            sequence_payoffs=sequence_payoffs,
            delayed_credit_summary=delayed_credit_summary,
            conversation_knowledge_candidates=job.conversation_knowledge_candidates,
            application_prior_update=application_prior_update,
            application_prior_writeback_report=application_prior_writeback_report,
            application_prior_audits=application_prior_audits,
            continuum_profile_id=job.continuum_profile_id,
            case_band_ids=job.case_band_ids,
            playbook_band_ids=job.playbook_band_ids,
            semantic_state_descriptions=job.semantic_state_descriptions,
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
        if not completed_results and self._experience_consolidation_snapshot is not None:
            return self._experience_consolidation_snapshot
        self._experience_consolidation_snapshot = self._experience_consolidation_module.publish_snapshot(
            completed_results=completed_results,
        )
        return self._experience_consolidation_snapshot

    def _publish_experience_fast_prior_snapshot(self) -> Snapshot[ExperienceFastPriorSnapshot]:
        experience_consolidation = (
            self._experience_consolidation_snapshot.value
            if self._experience_consolidation_snapshot is not None
            else None
        )
        self._experience_fast_prior_snapshot = self._experience_fast_prior_module.publish_snapshot(
            experience_consolidation_snapshot=experience_consolidation,
        )
        return self._experience_fast_prior_snapshot

    def _collect_session_post_writeback_result(self) -> WritebackResult | None:
        completed = self._session_post_queue.consume_completed_results()
        if completed:
            self._record_application_delayed_evidence(completed_results=completed)
            self._upstream_snapshots["session_post_slow_loop"] = self._publish_session_post_snapshot(
                completed_results=completed
            )
            self._upstream_snapshots["experience_consolidation"] = self._publish_experience_consolidation_snapshot(
                completed_results=completed
            )
            self._upstream_snapshots["experience_fast_prior"] = self._publish_experience_fast_prior_snapshot()
        else:
            self._upstream_snapshots["session_post_slow_loop"] = self._publish_session_post_snapshot()
            self._upstream_snapshots["experience_consolidation"] = self._publish_experience_consolidation_snapshot()
            self._upstream_snapshots["experience_fast_prior"] = self._publish_experience_fast_prior_snapshot()
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
        knowledge_import_operations: tuple[str, ...] = ()
        if artifact.application_checkpoint is not None and artifact.application_checkpoint.reviewed_knowledge_candidates:
            reviewed = artifact.application_checkpoint.reviewed_knowledge_candidates
            knowledge_updates = domain_knowledge_prior_updates_from_reviewed(
                job_id=f"{artifact.artifact_id}:rare-heavy-knowledge-import",
                reviewed=reviewed,
            )
            if knowledge_updates:
                knowledge_prior = ApplicationPriorUpdate(
                    source_session_post_job_id=f"{artifact.artifact_id}:rare-heavy-knowledge-import",
                    domain_knowledge_updates=knowledge_updates,
                    description="Rare-heavy reviewed knowledge import batch (owner-side gated writeback).",
                )
                (
                    knowledge_import_operations,
                    _knowledge_blocks,
                    _knowledge_audits,
                    _knowledge_report,
                ) = _apply_application_prior_writeback(
                    prior_update=knowledge_prior,
                    domain_knowledge_store=self._domain_knowledge_store,
                    case_memory_store=self._case_memory_store,
                    application_rare_heavy_state=self._application_rare_heavy_state,
                    credit_snapshot=None,
                    timestamp_ms=max(self._turn_index, 1) + 3,
                    checkpoint_id=checkpoint_id or artifact.artifact_id,
                    apply_enabled=True,
                    retrieval_apply_enabled=True,
                    blocked_reason="allow",
                )
        tail_operations = application_operations + knowledge_import_operations + reset_operations
        if not tail_operations:
            return result
        return replace(
            result,
            applied_operations=result.applied_operations + tail_operations,
            description=(
                f"{result.description} "
                f"{'Application rare-heavy state imported. ' if application_operations else ''}"
                f"{'Reviewed domain knowledge import applied. ' if knowledge_import_operations else ''}"
                f"{'Nested context reset applied after import.' if reset_operations else ''}"
            ).strip(),
        )

    def review_rare_heavy_artifact(
        self,
        artifact: RareHeavyArtifact,
        *,
        checkpoint_id: str | None = None,
    ) -> RareHeavyImportResult:
        application_checkpoint = self._application_rare_heavy_state.export_rare_heavy_state(
            checkpoint_id=f"{checkpoint_id or artifact.artifact_id}:application-review"
        )
        result = self._joint_loop.review_rare_heavy_artifact(
            artifact,
            checkpoint_id=checkpoint_id,
        )
        return replace(
            result,
            checkpoint=replace(result.checkpoint, application_checkpoint=application_checkpoint),
            description=(
                f"{result.description} "
                "Application rare-heavy state remained under session-owned review."
            ).strip(),
        )

    def rollback_rare_heavy_import(
        self,
        checkpoint: RareHeavyImportCheckpoint,
    ) -> tuple[str, ...]:
        operations = list(self._joint_loop.rollback_rare_heavy_import(checkpoint))
        if checkpoint.application_checkpoint is not None:
            operations.extend(self._application_rare_heavy_state.restore_rare_heavy_state(checkpoint.application_checkpoint))
        return tuple(operations)

    async def run_turn(
        self,
        user_input: str,
        *,
        environment_event: EnvironmentEvent | None = None,
    ) -> AgentTurnResult:
        deferred_writeback_result = self._collect_session_post_writeback_result()
        self._session_post_queue.schedule()
        async with self._session_post_lock:
            self._turn_index += 1
            wave_id = f"wave-{self._turn_index}"
            context_session_id = self.active_context_session_id
            if environment_event is None:
                environment_event = build_user_input_environment_event(
                    event_id=f"{wave_id}-environment-event",
                    user_input=user_input,
                    scene_id=context_session_id,
                    timestamp_ms=self._turn_index,
                    provenance="AgentSessionRunner.run_turn",
                )
            pre_turn_world_temporal_snapshot = self._joint_loop.world_temporal_policy.export_rare_heavy_snapshot()
            pre_turn_self_temporal_snapshot = self._joint_loop.self_temporal_policy.export_rare_heavy_snapshot()
            substrate_adapter = self._build_substrate_adapter(user_input=user_input)
            trace = self._build_training_trace_from_substrate(user_input=user_input)
            self._record_training_trace(trace)
            pe_task_error = self._previous_prediction_error.task_error if self._previous_prediction_error is not None else 0.0
            pe_relationship_error = (
                self._previous_prediction_error.relationship_error if self._previous_prediction_error is not None else 0.0
            )
            pe_regime_error = self._previous_prediction_error.regime_error if self._previous_prediction_error is not None else 0.0
            pe_action_error = self._previous_prediction_error.action_error if self._previous_prediction_error is not None else 0.0
            experience_eta_signals = self._experience_eta_signals()
            if self._external_prediction_error_drive:
                pe_signals = (
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
                self._joint_loop.set_external_learning_signals(
                    {
                        **experience_eta_signals,
                        **pe_signals,
                    }
                )
            else:
                readout_only_signals = dict(experience_eta_signals)
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
            pending_semantic_events = self._drain_pending_semantic_events()
            active_semantic_runtime: SemanticProposalRuntime = (
                AdapterSemanticProposalRuntime(
                    base_runtime=self._semantic_proposal_runtime,
                    external_events=pending_semantic_events,
                )
                if pending_semantic_events
                else self._semantic_proposal_runtime
            )
            integration_result = await run_final_wiring_turn(
                config=self._config,
                substrate_adapter=substrate_adapter,
                user_input=user_input,
                application_rare_heavy_state=self._application_rare_heavy_state,
                domain_knowledge_store=self._domain_knowledge_store,
                case_memory_store=self._case_memory_store,
                memory_store=self._memory_store,
                semantic_state_store=self._semantic_state_store,
                semantic_proposal_runtime=active_semantic_runtime,
                evaluation_backbone=self._evaluation_backbone,
                prior_session_reports=self.completed_session_reports,
                upstream_snapshots=self._upstream_snapshots,
                joint_loop_result=joint_result,
                environment_event=environment_event,
                credit_proposals=self._credit_proposals,
                reflection_mode=self._reflection_mode,
                world_temporal_policy=self._world_temporal_policy,
                self_temporal_policy=self._self_temporal_policy,
                prediction_module=self._prediction_module,
                regime_module=self._regime_module,
                session_id=context_session_id,
                wave_id=wave_id,
                turn_index=self._turn_index,
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
                pre_turn_world_temporal_snapshot=pre_turn_world_temporal_snapshot,
                pre_turn_self_temporal_snapshot=pre_turn_self_temporal_snapshot,
            )
            self._maybe_apply_delayed_substrate_rollback(
                integration_result=integration_result,
            )
        self._session_post_queue.schedule()
        return self._to_turn_result(
            user_input=user_input,
            wave_id=wave_id,
            environment_event=environment_event,
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
                EvaluationScore(
                    family="learning",
                    metric_name="substrate_online_fast_experimental_mode",
                    value=1.0 if result.experimental_live_mutation else 0.0,
                    confidence=0.82,
                    evidence=(
                        "Derived from session/runtime capability for experimental bounded live mutation."
                    ),
                ),
                EvaluationScore(
                    family="safety",
                    metric_name="substrate_online_fast_rollback_integrity",
                    value=1.0
                    if (not result.rollback_reason or result.rollback_applied)
                    else 0.0,
                    confidence=0.78,
                    evidence=(
                        result.description
                        if not result.rollback_reason
                        else (
                            f"rollback_reason={result.rollback_reason}, "
                            f"rollback_ops={len(result.rollback_operations)}."
                        )
                    ),
                ),
                EvaluationScore(
                    family="safety",
                    metric_name="substrate_online_fast_review_or_revert_safe",
                    value=1.0 if (not result.applied or result.rollback_applied or result.experimental_live_mutation) else 0.8,
                    confidence=0.72,
                    evidence=(
                        f"gate_decision={result.gate_decision}, experimental={result.experimental_live_mutation}, "
                        f"rollback={result.rollback_applied}."
                    ),
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

    def _refresh_memory_snapshot_after_online_fast_evidence(
        self,
        *,
        integration_result: FinalIntegrationResult,
    ) -> None:
        memory_snapshot = integration_result.active_snapshots.get("memory")
        if memory_snapshot is None or not isinstance(memory_snapshot.value, MemorySnapshot):
            return
        integration_result.active_snapshots["memory"] = Snapshot(
            slot_name=memory_snapshot.slot_name,
            owner=memory_snapshot.owner,
            version=memory_snapshot.version + 1,
            timestamp_ms=max(memory_snapshot.timestamp_ms + 1, self._turn_index),
            value=self._memory_store.snapshot(
                retrieved_entries=memory_snapshot.value.retrieved_entries,
            ),
        )

    def _delayed_substrate_rollback_reasons(
        self,
        *,
        integration_result: FinalIntegrationResult,
    ) -> tuple[str, ...]:
        del integration_result
        return ()

    def _append_delayed_rollback_evaluation_evidence(
        self,
        *,
        integration_result: FinalIntegrationResult,
        reasons: tuple[str, ...],
        operations: tuple[str, ...],
    ) -> None:
        evaluation_snapshot = integration_result.active_snapshots.get("evaluation")
        if evaluation_snapshot is None or not isinstance(evaluation_snapshot.value, EvaluationSnapshot):
            return
        enriched = self._evaluation_backbone.record_external_scores(
            session_id=self.active_context_session_id,
            wave_id=f"wave-{self._turn_index}",
            timestamp_ms=max(evaluation_snapshot.timestamp_ms + 1, self._turn_index),
            base_snapshot=evaluation_snapshot.value,
            scores=(
                EvaluationScore(
                    family="safety",
                    metric_name="substrate_delayed_rollback_applied",
                    value=1.0 if operations else 0.0,
                    confidence=0.82,
                    evidence=(
                        f"Derived from delayed rollback reasons={reasons} "
                        f"and operation_count={len(operations)}."
                    ),
                ),
            ),
            description_suffix="Session owner appended delayed substrate rollback evidence.",
        )
        integration_result.active_snapshots["evaluation"] = Snapshot(
            slot_name="evaluation",
            owner="EvaluationModule",
            version=evaluation_snapshot.version + 1,
            timestamp_ms=max(evaluation_snapshot.timestamp_ms + 1, self._turn_index),
            value=enriched,
        )

    def _maybe_apply_delayed_substrate_rollback(
        self,
        *,
        integration_result: FinalIntegrationResult,
    ) -> tuple[str, ...]:
        reasons = tuple(self._delayed_substrate_rollback_reasons(integration_result=integration_result))
        if not reasons:
            return ()
        operations: list[str] = []
        if self._last_online_fast_import_checkpoint is not None:
            operations.extend(
                self._joint_loop.rollback_online_fast_substrate_import(
                    self._last_online_fast_import_checkpoint
                )
            )
            self._last_online_fast_import_checkpoint = None
        if self._last_rare_heavy_import_checkpoint is not None:
            operations.extend(
                self.rollback_rare_heavy_import(self._last_rare_heavy_import_checkpoint)
            )
            self._last_rare_heavy_import_checkpoint = None
        applied = tuple(operations)
        self._append_delayed_rollback_evaluation_evidence(
            integration_result=integration_result,
            reasons=reasons,
            operations=applied,
        )
        return applied

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
                rollback_operations=(),
                parameter_change_rate=substrate_self_mod.parameter_change_rate,
                optimizer_state_norm=substrate_self_mod.optimizer_state_norm,
                checkpoint_id=substrate_self_mod.checkpoint.checkpoint_id,
                fast_state_hash=substrate_self_mod.checkpoint.fast_state_hash,
                source_fast_state_hash=substrate_self_mod.checkpoint.source_fast_state_hash,
                optimizer_state_description=substrate_self_mod.checkpoint.optimizer_state_description,
                fast_memory_signal=substrate_self_mod.checkpoint.fast_memory_signal,
                description="Online-fast substrate self-mod proposal was present, but schedule was not due.",
                experimental_live_mutation=self.residual_runtime.supports_live_substrate_mutation,
            )
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
                rollback_operations=(),
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
                experimental_live_mutation=False,
            )
            self._append_online_fast_evaluation_evidence(
                integration_result=integration_result,
                wave_id=wave_id,
                result=blocked_result,
            )
            if substrate_self_mod.checkpoint.fast_memory_signal:
                self._memory_store.observe_fast_memory_signal(
                    signal=substrate_self_mod.checkpoint.fast_memory_signal,
                    timestamp_ms=max(self._turn_index, 1),
                )
                self._refresh_memory_snapshot_after_online_fast_evidence(
                    integration_result=integration_result,
                )
            return blocked_result
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
                rollback_operations=(),
                parameter_change_rate=substrate_self_mod.parameter_change_rate,
                optimizer_state_norm=substrate_self_mod.optimizer_state_norm,
                checkpoint_id=substrate_self_mod.checkpoint.checkpoint_id,
                fast_state_hash=substrate_self_mod.checkpoint.fast_state_hash,
                source_fast_state_hash=substrate_self_mod.checkpoint.source_fast_state_hash,
                optimizer_state_description=substrate_self_mod.checkpoint.optimizer_state_description,
                fast_memory_signal=substrate_self_mod.checkpoint.fast_memory_signal,
                description="Online-fast substrate self-mod proposal was blocked by the ONLINE evaluation gate.",
                experimental_live_mutation=self.residual_runtime.supports_live_substrate_mutation,
            )
            self._append_online_fast_evaluation_evidence(
                integration_result=integration_result,
                wave_id=wave_id,
                result=blocked_result,
            )
            if substrate_self_mod.checkpoint.fast_memory_signal:
                self._memory_store.observe_fast_memory_signal(
                    signal=substrate_self_mod.checkpoint.fast_memory_signal,
                    timestamp_ms=max(self._turn_index, 1),
                )
                self._refresh_memory_snapshot_after_online_fast_evidence(
                    integration_result=integration_result,
                )
            return blocked_result
        import_result = self._joint_loop.apply_online_fast_substrate_checkpoint(
            substrate_self_mod.checkpoint,
            checkpoint_id=f"{self._session_id}:{wave_id}:online-fast-substrate",
        )
        self._last_online_fast_import_checkpoint = import_result.checkpoint
        prior_checkpoint = import_result.checkpoint.substrate_checkpoint
        if substrate_self_mod.checkpoint.fast_memory_signal:
            self._memory_store.observe_fast_memory_signal(
                signal=substrate_self_mod.checkpoint.fast_memory_signal,
                timestamp_ms=max(self._turn_index, 1),
            )
            self._refresh_memory_snapshot_after_online_fast_evidence(
                integration_result=integration_result,
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
        rollback_reason = ""
        rollback_operations: tuple[str, ...] = ()
        if (
            substrate_self_mod.parameter_change_rate > 0.85
            and substrate_self_mod.optimizer_state_norm > 0.85
        ):
            rollback_reason = "online-fast-integrity-guard"
            rollback_operations = self._joint_loop.rollback_online_fast_substrate_import(
                import_result.checkpoint
            )
            self._last_online_fast_import_checkpoint = None
            self._append_online_fast_credit_audit(
                integration_result=integration_result,
                record=SelfModificationRecord(
                    target=substrate_self_mod.target,
                    gate=ModificationGate.ONLINE,
                    decision=GateDecision.BLOCK,
                    old_value_hash=substrate_self_mod.checkpoint.checkpoint_id,
                    new_value_hash=prior_checkpoint.checkpoint_id if prior_checkpoint is not None else "none",
                    justification=(
                        "Online-fast substrate self-mod proposal was rolled back by the session integrity guard. "
                        f"{substrate_self_mod.description}"
                    ),
                    timestamp_ms=self._turn_index,
                    is_reversible=True,
                    checkpoint_id=(
                        prior_checkpoint.checkpoint_id if prior_checkpoint is not None else substrate_self_mod.checkpoint.checkpoint_id
                    ),
                    lineage_hash=substrate_self_mod.checkpoint.fast_state_hash,
                    proposal_hash=substrate_self_mod.checkpoint_hash,
                ),
            )
        applied_result = OnlineFastSubstrateTurnResult(
            recommended=True,
            applied=not bool(rollback_operations),
            gate_decision=(
                "allowed-then-rolled-back" if rollback_operations else gate_decision.value
            ),
            applied_operations=import_result.applied_operations,
            blocked_operations=(),
            rollback_operations=rollback_operations,
            parameter_change_rate=substrate_self_mod.parameter_change_rate,
            optimizer_state_norm=substrate_self_mod.optimizer_state_norm,
            checkpoint_id=substrate_self_mod.checkpoint.checkpoint_id,
            fast_state_hash=substrate_self_mod.checkpoint.fast_state_hash,
            source_fast_state_hash=substrate_self_mod.checkpoint.source_fast_state_hash,
            optimizer_state_description=substrate_self_mod.checkpoint.optimizer_state_description,
            fast_memory_signal=substrate_self_mod.checkpoint.fast_memory_signal,
            description=(
                import_result.description
                if not rollback_operations
                else (
                    f"{import_result.description} Session integrity guard rolled the substrate back "
                    f"via {len(rollback_operations)} owner-side operations."
                )
            ),
            experimental_live_mutation=True,
            rollback_applied=bool(rollback_operations),
            rollback_reason=rollback_reason,
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
        pre_turn_world_temporal_snapshot: object,
        pre_turn_self_temporal_snapshot: object,
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
        self._last_rare_heavy_import_checkpoint = replace(
            import_result.checkpoint,
            world_temporal_snapshot=pre_turn_world_temporal_snapshot,
            self_temporal_snapshot=pre_turn_self_temporal_snapshot,
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
        environment_event: EnvironmentEvent,
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
        runtime_backbone_evidence_active = False
        runtime_backbone_signal_norm = 0.0
        runtime_backbone_signal_quality = 0.0
        runtime_backbone_signal_strength = 0.0
        runtime_backbone_hook_coverage = 0.0
        fast_memory_signal_norm = 0.0
        fast_memory_runtime_alignment = 0.0
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
            runtime_backbone_signal_norm = lifecycle_metrics.get("last_runtime_backbone_signal_norm", 0.0)
            runtime_backbone_signal_quality = lifecycle_metrics.get("last_runtime_backbone_signal_quality", 0.0)
            runtime_backbone_signal_strength = lifecycle_metrics.get("last_runtime_backbone_signal_strength", 0.0)
            runtime_backbone_hook_coverage = lifecycle_metrics.get("last_runtime_backbone_hook_coverage", 0.0)
            runtime_backbone_evidence_active = (
                lifecycle_metrics.get("last_runtime_backbone_residual_stream_active", 0.0) > 0.0
                and runtime_backbone_signal_quality > 0.0
            )
            fast_memory_signal_norm = lifecycle_metrics.get("last_fast_memory_signal_norm", 0.0)
            fast_memory_runtime_alignment = lifecycle_metrics.get("last_fast_memory_runtime_alignment", 0.0)
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

        response_assembly_snapshot = integration_result.active_snapshots.get("response_assembly")
        response_assembly = (
            response_assembly_snapshot.value
            if response_assembly_snapshot is not None and isinstance(response_assembly_snapshot.value, ResponseAssemblySnapshot)
            else None
        )
        domain_knowledge_snapshot = integration_result.active_snapshots.get("domain_knowledge")
        case_memory_snapshot = integration_result.active_snapshots.get("case_memory")
        strategy_playbook_snapshot = integration_result.active_snapshots.get("strategy_playbook")
        boundary_policy_snapshot = integration_result.active_snapshots.get("boundary_policy")
        multi_party_identity_snapshot = integration_result.active_snapshots.get(
            "multi_party_identity"
        ) or integration_result.shadow_snapshots.get("multi_party_identity")
        active_speaker_id = PRIMARY_INTERLOCUTOR_ID
        addressee_ids = (SELF_INTERLOCUTOR_ID,)
        subject_ids = (PRIMARY_INTERLOCUTOR_ID,)
        audience_ids = (SELF_INTERLOCUTOR_ID,)
        if (
            multi_party_identity_snapshot is not None
            and isinstance(multi_party_identity_snapshot.value, MultiPartyIdentitySnapshot)
        ):
            identity_scope = multi_party_identity_snapshot.value
            active_speaker_id = identity_scope.active_speaker_id
            addressee_ids = identity_scope.addressee_ids
            subject_ids = identity_scope.subject_ids
            audience_ids = identity_scope.audience_ids

        response = self._response_synthesizer.synthesize(
            context=ResponseContext(
                regime_id=active_regime,
                regime_name=regime_snapshot.value.active_regime.name
                if regime_snapshot is not None and isinstance(regime_snapshot.value, RegimeSnapshot)
                else "current context",
                regime_switched=regime_switched,
                abstract_action=active_abstract_action,
                alert_count=len(evaluation_alerts),
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
                active_speaker_id=active_speaker_id,
                addressee_ids=addressee_ids,
                subject_ids=subject_ids,
                audience_ids=audience_ids,
            ),
            assembly=response_assembly,
        )
        effective_writeback_result = deferred_writeback_result or integration_result.writeback_result
        effective_queue_state = queue_state or self.session_post_queue_state
        session_post_snapshot = self._publish_session_post_snapshot()
        experience_consolidation_snapshot = self._publish_experience_consolidation_snapshot()
        experience_fast_prior_snapshot = self._publish_experience_fast_prior_snapshot()
        evaluation_snapshot = integration_result.active_snapshots.get("evaluation")
        if (
            evaluation_snapshot is not None
            and isinstance(evaluation_snapshot.value, EvaluationSnapshot)
            and (
                experience_consolidation_snapshot.value.delayed_outcome_ledger
                or experience_consolidation_snapshot.value.sequence_payoffs
            )
        ):
            enriched_evaluation = self._evaluation_backbone.record_learning_evidence(
                session_id=self.active_context_session_id,
                wave_id=wave_id,
                timestamp_ms=evaluation_snapshot.timestamp_ms + 20,
                base_snapshot=evaluation_snapshot.value,
                memory_snapshot=memory_snapshot.value if memory_snapshot is not None and isinstance(memory_snapshot.value, MemorySnapshot) else None,
                reflection_snapshot=reflection_snapshot.value if reflection_snapshot is not None else None,
                writeback_result=integration_result.writeback_result,
                joint_loop_result=joint_result,
                regime_snapshot=regime_snapshot.value if regime_snapshot is not None and isinstance(regime_snapshot.value, RegimeSnapshot) else None,
                domain_knowledge_snapshot=(
                    domain_knowledge_snapshot.value
                    if domain_knowledge_snapshot is not None and isinstance(domain_knowledge_snapshot.value, DomainKnowledgeSnapshot)
                    else None
                ),
                case_memory_snapshot=(
                    case_memory_snapshot.value
                    if case_memory_snapshot is not None and isinstance(case_memory_snapshot.value, CaseMemorySnapshot)
                    else None
                ),
                strategy_playbook_snapshot=(
                    strategy_playbook_snapshot.value
                    if strategy_playbook_snapshot is not None and isinstance(strategy_playbook_snapshot.value, StrategyPlaybookSnapshot)
                    else None
                ),
                boundary_policy_snapshot=(
                    boundary_policy_snapshot.value
                    if boundary_policy_snapshot is not None and isinstance(boundary_policy_snapshot.value, BoundaryPolicySnapshot)
                    else None
                ),
                experience_fast_prior_snapshot=experience_fast_prior_snapshot.value,
                response_assembly_snapshot=response_assembly,
                delayed_outcome_ledger=experience_consolidation_snapshot.value.delayed_outcome_ledger,
                sequence_payoffs=experience_consolidation_snapshot.value.sequence_payoffs,
            )
            integration_result.active_snapshots["evaluation"] = Snapshot(
                slot_name=evaluation_snapshot.slot_name,
                owner=evaluation_snapshot.owner,
                version=evaluation_snapshot.version + 1,
                timestamp_ms=evaluation_snapshot.timestamp_ms + 20,
                value=enriched_evaluation,
            )
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
        if self._config.is_active("experience_fast_prior"):
            active_snapshots["experience_fast_prior"] = experience_fast_prior_snapshot
        else:
            shadow_snapshots["experience_fast_prior"] = experience_fast_prior_snapshot

        dialogue_trace, dialogue_outcome_resolution = self._dialogue_trace_store.record_action(
            session_id=self.active_context_session_id,
            wave_id=wave_id,
            turn_index=self._turn_index,
            environment_event=environment_event,
            active_regime=active_regime,
            active_abstract_action=active_abstract_action,
            response_text=response.text,
            response_rationale=response.rationale,
            next_prediction=next_prediction,
            evaluated_prediction=evaluated_prediction,
            actual_outcome=actual_outcome,
            prediction_error=prediction_error,
        )
        dialogue_trace_snapshot = self._dialogue_trace_store.snapshot()

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
            default_continual_learning_surface=joint_result.default_continual_learning_surface,
            response=response,
            event_count=integration_result.event_count,
            environment_event_id=environment_event.event_id,
            environment_event_kind=environment_event.event_kind.value,
            environment_trigger_kind=environment_event.trigger_kind,
            dialogue_trace=dialogue_trace,
            dialogue_outcome_resolution=dialogue_outcome_resolution,
            dialogue_trace_snapshot=dialogue_trace_snapshot,
            active_speaker_id=active_speaker_id,
            addressee_ids=addressee_ids,
            subject_ids=subject_ids,
            audience_ids=audience_ids,
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
            runtime_backbone_evidence_active=runtime_backbone_evidence_active,
            runtime_backbone_signal_norm=runtime_backbone_signal_norm,
            runtime_backbone_signal_quality=runtime_backbone_signal_quality,
            runtime_backbone_signal_strength=runtime_backbone_signal_strength,
            runtime_backbone_hook_coverage=runtime_backbone_hook_coverage,
            fast_memory_signal_norm=fast_memory_signal_norm,
            fast_memory_runtime_alignment=fast_memory_runtime_alignment,
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
