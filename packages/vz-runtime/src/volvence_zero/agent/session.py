from __future__ import annotations

from collections.abc import Callable, Mapping
import asyncio
from dataclasses import dataclass, replace
import os
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from volvence_zero.owner_hydration_store import OwnerHydrationStore

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
from volvence_zero.agent.response import (
    AgentResponse,
    LLMResponseSynthesizer,
    RepairExpressionAdvisory,
    ResponseContext,
    ResponseSynthesizer,
)
from volvence_zero.agent.dialogue_trace import DialogueTraceStore
from volvence_zero.agent.dialogue_outcome_producers import (
    commitment_outcome_evidence_from_commitment,
    pe_continued_evidence_from_prediction_error,
    structural_outcome_evidence_from_external,
)
from volvence_zero.dialogue_external_outcome import DialogueExternalOutcomeModule
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
    DialogueOutcomeEvidence,
)
from volvence_zero.credit.gate import (
    CreditSnapshot,
    GateDecision,
    ModificationGate,
    ModificationProposal,
    SelfModificationRecord,
    derive_dialogue_outcome_credit_records,
    extend_credit_snapshot,
    evaluate_gate,
)
from volvence_zero.evaluation import (
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
    PredictionErrorSnapshot,
)
from volvence_zero.reflection import ReflectionSnapshot, WritebackMode, WritebackResult
from volvence_zero.identity_seed import IdentitySeed
from volvence_zero.regime import RegimeBootstrap, RegimeModule, RegimeSnapshot
from volvence_zero.rupture_state import RuptureStateSnapshot
from volvence_zero.protocol_runtime import ProtocolRegistryModule
from volvence_zero.runtime import Snapshot, WiringLevel
from volvence_zero.semantic_state import (
    AdapterSemanticProposalRuntime,
    ExternalSemanticEvent,
    ExternalSemanticEventBatch,
    NoOpSemanticProposalRuntime,
    SemanticProposalRuntime,
    SemanticStateStore,
)
from volvence_zero.semantic_state.llm_runtime import LLMSemanticProposalRuntime
from volvence_zero.social import (
    LLMCommonGroundProposalRuntime,
    LLMToMProposalRuntime,
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
from volvence_zero.dual_track import DualTrackGateLearner
from volvence_zero.social import SocialRecordStore
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
# Debt #9 wave 1: ``AgentSessionRunner`` is composed from four
# phase-grouped mixins. The mixins host the bulk of method bodies so
# this file shrinks to instance setup (``__init__``), the public
# property surface, the ``run_turn`` orchestrator, and the module-
# level factories. Each mixin is a pure container of methods that
# read ``self._*`` attributes set by ``__init__``; MRO order matches
# the W5 logical phase order so any future cross-mixin ``super()``
# call resolves predictably.
from volvence_zero.agent.session_lifecycle import SessionLifecycleMixin
from volvence_zero.agent.session_observation import SessionObservationMixin
from volvence_zero.agent.session_training_phase import SessionTrainingPhaseMixin
from volvence_zero.agent.session_writeback_phase import SessionWritebackPhaseMixin


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


# W5 of ssot-cleanup-p0-p4: pure helper functions extracted to
# ``agent/session_helpers.py``. The legacy private names are kept here
# as aliases for backward compat (any code that imported the leading-
# underscore symbols directly continues to work). New code should
# import the public names from ``volvence_zero.agent.session_helpers``.
from volvence_zero.agent.session_helpers import (
    abstract_action_alignment as _abstract_action_alignment,
    application_outcome_score as _application_outcome_score,
    clamp01 as _clamp,
    regime_alignment as _regime_alignment,
    repair_expression_advisory_from_snapshots as _repair_expression_advisory_from_snapshots,
    retrieval_mix_alignment as _retrieval_mix_alignment,
)


# Debt #9 wave 1: ``_APPLICATION_PRIOR_PROPOSAL_BUILDER`` and the
# ``_experience_deltas_from_prior_update`` alias were moved to
# ``session_writeback_phase.py`` together with the methods that use
# them; nothing left in ``session.py`` references either symbol.


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


class AgentSessionRunner(
    SessionLifecycleMixin,
    SessionWritebackPhaseMixin,
    SessionTrainingPhaseMixin,
    SessionObservationMixin,
):
    """Minimal session runner over the final wiring graph.

    Implementation surface is split across four phase-grouped mixins
    (debt #9 wave 1). The MRO is ``AgentSessionRunner ->
    SessionLifecycleMixin -> SessionWritebackPhaseMixin ->
    SessionTrainingPhaseMixin -> SessionObservationMixin -> object``.
    """

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
        temporal_latent_dim: int = 3,
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
        identity_seed: IdentitySeed | None = None,
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
        dialogue_pe_continued_evidence_enabled: bool = True,
        dialogue_commitment_outcome_evidence_enabled: bool = True,
        allow_llm_outcome_proposals: bool = False,
        user_scope: str = "anonymous",
        owner_hydration_store: Any = None,
        seed_protocols: tuple[Any, ...] = (),
    ) -> None:
        if temporal_latent_dim < 3:
            raise ValueError(
                f"temporal_latent_dim must be >= 3, got {temporal_latent_dim!r}"
            )
        self._session_id = session_id
        self._dialogue_pe_continued_evidence_enabled = dialogue_pe_continued_evidence_enabled
        self._dialogue_commitment_outcome_evidence_enabled = dialogue_commitment_outcome_evidence_enabled
        # Rupture-and-Repair v0: LLM-sourced external outcome proposals
        # are OFF by default. Callers that want to experiment with an
        # LLM proposal adapter must explicitly opt in (see docs/specs/
        # rupture-and-repair.md Risk 2).
        self._allow_llm_outcome_proposals = bool(allow_llm_outcome_proposals)
        # Rupture-and-Repair v0 per-user scope: rupture-repair memory
        # entries are tagged ``user_scope:<scope>``. Default is
        # ``anonymous`` (current behavior preserved). Identified
        # sessions pass the user's scope key here through the Brain
        # facade so rupture_repair entries stay attributable.
        self._user_scope = str(user_scope) if user_scope else "anonymous"
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
            self._world_temporal_policy = FullLearnedTemporalPolicy(
                parameter_store=MetacontrollerParameterStore(
                    n_z=temporal_latent_dim
                )
            )
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
        # autograd-owner-integration: apply the configured runtime metacontroller
        # backend (DISABLED default = pure rollback baseline). Reversible via
        # FinalRolloutConfig.temporal_runtime_backend.
        _runtime_backend = self._config.temporal_runtime_backend
        if isinstance(self._world_temporal_policy, FullLearnedTemporalPolicy):
            self._world_temporal_policy.set_runtime_backend(_runtime_backend)
        if isinstance(self._self_temporal_policy, FullLearnedTemporalPolicy):
            self._self_temporal_policy.set_runtime_backend(_runtime_backend)
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
        # ``parameter_store`` is only declared on ``FullLearnedTemporalPolicy``;
        # the abstract ``TemporalPolicy`` base lacks it. Use isinstance
        # dispatch instead of getattr-default so non-learned policies
        # surface explicitly (R8 / SSOT + no-hasattr-abuse).
        if isinstance(self._world_temporal_policy, FullLearnedTemporalPolicy):
            default_latent_dim = self._world_temporal_policy.parameter_store.n_z
        else:
            default_latent_dim = 16
        self._memory_store = memory_store or build_default_memory_store(
            latent_dim=default_latent_dim,
            cms_torch_backend=self._config.cms_torch_backend,
        )
        self._semantic_state_store = SemanticStateStore()
        # Packet D (long-horizon-closure): when an OwnerHydrationStore
        # is supplied (the orchestrator built it from
        # MemoryStore.persistence_backend), eagerly hydrate the
        # SemanticStateStore from the prior session's persisted
        # snapshot so commitment / open_loop / relationship_state
        # records survive across BrainSession instances. Hydration
        # errors fail-loudly per the no-swallow rule. SHADOW mode
        # returns False from load_snapshot so this is a no-op there.
        self._owner_hydration_store = owner_hydration_store
        if self._owner_hydration_store is not None:
            self._owner_hydration_store.hydrate_owner_if_present(
                self._semantic_state_store, "semantic_state"
            )
        # ``protocol-online-learning-active`` packet (sub-packet A):
        # construct ONE stable ``ProtocolRegistryModule`` here in
        # ``__init__`` so its ``_alpha`` / ``_beta`` / ``_pe_utility``
        # / ``_strategy_weights`` learning state survives across every
        # turn of this session. The default-construction branch inside
        # ``build_final_runtime_modules`` would otherwise rebuild this
        # owner per turn and reset all the PE-driven dials —
        # mirroring the same fix applied to ``_tom_proposal_runtime``
        # / ``_common_ground_proposal_runtime`` below for debt #10B
        # item 3. Sibling owners (ProtocolPhase / Introspection /
        # RevisionLog / RevisionQueue) read ``module.registry`` so
        # threading the SAME instance via ``run_final_wiring_turn``
        # also keeps their views aligned.
        #
        # Each entry in ``seed_protocols`` is loaded right away; when
        # the application stores are injected (always true here),
        # ``load_protocol`` auto-applies the protocol's compiled
        # ``BoundaryPriorHint`` / ``PlaybookRule`` /
        # ``DomainKnowledgeRecord`` / ``CaseMemoryRecord`` artifacts
        # into the owner stores in the same call. This is the single-
        # path replacement for the SessionManager's old
        # ``with_domain_experience(...)`` injection (which went via
        # ``apply_domain_experience_packages``); both paths produce
        # the same store mutations, but the new path also enables
        # α/β learning on the seeded protocols.
        self._protocol_registry_module = ProtocolRegistryModule(
            wiring_level=self._config.level_for(
                "protocol_runtime", WiringLevel.SHADOW
            ),
            application_rare_heavy_state=self._application_rare_heavy_state,
            domain_knowledge_store=self._domain_knowledge_store,
            case_memory_store=self._case_memory_store,
        )
        for protocol in seed_protocols:
            self._protocol_registry_module.load_protocol(protocol)
        # ``protocol-online-learning-active`` packet (sub-packet C):
        # rehydrate the per-session learning state (α / β /
        # _pe_utility / _strategy_weights / _last_strategy_reward
        # / _last_pe_turn_index) from the prior session's persisted
        # snapshot when one exists. MUST run AFTER the seed
        # protocols are loaded so the hydrate path can intersect
        # the persisted strategy_weights with the currently-loaded
        # protocols' rule ids (rules that disappeared between
        # sessions are dropped; rules added stay at their
        # initial_weight). SHADOW hydration store returns False
        # from load_snapshot so this is a no-op there. Errors
        # propagate as typed ``HydrationError`` per the no-swallow
        # rule.
        if self._owner_hydration_store is not None:
            self._owner_hydration_store.hydrate_owner_if_present(
                self._protocol_registry_module, "protocol_registry"
            )
        self._semantic_proposal_runtime = semantic_proposal_runtime or NoOpSemanticProposalRuntime()
        # Wave E1 follow-up (option B / debt #10B item 3): when the
        # session is wired with an LLM-backed semantic runtime, derive
        # the ToM and common-ground LLM proposal runtimes ONCE here so
        # their typed ``LLMProposalAttemptAccumulator`` accumulates
        # across every turn of the session. The default-construction
        # branch inside ``build_final_runtime_modules`` runs per turn
        # and would otherwise reset the counters every turn (visible
        # in ``per_round_*_proposal_attempts_total = [1]`` even after
        # a 5-turn session). Constructing from the unwrapped runtime
        # here also avoids the latent secondary regression where
        # ``AdapterSemanticProposalRuntime`` (added per turn when
        # external semantic events fire) fails the strict isinstance
        # check inside ``build_final_runtime_modules`` and silently
        # fail-closes the ToM / common-ground LLM auto-wire.
        self._tom_proposal_runtime: SemanticProposalRuntime | None = None
        self._common_ground_proposal_runtime: LLMCommonGroundProposalRuntime | None = None
        if isinstance(self._semantic_proposal_runtime, LLMSemanticProposalRuntime):
            self._tom_proposal_runtime = LLMToMProposalRuntime(
                provider=self._semantic_proposal_runtime.text_provider
            )
            self._common_ground_proposal_runtime = LLMCommonGroundProposalRuntime(
                provider=self._semantic_proposal_runtime.text_provider
            )
        self._pending_semantic_events: list[ExternalSemanticEvent] = []
        self._pending_environment_outcome_id: str = ""
        # Packet A (long-horizon-closure): mirror of
        # ``_pending_environment_outcome_id`` for the prediction_id /
        # plan_ref lineage. Populated by ``remember_environment_prediction_id``
        # in the same call where the outcome id is remembered, drained at
        # the start of the next turn so the next-turn PredictionActionContext
        # can carry the affordance call's plan_ref.
        self._pending_environment_prediction_id: str = ""
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
        # Behavior Protocol Runtime packet 1.3''' (production wiring of
        # 1.3'' machinery): the identity seed flows through to
        # ``run_final_wiring_turn`` each turn, populating
        # ``DualTrackSnapshot.self_track.traits`` for the protocol
        # identity gate.
        self._identity_seed = identity_seed
        # W1.A (intent-alignment remediation): session-held gate learner so
        # the dual-track learned-gate SHADOW readout is a genuine bounded
        # online-SGD learner whose weights survive the per-turn
        # ``DualTrackModule`` rebuild (same lifetime pattern as
        # ``_tom_proposal_runtime``). Fed after each turn from the PE
        # owner's published realized outcome; report-only.
        self._dual_track_gate_learner = DualTrackGateLearner()
        # W1.C (CP-16/17): session-held ToM / common-ground record store
        # so those owners keep cross-turn records, settle prior-turn
        # predictions, and drive PE-weighted promote/retire.
        self._social_record_store = SocialRecordStore()
        self._prediction_module = PredictionErrorModule(
            wiring_level=self._config.level_for("prediction_error", WiringLevel.ACTIVE),
        )
        self._dialogue_external_outcome_module = DialogueExternalOutcomeModule(
            wiring_level=self._config.level_for(
                "dialogue_external_outcome", WiringLevel.ACTIVE
            ),
            allow_llm_outcome_proposals=self._allow_llm_outcome_proposals,
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
            ssl_backend=self._config.temporal_ssl_backend,
            internal_rl_backend=self._config.internal_rl_backend,
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
            wiring_level=self._config.level_for("session_post_slow_loop", WiringLevel.ACTIVE),
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
    def owner_hydration_store(self) -> "OwnerHydrationStore | None":
        """Public typed handle to the OwnerHydrationStore (Packet D).

        Returns ``None`` when ``BrainConfig.owner_hydration_wiring`` is
        DISABLED or when the MemoryStore has no persistence backend
        (anonymous session). Consumers (BrainSession / LifeformSession)
        read this directly per R8 / SSOT instead of poking at the
        private ``_owner_hydration_store`` field.
        """
        return self._owner_hydration_store

    @property
    def user_scope(self) -> str:
        """Per-user memory / rupture-repair scope key set at construction.

        Public R8 readout used by export / audit consumers
        (``vz-runtime.open_dialogue_artifact`` / ``lifeform-evolution``)
        instead of the private ``_user_scope`` field.
        """
        return self._user_scope

    @property
    def upstream_snapshots(self) -> Mapping[str, "Snapshot[Any]"]:
        """Read-only view of the latest ``slot_name -> Snapshot`` map
        published by the runtime graph this turn (active + shadow).

        Returns a defensive shallow copy so callers cannot mutate the
        runner's internal dict; the snapshot values themselves are
        immutable (``vz-contracts`` invariant). Public R8 readout
        replacing prior ``_upstream_snapshots`` SLF001 paths.
        """
        return dict(self._upstream_snapshots)

    @property
    def turn_index(self) -> int:
        return self._turn_index

    @property
    def temporal_latent_dim(self) -> int:
        return self._joint_loop.temporal_policy.parameter_store.n_z

    @property
    def joint_loop(self) -> ETANLJointLoop:
        """Read-only handle for evidence exporters (do not drive turns via this)."""

        return self._joint_loop

    @property
    def rollout_config(self) -> FinalRolloutConfig:
        return self._config

    @property
    def world_temporal_policy(self) -> TemporalPolicy:
        return self._world_temporal_policy

    @property
    def self_temporal_policy(self) -> TemporalPolicy:
        return self._self_temporal_policy

    @property
    def dual_track_gate_learner(self) -> DualTrackGateLearner:
        return self._dual_track_gate_learner

    @property
    def social_record_store(self) -> SocialRecordStore:
        return self._social_record_store

    @property
    def prediction_module(self) -> PredictionErrorModule:
        """Session-held PE owner. Exposed for the CP-11 gate-completeness
        surfaces (checkpoint export/restore + kill-criteria readout); the
        published PredictionErrorSnapshot remains the runtime data channel.
        """
        return self._prediction_module

    @property
    def semantic_state_store(self) -> SemanticStateStore:
        """Session-held semantic owner store (W1.B learned-forecast readout)."""
        return self._semantic_state_store

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

    # ----- session lifecycle / dialogue trace export / dialogue outcome
    # submission / semantic event enqueue / environment outcome stash /
    # context lifecycle / public rare-heavy artifact API -----
    # Implementations live in ``SessionLifecycleMixin``
    # (``session_lifecycle.py``). The methods that previously sat
    # here -- ``export_dialogue_trace_replay_artifact``,
    # ``export_snapshot_replay_artifact``,
    # ``attach_dialogue_outcome_evidence``, ``submit_dialogue_outcome``,
    # ``enqueue_semantic_events``,
    # ``_drain_pending_semantic_events``,
    # ``remember_environment_outcome``, and
    # ``_consume_pending_environment_outcome_id`` -- were moved
    # verbatim to the lifecycle mixin during the wave-1 split.

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

    # ----- writeback / lifecycle methods (12 + 4 entries) -----
    # Implementations live in ``SessionWritebackPhaseMixin``
    # (``session_writeback_phase.py``) and ``SessionLifecycleMixin``
    # (``session_lifecycle.py``). Wave-1 split moved them out
    # verbatim; the public surface is unchanged.

    async def run_turn(
        self,
        user_input: str,
        *,
        environment_event: EnvironmentEvent | None = None,
        apprenticeship_turn: bool = False,
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
            environment_outcome_id = self._consume_pending_environment_outcome_id()
            # Packet A: drain plan_ref / prediction_id buffer in the same
            # spot so it lands on this turn's PE action context together
            # with environment_outcome_id (both produced by the same
            # ``submit_tool_result`` call last turn).
            environment_prediction_id = (
                self._consume_pending_environment_prediction_id()
            )
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
                tom_proposal_runtime=self._tom_proposal_runtime,
                common_ground_proposal_runtime=self._common_ground_proposal_runtime,
                evaluation_backbone=self._evaluation_backbone,
                prior_session_reports=self.completed_session_reports,
                upstream_snapshots=self._upstream_snapshots,
                joint_loop_result=joint_result,
                environment_event=environment_event,
                environment_outcome_id=environment_outcome_id,
                environment_prediction_id=environment_prediction_id,
                apprenticeship_turn=apprenticeship_turn,
                credit_proposals=self._credit_proposals,
                reflection_mode=self._reflection_mode,
                world_temporal_policy=self._world_temporal_policy,
                self_temporal_policy=self._self_temporal_policy,
                prediction_module=self._prediction_module,
                regime_module=self._regime_module,
                dialogue_external_outcome_module=self._dialogue_external_outcome_module,
                protocol_registry_module=self._protocol_registry_module,
                user_scope=self._user_scope,
                session_id=context_session_id,
                wave_id=wave_id,
                turn_index=self._turn_index,
                apply_slow_writeback=False,
                substrate_self_mod_pe_magnitude=self._previous_prediction_magnitude,
                substrate_self_mod_pe_reward=self._previous_prediction_reward,
                substrate_self_mod_pe_threshold=self._joint_schedule.pe_substrate_online_fast_threshold,
                identity_seed=self._identity_seed,
                dual_track_gate_learner=self._dual_track_gate_learner,
                social_record_store=self._social_record_store,
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
                # W1.A: settle the previous turn's dual-track gate candidate
                # against the PE owner's published realized outcome. The
                # session reads the snapshot (never the PE module's
                # internals) and feeds the session-held learner; bootstrap
                # turns are skipped inside the learner (no prior features).
                pe_actual = integration_result.prediction_error_snapshot.actual_outcome
                if not integration_result.prediction_error_snapshot.bootstrap:
                    self._dual_track_gate_learner.observe_realized_outcome(
                        task_progress=pe_actual.task_progress,
                        relationship_delta=pe_actual.relationship_delta,
                    )
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

    # ----- training (rare-heavy + online-fast) +
    # observation (substrate adapter / training trace /
    # _to_turn_result / _run_imagination) -----
    # Implementations live in ``SessionTrainingPhaseMixin``
    # (``session_training_phase.py``) and ``SessionObservationMixin``
    # (``session_observation.py``). Wave-1 split moved them
    # out verbatim; the public surface is unchanged.

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
