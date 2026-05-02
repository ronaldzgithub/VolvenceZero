from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from volvence_zero.application.runtime import (
    ApplicationPriorUpdate,
    ApplicationPriorWritebackReport,
    ApplicationRareHeavyState,
    BoundaryPolicyModule,
    BoundaryPolicySnapshot,
    CaseMemoryModule,
    CaseMemorySnapshot,
    DomainKnowledgeModule,
    DomainKnowledgePriorUpdate,
    DomainKnowledgeSnapshot,
    ExperienceFastPriorModule,
    ExperienceFastPriorSnapshot,
    KnowledgeReviewStatus,
    KnowledgeSourceKind,
    ResponseAssemblyModule,
    ResponseAssemblySnapshot,
    RetrievalPolicyModule,
    RetrievalPolicySnapshot,
    StrategyPlaybookModule,
    StrategyPlaybookSnapshot,
    RetrievalReadoutPriorUpdate,
)
from volvence_zero.application.domain_experience import (
    DomainExperiencePackage,
    apply_domain_experience_packages,
)
from volvence_zero.application.storage import (
    ApplicationCaseMemoryStore,
    ApplicationDomainKnowledgeStore,
    DomainKnowledgeCheckpoint,
    build_default_case_memory_store,
    build_default_domain_knowledge_store,
)
from volvence_zero.credit.gate import (
    CreditSnapshot,
    CreditModule,
    GateDecision,
    ModificationGate,
    ModificationProposal,
    SelfModificationRecord,
    derive_delayed_attribution_credit_records,
    derive_learning_evidence_credit_records,
    derive_prediction_error_credit_records,
    derive_social_prediction_error_credit_records,
    has_blocking_writeback,
)
from volvence_zero.dual_track import DualTrackModule, DualTrackSnapshot
from volvence_zero.evaluation.backbone import (
    CrossSessionBenchmarkSuite,
    EvaluationBackbone,
    EvaluationReport,
    EvaluationModule,
    EvaluationScore,
    EvaluationSnapshot,
    EvolutionDecision,
    EvolutionJudgement,
    JudgementCategory,
)
from volvence_zero.environment import EnvironmentEvent
from volvence_zero.memory import MemoryModule, MemoryStore, Track, build_default_memory_store
from volvence_zero.prediction.error import (
    PredictedOutcome,
    PredictionActionContext,
    PredictionErrorModule,
    PredictionErrorSnapshot,
)
from volvence_zero.reflection import (
    ReflectionEngine,
    ReflectionModule,
    ReflectionSnapshot,
    WritebackMode,
    WritebackResult,
)
from volvence_zero.regime import RegimeModule, RegimeSnapshot
from volvence_zero.runtime import (
    EventRecorder,
    RuntimePlaceholderValue,
    SlotRegistry,
    Snapshot,
    WiringLevel,
    propagate,
)
from volvence_zero.social_cognition import (
    CommonGroundAtom,
    GroupIdentity,
    SocialPredictionError,
    SocialPredictionErrorSnapshot,
)
from volvence_zero.runtime.kernel import stable_value_hash
from volvence_zero.semantic_state import (
    SEMANTIC_OWNER_SLOTS,
    BeliefAssumptionSnapshot,
    BoundaryConsentSnapshot,
    CommitmentSnapshot,
    ExecutionResultSnapshot,
    GoalValueSnapshot,
    NoOpSemanticProposalRuntime,
    OpenLoopSnapshot,
    PlanIntentSnapshot,
    RelationshipStateSnapshot,
    SemanticProposalRuntime,
    SemanticStateStore,
    UserModelSnapshot,
    build_semantic_modules,
)
from volvence_zero.social_identity import (
    MultiPartyIdentityModule,
    SocialPredictionAggregateModule,
    SocialPredictionErrorModule,
)
from volvence_zero.social_common_ground import CommonGroundModule
from volvence_zero.social_common_ground_runtime import LLMCommonGroundProposalRuntime
from volvence_zero.social_group import GroupModule
from volvence_zero.social_role import ConversationalRoleModule
from volvence_zero.social_tom import (
    BeliefAboutOtherModule,
    FeelingAboutOtherModule,
    IntentAboutOtherModule,
    PreferenceAboutOtherModule,
)
from volvence_zero.substrate import (
    SubstrateAdapter,
    SubstrateModule,
    SubstrateSelfModModule,
    SubstrateSelfModSnapshot,
)
from volvence_zero.temporal import (
    FullLearnedTemporalPolicy,
    MetacontrollerRuntimeState,
    TemporalAbstractionSnapshot,
    TemporalConsolidationSnapshot,
    TemporalPolicy,
    TemporalAggregateModule,
    TemporalModule,
    TrackTemporalConsolidationModule,
    TrackTemporalModule,
    build_temporal_runtime_state_aggregate,
    clone_full_learned_temporal_policy,
)


@dataclass(frozen=True)
class FinalRolloutConfig:
    substrate: WiringLevel = WiringLevel.ACTIVE
    substrate_self_mod: WiringLevel = WiringLevel.ACTIVE
    memory: WiringLevel = WiringLevel.ACTIVE
    retrieval_policy: WiringLevel = WiringLevel.ACTIVE
    domain_knowledge: WiringLevel = WiringLevel.ACTIVE
    case_memory: WiringLevel = WiringLevel.ACTIVE
    strategy_playbook: WiringLevel = WiringLevel.ACTIVE
    experience_fast_prior: WiringLevel = WiringLevel.ACTIVE
    boundary_policy: WiringLevel = WiringLevel.ACTIVE
    response_assembly: WiringLevel = WiringLevel.ACTIVE
    dual_track: WiringLevel = WiringLevel.ACTIVE
    evaluation: WiringLevel = WiringLevel.ACTIVE
    prediction_error: WiringLevel = WiringLevel.ACTIVE
    regime: WiringLevel = WiringLevel.ACTIVE
    credit: WiringLevel = WiringLevel.ACTIVE
    reflection: WiringLevel = WiringLevel.ACTIVE
    temporal: WiringLevel = WiringLevel.ACTIVE
    eta_open_weight_runtime: WiringLevel = WiringLevel.SHADOW
    session_post_slow_loop: WiringLevel = WiringLevel.SHADOW
    experience_consolidation: WiringLevel = WiringLevel.ACTIVE
    plan_intent: WiringLevel = WiringLevel.ACTIVE
    commitment: WiringLevel = WiringLevel.ACTIVE
    open_loop: WiringLevel = WiringLevel.ACTIVE
    user_model: WiringLevel = WiringLevel.ACTIVE
    execution_result: WiringLevel = WiringLevel.ACTIVE
    belief_assumption: WiringLevel = WiringLevel.ACTIVE
    relationship_state: WiringLevel = WiringLevel.ACTIVE
    goal_value: WiringLevel = WiringLevel.ACTIVE
    boundary_consent: WiringLevel = WiringLevel.ACTIVE
    multi_party_identity: WiringLevel = WiringLevel.ACTIVE
    social_prediction: WiringLevel = WiringLevel.ACTIVE
    social_prediction_error: WiringLevel = WiringLevel.ACTIVE
    conversational_role: WiringLevel = WiringLevel.ACTIVE
    common_ground: WiringLevel = WiringLevel.SHADOW
    groups: WiringLevel = WiringLevel.SHADOW
    belief_about_other: WiringLevel = WiringLevel.SHADOW
    intent_about_other: WiringLevel = WiringLevel.SHADOW
    feeling_about_other: WiringLevel = WiringLevel.SHADOW
    preference_about_other: WiringLevel = WiringLevel.SHADOW
    kill_switches: frozenset[str] = frozenset()

    def level_for(self, module_name: str, default: WiringLevel) -> WiringLevel:
        if module_name in self.kill_switches:
            return WiringLevel.DISABLED
        return {
            "substrate": self.substrate,
            "substrate_self_mod": self.substrate_self_mod,
            "memory": self.memory,
            "retrieval_policy": self.retrieval_policy,
            "domain_knowledge": self.domain_knowledge,
            "case_memory": self.case_memory,
            "strategy_playbook": self.strategy_playbook,
            "experience_fast_prior": self.experience_fast_prior,
            "boundary_policy": self.boundary_policy,
            "response_assembly": self.response_assembly,
            "dual_track": self.dual_track,
            "evaluation": self.evaluation,
            "prediction_error": self.prediction_error,
            "regime": self.regime,
            "credit": self.credit,
            "reflection": self.reflection,
            "temporal": self.temporal,
            "eta_open_weight_runtime": self.eta_open_weight_runtime,
            "session_post_slow_loop": self.session_post_slow_loop,
            "experience_consolidation": self.experience_consolidation,
            "plan_intent": self.plan_intent,
            "commitment": self.commitment,
            "open_loop": self.open_loop,
            "user_model": self.user_model,
            "execution_result": self.execution_result,
            "belief_assumption": self.belief_assumption,
            "relationship_state": self.relationship_state,
            "goal_value": self.goal_value,
            "boundary_consent": self.boundary_consent,
            "multi_party_identity": self.multi_party_identity,
            "social_prediction": self.social_prediction,
            "social_prediction_error": self.social_prediction_error,
            "conversational_role": self.conversational_role,
            "common_ground": self.common_ground,
            "groups": self.groups,
            "belief_about_other": self.belief_about_other,
            "intent_about_other": self.intent_about_other,
            "feeling_about_other": self.feeling_about_other,
            "preference_about_other": self.preference_about_other,
        }.get(module_name, default)

    def is_active(self, module_name: str, default: WiringLevel = WiringLevel.DISABLED) -> bool:
        return self.level_for(module_name, default) is WiringLevel.ACTIVE


def reflection_promotion_eligible(
    *,
    evaluation_snapshot: EvaluationSnapshot,
    min_accuracy: float = 0.6,
    min_observations: int = 5,
    reflection_engine: ReflectionEngine | None = None,
) -> tuple[bool, str]:
    """Evaluate whether reflection should be promoted from SHADOW to ACTIVE.

    Returns (eligible, reason). Promotion requires:
    1. reflection_accuracy >= min_accuracy
    2. At least min_observations proposal outcomes tracked

    This is a read-only evaluation — actual promotion is a manual config change.
    The caller should separately verify via evolution judge that no
    UNSAFE_MUTATION verdicts have occurred in the relevant window.
    """
    accuracy = evaluation_snapshot.reflection_accuracy
    if reflection_engine is None:
        return (False, f"no reflection engine; accuracy={accuracy:.2f}")
    ledger = reflection_engine.proposal_outcome_ledger
    if len(ledger) < min_observations:
        return (False, f"insufficient observations: {len(ledger)}/{min_observations}")
    if accuracy < min_accuracy:
        return (False, f"accuracy {accuracy:.2f} < threshold {min_accuracy:.2f}")
    return (True, f"eligible: accuracy={accuracy:.2f} observations={len(ledger)}")


@dataclass(frozen=True)
class FinalAcceptanceReport:
    passed: bool
    active_slots: tuple[str, ...]
    shadow_slots: tuple[str, ...]
    disabled_slots: tuple[str, ...]
    issues: tuple[str, ...]
    recommendations: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class FinalIntegrationResult:
    active_snapshots: dict[str, Snapshot[Any]]
    shadow_snapshots: dict[str, Snapshot[Any]]
    acceptance_report: FinalAcceptanceReport
    event_count: int
    writeback_result: WritebackResult | None
    writeback_source: str | None
    temporal_runtime_state: MetacontrollerRuntimeState | None
    track_runtime_states: tuple[tuple[str, MetacontrollerRuntimeState], ...] = ()
    prediction_error_snapshot: PredictionErrorSnapshot | None = None
    reflection_promotion_eligible: bool = False
    reflection_promotion_reason: str = ""
    evolution_judgement: EvolutionJudgement | None = None
    cross_session_verdict: str = ""
    session_post_writeback_request: "SessionPostWritebackRequest | None" = None


@dataclass(frozen=True)
class SessionPostWritebackRequest:
    context_session_id: str
    source_wave_id: str
    session_report: EvaluationReport
    reflection_snapshot: ReflectionSnapshot
    credit_snapshot: CreditSnapshot | None
    evolution_judgement: EvolutionJudgement | None
    cross_session_verdict: str
    writeback_source: str | None
    reflection_apply_enabled: bool
    structural_writeback_allowed: bool
    checkpoint_id: str
    description: str
    semantic_state_descriptions: tuple[str, ...] = ()


def _judge_allows_structural_writeback(evolution_judgement: EvolutionJudgement | None) -> bool:
    if evolution_judgement is None:
        return True
    return evolution_judgement.decision is EvolutionDecision.PROMOTE or (
        evolution_judgement.decision is EvolutionDecision.HOLD
        and evolution_judgement.category is not JudgementCategory.UNSAFE_MUTATION
    )


def _apply_temporal_reflection_writeback(
    *,
    temporal_module: TemporalModule | None,
    reflection_snapshot: Snapshot[Any] | None,
    credit_snapshot: CreditSnapshot | None,
    timestamp_ms: int,
    apply_enabled: bool,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[SelfModificationRecord, ...]]:
    if (
        temporal_module is None
        or reflection_snapshot is None
        or not isinstance(reflection_snapshot.value, ReflectionSnapshot)
    ):
        return ((), (), ())
    temporal_prior_update = reflection_snapshot.value.policy_consolidation.temporal_prior_update
    if temporal_prior_update is None:
        return ((), (), ())
    target_groups = temporal_prior_update.target_groups or ("base-weights",)
    before_hash = stable_value_hash(temporal_module.policy.export_parameters())
    if not apply_enabled:
        blocked_operations = ("temporal-prior:writeback-mode-not-apply",)
        return (
            (),
            blocked_operations,
            tuple(
                SelfModificationRecord(
                    target=f"{temporal_prior_update.target}.{group}",
                    gate=ModificationGate.BACKGROUND,
                    decision=GateDecision.BLOCK,
                    old_value_hash=before_hash,
                    new_value_hash=before_hash,
                    justification="Reflection-to-temporal writeback skipped because reflection apply mode is disabled.",
                    timestamp_ms=timestamp_ms,
                    is_reversible=True,
                )
                for group in target_groups
            ),
        )
    blocked_groups = (
        tuple(
            group
            for group in target_groups
            if credit_snapshot is not None
            and has_blocking_writeback(
                credit_snapshot,
                target_prefix=f"{temporal_prior_update.target}.{group}",
            )
        )
    )
    allowed_groups = tuple(group for group in target_groups if group not in blocked_groups)
    if not allowed_groups:
        blocked_operations = ("temporal-prior:credit-gate-block",)
        return (
            (),
            blocked_operations,
            tuple(
                SelfModificationRecord(
                    target=f"{temporal_prior_update.target}.{group}",
                    gate=ModificationGate.BACKGROUND,
                    decision=GateDecision.BLOCK,
                    old_value_hash=before_hash,
                    new_value_hash=before_hash,
                    justification="Reflection-to-temporal writeback blocked by target-specific credit gate.",
                    timestamp_ms=timestamp_ms,
                    is_reversible=True,
                )
                for group in blocked_groups
            ),
        )
    applied_operations = temporal_module.policy.apply_reflection_prior_update(
        update=temporal_prior_update,
        allowed_target_groups=allowed_groups,
    )
    blocked_operations = ("temporal-prior:partial-credit-gate-block",) if blocked_groups else ()
    after_hash = stable_value_hash(temporal_module.policy.export_parameters())
    audit_records = tuple(
        [
            SelfModificationRecord(
                target=f"{temporal_prior_update.target}.{group}",
                gate=ModificationGate.BACKGROUND,
                decision=GateDecision.ALLOW,
                old_value_hash=before_hash,
                new_value_hash=after_hash,
                justification=temporal_prior_update.description,
                timestamp_ms=timestamp_ms,
                is_reversible=True,
            )
            for group in allowed_groups
        ]
        + [
            SelfModificationRecord(
                target=f"{temporal_prior_update.target}.{group}",
                gate=ModificationGate.BACKGROUND,
                decision=GateDecision.BLOCK,
                old_value_hash=before_hash,
                new_value_hash=before_hash,
                justification="Reflection-to-temporal writeback blocked by target-specific credit gate.",
                timestamp_ms=timestamp_ms,
                is_reversible=True,
            )
            for group in blocked_groups
        ]
    )
    return (applied_operations, blocked_operations, audit_records)


def _build_session_post_writeback_request(
    *,
    context_session_id: str,
    source_wave_id: str,
    session_report: EvaluationReport | None,
    reflection_snapshot: Snapshot[Any] | None,
    credit_snapshot: Snapshot[Any] | None,
    evolution_judgement: EvolutionJudgement | None,
    cross_session_verdict: str,
    writeback_source: str | None,
    reflection_mode: WritebackMode,
    structural_writeback_allowed: bool,
    semantic_state_values: tuple[object, ...] = (),
) -> SessionPostWritebackRequest | None:
    if (
        session_report is None
        or reflection_snapshot is None
        or not isinstance(reflection_snapshot.value, ReflectionSnapshot)
    ):
        return None
    credit_value = (
        credit_snapshot.value
        if credit_snapshot is not None and isinstance(credit_snapshot.value, CreditSnapshot)
        else None
    )
    return SessionPostWritebackRequest(
        context_session_id=context_session_id,
        source_wave_id=source_wave_id,
        session_report=session_report,
        reflection_snapshot=reflection_snapshot.value,
        credit_snapshot=credit_value,
        evolution_judgement=evolution_judgement,
        cross_session_verdict=cross_session_verdict,
        writeback_source=writeback_source,
        reflection_apply_enabled=reflection_mode is WritebackMode.APPLY,
        structural_writeback_allowed=structural_writeback_allowed,
        checkpoint_id=f"{context_session_id}:{source_wave_id}:session-post",
        description=(
            f"Deferred session-post slow loop request for {context_session_id} from {source_wave_id}; "
            f"apply={'on' if reflection_mode is WritebackMode.APPLY else 'off'} "
            f"structural={'allow' if structural_writeback_allowed else 'block'}."
        ),
        semantic_state_descriptions=tuple(
            description
            for value in semantic_state_values
            for description in (getattr(value, "description", ""),)
            if isinstance(description, str) and description
        ),
    )


def _apply_application_prior_writeback(
    *,
    prior_update: ApplicationPriorUpdate | None,
    domain_knowledge_store: ApplicationDomainKnowledgeStore,
    case_memory_store: ApplicationCaseMemoryStore,
    application_rare_heavy_state: ApplicationRareHeavyState,
    credit_snapshot: CreditSnapshot | None,
    timestamp_ms: int,
    checkpoint_id: str,
    apply_enabled: bool,
    retrieval_apply_enabled: bool | None = None,
    blocked_reason: str,
    domain_knowledge_pre_checkpoint_out: list[DomainKnowledgeCheckpoint] | None = None,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[SelfModificationRecord, ...],
    ApplicationPriorWritebackReport | None,
]:
    if prior_update is None:
        return ((), (), (), None)
    resolved_retrieval_apply_enabled = apply_enabled if retrieval_apply_enabled is None else retrieval_apply_enabled
    resolved_domain_apply_enabled = resolved_retrieval_apply_enabled
    proposed_targets = tuple(
        update.target
        for update in (
            *prior_update.case_memory_updates,
            *prior_update.strategy_playbook_updates,
            *prior_update.boundary_policy_updates,
            *prior_update.domain_knowledge_updates,
            *prior_update.retrieval_readout_updates,
        )
    )
    if not proposed_targets:
        return (
            (),
            (),
            (),
            ApplicationPriorWritebackReport(
                proposed_target_count=0,
                applied_targets=(),
                blocked_targets=(),
                audit_record_count=0,
                description="Application prior update contained no targets.",
            ),
        )
    applied_operations: list[str] = []
    blocked_operations: list[str] = []
    audit_records: list[SelfModificationRecord] = []
    applied_targets: list[str] = []
    blocked_targets: list[str] = []

    def current_case_hash() -> str:
        return stable_value_hash(case_memory_store.records)

    def current_domain_knowledge_hash() -> str:
        return stable_value_hash(domain_knowledge_store.records)

    def current_playbook_hash() -> str:
        return stable_value_hash(application_rare_heavy_state.distilled_playbook_rules)

    def current_boundary_hash() -> str:
        return stable_value_hash(application_rare_heavy_state.boundary_prior_hints)

    def current_retrieval_readout_hash() -> str:
        return stable_value_hash(application_rare_heavy_state.retrieval_readout_checkpoint)

    def should_block_target(target: str) -> bool:
        return credit_snapshot is not None and has_blocking_writeback(
            credit_snapshot,
            target_prefix=target,
        )

    def should_block_domain_knowledge_prior(update: DomainKnowledgePriorUpdate) -> tuple[bool, str]:
        if update.review_status is not KnowledgeReviewStatus.APPROVED:
            return True, "knowledge-review-not-approved"
        if update.source_kind is KnowledgeSourceKind.CONVERSATION:
            if not update.citation_ids:
                return True, "knowledge-citation-missing"
            if update.record.locator == "surface-fallback":
                return True, "knowledge-fallback-record-blocked"
        return False, "allow"

    domain_pre_checkpoint_captured = False

    for update in prior_update.case_memory_updates:
        before_hash = current_case_hash()
        if not apply_enabled:
            blocked_targets.append(update.target)
            blocked_operations.append(f"application-prior:block:{update.target}:{blocked_reason}")
            audit_records.append(
                SelfModificationRecord(
                    target=update.target,
                    gate=ModificationGate.BACKGROUND,
                    decision=GateDecision.BLOCK,
                    old_value_hash=before_hash,
                    new_value_hash=before_hash,
                    justification=(
                        "Application case-memory prior writeback skipped because "
                        f"{blocked_reason.replace('-', ' ')}."
                    ),
                    timestamp_ms=timestamp_ms,
                    is_reversible=True,
                    checkpoint_id=checkpoint_id,
                )
            )
            continue
        if should_block_target(update.target):
            blocked_targets.append(update.target)
            blocked_operations.append(f"application-prior:block:{update.target}:credit-gate-block")
            audit_records.append(
                SelfModificationRecord(
                    target=update.target,
                    gate=ModificationGate.BACKGROUND,
                    decision=GateDecision.BLOCK,
                    old_value_hash=before_hash,
                    new_value_hash=before_hash,
                    justification="Application case-memory prior writeback blocked by target-specific credit gate.",
                    timestamp_ms=timestamp_ms,
                    is_reversible=True,
                    checkpoint_id=checkpoint_id,
                )
            )
            continue
        case_memory_store.upsert_records((update.record,))
        case_memory_store.save_to_backend()
        after_hash = current_case_hash()
        applied_targets.append(update.target)
        applied_operations.append(f"application-prior:case-memory:{update.record.case_id}")
        audit_records.append(
            SelfModificationRecord(
                target=update.target,
                gate=ModificationGate.BACKGROUND,
                decision=GateDecision.ALLOW,
                old_value_hash=before_hash,
                new_value_hash=after_hash,
                justification=update.description,
                timestamp_ms=timestamp_ms,
                is_reversible=True,
                checkpoint_id=checkpoint_id,
            )
        )

    for update in prior_update.domain_knowledge_updates:
        before_hash = current_domain_knowledge_hash()
        if not resolved_domain_apply_enabled:
            blocked_targets.append(update.target)
            blocked_operations.append(f"application-prior:block:{update.target}:{blocked_reason}")
            audit_records.append(
                SelfModificationRecord(
                    target=update.target,
                    gate=ModificationGate.BACKGROUND,
                    decision=GateDecision.BLOCK,
                    old_value_hash=before_hash,
                    new_value_hash=before_hash,
                    justification=(
                        "Application domain-knowledge prior writeback skipped because "
                        f"{blocked_reason.replace('-', ' ')}."
                    ),
                    timestamp_ms=timestamp_ms,
                    is_reversible=True,
                    checkpoint_id=checkpoint_id,
                )
            )
            continue
        if should_block_target(update.target):
            blocked_targets.append(update.target)
            blocked_operations.append(f"application-prior:block:{update.target}:credit-gate-block")
            audit_records.append(
                SelfModificationRecord(
                    target=update.target,
                    gate=ModificationGate.BACKGROUND,
                    decision=GateDecision.BLOCK,
                    old_value_hash=before_hash,
                    new_value_hash=before_hash,
                    justification="Application domain-knowledge prior writeback blocked by target-specific credit gate.",
                    timestamp_ms=timestamp_ms,
                    is_reversible=True,
                    checkpoint_id=checkpoint_id,
                )
            )
            continue
        schema_block, schema_reason = should_block_domain_knowledge_prior(update)
        if schema_block:
            blocked_targets.append(update.target)
            blocked_operations.append(f"application-prior:block:{update.target}:{schema_reason}")
            audit_records.append(
                SelfModificationRecord(
                    target=update.target,
                    gate=ModificationGate.BACKGROUND,
                    decision=GateDecision.BLOCK,
                    old_value_hash=before_hash,
                    new_value_hash=before_hash,
                    justification=(
                        "Application domain-knowledge prior writeback blocked by shared knowledge gate "
                        f"({schema_reason.replace('-', ' ')})."
                    ),
                    timestamp_ms=timestamp_ms,
                    is_reversible=True,
                    checkpoint_id=checkpoint_id,
                )
            )
            continue
        if domain_knowledge_pre_checkpoint_out is not None and not domain_pre_checkpoint_captured:
            domain_knowledge_pre_checkpoint_out.append(
                domain_knowledge_store.create_checkpoint(
                    checkpoint_id=f"{checkpoint_id}:domain-knowledge-pre",
                )
            )
            domain_pre_checkpoint_captured = True
        domain_knowledge_store.upsert_records((update.record,))
        domain_knowledge_store.save_to_backend()
        after_hash = current_domain_knowledge_hash()
        applied_targets.append(update.target)
        applied_operations.append(f"application-prior:domain-knowledge:{update.record.record_id}")
        audit_records.append(
            SelfModificationRecord(
                target=update.target,
                gate=ModificationGate.BACKGROUND,
                decision=GateDecision.ALLOW,
                old_value_hash=before_hash,
                new_value_hash=after_hash,
                justification=update.description,
                timestamp_ms=timestamp_ms,
                is_reversible=True,
                checkpoint_id=checkpoint_id,
            )
        )

    for update in prior_update.strategy_playbook_updates:
        before_hash = current_playbook_hash()
        if not apply_enabled:
            blocked_targets.append(update.target)
            blocked_operations.append(f"application-prior:block:{update.target}:{blocked_reason}")
            audit_records.append(
                SelfModificationRecord(
                    target=update.target,
                    gate=ModificationGate.BACKGROUND,
                    decision=GateDecision.BLOCK,
                    old_value_hash=before_hash,
                    new_value_hash=before_hash,
                    justification=(
                        "Application playbook prior writeback skipped because "
                        f"{blocked_reason.replace('-', ' ')}."
                    ),
                    timestamp_ms=timestamp_ms,
                    is_reversible=True,
                    checkpoint_id=checkpoint_id,
                )
            )
            continue
        if should_block_target(update.target):
            blocked_targets.append(update.target)
            blocked_operations.append(f"application-prior:block:{update.target}:credit-gate-block")
            audit_records.append(
                SelfModificationRecord(
                    target=update.target,
                    gate=ModificationGate.BACKGROUND,
                    decision=GateDecision.BLOCK,
                    old_value_hash=before_hash,
                    new_value_hash=before_hash,
                    justification="Application playbook prior writeback blocked by target-specific credit gate.",
                    timestamp_ms=timestamp_ms,
                    is_reversible=True,
                    checkpoint_id=checkpoint_id,
                )
            )
            continue
        application_rare_heavy_state.upsert_distilled_playbook_rules((update.rule,))
        after_hash = current_playbook_hash()
        applied_targets.append(update.target)
        applied_operations.append(f"application-prior:playbook:{update.rule.problem_pattern}")
        audit_records.append(
            SelfModificationRecord(
                target=update.target,
                gate=ModificationGate.BACKGROUND,
                decision=GateDecision.ALLOW,
                old_value_hash=before_hash,
                new_value_hash=after_hash,
                justification=update.description,
                timestamp_ms=timestamp_ms,
                is_reversible=True,
                checkpoint_id=checkpoint_id,
            )
        )

    for update in prior_update.boundary_policy_updates:
        before_hash = current_boundary_hash()
        if not apply_enabled:
            blocked_targets.append(update.target)
            blocked_operations.append(f"application-prior:block:{update.target}:{blocked_reason}")
            audit_records.append(
                SelfModificationRecord(
                    target=update.target,
                    gate=ModificationGate.BACKGROUND,
                    decision=GateDecision.BLOCK,
                    old_value_hash=before_hash,
                    new_value_hash=before_hash,
                    justification=(
                        "Application boundary prior writeback skipped because "
                        f"{blocked_reason.replace('-', ' ')}."
                    ),
                    timestamp_ms=timestamp_ms,
                    is_reversible=True,
                    checkpoint_id=checkpoint_id,
                )
            )
            continue
        if should_block_target(update.target):
            blocked_targets.append(update.target)
            blocked_operations.append(f"application-prior:block:{update.target}:credit-gate-block")
            audit_records.append(
                SelfModificationRecord(
                    target=update.target,
                    gate=ModificationGate.BACKGROUND,
                    decision=GateDecision.BLOCK,
                    old_value_hash=before_hash,
                    new_value_hash=before_hash,
                    justification="Application boundary prior writeback blocked by target-specific credit gate.",
                    timestamp_ms=timestamp_ms,
                    is_reversible=True,
                    checkpoint_id=checkpoint_id,
                )
            )
            continue
        application_rare_heavy_state.upsert_boundary_prior_hints((update.hint,))
        after_hash = current_boundary_hash()
        applied_targets.append(update.target)
        applied_operations.append(f"application-prior:boundary:{update.hint.hint_id}")
        audit_records.append(
            SelfModificationRecord(
                target=update.target,
                gate=ModificationGate.BACKGROUND,
                decision=GateDecision.ALLOW,
                old_value_hash=before_hash,
                new_value_hash=after_hash,
                justification=update.description,
                timestamp_ms=timestamp_ms,
                is_reversible=True,
                checkpoint_id=checkpoint_id,
            )
        )

    for update in prior_update.retrieval_readout_updates:
        before_hash = current_retrieval_readout_hash()
        if not resolved_retrieval_apply_enabled:
            blocked_targets.append(update.target)
            blocked_operations.append(f"application-prior:block:{update.target}:{blocked_reason}")
            audit_records.append(
                SelfModificationRecord(
                    target=update.target,
                    gate=ModificationGate.BACKGROUND,
                    decision=GateDecision.BLOCK,
                    old_value_hash=before_hash,
                    new_value_hash=before_hash,
                    justification=(
                        "Application retrieval readout checkpoint writeback skipped because "
                        f"{blocked_reason.replace('-', ' ')}."
                    ),
                    timestamp_ms=timestamp_ms,
                    is_reversible=True,
                    checkpoint_id=checkpoint_id,
                )
            )
            continue
        if should_block_target(update.target):
            blocked_targets.append(update.target)
            blocked_operations.append(f"application-prior:block:{update.target}:credit-gate-block")
            audit_records.append(
                SelfModificationRecord(
                    target=update.target,
                    gate=ModificationGate.BACKGROUND,
                    decision=GateDecision.BLOCK,
                    old_value_hash=before_hash,
                    new_value_hash=before_hash,
                    justification="Application retrieval readout checkpoint writeback blocked by target-specific credit gate.",
                    timestamp_ms=timestamp_ms,
                    is_reversible=True,
                    checkpoint_id=checkpoint_id,
                )
            )
            continue
        application_rare_heavy_state.apply_retrieval_readout_checkpoint(update.checkpoint)
        after_hash = current_retrieval_readout_hash()
        applied_targets.append(update.target)
        applied_operations.append(f"application-prior:retrieval-readout:{update.checkpoint.checkpoint_id}")
        audit_records.append(
            SelfModificationRecord(
                target=update.target,
                gate=ModificationGate.BACKGROUND,
                decision=GateDecision.ALLOW,
                old_value_hash=before_hash,
                new_value_hash=after_hash,
                justification=update.description,
                timestamp_ms=timestamp_ms,
                is_reversible=True,
                checkpoint_id=checkpoint_id,
            )
        )

    return (
        tuple(applied_operations),
        tuple(blocked_operations),
        tuple(audit_records),
        ApplicationPriorWritebackReport(
            proposed_target_count=len(proposed_targets),
            applied_targets=tuple(applied_targets),
            blocked_targets=tuple(blocked_targets),
            audit_record_count=len(audit_records),
            description=(
                f"Application prior writeback proposed={len(proposed_targets)} "
                f"applied={len(applied_targets)} blocked={len(blocked_targets)}."
            ),
        ),
    )


def _prune_disabled_stub_snapshots(snapshots: dict[str, Snapshot[Any]]) -> dict[str, Snapshot[Any]]:
    return {
        slot_name: snapshot
        for slot_name, snapshot in snapshots.items()
        if not (
            isinstance(snapshot.value, RuntimePlaceholderValue)
            and snapshot.value.reason == "disabled-module"
        )
    }


def apply_session_post_writeback_request(
    *,
    request: SessionPostWritebackRequest,
    memory_store: MemoryStore,
    temporal_policy: TemporalPolicy | None,
    regime_module: RegimeModule | None,
) -> tuple[WritebackResult | None, tuple[SelfModificationRecord, ...]]:
    structural_apply_enabled = (
        request.reflection_apply_enabled and request.structural_writeback_allowed
    )
    if request.reflection_apply_enabled:
        writeback_result = ReflectionEngine(writeback_mode=WritebackMode.APPLY).apply(
            memory_store=memory_store,
            reflection_snapshot=request.reflection_snapshot,
            credit_snapshot=request.credit_snapshot,
            regime_module=regime_module if structural_apply_enabled else None,
            checkpoint_id=request.checkpoint_id,
        )
        if (
            not structural_apply_enabled
            and writeback_result is not None
            and not writeback_result.blocked_operations
        ):
            writeback_result = replace(
                writeback_result,
                blocked_operations=writeback_result.blocked_operations
                + ("evolution-judge-block:structural-only",),
                description=(
                    f"{writeback_result.description} "
                    "Structural writeback was gated by the evolution judge; "
                    "memory-owner bounded consolidation still applied."
                ),
            )
    else:
        writeback_result = WritebackResult(
            applied_operations=(),
            blocked_operations=("writeback-mode-not-apply",),
            checkpoint=None,
            description="Session-post reflection writeback is disabled because reflection apply mode is not active.",
        )
    temporal_module = TemporalModule(policy=temporal_policy, wiring_level=WiringLevel.ACTIVE)
    reflection_snapshot = Snapshot(
        slot_name="reflection",
        owner="ReflectionModule",
        version=0,
        timestamp_ms=request.session_report.timestamp_ms,
        value=request.reflection_snapshot,
    )
    temporal_writeback_operations, temporal_writeback_blocks, temporal_audits = _apply_temporal_reflection_writeback(
        temporal_module=temporal_module,
        reflection_snapshot=reflection_snapshot,
        credit_snapshot=request.credit_snapshot,
        timestamp_ms=request.session_report.timestamp_ms + 1,
        apply_enabled=structural_apply_enabled,
    )
    if (temporal_writeback_operations or temporal_writeback_blocks) and writeback_result is not None:
        writeback_result = replace(
            writeback_result,
            applied_operations=writeback_result.applied_operations + temporal_writeback_operations,
            blocked_operations=writeback_result.blocked_operations + temporal_writeback_blocks,
            description=(
                f"{writeback_result.description} temporal_ops={len(temporal_writeback_operations)} "
                f"temporal_blocks={len(temporal_writeback_blocks)}."
            ),
        )
    elif temporal_writeback_operations or temporal_writeback_blocks:
        writeback_result = WritebackResult(
            applied_operations=temporal_writeback_operations,
            blocked_operations=temporal_writeback_blocks,
            checkpoint=None,
            description=(
                f"Session-post temporal writeback produced {len(temporal_writeback_operations)} applied ops and "
                f"{len(temporal_writeback_blocks)} blocked ops."
            ),
        )
    return (writeback_result, temporal_audits)


def _prediction_action_context_from_upstream(
    *,
    upstream_snapshots: dict[str, Snapshot[Any]] | None,
    environment_event: EnvironmentEvent | None,
) -> PredictionActionContext:
    temporal_snapshot = (
        upstream_snapshots.get("temporal_abstraction")
        if upstream_snapshots is not None
        else None
    )
    temporal_value = (
        temporal_snapshot.value
        if temporal_snapshot is not None
        and isinstance(temporal_snapshot.value, TemporalAbstractionSnapshot)
        else None
    )
    segment = next(iter(temporal_value.closed_segments), None) if temporal_value else None
    return PredictionActionContext(
        segment_id=segment.segment_id if segment is not None else "",
        abstract_action_id=(
            segment.abstract_action_id
            if segment is not None
            else (temporal_value.active_abstract_action if temporal_value is not None else "")
        ),
        z_t_digest=segment.z_t_digest if segment is not None else (),
        regime_id="",
        affordance_name=(segment.affordance_name or "") if segment is not None else "",
        environment_event_id=environment_event.event_id if environment_event is not None else "",
    )


def build_final_runtime_modules(
    *,
    config: FinalRolloutConfig,
    substrate_adapter: SubstrateAdapter,
    user_input: str | None = None,
    application_rare_heavy_state: ApplicationRareHeavyState | None = None,
    domain_knowledge_store: ApplicationDomainKnowledgeStore | None = None,
    case_memory_store: ApplicationCaseMemoryStore | None = None,
    domain_experience_packages: tuple[DomainExperiencePackage, ...] = (),
    memory_store: MemoryStore | None = None,
    semantic_state_store: SemanticStateStore | None = None,
    semantic_proposal_runtime: SemanticProposalRuntime | None = None,
    tom_proposal_runtime: SemanticProposalRuntime | None = None,
    common_ground_proposal_runtime: LLMCommonGroundProposalRuntime | None = None,
    evaluation_backbone: EvaluationBackbone | None = None,
    credit_proposals: tuple[ModificationProposal, ...] = (),
    reflection_mode: WritebackMode = WritebackMode.PROPOSAL_ONLY,
    temporal_policy: TemporalPolicy | None = None,
    world_temporal_policy: TemporalPolicy | None = None,
    self_temporal_policy: TemporalPolicy | None = None,
    prediction_module: PredictionErrorModule | None = None,
    regime_module: RegimeModule | None = None,
    session_id: str = "runtime-session",
    wave_id: str = "wave-0",
    turn_index: int = 0,
    substrate_self_mod_pe_magnitude: float = 0.0,
    substrate_self_mod_pe_reward: float = 0.0,
    substrate_self_mod_pe_threshold: float = 0.18,
    social_prediction_errors: tuple[SocialPredictionError, ...] = (),
    environment_event: EnvironmentEvent | None = None,
    prediction_action_context: PredictionActionContext | None = None,
    common_ground_dyad_atoms: tuple[CommonGroundAtom, ...] = (),
    common_ground_group_atoms: tuple[CommonGroundAtom, ...] = (),
    group_identities: tuple[GroupIdentity, ...] = (),
    active_group_id: str | None = None,
    group_joint_attention: tuple[str, ...] = (),
    group_joint_commitments: tuple[str, ...] = (),
    group_regime_id: str | None = None,
) -> list[Any]:
    if domain_experience_packages:
        application_rare_heavy_state = application_rare_heavy_state or ApplicationRareHeavyState()
        domain_knowledge_store = domain_knowledge_store or build_default_domain_knowledge_store()
        case_memory_store = case_memory_store or build_default_case_memory_store()
        apply_domain_experience_packages(
            packages=domain_experience_packages,
            domain_knowledge_store=domain_knowledge_store,
            case_memory_store=case_memory_store,
            application_rare_heavy_state=application_rare_heavy_state,
        )
    resolved_world_temporal_policy = world_temporal_policy or temporal_policy or FullLearnedTemporalPolicy()
    if self_temporal_policy is not None:
        resolved_self_temporal_policy = self_temporal_policy
    elif isinstance(resolved_world_temporal_policy, FullLearnedTemporalPolicy):
        resolved_self_temporal_policy = clone_full_learned_temporal_policy(resolved_world_temporal_policy)
    else:
        resolved_self_temporal_policy = resolved_world_temporal_policy
    semantic_store = semantic_state_store or SemanticStateStore()
    semantic_runtime = semantic_proposal_runtime or NoOpSemanticProposalRuntime()
    return [
        SubstrateModule(
            adapter=substrate_adapter,
            wiring_level=config.level_for("substrate", WiringLevel.ACTIVE),
        ),
        SubstrateSelfModModule(
            session_id=session_id,
            wave_id=wave_id,
            pe_threshold=substrate_self_mod_pe_threshold,
            external_pe_magnitude=substrate_self_mod_pe_magnitude,
            external_pe_reward=substrate_self_mod_pe_reward,
            wiring_level=config.level_for("substrate_self_mod", WiringLevel.SHADOW),
        ),
        MultiPartyIdentityModule(
            environment_event=environment_event,
            wiring_level=config.level_for("multi_party_identity", WiringLevel.SHADOW),
        ),
        MemoryModule(
            store=memory_store or build_default_memory_store(),
            wiring_level=config.level_for("memory", WiringLevel.SHADOW),
            user_text=user_input,
        ),
        SocialPredictionAggregateModule(
            wiring_level=config.level_for("social_prediction", WiringLevel.SHADOW),
        ),
        SocialPredictionErrorModule(
            wiring_level=config.level_for("social_prediction_error", WiringLevel.SHADOW),
            pending_errors=social_prediction_errors,
        ),
        ConversationalRoleModule(
            wiring_level=config.level_for("conversational_role", WiringLevel.SHADOW),
            environment_event=environment_event,
        ),
        BeliefAboutOtherModule(
            wiring_level=config.level_for("belief_about_other", WiringLevel.SHADOW),
            proposal_runtime=tom_proposal_runtime,
            user_input=user_input,
            turn_index=turn_index,
        ),
        IntentAboutOtherModule(
            wiring_level=config.level_for("intent_about_other", WiringLevel.SHADOW),
            proposal_runtime=tom_proposal_runtime,
            user_input=user_input,
            turn_index=turn_index,
        ),
        FeelingAboutOtherModule(
            wiring_level=config.level_for("feeling_about_other", WiringLevel.SHADOW),
            proposal_runtime=tom_proposal_runtime,
            user_input=user_input,
            turn_index=turn_index,
        ),
        PreferenceAboutOtherModule(
            wiring_level=config.level_for("preference_about_other", WiringLevel.SHADOW),
            proposal_runtime=tom_proposal_runtime,
            user_input=user_input,
            turn_index=turn_index,
        ),
        CommonGroundModule(
            wiring_level=config.level_for("common_ground", WiringLevel.SHADOW),
            dyad_atoms=common_ground_dyad_atoms,
            group_atoms=common_ground_group_atoms,
            proposal_runtime=common_ground_proposal_runtime,
            user_input=user_input,
            turn_index=turn_index,
        ),
        GroupModule(
            wiring_level=config.level_for("groups", WiringLevel.SHADOW),
            groups=group_identities,
            active_group_id=active_group_id,
            joint_attention=group_joint_attention,
            joint_commitments=group_joint_commitments,
            group_regime_id=group_regime_id,
        ),
        *build_semantic_modules(
            store=semantic_store,
            proposal_runtime=semantic_runtime,
            user_input=user_input,
            turn_index=turn_index,
            level_for=config.level_for,
        ),
        TrackTemporalModule(
            track=Track.WORLD,
            policy=resolved_world_temporal_policy,
            wiring_level=config.level_for("temporal", WiringLevel.SHADOW),
        ),
        TrackTemporalModule(
            track=Track.SELF,
            policy=resolved_self_temporal_policy,
            wiring_level=config.level_for("temporal", WiringLevel.SHADOW),
        ),
        TemporalAggregateModule(
            wiring_level=config.level_for("temporal", WiringLevel.SHADOW),
        ),
        DualTrackModule(
            wiring_level=config.level_for("dual_track", WiringLevel.SHADOW),
        ),
        ExperienceFastPriorModule(
            wiring_level=config.level_for("experience_fast_prior", WiringLevel.SHADOW),
        ),
        regime_module
        or RegimeModule(
            wiring_level=config.level_for("regime", WiringLevel.SHADOW),
        ),
        RetrievalPolicyModule(
            rare_heavy_state=application_rare_heavy_state,
            wiring_level=config.level_for("retrieval_policy", WiringLevel.ACTIVE),
        ),
        DomainKnowledgeModule(
            rare_heavy_state=application_rare_heavy_state,
            store=domain_knowledge_store,
            wiring_level=config.level_for("domain_knowledge", WiringLevel.ACTIVE),
        ),
        EvaluationModule(
            backbone=evaluation_backbone,
            session_id=session_id,
            wave_id=wave_id,
            wiring_level=config.level_for("evaluation", WiringLevel.ACTIVE),
        ),
        prediction_module
        or PredictionErrorModule(
            wiring_level=config.level_for("prediction_error", WiringLevel.ACTIVE),
            action_context=prediction_action_context,
        ),
        CaseMemoryModule(
            rare_heavy_state=application_rare_heavy_state,
            store=case_memory_store,
            wiring_level=config.level_for("case_memory", WiringLevel.SHADOW),
        ),
        StrategyPlaybookModule(
            rare_heavy_state=application_rare_heavy_state,
            wiring_level=config.level_for("strategy_playbook", WiringLevel.SHADOW),
        ),
        BoundaryPolicyModule(
            rare_heavy_state=application_rare_heavy_state,
            wiring_level=config.level_for("boundary_policy", WiringLevel.ACTIVE),
        ),
        ResponseAssemblyModule(
            wiring_level=config.level_for("response_assembly", WiringLevel.ACTIVE),
        ),
        CreditModule(
            pending_proposals=credit_proposals,
            wiring_level=config.level_for("credit", WiringLevel.SHADOW),
        ),
        ReflectionModule(
            engine=ReflectionEngine(writeback_mode=reflection_mode),
            wiring_level=config.level_for("reflection", WiringLevel.SHADOW),
        ),
        TrackTemporalConsolidationModule(
            track=Track.WORLD,
            policy=resolved_world_temporal_policy,
            wiring_level=config.level_for("temporal", WiringLevel.SHADOW),
        ),
        TrackTemporalConsolidationModule(
            track=Track.SELF,
            policy=resolved_self_temporal_policy,
            wiring_level=config.level_for("temporal", WiringLevel.SHADOW),
        ),
    ]


async def run_final_wiring_turn(
    *,
    config: FinalRolloutConfig,
    substrate_adapter: SubstrateAdapter,
    user_input: str | None = None,
    application_rare_heavy_state: ApplicationRareHeavyState | None = None,
    domain_knowledge_store: ApplicationDomainKnowledgeStore | None = None,
    case_memory_store: ApplicationCaseMemoryStore | None = None,
    domain_experience_packages: tuple[DomainExperiencePackage, ...] = (),
    memory_store: MemoryStore | None = None,
    semantic_state_store: SemanticStateStore | None = None,
    semantic_proposal_runtime: SemanticProposalRuntime | None = None,
    tom_proposal_runtime: SemanticProposalRuntime | None = None,
    common_ground_proposal_runtime: LLMCommonGroundProposalRuntime | None = None,
    evaluation_backbone: EvaluationBackbone | None = None,
    prior_session_reports: tuple[EvaluationReport, ...] = (),
    upstream_snapshots: dict[str, Snapshot[Any]] | None = None,
    joint_loop_result: object | None = None,
    credit_proposals: tuple[ModificationProposal, ...] = (),
    reflection_mode: WritebackMode = WritebackMode.PROPOSAL_ONLY,
    temporal_policy: TemporalPolicy | None = None,
    world_temporal_policy: TemporalPolicy | None = None,
    self_temporal_policy: TemporalPolicy | None = None,
    prediction_module: PredictionErrorModule | None = None,
    regime_module: RegimeModule | None = None,
    session_id: str = "runtime-session",
    wave_id: str = "wave-0",
    turn_index: int = 0,
    apply_slow_writeback: bool = True,
    substrate_self_mod_pe_magnitude: float = 0.0,
    substrate_self_mod_pe_reward: float = 0.0,
    substrate_self_mod_pe_threshold: float = 0.18,
    social_prediction_errors: tuple[SocialPredictionError, ...] = (),
    environment_event: EnvironmentEvent | None = None,
    common_ground_dyad_atoms: tuple[CommonGroundAtom, ...] = (),
    common_ground_group_atoms: tuple[CommonGroundAtom, ...] = (),
    group_identities: tuple[GroupIdentity, ...] = (),
    active_group_id: str | None = None,
    group_joint_attention: tuple[str, ...] = (),
    group_joint_commitments: tuple[str, ...] = (),
    group_regime_id: str | None = None,
) -> FinalIntegrationResult:
    prediction_action_context = _prediction_action_context_from_upstream(
        upstream_snapshots=upstream_snapshots,
        environment_event=environment_event,
    )
    modules = build_final_runtime_modules(
        config=config,
        substrate_adapter=substrate_adapter,
        user_input=user_input,
        application_rare_heavy_state=application_rare_heavy_state,
        domain_knowledge_store=domain_knowledge_store,
        case_memory_store=case_memory_store,
        domain_experience_packages=domain_experience_packages,
        memory_store=memory_store,
        semantic_state_store=semantic_state_store,
        semantic_proposal_runtime=semantic_proposal_runtime,
        tom_proposal_runtime=tom_proposal_runtime,
        common_ground_proposal_runtime=common_ground_proposal_runtime,
        evaluation_backbone=evaluation_backbone,
        credit_proposals=credit_proposals,
        reflection_mode=reflection_mode,
        temporal_policy=temporal_policy,
        world_temporal_policy=world_temporal_policy,
        self_temporal_policy=self_temporal_policy,
        prediction_module=prediction_module,
        regime_module=regime_module,
        session_id=session_id,
        wave_id=wave_id,
        turn_index=turn_index,
        substrate_self_mod_pe_magnitude=substrate_self_mod_pe_magnitude,
        substrate_self_mod_pe_reward=substrate_self_mod_pe_reward,
        substrate_self_mod_pe_threshold=substrate_self_mod_pe_threshold,
        social_prediction_errors=social_prediction_errors,
        environment_event=environment_event,
        prediction_action_context=prediction_action_context,
        common_ground_dyad_atoms=common_ground_dyad_atoms,
        common_ground_group_atoms=common_ground_group_atoms,
        group_identities=group_identities,
        active_group_id=active_group_id,
        group_joint_attention=group_joint_attention,
        group_joint_commitments=group_joint_commitments,
        group_regime_id=group_regime_id,
    )
    if upstream_snapshots:
        for module in modules:
            previous_snapshot = upstream_snapshots.get(module.slot_name)
            if previous_snapshot is not None:
                module.seed_version(previous_snapshot.version)
    recorder = EventRecorder()
    registry = SlotRegistry()
    if upstream_snapshots:
        registry.seed_versions(upstream_snapshots)
    shadow_snapshots: dict[str, Snapshot[Any]] = {}
    active_snapshots = await propagate(
        modules,
        upstream=upstream_snapshots,
        registry=registry,
        recorder=recorder,
        shadow_snapshots=shadow_snapshots,
        session_id=session_id,
        wave_id=wave_id,
    )
    if upstream_snapshots:
        for slot_name in ("session_post_slow_loop", "experience_consolidation"):
            upstream_snapshot = upstream_snapshots.get(slot_name)
            if upstream_snapshot is None:
                continue
            resolved_level = config.level_for(slot_name, WiringLevel.DISABLED)
            if resolved_level is WiringLevel.ACTIVE:
                active_snapshots.setdefault(slot_name, upstream_snapshot)
                shadow_snapshots.pop(slot_name, None)
            elif resolved_level is WiringLevel.SHADOW:
                shadow_snapshots.setdefault(slot_name, upstream_snapshot)
    writeback_result: WritebackResult | None = None
    writeback_source: str | None = None
    reflection_module = next((module for module in modules if isinstance(module, ReflectionModule)), None)
    evaluation_module = next((module for module in modules if isinstance(module, EvaluationModule)), None)
    credit_module = next((module for module in modules if isinstance(module, CreditModule)), None)
    regime_module = next((module for module in modules if isinstance(module, RegimeModule)), None)
    temporal_modules = tuple(
        module
        for module in modules
        if isinstance(module, TrackTemporalModule)
    )
    reflection_snapshot = active_snapshots.get("reflection")
    if reflection_snapshot is None:
        reflection_snapshot = shadow_snapshots.get("reflection")
        if reflection_snapshot is not None:
            writeback_source = "shadow"
    else:
        writeback_source = "active"
    credit_snapshot = active_snapshots.get("credit") or shadow_snapshots.get("credit")
    evolution_judgement: EvolutionJudgement | None = None
    cross_session_verdict = ""
    judge_allows_structural_writeback = reflection_mode is WritebackMode.APPLY
    session_report: EvaluationReport | None = None
    session_post_writeback_request: SessionPostWritebackRequest | None = None
    evaluation_snapshot = active_snapshots.get("evaluation")
    retrieval_policy_snapshot = active_snapshots.get("retrieval_policy")
    domain_knowledge_snapshot = active_snapshots.get("domain_knowledge")
    case_memory_snapshot = active_snapshots.get("case_memory")
    strategy_playbook_snapshot = active_snapshots.get("strategy_playbook")
    experience_fast_prior_snapshot = active_snapshots.get("experience_fast_prior")
    boundary_policy_snapshot = active_snapshots.get("boundary_policy")
    response_assembly_snapshot = active_snapshots.get("response_assembly")
    semantic_state_values = tuple(
        snapshot.value
        for slot in SEMANTIC_OWNER_SLOTS
        for snapshot in (active_snapshots.get(slot),)
        if snapshot is not None
        and isinstance(
            snapshot.value,
            (
                PlanIntentSnapshot,
                CommitmentSnapshot,
                OpenLoopSnapshot,
                UserModelSnapshot,
                ExecutionResultSnapshot,
                BeliefAssumptionSnapshot,
                RelationshipStateSnapshot,
                GoalValueSnapshot,
                BoundaryConsentSnapshot,
            ),
        )
    )
    prediction_snapshot_value = (
        active_snapshots.get("prediction_error").value
        if active_snapshots.get("prediction_error") is not None
        and isinstance(active_snapshots.get("prediction_error").value, PredictionErrorSnapshot)
        else None
    )
    track_runtime_states_for_evaluation = tuple(
        (module.track.value, runtime_state)
        for module in temporal_modules
        for runtime_state in (module.export_runtime_state(),)
        if runtime_state is not None
    )
    temporal_runtime_state_for_evaluation = None
    if len(track_runtime_states_for_evaluation) == 2:
        track_state_map = dict(track_runtime_states_for_evaluation)
        temporal_runtime_state_for_evaluation = build_temporal_runtime_state_aggregate(
            world_state=track_state_map["world"],
            self_state=track_state_map["self"],
        )
    reflection_promote = False
    reflection_promote_reason = ""
    if (
        evaluation_module is not None
        and evaluation_snapshot is not None
        and isinstance(evaluation_snapshot.value, EvaluationSnapshot)
    ):
        from volvence_zero.joint_loop.runtime import ScheduledJointLoopResult

        enriched_evaluation = evaluation_snapshot.value
        temporal_snapshot = active_snapshots.get("temporal_abstraction")
        temporal_value = (
            temporal_snapshot.value
            if temporal_snapshot is not None and isinstance(temporal_snapshot.value, TemporalAbstractionSnapshot)
            else None
        )
        enriched_evaluation = evaluation_module.backbone.record_temporal_public_evidence(
            session_id=session_id,
            wave_id=wave_id,
            timestamp_ms=evaluation_snapshot.timestamp_ms + 1,
            base_snapshot=enriched_evaluation,
            temporal_snapshot=temporal_value,
            metacontroller_state=temporal_runtime_state_for_evaluation,
        )
        joint_kernel_scores = (
            joint_loop_result.kernel_scores
            if isinstance(joint_loop_result, ScheduledJointLoopResult)
            else ()
        )
        if joint_kernel_scores:
            enriched_evaluation = evaluation_module.backbone.record_external_scores(
                session_id=session_id,
                wave_id=wave_id,
                timestamp_ms=evaluation_snapshot.timestamp_ms + 2,
                base_snapshot=enriched_evaluation,
                scores=joint_kernel_scores,
                description_suffix=f"Enriched with {len(joint_kernel_scores)} ETA kernel scores.",
            )
        enriched_evaluation = evaluation_module.backbone.record_prediction_error_evidence(
            session_id=session_id,
            wave_id=wave_id,
            timestamp_ms=evaluation_snapshot.timestamp_ms + 3,
            base_snapshot=enriched_evaluation,
            prediction_error_snapshot=prediction_snapshot_value,
        )
        enriched_evaluation = evaluation_module.backbone.record_learning_evidence(
            session_id=session_id,
            wave_id=wave_id,
            timestamp_ms=evaluation_snapshot.timestamp_ms + 4,
            base_snapshot=enriched_evaluation,
            memory_snapshot=active_snapshots.get("memory").value if active_snapshots.get("memory") is not None else None,
            reflection_snapshot=reflection_snapshot.value if reflection_snapshot is not None else None,
            writeback_result=writeback_result,
            joint_loop_result=joint_loop_result,
            regime_snapshot=active_snapshots.get("regime").value if active_snapshots.get("regime") is not None else None,
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
                if strategy_playbook_snapshot is not None
                and isinstance(strategy_playbook_snapshot.value, StrategyPlaybookSnapshot)
                else None
            ),
            experience_fast_prior_snapshot=(
                experience_fast_prior_snapshot.value
                if experience_fast_prior_snapshot is not None
                and isinstance(experience_fast_prior_snapshot.value, ExperienceFastPriorSnapshot)
                else None
            ),
            boundary_policy_snapshot=(
                boundary_policy_snapshot.value
                if boundary_policy_snapshot is not None and isinstance(boundary_policy_snapshot.value, BoundaryPolicySnapshot)
                else None
            ),
            response_assembly_snapshot=(
                response_assembly_snapshot.value
                if response_assembly_snapshot is not None and isinstance(response_assembly_snapshot.value, ResponseAssemblySnapshot)
                else None
            ),
            semantic_state_snapshots=semantic_state_values,
        )
        substrate_self_mod_snapshot = active_snapshots.get("substrate_self_mod")
        substrate_self_mod_value = (
            substrate_self_mod_snapshot.value
            if substrate_self_mod_snapshot is not None
            and isinstance(substrate_self_mod_snapshot.value, SubstrateSelfModSnapshot)
            else None
        )
        if substrate_self_mod_value is not None:
            enriched_evaluation = evaluation_module.backbone.record_external_scores(
                session_id=session_id,
                wave_id=wave_id,
                timestamp_ms=evaluation_snapshot.timestamp_ms + 5,
                base_snapshot=enriched_evaluation,
                scores=(
                    EvaluationScore(
                        family="learning",
                        metric_name="substrate_online_fast_change_rate",
                        value=substrate_self_mod_value.parameter_change_rate,
                        confidence=0.7,
                        evidence=substrate_self_mod_value.description,
                    ),
                    EvaluationScore(
                        family="learning",
                        metric_name="substrate_online_fast_gate_preview",
                        value=1.0 if substrate_self_mod_value.gate_preview == GateDecision.ALLOW.value else 0.0,
                        confidence=0.7,
                        evidence=substrate_self_mod_value.description,
                    ),
                    EvaluationScore(
                        family="learning",
                        metric_name="substrate_online_fast_optimizer_norm",
                        value=substrate_self_mod_value.optimizer_state_norm,
                        confidence=0.7,
                        evidence=substrate_self_mod_value.description,
                    ),
                    EvaluationScore(
                        family="learning",
                        metric_name="substrate_online_fast_recommended",
                        value=1.0 if substrate_self_mod_value.recommended else 0.0,
                        confidence=0.7,
                        evidence=substrate_self_mod_value.description,
                    ),
                    EvaluationScore(
                        family="learning",
                        metric_name="substrate_online_fast_runtime_evidence_strength",
                        value=substrate_self_mod_value.runtime_evidence_strength,
                        confidence=0.72,
                        evidence=substrate_self_mod_value.description,
                    ),
                    EvaluationScore(
                        family="learning",
                        metric_name="substrate_online_fast_proposal_readiness",
                        value=substrate_self_mod_value.proposal_readiness,
                        confidence=0.72,
                        evidence=substrate_self_mod_value.description,
                    ),
                ),
                description_suffix="Enriched with online-fast substrate self-mod evidence.",
            )
        reflection_accuracy = 0.0
        if reflection_module is not None:
            reflection_accuracy = reflection_module.engine.proposal_success_rate
        session_report = evaluation_module.backbone.build_session_report(
            session_id=session_id,
            timestamp_ms=evaluation_snapshot.timestamp_ms + 6,
        )
        cross_session_report = None
        if prior_session_reports:
            cross_session_report = evaluation_module.backbone.run_cross_session_benchmark(
                suite=CrossSessionBenchmarkSuite(
                    session_reports=prior_session_reports + (session_report,),
                )
            )
            cross_session_verdict = cross_session_report.verdict
        cycle_report = (
            joint_loop_result.cycle_report
            if isinstance(joint_loop_result, ScheduledJointLoopResult)
            else None
        )
        evolution_judgement = cycle_report.evolution_judgement if cycle_report is not None else None
        if evolution_judgement is None:
            replay_result = evaluation_module.backbone.run_default_evolution_benchmark(
                timestamp_ms=evaluation_snapshot.timestamp_ms + 6,
            )
            evolution_judgement = evaluation_module.backbone.judge_evolution_candidate(
                replay_suite_result=replay_result,
                session_report=session_report,
                cross_session_report=cross_session_report,
            )
        judge_allows_structural_writeback = _judge_allows_structural_writeback(evolution_judgement) and (
            reflection_mode is WritebackMode.APPLY
        )
        enriched_evaluation = replace(
            enriched_evaluation,
            reflection_accuracy=reflection_accuracy,
            longitudinal_verdict=cross_session_verdict,
        )
        active_snapshots["evaluation"] = evaluation_module.publish(enriched_evaluation)
        reflection_promote, reflection_promote_reason = reflection_promotion_eligible(
            evaluation_snapshot=enriched_evaluation,
            reflection_engine=reflection_module.engine if reflection_module is not None else None,
        )
        if credit_module is not None:
            extra_credits = derive_learning_evidence_credit_records(
                evaluation_snapshot=enriched_evaluation,
                timestamp_ms=active_snapshots["evaluation"].timestamp_ms + 1,
            )
            delayed_credits = derive_delayed_attribution_credit_records(
                regime_snapshot=active_snapshots.get("regime").value if active_snapshots.get("regime") is not None else None,
                timestamp_ms=active_snapshots["evaluation"].timestamp_ms + 2,
            )
            extra_credits = extra_credits + delayed_credits
            if prediction_snapshot_value is not None:
                extra_credits = extra_credits + derive_prediction_error_credit_records(
                    prediction_error=prediction_snapshot_value.error,
                    timestamp_ms=active_snapshots["evaluation"].timestamp_ms + 3,
                )
            social_prediction_error_snapshot = active_snapshots.get(
                "social_prediction_error"
            ) or shadow_snapshots.get("social_prediction_error")
            if (
                social_prediction_error_snapshot is not None
                and isinstance(
                    social_prediction_error_snapshot.value,
                    SocialPredictionErrorSnapshot,
                )
                and social_prediction_error_snapshot.value.errors
            ):
                extra_credits = (
                    extra_credits
                    + derive_social_prediction_error_credit_records(
                        social_errors=social_prediction_error_snapshot.value.errors,
                        timestamp_ms=active_snapshots["evaluation"].timestamp_ms + 4,
                    )
                )
            if extra_credits:
                credit_module.ledger.record_credits(extra_credits)
            active_snapshots["credit"] = credit_module.publish(credit_module.ledger.snapshot())
            credit_snapshot = active_snapshots.get("credit")
    session_post_writeback_request = _build_session_post_writeback_request(
        context_session_id=session_id,
        source_wave_id=wave_id,
        session_report=session_report,
        reflection_snapshot=reflection_snapshot,
        credit_snapshot=credit_snapshot,
        evolution_judgement=evolution_judgement,
        cross_session_verdict=cross_session_verdict,
        writeback_source=writeback_source,
        reflection_mode=reflection_mode,
        structural_writeback_allowed=judge_allows_structural_writeback,
        semantic_state_values=semantic_state_values,
    )
    if (
        apply_slow_writeback
        and session_post_writeback_request is not None
        and memory_store is not None
    ):
        writeback_result, temporal_audits = apply_session_post_writeback_request(
            request=session_post_writeback_request,
            memory_store=memory_store,
            temporal_policy=world_temporal_policy or temporal_policy,
            regime_module=regime_module,
        )
        if credit_module is not None and temporal_audits:
            for record in temporal_audits:
                credit_module.ledger.record_modification(record)
            active_snapshots["credit"] = credit_module.publish(credit_module.ledger.snapshot())
            credit_snapshot = active_snapshots.get("credit")
    elif (
        memory_store is not None
        and reflection_module is not None
        and reflection_snapshot is not None
        and isinstance(reflection_snapshot.value, ReflectionSnapshot)
    ):
        writeback_result = None
    if credit_module is not None:
        active_snapshots["credit"] = credit_module.publish(credit_module.ledger.snapshot())
    active_snapshots = _prune_disabled_stub_snapshots(active_snapshots)
    acceptance_report = build_acceptance_report(
        config=config,
        active_snapshots=active_snapshots,
        shadow_snapshots=shadow_snapshots,
        recorder=recorder,
    )
    track_runtime_states = tuple(
        (module.track.value, runtime_state)
        for module in temporal_modules
        for runtime_state in (module.export_runtime_state(),)
        if runtime_state is not None
    )
    temporal_runtime_state = None
    if len(track_runtime_states) == 2:
        track_state_map = dict(track_runtime_states)
        temporal_runtime_state = build_temporal_runtime_state_aggregate(
            world_state=track_state_map["world"],
            self_state=track_state_map["self"],
        )
    if (
        evaluation_module is not None
        and temporal_runtime_state is not None
        and "evaluation" in active_snapshots
        and isinstance(active_snapshots["evaluation"].value, EvaluationSnapshot)
    ):
        active_snapshots["evaluation"] = evaluation_module.publish(
            evaluation_module.backbone.record_temporal_public_evidence(
                session_id=session_id,
                wave_id=wave_id,
                timestamp_ms=active_snapshots["evaluation"].timestamp_ms + 10,
                base_snapshot=active_snapshots["evaluation"].value,
                temporal_snapshot=(
                    active_snapshots["temporal_abstraction"].value
                    if "temporal_abstraction" in active_snapshots
                    and isinstance(active_snapshots["temporal_abstraction"].value, TemporalAbstractionSnapshot)
                    else None
                ),
                metacontroller_state=temporal_runtime_state,
            )
        )
    return FinalIntegrationResult(
        active_snapshots=active_snapshots,
        shadow_snapshots=shadow_snapshots,
        acceptance_report=acceptance_report,
        event_count=len(recorder.events),
        writeback_result=writeback_result,
        writeback_source=writeback_source,
        temporal_runtime_state=temporal_runtime_state,
        track_runtime_states=track_runtime_states,
        prediction_error_snapshot=prediction_snapshot_value,
        reflection_promotion_eligible=reflection_promote,
        reflection_promotion_reason=reflection_promote_reason,
        evolution_judgement=evolution_judgement,
        cross_session_verdict=cross_session_verdict,
        session_post_writeback_request=None if apply_slow_writeback else session_post_writeback_request,
    )


def build_acceptance_report(
    *,
    config: FinalRolloutConfig,
    active_snapshots: dict[str, Snapshot[Any]],
    shadow_snapshots: dict[str, Snapshot[Any]],
    recorder: EventRecorder,
) -> FinalAcceptanceReport:
    def resolved_level(module_name: str) -> WiringLevel:
        return config.level_for(module_name, WiringLevel.DISABLED)

    active_slots = tuple(sorted(active_snapshots))
    shadow_slots = tuple(sorted(shadow_snapshots))
    disabled_slots = tuple(
        sorted(
            name
            for name in (
                "substrate",
                "substrate_self_mod",
                "memory",
                "retrieval_policy",
                "domain_knowledge",
                "case_memory",
                "strategy_playbook",
                "experience_fast_prior",
                "boundary_policy",
                "response_assembly",
                "dual_track",
                "evaluation",
                "prediction_error",
                "regime",
                "credit",
                "reflection",
                "temporal",
                *SEMANTIC_OWNER_SLOTS,
            )
            if resolved_level(name) is WiringLevel.DISABLED
        )
    )
    issues: list[str] = []
    recommendations: list[str] = []

    expected_active: set[str] = set()
    for module_name, slot_name in (
        ("substrate", "substrate"),
        ("substrate_self_mod", "substrate_self_mod"),
        ("memory", "memory"),
        ("retrieval_policy", "retrieval_policy"),
        ("domain_knowledge", "domain_knowledge"),
        ("case_memory", "case_memory"),
        ("strategy_playbook", "strategy_playbook"),
        ("experience_fast_prior", "experience_fast_prior"),
        ("boundary_policy", "boundary_policy"),
        ("response_assembly", "response_assembly"),
        ("dual_track", "dual_track"),
        ("evaluation", "evaluation"),
        ("prediction_error", "prediction_error"),
        ("regime", "regime"),
        ("credit", "credit"),
        ("reflection", "reflection"),
        *tuple((slot, slot) for slot in SEMANTIC_OWNER_SLOTS),
    ):
        if config.is_active(module_name):
            expected_active.add(slot_name)
    if config.is_active("temporal"):
        expected_active.add("temporal_abstraction")
        expected_active.add("world_temporal")
        expected_active.add("self_temporal")
    missing_active = sorted(expected_active - set(active_slots))
    if missing_active:
        issues.append(f"Missing active slots: {', '.join(missing_active)}")

    violation_count = sum(1 for event in recorder.events if event.event_type == "contract.violation")
    if violation_count:
        issues.append(f"Observed {violation_count} contract violation events during final wiring.")

    if resolved_level("reflection") is WiringLevel.SHADOW and "reflection" not in shadow_slots and "reflection" not in active_slots:
        issues.append("Reflection wiring configured but no reflection snapshot was produced.")
    if resolved_level("reflection") is WiringLevel.ACTIVE and "reflection" not in active_slots:
        issues.append("Reflection is configured ACTIVE but did not publish into the active chain.")

    if (
        resolved_level("substrate_self_mod") is WiringLevel.SHADOW
        and "substrate_self_mod" not in shadow_slots
        and "substrate_self_mod" not in active_slots
    ):
        issues.append("Substrate self-mod wiring configured but no substrate_self_mod snapshot was produced.")
    if resolved_level("substrate_self_mod") is WiringLevel.ACTIVE and "substrate_self_mod" not in active_slots:
        issues.append("Substrate self-mod is configured ACTIVE but did not publish into the active chain.")

    if (
        resolved_level("retrieval_policy") is WiringLevel.SHADOW
        and "retrieval_policy" not in shadow_slots
        and "retrieval_policy" not in active_slots
    ):
        issues.append("Retrieval policy wiring configured but no retrieval_policy snapshot was produced.")
    if resolved_level("retrieval_policy") is WiringLevel.ACTIVE and "retrieval_policy" not in active_slots:
        issues.append("Retrieval policy is configured ACTIVE but did not publish into the active chain.")

    if (
        resolved_level("domain_knowledge") is WiringLevel.SHADOW
        and "domain_knowledge" not in shadow_slots
        and "domain_knowledge" not in active_slots
    ):
        issues.append("Domain knowledge wiring configured but no domain_knowledge snapshot was produced.")
    if resolved_level("domain_knowledge") is WiringLevel.ACTIVE and "domain_knowledge" not in active_slots:
        issues.append("Domain knowledge is configured ACTIVE but did not publish into the active chain.")

    if (
        resolved_level("case_memory") is WiringLevel.SHADOW
        and "case_memory" not in shadow_slots
        and "case_memory" not in active_slots
    ):
        issues.append("Case memory wiring configured but no case_memory snapshot was produced.")
    if resolved_level("case_memory") is WiringLevel.ACTIVE and "case_memory" not in active_slots:
        issues.append("Case memory is configured ACTIVE but did not publish into the active chain.")

    if (
        resolved_level("strategy_playbook") is WiringLevel.SHADOW
        and "strategy_playbook" not in shadow_slots
        and "strategy_playbook" not in active_slots
    ):
        issues.append("Strategy playbook wiring configured but no strategy_playbook snapshot was produced.")
    if resolved_level("strategy_playbook") is WiringLevel.ACTIVE and "strategy_playbook" not in active_slots:
        issues.append("Strategy playbook is configured ACTIVE but did not publish into the active chain.")

    if (
        resolved_level("experience_fast_prior") is WiringLevel.SHADOW
        and "experience_fast_prior" not in shadow_slots
        and "experience_fast_prior" not in active_slots
    ):
        issues.append("Experience fast prior wiring configured but no experience_fast_prior snapshot was produced.")
    if resolved_level("experience_fast_prior") is WiringLevel.ACTIVE and "experience_fast_prior" not in active_slots:
        issues.append("Experience fast prior is configured ACTIVE but did not publish into the active chain.")

    if (
        resolved_level("boundary_policy") is WiringLevel.SHADOW
        and "boundary_policy" not in shadow_slots
        and "boundary_policy" not in active_slots
    ):
        issues.append("Boundary policy wiring configured but no boundary_policy snapshot was produced.")
    if resolved_level("boundary_policy") is WiringLevel.ACTIVE and "boundary_policy" not in active_slots:
        issues.append("Boundary policy is configured ACTIVE but did not publish into the active chain.")

    if (
        resolved_level("response_assembly") is WiringLevel.SHADOW
        and "response_assembly" not in shadow_slots
        and "response_assembly" not in active_slots
    ):
        issues.append("Response assembly wiring configured but no response_assembly snapshot was produced.")
    if resolved_level("response_assembly") is WiringLevel.ACTIVE and "response_assembly" not in active_slots:
        issues.append("Response assembly is configured ACTIVE but did not publish into the active chain.")

    if resolved_level("temporal") is WiringLevel.SHADOW and "temporal_abstraction" not in shadow_slots and "temporal_abstraction" not in active_slots:
        issues.append("Temporal wiring configured but no temporal snapshot was produced.")
    if resolved_level("temporal") is WiringLevel.SHADOW and "world_temporal" not in shadow_slots and "world_temporal" not in active_slots:
        issues.append("World temporal wiring configured but no world_temporal snapshot was produced.")
    if resolved_level("temporal") is WiringLevel.SHADOW and "self_temporal" not in shadow_slots and "self_temporal" not in active_slots:
        issues.append("Self temporal wiring configured but no self_temporal snapshot was produced.")
    if resolved_level("temporal") is WiringLevel.ACTIVE and "temporal_abstraction" not in active_slots:
        issues.append("Temporal is configured ACTIVE but did not publish into the active chain.")
    if resolved_level("temporal") is WiringLevel.ACTIVE and "world_temporal" not in active_slots:
        issues.append("Temporal is configured ACTIVE but world_temporal did not publish into the active chain.")
    if resolved_level("temporal") is WiringLevel.ACTIVE and "self_temporal" not in active_slots:
        issues.append("Temporal is configured ACTIVE but self_temporal did not publish into the active chain.")

    if config.is_active("reflection"):
        recommendations.append("Keep reflection in proposal-only mode until writeback acceptance is proven.")
    if config.is_active("temporal"):
        recommendations.append("Validate temporal active mode against rollout evidence before widening scope.")
    if config.is_active("substrate_self_mod"):
        recommendations.append("Keep online-fast substrate self-mod bounded and rollback-safe before widening scope.")
    if config.is_active("domain_knowledge"):
        recommendations.append("Keep domain knowledge evidence compact and citation-bounded before widening scope.")
    if config.is_active("case_memory"):
        recommendations.append("Keep case_memory as a sibling owner and avoid collapsing it back into memory.")
    if config.is_active("strategy_playbook"):
        recommendations.append("Keep strategy playbook as advisory prior evidence, not a second temporal owner.")
    if config.is_active("experience_fast_prior"):
        recommendations.append("Keep experience fast prior compact and advisory so downstream owners remain primary.")
    if config.is_active("experience_consolidation"):
        recommendations.append(
            "Treat experience_consolidation as a session-owned post surface and pass it forward via snapshots, not a second final-wiring owner."
        )
    if config.is_active("boundary_policy"):
        recommendations.append("Validate boundary policy triggers against rollout evidence before widening scope.")
    if config.is_active("response_assembly"):
        recommendations.append("Keep response assembly as the single public expression-control surface, not a second prompt owner.")
    if not recommendations:
        recommendations.append("Core chain is wired; next step is controlled widening via rollout evidence.")

    passed = not issues
    description = (
        f"Final wiring acceptance {'passed' if passed else 'failed'} with "
        f"{len(active_slots)} active slots, {len(shadow_slots)} shadow slots, "
        f"{len(disabled_slots)} disabled slots."
    )
    return FinalAcceptanceReport(
        passed=passed,
        active_slots=active_slots,
        shadow_slots=shadow_slots,
        disabled_slots=disabled_slots,
        issues=tuple(issues),
        recommendations=tuple(recommendations),
        description=description,
    )
