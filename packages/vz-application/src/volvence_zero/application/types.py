"""Application-tier dataclasses and enums (debt #9 wave 2).

All ``application``-tier typed surface that other layers consume:
snapshots (``BoundaryPolicySnapshot`` / ``CaseMemorySnapshot`` /
``DomainKnowledgeSnapshot`` / ``ExperienceFastPriorSnapshot`` /
``ResponseAssemblySnapshot`` / ``StrategyPlaybookSnapshot`` /
``RetrievalPolicySnapshot`` / ``ExperienceConsolidationSnapshot``),
their record / atom / hit value-types, prior-update payloads,
rare-heavy checkpoint payloads, plus the six application enums
(``EvidenceStrength`` / ``RiskBand`` / ``ProfessionalScope`` /
``KnowledgeSourceType`` / ``ExperienceOutcomeLabel`` /
``ResponseMode`` / ``KnowledgeDepth`` / ``KnowledgeSourceKind`` /
``KnowledgeReviewStatus``).

Wave 2 of debt #9 split: these were lines 55-676 of the original
monolithic ``runtime.py``. The legacy import path
``from volvence_zero.application.runtime import ...`` keeps
working via the re-export shell.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import TYPE_CHECKING, Any, Mapping

from volvence_zero.dual_track import DualTrackSnapshot
from volvence_zero.memory import MemoryEntry, MemorySnapshot, Track
from volvence_zero.runtime import RuntimeModule, RuntimePlaceholderValue, Snapshot, WiringLevel
from volvence_zero.social_cognition import (
    BeliefAboutOtherSnapshot,
    CommonGroundSnapshot,
    ConversationalRoleSnapshot,
    FeelingAboutOtherSnapshot,
    GroupSnapshot,
    IntentAboutOtherSnapshot,
    PreferenceAboutOtherSnapshot,
)
from volvence_zero.application.storage import (
    ApplicationCaseMemoryStore,
    ApplicationDomainKnowledgeStore,
    CaseMemoryRecord,
    DomainKnowledgeRecord,
)
from volvence_zero.application.retrieval_readout import (
    RetrievalControlReadoutInputs,
    RetrievalControlReadoutParameters,
    RetrievalReadoutCheckpoint,
    RetrievalControlReadoutStrategy,
)

if TYPE_CHECKING:
    from volvence_zero.prediction.error import PredictionErrorSnapshot
    from volvence_zero.regime import RegimeSnapshot
    from volvence_zero.temporal_types import TemporalAbstractionSnapshot


from volvence_zero.application.scoring_helpers import clamp01 as _clamp

class EvidenceStrength(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ProfessionalScope(str, Enum):
    GENERAL_SUPPORT = "general-support"
    DOMAIN_INFORMATION = "domain-information"
    DOMAIN_DECISION_SUPPORT = "domain-decision-support"
    PROFESSIONAL_HANDOFF = "professional-handoff"


class KnowledgeSourceType(str, Enum):
    LAW = "law"
    OFFICIAL_GUIDE = "official-guide"
    INTERNAL_GUIDE = "internal-guide"
    REVIEWED_ARTICLE = "reviewed-article"
    PLAYBOOK = "playbook"


class ExperienceOutcomeLabel(str, Enum):
    IMPROVED = "improved"
    STABLE = "stable"
    WORSENED = "worsened"
    UNKNOWN = "unknown"


class ResponseMode(str, Enum):
    SUPPORT = "support"
    CLARIFY = "clarify"
    STRUCTURE = "structure"
    REFER_OUT = "refer-out"


# ---------------------------------------------------------------------------
# Snapshot building blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnowledgeCitation:
    citation_id: str
    source_type: KnowledgeSourceType
    title: str
    locator: str
    snippet: str
    url: str | None


@dataclass(frozen=True)
class KnowledgeHit:
    hit_id: str
    domain: str
    topic_tags: tuple[str, ...]
    jurisdiction_tags: tuple[str, ...]
    freshness_label: str
    confidence: float
    evidence_strength: EvidenceStrength
    summary: str
    conflict_markers: tuple[str, ...]
    citations: tuple[KnowledgeCitation, ...]
    description: str


@dataclass(frozen=True)
class ContinuumLocation:
    profile_id: str
    band_id: str
    band_role: str
    position: float
    update_frequency: float
    reconstruction_source: str
    description: str


@dataclass(frozen=True)
class CaseInterventionStep:
    step_id: str
    step_order: int
    regime_id: str | None
    abstract_action: str | None
    action_label: str
    description: str


@dataclass(frozen=True)
class CaseOutcomeSummary:
    outcome_label: ExperienceOutcomeLabel
    delayed_signal_count: int
    escalation_observed: bool
    repair_observed: bool
    confidence: float
    description: str


@dataclass(frozen=True)
class CaseEpisodeHit:
    case_id: str
    domain: str
    problem_pattern: str
    user_state_pattern: str
    risk_markers: tuple[str, ...]
    track_tags: tuple[str, ...]
    regime_tags: tuple[str, ...]
    intervention_steps: tuple[CaseInterventionStep, ...]
    outcome: CaseOutcomeSummary
    relevance_score: float
    description: str
    continuum_location: ContinuumLocation | None = None


@dataclass(frozen=True)
class PlaybookRule:
    rule_id: str
    problem_pattern: str
    recommended_regime: str | None
    recommended_ordering: tuple[str, ...]
    recommended_pacing: str
    avoid_patterns: tuple[str, ...]
    knowledge_weight_hint: float
    experience_weight_hint: float
    applicability_scope: tuple[str, ...]
    confidence: float
    description: str
    continuum_band_id: str | None = None
    mean_continuum_position: float = 0.0


@dataclass(frozen=True)
class BoundaryDecision:
    decision_id: str
    risk_band: RiskBand
    professional_scope: ProfessionalScope
    answer_depth_limit: str
    citation_required: bool
    clarification_required: bool
    refer_out_required: bool
    blocked_topics: tuple[str, ...]
    required_disclaimers: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class ResponseSpeechPlan:
    cue: str
    inferred_need: str
    response_adjustment: str
    question_budget: int
    required_steps: tuple[str, ...] = ()
    description: str = ""


# ---------------------------------------------------------------------------
# ExperienceFastPrior helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperienceFastPriorRegimeBias:
    regime_id: str
    bias: float
    source_attribution_ids: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class ExperienceFastPriorActionBias:
    abstract_action: str
    bias: float
    source_attribution_ids: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class ExperienceFastPriorFamilyBias:
    action_family_version: int
    continuation_bias: float
    source_attribution_ids: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class ExperienceFastPriorSequenceBias:
    regime_sequence: tuple[str, ...]
    action_family_version: int
    payoff_bias: float
    source_sequence_ids: tuple[str, ...]
    description: str


# ---------------------------------------------------------------------------
# Top-level snapshot types (consumed by vz-cognition.evaluation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryPolicySnapshot:
    active_decision: BoundaryDecision
    trigger_reasons: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class CaseMemorySnapshot:
    retrieval_policy_id: str
    hits: tuple[CaseEpisodeHit, ...]
    active_problem_patterns: tuple[str, ...]
    active_risk_markers: tuple[str, ...]
    description: str
    continuum_profile_id: str | None = None
    active_band_ids: tuple[str, ...] = ()
    mean_continuum_position: float = 0.0
    support_prior: float = 0.0
    task_prior: float = 0.0


@dataclass(frozen=True)
class DomainKnowledgeSnapshot:
    retrieval_policy_id: str
    active_domains: tuple[str, ...]
    hits: tuple[KnowledgeHit, ...]
    citation_required: bool
    jurisdiction_required: bool
    unresolved_conflicts: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class ExperienceFastPriorSnapshot:
    regime_biases: tuple[ExperienceFastPriorRegimeBias, ...]
    knowledge_weight_bias: float
    experience_weight_bias: float
    action_biases: tuple[ExperienceFastPriorActionBias, ...]
    family_biases: tuple[ExperienceFastPriorFamilyBias, ...]
    sequence_biases: tuple[ExperienceFastPriorSequenceBias, ...]
    prior_strength: float
    source_attribution_ids: tuple[str, ...]
    source_sequence_ids: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class ResponseAssemblySnapshot:
    regime_id: str | None
    regime_name: str
    abstract_action: str | None
    response_mode: ResponseMode
    answer_depth_limit: str
    citation_mode: str
    clarification_required: bool
    refer_out_required: bool
    ordering_plan: tuple[str, ...]
    knowledge_briefs: tuple[str, ...]
    case_briefs: tuple[str, ...]
    playbook_ordering: tuple[str, ...]
    required_disclaimers: tuple[str, ...]
    required_disclaimer_phrases: tuple[str, ...]
    control_code: tuple[float, ...]
    control_scale: float
    max_questions: int
    prompt_residue_summary: str
    prompt_residue_ratio: float
    knowledge_hit_count: int
    case_hit_count: int
    playbook_rule_count: int
    risk_band: RiskBand
    description: str
    continuum_target_position: float = 0.0
    ordering_driver: str = "playbook-only"
    semantic_record_counts: tuple[tuple[str, int], ...] = ()
    semantic_control_signal: float = 0.0
    semantic_residue_summary: str = ""
    expression_intent: str = "direct-answer"
    judgment_focus: tuple[str, ...] = ()
    speech_plan: ResponseSpeechPlan | None = None
    support_before_decision_pressure: float = 0.0
    eta_action_family: str = ""


@dataclass(frozen=True)
class StrategyPlaybookSnapshot:
    matched_problem_patterns: tuple[str, ...]
    matched_rules: tuple[PlaybookRule, ...]
    description: str
    continuum_profile_id: str | None = None
    active_band_ids: tuple[str, ...] = ()
    support_prior: float = 0.0
    task_prior: float = 0.0


@dataclass(frozen=True)
class ProtocolAlignmentRef:
    """One guidance-vs-active-protocol comparison result.

    ``layer`` is which protocol artifact the comparison was against
    (``strategy`` / ``knowledge`` / ``boundary``); ``relation`` is the
    reliable-apprenticeship verdict at the protocol layer:
    ``covered`` (guidance reinforces an active protocol element =>
    agreement region), ``novel`` (no active element covers it =>
    disagreement region / informative), ``conflict`` (guidance opposes
    an active element, e.g. endorses an ``avoid_patterns`` entry or
    negates an active rule => protocol-layer contradiction).
    ``target_ref`` is the matched ``rule_id`` / ``hit_id`` /
    ``decision_id`` (empty for ``novel``).
    """

    guidance_constraint_id: str
    layer: str
    relation: str
    target_ref: str
    severity: float
    description: str


@dataclass(frozen=True)
class ApprenticeshipProtocolAlignmentSnapshot:
    """Protocol-layer reliable-apprenticeship readout (DRAFT Packet 1).

    Compares the operator-guidance constraints published by the
    vz-cognition ``apprenticeship_alignment`` owner against the
    currently-active, compiled protocol artifacts (``strategy_playbook``
    / ``domain_knowledge`` / ``boundary_policy``) — a FINITE structured
    option set, where the reliable-active-apprenticeship reliability /
    eluder notions are well-defined (unlike the open content layer).
    SHADOW-only readout in Packet 1: no PE overlay, no belief/protocol
    revision. See ``docs/specs/apprenticeship-alignment-protocol-layer-draft.md``.
    """

    version_space_status: str
    reliability: str
    in_agreement_region: bool
    guidance_surprise: float
    matched_protocol_count: int
    alignment_refs: tuple[ProtocolAlignmentRef, ...]
    contradiction_refs: tuple[ProtocolAlignmentRef, ...]
    description: str


@dataclass(frozen=True)
class ApplicationOutcomeAttribution:
    attribution_id: str
    source_context_session_id: str
    source_wave_id: str
    regime_id: str | None
    abstract_action: str | None
    action_family_version: int
    retrieval_policy_id: str | None
    knowledge_weight: float
    experience_weight: float
    retrieval_mix_alignment: float
    regime_alignment: float
    abstract_action_alignment: float
    outcome_score: float
    resolved_turn_index: int
    description: str
    continuum_profile_id: str | None = None
    dominant_band_id: str | None = None
    mean_continuum_position: float = 0.0
    continuum_alignment: float = 0.0


@dataclass(frozen=True)
class ApplicationSequencePayoff:
    sequence_id: str
    regime_sequence: tuple[str, ...]
    action_family_version: int
    sample_count: int
    rolling_payoff: float
    latest_outcome: float
    description: str
    continuum_profile_id: str | None = None
    dominant_band_id: str | None = None
    mean_continuum_position: float = 0.0


class KnowledgeDepth(str, Enum):
    LIGHT = "light"
    MEDIUM = "medium"
    DEEP = "deep"


@dataclass(frozen=True)
class RetrievalPolicySnapshot:
    knowledge_domains: tuple[str, ...]
    experience_domains: tuple[str, ...]
    knowledge_weight: float
    experience_weight: float
    world_weight: float
    self_weight: float
    retrieval_depth: KnowledgeDepth
    citation_required: bool
    jurisdiction_required: bool
    risk_band: RiskBand
    regime_id: str | None
    abstract_action: str | None
    intent_description: str
    description: str
    knowledge_domain_ranking: tuple[str, ...] = ()
    experience_domain_ranking: tuple[str, ...] = ()
    response_mode_hint: str = "support"
    clarification_bias: float = 0.0
    refer_out_bias: float = 0.0
    answer_depth_bias: float = 0.0
    continuum_target_position_hint: float = 0.5
    ordering_bias: tuple[str, ...] = ()
    ordering_driver_hint: str = "retrieval-fallback"


class KnowledgeSourceKind(str, Enum):
    CONVERSATION = "conversation"
    EXTERNAL_IMPORT = "external-import"
    RARE_HEAVY_IMPORT = "rare-heavy-import"


class KnowledgeReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SHADOW = "shadow"


@dataclass(frozen=True)
class ConversationKnowledgeCandidate:
    candidate_id: str
    source_context_session_id: str
    source_wave_id: str
    source_turn_index: int
    turn_reference: str
    domain: str
    knowledge_hit_id: str
    citation_ids: tuple[str, ...]
    summary: str
    confidence: float
    boundary_aligned: bool
    review_status: KnowledgeReviewStatus
    is_fallback_hit: bool
    description: str


@dataclass(frozen=True)
class ExternalKnowledgeCandidate:
    candidate_id: str
    source_label: str
    domain: str
    topic_tags: tuple[str, ...]
    jurisdiction_tags: tuple[str, ...]
    source_type: str
    title: str
    locator: str
    summary: str
    snippet: str
    freshness_label: str
    confidence: float
    evidence_strength: str
    conflict_markers: tuple[str, ...] = ()
    url: str | None = None
    description: str = ""


@dataclass(frozen=True)
class KnowledgeReviewDecision:
    candidate_id: str
    review_status: KnowledgeReviewStatus
    reviewer_id: str
    confidence: float
    note: str
    supersedes_record_id: str | None = None


@dataclass(frozen=True)
class ReviewedKnowledgeCandidate:
    candidate_id: str
    source_kind: KnowledgeSourceKind
    review_status: KnowledgeReviewStatus
    record: DomainKnowledgeRecord
    source_candidate_ids: tuple[str, ...]
    review_note: str
    confidence: float
    supersedes_record_id: str | None = None


@dataclass(frozen=True)
class KnowledgeImportSession:
    session_id: str
    source_kind: KnowledgeSourceKind
    source_label: str
    pending_candidates: tuple[ExternalKnowledgeCandidate, ...] = ()
    reviewed_candidates: tuple[ReviewedKnowledgeCandidate, ...] = ()
    applied_candidate_ids: tuple[str, ...] = ()
    blocked_candidate_ids: tuple[str, ...] = ()
    description: str = ""


KnowledgeImportBatch = KnowledgeImportSession


@dataclass(frozen=True)
class ExperienceDelta:
    delta_id: str
    delta_type: str
    target_slot: str
    summary: str
    confidence: float
    blocked: bool
    description: str
    continuum_band_id: str | None = None
    continuum_position: float = 0.0


@dataclass(frozen=True)
class BoundaryPriorHint:
    hint_id: str
    regime_id: str | None
    trigger_reasons: tuple[str, ...]
    answer_depth_limit_hint: str
    clarification_required: bool
    refer_out_required: bool
    blocked_topics: tuple[str, ...]
    required_disclaimers: tuple[str, ...]
    confidence: float
    description: str


@dataclass(frozen=True)
class CaseMemoryPriorUpdate:
    update_id: str
    target: str
    record: CaseMemoryRecord
    confidence: float
    description: str


@dataclass(frozen=True)
class StrategyPlaybookPriorUpdate:
    update_id: str
    target: str
    rule: PlaybookRule
    confidence: float
    description: str


@dataclass(frozen=True)
class BoundaryPolicyPriorUpdate:
    update_id: str
    target: str
    hint: BoundaryPriorHint
    confidence: float
    description: str


@dataclass(frozen=True)
class DomainKnowledgePriorUpdate:
    update_id: str
    target: str
    record: DomainKnowledgeRecord
    confidence: float
    description: str
    source_kind: KnowledgeSourceKind = KnowledgeSourceKind.CONVERSATION
    source_candidate_ids: tuple[str, ...] = ()
    review_status: KnowledgeReviewStatus = KnowledgeReviewStatus.APPROVED
    citation_ids: tuple[str, ...] = ()
    supersedes_record_id: str | None = None


@dataclass(frozen=True)
class RetrievalReadoutPriorUpdate:
    update_id: str
    target: str
    checkpoint: RetrievalReadoutCheckpoint
    confidence: float
    description: str


@dataclass(frozen=True)
class ApplicationPriorUpdate:
    source_session_post_job_id: str
    case_memory_updates: tuple[CaseMemoryPriorUpdate, ...] = ()
    strategy_playbook_updates: tuple[StrategyPlaybookPriorUpdate, ...] = ()
    boundary_policy_updates: tuple[BoundaryPolicyPriorUpdate, ...] = ()
    domain_knowledge_updates: tuple[DomainKnowledgePriorUpdate, ...] = ()
    retrieval_readout_updates: tuple[RetrievalReadoutPriorUpdate, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class ApplicationPriorWritebackReport:
    proposed_target_count: int
    applied_targets: tuple[str, ...]
    blocked_targets: tuple[str, ...]
    audit_record_count: int
    description: str


@dataclass(frozen=True)
class DelayedCreditSummary:
    summary_id: str
    regime_id: str | None
    abstract_action: str | None
    action_family_version: int
    retrieval_policy_id: str | None
    knowledge_weight: float
    experience_weight: float
    retrieval_mix_alignment: float
    regime_alignment: float
    abstract_action_alignment: float
    outcome_score: float
    sequence_payoff: float
    continuum_alignment: float
    attribution_count: int
    sequence_count: int
    description: str
    continuum_profile_id: str | None = None
    dominant_band_id: str | None = None
    mean_continuum_position: float = 0.0


@dataclass(frozen=True)
class ExperienceConsolidationSnapshot:
    source_session_post_job_id: str
    promoted_case_count: int
    playbook_delta_count: int
    boundary_delta_count: int
    deltas: tuple[ExperienceDelta, ...]
    description: str
    delayed_outcome_ledger: tuple[ApplicationOutcomeAttribution, ...] = ()
    sequence_payoffs: tuple[ApplicationSequencePayoff, ...] = ()
    latest_prior_update: ApplicationPriorUpdate | None = None
    latest_writeback_report: ApplicationPriorWritebackReport | None = None
    delayed_credit_summary: DelayedCreditSummary | None = None
    continuum_profile_id: str | None = None
    active_band_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ApplicationCaseCluster:
    cluster_id: str
    problem_pattern: str
    exemplar_count: int
    mean_relevance: float
    risk_markers: tuple[str, ...]
    description: str
    continuum_band_id: str | None = None
    continuum_position: float = 0.0


@dataclass(frozen=True)
class ApplicationRareHeavyCheckpoint:
    checkpoint_id: str
    domain_template_biases: tuple[tuple[str, float], ...]
    case_clusters: tuple[ApplicationCaseCluster, ...]
    distilled_playbook_rules: tuple[PlaybookRule, ...]
    description: str
    boundary_prior_hints: tuple[BoundaryPriorHint, ...] = ()
    reviewed_knowledge_candidates: tuple[ReviewedKnowledgeCandidate, ...] = ()
    continuum_profile_id: str | None = None
    retrieval_readout_checkpoint: RetrievalReadoutCheckpoint | None = None


from volvence_zero.application.scoring_helpers import (
    dedupe as _dedupe,
)

