from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import TYPE_CHECKING, Any, Mapping

from volvence_zero.dual_track import DualTrackSnapshot
from volvence_zero.memory import MemoryEntry, MemorySnapshot
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
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
    from volvence_zero.temporal import TemporalAbstractionSnapshot


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


class KnowledgeDepth(str, Enum):
    LIGHT = "light"
    MEDIUM = "medium"
    DEEP = "deep"


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
class DomainKnowledgeSnapshot:
    retrieval_policy_id: str
    active_domains: tuple[str, ...]
    hits: tuple[KnowledgeHit, ...]
    citation_required: bool
    jurisdiction_required: bool
    unresolved_conflicts: tuple[str, ...]
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
class CaseMemorySnapshot:
    retrieval_policy_id: str
    hits: tuple[CaseEpisodeHit, ...]
    active_problem_patterns: tuple[str, ...]
    active_risk_markers: tuple[str, ...]
    description: str
    continuum_profile_id: str | None = None
    active_band_ids: tuple[str, ...] = ()
    mean_continuum_position: float = 0.0


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
class StrategyPlaybookSnapshot:
    matched_problem_patterns: tuple[str, ...]
    matched_rules: tuple[PlaybookRule, ...]
    description: str
    continuum_profile_id: str | None = None
    active_band_ids: tuple[str, ...] = ()


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
class BoundaryPolicySnapshot:
    active_decision: BoundaryDecision
    trigger_reasons: tuple[str, ...]
    description: str


class ResponseMode(str, Enum):
    SUPPORT = "support"
    CLARIFY = "clarify"
    STRUCTURE = "structure"
    REFER_OUT = "refer-out"


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


def _dedupe(items: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def _memory_text(memory_snapshot: MemorySnapshot | None) -> str:
    if memory_snapshot is None:
        return ""
    parts = [
        memory_snapshot.transient_summary,
        memory_snapshot.episodic_summary,
        memory_snapshot.durable_summary,
        *(entry.content for entry in memory_snapshot.retrieved_entries[:8]),
    ]
    return " ".join(part for part in parts if part)


def _continuum_profile(memory_snapshot: MemorySnapshot | None) -> Any | None:
    if memory_snapshot is None or memory_snapshot.cms_state is None:
        return None
    return memory_snapshot.cms_state.continuum_profile


def _continuum_location_from_band(
    *,
    profile: Any | None,
    band_id: str | None,
    fallback_source: str,
) -> ContinuumLocation | None:
    if profile is None or band_id is None:
        return None
    band = next((candidate for candidate in profile.bands if candidate.band_id == band_id), None)
    if band is None:
        return None
    reconstruction_edge = next(
        (edge for edge in profile.reconstruction_edges if edge.target_band_id == band.band_id),
        None,
    )
    return ContinuumLocation(
        profile_id=profile.profile_id,
        band_id=band.band_id,
        band_role=band.role,
        position=band.persistence_bias,
        update_frequency=band.update_frequency,
        reconstruction_source=reconstruction_edge.transfer_kind if reconstruction_edge is not None else fallback_source,
        description=(
            f"Continuum location band={band.band_id} role={band.role} position={band.persistence_bias:.2f} "
            f"frequency={band.update_frequency:.2f}."
        ),
    )


def _fallback_continuum_location(
    *,
    band_id: str,
    position: float,
    update_frequency: float,
    source: str,
) -> ContinuumLocation:
    role = {
        "online-fast": "fast-band",
        "session-medium": "session-band",
        "background-slow": "slow-band",
        "tower-readout": "readout",
    }.get(band_id, "application-case-band")
    return ContinuumLocation(
        profile_id="application-continuum-fallback",
        band_id=band_id,
        band_role=role,
        position=position,
        update_frequency=update_frequency,
        reconstruction_source=source,
        description=(
            f"Fallback continuum location band={band_id} position={position:.2f} "
            f"frequency={update_frequency:.2f}."
        ),
    )


def _continuum_location_for_entry(
    *,
    entry: MemoryEntry,
    memory_snapshot: MemorySnapshot | None,
) -> ContinuumLocation | None:
    profile = _continuum_profile(memory_snapshot)
    band_by_stratum = {
        "transient": "online-fast",
        "episodic": "session-medium",
        "durable": "background-slow",
        "derived": "tower-readout",
    }
    location = _continuum_location_from_band(
        profile=profile,
        band_id=band_by_stratum.get(entry.stratum),
        fallback_source="artifact-anchor",
    )
    if location is not None:
        return location
    fallback_band_id = band_by_stratum.get(entry.stratum, "tower-readout")
    fallback_position = {
        "transient": 0.18,
        "episodic": 0.46,
        "durable": 0.82,
        "derived": 0.58,
    }.get(entry.stratum, 0.5)
    fallback_frequency = {
        "transient": 1.0,
        "episodic": 0.5,
        "durable": 0.25,
        "derived": 0.2,
    }.get(entry.stratum, 0.33)
    return _fallback_continuum_location(
        band_id=fallback_band_id,
        position=fallback_position,
        update_frequency=fallback_frequency,
        source="artifact-anchor",
    )


def _continuum_location_from_record(record: CaseMemoryRecord) -> ContinuumLocation | None:
    if record.continuum_profile_id is None or record.continuum_band_id is None:
        fallback_band_id = (
            "background-slow"
            if record.delayed_signal_count >= 4 or record.confidence >= 0.78
            else "session-medium"
        )
        fallback_position = 0.82 if fallback_band_id == "background-slow" else 0.48
        fallback_frequency = 0.25 if fallback_band_id == "background-slow" else 0.5
        return _fallback_continuum_location(
            band_id=fallback_band_id,
            position=fallback_position,
            update_frequency=fallback_frequency,
            source="persisted-case-fallback",
        )
    return ContinuumLocation(
        profile_id=record.continuum_profile_id,
        band_id=record.continuum_band_id,
        band_role="application-case-band",
        position=record.continuum_position,
        update_frequency=record.continuum_update_frequency,
        reconstruction_source=record.reconstruction_source,
        description=(
            f"Persisted case continuum band={record.continuum_band_id} position={record.continuum_position:.2f} "
            f"frequency={record.continuum_update_frequency:.2f}."
        ),
    )


def _semantic_tokens(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    ascii_buffer: list[str] = []
    compact = "".join(char for char in text.lower() if not char.isspace())
    for char in text.lower():
        if char.isascii() and char.isalnum():
            ascii_buffer.append(char)
            continue
        if ascii_buffer:
            tokens.append("".join(ascii_buffer))
            ascii_buffer.clear()
        if not char.isspace():
            tokens.append(char)
    if ascii_buffer:
        tokens.append("".join(ascii_buffer))
    tokens.extend(compact[index : index + 2] for index in range(len(compact) - 1))
    return tuple(tokens)


def _semantic_embedding(text: str, *, dim: int = 8) -> tuple[float, ...]:
    tokens = _semantic_tokens(text)
    if not tokens:
        return tuple(0.0 for _ in range(dim))
    vector = [0.0 for _ in range(dim)]
    for token in tokens:
        token_scale = max(len(token), 1)
        for index, char in enumerate(token):
            vector[(index + len(token)) % dim] += (ord(char) % 37) / 37.0 / token_scale
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-6:
        return tuple(0.0 for _ in range(dim))
    return tuple(value / norm for value in vector)


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right:
        return 0.0
    return sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))


def _semantic_similarity(text: str, prototype_text: str) -> float:
    embedding = _semantic_embedding(text)
    prototype = _semantic_embedding(prototype_text)
    return _clamp((_cosine_similarity(embedding, prototype) + 1.0) / 2.0)


def _signed_centered(score: float) -> float:
    return (score - 0.5) * 2.0


def _clamp_signed(value: float, *, magnitude: float = 0.25) -> float:
    return max(-magnitude, min(magnitude, value))


TASK_PRESSURE_PROTOTYPE = (
    "task pressure decision compare options concrete plan execution direct next step "
    "problem solving structure urgency organize"
)
SUPPORT_PRESENCE_PROTOTYPE = (
    "emotional support overwhelmed reassure steady warmth care calm gentle regulate "
    "feelings support-first stabilize"
)
REPAIR_PRESSURE_PROTOTYPE = (
    "repair trust conflict de-escalation apology rupture safety relational repair "
    "deescalate restore connection"
)
FAMILY_TRANSITION_PROTOTYPE = (
    "family transition separation co-parenting custody relationship breakdown household change "
    "emotionally intense transition"
)
PROFESSIONAL_PROCESS_PROTOTYPE = (
    "professional process legal procedural official guidance compliance sourced bounded advice "
    "jurisdiction sensitive process"
)
CAREER_DECISION_PROTOTYPE = (
    "career decision offer role transition trade-off professional path work change job choice"
)
JURISDICTION_CONTEXT_PROTOTYPE = (
    "local jurisdiction region local law court official local process local rules geography country city"
)
CHILD_IMPACT_PROTOTYPE = (
    "child family parenting custody co-parenting dependent vulnerable family impact"
)
DOMAIN_SENSITIVE_PROTOTYPE = (
    "legal professional official local law jurisdiction compliance procedural sensitive domain"
)
PROBLEM_PATTERN_PROTOTYPES: tuple[tuple[str, str], ...] = (
    ("family-transition-high-emotion", "emotionally intense family transition separation co-parenting overwhelm"),
    ("structured-decision-overwhelm", "too many options structure plan timeline organize compare next step"),
    ("relational-repair", "trust rupture conflict de-escalation apology relationship repair"),
)
USER_STATE_PROTOTYPES: tuple[tuple[str, str], ...] = (
    ("high-emotional-load", "overwhelmed anxious afraid distressed emotionally flooded tense"),
    ("needs-structure", "clarify organize structure sequence plan compare next step"),
    ("mixed-signal", "mixed ambivalent uncertain blended state both emotion and planning"),
)
PROBLEM_PATTERN_PROTOTYPES = PROBLEM_PATTERN_PROTOTYPES + (
    ("general-guidance", "general guidance mixed support and problem solving"),
)
RISK_BAND_ANCHORS: tuple[tuple[RiskBand, float], ...] = (
    (RiskBand.LOW, 0.18),
    (RiskBand.MEDIUM, 0.45),
    (RiskBand.HIGH, 0.72),
    (RiskBand.CRITICAL, 0.92),
)
KNOWLEDGE_DEPTH_ANCHORS: tuple[tuple[KnowledgeDepth, float], ...] = (
    (KnowledgeDepth.LIGHT, 0.22),
    (KnowledgeDepth.MEDIUM, 0.54),
    (KnowledgeDepth.DEEP, 0.84),
)
ANSWER_DEPTH_ANCHORS: tuple[tuple[str, float], ...] = (
    ("support-first", 0.28),
    ("standard", 0.56),
    ("high-level-only", 0.86),
)
PACING_ANCHORS: tuple[tuple[str, float], ...] = (
    ("structured", 0.20),
    ("slow", 0.52),
    ("gradual", 0.82),
)
OUTCOME_LABEL_ANCHORS: tuple[tuple[ExperienceOutcomeLabel, float], ...] = (
    (ExperienceOutcomeLabel.WORSENED, 0.18),
    (ExperienceOutcomeLabel.STABLE, 0.50),
    (ExperienceOutcomeLabel.IMPROVED, 0.82),
)


def _regime_bonus(regime_id: str | None, bonuses: Mapping[str, float]) -> float:
    if regime_id is None:
        return 0.0
    return bonuses.get(regime_id, 0.0)


def _nearest_anchor_value(score: float, anchors: tuple[tuple[Any, float], ...]) -> Any:
    return min(anchors, key=lambda item: abs(score - item[1]))[0]


def _truncate_text(text: str, *, max_chars: int = 96) -> str:
    compact = " ".join(part for part in text.split() if part)
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _ranked_labels(score_map: Mapping[str, float], *, max_count: int) -> tuple[str, ...]:
    ranked = sorted(score_map.items(), key=lambda item: (-item[1], item[0]))
    return tuple(label for label, _ in ranked[:max_count])


def _infer_track_weights(dual_track_snapshot: DualTrackSnapshot | None) -> tuple[float, float]:
    if dual_track_snapshot is None:
        return (0.5, 0.5)
    world_tension = max(0.05, dual_track_snapshot.world_track.tension_level)
    self_tension = max(0.05, dual_track_snapshot.self_track.tension_level)
    total = world_tension + self_tension
    return (_clamp(world_tension / total), _clamp(self_tension / total))


def _risk_band_from_state(
    *,
    dual_track_snapshot: DualTrackSnapshot | None,
    temporal_snapshot: "TemporalAbstractionSnapshot | None",
    regime_snapshot: "RegimeSnapshot | None",
) -> RiskBand:
    cross_track_tension = dual_track_snapshot.cross_track_tension if dual_track_snapshot is not None else 0.0
    switch_gate = (
        temporal_snapshot.controller_state.switch_gate
        if temporal_snapshot is not None
        else 0.0
    )
    regime_id = regime_snapshot.active_regime.regime_id if regime_snapshot is not None else None
    severity = _clamp(
        cross_track_tension * 0.56
        + switch_gate * 0.26
        + _regime_bonus(
            regime_id,
            {
                "repair_and_deescalation": 0.24,
                "emotional_support": 0.07,
                "problem_solving": 0.03,
            },
        )
    )
    return _nearest_anchor_value(severity, RISK_BAND_ANCHORS)


def _knowledge_domains(
    *,
    dual_track_snapshot: DualTrackSnapshot | None,
    regime_id: str | None,
    world_weight: float,
    self_weight: float,
    abstract_action: str | None,
) -> tuple[str, ...]:
    world_goals = " ".join(dual_track_snapshot.world_track.active_goals) if dual_track_snapshot is not None else ""
    self_goals = " ".join(dual_track_snapshot.self_track.active_goals) if dual_track_snapshot is not None else ""
    abstract_action_text = abstract_action or ""
    combined_text = " ".join(part for part in (world_goals, self_goals, abstract_action_text) if part)
    task_pull = _semantic_similarity(world_goals or combined_text, TASK_PRESSURE_PROTOTYPE)
    support_pull = _semantic_similarity(self_goals or combined_text, SUPPORT_PRESENCE_PROTOTYPE)
    repair_pull = _semantic_similarity(combined_text, REPAIR_PRESSURE_PROTOTYPE)
    family_transition_pull = _semantic_similarity(combined_text, FAMILY_TRANSITION_PROTOTYPE)
    professional_process_pull = _semantic_similarity(combined_text, PROFESSIONAL_PROCESS_PROTOTYPE)
    career_pull = _semantic_similarity(combined_text, CAREER_DECISION_PROTOTYPE)
    cross_track_tension = dual_track_snapshot.cross_track_tension if dual_track_snapshot is not None else abs(world_weight - self_weight)
    score_map = {
        "family_transition": _clamp(
            family_transition_pull * 0.60
            + support_pull * 0.08
            + repair_pull * 0.08
            + self_weight * 0.10
            + cross_track_tension * 0.08
        ),
        "professional_process": _clamp(
            professional_process_pull * 0.46
            + family_transition_pull * 0.18
            + task_pull * 0.10
            + world_weight * 0.14
            + _regime_bonus(regime_id, {"problem_solving": 0.08, "guided_exploration": 0.04})
        ),
        "career_decision": _clamp(
            career_pull * 0.60
            + task_pull * 0.14
            + world_weight * 0.16
            + _regime_bonus(regime_id, {"problem_solving": 0.08, "guided_exploration": 0.05})
        ),
        "structured_decision_support": _clamp(
            task_pull * 0.38
            + world_weight * 0.32
            + professional_process_pull * 0.08
            + career_pull * 0.08
            + _regime_bonus(regime_id, {"problem_solving": 0.12, "guided_exploration": 0.08})
        ),
        "emotional_support_basics": _clamp(
            support_pull * 0.42
            + self_weight * 0.26
            + repair_pull * 0.10
            + family_transition_pull * 0.06
            + _regime_bonus(regime_id, {"emotional_support": 0.12, "repair_and_deescalation": 0.08})
        ),
        "relational_repair": _clamp(
            repair_pull * 0.46
            + cross_track_tension * 0.18
            + support_pull * 0.10
            + self_weight * 0.08
            + _regime_bonus(regime_id, {"repair_and_deescalation": 0.16})
        ),
    }
    score_map["general_support_guidance"] = _clamp(0.18 + (1.0 - max(score_map.values(), default=0.0)) * 0.55)
    return _dedupe(_ranked_labels(score_map, max_count=3))


def _experience_domains(*, regime_id: str | None, self_weight: float, world_weight: float) -> tuple[str, ...]:
    task_bias = _clamp(world_weight * 0.78 + _regime_bonus(regime_id, {"problem_solving": 0.18, "guided_exploration": 0.08}))
    support_bias = _clamp(self_weight * 0.76 + _regime_bonus(regime_id, {"emotional_support": 0.18, "repair_and_deescalation": 0.06}))
    repair_bias = _clamp(self_weight * 0.32 + _regime_bonus(regime_id, {"repair_and_deescalation": 0.34}))
    score_map = {
        "structured_decision_patterns": task_bias,
        "repair_patterns": repair_bias,
        "stabilization_patterns": support_bias,
    }
    score_map["general_guidance_patterns"] = _clamp(0.20 + (1.0 - max(score_map.values(), default=0.0)) * 0.55)
    return _dedupe(_ranked_labels(score_map, max_count=2))


def _knowledge_weight(*, regime_id: str | None, world_weight: float, self_weight: float) -> float:
    base = 0.5 + (world_weight - self_weight) * 0.35
    base += _regime_bonus(
        regime_id,
        {
            "problem_solving": 0.18,
            "guided_exploration": 0.05,
            "emotional_support": -0.18,
            "repair_and_deescalation": -0.22,
        },
    )
    return _clamp(base)


def _regime_delayed_payoff_signal(
    *,
    regime_snapshot: "RegimeSnapshot | None",
    regime_id: str | None,
    abstract_action: str | None,
) -> tuple[float, float]:
    if regime_snapshot is None or regime_id is None:
        return (0.0, 0.0)
    payoffs = tuple(payoff for payoff in regime_snapshot.delayed_payoffs if payoff.regime_id == regime_id)
    if not payoffs:
        return (0.0, 0.0)
    matching_action_payoffs = tuple(
        payoff for payoff in payoffs if abstract_action is not None and payoff.abstract_action == abstract_action
    )
    selected = matching_action_payoffs or payoffs
    mean_payoff = sum(payoff.rolling_payoff for payoff in selected) / len(selected)
    sequence_payoff = next(
        (
            payoff.rolling_payoff
            for payoff in regime_snapshot.sequence_payoffs
            if payoff.regime_sequence and payoff.regime_sequence[-1] == regime_id
        ),
        mean_payoff,
    )
    return (_clamp(mean_payoff), _clamp(sequence_payoff))


def _rare_heavy_playbook_prior(
    *,
    rare_heavy_state: ApplicationRareHeavyState | None,
    regime_id: str | None,
) -> tuple[float, float]:
    if rare_heavy_state is None or not rare_heavy_state.distilled_playbook_rules:
        return (0.0, 0.0)
    matching_rules = tuple(
        rule
        for rule in rare_heavy_state.distilled_playbook_rules
        if regime_id is None or rule.recommended_regime in {None, regime_id}
    )
    if not matching_rules:
        return (0.0, 0.0)
    return (
        _clamp(sum(rule.knowledge_weight_hint for rule in matching_rules) / len(matching_rules)),
        _clamp(sum(rule.experience_weight_hint for rule in matching_rules) / len(matching_rules)),
    )


def _continuum_mixing_hints(memory_snapshot: MemorySnapshot | None) -> tuple[float, float, float, float, tuple[str, ...]]:
    profile = _continuum_profile(memory_snapshot)
    if profile is None or not profile.bands:
        return (0.5, 0.5, 0.0, 0.0, ())
    weighted_rows: list[tuple[Any, float]] = []
    for band in profile.bands:
        vector_strength = sum(abs(value) for value in band.vector) / max(len(band.vector), 1)
        pending_strength = sum(abs(value) for value in band.pending_signal) / max(len(band.pending_signal), 1)
        signal_strength = _clamp(
            vector_strength * 0.45
            + pending_strength * 0.18
            + band.retrieval_weight * 0.12
            + band.persistence_bias * 0.15
            + (1.0 - band.update_frequency) * 0.10
        )
        weighted_rows.append((band, max(signal_strength, 0.05)))
    total_weight = sum(weight for _, weight in weighted_rows)
    continuum_position = sum(band.persistence_bias * weight for band, weight in weighted_rows) / max(total_weight, 1e-6)
    continuum_frequency = sum(band.update_frequency * weight for band, weight in weighted_rows) / max(total_weight, 1e-6)
    slow_share = sum(
        weight
        for band, weight in weighted_rows
        if band.role in {"slow-band", "meta-init"}
    ) / max(total_weight, 1e-6)
    readout_share = sum(
        weight
        for band, weight in weighted_rows
        if band.role == "readout"
    ) / max(total_weight, 1e-6)
    edge_weight = max(len(profile.reconstruction_edges), 1)
    reconstruction_pressure = sum(edge.strength for edge in profile.reconstruction_edges) / edge_weight
    active_band_ids = tuple(
        band.band_id
        for band, weight in sorted(weighted_rows, key=lambda item: -item[1])[:3]
        if weight > 0.0
    )
    return (
        _clamp(continuum_position),
        _clamp(continuum_frequency),
        _clamp(slow_share),
        _clamp(readout_share * 0.65 + reconstruction_pressure * 0.35),
        active_band_ids,
    )


def _continuum_target_position(
    *,
    regime_id: str | None,
    world_weight: float,
    self_weight: float,
) -> float:
    if regime_id == "repair_and_deescalation":
        return 0.82
    if regime_id == "emotional_support":
        return 0.72
    if regime_id == "guided_exploration":
        return 0.56
    if regime_id == "problem_solving":
        return 0.44
    return _clamp(0.35 + self_weight * 0.35 + (1.0 - world_weight) * 0.12)


def _continuum_source_bonus(source: str) -> float:
    return {
        "context-reset-reconstruction": 0.10,
        "slow-to-fast-reuse": 0.08,
        "associative-readout": 0.06,
        "reset-prior": 0.05,
        "persisted-case-fallback": 0.04,
        "rare-heavy-cluster": 0.04,
        "artifact-anchor": 0.03,
        "direct": 0.02,
    }.get(source, 0.01)


def _case_hit_playbook_score(
    *,
    hit: CaseEpisodeHit,
    regime_id: str | None,
    world_weight: float,
    self_weight: float,
) -> float:
    target_position = _continuum_target_position(
        regime_id=regime_id,
        world_weight=world_weight,
        self_weight=self_weight,
    )
    if hit.continuum_location is None:
        position_alignment = 0.5
        source_bonus = 0.0
        role_bonus = 0.0
    else:
        position_alignment = _clamp(1.0 - abs(hit.continuum_location.position - target_position))
        source_bonus = _continuum_source_bonus(hit.continuum_location.reconstruction_source)
        role_bonus = 0.0
        if regime_id in {"emotional_support", "repair_and_deescalation"} and hit.continuum_location.band_role in {
            "slow-band",
            "meta-init",
        }:
            role_bonus += 0.08
        if regime_id in {"problem_solving", "guided_exploration"} and hit.continuum_location.band_role in {
            "session-band",
            "readout",
        }:
            role_bonus += 0.08
    ordering_bonus = 0.08 if _case_hit_ordering(hit) else 0.0
    return _clamp(
        hit.relevance_score * 0.42
        + hit.outcome.confidence * 0.22
        + position_alignment * 0.24
        + source_bonus
        + role_bonus
        + ordering_bonus
    )


def _retrieval_depth(
    *,
    regime_id: str | None,
    knowledge_weight: float,
    temporal_snapshot: "TemporalAbstractionSnapshot | None",
    continuum_position: float = 0.5,
    continuum_reconstruction_pressure: float = 0.0,
) -> KnowledgeDepth:
    switch_gate = temporal_snapshot.controller_state.switch_gate if temporal_snapshot is not None else 0.0
    depth_score = _clamp(
        knowledge_weight * 0.58
        + switch_gate * 0.28
        + continuum_position * 0.10
        + continuum_reconstruction_pressure * 0.10
        + _regime_bonus(
            regime_id,
            {
                "problem_solving": 0.14,
                "guided_exploration": 0.05,
                "emotional_support": -0.04,
                "repair_and_deescalation": -0.02,
            },
        )
    )
    return _nearest_anchor_value(depth_score, KNOWLEDGE_DEPTH_ANCHORS)


def _requires_citation(knowledge_domains: tuple[str, ...]) -> bool:
    return any(domain in {"family_transition", "professional_process", "career_decision"} for domain in knowledge_domains)


def _requires_jurisdiction(knowledge_domains: tuple[str, ...]) -> bool:
    return any(domain in {"family_transition", "professional_process"} for domain in knowledge_domains)


def _has_jurisdiction_context(text: str) -> bool:
    return _semantic_similarity(text, JURISDICTION_CONTEXT_PROTOTYPE) >= 0.52


def _domain_summary(domain: str, *, regime_id: str | None) -> str:
    if domain == "family_transition":
        return (
            "Separate emotional stabilization from legal or procedural next steps, and keep any "
            "child-safety or jurisdiction-sensitive guidance explicitly bounded."
        )
    if domain == "professional_process":
        return (
            "Use sourced high-level process guidance first, and avoid definitive professional conclusions "
            "before local specifics are confirmed."
        )
    if domain == "career_decision":
        return (
            "Frame trade-offs explicitly, reduce ambiguity, and prefer the smallest next step over a full "
            "life-plan answer."
        )
    if domain == "structured_decision_support":
        return (
            "Prefer option framing, trade-off comparison, and one grounded next action instead of broad "
            "multi-branch advice."
        )
    if domain == "relational_repair":
        return (
            "Prioritize de-escalation, acknowledgement, and safety before moving into explanation or planning."
        )
    if domain == "emotional_support_basics":
        return (
            "Acknowledge the felt experience first, then add structure gradually so the response does not "
            "skip past distress."
        )
    return (
        "Keep the response grounded, bounded, and shaped by the current regime rather than defaulting to a "
        "generic information dump."
    )


def _domain_topic_tags(domain: str) -> tuple[str, ...]:
    tags = {
        "family_transition": ("family", "transition", "procedure"),
        "professional_process": ("professional", "process", "bounded-advice"),
        "career_decision": ("career", "tradeoff", "next-step"),
        "structured_decision_support": ("decision", "structure", "options"),
        "relational_repair": ("repair", "de-escalation", "safety"),
        "emotional_support_basics": ("support", "stabilization", "presence"),
        "general_support_guidance": ("support", "boundedness"),
    }
    return tags.get(domain, ("general",))


def _domain_source_type(domain: str) -> KnowledgeSourceType:
    if domain in {"family_transition", "professional_process"}:
        return KnowledgeSourceType.OFFICIAL_GUIDE
    if domain in {"relational_repair", "emotional_support_basics"}:
        return KnowledgeSourceType.INTERNAL_GUIDE
    return KnowledgeSourceType.REVIEWED_ARTICLE


def _domain_jurisdiction_tags(domain: str, *, jurisdiction_required: bool) -> tuple[str, ...]:
    if jurisdiction_required and domain in {"family_transition", "professional_process"}:
        return ("local-law-sensitive",)
    return ("general",)


def _entry_problem_pattern(entry: MemoryEntry) -> str:
    best_label = "general-guidance"
    best_score = -1.0
    for label, prototype in PROBLEM_PATTERN_PROTOTYPES:
        score = _semantic_similarity(entry.content, prototype)
        if score > best_score:
            best_label = label
            best_score = score
    return best_label


def _entry_user_state_pattern(entry: MemoryEntry) -> str:
    best_label = "mixed-signal"
    best_score = -1.0
    for label, prototype in USER_STATE_PROTOTYPES:
        score = _semantic_similarity(entry.content, prototype)
        if score > best_score:
            best_label = label
            best_score = score
    return best_label


def _entry_risk_markers(
    *,
    entry: MemoryEntry,
    prediction_error: "PredictionErrorSnapshot | None",
    retrieval_policy: RetrievalPolicySnapshot,
) -> tuple[str, ...]:
    markers: list[str] = []
    if retrieval_policy.risk_band in {RiskBand.HIGH, RiskBand.CRITICAL}:
        markers.append(f"risk-{retrieval_policy.risk_band.value}")
    child_impact_score = _semantic_similarity(entry.content, CHILD_IMPACT_PROTOTYPE)
    domain_sensitive_score = _semantic_similarity(entry.content, DOMAIN_SENSITIVE_PROTOTYPE)
    if child_impact_score >= 0.52:
        markers.append("child-impact")
    if domain_sensitive_score >= 0.52:
        markers.append("domain-sensitive")
    if prediction_error is not None and prediction_error.error.relationship_error <= -0.4:
        markers.append("relationship-instability")
    return _dedupe(tuple(markers))


def _entry_outcome_summary(
    *,
    prediction_error: "PredictionErrorSnapshot | None",
    entry: MemoryEntry,
) -> CaseOutcomeSummary:
    if prediction_error is None:
        return CaseOutcomeSummary(
            outcome_label=ExperienceOutcomeLabel.UNKNOWN,
            delayed_signal_count=0,
            escalation_observed=False,
            repair_observed=False,
            confidence=0.45,
            description=f"No prediction-error evidence yet for entry {entry.entry_id}.",
        )
    reward = prediction_error.error.signed_reward
    reward_alignment = _clamp(0.5 + reward * 0.5)
    relationship_repair_score = _clamp(0.5 + prediction_error.error.relationship_error * 0.5)
    relationship_escalation_score = _clamp(0.5 + (-prediction_error.error.relationship_error) * 0.5)
    outcome_label = _nearest_anchor_value(reward_alignment, OUTCOME_LABEL_ANCHORS)
    escalation_observed = relationship_escalation_score >= 0.68
    repair_observed = relationship_repair_score >= 0.58
    return CaseOutcomeSummary(
        outcome_label=outcome_label,
        delayed_signal_count=4,
        escalation_observed=escalation_observed,
        repair_observed=repair_observed,
        confidence=_clamp(0.52 + abs(reward - prediction_error.error.relationship_error) * 0.18 + abs(reward) * 0.18),
        description=(
            f"Outcome derived from prediction_error reward={reward:.2f} "
            f"relationship_error={prediction_error.error.relationship_error:.2f}."
        ),
    )


def _case_entries(memory_snapshot: MemorySnapshot | None, *, retrieval_policy: RetrievalPolicySnapshot) -> tuple[MemoryEntry, ...]:
    if memory_snapshot is None:
        return ()
    preferred_tracks: tuple[str, ...]
    if retrieval_policy.self_weight > retrieval_policy.world_weight:
        preferred_tracks = ("self", "shared", "world")
    else:
        preferred_tracks = ("world", "shared", "self")

    def score(entry: MemoryEntry) -> tuple[float, float, int]:
        track_priority = preferred_tracks.index(entry.track.value) if entry.track.value in preferred_tracks else len(preferred_tracks)
        experience_domain_bonus = 0.1 if retrieval_policy.experience_domains else 0.0
        return (float(track_priority), -(entry.strength + experience_domain_bonus), -entry.last_accessed_ms)

    ranked = sorted(memory_snapshot.retrieved_entries, key=score)
    return tuple(ranked[:3])


def _playbook_template(
    *,
    problem_pattern: str,
    regime_id: str | None,
    world_weight: float,
    self_weight: float,
) -> tuple[tuple[str, ...], str, tuple[str, ...]]:
    if problem_pattern == "family-transition-high-emotion":
        return (
            ("stabilize", "split_axes", "smallest_next_step"),
            "gradual",
            ("procedure-dump-too-early", "definitive-legal-conclusion"),
        )
    if problem_pattern == "structured-decision-overwhelm":
        return (
            ("narrow_scope", "option_compare", "smallest_next_step"),
            "structured",
            ("multi-branch-overload",),
        )
    if problem_pattern == "relational-repair":
        return (
            ("acknowledge", "deescalate", "bounded_next_step"),
            "slow",
            ("defensive-argumentation",),
        )
    if regime_id == "problem_solving" or world_weight >= self_weight:
        return (
            ("clarify_goal", "structure_options", "commit_next_step"),
            "structured",
            ("premature-emotional-bypass",),
        )
    return (
        ("acknowledge", "stabilize", "smallest_next_step"),
        "support-first",
        ("over-directive-solutioning",),
    )


def _case_hit_ordering(hit: CaseEpisodeHit) -> tuple[str, ...]:
    generic_labels = {"maintain-current-regime", "cluster-guided-ordering"}
    ordering = tuple(
        step.action_label
        for step in hit.intervention_steps
        if step.action_label not in generic_labels
    )
    return _dedupe(ordering)


def _case_hit_pacing(
    *,
    hit: CaseEpisodeHit,
    world_weight: float,
    self_weight: float,
) -> str:
    pacing_score = _clamp(
        (1.0 if "risk-high" in hit.risk_markers or "risk-critical" in hit.risk_markers else 0.0) * 0.32
        + (1.0 if hit.outcome.escalation_observed else 0.0) * 0.26
        + self_weight * 0.24
        + (1.0 - world_weight) * 0.18
    )
    return _nearest_anchor_value(pacing_score, PACING_ANCHORS)


def _case_hit_avoid_patterns(hit: CaseEpisodeHit) -> tuple[str, ...]:
    avoid_patterns: list[str] = []
    if "domain-sensitive" in hit.risk_markers:
        avoid_patterns.append("definitive-legal-conclusion")
    if "child-impact" in hit.risk_markers:
        avoid_patterns.append("procedure-dump-too-early")
    if hit.outcome.escalation_observed:
        avoid_patterns.append("premature-emotional-bypass")
    if not avoid_patterns:
        avoid_patterns.append("over-directive-solutioning")
    return _dedupe(tuple(avoid_patterns))


def _response_mode(
    *,
    regime_id: str | None,
    boundary_policy_snapshot: BoundaryPolicySnapshot,
    retrieval_policy_snapshot: RetrievalPolicySnapshot | None = None,
) -> ResponseMode:
    decision = boundary_policy_snapshot.active_decision
    if decision.refer_out_required:
        return ResponseMode.REFER_OUT
    if decision.clarification_required:
        return ResponseMode.CLARIFY
    if retrieval_policy_snapshot is not None:
        if retrieval_policy_snapshot.response_mode_hint == "refer-out":
            return ResponseMode.REFER_OUT
        if retrieval_policy_snapshot.response_mode_hint == "clarify":
            return ResponseMode.CLARIFY
        if retrieval_policy_snapshot.response_mode_hint == "structure":
            return ResponseMode.STRUCTURE
        if retrieval_policy_snapshot.response_mode_hint == "support":
            return ResponseMode.SUPPORT
    if regime_id in {"problem_solving", "guided_exploration"}:
        return ResponseMode.STRUCTURE
    return ResponseMode.SUPPORT


def _required_disclaimer_phrase(disclaimer: str) -> str | None:
    if disclaimer == "jurisdiction-variance":
        return "Local rules and procedures can vary by jurisdiction."
    if disclaimer == "clarify-before-concluding":
        return "I may need one missing local detail before going further."
    if disclaimer == "professional-handoff":
        return "If this has real-world consequences, appropriate professional follow-up would be the safest next step."
    return None


def _response_control_scale(
    *,
    temporal_snapshot: "TemporalAbstractionSnapshot | None",
    ordering_plan: tuple[str, ...],
    response_mode: ResponseMode,
    boundary_policy_snapshot: BoundaryPolicySnapshot,
    retrieval_policy_snapshot: RetrievalPolicySnapshot | None = None,
) -> float:
    if temporal_snapshot is None or not temporal_snapshot.controller_state.code:
        return 0.0
    gate = temporal_snapshot.controller_state.switch_gate
    boundary_bonus = 0.05 if boundary_policy_snapshot.active_decision.refer_out_required else 0.03
    ordering_bonus = 0.03 if ordering_plan else 0.0
    mode_bonus = 0.03 if response_mode in {ResponseMode.CLARIFY, ResponseMode.STRUCTURE} else 0.0
    retrieval_bonus = (
        retrieval_policy_snapshot.answer_depth_bias * 0.04
        + retrieval_policy_snapshot.clarification_bias * 0.03
        if retrieval_policy_snapshot is not None
        else 0.0
    )
    return max(0.08, min(0.28, 0.06 + gate * 0.16 + boundary_bonus + ordering_bonus + mode_bonus + retrieval_bonus))


def _response_target_position(
    *,
    regime_id: str | None,
    response_mode: ResponseMode,
    boundary_policy_snapshot: BoundaryPolicySnapshot,
    case_memory_snapshot: CaseMemorySnapshot | None,
    strategy_playbook_snapshot: StrategyPlaybookSnapshot | None,
) -> float:
    if response_mode is ResponseMode.REFER_OUT:
        return 0.88
    if response_mode is ResponseMode.CLARIFY:
        if regime_id in {"emotional_support", "repair_and_deescalation"}:
            return 0.74
        case_position = case_memory_snapshot.mean_continuum_position if case_memory_snapshot is not None else 0.0
        if case_position >= 0.68:
            return 0.74
        return 0.58
    if regime_id == "repair_and_deescalation":
        return 0.82
    if regime_id == "emotional_support":
        return 0.72
    if regime_id == "guided_exploration":
        return 0.56
    if regime_id == "problem_solving":
        playbook_position = (
            max((rule.mean_continuum_position for rule in strategy_playbook_snapshot.matched_rules), default=0.0)
            if strategy_playbook_snapshot is not None
            else 0.0
        )
        return _clamp(max(0.36, min(0.52, playbook_position or 0.42)))
    return 0.5


def _response_ordering_plan(
    *,
    regime_id: str | None,
    response_mode: ResponseMode,
    boundary_policy_snapshot: BoundaryPolicySnapshot,
    case_memory_snapshot: CaseMemorySnapshot | None,
    strategy_playbook_snapshot: StrategyPlaybookSnapshot | None,
    retrieval_policy_snapshot: RetrievalPolicySnapshot | None = None,
) -> tuple[tuple[str, ...], float, str]:
    target_position = (
        retrieval_policy_snapshot.continuum_target_position_hint
        if retrieval_policy_snapshot is not None
        else _response_target_position(
            regime_id=regime_id,
            response_mode=response_mode,
            boundary_policy_snapshot=boundary_policy_snapshot,
            case_memory_snapshot=case_memory_snapshot,
            strategy_playbook_snapshot=strategy_playbook_snapshot,
        )
    )
    playbook_ordering = (
        strategy_playbook_snapshot.matched_rules[0].recommended_ordering
        if strategy_playbook_snapshot is not None and strategy_playbook_snapshot.matched_rules
        else ()
    )
    retrieval_ordering = retrieval_policy_snapshot.ordering_bias if retrieval_policy_snapshot is not None else ()
    playbook_support_first = bool(playbook_ordering) and playbook_ordering[0] in {"stabilize", "acknowledge", "deescalate"}
    if response_mode is ResponseMode.REFER_OUT:
        prefix = ("stabilize", "bounded_handoff")
        driver = "continuum-refer-out"
    elif response_mode is ResponseMode.CLARIFY:
        if playbook_support_first or target_position >= 0.66:
            prefix = ("stabilize", "clarify_goal")
            driver = "continuum-support-clarify"
        else:
            prefix = ("clarify_goal",)
            driver = "continuum-clarify-first"
    elif target_position >= 0.66:
        prefix = ("stabilize", "acknowledge")
        driver = (
            retrieval_policy_snapshot.ordering_driver_hint
            if retrieval_policy_snapshot is not None and retrieval_policy_snapshot.ordering_driver_hint != "retrieval-fallback"
            else "continuum-support-first"
        )
    elif target_position >= 0.52:
        prefix = ("clarify_goal", "split_axes")
        driver = (
            retrieval_policy_snapshot.ordering_driver_hint
            if retrieval_policy_snapshot is not None and retrieval_policy_snapshot.ordering_driver_hint != "retrieval-fallback"
            else "continuum-guided-clarify"
        )
    else:
        prefix = ("structure_options", "smallest_next_step")
        driver = (
            retrieval_policy_snapshot.ordering_driver_hint
            if retrieval_policy_snapshot is not None and retrieval_policy_snapshot.ordering_driver_hint != "retrieval-fallback"
            else "continuum-structure-first"
        )
    fallback_suffix = {
        "continuum-refer-out": ("smallest_next_step",),
        "continuum-support-clarify": ("smallest_next_step",),
        "continuum-clarify-first": ("smallest_next_step",),
        "continuum-support-first": ("smallest_next_step",),
        "continuum-guided-clarify": ("smallest_next_step",),
        "continuum-structure-first": ("commit_next_step",),
    }.get(driver, ("smallest_next_step",))
    ordering_plan = _dedupe(prefix + retrieval_ordering + playbook_ordering + fallback_suffix)
    return ordering_plan, target_position, driver


def _prompt_residue_summary(
    *,
    regime_name: str,
    regime_switched: bool,
    memory_snapshot: MemorySnapshot | None,
    reflection_snapshot: object | None,
    temporal_snapshot: "TemporalAbstractionSnapshot | None",
) -> tuple[str, float]:
    from volvence_zero.reflection import ReflectionSnapshot

    residue_parts = [f"Current mode: {regime_name}."]
    if regime_switched:
        residue_parts.append("The interaction frame has shifted, so acknowledge that naturally.")
    if memory_snapshot is not None and memory_snapshot.retrieved_entries:
        residue_parts.append(
            "Carry forward continuity from prior context: "
            + _truncate_text(memory_snapshot.retrieved_entries[0].content, max_chars=88)
        )
    if isinstance(reflection_snapshot, ReflectionSnapshot) and reflection_snapshot.lessons_extracted:
        residue_parts.append(f"Recent lesson: {next(iter(reflection_snapshot.lessons_extracted))}.")
    if isinstance(reflection_snapshot, ReflectionSnapshot) and reflection_snapshot.tensions_identified:
        residue_parts.append(f"Watch this tension: {next(iter(reflection_snapshot.tensions_identified))}.")
    if temporal_snapshot is not None:
        residue_parts.append(
            "Internal control frame: "
            + _truncate_text(temporal_snapshot.description, max_chars=96)
        )
    summary = " ".join(residue_parts)
    prompt_signal_count = len(residue_parts)
    explicit_signal_count = 5
    return summary, _clamp(prompt_signal_count / max(prompt_signal_count + explicit_signal_count, 1))


class ApplicationRareHeavyState:
    def __init__(self) -> None:
        self._domain_template_biases: dict[str, float] = {}
        self._case_clusters: tuple[ApplicationCaseCluster, ...] = ()
        self._distilled_playbook_rules: tuple[PlaybookRule, ...] = ()
        self._boundary_prior_hints: tuple[BoundaryPriorHint, ...] = ()
        self._retrieval_readout_checkpoint: RetrievalReadoutCheckpoint | None = None
        self._reviewed_knowledge_candidates: tuple[ReviewedKnowledgeCandidate, ...] = ()

    @property
    def domain_template_biases(self) -> tuple[tuple[str, float], ...]:
        return tuple(sorted(self._domain_template_biases.items()))

    @property
    def case_clusters(self) -> tuple[ApplicationCaseCluster, ...]:
        return self._case_clusters

    @property
    def distilled_playbook_rules(self) -> tuple[PlaybookRule, ...]:
        return self._distilled_playbook_rules

    @property
    def boundary_prior_hints(self) -> tuple[BoundaryPriorHint, ...]:
        return self._boundary_prior_hints

    @property
    def retrieval_readout_checkpoint(self) -> RetrievalReadoutCheckpoint | None:
        return self._retrieval_readout_checkpoint

    @property
    def reviewed_knowledge_candidates(self) -> tuple[ReviewedKnowledgeCandidate, ...]:
        return self._reviewed_knowledge_candidates

    def upsert_distilled_playbook_rules(self, rules: tuple[PlaybookRule, ...]) -> tuple[str, ...]:
        by_pattern = {rule.problem_pattern: rule for rule in self._distilled_playbook_rules}
        for rule in rules:
            existing = by_pattern.get(rule.problem_pattern)
            if existing is None or rule.confidence >= existing.confidence:
                by_pattern[rule.problem_pattern] = rule
        self._distilled_playbook_rules = tuple(
            sorted(by_pattern.values(), key=lambda rule: (rule.problem_pattern, rule.rule_id))
        )
        return tuple(f"application-playbook-upsert:{rule.problem_pattern}" for rule in rules)

    def upsert_boundary_prior_hints(self, hints: tuple[BoundaryPriorHint, ...]) -> tuple[str, ...]:
        by_key = {
            (hint.regime_id, hint.trigger_reasons): hint
            for hint in self._boundary_prior_hints
        }
        for hint in hints:
            key = (hint.regime_id, hint.trigger_reasons)
            existing = by_key.get(key)
            if existing is None or hint.confidence >= existing.confidence:
                by_key[key] = hint
        self._boundary_prior_hints = tuple(
            sorted(
                by_key.values(),
                key=lambda hint: (
                    hint.regime_id or "",
                    ",".join(hint.trigger_reasons),
                    hint.hint_id,
                ),
            )
        )
        return tuple(
            f"application-boundary-hint-upsert:{hint.regime_id or 'shared'}:{len(hint.trigger_reasons)}"
            for hint in hints
        )

    def apply_retrieval_readout_checkpoint(self, checkpoint: RetrievalReadoutCheckpoint) -> tuple[str, ...]:
        existing = self._retrieval_readout_checkpoint
        if existing is None or checkpoint.confidence >= existing.confidence:
            self._retrieval_readout_checkpoint = checkpoint
            return ("application-retrieval-readout-checkpoint-upsert",)
        return ("application-retrieval-readout-checkpoint-skip-lower-confidence",)

    def export_rare_heavy_state(self, *, checkpoint_id: str) -> ApplicationRareHeavyCheckpoint:
        return ApplicationRareHeavyCheckpoint(
            checkpoint_id=checkpoint_id,
            domain_template_biases=self.domain_template_biases,
            case_clusters=self._case_clusters,
            distilled_playbook_rules=self._distilled_playbook_rules,
            boundary_prior_hints=self._boundary_prior_hints,
            reviewed_knowledge_candidates=self._reviewed_knowledge_candidates,
            continuum_profile_id=None,
            retrieval_readout_checkpoint=self._retrieval_readout_checkpoint,
            description=(
                f"Application rare-heavy checkpoint with {len(self._domain_template_biases)} domain biases, "
                f"{len(self._case_clusters)} case clusters, {len(self._distilled_playbook_rules)} playbook rules, "
                f"{len(self._boundary_prior_hints)} boundary prior hints, "
                f"{len(self._reviewed_knowledge_candidates)} reviewed knowledge candidates, and "
                f"{'a' if self._retrieval_readout_checkpoint is not None else 'no'} retrieval readout checkpoint."
            ),
        )

    def import_rare_heavy_state(self, checkpoint: ApplicationRareHeavyCheckpoint) -> tuple[str, ...]:
        self._domain_template_biases = dict(checkpoint.domain_template_biases)
        self._case_clusters = checkpoint.case_clusters
        self._distilled_playbook_rules = checkpoint.distilled_playbook_rules
        self._boundary_prior_hints = checkpoint.boundary_prior_hints
        self._retrieval_readout_checkpoint = checkpoint.retrieval_readout_checkpoint
        self._reviewed_knowledge_candidates = checkpoint.reviewed_knowledge_candidates
        return (
            "rare-heavy:application-domain-refresh",
            "rare-heavy:application-case-clusters-import",
            "rare-heavy:application-playbook-import",
            "rare-heavy:application-boundary-import",
            "rare-heavy:application-retrieval-readout-import",
            "rare-heavy:application-reviewed-knowledge-import",
        )

    def restore_rare_heavy_state(self, checkpoint: ApplicationRareHeavyCheckpoint) -> tuple[str, ...]:
        self._domain_template_biases = dict(checkpoint.domain_template_biases)
        self._case_clusters = checkpoint.case_clusters
        self._distilled_playbook_rules = checkpoint.distilled_playbook_rules
        self._boundary_prior_hints = checkpoint.boundary_prior_hints
        self._retrieval_readout_checkpoint = checkpoint.retrieval_readout_checkpoint
        self._reviewed_knowledge_candidates = checkpoint.reviewed_knowledge_candidates
        return (
            "rare-heavy:application-domain-rollback",
            "rare-heavy:application-case-clusters-rollback",
            "rare-heavy:application-playbook-rollback",
            "rare-heavy:application-boundary-rollback",
            "rare-heavy:application-retrieval-readout-rollback",
            "rare-heavy:application-reviewed-knowledge-rollback",
        )


class ExperienceFastPriorModule(RuntimeModule[ExperienceFastPriorSnapshot]):
    slot_name = "experience_fast_prior"
    owner = "ExperienceFastPriorModule"
    value_type = ExperienceFastPriorSnapshot
    dependencies = ("experience_consolidation",)
    default_wiring_level = WiringLevel.SHADOW

    def publish_snapshot(
        self,
        *,
        experience_consolidation_snapshot: ExperienceConsolidationSnapshot | None = None,
    ) -> Snapshot[ExperienceFastPriorSnapshot]:
        consolidation = experience_consolidation_snapshot
        if consolidation is None:
            return self.publish(
                ExperienceFastPriorSnapshot(
                    regime_biases=(),
                    knowledge_weight_bias=0.0,
                    experience_weight_bias=0.0,
                    action_biases=(),
                    family_biases=(),
                    sequence_biases=(),
                    prior_strength=0.0,
                    source_attribution_ids=(),
                    source_sequence_ids=(),
                    description="Experience fast prior unavailable because experience_consolidation is absent.",
                )
            )

        attribution_ids: list[str] = []
        sequence_ids: list[str] = []
        regime_bias_map: dict[str, list[tuple[float, str]]] = {}
        action_bias_map: dict[str, list[tuple[float, str]]] = {}
        family_bias_map: dict[int, list[tuple[float, str]]] = {}
        sequence_bias_map: dict[tuple[tuple[str, ...], int], list[tuple[float, str]]] = {}
        mix_bias_values: list[float] = []

        for attribution in consolidation.delayed_outcome_ledger:
            attribution_ids.append(attribution.attribution_id)
            outcome_direction = _signed_centered(attribution.outcome_score)
            regime_bias = outcome_direction * _signed_centered(attribution.regime_alignment) * 0.20
            if attribution.regime_id is not None:
                regime_bias_map.setdefault(attribution.regime_id, []).append((regime_bias, attribution.attribution_id))
            action_bias = outcome_direction * _signed_centered(attribution.abstract_action_alignment) * 0.18
            if attribution.abstract_action is not None:
                action_bias_map.setdefault(attribution.abstract_action, []).append(
                    (action_bias, attribution.attribution_id)
                )
            family_bias = (
                outcome_direction * 0.10
                + _signed_centered(attribution.abstract_action_alignment) * 0.08
                + _signed_centered(attribution.regime_alignment) * 0.04
            )
            if attribution.action_family_version > 0:
                family_bias_map.setdefault(attribution.action_family_version, []).append(
                    (family_bias, attribution.attribution_id)
                )
            mix_alignment_direction = _signed_centered(attribution.retrieval_mix_alignment)
            retrieval_lean = attribution.experience_weight - attribution.knowledge_weight
            mix_bias_values.append(outcome_direction * mix_alignment_direction * retrieval_lean * 0.22)

        for sequence_payoff in consolidation.sequence_payoffs:
            sequence_ids.append(sequence_payoff.sequence_id)
            payoff_bias = (
                _signed_centered(sequence_payoff.rolling_payoff)
                * min(sequence_payoff.sample_count / 3.0, 1.0)
                * 0.18
            )
            sequence_key = (sequence_payoff.regime_sequence, sequence_payoff.action_family_version)
            sequence_bias_map.setdefault(sequence_key, []).append((payoff_bias, sequence_payoff.sequence_id))

        delayed_credit_summary = consolidation.delayed_credit_summary
        if delayed_credit_summary is not None:
            summary_id = delayed_credit_summary.summary_id
            attribution_ids.append(summary_id)
            summary_direction = _signed_centered(delayed_credit_summary.outcome_score)
            if delayed_credit_summary.regime_id is not None:
                regime_bias_map.setdefault(delayed_credit_summary.regime_id, []).append(
                    (
                        summary_direction * _signed_centered(delayed_credit_summary.regime_alignment) * 0.22,
                        summary_id,
                    )
                )
            if delayed_credit_summary.abstract_action is not None:
                action_bias_map.setdefault(delayed_credit_summary.abstract_action, []).append(
                    (
                        summary_direction * _signed_centered(delayed_credit_summary.abstract_action_alignment) * 0.20,
                        summary_id,
                    )
                )
            if delayed_credit_summary.action_family_version > 0:
                family_bias_map.setdefault(delayed_credit_summary.action_family_version, []).append(
                    (
                        summary_direction * 0.12
                        + _signed_centered(delayed_credit_summary.abstract_action_alignment) * 0.10
                        + _signed_centered(delayed_credit_summary.sequence_payoff) * 0.08,
                        summary_id,
                    )
                )
                sequence_key = (
                    ((delayed_credit_summary.regime_id,) if delayed_credit_summary.regime_id is not None else ()),
                    delayed_credit_summary.action_family_version,
                )
                sequence_bias_map.setdefault(sequence_key, []).append(
                    (_signed_centered(delayed_credit_summary.sequence_payoff) * 0.20, summary_id)
                )
            retrieval_lean = delayed_credit_summary.experience_weight - delayed_credit_summary.knowledge_weight
            mix_bias_values.append(
                summary_direction
                * _signed_centered(delayed_credit_summary.retrieval_mix_alignment)
                * retrieval_lean
                * 0.24
            )

        regime_biases = tuple(
            ExperienceFastPriorRegimeBias(
                regime_id=regime_id,
                bias=_clamp_signed(sum(value for value, _ in values) / len(values)),
                source_attribution_ids=tuple(source_id for _, source_id in values),
                description=(
                    f"Fast prior regime bias for {regime_id} derived from {len(values)} delayed attribution(s)."
                ),
            )
            for regime_id, values in sorted(regime_bias_map.items())
        )
        action_biases = tuple(
            ExperienceFastPriorActionBias(
                abstract_action=abstract_action,
                bias=_clamp_signed(sum(value for value, _ in values) / len(values)),
                source_attribution_ids=tuple(source_id for _, source_id in values),
                description=(
                    f"Fast prior action bias for {abstract_action} derived from {len(values)} delayed attribution(s)."
                ),
            )
            for abstract_action, values in sorted(action_bias_map.items())
        )
        family_biases = tuple(
            ExperienceFastPriorFamilyBias(
                action_family_version=family_version,
                continuation_bias=_clamp_signed(sum(value for value, _ in values) / len(values)),
                source_attribution_ids=tuple(source_id for _, source_id in values),
                description=(
                    f"Fast prior family bias for action_family_version={family_version} derived from "
                    f"{len(values)} delayed attribution(s)."
                ),
            )
            for family_version, values in sorted(family_bias_map.items())
        )
        sequence_biases = tuple(
            ExperienceFastPriorSequenceBias(
                regime_sequence=regime_sequence,
                action_family_version=family_version,
                payoff_bias=_clamp_signed(sum(value for value, _ in values) / len(values)),
                source_sequence_ids=tuple(source_id for _, source_id in values),
                description=(
                    f"Fast prior sequence bias for regime_sequence={regime_sequence} family_version={family_version} "
                    f"derived from {len(values)} sequence payoff(s)."
                ),
            )
            for (regime_sequence, family_version), values in sorted(sequence_bias_map.items())
        )
        experience_weight_bias = _clamp_signed(
            sum(mix_bias_values) / len(mix_bias_values) if mix_bias_values else 0.0
        )
        knowledge_weight_bias = _clamp_signed(-experience_weight_bias)
        prior_strength_terms = (
            tuple(abs(item.bias) for item in regime_biases)
            + tuple(abs(item.bias) for item in action_biases)
            + tuple(abs(item.continuation_bias) for item in family_biases)
            + tuple(abs(item.payoff_bias) for item in sequence_biases)
            + ((abs(experience_weight_bias) + abs(knowledge_weight_bias)) / 2.0,)
        )
        prior_strength = _clamp(
            sum(prior_strength_terms) / len(prior_strength_terms) if prior_strength_terms else 0.0
        )
        return self.publish(
            ExperienceFastPriorSnapshot(
                regime_biases=regime_biases,
                knowledge_weight_bias=knowledge_weight_bias,
                experience_weight_bias=experience_weight_bias,
                action_biases=action_biases,
                family_biases=family_biases,
                sequence_biases=sequence_biases,
                prior_strength=prior_strength,
                source_attribution_ids=tuple(attribution_ids),
                source_sequence_ids=tuple(sequence_ids),
                description=(
                    f"Experience fast prior published {len(regime_biases)} regime bias(es), "
                    f"{len(action_biases)} action bias(es), {len(family_biases)} family bias(es), and "
                    f"{len(sequence_biases)} sequence bias(es)."
                ),
            )
        )

    async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[ExperienceFastPriorSnapshot]:
        consolidation_value = upstream["experience_consolidation"].value
        consolidation = (
            consolidation_value if isinstance(consolidation_value, ExperienceConsolidationSnapshot) else None
        )
        return self.publish_snapshot(experience_consolidation_snapshot=consolidation)

    async def process_standalone(
        self,
        *,
        experience_consolidation_snapshot: ExperienceConsolidationSnapshot | None = None,
        **kwargs: Any,
    ) -> Snapshot[ExperienceFastPriorSnapshot]:
        del kwargs
        return self.publish_snapshot(experience_consolidation_snapshot=experience_consolidation_snapshot)


class RetrievalPolicyModule(RuntimeModule[RetrievalPolicySnapshot]):
    slot_name = "retrieval_policy"
    owner = "RetrievalPolicyModule"
    value_type = RetrievalPolicySnapshot
    dependencies = ("world_temporal", "self_temporal", "dual_track", "regime", "memory", "experience_fast_prior")
    default_wiring_level = WiringLevel.ACTIVE

    def __init__(
        self,
        *,
        rare_heavy_state: ApplicationRareHeavyState | None = None,
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._rare_heavy_state = rare_heavy_state

    async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[RetrievalPolicySnapshot]:
        from volvence_zero.regime import RegimeSnapshot
        from volvence_zero.temporal import ControllerState, TemporalAbstractionSnapshot

        world_temporal_snapshot = upstream["world_temporal"].value
        self_temporal_snapshot = upstream["self_temporal"].value
        dual_track_snapshot = upstream["dual_track"].value
        regime_snapshot = upstream["regime"].value
        memory_snapshot = upstream["memory"].value
        experience_fast_prior_snapshot = upstream["experience_fast_prior"].value
        if not isinstance(world_temporal_snapshot, TemporalAbstractionSnapshot):
            world_temporal_snapshot = TemporalAbstractionSnapshot(
                controller_state=ControllerState(
                    code=(0.0, 0.0, 0.0),
                    code_dim=3,
                    switch_gate=0.0,
                    is_switching=False,
                    steps_since_switch=0,
                ),
                active_abstract_action="temporal-disabled-placeholder",
                controller_params_hash="temporal-disabled",
                description="world_temporal disabled; retrieval policy fell back to placeholder temporal state.",
            )
        if not isinstance(self_temporal_snapshot, TemporalAbstractionSnapshot):
            self_temporal_snapshot = TemporalAbstractionSnapshot(
                controller_state=ControllerState(
                    code=(0.0, 0.0, 0.0),
                    code_dim=3,
                    switch_gate=0.0,
                    is_switching=False,
                    steps_since_switch=0,
                ),
                active_abstract_action="temporal-disabled-placeholder",
                controller_params_hash="temporal-disabled",
                description="self_temporal disabled; retrieval policy fell back to placeholder temporal state.",
            )
        if not isinstance(dual_track_snapshot, DualTrackSnapshot):
            raise TypeError("dual_track must publish DualTrackSnapshot.")
        if not isinstance(regime_snapshot, RegimeSnapshot):
            raise TypeError("regime must publish RegimeSnapshot.")
        experience_fast_prior = (
            experience_fast_prior_snapshot
            if isinstance(experience_fast_prior_snapshot, ExperienceFastPriorSnapshot)
            else None
        )
        memory_value = memory_snapshot if isinstance(memory_snapshot, MemorySnapshot) else None
        world_weight, self_weight = _infer_track_weights(dual_track_snapshot)
        regime_id = regime_snapshot.active_regime.regime_id
        active_temporal_snapshot = (
            world_temporal_snapshot if world_weight >= self_weight else self_temporal_snapshot
        )
        abstract_action = active_temporal_snapshot.active_abstract_action
        knowledge_domains = _knowledge_domains(
            dual_track_snapshot=dual_track_snapshot,
            regime_id=regime_id,
            world_weight=world_weight,
            self_weight=self_weight,
            abstract_action=abstract_action,
        )
        experience_domains = _experience_domains(
            regime_id=regime_id,
            self_weight=self_weight,
            world_weight=world_weight,
        )
        (
            continuum_position,
            continuum_frequency,
            continuum_slow_share,
            continuum_reconstruction_pressure,
            continuum_active_band_ids,
        ) = _continuum_mixing_hints(memory_value)
        if self._rare_heavy_state is not None and self._rare_heavy_state.domain_template_biases:
            boosted = sorted(
                self._rare_heavy_state.domain_template_biases,
                key=lambda item: -item[1],
            )
            for domain, weight in boosted[:2]:
                if weight >= 0.55 and domain not in knowledge_domains:
                    knowledge_domains = knowledge_domains + (domain,)
        delayed_payoff_signal, sequence_payoff_signal = _regime_delayed_payoff_signal(
            regime_snapshot=regime_snapshot,
            regime_id=regime_id,
            abstract_action=abstract_action,
        )
        playbook_knowledge_hint, playbook_experience_hint = _rare_heavy_playbook_prior(
            rare_heavy_state=self._rare_heavy_state,
            regime_id=regime_id,
        )
        regime_fast_prior_bias = 0.0
        action_fast_prior_bias = 0.0
        family_fast_prior_bias = 0.0
        if experience_fast_prior is not None:
            regime_fast_prior_bias = next(
                (item.bias for item in experience_fast_prior.regime_biases if item.regime_id == regime_id),
                0.0,
            )
            action_fast_prior_bias = next(
                (
                    item.bias
                    for item in experience_fast_prior.action_biases
                    if item.abstract_action == abstract_action
                ),
                0.0,
            )
            family_fast_prior_bias = next(
                (
                    item.continuation_bias
                    for item in experience_fast_prior.family_biases
                    if item.action_family_version == active_temporal_snapshot.action_family_version
                ),
                0.0,
            )
        retrieval_readout_parameters = (
            self._rare_heavy_state.retrieval_readout_checkpoint.parameters
            if self._rare_heavy_state is not None and self._rare_heavy_state.retrieval_readout_checkpoint is not None
            else RetrievalControlReadoutParameters.default()
        )
        control_readout = RetrievalControlReadoutStrategy(parameters=retrieval_readout_parameters).build(
            RetrievalControlReadoutInputs(
                regime_id=regime_id,
                abstract_action=abstract_action,
                action_family_version=active_temporal_snapshot.action_family_version,
                switch_gate=active_temporal_snapshot.controller_state.switch_gate,
                knowledge_domains=knowledge_domains,
                experience_domains=experience_domains,
                world_weight=world_weight,
                self_weight=self_weight,
                continuum_position=continuum_position,
                continuum_frequency=continuum_frequency,
                continuum_slow_share=continuum_slow_share,
                continuum_reconstruction_pressure=continuum_reconstruction_pressure,
                delayed_payoff_signal=delayed_payoff_signal,
                sequence_payoff_signal=sequence_payoff_signal,
                playbook_knowledge_hint=playbook_knowledge_hint,
                playbook_experience_hint=playbook_experience_hint,
                knowledge_weight_bias=(
                    experience_fast_prior.knowledge_weight_bias if experience_fast_prior is not None else 0.0
                ),
                experience_weight_bias=(
                    experience_fast_prior.experience_weight_bias if experience_fast_prior is not None else 0.0
                ),
                regime_fast_prior_bias=regime_fast_prior_bias,
                action_fast_prior_bias=action_fast_prior_bias,
                family_fast_prior_bias=family_fast_prior_bias,
                fast_prior_strength=experience_fast_prior.prior_strength if experience_fast_prior is not None else 0.0,
                fast_prior_attribution_count=(
                    len(experience_fast_prior.source_attribution_ids) if experience_fast_prior is not None else 0
                ),
                fast_prior_sequence_count=(
                    len(experience_fast_prior.source_sequence_ids) if experience_fast_prior is not None else 0
                ),
                continuum_active_band_ids=continuum_active_band_ids,
            )
        )
        knowledge_domains = control_readout.knowledge_domains
        experience_domains = control_readout.experience_domains
        knowledge_weight = control_readout.knowledge_weight
        retrieval_depth = _retrieval_depth(
            regime_id=regime_id,
            knowledge_weight=knowledge_weight,
            temporal_snapshot=active_temporal_snapshot,
            continuum_position=continuum_position,
            continuum_reconstruction_pressure=continuum_reconstruction_pressure,
        )
        citation_required = _requires_citation(knowledge_domains)
        jurisdiction_required = _requires_jurisdiction(knowledge_domains)
        risk_band = _risk_band_from_state(
            dual_track_snapshot=dual_track_snapshot,
            temporal_snapshot=active_temporal_snapshot,
            regime_snapshot=regime_snapshot,
        )
        experience_weight = control_readout.experience_weight
        intent_description = (
            f"retrieval policy regime={regime_id} abstract_action={abstract_action} "
            f"knowledge_weight={knowledge_weight:.2f} experience_weight={experience_weight:.2f} "
            f"world_weight={world_weight:.2f} self_weight={self_weight:.2f} "
            f"continuum_position={continuum_position:.2f} slow_share={continuum_slow_share:.2f} "
            f"reconstruction_pressure={continuum_reconstruction_pressure:.2f}."
        )
        return self.publish(
            RetrievalPolicySnapshot(
                knowledge_domains=knowledge_domains,
                experience_domains=experience_domains,
                knowledge_weight=knowledge_weight,
                experience_weight=experience_weight,
                world_weight=world_weight,
                self_weight=self_weight,
                retrieval_depth=retrieval_depth,
                citation_required=citation_required,
                jurisdiction_required=jurisdiction_required,
                risk_band=risk_band,
                regime_id=regime_id,
                abstract_action=abstract_action,
                intent_description=intent_description,
                knowledge_domain_ranking=control_readout.knowledge_domain_ranking,
                experience_domain_ranking=control_readout.experience_domain_ranking,
                response_mode_hint=control_readout.response_mode_hint,
                clarification_bias=control_readout.clarification_bias,
                refer_out_bias=control_readout.refer_out_bias,
                answer_depth_bias=control_readout.answer_depth_bias,
                continuum_target_position_hint=control_readout.continuum_target_position,
                ordering_bias=control_readout.ordering_bias,
                ordering_driver_hint=control_readout.ordering_driver,
                description=(
                    f"{control_readout.description} Retrieval policy remains a compact control surface; "
                    "knowledge and experience owners publish evidence and priors without entering ETA ownership. "
                    f"checkpoint={'present' if self._rare_heavy_state is not None and self._rare_heavy_state.retrieval_readout_checkpoint is not None else 'default'}."
                ),
            )
        )

    async def process_standalone(self, **kwargs: Any) -> Snapshot[RetrievalPolicySnapshot]:
        raise NotImplementedError("RetrievalPolicyModule should be driven by upstream runtime state.")


class DomainKnowledgeModule(RuntimeModule[DomainKnowledgeSnapshot]):
    slot_name = "domain_knowledge"
    owner = "DomainKnowledgeModule"
    value_type = DomainKnowledgeSnapshot
    dependencies = ("retrieval_policy", "memory", "dual_track", "regime")
    default_wiring_level = WiringLevel.ACTIVE

    def __init__(
        self,
        *,
        rare_heavy_state: ApplicationRareHeavyState | None = None,
        store: ApplicationDomainKnowledgeStore | None = None,
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._rare_heavy_state = rare_heavy_state
        self._store = store

    async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[DomainKnowledgeSnapshot]:
        from volvence_zero.regime import RegimeSnapshot

        retrieval_policy = upstream["retrieval_policy"].value
        memory_snapshot = upstream["memory"].value
        dual_track_snapshot = upstream["dual_track"].value
        regime_snapshot = upstream["regime"].value
        if not isinstance(retrieval_policy, RetrievalPolicySnapshot):
            raise TypeError("retrieval_policy must publish RetrievalPolicySnapshot.")
        if not isinstance(dual_track_snapshot, DualTrackSnapshot):
            raise TypeError("dual_track must publish DualTrackSnapshot.")
        if not isinstance(regime_snapshot, RegimeSnapshot):
            raise TypeError("regime must publish RegimeSnapshot.")
        memory_value = memory_snapshot if isinstance(memory_snapshot, MemorySnapshot) else None
        memory_text = _memory_text(memory_value)
        unresolved_conflicts: list[str] = []
        if retrieval_policy.jurisdiction_required and not _has_jurisdiction_context(memory_text):
            unresolved_conflicts.append("jurisdiction-unspecified")
        hits: list[KnowledgeHit] = []
        domain_biases = dict(self._rare_heavy_state.domain_template_biases) if self._rare_heavy_state is not None else {}
        records = (
            self._store.query(
                domains=retrieval_policy.knowledge_domains,
                query_text=f"{memory_text} {retrieval_policy.intent_description}",
                jurisdiction_required=retrieval_policy.jurisdiction_required,
                limit=3,
            )
            if self._store is not None
            else ()
        )
        if records:
            for record in records:
                confidence = _clamp(record.confidence + domain_biases.get(record.domain, 0.0) * 0.08)
                hits.append(
                    KnowledgeHit(
                        hit_id=record.record_id,
                        domain=record.domain,
                        topic_tags=record.topic_tags,
                        jurisdiction_tags=record.jurisdiction_tags,
                        freshness_label=record.freshness_label,
                        confidence=confidence,
                        evidence_strength=EvidenceStrength(record.evidence_strength),
                        summary=record.summary,
                        conflict_markers=record.conflict_markers,
                        citations=(
                            KnowledgeCitation(
                                citation_id=f"{record.record_id}:primary",
                                source_type=KnowledgeSourceType(record.source_type),
                                title=record.title,
                                locator=record.locator,
                                snippet=record.snippet,
                                url=record.url,
                            ),
                        ),
                        description=(
                            f"Knowledge record {record.record_id} aligned to regime={retrieval_policy.regime_id} "
                            f"and world_weight={retrieval_policy.world_weight:.2f}."
                        ),
                    )
                )
        else:
            for index, domain in enumerate(retrieval_policy.knowledge_domains[:3], start=1):
                source_type = _domain_source_type(domain)
                confidence = _clamp(
                    0.48
                    + retrieval_policy.knowledge_weight * 0.35
                    + (0.05 if memory_text else 0.0)
                    + domain_biases.get(domain, 0.0) * 0.08
                )
                hit = KnowledgeHit(
                    hit_id=f"{domain}:{index}",
                    domain=domain,
                    topic_tags=_domain_topic_tags(domain),
                    jurisdiction_tags=_domain_jurisdiction_tags(
                        domain,
                        jurisdiction_required=retrieval_policy.jurisdiction_required,
                    ),
                    freshness_label="surface-fallback-current",
                    confidence=confidence,
                    evidence_strength=(
                        EvidenceStrength.HIGH if confidence >= 0.8 else
                        EvidenceStrength.MEDIUM if confidence >= 0.6 else
                        EvidenceStrength.LOW
                    ),
                    summary=_domain_summary(domain, regime_id=retrieval_policy.regime_id),
                    conflict_markers=("jurisdiction-unspecified",)
                    if "jurisdiction-unspecified" in unresolved_conflicts and domain in {"family_transition", "professional_process"}
                    else (),
                    citations=(
                        KnowledgeCitation(
                            citation_id=f"{domain}:primary",
                            source_type=source_type,
                            title=f"{domain.replace('_', ' ')} guidance",
                            locator="surface-fallback",
                            snippet=_domain_summary(domain, regime_id=retrieval_policy.regime_id),
                            url=None,
                        ),
                    ),
                    description=(
                        f"Fallback knowledge hit for {domain} aligned to regime={retrieval_policy.regime_id} "
                        f"and world_weight={retrieval_policy.world_weight:.2f}; compact evidence only."
                    ),
                )
                hits.append(hit)
        retrieval_policy_id = f"policy:{hash(retrieval_policy.intent_description) & 0xFFFF:04x}"
        return self.publish(
            DomainKnowledgeSnapshot(
                retrieval_policy_id=retrieval_policy_id,
                active_domains=retrieval_policy.knowledge_domains,
                hits=tuple(hits),
                citation_required=retrieval_policy.citation_required,
                jurisdiction_required=retrieval_policy.jurisdiction_required,
                unresolved_conflicts=tuple(unresolved_conflicts),
                description=(
                    f"Domain knowledge produced {len(hits)} compact hits for regime={regime_snapshot.active_regime.regime_id} "
                    f"with citation_required={retrieval_policy.citation_required}."
                ),
            )
        )

    async def process_standalone(self, **kwargs: Any) -> Snapshot[DomainKnowledgeSnapshot]:
        raise NotImplementedError("DomainKnowledgeModule should be driven by RetrievalPolicySnapshot.")


class CaseMemoryModule(RuntimeModule[CaseMemorySnapshot]):
    slot_name = "case_memory"
    owner = "CaseMemoryModule"
    value_type = CaseMemorySnapshot
    dependencies = ("retrieval_policy", "memory", "dual_track", "prediction_error")
    default_wiring_level = WiringLevel.ACTIVE

    def __init__(
        self,
        *,
        rare_heavy_state: ApplicationRareHeavyState | None = None,
        store: ApplicationCaseMemoryStore | None = None,
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._rare_heavy_state = rare_heavy_state
        self._store = store

    async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[CaseMemorySnapshot]:
        from volvence_zero.prediction.error import PredictionErrorSnapshot

        retrieval_policy = upstream["retrieval_policy"].value
        memory_snapshot = upstream["memory"].value
        dual_track_snapshot = upstream["dual_track"].value
        prediction_error_snapshot = upstream["prediction_error"].value
        if not isinstance(retrieval_policy, RetrievalPolicySnapshot):
            raise TypeError("retrieval_policy must publish RetrievalPolicySnapshot.")
        if not isinstance(dual_track_snapshot, DualTrackSnapshot):
            raise TypeError("dual_track must publish DualTrackSnapshot.")
        prediction_error = (
            prediction_error_snapshot if isinstance(prediction_error_snapshot, PredictionErrorSnapshot) else None
        )
        memory_value = memory_snapshot if isinstance(memory_snapshot, MemorySnapshot) else None
        memory_continuum_profile = _continuum_profile(memory_value)
        entries = _case_entries(memory_value, retrieval_policy=retrieval_policy)
        hits: list[CaseEpisodeHit] = []
        active_problem_patterns: list[str] = []
        active_risk_markers: list[str] = []
        active_band_ids: list[str] = []
        continuum_positions: list[float] = []
        retrieval_policy_id = f"policy:{hash(retrieval_policy.intent_description) & 0xFFFF:04x}"
        records = (
            self._store.query(
                experience_domains=retrieval_policy.experience_domains,
                regime_id=retrieval_policy.regime_id,
                risk_band=retrieval_policy.risk_band.value,
                limit=3,
            )
            if self._store is not None
            else ()
        )
        for record in records:
            active_problem_patterns.append(record.problem_pattern)
            active_risk_markers.extend(record.risk_markers)
            continuum_location = _continuum_location_from_record(record)
            if continuum_location is not None:
                active_band_ids.append(continuum_location.band_id)
                continuum_positions.append(continuum_location.position)
            hits.append(
                CaseEpisodeHit(
                    case_id=record.case_id,
                    domain=record.domain,
                    problem_pattern=record.problem_pattern,
                    user_state_pattern=record.user_state_pattern,
                    risk_markers=record.risk_markers,
                    track_tags=record.track_tags,
                    regime_tags=record.regime_tags,
                    intervention_steps=tuple(
                        CaseInterventionStep(
                            step_id=f"{record.case_id}:{step_index}",
                            step_order=step_index,
                            regime_id=retrieval_policy.regime_id,
                            abstract_action=retrieval_policy.abstract_action,
                            action_label=step,
                            description=f"Persisted case ordering step {step}.",
                        )
                        for step_index, step in enumerate(record.intervention_ordering, start=1)
                    ),
                    outcome=CaseOutcomeSummary(
                        outcome_label=ExperienceOutcomeLabel(record.outcome_label),
                        delayed_signal_count=record.delayed_signal_count,
                        escalation_observed=record.escalation_observed,
                        repair_observed=record.repair_observed,
                        confidence=record.confidence,
                        description=record.description,
                    ),
                    relevance_score=record.relevance_score,
                    description=record.description,
                    continuum_location=continuum_location,
                )
            )
        for index, entry in enumerate(entries, start=1):
            problem_pattern = _entry_problem_pattern(entry)
            user_state_pattern = _entry_user_state_pattern(entry)
            risk_markers = _entry_risk_markers(
                entry=entry,
                prediction_error=prediction_error,
                retrieval_policy=retrieval_policy,
            )
            outcome = _entry_outcome_summary(
                prediction_error=prediction_error,
                entry=entry,
            )
            active_problem_patterns.append(problem_pattern)
            active_risk_markers.extend(risk_markers)
            continuum_location = _continuum_location_for_entry(
                entry=entry,
                memory_snapshot=memory_value,
            )
            if continuum_location is not None:
                active_band_ids.append(continuum_location.band_id)
                continuum_positions.append(continuum_location.position)
            hits.append(
                CaseEpisodeHit(
                    case_id=f"case:{entry.entry_id}:{index}",
                    domain=next(iter(retrieval_policy.experience_domains), "general_guidance_patterns"),
                    problem_pattern=problem_pattern,
                    user_state_pattern=user_state_pattern,
                    risk_markers=risk_markers,
                    track_tags=(entry.track.value,),
                    regime_tags=(retrieval_policy.regime_id,) if retrieval_policy.regime_id is not None else (),
                    intervention_steps=(
                        CaseInterventionStep(
                            step_id=f"{entry.entry_id}:1",
                            step_order=1,
                            regime_id=retrieval_policy.regime_id,
                            abstract_action=retrieval_policy.abstract_action,
                            action_label="maintain-current-regime",
                            description=(
                                f"Historical entry '{entry.content[:48]}' aligned with regime={retrieval_policy.regime_id} "
                                f"and abstract_action={retrieval_policy.abstract_action}."
                            ),
                        ),
                    ),
                    outcome=outcome,
                    relevance_score=_clamp(entry.strength * 0.7 + retrieval_policy.experience_weight * 0.3),
                    description=(
                        f"Case memory hit derived from {entry.track.value}-track memory entry "
                        f"with problem_pattern={problem_pattern}."
                    ),
                    continuum_location=continuum_location,
                )
            )
        if self._rare_heavy_state is not None and self._rare_heavy_state.case_clusters:
            existing_patterns = set(active_problem_patterns)
            for index, cluster in enumerate(self._rare_heavy_state.case_clusters[:2], start=1):
                if cluster.problem_pattern in existing_patterns:
                    active_risk_markers.extend(cluster.risk_markers)
                    continue
                continuum_location = _continuum_location_from_band(
                    profile=memory_continuum_profile,
                    band_id=cluster.continuum_band_id or "background-slow",
                    fallback_source="rare-heavy-cluster",
                )
                if continuum_location is not None:
                    continuum_location = ContinuumLocation(
                        profile_id=continuum_location.profile_id,
                        band_id=continuum_location.band_id,
                        band_role=continuum_location.band_role,
                        position=cluster.continuum_position or continuum_location.position,
                        update_frequency=continuum_location.update_frequency,
                        reconstruction_source=continuum_location.reconstruction_source,
                        description=continuum_location.description,
                    )
                    active_band_ids.append(continuum_location.band_id)
                    continuum_positions.append(continuum_location.position)
                hits.append(
                    CaseEpisodeHit(
                        case_id=f"cluster:{cluster.cluster_id}:{index}",
                        domain=next(iter(retrieval_policy.experience_domains), "general_guidance_patterns"),
                        problem_pattern=cluster.problem_pattern,
                        user_state_pattern="cluster-derived",
                        risk_markers=cluster.risk_markers,
                        track_tags=("shared",),
                        regime_tags=(retrieval_policy.regime_id,) if retrieval_policy.regime_id is not None else (),
                        intervention_steps=(
                            CaseInterventionStep(
                                step_id=f"{cluster.cluster_id}:1",
                                step_order=1,
                                regime_id=retrieval_policy.regime_id,
                                abstract_action=retrieval_policy.abstract_action,
                                action_label="cluster-guided-ordering",
                                description=cluster.description,
                            ),
                        ),
                        outcome=CaseOutcomeSummary(
                            outcome_label=ExperienceOutcomeLabel.STABLE,
                            delayed_signal_count=max(cluster.exemplar_count, 1),
                            escalation_observed=False,
                            repair_observed=False,
                            confidence=_clamp(0.5 + cluster.mean_relevance * 0.3),
                            description=f"Derived from case cluster {cluster.cluster_id}.",
                        ),
                        relevance_score=cluster.mean_relevance,
                        description=f"Cluster-derived case memory hit for pattern={cluster.problem_pattern}.",
                        continuum_location=continuum_location,
                    )
                )
                active_problem_patterns.append(cluster.problem_pattern)
                active_risk_markers.extend(cluster.risk_markers)
                existing_patterns.add(cluster.problem_pattern)
        return self.publish(
            CaseMemorySnapshot(
                retrieval_policy_id=retrieval_policy_id,
                hits=tuple(hits),
                active_problem_patterns=_dedupe(tuple(active_problem_patterns)),
                active_risk_markers=_dedupe(tuple(active_risk_markers)),
                continuum_profile_id=(
                    memory_continuum_profile.profile_id
                    if memory_continuum_profile is not None
                    else hits[0].continuum_location.profile_id
                    if hits and hits[0].continuum_location is not None
                    else None
                ),
                active_band_ids=_dedupe(tuple(active_band_ids)),
                mean_continuum_position=(
                    sum(continuum_positions) / len(continuum_positions) if continuum_positions else 0.0
                ),
                description=(
                    f"Case memory produced {len(hits)} compact case hits for "
                    f"{len(retrieval_policy.experience_domains)} experience domains."
                ),
            )
        )

    async def process_standalone(self, **kwargs: Any) -> Snapshot[CaseMemorySnapshot]:
        raise NotImplementedError("CaseMemoryModule should be driven by RetrievalPolicySnapshot.")

    @staticmethod
    def records_from_snapshot(snapshot: CaseMemorySnapshot) -> tuple[CaseMemoryRecord, ...]:
        return tuple(
            CaseMemoryRecord(
                case_id=hit.case_id,
                domain=hit.domain,
                problem_pattern=hit.problem_pattern,
                user_state_pattern=hit.user_state_pattern,
                risk_markers=hit.risk_markers,
                track_tags=hit.track_tags,
                regime_tags=hit.regime_tags,
                intervention_ordering=tuple(step.action_label for step in hit.intervention_steps),
                outcome_label=hit.outcome.outcome_label,
                delayed_signal_count=hit.outcome.delayed_signal_count,
                escalation_observed=hit.outcome.escalation_observed,
                repair_observed=hit.outcome.repair_observed,
                confidence=hit.outcome.confidence,
                relevance_score=hit.relevance_score,
                description=hit.description,
                continuum_profile_id=(
                    hit.continuum_location.profile_id if hit.continuum_location is not None else None
                ),
                continuum_band_id=(
                    hit.continuum_location.band_id if hit.continuum_location is not None else None
                ),
                continuum_position=(
                    hit.continuum_location.position if hit.continuum_location is not None else 0.0
                ),
                continuum_update_frequency=(
                    hit.continuum_location.update_frequency if hit.continuum_location is not None else 0.0
                ),
                reconstruction_source=(
                    hit.continuum_location.reconstruction_source if hit.continuum_location is not None else "direct"
                ),
            )
            for hit in snapshot.hits
        )


class StrategyPlaybookModule(RuntimeModule[StrategyPlaybookSnapshot]):
    slot_name = "strategy_playbook"
    owner = "StrategyPlaybookModule"
    value_type = StrategyPlaybookSnapshot
    dependencies = ("case_memory", "regime", "dual_track")
    default_wiring_level = WiringLevel.ACTIVE

    def __init__(
        self,
        *,
        rare_heavy_state: ApplicationRareHeavyState | None = None,
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._rare_heavy_state = rare_heavy_state

    async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[StrategyPlaybookSnapshot]:
        from volvence_zero.regime import RegimeSnapshot

        case_memory_snapshot = upstream["case_memory"].value
        regime_snapshot = upstream["regime"].value
        dual_track_snapshot = upstream["dual_track"].value
        if not isinstance(case_memory_snapshot, CaseMemorySnapshot):
            raise TypeError("case_memory must publish CaseMemorySnapshot.")
        if not isinstance(regime_snapshot, RegimeSnapshot):
            raise TypeError("regime must publish RegimeSnapshot.")
        if not isinstance(dual_track_snapshot, DualTrackSnapshot):
            raise TypeError("dual_track must publish DualTrackSnapshot.")
        world_weight, self_weight = _infer_track_weights(dual_track_snapshot)
        rules: list[PlaybookRule] = []
        active_band_ids: list[str] = []
        rare_heavy_rules = {
            rule.problem_pattern: rule
            for rule in (self._rare_heavy_state.distilled_playbook_rules if self._rare_heavy_state is not None else ())
        }
        for index, pattern in enumerate(case_memory_snapshot.active_problem_patterns, start=1):
            if pattern in rare_heavy_rules:
                rare_heavy_rule = rare_heavy_rules[pattern]
                if rare_heavy_rule.continuum_band_id is not None:
                    active_band_ids.append(rare_heavy_rule.continuum_band_id)
                rules.append(rare_heavy_rule)
                continue
            matching_hits = tuple(hit for hit in case_memory_snapshot.hits if hit.problem_pattern == pattern)
            ranked_hits = tuple(
                sorted(
                    matching_hits,
                    key=lambda hit: -_case_hit_playbook_score(
                        hit=hit,
                        regime_id=regime_snapshot.active_regime.regime_id,
                        world_weight=world_weight,
                        self_weight=self_weight,
                    ),
                )
            )
            source_hit = ranked_hits[0] if ranked_hits else None
            source_continuum_location = (
                source_hit.continuum_location if source_hit is not None else None
            )
            direct_ordering = next(
                (ordering for ordering in (_case_hit_ordering(hit) for hit in ranked_hits) if ordering),
                (),
            )
            if direct_ordering:
                if source_continuum_location is not None:
                    active_band_ids.append(source_continuum_location.band_id)
                rules.append(
                    PlaybookRule(
                        rule_id=f"playbook:case-derived:{pattern}:{index}",
                        problem_pattern=pattern,
                        recommended_regime=regime_snapshot.active_regime.regime_id,
                        recommended_ordering=direct_ordering,
                        recommended_pacing=_case_hit_pacing(
                            hit=source_hit,
                            world_weight=world_weight,
                            self_weight=self_weight,
                        ),
                        avoid_patterns=_case_hit_avoid_patterns(source_hit),
                        knowledge_weight_hint=_clamp(0.45 + world_weight * 0.30),
                        experience_weight_hint=_clamp(0.45 + self_weight * 0.30),
                        applicability_scope=(regime_snapshot.active_regime.regime_id, *source_hit.risk_markers[:2]),
                        confidence=_clamp(
                            max(source_hit.outcome.confidence, source_hit.relevance_score) * 0.7
                            + _case_hit_playbook_score(
                                hit=source_hit,
                                regime_id=regime_snapshot.active_regime.regime_id,
                                world_weight=world_weight,
                                self_weight=self_weight,
                            )
                            * 0.3
                        ),
                        continuum_band_id=(
                            source_continuum_location.band_id if source_continuum_location is not None else None
                        ),
                        mean_continuum_position=(
                            source_continuum_location.position if source_continuum_location is not None else 0.0
                        ),
                        description=(
                            f"Case-derived playbook rule for pattern={pattern} from "
                            f"{len(source_hit.intervention_steps)} intervention steps."
                        ),
                    )
                )
                continue
            ordering, pacing, avoid_patterns = _playbook_template(
                problem_pattern=pattern,
                regime_id=regime_snapshot.active_regime.regime_id,
                world_weight=world_weight,
                self_weight=self_weight,
            )
            if source_continuum_location is not None:
                active_band_ids.append(source_continuum_location.band_id)
            rules.append(
                PlaybookRule(
                    rule_id=f"playbook:{pattern}:{index}",
                    problem_pattern=pattern,
                    recommended_regime=regime_snapshot.active_regime.regime_id,
                    recommended_ordering=ordering,
                    recommended_pacing=pacing,
                    avoid_patterns=avoid_patterns,
                    knowledge_weight_hint=_clamp(0.45 + world_weight * 0.30),
                    experience_weight_hint=_clamp(0.45 + self_weight * 0.30),
                    applicability_scope=(regime_snapshot.active_regime.regime_id, *case_memory_snapshot.active_risk_markers[:2]),
                    confidence=_clamp(
                        0.45
                        + len(case_memory_snapshot.hits) * 0.06
                        + (
                            _case_hit_playbook_score(
                                hit=source_hit,
                                regime_id=regime_snapshot.active_regime.regime_id,
                                world_weight=world_weight,
                                self_weight=self_weight,
                            )
                            * 0.18
                            if source_hit is not None
                            else 0.0
                        )
                    ),
                    continuum_band_id=(
                        source_continuum_location.band_id if source_continuum_location is not None else None
                    ),
                    mean_continuum_position=(
                        source_continuum_location.position if source_continuum_location is not None else 0.0
                    ),
                    description=(
                        f"Playbook rule for pattern={pattern} aligned to regime={regime_snapshot.active_regime.regime_id} "
                        f"and cross_track_tension={dual_track_snapshot.cross_track_tension:.2f}."
                    ),
                )
            )
        return self.publish(
            StrategyPlaybookSnapshot(
                matched_problem_patterns=case_memory_snapshot.active_problem_patterns,
                matched_rules=tuple(rules),
                continuum_profile_id=case_memory_snapshot.continuum_profile_id,
                active_band_ids=_dedupe(tuple(active_band_ids)),
                description=(
                    f"Strategy playbook produced {len(rules)} matched rules from "
                    f"{len(case_memory_snapshot.hits)} case hits."
                ),
            )
        )

    async def process_standalone(self, **kwargs: Any) -> Snapshot[StrategyPlaybookSnapshot]:
        raise NotImplementedError("StrategyPlaybookModule should be driven by CaseMemorySnapshot.")


class BoundaryPolicyModule(RuntimeModule[BoundaryPolicySnapshot]):
    slot_name = "boundary_policy"
    owner = "BoundaryPolicyModule"
    value_type = BoundaryPolicySnapshot
    dependencies = ("retrieval_policy", "domain_knowledge", "regime", "prediction_error")
    default_wiring_level = WiringLevel.ACTIVE

    def __init__(
        self,
        *,
        rare_heavy_state: ApplicationRareHeavyState | None = None,
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._rare_heavy_state = rare_heavy_state

    async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[BoundaryPolicySnapshot]:
        from volvence_zero.prediction.error import PredictionErrorSnapshot
        from volvence_zero.regime import RegimeSnapshot

        retrieval_policy = upstream["retrieval_policy"].value
        domain_knowledge = upstream["domain_knowledge"].value
        regime_snapshot = upstream["regime"].value
        prediction_error = upstream["prediction_error"].value
        if not isinstance(retrieval_policy, RetrievalPolicySnapshot):
            raise TypeError("retrieval_policy must publish RetrievalPolicySnapshot.")
        if not isinstance(domain_knowledge, DomainKnowledgeSnapshot):
            raise TypeError("domain_knowledge must publish DomainKnowledgeSnapshot.")
        if not isinstance(regime_snapshot, RegimeSnapshot):
            raise TypeError("regime must publish RegimeSnapshot.")
        if not isinstance(prediction_error, PredictionErrorSnapshot):
            raise TypeError("prediction_error must publish PredictionErrorSnapshot.")
        trigger_reasons: list[str] = []
        retrieval_risk_score = dict(RISK_BAND_ANCHORS).get(retrieval_policy.risk_band, 0.18)
        prediction_risk_score = _clamp(
            min(max(prediction_error.error.magnitude, 0.0) / 4.0, 1.0) * 0.58
            + _clamp(-prediction_error.error.relationship_error) * 0.42
        )
        effective_risk_score = max(retrieval_risk_score, prediction_risk_score)
        effective_risk_band = _nearest_anchor_value(effective_risk_score, RISK_BAND_ANCHORS)
        if prediction_risk_score >= 0.82:
            trigger_reasons.append("prediction-error-critical")
        elif prediction_risk_score >= 0.58:
            trigger_reasons.append("prediction-error-high")
        clarification_score = _clamp(
            (1.0 if retrieval_policy.jurisdiction_required else 0.0) * 0.40
            + min(len(domain_knowledge.unresolved_conflicts), 3) / 3.0 * 0.45
            + (1.0 if retrieval_policy.citation_required else 0.0) * 0.15
            + retrieval_policy.clarification_bias * 0.20
        )
        clarification_required = clarification_score >= 0.5
        if clarification_required:
            trigger_reasons.append("jurisdiction-clarification-required")
        refer_out_score = _clamp(
            effective_risk_score * 0.74
            + clarification_score * 0.16
            + retrieval_policy.refer_out_bias * 0.16
            + _regime_bonus(
                retrieval_policy.regime_id,
                {
                    "repair_and_deescalation": 0.10,
                    "emotional_support": 0.04,
                },
            )
        )
        refer_out_required = refer_out_score >= 0.84
        if refer_out_required:
            trigger_reasons.append("refer-out-required")
        citation_required = retrieval_policy.citation_required
        if citation_required:
            trigger_reasons.append("citation-required")
        answer_depth_score = _clamp(
            effective_risk_score * 0.28
            + clarification_score * 0.30
            + (1.0 if citation_required else 0.0) * 0.24
            + retrieval_policy.answer_depth_bias * 0.18
            + _regime_bonus(
                retrieval_policy.regime_id,
                {
                    "emotional_support": -0.10,
                    "repair_and_deescalation": -0.08,
                    "problem_solving": 0.05,
                },
            )
        )
        answer_depth_limit = _nearest_anchor_value(answer_depth_score, ANSWER_DEPTH_ANCHORS)
        professional_scope = (
            ProfessionalScope.PROFESSIONAL_HANDOFF
            if refer_out_required
            else ProfessionalScope.DOMAIN_INFORMATION
            if citation_required or retrieval_policy.jurisdiction_required
            else ProfessionalScope.GENERAL_SUPPORT
        )
        required_disclaimers: list[str] = []
        if retrieval_policy.jurisdiction_required:
            required_disclaimers.append("jurisdiction-variance")
        if clarification_required:
            required_disclaimers.append("clarify-before-concluding")
        if refer_out_required:
            required_disclaimers.append("professional-handoff")
        blocked_topics: tuple[str, ...] = ("definitive-domain-conclusion",) if citation_required else ()
        if self._rare_heavy_state is not None and self._rare_heavy_state.boundary_prior_hints:
            matching_hints = tuple(
                hint
                for hint in self._rare_heavy_state.boundary_prior_hints
                if (
                    hint.regime_id is None
                    or hint.regime_id == retrieval_policy.regime_id
                )
                and (
                    not hint.trigger_reasons
                    or set(hint.trigger_reasons).intersection(trigger_reasons)
                )
            )
            if matching_hints:
                clarification_required = clarification_required or any(
                    hint.clarification_required for hint in matching_hints
                )
                refer_out_required = refer_out_required or any(
                    hint.refer_out_required for hint in matching_hints
                )
                if any(hint.answer_depth_limit_hint == "high-level-only" for hint in matching_hints):
                    answer_depth_limit = "high-level-only"
                elif (
                    answer_depth_limit != "high-level-only"
                    and any(hint.answer_depth_limit_hint == "support-first" for hint in matching_hints)
                ):
                    answer_depth_limit = "support-first"
                blocked_topics = _dedupe(
                    blocked_topics
                    + tuple(topic for hint in matching_hints for topic in hint.blocked_topics)
                )
                required_disclaimers = list(
                    _dedupe(
                        tuple(required_disclaimers)
                        + tuple(
                            disclaimer
                            for hint in matching_hints
                            for disclaimer in hint.required_disclaimers
                        )
                    )
                )
                trigger_reasons.append("boundary-prior-consumed")
                professional_scope = (
                    ProfessionalScope.PROFESSIONAL_HANDOFF
                    if refer_out_required
                    else professional_scope
                )
        decision = BoundaryDecision(
            decision_id=f"boundary:{regime_snapshot.active_regime.regime_id}:{answer_depth_limit}",
            risk_band=effective_risk_band,
            professional_scope=professional_scope,
            answer_depth_limit=answer_depth_limit,
            citation_required=citation_required,
            clarification_required=clarification_required,
            refer_out_required=refer_out_required,
            blocked_topics=blocked_topics,
            required_disclaimers=tuple(required_disclaimers),
            description=(
                f"Boundary policy regime={regime_snapshot.active_regime.regime_id} "
                f"risk={effective_risk_band.value} scope={professional_scope.value} depth={answer_depth_limit}."
            ),
        )
        return self.publish(
            BoundaryPolicySnapshot(
                active_decision=decision,
                trigger_reasons=tuple(trigger_reasons),
                description=(
                    f"Boundary policy published with clarification={clarification_required} "
                    f"refer_out={refer_out_required} and {len(domain_knowledge.hits)} knowledge hits."
                ),
            )
        )

    async def process_standalone(self, **kwargs: Any) -> Snapshot[BoundaryPolicySnapshot]:
        raise NotImplementedError("BoundaryPolicyModule should be driven by runtime upstream state.")


class ResponseAssemblyModule(RuntimeModule[ResponseAssemblySnapshot]):
    slot_name = "response_assembly"
    owner = "ResponseAssemblyModule"
    value_type = ResponseAssemblySnapshot
    dependencies = (
        "retrieval_policy",
        "regime",
        "temporal_abstraction",
        "memory",
        "reflection",
        "domain_knowledge",
        "case_memory",
        "strategy_playbook",
        "boundary_policy",
    )
    default_wiring_level = WiringLevel.ACTIVE

    async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[ResponseAssemblySnapshot]:
        from volvence_zero.reflection import ReflectionSnapshot
        from volvence_zero.regime import RegimeSnapshot
        from volvence_zero.temporal import TemporalAbstractionSnapshot

        regime_value = upstream["regime"].value
        temporal_value = upstream["temporal_abstraction"].value
        memory_value = upstream["memory"].value
        reflection_value = upstream["reflection"].value
        retrieval_policy_value = upstream["retrieval_policy"].value
        domain_knowledge_value = upstream["domain_knowledge"].value
        case_memory_value = upstream["case_memory"].value
        strategy_playbook_value = upstream["strategy_playbook"].value
        boundary_policy_value = upstream["boundary_policy"].value
        if not isinstance(regime_value, RegimeSnapshot):
            raise TypeError("regime must publish RegimeSnapshot.")
        if not isinstance(boundary_policy_value, BoundaryPolicySnapshot):
            raise TypeError("boundary_policy must publish BoundaryPolicySnapshot.")
        temporal_snapshot = temporal_value if isinstance(temporal_value, TemporalAbstractionSnapshot) else None
        memory_snapshot = memory_value if isinstance(memory_value, MemorySnapshot) else None
        reflection_snapshot = reflection_value if isinstance(reflection_value, ReflectionSnapshot) else None
        retrieval_policy_snapshot = (
            retrieval_policy_value if isinstance(retrieval_policy_value, RetrievalPolicySnapshot) else None
        )
        domain_knowledge_snapshot = (
            domain_knowledge_value if isinstance(domain_knowledge_value, DomainKnowledgeSnapshot) else None
        )
        case_memory_snapshot = case_memory_value if isinstance(case_memory_value, CaseMemorySnapshot) else None
        strategy_playbook_snapshot = (
            strategy_playbook_value if isinstance(strategy_playbook_value, StrategyPlaybookSnapshot) else None
        )
        regime_id = regime_value.active_regime.regime_id
        response_mode = _response_mode(
            regime_id=regime_id,
            boundary_policy_snapshot=boundary_policy_value,
            retrieval_policy_snapshot=retrieval_policy_snapshot,
        )
        ordering_plan, continuum_target_position, ordering_driver = _response_ordering_plan(
            regime_id=regime_id,
            response_mode=response_mode,
            boundary_policy_snapshot=boundary_policy_value,
            case_memory_snapshot=case_memory_snapshot,
            strategy_playbook_snapshot=strategy_playbook_snapshot,
            retrieval_policy_snapshot=retrieval_policy_snapshot,
        )
        control_code = temporal_snapshot.controller_state.code if temporal_snapshot is not None else ()
        control_scale = _response_control_scale(
            temporal_snapshot=temporal_snapshot,
            ordering_plan=ordering_plan,
            response_mode=response_mode,
            boundary_policy_snapshot=boundary_policy_value,
            retrieval_policy_snapshot=retrieval_policy_snapshot,
        )
        prompt_residue_summary, prompt_residue_ratio = _prompt_residue_summary(
            regime_name=regime_value.active_regime.name,
            regime_switched=(
                regime_value.previous_regime is not None
                and regime_value.previous_regime.regime_id != regime_id
            ),
            memory_snapshot=memory_snapshot,
            reflection_snapshot=reflection_snapshot,
            temporal_snapshot=temporal_snapshot,
        )
        required_disclaimers = boundary_policy_value.active_decision.required_disclaimers
        required_disclaimer_phrases = tuple(
            phrase
            for phrase in (
                _required_disclaimer_phrase(disclaimer)
                for disclaimer in required_disclaimers
            )
            if phrase is not None
        )
        knowledge_briefs = (
            tuple(hit.summary for hit in domain_knowledge_snapshot.hits[:2])
            if domain_knowledge_snapshot is not None
            else ()
        )
        case_briefs = (
            tuple(
                f"{hit.problem_pattern}: {hit.user_state_pattern}"
                + (
                    f" [band={hit.continuum_location.band_id} pos={hit.continuum_location.position:.2f}]"
                    if hit.continuum_location is not None
                    else ""
                )
                for hit in case_memory_snapshot.hits[:2]
            )
            if case_memory_snapshot is not None
            else ()
        )
        playbook_ordering = tuple(ordering_plan[:4])
        return self.publish(
            ResponseAssemblySnapshot(
                regime_id=regime_id,
                regime_name=regime_value.active_regime.name,
                abstract_action=temporal_snapshot.active_abstract_action if temporal_snapshot is not None else None,
                response_mode=response_mode,
                answer_depth_limit=boundary_policy_value.active_decision.answer_depth_limit,
                citation_mode="required" if boundary_policy_value.active_decision.citation_required else "optional",
                clarification_required=boundary_policy_value.active_decision.clarification_required,
                refer_out_required=boundary_policy_value.active_decision.refer_out_required,
                ordering_plan=ordering_plan,
                knowledge_briefs=knowledge_briefs,
                case_briefs=case_briefs,
                playbook_ordering=playbook_ordering,
                required_disclaimers=required_disclaimers,
                required_disclaimer_phrases=required_disclaimer_phrases,
                control_code=control_code,
                control_scale=control_scale,
                max_questions=1 if boundary_policy_value.active_decision.clarification_required else 0,
                prompt_residue_summary=prompt_residue_summary,
                prompt_residue_ratio=prompt_residue_ratio,
                knowledge_hit_count=len(domain_knowledge_snapshot.hits) if domain_knowledge_snapshot is not None else 0,
                case_hit_count=len(case_memory_snapshot.hits) if case_memory_snapshot is not None else 0,
                playbook_rule_count=(
                    len(strategy_playbook_snapshot.matched_rules)
                    if strategy_playbook_snapshot is not None
                    else 0
                ),
                risk_band=boundary_policy_value.active_decision.risk_band,
                continuum_target_position=continuum_target_position,
                ordering_driver=ordering_driver,
                description=(
                    f"Response assembly published mode={response_mode.value} depth="
                    f"{boundary_policy_value.active_decision.answer_depth_limit} "
                    f"knowledge={len(knowledge_briefs)} case={len(case_briefs)} "
                    f"ordering={len(ordering_plan)} control_scale={control_scale:.2f} "
                    f"continuum_target={continuum_target_position:.2f} driver={ordering_driver}."
                ),
            )
        )

    async def process_standalone(self, **kwargs: Any) -> Snapshot[ResponseAssemblySnapshot]:
        raise NotImplementedError("ResponseAssemblyModule should be driven by runtime upstream state.")


class ExperienceConsolidationModule(RuntimeModule[ExperienceConsolidationSnapshot]):
    slot_name = "experience_consolidation"
    owner = "ExperienceConsolidationModule"
    value_type = ExperienceConsolidationSnapshot
    dependencies = ()
    default_wiring_level = WiringLevel.ACTIVE

    def publish_snapshot(
        self,
        *,
        completed_results: tuple[Any, ...] = (),
    ) -> Snapshot[ExperienceConsolidationSnapshot]:
        deltas: list[ExperienceDelta] = []
        delayed_outcome_ledger: list[ApplicationOutcomeAttribution] = []
        sequence_payoffs: list[ApplicationSequencePayoff] = []
        source_job_id = "experience-consolidation:none"
        continuum_profile_id: str | None = None
        active_band_ids: list[str] = []
        latest_prior_update: ApplicationPriorUpdate | None = None
        latest_writeback_report: ApplicationPriorWritebackReport | None = None
        delayed_credit_summary: DelayedCreditSummary | None = None
        for result in completed_results[-4:]:
            result_deltas = getattr(result, "experience_deltas", ())
            if getattr(result, "job_id", None) is not None:
                source_job_id = result.job_id
            deltas.extend(result_deltas)
            delayed_outcome_ledger.extend(getattr(result, "delayed_outcome_ledger", ()))
            sequence_payoffs.extend(getattr(result, "sequence_payoffs", ()))
            continuum_profile_id = continuum_profile_id or getattr(result, "continuum_profile_id", None)
            active_band_ids.extend(getattr(result, "case_band_ids", ()))
            active_band_ids.extend(getattr(result, "playbook_band_ids", ()))
            result_prior_update = getattr(result, "application_prior_update", None)
            if isinstance(result_prior_update, ApplicationPriorUpdate):
                latest_prior_update = result_prior_update
            result_writeback_report = getattr(result, "application_prior_writeback_report", None)
            if isinstance(result_writeback_report, ApplicationPriorWritebackReport):
                latest_writeback_report = result_writeback_report
            result_delayed_credit_summary = getattr(result, "delayed_credit_summary", None)
            if isinstance(result_delayed_credit_summary, DelayedCreditSummary):
                delayed_credit_summary = result_delayed_credit_summary
        promoted_case_count = sum(1 for delta in deltas if delta.target_slot == "case_memory" and not delta.blocked)
        playbook_delta_count = sum(1 for delta in deltas if delta.target_slot == "strategy_playbook")
        boundary_delta_count = sum(1 for delta in deltas if delta.target_slot == "boundary_policy")
        return self.publish(
            ExperienceConsolidationSnapshot(
                source_session_post_job_id=source_job_id,
                promoted_case_count=promoted_case_count,
                playbook_delta_count=playbook_delta_count,
                boundary_delta_count=boundary_delta_count,
                deltas=tuple(deltas[-6:]),
                description=(
                    f"Experience consolidation published {len(deltas[-6:])} recent deltas, "
                    f"{len(delayed_outcome_ledger[-6:])} delayed outcomes, and "
                    f"{len(sequence_payoffs[-4:])} sequence payoff summaries from "
                    f"{len(completed_results)} completed slow-loop result(s)."
                ),
                delayed_outcome_ledger=tuple(delayed_outcome_ledger[-6:]),
                sequence_payoffs=tuple(sequence_payoffs[-4:]),
                latest_prior_update=latest_prior_update,
                latest_writeback_report=latest_writeback_report,
                delayed_credit_summary=delayed_credit_summary,
                continuum_profile_id=continuum_profile_id,
                active_band_ids=_dedupe(tuple(active_band_ids)),
            )
        )

    async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[ExperienceConsolidationSnapshot]:
        raise NotImplementedError("ExperienceConsolidationModule is published via process_standalone().")

    async def process_standalone(
        self,
        *,
        completed_results: tuple[Any, ...] = (),
        **kwargs: Any,
    ) -> Snapshot[ExperienceConsolidationSnapshot]:
        del kwargs
        return self.publish_snapshot(completed_results=completed_results)
