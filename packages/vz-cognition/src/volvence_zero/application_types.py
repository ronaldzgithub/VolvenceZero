"""Application snapshot type definitions \u2014 lives in vz-cognition.

This module is the **cycle break** between ``vz-cognition`` and the
forthcoming ``vz-application`` wheel. Background:

* ``vz-cognition.evaluation`` consumes a handful of frozen-dataclass types
  produced by application owners (``BoundaryPolicySnapshot`` etc.).
* The application owners themselves consume ``DualTrackSnapshot`` and
  ``MemorySnapshot`` from ``vz-cognition``.
* If we keep the type definitions inside the application owner package and
  then split that package into its own wheel, we get a wheel-level cycle:
  ``vz-cognition \u2192 vz-application \u2192 vz-cognition``.

Solution: define the types exactly **once** here, in vz-cognition. The
application owner package re-exports them (so existing consumers keep
working unchanged) but does not own them. After the eventual physical
split, ``vz-application`` imports these types from ``vz-cognition``,
and ``vz-cognition.evaluation`` does too \u2014 no cycle.

Only types that ``evaluation`` directly consumes plus their transitive
type closure are moved here. Other types that bind storage records or
retrieval-readout checkpoints stay in ``application/runtime.py`` because
those bindings cross into the future ``vz-application`` wheel anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


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


@dataclass(frozen=True)
class StrategyPlaybookSnapshot:
    matched_problem_patterns: tuple[str, ...]
    matched_rules: tuple[PlaybookRule, ...]
    description: str
    continuum_profile_id: str | None = None
    active_band_ids: tuple[str, ...] = ()


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
