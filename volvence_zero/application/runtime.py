from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping

from volvence_zero.dual_track import DualTrackSnapshot
from volvence_zero.memory import MemoryEntry, MemorySnapshot
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel

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
class DomainKnowledgeSnapshot:
    retrieval_policy_id: str
    active_domains: tuple[str, ...]
    hits: tuple[KnowledgeHit, ...]
    citation_required: bool
    jurisdiction_required: bool
    unresolved_conflicts: tuple[str, ...]
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


@dataclass(frozen=True)
class CaseMemorySnapshot:
    retrieval_policy_id: str
    hits: tuple[CaseEpisodeHit, ...]
    active_problem_patterns: tuple[str, ...]
    active_risk_markers: tuple[str, ...]
    description: str


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


@dataclass(frozen=True)
class StrategyPlaybookSnapshot:
    matched_problem_patterns: tuple[str, ...]
    matched_rules: tuple[PlaybookRule, ...]
    description: str


@dataclass(frozen=True)
class ExperienceDelta:
    delta_id: str
    delta_type: str
    target_slot: str
    summary: str
    confidence: float
    blocked: bool
    description: str


@dataclass(frozen=True)
class ExperienceConsolidationSnapshot:
    source_session_post_job_id: str
    promoted_case_count: int
    playbook_delta_count: int
    boundary_delta_count: int
    deltas: tuple[ExperienceDelta, ...]
    description: str


@dataclass(frozen=True)
class ApplicationCaseCluster:
    cluster_id: str
    problem_pattern: str
    exemplar_count: int
    mean_relevance: float
    risk_markers: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class ApplicationRareHeavyCheckpoint:
    checkpoint_id: str
    domain_template_biases: tuple[tuple[str, float], ...]
    case_clusters: tuple[ApplicationCaseCluster, ...]
    distilled_playbook_rules: tuple[PlaybookRule, ...]
    description: str


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


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


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
    severity = max(cross_track_tension, switch_gate)
    if regime_id == "repair_and_deescalation":
        severity = max(severity, 0.75)
    if severity >= 0.9:
        return RiskBand.CRITICAL
    if severity >= 0.7:
        return RiskBand.HIGH
    if severity >= 0.4:
        return RiskBand.MEDIUM
    return RiskBand.LOW


def _knowledge_domains(
    *,
    text: str,
    regime_id: str | None,
    world_weight: float,
    self_weight: float,
) -> tuple[str, ...]:
    domains: list[str] = []
    if _contains_any(
        text,
        (
            "离婚",
            "婚姻",
            "抚养",
            "custody",
            "divorce",
            "marriage",
            "law",
            "legal",
            "court",
        ),
    ):
        domains.append("family_transition")
        domains.append("professional_process")
    if _contains_any(
        text,
        (
            "工作",
            "职业",
            "offer",
            "career",
            "job",
            "辞职",
            "转行",
        ),
    ):
        domains.append("career_decision")
    if regime_id == "problem_solving" or world_weight >= 0.65:
        domains.append("structured_decision_support")
    if regime_id in {"emotional_support", "repair_and_deescalation"} or self_weight >= 0.6:
        domains.append("emotional_support_basics")
    if regime_id == "repair_and_deescalation":
        domains.append("relational_repair")
    if not domains:
        domains.append("general_support_guidance")
    return _dedupe(tuple(domains))


def _experience_domains(*, regime_id: str | None, self_weight: float, world_weight: float) -> tuple[str, ...]:
    domains: list[str] = []
    if regime_id == "problem_solving" or world_weight >= 0.65:
        domains.append("structured_decision_patterns")
    if regime_id == "repair_and_deescalation":
        domains.append("repair_patterns")
    if regime_id == "emotional_support" or self_weight >= 0.6:
        domains.append("stabilization_patterns")
    if not domains:
        domains.append("general_guidance_patterns")
    return _dedupe(tuple(domains))


def _knowledge_weight(*, regime_id: str | None, world_weight: float, self_weight: float) -> float:
    base = 0.5 + (world_weight - self_weight) * 0.35
    if regime_id == "problem_solving":
        base += 0.18
    elif regime_id == "guided_exploration":
        base += 0.05
    elif regime_id == "emotional_support":
        base -= 0.18
    elif regime_id == "repair_and_deescalation":
        base -= 0.22
    return _clamp(base)


def _retrieval_depth(
    *,
    regime_id: str | None,
    knowledge_weight: float,
    temporal_snapshot: "TemporalAbstractionSnapshot | None",
) -> KnowledgeDepth:
    switch_gate = temporal_snapshot.controller_state.switch_gate if temporal_snapshot is not None else 0.0
    if regime_id == "problem_solving" and (knowledge_weight >= 0.72 or switch_gate >= 0.55):
        return KnowledgeDepth.DEEP
    if knowledge_weight >= 0.5 or switch_gate >= 0.35:
        return KnowledgeDepth.MEDIUM
    return KnowledgeDepth.LIGHT


def _requires_citation(knowledge_domains: tuple[str, ...]) -> bool:
    return any(domain in {"family_transition", "professional_process", "career_decision"} for domain in knowledge_domains)


def _requires_jurisdiction(knowledge_domains: tuple[str, ...]) -> bool:
    return any(domain in {"family_transition", "professional_process"} for domain in knowledge_domains)


def _has_jurisdiction_context(text: str) -> bool:
    return _contains_any(
        text,
        (
            "shanghai",
            "beijing",
            "shenzhen",
            "guangzhou",
            "中国",
            "大陆",
            "本地",
            "当地",
            "jurisdiction",
            "local law",
        ),
    )


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
    lowered = entry.content.lower()
    if _contains_any(lowered, ("离婚", "divorce", "婚姻", "custody", "抚养")):
        return "family-transition-high-emotion"
    if _contains_any(lowered, ("timeline", "步骤", "plan", "project", "organize")):
        return "structured-decision-overwhelm"
    if _contains_any(lowered, ("repair", "冲突", "de-escalate", "trust", "修复")):
        return "relational-repair"
    return "general-guidance"


def _entry_user_state_pattern(entry: MemoryEntry) -> str:
    lowered = entry.content.lower()
    if _contains_any(lowered, ("overwhelmed", "乱", "怕", "anxious", "焦虑", "情绪")):
        return "high-emotional-load"
    if _contains_any(lowered, ("plan", "步骤", "clarify", "结构", "整理")):
        return "needs-structure"
    return "mixed-signal"


def _entry_risk_markers(
    *,
    entry: MemoryEntry,
    prediction_error: "PredictionErrorSnapshot | None",
    retrieval_policy: RetrievalPolicySnapshot,
) -> tuple[str, ...]:
    markers: list[str] = []
    if retrieval_policy.risk_band in {RiskBand.HIGH, RiskBand.CRITICAL}:
        markers.append(f"risk-{retrieval_policy.risk_band.value}")
    lowered = entry.content.lower()
    if _contains_any(lowered, ("孩子", "child", "custody", "抚养")):
        markers.append("child-impact")
    if _contains_any(lowered, ("law", "legal", "本地", "当地", "jurisdiction")):
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
    outcome_label = (
        ExperienceOutcomeLabel.IMPROVED if reward >= 0.2 else
        ExperienceOutcomeLabel.WORSENED if reward <= -0.2 else
        ExperienceOutcomeLabel.STABLE
    )
    escalation_observed = prediction_error.error.relationship_error <= -0.45
    repair_observed = prediction_error.error.relationship_error >= 0.15
    return CaseOutcomeSummary(
        outcome_label=outcome_label,
        delayed_signal_count=4,
        escalation_observed=escalation_observed,
        repair_observed=repair_observed,
        confidence=_clamp(0.55 + abs(reward) * 0.25),
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


class ApplicationRareHeavyState:
    def __init__(self) -> None:
        self._domain_template_biases: dict[str, float] = {}
        self._case_clusters: tuple[ApplicationCaseCluster, ...] = ()
        self._distilled_playbook_rules: tuple[PlaybookRule, ...] = ()

    @property
    def domain_template_biases(self) -> tuple[tuple[str, float], ...]:
        return tuple(sorted(self._domain_template_biases.items()))

    @property
    def case_clusters(self) -> tuple[ApplicationCaseCluster, ...]:
        return self._case_clusters

    @property
    def distilled_playbook_rules(self) -> tuple[PlaybookRule, ...]:
        return self._distilled_playbook_rules

    def export_rare_heavy_state(self, *, checkpoint_id: str) -> ApplicationRareHeavyCheckpoint:
        return ApplicationRareHeavyCheckpoint(
            checkpoint_id=checkpoint_id,
            domain_template_biases=self.domain_template_biases,
            case_clusters=self._case_clusters,
            distilled_playbook_rules=self._distilled_playbook_rules,
            description=(
                f"Application rare-heavy checkpoint with {len(self._domain_template_biases)} domain biases, "
                f"{len(self._case_clusters)} case clusters, and {len(self._distilled_playbook_rules)} playbook rules."
            ),
        )

    def import_rare_heavy_state(self, checkpoint: ApplicationRareHeavyCheckpoint) -> tuple[str, ...]:
        self._domain_template_biases = dict(checkpoint.domain_template_biases)
        self._case_clusters = checkpoint.case_clusters
        self._distilled_playbook_rules = checkpoint.distilled_playbook_rules
        return (
            "rare-heavy:application-domain-refresh",
            "rare-heavy:application-case-clusters-import",
            "rare-heavy:application-playbook-import",
        )

    def restore_rare_heavy_state(self, checkpoint: ApplicationRareHeavyCheckpoint) -> tuple[str, ...]:
        self._domain_template_biases = dict(checkpoint.domain_template_biases)
        self._case_clusters = checkpoint.case_clusters
        self._distilled_playbook_rules = checkpoint.distilled_playbook_rules
        return (
            "rare-heavy:application-domain-rollback",
            "rare-heavy:application-case-clusters-rollback",
            "rare-heavy:application-playbook-rollback",
        )


class RetrievalPolicyModule(RuntimeModule[RetrievalPolicySnapshot]):
    slot_name = "retrieval_policy"
    owner = "RetrievalPolicyModule"
    value_type = RetrievalPolicySnapshot
    dependencies = ("world_temporal", "self_temporal", "dual_track", "regime")
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
        world_weight, self_weight = _infer_track_weights(dual_track_snapshot)
        regime_id = regime_snapshot.active_regime.regime_id
        knowledge_domains = _knowledge_domains(
            text=(
                " ".join(dual_track_snapshot.world_track.active_goals)
                + " "
                + " ".join(dual_track_snapshot.self_track.active_goals)
            ),
            regime_id=regime_id,
            world_weight=world_weight,
            self_weight=self_weight,
        )
        experience_domains = _experience_domains(
            regime_id=regime_id,
            self_weight=self_weight,
            world_weight=world_weight,
        )
        if self._rare_heavy_state is not None and self._rare_heavy_state.domain_template_biases:
            boosted = sorted(
                self._rare_heavy_state.domain_template_biases,
                key=lambda item: -item[1],
            )
            for domain, weight in boosted[:2]:
                if weight >= 0.55 and domain not in knowledge_domains:
                    knowledge_domains = knowledge_domains + (domain,)
        knowledge_weight = _knowledge_weight(
            regime_id=regime_id,
            world_weight=world_weight,
            self_weight=self_weight,
        )
        retrieval_depth = _retrieval_depth(
            regime_id=regime_id,
            knowledge_weight=knowledge_weight,
            temporal_snapshot=world_temporal_snapshot if world_weight >= self_weight else self_temporal_snapshot,
        )
        citation_required = _requires_citation(knowledge_domains)
        jurisdiction_required = _requires_jurisdiction(knowledge_domains)
        risk_band = _risk_band_from_state(
            dual_track_snapshot=dual_track_snapshot,
            temporal_snapshot=world_temporal_snapshot if world_weight >= self_weight else self_temporal_snapshot,
            regime_snapshot=regime_snapshot,
        )
        abstract_action = (
            world_temporal_snapshot.active_abstract_action
            if world_weight >= self_weight
            else self_temporal_snapshot.active_abstract_action
        )
        experience_weight = _clamp(1.0 - knowledge_weight)
        intent_description = (
            f"retrieval policy regime={regime_id} abstract_action={abstract_action} "
            f"knowledge_weight={knowledge_weight:.2f} experience_weight={experience_weight:.2f} "
            f"world_weight={world_weight:.2f} self_weight={self_weight:.2f}."
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
                description=(
                    f"Retrieval policy selected {len(knowledge_domains)} knowledge domains and "
                    f"{len(experience_domains)} experience domains for regime={regime_id}."
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
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._rare_heavy_state = rare_heavy_state

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
                freshness_label="phase1-mock-current",
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
                        locator="phase1-placeholder",
                        snippet=_domain_summary(domain, regime_id=retrieval_policy.regime_id),
                        url=None,
                    ),
                ),
                description=(
                    f"Knowledge hit for {domain} aligned to regime={retrieval_policy.regime_id} "
                    f"and world_weight={retrieval_policy.world_weight:.2f}."
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
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._rare_heavy_state = rare_heavy_state

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
        entries = _case_entries(memory_value, retrieval_policy=retrieval_policy)
        hits: list[CaseEpisodeHit] = []
        active_problem_patterns: list[str] = []
        active_risk_markers: list[str] = []
        retrieval_policy_id = f"policy:{hash(retrieval_policy.intent_description) & 0xFFFF:04x}"
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
                )
            )
        if self._rare_heavy_state is not None and self._rare_heavy_state.case_clusters:
            existing_patterns = set(active_problem_patterns)
            for index, cluster in enumerate(self._rare_heavy_state.case_clusters[:2], start=1):
                if cluster.problem_pattern in existing_patterns:
                    active_risk_markers.extend(cluster.risk_markers)
                    continue
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
                description=(
                    f"Case memory produced {len(hits)} compact case hits for "
                    f"{len(retrieval_policy.experience_domains)} experience domains."
                ),
            )
        )

    async def process_standalone(self, **kwargs: Any) -> Snapshot[CaseMemorySnapshot]:
        raise NotImplementedError("CaseMemoryModule should be driven by RetrievalPolicySnapshot.")


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
        for index, pattern in enumerate(case_memory_snapshot.active_problem_patterns, start=1):
            ordering, pacing, avoid_patterns = _playbook_template(
                problem_pattern=pattern,
                regime_id=regime_snapshot.active_regime.regime_id,
                world_weight=world_weight,
                self_weight=self_weight,
            )
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
                    confidence=_clamp(0.55 + len(case_memory_snapshot.hits) * 0.08),
                    description=(
                        f"Playbook rule for pattern={pattern} aligned to regime={regime_snapshot.active_regime.regime_id} "
                        f"and cross_track_tension={dual_track_snapshot.cross_track_tension:.2f}."
                    ),
                )
            )
        if self._rare_heavy_state is not None and self._rare_heavy_state.distilled_playbook_rules:
            for distilled_rule in self._rare_heavy_state.distilled_playbook_rules:
                if distilled_rule.problem_pattern in case_memory_snapshot.active_problem_patterns:
                    rules.append(distilled_rule)
        return self.publish(
            StrategyPlaybookSnapshot(
                matched_problem_patterns=case_memory_snapshot.active_problem_patterns,
                matched_rules=tuple(rules),
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
        effective_risk_band = retrieval_policy.risk_band
        if prediction_error.error.magnitude >= 2.5 or prediction_error.error.relationship_error <= -0.55:
            effective_risk_band = RiskBand.CRITICAL
            trigger_reasons.append("prediction-error-critical")
        elif prediction_error.error.magnitude >= 1.4:
            effective_risk_band = RiskBand.HIGH
            trigger_reasons.append("prediction-error-high")
        clarification_required = bool(
            retrieval_policy.jurisdiction_required and domain_knowledge.unresolved_conflicts
        )
        if clarification_required:
            trigger_reasons.append("jurisdiction-clarification-required")
        refer_out_required = effective_risk_band is RiskBand.CRITICAL
        if refer_out_required:
            trigger_reasons.append("refer-out-required")
        citation_required = retrieval_policy.citation_required
        if citation_required:
            trigger_reasons.append("citation-required")
        if citation_required or clarification_required:
            answer_depth_limit = "high-level-only"
        elif retrieval_policy.regime_id in {"emotional_support", "repair_and_deescalation"}:
            answer_depth_limit = "support-first"
        else:
            answer_depth_limit = "standard"
        professional_scope = (
            ProfessionalScope.PROFESSIONAL_HANDOFF
            if refer_out_required
            else ProfessionalScope.DOMAIN_INFORMATION
            if citation_required
            else ProfessionalScope.GENERAL_SUPPORT
        )
        required_disclaimers: list[str] = []
        if retrieval_policy.jurisdiction_required:
            required_disclaimers.append("jurisdiction-variance")
        if clarification_required:
            required_disclaimers.append("clarify-before-concluding")
        if refer_out_required:
            required_disclaimers.append("professional-handoff")
        decision = BoundaryDecision(
            decision_id=f"boundary:{regime_snapshot.active_regime.regime_id}:{answer_depth_limit}",
            risk_band=effective_risk_band,
            professional_scope=professional_scope,
            answer_depth_limit=answer_depth_limit,
            citation_required=citation_required,
            clarification_required=clarification_required,
            refer_out_required=refer_out_required,
            blocked_topics=("definitive-domain-conclusion",) if citation_required else (),
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
        source_job_id = "experience-consolidation:none"
        for result in completed_results[-4:]:
            result_deltas = getattr(result, "experience_deltas", ())
            if getattr(result, "job_id", None) is not None:
                source_job_id = result.job_id
            deltas.extend(result_deltas)
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
                    f"Experience consolidation published {len(deltas[-6:])} recent deltas from "
                    f"{len(completed_results)} completed slow-loop result(s)."
                ),
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
