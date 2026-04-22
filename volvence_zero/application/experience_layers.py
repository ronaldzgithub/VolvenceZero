from __future__ import annotations

from dataclasses import dataclass

from volvence_zero.application.runtime import (
    ApplicationPriorUpdate,
    BoundaryPolicyPriorUpdate,
    BoundaryPriorHint,
    CaseMemoryPriorUpdate,
    DomainKnowledgePriorUpdate,
    KnowledgeHit,
    PlaybookRule,
    RetrievalReadoutPriorUpdate,
    StrategyPlaybookPriorUpdate,
)
from volvence_zero.application.retrieval_readout import (
    RetrievalControlReadoutParameters,
    RetrievalReadoutCheckpoint,
)
from volvence_zero.application.storage import CaseMemoryRecord, DomainKnowledgeRecord


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class ApplicationPriorProposalInputs:
    job_id: str
    closed_at_turn: int
    regime_id: str | None
    knowledge_domains: tuple[str, ...]
    experience_domains: tuple[str, ...]
    case_problem_patterns: tuple[str, ...]
    case_risk_markers: tuple[str, ...]
    boundary_trigger_reasons: tuple[str, ...]
    knowledge_weight: float
    experience_weight: float
    case_hit_count: int
    mean_experience_quality: float
    knowledge_hits: tuple[KnowledgeHit, ...] = ()
    retrieval_readout_checkpoint: RetrievalReadoutCheckpoint | None = None
    retrieval_fast_prior_strength: float = 0.0
    retrieval_fast_prior_attribution_count: int = 0
    retrieval_fast_prior_sequence_count: int = 0
    retrieval_regime_bias: float = 0.0
    retrieval_action_bias: float = 0.0
    retrieval_family_bias: float = 0.0
    retrieval_knowledge_weight_bias: float = 0.0
    retrieval_experience_weight_bias: float = 0.0
    retrieval_source_attribution_ids: tuple[str, ...] = ()
    retrieval_source_sequence_ids: tuple[str, ...] = ()
    retrieval_mean_retrieval_mix_alignment: float = 0.0
    retrieval_mean_regime_alignment: float = 0.0
    retrieval_mean_action_alignment: float = 0.0
    retrieval_mean_sequence_payoff: float = 0.0


class ApplicationPriorProposalBuilder:
    """Owner-side helper that turns slow-loop evidence into application priors."""

    def build(self, *, inputs: ApplicationPriorProposalInputs) -> ApplicationPriorUpdate | None:
        case_updates: list[CaseMemoryPriorUpdate] = []
        playbook_updates: list[StrategyPlaybookPriorUpdate] = []
        boundary_updates: list[BoundaryPolicyPriorUpdate] = []
        knowledge_updates: list[DomainKnowledgePriorUpdate] = []
        retrieval_updates: list[RetrievalReadoutPriorUpdate] = []
        primary_domain = next(iter(inputs.experience_domains), "general_guidance_patterns")
        outcome_label = "improved" if inputs.mean_experience_quality >= 0.6 else "stable"
        for pattern in inputs.case_problem_patterns:
            ordering = self._application_ordering_for_pattern(problem_pattern=pattern, regime_id=inputs.regime_id)
            case_updates.append(
                CaseMemoryPriorUpdate(
                    update_id=f"{inputs.job_id}:case-update:{pattern}",
                    target=f"application.case_memory.records.{pattern}",
                    record=CaseMemoryRecord(
                        case_id=f"case:slow-loop:{inputs.job_id}:{pattern}",
                        domain=primary_domain,
                        problem_pattern=pattern,
                        user_state_pattern="slow-loop-promoted",
                        risk_markers=inputs.case_risk_markers,
                        track_tags=("self",)
                        if inputs.regime_id in {"emotional_support", "repair_and_deescalation"}
                        else ("world",),
                        regime_tags=(inputs.regime_id,) if inputs.regime_id is not None else (),
                        intervention_ordering=ordering,
                        outcome_label=outcome_label,
                        delayed_signal_count=max(inputs.case_hit_count, 1),
                        escalation_observed="refer-out-required" in inputs.boundary_trigger_reasons,
                        repair_observed=inputs.regime_id == "repair_and_deescalation",
                        confidence=_clamp(0.52 + inputs.mean_experience_quality * 0.36),
                        relevance_score=_clamp(0.48 + inputs.mean_experience_quality * 0.42),
                        description=(
                            f"Session-post promoted case prior for pattern={pattern} "
                            f"quality={inputs.mean_experience_quality:.2f}."
                        ),
                    ),
                    confidence=_clamp(0.52 + inputs.mean_experience_quality * 0.36),
                    description=f"Promote case prior for pattern={pattern} from session-post evidence.",
                )
            )
            playbook_updates.append(
                StrategyPlaybookPriorUpdate(
                    update_id=f"{inputs.job_id}:playbook-update:{pattern}",
                    target=f"application.strategy_playbook.rules.{pattern}",
                    rule=PlaybookRule(
                        rule_id=f"playbook:slow-loop:{pattern}:{inputs.closed_at_turn}",
                        problem_pattern=pattern,
                        recommended_regime=inputs.regime_id,
                        recommended_ordering=ordering,
                        recommended_pacing=self._application_pacing_for_regime(inputs.regime_id),
                        avoid_patterns=("procedure-dump-too-early",)
                        if "child-impact" in inputs.case_risk_markers
                        else ("over-directive-solutioning",),
                        knowledge_weight_hint=_clamp(inputs.knowledge_weight + 0.08),
                        experience_weight_hint=_clamp(inputs.experience_weight + 0.12),
                        applicability_scope=((inputs.regime_id,) if inputs.regime_id is not None else ())
                        + inputs.case_risk_markers[:2],
                        confidence=_clamp(0.5 + inputs.mean_experience_quality * 0.4),
                        description=(
                            f"Session-post promoted playbook prior for pattern={pattern} "
                            f"with regime={inputs.regime_id}."
                        ),
                    ),
                    confidence=_clamp(0.5 + inputs.mean_experience_quality * 0.4),
                    description=f"Promote playbook prior for pattern={pattern} from session-post evidence.",
                )
            )
        if inputs.boundary_trigger_reasons:
            boundary_updates.append(
                BoundaryPolicyPriorUpdate(
                    update_id=f"{inputs.job_id}:boundary-update",
                    target=(
                        f"application.boundary_policy.hints.{inputs.regime_id or 'shared'}."
                        f"{len(inputs.boundary_trigger_reasons)}"
                    ),
                    hint=BoundaryPriorHint(
                        hint_id=f"boundary-hint:{inputs.job_id}",
                        regime_id=inputs.regime_id,
                        trigger_reasons=inputs.boundary_trigger_reasons,
                        answer_depth_limit_hint=(
                            "high-level-only"
                            if "refer-out-required" in inputs.boundary_trigger_reasons
                            or "citation-required" in inputs.boundary_trigger_reasons
                            else "support-first"
                        ),
                        clarification_required="jurisdiction-clarification-required" in inputs.boundary_trigger_reasons,
                        refer_out_required="refer-out-required" in inputs.boundary_trigger_reasons,
                        blocked_topics=("definitive-domain-conclusion",)
                        if "citation-required" in inputs.boundary_trigger_reasons
                        else (),
                        required_disclaimers=(
                            ("professional-handoff",)
                            if "refer-out-required" in inputs.boundary_trigger_reasons
                            else ()
                        )
                        + (
                            ("clarify-before-concluding",)
                            if "jurisdiction-clarification-required" in inputs.boundary_trigger_reasons
                            else ()
                        ),
                        confidence=_clamp(0.5 + inputs.mean_experience_quality * 0.34),
                        description=(
                            f"Session-post boundary prior from triggers={inputs.boundary_trigger_reasons} "
                            f"quality={inputs.mean_experience_quality:.2f}."
                        ),
                    ),
                    confidence=_clamp(0.5 + inputs.mean_experience_quality * 0.34),
                    description="Promote boundary prior from repeated slow-loop boundary triggers.",
                )
            )
        for index, hit in enumerate(inputs.knowledge_hits, start=1):
            citation = hit.citations[0] if hit.citations else None
            title = citation.title if citation is not None else f"{hit.domain.replace('_', ' ')} guidance"
            locator = citation.locator if citation is not None else f"promoted:{hit.hit_id}"
            snippet = citation.snippet if citation is not None else hit.summary
            source_type = citation.source_type.value if citation is not None else "internal-guide"
            stable_id = hit.hit_id.replace(":", "-")
            knowledge_updates.append(
                DomainKnowledgePriorUpdate(
                    update_id=f"{inputs.job_id}:knowledge-update:{stable_id}:{index}",
                    target=f"application.domain_knowledge.records.{hit.domain}.{stable_id}",
                    record=DomainKnowledgeRecord(
                        record_id=f"knowledge:slow-loop:{inputs.job_id}:{stable_id}:{index}",
                        domain=hit.domain,
                        topic_tags=hit.topic_tags,
                        jurisdiction_tags=hit.jurisdiction_tags,
                        source_type=source_type,
                        title=title,
                        locator=locator,
                        summary=hit.summary,
                        snippet=snippet,
                        freshness_label="session-post-promoted",
                        confidence=_clamp(hit.confidence * 0.55 + inputs.mean_experience_quality * 0.35 + 0.10),
                        evidence_strength=hit.evidence_strength.value,
                        conflict_markers=hit.conflict_markers,
                        url=citation.url if citation is not None else None,
                    ),
                    confidence=_clamp(hit.confidence * 0.45 + inputs.mean_experience_quality * 0.40 + 0.10),
                    description=(
                        f"Promote knowledge prior for domain={hit.domain} from session-post evidence "
                        f"using hit={hit.hit_id}."
                    ),
                )
            )
        retrieval_checkpoint = self._build_retrieval_readout_checkpoint(inputs=inputs)
        if retrieval_checkpoint is not None:
            retrieval_updates.append(
                RetrievalReadoutPriorUpdate(
                    update_id=f"{inputs.job_id}:retrieval-readout-update",
                    target="application.retrieval_readout.checkpoint",
                    checkpoint=retrieval_checkpoint,
                    confidence=retrieval_checkpoint.confidence,
                    description="Promote retrieval readout checkpoint from delayed experience evidence.",
                )
            )
        if not case_updates and not playbook_updates and not boundary_updates and not knowledge_updates and not retrieval_updates:
            return None
        return ApplicationPriorUpdate(
            source_session_post_job_id=inputs.job_id,
            case_memory_updates=tuple(case_updates),
            strategy_playbook_updates=tuple(playbook_updates),
            boundary_policy_updates=tuple(boundary_updates),
            domain_knowledge_updates=tuple(knowledge_updates),
            retrieval_readout_updates=tuple(retrieval_updates),
            description=(
                f"Application prior update proposed from {inputs.job_id} with "
                f"{len(case_updates)} case updates, {len(playbook_updates)} playbook updates, "
                f"{len(boundary_updates)} boundary updates, {len(knowledge_updates)} knowledge updates, "
                f"and {len(retrieval_updates)} retrieval readout updates."
            ),
        )

    def _build_retrieval_readout_checkpoint(
        self,
        *,
        inputs: ApplicationPriorProposalInputs,
    ) -> RetrievalReadoutCheckpoint | None:
        evidence_present = (
            inputs.retrieval_fast_prior_strength > 0.0
            or inputs.retrieval_fast_prior_attribution_count > 0
            or inputs.retrieval_fast_prior_sequence_count > 0
        )
        if not evidence_present:
            return None
        base_parameters = (
            inputs.retrieval_readout_checkpoint.parameters
            if inputs.retrieval_readout_checkpoint is not None
            else RetrievalControlReadoutParameters.default()
        )
        updated_parameters = base_parameters.updated_from_slow_prior(
            strength=inputs.retrieval_fast_prior_strength,
            attribution_count=inputs.retrieval_fast_prior_attribution_count,
            sequence_count=inputs.retrieval_fast_prior_sequence_count,
            regime_bias=inputs.retrieval_regime_bias,
            action_bias=inputs.retrieval_action_bias,
            family_bias=inputs.retrieval_family_bias,
            knowledge_weight_bias=inputs.retrieval_knowledge_weight_bias,
            experience_weight_bias=inputs.retrieval_experience_weight_bias,
        )
        if updated_parameters == base_parameters:
            return None
        return RetrievalReadoutCheckpoint(
            checkpoint_id=f"retrieval-readout:{inputs.job_id}",
            parameters=updated_parameters,
            confidence=_clamp(0.5 + inputs.mean_experience_quality * 0.35),
            source_session_post_job_id=inputs.job_id,
            source_attribution_ids=inputs.retrieval_source_attribution_ids,
            source_sequence_ids=inputs.retrieval_source_sequence_ids,
            mean_retrieval_mix_alignment=inputs.retrieval_mean_retrieval_mix_alignment,
            mean_regime_alignment=inputs.retrieval_mean_regime_alignment,
            mean_action_alignment=inputs.retrieval_mean_action_alignment,
            mean_sequence_payoff=inputs.retrieval_mean_sequence_payoff,
            description=(
                f"Session-post retrieval readout checkpoint from {inputs.job_id} with "
                f"strength={inputs.retrieval_fast_prior_strength:.2f} "
                f"attr={len(inputs.retrieval_source_attribution_ids)} seq={len(inputs.retrieval_source_sequence_ids)}."
            ),
        )

    def _application_ordering_for_pattern(
        self,
        *,
        problem_pattern: str,
        regime_id: str | None,
    ) -> tuple[str, ...]:
        if problem_pattern == "family-transition-high-emotion":
            return ("stabilize", "split_axes", "smallest_next_step")
        if problem_pattern == "relational-repair" or regime_id == "repair_and_deescalation":
            return ("acknowledge", "deescalate", "repair-next-step")
        if problem_pattern == "structured-decision-overwhelm" or regime_id == "problem_solving":
            return ("narrow_scope", "option_compare", "smallest_next_step")
        if regime_id == "emotional_support":
            return ("acknowledge", "stabilize", "smallest_next_step")
        return ("acknowledge", "smallest_next_step")

    def _application_pacing_for_regime(self, regime_id: str | None) -> str:
        if regime_id in {"emotional_support", "repair_and_deescalation"}:
            return "gradual"
        if regime_id == "problem_solving":
            return "structured"
        return "balanced"
