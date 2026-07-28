from __future__ import annotations

from dataclasses import dataclass

from volvence_zero.application.runtime import (
    ApplicationModificationEvidence,
    ApplicationPriorUpdate,
    BoundaryPolicyPriorUpdate,
    BoundaryPriorHint,
    CaseMemoryPriorUpdate,
    ConversationKnowledgeCandidate,
    DomainKnowledgePriorUpdate,
    ExperiencedActionEvidence,
    KnowledgeHit,
    KnowledgeReviewStatus,
    KnowledgeSourceKind,
    PlaybookRule,
    RetrievalReadoutPriorUpdate,
    StrategyPlaybookPriorUpdate,
)
from volvence_zero.application.action_abstraction import (
    ActionAbstractionDecoder,
    ActionAbstractionExperience,
    ActionAbstractionOwner,
    merge_action_abstraction_experiences,
)
from volvence_zero.application.retrieval_readout import (
    RetrievalControlReadoutParameters,
    RetrievalReadoutCheckpoint,
)
from volvence_zero.application.storage import (
    CaseActionAbstractionEvidence,
    CaseActionAbstractionPromotion,
    CaseMemoryRecord,
    DomainKnowledgeRecord,
)


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
    conversation_knowledge_candidates: tuple[ConversationKnowledgeCandidate, ...] = ()
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
    experienced_actions: tuple[ExperiencedActionEvidence, ...] = ()
    prior_action_abstraction_evidence: tuple[
        CaseActionAbstractionEvidence, ...
    ] = ()
    promoted_action_abstraction_family_versions: tuple[
        tuple[str, int], ...
    ] = ()
    action_abstraction_decoder: ActionAbstractionDecoder | None = None


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
        promoted_action_family_ids = {
            family_id
            for family_id, _family_version in (
                inputs.promoted_action_abstraction_family_versions
            )
        }
        current_action_abstraction_experiences = tuple(
            ActionAbstractionExperience(
                outcome_id=evidence.outcome_id,
                action_id=evidence.action_id,
                action_family_id=evidence.action_family_id,
                action_family_version=evidence.action_family_version,
                situation_statement=evidence.situation_statement,
                action_statement=evidence.action_statement,
                evidence=evidence.evidence,
                confidence=evidence.confidence,
                controller_code_digest=evidence.controller_code_digest,
            )
            for evidence in inputs.experienced_actions
            if (
                evidence.action_schema is None
                and evidence.action_family_id
                and evidence.action_family_version > 0
                and evidence.situation_statement.strip()
                and evidence.action_family_id not in promoted_action_family_ids
            )
        )
        prior_action_abstraction_experiences = tuple(
            ActionAbstractionExperience(
                outcome_id=evidence.outcome_id,
                action_id=evidence.action_id,
                action_family_id=evidence.action_family_id,
                action_family_version=evidence.action_family_version,
                situation_statement=evidence.situation_statement,
                action_statement=evidence.action_statement,
                evidence=evidence.evidence,
                confidence=evidence.confidence,
                controller_code_digest=evidence.controller_code_digest,
            )
            for evidence in inputs.prior_action_abstraction_evidence
            if evidence.action_family_id not in promoted_action_family_ids
        )
        action_abstraction_experiences = merge_action_abstraction_experiences(
            prior_action_abstraction_experiences,
            current_action_abstraction_experiences,
        )
        action_abstraction_groups: dict[
            str,
            list[ActionAbstractionExperience],
        ] = {}
        for experience in action_abstraction_experiences:
            action_abstraction_groups.setdefault(
                experience.action_family_id,
                [],
            ).append(experience)
        for evidence in inputs.experienced_actions:
            action_schema = evidence.action_schema
            intervention_ordering = (
                action_schema.action_steps
                if action_schema is not None
                else ()
            )
            problem_pattern = (
                " ".join(
                    (
                        f"lived-action-schema:{action_schema.schema_id}",
                        *action_schema.applicability_conditions,
                    )
                )
                if action_schema is not None
                else (
                    f"latent-action-family:{evidence.action_family_id}"
                    if evidence.action_family_id
                    else f"unresolved-lived-action:{evidence.action_id}"
                )
            )
            confidence = _clamp(
                evidence.confidence * 0.7
                + inputs.mean_experience_quality * 0.3
            )
            relevance = _clamp(
                0.60
                + evidence.confidence * 0.25
                + inputs.mean_experience_quality * 0.15
            )
            case_updates.append(
                CaseMemoryPriorUpdate(
                    update_id=(
                        f"{inputs.job_id}:experienced-action:"
                        f"{evidence.outcome_id}"
                    ),
                    target=(
                        "application.case_memory.records."
                        f"experienced-action.{evidence.outcome_id}"
                    ),
                    record=CaseMemoryRecord(
                        case_id=(
                            f"case:slow-loop:{inputs.job_id}:"
                            f"experienced-action:{evidence.action_id}"
                        ),
                        domain=primary_domain,
                        problem_pattern=problem_pattern,
                        user_state_pattern=(
                            "reviewed-environment-outcome"
                            if action_schema is not None
                            else "schema-pending:latent-family-linked"
                            if evidence.action_family_id
                            else "schema-pending:unclassified"
                        ),
                        risk_markers=inputs.case_risk_markers,
                        track_tags=("world", "self"),
                        regime_tags=(
                            (inputs.regime_id,)
                            if inputs.regime_id is not None
                            else ()
                        ),
                        intervention_ordering=intervention_ordering,
                        outcome_label=outcome_label,
                        delayed_signal_count=1,
                        escalation_observed=False,
                        repair_observed=False,
                        confidence=confidence,
                        relevance_score=relevance,
                        description=(
                            (
                                "Reviewed reusable action conditions: "
                                + "; ".join(
                                    action_schema.applicability_conditions
                                )
                                + ". "
                            )
                            if action_schema is not None
                            else ""
                        )
                        + (
                            "Reviewed lived action: "
                            f"{evidence.action_statement} "
                            "Observed outcome: "
                            f"{evidence.outcome_statement}"
                        )
                        + (
                            (
                                " Latent family lineage: "
                                f"{evidence.action_family_id} "
                                f"version={evidence.action_family_version} "
                                "controller_code_digest="
                                f"{evidence.controller_code_digest}."
                            )
                            if evidence.action_family_id
                            else ""
                        ),
                        action_abstraction_evidence=(
                            CaseActionAbstractionEvidence(
                                outcome_id=evidence.outcome_id,
                                action_id=evidence.action_id,
                                action_family_id=evidence.action_family_id,
                                action_family_version=(
                                    evidence.action_family_version
                                ),
                                situation_statement=(
                                    evidence.situation_statement
                                ),
                                action_statement=evidence.action_statement,
                                evidence=evidence.evidence,
                                confidence=evidence.confidence,
                                controller_code_digest=(
                                    evidence.controller_code_digest
                                ),
                            )
                            if (
                                evidence.action_schema is None
                                and evidence.action_family_id
                                and evidence.action_family_version > 0
                                and evidence.situation_statement.strip()
                                and evidence.action_family_id
                                not in promoted_action_family_ids
                            )
                            else None
                        ),
                    ),
                    confidence=confidence,
                    description=(
                        "Compile canonical environment action "
                        f"{evidence.action_id} into CaseMemory."
                    ),
                )
            )
        if inputs.action_abstraction_decoder is not None:
            for family_key in sorted(action_abstraction_groups):
                family_experiences = tuple(
                    action_abstraction_groups[family_key]
                )
                candidate = ActionAbstractionOwner().propose(
                    experiences=family_experiences,
                    decoder=inputs.action_abstraction_decoder,
                )
                if candidate is None:
                    continue
                source_confidence = min(
                    evidence.confidence
                    for evidence in family_experiences
                    if evidence.outcome_id in candidate.source_outcome_ids
                )
                confidence = _clamp(
                    candidate.confidence * 0.7
                    + source_confidence * 0.3
                )
                target = (
                    "application.case_memory.records.action-abstraction."
                    f"{candidate.action_family_id}."
                    f"{candidate.action_family_version}"
                )
                case_updates.append(
                    CaseMemoryPriorUpdate(
                        update_id=(
                            f"{inputs.job_id}:action-abstraction:"
                            f"{candidate.schema_id}"
                        ),
                        target=target,
                        record=CaseMemoryRecord(
                            case_id=(
                                "case:slow-loop:action-abstraction:"
                                f"{candidate.action_family_id}:"
                                f"{candidate.action_family_version}"
                            ),
                            domain=primary_domain,
                            problem_pattern=" ".join(
                                (
                                    "learned-action-schema:"
                                    f"{candidate.schema_id}",
                                    *candidate.applicability_conditions,
                                )
                            ),
                            user_state_pattern=(
                                "background-slow:gated-action-abstraction"
                            ),
                            risk_markers=inputs.case_risk_markers,
                            track_tags=("world", "self"),
                            regime_tags=(
                                (inputs.regime_id,)
                                if inputs.regime_id is not None
                                else ()
                            ),
                            intervention_ordering=candidate.action_steps,
                            outcome_label=outcome_label,
                            delayed_signal_count=len(
                                candidate.source_outcome_ids
                            ),
                            escalation_observed=False,
                            repair_observed=False,
                            confidence=confidence,
                            relevance_score=confidence,
                            description=(
                                "Background-slow learned action abstraction "
                                f"for family={candidate.action_family_id} "
                                f"version={candidate.action_family_version}; "
                                "source_outcome_ids="
                                f"{candidate.source_outcome_ids}. "
                                f"{candidate.description}"
                            ),
                            action_abstraction_promotion=(
                                CaseActionAbstractionPromotion(
                                    schema_id=candidate.schema_id,
                                    action_family_id=(
                                        candidate.action_family_id
                                    ),
                                    action_family_version=(
                                        candidate.action_family_version
                                    ),
                                    source_outcome_ids=(
                                        candidate.source_outcome_ids
                                    ),
                                    applicability_conditions=(
                                        candidate.applicability_conditions
                                    ),
                                )
                            ),
                        ),
                        confidence=confidence,
                        description=(
                            "Promote a multi-experience latent-family "
                            "semantic abstraction into CaseMemory."
                        ),
                        modification_evidence=(
                            ApplicationModificationEvidence(
                                validation_delta=max(
                                    0.0,
                                    min(
                                        candidate.confidence,
                                        source_confidence,
                                    )
                                    - 0.70,
                                ),
                                capacity_cost=min(
                                    0.40,
                                    0.02
                                    * (
                                        len(
                                            candidate.applicability_conditions
                                        )
                                        + len(candidate.action_steps)
                                    ),
                                ),
                                rollback_evidence=(
                                    "Remove CaseMemory target "
                                    f"{target} and restore the prior "
                                    "application case-memory checkpoint."
                                ),
                            )
                        ),
                    )
                )
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
                        if _brief_is_self_track(inputs.regime_id)
                        else ("world",),
                        regime_tags=(inputs.regime_id,) if inputs.regime_id is not None else (),
                        intervention_ordering=ordering,
                        outcome_label=outcome_label,
                        delayed_signal_count=max(inputs.case_hit_count, 1),
                        escalation_observed="refer-out-required" in inputs.boundary_trigger_reasons,
                        repair_observed=_brief_is_repair(inputs.regime_id),
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
        hits_by_id = {hit.hit_id: hit for hit in inputs.knowledge_hits}
        for index, candidate in enumerate(inputs.conversation_knowledge_candidates[:1], start=1):
            if candidate.review_status is not KnowledgeReviewStatus.APPROVED:
                continue
            hit = hits_by_id.get(candidate.knowledge_hit_id)
            if hit is None:
                continue
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
                        f"using hit={hit.hit_id} candidate={candidate.candidate_id}."
                    ),
                    source_kind=KnowledgeSourceKind.CONVERSATION,
                    source_candidate_ids=(candidate.candidate_id,),
                    review_status=candidate.review_status,
                    citation_ids=tuple(candidate.citation_ids),
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
        # W4 SSOT: read structured decision_kind_hint from the
        # ApplicationBrief instead of regime-id strings.
        from volvence_zero.regime import application_brief_for_regime

        decision_kind = application_brief_for_regime(regime_id).decision_kind_hint
        if problem_pattern == "family-transition-high-emotion":
            return ("stabilize", "split_axes", "smallest_next_step")
        if problem_pattern == "relational-repair" or decision_kind == "repair-first":
            return ("acknowledge", "deescalate", "repair-next-step")
        if (
            problem_pattern == "structured-decision-overwhelm"
            or decision_kind == "structure-first"
        ):
            return ("narrow_scope", "option_compare", "smallest_next_step")
        if decision_kind == "support-first":
            return ("acknowledge", "stabilize", "smallest_next_step")
        return ("acknowledge", "smallest_next_step")

    def _application_pacing_for_regime(self, regime_id: str | None) -> str:
        from volvence_zero.regime import application_brief_for_regime

        brief = application_brief_for_regime(regime_id)
        if _brief_is_self_track_brief(brief):
            return "gradual"
        if brief.decision_kind_hint == "structure-first":
            return "structured"
        return "balanced"


def _brief_is_self_track(regime_id: str | None) -> bool:
    """Wave 4 SSOT: a regime is "self-track-leaning" when the brief
    publishes elevated support_focus or repair_focus.

    Replaces ``regime_id in {"emotional_support", "repair_and_deescalation"}``.
    Anywhere this used to be true now corresponds to either
    ``brief.support_focus >= 0.6`` (emotional_support is 0.85) or
    ``brief.repair_focus >= 0.4`` (repair_and_deescalation is 0.85).
    """

    from volvence_zero.regime import application_brief_for_regime

    return _brief_is_self_track_brief(application_brief_for_regime(regime_id))


def _brief_is_self_track_brief(brief) -> bool:
    return brief.support_focus >= 0.6 or brief.repair_focus >= 0.4


def _brief_is_repair(regime_id: str | None) -> bool:
    """Wave 4 SSOT: a regime is "repair-shaped" iff the brief
    publishes elevated repair_focus. Replaces ``regime_id ==
    "repair_and_deescalation"``.
    """

    from volvence_zero.regime import application_brief_for_regime

    return application_brief_for_regime(regime_id).repair_focus >= 0.4
