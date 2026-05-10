"""Compile reviewed growth-advisor profiles into canonical lifeform inputs.

The three public functions in this module are pure (no I/O, no global
state). They take a ``GrowthAdvisorProfile`` and return canonical
runtime inputs:

* :func:`build_growth_advisor_package` -> :class:`DomainExperiencePackage`
  for the ``vz-application`` knowledge / case / playbook / boundary
  owners.
* :func:`build_growth_advisor_vitals_bootstrap` ->
  :class:`VitalsBootstrap` for ``lifeform-core`` drive priors.
* :func:`build_growth_advisor_ingestion_envelope` ->
  :class:`IngestionEnvelope` for replaying livestream-channel excerpts
  / 7-day dialogue samples through the canonical R6 ingestion path.

Mirrors :mod:`lifeform_domain_character.compiler` in shape; keeps a
distinct namespace so the two verticals never collide.
"""

from __future__ import annotations

from lifeform_core import DriveSpec, VitalsBootstrap
from lifeform_ingestion import (
    IngestionComplianceProfile,
    IngestionEnvelope,
    IngestionSourceKind,
)
from lifeform_ingestion.sources.plain_text import envelope_from_text
from volvence_zero.application import (
    BoundaryPriorHint,
    CaseMemoryRecord,
    DomainExperienceManifest,
    DomainExperiencePackage,
    DomainKnowledgeRecord,
    PlaybookRule,
)

from lifeform_domain_growth_advisor.profile import (
    GrowthAdvisorBoundaryPrior,
    GrowthAdvisorKnowledgeSeed,
    GrowthAdvisorProfile,
    GrowthAdvisorSignatureCase,
    GrowthAdvisorStrategyPrior,
)


_PACKAGE_PREFIX = "lifeform-growth-advisor"


def build_growth_advisor_package(
    profile: GrowthAdvisorProfile,
) -> DomainExperiencePackage:
    """Compile a reviewed growth-advisor profile into a DomainExperiencePackage.

    The package is data only. It enters the kernel through the existing
    ``domain_knowledge``, ``case_memory``, ``strategy_playbook``, and
    ``boundary_policy`` owners; no growth-advisor-specific kernel owner is
    introduced.
    """

    return DomainExperiencePackage(
        manifest=DomainExperienceManifest(
            package_id=f"{_PACKAGE_PREFIX}:{profile.profile_id}",
            version=profile.version,
            display_name=f"{profile.advisor_name} growth-advisor bootstrap",
            domain_ids=_domain_ids(profile),
            target_contexts=profile.target_contexts,
            evidence_level="reviewed-seed",
            owner="lifeform-domain-growth-advisor",
            description=(
                f"Reviewed growth-advisor bootstrap for {profile.advisor_name} "
                f"from {profile.source_title}. {profile.description}"
            ),
        ),
        knowledge_records=tuple(
            _knowledge_record(profile, seed) for seed in profile.knowledge_seeds
        ),
        case_records=tuple(_case_record(case) for case in profile.signature_cases),
        playbook_rules=tuple(
            _playbook_rule(rule) for rule in profile.strategy_priors
        ),
        boundary_hints=tuple(
            _boundary_hint(boundary) for boundary in profile.boundary_priors
        ),
    )


def build_growth_advisor_vitals_bootstrap(
    profile: GrowthAdvisorProfile,
    *,
    proactive_pe_threshold: float = 1.0,
    proactive_followup_priority: float = 0.55,
    proactive_cooldown_ticks: int = 60,
) -> VitalsBootstrap:
    """Compile growth-advisor drive priors into a VitalsBootstrap."""

    return VitalsBootstrap(
        schema_version=1,
        drives=tuple(
            DriveSpec(
                name=drive.name,
                target=drive.target,
                homeostatic_band=drive.homeostatic_band,
                decay_per_tick=drive.decay_per_tick,
                pe_weight=drive.pe_weight,
                initial_level=drive.initial_level,
                recharge_per_turn=drive.recharge_per_turn,
                recharge_per_regime=dict(drive.recharge_per_regime),
            )
            for drive in profile.drive_priors
        ),
        proactive_pe_threshold=proactive_pe_threshold,
        proactive_followup_priority=proactive_followup_priority,
        proactive_cooldown_ticks=proactive_cooldown_ticks,
    )


def build_growth_advisor_ingestion_envelope(
    profile: GrowthAdvisorProfile,
    sample_text: str,
    *,
    uploader: str,
    upload_ts_ms: int | None = None,
    max_chunk_chars: int = 1024,
    compliance_profile: IngestionComplianceProfile = (
        IngestionComplianceProfile.FORCED
    ),
) -> IngestionEnvelope:
    """Build an ingestion envelope for the sample-excerpt corpus.

    The growth-advisor vertical uses ``IngestionSourceKind.CORPUS`` (not
    ``BOOK`` like the character vertical) because the source material is
    a corpus of livestream-channel introduction text + structured 7-day
    dialogue samples, not a single literary work.
    """

    return envelope_from_text(
        sample_text,
        source_uri=profile.source_uri,
        uploader=uploader,
        upload_ts_ms=upload_ts_ms,
        envelope_id=f"growth-advisor-ingestion:{profile.profile_id}",
        source_kind=IngestionSourceKind.CORPUS,
        compliance_profile=compliance_profile,
        max_chunk_chars=max_chunk_chars,
    )


def _domain_ids(profile: GrowthAdvisorProfile) -> tuple[str, ...]:
    ordered: list[str] = []
    for domain in tuple(seed.domain for seed in profile.knowledge_seeds) + tuple(
        case.domain for case in profile.signature_cases
    ):
        if domain not in ordered:
            ordered.append(domain)
    if not ordered:
        ordered.append(f"growth_advisor:{profile.profile_id}")
    return tuple(ordered)


def _knowledge_record(
    profile: GrowthAdvisorProfile,
    seed: GrowthAdvisorKnowledgeSeed,
) -> DomainKnowledgeRecord:
    return DomainKnowledgeRecord(
        record_id=f"rid-growth-advisor:{profile.profile_id}:knowledge:{seed.seed_id}",
        domain=seed.domain,
        topic_tags=seed.topic_tags,
        jurisdiction_tags=("private-domain-companion",),
        source_type=seed.source_type,
        title=seed.title,
        locator=seed.evidence_locator,
        summary=seed.summary,
        snippet=seed.snippet,
        freshness_label=seed.freshness_label,
        confidence=seed.confidence,
        evidence_strength=seed.evidence_strength,
        url=profile.source_uri,
    )


def _case_record(case: GrowthAdvisorSignatureCase) -> CaseMemoryRecord:
    return CaseMemoryRecord(
        case_id=f"rid-growth-advisor:case:{case.case_id}",
        domain=case.domain,
        problem_pattern=case.problem_pattern,
        user_state_pattern=case.user_state_pattern,
        risk_markers=case.risk_markers,
        track_tags=case.track_tags,
        regime_tags=case.regime_tags,
        intervention_ordering=case.intervention_ordering,
        outcome_label=case.outcome_label,
        delayed_signal_count=1,
        escalation_observed=case.escalation_observed,
        repair_observed=case.repair_observed,
        confidence=case.confidence,
        relevance_score=case.relevance_score,
        description=case.description,
        reconstruction_source="reviewed-growth-advisor-profile",
    )


def _playbook_rule(rule: GrowthAdvisorStrategyPrior) -> PlaybookRule:
    return PlaybookRule(
        rule_id=f"rid-growth-advisor:playbook:{rule.rule_id}",
        problem_pattern=rule.problem_pattern,
        recommended_regime=rule.recommended_regime,
        recommended_ordering=rule.recommended_ordering,
        recommended_pacing=rule.recommended_pacing,
        avoid_patterns=rule.avoid_patterns,
        knowledge_weight_hint=rule.knowledge_weight_hint,
        experience_weight_hint=rule.experience_weight_hint,
        applicability_scope=rule.applicability_scope,
        confidence=rule.confidence,
        description=rule.description,
    )


def _boundary_hint(boundary: GrowthAdvisorBoundaryPrior) -> BoundaryPriorHint:
    return BoundaryPriorHint(
        hint_id=f"rid-growth-advisor:boundary:{boundary.boundary_id}",
        regime_id=boundary.regime_id,
        trigger_reasons=boundary.trigger_reasons,
        answer_depth_limit_hint=boundary.answer_depth_limit_hint,
        clarification_required=boundary.clarification_required,
        refer_out_required=boundary.refer_out_required,
        blocked_topics=boundary.blocked_topics,
        required_disclaimers=boundary.required_disclaimers,
        confidence=boundary.confidence,
        description=boundary.description,
    )


__all__ = [
    "build_growth_advisor_ingestion_envelope",
    "build_growth_advisor_package",
    "build_growth_advisor_vitals_bootstrap",
]
