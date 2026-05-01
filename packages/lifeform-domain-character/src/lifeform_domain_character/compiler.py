"""Compile reviewed character profiles into canonical lifeform inputs."""

from __future__ import annotations

from lifeform_core import DriveSpec, VitalsBootstrap
from lifeform_ingestion import IngestionComplianceProfile, IngestionEnvelope, IngestionSourceKind
from lifeform_ingestion.sources.plain_text import envelope_from_text
from volvence_zero.application import (
    BoundaryPriorHint,
    CaseMemoryRecord,
    DomainExperienceManifest,
    DomainExperiencePackage,
    DomainKnowledgeRecord,
    PlaybookRule,
)

from lifeform_domain_character.profile import (
    CharacterBoundaryPrior,
    CharacterKnowledgeSeed,
    CharacterSignatureCase,
    CharacterSoulProfile,
    CharacterStrategyPrior,
)


_PACKAGE_PREFIX = "lifeform-character"


def build_character_package(profile: CharacterSoulProfile) -> DomainExperiencePackage:
    """Compile a reviewed character profile into a DomainExperiencePackage.

    The package is data only. It enters the kernel through the existing
    ``domain_knowledge``, ``case_memory``, ``strategy_playbook``, and
    ``boundary_policy`` owners; no character-specific owner is introduced.
    """

    return DomainExperiencePackage(
        manifest=DomainExperienceManifest(
            package_id=f"{_PACKAGE_PREFIX}:{profile.profile_id}",
            version=profile.version,
            display_name=f"{profile.character_name} character bootstrap",
            domain_ids=_domain_ids(profile),
            target_contexts=profile.target_contexts,
            evidence_level="reviewed-seed",
            owner="lifeform-domain-character",
            description=(
                f"Reviewed fictional-character bootstrap for {profile.character_name} "
                f"from {profile.source_title}. {profile.description}"
            ),
        ),
        knowledge_records=tuple(_knowledge_record(profile, seed) for seed in profile.knowledge_seeds),
        case_records=tuple(_case_record(case) for case in profile.signature_cases),
        playbook_rules=tuple(_playbook_rule(rule) for rule in profile.strategy_priors),
        boundary_hints=tuple(_boundary_hint(boundary) for boundary in profile.boundary_priors),
    )


def build_character_vitals_bootstrap(
    profile: CharacterSoulProfile,
    *,
    proactive_pe_threshold: float = 1.0,
    proactive_followup_priority: float = 0.55,
    proactive_cooldown_ticks: int = 60,
) -> VitalsBootstrap:
    """Compile character drive priors into a VitalsBootstrap."""

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


def build_character_ingestion_envelope(
    profile: CharacterSoulProfile,
    novel_text: str,
    *,
    uploader: str,
    upload_ts_ms: int | None = None,
    max_chunk_chars: int = 2048,
    compliance_profile: IngestionComplianceProfile = IngestionComplianceProfile.FORCED,
) -> IngestionEnvelope:
    """Build a book ingestion envelope for the source text behind a profile."""

    return envelope_from_text(
        novel_text,
        source_uri=profile.source_uri,
        uploader=uploader,
        upload_ts_ms=upload_ts_ms,
        envelope_id=f"character-ingestion:{profile.profile_id}",
        source_kind=IngestionSourceKind.BOOK,
        compliance_profile=compliance_profile,
        max_chunk_chars=max_chunk_chars,
    )


def _domain_ids(profile: CharacterSoulProfile) -> tuple[str, ...]:
    ordered: list[str] = []
    for domain in tuple(seed.domain for seed in profile.knowledge_seeds) + tuple(
        case.domain for case in profile.signature_cases
    ):
        if domain not in ordered:
            ordered.append(domain)
    if not ordered:
        ordered.append(f"character:{profile.profile_id}")
    return tuple(ordered)


def _knowledge_record(profile: CharacterSoulProfile, seed: CharacterKnowledgeSeed) -> DomainKnowledgeRecord:
    return DomainKnowledgeRecord(
        record_id=f"rid-character:{profile.profile_id}:knowledge:{seed.seed_id}",
        domain=seed.domain,
        topic_tags=seed.topic_tags,
        jurisdiction_tags=("fictional-character",),
        source_type="internal-guide",
        title=seed.title,
        locator=seed.evidence_locator,
        summary=seed.summary,
        snippet=seed.snippet,
        freshness_label="reviewed",
        confidence=seed.confidence,
        evidence_strength=seed.evidence_strength,
        url=profile.source_uri,
    )


def _case_record(case: CharacterSignatureCase) -> CaseMemoryRecord:
    return CaseMemoryRecord(
        case_id=f"rid-character:case:{case.case_id}",
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
        reconstruction_source="reviewed-character-profile",
    )


def _playbook_rule(rule: CharacterStrategyPrior) -> PlaybookRule:
    return PlaybookRule(
        rule_id=f"rid-character:playbook:{rule.rule_id}",
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


def _boundary_hint(boundary: CharacterBoundaryPrior) -> BoundaryPriorHint:
    return BoundaryPriorHint(
        hint_id=f"rid-character:boundary:{boundary.boundary_id}",
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
    "build_character_ingestion_envelope",
    "build_character_package",
    "build_character_vitals_bootstrap",
]
