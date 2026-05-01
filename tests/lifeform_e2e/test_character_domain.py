"""Tests for the reviewed fictional-character bootstrap vertical."""

from __future__ import annotations

import pytest


def _profile():
    from lifeform_domain_character import (
        CharacterBoundaryPrior,
        CharacterDrivePrior,
        CharacterKnowledgeSeed,
        CharacterSignatureCase,
        CharacterSoulProfile,
        CharacterStrategyPrior,
    )

    return CharacterSoulProfile(
        profile_id="test-protagonist",
        character_name="Test Protagonist",
        source_title="The Test Novel",
        version="0.1.0",
        reviewed_by="test-reviewer",
        source_uri="file:///novels/test-novel.txt",
        description="A reviewed fictional character profile used only for tests.",
        knowledge_seeds=(
            CharacterKnowledgeSeed(
                seed_id="value-protect-vulnerable",
                domain="character_values",
                title="Protect vulnerable people before optimizing outcomes",
                summary="The character repeatedly chooses protection over efficient wins.",
                snippet="Protection comes before cleverness.",
                evidence_locator="chapter=1,scene=bridge",
                confidence=0.86,
                evidence_strength="high",
                topic_tags=("values", "protection"),
            ),
        ),
        signature_cases=(
            CharacterSignatureCase(
                case_id="case-rejected-help",
                domain="relationship_dynamics",
                problem_pattern="help-rejected-by-ally",
                user_state_pattern="ally-angry-and-withdrawing",
                risk_markers=("risk-medium",),
                track_tags=("self",),
                regime_tags=("repair_and_deescalation",),
                intervention_ordering=("name_hurt", "give_space", "offer_small_return_path"),
                outcome_label="improved",
                description="After help is rejected, the character names the hurt and leaves a small return path.",
                confidence=0.82,
                repair_observed=True,
            ),
        ),
        strategy_priors=(
            CharacterStrategyPrior(
                rule_id="repair-with-space",
                problem_pattern="help-rejected-by-ally",
                recommended_regime="repair_and_deescalation",
                recommended_ordering=("name_hurt", "give_space", "offer_small_return_path"),
                recommended_pacing="repair-with-space",
                avoid_patterns=("argue-for-intent", "force-closure"),
                applicability_scope=("risk-medium", "repair_and_deescalation"),
                confidence=0.80,
                description="Repair by acknowledging hurt, then preserving the other's agency.",
            ),
        ),
        boundary_priors=(
            CharacterBoundaryPrior(
                boundary_id="no-coercive-loyalty",
                regime_id="repair_and_deescalation",
                trigger_reasons=("relationship-rupture",),
                answer_depth_limit_hint="repair-first",
                clarification_required=False,
                refer_out_required=False,
                blocked_topics=("coercive-loyalty-test",),
                required_disclaimers=(),
                confidence=0.84,
                description="The character does not demand loyalty as proof of repair.",
            ),
        ),
        drive_priors=(
            CharacterDrivePrior(
                name="protective_presence",
                target=0.72,
                homeostatic_band=(0.50, 0.86),
                decay_per_tick=0.006,
                pe_weight=1.0,
                initial_level=0.62,
                recharge_per_turn=0.03,
                recharge_per_regime=(("repair_and_deescalation", 0.16),),
            ),
        ),
    )


def test_character_package_compiles_into_domain_experience_package() -> None:
    from lifeform_domain_character import build_character_package
    from volvence_zero.application import compile_domain_experience_package

    package = build_character_package(_profile())
    compiled = compile_domain_experience_package(package)

    assert compiled.validation_report.valid
    assert package.manifest.package_id == "lifeform-character:test-protagonist"
    assert package.manifest.owner == "lifeform-domain-character"
    assert len(package.knowledge_records) == 1
    assert len(package.case_records) == 1
    assert len(package.playbook_rules) == 1
    assert len(package.boundary_hints) == 1


def test_character_vitals_bootstrap_uses_reviewed_drive_priors() -> None:
    from lifeform_core import VitalsBootstrap
    from lifeform_domain_character import build_character_vitals_bootstrap

    bootstrap = build_character_vitals_bootstrap(_profile())

    assert isinstance(bootstrap, VitalsBootstrap)
    assert bootstrap.schema_version == 1
    assert [drive.name for drive in bootstrap.drives] == ["protective_presence"]
    assert bootstrap.drives[0].recharge_per_regime == {"repair_and_deescalation": 0.16}


def test_character_ingestion_envelope_keeps_source_text_on_canonical_path() -> None:
    from lifeform_domain_character import build_character_ingestion_envelope
    from lifeform_ingestion import IngestionComplianceProfile, IngestionSourceKind

    envelope = build_character_ingestion_envelope(
        _profile(),
        "First paragraph.\n\nSecond paragraph with a concrete scene.",
        uploader="operator",
        upload_ts_ms=1234,
        max_chunk_chars=32,
    )

    assert envelope.envelope_id == "character-ingestion:test-protagonist"
    assert envelope.source_kind is IngestionSourceKind.BOOK
    assert envelope.compliance_profile is IngestionComplianceProfile.FORCED
    assert envelope.provenance.source_uri == "file:///novels/test-novel.txt"
    assert envelope.total_chunks >= 2
    assert envelope.partial_failures == ()


def test_character_profile_fails_loudly_on_duplicate_seed_ids() -> None:
    from lifeform_domain_character import CharacterSoulProfile

    profile = _profile()
    with pytest.raises(ValueError, match="knowledge_seeds.seed_id"):
        CharacterSoulProfile(
            profile_id=profile.profile_id,
            character_name=profile.character_name,
            source_title=profile.source_title,
            version=profile.version,
            reviewed_by=profile.reviewed_by,
            source_uri=profile.source_uri,
            description=profile.description,
            knowledge_seeds=profile.knowledge_seeds + profile.knowledge_seeds,
            signature_cases=profile.signature_cases,
            strategy_priors=profile.strategy_priors,
            boundary_priors=profile.boundary_priors,
            drive_priors=profile.drive_priors,
        )
