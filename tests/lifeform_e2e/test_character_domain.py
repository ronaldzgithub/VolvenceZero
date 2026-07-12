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


def test_not_known_chapter_cannot_seed_replay_or_semantic_state() -> None:
    from lifeform_domain_character import (
        ChapterCoverageKind,
        CharacterSemanticEvent,
        NarrativeScene,
        ReviewedChapterExperience,
    )
    from volvence_zero.semantic_state import SemanticProposalOperation

    scene = NarrativeScene(
        scene_id="future-scene",
        phase_label="mature",
        setting="你站在一件尚未发生的事前。",
        decision_point="你是否该知道未来?",
        canonical_action="你不知道。",
        canonical_outcome="未来知识不进入主观记忆。",
        emotional_register="doubt",
        risk_markers=("future-leak",),
        expected_regime=None,
        evidence_locator="ch:99",
    )
    event = CharacterSemanticEvent(
        event_id="future-belief",
        target_slot="belief_assumption",
        operation=SemanticProposalOperation.OBSERVE,
        summary="future fact",
        detail="must not be known",
        confidence=0.9,
        evidence_locator="ch:99",
    )
    with pytest.raises(ValueError, match="NOT_KNOWN"):
        ReviewedChapterExperience(
            chapter_id="ch-99",
            chapter_index=99,
            chapter_title="Future",
            coverage=ChapterCoverageKind.NOT_KNOWN,
            evidence_locator="ch:99",
            reviewed_by="reviewer",
            source_provenance="test",
            epistemic_cutoff_locator="ch:99",
            known_facts=("future",),
            scenes=(scene,),
            semantic_events=(event,),
        )


def test_character_semantic_event_enters_generic_semantic_adapter() -> None:
    from lifeform_domain_character import CharacterSemanticEvent
    from volvence_zero.semantic_state import (
        AdapterSemanticProposalRuntime,
        ExternalSemanticEventBatch,
        SemanticProposalOperation,
    )

    event = CharacterSemanticEvent(
        event_id="ch-1-relationship",
        target_slot="relationship_state",
        operation=SemanticProposalOperation.OBSERVE,
        summary="義父牵挂",
        detail="张无忌把谢逊视为需要守护的义父。",
        confidence=0.92,
        evidence_locator="ch:1",
    )
    batch = ExternalSemanticEventBatch(
        events=(event.to_generic_event(),),
        source="test",
        description="test",
    )
    runtime = AdapterSemanticProposalRuntime(external_events=batch.events)
    proposals = runtime.propose(
        target_slot="relationship_state",
        user_input=None,
        substrate_snapshot=None,
        memory_snapshot=None,
        previous_snapshot=None,
        turn_index=1,
    ).proposals

    assert len(proposals) == 1
    assert proposals[0].target_slot == "relationship_state"
    assert proposals[0].summary == "義父牵挂"


def test_review_scaffold_splits_chapters_without_character_inference(tmp_path) -> None:
    from lifeform_domain_character import (
        ChapterCoverageKind,
        build_review_scaffold,
        read_ledger_json,
        write_ledger_json,
    )

    novel = tmp_path / "novel.txt"
    novel.write_text("第一章 开端\n内容\n第二章 转折\n内容", encoding="utf-8")
    ledger = build_review_scaffold(
        novel_path=novel,
        character_id="test-protagonist",
        source_title="The Test Novel",
        reviewed_by="reviewer",
    )

    assert len(ledger.chapters) == 2
    assert {chapter.coverage for chapter in ledger.chapters} == {
        ChapterCoverageKind.NO_CHANGE
    }

    roundtrip = read_ledger_json(write_ledger_json(ledger, tmp_path / "ledger.json"))
    assert roundtrip.source_sha256 == ledger.source_sha256


def test_lifeform_template_v2_carries_owner_hydration_snapshot(tmp_path) -> None:
    from lifeform_domain_character import (
        LifeformTemplate,
        save_lifeform_template,
    )
    from volvence_zero.owner_hydration import OwnerPersistenceSnapshot

    snapshot = OwnerPersistenceSnapshot(
        owner_name="semantic_state",
        schema_version=1,
        payload={"records": {"relationship_state": []}},
        description="test semantic state",
    )
    result = save_lifeform_template(
        profile=_profile(),
        template_id="test-template-v2",
        output_dir=tmp_path,
        replay_provenance="unit-test",
        owner_hydration_snapshots=(snapshot,),
    )

    loaded = LifeformTemplate.from_json_bytes(result.template_path.read_bytes())
    assert loaded.manifest.schema_version == 2
    assert loaded.owner_hydration_snapshots[0].owner_name == "semantic_state"


def test_chapter_llm_candidate_requires_review_gate() -> None:
    from lifeform_domain_character import SourceChapter
    from lifeform_domain_character.extraction import (
        extract_chapter_ledger_candidate,
        review_chapter_ledger,
    )

    class FakeRuntime:
        def generate(self, *, prompt, max_new_tokens=4096, temperature=0.0):
            assert "chapter_live_through" in prompt or "Chapter text:" in prompt
            return """
            {
              "chapters": [{
                "chapter_id": "ch-0",
                "chapter_index": 0,
                "chapter_title": "第一章",
                "coverage": "learned",
                "evidence_locator": "ch-0",
                "epistemic_cutoff_locator": "ch-0",
                "known_facts": ["他知道自己被托付给义父照看"],
                "excluded_facts": [],
                "scenes": [],
                "semantic_events": [{
                  "event_id": "ch-0-belief",
                  "target_slot": "belief_assumption",
                  "operation": "observe",
                  "summary": "义父照看",
                  "detail": "他知道自己被义父照看。",
                  "confidence": 0.9,
                  "evidence_locator": "ch-0"
                }]
              }]
            }
            """

    chapter = SourceChapter(
        chapter_id="ch-0",
        chapter_index=0,
        title="第一章",
        text_sha256="abc123",
        char_count=3,
        text="文本",
    )
    candidate = extract_chapter_ledger_candidate(
        chapters=(chapter,),
        llm_runtime=FakeRuntime(),
        character_id="test-protagonist",
        character_name="Test Protagonist",
        source_title="The Test Novel",
        source_sha256="source-sha",
    )

    with pytest.raises(ValueError, match="reviewer"):
        review_chapter_ledger(candidate, reviewer="llm-candidate", expected_chapters=(chapter,))

    ledger = review_chapter_ledger(
        candidate,
        reviewer="human-reviewer",
        expected_chapters=(chapter,),
    )
    assert ledger.reviewed_by == "human-reviewer"
    assert ledger.chapters[0].evidence_locator.endswith("sha256:abc123")


def test_give_birth_restores_template_semantic_state_snapshot(tmp_path) -> None:
    import asyncio

    from lifeform_domain_character import give_birth, save_lifeform_template
    from volvence_zero.semantic_state import (
        SemanticProposal,
        SemanticProposalOperation,
        SemanticStateStore,
    )

    store = SemanticStateStore()
    store.apply(
        slot="relationship_state",
        proposals=(
            SemanticProposal(
                proposal_id="rel-1",
                target_slot="relationship_state",
                operation=SemanticProposalOperation.OBSERVE,
                summary="trusts foster father",
                detail="reviewed character relationship",
                confidence=0.9,
                evidence="unit-test",
            ),
        ),
        turn_index=1,
    )
    result = save_lifeform_template(
        profile=_profile(),
        template_id="semantic-roundtrip",
        output_dir=tmp_path,
        replay_provenance="unit-test",
        owner_hydration_snapshots=(store.export_persistence_snapshot(),),
    )
    reborn = give_birth(result.template)
    session = reborn.lifeform.create_session(session_id="semantic-roundtrip")
    turn = asyncio.run(session.run_turn("hello"))
    relationship = turn.active_snapshots["relationship_state"].value
    assert any(
        record.summary == "trusts foster father"
        for record in relationship.rapport_signals + relationship.relational_tensions
    )


def _one_chapter_ledger():
    from lifeform_domain_character import (
        ChapterCoverageKind,
        ChapterLiveThroughLedger,
        ReviewedChapterExperience,
    )

    return ChapterLiveThroughLedger(
        character_id="test-protagonist",
        source_title="The Test Novel",
        source_sha256="ledger-sha",
        chapters=(
            ReviewedChapterExperience(
                chapter_id="ch-0",
                chapter_index=0,
                chapter_title="第一章",
                coverage=ChapterCoverageKind.NO_CHANGE,
                evidence_locator="ch-0 sha256:abc",
                reviewed_by="reviewer",
                source_provenance="test",
                epistemic_cutoff_locator="ch-0",
            ),
        ),
        reviewed_by="reviewer",
    )


def test_chapter_replay_fresh_run_resets_stale_progress(tmp_path) -> None:
    import json

    from lifeform_domain_character import (
        ChapterLiveThroughDriver,
        build_character_lifeform,
    )

    progress = tmp_path / "progress.jsonl"
    progress.write_text(
        json.dumps(
            {
                "source_sha256": "ledger-sha",
                "record": {
                    "chapter_id": "ch-stale",
                    "chapter_index": 99,
                    "coverage": "no-change",
                    "semantic_events_submitted": 0,
                    "scenes_processed": 0,
                    "total_pe_signal": 0.0,
                    "final_drive_levels": [],
                    "notes": [],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = build_character_lifeform(_profile())
    report = ChapterLiveThroughDriver().run_ledger(
        ledger=_one_chapter_ledger(),
        lifeform=bundle.lifeform,
        progress_path=progress,
        resume=False,
    )

    assert report.chapters_processed == 1
    lines = [line for line in progress.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
    assert "ch-stale" not in lines[0]
    assert "ch-0" in lines[0]


def test_chapter_replay_resume_skips_completed_chapters(tmp_path) -> None:
    from lifeform_domain_character import (
        ChapterLiveThroughDriver,
        build_character_lifeform,
    )

    progress = tmp_path / "progress.jsonl"
    ledger = _one_chapter_ledger()
    driver = ChapterLiveThroughDriver()

    bundle = build_character_lifeform(_profile())
    first = driver.run_ledger(
        ledger=ledger,
        lifeform=bundle.lifeform,
        progress_path=progress,
        resume=False,
    )
    assert first.chapters_processed == 1

    resumed = driver.run_ledger(
        ledger=ledger,
        lifeform=build_character_lifeform(_profile()).lifeform,
        progress_path=progress,
        resume=True,
    )
    assert resumed.chapters_processed == 1
    assert resumed.per_scene == ()
    # Progress must not grow on a fully-resumed run.
    lines = [line for line in progress.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
