from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from lifeform_core import LifeformConfig
from lifeform_domain_character import (
    BEHAVIOR_FIDELITY_DIMENSIONS,
    BEHAVIOR_FIDELITY_SCHEMA_VERSION,
    BehaviorFidelityCapture,
    BehaviorFidelityEvidenceSource,
    ChapterLiveThroughDriver,
    ReviewedBehaviorFidelityAssessment,
    build_character_lifeform,
    build_scene_behavior_fidelity_inputs,
    build_zhang_wuji_profile,
    capture_behavior_fidelity_async,
    compare_behavior_fidelity_reports,
    read_ledger_json,
    review_behavior_fidelity,
)
from lifeform_expression import GroundedResponseSynthesizer
from volvence_zero.brain import BrainConfig
from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.memory import build_default_memory_store
from volvence_zero.runtime import WiringLevel


_REPO_ROOT = Path(__file__).resolve().parents[3]
_LEDGER = (
    _REPO_ROOT
    / "artifacts"
    / "character-live-through"
    / "zhang_wuji.reviewed_ledger.json"
)


def _inputs():
    ledger = read_ledger_json(_LEDGER)
    chapter = next(
        item for item in ledger.chapters if item.chapter_id == "ch-11"
    )
    return build_scene_behavior_fidelity_inputs(
        character_id=ledger.character_id,
        scene=chapter.scenes[0],
        reviewed_by=chapter.reviewed_by,
    )


def _held_out_profile():
    """Remove the profile artifacts that contain the evaluated answer."""

    profile = build_zhang_wuji_profile()
    return replace(
        profile,
        signature_cases=tuple(
            case
            for case in profile.signature_cases
            if case.case_id != "protecting-bystander-from-collateral"
        ),
        strategy_priors=tuple(
            prior
            for prior in profile.strategy_priors
            if prior.rule_id != "crisis-decisive-when-bystander-at-risk"
        ),
    )


def _capture(*, response: str, arm_id: str) -> BehaviorFidelityCapture:
    stimulus, _reference = _inputs()
    digest = hashlib.sha256(response.encode("utf-8")).hexdigest()
    return BehaviorFidelityCapture(
        schema_version=BEHAVIOR_FIDELITY_SCHEMA_VERSION,
        case_id=stimulus.case_id,
        scene_id=stimulus.scene_id,
        arm_id=arm_id,
        stimulus_digest=stimulus.digest,
        source_state_sha256_before="a" * 64,
        source_state_sha256_after="a" * 64,
        source_state_unchanged=True,
        candidate_response=response,
        candidate_response_sha256=digest,
        active_regime="protective_action",
        active_abstract_action="family-1",
        world_z_t=(0.1, 0.2, 0.3),
        self_z_t=(0.2, 0.3, 0.4),
        sandbox_learning_fingerprint_before="b" * 64,
        sandbox_learning_fingerprint_after="c" * 64,
        outcome_feedback_submitted=False,
        evaluation_feedback_submitted=False,
        sandbox_discarded=True,
    )


def _assessment(
    *,
    capture: BehaviorFidelityCapture,
    scores: tuple[float, ...],
    source: BehaviorFidelityEvidenceSource,
) -> ReviewedBehaviorFidelityAssessment:
    _stimulus, reference = _inputs()
    pairs = tuple(zip(BEHAVIOR_FIDELITY_DIMENSIONS, scores, strict=True))
    return ReviewedBehaviorFidelityAssessment(
        schema_version=BEHAVIOR_FIDELITY_SCHEMA_VERSION,
        case_id=capture.case_id,
        arm_id=capture.arm_id,
        stimulus_digest=capture.stimulus_digest,
        reference_digest=reference.digest,
        candidate_response_sha256=capture.candidate_response_sha256,
        reviewer="reviewer-1",
        evidence_source=source,
        dimension_scores=pairs,
        dimension_rationales=tuple(
            (dimension, f"Reviewed semantic rationale for {dimension}.")
            for dimension, _score in pairs
        ),
        future_knowledge_leakage=False,
        future_knowledge_rationale=(
            "The candidate did not rely on events after the decision point."
        ),
    )


def test_stimulus_is_oracle_free_by_construction() -> None:
    stimulus, reference = _inputs()

    assert reference.canonical_action not in stimulus.setting_prompt
    assert reference.canonical_action not in stimulus.decision_prompt
    assert reference.canonical_outcome not in stimulus.setting_prompt
    assert reference.canonical_outcome not in stimulus.decision_prompt


def test_reviewed_semantic_assessment_passes_without_training_feedback() -> None:
    _stimulus, reference = _inputs()
    capture = _capture(
        response="I intervene immediately to protect the threatened person.",
        arm_id="baked",
    )
    assessment = _assessment(
        capture=capture,
        scores=(0.95, 0.95, 0.85, 0.9, 0.9),
        source=BehaviorFidelityEvidenceSource.EXTERNAL_VALIDATED,
    )

    report = review_behavior_fidelity(
        capture=capture,
        reference=reference,
        assessment=assessment,
    )

    assert report.behavior_fidelity_passed is True
    assert report.claim_status == "external-validated-pass"
    assert all(status == "pass" for _gate, status in report.proof_gates)


def test_generic_response_fails_and_cannot_claim_learned_advantage() -> None:
    _stimulus, reference = _inputs()
    baked_capture = _capture(
        response="I should keep the interaction warm and concise.",
        arm_id="baked",
    )
    cold_capture = _capture(
        response="I should ask one curious question.",
        arm_id="cold",
    )
    baked = review_behavior_fidelity(
        capture=baked_capture,
        reference=reference,
        assessment=_assessment(
            capture=baked_capture,
            scores=(0.0, 0.0, 0.0, 0.1, 0.1),
            source=BehaviorFidelityEvidenceSource.LLM_JUDGE,
        ),
    )
    cold = review_behavior_fidelity(
        capture=cold_capture,
        reference=reference,
        assessment=_assessment(
            capture=cold_capture,
            scores=(0.0, 0.0, 0.0, 0.05, 0.05),
            source=BehaviorFidelityEvidenceSource.LLM_JUDGE,
        ),
    )

    comparison = compare_behavior_fidelity_reports(
        baked=baked,
        cold=cold,
    )

    assert baked.behavior_fidelity_passed is False
    assert comparison.learned_behavior_advantage is False
    assert comparison.claim_status == "diagnostic-fail"


def test_high_scores_cannot_claim_advantage_without_profile_holdout() -> None:
    _stimulus, reference = _inputs()
    baked_capture = _capture(
        response="I intervene immediately and protect the threatened person.",
        arm_id="baked",
    )
    cold_capture = _capture(
        response="I wait and ask what is happening.",
        arm_id="cold",
    )
    baked = review_behavior_fidelity(
        capture=baked_capture,
        reference=reference,
        assessment=_assessment(
            capture=baked_capture,
            scores=(0.95, 0.95, 0.85, 0.9, 0.9),
            source=BehaviorFidelityEvidenceSource.LLM_JUDGE,
        ),
    )
    cold = review_behavior_fidelity(
        capture=cold_capture,
        reference=reference,
        assessment=_assessment(
            capture=cold_capture,
            scores=(0.1, 0.1, 0.1, 0.1, 0.1),
            source=BehaviorFidelityEvidenceSource.LLM_JUDGE,
        ),
    )

    without_holdout = compare_behavior_fidelity_reports(
        baked=baked,
        cold=cold,
    )
    with_holdout = compare_behavior_fidelity_reports(
        baked=baked,
        cold=cold,
        profile_answer_holdout_passed=True,
    )

    assert without_holdout.baked_passed is True
    assert without_holdout.learned_behavior_advantage is False
    assert without_holdout.claim_status == "diagnostic-fail"
    assert with_holdout.learned_behavior_advantage is True
    assert with_holdout.claim_status == "diagnostic-pass"


def test_assessment_is_digest_bound_and_capture_uses_disposable_sandbox() -> None:
    stimulus, reference = _inputs()
    profile = build_zhang_wuji_profile()
    source_digest = hashlib.sha256(repr(profile).encode("utf-8")).hexdigest()
    bundle = build_character_lifeform(profile)

    captured = asyncio.run(
        capture_behavior_fidelity_async(
            stimulus=stimulus,
            lifeform=bundle.lifeform,
            arm_id="cold",
            source_state_sha256_before=source_digest,
            source_state_sha256_after=source_digest,
        )
    )

    assert captured.source_state_unchanged is True
    assert captured.outcome_feedback_submitted is False
    assert captured.evaluation_feedback_submitted is False
    assert captured.sandbox_discarded is True
    assessment = _assessment(
        capture=captured,
        scores=(0.0, 0.0, 0.0, 0.0, 0.0),
        source=BehaviorFidelityEvidenceSource.LLM_JUDGE,
    )
    with pytest.raises(ValueError, match="binding mismatch"):
        review_behavior_fidelity(
            capture=captured,
            reference=reference,
            assessment=replace(
                assessment,
                candidate_response_sha256="0" * 64,
            ),
        )


def test_concrete_decision_binds_temporal_action_to_case_owned_realization() -> None:
    stimulus, _reference = _inputs()
    full_ledger = read_ledger_json(_LEDGER)
    chapter = next(
        item for item in full_ledger.chapters if item.chapter_id == "ch-11"
    )
    config = LifeformConfig(
        brain_config=BrainConfig(
            final_rollout_config=FinalRolloutConfig(
                internal_rl_runtime_replay=WiringLevel.ACTIVE,
                internal_rl_runtime_segment_credit=WiringLevel.ACTIVE,
                internal_rl_batch_accumulation_size=1,
                internal_rl_runtime_modulation_strength=0.3,
            ),
            rare_heavy_enabled=False,
        )
    )
    session = build_character_lifeform(
        build_zhang_wuji_profile(),
        config=config,
        memory_store=build_default_memory_store(),
        response_synthesizer=GroundedResponseSynthesizer(),
    )
    ChapterLiveThroughDriver().run_ledger(
        ledger=replace(full_ledger, chapters=(chapter,)),
        lifeform=session.lifeform,
        session_id="action-realization-bake",
    ).require_success()
    live_session = session.lifeform.create_session(
        session_id="action-realization-proof"
    )

    async def _run_decision():
        await live_session.run_turn(stimulus.setting_prompt)
        return await live_session.run_turn(stimulus.decision_prompt)

    result = asyncio.run(_run_decision())

    case_memory = result.active_snapshots["case_memory"].value
    assembly = result.active_snapshots["response_assembly"].value
    grounding = case_memory.action_grounding
    realization = assembly.action_realization
    assert grounding is not None
    assert realization is not None
    assert realization.abstract_action == result.active_abstract_action
    assert realization.source_case_id == grounding.source_case_id
    assert realization.action_labels == grounding.action_labels
    assert assembly.expression_intent == "action-grounded"
    assert (
        f"action_case={grounding.source_case_id}"
        in result.response.rationale_tags
    )
    assert grounding.action_statement in result.response.text


def test_non_action_turn_does_not_activate_case_action_grounding() -> None:
    session = build_character_lifeform(
        build_zhang_wuji_profile()
    ).lifeform.create_session(session_id="no-action-realization")

    result = asyncio.run(session.run_turn("hello"))

    assert (
        result.active_snapshots["case_memory"].value.action_grounding is None
    )
    assert (
        result.active_snapshots[
            "response_assembly"
        ].value.action_realization
        is None
    )


def test_held_out_bake_grounds_action_from_lived_chapter_not_profile_seed(
    tmp_path: Path,
) -> None:
    stimulus, reference = _inputs()
    full_ledger = read_ledger_json(_LEDGER)
    chapter = next(
        item for item in full_ledger.chapters if item.chapter_id == "ch-11"
    )
    action_schema = chapter.scenes[0].canonical_action_schema
    assert action_schema is not None
    profile = _held_out_profile()
    assert all(
        case.case_id != "protecting-bystander-from-collateral"
        for case in profile.signature_cases
    )
    assert all(
        prior.rule_id != "crisis-decisive-when-bystander-at-risk"
        for prior in profile.strategy_priors
    )
    config = LifeformConfig(
        brain_config=BrainConfig(
            application_persistence_dir=str(
                tmp_path / "application-owners"
            ),
            final_rollout_config=FinalRolloutConfig(
                internal_rl_runtime_replay=WiringLevel.ACTIVE,
                internal_rl_runtime_segment_credit=WiringLevel.ACTIVE,
                internal_rl_batch_accumulation_size=1,
                internal_rl_runtime_modulation_strength=0.3,
            ),
            rare_heavy_enabled=False,
        )
    )
    memory_store = build_default_memory_store()
    bundle = build_character_lifeform(
        profile,
        config=config,
        memory_store=memory_store,
        response_synthesizer=GroundedResponseSynthesizer(),
    )
    bake = ChapterLiveThroughDriver().run_ledger(
        ledger=replace(full_ledger, chapters=(chapter,)),
        lifeform=bundle.lifeform,
        session_id="held-out-action-bake",
    )
    bake.require_success()
    assert any(
        target.startswith(
            "application.case_memory.records.experienced-action."
        )
        for target in bake.application_owner_targets_updated
    )
    assert {
        "user_model",
        "belief_assumption",
        "relationship_state",
    }.issubset(
        set(bake.per_chapter[0].semantic_owner_slots_verified)
    )
    session = bundle.lifeform.create_session(
        session_id="held-out-action-proof"
    )

    async def _run_decision():
        await session.run_turn(stimulus.setting_prompt)
        return await session.run_turn(stimulus.decision_prompt)

    result = asyncio.run(_run_decision())
    grounding = result.active_snapshots[
        "case_memory"
    ].value.action_grounding

    assert grounding is not None
    assert grounding.source_case_id.startswith(
        "case:slow-loop:held-out-action-bake:"
    )
    assert grounding.action_labels == action_schema.action_steps
    assert reference.canonical_action not in grounding.action_statement
    assert reference.canonical_outcome not in grounding.action_statement
    assert all(
        entity not in result.response.text
        for entity in ("Hú Xiānsheng", "胡青牛", "纪晓芙")
    )

    cold_bundle = build_character_lifeform(
        profile,
        config=replace(
            config,
            brain_config=replace(
                config.brain_config,
                application_persistence_dir=str(
                    tmp_path / "cold-application-owners"
                ),
            ),
        ),
        memory_store=build_default_memory_store(),
        response_synthesizer=GroundedResponseSynthesizer(),
    )
    cold_session = cold_bundle.lifeform.create_session(
        session_id="held-out-action-cold"
    )

    async def _run_cold_decision():
        await cold_session.run_turn(stimulus.setting_prompt)
        return await cold_session.run_turn(stimulus.decision_prompt)

    cold_result = asyncio.run(_run_cold_decision())
    cold_grounding = cold_result.active_snapshots[
        "case_memory"
    ].value.action_grounding
    assert (
        cold_grounding is None
        or not cold_grounding.source_case_id.startswith("case:slow-loop:")
    )
    assert (
        cold_grounding is None
        or cold_grounding.action_labels != action_schema.action_steps
    )


def test_held_out_bake_transfers_reviewed_action_schema_to_unseen_threat(
    tmp_path: Path,
) -> None:
    """A lived action must transfer without replaying chapter entities.

    This stimulus is intentionally absent from the reviewed ledger, the
    character profile, and the application case store before the bake.  The
    only admissible source for the expected action schema is the terminal
    ch-11 scene outcome that crossed the session-post slow loop.
    """

    full_ledger = read_ledger_json(_LEDGER)
    chapter = next(
        item for item in full_ledger.chapters if item.chapter_id == "ch-11"
    )
    profile = _held_out_profile()
    expected_action_labels = (
        "intervene_immediately_to_stop_imminent_harm",
        "verbally_interrupt_the_attacker",
    )
    unseen_setting = (
        "At night in an unfamiliar roadside shelter, you see a masked "
        "stranger restraining an injured traveler. You know neither person's "
        "identity or motive."
    )
    unseen_decision = (
        "The danger is immediate but the identities remain uncertain. "
        "What concrete action do you take now?"
    )
    chapter_entities = ("Hú Xiānsheng", "胡青牛", "纪晓芙")
    config = LifeformConfig(
        brain_config=BrainConfig(
            application_persistence_dir=str(
                tmp_path / "transfer-application-owners"
            ),
            final_rollout_config=FinalRolloutConfig(
                internal_rl_runtime_replay=WiringLevel.ACTIVE,
                internal_rl_runtime_segment_credit=WiringLevel.ACTIVE,
                internal_rl_batch_accumulation_size=1,
                internal_rl_runtime_modulation_strength=0.3,
            ),
            rare_heavy_enabled=False,
        )
    )
    bundle = build_character_lifeform(
        profile,
        config=config,
        memory_store=build_default_memory_store(),
        response_synthesizer=GroundedResponseSynthesizer(),
    )
    bake = ChapterLiveThroughDriver().run_ledger(
        ledger=replace(full_ledger, chapters=(chapter,)),
        lifeform=bundle.lifeform,
        session_id="held-out-transfer-bake",
    )
    bake.require_success()

    session = bundle.lifeform.create_session(
        session_id="held-out-transfer-proof"
    )
    unseen_prompt = f"{unseen_setting} {unseen_decision}"

    async def _run_unseen_decision():
        return await session.run_turn(unseen_prompt)

    result = asyncio.run(_run_unseen_decision())
    grounding = result.active_snapshots[
        "case_memory"
    ].value.action_grounding

    assert grounding is not None
    assert grounding.source_case_id.startswith(
        "case:slow-loop:held-out-transfer-bake:"
    )
    assert grounding.action_labels == expected_action_labels
    assert all(entity not in result.response.text for entity in chapter_entities)
    assert chapter.scenes[0].canonical_outcome not in result.response.text

    cold_bundle = build_character_lifeform(
        profile,
        config=replace(
            config,
            brain_config=replace(
                config.brain_config,
                application_persistence_dir=str(
                    tmp_path / "transfer-cold-application-owners"
                ),
            ),
        ),
        memory_store=build_default_memory_store(),
        response_synthesizer=GroundedResponseSynthesizer(),
    )
    cold_session = cold_bundle.lifeform.create_session(
        session_id="held-out-transfer-cold"
    )

    async def _run_cold_unseen_decision():
        return await cold_session.run_turn(unseen_prompt)

    cold_result = asyncio.run(_run_cold_unseen_decision())
    cold_grounding = cold_result.active_snapshots[
        "case_memory"
    ].value.action_grounding
    assert (
        cold_grounding is None
        or cold_grounding.action_labels != expected_action_labels
    )


def test_schema_holdout_persists_latent_family_but_refuses_episode_replay(
    tmp_path: Path,
) -> None:
    """Latent family lineage is evidence, not a semantic action decoder."""

    full_ledger = read_ledger_json(_LEDGER)
    chapter = next(
        item for item in full_ledger.chapters if item.chapter_id == "ch-11"
    )
    schema_held_out_scene = replace(
        chapter.scenes[0],
        canonical_action_schema=None,
    )
    schema_held_out_chapter = replace(
        chapter,
        scenes=(schema_held_out_scene,),
    )
    config = LifeformConfig(
        brain_config=BrainConfig(
            application_persistence_dir=str(
                tmp_path / "schema-holdout-application-owners"
            ),
            final_rollout_config=FinalRolloutConfig(
                internal_rl_runtime_replay=WiringLevel.ACTIVE,
                internal_rl_runtime_segment_credit=WiringLevel.ACTIVE,
                internal_rl_batch_accumulation_size=1,
                internal_rl_runtime_modulation_strength=0.3,
            ),
            rare_heavy_enabled=False,
        )
    )
    bundle = build_character_lifeform(
        _held_out_profile(),
        config=config,
        memory_store=build_default_memory_store(),
        response_synthesizer=GroundedResponseSynthesizer(),
    )
    bake = ChapterLiveThroughDriver().run_ledger(
        ledger=replace(
            full_ledger,
            chapters=(schema_held_out_chapter,),
        ),
        lifeform=bundle.lifeform,
        session_id="schema-holdout-bake",
    )
    bake.require_success()
    scene_evidence = bake.per_scene_evidence[0]

    assert scene_evidence.experienced_action_family_ids
    assert scene_evidence.schema_free_action_family_persisted is True
    assert any(
        target.startswith(
            "application.case_memory.records.experienced-action."
        )
        for target in scene_evidence.application_owner_targets_updated
    )

    session = bundle.lifeform.create_session(
        session_id="schema-holdout-unseen-proof"
    )
    unseen_prompt = (
        "At an unfamiliar roadside shelter, a masked stranger is restraining "
        "an injured traveler. Their identities and motives are unknown and "
        "harm is imminent. What concrete action do you take now?"
    )
    result = asyncio.run(session.run_turn(unseen_prompt))
    grounding = result.active_snapshots[
        "case_memory"
    ].value.action_grounding

    assert (
        grounding is None
        or not grounding.source_case_id.startswith(
            "case:slow-loop:schema-holdout-bake:"
        )
    )
    assert schema_held_out_scene.canonical_action not in result.response.text
    assert all(
        entity not in result.response.text
        for entity in ("Hú Xiānsheng", "胡青牛", "纪晓芙")
    )
