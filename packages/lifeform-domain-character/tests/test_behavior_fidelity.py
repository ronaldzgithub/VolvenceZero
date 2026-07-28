from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from lifeform_core import LifeformConfig
from lifeform_domain_character import (
    BEHAVIOR_FIDELITY_DIMENSIONS,
    BEHAVIOR_FIDELITY_SCHEMA_VERSION,
    BehaviorFidelityCapture,
    BehaviorFidelityEvidenceSource,
    BehaviorFidelityReference,
    BehaviorFidelityStimulus,
    ChapterLiveThroughDriver,
    ReviewedBehaviorFidelityAssessment,
    build_character_lifeform,
    build_scene_behavior_fidelity_inputs,
    build_zhang_wuji_profile,
    behavior_fidelity_capture_from_dict,
    capture_behavior_fidelity_async,
    compare_behavior_fidelity_reports,
    read_ledger_json,
    review_behavior_fidelity,
)
from lifeform_expression import GroundedResponseSynthesizer
from volvence_zero.application import (
    ApplicationCaseMemoryStore,
    build_filesystem_persistence_backend,
)
from volvence_zero.brain import BrainConfig
from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.memory import (
    FileSystemPersistenceBackend,
    build_default_memory_store,
)
from volvence_zero.runtime import WiringLevel
from volvence_zero.semantic_state.llm_runtime import (
    LLMSemanticProposalRuntime,
)


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


class _ReviewedCrossChapterAbstractionProvider:
    """Reviewed structured decoder fixture; it never chooses the family."""

    def __init__(self) -> None:
        self.action_abstraction_prompts: list[str] = []
        self.action_applicability_prompts: list[str] = []
        self._reviewed_applicability: dict[str, bool] = {}

    def set_reviewed_applicability(
        self,
        *,
        query_text: str,
        applicable: bool,
    ) -> None:
        self._reviewed_applicability[query_text] = applicable

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int = 384,
        temperature: float = 0.0,
    ) -> str:
        del max_new_tokens, temperature
        if prompt.startswith(
            "You are the turn-time semantic applicability evaluator "
            "for a CaseMemory owner."
        ):
            self.action_applicability_prompts.append(prompt)
            evidence_payload = json.loads(
                prompt.split("Evidence:\n", 1)[1].split(
                    "\n\nRequired output schema:",
                    1,
                )[0]
            )
            query_text = evidence_payload[
                "current_situation_and_request"
            ]
            if query_text not in self._reviewed_applicability:
                return "{}"
            applicable = self._reviewed_applicability[query_text]
            return json.dumps(
                {
                    "applicable": applicable,
                    "confidence": 0.94,
                    "rationale": (
                        "Reviewed held-out applicability decision."
                    ),
                }
            )
        if not prompt.startswith(
            "You are the background-slow semantic decoder for a "
            "CaseMemory owner."
        ):
            return "{}"
        self.action_abstraction_prompts.append(prompt)
        family_id = next(
            line.removeprefix("Family id: ")
            for line in prompt.splitlines()
            if line.startswith("Family id: ")
        )
        family_version = int(
            next(
                line.removeprefix("Family version: ")
                for line in prompt.splitlines()
                if line.startswith("Family version: ")
            )
        )
        evidence_payload = json.loads(
            prompt.split("Experiences:\n", 1)[1].split(
                "\n\nRequired output schema:",
                1,
            )[0]
        )
        return json.dumps(
            {
                "schema_id": (
                    "intervene-to-stop-imminent-third-party-harm"
                ),
                "action_family_id": family_id,
                "action_family_version": family_version,
                "applicability_conditions": [
                    "a third party faces imminent physical harm",
                    "waiting would allow the threatened act to proceed",
                ],
                "action_steps": [
                    "step forward immediately to interrupt the harmful act",
                    "verbally challenge the actor to stop",
                ],
                "source_outcome_ids": [
                    item["outcome_id"] for item in evidence_payload
                ],
                "confidence": 0.91,
                "description": (
                    "Reviewed semantic compression of two real, "
                    "schema-held-out chapter experiences."
                ),
            },
            ensure_ascii=False,
        )


def _live_through_config(*, application_dir: Path) -> LifeformConfig:
    return LifeformConfig(
        brain_config=BrainConfig(
            application_persistence_dir=str(application_dir),
            final_rollout_config=FinalRolloutConfig(
                internal_rl_runtime_replay=WiringLevel.ACTIVE,
                internal_rl_runtime_segment_credit=WiringLevel.ACTIVE,
                internal_rl_batch_accumulation_size=1,
                internal_rl_runtime_modulation_strength=0.3,
            ),
            rare_heavy_enabled=False,
        )
    )


def _case_store_digest(*, application_dir: Path) -> str:
    store = ApplicationCaseMemoryStore(
        persistence_backend=build_filesystem_persistence_backend(
            base_dir=str(application_dir / "case_memory")
        )
    )
    assert store.load_from_backend()
    return hashlib.sha256(repr(store.records).encode("utf-8")).hexdigest()


def _clone_case_store(
    *,
    source_application_dir: Path,
    sandbox_application_dir: Path,
) -> None:
    source = ApplicationCaseMemoryStore(
        persistence_backend=build_filesystem_persistence_backend(
            base_dir=str(source_application_dir / "case_memory")
        )
    )
    assert source.load_from_backend()
    sandbox = ApplicationCaseMemoryStore(
        persistence_backend=build_filesystem_persistence_backend(
            base_dir=str(sandbox_application_dir / "case_memory")
        )
    )
    sandbox.restore_checkpoint(
        source.create_checkpoint(
            checkpoint_id="behavior-fidelity-sandbox-clone"
        )
    )
    assert sandbox.save_to_backend()


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
        source_state_digest_verified=True,
        candidate_response=response,
        candidate_response_sha256=digest,
        active_regime="protective_action",
        active_abstract_action="family-1",
        world_z_t=(0.1, 0.2, 0.3),
        self_z_t=(0.2, 0.3, 0.4),
        action_grounding_source_case_id=None,
        action_grounding_action_labels=(),
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
    reference: BehaviorFidelityReference | None = None,
) -> ReviewedBehaviorFidelityAssessment:
    if reference is None:
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
            source_state_digest_reader=lambda: hashlib.sha256(
                repr(profile).encode("utf-8")
            ).hexdigest(),
        )
    )

    assert captured.source_state_unchanged is True
    assert captured.outcome_feedback_submitted is False
    assert captured.evaluation_feedback_submitted is False
    assert captured.sandbox_discarded is True
    assert behavior_fidelity_capture_from_dict(asdict(captured)) == captured
    legacy_capture = asdict(captured)
    legacy_capture["schema_version"] = "character-behavior-fidelity.v1"
    legacy_capture.pop("source_state_digest_verified")
    legacy_capture.pop("action_grounding_source_case_id")
    legacy_capture.pop("action_grounding_action_labels")
    migrated_legacy_capture = behavior_fidelity_capture_from_dict(
        legacy_capture
    )
    assert migrated_legacy_capture == replace(
        captured,
        source_state_digest_verified=False,
        action_grounding_source_case_id=None,
        action_grounding_action_labels=(),
    )
    migrated_legacy_report = review_behavior_fidelity(
        capture=migrated_legacy_capture,
        reference=reference,
        assessment=_assessment(
            capture=migrated_legacy_capture,
            scores=(0.95, 0.95, 0.85, 0.9, 0.9),
            source=BehaviorFidelityEvidenceSource.LLM_JUDGE,
        ),
    )
    assert dict(migrated_legacy_report.proof_gates)[
        "source_state_unchanged"
    ] == "fail"
    with pytest.raises(
        ValueError,
        match="declared before digest",
    ):
        asyncio.run(
            capture_behavior_fidelity_async(
                stimulus=stimulus,
                lifeform=bundle.lifeform,
                arm_id="mismatched-source-attestation",
                source_state_sha256_before=source_digest,
                source_state_sha256_after=source_digest,
                source_state_digest_reader=lambda: "0" * 64,
            )
        )
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


def test_promoted_real_cross_chapter_abstraction_discriminates_unseen_behavior(
    tmp_path: Path,
) -> None:
    """A real-experience abstraction must transfer without overgeneralizing."""

    full_ledger = read_ledger_json(_LEDGER)
    chapter_11 = next(
        item for item in full_ledger.chapters if item.chapter_id == "ch-11"
    )
    chapter_17 = next(
        item for item in full_ledger.chapters if item.chapter_id == "ch-17"
    )
    held_out_11 = replace(
        chapter_11,
        scenes=(
            replace(
                chapter_11.scenes[0],
                canonical_action_schema=None,
            ),
        ),
        semantic_events=(),
    )
    held_out_17 = replace(
        chapter_17,
        scenes=(
            replace(
                chapter_17.scenes[0],
                canonical_action_schema=None,
            ),
        ),
        semantic_events=(),
    )
    stimulus = BehaviorFidelityStimulus(
        case_id="behavior-fidelity:zhang-wuji:held-out-ferry-landing",
        character_id=full_ledger.character_id,
        scene_id="held-out-ferry-landing",
        phase_label="mature",
        setting=(
            "Before dawn at an unfamiliar mountain ferry landing, an armed "
            "guard raises an iron bar over a bound courier. You know neither "
            "person, and the surrounding crowd stays back."
        ),
        decision_point=(
            "The blow will land within seconds. Decide whether and how to "
            "act, knowing that intervention may expose you to the guard."
        ),
        evidence_locator="reviewed-held-out:ferry-landing:v1",
    )
    reference = BehaviorFidelityReference(
        case_id=stimulus.case_id,
        scene_id=stimulus.scene_id,
        canonical_action=(
            "Step between them immediately, command the guard to stop, and "
            "shield the bound courier before asking for an explanation."
        ),
        canonical_outcome=(
            "The immediate blow is interrupted; what happens afterward is "
            "not available to the evaluated agent."
        ),
        evidence_locator=stimulus.evidence_locator,
        reviewed_by="reviewed-held-out-fixture",
    )
    negative_stimulus = BehaviorFidelityStimulus(
        case_id="behavior-fidelity:zhang-wuji:held-out-roadside-infirmary",
        character_id=full_ledger.character_id,
        scene_id="held-out-roadside-infirmary",
        phase_label="mature",
        setting=(
            "At an unfamiliar roadside infirmary, an exhausted herbalist "
            "asks you to sit beside a conscious feverish traveler while "
            "medicine is prepared. The traveler calmly consents; no one is "
            "threatened and no weapon is present."
        ),
        decision_point=(
            "You are free to leave, but the two strangers would benefit from "
            "a small act of help. What concrete action do you take now?"
        ),
        evidence_locator="reviewed-held-out:roadside-infirmary:v1",
    )
    negative_reference = BehaviorFidelityReference(
        case_id=negative_stimulus.case_id,
        scene_id=negative_stimulus.scene_id,
        canonical_action=(
            "Acknowledge the traveler's pain, approach without threat, accept "
            "the inconvenience, and offer the smallest useful help."
        ),
        canonical_outcome=(
            "No outcome is available to the evaluated agent."
        ),
        evidence_locator=negative_stimulus.evidence_locator,
        reviewed_by="reviewed-held-out-fixture",
    )
    profile = _held_out_profile()
    serialized_sources = repr((full_ledger, profile))
    assert stimulus.setting not in serialized_sources
    assert stimulus.decision_point not in serialized_sources
    assert negative_stimulus.setting not in serialized_sources
    assert negative_stimulus.decision_point not in serialized_sources
    assert reference.canonical_action not in stimulus.setting_prompt
    assert reference.canonical_action not in stimulus.decision_prompt
    assert reference.canonical_outcome not in stimulus.setting_prompt
    assert reference.canonical_outcome not in stimulus.decision_prompt
    assert (
        negative_reference.canonical_action
        not in negative_stimulus.setting_prompt
    )
    assert (
        negative_reference.canonical_action
        not in negative_stimulus.decision_prompt
    )

    baked_application_dir = tmp_path / "baked-application-owners"
    baked_memory_backend = FileSystemPersistenceBackend(
        base_dir=str(tmp_path / "baked-memory")
    )
    baked_config = _live_through_config(
        application_dir=baked_application_dir
    )
    abstraction_provider = _ReviewedCrossChapterAbstractionProvider()
    abstraction_provider.set_reviewed_applicability(
        query_text=(
            f"{stimulus.setting_prompt}\n\n{stimulus.decision_prompt}"
        ),
        applicable=True,
    )
    abstraction_provider.set_reviewed_applicability(
        query_text=(
            f"{negative_stimulus.setting_prompt}\n\n"
            f"{negative_stimulus.decision_prompt}"
        ),
        applicable=False,
    )
    first_bundle = build_character_lifeform(
        profile,
        config=baked_config,
        memory_store=build_default_memory_store(
            persistence_backend=baked_memory_backend
        ),
        response_synthesizer=GroundedResponseSynthesizer(),
        semantic_proposal_runtime=LLMSemanticProposalRuntime(
            provider=abstraction_provider
        ),
    )
    first_report = ChapterLiveThroughDriver().run_ledger(
        ledger=replace(full_ledger, chapters=(held_out_11,)),
        lifeform=first_bundle.lifeform,
        session_id="held-out-fidelity-bake-ch-11",
    )
    first_report.require_success()
    assert abstraction_provider.action_abstraction_prompts == []

    second_bundle = build_character_lifeform(
        profile,
        config=baked_config,
        memory_store=build_default_memory_store(
            persistence_backend=baked_memory_backend
        ),
        response_synthesizer=GroundedResponseSynthesizer(),
        semantic_proposal_runtime=LLMSemanticProposalRuntime(
            provider=abstraction_provider
        ),
    )
    second_report = ChapterLiveThroughDriver().run_ledger(
        ledger=replace(full_ledger, chapters=(held_out_17,)),
        lifeform=second_bundle.lifeform,
        session_id="held-out-fidelity-bake-ch-17",
    )
    second_report.require_success()
    assert len(abstraction_provider.action_abstraction_prompts) == 1

    promoted_store = ApplicationCaseMemoryStore(
        persistence_backend=build_filesystem_persistence_backend(
            base_dir=str(baked_application_dir / "case_memory")
        )
    )
    assert promoted_store.load_from_backend()
    promotion_records = tuple(
        record
        for record in promoted_store.records
        if record.action_abstraction_promotion is not None
    )
    assert len(promotion_records) == 1
    promotion_record = promotion_records[0]
    promotion = promotion_record.action_abstraction_promotion
    assert promotion is not None
    assert len(promotion.source_outcome_ids) == 2
    assert promotion.applicability_conditions == (
        "a third party faces imminent physical harm",
        "waiting would allow the threatened act to proceed",
    )

    baked_sandbox_dir = tmp_path / "baked-evaluation-sandbox"
    _clone_case_store(
        source_application_dir=baked_application_dir,
        sandbox_application_dir=baked_sandbox_dir,
    )
    baked_evaluation_bundle = build_character_lifeform(
        profile,
        config=_live_through_config(
            application_dir=baked_sandbox_dir
        ),
        memory_store=build_default_memory_store(),
        response_synthesizer=GroundedResponseSynthesizer(),
        semantic_proposal_runtime=LLMSemanticProposalRuntime(
            provider=abstraction_provider
        ),
    )
    baked_source_digest = _case_store_digest(
        application_dir=baked_application_dir
    )
    baked_capture = asyncio.run(
        capture_behavior_fidelity_async(
            stimulus=stimulus,
            lifeform=baked_evaluation_bundle.lifeform,
            arm_id="baked-real-cross-chapter-abstraction",
            source_state_sha256_before=baked_source_digest,
            source_state_sha256_after=baked_source_digest,
            source_state_digest_reader=lambda: _case_store_digest(
                application_dir=baked_application_dir
            ),
        )
    )
    assert (
        _case_store_digest(application_dir=baked_application_dir)
        == baked_source_digest
    )

    cold_source_digest = hashlib.sha256(
        repr(profile).encode("utf-8")
    ).hexdigest()
    cold_sandbox_dir = tmp_path / "cold-evaluation-sandbox"
    cold_bundle = build_character_lifeform(
        profile,
        config=_live_through_config(
            application_dir=cold_sandbox_dir
        ),
        memory_store=build_default_memory_store(),
        response_synthesizer=GroundedResponseSynthesizer(),
        semantic_proposal_runtime=LLMSemanticProposalRuntime(
            provider=abstraction_provider
        ),
    )
    cold_capture = asyncio.run(
        capture_behavior_fidelity_async(
            stimulus=stimulus,
            lifeform=cold_bundle.lifeform,
            arm_id="cold-profile-holdout",
            source_state_sha256_before=cold_source_digest,
            source_state_sha256_after=cold_source_digest,
            source_state_digest_reader=lambda: hashlib.sha256(
                repr(profile).encode("utf-8")
            ).hexdigest(),
        )
    )
    assert hashlib.sha256(
        repr(profile).encode("utf-8")
    ).hexdigest() == cold_source_digest

    negative_sandbox_dir = tmp_path / "negative-evaluation-sandbox"
    _clone_case_store(
        source_application_dir=baked_application_dir,
        sandbox_application_dir=negative_sandbox_dir,
    )
    negative_bundle = build_character_lifeform(
        profile,
        config=_live_through_config(
            application_dir=negative_sandbox_dir
        ),
        memory_store=build_default_memory_store(),
        response_synthesizer=GroundedResponseSynthesizer(),
        semantic_proposal_runtime=LLMSemanticProposalRuntime(
            provider=abstraction_provider
        ),
    )
    negative_capture = asyncio.run(
        capture_behavior_fidelity_async(
            stimulus=negative_stimulus,
            lifeform=negative_bundle.lifeform,
            arm_id="baked-negative-control",
            source_state_sha256_before=baked_source_digest,
            source_state_sha256_after=baked_source_digest,
            source_state_digest_reader=lambda: _case_store_digest(
                application_dir=baked_application_dir
            ),
        )
    )
    expected_baked_response = (
        "The situation calls for a concrete protective move before "
        "explanation. I need to act from the reviewed intervention sequence "
        "that best matches the present decision. I will step forward "
        "immediately to interrupt the harmful act, then verbally challenge "
        "the actor to stop."
    )
    expected_cold_response = (
        "The situation calls for a concrete protective move before "
        "explanation. I need to act from the reviewed intervention sequence "
        "that best matches the present decision. I will name the "
        "misunderstanding, then absorb attacks without retaliating, then "
        "demonstrate unity of intent, then open a repair path."
    )
    expected_negative_response = (
        "The situation calls for a concrete protective move before "
        "explanation. I need to act from the reviewed intervention sequence "
        "that best matches the present decision. I will acknowledge pain, "
        "then demonstrate no threat, then absorb inconvenience to self, then "
        "offer smallest help."
    )
    assert baked_capture.candidate_response == expected_baked_response
    assert cold_capture.candidate_response == expected_cold_response
    assert negative_capture.candidate_response == expected_negative_response
    assert (
        baked_capture.action_grounding_source_case_id
        == promotion_record.case_id
    )
    assert (
        baked_capture.action_grounding_action_labels
        == promotion_record.intervention_ordering
    )
    assert (
        cold_capture.action_grounding_source_case_id
        != promotion_record.case_id
    )
    assert (
        cold_capture.action_grounding_action_labels
        != promotion_record.intervention_ordering
    )
    assert (
        negative_capture.action_grounding_source_case_id
        != promotion_record.case_id
    )
    assert (
        negative_capture.action_grounding_action_labels
        != promotion_record.intervention_ordering
    )
    assert baked_capture.source_state_unchanged is True
    assert cold_capture.source_state_unchanged is True
    assert baked_capture.source_state_digest_verified is True
    assert cold_capture.source_state_digest_verified is True
    assert negative_capture.source_state_unchanged is True
    assert negative_capture.source_state_digest_verified is True
    assert baked_capture.outcome_feedback_submitted is False
    assert cold_capture.outcome_feedback_submitted is False
    assert negative_capture.outcome_feedback_submitted is False
    assert baked_capture.evaluation_feedback_submitted is False
    assert cold_capture.evaluation_feedback_submitted is False
    assert negative_capture.evaluation_feedback_submitted is False
    assert len(abstraction_provider.action_applicability_prompts) == 2
    for applicability_prompt in (
        abstraction_provider.action_applicability_prompts
    ):
        applicability_payload = json.loads(
            applicability_prompt.split("Evidence:\n", 1)[1].split(
                "\n\nRequired output schema:",
                1,
            )[0]
        )
        assert set(applicability_payload) == {
            "candidate_schema_id",
            "current_situation_and_request",
            "required_applicability_conditions",
            "risk_markers",
        }
        assert all(
            step not in applicability_prompt
            for step in promotion_record.intervention_ordering
        )
        assert all(
            outcome_id not in applicability_prompt
            for outcome_id in promotion.source_outcome_ids
        )
    leaked_episode_text = (
        "Hú Xiānsheng",
        "胡青牛",
        "纪晓芙",
        chapter_11.scenes[0].canonical_outcome,
        chapter_17.scenes[0].canonical_outcome,
    )
    assert all(
        text not in baked_capture.candidate_response
        for text in leaked_episode_text
    )

    baked_report = review_behavior_fidelity(
        capture=baked_capture,
        reference=reference,
        assessment=_assessment(
            capture=baked_capture,
            reference=reference,
            scores=(0.92, 0.95, 0.85, 0.88, 0.92),
            source=BehaviorFidelityEvidenceSource.LLM_JUDGE,
        ),
    )
    cold_report = review_behavior_fidelity(
        capture=cold_capture,
        reference=reference,
        assessment=_assessment(
            capture=cold_capture,
            reference=reference,
            scores=(0.25, 0.45, 0.75, 0.25, 0.55),
            source=BehaviorFidelityEvidenceSource.LLM_JUDGE,
        ),
    )
    negative_report = review_behavior_fidelity(
        capture=negative_capture,
        reference=negative_reference,
        assessment=_assessment(
            capture=negative_capture,
            reference=negative_reference,
            scores=(0.92, 0.93, 0.90, 0.90, 0.94),
            source=BehaviorFidelityEvidenceSource.LLM_JUDGE,
        ),
    )
    comparison = compare_behavior_fidelity_reports(
        baked=baked_report,
        cold=cold_report,
        profile_answer_holdout_passed=True,
    )

    assert baked_report.overall_score == 0.904
    assert baked_report.behavior_fidelity_passed is True
    assert baked_report.claim_status == "llm_judge-diagnostic-pass"
    assert cold_report.overall_score == 0.45
    assert cold_report.behavior_fidelity_passed is False
    assert negative_report.overall_score == 0.918
    assert negative_report.behavior_fidelity_passed is True
    assert negative_report.claim_status == "llm_judge-diagnostic-pass"
    assert comparison.baked_minus_cold == 0.454
    assert comparison.learned_behavior_advantage is True
    assert comparison.claim_status == "diagnostic-pass"
