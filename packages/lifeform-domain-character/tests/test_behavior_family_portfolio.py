from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from lifeform_core import LifeformConfig
from lifeform_domain_character import (
    ActionEvidenceOnlyTextProvider,
    BehaviorFamilyPortfolioReport,
    BehaviorFamilyPromptKind,
    BehaviorFamilyRoutingObservation,
    BehaviorFidelityStimulus,
    ChapterLiveThroughDriver,
    build_character_lifeform,
    build_zhang_wuji_profile,
    capture_behavior_fidelity_async,
    evaluate_behavior_family_portfolio,
    evaluate_real_provider_behavior_evidence,
    read_ledger_json,
)
from lifeform_expression import GroundedResponseSynthesizer
from volvence_zero.application import (
    ApplicationCaseMemoryStore,
    CaseActionAbstractionPromotion,
    LearnedActionSchemaCandidate,
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
_BOUNDARY_SCHEMA = "withhold-disclosure-until-moral-clarity"
_PROTECTION_SCHEMA = "intervene-immediately-to-protect-life"


class _RecordingStructuredProvider:
    def __init__(self, *, fenced: bool = False) -> None:
        self.prompts: list[str] = []
        self._fenced = fenced

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int = 384,
        temperature: float = 0.0,
    ) -> str:
        del max_new_tokens, temperature
        self.prompts.append(prompt)
        payload = json.dumps({"structured": True})
        return f"```json\n{payload}\n```" if self._fenced else payload


class _ReviewedPortfolioProvider:
    """Exact-protocol fixture; temporal ownership chooses every family id."""

    def __init__(self) -> None:
        self._routing_expectations: dict[str, str] = {}
        self.decoded_family_ids: list[str] = []

    def register_route(
        self,
        *,
        stimulus: BehaviorFidelityStimulus,
        expected_schema_id: str,
    ) -> None:
        context = (
            f"{stimulus.setting_prompt}\n\n"
            f"{stimulus.decision_prompt}"
        )
        self._routing_expectations[context] = expected_schema_id

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int = 384,
        temperature: float = 0.0,
    ) -> str:
        del max_new_tokens, temperature
        if prompt.startswith(
            "You are the independent second-pass semantic generalization "
            "auditor for a"
        ):
            return json.dumps(
                {
                    "shared_structure_supported": True,
                    "episode_specificity_absent": True,
                    "conditions_reusable": True,
                    "steps_reusable": True,
                    "confidence": 0.95,
                    "rationale": "Frozen reviewed portfolio generalization.",
                }
            )
        if prompt.startswith(
            "You are the turn-time semantic applicability evaluator "
            "for a CaseMemory owner."
        ):
            payload = json.loads(
                prompt.split("Evidence:\n", 1)[1].split(
                    "\n\nRequired output schema:",
                    1,
                )[0]
            )
            expected = self._routing_expectations.get(
                payload["current_situation_and_request"]
            )
            return (
                json.dumps(
                    {
                        "applicable": (
                            payload["candidate_schema_id"] == expected
                        ),
                        "confidence": 0.95,
                        "rationale": (
                            "Frozen reviewed portfolio applicability."
                        ),
                    }
                )
                if expected is not None
                else "{}"
            )
        if not prompt.startswith(
            "You are the background-slow semantic decoder for a "
            "CaseMemory owner."
        ):
            return "{}"
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
        experiences = json.loads(
            prompt.split("Experiences:\n", 1)[1].split(
                "\n\nRequired output schema:",
                1,
            )[0]
        )
        source_outcomes = {
            item["outcome_id"] for item in experiences
        }
        boundary_outcomes = {
            item
            for item in source_outcomes
            if item.startswith(
                "chapter-replay:ch26-winehouse-identity-revelation:outcome:"
            )
            or item.startswith(
                "chapter-replay:ch-30-scene-2:outcome:"
            )
        }
        protection_outcomes = {
            item
            for item in source_outcomes
            if item.startswith(
                "chapter-replay:ch-11-scene-1:outcome:"
            )
            or item.startswith(
                "chapter-replay:ch-12-scene-4:outcome:"
            )
        }
        if source_outcomes == boundary_outcomes and len(source_outcomes) == 2:
            schema_id = _BOUNDARY_SCHEMA
            conditions = (
                "social pressure demands an immediate definitive disclosure",
                "an immediate answer would compromise moral autonomy or "
                "protected context",
            )
            steps = (
                "decline the premature definitive disclosure",
                "state the governing boundary without inventing certainty",
            )
        elif (
            source_outcomes == protection_outcomes
            and len(source_outcomes) == 2
        ):
            schema_id = _PROTECTION_SCHEMA
            conditions = (
                "a person's life or bodily safety faces immediate danger",
                "timely intervention can reduce the present harm",
            )
            steps = (
                "intervene immediately to protect the endangered person",
                "stabilize the immediate danger before seeking explanation",
            )
        else:
            return "{}"
        self.decoded_family_ids.append(family_id)
        return json.dumps(
            {
                "schema_id": schema_id,
                "action_family_id": family_id,
                "action_family_version": family_version,
                "applicability_conditions": conditions,
                "action_steps": steps,
                "source_outcome_ids": sorted(source_outcomes),
                "confidence": 0.92,
                "description": (
                    "Reviewed semantic compression of two independent "
                    "schema-held-out chapter outcomes."
                ),
            }
        )


def _held_out_profile():
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


def _config(*, application_dir: Path) -> LifeformConfig:
    return LifeformConfig(
        brain_config=BrainConfig(
            application_persistence_dir=str(application_dir),
            temporal_latent_dim=4,
            final_rollout_config=FinalRolloutConfig(
                internal_rl_runtime_replay=WiringLevel.ACTIVE,
                internal_rl_runtime_segment_credit=WiringLevel.ACTIVE,
                internal_rl_batch_accumulation_size=1,
                internal_rl_runtime_modulation_strength=0.3,
            ),
            rare_heavy_enabled=False,
        )
    )


def _case_store(*, application_dir: Path) -> ApplicationCaseMemoryStore:
    return ApplicationCaseMemoryStore(
        persistence_backend=build_filesystem_persistence_backend(
            base_dir=str(application_dir / "case_memory")
        )
    )


def _case_store_digest(*, application_dir: Path) -> str:
    store = _case_store(application_dir=application_dir)
    assert store.load_from_backend()
    return hashlib.sha256(repr(store.records).encode()).hexdigest()


def _clone_case_store(
    *,
    source_application_dir: Path,
    sandbox_application_dir: Path,
) -> None:
    source = _case_store(application_dir=source_application_dir)
    assert source.load_from_backend()
    sandbox = _case_store(application_dir=sandbox_application_dir)
    sandbox.restore_checkpoint(
        source.create_checkpoint(
            checkpoint_id="multi-family-routing-clone"
        )
    )
    assert sandbox.save_to_backend()


def _held_out_stimuli() -> tuple[BehaviorFidelityStimulus, ...]:
    return (
        BehaviorFidelityStimulus(
            case_id="behavior-family:boundary:sealed-route",
            character_id="zhang-wuji",
            scene_id="portfolio-boundary",
            phase_label="mature",
            setting=(
                "At an unfamiliar council, a former ally presses you to "
                "reveal another person's entrusted act and declare final "
                "judgment immediately."
            ),
            decision_point=(
                "The evidence remains incomplete and you are not authorized "
                "to disclose the confidence. Decide what you say now."
            ),
            evidence_locator="reviewed-portfolio:boundary:v1",
        ),
        BehaviorFidelityStimulus(
            case_id="behavior-family:protection:falling-beam",
            character_id="zhang-wuji",
            scene_id="portfolio-protection",
            phase_label="mature",
            setting=(
                "Inside an unfamiliar storehouse, a support beam starts "
                "falling toward an injured porter who cannot move away."
            ),
            decision_point=(
                "The beam will strike within seconds. Decide what concrete "
                "action you take before asking how the accident happened."
            ),
            evidence_locator="reviewed-portfolio:protection:v1",
        ),
    )


def test_real_chapters_promote_and_route_two_distinct_behavior_families(
    tmp_path: Path,
) -> None:
    ledger = read_ledger_json(_LEDGER)
    chapters = {
        chapter.chapter_id: chapter for chapter in ledger.chapters
    }
    selected_scenes = (
        ("ch-8", 0),
        ("ch-9", 0),
        ("ch-10", 2),
        ("ch-11", 0),
        ("ch-12", 3),
        ("ch-26", 1),
        ("ch-30", 1),
    )
    provider = _ReviewedPortfolioProvider()
    source_application_dir = tmp_path / "source-application"
    memory_backend = FileSystemPersistenceBackend(
        base_dir=str(tmp_path / "memory")
    )
    family_ids_by_scene: dict[str, tuple[str, ...]] = {}
    for ordinal, (chapter_id, scene_index) in enumerate(selected_scenes):
        bundle = build_character_lifeform(
            _held_out_profile(),
            config=_config(application_dir=source_application_dir),
            memory_store=build_default_memory_store(
                persistence_backend=memory_backend
            ),
            response_synthesizer=GroundedResponseSynthesizer(),
            semantic_proposal_runtime=LLMSemanticProposalRuntime(
                provider=provider
            ),
        )
        chapter = chapters[chapter_id]
        scene = replace(
            chapter.scenes[scene_index],
            canonical_action_schema=None,
        )
        report = ChapterLiveThroughDriver().run_ledger(
            ledger=replace(
                ledger,
                chapters=(
                    replace(
                        chapter,
                        scenes=(scene,),
                        semantic_events=(),
                    ),
                ),
            ),
            lifeform=bundle.lifeform,
            session_id=f"portfolio-{ordinal}-{chapter_id}",
        )
        report.require_success()
        family_ids_by_scene[scene.scene_id] = (
            report.per_scene_evidence[0].experienced_action_family_ids
        )

    store = _case_store(application_dir=source_application_dir)
    assert store.load_from_backend()
    promotion_records = tuple(
        record
        for record in store.records
        if record.action_abstraction_promotion is not None
    )
    promotions = tuple(
        record.action_abstraction_promotion
        for record in promotion_records
        if record.action_abstraction_promotion is not None
    )
    assert len(promotions) == 2, (
        family_ids_by_scene,
        provider.decoded_family_ids,
        tuple(
            (
                item.outcome_id,
                item.action_family_id,
            )
            for item in store.pending_action_abstraction_evidence()
        ),
    )
    assert len(set(provider.decoded_family_ids)) == 2
    assert family_ids_by_scene["ch-11-scene-1"] == (
        provider.decoded_family_ids[0],
    )
    assert family_ids_by_scene[
        "ch26-winehouse-identity-revelation"
    ] == (
        provider.decoded_family_ids[1],
    )

    stimuli = _held_out_stimuli()
    for stimulus, expected_schema in zip(
        stimuli,
        (_BOUNDARY_SCHEMA, _PROTECTION_SCHEMA),
        strict=True,
    ):
        provider.register_route(
            stimulus=stimulus,
            expected_schema_id=expected_schema,
        )
    source_digest = _case_store_digest(
        application_dir=source_application_dir
    )
    schema_by_case_id = {
        record.case_id: record.action_abstraction_promotion.schema_id
        for record in promotion_records
        if record.action_abstraction_promotion is not None
    }
    routing_observations = []
    for index, (stimulus, expected_schema) in enumerate(
        zip(
            stimuli,
            (_BOUNDARY_SCHEMA, _PROTECTION_SCHEMA),
            strict=True,
        )
    ):
        sandbox_dir = tmp_path / "routing-sandboxes" / str(index)
        _clone_case_store(
            source_application_dir=source_application_dir,
            sandbox_application_dir=sandbox_dir,
        )
        sandbox_bundle = build_character_lifeform(
            _held_out_profile(),
            config=_config(application_dir=sandbox_dir),
            memory_store=build_default_memory_store(),
            response_synthesizer=GroundedResponseSynthesizer(),
            semantic_proposal_runtime=LLMSemanticProposalRuntime(
                provider=provider
            ),
        )
        capture = asyncio.run(
            capture_behavior_fidelity_async(
                stimulus=stimulus,
                lifeform=sandbox_bundle.lifeform,
                arm_id="multi-family",
                source_state_sha256_before=source_digest,
                source_state_sha256_after=source_digest,
                source_state_digest_reader=lambda: _case_store_digest(
                    application_dir=source_application_dir
                ),
            )
        )
        routing_observations.append(
            BehaviorFamilyRoutingObservation(
                case_id=stimulus.case_id,
                expected_schema_id=expected_schema,
                selected_schema_id=(
                    schema_by_case_id.get(
                        capture.action_grounding_source_case_id
                    )
                    if capture.action_grounding_source_case_id is not None
                    else None
                ),
                source_state_digest_verified=(
                    capture.source_state_digest_verified
                ),
                outcome_feedback_submitted=(
                    capture.outcome_feedback_submitted
                ),
                evaluation_feedback_submitted=(
                    capture.evaluation_feedback_submitted
                ),
            )
        )

    portfolio = evaluate_behavior_family_portfolio(
        suite_id="zhang-wuji-two-family-longitudinal-v1",
        expected_schema_ids=(_BOUNDARY_SCHEMA, _PROTECTION_SCHEMA),
        promotions=promotions,
        routing_observations=tuple(routing_observations),
        pending_family_ids=tuple(
            sorted(
                {
                    item.action_family_id
                    for item in (
                        store.pending_action_abstraction_evidence()
                    )
                }
            )
        ),
    )

    assert portfolio.promoted_family_ids[0] != (
        portfolio.promoted_family_ids[1]
    )
    assert portfolio.correct_routing_count == 2
    assert portfolio.pending_family_count == 2
    assert portfolio.multi_family_owner_supported is True
    assert portfolio.claim_status == "multi-family-owner-diagnostic-pass"
    assert all(status == "pass" for _gate, status in portfolio.gate_statuses)


def test_action_evidence_provider_delegates_only_exact_action_protocols() -> None:
    provider = _RecordingStructuredProvider()
    scoped = ActionEvidenceOnlyTextProvider(
        provider=provider,
        provider_id="local:test-model",
    )

    abstraction_response = scoped.generate(
        prompt=(
            "You are the background-slow semantic decoder for a "
            "CaseMemory owner.\nEvidence"
        )
    )
    applicability_response = scoped.generate(
        prompt=(
            "You are the turn-time semantic applicability evaluator "
            "for a CaseMemory owner.\nEvidence"
        )
    )
    generalization_response = scoped.generate(
        prompt=(
            "You are the independent second-pass semantic generalization "
            "auditor for a CaseMemory owner.\nEvidence"
        )
    )
    excluded_response = scoped.generate(
        prompt="You classify a user's message in a multi-turn dialogue."
    )

    assert json.loads(abstraction_response) == {"structured": True}
    assert json.loads(applicability_response) == {"structured": True}
    assert json.loads(generalization_response) == {"structured": True}
    assert excluded_response == "{}"
    assert len(provider.prompts) == 3
    assert tuple(trace.prompt_kind for trace in scoped.traces) == (
        BehaviorFamilyPromptKind.ACTION_ABSTRACTION,
        BehaviorFamilyPromptKind.ACTION_APPLICABILITY,
        BehaviorFamilyPromptKind.ACTION_GENERALIZATION_AUDIT,
        BehaviorFamilyPromptKind.EXCLUDED_OTHER,
    )
    assert tuple(trace.delegated_to_provider for trace in scoped.traces) == (
        True,
        True,
        True,
        False,
    )


def test_owner_portfolio_rejects_legacy_promotions_without_audit() -> None:
    promotions = tuple(
        CaseActionAbstractionPromotion(
            schema_id=f"schema-{suffix}",
            action_family_id=f"family-{suffix}",
            action_family_version=index,
            source_outcome_ids=(
                f"outcome-{suffix}-1",
                f"outcome-{suffix}-2",
            ),
            applicability_conditions=("a reusable condition is observed",),
        )
        for index, suffix in enumerate(("a", "b"), start=1)
    )

    report = evaluate_behavior_family_portfolio(
        suite_id="legacy-promotion-audit-probe",
        expected_schema_ids=("schema-a", "schema-b"),
        promotions=promotions,
        routing_observations=(),
        pending_family_ids=(),
    )

    assert report.generalization_audited_promotion_count == 0
    assert (
        dict(report.gate_statuses)["promotions_generalization_audited"]
        == "fail"
    )
    assert report.multi_family_owner_supported is False


def test_real_provider_report_requires_scoped_outputs_and_owner_pass() -> None:
    provider = _RecordingStructuredProvider()
    scoped = ActionEvidenceOnlyTextProvider(
        provider=provider,
        provider_id="local:test-model",
    )
    for prompt in (
        (
            "You are the background-slow semantic decoder for a "
            "CaseMemory owner."
        ),
        (
            "You are the background-slow semantic decoder for a "
            "CaseMemory owner.\nsecond family"
        ),
        (
            "You are the turn-time semantic applicability evaluator "
            "for a CaseMemory owner."
        ),
        (
            "You are the turn-time semantic applicability evaluator "
            "for a CaseMemory owner.\nsecond route"
        ),
        (
            "You are the independent second-pass semantic generalization "
            "auditor for a CaseMemory owner."
        ),
        (
            "You are the independent second-pass semantic generalization "
            "auditor for a CaseMemory owner.\nsecond family"
        ),
        "unrelated semantic owner protocol",
    ):
        scoped.generate(prompt=prompt)
    portfolio = BehaviorFamilyPortfolioReport(
        suite_id="test-suite",
        expected_schema_ids=("schema-a", "schema-b"),
        promoted_schema_ids=("schema-a", "schema-b"),
        promoted_family_ids=("family-a", "family-b"),
        promotion_count=2,
        generalization_audited_promotion_count=2,
        distinct_family_count=2,
        pending_family_count=0,
        routing_case_count=2,
        correct_routing_count=2,
        gate_statuses=(("owner", "pass"),),
        multi_family_owner_supported=True,
        claim_status="multi-family-owner-diagnostic-pass",
        description="test",
    )

    report = evaluate_real_provider_behavior_evidence(
        provider_id="local:test-model",
        traces=scoped.traces,
        portfolio=portfolio,
    )

    assert report.real_provider_supported is True
    assert report.claim_status == "real-structured-provider-diagnostic-pass"
    assert report.delegated_abstraction_call_count == 2
    assert report.delegated_generalization_audit_call_count == 2
    assert report.delegated_applicability_call_count == 2
    assert report.excluded_other_call_count == 1
    assert all(status == "pass" for _gate, status in report.gate_statuses)


def test_real_provider_report_rejects_zero_promotion_vacuous_pass() -> None:
    provider = _RecordingStructuredProvider(fenced=True)
    scoped = ActionEvidenceOnlyTextProvider(
        provider=provider,
        provider_id="local:under-capacity-model",
    )
    scoped.generate(
        prompt=(
            "You are the background-slow semantic decoder for a "
            "CaseMemory owner."
        )
    )
    failed_portfolio = BehaviorFamilyPortfolioReport(
        suite_id="failed-provider-suite",
        expected_schema_ids=("schema-a", "schema-b"),
        promoted_schema_ids=(),
        promoted_family_ids=(),
        promotion_count=0,
        generalization_audited_promotion_count=0,
        distinct_family_count=0,
        pending_family_count=2,
        routing_case_count=0,
        correct_routing_count=0,
        gate_statuses=(("owner", "fail"),),
        multi_family_owner_supported=False,
        claim_status="diagnostic-fail",
        description="test",
    )

    report = evaluate_real_provider_behavior_evidence(
        provider_id="local:under-capacity-model",
        traces=scoped.traces,
        portfolio=failed_portfolio,
    )

    assert report.real_provider_supported is False
    assert report.claim_status == "diagnostic-fail"
    assert dict(report.gate_statuses) == {
        "action_protocol_scope": "pass",
        "structured_provider_outputs": "pass",
        "provider_abstraction_consumed": "fail",
        "provider_generalization_audit_consumed": "fail",
        "provider_applicability_consumed": "fail",
        "owner_portfolio_pass": "fail",
    }


def test_action_schema_candidate_rejects_non_kebab_identifier() -> None:
    with pytest.raises(
        ValueError,
        match="schema_id must be a lowercase kebab-case identifier",
    ):
        LearnedActionSchemaCandidate(
            schema_id="confront_and_treat_poison",
            action_family_id="discovered_family_5",
            action_family_version=180,
            applicability_conditions=("immediate danger is observable",),
            action_steps=("intervene to reduce present harm",),
            source_outcome_ids=("outcome-1", "outcome-2"),
            confidence=0.95,
            description="Invalid provider output.",
        )
