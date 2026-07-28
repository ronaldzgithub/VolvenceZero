from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from lifeform_core import LifeformConfig
from lifeform_domain_character import (
    BehaviorFidelityAblationArm,
    BehaviorFidelityCaseKind,
    BehaviorFidelityCaseObservation,
    ChapterLiveThroughDriver,
    ChapterLiveThroughReport,
    PromotionExpectation,
    build_character_lifeform,
    build_zhang_wuji_profile,
    capture_behavior_fidelity_async,
    evaluate_behavior_fidelity_ablation,
    load_zhang_wuji_action_applicability_matrix,
    read_ledger_json,
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


class _FrozenMatrixProvider:
    """Reviewed exact-protocol fixture shared byte-identically by every arm."""

    def __init__(self) -> None:
        self._applicability: dict[str, bool] = {}
        self.action_abstraction_call_count = 0

    def register_matrix(self) -> None:
        matrix = load_zhang_wuji_action_applicability_matrix()
        for case in matrix.cases:
            context = (
                f"{case.stimulus.setting_prompt}\n\n"
                f"{case.stimulus.decision_prompt}"
            )
            self._applicability[context] = (
                case.promotion_expectation
                is PromotionExpectation.REQUIRED
            )

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
            evidence = json.loads(
                prompt.split("Evidence:\n", 1)[1].split(
                    "\n\nRequired output schema:",
                    1,
                )[0]
            )
            context = evidence["current_situation_and_request"]
            if context not in self._applicability:
                return "{}"
            return json.dumps(
                {
                    "applicable": self._applicability[context],
                    "confidence": 0.94,
                    "rationale": "Frozen reviewed matrix applicability.",
                }
            )
        if not prompt.startswith(
            "You are the background-slow semantic decoder for a "
            "CaseMemory owner."
        ):
            return "{}"
        self.action_abstraction_call_count += 1
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
                    item["outcome_id"] for item in experiences
                ],
                "confidence": 0.91,
                "description": (
                    "Frozen reviewed semantic compression for causal "
                    "ablation."
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


def _config(
    *,
    application_dir: Path,
    internal_rl_level: WiringLevel,
) -> LifeformConfig:
    return LifeformConfig(
        brain_config=BrainConfig(
            application_persistence_dir=str(application_dir),
            final_rollout_config=FinalRolloutConfig(
                internal_rl_runtime_replay=internal_rl_level,
                internal_rl_runtime_segment_credit=internal_rl_level,
                internal_rl_batch_accumulation_size=1,
                internal_rl_runtime_modulation_strength=0.3,
            ),
            rare_heavy_enabled=False,
        )
    )


def _schema_held_out_chapters():
    ledger = read_ledger_json(_LEDGER)
    chapters = {
        chapter.chapter_id: chapter for chapter in ledger.chapters
    }
    return (
        ledger,
        replace(
            chapters["ch-11"],
            scenes=(
                replace(
                    chapters["ch-11"].scenes[0],
                    canonical_action_schema=None,
                ),
            ),
            semantic_events=(),
        ),
        replace(
            chapters["ch-17"],
            scenes=(
                replace(
                    chapters["ch-17"].scenes[0],
                    canonical_action_schema=None,
                ),
            ),
            semantic_events=(),
        ),
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
        source.create_checkpoint(checkpoint_id="matrix-ablation-clone")
    )
    assert sandbox.save_to_backend()


def _shuffle_pending_family(*, application_dir: Path) -> None:
    store = _case_store(application_dir=application_dir)
    assert store.load_from_backend()
    shuffled_count = 0
    records = []
    for record in store.records:
        evidence = record.action_abstraction_evidence
        if evidence is None:
            records.append(record)
            continue
        shuffled_count += 1
        records.append(
            replace(
                record,
                action_abstraction_evidence=replace(
                    evidence,
                    action_family_id=(
                        f"ablation_shuffled_{evidence.action_family_id}"
                    ),
                ),
            )
        )
    assert shuffled_count == 1
    checkpoint = store.create_checkpoint(
        checkpoint_id="pre-shuffle-lineage"
    )
    store.restore_checkpoint(
        replace(
            checkpoint,
            checkpoint_id="shuffled-lineage",
            records=tuple(records),
        )
    )
    assert store.save_to_backend()


def _bake_arm(
    *,
    application_dir: Path,
    memory_dir: Path,
    internal_rl_level: WiringLevel,
    shuffle_after_first_chapter: bool,
    provider: _FrozenMatrixProvider,
) -> tuple[str, ...]:
    ledger, chapter_11, chapter_17 = _schema_held_out_chapters()
    profile = _held_out_profile()
    config = _config(
        application_dir=application_dir,
        internal_rl_level=internal_rl_level,
    )
    memory_backend = FileSystemPersistenceBackend(
        base_dir=str(memory_dir)
    )
    first = build_character_lifeform(
        profile,
        config=config,
        memory_store=build_default_memory_store(
            persistence_backend=memory_backend
        ),
        response_synthesizer=GroundedResponseSynthesizer(),
        semantic_proposal_runtime=LLMSemanticProposalRuntime(
            provider=provider
        ),
    )
    first_report = ChapterLiveThroughDriver().run_ledger(
        ledger=replace(ledger, chapters=(chapter_11,)),
        lifeform=first.lifeform,
        session_id=f"{application_dir.name}-ch-11",
    )
    _assert_expected_bake_status(
        report=first_report,
        internal_rl_level=internal_rl_level,
    )
    first_store = _case_store(application_dir=application_dir)
    assert first_store.load_from_backend()
    pending_after_first = (
        first_store.pending_action_abstraction_evidence()
    )
    if internal_rl_level is WiringLevel.ACTIVE:
        assert len(pending_after_first) == 1
        assert pending_after_first[0].learning_lineage is not None
        assert pending_after_first[0].learning_lineage.admission_ready
    else:
        assert pending_after_first == ()
    if shuffle_after_first_chapter:
        _shuffle_pending_family(application_dir=application_dir)
    second = build_character_lifeform(
        profile,
        config=config,
        memory_store=build_default_memory_store(
            persistence_backend=memory_backend
        ),
        response_synthesizer=GroundedResponseSynthesizer(),
        semantic_proposal_runtime=LLMSemanticProposalRuntime(
            provider=provider
        ),
    )
    second_report = ChapterLiveThroughDriver().run_ledger(
        ledger=replace(ledger, chapters=(chapter_17,)),
        lifeform=second.lifeform,
        session_id=f"{application_dir.name}-ch-17",
    )
    _assert_expected_bake_status(
        report=second_report,
        internal_rl_level=internal_rl_level,
    )
    store = _case_store(application_dir=application_dir)
    assert store.load_from_backend()
    return tuple(
        record.case_id
        for record in store.records
        if record.action_abstraction_promotion is not None
        and record.action_abstraction_promotion.schema_id
        == "intervene-to-stop-imminent-third-party-harm"
    )


def _assert_expected_bake_status(
    *,
    report: ChapterLiveThroughReport,
    internal_rl_level: WiringLevel,
) -> None:
    if internal_rl_level is WiringLevel.ACTIVE:
        report.require_success()
        return
    assert report.bake_succeeded is False
    assert any(
        "internal-rl-runtime-replay-not-active" in failure
        for failure in report.verification_failures
    )
    assert any(
        "internal-rl-did-not-consume-runtime-replay" in failure
        for failure in report.verification_failures
    )


async def _capture_persisted_arm(
    *,
    arm: BehaviorFidelityAblationArm,
    source_application_dir: Path,
    sandbox_root: Path,
    promotion_case_ids: tuple[str, ...],
    provider: _FrozenMatrixProvider,
) -> tuple[BehaviorFidelityCaseObservation, ...]:
    matrix = load_zhang_wuji_action_applicability_matrix()
    profile = _held_out_profile()
    source_digest = _case_store_digest(
        application_dir=source_application_dir
    )
    observations = []
    for index, case in enumerate(matrix.cases):
        sandbox_dir = sandbox_root / f"{index:02d}"
        _clone_case_store(
            source_application_dir=source_application_dir,
            sandbox_application_dir=sandbox_dir,
        )
        bundle = build_character_lifeform(
            profile,
            config=_config(
                application_dir=sandbox_dir,
                internal_rl_level=WiringLevel.ACTIVE,
            ),
            memory_store=build_default_memory_store(),
            response_synthesizer=GroundedResponseSynthesizer(),
            semantic_proposal_runtime=LLMSemanticProposalRuntime(
                provider=provider
            ),
        )
        capture = await capture_behavior_fidelity_async(
            stimulus=case.stimulus,
            lifeform=bundle.lifeform,
            arm_id=arm.value,
            source_state_sha256_before=source_digest,
            source_state_sha256_after=source_digest,
            source_state_digest_reader=lambda: _case_store_digest(
                application_dir=source_application_dir
            ),
        )
        observations.append(
            BehaviorFidelityCaseObservation(
                suite_digest=matrix.digest,
                arm=arm,
                case_id=case.stimulus.case_id,
                kind=case.kind,
                promotion_expectation=case.promotion_expectation,
                target_promotion_used=(
                    capture.action_grounding_source_case_id
                    in promotion_case_ids
                ),
                action_grounding_source_case_id=(
                    capture.action_grounding_source_case_id
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
    return tuple(observations)


async def _capture_cold_arm(
    *,
    sandbox_root: Path,
    provider: _FrozenMatrixProvider,
) -> tuple[BehaviorFidelityCaseObservation, ...]:
    matrix = load_zhang_wuji_action_applicability_matrix()
    profile = _held_out_profile()
    source_digest = hashlib.sha256(repr(profile).encode()).hexdigest()
    observations = []
    for index, case in enumerate(matrix.cases):
        bundle = build_character_lifeform(
            profile,
            config=_config(
                application_dir=sandbox_root / f"{index:02d}",
                internal_rl_level=WiringLevel.ACTIVE,
            ),
            memory_store=build_default_memory_store(),
            response_synthesizer=GroundedResponseSynthesizer(),
            semantic_proposal_runtime=LLMSemanticProposalRuntime(
                provider=provider
            ),
        )
        capture = await capture_behavior_fidelity_async(
            stimulus=case.stimulus,
            lifeform=bundle.lifeform,
            arm_id=BehaviorFidelityAblationArm.COLD.value,
            source_state_sha256_before=source_digest,
            source_state_sha256_after=source_digest,
            source_state_digest_reader=lambda: hashlib.sha256(
                repr(profile).encode()
            ).hexdigest(),
        )
        observations.append(
            BehaviorFidelityCaseObservation(
                suite_digest=matrix.digest,
                arm=BehaviorFidelityAblationArm.COLD,
                case_id=case.stimulus.case_id,
                kind=case.kind,
                promotion_expectation=case.promotion_expectation,
                target_promotion_used=False,
                action_grounding_source_case_id=(
                    capture.action_grounding_source_case_id
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
    return tuple(observations)


def _synthetic_observations(
    *,
    no_rl_uses_promotion: bool = False,
) -> tuple[BehaviorFidelityCaseObservation, ...]:
    matrix = load_zhang_wuji_action_applicability_matrix()
    observations = []
    for arm in BehaviorFidelityAblationArm:
        for case in matrix.cases:
            positive = case.kind is BehaviorFidelityCaseKind.POSITIVE
            uses_promotion = positive and (
                arm is BehaviorFidelityAblationArm.BAKED
                or (
                    no_rl_uses_promotion
                    and arm is BehaviorFidelityAblationArm.NO_RL
                )
            )
            observations.append(
                BehaviorFidelityCaseObservation(
                    suite_digest=matrix.digest,
                    arm=arm,
                    case_id=case.stimulus.case_id,
                    kind=case.kind,
                    promotion_expectation=case.promotion_expectation,
                    target_promotion_used=uses_promotion,
                    action_grounding_source_case_id=(
                        "case:promotion" if uses_promotion else "case:other"
                    ),
                    source_state_digest_verified=True,
                    outcome_feedback_submitted=False,
                    evaluation_feedback_submitted=False,
                    fidelity_score=(
                        0.9
                        if arm is BehaviorFidelityAblationArm.BAKED
                        else 0.6
                    ),
                    competing_behavior_family_matched=(
                        arm is BehaviorFidelityAblationArm.BAKED
                        if case.kind
                        is BehaviorFidelityCaseKind.COMPETING_BEHAVIOR
                        else None
                    ),
                )
            )
    return tuple(observations)


def test_ablation_report_separates_lineage_and_behavior_claims() -> None:
    matrix = load_zhang_wuji_action_applicability_matrix()

    report = evaluate_behavior_fidelity_ablation(
        matrix=matrix,
        observations=_synthetic_observations(),
    )

    assert report.lineage_causal_supported is True
    assert report.behavior_causal_supported is True
    assert report.claim_status == "behavior-causal-diagnostic-pass"
    by_arm = {item.arm: item for item in report.arm_reports}
    baked = by_arm[BehaviorFidelityAblationArm.BAKED]
    cold = by_arm[BehaviorFidelityAblationArm.COLD]
    assert (
        baked.true_positive_count,
        baked.false_positive_count,
        baked.false_negative_count,
        baked.true_negative_count,
    ) == (4, 0, 0, 12)
    assert baked.promotion_precision == 1.0
    assert baked.promotion_recall == 1.0
    assert baked.promotion_specificity == 1.0
    assert cold.promotion_precision is None
    assert cold.promotion_recall == 0.0
    assert cold.promotion_specificity == 1.0

    confounded = evaluate_behavior_fidelity_ablation(
        matrix=matrix,
        observations=_synthetic_observations(
            no_rl_uses_promotion=True,
        ),
    )
    assert confounded.lineage_causal_supported is False
    assert confounded.behavior_causal_supported is False
    assert confounded.claim_status == "diagnostic-fail"


def test_ablation_report_fails_closed_on_coverage_and_digest_mismatch() -> None:
    matrix = load_zhang_wuji_action_applicability_matrix()
    observations = _synthetic_observations()

    with pytest.raises(ValueError, match="coverage mismatch"):
        evaluate_behavior_fidelity_ablation(
            matrix=matrix,
            observations=observations[:-1],
        )

    with pytest.raises(ValueError, match="suite digest mismatch"):
        evaluate_behavior_fidelity_ablation(
            matrix=matrix,
            observations=(
                replace(observations[0], suite_digest="0" * 64),
                *observations[1:],
            ),
        )


def test_ablation_report_publishes_false_positive_and_false_negative_rates() -> None:
    matrix = load_zhang_wuji_action_applicability_matrix()
    observations = list(_synthetic_observations())
    baked_positive_index = next(
        index
        for index, observation in enumerate(observations)
        if observation.arm is BehaviorFidelityAblationArm.BAKED
        and observation.kind is BehaviorFidelityCaseKind.POSITIVE
    )
    baked_negative_index = next(
        index
        for index, observation in enumerate(observations)
        if observation.arm is BehaviorFidelityAblationArm.BAKED
        and observation.kind is BehaviorFidelityCaseKind.NEAR_NEGATIVE
    )
    observations[baked_positive_index] = replace(
        observations[baked_positive_index],
        target_promotion_used=False,
        action_grounding_source_case_id="case:other",
    )
    observations[baked_negative_index] = replace(
        observations[baked_negative_index],
        target_promotion_used=True,
        action_grounding_source_case_id="case:promotion",
    )

    report = evaluate_behavior_fidelity_ablation(
        matrix=matrix,
        observations=tuple(observations),
    )
    baked = next(
        item
        for item in report.arm_reports
        if item.arm is BehaviorFidelityAblationArm.BAKED
    )

    assert (
        baked.true_positive_count,
        baked.false_positive_count,
        baked.false_negative_count,
        baked.true_negative_count,
    ) == (3, 1, 1, 11)
    assert baked.promotion_precision == 0.75
    assert baked.promotion_recall == 0.75
    assert baked.promotion_specificity == pytest.approx(11 / 12, abs=1e-6)
    assert report.lineage_causal_supported is False
    assert report.claim_status == "diagnostic-fail"


def test_frozen_matrix_four_arm_runtime_ablation(tmp_path: Path) -> None:
    matrix = load_zhang_wuji_action_applicability_matrix()
    provider = _FrozenMatrixProvider()
    provider.register_matrix()
    baked_dir = tmp_path / "baked-source"
    no_rl_dir = tmp_path / "no-rl-source"
    shuffled_dir = tmp_path / "shuffled-source"

    baked_promotions = _bake_arm(
        application_dir=baked_dir,
        memory_dir=tmp_path / "baked-memory",
        internal_rl_level=WiringLevel.ACTIVE,
        shuffle_after_first_chapter=False,
        provider=provider,
    )
    no_rl_promotions = _bake_arm(
        application_dir=no_rl_dir,
        memory_dir=tmp_path / "no-rl-memory",
        internal_rl_level=WiringLevel.DISABLED,
        shuffle_after_first_chapter=False,
        provider=provider,
    )
    shuffled_promotions = _bake_arm(
        application_dir=shuffled_dir,
        memory_dir=tmp_path / "shuffled-memory",
        internal_rl_level=WiringLevel.ACTIVE,
        shuffle_after_first_chapter=True,
        provider=provider,
    )
    observations = asyncio.run(
        _capture_all_arms(
            baked_dir=baked_dir,
            no_rl_dir=no_rl_dir,
            shuffled_dir=shuffled_dir,
            tmp_path=tmp_path,
            baked_promotions=baked_promotions,
            no_rl_promotions=no_rl_promotions,
            shuffled_promotions=shuffled_promotions,
            provider=provider,
        )
    )

    report = evaluate_behavior_fidelity_ablation(
        matrix=matrix,
        observations=observations,
    )

    assert len(baked_promotions) == 1
    assert no_rl_promotions == ()
    assert shuffled_promotions == ()
    assert provider.action_abstraction_call_count == 1
    by_arm = {arm_report.arm: arm_report for arm_report in report.arm_reports}
    assert (
        by_arm[
            BehaviorFidelityAblationArm.BAKED
        ].positive_promotion_hits
        == 4
    )
    assert (
        by_arm[BehaviorFidelityAblationArm.BAKED].promotion_precision
        == 1.0
    )
    assert (
        by_arm[BehaviorFidelityAblationArm.BAKED].promotion_recall
        == 1.0
    )
    assert (
        by_arm[BehaviorFidelityAblationArm.BAKED].promotion_specificity
        == 1.0
    )
    assert (
        by_arm[
            BehaviorFidelityAblationArm.NO_RL
        ].positive_promotion_hits
        == 0
    )
    assert (
        by_arm[
            BehaviorFidelityAblationArm.COLD
        ].positive_promotion_hits
        == 0
    )
    assert (
        by_arm[
            BehaviorFidelityAblationArm.SHUFFLED_LINEAGE
        ].positive_promotion_hits
        == 0
    )
    assert all(
        arm_report.non_positive_promotion_hits == 0
        and arm_report.source_integrity_passed
        and arm_report.no_feedback_passed
        for arm_report in report.arm_reports
    )
    gates = dict(report.causal_gate_statuses)
    assert gates["no_rl_target_promotion_absent"] == "pass"
    assert gates["reviewed_behavior_evidence_complete"] == (
        "insufficient_data"
    )
    assert report.lineage_causal_supported is True
    assert report.behavior_causal_supported is False
    assert report.claim_status == "lineage-causal-diagnostic-pass"


async def _capture_all_arms(
    *,
    baked_dir: Path,
    no_rl_dir: Path,
    shuffled_dir: Path,
    tmp_path: Path,
    baked_promotions: tuple[str, ...],
    no_rl_promotions: tuple[str, ...],
    shuffled_promotions: tuple[str, ...],
    provider: _FrozenMatrixProvider,
) -> tuple[BehaviorFidelityCaseObservation, ...]:
    baked = await _capture_persisted_arm(
        arm=BehaviorFidelityAblationArm.BAKED,
        source_application_dir=baked_dir,
        sandbox_root=tmp_path / "baked-eval",
        promotion_case_ids=baked_promotions,
        provider=provider,
    )
    cold = await _capture_cold_arm(
        sandbox_root=tmp_path / "cold-eval",
        provider=provider,
    )
    no_rl = await _capture_persisted_arm(
        arm=BehaviorFidelityAblationArm.NO_RL,
        source_application_dir=no_rl_dir,
        sandbox_root=tmp_path / "no-rl-eval",
        promotion_case_ids=no_rl_promotions,
        provider=provider,
    )
    shuffled = await _capture_persisted_arm(
        arm=BehaviorFidelityAblationArm.SHUFFLED_LINEAGE,
        source_application_dir=shuffled_dir,
        sandbox_root=tmp_path / "shuffled-eval",
        promotion_case_ids=shuffled_promotions,
        provider=provider,
    )
    return baked + cold + no_rl + shuffled
