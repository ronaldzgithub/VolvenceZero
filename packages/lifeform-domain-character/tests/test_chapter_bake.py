from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from lifeform_core import LifeformConfig
from lifeform_domain_character import (
    ChapterLiveThroughDriver,
    build_character_lifeform,
    build_zhang_wuji_profile,
    read_ledger_json,
)
from volvence_zero.agent import decode_agent_learning_archive
from volvence_zero.application import (
    ApplicationCaseMemoryStore,
    build_filesystem_persistence_backend,
)
from volvence_zero.brain import BrainConfig
from volvence_zero.canonical_json import typed_to_json
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
_REVIEWED_LEDGER = (
    _REPO_ROOT
    / "artifacts"
    / "character-live-through"
    / "zhang_wuji.reviewed_ledger.json"
)


def _bake_config() -> LifeformConfig:
    return LifeformConfig(
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


class _ReviewedCrossChapterAbstractionProvider:
    """Deterministic reviewer stand-in for the structured decoder only."""

    def __init__(self) -> None:
        self.action_abstraction_prompts: list[str] = []

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
                    "rationale": "Reviewed cross-chapter generalization.",
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
        evidence_json = prompt.split("Experiences:\n", 1)[1].split(
            "\n\nRequired output schema:",
            1,
        )[0]
        experiences = json.loads(evidence_json)
        return json.dumps(
            {
                "schema_id": "intervene-to-stop-imminent-third-party-harm",
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
                    "Reviewed semantic compression of two real, "
                    "schema-held-out chapter experiences."
                ),
            },
            ensure_ascii=False,
        )


def test_zhang_wuji_chapter_12_live_through_reaches_owners_and_internal_rl(
    tmp_path: Path,
) -> None:
    full_ledger = read_ledger_json(_REVIEWED_LEDGER)
    chapter = next(
        item for item in full_ledger.chapters if item.chapter_id == "ch-11"
    )
    assert chapter.chapter_title == "十二　针其膏兮药其育"
    assert len(chapter.scenes) == 1
    ledger = replace(full_ledger, chapters=(chapter,))

    backend = FileSystemPersistenceBackend(base_dir=str(tmp_path / "state"))
    memory_store = build_default_memory_store(persistence_backend=backend)
    bundle = build_character_lifeform(
        build_zhang_wuji_profile(),
        config=_bake_config(),
        memory_store=memory_store,
    )

    report = ChapterLiveThroughDriver().run_ledger(
        ledger=ledger,
        lifeform=bundle.lifeform,
        session_id="zhang-wuji-chapter-12-proof",
        progress_path=tmp_path / "progress.jsonl",
    )

    report.require_success()
    assert report.bake_succeeded is True
    assert dict(report.proof_gates) == {
        "full_chapter_coverage": "pass",
        "semantic_events_reached_declared_owners_after_outcome": "pass",
        "canonical_outcomes_settled_by_prediction_error": "pass",
        "runtime_replay_lineage_reached_internal_rl": "pass",
        "internal_rl_updated_z_policy": "pass",
        "background_slow_memory_and_policy_integration": "pass",
    }
    chapter_record = report.per_chapter[0]
    assert chapter_record.semantic_events_verified == 3
    assert set(chapter_record.semantic_owner_slots_verified) == {
        "belief_assumption",
        "relationship_state",
        "user_model",
    }
    assert chapter_record.semantic_events_applied_after_outcome is True

    evidence = report.per_scene_evidence[0]
    assert evidence.evaluated_prediction_id == evidence.prediction_id
    assert evidence.pe_environment_outcome_id == evidence.environment_outcome_id
    assert evidence.world_z_t
    assert evidence.self_z_t
    assert evidence.runtime_replay_wiring == "active"
    assert evidence.runtime_settled_delta == 2
    assert evidence.runtime_transition_delta == 2
    assert evidence.runtime_lineage_match_delta == 2
    assert evidence.internal_rl_backend == "runtime-replay"
    assert evidence.internal_rl_policy_update_applied is True
    assert evidence.internal_rl_policy_epochs > 0
    assert evidence.internal_rl_rollback_reasons == ()
    assert evidence.memory_entry_delta > 0
    assert evidence.delayed_credit_published is True
    assert any(
        operation.startswith("temporal-prior:")
        for operation in evidence.slow_loop_applied_operations
    )
    target_prefixes = {
        target.split(".", 2)[0] + "." + target.split(".", 2)[1]
        for target in evidence.application_owner_targets_updated
    }
    assert {
        "application.boundary_policy",
        "application.case_memory",
        "application.domain_knowledge",
        "application.strategy_playbook",
    } <= target_prefixes

    persisted = backend.load_checkpoint(
        key="owner_hydration/joint_loop.learning"
    )
    assert persisted is not None
    payload = json.loads(persisted[0].decode("utf-8"))
    assert payload["owner_name"] == "joint_loop.learning"
    assert payload["description"].endswith("runtime_replay=excluded")

    reborn_bundle = build_character_lifeform(
        build_zhang_wuji_profile(),
        config=_bake_config(),
        memory_store=build_default_memory_store(persistence_backend=backend),
    )
    reborn_session = reborn_bundle.lifeform.create_session(
        session_id="zhang-wuji-chapter-12-rebirth-proof"
    )
    restored_archive = decode_agent_learning_archive(
        reborn_session.brain_session.runner.export_learning_checkpoint_archive(
            checkpoint_id="zhang-wuji-chapter-12-rebirth-proof",
            include_runtime_replay=False,
        )
    )
    restored_joint_loop = next(
        snapshot
        for snapshot in restored_archive.owner_snapshots
        if snapshot.owner_name == "joint_loop.learning"
    )
    assert (
        typed_to_json(restored_joint_loop.payload, Mapping[str, Any])
        == payload["payload"]
    )


def test_chapter_bake_fails_when_runtime_replay_is_disabled(
    tmp_path: Path,
) -> None:
    full_ledger = read_ledger_json(_REVIEWED_LEDGER)
    chapter = next(
        item for item in full_ledger.chapters if item.chapter_id == "ch-11"
    )
    ledger = replace(full_ledger, chapters=(chapter,))
    backend = FileSystemPersistenceBackend(base_dir=str(tmp_path / "state"))
    bundle = build_character_lifeform(
        build_zhang_wuji_profile(),
        memory_store=build_default_memory_store(persistence_backend=backend),
    )

    report = ChapterLiveThroughDriver().run_ledger(
        ledger=ledger,
        lifeform=bundle.lifeform,
        session_id="zhang-wuji-disabled-runtime-replay",
    )

    assert report.bake_succeeded is False
    assert dict(report.proof_gates)[
        "runtime_replay_lineage_reached_internal_rl"
    ] == "fail"
    assert dict(report.proof_gates)["internal_rl_updated_z_policy"] == "fail"
    with pytest.raises(
        RuntimeError,
        match="chapter live-through bake did not satisfy proof gates",
    ):
        report.require_success()


def test_real_cross_chapter_schema_holdout_promotes_natural_family(
    tmp_path: Path,
) -> None:
    """Two real scenes must promote only through owner-issued family identity."""

    full_ledger = read_ledger_json(_REVIEWED_LEDGER)
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
    application_dir = tmp_path / "application-owners"
    memory_backend = FileSystemPersistenceBackend(
        base_dir=str(tmp_path / "memory")
    )
    config = replace(
        _bake_config(),
        brain_config=replace(
            _bake_config().brain_config,
            application_persistence_dir=str(application_dir),
        ),
    )
    abstraction_provider = _ReviewedCrossChapterAbstractionProvider()
    bundle = build_character_lifeform(
        build_zhang_wuji_profile(),
        config=config,
        memory_store=build_default_memory_store(
            persistence_backend=memory_backend
        ),
        semantic_proposal_runtime=LLMSemanticProposalRuntime(
            provider=abstraction_provider
        ),
    )
    first_report = ChapterLiveThroughDriver().run_ledger(
        ledger=replace(full_ledger, chapters=(held_out_11,)),
        lifeform=bundle.lifeform,
        session_id="real-family-ch-11",
    )
    first_report.require_success()

    restored_after_first = ApplicationCaseMemoryStore(
        persistence_backend=build_filesystem_persistence_backend(
            base_dir=str(application_dir / "case_memory")
        )
    )
    assert restored_after_first.load_from_backend()
    first_pending = (
        restored_after_first.pending_action_abstraction_evidence()
    )
    assert len(first_pending) == 1
    assert abstraction_provider.action_abstraction_prompts == []

    reborn_bundle = build_character_lifeform(
        build_zhang_wuji_profile(),
        config=config,
        memory_store=build_default_memory_store(
            persistence_backend=memory_backend
        ),
        semantic_proposal_runtime=LLMSemanticProposalRuntime(
            provider=abstraction_provider
        ),
    )
    second_session = reborn_bundle.lifeform.create_session(
        session_id="real-family-ch-17"
    )
    restored_at_second_start = ApplicationCaseMemoryStore(
        persistence_backend=build_filesystem_persistence_backend(
            base_dir=str(application_dir / "case_memory")
        )
    )
    assert restored_at_second_start.load_from_backend()
    assert len(
        restored_at_second_start.pending_action_abstraction_evidence()
    ) == 1
    (
        second_chapter_record,
        _second_scene_records,
        second_scene_evidence,
    ) = asyncio.run(
        ChapterLiveThroughDriver().run_chapter_async(
            chapter=held_out_17,
            session=second_session,
        )
    )
    assert second_chapter_record.bake_verified is True
    first_families = (
        first_report.per_scene_evidence[0].experienced_action_family_ids
    )
    second_families = (
        second_scene_evidence[0].experienced_action_family_ids
    )

    assert first_families
    assert second_families
    assert first_families == second_families
    assert len(abstraction_provider.action_abstraction_prompts) == 1
    abstraction_prompt = (
        abstraction_provider.action_abstraction_prompts[0]
    )
    assert chapter_11.scenes[0].canonical_outcome not in abstraction_prompt
    assert chapter_17.scenes[0].canonical_outcome not in abstraction_prompt
    assert any(
        target.startswith(
            "application.case_memory.records.action-abstraction."
        )
        for target in (
            second_scene_evidence[0]
            .application_owner_targets_updated
        )
    )

    restored_after_second = ApplicationCaseMemoryStore(
        persistence_backend=build_filesystem_persistence_backend(
            base_dir=str(application_dir / "case_memory")
        )
    )
    assert restored_after_second.load_from_backend()
    second_pending = (
        restored_after_second.pending_action_abstraction_evidence()
    )
    assert second_pending == ()
    evidence_records = tuple(
        record.action_abstraction_evidence
        for record in restored_after_second.records
        if record.action_abstraction_evidence is not None
    )
    assert len(evidence_records) == 2
    assert {
        item.action_family_id for item in evidence_records
    } == {first_families[0]}
    evidence_versions = tuple(
        sorted(item.action_family_version for item in evidence_records)
    )
    assert evidence_versions[0] < evidence_versions[1]
    promotion_records = tuple(
        record.action_abstraction_promotion
        for record in restored_after_second.records
        if record.action_abstraction_promotion is not None
    )
    assert len(promotion_records) == 1
    promotion = promotion_records[0]
    assert promotion is not None
    assert promotion.action_family_id == first_families[0]
    assert promotion.action_family_version == evidence_versions[-1]
    assert set(promotion.source_outcome_ids) == {
        item.outcome_id for item in evidence_records
    }
