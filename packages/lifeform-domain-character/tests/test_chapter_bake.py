from __future__ import annotations

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
from volvence_zero.brain import BrainConfig
from volvence_zero.canonical_json import typed_to_json
from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.memory import (
    FileSystemPersistenceBackend,
    build_default_memory_store,
)
from volvence_zero.runtime import WiringLevel


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
