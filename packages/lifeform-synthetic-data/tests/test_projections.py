from __future__ import annotations

import gzip
import json
from dataclasses import replace
from pathlib import Path

from companion_encoder import examples_from_trajectory

from lifeform_synthetic_data.canonical import canonical_json, stable_hash
from lifeform_synthetic_data.contracts import (
    GenerationTier,
    SnapshotFrame,
    TrainingUse,
    TurnRole,
)
from lifeform_synthetic_data.projections import (
    SEMANTIC_OWNER_NAMES,
    SOCIAL_OWNER_NAMES,
    ProjectionView,
    project_evaluation_only,
    project_expression_sft,
    project_human_review_queue,
    project_internal_rl,
    project_master_run,
    project_memory_retrieval,
    project_relationship_encoder,
    project_relationship_record,
    project_semantic_owner,
    project_social_cognition,
    project_temporal_ssl,
    write_projection_view,
    write_relationship_encoder_layout,
)
from lifeform_synthetic_data.scenario import load_unified_v1_blueprints
from lifeform_synthetic_data.world import (
    compile_structural_trajectory,
    replace_rendered_text,
)


def _structural(family: str = "relationship_continuity"):
    blueprint = next(item for item in load_unified_v1_blueprints() if item.family == family)
    return compile_structural_trajectory(
        blueprint,
        replicate_index=0,
        seed=42,
        run_id="projection-test",
        created_at="2026-07-20T00:00:00Z",
        git_sha="test",
    )


def _rendered(family: str = "relationship_continuity"):
    structural = _structural(family)
    slots = tuple(
        (turn.turn_id, f"rendered::{turn.role.value}::{turn.turn_index}")
        for session in structural.sessions
        for turn in session.turns
    )
    return replace_rendered_text(
        structural,
        rendered_slots=slots,
        model_id="teacher-model",
        prompt_hash="a" * 64,
    )


def _live():
    structural = _structural("environment_delayed_credit")
    frames: list[SnapshotFrame] = []
    output_sessions = []
    for session in structural.sessions:
        output_turns = []
        for turn in session.turns:
            if turn.role is not TurnRole.USER:
                output_turns.append(turn)
                continue
            refs: list[str] = []
            for slot in ("temporal_abstraction", "prediction_error", "credit"):
                payload_json = json.dumps(
                    {
                        "slot": slot,
                        "turn_id": turn.turn_id,
                        "z_t": [0.1, 0.2] if slot == "temporal_abstraction" else [],
                        "beta_t": 0.25 if slot == "temporal_abstraction" else None,
                    },
                    sort_keys=True,
                )
                snapshot_id = f"{turn.turn_id}:snapshot:{slot}"
                frames.append(
                    SnapshotFrame(
                        snapshot_id=snapshot_id,
                        turn_ref=turn.turn_id,
                        slot_name=slot,
                        owner=f"{slot}_owner",
                        version=turn.turn_index + 1,
                        timestamp_ms=turn.turn_index + 1,
                        value_type="test.SnapshotValue",
                        value_hash=stable_hash(json.loads(payload_json)),
                        payload_json=payload_json,
                        wiring_level="active",
                        description="",
                    )
                )
                refs.append(snapshot_id)
            output_turns.append(replace(turn, snapshot_refs=tuple(refs)))
        output_sessions.append(replace(session, turns=tuple(output_turns)))
    return replace(
        structural,
        generation_tier=GenerationTier.LIVE_THROUGH,
        sessions=tuple(output_sessions),
        snapshot_frames=tuple(frames),
    )


def test_relationship_projection_uses_only_fsm_ground_truth() -> None:
    trajectory = _rendered("rupture_repair")

    projected = project_relationship_encoder(trajectory)
    record = project_relationship_record(trajectory)
    examples = examples_from_trajectory(projected, split=record.split.value)

    assert projected.labels
    assert all(label.source.value == "fsm_ground_truth" for label in projected.labels)
    assert any(label.phase.value == "ruptured" for label in projected.labels)
    assert all("generator_truth:" in label.evidence for label in projected.labels)
    assert examples
    assert record.master_trajectory_hash == stable_hash(trajectory)
    assert record.split.value == "test"


def test_expression_sft_requires_rendered_teacher_text() -> None:
    rendered = _rendered()

    records = project_expression_sft(rendered)

    assert records
    assert all(record.training_use is TrainingUse.TARGET for record in records)
    payload = json.loads(records[0].payload_json)
    assert payload["response"].startswith("rendered::assistant")
    assert payload["response_contract"]


def test_semantic_and_social_projection_cover_declared_owners() -> None:
    trajectories = tuple(
        _structural(family)
        for family in (
            "relationship_continuity",
            "preference_personalization",
            "boundary_consent_autonomy",
            "goal_value_drift",
            "plan_commitment_open_loop",
            "task_tool_execution",
            "belief_uncertainty_verification",
            "multi_party_identity_privacy",
            "tom_common_ground_group",
        )
    )
    semantic_records = tuple(record for trajectory in trajectories for record in project_semantic_owner(trajectory))
    social_records = tuple(record for trajectory in trajectories for record in project_social_cognition(trajectory))
    semantic_owners = {json.loads(record.payload_json)["target_owner"] for record in semantic_records}
    social_owners = {json.loads(record.payload_json)["target_owner"] for record in social_records}

    assert semantic_owners == set(SEMANTIC_OWNER_NAMES)
    assert social_owners.issubset(SOCIAL_OWNER_NAMES)
    assert {
        "conversational_role",
        "belief_about_other",
        "intent_about_other",
        "common_ground",
        "groups",
    }.issubset(social_owners)


def test_memory_temporal_rl_and_review_views_keep_lineage() -> None:
    rendered = _rendered()
    live = _live()

    memory_records = project_memory_retrieval(rendered)
    ssl_records = project_temporal_ssl(live)
    rl_records = project_internal_rl(live)
    eval_records = project_evaluation_only(live)
    review_records = project_human_review_queue(live, sample_rate=1.0)

    assert memory_records and all(record.source_refs for record in memory_records)
    assert ssl_records and all(record.source_refs for record in ssl_records)
    assert rl_records and all(record.source_refs for record in rl_records)
    assert all(json.loads(record.payload_json)["manual_semantic_label"] is None for record in rl_records)
    assert eval_records[0].training_use is TrainingUse.EVAL_ONLY
    assert review_records[0].training_use is TrainingUse.QUARANTINED


def test_projection_files_and_relationship_layout_are_deterministic(
    tmp_path: Path,
) -> None:
    train = _structural("relationship_continuity")
    val = _structural("absence_reengagement")
    records = (
        project_relationship_record(train),
        project_relationship_record(val),
    )

    first = write_projection_view(
        records,
        view=ProjectionView.RELATIONSHIP_ENCODER,
        output_dir=tmp_path / "views",
    )
    first_bytes = (tmp_path / "views" / first.records_uri).read_bytes()
    second = write_projection_view(
        records,
        view=ProjectionView.RELATIONSHIP_ENCODER,
        output_dir=tmp_path / "views",
    )
    manifest_path = write_relationship_encoder_layout(
        (train, val),
        output_root=tmp_path / "encoder",
    )

    assert second.records_sha256 == first.records_sha256
    assert (tmp_path / "views" / second.records_uri).read_bytes() == first_bytes
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["policy"] == "whole_scenario_family"
    assert manifest["split_families"]["train"] == ["relationship_continuity"]
    assert manifest["split_families"]["val"] == ["absence_reengagement"]


def test_project_master_run_streams_shards_and_writes_all_views(
    tmp_path: Path,
) -> None:
    trajectories = (
        _rendered("relationship_continuity"),
        _rendered("absence_reengagement"),
    )
    master_root = tmp_path / "master"
    master_root.mkdir()
    with gzip.open(
        master_root / "shard-00000.jsonl.gz",
        "wt",
        encoding="utf-8",
    ) as sink:
        for trajectory in sorted(
            trajectories,
            key=lambda item: item.trajectory_id,
        ):
            sink.write(canonical_json(trajectory) + "\n")

    manifests = project_master_run(tmp_path)

    by_view = {manifest.view: manifest for manifest in manifests}
    assert set(by_view) == set(ProjectionView)
    assert by_view[ProjectionView.RELATIONSHIP_ENCODER].record_count == 2
    assert by_view[ProjectionView.EXPRESSION_SFT].record_count > 0
    projection_audit = json.loads((tmp_path / "projections" / "projection-audit.json").read_text(encoding="utf-8"))
    assert projection_audit["passed"] is True
