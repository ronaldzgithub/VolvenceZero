from __future__ import annotations

from dataclasses import replace
import json
import pathlib
import shutil

import pytest

import lifeform_evolution.relationship_product_horizon_theta0_calibration as subject
from lifeform_domain_emogpt.relationship_action_gate import (
    RelationshipActionGate,
    RelationshipActionGateCheckpoint,
    RelationshipActionGateTheta0Artifact,
)


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SOURCE_V3 = _REPO_ROOT / (
    "artifacts/relationship_lab/"
    "relationship_product_source_v3_campaign_admission_"
    "20260826_p98d51d845fd0"
)
_PREFLIGHT = _REPO_ROOT / (
    "artifacts/relationship_lab/"
    "relationship_condition_reader_qualification_preflight_v6_"
    "20260826_p723796027a64_c7381743b0"
)
_READER = _REPO_ROOT / (
    "artifacts/relationship_lab/"
    "relationship_product_horizon_development_reader_"
    "20260826_pa1ea1e30fd7b_ce8272ce7d3da"
)
_SOURCE_V4 = _REPO_ROOT / (
    "artifacts/relationship_lab/"
    "relationship_product_horizon_source_v4_admission_20260826_b3988b21"
)


def _public_only_source_v3(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "source-v3-public-envelope"
    for relative in (
        "manifest.json",
        "replay_a/manifest.json",
        "replay_a/protocol.json",
        "replay_a/public/source_plan.json",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_SOURCE_V3 / relative, target)
    return root


def _load_public_dependencies(source_v3_root: pathlib.Path) -> subject._Dependencies:
    return subject._load_dependencies(
        source_v3_admission_root=source_v3_root,
        preflight_root=_PREFLIGHT,
        reader_root=_READER,
        source_v4_admission_root=_SOURCE_V4,
    )


def test_protocol_rejects_bool_integer_aliases_and_nonfinite_json(
    tmp_path: pathlib.Path,
) -> None:
    loaded = subject.load_relationship_product_horizon_theta0_calibration_protocol()
    assert loaded.protocol_id == (
        "0e51c343646b971daa144ac098b76221c0b5811d452040c80e776aa003e3678c"
    )
    assert loaded.raw_sha256 == (
        "458ad3af30f35e50425c28f6fef37529addd3703c5ff3956cd215590ad2d5d7b"
    )
    protocol = json.loads(
        subject.relationship_product_horizon_theta0_calibration_protocol_path().read_text(
            encoding="utf-8"
        )
    )
    variants = (
        (("gate", "artifact_version"), True),
        (("topology", "owner_reset_each_root"), 1),
        (("terminal_gates", "pending_decision_count"), False),
        (("claims", "model_output_count"), False),
        (("claim_boundary",), {}),
    )
    for index, (path, replacement) in enumerate(variants):
        mutated = json.loads(json.dumps(protocol))
        cursor = mutated
        for part in path[:-1]:
            cursor = cursor[part]
        cursor[path[-1]] = replacement
        candidate = tmp_path / f"protocol-{index}.json"
        candidate.write_text(json.dumps(mutated), encoding="utf-8")
        with pytest.raises(ValueError):
            subject.load_relationship_product_horizon_theta0_calibration_protocol(
                candidate
            )

    for token in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(ValueError, match="non-finite JSON constant"):
            subject._parse_json_bytes(
                f'{{"value":{token}}}'.encode(),
                source="strict JSON probe",
            )


def test_public_join_is_exact_and_rejects_duplicate_or_unmatched_world(
    tmp_path: pathlib.Path,
) -> None:
    source_v3_root = _public_only_source_v3(tmp_path)
    dependencies = _load_public_dependencies(source_v3_root)

    public_join = subject._build_public_join(dependencies)

    assert public_join["row_count"] == 224
    assert [row["source_sequence_index"] for row in public_join["rows"]] == list(
        range(224)
    )
    assert len({row["join_row_id"] for row in public_join["rows"]}) == 224
    assert len({row["challenge_input_index"] for row in public_join["rows"]}) == 224
    assert not (source_v3_root / "replay_a/sealed").exists()
    assert not (source_v3_root / "replay_b").exists()
    assert not (source_v3_root / "comparison.json").exists()

    corpus = dict(dependencies.preflight_corpus)
    challenge = list(corpus["challenge_inputs"])
    challenge[-1] = challenge[0]
    corpus["challenge_inputs"] = challenge
    with pytest.raises(ValueError, match="challenge text digest is not unique"):
        subject._build_public_join(replace(dependencies, preflight_corpus=corpus))

    public_view = dependencies.public_view
    first = public_view.subjects[0]
    bad_first = replace(first, world_clone_id="f" * 64)
    bad_public_view = replace(
        public_view,
        subjects=(bad_first, *public_view.subjects[1:]),
    )
    with pytest.raises(ValueError, match="world_clone_id did not uniquely join"):
        subject._EnvironmentScope(
            dependencies=replace(dependencies, public_view=bad_public_view)
        )


def test_materialization_closes_192_updates_and_cold_theta0_replays(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_v3_root = _public_only_source_v3(tmp_path)
    dependencies = _load_public_dependencies(source_v3_root)
    monkeypatch.setattr(subject, "_load_dependencies", lambda **_kwargs: dependencies)
    output = tmp_path / "theta0"

    manifest = subject.materialize_relationship_product_horizon_theta0_calibration(
        source_v3_admission_root=source_v3_root,
        preflight_root=_PREFLIGHT,
        reader_root=_READER,
        source_v4_admission_root=_SOURCE_V4,
        output_dir=output,
        implementation_git_commit="a" * 40,
    )

    assert manifest["status"] == "development_theta0_materialized_effect_not_tested"
    assert manifest["decision_count"] == 192
    assert manifest["gate_update_count"] == 192
    assert manifest["processed_credit_id_count"] == 192
    assert manifest["unique_credit_id_count"] == 192
    assert manifest["pending_decision_count"] == 0
    assert manifest["challenge_label_file_read_count"] == 0
    assert manifest["group_split_file_read_count"] == 0
    assert manifest["admitted_sealed_file_runtime_read_count"] == 0
    assert manifest["claims"]["development_theta0_materialized"] is True
    assert manifest["claims"]["learnable_effect"] is False
    assert manifest["claims"]["steerable_effect"] is False
    assert {path.name for path in output.iterdir()} == {
        "protocol.json",
        "public_join.json",
        "calibration_trace.jsonl",
        "theta0_artifact.json",
        "manifest.json",
    }

    records = [
        json.loads(line)
        for line in (output / "calibration_trace.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    preactions = [item for item in records if item["record_type"] == "preaction"]
    postactions = [item for item in records if item["record_type"] == "postaction"]
    roots = [item for item in records if item["record_type"] == "root_start"]
    assert len(roots) == 8
    assert [item["gate_update_count"] for item in roots] == list(range(0, 192, 24))
    assert [item["global_sequence_index"] for item in preactions] == list(range(192))
    assert [item["gate_update_count_before"] for item in preactions] == list(range(192))
    assert {item["gate_pending_count_before"] for item in preactions} == {0}
    assert {item["gate_pending_count_after_preaction"] for item in preactions} == {1}
    assert [item["gate_update_count"] for item in postactions] == list(range(1, 193))
    assert {item["gate_pending_count"] for item in postactions} == {0}
    assert len({item["credit"]["record_id"] for item in postactions}) == 192
    assert [item["credit"]["timestamp_ms"] for item in postactions] == [
        root_index * 52 + 5 + 2 * decision_index
        for root_index in range(8)
        for decision_index in range(24)
    ]
    safe_environment_fields = {
        "environment_subject_id",
        "selected_action_id",
        "typed_outcome_id",
        "rendered_user_reaction_sha256",
        "environment_evidence_ref",
        "environment_version",
    }
    assert all(set(item["environment"]) == safe_environment_fields for item in postactions)

    terminal = records[-1]
    checkpoint = RelationshipActionGateCheckpoint.from_payload(
        terminal["final_checkpoint"]
    )
    theta0 = RelationshipActionGateTheta0Artifact.from_payload(
        json.loads((output / "theta0_artifact.json").read_text(encoding="utf-8"))
    )
    theta0.validate_source_checkpoint(checkpoint)
    cold = RelationshipActionGate.from_theta0(theta0).export_checkpoint()
    assert cold.weights == checkpoint.weights
    assert cold.bias == checkpoint.bias
    assert cold.update_count == 0
    assert cold.processed_credit_ids == ()
    assert cold.pending_decisions == ()

    validated = subject.validate_relationship_product_horizon_theta0_calibration(
        source_v3_admission_root=source_v3_root,
        preflight_root=_PREFLIGHT,
        reader_root=_READER,
        source_v4_admission_root=_SOURCE_V4,
        output_dir=output,
        expected_protocol_id=manifest["protocol_id"],
        expected_artifact_id=manifest["artifact_id"],
    )
    assert validated == manifest
    with pytest.raises(ValueError, match="external expected protocol"):
        subject.validate_relationship_product_horizon_theta0_calibration(
            source_v3_admission_root=source_v3_root,
            preflight_root=_PREFLIGHT,
            reader_root=_READER,
            source_v4_admission_root=_SOURCE_V4,
            output_dir=output,
            expected_protocol_id="0" * 64,
            expected_artifact_id=manifest["artifact_id"],
        )
    with pytest.raises(FileExistsError, match="create-only"):
        subject.materialize_relationship_product_horizon_theta0_calibration(
            source_v3_admission_root=source_v3_root,
            preflight_root=_PREFLIGHT,
            reader_root=_READER,
            source_v4_admission_root=_SOURCE_V4,
            output_dir=output,
            implementation_git_commit="a" * 40,
        )


def test_all_zero_terminal_writes_no_consumable_theta0(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_v3_root = _public_only_source_v3(tmp_path)
    dependencies = _load_public_dependencies(source_v3_root)
    monkeypatch.setattr(subject, "_load_dependencies", lambda **_kwargs: dependencies)
    credit_ids = tuple(f"credit-{index:03d}" for index in range(192))
    checkpoint = RelationshipActionGateCheckpoint(
        artifact_id="relationship-action-gate-zero-init",
        artifact_version=1,
        weights=(0.0, 0.0, 0.0, 0.0, 0.0),
        bias=0.0,
        update_count=192,
        processed_credit_ids=credit_ids,
        pending_decisions=(),
    )

    async def fake_run(
        *,
        dependencies: subject._Dependencies,
        public_join: object,
        sink: subject._TraceSink,
    ) -> subject._CalibrationReplay:
        del dependencies, public_join
        sink.append(
            {
                "schema_version": subject.THETA0_TRACE_SCHEMA_VERSION,
                "record_type": "terminal",
                "final_parameters_nonzero": False,
                "terminal_status": "calibration_completed_no_nonzero_theta0",
            }
        )
        return subject._CalibrationReplay(
            final_checkpoint=checkpoint,
            credit_ids=credit_ids,
            root_mapping=(),
            terminal_status="calibration_completed_no_nonzero_theta0",
        )

    monkeypatch.setattr(subject, "_run_calibration", fake_run)
    output = tmp_path / "all-zero"
    manifest = subject.materialize_relationship_product_horizon_theta0_calibration(
        source_v3_admission_root=source_v3_root,
        preflight_root=_PREFLIGHT,
        reader_root=_READER,
        source_v4_admission_root=_SOURCE_V4,
        output_dir=output,
        implementation_git_commit="b" * 40,
    )

    assert manifest["status"] == "calibration_completed_no_nonzero_theta0"
    assert manifest["theta0_artifact_id"] is None
    assert manifest["claims"]["development_theta0_materialized"] is False
    assert not (output / "theta0_artifact.json").exists()
    assert {path.name for path in output.iterdir()} == {
        "protocol.json",
        "public_join.json",
        "calibration_trace.jsonl",
        "manifest.json",
    }
