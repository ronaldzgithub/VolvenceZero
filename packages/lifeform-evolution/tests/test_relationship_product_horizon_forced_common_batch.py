from __future__ import annotations

import asyncio
from collections import Counter
import json
import pathlib

import pytest

import lifeform_evolution.relationship_product_horizon_forced_common_batch as subject
from lifeform_domain_emogpt.relationship_action_gate import (
    RelationshipActionGate,
    RelationshipActionGateBatchDisposition,
    RelationshipActionGateCreditBatch,
    RelationshipActionGateTheta0Artifact,
)


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SOURCE_ROOT = _REPO_ROOT / (
    "artifacts/relationship_lab/"
    "relationship_product_horizon_source_v4_admission_20260826_b3988b21"
)
_READER_ROOT = _REPO_ROOT / (
    "artifacts/relationship_lab/"
    "relationship_product_horizon_development_reader_20260826_"
    "pa1ea1e30fd7b_ce8272ce7d3da"
)
_THETA_ROOT = _REPO_ROOT / (
    "artifacts/relationship_lab/"
    "relationship_product_horizon_theta0_v2_bootstrap_20260826_"
    "pdfefb9faa240_c66c2d83a"
)
_BASE_THETA_ROOT = _REPO_ROOT / (
    "artifacts/relationship_lab/"
    "relationship_product_horizon_theta0_calibration_20260826_"
    "p0e51c343646b_c7922c8a8fe23"
)
_SCANNER_ROOT = _REPO_ROOT / (
    "artifacts/relationship_lab/"
    "relationship_product_horizon_transductive_public_opportunity_20260826_"
    "p4471c9ab49bc_c0ffda0a1976d"
)
_DYNAMIC_ROOT = _REPO_ROOT / (
    "artifacts/relationship_lab/"
    "relationship_product_horizon_dynamic_collection_prefix_20260826_"
    "p47cea5fae3be_cc275bd908afd"
)


def _dependencies() -> subject._Dependencies:
    return subject._load_dependencies(
        source_v4_admission_root=_SOURCE_ROOT,
        reader_root=_READER_ROOT,
        theta0_v2_root=_THETA_ROOT,
        scanner_root=_SCANNER_ROOT,
        dynamic_root=_DYNAMIC_ROOT,
    )


def _successful_replay() -> subject._ForcedBatchReplay:
    return subject._ForcedBatchReplay(
        completed_root_count=112,
        onboarding_count=448,
        preaction_count=896,
        postaction_count=896,
        first_preaction_exact_match_count=112,
        first_preaction_projection_sha256="a" * 64,
        later_owner_handoff_count=784,
        owner_writeback_change_count=896,
        selected_branch_resolution_count=896,
        selected_branch_commitment_match_count=896,
        unique_command_count=896,
        unique_receipt_count=896,
        unique_exposure_count=896,
        unique_forecast_count=896,
        unique_commitment_count=896,
        unique_settlement_count=896,
        unique_environment_evidence_ref_count=896,
        unique_credit_count=896,
        cold_checkpoint_unchanged_count=896,
        root_batch_count=112,
        unique_batch_count=112,
        apply_receipt_count=112,
        withhold_receipt_count=112,
        full_owner_replay_count=112,
        owner_roundtrip_count=112,
        parameter_delta_nonzero_root_count=1,
        parameter_cap_hit_root_count=0,
        scheduled_role_counts={"neutral_noop": 448, "owner_recommendation": 448},
        delivered_action_counts={"neutral_noop": 448, "stay_present_without_probe": 448},
        terminal_failure_reasons=(),
        terminal_status=subject._SUCCESS_STATUS,
    )


def test_protocol_freezes_root_local_batches_and_claim_ceiling(
    tmp_path: pathlib.Path,
) -> None:
    loaded = subject.load_relationship_product_horizon_forced_common_batch_protocol()
    assert loaded.protocol_id == (
        "dd0d28a72f9e3d046f6aaab2ab2b24c0528c42f767c08610857de8a77f4aff93"
    )
    assert loaded.payload["root_local_batch_transition"]["global_batch_forbidden"]
    assert loaded.payload["forced_schedule_index"]["schedule_artifact_count"] == 112
    assert loaded.payload["collection"]["forced_exposure_sequence_scope"] == (
        "root_local_zero_through_seven"
    )
    assert loaded.payload["runtime_order"]["dynamic_trace_read"] is False
    assert loaded.payload["claims"]["forced_common_batch_execution_authorized"]
    assert not loaded.payload["claims"]["campaign_execution_authorized"]
    assert not loaded.payload["claims"]["learnable_effect"]
    assert not loaded.payload["claims"]["steerable_effect"]

    protocol = json.loads(
        subject.relationship_product_horizon_forced_common_batch_protocol_path().read_text(
            encoding="utf-8"
        )
    )
    variants = (
        (("adaptive_lineage", "unseen"), True),
        (("forced_schedule_index", "root_count"), 111),
        (("collection", "credit_applied_online"), True),
        (("root_local_batch_transition", "global_batch_forbidden"), False),
        (("runtime_order", "dynamic_trace_read"), True),
        (("terminal_gates", "campaign_execution_authorized_on_success"), True),
        (("claims", "campaign_protocol_freeze_authorized"), True),
        (("causal_firewall", "cuda_execution_count"), False),
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
            subject.load_relationship_product_horizon_forced_common_batch_protocol(
                candidate
            )


def test_schedule_index_is_public_position_only_and_root_local() -> None:
    dependencies = _dependencies()
    index, schedules = subject._build_schedule_index(dependencies.public_view)

    assert index == dependencies.schedule_index
    assert index["schedule_index_id"] == (
        "13c3f8e251d1679b38491569fde267a7177429ad251688c3528f993af5cb4e9f"
    )
    assert len(schedules) == 112
    assert len({item.artifact_id for item in schedules}) == 112
    global_counts: Counter[str] = Counter()
    column_counts = {index: Counter() for index in range(8)}
    for schedule in schedules:
        assert tuple(item.sequence_index for item in schedule.entries) == tuple(range(8))
        root_counts = Counter(item.forced_action_role.value for item in schedule.entries)
        assert root_counts == {"neutral_noop": 4, "owner_recommendation": 4}
        for entry in schedule.entries:
            global_counts[entry.forced_action_role.value] += 1
            column_counts[entry.sequence_index][entry.forced_action_role.value] += 1
    assert global_counts == {"neutral_noop": 448, "owner_recommendation": 448}
    assert all(
        counts == {"neutral_noop": 56, "owner_recommendation": 56}
        for counts in column_counts.values()
    )

    with pytest.raises(ValueError, match="contiguous and ordered from zero"):
        subject.RelationshipProductForcedCollectionScheduleArtifact(
            entries=(*schedules[0].entries, *schedules[1].entries)
        )


def test_dependency_loader_reads_dynamic_manifest_but_never_dynamic_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads: list[pathlib.Path] = []
    original = pathlib.Path.read_bytes

    def tracked(path: pathlib.Path) -> bytes:
        resolved = path.resolve()
        reads.append(resolved)
        if resolved == (_DYNAMIC_ROOT / "dynamic_collection_prefix.jsonl").resolve():
            raise AssertionError("natural dynamic trace must never be opened")
        return original(path)

    monkeypatch.setattr(pathlib.Path, "read_bytes", tracked)
    dependencies = _dependencies()

    assert len(dependencies.schedules) == 112
    assert reads.count((_DYNAMIC_ROOT / "manifest.json").resolve()) == 1
    assert (_DYNAMIC_ROOT / "dynamic_collection_prefix.jsonl").resolve() not in reads


def test_forced_safe_projection_excludes_natural_delivery_fields() -> None:
    dependencies = _dependencies()
    projections = tuple(
        subject._forced_safe_projection_from_scanner(item)
        for item in dependencies.dynamic_dependencies.expected_first_projections
    )

    assert len(projections) == 112
    assert tuple(projections[0]) == subject.FORCED_SAFE_FIRST_PROJECTION_FIELDS
    excluded = {
        "executor_disposition",
        "executor_status",
        "delivered_advisory_id",
        "temporal_delivered_action_id",
        "temporal_controller_params_hash",
        "temporal_action_family_version",
        "temporal_action_advisory_status",
    }
    assert excluded.isdisjoint(projections[0])


def test_owner_envelope_roundtrips_without_consumer_interpretation() -> None:
    dependencies = _dependencies()
    owner = asyncio.run(
        subject.dynamic.scan._post_onboarding_state(dependencies.public_view.roots[0])
    )
    payload = subject._owner_snapshot_payload(owner)
    restored = subject._owner_snapshot_from_payload(payload)

    assert restored == owner
    assert subject.social_record_store_persistence_sha256(restored) == (
        subject.social_record_store_persistence_sha256(owner)
    )


def test_existing_forced_batch_contract_supports_root_local_three_arm_transition() -> None:
    source_batch_payload = subject.cal._parse_json_bytes(
        (_THETA_ROOT / "credit_batch.json").read_bytes(),
        source="existing theta0-v2 credit batch",
    )
    source_batch = RelationshipActionGateCreditBatch.from_payload(
        source_batch_payload
    )
    batch = RelationshipActionGateCreditBatch(
        exposures=source_batch.exposures[:8],
        credits=source_batch.credits[:8],
    )
    theta0 = RelationshipActionGateTheta0Artifact.from_payload(
        subject.cal._parse_json_bytes(
            (_BASE_THETA_ROOT / "theta0_artifact.json").read_bytes(),
            source="base theta0 artifact",
        )
    )
    full = RelationshipActionGate.from_theta0(theta0)
    frozen = RelationshipActionGate.from_theta0(theta0)
    strict = RelationshipActionGate.from_theta0(theta0)
    full_plan = full.plan_credit_batch(batch)
    frozen_plan = frozen.plan_credit_batch(batch)
    strict_plan = strict.plan_credit_batch(batch)
    assert full_plan == frozen_plan == strict_plan
    apply = full.commit_credit_batch(
        full_plan, disposition=RelationshipActionGateBatchDisposition.APPLY
    )
    withheld = frozen.commit_credit_batch(
        frozen_plan, disposition=RelationshipActionGateBatchDisposition.WITHHOLD
    )
    strict_withheld = strict.commit_credit_batch(
        strict_plan, disposition=RelationshipActionGateBatchDisposition.WITHHOLD
    )

    assert apply.update_count_delta == 8
    assert apply.atomic_commit_count == 1
    assert withheld == strict_withheld
    assert withheld.update_count_delta == 0
    replayed = RelationshipActionGate.from_applied_credit_batch(
        theta0, batch=batch, receipt=apply
    )
    assert replayed.freeze_for_evaluation() == full.freeze_for_evaluation()
    assert frozen.freeze_for_evaluation() == strict.freeze_for_evaluation()


def test_artifact_plumbing_is_create_only_schedule_first_and_manifest_last(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "forced-common-batch"
    dependencies = _dependencies()
    replay = _successful_replay()
    order_checked: list[bool] = []
    real_validate_persisted_evidence = subject._validate_persisted_evidence

    def fake_load_dependencies(**_: pathlib.Path) -> subject._Dependencies:
        return dependencies

    async def fake_run(
        *,
        dependencies: subject._Dependencies,
        trace_sink: subject.cal._TraceSink,
        state_sink: subject.cal._TraceSink,
        transition_sink: subject.cal._TraceSink,
    ) -> subject._ForcedBatchReplay:
        if isinstance(trace_sink, subject.cal._FsyncTraceSink):
            root = pathlib.Path(trace_sink._handle.name).parent
            assert (root / "protocol.json").is_file()
            assert (root / "schedule_index.json").is_file()
            assert not (root / "manifest.json").exists()
            order_checked.append(True)
        trace_sink.append({"record_type": "unit-trace"})
        state_sink.append({"record_type": "unit-state"})
        transition_sink.append({"record_type": "unit-transition"})
        return replay

    monkeypatch.setattr(subject, "_load_dependencies", fake_load_dependencies)
    monkeypatch.setattr(subject, "_run_forced_common_batch", fake_run)
    manifest = subject.materialize_relationship_product_horizon_forced_common_batch(
        source_v4_admission_root=_SOURCE_ROOT,
        reader_root=_READER_ROOT,
        theta0_v2_root=_THETA_ROOT,
        scanner_root=_SCANNER_ROOT,
        dynamic_root=_DYNAMIC_ROOT,
        output_dir=output,
        implementation_git_commit="a" * 40,
    )

    assert order_checked == [True]
    assert {path.name for path in output.iterdir()} == subject._OUTPUT_FILES
    before = {
        path.name: (path.read_bytes(), path.stat().st_size, path.stat().st_mtime_ns)
        for path in output.iterdir()
    }
    monkeypatch.setattr(
        subject,
        "_validate_persisted_evidence",
        lambda **_: replay,
    )
    validated = subject.validate_relationship_product_horizon_forced_common_batch(
        source_v4_admission_root=_SOURCE_ROOT,
        reader_root=_READER_ROOT,
        theta0_v2_root=_THETA_ROOT,
        scanner_root=_SCANNER_ROOT,
        dynamic_root=_DYNAMIC_ROOT,
        output_dir=output,
        expected_protocol_id=manifest["protocol_id"],
        expected_artifact_id=manifest["artifact_id"],
    )
    after = {
        path.name: (path.read_bytes(), path.stat().st_size, path.stat().st_mtime_ns)
        for path in output.iterdir()
    }
    assert validated == manifest
    assert after == before

    schedule_path = output / "schedule_index.json"
    schedule_raw = schedule_path.read_bytes()
    schedule_path.write_bytes(b"!" + schedule_raw[1:])
    monkeypatch.setattr(
        subject,
        "_validate_persisted_evidence",
        real_validate_persisted_evidence,
    )
    try:
        with pytest.raises(
            ValueError, match="persisted forced schedule-index bytes drifted"
        ):
            subject.validate_relationship_product_horizon_forced_common_batch(
                source_v4_admission_root=_SOURCE_ROOT,
                reader_root=_READER_ROOT,
                theta0_v2_root=_THETA_ROOT,
                scanner_root=_SCANNER_ROOT,
                dynamic_root=_DYNAMIC_ROOT,
                output_dir=output,
                expected_protocol_id=manifest["protocol_id"],
                expected_artifact_id=manifest["artifact_id"],
            )
    finally:
        schedule_path.write_bytes(schedule_raw)

    monkeypatch.setattr(
        subject,
        "_validate_persisted_evidence",
        lambda **_: replay,
    )
    extra = output / "unexpected.txt"
    extra.write_text("unexpected", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="output inventory drifted"):
            subject.validate_relationship_product_horizon_forced_common_batch(
                source_v4_admission_root=_SOURCE_ROOT,
                reader_root=_READER_ROOT,
                theta0_v2_root=_THETA_ROOT,
                scanner_root=_SCANNER_ROOT,
                dynamic_root=_DYNAMIC_ROOT,
                output_dir=output,
                expected_protocol_id=manifest["protocol_id"],
                expected_artifact_id=manifest["artifact_id"],
            )
    finally:
        extra.unlink()

    with pytest.raises(FileExistsError, match="create-only"):
        subject.materialize_relationship_product_horizon_forced_common_batch(
            source_v4_admission_root=_SOURCE_ROOT,
            reader_root=_READER_ROOT,
            theta0_v2_root=_THETA_ROOT,
            scanner_root=_SCANNER_ROOT,
            dynamic_root=_DYNAMIC_ROOT,
            output_dir=output,
            implementation_git_commit="a" * 40,
        )
    with pytest.raises(ValueError, match="artifact ID drifted"):
        subject.validate_relationship_product_horizon_forced_common_batch(
            source_v4_admission_root=_SOURCE_ROOT,
            reader_root=_READER_ROOT,
            theta0_v2_root=_THETA_ROOT,
            scanner_root=_SCANNER_ROOT,
            dynamic_root=_DYNAMIC_ROOT,
            output_dir=output,
            expected_protocol_id=manifest["protocol_id"],
            expected_artifact_id="b" * 64,
        )

    failed_output = tmp_path / "forced-common-batch-incomplete"

    async def failed_run(**_: object) -> subject._ForcedBatchReplay:
        raise RuntimeError("injected collection failure")

    monkeypatch.setattr(subject, "_run_forced_common_batch", failed_run)
    with pytest.raises(RuntimeError, match="injected collection failure"):
        subject.materialize_relationship_product_horizon_forced_common_batch(
            source_v4_admission_root=_SOURCE_ROOT,
            reader_root=_READER_ROOT,
            theta0_v2_root=_THETA_ROOT,
            scanner_root=_SCANNER_ROOT,
            dynamic_root=_DYNAMIC_ROOT,
            output_dir=failed_output,
            implementation_git_commit="a" * 40,
        )
    assert failed_output.is_dir()
    assert not (failed_output / "manifest.json").exists()


def test_incomplete_or_degenerate_transition_has_explicit_terminal() -> None:
    reasons = subject._terminal_failure_reasons(
        completed_root_count=0,
        onboarding_count=0,
        preaction_count=0,
        postaction_count=0,
        first_preaction_exact_match_count=0,
        later_owner_handoff_count=0,
        owner_writeback_change_count=0,
        selected_branch_resolution_count=0,
        selected_branch_commitment_match_count=0,
        unique_command_count=0,
        unique_receipt_count=0,
        unique_exposure_count=0,
        unique_forecast_count=0,
        unique_commitment_count=0,
        unique_settlement_count=0,
        unique_environment_evidence_ref_count=0,
        unique_credit_count=0,
        cold_checkpoint_unchanged_count=0,
        root_batch_count=0,
        unique_batch_count=0,
        apply_receipt_count=0,
        withhold_receipt_count=0,
        full_owner_replay_count=0,
        owner_roundtrip_count=0,
        parameter_cap_hit_root_count=0,
        delivered_noop_count=0,
        delivered_nonnoop_count=0,
    )
    assert "root_batch_count_not_112" in reasons
    assert "unique_batch_count_not_112" in reasons
    assert "delivered_noop_count_below_one" in reasons
    assert "delivered_nonnoop_count_below_one" in reasons
    assert subject._DEGENERATE_STATUS == "arm_degeneracy_invalid_contrast_no_claim"
