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
_FORCED_ROOT = _REPO_ROOT / (
    "artifacts/relationship_lab/"
    "relationship_product_horizon_forced_common_batch_20260827_"
    "pdd0d28a72f9e_c5d028d9ce41b"
)
_FORCED_PROTOCOL_ID = (
    "dd0d28a72f9e3d046f6aaab2ab2b24c0528c42f767c08610857de8a77f4aff93"
)
_FORCED_ARTIFACT_ID = (
    "92880fb7f6f692475c0fa5f57705e175508788574bc6f0b356887312e57f40d7"
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


def test_public_campaign_input_loader_returns_fresh_typed_arm_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = {
        name: (
            (_FORCED_ROOT / name).read_bytes(),
            (_FORCED_ROOT / name).stat().st_size,
            (_FORCED_ROOT / name).stat().st_mtime_ns,
        )
        for name in subject._OUTPUT_FILES
    }
    reads: list[pathlib.Path] = []
    original_read_bytes = pathlib.Path.read_bytes

    def tracked_read_bytes(path: pathlib.Path) -> bytes:
        reads.append(path.resolve())
        return original_read_bytes(path)

    monkeypatch.setattr(pathlib.Path, "read_bytes", tracked_read_bytes)
    inputs = subject.load_relationship_product_horizon_forced_campaign_inputs(
        source_v4_admission_root=_SOURCE_ROOT,
        reader_root=_READER_ROOT,
        theta0_v2_root=_THETA_ROOT,
        scanner_root=_SCANNER_ROOT,
        dynamic_root=_DYNAMIC_ROOT,
        forced_common_batch_root=_FORCED_ROOT,
        expected_forced_protocol_id=_FORCED_PROTOCOL_ID,
        expected_forced_artifact_id=_FORCED_ARTIFACT_ID,
    )

    assert inputs.forced_protocol_id == _FORCED_PROTOCOL_ID
    assert inputs.forced_artifact_id == _FORCED_ARTIFACT_ID
    assert inputs.public_view.public_plan_sha256 == inputs.public_plan_sha256
    assert len(inputs.roots) == 112
    assert not hasattr(inputs, "_dynamic_dependencies")
    assert not hasattr(inputs, "open_selected_branch_environment")
    lineage = {item.name: item.value for item in inputs.lineage}
    assert len(lineage) == len(inputs.lineage)
    assert inputs.lineage_schema_version == (
        subject.FORCED_CAMPAIGN_INPUT_LINEAGE_SCHEMA_VERSION
    )
    assert inputs.lineage_id == subject.sha256_json(
        {
            "schema_version": inputs.lineage_schema_version,
            "entries": [
                {"name": item.name, "value": item.value}
                for item in inputs.lineage
            ],
        }
    )
    assert lineage["forced_schedule_index_id"] == (
        "13c3f8e251d1679b38491569fde267a7177429ad251688c3528f993af5cb4e9f"
    )
    assert lineage["dynamic_artifact_id"] == (
        "f1a5b2f6093f0c2a8c86401feb1468867ca8948dabe4a61b832dca1970b4aaf4"
    )
    assert lineage["source_v4_source_protocol_id"] == (
        "dbf0526299558842b52f293875520e4524afff0e5a1636ab1fd10da9f74d1d91"
    )
    assert lineage["reader_artifact_id"] == (
        "ded8c0dcef7ef3aa0c7cdaacfc16e2070787a0f5a3d13f391a16d7f2623dcbb6"
    )
    assert lineage["theta0_v2_artifact_id"].startswith(
        "relationship-action-gate-theta0-sha256:"
    )
    forbidden_environment_inputs = {
        (_SOURCE_ROOT / "source/source_protocol.json").resolve(),
        (_SOURCE_ROOT / "sealed/evaluator_bundle.json").resolve(),
        (
            _SOURCE_ROOT
            / "sealed/action_counterfactual_commitment_index.json"
        ).resolve(),
    }
    assert forbidden_environment_inputs.isdisjoint(reads)
    after = {
        name: (
            (_FORCED_ROOT / name).read_bytes(),
            (_FORCED_ROOT / name).stat().st_size,
            (_FORCED_ROOT / name).stat().st_mtime_ns,
        )
        for name in subject._OUTPUT_FILES
    }
    assert after == before
    first = inputs.roots[0]
    assert first.root_sequence_index == 0
    assert first.public_root.decision_sessions[8].decision_index == 8
    assert first.schedule_artifact_id.startswith(
        "relationship-product-forced-collection-schedule-sha256:"
    )
    arms = first.fresh_arm_initializations()
    assert tuple(item.arm_id.value for item in arms) == (
        "full",
        "frozen_theta0",
        "strict_noop",
    )
    full, frozen, strict = arms
    owners = tuple(item.owner_persistence_snapshot for item in arms)
    assert owners[0] == owners[1] == owners[2]
    assert len({id(item) for item in owners}) == 3
    assert len({id(item.payload) for item in owners}) == 3
    assert full.batch_receipt == first.apply_receipt
    assert frozen.batch_receipt == strict.batch_receipt == first.withhold_receipt
    assert full.frozen_policy.policy_id == first.full_policy_id
    assert full.frozen_policy.checkpoint.update_count == 8
    assert frozen.frozen_policy == strict.frozen_policy
    assert frozen.frozen_policy is not strict.frozen_policy
    assert frozen.frozen_policy.checkpoint.update_count == 0
    assert not frozen.frozen_policy.checkpoint.processed_credit_ids
    assert frozen.frozen_policy.policy_id == first.cold_frozen_policy_id
    assert full.executor_disposition.value == "apply_candidate"
    assert frozen.executor_disposition.value == "apply_candidate"
    assert strict.executor_disposition.value == "force_strict_noop"
    assert len({id(item.forecast_runtime) for item in arms}) == 3
    assert len({item.forecast_runtime.runtime_id for item in arms}) == 1
    with pytest.raises(KeyError, match="absent from table"):
        full.forecast_runtime.read_condition(
            "this text is intentionally absent from the frozen table"
        )

    with pytest.raises(ValueError, match="protocol ID drifted"):
        subject.load_relationship_product_horizon_forced_campaign_inputs(
            source_v4_admission_root=_SOURCE_ROOT,
            reader_root=_READER_ROOT,
            theta0_v2_root=_THETA_ROOT,
            scanner_root=_SCANNER_ROOT,
            dynamic_root=_DYNAMIC_ROOT,
            forced_common_batch_root=_FORCED_ROOT,
            expected_forced_protocol_id="e" * 64,
            expected_forced_artifact_id=_FORCED_ARTIFACT_ID,
        )

    with pytest.raises(ValueError, match="artifact ID drifted"):
        subject.load_relationship_product_horizon_forced_campaign_inputs(
            source_v4_admission_root=_SOURCE_ROOT,
            reader_root=_READER_ROOT,
            theta0_v2_root=_THETA_ROOT,
            scanner_root=_SCANNER_ROOT,
            dynamic_root=_DYNAMIC_ROOT,
            forced_common_batch_root=_FORCED_ROOT,
            expected_forced_protocol_id=_FORCED_PROTOCOL_ID,
            expected_forced_artifact_id="f" * 64,
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
