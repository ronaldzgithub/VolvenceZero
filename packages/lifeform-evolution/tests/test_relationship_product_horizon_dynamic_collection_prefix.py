from __future__ import annotations

import asyncio
from dataclasses import fields, replace
import json
import pathlib

import pytest

import lifeform_evolution.relationship_product_horizon_dynamic_collection_prefix as subject
from lifeform_domain_emogpt.lab.relationship_product_pulse import (
    RelationshipProductExecutorDisposition,
    settle_relationship_product_frozen_pulse,
)
from volvence_zero.dialogue_trace import DialogueExternalOutcomeKind
from volvence_zero.social import social_record_store_persistence_sha256


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
_DYNAMIC_PROTOCOL_ID = (
    "47cea5fae3be13067492785893a7b621285ddce36efbe6c2eefae3331c50dbb2"
)
_DYNAMIC_ARTIFACT_ID = (
    "f1a5b2f6093f0c2a8c86401feb1468867ca8948dabe4a61b832dca1970b4aaf4"
)


def _dependencies() -> subject._Dependencies:
    return subject._load_dependencies(
        source_v4_admission_root=_SOURCE_ROOT,
        reader_root=_READER_ROOT,
        theta0_v2_root=_THETA_ROOT,
        scanner_root=_SCANNER_ROOT,
    )


def _successful_replay() -> subject._DynamicReplay:
    return subject._DynamicReplay(
        completed_root_count=112,
        onboarding_count=448,
        preaction_count=896,
        postaction_count=896,
        first_preaction_exact_match_count=112,
        later_owner_handoff_count=784,
        owner_writeback_change_count=896,
        selected_branch_resolution_count=896,
        selected_branch_commitment_match_count=896,
        unique_selected_branch_commitment_count=896,
        prediction_error_count=896,
        credit_count=896,
        unique_credit_count=896,
        unique_settlement_count=896,
        unique_environment_evidence_ref_count=896,
        cold_checkpoint_unchanged_count=896,
        temporal_delivered_action_counts={
            subject.RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value: 896,
        },
        temporal_delivered_nonnoop_count=896,
        temporal_delivered_nonnoop_root_count=112,
        first_preaction_projection_sha256=(
            "5eee5690e366262eed331baadb217ef3259bf958e6d125687558f5f4bb4284d1"
        ),
        terminal_failure_reasons=(),
        terminal_status=subject._SUCCESS_STATUS,
    )


def test_protocol_freezes_natural_dynamic_gate_and_claim_ceiling(
    tmp_path: pathlib.Path,
) -> None:
    loaded = (
        subject.load_relationship_product_horizon_dynamic_collection_prefix_protocol()
    )
    assert loaded.payload["adaptive_lineage"]["transductive"] is True
    assert loaded.payload["adaptive_lineage"]["unseen"] is False
    assert loaded.payload["collection_prefix"]["executor_disposition"] == (
        "apply_candidate"
    )
    assert loaded.payload["collection_prefix"]["decision_count"] == 896
    assert loaded.payload["collection_prefix"]["gate_update_count"] == 0
    assert loaded.payload["claims"]["dynamic_collection_gate_execution_authorized"]
    assert not loaded.payload["claims"][
        "forced_common_batch_protocol_freeze_authorized"
    ]
    assert not loaded.payload["claims"]["campaign_execution_authorized"]
    protocol = json.loads(
        subject.relationship_product_horizon_dynamic_collection_prefix_protocol_path().read_text(
            encoding="utf-8"
        )
    )
    variants = (
        (("adaptive_lineage", "unseen"), True),
        (("upstream_scanner", "first_preaction_projection_count"), 111),
        (("collection_prefix", "decision_count_per_root"), 7),
        (("collection_prefix", "credit_applied_online"), True),
        (("runtime_order", "sealed_truth_passed_to_forecast_reader_gate_or_executor"), True),
        (("terminal_gates", "forced_common_batch_execution_authorized_on_success"), True),
        (("claims", "forced_common_batch_protocol_freeze_authorized"), True),
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
            subject.load_relationship_product_horizon_dynamic_collection_prefix_protocol(
                candidate
            )

    mutated = json.loads(json.dumps(protocol))
    mutated["collection_prefix"]["forced_schedule"] = "forbidden"
    candidate = tmp_path / "extra-key.json"
    candidate.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="collection_prefix fields do not match schema"):
        subject.load_relationship_product_horizon_dynamic_collection_prefix_protocol(
            candidate
        )


def test_dependency_loader_restores_112_scanner_rows_without_sealed_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads: list[pathlib.Path] = []
    original = pathlib.Path.read_bytes

    def tracked(path: pathlib.Path) -> bytes:
        resolved = path.resolve()
        reads.append(resolved)
        return original(path)

    monkeypatch.setattr(pathlib.Path, "read_bytes", tracked)
    dependencies = _dependencies()

    assert len(dependencies.expected_first_projections) == 112
    assert subject.sha256_json(dependencies.expected_first_projections) == (
        "5eee5690e366262eed331baadb217ef3259bf958e6d125687558f5f4bb4284d1"
    )
    assert tuple(
        item["root_sequence_index"]
        for item in dependencies.expected_first_projections
    ) == tuple(range(112))
    assert not any("sealed" in path.parts for path in reads)
    assert not any(path.name == "source_protocol.json" for path in reads)


def test_all_first_preactions_match_scanner_before_environment_scope() -> None:
    dependencies = _dependencies()

    async def run() -> tuple[dict[str, object], ...]:
        policy = dependencies.scanner_dependencies.frozen_policy
        authorization = subject.scan._authorization(
            protocol_id=dependencies.protocol.protocol_id,
            frozen_policy=policy,
        )
        projections = []
        for root_index, root in enumerate(dependencies.public_view.roots):
            decision = root.decision_sessions[0]
            owner = await subject.scan._post_onboarding_state(root)
            owner_sha = social_record_store_persistence_sha256(owner)
            preaction = await subject.scan.prepare_relationship_product_frozen_preaction(
                request=subject.scan._request(
                    subject_id=root.subject_id,
                    decision=decision,
                ),
                owner_persistence_snapshot=owner,
                forecast_runtime=dependencies.scanner_dependencies.forecast_runtime,
                frozen_policy=policy,
                executor_disposition=(
                    RelationshipProductExecutorDisposition.APPLY_CANDIDATE
                ),
                authorization=authorization,
                substrate_snapshot=subject.cal._placeholder_substrate(),
            )
            projection = subject._stable_first_preaction_projection(
                root_sequence_index=root_index,
                root=root,
                decision=decision,
                owner_input_sha256=owner_sha,
                preaction=preaction,
            )
            assert projection == dependencies.expected_first_projections[root_index]
            projections.append(dict(projection))
        return tuple(projections)

    projections = asyncio.run(run())
    assert len(projections) == 112
    assert subject.sha256_json(projections) == (
        "5eee5690e366262eed331baadb217ef3259bf958e6d125687558f5f4bb4284d1"
    )


def test_one_root_closes_eight_sequential_actual_action_settlements() -> None:
    dependencies = _dependencies()

    async def run() -> tuple[int, int, int]:
        root = dependencies.public_view.roots[0]
        owner = await subject.scan._post_onboarding_state(root)
        policy = dependencies.scanner_dependencies.frozen_policy
        authorization = subject.scan._authorization(
            protocol_id=dependencies.protocol.protocol_id,
            frozen_policy=policy,
        )
        scope: subject._SelectedBranchEnvironmentScope | None = None
        prior_post_sha: str | None = None
        handoffs = 0
        credits: set[str] = set()
        nonnoop = 0
        for decision in root.decision_sessions[:8]:
            input_sha = social_record_store_persistence_sha256(owner)
            if decision.decision_index > 0:
                assert input_sha == prior_post_sha
                handoffs += 1
            preaction = await subject.scan.prepare_relationship_product_frozen_preaction(
                request=subject.scan._request(
                    subject_id=root.subject_id,
                    decision=decision,
                ),
                owner_persistence_snapshot=owner,
                forecast_runtime=dependencies.scanner_dependencies.forecast_runtime,
                frozen_policy=policy,
                executor_disposition=(
                    RelationshipProductExecutorDisposition.APPLY_CANDIDATE
                ),
                authorization=authorization,
                substrate_snapshot=subject.cal._placeholder_substrate(),
            )
            if decision.decision_index == 0:
                projection = subject._stable_first_preaction_projection(
                    root_sequence_index=0,
                    root=root,
                    decision=decision,
                    owner_input_sha256=input_sha,
                    preaction=preaction,
                )
                assert projection == dependencies.expected_first_projections[0]
                scope = subject._SelectedBranchEnvironmentScope(
                    dependencies=dependencies
                )
            assert scope is not None
            branch = scope.settle(
                public_root=root,
                public_decision=decision,
                delivered_action_id=preaction.delivered_action_id,
            )
            action_turn = 4 + 2 * decision.decision_index
            settlement_input = replace(
                subject.cal._settlement_input(
                    subject_scope=root.subject_id,
                    decision=decision,
                    forecast_id=preaction.forecast.forecast_id,
                    selected_action_id=preaction.delivered_action_id,
                    environment_outcome=branch,
                    action_turn=action_turn,
                    credit_timestamp=subject._credit_timestamp(
                        0,
                        decision.decision_index,
                    ),
                ),
                apply_credit_to_gate=False,
            )
            settled = await settle_relationship_product_frozen_pulse(
                preaction=preaction,
                settlement_input=settlement_input,
            )
            assert not settled.credit_applied_to_gate
            assert settled.evaluation_gate_update_delta == 0
            assert settled.gate_checkpoint == policy.checkpoint
            assert settled.credit.abstract_action_id == preaction.delivered_action_id
            credits.add(settled.credit.record_id)
            nonnoop += int(
                preaction.delivered_action_id
                != subject.RelationshipAction.NEUTRAL_NOOP.value
            )
            owner = settled.owner_persistence_snapshot
            prior_post_sha = social_record_store_persistence_sha256(owner)
        return handoffs, len(credits), nonnoop

    handoffs, credit_count, nonnoop = asyncio.run(run())
    assert handoffs == 7
    assert credit_count == 8
    assert nonnoop >= 1


def test_public_selected_branch_facade_binds_canonical_view_and_typed_action() -> None:
    dependencies = _dependencies()
    environment = (
        subject.open_relationship_product_horizon_selected_branch_environment(
            source_v4_admission_root=_SOURCE_ROOT,
            reader_root=_READER_ROOT,
            theta0_v2_root=_THETA_ROOT,
            scanner_root=_SCANNER_ROOT,
            dynamic_collection_prefix_root=_DYNAMIC_ROOT,
            expected_dynamic_protocol_id=_DYNAMIC_PROTOCOL_ID,
            expected_dynamic_artifact_id=_DYNAMIC_ARTIFACT_ID,
        )
    )
    root = dependencies.public_view.roots[0]
    decision = root.decision_sessions[8]
    action = subject.RelationshipAction.STAY_PRESENT_WITHOUT_PROBE
    outcome = environment.settle(
        public_root=root,
        public_decision=decision,
        selected_action=action,
    )

    assert isinstance(
        environment,
        subject.RelationshipProductHorizonSelectedBranchEnvironment,
    )
    assert not hasattr(environment, "__dict__")
    for forbidden in (
        "dependencies",
        "source_v4_admission_root",
        "source_path",
        "condition",
        "policy",
        "preferred_action",
    ):
        assert not hasattr(environment, forbidden)
        assert not hasattr(outcome, forbidden)
    assert isinstance(
        outcome,
        subject.RelationshipProductHorizonSelectedBranchOutcome,
    )
    assert outcome.environment_subject_id == root.subject_id
    assert outcome.selected_action is action
    assert isinstance(outcome.typed_outcome, DialogueExternalOutcomeKind)
    assert {field.name for field in fields(outcome)} == {
        "environment_subject_id",
        "selected_action",
        "typed_outcome",
        "rendered_user_reaction",
        "environment_evidence_ref",
        "environment_version",
        "commitment_id",
    }
    assert outcome.to_payload()["selected_action_id"] == action.value
    assert outcome.to_payload()["typed_outcome_id"] == outcome.typed_outcome.value

    changed_decision = replace(
        decision,
        current_input=f"{decision.current_input} altered",
    )
    with pytest.raises(ValueError, match="public decision is not canonical"):
        environment.settle(
            public_root=root,
            public_decision=changed_decision,
            selected_action=action,
        )
    changed_root = replace(
        root,
        decision_sessions=(
            *root.decision_sessions[:8],
            changed_decision,
            *root.decision_sessions[9:],
        ),
    )
    with pytest.raises(ValueError, match="public root is not canonical"):
        environment.settle(
            public_root=changed_root,
            public_decision=changed_root.decision_sessions[8],
            selected_action=action,
        )
    with pytest.raises(TypeError, match="selected_action must be RelationshipAction"):
        environment.settle(
            public_root=root,
            public_decision=decision,
            selected_action=action.value,
        )


def test_public_selected_branch_opener_rejects_external_ids_before_sealed_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[bool] = []

    def forbidden_scope(**_: object) -> None:
        opened.append(True)
        raise AssertionError("sealed scope must not open after an external ID mismatch")

    monkeypatch.setattr(subject, "_SelectedBranchEnvironmentScope", forbidden_scope)
    common = {
        "source_v4_admission_root": _SOURCE_ROOT,
        "reader_root": _READER_ROOT,
        "theta0_v2_root": _THETA_ROOT,
        "scanner_root": _SCANNER_ROOT,
        "dynamic_collection_prefix_root": _DYNAMIC_ROOT,
    }
    with pytest.raises(ValueError, match="protocol ID drifted"):
        subject.open_relationship_product_horizon_selected_branch_environment(
            **common,
            expected_dynamic_protocol_id="0" * 64,
            expected_dynamic_artifact_id=_DYNAMIC_ARTIFACT_ID,
        )
    with pytest.raises(ValueError, match="artifact ID drifted"):
        subject.open_relationship_product_horizon_selected_branch_environment(
            **common,
            expected_dynamic_protocol_id=_DYNAMIC_PROTOCOL_ID,
            expected_dynamic_artifact_id="0" * 64,
        )
    assert opened == []


def test_artifact_lifecycle_is_create_only_manifest_last_and_read_only(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "dynamic-prefix"
    materialization_order_checked: list[bool] = []
    dependencies = _dependencies()

    def fake_load_dependencies(**_: pathlib.Path) -> subject._Dependencies:
        return dependencies

    async def fake_run(
        *,
        dependencies: subject._Dependencies,
        sink: subject.cal._TraceSink,
    ) -> subject._DynamicReplay:
        assert dependencies.protocol.protocol_id == (
            "47cea5fae3be13067492785893a7b621285ddce36efbe6c2eefae3331c50dbb2"
        )
        if isinstance(sink, subject.cal._FsyncTraceSink):
            trace_path = pathlib.Path(sink._handle.name)
            assert (trace_path.parent / "protocol.json").is_file()
            assert not (trace_path.parent / "manifest.json").exists()
            materialization_order_checked.append(True)
        sink.append(
            {
                "schema_version": subject.DYNAMIC_COLLECTION_PREFIX_TRACE_SCHEMA_VERSION,
                "record_type": "unit_artifact_plumbing",
                "protocol_id": dependencies.protocol.protocol_id,
            }
        )
        return _successful_replay()

    monkeypatch.setattr(subject, "_load_dependencies", fake_load_dependencies)
    monkeypatch.setattr(subject, "_run_dynamic_collection_prefix", fake_run)
    manifest = (
        subject.materialize_relationship_product_horizon_dynamic_collection_prefix(
            source_v4_admission_root=_SOURCE_ROOT,
            reader_root=_READER_ROOT,
            theta0_v2_root=_THETA_ROOT,
            scanner_root=_SCANNER_ROOT,
            output_dir=output,
            implementation_git_commit="a" * 40,
        )
    )

    assert materialization_order_checked == [True]
    assert {path.name for path in output.iterdir()} == {
        "protocol.json",
        "dynamic_collection_prefix.jsonl",
        "manifest.json",
    }
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in output.iterdir()
    }
    validated = subject.validate_relationship_product_horizon_dynamic_collection_prefix(
        source_v4_admission_root=_SOURCE_ROOT,
        reader_root=_READER_ROOT,
        theta0_v2_root=_THETA_ROOT,
        scanner_root=_SCANNER_ROOT,
        output_dir=output,
        expected_protocol_id=manifest["protocol_id"],
        expected_artifact_id=manifest["artifact_id"],
    )
    after = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in output.iterdir()
    }
    assert validated == manifest
    assert after == before

    with pytest.raises(FileExistsError, match="create-only"):
        subject.materialize_relationship_product_horizon_dynamic_collection_prefix(
            source_v4_admission_root=_SOURCE_ROOT,
            reader_root=_READER_ROOT,
            theta0_v2_root=_THETA_ROOT,
            scanner_root=_SCANNER_ROOT,
            output_dir=output,
            implementation_git_commit="a" * 40,
        )
    with pytest.raises(ValueError, match="artifact ID drifted"):
        subject.validate_relationship_product_horizon_dynamic_collection_prefix(
            source_v4_admission_root=_SOURCE_ROOT,
            reader_root=_READER_ROOT,
            theta0_v2_root=_THETA_ROOT,
            scanner_root=_SCANNER_ROOT,
            output_dir=output,
            expected_protocol_id=manifest["protocol_id"],
            expected_artifact_id="b" * 64,
        )

    extra = output / "extra.txt"
    extra.write_text("unexpected", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="output inventory drifted"):
            subject.validate_relationship_product_horizon_dynamic_collection_prefix(
                source_v4_admission_root=_SOURCE_ROOT,
                reader_root=_READER_ROOT,
                theta0_v2_root=_THETA_ROOT,
                scanner_root=_SCANNER_ROOT,
                output_dir=output,
                expected_protocol_id=manifest["protocol_id"],
                expected_artifact_id=manifest["artifact_id"],
            )
    finally:
        extra.unlink()

    trace = output / "dynamic_collection_prefix.jsonl"
    trace_raw = trace.read_bytes()
    trace.write_bytes(b"!" + trace_raw[1:])
    try:
        with pytest.raises(ValueError, match="stable trace bytes drifted"):
            subject.validate_relationship_product_horizon_dynamic_collection_prefix(
                source_v4_admission_root=_SOURCE_ROOT,
                reader_root=_READER_ROOT,
                theta0_v2_root=_THETA_ROOT,
                scanner_root=_SCANNER_ROOT,
                output_dir=output,
                expected_protocol_id=manifest["protocol_id"],
                expected_artifact_id=manifest["artifact_id"],
            )
    finally:
        trace.write_bytes(trace_raw)


def test_incomplete_or_all_noop_dynamic_chain_has_explicit_terminal_reasons() -> None:
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
        unique_selected_branch_commitment_count=0,
        prediction_error_count=0,
        credit_count=0,
        unique_credit_count=0,
        unique_settlement_count=0,
        unique_environment_evidence_ref_count=0,
        cold_checkpoint_unchanged_count=0,
        temporal_delivered_nonnoop_count=0,
    )
    assert reasons == (
        "completed_root_count_not_112",
        "onboarding_count_not_448",
        "preaction_count_not_896",
        "postaction_count_not_896",
        "first_preaction_exact_match_count_not_112",
        "later_owner_handoff_count_not_784",
        "owner_writeback_change_count_not_896",
        "selected_branch_resolution_count_not_896",
        "selected_branch_commitment_match_count_not_896",
        "unique_selected_branch_commitment_count_not_896",
        "prediction_error_count_not_896",
        "credit_count_not_896",
        "unique_credit_count_not_896",
        "unique_settlement_count_not_896",
        "unique_environment_evidence_ref_count_not_896",
        "cold_checkpoint_unchanged_count_not_896",
        "temporal_delivered_nonnoop_count_below_one",
    )
