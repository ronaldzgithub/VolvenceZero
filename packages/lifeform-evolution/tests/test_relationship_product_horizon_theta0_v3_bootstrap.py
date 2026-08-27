from __future__ import annotations

import ast
from collections import Counter
import inspect
import json
import pathlib
import subprocess

import pytest

import lifeform_evolution.relationship_product_horizon_theta0_v3_bootstrap as subject
from lifeform_domain_emogpt.relationship_action_gate_v2 import (
    RELATIONSHIP_ACTION_GATE_V2_FEATURE_ORDER,
    RELATIONSHIP_ACTION_GATE_V2_THRESHOLD_RULE,
    RelationshipActionGateV2AssignmentRole,
)
from lifeform_evolution import relationship_product_horizon_theta0_calibration as cal


_SOURCE_ROOT = pathlib.Path(
    "artifacts/relationship_lab/"
    "relationship_product_horizon_source_v4_admission_20260826_b3988b21"
)
_READER_ROOT = pathlib.Path(
    "artifacts/relationship_lab/"
    "relationship_product_horizon_development_reader_20260826_"
    "pa1ea1e30fd7b_ce8272ce7d3da"
)


def _dependencies() -> subject._Dependencies:
    return subject._load_dependencies(
        source_v4_admission_root=_SOURCE_ROOT,
        reader_root=_READER_ROOT,
    )


def _checkout() -> subject._ImplementationCheckoutReceipt:
    return subject._ImplementationCheckoutReceipt(
        implementation_git_commit="a" * 40,
        owned_blob_ids=tuple(
            (path, f"{index + 1:040x}")
            for index, path in enumerate(subject._IMPLEMENTATION_OWNED_PATHS)
        ),
    )


def test_protocol_freezes_non_rehearsal_adaptive_claim_ceiling(
    tmp_path: pathlib.Path,
) -> None:
    loaded = subject.load_relationship_product_horizon_theta0_v3_bootstrap_protocol()
    assert loaded.raw_sha256 == subject.THETA0_V3_BOOTSTRAP_PROTOCOL_RAW_SHA256
    assert loaded.payload["adaptive_lineage"]["rehearsal_execution_authorized"] is False
    assert loaded.payload["adaptive_lineage"]["source_v4_unseen_evidence"] is False
    assert loaded.payload["implementation_lineage"]["owned_paths"] == list(
        subject._IMPLEMENTATION_OWNED_PATHS
    )
    assert loaded.payload["development_reader"]["condition_reader_qualified"] is False
    assert loaded.payload["gate_v2"]["feature_order"] == list(
        RELATIONSHIP_ACTION_GATE_V2_FEATURE_ORDER
    )
    assert (
        loaded.payload["gate_v2"]["threshold_rule"]
        == RELATIONSHIP_ACTION_GATE_V2_THRESHOLD_RULE
    )
    assert loaded.payload["claims"]["learnable_effect"] is False
    assert loaded.payload["claims"]["steerable_effect"] is False
    assert loaded.payload["claims"]["campaign_execution_authorized"] is False

    payload = json.loads(
        subject.relationship_product_horizon_theta0_v3_bootstrap_protocol_path().read_text(
            encoding="utf-8"
        )
    )
    payload["federated_schedule"]["entry_count"] = 896.0
    mutated = tmp_path / "mutated.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="raw bytes drifted"):
        subject.load_relationship_product_horizon_theta0_v3_bootstrap_protocol(
            mutated
        )


def test_public_only_schedule_and_seed_match_every_frozen_pin() -> None:
    dependencies = _dependencies()
    seed = subject._build_seed(dependencies.protocol)
    parent, children = subject._build_federated_schedule(dependencies.public_view)
    subject._validate_frozen_public_inputs(
        dependencies=dependencies,
        seed_artifact=seed,
        parent_schedule=parent,
    )

    assert len(children) == 112
    assert len(parent.segments) == 112
    assert len(parent.flattened_entries) == 896
    assert tuple(item.global_start_index for item in parent.segments) == tuple(
        range(0, 896, 8)
    )
    assert len({item.artifact_id for item in children}) == 112
    assert len({item.schedule_scope_id for item in children}) == 112
    assert Counter(item.assignment_role for item in parent.flattened_entries) == {
        RelationshipActionGateV2AssignmentRole.CANDIDATE: 448,
        RelationshipActionGateV2AssignmentRole.NEUTRAL_NOOP: 448,
    }
    for child in children:
        assert Counter(item.assignment_role for item in child.entries) == {
            RelationshipActionGateV2AssignmentRole.CANDIDATE: 4,
            RelationshipActionGateV2AssignmentRole.NEUTRAL_NOOP: 4,
        }


def test_parent_schedule_writer_mints_only_row_zero_durable_capability(
    tmp_path: pathlib.Path,
) -> None:
    dependencies = _dependencies()
    parent, _children = subject._build_federated_schedule(dependencies.public_view)
    raw = subject._canonical_bytes(parent.to_payload())
    path = tmp_path / "parent_schedule.json"
    subject._write_and_reopen_exact(path, raw)
    receipt = subject._durable_parent_receipt(
        protocol_id=dependencies.protocol.protocol_id,
        implementation_git_commit="a" * 40,
        implementation_checkout=_checkout(),
        parent_schedule=parent,
        persisted_raw=path.read_bytes(),
    )
    sink = cal._MemoryTraceSink()
    ledger = subject._Ledger(sink)
    active = subject._activate_parent_receipt(
        ledger=ledger,
        durable=receipt,
        protocol_id=dependencies.protocol.protocol_id,
        implementation_git_commit="a" * 40,
        implementation_checkout=_checkout(),
        parent_schedule=parent,
    )

    first = json.loads(ledger.raw_bytes.decode("utf-8"))
    assert first["record_type"] == "parent_schedule_durable"
    assert first["physical_sequence_index"] == 0
    assert first["row_id"] == active.ledger_row_id
    assert first["forecast_count_before_receipt"] == 0
    assert first["outcome_count_before_receipt"] == 0
    assert first["child_transition_count_before_receipt"] == 0
    subject._require_active_parent(
        active=active,
        ledger=ledger,
        protocol_id=dependencies.protocol.protocol_id,
        implementation_git_commit="a" * 40,
        parent_schedule=parent,
    )
    with pytest.raises(ValueError, match="first ledger row"):
        subject._activate_parent_receipt(
            ledger=ledger,
            durable=receipt,
            protocol_id=dependencies.protocol.protocol_id,
            implementation_git_commit="a" * 40,
            implementation_checkout=_checkout(),
            parent_schedule=parent,
        )
    with pytest.raises(FileExistsError):
        subject._write_and_reopen_exact(path, raw)


def test_terminal_gate_seals_cap_zero_information_and_partial_parent_as_failures() -> None:
    baseline = {
        "completed_root_count": 112,
        "onboarding_count": 448,
        "onboarding_write_change_count": 448,
        "preaction_count": 896,
        "postaction_count": 896,
        "owner_handoff_count": 784,
        "owner_writeback_change_count": 896,
        "child_collection_count": 112,
        "unique_child_collection_count": 112,
        "unique_credit_count": 896,
        "federated_credit_count": 896,
        "apply_child_batch_count": 112,
        "apply_child_transition_count": 0,
        "apply_credit_count": 896,
        "apply_atomic_commit_count": 1,
        "apply_update_count_delta": 896,
        "apply_informative_update_count_delta": 1,
        "apply_cap_hit_count": 0,
        "withhold_child_transition_count": 0,
        "withhold_atomic_commit_count": 0,
        "withhold_update_count_delta": 0,
        "withhold_checkpoint_unchanged": True,
        "terminal_parameter_finite": True,
        "terminal_parameter_nonzero": True,
    }
    assert subject._terminal_failure_reasons(**baseline) == ()
    variants = {
        "apply_cap_hit_count": (1, "apply_cap_hit_count_not_zero"),
        "apply_informative_update_count_delta": (
            0,
            "apply_informative_update_count_below_one",
        ),
        "child_collection_count": (111, "child_collection_count_not_112"),
        "onboarding_write_change_count": (
            447,
            "onboarding_write_change_count_not_448",
        ),
        "owner_writeback_change_count": (
            895,
            "owner_writeback_change_count_not_896",
        ),
        "apply_child_transition_count": (
            1,
            "apply_child_transition_count_not_zero",
        ),
        "terminal_parameter_nonzero": (False, "terminal_parameter_all_zero"),
    }
    for field, (replacement, expected_reason) in variants.items():
        candidate = {**baseline, field: replacement}
        assert expected_reason in subject._terminal_failure_reasons(**candidate)


def test_owner_has_one_parent_commit_and_literal_none_root_reset_without_legacy_inputs() -> None:
    source = inspect.getsource(subject)
    assert "commit_relationship_product_v2_matched_gate_transitions" not in source
    assert "commit_relationship_product_v2_segmented_matched_gate_transitions" not in source
    assert source.count("commit_relationship_product_v2_federated_matched_gate_transitions(") == 1
    signature = inspect.signature(
        subject.materialize_relationship_product_horizon_theta0_v3_bootstrap
    )
    assert tuple(signature.parameters) == (
        "source_v4_admission_root",
        "reader_root",
        "output_dir",
        "implementation_git_commit",
    )
    assert not {
        "theta0_v2_root",
        "scanner_root",
        "dynamic_root",
        "forced_common_batch_root",
    } & set(signature.parameters)

    tree = ast.parse(inspect.getsource(subject._run_bootstrap))
    literal_resets = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "owner_persistence"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and node.value.value is None
    ]
    assert len(literal_resets) == 1


def test_public_dependency_load_does_not_open_sealed_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_environment(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("sealed environment opened during public dependency load")

    monkeypatch.setattr(subject, "_SelectedBranchEnvironmentScope", forbidden_environment)
    dependencies = _dependencies()
    assert len(dependencies.public_view.roots) == 112
    assert dependencies.protocol.payload["causal_firewall"]["model_output_count"] == 0


def test_git_lineage_requires_head_and_clean_materialization_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = subject.load_relationship_product_horizon_theta0_v3_bootstrap_protocol()
    commit = "a" * 40
    head = commit
    dirty = False

    def fake_git(
        *args: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        del check
        if args == ("rev-parse", "--show-toplevel"):
            stdout = str(subject._REPOSITORY_ROOT)
        elif args == ("rev-parse", "--verify", f"{commit}^{{commit}}"):
            stdout = commit
        elif args == ("rev-parse", "HEAD"):
            stdout = head
        elif args[:2] == ("ls-files", "--"):
            stdout = "\n".join(subject._IMPLEMENTATION_OWNED_PATHS)
        elif args[:3] == ("diff", "--quiet", commit):
            return subprocess.CompletedProcess(args, 0, "", "")
        elif args[:3] == ("status", "--porcelain=v1", "--untracked-files=all"):
            stdout = "?? packages/drift.py\n" if dirty else ""
        elif (
            len(args) == 2
            and args[0] == "rev-parse"
            and args[1].startswith(f"{commit}:")
        ):
            stdout = "b" * 40
        else:
            raise AssertionError(f"unexpected git invocation: {args!r}")
        return subprocess.CompletedProcess(args, 0, stdout, "")

    monkeypatch.setattr(subject, "_run_git", fake_git)
    receipt = subject._verify_implementation_checkout(
        protocol=protocol,
        implementation_git_commit=commit,
        require_current_head=True,
    )
    assert receipt.implementation_git_commit == commit
    assert len(receipt.owned_blob_ids) == 3

    head = "c" * 40
    with pytest.raises(ValueError, match="does not match HEAD"):
        subject._verify_implementation_checkout(
            protocol=protocol,
            implementation_git_commit=commit,
            require_current_head=True,
        )

    head = commit
    dirty = True
    with pytest.raises(ValueError, match="differs from frozen HEAD"):
        subject._verify_implementation_checkout(
            protocol=protocol,
            implementation_git_commit=commit,
            require_current_head=True,
        )


def test_actual_child_transition_counts_drive_terminal_serializers() -> None:
    observation = subject._ChildTransitionObservation(
        apply_count=1,
        withhold_count=0,
    )
    assert observation.total_count == 1
    assert observation.accepted_lineage_has_no_child_transition is False
    assert '"child_transition_count": 0' not in inspect.getsource(
        subject._transition_bundle_payload
    )
    assert '"child_transition_count": 0' not in inspect.getsource(
        subject._build_manifest
    )
    bootstrap_source = inspect.getsource(subject._run_bootstrap)
    assert "assignment.receipt_id" not in bootstrap_source
    assert "preaction.forced_exposure.assignment_receipt_id" in bootstrap_source


def test_output_root_must_be_disjoint_and_closed_trace_must_match(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    reader_root = tmp_path / "reader"
    source_root.mkdir()
    reader_root.mkdir()
    with pytest.raises(ValueError, match="disjoint"):
        subject._require_disjoint_output_root(
            output_root=source_root / "new-output",
            input_roots=(source_root, reader_root),
        )
    subject._require_disjoint_output_root(
        output_root=tmp_path / "output",
        input_roots=(source_root, reader_root),
    )
    with pytest.raises(ValueError, match="disjoint"):
        subject._require_disjoint_output_root(
            output_root=subject._REPOSITORY_ROOT / "packages" / "theta0-output",
            input_roots=subject._OUTPUT_FORBIDDEN_REPOSITORY_ROOTS,
        )

    def _different_volume(_paths: object) -> str:
        raise ValueError

    monkeypatch.setattr(subject.os.path, "commonpath", _different_volume)
    subject._require_disjoint_output_root(
        output_root=tmp_path / "other-volume-output",
        input_roots=(source_root, reader_root),
    )

    ledger = subject._Ledger(cal._MemoryTraceSink())
    ledger.append(record_type="test", payload={"value": 1})
    trace = tmp_path / "trace.jsonl"
    trace.write_bytes(ledger.raw_bytes)
    subject._require_closed_trace_exact(path=trace, ledger=ledger)
    trace.write_bytes(ledger.raw_bytes + b"drift")
    with pytest.raises(OSError, match="closed trace bytes drifted"):
        subject._require_closed_trace_exact(path=trace, ledger=ledger)
