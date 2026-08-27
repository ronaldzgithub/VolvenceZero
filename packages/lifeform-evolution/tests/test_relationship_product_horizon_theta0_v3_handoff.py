from __future__ import annotations

import copy
import inspect
import pathlib
import subprocess

import pytest

import lifeform_evolution.relationship_product_horizon_theta0_v3_handoff as subject
from lifeform_evolution import relationship_product_horizon_theta0_v3_bootstrap


_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
_THETA0_ROOT = _REPOSITORY_ROOT / (
    "artifacts/relationship_lab/relationship_product_horizon_theta0_v3_bootstrap_20260827_pf5c33f5c_iaf6cc60b_attempt03"
)
_VALIDATION_REPORT = _REPOSITORY_ROOT / (
    "artifacts/relationship_lab/"
    "relationship_product_horizon_theta0_v3_bootstrap_validation_pass_"
    "20260827_pf5c33f5c_iaf6cc60b_attempt03/report.json"
)


def test_protocol_freezes_receipt_first_full_graph_handoff_claim_ceiling() -> None:
    protocol = subject.load_relationship_product_horizon_theta0_v3_handoff_protocol()

    assert protocol.payload["evidence_tier"] == "development"
    assert protocol.theta0_input["artifact_id"] == ("dde0fc78d8957f78fdeeb23203635d772e419aab22d8ab02d311ce9e31109777")
    assert protocol.theta0_input["credit_count"] == 896
    assert protocol.theta0_input["apply_update_count"] == 896
    assert protocol.theta0_input["withhold_update_count"] == 0
    assert protocol.payload["compatibility_replay"] == {
        "historical_validate_existing_reexecuted": False,
        "historical_acceptance_receipt_must_be_bound_before_replay": True,
        "artifact_specific_current_compatibility": True,
        "general_backward_compatibility": False,
        "current_full_typed_federation_rehydration_required": True,
        "all_six_persisted_files_byte_exact_required": True,
        "before_after_no_write_fingerprint_required": True,
        "theta_json_only_handoff_forbidden": True,
        "compact_projection_bypass_forbidden": True,
        "pickle_or_untyped_deserialization_forbidden": True,
    }
    assert protocol.payload["claims_ceiling"]["theta_handoff_materialized"] is True
    assert protocol.payload["claims_ceiling"]["reader_qualified"] is False
    assert protocol.payload["claims_ceiling"]["learnable_effect"] is False
    assert protocol.payload["claims_ceiling"]["steerable_effect"] is False
    assert protocol.payload["claims_ceiling"]["source_v5_bound"] is False
    assert protocol.materialization_contract["compatibility_replay_reader_inference_count"] == 896
    assert protocol.materialization_contract["compatibility_replay_credit_derivation_count"] == 896
    assert protocol.materialization_contract["new_scientific_credit_observation_count"] == 0
    assert protocol.payload["current_execution_closure"] == [
        "packages",
        "scripts/run_relationship_product_horizon_theta0_v3_handoff.py",
    ]


def test_current_algorithm_owner_modules_resolve_inside_pinned_packages_tree() -> None:
    subject._verify_module_origins()


@pytest.mark.parametrize(
    ("section", "key"),
    [
        *(("authorization_contract", key) for key in subject._EXPECTED_AUTHORIZATION_CONTRACT),
        *(("materialization_contract", key) for key in subject._EXPECTED_MATERIALIZATION_CONTRACT),
    ],
)
def test_protocol_rejects_each_nested_contract_field_drift(
    tmp_path: pathlib.Path,
    section: str,
    key: str,
) -> None:
    protocol = subject.load_relationship_product_horizon_theta0_v3_handoff_protocol()
    payload = copy.deepcopy(dict(protocol.payload))
    nested = payload[section]
    assert type(nested) is dict
    value = nested[key]
    if type(value) is bool:
        nested[key] = not value
    elif value is None:
        nested[key] = "drift"
    elif type(value) is int:
        nested[key] = False
    elif type(value) is str:
        nested[key] = f"{value}-drift"
    elif type(value) is list:
        nested[key] = [*value, "unexpected.json"]
    else:
        raise AssertionError(f"unsupported protocol test value: {value!r}")
    path = tmp_path / f"{section}-{key}.json"
    path.write_bytes(subject._artifact_bytes(payload))

    with pytest.raises(ValueError, match=f"theta0 v3 handoff {section.replace('_', ' ')}"):
        subject.load_relationship_product_horizon_theta0_v3_handoff_protocol(path)


@pytest.mark.parametrize(
    ("section", "key", "replacement"),
    [
        ("theta0_input", "withhold_update_count", False),
        ("historical_acceptance", "report_raw_bytes", False),
        ("theta0_input", "unexpected", 0),
        ("historical_acceptance", "unexpected", False),
    ],
)
def test_protocol_rejects_upstream_pin_type_aliases_and_extra_keys(
    tmp_path: pathlib.Path,
    section: str,
    key: str,
    replacement: object,
) -> None:
    protocol = subject.load_relationship_product_horizon_theta0_v3_handoff_protocol()
    payload = copy.deepcopy(dict(protocol.payload))
    nested = payload[section]
    assert type(nested) is dict
    nested[key] = replacement
    path = tmp_path / f"{section}-{key}.json"
    path.write_bytes(subject._artifact_bytes(payload))

    with pytest.raises(ValueError, match=section):
        subject.load_relationship_product_horizon_theta0_v3_handoff_protocol(path)


def test_historical_acceptance_joins_report_and_all_six_root_files() -> None:
    protocol = subject.load_relationship_product_horizon_theta0_v3_handoff_protocol()

    receipt = subject._validate_historical_acceptance(
        protocol=protocol,
        theta0_root=_THETA0_ROOT,
        historical_validation_report_path=_VALIDATION_REPORT,
    )

    assert receipt.report_artifact_id == (
        "relationship-product-horizon-theta0-v3-bootstrap-validation-pass-"
        "report-sha256:4c3c414b695c0de376fd9c8bde4cbacbbb3a4432b0815482467e4c962364aab5"
    )
    assert receipt.report_raw_sha256 == ("cae85ed0f6481001772340ecd2f4a42ee2053aae7d75efaac6a90cee7b21af6c")
    assert receipt.theta_manifest["completed_root_count"] == 112
    assert receipt.theta_manifest["credit_count"] == 896


def test_historical_acceptance_rejects_any_report_byte_drift(
    tmp_path: pathlib.Path,
) -> None:
    protocol = subject.load_relationship_product_horizon_theta0_v3_handoff_protocol()
    raw = bytearray(_VALIDATION_REPORT.read_bytes())
    raw[-2] = ord(" ") if raw[-2] != ord(" ") else ord("\t")
    mutated = tmp_path / "report.json"
    mutated.write_bytes(bytes(raw))

    with pytest.raises(ValueError, match="report bytes drifted"):
        subject._validate_historical_acceptance(
            protocol=protocol,
            theta0_root=_THETA0_ROOT,
            historical_validation_report_path=mutated,
        )


def test_cross_commit_bridge_is_private_and_receipt_precedes_replay() -> None:
    assert (
        "_replay_relationship_product_horizon_theta0_v3_bundle_for_cross_commit_handoff"
        not in relationship_product_horizon_theta0_v3_bootstrap.__all__
    )
    source = inspect.getsource(subject._replay_and_authorize)
    acceptance = source.index("_validate_historical_acceptance")
    current_closure = source.index("_verify_current_execution_closure")
    full_replay = source.index("_replay_relationship_product_horizon_theta0_v3_bundle_for_cross_commit_handoff")
    constructor = source.index("RelationshipProductV2CondensedTheta0FrozenPulseAuthorization(")
    assert acceptance < current_closure < full_replay < constructor
    assert "bundle.matched_transitions" in source
    assert "theta0_artifact.json" not in source


def test_current_closure_can_follow_docs_head_but_never_drift_owned_blobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = subject.load_relationship_product_horizon_theta0_v3_handoff_protocol()
    commit = "a" * 40
    current_head = "c" * 40
    diff_return_code = 0

    def fake_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        del check
        if args == ("rev-parse", "--show-toplevel"):
            stdout = str(subject._REPOSITORY_ROOT)
        elif args == ("rev-parse", "--verify", f"{commit}^{{commit}}"):
            stdout = commit
        elif args == ("rev-parse", "HEAD"):
            stdout = current_head
        elif args[:3] == ("diff", "--quiet", commit):
            return subprocess.CompletedProcess(args, diff_return_code, "", "")
        elif args[:4] == ("ls-files", "--others", "--exclude-standard", "--"):
            stdout = ""
        elif len(args) == 3 and args[:2] == ("cat-file", "-t"):
            relative = args[2].split(":", 1)[1]
            stdout = subject._CURRENT_EXECUTION_CLOSURE_OBJECT_TYPES[relative]
        elif len(args) == 2 and args[0] == "rev-parse" and args[1].startswith(f"{commit}:"):
            stdout = "b" * 40
        else:
            raise AssertionError(f"unexpected git invocation: {args!r}")
        return subprocess.CompletedProcess(args, 0, stdout, "")

    monkeypatch.setattr(subject, "_run_git", fake_git)
    receipt = subject._verify_current_execution_closure(
        protocol=protocol,
        implementation_git_commit=commit,
        require_current_head=False,
    )
    assert receipt.implementation_git_commit == commit

    with pytest.raises(ValueError, match="does not match HEAD"):
        subject._verify_current_execution_closure(
            protocol=protocol,
            implementation_git_commit=commit,
            require_current_head=True,
        )

    diff_return_code = 1
    with pytest.raises(ValueError, match="differs from commit"):
        subject._verify_current_execution_closure(
            protocol=protocol,
            implementation_git_commit=commit,
            require_current_head=False,
        )
