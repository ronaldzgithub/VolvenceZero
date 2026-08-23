from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Any, Callable

import pytest

import lifeform_evolution.relationship_lab_p4_long_context_causal_campaign as owner
from lifeform_evolution.relationship_lab_p4_long_context_causal_campaign import (
    load_relationship_p4_long_context_external_anchor_request_protocol,
    prepare_relationship_p4_long_context_external_anchor_request,
    validate_relationship_p4_long_context_external_anchor_request,
)


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_PACKAGE_ROOT = _REPO_ROOT / "packages" / "lifeform-evolution"
_OWNER_SOURCE = _PACKAGE_ROOT / "src" / "lifeform_evolution" / "relationship_lab_p4_long_context_causal_campaign.py"
_PROTOCOL_PATH = (
    _PACKAGE_ROOT
    / "src"
    / "lifeform_evolution"
    / "protocols"
    / "relationship_p4_long_context_v4_external_publication_anchor_v1.json"
)
_CLI_SOURCE = _REPO_ROOT / "scripts" / "run_relationship_lab_p4_long_context_external_publication_anchor.py"
_V4A_PLANNING = (
    _REPO_ROOT / "artifacts" / "relationship_lab" / "p4_independent_long_context_v4a_zero_output_planning_20260823"
)
_V3_PREPARATION = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "p4_independent_long_context_causal_campaign_design_prereg_v3_20260823"
)
_V2_ADMISSION = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "p4_independent_long_context_power_admission_v2_under_specified_20260823"
)
_SOURCE_PREFLIGHT_ARTIFACT = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "p4_independent_long_context_source_opportunity_preflight_v1_20260823"
)
_ANCHOR_REQUEST_ARTIFACT = (
    _REPO_ROOT / "artifacts" / "relationship_lab" / "p4_independent_long_context_external_anchor_request_v1_20260823"
)

_REQUEST_FILE = "external_publication_anchor_request.json"
_MANIFEST_FILE = "manifest.json"

_EXPECTED_PROTOCOL_ID = "dedfc7ff42f1be0030cdfbe64fd6b1d6dc868adf9db6a9f1150883a9a96a4bee"
_EXPECTED_PROTOCOL_RAW = "38ce85d479c4359c252de8e5293ca1c15d886c5e1757610435aad136feeca8c6"
_EXPECTED_PROTOCOL_BYTE_COUNT = 16006
_EXPECTED_REQUEST_ID = "7897e3285299eac33385f69fb560a7d68e9f3316fdaf200f27cfa9bbfda489d1"
_EXPECTED_REQUEST_RAW = "0d5147cdf11db9fcaaa793bd9bf9bf8bfb6d07511f2307f5e77cdc0dfd263057"
_EXPECTED_REQUEST_BYTE_COUNT = 12115
_EXPECTED_ARTIFACT_ID = "5496fa80bba07c6b2234e0e2ca9293111d7ed6edf0a676ee4f561a7893c22900"
_EXPECTED_MANIFEST_RAW = "17496a50035d5b1dd3849455940ef2bcf4c6c4089dc2093bb194acb14a514125"
_EXPECTED_MANIFEST_BYTE_COUNT = 1307
_EXPECTED_SOURCE_PROTOCOL_ID = "47bcf6561be1ace0698cc0f96e2e7e35701f46d15baac9eb87ad1d662576494a"
_EXPECTED_SOURCE_ARTIFACT_ID = "8a36d2de9077bb5550db8018338eded27b6ce30d77eea17739ffe35b73e00a99"
_EXPECTED_SOURCE_PROJECTION_ID = "b8b7823a6fd2c7ad706c4ffa143438b730da667c26a925f0be87df14212e6f1b"
_EXPECTED_SOURCE_CERTIFICATE_ID = "64d879c4f41ca873f8e40f0344234771343f6efee229b668914b61d31c96c95a"

_EXPECTED_SECTION_SHA256 = {
    "schema_version": "5ce63abdc943432e02d932d8f38036d2c630982d8dcab493438ce530a038747f",
    "protocol_id_contract": "34c79b317bf8a9de446937cbfe2255d9409a7d831323787de601a2001142f2fe",
    "frozen_at_utc": "cf35b37fdcb58ba82055529a45fa9cadff04f15877d3bf810ab4f058eb137be1",
    "owner": "9d3c16023430115cb2ee14603d6af3209e6e7fc25912dfcb1e408b296b8f34a4",
    "supersession": "d356c00957a170fabb20a218302f5fa21b3e2299984d6ac48f49dcd2d9257046",
    "question": "3bf6fc3e45a1f723e647cc06a3e5e253d09b17819af4c536bb6f6059d1c6c0d6",
    "input_lineage": "ae1a18aeaf917fcd8a4c6dbb873f4bd06f3814f576e0a185309e3d50fb4c4920",
    "anchor_stage": "baa251cbcceb889089939838a3bd77a4bfd1a28eb2d6432e7130babad7759a76",
    "publication_subject_contract": "bfa95ccb8f5dd836269c7b4c84ac1727010d60c42c3a1228c073a87bcf7ec112",
    "publication_target_contract": "41798c349218be5e58e324e434f35f51379854503e12e00f654f9983ad0eab7d",
    "self_publication_binding": "06e28786b66320232f93d4634b2fee9bba6c5a6bf3fe13034ea83a54fd51769f",
    "future_receipt_requirements": "5b5ec66c404c20674abdd74dc4170ac87a71fa338e01bcf6b42ff70e7a02bd68",
    "authorization_firewall": "9740ae863e28a3028ec13e3d5c4e6989afd1c3abb1cda8b635e5fc9635fb3bb2",
    "zero_output_firewall": "a1874b50d34dfea446bddbad68c6887842ab4a96974d335ff02c6938283cf959",
    "terminal": "02137f0a4009feb3982d5aac2e378e679bcdae8e08a0deeed20031a00db7ce7c",
    "claim_boundary": "1bda30c5cd3efad160b41455ed6af1dd1fd570a0fe49ac365aac846623e5f883",
}

_EXPECTED_REQUEST_KEYS = {
    "anchor_stage",
    "authorization_firewall",
    "claim_boundary",
    "frozen_at_utc",
    "future_receipt_requirements",
    "identity",
    "publication_subjects",
    "publication_target",
    "request_id",
    "request_id_contract",
    "schema_version",
    "self_publication_binding",
    "terminal",
    "zero_output_firewall",
}
_EXPECTED_MANIFEST_KEYS = {
    "CUDA_planner_authorized",
    "CUDA_run_count",
    "Git_commit_count",
    "Git_push_count",
    "anchor_request_protocol_id",
    "artifact_id",
    "claim_boundary",
    "empirical_outcome_count",
    "external_anchor_admitted",
    "external_publication_anchor_present",
    "external_request_dispatched",
    "files",
    "model_output_authorized",
    "model_output_count",
    "network_request_count",
    "publication_performed",
    "publication_request_contract_frozen",
    "request_id",
    "schema_version",
    "source_execution_authorized",
    "source_structural_inventory_artifact_count",
    "status",
    "structural_inventory_materialization_authorized",
    "tuple_feasibility_authorized",
}


def test_protocol_raw_canonical_and_every_section_are_literal_pins() -> None:
    protocol_bytes = _PROTOCOL_PATH.read_bytes()
    raw = _strict_json(_PROTOCOL_PATH)

    assert len(protocol_bytes) == _EXPECTED_PROTOCOL_BYTE_COUNT
    assert hashlib.sha256(protocol_bytes).hexdigest() == _EXPECTED_PROTOCOL_RAW
    assert hashlib.sha256(_canonical_bytes(raw)).hexdigest() == _EXPECTED_PROTOCOL_ID
    assert set(raw) == set(_EXPECTED_SECTION_SHA256)
    assert {
        section: hashlib.sha256(_canonical_bytes(value)).hexdigest() for section, value in raw.items()
    } == _EXPECTED_SECTION_SHA256
    assert dict(owner._V4_EXTERNAL_ANCHOR_REQUEST_PROTOCOL_SECTION_SHA256_V1) == (_EXPECTED_SECTION_SHA256)
    assert owner.P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_PROTOCOL_ID_V1 == _EXPECTED_PROTOCOL_ID
    assert owner.P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_PROTOCOL_RAW_SHA256_V1 == (_EXPECTED_PROTOCOL_RAW)
    assert owner.P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_ID_V1 == _EXPECTED_REQUEST_ID
    assert owner.P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_ARTIFACT_ID_V1 == _EXPECTED_ARTIFACT_ID

    loaded = load_relationship_p4_long_context_external_anchor_request_protocol()
    assert loaded.protocol_id == _EXPECTED_PROTOCOL_ID
    assert loaded.source_preflight_protocol_id == _EXPECTED_SOURCE_PROTOCOL_ID
    assert loaded.source_preflight_artifact_id == _EXPECTED_SOURCE_ARTIFACT_ID
    assert loaded.publication_subject_count == 5
    assert loaded.provider == "github_public_gist_first_revision_v1"
    assert loaded.expected_owner_login == "ronaldzgithub"
    assert loaded.required_filename == "volvence_p4_7_source_opportunity_a0_anchor_request.json"
    assert loaded.publication_request_contract_frozen is True
    assert loaded.external_publication_anchor_present is False
    assert loaded.structural_inventory_materialization_authorized is False

    target = raw["publication_target_contract"]
    assert target["required_visibility"] == "public"
    assert target["unauthenticated_read_required"] is True
    assert target["new_gist_required"] is True
    assert target["first_revision_required"] is True
    assert target["required_parent_count"] == 0
    assert target["required_exact_file_count"] == 1
    assert target["mutable_latest_URL_is_authority"] is False
    for key in (
        "actual_gist_id",
        "actual_revision_oid",
        "actual_raw_permalink",
        "actual_HTML_permalink",
    ):
        assert target[key] is None


def test_receipt_contract_requires_one_public_object_graph_and_splits_observer_from_publisher_auth() -> None:
    raw = _strict_json(_PROTOCOL_PATH)
    receipt = raw["future_receipt_requirements"]
    same_object_graph_requirements = {
        "same_gist_owner_id_revision_and_filename_identity_join_required",
        "revision_commit_must_be_loaded_from_the_same_gist_and_have_zero_parents",
        "revision_tree_must_have_exactly_one_entry_with_mode_100644_and_required_filename",
        "tree_entry_blob_OID_must_equal_Git_blob_SHA1_of_observed_request_bytes",
        "revision_pinned_raw_final_URL_API_identity_and_HTML_identity_must_resolve_to_the_same_gist_revision_and_filename",
        "observed_request_bytes_must_equal_local_request_payload_bytes_with_exact_SHA256_and_byte_count",
    }
    assert all(receipt[field] is True for field in same_object_graph_requirements)
    assert receipt["all_observed_HTTP_URLs_must_use_HTTPS_without_userinfo_or_nondefault_port"] is True
    assert receipt["required_empty_Gist_description_must_be_observed"] is True

    assert receipt["observer_requests_must_send_no_Authorization_header_and_no_Cookie_header"] is True
    assert receipt["publisher_creation_authentication_requires_separate_explicit_user_authority"] is True
    assert receipt["publisher_credentials_headers_cookies_and_tokens_must_never_be_serialized"] is True
    assert "authentication_used_must_be_false" not in receipt
    assert receipt["receipt_alone_may_authorize_source_materialization"] is False

    target = raw["publication_target_contract"]
    assert target["required_description"] == ""
    assert target["HTTPS_required"] is True
    assert target["URL_userinfo_forbidden"] is True
    assert target["nondefault_port_forbidden"] is True
    binding = raw["self_publication_binding"]
    assert binding["prepare_upstream_repo_relative_roots"] == {
        "source_preflight": (
            "artifacts/relationship_lab/p4_independent_long_context_source_opportunity_preflight_v1_20260823"
        ),
        "v4a_planning": ("artifacts/relationship_lab/p4_independent_long_context_v4a_zero_output_planning_20260823"),
        "v3_preparation": (
            "artifacts/relationship_lab/p4_independent_long_context_causal_campaign_design_prereg_v3_20260823"
        ),
        "v2_admission": (
            "artifacts/relationship_lab/p4_independent_long_context_power_admission_v2_under_specified_20260823"
        ),
    }
    assert (
        binding["prepare_must_read_all_upstream_lineage_artifacts_from_their_frozen_canonical_repository_roots"] is True
    )


def test_ephemeral_derivation_objects_are_not_persisted_materialization() -> None:
    raw = _strict_json(_PROTOCOL_PATH)
    zero = raw["zero_output_firewall"]
    assert (
        zero[
            "all_materialization_counts_refer_to_persisted_or_published_artifact_rows_not_ephemeral_in_memory_exact_derivation_objects"
        ]
        is True
    )
    assert (
        zero["ephemeral_in_memory_structural_derivation_objects_are_not_source_content_or_persistent_materialization"]
        is True
    )
    assert all(value == 0 for key, value in zero.items() if key.endswith("_count"))
    assert "Exact ephemeral in-memory derivations" in raw["claim_boundary"]


def test_custom_protocol_with_same_canonical_json_but_different_bytes_is_rejected(
    tmp_path: pathlib.Path,
) -> None:
    variant = tmp_path / "byte-variant.json"
    exact_bytes = _PROTOCOL_PATH.read_bytes()
    variant.write_bytes(exact_bytes + b"\n")
    assert _strict_json(variant) == _strict_json(_PROTOCOL_PATH)
    assert hashlib.sha256(_canonical_bytes(_strict_json(variant))).hexdigest() == _EXPECTED_PROTOCOL_ID
    assert hashlib.sha256(variant.read_bytes()).hexdigest() != _EXPECTED_PROTOCOL_RAW
    with pytest.raises(ValueError, match="protocol raw bytes drift"):
        load_relationship_p4_long_context_external_anchor_request_protocol(variant)


def test_source_preflight_replay_binds_five_subject_bytes_sha256_and_git_blobs() -> None:
    protocol, raw, source_certificate, subjects = _external_inputs()
    lineage = raw["input_lineage"]

    assert protocol.protocol_id == _EXPECTED_PROTOCOL_ID
    assert source_certificate.artifact_id == _EXPECTED_SOURCE_ARTIFACT_ID
    assert source_certificate.contract_projection_id == _EXPECTED_SOURCE_PROJECTION_ID
    assert source_certificate.certificate_id == _EXPECTED_SOURCE_CERTIFICATE_ID
    assert source_certificate.artifact_id == lineage["source_preflight_artifact_id"]
    assert source_certificate.contract_projection_id == lineage["source_preflight_contract_projection_id"]
    assert source_certificate.certificate_id == lineage["source_preflight_certificate_id"]
    assert source_certificate.source_opportunity_stage_completed is False
    assert source_certificate.source_structural_inventory_materialized is False
    assert source_certificate.current_source_execution_authorized is False
    assert source_certificate.model_output_authorized is False
    assert source_certificate.cuda_planner_authorized is False

    expected_subjects = raw["publication_subject_contract"]["ordered_subjects"]
    assert len(subjects) == len(expected_subjects) == 5
    assert [dict(item) for item in subjects] == expected_subjects
    seen_paths: set[str] = set()
    seen_casefolded: set[str] = set()
    for subject in expected_subjects:
        repo_path = subject["repo_relative_posix_path"]
        pure_path = pathlib.PurePosixPath(repo_path)
        assert not pure_path.is_absolute()
        assert str(pure_path) == repo_path
        assert not {"", ".", ".."}.intersection(pure_path.parts)
        assert "\\" not in repo_path and ":" not in repo_path and "%" not in repo_path
        assert repo_path not in seen_paths
        assert repo_path.casefold() not in seen_casefolded
        seen_paths.add(repo_path)
        seen_casefolded.add(repo_path.casefold())

        subject_path = _REPO_ROOT.joinpath(*pure_path.parts)
        payload = subject_path.read_bytes()
        assert len(payload) == subject["byte_count"]
        assert hashlib.sha256(payload).hexdigest() == subject["raw_sha256"]
        assert _git_blob_oid_sha1(payload) == subject["expected_git_blob_oid_sha1"]
        assert subject["expected_git_mode"] == "100644"
        assert subject["expected_git_object_type"] == "blob"
        assert not payload.startswith(b"version https://git-lfs.github.com/spec/v1")

    semantic_identities = {
        "source_preflight_protocol": _EXPECTED_SOURCE_PROTOCOL_ID,
        "source_derivation_helper": hashlib.sha256(
            (
                _PACKAGE_ROOT
                / "src"
                / "lifeform_evolution"
                / "relationship_lab_p4_long_context_source_opportunity_derivation.py"
            ).read_bytes()
        ).hexdigest(),
        "source_preflight_contract_projection": _strict_json(
            _SOURCE_PREFLIGHT_ARTIFACT / "source_opportunity_contract_projection.json"
        )["contract_projection_id"],
        "source_preflight_certificate": _strict_json(
            _SOURCE_PREFLIGHT_ARTIFACT / "source_opportunity_preflight_certificate.json"
        )["certificate_id"],
        "source_preflight_manifest": _strict_json(_SOURCE_PREFLIGHT_ARTIFACT / _MANIFEST_FILE)["artifact_id"],
    }
    assert {item["role"]: item["semantic_identity"] for item in expected_subjects} == (semantic_identities)


@pytest.mark.parametrize(
    ("repo_path", "expected_eol", "expected_raw_sha256", "expected_filtered_oid"),
    (
        (
            "packages/lifeform-evolution/src/lifeform_evolution/"
            "relationship_lab_p4_long_context_v4_planning_derivation.py",
            "lf",
            "bf38e7ab89c56bdae8844f533cac077443d157a793c698adbb11a9591e32a0ef",
            None,
        ),
        (
            "packages/lifeform-evolution/src/lifeform_evolution/"
            "relationship_lab_p4_long_context_source_opportunity_derivation.py",
            "lf",
            "72efc093b815c2ca07872f6cb6a78f53a4d4d5ada5975222b36cf90c640746f8",
            "f0e20481ab180d794d5551b35b3a24c5550476f2",
        ),
        (
            "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/relationship_action_contracts.py",
            "crlf",
            "dc1907cc67d76536b88894f5e06c907ec4651a76acab0fe28531ffe14db2b526",
            None,
        ),
        (
            "packages/lifeform-evolution/src/lifeform_evolution/schemas/relationship_action_choice.schema.json",
            "crlf",
            "764309ff7b1d4aa6e9001a73a8c72407a1fabfd1e9d5c89e7cdf37360054efea",
            None,
        ),
    ),
    ids=("planning-helper-lf", "source-helper-lf", "action-owner-crlf", "action-schema-crlf"),
)
def test_raw_pinned_sources_have_explicit_git_eol_contracts(
    repo_path: str,
    expected_eol: str,
    expected_raw_sha256: str,
    expected_filtered_oid: str | None,
) -> None:
    payload = _REPO_ROOT.joinpath(*pathlib.PurePosixPath(repo_path).parts).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == expected_raw_sha256
    if expected_eol == "lf":
        assert b"\r\n" not in payload
    else:
        assert b"\r\n" in payload
        assert b"\n" not in payload.replace(b"\r\n", b"")

    attributes = subprocess.run(
        [
            "git",
            "-c",
            "safe.directory=D:/volvence",
            "check-attr",
            "text",
            "eol",
            "--",
            repo_path,
        ],
        check=True,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    parsed_attributes = {
        attribute: value
        for line in attributes.stdout.splitlines()
        for _path, attribute, value in (line.split(": ", 2),)
    }
    assert parsed_attributes == {"text": "set", "eol": expected_eol}

    if expected_eol == "lf":
        filtered_oid = subprocess.run(
            [
                "git",
                "-c",
                "safe.directory=D:/volvence",
                "hash-object",
                f"--path={repo_path}",
                repo_path,
            ],
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert filtered_oid == _git_blob_oid_sha1(payload)
        if expected_filtered_oid is not None:
            assert filtered_oid == expected_filtered_oid


def test_importing_owner_does_not_eagerly_load_derivation_helpers() -> None:
    probe = """
import json
import sys
import lifeform_evolution.relationship_lab_p4_long_context_causal_campaign

forbidden = sorted(
    name
    for name in sys.modules
    if name in {
        "lifeform_evolution.relationship_lab_p4_long_context_v4_planning_derivation",
        "lifeform_evolution.relationship_lab_p4_long_context_source_opportunity_derivation",
    }
    or name.startswith("_volvence_verified_p4_v4_")
)
print(json.dumps(forbidden, separators=(",", ":")))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []


def test_verified_helper_loader_compiles_the_verified_buffer_without_rereading_path(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper_path = tmp_path / "changing_helper.py"
    verified_payload = b"SAME_BUFFER_SENTINEL = 'verified'\n"
    changed_payload = b"raise RuntimeError('changed path was reread')\n"
    helper_path.write_bytes(verified_payload)
    expected_raw = hashlib.sha256(verified_payload).hexdigest()
    original_read_bytes = pathlib.Path.read_bytes
    read_count = 0

    def changing_read_bytes(path: pathlib.Path) -> bytes:
        nonlocal read_count
        if path == helper_path:
            read_count += 1
            if read_count == 1:
                helper_path.write_bytes(changed_payload)
                return verified_payload
            return changed_payload
        return original_read_bytes(path)

    module_name = f"_volvence_verified_a0_same_buffer_test_{expected_raw}"
    sys.modules.pop(module_name, None)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(pathlib.Path, "read_bytes", changing_read_bytes)
            module = owner._load_verified_derivation_helper(
                path=helper_path,
                expected_raw_sha256=expected_raw,
                module_label="a0_same_buffer_test",
            )
        assert module.SAME_BUFFER_SENTINEL == "verified"
        assert read_count == 1
        assert helper_path.read_bytes() == changed_payload
    finally:
        sys.modules.pop(module_name, None)


@pytest.mark.skipif(os.name != "nt", reason="Windows namespace and NTFS ADS syntax")
@pytest.mark.parametrize(
    "unsafe_path",
    (
        r"\\server\share\request",
        r"\\?\C:\request",
        r"\\.\PhysicalDrive0",
        r"D:\volvence\request.json:alternate",
    ),
    ids=("unc", "extended-device", "physical-device", "ads"),
)
def test_local_default_stream_guard_rejects_unc_device_and_ads_paths(unsafe_path: str) -> None:
    with pytest.raises(ValueError, match="UNC|device namespace|default data stream"):
        owner._require_local_default_stream_path(unsafe_path, "A0 path test")


def test_local_default_stream_guard_accepts_an_ordinary_local_path(tmp_path: pathlib.Path) -> None:
    ordinary = tmp_path / "ordinary"
    assert owner._require_local_default_stream_path(ordinary, "A0 path test") == pathlib.Path(os.path.abspath(ordinary))
    for unsafe in ("", "embedded\x00nul"):
        with pytest.raises(ValueError, match="non-empty local text path"):
            owner._require_local_default_stream_path(unsafe, "A0 path test")


def test_published_request_replays_as_exact_two_file_all_no_go_artifact() -> None:
    assert {item.name for item in _ANCHOR_REQUEST_ARTIFACT.iterdir()} == {
        _REQUEST_FILE,
        _MANIFEST_FILE,
    }
    request_path = _ANCHOR_REQUEST_ARTIFACT / _REQUEST_FILE
    manifest_path = _ANCHOR_REQUEST_ARTIFACT / _MANIFEST_FILE
    request_bytes = request_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    request = _strict_json(request_path)
    manifest = _strict_json(manifest_path)

    assert len(request_bytes) == _EXPECTED_REQUEST_BYTE_COUNT
    assert len(manifest_bytes) == _EXPECTED_MANIFEST_BYTE_COUNT
    assert request_bytes == _canonical_bytes(request)
    assert manifest_bytes == _canonical_bytes(manifest)
    assert hashlib.sha256(request_bytes).hexdigest() == _EXPECTED_REQUEST_RAW
    assert hashlib.sha256(manifest_bytes).hexdigest() == _EXPECTED_MANIFEST_RAW
    request_core = dict(request)
    assert request_core.pop("request_id") == _EXPECTED_REQUEST_ID
    assert hashlib.sha256(_canonical_bytes(request_core)).hexdigest() == _EXPECTED_REQUEST_ID
    manifest_core = dict(manifest)
    assert manifest_core.pop("artifact_id") == _EXPECTED_ARTIFACT_ID
    assert hashlib.sha256(_canonical_bytes(manifest_core)).hexdigest() == _EXPECTED_ARTIFACT_ID
    assert set(request) == _EXPECTED_REQUEST_KEYS
    assert set(manifest) == _EXPECTED_MANIFEST_KEYS
    assert manifest["files"] == [
        {
            "byte_count": _EXPECTED_REQUEST_BYTE_COUNT,
            "path": _REQUEST_FILE,
            "sha256": _EXPECTED_REQUEST_RAW,
        }
    ]
    assert (
        request["publication_subjects"]
        == _strict_json(_PROTOCOL_PATH)["publication_subject_contract"]["ordered_subjects"]
    )
    _assert_request_is_all_no_go(request, manifest)

    replayed = validate_relationship_p4_long_context_external_anchor_request(
        output_dir=_ANCHOR_REQUEST_ARTIFACT,
        source_preflight_dir=_SOURCE_PREFLIGHT_ARTIFACT,
        v4a_planning_dir=_V4A_PLANNING,
        v3_preparation_dir=_V3_PREPARATION,
        v2_admission_dir=_V2_ADMISSION,
    )
    _assert_result_is_all_no_go(replayed)
    assert replayed.request_id == _EXPECTED_REQUEST_ID
    assert replayed.artifact_id == _EXPECTED_ARTIFACT_ID


def test_prepare_and_validate_reproduce_the_published_bytes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cached_inputs = _external_inputs()
    repository_root = tmp_path / "repository"
    upstream_roots = _mirror_prepare_repository(repository_root)
    output = _canonical_request_output(repository_root)
    with monkeypatch.context() as patch:
        patch.setattr(owner, "_validated_v4_external_anchor_request_inputs", lambda **_kwargs: cached_inputs)
        _patch_prepare_repository(patch, repository_root)
        prepared = prepare_relationship_p4_long_context_external_anchor_request(
            output_dir=output,
            source_preflight_dir=upstream_roots["source_preflight"],
            v4a_planning_dir=upstream_roots["v4a_planning"],
            v3_preparation_dir=upstream_roots["v3_preparation"],
            v2_admission_dir=upstream_roots["v2_admission"],
        )
        replayed = validate_relationship_p4_long_context_external_anchor_request(
            output_dir=output,
            source_preflight_dir=upstream_roots["source_preflight"],
            v4a_planning_dir=upstream_roots["v4a_planning"],
            v3_preparation_dir=upstream_roots["v3_preparation"],
            v2_admission_dir=upstream_roots["v2_admission"],
        )

    assert {item.name for item in output.iterdir()} == {_REQUEST_FILE, _MANIFEST_FILE}
    assert {name: (output / name).read_bytes() for name in (_REQUEST_FILE, _MANIFEST_FILE)} == {
        name: (_ANCHOR_REQUEST_ARTIFACT / name).read_bytes() for name in (_REQUEST_FILE, _MANIFEST_FILE)
    }
    assert prepared == replayed
    _assert_result_is_all_no_go(prepared)
    assert prepared.request_id == _EXPECTED_REQUEST_ID
    assert prepared.artifact_id == _EXPECTED_ARTIFACT_ID


def test_prepare_rejects_noncanonical_output_and_every_noncanonical_upstream_before_validation(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def validation_must_not_run(**_kwargs: object) -> object:
        raise AssertionError("input validation must not run before canonical path checks close")

    canonical_arguments = {
        "output_dir": _ANCHOR_REQUEST_ARTIFACT,
        "source_preflight_dir": _SOURCE_PREFLIGHT_ARTIFACT,
        "v4a_planning_dir": _V4A_PLANNING,
        "v3_preparation_dir": _V3_PREPARATION,
        "v2_admission_dir": _V2_ADMISSION,
    }
    with monkeypatch.context() as patch:
        patch.setattr(owner, "_validated_v4_external_anchor_request_inputs", validation_must_not_run)
        with pytest.raises(ValueError, match="frozen canonical repository path"):
            prepare_relationship_p4_long_context_external_anchor_request(
                **{**canonical_arguments, "output_dir": tmp_path / "noncanonical-output"}
            )

        for field in (
            "source_preflight_dir",
            "v4a_planning_dir",
            "v3_preparation_dir",
            "v2_admission_dir",
        ):
            with pytest.raises(ValueError, match="frozen canonical repository (path|root)"):
                prepare_relationship_p4_long_context_external_anchor_request(
                    **{**canonical_arguments, field: tmp_path / f"relocated-{field}"}
                )


@pytest.mark.parametrize(
    "mutator",
    (
        lambda request: request["authorization_firewall"].__setitem__("external_publication_anchor_present", True),
        lambda request: request["publication_target"].__setitem__("actual_gist_id", "forged-gist"),
        lambda request: request["terminal"].__setitem__("structural_inventory_materialization_authorized", True),
        lambda request: request["zero_output_firewall"].__setitem__("network_request_count", 1),
    ),
    ids=("open-anchor", "prefill-remote-id", "authorize-materialization", "claim-network"),
)
def test_tamper_and_full_local_reseal_cannot_change_frozen_request(
    tmp_path: pathlib.Path,
    mutator: Callable[[dict[str, Any]], object],
) -> None:
    tampered = _copy_request_artifact(tmp_path, "tampered")
    request_id, artifact_id = _resign_request_artifact(tampered, mutator)
    assert request_id != _EXPECTED_REQUEST_ID
    assert artifact_id != _EXPECTED_ARTIFACT_ID
    _assert_internal_hash_chain(tampered, request_id=request_id, artifact_id=artifact_id)
    with pytest.raises((TypeError, ValueError), match="request payload"):
        _validate_request_fast(tampered)


def test_manifest_tamper_and_reseal_cannot_claim_an_external_action(tmp_path: pathlib.Path) -> None:
    tampered = _copy_request_artifact(tmp_path, "manifest-tampered")
    manifest = _strict_json(tampered / _MANIFEST_FILE)
    manifest.pop("artifact_id")
    manifest["Git_push_count"] = 1
    artifact_id = hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
    manifest["artifact_id"] = artifact_id
    (tampered / _MANIFEST_FILE).write_bytes(_canonical_bytes(manifest))
    _assert_internal_hash_chain(tampered, request_id=_EXPECTED_REQUEST_ID, artifact_id=artifact_id)
    with pytest.raises(ValueError, match="request manifest"):
        _validate_request_fast(tampered)


def test_strict_json_and_exact_file_set_reject_bom_duplicate_extra_and_missing(
    tmp_path: pathlib.Path,
) -> None:
    bom = _copy_request_artifact(tmp_path, "bom")
    request_path = bom / _REQUEST_FILE
    request_path.write_bytes(b"\xef\xbb\xbf" + request_path.read_bytes())
    with pytest.raises(ValueError, match="must not carry a UTF-8 BOM"):
        _validate_request_fast(bom)

    duplicate = _copy_request_artifact(tmp_path, "duplicate")
    request_path = duplicate / _REQUEST_FILE
    original = request_path.read_bytes()
    assert original.startswith(b"{")
    request_path.write_bytes(b'{"schema_version":"duplicate",' + original[1:])
    with pytest.raises(ValueError, match="duplicate JSON key: schema_version"):
        _validate_request_fast(duplicate)

    extra_file = _copy_request_artifact(tmp_path, "extra-file")
    (extra_file / "unexpected.json").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="file set drift"):
        _validate_request_fast(extra_file)

    missing_file = _copy_request_artifact(tmp_path, "missing-file")
    (missing_file / _MANIFEST_FILE).unlink()
    with pytest.raises(ValueError, match="file set drift"):
        _validate_request_fast(missing_file)

    extra_key = _copy_request_artifact(tmp_path, "extra-key")
    _resign_request_artifact(extra_key, lambda request: request.__setitem__("unexpected", False))
    with pytest.raises(ValueError, match="keys drift"):
        _validate_request_fast(extra_key)

    missing_key = _copy_request_artifact(tmp_path, "missing-key")
    _resign_request_artifact(missing_key, lambda request: request.pop("claim_boundary"))
    with pytest.raises(ValueError, match="keys drift"):
        _validate_request_fast(missing_key)


def test_request_artifact_rejects_hardlinks_symlinks_and_reparse_roots_when_supported(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hardlinked = _copy_request_artifact(tmp_path, "hardlinked")
    independent = tmp_path / "independent-request.json"
    shutil.copy2(hardlinked / _REQUEST_FILE, independent)
    (hardlinked / _REQUEST_FILE).unlink()
    try:
        os.link(independent, hardlinked / _REQUEST_FILE)
    except OSError as exc:
        pytest.skip(f"hardlink creation is unavailable: {exc}")
    with pytest.raises(ValueError, match="exactly one hard link"):
        _validate_request_fast(hardlinked)

    symlinked = _copy_request_artifact(tmp_path, "symlinked")
    linked_request = symlinked / _REQUEST_FILE
    linked_request.unlink()
    try:
        linked_request.symlink_to(_ANCHOR_REQUEST_ARTIFACT / _REQUEST_FILE)
    except OSError:
        pass
    else:
        with pytest.raises((FileNotFoundError, ValueError), match="regular file|symlink|reparse point"):
            _validate_request_fast(symlinked)

    artifact_target = _copy_request_artifact(tmp_path, "artifact-target")
    artifact_alias = tmp_path / "artifact-alias"
    if _create_directory_alias(artifact_alias, artifact_target):
        try:
            cached_inputs = _external_inputs()
            with monkeypatch.context() as patch:
                patch.setattr(
                    owner,
                    "_validated_v4_external_anchor_request_inputs",
                    lambda **_kwargs: cached_inputs,
                )
                with pytest.raises(ValueError, match="symlink|reparse point"):
                    validate_relationship_p4_long_context_external_anchor_request(
                        output_dir=artifact_alias,
                        source_preflight_dir=_SOURCE_PREFLIGHT_ARTIFACT,
                        v4a_planning_dir=_V4A_PLANNING,
                        v3_preparation_dir=_V3_PREPARATION,
                        v2_admission_dir=_V2_ADMISSION,
                    )
        finally:
            _remove_directory_alias(artifact_alias)


def test_prepare_is_create_only_and_bytes_ignore_path_host_cuda_and_environment(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached_inputs = _external_inputs()
    repository_root_a = tmp_path / "repository-a"
    repository_root_b = tmp_path / "different-parent" / "repository-b"
    upstream_roots_a = _mirror_prepare_repository(repository_root_a)
    upstream_roots_b = _mirror_prepare_repository(repository_root_b)
    output_a = _canonical_request_output(repository_root_a)
    output_b = _canonical_request_output(repository_root_b)
    environment_keys = (
        "COMPUTERNAME",
        "HOSTNAME",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "GITHUB_TOKEN",
        "GH_TOKEN",
    )
    first_environment = tuple(f"alpha-a0-sentinel-{index}" for index in range(len(environment_keys)))
    second_environment = tuple(f"beta-a0-sentinel-{index}" for index in range(len(environment_keys)))
    with monkeypatch.context() as patch:
        patch.setattr(owner, "_validated_v4_external_anchor_request_inputs", lambda **_kwargs: cached_inputs)
        _patch_prepare_repository(patch, repository_root_a)
        for key, value in zip(environment_keys, first_environment, strict=True):
            patch.setenv(key, value)
        result_a = prepare_relationship_p4_long_context_external_anchor_request(
            output_dir=output_a,
            source_preflight_dir=upstream_roots_a["source_preflight"],
            v4a_planning_dir=upstream_roots_a["v4a_planning"],
            v3_preparation_dir=upstream_roots_a["v3_preparation"],
            v2_admission_dir=upstream_roots_a["v2_admission"],
        )
        for key, value in zip(environment_keys, second_environment, strict=True):
            patch.setenv(key, value)
        _patch_prepare_repository(patch, repository_root_b)
        result_b = prepare_relationship_p4_long_context_external_anchor_request(
            output_dir=output_b,
            source_preflight_dir=upstream_roots_b["source_preflight"],
            v4a_planning_dir=upstream_roots_b["v4a_planning"],
            v3_preparation_dir=upstream_roots_b["v3_preparation"],
            v2_admission_dir=upstream_roots_b["v2_admission"],
        )

        filenames = (_REQUEST_FILE, _MANIFEST_FILE)
        bytes_a = {name: (output_a / name).read_bytes() for name in filenames}
        bytes_b = {name: (output_b / name).read_bytes() for name in filenames}
        published_bytes = {name: (_ANCHOR_REQUEST_ARTIFACT / name).read_bytes() for name in filenames}
        assert bytes_a == bytes_b == published_bytes
        assert result_a.request_id == result_b.request_id == _EXPECTED_REQUEST_ID
        assert result_a.artifact_id == result_b.artifact_id == _EXPECTED_ARTIFACT_ID
        combined = b"".join(bytes_a.values())
        for sentinel in (*first_environment, *second_environment):
            assert sentinel.encode("utf-8") not in combined
        assert str(output_a).encode("utf-8") not in combined
        assert str(output_b).encode("utf-8") not in combined

        before = dict(bytes_a)
        _patch_prepare_repository(patch, repository_root_a)
        with pytest.raises(FileExistsError, match="output already exists"):
            prepare_relationship_p4_long_context_external_anchor_request(
                output_dir=output_a,
                source_preflight_dir=upstream_roots_a["source_preflight"],
                v4a_planning_dir=upstream_roots_a["v4a_planning"],
                v3_preparation_dir=upstream_roots_a["v3_preparation"],
                v2_admission_dir=upstream_roots_a["v2_admission"],
            )
        assert {name: (output_a / name).read_bytes() for name in filenames} == before
        assert not tuple(output_a.parent.glob(f".{output_a.name}.tmp-*"))


def test_cli_has_only_local_request_commands_and_keeps_heavy_modules_out_of_import_closure(
    tmp_path: pathlib.Path,
) -> None:
    cli_tree = ast.parse(_CLI_SOURCE.read_text(encoding="utf-8"))
    option_strings = {
        argument.value
        for node in ast.walk(cli_tree)
        if isinstance(node, ast.Call)
        for argument in node.args
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str) and argument.value.startswith("--")
    }
    subcommands = {
        node.args[0].value
        for node in ast.walk(cli_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_parser"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    assert subcommands == {"show-protocol", "prepare-request", "validate-request"}
    assert option_strings == {
        "--output-dir",
        "--source-preflight-dir",
        "--v4a-planning-dir",
        "--v3-preparation-dir",
        "--v2-admission-dir",
    }
    assert option_strings.isdisjoint(
        {
            "--admit",
            "--commit",
            "--cuda",
            "--device",
            "--materialize",
            "--model",
            "--publish",
            "--push",
            "--receipt",
            "--source",
            "--source-dir",
            "--source-path",
            "--token",
        }
    )

    cli_repository_root = tmp_path / "cli-repository"
    cli_upstream_roots = _mirror_prepare_repository(cli_repository_root)
    prepared = _canonical_request_output(cli_repository_root)
    common_arguments = (
        "--output-dir",
        str(prepared),
        "--source-preflight-dir",
        str(cli_upstream_roots["source_preflight"]),
        "--v4a-planning-dir",
        str(cli_upstream_roots["v4a_planning"]),
        "--v3-preparation-dir",
        str(cli_upstream_roots["v3_preparation"]),
        "--v2-admission-dir",
        str(cli_upstream_roots["v2_admission"]),
    )
    help_payload, help_loaded = _run_cli_import_closure(("--help",))
    assert help_payload is None
    assert help_loaded == []
    show_payload, show_loaded = _run_cli_import_closure(("show-protocol",))
    assert show_loaded == []
    assert show_payload is not None
    assert show_payload["protocol_id"] == _EXPECTED_PROTOCOL_ID
    assert show_payload["external_publication_anchor_present"] is False
    assert show_payload["structural_inventory_materialization_authorized"] is False

    prepare_payload, prepare_loaded = _run_cli_import_closure(
        ("prepare-request", *common_arguments),
        repository_root_override=cli_repository_root,
    )
    assert prepare_loaded == []
    assert prepare_payload is not None
    assert prepare_payload["request_id"] == _EXPECTED_REQUEST_ID
    assert prepare_payload["artifact_id"] == _EXPECTED_ARTIFACT_ID
    assert {item.name for item in prepared.iterdir()} == {_REQUEST_FILE, _MANIFEST_FILE}
    validate_payload, validate_loaded = _run_cli_import_closure(
        ("validate-request", *common_arguments),
        repository_root_override=cli_repository_root,
    )
    assert validate_loaded == []
    assert validate_payload == prepare_payload


@lru_cache(maxsize=1)
def _external_inputs() -> tuple[Any, Any, Any, Any]:
    return owner._validated_v4_external_anchor_request_inputs(
        source_preflight_dir=_SOURCE_PREFLIGHT_ARTIFACT,
        v4a_planning_dir=_V4A_PLANNING,
        v3_preparation_dir=_V3_PREPARATION,
        v2_admission_dir=_V2_ADMISSION,
        protocol_path=None,
    )


def _validate_request_fast(path: pathlib.Path) -> object:
    protocol, raw, source_certificate, subjects = _external_inputs()
    return owner._validate_v4_external_anchor_request_root(
        path,
        protocol=protocol,
        raw=raw,
        source_certificate=source_certificate,
        subjects=subjects,
    )


def _strict_json(path: pathlib.Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_bytes().decode("utf-8"), object_pairs_hook=reject_duplicates)
    assert type(value) is dict
    return value


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _git_blob_oid_sha1(payload: bytes) -> str:
    framed = b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
    return hashlib.sha1(framed, usedforsecurity=False).hexdigest()


def _copy_request_artifact(tmp_path: pathlib.Path, name: str) -> pathlib.Path:
    destination = tmp_path / name
    shutil.copytree(_ANCHOR_REQUEST_ARTIFACT, destination)
    return destination


def _canonical_request_output(repository_root: pathlib.Path) -> pathlib.Path:
    binding = _strict_json(_PROTOCOL_PATH)["self_publication_binding"]
    relative = pathlib.PurePosixPath(binding["request_payload_repo_relative_path"])
    return repository_root.joinpath(*relative.parent.parts)


def _mirror_prepare_repository(repository_root: pathlib.Path) -> dict[str, pathlib.Path]:
    raw = _strict_json(_PROTOCOL_PATH)
    binding = raw["self_publication_binding"]
    relative_roots = binding["prepare_upstream_repo_relative_roots"]
    actual_roots = {
        "source_preflight": _SOURCE_PREFLIGHT_ARTIFACT,
        "v4a_planning": _V4A_PLANNING,
        "v3_preparation": _V3_PREPARATION,
        "v2_admission": _V2_ADMISSION,
    }
    mirrored_roots: dict[str, pathlib.Path] = {}
    for name, actual_root in actual_roots.items():
        relative = pathlib.PurePosixPath(relative_roots[name])
        mirrored_root = repository_root.joinpath(*relative.parts)
        shutil.copytree(actual_root, mirrored_root)
        mirrored_roots[name] = mirrored_root

    for subject in raw["publication_subject_contract"]["ordered_subjects"]:
        relative = pathlib.PurePosixPath(subject["repo_relative_posix_path"])
        source = _REPO_ROOT.joinpath(*relative.parts)
        destination = repository_root.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return mirrored_roots


def _patch_prepare_repository(
    patch: pytest.MonkeyPatch,
    repository_root: pathlib.Path,
) -> None:
    patch.setattr(owner, "_REPOSITORY_ROOT", repository_root)
    patch.setattr(
        owner,
        "_V4_SOURCE_OPPORTUNITY_PREFLIGHT_PROTOCOL_PATH_V1",
        repository_root
        / "packages"
        / "lifeform-evolution"
        / "src"
        / "lifeform_evolution"
        / "protocols"
        / "relationship_p4_long_context_v4_source_opportunity_preflight_v1.json",
    )
    patch.setattr(
        owner,
        "_V4_SOURCE_OPPORTUNITY_DERIVATION_HELPER_PATH",
        repository_root
        / "packages"
        / "lifeform-evolution"
        / "src"
        / "lifeform_evolution"
        / "relationship_lab_p4_long_context_source_opportunity_derivation.py",
    )


def _resign_request_artifact(
    root: pathlib.Path,
    mutator: Callable[[dict[str, Any]], object],
) -> tuple[str, str]:
    request_path = root / _REQUEST_FILE
    request = _strict_json(request_path)
    request.pop("request_id")
    mutator(request)
    request_id = hashlib.sha256(_canonical_bytes(request)).hexdigest()
    request["request_id"] = request_id
    request_bytes = _canonical_bytes(request)
    request_path.write_bytes(request_bytes)

    manifest_path = root / _MANIFEST_FILE
    manifest = _strict_json(manifest_path)
    manifest.pop("artifact_id")
    manifest["request_id"] = request_id
    manifest["files"] = [
        {
            "path": _REQUEST_FILE,
            "byte_count": len(request_bytes),
            "sha256": hashlib.sha256(request_bytes).hexdigest(),
        }
    ]
    artifact_id = hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
    manifest["artifact_id"] = artifact_id
    manifest_path.write_bytes(_canonical_bytes(manifest))
    return request_id, artifact_id


def _assert_internal_hash_chain(
    root: pathlib.Path,
    *,
    request_id: str,
    artifact_id: str,
) -> None:
    request = _strict_json(root / _REQUEST_FILE)
    request_core = dict(request)
    assert request_core.pop("request_id") == request_id
    assert hashlib.sha256(_canonical_bytes(request_core)).hexdigest() == request_id
    request_bytes = (root / _REQUEST_FILE).read_bytes()
    manifest = _strict_json(root / _MANIFEST_FILE)
    manifest_core = dict(manifest)
    assert manifest_core.pop("artifact_id") == artifact_id
    assert hashlib.sha256(_canonical_bytes(manifest_core)).hexdigest() == artifact_id
    assert manifest["request_id"] == request_id
    assert manifest["files"] == [
        {
            "path": _REQUEST_FILE,
            "byte_count": len(request_bytes),
            "sha256": hashlib.sha256(request_bytes).hexdigest(),
        }
    ]


def _assert_request_is_all_no_go(request: dict[str, Any], manifest: dict[str, Any]) -> None:
    authorization = request["authorization_firewall"]
    assert authorization["publication_request_contract_frozen"] is True
    assert all(value is False for key, value in authorization.items() if key != "publication_request_contract_frozen")
    zero = request["zero_output_firewall"]
    assert all(value == 0 for key, value in zero.items() if key.endswith("_count"))
    assert all(value is False for key, value in zero.items() if key.endswith("_claimed"))
    assert all(value is False for key, value in zero.items() if key.endswith("_supported"))
    terminal = request["terminal"]
    assert terminal["publication_request_contract_frozen"] is True
    assert terminal["external_publication_anchor_present"] is False
    assert terminal["structural_inventory_materialization_authorized"] is False
    for key in (
        "publication_request_artifact_id",
        "GitHub_gist_id",
        "GitHub_revision_oid",
        "external_receipt_artifact_id",
        "A0_admission_artifact_id",
    ):
        assert terminal[key] is None
    assert request["future_receipt_requirements"]["receipt_alone_may_authorize_source_materialization"] is False
    assert manifest["publication_request_contract_frozen"] is True
    for key, value in manifest.items():
        if key.endswith("_authorized") or key in {
            "external_request_dispatched",
            "publication_performed",
            "external_publication_anchor_present",
            "external_anchor_admitted",
        }:
            assert value is False
        if key.endswith("_count"):
            assert value == 0


def _assert_result_is_all_no_go(result: object) -> None:
    assert result.publication_request_contract_frozen is True
    assert result.external_request_dispatched is False
    assert result.publication_performed is False
    assert result.external_publication_anchor_present is False
    assert result.external_anchor_admitted is False
    assert result.structural_inventory_materialization_authorized is False
    assert result.source_execution_authorized is False
    assert result.tuple_feasibility_authorized is False
    assert result.model_output_authorized is False
    assert result.cuda_planner_authorized is False


def _run_cli_import_closure(
    arguments: tuple[str, ...],
    *,
    repository_root_override: pathlib.Path | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    probe = """
import importlib.util
import json
import pathlib
import sys

cli_path = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("_external_anchor_cli_probe", cli_path)
if spec is None or spec.loader is None:
    raise RuntimeError("CLI probe could not create an import spec")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if sys.argv[2] != "-":
    module._install_exact_source_roots()
    import lifeform_evolution.relationship_lab_p4_long_context_causal_campaign as owner
    repository_root = pathlib.Path(sys.argv[2])
    owner._REPOSITORY_ROOT = repository_root
    owner._V4_SOURCE_OPPORTUNITY_PREFLIGHT_PROTOCOL_PATH_V1 = (
        repository_root
        / "packages"
        / "lifeform-evolution"
        / "src"
        / "lifeform_evolution"
        / "protocols"
        / "relationship_p4_long_context_v4_source_opportunity_preflight_v1.json"
    )
    owner._V4_SOURCE_OPPORTUNITY_DERIVATION_HELPER_PATH = (
        repository_root
        / "packages"
        / "lifeform-evolution"
        / "src"
        / "lifeform_evolution"
        / "relationship_lab_p4_long_context_source_opportunity_derivation.py"
    )
try:
    return_code = module.main(sys.argv[3:])
except SystemExit as exc:
    return_code = exc.code
if return_code != 0:
    raise RuntimeError(f"CLI returned {return_code!r}")
forbidden_roots = (
    "lifeform_core",
    "lifeform_domain_emogpt",
    "volvence_zero.substrate",
    "torch",
    "transformers",
    "vllm",
)
loaded = sorted(
    name
    for name in sys.modules
    if any(name == root or name.startswith(root + ".") for root in forbidden_roots)
)
print("IMPORT_CLOSURE=" + json.dumps(loaded, separators=(",", ":")))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            probe,
            str(_CLI_SOURCE),
            "-" if repository_root_override is None else str(repository_root_override),
            *arguments,
        ],
        check=True,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    marker = next(
        (line for line in reversed(completed.stdout.splitlines()) if line.startswith("IMPORT_CLOSURE=")),
        None,
    )
    assert marker is not None, completed.stdout
    loaded = json.loads(marker.removeprefix("IMPORT_CLOSURE="))
    assert isinstance(loaded, list)
    assert all(isinstance(item, str) for item in loaded)
    payload_line = next(
        (line for line in reversed(completed.stdout.splitlines()) if line.startswith("{") and line.endswith("}")),
        None,
    )
    payload = None if payload_line is None else json.loads(payload_line)
    if payload is not None:
        assert isinstance(payload, dict)
    return payload, loaded


def _create_directory_alias(alias: pathlib.Path, target: pathlib.Path) -> bool:
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(alias), str(target)],
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.returncode == 0
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError:
        return False
    return True


def _remove_directory_alias(alias: pathlib.Path) -> None:
    if not os.path.lexists(alias):
        return
    if os.name == "nt":
        alias.rmdir()
    else:
        alias.unlink()
