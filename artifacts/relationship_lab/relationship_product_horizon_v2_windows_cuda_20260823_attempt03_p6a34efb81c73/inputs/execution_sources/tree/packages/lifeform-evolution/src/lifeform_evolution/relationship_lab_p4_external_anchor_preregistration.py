"""Create-only A0 v2 preregistration for an independently observed public Gist anchor.

The v1 request remains immutable historical evidence of an under-specified receipt
contract.  This module is the sole active writer for the v2 successor.  It performs
only local, content-addressed preparation and replay; publication and observation
live in separately pinned adapters and cannot be triggered from here.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import pathlib
import shutil
import stat
import uuid
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping


P4_EXTERNAL_ANCHOR_PROTOCOL_SCHEMA_VERSION_V2 = (
    "relationship-p4-independent-long-context-v4-external-publication-anchor-request-protocol.v2"
)
P4_EXTERNAL_ANCHOR_REQUEST_SCHEMA_VERSION_V2 = (
    "relationship-p4-long-context-external-publication-anchor-request.v2"
)
P4_EXTERNAL_ANCHOR_REQUEST_MANIFEST_SCHEMA_VERSION_V2 = (
    "relationship-p4-long-context-external-publication-anchor-request-manifest.v2"
)
P4_EXTERNAL_ANCHOR_REQUEST_STATUS_V2 = (
    "external_publication_anchor_v2_request_frozen_publication_not_observed_no_authority"
)

# Filled only after the corresponding immutable file/artifact exists.  The build
# graph is protocol -> request -> manifest, so no generated identity is fed back
# into an upstream object.
P4_EXTERNAL_ANCHOR_PROTOCOL_ID_V2 = "__P4_EXTERNAL_ANCHOR_PROTOCOL_ID_V2__"
P4_EXTERNAL_ANCHOR_PROTOCOL_RAW_SHA256_V2 = "__P4_EXTERNAL_ANCHOR_PROTOCOL_RAW_SHA256_V2__"
P4_EXTERNAL_ANCHOR_REQUEST_ID_V2 = "__P4_EXTERNAL_ANCHOR_REQUEST_ID_V2__"
P4_EXTERNAL_ANCHOR_REQUEST_ARTIFACT_ID_V2 = "__P4_EXTERNAL_ANCHOR_REQUEST_ARTIFACT_ID_V2__"

_MODULE_PATH = pathlib.Path(os.path.abspath(__file__))
_MODULE_DIR = _MODULE_PATH.parent
_REPOSITORY_ROOT = _MODULE_PATH.parents[4]
_PROTOCOL_PATH = (
    _MODULE_DIR / "protocols" / "relationship_p4_long_context_v4_external_publication_anchor_v2.json"
)
_REQUEST_FILE = "external_publication_anchor_request.json"
_MANIFEST_FILE = "manifest.json"
_HEX = frozenset("0123456789abcdef")
_MAX_LOCAL_INPUT_BYTES = 8 * 1024 * 1024

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id_contract",
        "frozen_at_utc",
        "owner",
        "supersession",
        "question",
        "input_lineage",
        "anchor_stage",
        "publication_subject_contract",
        "publication_target_contract",
        "verification_preregistration",
        "observer_acquisition_contract",
        "git_object_closure_contract",
        "receipt_contract",
        "admission_contract",
        "self_publication_binding",
        "authorization_firewall",
        "zero_output_firewall",
        "terminal",
        "claim_boundary",
    }
)

# Patched from the final v2 protocol.  Per-section hashes make accidental edits
# localizable; the raw-file hash remains the primary byte identity.
_PROTOCOL_SECTION_SHA256_V2: Mapping[str, str] = MappingProxyType({})


@dataclass(frozen=True)
class RelationshipP4ExternalAnchorProtocolV2:
    """Validated view of the v2 A0 request protocol."""

    protocol_id: str
    schema_version: str
    frozen_at_utc: str
    required_filename: str
    expected_owner_login: str
    expected_owner_id: int
    expected_owner_node_id: str
    publication_subject_count: int
    normative_file_count: int
    status: str
    claim_boundary: str

    def __post_init__(self) -> None:
        _require_sha256(self.protocol_id, "A0 v2 protocol id")
        if self.protocol_id != P4_EXTERNAL_ANCHOR_PROTOCOL_ID_V2:
            raise ValueError("A0 v2 protocol id drift")
        if self.schema_version != P4_EXTERNAL_ANCHOR_PROTOCOL_SCHEMA_VERSION_V2:
            raise ValueError("A0 v2 protocol schema drift")
        if not self.frozen_at_utc.endswith("Z"):
            raise ValueError("A0 v2 frozen_at_utc must be UTC")
        if self.required_filename != "volvence_p4_7_source_opportunity_a0_anchor_request.json":
            raise ValueError("A0 v2 publication filename drift")
        if (
            self.expected_owner_login != "ronaldzgithub"
            or self.expected_owner_id != 36839548
            or self.expected_owner_node_id != "MDQ6VXNlcjM2ODM5NTQ4"
        ):
            raise ValueError("A0 v2 frozen owner identity drift")
        if self.publication_subject_count != 5:
            raise ValueError("A0 v2 publication subject count drift")
        if self.normative_file_count != 6:
            raise ValueError("A0 v2 normative verifier closure is incomplete")
        if self.status != P4_EXTERNAL_ANCHOR_REQUEST_STATUS_V2:
            raise ValueError("A0 v2 terminal status drift")
        if not self.claim_boundary.strip():
            raise ValueError("A0 v2 claim boundary is empty")


@dataclass(frozen=True)
class RelationshipP4ExternalAnchorRequestV2:
    """Validated local v2 request artifact; every external authority is false."""

    artifact_id: str
    request_id: str
    protocol_id: str
    status: str
    publication_request_contract_frozen: bool
    external_request_dispatched: bool
    publication_object_exists_observed: bool
    publisher_action_or_identity_proven: bool
    external_publication_observed: bool
    external_publication_anchor_present: bool
    external_anchor_admitted: bool
    a1_contract_and_materializer_implementation_authorized: bool
    structural_inventory_materialization_authorized: bool
    source_execution_authorized: bool
    model_output_authorized: bool
    cuda_planner_authorized: bool
    output_dir: pathlib.Path

    def __post_init__(self) -> None:
        _require_sha256(self.artifact_id, "A0 v2 request artifact id")
        _require_sha256(self.request_id, "A0 v2 request id")
        if self.artifact_id != P4_EXTERNAL_ANCHOR_REQUEST_ARTIFACT_ID_V2:
            raise ValueError("A0 v2 request artifact id drift")
        if self.request_id != P4_EXTERNAL_ANCHOR_REQUEST_ID_V2:
            raise ValueError("A0 v2 request id drift")
        if self.protocol_id != P4_EXTERNAL_ANCHOR_PROTOCOL_ID_V2:
            raise ValueError("A0 v2 request protocol lineage drift")
        if self.status != P4_EXTERNAL_ANCHOR_REQUEST_STATUS_V2:
            raise ValueError("A0 v2 request status drift")
        if not self.publication_request_contract_frozen:
            raise ValueError("A0 v2 request contract is not frozen")
        if any(
            (
                self.external_request_dispatched,
                self.publication_object_exists_observed,
                self.publisher_action_or_identity_proven,
                self.external_publication_observed,
                self.external_publication_anchor_present,
                self.external_anchor_admitted,
                self.a1_contract_and_materializer_implementation_authorized,
                self.structural_inventory_materialization_authorized,
                self.source_execution_authorized,
                self.model_output_authorized,
                self.cuda_planner_authorized,
            )
        ):
            raise ValueError("A0 v2 request opened downstream authority")


def protocol_path() -> pathlib.Path:
    """Return the canonical v2 protocol path without resolving filesystem links."""

    return _PROTOCOL_PATH


def load_protocol(path: pathlib.Path | None = None) -> RelationshipP4ExternalAnchorProtocolV2:
    """Load and validate the complete local protocol and normative closure."""

    protocol, _raw = _load_protocol_bundle(_PROTOCOL_PATH if path is None else path)
    return protocol


def _load_protocol_bundle(
    path: pathlib.Path,
) -> tuple[RelationshipP4ExternalAnchorProtocolV2, Mapping[str, Any]]:
    """Read every protocol-controlled local input once for one validation operation."""

    protocol_source = _plain_file_buffer(
        path,
        "A0 v2 protocol",
    )
    protocol_bytes = protocol_source[1]
    if _sha256(protocol_bytes) != P4_EXTERNAL_ANCHOR_PROTOCOL_RAW_SHA256_V2:
        raise ValueError("A0 v2 protocol raw bytes drift")
    raw = _strict_json_object(protocol_bytes, "A0 v2 protocol")
    _require_exact_keys(raw, _TOP_LEVEL_KEYS, "A0 v2 protocol")
    if raw["schema_version"] != P4_EXTERNAL_ANCHOR_PROTOCOL_SCHEMA_VERSION_V2:
        raise ValueError("A0 v2 protocol schema drift")
    if raw["protocol_id_contract"] != "sha256_canonical_json_utf8_newline_v1":
        raise ValueError("A0 v2 protocol id contract drift")
    protocol_id = _sha256(_canonical_bytes(raw))
    if protocol_id != P4_EXTERNAL_ANCHOR_PROTOCOL_ID_V2:
        raise ValueError("A0 v2 protocol semantic id drift")
    if set(_PROTOCOL_SECTION_SHA256_V2) != set(_TOP_LEVEL_KEYS):
        raise ValueError("A0 v2 protocol section registry is incomplete")
    for section, expected in _PROTOCOL_SECTION_SHA256_V2.items():
        if _sha256(_canonical_bytes(raw[section])) != expected:
            raise ValueError(f"A0 v2 protocol section drift: {section}")
    _validate_protocol_semantics(raw)
    _validate_normative_closure(raw)
    return _protocol_view(raw, protocol_id), MappingProxyType(raw)


def prepare_request(
    *,
    output_dir: pathlib.Path,
    protocol_path_override: pathlib.Path | None = None,
) -> RelationshipP4ExternalAnchorRequestV2:
    """Create the sole canonical v2 request without network, Git, source, model, or CUDA."""

    protocol_source = _PROTOCOL_PATH if protocol_path_override is None else protocol_path_override
    protocol, raw = _load_protocol_bundle(protocol_source)
    output = _require_local_default_stream_path(output_dir, "A0 v2 request output")
    expected_output = _canonical_output(raw)
    if output != expected_output:
        raise ValueError(f"A0 v2 request must target its frozen canonical path: {expected_output}")
    _validate_frozen_inputs(raw)
    _reject_reparse_components(output, "A0 v2 request output")
    if os.path.lexists(output):
        raise FileExistsError(f"A0 v2 request output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    published = False
    try:
        request_core = _request_core(protocol, raw)
        request_id = _sha256(_canonical_bytes(request_core))
        request_bytes = _canonical_bytes({**request_core, "request_id": request_id})
        _write_create(temporary / _REQUEST_FILE, request_bytes)
        manifest_core = _manifest_core(protocol, request_id, request_bytes)
        artifact_id = _sha256(_canonical_bytes(manifest_core))
        _write_create(
            temporary / _MANIFEST_FILE,
            _canonical_bytes({**manifest_core, "artifact_id": artifact_id}),
        )
        result = _validate_request_root(temporary, protocol=protocol, raw=raw)
        if os.path.lexists(output):
            raise FileExistsError("A0 v2 canonical output appeared during create-only preparation")
        temporary.rename(output)
        published = True
        return replace(result, output_dir=output)
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)


def validate_request(
    *,
    output_dir: pathlib.Path,
    protocol_path_override: pathlib.Path | None = None,
) -> RelationshipP4ExternalAnchorRequestV2:
    """Replay an existing v2 request from immutable local buffers only."""

    protocol_source = _PROTOCOL_PATH if protocol_path_override is None else protocol_path_override
    protocol, raw = _load_protocol_bundle(protocol_source)
    _validate_frozen_inputs(raw)
    output = _require_local_default_stream_path(output_dir, "A0 v2 request root")
    _reject_reparse_components(output, "A0 v2 request root")
    return _validate_request_root(output, protocol=protocol, raw=raw)


def _validate_protocol_semantics(raw: Mapping[str, Any]) -> None:
    owner = _mapping(raw["owner"], "A0 v2 owner")
    _require_literal(
        owner,
        {
            "wheel": "lifeform-evolution",
            "module": "lifeform_evolution.relationship_lab_p4_external_anchor_preregistration",
            "data_owner": "relationship_p4_long_context_external_publication_anchor",
            "wiring_level": "OFFLINE_READOUT_ONLY",
            "runtime_slot_registered": False,
            "v1_writer_status": "historical_immutable_not_active",
            "v2_writer_status": "sole_active_create_only_request_writer",
        },
        "A0 v2 owner",
    )
    supersession = _mapping(raw["supersession"], "A0 v2 supersession")
    if (
        supersession["v1_protocol_id"]
        != "dedfc7ff42f1be0030cdfbe64fd6b1d6dc868adf9db6a9f1150883a9a96a4bee"
        or supersession["v1_protocol_raw_sha256"]
        != "38ce85d479c4359c252de8e5293ca1c15d886c5e1757610435aad136feeca8c6"
        or supersession["v1_request_id"]
        != "7897e3285299eac33385f69fb560a7d68e9f3316fdaf200f27cfa9bbfda489d1"
        or supersession["v1_request_artifact_id"]
        != "5496fa80bba07c6b2234e0e2ca9293111d7ed6edf0a676ee4f561a7893c22900"
        or supersession["v1_publication_performed"] is not False
        or supersession["v1_must_remain_byte_identical"] is not True
        or supersession["reason"]
        != "v1_did_not_mechanically_freeze_receipt_admission_and_observer_acceptance_rules"
    ):
        raise ValueError("A0 v2 supersession boundary drift")
    target = _mapping(raw["publication_target_contract"], "A0 v2 target")
    expected_owner = _mapping(target["expected_owner"], "A0 v2 expected owner")
    if expected_owner != {
        "login": "ronaldzgithub",
        "id": 36839548,
        "node_id": "MDQ6VXNlcjM2ODM5NTQ4",
        "type": "User",
        "site_admin": False,
    }:
        raise ValueError("A0 v2 expected owner drift")
    role_hosts = _mapping(target["role_hosts"], "A0 v2 role hosts")
    if role_hosts != {
        "api": "api.github.com",
        "raw": "gist.githubusercontent.com",
        "html_and_git": "gist.github.com",
        "owner_profile_field_only_not_fetched_by_observer": "github.com",
    }:
        raise ValueError("A0 v2 role host contract drift")
    if (
        target["provider"] != "github_public_gist_sole_first_revision_v2"
        or target["required_filename"] != "volvence_p4_7_source_opportunity_a0_anchor_request.json"
        or target["required_description"] != ""
        or target["required_visibility"] != "public"
        or target["required_exact_file_count"] != 1
        or target["required_exact_history_count"] != 1
        or target["required_exact_advertised_head_count"] != 1
        or target["required_parent_count"] != 0
        or target["actual_gist_id"] is not None
        or target["actual_gist_node_id"] is not None
        or target["actual_revision_oid"] is not None
        or target["actual_raw_URL"] is not None
        or target["actual_HTML_URL"] is not None
        or target["actual_git_pull_URL"] is not None
    ):
        raise ValueError("A0 v2 target identity or unknown locator drift")
    acquisition = _mapping(raw["observer_acquisition_contract"], "A0 v2 acquisition")
    if (
        acquisition["observation_pass_ids"] != ["R0", "R1"]
        or acquisition["observation_pass_count"] != 2
        or acquisition["per_request_retry_count"] != 0
        or acquisition["api_redirect_max_hops"] != 0
        or acquisition["raw_redirect_max_hops"] != 3
        or acquisition["html_redirect_max_hops"] != 3
        or acquisition["git_follow_redirects"] is not False
        or acquisition["success_HTTP_statuses"] != [200]
    ):
        raise ValueError("A0 v2 acquisition cardinality or redirect contract drift")
    _require_literal(
        acquisition["Git_process_resource_contract"],
        {
            "production_platform": "Windows",
            "spawn_sequence": (
                "CREATE_SUSPENDED_then_assign_to_fresh_Job_Object_then_resume_only_initial_thread"
            ),
            "Job_Object_kill_on_close": True,
            "timeout_failure_or_output_cap_terminates_and_reaps_the_whole_child_process_tree": True,
            "successful_command_requires_zero_active_Job_processes_before_return": True,
            "process_tree_reap_deadline_seconds": 10,
            "stdout_max_bytes_per_command": 4194304,
            "stderr_max_bytes_per_command": 4194304,
            "stdin_max_bytes_per_command": 4096,
            "stdin_write_must_remain_inside_the_total_command_deadline": True,
            "Git_version_preflight_uses_the_same_bounded_process_tree_runner": True,
            "Job_PID_and_process_ledgers_are_cooperative_local_metadata_not_remote_attestation": True,
        },
        "A0 v2 Git process resource contract",
    )
    _require_literal(
        acquisition["production_Windows_Git_executable"],
        {
            "absolute_path": "C:\\Program Files\\Git\\mingw64\\libexec\\git-core\\git.exe",
            "byte_count": 3872760,
            "raw_sha256": "bf50371f964f7be61a76ebad7dce8b6197afd47b5b588fc62780823aa097ff63",
            "version_stdout": "git version 2.44.0.windows.1",
            "transport_helpers": [
                {
                    "absolute_path": (
                        "C:\\Program Files\\Git\\mingw64\\libexec\\git-core\\git-remote-http.exe"
                    ),
                    "byte_count": 2375176,
                    "raw_sha256": (
                        "e6bc0697e9405ffc7e94abaecd33e924d8b8b1634b57738629515b237c568c60"
                    ),
                },
                {
                    "absolute_path": (
                        "C:\\Program Files\\Git\\mingw64\\libexec\\git-core\\git-remote-https.exe"
                    ),
                    "byte_count": 2375176,
                    "raw_sha256": (
                        "e6bc0697e9405ffc7e94abaecd33e924d8b8b1634b57738629515b237c568c60"
                    ),
                },
            ],
            "must_be_revalidated_from_one_regular_default_stream_buffer_before_use": True,
            "version_is_environment_metadata_but_executable_hash_is_required_for_production_backend": True,
            "transitive_OS_DLL_and_TLS_library_closure_is_fully_pinned": False,
            "cooperative_local_Git_toolchain_trust_remains": True,
        },
        "A0 v2 production Git executable",
    )
    admission = _mapping(raw["admission_contract"], "A0 v2 admission")
    receipt = _mapping(raw["receipt_contract"], "A0 v2 receipt")
    if (
        receipt["schema_version"]
        != "relationship-p4-long-context-github-public-gist-anchor-receipt.v2"
        or admission["schema_version"]
        != "relationship-p4-long-context-github-public-gist-anchor-admission.v2"
        or admission["truth_expression"]
        != "L_and_H_and_T_and_J_and_G_and_B_and_E_for_R0_and_R1_and_D_and_S"
        or admission["A1_required_before_materialization"] is not True
        or admission["receipt_alone_may_admit_anchor"] is not False
    ):
        raise ValueError("A0 v2 admission truth table drift")
    firewall = _mapping(raw["authorization_firewall"], "A0 v2 firewall")
    if firewall["publication_request_contract_frozen"] is not True:
        raise ValueError("A0 v2 request contract is not frozen")
    for field, value in firewall.items():
        if field == "publication_request_contract_frozen":
            continue
        if value is not False:
            raise ValueError(f"A0 v2 authority opened: {field}")
    zero = _mapping(raw["zero_output_firewall"], "A0 v2 zero-output firewall")
    for field, value in zero.items():
        if field.endswith("_count") and value != 0:
            raise ValueError(f"A0 v2 zero-output count is nonzero: {field}")
        if (field.endswith("_claimed") or field.endswith("_supported")) and value is not False:
            raise ValueError(f"A0 v2 claim firewall opened: {field}")
    terminal = _mapping(raw["terminal"], "A0 v2 terminal")
    if (
        terminal["status"] != P4_EXTERNAL_ANCHOR_REQUEST_STATUS_V2
        or terminal["external_publication_anchor_present"] is not False
        or terminal["A1_required_before_materialization"] is not True
        or terminal["structural_inventory_materialization_authorized"] is not False
    ):
        raise ValueError("A0 v2 terminal boundary drift")


def _protocol_view(raw: Mapping[str, Any], protocol_id: str) -> RelationshipP4ExternalAnchorProtocolV2:
    subjects = _mapping(raw["publication_subject_contract"], "A0 v2 subjects")
    verification = _mapping(raw["verification_preregistration"], "A0 v2 verification")
    target = _mapping(raw["publication_target_contract"], "A0 v2 target")
    owner = _mapping(target["expected_owner"], "A0 v2 expected owner")
    terminal = _mapping(raw["terminal"], "A0 v2 terminal")
    return RelationshipP4ExternalAnchorProtocolV2(
        protocol_id=protocol_id,
        schema_version=_text(raw["schema_version"], "A0 v2 schema"),
        frozen_at_utc=_text(raw["frozen_at_utc"], "A0 v2 freeze time"),
        required_filename=_text(target["required_filename"], "A0 v2 filename"),
        expected_owner_login=_text(owner["login"], "A0 v2 owner login"),
        expected_owner_id=_integer(owner["id"], "A0 v2 owner id"),
        expected_owner_node_id=_text(owner["node_id"], "A0 v2 owner node id"),
        publication_subject_count=_integer(subjects["subject_count"], "A0 v2 subject count"),
        normative_file_count=_integer(verification["normative_file_count"], "A0 v2 normative file count"),
        status=_text(terminal["status"], "A0 v2 status"),
        claim_boundary=_text(raw["claim_boundary"], "A0 v2 claim boundary"),
    )


def _validate_normative_closure(raw: Mapping[str, Any]) -> None:
    verification = _mapping(raw["verification_preregistration"], "A0 v2 verification")
    files = verification["normative_files"]
    if type(files) is not list or len(files) != verification["normative_file_count"]:
        raise ValueError("A0 v2 normative file count drift")
    roles: set[str] = set()
    paths: set[str] = set()
    for index, value in enumerate(files):
        descriptor = _mapping(value, f"A0 v2 normative file {index}")
        _require_exact_keys(
            descriptor,
            frozenset(
                {
                    "role",
                    "repo_relative_posix_path",
                    "media_type",
                    "byte_count",
                    "raw_sha256",
                    "git_blob_oid_sha1",
                    "expected_eol",
                    "same_buffer_operation",
                }
            ),
            f"A0 v2 normative file {index}",
        )
        role = _text(descriptor["role"], f"A0 v2 normative role {index}")
        relative = _safe_repo_path(descriptor["repo_relative_posix_path"], f"A0 v2 normative path {index}")
        if role in roles or relative in paths:
            raise ValueError("A0 v2 normative closure contains duplicate role or path")
        roles.add(role)
        paths.add(relative)
        path = _REPOSITORY_ROOT.joinpath(*pathlib.PurePosixPath(relative).parts)
        _, payload = _plain_file_buffer(path, f"A0 v2 normative file {role}")
        _validate_descriptor_bytes(descriptor, payload, f"A0 v2 normative file {role}")
        media_type = descriptor["media_type"]
        if media_type == "application/json":
            _strict_json_value(payload, f"A0 v2 normative JSON {role}")
        elif media_type == "text/x-python":
            source = _strict_lf_text(payload, f"A0 v2 normative Python {role}")
            compile(source, str(path), "exec", dont_inherit=True)
            ast.parse(source, filename=str(path))
        else:
            raise ValueError(f"A0 v2 unsupported normative media type: {media_type}")
    required_roles = {
        "receipt_schema",
        "admission_schema",
        "pure_verifier_and_admission_judge",
        "fixed_production_observer",
        "observer_CLI",
        "golden_vectors",
    }
    if roles != required_roles:
        raise ValueError("A0 v2 normative closure omits a required role")
    if verification["same_buffer_hash_parse_compile_required"] is not True:
        raise ValueError("A0 v2 same-buffer verifier requirement is disabled")
    if verification["ordinary_import_may_substitute_for_pinned_same_buffer_load"] is not False:
        raise ValueError("A0 v2 allows an unpinned verifier import")
    if (
        verification[
            "CLI_requires_expected_protocol_raw_SHA256_as_an_external_invocation_pin_to_avoid_self_bootstrap"
        ]
        is not True
        or verification[
            "the_OS_Python_launch_of_the_outer_CLI_bootstrap_is_cooperative_local_trust_not_self_proven_by_the_CLI"
        ]
        is not True
        or verification["pinned_CLI_allows_exactly_one_stage_per_process_and_then_exits"] is not True
        or verification["R0_stage_and_R1_admission_stage_may_not_share_one_CLI_invocation"] is not True
    ):
        raise ValueError("A0 v2 CLI trust or process boundary drift")


def _validate_frozen_inputs(raw: Mapping[str, Any]) -> None:
    subjects = _mapping(raw["publication_subject_contract"], "A0 v2 subjects")
    values = subjects["ordered_subjects"]
    if type(values) is not list or len(values) != subjects["subject_count"]:
        raise ValueError("A0 v2 subject inventory drift")
    seen: set[str] = set()
    for index, value in enumerate(values):
        descriptor = _mapping(value, f"A0 v2 subject {index}")
        relative = _safe_repo_path(descriptor["repo_relative_posix_path"], f"A0 v2 subject path {index}")
        if relative.casefold() in seen:
            raise ValueError("A0 v2 subject paths casefold-collide")
        seen.add(relative.casefold())
        path = _REPOSITORY_ROOT.joinpath(*pathlib.PurePosixPath(relative).parts)
        _, payload = _plain_file_buffer(path, f"A0 v2 subject {index}")
        if payload.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError("A0 v2 publication subject is a Git LFS pointer")
        _validate_descriptor_bytes(descriptor, payload, f"A0 v2 subject {index}")
    binding = _mapping(raw["self_publication_binding"], "A0 v2 binding")
    v1_root = _REPOSITORY_ROOT.joinpath(
        *pathlib.PurePosixPath(
            _safe_repo_path(binding["v1_request_artifact_repo_relative_root"], "A0 v2 v1 artifact root")
        ).parts
    )
    _plain_directory(v1_root, (_REQUEST_FILE, _MANIFEST_FILE), "A0 v2 frozen v1 request artifact")
    lineage = _mapping(raw["input_lineage"], "A0 v2 lineage")
    v1_request = _plain_file_buffer(v1_root / _REQUEST_FILE, "A0 v2 v1 request")[1]
    v1_manifest = _plain_file_buffer(v1_root / _MANIFEST_FILE, "A0 v2 v1 manifest")[1]
    if (
        len(v1_request) != lineage["v1_request_byte_count"]
        or _sha256(v1_request) != lineage["v1_request_raw_sha256"]
        or len(v1_manifest) != lineage["v1_manifest_byte_count"]
        or _sha256(v1_manifest) != lineage["v1_manifest_raw_sha256"]
    ):
        raise ValueError("A0 v2 v1 request lineage bytes drift")


def _request_core(
    protocol: RelationshipP4ExternalAnchorProtocolV2,
    raw: Mapping[str, Any],
) -> dict[str, object]:
    return {
        "schema_version": P4_EXTERNAL_ANCHOR_REQUEST_SCHEMA_VERSION_V2,
        "request_id_contract": "sha256_canonical_json_utf8_newline_without_request_id_v1",
        "frozen_at_utc": protocol.frozen_at_utc,
        "identity": {
            "anchor_request_protocol_id": protocol.protocol_id,
            "anchor_request_protocol_raw_sha256": P4_EXTERNAL_ANCHOR_PROTOCOL_RAW_SHA256_V2,
            "superseded_v1_protocol_id": raw["supersession"]["v1_protocol_id"],
            "superseded_v1_request_id": raw["supersession"]["v1_request_id"],
            "source_preflight_protocol_id": raw["input_lineage"]["source_preflight_protocol_id"],
            "source_preflight_artifact_id": raw["input_lineage"]["source_preflight_artifact_id"],
        },
        "supersession": raw["supersession"],
        "anchor_stage": raw["anchor_stage"],
        "publication_subjects": raw["publication_subject_contract"]["ordered_subjects"],
        "publication_target": raw["publication_target_contract"],
        "verification_preregistration": raw["verification_preregistration"],
        "observer_acquisition_contract": raw["observer_acquisition_contract"],
        "git_object_closure_contract": raw["git_object_closure_contract"],
        "receipt_contract": raw["receipt_contract"],
        "admission_contract": raw["admission_contract"],
        "self_publication_binding": raw["self_publication_binding"],
        "authorization_firewall": raw["authorization_firewall"],
        "zero_output_firewall": raw["zero_output_firewall"],
        "terminal": raw["terminal"],
        "claim_boundary": protocol.claim_boundary,
    }


def _manifest_core(
    protocol: RelationshipP4ExternalAnchorProtocolV2,
    request_id: str,
    request_bytes: bytes,
) -> dict[str, object]:
    return {
        "schema_version": P4_EXTERNAL_ANCHOR_REQUEST_MANIFEST_SCHEMA_VERSION_V2,
        "anchor_request_protocol_id": protocol.protocol_id,
        "request_id": request_id,
        "status": protocol.status,
        "files": [
            {
                "path": _REQUEST_FILE,
                "byte_count": len(request_bytes),
                "sha256": _sha256(request_bytes),
                "git_blob_oid_sha1": _git_oid("blob", request_bytes),
            }
        ],
        "publication_request_contract_frozen": True,
        "external_request_dispatched": False,
        "publication_object_exists_observed": False,
        "publisher_action_or_identity_proven": False,
        "external_publication_observed": False,
        "external_publication_anchor_present": False,
        "external_anchor_admitted": False,
        "A1_contract_and_materializer_implementation_authorized": False,
        "A1_required_before_materialization": True,
        "structural_inventory_materialization_authorized": False,
        "source_execution_authorized": False,
        "tuple_feasibility_authorized": False,
        "power_search_authorized": False,
        "model_output_authorized": False,
        "CUDA_planner_authorized": False,
        "development_authorized": False,
        "qualification_authorized": False,
        "formal_authorized": False,
        "network_request_count": 0,
        "Git_commit_count": 0,
        "Git_push_count": 0,
        "external_publication_count": 0,
        "external_receipt_count": 0,
        "A0_admission_count": 0,
        "source_structural_inventory_artifact_count": 0,
        "model_output_count": 0,
        "CUDA_run_count": 0,
        "empirical_outcome_count": 0,
        "claim_boundary": (
            "Local A0 v2 publication request only; the acceptance mechanism is pinned but no external "
            "publication, observation, admission, source, model, CUDA, or four-axis evidence exists."
        ),
    }


def _validate_request_root(
    output: pathlib.Path,
    *,
    protocol: RelationshipP4ExternalAnchorProtocolV2,
    raw: Mapping[str, Any],
) -> RelationshipP4ExternalAnchorRequestV2:
    root = _plain_directory(output, (_REQUEST_FILE, _MANIFEST_FILE), "A0 v2 request artifact")
    request_bytes = _plain_file_buffer(root / _REQUEST_FILE, "A0 v2 request payload")[1]
    request = _strict_json_object(request_bytes, "A0 v2 request payload")
    if request_bytes != _canonical_bytes(request):
        raise ValueError("A0 v2 request is not canonical JSON")
    request_id = _require_sha256(request.get("request_id"), "A0 v2 request id")
    request_core = dict(request)
    del request_core["request_id"]
    if request_id != _sha256(_canonical_bytes(request_core)):
        raise ValueError("A0 v2 request id does not close")
    _require_literal(request_core, _request_core(protocol, raw), "A0 v2 request payload")
    if request_id != P4_EXTERNAL_ANCHOR_REQUEST_ID_V2:
        raise ValueError("A0 v2 frozen request id drift")
    manifest_bytes = _plain_file_buffer(root / _MANIFEST_FILE, "A0 v2 request manifest")[1]
    manifest = _strict_json_object(manifest_bytes, "A0 v2 request manifest")
    if manifest_bytes != _canonical_bytes(manifest):
        raise ValueError("A0 v2 request manifest is not canonical JSON")
    artifact_id = _require_sha256(manifest.get("artifact_id"), "A0 v2 artifact id")
    manifest_core = dict(manifest)
    del manifest_core["artifact_id"]
    if artifact_id != _sha256(_canonical_bytes(manifest_core)):
        raise ValueError("A0 v2 artifact id does not close")
    _require_literal(manifest_core, _manifest_core(protocol, request_id, request_bytes), "A0 v2 manifest")
    if artifact_id != P4_EXTERNAL_ANCHOR_REQUEST_ARTIFACT_ID_V2:
        raise ValueError("A0 v2 frozen artifact id drift")
    firewall = _mapping(request["authorization_firewall"], "A0 v2 request firewall")
    return RelationshipP4ExternalAnchorRequestV2(
        artifact_id=artifact_id,
        request_id=request_id,
        protocol_id=protocol.protocol_id,
        status=protocol.status,
        publication_request_contract_frozen=_boolean(
            firewall["publication_request_contract_frozen"], "A0 v2 request frozen"
        ),
        external_request_dispatched=_boolean(
            firewall["external_request_dispatched"], "A0 v2 request dispatched"
        ),
        publication_object_exists_observed=_boolean(
            firewall["publication_object_exists_observed"], "A0 v2 object observed"
        ),
        publisher_action_or_identity_proven=_boolean(
            firewall["publisher_action_or_identity_proven"], "A0 v2 publisher identity proven"
        ),
        external_publication_observed=_boolean(
            firewall["external_publication_observed"], "A0 v2 publication observed"
        ),
        external_publication_anchor_present=_boolean(
            firewall["external_publication_anchor_present"], "A0 v2 anchor present"
        ),
        external_anchor_admitted=_boolean(firewall["external_anchor_admitted"], "A0 v2 anchor admitted"),
        a1_contract_and_materializer_implementation_authorized=_boolean(
            firewall["A1_contract_and_materializer_implementation_authorized"],
            "A0 v2 A1 implementation authority",
        ),
        structural_inventory_materialization_authorized=_boolean(
            firewall["structural_inventory_materialization_authorized"],
            "A0 v2 materialization authority",
        ),
        source_execution_authorized=_boolean(
            firewall["source_execution_authorized"], "A0 v2 source authority"
        ),
        model_output_authorized=_boolean(firewall["model_output_authorized"], "A0 v2 model authority"),
        cuda_planner_authorized=_boolean(firewall["CUDA_planner_authorized"], "A0 v2 CUDA authority"),
        output_dir=root,
    )


def _canonical_output(raw: Mapping[str, Any]) -> pathlib.Path:
    binding = _mapping(raw["self_publication_binding"], "A0 v2 binding")
    request = pathlib.PurePosixPath(
        _safe_repo_path(binding["request_payload_repo_relative_path"], "A0 v2 request path")
    )
    manifest = pathlib.PurePosixPath(
        _safe_repo_path(binding["request_manifest_repo_relative_path"], "A0 v2 manifest path")
    )
    if request.name != _REQUEST_FILE or manifest.name != _MANIFEST_FILE or request.parent != manifest.parent:
        raise ValueError("A0 v2 canonical request/manifest path drift")
    return _absolute_without_resolving(_REPOSITORY_ROOT.joinpath(*request.parent.parts))


def _validate_descriptor_bytes(descriptor: Mapping[str, Any], payload: bytes, label: str) -> None:
    if _integer(descriptor["byte_count"], f"{label} byte count") != len(payload):
        raise ValueError(f"{label} byte count drift")
    if _require_sha256(descriptor["raw_sha256"], f"{label} SHA-256") != _sha256(payload):
        raise ValueError(f"{label} SHA-256 drift")
    expected_oid = _text(descriptor["git_blob_oid_sha1"], f"{label} Git blob OID")
    if expected_oid != _git_oid("blob", payload):
        raise ValueError(f"{label} Git blob OID drift")
    if descriptor.get("expected_eol") == "LF" and (b"\r\n" in payload or not payload.endswith(b"\n")):
        raise ValueError(f"{label} LF contract drift")


def _plain_directory(path: pathlib.Path, expected_names: tuple[str, ...], label: str) -> pathlib.Path:
    root = _require_local_default_stream_path(path, label)
    _reject_reparse_components(root, label)
    if not root.is_dir():
        raise FileNotFoundError(f"{label} is missing: {root}")
    entries = tuple(sorted(item.name for item in root.iterdir()))
    if entries != tuple(sorted(expected_names)):
        raise ValueError(f"{label} file set drift")
    for name in expected_names:
        _plain_file_metadata(root / name, f"{label} file")
    return root


def _plain_file_buffer(path: os.PathLike[str] | str, label: str) -> tuple[pathlib.Path, bytes]:
    candidate, before = _plain_file_metadata(path, label)
    with candidate.open("rb", buffering=0) as stream:
        payload = stream.read(_MAX_LOCAL_INPUT_BYTES + 1)
        during = os.fstat(stream.fileno())
    after = os.lstat(candidate)
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    during_identity = (during.st_dev, during.st_ino, during.st_size, during.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != during_identity or before_identity != after_identity:
        raise ValueError(f"{label} changed during one-buffer read")
    if (
        len(payload) > _MAX_LOCAL_INPUT_BYTES
        or len(payload) != during.st_size
        or during.st_nlink != 1
        or after.st_nlink != 1
        or stat.S_ISLNK(after.st_mode)
        or not stat.S_ISREG(after.st_mode)
    ):
        raise ValueError(f"{label} violated its bounded unique regular-file contract")
    _reject_reparse_components(candidate, label)
    return candidate, payload


def _plain_file_metadata(
    path: os.PathLike[str] | str,
    label: str,
) -> tuple[pathlib.Path, os.stat_result]:
    candidate = _require_local_default_stream_path(path, label)
    _reject_reparse_components(candidate, label)
    try:
        metadata = os.lstat(candidate)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} is missing: {candidate}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise FileNotFoundError(f"{label} must be a regular file: {candidate}")
    if metadata.st_nlink != 1:
        raise ValueError(f"{label} must have exactly one hard link")
    return candidate, metadata


def _absolute_without_resolving(path: os.PathLike[str] | str) -> pathlib.Path:
    return pathlib.Path(os.path.abspath(os.fspath(path)))


def _require_local_default_stream_path(path: os.PathLike[str] | str, label: str) -> pathlib.Path:
    raw = os.fspath(path)
    if type(raw) is not str or not raw or "\x00" in raw:
        raise ValueError(f"{label} must be a non-empty local text path")
    if os.name == "nt":
        windows = raw.replace("/", "\\")
        drive, remainder = os.path.splitdrive(windows)
        if windows.startswith("\\\\") or drive.startswith("\\\\"):
            raise ValueError(f"{label} must not use UNC or device namespaces")
        if ":" in remainder:
            raise ValueError(f"{label} must use the NTFS default data stream")
    absolute = _absolute_without_resolving(raw)
    if os.name == "nt" and str(absolute).replace("/", "\\").startswith("\\\\"):
        raise ValueError(f"{label} must remain local")
    return absolute


def _reject_reparse_components(path: pathlib.Path, label: str) -> None:
    for candidate in (path, *path.parents):
        if not os.path.lexists(candidate):
            continue
        metadata = os.lstat(candidate)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} must not traverse a symlink: {candidate}")
        if os.name == "nt" and metadata.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError(f"{label} must not traverse a reparse point: {candidate}")


def _strict_json_value(payload: bytes, label: str) -> Any:
    if payload.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{label} must not carry a UTF-8 BOM")
    if b"\x00" in payload:
        raise ValueError(f"{label} must not contain NUL")
    try:
        text = payload.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_raise_json_constant(value)),
        )
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not strict UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON") from exc


def _strict_json_object(payload: bytes, label: str) -> dict[str, Any]:
    value = _strict_json_value(payload, label)
    if type(value) is not dict:
        raise ValueError(f"{label} root must be an object")
    return value


def _raise_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_lf_text(payload: bytes, label: str) -> str:
    if payload.startswith(b"\xef\xbb\xbf") or b"\r" in payload or not payload.endswith(b"\n"):
        raise ValueError(f"{label} must be BOM-free LF text ending in LF")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not strict UTF-8") from exc


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_create(path: pathlib.Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_oid(kind: str, payload: bytes) -> str:
    framed = kind.encode("ascii") + b" " + str(len(payload)).encode("ascii") + b"\0" + payload
    return hashlib.sha1(framed, usedforsecurity=False).hexdigest()


def _safe_repo_path(value: object, label: str) -> str:
    text = _text(value, label)
    path = pathlib.PurePosixPath(text)
    if (
        path.is_absolute()
        or str(path) != text
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in text
        or ":" in text
        or "%" in text
    ):
        raise ValueError(f"{label} must be an unambiguous repository-relative POSIX path")
    return text


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be an object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str] | frozenset[str], label: str) -> None:
    missing = sorted(set(expected) - set(value))
    extra = sorted(set(value) - set(expected))
    if missing or extra:
        raise ValueError(f"{label} keys drift; missing={missing}, extra={extra}")


def _require_literal(actual: object, expected: object, label: str) -> None:
    if type(actual) is not type(expected):
        raise TypeError(f"{label} type drift")
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        _require_exact_keys(actual, set(expected), label)
        for key, expected_value in expected.items():
            _require_literal(actual[key], expected_value, f"{label}.{key}")
        return
    if isinstance(expected, list):
        assert isinstance(actual, list)
        if len(actual) != len(expected):
            raise ValueError(f"{label} length drift")
        for index, expected_value in enumerate(expected):
            _require_literal(actual[index], expected_value, f"{label}[{index}]")
        return
    if actual != expected:
        raise ValueError(f"{label} value drift")


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise TypeError(f"{label} must be non-empty text")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    return value


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be boolean")
    return value


def _require_sha256(value: object, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


__all__ = (
    "P4_EXTERNAL_ANCHOR_PROTOCOL_ID_V2",
    "P4_EXTERNAL_ANCHOR_PROTOCOL_RAW_SHA256_V2",
    "P4_EXTERNAL_ANCHOR_PROTOCOL_SCHEMA_VERSION_V2",
    "P4_EXTERNAL_ANCHOR_REQUEST_ARTIFACT_ID_V2",
    "P4_EXTERNAL_ANCHOR_REQUEST_ID_V2",
    "P4_EXTERNAL_ANCHOR_REQUEST_MANIFEST_SCHEMA_VERSION_V2",
    "P4_EXTERNAL_ANCHOR_REQUEST_SCHEMA_VERSION_V2",
    "P4_EXTERNAL_ANCHOR_REQUEST_STATUS_V2",
    "RelationshipP4ExternalAnchorProtocolV2",
    "RelationshipP4ExternalAnchorRequestV2",
    "load_protocol",
    "prepare_request",
    "protocol_path",
    "validate_request",
)
