"""Identity-free, non-authorizing acquisition for a public P4.7 A0 Gist anchor.

The adapter deliberately stops at byte capture.  A separate verifier owns every
semantic join and every authority decision.  Production acquisition uses only
stdlib ``http.client`` plus an explicitly selected absolute Git executable.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import pathlib
import re
import secrets
import ssl
import stat
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, Sequence
from urllib.parse import urlsplit


HTTP_METHOD = "GET"
GITHUB_API_VERSION = "2026-03-10"
HTTP_ACCEPT = "application/vnd.github+json"
HTTP_USER_AGENT = "volvence-a0-gist-observer/1"
HTTP_ACCEPT_ENCODING = "identity"
HTTP_CACHE_CONTROL = "no-cache"
HTTP_PRAGMA = "no-cache"

ROLE_API = "api"
ROLE_RAW = "raw"
ROLE_HTML_GIT = "html_git"
ROLE_HOSTS = MappingProxyType(
    {
        ROLE_API: "api.github.com",
        ROLE_RAW: "gist.githubusercontent.com",
        ROLE_HTML_GIT: "gist.github.com",
    }
)

REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
MAX_REDIRECTS = 3
MAX_REDIRECTS_BY_ROLE = MappingProxyType(
    {
        ROLE_API: 0,
        ROLE_RAW: 3,
        ROLE_HTML_GIT: 3,
    }
)
SUCCESS_STATUS = 200
MAX_URL_BYTES = 2_048
MAX_HEADER_BYTES = 65_536
MAX_HEADER_COUNT = 100
MAX_API_BODY_BYTES = 262_144
MAX_HTML_BODY_BYTES = 2_097_152
CONNECT_TIMEOUT_SECONDS = 10.0
READ_TIMEOUT_SECONDS = 10.0
HTTP_OVERALL_TIMEOUT_SECONDS = 30.0
GIT_TOTAL_TIMEOUT_SECONDS = 120.0
GIT_REAP_TIMEOUT_SECONDS = 10.0
MAX_GIT_PROCESS_STREAM_BYTES = 4 * 1024 * 1024
MAX_GIT_OBJECT_STORE_BYTES = 4 * 1024 * 1024
MAX_GIT_REPOSITORY_METADATA_BYTES = 4 * 1024 * 1024
MAX_GIT_STDIN_BYTES = 4_096
RETRY_COUNT = 0

PRODUCTION_GIT_EXECUTABLE = pathlib.PureWindowsPath(r"C:\Program Files\Git\mingw64\libexec\git-core\git.exe")
PRODUCTION_GIT_EXECUTABLE_BYTE_COUNT = 3_872_760
PRODUCTION_GIT_EXECUTABLE_RAW_SHA256 = "bf50371f964f7be61a76ebad7dce8b6197afd47b5b588fc62780823aa097ff63"
PRODUCTION_GIT_VERSION_STDOUT = "git version 2.44.0.windows.1"
PRODUCTION_GIT_HELPER_BYTE_COUNT = 2_375_176
PRODUCTION_GIT_HELPER_RAW_SHA256 = "e6bc0697e9405ffc7e94abaecd33e924d8b8b1634b57738629515b237c568c60"
PRODUCTION_GIT_HELPER_PATHS = (
    pathlib.PureWindowsPath(r"C:\Program Files\Git\mingw64\libexec\git-core\git-remote-http.exe"),
    pathlib.PureWindowsPath(r"C:\Program Files\Git\mingw64\libexec\git-core\git-remote-https.exe"),
)

CLAIM_FILE = "000_observer_claim.json"
API_START_HTTP_FILE = "010_api_revision_start.http.json"
API_START_BODY_FILE = "011_api_revision_start.body"
RAW_HTTP_FILE = "020_returned_raw.http.json"
RAW_BODY_FILE = "021_returned_raw.body"
HTML_HTTP_FILE = "030_returned_html.http.json"
HTML_BODY_FILE = "031_returned_html.body"
GIT_CAPTURE_FILE = "040_git_capture.json"
GIT_ADVERTISED_REFS_FILE = "040_git_advertised_refs.body"
GIT_FETCHED_REFS_FILE = "040_git_fetched_refs.body"
GIT_OBJECT_INVENTORY_FILE = "040_git_object_inventory.body"
GIT_COMMIT_FILE = "041_git_commit.body"
GIT_TREE_FILE = "042_git_tree.body"
GIT_BLOB_FILE = "043_git_blob.body"
API_END_HTTP_FILE = "050_api_revision_end.http.json"
API_END_BODY_FILE = "051_api_revision_end.body"
CAPTURE_MAP_FILE = "090_capture_map.json"
TERMINAL_FILE = "099_terminal.json"

_CAPTURE_SEQUENCE = (
    "api_exact_revision_start",
    "returned_raw",
    "returned_html",
    "fresh_isolated_bare_git",
    "api_exact_revision_end",
)
_OBSERVATION_STAGES = frozenset({"R0", "R1"})
_LOWER_HEX = frozenset("0123456789abcdef")
_OWNER_LOGIN_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?\Z")
_SAFE_FILENAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}\Z")
_SAFE_NODE_ID_PATTERN = re.compile(r"[A-Za-z0-9_+/=-]{1,256}\Z")
_AUTHORITY_FIREWALL = MappingProxyType(
    {
        "external_request_dispatched": False,
        "publication_object_exists_observed": False,
        "publisher_action_or_identity_proven": False,
        "external_publication_observed": False,
        "external_publication_anchor_present": False,
        "external_anchor_admitted": False,
        "A1_contract_and_materializer_implementation_authorized": False,
        "structural_inventory_materialization_authorized": False,
        "source_execution_authorized": False,
        "tuple_feasibility_authorized": False,
        "power_search_authorized": False,
        "model_output_authorized": False,
        "CUDA_planner_authorized": False,
        "development_authorized": False,
        "qualification_authorized": False,
        "formal_authorized": False,
        "appendable_formal_supported": False,
        "readable_formal_supported": False,
        "learnable_formal_supported": False,
        "steerable_formal_supported": False,
        "integrated_four_axis_supported": False,
    }
)
_REQUEST_HEADER_ITEMS = (
    ("Accept", HTTP_ACCEPT),
    ("User-Agent", HTTP_USER_AGENT),
    ("Accept-Encoding", HTTP_ACCEPT_ENCODING),
    ("Cache-Control", HTTP_CACHE_CONTROL),
    ("Pragma", HTTP_PRAGMA),
    ("X-GitHub-Api-Version", GITHUB_API_VERSION),
)
_FORBIDDEN_REQUEST_HEADER_NAMES = frozenset(
    {
        "authorization",
        "cookie",
        "if-match",
        "if-modified-since",
        "if-none-match",
        "if-range",
        "proxy-authorization",
        "range",
    }
)
_GIT_AMBIENT_ALLOWLIST_WINDOWS = (
    "SYSTEMROOT",
    "WINDIR",
)
_GIT_FIXED_ENVIRONMENT = MappingProxyType(
    {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GCM_INTERACTIVE": "Never",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PROTOCOL_FROM_USER": "0",
        "GIT_HTTP_USER_AGENT": HTTP_USER_AGENT,
        "LANG": "C",
        "LC_ALL": "C",
    }
)
_CLAIM_BOUNDARY = (
    "Facts-only external acquisition. This adapter does not publish a Gist, authenticate, retry, "
    "verify the A0 object graph, admit an external anchor, authorize A1, materialize source rows, "
    "run a model or CUDA, or establish any four-axis evidence."
)

_PRE_MAP_CAPTURE_ROLES = MappingProxyType(
    {
        CLAIM_FILE: "claim",
        API_START_HTTP_FILE: "api_exact_revision_start_http",
        API_START_BODY_FILE: "api_exact_revision_start_body",
        RAW_HTTP_FILE: "returned_raw_http",
        RAW_BODY_FILE: "returned_raw_body",
        HTML_HTTP_FILE: "returned_html_http",
        HTML_BODY_FILE: "returned_html_body",
        GIT_CAPTURE_FILE: "git_capture",
        GIT_ADVERTISED_REFS_FILE: "git_advertised_refs_raw_stdout",
        GIT_FETCHED_REFS_FILE: "git_fetched_refs_raw_stdout",
        GIT_OBJECT_INVENTORY_FILE: "git_object_inventory_raw_stdout",
        GIT_COMMIT_FILE: "git_commit_body",
        GIT_TREE_FILE: "git_tree_body",
        GIT_BLOB_FILE: "git_blob_body",
        API_END_HTTP_FILE: "api_exact_revision_end_http",
        API_END_BODY_FILE: "api_exact_revision_end_body",
    }
)

OBSERVER_FAILURE_CODES = frozenset(
    {
        "api_bookend_projection_drift",
        "api_content_type_invalid",
        "api_file_encoding_invalid",
        "api_file_missing",
        "api_file_set_invalid",
        "api_file_size_mismatch",
        "api_file_truncated",
        "api_filename_mismatch",
        "api_gist_id_mismatch",
        "api_gist_node_id_invalid",
        "api_git_url_path_mismatch",
        "api_history_invalid",
        "api_html_url_path_mismatch",
        "api_inline_content_mismatch",
        "api_inline_content_missing",
        "api_owner_mismatch",
        "api_public_shape_invalid",
        "api_raw_url_missing",
        "api_raw_url_path_mismatch",
        "api_shape_invalid",
        "api_truncated",
        "body_cap_exceeded",
        "capture_root_closure_invalid",
        "content_encoding_duplicated",
        "content_encoding_invalid",
        "content_length_invalid",
        "content_length_mismatch",
        "content_type_duplicated",
        "git_advertised_head_count_invalid",
        "git_advertised_head_mismatch",
        "git_advertised_raw_drift",
        "git_advertised_ref_scope_invalid",
        "git_advertisement_encoding",
        "git_advertisement_invalid",
        "git_blob_invalid",
        "git_blob_oid_mismatch",
        "git_capture_cap_exceeded",
        "git_cat_file_framing_invalid",
        "git_cat_file_header_invalid",
        "git_cat_file_header_missing",
        "git_command_failed",
        "git_commit_has_parent",
        "git_commit_headers_invalid",
        "git_commit_oid_mismatch",
        "git_commit_tree_encoding",
        "git_commit_tree_invalid",
        "git_commit_tree_mismatch",
        "git_executable_identity_drift",
        "git_executable_invalid",
        "git_executable_missing",
        "git_executable_not_absolute",
        "git_executable_path_drift",
        "git_environment_invalid",
        "git_fetched_head_mismatch",
        "git_fetched_raw_drift",
        "git_filename_invalid",
        "git_helper_identity_drift",
        "git_job_assignment_failed",
        "git_job_configuration_failed",
        "git_job_create_failed",
        "git_job_query_failed",
        "git_job_termination_failed",
        "git_local_config_forbidden",
        "git_object_body_cap_exceeded",
        "git_object_inventory_count_invalid",
        "git_object_inventory_encoding",
        "git_object_inventory_invalid",
        "git_object_inventory_mismatch",
        "git_object_oid_mismatch",
        "git_object_store_cap_exceeded",
        "git_preflight_failed",
        "git_process_output_cap_exceeded",
        "git_process_input_cap_exceeded",
        "git_process_start_failed",
        "git_process_stream_failure",
        "git_process_tree_reap_timeout",
        "git_raw_blob_mismatch",
        "git_ref_count_invalid",
        "git_ref_output_encoding",
        "git_ref_output_invalid",
        "git_ref_scope_invalid",
        "git_repository_escape_surface",
        "git_repository_metadata_invalid",
        "git_required_entry_invalid",
        "git_revision_commit_mismatch",
        "git_revision_not_commit",
        "git_tool_changed_during_read",
        "git_tool_identity_mismatch",
        "git_tool_not_unique_regular_file",
        "git_toolchain_identity_drift",
        "git_total_timeout",
        "git_thread_resume_failed",
        "git_tree_binding_mismatch",
        "git_tree_encoding_invalid",
        "git_tree_entry_mode_invalid",
        "git_tree_entry_name_invalid",
        "git_tree_framing_invalid",
        "git_tree_invalid",
        "git_tree_oid_mismatch",
        "git_tree_shape_invalid",
        "git_version_drift",
        "git_version_encoding_invalid",
        "header_bytes_exceeded",
        "header_count_exceeded",
        "header_encoding_invalid",
        "header_folding_rejected",
        "header_type_invalid",
        "html_content_type_invalid",
        "http_framing_ambiguous",
        "http_overall_timeout",
        "http_role_drift",
        "http_socket_missing",
        "http_status_not_200",
        "invalid_body_cap",
        "json_duplicate_key",
        "json_invalid",
        "json_nonfinite_number",
        "json_not_utf8",
        "json_root_invalid",
        "raw_request_bytes_mismatch",
        "redirect_chain_invalid",
        "redirect_final_url_drift",
        "redirect_header_facts_drift",
        "redirect_limit_exceeded",
        "redirect_location_invalid",
        "redirect_status_invalid",
        "role_url_invalid",
        "role_url_noncanonical",
        "synthetic_call_order_drift",
        "synthetic_fixture_exhausted",
        "synthetic_git_filename_drift",
        "synthetic_git_identity_drift",
        "system_trust_store_missing",
        "transfer_encoding_duplicated",
        "transfer_encoding_invalid",
        "unclassified_acquisition_failure",
        "unknown_role",
        "url_invalid",
        "url_lexical_invalid",
        "url_non_ascii",
    }
)


class ObserverError(RuntimeError):
    """A bounded acquisition or contract failure."""

    def __init__(self, code: str, message: str) -> None:
        if code not in OBSERVER_FAILURE_CODES:
            raise ValueError(f"observer failure code is not frozen: {code!r}")
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ObserverTarget:
    """Frozen A0 lineage, public identity, and request byte commitments."""

    observation_stage: str
    predecessor_receipt_id: str | None
    predecessor_receipt_bundle_manifest_raw_sha256: str | None
    protocol_id: str
    protocol_raw_sha256: str
    protocol_raw_byte_count: int
    request_id: str
    request_artifact_id: str
    request_raw_sha256: str
    request_raw_byte_count: int
    request_manifest_raw_sha256: str
    request_manifest_raw_byte_count: int
    gist_id: str
    revision_oid: str
    expected_owner_login: str
    expected_owner_id: int
    expected_owner_node_id: str
    required_filename: str

    def __post_init__(self) -> None:
        if self.observation_stage not in _OBSERVATION_STAGES:
            raise ValueError("observation_stage must be R0 or R1")
        if self.observation_stage == "R0" and (
            self.predecessor_receipt_id is not None or self.predecessor_receipt_bundle_manifest_raw_sha256 is not None
        ):
            raise ValueError("R0 must not bind predecessor R0 receipt artifacts")
        if self.observation_stage == "R1":
            _require_lower_hex(
                self.predecessor_receipt_id,
                "R1 predecessor_receipt_id",
                minimum=64,
                maximum=64,
            )
            _require_lower_hex(
                self.predecessor_receipt_bundle_manifest_raw_sha256,
                "R1 predecessor_receipt_bundle_manifest_raw_sha256",
                minimum=64,
                maximum=64,
            )
        for label, value in (
            ("protocol_id", self.protocol_id),
            ("protocol_raw_sha256", self.protocol_raw_sha256),
            ("request_id", self.request_id),
            ("request_artifact_id", self.request_artifact_id),
            ("request_raw_sha256", self.request_raw_sha256),
            ("request_manifest_raw_sha256", self.request_manifest_raw_sha256),
        ):
            _require_lower_hex(value, label, minimum=64, maximum=64)
        for label, value in (
            ("protocol_raw_byte_count", self.protocol_raw_byte_count),
            ("request_raw_byte_count", self.request_raw_byte_count),
            ("request_manifest_raw_byte_count", self.request_manifest_raw_byte_count),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        _require_lower_hex(self.gist_id, "gist_id", minimum=20, maximum=64)
        _require_lower_hex(self.revision_oid, "revision_oid", minimum=40, maximum=40)
        if not _OWNER_LOGIN_PATTERN.fullmatch(self.expected_owner_login):
            raise ValueError("expected_owner_login is not a canonical GitHub login")
        if type(self.expected_owner_id) is not int or self.expected_owner_id <= 0:
            raise ValueError("expected_owner_id must be a positive integer")
        if not _SAFE_NODE_ID_PATTERN.fullmatch(self.expected_owner_node_id):
            raise ValueError("expected_owner_node_id is invalid")
        if not _SAFE_FILENAME_PATTERN.fullmatch(self.required_filename):
            raise ValueError("required_filename is not a safe root filename")
        if self.request_raw_byte_count > MAX_HTML_BODY_BYTES:
            raise ValueError("request_raw_byte_count exceeds the fixed observer bound")

    @property
    def api_revision_url(self) -> str:
        return f"https://{ROLE_HOSTS[ROLE_API]}/gists/{self.gist_id}/{self.revision_oid}"

    @property
    def git_remote_url(self) -> str:
        return f"https://{ROLE_HOSTS[ROLE_HTML_GIT]}/{self.gist_id}.git"


@dataclass(frozen=True)
class RedirectHop:
    requested_url: str
    status: int
    location: str
    response_header_count: int
    response_header_wire_bytes: int
    response_header_ledger_bytes: int
    response_header_ledger_sha256: str
    response_headers: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class HttpObservation:
    role: str
    requested_url: str
    final_url: str
    status: int
    response_headers: tuple[tuple[str, str], ...]
    body: bytes
    redirects: tuple[RedirectHop, ...] = ()


@dataclass(frozen=True)
class GitObjectCapture:
    remote_url: str
    revision_oid: str
    commit_oid: str
    tree_oid: str
    blob_oid: str
    tree_entry_mode: str
    tree_entry_name: str
    advertised_refs: tuple[tuple[str, str], ...]
    fetched_refs: tuple[tuple[str, str], ...]
    object_inventory: tuple[tuple[str, str, int], ...]
    object_store_byte_count: int
    commit_body: bytes
    tree_body: bytes
    blob_body: bytes
    advertised_refs_body: bytes
    fetched_refs_body: bytes
    object_inventory_body: bytes
    fsck_stdout_sha256: str
    fsck_stderr_sha256: str
    git_executable_byte_count: int
    git_executable_raw_sha256: str
    git_version_stdout: str
    git_helper_identities: tuple[tuple[str, int, str], ...]
    command_argv_ledger: tuple[tuple[str, ...], ...]
    environment_ledger: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class GitExecutableIdentity:
    path: pathlib.Path
    byte_count: int
    raw_sha256: str
    version_stdout: str
    helper_identities: tuple[tuple[str, int, str], ...]


@dataclass(frozen=True)
class ObserverResult:
    output_dir: pathlib.Path
    succeeded: bool
    status: str
    claim_id: str
    capture_map_id: str
    terminal_id: str


class ObserverBackend(Protocol):
    kind: str

    def http_get(self, *, role: str, url: str, body_cap: int) -> HttpObservation: ...

    def git_capture(
        self,
        *,
        remote_url: str,
        revision_oid: str,
        required_filename: str,
        workspace: pathlib.Path,
        git_executable: pathlib.Path | None,
        git_identity: GitExecutableIdentity | None,
    ) -> GitObjectCapture: ...


class SyntheticObserverBackend:
    """Deterministic fixture backend; it never constructs a network or Git client."""

    kind = "synthetic"

    def __init__(
        self,
        *,
        http_observations: Sequence[HttpObservation],
        git_capture: GitObjectCapture,
    ) -> None:
        self._http_observations = tuple(http_observations)
        self._git_capture = git_capture
        self._http_index = 0
        self.http_calls: list[tuple[str, str, int]] = []
        self.git_calls: list[tuple[str, str, str]] = []

    def http_get(self, *, role: str, url: str, body_cap: int) -> HttpObservation:
        self.http_calls.append((role, url, body_cap))
        if self._http_index >= len(self._http_observations):
            raise ObserverError("synthetic_fixture_exhausted", "synthetic HTTP fixture is exhausted")
        observation = self._http_observations[self._http_index]
        self._http_index += 1
        if observation.role != role or observation.requested_url != url:
            raise ObserverError("synthetic_call_order_drift", "synthetic HTTP call order drift")
        _validate_http_observation(observation, expected_role=role, body_cap=body_cap)
        return observation

    def git_capture(
        self,
        *,
        remote_url: str,
        revision_oid: str,
        required_filename: str,
        workspace: pathlib.Path,
        git_executable: pathlib.Path | None,
        git_identity: GitExecutableIdentity | None,
    ) -> GitObjectCapture:
        del workspace, git_executable, git_identity
        self.git_calls.append((remote_url, revision_oid, required_filename))
        capture = self._git_capture
        if capture.remote_url != remote_url or capture.revision_oid != revision_oid:
            raise ObserverError("synthetic_git_identity_drift", "synthetic Git identity drift")
        if capture.tree_entry_name != required_filename:
            raise ObserverError("synthetic_git_filename_drift", "synthetic Git filename drift")
        _validate_git_capture(capture)
        return capture


class ProductionObserverBackend:
    """Fixed direct HTTPS plus fresh isolated bare-Git production backend."""

    kind = "production"

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock

    def http_get(self, *, role: str, url: str, body_cap: int) -> HttpObservation:
        return _direct_https_get(
            role=role,
            url=url,
            body_cap=body_cap,
            clock=self._clock,
        )

    def git_capture(
        self,
        *,
        remote_url: str,
        revision_oid: str,
        required_filename: str,
        workspace: pathlib.Path,
        git_executable: pathlib.Path | None,
        git_identity: GitExecutableIdentity | None,
    ) -> GitObjectCapture:
        if git_executable is None or git_identity is None:
            raise ObserverError("git_executable_missing", "production acquisition requires an absolute Git executable")
        return _capture_fresh_bare_git(
            git_executable=git_executable,
            git_identity=git_identity,
            remote_url=remote_url,
            revision_oid=revision_oid,
            required_filename=required_filename,
            workspace=workspace,
            clock=self._clock,
        )


def acquire_external_anchor_observation(
    *,
    output_dir: os.PathLike[str] | str,
    target: ObserverTarget,
    git_executable: os.PathLike[str] | str | None = None,
    backend: ObserverBackend | None = None,
) -> ObserverResult:
    """Create one facts-only observation root; failures are terminal and never retried."""

    if not isinstance(target, ObserverTarget):
        raise TypeError("target must be ObserverTarget")
    production = backend is None
    selected_backend: ObserverBackend = ProductionObserverBackend() if production else backend
    if production:
        backend_kind = "production"
    else:
        backend_kind = "synthetic"
    git_path = None if git_executable is None else pathlib.Path(os.path.abspath(os.fspath(git_executable)))
    if production:
        if git_path is None or not pathlib.Path(os.fspath(git_executable)).is_absolute():
            raise ValueError("production git_executable must be an absolute path")
        if os.name != "nt" or os.path.normcase(str(git_path)) != os.path.normcase(str(PRODUCTION_GIT_EXECUTABLE)):
            raise ValueError("production git_executable must equal the frozen Windows Git path")
    output = _prepare_create_only_root(output_dir)

    claim_core = {
        "schema_version": "relationship-p4-external-anchor-observer-claim.v1",
        "backend_kind": backend_kind,
        "target": _target_payload(target),
        "fixed_acquisition_contract": _fixed_acquisition_contract(),
        "authority_firewall": dict(_AUTHORITY_FIREWALL),
        "A1_required_before_materialization": True,
        "process_id": os.getpid(),
        "process_instance_nonce": secrets.token_hex(32),
        "claim_boundary": _CLAIM_BOUNDARY,
    }
    claim_id = _sha256(_canonical_bytes(claim_core))
    claim = {**claim_core, "claim_id": claim_id}
    completed_stages: list[str] = []
    _write_json_capture(output, CLAIM_FILE, claim, role="claim")

    failure: ObserverError | None = None
    git_identity: GitExecutableIdentity | None = None
    try:
        if production:
            assert git_path is not None
            git_identity = _preflight_production_git_executable(git_path, workspace=output)
        api_start = selected_backend.http_get(
            role=ROLE_API,
            url=target.api_revision_url,
            body_cap=MAX_API_BODY_BYTES,
        )
        _validate_http_observation(api_start, expected_role=ROLE_API, body_cap=MAX_API_BODY_BYTES)
        _write_http_capture(
            output,
            observation=api_start,
            metadata_file=API_START_HTTP_FILE,
            body_file=API_START_BODY_FILE,
            role="api_exact_revision_start",
            body_cap=MAX_API_BODY_BYTES,
        )
        completed_stages.append("api_exact_revision_start")
        api_start_payload = _strict_json_object(api_start.body, "API exact revision start")
        api_start_projection = _api_core_projection(api_start_payload, target)
        raw_url = str(api_start_projection["raw_url"])
        html_url = str(api_start_projection["html_url"])
        git_remote_url = str(api_start_projection["git_pull_url"])

        raw_observation = selected_backend.http_get(
            role=ROLE_RAW,
            url=raw_url,
            body_cap=target.request_raw_byte_count,
        )
        _validate_http_observation(
            raw_observation,
            expected_role=ROLE_RAW,
            body_cap=target.request_raw_byte_count,
        )
        if (
            len(raw_observation.body) != target.request_raw_byte_count
            or _sha256(raw_observation.body) != target.request_raw_sha256
        ):
            raise ObserverError("raw_request_bytes_mismatch", "returned raw bytes differ from request commitment")
        _write_http_capture(
            output,
            observation=raw_observation,
            metadata_file=RAW_HTTP_FILE,
            body_file=RAW_BODY_FILE,
            role="returned_raw",
            body_cap=target.request_raw_byte_count,
        )
        completed_stages.append("returned_raw")

        html_observation = selected_backend.http_get(
            role=ROLE_HTML_GIT,
            url=html_url,
            body_cap=MAX_HTML_BODY_BYTES,
        )
        _validate_http_observation(
            html_observation,
            expected_role=ROLE_HTML_GIT,
            body_cap=MAX_HTML_BODY_BYTES,
        )
        _write_http_capture(
            output,
            observation=html_observation,
            metadata_file=HTML_HTTP_FILE,
            body_file=HTML_BODY_FILE,
            role="returned_html",
            body_cap=MAX_HTML_BODY_BYTES,
        )
        completed_stages.append("returned_html")

        git_capture = selected_backend.git_capture(
            remote_url=git_remote_url,
            revision_oid=target.revision_oid,
            required_filename=target.required_filename,
            workspace=output,
            git_executable=git_path,
            git_identity=git_identity,
        )
        _validate_git_capture(git_capture)
        if git_capture.blob_body != raw_observation.body:
            raise ObserverError("git_raw_blob_mismatch", "Git blob bytes differ from returned raw bytes")
        _write_git_capture(output, git_capture, production=production)
        completed_stages.append("fresh_isolated_bare_git")

        api_end = selected_backend.http_get(
            role=ROLE_API,
            url=target.api_revision_url,
            body_cap=MAX_API_BODY_BYTES,
        )
        _validate_http_observation(api_end, expected_role=ROLE_API, body_cap=MAX_API_BODY_BYTES)
        _write_http_capture(
            output,
            observation=api_end,
            metadata_file=API_END_HTTP_FILE,
            body_file=API_END_BODY_FILE,
            role="api_exact_revision_end",
            body_cap=MAX_API_BODY_BYTES,
        )
        completed_stages.append("api_exact_revision_end")
        api_end_payload = _strict_json_object(api_end.body, "API exact revision end")
        api_end_projection = _api_core_projection(api_end_payload, target)
        if api_end_projection != api_start_projection:
            raise ObserverError("api_bookend_projection_drift", "API bookend core projection changed")
    except ObserverError as exc:
        failure = exc
    except Exception as exc:  # process boundary: terminalize every ordinary acquisition failure
        del exc
        failure = ObserverError(
            "unclassified_acquisition_failure",
            "acquisition failed at a nonclassified boundary",
        )

    actual_files, root_anomalies = _inventory_pre_map_capture_root(output)
    actual_names = [str(entry["path"]) for entry in actual_files]
    expected_names = sorted(_PRE_MAP_CAPTURE_ROLES)
    unexpected_names = sorted(set(actual_names) - set(expected_names))
    missing_names = sorted(set(expected_names) - set(actual_names))
    exact_complete_root = not root_anomalies and not unexpected_names and not missing_names
    if failure is None and not exact_complete_root:
        failure = ObserverError(
            "capture_root_closure_invalid",
            "successful acquisition did not close an exact pre-map capture root",
        )
    if failure is None:
        root_closure_status = "complete_exact_pre_map_root"
    elif root_anomalies or unexpected_names:
        root_closure_status = "invalid_failure_pre_map_root"
    elif not missing_names:
        root_closure_status = "complete_but_failed_pre_map_root"
    else:
        root_closure_status = "partial_failure_pre_map_root"

    map_core = {
        "schema_version": "relationship-p4-external-anchor-observer-capture-map.v1",
        "claim_id": claim_id,
        "backend_kind": backend_kind,
        "observation_stage": target.observation_stage,
        "capture_sequence": list(_CAPTURE_SEQUENCE),
        "completed_stages": completed_stages,
        "files": actual_files,
        "expected_pre_map_files": expected_names,
        "actual_pre_map_files": actual_names,
        "missing_pre_map_files": missing_names,
        "unexpected_pre_map_files": unexpected_names,
        "root_anomalies": root_anomalies,
        "root_closure_status": root_closure_status,
        "acquisition_complete": failure is None,
        "failure_code": None if failure is None else failure.code,
        "retry_count": RETRY_COUNT,
        "root_anomaly_count": len(root_anomalies),
        "authority_firewall": dict(_AUTHORITY_FIREWALL),
        "A1_required_before_materialization": True,
    }
    capture_map_id = _sha256(_canonical_bytes(map_core))
    capture_map = {**map_core, "capture_map_id": capture_map_id}
    map_entry = _write_json_capture(output, CAPTURE_MAP_FILE, capture_map, role="capture_map")

    terminal_core = {
        "schema_version": "relationship-p4-external-anchor-observer-terminal.v1",
        "claim_id": claim_id,
        "capture_map_id": capture_map_id,
        "capture_map_raw_sha256": map_entry["sha256"],
        "backend_kind": backend_kind,
        "observation_stage": target.observation_stage,
        "status": (
            "facts_only_observation_complete_non_authorizing"
            if failure is None
            else "facts_only_observation_failed_non_authorizing"
        ),
        "acquisition_complete": failure is None,
        "failure": (
            None
            if failure is None
            else {
                "code": failure.code,
                "message": str(failure),
            }
        ),
        "retry_count": RETRY_COUNT,
        "root_closure_status": root_closure_status,
        "root_anomaly_count": len(root_anomalies),
        "authority_firewall": dict(_AUTHORITY_FIREWALL),
        "A1_required_before_materialization": True,
        "claim_boundary": _CLAIM_BOUNDARY,
    }
    terminal_id = _sha256(_canonical_bytes(terminal_core))
    terminal = {**terminal_core, "terminal_id": terminal_id}
    _write_json_capture(output, TERMINAL_FILE, terminal, role="terminal")
    return ObserverResult(
        output_dir=output,
        succeeded=failure is None,
        status=terminal_core["status"],
        claim_id=claim_id,
        capture_map_id=capture_map_id,
        terminal_id=terminal_id,
    )


def _target_payload(target: ObserverTarget) -> dict[str, object]:
    return {
        "observation_stage": target.observation_stage,
        "predecessor_receipt_id": target.predecessor_receipt_id,
        "predecessor_receipt_bundle_manifest_raw_sha256": (target.predecessor_receipt_bundle_manifest_raw_sha256),
        "protocol_id": target.protocol_id,
        "protocol_raw_sha256": target.protocol_raw_sha256,
        "protocol_raw_byte_count": target.protocol_raw_byte_count,
        "request_id": target.request_id,
        "request_artifact_id": target.request_artifact_id,
        "request_raw_sha256": target.request_raw_sha256,
        "request_raw_byte_count": target.request_raw_byte_count,
        "request_manifest_raw_sha256": target.request_manifest_raw_sha256,
        "request_manifest_raw_byte_count": target.request_manifest_raw_byte_count,
        "gist_id": target.gist_id,
        "revision_oid": target.revision_oid,
        "expected_owner_login": target.expected_owner_login,
        "expected_owner_id": target.expected_owner_id,
        "expected_owner_node_id": target.expected_owner_node_id,
        "required_filename": target.required_filename,
        "local_protocol_request_manifest_buffers_recomputed_by_observer": False,
        "local_buffer_recomputation_owner": "separate_pinned_verifier",
    }


def _fixed_acquisition_contract() -> dict[str, object]:
    return {
        "method": HTTP_METHOD,
        "github_api_version": GITHUB_API_VERSION,
        "request_headers": {key: value for key, value in _REQUEST_HEADER_ITEMS},
        "derived_Host_request_header_by_role": dict(ROLE_HOSTS),
        "role_hosts": dict(ROLE_HOSTS),
        "same_role_manual_redirect_statuses": sorted(REDIRECT_STATUSES),
        "maximum_redirect_count_by_role": dict(MAX_REDIRECTS_BY_ROLE),
        "success_status": SUCCESS_STATUS,
        "maximum_header_bytes": MAX_HEADER_BYTES,
        "maximum_header_count": MAX_HEADER_COUNT,
        "header_wire_byte_cap_metric": "ordered_original_name_value_pairs_as_ISO-8859-1",
        "header_ledger_metric": "ordered_pairs_with_Set-Cookie_value_redacted_and_indexed_length_hash_facts",
        "maximum_api_body_bytes": MAX_API_BODY_BYTES,
        "maximum_html_body_bytes": MAX_HTML_BODY_BYTES,
        "raw_body_cap_is_expected_request_byte_count": True,
        "connect_timeout_seconds": int(CONNECT_TIMEOUT_SECONDS),
        "read_timeout_seconds": int(READ_TIMEOUT_SECONDS),
        "HTTP_overall_timeout_seconds": int(HTTP_OVERALL_TIMEOUT_SECONDS),
        "retry_count": RETRY_COUNT,
        "git_total_timeout_seconds": int(GIT_TOTAL_TIMEOUT_SECONDS),
        "git_process_execution_platform": "Windows_only_production",
        "git_process_creation_sequence": [
            "CREATE_SUSPENDED_and_CREATE_NO_WINDOW",
            "assign_root_process_to_noninheritable_Job_Object",
            "resume_exactly_one_primary_thread",
        ],
        "git_job_KILL_ON_JOB_CLOSE": True,
        "git_job_breakaway_allowed": False,
        "git_timeout_or_failure_terminates_whole_job_tree": True,
        "git_success_requires_root_exit_and_Job_ActiveProcesses_zero": True,
        "git_process_tree_reap_timeout_seconds": int(GIT_REAP_TIMEOUT_SECONDS),
        "maximum_git_process_stdout_bytes_per_command": MAX_GIT_PROCESS_STREAM_BYTES,
        "maximum_git_process_stderr_bytes_per_command": MAX_GIT_PROCESS_STREAM_BYTES,
        "maximum_git_process_stdin_bytes_per_command": MAX_GIT_STDIN_BYTES,
        "git_stdin_writer_is_concurrent_and_deadline_bounded": True,
        "git_version_preflight_uses_same_bounded_job_runner": True,
        "maximum_git_object_store_bytes": MAX_GIT_OBJECT_STORE_BYTES,
        "git_object_store_cap_is_post_fetch_acceptance_not_fetch_time_limit": True,
        "production_git_executable": {
            "path": str(PRODUCTION_GIT_EXECUTABLE),
            "byte_count": PRODUCTION_GIT_EXECUTABLE_BYTE_COUNT,
            "raw_sha256": PRODUCTION_GIT_EXECUTABLE_RAW_SHA256,
            "version_stdout": PRODUCTION_GIT_VERSION_STDOUT,
            "one_buffer_preflight_before_HTTP": True,
        },
        "production_git_HTTPS_helpers": [
            {
                "path": str(path),
                "byte_count": PRODUCTION_GIT_HELPER_BYTE_COUNT,
                "raw_sha256": PRODUCTION_GIT_HELPER_RAW_SHA256,
                "one_buffer_preflight_before_HTTP": True,
            }
            for path in PRODUCTION_GIT_HELPER_PATHS
        ],
        "capture_sequence": list(_CAPTURE_SEQUENCE),
        "identity_free": True,
        "proxy_netrc_auth_cookie_forbidden": True,
        "caller_supplied_backend_forced_synthetic": True,
        "facts_only_no_verdict": True,
    }


def _direct_https_get(
    *,
    role: str,
    url: str,
    body_cap: int,
    clock: Callable[[], float],
) -> HttpObservation:
    if type(body_cap) is not int or body_cap <= 0:
        raise ObserverError("invalid_body_cap", "body cap must be a positive integer")
    current_url = _require_role_url(url, role)
    started = clock()
    redirects: list[RedirectHop] = []
    while True:
        _require_before_deadline(started, HTTP_OVERALL_TIMEOUT_SECONDS, clock, "HTTP overall timeout")
        split = urlsplit(current_url)
        path = split.path or "/"
        if split.query:
            path = f"{path}?{split.query}"
        connection = http.client.HTTPSConnection(
            host=ROLE_HOSTS[role],
            port=443,
            timeout=CONNECT_TIMEOUT_SECONDS,
            context=_system_tls_context(),
        )
        try:
            connection.connect()
            if connection.sock is None:
                raise ObserverError("http_socket_missing", "HTTPS connection has no socket")
            connection.sock.settimeout(READ_TIMEOUT_SECONDS)
            connection.putrequest(HTTP_METHOD, path, skip_host=True, skip_accept_encoding=True)
            connection.putheader("Host", ROLE_HOSTS[role])
            for name, value in _REQUEST_HEADER_ITEMS:
                if name.casefold() in _FORBIDDEN_REQUEST_HEADER_NAMES:
                    raise AssertionError("fixed request headers contain a forbidden identity header")
                connection.putheader(name, value)
            connection.endheaders()
            response = connection.getresponse()
            headers = tuple((str(name), str(value)) for name, value in response.getheaders())
            header_count, header_wire_bytes, header_ledger_bytes, header_sha = _header_facts(headers)
            _response_header_safety(headers, role=role, final_body_length=None)
            _require_before_deadline(started, HTTP_OVERALL_TIMEOUT_SECONDS, clock, "HTTP overall timeout")
            if response.status in REDIRECT_STATUSES:
                locations = [value for name, value in headers if name.casefold() == "location"]
                if len(locations) != 1:
                    raise ObserverError("redirect_location_invalid", "redirect must contain exactly one Location")
                if len(redirects) >= MAX_REDIRECTS_BY_ROLE[role]:
                    raise ObserverError("redirect_limit_exceeded", "manual redirect limit exceeded")
                redirected = _require_role_url(locations[0], role)
                redirects.append(
                    RedirectHop(
                        requested_url=current_url,
                        status=response.status,
                        location=redirected,
                        response_header_count=header_count,
                        response_header_wire_bytes=header_wire_bytes,
                        response_header_ledger_bytes=header_ledger_bytes,
                        response_header_ledger_sha256=header_sha,
                        response_headers=headers,
                    )
                )
                current_url = redirected
                continue
            if response.status != SUCCESS_STATUS:
                raise ObserverError("http_status_not_200", f"HTTP status is {response.status}, expected 200")
            declared_length = _declared_content_length(headers)
            if declared_length is not None and declared_length > body_cap:
                raise ObserverError("body_cap_exceeded", "declared response body exceeds the fixed cap")
            body = _bounded_response_body(
                response,
                body_cap=body_cap,
                started=started,
                clock=clock,
            )
            _response_header_safety(headers, role=role, final_body_length=len(body))
            observation = HttpObservation(
                role=role,
                requested_url=url,
                final_url=current_url,
                status=response.status,
                response_headers=headers,
                body=body,
                redirects=tuple(redirects),
            )
            _validate_http_observation(observation, expected_role=role, body_cap=body_cap)
            return observation
        finally:
            connection.close()


def _system_tls_context() -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    if os.name == "nt":
        context.load_default_certs(ssl.Purpose.SERVER_AUTH)
        return context
    paths = ssl.get_default_verify_paths()
    cafile = paths.openssl_cafile if paths.openssl_cafile and os.path.isfile(paths.openssl_cafile) else None
    capath = paths.openssl_capath if paths.openssl_capath and os.path.isdir(paths.openssl_capath) else None
    if cafile is None and capath is None:
        raise ObserverError("system_trust_store_missing", "system TLS trust store is unavailable")
    context.load_verify_locations(cafile=cafile, capath=capath)
    return context


def _bounded_response_body(
    response: http.client.HTTPResponse,
    *,
    body_cap: int,
    started: float,
    clock: Callable[[], float],
) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        _require_before_deadline(started, HTTP_OVERALL_TIMEOUT_SECONDS, clock, "HTTP overall timeout")
        chunk = response.read(min(65_536, body_cap + 1 - total))
        _require_before_deadline(started, HTTP_OVERALL_TIMEOUT_SECONDS, clock, "HTTP overall timeout")
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        total += len(chunk)
        if total > body_cap:
            raise ObserverError("body_cap_exceeded", "response body exceeds the fixed cap")


def _validate_http_observation(
    observation: HttpObservation,
    *,
    expected_role: str,
    body_cap: int,
) -> None:
    if observation.role != expected_role:
        raise ObserverError("http_role_drift", "HTTP observation role drift")
    _require_role_url(observation.requested_url, expected_role)
    _require_role_url(observation.final_url, expected_role)
    if observation.status != SUCCESS_STATUS:
        raise ObserverError("http_status_not_200", f"HTTP status is {observation.status}, expected 200")
    if len(observation.redirects) > MAX_REDIRECTS_BY_ROLE[expected_role]:
        raise ObserverError("redirect_limit_exceeded", "manual redirect limit exceeded")
    current_url = observation.requested_url
    for hop in observation.redirects:
        if hop.requested_url != current_url:
            raise ObserverError("redirect_chain_invalid", "redirect chain is not contiguous")
        _require_role_url(hop.requested_url, expected_role)
        _require_role_url(hop.location, expected_role)
        if hop.status not in REDIRECT_STATUSES:
            raise ObserverError("redirect_status_invalid", "redirect status is outside the frozen set")
        count, wire_bytes, ledger_bytes, header_sha = _header_facts(hop.response_headers)
        if (
            count != hop.response_header_count
            or wire_bytes != hop.response_header_wire_bytes
            or ledger_bytes != hop.response_header_ledger_bytes
            or header_sha != hop.response_header_ledger_sha256
        ):
            raise ObserverError("redirect_header_facts_drift", "redirect header facts do not match raw pairs")
        _response_header_safety(hop.response_headers, role=expected_role, final_body_length=None)
        current_url = hop.location
    if current_url != observation.final_url:
        raise ObserverError("redirect_final_url_drift", "final URL does not close the redirect chain")
    _header_facts(observation.response_headers)
    if len(observation.body) > body_cap:
        raise ObserverError("body_cap_exceeded", "response body exceeds the fixed cap")
    _response_header_safety(
        observation.response_headers,
        role=expected_role,
        final_body_length=len(observation.body),
    )


def _header_values(headers: Sequence[tuple[str, str]], name: str) -> list[str]:
    folded = name.casefold()
    return [value for header_name, value in headers if header_name.casefold() == folded]


def _declared_content_length(headers: Sequence[tuple[str, str]]) -> int | None:
    values = _header_values(headers, "content-length")
    if not values:
        return None
    value = values[0]
    if len(values) != 1 or not value.isascii() or not value.isdecimal():
        raise ObserverError("content_length_invalid", "Content-Length is duplicated or invalid")
    return int(value)


def _response_header_safety(
    headers: Sequence[tuple[str, str]],
    *,
    role: str,
    final_body_length: int | None,
) -> None:
    content_length = _declared_content_length(headers)
    transfer_encodings = _header_values(headers, "transfer-encoding")
    content_encodings = _header_values(headers, "content-encoding")
    if len(transfer_encodings) > 1:
        raise ObserverError("transfer_encoding_duplicated", "Transfer-Encoding must not be duplicated")
    if transfer_encodings:
        transfer_encoding = transfer_encodings[0]
        if content_length is not None:
            raise ObserverError("http_framing_ambiguous", "Transfer-Encoding and Content-Length coexist")
        if transfer_encoding.casefold() != "chunked":
            raise ObserverError("transfer_encoding_invalid", "Transfer-Encoding must be exactly chunked")
    if len(content_encodings) > 1:
        raise ObserverError("content_encoding_duplicated", "Content-Encoding must not be duplicated")
    if content_encodings and content_encodings[0].casefold() != "identity":
        raise ObserverError("content_encoding_invalid", "Content-Encoding must be absent or identity")
    if final_body_length is not None and content_length is not None and content_length != final_body_length:
        raise ObserverError("content_length_mismatch", "Content-Length does not equal the captured body length")
    content_types = _header_values(headers, "content-type")
    if len(content_types) > 1:
        raise ObserverError("content_type_duplicated", "Content-Type must not be duplicated")
    if final_body_length is None:
        return
    raw_content_type = "" if not content_types else content_types[0].strip().casefold()
    media_type = raw_content_type.partition(";")[0].strip()
    if role == ROLE_API and media_type != "application/json":
        raise ObserverError("api_content_type_invalid", "API Content-Type must be JSON")
    if role == ROLE_HTML_GIT and media_type != "text/html":
        raise ObserverError("html_content_type_invalid", "HTML Content-Type must be text/html")


def _header_facts(headers: Sequence[tuple[str, str]]) -> tuple[int, int, int, str]:
    if len(headers) > MAX_HEADER_COUNT:
        raise ObserverError("header_count_exceeded", "response header count exceeds the fixed cap")
    normalized: list[tuple[str, str]] = []
    wire_byte_count = 0
    ledger_byte_count = 0
    for name, value in headers:
        if not isinstance(name, str) or not isinstance(value, str):
            raise ObserverError("header_type_invalid", "response header is not text")
        if "\r" in name or "\n" in name or "\r" in value or "\n" in value:
            raise ObserverError("header_folding_rejected", "response header contains CR/LF")
        serialized_value = "<redacted-set-cookie-value>" if name.casefold() == "set-cookie" else value
        try:
            wire_encoded = f"{name}: {value}\r\n".encode("latin-1")
            ledger_encoded = f"{name}: {serialized_value}\r\n".encode("latin-1")
        except UnicodeEncodeError as exc:
            raise ObserverError("header_encoding_invalid", "response header is not ISO-8859-1") from exc
        wire_byte_count += len(wire_encoded)
        ledger_byte_count += len(ledger_encoded)
        normalized.append((name.casefold(), serialized_value))
    if wire_byte_count > MAX_HEADER_BYTES:
        raise ObserverError("header_bytes_exceeded", "response headers exceed the fixed byte cap")
    return len(headers), wire_byte_count, ledger_byte_count, _sha256(_canonical_bytes(normalized))


def _serialized_header_pairs(headers: Sequence[tuple[str, str]]) -> list[list[str]]:
    return [
        [name, "<redacted-set-cookie-value>" if name.casefold() == "set-cookie" else value] for name, value in headers
    ]


def _redacted_set_cookie_facts(headers: Sequence[tuple[str, str]]) -> list[dict[str, object]]:
    facts: list[dict[str, object]] = []
    for index, (name, value) in enumerate(headers):
        if name.casefold() != "set-cookie":
            continue
        encoded = value.encode("latin-1")
        facts.append(
            {
                "pair_index": index,
                "value_byte_count": len(encoded),
                "value_sha256": _sha256(encoded),
            }
        )
    return facts


def _require_role_url(url: str, role: str) -> str:
    if role not in ROLE_HOSTS:
        raise ObserverError("unknown_role", "URL role is not frozen")
    if not isinstance(url, str) or not url:
        raise ObserverError("url_invalid", "URL must be non-empty text")
    try:
        encoded = url.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ObserverError("url_non_ascii", "URL must be ASCII") from exc
    if (
        len(encoded) > MAX_URL_BYTES
        or "\\" in url
        or any(ord(character) <= 0x20 or ord(character) == 0x7F for character in url)
        or not url.startswith(f"https://{ROLE_HOSTS[role]}/")
    ):
        raise ObserverError("url_lexical_invalid", "URL exceeds or violates the frozen lexical contract")
    try:
        split = urlsplit(url)
        port = split.port
    except ValueError as exc:
        raise ObserverError("url_invalid", "URL authority is invalid") from exc
    if (
        split.scheme != "https"
        or split.hostname != ROLE_HOSTS[role]
        or split.username is not None
        or split.password is not None
        or port not in {None, 443}
        or split.fragment
        or split.query
        or not split.path
        or split.path.startswith("//")
    ):
        raise ObserverError("role_url_invalid", f"URL does not match the exact {role} role host")
    if split.netloc != ROLE_HOSTS[role]:
        raise ObserverError("role_url_noncanonical", "URL must not spell an explicit or noncanonical port")
    return url


def _strict_json_object(payload: bytes, label: str) -> Mapping[str, object]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ObserverError("json_not_utf8", f"{label} is not strict UTF-8") from exc

    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ObserverError("json_duplicate_key", f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> object:
        raise ObserverError("json_nonfinite_number", f"{label} contains non-finite number {token}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ObserverError("json_invalid", f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ObserverError("json_root_invalid", f"{label} root must be an object")
    return value


def _api_core_projection(
    api_payload: Mapping[str, object],
    target: ObserverTarget,
) -> dict[str, object]:
    html_url = api_payload.get("html_url")
    git_pull_url = api_payload.get("git_pull_url")
    files = api_payload.get("files")
    history = api_payload.get("history")
    owner = api_payload.get("owner")
    if (
        not isinstance(html_url, str)
        or not isinstance(git_pull_url, str)
        or not isinstance(files, dict)
        or not isinstance(history, list)
        or not isinstance(owner, dict)
    ):
        raise ObserverError("api_shape_invalid", "API response lacks a required identity field")
    if api_payload.get("id") != target.gist_id:
        raise ObserverError("api_gist_id_mismatch", "API Gist id does not match the frozen target")
    gist_node_id = api_payload.get("node_id")
    if not isinstance(gist_node_id, str) or not _SAFE_NODE_ID_PATTERN.fullmatch(gist_node_id):
        raise ObserverError("api_gist_node_id_invalid", "API Gist node id is invalid")
    if api_payload.get("public") is not True or api_payload.get("description") != "":
        raise ObserverError("api_public_shape_invalid", "API Gist must be public with empty description")
    if api_payload.get("truncated") is not False:
        raise ObserverError("api_truncated", "API Gist response must not be truncated")
    if (
        owner.get("login") != target.expected_owner_login
        or owner.get("id") != target.expected_owner_id
        or owner.get("node_id") != target.expected_owner_node_id
        or owner.get("type") != "User"
        or owner.get("site_admin") is not False
    ):
        raise ObserverError("api_owner_mismatch", "API owner identity does not match the frozen target")
    if len(history) != 1 or not isinstance(history[0], dict) or history[0].get("version") != target.revision_oid:
        raise ObserverError("api_history_invalid", "API history must contain exactly the target revision")
    if set(files) != {target.required_filename}:
        raise ObserverError("api_file_set_invalid", "API files must contain exactly the required filename")
    file_payload = files.get(target.required_filename)
    if not isinstance(file_payload, dict):
        raise ObserverError("api_file_missing", "API response lacks the required filename")
    raw_url = file_payload.get("raw_url")
    if not isinstance(raw_url, str):
        raise ObserverError("api_raw_url_missing", "API file response lacks raw_url")
    if file_payload.get("filename") != target.required_filename:
        raise ObserverError("api_filename_mismatch", "API filename does not match the frozen target")
    if file_payload.get("size") != target.request_raw_byte_count:
        raise ObserverError("api_file_size_mismatch", "API file size does not match the frozen request bytes")
    if file_payload.get("truncated") is not False:
        raise ObserverError("api_file_truncated", "API required file is truncated")
    if file_payload.get("encoding") != "utf-8":
        raise ObserverError("api_file_encoding_invalid", "API required file encoding must be utf-8")
    inline_content = file_payload.get("content")
    if not isinstance(inline_content, str):
        raise ObserverError("api_inline_content_missing", "API required file lacks inline UTF-8 content")
    inline_bytes = inline_content.encode("utf-8")
    if len(inline_bytes) != target.request_raw_byte_count or _sha256(inline_bytes) != target.request_raw_sha256:
        raise ObserverError("api_inline_content_mismatch", "API inline content differs from request commitment")
    raw_url = _require_role_url(raw_url, ROLE_RAW)
    html_url = _require_role_url(html_url, ROLE_HTML_GIT)
    git_pull_url = _require_role_url(git_pull_url, ROLE_HTML_GIT)
    raw_path_parts = urlsplit(raw_url).path.split("/")
    if (
        len(raw_path_parts) != 6
        or raw_path_parts[:4] != ["", target.expected_owner_login, target.gist_id, "raw"]
        or raw_path_parts[5] != target.required_filename
    ):
        raise ObserverError("api_raw_url_path_mismatch", "API raw URL path does not bind the target Gist")
    raw_revision_token = raw_path_parts[4]
    try:
        _require_lower_hex(raw_revision_token, "raw URL revision token", minimum=40, maximum=40)
    except ValueError as exc:
        raise ObserverError("api_raw_url_path_mismatch", "API raw URL revision token is invalid") from exc
    if urlsplit(html_url).path != f"/{target.gist_id}":
        raise ObserverError("api_html_url_path_mismatch", "API HTML URL path does not bind the target Gist")
    if urlsplit(git_pull_url).path != f"/{target.gist_id}.git":
        raise ObserverError("api_git_url_path_mismatch", "API Git URL path does not bind the target Gist")
    return {
        "gist_id": target.gist_id,
        "gist_node_id": gist_node_id,
        "public": True,
        "description": "",
        "owner_login": target.expected_owner_login,
        "owner_id": target.expected_owner_id,
        "owner_node_id": target.expected_owner_node_id,
        "revision_oid": target.revision_oid,
        "required_filename": target.required_filename,
        "request_raw_byte_count": target.request_raw_byte_count,
        "request_raw_sha256": target.request_raw_sha256,
        "raw_url": raw_url,
        "raw_revision_token": raw_revision_token,
        "html_url": html_url,
        "git_pull_url": git_pull_url,
    }


def _http_metadata(
    observation: HttpObservation,
    *,
    body_file: str,
    body_cap: int,
) -> dict[str, object]:
    count, wire_bytes, ledger_bytes, header_sha = _header_facts(observation.response_headers)
    set_cookie_count = len(_header_values(observation.response_headers, "set-cookie"))
    content_lengths = _header_values(observation.response_headers, "content-length")
    transfer_encodings = _header_values(observation.response_headers, "transfer-encoding")
    content_encodings = _header_values(observation.response_headers, "content-encoding")
    content_types = _header_values(observation.response_headers, "content-type")
    return {
        "schema_version": "relationship-p4-external-anchor-observer-http.v1",
        "role": observation.role,
        "method": HTTP_METHOD,
        "requested_url": observation.requested_url,
        "final_url": observation.final_url,
        "status": observation.status,
        "request_headers": {key: value for key, value in _REQUEST_HEADER_ITEMS},
        "effective_request_headers": [
            ["Host", ROLE_HOSTS[observation.role]],
            *[[key, value] for key, value in _REQUEST_HEADER_ITEMS],
        ],
        "authorization_header_sent": False,
        "cookie_header_sent": False,
        "proxy_used": False,
        "netrc_used": False,
        "response_header_count": count,
        "response_header_wire_bytes": wire_bytes,
        "response_header_ledger_bytes": ledger_bytes,
        "response_header_ledger_sha256": header_sha,
        "response_header_pairs": _serialized_header_pairs(observation.response_headers),
        "set_cookie_present": set_cookie_count > 0,
        "set_cookie_count": set_cookie_count,
        "set_cookie_values_serialized": False,
        "set_cookie_redaction_facts": _redacted_set_cookie_facts(observation.response_headers),
        "response_framing": {
            "content_length_values": content_lengths,
            "transfer_encoding_values": transfer_encodings,
            "content_encoding_values": content_encodings,
            "content_type_values": content_types,
            "duplicate_content_length_rejected": True,
            "duplicate_transfer_encoding_rejected": True,
            "duplicate_content_encoding_rejected": True,
            "transfer_encoding_and_content_length_coexistence_rejected": True,
            "content_encoding_allowed": ["absent", "identity"],
            "declared_content_length_must_equal_captured_body": True,
        },
        "redirects": [
            {
                "requested_url": hop.requested_url,
                "status": hop.status,
                "location": hop.location,
                "response_header_count": hop.response_header_count,
                "response_header_wire_bytes": hop.response_header_wire_bytes,
                "response_header_ledger_bytes": hop.response_header_ledger_bytes,
                "response_header_ledger_sha256": hop.response_header_ledger_sha256,
                "effective_request_headers": [
                    ["Host", ROLE_HOSTS[observation.role]],
                    *[[key, value] for key, value in _REQUEST_HEADER_ITEMS],
                ],
                "response_header_pairs": _serialized_header_pairs(hop.response_headers),
                "set_cookie_present": bool(_header_values(hop.response_headers, "set-cookie")),
                "set_cookie_count": len(_header_values(hop.response_headers, "set-cookie")),
                "set_cookie_values_serialized": False,
                "set_cookie_redaction_facts": _redacted_set_cookie_facts(hop.response_headers),
            }
            for hop in observation.redirects
        ],
        "body": {
            "path": body_file,
            "byte_count": len(observation.body),
            "sha256": _sha256(observation.body),
            "body_cap": body_cap,
        },
        "role_redirect_max_hops": MAX_REDIRECTS_BY_ROLE[observation.role],
        "connect_timeout_seconds": int(CONNECT_TIMEOUT_SECONDS),
        "read_idle_timeout_seconds": int(READ_TIMEOUT_SECONDS),
        "request_total_timeout_seconds": int(HTTP_OVERALL_TIMEOUT_SECONDS),
        "retry_count": RETRY_COUNT,
        "facts_only_no_verdict": True,
    }


def _write_http_capture(
    output: pathlib.Path,
    *,
    observation: HttpObservation,
    metadata_file: str,
    body_file: str,
    role: str,
    body_cap: int,
) -> list[dict[str, object]]:
    body_entry = _write_bytes_capture(output, body_file, observation.body, role=f"{role}_body")
    metadata = _http_metadata(observation, body_file=body_file, body_cap=body_cap)
    metadata_entry = _write_json_capture(output, metadata_file, metadata, role=f"{role}_http")
    return [metadata_entry, body_entry]


def _write_git_capture(
    output: pathlib.Path,
    capture: GitObjectCapture,
    *,
    production: bool,
) -> list[dict[str, object]]:
    advertised_entry = _write_bytes_capture(
        output,
        GIT_ADVERTISED_REFS_FILE,
        capture.advertised_refs_body,
        role="git_advertised_refs_raw_stdout",
    )
    fetched_entry = _write_bytes_capture(
        output,
        GIT_FETCHED_REFS_FILE,
        capture.fetched_refs_body,
        role="git_fetched_refs_raw_stdout",
    )
    inventory_entry = _write_bytes_capture(
        output,
        GIT_OBJECT_INVENTORY_FILE,
        capture.object_inventory_body,
        role="git_object_inventory_raw_stdout",
    )
    commit_entry = _write_bytes_capture(output, GIT_COMMIT_FILE, capture.commit_body, role="git_commit_body")
    tree_entry = _write_bytes_capture(output, GIT_TREE_FILE, capture.tree_body, role="git_tree_body")
    blob_entry = _write_bytes_capture(output, GIT_BLOB_FILE, capture.blob_body, role="git_blob_body")
    metadata = {
        "schema_version": "relationship-p4-external-anchor-observer-git.v1",
        "remote_url": capture.remote_url,
        "revision_oid": capture.revision_oid,
        "commit_oid": capture.commit_oid,
        "tree_oid": capture.tree_oid,
        "blob_oid": capture.blob_oid,
        "tree_entry_mode": capture.tree_entry_mode,
        "tree_entry_name": capture.tree_entry_name,
        "advertised_refs": [list(item) for item in capture.advertised_refs],
        "fetched_refs": [list(item) for item in capture.fetched_refs],
        "object_inventory": [list(item) for item in capture.object_inventory],
        "advertised_refs_raw_stdout": _file_ref(advertised_entry),
        "fetched_refs_raw_stdout": _file_ref(fetched_entry),
        "object_inventory_raw_stdout": _file_ref(inventory_entry),
        "object_store_byte_count": capture.object_store_byte_count,
        "commit_body": _file_ref(commit_entry),
        "tree_body": _file_ref(tree_entry),
        "blob_body": _file_ref(blob_entry),
        "fsck_stdout_sha256": capture.fsck_stdout_sha256,
        "fsck_stderr_sha256": capture.fsck_stderr_sha256,
        "production_git_toolchain": {
            "executable_path": str(PRODUCTION_GIT_EXECUTABLE),
            "executable_byte_count": capture.git_executable_byte_count,
            "executable_raw_sha256": capture.git_executable_raw_sha256,
            "version_stdout": capture.git_version_stdout,
            "helper_identities": [list(item) for item in capture.git_helper_identities],
            "preflight_completed_before_HTTP": production,
        },
        "command_argv_ledger": [list(argv) for argv in capture.command_argv_ledger],
        "environment_ledger": [list(item) for item in capture.environment_ledger],
        "isolation_artifacts": {
            "global_config": "create_only_empty_file_inside_fresh_temporary_root",
            "XDG_CONFIG_HOME": "fresh_empty_directory_inside_fresh_temporary_root",
            "hooks_path": "fresh_empty_directory_inside_fresh_temporary_root",
            "system_config_disabled": production,
            "HOME_and_USERPROFILE_absent": production,
            "applied_in_production_backend": production,
        },
        "fresh_bare_repository": production,
        "all_heads_fetch_refspec": "+refs/heads/*:refs/remotes/origin/*",
        "system_and_global_config_disabled": production,
        (
            "credentials_askpass_extra_headers_cookies_proxy_custom_CA_redirects_hooks_"
            "alternates_replace_and_shallow_disabled"
        ): production,
        "facts_only_no_verdict": True,
    }
    metadata_entry = _write_json_capture(output, GIT_CAPTURE_FILE, metadata, role="git_capture")
    return [
        metadata_entry,
        advertised_entry,
        fetched_entry,
        inventory_entry,
        commit_entry,
        tree_entry,
        blob_entry,
    ]


def _file_ref(entry: Mapping[str, object]) -> dict[str, object]:
    return {
        "path": entry["path"],
        "byte_count": entry["byte_count"],
        "sha256": entry["sha256"],
    }


def _capture_fresh_bare_git(
    *,
    git_executable: pathlib.Path,
    git_identity: GitExecutableIdentity,
    remote_url: str,
    revision_oid: str,
    required_filename: str,
    workspace: pathlib.Path,
    clock: Callable[[], float],
) -> GitObjectCapture:
    git_path = _require_absolute_git_executable(git_executable)
    if git_path != git_identity.path:
        raise ObserverError("git_executable_identity_drift", "Git executable differs from its preflight")
    _require_role_url(remote_url, ROLE_HTML_GIT)
    _require_lower_hex(revision_oid, "revision_oid", minimum=40, maximum=40)
    if not _SAFE_FILENAME_PATTERN.fullmatch(required_filename):
        raise ObserverError("git_filename_invalid", "required Git filename is invalid")
    started = clock()
    with tempfile.TemporaryDirectory(prefix=".observer-git-", dir=workspace) as temporary_text:
        temporary = pathlib.Path(temporary_text)
        bare = temporary / "repository.git"
        template = temporary / "empty-template"
        hooks = temporary / "disabled-hooks"
        temp_io = temporary / "io"
        xdg_config_home = temporary / "empty-xdg-config"
        global_config = temporary / "empty-global-config"
        for directory in (template, hooks, temp_io, xdg_config_home):
            directory.mkdir()
        _write_empty_create_only(global_config)
        environment = _git_environment(
            temp_io,
            global_config=global_config,
            xdg_config_home=xdg_config_home,
        )
        fixed = _git_fixed_arguments(hooks)
        command_ledger: list[tuple[str, ...]] = []
        _run_git_with_ledger(
            git_path,
            (
                *fixed,
                "init",
                "--bare",
                "--object-format=sha1",
                f"--template={template}",
                str(bare),
            ),
            environment=environment,
            started=started,
            clock=clock,
            command_ledger=command_ledger,
        )
        advertised_result = _run_git_with_ledger(
            git_path,
            (*fixed, "ls-remote", "--heads", remote_url),
            environment=environment,
            started=started,
            clock=clock,
            command_ledger=command_ledger,
        )
        advertised_refs = _parse_advertised_refs(advertised_result.stdout)
        _require_single_target_head(advertised_refs, revision_oid, label="advertised")
        _run_git_with_ledger(
            git_path,
            (
                *fixed,
                "-C",
                str(bare),
                "fetch",
                "--quiet",
                "--no-tags",
                "--force",
                "--prune",
                "--no-recurse-submodules",
                remote_url,
                "+refs/heads/*:refs/remotes/origin/*",
            ),
            environment=environment,
            started=started,
            clock=clock,
            command_ledger=command_ledger,
        )
        object_store_bytes = _validate_fresh_git_repository(bare)
        fsck = _run_git_with_ledger(
            git_path,
            (*fixed, "-C", str(bare), "fsck", "--full", "--strict", "--no-reflogs"),
            environment=environment,
            started=started,
            clock=clock,
            command_ledger=command_ledger,
        )
        refs_result = _run_git_with_ledger(
            git_path,
            (
                *fixed,
                "-C",
                str(bare),
                "for-each-ref",
                "--format=%(refname)%00%(objectname)",
                "refs/remotes/origin",
            ),
            environment=environment,
            started=started,
            clock=clock,
            command_ledger=command_ledger,
        )
        fetched_refs = _parse_fetched_refs(refs_result.stdout)
        _require_single_target_head(fetched_refs, revision_oid, label="fetched")
        commit_oid, commit_type, commit_body = _git_cat_file_raw(
            git_path,
            fixed=fixed,
            bare=bare,
            object_name=revision_oid,
            environment=environment,
            started=started,
            clock=clock,
            command_ledger=command_ledger,
        )
        if commit_oid != revision_oid or commit_type != "commit":
            raise ObserverError("git_revision_not_commit", "frozen revision is not the exact captured commit")
        tree_oid = _commit_tree_oid(commit_body)
        captured_tree_oid, tree_type, tree_body = _git_cat_file_raw(
            git_path,
            fixed=fixed,
            bare=bare,
            object_name=tree_oid,
            environment=environment,
            started=started,
            clock=clock,
            command_ledger=command_ledger,
        )
        if captured_tree_oid != tree_oid or tree_type != "tree":
            raise ObserverError("git_tree_invalid", "commit tree object is invalid")
        tree_entry_mode, blob_oid = _required_tree_entry(tree_body, required_filename)
        captured_blob_oid, blob_type, blob_body = _git_cat_file_raw(
            git_path,
            fixed=fixed,
            bare=bare,
            object_name=blob_oid,
            environment=environment,
            started=started,
            clock=clock,
            command_ledger=command_ledger,
        )
        if captured_blob_oid != blob_oid or blob_type != "blob":
            raise ObserverError("git_blob_invalid", "required tree entry is not a blob")
        inventory_result = _run_git_with_ledger(
            git_path,
            (
                *fixed,
                "-C",
                str(bare),
                "cat-file",
                "--batch-all-objects",
                "--batch-check=%(objectname)%00%(objecttype)%00%(objectsize)",
            ),
            environment=environment,
            started=started,
            clock=clock,
            command_ledger=command_ledger,
        )
        object_inventory = _parse_object_inventory(inventory_result.stdout)
        expected_inventory = tuple(
            sorted(
                (
                    (commit_oid, "commit", len(commit_body)),
                    (tree_oid, "tree", len(tree_body)),
                    (blob_oid, "blob", len(blob_body)),
                )
            )
        )
        if object_inventory != expected_inventory:
            raise ObserverError(
                "git_object_inventory_mismatch",
                "Git object inventory must contain exactly the captured commit, tree, and blob",
            )
        _require_before_deadline(started, GIT_TOTAL_TIMEOUT_SECONDS, clock, "Git total timeout")
        capture = GitObjectCapture(
            remote_url=remote_url,
            revision_oid=revision_oid,
            commit_oid=commit_oid,
            tree_oid=tree_oid,
            blob_oid=blob_oid,
            tree_entry_mode=tree_entry_mode,
            tree_entry_name=required_filename,
            advertised_refs=advertised_refs,
            fetched_refs=fetched_refs,
            object_inventory=object_inventory,
            object_store_byte_count=object_store_bytes,
            commit_body=commit_body,
            tree_body=tree_body,
            blob_body=blob_body,
            advertised_refs_body=advertised_result.stdout,
            fetched_refs_body=refs_result.stdout,
            object_inventory_body=inventory_result.stdout,
            fsck_stdout_sha256=_sha256(fsck.stdout),
            fsck_stderr_sha256=_sha256(fsck.stderr),
            git_executable_byte_count=git_identity.byte_count,
            git_executable_raw_sha256=git_identity.raw_sha256,
            git_version_stdout=git_identity.version_stdout,
            git_helper_identities=git_identity.helper_identities,
            command_argv_ledger=tuple(command_ledger),
            environment_ledger=tuple(sorted(environment.items())),
        )
        _validate_git_capture(capture)
        return capture


def _require_absolute_git_executable(path: pathlib.Path) -> pathlib.Path:
    raw = pathlib.Path(path)
    if not raw.is_absolute():
        raise ObserverError("git_executable_not_absolute", "Git executable path must be absolute")
    absolute = pathlib.Path(os.path.abspath(raw))
    _reject_reparse_components(absolute, "Git executable")
    if absolute.is_symlink() or not absolute.is_file():
        raise ObserverError("git_executable_invalid", "Git executable is not a regular file")
    return absolute


def _preflight_production_git_executable(
    path: pathlib.Path,
    *,
    workspace: pathlib.Path,
) -> GitExecutableIdentity:
    executable = _require_absolute_git_executable(path)
    if os.name != "nt" or os.path.normcase(str(executable)) != os.path.normcase(str(PRODUCTION_GIT_EXECUTABLE)):
        raise ObserverError("git_executable_path_drift", "Git executable path differs from the frozen path")
    executable_buffer = _read_frozen_regular_buffer(
        executable,
        expected_byte_count=PRODUCTION_GIT_EXECUTABLE_BYTE_COUNT,
        expected_sha256=PRODUCTION_GIT_EXECUTABLE_RAW_SHA256,
        label="production Git executable",
    )
    helper_identities: list[tuple[str, int, str]] = []
    for helper_path_value in PRODUCTION_GIT_HELPER_PATHS:
        helper_path = pathlib.Path(helper_path_value)
        helper_buffer = _read_frozen_regular_buffer(
            helper_path,
            expected_byte_count=PRODUCTION_GIT_HELPER_BYTE_COUNT,
            expected_sha256=PRODUCTION_GIT_HELPER_RAW_SHA256,
            label=helper_path.name,
        )
        helper_identities.append((str(helper_path), len(helper_buffer), _sha256(helper_buffer)))
    with tempfile.TemporaryDirectory(prefix=".observer-git-preflight-", dir=workspace) as temporary_text:
        temporary = pathlib.Path(temporary_text)
        temp_io = temporary / "io"
        global_config = temporary / "empty-global-config"
        xdg_config_home = temporary / "empty-xdg-config"
        temp_io.mkdir()
        xdg_config_home.mkdir()
        _write_empty_create_only(global_config)
        environment = _git_environment(
            temp_io,
            global_config=global_config,
            xdg_config_home=xdg_config_home,
        )
        version_started = time.monotonic()
        version = _run_git(
            executable,
            ("--version",),
            environment=environment,
            started=version_started,
            clock=time.monotonic,
        )
        if version.stderr:
            raise ObserverError("git_preflight_failed", "Git version preflight failed")
        try:
            version_stdout = version.stdout.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise ObserverError("git_version_encoding_invalid", "Git version output is not ASCII") from exc
        if version_stdout != PRODUCTION_GIT_VERSION_STDOUT:
            raise ObserverError("git_version_drift", "Git version differs from the frozen production version")
    return GitExecutableIdentity(
        path=executable,
        byte_count=len(executable_buffer),
        raw_sha256=_sha256(executable_buffer),
        version_stdout=version_stdout,
        helper_identities=tuple(helper_identities),
    )


def _read_frozen_regular_buffer(
    path: pathlib.Path,
    *,
    expected_byte_count: int,
    expected_sha256: str,
    label: str,
) -> bytes:
    _reject_reparse_components(path, label)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or _is_windows_reparse(path):
        raise ObserverError("git_tool_not_unique_regular_file", f"{label} is not a unique regular file")
    with path.open("rb", buffering=0) as stream:
        payload = stream.read(expected_byte_count + 1)
        during = os.fstat(stream.fileno())
    after = path.lstat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_during = (during.st_dev, during.st_ino, during.st_size, during.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_during or identity_before != identity_after:
        raise ObserverError("git_tool_changed_during_read", f"{label} changed during one-buffer read")
    if len(payload) != expected_byte_count or _sha256(payload) != expected_sha256:
        raise ObserverError("git_tool_identity_mismatch", f"{label} does not match the frozen identity")
    return payload


def _write_empty_create_only(path: pathlib.Path) -> None:
    with path.open("xb") as stream:
        stream.flush()
        os.fsync(stream.fileno())


def _git_environment(
    temp_io: pathlib.Path,
    *,
    global_config: pathlib.Path,
    xdg_config_home: pathlib.Path,
) -> dict[str, str]:
    environment = dict(_GIT_FIXED_ENVIRONMENT)
    if os.name == "nt":
        ambient_by_upper = {key.upper(): value for key, value in os.environ.items()}
        for key in _GIT_AMBIENT_ALLOWLIST_WINDOWS:
            value = ambient_by_upper.get(key)
            if value:
                environment[key] = value
        if "SYSTEMROOT" not in environment or "WINDIR" not in environment:
            raise ObserverError(
                "git_environment_invalid",
                "Windows Git requires matching SYSTEMROOT and WINDIR",
            )
        if os.path.normcase(environment["SYSTEMROOT"]) != os.path.normcase(environment["WINDIR"]):
            raise ObserverError(
                "git_environment_invalid",
                "Windows Git requires matching SYSTEMROOT and WINDIR",
            )
    temp_text = str(temp_io)
    environment.update(
        {
            "TMP": temp_text,
            "TEMP": temp_text,
            "TMPDIR": temp_text,
            "GIT_CONFIG_GLOBAL": str(global_config),
            "XDG_CONFIG_HOME": str(xdg_config_home),
            "GIT_EXEC_PATH": str(pathlib.Path(PRODUCTION_GIT_HELPER_PATHS[0]).parent),
        }
    )
    if any(key.upper() in {"HOME", "USERPROFILE"} for key in environment):
        raise AssertionError("isolated Git environment must not set HOME or USERPROFILE")
    return environment


def _git_fixed_arguments(hooks: pathlib.Path) -> tuple[str, ...]:
    return (
        "--no-optional-locks",
        "-c",
        "credential.helper=",
        "-c",
        "core.askPass=",
        "-c",
        "http.extraHeader=",
        "-c",
        "http.cookieFile=",
        "-c",
        "http.saveCookies=false",
        "-c",
        "http.proxy=",
        "-c",
        "http.followRedirects=false",
        "-c",
        "http.sslVerify=true",
        "-c",
        "http.version=HTTP/1.1",
        "-c",
        "protocol.allow=never",
        "-c",
        "protocol.https.allow=always",
        "-c",
        "protocol.file.allow=never",
        "-c",
        "protocol.ext.allow=never",
        "-c",
        "transfer.fsckObjects=true",
        "-c",
        "fetch.fsckObjects=true",
        "-c",
        "core.hooksPath=" + str(hooks),
        "-c",
        "core.useReplaceRefs=false",
        "-c",
        "fetch.writeCommitGraph=false",
    )


@dataclass
class _PipeReadState:
    chunks: list[bytes] = field(default_factory=list)
    byte_count: int = 0
    exceeded: bool = False
    failed: bool = False

    @property
    def payload(self) -> bytes:
        return b"".join(self.chunks)


@dataclass
class _PipeWriteState:
    completed: bool = False
    failed: bool = False


class _WindowsJob:
    """Kill-on-close Windows Job Object for one suspended Git process tree."""

    def __init__(self) -> None:
        import ctypes
        import ctypes.wintypes as wintypes

        class BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        class BasicAccountingInformation(ctypes.Structure):
            _fields_ = [
                ("TotalUserTime", ctypes.c_longlong),
                ("TotalKernelTime", ctypes.c_longlong),
                ("ThisPeriodTotalUserTime", ctypes.c_longlong),
                ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
                ("TotalPageFaultCount", wintypes.DWORD),
                ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD),
                ("TotalTerminatedProcesses", wintypes.DWORD),
            ]

        self._ctypes = ctypes
        self._wintypes = wintypes
        self._accounting_type = BasicAccountingInformation
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        self._kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        self._kernel32.SetInformationJobObject.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        )
        self._kernel32.SetInformationJobObject.restype = wintypes.BOOL
        self._kernel32.SetHandleInformation.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.DWORD,
        )
        self._kernel32.SetHandleInformation.restype = wintypes.BOOL
        self._kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        self._kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        self._kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
        self._kernel32.TerminateJobObject.restype = wintypes.BOOL
        self._kernel32.QueryInformationJobObject.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_void_p,
        )
        self._kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        self._kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        self._kernel32.CloseHandle.restype = wintypes.BOOL
        self._handle = self._kernel32.CreateJobObjectW(None, None)
        if not self._handle:
            raise ObserverError("git_job_create_failed", "Windows Git Job Object creation failed")
        if not self._kernel32.SetHandleInformation(self._handle, 1, 0):
            self.close()
            raise ObserverError("git_job_configuration_failed", "Windows Git Job Object inheritance failed")
        limits = ExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = 0x00002000
        if not self._kernel32.SetInformationJobObject(
            self._handle,
            9,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            self.close()
            raise ObserverError("git_job_configuration_failed", "Windows Git Job Object limit setup failed")

    def assign_and_resume(self, process: subprocess.Popen[bytes]) -> None:
        process_handle = self._wintypes.HANDLE(int(process._handle))  # type: ignore[attr-defined]
        if not self._kernel32.AssignProcessToJobObject(self._handle, process_handle):
            raise ObserverError("git_job_assignment_failed", "Git process could not enter its Job Object")
        self._resume_only_suspended_thread(process.pid)

    def _resume_only_suspended_thread(self, process_id: int) -> None:
        ctypes = self._ctypes
        wintypes = self._wintypes

        class ThreadEntry32(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ThreadID", wintypes.DWORD),
                ("th32OwnerProcessID", wintypes.DWORD),
                ("tpBasePri", wintypes.LONG),
                ("tpDeltaPri", wintypes.LONG),
                ("dwFlags", wintypes.DWORD),
            ]

        self._kernel32.CreateToolhelp32Snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
        self._kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        self._kernel32.Thread32First.argtypes = (wintypes.HANDLE, ctypes.POINTER(ThreadEntry32))
        self._kernel32.Thread32First.restype = wintypes.BOOL
        self._kernel32.Thread32Next.argtypes = (wintypes.HANDLE, ctypes.POINTER(ThreadEntry32))
        self._kernel32.Thread32Next.restype = wintypes.BOOL
        self._kernel32.OpenThread.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        self._kernel32.OpenThread.restype = wintypes.HANDLE
        self._kernel32.ResumeThread.argtypes = (wintypes.HANDLE,)
        self._kernel32.ResumeThread.restype = wintypes.DWORD
        snapshot = self._kernel32.CreateToolhelp32Snapshot(0x00000004, 0)
        if int(snapshot) == ctypes.c_void_p(-1).value:
            raise ObserverError("git_thread_resume_failed", "Git suspended thread snapshot failed")
        thread_ids: list[int] = []
        try:
            entry = ThreadEntry32()
            entry.dwSize = ctypes.sizeof(entry)
            present = self._kernel32.Thread32First(snapshot, ctypes.byref(entry))
            while present:
                if entry.th32OwnerProcessID == process_id:
                    thread_ids.append(int(entry.th32ThreadID))
                present = self._kernel32.Thread32Next(snapshot, ctypes.byref(entry))
        finally:
            self._kernel32.CloseHandle(snapshot)
        if len(thread_ids) != 1:
            raise ObserverError("git_thread_resume_failed", "Git suspended process thread set is not singular")
        thread_handle = self._kernel32.OpenThread(0x0002, False, thread_ids[0])
        if not thread_handle:
            raise ObserverError("git_thread_resume_failed", "Git suspended thread could not be opened")
        try:
            if self._kernel32.ResumeThread(thread_handle) == 0xFFFFFFFF:
                raise ObserverError("git_thread_resume_failed", "Git suspended thread could not resume")
        finally:
            self._kernel32.CloseHandle(thread_handle)

    def terminate(self) -> None:
        if self._handle and not self._kernel32.TerminateJobObject(self._handle, 0xE0000001):
            raise ObserverError("git_job_termination_failed", "Windows Git Job Object termination failed")

    def active_process_count(self) -> int:
        accounting = self._accounting_type()
        if not self._kernel32.QueryInformationJobObject(
            self._handle,
            1,
            self._ctypes.byref(accounting),
            self._ctypes.sizeof(accounting),
            None,
        ):
            raise ObserverError("git_job_query_failed", "Windows Git Job Object query failed")
        return int(accounting.ActiveProcesses)

    def wait_empty(self, deadline: float) -> bool:
        while time.monotonic() < deadline:
            if self.active_process_count() == 0:
                return True
            time.sleep(0.01)
        return self.active_process_count() == 0

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


class _SingleProcessGuard:
    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self._process = process

    def terminate(self) -> None:
        if self._process.poll() is None:
            self._process.kill()

    def active_process_count(self) -> int:
        return int(self._process.poll() is None)

    def wait_empty(self, deadline: float) -> bool:
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                return True
            time.sleep(0.01)
        return self._process.poll() is not None

    def close(self) -> None:
        return None


def _read_capped_pipe(
    stream: object,
    state: _PipeReadState,
    wake: threading.Event,
) -> None:
    try:
        while True:
            chunk = stream.read(65_536)  # type: ignore[attr-defined]
            if not chunk:
                return
            remaining = MAX_GIT_PROCESS_STREAM_BYTES - state.byte_count
            if len(chunk) > remaining:
                if remaining > 0:
                    state.chunks.append(chunk[:remaining])
                    state.byte_count += remaining
                state.exceeded = True
                wake.set()
                return
            state.chunks.append(chunk)
            state.byte_count += len(chunk)
    except (OSError, ValueError):
        state.failed = True
        wake.set()
    finally:
        try:
            stream.close()  # type: ignore[attr-defined]
        except (OSError, ValueError):
            state.failed = True
            wake.set()


def _write_bounded_stdin(
    stream: object,
    payload: bytes,
    state: _PipeWriteState,
    wake: threading.Event,
) -> None:
    try:
        stream.write(payload)  # type: ignore[attr-defined]
        stream.flush()  # type: ignore[attr-defined]
        state.completed = True
    except (BrokenPipeError, OSError, ValueError):
        state.failed = True
    finally:
        try:
            stream.close()  # type: ignore[attr-defined]
        except (OSError, ValueError):
            state.failed = True
        wake.set()


def _start_git_process(
    argv: tuple[str, ...],
    *,
    environment: Mapping[str, str],
    input_required: bool,
) -> tuple[subprocess.Popen[bytes], _WindowsJob | _SingleProcessGuard]:
    job: _WindowsJob | None = None
    creationflags = 0
    start_new_session = False
    if os.name == "nt":
        job = _WindowsJob()
        creationflags = 0x00000004 | 0x08000000
    else:
        start_new_session = True
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE if input_required else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            env=dict(environment),
            close_fds=True,
            bufsize=0,
            creationflags=creationflags,
            start_new_session=start_new_session,
        )
    except Exception as exc:  # subprocess creation boundary; always close the pre-created Job handle
        if job is not None:
            job.close()
        raise ObserverError("git_process_start_failed", "Git process creation failed") from exc
    guard: _WindowsJob | _SingleProcessGuard = job if job is not None else _SingleProcessGuard(process)
    if job is not None:
        try:
            job.assign_and_resume(process)
        except Exception as exc:  # assignment/resume boundary; never leak a suspended process or Job
            cleanup_failed = False
            try:
                try:
                    job.terminate()
                except ObserverError:
                    cleanup_failed = True
                if process.poll() is None:
                    process.kill()
                try:
                    process.wait(timeout=GIT_REAP_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    cleanup_failed = True
                try:
                    if not job.wait_empty(time.monotonic() + GIT_REAP_TIMEOUT_SECONDS):
                        cleanup_failed = True
                except ObserverError:
                    cleanup_failed = True
            finally:
                try:
                    for stream in (process.stdin, process.stdout, process.stderr):
                        if stream is not None:
                            try:
                                stream.close()
                            except (OSError, ValueError):
                                cleanup_failed = True
                finally:
                    job.close()
            if cleanup_failed:
                raise ObserverError(
                    "git_process_tree_reap_timeout",
                    "Git process tree could not be proven terminated after setup failure",
                ) from exc
            if isinstance(exc, ObserverError):
                raise
            raise ObserverError(
                "git_job_assignment_failed",
                "Git process assignment or suspended-thread resume failed",
            ) from exc
    return process, guard


def _run_git(
    executable: pathlib.Path,
    arguments: Sequence[str],
    *,
    environment: Mapping[str, str],
    started: float,
    clock: Callable[[], float],
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    remaining = GIT_TOTAL_TIMEOUT_SECONDS - (clock() - started)
    if remaining <= 0:
        raise ObserverError("git_total_timeout", "Git total timeout expired")
    if input_bytes is not None and len(input_bytes) > MAX_GIT_STDIN_BYTES:
        raise ObserverError("git_process_input_cap_exceeded", "Git process input exceeds 4 KiB")
    argv = (str(executable), *arguments)
    process, guard = _start_git_process(argv, environment=environment, input_required=input_bytes is not None)
    workers: list[threading.Thread] = []
    wake = threading.Event()
    stdout_state = _PipeReadState()
    stderr_state = _PipeReadState()
    writer_state = _PipeWriteState(completed=input_bytes is None)
    try:
        if process.stdout is None or process.stderr is None:
            raise ObserverError("git_process_stream_failure", "Git process pipes were not created")
        workers.extend(
            (
                threading.Thread(
                    target=_read_capped_pipe,
                    args=(process.stdout, stdout_state, wake),
                    name="volvence-git-stdout-reader",
                    daemon=False,
                ),
                threading.Thread(
                    target=_read_capped_pipe,
                    args=(process.stderr, stderr_state, wake),
                    name="volvence-git-stderr-reader",
                    daemon=False,
                ),
            )
        )
        if input_bytes is not None:
            if process.stdin is None:
                raise ObserverError("git_process_stream_failure", "Git stdin pipe was not created")
            workers.append(
                threading.Thread(
                    target=_write_bounded_stdin,
                    args=(process.stdin, input_bytes, writer_state, wake),
                    name="volvence-git-stdin-writer",
                    daemon=False,
                )
            )
        for worker in workers:
            worker.start()

        failure_code: str | None = None
        deadline = time.monotonic() + remaining
        while failure_code is None and process.poll() is None:
            if stdout_state.exceeded or stderr_state.exceeded:
                failure_code = "git_process_output_cap_exceeded"
                break
            if stdout_state.failed or stderr_state.failed or writer_state.failed:
                failure_code = "git_process_stream_failure"
                break
            wait_remaining = deadline - time.monotonic()
            if wait_remaining <= 0:
                failure_code = "git_total_timeout"
                break
            wake.wait(min(0.02, wait_remaining))
            wake.clear()
        if failure_code is None and process.returncode not in {None, 0}:
            failure_code = "git_command_failed"
        if failure_code is not None:
            guard.terminate()
        elif not guard.wait_empty(deadline):
            failure_code = "git_process_tree_reap_timeout"
            guard.terminate()
        reap_deadline = time.monotonic() + GIT_REAP_TIMEOUT_SECONDS
        if not guard.wait_empty(reap_deadline):
            failure_code = "git_process_tree_reap_timeout"
            guard.terminate()
        try:
            process.wait(timeout=max(0.01, reap_deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            guard.terminate()
            failure_code = "git_process_tree_reap_timeout"
        for worker in workers:
            worker.join(timeout=max(0.01, reap_deadline - time.monotonic()))
        if any(worker.is_alive() for worker in workers):
            guard.terminate()
            guard.wait_empty(time.monotonic() + GIT_REAP_TIMEOUT_SECONDS)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()
            for worker in workers:
                worker.join(timeout=GIT_REAP_TIMEOUT_SECONDS)
            failure_code = "git_process_stream_failure"
        if stdout_state.exceeded or stderr_state.exceeded:
            failure_code = "git_process_output_cap_exceeded"
        elif stdout_state.failed or stderr_state.failed or writer_state.failed or not writer_state.completed:
            failure_code = "git_process_stream_failure"
        if failure_code is not None:
            if failure_code == "git_command_failed":
                raise ObserverError("git_command_failed", "Git command returned a nonzero status")
            if failure_code == "git_process_output_cap_exceeded":
                raise ObserverError("git_process_output_cap_exceeded", "Git process output exceeds 4 MiB")
            if failure_code == "git_process_stream_failure":
                raise ObserverError("git_process_stream_failure", "Git process stream capture failed")
            if failure_code == "git_process_tree_reap_timeout":
                raise ObserverError(
                    "git_process_tree_reap_timeout",
                    "Git process tree did not terminate within its bound",
                )
            if failure_code == "git_total_timeout":
                raise ObserverError("git_total_timeout", "Git total timeout expired")
            raise AssertionError("Git process failure code is not frozen")
        return subprocess.CompletedProcess(
            argv,
            int(process.returncode),
            stdout_state.payload,
            stderr_state.payload,
        )
    finally:
        cleanup_failed = False
        cleanup_deadline = time.monotonic() + GIT_REAP_TIMEOUT_SECONDS
        try:
            try:
                guard.terminate()
            except ObserverError:
                cleanup_failed = True
            try:
                if not guard.wait_empty(cleanup_deadline):
                    cleanup_failed = True
            except ObserverError:
                cleanup_failed = True
            try:
                process.wait(timeout=max(0.01, cleanup_deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                cleanup_failed = True
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    try:
                        stream.close()
                    except OSError:
                        cleanup_failed = True
            for worker in workers:
                worker.join(timeout=max(0.01, cleanup_deadline - time.monotonic()))
            if any(worker.is_alive() for worker in workers):
                cleanup_failed = True
        finally:
            guard.close()
        if cleanup_failed:
            raise ObserverError(
                "git_process_tree_reap_timeout",
                "Git process tree or stream workers did not terminate within their cleanup bound",
            )


def _run_git_with_ledger(
    executable: pathlib.Path,
    arguments: Sequence[str],
    *,
    environment: Mapping[str, str],
    started: float,
    clock: Callable[[], float],
    command_ledger: list[tuple[str, ...]],
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    command_ledger.append((str(executable), *arguments))
    return _run_git(
        executable,
        arguments,
        environment=environment,
        started=started,
        clock=clock,
        input_bytes=input_bytes,
    )


def _repository_entry_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _read_unique_repository_file(
    path: pathlib.Path,
) -> tuple[bytes, tuple[int, int, int, int, int, int]]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ObserverError("git_repository_escape_surface", "Git repository entry lstat failed") from exc
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or path.is_symlink() or _is_windows_reparse(path):
        raise ObserverError(
            "git_repository_escape_surface",
            "Git repository contains a non-unique or non-plain regular file",
        )
    try:
        with path.open("rb", buffering=0) as stream:
            payload = stream.read(MAX_GIT_REPOSITORY_METADATA_BYTES + 1)
            during = os.fstat(stream.fileno())
        after = path.lstat()
    except OSError as exc:
        raise ObserverError("git_repository_escape_surface", "Git repository file read failed") from exc
    identity = _repository_entry_identity(before)
    if identity != _repository_entry_identity(during) or identity != _repository_entry_identity(after):
        raise ObserverError("git_repository_escape_surface", "Git repository file changed during capture")
    if len(payload) > MAX_GIT_REPOSITORY_METADATA_BYTES:
        raise ObserverError("git_repository_escape_surface", "Git repository file exceeds its hard cap")
    return payload, identity


def _require_plain_repository_directory(
    path: pathlib.Path,
) -> tuple[int, int, int, int, int, int]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ObserverError("git_repository_escape_surface", "Git repository directory lstat failed") from exc
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink() or _is_windows_reparse(path):
        raise ObserverError("git_repository_escape_surface", "Git repository traverses a link/reparse point")
    return _repository_entry_identity(metadata)


def _validate_fresh_git_repository(bare: pathlib.Path) -> int:
    forbidden = (
        bare / "shallow",
        bare / "objects" / "info" / "alternates",
        bare / "info" / "grafts",
        bare / "refs" / "replace",
    )
    if any(os.path.lexists(path) for path in forbidden):
        raise ObserverError("git_repository_escape_surface", "Git shallow/alternate/graft/replace state exists")
    object_root = bare / "objects"
    directory_identities: dict[pathlib.Path, tuple[int, int, int, int, int, int]] = {}
    file_identities: dict[pathlib.Path, tuple[int, int, int, int, int, int]] = {}
    config_payload: bytes | None = None
    object_total = 0
    metadata_total = 0

    def walk_error(exc: OSError) -> None:
        raise ObserverError("git_repository_escape_surface", "Git repository tree walk failed") from exc

    for directory, directory_names, filenames in os.walk(
        bare,
        topdown=True,
        followlinks=False,
        onerror=walk_error,
    ):
        directory_path = pathlib.Path(directory)
        directory_identities[directory_path] = _require_plain_repository_directory(directory_path)
        directory_names.sort()
        filenames.sort()
        for name in directory_names:
            candidate = directory_path / name
            directory_identities[candidate] = _require_plain_repository_directory(candidate)
        for name in filenames:
            candidate = directory_path / name
            payload, identity = _read_unique_repository_file(candidate)
            file_identities[candidate] = identity
            if object_root == candidate.parent or object_root in candidate.parents:
                object_total += len(payload)
                if object_total > MAX_GIT_OBJECT_STORE_BYTES:
                    raise ObserverError("git_object_store_cap_exceeded", "Git object store exceeds 4 MiB")
            else:
                metadata_total += len(payload)
                if metadata_total > MAX_GIT_REPOSITORY_METADATA_BYTES:
                    raise ObserverError("git_repository_escape_surface", "Git repository metadata exceeds 4 MiB")
            if candidate == bare / "config":
                config_payload = payload

    for path, identity in directory_identities.items():
        if _require_plain_repository_directory(path) != identity:
            raise ObserverError("git_repository_escape_surface", "Git repository directory changed during capture")
    for path, identity in file_identities.items():
        try:
            after = path.lstat()
        except OSError as exc:
            raise ObserverError(
                "git_repository_escape_surface", "Git repository file disappeared during capture"
            ) from exc
        if _repository_entry_identity(after) != identity:
            raise ObserverError("git_repository_escape_surface", "Git repository file changed after capture")
    if config_payload is None:
        raise ObserverError("git_repository_metadata_invalid", "Git repository config is missing")
    try:
        config_text = config_payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ObserverError("git_repository_metadata_invalid", "Git repository config is not strict UTF-8") from exc
    forbidden_config_terms = (
        "credential",
        "proxy",
        "cookie",
        "extraheader",
        "sslca",
        "url ",
        "insteadOf",
    )
    if any(term.casefold() in config_text.casefold() for term in forbidden_config_terms):
        raise ObserverError("git_local_config_forbidden", "fresh bare repository contains forbidden local config")
    return object_total


def _parse_fetched_refs(payload: bytes) -> tuple[tuple[str, str], ...]:
    refs: list[tuple[str, str]] = []
    for line in payload.splitlines():
        if not line:
            continue
        pieces = line.split(b"\0")
        if len(pieces) != 2:
            raise ObserverError("git_ref_output_invalid", "Git ref output is malformed")
        try:
            name = pieces[0].decode("utf-8")
            oid = pieces[1].decode("ascii")
        except UnicodeDecodeError as exc:
            raise ObserverError("git_ref_output_encoding", "Git ref output encoding is invalid") from exc
        if not name.startswith("refs/remotes/origin/"):
            raise ObserverError("git_ref_scope_invalid", "captured Git ref is outside origin heads")
        _require_lower_hex(oid, "fetched ref OID", minimum=40, maximum=40)
        refs.append((name, oid))
    if len(refs) != 1:
        raise ObserverError("git_ref_count_invalid", "captured Git head count must be exactly one")
    return tuple(sorted(refs))


def _parse_advertised_refs(payload: bytes) -> tuple[tuple[str, str], ...]:
    refs: list[tuple[str, str]] = []
    for line in payload.splitlines():
        if not line:
            continue
        pieces = line.split(b"\t")
        if len(pieces) != 2:
            raise ObserverError("git_advertisement_invalid", "Git advertised ref output is malformed")
        try:
            oid = pieces[0].decode("ascii")
            name = pieces[1].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ObserverError("git_advertisement_encoding", "Git advertised ref encoding is invalid") from exc
        if not name.startswith("refs/heads/") or name == "refs/heads/":
            raise ObserverError("git_advertised_ref_scope_invalid", "Git advertised ref is not a head")
        _require_lower_hex(oid, "advertised ref OID", minimum=40, maximum=40)
        refs.append((name, oid))
    if len(refs) != 1:
        raise ObserverError("git_advertised_head_count_invalid", "advertised Git head count must be exactly one")
    return tuple(refs)


def _parse_object_inventory(payload: bytes) -> tuple[tuple[str, str, int], ...]:
    objects: list[tuple[str, str, int]] = []
    for line in payload.splitlines():
        if not line:
            continue
        pieces = line.split(b"\0")
        if len(pieces) != 3:
            raise ObserverError("git_object_inventory_invalid", "Git object inventory row is malformed")
        try:
            oid = pieces[0].decode("ascii")
            object_type = pieces[1].decode("ascii")
            size_text = pieces[2].decode("ascii")
        except UnicodeDecodeError as exc:
            raise ObserverError(
                "git_object_inventory_encoding",
                "Git object inventory encoding is invalid",
            ) from exc
        _require_lower_hex(oid, "inventory object OID", minimum=40, maximum=40)
        if object_type not in {"blob", "commit", "tree"} or not size_text.isdecimal():
            raise ObserverError("git_object_inventory_invalid", "Git object inventory value is invalid")
        size = int(size_text)
        if size < 0 or size > MAX_GIT_OBJECT_STORE_BYTES:
            raise ObserverError("git_object_inventory_invalid", "Git object inventory size is invalid")
        objects.append((oid, object_type, size))
    if len(objects) != 3 or len({oid for oid, _kind, _size in objects}) != 3:
        raise ObserverError(
            "git_object_inventory_count_invalid",
            "Git object inventory must contain exactly three unique objects",
        )
    return tuple(sorted(objects))


def _require_single_target_head(
    refs: Sequence[tuple[str, str]],
    revision_oid: str,
    *,
    label: str,
) -> None:
    if len(refs) != 1 or refs[0][1] != revision_oid:
        if label == "advertised":
            raise ObserverError(
                "git_advertised_head_mismatch",
                "advertised Git head must be exactly the frozen revision",
            )
        if label == "fetched":
            raise ObserverError(
                "git_fetched_head_mismatch",
                "fetched Git head must be exactly the frozen revision",
            )
        raise AssertionError("single-target-head label is not frozen")


def _git_cat_file_raw(
    executable: pathlib.Path,
    *,
    fixed: Sequence[str],
    bare: pathlib.Path,
    object_name: str,
    environment: Mapping[str, str],
    started: float,
    clock: Callable[[], float],
    command_ledger: list[tuple[str, ...]],
) -> tuple[str, str, bytes]:
    result = _run_git_with_ledger(
        executable,
        (*fixed, "-C", str(bare), "cat-file", "--batch"),
        environment=environment,
        started=started,
        clock=clock,
        command_ledger=command_ledger,
        input_bytes=f"{object_name}\n".encode("ascii"),
    )
    header, separator, remainder = result.stdout.partition(b"\n")
    if not separator:
        raise ObserverError("git_cat_file_header_missing", "Git cat-file header is missing")
    pieces = header.split(b" ")
    if len(pieces) != 3:
        raise ObserverError("git_cat_file_header_invalid", "Git cat-file header is invalid")
    try:
        oid = pieces[0].decode("ascii")
        object_type = pieces[1].decode("ascii")
        size = int(pieces[2].decode("ascii"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ObserverError("git_cat_file_header_invalid", "Git cat-file header is invalid") from exc
    _require_lower_hex(oid, "cat-file OID", minimum=40, maximum=40)
    if size < 0 or size > MAX_GIT_OBJECT_STORE_BYTES:
        raise ObserverError("git_object_body_cap_exceeded", "Git object body exceeds 4 MiB")
    if len(remainder) != size + 1 or remainder[-1:] != b"\n":
        raise ObserverError("git_cat_file_framing_invalid", "Git cat-file body framing is invalid")
    body = remainder[:size]
    framed = f"{object_type} {size}\0".encode("ascii") + body
    recomputed_oid = hashlib.sha1(framed, usedforsecurity=False).hexdigest()
    if recomputed_oid != oid:
        raise ObserverError("git_object_oid_mismatch", "Git object OID does not match its raw body")
    return oid, object_type, body


def _commit_tree_oid(commit_body: bytes) -> str:
    headers, separator, _message = commit_body.partition(b"\n\n")
    if not separator:
        raise ObserverError("git_commit_headers_invalid", "Git commit lacks a header terminator")
    header_lines = headers.split(b"\n")
    tree_lines = [line for line in header_lines if line.startswith(b"tree ")]
    parent_lines = [line for line in header_lines if line.startswith(b"parent ")]
    if len(tree_lines) != 1 or tree_lines[0] != header_lines[0]:
        raise ObserverError("git_commit_tree_invalid", "Git commit must have one first-line tree header")
    if parent_lines:
        raise ObserverError("git_commit_has_parent", "Git first revision commit must have zero parents")
    try:
        tree_oid = tree_lines[0][5:].decode("ascii")
    except UnicodeDecodeError as exc:
        raise ObserverError("git_commit_tree_encoding", "Git tree OID is not ASCII") from exc
    _require_lower_hex(tree_oid, "tree OID", minimum=40, maximum=40)
    return tree_oid


def _required_tree_entry(tree_body: bytes, required_filename: str) -> tuple[str, str]:
    offset = 0
    matches: list[tuple[str, str]] = []
    while offset < len(tree_body):
        space = tree_body.find(b" ", offset)
        nul = tree_body.find(b"\0", space + 1)
        if space <= offset or nul < 0 or nul + 21 > len(tree_body):
            raise ObserverError("git_tree_framing_invalid", "Git tree body is malformed")
        try:
            mode = tree_body[offset:space].decode("ascii")
            name = tree_body[space + 1 : nul].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ObserverError("git_tree_encoding_invalid", "Git tree entry encoding is invalid") from exc
        oid = tree_body[nul + 1 : nul + 21].hex()
        if name == required_filename:
            matches.append((mode, oid))
        offset = nul + 21
    if offset != len(tree_body) or len(matches) != 1 or offset == 0:
        raise ObserverError("git_required_entry_invalid", "required Git tree entry is absent or ambiguous")
    # An exact one-entry tree has one match and no other framed entry. Reparse the
    # body boundary from the sole entry rather than trusting the filename match.
    first_nul = tree_body.find(b"\0")
    if first_nul < 0 or first_nul + 21 != len(tree_body) or matches[0][0] != "100644":
        raise ObserverError("git_tree_shape_invalid", "Git tree must be exactly one mode-100644 file")
    return matches[0]


def _validate_git_capture(capture: GitObjectCapture) -> None:
    _require_role_url(capture.remote_url, ROLE_HTML_GIT)
    for label, value in (
        ("revision_oid", capture.revision_oid),
        ("commit_oid", capture.commit_oid),
        ("tree_oid", capture.tree_oid),
        ("blob_oid", capture.blob_oid),
    ):
        _require_lower_hex(value, label, minimum=40, maximum=40)
    if capture.commit_oid != capture.revision_oid:
        raise ObserverError("git_revision_commit_mismatch", "captured commit does not equal frozen revision")
    _require_single_target_head(capture.advertised_refs, capture.revision_oid, label="advertised")
    _require_single_target_head(capture.fetched_refs, capture.revision_oid, label="fetched")
    if _parse_advertised_refs(capture.advertised_refs_body) != capture.advertised_refs:
        raise ObserverError("git_advertised_raw_drift", "advertised refs do not match raw stdout")
    if _parse_fetched_refs(capture.fetched_refs_body) != capture.fetched_refs:
        raise ObserverError("git_fetched_raw_drift", "fetched refs do not match raw stdout")
    if _parse_object_inventory(capture.object_inventory_body) != capture.object_inventory:
        raise ObserverError("git_object_inventory_mismatch", "object inventory differs from raw stdout")
    if not capture.tree_entry_name or "\0" in capture.tree_entry_name:
        raise ObserverError("git_tree_entry_name_invalid", "captured tree entry name is invalid")
    if not capture.tree_entry_mode.isdigit():
        raise ObserverError("git_tree_entry_mode_invalid", "captured tree entry mode is invalid")
    if _git_object_oid("commit", capture.commit_body) != capture.commit_oid:
        raise ObserverError("git_commit_oid_mismatch", "captured commit body does not match its OID")
    if _commit_tree_oid(capture.commit_body) != capture.tree_oid:
        raise ObserverError("git_commit_tree_mismatch", "captured commit does not bind the captured tree")
    if _git_object_oid("tree", capture.tree_body) != capture.tree_oid:
        raise ObserverError("git_tree_oid_mismatch", "captured tree body does not match its OID")
    mode, blob_oid = _required_tree_entry(capture.tree_body, capture.tree_entry_name)
    if mode != capture.tree_entry_mode or blob_oid != capture.blob_oid:
        raise ObserverError("git_tree_binding_mismatch", "captured tree metadata differs from the raw tree")
    if _git_object_oid("blob", capture.blob_body) != capture.blob_oid:
        raise ObserverError("git_blob_oid_mismatch", "captured blob body does not match its OID")
    expected_inventory = tuple(
        sorted(
            (
                (capture.commit_oid, "commit", len(capture.commit_body)),
                (capture.tree_oid, "tree", len(capture.tree_body)),
                (capture.blob_oid, "blob", len(capture.blob_body)),
            )
        )
    )
    if capture.object_inventory != expected_inventory:
        raise ObserverError(
            "git_object_inventory_mismatch",
            "object inventory is not the exact commit/tree/blob closure",
        )
    if (
        capture.object_store_byte_count < 0
        or capture.object_store_byte_count > MAX_GIT_OBJECT_STORE_BYTES
        or any(
            len(body) > MAX_GIT_OBJECT_STORE_BYTES
            for body in (capture.commit_body, capture.tree_body, capture.blob_body)
        )
    ):
        raise ObserverError("git_capture_cap_exceeded", "Git capture exceeds the fixed cap")
    _require_lower_hex(capture.fsck_stdout_sha256, "fsck stdout SHA-256", minimum=64, maximum=64)
    _require_lower_hex(capture.fsck_stderr_sha256, "fsck stderr SHA-256", minimum=64, maximum=64)
    if (
        capture.git_executable_byte_count != PRODUCTION_GIT_EXECUTABLE_BYTE_COUNT
        or capture.git_executable_raw_sha256 != PRODUCTION_GIT_EXECUTABLE_RAW_SHA256
        or capture.git_version_stdout != PRODUCTION_GIT_VERSION_STDOUT
    ):
        raise ObserverError("git_toolchain_identity_drift", "captured Git executable identity drift")
    expected_helpers = tuple(
        (str(path), PRODUCTION_GIT_HELPER_BYTE_COUNT, PRODUCTION_GIT_HELPER_RAW_SHA256)
        for path in PRODUCTION_GIT_HELPER_PATHS
    )
    if capture.git_helper_identities != expected_helpers:
        raise ObserverError("git_helper_identity_drift", "captured Git helper identities drift")


def _git_object_oid(object_type: str, body: bytes) -> str:
    framed = f"{object_type} {len(body)}\0".encode("ascii") + body
    return hashlib.sha1(framed, usedforsecurity=False).hexdigest()


def _prepare_create_only_root(path: os.PathLike[str] | str) -> pathlib.Path:
    output = _require_local_default_stream_path(path, "observer output root")
    _reject_reparse_components(output.parent, "observer output parent")
    if os.path.lexists(output):
        raise FileExistsError(f"observer output root is create-only and already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    root_stat = output.lstat()
    if not stat.S_ISDIR(root_stat.st_mode) or output.is_symlink() or _is_windows_reparse(output):
        raise ValueError("observer output root is not a plain local directory")
    return output


def _inventory_pre_map_capture_root(
    root: pathlib.Path,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    _require_local_default_stream_path(root, "observer capture root")
    _reject_reparse_components(root, "observer capture root")
    files: list[dict[str, object]] = []
    anomalies: list[dict[str, str]] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        name = child.name
        if name in {CAPTURE_MAP_FILE, TERMINAL_FILE}:
            anomalies.append({"path": name, "status": "pre_map_reserved_file_already_exists"})
            continue
        try:
            _require_local_default_stream_path(child, f"capture entry {name}")
            before = child.lstat()
        except (OSError, ValueError):
            anomalies.append({"path": name, "status": "entry_lstat_or_path_contract_failed"})
            continue
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or child.is_symlink() or _is_windows_reparse(child):
            anomalies.append({"path": name, "status": "entry_not_unique_plain_regular_file"})
            continue
        try:
            with child.open("rb", buffering=0) as stream:
                payload = stream.read(MAX_GIT_OBJECT_STORE_BYTES + 1)
                during = os.fstat(stream.fileno())
            after = child.lstat()
        except OSError:
            anomalies.append({"path": name, "status": "entry_same_buffer_read_failed"})
            continue
        before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        during_identity = (during.st_dev, during.st_ino, during.st_size, during.st_mtime_ns)
        after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if before_identity != during_identity or before_identity != after_identity:
            anomalies.append({"path": name, "status": "entry_changed_during_same_buffer_read"})
            continue
        if len(payload) > MAX_GIT_OBJECT_STORE_BYTES:
            anomalies.append({"path": name, "status": "entry_exceeds_capture_inventory_cap"})
            continue
        role = _PRE_MAP_CAPTURE_ROLES.get(name, "unexpected_capture_file")
        files.append(
            {
                "role": role,
                "path": name,
                "byte_count": len(payload),
                "sha256": _sha256(payload),
            }
        )
        if name not in _PRE_MAP_CAPTURE_ROLES:
            anomalies.append({"path": name, "status": "unexpected_readable_regular_file"})
    return files, anomalies


def _require_local_default_stream_path(
    path: os.PathLike[str] | str,
    label: str,
) -> pathlib.Path:
    raw_text = os.fspath(path)
    if not isinstance(raw_text, str) or not raw_text or "\0" in raw_text:
        raise ValueError(f"{label} must be a non-empty text path")
    raw = pathlib.Path(raw_text)
    if not raw.is_absolute():
        raise ValueError(f"{label} must be absolute")
    absolute = pathlib.Path(os.path.abspath(raw))
    if os.name == "nt":
        normalized = str(absolute).replace("/", "\\")
        if normalized.startswith(("\\\\", "\\?\\", "\\.\\")):
            raise ValueError(f"{label} must not use UNC or a Windows device namespace")
        drive, tail = os.path.splitdrive(normalized)
        if not drive or ":" in tail:
            raise ValueError(f"{label} must be a local default-stream path")
        for component in pathlib.PureWindowsPath(normalized).parts[1:]:
            if component.rstrip(" .") != component:
                raise ValueError(f"{label} contains a noncanonical Windows component")
    return absolute


def _write_json_capture(
    root: pathlib.Path,
    name: str,
    value: Mapping[str, object],
    *,
    role: str,
) -> dict[str, object]:
    return _write_bytes_capture(root, name, _canonical_bytes(value), role=role)


def _write_bytes_capture(
    root: pathlib.Path,
    name: str,
    payload: bytes,
    *,
    role: str,
) -> dict[str, object]:
    if pathlib.PurePath(name).name != name:
        raise ValueError("capture filename must be a single path component")
    path = root / name
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "role": role,
        "path": name,
        "byte_count": len(payload),
        "sha256": _sha256(payload),
    }


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_lower_hex(value: str, label: str, *, minimum: int, maximum: int) -> None:
    if (
        not isinstance(value, str)
        or not minimum <= len(value) <= maximum
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{label} must be {minimum}..{maximum} lowercase hexadecimal characters")


def _require_before_deadline(
    started: float,
    duration: float,
    clock: Callable[[], float],
    label: str,
) -> None:
    if clock() - started > duration:
        if label.startswith("Git"):
            raise ObserverError("git_total_timeout", label)
        raise ObserverError("http_overall_timeout", label)


def _reject_reparse_components(path: pathlib.Path, label: str) -> None:
    for candidate in (path, *path.parents):
        if not os.path.lexists(candidate):
            continue
        if candidate.is_symlink() or _is_windows_reparse(candidate):
            raise ValueError(f"{label} must not traverse a symlink or Windows reparse point: {candidate}")


def _is_windows_reparse(path: pathlib.Path) -> bool:
    if os.name != "nt" or not os.path.lexists(path):
        return False
    return bool(os.lstat(path).st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


__all__ = [
    "API_END_BODY_FILE",
    "API_END_HTTP_FILE",
    "API_START_BODY_FILE",
    "API_START_HTTP_FILE",
    "CAPTURE_MAP_FILE",
    "CLAIM_FILE",
    "CONNECT_TIMEOUT_SECONDS",
    "GITHUB_API_VERSION",
    "GIT_ADVERTISED_REFS_FILE",
    "GIT_BLOB_FILE",
    "GIT_CAPTURE_FILE",
    "GIT_COMMIT_FILE",
    "GIT_FETCHED_REFS_FILE",
    "GIT_OBJECT_INVENTORY_FILE",
    "GIT_TOTAL_TIMEOUT_SECONDS",
    "GIT_TREE_FILE",
    "GitExecutableIdentity",
    "HTML_BODY_FILE",
    "HTML_HTTP_FILE",
    "HTTP_ACCEPT",
    "HTTP_ACCEPT_ENCODING",
    "HTTP_CACHE_CONTROL",
    "HTTP_METHOD",
    "HTTP_OVERALL_TIMEOUT_SECONDS",
    "HTTP_PRAGMA",
    "HTTP_USER_AGENT",
    "HttpObservation",
    "GitObjectCapture",
    "MAX_API_BODY_BYTES",
    "MAX_GIT_OBJECT_STORE_BYTES",
    "MAX_HEADER_BYTES",
    "MAX_HEADER_COUNT",
    "MAX_HTML_BODY_BYTES",
    "MAX_REDIRECTS",
    "MAX_REDIRECTS_BY_ROLE",
    "MAX_URL_BYTES",
    "OBSERVER_FAILURE_CODES",
    "ObserverError",
    "ObserverResult",
    "ObserverTarget",
    "ProductionObserverBackend",
    "PRODUCTION_GIT_EXECUTABLE",
    "PRODUCTION_GIT_EXECUTABLE_BYTE_COUNT",
    "PRODUCTION_GIT_EXECUTABLE_RAW_SHA256",
    "PRODUCTION_GIT_HELPER_BYTE_COUNT",
    "PRODUCTION_GIT_HELPER_PATHS",
    "PRODUCTION_GIT_HELPER_RAW_SHA256",
    "PRODUCTION_GIT_VERSION_STDOUT",
    "RAW_BODY_FILE",
    "RAW_HTTP_FILE",
    "READ_TIMEOUT_SECONDS",
    "REDIRECT_STATUSES",
    "RETRY_COUNT",
    "ROLE_API",
    "ROLE_HOSTS",
    "ROLE_HTML_GIT",
    "ROLE_RAW",
    "RedirectHop",
    "SUCCESS_STATUS",
    "SyntheticObserverBackend",
    "TERMINAL_FILE",
    "acquire_external_anchor_observation",
]
