"""Normative pure verifier for the P4.7 A0 public-Gist anchor.

The verifier is identity-free: no Volvence protocol/request id, GitHub owner,
Gist id, revision, or URL is compiled into this module.  Its caller supplies a
raw protocol pin and the exact local/capture buffers.  Every acceptance fact is
then recomputed from those buffers.  Caller-provided booleans, receipt verdicts,
and admission summaries are never authority inputs.

There is deliberately no filesystem, clock, process, environment, network,
Git CLI, model, or CUDA access here.  A separately pinned observer acquires a
create-only bundle; this module only replays bytes.  Synthetic bundles may test
integrity but can never be observed or admitted.  A successful A0 admission
still cannot authorize source materialization: separately anchored A1 remains
mandatory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import ntpath
import re
from typing import Any, Final
from urllib.parse import urlsplit


EXTERNAL_ANCHOR_RECEIPT_SCHEMA_VERSION: Final = "relationship-p4-long-context-github-public-gist-anchor-receipt.v2"
EXTERNAL_ANCHOR_ADMISSION_SCHEMA_VERSION: Final = "relationship-p4-long-context-github-public-gist-anchor-admission.v2"
EXTERNAL_ANCHOR_GOLDEN_VECTORS_SCHEMA_VERSION: Final = "relationship-p4-external-anchor-verifier-golden-vectors.v2"
EXTERNAL_ANCHOR_RECEIPT_BUNDLE_MANIFEST_SCHEMA_VERSION: Final = (
    "relationship-p4-external-anchor-receipt-bundle-manifest.v1"
)
EXTERNAL_ANCHOR_ADMISSION_BUNDLE_MANIFEST_SCHEMA_VERSION: Final = (
    "relationship-p4-external-anchor-admission-bundle-manifest.v1"
)

REAL_OBSERVER_BACKEND: Final = "production"
SYNTHETIC_OBSERVER_BACKEND: Final = "synthetic"

CLAIM_FILE: Final = "000_observer_claim.json"
API_START_HTTP_FILE: Final = "010_api_revision_start.http.json"
API_START_BODY_FILE: Final = "011_api_revision_start.body"
RAW_HTTP_FILE: Final = "020_returned_raw.http.json"
RAW_BODY_FILE: Final = "021_returned_raw.body"
HTML_HTTP_FILE: Final = "030_returned_html.http.json"
HTML_BODY_FILE: Final = "031_returned_html.body"
GIT_CAPTURE_FILE: Final = "040_git_capture.json"
GIT_ADVERTISED_REFS_FILE: Final = "040_git_advertised_refs.body"
GIT_FETCHED_REFS_FILE: Final = "040_git_fetched_refs.body"
GIT_OBJECT_INVENTORY_FILE: Final = "040_git_object_inventory.body"
GIT_COMMIT_FILE: Final = "041_git_commit.body"
GIT_TREE_FILE: Final = "042_git_tree.body"
GIT_BLOB_FILE: Final = "043_git_blob.body"
API_END_HTTP_FILE: Final = "050_api_revision_end.http.json"
API_END_BODY_FILE: Final = "051_api_revision_end.body"
CAPTURE_MAP_FILE: Final = "090_capture_map.json"
TERMINAL_FILE: Final = "099_terminal.json"

CAPTURE_FILE_NAMES: Final = (
    CLAIM_FILE,
    API_START_HTTP_FILE,
    API_START_BODY_FILE,
    RAW_HTTP_FILE,
    RAW_BODY_FILE,
    HTML_HTTP_FILE,
    HTML_BODY_FILE,
    GIT_CAPTURE_FILE,
    GIT_ADVERTISED_REFS_FILE,
    GIT_FETCHED_REFS_FILE,
    GIT_OBJECT_INVENTORY_FILE,
    GIT_COMMIT_FILE,
    GIT_TREE_FILE,
    GIT_BLOB_FILE,
    API_END_HTTP_FILE,
    API_END_BODY_FILE,
    CAPTURE_MAP_FILE,
    TERMINAL_FILE,
)

_MAPPED_CAPTURE_FILES = tuple(sorted(CAPTURE_FILE_NAMES[:-2]))
_CAPTURE_SEQUENCE = (
    "api_exact_revision_start",
    "returned_raw",
    "returned_html",
    "fresh_isolated_bare_git",
    "api_exact_revision_end",
)
_FILE_ROLES = {
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
_RECEIPT_BUNDLE_CAPTURE_ROLES = {
    **_FILE_ROLES,
    CAPTURE_MAP_FILE: "capture_map",
    TERMINAL_FILE: "terminal",
}
_RECEIPT_BUNDLE_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "artifact_id_contract",
        "observation_role",
        "receipt_id",
        "file_count",
        "files",
        "authority_firewall",
        "artifact_id",
    }
)
_RECEIPT_BUNDLE_FILE_REF_KEYS = frozenset({"role", "name", "byte_count", "sha256", "git_blob_oid_sha1"})
_ADMISSION_BUNDLE_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "artifact_id_contract",
        "admission_id",
        "r0_binding",
        "r1_binding",
        "authority_verdict",
        "file_count",
        "files",
        "artifact_id",
    }
)
_ROLE_HOSTS = {
    "api": "api.github.com",
    "raw": "gist.githubusercontent.com",
    "html_git": "gist.github.com",
}
_REQUEST_HEADERS = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "volvence-a0-gist-observer/1",
    "Accept-Encoding": "identity",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "X-GitHub-Api-Version": "2026-03-10",
}
_GIT_REQUIRED_ENVIRONMENT = {
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "GCM_INTERACTIVE": "Never",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_PROTOCOL_FROM_USER": "0",
    "GIT_HTTP_USER_AGENT": "volvence-a0-gist-observer/1",
    "LANG": "C",
    "LC_ALL": "C",
}
_GIT_REQUIRED_CONFIG_ARGUMENTS = (
    "credential.helper=",
    "core.askPass=",
    "http.extraHeader=",
    "http.cookieFile=",
    "http.saveCookies=false",
    "http.proxy=",
    "http.followRedirects=false",
    "http.sslVerify=true",
    "http.version=HTTP/1.1",
    "protocol.allow=never",
    "protocol.https.allow=always",
    "protocol.file.allow=never",
    "protocol.ext.allow=never",
    "transfer.fsckObjects=true",
    "fetch.fsckObjects=true",
    "core.useReplaceRefs=false",
    "fetch.writeCommitGraph=false",
)
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_LOWER_HEX_RE = re.compile(r"^[0-9a-f]+$")
_GITHUB_OWNER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")
_GIT_OBJECT_TYPES = frozenset({"blob", "tree", "commit"})

_REQUIRED_NORMATIVE_ROLES = (
    "receipt_schema",
    "admission_schema",
    "pure_verifier_and_admission_judge",
    "fixed_production_observer",
    "observer_CLI",
    "golden_vectors",
)
_NORMATIVE_SAME_BUFFER_OPERATIONS = {
    "receipt_schema": "hash_then_strict_JSON_parse",
    "admission_schema": "hash_then_strict_JSON_parse",
    "pure_verifier_and_admission_judge": ("hash_then_compile_and_execute_the_same_immutable_buffer"),
    "fixed_production_observer": "hash_then_compile_and_execute_the_same_immutable_buffer",
    "observer_CLI": "hash_then_compile_and_execute_the_same_immutable_buffer",
    "golden_vectors": "hash_then_strict_JSON_parse_and_execute_all_vectors",
}
_RECEIPT_BUNDLE_NORMATIVE_NAMES = {
    "receipt_schema": "110_receipt_schema.json",
    "admission_schema": "111_admission_schema.json",
    "pure_verifier_and_admission_judge": "112_verifier.py",
    "fixed_production_observer": "113_observer.py",
    "observer_CLI": "114_observer_cli.py",
    "golden_vectors": "115_golden_vectors.json",
}
_NORMATIVE_DESCRIPTOR_KEYS = frozenset(
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
)
_AUTHORITY_FIREWALL_KEYS = frozenset(
    {
        "external_request_dispatched",
        "publication_object_exists_observed",
        "publisher_action_or_identity_proven",
        "external_publication_observed",
        "external_publication_anchor_present",
        "external_anchor_admitted",
        "A1_contract_and_materializer_implementation_authorized",
        "structural_inventory_materialization_authorized",
        "source_execution_authorized",
        "tuple_feasibility_authorized",
        "power_search_authorized",
        "model_output_authorized",
        "CUDA_planner_authorized",
        "development_authorized",
        "qualification_authorized",
        "formal_authorized",
        "appendable_formal_supported",
        "readable_formal_supported",
        "learnable_formal_supported",
        "steerable_formal_supported",
        "integrated_four_axis_supported",
    }
)
_CLAIM_KEYS = frozenset(
    {
        "schema_version",
        "backend_kind",
        "target",
        "fixed_acquisition_contract",
        "authority_firewall",
        "A1_required_before_materialization",
        "process_id",
        "process_instance_nonce",
        "claim_boundary",
        "claim_id",
    }
)
_CLAIM_TARGET_KEYS = frozenset(
    {
        "observation_stage",
        "predecessor_receipt_id",
        "predecessor_receipt_bundle_manifest_raw_sha256",
        "protocol_id",
        "protocol_raw_sha256",
        "protocol_raw_byte_count",
        "request_id",
        "request_artifact_id",
        "request_raw_sha256",
        "request_raw_byte_count",
        "request_manifest_raw_sha256",
        "request_manifest_raw_byte_count",
        "gist_id",
        "revision_oid",
        "expected_owner_login",
        "expected_owner_id",
        "expected_owner_node_id",
        "required_filename",
        "local_protocol_request_manifest_buffers_recomputed_by_observer",
        "local_buffer_recomputation_owner",
    }
)
_CAPTURE_MAP_KEYS = frozenset(
    {
        "schema_version",
        "claim_id",
        "backend_kind",
        "observation_stage",
        "capture_sequence",
        "completed_stages",
        "files",
        "expected_pre_map_files",
        "actual_pre_map_files",
        "missing_pre_map_files",
        "unexpected_pre_map_files",
        "root_anomalies",
        "root_closure_status",
        "acquisition_complete",
        "failure_code",
        "retry_count",
        "root_anomaly_count",
        "authority_firewall",
        "A1_required_before_materialization",
        "capture_map_id",
    }
)
_TERMINAL_KEYS = frozenset(
    {
        "schema_version",
        "claim_id",
        "capture_map_id",
        "capture_map_raw_sha256",
        "backend_kind",
        "observation_stage",
        "status",
        "acquisition_complete",
        "failure",
        "retry_count",
        "root_closure_status",
        "root_anomaly_count",
        "authority_firewall",
        "A1_required_before_materialization",
        "claim_boundary",
        "terminal_id",
    }
)
_FILE_REF_KEYS = frozenset({"role", "path", "byte_count", "sha256"})
_BODY_REF_KEYS = frozenset({"path", "byte_count", "sha256"})
_HTTP_BODY_REF_KEYS = frozenset({"path", "byte_count", "sha256", "body_cap"})
_HTTP_KEYS = frozenset(
    {
        "schema_version",
        "role",
        "method",
        "requested_url",
        "final_url",
        "status",
        "request_headers",
        "effective_request_headers",
        "authorization_header_sent",
        "cookie_header_sent",
        "proxy_used",
        "netrc_used",
        "response_header_count",
        "response_header_wire_bytes",
        "response_header_ledger_bytes",
        "response_header_ledger_sha256",
        "response_header_pairs",
        "set_cookie_present",
        "set_cookie_count",
        "set_cookie_values_serialized",
        "set_cookie_redaction_facts",
        "response_framing",
        "redirects",
        "body",
        "role_redirect_max_hops",
        "connect_timeout_seconds",
        "read_idle_timeout_seconds",
        "request_total_timeout_seconds",
        "retry_count",
        "facts_only_no_verdict",
    }
)
_REDIRECT_KEYS = frozenset(
    {
        "requested_url",
        "status",
        "location",
        "response_header_count",
        "response_header_wire_bytes",
        "response_header_ledger_bytes",
        "response_header_ledger_sha256",
        "effective_request_headers",
        "response_header_pairs",
        "set_cookie_present",
        "set_cookie_count",
        "set_cookie_values_serialized",
        "set_cookie_redaction_facts",
    }
)
_GIT_KEYS = frozenset(
    {
        "schema_version",
        "remote_url",
        "revision_oid",
        "commit_oid",
        "tree_oid",
        "blob_oid",
        "tree_entry_mode",
        "tree_entry_name",
        "advertised_refs",
        "fetched_refs",
        "object_inventory",
        "advertised_refs_raw_stdout",
        "fetched_refs_raw_stdout",
        "object_inventory_raw_stdout",
        "object_store_byte_count",
        "commit_body",
        "tree_body",
        "blob_body",
        "fsck_stdout_sha256",
        "fsck_stderr_sha256",
        "production_git_toolchain",
        "command_argv_ledger",
        "environment_ledger",
        "isolation_artifacts",
        "fresh_bare_repository",
        "all_heads_fetch_refspec",
        "system_and_global_config_disabled",
        "credentials_askpass_extra_headers_cookies_proxy_custom_CA_redirects_hooks_alternates_replace_and_shallow_disabled",
        "facts_only_no_verdict",
    }
)


@dataclass(frozen=True, slots=True)
class ParsedZeroParentCommit:
    tree_oid_sha1: str
    headers: tuple[tuple[str, bytes], ...]
    message: bytes


@dataclass(frozen=True, slots=True)
class ParsedSingleEntryTree:
    mode: str
    filename: str
    blob_oid_sha1: str


@dataclass(frozen=True, slots=True)
class VerifiedGitClosure:
    commit_oid_sha1: str
    tree_oid_sha1: str
    blob_oid_sha1: str
    filename: str
    blob_byte_count: int


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _reject_json_nul(value: object, *, label: str) -> None:
    if type(value) is str:
        if "\0" in value:
            raise ValueError(f"{label} contains a forbidden NUL string value")
        return
    if type(value) is list:
        for item in value:
            _reject_json_nul(item, label=label)
        return
    if type(value) is dict:
        for key, item in value.items():
            if "\0" in key:
                raise ValueError(f"{label} contains a forbidden NUL object key")
            _reject_json_nul(item, label=label)


def strict_json_loads(payload: bytes, *, label: str = "JSON payload") -> object:
    if type(payload) is not bytes:
        raise TypeError(f"{label} must be bytes")
    if payload.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{label} must not carry a UTF-8 BOM")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not strict UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    _reject_json_nul(value, label=label)
    return value


def strict_json_object_from_bytes(payload: bytes, *, label: str = "JSON payload") -> dict[str, Any]:
    value = strict_json_loads(payload, label=label)
    if type(value) is not dict:
        raise ValueError(f"{label} root must be an object")
    return value


def canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value cannot be encoded as canonical JSON") from exc
    return (encoded + "\n").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git_object_frame_sha1(object_type: str, payload: bytes) -> bytes:
    if object_type not in _GIT_OBJECT_TYPES:
        raise ValueError(f"unsupported Git object type: {object_type!r}")
    if type(payload) is not bytes:
        raise TypeError("Git object payload must be bytes")
    return f"{object_type} {len(payload)}\0".encode("ascii") + payload


def git_object_oid_sha1(object_type: str, payload: bytes) -> str:
    return hashlib.sha1(git_object_frame_sha1(object_type, payload), usedforsecurity=False).hexdigest()


def _parse_commit_headers(header_payload: bytes) -> tuple[tuple[str, bytes], ...]:
    if not header_payload or b"\r" in header_payload or b"\0" in header_payload:
        raise ValueError("Git commit headers are malformed")
    logical: list[tuple[bytes, bytes]] = []
    for line in header_payload.split(b"\n"):
        if not line:
            raise ValueError("Git commit contains an empty header line")
        if line.startswith(b" "):
            if not logical:
                raise ValueError("Git commit starts with a continuation header")
            name, prior = logical[-1]
            logical[-1] = (name, prior + b"\n" + line)
            continue
        separator = line.find(b" ")
        if separator <= 0 or separator == len(line) - 1:
            raise ValueError("Git commit header must contain a name and value")
        name = line[:separator]
        if any(byte < 0x21 or byte > 0x7E for byte in name):
            raise ValueError("Git commit header name is not printable ASCII")
        logical.append((name, line[separator + 1 :]))
    return tuple((name.decode("ascii"), value) for name, value in logical)


def parse_zero_parent_commit(payload: bytes) -> ParsedZeroParentCommit:
    if type(payload) is not bytes:
        raise TypeError("Git commit payload must be bytes")
    separator = payload.find(b"\n\n")
    if separator <= 0:
        raise ValueError("Git commit must contain headers followed by a blank line")
    headers = _parse_commit_headers(payload[:separator])
    if headers[0][0] != "tree":
        raise ValueError("Git commit's first header must be tree")
    trees = [value for name, value in headers if name == "tree"]
    parents = [value for name, value in headers if name == "parent"]
    authors = [value for name, value in headers if name == "author"]
    committers = [value for name, value in headers if name == "committer"]
    if len(trees) != 1:
        raise ValueError("Git commit must contain exactly one tree header")
    if parents:
        raise ValueError("Git commit must have zero parent headers")
    if len(authors) != 1 or len(committers) != 1:
        raise ValueError("Git commit must contain exactly one author and committer")
    try:
        tree_oid = trees[0].decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("Git commit tree OID is not ASCII") from exc
    _require_lower_hex(tree_oid, 40, "Git commit tree OID")
    return ParsedZeroParentCommit(tree_oid, headers, payload[separator + 2 :])


def parse_single_entry_tree(payload: bytes) -> ParsedSingleEntryTree:
    if type(payload) is not bytes:
        raise TypeError("Git tree payload must be bytes")
    space = payload.find(b" ")
    nul = payload.find(b"\0", space + 1)
    if space <= 0 or nul <= space + 1 or len(payload) != nul + 21:
        raise ValueError("Git tree must contain exactly one complete SHA-1 entry")
    try:
        mode = payload[:space].decode("ascii")
        filename = payload[space + 1 : nul].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Git tree mode or filename encoding is invalid") from exc
    if mode != "100644":
        raise ValueError("Git tree's only entry must use mode 100644")
    _require_safe_filename(filename, "Git tree filename")
    return ParsedSingleEntryTree(mode, filename, payload[nul + 1 :].hex())


def verify_zero_parent_single_file_git_closure(
    *,
    commit_payload: bytes,
    tree_payload: bytes,
    blob_payload: bytes,
    required_filename: str,
    expected_revision_oid_sha1: str | None = None,
) -> VerifiedGitClosure:
    _require_safe_filename(required_filename, "required filename")
    commit = parse_zero_parent_commit(commit_payload)
    tree = parse_single_entry_tree(tree_payload)
    commit_oid = git_object_oid_sha1("commit", commit_payload)
    tree_oid = git_object_oid_sha1("tree", tree_payload)
    blob_oid = git_object_oid_sha1("blob", blob_payload)
    if expected_revision_oid_sha1 is not None:
        expected = _require_lower_hex(expected_revision_oid_sha1, 40, "expected revision OID")
        if commit_oid != expected:
            raise ValueError("Git commit OID does not equal the expected revision OID")
    if commit.tree_oid_sha1 != tree_oid:
        raise ValueError("Git commit does not identify the captured tree")
    if tree.filename != required_filename:
        raise ValueError("Git tree filename does not equal the required filename")
    if tree.blob_oid_sha1 != blob_oid:
        raise ValueError("Git tree does not identify the captured blob")
    return VerifiedGitClosure(commit_oid, tree_oid, blob_oid, tree.filename, len(blob_payload))


def exact_request_bytes_match(
    *,
    local_request_payload: bytes,
    git_blob_payload: bytes,
    observed_raw_payload: bytes,
    expected_raw_sha256: str,
    expected_byte_count: int,
) -> bool:
    for label, payload in (
        ("local request payload", local_request_payload),
        ("Git blob payload", git_blob_payload),
        ("observed raw payload", observed_raw_payload),
    ):
        if type(payload) is not bytes:
            raise TypeError(f"{label} must be bytes")
    expected_hash = _require_lower_hex(expected_raw_sha256, 64, "expected request SHA-256")
    expected_count = _require_nonnegative_int(expected_byte_count, "expected request byte count")
    return (
        len(local_request_payload) == expected_count
        and _sha256(local_request_payload) == expected_hash
        and git_blob_payload == local_request_payload
        and observed_raw_payload == local_request_payload
    )


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{label} keys must be strings")
    return value


def _require_exact_keys(value: Mapping[str, object], expected: frozenset[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys do not match schema; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _require_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a JSON boolean")
    return value


def _require_int(value: object, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        qualifier = "an integer" if minimum is None else f"an integer >= {minimum}"
        raise ValueError(f"{label} must be {qualifier}")
    return value


def _require_nonnegative_int(value: object, label: str) -> int:
    return _require_int(value, label, minimum=0)


def _require_text(value: object, label: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str or (not allow_empty and not value):
        qualifier = "text" if allow_empty else "non-empty text"
        raise ValueError(f"{label} must be {qualifier}")
    if "\0" in value or any(ord(character) < 0x20 and character not in "\t" for character in value):
        raise ValueError(f"{label} contains a forbidden control character")
    return value


def _require_lower_hex(value: object, length: int, label: str) -> str:
    if type(value) is not str or len(value) != length or _LOWER_HEX_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be exactly {length} lowercase hexadecimal characters")
    return value


def _require_variable_lower_hex(value: object, label: str, *, minimum: int, maximum: int) -> str:
    if type(value) is not str or not minimum <= len(value) <= maximum or _LOWER_HEX_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be {minimum}..{maximum} lowercase hexadecimal characters")
    return value


def _require_safe_filename(value: object, label: str) -> str:
    filename = _require_text(value, label)
    if filename in {".", ".."} or "/" in filename or "\\" in filename:
        raise ValueError(f"{label} must be a single safe path component")
    return filename


def _require_role_url(value: object, role: str, label: str) -> str:
    url = _require_text(value, label)
    expected_host = _ROLE_HOSTS[role]
    try:
        encoded = url.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must be ASCII") from exc
    if (
        len(encoded) > 2_048
        or not url.startswith(f"https://{expected_host}/")
        or "\\" in url
        or any(character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F for character in url)
    ):
        raise ValueError(f"{label} violates the frozen lexical URL contract")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid URL") from exc
    if (
        parsed.scheme != "https"
        or parsed.netloc != expected_host
        or parsed.hostname != expected_host
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or not parsed.path.startswith("/")
        or parsed.path.startswith("//")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{label} must use exact HTTPS role host {expected_host}")
    return url


def _require_canonical_windows_local_path(value: object, label: str) -> str:
    path = _require_text(value, label)
    if (
        "/" in path
        or path.startswith("\\\\")
        or len(path) < 4
        or not path[0].isalpha()
        or not path[0].isascii()
        or path[1:3] != ":\\"
        or ":" in path[2:]
        or path != ntpath.normpath(path)
    ):
        raise ValueError(f"{label} must be a canonical absolute local Windows path")
    components = path[3:].split("\\")
    reserved = {"CON", "PRN", "AUX", "NUL", *{f"COM{i}" for i in range(1, 10)}, *{f"LPT{i}" for i in range(1, 10)}}
    if any(
        not component
        or component in {".", ".."}
        or component.endswith((" ", "."))
        or component.split(".", 1)[0].upper() in reserved
        or any(ord(character) < 0x20 for character in component)
        for component in components
    ):
        raise ValueError(f"{label} contains a non-default-stream or ambiguous path component")
    return path


def _require_authority_firewall(value: object, label: str) -> dict[str, bool]:
    firewall = _require_mapping(value, label)
    _require_exact_keys(firewall, _AUTHORITY_FIREWALL_KEYS, label)
    normalized = {key: _require_bool(firewall[key], f"{label}.{key}") for key in sorted(firewall)}
    if any(normalized.values()):
        raise ValueError(f"{label} opens downstream authority")
    return normalized


def _without_id(value: Mapping[str, object], field: str, label: str) -> tuple[dict[str, object], str]:
    identifier = _require_lower_hex(value.get(field), 64, f"{label}.{field}")
    core = dict(value)
    del core[field]
    if _sha256(canonical_json_bytes(core)) != identifier:
        raise ValueError(f"{label}.{field} does not close over its canonical core")
    return core, identifier


def _canonical_object(payload: bytes, label: str) -> dict[str, Any]:
    value = strict_json_object_from_bytes(payload, label=label)
    if payload != canonical_json_bytes(value):
        raise ValueError(f"{label} must be canonical JSON UTF-8 plus one LF")
    return value


def _file_ref(name: str, payload: bytes, role: str | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "path": name,
        "byte_count": len(payload),
        "sha256": _sha256(payload),
    }
    if role is not None:
        result = {"role": role, **result}
    return result


def _validate_body_ref(value: object, *, expected_name: str, payload: bytes, label: str) -> dict[str, object]:
    reference = _require_mapping(value, label)
    _require_exact_keys(reference, _BODY_REF_KEYS, label)
    expected = _file_ref(expected_name, payload)
    if dict(reference) != expected:
        raise ValueError(f"{label} does not identify the supplied body buffer")
    return expected


def _validate_capture_file_mapping(capture_files: Mapping[str, bytes]) -> dict[str, bytes]:
    if not isinstance(capture_files, Mapping) or any(type(key) is not str for key in capture_files):
        raise ValueError("capture_files must be a string-keyed mapping")
    if set(capture_files) != set(CAPTURE_FILE_NAMES):
        raise ValueError(
            "capture_files do not match the exact bundle inventory; "
            f"missing={sorted(set(CAPTURE_FILE_NAMES) - set(capture_files))}, "
            f"extra={sorted(set(capture_files) - set(CAPTURE_FILE_NAMES))}"
        )
    result: dict[str, bytes] = {}
    for name in CAPTURE_FILE_NAMES:
        payload = capture_files[name]
        if type(payload) is not bytes:
            raise TypeError(f"capture buffer {name} must be bytes")
        result[name] = payload
    return result


def _validate_normative_binding(
    protocol: Mapping[str, object],
    normative_file_buffers: Mapping[str, bytes],
) -> dict[str, object]:
    registration = _require_mapping(protocol.get("verification_preregistration"), "verification_preregistration")
    files = registration.get("normative_files")
    if type(files) is not list:
        raise ValueError("verification_preregistration.normative_files must be an array")
    if registration.get("normative_file_count") != len(files):
        raise ValueError("verification_preregistration normative file count does not close")
    by_role: dict[str, dict[str, object]] = {}
    for index, item in enumerate(files):
        descriptor = _require_mapping(item, f"normative_files[{index}]")
        _require_exact_keys(descriptor, _NORMATIVE_DESCRIPTOR_KEYS, f"normative_files[{index}]")
        role = _require_text(descriptor["role"], f"normative_files[{index}].role")
        if role in by_role:
            raise ValueError(f"duplicate normative role: {role}")
        by_role[role] = {
            "repo_relative_posix_path": _require_text(
                descriptor["repo_relative_posix_path"], f"normative_files[{index}].path"
            ),
            "byte_count": _require_nonnegative_int(descriptor["byte_count"], f"normative_files[{index}].byte_count"),
            "raw_sha256": _require_lower_hex(descriptor["raw_sha256"], 64, f"normative_files[{index}].raw_sha256"),
            "git_blob_oid_sha1": _require_lower_hex(
                descriptor["git_blob_oid_sha1"], 40, f"normative_files[{index}].git_blob_oid_sha1"
            ),
            "media_type": _require_text(descriptor["media_type"], f"normative_files[{index}].media_type"),
            "expected_eol": _require_text(descriptor["expected_eol"], f"normative_files[{index}].expected_eol"),
            "same_buffer_operation": _require_text(
                descriptor["same_buffer_operation"], f"normative_files[{index}].same_buffer_operation"
            ),
        }
    if set(by_role) != set(_REQUIRED_NORMATIVE_ROLES):
        raise ValueError(
            "verification_preregistration must contain exactly the six normative roles; "
            f"missing={sorted(set(_REQUIRED_NORMATIVE_ROLES) - set(by_role))}, "
            f"extra={sorted(set(by_role) - set(_REQUIRED_NORMATIVE_ROLES))}"
        )
    if not isinstance(normative_file_buffers, Mapping) or any(type(role) is not str for role in normative_file_buffers):
        raise ValueError("normative_file_buffers must be a string-keyed mapping")
    if set(normative_file_buffers) != set(by_role):
        raise ValueError(
            "normative_file_buffers roles do not match protocol descriptors; "
            f"missing={sorted(set(by_role) - set(normative_file_buffers))}, "
            f"extra={sorted(set(normative_file_buffers) - set(by_role))}"
        )
    for role, descriptor in by_role.items():
        payload = normative_file_buffers[role]
        if type(payload) is not bytes:
            raise TypeError(f"normative buffer {role} must be bytes")
        if (
            len(payload) != descriptor["byte_count"]
            or _sha256(payload) != descriptor["raw_sha256"]
            or git_object_oid_sha1("blob", payload) != descriptor["git_blob_oid_sha1"]
        ):
            raise ValueError(f"normative buffer bytes drift: {role}")
        if descriptor["same_buffer_operation"] != _NORMATIVE_SAME_BUFFER_OPERATIONS[role]:
            raise ValueError(f"normative same-buffer operation drift: {role}")
        eol = descriptor["expected_eol"]
        if eol == "LF" and (not payload.endswith(b"\n") or b"\r" in payload):
            raise ValueError(f"normative buffer LF contract drift: {role}")
        if eol == "CRLF" and (not payload.endswith(b"\r\n") or payload.replace(b"\r\n", b"").find(b"\n") >= 0):
            raise ValueError(f"normative buffer CRLF contract drift: {role}")
        media_type = descriptor["media_type"]
        if media_type in {"application/json", "application/schema+json"}:
            strict_json_loads(payload, label=f"normative JSON {role}")
            if role == "golden_vectors":
                run_golden_vectors(payload)
        elif media_type == "text/x-python":
            try:
                source = payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"normative Python is not UTF-8: {role}") from exc
            compile(source, descriptor["repo_relative_posix_path"], "exec", dont_inherit=True)
        else:
            raise ValueError(f"unsupported normative media_type for {role}: {media_type}")
    return {role: by_role[role] for role in sorted(by_role)}


def _local_lineage(
    *,
    expected_protocol_raw_sha256: str,
    local_protocol_bytes: bytes,
    local_request_bytes: bytes,
    local_manifest_bytes: bytes,
    normative_file_buffers: Mapping[str, bytes],
) -> tuple[dict[str, object], dict[str, Any], dict[str, Any], dict[str, Any], bool]:
    for label, payload in (
        ("local protocol", local_protocol_bytes),
        ("local request", local_request_bytes),
        ("local manifest", local_manifest_bytes),
    ):
        if type(payload) is not bytes:
            raise TypeError(f"{label} must be bytes")
    expected_protocol_raw = _require_lower_hex(expected_protocol_raw_sha256, 64, "expected protocol raw SHA-256")
    if _sha256(local_protocol_bytes) != expected_protocol_raw:
        raise ValueError("local protocol buffer does not equal the externally pinned raw SHA-256")
    protocol = strict_json_object_from_bytes(local_protocol_bytes, label="local protocol")
    if protocol.get("protocol_id_contract") != "sha256_canonical_json_utf8_newline_v1":
        raise ValueError("local protocol id contract is unsupported")
    protocol_id = _sha256(canonical_json_bytes(protocol))
    normative = _validate_normative_binding(protocol, normative_file_buffers)

    request = _canonical_object(local_request_bytes, "local request")
    request_core, request_id = _without_id(request, "request_id", "local request")
    manifest = _canonical_object(local_manifest_bytes, "local manifest")
    manifest_core, artifact_id = _without_id(manifest, "artifact_id", "local manifest")

    identity = _require_mapping(request.get("identity"), "local request identity")
    files = manifest.get("files")
    lineage_valid = (
        identity.get("anchor_request_protocol_id") == protocol_id
        and identity.get("anchor_request_protocol_raw_sha256") == expected_protocol_raw
        and manifest.get("anchor_request_protocol_id") == protocol_id
        and manifest.get("request_id") == request_id
        and type(files) is list
        and len(files) == 1
    )
    if type(files) is list and len(files) == 1:
        request_ref = _require_mapping(files[0], "local manifest request file")
        expected_ref = {
            "path": "external_publication_anchor_request.json",
            "byte_count": len(local_request_bytes),
            "sha256": _sha256(local_request_bytes),
            "git_blob_oid_sha1": git_object_oid_sha1("blob", local_request_bytes),
        }
        lineage_valid = lineage_valid and dict(request_ref) == expected_ref
    projected_sections = (
        ("publication_target", "publication_target_contract"),
        ("verification_preregistration", "verification_preregistration"),
        ("observer_acquisition_contract", "observer_acquisition_contract"),
        ("git_object_closure_contract", "git_object_closure_contract"),
        ("receipt_contract", "receipt_contract"),
        ("admission_contract", "admission_contract"),
    )
    for request_key, protocol_key in projected_sections:
        lineage_valid = lineage_valid and request.get(request_key) == protocol.get(protocol_key)
    binding = {
        "protocol_id": protocol_id,
        "protocol_raw_sha256": expected_protocol_raw,
        "protocol_byte_count": len(local_protocol_bytes),
        "request_id": request_id,
        "request_artifact_id": artifact_id,
        "request_payload_raw_sha256": _sha256(local_request_bytes),
        "request_payload_byte_count": len(local_request_bytes),
        "request_manifest_raw_sha256": _sha256(local_manifest_bytes),
        "request_manifest_byte_count": len(local_manifest_bytes),
        "normative_files": normative,
    }
    del request_core, manifest_core
    return binding, protocol, request, manifest, lineage_valid


def _target_contract(request: Mapping[str, object]) -> dict[str, object]:
    target = _require_mapping(request.get("publication_target"), "request publication_target")
    owner = _require_mapping(target.get("expected_owner"), "publication_target.expected_owner")
    login = _require_text(owner.get("login"), "expected owner login")
    if _GITHUB_OWNER_RE.fullmatch(login) is None:
        raise ValueError("expected owner login is not canonical")
    role_hosts = _require_mapping(target.get("role_hosts"), "publication target role_hosts")
    expected_hosts = {
        "api": _ROLE_HOSTS["api"],
        "raw": _ROLE_HOSTS["raw"],
        "html_and_git": _ROLE_HOSTS["html_git"],
        "owner_profile_field_only_not_fetched_by_observer": "github.com",
    }
    if dict(role_hosts) != expected_hosts:
        raise ValueError("publication target role hosts drift from the verifier's exact roles")
    if target.get("required_visibility") != "public":
        raise ValueError("publication target visibility must be public")
    return {
        "expected_owner_login": login,
        "expected_owner_id": _require_int(owner.get("id"), "expected owner id", minimum=1),
        "expected_owner_node_id": _require_text(owner.get("node_id"), "expected owner node id"),
        "expected_owner_type": _require_text(owner.get("type"), "expected owner type"),
        "expected_owner_site_admin": _require_bool(owner.get("site_admin"), "expected owner site_admin"),
        "required_filename": _require_safe_filename(target.get("required_filename"), "required filename"),
        "required_description": _require_text(
            target.get("required_description"), "required description", allow_empty=True
        ),
        "required_file_count": _require_int(target.get("required_exact_file_count"), "required file count", minimum=1),
        "required_history_count": _require_int(
            target.get("required_exact_history_count"), "required history count", minimum=1
        ),
        "required_head_count": _require_int(
            target.get("required_exact_advertised_head_count"), "required head count", minimum=1
        ),
    }


def _validate_claim(
    payload: bytes,
    *,
    local_binding: Mapping[str, object],
    target: Mapping[str, object],
) -> tuple[dict[str, Any], dict[str, object], bool]:
    claim = _canonical_object(payload, "observer claim")
    _require_exact_keys(claim, _CLAIM_KEYS, "observer claim")
    core, claim_id = _without_id(claim, "claim_id", "observer claim")
    if claim.get("schema_version") != "relationship-p4-external-anchor-observer-claim.v1":
        raise ValueError("observer claim schema_version mismatch")
    backend = claim.get("backend_kind")
    if backend not in {REAL_OBSERVER_BACKEND, SYNTHETIC_OBSERVER_BACKEND}:
        raise ValueError("observer claim backend_kind is unsupported")
    claim_target = _require_mapping(claim.get("target"), "observer claim target")
    _require_exact_keys(claim_target, _CLAIM_TARGET_KEYS, "observer claim target")
    stage = claim_target.get("observation_stage")
    if stage not in {"R0", "R1"}:
        raise ValueError("observer claim stage must be R0 or R1")
    predecessor = claim_target.get("predecessor_receipt_id")
    if predecessor is not None:
        predecessor = _require_lower_hex(predecessor, 64, "claim predecessor receipt id")
    predecessor_bundle_manifest = claim_target.get("predecessor_receipt_bundle_manifest_raw_sha256")
    if predecessor_bundle_manifest is not None:
        predecessor_bundle_manifest = _require_lower_hex(
            predecessor_bundle_manifest,
            64,
            "claim predecessor receipt bundle manifest SHA-256",
        )
    if stage == "R0" and (predecessor is not None or predecessor_bundle_manifest is not None):
        raise ValueError("R0 claim must not bind predecessor receipt identities")
    if stage == "R1" and (predecessor is None or predecessor_bundle_manifest is None):
        raise ValueError("R1 claim must bind predecessor receipt and bundle-manifest identities")
    gist_id = _require_variable_lower_hex(claim_target.get("gist_id"), "claim gist id", minimum=20, maximum=64)
    revision = _require_lower_hex(claim_target.get("revision_oid"), 40, "claim revision OID")
    lineage_valid = (
        claim_target.get("protocol_id") == local_binding["protocol_id"]
        and claim_target.get("protocol_raw_sha256") == local_binding["protocol_raw_sha256"]
        and claim_target.get("protocol_raw_byte_count") == local_binding["protocol_byte_count"]
        and claim_target.get("request_id") == local_binding["request_id"]
        and claim_target.get("request_artifact_id") == local_binding["request_artifact_id"]
        and claim_target.get("request_raw_sha256") == local_binding["request_payload_raw_sha256"]
        and claim_target.get("request_raw_byte_count") == local_binding["request_payload_byte_count"]
        and claim_target.get("request_manifest_raw_sha256") == local_binding["request_manifest_raw_sha256"]
        and claim_target.get("request_manifest_raw_byte_count") == local_binding["request_manifest_byte_count"]
        and claim_target.get("expected_owner_login") == target["expected_owner_login"]
        and claim_target.get("expected_owner_id") == target["expected_owner_id"]
        and claim_target.get("expected_owner_node_id") == target["expected_owner_node_id"]
        and claim_target.get("required_filename") == target["required_filename"]
        and claim_target.get("local_protocol_request_manifest_buffers_recomputed_by_observer") is False
        and claim_target.get("local_buffer_recomputation_owner") == "separate_pinned_verifier"
    )
    _require_authority_firewall(claim.get("authority_firewall"), "observer claim authority_firewall")
    if claim.get("A1_required_before_materialization") is not True:
        raise ValueError("observer claim does not preserve the A1 firewall")
    fixed = _require_mapping(claim.get("fixed_acquisition_contract"), "fixed acquisition contract")
    if (
        fixed.get("method") != "GET"
        or fixed.get("github_api_version") != _REQUEST_HEADERS["X-GitHub-Api-Version"]
        or fixed.get("request_headers") != _REQUEST_HEADERS
        or fixed.get("role_hosts") != _ROLE_HOSTS
        or fixed.get("success_status") != 200
        or fixed.get("retry_count") != 0
        or fixed.get("identity_free") is not True
        or fixed.get("proxy_netrc_auth_cookie_forbidden") is not True
        or fixed.get("facts_only_no_verdict") is not True
    ):
        raise ValueError("observer claim fixed acquisition contract drift")
    projection = {
        "claim_id": claim_id,
        "backend_kind": backend,
        "observation_stage": stage,
        "predecessor_receipt_id": predecessor,
        "predecessor_receipt_bundle_manifest_raw_sha256": predecessor_bundle_manifest,
        "process_id": _require_int(claim.get("process_id"), "observer process_id", minimum=1),
        "process_instance_nonce": _require_lower_hex(
            claim.get("process_instance_nonce"), 64, "observer process_instance_nonce"
        ),
        "gist_id": gist_id,
        "revision_oid": revision,
        "expected_owner_login": target["expected_owner_login"],
        "expected_owner_id": target["expected_owner_id"],
        "expected_owner_node_id": target["expected_owner_node_id"],
        "required_filename": target["required_filename"],
    }
    del core
    return claim, projection, lineage_valid


def _validate_capture_map_and_terminal(
    capture_files: Mapping[str, bytes],
    *,
    claim_projection: Mapping[str, object],
) -> tuple[dict[str, object], bool]:
    capture_map = _canonical_object(capture_files[CAPTURE_MAP_FILE], "observer capture map")
    _require_exact_keys(capture_map, _CAPTURE_MAP_KEYS, "observer capture map")
    _map_core, capture_map_id = _without_id(capture_map, "capture_map_id", "observer capture map")
    refs = capture_map.get("files")
    if type(refs) is not list or len(refs) != len(_MAPPED_CAPTURE_FILES):
        raise ValueError("observer capture map does not contain the exact file-ref count")
    normalized_refs: list[dict[str, object]] = []
    for index, name in enumerate(_MAPPED_CAPTURE_FILES):
        value = _require_mapping(refs[index], f"capture map files[{index}]")
        _require_exact_keys(value, _FILE_REF_KEYS, f"capture map files[{index}]")
        expected = _file_ref(name, capture_files[name], _FILE_ROLES[name])
        if dict(value) != expected:
            raise ValueError(f"capture map file ref does not bind supplied buffer: {name}")
        normalized_refs.append(expected)
    _require_authority_firewall(capture_map.get("authority_firewall"), "capture map authority_firewall")
    map_complete = (
        capture_map.get("schema_version") == "relationship-p4-external-anchor-observer-capture-map.v1"
        and capture_map.get("claim_id") == claim_projection["claim_id"]
        and capture_map.get("backend_kind") == claim_projection["backend_kind"]
        and capture_map.get("observation_stage") == claim_projection["observation_stage"]
        and capture_map.get("capture_sequence") == list(_CAPTURE_SEQUENCE)
        and capture_map.get("completed_stages") == list(_CAPTURE_SEQUENCE)
        and capture_map.get("expected_pre_map_files") == list(_MAPPED_CAPTURE_FILES)
        and capture_map.get("actual_pre_map_files") == list(_MAPPED_CAPTURE_FILES)
        and capture_map.get("missing_pre_map_files") == []
        and capture_map.get("unexpected_pre_map_files") == []
        and capture_map.get("root_anomalies") == []
        and capture_map.get("root_closure_status") == "complete_exact_pre_map_root"
        and capture_map.get("acquisition_complete") is True
        and capture_map.get("failure_code") is None
        and capture_map.get("retry_count") == 0
        and capture_map.get("root_anomaly_count") == 0
        and capture_map.get("A1_required_before_materialization") is True
    )

    terminal = _canonical_object(capture_files[TERMINAL_FILE], "observer terminal")
    _require_exact_keys(terminal, _TERMINAL_KEYS, "observer terminal")
    _terminal_core, terminal_id = _without_id(terminal, "terminal_id", "observer terminal")
    _require_authority_firewall(terminal.get("authority_firewall"), "terminal authority_firewall")
    terminal_complete = (
        terminal.get("schema_version") == "relationship-p4-external-anchor-observer-terminal.v1"
        and terminal.get("claim_id") == claim_projection["claim_id"]
        and terminal.get("capture_map_id") == capture_map_id
        and terminal.get("capture_map_raw_sha256") == _sha256(capture_files[CAPTURE_MAP_FILE])
        and terminal.get("backend_kind") == claim_projection["backend_kind"]
        and terminal.get("observation_stage") == claim_projection["observation_stage"]
        and terminal.get("status") == "facts_only_observation_complete_non_authorizing"
        and terminal.get("acquisition_complete") is True
        and terminal.get("failure") is None
        and terminal.get("retry_count") == 0
        and terminal.get("root_closure_status") == "complete_exact_pre_map_root"
        and terminal.get("root_anomaly_count") == 0
        and terminal.get("A1_required_before_materialization") is True
    )
    binding = {
        "claim_id": claim_projection["claim_id"],
        "claim_raw_sha256": _sha256(capture_files[CLAIM_FILE]),
        "process_id": claim_projection["process_id"],
        "process_instance_nonce": claim_projection["process_instance_nonce"],
        "capture_map_id": capture_map_id,
        "capture_map_raw_sha256": _sha256(capture_files[CAPTURE_MAP_FILE]),
        "terminal_id": terminal_id,
        "terminal_raw_sha256": _sha256(capture_files[TERMINAL_FILE]),
        "ordered_file_refs": normalized_refs,
    }
    return binding, map_complete and terminal_complete


def _normalized_headers(value: object, label: str) -> tuple[list[list[str]], int, int, str]:
    if type(value) is not list:
        raise ValueError(f"{label} must be an array")
    normalized: list[list[str]] = []
    byte_count = 0
    for index, item in enumerate(value):
        if type(item) is not list or len(item) != 2:
            raise ValueError(f"{label}[{index}] must be a two-item array")
        name = _require_text(item[0], f"{label}[{index}].name")
        header_value = _require_text(item[1], f"{label}[{index}].value", allow_empty=True)
        if "\r" in name or "\n" in name or "\r" in header_value or "\n" in header_value:
            raise ValueError(f"{label}[{index}] contains header folding")
        try:
            byte_count += len(f"{name}: {header_value}\r\n".encode("latin-1"))
        except UnicodeEncodeError as exc:
            raise ValueError(f"{label}[{index}] is not ISO-8859-1") from exc
        normalized.append([name.casefold(), header_value])
    if len(value) > 100:
        raise ValueError(f"{label} exceeds the frozen header count budget")
    return normalized, len(value), byte_count, _sha256(canonical_json_bytes(normalized))


def _header_wire_facts(
    normalized: Sequence[Sequence[str]],
    redaction_facts_value: object,
    *,
    ledger_byte_count: int,
    label: str,
) -> tuple[int, bool]:
    if type(redaction_facts_value) is not list:
        raise ValueError(f"{label} must be an array")
    facts_by_index: dict[int, tuple[int, str]] = {}
    for fact_index, item in enumerate(redaction_facts_value):
        fact = _require_mapping(item, f"{label}[{fact_index}]")
        _require_exact_keys(
            fact,
            frozenset({"pair_index", "value_byte_count", "value_sha256"}),
            f"{label}[{fact_index}]",
        )
        pair_index = _require_nonnegative_int(fact["pair_index"], f"{label}[{fact_index}].pair_index")
        value_byte_count = _require_nonnegative_int(fact["value_byte_count"], f"{label}[{fact_index}].value_byte_count")
        value_sha256 = _require_lower_hex(fact["value_sha256"], 64, f"{label}[{fact_index}].value_sha256")
        if pair_index in facts_by_index:
            raise ValueError(f"{label} duplicates pair_index {pair_index}")
        facts_by_index[pair_index] = (value_byte_count, value_sha256)
    cookie_indexes = {index for index, (name, _value) in enumerate(normalized) if name == "set-cookie"}
    redaction_valid = set(facts_by_index) == cookie_indexes
    placeholder_size = len("<redacted-set-cookie-value>".encode("latin-1"))
    wire_byte_count = ledger_byte_count
    for index in cookie_indexes:
        _count, _hash = facts_by_index.get(index, (0, ""))
        wire_byte_count += _count - placeholder_size
    return wire_byte_count, redaction_valid and wire_byte_count <= 65_536


def _header_semantics(
    headers: Sequence[Sequence[str]],
    *,
    role: str,
    final_body_length: int | None,
) -> bool:
    by_name: dict[str, list[str]] = {}
    for name, value in headers:
        by_name.setdefault(name, []).append(value)
    content_lengths = by_name.get("content-length", [])
    transfer_encodings = by_name.get("transfer-encoding", [])
    content_encodings = by_name.get("content-encoding", [])
    content_types = by_name.get("content-type", [])
    if len(content_lengths) > 1 or len(transfer_encodings) > 1 or len(content_encodings) > 1:
        return False
    if content_lengths:
        value = content_lengths[0]
        if not value.isascii() or not value.isdecimal():
            return False
        if final_body_length is not None and int(value) != final_body_length:
            return False
    if transfer_encodings and (content_lengths or transfer_encodings[0].casefold() != "chunked"):
        return False
    if content_encodings and content_encodings[0].casefold() != "identity":
        return False
    if len(content_types) > 1:
        return False
    if final_body_length is not None:
        media_type = "" if not content_types else content_types[0].partition(";")[0].strip().casefold()
        if role == "api" and media_type != "application/json":
            return False
        if role == "html_git" and media_type != "text/html":
            return False
    return True


def _golden_redirect_chain(
    *,
    role: str,
    requested_url: object,
    final_url: object,
    locations: object,
) -> bool:
    if role not in _ROLE_HOSTS or type(locations) is not list:
        return False
    try:
        current = _require_role_url(requested_url, role, "golden redirect requested_url")
        final = _require_role_url(final_url, role, "golden redirect final_url")
    except (KeyError, ValueError):
        return False
    limit = 0 if role == "api" else 3
    if len(locations) > limit:
        return False
    seen = {current}
    for index, location in enumerate(locations):
        try:
            current = _require_role_url(location, role, f"golden redirect location {index}")
        except (KeyError, ValueError):
            return False
        if current in seen:
            return False
        seen.add(current)
    return current == final


def run_golden_vectors(golden_bytes: bytes) -> dict[str, object]:
    """Parse and execute every vector from the pinned immutable JSON buffer."""

    root = strict_json_object_from_bytes(golden_bytes, label="external anchor golden vectors")
    expected_root_keys = frozenset(
        {
            "schema_version",
            "git_object_vectors",
            "strict_json_vectors",
            "url_vectors",
            "header_vectors",
            "redirect_vectors",
        }
    )
    _require_exact_keys(root, expected_root_keys, "external anchor golden vectors")
    if root.get("schema_version") != EXTERNAL_ANCHOR_GOLDEN_VECTORS_SCHEMA_VERSION:
        raise ValueError("external anchor golden vector schema_version mismatch")
    executed: list[str] = []
    seen_case_ids: set[str] = set()

    def record(case_id: object, *, expected: object, actual: bool, label: str) -> None:
        normalized_id = _require_text(case_id, f"{label}.case_id")
        if normalized_id in seen_case_ids:
            raise ValueError(f"duplicate golden vector case_id: {normalized_id}")
        seen_case_ids.add(normalized_id)
        if type(expected) is not bool or actual is not expected:
            raise ValueError(
                f"golden vector {normalized_id} did not produce its pinned result: "
                f"expected={expected!r}, actual={actual!r}"
            )
        executed.append(normalized_id)

    category_outcomes: dict[str, set[bool]] = {}
    git_vectors = root.get("git_object_vectors")
    if type(git_vectors) is not list:
        raise ValueError("git_object_vectors must be an array")
    for index, item in enumerate(git_vectors):
        label = f"git_object_vectors[{index}]"
        vector = _require_mapping(item, label)
        _require_exact_keys(
            vector,
            frozenset(
                {
                    "case_id",
                    "object_type",
                    "payload_hex",
                    "expected_oid_sha1",
                    "expected_framed_sha256",
                    "should_pass",
                }
            ),
            label,
        )
        actual = False
        try:
            payload_hex = _require_text(vector["payload_hex"], f"{label}.payload_hex", allow_empty=True)
            if len(payload_hex) % 2 or (payload_hex and _LOWER_HEX_RE.fullmatch(payload_hex) is None):
                raise ValueError("golden Git payload_hex is not canonical lowercase hex")
            payload = bytes.fromhex(payload_hex)
            object_type = _require_text(vector["object_type"], f"{label}.object_type")
            actual = (
                git_object_oid_sha1(object_type, payload) == vector["expected_oid_sha1"]
                and _sha256(git_object_frame_sha1(object_type, payload)) == vector["expected_framed_sha256"]
            )
        except (TypeError, ValueError):
            actual = False
        should_pass = vector["should_pass"]
        record(vector["case_id"], expected=should_pass, actual=actual, label=label)
        category_outcomes.setdefault("git_object_vectors", set()).add(should_pass)

    strict_vectors = root.get("strict_json_vectors")
    if type(strict_vectors) is not list:
        raise ValueError("strict_json_vectors must be an array")
    for index, item in enumerate(strict_vectors):
        label = f"strict_json_vectors[{index}]"
        vector = _require_mapping(item, label)
        _require_exact_keys(
            vector,
            frozenset({"case_id", "payload_hex", "should_pass"}),
            label,
        )
        actual = True
        try:
            payload_hex = _require_text(vector["payload_hex"], f"{label}.payload_hex", allow_empty=True)
            if len(payload_hex) % 2 or (payload_hex and _LOWER_HEX_RE.fullmatch(payload_hex) is None):
                raise ValueError("golden JSON payload_hex is not canonical lowercase hex")
            strict_json_loads(bytes.fromhex(payload_hex), label=label)
        except (TypeError, ValueError):
            actual = False
        should_pass = vector["should_pass"]
        record(vector["case_id"], expected=should_pass, actual=actual, label=label)
        category_outcomes.setdefault("strict_json_vectors", set()).add(should_pass)

    url_vectors = root.get("url_vectors")
    if type(url_vectors) is not list:
        raise ValueError("url_vectors must be an array")
    for index, item in enumerate(url_vectors):
        label = f"url_vectors[{index}]"
        vector = _require_mapping(item, label)
        _require_exact_keys(
            vector,
            frozenset({"case_id", "url", "role", "should_pass"}),
            label,
        )
        actual = True
        try:
            role = _require_text(vector["role"], f"{label}.role")
            if role not in _ROLE_HOSTS:
                raise ValueError("golden URL role is not frozen")
            _require_role_url(vector["url"], role, label)
        except (KeyError, TypeError, ValueError):
            actual = False
        should_pass = vector["should_pass"]
        record(vector["case_id"], expected=should_pass, actual=actual, label=label)
        category_outcomes.setdefault("url_vectors", set()).add(should_pass)

    header_vectors = root.get("header_vectors")
    if type(header_vectors) is not list:
        raise ValueError("header_vectors must be an array")
    for index, item in enumerate(header_vectors):
        label = f"header_vectors[{index}]"
        vector = _require_mapping(item, label)
        _require_exact_keys(
            vector,
            frozenset(
                {
                    "case_id",
                    "role",
                    "body_length",
                    "pairs",
                    "redaction_facts",
                    "expected_wire_bytes",
                    "should_pass",
                }
            ),
            label,
        )
        actual = False
        try:
            role = _require_text(vector["role"], f"{label}.role")
            if role not in _ROLE_HOSTS:
                raise ValueError("golden header role is not frozen")
            body_length = _require_nonnegative_int(vector["body_length"], f"{label}.body_length")
            headers, _count, ledger_bytes, _hash = _normalized_headers(vector["pairs"], f"{label}.pairs")
            wire_bytes, redaction_valid = _header_wire_facts(
                headers,
                vector["redaction_facts"],
                ledger_byte_count=ledger_bytes,
                label=f"{label}.redaction_facts",
            )
            actual = (
                _header_semantics(headers, role=role, final_body_length=body_length)
                and redaction_valid
                and wire_bytes
                == _require_nonnegative_int(vector["expected_wire_bytes"], f"{label}.expected_wire_bytes")
            )
        except (KeyError, TypeError, ValueError):
            actual = False
        should_pass = vector["should_pass"]
        record(vector["case_id"], expected=should_pass, actual=actual, label=label)
        category_outcomes.setdefault("header_vectors", set()).add(should_pass)

    redirect_vectors = root.get("redirect_vectors")
    if type(redirect_vectors) is not list:
        raise ValueError("redirect_vectors must be an array")
    for index, item in enumerate(redirect_vectors):
        label = f"redirect_vectors[{index}]"
        vector = _require_mapping(item, label)
        _require_exact_keys(
            vector,
            frozenset({"case_id", "role", "requested_url", "final_url", "locations", "should_pass"}),
            label,
        )
        actual = _golden_redirect_chain(
            role=vector["role"],
            requested_url=vector["requested_url"],
            final_url=vector["final_url"],
            locations=vector["locations"],
        )
        should_pass = vector["should_pass"]
        record(vector["case_id"], expected=should_pass, actual=actual, label=label)
        category_outcomes.setdefault("redirect_vectors", set()).add(should_pass)

    if set(category_outcomes) != {
        "git_object_vectors",
        "strict_json_vectors",
        "url_vectors",
        "header_vectors",
        "redirect_vectors",
    } or any(outcomes != {False, True} for outcomes in category_outcomes.values()):
        raise ValueError("each golden-vector category must contain a passing and failing case")
    return {
        "schema_version": EXTERNAL_ANCHOR_GOLDEN_VECTORS_SCHEMA_VERSION,
        "case_count": len(executed),
        "executed_case_ids": executed,
        "all_vectors_passed": True,
    }


def _validate_http_capture(
    metadata_payload: bytes,
    body_payload: bytes,
    *,
    expected_metadata_name: str,
    expected_body_name: str,
    expected_role: str,
    expected_body_cap: int,
) -> tuple[dict[str, object], bool]:
    metadata = _canonical_object(metadata_payload, f"HTTP metadata {expected_metadata_name}")
    _require_exact_keys(metadata, _HTTP_KEYS, f"HTTP metadata {expected_metadata_name}")
    if metadata.get("schema_version") != "relationship-p4-external-anchor-observer-http.v1":
        raise ValueError("HTTP capture schema_version mismatch")
    role = metadata.get("role")
    if role != expected_role:
        raise ValueError(f"HTTP capture role mismatch for {expected_metadata_name}")
    requested_url = _require_text(metadata.get("requested_url"), "HTTP requested_url")
    final_url = _require_text(metadata.get("final_url"), "HTTP final_url")
    url_valid = True
    try:
        _require_role_url(requested_url, expected_role, "HTTP requested_url")
        _require_role_url(final_url, expected_role, "HTTP final_url")
    except ValueError:
        url_valid = False
    normalized, count, ledger_byte_count, header_hash = _normalized_headers(
        metadata.get("response_header_pairs"), "HTTP response_header_pairs"
    )
    wire_byte_count, redaction_facts_valid = _header_wire_facts(
        normalized,
        metadata.get("set_cookie_redaction_facts"),
        ledger_byte_count=ledger_byte_count,
        label="HTTP set_cookie_redaction_facts",
    )
    set_cookie_values = [value for name, value in normalized if name == "set-cookie"]
    set_cookie_closed = (
        metadata.get("set_cookie_present") is bool(set_cookie_values)
        and metadata.get("set_cookie_count") == len(set_cookie_values)
        and metadata.get("set_cookie_values_serialized") is False
        and all(value == "<redacted-set-cookie-value>" for value in set_cookie_values)
    )
    framing = _require_mapping(metadata.get("response_framing"), "HTTP response_framing")
    expected_framing_keys = frozenset(
        {
            "content_length_values",
            "transfer_encoding_values",
            "content_encoding_values",
            "content_type_values",
            "duplicate_content_length_rejected",
            "duplicate_transfer_encoding_rejected",
            "duplicate_content_encoding_rejected",
            "transfer_encoding_and_content_length_coexistence_rejected",
            "content_encoding_allowed",
            "declared_content_length_must_equal_captured_body",
        }
    )
    _require_exact_keys(framing, expected_framing_keys, "HTTP response_framing")
    by_name: dict[str, list[str]] = {}
    for name, value in normalized:
        by_name.setdefault(name, []).append(value)
    framing_closed = framing == {
        "content_length_values": by_name.get("content-length", []),
        "transfer_encoding_values": by_name.get("transfer-encoding", []),
        "content_encoding_values": by_name.get("content-encoding", []),
        "content_type_values": by_name.get("content-type", []),
        "duplicate_content_length_rejected": True,
        "duplicate_transfer_encoding_rejected": True,
        "duplicate_content_encoding_rejected": True,
        "transfer_encoding_and_content_length_coexistence_rejected": True,
        "content_encoding_allowed": ["absent", "identity"],
        "declared_content_length_must_equal_captured_body": True,
    }
    headers_closed = (
        metadata.get("response_header_count") == count
        and metadata.get("response_header_wire_bytes") == wire_byte_count
        and metadata.get("response_header_ledger_bytes") == ledger_byte_count
        and metadata.get("response_header_ledger_sha256") == header_hash
        and _header_semantics(normalized, role=expected_role, final_body_length=len(body_payload))
        and set_cookie_closed
        and redaction_facts_valid
        and framing_closed
    )
    redirects = metadata.get("redirects")
    if type(redirects) is not list:
        raise ValueError("HTTP redirects must be an array")
    redirect_limit = 0 if expected_role == "api" else 3
    redirect_valid = len(redirects) <= redirect_limit
    current_url = requested_url
    seen_urls = {current_url}
    for index, item in enumerate(redirects):
        hop = _require_mapping(item, f"HTTP redirects[{index}]")
        _require_exact_keys(hop, _REDIRECT_KEYS, f"HTTP redirects[{index}]")
        try:
            _require_role_url(hop.get("requested_url"), expected_role, "redirect requested_url")
            _require_role_url(hop.get("location"), expected_role, "redirect location")
        except ValueError:
            redirect_valid = False
        hop_headers, hop_count, hop_ledger_bytes, hop_hash = _normalized_headers(
            hop.get("response_header_pairs"), f"redirect[{index}] response_header_pairs"
        )
        hop_wire_bytes, hop_redaction_facts_valid = _header_wire_facts(
            hop_headers,
            hop.get("set_cookie_redaction_facts"),
            ledger_byte_count=hop_ledger_bytes,
            label=f"redirect[{index}] set_cookie_redaction_facts",
        )
        locations = [value for name, value in hop_headers if name == "location"]
        hop_cookies = [value for name, value in hop_headers if name == "set-cookie"]
        location = hop.get("location")
        redirect_valid = (
            redirect_valid
            and hop.get("requested_url") == current_url
            and hop.get("status") in _REDIRECT_STATUSES
            and hop.get("response_header_count") == hop_count
            and hop.get("response_header_wire_bytes") == hop_wire_bytes
            and hop.get("response_header_ledger_bytes") == hop_ledger_bytes
            and hop.get("response_header_ledger_sha256") == hop_hash
            and _header_semantics(hop_headers, role=expected_role, final_body_length=None)
            and locations == [location]
            and hop.get("set_cookie_present") is bool(hop_cookies)
            and hop.get("set_cookie_count") == len(hop_cookies)
            and hop.get("set_cookie_values_serialized") is False
            and all(value == "<redacted-set-cookie-value>" for value in hop_cookies)
            and hop_redaction_facts_valid
            and location not in seen_urls
        )
        if type(location) is str:
            current_url = location
            seen_urls.add(location)
    redirect_valid = redirect_valid and current_url == final_url
    body_reference = _require_mapping(metadata.get("body"), "HTTP body ref")
    _require_exact_keys(body_reference, _HTTP_BODY_REF_KEYS, "HTTP body ref")
    body_ref_valid = (
        {key: body_reference[key] for key in _BODY_REF_KEYS} == _file_ref(expected_body_name, body_payload)
        and body_reference.get("body_cap") == expected_body_cap
        and len(body_payload) <= expected_body_cap
    )
    request_headers = metadata.get("request_headers")
    expected_effective_headers = [
        ["Host", _ROLE_HOSTS[expected_role]],
        *[[name, value] for name, value in _REQUEST_HEADERS.items()],
    ]
    outbound_valid = (
        type(request_headers) is dict
        and request_headers == _REQUEST_HEADERS
        and metadata.get("effective_request_headers") == expected_effective_headers
        and all(
            hop.get("effective_request_headers") == expected_effective_headers
            for hop in redirects
            if isinstance(hop, Mapping)
        )
        and metadata.get("authorization_header_sent") is False
        and metadata.get("cookie_header_sent") is False
        and metadata.get("proxy_used") is False
        and metadata.get("netrc_used") is False
    )
    semantics = (
        metadata.get("method") == "GET"
        and metadata.get("status") == 200
        and metadata.get("retry_count") == 0
        and metadata.get("facts_only_no_verdict") is True
        and url_valid
        and headers_closed
        and redirect_valid
        and body_ref_valid
        and outbound_valid
        and metadata.get("role_redirect_max_hops") == redirect_limit
        and metadata.get("connect_timeout_seconds") == 10
        and metadata.get("read_idle_timeout_seconds") == 10
        and metadata.get("request_total_timeout_seconds") == 30
    )
    projection = {
        "role": expected_role,
        "requested_url": requested_url,
        "final_url": final_url,
        "status": _require_int(metadata.get("status"), "HTTP status", minimum=100),
        "response_header_count": count,
        "response_header_wire_bytes": wire_byte_count,
        "response_header_ledger_bytes": ledger_byte_count,
        "response_header_ledger_sha256": header_hash,
        "redirect_count": len(redirects),
        "body": _file_ref(expected_body_name, body_payload),
    }
    del expected_metadata_name
    return projection, semantics


def _api_body_projection(
    payload: bytes,
    *,
    claim: Mapping[str, object],
    target: Mapping[str, object],
    local_request_bytes: bytes,
) -> tuple[dict[str, object], bool, bool]:
    api = strict_json_object_from_bytes(payload, label="GitHub exact-revision API body")
    owner = _require_mapping(api.get("owner"), "GitHub API owner")
    files = api.get("files")
    history = api.get("history")
    if type(files) is not dict or type(history) is not list:
        raise ValueError("GitHub API files/history shape is invalid")
    filename = target["required_filename"]
    file_payload = _require_mapping(files.get(filename), "GitHub API sole file")
    raw_url = _require_text(file_payload.get("raw_url"), "GitHub API raw_url")
    html_base = _require_text(api.get("html_url"), "GitHub API html_url")
    git_pull_url = _require_text(api.get("git_pull_url"), "GitHub API git_pull_url")
    url_shapes_valid = True
    for value, role, label in (
        (raw_url, "raw", "GitHub API raw_url"),
        (html_base, "html_git", "GitHub API html_url"),
        (git_pull_url, "html_git", "GitHub API git_pull_url"),
    ):
        try:
            _require_role_url(value, role, label)
        except ValueError:
            url_shapes_valid = False
    raw_path_match = re.fullmatch(
        rf"/{re.escape(target['expected_owner_login'])}/{claim['gist_id']}/raw/"
        rf"([0-9a-f]{{40}})/{re.escape(target['required_filename'])}",
        urlsplit(raw_url).path,
    )
    url_shapes_valid = (
        url_shapes_valid
        and raw_path_match is not None
        and urlsplit(html_base).path == f"/{claim['gist_id']}"
        and urlsplit(git_pull_url).path == f"/{claim['gist_id']}.git"
    )
    owner_valid = (
        owner.get("login") == target["expected_owner_login"]
        and owner.get("id") == target["expected_owner_id"]
        and owner.get("node_id") == target["expected_owner_node_id"]
        and owner.get("type") == target["expected_owner_type"]
        and owner.get("site_admin") == target["expected_owner_site_admin"]
    )
    history_valid = (
        len(history) == target["required_history_count"] == 1
        and isinstance(history[0], Mapping)
        and history[0].get("version") == claim["revision_oid"]
    )
    metadata_valid = (
        api.get("id") == claim["gist_id"]
        and type(api.get("node_id")) is str
        and re.fullmatch(r"[A-Za-z0-9_+/=-]{1,256}", api["node_id"]) is not None
        and api.get("public") is True
        and api.get("description") == target["required_description"] == ""
        and api.get("truncated") is False
        and len(files) == target["required_file_count"] == 1
        and owner_valid
        and history_valid
        and file_payload.get("filename") == filename
        and file_payload.get("size") == len(local_request_bytes)
        and file_payload.get("truncated") is False
        and file_payload.get("encoding") == "utf-8"
        and url_shapes_valid
    )
    content = file_payload.get("content")
    content_valid = type(content) is str and content.encode("utf-8") == local_request_bytes
    projection = {
        "gist_id": api.get("id"),
        "gist_node_id": api.get("node_id"),
        "owner_login": owner.get("login"),
        "owner_id": owner.get("id"),
        "owner_node_id": owner.get("node_id"),
        "public": api.get("public"),
        "description": api.get("description"),
        "revision_oid": history[0].get("version") if history and isinstance(history[0], Mapping) else None,
        "history_count": len(history),
        "file_count": len(files),
        "filename": file_payload.get("filename"),
        "file_size": file_payload.get("size"),
        "file_truncated": file_payload.get("truncated"),
        "api_content_sha256": _sha256(content.encode("utf-8")) if type(content) is str else None,
        "raw_revision_token": raw_path_match.group(1) if raw_path_match is not None else None,
        "raw_url": raw_url,
        "html_url": html_base,
        "git_pull_url": git_pull_url,
    }
    return projection, metadata_valid, content_valid


def _git_toolchain_contract(request: Mapping[str, object]) -> dict[str, object]:
    acquisition = _require_mapping(
        request.get("observer_acquisition_contract"), "request observer_acquisition_contract"
    )
    toolchain = _require_mapping(
        acquisition.get("production_Windows_Git_executable"),
        "observer_acquisition_contract.production_Windows_Git_executable",
    )
    expected_keys = frozenset(
        {
            "absolute_path",
            "byte_count",
            "raw_sha256",
            "version_stdout",
            "transport_helpers",
            "must_be_revalidated_from_one_regular_default_stream_buffer_before_use",
            "version_is_environment_metadata_but_executable_hash_is_required_for_production_backend",
            "transitive_OS_DLL_and_TLS_library_closure_is_fully_pinned",
            "cooperative_local_Git_toolchain_trust_remains",
        }
    )
    _require_exact_keys(toolchain, expected_keys, "production Git toolchain contract")
    helpers = toolchain.get("transport_helpers")
    if type(helpers) is not list or not helpers:
        raise ValueError("production Git helper identities must be a non-empty array")
    normalized_helpers: list[list[object]] = []
    for index, item in enumerate(helpers):
        helper = _require_mapping(item, f"production Git helper identity {index}")
        _require_exact_keys(
            helper,
            frozenset({"absolute_path", "byte_count", "raw_sha256"}),
            f"production Git helper identity {index}",
        )
        normalized_helpers.append(
            [
                _require_canonical_windows_local_path(helper["absolute_path"], f"production Git helper {index} path"),
                _require_nonnegative_int(helper["byte_count"], f"production Git helper {index} byte count"),
                _require_lower_hex(helper["raw_sha256"], 64, f"production Git helper {index} SHA-256"),
            ]
        )
    if (
        toolchain.get("must_be_revalidated_from_one_regular_default_stream_buffer_before_use") is not True
        or toolchain.get("version_is_environment_metadata_but_executable_hash_is_required_for_production_backend")
        is not True
        or toolchain.get("transitive_OS_DLL_and_TLS_library_closure_is_fully_pinned") is not False
        or toolchain.get("cooperative_local_Git_toolchain_trust_remains") is not True
    ):
        raise ValueError("production Git toolchain trust-boundary contract drift")
    return {
        "executable_path": _require_canonical_windows_local_path(
            toolchain.get("absolute_path"), "production Git executable path"
        ),
        "executable_byte_count": _require_nonnegative_int(
            toolchain.get("byte_count"), "production Git executable byte count"
        ),
        "executable_raw_sha256": _require_lower_hex(
            toolchain.get("raw_sha256"), 64, "production Git executable SHA-256"
        ),
        "version_stdout": _require_text(toolchain.get("version_stdout"), "production Git version stdout"),
        "helper_identities": normalized_helpers,
    }


def _parse_advertised_refs(payload: bytes) -> list[list[str]]:
    refs: list[list[str]] = []
    for line in payload.splitlines():
        if not line:
            continue
        pieces = line.split(b"\t")
        if len(pieces) != 2:
            raise ValueError("raw advertised refs line is malformed")
        try:
            oid = pieces[0].decode("ascii")
            name = pieces[1].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("raw advertised refs encoding is invalid") from exc
        _require_lower_hex(oid, 40, "advertised ref OID")
        if not name.startswith("refs/heads/") or name == "refs/heads/":
            raise ValueError("advertised ref is not a named head")
        refs.append([name, oid])
    if len(refs) != 1:
        raise ValueError("raw advertised refs must contain exactly one head")
    return refs


def _parse_fetched_refs(payload: bytes) -> list[list[str]]:
    refs: list[list[str]] = []
    for line in payload.splitlines():
        if not line:
            continue
        pieces = line.split(b"\0")
        if len(pieces) != 2:
            raise ValueError("raw fetched refs line is malformed")
        try:
            name = pieces[0].decode("utf-8")
            oid = pieces[1].decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("raw fetched refs encoding is invalid") from exc
        _require_lower_hex(oid, 40, "fetched ref OID")
        if not name.startswith("refs/remotes/origin/") or name == "refs/remotes/origin/HEAD":
            raise ValueError("fetched ref is not a named origin head")
        refs.append([name, oid])
    if len(refs) != 1:
        raise ValueError("raw fetched refs must contain exactly one head")
    return refs


def _parse_object_inventory(payload: bytes) -> list[list[object]]:
    objects: list[list[object]] = []
    for line in payload.splitlines():
        if not line:
            continue
        pieces = line.split(b"\0")
        if len(pieces) != 3:
            raise ValueError("raw Git object inventory row is malformed")
        try:
            oid = pieces[0].decode("ascii")
            object_type = pieces[1].decode("ascii")
            size_text = pieces[2].decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("raw Git object inventory encoding is invalid") from exc
        _require_lower_hex(oid, 40, "inventory object OID")
        if object_type not in _GIT_OBJECT_TYPES or not size_text.isdecimal():
            raise ValueError("raw Git object inventory value is invalid")
        size = int(size_text)
        if size > 4_194_304:
            raise ValueError("raw Git object inventory size exceeds the fixed cap")
        objects.append([oid, object_type, size])
    if len(objects) != 3 or len({item[0] for item in objects}) != 3:
        raise ValueError("raw Git object inventory must contain exactly three unique objects")
    return sorted(objects, key=lambda item: item[0])


def _validate_git_isolation_ledger(
    metadata: Mapping[str, object],
    *,
    expected_toolchain: Mapping[str, object],
    remote_url: str,
    production: bool,
) -> bool:
    commands = metadata.get("command_argv_ledger")
    environment = metadata.get("environment_ledger")
    artifacts = metadata.get("isolation_artifacts")
    if type(commands) is not list or len(commands) != 9 or type(environment) is not list:
        return False
    normalized_environment: dict[str, str] = {}
    for item in environment:
        if type(item) is not list or len(item) != 2 or any(type(value) is not str for value in item):
            return False
        key, value = item
        if key in normalized_environment:
            return False
        normalized_environment[key] = value
    required_windows_environment = {"SYSTEMROOT", "WINDIR"}
    required_dynamic_environment = {
        "TMP",
        "TEMP",
        "TMPDIR",
        "GIT_CONFIG_GLOBAL",
        "XDG_CONFIG_HOME",
        "GIT_EXEC_PATH",
    }
    expected_environment_keys = (
        set(_GIT_REQUIRED_ENVIRONMENT) | required_dynamic_environment | required_windows_environment
    )
    if set(normalized_environment) != expected_environment_keys:
        return False
    if any(normalized_environment.get(key) != value for key, value in _GIT_REQUIRED_ENVIRONMENT.items()):
        return False
    try:
        system_root = _require_canonical_windows_local_path(
            normalized_environment["SYSTEMROOT"], "Git environment SYSTEMROOT"
        )
        windows_directory = _require_canonical_windows_local_path(
            normalized_environment["WINDIR"], "Git environment WINDIR"
        )
        dynamic_paths = {
            key: _require_canonical_windows_local_path(normalized_environment[key], f"Git environment {key}")
            for key in required_dynamic_environment
        }
    except ValueError:
        return False
    if ntpath.normcase(system_root) != ntpath.normcase(windows_directory):
        return False
    if not (dynamic_paths["TMP"] == dynamic_paths["TEMP"] == dynamic_paths["TMPDIR"]):
        return False
    temporary_root = ntpath.dirname(dynamic_paths["TMP"])
    if (
        not temporary_root
        or not ntpath.basename(temporary_root).startswith(".observer-git-")
        or ntpath.basename(dynamic_paths["TMP"]) != "io"
        or dynamic_paths["GIT_CONFIG_GLOBAL"] != ntpath.join(temporary_root, "empty-global-config")
        or dynamic_paths["XDG_CONFIG_HOME"] != ntpath.join(temporary_root, "empty-xdg-config")
    ):
        return False
    helper_identities = expected_toolchain.get("helper_identities")
    if type(helper_identities) is not list or not helper_identities:
        return False
    helper_path = helper_identities[0][0]
    if type(helper_path) is not str or dynamic_paths["GIT_EXEC_PATH"] != ntpath.dirname(helper_path):
        return False

    executable = expected_toolchain["executable_path"]
    hooks = ntpath.join(temporary_root, "disabled-hooks")
    bare = ntpath.join(temporary_root, "repository.git")
    template = ntpath.join(temporary_root, "empty-template")
    fresh_children = {
        dynamic_paths["TMP"],
        dynamic_paths["GIT_CONFIG_GLOBAL"],
        dynamic_paths["XDG_CONFIG_HOME"],
        hooks,
        bare,
        template,
    }
    if len({ntpath.normcase(path) for path in fresh_children}) != 6 or any(
        ntpath.dirname(path) != temporary_root for path in fresh_children
    ):
        return False
    fixed: list[str] = ["--no-optional-locks"]
    for value in _GIT_REQUIRED_CONFIG_ARGUMENTS[:-2]:
        fixed.extend(["-c", value])
    fixed.extend(["-c", f"core.hooksPath={hooks}"])
    for value in _GIT_REQUIRED_CONFIG_ARGUMENTS[-2:]:
        fixed.extend(["-c", value])
    tails = [
        ["init", "--bare", "--object-format=sha1", f"--template={template}", bare],
        ["ls-remote", "--heads", remote_url],
        [
            "-C",
            bare,
            "fetch",
            "--quiet",
            "--no-tags",
            "--force",
            "--prune",
            "--no-recurse-submodules",
            remote_url,
            "+refs/heads/*:refs/remotes/origin/*",
        ],
        ["-C", bare, "fsck", "--full", "--strict", "--no-reflogs"],
        [
            "-C",
            bare,
            "for-each-ref",
            "--format=%(refname)%00%(objectname)",
            "refs/remotes/origin",
        ],
        ["-C", bare, "cat-file", "--batch"],
        ["-C", bare, "cat-file", "--batch"],
        ["-C", bare, "cat-file", "--batch"],
        [
            "-C",
            bare,
            "cat-file",
            "--batch-all-objects",
            "--batch-check=%(objectname)%00%(objecttype)%00%(objectsize)",
        ],
    ]
    expected_commands = [[executable, *fixed, *tail] for tail in tails]
    if commands != expected_commands:
        return False
    return artifacts == {
        "global_config": "create_only_empty_file_inside_fresh_temporary_root",
        "XDG_CONFIG_HOME": "fresh_empty_directory_inside_fresh_temporary_root",
        "hooks_path": "fresh_empty_directory_inside_fresh_temporary_root",
        "system_config_disabled": production,
        "HOME_and_USERPROFILE_absent": production,
        "applied_in_production_backend": production,
    }


def _validate_git_capture(
    metadata_payload: bytes,
    *,
    commit_payload: bytes,
    tree_payload: bytes,
    blob_payload: bytes,
    claim: Mapping[str, object],
    target: Mapping[str, object],
    backend_kind: str,
    expected_toolchain: Mapping[str, object],
    advertised_refs_payload: bytes,
    fetched_refs_payload: bytes,
    object_inventory_payload: bytes,
) -> tuple[dict[str, object], bool]:
    metadata = _canonical_object(metadata_payload, "Git capture metadata")
    _require_exact_keys(metadata, _GIT_KEYS, "Git capture metadata")
    if metadata.get("schema_version") != "relationship-p4-external-anchor-observer-git.v1":
        raise ValueError("Git capture schema_version mismatch")
    for field, name, payload in (
        ("advertised_refs_raw_stdout", GIT_ADVERTISED_REFS_FILE, advertised_refs_payload),
        ("fetched_refs_raw_stdout", GIT_FETCHED_REFS_FILE, fetched_refs_payload),
        ("object_inventory_raw_stdout", GIT_OBJECT_INVENTORY_FILE, object_inventory_payload),
        ("commit_body", GIT_COMMIT_FILE, commit_payload),
        ("tree_body", GIT_TREE_FILE, tree_payload),
        ("blob_body", GIT_BLOB_FILE, blob_payload),
    ):
        _validate_body_ref(metadata.get(field), expected_name=name, payload=payload, label=f"Git {field}")
    revision = _require_lower_hex(metadata.get("revision_oid"), 40, "Git revision OID")
    metadata_commit = _require_lower_hex(metadata.get("commit_oid"), 40, "Git commit OID")
    metadata_tree = _require_lower_hex(metadata.get("tree_oid"), 40, "Git tree OID")
    metadata_blob = _require_lower_hex(metadata.get("blob_oid"), 40, "Git blob OID")
    advertised_refs = _parse_advertised_refs(advertised_refs_payload)
    fetched_refs = _parse_fetched_refs(fetched_refs_payload)
    object_inventory = _parse_object_inventory(object_inventory_payload)
    refs_valid = (
        target["required_head_count"] == 1
        and advertised_refs[0][1] == revision
        and fetched_refs[0][1] == revision
        and metadata.get("advertised_refs") == advertised_refs
        and metadata.get("fetched_refs") == fetched_refs
    )
    closure: VerifiedGitClosure | None = None
    try:
        closure = verify_zero_parent_single_file_git_closure(
            commit_payload=commit_payload,
            tree_payload=tree_payload,
            blob_payload=blob_payload,
            required_filename=target["required_filename"],
            expected_revision_oid_sha1=claim["revision_oid"],
        )
    except (TypeError, ValueError):
        closure = None
    remote_valid = True
    try:
        _require_role_url(metadata.get("remote_url"), "html_git", "Git remote_url")
    except ValueError:
        remote_valid = False
    toolchain = _require_mapping(metadata.get("production_git_toolchain"), "Git production_git_toolchain")
    expected_metadata_toolchain = {
        **dict(expected_toolchain),
        "preflight_completed_before_HTTP": backend_kind == REAL_OBSERVER_BACKEND,
    }
    toolchain_valid = dict(toolchain) == expected_metadata_toolchain
    ledger_valid = _validate_git_isolation_ledger(
        metadata,
        expected_toolchain=expected_toolchain,
        remote_url=metadata.get("remote_url"),
        production=backend_kind == REAL_OBSERVER_BACKEND,
    )
    expected_inventory = sorted(
        [
            [git_object_oid_sha1("commit", commit_payload), "commit", len(commit_payload)],
            [git_object_oid_sha1("tree", tree_payload), "tree", len(tree_payload)],
            [git_object_oid_sha1("blob", blob_payload), "blob", len(blob_payload)],
        ],
        key=lambda item: item[0],
    )
    production = backend_kind == REAL_OBSERVER_BACKEND
    isolation_facts_valid = (
        metadata.get("fresh_bare_repository") is production
        and metadata.get("all_heads_fetch_refspec") == "+refs/heads/*:refs/remotes/origin/*"
        and metadata.get("system_and_global_config_disabled") is production
        and metadata.get(
            "credentials_askpass_extra_headers_cookies_proxy_custom_CA_redirects_hooks_"
            "alternates_replace_and_shallow_disabled"
        )
        is production
    )
    object_store_byte_count = _require_nonnegative_int(
        metadata.get("object_store_byte_count"), "Git object store byte count"
    )
    closure_valid = (
        closure is not None
        and revision == claim["revision_oid"]
        and metadata_commit == closure.commit_oid_sha1
        and metadata_tree == closure.tree_oid_sha1
        and metadata_blob == closure.blob_oid_sha1
        and metadata.get("tree_entry_mode") == "100644"
        and metadata.get("tree_entry_name") == target["required_filename"]
        and refs_valid
        and remote_valid
        and toolchain_valid
        and ledger_valid
        and isolation_facts_valid
        and metadata.get("object_inventory") == object_inventory == expected_inventory
        and object_store_byte_count <= 4_194_304
        and metadata.get("facts_only_no_verdict") is True
    )
    fsck_stdout = _require_lower_hex(metadata.get("fsck_stdout_sha256"), 64, "Git fsck stdout SHA-256")
    fsck_stderr = _require_lower_hex(metadata.get("fsck_stderr_sha256"), 64, "Git fsck stderr SHA-256")
    projection = {
        "remote_url": metadata.get("remote_url"),
        "advertised_heads": advertised_refs,
        "fetched_heads": fetched_refs,
        "object_inventory": object_inventory,
        "object_inventory_payload": _file_ref(GIT_OBJECT_INVENTORY_FILE, object_inventory_payload),
        "commit_oid_sha1": git_object_oid_sha1("commit", commit_payload),
        "commit_framed_sha256": _sha256(git_object_frame_sha1("commit", commit_payload)),
        "commit_payload": _file_ref(GIT_COMMIT_FILE, commit_payload),
        "commit_parent_count": 0 if closure is not None else None,
        "tree_oid_sha1": git_object_oid_sha1("tree", tree_payload),
        "tree_framed_sha256": _sha256(git_object_frame_sha1("tree", tree_payload)),
        "tree_payload": _file_ref(GIT_TREE_FILE, tree_payload),
        "tree_entry_count": 1 if closure is not None else None,
        "tree_entry_mode": metadata.get("tree_entry_mode"),
        "tree_entry_filename": metadata.get("tree_entry_name"),
        "blob_oid_sha1": git_object_oid_sha1("blob", blob_payload),
        "blob_framed_sha256": _sha256(git_object_frame_sha1("blob", blob_payload)),
        "blob_payload": _file_ref(GIT_BLOB_FILE, blob_payload),
        "fsck_stdout_sha256": fsck_stdout,
        "fsck_stderr_sha256": fsck_stderr,
        "production_git_toolchain": dict(toolchain),
    }
    return projection, closure_valid


def _same_anchor_projection(
    *,
    api_start: Mapping[str, object],
    api_end: Mapping[str, object],
    claim: Mapping[str, object],
    target: Mapping[str, object],
    api_start_http: Mapping[str, object],
    api_end_http: Mapping[str, object],
    raw_http: Mapping[str, object],
    html_http: Mapping[str, object],
    git_projection: Mapping[str, object],
) -> bool:
    expected_api = f"https://{_ROLE_HOSTS['api']}/gists/{claim['gist_id']}/{claim['revision_oid']}"
    expected_raw = api_start["raw_url"]
    expected_html = api_start["html_url"]
    return (
        api_start == api_end
        and api_start["gist_id"] == claim["gist_id"]
        and api_start["owner_login"] == target["expected_owner_login"]
        and api_start["owner_id"] == target["expected_owner_id"]
        and api_start["owner_node_id"] == target["expected_owner_node_id"]
        and api_start["revision_oid"] == claim["revision_oid"]
        and api_start_http["requested_url"] == api_start_http["final_url"] == expected_api
        and api_end_http["requested_url"] == api_end_http["final_url"] == expected_api
        and raw_http["requested_url"] == expected_raw
        and html_http["requested_url"] == expected_html
        and git_projection["remote_url"] == api_start["git_pull_url"]
        and git_projection["commit_oid_sha1"] == claim["revision_oid"]
        and git_projection["tree_entry_filename"] == target["required_filename"]
    )


def _receipt_verdict(checks: Mapping[str, bool]) -> dict[str, bool]:
    integrity = all(checks[name] for name in ("L", "H", "T", "J", "G", "B"))
    observed = integrity and checks["E"]
    authority = {key: False for key in sorted(_AUTHORITY_FIREWALL_KEYS)}
    return {
        "integrity_valid": integrity,
        "observation_complete": observed,
        **authority,
    }


def verify_receipt_bundle(
    *,
    expected_protocol_raw_sha256: str,
    local_protocol_bytes: bytes,
    local_request_bytes: bytes,
    local_manifest_bytes: bytes,
    normative_file_buffers: Mapping[str, bytes],
    capture_files: Mapping[str, bytes],
) -> dict[str, object]:
    """Replay one complete raw R0/R1 bundle and construct its canonical receipt."""

    buffers = _validate_capture_file_mapping(capture_files)
    local_binding, _protocol, request, _manifest, local_valid = _local_lineage(
        expected_protocol_raw_sha256=expected_protocol_raw_sha256,
        local_protocol_bytes=local_protocol_bytes,
        local_request_bytes=local_request_bytes,
        local_manifest_bytes=local_manifest_bytes,
        normative_file_buffers=normative_file_buffers,
    )
    target = _target_contract(request)
    _claim, claim, claim_lineage_valid = _validate_claim(
        buffers[CLAIM_FILE],
        local_binding=local_binding,
        target=target,
    )
    capture_binding, capture_complete = _validate_capture_map_and_terminal(
        buffers,
        claim_projection=claim,
    )

    api_start_http, api_start_http_valid = _validate_http_capture(
        buffers[API_START_HTTP_FILE],
        buffers[API_START_BODY_FILE],
        expected_metadata_name=API_START_HTTP_FILE,
        expected_body_name=API_START_BODY_FILE,
        expected_role="api",
        expected_body_cap=262_144,
    )
    raw_http, raw_http_valid = _validate_http_capture(
        buffers[RAW_HTTP_FILE],
        buffers[RAW_BODY_FILE],
        expected_metadata_name=RAW_HTTP_FILE,
        expected_body_name=RAW_BODY_FILE,
        expected_role="raw",
        expected_body_cap=len(local_request_bytes),
    )
    html_http, html_http_valid = _validate_http_capture(
        buffers[HTML_HTTP_FILE],
        buffers[HTML_BODY_FILE],
        expected_metadata_name=HTML_HTTP_FILE,
        expected_body_name=HTML_BODY_FILE,
        expected_role="html_git",
        expected_body_cap=2_097_152,
    )
    api_end_http, api_end_http_valid = _validate_http_capture(
        buffers[API_END_HTTP_FILE],
        buffers[API_END_BODY_FILE],
        expected_metadata_name=API_END_HTTP_FILE,
        expected_body_name=API_END_BODY_FILE,
        expected_role="api",
        expected_body_cap=262_144,
    )
    api_start, start_target_valid, start_content_valid = _api_body_projection(
        buffers[API_START_BODY_FILE],
        claim=claim,
        target=target,
        local_request_bytes=local_request_bytes,
    )
    api_end, end_target_valid, end_content_valid = _api_body_projection(
        buffers[API_END_BODY_FILE],
        claim=claim,
        target=target,
        local_request_bytes=local_request_bytes,
    )
    expected_git_toolchain = _git_toolchain_contract(request)
    git_projection, git_valid = _validate_git_capture(
        buffers[GIT_CAPTURE_FILE],
        advertised_refs_payload=buffers[GIT_ADVERTISED_REFS_FILE],
        fetched_refs_payload=buffers[GIT_FETCHED_REFS_FILE],
        object_inventory_payload=buffers[GIT_OBJECT_INVENTORY_FILE],
        commit_payload=buffers[GIT_COMMIT_FILE],
        tree_payload=buffers[GIT_TREE_FILE],
        blob_payload=buffers[GIT_BLOB_FILE],
        claim=claim,
        target=target,
        backend_kind=claim["backend_kind"],
        expected_toolchain=expected_git_toolchain,
    )
    identity_join_valid = _same_anchor_projection(
        api_start=api_start,
        api_end=api_end,
        claim=claim,
        target=target,
        api_start_http=api_start_http,
        api_end_http=api_end_http,
        raw_http=raw_http,
        html_http=html_http,
        git_projection=git_projection,
    )
    request_bytes_valid = (
        exact_request_bytes_match(
            local_request_payload=local_request_bytes,
            git_blob_payload=buffers[GIT_BLOB_FILE],
            observed_raw_payload=buffers[RAW_BODY_FILE],
            expected_raw_sha256=local_binding["request_payload_raw_sha256"],
            expected_byte_count=local_binding["request_payload_byte_count"],
        )
        and start_content_valid
        and end_content_valid
    )
    checks = {
        "L": local_valid and claim_lineage_valid and capture_complete,
        "H": api_start_http_valid and raw_http_valid and html_http_valid and api_end_http_valid,
        "T": start_target_valid and end_target_valid,
        "J": identity_join_valid,
        "G": git_valid,
        "B": request_bytes_valid,
        "E": claim["backend_kind"] == REAL_OBSERVER_BACKEND,
    }
    anchor_projection = {
        "gist_id": api_start["gist_id"],
        "gist_node_id": api_start["gist_node_id"],
        "owner_login": api_start["owner_login"],
        "owner_id": api_start["owner_id"],
        "owner_node_id": api_start["owner_node_id"],
        "public": api_start["public"],
        "description": api_start["description"],
        "revision_oid": api_start["revision_oid"],
        "filename": api_start["filename"],
        "raw_revision_token": api_start["raw_revision_token"],
        "api_revision_url": api_start_http["final_url"],
        "raw_revision_url": raw_http["final_url"],
        "html_presentation_url": html_http["final_url"],
        "git_pull_url": git_projection["remote_url"],
    }
    core: dict[str, object] = {
        "schema_version": EXTERNAL_ANCHOR_RECEIPT_SCHEMA_VERSION,
        "receipt_id_contract": "sha256_canonical_json_utf8_newline_without_receipt_id_v1",
        "observation_role": claim["observation_stage"],
        "predecessor_receipt_id": claim["predecessor_receipt_id"],
        "predecessor_receipt_bundle_manifest_raw_sha256": claim["predecessor_receipt_bundle_manifest_raw_sha256"],
        "normative_binding": {
            "protocol_id": local_binding["protocol_id"],
            "protocol_raw_sha256": local_binding["protocol_raw_sha256"],
            "protocol_byte_count": local_binding["protocol_byte_count"],
            "normative_files": local_binding["normative_files"],
        },
        "request_binding": {
            key: local_binding[key]
            for key in (
                "request_id",
                "request_artifact_id",
                "request_payload_raw_sha256",
                "request_payload_byte_count",
                "request_manifest_raw_sha256",
                "request_manifest_byte_count",
            )
        },
        "capture_binding": capture_binding,
        "anchor_projection": anchor_projection,
        "http_projection": {
            "api_start": api_start_http,
            "returned_raw": raw_http,
            "returned_html": html_http,
            "api_end": api_end_http,
        },
        "git_projection": git_projection,
        "derived_checks": checks,
        "verdict": _receipt_verdict(checks),
    }
    receipt_id = _sha256(canonical_json_bytes(core))
    return {**core, "receipt_id": receipt_id}


def validate_external_anchor_receipt(
    receipt_bytes: bytes,
    *,
    expected_protocol_raw_sha256: str,
    local_protocol_bytes: bytes,
    local_request_bytes: bytes,
    local_manifest_bytes: bytes,
    normative_file_buffers: Mapping[str, bytes],
    capture_files: Mapping[str, bytes],
) -> dict[str, object]:
    """Replay raw capture bytes and require an exact canonical receipt."""

    supplied = _canonical_object(receipt_bytes, "external anchor receipt")
    expected = verify_receipt_bundle(
        expected_protocol_raw_sha256=expected_protocol_raw_sha256,
        local_protocol_bytes=local_protocol_bytes,
        local_request_bytes=local_request_bytes,
        local_manifest_bytes=local_manifest_bytes,
        normative_file_buffers=normative_file_buffers,
        capture_files=capture_files,
    )
    if supplied != expected or receipt_bytes != canonical_json_bytes(expected):
        raise ValueError("external anchor receipt does not equal full raw-bundle replay")
    return expected


def _receipt_bundle_file_ref(role: str, name: str, payload: bytes) -> dict[str, object]:
    if type(payload) is not bytes:
        raise TypeError(f"receipt-bundle file {role} must be bytes")
    return {
        "role": _require_text(role, "receipt-bundle file role"),
        "name": _require_text(name, "receipt-bundle file name"),
        "byte_count": len(payload),
        "sha256": _sha256(payload),
        "git_blob_oid_sha1": git_object_oid_sha1("blob", payload),
    }


def build_receipt_bundle_manifest(
    *,
    local_protocol_bytes: bytes,
    local_request_bytes: bytes,
    local_manifest_bytes: bytes,
    normative_file_buffers: Mapping[str, bytes],
    capture_files: Mapping[str, bytes],
    receipt_bytes: bytes,
) -> dict[str, object]:
    """Seal replay inputs; construction alone is explicitly non-authorizing."""

    receipt = _canonical_object(receipt_bytes, "external anchor receipt for bundle manifest")
    _receipt_core, receipt_id = _without_id(receipt, "receipt_id", "external anchor receipt")
    observation_role = receipt.get("observation_role")
    if observation_role not in {"R0", "R1"}:
        raise ValueError("receipt-bundle manifest requires an R0 or R1 receipt")
    normative_binding = _require_mapping(receipt.get("normative_binding"), "receipt normative binding")
    normative_descriptors = _require_mapping(normative_binding.get("normative_files"), "receipt normative files")
    if set(normative_file_buffers) != set(_REQUIRED_NORMATIVE_ROLES):
        raise ValueError("receipt-bundle normative buffers must contain exactly six roles")
    buffers = _validate_capture_file_mapping(capture_files)
    files = [
        _receipt_bundle_file_ref("local_protocol", "100_local_protocol.json", local_protocol_bytes),
        _receipt_bundle_file_ref("local_request", "101_local_request.json", local_request_bytes),
        _receipt_bundle_file_ref("local_request_manifest", "102_local_request_manifest.json", local_manifest_bytes),
    ]
    for role in _REQUIRED_NORMATIVE_ROLES:
        descriptor = _require_mapping(normative_descriptors.get(role), f"receipt normative descriptor {role}")
        _require_text(descriptor.get("repo_relative_posix_path"), f"receipt normative path {role}")
        name = _RECEIPT_BUNDLE_NORMATIVE_NAMES[role]
        files.append(_receipt_bundle_file_ref(f"normative:{role}", name, normative_file_buffers[role]))
    files.extend(
        _receipt_bundle_file_ref(f"capture:{_RECEIPT_BUNDLE_CAPTURE_ROLES[name]}", name, buffers[name])
        for name in CAPTURE_FILE_NAMES
    )
    files.append(_receipt_bundle_file_ref("canonical_receipt", "200_external_anchor_receipt.json", receipt_bytes))
    firewall = {key: False for key in sorted(_AUTHORITY_FIREWALL_KEYS)}
    core: dict[str, object] = {
        "schema_version": EXTERNAL_ANCHOR_RECEIPT_BUNDLE_MANIFEST_SCHEMA_VERSION,
        "artifact_id_contract": "sha256_canonical_json_utf8_newline_without_artifact_id_v1",
        "observation_role": observation_role,
        "receipt_id": receipt_id,
        "file_count": len(files),
        "files": files,
        "authority_firewall": firewall,
    }
    return {**core, "artifact_id": _sha256(canonical_json_bytes(core))}


def validate_receipt_bundle_manifest(
    manifest_bytes: bytes,
    *,
    local_protocol_bytes: bytes,
    local_request_bytes: bytes,
    local_manifest_bytes: bytes,
    normative_file_buffers: Mapping[str, bytes],
    capture_files: Mapping[str, bytes],
    receipt_bytes: bytes,
) -> dict[str, object]:
    """Validate byte sealing only; this low-level helper grants no authority."""

    supplied = _canonical_object(manifest_bytes, "external anchor receipt-bundle manifest")
    _require_exact_keys(
        supplied,
        _RECEIPT_BUNDLE_MANIFEST_KEYS,
        "external anchor receipt-bundle manifest",
    )
    _manifest_core, _artifact_id = _without_id(
        supplied,
        "artifact_id",
        "external anchor receipt-bundle manifest",
    )
    files = supplied.get("files")
    if type(files) is not list:
        raise ValueError("external anchor receipt-bundle manifest files must be an array")
    for index, value in enumerate(files):
        reference = _require_mapping(value, f"receipt-bundle manifest files[{index}]")
        _require_exact_keys(
            reference,
            _RECEIPT_BUNDLE_FILE_REF_KEYS,
            f"receipt-bundle manifest files[{index}]",
        )
    _require_authority_firewall(supplied.get("authority_firewall"), "receipt-bundle manifest authority_firewall")
    expected = build_receipt_bundle_manifest(
        local_protocol_bytes=local_protocol_bytes,
        local_request_bytes=local_request_bytes,
        local_manifest_bytes=local_manifest_bytes,
        normative_file_buffers=normative_file_buffers,
        capture_files=capture_files,
        receipt_bytes=receipt_bytes,
    )
    if supplied != expected or manifest_bytes != canonical_json_bytes(expected):
        raise ValueError("receipt-bundle manifest does not seal the exact replay buffers")
    return expected


_RECEIPT_BUNDLE_KEYS = frozenset(
    {
        "expected_protocol_raw_sha256",
        "local_protocol_bytes",
        "local_request_bytes",
        "local_manifest_bytes",
        "normative_file_buffers",
        "capture_files",
        "receipt_bytes",
        "receipt_bundle_manifest_bytes",
    }
)


def _replay_receipt_bundle(
    value: object,
    label: str,
    *,
    expected_protocol_raw_sha256: str,
) -> tuple[dict[str, object], Mapping[str, object], dict[str, object]]:
    bundle = _require_mapping(value, label)
    _require_exact_keys(bundle, _RECEIPT_BUNDLE_KEYS, label)
    external_pin = _require_lower_hex(
        expected_protocol_raw_sha256,
        64,
        "externally supplied protocol raw SHA-256",
    )
    if bundle["expected_protocol_raw_sha256"] != external_pin:
        raise ValueError(f"{label} attempts to substitute its own protocol trust pin")
    capture_files = _require_mapping(bundle["capture_files"], f"{label}.capture_files")
    receipt = validate_external_anchor_receipt(
        bundle["receipt_bytes"],
        expected_protocol_raw_sha256=external_pin,
        local_protocol_bytes=bundle["local_protocol_bytes"],
        local_request_bytes=bundle["local_request_bytes"],
        local_manifest_bytes=bundle["local_manifest_bytes"],
        normative_file_buffers=_require_mapping(bundle["normative_file_buffers"], f"{label}.normative_file_buffers"),
        capture_files=capture_files,
    )
    manifest = validate_receipt_bundle_manifest(
        bundle["receipt_bundle_manifest_bytes"],
        local_protocol_bytes=bundle["local_protocol_bytes"],
        local_request_bytes=bundle["local_request_bytes"],
        local_manifest_bytes=bundle["local_manifest_bytes"],
        normative_file_buffers=_require_mapping(bundle["normative_file_buffers"], f"{label}.normative_file_buffers"),
        capture_files=capture_files,
        receipt_bytes=bundle["receipt_bytes"],
    )
    return receipt, bundle, manifest


def validate_complete_external_anchor_receipt(
    *,
    expected_protocol_raw_sha256: str,
    bundle: Mapping[str, object],
) -> dict[str, object]:
    """Public complete receipt entry: external trust pin, raw replay, then manifest seal."""

    receipt, _raw_bundle, manifest = _replay_receipt_bundle(
        bundle,
        "external anchor receipt bundle",
        expected_protocol_raw_sha256=expected_protocol_raw_sha256,
    )
    return {"receipt": receipt, "manifest": manifest}


def _git_core(projection: Mapping[str, object]) -> dict[str, object]:
    return {
        key: projection[key]
        for key in (
            "remote_url",
            "advertised_heads",
            "fetched_heads",
            "object_inventory",
            "object_inventory_payload",
            "commit_oid_sha1",
            "commit_framed_sha256",
            "commit_parent_count",
            "tree_oid_sha1",
            "tree_framed_sha256",
            "tree_entry_count",
            "tree_entry_mode",
            "tree_entry_filename",
            "blob_oid_sha1",
            "blob_framed_sha256",
        )
    }


def _admission_verdict(checks: Mapping[str, bool]) -> dict[str, bool]:
    admitted = all(checks.values())
    authority = {key: False for key in sorted(_AUTHORITY_FIREWALL_KEYS)}
    for field in (
        "publication_object_exists_observed",
        "external_publication_observed",
        "external_publication_anchor_present",
        "external_anchor_admitted",
        "A1_contract_and_materializer_implementation_authorized",
    ):
        authority[field] = admitted
    return {
        **authority,
        "external_publication_anchor_present": admitted,
        "external_anchor_admitted": admitted,
        "A1_contract_and_materializer_implementation_authorized": admitted,
        "A1_admission_required_before_materialization": True,
    }


def judge_external_anchor_admission(
    *,
    expected_protocol_raw_sha256: str,
    r0_bundle: Mapping[str, object],
    r1_bundle: Mapping[str, object],
) -> dict[str, object]:
    """Build from two replays; non-authorizing until the composite manifest check."""

    r0, r0_raw, r0_manifest = _replay_receipt_bundle(
        r0_bundle,
        "R0 bundle",
        expected_protocol_raw_sha256=expected_protocol_raw_sha256,
    )
    r1, r1_raw, r1_manifest = _replay_receipt_bundle(
        r1_bundle,
        "R1 bundle",
        expected_protocol_raw_sha256=expected_protocol_raw_sha256,
    )
    r0_capture = _require_mapping(r0["capture_binding"], "R0 capture binding")
    r1_capture = _require_mapping(r1["capture_binding"], "R1 capture binding")
    r0_git = _require_mapping(r0["git_projection"], "R0 Git projection")
    r1_git = _require_mapping(r1["git_projection"], "R1 Git projection")
    r0_verdict = _require_mapping(r0["verdict"], "R0 verdict")
    r1_verdict = _require_mapping(r1["verdict"], "R1 verdict")
    same_local_buffers = all(
        r0_raw[field] == r1_raw[field]
        for field in (
            "expected_protocol_raw_sha256",
            "local_protocol_bytes",
            "local_request_bytes",
            "local_manifest_bytes",
            "normative_file_buffers",
        )
    )
    checks = {
        "r0_full_replay_observed": r0_verdict["observation_complete"] is True,
        "r1_full_replay_observed": r1_verdict["observation_complete"] is True,
        "ordered_roles_valid": r0["observation_role"] == "R0" and r1["observation_role"] == "R1",
        "distinct_process_metadata_claims_and_sealed_lineage_valid": (
            r0["receipt_id"] != r1["receipt_id"]
            and r0_capture["claim_id"] != r1_capture["claim_id"]
            and r0_capture["capture_map_id"] != r1_capture["capture_map_id"]
            and r0_capture["terminal_id"] != r1_capture["terminal_id"]
            and r0_capture["process_id"] != r1_capture["process_id"]
            and r0_capture["process_instance_nonce"] != r1_capture["process_instance_nonce"]
            and r0["predecessor_receipt_id"] is None
            and r0["predecessor_receipt_bundle_manifest_raw_sha256"] is None
            and r1["predecessor_receipt_id"] == r0["receipt_id"]
            and r1["predecessor_receipt_bundle_manifest_raw_sha256"] == _sha256(r0_raw["receipt_bundle_manifest_bytes"])
        ),
        "same_local_buffers": same_local_buffers,
        "same_normative_binding": r0["normative_binding"] == r1["normative_binding"],
        "same_request_binding": r0["request_binding"] == r1["request_binding"],
        "same_anchor_projection": r0["anchor_projection"] == r1["anchor_projection"],
        "same_git_object_graph": _git_core(r0_git) == _git_core(r1_git),
    }
    r0_binding = {
        "receipt_id": r0["receipt_id"],
        "claim_id": r0_capture["claim_id"],
        "capture_map_id": r0_capture["capture_map_id"],
        "capture_map_raw_sha256": r0_capture["capture_map_raw_sha256"],
        "terminal_id": r0_capture["terminal_id"],
        "receipt_bundle_manifest_artifact_id": r0_manifest["artifact_id"],
        "receipt_bundle_manifest_raw_sha256": _sha256(r0_raw["receipt_bundle_manifest_bytes"]),
        "receipt_bundle_manifest_byte_count": len(r0_raw["receipt_bundle_manifest_bytes"]),
    }
    r1_binding = {
        "receipt_id": r1["receipt_id"],
        "claim_id": r1_capture["claim_id"],
        "capture_map_id": r1_capture["capture_map_id"],
        "capture_map_raw_sha256": r1_capture["capture_map_raw_sha256"],
        "terminal_id": r1_capture["terminal_id"],
        "receipt_bundle_manifest_artifact_id": r1_manifest["artifact_id"],
        "receipt_bundle_manifest_raw_sha256": _sha256(r1_raw["receipt_bundle_manifest_bytes"]),
        "receipt_bundle_manifest_byte_count": len(r1_raw["receipt_bundle_manifest_bytes"]),
    }
    core: dict[str, object] = {
        "schema_version": EXTERNAL_ANCHOR_ADMISSION_SCHEMA_VERSION,
        "admission_id_contract": "sha256_canonical_json_utf8_newline_without_admission_id_v1",
        "r0_binding": r0_binding,
        "r1_binding": r1_binding,
        "shared_normative_binding": r0["normative_binding"],
        "shared_request_binding": r0["request_binding"],
        "shared_anchor_projection": r0["anchor_projection"],
        "shared_git_object_graph": _git_core(r0_git),
        "derived_checks": checks,
        "verdict": _admission_verdict(checks),
    }
    admission_id = _sha256(canonical_json_bytes(core))
    return {**core, "admission_id": admission_id}


def validate_external_anchor_admission(
    admission_bytes: bytes,
    *,
    expected_protocol_raw_sha256: str,
    r0_bundle: Mapping[str, object],
    r1_bundle: Mapping[str, object],
) -> dict[str, object]:
    """Validate replay equality; non-authorizing until the composite manifest check."""

    supplied = _canonical_object(admission_bytes, "external anchor admission")
    expected = judge_external_anchor_admission(
        expected_protocol_raw_sha256=expected_protocol_raw_sha256,
        r0_bundle=r0_bundle,
        r1_bundle=r1_bundle,
    )
    if supplied != expected or admission_bytes != canonical_json_bytes(expected):
        raise ValueError("external anchor admission does not equal full R0/R1 replay")
    return expected


def build_admission_bundle_manifest(
    *,
    r0_receipt_bytes: bytes,
    r0_receipt_bundle_manifest_bytes: bytes,
    r1_receipt_bytes: bytes,
    r1_receipt_bundle_manifest_bytes: bytes,
    admission_bytes: bytes,
) -> dict[str, object]:
    """Build an A0 manifest; construction alone is explicitly non-authorizing."""

    admission = _canonical_object(admission_bytes, "external anchor admission for manifest")
    _admission_core, admission_id = _without_id(admission, "admission_id", "external anchor admission")
    receipts: dict[str, dict[str, object]] = {}
    manifests: dict[str, dict[str, object]] = {}
    for role, receipt_bytes, manifest_bytes in (
        ("r0", r0_receipt_bytes, r0_receipt_bundle_manifest_bytes),
        ("r1", r1_receipt_bytes, r1_receipt_bundle_manifest_bytes),
    ):
        receipt = _canonical_object(receipt_bytes, f"{role.upper()} external anchor receipt")
        _receipt_core, receipt_id = _without_id(receipt, "receipt_id", f"{role.upper()} external anchor receipt")
        manifest = _canonical_object(manifest_bytes, f"{role.upper()} receipt-bundle manifest")
        _require_exact_keys(
            manifest,
            _RECEIPT_BUNDLE_MANIFEST_KEYS,
            f"{role.upper()} receipt-bundle manifest",
        )
        _manifest_core, _manifest_id = _without_id(
            manifest,
            "artifact_id",
            f"{role.upper()} receipt-bundle manifest",
        )
        if manifest.get("receipt_id") != receipt_id:
            raise ValueError(f"{role.upper()} bundle manifest does not bind its receipt")
        receipts[role] = receipt
        manifests[role] = manifest
    for role in ("r0", "r1"):
        admission_binding = _require_mapping(admission.get(f"{role}_binding"), f"admission {role.upper()} binding")
        receipt = receipts[role]
        manifest = manifests[role]
        manifest_bytes = r0_receipt_bundle_manifest_bytes if role == "r0" else r1_receipt_bundle_manifest_bytes
        if (
            admission_binding.get("receipt_id") != receipt["receipt_id"]
            or admission_binding.get("receipt_bundle_manifest_artifact_id") != manifest["artifact_id"]
            or admission_binding.get("receipt_bundle_manifest_raw_sha256") != _sha256(manifest_bytes)
            or admission_binding.get("receipt_bundle_manifest_byte_count") != len(manifest_bytes)
        ):
            raise ValueError(f"admission {role.upper()} binding does not close over exact bytes")
    verdict = _require_mapping(admission.get("verdict"), "admission authority verdict")
    expected_verdict_keys = frozenset({*_AUTHORITY_FIREWALL_KEYS, "A1_admission_required_before_materialization"})
    _require_exact_keys(verdict, expected_verdict_keys, "admission authority verdict")
    files = [
        _receipt_bundle_file_ref("R0_receipt", "100_R0_receipt.json", r0_receipt_bytes),
        _receipt_bundle_file_ref(
            "R0_receipt_bundle_manifest",
            "101_R0_receipt_bundle_manifest.json",
            r0_receipt_bundle_manifest_bytes,
        ),
        _receipt_bundle_file_ref("R1_receipt", "110_R1_receipt.json", r1_receipt_bytes),
        _receipt_bundle_file_ref(
            "R1_receipt_bundle_manifest",
            "111_R1_receipt_bundle_manifest.json",
            r1_receipt_bundle_manifest_bytes,
        ),
        _receipt_bundle_file_ref("canonical_admission", "200_external_anchor_admission.json", admission_bytes),
    ]
    core: dict[str, object] = {
        "schema_version": EXTERNAL_ANCHOR_ADMISSION_BUNDLE_MANIFEST_SCHEMA_VERSION,
        "artifact_id_contract": "sha256_canonical_json_utf8_newline_without_artifact_id_v1",
        "admission_id": admission_id,
        "r0_binding": admission["r0_binding"],
        "r1_binding": admission["r1_binding"],
        "authority_verdict": dict(verdict),
        "file_count": len(files),
        "files": files,
    }
    return {**core, "artifact_id": _sha256(canonical_json_bytes(core))}


def validate_admission_bundle_manifest(
    manifest_bytes: bytes,
    *,
    r0_receipt_bytes: bytes,
    r0_receipt_bundle_manifest_bytes: bytes,
    r1_receipt_bytes: bytes,
    r1_receipt_bundle_manifest_bytes: bytes,
    admission_bytes: bytes,
) -> dict[str, object]:
    """Validate manifest byte sealing only; this helper is not an A0 trust decision."""

    supplied = _canonical_object(manifest_bytes, "external anchor admission-bundle manifest")
    _require_exact_keys(
        supplied,
        _ADMISSION_BUNDLE_MANIFEST_KEYS,
        "external anchor admission-bundle manifest",
    )
    _manifest_core, _artifact_id = _without_id(
        supplied,
        "artifact_id",
        "external anchor admission-bundle manifest",
    )
    expected = build_admission_bundle_manifest(
        r0_receipt_bytes=r0_receipt_bytes,
        r0_receipt_bundle_manifest_bytes=r0_receipt_bundle_manifest_bytes,
        r1_receipt_bytes=r1_receipt_bytes,
        r1_receipt_bundle_manifest_bytes=r1_receipt_bundle_manifest_bytes,
        admission_bytes=admission_bytes,
    )
    if supplied != expected or manifest_bytes != canonical_json_bytes(expected):
        raise ValueError("admission-bundle manifest does not seal exact R0/R1/admission bytes")
    return expected


def validate_complete_A0_admission(
    manifest_bytes: bytes,
    admission_bytes: bytes,
    *,
    expected_protocol_raw_sha256: str,
    r0_bundle: Mapping[str, object],
    r1_bundle: Mapping[str, object],
) -> dict[str, object]:
    """Authoritative A0 entry: external pin, two full replays, admission, then manifest."""

    admission = validate_external_anchor_admission(
        admission_bytes,
        expected_protocol_raw_sha256=expected_protocol_raw_sha256,
        r0_bundle=r0_bundle,
        r1_bundle=r1_bundle,
    )
    manifest = validate_admission_bundle_manifest(
        manifest_bytes,
        r0_receipt_bytes=r0_bundle["receipt_bytes"],
        r0_receipt_bundle_manifest_bytes=r0_bundle["receipt_bundle_manifest_bytes"],
        r1_receipt_bytes=r1_bundle["receipt_bytes"],
        r1_receipt_bundle_manifest_bytes=r1_bundle["receipt_bundle_manifest_bytes"],
        admission_bytes=admission_bytes,
    )
    return {"admission": admission, "manifest": manifest}


__all__ = [
    "API_END_BODY_FILE",
    "API_END_HTTP_FILE",
    "API_START_BODY_FILE",
    "API_START_HTTP_FILE",
    "CAPTURE_FILE_NAMES",
    "CAPTURE_MAP_FILE",
    "CLAIM_FILE",
    "EXTERNAL_ANCHOR_ADMISSION_SCHEMA_VERSION",
    "EXTERNAL_ANCHOR_ADMISSION_BUNDLE_MANIFEST_SCHEMA_VERSION",
    "EXTERNAL_ANCHOR_GOLDEN_VECTORS_SCHEMA_VERSION",
    "EXTERNAL_ANCHOR_RECEIPT_SCHEMA_VERSION",
    "EXTERNAL_ANCHOR_RECEIPT_BUNDLE_MANIFEST_SCHEMA_VERSION",
    "GIT_ADVERTISED_REFS_FILE",
    "GIT_BLOB_FILE",
    "GIT_CAPTURE_FILE",
    "GIT_COMMIT_FILE",
    "GIT_FETCHED_REFS_FILE",
    "GIT_OBJECT_INVENTORY_FILE",
    "GIT_TREE_FILE",
    "HTML_BODY_FILE",
    "HTML_HTTP_FILE",
    "ParsedSingleEntryTree",
    "ParsedZeroParentCommit",
    "RAW_BODY_FILE",
    "RAW_HTTP_FILE",
    "REAL_OBSERVER_BACKEND",
    "SYNTHETIC_OBSERVER_BACKEND",
    "TERMINAL_FILE",
    "VerifiedGitClosure",
    "canonical_json_bytes",
    "exact_request_bytes_match",
    "git_object_frame_sha1",
    "git_object_oid_sha1",
    "parse_single_entry_tree",
    "parse_zero_parent_commit",
    "run_golden_vectors",
    "strict_json_loads",
    "strict_json_object_from_bytes",
    "validate_complete_A0_admission",
    "validate_complete_external_anchor_receipt",
    "validate_external_anchor_receipt",
    "verify_receipt_bundle",
    "verify_zero_parent_single_file_git_closure",
]
