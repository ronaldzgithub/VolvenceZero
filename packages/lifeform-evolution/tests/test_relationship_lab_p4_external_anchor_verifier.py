from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import inspect
import json
import ntpath
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

import lifeform_evolution.relationship_lab_p4_external_anchor_verifier as verifier


_OWNER = "ronaldzgithub"
_OWNER_ID = 36_839_548
_OWNER_NODE_ID = "MDQ6VXNlcjM2ODM5NTQ4"
_GIST_ID = "0123456789abcdef0123456789abcdef"
_GIST_NODE_ID = "R2lzdDAxMjM0NTY3ODlhYmNkZWY="
_FILENAME = "volvence_p4_7_source_opportunity_a0_anchor_request.json"
_EXE = r"C:\Program Files\Git\mingw64\libexec\git-core\git.exe"
_HELPERS = [
    [
        r"C:\Program Files\Git\mingw64\libexec\git-core\git-remote-http.exe",
        101,
        "1" * 64,
    ],
    [
        r"C:\Program Files\Git\mingw64\libexec\git-core\git-remote-https.exe",
        102,
        "2" * 64,
    ],
]


@dataclass(frozen=True)
class LocalInputs:
    protocol: bytes
    request: bytes
    manifest: bytes
    normative: dict[str, bytes]
    protocol_raw_sha256: str
    protocol_id: str
    request_id: str
    request_artifact_id: str


def _canonical(value: object) -> bytes:
    return verifier.canonical_json_bytes(value)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _seal(core: dict[str, Any], field: str) -> dict[str, Any]:
    return {**core, field: _sha(_canonical(core))}


def _firewall() -> dict[str, bool]:
    return {key: False for key in sorted(verifier._AUTHORITY_FIREWALL_KEYS)}


def _file_ref(path: str, payload: bytes, role: str | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "path": path,
        "byte_count": len(payload),
        "sha256": _sha(payload),
    }
    return result if role is None else {"role": role, **result}


def _normative_buffers() -> dict[str, bytes]:
    golden = (
        Path(verifier.__file__).parent / "protocols" / "relationship_p4_external_anchor_verifier_golden_vectors_v1.json"
    ).read_bytes()
    return {
        "receipt_schema": b"{}\n",
        "admission_schema": b"{}\n",
        "pure_verifier_and_admission_judge": b"VALUE = 1\n",
        "fixed_production_observer": b"VALUE = 2\n",
        "observer_CLI": b"VALUE = 3\n",
        "golden_vectors": golden,
    }


def _git_toolchain_protocol() -> dict[str, object]:
    return {
        "absolute_path": _EXE,
        "byte_count": 123,
        "raw_sha256": "0" * 64,
        "version_stdout": "git version fixture",
        "transport_helpers": [
            {"absolute_path": path, "byte_count": count, "raw_sha256": digest} for path, count, digest in _HELPERS
        ],
        "must_be_revalidated_from_one_regular_default_stream_buffer_before_use": True,
        "version_is_environment_metadata_but_executable_hash_is_required_for_production_backend": True,
        "transitive_OS_DLL_and_TLS_library_closure_is_fully_pinned": False,
        "cooperative_local_Git_toolchain_trust_remains": True,
    }


def _make_local_inputs() -> LocalInputs:
    normative = _normative_buffers()
    paths = {
        "receipt_schema": "schemas/receipt.json",
        "admission_schema": "schemas/admission.json",
        "pure_verifier_and_admission_judge": "verifier.py",
        "fixed_production_observer": "observer.py",
        "observer_CLI": "observer_cli.py",
        "golden_vectors": "golden.json",
    }
    media = {
        "receipt_schema": "application/schema+json",
        "admission_schema": "application/schema+json",
        "pure_verifier_and_admission_judge": "text/x-python",
        "fixed_production_observer": "text/x-python",
        "observer_CLI": "text/x-python",
        "golden_vectors": "application/json",
    }
    descriptors = [
        {
            "role": role,
            "repo_relative_posix_path": paths[role],
            "media_type": media[role],
            "byte_count": len(normative[role]),
            "raw_sha256": _sha(normative[role]),
            "git_blob_oid_sha1": verifier.git_object_oid_sha1("blob", normative[role]),
            "expected_eol": "LF",
            "same_buffer_operation": verifier._NORMATIVE_SAME_BUFFER_OPERATIONS[role],
        }
        for role in verifier._REQUIRED_NORMATIVE_ROLES
    ]
    target = {
        "expected_owner": {
            "login": _OWNER,
            "id": _OWNER_ID,
            "node_id": _OWNER_NODE_ID,
            "type": "User",
            "site_admin": False,
        },
        "role_hosts": {
            "api": "api.github.com",
            "raw": "gist.githubusercontent.com",
            "html_and_git": "gist.github.com",
            "owner_profile_field_only_not_fetched_by_observer": "github.com",
        },
        "required_visibility": "public",
        "required_filename": _FILENAME,
        "required_description": "",
        "required_exact_file_count": 1,
        "required_exact_history_count": 1,
        "required_exact_advertised_head_count": 1,
    }
    verification = {"normative_files": descriptors, "normative_file_count": 6}
    acquisition = {"production_Windows_Git_executable": _git_toolchain_protocol()}
    protocol_object = {
        "protocol_id_contract": "sha256_canonical_json_utf8_newline_v1",
        "publication_target_contract": target,
        "verification_preregistration": verification,
        "observer_acquisition_contract": acquisition,
        "git_object_closure_contract": {"exact_three_objects": True},
        "receipt_contract": {"schema_version": verifier.EXTERNAL_ANCHOR_RECEIPT_SCHEMA_VERSION},
        "admission_contract": {"schema_version": verifier.EXTERNAL_ANCHOR_ADMISSION_SCHEMA_VERSION},
    }
    protocol = _canonical(protocol_object)
    protocol_id = _sha(_canonical(protocol_object))
    protocol_raw = _sha(protocol)
    request_core = {
        "identity": {
            "anchor_request_protocol_id": protocol_id,
            "anchor_request_protocol_raw_sha256": protocol_raw,
        },
        "publication_target": target,
        "verification_preregistration": verification,
        "observer_acquisition_contract": acquisition,
        "git_object_closure_contract": protocol_object["git_object_closure_contract"],
        "receipt_contract": protocol_object["receipt_contract"],
        "admission_contract": protocol_object["admission_contract"],
    }
    request_object = _seal(request_core, "request_id")
    request = _canonical(request_object)
    manifest_core = {
        "anchor_request_protocol_id": protocol_id,
        "request_id": request_object["request_id"],
        "files": [
            {
                "path": "external_publication_anchor_request.json",
                "byte_count": len(request),
                "sha256": _sha(request),
                "git_blob_oid_sha1": verifier.git_object_oid_sha1("blob", request),
            }
        ],
    }
    manifest_object = _seal(manifest_core, "artifact_id")
    return LocalInputs(
        protocol=protocol,
        request=request,
        manifest=_canonical(manifest_object),
        normative=normative,
        protocol_raw_sha256=protocol_raw,
        protocol_id=protocol_id,
        request_id=request_object["request_id"],
        request_artifact_id=manifest_object["artifact_id"],
    )


def _header_facts(pairs: list[list[str]]) -> tuple[int, int, int, str]:
    normalized = [[name.casefold(), value] for name, value in pairs]
    ledger_bytes = sum(len(f"{name}: {value}\r\n".encode("latin-1")) for name, value in pairs)
    return len(pairs), ledger_bytes, ledger_bytes, _sha(_canonical(normalized))


def _http_metadata(role: str, url: str, body_name: str, body: bytes) -> bytes:
    content_type = {"api": "application/json", "html_git": "text/html"}.get(role)
    pairs = [["Content-Length", str(len(body))]]
    if content_type is not None:
        pairs.append(["Content-Type", content_type])
    count, wire_bytes, ledger_bytes, ledger_hash = _header_facts(pairs)
    by_name = {name.casefold(): [value] for name, value in pairs}
    redirect_limit = 0 if role == "api" else 3
    core = {
        "schema_version": "relationship-p4-external-anchor-observer-http.v1",
        "role": role,
        "method": "GET",
        "requested_url": url,
        "final_url": url,
        "status": 200,
        "request_headers": dict(verifier._REQUEST_HEADERS),
        "effective_request_headers": [
            ["Host", verifier._ROLE_HOSTS[role]],
            *[[key, value] for key, value in verifier._REQUEST_HEADERS.items()],
        ],
        "authorization_header_sent": False,
        "cookie_header_sent": False,
        "proxy_used": False,
        "netrc_used": False,
        "response_header_count": count,
        "response_header_wire_bytes": wire_bytes,
        "response_header_ledger_bytes": ledger_bytes,
        "response_header_ledger_sha256": ledger_hash,
        "response_header_pairs": pairs,
        "set_cookie_present": False,
        "set_cookie_count": 0,
        "set_cookie_values_serialized": False,
        "set_cookie_redaction_facts": [],
        "response_framing": {
            "content_length_values": by_name.get("content-length", []),
            "transfer_encoding_values": [],
            "content_encoding_values": [],
            "content_type_values": by_name.get("content-type", []),
            "duplicate_content_length_rejected": True,
            "duplicate_transfer_encoding_rejected": True,
            "duplicate_content_encoding_rejected": True,
            "transfer_encoding_and_content_length_coexistence_rejected": True,
            "content_encoding_allowed": ["absent", "identity"],
            "declared_content_length_must_equal_captured_body": True,
        },
        "redirects": [],
        "body": {
            **_file_ref(body_name, body),
            "body_cap": {
                "api": 262_144,
                "raw": len(body),
                "html_git": 2_097_152,
            }[role],
        },
        "role_redirect_max_hops": redirect_limit,
        "connect_timeout_seconds": 10,
        "read_idle_timeout_seconds": 10,
        "request_total_timeout_seconds": 30,
        "retry_count": 0,
        "facts_only_no_verdict": True,
    }
    return _canonical(core)


def _fixed_git_commands(remote_url: str) -> tuple[list[list[str]], list[list[str]]]:
    root = r"C:\fixture\.observer-git-0001"
    hooks = ntpath.join(root, "disabled-hooks")
    bare = ntpath.join(root, "repository.git")
    template = ntpath.join(root, "empty-template")
    fixed = ["--no-optional-locks"]
    for value in verifier._GIT_REQUIRED_CONFIG_ARGUMENTS[:-2]:
        fixed.extend(["-c", value])
    fixed.extend(["-c", f"core.hooksPath={hooks}"])
    for value in verifier._GIT_REQUIRED_CONFIG_ARGUMENTS[-2:]:
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
        ["-C", bare, "for-each-ref", "--format=%(refname)%00%(objectname)", "refs/remotes/origin"],
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
    commands = [[_EXE, *fixed, *tail] for tail in tails]
    environment = {
        **verifier._GIT_REQUIRED_ENVIRONMENT,
        "TMP": ntpath.join(root, "io"),
        "TEMP": ntpath.join(root, "io"),
        "TMPDIR": ntpath.join(root, "io"),
        "GIT_CONFIG_GLOBAL": ntpath.join(root, "empty-global-config"),
        "XDG_CONFIG_HOME": ntpath.join(root, "empty-xdg-config"),
        "GIT_EXEC_PATH": ntpath.dirname(_HELPERS[0][0]),
        "SYSTEMROOT": r"C:\Windows",
        "WINDIR": r"C:\Windows",
    }
    return commands, [[key, value] for key, value in sorted(environment.items())]


def _api_body(request: bytes, revision: str, raw_token: str | None = None) -> bytes:
    token = revision if raw_token is None else raw_token
    raw_url = f"https://gist.githubusercontent.com/{_OWNER}/{_GIST_ID}/raw/{token}/{_FILENAME}"
    return _canonical(
        {
            "id": _GIST_ID,
            "node_id": _GIST_NODE_ID,
            "public": True,
            "description": "",
            "truncated": False,
            "owner": {
                "login": _OWNER,
                "id": _OWNER_ID,
                "node_id": _OWNER_NODE_ID,
                "type": "User",
                "site_admin": False,
            },
            "files": {
                _FILENAME: {
                    "filename": _FILENAME,
                    "size": len(request),
                    "truncated": False,
                    "encoding": "utf-8",
                    "content": request.decode("utf-8"),
                    "raw_url": raw_url,
                }
            },
            "history": [{"version": revision}],
            "html_url": f"https://gist.github.com/{_GIST_ID}",
            "git_pull_url": f"https://gist.github.com/{_GIST_ID}.git",
        }
    )


def _reseal_capture(capture: dict[str, bytes]) -> None:
    refs = [_file_ref(name, capture[name], verifier._FILE_ROLES[name]) for name in verifier._MAPPED_CAPTURE_FILES]
    old_map = json.loads(capture[verifier.CAPTURE_MAP_FILE])
    map_core = {key: value for key, value in old_map.items() if key != "capture_map_id"}
    map_core["files"] = refs
    capture_map = _seal(map_core, "capture_map_id")
    capture[verifier.CAPTURE_MAP_FILE] = _canonical(capture_map)
    old_terminal = json.loads(capture[verifier.TERMINAL_FILE])
    terminal_core = {key: value for key, value in old_terminal.items() if key != "terminal_id"}
    terminal_core["capture_map_id"] = capture_map["capture_map_id"]
    terminal_core["capture_map_raw_sha256"] = _sha(capture[verifier.CAPTURE_MAP_FILE])
    capture[verifier.TERMINAL_FILE] = _canonical(_seal(terminal_core, "terminal_id"))


def _make_capture(
    local: LocalInputs,
    *,
    stage: str,
    process_id: int,
    nonce_digit: str,
    predecessor_receipt_id: str | None,
    predecessor_manifest_sha256: str | None,
    backend: str = verifier.REAL_OBSERVER_BACKEND,
) -> dict[str, bytes]:
    blob = local.request
    blob_oid = verifier.git_object_oid_sha1("blob", blob)
    tree = b"100644 " + _FILENAME.encode() + b"\0" + bytes.fromhex(blob_oid)
    tree_oid = verifier.git_object_oid_sha1("tree", tree)
    commit = (
        f"tree {tree_oid}\n"
        "author Fixture <fixture@example.invalid> 1700000000 +0000\n"
        "committer Fixture <fixture@example.invalid> 1700000000 +0000\n"
        "\nA0 fixture root commit\n"
    ).encode()
    revision = verifier.git_object_oid_sha1("commit", commit)
    api_url = f"https://api.github.com/gists/{_GIST_ID}/{revision}"
    api_body = _api_body(local.request, revision)
    api_object = json.loads(api_body)
    raw_url = api_object["files"][_FILENAME]["raw_url"]
    html_url = api_object["html_url"]
    git_url = api_object["git_pull_url"]
    html_body = b"<!doctype html><title>public gist</title>"
    advertised = f"{revision}\trefs/heads/main\n".encode()
    fetched = f"refs/remotes/origin/main\0{revision}\n".encode()
    inventory_rows = sorted(
        [
            (revision, "commit", len(commit)),
            (tree_oid, "tree", len(tree)),
            (blob_oid, "blob", len(blob)),
        ]
    )
    inventory = b"".join(f"{oid}\0{kind}\0{size}\n".encode() for oid, kind, size in inventory_rows)
    commands, environment = _fixed_git_commands(git_url)
    production = backend == verifier.REAL_OBSERVER_BACKEND
    toolchain = {
        "executable_path": _EXE,
        "executable_byte_count": 123,
        "executable_raw_sha256": "0" * 64,
        "version_stdout": "git version fixture",
        "helper_identities": _HELPERS,
        "preflight_completed_before_HTTP": production,
    }
    git_meta = {
        "schema_version": "relationship-p4-external-anchor-observer-git.v1",
        "remote_url": git_url,
        "revision_oid": revision,
        "commit_oid": revision,
        "tree_oid": tree_oid,
        "blob_oid": blob_oid,
        "tree_entry_mode": "100644",
        "tree_entry_name": _FILENAME,
        "advertised_refs": [["refs/heads/main", revision]],
        "fetched_refs": [["refs/remotes/origin/main", revision]],
        "object_inventory": [list(item) for item in inventory_rows],
        "advertised_refs_raw_stdout": _file_ref(verifier.GIT_ADVERTISED_REFS_FILE, advertised),
        "fetched_refs_raw_stdout": _file_ref(verifier.GIT_FETCHED_REFS_FILE, fetched),
        "object_inventory_raw_stdout": _file_ref(verifier.GIT_OBJECT_INVENTORY_FILE, inventory),
        "object_store_byte_count": len(commit) + len(tree) + len(blob),
        "commit_body": _file_ref(verifier.GIT_COMMIT_FILE, commit),
        "tree_body": _file_ref(verifier.GIT_TREE_FILE, tree),
        "blob_body": _file_ref(verifier.GIT_BLOB_FILE, blob),
        "fsck_stdout_sha256": _sha(b""),
        "fsck_stderr_sha256": _sha(b""),
        "production_git_toolchain": toolchain,
        "command_argv_ledger": commands,
        "environment_ledger": environment,
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
    claim_target = {
        "observation_stage": stage,
        "predecessor_receipt_id": predecessor_receipt_id,
        "predecessor_receipt_bundle_manifest_raw_sha256": predecessor_manifest_sha256,
        "protocol_id": local.protocol_id,
        "protocol_raw_sha256": local.protocol_raw_sha256,
        "protocol_raw_byte_count": len(local.protocol),
        "request_id": local.request_id,
        "request_artifact_id": local.request_artifact_id,
        "request_raw_sha256": _sha(local.request),
        "request_raw_byte_count": len(local.request),
        "request_manifest_raw_sha256": _sha(local.manifest),
        "request_manifest_raw_byte_count": len(local.manifest),
        "gist_id": _GIST_ID,
        "revision_oid": revision,
        "expected_owner_login": _OWNER,
        "expected_owner_id": _OWNER_ID,
        "expected_owner_node_id": _OWNER_NODE_ID,
        "required_filename": _FILENAME,
        "local_protocol_request_manifest_buffers_recomputed_by_observer": False,
        "local_buffer_recomputation_owner": "separate_pinned_verifier",
    }
    claim_core = {
        "schema_version": "relationship-p4-external-anchor-observer-claim.v1",
        "backend_kind": backend,
        "target": claim_target,
        "fixed_acquisition_contract": {
            "method": "GET",
            "github_api_version": "2026-03-10",
            "request_headers": dict(verifier._REQUEST_HEADERS),
            "role_hosts": dict(verifier._ROLE_HOSTS),
            "success_status": 200,
            "retry_count": 0,
            "identity_free": True,
            "proxy_netrc_auth_cookie_forbidden": True,
            "facts_only_no_verdict": True,
        },
        "authority_firewall": _firewall(),
        "A1_required_before_materialization": True,
        "process_id": process_id,
        "process_instance_nonce": nonce_digit * 64,
        "claim_boundary": "facts only",
    }
    claim = _seal(claim_core, "claim_id")
    capture: dict[str, bytes] = {
        verifier.CLAIM_FILE: _canonical(claim),
        verifier.API_START_HTTP_FILE: _http_metadata("api", api_url, verifier.API_START_BODY_FILE, api_body),
        verifier.API_START_BODY_FILE: api_body,
        verifier.RAW_HTTP_FILE: _http_metadata("raw", raw_url, verifier.RAW_BODY_FILE, local.request),
        verifier.RAW_BODY_FILE: local.request,
        verifier.HTML_HTTP_FILE: _http_metadata("html_git", html_url, verifier.HTML_BODY_FILE, html_body),
        verifier.HTML_BODY_FILE: html_body,
        verifier.GIT_CAPTURE_FILE: _canonical(git_meta),
        verifier.GIT_ADVERTISED_REFS_FILE: advertised,
        verifier.GIT_FETCHED_REFS_FILE: fetched,
        verifier.GIT_OBJECT_INVENTORY_FILE: inventory,
        verifier.GIT_COMMIT_FILE: commit,
        verifier.GIT_TREE_FILE: tree,
        verifier.GIT_BLOB_FILE: blob,
        verifier.API_END_HTTP_FILE: _http_metadata("api", api_url, verifier.API_END_BODY_FILE, api_body),
        verifier.API_END_BODY_FILE: api_body,
    }
    refs = [_file_ref(name, capture[name], verifier._FILE_ROLES[name]) for name in verifier._MAPPED_CAPTURE_FILES]
    map_core = {
        "schema_version": "relationship-p4-external-anchor-observer-capture-map.v1",
        "claim_id": claim["claim_id"],
        "backend_kind": backend,
        "observation_stage": stage,
        "capture_sequence": list(verifier._CAPTURE_SEQUENCE),
        "completed_stages": list(verifier._CAPTURE_SEQUENCE),
        "files": refs,
        "expected_pre_map_files": list(verifier._MAPPED_CAPTURE_FILES),
        "actual_pre_map_files": list(verifier._MAPPED_CAPTURE_FILES),
        "missing_pre_map_files": [],
        "unexpected_pre_map_files": [],
        "root_anomalies": [],
        "root_closure_status": "complete_exact_pre_map_root",
        "acquisition_complete": True,
        "failure_code": None,
        "retry_count": 0,
        "root_anomaly_count": 0,
        "authority_firewall": _firewall(),
        "A1_required_before_materialization": True,
    }
    capture_map = _seal(map_core, "capture_map_id")
    capture[verifier.CAPTURE_MAP_FILE] = _canonical(capture_map)
    terminal_core = {
        "schema_version": "relationship-p4-external-anchor-observer-terminal.v1",
        "claim_id": claim["claim_id"],
        "capture_map_id": capture_map["capture_map_id"],
        "capture_map_raw_sha256": _sha(capture[verifier.CAPTURE_MAP_FILE]),
        "backend_kind": backend,
        "observation_stage": stage,
        "status": "facts_only_observation_complete_non_authorizing",
        "acquisition_complete": True,
        "failure": None,
        "retry_count": 0,
        "root_closure_status": "complete_exact_pre_map_root",
        "root_anomaly_count": 0,
        "authority_firewall": _firewall(),
        "A1_required_before_materialization": True,
        "claim_boundary": "facts only",
    }
    capture[verifier.TERMINAL_FILE] = _canonical(_seal(terminal_core, "terminal_id"))
    return capture


def _verify(local: LocalInputs, capture: dict[str, bytes]) -> dict[str, object]:
    return verifier.verify_receipt_bundle(
        expected_protocol_raw_sha256=local.protocol_raw_sha256,
        local_protocol_bytes=local.protocol,
        local_request_bytes=local.request,
        local_manifest_bytes=local.manifest,
        normative_file_buffers=local.normative,
        capture_files=capture,
    )


def _bundle(local: LocalInputs, capture: dict[str, bytes]) -> dict[str, object]:
    receipt = _verify(local, capture)
    receipt_bytes = _canonical(receipt)
    manifest = verifier.build_receipt_bundle_manifest(
        local_protocol_bytes=local.protocol,
        local_request_bytes=local.request,
        local_manifest_bytes=local.manifest,
        normative_file_buffers=local.normative,
        capture_files=capture,
        receipt_bytes=receipt_bytes,
    )
    return {
        "expected_protocol_raw_sha256": local.protocol_raw_sha256,
        "local_protocol_bytes": local.protocol,
        "local_request_bytes": local.request,
        "local_manifest_bytes": local.manifest,
        "normative_file_buffers": local.normative,
        "capture_files": capture,
        "receipt_bytes": receipt_bytes,
        "receipt_bundle_manifest_bytes": _canonical(manifest),
    }


def _r0_r1() -> tuple[LocalInputs, dict[str, object], dict[str, object]]:
    local = _make_local_inputs()
    r0 = _bundle(
        local,
        _make_capture(
            local,
            stage="R0",
            process_id=100,
            nonce_digit="a",
            predecessor_receipt_id=None,
            predecessor_manifest_sha256=None,
        ),
    )
    r0_receipt = json.loads(r0["receipt_bytes"])
    r1 = _bundle(
        local,
        _make_capture(
            local,
            stage="R1",
            process_id=101,
            nonce_digit="b",
            predecessor_receipt_id=r0_receipt["receipt_id"],
            predecessor_manifest_sha256=_sha(r0["receipt_bundle_manifest_bytes"]),
        ),
    )
    return local, r0, r1


def test_golden_vectors_execute_from_the_exact_buffer() -> None:
    golden = _normative_buffers()["golden_vectors"]
    result = verifier.run_golden_vectors(golden)
    assert result["all_vectors_passed"] is True
    assert result["case_count"] == 23
    drifted = golden.replace(b'"blob_frame_positive"', b'"blob_frame_drifted"', 1)
    result = verifier.run_golden_vectors(drifted)
    assert result["executed_case_ids"][0] == "blob_frame_drifted"


@pytest.mark.parametrize(
    "payload",
    [b"\xef\xbb\xbf{}", b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":"\\u0000"}', b"[]"],
)
def test_strict_json_rejects_bom_duplicate_nonfinite_and_nonobject(payload: bytes) -> None:
    with pytest.raises(ValueError):
        verifier.strict_json_object_from_bytes(payload)


def test_git_framing_zero_parent_tree_and_exact_request_bytes() -> None:
    local = _make_local_inputs()
    capture = _make_capture(
        local,
        stage="R0",
        process_id=1,
        nonce_digit="a",
        predecessor_receipt_id=None,
        predecessor_manifest_sha256=None,
    )
    closure = verifier.verify_zero_parent_single_file_git_closure(
        commit_payload=capture[verifier.GIT_COMMIT_FILE],
        tree_payload=capture[verifier.GIT_TREE_FILE],
        blob_payload=capture[verifier.GIT_BLOB_FILE],
        required_filename=_FILENAME,
    )
    assert closure.blob_oid_sha1 == verifier.git_object_oid_sha1("blob", local.request)
    with pytest.raises(ValueError, match="zero parent"):
        verifier.parse_zero_parent_commit(
            capture[verifier.GIT_COMMIT_FILE].replace(b"author ", b"parent " + b"0" * 40 + b"\nauthor ", 1)
        )
    with pytest.raises(ValueError, match="exactly one"):
        verifier.parse_single_entry_tree(capture[verifier.GIT_TREE_FILE] + capture[verifier.GIT_TREE_FILE])


def test_full_real_receipt_replay_is_observation_only() -> None:
    local = _make_local_inputs()
    receipt = _verify(
        local,
        _make_capture(
            local,
            stage="R0",
            process_id=100,
            nonce_digit="a",
            predecessor_receipt_id=None,
            predecessor_manifest_sha256=None,
        ),
    )
    assert receipt["derived_checks"] == {key: True for key in "LHTJGBE"}
    assert receipt["verdict"]["integrity_valid"] is True
    assert receipt["verdict"]["observation_complete"] is True
    assert all(receipt["verdict"][key] is False for key in verifier._AUTHORITY_FIREWALL_KEYS)


def test_synthetic_can_have_integrity_but_never_observation_or_authority() -> None:
    local = _make_local_inputs()
    receipt = _verify(
        local,
        _make_capture(
            local,
            stage="R0",
            process_id=100,
            nonce_digit="a",
            predecessor_receipt_id=None,
            predecessor_manifest_sha256=None,
            backend=verifier.SYNTHETIC_OBSERVER_BACKEND,
        ),
    )
    assert receipt["verdict"]["integrity_valid"] is True
    assert receipt["derived_checks"]["E"] is False
    assert receipt["verdict"]["observation_complete"] is False


def test_normative_buffers_are_exact_six_same_buffers_and_golden_is_executed() -> None:
    local = _make_local_inputs()
    capture = _make_capture(
        local,
        stage="R0",
        process_id=100,
        nonce_digit="a",
        predecessor_receipt_id=None,
        predecessor_manifest_sha256=None,
    )
    drifted = dict(local.normative)
    drifted["observer_CLI"] += b"# drift\n"
    with pytest.raises(ValueError, match="bytes drift"):
        verifier.verify_receipt_bundle(
            expected_protocol_raw_sha256=local.protocol_raw_sha256,
            local_protocol_bytes=local.protocol,
            local_request_bytes=local.request,
            local_manifest_bytes=local.manifest,
            normative_file_buffers=drifted,
            capture_files=capture,
        )
    extra = dict(local.normative)
    extra["unregistered"] = b"{}\n"
    with pytest.raises(ValueError, match="roles do not match"):
        verifier.verify_receipt_bundle(
            expected_protocol_raw_sha256=local.protocol_raw_sha256,
            local_protocol_bytes=local.protocol,
            local_request_bytes=local.request,
            local_manifest_bytes=local.manifest,
            normative_file_buffers=extra,
            capture_files=capture,
        )


def test_capture_orphan_extra_and_inventory_tamper_fail_closed() -> None:
    local = _make_local_inputs()
    capture = _make_capture(
        local,
        stage="R0",
        process_id=100,
        nonce_digit="a",
        predecessor_receipt_id=None,
        predecessor_manifest_sha256=None,
    )
    extra = dict(capture)
    extra["orphan.bin"] = b"orphan"
    with pytest.raises(ValueError, match="exact bundle inventory"):
        _verify(local, extra)
    inventory_tamper = copy.deepcopy(capture)
    inventory_tamper[verifier.GIT_OBJECT_INVENTORY_FILE] += b"0" * 40 + b"\0blob\0" + b"1\n"
    git_meta = json.loads(inventory_tamper[verifier.GIT_CAPTURE_FILE])
    git_meta["object_inventory_raw_stdout"] = _file_ref(
        verifier.GIT_OBJECT_INVENTORY_FILE,
        inventory_tamper[verifier.GIT_OBJECT_INVENTORY_FILE],
    )
    inventory_tamper[verifier.GIT_CAPTURE_FILE] = _canonical(git_meta)
    _reseal_capture(inventory_tamper)
    with pytest.raises(ValueError, match="exactly three"):
        _verify(local, inventory_tamper)


def test_http_wire_ledger_and_git_argv_environment_tamper_are_no_go() -> None:
    local = _make_local_inputs()
    capture = _make_capture(
        local,
        stage="R0",
        process_id=100,
        nonce_digit="a",
        predecessor_receipt_id=None,
        predecessor_manifest_sha256=None,
    )
    wire_tamper = copy.deepcopy(capture)
    raw_meta = json.loads(wire_tamper[verifier.RAW_HTTP_FILE])
    raw_meta["response_header_wire_bytes"] += 1
    wire_tamper[verifier.RAW_HTTP_FILE] = _canonical(raw_meta)
    _reseal_capture(wire_tamper)
    assert _verify(local, wire_tamper)["derived_checks"]["H"] is False

    git_tamper = copy.deepcopy(capture)
    git_meta = json.loads(git_tamper[verifier.GIT_CAPTURE_FILE])
    git_meta["command_argv_ledger"][0].extend(["-c", "http.followRedirects=true"])
    git_meta["environment_ledger"].append(["HTTPS_PROXY", "https://proxy.invalid/"])
    git_tamper[verifier.GIT_CAPTURE_FILE] = _canonical(git_meta)
    _reseal_capture(git_tamper)
    assert _verify(local, git_tamper)["derived_checks"]["G"] is False


def test_cross_gist_returned_url_and_fake_all_true_receipt_cannot_authorize() -> None:
    local = _make_local_inputs()
    capture = _make_capture(
        local,
        stage="R0",
        process_id=100,
        nonce_digit="a",
        predecessor_receipt_id=None,
        predecessor_manifest_sha256=None,
    )
    api = json.loads(capture[verifier.API_START_BODY_FILE])
    api["files"][_FILENAME]["raw_url"] = api["files"][_FILENAME]["raw_url"].replace(_GIST_ID, "f" * len(_GIST_ID))
    capture[verifier.API_START_BODY_FILE] = _canonical(api)
    http_meta = json.loads(capture[verifier.API_START_HTTP_FILE])
    http_meta["body"] = {
        **_file_ref(verifier.API_START_BODY_FILE, capture[verifier.API_START_BODY_FILE]),
        "body_cap": 262_144,
    }
    pairs = http_meta["response_header_pairs"]
    pairs[0][1] = str(len(capture[verifier.API_START_BODY_FILE]))
    count, wire, ledger, digest = _header_facts(pairs)
    http_meta.update(
        response_header_count=count,
        response_header_wire_bytes=wire,
        response_header_ledger_bytes=ledger,
        response_header_ledger_sha256=digest,
    )
    http_meta["response_framing"]["content_length_values"] = [str(len(capture[verifier.API_START_BODY_FILE]))]
    capture[verifier.API_START_HTTP_FILE] = _canonical(http_meta)
    _reseal_capture(capture)
    replay = _verify(local, capture)
    assert replay["derived_checks"]["T"] is False
    forged = copy.deepcopy(replay)
    forged["derived_checks"] = {key: True for key in "LHTJGBE"}
    forged["verdict"]["integrity_valid"] = True
    forged["verdict"]["observation_complete"] = True
    forged_core = {key: value for key, value in forged.items() if key != "receipt_id"}
    forged["receipt_id"] = _sha(_canonical(forged_core))
    with pytest.raises(ValueError, match="full raw-bundle replay"):
        verifier.validate_external_anchor_receipt(
            _canonical(forged),
            expected_protocol_raw_sha256=local.protocol_raw_sha256,
            local_protocol_bytes=local.protocol,
            local_request_bytes=local.request,
            local_manifest_bytes=local.manifest,
            normative_file_buffers=local.normative,
            capture_files=capture,
        )


def test_r0_r1_full_replay_admission_and_both_create_only_manifests() -> None:
    _local, r0, r1 = _r0_r1()
    complete_receipt = verifier.validate_complete_external_anchor_receipt(
        expected_protocol_raw_sha256=_local.protocol_raw_sha256,
        bundle=r0,
    )
    assert complete_receipt["receipt"]["observation_role"] == "R0"
    admission = verifier.judge_external_anchor_admission(
        expected_protocol_raw_sha256=_local.protocol_raw_sha256,
        r0_bundle=r0,
        r1_bundle=r1,
    )
    assert all(admission["derived_checks"].values())
    go_fields = {
        "publication_object_exists_observed",
        "external_publication_observed",
        "external_publication_anchor_present",
        "external_anchor_admitted",
        "A1_contract_and_materializer_implementation_authorized",
    }
    assert {key for key, value in admission["verdict"].items() if value is True} == {
        *go_fields,
        "A1_admission_required_before_materialization",
    }
    admission_bytes = _canonical(admission)
    manifest = verifier.build_admission_bundle_manifest(
        r0_receipt_bytes=r0["receipt_bytes"],
        r0_receipt_bundle_manifest_bytes=r0["receipt_bundle_manifest_bytes"],
        r1_receipt_bytes=r1["receipt_bytes"],
        r1_receipt_bundle_manifest_bytes=r1["receipt_bundle_manifest_bytes"],
        admission_bytes=admission_bytes,
    )
    validated = verifier.validate_admission_bundle_manifest(
        _canonical(manifest),
        r0_receipt_bytes=r0["receipt_bytes"],
        r0_receipt_bundle_manifest_bytes=r0["receipt_bundle_manifest_bytes"],
        r1_receipt_bytes=r1["receipt_bytes"],
        r1_receipt_bundle_manifest_bytes=r1["receipt_bundle_manifest_bytes"],
        admission_bytes=admission_bytes,
    )
    assert validated["artifact_id"] == manifest["artifact_id"]
    complete = verifier.validate_complete_A0_admission(
        _canonical(manifest),
        admission_bytes,
        expected_protocol_raw_sha256=_local.protocol_raw_sha256,
        r0_bundle=r0,
        r1_bundle=r1,
    )
    assert complete["admission"]["admission_id"] == admission["admission_id"]

    with pytest.raises(ValueError, match="substitute its own protocol trust pin"):
        verifier.validate_complete_A0_admission(
            _canonical(manifest),
            admission_bytes,
            expected_protocol_raw_sha256="f" * 64,
            r0_bundle=r0,
            r1_bundle=r1,
        )

    forged = copy.deepcopy(admission)
    forged["verdict"]["source_execution_authorized"] = True
    forged_core = {key: value for key, value in forged.items() if key != "admission_id"}
    forged["admission_id"] = _sha(_canonical(forged_core))
    forged_bytes = _canonical(forged)
    forged_manifest = verifier.build_admission_bundle_manifest(
        r0_receipt_bytes=r0["receipt_bytes"],
        r0_receipt_bundle_manifest_bytes=r0["receipt_bundle_manifest_bytes"],
        r1_receipt_bytes=r1["receipt_bytes"],
        r1_receipt_bundle_manifest_bytes=r1["receipt_bundle_manifest_bytes"],
        admission_bytes=forged_bytes,
    )
    with pytest.raises(ValueError, match="full R0/R1 replay"):
        verifier.validate_complete_A0_admission(
            _canonical(forged_manifest),
            forged_bytes,
            expected_protocol_raw_sha256=_local.protocol_raw_sha256,
            r0_bundle=r0,
            r1_bundle=r1,
        )


def test_admission_requires_exact_r0_bundle_manifest_lineage_and_distinct_metadata() -> None:
    local, r0, _r1 = _r0_r1()
    r0_receipt = json.loads(r0["receipt_bytes"])
    wrong_r1 = _bundle(
        local,
        _make_capture(
            local,
            stage="R1",
            process_id=100,
            nonce_digit="a",
            predecessor_receipt_id=r0_receipt["receipt_id"],
            predecessor_manifest_sha256="f" * 64,
        ),
    )
    admission = verifier.judge_external_anchor_admission(
        expected_protocol_raw_sha256=local.protocol_raw_sha256,
        r0_bundle=r0,
        r1_bundle=wrong_r1,
    )
    assert admission["derived_checks"]["distinct_process_metadata_claims_and_sealed_lineage_valid"] is False
    assert admission["verdict"]["external_anchor_admitted"] is False


def test_receipt_bundle_manifest_seals_flat_exact_files_and_rejects_tamper() -> None:
    _local, r0, _r1 = _r0_r1()
    manifest = json.loads(r0["receipt_bundle_manifest_bytes"])
    names = [item["name"] for item in manifest["files"]]
    assert names[:9] == [
        "100_local_protocol.json",
        "101_local_request.json",
        "102_local_request_manifest.json",
        "110_receipt_schema.json",
        "111_admission_schema.json",
        "112_verifier.py",
        "113_observer.py",
        "114_observer_cli.py",
        "115_golden_vectors.json",
    ]
    tampered = copy.deepcopy(r0)
    raw_manifest = bytearray(tampered["receipt_bundle_manifest_bytes"])
    raw_manifest[-2] = ord(" ")
    tampered["receipt_bundle_manifest_bytes"] = bytes(raw_manifest)
    with pytest.raises(ValueError):
        verifier.judge_external_anchor_admission(
            expected_protocol_raw_sha256=tampered["expected_protocol_raw_sha256"],
            r0_bundle=tampered,
            r1_bundle=r0,
        )


def test_public_apis_accept_raw_buffers_not_authority_summaries() -> None:
    verify_parameters = inspect.signature(verifier.verify_receipt_bundle).parameters
    assert set(verify_parameters) == {
        "expected_protocol_raw_sha256",
        "local_protocol_bytes",
        "local_request_bytes",
        "local_manifest_bytes",
        "normative_file_buffers",
        "capture_files",
    }
    assert "capture_facts" not in inspect.getsource(verifier.verify_receipt_bundle)
    assert "checks" not in inspect.signature(verifier.judge_external_anchor_admission).parameters
    assert "validate_complete_A0_admission" in verifier.__all__
    assert "validate_complete_external_anchor_receipt" in verifier.__all__
    assert "judge_external_anchor_admission" not in verifier.__all__
    assert "validate_admission_bundle_manifest" not in verifier.__all__


def test_frozen_schemas_have_exact_top_level_and_verdict_key_sets() -> None:
    local, r0, r1 = _r0_r1()
    receipt = json.loads(r0["receipt_bytes"])
    admission = verifier.judge_external_anchor_admission(
        expected_protocol_raw_sha256=local.protocol_raw_sha256,
        r0_bundle=r0,
        r1_bundle=r1,
    )
    schema_root = Path(verifier.__file__).parent / "schemas"
    receipt_schema = verifier.strict_json_object_from_bytes(
        (schema_root / "relationship_p4_external_anchor_receipt.schema.json").read_bytes()
    )
    admission_schema = verifier.strict_json_object_from_bytes(
        (schema_root / "relationship_p4_external_anchor_admission.schema.json").read_bytes()
    )
    assert set(receipt) == set(receipt_schema["required"])
    assert set(admission) == set(admission_schema["required"])
    assert set(receipt["verdict"]) == set(receipt_schema["properties"]["verdict"]["required"])
    assert set(admission["verdict"]) == set(admission_schema["$defs"]["verdict"]["required"])
    for value_key, definition_key in (
        ("normative_binding", "normative_binding"),
        ("request_binding", "request_binding"),
        ("capture_binding", "capture_binding"),
        ("anchor_projection", "anchor_projection"),
        ("git_projection", "git_projection"),
    ):
        assert set(receipt[value_key]) == set(receipt_schema["$defs"][definition_key]["required"])
    for projection in receipt["http_projection"].values():
        assert set(projection) == set(receipt_schema["$defs"]["http_projection"]["required"])
    assert set(admission["shared_git_object_graph"]) == set(admission_schema["$defs"]["git_object_graph"]["required"])


def test_host_cuda_and_unrelated_environment_do_not_change_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    local = _make_local_inputs()
    capture = _make_capture(
        local,
        stage="R0",
        process_id=100,
        nonce_digit="a",
        predecessor_receipt_id=None,
        predecessor_manifest_sha256=None,
    )
    before = _verify(local, capture)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "999")
    monkeypatch.setenv("VOLVENCE_UNRELATED_HOST_FACT", "drift")
    assert _verify(local, capture) == before


@pytest.mark.parametrize(
    "path",
    [
        r"\\server\share\git.exe",
        r"\\?\C:\Git\git.exe",
        r"C:\Git\git.exe:stream",
        r"C:\Git\..\git.exe",
        r"C:\Git\trailing.\git.exe",
        r"Git\git.exe",
    ],
)
def test_windows_git_paths_reject_unc_device_ads_and_alias_forms(path: str) -> None:
    with pytest.raises(ValueError):
        verifier._require_canonical_windows_local_path(path, "fixture path")


def test_verifier_exact_inventory_roles_and_firewall_match_the_pinned_observer_shape() -> None:
    import lifeform_evolution.relationship_lab_p4_external_anchor_observer as observer

    assert set(verifier._MAPPED_CAPTURE_FILES) == set(observer._PRE_MAP_CAPTURE_ROLES)
    assert verifier._FILE_ROLES == dict(observer._PRE_MAP_CAPTURE_ROLES)
    assert verifier._AUTHORITY_FIREWALL_KEYS == frozenset(observer._AUTHORITY_FIREWALL)


def test_import_closure_is_stdlib_only_and_does_not_load_runtime_model_cuda_modules() -> None:
    forbidden = {
        "lifeform_core",
        "lifeform_domain_emogpt",
        "volvence_zero.substrate",
        "torch",
        "transformers",
        "vllm",
    }
    script = (
        "import sys; "
        "import lifeform_evolution.relationship_lab_p4_external_anchor_verifier; "
        f"print(sorted({forbidden!r} & set(sys.modules)))"
    )
    environment = dict(os.environ)
    source_root = str(Path(verifier.__file__).parents[1])
    environment["PYTHONPATH"] = source_root
    result = subprocess.run(
        [sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {source_root!r}); {script}"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.stdout.strip() == "[]"
