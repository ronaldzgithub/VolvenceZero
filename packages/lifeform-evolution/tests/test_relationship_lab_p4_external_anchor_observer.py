from __future__ import annotations

import ast
import hashlib
import inspect
import io
import json
import os
import pathlib
import subprocess
import sys
from dataclasses import replace
from typing import Any

import pytest

import lifeform_evolution.relationship_lab_p4_external_anchor_observer as observer


_GIST_ID = "0123456789abcdef0123456789abcdef"
_OWNER = "ronaldzgithub"
_OWNER_ID = 36_839_548
_OWNER_NODE_ID = "MDQ6VXNlcjM2ODM5NTQ4"
_GIST_NODE_ID = "R2lzdDAxMjM0NTY3ODlhYmNkZWY="
_FILENAME = "volvence_p4_7_source_opportunity_a0_anchor_request.json"
_RAW_BODY = b'{"frozen":"a0-request"}\n'
_HTML_BODY = b"<!doctype html><title>public gist</title>"
_PROTOCOL_ID = "1" * 64
_PROTOCOL_RAW_SHA256 = "2" * 64
_REQUEST_ID = "3" * 64
_REQUEST_ARTIFACT_ID = "4" * 64
_REQUEST_MANIFEST_RAW_SHA256 = "5" * 64


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _git_oid(kind: str, body: bytes) -> str:
    framed = f"{kind} {len(body)}\0".encode("ascii") + body
    return hashlib.sha1(framed, usedforsecurity=False).hexdigest()


def _fixture_bundle(
    stage: str = "R0",
) -> tuple[
    observer.ObserverTarget,
    tuple[observer.HttpObservation, ...],
    observer.GitObjectCapture,
]:
    blob_oid = _git_oid("blob", _RAW_BODY)
    tree_body = b"100644 " + _FILENAME.encode("utf-8") + b"\0" + bytes.fromhex(blob_oid)
    tree_oid = _git_oid("tree", tree_body)
    commit_body = (
        f"tree {tree_oid}\n"
        "author Public Observer <public@example.invalid> 0 +0000\n"
        "committer Public Observer <public@example.invalid> 0 +0000\n"
        "\nA0 public commitment\n"
    ).encode("utf-8")
    revision_oid = _git_oid("commit", commit_body)
    target = observer.ObserverTarget(
        observation_stage=stage,
        predecessor_receipt_id=None if stage == "R0" else "6" * 64,
        predecessor_receipt_bundle_manifest_raw_sha256=None if stage == "R0" else "7" * 64,
        protocol_id=_PROTOCOL_ID,
        protocol_raw_sha256=_PROTOCOL_RAW_SHA256,
        protocol_raw_byte_count=16_006,
        request_id=_REQUEST_ID,
        request_artifact_id=_REQUEST_ARTIFACT_ID,
        request_raw_sha256=hashlib.sha256(_RAW_BODY).hexdigest(),
        request_raw_byte_count=len(_RAW_BODY),
        request_manifest_raw_sha256=_REQUEST_MANIFEST_RAW_SHA256,
        request_manifest_raw_byte_count=1_307,
        gist_id=_GIST_ID,
        revision_oid=revision_oid,
        expected_owner_login=_OWNER,
        expected_owner_id=_OWNER_ID,
        expected_owner_node_id=_OWNER_NODE_ID,
        required_filename=_FILENAME,
    )
    raw_url = f"https://{observer.ROLE_HOSTS[observer.ROLE_RAW]}/{_OWNER}/{_GIST_ID}/raw/{revision_oid}/{_FILENAME}"
    html_url = f"https://{observer.ROLE_HOSTS[observer.ROLE_HTML_GIT]}/{_GIST_ID}"
    api_body = _canonical_bytes(
        {
            "files": {
                _FILENAME: {
                    "filename": _FILENAME,
                    "raw_url": raw_url,
                    "size": len(_RAW_BODY),
                    "truncated": False,
                    "encoding": "utf-8",
                    "content": _RAW_BODY.decode("utf-8"),
                }
            },
            "html_url": html_url,
            "git_pull_url": target.git_remote_url,
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
            "history": [{"version": revision_oid}],
        }
    )
    api_headers = (
        ("Content-Type", "application/json; charset=utf-8"),
        ("Date", "Sun, 23 Aug 2026 04:00:00 GMT"),
        ("ETag", '"fixture-etag"'),
        ("X-GitHub-Request-Id", "fixture-request"),
        ("Set-Cookie", "secret-cookie-must-not-be-serialized"),
    )
    observations = (
        observer.HttpObservation(
            role=observer.ROLE_API,
            requested_url=target.api_revision_url,
            final_url=target.api_revision_url,
            status=200,
            response_headers=api_headers,
            body=api_body,
        ),
        observer.HttpObservation(
            role=observer.ROLE_RAW,
            requested_url=raw_url,
            final_url=raw_url,
            status=200,
            response_headers=(("Content-Length", str(len(_RAW_BODY))),),
            body=_RAW_BODY,
        ),
        observer.HttpObservation(
            role=observer.ROLE_HTML_GIT,
            requested_url=html_url,
            final_url=html_url,
            status=200,
            response_headers=(("Content-Type", "text/html; charset=utf-8"),),
            body=_HTML_BODY,
        ),
        observer.HttpObservation(
            role=observer.ROLE_API,
            requested_url=target.api_revision_url,
            final_url=target.api_revision_url,
            status=200,
            response_headers=api_headers,
            body=api_body,
        ),
    )
    git_capture = observer.GitObjectCapture(
        remote_url=target.git_remote_url,
        revision_oid=revision_oid,
        commit_oid=revision_oid,
        tree_oid=tree_oid,
        blob_oid=blob_oid,
        tree_entry_mode="100644",
        tree_entry_name=_FILENAME,
        advertised_refs=(("refs/heads/main", revision_oid),),
        fetched_refs=(("refs/remotes/origin/main", revision_oid),),
        object_inventory=tuple(
            sorted(
                (
                    (revision_oid, "commit", len(commit_body)),
                    (tree_oid, "tree", len(tree_body)),
                    (blob_oid, "blob", len(_RAW_BODY)),
                )
            )
        ),
        object_store_byte_count=4096,
        commit_body=commit_body,
        tree_body=tree_body,
        blob_body=_RAW_BODY,
        advertised_refs_body=f"{revision_oid}\trefs/heads/main\n".encode("ascii"),
        fetched_refs_body=f"refs/remotes/origin/main\0{revision_oid}\n".encode("ascii"),
        object_inventory_body=b"".join(
            f"{oid}\0{kind}\0{size}\n".encode("ascii")
            for oid, kind, size in sorted(
                (
                    (revision_oid, "commit", len(commit_body)),
                    (tree_oid, "tree", len(tree_body)),
                    (blob_oid, "blob", len(_RAW_BODY)),
                )
            )
        ),
        fsck_stdout_sha256=hashlib.sha256(b"").hexdigest(),
        fsck_stderr_sha256=hashlib.sha256(b"").hexdigest(),
        git_executable_byte_count=observer.PRODUCTION_GIT_EXECUTABLE_BYTE_COUNT,
        git_executable_raw_sha256=observer.PRODUCTION_GIT_EXECUTABLE_RAW_SHA256,
        git_version_stdout=observer.PRODUCTION_GIT_VERSION_STDOUT,
        git_helper_identities=tuple(
            (
                str(path),
                observer.PRODUCTION_GIT_HELPER_BYTE_COUNT,
                observer.PRODUCTION_GIT_HELPER_RAW_SHA256,
            )
            for path in observer.PRODUCTION_GIT_HELPER_PATHS
        ),
        command_argv_ledger=(),
        environment_ledger=(),
    )
    return target, observations, git_capture


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_synthetic_observer_claims_before_io_and_captures_fixed_non_authorizing_sequence(
    tmp_path: pathlib.Path,
) -> None:
    target, observations, git_capture = _fixture_bundle()
    output = tmp_path / "observation"
    events: list[str] = []

    class InspectingBackend(observer.SyntheticObserverBackend):
        kind = "production"  # caller labels cannot turn an injected backend into production evidence

        def http_get(self, *, role: str, url: str, body_cap: int) -> observer.HttpObservation:
            assert (output / observer.CLAIM_FILE).is_file()
            events.append(f"http:{role}")
            return super().http_get(role=role, url=url, body_cap=body_cap)

        def git_capture(
            self,
            *,
            remote_url: str,
            revision_oid: str,
            required_filename: str,
            workspace: pathlib.Path,
            git_executable: pathlib.Path | None,
            git_identity: observer.GitExecutableIdentity | None,
        ) -> observer.GitObjectCapture:
            assert (output / observer.CLAIM_FILE).is_file()
            events.append("git")
            return super().git_capture(
                remote_url=remote_url,
                revision_oid=revision_oid,
                required_filename=required_filename,
                workspace=workspace,
                git_executable=git_executable,
                git_identity=git_identity,
            )

    backend = InspectingBackend(http_observations=observations, git_capture=git_capture)
    result = observer.acquire_external_anchor_observation(
        output_dir=output,
        target=target,
        backend=backend,
    )

    assert result.succeeded is True
    assert result.status == "facts_only_observation_complete_non_authorizing"
    assert events == ["http:api", "http:raw", "http:html_git", "git", "http:api"]
    assert backend.http_calls == [
        (observer.ROLE_API, target.api_revision_url, observer.MAX_API_BODY_BYTES),
        (observer.ROLE_RAW, observations[1].requested_url, len(_RAW_BODY)),
        (observer.ROLE_HTML_GIT, observations[2].requested_url, observer.MAX_HTML_BODY_BYTES),
        (observer.ROLE_API, target.api_revision_url, observer.MAX_API_BODY_BYTES),
    ]
    assert backend.git_calls == [(target.git_remote_url, target.revision_oid, _FILENAME)]

    expected_files = {
        observer.CLAIM_FILE,
        observer.API_START_HTTP_FILE,
        observer.API_START_BODY_FILE,
        observer.RAW_HTTP_FILE,
        observer.RAW_BODY_FILE,
        observer.HTML_HTTP_FILE,
        observer.HTML_BODY_FILE,
        observer.GIT_CAPTURE_FILE,
        observer.GIT_ADVERTISED_REFS_FILE,
        observer.GIT_FETCHED_REFS_FILE,
        observer.GIT_OBJECT_INVENTORY_FILE,
        observer.GIT_COMMIT_FILE,
        observer.GIT_TREE_FILE,
        observer.GIT_BLOB_FILE,
        observer.API_END_HTTP_FILE,
        observer.API_END_BODY_FILE,
        observer.CAPTURE_MAP_FILE,
        observer.TERMINAL_FILE,
    }
    assert {item.name for item in output.iterdir()} == expected_files
    assert (output / observer.RAW_BODY_FILE).read_bytes() == _RAW_BODY
    assert (output / observer.GIT_BLOB_FILE).read_bytes() == _RAW_BODY

    claim = _load_json(output / observer.CLAIM_FILE)
    assert claim["backend_kind"] == "synthetic"
    contract = claim["fixed_acquisition_contract"]
    assert contract["method"] == "GET"
    assert contract["github_api_version"] == "2026-03-10"
    assert contract["request_headers"] == {
        "Accept": "application/vnd.github+json",
        "Accept-Encoding": "identity",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "User-Agent": "volvence-a0-gist-observer/1",
        "X-GitHub-Api-Version": "2026-03-10",
    }
    assert contract["role_hosts"] == {
        "api": "api.github.com",
        "html_git": "gist.github.com",
        "raw": "gist.githubusercontent.com",
    }
    assert contract["maximum_redirect_count_by_role"] == {"api": 0, "html_git": 3, "raw": 3}
    assert contract["retry_count"] == 0
    assert contract["connect_timeout_seconds"] == 10
    assert contract["read_timeout_seconds"] == 10
    assert contract["HTTP_overall_timeout_seconds"] == 30
    assert contract["git_total_timeout_seconds"] == 120
    assert contract["maximum_git_object_store_bytes"] == 4 * 1024 * 1024
    assert contract["production_git_executable"] == {
        "path": str(observer.PRODUCTION_GIT_EXECUTABLE),
        "byte_count": observer.PRODUCTION_GIT_EXECUTABLE_BYTE_COUNT,
        "raw_sha256": observer.PRODUCTION_GIT_EXECUTABLE_RAW_SHA256,
        "version_stdout": observer.PRODUCTION_GIT_VERSION_STDOUT,
        "one_buffer_preflight_before_HTTP": True,
    }
    assert claim["target"] == {
        "observation_stage": "R0",
        "predecessor_receipt_id": None,
        "predecessor_receipt_bundle_manifest_raw_sha256": None,
        "protocol_id": _PROTOCOL_ID,
        "protocol_raw_sha256": _PROTOCOL_RAW_SHA256,
        "protocol_raw_byte_count": 16_006,
        "request_id": _REQUEST_ID,
        "request_artifact_id": _REQUEST_ARTIFACT_ID,
        "request_raw_sha256": hashlib.sha256(_RAW_BODY).hexdigest(),
        "request_raw_byte_count": len(_RAW_BODY),
        "request_manifest_raw_sha256": _REQUEST_MANIFEST_RAW_SHA256,
        "request_manifest_raw_byte_count": 1_307,
        "gist_id": _GIST_ID,
        "revision_oid": target.revision_oid,
        "expected_owner_login": _OWNER,
        "expected_owner_id": _OWNER_ID,
        "expected_owner_node_id": _OWNER_NODE_ID,
        "required_filename": _FILENAME,
        "local_protocol_request_manifest_buffers_recomputed_by_observer": False,
        "local_buffer_recomputation_owner": "separate_pinned_verifier",
    }
    assert claim["process_id"] == os.getpid()
    assert len(claim["process_instance_nonce"]) == 64

    api_metadata = _load_json(output / observer.API_START_HTTP_FILE)
    assert api_metadata["response_header_pairs"][-1] == [
        "Set-Cookie",
        "<redacted-set-cookie-value>",
    ]
    assert api_metadata["set_cookie_present"] is True
    assert api_metadata["set_cookie_count"] == 1
    assert "secret-cookie" not in (output / observer.API_START_HTTP_FILE).read_text(encoding="utf-8")

    capture_map = _load_json(output / observer.CAPTURE_MAP_FILE)
    terminal = _load_json(output / observer.TERMINAL_FILE)
    assert capture_map["acquisition_complete"] is True
    assert capture_map["completed_stages"] == [
        "api_exact_revision_start",
        "returned_raw",
        "returned_html",
        "fresh_isolated_bare_git",
        "api_exact_revision_end",
    ]
    assert capture_map["retry_count"] == 0
    assert terminal["A1_required_before_materialization"] is True
    assert terminal["failure"] is None
    assert all(value is False for value in terminal["authority_firewall"].values())
    assert terminal["terminal_id"] == result.terminal_id


def test_failed_observation_is_terminal_create_only_and_never_retries(tmp_path: pathlib.Path) -> None:
    target, observations, git_capture = _fixture_bundle()
    failed_start = observer.HttpObservation(
        role=observer.ROLE_API,
        requested_url=target.api_revision_url,
        final_url=target.api_revision_url,
        status=503,
        response_headers=(("Content-Length", "0"),),
        body=b"",
    )
    backend = observer.SyntheticObserverBackend(
        http_observations=(failed_start, *observations[1:]),
        git_capture=git_capture,
    )
    output = tmp_path / "failed"
    result = observer.acquire_external_anchor_observation(
        output_dir=output,
        target=target,
        backend=backend,
    )

    assert result.succeeded is False
    assert backend.http_calls == [(observer.ROLE_API, target.api_revision_url, observer.MAX_API_BODY_BYTES)]
    assert backend.git_calls == []
    assert {item.name for item in output.iterdir()} == {
        observer.CLAIM_FILE,
        observer.CAPTURE_MAP_FILE,
        observer.TERMINAL_FILE,
    }
    terminal = _load_json(output / observer.TERMINAL_FILE)
    assert terminal["status"] == "facts_only_observation_failed_non_authorizing"
    assert terminal["retry_count"] == 0
    assert terminal["failure"]["code"] == "http_status_not_200"
    assert all(value is False for value in terminal["authority_firewall"].values())
    with pytest.raises(FileExistsError, match="create-only"):
        observer.acquire_external_anchor_observation(
            output_dir=output,
            target=target,
            backend=backend,
        )


def test_api_history_and_git_head_cardinality_are_exact_not_proxy_checks(
    tmp_path: pathlib.Path,
) -> None:
    target, observations, git_capture = _fixture_bundle()
    api_payload = json.loads(observations[0].body)
    api_payload["history"].append({"version": "f" * 40})
    invalid_api = replace(observations[0], body=_canonical_bytes(api_payload))
    result = observer.acquire_external_anchor_observation(
        output_dir=tmp_path / "history-failure",
        target=target,
        backend=observer.SyntheticObserverBackend(
            http_observations=(invalid_api, *observations[1:]),
            git_capture=git_capture,
        ),
    )
    assert result.succeeded is False
    terminal = _load_json(result.output_dir / observer.TERMINAL_FILE)
    assert terminal["failure"]["code"] == "api_history_invalid"

    two_heads = replace(
        git_capture,
        advertised_refs=(
            ("refs/heads/main", target.revision_oid),
            ("refs/heads/other", target.revision_oid),
        ),
    )
    with pytest.raises(observer.ObserverError) as captured:
        observer._validate_git_capture(two_heads)
    assert captured.value.code == "git_advertised_head_mismatch"


def test_api_returned_url_must_bind_the_same_gist_path(tmp_path: pathlib.Path) -> None:
    target, observations, git_capture = _fixture_bundle()
    api_payload = json.loads(observations[0].body)
    api_payload["files"][_FILENAME]["raw_url"] = (
        f"https://{observer.ROLE_HOSTS[observer.ROLE_RAW]}/{_OWNER}/"
        f"{'a' * len(_GIST_ID)}/raw/{target.revision_oid}/{_FILENAME}"
    )
    invalid_api = replace(observations[0], body=_canonical_bytes(api_payload))
    result = observer.acquire_external_anchor_observation(
        output_dir=tmp_path / "cross-gist-url-failure",
        target=target,
        backend=observer.SyntheticObserverBackend(
            http_observations=(invalid_api, *observations[1:]),
            git_capture=git_capture,
        ),
    )
    assert result.succeeded is False
    terminal = _load_json(result.output_dir / observer.TERMINAL_FILE)
    assert terminal["failure"]["code"] == "api_raw_url_path_mismatch"


def test_partial_multi_file_write_is_reinventoried_without_orphan_omission(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, observations, git_capture = _fixture_bundle()
    original = observer._write_bytes_capture
    injected = False

    def fail_once(
        root: pathlib.Path,
        name: str,
        payload: bytes,
        *,
        role: str,
    ) -> dict[str, object]:
        nonlocal injected
        if name == observer.GIT_TREE_FILE and not injected:
            injected = True
            raise OSError("injected write failure")
        return original(root, name, payload, role=role)

    monkeypatch.setattr(observer, "_write_bytes_capture", fail_once)
    result = observer.acquire_external_anchor_observation(
        output_dir=tmp_path / "partial-write",
        target=target,
        backend=observer.SyntheticObserverBackend(
            http_observations=observations,
            git_capture=git_capture,
        ),
    )
    assert result.succeeded is False
    capture_map = _load_json(result.output_dir / observer.CAPTURE_MAP_FILE)
    terminal = _load_json(result.output_dir / observer.TERMINAL_FILE)
    actual_root_files = {
        item.name
        for item in result.output_dir.iterdir()
        if item.name not in {observer.CAPTURE_MAP_FILE, observer.TERMINAL_FILE}
    }
    inventoried_files = {entry["path"] for entry in capture_map["files"]}
    assert inventoried_files == actual_root_files
    assert observer.GIT_COMMIT_FILE in inventoried_files
    assert observer.GIT_TREE_FILE not in inventoried_files
    assert capture_map["root_closure_status"] == "partial_failure_pre_map_root"
    assert capture_map["acquisition_complete"] is False
    assert terminal["failure"]["code"] == "unclassified_acquisition_failure"
    assert all(value is False for value in terminal["authority_firewall"].values())


class _FakeSocket:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def settimeout(self, value: float) -> None:
        self.timeouts.append(value)


class _FakeResponse:
    def __init__(self, status: int, headers: tuple[tuple[str, str], ...], body: bytes = b"") -> None:
        self.status = status
        self._headers = headers
        self._body = body
        self._offset = 0

    def getheaders(self) -> list[tuple[str, str]]:
        return list(self._headers)

    def read(self, amount: int) -> bytes:
        if self._offset >= len(self._body):
            return b""
        chunk = self._body[self._offset : self._offset + amount]
        self._offset += len(chunk)
        return chunk


def test_production_http_uses_fixed_identity_free_headers_and_manual_raw_role_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = "https://gist.githubusercontent.com/owner/id/raw/revision/file"
    redirected = "https://gist.githubusercontent.com/owner/id/raw/revision-2/file"
    responses = [
        _FakeResponse(302, (("Location", redirected),)),
        _FakeResponse(200, (("Content-Length", "2"),), b"{}"),
    ]
    connections: list[Any] = []

    class FakeConnection:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.sock = _FakeSocket()
            self.request: tuple[str, str, bool, bool] | None = None
            self.headers: list[tuple[str, str]] = []
            self.closed = False
            connections.append(self)

        def connect(self) -> None:
            return None

        def putrequest(self, method: str, path: str, *, skip_host: bool, skip_accept_encoding: bool) -> None:
            self.request = (method, path, skip_host, skip_accept_encoding)

        def putheader(self, name: str, value: str) -> None:
            self.headers.append((name, value))

        def endheaders(self) -> None:
            return None

        def getresponse(self) -> _FakeResponse:
            return responses.pop(0)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(observer, "_system_tls_context", lambda: object())
    monkeypatch.setattr(observer.http.client, "HTTPSConnection", FakeConnection)
    observation = observer.ProductionObserverBackend().http_get(
        role=observer.ROLE_RAW,
        url=start,
        body_cap=100,
    )

    assert observation.body == b"{}"
    assert observation.final_url == redirected
    assert len(observation.redirects) == 1
    assert len(connections) == 2
    expected_headers = {
        "Host": "gist.githubusercontent.com",
        "Accept": "application/vnd.github+json",
        "User-Agent": "volvence-a0-gist-observer/1",
        "Accept-Encoding": "identity",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "X-GitHub-Api-Version": "2026-03-10",
    }
    for connection in connections:
        assert connection.kwargs == {
            "host": "gist.githubusercontent.com",
            "port": 443,
            "timeout": 10.0,
            "context": connection.kwargs["context"],
        }
        assert connection.request is not None
        assert connection.request[0] == "GET"
        assert connection.request[2:] == (True, True)
        assert dict(connection.headers) == expected_headers
        assert "Authorization" not in dict(connection.headers)
        assert "Cookie" not in dict(connection.headers)
        assert connection.sock.timeouts == [10.0]
        assert connection.closed is True


@pytest.mark.parametrize(
    ("role", "start", "locations", "message"),
    [
        (
            observer.ROLE_API,
            "https://api.github.com/start",
            ("https://api.github.com/redirected",),
            "redirect_limit_exceeded",
        ),
        (
            observer.ROLE_RAW,
            "https://gist.githubusercontent.com/start",
            ("https://gist.github.com/owner/id",),
            "url_lexical_invalid",
        ),
        (
            observer.ROLE_RAW,
            "https://gist.githubusercontent.com/start",
            (
                "https://gist.githubusercontent.com/r1",
                "https://gist.githubusercontent.com/r2",
                "https://gist.githubusercontent.com/r3",
                "https://gist.githubusercontent.com/r4",
            ),
            "redirect_limit_exceeded",
        ),
    ],
)
def test_production_http_enforces_role_specific_redirect_contract(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    start: str,
    locations: tuple[str, ...],
    message: str,
) -> None:
    responses = [_FakeResponse(302, (("Location", location),)) for location in locations]

    class FakeConnection:
        def __init__(self, **_kwargs: object) -> None:
            self.sock = _FakeSocket()

        def connect(self) -> None:
            return None

        def putrequest(self, *_args: object, **_kwargs: object) -> None:
            return None

        def putheader(self, *_args: object) -> None:
            return None

        def endheaders(self) -> None:
            return None

        def getresponse(self) -> _FakeResponse:
            return responses.pop(0)

        def close(self) -> None:
            return None

    monkeypatch.setattr(observer, "_system_tls_context", lambda: object())
    monkeypatch.setattr(observer.http.client, "HTTPSConnection", FakeConnection)
    with pytest.raises(observer.ObserverError) as captured:
        observer.ProductionObserverBackend().http_get(
            role=role,
            url=start,
            body_cap=100,
        )
    assert captured.value.code == message


def test_production_git_plan_is_fresh_isolated_all_heads_fsck_and_raw_object_capture(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, _observations, expected = _fixture_bundle()
    executable = tmp_path / ("git.exe" if os.name == "nt" else "git")
    executable.write_bytes(b"test executable placeholder")
    calls: list[dict[str, object]] = []

    def fake_run_git(
        selected_executable: pathlib.Path,
        arguments: tuple[str, ...],
        *,
        environment: dict[str, str],
        started: float,
        clock: Any,
        input_bytes: bytes | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        del started, clock
        calls.append(
            {
                "executable": selected_executable,
                "arguments": arguments,
                "environment": dict(environment),
                "input": input_bytes,
            }
        )
        if "init" in arguments:
            bare = pathlib.Path(arguments[-1])
            (bare / "objects" / "aa").mkdir(parents=True)
            (bare / "objects" / "aa" / "object").write_bytes(b"object store fixture")
            (bare / "config").write_text(
                "[core]\n\trepositoryformatversion = 0\n\tbare = true\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(arguments, 0, b"", b"")
        if "for-each-ref" in arguments:
            stdout = f"refs/remotes/origin/main\0{target.revision_oid}\n".encode("ascii")
            return subprocess.CompletedProcess(arguments, 0, stdout, b"")
        if "ls-remote" in arguments:
            stdout = f"{target.revision_oid}\trefs/heads/main\n".encode("ascii")
            return subprocess.CompletedProcess(arguments, 0, stdout, b"")
        if "--batch-all-objects" in arguments:
            return subprocess.CompletedProcess(arguments, 0, expected.object_inventory_body, b"")
        if "cat-file" in arguments:
            assert input_bytes is not None
            object_name = input_bytes.decode("ascii").strip()
            objects = {
                expected.commit_oid: ("commit", expected.commit_body),
                expected.tree_oid: ("tree", expected.tree_body),
                expected.blob_oid: ("blob", expected.blob_body),
            }
            kind, body = objects[object_name]
            stdout = f"{object_name} {kind} {len(body)}\n".encode("ascii") + body + b"\n"
            return subprocess.CompletedProcess(arguments, 0, stdout, b"")
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    monkeypatch.setattr(observer, "_run_git", fake_run_git)
    git_identity = observer.GitExecutableIdentity(
        path=executable,
        byte_count=observer.PRODUCTION_GIT_EXECUTABLE_BYTE_COUNT,
        raw_sha256=observer.PRODUCTION_GIT_EXECUTABLE_RAW_SHA256,
        version_stdout=observer.PRODUCTION_GIT_VERSION_STDOUT,
        helper_identities=expected.git_helper_identities,
    )
    capture = observer._capture_fresh_bare_git(
        git_executable=executable,
        git_identity=git_identity,
        remote_url=target.git_remote_url,
        revision_oid=target.revision_oid,
        required_filename=target.required_filename,
        workspace=tmp_path,
        clock=lambda: 1.0,
    )

    assert capture.commit_body == expected.commit_body
    assert capture.tree_body == expected.tree_body
    assert capture.blob_body == expected.blob_body
    assert capture.tree_entry_mode == "100644"
    commands = [tuple(call["arguments"]) for call in calls]
    assert capture.command_argv_ledger == tuple((str(executable), *tuple(call["arguments"])) for call in calls)
    assert dict(capture.environment_ledger) == calls[0]["environment"]
    flattened = "\n".join(" ".join(command) for command in commands)
    assert "init --bare --object-format=sha1" in flattened
    assert "ls-remote --heads" in flattened
    assert "+refs/heads/*:refs/remotes/origin/*" in flattened
    assert "--no-tags" in flattened
    assert "--no-recurse-submodules" in flattened
    assert "fsck --full --strict --no-reflogs" in flattened
    assert "--depth" not in flattened
    assert "http.followRedirects=false" in flattened
    assert "credential.helper=" in flattened
    assert "core.askPass=" in flattened
    assert "http.extraHeader=" in flattened
    assert "http.cookieFile=" in flattened
    assert "http.proxy=" in flattened
    assert "core.useReplaceRefs=false" in flattened
    assert "fetch.fsckObjects=true" in flattened
    assert "transfer.fsckObjects=true" in flattened

    for call in calls:
        environment = call["environment"]
        assert "HOME" not in environment
        assert "USERPROFILE" not in environment
        assert "HTTP_PROXY" not in environment
        assert "HTTPS_PROXY" not in environment
        assert "GITHUB_TOKEN" not in environment
        assert "GH_TOKEN" not in environment
        assert "GIT_ASKPASS" not in environment
        assert "SSL_CERT_FILE" not in environment
        assert "COMSPEC" not in environment
        assert "PATHEXT" not in environment
        assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
        assert environment["GIT_CONFIG_GLOBAL"].endswith("empty-global-config")
        assert environment["XDG_CONFIG_HOME"].endswith("empty-xdg-config")
        assert environment["GIT_EXEC_PATH"] == str(pathlib.Path(observer.PRODUCTION_GIT_HELPER_PATHS[0]).parent)
        assert environment["GIT_TERMINAL_PROMPT"] == "0"
        assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"


def test_production_git_preflight_hashes_main_and_https_helpers_before_use(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "git.exe"
    remote_http = tmp_path / "git-remote-http.exe"
    remote_https = tmp_path / "git-remote-https.exe"
    executable_bytes = b"frozen-main-git"
    helper_bytes = b"frozen-https-helper"
    executable.write_bytes(executable_bytes)
    remote_http.write_bytes(helper_bytes)
    remote_https.write_bytes(helper_bytes)
    monkeypatch.setattr(observer, "PRODUCTION_GIT_EXECUTABLE", pathlib.PureWindowsPath(executable))
    monkeypatch.setattr(observer, "PRODUCTION_GIT_EXECUTABLE_BYTE_COUNT", len(executable_bytes))
    monkeypatch.setattr(
        observer,
        "PRODUCTION_GIT_EXECUTABLE_RAW_SHA256",
        hashlib.sha256(executable_bytes).hexdigest(),
    )
    monkeypatch.setattr(
        observer,
        "PRODUCTION_GIT_HELPER_PATHS",
        (pathlib.PureWindowsPath(remote_http), pathlib.PureWindowsPath(remote_https)),
    )
    monkeypatch.setattr(observer, "PRODUCTION_GIT_HELPER_BYTE_COUNT", len(helper_bytes))
    monkeypatch.setattr(
        observer,
        "PRODUCTION_GIT_HELPER_RAW_SHA256",
        hashlib.sha256(helper_bytes).hexdigest(),
    )
    monkeypatch.setattr(observer, "PRODUCTION_GIT_VERSION_STDOUT", "git version test")

    def fake_version(
        selected_executable: pathlib.Path,
        arguments: tuple[str, ...],
        *,
        environment: dict[str, str],
        started: float,
        clock: Any,
        input_bytes: bytes | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        del started, clock, input_bytes
        assert selected_executable == executable
        assert arguments == ("--version",)
        assert "HOME" not in environment
        assert environment["GIT_CONFIG_GLOBAL"].endswith("empty-global-config")
        assert environment["XDG_CONFIG_HOME"].endswith("empty-xdg-config")
        return subprocess.CompletedProcess((str(executable), "--version"), 0, b"git version test\n", b"")

    monkeypatch.setattr(observer, "_run_git", fake_version)
    identity = observer._preflight_production_git_executable(executable, workspace=tmp_path)
    assert identity.raw_sha256 == hashlib.sha256(executable_bytes).hexdigest()
    assert identity.byte_count == len(executable_bytes)
    assert identity.helper_identities == (
        (str(remote_http), len(helper_bytes), hashlib.sha256(helper_bytes).hexdigest()),
        (str(remote_https), len(helper_bytes), hashlib.sha256(helper_bytes).hexdigest()),
    )


def test_module_has_no_network_capability_in_synthetic_backend_and_no_verdict_surface() -> None:
    synthetic_source = inspect.getsource(observer.SyntheticObserverBackend)
    assert "http.client" not in synthetic_source
    assert "subprocess" not in synthetic_source
    assert "socket" not in synthetic_source
    assert "urlopen" not in synthetic_source
    assert "verdict" not in synthetic_source.casefold()

    module_source = pathlib.Path(observer.__file__).read_text(encoding="utf-8")
    assert "urllib.request" not in module_source
    assert "shell=True" not in module_source
    assert 'retry_count": 0' not in module_source  # emitted from the frozen constant, not a mutable literal
    assert "socket.gethostname" not in module_source
    assert "getpass" not in module_source


def test_failure_code_registry_exactly_covers_literal_observer_errors() -> None:
    source = pathlib.Path(observer.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "ObserverError"
    ]
    assert calls
    assert all(
        node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str) for node in calls
    )
    literal_codes = {node.args[0].value for node in calls}
    assert literal_codes == observer.OBSERVER_FAILURE_CODES
    with pytest.raises(ValueError, match="not frozen"):
        observer.ObserverError("caller_invented_failure", "must fail")


@pytest.mark.parametrize(
    ("headers", "expected_code"),
    [
        (
            (("Content-Length", "2"), ("Content-Length", "2")),
            "content_length_invalid",
        ),
        (
            (("Transfer-Encoding", "chunked"), ("Transfer-Encoding", "chunked")),
            "transfer_encoding_duplicated",
        ),
        (
            (("Content-Encoding", "identity"), ("Content-Encoding", "identity")),
            "content_encoding_duplicated",
        ),
        (
            (("Content-Length", "2"), ("Transfer-Encoding", "chunked")),
            "http_framing_ambiguous",
        ),
    ],
)
def test_http_framing_rejects_duplicate_or_ambiguous_length_and_encoding(
    headers: tuple[tuple[str, str], tuple[str, str]],
    expected_code: str,
) -> None:
    with pytest.raises(observer.ObserverError) as captured:
        observer._response_header_safety(headers, role=observer.ROLE_RAW, final_body_length=2)
    assert captured.value.code == expected_code


def test_api_response_content_type_rejects_vendor_media_type() -> None:
    with pytest.raises(observer.ObserverError) as captured:
        observer._response_header_safety(
            (("Content-Type", "application/vnd.github+json"),),
            role=observer.ROLE_API,
            final_body_length=0,
        )
    assert captured.value.code == "api_content_type_invalid"


def test_strict_json_rejects_nonfinite_numbers_and_relative_redirect_locations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(observer.ObserverError) as nonfinite:
        observer._strict_json_object(b'{"value":NaN}', "fixture")
    assert nonfinite.value.code == "json_nonfinite_number"

    responses = [_FakeResponse(302, (("Location", "/relative"),))]

    class FakeConnection:
        def __init__(self, **_kwargs: object) -> None:
            self.sock = _FakeSocket()

        def connect(self) -> None:
            return None

        def putrequest(self, *_args: object, **_kwargs: object) -> None:
            return None

        def putheader(self, *_args: object) -> None:
            return None

        def endheaders(self) -> None:
            return None

        def getresponse(self) -> _FakeResponse:
            return responses.pop(0)

        def close(self) -> None:
            return None

    monkeypatch.setattr(observer, "_system_tls_context", lambda: object())
    monkeypatch.setattr(observer.http.client, "HTTPSConnection", FakeConnection)
    with pytest.raises(observer.ObserverError) as relative:
        observer.ProductionObserverBackend().http_get(
            role=observer.ROLE_RAW,
            url="https://gist.githubusercontent.com/start",
            body_cap=10,
        )
    assert relative.value.code == "url_lexical_invalid"


def test_git_runner_stream_cap_and_nonzero_failure_terminate_the_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = observer._PipeReadState()
    wake = observer.threading.Event()
    observer._read_capped_pipe(
        io.BytesIO(b"x" * (observer.MAX_GIT_PROCESS_STREAM_BYTES + 1)),
        state,
        wake,
    )
    assert state.exceeded is True
    assert state.byte_count == observer.MAX_GIT_PROCESS_STREAM_BYTES

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = io.BytesIO(b"")
            self.stderr = io.BytesIO(b"fixed failure")
            self.stdin = None
            self.returncode = 1

        def poll(self) -> int:
            return self.returncode

        def wait(self, timeout: float) -> int:
            del timeout
            return self.returncode

    class FakeGuard:
        def __init__(self) -> None:
            self.terminated = False
            self.closed = False

        def terminate(self) -> None:
            self.terminated = True

        def active_process_count(self) -> int:
            return int(not self.terminated)

        def wait_empty(self, deadline: float) -> bool:
            del deadline
            return self.terminated

        def close(self) -> None:
            self.closed = True

    process = FakeProcess()
    guard = FakeGuard()
    monkeypatch.setattr(observer, "_start_git_process", lambda *_args, **_kwargs: (process, guard))
    with pytest.raises(observer.ObserverError) as captured:
        observer._run_git(
            pathlib.Path(sys.executable),
            ("--version",),
            environment={},
            started=0.0,
            clock=lambda: 0.0,
        )
    assert captured.value.code == "git_command_failed"
    assert guard.terminated is True
    assert guard.closed is True


def test_git_runner_rejects_oversized_stdin_before_process_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process_started = False

    def forbidden_start(*_args: object, **_kwargs: object) -> None:
        nonlocal process_started
        process_started = True

    monkeypatch.setattr(observer, "_start_git_process", forbidden_start)
    with pytest.raises(observer.ObserverError) as captured:
        observer._run_git(
            pathlib.Path(sys.executable),
            ("-c", "pass"),
            environment={},
            started=0.0,
            clock=lambda: 0.0,
            input_bytes=b"x" * (observer.MAX_GIT_STDIN_BYTES + 1),
        )
    assert captured.value.code == "git_process_input_cap_exceeded"
    assert process_started is False


def test_fixed_contract_exposes_job_tree_and_per_stream_resource_bounds() -> None:
    contract = observer._fixed_acquisition_contract()
    assert contract["git_process_execution_platform"] == "Windows_only_production"
    assert contract["git_job_KILL_ON_JOB_CLOSE"] is True
    assert contract["git_job_breakaway_allowed"] is False
    assert contract["git_timeout_or_failure_terminates_whole_job_tree"] is True
    assert contract["git_success_requires_root_exit_and_Job_ActiveProcesses_zero"] is True
    assert contract["git_process_tree_reap_timeout_seconds"] == observer.GIT_REAP_TIMEOUT_SECONDS
    assert contract["maximum_git_process_stdout_bytes_per_command"] == observer.MAX_GIT_PROCESS_STREAM_BYTES
    assert contract["maximum_git_process_stderr_bytes_per_command"] == observer.MAX_GIT_PROCESS_STREAM_BYTES
    assert contract["maximum_git_process_stdin_bytes_per_command"] == observer.MAX_GIT_STDIN_BYTES
    assert contract["git_stdin_writer_is_concurrent_and_deadline_bounded"] is True
    assert contract["git_version_preflight_uses_same_bounded_job_runner"] is True
    assert contract["git_object_store_cap_is_post_fetch_acceptance_not_fetch_time_limit"] is True


def test_fresh_git_repository_rejects_hardlinked_files(tmp_path: pathlib.Path) -> None:
    bare = tmp_path / "bare"
    object_directory = bare / "objects" / "aa"
    object_directory.mkdir(parents=True)
    (bare / "config").write_text("[core]\n\tbare = true\n", encoding="utf-8")
    object_file = object_directory / "object"
    object_file.write_bytes(b"object")
    os.link(object_file, object_directory / "hardlink")

    with pytest.raises(observer.ObserverError) as captured:
        observer._validate_fresh_git_repository(bare)
    assert captured.value.code == "git_repository_escape_surface"


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_windows_git_runner_enforces_stream_cap_during_process_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(observer, "GIT_TOTAL_TIMEOUT_SECONDS", 5.0)
    script = f"import sys;sys.stdout.buffer.write(b'x'*{observer.MAX_GIT_PROCESS_STREAM_BYTES + 1})"
    with pytest.raises(observer.ObserverError) as captured:
        observer._run_git(
            pathlib.Path(sys.executable),
            ("-c", script),
            environment=dict(os.environ),
            started=0.0,
            clock=lambda: 0.0,
        )
    assert captured.value.code == "git_process_output_cap_exceeded"


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_windows_git_runner_timeout_reaps_spawned_child_process(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ctypes
    import ctypes.wintypes as wintypes

    child_pid_file = tmp_path / "child.pid"
    child_script = "import time; time.sleep(60)"
    parent_script = (
        "import pathlib,subprocess,sys,time;"
        f"p=subprocess.Popen([sys.executable,'-c',{child_script!r}]);"
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(p.pid),encoding='ascii');"
        "time.sleep(60)"
    )
    monkeypatch.setattr(observer, "GIT_TOTAL_TIMEOUT_SECONDS", 0.4)
    with pytest.raises(observer.ObserverError) as captured:
        observer._run_git(
            pathlib.Path(sys.executable),
            ("-c", parent_script),
            environment=dict(os.environ),
            started=0.0,
            clock=lambda: 0.0,
        )
    assert captured.value.code == "git_total_timeout"
    child_pid = int(child_pid_file.read_text(encoding="ascii"))
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
    kernel32.TerminateProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(0x00100001, False, child_pid)
    if handle:
        try:
            wait_result = kernel32.WaitForSingleObject(handle, 0)
            if wait_result != 0:
                kernel32.TerminateProcess(handle, 0xE0000002)
                kernel32.WaitForSingleObject(handle, 10_000)
            assert wait_result == 0
        finally:
            kernel32.CloseHandle(handle)
