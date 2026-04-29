"""Tests for coding vertical affordances (Gap 1 slice 2b).

Covers:

* Sandbox path resolution (``resolve_sandbox_path``): reject
  parent-escape, absolute-outside, symlink-escape, non-existent,
  non-directory.
* Each backend's happy path (``read_file`` / ``list_dir`` / ``grep``).
* Each backend's failure mode (file missing, grep pattern empty,
  grep budget truncation, list_dir on a file, read_file byte cap,
  non-UTF-8 rejection).
* The coding-vertical registry wiring: every descriptor has a
  backend and the full boundary gate (consent + regime block) fires.
* End-to-end: a real ``build_coding_lifeform`` session + invoker
  reads a sandbox file and gets the content back through the
  kernel's execution_result owner.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import sys

import pytest

from lifeform_affordance import (
    AffordanceInvocationStatus,
    AffordanceRegistry,
)
from lifeform_domain_coding import (
    CODING_AFFORDANCE_DESCRIPTORS,
    CONSENT_FILESYSTEM_READ,
    SandboxPathError,
    build_coding_affordance_backends,
    build_coding_affordance_invoker,
    build_coding_affordance_registry,
    resolve_sandbox_path,
)


# ---------------------------------------------------------------------------
# Sandbox fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sandbox(tmp_path: pathlib.Path) -> pathlib.Path:
    """A sandbox with a realistic mix of files + a subdirectory."""
    (tmp_path / "README.md").write_text(
        "# Sample workspace\n\nThis is a placeholder for grep tests.\n",
        encoding="utf-8",
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        "def main() -> None:\n"
        "    print('hello world')\n"
        "\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "util.py").write_text(
        "def helper() -> int:\n    return 42\n\n\ndef main() -> None:\n    helper()\n",
        encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text(
        "def test_helper():\n    from src.util import helper\n    assert helper() == 42\n",
        encoding="utf-8",
    )
    # Binary file to ensure grep skips + read_file surfaces utf-8 error.
    (tmp_path / "binary.bin").write_bytes(bytes(range(256)))
    return tmp_path


# ---------------------------------------------------------------------------
# Sandbox path resolution
# ---------------------------------------------------------------------------


def test_resolve_sandbox_path_accepts_relative_inside(sandbox: pathlib.Path) -> None:
    resolved = resolve_sandbox_path("src/main.py", sandbox_root=sandbox)
    assert resolved.is_file()
    assert resolved.samefile(sandbox / "src" / "main.py")


def test_resolve_sandbox_path_accepts_absolute_inside(sandbox: pathlib.Path) -> None:
    absolute = str(sandbox / "src" / "main.py")
    resolved = resolve_sandbox_path(absolute, sandbox_root=sandbox)
    assert resolved.is_file()


def test_resolve_sandbox_path_rejects_parent_escape(sandbox: pathlib.Path) -> None:
    # ../ escape.
    with pytest.raises(SandboxPathError, match="outside sandbox"):
        resolve_sandbox_path("../outside.txt", sandbox_root=sandbox)


def test_resolve_sandbox_path_rejects_absolute_outside(tmp_path: pathlib.Path) -> None:
    sandbox_root = tmp_path / "work"
    sandbox_root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("data", encoding="utf-8")
    with pytest.raises(SandboxPathError, match="outside sandbox"):
        resolve_sandbox_path(str(outside), sandbox_root=sandbox_root)


def test_resolve_sandbox_path_rejects_missing_file(sandbox: pathlib.Path) -> None:
    with pytest.raises(SandboxPathError, match="does not exist"):
        resolve_sandbox_path("src/ghost.py", sandbox_root=sandbox)


def test_resolve_sandbox_path_rejects_empty_path(sandbox: pathlib.Path) -> None:
    with pytest.raises(SandboxPathError, match="non-empty"):
        resolve_sandbox_path("", sandbox_root=sandbox)


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="symlink tests require admin on Windows; rely on Unix runners",
)
def test_resolve_sandbox_path_rejects_symlink_escape(
    tmp_path: pathlib.Path,
) -> None:
    sandbox_root = tmp_path / "work"
    sandbox_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    link = sandbox_root / "escape"
    os.symlink(outside, link)  # symlink inside sandbox pointing outside
    with pytest.raises(SandboxPathError, match="outside sandbox"):
        resolve_sandbox_path("escape/secret.txt", sandbox_root=sandbox_root)


# ---------------------------------------------------------------------------
# read_file backend
# ---------------------------------------------------------------------------


def test_read_file_success_returns_content(sandbox: pathlib.Path) -> None:
    backends = build_coding_affordance_backends(sandbox)
    result = asyncio.run(backends["read_file"]({"path": "README.md"}))
    assert "Sample workspace" in result["content"]
    assert result["truncated"] is False
    assert result["byte_count"] > 0
    assert str(sandbox / "README.md") == result["resolved_path"] or (
        pathlib.Path(result["resolved_path"]).samefile(sandbox / "README.md")
    )


def test_read_file_truncates_at_max_bytes(sandbox: pathlib.Path) -> None:
    backends = build_coding_affordance_backends(sandbox)
    result = asyncio.run(
        backends["read_file"]({"path": "README.md", "max_bytes": 8})
    )
    assert result["truncated"] is True
    assert len(result["content"].encode("utf-8")) <= 8


def test_read_file_fails_on_non_utf8_binary(sandbox: pathlib.Path) -> None:
    backends = build_coding_affordance_backends(sandbox)
    with pytest.raises(UnicodeDecodeError):
        asyncio.run(backends["read_file"]({"path": "binary.bin"}))


def test_read_file_fails_on_directory(sandbox: pathlib.Path) -> None:
    backends = build_coding_affordance_backends(sandbox)
    with pytest.raises(SandboxPathError, match="not a regular file"):
        asyncio.run(backends["read_file"]({"path": "src"}))


def test_read_file_fails_on_parent_escape(sandbox: pathlib.Path) -> None:
    backends = build_coding_affordance_backends(sandbox)
    with pytest.raises(SandboxPathError, match="outside sandbox"):
        asyncio.run(backends["read_file"]({"path": "../escape.txt"}))


# ---------------------------------------------------------------------------
# list_dir backend
# ---------------------------------------------------------------------------


def test_list_dir_returns_sorted_entries(sandbox: pathlib.Path) -> None:
    backends = build_coding_affordance_backends(sandbox)
    result = asyncio.run(backends["list_dir"]({"path": "src"}))
    names = [e["name"] for e in result["entries"]]
    assert names == sorted(names)
    assert set(names) == {"main.py", "util.py"}
    for e in result["entries"]:
        assert e["kind"] == "file"


def test_list_dir_root_includes_all_top_level(sandbox: pathlib.Path) -> None:
    backends = build_coding_affordance_backends(sandbox)
    result = asyncio.run(backends["list_dir"]({"path": "."}))
    names = {e["name"] for e in result["entries"]}
    # "src" and "tests" are directories; "README.md" and "binary.bin" are files.
    assert {"src", "tests", "README.md", "binary.bin"} <= names
    kinds = {e["name"]: e["kind"] for e in result["entries"]}
    assert kinds["src"] == "dir"
    assert kinds["README.md"] == "file"


def test_list_dir_rejects_when_path_is_file(sandbox: pathlib.Path) -> None:
    backends = build_coding_affordance_backends(sandbox)
    with pytest.raises(SandboxPathError, match="not a directory"):
        asyncio.run(backends["list_dir"]({"path": "README.md"}))


# ---------------------------------------------------------------------------
# grep backend
# ---------------------------------------------------------------------------


def test_grep_finds_matches_across_files(sandbox: pathlib.Path) -> None:
    backends = build_coding_affordance_backends(sandbox)
    result = asyncio.run(backends["grep"]({"pattern": "def main"}))
    assert result["truncated"] is False
    assert result["scanned_files"] >= 2
    # main() defined in both src/main.py and src/util.py.
    paths_matched = {m["path"] for m in result["matches"]}
    assert any("main.py" in p for p in paths_matched)
    assert any("util.py" in p for p in paths_matched)


def test_grep_respects_subdir_scope(sandbox: pathlib.Path) -> None:
    backends = build_coding_affordance_backends(sandbox)
    result = asyncio.run(
        backends["grep"]({"pattern": "def", "subdir": "tests"})
    )
    paths_matched = {m["path"] for m in result["matches"]}
    # Only tests/ files scanned.
    assert all("tests" in p for p in paths_matched)


def test_grep_truncates_at_max_results(sandbox: pathlib.Path) -> None:
    backends = build_coding_affordance_backends(sandbox)
    result = asyncio.run(
        backends["grep"]({"pattern": "def", "max_results": 1})
    )
    assert result["truncated"] is True
    assert len(result["matches"]) == 1


def test_grep_skips_binary_files(sandbox: pathlib.Path) -> None:
    """binary.bin should not appear in grep results even if the
    pattern coincidentally matches bytes \u2014 UTF-8 decode will
    fail and the file is skipped.
    """
    backends = build_coding_affordance_backends(sandbox)
    # Pattern with only ASCII bytes that ARE present in binary.bin
    # (0x6D 0x61 0x69 0x6E = "main"). But grep should skip the
    # binary because read_text with utf-8 raises on the full range.
    result = asyncio.run(backends["grep"]({"pattern": "main"}))
    paths_matched = {m["path"] for m in result["matches"]}
    assert all("binary.bin" not in p for p in paths_matched)


def test_grep_rejects_empty_pattern(sandbox: pathlib.Path) -> None:
    backends = build_coding_affordance_backends(sandbox)
    with pytest.raises(SandboxPathError, match="non-empty"):
        asyncio.run(backends["grep"]({"pattern": ""}))


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_build_coding_registry_contains_three_descriptors() -> None:
    registry = build_coding_affordance_registry()
    assert set(registry.names()) == {"read_file", "list_dir", "grep"}
    assert registry.sealed is True


def test_build_coding_registry_allows_extension_when_seal_false() -> None:
    registry = build_coding_affordance_registry(seal=False)
    assert registry.sealed is False
    # Caller can seal later.
    registry.seal()
    assert registry.sealed is True


def test_coding_descriptors_all_require_filesystem_read_consent() -> None:
    for descriptor in CODING_AFFORDANCE_DESCRIPTORS:
        assert CONSENT_FILESYSTEM_READ in descriptor.safety_model.requires_consent_grant


def test_coding_descriptors_block_in_social_regimes() -> None:
    for descriptor in CODING_AFFORDANCE_DESCRIPTORS:
        blocked = descriptor.safety_model.blocked_in_regimes
        assert "casual_social" in blocked
        assert "emotional_support" in blocked


# ---------------------------------------------------------------------------
# Invoker boundary gates
# ---------------------------------------------------------------------------


async def test_invoker_denies_without_consent(sandbox: pathlib.Path) -> None:
    invoker = build_coding_affordance_invoker(sandbox_root=sandbox)
    result = await invoker.invoke(
        "read_file",
        {"path": "README.md"},
        granted_consents=frozenset(),
        active_regime_id="problem_solving",
    )
    assert result.status is AffordanceInvocationStatus.DENIED_BY_BOUNDARY
    assert result.error_class == "consent_not_granted"
    assert "filesystem_read" in result.error_detail


async def test_invoker_denies_in_blocked_regime(sandbox: pathlib.Path) -> None:
    invoker = build_coding_affordance_invoker(sandbox_root=sandbox)
    result = await invoker.invoke(
        "grep",
        {"pattern": "def"},
        granted_consents=frozenset({CONSENT_FILESYSTEM_READ}),
        active_regime_id="casual_social",
    )
    assert result.status is AffordanceInvocationStatus.DENIED_BY_BOUNDARY
    assert result.error_class == "regime_blocked"


async def test_invoker_read_file_happy_path(sandbox: pathlib.Path) -> None:
    invoker = build_coding_affordance_invoker(sandbox_root=sandbox)
    result = await invoker.invoke(
        "read_file",
        {"path": "src/main.py"},
        granted_consents=frozenset({CONSENT_FILESYSTEM_READ}),
        active_regime_id="problem_solving",
    )
    assert result.status is AffordanceInvocationStatus.SUCCEEDED
    assert "hello world" in result.payload["content"]
    assert result.payload["truncated"] is False


async def test_invoker_grep_happy_path(sandbox: pathlib.Path) -> None:
    invoker = build_coding_affordance_invoker(sandbox_root=sandbox)
    result = await invoker.invoke(
        "grep",
        {"pattern": "helper", "subdir": "src"},
        granted_consents=frozenset({CONSENT_FILESYSTEM_READ}),
        active_regime_id="problem_solving",
    )
    assert result.status is AffordanceInvocationStatus.SUCCEEDED
    paths = {m["path"] for m in result.payload["matches"]}
    assert any("util.py" in p for p in paths)


async def test_invoker_list_dir_happy_path(sandbox: pathlib.Path) -> None:
    invoker = build_coding_affordance_invoker(sandbox_root=sandbox)
    result = await invoker.invoke(
        "list_dir",
        {"path": "src"},
        granted_consents=frozenset({CONSENT_FILESYSTEM_READ}),
        active_regime_id="problem_solving",
    )
    assert result.status is AffordanceInvocationStatus.SUCCEEDED
    names = {e["name"] for e in result.payload["entries"]}
    assert names == {"main.py", "util.py"}


async def test_invoker_param_invalid_rejected_before_backend(sandbox: pathlib.Path) -> None:
    invoker = build_coding_affordance_invoker(sandbox_root=sandbox)
    result = await invoker.invoke(
        "read_file",
        {"max_bytes": 10},  # missing required "path"
        granted_consents=frozenset({CONSENT_FILESYSTEM_READ}),
        active_regime_id="problem_solving",
    )
    assert result.status is AffordanceInvocationStatus.PARAMETER_INVALID
    assert "path" in result.error_detail


async def test_invoker_sandbox_escape_becomes_backend_failed(
    sandbox: pathlib.Path,
) -> None:
    """Sandbox escape is caught by the backend (not the descriptor
    schema), so it surfaces as BACKEND_FAILED with the SandboxPathError
    class name. The invocation reaches the backend; the sandbox
    guard inside the backend refuses the access.
    """
    invoker = build_coding_affordance_invoker(sandbox_root=sandbox)
    result = await invoker.invoke(
        "read_file",
        {"path": "../outside.txt"},
        granted_consents=frozenset({CONSENT_FILESYSTEM_READ}),
        active_regime_id="problem_solving",
    )
    assert result.status is AffordanceInvocationStatus.BACKEND_FAILED
    assert result.error_class == "SandboxPathError"


# ---------------------------------------------------------------------------
# End-to-end: real LifeformSession + invoker
# ---------------------------------------------------------------------------


async def test_coding_lifeform_with_invoker_reads_sandbox_file(
    sandbox: pathlib.Path,
) -> None:
    """Full path: build the coding lifeform, start a session, open
    a scene, invoke read_file with kernel wiring. The kernel's
    execution_result owner should pick up the tool event.
    """
    from lifeform_domain_coding import build_coding_lifeform

    lifeform = build_coding_lifeform()
    session = lifeform.create_session(session_id="coding-invoker-e2e")
    await session.run_turn("Please take a look at src/main.py.")

    invoker = build_coding_affordance_invoker(sandbox_root=sandbox)
    result = await invoker.invoke(
        "read_file",
        {"path": "src/main.py"},
        granted_consents=frozenset({CONSENT_FILESYSTEM_READ}),
        active_regime_id="problem_solving",
        session=session.brain_session,
        event_id="coding-e2e-read-1",
    )
    assert result.status is AffordanceInvocationStatus.SUCCEEDED
    assert "hello world" in result.payload["content"]
    assert result.tool_event_ids  # kernel saw the tool event

    # Run another turn so the kernel processes the queued tool event.
    await session.run_turn("What did you find?")
    execution_snap = session.latest_active_snapshots.get("execution_result")
    assert execution_snap is not None
    completed = execution_snap.value.completed_actions
    assert any(
        "coding-e2e-read-1" in record.record_id for record in completed
    ), (
        f"Expected 'coding-e2e-read-1' in execution_result.completed_actions; "
        f"got ids: {[r.record_id for r in completed]!r}"
    )
