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
    CONSENT_FILESYSTEM_WRITE,
    CONSENT_RUN_SHELL_COMMANDS,
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


def test_build_coding_registry_contains_all_descriptors() -> None:
    registry = build_coding_affordance_registry()
    # slice 2b: read_file, list_dir, grep
    # slice 2c: write_file, run_test
    assert set(registry.names()) == {
        "read_file",
        "list_dir",
        "grep",
        "write_file",
        "run_test",
    }
    assert registry.sealed is True


def test_build_coding_registry_allows_extension_when_seal_false() -> None:
    registry = build_coding_affordance_registry(seal=False)
    assert registry.sealed is False
    # Caller can seal later.
    registry.seal()
    assert registry.sealed is True


def test_coding_descriptors_all_require_a_consent_grant() -> None:
    """Every coding affordance must gate on SOME consent grant.

    Read-only tools gate on ``filesystem_read``; write_file gates on
    ``filesystem_write``; run_test gates on ``run_shell_commands``.
    The invariant here is "never ungated", not "all share one grant".
    """
    for descriptor in CODING_AFFORDANCE_DESCRIPTORS:
        assert descriptor.safety_model.requires_consent_grant, (
            f"{descriptor.name!r} must declare at least one consent grant"
        )


def test_read_only_descriptors_require_filesystem_read() -> None:
    """The three slice-2b read-only tools share ``filesystem_read``."""
    registry = build_coding_affordance_registry()
    for name in ("read_file", "list_dir", "grep"):
        desc = registry.get(name)
        assert CONSENT_FILESYSTEM_READ in desc.safety_model.requires_consent_grant


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


# ===========================================================================
# Gap 1 slice 2c: write_file + run_test
# ===========================================================================


# ---------------------------------------------------------------------------
# write_file backend
# ---------------------------------------------------------------------------


def test_write_file_create_new_file(sandbox: pathlib.Path) -> None:
    backends = build_coding_affordance_backends(sandbox)
    result = asyncio.run(
        backends["write_file"]({
            "path": "src/new.py",
            "content": "print('new')\n",
            "mode": "create",
        })
    )
    target = sandbox / "src" / "new.py"
    assert target.is_file()
    assert target.read_text(encoding="utf-8") == "print('new')\n"
    assert result["mode"] == "create"
    assert result["created"] is True
    assert result["bytes_written"] == len(b"print('new')\n")


def test_write_file_create_rejects_existing(sandbox: pathlib.Path) -> None:
    backends = build_coding_affordance_backends(sandbox)
    with pytest.raises(SandboxPathError, match="already exists"):
        asyncio.run(backends["write_file"]({
            "path": "src/main.py",  # exists in the fixture
            "content": "overwrite attempt",
            "mode": "create",
        }))


def test_write_file_overwrite_replaces_content(sandbox: pathlib.Path) -> None:
    backends = build_coding_affordance_backends(sandbox)
    result = asyncio.run(backends["write_file"]({
        "path": "src/main.py",
        "content": "replaced\n",
        "mode": "overwrite",
    }))
    assert (sandbox / "src" / "main.py").read_text(encoding="utf-8") == "replaced\n"
    assert result["mode"] == "overwrite"
    assert result["created"] is False


def test_write_file_append_requires_existing(sandbox: pathlib.Path) -> None:
    backends = build_coding_affordance_backends(sandbox)
    with pytest.raises(SandboxPathError, match="does not exist"):
        asyncio.run(backends["write_file"]({
            "path": "src/nonexistent.py",
            "content": "x",
            "mode": "append",
        }))


def test_write_file_append_extends_existing(sandbox: pathlib.Path) -> None:
    backends = build_coding_affordance_backends(sandbox)
    before = (sandbox / "src" / "main.py").read_text(encoding="utf-8")
    asyncio.run(backends["write_file"]({
        "path": "src/main.py",
        "content": "# APPENDED\n",
        "mode": "append",
    }))
    after = (sandbox / "src" / "main.py").read_text(encoding="utf-8")
    assert after == before + "# APPENDED\n"


def test_write_file_rejects_unknown_mode(sandbox: pathlib.Path) -> None:
    backends = build_coding_affordance_backends(sandbox)
    with pytest.raises(SandboxPathError, match="mode must be one of"):
        asyncio.run(backends["write_file"]({
            "path": "src/new.py",
            "content": "x",
            "mode": "bogus",
        }))


def test_write_file_rejects_parent_escape(sandbox: pathlib.Path) -> None:
    backends = build_coding_affordance_backends(sandbox)
    with pytest.raises(SandboxPathError, match="outside sandbox"):
        asyncio.run(backends["write_file"]({
            "path": "../escape.py",
            "content": "x",
            "mode": "create",
        }))


def test_write_file_rejects_missing_parent_directory(sandbox: pathlib.Path) -> None:
    backends = build_coding_affordance_backends(sandbox)
    with pytest.raises(SandboxPathError, match="parent directory"):
        asyncio.run(backends["write_file"]({
            "path": "nonexistent_dir/new.py",
            "content": "x",
            "mode": "create",
        }))


def test_write_file_rejects_directory_target(sandbox: pathlib.Path) -> None:
    backends = build_coding_affordance_backends(sandbox)
    with pytest.raises(SandboxPathError, match="is a directory"):
        asyncio.run(backends["write_file"]({
            "path": "src",  # directory, not a file
            "content": "x",
            "mode": "overwrite",
        }))


def test_write_file_rejects_content_over_hard_cap(sandbox: pathlib.Path) -> None:
    backends = build_coding_affordance_backends(sandbox)
    huge = "x" * (10 * 1024 * 1024 + 1)
    with pytest.raises(SandboxPathError, match="hard cap"):
        asyncio.run(backends["write_file"]({
            "path": "src/huge.py",
            "content": huge,
            "mode": "create",
        }))


def test_write_file_rejects_non_string_content(sandbox: pathlib.Path) -> None:
    backends = build_coding_affordance_backends(sandbox)
    with pytest.raises(SandboxPathError, match="must be a string"):
        asyncio.run(backends["write_file"]({
            "path": "src/x.py",
            "content": 123,  # not a string
            "mode": "create",
        }))


# ---------------------------------------------------------------------------
# write_file descriptor safety flags
# ---------------------------------------------------------------------------


def test_write_file_descriptor_safety_flags_are_strict() -> None:
    registry = build_coding_affordance_registry()
    descriptor = registry.get("write_file")
    assert descriptor.safety_model.requires_user_confirmation is True
    assert descriptor.safety_model.irreversible is True
    assert descriptor.safety_model.audit_required is True
    assert CONSENT_FILESYSTEM_WRITE in descriptor.safety_model.requires_consent_grant
    assert CONSENT_FILESYSTEM_READ not in descriptor.safety_model.requires_consent_grant


# ---------------------------------------------------------------------------
# write_file via the invoker (consent + confirmation gates)
# ---------------------------------------------------------------------------


async def test_invoker_write_file_denied_without_write_consent(
    sandbox: pathlib.Path,
) -> None:
    invoker = build_coding_affordance_invoker(sandbox_root=sandbox)
    # Caller has READ consent but NOT write consent.
    result = await invoker.invoke(
        "write_file",
        {"path": "src/new.py", "content": "x", "mode": "create"},
        granted_consents=frozenset({CONSENT_FILESYSTEM_READ}),
        active_regime_id="problem_solving",
        user_confirmed=True,
    )
    assert result.status is AffordanceInvocationStatus.DENIED_BY_BOUNDARY
    assert result.error_class == "consent_not_granted"
    assert "filesystem_write" in result.error_detail


async def test_invoker_write_file_denied_without_confirmation(
    sandbox: pathlib.Path,
) -> None:
    invoker = build_coding_affordance_invoker(sandbox_root=sandbox)
    result = await invoker.invoke(
        "write_file",
        {"path": "src/new.py", "content": "x", "mode": "create"},
        granted_consents=frozenset({CONSENT_FILESYSTEM_WRITE}),
        active_regime_id="problem_solving",
        user_confirmed=False,
    )
    assert result.status is AffordanceInvocationStatus.DENIED_BY_BOUNDARY
    assert result.error_class == "confirmation_required"


async def test_invoker_write_file_success_with_confirmation_and_consent(
    sandbox: pathlib.Path,
) -> None:
    invoker = build_coding_affordance_invoker(sandbox_root=sandbox)
    result = await invoker.invoke(
        "write_file",
        {"path": "src/new.py", "content": "hello\n", "mode": "create"},
        granted_consents=frozenset({CONSENT_FILESYSTEM_WRITE}),
        active_regime_id="problem_solving",
        user_confirmed=True,
    )
    assert result.status is AffordanceInvocationStatus.SUCCEEDED
    assert (sandbox / "src" / "new.py").read_text(encoding="utf-8") == "hello\n"
    assert result.payload["bytes_written"] == len(b"hello\n")
    assert result.payload["mode"] == "create"


# ---------------------------------------------------------------------------
# run_test backend (real subprocess)
# ---------------------------------------------------------------------------


@pytest.fixture
def runnable_test_sandbox(tmp_path: pathlib.Path) -> pathlib.Path:
    """A sandbox containing a passing + a failing pytest module.

    We deliberately write TINY tests so the subprocess returns
    quickly; the full pytest startup is usually a second or two.
    """
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_pass.py").write_text(
        "def test_ok():\n    assert 1 + 1 == 2\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_fail.py").write_text(
        "def test_bad():\n    assert 1 + 1 == 3\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_slow.py").write_text(
        "import time\n\n\ndef test_slow():\n    time.sleep(30)\n",
        encoding="utf-8",
    )
    return tmp_path


def test_run_test_passing_target_returns_exit_zero(
    runnable_test_sandbox: pathlib.Path,
) -> None:
    backends = build_coding_affordance_backends(runnable_test_sandbox)
    result = asyncio.run(
        backends["run_test"]({"test_path": "tests/test_pass.py"})
    )
    assert result["exit_code"] == 0
    assert result["timed_out"] is False
    assert "1 passed" in result["stdout"] or "1 passed" in result["stderr"]


def test_run_test_failing_target_returns_nonzero(
    runnable_test_sandbox: pathlib.Path,
) -> None:
    backends = build_coding_affordance_backends(runnable_test_sandbox)
    result = asyncio.run(
        backends["run_test"]({"test_path": "tests/test_fail.py"})
    )
    assert result["exit_code"] != 0
    assert result["timed_out"] is False
    # pytest's assertion diff mentions the assert line.
    combined = result["stdout"] + result["stderr"]
    assert "1 failed" in combined or "assert" in combined


def test_run_test_timeout_triggers_timed_out_flag(
    runnable_test_sandbox: pathlib.Path,
) -> None:
    backends = build_coding_affordance_backends(runnable_test_sandbox)
    result = asyncio.run(
        backends["run_test"]({
            "test_path": "tests/test_slow.py",
            "max_seconds": 2,  # slow test sleeps 30s; must time out
        })
    )
    assert result["timed_out"] is True
    assert result["duration_seconds"] < 10.0  # killed quickly
    # Exit code is non-zero (killed) when timed out.
    assert result["exit_code"] != 0


def test_run_test_rejects_path_outside_sandbox(
    runnable_test_sandbox: pathlib.Path,
) -> None:
    backends = build_coding_affordance_backends(runnable_test_sandbox)
    with pytest.raises(SandboxPathError, match="outside sandbox"):
        asyncio.run(backends["run_test"]({
            "test_path": "../sneaky.py",
        }))


def test_run_test_rejects_empty_path(
    runnable_test_sandbox: pathlib.Path,
) -> None:
    backends = build_coding_affordance_backends(runnable_test_sandbox)
    with pytest.raises(SandboxPathError, match="test_path must be non-empty"):
        asyncio.run(backends["run_test"]({"test_path": ""}))


def test_run_test_rejects_non_int_max_seconds(
    runnable_test_sandbox: pathlib.Path,
) -> None:
    backends = build_coding_affordance_backends(runnable_test_sandbox)
    with pytest.raises(SandboxPathError, match="max_seconds"):
        asyncio.run(backends["run_test"]({
            "test_path": "tests/test_pass.py",
            "max_seconds": "20",
        }))


# ---------------------------------------------------------------------------
# run_test descriptor + invoker
# ---------------------------------------------------------------------------


def test_run_test_descriptor_has_rate_limit_and_audit() -> None:
    registry = build_coding_affordance_registry()
    descriptor = registry.get("run_test")
    assert descriptor.cost_model.rate_limit_per_minute == 6
    assert descriptor.safety_model.audit_required is True
    assert CONSENT_RUN_SHELL_COMMANDS in descriptor.safety_model.requires_consent_grant


async def test_invoker_run_test_denied_without_shell_consent(
    runnable_test_sandbox: pathlib.Path,
) -> None:
    invoker = build_coding_affordance_invoker(sandbox_root=runnable_test_sandbox)
    result = await invoker.invoke(
        "run_test",
        {"test_path": "tests/test_pass.py"},
        granted_consents=frozenset({CONSENT_FILESYSTEM_WRITE}),  # wrong grant
        active_regime_id="problem_solving",
    )
    assert result.status is AffordanceInvocationStatus.DENIED_BY_BOUNDARY
    assert "run_shell_commands" in result.error_detail


async def test_invoker_run_test_rate_limit_fires_after_budget(
    runnable_test_sandbox: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_test has rate_limit_per_minute=6. Invoke 7 times in
    monotonic-clock window -> the 7th fires RATE_LIMITED.

    We DO NOT want to actually invoke pytest 7 times (slow). We
    use a stub backend that returns immediately so the rate
    limiter is exercised in isolation.
    """
    invoker = build_coding_affordance_invoker(sandbox_root=runnable_test_sandbox)

    async def _fast_stub(_parameters):
        return {
            "exit_code": 0,
            "stdout": "stub",
            "stderr": "",
            "duration_seconds": 0.0,
            "timed_out": False,
            "resolved_path": "stub",
        }

    invoker.register_backend("run_test", _fast_stub)
    for _ in range(6):
        result = await invoker.invoke(
            "run_test",
            {"test_path": "tests/test_pass.py"},
            granted_consents=frozenset({CONSENT_RUN_SHELL_COMMANDS}),
            active_regime_id="problem_solving",
        )
        assert result.status is AffordanceInvocationStatus.SUCCEEDED
    # 7th call in the same minute window should fire rate limit.
    result = await invoker.invoke(
        "run_test",
        {"test_path": "tests/test_pass.py"},
        granted_consents=frozenset({CONSENT_RUN_SHELL_COMMANDS}),
        active_regime_id="problem_solving",
    )
    assert result.status is AffordanceInvocationStatus.RATE_LIMITED
    assert result.error_class == "rate_limited"


# ---------------------------------------------------------------------------
# End-to-end: real LifeformSession + write_file + run_test
# ---------------------------------------------------------------------------


async def test_coding_lifeform_end_to_end_write_then_run(
    runnable_test_sandbox: pathlib.Path,
) -> None:
    """Complete coding flow: session open -> write a new test
    -> run pytest on it -> both tool events land on the kernel
    execution_result bus as succeeded.

    This is the Gap 1 slice 2c proof-of-life: coding vertical
    can actually modify the workspace AND verify its own work.
    """
    from lifeform_domain_coding import build_coding_lifeform

    lifeform = build_coding_lifeform()
    session = lifeform.create_session(session_id="coding-e2e-write-run")
    await session.run_turn("Let me add a new test.")

    invoker = build_coding_affordance_invoker(sandbox_root=runnable_test_sandbox)

    # Step 1: write a brand-new passing test file.
    write_result = await invoker.invoke(
        "write_file",
        {
            "path": "tests/test_added.py",
            "content": "def test_added():\n    assert True\n",
            "mode": "create",
        },
        granted_consents=frozenset({CONSENT_FILESYSTEM_WRITE}),
        active_regime_id="problem_solving",
        user_confirmed=True,
        session=session.brain_session,
        event_id="e2e-write-1",
    )
    assert write_result.status is AffordanceInvocationStatus.SUCCEEDED
    assert write_result.tool_event_ids
    assert (runnable_test_sandbox / "tests" / "test_added.py").is_file()

    # Step 2: run pytest on the file we just wrote.
    run_result = await invoker.invoke(
        "run_test",
        {"test_path": "tests/test_added.py"},
        granted_consents=frozenset({CONSENT_RUN_SHELL_COMMANDS}),
        active_regime_id="problem_solving",
        session=session.brain_session,
        event_id="e2e-run-1",
    )
    assert run_result.status is AffordanceInvocationStatus.SUCCEEDED, (
        f"run_test failed: stdout={run_result.payload and run_result.payload.get('stdout')!r}"
    )
    assert run_result.payload["exit_code"] == 0
    assert run_result.payload["timed_out"] is False
    assert run_result.tool_event_ids

    # Step 3: next turn -- kernel execution_result owner should
    # see BOTH event ids as completed actions.
    await session.run_turn("How did the tests go?")
    execution_snap = session.latest_active_snapshots.get("execution_result")
    assert execution_snap is not None
    completed_ids = [r.record_id for r in execution_snap.value.completed_actions]
    assert any("e2e-write-1" in rid for rid in completed_ids), (
        f"write event missing from execution_result; ids={completed_ids!r}"
    )
    assert any("e2e-run-1" in rid for rid in completed_ids), (
        f"run_test event missing from execution_result; ids={completed_ids!r}"
    )
