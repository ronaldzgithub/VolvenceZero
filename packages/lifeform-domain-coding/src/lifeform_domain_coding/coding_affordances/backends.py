"""Coding vertical affordance backends (Gap 1 slice 2b).

Three read-only async backends, each scoped to a ``sandbox_root``
supplied at construction time. Safety protocol:

* **Path resolution** goes through ``resolve_sandbox_path`` which
  calls ``Path.resolve(strict=True)`` then checks the result
  is under ``sandbox_root.resolve(strict=True)``. Any path that
  resolves outside the sandbox \u2014 symlink or ``..`` escape \u2014
  raises ``SandboxPathError``. The invoker converts that exception
  into a ``BACKEND_FAILED`` status with ``error_class="SandboxPathError"``.
* **No subprocess spawning** in slice 2b. ``grep`` is a pure-Python
  recursive walk.
* **Byte / result budgets** are enforced so a pathological input
  cannot DoS the host.

Each backend returns a JSON-serialisable ``dict`` matching the
descriptor's ``output_schema``.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
import time
from collections.abc import Mapping
from typing import Any


class SandboxPathError(ValueError):
    """Raised when a requested path resolves outside the sandbox.

    Subclasses ``ValueError`` so callers that catch either "bad
    input" or "sandbox violation" can use a common handler. The
    message names the sandbox and the offending path so an
    operator can trace what happened.
    """


_DEFAULT_READ_MAX_BYTES: int = 65536
_DEFAULT_GREP_MAX_RESULTS: int = 200
# Hard cap \u2014 the caller cannot raise max_results beyond this even
# if they try, so a pathological pattern cannot walk the registry
# through millions of matches.
_HARD_GREP_MAX_RESULTS: int = 2000
# Hard cap on max_bytes too. A caller cannot bypass this.
_HARD_READ_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
# Files larger than this scanning threshold are skipped during grep.
# Prevents grep from slurping a 500MB log into memory.
_GREP_SKIP_LARGE_FILE_BYTES: int = 2 * 1024 * 1024  # 2 MB


def resolve_sandbox_path(
    requested_path: str,
    *,
    sandbox_root: pathlib.Path,
) -> pathlib.Path:
    """Resolve ``requested_path`` relative to ``sandbox_root``.

    Returns the fully-resolved ``Path`` when it is inside the
    sandbox; raises ``SandboxPathError`` otherwise. Symlinks are
    followed during resolution, so a symlink pointing outside
    the sandbox still triggers a rejection.

    Accepts both absolute paths (validated to be under the
    sandbox) and relative paths (joined to sandbox_root first).
    """
    if not requested_path:
        raise SandboxPathError(
            f"path must be non-empty; sandbox_root={sandbox_root!s}"
        )
    # Resolve sandbox_root strictly so symlinks in the root itself
    # are evaluated once and compared against consistently.
    resolved_root = sandbox_root.resolve(strict=True)
    candidate_raw = pathlib.Path(requested_path)
    if candidate_raw.is_absolute():
        candidate = candidate_raw
    else:
        candidate = resolved_root / candidate_raw
    # Two-phase resolution:
    # 1. Non-strict resolve normalises ``..`` segments even for
    #    non-existent last components, so the containment check
    #    rejects attempts to escape the sandbox BEFORE we reveal
    #    whether the outside path exists.
    # 2. Strict resolve follows symlinks, which the second
    #    containment check then validates \u2014 a symlink inside the
    #    sandbox pointing outside is rejected here.
    lenient_resolved = candidate.resolve(strict=False)
    try:
        lenient_resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise SandboxPathError(
            f"path {requested_path!r} resolves to {lenient_resolved!s}, "
            f"which is outside sandbox {resolved_root!s}"
        ) from exc
    if not lenient_resolved.exists():
        raise SandboxPathError(
            f"path {requested_path!r} does not exist under sandbox {resolved_root!s}"
        )
    strict_resolved = lenient_resolved.resolve(strict=True)
    try:
        strict_resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise SandboxPathError(
            f"path {requested_path!r} (via symlink) resolves to "
            f"{strict_resolved!s}, which is outside sandbox {resolved_root!s}"
        ) from exc
    return strict_resolved


async def _read_file_backend(
    parameters: Mapping[str, Any],
    *,
    sandbox_root: pathlib.Path,
) -> Mapping[str, Any]:
    """Read a UTF-8 text file under the sandbox.

    Returns ``content`` truncated to ``max_bytes`` (default 64 KB,
    hard-capped at 10 MB). Non-text files (decode error) raise
    ``UnicodeDecodeError`` which the invoker surfaces as
    ``BACKEND_FAILED``.
    """
    path_str = str(parameters.get("path", ""))
    max_bytes_raw = parameters.get("max_bytes", _DEFAULT_READ_MAX_BYTES)
    # max_bytes validation matches the descriptor schema type but
    # still clamps to a sane range; the schema allows any integer.
    if not isinstance(max_bytes_raw, int) or isinstance(max_bytes_raw, bool):
        raise SandboxPathError(
            f"max_bytes must be an integer, got {type(max_bytes_raw).__name__}"
        )
    max_bytes = min(max(1, max_bytes_raw), _HARD_READ_MAX_BYTES)
    resolved = resolve_sandbox_path(path_str, sandbox_root=sandbox_root)
    if not resolved.is_file():
        raise SandboxPathError(
            f"path {path_str!r} resolves to {resolved!s} which is not a regular file"
        )
    raw = resolved.read_bytes()
    truncated = len(raw) > max_bytes
    content = raw[:max_bytes].decode("utf-8")
    return {
        "content": content,
        "truncated": truncated,
        "byte_count": len(raw),
        "resolved_path": str(resolved),
    }


async def _list_dir_backend(
    parameters: Mapping[str, Any],
    *,
    sandbox_root: pathlib.Path,
) -> Mapping[str, Any]:
    """List direct children of ``path`` under the sandbox.

    Returns ``entries`` ordered by name (case-sensitive), each
    tagged with ``kind`` in ``{"file", "dir", "other"}``. Hidden
    entries (starting with ``.``) are included; the caller chooses
    whether to ignore them.
    """
    path_str = str(parameters.get("path", ""))
    resolved = resolve_sandbox_path(path_str, sandbox_root=sandbox_root)
    if not resolved.is_dir():
        raise SandboxPathError(
            f"path {path_str!r} resolves to {resolved!s} which is not a directory"
        )
    entries = []
    for child in sorted(resolved.iterdir(), key=lambda p: p.name):
        if child.is_dir():
            kind = "dir"
        elif child.is_file():
            kind = "file"
        else:
            kind = "other"
        entries.append({"name": child.name, "kind": kind})
    return {"entries": entries, "resolved_path": str(resolved)}


async def _grep_backend(
    parameters: Mapping[str, Any],
    *,
    sandbox_root: pathlib.Path,
) -> Mapping[str, Any]:
    """Pure-Python case-sensitive grep across text files in the sandbox.

    Skips files larger than 2 MB (binary or log dump); skips
    files that fail UTF-8 decode. Honours ``subdir`` for scoping
    and ``max_results`` for budget (hard-capped at 2000).
    """
    pattern = str(parameters.get("pattern", ""))
    if not pattern:
        raise SandboxPathError("grep pattern must be non-empty")
    subdir = str(parameters.get("subdir", "."))
    max_results_raw = parameters.get("max_results", _DEFAULT_GREP_MAX_RESULTS)
    if not isinstance(max_results_raw, int) or isinstance(max_results_raw, bool):
        raise SandboxPathError(
            f"max_results must be an integer, got {type(max_results_raw).__name__}"
        )
    max_results = min(max(1, max_results_raw), _HARD_GREP_MAX_RESULTS)
    scope = resolve_sandbox_path(subdir, sandbox_root=sandbox_root)
    if not scope.is_dir():
        raise SandboxPathError(
            f"subdir {subdir!r} resolves to {scope!s} which is not a directory"
        )
    matches: list[dict[str, Any]] = []
    scanned = 0
    truncated = False
    for candidate in sorted(scope.rglob("*")):
        if not candidate.is_file():
            continue
        try:
            size = candidate.stat().st_size
        except OSError:
            continue
        if size > _GREP_SKIP_LARGE_FILE_BYTES:
            continue
        scanned += 1
        try:
            text = candidate.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if pattern in line:
                matches.append(
                    {
                        "path": str(candidate),
                        "line_number": line_number,
                        "line": line[:320],
                    }
                )
                if len(matches) >= max_results:
                    truncated = True
                    break
        if truncated:
            break
    return {
        "matches": matches,
        "truncated": truncated,
        "scanned_files": scanned,
    }


_VALID_WRITE_MODES: frozenset[str] = frozenset({"create", "overwrite", "append"})
_HARD_WRITE_CONTENT_BYTES: int = 10 * 1024 * 1024  # 10 MB hard cap per call
_DEFAULT_RUN_TEST_SECONDS: int = 30
_MAX_RUN_TEST_SECONDS: int = 300
_RUN_TEST_OUTPUT_MAX_BYTES: int = 64 * 1024  # 64 KB per stream, truncated


def _resolve_parent_under_sandbox(
    path_str: str, *, sandbox_root: pathlib.Path
) -> pathlib.Path:
    """Resolve a target path that may NOT exist yet (for write_file).

    Sibling of ``resolve_sandbox_path`` but for write paths where
    the leaf file is allowed to be absent. The parent directory
    MUST exist (we do not silently mkdir \u2014 that would let a
    mis-typed path create unexpected subtrees) and the resolved
    location must still sit inside the sandbox.
    """
    if not path_str:
        raise SandboxPathError(
            f"path must be non-empty; sandbox_root={sandbox_root!s}"
        )
    resolved_root = sandbox_root.resolve(strict=True)
    candidate_raw = pathlib.Path(path_str)
    if candidate_raw.is_absolute():
        candidate = candidate_raw
    else:
        candidate = resolved_root / candidate_raw
    # Non-strict resolve normalises ``..`` segments even for
    # non-existent last components \u2014 the containment check rejects
    # escape before the existence check fires.
    lenient = candidate.resolve(strict=False)
    try:
        lenient.relative_to(resolved_root)
    except ValueError as exc:
        raise SandboxPathError(
            f"path {path_str!r} resolves to {lenient!s}, "
            f"which is outside sandbox {resolved_root!s}"
        ) from exc
    parent = lenient.parent
    if not parent.exists():
        raise SandboxPathError(
            f"parent directory {parent!s} does not exist; "
            f"write_file does not auto-create directories"
        )
    if not parent.is_dir():
        raise SandboxPathError(
            f"parent {parent!s} is not a directory"
        )
    # Final containment check against the STRICT-resolved parent
    # (so a symlink ``parent -> outside`` is rejected).
    strict_parent = parent.resolve(strict=True)
    try:
        strict_parent.relative_to(resolved_root)
    except ValueError as exc:
        raise SandboxPathError(
            f"write target's parent {strict_parent!s} resolves outside "
            f"sandbox {resolved_root!s}"
        ) from exc
    return strict_parent / lenient.name


async def _write_file_backend(
    parameters: Mapping[str, Any],
    *,
    sandbox_root: pathlib.Path,
) -> Mapping[str, Any]:
    """Write a UTF-8 text file under the sandbox.

    Modes:

    * ``create`` \u2014 target must NOT already exist. Fails loudly
      on collision to protect accidental overwrites.
    * ``overwrite`` \u2014 replace existing content; target may or
      may not exist.
    * ``append`` \u2014 target must exist; content is appended.

    Returns a dict matching the descriptor output_schema. The
    invocation is irreversible from the backend's perspective; the
    host is responsible for the user-confirmation gate upstream.
    """
    path_str = str(parameters.get("path", ""))
    content = parameters.get("content", "")
    if not isinstance(content, str):
        raise SandboxPathError(
            f"write_file.content must be a string, got {type(content).__name__}"
        )
    mode = str(parameters.get("mode", "create"))
    if mode not in _VALID_WRITE_MODES:
        raise SandboxPathError(
            f"write_file.mode must be one of {sorted(_VALID_WRITE_MODES)!r}, "
            f"got {mode!r}"
        )
    encoded = content.encode("utf-8")
    if len(encoded) > _HARD_WRITE_CONTENT_BYTES:
        raise SandboxPathError(
            f"write_file.content exceeds hard cap of "
            f"{_HARD_WRITE_CONTENT_BYTES} bytes "
            f"(got {len(encoded)}); split into multiple smaller writes"
        )
    resolved = _resolve_parent_under_sandbox(path_str, sandbox_root=sandbox_root)
    already_exists = resolved.exists()
    if already_exists and resolved.is_dir():
        raise SandboxPathError(
            f"write target {resolved!s} is a directory, not a file"
        )
    if mode == "create" and already_exists:
        raise SandboxPathError(
            f"write_file mode='create' but {resolved!s} already exists; "
            f"use mode='overwrite' to replace it"
        )
    if mode == "append" and not already_exists:
        raise SandboxPathError(
            f"write_file mode='append' but {resolved!s} does not exist; "
            f"use mode='create' first"
        )
    open_mode = {
        "create": "wb",
        "overwrite": "wb",
        "append": "ab",
    }[mode]
    with resolved.open(open_mode) as handle:
        handle.write(encoded)
    final_size = resolved.stat().st_size
    return {
        "resolved_path": str(resolved),
        "mode": mode,
        "bytes_written": len(encoded),
        "final_size_bytes": final_size,
        "created": not already_exists,
    }


def _bounded_output(raw: bytes, *, max_bytes: int) -> tuple[str, bool]:
    """Decode ``raw`` (at most ``max_bytes``) as UTF-8 with replacement.

    Returns ``(decoded, truncated)``. Output truncation is silent on
    the wire (caller appends a marker); subprocesses often produce
    very long logs and we don't want to OOM the kernel tool bus.
    """
    truncated = len(raw) > max_bytes
    body = raw[:max_bytes].decode("utf-8", errors="replace")
    if truncated:
        body = body + "\n... [output truncated] ..."
    return body, truncated


async def _run_test_backend(
    parameters: Mapping[str, Any],
    *,
    sandbox_root: pathlib.Path,
) -> Mapping[str, Any]:
    """Run pytest against a single target inside the sandbox.

    Implementation:

    * Target path is resolved through the sandbox guard so a
      caller cannot run tests outside the workspace.
    * Subprocess is launched via ``sys.executable -m pytest -q``
      so pytest doesn't need to be on PATH; ``cwd=sandbox_root``
      so relative-path fixtures behave as expected.
    * Timeout uses ``asyncio.wait_for``; on timeout the process is
      terminated (``proc.kill()``) and we still return a structured
      result with ``timed_out=True`` and whatever output was
      captured before the kill.
    * stdout + stderr are each bounded to 64 KB; above that the
      tail is replaced with ``... [output truncated] ...``.
    """
    test_path_str = str(parameters.get("test_path", ""))
    if not test_path_str:
        raise SandboxPathError("run_test.test_path must be non-empty")
    max_seconds_raw = parameters.get("max_seconds", _DEFAULT_RUN_TEST_SECONDS)
    if not isinstance(max_seconds_raw, int) or isinstance(max_seconds_raw, bool):
        raise SandboxPathError(
            f"run_test.max_seconds must be an integer, got "
            f"{type(max_seconds_raw).__name__}"
        )
    max_seconds = min(max(1, max_seconds_raw), _MAX_RUN_TEST_SECONDS)
    # pytest accepts "path::node_id" selectors; split off the node id
    # so the sandbox check only validates the file portion.
    file_portion = test_path_str.split("::", 1)[0]
    resolved_file = resolve_sandbox_path(file_portion, sandbox_root=sandbox_root)
    # Reattach node id (if any) to the resolved file for the pytest arg.
    node_suffix = test_path_str[len(file_portion):]
    pytest_arg = str(resolved_file) + node_suffix
    started_at = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "pytest",
        "-q",
        pytest_arg,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(sandbox_root),
    )
    timed_out = False
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=max_seconds
        )
        exit_code = proc.returncode if proc.returncode is not None else -1
    except asyncio.TimeoutError:
        timed_out = True
        # Fire-and-forget kill + collect whatever we have. On Windows
        # ``proc.kill()`` terminates the process immediately; on POSIX
        # it sends SIGKILL.
        try:
            proc.kill()
        except ProcessLookupError:
            # Already exited between the timeout firing and kill.
            pass
        # Drain the pipes so the fds don't leak; give communicate a
        # short grace period now that the process is signalled.
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=5.0
            )
        except asyncio.TimeoutError:
            stdout_bytes, stderr_bytes = b"", b""
        exit_code = proc.returncode if proc.returncode is not None else -9
    duration = round(time.monotonic() - started_at, 3)
    stdout_str, _ = _bounded_output(stdout_bytes, max_bytes=_RUN_TEST_OUTPUT_MAX_BYTES)
    stderr_str, _ = _bounded_output(stderr_bytes, max_bytes=_RUN_TEST_OUTPUT_MAX_BYTES)
    return {
        "exit_code": int(exit_code),
        "stdout": stdout_str,
        "stderr": stderr_str,
        "duration_seconds": duration,
        "timed_out": timed_out,
        "resolved_path": str(resolved_file),
    }


def build_coding_affordance_backends(
    sandbox_root: pathlib.Path | str,
) -> dict[str, Any]:
    """Return a ``dict`` mapping descriptor name \u2192 async backend.

    All backends close over ``sandbox_root`` so each is bound to a
    specific workspace. Resolving the root once at construction
    time (``strict=True``) fails loud on a non-existent / inaccessible
    sandbox before any invocation.
    """
    root = pathlib.Path(sandbox_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(
            f"sandbox_root {sandbox_root!s} resolves to {root!s} which is not a directory"
        )

    async def read_file(parameters):
        return await _read_file_backend(parameters, sandbox_root=root)

    async def list_dir(parameters):
        return await _list_dir_backend(parameters, sandbox_root=root)

    async def grep(parameters):
        return await _grep_backend(parameters, sandbox_root=root)

    async def write_file(parameters):
        return await _write_file_backend(parameters, sandbox_root=root)

    async def run_test(parameters):
        return await _run_test_backend(parameters, sandbox_root=root)

    return {
        "read_file": read_file,
        "list_dir": list_dir,
        "grep": grep,
        "write_file": write_file,
        "run_test": run_test,
    }


__all__ = [
    "SandboxPathError",
    "build_coding_affordance_backends",
    "resolve_sandbox_path",
]
