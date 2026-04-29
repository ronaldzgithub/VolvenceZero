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

import pathlib
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

    return {"read_file": read_file, "list_dir": list_dir, "grep": grep}


__all__ = [
    "SandboxPathError",
    "build_coding_affordance_backends",
    "resolve_sandbox_path",
]
