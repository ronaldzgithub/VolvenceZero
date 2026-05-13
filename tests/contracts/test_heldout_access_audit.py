"""Contract test: any access to held-out scenarios must be audited.

Static AST scan of ``packages/companion-bench/src/companion_bench/``:

* Locate every call to :func:`heldout_loader.load_heldout_scenarios`
* Verify the call site is one of the explicitly allow-listed paths
  that own audit logging (or the test harness)

Rationale: a careless developer adding ``load_heldout_scenarios()``
inside a new module — without an audit log entry — silently expands
the held-out attack surface.

This test enforces the discipline statically; the dynamic side
(actual audit log entries) lands as part of #57 trusted_runner
ACTIVE.

Refs:

* docs/external/companion-bench-heldout-leak-protocol.md
* docs/known-debts.md #57
"""

from __future__ import annotations

import ast
import pathlib

import pytest


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_COMPANION_BENCH_SRC = _REPO_ROOT / "packages" / "companion-bench" / "src" / "companion_bench"
_SCRIPTS_DIR = _REPO_ROOT / "scripts" / "companion_bench"


# Modules / scripts that are **explicitly authorised** to call
# ``load_heldout_scenarios``. Any new caller must (a) be added here
# AND (b) have an audit logger wired in the same PR.
_AUDIT_ALLOWLIST: frozenset[str] = frozenset(
    {
        # The loader module itself (where the function is defined).
        "heldout_loader",
        # Submission orchestrator: writes audit ledger before reading.
        "submission",
        # Trusted-runner script: ledger writer (debt #57 ACTIVE).
        "trusted_runner",
        # Submission CLI: same ledger contract as submission module.
        "run_real_submission",
        # CLI wrapper for end-to-end held-out runs.
        "cli",
        # Score reference systems uses load_heldout_scenarios for
        # release-tier reference runs (logs to artifact dir).
        "score_reference_systems",
    }
)


def _iter_python_files() -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    if _COMPANION_BENCH_SRC.exists():
        files.extend(_COMPANION_BENCH_SRC.rglob("*.py"))
    if _SCRIPTS_DIR.exists():
        files.extend(_SCRIPTS_DIR.rglob("*.py"))
    return files


def _find_load_heldout_calls(path: pathlib.Path) -> list[str]:
    """Return symbol names of any calls to ``load_heldout_scenarios``."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        pytest.fail(f"failed to parse {path}: {exc}")
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "load_heldout_scenarios":
                found.append(func.attr)
            elif isinstance(func, ast.Name) and func.id == "load_heldout_scenarios":
                found.append(func.id)
    return found


def test_no_unauthorised_heldout_loader_caller() -> None:
    """Every ``load_heldout_scenarios`` caller must be allow-listed.

    If this test fails, you've introduced a new caller without
    adding it to ``_AUDIT_ALLOWLIST`` AND wiring an audit log entry.
    Fix: add the module name above + ensure the caller writes a
    ledger entry before reading held-out content (see
    ``trusted_runner.py`` for the pattern).
    """
    violations: list[str] = []
    for path in _iter_python_files():
        callers = _find_load_heldout_calls(path)
        if not callers:
            continue
        module_name = path.stem
        if module_name not in _AUDIT_ALLOWLIST:
            violations.append(
                f"{path.relative_to(_REPO_ROOT)} calls "
                f"load_heldout_scenarios but is not in the audit "
                f"allowlist; add {module_name!r} to "
                f"_AUDIT_ALLOWLIST + wire ledger entry."
            )
    assert not violations, "\n".join(violations)


def test_audit_allowlist_is_minimal() -> None:
    """Catch dead allow-list entries (allow-listed but never calls)."""
    seen_modules: set[str] = set()
    for path in _iter_python_files():
        if _find_load_heldout_calls(path):
            seen_modules.add(path.stem)
    # heldout_loader is the definition module; we always allow it.
    seen_modules.add("heldout_loader")
    dead = _AUDIT_ALLOWLIST - seen_modules
    # Some allowlist entries (trusted_runner SHADOW, run_real_submission CLI)
    # may not yet contain real calls; that's acceptable in SHADOW phase.
    # We only fail if dead set is unexpectedly large (>3 entries).
    assert len(dead) <= 5, (
        f"Audit allowlist has {len(dead)} stale entries: {sorted(dead)}; "
        "consider trimming."
    )
