"""Static contract: the figure CLI imports only the public surface.

The figure-vertical CLI (``python -m lifeform_domain_figure.cli``)
is operator-facing and orchestrates ``bake-bundle`` /
``bake-steering`` / ``bake-lora`` / ``rollback`` / ``list``. Per
known-debts #23 plan, the CLI must drive every operation through
:mod:`lifeform_domain_figure.__init__` (the wheel's public surface)
plus a tightly-scoped allowlist of upstream kernel hooks
(:class:`PersonaLoRAPool`, :class:`GateDecision`,
:class:`EvaluationSnapshot`).

This test scans the CLI source files at AST level and rejects any
``from lifeform_domain_figure.<internal_module>`` import, except
sibling cli internals (``lifeform_domain_figure.cli.<x>``) and the
two wheel-internal modules whose typed records the CLI must
construct directly: :mod:`lifeform_domain_figure.audit` (typed
audit record + write helpers; the CLI is the only audit producer
in the repo) and the eval-snapshot loader inside
:mod:`lifeform_domain_figure.cli` itself.

Why a static check:

* Every internal-module import is a potential R8 violation: the
  CLI bypassing the public surface lets future feature work
  silently couple to wheel internals.
* The static check is fast, runs in CI before pytest, and gives
  a clear error message pointing at the violating import line.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CLI_PACKAGE = (
    REPO_ROOT
    / "packages"
    / "lifeform-domain-figure"
    / "src"
    / "lifeform_domain_figure"
    / "cli"
)

# Modules the CLI may import directly (whitelist). Anything else
# under ``lifeform_domain_figure.<x>`` must go through the wheel
# root (``from lifeform_domain_figure import X``).
WHEEL_ROOT_MODULE = "lifeform_domain_figure"
ALLOWED_CLI_INTERNAL_PREFIXES = (
    "lifeform_domain_figure.cli",       # sibling cli files
    "lifeform_domain_figure.audit",     # typed audit record types + writers
)


def _cli_source_files() -> list[pathlib.Path]:
    return sorted(p for p in CLI_PACKAGE.rglob("*.py") if p.is_file())


@pytest.mark.parametrize(
    "source_path",
    _cli_source_files(),
    ids=lambda p: p.relative_to(REPO_ROOT).as_posix(),
)
def test_cli_only_imports_public_surface_or_allowed_internal(
    source_path: pathlib.Path,
) -> None:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if not module.startswith(f"{WHEEL_ROOT_MODULE}."):
                continue
            if module == WHEEL_ROOT_MODULE:
                continue
            if any(
                module == prefix or module.startswith(f"{prefix}.")
                for prefix in ALLOWED_CLI_INTERNAL_PREFIXES
            ):
                continue
            violations.append(
                f"line {node.lineno}: from {module} import "
                f"{', '.join(alias.name for alias in node.names)}"
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if not alias.name.startswith(f"{WHEEL_ROOT_MODULE}."):
                    continue
                if alias.name == WHEEL_ROOT_MODULE:
                    continue
                if any(
                    alias.name == prefix or alias.name.startswith(f"{prefix}.")
                    for prefix in ALLOWED_CLI_INTERNAL_PREFIXES
                ):
                    continue
                violations.append(
                    f"line {node.lineno}: import {alias.name}"
                )
    assert not violations, (
        f"{source_path.relative_to(REPO_ROOT)} reaches into "
        f"figure-wheel internals (violates known-debts #23 R8 "
        f"contract). Each violation must be replaced with an import "
        f"from the wheel root (``from lifeform_domain_figure import "
        f"X``):\n  - " + "\n  - ".join(violations)
    )


def test_cli_internal_allowlist_is_tight() -> None:
    """Sanity check: the allowlist is exactly the modules the plan
    documents. If a future change adds a new internal module the
    CLI is allowed to reach into, the allowlist update must come
    paired with this test update so reviewers see it in the diff."""

    assert ALLOWED_CLI_INTERNAL_PREFIXES == (
        "lifeform_domain_figure.cli",
        "lifeform_domain_figure.audit",
    ), (
        "ALLOWED_CLI_INTERNAL_PREFIXES drifted from the known-debts "
        "#23 plan; update both this test and the plan note in "
        "docs/known-debts.md before merging."
    )
