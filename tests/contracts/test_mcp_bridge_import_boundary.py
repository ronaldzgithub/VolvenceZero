"""mcp-tools-bundle-bridge: AST import-boundary contract.

Per ``docs/specs/mcp-bridge.md`` invariant 6: the
``lifeform-mcp-bridge`` wheel is a lifeform-side wheel and must NOT
reverse-import any kernel internals (``volvence_zero.cognition`` /
``volvence_zero.memory`` / ``volvence_zero.temporal`` /
``volvence_zero.substrate`` / ``volvence_zero.application`` /
``volvence_zero.runtime``). It IS allowed to import:

* ``volvence_zero.affordance`` / ``volvence_zero.environment`` — these
  are stable contracts living in ``vz-contracts``
* ``volvence_zero.runtime`` is BORDERLINE — kernel orchestration code
  lives there too. Per the wheel-boundary policy in this packet the
  bridge does NOT depend on ``vz-runtime``; only ``vz-contracts``,
  ``lifeform-affordance``, ``lifeform-ingestion``, and ``mcp`` /
  ``pyyaml``.

This is an AST-only test (no actual imports run), so it stays
deterministic and fast even when ``mcp`` / ``pyyaml`` are not
available.
"""

from __future__ import annotations

import ast
import pathlib

import pytest


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_BRIDGE_SRC = _REPO_ROOT / "packages" / "lifeform-mcp-bridge" / "src" / "lifeform_mcp_bridge"

# Modules under volvence_zero.* the bridge MUST NOT import. Each entry
# is a top-level segment after ``volvence_zero.``.
_FORBIDDEN_KERNEL_SEGMENTS = frozenset(
    {
        "cognition",
        "memory",
        "temporal",
        "substrate",
        "application",
        "runtime",
        "regime",
        "agent",
        "evaluation",
        "credit",
        "prediction",
        "reflection",
        "dual_track",
        "semantic_state",
        "social",
        "rupture_state",
        "interlocutor_state",
        "protocol_runtime",
        "integration",
    }
)


def _python_files() -> list[pathlib.Path]:
    if not _BRIDGE_SRC.is_dir():
        pytest.skip(
            "lifeform-mcp-bridge wheel not present at expected path "
            f"{_BRIDGE_SRC}; ran from a partial checkout?"
        )
    return sorted(_BRIDGE_SRC.rglob("*.py"))


def _imports_in(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return ``(lineno, dotted_name)`` for every ``import``/``from``."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            out.append((node.lineno, module))
    return out


def test_bridge_does_not_import_forbidden_kernel_segments() -> None:
    violations: list[str] = []
    for path in _python_files():
        for lineno, dotted in _imports_in(path):
            if not dotted.startswith("volvence_zero."):
                continue
            segment = dotted[len("volvence_zero."):].split(".", 1)[0]
            if segment in _FORBIDDEN_KERNEL_SEGMENTS:
                rel = path.relative_to(_REPO_ROOT)
                violations.append(
                    f"  {rel}:{lineno}: imports volvence_zero.{segment} "
                    f"({dotted!r})"
                )
    assert not violations, (
        "lifeform-mcp-bridge must not reverse-import kernel internals.\n"
        "Forbidden segments: "
        + ", ".join(sorted(_FORBIDDEN_KERNEL_SEGMENTS))
        + "\nViolations:\n"
        + "\n".join(violations)
    )


def test_bridge_does_not_import_lifeform_core() -> None:
    """Bridge does not depend on ``lifeform-core`` either; the
    integration glue lives in ``lifeform-core`` (Lifeform.start) which
    imports the bridge, not the other way around. This keeps the
    wheel install graph one-way.
    """
    violations: list[str] = []
    for path in _python_files():
        for lineno, dotted in _imports_in(path):
            if dotted.startswith("lifeform_core") or dotted.startswith(
                "lifeform-core"
            ):
                rel = path.relative_to(_REPO_ROOT)
                violations.append(f"  {rel}:{lineno}: imports {dotted!r}")
    assert not violations, (
        "lifeform-mcp-bridge must not import lifeform-core; the dependency "
        "is one-way.\n" + "\n".join(violations)
    )


def test_bridge_only_uses_allowed_third_party_imports() -> None:
    """Whitelist of third-party imports the bridge is allowed to use.

    Update the whitelist consciously when you add a new dependency
    to ``lifeform-mcp-bridge/pyproject.toml``. The check is
    intentionally narrow so a transitive dep creep gets surfaced.
    """
    allowed_third_party = frozenset(
        {
            # Standard library
            "__future__",
            "asyncio", "collections", "dataclasses", "enum", "hashlib",
            "json", "logging", "os", "pathlib", "re", "subprocess",
            "sys", "time", "typing", "uuid",
            # Project wheels we ARE allowed to depend on
            "volvence_zero",        # via vz-contracts shared types
            "lifeform_affordance",  # invoker / registry
            "lifeform_ingestion",   # IngestionEnvelope shapes
            "lifeform_mcp_bridge",  # internal sibling imports
            # Declared third-party deps (see pyproject.toml)
            "yaml",                 # PyYAML
            # Optional extras (fine to import-guard if needed)
            "mcp",
            "aiohttp",
        }
    )
    violations: list[str] = []
    for path in _python_files():
        for lineno, dotted in _imports_in(path):
            if not dotted:
                continue
            top = dotted.split(".", 1)[0]
            if top in allowed_third_party:
                continue
            rel = path.relative_to(_REPO_ROOT)
            violations.append(
                f"  {rel}:{lineno}: imports {dotted!r} "
                f"(top-level {top!r} not in allowlist)"
            )
    assert not violations, (
        "lifeform-mcp-bridge import allowlist exceeded. If a new "
        "third-party dep was added intentionally, update the test + "
        "the wheel pyproject.toml together.\n" + "\n".join(violations)
    )
