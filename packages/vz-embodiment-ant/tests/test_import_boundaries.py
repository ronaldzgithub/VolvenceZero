"""SSOT guard: the embodiment package must not import kernel internals.

The digital ant enters the brain core ONLY through the ``vz-runtime`` facade
and the ``vz-contracts`` / ``vz-substrate`` public contracts. It must never
import ``volvence_zero.temporal`` / ``.memory`` / ``.prediction`` / etc.
directly, otherwise it would become a hidden second owner of kernel internals
(violates R8 + the module boundary rule).
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "volvence_ant"

# Kernel-internal top-level modules the embodiment must not touch directly.
_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "volvence_zero.temporal.",
    "volvence_zero.memory.",
    "volvence_zero.prediction.",
    "volvence_zero.internal_rl.",
    "volvence_zero.joint_loop.",
    "volvence_zero.credit.",
    "volvence_zero.dual_track.",
    "volvence_zero.regime.",
    "volvence_zero.semantic_state.",
    "volvence_zero.evaluation.",
    "volvence_zero.reflection.",
)

# Exact modules that are also forbidden (the bare package, not just submodules).
_FORBIDDEN_EXACT: frozenset[str] = frozenset(
    prefix.rstrip(".") for prefix in _FORBIDDEN_PREFIXES
)


def _iter_imported_modules(tree: ast.AST) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                modules.append(node.module)
    return modules


def test_no_kernel_internal_imports() -> None:
    offenders: list[str] = []
    for path in _SRC_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module in _iter_imported_modules(tree):
            if module in _FORBIDDEN_EXACT or module.startswith(_FORBIDDEN_PREFIXES):
                offenders.append(f"{path.relative_to(_SRC_ROOT)} -> {module}")
    assert not offenders, (
        "embodiment package imported kernel internals directly (use the "
        "vz-runtime facade / vz-contracts / vz-substrate instead):\n"
        + "\n".join(offenders)
    )


def test_only_allowed_volvence_zero_prefixes() -> None:
    """Positive guard: every volvence_zero import resolves to an allowed wheel."""

    allowed_prefixes = (
        "volvence_zero.environment",  # vz-contracts environment event types
        "volvence_zero.substrate",  # vz-substrate public contract
        "volvence_zero.runtime",  # vz-contracts kernel container
        "volvence_zero.temporal_types",  # vz-contracts controller state types
        "volvence_zero.agent",  # vz-runtime orchestration facade
        "volvence_zero.integration",  # vz-runtime rollout config facade
    )
    unexpected: list[str] = []
    for path in _SRC_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module in _iter_imported_modules(tree):
            if module.startswith("volvence_zero") and not module.startswith(allowed_prefixes):
                unexpected.append(f"{path.relative_to(_SRC_ROOT)} -> {module}")
    assert not unexpected, (
        "embodiment imported an unexpected volvence_zero module; add it to the "
        "allowed facade list only if it is a vz-contracts / vz-runtime / "
        "vz-substrate public surface:\n" + "\n".join(unexpected)
    )
