# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Static guard: ``companion-ref-harness`` must NOT import internal wheels.

``companion-ref-harness`` is the vendor-neutral CompanionBench reference
baseline. If it ever imports ``volvence_zero.*``, ``lifeform_*``, or
``companion_bench.*`` it becomes our-stack-specific and breaks the
RFC §3 P4 outcome-level evaluation contract plus the
``docs/moving forward/companion-ref-harness-packet.md`` §6 boundary
守门 ("ref-harness wheel must not import internal wheels").

The two CompanionBench wheels (``companion-bench`` benchmark code and
``companion-ref-harness`` baseline code) must remain mutually
unaware: bench is the test runner, harness is one of many SUTs.

This test mirrors
``tests/contracts/test_companion_bench_no_internal_imports.py``.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CRH_SRC = REPO_ROOT / "packages" / "companion-ref-harness" / "src"

# Roots whose sub-packages must never appear in companion-ref-harness source.
_FORBIDDEN_ROOTS: frozenset[str] = frozenset(
    {
        "volvence_zero",
        "companion_bench",
        "lifeform_core",
        "lifeform_service",
        "lifeform_openai_compat",
        "lifeform_evolution",
        "lifeform_expression",
        "lifeform_ingestion",
        "lifeform_thinking",
        "lifeform_affordance",
    }
)
# Any ``lifeform_domain_*`` and ``dlaas_platform_*`` are also forbidden.
_FORBIDDEN_PREFIXES: tuple[str, ...] = ("lifeform_domain_", "dlaas_platform_")


def _iter_python_sources() -> list[pathlib.Path]:
    if not CRH_SRC.exists():
        return []
    return sorted(CRH_SRC.rglob("*.py"))


def _module_root(name: str) -> str:
    return name.split(".", 1)[0] if name else ""


def _is_forbidden(module_name: str) -> bool:
    root = _module_root(module_name)
    if root in _FORBIDDEN_ROOTS:
        return True
    return any(root.startswith(p) for p in _FORBIDDEN_PREFIXES)


@pytest.mark.parametrize("path", _iter_python_sources(), ids=lambda p: p.name)
def test_companion_ref_harness_does_not_import_internal_wheels(
    path: pathlib.Path,
) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden(alias.name):
                    offenders.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if _is_forbidden(mod):
                offenders.append((node.lineno, mod))
    assert not offenders, (
        f"{path} imports forbidden internal wheels: {offenders}. "
        f"companion-ref-harness must stay system-agnostic; talk to upstream "
        f"models through the OpenAI / Anthropic HTTP contract only."
    )


def test_at_least_one_companion_ref_harness_source_was_scanned() -> None:
    """Sanity: catches accidental empty-glob configurations."""
    files = _iter_python_sources()
    assert files, (
        "No .py files found under packages/companion-ref-harness/src/. "
        "Either the wheel was deleted or the path moved; update this test."
    )
