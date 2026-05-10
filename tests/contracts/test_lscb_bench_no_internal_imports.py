# Copyright 2026 LSCB Contributors
# Licensed under the Apache License, Version 2.0.

"""Static guard: ``lscb-bench`` must NOT import any internal product wheel.

LSCB is a system-agnostic benchmark; it consumes the OpenAI
``/v1/chat/completions`` HTTP contract and nothing else. If
``lscb_bench`` ever imports ``volvence_zero.*`` or ``lifeform_*`` it
becomes our-stack-specific and breaks RFC §3 P4 ("any system reachable
via an OpenAI-compatible chat completion API can be evaluated") +
§11 governance neutrality.

This test runs an AST-only sweep over every ``.py`` under
``packages/lscb-bench/src/`` and rejects any forbidden import.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LSCB_SRC = REPO_ROOT / "packages" / "lscb-bench" / "src"

# Roots whose sub-packages must never appear in lscb-bench source.
_FORBIDDEN_ROOTS: frozenset[str] = frozenset(
    {
        "volvence_zero",
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
# Any ``lifeform_domain_*`` is also forbidden.
_FORBIDDEN_PREFIXES: tuple[str, ...] = ("lifeform_domain_", "dlaas_platform_")


def _iter_python_sources() -> list[pathlib.Path]:
    if not LSCB_SRC.exists():
        return []
    return sorted(LSCB_SRC.rglob("*.py"))


def _module_root(name: str) -> str:
    return name.split(".", 1)[0] if name else ""


def _is_forbidden(module_name: str) -> bool:
    root = _module_root(module_name)
    if root in _FORBIDDEN_ROOTS:
        return True
    return any(root.startswith(p) for p in _FORBIDDEN_PREFIXES)


@pytest.mark.parametrize("path", _iter_python_sources(), ids=lambda p: p.name)
def test_lscb_bench_does_not_import_internal_wheels(path: pathlib.Path) -> None:
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
        f"lscb-bench must stay system-agnostic; reach kernels through "
        f"the OpenAI HTTP contract only."
    )


def test_at_least_one_lscb_bench_source_was_scanned() -> None:
    """Sanity: catches accidental empty-glob configurations."""
    files = _iter_python_sources()
    assert files, (
        "No .py files found under packages/lscb-bench/src/. "
        "Either the wheel was deleted or the path moved; update this test."
    )
