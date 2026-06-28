# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Isolation contract: this wheel must not import the monorepo internals.

The CAMEL baseline is an external, vendor-neutral agent baseline. It must never
import ``volvence_zero.*``, ``lifeform_*``, ``companion_bench.*``, or
``companion_ref_harness.*`` — otherwise a "win vs CAMEL" comparison would be
contaminated by our own kernel code. Enforced statically via AST so the check
does not require importing (and thus installing) any of those packages.

Also asserts the Apache 2.0 license header is present on every source file, so
the Apache-licensed wheel cannot accidentally pick up Proprietary-licensed code
from the rest of the monorepo.
"""

from __future__ import annotations

import ast
import pathlib

# Exact top-level package names (matched as ``name`` or ``name.<sub>``).
_FORBIDDEN_EXACT: tuple[str, ...] = (
    "volvence_zero",
    "companion_bench",
    "companion_ref_harness",
)
# Prefix family (any module whose top segment starts with this, e.g.
# ``lifeform_core``, ``lifeform_service``).
_FORBIDDEN_PREFIX: tuple[str, ...] = ("lifeform_",)

_SRC_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src" / "companion_camel_baseline"


def _iter_source_files() -> list[pathlib.Path]:
    return sorted(_SRC_ROOT.rglob("*.py"))


def _is_forbidden(module: str) -> bool:
    if not module:
        return False
    top = module.split(".", 1)[0]
    if top in _FORBIDDEN_EXACT:
        return True
    return any(top.startswith(prefix) for prefix in _FORBIDDEN_PREFIX)


def test_source_root_exists() -> None:
    assert _SRC_ROOT.is_dir(), f"source root not found: {_SRC_ROOT}"
    assert _iter_source_files(), "no source files discovered"


def test_no_forbidden_imports() -> None:
    offenders: list[str] = []
    for path in _iter_source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_forbidden(alias.name):
                        offenders.append(f"{path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if _is_forbidden(module):
                    offenders.append(f"{path.name}: from {module} import ...")
    assert not offenders, "forbidden internal imports found:\n" + "\n".join(offenders)


def test_apache_license_header_present() -> None:
    missing: list[str] = []
    for path in _iter_source_files():
        head = path.read_text(encoding="utf-8")[:400]
        if "Apache License, Version 2.0" not in head:
            missing.append(path.name)
    assert not missing, "files missing Apache 2.0 header:\n" + "\n".join(missing)
