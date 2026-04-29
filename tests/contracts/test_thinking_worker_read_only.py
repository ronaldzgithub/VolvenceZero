"""Contract test: lifeform-thinking workers must be read-only.

Workers in ``packages/lifeform-thinking/src/lifeform_thinking/workers/``
must not reach into kernel owner mutation APIs. Specifically they
cannot import:

* any ``volvence_zero.application.storage`` store class (they'd be
  tempted to call ``upsert_records`` / ``reconcile_provisional_cases``
  directly, bypassing the scheduler-driven artifact apply path)
* any ``volvence_zero.semantic_state.SemanticStateStore``
* ``MemoryStore`` (the learned core is a write target)

This is a *grep-based* contract test: we parse worker files and
forbid those specific module names at import time. The scheduler
itself IS allowed to import immutable contract types
(``ThinkingTask`` etc.) but workers should only need the thinking
package's public surface + ``typing``.

Separate from ``test_import_boundaries.py`` because that test is
about wheel tiers; this one is about a narrow code-smell within one
wheel.
"""

from __future__ import annotations

import ast
import pathlib

import pytest


_WORKERS_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "packages"
    / "lifeform-thinking"
    / "src"
    / "lifeform_thinking"
    / "workers"
)


_FORBIDDEN_IMPORTS = frozenset(
    {
        "volvence_zero.application.storage",
        "volvence_zero.memory.store",
        "volvence_zero.regime",
        # semantic state has SemanticStateStore which is a write target.
        "volvence_zero.semantic_state",
    }
)


def _worker_py_files() -> list[pathlib.Path]:
    if not _WORKERS_DIR.is_dir():
        return []
    return sorted(p for p in _WORKERS_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def _module_imports(py_file: pathlib.Path) -> list[str]:
    source = py_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(py_file))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and not node.level:
                modules.append(node.module)
    return modules


@pytest.mark.parametrize(
    "py_file",
    _worker_py_files(),
    ids=lambda p: p.name,
)
def test_workers_do_not_import_owner_store_modules(py_file: pathlib.Path) -> None:
    """Forbid workers from reaching into owner-mutation module paths.

    Workers may read upstream snapshots (passed in as an already-typed
    mapping by the scheduler) but must not import any store or owner
    module directly. If a worker legitimately needs a new owner's
    snapshot shape it should import the snapshot *type* (e.g.
    ``CaseMemorySnapshot``) from the public ``volvence_zero.application``
    surface; that import is narrower and easier to audit.
    """
    imports = _module_imports(py_file)
    offending = [m for m in imports if m in _FORBIDDEN_IMPORTS]
    assert not offending, (
        f"Worker {py_file.name} imports forbidden owner-mutation module(s): "
        f"{offending}. Workers must be read-only; reach upstream snapshots "
        "through the scheduler-supplied mapping instead. See "
        "docs/specs/thinking-loop.md and "
        ".cursor/rules/ssot-module-boundaries.mdc."
    )


def test_worker_files_exist() -> None:
    """Sanity: the workers directory is non-empty; the parametrized
    test above would otherwise silently pass with zero cases.
    """
    assert _worker_py_files(), (
        "lifeform-thinking/workers/ directory contains no Python files; "
        "test_workers_do_not_import_owner_store_modules would be a no-op."
    )
