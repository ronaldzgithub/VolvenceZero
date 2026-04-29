"""Contract test: lifeform-ingestion must not import owner-mutation modules.

Ingestion is a chunker + pipeline that drives ``LifeformSession.run_turn``.
It must NEVER reach into kernel owner stores directly \u2014 the entire
point of Gap 3 is that external corpora flow through the same
canonical turn path that user input does.

Also forbid importing ``LifeformSession`` internals: the pipeline
uses a structural Protocol so tests can substitute stubs; real
production callers compose ``LifeformSession`` + ``IngestionPipeline``
themselves.

This is an AST-grep style test similar to
``tests/contracts/test_thinking_worker_read_only.py``. It is
narrowly scoped to the ingestion package.
"""

from __future__ import annotations

import ast
import pathlib

import pytest


_INGESTION_SRC = (
    pathlib.Path(__file__).resolve().parents[2]
    / "packages"
    / "lifeform-ingestion"
    / "src"
    / "lifeform_ingestion"
)


_FORBIDDEN_IMPORTS = frozenset(
    {
        # Kernel-owner stores \u2014 the pipeline must never call these.
        "volvence_zero.application.storage",
        "volvence_zero.memory.store",
        "volvence_zero.regime",
        "volvence_zero.semantic_state",
        # Runtime internals \u2014 ingestion drives a session, not a runner.
        "volvence_zero.agent.session",
        "volvence_zero.integration.final_wiring",
    }
)


def _ingestion_py_files() -> list[pathlib.Path]:
    if not _INGESTION_SRC.is_dir():
        return []
    return sorted(p for p in _INGESTION_SRC.rglob("*.py") if "__pycache__" not in p.parts)


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
    _ingestion_py_files(),
    ids=lambda p: p.relative_to(_INGESTION_SRC).as_posix(),
)
def test_ingestion_does_not_import_owner_store_or_runtime_internals(
    py_file: pathlib.Path,
) -> None:
    imports = _module_imports(py_file)
    offending = [m for m in imports if m in _FORBIDDEN_IMPORTS]
    assert not offending, (
        f"{py_file.relative_to(_INGESTION_SRC)} imports forbidden module(s): "
        f"{offending}. lifeform-ingestion must only drive sessions through "
        "LifeformSession.run_turn; it must not reach into owner stores or "
        "kernel runtime internals. See docs/specs/runtime-ingestion.md."
    )


def test_ingestion_src_is_non_empty() -> None:
    assert _ingestion_py_files(), (
        "lifeform-ingestion/src/lifeform_ingestion/ contains no Python files; "
        "the parametrized isolation test would otherwise silently pass."
    )
