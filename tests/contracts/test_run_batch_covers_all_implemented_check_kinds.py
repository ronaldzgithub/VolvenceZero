"""Contract: ``figure_verify run-batch`` covers every IMPLEMENTED_CHECK_KIND.

Wave I closure invariant: when a future packet adds a new
:class:`CheckKind` to ``IMPLEMENTED_CHECK_KINDS``, the run-batch
CLI MUST also import + invoke its verifier; otherwise the bundle
gate (which already iterates over ``IMPLEMENTED_CHECK_KINDS``)
would silently start failing every newly-added axis with
"missing-check".

The test scans the script's source AST for two things:

1. The script imports every verifier function name in
   ``SINGLE_SOURCE_AUTO_VERIFIERS`` plus ``verify_cross_source_byte``.
2. The ``_cmd_run_batch`` body references every verifier function
   (so a new axis cannot land in IMPLEMENTED_CHECK_KINDS without
   the script forming a corresponding ``ledger.append(...)`` row).

This is a static guard. The runtime smoke test
``test_verification_run_batch_smoke.py`` covers the actual
behaviour.
"""

from __future__ import annotations

import ast
from pathlib import Path

from lifeform_domain_figure.verification import (
    SINGLE_SOURCE_AUTO_VERIFIERS,
)


_FIGURE_VERIFY_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "packages"
    / "lifeform-domain-figure"
    / "scripts"
    / "figure_verify.py"
)


def _source_text() -> str:
    return _FIGURE_VERIFY_SCRIPT.read_text(encoding="utf-8")


def _imported_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _called_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def test_figure_verify_script_imports_every_single_source_verifier() -> None:
    tree = ast.parse(_source_text())
    imported = _imported_names(tree)
    expected = {fn.__name__ for fn in SINGLE_SOURCE_AUTO_VERIFIERS.values()}
    expected.add("verify_cross_source_byte")
    missing = expected - imported
    assert not missing, (
        f"figure_verify.py must import every IMPLEMENTED_CHECK_KIND verifier; "
        f"missing imports: {sorted(missing)!r}. Adding a new check kind? Wire "
        f"the import + the run-batch dispatch in the same packet."
    )


def test_figure_verify_script_invokes_every_single_source_verifier() -> None:
    tree = ast.parse(_source_text())
    called = _called_names(tree)
    expected = {fn.__name__ for fn in SINGLE_SOURCE_AUTO_VERIFIERS.values()}
    expected.add("verify_cross_source_byte")
    missing = expected - called
    assert not missing, (
        f"figure_verify.py must call every IMPLEMENTED_CHECK_KIND verifier "
        f"in run-batch; missing calls: {sorted(missing)!r}."
    )


def test_figure_verify_script_imports_metadata_clients() -> None:
    """When --metadata-mode toggles, run-batch must build all three
    V2 metadata clients (Wikidata / OpenAlex / Crossref)."""

    tree = ast.parse(_source_text())
    imported = _imported_names(tree)
    for name in (
        "live_wikidata_client",
        "live_openalex_client",
        "live_crossref_client",
        "offline_wikidata_client",
        "offline_openalex_client",
        "offline_crossref_client",
    ):
        assert name in imported, (
            f"figure_verify.py must import {name!r} for --metadata-mode "
            "switching"
        )
