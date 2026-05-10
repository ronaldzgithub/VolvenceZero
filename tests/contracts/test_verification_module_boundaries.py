"""Cross-cutting AST contract guards for the L2 verification subpackage.

Mirrors :mod:`tests.contracts.test_cleaning_pipeline_versions` for L1.
The verification subpackage MUST stay decoupled from:

1. Typed figure source records (``Figure*Source``) — bridging into
   typed sources is L1's `cleaning/bridging.py` responsibility, NOT
   the verifier's job (R8 / `ssot-module-boundaries.mdc`).
2. HTTP clients — the L2 pipeline reads in-memory provenance records
   produced by the curator or by L1 bridging; it never fetches.
3. Internal kernel modules (`volvence_zero.cognition` / `temporal` /
   `memory` / `substrate` / `application` / `runtime`) — the
   verifier produces audit readouts and bundle gate decisions, never
   writes back into kernel owners (R12 evaluation single-direction).
"""

from __future__ import annotations

import ast
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_VERIFICATION_ROOT = (
    _REPO_ROOT
    / "packages"
    / "lifeform-domain-figure"
    / "src"
    / "lifeform_domain_figure"
    / "verification"
)


_FORBIDDEN_TYPED_SOURCE_NAMES = {
    "FigurePaperSource",
    "FigureLetterSource",
    "FigureLectureSource",
    "FigureNotebookSource",
}


_FORBIDDEN_HTTP_MODULES = {
    "requests",
    "httpx",
    "aiohttp",
    "urllib",
    "urllib2",
    "urllib3",
    "http",
    "http.client",
    "tornado.httpclient",
}


_FORBIDDEN_KERNEL_PREFIXES = (
    "volvence_zero.cognition",
    "volvence_zero.temporal",
    "volvence_zero.memory",
    "volvence_zero.substrate",
    "volvence_zero.application",
    "volvence_zero.runtime",
)


def _iter_verification_python_files() -> list[Path]:
    return sorted(p for p in _VERIFICATION_ROOT.rglob("*.py"))


def test_verification_root_directory_exists() -> None:
    assert _VERIFICATION_ROOT.exists(), (
        f"contract anchor directory missing: {_VERIFICATION_ROOT}"
    )
    files = _iter_verification_python_files()
    assert files, f"no Python files found under {_VERIFICATION_ROOT}"


def test_no_verification_module_imports_typed_figure_source() -> None:
    violations: list[tuple[Path, int, str]] = []
    for path in _iter_verification_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in _FORBIDDEN_TYPED_SOURCE_NAMES:
                        violations.append((path, node.lineno, alias.name))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name.split(".")[-1]
                    if name in _FORBIDDEN_TYPED_SOURCE_NAMES:
                        violations.append((path, node.lineno, alias.name))
    assert not violations, (
        "verification/ files must not import Figure*Source typed records "
        "(typed source records are L1 bridging's responsibility, not "
        "verifier input). "
        f"Violations: {violations}"
    )


def test_no_verification_module_imports_http_client() -> None:
    violations: list[tuple[Path, int, str]] = []
    for path in _iter_verification_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in _FORBIDDEN_HTTP_MODULES:
                        violations.append((path, node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                root = node.module.split(".")[0]
                if root in _FORBIDDEN_HTTP_MODULES or node.module in _FORBIDDEN_HTTP_MODULES:
                    violations.append((path, node.lineno, node.module))
    assert not violations, (
        "verification/ files must not import HTTP clients "
        "(L2 reads in-memory provenance, never fetches). "
        f"Violations: {violations}"
    )


def test_no_verification_module_imports_kernel() -> None:
    violations: list[tuple[Path, int, str]] = []
    for path in _iter_verification_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for prefix in _FORBIDDEN_KERNEL_PREFIXES:
                    if node.module == prefix or node.module.startswith(prefix + "."):
                        violations.append((path, node.lineno, node.module))
                        break
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    for prefix in _FORBIDDEN_KERNEL_PREFIXES:
                        if alias.name == prefix or alias.name.startswith(prefix + "."):
                            violations.append((path, node.lineno, alias.name))
                            break
    assert not violations, (
        "verification/ files must not import kernel modules "
        "(verifier is a readout / gate; never writes back into kernel "
        "owners; R12 evaluation single-direction). "
        f"Violations: {violations}"
    )
