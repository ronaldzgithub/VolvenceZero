"""Cross-cutting AST contract guards for the L0 crawler subpackage.

The crawler is the **only** figure-vertical layer permitted to issue
HTTP requests; the L1 cleaning + L2 verification subpackages have
no-HTTP guards. This file inverts that posture for ``crawl/`` while
keeping the other R8 / R12 / R15 invariants in place:

1. ``crawl/`` MAY import ``requests`` / ``urllib.parse`` /
   ``urllib.robotparser``. (No assertion — just absence of a
   no-HTTP guard.)
2. ``crawl/`` MUST NOT import any typed ``Figure*Source`` record;
   typed sources stay L1's concern (R8 / `ssot-module-boundaries.mdc`).
3. ``crawl/`` MUST NOT import any kernel module
   (``volvence_zero.{cognition,...,runtime}``); the crawler is an
   artefact producer, never a kernel writer (R12).
4. ``crawl/`` MUST NOT import the L2 ``verification`` subpackage;
   the crawler does not consult the verifier (the verifier reads
   provenance derived from L1, not from crawl results).

Adding a new HTTP client library is allowed (e.g., httpx); but the
guards on typed-source / kernel / verification imports must remain.
"""

from __future__ import annotations

import ast
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CRAWL_ROOT = (
    _REPO_ROOT
    / "packages"
    / "lifeform-domain-figure"
    / "src"
    / "lifeform_domain_figure"
    / "crawl"
)


_FORBIDDEN_TYPED_SOURCE_NAMES = {
    "FigurePaperSource",
    "FigureLetterSource",
    "FigureLectureSource",
    "FigureNotebookSource",
}


_FORBIDDEN_KERNEL_PREFIXES = (
    "volvence_zero.cognition",
    "volvence_zero.temporal",
    "volvence_zero.memory",
    "volvence_zero.substrate",
    "volvence_zero.application",
    "volvence_zero.runtime",
)


_FORBIDDEN_VERIFICATION_PREFIX = "lifeform_domain_figure.verification"


def _iter_crawl_python_files() -> list[Path]:
    return sorted(p for p in _CRAWL_ROOT.rglob("*.py"))


def test_crawl_root_directory_exists() -> None:
    assert _CRAWL_ROOT.exists(), f"contract anchor directory missing: {_CRAWL_ROOT}"
    assert _iter_crawl_python_files()


def test_no_crawl_module_imports_typed_figure_source() -> None:
    violations: list[tuple[Path, int, str]] = []
    for path in _iter_crawl_python_files():
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
        "crawl/ files must not import Figure*Source typed records "
        "(typed sources are L1's responsibility). "
        f"Violations: {violations}"
    )


def test_no_crawl_module_imports_kernel() -> None:
    violations: list[tuple[Path, int, str]] = []
    for path in _iter_crawl_python_files():
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
        "crawl/ files must not import kernel modules "
        "(crawler is artefact producer; never writes kernel; R12)."
        f" Violations: {violations}"
    )


def test_no_crawl_module_imports_verification() -> None:
    violations: list[tuple[Path, int, str]] = []
    for path in _iter_crawl_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
                if (
                    module == _FORBIDDEN_VERIFICATION_PREFIX
                    or module.startswith(_FORBIDDEN_VERIFICATION_PREFIX + ".")
                ):
                    violations.append((path, node.lineno, module))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if (
                        alias.name == _FORBIDDEN_VERIFICATION_PREFIX
                        or alias.name.startswith(_FORBIDDEN_VERIFICATION_PREFIX + ".")
                    ):
                        violations.append((path, node.lineno, alias.name))
    assert not violations, (
        "crawl/ files must not import lifeform_domain_figure.verification.* "
        "(crawler does not consult L2 verifier). "
        f"Violations: {violations}"
    )
