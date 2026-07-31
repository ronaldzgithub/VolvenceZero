"""Physical isolation contract for the development-loop RSI Forge."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES_ROOT = REPO_ROOT / "packages"
FORGE_SOURCE_ROOT = REPO_ROOT / "forge" / "src"


def _python_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _parse(path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        pytest.fail(f"Cannot parse {path.relative_to(REPO_ROOT)}: {exc}")


def _all_imports(path: Path) -> tuple[str, ...]:
    modules: list[str] = []
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.append(node.module)
    return tuple(modules)


def _module_level_imports(path: Path) -> tuple[str, ...]:
    modules: list[str] = []
    for node in _parse(path).body:
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.append(node.module)
    return tuple(modules)


@pytest.mark.parametrize("py_file", _python_files(PACKAGES_ROOT), ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_business_wheels_do_not_import_forge(py_file: Path) -> None:
    for module in _all_imports(py_file):
        assert module != "volvence_forge" and not module.startswith("volvence_forge."), (
            f"{py_file.relative_to(REPO_ROOT)} imports {module!r}: runtime/business wheels must not depend on "
            "the development-loop Forge."
        )


@pytest.mark.parametrize(
    "py_file", _python_files(FORGE_SOURCE_ROOT), ids=lambda path: str(path.relative_to(REPO_ROOT))
)
def test_forge_does_not_import_runtime_or_lifeform(py_file: Path) -> None:
    for module in _module_level_imports(py_file):
        assert module != "volvence_zero" and not module.startswith("volvence_zero."), (
            f"{py_file.relative_to(REPO_ROOT)} imports {module!r}: Forge reads public artifacts, not kernel code."
        )
        assert module != "lifeform" and not module.startswith("lifeform_"), (
            f"{py_file.relative_to(REPO_ROOT)} imports {module!r}: Forge must remain outside product wheels."
        )


def test_root_workspace_does_not_install_forge() -> None:
    root_pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "volvence-forge" not in root_pyproject
    assert '"forge"' not in root_pyproject
