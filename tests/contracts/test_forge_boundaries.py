"""Physical isolation contract for the development-loop RSI Forge."""

from __future__ import annotations

import ast
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES_ROOT = REPO_ROOT / "packages"
FORGE_SOURCE_ROOT = REPO_ROOT / "forge" / "src"
sys.path.insert(0, str(FORGE_SOURCE_ROOT))

from volvence_forge.config import ForgeConfig, ForgePaths  # noqa: E402


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


@pytest.mark.parametrize(
    "protected_target",
    (
        "packages/lifeform-domain-character/src/lifeform_domain_character/"
        "scenario_packages/zhang_wuji_character_migration_v1/scenes.yaml",
        "packages/lifeform-domain-character/src/lifeform_domain_character/"
        "scenario_packages/zhang_wuji_character_migration_v1/ssot_fragment.json",
        "packages/lifeform-domain-character/src/lifeform_domain_character/"
        "scenario_packages/zhang_wuji_character_migration_v1/test_suite.yaml",
        "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/"
        "runtime_assets/test_suite.yaml",
        "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/"
        "schemas/companion_playbook_overlay.schema.json",
        "packages/companion-bench/src/companion_bench/scenarios/seven_day/manifest.yaml",
        "packages/lifeform-domain-character/src/lifeform_domain_character/evaluation/judge.json",
        "scripts/forge_gate_adjudicator.py",
    ),
)
def test_runtime_evaluator_and_gate_surfaces_are_not_editable(protected_target: str) -> None:
    config = ForgeConfig.load(ForgePaths.discover(repo_root=REPO_ROOT))
    assert config.is_read_only(protected_target)
    assert config.editable_entry_for(protected_target) is None


def test_only_owner_bound_companion_overlay_is_runtime_editable() -> None:
    config = ForgeConfig.load(ForgePaths.discover(repo_root=REPO_ROOT))
    overlay = (
        "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/"
        "runtime_assets/companion_playbook_overlay.json"
    )
    entry = config.editable_entry_for(overlay)
    assert entry is not None
    assert entry.component == "companion_runtime_playbook_overlay"
    assert entry.requires_offline_gate
    assert config.editable_entry_for(
        "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/companion_pack.py"
    ) is None
    assert config.editable_entry_for(
        "packages/lifeform-domain-coding/src/lifeform_domain_coding/coding_pack.py"
    ) is None
