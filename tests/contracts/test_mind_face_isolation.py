"""Mind/Face isolation contract tests.

The expression layer is the Face: it renders owner snapshots and exposes
typed rationale tags. It must not become a hidden learning / gate owner by
importing credit, evaluation, PE, memory, temporal, regime, or audit modules.
"""

from __future__ import annotations

import ast
import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LIFEFORM_EXPRESSION_SRC = (
    REPO_ROOT
    / "packages"
    / "lifeform-expression"
    / "src"
    / "lifeform_expression"
)

FORBIDDEN_FACE_IMPORT_PREFIXES: tuple[str, ...] = (
    "volvence_zero.audit",
    "volvence_zero.credit",
    "volvence_zero.evaluation",
    "volvence_zero.memory",
    "volvence_zero.prediction",
    "volvence_zero.regime",
    "volvence_zero.temporal",
)

# The Face may read contract surfaces used to render a response. These are
# intentionally narrow; adding an allowed prefix must be reviewed as an R8
# boundary change.
ALLOWED_VOLVENCE_IMPORT_PREFIXES: tuple[str, ...] = (
    "volvence_zero.agent.response",
    "volvence_zero.application.runtime",
    "volvence_zero.regime.hints",
)


def _python_files(root: pathlib.Path) -> tuple[pathlib.Path, ...]:
    if not root.is_dir():
        raise FileNotFoundError(f"lifeform-expression source dir not found: {root}")
    return tuple(sorted(root.rglob("*.py")))


def _imported_modules(path: pathlib.Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return tuple(imports)


def test_lifeform_expression_does_not_import_mind_or_gate_owners() -> None:
    """OA-2: Face layer must not import Mind / gate owner packages."""
    violations: list[str] = []
    for path in _python_files(LIFEFORM_EXPRESSION_SRC):
        for module_name in _imported_modules(path):
            if not module_name.startswith("volvence_zero."):
                continue
            if module_name.startswith(ALLOWED_VOLVENCE_IMPORT_PREFIXES):
                continue
            if module_name.startswith(FORBIDDEN_FACE_IMPORT_PREFIXES):
                violations.append(
                    f"{path.relative_to(REPO_ROOT)} imports forbidden owner {module_name!r}"
                )

    assert not violations, (
        "Mind/Face isolation violation: lifeform-expression must render/read "
        "contract surfaces only, not import learning or gate owners:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


def test_expression_layer_spec_names_static_guard() -> None:
    """The spec must point reviewers to this machine-enforced guard."""
    spec_path = REPO_ROOT / "docs" / "specs" / "expression-layer.md"
    text = spec_path.read_text(encoding="utf-8")
    assert "Mind/Face" in text
    assert "tests/contracts/test_mind_face_isolation.py" in text
