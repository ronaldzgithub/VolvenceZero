from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

from companion_standard import trajectory_from_jsonable
from companion_standard.canonical import to_jsonable

from lifeform_synthetic_data.projections import project_relationship_encoder
from lifeform_synthetic_data.scenario import load_unified_v1_blueprints
from lifeform_synthetic_data.world import compile_structural_trajectory

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGES_ROOT = REPO_ROOT / "packages"


def _imports(root: Path) -> tuple[tuple[Path, str], ...]:
    output: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                output.extend((path, alias.name) for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                output.append((path, node.module))
    return tuple(output)


def test_companion_standard_remains_independent() -> None:
    source = PACKAGES_ROOT / "companion-standard" / "src"
    allowed_roots = set(sys.stdlib_module_names) | {"companion_standard"}

    violations = [
        f"{path.name}:{module}"
        for path, module in _imports(source)
        if module.split(".", maxsplit=1)[0] not in allowed_roots
    ]

    assert violations == []


def test_companion_trajgen_does_not_import_product_or_encoder_wheels() -> None:
    source = PACKAGES_ROOT / "companion-trajgen" / "src"
    forbidden = (
        "lifeform_",
        "volvence_zero",
        "companion_encoder",
        "dlaas_platform_",
    )

    violations = [f"{path.name}:{module}" for path, module in _imports(source) if module.startswith(forbidden)]

    assert violations == []


def test_companion_encoder_reads_only_standard_trajectory_contract() -> None:
    source = PACKAGES_ROOT / "companion-encoder" / "src"
    forbidden = (
        "companion_bench",
        "companion_trajgen",
        "lifeform_",
        "volvence_zero",
        "dlaas_platform_",
    )

    violations = [f"{path.name}:{module}" for path, module in _imports(source) if module.startswith(forbidden)]

    assert violations == []


def test_synthetic_data_never_imports_heldout_loader() -> None:
    source = PACKAGES_ROOT / "lifeform-synthetic-data" / "src"

    violations = [f"{path.name}:{module}" for path, module in _imports(source) if "heldout_loader" in module]

    assert violations == []


def test_relationship_projection_conforms_to_public_round_trip() -> None:
    blueprint = load_unified_v1_blueprints()[0]
    trajectory = compile_structural_trajectory(
        blueprint,
        replicate_index=0,
        seed=1,
        run_id="standard-conformance",
        created_at="2026-07-20T00:00:00Z",
        git_sha="test",
    )
    projected = project_relationship_encoder(trajectory)
    jsonable = json.loads(json.dumps(to_jsonable(projected)))

    assert trajectory_from_jsonable(jsonable) == projected
