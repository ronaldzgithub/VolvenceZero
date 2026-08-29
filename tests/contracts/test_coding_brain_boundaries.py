from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_CONTROLLER = (
    _ROOT
    / "packages/lifeform-domain-coding/src/lifeform_domain_coding/coding_brain.py"
)
_ROUTES = (
    _ROOT / "packages/lifeform-service/src/lifeform_service/coding_brain_routes.py"
)


def test_coding_brain_uses_brain_memory_facades_not_owner_internals() -> None:
    source = _CONTROLLER.read_text(encoding="utf-8")
    assert "session.retrieve_memory(" in source
    assert "session.write_memory(" in source
    assert "session.persist_memory(" in source
    assert "runner.memory_store" not in source
    assert "._runner" not in source


def test_coding_http_projection_does_not_import_kernel_owner_types() -> None:
    source = _ROUTES.read_text(encoding="utf-8")
    assert "from lifeform_domain_coding import" in source
    assert "volvence_zero.memory" not in source
    assert "volvence_zero.prediction" not in source
    assert "volvence_zero.dialogue_trace" not in source


def test_vz_wheels_do_not_import_the_coding_product_owner() -> None:
    violations: list[str] = []
    for wheel in sorted((_ROOT / "packages").glob("vz-*")):
        for path in wheel.rglob("*.py"):
            if "lifeform_domain_coding" in path.read_text(encoding="utf-8"):
                violations.append(str(path.relative_to(_ROOT)))
    assert violations == []
