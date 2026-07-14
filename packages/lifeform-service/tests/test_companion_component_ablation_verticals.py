"""Discovery tests for same-substrate companion component arms."""

from __future__ import annotations

from lifeform_core import Lifeform
from lifeform_service import discover_verticals


_COMPONENT_VERTICALS = {
    "companion-pe-drive-off": {
        "temporal_bootstrap": True,
        "regime_bootstrap": True,
    },
    "companion-eta-off": {
        "temporal_bootstrap": False,
        "regime_bootstrap": True,
    },
    "companion-active-learning-off": {
        "temporal_bootstrap": True,
        "regime_bootstrap": True,
    },
    "companion-lora-adapter": {
        "temporal_bootstrap": False,
        "regime_bootstrap": False,
    },
}


def test_companion_component_ablation_verticals_are_discovered() -> None:
    specs = discover_verticals()
    for name, expected in _COMPONENT_VERTICALS.items():
        assert name in specs
        spec = specs[name]
        assert spec.name == name
        assert callable(spec.factory)
        assert spec.has_temporal_bootstrap is expected["temporal_bootstrap"]
        assert spec.has_regime_bootstrap is expected["regime_bootstrap"]


def test_component_factories_build_synthetic_lifeforms() -> None:
    specs = discover_verticals()
    for name in _COMPONENT_VERTICALS:
        lifeform = specs[name].factory(None)
        assert isinstance(lifeform, Lifeform), name

