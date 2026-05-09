"""Smoke tests for the F4.2 Einstein vertical wiring.

Validates:

* :func:`discover_verticals` enumerates the new ``einstein`` entry.
* The vertical's ``factory(None)`` produces a runnable
  :class:`Lifeform` (synthetic substrate path).
* The constructed lifeform's expression synthesizer carries the
  figure bundle (so L1 / L3 / L4 enforcement is reachable at
  session-construction time).
* :func:`default_figure_bundle_store` is seeded with the Einstein
  bundle keyed by both ``bundle_id`` and ``figure_id``.
* :class:`FigureBundleStore` round-trips registration and lookup.
"""

from __future__ import annotations

import pytest

from lifeform_core import Lifeform
from lifeform_service import (
    FigureBundleNotFound,
    FigureBundleStore,
    default_figure_bundle_store,
    discover_verticals,
    lookup_figure_bundle,
)


def test_einstein_vertical_is_discovered() -> None:
    specs = discover_verticals()
    assert "einstein" in specs
    spec = specs["einstein"]
    assert spec.name == "einstein"
    assert callable(spec.factory)


def test_einstein_factory_produces_lifeform_with_bundle() -> None:
    spec = discover_verticals()["einstein"]
    lifeform = spec.factory(None)
    assert isinstance(lifeform, Lifeform)
    # The init kwargs carry the synthesizer that the factory bound;
    # its figure_bundle attribute is what enforces L1 / L3 / L4 at
    # session-construction time. Reading the kwargs is the smoke
    # path that does not depend on a session being created.
    init_kwargs = getattr(lifeform, "_init_kwargs", {})
    synthesizer = init_kwargs.get("response_synthesizer")
    assert synthesizer is not None, "Einstein factory must wire a synthesizer"
    bundle = getattr(synthesizer, "figure_bundle", None)
    assert bundle is not None, "Einstein synthesizer must carry a figure_bundle"
    assert getattr(bundle, "figure_id", "") == "einstein"


def test_default_store_seeds_einstein() -> None:
    store = default_figure_bundle_store()
    assert store.has("einstein")
    bundle = store.lookup("einstein")
    assert getattr(bundle, "figure_id", "") == "einstein"
    bundle_id = getattr(bundle, "bundle_id", "")
    assert bundle_id
    assert store.has(bundle_id)
    assert store.lookup(bundle_id) is bundle


def test_lookup_figure_bundle_helper_roundtrips() -> None:
    bundle = lookup_figure_bundle(bundle_id="einstein")
    assert getattr(bundle, "figure_id", "") == "einstein"


def test_lookup_figure_bundle_helper_default_for_empty_id() -> None:
    sentinel = object()
    assert lookup_figure_bundle(default=sentinel, bundle_id="") is sentinel


def test_figure_bundle_store_register_and_lookup() -> None:
    store = FigureBundleStore()

    class _FakeBundle:
        bundle_id = "fake-bundle:abc"
        figure_id = "fake-figure"

    fake = _FakeBundle()
    bundle_id = store.register(fake)
    assert bundle_id == "fake-bundle:abc"
    assert store.lookup("fake-bundle:abc") is fake
    assert store.lookup("fake-figure") is fake


def test_figure_bundle_store_missing_raises() -> None:
    store = FigureBundleStore()
    with pytest.raises(FigureBundleNotFound):
        store.lookup("not-registered")


def test_figure_bundle_store_register_invalid_raises() -> None:
    store = FigureBundleStore()

    class _BadBundle:
        bundle_id = "  "
        figure_id = "f"

    with pytest.raises(ValueError, match="bundle_id"):
        store.register(_BadBundle())

    class _NoBundleId:
        bundle_id = "ok"
        figure_id = ""

    with pytest.raises(ValueError, match="figure_id"):
        store.register(_NoBundleId())
