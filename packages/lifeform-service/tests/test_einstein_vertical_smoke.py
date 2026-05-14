"""Smoke tests for the F4.2 Einstein vertical wiring.

Validates:

* :func:`discover_verticals` enumerates the legacy ``einstein`` alias
  plus the three ablation arms ``einstein-raw`` / ``einstein-bundle``
  / ``einstein-full`` that mirror
  :class:`PersonaCondition.{RAW,BUNDLE,BUNDLE_LORA}` from the
  verification harness.
* Each factory ``factory(None)`` produces a runnable :class:`Lifeform`
  on the synthetic substrate path.
* The RAW factory does **not** attach a figure bundle to the
  response synthesizer; the BUNDLE / BUNDLE_LORA / legacy
  ``einstein`` factories do.
* :func:`default_figure_bundle_store` is seeded with the Einstein
  bundle keyed by both ``bundle_id`` and ``figure_id``.
* :class:`FigureBundleStore` round-trips registration and lookup.
* :func:`resolve_einstein_bundle` falls back to ``synthetic`` when
  no disk bundle is reachable, fail-louds when the operator pins a
  missing bundle id, and fail-louds when ``EINSTEIN_REQUIRE_REAL_BUNDLE``
  is set against an empty root.
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
    resolve_einstein_bundle,
)


_ALL_EINSTEIN_NAMES = (
    "einstein",
    "einstein-raw",
    "einstein-bundle",
    "einstein-full",
)


def _read_bound_synthesizer(lifeform: Lifeform):
    init_kwargs = getattr(lifeform, "_init_kwargs", {})
    return init_kwargs.get("response_synthesizer")


def test_einstein_verticals_are_discovered() -> None:
    specs = discover_verticals()
    for name in _ALL_EINSTEIN_NAMES:
        assert name in specs, f"missing Einstein vertical {name!r}"
        assert specs[name].name == name
        assert callable(specs[name].factory)


def test_einstein_bundle_factory_attaches_bundle() -> None:
    spec = discover_verticals()["einstein-bundle"]
    lifeform = spec.factory(None)
    assert isinstance(lifeform, Lifeform)
    synthesizer = _read_bound_synthesizer(lifeform)
    assert synthesizer is not None
    bundle = getattr(synthesizer, "figure_bundle", None)
    assert bundle is not None, (
        "einstein-bundle factory must attach a figure_bundle"
    )
    assert getattr(bundle, "figure_id", "") == "einstein"


def test_einstein_full_factory_attaches_bundle() -> None:
    spec = discover_verticals()["einstein-full"]
    lifeform = spec.factory(None)
    assert isinstance(lifeform, Lifeform)
    synthesizer = _read_bound_synthesizer(lifeform)
    assert synthesizer is not None
    bundle = getattr(synthesizer, "figure_bundle", None)
    assert bundle is not None, (
        "einstein-full factory must attach a figure_bundle"
    )
    assert getattr(bundle, "figure_id", "") == "einstein"


def test_einstein_raw_factory_skips_bundle_attachment() -> None:
    spec = discover_verticals()["einstein-raw"]
    lifeform = spec.factory(None)
    assert isinstance(lifeform, Lifeform)
    synthesizer = _read_bound_synthesizer(lifeform)
    assert synthesizer is not None
    bundle = getattr(synthesizer, "figure_bundle", None)
    assert bundle is None, (
        "einstein-raw factory must keep figure_bundle unbound "
        "(no L1/L3/L4 enforcement); got "
        f"{bundle!r}"
    )


def test_legacy_einstein_alias_matches_bundle_arm() -> None:
    spec = discover_verticals()["einstein"]
    lifeform = spec.factory(None)
    synthesizer = _read_bound_synthesizer(lifeform)
    assert synthesizer is not None
    bundle = getattr(synthesizer, "figure_bundle", None)
    assert bundle is not None, (
        "legacy einstein alias must keep the bundle-attached behaviour"
    )


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


def test_resolver_falls_back_when_root_missing(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("EINSTEIN_BUNDLE_ROOT", str(tmp_path / "absent"))
    monkeypatch.delenv("EINSTEIN_BUNDLE_ID", raising=False)
    monkeypatch.delenv("EINSTEIN_REQUIRE_REAL_BUNDLE", raising=False)
    resolution = resolve_einstein_bundle()
    assert resolution.bundle is None
    assert resolution.bundle_id == ""
    assert resolution.source == "synthetic"


def test_resolver_fail_louds_when_required_but_missing(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("EINSTEIN_BUNDLE_ROOT", str(tmp_path))
    monkeypatch.setenv("EINSTEIN_REQUIRE_REAL_BUNDLE", "1")
    monkeypatch.delenv("EINSTEIN_BUNDLE_ID", raising=False)
    with pytest.raises(FileNotFoundError, match="resolver found no"):
        resolve_einstein_bundle()


def test_resolver_fail_louds_on_pinned_unknown_id(
    tmp_path, monkeypatch
) -> None:
    figure_dir = tmp_path / "einstein"
    figure_dir.mkdir()
    # Touch a manifest with an unrelated id so the resolver actually
    # gets past the "no manifests" branch and exercises the pin check.
    bundle_dir = figure_dir / "figure-bundle__einstein__deadbeef"
    bundle_dir.mkdir()
    (bundle_dir / "manifest.json").write_text(
        '{"manifest_schema_version": "vz-figure-bundle-manifest.v1", '
        '"bundle_schema_version": 1, "figure_id": "einstein", '
        '"bundle_id": "figure-bundle:einstein:deadbeef", '
        '"profile_version": "0.1.0", "version_window": [0, 0], '
        '"integrity_hash": "deadbeef", '
        '"created_at_iso": "2026-01-01T00:00:00Z", '
        '"steering_present": false, "lora_present": false}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("EINSTEIN_BUNDLE_ROOT", str(tmp_path))
    monkeypatch.setenv(
        "EINSTEIN_BUNDLE_ID", "figure-bundle:einstein:notreal"
    )
    monkeypatch.delenv("EINSTEIN_REQUIRE_REAL_BUNDLE", raising=False)
    with pytest.raises(FileNotFoundError, match="no matching manifest"):
        resolve_einstein_bundle()
