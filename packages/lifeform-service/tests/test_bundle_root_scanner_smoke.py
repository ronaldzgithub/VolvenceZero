"""Contract test for the U1 bundle-root scanner.

The scanner is the bridge between operator-side
``save_figure_bundle(...)`` calls (e.g. the family-bake-worker in
VolvenceDeploy) and the runtime ``FigureBundleStore``. Without it,
disk-baked bundles are invisible to ``lookup_figure_bundle`` and the
DLaaS adopt path cannot bind a per-``ai_id`` bundle to a template's
``figure_artifact_id``.

This test locks the contract that the family-memorial product depends
on:

1. A freshly-baked bundle written under ``<root>/<figure_id>/<bundle_id>/``
   is registered into the supplied :class:`FigureBundleStore` when the
   scanner runs.
2. The store remains capable of holding more than one bundle keyed
   by distinct ``bundle_id`` (two ``ai_id``s in the same process can
   bind to two distinct family bundles without colliding).
3. Re-running the scanner is idempotent (operators can call it on
   every health-check tick to pick up newly-baked bundles without
   restarting the platform).
4. Missing root directory does NOT raise — fresh installs report
   "registered zero bundles" instead of crashing the service.
5. A non-directory at the root path DOES raise — surface
   misconfiguration loudly rather than silently registering nothing.
"""

from __future__ import annotations

import pytest

from lifeform_domain_figure import (
    FigureBundleInputs,
    build_einstein_profile,
    build_figure_artifact_bundle,
    build_figure_ingestion_envelope,
    save_figure_bundle,
    synthetic_einstein_corpus,
)
from lifeform_domain_figure.envelope_builder import FigureCorpusSourceBundle

from lifeform_service import FigureBundleStore, scan_and_register_bundles
from lifeform_service.bundle_root_scanner import BundleScanReport


def _einstein_bundle():
    papers, letters, lectures, notebooks = synthetic_einstein_corpus()
    envelopes = build_figure_ingestion_envelope(
        FigureCorpusSourceBundle(
            figure_id="einstein",
            papers=papers,
            letters=letters,
            lectures=lectures,
            notebooks=notebooks,
        ),
        uploader="lifeform-service-tests:bundle-root-scanner",
    ).envelopes
    return build_figure_artifact_bundle(
        FigureBundleInputs(
            profile=build_einstein_profile(),
            envelopes=envelopes,
        )
    )


class _FakeMemorialBundle:
    """Duck-typed second bundle.

    The store does not require a real ``FigureArtifactBundle`` — it
    keys on ``bundle_id`` + ``figure_id`` attributes. We use a
    minimal stand-in to prove that a second family-memorial bundle
    co-resident in the same store is independently lookupable from
    the Einstein bundle the scanner registered from disk.
    """

    def __init__(self, bundle_id: str, figure_id: str) -> None:
        self.bundle_id = bundle_id
        self.figure_id = figure_id


def test_scanner_registers_freshly_saved_bundle(tmp_path) -> None:
    bundle = _einstein_bundle()
    bundle_dir = save_figure_bundle(bundle, root_dir=tmp_path)
    assert bundle_dir.is_dir()

    store = FigureBundleStore()
    report = scan_and_register_bundles(tmp_path, store=store)

    assert isinstance(report, BundleScanReport)
    assert report.registered_count == 1
    assert report.already_registered_count == 0
    assert bundle.bundle_id in report.bundle_ids
    assert store.has(bundle.bundle_id)
    looked_up = store.lookup(bundle.bundle_id)
    assert looked_up.bundle_id == bundle.bundle_id
    assert looked_up.figure_id == bundle.figure_id


def test_scanner_is_idempotent(tmp_path) -> None:
    bundle = _einstein_bundle()
    save_figure_bundle(bundle, root_dir=tmp_path)
    store = FigureBundleStore()

    first = scan_and_register_bundles(tmp_path, store=store)
    second = scan_and_register_bundles(tmp_path, store=store)

    assert first.registered_count == 1
    assert second.registered_count == 0
    assert second.already_registered_count == 1


def test_store_holds_multiple_bundles_keyed_by_bundle_id(tmp_path) -> None:
    """Two bundles in the same process; one from disk via scanner,
    one a memorial-style fake registered directly. Both must remain
    independently lookupable by distinct ``bundle_id``s.
    """

    einstein = _einstein_bundle()
    save_figure_bundle(einstein, root_dir=tmp_path)
    store = FigureBundleStore()
    scan_and_register_bundles(tmp_path, store=store)

    memorial = _FakeMemorialBundle(
        bundle_id="figure-bundle:family_grandpa01:0123456789abcdef",
        figure_id="family_grandpa01",
    )
    store.register(memorial)

    assert store.lookup(einstein.bundle_id).bundle_id == einstein.bundle_id
    assert store.lookup(memorial.bundle_id) is memorial
    # Distinct figure_ids must not collide on bundle_id lookup.
    assert store.lookup(einstein.bundle_id) is not memorial


def test_scanner_missing_root_returns_empty_report(tmp_path) -> None:
    absent = tmp_path / "no-bundles-yet"
    store = FigureBundleStore()
    report = scan_and_register_bundles(absent, store=store)
    assert report.registered_count == 0
    assert report.already_registered_count == 0
    assert report.bundle_ids == ()
    assert not store.has("anything")


def test_scanner_rejects_non_directory_root(tmp_path) -> None:
    bogus = tmp_path / "not-a-dir.txt"
    bogus.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        scan_and_register_bundles(bogus, store=FigureBundleStore())


def test_scanner_uses_default_store_when_none_provided(tmp_path) -> None:
    """Calling without ``store=`` must register into the process-wide
    default store. The DLaaS startup path will use this overload."""

    bundle = _einstein_bundle()
    save_figure_bundle(bundle, root_dir=tmp_path)
    # We don't reset the default store because that's a global; we
    # just assert the bundle becomes lookup-able from the helper.
    from lifeform_service import default_figure_bundle_store

    report = scan_and_register_bundles(tmp_path)
    default = default_figure_bundle_store()
    assert default.has(bundle.bundle_id)
    # Report tracks fresh registrations: depending on test ordering the
    # default store may already have this bundle, but the count must be
    # consistent with the report's `already_registered_count`.
    assert report.registered_count + report.already_registered_count == 1
