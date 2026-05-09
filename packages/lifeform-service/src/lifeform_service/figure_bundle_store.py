"""In-memory store mapping ``figure_artifact_id`` to a frozen bundle.

The DLaaS adopt path in :mod:`lifeform_service.app` reads
``TemplateSpec.figure_artifact_id`` from the registry and looks the
bundle up here so the runtime ``LifeformLLMResponseSynthesizer`` can
be bound to it via :meth:`with_figure_bundle`. The store stays in
the lifeform tier so the platform tier never imports
``lifeform_domain_figure`` directly (DLaaS allowlist invariant).

The store is process-level: bundles are constructed once at activate
or service-startup time, then served read-only across sessions. This
mirrors how ``OpenWeightResidualRuntime`` is loaded once per process
and shared across :class:`Lifeform` instances on the same GPU.

Thread-safety: the store is a thin wrapper over ``dict``; reads are
lock-free, writes use a single :class:`threading.Lock`. Activate /
adopt calls are infrequent compared to chat traffic, so the lock is
not a hot path. Use the **frozen** ``FigureArtifactBundle`` as the
returned value; mutating callers would be making a different bundle
and registering a new id (R15: every identity-bearing change yields
a fresh bundle id).
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import Any


class FigureBundleNotFound(LookupError):
    """Raised when a requested figure_artifact_id is not registered."""


class FigureBundleStore:
    """Process-level read-mostly registry of ``FigureArtifactBundle``.

    The store deliberately holds ``object`` rather than the typed
    ``FigureArtifactBundle`` so this module does not have to import
    ``lifeform_domain_figure`` for type checking. Callers that want
    typed access cast the returned value at the use site.
    """

    def __init__(self) -> None:
        self._bundles: dict[str, object] = {}
        self._lock = threading.Lock()

    def register(self, bundle: object) -> str:
        """Register ``bundle`` and return its bundle id.

        ``bundle`` must expose a ``bundle_id`` string attribute and a
        ``figure_id`` string attribute (we read them via single
        ``getattr`` calls — the duck-typed surface required from
        :class:`lifeform_domain_figure.FigureArtifactBundle`).
        """

        bundle_id = getattr(bundle, "bundle_id", "")
        if not isinstance(bundle_id, str) or not bundle_id.strip():
            raise ValueError(
                "FigureBundleStore.register: bundle.bundle_id must be a "
                "non-empty string"
            )
        figure_id = getattr(bundle, "figure_id", "")
        if not isinstance(figure_id, str) or not figure_id.strip():
            raise ValueError(
                "FigureBundleStore.register: bundle.figure_id must be a "
                "non-empty string"
            )
        with self._lock:
            self._bundles[bundle_id] = bundle
            self._bundles[figure_id] = bundle
        return bundle_id

    def lookup(self, bundle_or_figure_id: str) -> object:
        """Return the bundle for ``bundle_id`` or ``figure_id``.

        Raises :class:`FigureBundleNotFound` if no matching bundle
        is registered. Fail-loud per
        ``no-swallow-errors-no-hasattr-abuse.mdc``.
        """

        if not bundle_or_figure_id:
            raise ValueError(
                "FigureBundleStore.lookup: id must be non-empty"
            )
        bundle = self._bundles.get(bundle_or_figure_id)
        if bundle is None:
            raise FigureBundleNotFound(
                f"FigureBundleStore: no bundle registered for "
                f"{bundle_or_figure_id!r}"
            )
        return bundle

    def has(self, bundle_or_figure_id: str) -> bool:
        """Return whether the store can resolve ``bundle_or_figure_id``."""

        return bool(bundle_or_figure_id) and bundle_or_figure_id in self._bundles

    def keys(self) -> Iterable[str]:
        """Return a snapshot of registered ids (for diagnostics only)."""

        with self._lock:
            return tuple(self._bundles.keys())


_DEFAULT_STORE: FigureBundleStore | None = None
_DEFAULT_STORE_LOCK = threading.Lock()


def default_store() -> FigureBundleStore:
    """Return the process-wide default :class:`FigureBundleStore`.

    The default store is lazily initialised on first access and
    immediately seeded with the figure-vertical's shipped reviewed
    profiles (currently Einstein). Tests that want isolation should
    construct their own :class:`FigureBundleStore` instance instead
    of mutating this default.
    """

    global _DEFAULT_STORE
    if _DEFAULT_STORE is not None:
        return _DEFAULT_STORE
    with _DEFAULT_STORE_LOCK:
        if _DEFAULT_STORE is not None:
            return _DEFAULT_STORE
        _DEFAULT_STORE = FigureBundleStore()
        _seed_default_store(_DEFAULT_STORE)
    return _DEFAULT_STORE


def _seed_default_store(store: FigureBundleStore) -> None:
    """Populate ``store`` with every shipped figure profile.

    Bundles are constructed using the synthetic placeholder corpus
    that lifeform-domain-figure ships. Real-corpus replacement is a
    deployment-time concern; the bundle id changes when the corpus
    changes (R15) so existing templates pointing at the old id
    continue resolving to the old bundle until callers update.
    """

    try:
        from lifeform_domain_figure import (
            FigureBundleInputs,
            FigureCorpusSourceBundle,
            build_einstein_profile,
            build_figure_artifact_bundle,
            build_figure_ingestion_envelope,
            synthetic_einstein_corpus,
        )
    except ImportError:
        return
    profile = build_einstein_profile()
    papers, letters, lectures, notebooks = synthetic_einstein_corpus()
    bundle_inputs_corpus = FigureCorpusSourceBundle(
        figure_id="einstein",
        papers=papers,
        letters=letters,
        lectures=lectures,
        notebooks=notebooks,
    )
    envelope_set = build_figure_ingestion_envelope(
        bundle_inputs_corpus, uploader="lifeform-service:default-store"
    )
    bundle = build_figure_artifact_bundle(
        FigureBundleInputs(
            profile=profile,
            envelopes=envelope_set.envelopes,
        )
    )
    store.register(bundle)


def lookup_bundle(default: Any = None, *, bundle_id: str) -> object:
    """Convenience wrapper: ``default_store().lookup(bundle_id)`` or default.

    Used by DLaaS adopt code that wants to gracefully no-op when a
    template does not declare a figure artifact, by passing
    ``default=None``. When the template DOES declare an id and the
    store has no entry, this still raises — fail-loud is the L4
    contract surface for "the template asked for a figure we don't
    have".
    """

    if not bundle_id:
        return default
    return default_store().lookup(bundle_id)


__all__ = [
    "FigureBundleNotFound",
    "FigureBundleStore",
    "default_store",
    "lookup_bundle",
]
