"""Filesystem-to-store bridge for baked figure bundles.

When the operator bake pipeline (see family-bake-worker in
VolvenceDeploy or :mod:`lifeform_domain_figure.cli` invocations in
ops scripts) writes a ``FigureArtifactBundle`` under a shared bundle
root, the runtime ``FigureBundleStore`` is empty until something
registers that bundle. The DLaaS adopt path then cannot resolve the
template's ``figure_artifact_id`` and the L3/L4 contract degrades to
the global default bundle.

This module closes that gap. ``scan_and_register_bundles`` walks a
bundle root directory, calls :func:`load_figure_bundle` for every
manifest it finds, and registers each bundle with the supplied (or
default) :class:`FigureBundleStore`. It is intended to run **once at
service startup**, after ``build_dlaas_app`` returns and before
``aiohttp.web.run_app``.

Re-running the scanner is safe: :meth:`FigureBundleStore.register`
overwrites by ``bundle_id`` key, and bundle ids are content-addressed
(R15), so the same on-disk bundle deterministically produces the same
key. Operators can therefore call the scanner periodically to pick up
newly-baked bundles without restarting the platform.

The scanner is fail-loud per
:rule:`no-swallow-errors-no-hasattr-abuse`. A corrupted manifest or a
bundle whose integrity hash does not match raises immediately; the
caller logs and decides whether the platform should still serve
traffic. We deliberately do NOT swallow per-bundle errors into a
"skip-and-continue" path — that would mask schema drift between bake
and load.
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
from typing import Optional

from lifeform_service.figure_bundle_store import (
    FigureBundleStore,
    default_store,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BundleScanReport:
    """Outcome of a single ``scan_and_register_bundles`` invocation.

    Fields are typed so a caller's log line can include exact counts
    (operators routinely check this on a startup that suddenly has
    "zero memorials reachable").
    """

    root_dir: pathlib.Path
    registered_count: int
    already_registered_count: int
    bundle_ids: tuple[str, ...]


def scan_and_register_bundles(
    root_dir: str | pathlib.Path,
    *,
    store: Optional[FigureBundleStore] = None,
    figure_id: Optional[str] = None,
) -> BundleScanReport:
    """Scan ``root_dir`` and register every saved bundle into ``store``.

    Arguments:
        root_dir: filesystem path containing ``<figure_id>/<bundle_id>/``
            subdirectories. May be absent (typical for a fresh install) —
            the scanner returns an empty report in that case.
        store: target store. Defaults to the process-wide default
            (:func:`default_store`).
        figure_id: optional filter to register only bundles for a
            particular figure id.

    Returns:
        :class:`BundleScanReport` summarising what was registered.

    Raises:
        ValueError: when a bundle's manifest is malformed, when the
            pickle integrity hash does not match, or when a bundle id is
            empty. The DLaaS startup sequence must surface these — a
            silent skip would let a corrupted bundle root masquerade as
            empty and the operator would assume "no memorials yet".
    """

    target = store if store is not None else default_store()
    root = pathlib.Path(root_dir)
    if not root.exists():
        logger.info(
            "bundle_root_scanner: root_dir=%s does not exist (yet); "
            "registering zero bundles. This is normal for a fresh "
            "install before the first bake.",
            root,
        )
        return BundleScanReport(
            root_dir=root.resolve() if root.is_absolute() else root,
            registered_count=0,
            already_registered_count=0,
            bundle_ids=(),
        )
    if not root.is_dir():
        raise ValueError(
            f"scan_and_register_bundles: root_dir={root!s} exists but is "
            f"not a directory."
        )

    # Import locally so this module does not pull lifeform-domain-figure
    # at package-import time (mirrors figure_bundle_store._seed_default_store).
    from lifeform_domain_figure.bundle_io import (
        list_figure_bundles,
        load_figure_bundle,
    )

    manifests = list_figure_bundles(root_dir=root, figure_id=figure_id)
    registered: list[str] = []
    already_registered: list[str] = []
    for manifest in manifests:
        if target.has(manifest.bundle_id):
            already_registered.append(manifest.bundle_id)
            logger.debug(
                "bundle_root_scanner: bundle_id=%s already registered; "
                "skipping (id is content-addressed so re-registering is "
                "a no-op).",
                manifest.bundle_id,
            )
            continue
        bundle = load_figure_bundle(
            root_dir=root,
            bundle_id=manifest.bundle_id,
            figure_id=manifest.figure_id,
        )
        target.register(bundle)
        registered.append(manifest.bundle_id)
        logger.info(
            "bundle_root_scanner: registered bundle_id=%s figure_id=%s "
            "from %s",
            manifest.bundle_id,
            manifest.figure_id,
            manifest.bundle_dir,
        )

    return BundleScanReport(
        root_dir=root.resolve(),
        registered_count=len(registered),
        already_registered_count=len(already_registered),
        bundle_ids=tuple(registered),
    )


__all__ = (
    "BundleScanReport",
    "scan_and_register_bundles",
)
