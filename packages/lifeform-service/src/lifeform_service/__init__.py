"""HTTP service surface for the Volvence Zero lifeform.

This wheel turns an in-process ``Lifeform`` (from ``lifeform-core``,
typically constructed by a vertical's ``build_*_lifeform()`` factory)
into a multi-tenant network service. Routes are deliberately thin
HTTP-shaped projections of the lifeform's API; no business logic lives
in this layer.

Public surfaces:

* ``create_app`` \u2014 ``aiohttp`` ``Application`` factory.
* ``SessionManager`` \u2014 multi-tenant lifecycle for ``LifeformSession``s.
* ``VerticalSpec`` \u2014 declares a vertical (name + ``Lifeform`` factory).
* ``discover_verticals`` \u2014 enumerates installed verticals.
* ``main`` \u2014 ``lifeform-serve`` CLI entry point.

Routes (all under ``/v1``):

* ``GET  /v1/health`` \u2014 liveness + session count
* ``GET  /v1/info`` \u2014 active vertical, bootstraps available, paths
* ``POST /v1/sessions`` \u2014 create a session (optional ``session_id`` in body)
* ``DELETE /v1/sessions/{id}`` \u2014 close a session
* ``GET  /v1/sessions/{id}/state`` \u2014 read-only state summary
* ``POST /v1/sessions/{id}/turns`` \u2014 run a turn
* ``POST /v1/sessions/{id}/end-scene`` \u2014 close the open scene + drain slow loop
"""

from __future__ import annotations

from lifeform_service.app import create_app
from lifeform_service.bundle_root_scanner import (
    BundleScanReport,
    scan_and_register_bundles,
)
from lifeform_service.cli import main
from lifeform_service.einstein_resolver import (
    EinsteinBundleResolution,
    resolve_einstein_bundle,
)
from lifeform_service.figure_bundle_store import (
    FigureBundleNotFound,
    FigureBundleStore,
    default_store as default_figure_bundle_store,
    lookup_bundle as lookup_figure_bundle,
    register_bundle_persona_lora,
)
from lifeform_service.session_manager import (
    InvalidTemporalForkError,
    ScopeNotAuthorizedError,
    SessionAlreadyExistsError,
    SessionManager,
    SessionNotFoundError,
    SnapshotNotRestorableError,
    TimeNodeNotFoundError,
    TimeNodeSnapshot,
)
from lifeform_service.verticals import (
    COMPANION_ABLATION_VERTICAL_NAMES,
    VerticalSpec,
    default_vertical_name,
    discover_companion_ablation_verticals,
    discover_verticals,
)

__all__ = (
    "BundleScanReport",
    "EinsteinBundleResolution",
    "FigureBundleNotFound",
    "FigureBundleStore",
    "InvalidTemporalForkError",
    "ScopeNotAuthorizedError",
    "SessionAlreadyExistsError",
    "SessionManager",
    "SessionNotFoundError",
    "SnapshotNotRestorableError",
    "TimeNodeNotFoundError",
    "TimeNodeSnapshot",
    "COMPANION_ABLATION_VERTICAL_NAMES",
    "VerticalSpec",
    "create_app",
    "default_figure_bundle_store",
    "default_vertical_name",
    "discover_companion_ablation_verticals",
    "discover_verticals",
    "lookup_figure_bundle",
    "main",
    "register_bundle_persona_lora",
    "resolve_einstein_bundle",
    "scan_and_register_bundles",
)
