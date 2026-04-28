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
from lifeform_service.cli import main
from lifeform_service.session_manager import (
    SessionAlreadyExistsError,
    SessionManager,
    SessionNotFoundError,
)
from lifeform_service.verticals import (
    VerticalSpec,
    default_vertical_name,
    discover_verticals,
)

__all__ = (
    "SessionAlreadyExistsError",
    "SessionManager",
    "SessionNotFoundError",
    "VerticalSpec",
    "create_app",
    "default_vertical_name",
    "discover_verticals",
    "main",
)
