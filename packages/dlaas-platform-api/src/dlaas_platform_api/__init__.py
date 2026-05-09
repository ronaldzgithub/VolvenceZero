"""DLaaS platform-tier HTTP router.

Slice 2 ships the typed ``InteractionEnvelope`` dispatch endpoint
(``POST /dlaas/instances/{ai_id}/interactions``) covering all seven
``InteractionType`` values (chat / feedback / observe / teach / task /
report / command). It is still bound to a hardcoded single-instance
``LifeformSession`` until Slice 3.5 introduces the multi-tenant
``InstanceManager``.

Public exports:

* :func:`attach_dlaas_routes` — register ``/dlaas/*`` routes onto an
  existing aiohttp ``Application`` produced by ``lifeform-service``.
* :func:`build_dlaas_app` — convenience wrapper that builds a
  ``lifeform-service`` app and attaches the DLaaS router in one call,
  used by the new ``dlaas-serve`` CLI.
* :func:`dispatch_envelope` / :class:`DispatchError` — typed dispatch
  surface (importable for unit tests / Slice 7 contract suite without
  a running aiohttp server).
"""

from __future__ import annotations

from dlaas_platform_api.app import (
    attach_dlaas_full_stack,
    attach_dlaas_routes,
    build_dlaas_app,
)
from dlaas_platform_api.control_plane import attach_control_plane_routes
from dlaas_platform_api.dispatch import DispatchError, dispatch_envelope

__all__ = (
    "DispatchError",
    "attach_control_plane_routes",
    "attach_dlaas_full_stack",
    "attach_dlaas_routes",
    "build_dlaas_app",
    "dispatch_envelope",
)
