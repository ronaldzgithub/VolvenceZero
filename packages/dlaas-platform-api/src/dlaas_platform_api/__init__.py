"""DLaaS platform-tier HTTP router.

Slice 1 ships only the typed ``InteractionEnvelope`` dispatch endpoint
(``POST /dlaas/instances/{ai_id}/interactions``) bound to a hardcoded
single-instance ``LifeformSession`` so the architecture can be exercised
end-to-end before tenant / contract / launcher persistence lands.

Public exports:

* :func:`attach_dlaas_routes` — register ``/dlaas/*`` routes onto an
  existing aiohttp ``Application`` produced by ``lifeform-service``.
* :func:`build_dlaas_app` — convenience wrapper that builds a
  ``lifeform-service`` app and attaches the DLaaS router in one call,
  used by the new ``dlaas-serve`` CLI.
"""

from __future__ import annotations

from dlaas_platform_api.app import attach_dlaas_routes, build_dlaas_app

__all__ = ("attach_dlaas_routes", "build_dlaas_app")
