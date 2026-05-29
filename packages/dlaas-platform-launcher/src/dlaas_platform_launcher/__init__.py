"""DLaaS platform-tier instance launcher.

Public surface:

* :class:`InstanceManager` — owns the ``{ai_id -> SessionManager}``
  map. Each adoption produces one ``SessionManager`` bound to the
  vertical resolved from ``runtime_template_id``. The same shared
  ``OpenWeightResidualRuntime`` is passed into every vertical
  factory so one Qwen on one GPU backs every concurrent ai_id.
* :class:`InstanceNotFound` — typed lookup error.
* :data:`INSTANCE_MANAGER_APP_KEY` — aiohttp ``app[...]`` key the
  launcher uses to publish itself; the api wheel reads it during
  dispatch to look up the right SessionManager per ai_id.
"""

from __future__ import annotations

from dlaas_platform_launcher.instance_manager import (
    INSTANCE_MANAGER_APP_KEY,
    InstanceManager,
    InstanceNotFound,
)
from dlaas_platform_launcher.multi_pod_launcher import MultiPodLauncher
from dlaas_platform_launcher.placement import (
    AiIdPlacementRouter,
    PlacementCapacityError,
    PlacementNotFound,
    PlacementRecord,
    PodNotRegistered,
    RuntimePod,
)

__all__ = (
    "INSTANCE_MANAGER_APP_KEY",
    "InstanceManager",
    "InstanceNotFound",
    "MultiPodLauncher",
    "AiIdPlacementRouter",
    "PlacementCapacityError",
    "PlacementNotFound",
    "PlacementRecord",
    "PodNotRegistered",
    "RuntimePod",
)
