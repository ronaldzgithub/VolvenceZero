"""Lifeform core — public API.

The kernel side (``vz-*``) MUST NOT import from this package. The dependency
direction is enforced by ``tests/contracts/test_import_boundaries.py``.
"""

from __future__ import annotations

from lifeform_core.followup_manager import FollowupManager
from lifeform_core.lifeform import Lifeform, LifeformConfig, LifeformSession
from lifeform_core.scene_manager import SceneManager
from lifeform_core.tick_engine import TickEngine, TickEngineConfig
from lifeform_core.types import (
    FollowupItem,
    Scene,
    SceneStatus,
    TickEvent,
    TickKind,
    TurnSummary,
    TurnTriggerKind,
    is_apprenticeship_trigger,
)
from lifeform_core.vitals import (
    DriveLevel,
    DriveSpec,
    VitalsBootstrap,
    VitalsModule,
    VitalsSnapshot,
)

__all__ = (
    "DriveLevel",
    "DriveSpec",
    "FollowupItem",
    "FollowupManager",
    "is_apprenticeship_trigger",
    "Lifeform",
    "LifeformConfig",
    "LifeformSession",
    "Scene",
    "SceneManager",
    "SceneStatus",
    "TickEngine",
    "TickEngineConfig",
    "TickEvent",
    "TickKind",
    "TurnSummary",
    "TurnTriggerKind",
    "VitalsBootstrap",
    "VitalsModule",
    "VitalsSnapshot",
)
