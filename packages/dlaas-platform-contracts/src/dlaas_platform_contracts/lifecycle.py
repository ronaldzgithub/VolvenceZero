"""DLaaS instance lifecycle contracts.

These types are platform-tier state. They describe whether an
externally-addressable ``ai_id`` is awake / asleep / failed, but they
do not describe cognitive state and must not be mirrored into kernel
owners.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class InstanceLifecycleState(str, Enum):
    """Platform-owned lifecycle for a hosted ``ai_id``."""

    PROVISIONING = "provisioning"
    ASLEEP = "asleep"
    WAKING = "waking"
    AWAKE = "awake"
    SLEEPING = "sleeping"
    PAUSED = "paused"
    SUSPENDED = "suspended"
    FAILED = "failed"


@dataclass(frozen=True)
class WakeRequest:
    """Request body for ``POST /dlaas/v1/instances/{ai_id}/wake``.

    ``template_id`` (U5 / family-memorial enabler): when set, the wake
    handler resolves the named template's ``figure_artifact_id`` and
    binds the corresponding ``FigureArtifactBundle`` to the awakened
    ``ai_id``'s ``SessionManager``. This lets bake-worker-style
    operator flows (mint template -> wake) inherit the per-ai_id L3
    retrieval index / L4 coverage map without going through the full
    tenant-onboarding adopt path. Empty string keeps the legacy
    "wake without binding" behaviour.
    """

    contract_id: str = ""
    runtime_template_id: str = ""
    reason: str = "on_demand"
    strategy: str = "on_demand"
    prewarm: Mapping[str, Any] = field(default_factory=dict)
    template_id: str = ""

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | None) -> "WakeRequest":
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError("WakeRequest payload must be a JSON object")
        prewarm = data.get("prewarm") or {}
        if not isinstance(prewarm, Mapping):
            raise ValueError("WakeRequest.prewarm must be an object")
        return cls(
            contract_id=str(data.get("contract_id", "") or ""),
            runtime_template_id=str(data.get("runtime_template_id", "") or ""),
            reason=str(data.get("reason", "on_demand") or "on_demand"),
            strategy=str(data.get("strategy", "on_demand") or "on_demand"),
            prewarm=dict(prewarm),
            template_id=str(data.get("template_id", "") or ""),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "runtime_template_id": self.runtime_template_id,
            "reason": self.reason,
            "strategy": self.strategy,
            "prewarm": dict(self.prewarm),
        }


@dataclass(frozen=True)
class SleepRequest:
    """Request body for ``POST /dlaas/v1/instances/{ai_id}/sleep``."""

    reason: str = "idle_timeout"
    drain_slow_loop: bool = True
    close_sessions: bool = False
    release_instance: bool = False

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | None) -> "SleepRequest":
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError("SleepRequest payload must be a JSON object")
        return cls(
            reason=str(data.get("reason", "idle_timeout") or "idle_timeout"),
            drain_slow_loop=bool(data.get("drain_slow_loop", True)),
            close_sessions=bool(data.get("close_sessions", False)),
            release_instance=bool(data.get("release_instance", False)),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "drain_slow_loop": self.drain_slow_loop,
            "close_sessions": self.close_sessions,
            "release_instance": self.release_instance,
        }


@dataclass(frozen=True)
class InstanceStatus:
    """Serializable readout for one hosted ``ai_id``."""

    ai_id: str
    lifecycle_state: InstanceLifecycleState
    vertical: str = "unknown"
    session_count: int = 0
    max_sessions: int = 0
    last_interaction_at_ms: int = 0
    last_wake_reason: str = ""
    last_sleep_reason: str = ""
    failure_reason: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "ai_id": self.ai_id,
            "lifecycle_state": self.lifecycle_state.value,
            "vertical": self.vertical,
            "session_count": self.session_count,
            "max_sessions": self.max_sessions,
            "last_interaction_at_ms": self.last_interaction_at_ms,
            "last_wake_reason": self.last_wake_reason,
            "last_sleep_reason": self.last_sleep_reason,
            "failure_reason": self.failure_reason,
        }


__all__ = [
    "InstanceLifecycleState",
    "InstanceStatus",
    "SleepRequest",
    "WakeRequest",
]
