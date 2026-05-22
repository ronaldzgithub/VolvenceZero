"""Versioned DLaaS adoption configuration.

Adoption is where a tenant chooses the class of life being adopted:
vertical, substrate profile, protocol set, memory policy, tool policy,
ops policy, and training policy. The platform owns this contract; the
kernel only receives the resolved runtime inputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


def _mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key) or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"AdoptionConfig.{key} must be an object")
    return value


def _str_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raise ValueError("expected list of strings, got string")
    return tuple(str(item) for item in (value or ()))


@dataclass(frozen=True)
class VerticalSelection:
    vertical_id: str = ""
    runtime_template_id: str = ""
    profile_id: str = ""

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | None) -> "VerticalSelection":
        raw = data or {}
        if not isinstance(raw, Mapping):
            raise ValueError("VerticalSelection must be an object")
        vertical_id = str(raw.get("vertical_id", "") or "")
        runtime_template_id = str(
            raw.get("runtime_template_id", vertical_id) or vertical_id
        )
        return cls(
            vertical_id=vertical_id,
            runtime_template_id=runtime_template_id,
            profile_id=str(raw.get("profile_id", "") or ""),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "vertical_id": self.vertical_id,
            "runtime_template_id": self.runtime_template_id,
            "profile_id": self.profile_id,
        }


@dataclass(frozen=True)
class SubstrateSelection:
    substrate_profile_id: str = ""
    mode: str = "shared_frozen"
    adapter_policy: str = "none"
    allow_rare_heavy_refresh: bool = False

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | None) -> "SubstrateSelection":
        raw = data or {}
        if not isinstance(raw, Mapping):
            raise ValueError("SubstrateSelection must be an object")
        return cls(
            substrate_profile_id=str(raw.get("substrate_profile_id", "") or ""),
            mode=str(raw.get("mode", "shared_frozen") or "shared_frozen"),
            adapter_policy=str(raw.get("adapter_policy", "none") or "none"),
            allow_rare_heavy_refresh=bool(
                raw.get("allow_rare_heavy_refresh", False)
            ),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "substrate_profile_id": self.substrate_profile_id,
            "mode": self.mode,
            "adapter_policy": self.adapter_policy,
            "allow_rare_heavy_refresh": self.allow_rare_heavy_refresh,
        }


@dataclass(frozen=True)
class ProtocolSelection:
    autoload: tuple[str, ...] = ()
    library_ids: tuple[str, ...] = ()
    activation_policy: str = "explicit_load"
    allow_runtime_upload: bool = False
    review_level_required: str = "L3"

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | None) -> "ProtocolSelection":
        raw = data or {}
        if not isinstance(raw, Mapping):
            raise ValueError("ProtocolSelection must be an object")
        return cls(
            autoload=_str_tuple(raw.get("autoload")),
            library_ids=_str_tuple(raw.get("library_ids")),
            activation_policy=str(
                raw.get("activation_policy", "explicit_load") or "explicit_load"
            ),
            allow_runtime_upload=bool(raw.get("allow_runtime_upload", False)),
            review_level_required=str(
                raw.get("review_level_required", "L3") or "L3"
            ),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "autoload": list(self.autoload),
            "library_ids": list(self.library_ids),
            "activation_policy": self.activation_policy,
            "allow_runtime_upload": self.allow_runtime_upload,
            "review_level_required": self.review_level_required,
        }


@dataclass(frozen=True)
class MemoryPolicySelection:
    scope_strategy: str = "tenant_ai_end_user"
    retention_policy_id: str = ""
    deletion_policy_id: str = ""

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | None) -> "MemoryPolicySelection":
        raw = data or {}
        if not isinstance(raw, Mapping):
            raise ValueError("MemoryPolicySelection must be an object")
        return cls(
            scope_strategy=str(
                raw.get("scope_strategy", "tenant_ai_end_user")
                or "tenant_ai_end_user"
            ),
            retention_policy_id=str(raw.get("retention_policy_id", "") or ""),
            deletion_policy_id=str(raw.get("deletion_policy_id", "") or ""),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "scope_strategy": self.scope_strategy,
            "retention_policy_id": self.retention_policy_id,
            "deletion_policy_id": self.deletion_policy_id,
        }


@dataclass(frozen=True)
class ToolPolicySelection:
    tool_policy_id: str = ""
    allowed_capabilities: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | None) -> "ToolPolicySelection":
        raw = data or {}
        if not isinstance(raw, Mapping):
            raise ValueError("ToolPolicySelection must be an object")
        return cls(
            tool_policy_id=str(raw.get("tool_policy_id", "") or ""),
            allowed_capabilities=_str_tuple(raw.get("allowed_capabilities")),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "tool_policy_id": self.tool_policy_id,
            "allowed_capabilities": list(self.allowed_capabilities),
        }


@dataclass(frozen=True)
class OpsPolicySelection:
    awake_strategy: str = "on_demand"
    idle_sleep_seconds: int = 1800
    handoff_policy_id: str = ""
    pause_on_handoff: bool = True

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | None) -> "OpsPolicySelection":
        raw = data or {}
        if not isinstance(raw, Mapping):
            raise ValueError("OpsPolicySelection must be an object")
        return cls(
            awake_strategy=str(raw.get("awake_strategy", "on_demand") or "on_demand"),
            idle_sleep_seconds=int(raw.get("idle_sleep_seconds", 1800) or 1800),
            handoff_policy_id=str(raw.get("handoff_policy_id", "") or ""),
            pause_on_handoff=bool(raw.get("pause_on_handoff", True)),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "awake_strategy": self.awake_strategy,
            "idle_sleep_seconds": self.idle_sleep_seconds,
            "handoff_policy_id": self.handoff_policy_id,
            "pause_on_handoff": self.pause_on_handoff,
        }


@dataclass(frozen=True)
class TrainingPolicySelection:
    allow_protocol_intake: bool = True
    allow_corpus_intake: bool = True
    allow_adapter_training: bool = False
    promotion_gate: str = "reviewed_protocol_only"

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | None) -> "TrainingPolicySelection":
        raw = data or {}
        if not isinstance(raw, Mapping):
            raise ValueError("TrainingPolicySelection must be an object")
        return cls(
            allow_protocol_intake=bool(raw.get("allow_protocol_intake", True)),
            allow_corpus_intake=bool(raw.get("allow_corpus_intake", True)),
            allow_adapter_training=bool(raw.get("allow_adapter_training", False)),
            promotion_gate=str(
                raw.get("promotion_gate", "reviewed_protocol_only")
                or "reviewed_protocol_only"
            ),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "allow_protocol_intake": self.allow_protocol_intake,
            "allow_corpus_intake": self.allow_corpus_intake,
            "allow_adapter_training": self.allow_adapter_training,
            "promotion_gate": self.promotion_gate,
        }


@dataclass(frozen=True)
class AdoptionConfig:
    vertical: VerticalSelection = field(default_factory=VerticalSelection)
    substrate: SubstrateSelection = field(default_factory=SubstrateSelection)
    protocols: ProtocolSelection = field(default_factory=ProtocolSelection)
    memory: MemoryPolicySelection = field(default_factory=MemoryPolicySelection)
    tools: ToolPolicySelection = field(default_factory=ToolPolicySelection)
    ops: OpsPolicySelection = field(default_factory=OpsPolicySelection)
    training: TrainingPolicySelection = field(default_factory=TrainingPolicySelection)
    version: int = 1

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | None) -> "AdoptionConfig":
        raw = data or {}
        if not isinstance(raw, Mapping):
            raise ValueError("AdoptionConfig must be an object")
        return cls(
            vertical=VerticalSelection.from_json(_mapping(raw, "vertical")),
            substrate=SubstrateSelection.from_json(_mapping(raw, "substrate")),
            protocols=ProtocolSelection.from_json(_mapping(raw, "protocols")),
            memory=MemoryPolicySelection.from_json(_mapping(raw, "memory")),
            tools=ToolPolicySelection.from_json(_mapping(raw, "tools")),
            ops=OpsPolicySelection.from_json(_mapping(raw, "ops")),
            training=TrainingPolicySelection.from_json(_mapping(raw, "training")),
            version=int(raw.get("version", 1) or 1),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "vertical": self.vertical.to_json(),
            "substrate": self.substrate.to_json(),
            "protocols": self.protocols.to_json(),
            "memory": self.memory.to_json(),
            "tools": self.tools.to_json(),
            "ops": self.ops.to_json(),
            "training": self.training.to_json(),
        }


__all__ = [
    "AdoptionConfig",
    "MemoryPolicySelection",
    "OpsPolicySelection",
    "ProtocolSelection",
    "SubstrateSelection",
    "ToolPolicySelection",
    "TrainingPolicySelection",
    "VerticalSelection",
]
