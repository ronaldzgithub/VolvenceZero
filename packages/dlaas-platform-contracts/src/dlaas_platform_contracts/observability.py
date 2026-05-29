"""DLaaS observability and life-blueprint contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from dlaas_platform_contracts.adoption import (
    MemoryPolicySelection,
    OpsPolicySelection,
    ProtocolSelection,
    SubstrateSelection,
    ToolPolicySelection,
    TrainingPolicySelection,
    VerticalSelection,
)


class ReadoutView(str, Enum):
    SUMMARY = "summary"
    FULL = "full"


@dataclass(frozen=True)
class ReadoutBundle:
    ai_id: str
    session_id: str
    view: ReadoutView = ReadoutView.SUMMARY
    body: Mapping[str, Any] = field(default_factory=dict)
    cognition: Mapping[str, Any] = field(default_factory=dict)
    knowledge: Mapping[str, Any] = field(default_factory=dict)
    strategy: Mapping[str, Any] = field(default_factory=dict)
    protocol: Mapping[str, Any] = field(default_factory=dict)
    safety: Mapping[str, Any] = field(default_factory=dict)
    training: Mapping[str, Any] = field(default_factory=dict)
    # Social cognition readout (R16-R20): the InterlocutorState axes plus
    # conversational-role / common-ground projections. Additive + optional
    # (defaults empty) so existing readout consumers are unaffected; the
    # `/dlaas/v1/cognition/interlocutor` endpoint surfaces this block.
    social: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "ai_id": self.ai_id,
            "session_id": self.session_id,
            "view": self.view.value,
            "body": dict(self.body),
            "cognition": dict(self.cognition),
            "knowledge": dict(self.knowledge),
            "strategy": dict(self.strategy),
            "protocol": dict(self.protocol),
            "safety": dict(self.safety),
            "training": dict(self.training),
            "social": dict(self.social),
        }


@dataclass(frozen=True)
class SnapshotExportRequest:
    ai_id: str
    session_id: str
    slots: tuple[str, ...] = ()
    include_shadow: bool = False

    @classmethod
    def from_query(
        cls,
        *,
        ai_id: str,
        session_id: str,
        slots: tuple[str, ...],
        include_shadow: bool,
    ) -> "SnapshotExportRequest":
        if not session_id.strip():
            raise ValueError("SnapshotExportRequest.session_id must be non-empty")
        return cls(
            ai_id=ai_id,
            session_id=session_id,
            slots=slots,
            include_shadow=include_shadow,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "ai_id": self.ai_id,
            "session_id": self.session_id,
            "slots": list(self.slots),
            "include_shadow": self.include_shadow,
        }


@dataclass(frozen=True)
class ExplainTrace:
    ai_id: str
    session_id: str
    turn_index: str = "latest"
    chain: tuple[Mapping[str, Any], ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "ai_id": self.ai_id,
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "chain": [dict(step) for step in self.chain],
        }


@dataclass(frozen=True)
class LifeBlueprint:
    blueprint_id: str
    display_name: str
    vertical: VerticalSelection
    substrate: SubstrateSelection = field(default_factory=SubstrateSelection)
    protocols: ProtocolSelection = field(default_factory=ProtocolSelection)
    memory: MemoryPolicySelection = field(default_factory=MemoryPolicySelection)
    tools: ToolPolicySelection = field(default_factory=ToolPolicySelection)
    ops: OpsPolicySelection = field(default_factory=OpsPolicySelection)
    training: TrainingPolicySelection = field(default_factory=TrainingPolicySelection)
    evaluation_gates: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "blueprint_id": self.blueprint_id,
            "display_name": self.display_name,
            "vertical": self.vertical.to_json(),
            "substrate": self.substrate.to_json(),
            "protocols": self.protocols.to_json(),
            "memory": self.memory.to_json(),
            "tools": self.tools.to_json(),
            "ops": self.ops.to_json(),
            "training": self.training.to_json(),
            "evaluation_gates": list(self.evaluation_gates),
        }


__all__ = [
    "ExplainTrace",
    "LifeBlueprint",
    "ReadoutBundle",
    "ReadoutView",
    "SnapshotExportRequest",
]
