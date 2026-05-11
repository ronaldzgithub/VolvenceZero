"""Packet 6.8: introspection sibling owners for ProtocolRegistry.

Two new SHADOW-default RuntimeModule owners:

* ``ProtocolRegistryIntrospectionModule`` publishes
  ``protocol_registry`` slot — per-protocol summary including
  review_status, parent linkage, content counts.
* ``ProtocolRevisionLogModule`` publishes
  ``protocol_revision_log`` slot — flattened cross-protocol
  revision audit trail.

Both share the same ``ProtocolRegistry`` handle as
``ProtocolRegistryModule`` (constructor injection) so the
snapshots reflect the live registry state without rebuilding
or duplicating it.

Why split into two modules instead of folding into the
existing ``ProtocolRegistryModule``: a RuntimeModule owns
exactly one slot (R8 SSOT). The active_mixture slot is
already taken; introspection slots get their own owners.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

from volvence_zero.behavior_protocol import (
    ProtocolRegistryEntry,
    ProtocolRegistrySnapshot,
    ProtocolRevisionLogEntry,
    ProtocolRevisionLogSnapshot,
    ReviewStatus,
)
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel

from volvence_zero.protocol_runtime.registry import ProtocolRegistry


class ProtocolRegistryIntrospectionModule(
    RuntimeModule[ProtocolRegistrySnapshot]
):
    slot_name: ClassVar[str] = "protocol_registry"
    owner: ClassVar[str] = "ProtocolRegistryIntrospectionModule"
    value_type: ClassVar[type[Any]] = ProtocolRegistrySnapshot
    dependencies: ClassVar[tuple[str, ...]] = ()
    default_wiring_level: ClassVar[WiringLevel] = WiringLevel.SHADOW

    def __init__(
        self,
        *,
        wiring_level: WiringLevel | None = None,
        registry: ProtocolRegistry,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._registry = registry

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[ProtocolRegistrySnapshot]:
        all_protocols = self._registry.loaded_all()
        entries = tuple(
            ProtocolRegistryEntry(
                protocol_id=p.protocol_id,
                version=p.version,
                advisor_name=p.advisor_name,
                review_status=p.review_status,
                parent_protocol_id=p.parent_protocol_id,
                boundary_count=len(p.boundary_contracts),
                strategy_count=len(p.strategy_priors),
                knowledge_seed_count=len(p.knowledge_seeds),
                signature_case_count=len(p.signature_cases),
                revision_count=len(p.revision_log),
            )
            for p in all_protocols
        )
        active = sum(
            1
            for p in all_protocols
            if p.review_status in {ReviewStatus.SHADOW, ReviewStatus.ACTIVE}
        )
        retired = sum(
            1
            for p in all_protocols
            if p.review_status is ReviewStatus.RETIRED
        )
        snapshot = ProtocolRegistrySnapshot(
            entries=entries,
            active_count=active,
            retired_count=retired,
            description=(
                f"protocol_registry: {len(entries)} loaded, "
                f"{active} active, {retired} retired"
            ),
        )
        return self.publish(snapshot)


class ProtocolRevisionLogModule(
    RuntimeModule[ProtocolRevisionLogSnapshot]
):
    slot_name: ClassVar[str] = "protocol_revision_log"
    owner: ClassVar[str] = "ProtocolRevisionLogModule"
    value_type: ClassVar[type[Any]] = ProtocolRevisionLogSnapshot
    dependencies: ClassVar[tuple[str, ...]] = ()
    default_wiring_level: ClassVar[WiringLevel] = WiringLevel.SHADOW

    def __init__(
        self,
        *,
        wiring_level: WiringLevel | None = None,
        registry: ProtocolRegistry,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._registry = registry

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[ProtocolRevisionLogSnapshot]:
        all_protocols = self._registry.loaded_all()
        entries: list[ProtocolRevisionLogEntry] = []
        for protocol in all_protocols:
            for rev in protocol.revision_log:
                entries.append(
                    ProtocolRevisionLogEntry(
                        protocol_id=protocol.protocol_id,
                        revision_id=rev.revision_id,
                        revised_at_tick=rev.revised_at_tick,
                        revised_by=rev.revised_by,
                        description=rev.description,
                        affected_field=rev.affected_field,
                    )
                )
        snapshot = ProtocolRevisionLogSnapshot(
            entries=tuple(entries),
            description=(
                f"protocol_revision_log: {len(entries)} revision(s) "
                f"across {len(all_protocols)} protocol(s)"
            ),
        )
        return self.publish(snapshot)


__all__ = [
    "ProtocolRegistryIntrospectionModule",
    "ProtocolRevisionLogModule",
]
