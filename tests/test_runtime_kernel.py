from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from volvence_zero.runtime import (
    DependencyViolationError,
    EventRecorder,
    OwnershipViolationError,
    RuntimeModule,
    SlotRegistry,
    Snapshot,
    WiringLevel,
    propagate,
)


@dataclass(frozen=True)
class ProducerValue:
    description: str
    value: int


@dataclass(frozen=True)
class ConsumerValue:
    description: str
    seen_value: int


class ProducerModule(RuntimeModule[ProducerValue]):
    slot_name = "producer"
    owner = "ProducerModule"
    value_type = ProducerValue

    async def process(self, upstream):
        return self.publish(ProducerValue(description="producer snapshot", value=7))


class ConsumerModule(RuntimeModule[ConsumerValue]):
    slot_name = "consumer"
    owner = "ConsumerModule"
    value_type = ConsumerValue
    dependencies = ("producer",)

    async def process(self, upstream):
        producer_snapshot = upstream["producer"]
        value = producer_snapshot.value.value
        return self.publish(ConsumerValue(description="consumer snapshot", seen_value=value))


class BadDependencyModule(RuntimeModule[ConsumerValue]):
    slot_name = "bad_consumer"
    owner = "BadConsumerModule"
    value_type = ConsumerValue
    dependencies = ()

    async def process(self, upstream):
        upstream["producer"]
        return self.publish(ConsumerValue(description="bad", seen_value=0))


class ShadowProducerModule(ProducerModule):
    default_wiring_level = WiringLevel.SHADOW


class DisabledProducerModule(ProducerModule):
    default_wiring_level = WiringLevel.DISABLED


class DuplicateProducerModule(RuntimeModule[ProducerValue]):
    slot_name = "producer"
    owner = "AnotherProducerModule"
    value_type = ProducerValue

    async def process(self, upstream):
        return self.publish(ProducerValue(description="duplicate", value=1))


def test_active_modules_publish_into_active_chain():
    recorder = EventRecorder()
    result = asyncio.run(
        propagate(
            [ProducerModule(), ConsumerModule()],
            recorder=recorder,
            session_id="s1",
            wave_id="w1",
        )
    )

    assert set(result) == {"producer", "consumer"}
    assert result["producer"].value.value == 7
    assert result["consumer"].value.seen_value == 7
    assert [event.event_type for event in recorder.events].count("snapshot.published") == 2
    assert [event.event_type for event in recorder.events].count("snapshot.consumed") == 1


def test_shadow_modules_do_not_mutate_active_chain():
    shadow_snapshots: dict[str, Snapshot[object]] = {}
    result = asyncio.run(
        propagate(
            [ShadowProducerModule()],
            shadow_snapshots=shadow_snapshots,
            session_id="s1",
            wave_id="w1",
        )
    )

    assert "producer" not in result
    assert "producer" in shadow_snapshots
    assert shadow_snapshots["producer"].value.value == 7


def test_disabled_modules_publish_placeholder_into_active_chain():
    result = asyncio.run(
        propagate(
            [DisabledProducerModule()],
            session_id="s1",
            wave_id="w1",
        )
    )

    snapshot = result["producer"]
    assert snapshot.value.reason == "disabled-module"
    assert snapshot.version == 1


def test_dependency_guard_fails_loudly_for_undeclared_access():
    with pytest.raises(DependencyViolationError):
        asyncio.run(
            propagate(
                [ProducerModule(), BadDependencyModule()],
                session_id="s1",
                wave_id="w1",
            )
        )


def test_slot_registry_rejects_second_owner():
    registry = SlotRegistry()
    registry.register(ProducerModule())
    with pytest.raises(OwnershipViolationError):
        registry.register(DuplicateProducerModule())
