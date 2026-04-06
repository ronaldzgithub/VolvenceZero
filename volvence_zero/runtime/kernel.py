from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import Any, ClassVar, Generic, Mapping, MutableMapping, TypeVar
from uuid import uuid4


ValueT = TypeVar("ValueT")
UpstreamDict = dict[str, "Snapshot[Any]"]


def utc_now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


class ContractViolationError(RuntimeError):
    """Base error for snapshot-contract violations."""


class OwnershipViolationError(ContractViolationError):
    """Raised when multiple owners publish to the same slot."""


class DependencyViolationError(ContractViolationError):
    """Raised when a module consumes an undeclared upstream slot."""


class SchemaViolationError(ContractViolationError):
    """Raised when a snapshot does not match its declared schema."""


class ImmutabilityViolationError(ContractViolationError):
    """Raised when a published snapshot changes after publication."""


class WiringLevel(str, Enum):
    DISABLED = "disabled"
    SHADOW = "shadow"
    ACTIVE = "active"


@dataclass(frozen=True)
class RuntimePlaceholderValue:
    reason: str
    expected_slot: str
    produced_by: str
    detail: str


@dataclass(frozen=True)
class Snapshot(Generic[ValueT]):
    slot_name: str
    owner: str
    version: int
    timestamp_ms: int
    value: ValueT


@dataclass(frozen=True)
class DebugEvent:
    event_id: str
    timestamp_ms: int
    event_type: str
    module_owner: str
    wave_id: str
    session_id: str
    payload: dict[str, Any]
    parent_event_id: str | None = None


@dataclass(frozen=True)
class ModuleRegistration:
    slot_name: str
    owner: str
    value_type: type[Any]
    dependencies: tuple[str, ...]
    wiring_level: WiringLevel


class EventRecorder:
    """In-memory recorder for Layer 1 structured events."""

    def __init__(self) -> None:
        self._events: list[DebugEvent] = []

    @property
    def events(self) -> tuple[DebugEvent, ...]:
        return tuple(self._events)

    def emit(
        self,
        *,
        event_type: str,
        module_owner: str,
        wave_id: str,
        session_id: str,
        payload: dict[str, Any],
        parent_event_id: str | None = None,
    ) -> DebugEvent:
        event = DebugEvent(
            event_id=str(uuid4()),
            timestamp_ms=utc_now_ms(),
            event_type=event_type,
            module_owner=module_owner,
            wave_id=wave_id,
            session_id=session_id,
            payload=payload,
            parent_event_id=parent_event_id,
        )
        self._events.append(event)
        return event


def _normalize(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _normalize(asdict(value))
    if isinstance(value, dict):
        return {key: _normalize(val) for key, val in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def stable_value_hash(value: Any) -> str:
    payload = json.dumps(_normalize(value), sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def is_frozen_dataclass_instance(value: Any) -> bool:
    if not is_dataclass(value) or isinstance(value, type):
        return False
    return bool(value.__dataclass_params__.frozen)


def is_placeholder_snapshot(snapshot: Snapshot[Any]) -> bool:
    return isinstance(snapshot.value, RuntimePlaceholderValue)


def snapshot_description(snapshot: Snapshot[Any]) -> str:
    description = getattr(snapshot.value, "description", None)
    if isinstance(description, str) and description:
        return description
    if is_placeholder_snapshot(snapshot):
        return snapshot.value.detail
    return type(snapshot.value).__name__


def make_placeholder_snapshot(
    *,
    slot_name: str,
    owner: str,
    version: int,
    reason: str,
    detail: str,
    timestamp_ms: int | None = None,
) -> Snapshot[RuntimePlaceholderValue]:
    return Snapshot(
        slot_name=slot_name,
        owner=owner,
        version=version,
        timestamp_ms=utc_now_ms() if timestamp_ms is None else timestamp_ms,
        value=RuntimePlaceholderValue(
            reason=reason,
            expected_slot=slot_name,
            produced_by=owner,
            detail=detail,
        ),
    )


class SlotRegistry:
    """Tracks slot ownership, schema declarations, and latest versions."""

    def __init__(self) -> None:
        self._registrations: dict[str, ModuleRegistration] = {}
        self._latest_versions: dict[str, int] = {}

    def register(self, module: "RuntimeModule[Any]") -> ModuleRegistration:
        registration = ModuleRegistration(
            slot_name=module.slot_name,
            owner=module.owner,
            value_type=module.value_type,
            dependencies=tuple(module.dependencies),
            wiring_level=module.wiring_level,
        )
        existing = self._registrations.get(module.slot_name)
        if existing is not None and existing.owner != module.owner:
            raise OwnershipViolationError(
                f"Slot '{module.slot_name}' already owned by '{existing.owner}', "
                f"cannot register '{module.owner}'."
            )
        self._registrations[module.slot_name] = registration
        self._latest_versions.setdefault(module.slot_name, 0)
        return registration

    def get_registration(self, slot_name: str) -> ModuleRegistration:
        registration = self._registrations.get(slot_name)
        if registration is None:
            raise OwnershipViolationError(f"Slot '{slot_name}' is not registered.")
        return registration

    def next_version(self, slot_name: str) -> int:
        current = self._latest_versions.get(slot_name, 0)
        return current + 1

    def seed_versions(self, snapshots: Mapping[str, Snapshot[Any]]) -> None:
        for slot_name, snapshot in snapshots.items():
            current = self._latest_versions.get(slot_name, 0)
            if snapshot.version > current:
                self._latest_versions[slot_name] = snapshot.version

    def record_publication(
        self,
        *,
        module: "RuntimeModule[Any]",
        snapshot: Snapshot[Any],
    ) -> ModuleRegistration:
        registration = self.get_registration(module.slot_name)
        if snapshot.slot_name != registration.slot_name:
            raise OwnershipViolationError(
                f"Module '{module.owner}' published slot '{snapshot.slot_name}' "
                f"but owns '{registration.slot_name}'."
            )
        if snapshot.owner != registration.owner:
            raise OwnershipViolationError(
                f"Slot '{snapshot.slot_name}' must be published by '{registration.owner}', "
                f"got '{snapshot.owner}'."
            )
        latest = self._latest_versions.get(snapshot.slot_name, 0)
        if snapshot.version <= latest:
            raise OwnershipViolationError(
                f"Slot '{snapshot.slot_name}' version must increase monotonically: "
                f"got {snapshot.version}, latest is {latest}."
            )
        self._latest_versions[snapshot.slot_name] = snapshot.version
        return registration


class OwnershipGuard:
    """Ensures each slot has a single registered owner and monotonic versions."""

    def __init__(self, registry: SlotRegistry) -> None:
        self._registry = registry

    def register(self, module: "RuntimeModule[Any]") -> ModuleRegistration:
        return self._registry.register(module)

    def validate_publication(
        self,
        *,
        module: "RuntimeModule[Any]",
        snapshot: Snapshot[Any],
    ) -> ModuleRegistration:
        return self._registry.record_publication(module=module, snapshot=snapshot)


class ImmutabilityGuard:
    """Verifies that a published snapshot has not changed since publication."""

    def __init__(self) -> None:
        self._hashes: dict[tuple[str, int], str] = {}

    def remember(self, snapshot: Snapshot[Any]) -> None:
        self._hashes[(snapshot.slot_name, snapshot.version)] = stable_value_hash(snapshot.value)

    def verify(self, snapshot: Snapshot[Any]) -> None:
        key = (snapshot.slot_name, snapshot.version)
        expected = self._hashes.get(key)
        if expected is None:
            self.remember(snapshot)
            return
        current = stable_value_hash(snapshot.value)
        if current != expected:
            raise ImmutabilityViolationError(
                f"Snapshot '{snapshot.slot_name}' version {snapshot.version} changed after publication."
            )


class DependencyGuard:
    """Ensures modules only consume declared upstream slots."""

    def validate_access(
        self,
        *,
        consumer_owner: str,
        declared_dependencies: tuple[str, ...],
        slot_name: str,
    ) -> None:
        if slot_name not in declared_dependencies:
            raise DependencyViolationError(
                f"Module '{consumer_owner}' attempted to consume undeclared slot '{slot_name}'."
            )


class SchemaGuard:
    """Ensures published values match their declared frozen dataclass schema."""

    def validate_snapshot(self, snapshot: Snapshot[Any], expected_value_type: type[Any]) -> None:
        if is_placeholder_snapshot(snapshot):
            return
        if not is_frozen_dataclass_instance(snapshot.value):
            raise SchemaViolationError(
                f"Snapshot '{snapshot.slot_name}' must publish a frozen dataclass value."
            )
        if not isinstance(snapshot.value, expected_value_type):
            raise SchemaViolationError(
                f"Snapshot '{snapshot.slot_name}' expected value type "
                f"'{expected_value_type.__name__}', got '{type(snapshot.value).__name__}'."
            )


class UpstreamView(Mapping[str, Snapshot[Any]]):
    """Guarded upstream mapping with standardized missing-slot semantics."""

    def __init__(
        self,
        *,
        module: "RuntimeModule[Any]",
        active_snapshots: UpstreamDict,
        missing_snapshots: MutableMapping[str, Snapshot[Any]],
        dependency_guard: DependencyGuard,
        immutability_guard: ImmutabilityGuard,
        recorder: EventRecorder,
        session_id: str,
        wave_id: str,
    ) -> None:
        self._module = module
        self._active_snapshots = active_snapshots
        self._missing_snapshots = missing_snapshots
        self._dependency_guard = dependency_guard
        self._immutability_guard = immutability_guard
        self._recorder = recorder
        self._session_id = session_id
        self._wave_id = wave_id

    def _emit_violation(self, slot_name: str, error: ContractViolationError) -> None:
        self._recorder.emit(
            event_type="contract.violation",
            module_owner=self._module.owner,
            wave_id=self._wave_id,
            session_id=self._session_id,
            payload={
                "violation_type": type(error).__name__,
                "module": self._module.owner,
                "slot_name": slot_name,
                "details": str(error),
            },
        )

    def _consume(self, slot_name: str) -> Snapshot[Any]:
        try:
            self._dependency_guard.validate_access(
                consumer_owner=self._module.owner,
                declared_dependencies=tuple(self._module.dependencies),
                slot_name=slot_name,
            )
            snapshot = self._active_snapshots.get(slot_name)
            if snapshot is None:
                snapshot = self._missing_snapshots.get(slot_name)
                if snapshot is None:
                    snapshot = make_placeholder_snapshot(
                        slot_name=slot_name,
                        owner="runtime.missing",
                        version=0,
                        reason="missing-upstream",
                        detail=f"Upstream slot '{slot_name}' is unavailable for '{self._module.owner}'.",
                    )
                    self._missing_snapshots[slot_name] = snapshot
            self._immutability_guard.verify(snapshot)
        except ContractViolationError as error:
            self._emit_violation(slot_name, error)
            raise

        self._recorder.emit(
            event_type="snapshot.consumed",
            module_owner=self._module.owner,
            wave_id=self._wave_id,
            session_id=self._session_id,
            payload={
                "consumer": self._module.owner,
                "slot_name": snapshot.slot_name,
                "version": snapshot.version,
            },
        )
        return snapshot

    def __getitem__(self, key: str) -> Snapshot[Any]:
        return self._consume(key)

    def __iter__(self):
        return iter(self._active_snapshots)

    def __len__(self) -> int:
        return len(self._active_snapshots)

    def get(self, key: str, default: Any = None) -> Snapshot[Any] | Any:
        if key in self._active_snapshots or key in self._module.dependencies:
            return self._consume(key)
        return default


class RuntimeModule(ABC, Generic[ValueT]):
    """Base module contract for all runtime owners."""

    slot_name: ClassVar[str]
    owner: ClassVar[str]
    value_type: ClassVar[type[Any]]
    dependencies: ClassVar[tuple[str, ...]] = ()
    default_wiring_level: ClassVar[WiringLevel] = WiringLevel.ACTIVE

    def __init__(self, *, wiring_level: WiringLevel | None = None) -> None:
        self._wiring_level = wiring_level or self.default_wiring_level
        self._version = 0

    @property
    def wiring_level(self) -> WiringLevel:
        return self._wiring_level

    def seed_version(self, version: int) -> None:
        if version > self._version:
            self._version = version

    def publish(self, value: ValueT) -> Snapshot[ValueT]:
        self._version += 1
        return Snapshot(
            slot_name=self.slot_name,
            owner=self.owner,
            version=self._version,
            timestamp_ms=utc_now_ms(),
            value=value,
        )

    @abstractmethod
    async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[ValueT]:
        """Produce the module's next snapshot from upstream snapshots."""

    async def process_standalone(self, **kwargs: Any) -> Snapshot[ValueT]:
        raise NotImplementedError


async def propagate(
    modules: list[RuntimeModule[Any]],
    *,
    upstream: UpstreamDict | None = None,
    registry: SlotRegistry | None = None,
    recorder: EventRecorder | None = None,
    shadow_snapshots: MutableMapping[str, Snapshot[Any]] | None = None,
    session_id: str = "runtime",
    wave_id: str = "wave-0",
) -> UpstreamDict:
    """
    Execute modules in order against active upstream snapshots.

    Wiring semantics:
    - disabled: publish a stub placeholder snapshot into the active chain
    - shadow: execute and validate, but keep output out of the active chain
    - active: execute, validate, and publish into the active chain
    """

    active_snapshots = dict(upstream or {})
    registry = registry or SlotRegistry()
    ownership_guard = OwnershipGuard(registry)
    recorder = recorder or EventRecorder()
    dependency_guard = DependencyGuard()
    immutability_guard = ImmutabilityGuard()
    schema_guard = SchemaGuard()
    missing_snapshots: dict[str, Snapshot[Any]] = {}

    for snapshot in active_snapshots.values():
        immutability_guard.remember(snapshot)

    for module in modules:
        registration = ownership_guard.register(module)
        if module.wiring_level is WiringLevel.DISABLED:
            placeholder = make_placeholder_snapshot(
                slot_name=module.slot_name,
                owner=module.owner,
                version=registry.next_version(module.slot_name),
                reason="disabled-module",
                detail=f"Module '{module.owner}' is disabled and publishes a runtime stub.",
            )
            ownership_guard.validate_publication(module=module, snapshot=placeholder)
            immutability_guard.remember(placeholder)
            recorder.emit(
                event_type="snapshot.published",
                module_owner=module.owner,
                wave_id=wave_id,
                session_id=session_id,
                payload={
                    "slot_name": placeholder.slot_name,
                    "version": placeholder.version,
                    "value_hash": stable_value_hash(placeholder.value),
                    "description": snapshot_description(placeholder),
                    "wiring_level": module.wiring_level.value,
                },
            )
            active_snapshots[placeholder.slot_name] = placeholder
            continue

        upstream_view = UpstreamView(
            module=module,
            active_snapshots=active_snapshots,
            missing_snapshots=missing_snapshots,
            dependency_guard=dependency_guard,
            immutability_guard=immutability_guard,
            recorder=recorder,
            session_id=session_id,
            wave_id=wave_id,
        )

        try:
            snapshot = await module.process(upstream_view)
            schema_guard.validate_snapshot(snapshot, registration.value_type)
            ownership_guard.validate_publication(module=module, snapshot=snapshot)
        except ContractViolationError as error:
            recorder.emit(
                event_type="contract.violation",
                module_owner=module.owner,
                wave_id=wave_id,
                session_id=session_id,
                payload={
                    "violation_type": type(error).__name__,
                    "module": module.owner,
                    "slot_name": module.slot_name,
                    "details": str(error),
                },
            )
            raise

        immutability_guard.remember(snapshot)
        recorder.emit(
            event_type="snapshot.published",
            module_owner=module.owner,
            wave_id=wave_id,
            session_id=session_id,
            payload={
                "slot_name": snapshot.slot_name,
                "version": snapshot.version,
                "value_hash": stable_value_hash(snapshot.value),
                "description": snapshot_description(snapshot),
                "wiring_level": module.wiring_level.value,
            },
        )

        if module.wiring_level is WiringLevel.SHADOW:
            if shadow_snapshots is not None:
                shadow_snapshots[snapshot.slot_name] = snapshot
            continue

        active_snapshots[snapshot.slot_name] = snapshot

    return active_snapshots
