from __future__ import annotations

import base64
from pathlib import Path

import pytest

from lifeform_service.session_manager import (
    MemoryScopeNotConfiguredError,
    SessionManager,
)
from volvence_zero.memory import (
    MemoryStratum,
    MemoryWriteRequest,
    Track,
    UserIdentity,
    build_scoped_memory_store,
)


def _manager(*, root: Path | None, tenant: str, strategy: str) -> SessionManager:
    manager = SessionManager.__new__(SessionManager)
    manager._alpha_memory_scope_root_dir = str(root) if root is not None else None  # noqa: SLF001
    manager._tenant_id = tenant  # noqa: SLF001
    manager._scope_strategy = strategy  # noqa: SLF001
    manager._memory_backend_name = ""  # noqa: SLF001
    return manager


def _persist(root: Path, *, user_id: str, scope_key: str) -> None:
    store = build_scoped_memory_store(
        identity=UserIdentity(user_id=user_id, scope_key=scope_key),
        root_dir=root,
    )
    store.write(
        MemoryWriteRequest(
            content="The player changed the character's destination.",
            track=Track.WORLD,
            stratum=MemoryStratum.DURABLE,
            strength=0.9,
        ),
        timestamp_ms=1_800_000_000_000,
    )
    assert store.save_to_backend()


def test_export_does_not_require_a_live_session_and_uses_two_layer_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _persist(tmp_path, user_id="player-1", scope_key="tenant-a:player-1")
    manager = _manager(
        root=tmp_path,
        tenant="tenant-a",
        strategy="tenant_ai_end_user",
    )
    # A later process-env mutation must not redirect this already-resolved
    # manager away from its configured filesystem lane.
    monkeypatch.setenv("VZ_MEMORY_BACKEND", "memory")

    exported = manager.export_persisted_memory_scope("player-1")

    assert exported is not None
    assert exported["encoding"] == "base64"
    assert base64.b64decode(str(exported["payload_base64"]), validate=True)
    receipt = exported["receipt"]
    assert isinstance(receipt, dict)
    assert receipt["durability"] == "restart_durable"
    assert receipt["entry_count"] == 1


def test_export_missing_scope_is_none_without_creating_session(tmp_path: Path) -> None:
    manager = _manager(root=tmp_path, tenant="", strategy="")
    assert manager.export_persisted_memory_scope("missing") is None


def test_export_requires_a_configured_durable_scope_root() -> None:
    manager = _manager(root=None, tenant="tenant-a", strategy="tenant_ai_end_user")
    with pytest.raises(MemoryScopeNotConfiguredError):
        manager.export_persisted_memory_scope("player-1")
