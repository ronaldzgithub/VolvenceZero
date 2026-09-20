from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from volvence_zero.memory import (
    InMemoryPersistenceBackend,
    MemoryCheckpointExport,
    MemoryCheckpointExportError,
    MemoryStratum,
    MemoryWriteRequest,
    Track,
    UserIdentity,
    build_scoped_memory_store,
    export_scoped_memory_checkpoint,
    scoped_memory_dir,
)


def _persist_witness(*, root: Path, identity: UserIdentity) -> bytes:
    store = build_scoped_memory_store(identity=identity, root_dir=root)
    store.write(
        MemoryWriteRequest(
            content="The player moved the rendezvous from the north gate.",
            track=Track.WORLD,
            stratum=MemoryStratum.DURABLE,
            tags=("scene:north-gate", f"user_scope:{identity.scope_key}"),
            strength=0.9,
        ),
        timestamp_ms=1_800_000_000_000,
    )
    assert store.save_to_backend()
    scope_dir = scoped_memory_dir(root_dir=root, user_id=identity.user_id)
    return (scope_dir / "memory__store_v1.json").read_bytes()


def test_scoped_export_returns_exact_restart_durable_backend_bytes(
    tmp_path: Path,
) -> None:
    identity = UserIdentity(user_id="player-1", scope_key="tenant:player-1")
    persisted = _persist_witness(root=tmp_path, identity=identity)

    first = export_scoped_memory_checkpoint(identity=identity, root_dir=tmp_path)
    restarted = export_scoped_memory_checkpoint(identity=identity, root_dir=tmp_path)

    assert first is not None
    assert restarted is not None
    assert first.payload == restarted.payload == persisted
    assert first.receipt.checkpoint_version == 1
    assert first.receipt.entry_count == 1
    assert first.receipt.payload_sha256 == hashlib.sha256(persisted).hexdigest()
    assert first.receipt.durability == "restart_durable"
    assert first.receipt.is_restart_durable is True
    wire = first.to_json()
    assert base64.b64decode(str(wire["payload_base64"]), validate=True) == persisted
    assert wire["receipt"] == first.receipt.to_json()


def test_missing_filesystem_scope_does_not_create_a_directory(tmp_path: Path) -> None:
    identity = UserIdentity(user_id="missing", scope_key="missing")
    scope_dir = scoped_memory_dir(root_dir=tmp_path, user_id=identity.user_id)

    assert export_scoped_memory_checkpoint(identity=identity, root_dir=tmp_path) is None
    assert not scope_dir.exists()


def test_export_reports_process_local_backend_without_upgrading_durability() -> None:
    identity = UserIdentity(user_id="player-1", scope_key="player-1")
    backend = InMemoryPersistenceBackend()
    store = build_scoped_memory_store(
        identity=identity,
        persistence_backend=backend,
    )
    store.write(
        MemoryWriteRequest(
            content="process-local witness",
            track=Track.SELF,
            stratum=MemoryStratum.DURABLE,
            strength=0.8,
        ),
        timestamp_ms=1_800_000_000_000,
    )
    assert store.save_to_backend()

    exported = export_scoped_memory_checkpoint(
        identity=identity,
        persistence_backend=backend,
    )

    assert exported is not None
    assert exported.receipt.durability == "process_local"
    assert exported.receipt.is_restart_durable is False


def test_noncanonical_or_corrupt_persisted_bytes_fail_loud(tmp_path: Path) -> None:
    identity = UserIdentity(user_id="player-1", scope_key="player-1")
    persisted = _persist_witness(root=tmp_path, identity=identity)
    scope_dir = scoped_memory_dir(root_dir=tmp_path, user_id=identity.user_id)
    (scope_dir / "memory__store_v1.json").write_bytes(persisted + b" ")

    with pytest.raises(MemoryCheckpointExportError, match="canonically"):
        export_scoped_memory_checkpoint(identity=identity, root_dir=tmp_path)


def test_export_contract_rejects_payload_hash_mismatch(tmp_path: Path) -> None:
    identity = UserIdentity(user_id="player-1", scope_key="player-1")
    _persist_witness(root=tmp_path, identity=identity)
    exported = export_scoped_memory_checkpoint(identity=identity, root_dir=tmp_path)
    assert exported is not None

    with pytest.raises(ValueError, match="hash disagrees"):
        MemoryCheckpointExport(
            payload=exported.payload,
            receipt=replace(exported.receipt, payload_sha256="0" * 64),
        )
