from __future__ import annotations

import hashlib
import os
from pathlib import Path

from volvence_zero.memory import (
    MEMORY_CHECKPOINT_RECEIPT_SCHEMA,
    MEMORY_CHECKPOINT_RECEIPT_SCHEMA_VERSION,
    FileSystemPersistenceBackend,
    InMemoryPersistenceBackend,
    MemoryStore,
    MemoryStratum,
    MemoryWriteRequest,
    Track,
    build_default_memory_store,
)


_TIMESTAMP_MS = 1_800_000_000_000


def _write_restart_witness(store: MemoryStore) -> None:
    store.write(
        MemoryWriteRequest(
            content="The north gate witness remembered the changed rendezvous.",
            track=Track.WORLD,
            stratum=MemoryStratum.DURABLE,
            tags=("scene:north-gate", "event:changed-rendezvous"),
            strength=0.91,
        ),
        timestamp_ms=_TIMESTAMP_MS,
    )


def test_filesystem_save_uses_same_directory_fsync_and_atomic_replace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backend = FileSystemPersistenceBackend(base_dir=str(tmp_path))
    fsync_calls: list[int] = []
    replace_calls: list[tuple[Path, Path]] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def observed_fsync(file_descriptor: int) -> None:
        fsync_calls.append(file_descriptor)
        real_fsync(file_descriptor)

    def observed_replace(source: str | bytes, destination: str | bytes) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        assert source_path.parent == destination_path.parent == tmp_path
        assert source_path.read_bytes() == b'{"checkpoint":"actual-bytes"}'
        replace_calls.append((source_path, destination_path))
        real_replace(source, destination)

    monkeypatch.setattr(os, "fsync", observed_fsync)
    monkeypatch.setattr(os, "replace", observed_replace)

    backend.save_checkpoint(
        key="tenant/character",
        data=b'{"checkpoint":"actual-bytes"}',
        version=1,
    )

    assert fsync_calls
    assert len(replace_calls) == 1
    assert backend.durability == "restart_durable"
    assert backend.load_checkpoint(key="tenant/character") == (
        b'{"checkpoint":"actual-bytes"}',
        1,
    )
    assert tuple(tmp_path.glob("*.tmp")) == ()


def test_filesystem_receipt_proves_save_then_fresh_store_load_equivalence(
    tmp_path: Path,
) -> None:
    backend = FileSystemPersistenceBackend(base_dir=str(tmp_path))
    source = build_default_memory_store(persistence_backend=backend)
    _write_restart_witness(source)
    expected_checkpoint = source.create_checkpoint(
        checkpoint_id="persist-memory/store"
    )

    save_receipt = source.save_to_backend_with_receipt()

    assert save_receipt is not None
    persisted_path = tmp_path / "memory__store_v1.json"
    persisted_bytes = persisted_path.read_bytes()
    persisted_sha256 = hashlib.sha256(persisted_bytes).hexdigest()
    assert save_receipt.schema_id == MEMORY_CHECKPOINT_RECEIPT_SCHEMA
    assert save_receipt.schema_version == MEMORY_CHECKPOINT_RECEIPT_SCHEMA_VERSION
    assert save_receipt.operation == "save"
    assert save_receipt.checkpoint_id == "persist-memory/store"
    assert save_receipt.checkpoint_key == "memory/store"
    assert save_receipt.checkpoint_version == 1
    assert save_receipt.payload_sha256 == persisted_sha256
    assert save_receipt.payload_bytes == len(persisted_bytes)
    assert save_receipt.entry_count == 1
    assert save_receipt.durability == "restart_durable"
    assert save_receipt.completed_at_ms > 0
    assert save_receipt.restored_payload_sha256 is None
    assert save_receipt.restored_matches_persisted is None
    assert save_receipt.is_restart_durable is True
    assert save_receipt.to_json() == {
        "schema_id": save_receipt.schema_id,
        "schema_version": save_receipt.schema_version,
        "operation": save_receipt.operation,
        "checkpoint_id": save_receipt.checkpoint_id,
        "checkpoint_key": save_receipt.checkpoint_key,
        "checkpoint_version": save_receipt.checkpoint_version,
        "payload_sha256": save_receipt.payload_sha256,
        "payload_bytes": save_receipt.payload_bytes,
        "entry_count": save_receipt.entry_count,
        "durability": save_receipt.durability,
        "completed_at_ms": save_receipt.completed_at_ms,
        "restored_payload_sha256": None,
        "restored_matches_persisted": None,
    }
    assert source.latest_persistence_receipt == save_receipt

    restarted_backend = FileSystemPersistenceBackend(base_dir=str(tmp_path))
    restarted = build_default_memory_store(persistence_backend=restarted_backend)
    load_receipt = restarted.load_from_backend_with_receipt()

    assert load_receipt is not None
    assert load_receipt.operation == "load"
    assert load_receipt.checkpoint_id == save_receipt.checkpoint_id
    assert load_receipt.checkpoint_key == save_receipt.checkpoint_key
    assert load_receipt.checkpoint_version == save_receipt.checkpoint_version
    assert load_receipt.payload_sha256 == save_receipt.payload_sha256
    assert load_receipt.payload_bytes == save_receipt.payload_bytes
    assert load_receipt.entry_count == save_receipt.entry_count
    assert load_receipt.durability == "restart_durable"
    assert load_receipt.restored_payload_sha256 == save_receipt.payload_sha256
    assert load_receipt.restored_matches_persisted is True
    assert restarted.latest_persistence_receipt == load_receipt
    assert restarted.create_checkpoint(
        checkpoint_id="persist-memory/store"
    ) == expected_checkpoint


def test_bool_api_remains_compatible_and_retains_latest_load_receipt(
    tmp_path: Path,
) -> None:
    source = MemoryStore(
        persistence_backend=FileSystemPersistenceBackend(base_dir=str(tmp_path))
    )
    _write_restart_witness(source)
    assert source.save_to_backend() is True

    restarted = MemoryStore(
        persistence_backend=FileSystemPersistenceBackend(base_dir=str(tmp_path))
    )
    assert restarted.load_from_backend() is True
    assert restarted.latest_persistence_receipt is not None
    assert restarted.latest_persistence_receipt.operation == "load"
    assert restarted.latest_persistence_receipt.restored_matches_persisted is True

    unconfigured = MemoryStore()
    assert unconfigured.save_to_backend() is False
    assert unconfigured.load_from_backend() is False
    assert unconfigured.save_to_backend_with_receipt() is None
    assert unconfigured.load_from_backend_with_receipt() is None
    assert unconfigured.latest_persistence_receipt is None


def test_load_receipt_reports_noncanonical_persisted_bytes_as_mismatch(
    tmp_path: Path,
) -> None:
    source = MemoryStore(
        persistence_backend=FileSystemPersistenceBackend(base_dir=str(tmp_path))
    )
    _write_restart_witness(source)
    assert source.save_to_backend()
    persisted_path = tmp_path / "memory__store_v1.json"
    persisted_path.write_bytes(persisted_path.read_bytes() + b" ")

    restarted = MemoryStore(
        persistence_backend=FileSystemPersistenceBackend(base_dir=str(tmp_path))
    )
    receipt = restarted.load_from_backend_with_receipt()

    assert receipt is not None
    assert receipt.operation == "load"
    assert receipt.restored_matches_persisted is False
    assert receipt.restored_payload_sha256 != receipt.payload_sha256


def test_in_memory_receipt_is_explicitly_process_local() -> None:
    backend = InMemoryPersistenceBackend()
    source = MemoryStore(persistence_backend=backend)
    _write_restart_witness(source)

    save_receipt = source.save_to_backend_with_receipt()
    restarted = MemoryStore(persistence_backend=backend)
    load_receipt = restarted.load_from_backend_with_receipt()

    assert backend.durability == "process_local"
    assert save_receipt is not None
    assert save_receipt.durability == "process_local"
    assert save_receipt.is_restart_durable is False
    assert load_receipt is not None
    assert load_receipt.durability == "process_local"
    assert load_receipt.restored_matches_persisted is True
    assert restarted.create_checkpoint(
        checkpoint_id="persist-memory/store"
    ) == source.create_checkpoint(checkpoint_id="persist-memory/store")
