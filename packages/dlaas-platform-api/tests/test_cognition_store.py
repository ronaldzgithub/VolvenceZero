"""Tests for the durable cognition snapshot store (Track 4 FULL).

* In-memory backend keeps the historical behaviour.
* SQLite backend persists across a store re-open (simulating a
  platform-api restart) and enforces ``DLAAS_COGNITION_RETENTION_DAYS``.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from dlaas_platform_api.cognition import (
    COGNITION_SNAPSHOTS_KEY,
    CognitionSnapshotStore,
    record_cognition_snapshot,
)


def _row(
    snapshot_id: str,
    captured_at_ms: int,
    ai_id: str = "ai-1",
    *,
    tenant_id: str = "t-1",
    session_id: str = "s-1",
    snapshot_sequence: int | None = None,
) -> dict:
    return {
        "snapshot_id": snapshot_id,
        "tenant_id": tenant_id,
        "ai_id": ai_id,
        "session_id": session_id,
        "source": "interaction",
        "captured_at_ms": captured_at_ms,
        "regime_id": "acquaintance",
        "raw_readout": {"social": {"present": True}},
        **(
            {"snapshot_sequence": snapshot_sequence}
            if snapshot_sequence is not None
            else {}
        ),
    }


def test_in_memory_append_and_iterate() -> None:
    store = CognitionSnapshotStore()
    assert store.durable is False
    store.append(_row("c1", 1000))
    store.append(_row("c2", 2000))
    rows = list(store)
    assert {r["snapshot_id"] for r in rows} == {"c1", "c2"}
    # Row shape (incl. nested raw_readout) is preserved verbatim.
    assert rows[0]["raw_readout"]["social"]["present"] is True


def test_in_memory_sequence_is_partitioned_and_replay_safe() -> None:
    store = CognitionSnapshotStore()
    first = store.append(_row("c1", 3000))
    second = store.append(_row("c2", 1000))
    other_session = store.append(_row("c3", 2000, session_id="s-2"))

    assert first["snapshot_sequence"] == 1
    assert second["snapshot_sequence"] == 2
    assert other_session["snapshot_sequence"] == 1

    replayed = store.append(_row("c1", 4000))
    assert replayed["snapshot_sequence"] == 1
    assert len(list(store)) == 3

    with pytest.raises(ValueError, match="another snapshot"):
        store.append(_row("conflict", 5000, snapshot_sequence=2))


def test_in_memory_explicit_replay_sequence_advances_allocator() -> None:
    store = CognitionSnapshotStore()
    replayed = store.append(_row("restored", 1000, snapshot_sequence=7))
    appended = store.append(_row("new", 2000))
    assert replayed["snapshot_sequence"] == 7
    assert appended["snapshot_sequence"] == 8


def test_record_cognition_snapshot_returns_allocated_sequence() -> None:
    store = CognitionSnapshotStore()
    request = type("Request", (), {"app": {COGNITION_SNAPSHOTS_KEY: store}})()
    snapshot = record_cognition_snapshot(
        request,
        ai_id="ai-1",
        session_id="s-1",
        snapshots={},
        readout_bundle_json={},
        tenant_id="t-1",
    )
    assert snapshot.snapshot_sequence == 1
    assert list(store)[0]["snapshot_sequence"] == 1


def test_sqlite_persists_across_reopen(tmp_path: Path) -> None:
    db = str(tmp_path / "cog.db")
    store = CognitionSnapshotStore(db_path=db)
    assert store.durable is True
    store.append(_row("c1", 1000))
    store.append(_row("c2", 2000))
    # Re-open a fresh store on the same file -> rows survive (durable).
    reopened = CognitionSnapshotStore(db_path=db)
    ids = {r["snapshot_id"] for r in reopened}
    assert ids == {"c1", "c2"}
    assert next(iter(reopened))["raw_readout"]["social"]["present"] is True


def test_sqlite_sequence_is_atomic_partitioned_and_replay_safe(tmp_path: Path) -> None:
    db = str(tmp_path / "cog.db")
    first_store = CognitionSnapshotStore(db_path=db)
    assert first_store.append(_row("c1", 3000))["snapshot_sequence"] == 1
    assert first_store.append(_row("c2", 1000))["snapshot_sequence"] == 2
    assert (
        first_store.append(_row("other", 1000, tenant_id="t-2"))[
            "snapshot_sequence"
        ]
        == 1
    )

    reopened = CognitionSnapshotStore(db_path=db)
    assert reopened.append(_row("c1", 4000))["snapshot_sequence"] == 1
    assert reopened.append(_row("c3", 2000))["snapshot_sequence"] == 3
    by_id = {row["snapshot_id"]: row for row in reopened}
    assert by_id["c1"]["captured_at_ms"] == 4000
    assert by_id["c1"]["snapshot_sequence"] == 1

    with pytest.raises(sqlite3.IntegrityError):
        reopened.append(_row("conflict", 5000, snapshot_sequence=3))


def test_sqlite_legacy_schema_migrates_sequence_and_row_json_atomically(
    tmp_path: Path,
) -> None:
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE cognition_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL DEFAULT '',
            ai_id TEXT NOT NULL DEFAULT '',
            session_id TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT '',
            captured_at_ms INTEGER NOT NULL DEFAULT 0,
            row_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    for row in (
        _row("later", 2000),
        _row("earlier", 1000),
        _row("other-session", 500, session_id="s-2"),
    ):
        conn.execute(
            "INSERT INTO cognition_snapshots VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                row["snapshot_id"],
                row["tenant_id"],
                row["ai_id"],
                row["session_id"],
                row["source"],
                row["captured_at_ms"],
                json.dumps(row),
            ),
        )
    conn.commit()
    conn.close()

    migrated = CognitionSnapshotStore(db_path=str(db))
    by_id = {row["snapshot_id"]: row for row in migrated}
    assert by_id["earlier"]["snapshot_sequence"] == 1
    assert by_id["later"]["snapshot_sequence"] == 2
    assert by_id["other-session"]["snapshot_sequence"] == 1

    check = sqlite3.connect(db)
    check.row_factory = sqlite3.Row
    columns = {
        row["name"]: row for row in check.execute("PRAGMA table_info(cognition_snapshots)")
    }
    assert columns["snapshot_sequence"]["notnull"] == 1
    stored = check.execute(
        "SELECT snapshot_sequence, row_json FROM cognition_snapshots "
        "WHERE snapshot_id = 'later'"
    ).fetchone()
    assert stored["snapshot_sequence"] == 2
    assert json.loads(stored["row_json"])["snapshot_sequence"] == 2
    check.close()


def test_sqlite_legacy_migration_rolls_back_on_invalid_row_json(tmp_path: Path) -> None:
    db = tmp_path / "invalid-legacy.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE cognition_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL DEFAULT '',
            ai_id TEXT NOT NULL DEFAULT '',
            session_id TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT '',
            captured_at_ms INTEGER NOT NULL DEFAULT 0,
            row_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        "INSERT INTO cognition_snapshots VALUES "
        "('broken', 't-1', 'ai-1', 's-1', 'interaction', 1, '{')"
    )
    conn.commit()
    conn.close()

    with pytest.raises(json.JSONDecodeError):
        CognitionSnapshotStore(db_path=str(db))

    check = sqlite3.connect(db)
    columns = {
        row[1] for row in check.execute("PRAGMA table_info(cognition_snapshots)")
    }
    assert "snapshot_sequence" not in columns
    assert check.execute("SELECT COUNT(*) FROM cognition_snapshots").fetchone()[0] == 1
    check.close()


def test_sqlite_insert_or_replace_is_idempotent(tmp_path: Path) -> None:
    db = str(tmp_path / "cog.db")
    store = CognitionSnapshotStore(db_path=db)
    store.append(_row("c1", 1000))
    store.append(_row("c1", 1500))  # same id -> replace, not duplicate
    rows = list(store)
    assert len(rows) == 1
    assert rows[0]["captured_at_ms"] == 1500


def test_retention_sweep_deletes_old_rows_memory() -> None:
    # No retention at construction -> no auto-sweep on append; we drive
    # the sweep explicitly with an injected horizon + clock.
    store = CognitionSnapshotStore()
    now = 30 * 86_400_000
    store.append(_row("old", now - 10 * 86_400_000))
    store.append(_row("fresh", now - 1 * 86_400_000))
    deleted = store.sweep(retention_days=7, now_ms=now)
    assert deleted == 1
    assert {r["snapshot_id"] for r in store} == {"fresh"}


def test_retention_sweep_deletes_old_rows_sqlite(tmp_path: Path) -> None:
    db = str(tmp_path / "cog.db")
    store = CognitionSnapshotStore(db_path=db)
    now = 30 * 86_400_000
    store.append(_row("old", now - 10 * 86_400_000))
    store.append(_row("fresh", now - 1 * 86_400_000))
    deleted = store.sweep(retention_days=7, now_ms=now)
    assert deleted == 1
    assert {r["snapshot_id"] for r in store} == {"fresh"}


def test_auto_sweep_on_append_evicts_old_rows() -> None:
    # With retention configured, appending fresh-then-stale rows lets the
    # throttled auto-sweep evict anything past the horizon relative to
    # the real clock. Rows far in the past are removed on the next append.
    store = CognitionSnapshotStore(retention_days=7)
    store.append(_row("ancient", 1000))  # ~1970 -> well past any horizon
    assert "ancient" not in {r["snapshot_id"] for r in store}


def test_sweep_noop_without_retention() -> None:
    store = CognitionSnapshotStore()
    store.append(_row("c1", 1000))
    assert store.sweep(now_ms=10**12) == 0
    assert len(list(store)) == 1
